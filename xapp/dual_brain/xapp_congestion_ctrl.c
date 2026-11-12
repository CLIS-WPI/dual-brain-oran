/*
 * Harmony Dual-Brain congestion-control xApp.
 *
 * E2SM-KPM (report style 4, per UE) → aggregate cell PRB utilization (sum of
 * per-UE RRU.PrbTot{Dl,Ul} %) → sliding-window features (mean, std, min, slope)
 * → NeuralSmith ONNX model → on predicted congestion, FlexRIC SLICE control
 * reserving `reserve_pct` of the PRBs for the edge UE (Harmony gNB: SLICE SM →
 * AURA quota, DL only). Releases the reservation after sustained normal load.
 *
 * Modes (config "mode"):
 *   collect  emit samples + features only (training data for the rApp)
 *   control  collect + predict + act
 *
 * Config: JSON at $XAPP_CC_CONFIG (default /tmp/dualbrain/config.json); see
 * docs/features/dual-brain-demo.md. Events: JSON lines at "events_path".
 * UEs are discovered from the KPM reports (RAN UE ID); nothing is hard-coded.
 * Run with XAPP_DURATION=-1 to keep it running.
 */
#include "../../../../src/xApp/e42_xapp_api.h"
#include "../../../../src/sm/slice_sm/slice_sm_id.h"
#include "cc_assoc.h"
#include "cc_policy.h"
#include "../../../../src/util/alg_ds/alg/defer.h"
#include "../../../../src/util/time_now_us.h"
#include "../../../../src/util/e.h"

#include "cc_model.h"

#include <cjson/cJSON.h>

#include <assert.h>
#include <stdarg.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_UES 32
#define MAX_WINDOW 256
#define KPM_RAN_FUNC_ID 2

/* ------------------------------------------------------------------ config */

typedef struct {
  char mode[16];
  char policy[16]; /* ml (NeuralSmith model) | threshold (persistence baseline: cell PRB >= threshold_cell_prb
                      sustained for min_duration_samples, cc_policy.h) */
  double threshold_cell_prb;
  int min_duration_samples;
  char model_path[512];
  char preprocess_path[512];
  char events_path[512];
  char raw_dump_path[512];
  char aux_metric[64]; /* optional 2nd KPM measurement in the same subscription ("" = none); logged only */ /* debug: every KPM indication, unprocessed ("" = off) */
  char metric[64];
  char positive_label[16];
  char edge_policy[32];
  int kpm_period_ms;
  int window;
  double prob_threshold;
  int trigger_consecutive;
  int release_after_normal;
  int min_hold_samples;
  int lease_samples; /* > 0: lease release (lease_s from config.json), else the legacy calm rule */
  double lease_s;
  double release_below_cell_prb;
  int reserve_pct;
  int edge_window;
  int congestor_window;
  int be_slice_id;
  long long protected_ue_ref; /* >= 0: cap every UE except this KPM RAN UE ID; -1: cap the congestor only */
  int edge_slice_id;
  int nssai_sst;
  long nssai_sd; /* -1 = SST only */
} cc_cfg_t;

static cc_cfg_t cfg;

static void cfg_str(cJSON* j, const char* k, char* dst, size_t n, const char* def)
{
  cJSON* v = cJSON_GetObjectItem(j, k);
  snprintf(dst, n, "%s", cJSON_IsString(v) ? v->valuestring : def);
}

static double cfg_num(cJSON* j, const char* k, double def)
{
  cJSON* v = cJSON_GetObjectItem(j, k);
  return cJSON_IsNumber(v) ? v->valuedouble : def;
}

static int load_cfg(const char* path)
{
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "[CC] cannot open config %s\n", path);
    return -1;
  }
  char buf[16384];
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  buf[n] = 0;
  cJSON* j = cJSON_Parse(buf);
  if (!j) {
    fprintf(stderr, "[CC] invalid JSON in %s\n", path);
    return -1;
  }
  cfg_str(j, "mode", cfg.mode, sizeof(cfg.mode), "collect");
  cfg_str(j, "policy", cfg.policy, sizeof(cfg.policy), "ml");
  cfg.threshold_cell_prb = cfg_num(j, "threshold_cell_prb", 80.0);
  cfg.min_duration_samples = (int)cfg_num(j, "min_duration_samples", 1);
  cfg_str(j, "model_path", cfg.model_path, sizeof(cfg.model_path), "/tmp/dualbrain/model.onnx");
  cfg_str(j, "preprocess_path", cfg.preprocess_path, sizeof(cfg.preprocess_path), "/tmp/dualbrain/preprocess.json");
  cfg_str(j, "events_path", cfg.events_path, sizeof(cfg.events_path), "/tmp/dualbrain/events.jsonl");
  cfg_str(j, "raw_dump_path", cfg.raw_dump_path, sizeof(cfg.raw_dump_path), "");
  cfg_str(j, "aux_metric", cfg.aux_metric, sizeof(cfg.aux_metric), "");
  char dir[8];
  cfg_str(j, "direction", dir, sizeof(dir), "dl");
  cfg_str(j, "metric", cfg.metric, sizeof(cfg.metric), strcmp(dir, "ul") == 0 ? "RRU.PrbTotUl" : "RRU.PrbTotDl");
  cfg_str(j, "positive_label", cfg.positive_label, sizeof(cfg.positive_label), "1");
  cfg_str(j, "edge_policy", cfg.edge_policy, sizeof(cfg.edge_policy), "lowest_prb_share");
  cfg.kpm_period_ms = (int)cfg_num(j, "kpm_period_ms", 100);
  cfg.window = (int)cfg_num(j, "window_samples", 10);
  cfg.prob_threshold = cfg_num(j, "prob_threshold", 0.5);
  cfg.trigger_consecutive = (int)cfg_num(j, "trigger_consecutive", 2);
  cfg.release_after_normal = (int)cfg_num(j, "release_after_normal_samples", 50);
  cfg.min_hold_samples = (int)cfg_num(j, "min_hold_samples", 50);
  cfg.lease_s = cfg_num(j, "lease_s", 0);
  cfg.lease_samples = cfg.lease_s > 0 ? (int)ceil(cfg.lease_s * 1000.0 / (cfg.kpm_period_ms > 0 ? cfg.kpm_period_ms : 100)) : 0;
  /* While reserved, the cap itself holds cell PRB near (100 - reserve_pct)%, which
   * looks "normal" to the model. Release only when load is really gone. */
  cfg.release_below_cell_prb = cfg_num(j, "release_below_cell_prb", 60.0);
  cfg.reserve_pct = (int)cfg_num(j, "reserve_pct", 20);
  cfg.edge_window = (int)cfg_num(j, "edge_window_samples", 100);
  /* The UE that dominated the last congestor_window samples is the one capped. */
  cfg.congestor_window = (int)cfg_num(j, "congestor_window_samples", 5);
  cfg.be_slice_id = (int)cfg_num(j, "best_effort_slice_id", 1);
  /* R1: the protected (cell-edge) UE by E2SM-KPM RAN UE ID. With two heavy UEs, capping
   * only the heaviest lets the other absorb the freed PRBs; capping every UE except the
   * protected one into the same slice caps them *together* (one shared quota). */
  cfg.protected_ue_ref = (long long)cfg_num(j, "protected_ue_ref", -1);
  cfg.edge_slice_id = (int)cfg_num(j, "edge_slice_id", 2);
  cfg.nssai_sst = (int)cfg_num(j, "nssai_sst", 1);
  char sd[16];
  cfg_str(j, "nssai_sd", sd, sizeof(sd), "000001");
  cfg.nssai_sd = sd[0] ? strtol(sd, NULL, 16) : -1;
  cJSON_Delete(j);
  if (cfg.window < 2 || cfg.window > MAX_WINDOW || cfg.reserve_pct < 1 || cfg.reserve_pct > 99) {
    fprintf(stderr, "[CC] config: window_samples in [2,%d], reserve_pct in [1,99]\n", MAX_WINDOW);
    return -1;
  }
  return 0;
}

/* ------------------------------------------------------------------ state */

typedef struct {
  uint64_t ref;   /* RAN UE ID from KPM (gNB-CU UE ID) */
  double last;    /* last per-UE PRB share, % */
  double ema;     /* long average share (reporting) */
  double recent;  /* short average share: who is congesting the cell right now */
  int64_t seen_us;
  int used;
} ue_t;

static struct {
  pthread_mutex_t mtx;
  ue_t ue[MAX_UES];
  double win[MAX_WINDOW];
  int win_len, win_pos;
  uint64_t samples;
  int hot_streak, normal_streak, above_streak;
  int reserved;
  int extended; /* past the lease minimum hold, kept while hot */
  int armed;        /* trigger armed (dualbrain11: disarmed by a lease release, cc_rearm_step) */
  int below_streak; /* samples with cell PRB < threshold_cell_prb while disarmed */
  uint64_t reserved_at;
  uint64_t edge_ref;
  uint32_t assoc_sig; /* UE set signature at last ASSOC */
  int pending;        /* 1 reserve, 2 release, 3 re-assoc */
} st = {.mtx = PTHREAD_MUTEX_INITIALIZER, .armed = 1};

static cc_model_t* model;
static int pos_idx = -1;
static FILE* events;
static global_e2_node_id_t node_id;
static _Atomic int have_node;

static void emit(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void emit(const char* fmt, ...)
{
  if (!events)
    return;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(events, fmt, ap);
  va_end(ap);
  fputc('\n', events);
  fflush(events);
}

static ue_t* ue_get(uint64_t ref)
{
  ue_t* free_slot = NULL;
  for (int i = 0; i < MAX_UES; i++) {
    if (st.ue[i].used && st.ue[i].ref == ref)
      return &st.ue[i];
    if (!st.ue[i].used && !free_slot)
      free_slot = &st.ue[i];
  }
  if (free_slot) {
    memset(free_slot, 0, sizeof(*free_slot));
    free_slot->used = 1;
    free_slot->ref = ref;
  }
  return free_slot;
}

/* UEs reported within the last 2 s. */
static int ue_active(ue_t const* u, int64_t now) { return u->used && now - u->seen_us < 2000000; }

static uint32_t ue_signature(int64_t now)
{
  uint32_t h = 2166136261u;
  for (int i = 0; i < MAX_UES; i++)
    if (ue_active(&st.ue[i], now))
      h = (h ^ (uint32_t)st.ue[i].ref) * 16777619u;
  return h;
}

/* The congestor = the UE with the highest recent PRB share. It is the only UE
 * capped; every other UE (the "edge users") keeps the reserved floor. A long
 * average is wrong here: at a burst's start the bursting UE has been idle, so
 * its long average is the lowest of all. */
static int pick_congestor(int64_t now, uint64_t* congestor)
{
  int n = 0;
  double best = -1;
  for (int i = 0; i < MAX_UES; i++) {
    if (!ue_active(&st.ue[i], now))
      continue;
    n++;
    if (st.ue[i].recent > best) {
      best = st.ue[i].recent;
      *congestor = st.ue[i].ref;
    }
  }
  return n >= 2 ? 0 : -1;
}

/* ------------------------------------------------------------------ KPM */

static int ba_eq(byte_array_t ba, const char* s)
{
  size_t n = strlen(s);
  return ba.len == n && memcmp(ba.buf, s, n) == 0;
}

static int ue_ref_of(ue_id_e2sm_t const* id, uint64_t* ref)
{
  if (id->type == GNB_UE_ID_E2SM && id->gnb.ran_ue_id) {
    *ref = *id->gnb.ran_ue_id;
    return 0;
  }
  if (id->type == GNB_DU_UE_ID_E2SM && id->gnb_du.ran_ue_id) {
    *ref = *id->gnb_du.ran_ue_id;
    return 0;
  }
  if (id->type == GNB_UE_ID_E2SM) { /* fall back to AMF UE NGAP ID */
    *ref = id->gnb.amf_ue_ngap_id;
    return 0;
  }
  return -1;
}

static int metric_value(kpm_ind_msg_format_1_t const* m, double* v)
{
  if (m->meas_data_lst_len == 0)
    return -1;
  meas_data_lst_t const* d = &m->meas_data_lst[0];
  for (size_t i = 0; i < m->meas_info_lst_len && i < d->meas_record_len; i++) {
    meas_type_t const* t = &m->meas_info_lst[i].meas_type;
    if (t->type != NAME_MEAS_TYPE || !ba_eq(t->name, cfg.metric))
      continue;
    meas_record_lst_t const* r = &d->meas_record_lst[i];
    if (r->value == INTEGER_MEAS_VALUE)
      *v = (double)r->int_val;
    else if (r->value == REAL_MEAS_VALUE)
      *v = r->real_val;
    else
      return -1;
    return 0;
  }
  return -1;
}

static void on_sample(double cell, int64_t now)
{
  st.win[st.win_pos] = cell;
  st.win_pos = (st.win_pos + 1) % cfg.window;
  if (st.win_len < cfg.window)
    st.win_len++;
  st.samples++;

  char ues[1024];
  size_t o = 0;
  ues[0] = 0;
  for (int i = 0; i < MAX_UES; i++)
    if (ue_active(&st.ue[i], now) && o < sizeof(ues) - 48)
      o += (size_t)snprintf(ues + o, sizeof(ues) - o, "%s\"%llu\":%.1f", o ? "," : "",
                            (unsigned long long)st.ue[i].ref, st.ue[i].last);

  if (st.win_len < cfg.window) {
    emit("{\"t_us\":%lld,\"type\":\"sample\",\"cell_prb\":%.2f,\"ues\":{%s},\"warmup\":true}", (long long)now, cell, ues);
    return;
  }
  double ordered[MAX_WINDOW], feat[CC_N_WINDOW_FEATURES];
  for (int i = 0; i < cfg.window; i++)
    ordered[i] = st.win[(st.win_pos + i) % cfg.window];
  cc_window_features(ordered, (size_t)cfg.window, (double)cfg.kpm_period_ms, feat);

  int threshold_policy = strcmp(cfg.mode, "control") == 0 && strcmp(cfg.policy, "threshold") == 0;
  if (!model && !threshold_policy) {
    emit("{\"t_us\":%lld,\"type\":\"sample\",\"cell_prb\":%.2f,\"ues\":{%s},"
         "\"features\":{\"prb_mean\":%.4f,\"prb_std\":%.4f,\"prb_min\":%.4f,\"prb_slope\":%.5f,"
         "\"max_prb\":%.4f,\"slope_prb\":%.5f,\"last_prb\":%.4f}}",
         (long long)now, cell, ues, feat[0], feat[1], feat[2], feat[3], feat[7], feat[8], feat[9]);
    return;
  }

  /* Model input in the order the model expects (preprocess.json features). */
  double raw[CC_MAX_FEATURES];
  for (size_t k = 0; model && k < cc_model_n_features(model); k++) {
    raw[k] = NAN;
    for (int w = 0; w < CC_N_WINDOW_FEATURES; w++)
      if (strcmp(cc_model_feature_name(model, k), cc_window_feature_names[w]) == 0)
        raw[k] = feat[w];
  }
  double proba[8] = {0}, lat_us = 0, p;
  int idx = -1, hot;
  char err[256];
  if (threshold_policy) { /* baseline: reactive threshold on the current cell utilization */
    hot = cc_threshold_hot(&st.above_streak, cell, cfg.threshold_cell_prb, cfg.min_duration_samples);
    p = hot ? 1.0 : 0.0;
  } else {
    if (cc_model_predict(model, raw, &idx, proba, &lat_us, err, sizeof(err)) != 0) {
      fprintf(stderr, "[CC] predict: %s\n", err);
      return;
    }
    p = proba[pos_idx];
    hot = p >= cfg.prob_threshold;
  }
  st.hot_streak = hot ? st.hot_streak + 1 : 0;
  int calm = !hot && (!st.reserved || cell < cfg.release_below_cell_prb);
  st.normal_streak = calm ? st.normal_streak + 1 : 0;

  emit("{\"t_us\":%lld,\"type\":\"sample\",\"cell_prb\":%.2f,\"ues\":{%s},"
       "\"features\":{\"prb_mean\":%.4f,\"prb_std\":%.4f,\"prb_min\":%.4f,\"prb_slope\":%.5f,"
       "\"max_prb\":%.4f,\"slope_prb\":%.5f,\"last_prb\":%.4f},"
       "\"prob\":%.4f,\"pred\":%d,\"latency_us\":%.1f,\"reserved\":%s,\"armed\":%s,\"policy\":\"%s\"}",
       (long long)now, cell, ues, feat[0], feat[1], feat[2], feat[3], feat[7], feat[8], feat[9], p, hot, lat_us,
       st.reserved ? "true" : "false", st.armed ? "true" : "false", threshold_policy ? "threshold" : "ml");

  if (cfg.lease_samples > 0 && !st.reserved
      && cc_rearm_step(&st.armed, &st.below_streak, cell, cfg.threshold_cell_prb, cfg.min_duration_samples))
    emit("{\"t_us\":%lld,\"type\":\"control\",\"action\":\"rearm\",\"cell_prb\":%.2f}", (long long)now, cell);
  if (strcmp(cfg.mode, "control") != 0 || st.pending)
    return;
  if (!st.reserved && st.armed && cc_should_trigger(threshold_policy, st.hot_streak, cfg.trigger_consecutive)) {
    uint64_t congestor = 0;
    if (pick_congestor(now, &congestor) == 0) {
      st.edge_ref = congestor; /* field reused: the capped UE */
      st.pending = 1;
    } else if (st.hot_streak == (threshold_policy ? 1 : cfg.trigger_consecutive)) {
      emit("{\"t_us\":%lld,\"type\":\"control\",\"action\":\"skip\",\"reason\":\"need >= 2 active UEs\"}", (long long)now);
    }
  } else if (st.reserved && cfg.lease_samples > 0) {
    unsigned long long start = st.reserved_at;
    int a = cc_lease_step(st.samples, &start, cfg.lease_samples, hot);
    st.reserved_at = start;
    if (a == CC_LEASE_RELEASE)
      st.pending = 2;
    else if (a == CC_LEASE_RENEW && !st.extended) {
      st.extended = 1; /* one event when the minimum hold is over and the cap stays because the policy is hot */
      emit("{\"t_us\":%lld,\"type\":\"control\",\"action\":\"extend\",\"lease_s\":%.1f}", (long long)now, cfg.lease_s);
    }
    else if (ue_signature(now) != st.assoc_sig)
      st.pending = 3; /* a UE joined/left while reserved: re-associate */
  } else if (st.reserved && st.normal_streak >= cfg.release_after_normal
             && st.samples - st.reserved_at >= (uint64_t)cfg.min_hold_samples) {
    st.pending = 2;
  } else if (st.reserved && ue_signature(now) != st.assoc_sig) {
    st.pending = 3; /* a UE joined/left while reserved: re-associate */
  }
}

/* Debug: one JSON line per KPM indication, values exactly as decoded (no scaling,
 * no rounding): header, every UE report's identifiers, every measurement record of
 * every meas_data entry (record type + raw int/real), and which known UEs are absent. */
static FILE* raw_dump;
static void dump_raw(kpm_ind_data_t const* ind, int64_t now)
{
  if (!raw_dump)
    return;
  kpm_ind_msg_format_3_t const* f3 = &ind->msg.frm_3;
  fprintf(raw_dump, "{\"t_us\":%lld,\"hdr_type\":%d,\"collectStartTime\":%llu,\"edge_ref\":%llu,\"reserved\":%d,"
          "\"n_ue_reports\":%zu,\"ues\":[",
          (long long)now, (int)ind->hdr.type,
          ind->hdr.type == FORMAT_1_INDICATION_HEADER ? (unsigned long long)ind->hdr.kpm_ric_ind_hdr_format_1.collectStartTime : 0ULL,
          (unsigned long long)st.edge_ref, st.reserved, f3->ue_meas_report_lst_len);
  uint64_t seen[MAX_UES];
  size_t nseen = 0;
  for (size_t i = 0; i < f3->ue_meas_report_lst_len; i++) {
    ue_id_e2sm_t const* id = &f3->meas_report_per_ue[i].ue_meas_report_lst;
    kpm_ind_msg_format_1_t const* m = &f3->meas_report_per_ue[i].ind_msg_format_1;
    uint64_t ref = 0;
    int ok = ue_ref_of(id, &ref) == 0;
    if (ok && nseen < MAX_UES)
      seen[nseen++] = ref;
    fprintf(raw_dump, "%s{\"id_type\":%d", i ? "," : "", (int)id->type);
    if (id->type == GNB_UE_ID_E2SM)
      fprintf(raw_dump, ",\"amf_ue_ngap_id\":%llu,\"ran_ue_id\":%s%llu%s", (unsigned long long)id->gnb.amf_ue_ngap_id,
              id->gnb.ran_ue_id ? "" : "null,\"_x\":", id->gnb.ran_ue_id ? (unsigned long long)*id->gnb.ran_ue_id : 0ULL, "");
    else if (id->type == GNB_DU_UE_ID_E2SM)
      fprintf(raw_dump, ",\"gnb_cu_ue_f1ap\":%u,\"ran_ue_id\":%s%llu", id->gnb_du.gnb_cu_ue_f1ap,
              id->gnb_du.ran_ue_id ? "" : "null,\"_x\":", id->gnb_du.ran_ue_id ? (unsigned long long)*id->gnb_du.ran_ue_id : 0ULL);
    fprintf(raw_dump, ",\"xapp_ref\":%s%llu,\"meas_info_len\":%zu,\"meas_data_len\":%zu,\"data\":[", ok ? "" : "null,\"_y\":",
            (unsigned long long)ref, m->meas_info_lst_len, m->meas_data_lst_len);
    for (size_t d = 0; d < m->meas_data_lst_len; d++) {
      meas_data_lst_t const* md = &m->meas_data_lst[d];
      fprintf(raw_dump, "%s{\"incomplete\":%s,\"records\":[", d ? "," : "",
              md->incomplete_flag ? "true" : "false");
      for (size_t k = 0; k < md->meas_record_len; k++) {
        meas_record_lst_t const* rec = &md->meas_record_lst[k];
        char name[64] = "?";
        if (k < m->meas_info_lst_len && m->meas_info_lst[k].meas_type.type == NAME_MEAS_TYPE) {
          size_t n = m->meas_info_lst[k].meas_type.name.len;
          n = n < sizeof(name) - 1 ? n : sizeof(name) - 1;
          memcpy(name, m->meas_info_lst[k].meas_type.name.buf, n);
          name[n] = 0;
        }
        if (rec->value == INTEGER_MEAS_VALUE)
          fprintf(raw_dump, "%s{\"name\":\"%s\",\"type\":\"INTEGER\",\"int_val\":%u}", k ? "," : "", name, rec->int_val);
        else if (rec->value == REAL_MEAS_VALUE)
          fprintf(raw_dump, "%s{\"name\":\"%s\",\"type\":\"REAL\",\"real_val\":%.17g}", k ? "," : "", name, rec->real_val);
        else
          fprintf(raw_dump, "%s{\"name\":\"%s\",\"type\":\"NO_VALUE(%d)\"}", k ? "," : "", name, (int)rec->value);
      }
      fprintf(raw_dump, "]}");
    }
    fprintf(raw_dump, "]}");
  }
  fprintf(raw_dump, "],\"known_absent\":[");
  int first = 1;
  for (int i = 0; i < MAX_UES; i++) {
    if (!st.ue[i].used)
      continue;
    int found = 0;
    for (size_t k = 0; k < nseen; k++)
      found |= seen[k] == st.ue[i].ref;
    if (!found) {
      fprintf(raw_dump, "%s%llu", first ? "" : ",", (unsigned long long)st.ue[i].ref);
      first = 0;
    }
  }
  fprintf(raw_dump, "]}\n");
  fflush(raw_dump);
}

static void sm_cb_kpm(sm_ag_if_rd_t const* rd)
{
  assert(rd->type == INDICATION_MSG_AGENT_IF_ANS_V0);
  kpm_ind_data_t const* ind = &rd->ind.kpm.ind;
  if (ind->msg.type != FORMAT_3_INDICATION_MESSAGE)
    return;
  kpm_ind_msg_format_3_t const* f3 = &ind->msg.frm_3;
  int64_t now = time_now_us();
  pthread_mutex_lock(&st.mtx);
  dump_raw(ind, now);
  double cell = 0;
  int n = 0;
  for (size_t i = 0; i < f3->ue_meas_report_lst_len; i++) {
    uint64_t ref;
    double v;
    if (ue_ref_of(&f3->meas_report_per_ue[i].ue_meas_report_lst, &ref) != 0
        || metric_value(&f3->meas_report_per_ue[i].ind_msg_format_1, &v) != 0)
      continue;
    ue_t* u = ue_get(ref);
    if (!u)
      continue;
    double a = 1.0 / (double)cfg.edge_window;
    double r = 2.0 / ((double)cfg.congestor_window + 1.0);
    u->ema = u->seen_us ? (1 - a) * u->ema + a * v : v;
    u->recent = u->seen_us ? (1 - r) * u->recent + r * v : v;
    u->last = v;
    u->seen_us = now;
    cell += v;
    n++;
  }
  if (n > 0)
    on_sample(cell > 100 ? 100 : cell, now);
  pthread_mutex_unlock(&st.mtx);
}

static test_info_lst_t nssai_filter(void)
{
  test_info_lst_t dst = {0};
  dst.test_cond_type = S_NSSAI_TEST_COND_TYPE;
  dst.S_NSSAI = TRUE_TEST_COND_TYPE;
  dst.test_cond = ecalloc(1, sizeof(test_cond_e));
  *dst.test_cond = EQUAL_TEST_COND;
  dst.test_cond_value = ecalloc(1, sizeof(test_cond_value_t));
  dst.test_cond_value->type = OCTET_STRING_TEST_COND_VALUE;
  dst.test_cond_value->octet_string_value = ecalloc(1, sizeof(byte_array_t));
  size_t len = cfg.nssai_sd >= 0 ? 4 : 1;
  dst.test_cond_value->octet_string_value->len = len;
  uint8_t* b = ecalloc(len, 1);
  b[0] = (uint8_t)cfg.nssai_sst;
  if (cfg.nssai_sd >= 0) {
    b[1] = (cfg.nssai_sd >> 16) & 0xff;
    b[2] = (cfg.nssai_sd >> 8) & 0xff;
    b[3] = cfg.nssai_sd & 0xff;
  }
  dst.test_cond_value->octet_string_value->buf = b;
  return dst;
}

static int build_kpm_sub(kpm_ran_function_def_t const* rf, kpm_sub_data_t* sub)
{
  ric_report_style_item_t const* style4 = NULL;
  for (size_t i = 0; i < rf->sz_ric_report_style_list; i++)
    if (rf->ric_report_style_list[i].report_style_type == STYLE_4_RIC_SERVICE_REPORT)
      style4 = &rf->ric_report_style_list[i];
  if (!style4) {
    fprintf(stderr, "[CC] gNB offers no KPM report style 4 (per-UE)\n");
    return -1;
  }
  int has_metric = 0;
  for (size_t i = 0; i < style4->meas_info_for_action_lst_len; i++)
    if (ba_eq(style4->meas_info_for_action_lst[i].name, cfg.metric))
      has_metric = 1;
  if (!has_metric) {
    fprintf(stderr, "[CC] gNB does not offer %s in KPM style 4\n", cfg.metric);
    return -1;
  }
  memset(sub, 0, sizeof(*sub));
  sub->ev_trg_def.type = FORMAT_1_RIC_EVENT_TRIGGER;
  sub->ev_trg_def.kpm_ric_event_trigger_format_1.report_period_ms = (uint32_t)cfg.kpm_period_ms;
  sub->sz_ad = 1;
  sub->ad = ecalloc(1, sizeof(kpm_act_def_t));
  kpm_act_def_t* ad = sub->ad;
  ad->type = FORMAT_4_ACTION_DEFINITION;
  ad->frm_4.matching_cond_lst_len = 1;
  ad->frm_4.matching_cond_lst = ecalloc(1, sizeof(matching_condition_format_4_lst_t));
  ad->frm_4.matching_cond_lst[0].test_info_lst = nssai_filter();
  kpm_act_def_format_1_t* f1 = &ad->frm_4.action_def_format_1;
  int aux = 0;
  for (size_t i = 0; cfg.aux_metric[0] && i < style4->meas_info_for_action_lst_len; i++)
    aux |= ba_eq(style4->meas_info_for_action_lst[i].name, cfg.aux_metric);
  if (cfg.aux_metric[0] && !aux)
    fprintf(stderr, "[CC] gNB does not offer aux metric %s; subscribing to %s only\n", cfg.aux_metric, cfg.metric);
  f1->meas_info_lst_len = 1 + (size_t)aux;
  f1->meas_info_lst = ecalloc(f1->meas_info_lst_len, sizeof(meas_info_format_1_lst_t));
  for (size_t i = 0; i < f1->meas_info_lst_len; i++) {
    f1->meas_info_lst[i].meas_type.type = NAME_MEAS_TYPE;
    f1->meas_info_lst[i].meas_type.name = cp_str_to_ba(i == 0 ? cfg.metric : cfg.aux_metric);
    f1->meas_info_lst[i].label_info_lst_len = 1;
    f1->meas_info_lst[i].label_info_lst = ecalloc(1, sizeof(label_info_lst_t));
    f1->meas_info_lst[i].label_info_lst[0].noLabel = ecalloc(1, sizeof(enum_value_e));
    *f1->meas_info_lst[i].label_info_lst[0].noLabel = TRUE_ENUM_VALUE;
  }
  f1->gran_period_ms = (uint32_t)cfg.kpm_period_ms;
  f1->cell_global_id = NULL;
#if defined KPM_V2_03 || defined KPM_V3_00
  f1->meas_bin_range_info_lst_len = 0;
  f1->meas_bin_info_lst = NULL;
#endif
  return 0;
}

/* ------------------------------------------------------------------ SLICE control */

static char* dup_n(const char* s, uint32_t* len)
{
  *len = (uint32_t)strlen(s);
  char* d = malloc(*len);
  memcpy(d, s, *len);
  return d;
}

static void fill_slice(fr_slice_t* s, uint32_t id, const char* label, int pct)
{
  s->id = id;
  s->label = dup_n(label, &s->len_label);
  s->sched = dup_n("PF", &s->len_sched);
  s->params.type = SLICE_ALG_SM_V0_STATIC;
  /* Harmony gNB convention: pos_low..pos_high = share of the DL BWP in percent. */
  s->params.u.sta.pos_low = 0;
  s->params.u.sta.pos_high = (uint32_t)pct - 1;
}

static void send_ctrl(slice_ctrl_req_data_t* req)
{
  control_sm_xapp_api(&node_id, SM_SLICE_ID, req);
  free_slice_ctrl_msg(&req->msg);
}

static void ctrl_reserve(void)
{
  slice_ctrl_req_data_t add = {0};
  add.msg.type = SLICE_CTRL_SM_V0_ADD;
  ul_dl_slice_conf_t* dl = &add.msg.u.add_mod_slice.dl;
  dl->sched_name = dup_n("PF", &dl->len_sched_name);
  /* A reservation is a floor for the edge UE, not a cap: only the other UEs are
   * capped (best-effort = 100 - reserve_pct %); the edge UE stays unassociated
   * (uncapped) and always finds >= reserve_pct % of the PRBs free. */
  dl->len_slices = 1;
  dl->slices = ecalloc(1, sizeof(fr_slice_t));
  fill_slice(&dl->slices[0], (uint32_t)cfg.be_slice_id, "best-effort", 100 - cfg.reserve_pct);
  ul_dl_slice_conf_t* ul = &add.msg.u.add_mod_slice.ul;
  ul->sched_name = dup_n("PF", &ul->len_sched_name);
  send_ctrl(&add);
}

static void ctrl_assoc(int64_t now)
{
  slice_ctrl_req_data_t req = {0};
  req.msg.type = SLICE_CTRL_SM_V0_UE_SLICE_ASSOC;
  ue_slice_conf_t* a = &req.msg.u.ue_slice;
  a->ues = ecalloc(MAX_UES, sizeof(ue_slice_assoc_t));
  char list[512];
  size_t o = 0;
  list[0] = 0;
  for (int i = 0; i < MAX_UES; i++) {
    if (!ue_active(&st.ue[i], now) || st.ue[i].ref > 0xffff)
      continue;
    if (!cc_capped(st.ue[i].ref, st.edge_ref, cfg.protected_ue_ref))
      continue; /* protected UEs stay unassociated (uncapped, scheduled first) */
    ue_slice_assoc_t* u = &a->ues[a->len_ue_slice++];
    u->rnti = (uint16_t)st.ue[i].ref; /* Harmony gNB resolves RAN UE ID → RNTI */
    u->dl_id = (uint32_t)cfg.be_slice_id;
    u->ul_id = u->dl_id;
    if (o < sizeof(list) - 32)
      o += (size_t)snprintf(list + o, sizeof(list) - o, "%s\"%llu\":%u", o ? "," : "",
                            (unsigned long long)st.ue[i].ref, u->dl_id);
  }
  st.assoc_sig = ue_signature(now);
  send_ctrl(&req);
  emit("{\"t_us\":%lld,\"type\":\"control\",\"action\":\"assoc\",\"ues\":{%s}}", (long long)now, list);
}

static void ctrl_release(void)
{
  slice_ctrl_req_data_t del = {0};
  del.msg.type = SLICE_CTRL_SM_V0_DEL;
  del.msg.u.del_slice.len_dl = 2;
  del.msg.u.del_slice.dl = ecalloc(2, sizeof(uint32_t));
  del.msg.u.del_slice.dl[0] = (uint32_t)cfg.be_slice_id;
  del.msg.u.del_slice.dl[1] = (uint32_t)cfg.edge_slice_id;
  send_ctrl(&del);
}

/* Control runs here, never inside the KPM callback. */
static void control_step(void)
{
  pthread_mutex_lock(&st.mtx);
  int action = st.pending;
  int64_t now = time_now_us();
  if (!action || !have_node) {
    pthread_mutex_unlock(&st.mtx);
    return;
  }
  if (action == 1) {
    ctrl_reserve();
    ctrl_assoc(now);
    st.reserved = 1;
    st.extended = 0;
    st.reserved_at = st.samples;
    st.normal_streak = 0;
    char prot[256];
    size_t o = 0;
    uint64_t first_protected = 0;
    prot[0] = 0;
    for (int i = 0; i < MAX_UES; i++)
      if (ue_active(&st.ue[i], now) && o < sizeof(prot) - 24 && !cc_capped(st.ue[i].ref, st.edge_ref, cfg.protected_ue_ref)) {
        if (!o)
          first_protected = st.ue[i].ref;
        o += (size_t)snprintf(prot + o, sizeof(prot) - o, "%s%llu", o ? "," : "", (unsigned long long)st.ue[i].ref);
      }
    emit("{\"t_us\":%lld,\"type\":\"control\",\"action\":\"reserve\",\"cap\":\"%s\",\"congestor_ue\":%llu,\"protected_ues\":[%s],"
         "\"edge_ue\":%llu,\"reserve_pct\":%d,\"best_effort_pct\":%d,\"direction\":\"%s\"}",
         (long long)now, cfg.protected_ue_ref >= 0 ? "all_but_protected" : "congestor",
         (unsigned long long)st.edge_ref, prot, (unsigned long long)first_protected, cfg.reserve_pct,
         100 - cfg.reserve_pct, strstr(cfg.metric, "Ul") ? "ul" : "dl");
    if (cfg.protected_ue_ref >= 0)
      printf("[CC] RESERVE: cap all UEs except %lld together at %d%%, protect [%s] (>= %d%%)\n",
             cfg.protected_ue_ref, 100 - cfg.reserve_pct, prot, cfg.reserve_pct);
    else
      printf("[CC] RESERVE: cap UE %llu at %d%%, protect [%s] (>= %d%%)\n", (unsigned long long)st.edge_ref,
             100 - cfg.reserve_pct, prot, cfg.reserve_pct);
  } else if (action == 2) {
    ctrl_release();
    st.reserved = 0;
    if (cfg.lease_samples > 0) { /* R1 lease path: re-arm only after the cell has been below threshold (cc_policy.h) */
      st.armed = 0;
      st.below_streak = 0;
    }
    emit("{\"t_us\":%lld,\"type\":\"control\",\"action\":\"release\",\"congestor_ue\":%llu}", (long long)now,
         (unsigned long long)st.edge_ref);
    printf("[CC] RELEASE reservation\n");
  } else if (action == 3) {
    ctrl_assoc(now);
  }
  st.pending = 0;
  pthread_mutex_unlock(&st.mtx);
}

static _Atomic int stop;
static void on_signal(int s)
{
  (void)s;
  stop = 1;
}

int main(int argc, char* argv[])
{
  const char* cfg_path = getenv("XAPP_CC_CONFIG");
  if (!cfg_path || !*cfg_path)
    cfg_path = "/tmp/dualbrain/config.json";
  if (load_cfg(cfg_path) != 0)
    return 1;
  if (cfg.raw_dump_path[0] && !(raw_dump = fopen(cfg.raw_dump_path, "a")))
    fprintf(stderr, "[CC] cannot open raw dump %s\n", cfg.raw_dump_path);
  events = fopen(cfg.events_path, "a");
  if (!events) {
    fprintf(stderr, "[CC] cannot open events file %s\n", cfg.events_path);
    return 1;
  }
  if (strcmp(cfg.mode, "control") == 0 && strcmp(cfg.policy, "threshold") != 0) {
    char err[512];
    model = cc_model_load(cfg.model_path, cfg.preprocess_path, err, sizeof(err));
    if (!model) {
      fprintf(stderr, "[CC] model: %s\n", err);
      return 1;
    }
    pos_idx = cc_model_class_index(model, cfg.positive_label);
    if (pos_idx < 0) {
      fprintf(stderr, "[CC] positive_label %s not among model classes\n", cfg.positive_label);
      return 1;
    }
  }
  emit("{\"t_us\":%lld,\"type\":\"start\",\"mode\":\"%s\",\"policy\":\"%s\",\"metric\":\"%s\",\"window\":%d,"
       "\"period_ms\":%d,\"reserve_pct\":%d,\"prob_threshold\":%.3f,\"threshold_cell_prb\":%.1f,\"min_duration_samples\":%d,"
       "\"trigger_consecutive\":%d,\"lease_s\":%.1f,\"lease_samples\":%d,\"model\":\"%s\"}",
       (long long)time_now_us(), cfg.mode, cfg.policy, cfg.metric, cfg.window, cfg.kpm_period_ms, cfg.reserve_pct,
       cfg.prob_threshold, cfg.threshold_cell_prb, cfg.min_duration_samples, cfg.trigger_consecutive, cfg.lease_s,
       cfg.lease_samples,
       model ? cfg.model_path : "");

  fr_args_t args = init_fr_args(argc, argv);
  init_xapp_api(&args);
  sleep(1);
  e2_node_arr_xapp_t nodes = e2_nodes_xapp_api();
  defer({ free_e2_node_arr_xapp(&nodes); });
  if (nodes.len == 0) {
    fprintf(stderr, "[CC] no E2 node connected to the RIC\n");
    return 1;
  }

  /* Newest KPM-capable node first: after a gNB restart FlexRIC can still list the
   * dead association before the live one (same node ID). */
  sm_ans_xapp_t h = {0};
  for (size_t ii = nodes.len; ii-- > 0 && !h.success;) {
    e2_node_connected_xapp_t* n = &nodes.n[ii];
    for (size_t j = 0; j < n->len_rf; j++) {
      if (n->rf[j].id != KPM_RAN_FUNC_ID || n->rf[j].defn.type != KPM_RAN_FUNC_DEF_E)
        continue;
      kpm_sub_data_t sub;
      if (build_kpm_sub(&n->rf[j].defn.kpm, &sub) != 0)
        return 1;
      h = report_sm_xapp_api(&n->id, KPM_RAN_FUNC_ID, &sub, sm_cb_kpm);
      free_kpm_sub_data(&sub);
      if (h.success) {
        node_id = cp_global_e2_node_id(&n->id);
        have_node = 1;
      }
      break;
    }
  }
  if (!h.success) {
    fprintf(stderr, "[CC] KPM subscription failed\n");
    return 1;
  }
  if (strcmp(cfg.mode, "control") == 0) {
    /* Clean slate: drop a reservation left behind by a killed predecessor. */
    pthread_mutex_lock(&st.mtx);
    ctrl_release();
    pthread_mutex_unlock(&st.mtx);
  }
  printf("[CC] mode=%s metric=%s period=%dms window=%d — running\n", cfg.mode, cfg.metric, cfg.kpm_period_ms,
         cfg.window);

  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);
  const char* dur = getenv("XAPP_DURATION");
  int64_t deadline = (dur && atoi(dur) > 0) ? time_now_us() + (int64_t)atoi(dur) * 1000000 : 0;
  while (!stop && (!deadline || time_now_us() < deadline)) {
    control_step();
    usleep(10000);
  }
  pthread_mutex_lock(&st.mtx);
  if (st.reserved) { /* never leave a reservation behind */
    ctrl_release();
    emit("{\"t_us\":%lld,\"type\":\"control\",\"action\":\"release\",\"reason\":\"xapp_stop\"}", (long long)time_now_us());
  }
  pthread_mutex_unlock(&st.mtx);
  rm_report_sm_xapp_api(h.u.handle);
  while (try_stop_xapp_api() == false)
    usleep(1000);
  cc_model_free(model);
  emit("{\"t_us\":%lld,\"type\":\"stop\"}", (long long)time_now_us());
  fclose(events);
  return 0;
}
