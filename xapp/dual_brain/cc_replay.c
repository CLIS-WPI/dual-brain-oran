/*
 * Feature/model parity replay (COMMAG-R1): runs recorded KPM windows through the xApp's OWN feature path
 * (cc_window_features) and model path (preprocess.json + ONNX Runtime) — the code the running xApp uses.
 *
 *   cc_replay <bundle_dir> <period_ms> <windows.csv> [positive_label=1]
 *   cc_replay --policy <bundle_dir> <period_ms> <series.txt> <window> <trigger_consecutive> <lease_samples>
 *             <threshold_cell_prb> <min_duration_samples>
 *
 * --policy (dualbrain10): replays a recorded cell-PRB series (one value per line, KPM order) through the xApp's
 * window → features → model → debounce (cc_should_trigger) → lease (cc_lease_step) logic, with actuation
 * simulated (reserve/extend/release at once; no ">= 2 active UEs" condition), with the dualbrain11 re-arm rule
 * (cc_rearm_step: after a release, no reserve until cell < threshold for min_duration_samples). stdout CSV:
 * i,cell,proba,pred,action,reserved,armed (action rearm on the re-arming sample unless it also reserves).
 * The caller checks that the lease expires and the cap lifts in OFF (r1/zto/parity.py).
 *
 * windows.csv: one window per line, cell PRB % values oldest → newest, comma-separated (no header).
 * stdout CSV: header, then per window the R1 builder features (mean_prb,std_prb,min_prb,max_prb,slope_prb,
 * last_prb) as the xApp computes them, the positive-class probability and the predicted label.
 * The comparison against r1/dataset/build.py and the offline ONNX prediction is done by the caller
 * (r1/zto/parity.py), so this tool only reports; exit 1 on a load/predict error.
 */
#include "cc_model.h"
#include "cc_policy.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_WIN 256

static int policy_replay(int argc, char** argv)
{
  if (argc < 10) {
    fprintf(stderr, "usage: %s --policy <bundle_dir> <period_ms> <series.txt> <window> <trigger> <lease_samples> "
                    "<threshold_cell_prb> <min_duration_samples>\n", argv[0]);
    return 2;
  }
  double period_ms = atof(argv[3]);
  int window = atoi(argv[5]), trigger = atoi(argv[6]), lease = atoi(argv[7]), min_dur = atoi(argv[9]);
  double thr = atof(argv[8]);
  char p_onnx[1024], p_pre[1024], err[512];
  snprintf(p_onnx, sizeof(p_onnx), "%s/model.onnx", argv[2]);
  snprintf(p_pre, sizeof(p_pre), "%s/preprocess.json", argv[2]);
  cc_model_t* m = cc_model_load(p_onnx, p_pre, err, sizeof(err));
  FILE* f = fopen(argv[4], "r");
  if (!m || !f || window < 2 || window > MAX_WIN) {
    fprintf(stderr, "load failed: %s\n", m ? "series/window" : err);
    return 1;
  }
  int pos = cc_model_class_index(m, "1"), hot_streak = 0, reserved = 0, extended = 0, armed = 1, below = 0;
  double win[MAX_WIN], cell;
  unsigned long long i = 0, lease_start = 0;
  char line[256];
  printf("i,cell,proba,pred,action,reserved,armed\n");
  while (fgets(line, sizeof(line), f)) {
    cell = strtod(line, NULL);
    for (int k = 0; k + 1 < window; k++)
      win[k] = win[k + 1];
    win[window - 1] = cell;
    const char* action = "";
    double proba[8] = {0}, lat = 0;
    int pred = 0;
    if (i + 1 >= (unsigned long long)window) {
      double feat[CC_N_WINDOW_FEATURES], raw[CC_MAX_FEATURES];
      cc_window_features(win, (size_t)window, period_ms, feat);
      for (size_t k = 0; k < cc_model_n_features(m); k++) {
        raw[k] = NAN;
        for (int w = 0; w < CC_N_WINDOW_FEATURES; w++)
          if (strcmp(cc_model_feature_name(m, k), cc_window_feature_names[w]) == 0)
            raw[k] = feat[w];
      }
      int idx = -1;
      if (cc_model_predict(m, raw, &idx, proba, &lat, err, sizeof(err)) != 0) {
        fprintf(stderr, "predict failed: %s\n", err);
        return 1;
      }
      pred = proba[pos] >= 0.5;
      hot_streak = pred ? hot_streak + 1 : 0;
      if (!reserved && cc_rearm_step(&armed, &below, cell, thr, min_dur))
        action = "rearm";
      if (!reserved && armed && cc_should_trigger(0, hot_streak, trigger)) {
        reserved = 1;
        extended = 0;
        lease_start = i;
        action = "reserve";
      } else if (reserved) {
        int a = cc_lease_step(i, &lease_start, lease, pred);
        if (a == CC_LEASE_RENEW && !extended) {
          extended = 1;
          action = "extend";
        }
        else if (a == CC_LEASE_RELEASE) {
          reserved = 0;
          armed = 0;
          below = 0;
          action = "release";
        }
      }
    }
    printf("%llu,%.2f,%.6f,%d,%s,%d,%d\n", i, cell, proba[pos], pred, action, reserved, armed);
    i++;
  }
  fclose(f);
  cc_model_free(m);
  return 0;
}

int main(int argc, char** argv)
{
  if (argc > 1 && strcmp(argv[1], "--policy") == 0)
    return policy_replay(argc, argv);
  if (argc < 4) {
    fprintf(stderr, "usage: %s <bundle_dir> <period_ms> <windows.csv> [positive_label]\n", argv[0]);
    return 2;
  }
  double period_ms = atof(argv[2]);
  const char* pos_label = argc > 4 ? argv[4] : "1";
  char p_onnx[1024], p_pre[1024], err[512];
  snprintf(p_onnx, sizeof(p_onnx), "%s/model.onnx", argv[1]);
  snprintf(p_pre, sizeof(p_pre), "%s/preprocess.json", argv[1]);
  cc_model_t* m = cc_model_load(p_onnx, p_pre, err, sizeof(err));
  if (!m) {
    fprintf(stderr, "load failed: %s\n", err);
    return 1;
  }
  int pos = cc_model_class_index(m, pos_label);
  if (pos < 0) {
    fprintf(stderr, "positive label %s not among model classes\n", pos_label);
    return 1;
  }
  FILE* f = fopen(argv[3], "r");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", argv[3]);
    return 1;
  }
  printf("mean_prb,std_prb,min_prb,max_prb,slope_prb,last_prb,proba,pred,unmatched_inputs\n");
  char line[16384];
  while (fgets(line, sizeof(line), f)) {
    double x[MAX_WIN];
    size_t n = 0;
    char* save = NULL;
    line[strcspn(line, "\r\n")] = 0;
    for (char* tok = strtok_r(line, ",", &save); tok && n < MAX_WIN; tok = strtok_r(NULL, ",", &save))
      x[n++] = strtod(tok, NULL);
    if (n < 2)
      continue;
    double feat[CC_N_WINDOW_FEATURES], raw[CC_MAX_FEATURES], proba[8] = {0}, lat = 0;
    cc_window_features(x, n, period_ms, feat);
    int unmatched = 0; /* same lookup as xapp_congestion_ctrl.c: model input by name */
    for (size_t k = 0; k < cc_model_n_features(m); k++) {
      raw[k] = NAN;
      for (int w = 0; w < CC_N_WINDOW_FEATURES; w++)
        if (strcmp(cc_model_feature_name(m, k), cc_window_feature_names[w]) == 0)
          raw[k] = feat[w];
      unmatched += isnan(raw[k]);
    }
    int idx = -1;
    if (cc_model_predict(m, raw, &idx, proba, &lat, err, sizeof(err)) != 0) {
      fprintf(stderr, "predict failed: %s\n", err);
      return 1;
    }
    printf("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%d\n", feat[4], feat[5], feat[6], feat[7], feat[8], feat[9],
           proba[pos], cc_model_class_label(m, idx), unmatched);
  }
  fclose(f);
  cc_model_free(m);
  return 0;
}
