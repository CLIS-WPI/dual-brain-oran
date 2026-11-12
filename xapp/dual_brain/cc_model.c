#include "cc_model.h"

#include <cjson/cJSON.h>
#include <onnxruntime_c_api.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const char* const cc_window_feature_names[CC_N_WINDOW_FEATURES] = {
    "prb_mean", "prb_std", "prb_min", "prb_slope",                               /* original names */
    "mean_prb", "std_prb", "min_prb", "max_prb", "slope_prb", "last_prb"};       /* R1 builder names */

void cc_window_features(const double* x, size_t n, double period_ms, double out[CC_N_WINDOW_FEATURES])
{
  double sum = 0, mn = n ? x[0] : 0, mx = n ? x[0] : 0;
  for (size_t i = 0; i < n; i++) {
    sum += x[i];
    if (x[i] < mn)
      mn = x[i];
    if (x[i] > mx)
      mx = x[i];
  }
  double mean = n ? sum / (double)n : 0;
  double var = 0, sxy = 0, sxx = 0, tbar = n ? (n - 1) / 2.0 : 0;
  for (size_t i = 0; i < n; i++) {
    var += (x[i] - mean) * (x[i] - mean);
    sxy += ((double)i - tbar) * (x[i] - mean);
    sxx += ((double)i - tbar) * ((double)i - tbar);
  }
  double std = n ? sqrt(var / (double)n) : 0, slope = sxx > 0 ? sxy / sxx : 0;
  out[0] = mean;
  out[1] = std;
  out[2] = mn;
  out[3] = slope;
  out[4] = mean;
  out[5] = std;
  out[6] = mn;
  out[7] = mx;
  out[8] = period_ms > 0 ? slope * 1000.0 / period_ms : 0; /* % per second */
  out[9] = n ? x[n - 1] : 0;
}

struct cc_model {
  const OrtApi* api;
  OrtEnv* env;
  OrtSession* sess;
  OrtMemoryInfo* mem;
  char* input_name;
  char* out_label;
  char* out_proba;
  size_t n_feat;
  char* feat[CC_MAX_FEATURES];
  double impute[CC_MAX_FEATURES], mean[CC_MAX_FEATURES], scale[CC_MAX_FEATURES];
  int mask[CC_MAX_FEATURES];
  size_t n_used;
  int n_classes;
  char* classes[8];
};

static char* read_file(const char* path)
{
  FILE* f = fopen(path, "rb");
  if (!f)
    return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  char* buf = calloc(1, (size_t)sz + 1);
  if (buf && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    free(buf);
    buf = NULL;
  }
  fclose(f);
  return buf;
}

static int ort_fail(cc_model_t* m, OrtStatus* st, char* err, size_t errlen)
{
  if (!st)
    return 0;
  snprintf(err, errlen, "onnxruntime: %s", m->api->GetErrorMessage(st));
  m->api->ReleaseStatus(st);
  return -1;
}

static int load_preprocess(cc_model_t* m, const char* path, char* err, size_t errlen)
{
  char* txt = read_file(path);
  if (!txt) {
    snprintf(err, errlen, "cannot read %s", path);
    return -1;
  }
  cJSON* j = cJSON_Parse(txt);
  free(txt);
  if (!j) {
    snprintf(err, errlen, "invalid JSON in %s", path);
    return -1;
  }
  int rc = -1;
  cJSON* feats = cJSON_GetObjectItem(j, "features");
  cJSON* imp = cJSON_GetObjectItem(j, "impute");
  cJSON* mean = cJSON_GetObjectItem(j, "mean");
  cJSON* scale = cJSON_GetObjectItem(j, "scale");
  cJSON* mask = cJSON_GetObjectItem(j, "mask");
  cJSON* classes = cJSON_GetObjectItem(j, "classes");
  int n = cJSON_GetArraySize(feats);
  if (n <= 0 || n > CC_MAX_FEATURES || cJSON_GetArraySize(imp) != n || cJSON_GetArraySize(mean) != n
      || cJSON_GetArraySize(scale) != n || cJSON_GetArraySize(mask) != n || cJSON_GetArraySize(classes) < 2
      || cJSON_GetArraySize(classes) > 8) {
    snprintf(err, errlen, "preprocess.json: inconsistent features/impute/mean/scale/mask/classes");
    goto out;
  }
  m->n_feat = (size_t)n;
  m->n_used = 0;
  for (int i = 0; i < n; i++) {
    m->feat[i] = strdup(cJSON_GetArrayItem(feats, i)->valuestring);
    m->impute[i] = cJSON_GetArrayItem(imp, i)->valuedouble;
    m->mean[i] = cJSON_GetArrayItem(mean, i)->valuedouble;
    m->scale[i] = cJSON_GetArrayItem(scale, i)->valuedouble;
    m->mask[i] = cJSON_IsTrue(cJSON_GetArrayItem(mask, i));
    m->n_used += (size_t)m->mask[i];
  }
  m->n_classes = cJSON_GetArraySize(classes);
  for (int i = 0; i < m->n_classes; i++) {
    cJSON* c = cJSON_GetArrayItem(classes, i);
    char buf[64];
    if (cJSON_IsString(c))
      snprintf(buf, sizeof(buf), "%s", c->valuestring);
    else if (c->valuedouble == (double)(long long)c->valuedouble)
      snprintf(buf, sizeof(buf), "%lld", (long long)c->valuedouble);
    else
      snprintf(buf, sizeof(buf), "%g", c->valuedouble);
    m->classes[i] = strdup(buf);
  }
  rc = 0;
out:
  cJSON_Delete(j);
  return rc;
}

cc_model_t* cc_model_load(const char* onnx_path, const char* preprocess_path, char* err, size_t errlen)
{
  cc_model_t* m = calloc(1, sizeof(*m));
  m->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (load_preprocess(m, preprocess_path, err, errlen) != 0)
    goto fail;

  OrtSessionOptions* so = NULL;
  if (ort_fail(m, m->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "harmony-cc", &m->env), err, errlen)
      || ort_fail(m, m->api->CreateSessionOptions(&so), err, errlen))
    goto fail;
  m->api->SetIntraOpNumThreads(so, 1);
  m->api->SetInterOpNumThreads(so, 1);
  m->api->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL);
  OrtStatus* st = m->api->CreateSession(m->env, onnx_path, so, &m->sess);
  m->api->ReleaseSessionOptions(so);
  if (ort_fail(m, st, err, errlen))
    goto fail;
  if (ort_fail(m, m->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &m->mem), err, errlen))
    goto fail;

  OrtAllocator* alloc = NULL;
  m->api->GetAllocatorWithDefaultOptions(&alloc);
  size_t n_in = 0, n_out = 0;
  m->api->SessionGetInputCount(m->sess, &n_in);
  m->api->SessionGetOutputCount(m->sess, &n_out);
  if (n_in != 1 || n_out < 2) {
    snprintf(err, errlen, "model must have 1 input and label+probabilities outputs (got %zu/%zu)", n_in, n_out);
    goto fail;
  }
  char* tmp = NULL;
  m->api->SessionGetInputName(m->sess, 0, alloc, &tmp);
  m->input_name = strdup(tmp);
  m->api->AllocatorFree(alloc, tmp);
  m->api->SessionGetOutputName(m->sess, 0, alloc, &tmp);
  m->out_label = strdup(tmp);
  m->api->AllocatorFree(alloc, tmp);
  m->api->SessionGetOutputName(m->sess, 1, alloc, &tmp);
  m->out_proba = strdup(tmp);
  m->api->AllocatorFree(alloc, tmp);

  /* Probabilities must be a plain tensor (skl2onnx without ZipMap). */
  OrtTypeInfo* ti = NULL;
  m->api->SessionGetOutputTypeInfo(m->sess, 1, &ti);
  enum ONNXType ot;
  m->api->GetOnnxTypeFromTypeInfo(ti, &ot);
  m->api->ReleaseTypeInfo(ti);
  if (ot != ONNX_TYPE_TENSOR) {
    snprintf(err, errlen, "output '%s' is not a tensor (ZipMap export not supported)", m->out_proba);
    goto fail;
  }
  return m;
fail:
  cc_model_free(m);
  return NULL;
}

void cc_model_free(cc_model_t* m)
{
  if (!m)
    return;
  if (m->sess)
    m->api->ReleaseSession(m->sess);
  if (m->mem)
    m->api->ReleaseMemoryInfo(m->mem);
  if (m->env)
    m->api->ReleaseEnv(m->env);
  for (size_t i = 0; i < m->n_feat; i++)
    free(m->feat[i]);
  for (int i = 0; i < m->n_classes; i++)
    free(m->classes[i]);
  free(m->input_name);
  free(m->out_label);
  free(m->out_proba);
  free(m);
}

size_t cc_model_n_features(cc_model_t const* m) { return m->n_feat; }
const char* cc_model_feature_name(cc_model_t const* m, size_t i) { return i < m->n_feat ? m->feat[i] : NULL; }
const char* cc_model_class_label(cc_model_t const* m, int idx) { return idx >= 0 && idx < m->n_classes ? m->classes[idx] : NULL; }

int cc_model_class_index(cc_model_t const* m, const char* label)
{
  for (int i = 0; i < m->n_classes; i++)
    if (strcmp(m->classes[i], label) == 0)
      return i;
  return -1;
}

static double now_us(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

int cc_model_predict(cc_model_t* m, const double* raw, int* label_idx, double* proba, double* latency_us,
                     char* err, size_t errlen)
{
  double t0 = now_us();
  float x[CC_MAX_FEATURES];
  size_t k = 0;
  for (size_t i = 0; i < m->n_feat; i++) {
    if (!m->mask[i])
      continue;
    double v = isnan(raw[i]) ? m->impute[i] : raw[i];
    x[k++] = (float)((v - m->mean[i]) / m->scale[i]);
  }
  int64_t shape[2] = {1, (int64_t)m->n_used};
  OrtValue* in = NULL;
  if (ort_fail(m, m->api->CreateTensorWithDataAsOrtValue(m->mem, x, k * sizeof(float), shape, 2,
                                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in), err, errlen))
    return -1;
  const char* in_names[1] = {m->input_name};
  const char* out_names[1] = {m->out_proba};
  OrtValue* out = NULL;
  OrtStatus* st = m->api->Run(m->sess, NULL, in_names, (const OrtValue* const*)&in, 1, out_names, 1, &out);
  m->api->ReleaseValue(in);
  if (ort_fail(m, st, err, errlen))
    return -1;

  /* Predicted class = argmax(probabilities) — what the estimator's predict() does. */
  float* p = NULL;
  m->api->GetTensorMutableData(out, (void**)&p);
  int best = 0;
  for (int c = 0; c < m->n_classes; c++) {
    proba[c] = p[c];
    if (p[c] > p[best])
      best = c;
  }
  *label_idx = best;
  m->api->ReleaseValue(out);
  if (latency_us)
    *latency_us = now_us() - t0;
  return 0;
}
