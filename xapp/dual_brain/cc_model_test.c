/*
 * Parity test: the xApp's C model path (preprocess.json + ONNX Runtime) must
 * reproduce NeuralSmith's own predictions on the held-out file the /train
 * wrapper exports (holdout.csv: features, label, ns_pred, ns_proba).
 *
 *   cc_model_test <bundle_dir> [positive_label=1] [proba_tol=1e-4]
 *
 * Exit 0 only when every row's class matches ns_pred and |p - ns_proba| <= tol.
 * Also prints per-inference latency (p50/p99) against the 5 ms budget.
 */
#include "cc_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_COLS 64
#define MAX_ROWS 100000

static int split_csv(char* line, char** cols, int max)
{
  int n = 0;
  char* save = NULL;
  line[strcspn(line, "\r\n")] = 0;
  for (char* tok = strtok_r(line, ",", &save); tok && n < max; tok = strtok_r(NULL, ",", &save))
    cols[n++] = tok;
  return n;
}

static int cmp_d(const void* a, const void* b)
{
  double x = *(const double*)a, y = *(const double*)b;
  return (x > y) - (x < y);
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    fprintf(stderr, "usage: %s <bundle_dir> [positive_label] [proba_tol]\n", argv[0]);
    return 2;
  }
  const char* pos_label = argc > 2 ? argv[2] : "1";
  double tol = argc > 3 ? atof(argv[3]) : 1e-4;
  char p_onnx[1024], p_pre[1024], p_hold[1024], err[512];
  snprintf(p_onnx, sizeof(p_onnx), "%s/model.onnx", argv[1]);
  snprintf(p_pre, sizeof(p_pre), "%s/preprocess.json", argv[1]);
  snprintf(p_hold, sizeof(p_hold), "%s/holdout.csv", argv[1]);

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

  FILE* f = fopen(p_hold, "r");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", p_hold);
    return 1;
  }
  char line[8192], *cols[MAX_COLS];
  if (!fgets(line, sizeof(line), f)) {
    fprintf(stderr, "empty holdout\n");
    return 1;
  }
  int nh = split_csv(line, cols, MAX_COLS);
  int fidx[CC_MAX_FEATURES], i_pred = -1, i_proba = -1;
  char hdr_copy[MAX_COLS][64];
  for (int c = 0; c < nh; c++)
    snprintf(hdr_copy[c], sizeof(hdr_copy[c]), "%s", cols[c]);
  for (size_t k = 0; k < cc_model_n_features(m); k++) {
    fidx[k] = -1;
    for (int c = 0; c < nh; c++)
      if (strcmp(hdr_copy[c], cc_model_feature_name(m, k)) == 0)
        fidx[k] = c;
    if (fidx[k] < 0) {
      fprintf(stderr, "holdout.csv lacks feature column %s\n", cc_model_feature_name(m, k));
      return 1;
    }
  }
  for (int c = 0; c < nh; c++) {
    if (strcmp(hdr_copy[c], "ns_pred") == 0)
      i_pred = c;
    if (strcmp(hdr_copy[c], "ns_proba") == 0)
      i_proba = c;
  }
  if (i_pred < 0 || i_proba < 0) {
    fprintf(stderr, "holdout.csv lacks ns_pred/ns_proba\n");
    return 1;
  }

  static double lat[MAX_ROWS];
  int rows = 0, label_mismatch = 0, proba_mismatch = 0;
  double max_diff = 0;
  while (fgets(line, sizeof(line), f) && rows < MAX_ROWS) {
    if (split_csv(line, cols, MAX_COLS) < nh)
      continue;
    double raw[CC_MAX_FEATURES], proba[8];
    for (size_t k = 0; k < cc_model_n_features(m); k++)
      raw[k] = strtod(cols[fidx[k]], NULL);
    int idx = -1;
    if (cc_model_predict(m, raw, &idx, proba, &lat[rows], err, sizeof(err)) != 0) {
      fprintf(stderr, "predict failed: %s\n", err);
      return 1;
    }
    /* ns_pred may be written as "1" or "1.0" */
    double want = strtod(cols[i_pred], NULL);
    double got = strtod(cc_model_class_label(m, idx), NULL);
    if (want != got)
      label_mismatch++;
    double d = fabs(proba[pos] - strtod(cols[i_proba], NULL));
    if (d > max_diff)
      max_diff = d;
    if (d > tol)
      proba_mismatch++;
    rows++;
  }
  fclose(f);
  qsort(lat, (size_t)rows, sizeof(double), cmp_d);
  double p50 = rows ? lat[rows / 2] / 1000.0 : 0, p99 = rows ? lat[(int)(rows * 0.99)] / 1000.0 : 0;
  printf("{\"rows\":%d,\"label_mismatch\":%d,\"proba_mismatch\":%d,\"max_proba_diff\":%.3g,"
         "\"latency_ms_p50\":%.4f,\"latency_ms_p99\":%.4f}\n",
         rows, label_mismatch, proba_mismatch, max_diff, p50, p99);
  cc_model_free(m);
  return (rows > 0 && label_mismatch == 0 && proba_mismatch == 0 && p99 < 5.0) ? 0 : 1;
}
