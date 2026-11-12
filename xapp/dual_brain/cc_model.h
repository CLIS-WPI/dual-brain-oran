/*
 * Harmony Dual-Brain congestion xApp — model layer.
 *
 * Loads a NeuralSmith bundle (model.onnx + preprocess.json) and reproduces
 * neuralsmith.preprocess_data exactly: column order → impute → StandardScaler →
 * feature mask, then ONNX Runtime (CPU, 1 thread). Also computes the sliding-
 * window features the model is trained on, so training and serving share one
 * implementation (the xApp emits these features in collect mode).
 */
#ifndef HARMONY_CC_MODEL_H
#define HARMONY_CC_MODEL_H

#include <stddef.h>

#define CC_MAX_FEATURES 32
#define CC_N_WINDOW_FEATURES 10

/* Window features by name. 0-3: the original rApp names (slope per sample; MILCOM/dualbrain6 bundles).
 * 4-9 (dualbrain9, COMMAG-R1): the R1 dataset builder's names and definitions (r1/dataset/build.py):
 * mean_prb, std_prb (population), min_prb, max_prb, slope_prb (least squares, % per SECOND, from the KPM
 * period), last_prb. A model input is looked up by its preprocess.json name in this table. */
extern const char* const cc_window_feature_names[CC_N_WINDOW_FEATURES];

/* x[0..n-1] oldest → newest, sampled every period_ms. */
void cc_window_features(const double* x, size_t n, double period_ms, double out[CC_N_WINDOW_FEATURES]);

typedef struct cc_model cc_model_t;

/* NULL on error (message in err). */
cc_model_t* cc_model_load(const char* onnx_path, const char* preprocess_path, char* err, size_t errlen);
void cc_model_free(cc_model_t* m);

/* Feature names the model expects (preprocess.json "features"), in order. */
size_t cc_model_n_features(cc_model_t const* m);
const char* cc_model_feature_name(cc_model_t const* m, size_t i);

/* Index of the class label given as string (e.g. "1"), or -1. */
int cc_model_class_index(cc_model_t const* m, const char* label);
const char* cc_model_class_label(cc_model_t const* m, int idx);

/* raw: values in cc_model_feature_name() order (NaN = missing → impute).
 * Returns 0 on success; *label_idx = predicted class index, proba[] = per-class
 * probability (len = number of classes), *latency_us = preprocess + inference. */
int cc_model_predict(cc_model_t* m, const double* raw, int* label_idx, double* proba, double* latency_us,
                     char* err, size_t errlen);

#endif
