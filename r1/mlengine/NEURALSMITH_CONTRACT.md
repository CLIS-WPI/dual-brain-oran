# NeuralSmith — client interface only

**NeuralSmith is proprietary and is NOT part of this release.** Only the code that *calls* it is included:

| File | Role |
|---|---|
| `r1/mlengine/neuralsmith_backend.py` | ML-engine backend that sends `train.csv` to a NeuralSmith service and evaluates the returned ONNX model on our time-ordered `test.csv` |
| `harmony_backend_subset/services/dual_brain/neuralsmith.py` | health/train client used by the Dual-Brain RIC helpers |

To reproduce the paper's ML results without NeuralSmith, use the **FLAML** backend (`r1/mlengine/flaml_backend.py`,
`server.py --backend flaml`), which implements the same training contract end to end. The archived NeuralSmith
predictions (`neuralsmith/pred.csv`, `report.json`) are in the DataPort archive, so every NeuralSmith metric in the
paper can be re-scored offline with `r1/dataset/evaluate.py`.

## Service contract (what a drop-in replacement must implement)

`GET /livez` → 200 when up. `GET /health` → `{"ok": bool, ...}` (ok only after the service's self-test training).

`POST /train` (multipart form):

| field | value |
|---|---|
| `file` | `train.csv`: header row, feature columns + target column, comma-separated |
| `target_column` | `label` |
| `positive_label` | `1` |
| `latency_budget_ms` | ONNX inference budget per sample (paper: 5) |
| `mode` | `fast++` (default) or `fast` (restricted retry) |
| `holdout_frac` | `0.05` (internal stratified holdout used only for the service's own ONNX parity check) |

Response 200 JSON:

```
{ "model_id": str,
  "onnx_b64": base64 ONNX model (binary classifier; ZipMap removed),
  "preprocess": { "features": [names in input order],
                  "impute": [fill per feature], "mean": [...], "scale": [...],   # x' = (x - mean) / scale after imputing NaN
                  "mask": [bool per feature] },                                  # features kept, in order
  "report": { "algorithm": str, "train_seconds": float, "features_used": [...], "n_train": int, "cli_version": str, ... } }
```

The xApp applies exactly this preprocessing in C (`xapp/dual_brain/cc_model.c`); `neuralsmith_backend.apply_preprocess`
is the Python mirror. Seed: fixed at 42 inside the service (version used: neuralsmith-cli 1.2.1, mode fast++, 8 CPUs).
