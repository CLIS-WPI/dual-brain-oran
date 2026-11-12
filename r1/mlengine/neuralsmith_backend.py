# CLIENT INTERFACE ONLY: NeuralSmith is proprietary and not included in this release.
# Request/response contract: r1/mlengine/NEURALSMITH_CONTRACT.md. Open replacement: r1/mlengine/flaml_backend.py.
"""NeuralSmith backend: the in-cluster NeuralSmith node (neuralsmith-cli 1.2.1, harmony-core/harmony-neuralsmith).

Trains on our train.csv only (the service splits off an internal stratified 5 % holdout for its own ONNX parity
check and fixes the CLI seed at 42; both are recorded). The returned bundle = model.onnx (ZipMap removed) +
preprocess.json (imputer fill, StandardScaler, feature mask); the xApp applies the same preprocessing in C, and
so do we here before ONNX inference on our time-ordered test.csv.
"""
from __future__ import annotations

import base64
import json
import os
import time

import httpx
import numpy as np

import common

URL = os.getenv('NEURALSMITH_URL', 'http://127.0.0.1:18090').rstrip('/')
SEED = 42  # fixed inside the NeuralSmith service (neuralsmith/service/app.py)
HOLDOUT_FRAC = 0.05  # the service's minimum internal holdout


def apply_preprocess(spec: dict, X: np.ndarray) -> np.ndarray:
    """Same as neuralsmith/service/export.py apply_preprocess (and the C xApp)."""
    X = np.asarray(X, dtype=np.float64).copy()
    nan = np.isnan(X)
    X[nan] = np.take(np.asarray(spec['impute']), np.nonzero(nan)[1])
    X = (X - np.asarray(spec['mean'])) / np.asarray(spec['scale'])
    return X[:, np.asarray(spec['mask'], dtype=bool)].astype(np.float32)


def train(train_csv: str, test_csv: str, spec: dict, out_dir: str, restricted: bool = False) -> dict:
    mode = 'fast' if restricted else (spec.get('mode') or 'fast++')
    t0 = time.time()
    with open(train_csv, 'rb') as fh:
        r = httpx.post(f'{URL}/train', timeout=3600,
                       files={'file': ('train.csv', fh, 'text/csv')},
                       data={'latency_budget_ms': str(spec['latency_budget_ms']), 'target_column': common.TARGET,
                             'positive_label': '1', 'mode': mode, 'holdout_frac': str(HOLDOUT_FRAC)})
    r.raise_for_status()
    body = r.json()
    wall = time.time() - t0
    onnx_path = os.path.join(out_dir, 'model.onnx')
    with open(onnx_path, 'wb') as fh:
        fh.write(base64.b64decode(body['onnx_b64']))
    pre = body['preprocess']
    json.dump(pre, open(os.path.join(out_dir, 'preprocess.json'), 'w'), indent=1)
    json.dump(body['report'], open(os.path.join(out_dir, 'neuralsmith_report.json'), 'w'), indent=1)
    feats, X, y = common.load_csv(test_csv, pre['features'])
    ev = common.evaluate(onnx_path, apply_preprocess(pre, X), y, out_dir)
    rep = body['report']
    return {**ev, 'algorithm': rep.get('algorithm'), 'train_time_s': rep.get('train_seconds'),
            'features_used': rep.get('features_used'), 'backend': 'neuralsmith', 'mode': mode, 'seed': SEED,
            'internal_holdout_frac': HOLDOUT_FRAC, 'n_train': rep.get('n_train'), 'cli_version': rep.get('cli_version'),
            'service_model_id': body.get('model_id'), 'request_wall_s': round(wall, 1), 'preprocess': 'preprocess.json'}
