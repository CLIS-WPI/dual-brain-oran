"""Shared by both backends: ONNX evaluation on our time-ordered test.csv, latency, size, report fields."""
from __future__ import annotations

import csv
import os
import time

import numpy as np
import onnxruntime as ort

TARGET = 'label'


def load_csv(path: str, features: list[str] | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(open(path)))
    feats = features or [c for c in rows[0] if c != TARGET]
    X = np.array([[float(r[f]) for f in feats] for r in rows], dtype=np.float64)
    y = np.array([int(float(r[TARGET])) for r in rows], dtype=np.int64)
    return feats, X, y


def session(onnx_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])


def predict(sess: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    """Class labels (0/1) from the model's first output."""
    name = sess.get_inputs()[0].name
    return np.asarray(sess.run(None, {name: X.astype(np.float32)})[0]).astype(np.int64).ravel()


def latency_ms(sess: ort.InferenceSession, X: np.ndarray, warmup: int = 100, runs: int = 1000) -> dict:
    """Single-row CPU latency: `warmup` discarded calls, then `runs` timed calls (brief §6 step 3)."""
    name = sess.get_inputs()[0].name
    rows = [X[i % len(X)][None, :].astype(np.float32) for i in range(warmup + runs)]
    for r in rows[:warmup]:
        sess.run(None, {name: r})
    t = []
    for r in rows[warmup:]:
        t0 = time.perf_counter_ns()
        sess.run(None, {name: r})
        t.append((time.perf_counter_ns() - t0) / 1e6)
    return {'p50': float(np.percentile(t, 50)), 'p99': float(np.percentile(t, 99)), 'runs': runs, 'warmup': warmup}


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    tp, tn = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 0)).sum())
    fp, fn = int(((y == 0) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
    div = lambda a, b: a / b if b else None
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {'accuracy': div(tp + tn, len(y)), 'precision': prec, 'recall': rec,
            'f1': 2 * prec * rec / (prec + rec) if prec and rec else 0.0, 'fpr': div(fp, fp + tn),
            'confusion': [[tn, fp], [fn, tp]]}


assert metrics(np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]))['confusion'] == [[1, 1], [1, 1]]


def evaluate(onnx_path: str, X_test: np.ndarray, y_test: np.ndarray, out_dir: str) -> dict:
    """Metrics on our test split + latency + size; writes pred.csv (column 'pred', row-aligned with test.csv)."""
    sess = session(onnx_path)
    p = predict(sess, X_test)
    with open(os.path.join(out_dir, 'pred.csv'), 'w') as fh:
        fh.write('pred\n' + ''.join(f'{int(v)}\n' for v in p))
    lat = latency_ms(sess, X_test)
    return {**metrics(y_test, p), 'onnx_latency_ms_p50': lat['p50'], 'onnx_latency_ms_p99': lat['p99'],
            'latency_runs': lat, 'size_kb': os.path.getsize(onnx_path) / 1024, 'n_test': int(len(y_test))}
