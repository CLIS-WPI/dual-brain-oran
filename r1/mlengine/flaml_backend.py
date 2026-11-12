"""FLAML backend (open AutoML baseline, Exp-05): same contract, same train/test, seed 42.

time_budget = spec['time_budget_s'] (the runner passes NeuralSmith's measured training wall-clock on the same
data so neither backend gets more search time; both are recorded). FLAML runs with its defaults (estimators,
metric); n_jobs matches NeuralSmith's CPU cap. The winning estimator is exported to ONNX without ZipMap:
LightGBM / XGBoost via onnxmltools, scikit-learn estimators via skl2onnx. No preprocessing is applied (FLAML's
numeric pipeline is the identity for these all-float features), so the ONNX input is the raw feature vector.
"""
from __future__ import annotations

import os
import time

import numpy as np

import common

SEED = 42
N_JOBS = int(os.getenv('MLENGINE_FLAML_N_JOBS', '8'))  # = NeuralSmith service CPUS


def to_onnx(model, n_features: int):
    from onnxmltools.convert.common.data_types import FloatTensorType
    name = type(model).__name__
    init = [('input', FloatTensorType([None, n_features]))]
    if name.startswith('LGBM'):
        import onnxmltools
        return onnxmltools.convert_lightgbm(model, initial_types=init, zipmap=False, target_opset=15)
    if name.startswith('XGB'):
        import onnxmltools
        return onnxmltools.convert_xgboost(model, initial_types=init, target_opset=15)
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType as SkFloat
    return convert_sklearn(model, initial_types=[('input', SkFloat([None, n_features]))],
                           options={id(model): {'zipmap': False}}, target_opset=15)


def train(train_csv: str, test_csv: str, spec: dict, out_dir: str, restricted: bool = False) -> dict:
    from flaml import AutoML
    budget = float(spec.get('time_budget_s') or 300)
    feats, X, y = common.load_csv(train_csv)
    automl = AutoML()
    kw = dict(task='classification', time_budget=budget, seed=SEED, n_jobs=N_JOBS, verbose=0,
              log_file_name=os.path.join(out_dir, 'flaml.log'))
    if restricted:  # budget loop retry: smaller search space (the analogue of NeuralSmith --mode fast)
        kw['estimator_list'] = ['lgbm']
    t0 = time.time()
    automl.fit(X_train=X.astype(np.float32), y_train=y, **kw)
    train_s = time.time() - t0
    model = automl.model.estimator
    onx = to_onnx(model, X.shape[1])
    onnx_path = os.path.join(out_dir, 'model.onnx')
    with open(onnx_path, 'wb') as fh:
        fh.write(onx.SerializeToString())
    _, Xt, yt = common.load_csv(test_csv, feats)
    ev = common.evaluate(onnx_path, Xt, yt, out_dir)
    sk = (automl.model.predict(Xt.astype(np.float32)) == common.predict(common.session(onnx_path), Xt)).mean()
    return {**ev, 'algorithm': f'{automl.best_estimator} ({type(model).__name__})', 'train_time_s': round(train_s, 1),
            'features_used': feats, 'backend': 'flaml', 'seed': SEED, 'time_budget_s': budget, 'n_jobs': N_JOBS,
            'n_train': int(len(y)), 'onnx_agreement': float(sk), 'best_config': automl.best_config,
            'flaml_metric': automl._state.metric if hasattr(automl, '_state') else None}
