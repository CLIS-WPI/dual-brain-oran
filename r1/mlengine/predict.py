"""Apply a frozen trained model (an ML-engine job output dir) to a test.csv → pred.csv (column 'pred').

usage: predict.py --model-dir <dir with model.onnx [+ preprocess.json]> --test test.csv --out pred.csv
NeuralSmith bundles carry preprocess.json (applied first, as in the xApp); FLAML models take raw features.
Feature order: preprocess.json 'features' if present, else the test.csv column order without 'label'.
"""
import argparse
import json
import os

import numpy as np

import common
import neuralsmith_backend as nsb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    pre_p = os.path.join(a.model_dir, 'preprocess.json')
    pre = json.load(open(pre_p)) if os.path.exists(pre_p) else None
    feats, X, y = common.load_csv(a.test, pre['features'] if pre else None)
    if pre:
        X = nsb.apply_preprocess(pre, X)
    p = common.predict(common.session(os.path.join(a.model_dir, 'model.onnx')), np.asarray(X))
    with open(a.out, 'w') as fh:
        fh.write('pred\n' + ''.join(f'{int(v)}\n' for v in p))
    print(json.dumps({'rows': int(len(p)), 'positive_pred': int(p.sum()), 'features': feats}))


if __name__ == '__main__':
    main()
