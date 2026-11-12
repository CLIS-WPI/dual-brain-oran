"""Offline 'deployed model' scoring on exp01-eval (no testbed): the model actually deployed in Exp-03 arm (c)
(= Exp-04 s1 artifact, H 2.5 s) and its H 7.5 s sibling (arm (d), exploratory), frozen, scored on the same exp01-eval
test rows as the headline models (both runs + pooled), with the same evaluate.py (default debounce).
These are labelled 'deployed model' and never replace the headline (select-run) numbers.

usage: exp01_eval_deployed.py [--eval-run exp01-eval-20261104] [--arms exp03-20261106-arms]
Output: r1/results/exp01-eval/<eval-run>-deployed/<scope>/h<H>/ (test rows copied, pred.csv, deployed_metrics.json)
+ deployed_summary.json (model sha256, provenance, metrics per scope; headline NeuralSmith alongside for reference).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

R1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(R1, 'results')
MLV = os.path.join(R1, '.venv-mlengine', 'bin', 'python')
MODELS = {2500: ('c', 'main H; deployed in Exp-03 arm (c) (Exp-04 s1 artifact)'),
          7500: ('d', 'exploratory H; deployed in Exp-03 arm (d)')}


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def run(*args):
    r = subprocess.run(list(map(str, args)), text=True, capture_output=True)
    if r.returncode:
        raise SystemExit(f'{args}: {r.stderr[-800:]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-run', default='exp01-eval-20261104')
    ap.add_argument('--arms', default='exp03-20261106-arms')
    a = ap.parse_args()
    src = os.path.join(RES, 'exp01-eval', a.eval_run)
    out = src + '-deployed'
    arms = os.path.join(RES, 'exp03', a.arms)
    man = json.load(open(os.path.join(arms, 'manifest.json')))
    head = json.load(open(os.path.join(src, 'exp01_eval_summary.json')))['headline']
    scopes = sorted(d for d in os.listdir(src) if d.startswith('run')) + ['pooled']
    summary = {'eval_run': a.eval_run, 'arms': a.arms, 'label': 'deployed model',
               'git_commit': subprocess.run(['git', '-C', R1, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
               'models': {}, 'metrics': {}}
    for h, (arm, role) in MODELS.items():
        md = os.path.join(arms, arm)
        model_sha = sha(os.path.join(md, 'model.onnx'))
        assert model_sha == man['arms'][arm]['model.onnx'], f'arm ({arm}) model hash != manifest'
        assert sha(os.path.join(md, 'preprocess.json')) == man['arms'][arm]['preprocess.json']
        summary['models'][f'h{h}'] = {'arm': arm, 'role': role, 'dir': os.path.relpath(md, R1), 'model_sha256': model_sha}
        for sc in scopes:
            s, d = os.path.join(src, sc, f'h{h}'), os.path.join(out, sc, f'h{h}')
            os.makedirs(d, exist_ok=True)
            for f in ('test.csv', 'test_index.csv', 'meta.json'):
                shutil.copy2(os.path.join(s, f), d)
            run(MLV, os.path.join(R1, 'mlengine', 'predict.py'), '--model-dir', md, '--test', os.path.join(d, 'test.csv'),
                '--out', os.path.join(d, 'pred.csv'))
            run(sys.executable, os.path.join(R1, 'dataset', 'evaluate.py'), '--dir', d, '--pred', os.path.join(d, 'pred.csv'),
                '--out', 'deployed_metrics.json')
            m = json.load(open(os.path.join(d, 'deployed_metrics.json')))
            sec = m['secondary']
            summary['metrics'][f'{sc}/h{h}'] = {
                'n_onsets': m['n_onsets'], 'onset_recall': m['onset_recall'], 'lead_ms_median': m['lead_ms_median'],
                'lead_ms_iqr': m['lead_ms_iqr'], 'fpr': m['fpr'], 'debounce_n': m['debounce_n'], 'test_rows': sec['n'],
                **{k: sec[k] for k in ('accuracy', 'precision', 'recall', 'f1', 'confusion')}}
        summary['metrics'][f'pooled/h{h}']['headline_neuralsmith_for_reference'] = head.get(f'h{h}/neuralsmith') \
            or json.load(open(os.path.join(src, 'exp01_eval_summary.json')))['exploratory'].get(f'h{h}/neuralsmith')
    json.dump(summary, open(os.path.join(out, 'deployed_summary.json'), 'w'), indent=1)
    for k, v in summary['metrics'].items():
        print(k, {x: v[x] for x in ('onset_recall', 'lead_ms_median', 'lead_ms_iqr', 'fpr', 'accuracy', 'f1', 'confusion')})


if __name__ == '__main__':
    main()
