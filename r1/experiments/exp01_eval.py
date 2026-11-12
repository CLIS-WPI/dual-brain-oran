"""exp01-eval: headline classifier metrics on independent runs (selection and evaluation are separate).

usage: exp01_eval.py --run-id <id> --select-run <exp01-select run_id> --ts <T_s> --horizon <H>[,<H2>...]
                     [--exploratory <H2>[,...]]
                     [--seeds 7,8] [--window 2000] [--min-duration 3] [--collection-min 20] [--stage all|collect|eval]
exp01-select (exp01_dataset.py) picked (T_s, H) on its 4-cycle validation split; those numbers are never headline.
Here: two new independent 20-min frozen-profile runs at T_s with different traffic seeds (one after the other,
nothing interleaved), and the models trained ONCE in exp01-select on its training cycles 1-9 at (T_s, H) —
NeuralSmith and FLAML, frozen, not retrained — are evaluated on ALL complete cycles of both runs. The persistence
baseline is scored on the same rows. Pooled over both runs (cycles renumbered run-by-run) = headline; per-run
numbers are reported too. Outputs in r1/results/exp01-eval/<run_id>/: raw/ (collector), run<k>/ (dataset, preds,
metrics), pooled/ (concatenated test rows + metrics), exp01_eval.csv, exp01_eval_summary.json (sim-speed stats
per run, model provenance, git commit, versions).
Several horizons (2026-11-04): one collection, each H evaluated with its own frozen select-run models in
run<k>/h<H>/ and pooled/h<H>/; H listed in --exploratory are labelled role=exploratory (chosen after the select
grid) and kept out of 'headline' (summary key 'exploratory').
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys

R1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R1, 'scripts'))
import r1env as E  # noqa: E402

START_COMMIT = E.git('rev-parse', 'HEAD')  # the commit the run started from (no commits during official runs)

PY = sys.executable
MLV = os.path.join(R1, '.venv-mlengine', 'bin', 'python')
PREDICTORS = ('persistence', 'neuralsmith', 'flaml')


def sh(*args):
    print('$', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), text=True, capture_output=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def score(d, predictor, model_root):
    """Write <predictor>_metrics.json in d (predictions for the trained models via predict.py)."""
    if predictor == 'persistence':
        r = sh(PY, os.path.join(R1, 'dataset', 'evaluate.py'), '--dir', d, '--pred', 'persistence')
    else:
        pd_ = os.path.join(d, predictor)
        os.makedirs(pd_, exist_ok=True)
        if not os.path.exists(os.path.join(pd_, 'pred.csv')):
            p = sh(MLV, os.path.join(R1, 'mlengine', 'predict.py'), '--model-dir', os.path.join(model_root, predictor),
                   '--test', os.path.join(d, 'test.csv'), '--out', os.path.join(pd_, 'pred.csv'))
            if p.returncode:
                raise SystemExit(p.stderr[-800:])
        r = sh(PY, os.path.join(R1, 'dataset', 'evaluate.py'), '--dir', d, '--pred', os.path.join(pd_, 'pred.csv'),
               '--out', f'{predictor}_metrics.json')
    if r.returncode:
        raise SystemExit(r.stderr[-800:])
    return json.load(open(os.path.join(d, f'{predictor}_metrics.json')))


def pool(run_dirs, out):
    """Concatenate test rows of several runs (cycles renumbered after the previous run's last cycle)."""
    os.makedirs(out, exist_ok=True)
    files = {'test.csv': None, 'test_index.csv': None, **{f'{p}/pred.csv': None for p in PREDICTORS[1:]}}
    rows = {f: [] for f in files}
    offset = 0
    for d in run_dirs:
        idx = list(csv.DictReader(open(os.path.join(d, 'test_index.csv'))))
        for r in idx:
            r['cycle'] = int(r['cycle']) + offset
        rows['test_index.csv'] += idx
        rows['test.csv'] += list(csv.DictReader(open(os.path.join(d, 'test.csv'))))
        for p in PREDICTORS[1:]:
            rows[f'{p}/pred.csv'] += list(csv.DictReader(open(os.path.join(d, p, 'pred.csv'))))
        offset = max(r['cycle'] for r in idx)
    for f, rr in rows.items():
        os.makedirs(os.path.dirname(os.path.join(out, f)), exist_ok=True)
        with open(os.path.join(out, f), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    meta = json.load(open(os.path.join(run_dirs[0], 'meta.json')))
    json.dump({**meta, 'pooled_from': run_dirs}, open(os.path.join(out, 'meta.json'), 'w'), indent=1)


def row(scope, predictor, m, horizon, role):
    s = m['secondary']
    lo, hi = m['lead_ms_iqr'] or [None, None]
    return {'horizon_ms': horizon, 'role': role, 'scope': scope, 'predictor': predictor, 'n_onsets': m['n_onsets'], 'onset_recall': m['onset_recall'],
            'lead_ms_median': m['lead_ms_median'], 'lead_ms_iqr_lo': lo, 'lead_ms_iqr_hi': hi, 'fpr': m['fpr'],
            'debounce_n': m['debounce_n'], 'test_rows': s['n'],
            'test_positive_rate': round((s['confusion'][1][0] + s['confusion'][1][1]) / s['n'], 4) if s['n'] else None,
            'accuracy': s['accuracy'], 'precision': s['precision'],
            'recall': s['recall'], 'f1': s['f1'], 'confusion_tn_fp_fn_tp': json.dumps(s['confusion']), **{f'recall_{k}': v for k, v in m['onset_recall_variants'].items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--select-run', required=True)
    ap.add_argument('--ts', type=int, required=True)
    ap.add_argument('--horizon', required=True, help='comma-separated H in ms')
    ap.add_argument('--exploratory', default='', help='H values chosen after the select grid')
    ap.add_argument('--seeds', default='7,8')
    ap.add_argument('--window', type=int, default=2000)
    ap.add_argument('--min-duration', type=float, default=3)
    ap.add_argument('--collection-min', type=float, default=20)
    ap.add_argument('--stage', choices=('all', 'collect', 'eval'), default='all')
    a = ap.parse_args()
    out = os.path.join(E.RESULTS, 'exp01-eval', a.run_id)
    os.makedirs(out, exist_ok=True)
    hs = [int(h) for h in a.horizon.split(',')]
    expl = {int(h) for h in a.exploratory.split(',') if h}
    assert expl <= set(hs) and set(hs) - expl, 'need >= 1 main H; exploratory H must be listed in --horizon'
    role = lambda h: 'exploratory' if h in expl else 'main'
    roots = {h: os.path.join(E.RESULTS, 'exp01', a.select_run, f'ts{a.ts}_h{h}') for h in hs}
    for h, model_root in roots.items():
        for p in ('neuralsmith', 'flaml'):
            assert os.path.exists(os.path.join(model_root, p, 'model.onnx')), f'no frozen {p} model in {model_root}'
    seeds = [int(s) for s in a.seeds.split(',')]
    raw = os.path.join(out, 'raw')
    os.makedirs(raw, exist_ok=True)
    runs = {}
    for s in seeds:
        rid = f'{a.run_id}-seed{s}'
        if a.stage in ('all', 'collect'):
            p = sh(PY, os.path.join(R1, 'collector', 'kpm_collector.py'), '--run-id', rid, '--ts', a.ts, '--seed', s,
                   '--collection-min', a.collection_min, '--out-root', raw)
            open(os.path.join(raw, f'{rid}.stdout.txt'), 'w').write(p.stdout + p.stderr)
        mp = os.path.join(raw, rid, 'meta.json')
        runs[s] = json.load(open(mp)) if os.path.exists(mp) else {'status': 'failed'}
        print(f'seed {s}: {runs[s].get("status")} rows={runs[s].get("rows")} sim_speed={runs[s].get("sim_speed_mean")}', flush=True)
    rows = []
    for h, model_root in (roots.items() if a.stage in ('all', 'eval') else ()):
        per_run_dirs = []
        for k, s in enumerate(seeds, 1):
            if runs[s].get('status') != 'complete':
                continue  # a failed run is kept and reported, never silently replaced
            d = os.path.join(out, f'run{k}-seed{s}', f'h{h}')
            n_cyc = len((runs[s].get('schedule') or {}).get('cycles', [])) or 13
            b = sh(PY, os.path.join(R1, 'dataset', 'build.py'), '--in', os.path.join(raw, f'{a.run_id}-seed{s}', 'telemetry.csv'),
                   '--ts', a.ts, '--window', a.window, '--horizon', h, '--threshold', 80, '--min-duration',
                   a.min_duration, '--train-cycles', '0', '--test-cycles', f'1-{n_cyc}', '--out', d)
            if b.returncode:
                raise SystemExit(b.stderr[-800:])
            for p in PREDICTORS:
                rows.append(row(f'run{k}-seed{s}', p, score(d, p, model_root), h, role(h)))
            per_run_dirs.append(d)
        if per_run_dirs:
            pooled = os.path.join(out, 'pooled', f'h{h}')
            pool(per_run_dirs, pooled)
            for p in PREDICTORS:
                rows.append(row('pooled', p, score(pooled, p, model_root), h, role(h)))
    if rows:
        with open(os.path.join(out, 'exp01_eval.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    keep = ('status', 'rows', 'gaps_gt_2ts', 'max_gap_ms', 'kpm_period_ms_median', 'crosses_80_per_cycle', 'sim_speed_mean',
            'sim_speed_start', 'skipped_starts', 'host_loadavg_start', 'host_loadavg_end', 'failure', 'missing', 'duration_s',
            'running_images', 'start_iso')
    summary = {'run_id': a.run_id, 'name': 'exp01-eval', 'select_run': a.select_run, 'git_commit': START_COMMIT,
               'provenance': E.provenance(__file__, extra=[os.path.join(R1, p) for p in (
                   'collector/kpm_collector.py', 'dataset/build.py', 'dataset/evaluate.py', 'mlengine/predict.py')]),
               'config': {'ts_ms': a.ts, 'horizons_ms': hs, 'exploratory_horizons_ms': sorted(expl), 'window_ms': a.window, 'min_duration_s': a.min_duration,
                          'threshold_pct': 80},
               'models': {f'h{h}': {'role': role(h), **{p: {'dir': os.path.relpath(os.path.join(mr, p), E.R1),
                                                             'model_sha256': sha(os.path.join(mr, p, 'model.onnx')),
                                                             'report': json.load(open(os.path.join(mr, p, 'report.json')))}
                                                         for p in ('neuralsmith', 'flaml')}} for h, mr in roots.items()},
               'runs': {s: {k: runs[s].get(k) for k in keep} for s in seeds},
               'eval_cycles': sum(len((runs[s].get('schedule') or {}).get('cycles', [])) for s in seeds
                                  if runs[s].get('status') == 'complete'),
               'headline': {f"h{r['horizon_ms']}/{r['predictor']}": r for r in rows if r['scope'] == 'pooled' and r['role'] == 'main'},
               'exploratory': {f"h{r['horizon_ms']}/{r['predictor']}": r for r in rows if r['scope'] == 'pooled' and r['role'] == 'exploratory'},
               'versions': json.load(open(os.path.join(R1, 'env', 'versions.json')))}
    json.dump(summary, open(os.path.join(out, 'exp01_eval_summary.json'), 'w'), indent=1)
    print(json.dumps({sec: {p: {k: r[k] for k in ('n_onsets', 'onset_recall', 'lead_ms_median', 'fpr')} for p, r in summary[sec].items()}
                      for sec in ('headline', 'exploratory')}, indent=1))
    print('files:', *sorted(os.listdir(out)))


if __name__ == '__main__':
    main()
