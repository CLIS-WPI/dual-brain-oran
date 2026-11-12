"""Exp-01: dataset grid. One 20-min frozen-traffic run per T_s → dataset per (T_s, H) → NeuralSmith + FLAML →
onset metrics vs the persistence baseline → exp01_grid.csv + exp01_summary.json with the main-(T_s, H) proposal.

usage: exp01_dataset.py --run-id <id> [--ts 100,250,500] [--horizons 0,1000,2500,5000,7500,10000]
                        [--window 2000] [--min-duration 3] [--collection-min 20] [--stage all|collect|grid]
                        [--train-cycles 1-9] [--test-cycles 10-13]
Writes only into r1/results/exp01/<run_id>/:
  raw/<run_id>-ts<T>/        collector output (telemetry.csv, meta.json, guard.jsonl, events, gnb.log [gitignored])
  ts<T>_h<H>/                dataset.csv, train/test.csv, split.json, meta.json, persistence_metrics.json,
                             neuralsmith/{report.json, model.onnx, pred.csv, metrics.json}, flaml/{...}
  exp01_grid.csv             one row per (T_s, H, predictor)
  exp01_summary.json         run_id, git_commit, provenance, versions snapshot, collector stats, rule, proposal
Main-H rule (agreed): largest H with headline onset recall (debounced action before the onset) >= 0.8 and FPR
<= 0.1, per T_s for NeuralSmith; the summary proposes (T_s, H) and lists the full grid — the choice is Milad's.
FLAML: time_budget = NeuralSmith's measured train_time_s on the same train.csv (both recorded).
Collector runs follow the measurement rules (start gate, live guard); a failed run is kept and reported, its
T_s is skipped in the grid, never silently rerun.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request

R1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R1, 'scripts'))
import r1env as E  # noqa: E402

START_COMMIT = E.git('rev-parse', 'HEAD')  # the commit the run started from (no commits during official runs)

PY = sys.executable
MLV = os.path.join(R1, '.venv-mlengine', 'bin', 'python')
PORTS = {'neuralsmith': 8095, 'flaml': 8096}
SPEC = {'task': 'classification', 'target_column': 'label', 'latency_budget_ms': 5, 'max_size_kb': 2048,
        'mode': 'fast++', 'seed': 42}


def sh(*args, **kw):
    print('$', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), text=True, capture_output=True, **kw)


def collect(out, run_id, ts, minutes):
    raw = os.path.join(out, 'raw')
    os.makedirs(raw, exist_ok=True)
    rid = f'{run_id}-ts{ts}'
    p = sh(PY, os.path.join(R1, 'collector', 'kpm_collector.py'), '--run-id', rid, '--ts', ts,
           '--collection-min', minutes, '--out-root', raw)
    open(os.path.join(raw, f'{rid}.stdout.txt'), 'w').write(p.stdout + p.stderr)
    meta_p = os.path.join(raw, rid, 'meta.json')
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {'status': 'failed', 'failure': p.stderr[-500:]}
    print(f'collect T_s={ts}: {meta.get("status")} rows={meta.get("rows")}', flush=True)
    return rid, meta


def start_servers(out):
    procs = []
    for be, port in PORTS.items():
        procs.append(subprocess.Popen([MLV, os.path.join(R1, 'mlengine', 'server.py'), '--backend', be, '--port', str(port)],
                                      env={**os.environ, 'MLENGINE_RUNS': os.path.join(out, 'mlengine-runs', be)},
                                      stdout=open(os.path.join(out, f'mlengine-{be}.log'), 'w'), stderr=subprocess.STDOUT))
    for _ in range(60):
        try:
            for port in PORTS.values():
                urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2)
            return procs
        except OSError:
            time.sleep(1)
    raise SystemExit('ML engine servers did not start')


def metrics_row(ts, h, predictor, m, rep=None, split=None):
    sec = m['secondary']
    lo, hi = (m['lead_ms_iqr'] or [None, None])
    return {'ts_ms': ts, 'horizon_ms': h, 'predictor': predictor,
            'test_rows': split['test']['rows'], 'test_positive_rate': split['test']['positive_rate'],
            'train_rows': split['train']['rows'], 'train_positive_rate': split['train']['positive_rate'],
            'n_onsets': m['n_onsets'], 'onset_recall': m['onset_recall'], 'lead_ms_median': m['lead_ms_median'],
            'lead_ms_iqr_lo': lo, 'lead_ms_iqr_hi': hi, 'fpr': m['fpr'], 'debounce_n': m['debounce_n'],
            'accuracy': sec['accuracy'], 'precision': sec['precision'], 'recall': sec['recall'], 'f1': sec['f1'],
            'algorithm': (rep or {}).get('algorithm'), 'train_time_s': (rep or {}).get('train_time_s'),
            'time_budget_s': (rep or {}).get('time_budget_s'), 'onnx_latency_ms_p50': (rep or {}).get('onnx_latency_ms_p50'),
            'onnx_latency_ms_p99': (rep or {}).get('onnx_latency_ms_p99'), 'size_kb': (rep or {}).get('size_kb'),
            'retried': (rep or {}).get('retried')}


def grid(out, telemetry, horizons, window, min_duration, train_cycles='1-9', test_cycles='10-13'):
    rows = []
    procs = start_servers(out)
    try:
        for ts, tel in telemetry.items():
            for h in horizons:
                d = os.path.join(out, f'ts{ts}_h{h}')
                b = sh(PY, os.path.join(R1, 'dataset', 'build.py'), '--in', tel, '--ts', ts, '--window', window,
                       '--horizon', h, '--threshold', 80, '--min-duration', min_duration,
                       '--train-cycles', train_cycles, '--test-cycles', test_cycles, '--out', d)
                if b.returncode:
                    print(b.stderr[-800:])
                    continue
                split = json.load(open(os.path.join(d, 'split.json')))
                sh(PY, os.path.join(R1, 'dataset', 'evaluate.py'), '--dir', d, '--pred', 'persistence')
                rows.append(metrics_row(ts, h, 'persistence', json.load(open(os.path.join(d, 'persistence_metrics.json'))),
                                        split=split))
                ns_time = None
                for be in ('neuralsmith', 'flaml'):
                    bd = os.path.join(d, be)
                    spec = dict(SPEC, time_budget_s=ns_time) if be == 'flaml' else dict(SPEC)
                    os.makedirs(bd, exist_ok=True)
                    json.dump(spec, open(os.path.join(bd, 'spec.json'), 'w'), indent=1)
                    c = sh(MLV, os.path.join(R1, 'mlengine', 'client.py'), '--url', f'http://127.0.0.1:{PORTS[be]}',
                           '--train', os.path.join(d, 'train.csv'), '--test', os.path.join(d, 'test.csv'),
                           '--spec', os.path.join(bd, 'spec.json'), '--out', bd)
                    if c.returncode:
                        print(f'{be} failed T_s={ts} H={h}:', c.stderr[-600:])
                        rows.append({'ts_ms': ts, 'horizon_ms': h, 'predictor': be, 'algorithm': 'FAILED'})
                        continue
                    rep = json.load(open(os.path.join(bd, 'report.json')))
                    ns_time = ns_time or rep['train_time_s']
                    sh(PY, os.path.join(R1, 'dataset', 'evaluate.py'), '--dir', d, '--pred', os.path.join(bd, 'pred.csv'),
                       '--out', f'{be}_metrics.json')
                    rows.append(metrics_row(ts, h, be, json.load(open(os.path.join(d, f'{be}_metrics.json'))), rep, split))
                print(f'T_s={ts} H={h}: ' + ' | '.join(f"{r['predictor']} recall={r.get('onset_recall')} fpr={r.get('fpr')}"
                                                         for r in rows if r['ts_ms'] == ts and r['horizon_ms'] == h), flush=True)
    finally:
        for p in procs:
            p.terminate()
    return rows


def propose(rows):
    ok = [r for r in rows if r['predictor'] == 'neuralsmith' and r.get('onset_recall') is not None
          and r['onset_recall'] >= 0.8 and r.get('fpr') is not None and r['fpr'] <= 0.1]
    per_ts = {}
    for r in ok:
        if r['ts_ms'] not in per_ts or r['horizon_ms'] > per_ts[r['ts_ms']]['horizon_ms']:
            per_ts[r['ts_ms']] = r
    best = max(per_ts.values(), key=lambda r: (r['horizon_ms'], r['onset_recall'], -r['fpr']), default=None)
    return {'rule': 'largest H with NeuralSmith onset_recall >= 0.8 and FPR <= 0.1 (per T_s)',
            'per_ts': {ts: {'horizon_ms': r['horizon_ms'], 'onset_recall': r['onset_recall'], 'fpr': r['fpr']}
                       for ts, r in sorted(per_ts.items())},
            'proposal': best and {'ts_ms': best['ts_ms'], 'horizon_ms': best['horizon_ms']},
            'note': 'proposal only; the main configuration is chosen by Milad'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--ts', default='100,250,500')
    ap.add_argument('--horizons', default='0,1000,2500,5000,7500,10000')
    ap.add_argument('--window', type=int, default=2000)
    ap.add_argument('--min-duration', type=float, default=3)
    ap.add_argument('--collection-min', type=float, default=20)
    ap.add_argument('--stage', choices=('all', 'collect', 'grid'), default='all')
    ap.add_argument('--train-cycles', default='1-9')
    ap.add_argument('--test-cycles', default='10-13')
    a = ap.parse_args()
    out = os.path.join(E.RESULTS, 'exp01', a.run_id)
    os.makedirs(out, exist_ok=True)
    prov = E.provenance(__file__, extra=[os.path.join(R1, p) for p in (
        'collector/kpm_collector.py', 'dataset/build.py', 'dataset/evaluate.py', 'mlengine/server.py',
        'mlengine/neuralsmith_backend.py', 'mlengine/flaml_backend.py', 'mlengine/common.py')])
    tss = [int(x) for x in a.ts.split(',')]
    coll = {}
    if a.stage in ('all', 'collect'):
        for ts in tss:  # one 20-min frozen-traffic run per T_s
            coll[ts] = collect(out, a.run_id, ts, a.collection_min)
    else:
        for ts in tss:
            mp = os.path.join(out, 'raw', f'{a.run_id}-ts{ts}', 'meta.json')
            coll[ts] = (f'{a.run_id}-ts{ts}', json.load(open(mp)) if os.path.exists(mp) else {'status': 'missing'})
    rows = []
    if a.stage in ('all', 'grid'):
        tel = {ts: os.path.join(out, 'raw', rid, 'telemetry.csv') for ts, (rid, m) in coll.items() if m.get('status') == 'complete'}
        rows = grid(out, tel, [int(h) for h in a.horizons.split(',')], a.window, a.min_duration, a.train_cycles, a.test_cycles)
        if rows:
            cols = list(dict.fromkeys(k for r in rows for k in r))
            with open(os.path.join(out, 'exp01_grid.csv'), 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
    keep = ('status', 'rows', 'gaps_gt_2ts', 'max_gap_ms', 'kpm_period_ms_median', 'crosses_80_per_cycle', 'sim_speed_mean',
            'sim_speed_start', 'skipped_starts', 'host_loadavg_start', 'host_loadavg_end', 'failure', 'missing',
            'duration_s', 'running_images', 'start_iso')
    summary = {'run_id': a.run_id, 'git_commit': START_COMMIT, 'provenance': prov,
               'versions': json.load(open(os.path.join(R1, 'env', 'versions.json'))),
               'args': vars(a), 'collection': {ts: {'run': rid, **{k: m.get(k) for k in keep}} for ts, (rid, m) in coll.items()},
               'grid_rows': len(rows), 'main_choice': propose(rows) if rows else None}
    json.dump(summary, open(os.path.join(out, 'exp01_summary.json'), 'w'), indent=1)
    print(json.dumps(summary['main_choice'], indent=1))
    print('files:', *sorted(os.listdir(out)))


if __name__ == '__main__':
    main()
