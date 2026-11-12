"""Telemetry → supervised table (features over a trailing window, sustained-congestion label at t + horizon).

usage: build.py --in <run>/telemetry.csv --ts 250 --window 2000 --horizon 750 --threshold 80 --min-duration 3
                --train-cycles 1-9 --test-cycles 10-13 --out <dir>
All times are wall-clock ms (the paper reports them in ms) and go into <dir>/meta.json with the other arguments.

Sustained congestion at τ: every KPM sample in the trailing min_duration window ending at τ has
cell_prb_util ≥ threshold (one 1-2 s post-idle MCS spike is not congestion). Label y(t) = congested(t + H);
the persistence baseline predicts congested(t) (causal, identical definition). H = 0 reproduces detection.
Features over the trailing window ending at t (window_ms / ts samples, t included): mean_prb, std_prb, min_prb,
max_prb, slope_prb (least squares, % per second), last_prb.
Split is time-ordered by the traffic schedule's cycles (telemetry `cycle` column); rows whose window or label
crosses a cycle-set boundary or a KPM gap > 2·T_s are dropped (counted in split.json).
Outputs: dataset.csv (t_ms, cycle, features, sustained_now, label), train.csv / test.csv (features + label, the
NeuralSmith input), test_index.csv (t_ms, cycle, phase, cell_prb_util, sustained_now), split.json, meta.json.
Score with evaluate.py (classifier or --pred persistence).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

FEATURES = ('mean_prb', 'std_prb', 'min_prb', 'max_prb', 'slope_prb', 'last_prb')
TARGET = 'label'


def cycles(spec: str) -> set[int]:
    a, _, b = spec.partition('-')
    return set(range(int(a), int(b or a) + 1))


def features(v: list[float], ts_ms: int) -> dict[str, float]:
    n = len(v)
    m = sum(v) / n
    xs = [i * ts_ms / 1000 for i in range(n)]
    xm = sum(xs) / n
    sxx = sum((x - xm) ** 2 for x in xs)
    slope = sum((x - xm) * (y - m) for x, y in zip(xs, v)) / sxx if sxx else 0.0
    return {'mean_prb': m, 'std_prb': math.sqrt(sum((y - m) ** 2 for y in v) / n), 'min_prb': min(v),
            'max_prb': max(v), 'slope_prb': slope, 'last_prb': v[-1]}


def sustained(prb: list[float], i: int, n: int, thr: float) -> int | None:
    """1 if prb[i-n+1 .. i] all ≥ thr, None if the window starts before the data."""
    return None if i - n + 1 < 0 else int(all(x >= thr for x in prb[i - n + 1:i + 1]))


_f = features([0, 10, 20], 1000)
assert abs(_f['slope_prb'] - 10) < 1e-9 and _f['last_prb'] == 20 and _f['max_prb'] == 20
assert sustained([70, 85, 90, 95], 3, 3, 80) == 1 and sustained([85, 75, 90, 95], 3, 3, 80) == 0
assert sustained([90, 90], 1, 3, 80) is None


def build(rows, *, ts, window, horizon, threshold, min_duration_s, train, test):
    w, h, d = window // ts, horizon // ts, max(1, round(min_duration_s * 1000 / ts))
    t = [int(r['t_ms']) for r in rows]
    prb = [float(r['cell_prb_util']) for r in rows]
    cyc = [int(r['cycle']) if r.get('cycle') else None for r in rows]
    split_of = lambda c: 'train' if c in train else 'test' if c in test else None
    gap = [False] + [b - a > 2 * ts for a, b in zip(t, t[1:])]
    out, dropped = [], {'history': 0, 'gap': 0, 'boundary': 0, 'unscheduled': 0}
    for i in range(len(rows)):
        s = split_of(cyc[i])
        if s is None:
            dropped['unscheduled'] += 1
            continue
        lo, j = i - max(w, d) + 1, i + h
        if lo < 0 or j >= len(rows) or sustained(prb, j, d, threshold) is None:
            dropped['history'] += 1
            continue
        if any(gap[lo + 1:j + 1]):
            dropped['gap'] += 1
            continue
        if split_of(cyc[lo]) != s or split_of(cyc[j]) != s:
            dropped['boundary'] += 1
            continue
        row = {'t_ms': t[i], 'cycle': cyc[i], 'split': s, 'phase': rows[i].get('phase', ''),
               'cell_prb_util': prb[i], **{k: round(v, 5) for k, v in features(prb[i - w + 1:i + 1], ts).items()},
               'sustained_now': sustained(prb, i, d, threshold), TARGET: sustained(prb, j, d, threshold)}
        out.append(row)
    return out, dropped, {'window_samples': w, 'horizon_samples': h, 'min_duration_samples': d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--ts', type=int, required=True)
    ap.add_argument('--window', type=int, required=True, help='ms')
    ap.add_argument('--horizon', type=int, required=True, help='ms')
    ap.add_argument('--threshold', type=float, default=80)
    ap.add_argument('--min-duration', type=float, default=3, help='s, sustained-congestion rule')
    ap.add_argument('--train-cycles', default='1-9')
    ap.add_argument('--test-cycles', default='10-13')
    ap.add_argument('--out', required=True, help='output directory')
    a = ap.parse_args()
    for name, v in (('window', a.window), ('horizon', a.horizon)):
        assert v % a.ts == 0, f'--{name} must be a multiple of --ts'
    assert 1 <= a.min_duration <= 10, '--min-duration range [1..10] s'
    assert 0 <= a.horizon <= 10000, '--horizon range [0..10000] ms (spec.schema.json)'
    rows = list(csv.DictReader(open(a.inp)))
    meta_in = os.path.join(os.path.dirname(os.path.abspath(a.inp)), 'meta.json')
    tel = json.load(open(meta_in)) if os.path.exists(meta_in) else {}
    if tel.get('T_s_ms') and tel['T_s_ms'] != a.ts:
        raise SystemExit(f"--ts {a.ts} does not match telemetry T_s {tel['T_s_ms']}")
    train, test = cycles(a.train_cycles), cycles(a.test_cycles)
    assert not train & test, 'train and test cycles overlap'
    out, dropped, samples = build(rows, ts=a.ts, window=a.window, horizon=a.horizon, threshold=a.threshold,
                                  min_duration_s=a.min_duration, train=train, test=test)
    os.makedirs(a.out, exist_ok=True)
    cols = ['t_ms', 'cycle', 'split', 'phase', 'cell_prb_util', *FEATURES, 'sustained_now', TARGET]
    with open(os.path.join(a.out, 'dataset.csv'), 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(out)
    for s in ('train', 'test'):
        with open(os.path.join(a.out, f'{s}.csv'), 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=[*FEATURES, TARGET], extrasaction='ignore')
            wr.writeheader()
            wr.writerows(r for r in out if r['split'] == s)
    with open(os.path.join(a.out, 'test_index.csv'), 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=['t_ms', 'cycle', 'phase', 'cell_prb_util', 'sustained_now'], extrasaction='ignore')
        wr.writeheader()
        wr.writerows(r for r in out if r['split'] == 'test')
    split = {s: {'rows': (n := sum(1 for r in out if r['split'] == s)),
                 'positive': (p := sum(r[TARGET] for r in out if r['split'] == s)),
                 'positive_rate': round(p / n, 4) if n else None,
                 'cycles': sorted(train if s == 'train' else test)} for s in ('train', 'test')}
    split['dropped'] = dropped
    json.dump(split, open(os.path.join(a.out, 'split.json'), 'w'), indent=1)
    json.dump({'in': os.path.abspath(a.inp), 'ts_ms': a.ts, 'window_ms': a.window, 'horizon_ms': a.horizon,
               'threshold_pct': a.threshold, 'min_duration_s': a.min_duration, 'op': '>=', **samples,
               'features': list(FEATURES), 'label': 'sustained(cell_prb_util >= threshold, min_duration) at t + horizon',
               'persistence_baseline': 'sustained(cell_prb_util >= threshold, min_duration) at t', 'telemetry_run': tel.get('run_id'),
               'telemetry_status': tel.get('status')},
              open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
    print(json.dumps(split))


if __name__ == '__main__':
    main()
