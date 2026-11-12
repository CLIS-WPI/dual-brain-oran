"""Score predictions on the test split: onset metrics (headline) + row metrics (secondary).

usage: evaluate.py --dir <build.py output dir> --pred <csv with column 'pred' row-aligned with test.csv | persistence>
                   [--debounce N] [--out <name>.json]
Headline onset recall = fraction of onsets whose debounced action came before the onset (lead > 0); the
main-H rule uses it (largest H with onset recall ≥ 0.8 and FPR ≤ 0.1). Other variants are secondary.
Onset: the time the sustained condition (cell PRB ≥ threshold for min_duration_s) first becomes true in a test
cycle (rows where it already holds at the cycle's first kept row are not an onset). Action: the N-th consecutive
positive prediction (the xApp's debounce; default N = ceil(1000 / T_s) for a classifier, 1 for the persistence
baseline, which already applies the sustained rule). Per onset: lead = onset − first action in that cycle (ms,
> 0 = ahead; no action before the cycle ends → missed). Headline: N onsets, onset recall = fraction acted on
before the onset (debounced lead > 0), lead median / IQR over onsets with an action in the cycle, row-level FPR.
Secondary: accuracy, precision, recall, F1, confusion [[tn, fp], [fn, tp]] at row level.
Persistence baseline = predict sustained congestion at t (same rule as the label, evaluated at t, not t + H).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os


def row_metrics(y: list[int], p: list[int]) -> dict:
    tp = sum(1 for a, b in zip(y, p) if a and b)
    tn = sum(1 for a, b in zip(y, p) if not a and not b)
    fp = sum(1 for a, b in zip(y, p) if not a and b)
    fn = sum(1 for a, b in zip(y, p) if a and not b)
    div = lambda a, b: round(a / b, 4) if b else None
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {'n': len(y), 'accuracy': div(tp + tn, len(y)), 'precision': prec, 'recall': rec,
            'f1': round(2 * prec * rec / (prec + rec), 4) if prec and rec else 0.0, 'fpr': div(fp, fp + tn),
            'confusion': [[tn, fp], [fn, tp]]}


def first_action(ts: list[int], pred: list[int], n: int) -> int | None:
    run = 0
    for t, p in zip(ts, pred):
        run = run + 1 if p else 0
        if run >= n:
            return t
    return None


def onsets(rows: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    """rows (cycle, t_ms, sustained_now) in time order → [(cycle, onset_t_ms)], first 0→1 per cycle."""
    out, prev, seen = [], {}, set()
    for c, t, s in rows:
        if s and prev.get(c) == 0 and c not in seen:
            out.append((c, t))
            seen.add(c)
        prev[c] = s
    return out


def quantile(v: list[float], q: float) -> float:
    s = sorted(v)
    x = (len(s) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return s[lo] + (s[hi] - s[lo]) * (x - lo)


assert row_metrics([1, 0, 1, 0], [1, 1, 0, 0])['confusion'] == [[1, 1], [1, 1]]
assert first_action([0, 1, 2, 3], [1, 0, 1, 1], 2) == 3 and first_action([0, 1], [0, 0], 1) is None
assert onsets([(1, 0, 1), (1, 1, 0), (1, 2, 1), (1, 3, 1), (2, 4, 0), (2, 5, 1)]) == [(1, 2), (2, 5)]
assert quantile([1, 2, 3, 4], 0.5) == 2.5


def evaluate(idx: list[dict], y: list[int], pred: list[int], *, n: int, horizon_ms: int, ts_ms: int) -> dict:
    rows = [(int(r['cycle']), int(r['t_ms']), int(r['sustained_now'])) for r in idx]
    per = []
    for c, onset in onsets(rows):
        cyc = [(t, p) for (cc, t, _), p in zip(rows, pred) if cc == c]
        ts_, ps_ = [t for t, _ in cyc], [p for _, p in cyc]
        first, act = first_action(ts_, ps_, 1), first_action(ts_, ps_, n)
        lead_first = onset - first if first is not None else None
        lead = onset - act if act is not None else None
        per.append({'cycle': c, 'onset_ms': onset, 'first_positive_ms': first, 'action_ms': act,
                    'lead_first_ms': lead_first, 'lead_ms': lead,
                    'flagged_ge_h': lead_first is not None and lead_first >= horizon_ms})
    leads = [x['lead_ms'] for x in per if x['lead_ms'] is not None]
    rate = lambda f: round(sum(1 for x in per if f(x)) / len(per), 4) if per else None
    return {'n_onsets': len(per),
            # headline: the debounced action came before the onset (did the xApp act before congestion was
            # established?); the persistence baseline acts at the onset → 0 by construction
            'onset_recall': rate(lambda x: x['lead_ms'] is not None and x['lead_ms'] > 0),
            # variants (a perfect predictor flags exactly H ahead; its debounced action is (N-1)*T_s later)
            'onset_recall_variants': {
                'first_ge_h': rate(lambda x: x['flagged_ge_h']),
                'first_gt_0': rate(lambda x: (x['lead_first_ms'] or 0) > 0),
                'action_ge_h': rate(lambda x: x['lead_ms'] is not None and x['lead_ms'] >= horizon_ms),
                'action_ge_h_minus_debounce': rate(lambda x: x['lead_ms'] is not None
                                                   and x['lead_ms'] >= horizon_ms - (n - 1) * ts_ms),
                'action_gt_0': rate(lambda x: (x['lead_ms'] or 0) > 0)},
            'lead_ms_median': quantile(leads, 0.5) if leads else None,
            'lead_ms_iqr': [quantile(leads, 0.25), quantile(leads, 0.75)] if leads else None,
            'fpr': row_metrics(y, pred)['fpr'], 'debounce_n': n, 'per_onset': per,
            'secondary': row_metrics(y, pred)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--pred', required=True, help="predictions CSV (column 'pred') or 'persistence'")
    ap.add_argument('--debounce', type=int)
    ap.add_argument('--out')
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.dir, 'meta.json')))
    idx = list(csv.DictReader(open(os.path.join(a.dir, 'test_index.csv'))))
    y = [int(r['label']) for r in csv.DictReader(open(os.path.join(a.dir, 'test.csv')))]
    persistence = a.pred == 'persistence'
    pred = ([int(r['sustained_now']) for r in idx] if persistence
            else [int(float(r['pred'])) for r in csv.DictReader(open(a.pred))])
    assert len(pred) == len(idx) == len(y), 'predictions, test.csv and test_index.csv are not row-aligned'
    n = a.debounce or (1 if persistence else math.ceil(1000 / meta['ts_ms']))
    out = {'predictor': 'persistence baseline' if persistence else os.path.abspath(a.pred),
           'ts_ms': meta['ts_ms'], 'horizon_ms': meta['horizon_ms'], 'threshold_pct': meta['threshold_pct'],
           'min_duration_s': meta['min_duration_s'],
           **evaluate(idx, y, pred, n=n, horizon_ms=meta['horizon_ms'], ts_ms=meta['ts_ms'])}
    name = a.out or ('persistence_metrics.json' if persistence else 'classifier_metrics.json')
    json.dump(out, open(os.path.join(a.dir, name), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ('per_onset', 'secondary')}))


if __name__ == '__main__':
    main()
