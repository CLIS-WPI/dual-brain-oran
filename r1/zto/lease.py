"""Reservation lease from the training data (ZTO-Agent data curation step; not an LLM/spec field).

lease_s = p90 duration of the sustained-congestion episodes in the TRAINING rows (the dataset's sustained_now
column = the label rule evaluated at t) + H, clipped to [LEASE_MIN_S, LEASE_MAX_S]. Rendered into config.json
(runtime.lease_s); the xApp releases lease_s after reserving unless the model predicts congestion at expiry.
Episode = maximal run of sustained_now == 1 within one cycle with no KPM gap > 2·T_s; duration = samples × T_s.
"""
from __future__ import annotations

import csv
import math
import os

LEASE_MIN_S, LEASE_MAX_S = 10.0, 120.0


def episodes_s(rows: list[dict], ts_ms: int) -> list[float]:
    out, run, prev = [], 0, None
    for r in rows:
        t, c, on = int(r['t_ms']), r['cycle'], r['sustained_now'] in ('1', 1, True)
        contiguous = prev is not None and prev[1] == c and t - prev[0] <= 2 * ts_ms
        if run and (not on or not contiguous):
            out.append(run * ts_ms / 1000)
            run = 0
        run = run + 1 if on else 0
        prev = (t, c)
    if run:
        out.append(run * ts_ms / 1000)
    return out


def p90(v: list[float]) -> float:
    v = sorted(v)
    return v[max(0, math.ceil(0.9 * len(v)) - 1)]  # nearest rank


def compute(dataset_dir: str, ts_ms: int, horizon_ms: int) -> dict:
    rows = [r for r in csv.DictReader(open(os.path.join(dataset_dir, 'dataset.csv'))) if r['split'] == 'train']
    eps = episodes_s(rows, ts_ms)
    if not eps:
        raise ValueError('no sustained-congestion episode in the training data: lease undefined')
    raw = p90(eps) + horizon_ms / 1000
    return {'episodes_s': eps, 'n_episodes': len(eps), 'p90_episode_s': p90(eps), 'horizon_s': horizon_ms / 1000,
            'lease_raw_s': raw, 'lease_s': round(min(LEASE_MAX_S, max(LEASE_MIN_S, raw)), 1),
            'clip_s': [LEASE_MIN_S, LEASE_MAX_S], 'rule': 'p90(sustained episodes, train rows) + H, clipped'}


_r = [{'t_ms': i * 500, 'cycle': '1' if i < 10 else '2', 'sustained_now': s}
      for i, s in enumerate('0111001111' '1100000111')]
assert episodes_s(_r, 500) == [1.5, 2.0, 1.0, 1.5], episodes_s(_r, 500)  # a cycle boundary splits an episode
assert p90([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 9
