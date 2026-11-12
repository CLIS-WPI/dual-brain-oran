"""When does UE3's backlog start, relative to the label onset and to cell PRB first reaching 85 %? (no-cap cycles)

usage: backlog_onset.py <oracle-check dir> [...]   (offline, reads gnb_aura.log, ue3_ping.log, events.jsonl)
Per no-cap cycle, times in s after ramp-up start (1 s resolution, gNB [AURA-PRB] lines and 1 s ping bins):
  backlog   first 1 s bin with UE3 RTT mean > 50 ms that stays > 50 ms for 3 consecutive bins
  onset80   first second of the label's sustained run (cell PRB >= 80 % for 3 consecutive s); label_true = onset80 + 3,
            the time a persistence policy with min_duration 3 s acts
  first85   first second with cell PRB >= 85 %, ignoring the first 3 s of ramp-up (1 s spike when the center UEs'
            iperf flows start in cycle 1)
Lead = event time - backlog (positive: the event comes after the backlog has started).
"""
import json
import os
import re
import statistics as st
import sys

import r1env  # noqa: F401  (sys.path)
import burst
from services.dual_brain import gnb_truth

N_SUST, RTT_MS, PRB_LAB, PRB_85 = 3, 50.0, 80.0, 85.0


def first_sustained(series, ok, n):
    """series: [(t, v)] at 1 s; first t whose n consecutive values satisfy ok."""
    for i in range(len(series) - n + 1):
        if all(ok(v) for _, v in series[i:i + n]):
            return series[i][0]
    return None


def analyse(d):
    evs = [json.loads(ln) for ln in open(os.path.join(d, 'events.jsonl'))]
    start = next(e for e in evs if e['event'] == 'started')
    t0, arms = start['t0'], start['arms']
    s = json.load(open(os.path.join(d, 'summary.json')))
    sched = burst.schedule('ramp', cycles=len(arms), seed=6, peak=s['peak_mbps_per_ue'], up_s=45.0)
    prb = [(r['t'], 100 * sum(x for _, x in r['ues'].values()) / r['budget'])
           for r in gnb_truth.parse(open(os.path.join(d, 'gnb_aura.log')).read())[0]]
    rtt = [(float(a), float(b)) for a, b in re.findall(r'\[(\d+\.\d+)\].*?time=([\d.]+) ms', open(os.path.join(d, 'ue3_ping.log')).read())]
    out = []
    for i, c in enumerate(sched['cycles']):
        if arms[i] != 'nocap':
            continue
        a, b = t0 + c['phases']['ramp_up'][0], t0 + c['phases']['off'][0]
        p = [(t - a, v) for t, v in prb if a <= t < b]
        bins = {}
        for t, v in rtt:
            if a <= t < b:
                bins.setdefault(int(t - a), []).append(v)
        r = [(k, st.mean(v)) for k, v in sorted(bins.items())]
        backlog = first_sustained(r, lambda v: v > RTT_MS, N_SUST)
        onset = first_sustained(p, lambda v: v >= PRB_LAB, N_SUST)
        f85 = next((t for t, v in p if v >= PRB_85 and t >= 3), None)
        rd = lambda x: None if x is None else round(x, 1)
        lead = lambda x: None if x is None or backlog is None else round(x - backlog, 1)
        out.append({'run': os.path.basename(d.rstrip('/')), 'ue3_mbps': s['ue3_rate_mbps'], 'cycle': i + 1,
                    'backlog_s': rd(backlog), 'onset80_s': rd(onset), 'label_true_s': rd(onset and onset + N_SUST),
                    'first85_s': rd(f85), 'onset80_minus_backlog': lead(onset),
                    'label_true_minus_backlog': lead(onset and onset + N_SUST), 'first85_minus_backlog': lead(f85),
                    'hold_start_s': c['phases']['hold'][0] - c['phases']['ramp_up'][0]})
    return out


if __name__ == '__main__':
    rows = [r for d in sys.argv[1:] for r in analyse(d)]
    for r in rows:
        print(json.dumps(r))
    json.dump(rows, open(os.path.join(r1env.RESULTS, 'calibration', 'backlog_onset.json'), 'w'), indent=1)
