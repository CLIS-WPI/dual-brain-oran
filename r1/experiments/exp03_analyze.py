"""Exp-03 analysis (plan: experiments/exp03_plan.md) over the runs of one batch; system python3 (scipy, matplotlib).

usage: python3 exp03_analyze.py results/exp03/<batch>
Uses only complete, in-regime runs (status complete); others are listed with their status. Ground truth only:
gNB [AURA-PRB] 1 s lines (cell/UE PRB), iperf2 receiver per-second goodput, UE3 raw 10 Hz ping. Windows per cycle
(from the run's schedule, identical in every arm):
  eval   second half of ramp-up + hold         full   ramp-up start → next ramp-up start (last: OFF end)
  off    OFF start + 3 s → next ramp-up start  queue  last 5 s before ramp-up start (UE3 RTT only)
Per cycle: UE3 RTT mean/p95, UE3 goodput, UE1+UE2 goodput, PRB shares (eval), reserve/release vs schedule, false
reservations (reserve outside ramp-up start − 10 s … OFF start). Operational cost: fraction of time a reservation is
active while the cell is not congested (gNB 1 s: cell PRB >= 80 % for 3 consecutive s = congested), per cycle and per
run; center-UE goodput cost = UE1+UE2 goodput (eval / full) vs arm (a)'s mean for the same cycle index.
Summary per arm: median + IQR of per-cycle eval p95, pooled mean RTT, UE3 goodput, center cost. Mann–Whitney U
(two-sided) on per-cycle eval p95: (c) vs (a), (c) vs (b); (d) vs (a), (d) vs (b) exploratory. Oracle reference from
results/calibration/oracle-7-*/summary.json (not a tested arm).
Outputs in the batch dir: exp03_timeseries.csv, exp03_cycles.csv, exp03_summary.json, fig_outcome.pdf.
"""
from __future__ import annotations

import csv
import datetime as dt
import glob
import json
import os
import re
import statistics as st
import sys

R1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R1, 'collector'))
from speed_rules import GNB_COVERAGE_MIN, gnb_log_coverage  # noqa: E402
BUDGET_RT, CONG_PCT, CONG_S = 61200, 80.0, 3
_PRB, _UE = re.compile(r'budget=(\d+)'), re.compile(r'rnti=0x([0-9a-f]+):slice=(-?\d+):tx_rb=(\d+)')


def parse_aura(log: str) -> list[dict]:
    """gNB [AURA-PRB] 1 s rows {t, budget, ues: {rnti: (slice, tx_rb)}} from a kubectl --timestamps log.
    shortcut: same regexes as backend services/dual_brain/gnb_truth.parse (that module imports the k8s bridge, which
    the system python3 used here for scipy/matplotlib lacks); keep the two in sync if the AURA line format changes."""
    rows = []
    for line in log.splitlines():
        if '[AURA-PRB] 1s' not in line:
            continue
        try:
            t = dt.datetime.fromisoformat(line.split(' ', 1)[0].replace('Z', '')[:26]).replace(tzinfo=dt.timezone.utc).timestamp()
        except ValueError:
            continue
        b = int(_PRB.search(line).group(1))
        if b > 0:
            rows.append({'t': t, 'budget': b, 'ues': {r: (int(sl), int(x)) for r, sl, x in _UE.findall(line)}})
    return rows


def p95(v):
    v = sorted(v)
    return v[int(0.95 * (len(v) - 1))] if v else None


def mean(v):
    v = list(v)
    return round(st.mean(v), 3) if v else None


def congested(cell: list[tuple[float, float]]) -> list[tuple[float, bool]]:
    """gNB 1 s cell PRB → sustained-80 % label (>= 80 % for the last 3 consecutive 1 s rows)."""
    out, run = [], 0
    for t, v in cell:
        run = run + 1 if v >= CONG_PCT else 0
        out.append((t, run >= CONG_S))
    return out


def reserved_intervals(ctrl: list[dict], t_end: float) -> list[tuple[float, float]]:
    iv, start = [], None
    for e in sorted(ctrl, key=lambda e: e['t_us']):
        t = e['t_us'] / 1e6
        if e.get('action') == 'reserve' and start is None:
            start = t
        elif e.get('action') == 'release' and start is not None:
            iv.append((start, t))
            start = None
    if start is not None:
        iv.append((start, t_end))
    return iv


def run_cycles(d: str) -> tuple[list[dict], list[dict], dict]:
    m = json.load(open(os.path.join(d, 'run_meta.json')))
    raw = os.path.join(d, 'raw')
    t0, cyc = m['t0_wall_s'], m['schedule']['cycles']
    prb = parse_aura(open(os.path.join(raw, 'gnb_aura.log')).read())
    rnti = {r: u for r, u in m['rnti_to_ue'].items()}
    cell = [(r['t'], 100 * sum(x for _, x in r['ues'].values()) / r['budget']) for r in prb]
    cong = congested(cell)
    cov = gnb_log_coverage([r['t'] for r in prb], t0, m['t_end_wall_s'])
    prb_ok = cov >= GNB_COVERAGE_MIN  # else no PRB-based metric for this run (Exp-03 b3: last 41 s of the log only)
    rtt = [(float(a), float(b)) for a, b in re.findall(r'\[(\d+\.\d+)\].*?time=([\d.]+) ms', open(os.path.join(raw, 'ue3_ping.log')).read())]
    recv = {u: [tuple(x) for x in v] for u, v in json.load(open(os.path.join(raw, 'recv.json'))).items()}
    ctrl = [json.loads(ln) for ln in open(os.path.join(d, 'control_events.jsonl')) if '"control"' in ln]
    ctrl = [e for e in ctrl if e.get('type') == 'control']
    resv = reserved_intervals(ctrl, m['t_end_wall_s'])
    is_res = lambda t: any(a <= t < b for a, b in resv)
    ts_rows = [{'run_id': m['run_id'], 'arm': m['arm'], 't_rel_s': round(r['t'] - t0, 1), 'cell_prb_pct': round(c, 1),
                'congested': int(k), 'reserved': int(is_res(r['t'])),
                **{f'prb_pct_{u}': round(100 * r['ues'].get(x, (0, 0))[1] / r['budget'], 1) for x, u in rnti.items()},
                'sim_speed': round(r['budget'] / BUDGET_RT, 3)}
               for r, (_, c), (_, k) in zip(prb, cell, cong) if prb_ok and t0 <= r['t'] <= m['t_end_wall_s']]
    rows = []
    for i, c in enumerate(cyc):
        ru, hold, off = [(t0 + a, t0 + b) for a, b in (c['ramp_up'], c['hold'], c['off'])]
        nxt = t0 + cyc[i + 1]['ramp_up'][0] if i + 1 < len(cyc) else off[1]
        win = {'eval': ((ru[0] + ru[1]) / 2, hold[1]), 'full': (ru[0], nxt), 'off': (off[0] + 3, nxt), 'queue': (ru[0] - 5, ru[0])}
        sp = (m.get('sim_speed_cycle_means') or [])
        row = {'run_id': m['run_id'], 'arm': m['arm'], 'cycle': i + 1, 'off_len_s': round(nxt - off[0], 1),
               'sim_speed': sp[i] if i < len(sp) else None}
        for w, (a, b) in win.items():
            r3 = [v for t, v in rtt if a <= t < b]
            row[f'{w}_ue3_rtt_mean_ms'], row[f'{w}_ue3_rtt_p95_ms'] = mean(r3), p95(r3)
            if w != 'queue':
                row[f'{w}_ue3_goodput_mbps'] = mean(v for t, v in recv.get('ue3', []) if a <= t < b)
                cg = [mean(v for t, v in recv.get(u, []) if a <= t < b) for u in ('ue1', 'ue2')]
                row[f'{w}_center_goodput_mbps'] = round(sum(x for x in cg if x is not None), 3)
        ev = [r for r in prb if win['eval'][0] <= r['t'] < win['eval'][1]]
        for x, u in rnti.items():
            row[f'eval_prb_share_{u}'] = mean(100 * r['ues'].get(x, (0, 0))[1] / r['budget'] for r in ev) if prb_ok else None
        res = [e['t_us'] / 1e6 for e in ctrl if e.get('action') == 'reserve' and ru[0] - 10 <= e['t_us'] / 1e6 < off[0]]
        row['reserve_s_after_ramp_start'] = round(res[0] - ru[0], 1) if res else None
        rel = [e['t_us'] / 1e6 for e in ctrl if e.get('action') == 'release' and res and e['t_us'] / 1e6 > res[0]]
        row['release_s_after_off_start'] = round(rel[0] - off[0], 1) if rel else None
        row['false_reservations'] = sum(1 for e in ctrl if e.get('action') == 'reserve' and win['full'][0] <= e['t_us'] / 1e6 < win['full'][1]
                                        and not any(t0 + x['ramp_up'][0] - 10 <= e['t_us'] / 1e6 < t0 + x['off'][0] for x in cyc))
        fc = [(t, k) for t, k in cong if prb_ok and win['full'][0] <= t < win['full'][1]]
        row['res_active_not_congested_frac'] = round(sum(1 for t, k in fc if not k and is_res(t)) / len(fc), 4) if fc else None
        row['res_active_frac'] = round(sum(1 for t, _ in fc if is_res(t)) / len(fc), 4) if fc else None
        rows.append(row)
    full = [(t, k) for t, k in cong if prb_ok and t0 <= t <= m['t_end_wall_s']]
    run = {'run_id': m['run_id'], 'arm': m['arm'], 'status': m['status'], 'regime': m.get('regime'), 't0_wall_s': t0,
           'mean_speed': (m.get('regime') or {}).get('mean_cycle_speed'), 'gnb_log_coverage': cov, 'prb_metrics': prb_ok,
           'control_actions': m.get('control_actions'),
           'res_active_not_congested_frac': round(sum(1 for t, k in full if not k and is_res(t)) / len(full), 4) if full else None}
    return rows, ts_rows, run


def speed_balance(runs: list[dict], cycles: list[dict], by) -> dict:
    """Milad 2026-11-06: per-run mean speed by arm; Kruskal-Wallis across arms on per-run speeds; within arm (a)
    Spearman of per-cycle eval p95 vs per-cycle speed; (a) vs (c) restricted to time-adjacent run pairs (each (c) run
    with the nearest-in-time unpaired (a) run) — primary when the arms are unbalanced (KW p < 0.05). No run dropped."""
    from scipy.stats import kruskal, mannwhitneyu, spearmanr
    used = [r for r in runs if r.get('used')]
    per_arm = {a: [r['mean_speed'] for r in used if r['arm'] == a and r.get('mean_speed')] for a in sorted({r['arm'] for r in used})}
    out = {'per_run': [{k: r.get(k) for k in ('run_id', 'arm', 'mean_speed')} for r in used], 'per_arm_speeds': per_arm,
           'per_arm_median_speed': {a: st.median(v) for a, v in per_arm.items() if v}}
    groups = [v for v in per_arm.values() if v]
    try:
        kw = kruskal(*groups) if len(groups) >= 2 else None
        out['kruskal_wallis'] = kw and {'H': float(kw.statistic), 'p': float(kw.pvalue), 'k': len(groups)}
    except ValueError as exc:  # e.g. all speeds identical
        out['kruskal_wallis'] = {'error': str(exc)}
    out['unbalanced'] = bool(out['kruskal_wallis'] and out['kruskal_wallis'].get('p', 1) < 0.05)
    xa = [(r['sim_speed'], r['eval_ue3_rtt_p95_ms']) for r in cycles if r['arm'] == 'a' and r['sim_speed'] and r['eval_ue3_rtt_p95_ms'] is not None]
    if len(xa) >= 3:
        rho = spearmanr([x for x, _ in xa], [y for _, y in xa])
        out['arm_a_p95_vs_speed_spearman'] = {'rho': float(rho.statistic), 'p': float(rho.pvalue), 'n_cycles': len(xa)}
    a_runs = sorted((r for r in used if r['arm'] == 'a'), key=lambda r: r['t0_wall_s'])
    pairs, free = [], list(a_runs)
    for c in sorted((r for r in used if r['arm'] == 'c'), key=lambda r: r['t0_wall_s']):
        if free:
            best = min(free, key=lambda r: abs(r['t0_wall_s'] - c['t0_wall_s']))
            free.remove(best)
            pairs.append((best, c))
    pa = [r['eval_ue3_rtt_p95_ms'] for a, _ in pairs for r in cycles if r['run_id'] == a['run_id'] and r['eval_ue3_rtt_p95_ms'] is not None]
    pc = [r['eval_ue3_rtt_p95_ms'] for _, c in pairs for r in cycles if r['run_id'] == c['run_id'] and r['eval_ue3_rtt_p95_ms'] is not None]
    adj = {'pairs': [{'a': a['run_id'], 'c': c['run_id'], 'dt_s': round(c['t0_wall_s'] - a['t0_wall_s']),
                      'speed_a': a['mean_speed'], 'speed_c': c['mean_speed']} for a, c in pairs],
           'eval_p95_median_ms': {'a': st.median(pa) if pa else None, 'c': st.median(pc) if pc else None},
           'primary': out['unbalanced']}
    if pa and pc:
        u = mannwhitneyu(pc, pa, alternative='two-sided')
        adj['mann_whitney_c_vs_a_eval_p95'] = {'U': float(u.statistic), 'p': float(u.pvalue), 'n': [len(pc), len(pa)]}
    out['a_vs_c_adjacent_pairs'] = adj
    return out


def main(batch_dir: str):
    from scipy.stats import mannwhitneyu
    runs, cycles, ts = [], [], []
    for d in sorted(glob.glob(os.path.join(batch_dir, '*', 'run_meta.json'))):
        d = os.path.dirname(d)
        m = json.load(open(os.path.join(d, 'run_meta.json')))
        if m.get('status') != 'complete':
            runs.append({'run_id': m['run_id'], 'arm': m['arm'], 'status': m.get('status'), 'used': False})
            continue
        r, t, info = run_cycles(d)
        cycles += r
        ts += t
        runs.append(dict(info, used=True))
    if not cycles:
        raise SystemExit('no complete in-regime run')
    base = {}  # arm (a) center goodput per cycle index (mean over its runs)
    for w in ('eval', 'full'):
        for c in range(1, 14):
            v = [r[f'{w}_center_goodput_mbps'] for r in cycles if r['arm'] == 'a' and r['cycle'] == c]
            base[(w, c)] = st.mean(v) if v else None
    for r in cycles:
        for w in ('eval', 'full'):
            b = base[(w, r['cycle'])]
            r[f'{w}_center_goodput_cost_pct'] = round(100 * (1 - r[f'{w}_center_goodput_mbps'] / b), 2) if b else None
    for name, rows in (('exp03_cycles.csv', cycles), ('exp03_timeseries.csv', ts)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(batch_dir, name), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    arms = sorted({r['arm'] for r in cycles})
    by = lambda a, k: [r[k] for r in cycles if r['arm'] == a and r[k] is not None]
    q = lambda v: [round(x, 1) for x in st.quantiles(v, n=4)] if len(v) >= 2 else None
    summ = {}
    for a in arms:
        s = {'n_runs': len({r['run_id'] for r in cycles if r['arm'] == a}), 'n_cycles': len(by(a, 'cycle'))}
        for w in ('eval', 'full', 'off', 'queue'):
            p = by(a, f'{w}_ue3_rtt_p95_ms')
            s[w] = {'ue3_rtt_p95_median_ms': st.median(p) if p else None, 'ue3_rtt_p95_iqr_ms': q(p),
                    'ue3_rtt_mean_pooled_ms': mean(by(a, f'{w}_ue3_rtt_mean_ms'))}
            if w in ('eval', 'full', 'off'):
                s[w]['ue3_goodput_mbps'] = mean(by(a, f'{w}_ue3_goodput_mbps'))
                s[w]['center_goodput_mbps'] = mean(by(a, f'{w}_center_goodput_mbps'))
            if w in ('eval', 'full'):
                s[w]['center_goodput_cost_pct'] = mean(by(a, f'{w}_center_goodput_cost_pct'))
        s['res_active_not_congested_frac'] = mean(by(a, 'res_active_not_congested_frac'))
        s['res_active_not_congested_frac_per_run'] = [r['res_active_not_congested_frac'] for r in runs if r.get('used') and r['arm'] == a]
        s['false_reservations'] = sum(by(a, 'false_reservations'))
        s['reserve_s_after_ramp_start_median'] = st.median(by(a, 'reserve_s_after_ramp_start')) if by(a, 'reserve_s_after_ramp_start') else None
        summ[a] = s
    tests = {}
    for x, y, expl in (('c', 'a', False), ('c', 'b', False), ('d', 'a', True), ('d', 'b', True)):
        for w in ('eval', 'full'):
            u, v = by(x, f'{w}_ue3_rtt_p95_ms'), by(y, f'{w}_ue3_rtt_p95_ms')
            if u and v:
                r = mannwhitneyu(u, v, alternative='two-sided')
                tests[f'{x}_vs_{y}_{w}_p95'] = {'U': float(r.statistic), 'p': float(r.pvalue), 'n': [len(u), len(v)], 'exploratory': expl}
    out_speed = speed_balance(runs, cycles, by)
    oracle = sorted(glob.glob(os.path.join(R1, 'results', 'calibration', 'oracle-7-*', 'summary.json')))
    out = {'batch_dir': os.path.relpath(batch_dir, R1), 'runs': runs, 'per_arm': summ, 'tests_mann_whitney': tests,
           'speed_balance': out_speed,
           'oracle_reference': oracle and {'source': os.path.relpath(oracle[-1], R1), 'pooled': json.load(open(oracle[-1]))['pooled']},
           'windows': {'eval': 'second half of ramp-up + hold', 'full': 'ramp-up start to next ramp-up start',
                       'off': 'OFF start + 3 s to next ramp-up start', 'queue': 'last 5 s before ramp-up start'},
           'congested_rule': f'gNB 1 s cell PRB >= {CONG_PCT} % for {CONG_S} consecutive s'}
    json.dump(out, open(os.path.join(batch_dir, 'exp03_summary.json'), 'w'), indent=1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    for i, w in enumerate(('eval', 'full')):
        data = [by(a, f'{w}_ue3_rtt_p95_ms') for a in arms]
        ax[i].boxplot(data, tick_labels=[f'({a})' for a in arms])
        ax[i].set_title({'eval': 'Evaluation window', 'full': 'Full cycle'}[w], fontsize=9)
        ax[i].set_ylabel('UE3 RTT p95 per cycle [ms]', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(batch_dir, 'fig_outcome.pdf'))
    print(json.dumps({'per_arm': {a: {'eval_p95_median': s['eval']['ue3_rtt_p95_median_ms'], 'n_cycles': s['n_cycles']}
                                  for a, s in summ.items()}, 'tests': tests}, indent=1))


assert [k for _, k in congested([(0, 85), (1, 90), (2, 81), (3, 70), (4, 99)])] == [False, False, True, False, False]
assert reserved_intervals([{'t_us': 1e6, 'action': 'reserve'}, {'t_us': 5e6, 'action': 'release'},
                           {'t_us': 7e6, 'action': 'reserve'}], 9.0) == [(1.0, 5.0), (7.0, 9.0)]

def self_test():
    """Synthetic batch, 2 cycles × arms (a) and (c), known answers: (c) caps at ramp-up + 25 s, lifts at OFF + 10 s."""
    import tempfile
    T0, ph = 1_000_000.0, [{'ramp_up': [0, 45], 'hold': [45, 55], 'ramp_down': [55, 60], 'off': [60, 90]},
                          {'ramp_up': [90, 135], 'hold': [135, 145], 'ramp_down': [145, 150], 'off': [150, 180]}]
    iso = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + '000Z'
    hot = lambda r: 30 <= r % 90 < 60  # cell >= 80 % from ramp-up + 30 s to OFF start
    with tempfile.TemporaryDirectory() as root:
        for arm in ('a', 'c'):
            d = os.path.join(root, f'st-{arm}')
            os.makedirs(os.path.join(d, 'raw'))
            ctrl = [] if arm == 'a' else [{'t_us': (T0 + x) * 1e6, 'type': 'control', 'action': a}
                                          for x, a in ((25, 'reserve'), (70, 'release'), (80, 'rearm'), (115, 'reserve'), (160, 'release'))]
            json.dump({'run_id': f'st-{arm}', 'arm': arm, 'status': 'complete', 'regime': {'in_regime': True},
                       'control_actions': {}, 't0_wall_s': T0, 't_end_wall_s': T0 + 180, 'schedule': {'cycles': ph},
                       'rnti_to_ue': {'aa': 'ue1', 'cc': 'ue3'}, 'sim_speed_cycle_means': [1.7, 1.75]},
                      open(os.path.join(d, 'run_meta.json'), 'w'))
            open(os.path.join(d, 'raw', 'gnb_aura.log'), 'w').write('\n'.join(
                f'{iso(T0 + r)} [AURA-PRB] 1s budget=100000 rnti=0xaa:slice=-1:tx_rb={80000 if hot(r) else 20000}:tbs=1 '
                f'rnti=0xcc:slice=-1:tx_rb=10000:tbs=1' for r in range(0, 181)))
            rtt = lambda r: (200 if arm == 'a' else 50) if 45 <= r % 90 < 55 else 20
            open(os.path.join(d, 'raw', 'ue3_ping.log'), 'w').write('\n'.join(
                f'[{T0 + k / 10:.6f}] 64 bytes from 192.0.2.1: icmp_seq={k} ttl=63 time={rtt(k / 10)} ms' for k in range(1800)))
            json.dump({'ue1': [[T0 + r + 0.5, 20.0 if arm == 'a' else 16.0] for r in range(180)],
                       'ue2': [], 'ue3': [[T0 + r + 0.5, 7.0] for r in range(180)]}, open(os.path.join(d, 'raw', 'recv.json'), 'w'))
            with open(os.path.join(d, 'control_events.jsonl'), 'w') as fh:
                fh.writelines(json.dumps(e) + '\n' for e in ctrl)
        main(root)
        cyc = list(csv.DictReader(open(os.path.join(root, 'exp03_cycles.csv'))))
        sm = json.load(open(os.path.join(root, 'exp03_summary.json')))
        c = [r for r in cyc if r['arm'] == 'c']
        assert len(cyc) == 4 and all(os.path.exists(os.path.join(root, f)) for f in ('exp03_timeseries.csv', 'fig_outcome.pdf'))
        assert [float(r['reserve_s_after_ramp_start']) for r in c] == [25.0, 25.0], c
        assert [float(r['release_s_after_off_start']) for r in c] == [10.0, 10.0] and all(r['false_reservations'] == '0' for r in c)
        assert sm['per_arm']['a']['eval']['ue3_rtt_p95_median_ms'] == 200 and sm['per_arm']['c']['eval']['ue3_rtt_p95_median_ms'] == 50
        assert abs(sm['per_arm']['c']['eval']['center_goodput_cost_pct'] - 20.0) < 1e-6, sm['per_arm']['c']['eval']
        # reserved 25..70 of 90 s per cycle; congested (>= 80 % for 3 s) from +32..+60 → 17 s reserved and not congested
        f = float(c[0]['res_active_not_congested_frac'])
        assert abs(f - 17 / 90) < 0.02, f
        assert 'c_vs_a_eval_p95' in sm['tests_mann_whitney']
        sb = sm['speed_balance']
        assert sb['a_vs_c_adjacent_pairs']['pairs'][0]['a'] == 'st-a' and sb['a_vs_c_adjacent_pairs']['eval_p95_median_ms'] == {'a': 200, 'c': 50}
        assert set(sb['per_arm_speeds']) == {'a', 'c'}
    print('exp03_analyze self-test: ok')


if __name__ == '__main__':
    self_test() if sys.argv[1:] == ['--self-test'] else main(sys.argv[1])
