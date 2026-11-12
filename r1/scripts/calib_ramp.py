"""COMMAG-R1 burst-shape calibration (no xApp). Run with the backend venv (see r1env.py), from anywhere.

Profiles (UE1/UE2 DL, r1/traffic/burst.py, seed 6): ramp@15/20/25 Mbps per UE, 3 cycles, OFF ~30 s, plus
ramp@20 with a 45 s ramp-up (the frozen R1 profile). Throughput: iperf2 receiver-side per-second reports
(wall-clock, payload) per UE; gNB bytes only as cross-check. UE3: --ue3-rate (default 5M) DL + 10 Hz ping.
Ground truth: gNB [AURA-PRB] 1s cell load, ping RTT. Output: table + calib_ramp.json (with provenance) + CSVs.
"""
import asyncio, csv, json, os, re, statistics as st, subprocess, sys, time

import r1env as E
from r1env import k, pod, pdu_ip
import burst, run_guard
from services.dual_brain import channel_guard as g, gnb_truth
from services.k8s.experiment_ctx import ExperimentRef, use_experiment

EXP, SLUG, NS, H = E.EXP, E.SLUG, E.NS, E.H
UE3_RATE = sys.argv[sys.argv.index('--ue3-rate') + 1] if '--ue3-rate' in sys.argv else f"{E.ENV['ue3_rate_mbps']:g}M"
SEED, GAP = 6, 20.0  # idle seconds between profiles
PROFILES = [('ramp', 15, 3, {}), ('ramp', 20, 3, {}), ('ramp', 25, 3, {}),
            ('ramp', 20, 3, {'up_s': 45.0})]  # OFF ~30 s (burst.py default); last: slow 45 s ramp-up
SMOKE = '--smoke' in sys.argv  # one short cycle, output under results/smoke (pipeline check, not a measurement)
if SMOKE:
    PROFILES = [('ramp', 15, 1, {})]
SAMPLER_PY = os.path.join(E.R1, 'scripts', 'host_sampler.py')

LOAD0 = os.getloadavg()
PROV = E.provenance(__file__)
print('provenance:', PROV, flush=True)
BACKEND0 = run_guard.backend_worker()
assert BACKEND0, 'portal backend worker not found'
print('backend worker at start:', {k_: BACKEND0[k_] for k_ in ('pid', 'start', 'reload')}, flush=True)
OUT = os.path.join(E.RESULTS, 'smoke' if SMOKE else 'calibration', 'ramp-' + time.strftime('%Y%m%d-%H%M%S'))
os.makedirs(OUT, exist_ok=True)
SAMPLER = subprocess.Popen(['python3', '-I', SAMPLER_PY, f'{OUT}/host_load.jsonl',
                            '--interval', '60', '--until-pid', str(os.getpid())])
RETRIES = {'iperf_start': 0}
trf_name, trf_ip = pod('traffic-gen')
center = {ue: pod('virtual-ue', ue)[0] for ue in ('ue1', 'ue2')}
ue3_pod = pod('virtual-ue', 'ue3')[0]
rnti = {ue: g.ue_rnti(k('logs', p, '-c', 'ue', '--since=30s')) for ue, p in {**center, 'ue3': ue3_pod}.items()}
bind = {ue: pdu_ip(p) for ue, p in center.items()}
for p in center.values():
    E.route_to(p, trf_ip)
print('rnti', rnti, 'bind', bind, 'trf', trf_ip, flush=True)

plans = []
t = 0.0
for prof, peak, cyc, over in PROFILES:
    s = burst.schedule(prof, cycles=cyc, seed=SEED, peak=peak, **over)
    plans.append({'name': f'{prof}@{peak}' + (f"-up{over['up_s']:g}" if 'up_s' in over else ''), 'offset': t, 'sched': s})
    t += s['total_s'] + GAP
total = t + 10

# UE3: one constant iperf2 DL flow (same receiver-side CSV as UE1/UE2); no portal API needed.
bind['ue3'] = pdu_ip(ue3_pod)
E.route_to(ue3_pod, trf_ip)
_ue3_sched = {'cycles': [{'flows': [(0.0, float(UE3_RATE.rstrip('M')), total + 20)]}]}
_out = subprocess.run(['kubectl', '-n', NS, 'exec', ue3_pod, '-c', 'ue', '--',
                       *burst.remote_command(_ue3_sched, server_ip=trf_ip, bind_ip=bind['ue3'])],
                      capture_output=True, text=True).stdout
assert 'started' in _out, _out
E.start_ping(ue3_pod, trf_ip, total + 15)
time.sleep(10)  # UE3 background settles

# One combined schedule per center UE: every profile's flows shifted by its offset.
combined = {'cycles': [{'flows': [(a + pl['offset'], r_, d) for a, r_, d in c['flows']]}
                       for pl in plans for c in pl['sched']['cycles']]}
t_start = {}
for ue, p in center.items():
    t_start[ue] = time.time()
    out = subprocess.run(['kubectl', '-n', NS, 'exec', p, '-c', 'ue', '--',
                          *burst.remote_command(combined, server_ip=trf_ip, bind_ip=bind[ue])],
                         capture_output=True, text=True).stdout
    assert 'started' in out, out
T0 = t_start['ue1']
print(f"schedules started (UE2 skew {t_start['ue2'] - T0:.2f} s); session {total:.0f} s", flush=True)
aborted = None
t_end = time.time() + total
while time.time() < t_end:  # backend_reloaded: abort the measurement, keep what we have
    time.sleep(min(10, max(0, t_end - time.time())))
    try:
        run_guard.assert_same(BACKEND0)
    except run_guard.BackendReloaded as e:
        aborted = str(e)
        print('ABORT', aborted, flush=True)
        break

glog = k('logs', pod('gnb-rfsim')[0], '-c', 'gnb', '--timestamps', f'--since={int(time.time() - T0) + 60}s')
ping = k('exec', ue3_pod, '-c', 'ue', '--', 'cat', E.PING_LOG)
recv = {ue: [(t_ - T0, v) for t_, v in burst.receiver_mbps(k('exec', p, '-c', 'ue', '--', 'cat', burst.RECV_LOG))]
        for ue, p in {**center, 'ue3': ue3_pod}.items()}


async def _crash():
    with use_experiment(ExperimentRef(experiment_id=EXP, slug=SLUG)):
        return await g.save_crash_logs(OUT + '/crash')
guard_crash_files = asyncio.run(_crash())
prb_rows = gnb_truth.parse(glog)[0]
prb = [(p['t'] - T0, 100 * sum(x for _, x in p['ues'].values()) / p['budget']) for p in prb_rows]
speed = [(p['t'] - T0, p['budget'] / 61200) for p in prb_rows]  # sim/wall ratio (51 PRB real-time budget)
dl3 = [(gnb_truth._ts(l) - T0, float(m.group(4))) for l in glog.splitlines()
       if (m := g._GNB_DL.search(l)) and m.group(1) == rnti['ue3'] and gnb_truth._ts(l)]
rtt = [(float(a) - T0, float(b)) for a, b in re.findall(r'\[(\d+\.\d+)\].*?time=([\d.]+) ms', ping)]


def mean(v):
    return round(st.mean(v), 2) if v else None


def p95(v):
    return round(sorted(v)[int(0.95 * (len(v) - 1))], 1) if v else None


def within(series, spans, skip=(0, 0)):
    return [v for t_, v in series if any(a + skip[0] <= t_ <= b - skip[1] for a, b in spans)]


report = []
for pl in plans:
    off0 = pl['offset']
    ph = lambda name: [(off0 + c['phases'][name][0], off0 + c['phases'][name][1]) for c in pl['sched']['cycles']
                       if name in c['phases']]
    on, off, hold = ph('on'), ph('off'), ph('hold') or ph('on')
    cyc = []
    for a, b in on:
        s = [(t_, v) for t_, v in prb if a <= t_ <= b + 3]
        t60 = next((t_ for t_, v in s if v >= 60), None)
        t80 = next((t_ for t_, v in s if v >= 80 and (t60 is None or t_ >= t60)), None)
        on_v = [v for t_, v in s if t_ <= b]
        cyc.append({'max_prb': round(max((v for _, v in s), default=0), 1), 'crossed_80': t80 is not None,
                    'rise_60_80_s': round(t80 - t60, 1) if t60 is not None and t80 is not None else None,
                    'pinned_frac': round(sum(v >= 99 for v in on_v) / len(on_v), 2) if on_v else None})
    for c_, (a, b), (oa, ob) in zip(cyc, on, off):
        def longest_below(series, lim=50):
            best = run0 = None
            for t_, v in series:
                if v < lim:
                    run0 = t_ if run0 is None else run0
                    best = max(best or 0, t_ - run0 + 1)
                else:
                    run0 = None
            return round(best or 0, 1)
        c_['off_below50_s'] = longest_below([(t_, v) for t_, v in prb if oa <= t_ <= ob])
        c_['drained'] = c_['off_below50_s'] >= 10
        c_['s_at_88'] = sum(1 for t_, v in prb if a <= t_ <= b and v >= 88)  # ~1 sample/s
    rtt_off, rtt_hold = within(rtt, off, (3, 0)), within(rtt, hold)
    rtt_clear = bool(rtt_off and rtt_hold and st.mean(rtt_hold) > 2 * st.mean(rtt_off) and st.mean(rtt_hold) - st.mean(rtt_off) >= 5)
    rises = [c_['rise_60_80_s'] for c_ in cyc]
    accept = {'drain_every_off': all(c_['drained'] for c_ in cyc),
              'crossed_80_every_cycle': all(c_['crossed_80'] for c_ in cyc),
              'rise_ge3_most': sum(1 for r_ in rises if r_ is not None and r_ >= 3) > len(rises) / 2,
              's_at_88_per_cycle': [c_['s_at_88'] for c_ in cyc],
              'ue3_rtt_hold_clearly_above_off': rtt_clear}
    row = {'profile': pl['name'], 'cycles': cyc, 'accept': accept,
           'iperf_rx_mbps': {ue: {'off': mean(within(recv[ue], off, (3, 0))), 'on': mean(within(recv[ue], on, (2, 0))),
                                  'hold': mean(within(recv[ue], hold))} for ue in recv},
           'crossed_80_all': all(c['crossed_80'] for c in cyc),
           'rise_60_80_s': [c['rise_60_80_s'] for c in cyc],
           'pinned_frac_on': mean([c['pinned_frac'] for c in cyc if c['pinned_frac'] is not None]),
           'prb_on_mean': mean(within(prb, on, (2, 0))), 'prb_off_mean': mean(within(prb, off, (3, 0))),
           'sim_speed': (lambda w: {'mean': mean(w), 'min': min(w, default=None), 'max': max(w, default=None)})(
               [round(st.mean(v), 2) for a10 in range(int(off0), int(off0 + pl['sched']['total_s']), 10)
                if (v := [r_ for t_, r_ in speed if a10 <= t_ < a10 + 10])]),
           'ue3_mbps': {'off': mean(within(dl3, off, (3, 0))), 'on': mean(within(dl3, on, (2, 0))),
                        'hold': mean(within(dl3, hold))},
           'ue3_rtt_ms': {k_: {'mean': mean(v), 'p95': p95(v)} for k_, v in
                          (('off', within(rtt, off, (3, 0))), ('on', within(rtt, on, (2, 0))), ('hold', within(rtt, hold)))}}
    report.append(row)
    with open(f"{OUT}/prb_{pl['name'].replace('@', '_')}.csv", 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['t_wall_s', 'cell_prb_pct', 'offered_mbps_per_ue', 'sim_speed_ratio'])
        for (t_, v), (_, sr) in zip(prb, speed):
            if off0 - 2 <= t_ <= off0 + pl['sched']['total_s']:
                w.writerow([round(t_ - off0, 2), round(v, 1), round(burst.offered_mbps(pl['sched'], t_ - off0), 2), round(sr, 3)])
json.dump({'provenance': PROV, 'retries': RETRIES, 'aborted': aborted, 'backend_worker': BACKEND0, 'host_loadavg': {'start': LOAD0, 'end': os.getloadavg()}, 'seed': SEED, 'rnti': rnti, 'plans': [{'name': p['name'], 'offset': p['offset'], 'params': p['sched']['params']}
                                                  for p in plans], 'report': report}, open(OUT + '/calib_ramp.json', 'w'), indent=1)

print('host load avg start/end:', LOAD0, os.getloadavg())
print('iperf start retries this run:', RETRIES['iperf_start'])
_missing = [] if any(v > 0 for _, v in dl3) else ['ue3']
_center_missing = [ue for ue, p in center.items() if not any(m.group(1) == rnti[ue] for l in glog.splitlines() if (m := g._GNB_DL.search(l)) and float(m.group(4)) > 0)]
_missing += _center_missing
_missing += [f'{ue} iperf receiver' for ue, v in recv.items() if not v]
print('RUN', 'FAILED: no DL data for ' + ', '.join(_missing) if _missing else 'complete: every UE has DL data')
print('host load per minute:\n' + subprocess.run(['python3', '-I', SAMPLER_PY,
                                                   f'{OUT}/host_load.jsonl', '--summary'], capture_output=True, text=True).stdout)
print('crash evidence:', guard_crash_files or 'none (no container restarted)')
rampup = {}
for pl in plans:  # per-second (cell PRB, offered) pairs during every ramp-up, to see proportional vs greedy growth
    rows_ = []
    for ci, c in enumerate(pl['sched']['cycles'], 1):
        a_, b_ = (pl['offset'] + x for x in c['phases']['ramp_up'])
        for t_, v in prb:
            if a_ <= t_ < b_:
                off_ = burst.offered_mbps(pl['sched'], t_ - pl['offset'])
                rows_.append((ci, round(t_ - a_, 1), round(off_, 2), round(2 * off_ + float(UE3_RATE.rstrip('M')), 2), round(v, 1)))
    rampup[pl['name']] = rows_
    with open(f"{OUT}/rampup_prb_vs_offered_{pl['name'].replace('@', '_')}.csv", 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['cycle', 't_in_rampup_s', 'offered_mbps_per_center_ue', 'offered_mbps_cell_total', 'cell_prb_pct'])
        w.writerows(rows_)
print('\nramp-up: mean cell PRB % by offered cell load (Mbps, 5 Mbps bins, all cycles)')
for name, rows_ in rampup.items():
    bins = {}
    for _, _, _, tot, v in rows_:
        bins.setdefault(int(tot // 5) * 5, []).append(v)
    print(f"  {name:14} " + '  '.join(f"{b}:{round(st.mean(v))}" for b, v in sorted(bins.items())))
print('provenance:', PROV)
print(f"\n{'profile':9} {'speed':>5} {'drain<50%≥10s':>13} {'OFF<50% s/cycle':>16} {'cross80':>7} {'rise60→80 s':>16} {'s≥88%/cycle':>12} {'PRB on/off':>11} | "
      f"{'UE3 RTT off→hold mean/p95 ms':>30} {'clear':>5} | iperf rx Mbps hold (off): UE1, UE2, UE3")
for r_ in report:
    A, rt, rx = r_['accept'], r_['ue3_rtt_ms'], r_['iperf_rx_mbps']
    print(f"{r_['profile']:9} {r_['sim_speed']['mean']!s:>5} {A['drain_every_off']!s:>13} {str([c_['off_below50_s'] for c_ in r_['cycles']]):>16} "
          f"{A['crossed_80_every_cycle']!s:>7} {str(r_['rise_60_80_s']):>16} {str(A['s_at_88_per_cycle']):>12} "
          f"{str(r_['prb_on_mean']) + '/' + str(r_['prb_off_mean']):>11} | "
          f"{str(rt['off']['mean']) + '/' + str(rt['off']['p95']) + ' → ' + str(rt['hold']['mean']) + '/' + str(rt['hold']['p95']):>30} "
          f"{A['ue3_rtt_hold_clearly_above_off']!s:>5} | " + ', '.join(f"{rx[u]['hold']} ({rx[u]['off']})" for u in ('ue1', 'ue2', 'ue3')))
print('gNB DL goodput UE3 (cross-check, per simulated s) off/on/hold:', {r_['profile']: r_['ue3_mbps'] for r_ in report})
for r_ in report:  # compact PRB series per profile: 1 value per 2 s over the first ON+OFF cycle
    pl = next(p for p in plans if p['name'] == r_['profile'])
    c0 = pl['sched']['cycles'][0]
    a, b = pl['offset'] + c0['phases']['on'][0], pl['offset'] + c0['phases']['off'][1]
    series = [round(st.mean([v for t_, v in prb if s_ <= t_ < s_ + 2]) or 0) if any(s_ <= t_ < s_ + 2 for t_, _ in prb) else '-'
              for s_ in [a + 2 * i for i in range(int((b - a) / 2))]]
    print(f"PRB% {r_['profile']:9} cycle 1, every 2 s: {series}")
