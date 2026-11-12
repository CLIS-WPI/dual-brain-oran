"""Oracle check: the forced 20 % reservation applied from ramp-up start (an ideal predictor) vs no cap, interleaved.

usage: oracle_check.py --ue3-rate 7 [--peak 28] [--pairs 3]
One continuous frozen-profile run (45 s up, 10 s hold, 5 s down, ~30 s OFF) of 2·pairs cycles, arms interleaved per
cycle B, A, B, A, … (B = no cap first, so the xApp can be armed during the preceding OFF for every A cycle).
Oracle (A): the congestion xApp (control, threshold policy at 0 %, protected_ue_ref = UE3, no self-release) is started
in the OFF before the cycle; it reserves as soon as >= 2 UEs are active, i.e. at ramp-up start, and is stopped at OFF
start (SIGTERM → SLICE DEL). Reserve/release times vs the schedule are recorded.
Measurement (the xApp occupies the KPM subscription, so ground truth only): per-UE/cell PRB from the gNB [AURA-PRB]
1 s lines, iperf2 receiver goodput per UE, UE3 raw 10 Hz ping. Per arm, pooled over its cycles and per cycle: UE3
goodput + RTT mean/p95 in hold and OFF, UE1+UE2 goodput (hold), PRB shares (hold), OFF drain (longest run < 50 %).
Isolation + fresh traffic generator first; traffic stopped on exit. Output: results/calibration/oracle-<rate>-<time>/.
"""
import asyncio
import json
import os
import re
import statistics as st
import sys
import time

import r1env as E
from r1env import k, pdu_ip, pod

import burst
from services.dual_brain import channel_guard as g, gnb_truth, profiles, ric
from services.k8s.experiment_ctx import ExperimentRef, use_experiment

arg = lambda key, d: type(d)(sys.argv[sys.argv.index(key) + 1]) if key in sys.argv else d
UE3, PEAK, PAIRS = arg('--ue3-rate', 7.0), arg('--peak', 28.0), arg('--pairs', 3)
ARM = lambda i: 'oracle' if i % 2 == 1 else 'nocap'  # cycle index 0-based: B, A, B, A, …


def p95(v):
    v = sorted(v)
    return v[int(0.95 * (len(v) - 1))] if v else None


async def main():
    out = os.path.join(E.RESULTS, 'calibration', f'oracle-{UE3:g}-' + time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(out)
    log = open(os.path.join(out, 'events.jsonl'), 'w')

    def ev(**x):
        x['wall_s'] = round(time.time(), 3)
        log.write(json.dumps(x) + '\n')
        log.flush()
        print(x, flush=True)

    prov = E.provenance(__file__)
    ev(event='isolated', **E.isolate(lambda e: None))
    trf_name, trf_ip = E.restart_trafficgen()
    import atexit
    atexit.register(E.stop_traffic)
    with use_experiment(ExperimentRef(experiment_id=E.EXP, slug=E.SLUG)):
        rows = await g.snapshot(window_s=20)
        if problem := g.check_mapping(rows):
            raise SystemExit(f'mapping: {problem}')
        prot = next(r for r in rows if r['imsi'] == E.IMSI['ue3'])
        rnti = {r['ue'].lower(): r['rnti'] for r in rows}
        pods0 = await g.pods_state()
        upods = E.ue_pods()
        for p in upods.values():
            E.route_to(p, trf_ip)
        bind = {u: pdu_ip(p) for u, p in upods.items()}
        sched = burst.schedule('ramp', cycles=2 * PAIRS, seed=6, peak=PEAK, up_s=45.0)
        prof = profiles.load('rfsim', {'kpm': {'period_ms': 250}, 'control': {
            'policy': 'threshold', 'threshold_cell_prb': 0, 'min_duration_samples': 1, 'trigger_consecutive': 1,
            'release_below_cell_prb': 0, 'release_after_normal_samples': 10 ** 9, 'min_hold_samples': 10 ** 9,
            'protected_ue_ref': int(prot['cu_ue_id'])}})
        await ric.stop_conflicting_kpm_monitor()
        await ric.stop()
        span = sched['total_s'] + 15
        k('exec', upods['ue3'], '-c', 'ue', '--', *burst.remote_command({'cycles': [{'flows': [(0.0, UE3, span)]}]},
                                                                      server_ip=trf_ip, bind_ip=bind['ue3']))
        E.start_ping(upods['ue3'], trf_ip, span)
        await asyncio.sleep(10)  # UE3 background settles before the first cycle
        t0 = time.time()
        for u in ('ue1', 'ue2'):
            r = k('exec', upods[u], '-c', 'ue', '--', *burst.remote_command(sched, server_ip=trf_ip, bind_ip=bind[u]))
            assert 'started' in r, r
        ev(event='started', t0=t0, arms=[ARM(i) for i in range(2 * PAIRS)])
        marks, trf_peak = [], 0
        for i, c in enumerate(sched['cycles']):
            ph = c['phases']
            if ARM(i) == 'oracle':
                arm_at = t0 + ph['ramp_up'][0] - 25  # in the previous OFF: KPM subscription takes ~12-16 s
                await asyncio.sleep(max(0, arm_at - time.time()))
                offset = await ric.start(prof, mode='control', reserve_pct=20)
                reserve_t = None
                while time.time() < t0 + ph['off'][0]:
                    evs, offset = await ric.read_events(prof, offset)
                    for e in evs:
                        if reserve_t is None and e.get('type') == 'control' and e.get('action') == 'reserve':
                            reserve_t = e['t_us'] / 1e6
                    cur, _ = E.trafficgen_memory()
                    trf_peak = max(trf_peak, cur or 0)
                    await asyncio.sleep(1)
                await ric.stop()  # SLICE DEL at OFF start
                rel_t = time.time()
                marks.append({'cycle': i + 1, 'reserve_s_after_ramp_start': reserve_t and round(reserve_t - (t0 + ph['ramp_up'][0]), 1),
                              'release_s_after_off_start': round(rel_t - (t0 + ph['off'][0]), 1)})
                ev(event='oracle_cycle', **marks[-1])
        await asyncio.sleep(max(0, t0 + sched['total_s'] + 3 - time.time()))
        changed = g.pod_changes(pods0, await g.pods_state())
        glog = k('logs', pod('gnb-rfsim')[0], '-c', 'gnb', '--timestamps', f'--since={int(time.time() - t0) + 30}s')
        ping = k('exec', upods['ue3'], '-c', 'ue', '--', 'cat', E.PING_LOG)
        recv = {u: burst.receiver_mbps(k('exec', p, '-c', 'ue', '--', 'cat', burst.RECV_LOG)) for u, p in upods.items()}
    E.stop_traffic()
    open(os.path.join(out, 'ue3_ping.log'), 'w').write(ping)
    open(os.path.join(out, 'gnb_aura.log'), 'w').write('\n'.join(ln for ln in glog.splitlines() if '[AURA-' in ln))
    prb = gnb_truth.parse(glog)[0]
    rtt = [(float(a), float(b)) for a, b in re.findall(r'\[(\d+\.\d+)\].*?time=([\d.]+) ms', ping)]

    def window(c, name, skip=0):
        a, b = c['phases'][name]
        return t0 + a + skip, t0 + b

    def stats(cyc_idx):
        cs = [sched['cycles'][i] for i in cyc_idx]
        inw = lambda t, name, skip=0: any(window(c, name, skip)[0] <= t < window(c, name, skip)[1] for c in cs)
        hold_prb = [r for r in prb if inw(r['t'], 'hold')]
        share = {u: round(st.mean(100 * r['ues'].get(rnti[u], (0, 0))[1] / r['budget'] for r in hold_prb), 1) for u in rnti} if hold_prb else {}
        drains = []
        for c in cs:
            run = best = 0
            for r in prb:
                if window(c, 'off')[0] <= r['t'] < window(c, 'off')[1]:
                    run = run + 1 if 100 * sum(x for _, x in r['ues'].values()) / r['budget'] < 50 else 0
                    best = max(best, run)
            drains.append(best)
        out_ = {}
        for name, skip in (('hold', 0), ('off', 3)):
            r3 = [v for t, v in rtt if inw(t, name, skip)]
            out_[name] = {'ue3_goodput_mbps': round(st.mean(v for t, v in recv['ue3'] if inw(t, name, skip)), 2),
                          'ue3_rtt_ms_mean': round(st.mean(r3), 1) if r3 else None, 'ue3_rtt_ms_p95': p95(r3)}
        out_['hold']['center_goodput_mbps'] = round(sum(st.mean(v for t, v in recv[u] if inw(t, 'hold')) for u in ('ue1', 'ue2')), 2)
        out_['hold']['prb_share_pct'] = share
        out_['hold']['cell_prb_pct'] = round(st.mean(100 * sum(x for _, x in r['ues'].values()) / r['budget'] for r in hold_prb), 1) if hold_prb else None
        out_['off_drain_s'] = drains
        out_['sim_speed'] = round(st.mean(r['budget'] / 61200 for r in prb if any(window(c, 'ramp_up')[0] <= r['t'] < window(c, 'off')[1] for c in cs)), 3)
        return out_

    arms = {a: [i for i in range(2 * PAIRS) if ARM(i) == a] for a in ('nocap', 'oracle')}
    res = {'ue3_rate_mbps': UE3, 'peak_mbps_per_ue': PEAK, 'arm_order': [ARM(i) for i in range(2 * PAIRS)],
           'pooled': {a: stats(ix) for a, ix in arms.items()},
           'per_cycle': [{'cycle': i + 1, 'arm': ARM(i), **stats([i])} for i in range(2 * PAIRS)],
           'oracle_marks': marks, 'pod_changes': changed, 'trafficgen_memory_peak_mib': round(trf_peak / 2**20, 1),
           'provenance': prov}
    json.dump(res, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    for a in ('nocap', 'oracle'):
        print(a, json.dumps(res['pooled'][a]))
    for c in res['per_cycle']:
        print(c['cycle'], c['arm'], 'hold', c['hold']['ue3_goodput_mbps'], c['hold']['ue3_rtt_ms_mean'], c['hold']['ue3_rtt_ms_p95'],
              '| off', c['off']['ue3_goodput_mbps'], c['off']['ue3_rtt_ms_mean'], c['off']['ue3_rtt_ms_p95'],
              '| center', c['hold']['center_goodput_mbps'], '| drain', c['off_drain_s'])
    print('oracle marks', marks, '| pod changes', changed, '| wrote', out)


if __name__ == '__main__':
    asyncio.run(main())
