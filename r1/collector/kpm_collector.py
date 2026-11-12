"""R1 telemetry collector: frozen traffic + FlexRIC KPM at T_s → data/raw/<run_id>/telemetry.csv + meta.json.

usage (backend venv, see r1/scripts/r1env.py):
  kpm_collector.py --run-id <id> [--ts 250] [--collection-min 20] [--seed 6] [--out-root r1/data/raw]
                   [--traffic frozen|none] [--no-start-gate]

Sources, all on one wall clock (CLOCK_REALTIME on the host; kubectl --timestamps UTC):
  KPM (FlexRIC, xApp xapp_congestion_ctrl in collect mode, period T_s): per-UE RRU.PrbTotDl % of cell PRBs and
      their sum = cell_prb_util. One telemetry row per KPM indication.
  gNB MAC stats (~1 s, sample-and-hold, causal): DL MCS, DL BLER, UL SNR per UE (`UE xxxx: ulsch_rounds … SNR`).
  gNB [AURA-PRB] 1 s lines: sim/wall speed ratio (budget / 61200) and a 1 s cell-PRB cross-check.
  iperf2 receiver-side per-second reports in each UE pod: thp_dl_ue* (Mbps, wall-clock, payload).
  UE3 ping at 10 Hz: rtt_ue3_ms = mean RTT of the pings sent in (t − T_s, t]; empty → blank.
Traffic (--traffic frozen): ramp@20, 45 s up, 10 s hold, 5 s down, ~30 s OFF (seeded), UE1/UE2 DL via
r1/traffic/burst.py; UE3 constant 5 Mbps DL. Only cycles that end within collection_min are scheduled.
Isolation (before every collection): stop leftover UE traffic, wait until no live flow exists and the cell PRB has
stayed at idle (< IDLE_PRB_PCT) for >= IDLE_HOLD_S, log the wait; then restart the traffic generator (fresh iperf2
server). On every exit (complete or abort) all UE traffic is stopped.
Guard (every 10 s): backend_reloaded, pod_changed (gNB, UEs, RIC, traffic-gen), sim_speed_drift (per-cycle mean
speed vs cycle 1's, ±10 %, two consecutive cycles; speed_rules.py), trafficgen_memory (> 70 % of its limit),
ul_traffic_detected, channel_mapping_violated → abort;
meta.status = failed with the code. Peak traffic-gen memory, sim-speed 10 s means and per-minute host load
(scripts/host_sampler.py) are recorded. Start gate: sim speed not below −10 % of the (one-sided since 2026-11-05 17:15; earlier ±15 %, then ±10 %)
IDLE reference (r1/env/experiment.json idle_reference_sim_speed, measured with no traffic)
plus a ~3 min loaded probe (2 frozen ramp cycles; mean loaded speed within the regime band 1.47-1.80);
if either fails, wait 15 min and retry without limit (until R1_DEADLINE_EPOCH if set); every attempt is logged. Run-level regime: meta.regime (mean of the
per-cycle speeds vs env loaded_reference_sim_speed, ±10 %; outside = off-regime, kept, excluded from the paper).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import r1env as E  # noqa: E402  (sets sys.path for backend + traffic)
from r1env import k, pdu_ip, pod  # noqa: E402

import burst  # noqa: E402
import speed_rules  # noqa: E402
import run_guard  # noqa: E402
from services.dual_brain import channel_guard as g, gnb_truth, profiles, ric  # noqa: E402
from services.k8s.experiment_ctx import ExperimentRef, use_experiment  # noqa: E402

FROZEN = {'profile': 'ramp', 'peak': 28.0, 'up_s': 45.0}  # calibration/NOTES.md 2026-11-04: frozen for Exp-01/03/04 (UE3 7 Mbps)
TOTAL_PRB = 51
REALTIME_BUDGET = 61200  # DL PRB-slots per wall second at real time (51 PRB x 1200 DL slots)
TRF_MEM_LIMIT_FRAC = 0.70  # trafficgen_memory guard
COLUMNS = (['t_ms', 'wall_s', 'cell_prb_util'] + [f'prb_pct_ue{i}' for i in (1, 2, 3)]
           + [f'dl_mcs_ue{i}' for i in (1, 2, 3)] + [f'bler_ue{i}' for i in (1, 2, 3)]
           + [f'ul_snr_ue{i}' for i in (1, 2, 3)] + [f'thp_dl_ue{i}' for i in (1, 2, 3)]
           + ['rtt_ue3_ms', 'sim_speed', 'cycle', 'phase', 'offered_mbps_center'])


def frozen_schedule(duration_s: float, seed: int, peak: float = FROZEN['peak'], up_s: float = FROZEN['up_s'],
                    cycles: int | None = None) -> dict:
    """Frozen ramp schedule: `cycles` cycles, or as many complete cycles as fit in duration_s."""
    n = cycles or 1
    while cycles is None and burst.schedule(FROZEN['profile'], cycles=n + 1, seed=seed, peak=peak,
                                            up_s=up_s)['total_s'] <= duration_s:
        n += 1
    return burst.schedule(FROZEN['profile'], cycles=n, seed=seed, peak=peak, up_s=up_s)


def phase_at(sched: dict, t: float) -> tuple[int | None, str | None]:
    for i, c in enumerate(sched['cycles'], 1):
        for name in ('ramp_up', 'hold', 'ramp_down', 'off'):
            a, b = c['phases'][name]
            if a <= t < b:
                return i, name
    return None, None


def hold_last(series: list[tuple[float, float]], t: float):
    """Last value at or before t (series sorted by time), else None: causal sample-and-hold."""
    lo, hi = 0, len(series)
    while lo < hi:
        mid = (lo + hi) // 2
        if series[mid][0] <= t:
            lo = mid + 1
        else:
            hi = mid
    return series[lo - 1][1] if lo else None


assert hold_last([(1.0, 'a'), (2.0, 'b')], 1.5) == 'a' and hold_last([(1.0, 'a')], 0.5) is None


def merge(samples, ref_to_ue, mac, aura, recv, rtt, t0, ts_ms, sched):
    """Telemetry rows on the KPM grid. samples: xApp sample events; mac[ue][field] = [(t, v)]."""
    rows = []
    thp = {ue: {round(t - 0.5): v for t, v in s} for ue, s in recv.items()}  # second index → Mbps
    rtt_i, rtt_sorted = 0, sorted(rtt)
    for e in samples:
        t = e['t_us'] / 1e6
        prb = {ue: 0.0 for ue in ('ue1', 'ue2', 'ue3')}
        for ref, v in (e.get('ues') or {}).items():
            if ref in ref_to_ue:
                prb[ref_to_ue[ref]] = float(v)
        while rtt_i < len(rtt_sorted) and rtt_sorted[rtt_i][0] <= t - ts_ms / 1000:
            rtt_i += 1
        win = [v for tt, v in rtt_sorted[rtt_i:rtt_i + 64] if t - ts_ms / 1000 < tt <= t]
        cyc, ph = phase_at(sched, t - t0) if sched else (None, None)
        row = {'t_ms': round((t - t0) * 1000), 'wall_s': round(t, 3), 'cell_prb_util': round(float(e['cell_prb']), 2),
               'rtt_ue3_ms': round(sum(win) / len(win), 2) if win else '',
               'sim_speed': (lambda s: round(s, 3) if s else '')(hold_last(aura, t)),
               'cycle': cyc or '', 'phase': ph or '',
               'offered_mbps_center': round(burst.offered_mbps(sched, t - t0), 2) if sched else ''}
        for i, ue in enumerate(('ue1', 'ue2', 'ue3'), 1):
            row[f'prb_pct_ue{i}'] = round(prb[ue], 2)
            for f, col in (('mcs', 'dl_mcs'), ('bler', 'bler'), ('ul_snr', 'ul_snr')):
                v = hold_last(mac.get(ue, {}).get(f, []), t)
                row[f'{col}_ue{i}'] = v if v is not None else ''
            row[f'thp_dl_ue{i}'] = round(thp.get(ue, {}).get(math.floor(t), 0.0), 3)
        rows.append(row)
    return rows


def parse_mac(glog: str, rnti_to_ue: dict) -> dict:
    out = {ue: {'mcs': [], 'bler': [], 'ul_snr': []} for ue in rnti_to_ue.values()}
    for line in glog.splitlines():
        if (m := g._GNB_DL.search(line)) and m.group(1) in rnti_to_ue:
            t = gnb_truth._ts(line)
            out[rnti_to_ue[m.group(1)]]['bler'].append((t, float(m.group(2))))
            out[rnti_to_ue[m.group(1)]]['mcs'].append((t, int(m.group(3))))
        elif (m := g._GNB_UL.search(line)) and m.group(1) in rnti_to_ue:
            out[rnti_to_ue[m.group(1)]]['ul_snr'].append((gnb_truth._ts(line), float(m.group(2))))
    return out


GATE_RETRY_S = 900  # 2026-11-04 (Milad): wait 15 min after a failed idle check or loaded probe


def loaded_probe(seed: int) -> dict:
    """~3 min of the frozen traffic (2 ramp cycles at the frozen peak, UE3 at its rate): mean sim/wall speed over the
    ramp-up + hold phases (gNB [AURA-PRB] 1 s budgets). Traffic is stopped and the cell isolated afterwards."""
    trf_ip = pod('traffic-gen')[1]
    upods = {u: pod('virtual-ue', u)[0] for u in ('ue1', 'ue2', 'ue3')}
    for p in upods.values():
        E.route_to(p, trf_ip)
    bind = {u: pdu_ip(p) for u, p in upods.items()}
    sched = burst.schedule(FROZEN['profile'], cycles=2, seed=seed + 5000, peak=FROZEN['peak'], up_s=FROZEN['up_s'])
    hold_end = sched['cycles'][-1]['phases']['hold'][1]
    k('exec', upods['ue3'], '-c', 'ue', '--', *burst.remote_command({'cycles': [{'flows': [(0.0, E.ENV['ue3_rate_mbps'], hold_end + 5)]}]},
                                                                  server_ip=trf_ip, bind_ip=bind['ue3']))
    t0 = time.time()
    for u in ('ue1', 'ue2'):
        k('exec', upods[u], '-c', 'ue', '--', *burst.remote_command(sched, server_ip=trf_ip, bind_ip=bind[u]))
    time.sleep(hold_end + 3)
    rows = gnb_truth.parse(k('logs', pod('gnb-rfsim')[0], '-c', 'gnb', '--timestamps', f'--since={int(time.time() - t0) + 10}s'))[0]
    E.stop_traffic()
    win = [(t0 + c['phases']['ramp_up'][0], t0 + c['phases']['hold'][1]) for c in sched['cycles']]
    v = [r['budget'] / REALTIME_BUDGET for r in rows if any(a <= r['t'] < b for a, b in win)]
    E.isolate(lambda e: None)
    return {'mean_loaded_speed': round(sum(v) / len(v), 4) if v else None, 'n_s': len(v), 'seed': seed + 5000}


async def start_gate(log, seed: int, loaded_reference: float | None = None, tol: float | None = None):
    """Idle check (±START_TOL of the idle reference) + loaded probe (mean within the regime band); on failure wait
    GATE_RETRY_S and retry, without limit, until R1_DEADLINE_EPOCH (env) if set. Every attempt is logged."""
    ref = E.ENV['idle_reference_sim_speed']
    lref = loaded_reference or E.ENV['loaded_reference_sim_speed']  # Exp-03 (2026-11-06): its own band, see exp03.py
    tol = speed_rules.REGIME_TOL if tol is None else tol
    deadline = float(os.environ.get('R1_DEADLINE_EPOCH', 0) or 0)
    i = 0
    while True:
        if deadline and time.time() > deadline:
            raise SystemExit('start gate: deadline reached (R1_DEADLINE_EPOCH)')
        s = await g.sim_speed(10, REALTIME_BUDGET)  # cell idle here (after isolation, before traffic)
        probe = None
        if speed_rules.start_gate_ok(s, ref):
            probe = loaded_probe(seed)
            if speed_rules.regime_check([probe['mean_loaded_speed']], lref, tol)['in_regime']:
                log({'event': 'start_gate_passed', 'idle_sim_speed': s, 'probe': probe, 'attempt': i + 1})
                return s, i, probe
        log({'event': 'start_skipped', 'idle_sim_speed': s, 'idle_reference': ref, 'probe': probe,
             'loaded_reference': lref, 'loadavg': os.getloadavg(), 'retry_in_s': GATE_RETRY_S})
        i += 1
        await asyncio.sleep(GATE_RETRY_S)


async def run(a):
    out = os.path.join(a.out_root, a.run_id)
    os.makedirs(out, exist_ok=False)
    guard_log = open(os.path.join(out, 'guard.jsonl'), 'w')

    def log(ev):
        guard_log.write(json.dumps({'wall_s': round(time.time(), 3), **ev}) + '\n')
        guard_log.flush()
        print(ev, flush=True)

    meta = {'run_id': a.run_id, 'T_s_ms': a.ts, 'total_prb': TOTAL_PRB, 'collection_min': a.collection_min,
            'traffic': {'mode': a.traffic, 'seed': a.seed, **FROZEN, 'peak': a.peak, 'up_s': a.up_s, 'cycles_arg': a.cycles,
                        'ue3_rate_mbps': a.ue3_rate,
                        'ping_interval_s': E.PING_INTERVAL_S},
            'versions': json.load(open(os.path.join(E.R1, 'env', 'versions.json'))),
            'provenance': E.provenance(__file__, extra=[  # every file this collector executes
                os.path.join(E.R1, 'collector', 'speed_rules.py'), os.path.join(E.R1, 'scripts', 'run_guard.py'),
                os.path.join(E.R1, 'scripts', 'host_sampler.py')] + [
                os.path.join(E.BACKEND, 'services', 'dual_brain', m) for m in
                ('channel_guard.py', 'gnb_truth.py', 'ric.py', 'profiles.py')]), 'host_loadavg_start': os.getloadavg(), 'status': 'running'}
    meta['running_images'] = dict(  # what actually ran (versions.json records what was built)
        ln.split('=', 1) for ln in k('get', 'pods', '-l', f'harmony.telecom/experiment={E.SLUG}', '-o',
                                     'jsonpath={range .items[*]}{.metadata.name}={.spec.containers[0].image}{"\\n"}{end}'
                                     ).splitlines() if '=' in ln)
    backend0 = run_guard.backend_worker()
    ref = ExperimentRef(experiment_id=E.EXP, slug=E.SLUG)
    sampler = subprocess.Popen(['python3', '-I', os.path.join(E.R1, 'scripts', 'host_sampler.py'),
                                os.path.join(out, 'host_load.jsonl'), '--interval', '60', '--until-pid', str(os.getpid())])
    with use_experiment(ref):
        meta['isolation'] = E.isolate(log)
        trf = E.restart_trafficgen()
        log({'event': 'trafficgen_restarted', 'pod': trf[0]})
        if not a.no_start_gate:
            meta['sim_speed_start'], meta['skipped_starts'], meta['start_probe'] = await start_gate(log, a.seed)
            trf = E.restart_trafficgen()  # fresh traffic generator after the probe
            log({'event': 'trafficgen_restarted', 'pod': trf[0]})
        rows0 = await g.snapshot(window_s=20)
        if problem := g.check_mapping(rows0):
            raise SystemExit(f'mapping: {problem}')
        rnti_to_ue = {r['rnti']: r['ue'].lower() for r in rows0}
        ref_to_ue = {str(r['cu_ue_id']): r['ue'].lower() for r in rows0}
        meta['ues'] = [{'label': u['label'], 'key': u['key']} for u in E.UES]
        pods0 = await g.pods_state()
        speed0 = await g.sim_speed(10, REALTIME_BUDGET)
        duration = a.collection_min * 60
        sched = frozen_schedule(duration, a.seed, a.peak, a.up_s, a.cycles) if a.traffic == 'frozen' else None
        if a.cycles and sched:  # a fixed number of cycles sets the duration (+10 s tail)
            duration = sched['total_s'] + 10
        trf_name, trf_ip = pod('traffic-gen')
        upods = {u: pod('virtual-ue', u)[0] for u in ('ue1', 'ue2', 'ue3')}
        for p in upods.values():
            E.route_to(p, trf_ip)
        bind = {u: pdu_ip(p) for u, p in upods.items()}

        prof = profiles.load('rfsim', {'kpm': {'period_ms': a.ts}})
        await ric.stop_conflicting_kpm_monitor()
        offset = await ric.start(prof, mode='collect', reserve_pct=20)  # returns once KPM samples flow
        if a.traffic != 'none':
            ue3 = {'cycles': [{'flows': [(0.0, a.ue3_rate, duration + 15)]}]}
            k('exec', upods['ue3'], '-c', 'ue', '--', *burst.remote_command(ue3, server_ip=trf_ip, bind_ip=bind['ue3']))
        E.start_ping(upods['ue3'], trf_ip, duration + 15)
        t0 = time.time()
        if sched:
            for u in ('ue1', 'ue2'):
                r = k('exec', upods[u], '-c', 'ue', '--', *burst.remote_command(sched, server_ip=trf_ip, bind_ip=bind[u]))
                assert 'started' in r, r
        meta.update(t0_wall_s=t0, start_iso=time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(t0)),
                    schedule=sched and {'cycles': [c['phases'] for c in sched['cycles']], 'total_s': sched['total_s'],
                                        'params': sched['params']})
        log({'event': 'started', 't0': t0, 'cycles': len(sched['cycles']) if sched else 0})
        import atexit
        atexit.register(E.stop_traffic)  # also on exceptions / SystemExit (not on SIGKILL)

        events, cyc_means, abort, trf_peak = [], [], None, 0
        trf_limit = E.trafficgen_memory()[1]
        while time.time() - t0 < duration and not abort:
            await asyncio.sleep(10)
            evs, offset = await ric.read_events(prof, offset)
            events += evs
            try:
                run_guard.assert_same(backend0)
            except run_guard.BackendReloaded as exc:
                abort = ('backend_reloaded', str(exc))
                break
            if msg := g.pod_changes(pods0, await g.pods_state()):
                abort = (g.POD_CHANGED, msg)
                break
            cur, lim = E.trafficgen_memory()
            if cur:
                trf_peak = max(trf_peak, cur)
                if lim and cur > TRF_MEM_LIMIT_FRAC * lim:
                    abort = ('trafficgen_memory', f'traffic-gen memory {cur / 2**20:.0f} MiB > '
                                                  f'{TRF_MEM_LIMIT_FRAC:.0%} of {lim / 2**20:.0f} MiB')
                    break
            done = [c['phases'] for c in (sched or {}).get('cycles', []) if t0 + c['phases']['off'][1] <= time.time()]
            if len(done) > len(cyc_means):  # a cycle just ended: its mean speed from the gNB's 1 s budgets
                for ph in done[len(cyc_means):]:
                    a_, b_ = t0 + ph['ramp_up'][0], t0 + ph['off'][1]
                    rows_ = gnb_truth.parse(k('logs', pod('gnb-rfsim')[0], '-c', 'gnb', '--timestamps',
                                              f'--since={int(time.time() - a_) + 5}s'))[0]
                    cyc_means += speed_rules.cycle_means([(r['t'], r['budget'] / REALTIME_BUDGET) for r in rows_], [(a_, b_)])
                log({'event': 'cycle_speed', 'cycle': len(cyc_means), 'mean': cyc_means[-1] and round(cyc_means[-1], 3)})
                if reason := speed_rules.drift_check(cyc_means):
                    abort = (g.SIM_SPEED, reason)
                    break
            if int(time.time() - t0) % 60 < 10:  # once a minute: UL guard + identity mapping
                snap = await g.snapshot(window_s=30)
                if msg := g.check_ul(snap) or g.check_mapping(snap):
                    abort = (g.UL_TRAFFIC if 'UL goodput' in msg else g.VIOLATION, msg)
        await ric.stop()
        t_end = time.time()
        left = E.stop_traffic()  # on every exit: no flow may outlive its collection
        log({'event': 'traffic_stopped', 'still_live': left})
        evs, offset = await ric.read_events(prof, offset)
        events += evs
        if abort:
            meta['crash_logs'] = await g.save_crash_logs(os.path.join(out, 'crash'))
            log({'event': 'abort', 'code': abort[0], 'detail': abort[1]})
        glog = k('logs', pod('gnb-rfsim')[0], '-c', 'gnb', '--timestamps', f'--since={int(time.time() - t0) + 60}s')
        ping = k('exec', upods['ue3'], '-c', 'ue', '--', 'cat', E.PING_LOG)
        open(os.path.join(out, 'ue3_ping.log'), 'w').write(ping)  # raw 10 Hz ping (RTT tails recomputable)
        recv = {u: burst.receiver_mbps(k('exec', p, '-c', 'ue', '--', 'cat', burst.RECV_LOG)) for u, p in upods.items()}

    with open(os.path.join(out, 'events.jsonl'), 'w') as fh:
        fh.writelines(json.dumps(e) + '\n' for e in events)
    with open(os.path.join(out, 'gnb.log'), 'w') as fh:
        fh.write(glog)
    import re
    rtt = [(float(x), float(y)) for x, y in re.findall(r'\[(\d+\.\d+)\].*?time=([\d.]+) ms', ping)]
    aura_rows = gnb_truth.parse(glog)[0]
    aura = [(r['t'], r['budget'] / REALTIME_BUDGET) for r in aura_rows]
    samples = [e for e in events if e.get('type') == 'sample' and t0 <= e['t_us'] / 1e6 <= t_end]
    rows = merge(samples, ref_to_ue, parse_mac(glog, rnti_to_ue), aura, recv, rtt, t0, a.ts, sched)
    with open(os.path.join(out, 'telemetry.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    sampler.terminate()
    gaps = [b['t_ms'] - x['t_ms'] for x, b in zip(rows, rows[1:])]
    w10 = {}
    for t_, sp in aura:
        if t0 <= t_ <= t_end:
            w10.setdefault(int((t_ - t0) // 10), []).append(sp)
    m10 = [round(sum(v) / len(v), 3) for _, v in sorted(w10.items())]
    phase_minmax = {}
    for i_, v in sorted(w10.items()):
        _, ph = phase_at(sched, i_ * 10 + 5) if sched else (None, None)
        if ph:
            mv = sum(v) / len(v)
            lo, hi = phase_minmax.get(ph, (mv, mv))
            phase_minmax[ph] = (round(min(lo, mv), 3), round(max(hi, mv), 3))
    hl = [json.loads(ln) for ln in open(os.path.join(out, 'host_load.jsonl'))] if os.path.exists(os.path.join(out, 'host_load.jsonl')) else []
    cross = []
    for i, c in enumerate((sched or {}).get('cycles', []), 1):
        on = [r['cell_prb_util'] for r in rows if r['cycle'] == i and r['phase'] in ('ramp_up', 'hold', 'ramp_down')]
        cross.append(bool(on) and max(on) >= 80)
    missing = [u for u in ('ue1', 'ue2', 'ue3') if not recv.get(u)] + ([] if rtt else ['ue3 ping'])
    meta['gnb_log_coverage'] = speed_rules.gnb_log_coverage([r['t'] for r in aura_rows], t0, t_end)
    if meta['gnb_log_coverage'] < speed_rules.GNB_COVERAGE_MIN:  # no valid PRB ground truth (Exp-03 b3)
        missing.append(f"gnb log (coverage {meta['gnb_log_coverage']})")
    speeds = [s for _, s in aura if t0 <= _ <= t_end]
    meta.update(trafficgen_memory_peak_mib=round(trf_peak / 2**20, 1), trafficgen_memory_limit_mib=trf_limit and round(trf_limit / 2**20),
                sim_speed_10s={'idle_start': speed0, 'means': m10, 'min': min(m10, default=None), 'max': max(m10, default=None),
                               'per_phase': phase_minmax},
                sim_speed_cycle_means=[m and round(m, 3) for m in cyc_means],
                regime=speed_rules.regime_check(cyc_means, E.ENV['loaded_reference_sim_speed']),
                sim_speed_cycle_max_dev_pct=round(100 * max((abs(m / cyc_means[0] - 1) for m in cyc_means[1:] if m), default=0), 1)
                if cyc_means and cyc_means[0] else None,
                host_load_per_minute=[{'wall': h.get('wall_iso'), 'load1': h.get('loadavg_1m'),
                                       'top': [(x['comm'], x['user'], x['pcpu']) for x in (h.get('top') or [])[:4]]} for h in hl],
                stop_iso=time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(t_end)), duration_s=round(t_end - t0, 1),
                rows=len(rows), gaps_gt_2ts=sum(1 for d in gaps if d > 2 * a.ts), max_gap_ms=max(gaps, default=None),
                kpm_period_ms_median=sorted(gaps)[len(gaps) // 2] if gaps else None,
                crosses_80_per_cycle=cross, sim_speed_mean=round(sum(speeds) / len(speeds), 3) if speeds else None,
                host_loadavg_end=os.getloadavg(), missing=missing,
                status='failed' if abort or missing or not rows else 'complete',
                failure=abort and {'code': abort[0], 'detail': abort[1]})
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print(json.dumps({x: meta[x] for x in ('status', 'rows', 'gaps_gt_2ts', 'max_gap_ms', 'kpm_period_ms_median',
                                           'crosses_80_per_cycle', 'sim_speed_mean', 'missing', 'failure', 'isolation',
                                           'trafficgen_memory_peak_mib', 'trafficgen_memory_limit_mib')}, indent=1))
    print('sim speed: idle start', meta['sim_speed_10s']['idle_start'], '| cycle means', meta['sim_speed_cycle_means'],
          '| max cycle dev %', meta['sim_speed_cycle_max_dev_pct'], '| 10 s per phase (min, max)', meta['sim_speed_10s']['per_phase'])
    print('wrote', *sorted(os.listdir(out)))
    return 0 if meta['status'] == 'complete' else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--ts', type=int, default=250, choices=(100, 250, 500))
    ap.add_argument('--collection-min', type=float, default=20)
    ap.add_argument('--seed', type=int, default=6)
    ap.add_argument('--traffic', choices=('frozen', 'none'), default='frozen')
    ap.add_argument('--out-root', default=os.path.join(E.R1, 'data', 'raw'))
    ap.add_argument('--no-start-gate', action='store_true')
    ap.add_argument('--peak', type=float, default=FROZEN['peak'], help='Mbps per center UE (unofficial profile checks)')
    ap.add_argument('--up-s', type=float, default=FROZEN['up_s'])
    ap.add_argument('--cycles', type=int, help='fixed number of cycles (overrides --collection-min)')
    ap.add_argument('--ue3-rate', type=float, default=E.ENV['ue3_rate_mbps'], help='UE3 constant DL Mbps')
    sys.exit(asyncio.run(run(ap.parse_args())))


if __name__ == '__main__':
    main()
