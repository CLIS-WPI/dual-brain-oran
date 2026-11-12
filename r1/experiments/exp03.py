"""Exp-03 outcome runs (plan: experiments/exp03_plan.md). One run = one arm, 13 frozen cycles (seed 6, same schedule in
every run), ground-truth measurement only.

usage: exp03.py run   --arm a|b|c|d --arms-dir <dir> --run-id <id> --out-root <dir> [--cycles 13]
       exp03.py dry   --arms-dir <dir> --batch <name> --deadline YYYY-mm-ddTHH:MM   (unofficial: (a), (b), (c), 2 cycles
            each, then exp03_analyze + checks: config byte-exact on the RIC, cap reserves and lifts within OFF in
            every (b)/(c) cycle, gate + regime recorded, all outputs and metrics produced → <batch>-dry/dry_check.json)
       exp03.py batch --arms-dir <dir> --batch <name> --deadline YYYY-mm-ddTHH:MM
Run: isolation + fresh traffic generator → start gate (kpm_collector.start_gate: idle ±10 % + ~3 min loaded probe,
retry every 15 min until R1_DEADLINE_EPOCH; Exp-03 regime band = arm (c) training speed ±5 %, exp03_band) → fresh traffic generator → arm setup → UE3 7 Mbps + 10 Hz ping →
UE1/UE2 frozen ramp → guard every 10 s (backend reload, pod change, traffic-gen memory, per-cycle sim-speed drift,
UL/mapping once a minute; same codes as the collector) → stop → raw logs → run_meta.json (status, regime).
Arms: (a) no xApp and no other KPM subscriber; (b)/(c)/(d) zto/deploy.py with the arm's files from <arms-dir>
(byte-for-byte, sha256-checked on the RIC; exp03_prep.py made them). The configs' protected_ue_ref must equal UE3's
current CU-UE-ID, else the run stops (status protected_ue_mismatch) — the config is never edited.
Outputs per run (results/exp03/<batch>/<run_id>/): run_meta.json, control_events.jsonl, guard.jsonl,
raw/{gnb_aura.log, ue3_ping.log, recv.json, xapp_events.jsonl, host_load.jsonl} (raw/ stays local, DataPort only).
Batch: order a,c,b × 3 then d × 3; every run is its own process (same gate before every run). off_regime, sim_speed_drift
or pod_changed (e.g. a UE3 LDPC crash) → kept, marked, repeated (max 2 repeats per slot). Arm (d) is skipped when its
model failed model_parity (manifest d_model.usable). At the deadline the current run is terminated
(stopped_deadline) and the remaining slots are marked not_started. At the end the stack is left idle.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time

R1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R1, 'scripts'), os.path.join(R1, 'zto'), os.path.join(R1, 'collector'), R1]
import r1env as E  # noqa: E402

START_COMMIT = E.git('rev-parse', 'HEAD')
SEED, CYCLES = 6, 13
ORDER = ['a', 'c', 'b'] * 3 + ['d'] * 3  # 2026-11-05 (Milad): main pair first; (a) vs (c) most complete if time runs out
SLOTS = [f'{x}{ORDER[:i + 1].count(x)}' for i, x in enumerate(ORDER)]  # a1, c1, b1, a2, …, d1, d2, d3
DRY_ARMS, DRY_CYCLES = ('a', 'b', 'c'), 2
REPEAT_CODES = {'sim_speed_drift', 'pod_changed'}
EXP03_REGIME_TOL = 0.05  # Milad 2026-11-06: Exp-03 band = arm (c) training speed ±5 % (gate probe + run-level check)


def exp03_band(arms_dir: str) -> tuple[float, float]:
    """(loaded reference, tol) for Exp-03: the mean loaded speed of the Exp-04 run whose model/lease are arm (c)
    (s1: 1.7905 → 1.70-1.88). The exp01 band (1.635 ±10 %) stays for exp01/Exp-04."""
    man = json.load(open(os.path.join(arms_dir, 'manifest.json')))
    m = json.load(open(os.path.join(E.RESULTS, 'exp04', man['source_exp04_run'], 'run_meta.json')))
    return m['regime']['mean_cycle_speed'], EXP03_REGIME_TOL
sha = lambda b: hashlib.sha256(b).hexdigest()


async def run(a):
    import burst
    import deploy
    import kpm_collector as K
    import run_guard
    import speed_rules
    from services.dual_brain import channel_guard as g, gnb_truth, profiles, ric
    from services.k8s.experiment_ctx import ExperimentRef, use_experiment

    out = os.path.join(a.out_root, a.run_id)
    raw = os.path.join(out, 'raw')
    os.makedirs(raw, exist_ok=False)
    glog_fh = open(os.path.join(out, 'guard.jsonl'), 'w')

    def log(ev):
        glog_fh.write(json.dumps({'wall_s': round(time.time(), 3), **ev}) + '\n')
        glog_fh.flush()
        print(ev, flush=True)

    files = {} if a.arm == 'a' else {f: open(os.path.join(a.arms_dir, a.arm, f), 'rb').read()
                                      for f in ('config.json', 'model.onnx', 'preprocess.json')}
    meta = {'run_id': a.run_id, 'arm': a.arm, 'git_commit': START_COMMIT, 'seed': SEED, 'cycles': a.cycles,
            'ue3_rate_mbps': E.ENV['ue3_rate_mbps'], 'hashes': {f: sha(v) for f, v in files.items()},
            'arms_manifest_sha256': sha(open(os.path.join(a.arms_dir, 'manifest.json'), 'rb').read()),
            'provenance': E.provenance(__file__, extra=[os.path.join(R1, p) for p in (
                'collector/kpm_collector.py', 'collector/speed_rules.py', 'zto/deploy.py', 'scripts/run_guard.py')]),
            'host_loadavg_start': os.getloadavg(), 'status': 'running'}
    save = lambda: json.dump(meta, open(os.path.join(out, 'run_meta.json'), 'w'), indent=1, default=str)
    save()
    import atexit
    atexit.register(E.stop_traffic)
    backend0 = run_guard.backend_worker()
    sampler = subprocess.Popen(['python3', '-I', os.path.join(R1, 'scripts', 'host_sampler.py'),
                                os.path.join(raw, 'host_load.jsonl'), '--interval', '60', '--until-pid', str(os.getpid())])
    with use_experiment(ExperimentRef(experiment_id=E.EXP, slug=E.SLUG)):
        await ric.stop()  # nothing of a previous arm may keep running through isolation / the gate
        meta['isolation'] = E.isolate(log)
        log({'event': 'trafficgen_restarted', 'pod': E.restart_trafficgen()[0]})
        lref, tol = exp03_band(a.arms_dir)
        meta['regime_band'] = {'loaded_reference': lref, 'tol': tol, 'band': [round(lref * (1 - tol), 4), round(lref * (1 + tol), 4)],
                               'source': 'arm (c) training run mean loaded speed (Milad 2026-11-06)'}
        meta['sim_speed_start'], meta['skipped_starts'], meta['start_probe'] = await K.start_gate(log, SEED, lref, tol)
        trf_name, trf_ip = E.restart_trafficgen()
        log({'event': 'trafficgen_restarted', 'pod': trf_name})
        rows0 = await g.snapshot(window_s=20)
        if problem := g.check_mapping(rows0):
            raise SystemExit(f'mapping: {problem}')
        rnti_to_ue = {r['rnti']: r['ue'].lower() for r in rows0}
        prot = next(r for r in rows0 if r['imsi'] == E.IMSI['ue3'])
        meta['ue3_cu_ue_id'] = int(prot['cu_ue_id'])
        prof = profiles.load('rfsim')
        await ric.stop_conflicting_kpm_monitor()
        offset = None
        if files:
            ref = json.loads(files['config.json'])['protected_ue_ref']
            if ref != meta['ue3_cu_ue_id']:
                meta['status'] = 'protected_ue_mismatch'
                meta['detail'] = f'config protected_ue_ref {ref} != UE3 CU-UE-ID {meta["ue3_cu_ue_id"]}'
                return save()
            offset, meta['config_sha256_on_ric'] = await deploy.deploy(
                prof, files['config.json'], {f: v for f, v in files.items() if f != 'config.json'})
            log({'event': 'xapp_deployed', 'config_sha256': meta['config_sha256_on_ric']})
        pods0 = await g.pods_state()
        upods = E.ue_pods()
        for p in upods.values():
            E.route_to(p, trf_ip)
        bind = {u: E.pdu_ip(p) for u, p in upods.items()}
        sched = burst.schedule('ramp', cycles=a.cycles, seed=SEED, peak=K.FROZEN['peak'], up_s=K.FROZEN['up_s'])
        span = sched['total_s'] + 15
        E.k('exec', upods['ue3'], '-c', 'ue', '--', *burst.remote_command(
            {'cycles': [{'flows': [(0.0, E.ENV['ue3_rate_mbps'], span)]}]}, server_ip=trf_ip, bind_ip=bind['ue3']))
        E.start_ping(upods['ue3'], trf_ip, span)
        await asyncio.sleep(10)  # UE3 background settles before the first cycle (as oracle_check)
        t0 = time.time()
        for u in ('ue1', 'ue2'):
            r = E.k('exec', upods[u], '-c', 'ue', '--', *burst.remote_command(sched, server_ip=trf_ip, bind_ip=bind[u]))
            assert 'started' in r, r
        meta.update(t0_wall_s=t0, start_iso=time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(t0)),
                    schedule={'cycles': [c['phases'] for c in sched['cycles']], 'total_s': sched['total_s'],
                              'params': sched['params']})
        save()
        log({'event': 'started', 't0': t0, 'cycles': a.cycles})
        events, cyc_means, abort, trf_peak = [], [], None, 0
        while time.time() < t0 + sched['total_s'] + 3 and not abort:
            await asyncio.sleep(10)
            if offset is not None:
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
                if lim and cur > K.TRF_MEM_LIMIT_FRAC * lim:
                    abort = ('trafficgen_memory', f'{cur / 2**20:.0f} MiB > {K.TRF_MEM_LIMIT_FRAC:.0%} of {lim / 2**20:.0f} MiB')
                    break
            done = [c['phases'] for c in sched['cycles'] if t0 + c['phases']['off'][1] <= time.time()]
            if len(done) > len(cyc_means):
                for ph in done[len(cyc_means):]:
                    a_, b_ = t0 + ph['ramp_up'][0], t0 + ph['off'][1]
                    rows_ = gnb_truth.parse(E.k('logs', E.pod('gnb-rfsim')[0], '-c', 'gnb', '--timestamps',
                                                f'--since={int(time.time() - a_) + 5}s'))[0]
                    cyc_means += speed_rules.cycle_means([(r['t'], r['budget'] / K.REALTIME_BUDGET) for r in rows_], [(a_, b_)])
                log({'event': 'cycle_speed', 'cycle': len(cyc_means), 'mean': cyc_means[-1] and round(cyc_means[-1], 3)})
                if reason := speed_rules.drift_check(cyc_means):
                    abort = (g.SIM_SPEED, reason)
                    break
            if int(time.time() - t0) % 60 < 10:
                snap = await g.snapshot(window_s=30)
                if msg := g.check_ul(snap) or g.check_mapping(snap):
                    abort = (g.UL_TRAFFIC if 'UL goodput' in msg else g.VIOLATION, msg)
        if offset is not None:
            await ric.stop()
            events += (await ric.read_events(prof, offset))[0]
        t_end = time.time()
        log({'event': 'traffic_stopped', 'still_live': E.stop_traffic()})
        if abort:
            meta['crash_logs'] = await g.save_crash_logs(os.path.join(raw, 'crash'))
            log({'event': 'abort', 'code': abort[0], 'detail': abort[1]})
        glog = E.k('logs', E.pod('gnb-rfsim')[0], '-c', 'gnb', '--timestamps', f'--since={int(time.time() - t0) + 60}s')
        ping = E.k('exec', upods['ue3'], '-c', 'ue', '--', 'cat', E.PING_LOG)
        recv = {u: burst.receiver_mbps(E.k('exec', p, '-c', 'ue', '--', 'cat', burst.RECV_LOG)) for u, p in upods.items()}
        meta['pod_changes_end'] = g.pod_changes(pods0, await g.pods_state())
    sampler.terminate()
    open(os.path.join(raw, 'gnb_aura.log'), 'w').write('\n'.join(ln for ln in glog.splitlines() if '[AURA-' in ln or 'CU-UE-ID' in ln))
    open(os.path.join(raw, 'ue3_ping.log'), 'w').write(ping)
    json.dump(recv, open(os.path.join(raw, 'recv.json'), 'w'))
    with open(os.path.join(raw, 'xapp_events.jsonl'), 'w') as fh:
        fh.writelines(json.dumps(e) + '\n' for e in events)
    with open(os.path.join(out, 'control_events.jsonl'), 'w') as fh:
        fh.writelines(json.dumps(e) + '\n' for e in events if e.get('type') != 'sample')
    rtt = re.findall(r'\[(\d+\.\d+)\].*?time=([\d.]+) ms', ping)
    missing = [u for u in ('ue1', 'ue2', 'ue3') if not recv.get(u)] + ([] if rtt else ['ue3 ping'])
    meta['gnb_log_coverage'] = speed_rules.gnb_log_coverage([r["t"] for r in gnb_truth.parse(glog)[0]], t0, t_end)
    if meta['gnb_log_coverage'] < speed_rules.GNB_COVERAGE_MIN:  # no valid PRB ground truth (Exp-03 b3)
        missing.append(f"gnb log (coverage {meta['gnb_log_coverage']})")
    meta.update(t_end_wall_s=t_end, duration_s=round(t_end - t0, 1), rnti_to_ue=rnti_to_ue,
                trafficgen_memory_peak_mib=round(trf_peak / 2**20, 1), sim_speed_cycle_means=[m and round(m, 3) for m in cyc_means],
                regime=speed_rules.regime_check(cyc_means, lref, tol),
                control_actions={x: sum(1 for e in events if e.get('type') == 'control' and e.get('action') == x)
                                 for x in ('reserve', 'extend', 'release')},
                host_loadavg_end=os.getloadavg(), missing=missing, failure=abort and {'code': abort[0], 'detail': abort[1]})
    meta['status'] = ('failed' if abort or missing else
                      'complete' if len(cyc_means) == a.cycles and meta['regime'].get('in_regime') else
                      'off_regime' if len(cyc_means) == a.cycles else 'failed')
    save()
    print(json.dumps({k: meta.get(k) for k in ('status', 'arm', 'regime', 'control_actions', 'failure', 'missing')}, default=str))


def _write_atomic(path: str, data: bytes) -> None:
    tmp = path + '.tmp'
    open(tmp, 'wb').write(data)
    os.replace(tmp, path)


def refresh_protected_ref(arms_dir: str, root: str) -> dict | None:
    """Milad 2026-11-06: if UE3 re-attached with a new CU-UE-ID (LDPC crash), re-derive the arm configs before the next
    run with the prep procedure (exp03_prep.with_ref: ONLY protected_ue_ref changes; model files, lease, policy
    byte-identical), record hashes, diff and crash time in the manifest, keep the crash log (root/raw/, local)."""
    from services.dual_brain import channel_guard as g
    from services.k8s.experiment_ctx import ExperimentRef, use_experiment

    async def live():
        with use_experiment(ExperimentRef(experiment_id=E.EXP, slug=E.SLUG)):
            rows = await g.snapshot(window_s=20)
            if problem := g.check_mapping(rows):
                raise RuntimeError(f'mapping: {problem}')
            return int(next(r for r in rows if r['imsi'] == E.IMSI['ue3'])['cu_ue_id'])
    ref = asyncio.run(live())
    mp = os.path.join(arms_dir, 'manifest.json')
    man = json.load(open(mp))
    cur = json.loads(open(os.path.join(arms_dir, 'c', 'config.json'), 'rb').read())['protected_ue_ref']
    if ref == cur:
        return None
    import exp03_prep as P
    pod = E.pod('virtual-ue', 'ue3')[0]
    st = json.loads(E.k('get', 'pod', pod, '-o', 'json'))['status']['containerStatuses'][0]
    fin = ((st.get('lastState') or {}).get('terminated') or {})
    os.makedirs(os.path.join(root, 'raw'), exist_ok=True)
    logf = os.path.join(root, 'raw', f"ue3_previous_{(fin.get('finishedAt') or 'unknown').replace(':', '')}.log")
    if not os.path.exists(logf):
        open(logf, 'w').write(E.k('logs', pod, '-c', 'ue', '--previous', '--timestamps'))
    rec = {'t': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'protected_ue_ref': [cur, ref], 'diff': {'protected_ue_ref': [cur, ref]},
           'ue3_crash_finished_at': fin.get('finishedAt'), 'ue3_exit_code': fin.get('exitCode'),
           'ue3_restart_count': st.get('restartCount'), 'crash_log': os.path.relpath(logf, R1), 'config_sha256': {}}
    for x in ('b', 'c', 'd'):
        f = os.path.join(arms_dir, x, 'config.json')
        old = open(f, 'rb').read()
        new = P.with_ref(old, ref)  # asserts: only protected_ue_ref differs
        for m in ('model.onnx', 'preprocess.json'):
            assert sha(open(os.path.join(arms_dir, x, m), 'rb').read()) == man['arms'][x][m], f'{x}/{m} changed'
        _write_atomic(f, new)
        rec['config_sha256'][x] = {'before': sha(old), 'after': sha(new)}
        man['arms'][x]['config.json'] = sha(new)
    man.setdefault('protected_ue_ref_refreshes', []).append(rec)
    _write_atomic(mp, (json.dumps(man, indent=1, default=str) + '\n').encode())
    return rec


def batch(a, order=None, slots=None):
    order, slots = order or ORDER, slots or SLOTS
    deadline = dt.datetime.fromisoformat(a.deadline).timestamp()
    root = os.path.join(E.RESULTS, 'exp03', a.batch)
    os.makedirs(root, exist_ok=True)
    man = json.load(open(os.path.join(a.arms_dir, 'manifest.json')))
    blog = os.path.join(root, 'batch.log.jsonl')

    def log(**x):
        x['t'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        print(json.dumps(x), flush=True)
        open(blog, 'a').write(json.dumps(x) + '\n')

    log(event='batch_start', deadline=a.deadline, start_commit=START_COMMIT, arms_manifest=man['arms'])
    out, verdict = [], None
    for name, arm in zip(slots, order):
        if arm == 'd' and not man['d_model']['usable']:
            out.append({'slot': name, 'arm': arm, 'status': 'skipped_d_model_parity_failed'})
            continue
        for kk in range(3):
            rid = f'{a.batch}-{name}' + (f'-r{kk}' if kk else '')
            if time.time() > deadline:
                out.append({'slot': name, 'arm': arm, 'run_id': rid, 'status': 'not_started'})
                break
            try:  # UE3 re-attached with a new CU-UE-ID → re-derive the configs before this run
                if rr := refresh_protected_ref(a.arms_dir, root):
                    log(event='protected_ue_ref_refreshed', **rr)
            except Exception as exc:  # outside the rules: stop this slot safely, keep going
                log(event='protected_ue_ref_refresh_failed', error=f'{type(exc).__name__}: {exc}')
            log(event='run_start', slot=name, arm=arm, run_id=rid)
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), 'run', '--arm', arm, '--arms-dir', a.arms_dir,
                                  '--run-id', rid, '--out-root', root, '--cycles', str(a.cycles)], cwd=R1, start_new_session=True,
                                 env={**os.environ, 'R1_DEADLINE_EPOCH': str(deadline)},
                                 stdout=open(os.path.join(root, f'{rid}.stdout.txt'), 'w'), stderr=subprocess.STDOUT)
            while p.poll() is None:
                if time.time() > deadline:
                    os.killpg(p.pid, signal.SIGTERM)
                    p.wait(60)
                    E.stop_traffic()
                    break
                time.sleep(10)
            mp = os.path.join(root, rid, 'run_meta.json')
            m = json.load(open(mp)) if os.path.exists(mp) else {}
            st = 'stopped_deadline' if time.time() > deadline and m.get('status') in (None, 'running') else m.get('status') or 'no_run_meta'
            rec = {'slot': name, 'arm': arm, 'run_id': rid, 'status': st, 'regime': m.get('regime'),
                   'failure': m.get('failure'), 'control_actions': m.get('control_actions'), 'returncode': p.returncode}
            out.append(rec)
            log(event='run_end', **rec)
            if st in ('off_regime',) or (st == 'failed' and (m.get('failure') or {}).get('code') in REPEAT_CODES):
                continue
            break
        if out and out[-1]['status'] in ('no_run_meta', 'running'):  # protected_ue_mismatch: slot stopped, refresh retried next slot
            verdict = 'stopped_' + out[-1]['status']
            break
        if time.time() > deadline:
            verdict = 'deadline'
    done = {r['slot'] for r in out}
    for name, arm in zip(slots, order):  # slots never reached
        if name not in done:
            out.append({'slot': name, 'arm': arm, 'status': 'not_started'})
    E.stop_traffic()

    async def idle():
        from services.dual_brain import ric
        from services.k8s.experiment_ctx import ExperimentRef, use_experiment
        with use_experiment(ExperimentRef(experiment_id=E.EXP, slug=E.SLUG)):
            await ric.stop()
    try:
        asyncio.run(idle())
    except Exception as exc:
        log(event='idle_xapp_stop_failed', error=str(exc))
    json.dump({'batch': a.batch, 'verdict': verdict or 'complete', 'deadline': a.deadline, 'start_commit': START_COMMIT,
               'arms_manifest': man['arms'], 'runs': out, 'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z')},
              open(os.path.join(root, 'batch.json'), 'w'), indent=1, default=str)
    log(event='batch_end', verdict=verdict or 'complete', n_runs=len(out))
    return root, out


def lift_check(run_dir: str) -> list[dict]:
    """Per cycle of a (b)/(c)/(d) run: where the cap lifted. OK (Milad 2026-11-06, same reading as Exp-04 s6) when the
    first release after the cycle's reserve falls in OFF, or in ramp-down provided it came at a non-hot sample after
    the lease expired (release − reserve >= lease_s and the xApp's last sample before the release has pred 0). A lift
    in ramp-up or hold, or no lift before the next ramp-up, fails."""
    m = json.load(open(os.path.join(run_dir, 'run_meta.json')))
    t0, cyc = m['t0_wall_s'], m['schedule']['cycles']
    evs = [json.loads(ln) for ln in open(os.path.join(run_dir, 'raw', 'xapp_events.jsonl'))]
    lease = next((e['lease_s'] for e in evs if e.get('type') == 'start'), None)
    ctrl = [(e['t_us'] / 1e6, e['action']) for e in evs if e.get('type') == 'control' and e.get('action') in ('reserve', 'release')]
    samples = [(e['t_us'] / 1e6, e.get('pred')) for e in evs if e.get('type') == 'sample' and 'pred' in e]
    out = []
    for i, c in enumerate(cyc):
        ru0, rd, off = t0 + c['ramp_up'][0], [t0 + x for x in c['ramp_down']], [t0 + x for x in c['off']]
        nxt = t0 + cyc[i + 1]['ramp_up'][0] if i + 1 < len(cyc) else off[1]
        res = next((t for t, a in ctrl if a == 'reserve' and ru0 - 10 <= t < off[0]), None)
        rel = next((t for t, a in ctrl if a == 'release' and res and t > res), None)
        pred = next((p for t, p in reversed(samples) if rel and t <= rel), None)
        where = None if rel is None else 'off' if off[0] <= rel < nxt else 'ramp_down' if rd[0] <= rel < rd[1] else 'on' if rel < rd[0] else 'late'
        ok = where == 'off' or (where == 'ramp_down' and lease is not None and rel - res >= lease - 0.5 and pred == 0)
        out.append({'cycle': i + 1, 'reserve_s_after_ramp_start': res and round(res - ru0, 1), 'release_where': where,
                    'release_s_after_off_start': rel and round(rel - off[0], 1), 'held_s': rel and res and round(rel - res, 1),
                    'lease_s': lease, 'pred_at_release': pred, 'ok': bool(res and ok)})
    return out


def dry_check(root: str, out: list[dict]) -> dict:
    an = subprocess.run(['python3', os.path.join(R1, 'experiments', 'exp03_analyze.py'), root], capture_output=True, text=True)
    chk = {'runs': {r['arm']: r['status'] for r in out}, 'analyze_rc': an.returncode, 'analyze_err': an.stderr[-800:] or None}
    cyc = list(csv.DictReader(open(os.path.join(root, 'exp03_cycles.csv')))) if an.returncode == 0 else []
    for r in out:
        if not r.get('run_id') or not os.path.exists(os.path.join(root, r['run_id'], 'run_meta.json')):
            continue
        m = json.load(open(os.path.join(root, r['run_id'], 'run_meta.json')))
        c = {'status': m.get('status'), 'gate_recorded': m.get('sim_speed_start') is not None and m.get('start_probe') is not None,
             'regime_recorded': bool(m.get('regime')), 'regime': m.get('regime')}
        if r['arm'] != 'a':
            c['config_byte_exact_on_ric'] = m.get('config_sha256_on_ric') == m['hashes']['config.json']
            c['cycles'] = lift_check(os.path.join(root, r['run_id']))
            c['cap_reserves_and_lifts_ok'] = bool(c['cycles']) and all(x['ok'] for x in c['cycles'])
        else:
            c['no_control_events'] = not any((m.get('control_actions') or {}).values())
        chk[r['arm']] = c
    chk['outputs'] = {f: os.path.exists(os.path.join(root, f)) for f in
                      ('exp03_cycles.csv', 'exp03_timeseries.csv', 'exp03_summary.json', 'fig_outcome.pdf')}
    ok = (an.returncode == 0 and all(chk['outputs'].values()) and all(chk.get(x, {}).get('status') == 'complete' for x in DRY_ARMS)
          and all(chk[x]['gate_recorded'] and chk[x]['regime_recorded'] for x in DRY_ARMS) and chk['a']['no_control_events']
          and all(chk[x]['config_byte_exact_on_ric'] and chk[x]['cap_reserves_and_lifts_ok'] for x in ('b', 'c')))
    chk['result'] = 'PASS' if ok else 'FAIL'
    chk['lift_rule'] = 'OFF, or ramp-down at a non-hot sample after the lease expired (Milad 2026-11-06)'
    json.dump(chk, open(os.path.join(root, 'dry_check.json'), 'w'), indent=1, default=str)
    print(json.dumps({'dry_check': chk['result'], 'runs': chk['runs']}))
    return chk


def dry(a):
    root, out = batch(a, order=list(DRY_ARMS), slots=[f'dry-{x}' for x in DRY_ARMS])
    return dry_check(root, out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run')
    r.add_argument('--arm', required=True, choices='abcd')
    r.add_argument('--arms-dir', required=True)
    r.add_argument('--run-id', required=True)
    r.add_argument('--out-root', required=True)
    r.add_argument('--cycles', type=int, default=CYCLES)
    k = sp.add_parser('drycheck', help='re-run the dry check on an existing dry batch (no runs)')
    k.add_argument('--batch', required=True)
    for cmd in ('batch', 'dry'):
        b = sp.add_parser(cmd)
        b.add_argument('--arms-dir', required=True)
        b.add_argument('--batch', required=True)
        b.add_argument('--deadline', required=True)
    a = ap.parse_args()
    if a.cmd == 'drycheck':
        root = os.path.join(E.RESULTS, 'exp03', a.batch)
        return dry_check(root, json.load(open(os.path.join(root, 'batch.json')))['runs'])
    if a.cmd == 'dry':
        a.batch, a.cycles = a.batch + '-dry', DRY_CYCLES
    elif a.cmd == 'batch':
        a.cycles = CYCLES
    a.arms_dir = os.path.abspath(a.arms_dir)
    if a.cmd == 'run':
        asyncio.run(run(a))
    elif a.cmd == 'dry':
        dry(a)
    else:
        batch(a)


assert SLOTS == ['a1', 'c1', 'b1', 'a2', 'c2', 'b2', 'a3', 'c3', 'b3', 'd1', 'd2', 'd3'], SLOTS

if __name__ == '__main__':
    main()
