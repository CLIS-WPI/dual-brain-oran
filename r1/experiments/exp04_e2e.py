"""Exp-04: end-to-end provisioning, intent → running xApp, one run per call (timing marks, no log parsing).

usage: exp04_e2e.py --run-id <id> --collection-min 20|10 --seed <traffic seed> [--intent I1] [--mode full]
                    [--verify-cycles 3] [--dry-run]
Residency "pair" (llama3.1:8b orchestrator + llama3.3:70b judge, both resident and warm) on the Ollama GPU, guarded
before and during the run (gpu_abort.json). Steps and marks (r1/common/timing.py, scripts/timing_report.py):
  intent_received → agent (full mode, frozen prompt) → spec_ready → gate (extraction read-back; gate_* marks)
    HOLD → the run stops: status "held" (an operator decision; counted as a human touchpoint)
  collection (collector/kpm_collector.py, frozen profile, T_s = spec.sampling_ms) → collection_start/_end
  dataset/build.py (spec window/horizon/label; split by cycles, first ~9/13 train) → train_post → NeuralSmith →
    model_received
  lease (zto/lease.py: p90 sustained-episode duration in the training rows + H, clipped [10, 120] s) → lease_computed
  render_start → zto/render.py (spec, protected UE3 from the RAN mapping, policy ml, lease_s) →
    template_rendered → render_end   (config.json saved in the run dir; these bytes are deployed unchanged)
  gate check model_parity (zto/gate.py → zto/parity.py; FAIL stops the run before registration)
  register_start → zto/deploy.py (config.json + model.onnx + preprocess.json written to the RIC, sha256-checked,
    xApp started, KPM flowing) → register_end
  verification traffic (frozen profile, --verify-cycles cycles, UE3 7 Mbps + 10 Hz ping): xApp reserve/release
    events vs the schedule; first reserve → first_e2_control_sent (mark placed at the xApp's event time).
    Check: the cap lifts within OFF in every cycle (cap_lifts_within_off_all).
Outputs in results/exp04/<run_id>/: timing.jsonl, agent_calls.jsonl, spec.json, gate.json, config.json, model/
(report.json, model.onnx, preprocess.json), dataset/ (telemetry split kept for the same-dataset H 7.5 s model),
raw/ (collector), verify_events.jsonl, ue3_ping.log, run_meta.json (human_touchpoints, hashes, verify, status).
"""
from __future__ import annotations

import argparse
import asyncio
import atexit
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

R1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R1, 'scripts'), os.path.join(R1, 'zto'), os.path.join(R1, 'experiments'), R1]
import r1env as E  # noqa: E402

START_COMMIT = E.git('rev-parse', 'HEAD')  # the commit the run started from (no commits during official runs)
import agent  # noqa: E402
import burst  # noqa: E402
import deploy  # noqa: E402
import gate  # noqa: E402
import lease  # noqa: E402
import ollama_env as O  # noqa: E402
import render  # noqa: E402
from common import timing  # noqa: E402
from exp01_dataset import SPEC as NS_SPEC  # noqa: E402  same NeuralSmith settings as Exp-01
from exp08_gate import JUDGE, ORCH, pair_residency  # noqa: E402
from services.dual_brain import channel_guard as g, profiles, ric  # noqa: E402
from services.k8s.experiment_ctx import ExperimentRef, use_experiment  # noqa: E402

PY, MLV, NS_PORT = sys.executable, os.path.join(R1, '.venv-mlengine', 'bin', 'python'), 8097
sha = lambda b: hashlib.sha256(b).hexdigest()


def sh(*args, **kw):
    print('$', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), text=True, capture_output=True, **kw)


def train_neuralsmith(d, spec, out):
    """NeuralSmith server on its own port for this run; returns the report (model files in out/)."""
    p = subprocess.Popen([MLV, os.path.join(R1, 'mlengine', 'server.py'), '--backend', 'neuralsmith', '--port', str(NS_PORT)],
                         env={**os.environ, 'MLENGINE_RUNS': os.path.join(out, 'mlengine-runs')},
                         stdout=open(os.path.join(out, 'mlengine.log'), 'w'), stderr=subprocess.STDOUT)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{NS_PORT}/health', timeout=2)
                break
            except OSError:
                time.sleep(1)
        s = dict(NS_SPEC, latency_budget_ms=spec['model']['latency_budget_ms'], max_size_kb=spec['model']['max_size_kb'])
        json.dump(s, open(os.path.join(out, 'spec.json'), 'w'), indent=1)
        c = sh(MLV, os.path.join(R1, 'mlengine', 'client.py'), '--url', f'http://127.0.0.1:{NS_PORT}', '--train',
               os.path.join(d, 'train.csv'), '--test', os.path.join(d, 'test.csv'), '--spec', os.path.join(out, 'spec.json'),
               '--out', out)
        if c.returncode:
            raise SystemExit('NeuralSmith training failed: ' + c.stderr[-600:])
        return json.load(open(os.path.join(out, 'report.json')))
    finally:
        p.terminate()


def mark_at(tm, event, wall_s, **extra):
    """A mark for an event that happened at wall time wall_s (e.g. the xApp's t_us), on the monotonic clock."""
    rec = {'run_id': tm.run_id, 'event': event, 'monotonic_ns': time.monotonic_ns() - int((time.time() - wall_s) * 1e9),
           'wall_iso': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(wall_s)), 'source': 'xapp_event', **extra}
    with open(tm.path, 'a') as fh:
        fh.write(json.dumps(rec) + '\n')


def cap_check(sched, t0, ctrl):
    """Per cycle: first reserve and the first release after it, relative to the schedule; release within OFF?"""
    rows = []
    for i, c in enumerate(sched['cycles']):
        ph = {k: (t0 + a, t0 + b) for k, (a, b) in c['phases'].items()}
        nxt = t0 + sched['cycles'][i + 1]['phases']['ramp_up'][0] if i + 1 < len(sched['cycles']) else ph['off'][1]
        res = [e['t'] for e in ctrl if e['action'] == 'reserve' and ph['ramp_up'][0] - 10 <= e['t'] < ph['off'][0]]
        r0 = res[0] if res else None
        rel = next((e['t'] for e in ctrl if e['action'] == 'release' and r0 and e['t'] > r0), None)
        rows.append({'cycle': i + 1, 'reserve_s_after_ramp_start': r0 and round(r0 - ph['ramp_up'][0], 1),
                     'release_s_after_off_start': rel and round(rel - ph['off'][0], 1),
                     'off_len_s': round(nxt - ph['off'][0], 1),
                     'cap_lifts_within_off': bool(r0 and rel and ph['off'][0] <= rel < nxt)})
    return rows


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--collection-min', type=float, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--intent', default='I1')
    ap.add_argument('--mode', default='full', help="final spec mode (Exp-02)")
    ap.add_argument('--verify-cycles', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    out = os.path.join(E.RESULTS, 'exp04', a.run_id)
    os.makedirs(out, exist_ok=False)
    meta = {'run_id': a.run_id, 'git_commit': START_COMMIT, 'dry_run': a.dry_run, 'collection_min': a.collection_min,
            'seed': a.seed, 'intent': a.intent, 'mode': a.mode, 'status': 'running', 'human_touchpoints': 1,
            'provenance': E.provenance(__file__, extra=[os.path.join(R1, p) for p in (
                'zto/agent.py', 'zto/gate.py', 'zto/render.py', 'zto/deploy.py', 'zto/xapp/config.json.j2', 'zto/defaults.json',
                'collector/kpm_collector.py', 'dataset/build.py', 'mlengine/client.py')]),
            'prompts': json.load(open(os.path.join(R1, 'zto', 'prompts', 'lock.json')))}
    save = lambda: json.dump(meta, open(os.path.join(out, 'run_meta.json'), 'w'), indent=1)
    atexit.register(E.stop_traffic)
    GPU = O.ollama_gpu()
    if busy := O.gpu_quiet(gpu=GPU):
        raise SystemExit(f"GPU{GPU['index']} (Ollama) not quiet: " + '; '.join(busy))
    O.start_gpu_watch(out, GPU)
    meta['residency'] = pair_residency()
    if not meta['residency']['both_resident']:
        raise SystemExit('pair residency not established')
    ref = json.load(open(os.path.join(R1, 'zto', 'intents', 'reference', f'{a.intent}.json')))
    tm = timing.Timer(a.run_id, out)

    # 1. intent → spec → gate
    tm.mark('intent_received', intent=a.intent)
    r = agent.run(ref['_intent_text'], model=ORCH, seed=a.seed, mode=a.mode, tm=tm)
    open(os.path.join(out, 'agent_calls.jsonl'), 'a').write(json.dumps(r) + '\n')
    if not r['valid']:
        meta['status'] = 'agent_failed'
        return save()
    spec = r['spec']
    tm.mark('spec_ready')
    json.dump(spec, open(os.path.join(out, 'spec.json'), 'w'), indent=1)
    gr = gate.run(ref['_intent_text'], r['raw_spec'], judge_model=JUDGE, seed=a.seed, readback='extract', tm=tm)
    json.dump(gr, open(os.path.join(out, 'gate.json'), 'w'), indent=1, default=str)
    meta['gate'] = {'final': gr['final'], 'caught_by': gr['caught_by']}
    if gr['final'] != 'PASS':
        meta['human_touchpoints'] += gr['final'] == 'HOLD'
        meta['status'] = 'held' if gr['final'] == 'HOLD' else 'gate_failed'
        return save()

    # 2. data collection at the spec's T_s → dataset → NeuralSmith
    tm.mark('collection_start')
    rid = f'{a.run_id}-collect'
    p = sh(PY, os.path.join(R1, 'collector', 'kpm_collector.py'), '--run-id', rid, '--ts', spec['sampling_ms'], '--seed', a.seed,
           '--collection-min', a.collection_min, '--out-root', os.path.join(out, 'raw'))
    open(os.path.join(out, 'raw', f'{rid}.stdout.txt'), 'w').write(p.stdout + p.stderr)
    cm = json.load(open(os.path.join(out, 'raw', rid, 'meta.json')))
    tm.mark('collection_end', status=cm['status'])
    meta['collection'] = {k: cm.get(k) for k in ('status', 'rows', 'sim_speed_mean', 'sim_speed_start', 'skipped_starts',
                                                 'failure', 'gaps_gt_2ts', 'start_iso')}
    if cm['status'] != 'complete':
        meta['status'] = 'collection_failed'
        return save()
    meta['regime'] = cm.get('regime')  # run-level validity (speed_rules.regime_check); off-regime runs are kept, never used
    if not (meta['regime'] or {}).get('in_regime'):
        meta['status'] = 'off_regime'
        return save()
    n = len(cm['schedule']['cycles'])
    n_train = max(1, round(n * 9 / 13))
    ds = os.path.join(out, 'dataset')
    b = sh(PY, os.path.join(R1, 'dataset', 'build.py'), '--in', os.path.join(out, 'raw', rid, 'telemetry.csv'), '--ts', spec['sampling_ms'],
           '--window', spec['window_ms'], '--horizon', spec['horizon_ms'], '--threshold', spec['label_rule']['threshold_pct'],
           '--min-duration', spec['label_rule']['min_duration_s'], '--train-cycles', f'1-{n_train}',
           '--test-cycles', f'{n_train + 1}-{n}', '--out', ds)
    if b.returncode:
        raise SystemExit('dataset build failed: ' + b.stderr[-600:])
    # data curation: reservation lease from the training episodes (zto/lease.py; not an LLM/spec field)
    meta['lease'] = lease.compute(ds, spec['sampling_ms'], spec['horizon_ms'])
    tm.mark('lease_computed', lease_s=meta['lease']['lease_s'])
    tm.mark('train_post', train_cycles=f'1-{n_train}')
    md = os.path.join(out, 'model')
    os.makedirs(md, exist_ok=True)
    rep = train_neuralsmith(ds, spec, md)
    tm.mark('model_received', algorithm=rep.get('algorithm'))
    files = {f: open(os.path.join(md, f), 'rb').read() for f in ('model.onnx', 'preprocess.json')}

    with use_experiment(ExperimentRef(experiment_id=E.EXP, slug=E.SLUG)):
        rows = await g.snapshot(window_s=20)
        if problem := g.check_mapping(rows):
            raise SystemExit(f'mapping: {problem}')
        prot = next(x for x in rows if x['imsi'] == E.IMSI['ue3'])
        prof = profiles.load('rfsim')
        # 3. render → register (deploy) → verification traffic
        with tm.span('render'):
            cfg = render.render(spec, protected_ue_ref=int(prot['cu_ue_id']), policy='ml', workdir=prof['ric']['workdir'],
                                lease_s=meta['lease']['lease_s'])
            cfg_bytes = (json.dumps(cfg, indent=1) + '\n').encode()
            open(os.path.join(out, 'config.json'), 'wb').write(cfg_bytes)
            tm.mark('template_rendered', sha256=sha(cfg_bytes))
        meta['hashes'] = {'config.json': sha(cfg_bytes), **{f: sha(v) for f, v in files.items()}}
        # gate check 'model_parity' (deterministic, before registration): xApp features + candidate model vs offline
        mp = gate.model_parity(md, spec, E.NS, E.pod('ric')[0], meta['lease']['lease_s'], tm)
        gr['model_parity'] = mp
        gr['final'] = max((gr['final'], mp['result']), key=lambda r: gate.SEVERITY[r])
        gr['caught_by'] = gr['caught_by'] + (['model_parity'] if mp['result'] in ('HOLD', 'FAIL') else [])
        json.dump(gr, open(os.path.join(out, 'gate.json'), 'w'), indent=1, default=str)
        meta['gate'] = {'final': gr['final'], 'caught_by': gr['caught_by'], 'model_parity': mp}
        if mp['result'] != 'PASS':
            meta['status'] = 'gate_failed_model_parity'
            return save()
        meta['protected_ue_ref'] = int(prot['cu_ue_id'])
        pods0 = await g.pods_state()
        E.isolate(lambda e: None)
        trf_name, trf_ip = E.restart_trafficgen()
        await ric.stop_conflicting_kpm_monitor()
        with tm.span('register'):
            offset, ric_sha = await deploy.deploy(prof, cfg_bytes, files)
        meta['config_sha256_on_ric'] = ric_sha
        upods = E.ue_pods()
        for q in upods.values():
            E.route_to(q, trf_ip)
        bind = {u: E.pdu_ip(q) for u, q in upods.items()}
        sched = burst.schedule('ramp', cycles=a.verify_cycles, seed=a.seed + 1000, peak=28.0, up_s=45.0)
        span = sched['total_s'] + 15
        E.k('exec', upods['ue3'], '-c', 'ue', '--', *burst.remote_command({'cycles': [{'flows': [(0.0, E.ENV['ue3_rate_mbps'], span)]}]},
                                                                        server_ip=trf_ip, bind_ip=bind['ue3']))
        E.start_ping(upods['ue3'], trf_ip, span)
        await asyncio.sleep(10)
        t0 = time.time()
        for u in ('ue1', 'ue2'):
            E.k('exec', upods[u], '-c', 'ue', '--', *burst.remote_command(sched, server_ip=trf_ip, bind_ip=bind[u]))
        evs = []
        while time.time() < t0 + sched['total_s'] + 3:
            await asyncio.sleep(2)
            new, offset = await ric.read_events(prof, offset)
            evs += new
        await ric.stop()
        evs += (await ric.read_events(prof, offset))[0]
        E.stop_traffic()
        ctrl = [{'t': e['t_us'] / 1e6, 'action': e.get('action')} for e in evs if e.get('type') == 'control']
        with open(os.path.join(out, 'verify_events.jsonl'), 'w') as fh:
            for e in evs:
                if e.get('type') != 'sample':
                    fh.write(json.dumps(e) + '\n')
        first = next((c for c in ctrl if c['action'] == 'reserve'), None)
        if first:
            mark_at(tm, 'first_e2_control_sent', first['t'], action='reserve')
        open(os.path.join(out, 'ue3_ping.log'), 'w').write(E.k('exec', upods['ue3'], '-c', 'ue', '--', 'cat', E.PING_LOG))
        cc = cap_check(sched, t0, ctrl)
        meta['verify'] = {'schedule_seed': a.seed + 1000, 'cycles': cc, 't0': t0,
                          'cap_lifts_within_off_all': all(x['cap_lifts_within_off'] for x in cc),
                          'reserves_outside_on': sum(1 for c in ctrl if c['action'] == 'reserve' and not any(
                              t0 + cy['phases']['ramp_up'][0] - 10 <= c['t'] < t0 + cy['phases']['off'][0] for cy in sched['cycles']))}
        meta['pod_changes'] = g.pod_changes(pods0, await g.pods_state())
    meta['model_report'] = {k: rep.get(k) for k in ('algorithm', 'train_time_s', 'size_kb', 'onnx_latency_ms_p50', 'onnx_latency_ms_p99')}
    meta['status'] = 'complete' if first else 'no_control_sent'
    save()
    print(json.dumps({k: meta[k] for k in ('status', 'human_touchpoints', 'gate', 'hashes', 'verify')}, indent=1, default=str))


if __name__ == '__main__':
    asyncio.run(main())
