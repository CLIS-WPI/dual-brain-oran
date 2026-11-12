"""make smoke: prove every R1 interface works before any experiment. Exit 0 only if every check passes.

  1. RAN: three UEs attached and mapped (IMSI → RNTI → KPM RAN UE ID) on the R1 gNB
  2. KPM: the congestion xApp (collect mode) receives indications at T_s = defaults.json sampling_ms (median period within ±10 %)
  3. control: with 2 Mbps DL on UE1/UE2, a stub reservation (threshold policy, threshold 0) is sent over SLICE (SM 145) and accepted: xApp
     'reserve' event + gNB [AURA-QUOTA] lines; then released (SLICE DEL)
  4. ML engine: POST /train on a toy CSV returns a full report from both backends (NeuralSmith, FLAML)
  5b. model parity: the main exp01 model through the xApp's feature + model path (cc_replay) on recorded
      telemetry: features == build.py, predictions == offline ONNX, not constant; lease replay lifts the cap in OFF
      in every cycle (zto/parity.py)
  5. ZTO: schema/render/gate self-checks, 6 template slots bounded, prompts match prompts/lock.json
  6. Ollama: reachable, the four R1 models present
Run with the backend venv (see r1env.py). Writes results/smoke/smoke-<time>.json.
"""
import asyncio
import json
import os
import random
import subprocess
import sys
import time

import r1env as E
from r1env import k, pod

R1, MLV = E.R1, os.path.join(E.R1, '.venv-mlengine', 'bin', 'python')
REPORT_FIELDS = ('algorithm', 'accuracy', 'f1', 'precision', 'recall', 'fpr', 'confusion', 'onnx_latency_ms_p50',
                 'onnx_latency_ms_p99', 'size_kb', 'train_time_s', 'features_used')
results = {}


def check(name, ok, **info):
    results[name] = {'ok': bool(ok), **info}
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(info)[:300]}", flush=True)


async def ran_kpm_control():
    from services.dual_brain import channel_guard as g, profiles, ric
    from services.k8s.experiment_ctx import ExperimentRef, use_experiment
    with use_experiment(ExperimentRef(experiment_id=E.EXP, slug=E.SLUG)):
        rows = await g.snapshot(window_s=20)
        problem = g.check_mapping(rows)
        check('ran_3_ues', not problem and len(rows) == 3, ues=[(r['ue'], r['rnti'], r['cu_ue_id']) for r in rows],
              problem=problem)
        if problem:
            return
        prot = next(r for r in rows if r['imsi'] == E.IMSI['ue3'])
        TS = json.load(open(os.path.join(R1, 'zto', 'defaults.json')))['sampling_ms']
        prof = profiles.load('rfsim', {'kpm': {'period_ms': TS}})
        await ric.stop_conflicting_kpm_monitor()
        off = await ric.start(prof, mode='collect', reserve_pct=20)
        await asyncio.sleep(15)
        evs, off = await ric.read_events(prof, off)
        ts = [e['t_us'] / 1e3 for e in evs if e.get('type') == 'sample']
        gaps = sorted(b - a for a, b in zip(ts, ts[1:]))
        med = gaps[len(gaps) // 2] if gaps else None
        check(f'kpm_rate_{TS}ms', med and abs(med - TS) <= 0.1 * TS, samples=len(ts), median_period_ms=med)
        prof = profiles.load('rfsim', {'kpm': {'period_ms': TS}, 'control': {
            'policy': 'threshold', 'threshold_cell_prb': 0, 'min_duration_samples': 1, 'trigger_consecutive': 1,
            'release_after_normal_samples': 10 ** 9, 'min_hold_samples': 10 ** 9, 'protected_ue_ref': int(prot['cu_ue_id'])}})
        import burst  # the xApp reserves only with >= 2 active UEs: 2 Mbps DL on UE1/UE2 for the check
        trf_ip = pod('traffic-gen')[1]
        for u in ('ue1', 'ue2'):
            p = pod('virtual-ue', u)[0]
            E.route_to(p, trf_ip)
            k('exec', p, '-c', 'ue', '--', *burst.remote_command({'cycles': [{'flows': [(0.0, 2.0, 90)]}]},
                                                               server_ip=trf_ip, bind_ip=E.pdu_ip(p)))
        t0 = time.time()
        off = await ric.start(prof, mode='control', reserve_pct=20)
        reserve = None
        while reserve is None and time.time() - t0 < 60:
            evs, off = await ric.read_events(prof, off)
            reserve = next((e for e in evs if e.get('type') == 'control' and e.get('action') == 'reserve'), None)
            await asyncio.sleep(1)
        await asyncio.sleep(5)
        await ric.stop()  # SIGTERM → SLICE DEL
        await asyncio.sleep(3)
        glog = k('logs', pod('gnb-rfsim')[0], '-c', 'gnb', f'--since={int(time.time() - t0) + 5}s')
        quota = [ln for ln in glog.splitlines() if '[AURA-QUOTA]' in ln]
        check('control_accepted', reserve is not None and quota, reserve=reserve, aura_quota_lines=len(quota),
              released=any('del' in ln.lower() for ln in quota))


def mlengine():
    import tempfile
    import urllib.request
    d = tempfile.mkdtemp(prefix='smoke-ml-')
    rnd = random.Random(42)
    rows = []
    for _ in range(400):  # toy: label = mean_prb + noise above 80
        m = rnd.uniform(20, 100)
        rows.append([m, rnd.uniform(0, 10), m - 5, m + 5, rnd.uniform(-5, 5), m + rnd.uniform(-3, 3), int(m + rnd.gauss(0, 4) > 80)])
    hdr = 'mean_prb,std_prb,min_prb,max_prb,slope_prb,last_prb,label\n'
    for name, part in (('train.csv', rows[:300]), ('test.csv', rows[300:])):
        open(os.path.join(d, name), 'w').write(hdr + ''.join(','.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in r) + '\n' for r in part))
    spec = {'task': 'classification', 'target_column': 'label', 'latency_budget_ms': 5, 'max_size_kb': 2048, 'mode': 'fast++', 'seed': 42}
    procs = []
    try:
        for be, port in (('neuralsmith', 8095), ('flaml', 8096)):
            procs.append(subprocess.Popen([MLV, os.path.join(R1, 'mlengine', 'server.py'), '--backend', be, '--port', str(port)],
                                          env={**os.environ, 'MLENGINE_RUNS': os.path.join(d, f'runs-{be}')},
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for _ in range(30):
            try:
                urllib.request.urlopen('http://127.0.0.1:8096/health', timeout=2)
                urllib.request.urlopen('http://127.0.0.1:8095/health', timeout=2)
                break
            except OSError:
                time.sleep(1)
        ns_time = None
        for be, port in (('neuralsmith', 8095), ('flaml', 8096)):
            s = dict(spec, time_budget_s=ns_time or 15)
            json.dump(s, open(os.path.join(d, f'spec_{be}.json'), 'w'))
            out = subprocess.run([MLV, os.path.join(R1, 'mlengine', 'client.py'), '--url', f'http://127.0.0.1:{port}',
                                  '--train', os.path.join(d, 'train.csv'), '--test', os.path.join(d, 'test.csv'),
                                  '--spec', os.path.join(d, f'spec_{be}.json'), '--out', os.path.join(d, f'out-{be}')],
                                 capture_output=True, text=True, timeout=1800)
            try:
                rep = json.loads(out.stdout)
            except ValueError:
                rep = {}
            missing = [f for f in REPORT_FIELDS if rep.get(f) is None]
            ns_time = ns_time or rep.get('train_time_s')
            check(f'mlengine_{be}', out.returncode == 0 and not missing, missing=missing, algorithm=rep.get('algorithm'),
                  train_time_s=rep.get('train_time_s'), err=out.stderr[-300:] if out.returncode else None)
    finally:
        for p in procs:
            p.terminate()


def zto():
    import importlib
    sys.path.insert(0, os.path.join(R1, 'zto'))
    try:
        for m in ('render', 'resolve', 'gate', 'agent'):
            importlib.import_module(m)  # module-level self-checks
        import agent
        lock = json.load(open(os.path.join(R1, 'zto', 'prompts', 'lock.json')))
        frozen = {m: agent.prompt_sha256(m) == h for m, h in lock['sha256'].items()}
        slots = json.loads(subprocess.run([sys.executable, os.path.join(R1, 'scripts', 'list_slots.py'), '--json'],
                                          capture_output=True, text=True).stdout)
        check('zto_selfchecks', all(frozen.values()) and not slots['errors'], prompts_frozen=frozen,
              n_slots=slots['n_slots'], lock_commit=lock['frozen_at_commit'])
    except Exception as exc:
        check('zto_selfchecks', False, error=f'{type(exc).__name__}: {exc}')


def model_parity():
    """The trained main model (exp01-select T_s 500 / H 2.5 s) through the xApp's own feature + model path in the RIC
    pod on recorded telemetry (zto/parity.py): features == build.py, predictions == offline ONNX, not constant."""
    sys.path.insert(0, os.path.join(R1, 'zto'))
    try:
        import parity
        d = json.load(open(os.path.join(R1, 'zto', 'defaults.json')))
        md = os.path.join(E.RESULTS, 'exp01', 'exp01-20261104', f"ts{d['sampling_ms']}_h{d['horizon_ms']}", 'neuralsmith')
        import lease
        ls = lease.compute(os.path.dirname(md), d['sampling_ms'], d['horizon_ms'])['lease_s']
        v = parity.check(md, d['sampling_ms'], d['window_ms'], E.NS, E.pod('ric')[0],
                         debounce_n=-(-1000 // d['sampling_ms']), lease_s=ls, threshold=d['label_rule']['threshold_pct'],
                         min_dur=-(-d['label_rule']['min_duration_s'] * 1000 // d['sampling_ms']))
        check('model_parity', v['result'] == 'PASS', model=os.path.relpath(md, R1), **v)
    except Exception as exc:
        check('model_parity', False, error=f'{type(exc).__name__}: {exc}')


def ollama():
    import ollama_env as O
    try:
        tags = {m['name'] for m in O.api('/api/tags').get('models', [])}
        check('ollama_models', set(O.MODELS) <= tags, version=O.api('/api/version').get('version'),
              missing=sorted(set(O.MODELS) - tags))
    except Exception as exc:
        check('ollama_models', False, error=str(exc))


def main():
    asyncio.run(ran_kpm_control())
    mlengine()
    zto()
    model_parity()
    ollama()
    out = os.path.join(E.RESULTS, 'smoke', time.strftime('smoke-%Y%m%d-%H%M%S.json'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({'provenance': E.provenance(__file__), 'checks': results}, open(out, 'w'), indent=1)
    ok = all(r['ok'] for r in results.values())
    print('SMOKE', 'OK' if ok else 'FAILED', '→', os.path.relpath(out, E.ROOT))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
