"""COMMAG-R1 UE3 calibration (option B): UE-side DL channel, 3 full restarts, 5-min DL run.

usage: calib_channel.py <ploss_dB> <noise_dB> [--measure-s 300]   (the run that set UE3 to 30/−10 dB; frozen)
Prints per-restart attach results and per-UE DL MCS/BLER/goodput + UL SNR over the run. Restarts gNB + UEs only
(the RIC keeps running, so the gNB finds it). Needs /tmp/portal_token. Portal iperf, 1 Hz ping era: historical.
"""
import json, os, subprocess, sys, time

import r1env as E
from r1env import req

EXP, SLUG, NS, H = E.EXP, E.SLUG, E.NS, E.H
UES = [(E.IMSI['ue1'], 'ue1', '20M'), (E.IMSI['ue2'], 'ue2', '20M'), (E.IMSI['ue3'], 'ue3', '1M')]
EDGE = UES[2][0]
ploss, noise = float(sys.argv[1]), float(sys.argv[2])
MEASURE_S = int(sys.argv[sys.argv.index('--measure-s') + 1]) if '--measure-s' in sys.argv else 300
ENV = {'PATH': os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin'), 'HOME': os.environ.get('HOME', '/tmp'),
       'DEPLOYMENT_MODE': 'k8s', 'K8S_TENANT_NAME': 'test-1'}  # never the portal .env (no JWT_SECRET)
BACKEND = E.BACKEND
RUN_ID = f"calib-{ploss:g}_{noise:g}-" + time.strftime('%Y%m%d-%H%M%S')
OUT = os.path.join(E.RESULTS, 'calibration', RUN_ID)
os.makedirs(OUT, exist_ok=True)
LOG = open(f'{OUT}/calibration.txt', 'w')
SAMPLER = subprocess.Popen(['python3', '-I', os.path.join(E.R1, 'scripts', 'host_sampler.py'), f'{OUT}/host_load.jsonl',
                            '--interval', '60', '--until-pid', str(os.getpid())])
RETRIES = {'iperf_start': 0}
_print = print


def print(*a, **kw):  # tee to the run's evidence folder
    _print(*a, **kw, flush=True)
    _print(*a, **{k: v for k, v in kw.items() if k != 'flush'}, file=LOG, flush=True)


def guard_call(expr):
    """Evaluate an async channel_guard expression in the R1 experiment scope (backend venv)."""
    code = f'''
import asyncio, json
from services.k8s.experiment_ctx import ExperimentRef, use_experiment
from services.dual_brain import channel_guard as g
async def m():
    with use_experiment(ExperimentRef(experiment_id="{EXP}", slug="{SLUG}")):
        print(json.dumps(await {expr}))
asyncio.run(m())'''
    p = subprocess.run(['./venv/bin/python3', '-c', code], cwd=BACKEND, capture_output=True, text=True, env=ENV)
    if p.returncode:
        raise RuntimeError(p.stderr[-800:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def save_crashes(tag):
    files = guard_call(f'g.save_crash_logs("{OUT}/{tag}")')
    if files:
        print(f'  crash evidence ({tag}):', [f.rsplit('/', 1)[-1] for f in files])
    return files


def k(*a):
    return subprocess.run(['kubectl', '-n', NS, *a], capture_output=True, text=True, env=ENV).stdout


def snapshot(window_s):
    code = f'''
import asyncio, json
from services.k8s.experiment_ctx import ExperimentRef, use_experiment
from services.dual_brain import channel_guard as g
async def m():
    with use_experiment(ExperimentRef(experiment_id="{EXP}", slug="{SLUG}")):
        rows = await g.snapshot(window_s={window_s})
        print(json.dumps({{"rows": rows, "mapping": g.check_mapping(rows), "edge": g.check(rows, "{EDGE}", 5)}}))
asyncio.run(m())'''
    p = subprocess.run(['./venv/bin/python3', '-c', code], cwd=BACKEND, capture_output=True, text=True, env=ENV)
    if p.returncode:
        raise RuntimeError(p.stderr[-800:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def set_canvas():
    st, exp = req('GET', f'/api/experiments/{EXP}')
    cd = exp['canvas_data']
    by_id = {n['id']: n for n in cd['nodes']}
    for n in cd['nodes']:
        if n['type'] == 'virtual_ue' and n['data']['config'].get('imsi') == EDGE:
            n['data']['config'].update(channel_ploss_db=ploss, channel_noise_db=noise)
    # Iperf nodes: DL UDP (UE1/UE2 20M center, UE3 1M background), found via their UE edge.
    rate = {imsi: r for imsi, _, r in UES}
    for e in cd['edges']:
        s, t = by_id.get(e['source']), by_id.get(e['target'])
        if s and t and s['type'] == 'virtual_ue' and t['type'] == 'iperf3':
            t['data']['config'].update(protocol='udp', direction='dl',
                                       bitrate=rate.get(s['data']['config'].get('imsi'), '1M'))
    print('canvas', req('PATCH', f'/api/experiments/{EXP}', {'canvas_data': cd})[0], f'UE3 channel {ploss}/{noise} dB')


def full_restart(n):
    req('POST', '/api/ue/virtual/batch/stop', {'imsis': [i for i, _, _ in UES], 'count': 3}, H, timeout=300)
    req('POST', '/api/ran/stop', {'mode': 'rfsim'}, H, timeout=300)
    k('wait', '--for=delete', 'pod', '-l',
      f'harmony.telecom/experiment={SLUG},app.kubernetes.io/component in (gnb-rfsim,virtual-ue)', '--timeout=180s')
    st, r = req('POST', '/api/ran/start', {'mode': 'rfsim', 'config_file': 'gnb.rfsim.band77.51prb.conf',
                                           'startup_cleanup': True}, H, timeout=300)
    k('rollout', 'status', f'deploy/test-1-{SLUG}-rfsim-gnb', '--timeout=180s')
    pdu = {}
    for imsi, ue, _ in UES:  # strictly in order: next UE only after this one has its PDU
        ip = None
        for _ in range(20):
            st, r = req('POST', f'/api/ue/virtual/{imsi}/start', {'interface': ''}, H, timeout=300)
            ip = (r or {}).get('ip')
            if ip:
                break
            time.sleep(6)
        pdu[ue] = ip
    time.sleep(20)  # a few MAC stats periods
    snap = snapshot(20)
    rows = {r['ue']: r for r in snap['rows']}
    ok = all(pdu.values()) and snap['mapping'] is None
    save_crashes(f'restart{n}')
    print(f'restart {n}: ' + ', '.join(f"{ue.upper()} pdu={pdu[ue]} rnti={rows.get(ue.upper(), {}).get('rnti')}"
                                       for _, ue, _ in UES) + f" -> {'ATTACHED' if ok else 'FAIL: ' + str(snap['mapping'])}")
    return ok


def measure():
    trf = f'test-1-{SLUG}-rfsim-trf'
    snap0 = snapshot(20)
    if snap0['mapping']:
        print('before measurement:', snap0['mapping'])
    for imsi, ue, rate in UES:
        for attempt in range(3):  # kubectl exec in the PDU lookup can time out under host load
            st, r = req('POST', f'/api/ue/{10000 + int(imsi[-4:])}/iperf/start',
                        {'server': trf, 'duration': MEASURE_S, 'direction': 'dl', 'protocol': 'udp',
                         'bandwidth': rate, 'imsi': imsi}, H, timeout=120)
            if (r or {}).get('success'):
                break
            RETRIES['iperf_start'] += 1
            time.sleep(5)
        print(f'iperf DL {ue} {rate}:', st, (r or {}).get('success'), str((r or {}).get('message') or '')[:120])
    load0 = os.getloadavg()
    t0 = time.time()
    time.sleep(MEASURE_S + 5)
    snap = snapshot(MEASURE_S)
    # sim/wall speed: gNB AURA-PRB budget / real-time budget, 10 s means over the window
    glog = k('logs', k('get', 'pods', '-l', f'harmony.telecom/experiment={SLUG},app.kubernetes.io/component=gnb-rfsim',
                       '-o', 'jsonpath={.items[0].metadata.name}'), '-c', 'gnb', '--timestamps', f'--since={MEASURE_S + 10}s')
    code = ("import sys,json; sys.path.insert(0,'.'); from services.dual_brain import channel_guard as g; "
            "print(json.dumps(g.speed_ratios(sys.stdin.read(), 61200)))")
    ratios = json.loads(subprocess.run(['./venv/bin/python3', '-c', code], cwd=BACKEND, input=glog, capture_output=True,
                                       text=True, env=ENV).stdout or '[]')
    ratios = [(t_, r_) for t_, r_ in ratios if t0 <= t_ <= t0 + MEASURE_S]
    w10 = [round(sum(r_ for t_, r_ in ratios if t0 + i <= t_ < t0 + i + 10) /
                 max(1, sum(1 for t_, _ in ratios if t0 + i <= t_ < t0 + i + 10)), 2) for i in range(0, MEASURE_S, 10)]
    w10 = [x for x in w10 if x]
    snap['sim_speed'] = {'per_10s': w10, 'mean': round(sum(w10) / len(w10), 2) if w10 else None,
                         'min': min(w10, default=None), 'max': max(w10, default=None)}
    snap['host_loadavg'] = {'start': load0, 'end': os.getloadavg()}
    print(f'\nDL over {MEASURE_S} s (gNB MAC stats, samples with goodput > 0):')
    print(f"{'UE':4} {'IMSI':16} {'RNTI':5} {'DL MCS mean':>11} {'DL BLER':>8} {'goodput Mbps':>12} {'n':>4} {'UL SNR dB':>9}  DL CQI")
    for r in snap['rows']:
        print(f"{r['ue']:4} {r['imsi']:16} {r['rnti'] or '-':5} {str(r['dl_mcs_mean']):>11} {str(r['dl_bler_mean']):>8} "
              f"{str(r['dl_goodput_mbps_mean']):>12} {r['dl_active_n']:>4} {str(r['ul_snr_db']):>9}  not reported")
    print('edge check (UE3 mean DL MCS >= 5 below UE1/UE2):', snap['edge'] or 'PASS')
    restarts = {ue: k('get', 'pods', '-l', f'harmony.telecom/experiment={SLUG},harmony.telecom/ue={ue}', '-o',
                      'jsonpath={.items[0].status.containerStatuses[0].restartCount}') for _, ue, _ in UES}
    print('UE container restarts during the run (must be 0):', restarts)
    snap['ue_restarts'] = restarts
    missing = [r['ue'] for r in snap['rows'] if not r['dl_active_n']]
    snap['retries'], snap['failed_missing_data'] = dict(RETRIES), missing
    print('iperf start retries this run:', RETRIES['iperf_start'])
    print('RUN', 'FAILED: no DL data for ' + ', '.join(missing) if missing else 'complete: every UE has DL data')
    hl = subprocess.run(['python3', '-I', os.path.join(E.R1, 'scripts', 'host_sampler.py'), f'{OUT}/host_load.jsonl',
                         '--summary'], capture_output=True, text=True).stdout
    print('host load per minute:\n' + hl)
    snap['crash_logs'] = save_crashes('after_measure')
    sp = snap['sim_speed']
    if sp['mean']:
        print(f"sim/wall speed: mean {sp['mean']}x, 10 s means {sp['min']}..{sp['max']}x "
              f"({(sp['max'] / sp['mean'] - 1):+.0%} / {(sp['min'] / sp['mean'] - 1):+.0%} around mean); series {sp['per_10s']}")
    else:
        print('sim/wall speed: no AURA-PRB lines in the window')
    print('host load avg start/end:', snap['host_loadavg'])
    json.dump({'ploss_dB': ploss, 'noise_power_dB': noise, 'measure_s': MEASURE_S, **snap},
              open(f'{OUT}/calibration.json', 'w'), indent=1)


set_canvas()
results = [full_restart(i) for i in (1, 2, 3)]
print(f'run {RUN_ID} → {OUT}')
print('attach on 3 consecutive full restarts:', 'PASS' if all(results) else f'FAIL {results}')
if all(results):
    measure()
