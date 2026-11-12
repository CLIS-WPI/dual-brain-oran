"""Shared setup for r1/scripts: paths, testbed identity, portal API, kubectl helpers, UE3 ping, provenance.

Run any r1 script from anywhere with the backend venv and env -i (never the portal .env):
  env -i PATH=$HOME/bin:/usr/local/bin:/usr/bin:/bin HOME=$HOME DEPLOYMENT_MODE=k8s K8S_TENANT_NAME=test-1 \
      testbed-manager/backend/venv/bin/python3 r1/scripts/<script>.py ...
The portal API reads its bearer token from /tmp/portal_token (written by the operator; never printed).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

R1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(R1)
BACKEND = os.path.join(ROOT, 'testbed-manager', 'backend')
RESULTS = os.path.join(R1, 'results')
for p in (BACKEND, os.path.join(R1, 'traffic'), os.path.join(R1, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

ENV = json.load(open(os.path.join(R1, 'env', 'experiment.json')))
EXP, SLUG, NS = ENV['experiment_id'], ENV['slug'], ENV['namespace']
H = {'X-Harmony-Experiment': EXP}
UES = ENV['ues']
IMSI = {u['key']: u['imsi'] for u in UES}
PING_INTERVAL_S = ENV['ping_interval_s']  # 10 Hz: ~0.009 Mbps UL, far below the guard's 2 Mbps UL limit
PORTAL = 'http://127.0.0.1:8888'
TOKEN_FILE = '/tmp/portal_token'


def req(method, path, body=None, headers=None, timeout=120):
    """Portal API call → (status, json|text). Token from TOKEN_FILE, never printed."""
    tok = open(TOKEN_FILE).read().strip()
    h = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + tok, **(headers or {})}
    r = urllib.request.Request(PORTAL + path, method=method, headers=h,
                               data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        txt = e.read().decode(errors='replace')
        try:
            return e.code, json.loads(txt)
        except ValueError:
            return e.code, txt[:500]


def k(*a):
    return subprocess.run(['kubectl', '-n', NS, *a], capture_output=True, text=True).stdout


def pod(comp, ue=None):
    """[name, podIP] of the running R1 pod for a component (gnb-rfsim, virtual-ue, traffic-gen, ric)."""
    sel = f'harmony.telecom/experiment={SLUG},app.kubernetes.io/component={comp}' + (f',harmony.telecom/ue={ue}' if ue else '')
    return k('get', 'pods', '-l', sel, '--field-selector=status.phase=Running', '-o',
             'jsonpath={.items[0].metadata.name} {.items[0].status.podIP}').split()


def pdu_ip(p):
    return k('exec', p, '-c', 'ue', '--', 'sh', '-c', "ip -4 -o addr show dev oaitun_ue1 | awk '{print $4}'").strip().split('/')[0]


def route_to(p, ip):
    """Same host route the portal iperf sets: traffic-gen via the PDU session."""
    k('exec', p, '-c', 'ue', '--', 'sh', '-c', f'ip route replace {ip}/32 dev oaitun_ue1')


PING_LOG = '/tmp/r1_ping.log'


def start_ping(p, dst, duration_s):
    """UE3 RTT probe: timestamped ping every PING_INTERVAL_S through the PDU session, detached."""
    k('exec', p, '-c', 'ue', '--', 'sh', '-c',
      f'rm -f {PING_LOG}; nohup ping -D -i {PING_INTERVAL_S} -w {int(duration_s)} -I oaitun_ue1 {dst} > {PING_LOG} 2>&1 &')


def git(*a):
    return subprocess.run(['git', '-C', ROOT, *a], capture_output=True, text=True).stdout.strip()


def provenance(script_file, extra=()):
    """Commit the run started from + blob of every file it depends on; dirty = uncommitted r1/ or backend changes."""
    files = [os.path.abspath(script_file), os.path.abspath(__file__), os.path.join(R1, 'traffic', 'burst.py'), *extra]
    rel = [os.path.relpath(f, ROOT) for f in files]
    return {'repo_head': git('rev-parse', 'HEAD'),
            'blobs': {r: git('hash-object', f) for r, f in zip(rel, files)},
            'uncommitted': [ln[3:] for ln in git('status', '--porcelain', '--', *rel,
                                                 'testbed-manager/backend/services/dual_brain').splitlines()]}


# ── traffic hygiene (exp01-select 2026-11-03: leftover flows after an abort contaminated the next run) ──

def _ps(p):
    """[(pid, stat, args)] in a UE container."""
    out = k('exec', p, '-c', 'ue', '--', 'ps', '-eo', 'pid=,stat=,args=')
    rows = []
    for ln in out.splitlines():
        parts = ln.split(None, 2)
        if len(parts) >= 2 and parts[0].isdigit():
            rows.append((int(parts[0]), parts[1], parts[2] if len(parts) > 2 else ''))
    return rows


def _traffic_pids(rows):
    """Live (non-zombie) R1 traffic processes: iperf / ping / sleep and the burst scheduler shells."""
    out = []
    for pid, stat, args in rows:
        if stat.startswith('Z') or pid == 1:
            continue
        cmd = args.split()[0].rsplit('/', 1)[-1] if args else ''
        if cmd in ('iperf', 'ping', 'sleep') or (cmd == 'sh' and 'r1_iperf.csv' in args):
            out.append(pid)
    return out


assert _traffic_pids([(1, 'Ss', '/bin/sh -c x'), (5, 'S', 'iperf -c 1'), (6, 'Z', '[iperf]'),
                      (7, 'S', 'sh -c rm -f /tmp/r1_iperf.csv ...'), (8, 'S', 'nr-uesoftmodem')]) == [5, 7]


def ue_pods():
    return {u: pod('virtual-ue', u)[0] for u in ('ue1', 'ue2', 'ue3')}


def live_flows(pods=None):
    """{ue: [pids]} of live traffic processes per UE pod."""
    return {u: _traffic_pids(_ps(p)) for u, p in (pods or ue_pods()).items()}


def stop_traffic(pods=None, tries=3):
    """Kill every live R1 traffic process in the UE pods (by pid; zombies are left to the container's PID 1)."""
    pods = pods or ue_pods()
    for _ in range(tries):
        live = live_flows(pods)
        if not any(live.values()):
            return {}
        for u, pids in live.items():
            if pids:
                k('exec', pods[u], '-c', 'ue', '--', 'kill', *map(str, pids))
        import time as _t
        _t.sleep(1)
    return live_flows(pods)


def trafficgen_memory():
    """(current bytes, limit bytes) of the traffic-gen container (cgroup v2)."""
    p = pod('traffic-gen')[0]
    out = k('exec', p, '--', 'sh', '-c', 'cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.max').split()
    cur = int(out[0]) if out and out[0].isdigit() else None
    lim = int(out[1]) if len(out) > 1 and out[1].isdigit() else None
    return cur, lim


def restart_trafficgen(timeout_s=180):
    """Fresh iperf2 server for a collection (reverse-UDP sessions accumulate in a long-lived server)."""
    dep = f'{ENV["tenant"]}-{SLUG}-rfsim-trf'
    old = set(k('get', 'pods', '-l', f'harmony.telecom/experiment={SLUG},app.kubernetes.io/component=traffic-gen',
                '-o', 'jsonpath={.items[*].metadata.name}').split())
    k('rollout', 'restart', f'deploy/{dep}')
    k('rollout', 'status', f'deploy/{dep}', f'--timeout={timeout_s}s')
    import time as _t
    deadline = _t.time() + timeout_s
    while _t.time() < deadline:  # the old pod stays 'Running' while it terminates: wait until only the new one is left
        names = set(k('get', 'pods', '-l', f'harmony.telecom/experiment={SLUG},app.kubernetes.io/component=traffic-gen',
                      '-o', 'jsonpath={.items[*].metadata.name}').split())
        if names and not names & old and len(names) == 1:
            return pod('traffic-gen')
        _t.sleep(2)
    raise RuntimeError(f'traffic-gen restart: old pod(s) {sorted(old)} still present after {timeout_s} s')


IDLE_PRB_PCT, IDLE_HOLD_S, IDLE_TIMEOUT_S = 5.0, 30, 600


def isolate(log=print) -> dict:
    """Stop leftover UE traffic, then wait for no live flow and an idle cell (PRB < IDLE_PRB_PCT for IDLE_HOLD_S)."""
    import time as _t
    from services.dual_brain import gnb_truth
    t0 = _t.time()
    left = stop_traffic()
    while _t.time() - t0 < IDLE_TIMEOUT_S:
        live = live_flows()
        rows = gnb_truth.parse(k('logs', pod('gnb-rfsim')[0], '-c', 'gnb', '--timestamps', f'--since={IDLE_HOLD_S + 5}s'))[0]
        cell = [100 * sum(x for _, x in r['ues'].values()) / r['budget'] for r in rows if r['t'] >= _t.time() - IDLE_HOLD_S]
        if not any(live.values()) and len(cell) >= IDLE_HOLD_S - 3 and max(cell) < IDLE_PRB_PCT:
            out = {'wait_s': round(_t.time() - t0, 1), 'stopped_leftover': bool(left), 'idle_cell_prb_max': round(max(cell), 2)}
            log({'event': 'isolated', **out})
            return out
        if any(live.values()):
            stop_traffic()
        _t.sleep(5)
    raise SystemExit(f'isolation: cell not idle / flows still live after {IDLE_TIMEOUT_S} s')
