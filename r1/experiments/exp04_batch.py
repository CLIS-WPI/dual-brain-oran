"""Unattended Exp-04 batch (Milad, 2026-11-04): dry run (slot dry7 since 2026-11-05), then the six official runs, then stop (no Exp-03).

usage: exp04_batch.py --deadline 'YYYY-mm-ddTHH:MM' [--batch exp04-20261104] [--skip-dry-run <passed dry run id>]
--skip-dry-run (Milad, 2026-11-05): reuse an earlier dry run instead of running one; accepted only if its run_meta is a
complete dry run with the cap lifting within OFF in every cycle (recorded in the batch json as dry_run_reused).
Every run is exp04_e2e.py in its own session; its collector runs the start gate (idle check + ~3 min loaded probe,
retry every 15 min, R1_DEADLINE_EPOCH). The dry run (10 min, seed 26) must be complete with the cap lifting within OFF
in every verification cycle, else the batch stops. Official slots, interleaved 20/10 min: (20, 31) (10, 41) (20, 32)
(10, 42) (20, 33) (10, 43). A run that comes out off_regime, or whose collection was aborted by the sim-speed drift
guard, is kept, marked, and repeated (same seed, run id suffix -r1/-r2; max 2 repeats per slot). Any other outcome
(complete, gate HOLD/FAIL, no control) is a result and is kept. A run without a final run_meta (GPU abort, crash) stops
the batch. At the deadline the current run is terminated (marked stopped_deadline) and nothing new starts.
At the end: traffic stopped, xApp stopped, Ollama models unloaded (stack idle); timing_report over the complete,
in-regime official runs → results/exp04/<batch>-timing_summary.json; results/exp04/<batch>-batch.json.
No commits (the operator commits after the batch).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time

R1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R1, 'scripts'), R1]
import r1env as E  # noqa: E402

PY = sys.executable
SLOTS = [(20, 31), (10, 41), (20, 32), (10, 42), (20, 33), (10, 43)]
REPEATABLE = {'off_regime'}


def log(batch, **x):
    x['t'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    print(json.dumps(x), flush=True)
    with open(os.path.join(E.RESULTS, 'exp04', f'{batch}-batch.log.jsonl'), 'a') as fh:
        fh.write(json.dumps(x) + '\n')


def run(rid, cmin, seed, dry, deadline):
    args = [PY, os.path.join(R1, 'experiments', 'exp04_e2e.py'), '--run-id', rid, '--collection-min', cmin, '--seed', seed]
    p = subprocess.Popen([str(x) for x in args + (['--dry-run'] if dry else [])], cwd=R1, start_new_session=True,
                         env={**os.environ, 'R1_DEADLINE_EPOCH': str(deadline)},
                         stdout=open(os.path.join(E.RESULTS, 'exp04', f'{rid}.stdout.txt'), 'w'), stderr=subprocess.STDOUT)
    while p.poll() is None:
        if time.time() > deadline:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(60)
            E.stop_traffic()
            return {'status': 'stopped_deadline'}
        time.sleep(10)
    mp = os.path.join(E.RESULTS, 'exp04', rid, 'run_meta.json')
    m = json.load(open(mp)) if os.path.exists(mp) else {}
    if m.get('status') == 'collection_failed' and ((m.get('collection') or {}).get('failure') or {}).get('code') == 'sim_speed_drift':
        m['status_batch'] = 'drift_abort'
    return m or {'status': 'no_run_meta', 'returncode': p.returncode}


def slot(batch, name, cmin, seed, dry, deadline, out):
    for k in range(3):  # first attempt + max 2 repeats
        rid = f'{batch}-{name}' + (f'-r{k}' if k else '')
        if time.time() > deadline:
            out.append({'slot': name, 'run_id': rid, 'status': 'not_started_deadline'})
            return None
        log(batch, event='run_start', slot=name, run_id=rid, collection_min=cmin, seed=seed)
        m = run(rid, cmin, seed, dry, deadline)
        rec = {'slot': name, 'run_id': rid, 'collection_min': cmin, 'seed': seed, 'status': m.get('status'),
               'status_batch': m.get('status_batch'), 'regime': m.get('regime'), 'gate': (m.get('gate') or {}).get('final'),
               'cap_lifts_within_off_all': (m.get('verify') or {}).get('cap_lifts_within_off_all'), 'lease_s': (m.get('lease') or {}).get('lease_s')}
        out.append(rec)
        log(batch, event='run_end', **rec)
        if m.get('status') in REPEATABLE or m.get('status_batch') == 'drift_abort':
            continue
        return m
    return None


async def idle():
    from services.dual_brain import ric
    from services.k8s.experiment_ctx import ExperimentRef, use_experiment
    with use_experiment(ExperimentRef(experiment_id=E.EXP, slug=E.SLUG)):
        await ric.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deadline', required=True)
    ap.add_argument('--batch', default='exp04-' + time.strftime('%Y%m%d'))
    ap.add_argument('--skip-dry-run')
    a = ap.parse_args()
    deadline = dt.datetime.fromisoformat(a.deadline).timestamp()
    os.makedirs(os.path.join(E.RESULTS, 'exp04'), exist_ok=True)
    out, verdict = [], None
    log(a.batch, event='batch_start', deadline=a.deadline, start_commit=E.git('rev-parse', 'HEAD'))
    import ollama_env as O
    try:  # pre-flight (2026-11-05: Ollama lost the GPU after a systemd reload): nvidia-smi in the container = GPU1
        got, want = O.ollama_gpu().get('uuid'), json.load(open(os.path.join(R1, 'env', 'versions.json')))['ollama']['gpu']['uuid']
        err = None if got == want else f'Ollama GPU {got} != recorded {want}'
    except Exception as exc:
        err = f'nvidia-smi in the Ollama container failed: {exc}'
    if err:
        log(a.batch, event='preflight_failed', error=err)
        json.dump({'batch': a.batch, 'verdict': 'preflight_failed', 'error': err, 'runs': []},
                  open(os.path.join(E.RESULTS, 'exp04', f'{a.batch}-batch.json'), 'w'), indent=1)
        sys.exit(1)
    log(a.batch, event='preflight_ok', ollama_gpu=got)
    if a.skip_dry_run:
        m = json.load(open(os.path.join(E.RESULTS, 'exp04', a.skip_dry_run, 'run_meta.json')))
        assert m.get('dry_run'), f'{a.skip_dry_run} is not a dry run'
        out.append({'slot': 'dry_reused', 'run_id': a.skip_dry_run, 'status': m.get('status'), 'regime': m.get('regime'),
                    'cap_lifts_within_off_all': (m.get('verify') or {}).get('cap_lifts_within_off_all'),
                    'lease_s': (m.get('lease') or {}).get('lease_s'), 'git_commit': m.get('git_commit')})
        log(a.batch, event='dry_run_reused', **out[-1])
    else:
        m = slot(a.batch, 'dry7', 10, 26, True, deadline, out)
    if not (m and m.get('status') == 'complete' and (m.get('verify') or {}).get('cap_lifts_within_off_all')):
        verdict = 'dry_run_failed'
    else:
        for i, (cmin, seed) in enumerate(SLOTS, 1):
            m = slot(a.batch, f's{i}-{cmin}m', cmin, seed, False, deadline, out)
            if out and out[-1]['status'] in ('no_run_meta', 'running', None):
                verdict = 'stopped_run_without_meta'
                break
            if time.time() > deadline:
                verdict = 'deadline'
                break
        verdict = verdict or 'complete'
    E.stop_traffic()
    try:
        asyncio.run(idle())
    except Exception as exc:  # leave a note; the stack is still left without traffic
        log(a.batch, event='idle_xapp_stop_failed', error=str(exc))
    O.unload_all()
    use = [os.path.join(E.RESULTS, 'exp04', r['run_id']) for r in out
           if not r['slot'].startswith('dry') and r['status'] == 'complete' and (r['regime'] or {}).get('in_regime')]
    if use:
        subprocess.run([PY, os.path.join(R1, 'scripts', 'timing_report.py'), *use, '--out',
                        os.path.join(E.RESULTS, 'exp04', f'{a.batch}-timing_summary.json')], cwd=R1)
    json.dump({'batch': a.batch, 'verdict': verdict, 'deadline': a.deadline, 'runs': out, 'timing_runs': use,
               'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z')},
              open(os.path.join(E.RESULTS, 'exp04', f'{a.batch}-batch.json'), 'w'), indent=1)
    log(a.batch, event='batch_end', verdict=verdict, n_runs=len(out), timing_runs=len(use))


if __name__ == '__main__':
    main()
