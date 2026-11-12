"""Exp-08: verification gate. (a) fault injection over the references; (b) the ten vague intents through the full
agent + gate. Residency "pair": orchestrator llama3.1:8b and judge llama3.3:70b both resident and warm.

usage: exp08_gate.py --run-id <id> [--seeds 1,2,3] [--readbacks extract,sentence] [--modes full,sparse]
(a) zto/gate_inject.py: I1–I3 × variants × seeds × read-backs → exp08_gate_results.csv (+ _summary.json with the
    corrupted-caught / correct-passed / false-HOLD rates and per-check attribution).
(b) every vague intent (zto/intents/vague.yaml) × spec mode × seed: agent (llama3.1:8b) → gate (each read-back) →
    exp08_vague_intents.csv: agent validity, gate final, caught_by, expected_outcome, correct (final == expected;
    an expected PASS also needs the spec to equal expected_spec resolved with defaults.json).
Outputs in results/exp08/<run_id>/ with exp08_summary.json (provenance, prompt hashes, residency check).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

R1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R1, 'scripts'), os.path.join(R1, 'zto'), R1]
import r1env as E  # noqa: E402

START_COMMIT = E.git('rev-parse', 'HEAD')  # the commit the run started from (no commits during official runs)
import agent  # noqa: E402
import gate  # noqa: E402
import ollama_env as O  # noqa: E402
import resolve  # noqa: E402
import yaml  # noqa: E402

ORCH, JUDGE = 'llama3.1:8b', 'llama3.3:70b'


def pair_residency() -> dict:
    O.unload_all()
    for m in (JUDGE, ORCH):  # load + one discarded warm-up call each (first call after load is slow)
        agent.call(m, [{'role': 'user', 'content': 'Reply with {}.'}], 0, None, {'type': 'object'})
        agent.call(m, [{'role': 'user', 'content': 'Reply with {}.'}], 0, None, {'type': 'object'})
    ps = {x['name']: x for x in O.api('/api/ps').get('models', [])}
    return {'mode': 'pair', 'orchestrator': ORCH, 'judge': JUDGE,
            'both_resident': all(m in ps and ps[m]['size_vram'] == ps[m]['size'] for m in (ORCH, JUDGE))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--seeds', default='1,2,3')
    ap.add_argument('--readbacks', default='extract,sentence')
    ap.add_argument('--modes', default='full,sparse')
    ap.add_argument('--final-mode', default=None, help="Exp-02's chosen spec mode (full|sparse): the paper's final configuration")
    a = ap.parse_args()
    out = os.path.join(E.RESULTS, 'exp08', a.run_id)
    os.makedirs(out, exist_ok=False)
    GPU = O.ollama_gpu()
    busy = O.gpu_quiet(gpu=GPU)
    if busy:
        raise SystemExit(f"GPU{GPU['index']} (Ollama) not quiet: " + '; '.join(busy))
    O.start_gpu_watch(out, GPU)  # aborts (gpu_abort.json) if the GPU is occupied mid-run
    res = pair_residency()
    print('residency', res, flush=True)
    if not res['both_resident']:
        raise SystemExit('pair residency not established')
    p = subprocess.run([sys.executable, os.path.join(R1, 'zto', 'gate_inject.py'), '--seeds', a.seeds,
                        '--readbacks', a.readbacks, '--out', os.path.join(out, 'exp08_gate_results.csv')],
                       capture_output=True, text=True)
    open(os.path.join(out, 'gate_inject.log'), 'w').write(p.stdout + p.stderr)
    inject = json.load(open(os.path.join(out, 'exp08_gate_results_summary.json'))) if p.returncode == 0 else {'error': p.stderr[-800:]}
    vague = yaml.safe_load(open(os.path.join(R1, 'zto', 'intents', 'vague.yaml')))['vague']
    rows = []
    for v in vague:
        want = resolve.resolve(v['expected_spec']) if v['expected_outcome'] == 'PASS' else None
        for mode in a.modes.split(','):
            for seed in map(int, a.seeds.split(',')):
                r = agent.run(v['text'], model=ORCH, seed=seed, mode=mode)
                for rb in a.readbacks.split(','):
                    g = gate.run(v['text'], r['raw_spec'] if r['raw_spec'] is not None else {}, judge_model=JUDGE, seed=seed,
                                 readback=rb)
                    spec_ok = want is None or (r['valid'] and resolve.resolve(r['spec']) == want)
                    rows.append({'id': v['id'], 'mode': mode, 'seed': seed, 'readback_variant': rb, 'agent_valid': r['valid'],
                                 'final': g['final'], 'caught_by': ';'.join(g['caught_by']), 'expected': v['expected_outcome'],
                                 'correct': g['final'] == v['expected_outcome'] and spec_ok,
                                 'spec_json': json.dumps(r['raw_spec'], sort_keys=True, separators=(',', ':')),
                                 'agent_wall_ms': round(sum(c['wall_s'] for c in r['calls']) * 1000, 1),
                                 'gate_deterministic_s': g['gate_deterministic_s'], 'gate_readback_s': g['gate_readback_s']})
                    print(json.dumps({k: rows[-1][k] for k in ('id', 'mode', 'seed', 'readback_variant', 'final',
                                                                 'expected', 'correct')}), flush=True)
    with open(os.path.join(out, 'exp08_vague_intents.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    vs = {}
    for rb in a.readbacks.split(','):
        for mode in a.modes.split(','):
            rr = [r for r in rows if r['readback_variant'] == rb and r['mode'] == mode]
            vs[f'{rb}|{mode}'] = {'n': len(rr), 'correct': sum(r['correct'] for r in rr),
                                  'misinterpretation': sum(r['final'] == 'PASS' and r['expected'] != 'PASS' for r in rr)}
    final = {}  # paper: final configuration (chosen mode, extraction read-back), majority verdict over the seeds
    if a.final_mode:
        for v in vague:
            rr = [r for r in rows if r['id'] == v['id'] and r['mode'] == a.final_mode and r['readback_variant'] == 'extract']
            votes = {}
            for r in rr:
                votes[r['final']] = votes.get(r['final'], 0) + 1
            maj = max(votes, key=votes.get) if votes else None
            final[v['id']] = {'majority': maj, 'votes': votes, 'expected': v['expected_outcome'],
                              'matched': maj == v['expected_outcome'] and all(r['correct'] for r in rr if r['final'] == maj)}
    gi = list(csv.DictReader(open(os.path.join(out, 'exp08_gate_results.csv')))) if p.returncode == 0 else []
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    inrange = ('reserve_to_50', 'threshold_to_72', 'min_duration_to_8', 'horizon_changed')
    sent = [r for r in gi if r['readback_variant'] == 'sentence' and r['variant'] in inrange]
    gate_stats = {'gate_deterministic_ms_median': med([float(r['gate_deterministic_s']) * 1000 for r in gi]),
                  'gate_readback_s_median_extract': med([float(r['gate_readback_s']) for r in gi if r['readback_variant'] == 'extract']),
                  'sentence_inrange_miss_pct': round(100 * sum(r['readback'] == 'PASS' for r in sent) / len(sent), 1) if sent else None,
                  'sentence_inrange_variants': list(inrange)}
    O.unload_all()
    lock = json.load(open(os.path.join(R1, 'zto', 'prompts', 'lock.json')))
    json.dump({'run_id': a.run_id, 'git_commit': START_COMMIT,
               'provenance': E.provenance(__file__, extra=[os.path.join(R1, 'zto', f) for f in
                                                           ('agent.py', 'gate.py', 'gate_inject.py', 'resolve.py', 'defaults.json',
                                                            'spec.schema.json', 'actuatable.json', 'field_classes.json',
                                                            'gate_registry.json', 'intents/vague.yaml')]),
               'prompts': lock, 'residency': res, 'think': False, 'args': vars(a),
               'versions': json.load(open(os.path.join(R1, 'env', 'versions.json'))),
               'fault_injection': inject, 'gate_stats': gate_stats, 'vague': vs,
               'vague_final': {'mode': a.final_mode, 'readback': 'extract', 'per_intent': final,
                               'matched': sum(x['matched'] for x in final.values()), 'of': len(final)}, 'not_applicable': ['action_swapped (one SLICE catalogue action)'],
               'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z')},
              open(os.path.join(out, 'exp08_summary.json'), 'w'), indent=1)
    print(json.dumps({'fault_injection': {k: v for k, v in inject.items() if k != 'caught_by'}, 'vague': vs}, indent=1))
    print('files:', *sorted(os.listdir(out)))


if __name__ == '__main__':
    main()
