"""Exp-02: spec consistency. 3 intents × 4 LLMs × 5 seeds × 2 spec modes (full, sparse) → scored against the
reference specs (no LLM judge).

usage: exp02_consistency.py --run-id <id> [--models llama3.1:8b,qwen2.5-coder:7b,gemma4:26b-a4b,llama3.3:70b]
                            [--intents I1,I2,I3] [--seeds 1,2,3,4,5] [--modes full,sparse]
Residency "single": one model resident at a time (unload_all, then one discarded warm-up call per model). Every
call: think=false, Ollama default temperature with the given seed, no token cap, frozen prompts (agent.py refuses a
prompt that differs from zto/prompts/lock.json). Scoring per call:
  schema_valid       the spec validates (sparse: after resolve.py fills the defaults)
  slots_correct      the 6 template slots (scripts/list_slots.py) equal the resolved reference
  nonslots_correct   every other spec field equals the reference (horizon_ms, collection_min, model.*,
                     label_rule.feature/op, slice_action.action/target)
  reference_correct  both
Per (model, intent, mode): identical = the 5 seeds produced the same spec. Outputs in results/exp02/<run_id>/:
exp02_runs.csv (one row per call), exp02_summary.json (counts per model × mode, prompt hashes, provenance),
slots.json (list_slots.py --json).
--allow-gpu0-coresident (2026-11-04, Milad): run although another process computes on the Ollama GPU; every snapshot
of the co-residents (pid, command, GPU memory) and that GPU's utilization/memory goes into exp02_summary.json
gpu_coresidency (part a, run on GPU0: key gpu0_coresidency). The guard checks the GPU Ollama uses (ollama_env.ollama_gpu).
Always: after the warm-up each model must be fully resident in VRAM (/api/ps size_vram == size), else abort (no
partial CPU offload).
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
import ollama_env as O  # noqa: E402
import render  # noqa: E402
import resolve  # noqa: E402


def flat(d, p=''):
    out = {}
    for k, v in (d or {}).items():
        out.update(flat(v, f'{p}{k}.') if isinstance(v, dict) else {f'{p}{k}': v})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--models', default=','.join(O.MODELS))
    ap.add_argument('--intents', default='I1,I2,I3')
    ap.add_argument('--seeds', default='1,2,3,4,5')
    ap.add_argument('--modes', default='full,sparse')
    ap.add_argument('--allow-gpu0-coresident', action='store_true')
    a = ap.parse_args()
    out = os.path.join(E.RESULTS, 'exp02', a.run_id)
    os.makedirs(out, exist_ok=False)
    slots_js = json.loads(subprocess.run([sys.executable, os.path.join(R1, 'scripts', 'list_slots.py'), '--json'],
                                         capture_output=True, text=True).stdout)
    json.dump(slots_js, open(os.path.join(out, 'slots.json'), 'w'), indent=1)
    slots = [s['slot'] for s in slots_js['slots']]
    refs = {}
    for i in a.intents.split(','):
        r = json.load(open(os.path.join(R1, 'zto', 'intents', 'reference', f'{i}.json')))
        refs[i] = (r['_intent_text'], resolve.validated(resolve.resolve(r)))  # refuses an invalid reference
    cores, GPU = [], O.ollama_gpu()

    def snap(when):
        cores.append({'when': when, 't': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'busy_sm': O.gpu_quiet(gpu=GPU),
                      'coresident': O.gpu_coresident(GPU), **O.gpu_state(GPU)})
        return cores[-1]

    busy = snap('start')['busy_sm']
    if busy and not a.allow_gpu0_coresident:
        raise SystemExit(f"GPU{GPU['index']} (Ollama) not quiet: " + '; '.join(busy))
    if not a.allow_gpu0_coresident:
        O.start_gpu_watch(out, GPU)  # aborts (gpu_abort.json) if the GPU is occupied mid-run
    rows = []
    for model in a.models.split(','):
        O.unload_all()  # residency "single"
        warm = agent.call(model, [{'role': 'user', 'content': 'Reply with {}.'}], 0, None, {'type': 'object'})
        print(f'{model}: warm-up {warm[1]:.1f} s (discarded)', flush=True)
        ps = [m for m in O.api('/api/ps').get('models', []) if m['name'] == model]
        if not ps or ps[0].get('size_vram') != ps[0].get('size'):
            raise SystemExit(f'{model} not fully in VRAM (partial CPU offload): {ps}')
        snap(f'{model} loaded')['ollama_ps'] = ps[0]
        for intent_id, (text, ref) in refs.items():
            want = flat(ref)
            for mode in a.modes.split(','):
                for seed in map(int, a.seeds.split(',')):
                    r = agent.run(text, model=model, seed=seed, mode=mode)
                    got = flat(r['spec']) if r['valid'] else {}
                    try:
                        render_ok = bool(r['valid']) and bool(render.render(r['spec'], protected_ue_ref=0))
                    except Exception:
                        render_ok = False
                    slot_ok = bool(got) and all(got.get(f) == want.get(f) for f in slots)
                    non_ok = bool(got) and all(got.get(f) == want[f] for f in want if f not in slots)
                    c = r['calls'][-1]
                    rows.append({'model': model, 'intent': intent_id, 'mode': mode, 'seed': seed, 'schema_valid': r['valid'],
                                 'rendered_ok': render_ok,
                                 'attempts': r['attempts'], 'slots_correct': slot_ok, 'nonslots_correct': non_ok,
                                 'reference_correct': slot_ok and non_ok,
                                 'wrong_fields': ';'.join(f for f in want if got.get(f) != want[f]) if got else 'invalid',
                                 'prompt_tokens': sum(x['prompt_tokens'] or 0 for x in r['calls']),
                                 'completion_tokens': sum(x['completion_tokens'] or 0 for x in r['calls']),
                                 'wall_ms': round(sum(x['wall_s'] for x in r['calls']) * 1000, 1),
                                 'total_duration_ms': round(sum(x['total_duration_ms'] for x in r['calls']), 1),
                                 'load_duration_ms': round(c['load_duration_ms'], 1),
                                 'spec_json': json.dumps(r['raw_spec'], sort_keys=True, separators=(',', ':')),
                                 'prompt_sha256': r['prompt_sha256']})
                    print(json.dumps({k: rows[-1][k] for k in ('model', 'intent', 'mode', 'seed', 'schema_valid',
                                                                 'reference_correct', 'wrong_fields', 'wall_ms')}), flush=True)
        snap(f'{model} done')
    snap('end')
    O.unload_all()
    with open(os.path.join(out, 'exp02_runs.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for model in a.models.split(','):
        for mode in a.modes.split(','):
            rr = [r for r in rows if r['model'] == model and r['mode'] == mode]
            groups = {}
            for r in rr:
                groups.setdefault(r['intent'], set()).add(r['spec_json'])
            summary[f'{model}|{mode}'] = {
                'calls': len(rr), 'schema_valid': sum(r['schema_valid'] for r in rr),
                'slots_correct': sum(r['slots_correct'] for r in rr), 'nonslots_correct': sum(r['nonslots_correct'] for r in rr),
                'reference_correct': sum(r['reference_correct'] for r in rr),
                'identical_intents': sum(len(v) == 1 for v in groups.values()), 'intents': len(groups)}
    per_mode = {}
    for mode in a.modes.split(','):
        rr = [r for r in rows if r['mode'] == mode]
        per_mode[mode] = {'calls': len(rr), 'reference_correct_pct': round(100 * sum(r['reference_correct'] for r in rr) / len(rr), 1),
                          'schema_valid_pct': round(100 * sum(r['schema_valid'] for r in rr) / len(rr), 1),
                          'rendered_ok': sum(r['rendered_ok'] for r in rr)}
    lock = json.load(open(os.path.join(R1, 'zto', 'prompts', 'lock.json')))
    json.dump({'run_id': a.run_id, 'git_commit': START_COMMIT,
               'provenance': E.provenance(__file__, extra=[os.path.join(R1, 'zto', f) for f in
                                                           ('agent.py', 'resolve.py', 'defaults.json', 'spec.schema.json')]),
               'prompts': lock, 'residency': 'single', 'think': False, 'temperature': 'ollama default', 'args': vars(a),
               'versions': json.load(open(os.path.join(R1, 'env', 'versions.json'))), 'per_model_mode': summary,
               'per_mode': per_mode, 'rendered_ok_total': sum(r['rendered_ok'] for r in rows), 'calls_total': len(rows),
               'ollama_gpu': GPU, 'gpu_coresidency': cores, 'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z')},
              open(os.path.join(out, 'exp02_summary.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    print('files:', *sorted(os.listdir(out)))


if __name__ == '__main__':
    main()
