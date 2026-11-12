"""GPU0 residency check for end-to-end runs (Exp-04/08): orchestrator + judge resident together.

usage: gpu0_residency.py [--judge llama3.3:70b] [--orchestrators llama3.1:8b,qwen2.5-coder:7b,gemma4:26b-a4b]
For each orchestrator: unload all → load judge → load orchestrator → /api/ps must list both fully on GPU
(size_vram == size) → nvidia-smi GPU0 memory → first call after load (slow: 70B ~32 s once) → warm call (timed).
Gate time is measured on warm models only: after load, one warm-up call per model. Unloads everything at the end
(Exp-06 runs one model at a time). Writes r1/env/gpu0_residency.json.
Exp-06 = residency 'single' (one model at a time); Exp-04/08 = 'pair' (orchestrator + judge, both warm).
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ollama_env as O  # noqa: E402

R1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gpu0_used_mib():  # name kept (output keys); measures the GPU Ollama uses (ollama_env.ollama_gpu)
    g = O.gpu_state()
    return g['mem_used_mib'], g['mem_total_mib']


def load(model):
    t0 = time.time()
    O.api('/api/generate', {'model': model, 'prompt': '', 'stream': False, 'keep_alive': -1})
    return round(time.time() - t0, 1)


def warm_call(model):
    t0 = time.time()
    r = O.api('/api/generate', {'model': model, 'prompt': 'Reply with the JSON {"ok": true}.', 'format': 'json',
                                'stream': False, 'keep_alive': -1, 'options': {'seed': 42, 'temperature': 0}})
    return {'wall_s': round(time.time() - t0, 2), 'load_ms': round(r.get('load_duration', 0) / 1e6, 1),
            'total_ms': round(r.get('total_duration', 0) / 1e6, 1)}


def main():
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    judge = arg('--judge', 'llama3.3:70b')
    orch = arg('--orchestrators', 'llama3.1:8b,qwen2.5-coder:7b,gemma4:26b-a4b').split(',')
    results = []
    for m in orch:
        O.unload_all()
        base, total = gpu0_used_mib()
        lj, lo = load(judge), load(m)
        ps = {x['name']: x for x in O.api('/api/ps').get('models', [])}
        used, _ = gpu0_used_mib()
        both = all(n in ps and ps[n]['size_vram'] == ps[n]['size'] for n in (judge, m))
        r = {'orchestrator': m, 'judge': judge, 'both_resident_full_gpu': both,
             'vram_gib': {n: round(ps[n]['size_vram'] / 2**30, 1) for n in ps},
             'context': {n: ps[n].get('context_length') for n in ps},
             'gpu0_used_mib': used, 'gpu0_total_mib': total, 'gpu0_before_mib': base,
             'cold_load_s': {judge: lj, m: lo},
             'first_call_after_load': {judge: warm_call(judge), m: warm_call(m)} if both else None,
             'warm_call': {judge: warm_call(judge), m: warm_call(m)} if both else None}
        r['warm_ok'] = bool(both and all(v['load_ms'] < 1000 for v in r['warm_call'].values()))
        print(json.dumps(r), flush=True)
        results.append(r)
    O.unload_all()
    out = {'date': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'ollama': O.api('/api/version').get('version'),
           'settings': O.container_settings(), 'gpu': O.ollama_gpu(), 'results': results}
    json.dump(out, open(os.path.join(R1, 'env', 'gpu0_residency.json'), 'w'), indent=1)
    sys.exit(0 if all(r['both_resident_full_gpu'] and r['warm_ok'] for r in results) else 1)


if __name__ == '__main__':
    main()
