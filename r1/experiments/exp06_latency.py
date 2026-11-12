"""Exp-06: latency. Residency "single" (one model resident at a time), think=false, the Ollama GPU (ollama_env.ollama_gpu) quiet.

usage: exp06_latency.py --run-id <id> --exp01-config <results/exp01/<run>/ts<T>_h<H>> [--warm 20] [--cold 5]
                        [--n-classify 100] [--models ...]
(a) orchestration latency, full spec: per LLM, `cold` calls with the model unloaded before each (load included) and
    `warm` calls with it resident (intent I1, full mode, seeds 1..N) → exp06_orchestration.csv (one row per call:
    model, cold, wall_ms, total/load/prompt_eval/eval durations, prompt/completion tokens, schema_valid).
(b) like-for-like classification: the same `n-classify` telemetry windows (seeded sample of the Exp-01 main
    test.csv) classified by each LLM (JSON {"congested": 0|1}, warm) and by the Exp-01 main ONNX model (ORT CPU,
    single row) → exp06_like_for_like.json (latency p50/p99 and accuracy vs the label, per predictor).
(c) model sizes (Ollama blob size per tag, ONNX size), env/hardware.json (scripts/hw_info.sh), env/versions.json.
Outputs in results/exp06/<run_id>/ with exp06_summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time

R1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R1, 'scripts'), os.path.join(R1, 'zto'), R1]
import r1env as E  # noqa: E402

START_COMMIT = E.git('rev-parse', 'HEAD')  # the commit the run started from (no commits during official runs)
import agent  # noqa: E402
import ollama_env as O  # noqa: E402

CLASSIFY_SCHEMA = {'type': 'object', 'required': ['congested'], 'additionalProperties': False,
                   'properties': {'congested': {'enum': [0, 1]}}}


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))] if v else None


def orchestration(model, intent, cold, warm):
    rows = []
    for i in range(cold + warm):
        is_cold = i < cold
        if is_cold:
            O.unload_all()
        r = agent.run(intent, model=model, seed=i + 1, mode='full')
        c = r['calls'][0]
        rows.append({'model': model, 'call': i + 1, 'cold': is_cold, 'wall_ms': round(c['wall_s'] * 1000, 1),
                     'total_duration_ms': round(c['total_duration_ms'], 1), 'load_duration_ms': round(c['load_duration_ms'], 1),
                     'prompt_eval_duration_ms': round(c['prompt_eval_duration_ms'], 1),
                     'eval_duration_ms': round(c['eval_duration_ms'], 1), 'prompt_tokens': c['prompt_tokens'],
                     'completion_tokens': c['completion_tokens'], 'attempts': r['attempts'], 'schema_valid': r['valid']})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def classify_llm(model, windows):
    lat, pred = [], []
    for w in windows:
        feats = ', '.join(f'{k}={float(w[k]):.3f}' for k in ('mean_prb', 'std_prb', 'min_prb', 'max_prb', 'slope_prb', 'last_prb'))
        msg = [{'role': 'user', 'content': f'Cell PRB utilization over the last window: {feats} (%, slope in %/s). '
                                           'Will the cell be congested at the prediction horizon? Reply JSON {"congested": 0 or 1}.'}]
        r, wall = agent.call(model, msg, 42, 0, CLASSIFY_SCHEMA)
        lat.append(wall * 1000)
        try:
            pred.append(int(json.loads(r['message']['content'])['congested']))
        except (ValueError, KeyError, TypeError):
            pred.append(-1)
    y = [int(w['label']) for w in windows]
    return {'latency_ms_p50': pct(lat, 0.5), 'latency_ms_p99': pct(lat, 0.99), 'n': len(lat),
            'accuracy': sum(p == t for p, t in zip(pred, y)) / len(y), 'invalid_replies': pred.count(-1)}


def classify_onnx(cfg_dir, windows):
    sys.path.insert(0, os.path.join(R1, 'mlengine'))
    import numpy as np
    import onnxruntime as ort
    import neuralsmith_backend as nsb
    pre = json.load(open(os.path.join(cfg_dir, 'neuralsmith', 'preprocess.json')))
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    sess = ort.InferenceSession(os.path.join(cfg_dir, 'neuralsmith', 'model.onnx'), so, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    X = nsb.apply_preprocess(pre, np.array([[float(w[f]) for f in pre['features']] for w in windows]))
    lat, pred = [], []
    for i in range(len(X)):
        t0 = time.perf_counter_ns()
        out = sess.run(None, {name: X[i:i + 1]})
        lat.append((time.perf_counter_ns() - t0) / 1e6)
        pred.append(int(np.asarray(out[0]).ravel()[0]))
    y = [int(w['label']) for w in windows]
    return {'latency_ms_p50': pct(lat, 0.5), 'latency_ms_p99': pct(lat, 0.99), 'n': len(lat),
            'accuracy': sum(p == t for p, t in zip(pred, y)) / len(y),
            'size_kb': os.path.getsize(os.path.join(cfg_dir, 'neuralsmith', 'model.onnx')) / 1024,
            'note': 'single-row ORT CPU calls on the same windows (the 1000-call p50/p99 is in the Exp-01 report.json)'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--exp01-config', required=True, help='results/exp01/<run>/ts<T>_h<H> (the main configuration)')
    ap.add_argument('--models', default=','.join(O.MODELS))
    ap.add_argument('--warm', type=int, default=20)
    ap.add_argument('--cold', type=int, default=5)
    ap.add_argument('--n-classify', type=int, default=100)
    a = ap.parse_args()
    out = os.path.join(E.RESULTS, 'exp06', a.run_id)
    os.makedirs(out, exist_ok=False)
    GPU = O.ollama_gpu()
    busy = O.gpu_quiet(gpu=GPU)
    if busy:
        raise SystemExit(f"GPU{GPU['index']} (Ollama) not quiet: " + '; '.join(busy))
    O.start_gpu_watch(out, GPU)  # aborts (gpu_abort.json) if the GPU is occupied mid-run
    i1 = json.load(open(os.path.join(R1, 'zto', 'intents', 'reference', 'I1.json')))['_intent_text']
    test = list(csv.DictReader(open(os.path.join(a.exp01_config, 'test.csv'))))
    windows = random.Random(42).sample(test, min(a.n_classify, len(test)))
    orch, like = [], {}
    for m in a.models.split(','):
        orch += orchestration(m, i1, a.cold, a.warm)
        like[m] = classify_llm(m, windows)  # model is resident and warm after the orchestration calls
        ps = next((x for x in O.api('/api/ps').get('models', []) if x['name'] == m), {})
        used = str(O.gpu_state(GPU)['mem_used_mib'])
        like[m]['vram_gb_ollama'] = round(ps.get('size_vram', 0) / 1e9, 2)  # /api/ps (under-reports gemma4)
        like[m]['gpu_used_gb_nvidia_smi'] = round(int(used) / 1024, 2) if used.isdigit() else None
        print(m, like[m], flush=True)
        O.unload_all()
    like['onnx_neuralsmith'] = classify_onnx(a.exp01_config, windows)
    with open(os.path.join(out, 'exp06_orchestration.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(orch[0]))
        w.writeheader()
        w.writerows(orch)
    json.dump({'windows': len(windows), 'source': os.path.relpath(a.exp01_config, E.R1), 'seed': 42, **like},
              open(os.path.join(out, 'exp06_like_for_like.json'), 'w'), indent=1)
    subprocess.run(['bash', os.path.join(R1, 'scripts', 'hw_info.sh')], check=True)
    sizes = {m['name']: round(m['size'] / 1e9, 2) for m in O.api('/api/tags').get('models', []) if m['name'] in O.MODELS}
    per = {}
    for m in a.models.split(','):
        for cold in (True, False):
            v = [r['wall_ms'] for r in orch if r['model'] == m and r['cold'] == cold]
            per[f"{m}|{'cold' if cold else 'warm'}"] = {'n': len(v), 'wall_ms_median': pct(v, 0.5),
                                                       'wall_ms_iqr': [pct(v, 0.25), pct(v, 0.75)]}
    json.dump({'run_id': a.run_id, 'git_commit': START_COMMIT, 'provenance': E.provenance(__file__),
               'prompts': json.load(open(os.path.join(R1, 'zto', 'prompts', 'lock.json'))), 'residency': 'single',
               'think': False, 'args': vars(a), 'model_size_gb': sizes, 'orchestration': per, 'like_for_like': like,
               'vram_gb': {m: like[m]['gpu_used_gb_nvidia_smi'] for m in a.models.split(',')},
               'completion_tokens_median': {m: pct([r['completion_tokens'] for r in orch if r['model'] == m and not r['cold']], 0.5)
                                            for m in a.models.split(',')},
               'hardware': json.load(open(os.path.join(R1, 'env', 'hardware.json'))),
               'versions': json.load(open(os.path.join(R1, 'env', 'versions.json'))),
               'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z')},
              open(os.path.join(out, 'exp06_summary.json'), 'w'), indent=1)
    print(json.dumps(per, indent=1))
    print('files:', *sorted(os.listdir(out)))


if __name__ == '__main__':
    main()
