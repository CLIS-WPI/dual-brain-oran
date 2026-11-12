"""Timing report for Exp-04 (and any run with timing.jsonl): the six quantities, median and IQR over runs.

usage: timing_report.py <run_dir> [<run_dir> ...] [--out timing_summary.json] [--group-by collection_min]
Computed from timing marks only (r1/common/timing.py), never from logs:
  orchestration latency   intent_received → spec_ready  (+ template_rendered − model_received, the rendering part)
  training time           train_post → model_received
  gate time               gate_start → gate_end   (also: deterministic checks / LLM read-back split, from marks)
  render + gate           render_start → render_end  +  gate_start → gate_end   (Table II "render+gate")
  registration            register_start → register_end
  end-to-end provisioning intent_received → first_e2_control_sent (emitted by the running xApp)
  inference latency       not from marks: the ML engine report (onnx_latency_ms_p50/p99 of 1000 ONNX calls),
                          read from <run_dir>/model/report.json
Per-LLM-call log (<run_dir>/agent_calls.jsonl): model, prompt/completion tokens, warm flag (load < 1 s); per model
the first 5 calls are cold by definition and reported separately.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import timing  # noqa: E402

QUANTITIES = {
    'orchestration_s': [('intent_received', 'spec_ready'), ('model_received', 'template_rendered')],
    'training_s': [('train_post', 'model_received')],
    'gate_s': [('gate_start', 'gate_end')],
    'gate_deterministic_s': [('gate_actuatable_start', 'gate_conflict_end')],
    'gate_readback_s': [('gate_readback_start', 'gate_readback_end')],
    'gate_model_parity_s': [('gate_model_parity_start', 'gate_model_parity_end')],  # after training, before registration
    'render_gate_s': [('render_start', 'render_end'), ('gate_start', 'gate_end')],  # Table II "render+gate"
    'registration_s': [('register_start', 'register_end')],
    'end_to_end_s': [('intent_received', 'first_e2_control_sent')],
}


def run_quantities(run_dir: str) -> dict:
    marks = timing.load(run_dir)
    out = {}
    for q, spans in QUANTITIES.items():
        parts = [timing.duration_s(marks, a, b) for a, b in spans]
        out[q] = round(sum(parts), 3) if all(p is not None for p in parts) else None
    rep = os.path.join(run_dir, 'model', 'report.json')
    if os.path.exists(rep):
        r = json.load(open(rep))
        out['inference_ms_p50'], out['inference_ms_p99'] = r.get('onnx_latency_ms_p50'), r.get('onnx_latency_ms_p99')
    meta = os.path.join(run_dir, 'run_meta.json')
    out['meta'] = json.load(open(meta)) if os.path.exists(meta) else {}
    return out


def stats(v: list[float]) -> dict | None:
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    q = lambda p: v[min(len(v) - 1, max(0, round(p * (len(v) - 1))))]
    return {'n': len(v), 'median': round(st.median(v), 3), 'iqr': [q(0.25), q(0.75)], 'min': v[0], 'max': v[-1]}


def llm_calls(run_dirs: list[str]) -> dict:
    per = {}
    for d in run_dirs:
        p = os.path.join(d, 'agent_calls.jsonl')
        for line in open(p) if os.path.exists(p) else []:
            r = json.loads(line)
            for c in r.get('calls', []):
                per.setdefault(r['model'], []).append(c)
    out = {}
    for m, calls in per.items():
        cold, warm = calls[:5], calls[5:]
        out[m] = {'n_calls': len(calls), 'cold_first5_wall_s': stats([c['wall_s'] for c in cold]),
                  'warm_wall_s': stats([c['wall_s'] for c in warm]),
                  'prompt_tokens': stats([c['prompt_tokens'] for c in calls]),
                  'completion_tokens': stats([c['completion_tokens'] for c in calls])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dirs', nargs='+')
    ap.add_argument('--out', default='timing_summary.json')
    ap.add_argument('--group-by', default='collection_min')
    a = ap.parse_args()
    runs = {d: run_quantities(d) for d in a.run_dirs}
    groups = {}
    for d, q in runs.items():
        groups.setdefault(str(q['meta'].get(a.group_by, 'all')), []).append(q)
    summary = {'runs': {os.path.basename(d.rstrip('/')): {k: v for k, v in q.items() if k != 'meta'} for d, q in runs.items()},
               'groups': {g: {k: stats([q.get(k) for q in qs]) for k in [*QUANTITIES, 'inference_ms_p50', 'inference_ms_p99']}
                          for g, qs in groups.items()},
               'llm_calls': llm_calls(a.run_dirs), 'group_by': a.group_by}
    json.dump(summary, open(a.out, 'w'), indent=1)
    for g, s in summary['groups'].items():
        print(f'\n{a.group_by} = {g}')
        print(f"{'quantity':24} {'n':>2} {'median':>9} {'IQR':>22}")
        for k, v in s.items():
            if v:
                print(f"{k:24} {v['n']:>2} {v['median']:>9} {str(v['iqr']):>22}")
            else:
                print(f'{k:24}  - (no marks)')
    print('\nwrote', a.out)


if __name__ == '__main__':
    main()
