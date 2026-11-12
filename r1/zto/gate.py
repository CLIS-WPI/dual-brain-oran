"""Verification gate between template rendering and FlexRIC deployment → PASS / HOLD / FAIL.

usage: gate.py --intent "<text>" --spec spec.json [--readback extract|sentence] [--judge llama3.3:70b]
               [--registry gate_registry.json] [--seed 42] [--run-id ID --run-dir DIR]
"Expressive extraction, restrictive actuation": specs may state what the operator asked even if the testbed
cannot do it; check 0 refuses to deploy anything outside zto/actuatable.json. Every check always runs and records
its verdict; the final verdict is the most severe (FAIL > HOLD > PASS); `caught_by` lists every non-PASS check.
Deterministic checks first, then the LLM read-back:
  0. actuatable (implemented, deterministic): label_rule.op, slice_action.target, .action, label_rule.feature must
     be in zto/actuatable.json → otherwise FAIL "not actuatable" naming the field. target "unspecified" → HOLD
     "beneficiary not stated" (target has no default).
  1. schema + range (implemented): validate against spec.schema.json, render the xApp config, check every rendered
     slot carries the spec's value. Violation → FAIL.
  2. engineering default (implemented, deterministic): engineering-level fields (field_classes.json) must equal
     zto/defaults.json → otherwise HOLD "engineering field changed" listing them (sparse mode: by construction).
  3. conflict (implemented, static): the candidate's resources vs every deployed xApp in gate_registry.json;
     overlap → HOLD naming the other xApp.
  R. intent read-back (implemented, LLM, two variants; Exp-08 reports both):
     extract  (default) the judge (llama3.3:70b, warm, "pair" residency) extracts the operator-level fields from
              the intent text only (agent.py --mode judge, never sees the orchestrator's spec). Judge target
              "unspecified" → HOLD "beneficiary not stated". Over operator-level fields: an extracted field that
              differs = stated mismatch; a non-extracted field whose spec value is not the default = unsupported
              deviation. Either → HOLD listing the fields.
     sentence fixed-template English sentence of the deployable behaviour (no LLM); the judge answers JSON
              {match, reason} to "Does this sentence fulfil the operator's intent?" → match=false = HOLD.
  4. dry-run (stub): logs "not implemented".
  5. rollback trigger (stub): logs "not implemented"; receives the KPI feedback (UE3 throughput) so it has data.
Timing marks gate_start/end and per check; gate_deterministic_s (checks 0-3) and gate_readback_s are separate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request

ZTO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ZTO)
sys.path.insert(0, os.path.dirname(ZTO))
import render  # noqa: E402
import resolve  # noqa: E402
from common import timing  # noqa: E402

OLLAMA = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11436')
REGISTRY = os.path.join(ZTO, 'gate_registry.json')
ACTUATABLE = {k: v for k, v in json.load(open(os.path.join(ZTO, 'actuatable.json'))).items() if not k.startswith('_')}
FIELD_CLASSES = json.load(open(os.path.join(ZTO, 'field_classes.json')))
SEVERITY = {'PASS': 0, 'SKIPPED': 0, 'HOLD': 1, 'FAIL': 2}
JUDGE_FORMAT = {'type': 'object', 'required': ['match', 'reason'], 'additionalProperties': False,
                'properties': {'match': {'type': 'boolean'}, 'reason': {'type': 'string'}}}


def _flat(d, p=''):
    out = {}
    for k, v in (d or {}).items():
        out.update(_flat(v, f'{p}{k}.') if isinstance(v, dict) else {f'{p}{k}': v})
    return out


def _eq(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) < 1e-9
    return a == b


def check_actuatable(spec: dict) -> dict:
    """Check 0: deterministic, no LLM. Values the deployed xApp cannot execute → FAIL; unstated beneficiary → HOLD."""
    f = _flat(resolve.resolve(spec)) if isinstance(spec, dict) else {}
    if f.get('slice_action.target') == 'unspecified':
        return {'result': 'HOLD', 'reason': 'beneficiary not stated', 'fields': ['slice_action.target']}
    bad = [{'field': k, 'value': f.get(k), 'actuatable': v} for k, v in ACTUATABLE.items() if f.get(k) not in v]
    return {'result': 'FAIL' if bad else 'PASS', 'reason': 'not actuatable' if bad else None, 'fields': bad}


def check_engineering(spec: dict) -> dict:
    """Check 2: engineering-level fields must carry their defaults.json value (deterministic)."""
    got, want = _flat(resolve.resolve(spec)), _flat(resolve.resolve({}))
    bad = [{'field': f, 'default': want.get(f), 'spec': got.get(f)} for f in FIELD_CLASSES['engineering']
           if not _eq(got.get(f), want.get(f))]
    return {'result': 'HOLD' if bad else 'PASS', 'reason': 'engineering field changed' if bad else None, 'fields': bad}


def check_schema(spec: dict) -> dict:
    """Check 1: schema + range, then the rendered config must carry the spec's values."""
    errs = render.validate(render.with_defaults(spec))
    if errs:
        return {'result': 'FAIL', 'errors': errs}
    s = resolve.resolve(spec)
    cfg = render.render(s, protected_ue_ref=0)
    want = {'kpm_period_ms': s['sampling_ms'], 'threshold_cell_prb': s['label_rule']['threshold_pct'],
            'reserve_pct': s['slice_action']['reserve_pct'], 'trigger_consecutive': s['control']['debounce_n'],
            'window_samples': max(1, s['window_ms'] // s['sampling_ms']),
            'min_duration_samples': max(1, -(-int(s['label_rule']['min_duration_s'] * 1000) // s['sampling_ms']))}
    bad = [f'{k}={cfg[k]!r} != {v!r}' for k, v in want.items() if cfg[k] != v]
    return {'result': 'FAIL' if bad else 'PASS', 'errors': bad}


def readback_sentence(spec: dict) -> str:
    """Fixed template over the deployable behaviour (actuatable values only reach here), no LLM."""
    s = resolve.resolve(spec)
    lr, sa = s['label_rule'], s['slice_action']
    return (f"UE3 (the protected cell-edge user) is served first and guaranteed at least {sa['reserve_pct']}% of the "
            f"downlink PRBs; all other UEs share at most {100 - sa['reserve_pct']}%. This is applied when cell PRB "
            f"utilization at or above {lr['threshold_pct']:g}% for at least {lr['min_duration_s']:g} s is predicted "
            f"{s['horizon_ms'] / 1000:g} s ahead (acting after {s['control']['debounce_n']} consecutive predictions, "
            f"KPM every {s['sampling_ms']} ms), and released when the load is normal.")


def judge(intent: str, sentence: str, model: str, seed: int) -> dict:
    prompt = (f'Operator intent: "{intent}"\nConfigured behaviour: "{sentence}"\n'
              'Does this sentence fulfil the operator\'s intent? Answer JSON {"match": true|false, "reason": "<one sentence>"}.')
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'stream': False, 'think': False,
            'format': JUDGE_FORMAT, 'keep_alive': -1, 'options': {'seed': seed, 'temperature': 0}}
    t0 = time.monotonic()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        f'{OLLAMA}/api/chat', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}),
        timeout=1800).read())
    wall = time.monotonic() - t0
    try:
        v = json.loads(r['message']['content'])
    except ValueError:
        v = {'match': False, 'reason': 'judge reply not JSON'}
    return {'match': bool(v.get('match')), 'reason': v.get('reason'), 'model': model, 'seed': seed,
            'prompt_tokens': r.get('prompt_eval_count'), 'completion_tokens': r.get('eval_count'),
            'total_duration_ms': r.get('total_duration', 0) / 1e6, 'load_duration_ms': r.get('load_duration', 0) / 1e6,
            'wall_s': round(wall, 3), 'warm': r.get('load_duration', 0) / 1e6 < 1000}


def diff_extracted(judge_sparse: dict, spec: dict) -> dict:
    """Over operator-level fields: stated = extracted and different; unsupported = not extracted and not default."""
    stated = _flat(judge_sparse)
    want, got = _flat(resolve.resolve(judge_sparse)), _flat(resolve.resolve(spec))
    out = {'stated': [], 'unsupported': []}
    for f in FIELD_CLASSES['operator']:
        if not _eq(got.get(f), want[f]):
            out['stated' if f in stated else 'unsupported'].append({'field': f, 'expected': want[f], 'spec': got.get(f)})
    return out


_d = diff_extracted({'slice_action': {'action': 'reserve_prb_for_protected_ue', 'reserve_pct': 20, 'target': 'protected_ue'}},
                    {'label_rule': {'feature': 'cell_prb_util', 'op': '>=', 'threshold_pct': 72, 'min_duration_s': 3},
                     'slice_action': {'action': 'reserve_prb_for_protected_ue', 'reserve_pct': 50, 'target': 'protected_ue'}})
assert [x['field'] for x in _d['stated']] == ['slice_action.reserve_pct']
assert [x['field'] for x in _d['unsupported']] == ['label_rule.threshold_pct']
assert check_actuatable({'slice_action': {'target': 'center_ues'}})['result'] == 'FAIL'
assert check_actuatable({'slice_action': {'target': 'unspecified'}})['result'] == 'HOLD'
assert check_actuatable({'label_rule': {'op': 'unspecified'}, 'slice_action': {'target': 'protected_ue'}})['result'] == 'PASS'


def extract_readback(intent: str, spec: dict, model: str, seed: int) -> dict:
    import agent
    r = agent.run(intent, model=model, seed=seed, mode='judge')
    j = {'model': model, 'seed': seed, 'judge_sparse': r['sparse_spec'], 'judge_valid': r['valid'],
         'attempts': r['attempts'], 'prompt_tokens': sum(x['prompt_tokens'] or 0 for x in r['calls']),
         'completion_tokens': sum(x['completion_tokens'] or 0 for x in r['calls']),
         'wall_s': round(sum(x['wall_s'] for x in r['calls']), 3),
         'warm': all(x['load_duration_ms'] < 1000 for x in r['calls']), 'prompt_sha256': r['prompt_sha256']}
    if not r['valid']:
        return {'result': 'HOLD', 'reason': 'judge extraction did not validate', 'judge': j, 'mismatches': None}
    if (r['sparse_spec'].get('slice_action') or {}).get('target') == 'unspecified':
        return {'result': 'HOLD', 'reason': 'beneficiary not stated', 'judge': j, 'mismatches': None}
    d = diff_extracted(r['sparse_spec'], spec)
    mism = d['stated'] + d['unsupported']
    return {'result': 'HOLD' if mism else 'PASS', 'mismatches': mism, 'stated_mismatch': d['stated'],
            'unsupported_deviation': d['unsupported'], 'judge': j, 'extracted': sorted(_flat(r['sparse_spec']))}


def check_conflict(registry: dict) -> dict:
    mine = set(registry['candidate']['resources'])
    hits = [x for x in registry['xapps'] if x.get('deployed') and mine & set(x['resources'])]
    return {'result': 'HOLD' if hits else 'PASS',
            'conflicts': [{'name': x['name'], 'resources': sorted(mine & set(x['resources']))} for x in hits]}


def run(intent: str, spec: dict, *, judge_model: str = 'llama3.3:70b', seed: int = 42, registry: dict | None = None,
        readback: str = 'extract', kpi_feedback: dict | None = None, tm: timing.Timer | None = None) -> dict:
    registry = registry or json.load(open(REGISTRY))
    mark = (lambda *a, **k: tm.mark(*a, **k)) if tm else (lambda *a, **k: None)
    ok = isinstance(spec, dict)
    mark('gate_start')
    t0 = time.monotonic()
    checks = {}
    for name, fn in (('actuatable', check_actuatable), ('schema', check_schema), ('engineering', check_engineering)):
        mark(f'gate_{name}_start')
        try:
            checks[name] = fn(spec) if ok else {'result': 'FAIL', 'reason': 'spec is not an object'}
        except Exception as exc:  # a malformed spec must not crash the gate: it fails the check
            checks[name] = {'result': 'FAIL', 'reason': f'{type(exc).__name__}: {exc}'}
        mark(f'gate_{name}_end', result=checks[name]['result'])
    mark('gate_conflict_start')
    checks['conflict'] = check_conflict(registry)
    mark('gate_conflict_end', result=checks['conflict']['result'])
    det_s = time.monotonic() - t0
    t1 = time.monotonic()
    mark('gate_readback_start', variant=readback)
    try:
        sentence = readback_sentence(spec)
    except Exception:  # an out-of-range spec still gets a read-back where possible
        sentence = None
    if readback == 'extract':
        checks['readback'] = {'variant': 'extract', 'intent': intent, 'readback': sentence,
                              **extract_readback(intent, spec, judge_model, seed)}
    else:
        j = judge(intent, sentence or json.dumps(spec), judge_model, seed)
        checks['readback'] = {'variant': 'sentence', 'result': 'PASS' if j['match'] else 'HOLD', 'intent': intent,
                              'readback': sentence, 'judge': j}
    mark('gate_readback_end', variant=readback, result=checks['readback']['result'],
         warm=checks['readback']['judge']['warm'])
    rb_s = time.monotonic() - t1
    final = max((c['result'] for c in checks.values()), key=lambda r: SEVERITY[r])
    caught_by = [n for n, c in checks.items() if c['result'] in ('HOLD', 'FAIL')]
    mark('gate_end', final=final, caught_by=caught_by)
    return {'final': final, 'caught_by': caught_by, **checks,
            'dry_run': {'result': 'STUB', 'log': 'dry-run: not implemented'},
            'rollback': {'result': 'STUB', 'log': 'rollback trigger: not implemented', 'kpi_feedback': kpi_feedback},
            'gate_deterministic_s': round(det_s, 4), 'gate_readback_s': round(rb_s, 3)}


def model_parity(model_dir: str, spec: dict, ns: str, ric_pod: str, lease_s: float, tm: timing.Timer | None = None) -> dict:
    """Deterministic check 'model_parity' (implemented 2026-11-04, after the Exp-04 dry-run finding): runs after
    training and BEFORE registration. On the fixed recorded windows (zto/parity.py) the xApp-side features + the
    candidate model must reproduce the offline predictions and the predictions must not be constant, and the recorded
    series replayed through the xApp's debounce + lease logic must show the lease expiring and the cap lifting in OFF
    in every cycle, with the dualbrain11 re-arm rule reproduced exactly; else FAIL."""
    import parity
    tm and tm.mark('gate_model_parity_start')
    try:
        ts = spec['sampling_ms']  # effective xApp values (render.py filters): debounce default, label rule in samples
        v = parity.check(model_dir, ts, spec['window_ms'], ns, ric_pod,
                         debounce_n=spec['control']['debounce_n'] or math.ceil(1000 / ts), lease_s=lease_s,
                         threshold=spec['label_rule']['threshold_pct'],
                         min_dur=max(1, math.ceil(float(spec['label_rule']['min_duration_s']) * 1000 / ts)))
    except Exception as exc:  # the check itself failing is a FAIL, never a silent pass
        v = {'result': 'FAIL', 'reason': f'{type(exc).__name__}: {exc}'}
    tm and tm.mark('gate_model_parity_end', result=v['result'])
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--intent', required=True)
    ap.add_argument('--spec', required=True)
    ap.add_argument('--judge', default='llama3.3:70b')
    ap.add_argument('--registry', default=REGISTRY)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--readback', choices=('extract', 'sentence'), default='extract')
    ap.add_argument('--run-id')
    ap.add_argument('--run-dir')
    a = ap.parse_args()
    tm = timing.Timer(a.run_id, a.run_dir) if a.run_id and a.run_dir else None
    res = run(a.intent, json.load(open(a.spec)), judge_model=a.judge, seed=a.seed,
              registry=json.load(open(a.registry)), readback=a.readback, tm=tm)
    print(json.dumps(res, indent=1))
    if res['final'] != 'PASS':
        print(f"\n{res['final']} (caught by {', '.join(res['caught_by'])}) for operator confirmation:\n  intent:   ",
              a.intent, '\n  read-back:', res['readback'].get('readback'),
              '\n  fields:', {n: res[n].get('fields') or res[n].get('mismatches') for n in res['caught_by']})
    sys.exit({'PASS': 0, 'HOLD': 2, 'FAIL': 1}[res['final']])


if __name__ == '__main__':
    main()
