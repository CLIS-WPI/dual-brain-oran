"""Validate a ZTO task spec against spec.schema.json and render the xApp config from the template.

usage: render.py --spec spec.json [--protected-ue-ref 5] [--policy ml|threshold] [--out config.json]
Defaults from the schema are applied first (label_rule.min_duration_s, control.debounce_n = ceil(1000/T_s)).
Rendering uses StrictUndefined: a slot missing from the spec is an error, never an empty string.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import jinja2
import jsonschema

ZTO = os.path.dirname(os.path.abspath(__file__))
SCHEMA = json.load(open(os.path.join(ZTO, 'spec.schema.json')))


def with_defaults(spec: dict) -> dict:
    s = json.loads(json.dumps(spec))
    s['label_rule'].setdefault('min_duration_s', SCHEMA['properties']['label_rule']['properties']['min_duration_s']['default'])
    s.setdefault('control', {})
    return s


def validate(spec: dict) -> list[str]:
    """All schema violations (empty = valid)."""
    v = jsonschema.Draft202012Validator(SCHEMA)
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in v.iter_errors(spec)]


def env() -> jinja2.Environment:
    e = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.join(ZTO, 'xapp')), undefined=jinja2.StrictUndefined,
                           keep_trailing_newline=True)
    e.filters['samples'] = lambda ms, ts: max(1, int(ms) // int(ts))
    e.filters['duration_samples'] = lambda s, ts: max(1, math.ceil(float(s) * 1000 / int(ts)))
    e.filters['default_debounce'] = lambda n, ts: int(n) if n else math.ceil(1000 / int(ts))
    return e


def render(spec: dict, *, protected_ue_ref: int, policy: str = 'ml', workdir: str = '/tmp/dualbrain',
           lease_s: float | None = None) -> dict:
    """lease_s: reservation lease from the training data (zto/lease.py, data curation; not a spec field). The xApp
    keeps the cap for lease_s (minimum hold), then only while the policy is hot (first non-hot sample lifts it).
    None before any data exists (gate/Exp-02 spec-level rendering): rendered as null; deploy.py refuses it."""
    s = with_defaults(spec)
    if lease_s is not None and not 10 <= float(lease_s) <= 120:
        raise ValueError(f'lease_s {lease_s} outside [10, 120] s')
    errors = validate(s)
    if errors:
        raise ValueError('spec violates spec.schema.json: ' + '; '.join(errors))
    ctx = {**s, 'control': {'debounce_n': s['control'].get('debounce_n')},
           'runtime': {'protected_ue_ref': int(protected_ue_ref), 'policy': policy, 'workdir': workdir,
                       'lease_s': None if lease_s is None else float(lease_s)}}
    return json.loads(env().get_template('config.json.j2').render(**ctx))


I1 = {'sampling_ms': 250, 'window_ms': 2000,
      'horizon_ms': 2500, 'collection_min': 20,
      'label_rule': {'feature': 'cell_prb_util', 'op': '>=', 'threshold_pct': 80, 'min_duration_s': 3},
      'model': {'latency_budget_ms': 5, 'max_size_kb': 2048},
      'slice_action': {'action': 'reserve_prb_for_protected_ue', 'reserve_pct': 20, 'target': 'protected_ue'}}
_c = render(I1, protected_ue_ref=5, lease_s=50.0)
assert render(I1, protected_ue_ref=5)['lease_s'] is None
assert _c['window_samples'] == 8 and _c['min_duration_samples'] == 12 and _c['trigger_consecutive'] == 4, _c
assert _c['lease_s'] == 50.0 and 'release_below_cell_prb' not in _c and 'min_hold_samples' not in _c, _c
assert _c['protected_ue_ref'] == 5 and _c['reserve_pct'] == 20 and _c['metric'] == 'RRU.PrbTotDl'
assert validate({**I1, 'label_rule': {**I1['label_rule'], 'threshold_pct': 95}})  # range violation detected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--protected-ue-ref', type=int, default=-1)
    ap.add_argument('--policy', choices=('ml', 'threshold'), default='ml')
    ap.add_argument('--out')
    a = ap.parse_args()
    cfg = render(json.load(open(a.spec)), protected_ue_ref=a.protected_ue_ref, policy=a.policy)
    txt = json.dumps(cfg, indent=1)
    if a.out:
        open(a.out, 'w').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
