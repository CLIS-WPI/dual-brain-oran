"""Resolve a spec or reference against zto/defaults.json (the single source of defaults).

  resolve.py intents/reference/I2.json            # print the full, schema-validated reference spec
  resolve.py --write-listing                      # regenerate paper/listing_example.json from reference I1
A reference file holds only what its intent states explicitly ("$defaults": "zto/defaults.json" marks it);
everything else comes from the defaults. control.debounce_n null → ceil(1000 / sampling_ms).
"""
from __future__ import annotations

import json
import math
import os
import sys

ZTO = os.path.dirname(os.path.abspath(__file__))
R1 = os.path.dirname(ZTO)
DEFAULTS = json.load(open(os.path.join(ZTO, 'defaults.json')))
SCHEMA = json.load(open(os.path.join(ZTO, 'spec.schema.json')))


def _merge(base, over):
    out = json.loads(json.dumps(base))
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def resolve(partial: dict) -> dict:
    """Defaults ⊕ the explicit fields; '_'/'$' keys dropped; derived debounce filled; key order = schema order."""
    explicit = {k: v for k, v in partial.items() if not k.startswith(('_', '$'))}
    d = {k: v for k, v in DEFAULTS.items() if not k.startswith('_')}
    full = _merge(d, explicit)
    if full.get('label_rule', {}).get('op') == 'unspecified':  # op has a default; target has none (stays as given)
        full['label_rule']['op'] = DEFAULTS['label_rule']['op']
    if full.get('control', {}).get('debounce_n') is None:
        full['control'] = {'debounce_n': math.ceil(1000 / full['sampling_ms'])}
    return {k: full[k] for k in SCHEMA['properties'] if k in full}


def validated(spec: dict) -> dict:
    import jsonschema
    jsonschema.Draft202012Validator(SCHEMA).validate(spec)
    return spec


assert resolve({'slice_action': {'reserve_pct': 30}})['slice_action'] == {
    'action': 'reserve_prb_for_protected_ue', 'reserve_pct': 30}  # no target default
assert resolve({'label_rule': {'op': 'unspecified'}})['label_rule']['op'] == '>='
assert resolve({'slice_action': {'target': 'unspecified'}})['slice_action']['target'] == 'unspecified'
assert resolve({'sampling_ms': 100})['control']['debounce_n'] == 10


def main():
    if '--write-listing' in sys.argv:
        spec = validated(resolve(json.load(open(os.path.join(ZTO, 'intents', 'reference', 'I1.json')))))
        p = os.path.join(R1, 'paper', 'listing_example.json')
        open(p, 'w').write(json.dumps(spec, indent=2) + '\n')
        print('wrote', os.path.relpath(p, R1))
        return
    print(json.dumps(validated(resolve(json.load(open(sys.argv[1])))), indent=2))


if __name__ == '__main__':
    main()
