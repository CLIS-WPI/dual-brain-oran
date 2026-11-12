"""Print the xApp template's slots (the number goes in the paper) and check each has a bound in the schema.

usage: list_slots.py [--json]
A slot = one `{{ field.path ... }}` expression in zto/xapp/config.json.j2 whose root is a spec field
(`runtime.*` values come from the testbed, not the spec, and are listed separately). Exits 1 if a slot has no
schema field or a numeric slot has no minimum/maximum.
"""
import json
import os
import re
import sys

R1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(R1, 'zto', 'xapp', 'config.json.j2')
SCHEMA = json.load(open(os.path.join(R1, 'zto', 'spec.schema.json')))


def schema_node(path):
    node = SCHEMA
    for part in path.split('.'):
        node = (node.get('properties') or {}).get(part)
        if node is None:
            return None
    return node


def main():
    text = re.sub(r'\{#.*?#\}', '', open(TEMPLATE).read(), flags=re.S)
    exprs = re.findall(r'\{\{\s*([a-zA-Z_][\w.]*)', text)
    slots = sorted({e for e in exprs if not e.startswith('runtime.')})
    runtime = sorted({e for e in exprs if e.startswith('runtime.')})
    rows, bad = [], []
    for s in slots:
        n = schema_node(s)
        if n is None:
            bad.append(f'{s}: no schema field')
            continue
        bound = ('enum ' + json.dumps(n['enum'])) if 'enum' in n else \
                (f"[{n['minimum']}..{n['maximum']}]" if 'minimum' in n and 'maximum' in n else None)
        if bound is None:
            bad.append(f'{s}: no bound in the schema')
        rows.append({'slot': s, 'bound': bound, 'type': n.get('type', 'enum')})
    if '--json' in sys.argv:
        print(json.dumps({'n_slots': len(rows), 'slots': rows, 'runtime': runtime, 'errors': bad}, indent=1))
    else:
        print(f'{len(rows)} slots in {os.path.relpath(TEMPLATE, R1)}:')
        for r in rows:
            print(f"  {r['slot']:28} {r['bound']}")
        print('runtime (testbed, not spec):', ', '.join(runtime))
        for b in bad:
            print('ERROR', b)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
