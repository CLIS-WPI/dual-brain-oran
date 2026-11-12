"""Exp-08 fault-injection harness: corrupt each correct reference spec, run the gate on every variant.

usage: gate_inject.py [--references I1,I2,I3] [--seeds 1,2,3] [--readbacks extract,sentence]
                      [--out exp08_gate_results.csv]
Variants (from each correct, resolved reference spec in zto/intents/reference/):
  correct            unchanged                                             → expected PASS
Every check runs on every variant (deterministic first, then the read-back); final = most severe.
  reserve_to_50      slice_action.reserve_pct → 50          (in range)     → expected HOLD (read-back)
  threshold_to_72    label_rule.threshold_pct → 72          (in range)     → expected HOLD (read-back)
  min_duration_to_8  label_rule.min_duration_s → 8          (in range)     → expected HOLD (read-back)
  horizon_changed    horizon_ms → 7500 (or 2500 if it was 7500)            → expected HOLD (read-back)
  debounce_changed   control.debounce_n → 10                (in range)     → expected HOLD (engineering default)
  target_center      slice_action.target → center_ues (schema-valid)       → expected FAIL (check 0: not actuatable)
  op_inverted        label_rule.op → "<=" (schema-valid)                   → expected FAIL (check 0: not actuatable)
  threshold_to_95    label_rule.threshold_pct → 95 (out of range)          → expected FAIL (check 1: schema)
  prb_pool_conflict  correct spec, energy_saver_xapp deployed              → expected HOLD (check 3: conflict)
Not applicable (recorded, not run): action_swapped — the SLICE catalogue has one action.
Each row: reference, readback_variant, variant, seed, final, expected, every check's verdict, caught_by (all
non-PASS checks), the read-back's stated/unsupported fields, judge wall-clock (warm), deterministic/read-back time.
Summary per read-back variant: corrupted-caught rate (final != PASS), correct-passed rate (final == PASS) and
false-HOLD rate (correct spec not passed), plus per-variant counts of which checks caught it.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import json
import os
import sys

ZTO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ZTO)
import gate  # noqa: E402
import resolve  # noqa: E402


def variants(spec: dict, registry: dict):
    def mod(path, value):
        s = copy.deepcopy(spec)
        d = s
        *head, last = path.split('.')
        for p in head:
            d = d[p]
        d[last] = value
        return s
    deployed = copy.deepcopy(registry)
    for x in deployed['xapps']:
        if x['name'].startswith('energy_saver'):
            x['deployed'] = True
    h = 2500 if spec['horizon_ms'] == 7500 else 7500
    return [('correct', spec, registry, 'PASS'),
            ('reserve_to_50', mod('slice_action.reserve_pct', 50), registry, 'HOLD'),
            ('threshold_to_72', mod('label_rule.threshold_pct', 72), registry, 'HOLD'),
            ('min_duration_to_8', mod('label_rule.min_duration_s', 8), registry, 'HOLD'),
            ('horizon_changed', mod('horizon_ms', h), registry, 'HOLD'),
            ('debounce_changed', mod('control.debounce_n', 10), registry, 'HOLD'),
            ('target_center', mod('slice_action.target', 'center_ues'), registry, 'FAIL'),
            ('op_inverted', mod('label_rule.op', '<='), registry, 'FAIL'),
            ('threshold_to_95', mod('label_rule.threshold_pct', 95), registry, 'FAIL'),
            ('prb_pool_conflict', spec, deployed, 'HOLD')]


def summarize(rows: list[dict]) -> dict:
    out = {}
    for rb in sorted({r['readback_variant'] for r in rows}):
        rr = [r for r in rows if r['readback_variant'] == rb]
        bad = [r for r in rr if r['variant'] != 'correct']
        good = [r for r in rr if r['variant'] == 'correct']
        by = collections.defaultdict(collections.Counter)
        for r in bad:
            for c in (r['caught_by'].split(';') if r['caught_by'] else ['none']):
                by[r['variant']][c] += 1
        out[rb] = {'corrupted_caught': f"{sum(r['final'] != 'PASS' for r in bad)}/{len(bad)}",
                   'correct_passed': f"{sum(r['final'] == 'PASS' for r in good)}/{len(good)}",
                   'false_hold': f"{sum(r['final'] != 'PASS' for r in good)}/{len(good)}",
                   'caught_by': {v: dict(c) for v, c in by.items()}}
    out['not_applicable'] = ['action_swapped (one SLICE catalogue action)']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--references', default='I1,I2,I3')
    ap.add_argument('--seeds', default='1,2,3')
    ap.add_argument('--judge', default='llama3.3:70b')
    ap.add_argument('--readbacks', default='extract,sentence')
    ap.add_argument('--out', default='exp08_gate_results.csv')
    a = ap.parse_args()
    registry = json.load(open(gate.REGISTRY))
    gate.judge('warm-up', 'warm-up', a.judge, 0)  # pair residency: gate time is measured on a warm judge only
    rows = []
    for ref_id in a.references.split(','):
        ref = json.load(open(os.path.join(ZTO, 'intents', 'reference', f'{ref_id}.json')))
        spec, intent = resolve.resolve(ref), ref['_intent_text']
        for rb in a.readbacks.split(','):
            for name, s, reg, expected in variants(spec, registry):
                for seed in map(int, a.seeds.split(',')):
                    r = gate.run(intent, s, judge_model=a.judge, seed=seed, registry=reg, readback=rb)
                    rbk = r['readback']
                    j = rbk.get('judge') or {}
                    rows.append({'reference': ref_id, 'readback_variant': rb, 'variant': name, 'seed': seed,
                                 'final': r['final'], 'expected': expected, 'caught_by': ';'.join(r['caught_by']),
                                 'actuatable': r['actuatable']['result'], 'schema': r['schema']['result'],
                                 'engineering': r['engineering']['result'], 'conflict': r['conflict']['result'],
                                 'readback': rbk['result'],
                                 'engineering_fields': ';'.join(x['field'] for x in r['engineering'].get('fields') or []),
                                 'stated_mismatch': ';'.join(m['field'] for m in rbk.get('stated_mismatch') or []),
                                 'unsupported_deviation': ';'.join(m['field'] for m in rbk.get('unsupported_deviation') or []),
                                 'readback_reason': rbk.get('reason') or j.get('reason'),
                                 'judge_wall_s': j.get('wall_s'), 'judge_warm': j.get('warm'),
                                 'gate_deterministic_s': r['gate_deterministic_s'], 'gate_readback_s': r['gate_readback_s']})
                    print(json.dumps(rows[-1]), flush=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    json.dump(summary, open(os.path.splitext(a.out)[0] + '_summary.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
