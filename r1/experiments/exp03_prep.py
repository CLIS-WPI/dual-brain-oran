"""Exp-03 arm artifacts from the Exp-04 batch (run after Exp-04, before Exp-03; never during measurements).

usage: exp03_prep.py --exp04-batch exp04-20261105b --out results/exp03/<batch>/arms
  (c) the FIRST official 20-min Exp-04 run that is complete and in regime: its gated config.json + model.onnx +
      preprocess.json, byte-for-byte (the gate record incl. model_parity must be PASS, else exit 2).
  (b) the same config.json with "policy" -> "threshold" only (same model files on the RIC; unused by that policy).
  (d) EXPLORATORY: the same config.json bytes + a model trained on the SAME Exp-04 dataset (same telemetry, same
      train/test cycles, same NeuralSmith settings) rebuilt by dataset/build.py with horizon_ms 7500; gate check
      model_parity on it with the same lease (FAIL → arm (d) is marked and skipped, (a)/(b)/(c) unaffected).
  Asserts each derived file differs from (c) only as stated; writes arms/<x>/{config.json, model.onnx,
  preprocess.json} and arms/manifest.json (sha256 per file, the diffs, source run, (d) parity, exp01-select hashes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

R1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R1, 'scripts'), os.path.join(R1, 'zto'), os.path.join(R1, 'experiments'), R1]
import r1env as E  # noqa: E402
import gate  # noqa: E402
from exp04_e2e import PY, sh, train_neuralsmith  # noqa: E402

sha = lambda b: hashlib.sha256(b).hexdigest()
FILES = ('config.json', 'model.onnx', 'preprocess.json')
H_D = 7500


def with_ref(cfg: bytes, ref: int) -> bytes:
    """Milad 2026-11-06: UE3's CU-UE-ID changed after its LDPC crash (7 → 1); the Exp-03 configs take the gated file with
    ONLY protected_ue_ref changed (a runtime field from the RAN mapping, never from the LLM or the model)."""
    c = json.loads(cfg)
    out = (json.dumps(dict(c, protected_ue_ref=ref), indent=1) + '\n').encode()
    d = json.loads(out)
    assert {k for k in set(c) | set(d) if c.get(k) != d.get(k)} <= {'protected_ue_ref'}
    return out


def derive_b(cfg_c: bytes) -> bytes:
    c = json.loads(cfg_c)
    b = dict(c, policy='threshold')
    out = (json.dumps(b, indent=1) + '\n').encode()  # exp04_e2e writes json.dumps(cfg, indent=1) + '\n'
    assert json.dumps(c, indent=1) + '\n' == cfg_c.decode(), 'arm (c) config is not in the runner\'s canonical form'
    diff = {k: (c.get(k), b.get(k)) for k in set(c) | set(b) if c.get(k) != b.get(k)}
    assert diff == {'policy': ('ml', 'threshold')}, diff
    return out


def select_c(batch: str) -> dict:
    bj = json.load(open(os.path.join(E.RESULTS, 'exp04', f'{batch}-batch.json')))
    for r in bj['runs']:
        if not r['slot'].startswith('dry') and r['collection_min'] == 20 and r['status'] == 'complete' \
                and (r['regime'] or {}).get('in_regime'):
            return r
    raise SystemExit('no complete in-regime official 20-min run in ' + batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp04-batch', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--protected-ue-ref', type=int, help="UE3's current CU-UE-ID if it differs from the gated config's")
    a = ap.parse_args()
    src = select_c(a.exp04_batch)
    run = os.path.join(E.RESULTS, 'exp04', src['run_id'])
    meta = json.load(open(os.path.join(run, 'run_meta.json')))
    gj = json.load(open(os.path.join(run, 'gate.json')))
    if gj['final'] != 'PASS' or gj.get('model_parity', {}).get('result') != 'PASS':
        print(json.dumps({'arm_c_gate': gj['final'], 'model_parity': gj.get('model_parity', {}).get('result')}))
        sys.exit(2)
    c = {f: open(os.path.join(run, 'model' if f != 'config.json' else '', f), 'rb').read() for f in FILES}
    assert {f: sha(v) for f, v in c.items()} == meta['hashes'], 'arm (c) files differ from the Exp-04 run_meta hashes'
    gated_ref = json.loads(c['config.json'])['protected_ue_ref']
    ref_change = None
    if a.protected_ue_ref is not None and a.protected_ue_ref != gated_ref:
        c = dict(c, **{'config.json': with_ref(c['config.json'], a.protected_ue_ref)})
        ref_change = {'protected_ue_ref': [gated_ref, a.protected_ue_ref], 'gated_config_sha256': meta['hashes']['config.json'],
                      'derived_config_sha256': sha(c['config.json']), 'reason': 'UE3 CU-UE-ID changed after its LDPC crash '
                      '(Exp-04 s6, 2026-11-05 20:31); Milad 2026-11-06: change only this runtime field'}
    arms = {'c': c, 'b': dict(c, **{'config.json': derive_b(c['config.json'])})}

    # (d): same telemetry + split, horizon 7500, same NeuralSmith settings
    spec = json.load(open(os.path.join(run, 'spec.json')))
    raw = next(os.path.join(run, 'raw', d) for d in os.listdir(os.path.join(run, 'raw')) if os.path.isdir(os.path.join(run, 'raw', d)))
    cm = json.load(open(os.path.join(raw, 'meta.json')))
    n = len(cm['schedule']['cycles'])
    n_train = max(1, round(n * 9 / 13))  # as exp04_e2e
    os.makedirs(a.out, exist_ok=True)
    dd = os.path.join(a.out, 'd_build')
    ds, md = os.path.join(dd, 'dataset'), os.path.join(dd, 'model')
    os.makedirs(md, exist_ok=True)
    b = sh(PY, os.path.join(R1, 'dataset', 'build.py'), '--in', os.path.join(raw, 'telemetry.csv'), '--ts', spec['sampling_ms'],
           '--window', spec['window_ms'], '--horizon', H_D, '--threshold', spec['label_rule']['threshold_pct'],
           '--min-duration', spec['label_rule']['min_duration_s'], '--train-cycles', f'1-{n_train}',
           '--test-cycles', f'{n_train + 1}-{n}', '--out', ds)
    if b.returncode:
        raise SystemExit('dataset build (H 7500) failed: ' + b.stderr[-600:])
    rep = train_neuralsmith(ds, spec, md)
    d_files = {f: open(os.path.join(md, f), 'rb').read() for f in ('model.onnx', 'preprocess.json')}
    lease_s = json.loads(c['config.json'])['lease_s']
    spec_d = dict(spec, horizon_ms=H_D)
    mp = gate.model_parity(md, spec_d, E.NS, E.pod('ric')[0], lease_s)
    arms['d'] = dict(c, **d_files)
    assert arms['d']['config.json'] == c['config.json'] and arms['d']['model.onnx'] != c['model.onnx']

    man = {'source_exp04_run': src['run_id'], 'source_hashes': meta['hashes'], 'arm_c_gate': {
        'final': gj['final'], 'model_parity': gj.get('model_parity')}, 'lease_s': lease_s,
        'protected_ue_ref_change': ref_change, 'arms': {}, 'diffs': {'b': {'config.json': {'policy': ['ml', 'threshold']}, 'model files': 'identical to (c)'},
                              'd': {'config.json': 'identical to (c)', 'model files': f'retrained, horizon_ms {H_D}'}},
        'd_model': {'horizon_ms': H_D, 'train_cycles': f'1-{n_train}', 'test_cycles': f'{n_train + 1}-{n}',
                    'report': {k: rep.get(k) for k in ('algorithm', 'train_time_s', 'size_kb', 'metrics')},
                    'model_parity': mp, 'usable': mp['result'] == 'PASS'},
        'exp01_select_models': {h: {f: sha(open(os.path.join(R1, 'results', 'exp01', 'exp01-20261104', f'ts500_h{h}', 'neuralsmith', f), 'rb').read())
                                    for f in ('model.onnx', 'preprocess.json')}
                                for h in (2500, 7500) if os.path.isdir(os.path.join(R1, 'results', 'exp01', 'exp01-20261104', f'ts500_h{h}', 'neuralsmith'))},
        'git_commit': E.git('rev-parse', 'HEAD')}
    for x, fs in arms.items():
        os.makedirs(os.path.join(a.out, x), exist_ok=True)
        for f, v in fs.items():
            open(os.path.join(a.out, x, f), 'wb').write(v)
        man['arms'][x] = {f: sha(v) for f, v in fs.items()}
    json.dump(man, open(os.path.join(a.out, 'manifest.json'), 'w'), indent=1, default=str)
    print(json.dumps({'source': src['run_id'], 'arms': man['arms'], 'd_parity': mp['result']}, indent=1))


_c = {'mode': 'control', 'policy': 'ml', 'lease_s': 50.0, 'protected_ue_ref': 7}
assert json.loads(with_ref((json.dumps(_c, indent=1) + '\n').encode(), 1)) == dict(_c, protected_ue_ref=1)
assert json.loads(derive_b((json.dumps(_c, indent=1) + '\n').encode())) == dict(_c, policy='threshold')

if __name__ == '__main__':
    main()
