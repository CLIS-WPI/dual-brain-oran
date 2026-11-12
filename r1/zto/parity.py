"""Feature/model parity between training (r1/dataset/build.py + offline ONNX) and the deployed xApp code path.

Used by make smoke (check model_parity_smoke) and by the gate's deterministic check "model_parity" (Exp-04, before
registration). On a FIXED set of recorded windows (zto/parity/series_ts<T>.csv: exp01-select test cycles, cut into
windows of window_ms / T_s consecutive KPM samples, KPM gaps > 2·T_s skipped) it compares
  features   the xApp's own C feature path (cc_replay in the RIC pod = cc_window_features) vs build.features()
             → max |difference| over the model's input features must be <= FEATURE_TOL
  inputs     every model input name must resolve in the xApp (unmatched_inputs == 0; the dualbrain8 failure)
  labels     the xApp's predictions (cc_model_predict) vs the offline ONNX prediction (mlengine/predict.py)
             on the build.py features → identical on every window
  variation  the xApp's predicted labels are not constant over the set (min != max)
  lease      (dualbrain10, with debounce_n + lease_s) the recorded series replayed through the xApp's debounce +
             lease logic (cc_replay --policy, actuation simulated): in every cycle the cap is reserved and the lease
             expires with the cap lifting in that cycle's OFF
  re-arm     (dualbrain11, with threshold_pct + min_duration_samples) the replay also applies the re-arm rule
             (after a release no reserve until cell < threshold for min_duration_samples): the binary's armed state
             must equal a Python reference of the rule on every sample (rearm_mismatch == 0) and every cycle must
             still reserve. LIMIT: the recorded series has no actuation, so the replay cannot show actuation effects
             on the network (e.g. the post-release backlog drain that re-triggered the cap in Exp-04 dry run 6);
             only a live (or digital-twin) dry run can.
→ {'result': 'PASS'|'FAIL', 'n', 'max_feature_abs_err', 'unmatched_inputs', 'label_mismatch', 'pred_min', 'pred_max',
   'proba_min', 'proba_max', 'reason'}.
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import uuid

ZTO = os.path.dirname(os.path.abspath(__file__))
R1 = os.path.dirname(ZTO)
sys.path.insert(0, os.path.join(R1, 'dataset'))
from build import FEATURES, features  # noqa: E402

FEATURE_TOL = 1e-4
MLV = os.path.join(R1, '.venv-mlengine', 'bin', 'python')
REPLAY = '/usr/local/flexric/xApp/c/dual_brain/cc_replay'


def windows(ts_ms: int, window_ms: int) -> list[list[float]]:
    path = os.path.join(ZTO, 'parity', f'series_ts{ts_ms}.csv')
    rows = list(csv.DictReader(ln for ln in open(path) if not ln.startswith('#')))
    t, v, n = [int(r['t_ms']) for r in rows], [float(r['cell_prb_util']) for r in rows], max(1, window_ms // ts_ms)
    return [v[i - n + 1:i + 1] for i in range(n - 1, len(v))
            if all(b - a <= 2 * ts_ms for a, b in zip(t[i - n + 1:i + 1], t[i - n + 2:i + 1]))]


_DOCKER_WD = None  # host dir bind-mounted at /tmp/r1-parity when replaying offline


def _exec_argv(ns: str, pod: str) -> list[str]:
    """kubectl exec into the RIC pod; ns == 'docker' → run the RIC image locally (pod = image), no cluster needed."""
    global _DOCKER_WD
    if ns != 'docker':
        return ['kubectl', '-n', ns, 'exec', '-i', pod, '--']
    _DOCKER_WD = _DOCKER_WD or tempfile.mkdtemp(prefix='r1-parity-')
    return ['docker', 'run', '--rm', '-i', '--network', 'none', '-v', f'{_DOCKER_WD}:/tmp/r1-parity', '--entrypoint', '',
            pod]


def _ric_exec(ns: str, pod: str, script: str, stdin: bytes | None = None) -> str:
    r = subprocess.run(_exec_argv(ns, pod) + ['sh', '-c', script], input=stdin, capture_output=True, timeout=120)
    if r.returncode:
        raise RuntimeError(f'RIC exec failed: {r.stderr.decode()[-400:]}')
    return r.stdout.decode()


def rearm_reference(cells: list[float], actions: list[str], threshold: float, min_dur: int) -> list[int]:
    """Python reference of cc_rearm_step as the replay applies it (armed after each sample)."""
    armed, below, reserved, out = 1, 0, 0, []
    for cell, act in zip(cells, actions):
        if not reserved and not armed:
            below = below + 1 if cell < threshold else 0
            if below >= max(1, min_dur):
                armed, below = 1, 0
        if act == 'reserve':
            reserved = 1
        elif act == 'release':
            reserved, armed, below = 0, 0, 0
        out.append(armed)
    return out


def lease_replay(model_dir: str, ts_ms: int, window_ms: int, debounce_n: int, lease_s: float, ns: str, ric_pod: str,
                 threshold: float, min_dur: int) -> dict:
    import math
    path = os.path.join(ZTO, 'parity', f'series_ts{ts_ms}.csv')
    rows = list(csv.DictReader(ln for ln in open(path) if not ln.startswith('#')))
    lease_n = math.ceil(lease_s * 1000 / ts_ms)
    wd = f'/tmp/r1-parity/{uuid.uuid4().hex[:8]}'
    try:
        for f, data in (('model.onnx', open(os.path.join(model_dir, 'model.onnx'), 'rb').read()),
                        ('preprocess.json', open(os.path.join(model_dir, 'preprocess.json'), 'rb').read()),
                        ('series.txt', ''.join(r['cell_prb_util'] + '\n' for r in rows).encode())):
            _ric_exec(ns, ric_pod, f'mkdir -p {wd} && cat > {wd}/{f}', data)
        out = _ric_exec(ns, ric_pod, f'{REPLAY} --policy {wd} {ts_ms} {wd}/series.txt {max(1, window_ms // ts_ms)} '
                                     f'{debounce_n} {lease_n} {threshold} {min_dur}')
    finally:
        subprocess.run(_exec_argv(ns, ric_pod) + ['rm', '-rf', wd], capture_output=True)
    xs = list(csv.DictReader(out.splitlines()))
    per = {}
    for r, x in zip(rows, xs):
        c = per.setdefault(r['cycle'], {'reserve': 0, 'extend': 0, 'release': 0, 'release_in_off': 0, 'rearm': 0})
        if x['action']:
            c[x['action']] += 1
            c['release_in_off'] += x['action'] == 'release' and r['phase'] == 'off'
    ref = rearm_reference([float(x['cell']) for x in xs], [x['action'] for x in xs], threshold, min_dur)
    mism = sum(int(x['armed']) != a for x, a in zip(xs, ref))
    ok = bool(per) and len(xs) == len(rows) and all(c['reserve'] >= 1 and c['release_in_off'] >= 1 for c in per.values())
    why = [m for m, bad in (('a cycle without reserve or without the cap lifting in OFF', not ok),
                            (f'xApp re-arm state differs from the rule on {mism} samples', mism)) if bad]
    return {'result': 'FAIL' if why else 'PASS', 'lease_s': lease_s, 'lease_samples': lease_n, 'debounce_n': debounce_n,
            'rearm_threshold_pct': threshold, 'rearm_min_duration_samples': min_dur, 'rearm_mismatch': mism,
            'per_cycle': per, 'reason': '; '.join(why) or None}


def check(model_dir: str, ts_ms: int, window_ms: int, ns: str, ric_pod: str, debounce_n: int | None = None,
          lease_s: float | None = None, threshold: float = 80.0, min_dur: int | None = None) -> dict:
    ws = windows(ts_ms, window_ms)
    ref = [features(w, ts_ms) for w in ws]
    pre = json.load(open(os.path.join(model_dir, 'preprocess.json')))
    names = pre['features']
    with tempfile.TemporaryDirectory() as d:  # offline prediction on the build.py features
        with open(os.path.join(d, 'x.csv'), 'w') as fh:
            fh.write(','.join(names) + ',label\n' + ''.join(','.join(repr(r[f]) for f in names) + ',0\n' for r in ref))
        p = subprocess.run([MLV, os.path.join(R1, 'mlengine', 'predict.py'), '--model-dir', model_dir, '--test',
                            os.path.join(d, 'x.csv'), '--out', os.path.join(d, 'pred.csv')], capture_output=True, text=True)
        if p.returncode:
            return {'result': 'FAIL', 'reason': 'offline predict failed: ' + p.stderr[-300:]}
        offline = [int(x) for x in open(os.path.join(d, 'pred.csv')).read().split()[1:]]
    wd = f'/tmp/r1-parity/{uuid.uuid4().hex[:8]}'
    try:
        for f, data in (('model.onnx', open(os.path.join(model_dir, 'model.onnx'), 'rb').read()),
                        ('preprocess.json', open(os.path.join(model_dir, 'preprocess.json'), 'rb').read()),
                        ('windows.csv', ''.join(','.join(repr(x) for x in w) + '\n' for w in ws).encode())):
            _ric_exec(ns, ric_pod, f'mkdir -p {wd} && cat > {wd}/{f}', data)
        out = _ric_exec(ns, ric_pod, f'{REPLAY} {wd} {ts_ms} {wd}/windows.csv 1')
    except RuntimeError as exc:
        return {'result': 'FAIL', 'reason': str(exc)}
    finally:
        subprocess.run(_exec_argv(ns, ric_pod) + ['rm', '-rf', wd], capture_output=True)
    xs = list(csv.DictReader(out.splitlines()))
    if len(xs) != len(ws):
        return {'result': 'FAIL', 'reason': f'xApp replay returned {len(xs)} rows for {len(ws)} windows'}
    used = [f for f in FEATURES if f in names]
    err = max(abs(float(x[f]) - r[f]) for x, r in zip(xs, ref) for f in used) if used else float('inf')
    preds, probas = [int(float(x['pred'])) for x in xs], [float(x['proba']) for x in xs]
    v = {'n': len(ws), 'features_checked': used, 'max_feature_abs_err': err,
         'unmatched_inputs': sum(int(x['unmatched_inputs']) for x in xs),
         'label_mismatch': sum(a != b for a, b in zip(preds, offline)), 'pred_min': min(preds), 'pred_max': max(preds),
         'proba_min': min(probas), 'proba_max': max(probas)}
    why = [m for m, bad in (('model inputs not computed by the xApp', v['unmatched_inputs'] or len(used) != len(names)),
                            (f'feature error > {FEATURE_TOL}', err > FEATURE_TOL),
                            ('xApp predictions differ from offline ONNX', v['label_mismatch']),
                            ('constant predictions', v['pred_min'] == v['pred_max'])) if bad]
    if lease_s is not None:
        try:
            v['lease'] = lease_replay(model_dir, ts_ms, window_ms, debounce_n or math.ceil(1000 / ts_ms), lease_s, ns,
                                      ric_pod, threshold, min_dur or math.ceil(3000 / ts_ms))
        except RuntimeError as exc:
            v['lease'] = {'result': 'FAIL', 'reason': str(exc)}
        if v['lease']['result'] != 'PASS':
            why.append('lease replay: ' + (v['lease']['reason'] or 'FAIL'))
    return {'result': 'FAIL' if why else 'PASS', **v, 'reason': '; '.join(why) or None}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--ts', type=int, default=500)
    ap.add_argument('--window', type=int, default=2000)
    ap.add_argument('--ns', default='tenant-test-1-space', help="'docker' = offline: --ric-pod is then the RIC image")
    ap.add_argument('--ric-pod', required=True)
    ap.add_argument('--debounce', type=int)
    ap.add_argument('--lease-s', type=float)
    ap.add_argument('--threshold', type=float, default=80.0)
    ap.add_argument('--min-duration-samples', type=int)
    a = ap.parse_args()
    print(json.dumps(check(a.model_dir, a.ts, a.window, a.ns, a.ric_pod, a.debounce, a.lease_s, a.threshold,
                           a.min_duration_samples), indent=1))


# rule self-check: a drain above the threshold does not re-arm; 3 samples below do; a release disarms again
assert rearm_reference([90, 90, 95, 70, 60, 50, 90, 90], ['reserve', '', 'release', '', '', '', 'reserve', 'release'], 80, 3) \
    == [1, 1, 0, 0, 0, 1, 1, 0]
