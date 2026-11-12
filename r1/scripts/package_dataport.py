"""Package the raw data of all official R1 runs into one archive with a manifest (IEEE DataPort).

usage: package_dataport.py [--out r1/dataport/commag-r1-<date>.tar.gz] [--include-invalid]
Runs = rows of r1/results/NOTES.md whose status is complete/valid (invalid, aborted or test runs are listed in the
manifest as excluded, with the reason, unless --include-invalid). For each run directory under r1/results/<exp>/<run_id>/
the archive takes: telemetry (telemetry.csv, events.jsonl, gnb.log, meta.json, guard.jsonl, host_load.jsonl), raw ping
(ue3_ping.log), timing (timing.jsonl), summaries (*_summary.json, *.csv, split.json, *_metrics.json, report.json).
Excluded always: model binaries (*.onnx), ML-engine work dirs, crash dumps. Paths inside the archive keep the
results/ layout. manifest.json lists every file with size and sha256, plus run_id, experiment, start commit, status;
README.md describes the layout and the units (one unit per CSV column, named in the header).
"""
from __future__ import annotations

import argparse
import hashlib
import glob
import io
import json
import os
import re
import tarfile
import time

R1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(R1, 'results')
KEEP = re.compile(r'(telemetry\.csv|events\.jsonl|gnb\.log|meta\.json|guard\.jsonl|host_load\.jsonl|ue3_ping\.log|'
                  r'timing\.jsonl|.*_summary\.json|.*\.csv|split\.json|.*_metrics\.json|report\.json|agent_calls\.jsonl|'
                  r'run_meta\.json|spec\.json|recv\.json|gnb_aura\.log|xapp_events\.jsonl|control_events\.jsonl|'
                  r'verify_events\.jsonl|gate\.json|config\.json)$')  # Exp-04: e2e_breakdown input, gate record, rendered config
SKIP_DIRS = {'mlengine-runs', 'crash'}
# Unofficial calibration runs the paper relies on (results/calibration/NOTES.md): ramp profile + rise (P036), peak and
# UE3-demand profile checks, UE3 capacity + unloaded baselines, oracle check, backlog-onset analysis ("queue starts
# ~4 s before congestion"). Taken whole (all files), labelled experiment 'calibration'.
CALIBRATION = ['calibration/NOTES.md', 'calibration/backlog_onset.json', 'calibration/ramp-20261103-115429',
               'calibration/retune-20261103', 'calibration/ue3demand-20261103', 'calibration/ue3cap-20261103-164022',
               'calibration/ue3cap-20261103-172012', 'calibration/ue3cap-20261103-173110',
               'calibration/oracle-7-20261103-172124', 'calibration/oracle-8-20261103-173254',
               'calibration/calib-30_-10-20261102-191230']  # edge channel (UE3 MCS ~27 -> ~14)
EXTRA = {'results/exp04/exp04-20261105d-timing_summary.json': 'Exp-04 (batch timing summary, six official runs)',
         'results/exp04/e2e_breakdown.json': 'Exp-04 (end-to-end segment breakdown, six official runs)',
         'scripts/e2e_breakdown.py': 'Exp-04 (script that computes e2e_breakdown.json)'}  # paths relative to r1/
# Deployed models (Exp-03 arms: c = Exp-04 s1 artifact H 2.5 s, b = same model with threshold policy, d = H 7.5 s +
# its training data in d_build/): model.onnx, preprocess.json, config.json, manifest.json. Needed offline by
# experiments/exp01_eval_deployed.py and the gate's model-parity replay (zto/parity.py).
DEPLOYED = 'exp03/exp03-20261106-arms'


def official_runs():
    """[(experiment, run_id, commit, status, note)] from results/NOTES.md."""
    rows = []
    for ln in open(os.path.join(RESULTS, 'NOTES.md')):
        cells = [c.strip() for c in ln.strip().strip('|').split('|')]
        if len(cells) == 5 and cells[0] and not cells[0].startswith(('Experiment', '---')):
            rows.append(tuple(cells))
    return rows


def run_dir(exp, run_id):
    """results/<exp>/<run_id> or results/<exp>/<batch>/<run_id>; 'Exp-04 ...' -> exp04, exp01-select -> exp01."""
    base = re.sub(r'^exp-', 'exp', exp.split()[0].lower()).replace('exp01-select', 'exp01')
    hits = [d for d in [os.path.join(RESULTS, base, run_id)] + glob.glob(os.path.join(RESULTS, base, '*', run_id))
            if os.path.isdir(d)]
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(R1, 'dataport', time.strftime('commag-r1-%Y%m%d.tar.gz')))
    ap.add_argument('--include-invalid', action='store_true')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    manifest = {'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'runs': [], 'excluded': [], 'calibration': [], 'files': []}
    batches = set()
    with tarfile.open(a.out, 'w:gz') as tar:
        for exp, run_id, commit, status, note in official_runs():
            ok = not re.search(r'invalid|test only|aborted|failed|off-regime', status, re.I)
            d = run_dir(exp, run_id)
            if not d or not (ok or a.include_invalid):
                manifest['excluded'].append({'experiment': exp, 'run_id': run_id, 'status': status,
                                             'reason': 'no directory' if not d else 'not a valid run', 'note': note})
                continue
            manifest['runs'].append({'experiment': exp, 'run_id': run_id, 'start_commit': commit, 'status': status})
            for root, dirs, files in os.walk(d):
                dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
                for fn in sorted(files):
                    if not KEEP.match(fn):
                        continue
                    p = os.path.join(root, fn)
                    arc = os.path.relpath(p, R1)
                    data = open(p, 'rb').read()
                    manifest['files'].append({'path': arc, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                                              'run_id': run_id, 'experiment': exp})
                    tar.add(p, arcname=arc)
            # batch-level analysis files (e.g. exp03_cycles.csv, timing_summary.json) once, top level only
            parent = os.path.dirname(d)
            if os.path.dirname(parent) != RESULTS and parent not in batches:
                batches.add(parent)
                for fn in sorted(os.listdir(parent)):
                    p = os.path.join(parent, fn)
                    if os.path.isfile(p) and KEEP.match(fn):
                        data = open(p, 'rb').read()
                        manifest['files'].append({'path': os.path.relpath(p, R1), 'bytes': len(data),
                                                  'sha256': hashlib.sha256(data).hexdigest(), 'run_id': None,
                                                  'experiment': exp, 'batch': os.path.basename(parent)})
                        tar.add(p, arcname=os.path.relpath(p, R1))
        def add(p, exp, **kw):
            data = open(p, 'rb').read()
            manifest['files'].append({'path': os.path.relpath(p, R1), 'bytes': len(data),
                                      'sha256': hashlib.sha256(data).hexdigest(), 'run_id': None, 'experiment': exp, **kw})
            tar.add(p, arcname=os.path.relpath(p, R1))
        for rel in CALIBRATION:
            p = os.path.join(RESULTS, rel)
            assert os.path.exists(p), f'missing calibration item {rel}'
            manifest['calibration'].append(rel)
            for f in ([p] if os.path.isfile(p) else sorted(os.path.join(r, x) for r, _, fs in os.walk(p) for x in fs)):
                add(f, 'calibration', run=os.path.basename(p) if os.path.isdir(p) else None)
        for root, dirs, files in os.walk(os.path.join(RESULTS, DEPLOYED)):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
            for fn in sorted(files):
                add(os.path.join(root, fn), 'Exp-03 (deployed models)', run=os.path.basename(DEPLOYED))
        for rel, exp in EXTRA.items():
            add(os.path.join(R1, rel), exp)
        readme = ('# COMMAG-R1 Dual-Brain raw data\n\nLayout: results/<experiment>/<run_id>/ as in the repository '
                  '(r1/results). manifest.json lists every file (bytes, sha256), the runs included and the runs excluded '
                  'with the reason. CSV: header row, UTF-8, comma-separated, one unit per column named in the header '
                  '(_ms, _mbps, _pct). Times are wall-clock; the rfsimulator is not paced to real time, so every run '
                  'records its sim/wall speed (meta.json). ue3_ping.log: raw 10 Hz ping of the protected UE '
                  '(`ping -D`, epoch-stamped). Provenance (start commit, blob of every executed file) is in each '
                  'meta.json / *_summary.json. Files with experiment "calibration" are unofficial calibration runs '
                  '(profile, UE3 capacity, oracle check, backlog onset) the paper relies on; see '
                  'results/calibration/NOTES.md.\n')
        for name, blob in (('manifest.json', json.dumps(manifest, indent=1).encode()), ('README.md', readme.encode())):
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(blob), int(time.time())
            tar.addfile(info, io.BytesIO(blob))
    print(f"wrote {a.out}: {len(manifest['runs'])} runs, {len(manifest['files'])} files, "
          f"{len(manifest['excluded'])} excluded")


if __name__ == '__main__':
    main()
