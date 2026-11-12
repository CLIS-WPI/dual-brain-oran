#!/usr/bin/env bash
# Offline self-tests and reproduction checks (no testbed, no LLM, no NeuralSmith).
#   ./run_offline_checks.sh <commag-r1-YYYYMMDD.tar.gz>     (the IEEE DataPort archive)
# Env: PY   analysis python (r1/requirements-analysis.txt; default python3)
#      MLPY ML-engine python (r1/mlengine/requirements.lock; default r1/.venv-mlengine/bin/python)
#      RIC_IMAGE RIC image with cc_replay for the offline parity replay (default localhost/harmony/flexric:kpm-sd-dualbrain11)
#      CC   C compiler for the xApp unit tests (default gcc; skipped if missing)
# The archive is extracted into r1/ (its paths start with results/). Scripts that write next to their inputs
# run on that extracted copy; every recomputed value is compared with the archived one. Exit 0 only if all pass.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; R1="$ROOT/r1"; RES="$R1/results"
PY="${PY:-python3}"; MLPY="${MLPY:-$R1/.venv-mlengine/bin/python}"; CC="${CC:-gcc}"
ARCHIVE="${1:?usage: $0 <DataPort archive .tar.gz>}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0; SKIP=0
ok()   { echo "PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "FAIL  $1"; FAIL=$((FAIL+1)); }
skip() { echo "SKIP  $1"; SKIP=$((SKIP+1)); }
run()  { local name="$1"; shift; if "$@" >"$TMP/log" 2>&1; then ok "$name"; else bad "$name"; tail -5 "$TMP/log" | sed 's/^/      /'; fi; }

[ -d "$RES" ] || { mkdir -p "$R1" && tar -xzf "$ARCHIVE" -C "$R1"; }
mkdir -p "$TMP/ref"
cp "$RES/exp03/exp03-20261106b/exp03_summary.json" "$RES/exp03/exp03-20261106b/exp03_cycles.csv" \
   "$RES/exp04/e2e_breakdown.json" "$RES/exp04/exp04-20261105d-timing_summary.json" "$R1/paper/cycle_timeline.json" "$TMP/ref/"
for m in neuralsmith flaml persistence; do cp "$RES/exp01-eval/exp01-eval-20261104/pooled/h2500/${m}_metrics.json" "$TMP/ref/${m}_pooled_h2500.json"; done

# same(<a.json> <b.json> [ignored top-level keys...]): equal after dropping volatile keys
same() { "$PY" - "$@" <<'EOF'
import json, sys
a, b, ign = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), set(sys.argv[3:])
def strip(d):
    d = {k: v for k, v in d.items() if k not in ign}
    if isinstance(d.get('runs'), list):  # runs not in the archive (failed/invalid, by policy) are listed only locally
        d['runs'] = [r for r in d['runs'] if not isinstance(r, dict) or r.get('status') == 'complete']
    return d
sys.exit(0 if strip(a) == strip(b) else (print('differs:', sorted(k for k in set(strip(a)) | set(strip(b))
                                                               if strip(a).get(k) != strip(b).get(k))) or 1))
EOF
}

echo "== unit self-tests"
run "speed_rules unit tests"            "$PY" "$R1/collector/test_speed_rules.py"
run "timing marks self-test"            "$PY" "$R1/common/timing.py"
run "traffic schedule self-check"       "$PY" "$R1/traffic/burst.py" plan --profile ramp --peak 28 --cycles 2 --seed 6
run "exp03_analyze synthetic self-test" "$PY" "$R1/experiments/exp03_analyze.py" --self-test
run "ZTO self-checks (schema/render/gate/agent import, frozen prompt hashes, slots bounded)" "$PY" - "$R1" <<'EOF'
import importlib, json, os, subprocess, sys
r1 = sys.argv[1]; sys.path.insert(0, os.path.join(r1, 'zto'))
for m in ('render', 'resolve', 'gate', 'agent'):
    importlib.import_module(m)  # module-level self-checks
import agent
lock = json.load(open(os.path.join(r1, 'zto', 'prompts', 'lock.json')))
bad = {m: agent.prompt_sha256(m) for m, h in lock['sha256'].items() if agent.prompt_sha256(m) != h}
slots = json.loads(subprocess.run([sys.executable, os.path.join(r1, 'scripts', 'list_slots.py'), '--json'],
                                  capture_output=True, text=True, check=True).stdout)
print('prompt hashes', 'match lock.json' if not bad else bad, '| slots', slots['n_slots'], slots['errors'])
sys.exit(1 if bad or slots['errors'] else 0)
EOF
if command -v "$CC" >/dev/null; then
  for t in cc_policy_test cc_assoc_test; do
    run "xApp C unit test $t" bash -c "'$CC' -O2 -o '$TMP/$t' '$ROOT/xapp/dual_brain/$t.c' -lm && '$TMP/$t'"
  done
else skip "xApp C unit tests (no $CC)"; fi

echo "== reproduction against the DataPort archive"
B="$RES/exp03/exp03-20261106b"
run "Exp-03 analysis rerun"                  "$PY" "$R1/experiments/exp03_analyze.py" "$B"
run "Exp-03 summary == archived"             same "$B/exp03_summary.json" "$TMP/ref/exp03_summary.json" generated git_commit
run "Exp-03 per-cycle table == archived"     cmp "$B/exp03_cycles.csv" "$TMP/ref/exp03_cycles.csv"
run "Fig. 3 (cycle timeline) rerun"          "$PY" "$R1/paper/fig3_cycle_timeline.py" "$B"
run "Fig. 3 medians == archived"             same "$R1/paper/cycle_timeline.json" "$TMP/ref/cycle_timeline.json"
run "Fig. 4 (edge-UE outcome) rerun"         "$PY" "$R1/paper/fig4_edge_ue_outcome.py" "$B"
run "Exp-04 end-to-end breakdown rerun"      "$PY" "$R1/scripts/e2e_breakdown.py"
run "Exp-04 breakdown == archived"           same "$RES/exp04/e2e_breakdown.json" "$TMP/ref/e2e_breakdown.json"
E4=$(ls -d "$RES"/exp04/exp04-20261105d-s*/)
run "Exp-04 timing summary rerun"            "$PY" "$R1/scripts/timing_report.py" $E4 --out "$TMP/timing.json" --group-by collection_min
run "Exp-04 timing summary == archived"      same "$TMP/timing.json" "$TMP/ref/exp04-20261105d-timing_summary.json" generated runs_dirs
P="$RES/exp01-eval/exp01-eval-20261104/pooled/h2500"   # the split the paper reports: exp01-eval, 2 runs pooled, 26 onsets
for m in neuralsmith flaml persistence; do
  pred="$P/$m/pred.csv"; [ "$m" = persistence ] && pred=persistence
  run "exp01-eval pooled $m scoring rerun"        "$PY" "$R1/dataset/evaluate.py" --dir "$P" --pred "$pred" --out "check_$m.json"
  run "exp01-eval pooled $m metrics == archived"  same "$P/check_$m.json" "$TMP/ref/${m}_pooled_h2500.json" predictor  # predictor = pred path
done
"$PY" - "$P" <<'EOF'
import json, sys
for m in ('neuralsmith', 'flaml', 'persistence'):
    x = json.load(open(f'{sys.argv[1]}/check_{m}.json')); s = x['secondary']
    print(f"      {m:11} exp01-eval pooled: onsets {x['n_onsets']} onset recall {x['onset_recall']} acc {s['accuracy']} "
          f"prec {s['precision']} rec {s['recall']} F1 {s['f1']} FPR {s['fpr']} confusion {s['confusion']}")
EOF
D="$RES/exp01/exp01-20261104/ts500_h2500"
run "exp01-select dataset rebuild (T_s 500, H 2.5 s)" "$PY" "$R1/dataset/build.py" \
    --in "$RES/exp01/exp01-20261104/raw/exp01-20261104-ts500/telemetry.csv" --ts 500 --window 2000 --horizon 2500 \
    --threshold 80 --min-duration 3 --train-cycles 1-9 --test-cycles 10-13 --out "$TMP/build"
for f in dataset.csv train.csv test.csv split.json; do run "rebuilt $f == archived" cmp "$TMP/build/$f" "$D/$f"; done
RIC_IMAGE="${RIC_IMAGE:-localhost/harmony/flexric:kpm-sd-dualbrain11}"
if [ -x "$MLPY" ]; then
  [ -e "$R1/.venv-mlengine" ] || ln -s "$(dirname "$(dirname "$MLPY")")" "$R1/.venv-mlengine"  # scripts call r1/.venv-mlengine/bin/python
  # FLAML retrain: its time-budgeted search (budget = NeuralSmith's training time, 12.2 s) is not bit-reproducible,
  # so the retrained model's scores are printed next to the archived ones, not asserted equal.
  run "FLAML retrain on exp01-select train (cycles 1-9), scored on exp01-eval pooled" "$MLPY" - "$R1" "$D" "$P" "$TMP/flaml" <<'EOF'
import json, os, shutil, subprocess, sys
r1, d, p, out = sys.argv[1:5]; os.makedirs(f'{out}/model'); os.makedirs(f'{out}/eval'); sys.path.insert(0, f'{r1}/mlengine')
import flaml_backend
flaml_backend.train(f'{d}/train.csv', f'{d}/test.csv', json.load(open(f'{d}/flaml/spec.json')), f'{out}/model')
for f in ('test.csv', 'test_index.csv', 'meta.json'):
    shutil.copy(f'{p}/{f}', f'{out}/eval/{f}')
subprocess.run([sys.executable, f'{r1}/mlengine/predict.py', '--model-dir', f'{out}/model', '--test', f'{out}/eval/test.csv',
                '--out', f'{out}/eval/pred.csv'], check=True, capture_output=True)
subprocess.run([sys.executable, f'{r1}/dataset/evaluate.py', '--dir', f'{out}/eval', '--pred', f'{out}/eval/pred.csv',
                '--out', 'm.json'], check=True, capture_output=True)
for name, path in (('archived', f'{p}/flaml_metrics.json'), ('retrained', f'{out}/eval/m.json')):
    x = json.load(open(path)); s = x['secondary']
    print(f"{name:9} onsets {x['n_onsets']} onset recall {x['onset_recall']} lead {x['lead_ms_median']} ms acc {s['accuracy']} "
          f"prec {s['precision']} rec {s['recall']} F1 {s['f1']} FPR {s['fpr']} confusion {s['confusion']}")
EOF
  sed 's/^/      /' "$TMP/log"
  E="$RES/exp01-eval/exp01-eval-20261104-deployed"; cp "$E/deployed_summary.json" "$TMP/ref/deployed_summary.json"
  run "deployed models (Exp-03 arms c, d) scored on exp01-eval" "$PY" "$R1/experiments/exp01_eval_deployed.py"
  run "deployed-model scores == archived"     same "$E/deployed_summary.json" "$TMP/ref/deployed_summary.json" git_commit
  run "lease derived from the selection data" "$MLPY" -c "import sys; sys.path.insert(0, '$R1/zto'); import lease; print(lease.compute('$D', 500, 2500))"
  if command -v docker >/dev/null && docker image inspect "$RIC_IMAGE" >/dev/null 2>&1; then
    for arm in c d; do
      run "model-parity + lease/re-arm replay, arm ($arm), offline in $RIC_IMAGE == gate record" "$PY" - "$R1" "$arm" "$RIC_IMAGE" <<'EOF'
import json, os, sys
r1, arm, img = sys.argv[1:4]; sys.path.insert(0, os.path.join(r1, 'zto')); import parity
a = os.path.join(r1, 'results', 'exp03', 'exp03-20261106-arms'); man = json.load(open(os.path.join(a, 'manifest.json')))
cfg = json.load(open(os.path.join(a, arm, 'config.json')))
rec = man['arm_c_gate']['model_parity'] if arm == 'c' else man['d_model']['model_parity']
v = parity.check(os.path.join(a, arm), cfg['kpm_period_ms'], cfg['window_samples'] * cfg['kpm_period_ms'], 'docker', img,
                 debounce_n=cfg['trigger_consecutive'], lease_s=cfg['lease_s'], threshold=cfg['threshold_cell_prb'],
                 min_dur=cfg['min_duration_samples'])
print({k: v.get(k) for k in ('result', 'n', 'max_feature_abs_err', 'label_mismatch')}, 'lease', v['lease']['result'])
sys.exit(0 if v == rec and v['result'] == 'PASS' else 1)
EOF
    done
  else skip "model-parity replay (needs docker + the RIC image $RIC_IMAGE, built from images/ric)"; fi
else skip "FLAML / deployed-model / parity checks (no ML-engine venv at $MLPY)"; fi

echo "== $PASS passed, $FAIL failed, $SKIP skipped"
[ "$FAIL" -eq 0 ]
