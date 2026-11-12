#!/usr/bin/env bash
# Parity test: xApp C model path vs NeuralSmith on a held-out file.
#   1. train a bundle with the NeuralSmith service image (offline: --network none)
#   2. run cc_model_test (the xApp's own preprocess + ONNX code) in the RIC image
# Passes only if every held-out prediction matches NeuralSmith's and p99 < 5 ms.
#   BUNDLE=<dir> skips step 1 (e.g. a bundle downloaded from /models/latest).
set -euo pipefail
NS_IMAGE="${NS_IMAGE:-localhost/harmony/neuralsmith:1.2.1}"
RIC_IMAGE="${RIC_IMAGE:-localhost/harmony/flexric:kpm-sd-dualbrain}"
WORK="${WORK:-$(mktemp -d)}"
DATA="${DATA:-}"

if [[ -z "${BUNDLE:-}" ]]; then
  mkdir -p "$WORK/out"
  data_mount=()
  data_arg=/app/selftest/selftest.csv
  if [[ -n "$DATA" ]]; then
    data_mount=(-v "$(cd "$(dirname "$DATA")" && pwd):/in:ro")
    data_arg="/in/$(basename "$DATA")"
  fi
  docker run --rm -i --network none --cpus 8 -u "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$WORK/out:/out" "${data_mount[@]}" "$NS_IMAGE" python - "$data_arg" <<'EOF'
import json, sys
from pathlib import Path
from service import export
r = export.train_bundle(Path(sys.argv[1]), Path("/out"), target="congested", latency_budget_ms=5.0,
                        mode="fast++", cpus=8, holdout_frac=0.2, seed=11, positive_label="1", timeout_s=1800)
print(json.dumps({k: r[k] for k in ("algorithm", "accuracy", "f1", "inference_latency_ms", "n_holdout")}))
EOF
  BUNDLE="$WORK/out/bundle"
fi

docker run --rm --network none -v "$(cd "$BUNDLE" && pwd):/b:ro" --entrypoint \
  /usr/local/flexric/xApp/c/dual_brain/cc_model_test "$RIC_IMAGE" /b 1 1e-4
echo "parity OK ($BUNDLE)"
