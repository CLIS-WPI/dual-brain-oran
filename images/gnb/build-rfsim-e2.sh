#!/usr/bin/env bash
# Overlay RFsim gNB with official OAI --build-e2, pinned to the live RIC
# (E2AP_V3 / KPM_V3_00). Does NOT load into minikube — run the pre-load
# grep yourself, then minikube image load + oai-rfsim tag only.
#
# Source = pinned OAI commit + Harmony patches (patches/oai:
# AURA DL PRB quota + E2 SLICE SM control). prepare-oai-src.sh builds that tree
# under .build/ — the OAI mirror checkout is never modified.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OAI_SRC="${OAI_SRC:-$ROOT/.build/oai-ran}"
IMAGE_BASE="${IMAGE_BASE:-localhost/harmony/oai-gnb:2026.w27-uhd410}"
IMAGE_OUT="${IMAGE_OUT:-localhost/harmony/oai-gnb:2026.w27-uhd410-e2-aura3}"
# OAI_BUILD_TARGET=SIMU (RFsim) | USRP (N310); see build-n310-e2.sh
OAI_BUILD_TARGET="${OAI_BUILD_TARGET:-SIMU}"
E2AP_VERSION="${E2AP_VERSION:-E2AP_V3}"
KPM_VERSION="${KPM_VERSION:-KPM_V3_00}"
CTR="${CTR:-harmony-oai-rfsim-e2-build}"
OVERLAY="${OVERLAY:-$ROOT/images/gnb/overlay-rfsim-e2}"
JOBS="${JOBS:-$(nproc)}"
DOCKERFILE="$ROOT/images/gnb/Dockerfile.rfsim-e2"

log() { printf '[build-rfsim-e2] %s\n' "$*"; }

if [[ "${SKIP_PREPARE:-0}" != "1" && "$OAI_SRC" == "$ROOT/.build/oai-ran" ]]; then
  OAI_BUILD_SRC="$OAI_SRC" "$ROOT/patches/oai/prepare-oai-src.sh"
fi

[[ -f "$OAI_SRC/cmake_targets/build_oai" ]] || { log "missing OAI_SRC=$OAI_SRC"; exit 1; }
[[ -f "$DOCKERFILE" ]] || { log "missing $DOCKERFILE"; exit 1; }

mkdir -p "$OVERLAY/libs" "$OVERLAY/flexric"

ensure_build_ctr() {
  if docker inspect "$CTR" >/dev/null 2>&1; then
    local src
    src="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/oai-ran"}}{{.Source}}{{end}}{{end}}' "$CTR")"
    if [[ "$src" != "$OAI_SRC" ]]; then
      log "remove $CTR (bound to $src, want $OAI_SRC)"
      docker rm -f "$CTR" >/dev/null
    fi
  fi
  if docker inspect "$CTR" >/dev/null 2>&1; then
    local st
    st="$(docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null || echo false)"
    if [[ "$st" == "true" ]]; then
      log "reuse running $CTR"
      return 0
    fi
    log "remove stopped $CTR"
    docker rm -f "$CTR" >/dev/null
  fi
  log "create $CTR from $IMAGE_BASE (bind $OAI_SRC)"
  docker run -d --name "$CTR" \
    --entrypoint sleep \
    -v "$OAI_SRC:/oai-ran" \
    -w /oai-ran \
    "$IMAGE_BASE" infinity
}

ensure_build_ctr

log "install build deps (idempotent) + FlexRIC submodule"
docker exec "$CTR" bash -lc '
  set -e
  export DEBIAN_FRONTEND=noninteractive
  git config --global --add safe.directory /oai-ran || true
  git config --global --add safe.directory /oai-ran/openair2/E2AP/flexric || true
  if ! command -v ninja >/dev/null || ! command -v cmake >/dev/null; then
    apt-get update
    apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build pkg-config git \
      libconfig-dev libsctp-dev libfftw3-dev libusb-1.0-0-dev \
      libblas-dev liblapack-dev libatlas-base-dev \
      libforms-dev libxpm-dev libssl-dev zlib1g-dev \
      libcap-dev \
      python3 ca-certificates wget
  fi
  if [[ ! -f /oai-ran/openair2/E2AP/flexric/CMakeLists.txt ]]; then
    cd /oai-ran
    git submodule update --init openair2/E2AP/flexric
  fi
'

CLEAN_FLAG=""
if [[ "${CLEAN:-0}" == "1" ]] || [[ ! -d "$OAI_SRC/cmake_targets/ran_build/build" ]]; then
  CLEAN_FLAG="-c"
  log "clean configure ($CLEAN_FLAG)"
fi

log "build_oai --gNB -w $OAI_BUILD_TARGET --build-e2 $E2AP_VERSION $KPM_VERSION --ninja $CLEAN_FLAG"
docker exec -e NUM_JOBS="$JOBS" "$CTR" bash -lc '
  set -e
  cd /oai-ran/cmake_targets
  if ! command -v asn1c >/dev/null 2>&1; then
    ./build_oai -I --install-optional-packages || true
  fi
  ./build_oai --gNB -w '"$OAI_BUILD_TARGET"' --build-e2 --ninja '"$CLEAN_FLAG"' \
    --cmake-opt -DE2AP_VERSION='"$E2AP_VERSION"' \
    --cmake-opt -DKPM_VERSION='"$KPM_VERSION"'
'

BIN="$OAI_SRC/cmake_targets/ran_build/build/nr-softmodem"
[[ -x "$BIN" ]] || { log "FAIL: no binary at $BIN"; exit 1; }

# Pre-load gate — refuse to tag an image without E2 symbols.
if ! grep -a -q near_ric_ip_addr "$BIN"; then
  log "FAIL: nr-softmodem has no near_ric_ip_addr — not tagging $IMAGE_OUT"
  exit 2
fi
log "pre-load grep OK: near_ric_ip_addr present in $BIN"
if ! grep -a -q "E2 slice add/mod" "$BIN"; then
  log "FAIL: nr-softmodem lacks the E2 SLICE → AURA patch — not tagging $IMAGE_OUT"
  exit 2
fi

cp -f "$BIN" "$OVERLAY/nr-softmodem"
BUILD_DIR="$OAI_SRC/cmake_targets/ran_build/build"
for f in librfsimulator.so libcoding.so libparams_libconfig.so libdfts.so \
         libparams_yaml.so libvrtsim.so libldpc.so libldpc_orig.so \
         liboai_usrpdevif.so; do
  [[ -f "$BUILD_DIR/$f" ]] && cp -f "$BUILD_DIR/$f" "$OVERLAY/libs/$f"
done

# SMs from the same FlexRIC submodule + E2AP/KPM pin (stock image SMs SIGSEGV).
log "build FlexRIC SMs $E2AP_VERSION $KPM_VERSION in $CTR"
docker exec "$CTR" bash -lc '
  set -e
  cd /oai-ran/openair2/E2AP/flexric
  mkdir -p build && cd build
  cmake -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DE2AP_VERSION='"$E2AP_VERSION"' -DKPM_VERSION='"$KPM_VERSION"' ..
  ninja
  ninja install
'
rm -rf "$OVERLAY/flexric"
mkdir -p "$OVERLAY/flexric"
docker cp "$CTR:/usr/local/lib/flexric/." "$OVERLAY/flexric/"

log "binary $(ls -lh "$OVERLAY/nr-softmodem" | awk '{print $5}') libs=$(ls "$OVERLAY/libs" 2>/dev/null | wc -l) flexric=$(ls "$OVERLAY/flexric" 2>/dev/null | wc -l)"

log "docker build → $IMAGE_OUT"
docker build \
  --build-arg "GNB_BASE_IMAGE=$IMAGE_BASE" \
  -f "$DOCKERFILE" \
  -t "$IMAGE_OUT" \
  "$OVERLAY"

log "OK: $IMAGE_OUT"
log "next (manual): minikube image load $IMAGE_OUT"
log "then: helm/kubectl set image on rfsim-gnb ONLY — do not retag USRP"
