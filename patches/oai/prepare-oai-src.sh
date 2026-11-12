#!/usr/bin/env bash
# Produce a clean, patched OAI source tree for Harmony gNB images.
#
#   OAI_BASE_COMMIT + FLEXRIC_SUBMODULE_COMMIT  (pinned, this directory)
#   + every NNNN-*.patch here, in order          (Harmony-owned changes)
#   → $OAI_BUILD_SRC (default <repo>/.build/oai-ran, gitignored)
#
# Works offline when OAI_MIRROR points at an existing OAI checkout (its
# objects and the FlexRIC submodule's objects are cloned locally); falls back
# to the upstream remotes in OAI_REMOTE otherwise. Never modifies the mirror.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OAI_MIRROR="${OAI_MIRROR:-}"
OAI_BUILD_SRC="${OAI_BUILD_SRC:-$ROOT/.build/oai-ran}"
BASE="$(cat "$HERE/OAI_BASE_COMMIT")"
FLEXRIC_BASE="$(cat "$HERE/FLEXRIC_SUBMODULE_COMMIT")"
FLEXRIC_PATH=openair2/E2AP/flexric

log() { printf '[prepare-oai-src] %s\n' "$*"; }

if [[ -d "$OAI_MIRROR/.git" ]]; then
  OAI_URL="$OAI_MIRROR"
  FLEXRIC_URL="$OAI_MIRROR/.git/modules/$FLEXRIC_PATH"
else
  OAI_URL="$(cat "$HERE/OAI_REMOTE")"
  FLEXRIC_URL="https://gitlab.eurecom.fr/mosaic5g/flexric.git"
fi

if [[ ! -d "$OAI_BUILD_SRC/.git" ]]; then
  log "clone $OAI_URL → $OAI_BUILD_SRC"
  mkdir -p "$(dirname "$OAI_BUILD_SRC")"
  git clone --quiet --no-checkout "$OAI_URL" "$OAI_BUILD_SRC"
fi

cd "$OAI_BUILD_SRC"
git fetch --quiet "$OAI_URL" "$BASE" 2>/dev/null || true
# Reset to the pinned base; drop any previous patch application.
git checkout --quiet --force --detach "$BASE"
git clean -fdq -e cmake_targets/ran_build -e "$FLEXRIC_PATH"

if [[ ! -d "$FLEXRIC_PATH/.git" && ! -f "$FLEXRIC_PATH/.git" ]]; then
  log "clone FlexRIC submodule from $FLEXRIC_URL"
  rm -rf "$FLEXRIC_PATH"
  git clone --quiet --no-checkout "$FLEXRIC_URL" "$FLEXRIC_PATH"
fi
git -C "$FLEXRIC_PATH" fetch --quiet "$FLEXRIC_URL" "$FLEXRIC_BASE" 2>/dev/null || true
git -C "$FLEXRIC_PATH" checkout --quiet --force --detach "$FLEXRIC_BASE"
git -C "$FLEXRIC_PATH" clean -fdq -e build

shopt -s nullglob
for p in "$HERE"/[0-9][0-9][0-9][0-9]-*.patch; do
  log "apply $(basename "$p")"
  git apply --whitespace=nowarn "$p"
done

log "OK: $OAI_BUILD_SRC @ ${BASE:0:10} + $(ls "$HERE"/[0-9][0-9][0-9][0-9]-*.patch | wc -l) patch(es)"
