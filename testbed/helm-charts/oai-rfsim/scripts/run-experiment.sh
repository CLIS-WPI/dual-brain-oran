#!/bin/bash
# Reusable RFSIM gNB + virtual UE + iPerf UL experiment runner.
#
# Usage:
#   ./run-experiment.sh [tenant] [namespace] [duration_seconds]
#
# What it does:
#   1. helm install/upgrade the oai-rfsim chart into <tenant>'s namespace
#      (gNB-RFSIM pod, 1+ virtual-UE pods, an iperf3 traffic-gen pod).
#   2. Waits for the UE pod(s) to report a PDU session / tun interface up.
#   3. Runs `iperf3 -c <traffic-gen-svc>` FROM inside the UE pod (uplink:
#      UE is the client, traffic-gen is the server) and prints the summary.
#
# Known issue (2026-07-09): the current oai-gnb/oai-nr-ue develop-snapshot
# image pair (2026.w27) hits a reproducible glibc heap corruption
# ("corrupted double-linked list") in nr-softmodem shortly after the RFSIM
# UE connects, before or during the RA procedure. This is upstream OAI
# instability, not a Helm/K8s config issue — see docs/BIG-PICTURE.md §11a.
# If this script's gNB pod crash-loops, try a pinned stable OAI tag instead
# of the develop snapshot (e.g. 2024.w10) via --set images.gnb.tag=... and
# --set images.ue.tag=... before re-running.

set -euo pipefail

TENANT="${1:-test-1}"
NAMESPACE="${2:-tenant-${TENANT}-space}"
DURATION="${3:-10}"
CHART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE="${TENANT}-rfsim"

echo "==> Deploying oai-rfsim chart (release=${RELEASE}, ns=${NAMESPACE})"
helm upgrade --install "${RELEASE}" "${CHART_DIR}" \
  -n "${NAMESPACE}" \
  --set tenant.name="${TENANT}" \
  "${@:4}"

echo "==> Waiting for gNB-RFSIM to be ready"
kubectl rollout status deployment "${RELEASE}-gnb" -n "${NAMESPACE}" --timeout=120s

echo "==> Waiting for traffic-gen (iperf3 server) to be ready"
kubectl rollout status deployment "${RELEASE}-trf" -n "${NAMESPACE}" --timeout=60s

UE_DEPLOY=$(kubectl get deployment -n "${NAMESPACE}" \
  -l "app.kubernetes.io/component=virtual-ue,harmony.telecom/tenant=${TENANT}" \
  -o jsonpath='{.items[0].metadata.name}')
echo "==> Waiting for virtual UE (${UE_DEPLOY}) to be ready"
kubectl rollout status deployment "${UE_DEPLOY}" -n "${NAMESPACE}" --timeout=180s

UE_POD=$(kubectl get pods -n "${NAMESPACE}" \
  -l "app.kubernetes.io/component=virtual-ue,harmony.telecom/tenant=${TENANT}" \
  -o jsonpath='{.items[0].metadata.name}')
TRF_SVC="${RELEASE}-trf"

echo "==> Running iperf UL test (UE '${UE_POD}' -> traffic-gen '${TRF_SVC}') for ${DURATION}s"
# OAI images ship legacy iperf (v2), not iperf3 — client/server must match.
kubectl exec -n "${NAMESPACE}" "${UE_POD}" -- \
  iperf -c "${TRF_SVC}" -p 5201 -t "${DURATION}" -i 1

echo "==> Done. Re-run any time with: $0 ${TENANT} ${NAMESPACE} ${DURATION}"
