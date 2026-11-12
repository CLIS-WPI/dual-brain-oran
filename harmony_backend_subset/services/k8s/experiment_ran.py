"""Per-experiment RAN provisioning (``ran_mode = 'isolated'`` experiments).

Builds an ``OaiExperiment`` (harmony.telecom/v1) from a Canvas experiment and lets
the Operator install it in-cluster with its Helm SDK — no helm on the host:

* ``<tenant>-<slug>-rfsim``: gNB (own gNB ID / cell ID / PCI, carrier profile from
  the Canvas gNB node) + one Virtual UE per IMSI + traffic generator.
* ``<tenant>-<slug>-ric``: FlexRIC + xApps, when the canvas has a FlexRIC node.

The tenant's 5G Core is shared. Current replica counts are sent with every
update so a Helm upgrade never starts or stops a component behind the Canvas.
Sizing is configuration (``HARMONY_EXPERIMENT_SIZING`` JSON), not a constant, and a
preflight rejects Run with a clear message instead of leaving pods Pending.
See docs/specs/experiment-isolation.md.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
from typing import Any

from services.k8s.core_bridge import _kubectl_json, get_k8s_core_config
from services.k8s.experiment_ctx import EXPERIMENT_LABEL, ExperimentRef, slug_for

logger = logging.getLogger(__name__)

TESTBED_NAMESPACE = os.getenv("HARMONY_TESTBED_NAMESPACE", "harmony-core")
LEGACY_GNB_ID = 0xE00  # the shared <tenant>-rfsim gNB (oai-rfsim default)

# Vetted, self-consistent carrier settings (band 78, 30 kHz SCS). A PRB change moves
# the SSB / Point A / initial BWP and the UE centre frequency together.
CARRIER_PROFILES: dict[str, dict[str, Any]] = {
    "b78-51prb": {
        "label": "Band 78 · 51 PRB (20 MHz)",
        "gnb": {
            "band": 78, "dlCarrierBandwidth": 51,
            "dlAbsoluteFrequencyPointA": 639996, "absoluteFrequencySSB": 640704,
            "initialDLBWPlocationAndBandwidth": 13750, "initialDLBWPcontrolResourceSetZero": 12,
            "ueCenterFrequencyHz": "3609120000", "ueSsbOffset": 234,
        },
    },
    "b78-106prb": {
        "label": "Band 78 · 106 PRB (40 MHz)",
        "gnb": {
            "band": 78, "dlCarrierBandwidth": 106,
            "dlAbsoluteFrequencyPointA": 640008, "absoluteFrequencySSB": 641280,
            "initialDLBWPlocationAndBandwidth": 28875, "initialDLBWPcontrolResourceSetZero": 12,
            "ueCenterFrequencyHz": "3619200000", "ueSsbOffset": 516,
        },
    },
}
DEFAULT_CARRIER_PROFILE = "b78-51prb"

# Default sizing per component (Kubernetes quantities). Measured on the reference lab
# (51 PRB, 2 UEs): gNB ≈ 2.3 CPU, UE ≈ 0.9 CPU. Override per installation with
# HARMONY_EXPERIMENT_SIZING='{"b78-106prb": {"gnb": {"limits": {"cpu": "8"}}}, "ue": …}'.
_DEFAULT_SIZING: dict[str, Any] = {
    "b78-51prb": {"gnb": {"requests": {"cpu": "2", "memory": "1Gi"}, "limits": {"cpu": "4", "memory": "4Gi"}}},
    "b78-106prb": {"gnb": {"requests": {"cpu": "3", "memory": "1Gi"}, "limits": {"cpu": "6", "memory": "4Gi"}}},
    "ue": {"requests": {"cpu": "500m", "memory": "512Mi"}, "limits": {"cpu": "1500m", "memory": "2Gi"}},
}


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def sizing() -> dict[str, Any]:
    raw = os.getenv("HARMONY_EXPERIMENT_SIZING", "").strip()
    if not raw:
        return _DEFAULT_SIZING
    try:
        return _deep_merge(_DEFAULT_SIZING, json.loads(raw))
    except ValueError:
        logger.warning("HARMONY_EXPERIMENT_SIZING is not valid JSON — using defaults")
        return _DEFAULT_SIZING


class ExperimentRanError(RuntimeError):
    """A provisioning problem the portal should show to the user verbatim."""


# ── experiments & canvas ─────────────────────────────────────────────────────

async def load_experiment(experiment_id: str) -> dict[str, Any] | None:
    from database import db_conn

    async with db_conn() as db:
        async with db.execute(
            "SELECT id, name, canvas_data, ran_mode FROM experiments WHERE id = ?", (experiment_id,)
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return None
    try:
        graph = json.loads(row["canvas_data"] or "{}")
    except ValueError:
        graph = {}
    return {"id": row["id"], "name": row["name"], "ran_mode": row["ran_mode"], "graph": graph}


def ref_for(experiment: dict[str, Any] | None) -> ExperimentRef | None:
    """Isolated experiments get a context; legacy ones use the shared stack."""
    if not experiment or experiment.get("ran_mode") != "isolated":
        return None
    return ExperimentRef(experiment_id=experiment["id"], slug=slug_for(experiment["id"]))


_ref_cache: dict[str, tuple[float, ExperimentRef | None]] = {}


async def resolve_ref(experiment_id: str) -> ExperimentRef | None:
    """ExperimentRef for an isolated experiment id, else None (cached 30 s)."""
    import time

    experiment_id = str(experiment_id or "").strip()
    if not experiment_id:
        return None
    hit = _ref_cache.get(experiment_id)
    if hit and time.monotonic() - hit[0] < 30:
        return hit[1]
    ref = ref_for(await load_experiment(experiment_id))
    _ref_cache[experiment_id] = (time.monotonic(), ref)
    return ref


def _node_type(node: dict[str, Any]) -> str:
    data = node.get("data") or {}
    return str(data.get("nodeType") or data.get("type") or node.get("type") or "")


def _nodes(graph: dict[str, Any], node_type: str) -> list[dict[str, Any]]:
    return [n for n in (graph.get("nodes") or []) if _node_type(n) == node_type]


# Dual-Brain needs the Harmony AURA gNB (patches 0001-0004: SLICE SM -> per-slice PRB quota,
# [AURA-PRB] ground truth). The chart default is the stock E2 image; MILCOM's legacy release
# carried this tag only as a hand-set Helm override. Built by scripts/build-rfsim-e2.sh.
DUAL_BRAIN_NODE_TYPES = {"zto_agent", "congestion_xapp"}
# Traffic-generator resources per experiment (keyed on the experiment id; others keep the chart default).
# COMMAG-R1: its 20-min runs open ~600 reverse-UDP iperf flows; the 256 Mi default was OOM-killed 3x (2026-11-03).
TRAFFIC_GEN_RESOURCES = json.loads(os.getenv("HARMONY_TRAFFIC_GEN_RESOURCES", "null") or "null") or {
    "ce70e6c7-ad83-464a-92bd-8902961deb5e": {"requests": {"cpu": "250m", "memory": "128Mi"},
                                             "limits": {"cpu": "2", "memory": "1Gi"}},
}
DUAL_BRAIN_GNB_IMAGE = {"registry": "localhost/harmony", "repository": "oai-gnb",
                        "tag": os.getenv("HARMONY_DUAL_BRAIN_GNB_TAG", "2026.w27-uhd410-e2-aura3")}
# ...and the RIC image that carries the congestion xApp + ONNX Runtime (chart default is plain
# flexric:latest). Built by scripts/build-dualbrain.sh.
DUAL_BRAIN_RIC_IMAGE = {"registry": "localhost/harmony", "repository": "flexric",
                        "tag": os.getenv("HARMONY_DUAL_BRAIN_RIC_TAG", "kpm-sd-dualbrain8")}
# Per-experiment RIC tag (keyed on the experiment id; others keep the default above).
# COMMAG-R1: dualbrain9 = R1 builder features (names, slope %/s); dualbrain10 = + lease release, 2026-11-04; dualbrain11 = + re-arm after release, 2026-11-05.
DUAL_BRAIN_RIC_TAG_BY_EXPERIMENT = json.loads(os.getenv("HARMONY_DUAL_BRAIN_RIC_TAGS", "null") or "null") or {
    "ce70e6c7-ad83-464a-92bd-8902961deb5e": "kpm-sd-dualbrain11",
}


def ue_channel(cfg: dict[str, Any]) -> dict[str, float] | None:
    """Virtual UE node rfsim DL channel (cell-edge UE, UE-side model only): {ploss_dB, noise_power_dB} or None."""
    ploss, noise = cfg.get("channel_ploss_db"), cfg.get("channel_noise_db")
    if ploss in (None, "") and noise in (None, ""):
        return None
    try:
        out = {"ploss_dB": float(ploss or 0), "noise_power_dB": float(-50 if noise in (None, "") else noise)}
    except (TypeError, ValueError) as exc:
        raise ExperimentRanError(f"Virtual UE channel settings must be numbers (dB): {exc}") from exc
    if not (0 <= out["ploss_dB"] <= 60 and -100 <= out["noise_power_dB"] <= 20):
        raise ExperimentRanError("Virtual UE channel: path loss 0..60 dB, noise power -100..20 dB")
    return out


def canvas_ue_instances(graph: dict[str, Any], extra_imsis: list[str] = ()) -> list[dict[str, Any]]:
    """One ue.instances entry per IMSI across all Virtual UE nodes (stable names)."""
    from services.k8s.ran_bridge import build_rfsim_ue_instances

    by_imsi: dict[str, dict[str, Any]] = {}
    for imsi in extra_imsis:
        if len(str(imsi)) == 15 and str(imsi).isdigit():
            by_imsi[str(imsi)] = build_rfsim_ue_instances(str(imsi), 1)[0]
    for node in _nodes(graph, "virtual_ue"):
        cfg = (node.get("data") or {}).get("config") or {}
        base = str(cfg.get("imsi") or "").strip()
        if len(base) != 15 or not base.isdigit():
            continue
        for inst in build_rfsim_ue_instances(base, int(cfg.get("count") or 1)):
            inst["dnn"] = str(cfg.get("dnn") or inst["dnn"])
            inst["sst"] = int(cfg.get("sst") or inst["sst"])
            inst["sd"] = int(cfg.get("sd") or inst["sd"])
            channel = ue_channel(cfg)
            if channel:
                inst["channel"] = channel
            by_imsi.setdefault(inst["imsi"], inst)
    return [dict(by_imsi[imsi], name=f"ue{i + 1}") for i, imsi in enumerate(sorted(by_imsi))]


# ── cluster state ────────────────────────────────────────────────────────────

async def _list_experiments_cr() -> list[dict[str, Any]]:
    try:
        data = await _kubectl_json(["get", "oaiexperiments.harmony.telecom", "-n", TESTBED_NAMESPACE])
    except RuntimeError:
        return []
    return data.get("items") or []


def _cr_name(slug: str) -> str:
    return f"exp-{slug}"


async def allocate_identity(slug: str, tenant: str) -> dict[str, Any]:
    """Unique gNB ID / NR cell ID / PCI per experiment (reused once allocated)."""
    used: set[int] = {LEGACY_GNB_ID}
    for item in await _list_experiments_cr():
        spec = item.get("spec") or {}
        gnb = ((spec.get("rfsim") or {}).get("gnb") or {})
        try:
            gid = int(str(gnb.get("gnbId", "0")), 0)
        except ValueError:
            continue
        if spec.get("slug") == slug and spec.get("testbed") == tenant:
            return {"gnbId": gnb.get("gnbId"), "nrCellId": gnb.get("nrCellId"), "physCellId": gnb.get("physCellId")}
        used.add(gid)
    n = 1
    while LEGACY_GNB_ID + n in used:
        n += 1
    return {"gnbId": hex(LEGACY_GNB_ID + n), "nrCellId": f"{12345678 + n}L", "physCellId": n % 1008}


async def live_replicas(tenant: str, slug: str) -> dict[str, int]:
    cfg = get_k8s_core_config()
    try:
        data = await _kubectl_json([
            "get", "deployments", "-n", cfg["namespace"],
            "-l", f"harmony.telecom/tenant={tenant},{EXPERIMENT_LABEL}={slug}",
        ])
    except RuntimeError:
        return {}
    return {
        (i.get("metadata") or {}).get("name", ""): int((i.get("spec") or {}).get("replicas") or 0)
        for i in data.get("items") or []
    }


# ── spec ─────────────────────────────────────────────────────────────────────

def build_spec(
    experiment: dict[str, Any],
    ref: ExperimentRef,
    tenant: str,
    identity: dict[str, Any],
    live: dict[str, int],
    *,
    start: set[str] = frozenset(),
    extra_imsis: list[str] = (),
) -> dict[str, Any]:
    """OaiExperiment manifest. ``start`` = components to run now: 'gnb', 'ric', 'ue:<imsi>'."""
    graph = experiment.get("graph") or {}
    gnb_nodes = _nodes(graph, "gnb_rfsim")
    gnb_cfg = ((gnb_nodes[0].get("data") or {}).get("config") or {}) if gnb_nodes else {}
    profile_key = str(gnb_cfg.get("carrier_profile") or DEFAULT_CARRIER_PROFILE)
    if profile_key not in CARRIER_PROFILES:
        raise ExperimentRanError(
            f"Unknown carrier profile {profile_key!r}; choose one of: " + ", ".join(CARRIER_PROFILES)
        )
    size = sizing()
    base = f"{tenant}-{ref.slug}-rfsim"

    gnb = dict(CARRIER_PROFILES[profile_key]["gnb"])
    gnb.update(identity)
    gnb["gnbName"] = f"gNB-{ref.slug}"
    gnb["resources"] = copy.deepcopy((size.get(profile_key) or {}).get("gnb") or {})
    if any(_nodes(graph, t) for t in DUAL_BRAIN_NODE_TYPES):
        # MILCOM's gNB carried this as a hand edit (kubectl set env): crash backtraces via gdb.
        gnb["extraEnv"] = [{"name": "OAI_GDBSTACKS", "value": "1"}]
    gnb["replicas"] = 1 if "gnb" in start else live.get(f"{base}-gnb", 0)

    instances = []
    for inst in canvas_ue_instances(graph, extra_imsis):
        dep = f"{base}-ue-{inst['name']}"
        inst["replicas"] = 1 if f"ue:{inst['imsi']}" in start else live.get(dep, 0)
        instances.append(inst)

    rfsim: dict[str, Any] = {
        **({"images": {"gnb": dict(DUAL_BRAIN_GNB_IMAGE)}}
           if any(_nodes(graph, t) for t in DUAL_BRAIN_NODE_TYPES) else {}),
        "gnb": gnb,
        "ue": {"instances": instances, "resources": copy.deepcopy(size.get("ue") or {})},
        "e2Agent": {"enabled": bool(gnb_cfg.get("enable_e2_agent", True))},
        "e3Agent": {"prbTotal": gnb["dlCarrierBandwidth"]},
    }
    if ref.experiment_id in TRAFFIC_GEN_RESOURCES:
        rfsim["trafficGen"] = {"resources": copy.deepcopy(TRAFFIC_GEN_RESOURCES[ref.experiment_id])}
    spec: dict[str, Any] = {
        "testbed": tenant, "slug": ref.slug, "experimentId": ref.experiment_id, "rfsim": rfsim,
    }
    if _nodes(graph, "flexric"):
        ric_replicas = 1 if "ric" in start else live.get(f"{tenant}-{ref.slug}-ric", 0)
        ric_image = dict(DUAL_BRAIN_RIC_IMAGE)
        if ref.experiment_id in DUAL_BRAIN_RIC_TAG_BY_EXPERIMENT:
            ric_image["tag"] = DUAL_BRAIN_RIC_TAG_BY_EXPERIMENT[ref.experiment_id]
        spec["ric"] = {**({"image": ric_image}
                           if any(_nodes(graph, t) for t in DUAL_BRAIN_NODE_TYPES) else {}), "ric": {
            "replicas": ric_replicas,
            "kpm": {"nssaiSst": str(instances[0]["sst"]) if instances else "1",
                    "nssaiSd": f"{int(instances[0]['sd']) if instances else 1:06x}"},
        }}
    return {
        "apiVersion": "harmony.telecom/v1", "kind": "OaiExperiment",
        "metadata": {"name": _cr_name(ref.slug), "namespace": TESTBED_NAMESPACE,
                     "labels": {EXPERIMENT_LABEL: ref.slug}},
        "spec": spec,
    }


# ── preflight ────────────────────────────────────────────────────────────────

def _cpu_to_milli(q: str) -> int:
    q = str(q).strip()
    return int(float(q[:-1])) if q.endswith("m") else int(float(q) * 1000)


def spec_cpu_limits_milli(spec: dict[str, Any], *, ric_cpu_milli: int = 1000, sidecars_milli: int = 1200) -> int:
    """Upper bound of CPU limits the experiment needs when running (quota check)."""
    rfsim = spec["spec"]["rfsim"]
    total = _cpu_to_milli(rfsim["gnb"]["resources"]["limits"]["cpu"]) + sidecars_milli  # e3/dApp + trf
    trf = ((rfsim.get("trafficGen") or {}).get("resources") or {}).get("limits", {}).get("cpu")
    if trf:  # sidecars_milli assumes the chart's 500m traffic generator
        total += _cpu_to_milli(trf) - 500
    ue_lim = _cpu_to_milli(rfsim["ue"]["resources"]["limits"]["cpu"])
    total += ue_lim * sum(1 for i in rfsim["ue"]["instances"] if i.get("replicas"))
    if spec["spec"].get("ric") and spec["spec"]["ric"]["ric"].get("replicas"):
        total += ric_cpu_milli
    return total


def spec_pods(spec: dict[str, Any]) -> int:
    """Pods the experiment runs with this spec (gNB, traffic generator, UEs, RIC)."""
    rfsim = spec["spec"]["rfsim"]
    pods = (1 if rfsim["gnb"].get("replicas") else 0) + 1  # trf always runs
    pods += sum(1 for i in rfsim["ue"]["instances"] if i.get("replicas"))
    if spec["spec"].get("ric") and spec["spec"]["ric"]["ric"].get("replicas"):
        pods += 1
    return pods


async def preflight(spec: dict[str, Any], current: dict[str, Any] | None) -> None:
    """Refuse up front when the tenant quota cannot fit the change (else pods sit Pending)."""
    cfg = get_k8s_core_config()
    try:
        quota = await _kubectl_json(["get", "resourcequota", "tenant-quota", "-n", cfg["namespace"]])
        hard, used = quota["status"]["hard"], quota["status"]["used"]
    except (RuntimeError, KeyError, ValueError):
        return  # no quota → nothing to check here
    fix = ("Stop another experiment, choose a smaller carrier profile, or raise the tenant quota "
           "(Operator env TENANT_CPU_QUOTA / TENANT_POD_QUOTA).")
    try:
        need = spec_cpu_limits_milli(spec) - (spec_cpu_limits_milli(current) if current else 0)
        free = _cpu_to_milli(hard["limits.cpu"]) - _cpu_to_milli(used["limits.cpu"])
        if need > 0 and need > free:
            raise ExperimentRanError(
                f"Not enough tenant CPU quota: this experiment needs {need / 1000:.1f} more CPU (limits), "
                f"{max(0, free) / 1000:.1f} of {_cpu_to_milli(hard['limits.cpu']) / 1000:.0f} free. " + fix
            )
        need_pods = spec_pods(spec) - (spec_pods(current) if current else 0)
        free_pods = int(hard["pods"]) - int(used["pods"])
        if need_pods > 0 and need_pods > free_pods:
            raise ExperimentRanError(
                f"Not enough tenant pod quota: this experiment needs {need_pods} more pod(s), "
                f"{max(0, free_pods)} of {hard['pods']} free. " + fix
            )
    except (KeyError, ValueError):
        return


async def imsi_conflicts(spec: dict[str, Any], slug: str, tenant: str) -> list[str]:
    """IMSIs this experiment wants to *run* that a running UE elsewhere already uses
    (a second attach with the same SUPI kicks the first UE off the Core)."""
    wanted = {i["imsi"] for i in spec["spec"]["rfsim"]["ue"]["instances"] if i.get("replicas")}
    if not wanted:
        return []
    from services.k8s.ran_bridge import _imsi_from_uicc_conf

    ns = get_k8s_core_config()["namespace"]
    sel = f"app.kubernetes.io/component=virtual-ue,harmony.telecom/tenant={tenant}"
    try:
        cms = await _kubectl_json(["get", "configmaps", "-n", ns, "-l", sel])
        deps = await _kubectl_json(["get", "deployments", "-n", ns, "-l", sel])
    except RuntimeError:
        return []
    # (experiment, ue) → imsi; experiment "" = shared legacy stack
    imsi_of = {}
    for cm in cms.get("items") or []:
        labels = (cm.get("metadata") or {}).get("labels") or {}
        imsi = _imsi_from_uicc_conf((cm.get("data") or {}).get("uicc.conf", ""))
        ue = labels.get("harmony.telecom/ue") or ""
        if not ue:  # ConfigMaps rendered before the label existed: <release>-ue-<name>-config
            name = (cm.get("metadata") or {}).get("name") or ""
            ue = name.rsplit("-ue-", 1)[-1].removesuffix("-config") if "-ue-" in name else ""
        imsi_of[(labels.get(EXPERIMENT_LABEL, ""), ue)] = imsi
    conflicts = []
    for dep in deps.get("items") or []:
        labels = (dep.get("metadata") or {}).get("labels") or {}
        owner = labels.get(EXPERIMENT_LABEL, "")
        if owner == slug or not int((dep.get("spec") or {}).get("replicas") or 0):
            continue
        imsi = imsi_of.get((owner, labels.get("harmony.telecom/ue", "")))
        if imsi in wanted:
            conflicts.append(f"{imsi} (running in {('experiment ' + owner) if owner else 'the shared legacy RAN'})")
    return conflicts


# ── apply ────────────────────────────────────────────────────────────────────

async def _apply(manifest: dict[str, Any]) -> None:
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "apply", "-f", "-",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await asyncio.wait_for(proc.communicate(json.dumps(manifest).encode()), timeout=30)
    if proc.returncode != 0:
        raise ExperimentRanError(f"Could not submit experiment RAN: {(err or out).decode().strip()}")


async def _wait_ready(slug: str, timeout: float = 120.0) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout
    status: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        try:
            cr = await _kubectl_json(["get", "oaiexperiments.harmony.telecom", _cr_name(slug), "-n", TESTBED_NAMESPACE])
        except RuntimeError:
            cr = {}
        status = cr.get("status") or {}
        current = status.get("observedGeneration") == (cr.get("metadata") or {}).get("generation")
        if current and status.get("phase") == "Ready":
            return status
        if current and status.get("phase") == "Failed":
            raise ExperimentRanError(f"Operator could not install the experiment RAN: {status.get('message')}")
        await asyncio.sleep(1.0)
    raise ExperimentRanError(
        f"Timed out waiting for the Operator ({status.get('phase') or 'no status'}: {status.get('message') or ''})."
    )


async def ensure_experiment_ran(
    ref: ExperimentRef, *, start: set[str] = frozenset(), extra_imsis: list[str] = (),
) -> dict[str, Any]:
    """Create/update this experiment's RAN and wait until the Operator installed it."""
    experiment = await load_experiment(ref.experiment_id)
    if not experiment:
        raise ExperimentRanError(f"Experiment {ref.experiment_id} not found")
    tenant = get_k8s_core_config()["tenant"]
    live = await live_replicas(tenant, ref.slug)
    identity = await allocate_identity(ref.slug, tenant)
    spec = build_spec(experiment, ref, tenant, identity, live, start=set(start), extra_imsis=list(extra_imsis))

    current_spec = build_spec(experiment, ref, tenant, identity, live, extra_imsis=list(extra_imsis))
    await preflight(spec, current_spec if live else None)
    conflicts = await imsi_conflicts(spec, ref.slug, tenant)
    if conflicts:
        raise ExperimentRanError(
            "IMSI already attached in another experiment: " + ", ".join(conflicts)
            + ". Give this experiment's Virtual UE a different IMSI."
        )

    from main import Subscriber, add_subscriber
    from services.k8s.ran_bridge import _RFSIM_UE_KEY, _RFSIM_UE_OPC

    for inst in spec["spec"]["rfsim"]["ue"]["instances"]:
        res = await add_subscriber(Subscriber(imsi=inst["imsi"], key=_RFSIM_UE_KEY, opc=_RFSIM_UE_OPC))
        if not res.get("success"):
            raise ExperimentRanError(f"Could not register subscriber {inst['imsi']}: {res.get('message')}")

    await _apply(spec)
    status = await _wait_ready(ref.slug)
    return {
        "ok": True, "slug": ref.slug, "namespace": status.get("namespace"),
        "rfsim_release": status.get("rfsimRelease"), "ric_release": status.get("ricRelease"),
        "gnb_id": spec["spec"]["rfsim"]["gnb"]["gnbId"],
        "carrier": spec["spec"]["rfsim"]["gnb"]["dlCarrierBandwidth"],
    }


async def delete_experiment_ran(experiment_id: str) -> None:
    """Remove an experiment's RAN (the Operator uninstalls its releases)."""
    _ref_cache.pop(str(experiment_id), None)
    slug = slug_for(experiment_id)
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "delete", "oaiexperiments.harmony.telecom", _cr_name(slug), "-n", TESTBED_NAMESPACE,
        "--ignore-not-found", "--wait=false",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
