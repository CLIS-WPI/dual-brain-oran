"""Cell-edge guard + run integrity for R1 runs.

The cell-edge UE (UE3) gets a UE-side DL channel model only (rfsimu_channel_enB0, applied
by the UE itself), so nothing depends on rfsim connection order. This module checks, never
silently, that the edge is real and that the RAN under the run did not change:

  mapping   UE label -> IMSI -> C-RNTI (UE pod log `UE 0 RNTI xxxx stats`)
            -> CU-UE-ID (gNB `UE RNTI xxxx CU-UE-ID n`) + gNB MAC stats per RNTI:
            DL MCS / BLER / goodput (`UE xxxx: dlsch_rounds … BLER b MCS (t) m … goodput g Mbps`),
            UL SNR (`UE xxxx: ulsch_rounds … SNR v dB`).
  edge      mean DL MCS over *active* samples (goodput > 0): the edge UE must be >= min_gap
            below every center UE carrying DL traffic, else `channel_mapping_violated`.
            DL CQI: this gNB build prints no CSI/CQI report, so the edge is defined on DL MCS.
  pods      gNB / UE / RIC pod UIDs + restart counts at run start; any change → `pod_changed`.
  UL        any UE with UL goodput > 2 Mbps for > 5 s → `ul_traffic_detected` (R1 traffic is DL
            only; the Canvas Iperf Run forces UDP UL on K8s UEs).
  speed     RFsim is CPU-bound, not real time: sim/wall ratio = gNB [AURA-PRB] 1s PRB budget /
            real-time budget. A 10 s mean > ±10 % off the run-start value twice → `sim_speed_drift`.
  evidence  save_crash_logs(): `kubectl logs --previous` of every restarted gNB/UE/RIC container.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
import statistics as st
from typing import Any

_UE_RNTI = re.compile(r"UE \d+ RNTI ([0-9a-f]{4}) stats")
_GNB_MAP = re.compile(r"UE RNTI ([0-9a-f]{4}) CU-UE-ID (\d+)")
_GNB_DL = re.compile(r"UE ([0-9a-f]{4}): dlsch_rounds .*?BLER (\d+(?:\.\d+)?) MCS \(\d+\) (\d+) .*?goodput (\d+(?:\.\d+)?) Mbps")
_GNB_UL = re.compile(r"UE ([0-9a-f]{4}): ulsch_rounds .*? SNR (-?\d+(?:\.\d+)?).*?goodput (\d+(?:\.\d+)?) Mbps")

VIOLATION = "channel_mapping_violated"
POD_CHANGED = "pod_changed"
UL_TRAFFIC = "ul_traffic_detected"
UL_LIMIT = {"goodput_mbps": 2.0, "max_s": 5.0}
SIM_SPEED = "sim_speed_drift"
SPEED_LIMIT = {"max_drift": 0.10, "window_s": 10}
EDGE_DEFINITION = {
    "metric": "dl_mcs_mean", "min_gap": 5,
    "source": "gNB MAC stats (dlsch MCS), samples with DL goodput > 0, 60 s window",
    "dl_cqi": "not reported by this gNB build (no CSI report in the MAC stats)",
}


class GuardViolation(RuntimeError):
    def __init__(self, code: str, detail: str, info: dict[str, Any]):
        super().__init__(f"{code}: {detail}")
        self.code, self.info = code, info


def ue_rnti(ue_log: str) -> str | None:
    """Latest C-RNTI the UE reports for itself."""
    found = _UE_RNTI.findall(ue_log)
    return found[-1] if found else None


def _ts(line: str) -> float | None:
    """Leading `kubectl logs --timestamps` RFC3339 time, else None."""
    head = line.split(" ", 1)[0]
    try:
        return dt.datetime.fromisoformat(head.replace("Z", "")[:26]).replace(tzinfo=dt.timezone.utc).timestamp()
    except ValueError:
        return None


def parse_gnb(gnb_log: str) -> dict[str, dict[str, Any]]:
    """rnti -> {cu_ue_id, dl: [(mcs, bler, goodput)], ul_snr: [dB], ul: [(t, goodput)]} from a gNB log window."""
    out: dict[str, dict[str, Any]] = {}

    def ue(r: str) -> dict[str, Any]:
        return out.setdefault(r, {"cu_ue_id": None, "dl": [], "ul_snr": [], "ul": []})
    for line in gnb_log.splitlines():
        if m := _GNB_MAP.search(line):
            ue(m.group(1))["cu_ue_id"] = m.group(2)
        elif m := _GNB_DL.search(line):
            ue(m.group(1))["dl"].append((int(m.group(3)), float(m.group(2)), float(m.group(4))))
        elif m := _GNB_UL.search(line):
            ue(m.group(1))["ul_snr"].append(float(m.group(2)))
            t = _ts(line)
            if t is not None:
                ue(m.group(1))["ul"].append((t, float(m.group(3))))
    return out


def ul_high_s(samples: list[tuple[float, float]], limit_mbps: float = UL_LIMIT["goodput_mbps"]) -> float:
    """Longest contiguous span (s) with UL goodput above limit_mbps."""
    best, start = 0.0, None
    for t, g in sorted(samples):
        if g > limit_mbps:
            start = t if start is None else start
            best = max(best, t - start)
        else:
            start = None
    return round(best, 1)


def _mean(v: list[float]) -> float | None:
    return round(st.mean(v), 2) if v else None


def build_mapping(ues: list[dict[str, Any]], gnb: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """ues: [{label, imsi, rnti}] → per-UE row; None = not seen on this gNB."""
    rows = []
    for u in ues:
        g = gnb.get(u.get("rnti") or "")
        active = [d for d in (g or {}).get("dl", []) if d[2] > 0]
        rows.append({
            "ue": u["label"], "imsi": u["imsi"], "rnti": u.get("rnti"), "on_gnb": g is not None,
            "cu_ue_id": (g or {}).get("cu_ue_id"),
            "dl_mcs_mean": _mean([d[0] for d in active]), "dl_bler_mean": _mean([d[1] for d in active]),
            "dl_goodput_mbps_mean": _mean([d[2] for d in active]), "dl_active_n": len(active),
            "dl_cqi": None,  # not reported by this gNB build (EDGE_DEFINITION)
            "ul_snr_db": round(st.median(g["ul_snr"]), 1) if g and g["ul_snr"] else None,
            "ul_high_s": ul_high_s((g or {}).get("ul", [])),
            "ul_goodput_mbps_max": max((x[1] for x in (g or {}).get("ul", [])), default=None),
        })
    return rows


def check_mapping(rows: list[dict[str, Any]]) -> str | None:
    """Every UE has a C-RNTI that this experiment's gNB reports stats for."""
    for r in rows:
        if r["rnti"] is None:
            return f"{r['ue']} ({r['imsi']}) has no C-RNTI (not attached?)"
        if not r["on_gnb"]:
            return f"{r['ue']} RNTI {r['rnti']} has no MAC stats on this gNB — wrong gNB or stale RNTI"
    return None


def check(rows: list[dict[str, Any]], edge_imsi: str, min_gap: float) -> str | None:
    """None if the edge UE's mean DL MCS is >= min_gap below every center UE with DL traffic."""
    problem = check_mapping(rows)
    if problem:
        return problem
    edge = next((r for r in rows if r["imsi"] == edge_imsi), None)
    if edge is None:
        return f"edge UE {edge_imsi} is not among this experiment's UEs"
    if edge["dl_mcs_mean"] is None:
        return f"{edge['ue']} has no DL traffic in the window — edge DL MCS unreadable"
    centers = [r for r in rows if r is not edge and r["dl_mcs_mean"] is not None]
    if not centers:
        return "no center UE carried DL traffic in the window — cannot compare DL MCS"
    for c in centers:
        gap = c["dl_mcs_mean"] - edge["dl_mcs_mean"]
        if gap < min_gap:
            return (f"{edge['ue']} mean DL MCS {edge['dl_mcs_mean']} is only {gap:.1f} below {c['ue']} "
                    f"({c['dl_mcs_mean']}), need >= {min_gap}")
    return None


def check_ul(rows: list[dict[str, Any]], max_s: float = UL_LIMIT["max_s"]) -> str | None:
    """R1 traffic is DL only: a UE with UL goodput > limit for > max_s means someone ran UL traffic."""
    for r in rows:
        if r.get("ul_high_s", 0) > max_s:
            return (f"{r['ue']} UL goodput > {UL_LIMIT['goodput_mbps']} Mbps for {r['ul_high_s']} s "
                    f"(max {r['ul_goodput_mbps_max']} Mbps) — R1 traffic must be DL (Canvas Iperf Run forces UDP UL)")
    return None


def speed_ratios(gnb_log: str, realtime_budget: float) -> list[tuple[float, float]]:
    """[(t, sim/wall ratio)] from timestamped gNB `[AURA-PRB] 1s` lines (budget / real-time budget)."""
    from services.dual_brain import gnb_truth

    return [(r["t"], r["budget"] / realtime_budget) for r in gnb_truth.parse(gnb_log)[0]]


def speed_drift(base: float, now: float, max_drift: float = SPEED_LIMIT["max_drift"]) -> str | None:
    d = now / base - 1
    return None if abs(d) <= max_drift else (
        f"sim/wall speed {now:.2f}x vs {base:.2f}x at run start ({d:+.0%}, limit ±{max_drift:.0%}) — "
        "host load changed; PRB % and timings are not comparable")


def pod_changes(base: dict[str, dict[str, Any]], now: dict[str, dict[str, Any]]) -> str | None:
    """Differences between two pods_state() snapshots, or None."""
    diffs = []
    for comp in sorted(set(base) | set(now)):
        b, n = base.get(comp), now.get(comp)
        if b is None or n is None:
            diffs.append(f"{comp} {'appeared' if b is None else 'disappeared'}")
        elif b["uid"] != n["uid"]:
            diffs.append(f"{comp} replaced ({b['pod']} → {n['pod']})")
        elif b["restarts"] != n["restarts"]:
            diffs.append(f"{comp} restarted ({b['restarts']} → {n['restarts']})")
    return "; ".join(diffs) or None


# --------------------------------------------------------------------------- live I/O

async def _logs(ns: str, pod: str, container: str, since_s: int, *, timestamps: bool = False) -> str:
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "-n", ns, "logs", pod, "-c", container, f"--since={int(since_s)}s",
        *(["--timestamps"] if timestamps else []),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    return out.decode(errors="replace")


async def _pods(component: str) -> list[dict[str, Any]]:
    from services.k8s.core_bridge import _kubectl_json, get_k8s_core_config
    from services.k8s.experiment_ctx import selector

    cfg = get_k8s_core_config()
    data = await _kubectl_json(["get", "pods", "-n", cfg["namespace"], "-l", selector(component, cfg["tenant"])])
    return [p for p in data.get("items") or [] if (p.get("status") or {}).get("phase") == "Running"
            and not (p.get("metadata") or {}).get("deletionTimestamp")]


async def gnb_pod() -> str | None:
    """Running gNB pod of the current experiment (or legacy tenant RAN)."""
    for comp in ("gnb-rfsim", "gnb"):
        try:
            pods = await _pods(comp)
        except RuntimeError:
            continue
        if pods:
            return pods[0]["metadata"]["name"]
    return None


async def sim_speed(window_s: int, realtime_budget: float) -> float | None:
    """Mean sim/wall ratio over the last window_s seconds (None = no AURA-PRB lines: non-AURA gNB)."""
    from services.k8s.core_bridge import get_k8s_core_config

    pod = await gnb_pod()
    if not pod:
        return None
    r = speed_ratios(await _logs(get_k8s_core_config()["namespace"], pod, "gnb", window_s, timestamps=True),
                     realtime_budget)
    return round(st.mean(v for _, v in r), 3) if r else None


async def save_crash_logs(dest: str) -> list[str]:
    """Previous-container logs + status of every restarted gNB/UE/RIC container → dest/. Returns files."""
    from services.k8s.core_bridge import get_k8s_core_config

    ns, saved = get_k8s_core_config()["namespace"], []
    for comp in ("gnb-rfsim", "virtual-ue", "ric", "traffic-gen"):
        for p in await _pods(comp):
            for c in (p.get("status") or {}).get("containerStatuses") or []:
                if not int(c.get("restartCount") or 0):
                    continue
                os.makedirs(dest, exist_ok=True)
                base = os.path.join(dest, f"{p['metadata']['name']}_{c['name']}")
                proc = await asyncio.create_subprocess_exec(
                    "kubectl", "-n", ns, "logs", p["metadata"]["name"], "-c", c["name"], "--previous", "--timestamps",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                out, err = await asyncio.wait_for(proc.communicate(), timeout=60)
                with open(base + "_previous.log", "wb") as fh:
                    fh.write(out or err)
                with open(base + "_status.json", "w") as fh:
                    json.dump({"pod": p["metadata"]["name"], "uid": p["metadata"]["uid"], "container": c["name"],
                               "restartCount": c.get("restartCount"), "lastState": c.get("lastState"),
                               "state": c.get("state")}, fh, indent=1)
                saved += [base + "_previous.log", base + "_status.json"]
    return saved


async def pods_state() -> dict[str, dict[str, Any]]:
    """{component[/ue]: {pod, uid, restarts}} for this experiment's gNB, UEs and RIC."""
    out = {}
    for comp in ("gnb-rfsim", "virtual-ue", "ric", "traffic-gen"):
        for p in await _pods(comp):
            md = p["metadata"]
            key = f"{comp}/{(md.get('labels') or {}).get('harmony.telecom/ue', '')}".rstrip("/")
            out[key] = {"pod": md["name"], "uid": md["uid"],
                        "restarts": sum(int(c.get("restartCount") or 0) for c in (p.get("status") or {}).get("containerStatuses") or [])}
    return out


async def snapshot(window_s: int = 60) -> list[dict[str, Any]]:
    """Live per-UE rows for the current experiment (60 s spans a full traffic on/off cycle)."""
    from services.k8s.core_bridge import get_k8s_core_config
    from services.k8s.ran_bridge import collect_k8s_virtual_ues

    ns = get_k8s_core_config()["namespace"]
    ues = sorted((u for u in await collect_k8s_virtual_ues() if u.get("imsi") and u.get("pod")),
                 key=lambda u: u["imsi"])
    # UE stats print every ~1.3 s (sim time); 30 s of log always holds the current RNTI.
    rntis = await asyncio.gather(*(_logs(ns, u["pod"], "ue", 30) for u in ues))
    pod = await gnb_pod()
    gnb = parse_gnb(await _logs(ns, pod, "gnb", window_s, timestamps=True)) if pod else {}
    return build_mapping([{"label": f"UE{i + 1}", "imsi": u["imsi"], "rnti": ue_rnti(log)}
                          for i, (u, log) in enumerate(zip(ues, rntis))], gnb)
