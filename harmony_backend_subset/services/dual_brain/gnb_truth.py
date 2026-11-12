"""gNB PRB ground truth for Dual-Brain runs (Harmony patch 0004 `[AURA-PRB] 1s` lines).

Every second the gNB logs the cell's DL PRB budget (DL BWP PRBs over the DL slots it
scheduled, the denominator of KPM RRU.PrbTotDl) and the PRBs each RNTI transmitted.
From that:

  sim speed   = budget / profile ground_truth.dl_prb_per_s_realtime
                (RFsim is not locked to wall time: ~2.5-2.9x on the lab host, ±10 %/s)
  cell load   = sum(tx_rb) / budget          (clock-independent, the congestion metric)
  UE share    = tx_rb / budget, mapped RNTI -> CU-UE-ID (= KPM RAN UE ID) from the
                gNB's `UE RNTI xxxx CU-UE-ID n` lines

Runs record this next to the KPM figures and the offered Mbps, so the sim-clock drift
stays visible and congestion is judged on measured load, not on the input rate.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import re
import statistics as st
from typing import Any

from services.k8s.core_bridge import get_k8s_core_config

_PRB = re.compile(r"budget=(\d+)")
_UE = re.compile(r"rnti=0x([0-9a-f]+):slice=(-?\d+):tx_rb=(\d+)")
_MAP = re.compile(r"UE RNTI ([0-9a-f]+) CU-UE-ID (\d+)")


def _ts(line: str) -> float:
    return dt.datetime.fromisoformat(line.split(" ", 1)[0].replace("Z", "")[:26]).replace(
        tzinfo=dt.timezone.utc).timestamp()


def parse(log: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Rows {t, budget, ues: {rnti: (slice, tx_rb)}} and rnti -> CU-UE-ID from a --timestamps gNB log."""
    rows, cu = [], {}
    for line in log.splitlines():
        m = _MAP.search(line)
        if m:
            cu[m.group(1)] = m.group(2)
        if "[AURA-PRB] 1s" not in line:
            continue
        try:
            t = _ts(line)
        except ValueError:
            continue
        b = int(_PRB.search(line).group(1))
        if b > 0:
            rows.append({"t": t, "budget": b,
                         "ues": {r: (int(s), int(x)) for r, s, x in _UE.findall(line)}})
    return rows, cu


async def fetch(since_s: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Ground truth from this experiment's running gNB pod log over the last `since_s` seconds."""
    from services.dual_brain.channel_guard import gnb_pod

    pod = await gnb_pod()  # experiment-scoped: never another canvas's gNB
    if not pod:
        return [], {}
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "-n", get_k8s_core_config()["namespace"], "logs", pod, "-c", "gnb", "--timestamps",
        f"--since={int(since_s)}s", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, _ = await proc.communicate()
    return parse(out.decode(errors="replace"))


def _stats(v: list[float]) -> dict[str, float] | None:
    if not v:
        return None
    s = sorted(v)
    return {"mean": round(st.mean(s), 2), "sd": round(st.pstdev(s), 2), "min": round(s[0], 2),
            "p10": round(s[len(s) // 10], 2), "max": round(s[-1], 2), "n": len(s)}


def summarize(rows: list[dict[str, Any]], cu: dict[str, str], windows: list[tuple[float, float]],
              realtime_budget: float | None, *, ramp_s: float = 2.0) -> dict[str, Any]:
    """Clock + measured load inside `windows` (wall-clock seconds, ramp skipped) and overall."""
    inside = [r for r in rows if any(a + ramp_s <= r["t"] <= b for a, b in windows)]
    shares: dict[str, list[float]] = {}
    slices: dict[str, set[int]] = {}
    for r in inside:
        for rnti, (sl, tx) in r["ues"].items():
            key = cu.get(rnti, f"rnti:{rnti}")
            shares.setdefault(key, []).append(100.0 * tx / r["budget"])
            slices.setdefault(key, set()).add(sl)
    return {
        "sim_speed_run": _stats([r["budget"] / realtime_budget for r in rows]) if realtime_budget else None,
        "sim_speed_windows": _stats([r["budget"] / realtime_budget for r in inside]) if realtime_budget else None,
        "budget_per_s_windows": _stats([float(r["budget"]) for r in inside]),
        "cell_load_windows": _stats([100.0 * sum(x for _, x in r["ues"].values()) / r["budget"] for r in inside]),
        "ue_share_windows": {k: {**(_stats(v) or {}), "slices": sorted(slices[k])} for k, v in sorted(shares.items())},
        "realtime_budget_per_s": realtime_budget,
        "source": "gNB [AURA-PRB] 1s (patch 0004)",
    }
