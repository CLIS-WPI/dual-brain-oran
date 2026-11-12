#!/usr/bin/env python3
"""Seeded DL burst schedules for the R1 center UEs (UE1/UE2), executed inside each UE pod.

Profiles
  step  constant `peak` Mbps for `on_s`, then OFF (the MILCOM-style step burst).
  ramp  `base` → `peak` linearly over `up_s`, hold `hold_s`, down over `down_s`, then OFF.
        iperf2 cannot change rate inside a flow, so the ramp is built from concurrent
        constant-rate flows: a `base` flow plus one Δ flow joining every `step_s`; on the way
        down the Δ flows end one after another. No reconnect gaps in the offered load.
OFF gaps get ±`off_jitter_s` from `seed`, so onsets are not strictly periodic but every run
with the same seed sees the same pattern (brief §2.3: deterministic, seeded).

  python3 burst.py plan --profile ramp --peak 30 --cycles 6 --seed 6   # JSON schedule
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import shlex
from typing import Any

PROFILES: dict[str, dict[str, Any]] = {
    "step": {"peak": 20.0, "on_s": 30.0, "off_s": 15.0, "off_jitter_s": 2.0},
    "ramp": {"base": 5.0, "peak": 30.0, "up_s": 18.0, "hold_s": 10.0, "down_s": 5.0, "step_s": 2.0,
             "off_s": 30.0, "off_jitter_s": 2.0},  # OFF ≥30 s: RLC AM keeps UDP excess queued; OFF must drain
}


def schedule(profile: str, *, cycles: int, seed: int, **over: float) -> dict[str, Any]:
    """{params, cycles: [{t0, phases: {name: [start, end]}, flows: [(start, mbps, dur)]}], total_s}."""
    p = {**PROFILES[profile], **{k: float(v) for k, v in over.items() if v is not None}}
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(cycles):
        off = p["off_s"] + rng.uniform(-p["off_jitter_s"], p["off_jitter_s"])
        if profile == "step":
            flows = [(t, p["peak"], p["on_s"])]
            phases = {"on": [t, t + p["on_s"]]}
            t_on_end = t + p["on_s"]
        else:
            n = max(1, round(p["up_s"] / p["step_s"]))  # Δ flows during the ramp-up
            delta = (p["peak"] - p["base"]) / n
            up_end, hold_end = t + n * p["step_s"], t + n * p["step_s"] + p["hold_s"]
            t_on_end = hold_end + p["down_s"]
            flows = [(t, p["base"], t_on_end - t)]  # base flow spans the whole ON phase
            for i in range(1, n + 1):  # flow i joins at i*step and leaves i-th from the end
                start = t + i * p["step_s"]
                end = hold_end + p["down_s"] * (n - i) / n
                flows.append((start, delta, end - start))
            phases = {"ramp_up": [t, up_end], "hold": [up_end, hold_end], "ramp_down": [hold_end, t_on_end],
                      "on": [t, t_on_end]}
        phases["off"] = [t_on_end, t_on_end + off]
        out.append({"t0": round(t, 3), "phases": {k: [round(a, 3), round(b, 3)] for k, (a, b) in phases.items()},
                    "flows": [(round(a, 3), round(r, 4), round(d, 3)) for a, r, d in flows]})
        t = t_on_end + off
    return {"profile": profile, "seed": seed, "params": p, "cycles": out, "total_s": round(t, 3)}


def offered_mbps(sched: dict[str, Any], t: float) -> float:
    """Offered load of one UE at time t (for plots and self-checks)."""
    return sum(r for c in sched["cycles"] for a, r, d in c["flows"] if a <= t < a + d)


RECV_LOG = "/tmp/r1_iperf.csv"


def pod_script(sched: dict[str, Any], *, server_ip: str, bind_ip: str, port: int = 5201, log: str = RECV_LOG) -> str:
    """sh script for the UE pod: every flow as a delayed, detached iperf2 UDP reverse (DL) client.

    Each flow appends its receiver-side per-second CSV (`-i 1 -y C`: wall-clock UTC, payload bytes)
    to `log`; parse it with `receiver_mbps`. TZ=UTC: the OAI UE image sets TZ=Europe/Paris and
    `-y C` timestamps are local time."""
    cmds = [f"rm -f {log}"]
    for c in sched["cycles"]:
        for a, r, d in c["flows"]:
            cmds.append(f"(sleep {a:.3f}; TZ=UTC iperf -c {server_ip} -B {bind_ip} -p {port} -u -R "
                        f"-b {round(r * 1e6)} -t {max(1, round(d))} -i 1 -y C >>{log} 2>/dev/null) &")
    return "\n".join(cmds) + "\nwait\n"


def receiver_mbps(csv_text: str) -> list[tuple[float, float]]:
    """[(epoch s at interval midpoint, Mbps)] per second, summed over concurrent flows.

    iperf2 `-y C` receiver lines: ts(UTC, end of interval),src,sport,dst,dport,id,"a-b",bytes,bps,...
    Per-flow summary lines (interval longer than 1.5 s) are skipped."""
    per_s: dict[float, int] = {}
    for line in csv_text.splitlines():
        f = line.split(",")
        if len(f) < 9 or "-" not in f[6]:
            continue
        a, b = (float(x) for x in f[6].split("-"))
        if b - a > 1.5:
            continue
        t = dt.datetime.strptime(f[0], "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc).timestamp() - 0.5
        per_s[t] = per_s.get(t, 0) + int(f[7])
    return [(t, n * 8 / 1e6) for t, n in sorted(per_s.items())]


def remote_command(sched: dict[str, Any], **kw: Any) -> list[str]:
    """`sh -c` argv that starts the whole schedule detached inside the pod and returns at once."""
    script = pod_script(sched, **kw)
    return ["sh", "-c", f"nohup sh -c {shlex.quote(script)} >/tmp/r1_burst.log 2>&1 </dev/null & echo started"]


def _self_check() -> None:
    s = schedule("ramp", cycles=2, seed=6, peak=40)
    c = s["cycles"][0]["phases"]
    assert abs(offered_mbps(s, c["ramp_up"][0] + 0.1) - 5.0) < 1e-6  # ramp starts at base
    assert abs(offered_mbps(s, c["hold"][0] + 1) - 40.0) < 1e-3  # hold at peak
    assert offered_mbps(s, c["off"][0] + 1) == 0.0  # OFF really off
    mid = [offered_mbps(s, c["ramp_up"][0] + k * 2 + 0.1) for k in range(9)]
    assert all(b > a for a, b in zip(mid, mid[1:]))  # monotonic rise
    assert schedule("ramp", cycles=3, seed=6) == schedule("ramp", cycles=3, seed=6)  # seeded
    assert schedule("ramp", cycles=3, seed=6) != schedule("ramp", cycles=3, seed=7)
    csv_ = ("20261103174438,192.0.2.10,5201,198.51.100.10,59501,1,0.0-1.0,26460,211680,0.479,0,18,0.000,0\n"
            "20261103174438,192.0.2.10,5201,198.51.100.10,59502,2,0.0-1.0,100000,800000,0.4,0,68,0.000,0\n"
            "20261103174439,192.0.2.10,5201,198.51.100.10,59501,1,0.0-3.1,82320,213826,0.412,0,56,0.000,0\n")
    assert receiver_mbps(csv_) == [(1793727877.5, 126460 * 8 / 1e6)]  # flows summed, summary skipped, UTC


if __name__ == "__main__":
    _self_check()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan"])
    ap.add_argument("--profile", choices=sorted(PROFILES), default="ramp")
    ap.add_argument("--peak", type=float)
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--seed", type=int, default=6)
    a = ap.parse_args()
    print(json.dumps(schedule(a.profile, cycles=a.cycles, seed=a.seed, peak=a.peak), indent=1))
