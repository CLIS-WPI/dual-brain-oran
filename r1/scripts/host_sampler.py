#!/usr/bin/env python3
"""Per-minute host load sampler for R1 calibration/measurement runs.

  python3 r1/scripts/host_sampler.py <out.jsonl> [--interval 60] [--top 8] [--until-pid PID]

Each line: {wall_iso, loadavg_1m/5m/15m, top: [{user, pid, pcpu, comm}], by_user: {user: pcpu}}.
`pcpu` = CPU used over a 1 s window from /proc/<pid>/stat deltas (100 = one core); `by_user` sums it.
summarize(path) → per-minute table for run summaries.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time


def _cpu_ticks() -> dict[int, tuple[int, str, int]]:
    """pid -> (uid, comm, utime+stime ticks) for every process."""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            stat = open(f"/proc/{d}/stat").read()
            comm = stat[stat.index("(") + 1:stat.rindex(")")]
            f = stat[stat.rindex(")") + 2:].split()
            out[int(d)] = (os.stat(f"/proc/{d}").st_uid, comm, int(f[11]) + int(f[12]))
        except (OSError, ValueError, IndexError):
            continue
    return out


def sample(top: int = 8, window_s: float = 1.0) -> dict:
    """CPU actually used over `window_s` (not ps's lifetime average): 100 = one core."""
    import pwd

    hz = os.sysconf("SC_CLK_TCK")
    a = _cpu_ticks()
    time.sleep(window_s)
    b = _cpu_ticks()
    rows, by_user = [], {}
    for pid, (uid, comm, t1) in b.items():
        t0 = a.get(pid, (uid, comm, t1))[2]
        pcpu = round(100.0 * (t1 - t0) / hz / window_s, 1)
        if pcpu <= 0:
            continue
        try:
            user = pwd.getpwuid(uid).pw_name
        except KeyError:
            user = str(uid)
        by_user[user] = round(by_user.get(user, 0) + pcpu, 1)
        rows.append({"user": user, "pid": pid, "pcpu": pcpu, "comm": comm})
    rows.sort(key=lambda r: -r["pcpu"])
    l1, l5, l15 = os.getloadavg()
    return {"wall_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), "loadavg_1m": round(l1, 2), "loadavg_5m": round(l5, 2),
            "loadavg_15m": round(l15, 2), "top": rows[:top],
            "by_user": dict(sorted(by_user.items(), key=lambda kv: -kv[1])[:6])}


def summarize(path: str) -> str:
    lines = [json.loads(l) for l in open(path) if l.strip()]
    out = [f"{'time':19} {'load1':>6}  top CPU users (pcpu %, 100 = 1 core)"]
    for s in lines:
        users = ", ".join(f"{u} {p:.0f}" for u, p in list(s["by_user"].items())[:4])
        procs = ", ".join(f"{t['comm']}({t['user']}) {t['pcpu']:.0f}" for t in s["top"][:4])
        out.append(f"{s['wall_iso']:19} {s['loadavg_1m']:>6}  users: {users} | procs: {procs}")
    return "\n".join(out)


if __name__ == "__main__":
    path = sys.argv[1]
    arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    interval, top, until = arg("--interval", 60), arg("--top", 8), arg("--until-pid", 0)
    if "--summary" in sys.argv:
        print(summarize(path))
        sys.exit()
    while True:
        with open(path, "a") as fh:
            fh.write(json.dumps(sample(top)) + "\n")
        if until and not os.path.exists(f"/proc/{until}"):
            break
        time.sleep(interval)
