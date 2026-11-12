#!/usr/bin/env python3
"""Run-integrity helpers for script-driven R1 runs (calibration, experiment runners).

backend_worker()  the Harmony portal backend process that serves the API: the `spawn_main`
                  child of `uvicorn main:app --reload`, or uvicorn itself when started without
                  --reload. Identity = (pid, start time) — a reload replaces the worker.
assert_same(b)    raise BackendReloaded (`backend_reloaded`) if the worker changed since `b`.

  python3 r1/scripts/run_guard.py      # print the current backend worker identity
"""
from __future__ import annotations

import json
import os
from pathlib import Path

BACKEND_RELOADED = "backend_reloaded"


class BackendReloaded(RuntimeError):
    def __init__(self, before: dict, now: dict | None):
        super().__init__(f"{BACKEND_RELOADED}: portal backend worker {before.get('pid')} "
                         f"→ {(now or {}).get('pid')} during the run (dev server reloaded on a file edit?)")
        self.before, self.now = before, now


def _start_epoch(pid: int) -> float:
    """Process start time (epoch s) from /proc/<pid>/stat field 22 + boot time."""
    stat = Path(f"/proc/{pid}/stat").read_text()
    ticks = int(stat.rsplit(")", 1)[1].split()[19])
    btime = next(int(l.split()[1]) for l in Path("/proc/stat").read_text().splitlines() if l.startswith("btime"))
    return round(btime + ticks / os.sysconf("SC_CLK_TCK"), 2)


def _procs() -> list[tuple[int, int, str]]:
    out = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
                ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
                out.append((int(d.name), ppid, cmd))
            except (OSError, ValueError, IndexError):
                continue
    return out


def backend_worker() -> dict | None:
    procs = _procs()
    for pid, _, cmd in procs:
        if "uvicorn main:app" in cmd and "--port 8888" in cmd:
            if "--reload" not in cmd:
                return {"pid": pid, "start": _start_epoch(pid), "reload": False, "cmd": cmd[:160]}
            kids = [p for p, pp, c in procs if pp == pid and "spawn_main" in c]
            if kids:
                return {"pid": kids[0], "start": _start_epoch(kids[0]), "reload": True, "parent": pid,
                        "cmd": cmd[:160]}
    return None


def assert_same(before: dict) -> None:
    now = backend_worker()
    if not now or (now["pid"], now["start"]) != (before["pid"], before["start"]):
        raise BackendReloaded(before, now)


if __name__ == "__main__":
    print(json.dumps(backend_worker(), indent=1))
