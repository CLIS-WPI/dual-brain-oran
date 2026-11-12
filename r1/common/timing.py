"""Timing marks for R1: one line per event in <run_dir>/timing.jsonl (nobody reads timestamps from logs).

    from common import timing
    tm = timing.Timer(run_id, run_dir)
    tm.mark('agent_request', model='llama3.1:8b')
    with tm.span('gate'):            # writes gate_start / gate_end
        ...
Record: {run_id, event, monotonic_ns, wall_iso, **extra}. monotonic_ns orders/diffs events within a host;
wall_iso (local time with offset, ms) aligns with other logs.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import time


class Timer:
    def __init__(self, run_id: str, run_dir: str):
        self.run_id = run_id
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, 'timing.jsonl')

    def mark(self, event: str, **extra) -> dict:
        rec = {'run_id': self.run_id, 'event': event, 'monotonic_ns': time.monotonic_ns(),
               'wall_iso': dt.datetime.now().astimezone().isoformat(timespec='milliseconds'), **extra}
        with open(self.path, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        return rec

    @contextlib.contextmanager
    def span(self, name: str, **extra):
        self.mark(f'{name}_start', **extra)
        try:
            yield
        finally:
            self.mark(f'{name}_end', **extra)


def load(run_dir: str) -> list[dict]:
    p = os.path.join(run_dir, 'timing.jsonl')
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def duration_s(marks: list[dict], start: str, end: str) -> float | None:
    """Seconds from the first `start` mark to the first `end` mark after it (monotonic clock)."""
    s = next((m for m in marks if m['event'] == start), None)
    e = next((m for m in marks if m['event'] == end and s and m['monotonic_ns'] >= s['monotonic_ns']), None)
    return (e['monotonic_ns'] - s['monotonic_ns']) / 1e9 if s and e else None


if __name__ == '__main__':
    import tempfile
    d = tempfile.mkdtemp()
    t = Timer('selftest', d)
    with t.span('x', k=1):
        time.sleep(0.01)
    m = load(d)
    assert [x['event'] for x in m] == ['x_start', 'x_end'] and m[0]['k'] == 1
    assert 0.009 < duration_s(m, 'x_start', 'x_end') < 1
    print('timing selftest ok')
