"""Deploy and drive the congestion xApp inside the (experiment-scoped) RIC pod.

Files live in profile["ric"]["workdir"] (default /tmp/dualbrain) in the RIC pod:
config.json, model.onnx, preprocess.json, events.jsonl. A retrained model is a
file copy + xApp restart — the binary never changes.
"""
from __future__ import annotations

import asyncio
import json
import shlex
from typing import Any

from services.k8s import xapp_bridge
from services.k8s.core_bridge import get_k8s_core_config

XAPP = "xapp_congestion_ctrl"


async def _pod() -> str:
    pod = await xapp_bridge._get_ric_pod(get_k8s_core_config())
    if not pod:
        raise RuntimeError("nearRT-RIC pod is not running (start FlexRIC first)")
    return pod


class RicCrashed(RuntimeError):
    """nearRT-RIC died (container restart or pod replaced) while a run depended on it."""

    def __init__(self, message: str, info: dict[str, Any]):
        super().__init__(message)
        self.info = info


async def ric_state() -> dict[str, Any]:
    """{pod, restarts, last_exit_code, last_reason, last_finished_at} of the nearRT-RIC pod,
    from the pod status (works while the RIC itself is down)."""
    from services.k8s.core_bridge import _kubectl_json

    cfg = get_k8s_core_config()
    pod = await xapp_bridge._get_ric_pod(cfg)
    if not pod:
        return {"pod": None}
    try:
        data = await _kubectl_json(["get", "pod", pod, "-n", cfg["namespace"]])
    except RuntimeError:
        return {"pod": pod}
    cs = ((data.get("status") or {}).get("containerStatuses") or [{}])[0]
    term = ((cs.get("lastState") or {}).get("terminated")) or {}
    return {"pod": pod, "restarts": cs.get("restartCount", 0), "ready": cs.get("ready"),
            "last_exit_code": term.get("exitCode"), "last_reason": term.get("reason"),
            "last_finished_at": term.get("finishedAt")}


def crashed_since(base: dict[str, Any], now: dict[str, Any]) -> bool:
    return bool(base.get("pod")) and (now.get("pod") != base.get("pod")
                                       or (now.get("restarts") or 0) > (base.get("restarts") or 0))


def crash_message(now: dict[str, Any]) -> str:
    if now.get("last_exit_code") is not None:
        how = f"exit {now['last_exit_code']}{' ' + now['last_reason'] if now.get('last_reason') else ''}" \
              f" at {str(now.get('last_finished_at') or '')[11:19]} UTC"
    else:
        how = "pod replaced or gone"
    return (f"nearRT-RIC crashed mid-run ({how}). The stack recovers by itself (demo keeper / heal); "
            "run the intent again in ~2 min.")


async def _exec(script: str, *, stdin: bytes | None = None, timeout: float = 30) -> str:
    cfg = get_k8s_core_config()
    pod = await _pod()
    args = ["kubectl", "exec", "-n", cfg["namespace"]]
    if stdin is not None:
        args.append("-i")
    args += [pod, "--", "sh", "-c", script]
    proc = await asyncio.create_subprocess_exec(
        *args, stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"RIC pod exec failed: {(err or out).decode(errors='replace').strip()[:300]}")
    return out.decode(errors="replace")


async def check_xapp_present(profile: dict[str, Any]) -> None:
    binary = profile["ric"]["xapp_binary"]
    out = await _exec(f"test -x {shlex.quote(binary)} && echo yes || echo no")
    if out.strip() != "yes":
        raise RuntimeError(
            f"{binary} not found in the RIC pod — the RIC image must be flexric:*-dualbrain "
            "(scripts/build-dualbrain.sh)")


async def write_file(path: str, data: bytes) -> None:
    d = path.rsplit("/", 1)[0]
    await _exec(f"mkdir -p {shlex.quote(d)} && cat > {shlex.quote(path)}", stdin=data)


def xapp_config(profile: dict[str, Any], *, mode: str, reserve_pct: int) -> dict[str, Any]:
    wd = profile["ric"]["workdir"]
    k, c = profile["kpm"], profile["control"]
    return {
        "mode": mode,
        "direction": profile["direction"],
        "model_path": f"{wd}/model.onnx",
        "preprocess_path": f"{wd}/preprocess.json",
        "events_path": f"{wd}/events.jsonl",
        "positive_label": "1",
        "kpm_period_ms": k["period_ms"],
        "window_samples": k["window_samples"],
        "nssai_sst": k["nssai_sst"],
        "nssai_sd": k["nssai_sd"],
        "prob_threshold": c["prob_threshold"],
        "trigger_consecutive": c["trigger_consecutive"],
        "release_after_normal_samples": c["release_after_normal_samples"],
        "min_hold_samples": c["min_hold_samples"],
        "release_below_cell_prb": c.get("release_below_cell_prb", 60),
        "policy": c.get("policy", "ml"),
        "threshold_cell_prb": c.get("threshold_cell_prb", profile["label"]["threshold_pct"]),
        # dualbrain8: threshold policy = persistence baseline, cell PRB >= threshold sustained this many samples
        "min_duration_samples": int(c.get("min_duration_samples", 1)),
        "reserve_pct": reserve_pct,
        "edge_policy": c["edge_policy"],
        "best_effort_slice_id": int(c.get("best_effort_slice_id", 1)),
        # R1: KPM RAN UE ID of the protected UE → cap every other UE together; -1 = cap the congestor only.
        "protected_ue_ref": int(c.get("protected_ue_ref", -1)),
    }


async def start(profile: dict[str, Any], *, mode: str, reserve_pct: int, model: dict[str, bytes] | None = None) -> int:
    """(Re)start the xApp; returns the events.jsonl byte offset to read from."""
    await stop()
    wd = profile["ric"]["workdir"]
    if mode == "control" and model is None and profile["control"].get("policy", "ml") != "threshold":
        present = await _exec(f"test -s {shlex.quote(wd)}/model.onnx && test -s {shlex.quote(wd)}/preprocess.json "
                              "&& echo yes || echo no")
        if present.strip() != "yes":  # RIC pod restarted: redeploy the last trained model
            from services.dual_brain import neuralsmith

            m = profile["model"]  # newest model that passed both gates (rejected ones are stored too)
            acc = await neuralsmith.latest_accepted(float(m["min_accuracy"]), float(m["latency_budget_ms"]))
            if not acc:
                raise RuntimeError("no accepted model on NeuralSmith to redeploy (train one first)")
            model = acc["bundle"]
    for name, data in (model or {}).items():
        await write_file(f"{wd}/{name}", data)
    await write_file(f"{wd}/config.json", json.dumps(xapp_config(profile, mode=mode, reserve_pct=reserve_pct)).encode())
    offset = int((await _exec(f"wc -c < {shlex.quote(wd)}/events.jsonl 2>/dev/null || echo 0")).strip() or 0)
    # The RIC can hold a stale pending subscription for a few minutes after another
    # KPM subscriber went away; keep retrying for ~3 min before giving up.
    attempts = 6
    for attempt in range(1, attempts + 1):
        res = await xapp_bridge.start_k8s_xapp(XAPP, duration=-1)
        if not res.get("success"):
            log = await _exec(f"tail -n 20 /tmp/{XAPP}.log 2>/dev/null || true")
            raise RuntimeError(f"{res.get('message')}: {res.get('error') or ''} {log[-600:]}")
        # KPM must actually flow: a subscription can be accepted while the RIC
        # still has a stale one (e.g. a just-killed KPM monitor) and nothing arrives.
        if await _samples_flowing(profile, offset):
            return offset
        await stop()
        if attempt < attempts:
            await asyncio.sleep(20)
    log = await _exec(f"tail -n 15 /tmp/{XAPP}.log 2>/dev/null || true")
    raise RuntimeError(f"{XAPP} subscribed but no KPM indications arrived ({attempts} tries, ~3 min); "
                       f"RIC/xApp log: {log[-600:]}")


# A fresh xApp's first KPM report arrives ~12-16 s after start on the lab RIC (its E42
# setup is answered only on the xApp's resend; RIC iApp one-read-per-wakeup, held for a
# RIC rebuild). A 10 s window failed most starts and each kill+retry paid the delay again
# (collect took 249 s for 60 s of traffic). Wait long enough instead.
async def _samples_flowing(profile: dict[str, Any], offset: int, timeout_s: float = 30.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(1.0)
        events, _ = await read_events(profile, offset)
        if any(e.get("type") == "sample" for e in events):
            return True
    return False


async def stop() -> None:
    await xapp_bridge.stop_k8s_xapp(XAPP)


async def read_events(profile: dict[str, Any], offset: int) -> tuple[list[dict[str, Any]], int]:
    path = f"{profile['ric']['workdir']}/events.jsonl"
    out = await _exec(f"tail -c +{int(offset) + 1} {shlex.quote(path)} 2>/dev/null || true", timeout=20)
    raw = out.encode()
    # Only consume complete lines (the xApp may be mid-write).
    end = raw.rfind(b"\n") + 1
    events = []
    for line in raw[:end].splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events, offset + end


async def stop_conflicting_kpm_monitor() -> bool:
    """xapp_kpm_moni shares the gNB's per-UE RRU.PrbTot* delta counters; two
    subscribers corrupt each other's values. Returns True if it was running."""
    running = await xapp_bridge._running_xapps()
    if "xapp_kpm_moni" in running:
        # SIGTERM first so it unsubscribes cleanly (a SIGKILLed subscriber leaves the
        # RIC with a stale subscription and pending-event timeouts), then KILL.
        await _exec("pkill -TERM -f '[x]app_kpm_moni' 2>/dev/null || true; "  # up to 15 s for a clean exit
                    "for i in $(seq 1 30); do pgrep -f '[x]app_kpm_moni' >/dev/null || break; sleep 0.5; done; "
                    "pkill -9 -f '[x]app_kpm_moni' 2>/dev/null || true")
        await asyncio.sleep(3)  # let the RIC drop the subscription
        return True
    return False
