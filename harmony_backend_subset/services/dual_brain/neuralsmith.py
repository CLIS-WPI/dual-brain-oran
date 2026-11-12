# CLIENT INTERFACE ONLY: NeuralSmith is proprietary and not included in this release.
# Request/response contract: r1/mlengine/NEURALSMITH_CONTRACT.md. Open replacement: r1/mlengine/flaml_backend.py.
"""Client for the NeuralSmith node (k8s Deployment harmony-core/harmony-neuralsmith).

NEURALSMITH_URL: in-cluster http://harmony-neuralsmith.harmony-core:8090; host
dev default http://127.0.0.1:18090 (port-forward kept alive automatically).
"""
from __future__ import annotations

import base64
import os
from typing import Any

import httpx

from services.k8s import port_forward

NS = os.getenv("NEURALSMITH_NAMESPACE", "harmony-core")
SERVICE = "harmony-neuralsmith"
URL = os.getenv("NEURALSMITH_URL", "http://127.0.0.1:18090").rstrip("/")


async def _reachable() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            return (await c.get(f"{URL}/livez")).status_code == 200
    except httpx.HTTPError:
        return False


async def _ensure_forward() -> dict[str, Any]:
    return await port_forward.ensure("neuralsmith", NS, SERVICE, URL, 8090, _reachable)


async def health() -> dict[str, Any]:
    """{"ok": bool, ...} — ok only when the service self-test training passed."""
    pf = await _ensure_forward()
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{URL}/health")
        body = r.json()
        body["ok"] = bool(r.status_code == 200 and body.get("ok"))
        return body
    except (httpx.HTTPError, ValueError) as exc:
        return {"ok": False, "stage": "unreachable", "detail": str(exc), "port_forward": pf}


async def train(csv_bytes: bytes, *, latency_budget_ms: float, target_column: str = "congested",
                positive_label: str = "1", timeout_s: float = 1800) -> dict[str, Any]:
    await _ensure_forward()
    async with httpx.AsyncClient(timeout=timeout_s) as c:
        r = await c.post(
            f"{URL}/train",
            files={"file": ("telemetry.csv", csv_bytes, "text/csv")},
            data={"latency_budget_ms": str(latency_budget_ms), "target_column": target_column,
                  "positive_label": positive_label},
        )
    if r.status_code != 200:
        raise RuntimeError(f"NeuralSmith /train {r.status_code}: {r.text[:400]}")
    out = r.json()
    out["onnx"] = base64.b64decode(out.pop("onnx_b64"))
    return out


async def latest_accepted(min_accuracy: float, latency_budget_ms: float, *, limit: int = 50) -> dict[str, Any] | None:
    """Newest stored model whose own report passed both gates (accuracy and latency).
    NeuralSmith also keeps rejected models, so "newest" alone is not safe to deploy.
    Returns {model_id, report, bundle: {model.onnx, preprocess.json}} or None."""
    await _ensure_forward()
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{URL}/models", params={"limit": limit})
        r.raise_for_status()
        for m in r.json().get("models", []):
            rep = m.get("report") or {}
            p99 = (rep.get("inference_latency_ms") or {}).get("p99")
            if (rep.get("accuracy") or 0) < min_accuracy or not rep.get("meets_latency_budget") \
                    or p99 is None or p99 > latency_budget_ms:
                continue
            bundle = {}
            for name in ("model.onnx", "preprocess.json"):
                f = await c.get(f"{URL}/models/{m['model_id']}/{name}")
                if f.status_code != 200:
                    break
                bundle[name] = f.content
            else:
                return {"model_id": m["model_id"], "report": rep, "bundle": bundle}
    return None


async def latest_bundle() -> dict[str, bytes]:
    """model.onnx + preprocess.json of the newest trained model (persisted on the
    NeuralSmith PVC) — used to redeploy after a RIC pod restart wiped its /tmp."""
    await _ensure_forward()
    out: dict[str, bytes] = {}
    async with httpx.AsyncClient(timeout=30) as c:
        for name in ("model.onnx", "preprocess.json"):
            r = await c.get(f"{URL}/models/latest/{name}")
            if r.status_code != 200:
                raise RuntimeError(f"NeuralSmith has no trained model yet ({name}: {r.status_code})")
            out[name] = r.content
    return out
