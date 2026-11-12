"""Scenario profiles: everything that differs between RFsim and hardware.

profiles/<name>.json ships with the code; DUAL_BRAIN_PROFILE_DIR may point at a
directory of site-specific overrides (same file names). Request-level overrides
are deep-merged on top.
"""
from __future__ import annotations

import copy
import math
import json
import os
from pathlib import Path
from typing import Any

_BUILTIN = Path(__file__).resolve().parent / "profiles"


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def available() -> list[str]:
    names = {p.stem for p in _BUILTIN.glob("*.json")}
    site = os.getenv("DUAL_BRAIN_PROFILE_DIR", "").strip()
    if site and Path(site).is_dir():
        names |= {p.stem for p in Path(site).glob("*.json")}
    return sorted(names)


def load(name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    if not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"bad profile name {name!r}")
    base_path = _BUILTIN / f"{name}.json"
    prof: dict[str, Any] = json.loads(base_path.read_text()) if base_path.is_file() else {}
    site = os.getenv("DUAL_BRAIN_PROFILE_DIR", "").strip()
    if site and (Path(site) / f"{name}.json").is_file():
        prof = _merge(prof, json.loads((Path(site) / f"{name}.json").read_text()))
    if not prof:
        raise ValueError(f"unknown profile {name!r}; available: {', '.join(available())}")
    return _merge(prof, overrides or {})


def for_canvas(graph: dict[str, Any]) -> str:
    """Pick the profile from the canvas gNB node (RFsim vs USRP/N310)."""
    types = {str(n.get("type") or (n.get("data") or {}).get("nodeType") or "") for n in graph.get("nodes") or []}
    if "gnb_usrp" in types:
        return "n310"
    return "rfsim"


ML_BACKENDS = ("neuralsmith",)  # "flaml" lands with the R1 ML-engine wrapper (Exp-05)


def canvas_overrides(graph: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """R1 experiment parameters set on the canvas → profile overrides.

    ZTO-Agent node: ollama_model, ml_backend, t_s_ms (KPM period), collection_min.
    A Virtual UE node with rfsim channel settings is the cell-edge UE (traffic.edge_ue).
    Empty fields keep the profile value. Raises ValueError on out-of-range values.
    """
    def nodes(t: str) -> list[dict[str, Any]]:
        return [n for n in graph.get("nodes") or []
                if str((n.get("data") or {}).get("nodeType") or n.get("type") or "") == t]

    out: dict[str, Any] = {}
    zto = nodes("zto_agent")
    cfg = ((zto[0].get("data") or {}).get("config") or {}) if zto else {}
    if str(cfg.get("ollama_model") or "").strip():
        out["intent"] = {"llm_model": str(cfg["ollama_model"]).strip()}
    backend = str(cfg.get("ml_backend") or "").strip()
    if backend:
        if backend not in ML_BACKENDS:
            raise ValueError(f"ML backend {backend!r} is not available yet (have: {', '.join(ML_BACKENDS)})")
        out["model"] = {"backend": backend}
    if cfg.get("t_s_ms") not in (None, ""):
        t_s = int(cfg["t_s_ms"])
        if not 100 <= t_s <= 1000:
            raise ValueError("T_s (KPM period) must be 100..1000 ms")
        out["kpm"] = {"period_ms": t_s}
    if cfg.get("collection_min") not in (None, ""):
        minutes = float(cfg["collection_min"])
        if not 1 <= minutes <= 60:
            raise ValueError("collection_min must be 1..60")
        t = base["traffic"]
        out["collect"] = {"cycles": max(1, math.ceil(minutes * 60 / (int(t["on_s"]) + int(t["off_s"])))),
                          "requested_min": minutes}
    edge = [n for n in nodes("virtual_ue")
            if any(((n.get("data") or {}).get("config") or {}).get(k) not in (None, "")
                   for k in ("channel_ploss_db", "channel_noise_db"))]
    if len(edge) > 1:
        raise ValueError("RFsim channel is set on more than one Virtual UE; R1 has one cell-edge UE")
    if edge:
        ecfg = (edge[0].get("data") or {}).get("config") or {}
        imsi = str(ecfg.get("imsi") or "").strip()
        if imsi:
            from services.k8s.experiment_ran import ue_channel

            out["traffic"] = {"edge_ue": {"select": "imsi", "value": imsi}}
            try:
                channel = ue_channel(ecfg)
            except RuntimeError as exc:  # ExperimentRanError: bad dB values
                raise ValueError(str(exc)) from exc
            # Edge = UE-side DL channel, verified live on DL MCS (channel_guard.EDGE_DEFINITION).
            from services.dual_brain.channel_guard import EDGE_DEFINITION, UL_LIMIT

            out["channel_guard"] = {"edge_imsi": imsi, "min_gap_dl_mcs": EDGE_DEFINITION["min_gap"],
                                    "channel": channel, "edge_definition": EDGE_DEFINITION, "ul_limit": UL_LIMIT}
    return out
