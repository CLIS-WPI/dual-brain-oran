#!/usr/bin/env python3
"""R1 Ollama environment: versions record + one-model-at-a-time warm-up.

  python3 r1/scripts/ollama_env.py versions   # env/versions.json["ollama"]
  python3 r1/scripts/ollama_env.py warm       # one JSON prompt per model, sequential

Uses the existing Ollama container through its API; never changes the container. Guards check the GPU the
container actually uses (ollama_gpu(): GPU1 since 2026-11-04), never a hard-coded index.
Only one model is resident at a time: each model is unloaded (keep_alive=0) and /api/ps is
confirmed empty before the next one loads, so co-resident models cannot skew latencies.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11436")
CONTAINER = os.environ.get("OLLAMA_CONTAINER", "ollama")
# The four R1 models (brief §2.4). All must be Q4_K_M (same quantization across the comparison).
MODELS = ["llama3.1:8b", "qwen2.5-coder:7b", "gemma4:26b-a4b", "llama3.3:70b"]
# Every R1 LLM call runs non-thinking: "think": false (0.40.13: disables gemma4 thinking; a no-op for the
# three models without thinking support, which reject "think": true). Recorded per model in versions.json.
THINK = False
QUANT = "Q4_K_M"
ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ROOT / "env" / "versions.json"


def api(path: str, body: dict | None = None, timeout: float = 900) -> dict:
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def _ollama_pids() -> set[int]:
    out = subprocess.run(["docker", "top", CONTAINER, "-o", "pid"], capture_output=True, text=True).stdout
    return {int(x) for x in out.split()[1:] if x.isdigit()}


def ollama_gpu() -> dict:
    """The host GPU the Ollama container computes on (its first visible GPU): host index, UUID, name, memory."""
    uuid = subprocess.run(["docker", "exec", CONTAINER, "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout.split()[0]
    for line in subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,vbios_version,driver_version",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.splitlines():
        idx, u, name, mem, vbios, drv = [x.strip() for x in line.split(",")]
        if u == uuid:
            return {"index": int(idx), "uuid": u, "name": name, "memory_total_mib": int(mem), "vbios": vbios, "driver": drv}
    raise SystemExit(f"Ollama GPU {uuid} not found on the host")


def gpu_state(gpu: dict | None = None) -> dict:
    """Utilization and memory of Ollama's GPU (nvidia-smi)."""
    g = gpu or ollama_gpu()
    u, used, total = subprocess.run(["nvidia-smi", "-i", str(g["index"]), "--query-gpu=utilization.gpu,memory.used,memory.total",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.strip().split(", ")
    return {"gpu_index": g["index"], "gpu_uuid": g["uuid"], "util_pct": int(u), "mem_used_mib": int(used), "mem_total_mib": int(total)}


def gpu_coresident(gpu: dict | None = None) -> list[dict]:
    """Non-Ollama compute processes on Ollama's GPU (pid, command, memory, role)."""
    uuid0 = (gpu or ollama_gpu())["uuid"]
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout.splitlines()
    ollama, rows = _ollama_pids(), []
    for line in apps:
        pid, mem, uuid = [x.strip() for x in line.split(",")]
        if uuid != uuid0 or int(pid) in ollama:
            continue
        try:
            args = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            args, cwd = "?", "?"
        role = ("Harmony portal backend (uvicorn worker): Knowledge Center embedder SentenceTransformer "
                "all-MiniLM-L6-v2, services/knowledge_base/vector_store.py; computes only on KB query/ingest"
                if cwd.endswith("testbed-manager/backend") else "unknown")
        rows.append({"pid": int(pid), "used_mib": int(mem), "cwd": cwd, "cmd": args[:200], "role": role})
    return rows


def gpu_quiet(samples: int = 3, gpu: dict | None = None) -> list[str]:
    """Problems if any non-Ollama process computes on Ollama's GPU (nvidia-smi pmon SM %), else []."""
    idx = str((gpu or ollama_gpu())["index"])
    ollama = _ollama_pids()
    out = subprocess.run(["nvidia-smi", "pmon", "-c", str(samples), "-s", "u"], capture_output=True, text=True).stdout
    busy = []
    for line in out.splitlines():
        f = line.split()
        if len(f) > 4 and f[0] == idx and f[1].isdigit() and int(f[1]) not in ollama and f[3] not in ("-", "0"):
            busy.append(f"pid {f[1]} ({f[-1]}) SM {f[3]}% on GPU{idx}")
    return sorted(set(busy))


def start_gpu_watch(out_dir: str, gpu: dict | None = None, interval_s: float = 20) -> None:
    """Mid-run guard: a daemon thread checks gpu_quiet every interval_s; if another process computes on Ollama's GPU it
    writes <out_dir>/gpu_abort.json and kills this run's process group (children included), so the run stops and is
    marked aborted instead of continuing on a shared GPU."""
    import signal
    import threading
    g = gpu or ollama_gpu()

    def watch():
        while True:
            time.sleep(interval_s)
            busy = gpu_quiet(gpu=g)
            if busy:
                Path(out_dir, "gpu_abort.json").write_text(json.dumps(
                    {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gpu": g, "busy": busy, "coresident": gpu_coresident(g)}, indent=1))
                print(f"GPU{g['index']} occupied mid-run: {busy} -> abort", flush=True)
                os.killpg(os.getpgid(0), signal.SIGTERM)

    threading.Thread(target=watch, daemon=True).start()


def container_settings() -> dict:
    out = subprocess.run(["docker", "inspect", CONTAINER, "--format",
                          "{{json .Config.Env}}|{{json .HostConfig.DeviceRequests}}|{{.Config.Image}}"],
                         capture_output=True, text=True, check=True).stdout.strip()
    env, dev, image = out.split("|", 2)
    keep = ("OLLAMA_", "CUDA_VISIBLE_DEVICES")
    return {"container": CONTAINER, "image": image,
            "env": {k: v for k, v in (e.split("=", 1) for e in json.loads(env)) if k.startswith(keep)},
            "gpu_device_ids": [d for r in json.loads(dev) or [] for d in (r.get("DeviceIDs") or [])]}


def model_record(name: str, tags: dict[str, dict]) -> dict:
    show = api("/api/show", {"model": name})
    d, info = show.get("details") or {}, show.get("model_info") or {}
    arch = info.get("general.architecture", "")
    rec = {"tag": name, "digest": (tags.get(name) or {}).get("digest"), "size_bytes": (tags.get(name) or {}).get("size"),
           "quantization": d.get("quantization_level"), "parameter_size": d.get("parameter_size"),
           "family": d.get("family"), "context_length_model": info.get(f"{arch}.context_length")}
    if info.get(f"{arch}.expert_count"):
        rec["moe"] = {"experts": info[f"{arch}.expert_count"], "experts_used": info.get(f"{arch}.expert_used_count")}
    return rec


def versions() -> dict:
    tags = {m["name"]: m for m in api("/api/tags").get("models", [])}
    missing = [m for m in MODELS if m not in tags]
    if missing:
        sys.exit(f"missing models in {CONTAINER}: {missing} (pull them first)")
    settings = container_settings()
    rec = {"version": api("/api/version").get("version"), **settings,
           "num_ctx": int(settings["env"].get("OLLAMA_CONTEXT_LENGTH", 0)) or None,
           "models": [model_record(m, tags) for m in MODELS],
           "gpu": ollama_gpu(), "gpu_coresident": gpu_coresident()}
    wrong = [m["tag"] for m in rec["models"] if m["quantization"] != QUANT]
    if wrong:
        sys.exit(f"not {QUANT}: {wrong} — pull the explicit -q4_K_M tag")
    VERSIONS.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(VERSIONS.read_text()) if VERSIONS.is_file() else {}
    old = data.get("ollama") or {}
    for k in ("residency", "think_policy"):  # recorded by gpu0_residency.py / the think check; keep them
        if k in old:
            rec[k] = old[k]
    think = {m["tag"]: m.get("think") for m in old.get("models", []) if m.get("think")}
    for m in rec["models"]:
        if m["tag"] in think:
            m["think"] = think[m["tag"]]
    data["ollama"] = rec
    VERSIONS.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {VERSIONS.relative_to(ROOT.parent)} (ollama {rec['version']}, {len(rec['models'])} models, all {QUANT})")
    return rec


def unload_all() -> None:
    for m in api("/api/ps").get("models", []):
        api("/api/generate", {"model": m["name"], "keep_alive": 0})
    for _ in range(60):
        if not api("/api/ps").get("models"):
            return
        time.sleep(1)
    sys.exit("could not unload models from the Ollama GPU")


def warm() -> list[dict]:
    busy = gpu_quiet()
    if busy:  # measurements must not share the Ollama GPU with another computing process
        sys.exit("Ollama GPU not quiet: " + "; ".join(busy))
    rows = []
    for m in MODELS:
        unload_all()  # one model resident at a time
        t0 = time.monotonic()
        r = api("/api/generate", {"model": m, "prompt": 'Reply with exactly this JSON: {"ok": true}',
                                  "format": "json", "stream": False, "think": THINK,
                                  "options": {"temperature": 0, "seed": 1}})
        ps = next((x for x in api("/api/ps").get("models", []) if x["name"] == m), {})
        try:
            ok = json.loads(r.get("response") or "{}").get("ok") is True
        except ValueError:
            ok = False
        rows.append({"model": m, "json_ok": ok, "wall_s": round(time.monotonic() - t0, 1),
                     "load_s": round((r.get("load_duration") or 0) / 1e9, 1),
                     "vram_gb": round((ps.get("size_vram") or 0) / 1e9, 1), "size_gb": round((ps.get("size") or 0) / 1e9, 1)})
        print(f"{m:18} json_ok={ok}  load {rows[-1]['load_s']} s  VRAM {rows[-1]['vram_gb']} / {rows[-1]['size_gb']} GB")
    unload_all()
    if not all(r["json_ok"] for r in rows):
        sys.exit("warm-up: not every model returned the JSON reply")
    return rows


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "versions":
        versions()
    elif cmd == "warm":
        warm()
    else:
        sys.exit(__doc__)
