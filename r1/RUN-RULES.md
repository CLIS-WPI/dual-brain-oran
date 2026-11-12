# COMMAG-R1 Dual-Brain reproduction

Testbed: Harmony canvas **COMMAG-R1 Dual-Brain** (isolated copy of *MILCOM Dual-Brain demo*,
which stays the frozen baseline). Feature notes: [../docs/features/dual-brain-demo.md](../docs/features/dual-brain-demo.md) (section "R1 canvas").
`results/` keeps summaries and final CSVs only; raw telemetry, datasets and per-run logs stay local (`.gitignore`).

## Measurement-run rules (all experiment runs; calibration is exempt from 1–2)

1. **Interleaved conditions.** Any A/B comparison (Exp-03 xApp OFF/ON, Exp-05 backends when run
   on-stack, Exp-06 models) runs in alternating order A, B, A, B, … — never blocked — so host-load
   drift hits both arms equally. Each run records its position in the order and its sim-speed mean.
2. **Start gate.** Before a run starts: record host load (load avg + other users' CPU) and the current
   sim/wall speed baseline. If the baseline is > 15 % off the calibration mean, wait and retry instead
   of starting; every skipped start is logged (time, baseline, load).
3. **Run conditions.** MILCOM RAN/RIC stopped; no builds or tests on the host; R1 pods are not
   CPU-pinned (kubelet CPU manager policy `none`; pinning would need a kubelet restart). Host load avg
   at start/end goes into `meta.json`.
4. **Never restart the Operator during an R1 run** (issue #53: it re-applies stale UE replica counts).
5. **Guard aborts** (`channel_guard`): `pod_changed` (crash logs saved to
   `results/calibration/<run_id>/` first), `ul_traffic_detected`, `sim_speed_drift` (±10 % on 10 s means,
   two windows), plus the cell-edge/identity check.
6. **GPU0 (Ollama, Exp-06).** One model resident at a time (unload + `/api/ps` empty before the next).
   `env/versions.json` → `ollama.gpu0_coresident` records every non-Ollama process on GPU0; today that is
   the Harmony portal backend's Knowledge Center embedder (`all-MiniLM-L6-v2`, ~0.6 GiB, uvicorn worker —
   its PID changes on every reload). `ollama_env.gpu0_quiet()` (nvidia-smi pmon) must report no other
   process computing on GPU0 before and during Exp-06; don't use the Knowledge Center / AI chat meanwhile.
   The other server user's jobs run on GPU1.
7. **Portal backend must not reload mid-run.** The dev server runs `uvicorn --reload`; any edit to a
   file it watches (`testbed-manager/backend/**`) replaces the worker, and a pipeline run inside it dies.
   During a measurement run either (a) freeze edits for the run's duration, or (b) run against a
   backend started **without** `--reload` from the clean worktree at the run's commit. Guard:
   `backend_reloaded` — script-driven runs record the worker (PID + start time, `scripts/run_guard.py`)
   at start and abort if it changes; pipeline runs record it in `meta.json`, and a worker that starts
   while a run is still marked running marks that run failed with `backend_reloaded`.
8. **Start order: RIC → gNB → UE1 → UE2 → UE3** (each UE after the previous one has its PDU session).
   The gNB resolves the RIC's headless service at start; with no RIC pod it falls back to `127.0.0.1`
   and E2 never connects. After any RIC restart, restart the gNB (and UEs) too.
9. **Clock.** RFsim is CPU-bound, not real time. All reported times (T_s, horizon, lead time, RTT) are
   wall-clock; report the mean sim/wall speed ratio next to them.

## Targets

- `make versions` → `env/versions.json` (Ollama version, container settings, per-model digest + quantization, all Q4_K_M)
- `make warm` → one JSON prompt per model, one model resident on GPU0 at a time
