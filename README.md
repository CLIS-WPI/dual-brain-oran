# COMMAG-R1 Dual-Brain — code release

Code to reproduce the results of the COMMAG-R1 Dual-Brain paper: an operator intent is turned by an LLM agent
(ZTO-Agent) into a validated task spec, a congestion-prediction model is trained on testbed telemetry, a verification
gate checks spec, configuration and model, and a fixed FlexRIC xApp reserves PRBs for a cell-edge UE before
congestion. Testbed: OAI 5G SA (RFsim) on Kubernetes with FlexRIC.

- **Data:** raw telemetry, per-run logs, datasets and summaries are in the IEEE DataPort archive
  `commag-r1-20261112.tar.gz` (716 files, sha256 per file in its `manifest.json`). This release holds code only.
- **Not included:** OAI and FlexRIC source trees (we ship patches against pinned commits), NeuralSmith (proprietary;
  client interface only, see `r1/mlengine/NEURALSMITH_CONTRACT.md`), the Harmony portal (see "Testbed runners"),
  container images, Ollama model weights, raw logs.
- `MANIFEST.sha256` lists every file of this release (`sha256sum -c MANIFEST.sha256`).

## Layout

| Path | What |
|---|---|
| `patches/oai/` | 4 patches against OAI RAN `09c9bb99` (+ its FlexRIC submodule `ef6d722`): per-slice DL PRB quota in the NR MAC scheduler (AURA), SLICE SM control programs the quota, level-triggered E2 agent socket, protected-UE-first allocation + `[AURA-PRB]` 1 s ground-truth log. `prepare-oai-src.sh` clones the pinned commits and applies them in order. Design notes: `r1/docs/aura3.md` |
| `patches/flexric/` | 3 patches against FlexRIC `340c36bc` (RIC side): robustness fixes, level-triggered E42 iApp socket, the Dual-Brain congestion xApp + its build (ONNX Runtime 1.30.0, cJSON) |
| `xapp/dual_brain/` | the deployed xApp source (image tag `kpm-sd-dualbrain11`) as plain files, identical to patch 0003: `xapp_congestion_ctrl.c` (KPM → features → ONNX → debounce → SLICE control, lease + re-arm), `cc_model.c` (preprocess + ONNX), `cc_policy.h`, `cc_replay.c` (offline replay used by the gate's parity check), unit tests |
| `images/gnb/`, `images/ric/` | Dockerfiles and build script for the gNB (`oai-gnb:2026.w27-uhd410-e2-aura3`) and RIC (`flexric:kpm-sd-dualbrain11`) images |
| `r1/zto/` | ZTO-Agent: `spec.schema.json` (task-spec schema), `defaults.json`, `catalog/`, `prompts/` (frozen prompts `full.txt`, `sparse.txt`, `judge.txt` + `lock.json` with their sha256; `agent.py` refuses to run on a hash mismatch), `agent.py` (intent → spec, schema-constrained), `resolve.py`, `render.py` + `xapp/config.json.j2` (render template), `gate.py` (verification gate: all checks incl. `model_parity` via `parity.py` and the lease/re-arm replay), `lease.py`, `deploy.py` (registration: config + model to the RIC byte for byte), `gate_inject.py` (Exp-08 fault injection), `intents/` (reference + vague intents), `FIELDS.md` |
| `r1/dataset/` | `build.py` (telemetry → windowed features + sustained-congestion label at t+H, train/test split by cycle), `evaluate.py` (onset recall, lead time, FPR; row metrics) |
| `r1/mlengine/` | ML engine with the paper's training contract: `server.py` (POST /train, GET /jobs), `flaml_backend.py` (FLAML, open), `neuralsmith_backend.py` (client only), `common.py`, `predict.py`, `client.py`; `requirements.in` / `requirements.lock` |
| `r1/collector/` | KPM collector with start gate and run guards (`kpm_collector.py`), sim-speed + gNB-log-coverage rules (`speed_rules.py`, unit test) |
| `r1/experiments/` | runners and analysis per experiment (table below), `exp03_plan.md` |
| `r1/traffic/burst.py` | seeded DL burst schedules for the center UEs (traffic profile) |
| `r1/common/timing.py` | timing marks (`timing.jsonl`) used for every reported duration |
| `r1/scripts/` | testbed helpers (`r1env.py`, `r1_restart.py`, `run_guard.py`, `smoke.py`, `ollama_env.py`, `gpu0_residency.py`, `host_sampler.py`, `hw_info.sh`), calibration (`calib_channel.py`, `calib_ramp.py`, `ramp_rise.py`, `ramp_hold_compare.py`, `profile_check.py`, `ue3_capacity.py`, `oracle_check.py`, `backlog_onset.py`), reporting (`timing_report.py`, `e2e_breakdown.py`, `list_slots.py`, `package_dataport.py`) |
| `r1/paper/` | figure scripts (`fig3_cycle_timeline.py`, `fig4_edge_ue_outcome.py`), `cycle_timeline.json` (Fig. 3 medians), `listing_example.json` (the spec listing), `slots.txt` |
| `r1/env/` | recorded environment: `versions.json` (images, commits, packages, Ollama), `hardware.json`, `gpu0_residency.json`, `experiment.json` (testbed identity; IMSIs are placeholders) |
| `r1/RUN-RULES.md` | measurement-run rules applied to every run (interleaving, start gate, guards, start order) |
| `harmony_backend_subset/` | the Harmony portal modules the runners import: `services/dual_brain/{ric,channel_guard,gnb_truth,profiles,neuralsmith}.py` + `profiles/rfsim.json`, `services/k8s/{experiment_ctx,experiment_ran}.py` |
| `testbed/` | Helm charts (`oai-5g-core`, `oai-rfsim` incl. UEs + traffic generator, `oai-ric`), `values-r1-rfsim.yaml` (R1 overrides incl. edge-UE channel), `gnb.rfsim.band77.51prb.conf`, `canvas_r1.json` (canvas topology), `ollama_models.json` (tags + digests) |
| `run_offline_checks.sh` | offline self-tests + reproduction of the paper's analysis from the DataPort archive |

## Versions

| Component | Version |
|---|---|
| OAI RAN | `09c9bb998fa2f30f2df452a0a9002787280389af`, https://gitlab.eurecom.fr/oai/openairinterface5g.git, + `patches/oai/0001-0004`; base image `oaisoftwarealliance/oai-gnb:2026.w27` + UHD 4.10.0.0 |
| FlexRIC in the gNB (E2 agent) | OAI submodule `ef6d722f22191eea74089966983da1f5ec1fedd4` (+ `patches/oai/0003`); E2AP v3, KPM v3.00 |
| FlexRIC RIC + xApp | `340c36bc8385dc0b9f5d8b2d51d16ff288acee79`, https://gitlab.eurecom.fr/mosaic5g/flexric.git (branch dev), + `patches/flexric/0001-0003`; ONNX Runtime 1.30.0 (C API) |
| OAI UE | `oaisoftwarealliance/oai-nr-ue:2026.w27` |
| Ollama | 0.40.13; models in `testbed/ollama_models.json` (llama3.1:8b, qwen2.5-coder:7b, gemma4:26b-a4b, llama3.3:70b; all Q4_K_M, num_ctx 8192) |
| ML engine (FLAML path) | Python 3.10; `r1/mlengine/requirements.lock` (FLAML 2.7.0, lightgbm 4.7.0, xgboost 2.1.4, scikit-learn 1.7.2, numpy 1.26.4, onnx 1.23.2, onnxruntime 1.23.2, skl2onnx 1.20.0, onnxmltools 1.16.0) |
| Analysis / figures | Python 3.10.12; `r1/requirements-analysis.txt` (numpy 1.26.4, scipy 1.15.3, matplotlib 3.10.7, jsonschema 4.26.0, Jinja2 3.1.3, PyYAML 5.4.1) |
| Runners | `r1/requirements-runner.txt` (portal backend venv) |
| Host | 2× AMD EPYC 9354 (64 logical cores), 377 GiB RAM, 2× NVIDIA H100 NVL (driver 580.126.09, CUDA 13.0), Ubuntu 22.04.5, kernel 6.8, Docker 29.4.0, Kubernetes v1.36 (minikube) |

## Rebuild the testbed

1. **Kubernetes:** single node (minikube, Docker driver). CPU placement: since 2026-11-05 14:41 the minikube container
   is restricted to NUMA node 1 (`docker update --cpuset-cpus 32-63 --cpuset-mems 1 minikube`), away from both GPUs
   and other users' jobs on node 0. Inside the cluster nothing is pinned (kubelet CPU manager policy `none`).
   Details: `r1/env/hardware.json`. The RFsim gNB is CPU-bound and not real-time: every run records its sim/wall speed.
2. **Images:**
   ```
   docker build --build-arg GNB_BASE_IMAGE=oaisoftwarealliance/oai-gnb:2026.w27 \
     -t localhost/harmony/oai-gnb:2026.w27-uhd410 images/gnb               # UHD 4.10 base
   images/gnb/build-rfsim-e2.sh      # patches/oai/prepare-oai-src.sh (OAI 09c9bb99 + FlexRIC ef6d722 + 0001-0004 → .build/oai-ran),
                                     # E2 build in a container → oai-gnb:2026.w27-uhd410-e2-aura3
   git clone https://gitlab.eurecom.fr/mosaic5g/flexric.git && cd flexric && git checkout 340c36bc8385dc0b9f5d8b2d51d16ff288acee79
   git apply ../patches/flexric/000{1,2,3}-*.patch
   docker build --target oai-flexric-builder -t localhost/harmony/flexric-builder:dualbrain -f docker/Dockerfile.flexric.ubuntu .
   docker build -t localhost/harmony/flexric:kpm-sd -f docker/Dockerfile.flexric.ubuntu .               # base RIC image
   docker build -f docker/Dockerfile.dualbrain-layer --build-arg BASE_RIC_IMAGE=localhost/harmony/flexric:kpm-sd \
     -t localhost/harmony/flexric:kpm-sd-dualbrain11 .                                               # + xApp, cc_replay
   ```
   `prepare-oai-src.sh` uses `OAI_MIRROR` (a local OAI checkout) if set, else the upstream remotes.
   Our lab base RIC image `flexric:kpm-sd` was built from the same FlexRIC tree; the layer only adds the xApp, ONNX
   Runtime and a `nearRT-RIC` built from the patched tree.
3. **Core:** `helm install core testbed/helm-charts/oai-5g-core` with PLMN 208/95, DNN `oai`, slice SST 1 / SD 000001.
   The subscriber database dump is not shipped (it holds SIM keys): provision three subscribers (UE1, UE2, UE3) with
   your IMSIs/K/OPc using the OAI CN5G `oai_db.sql` schema and set the MySQL passwords in `values.yaml`.
4. **RAN, UEs, traffic generator, RIC:** `helm install` `testbed/helm-charts/oai-ric` and `oai-rfsim` with the two
   documents of `testbed/values-r1-rfsim.yaml`. Start order matters: RIC → gNB → UE1 → UE2 → UE3 (each UE after the
   previous one has its PDU session; the gNB resolves the RIC at start). Cell: band 78, 51 PRB, 30 kHz SCS.
   Edge UE (UE3): UE-side RFsim AWGN DL channel, path loss 30 dB, noise power −10 dB (its DL MCS drops from ~27 to ~14).
   Topology as configured on the portal canvas: `testbed/canvas_r1.json`.
5. **Traffic:** UE1 and UE2 get seeded ramp bursts from `r1/traffic/burst.py` (iperf2 reverse UDP from the traffic
   generator pod). Exp-03/04 profile `ramp`: base 5 → peak 28 Mbps per UE over 45 s (2 s steps), hold 10 s, down 5 s,
   OFF 30 ± 2 s; 13 cycles per 20-min run; the exact schedule and seed of every run is in its `meta.json` /
   `run_meta.json` (`schedule`). UE3 (protected): constant 7 Mbps UDP DL + 10 Hz ping to the UPF.
6. **Ollama:** 0.40.13 on one H100, env in `testbed/ollama_models.json` (`server_env`); `ollama pull <tag>` and check
   the digests. One model resident at a time, except Exp-04 (orchestrator llama3.1:8b + judge llama3.3:70b).
7. **ML engine:** `python3.10 -m venv r1/.venv-mlengine && r1/.venv-mlengine/bin/pip install -r r1/mlengine/requirements.lock`,
   then `r1/.venv-mlengine/bin/python r1/mlengine/server.py --backend flaml --port 8095`.
8. **Check:** `make -C r1 smoke` (RAN with 3 UEs, KPM rate, SLICE control accepted, ML engine, ZTO self-checks, model
   parity in the RIC pod, Ollama models).

### Testbed runners and the Harmony portal

The testbed runners (everything that starts pods, traffic or xApps) were executed in the Harmony portal backend's
Python environment: they import `harmony_backend_subset/` (included) and the portal's Kubernetes bridges (not
included; Harmony platform code). What they use from those bridges is small and replaceable with `kubectl`:
`core_bridge.get_k8s_core_config()` (namespace/tenant) and `_kubectl_json(args)`; `ran_bridge.collect_k8s_virtual_ues()`
(UE pods → IMSI, RNTI, IP); `xapp_bridge.start_k8s_xapp / stop_k8s_xapp / _get_ric_pod / _running_xapps` (exec in the
RIC pod); `port_forward.ensure()`. `r1_restart.py` and `calib_channel.py` also call the portal REST API (bearer token
read from a local file, never stored in code). Put `harmony_backend_subset` (and your bridge implementation) on
`PYTHONPATH`, set `DEPLOYMENT_MODE=k8s K8S_TENANT_NAME=<tenant>`, and fill `r1/env/experiment.json`.

## Rerun the experiments

All runners write to `r1/results/<exp>/<run_id>/` and follow `r1/RUN-RULES.md`.

| Paper | Runner (testbed) | Analysis (offline) |
|---|---|---|
| Classifier selection (exp01-select) | `experiments/exp01_dataset.py --run-id exp01-<date> --ts 100,250,500` (one 20-min frozen-traffic collection per T_s, grid of H, NeuralSmith + FLAML + persistence) | grid in `exp01_summary.json`; final defaults in `zto/defaults.json` (T_s 500 ms, H 2.5 s) |
| Classifier evaluation (exp01-eval) | `experiments/exp01_eval.py --run-id exp01-eval-<date> --select-run exp01-20261104 --ts 500 --horizon 2500 --exploratory 7500 --seeds 7,8` | `dataset/evaluate.py`; deployed models: `experiments/exp01_eval_deployed.py` |
| Exp-02 spec consistency | `experiments/exp02_consistency.py --run-id <id>` (3 intents × 4 LLMs × 5 seeds × full/sparse) | `exp02_summary.json` |
| Exp-03 outcome (arms a/b/c, d exploratory) | `experiments/exp03_prep.py --exp04-batch exp04-20261105d --out results/exp03/<batch>-arms`, then `experiments/exp03.py dry …` and `exp03.py batch --arms-dir … --batch <batch> --deadline …` (3 interleaved runs per arm × 13 cycles) | `experiments/exp03_analyze.py results/exp03/<batch>`; Fig. 3 / Fig. 4 scripts |
| Exp-04 end-to-end | `experiments/exp04_batch.py --deadline … --batch exp04-<date>` (dry run + 6 official runs of `exp04_e2e.py`: 3 × 20-min, 3 × 10-min collection) | `scripts/timing_report.py <runs> --group-by collection_min`, `scripts/e2e_breakdown.py` |
| Exp-06 latency | `experiments/exp06_latency.py --run-id <id> --exp01-config results/exp01/exp01-20261104/ts500_h2500` | `exp06_summary.json` |
| Exp-08 gate | `experiments/exp08_gate.py --run-id <id>` ((a) `zto/gate_inject.py` fault injection, (b) the ten vague intents) | `exp08_summary.json`, `exp08_gate_results_summary.json` |
| Calibration | `scripts/calib_channel.py 30 -10`, `calib_ramp.py`, `ue3_capacity.py`, `oracle_check.py --ue3-rate 7` | `profile_check.py`, `ramp_rise.py`, `backlog_onset.py` |

Exp-07 (manual-development baseline) is a logged human study and has no runner.

## Offline reproduction from the DataPort archive

```
./run_offline_checks.sh /path/to/commag-r1-20261112.tar.gz   # PY=python3 MLPY=r1/.venv-mlengine/bin/python RIC_IMAGE=<RIC image>
```
Extracts the archive into `r1/` and runs:

- unit self-tests: sim-speed rules, timing marks, traffic schedule, Exp-03 analysis on synthetic data, ZTO
  schema/render/gate self-checks + frozen prompt hashes + bounded template slots, xApp policy/association C tests;
- recomputation compared with the archived values: Exp-03 summary and per-cycle table, Fig. 3 medians, Fig. 4,
  Exp-04 timing summary and end-to-end breakdown, the exp01-select dataset (byte-identical rebuild from raw telemetry);
- classifier scores on **exactly the split the paper reports** (exp01-eval, two runs pooled, 4,580 rows, 26 onsets)
  for NeuralSmith, FLAML and persistence, from the archived predictions — identical to the paper's values;
- the FLAML training path end to end: retrain on the exp01-select training split (cycles 1–9), score on exp01-eval
  pooled. FLAML's time-budgeted search (budget = NeuralSmith's training time, 12.2 s) is not bit-reproducible, so a
  retrain can pick other hyperparameters; its scores are printed next to the archived model's, not asserted equal;
- the deployed models (Exp-03 arm (c) = Exp-04 s1 artifact, H 2.5 s; arm (d), H 7.5 s) scored on exp01-eval
  (`exp01_eval_deployed.py`), and the gate's model-parity + lease/re-arm replay of both models through the xApp's own
  C code path (`cc_replay`), run offline with `docker run` of the RIC image (`zto/parity.py --ns docker`), compared
  with the gate records. Needs Docker and the RIC image built from `images/ric` (skipped otherwise).

Not reproducible offline: NeuralSmith training (proprietary; its outputs are archived) and LLM calls (need Ollama).

## Paper → script → DataPort file

DataPort paths are relative to the archive root (`results/…`). Exp-03 batch `exp03-20261106b`, Exp-04 batch
`exp04-20261105d` (official runs `s1`–`s6`), exp01-select `exp01-20261104`, exp01-eval `exp01-eval-20261104`.

| Paper item | Script | DataPort file → field |
|---|---|---|
| Fig. 1 (architecture) | drawn with a diagramming tool, no script | — |
| Fig. 3 (typical cycle: cell PRB, reservation times, edge-UE p95 RTT) | `r1/paper/fig3_cycle_timeline.py` | `exp03/exp03-20261106b/exp03_cycles.csv`, run dirs `raw/gnb_aura.log`, `raw/ue3_ping.log`, `run_meta.json`; medians → `r1/paper/cycle_timeline.json` |
| Fig. 4 (per-cycle outcome box plots) | `r1/paper/fig4_edge_ue_outcome.py` | `exp03/exp03-20261106b/exp03_cycles.csv` (`eval_ue3_rtt_p95_ms`, `eval_center_goodput_mbps`) |
| Edge-UE latency reduction, p95 medians/IQR, Mann-Whitney p, cycles per arm, center-UE cost, UE3 goodput | `experiments/exp03_analyze.py` | `exp03/exp03-20261106b/exp03_summary.json` (`rtt_p95_reduction_pct`, per-arm `eval`, `mann_whitney_p`, `center_cost_pct`) |
| End-to-end time intent → running xApp (20-/10-min), training, render+gate, registration | `scripts/timing_report.py` | `exp04/exp04-20261105d-timing_summary.json` → `groups["20"\|"10"].*_s.median` |
| End-to-end segment breakdown | `scripts/e2e_breakdown.py` | `exp04/e2e_breakdown.json` (inputs: run `timing.jsonl`, `verify_events.jsonl`, `run_meta.json`, collector `meta.json`) |
| Human touchpoints | `experiments/exp04_e2e.py` | `exp04/exp04-20261105d-s*/run_meta.json` → `human_touchpoints` |
| Classifier: accuracy, precision, recall, F1, FPR, onsets, onset recall, lead time (+IQR), test positive rate; FLAML and persistence | `experiments/exp01_eval.py`, `dataset/evaluate.py` | `exp01-eval/exp01-eval-20261104/exp01_eval_summary.json` → `headline["h2500/<model>"]` |
| Confusion matrix [TN FP; FN TP] | `dataset/evaluate.py` | `exp01-eval/exp01-eval-20261104/pooled/h2500/neuralsmith_metrics.json` → `secondary.confusion` |
| Deployed-model scores | `experiments/exp01_eval_deployed.py` | `exp01-eval/exp01-eval-20261104-deployed/deployed_summary.json`; models `exp03/exp03-20261106-arms/{c,d}/` |
| Model parity (gate) | `zto/parity.py` | `exp04/exp04-20261105d-s1-20m/gate.json`, `exp03/exp03-20261106-arms/manifest.json` |
| Training set size / positive rate, horizon samples | `dataset/build.py` | `exp01/exp01-20261104/ts500_h2500/split.json`, `meta.json` |
| Model family, ONNX latency p50, size, latency runs | `mlengine/*` | `exp01/exp01-20261104/ts500_h2500/neuralsmith/report.json`, `flaml/report.json` (FLAML `time_budget_s` = NeuralSmith training time) |
| T_s, H, debounce, window (spec listing) | `zto/resolve.py`, `scripts/list_slots.py` | `r1/zto/defaults.json`, `r1/paper/listing_example.json`, `r1/paper/slots.txt` (code) |
| Spec consistency (full/sparse correct %, rendered configs) | `experiments/exp02_consistency.py` | `exp02/exp02-20261104-{a,b}/exp02_summary.json` (part a: 3 LLMs, part b: llama3.3:70b) |
| Orchestration latency per LLM (warm/cold), tokens, VRAM | `experiments/exp06_latency.py` | `exp06/exp06-20261104/exp06_summary.json` |
| Gate: corrupted specs caught, false HOLDs, deterministic ms, read-back s, vague intents matched | `experiments/exp08_gate.py`, `zto/gate_inject.py` | `exp08/exp08-20261104/exp08_summary.json`, `exp08_gate_results_summary.json` |
| Edge-UE MCS ~27 → ~14 (channel) | `scripts/calib_channel.py` | `calibration/calib-30_-10-20261102-191230/calibration.json`, `calibration/NOTES.md` |
| Ramp rise time | `scripts/calib_ramp.py`, `ramp_rise.py` | `calibration/ramp-20261103-115429/rise_robust.json` |
| "Queue starts ~4 s before congestion" | `scripts/oracle_check.py`, `backlog_onset.py` | `calibration/backlog_onset.json`, `calibration/oracle-*` |
| sim/wall speed range | collectors (`speed_rules.py`) | every run's `meta.json` → `sim_speed_mean` |
| Hardware, Ollama version, models | `scripts/hw_info.sh`, `scripts/ollama_env.py` | `r1/env/hardware.json`, `r1/env/versions.json` (code) |

## License

Our code: MIT (`LICENSE`). The OAI patches modify files licensed under the OAI license of the pinned commit (OAI CSSL
v1.0 at `09c9bb99`; earlier OAI releases used the OAI Public License v1.1); the FlexRIC patches and the xApp fall
under the OAI Public License v1.1 (FlexRIC `340c36bc`). Details: `THIRD-PARTY-NOTICES.md`.
