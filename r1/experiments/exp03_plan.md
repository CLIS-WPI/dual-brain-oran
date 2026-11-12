# Exp-03 plan (four arms; (d) exploratory) — approved 2026-11-03, runs after the official Exp-04

Traffic: frozen profile (peak and UE3 demand from the oracle check), seed 6, 13 cycles per run, T_s = main T_s.
Arms (3 interleaved runs each, order a,b,c,d,a,b,c,d,a,b,c,d, isolation + fresh traffic generator before each run):
  (a) no xApp
  (b) reactive: xApp threshold (persistence) policy — acts at the sustained-threshold onset (min_duration_samples)
  (c) xApp ML policy, deployed Exp-04 artifact (T_s 500 ms, H 2.5 s, debounce ceil(1000/T_s) = 2)
  (d) EXPLORATORY: as (c) with the H = 7.5 s model trained on the same Exp-04 dataset (H chosen after the grid;
      never replaces (c))
xApp configuration and models (Milad, 2026-11-04):
  (c) = the full artifact of the FIRST official 20-min Exp-04 run: its gated config.json (rendered by the ZTO pipeline
      from intent I1) plus the model that run's pipeline trained, both byte-for-byte unchanged; the runner never
      assembles a config. Recorded: sha256 of config.json and model.onnx (+ preprocess.json) and the gate record.
  (d) = the same config.json with a model trained on the SAME Exp-04 dataset with the same NeuralSmith settings and
      horizon_ms = 7500 (EXPLORATORY); (c) and (d) differ only in H.
      "Same dataset" = the same Exp-04 telemetry and split, rebuilt by dataset/build.py with horizon_ms 7500 (labels
      depend on H); the Exp-04 runner keeps telemetry + train split for this.
  (b) = the same config.json with `policy` -> "threshold" only.
  Release: the lease rule in that config.json (lease_s from the Exp-04 training data, zto/lease.py) applies to
  (b), (c), (d) alike, so only the trigger differs.
  The runner asserts each derived file differs from (c) only as stated and records each sha256 and the diff.   exp03_summary.json also keeps the exp01-select models' sha256 (ts500_h2500, ts500_h7500) for reference.
Deployed-model scoring: the Exp-04 model (H 2.5 s) and its H 7.5 s sibling are scored offline on the two exp01-eval
runs with the same dataset/evaluate.py, labelled "deployed model", next to the headline (select-run) model.
Per-cycle window: second half of ramp-up + hold (from the schedule, same for every arm).
Measurement source (identical in all four arms): per-UE/cell PRB from the gNB [AURA-PRB] 1 s lines, goodput from
the iperf2 receiver reports, UE3 RTT from the raw 10 Hz ping log. KPM via the xApp exists only in arms (b), (c) and (d)
(it occupies the subscription there); it is logged for diagnosis, never used for an outcome metric. Arm (a) runs no
xApp and no other KPM subscriber.
Regime (2026-11-04): start gate = idle ±10 % of 1.70 + ~3 min loaded probe within 1.47-1.80 (retry every 15 min); each run's mean per-cycle
loaded speed must lie within ±10 % of the exp01 loaded mean 1.635 (speed_rules.regime_check), else the run is marked
off-regime, kept, and excluded from the paper numbers (never silently kept).
Queue state: per cycle, UE3 RTT mean/p95 over the last 5 s before ramp-up start (raw ping) is reported; separate runs
per arm with the isolation wait keep one arm's backlog out of another's cycles.
Per cycle: UE3 RTT mean and p95 (raw 10 Hz ping), UE3 goodput (iperf receiver), UE1+UE2 goodput, PRB shares,
reserve/release times vs the schedule (and false reservations outside ON phases).
Full-cycle metrics (added 2026-11-04, captures the post-release effect): per cycle and per arm, the same UE3 RTT
mean/p95, UE3 goodput and UE1+UE2 goodput over the whole cycle (ramp-up start to the next ramp-up start, OFF
included), plus OFF alone (OFF start + 3 s to the next ramp-up start). Reported next to the evaluation window,
same tests.
Operational cost (every arm): fraction of time a reservation is active while the cell is not congested
(sustained-80 % label false, gNB ground truth at 1 s: cell PRB >= 80 % for 3 consecutive s), over the whole run and
per cycle; plus the center-UE goodput cost (UE1+UE2 goodput vs arm (a), per cycle and pooled).
Summary per arm: median and IQR of per-cycle p95, pooled mean RTT, UE3 goodput, center-UE goodput cost.
Tests: Mann–Whitney U on per-cycle p95, (c) vs (a) and (c) vs (b); (d) vs (a) and (d) vs (b) reported as exploratory. Reference upper bound: the oracle check
(cap from ramp-up start), reported alongside, not a tested arm.
Outputs: exp03_timeseries.csv, exp03_cycles.csv (one row per cycle), exp03_summary.json, fig_outcome.pdf.
Runner (2026-11-05): exp03_prep.py (arm artifacts from the Exp-04 batch → arms/manifest.json; (d) model trained on the
same Exp-04 dataset with H 7500 + gate check model_parity, FAIL → (d) marked and skipped), exp03.py (run / batch:
order a,c,b × 3 then d × 3 (Milad 2026-11-05: (a) vs (c) most complete if time runs out), same start gate before every run, deadline → remaining runs "not started"; off_regime,
sim_speed_drift or pod_changed (e.g. UE3 LDPC crash) → kept, marked, repeated, max 2 repeats per slot; a config whose
protected_ue_ref differs from UE3's current CU-UE-ID is never edited: the run stops, protected_ue_mismatch),
exp03_analyze.py (complete in-regime runs only → exp03_cycles.csv, exp03_timeseries.csv, exp03_summary.json,
fig_outcome.pdf). The same seed-6 13-cycle schedule in every run.
Unofficial dry run before the first official run (Milad, 2026-11-05): `exp03.py dry` = (a), (b), (c) with 2 cycles each
(same gate), then exp03_analyze; PASS needs config byte-exact on the RIC, the cap reserving and lifting within OFF in
every (b)/(c) cycle, gate + regime recorded, no control events in (a), all outputs and metrics produced
(<batch>-dry/dry_check.json). Then commit, then the official batch. Deadline: Exp-03 (dry + official) 2026-11-05 23:30 (Milad, another user agreed); Exp-04 keeps 18:32.
Protected UE ref (Milad, 2026-11-06): UE3 re-attached after an LDPC crash with CU-UE-ID 1 (the gated arm (c) config has
7). exp03_prep.py --protected-ue-ref <live id> derives the Exp-03 configs from the gated file with ONLY
protected_ue_ref changed (asserted), recording the gated and derived sha256 and the diff (manifest
protected_ue_ref_change); exp03.py checks the ref against the live mapping before every run.
UE3 re-attach during Exp-03 (Milad, 2026-11-06): before every run the batch compares UE3's live CU-UE-ID with the arm
configs; if it changed, exp03.refresh_protected_ref re-derives (b)/(c)/(d) config.json with the prep procedure (ONLY
protected_ue_ref; model files, lease, policy byte-identical, asserted), records before/after sha256, the diff, the UE3
crash time/exit code/restart count in the manifest (protected_ue_ref_refreshes) and keeps the crash log (raw/, local).
The run during which the crash happened is kept (pod_changed) and repeated under the existing rule. A failed refresh
stops only that slot (protected_ue_mismatch) and is retried before the next slot.
Regime band for Exp-03 (Milad, 2026-11-06): arm (c) training speed (Exp-04 s1, 1.7905) ±5 % = 1.701-1.880 for the
gate's loaded probe and the run-level check (exp03.exp03_band); lift rule: OFF, or ramp-down at a non-hot sample after
the lease expired (exp03.lift_check); speed balance reported in exp03_analyze (speed_balance).
