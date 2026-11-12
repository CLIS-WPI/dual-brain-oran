# Spec field audit (2026-11-03): who consumes each field of spec.schema.json

"Expressive extraction, restrictive actuation": the schema can say what the operator asked; `actuatable.json`
(gate check 0) limits what can be deployed. Template slots = fields the xApp reads **and** that have more than one
actuatable value (`scripts/list_slots.py` → `paper/slots.txt`: **6**).

| Field | Consumer | xApp config key | Actuatable values | Template slot |
|---|---|---|---|---|
| sampling_ms | collector (KPM period), dataset builder (--ts), xApp | kpm_period_ms | [100..1000] | **yes** |
| window_ms | dataset builder (--window), xApp (feature window) | window_samples | [1000..10000] | **yes** |
| horizon_ms | dataset builder (label at t+H); the trained model encodes it | — (xApp never reads it) | [0..10000] | no |
| collection_min | collector (duration) | — | [5..60] | no |
| label_rule.feature | dataset builder (column), xApp (KPM metric) | metric (constant RRU.PrbTotDl) | cell_prb_util only | no (single value) |
| label_rule.op | dataset builder and xApp hard-code ">=" | — (never read) | ">=" only | no (check 0) |
| label_rule.threshold_pct | dataset builder (label), xApp threshold policy | threshold_cell_prb | [70..90] | **yes** |
| label_rule.min_duration_s | dataset builder (sustained label), xApp threshold policy | min_duration_samples | [1..10] | **yes** |
| model.latency_budget_ms | trainer (mlengine budget loop) | — | [1..10] | no |
| model.max_size_kb | trainer (mlengine budget loop) | — | [16..4096] | no |
| slice_action.action | xApp mode control (the only implemented action) | — (implied) | reserve_prb_for_protected_ue | no (check 0) |
| slice_action.reserve_pct | xApp | reserve_pct | [5..50] | **yes** |
| slice_action.target | none in code: the xApp protects protected_ue_ref (runtime, UE3) | — (never read) | protected_ue only; no default | no (check 0) |
| control.debounce_n | xApp ML policy | trigger_consecutive | [1..20], default ceil(1000/sampling_ms) | **yes** |

Removed from the template (consumed by no xApp code): horizon_ms, threshold_op, latency_budget_ms, action, target.
Removed from the spec and schema (no consumer at all): task, kpm_metrics.

Field classes (field_classes.json): **operator-level** (what an intent can state; the only fields the gate's judge
extracts) = target, op, threshold_pct, min_duration_s, reserve_pct, horizon_ms. **engineering-level** (set by the
system; gate check 'engineering default' requires the defaults.json value) = sampling_ms, window_ms, debounce_n,
collection_min, model.latency_budget_ms, model.max_size_kb.

## Hidden xApp behaviour the paper should state

- **Reservation needs ≥ 2 active UEs.** `pick_congestor()` (xapp_congestion_ctrl.c) refuses to reserve unless at least
  two UEs are *active*: listed in a KPM indication (RRU.PrbTotDl per UE) within the last 2 s. The gNB lists a UE only
  when it had DL activity in the period, so idle UEs do not count. With fewer than 2, the xApp logs
  `control/skip "need >= 2 active UEs"` once per hot streak and does not act. In the R1 profile all three UEs carry DL
  traffic during ramp-up and hold (UE1/UE2 5–20 Mbps, UE3 5 Mbps constant), so the condition holds whenever congestion
  can occur; during OFF only UE3 is active (no congestion to act on). Verified on the Exp-01 KPM samples (see
  results/NOTES.md).
- **Re-association while reserved.** If the set of active UEs changes while a reservation is held (a UE goes idle for
  > 2 s, e.g. at the start of OFF), the xApp re-sends UE_SLICE_ASSOC for the new set (pending = 3).
- **Release** needs `release_after_normal_samples` calm samples (cell PRB < `release_below_cell_prb` = 60 %) and at
  least `min_hold_samples` since the reservation (both runtime values, not spec fields). Since 2026-11-04 both are defined in seconds: ceil(12.5 s / T_s) each
  (zto/render.py release_samples; 25 at T_s = 500 ms, 50 at 250 ms), identical for arms (b), (c), (d).
- **Release condition (when the cap is lifted).** While reserved, a KPM sample is *calm* only if the policy is not hot
  AND cell PRB < `release_below_cell_prb` (60 %); the xApp releases (SLICE DEL) after `release_after_normal_samples`
  (ceil(12.5 s / T_s) = 25 at 500 ms) consecutive calm samples AND at least `min_hold_samples` (same count) since the reservation (xapp_congestion_ctrl.c
  `calm`, `normal_streak`). That is >= 12.5 s below 60 % at any T_s (with the old fixed 50 samples it would have been 25 s at 500 ms, too long for a ~30 s OFF). During a hold the cell runs at 76–88 % even
  under the cap (calm impossible), so the cap cannot be lifted mid-hold; it is released ~12.5 s after the cell drops
  below 60 % in OFF. All three are runtime constants in the template, not spec fields. Exp-03 logs every reserve/release
  event with its time relative to the traffic schedule, which verifies this on data.

## Gate check "model_parity" (deterministic, implemented 2026-11-04)
After training and before registration (Exp-04), zto/parity.py runs a fixed set of recorded windows
(zto/parity/series_ts500.csv: exp01-select test cycles 10-13) through the xApp's own feature and model path
(cc_replay in the RIC pod, dualbrain9) and compares with r1/dataset/build.py features and the offline ONNX
prediction. FAIL unless every model input is computed by the xApp, features agree within 1e-4, the predicted labels
are identical, and the labels are not constant. Added after the Exp-04 dry run (exp04-dryrun-20261104): the builder's
features had names/units the dualbrain8 xApp did not compute, so the deployed model saw only imputed constants.
Since dualbrain10 the check also replays the recorded series through the xApp's debounce + lease logic: every cycle
must reserve and lift the cap in OFF. Dry run and rollback remain planned (gate output: STUB). dualbrain9 computes mean_prb, std_prb, min_prb, max_prb,
slope_prb (% per second from kpm_period_ms), last_prb, and keeps prb_mean/prb_std/prb_min/prb_slope (per sample).

## Release = lease (dualbrain10, 2026-11-04; supersedes the calm-streak release above for R1)
config.json carries lease_s (runtime value from zto/lease.py: p90 sustained-episode duration in the training rows + H,
clipped [10, 120] s); release_after_normal_samples / min_hold_samples / release_below_cell_prb are no longer rendered.
The lease is a minimum hold: the xApp keeps the cap for lease_s after reserving; after that it keeps it only while the
policy is hot and lifts it at the first non-hot sample (amended 2026-11-04: full-lease renewal held the cap through the
next ON in 2 of 4 replayed cycles). Reason: under the
cap, cell PRB and the model cannot see the aftermath of congestion (results/NOTES.md finding).

## Re-arm after release (dualbrain11, 2026-11-05; Milad, option 1)
After a release the trigger is disarmed; it re-arms only once cell PRB has stayed below the label threshold
(label_rule.threshold_pct, 80 %) for label_rule.min_duration_s (in samples: ceil(min_duration_s·1000/T_s) = 6 at
500 ms; config.json threshold_cell_prb / min_duration_samples, already rendered). The debounce is unchanged. Same rule
for every policy (ml and threshold, arms b/c/d), so only the trigger differs between arms. Applies only with a lease
(lease_s set); the legacy calm-streak path is unchanged. Reason: Exp-04 dry run 6, the capped UEs' backlog drain after
each release re-triggered the cap 2.5 s later (cc_policy.h cc_rearm_step; xApp event action "rearm").
Gate check model_parity: the policy replay (cc_replay --policy … <threshold> <min_duration_samples>) applies the same
rule; the binary's armed state must equal the Python reference (zto/parity.py rearm_reference) on every sample, and
every cycle must still reserve and lift the cap in OFF. The replay cannot see actuation effects on the network.
