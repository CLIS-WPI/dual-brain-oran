# Exp-02 reference specs (review: Milad)

`I1.json`, `I2.json`, `I3.json` hold the operator sentence (`_intent_text`) and **only the fields the intent
states explicitly**. Every other field comes from `zto/defaults.json` — the same file the ZTO-Agent system prompt
shows — so references and prompt cannot drift. `zto/resolve.py <file>` prints the full, schema-validated
reference; `zto/resolve.py --write-listing` regenerates `paper/listing_example.json` from I1. Defaults are
provisional until Exp-01; Exp-02 runs after Exp-01.

| Intent | Explicit fields |
|---|---|
| I1 "Predict congestion and reserve 20% of PRBs for edge users." | reserve_pct 20 (= listing_example.json) |
| I2 "If the cell stays above 85% load for at least 5 seconds, guarantee the edge user 30% of the downlink, and see it coming about 5 seconds ahead." | threshold_pct 85, min_duration_s 5, op ">=", reserve_pct 30, target protected_ue, horizon_ms 5000 |
| I3 "Keep latency low for our cell-edge subscriber whenever the cell is about to get congested." | action reserve_prb_for_protected_ue, target protected_ue, op ">=" (all numbers = defaults) |

Note: with the current defaults I1 and I3 resolve to the same spec.

Exp-02 scoring per model × seed (no LLM judge): **schema-valid** (validates against spec.schema.json),
**reference-correct** (exact match, slot by slot: the 12 template slots of scripts/list_slots.py, and the
non-slot fields task, kpm_metrics, collection_min, model.max_size_kb reported separately), **identical** (same
spec across the 5 seeds of that model).
