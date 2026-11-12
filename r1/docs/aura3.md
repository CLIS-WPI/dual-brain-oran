# AURA gNB (`oai-gnb:2026.w27-uhd410-e2-aura3`): what it changes and how the xApp controls it

Read-only investigation, 2026-11-02, from `oai-telco-stack/oai-patches/` and
`flexric/examples/xApp/c/dual_brain/xapp_congestion_ctrl.c`. Use this file for the paper's
listing and grounding paragraph.

## Build

| Item | Value |
|---|---|
| Image | `localhost/harmony/oai-gnb:2026.w27-uhd410-e2-aura3` |
| Image ID | `sha256:535e30a6f392e7ca2cd1083918f71c53002bfcbda563e716101822e41546aef1` (local build, no registry digest), created 2026-10-05T18:36:11-04:00 |
| OAI base | `https://gitlab.eurecom.fr/oai/openairinterface5g.git` @ `09c9bb998fa2f30f2df452a0a9002787280389af` (2026.w27) |
| FlexRIC submodule | `ef6d722f22191eea74089966983da1f5ec1fedd4` |
| Patches | `0001`–`0004` (below), applied by `oai-patches/prepare-oai-src.sh`; built by `scripts/build-rfsim-e2.sh` |

Recorded in `env/versions.json` → `gnb`.

## Layers touched: MAC scheduler + E2 agent only, no PHY

Every patched file is under `openair2/` (L2 / E2). Nothing under `openair1/` (PHY), nothing in
the RF simulator.

| Patch | Files | Layer | What it does |
|---|---|---|---|
| 0001 nr-mac-aura-per-slice-dl-prb-quota | `openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_dlsch.c`, `aura_quota.{c,h}`, `CMakeLists.txt` | NR MAC DL scheduler | Per-slice DL PRB quota: before allocation, each sliced UE's `rbSize` is capped to its slice's remaining PRBs in that slot. Unsliced UEs are untouched. With no slices the scheduler is the stock one. |
| 0002 e2-slice-sm-programs-aura-quota | `openair2/E2AP/RAN_FUNCTION/CUSTOMIZED/ran_func_slice.c`, `aura_quota.{c,h}`, `gNB_scheduler_dlsch.c` | E2 agent (SLICE RAN function) → MAC | Programs the 0001 quota tables from FlexRIC **SLICE SM** control messages (below). Stock OAI only printed these messages. |
| 0003 flexric-agent-level-triggered-e2-socket | `openair2/E2AP/flexric/src/agent/{asio_agent.c,asio_agent.h,e2_agent.c}` | E2 agent I/O | Reads the E2 SCTP socket level-triggered (fixes stalled indications). No RAN behaviour change. |
| 0004 aura-protected-first-free-trimmed-prbs-prb-ground-truth | `aura_quota.{c,h}`, `gNB_scheduler_dlsch.c` | NR MAC DL scheduler | Unsliced (protected) UEs are allocated **first**, capped slices after them; PRBs trimmed by a cap are given back in `vrb_map`. Adds the `[AURA-PRB] 1s` ground-truth log line. |

## The control used for PRB reservation: FlexRIC SLICE SM — not E2SM-RC

**Important for the paper:** the PRB reservation is **not** an E2SM-RC control. The xApp uses
FlexRIC's **SLICE service model** (`SM_SLICE_ID = 145`, `src/sm/slice_sm/slice_sm_id.h`), served
by OAI's `RAN_FUNCTION/CUSTOMIZED/ran_func_slice.c`. There is no RC style, RC control action ID or
RC RAN parameter ID in this path. Monitoring uses E2SM-KPM (`RRU.PrbTotDl`, per UE, report period
`kpm_period_ms`, E2SM-KPM action definition format 1). The brief's `zto/catalog/rc_actions.yaml` and
the manuscript's "E2SM-RC style / action id" need to say **SLICE SM** with the fields below, or the
paper must describe the control as SLICE SM.

Control messages the xApp sends (`control_sm_xapp_api(&node_id, SM_SLICE_ID, req)`):

| Step | SLICE SM message | Fields used | gNB semantics (patch 0002) |
|---|---|---|---|
| reserve | `SLICE_CTRL_SM_V0_ADD`, DL, 1 slice | `id = best_effort_slice_id` (default 1), `label = "best-effort"`, `params.type = SLICE_ALG_SM_V0_STATIC`, `params.u.sta.pos_low = 0`, `pos_high = (100 − reserve_pct) − 1` | Only STATIC slices are accepted. `pos_low..pos_high` is read as the slice's share of the **DL BWP in percent**: quota = `pos_high − pos_low + 1` % → PRBs = `pct × BWPSize / 100` (51 PRB on RFsim, 106/273 on N310). |
| reserve | `SLICE_CTRL_SM_V0_UE_SLICE_ASSOC` | `ues[].rnti = <congestor's E2SM-KPM RAN UE ID>`, `dl_id = ul_id = best_effort_slice_id` | `rnti` may carry an RNTI **or** the E2SM-KPM RAN UE ID (= gNB-CU UE ID); the gNB resolves it to the current RNTI and re-resolves every 200 slots (re-attach keeps the slice). |
| release | `SLICE_CTRL_SM_V0_DEL` | DL slice id `best_effort_slice_id` | Removes the slice. With no E2 slice left the scheduler falls back to the quota file, and with no file to the stock scheduler. |

Value semantics: "reserve 20 % for the edge UE" is a **floor**, implemented as a **cap on the
congestor**. The UE with the highest recent PRB share is put in one slice of
`100 − reserve_pct` = 80 % of the DL BWP. Every other UE stays unassociated (uncapped and allocated
first, 0004), so at least `reserve_pct` % of the PRBs remain available to them. Only **DL** is
controlled: AURA caps the DL scheduler, so UL reservations are not supported.

## `[AURA-PRB] 1s` (patch 0004)

Printed by the gNB to stderr once per **wall-clock** second (`CLOCK_MONOTONIC` second boundary):

```
[AURA-PRB] 1s budget=<B> rnti=0x<RNTI>:slice=<S>:tx_rb=<T>:tbs=<TBS> …   (one entry per UE seen in the interval)
```

| Field | Meaning |
|---|---|
| `budget` | Sum of DL BWP PRBs over the DL slots of that wall second. This is the denominator of KPM `RRU.PrbTotDl`. |
| `rnti … slice` | UE (RNTI) and its AURA slice id (−1 = unsliced/protected) |
| `tx_rb` | PRBs actually transmitted to that UE (sum of `rbSize` at MAC PDU generation, new transmissions) |
| `tbs` | Transport block bytes for that UE in the interval |

Derived quantities used in R1:
- cell DL PRB % = `Σ tx_rb / budget × 100` (clock-independent)
- UE PRB share = `tx_rb / budget`
- **sim/wall speed ratio** = `budget / real-time budget`, where the real-time budget is 51 PRB × (2 DL + 1 mixed slot per 2.5 ms TDD period) × 400 periods/s = 61,200 PRB/s at 51 PRB. RFsim is CPU-bound, so this ratio moves with host load (see `r1/README.md`).

The companion `[AURA-QUOTA] 1s rnti=… slice=… asked=… granted=… tx=…` line is printed only while a
quota is active (per mapped UE: PRBs the scheduler asked for, PRBs granted after the cap, PRBs transmitted).
