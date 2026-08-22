# M1 Key Evidence

## Minimal evidence table

| Question | Authoritative answer | Source |
|---|---|---|
| What candidates were compared? | 16 frozen mechanistic candidates, C000-C015 | `CODE/FROZEN_V2/protocol_v2.json` |
| What was the selection rule? | Minimum FIT NRMSE after FIT-only affine calibration | `CODE/FROZEN_V2/protocol_v2.json` |
| Which candidate won? | C006, FIT NRMSE 0.2546119 | `RUNS/FIT_WORKER/SELECTION_FREEZE.json` |
| What is C006? | gain 0.82; `g_CT` 0.95; `g_TC` 1.10; `w_TR` 1.25 | `CODE/FROZEN_V2/protocol_v2.json` |
| Was it frozen before held-out access? | Yes; pre-heldout checkpoint and selection hash recorded | `STATE/CANDIDATE_FREEZE_CHECKPOINT.json` |
| Did held-out evidence support it? | C006 NRMSE 0.2478453 vs C000 0.4637754; 46.5592% improvement | `RUNS/HELDOUT_WORKER/HELDOUT_G6_RESULT.json` |
| Was uncertainty compatible with a positive advantage? | 95% paired bootstrap CI [0.0798002, 0.1014646] | `RUNS/HELDOUT_WORKER/HELDOUT_G6_RESULT.json` |
| Was the chain reproduced? | G1-G7 pass; G7 independent reproduction; no material disagreement | `TOURNAMENT/FINAL_M1_JUDGMENT.json` |

All paths above are relative to:

`D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933`

## Synthesis figure

`figures/M1_WHY_C006_SYNTHESIS_FROM_EXISTING_M1_EVIDENCE.png` is a rendered synthesis from slide 2 of the integration deck. It uses only the candidate FIT NRMSE values from `SELECTION_FREEZE.json` and the held-out comparison from `HELDOUT_G6_RESULT.json`. It is labeled `SYNTHESIS_FROM_EXISTING_M1_EVIDENCE` and is not new scientific evidence.

## The freeze boundary used by C1

All six C1 result JSON files record the same M1 protocol, model, and selection hashes. This exact identity match is the evidence that C1 controlled the already selected C006, rather than a refit or substituted plant.
