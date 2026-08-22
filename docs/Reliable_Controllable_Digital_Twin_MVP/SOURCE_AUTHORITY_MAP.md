# Source Authority Map

## Fixed authoritative roots

| Stage | Authoritative root |
|---|---|
| M1 - Reliable | `D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933` |
| C1 - Controllable | `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034` |

Both roots were present and readable during integration. No alternative M1 or C1 campaign was searched for or evaluated.

## M1 authority routing

| Evidence role | Authoritative artifact |
|---|---|
| Human-readable closure | `D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\HUMAN_REVIEW\FINAL_REPORT.md` |
| Final decision | `D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\TOURNAMENT\FINAL_M1_JUDGMENT.json` |
| Candidate comparison, calibration, and selection | `D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\RUNS\FIT_WORKER\SELECTION_FREEZE.json` |
| Frozen candidate bank and mechanistic parameters | `D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\CODE\FROZEN_V2\protocol_v2.json` |
| Held-out credibility gate | `D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\RUNS\HELDOUT_WORKER\HELDOUT_G6_RESULT.json` |
| Pre-heldout freeze provenance | `D:\Year3_Mao_Projects\sleep_loop\BigBang_V4D1\COSTA_MVP_FRESH_R4_20260816_023933\STATE\CANDIDATE_FREEZE_CHECKPOINT.json` |

## C1 authority routing

| Evidence role | Authoritative artifact |
|---|---|
| Human-readable closure | `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\HUMAN_REVIEW\C1_HIERARCHICAL_PRIMARY_TARGET_FINAL_SUMMARY.md` |
| Final decision | `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\DECISIONS\FINAL_C1_DECISION.json` |
| Prospectively frozen primary-target protocol | `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\PROTOCOL\C1_PRIMARY_TARGET_V1_T2.json` |
| Six-run aggregate | `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\RUNS\C1_PRIMARY_TARGET_AGGREGATE.json` |
| Lightweight metrics and ledger | `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\RUNS\C1_PRIMARY_TARGET_METRICS.csv`; `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\RUNS\EXPERIMENT_LEDGER.csv` |
| Frozen controller | `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\STATE\FROZEN_CONTROLLER.json` |
| Result hashes and terminal checkpoint | `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\CHECKPOINTS\LATEST_CHECKPOINT.json` |
| Authoritative quicklook | `D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034\FIGURES\C1_HIERARCHICAL_PRIMARY_TARGET_QUICKLOOK.png` |

The six per-seed result paths and hashes are listed in [EVIDENCE_MANIFEST.md](EVIDENCE_MANIFEST.md).

## Freeze continuity between M1 and C1

Every fresh C1 result records these M1 identities:

| Frozen element | M1 SHA-256 | C1 six-run check |
|---|---|---|
| `protocol_v2.json` | `338366593A2A4C8626349D70AA38306C5931F0FF4D22FD96DB546DD41505E3C3` | identical in 6/6 |
| Mechanistic model | `775A3961093C386A1E7DF8F0D033963EAB684F2566AE206E5BB6DB4A6CD03869` | identical in 6/6 |
| C006 selection freeze | `3D8C3D035F074CED2D608132864996D4AF9920C52889A11FE7C8E7DDA5DD8BBF` | identical in 6/6 |

This identity chain is the integration boundary: C1 tested control response in the same frozen C006 established by M1.
