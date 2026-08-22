# C1 Key Evidence

## Prospective freeze

| Frozen item | SHA-256 | Role |
|---|---|---|
| `C1_PRIMARY_TARGET_V1_T2.json` | `34729FE59A4B0E0AD811EE013F3A378A568F798913ADF9C582CD9E49712F3491` | Primary target, direction, seeds, durations, gates, limitations |
| `FROZEN_CONTROLLER.json` | `64B5C154BB02681650E5B730C5C9FD6F05E2D79EC911512E5400FF317AA71A79` | Frozen causal F01 architecture and legal actuator |

Both hashes are identical across all six fresh result files.

## Fresh per-seed evidence

| Seed | Duration | dT2 | I_SO | I_SP | dT6 | Duty fraction | Integrity |
|---:|---:|---:|---:|---:|---:|---:|---|
| 101003 | 220 s | +0.001112792 | +0.002272808 | -0.001518970 | -0.000940715 | 0.039024 | PASS |
| 101019 | 220 s | +0.009166627 | +0.012099545 | -0.003789832 | +0.028110145 | 0.033171 | PASS |
| 101033 | 220 s | +0.001912864 | +0.003756367 | -0.002293932 | -0.022641271 | 0.020488 | PASS |
| 101003 | 910 s | +0.000571455 | +0.001823686 | -0.001547398 | -0.001397511 | 0.038436 | PASS |
| 101019 | 910 s | +0.006794895 | +0.009018337 | -0.002889017 | +0.027090156 | 0.033631 | PASS |
| 101033 | 910 s | +0.001784399 | +0.003046250 | -0.000828056 | -0.009385300 | 0.019218 | PASS |

Each row was cross-checked against its authoritative per-seed `result.json`; each file SHA-256 matched `CHECKPOINTS/LATEST_CHECKPOINT.json`.

## Aggregate decision

| Gate | Result |
|---|---|
| 220 s primary T2 | PASS - median +0.1913%; positive 3/3 |
| 910 s primary T2 | PASS - median +0.1784%; positive 3/3 |
| I_SO at 220 s | SUPPORTIVE - positive 3/3 |
| I_SO at 910 s | SUPPORTIVE - positive 3/3 |
| Frozen C006 continuity | PASS - M1 protocol/model/selection hashes identical 6/6 |
| Frozen F01 integrity | PASS 6/6 |
| Terminal state | `RELIABLE_AND_CONTROLLABLE_DIGITAL_TWIN_MVP` |

All paths are relative to:

`D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_QSOLVER_C1_HIERARCHICAL_TARGET_20260820_204034`

## Figure

`figures/C1_HIERARCHICAL_PRIMARY_TARGET_QUICKLOOK.png` is copied byte-for-byte from the authoritative C1 `FIGURES` directory. SHA-256: `A54D369EA585D6E103116935BC8BCDCA112B8E8947A08D7858EB6CE52F60179B`.

## Interpretation guardrail

The fresh primary result is reproducible and duration-robust within the registered gate. `I_SP` is adverse 3/3 at both durations and `T6` is mixed/slightly negative. Therefore the result is **primary controllability demonstrated with secondary dynamical trade-offs**.
