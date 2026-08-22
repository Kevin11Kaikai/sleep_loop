# C1 Executive Summary - Why Frozen C006 Is Controllable

## Question and decision

C1 asked whether the **same frozen C006** would respond causally, boundedly, stably, and reproducibly to a frozen control intervention. The terminal decision was:

- `C1_CONTROLLABILITY_MVP_PASS_T2_PRIMARY`
- `RELIABLE_AND_CONTROLLABLE_DIGITAL_TWIN_MVP`

The claim ceiling is a **bounded in-silico reliable and controllable mechanistic digital twin MVP**.

## Prospective definition of controllability

Before fresh confirmation, C1 froze:

- Primary target: `T2`, with desired direction `dT2 > 0`.
- Supporting corroboration: `I_SO`.
- Fresh seeds: `[101003, 101019, 101033]`.
- Durations: 220 s and 910 s.
- Matched-SHAM design, analysis window, pass gates, integrity vetoes, and mandatory secondary diagnostics.
- Frozen controller F01.

The primary gate at each duration required median dT2 > 0, at least 2/3 positive seeds, and controller integrity PASS for all seeds.

## Frozen causal and bounded controller

F01 is a causal baseline-calibrated deep-event gate using the frozen C006 causal proxy. It applies additive cortical `u_E` pulses with:

| Parameter | Frozen value |
|---|---:|
| Quantile / direction | 0.20 / rising |
| Depth gate baseline quantile | 0.50 |
| Pulse amplitude | 0.045 |
| Pulse width | 0.10 s |
| Refractory interval | 1.0 s |
| Legal actuator range | [0.0, 0.05] |

No retuning or controller rescue was permitted during confirmation.

## Fresh primary evidence

| Fresh seed | 220 s dT2 | 220 s I_SO | 910 s dT2 | 910 s I_SO | Integrity |
|---:|---:|---:|---:|---:|---|
| 101003 | +0.1113% | +0.002273 | +0.0571% | +0.001824 | PASS / PASS |
| 101019 | +0.9167% | +0.012100 | +0.6795% | +0.009018 | PASS / PASS |
| 101033 | +0.1913% | +0.003756 | +0.1784% | +0.003046 | PASS / PASS |

- **220 s:** dT2 positive 3/3; median **+0.1913%**; I_SO supportive 3/3.
- **910 s:** dT2 positive 3/3; median **+0.1784%**; I_SO supportive 3/3.
- All six runs were finite, causal, engaged, within legal `u_E` bounds, below 10% duty, hash-consistent, and integrity/safety PASS.

## Mandatory secondary diagnostics

| Diagnostic | 220 s median / positive seeds | 910 s median / positive seeds |
|---|---:|---:|
| `N_SO_MIN` | +0.292683 / 2 of 3 | +0.737430 / 3 of 3 |
| `I_SP` | -0.002294 / 0 of 3 | -0.001547 / 0 of 3 |
| `N_SP_MIN` | +0.878049 / 3 of 3 | +0.201117 / 3 of 3 |
| `P_SO` | +0.008049 / 3 of 3 | +0.019117 / 3 of 3 |
| `T6` | -0.000941 / 1 of 3 | -0.001398 / 1 of 3 |
| `P_C_GIVEN_SP` | +0.048837 / 2 of 3 | +0.013260 / 2 of 3 |

The main limitations are adverse `I_SP` at both durations and mixed/slightly negative `T6`. These do not trigger a frozen veto, but they constrain the interpretation.

## Dissertation-ready interpretation

> The same frozen C006 showed a positive primary T2 response across three fresh seeds and retained that direction at 910 s under a causal, legal, bounded, and integrity-passing frozen controller. I_SO corroborated the response, while spindle/coupling trade-offs remained.

This establishes **primary controllability with secondary dynamical trade-offs**, not simultaneous optimization of all sleep dynamics.
