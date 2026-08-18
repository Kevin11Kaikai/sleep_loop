# Executive Brief

## Current answer

Neither formal C1 nor the exploratory C1 proxy MVP has been met. The final controller decision is:

`NO_CONTROLLER_MET_MVP_GATE`

The result is scientifically interpretable rather than computationally inconclusive: the final PLL/PV experiment completed 24/24 stable simulations, and the strict negative control was valid in all three evaluation seeds.

## What passed

- R4/M1 C006 frozen-input hashes and streaming equivalence checks.
- PLL and Phase Vocoder data-free phase-estimator preflights.
- Phase 2A exploratory execution and its minimum closeout gate.
- 24/24 final controller simulations were numerically stable.
- Three of three yoked negative controls matched pulse count, amplitude samples, width by construction, nonzero samples and total energy exactly.
- Realized adaptive-versus-shifted phase separation was 138.5°–163.7° in all three seeds.

## What failed

- No controller achieved median T2 improvement ≥10%.
- No controller achieved median composite response `R ≥10%`.
- PV adaptive did not beat its exactly yoked phase-shifted control; median R difference was −0.90 percentage points.
- No valid R4 PID baseline was obtained.
- The earlier SC4001/V1 personalization-anchor route failed and never entered its control phase.

## Controller comparison

| Controller | Median R | Median energy | Interpretation |
|---|---:|---:|---|
| PLL phase-locked | +0.26% | 0.004312 | Best observed R, but far below the gate; not a winner |
| PV phase-shifted yoked | −0.14% | 0.003465 | Valid negative control |
| PV adaptive | −1.03% | 0.003465 | No phase-specific advantage |
| PV fixed-phase | −1.10% | 0.003773 | Failed T2 and R gates |
| Fixed continuous | −3.36% | 0.073500 | Failed and used the most energy |

PLL used about 94% less median energy than Fixed, but its effect magnitude is too small to support provisional controller selection.

## Decision required from human reviewers

Choose exactly one direction before authorizing further simulations:

1. Preserve C006, `u_E`, and T2; accept the negative result and stop.
2. Preserve C006 and `u_E`, but redefine the primary physiological target, most plausibly toward T6/coupling.
3. Preserve C006 and T2, but replace `u_E` with a mechanistically justified actuator.
4. Change both target and actuator under a new exploratory protocol.

The review should not approve additional PID/PV tuning without first resolving this scientific choice.
