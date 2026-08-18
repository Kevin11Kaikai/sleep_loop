# Human Review Agenda — 45 Minutes

## Participants

- Sleep neurophysiology reviewer
- C006/mechanistic-model reviewer
- Control-method reviewer
- Decision owner and note-taker

## Agenda

| Time | Topic | Required outcome |
|---:|---|---|
| 0–5 min | Evidence boundary and frozen inputs | Confirm `EXPLORATORY_ONLY` and that C006 is not biological truth |
| 5–12 min | Phase 2A T1–T8 screen | Identify which response directions are physiologically plausible |
| 12–18 min | Failed C1 anchor and invalidated routes | Confirm which results cannot support C1 |
| 18–28 min | Final PLL/PV controller comparison | Review T2, T5, T6, R, energy and seed heterogeneity |
| 28–33 min | Yoked negative-control validity | Confirm that phase, not pulse count/amplitude/energy, is the intended contrast |
| 33–40 min | Target and actuator suitability | Choose whether the main mismatch is target, actuator, both, or true non-controllability |
| 40–45 min | Final decision | Complete the decision record and authorize one next state |

## Allowed next states

- `STOP_ACCEPT_NEGATIVE_RESULT`
- `DESIGN_T6_TARGETED_EXPLORATORY_MVP`
- `DESIGN_ALTERNATIVE_ACTUATOR_EXPLORATORY_MVP`
- `DESIGN_NEW_TARGET_AND_ACTUATOR_PROTOCOL`
- `REQUEST_SPECIFIC_ANALYSIS_ONLY`

Do not leave the meeting with a generic instruction to “try more tuning.”
