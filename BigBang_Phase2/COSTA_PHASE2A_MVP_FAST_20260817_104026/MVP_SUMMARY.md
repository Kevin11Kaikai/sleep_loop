# COSTA Phase 2A Fast Exploratory MVP

**Evidence ceiling:** EXPLORATORY_ONLY. These are model-internal, preliminary screening results. They do not support C1, C2, clinical, confirmatory, dissertation-final, efficacy, safety, or prescribing claims.

## Execution

- Candidate: accessible frozen development candidate `V1`; no claim that it is the prohibited stopped-lineage M1 judgment.
- Model: neurolib thalamocortical model, native cortical `ext_exc_current` control channel.
- Grid: sham plus 8 prospectively fixed active doses.
- Seeds: [11003, 22007, 33013].
- Planned/completed simulations: 27 / 27.
- Analysis: 5 s burn-in plus 30 s exploratory window per simulation.
- Numerical stability: 27 / 27 passed finite/range/non-flat checks.
- Wall time: 177.0 s.

## First-pass response screen

Responses in `mvp_phase2a_quicklook.csv` are relative changes from the three-seed sham mean, except T7, which is a wrapped phase shift in radians.

- T1 — Oscillation frequency / entrainment: largest coarse response +0.8 at dose -0.0500 mV/ms
- T2 — Slow-oscillation amplitude: largest coarse response -0.6844 at dose -0.0500 mV/ms
- T3 — Slow-oscillation density: largest coarse response +1.167 at dose -0.0200 mV/ms
- T4 — Spindle density: largest coarse response -0.6667 at dose -0.0500 mV/ms
- T5 — Spindle amplitude / power: largest coarse response -0.992 at dose -0.0500 mV/ms
- T6 — SO-spindle coupling strength: largest coarse response -0.7103 at dose -0.0350 mV/ms
- T7 — Preferred phase / timing: largest coarse response -0.3314 at dose -0.0200 mV/ms
- T8 — Dynamical regime / state transition: largest coarse response +2.951 at dose -0.0350 mV/ms

## Interpretation limits

The grid is intentionally coarse, the analysis windows are short, and no protected, Night-2, Sealed Bank, fresh-final, or confirmatory payload was used. Phase 2B was not executed. Any apparent dose-response pattern is hypothesis-generating only and requires independent development-only replication before stronger interpretation.
