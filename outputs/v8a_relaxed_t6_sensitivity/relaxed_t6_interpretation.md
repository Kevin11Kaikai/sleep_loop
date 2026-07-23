# V8a Relaxed-T6 Sensitivity Analysis

This is a read-only post-hoc sensitivity check. No simulations, DE, V8b, threshold changes, or T13 detector changes were run.

Strict V8a uses `T6_ibi_cv < 0.40`. The relaxed thresholds tested here are post-hoc sensitivity checks only:

- `T6_ibi_cv < 0.40`
- `T6_ibi_cv < 0.42`
- `T6_ibi_cv < 0.45`
- `T6_ibi_cv < 0.50`

A relaxed-feasible candidate must still pass T4, T13, and every available T constraint except T6. Only T6 is relaxed.

## Input Files Used

- `outputs\evolution_fig7_v8_records.csv`
- `outputs\v8a_t6_t13_coupling_sweep.csv`
- `outputs\v8a_ultra_narrow_t6_t13_search\ultra_narrow_records.csv`
- `outputs\v8a_local_de_t6_rescue_long\local_de_records.csv`
- `outputs\v8a_local_de_t6_rescue_narrow\narrow_local_de_records.csv`
- `outputs\v8a_coupling_sweep_long\phase1_cth2ctx_sweep.csv`
- `outputs\v8a_coupling_sweep_long\phase2_coupling_2d_sweep.csv`

## Missing Input Files

- none

## Summary

- Threshold `0.40`: relaxed 13/13 count = `0`, selected status = `closest_near_miss`, selected T6_ibi_cv = `0.417`, margin = `-0.017`.
- Threshold `0.42`: relaxed 13/13 count = `1`, selected status = `relaxed_feasible`, selected T6_ibi_cv = `0.417`, margin = `0.003`.
- Threshold `0.45`: relaxed 13/13 count = `1`, selected status = `relaxed_feasible`, selected T6_ibi_cv = `0.417`, margin = `0.033`.
- Threshold `0.50`: relaxed 13/13 count = `9`, selected status = `relaxed_feasible`, selected T6_ibi_cv = `0.417`, margin = `0.083`.

## Interpretation

If a threshold such as `0.42` produces a relaxed-feasible candidate, it should be described as a relaxed-T6 sensitivity result, not strict V8a success.

This analysis does not claim that the model is fully validated. It also does not claim that V8a succeeded under the original strict criteria.

The main V8a T6 threshold should not be changed based on this table alone. Any future threshold change would require explicit scientific justification and separate approval.
