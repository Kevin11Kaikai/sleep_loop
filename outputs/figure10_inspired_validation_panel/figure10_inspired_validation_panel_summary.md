# Figure-10-inspired Validation Panel Summary

This figure is inspired by SBI Figure 10, but no posterior estimator is available.
Candidate rows are a fitted-candidate ensemble / candidate archive, not posterior samples.

## Files Used

- `data\patient_params_fig7_v1_0418_2_SC4001.json`
- `data\patient_params_fig7_SC4001.json`
- `data\patient_params_fig7_v2_SC4001.json`
- `data\patient_params_fig7_v3_SC4001.json`
- `data\patient_params_fig7_v4_SC4001.json`
- `data\patient_params_fig7_v5_SC4001.json`
- `data\patient_params_fig7_v6_SC4001.json`
- `data\patient_params_fig7_v7_SC4001.json`
- `data\patient_params_fig7_v8_SC4001.json`
- `outputs\evolution_fig7_v3_records.csv`
- `outputs\evolution_fig7_v4_records.csv`
- `outputs\evolution_fig7_v5_records.csv`
- `outputs\evolution_fig7_v6_records.csv`
- `outputs\evolution_fig7_v7_records.csv`
- `outputs\evolution_fig7_v8_records.csv`
- `outputs\v8a_relaxed_t6_sensitivity\relaxed_t6_best_candidates.csv`
- `outputs\v8a_t6_t13_coupling_sweep.csv`
- `outputs\v8a_ultra_narrow_t6_t13_search\ultra_narrow_records.csv`
- `outputs\v8a_local_de_t6_rescue_long\local_de_records.csv`
- `outputs\v8a_local_de_t6_rescue_narrow\narrow_local_de_records.csv`

## Missing / Skipped Files

- `data\patient_params_fig7_v1_SC4001.json`

## Run Summary

- candidates collected after dedup/truncation: `50`
- versions represented: `V1, V2, V3, V4, V5, V6, V7, V8a, V8a-coupling, V8a-relaxed, V8a-ultra`
- simulations completed: `150`
- wall time seconds: `32.5`
- real observation summaries computed: `True`
- PNG: `outputs\figure10_inspired_validation_panel\figure10_inspired_validation_panel.png`
- PDF: `outputs\figure10_inspired_validation_panel\figure10_inspired_validation_panel.pdf`

## Wording Guardrails

- PPC-like predictive diagnostics, not calibrated posterior predictive checks.
- Synthetic parameter-recovery diagnostic, not SBC.
- Expected coverage / SBC / TARP / L-C2ST are not performed here.
- Future SBI diagnostics require q_phi(theta|x).

## Synthetic Recovery

| synthetic_id   | theta_true_version   | theta_recovered_version   |   same_version |   summary_distance |   mean_relative_error |
|:---------------|:---------------------|:--------------------------|---------------:|-------------------:|----------------------:|
| synth_00       | V1                   | V3                        |              0 |           0.923261 |             0.953798  |
| synth_01       | V7                   | V7                        |              1 |           0.294111 |             0.0681841 |
| synth_02       | V8a                  | V8a-coupling              |              0 |           0.377507 |             0.236962  |
| synth_03       | V8a-relaxed          | V8a-coupling              |              0 |           0.486862 |             0.0681829 |
| synth_04       | V3                   | V3                        |              1 |           0.549417 |             0         |