# Known Limitations

1. **Model identity:** C006 is a frozen discrete mechanistic candidate selected from a finite bank. It is not continuous parameter recovery, a posterior estimate or biological parameter truth.
2. **Observation:** model output is a `mau` proxy, not scalp-voltage EEG and not a validated clinical endpoint.
3. **Actuator:** `u_ctrl → cortical u_E` is a dimensionless model input. Its relationship to auditory, electrical or clinical intervention intensity is not established.
4. **Inputs:** controller MVP simulations use permitted synthetic/development inputs, not a fresh clinical intervention dataset.
5. **Sample size:** the final comparison uses three evaluation seeds; it is suitable for MVP screening, not statistical certification.
6. **Target validity:** T2 is a model-space slow-oscillation amplitude metric. A 10% model improvement has not been validated as a meaningful sleep-quality improvement.
7. **Controller family:** PLL and Phase Vocoder pulse controllers were evaluated; MPC, RL and broader controller searches were intentionally excluded.
8. **PID comparison:** earlier PID gains came from a different V1/neurolib plant. The C006 coarse PID attempt yielded `NO_VALID_R4_PID_BASELINE`.
9. **Seed heterogeneity:** development-phase response changed direction across seeds, reducing confidence in phase generality.
10. **T8:** `T8_MVP_cortical_state_crossing_rate` is `SAFETY_PROXY_ONLY — NOT a validated dynamical-regime-transition metric`.
11. **Failed anchor:** the SC4001/V1 PSD anchor is a failed auxiliary route and does not negate the frozen R4/M1 mechanistic result.
12. **Evidence ceiling:** nothing in this packet supports real-patient benefit, clinical efficacy, formal C1/C2 or dissertation-final claims.
