# Dissertation Story: Reliable + Controllable Digital Twin MVP

## Part II question

Can a mechanistic digital twin be both reliable and controllable within a clearly bounded in-silico use case?

## M1 - Reliable: why C006?

M1 began with a frozen bank of sixteen bidirectionally coupled thalamocortical candidates and a pre-specified selection rule: select the candidate with minimum FIT NRMSE after FIT-only affine calibration. Across 110 FIT epochs, C006 achieved the lowest NRMSE (0.254612); its calibration was fixed at intercept -0.053605 and slope 0.875498. C006 combined cortical gain 0.82, cortex-to-thalamus coupling 0.95, thalamus-to-cortex coupling 1.10, and relay-reticular inhibition 1.25.

C006 was then frozen before HELDOUT access. Across 110 HELDOUT epochs, frozen C006 achieved NRMSE 0.247845 compared with 0.463775 for matched non-personalized C000. This corresponds to 46.5592% relative improvement, exceeding the frozen 5% margin. The mean paired epoch MSE advantage was 0.090347, and its 1,000-draw 95% bootstrap interval [0.079800, 0.101465] remained above zero. All mandatory M1 gates passed, and the blind verifier independently reproduced the split, payloads, candidate/calibration identities, metrics, thresholds, and worker artifacts without material disagreement.

Within the registered scope, these results establish C006 as the reliable mechanistic twin candidate: selected by a frozen comparison, supported by held-out evidence, independently reproduced, and fixed by explicit provenance hashes.

## Freeze boundary

The freeze boundary is scientifically essential. M1 protocol, model, and C006 selection identities were fixed before controller confirmation. Every fresh C1 result carries exactly the same M1 hashes: protocol `338366...E3C3`, mechanistic model `775A39...3869`, and selection freeze `3D8C3D...8BBF`. C1 therefore did not substitute, refit, or retune the plant to obtain control success; it tested the response of the already frozen C006.

## C1 - Controllable: can the frozen twin respond reproducibly?

C1 prospectively froze the primary target (T2), direction (`dT2 > 0`), fresh seeds `[101003, 101019, 101033]`, durations (220 s and 910 s), matched-SHAM design, pass gates, secondary diagnostics, and F01 controller before fresh confirmation data. F01 is causal and applies additive cortical `u_E` pulses with amplitude 0.045, width 0.10 s, and 1.0 s refractory time within the legal actuator range [0, 0.05].

The primary response replicated. At 220 s, dT2 was positive in 3/3 seeds with median +0.1913%. At 910 s, dT2 remained positive in 3/3 seeds with median +0.1784%. I_SO was supportive in 3/3 seeds at both durations. All six runs passed the finite-trajectory, causality, hash, actuator, duty, numerical-integrity, and safety gates. The response therefore survived both new seeds and a prospectively fixed longer duration.

## Final result

M1 answers **why C006 is the frozen reliable twin**. C1 answers **whether the same frozen C006 has a reproducible response to bounded causal intervention**. Together they support:

> **Reliable + Controllable Digital Twin MVP**, under a bounded in-silico, T2-primary claim ceiling.

## Limitations

The result is primary-target controllability, not universal controller quality. `I_SP` was adverse in every fresh run, and `T6` remained mixed with slightly negative medians. The underlying M1 evidence remains one-subject, one-Night-1, proxy-level mechanistic consistency rather than population or clinical validation. These spindle/coupling trade-offs and translational limits belong in future controller-optimization and validation work; they do not alter the completed bounded MVP decision.
