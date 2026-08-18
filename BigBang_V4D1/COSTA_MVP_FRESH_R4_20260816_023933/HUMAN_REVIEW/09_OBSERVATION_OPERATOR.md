# Observation operator

The fixed model path is: four latent population states -> weighted source current -> causal baseline subtraction -> model-amplitude-unit proxy -> detrend/Welch -> four-band dimensionless log-relative-power vector. Empirical EEG uses detrend/Welch and the same log-relative-band-power feature definition.

G5 passed because the operator, ordering, units, and limits were frozen and coherent. The proxy is an arbitrary model-space observable, not volts or microvolts, and the mapping is not a validated scalp EEG forward model. Agreement therefore supports proxy-feature consistency only, not source localization, scalp causality, or physiological-unit equivalence.
