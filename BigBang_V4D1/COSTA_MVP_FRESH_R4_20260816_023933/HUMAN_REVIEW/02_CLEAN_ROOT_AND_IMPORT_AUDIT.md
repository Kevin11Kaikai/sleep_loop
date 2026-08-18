# Clean root and import audit

`CLEAN_ROOT = PASS`; `ROLE_ISOLATION = PASS`; R4 protected access = none.

The historical outer execution bundle was rejected. Its runner would traverse a non-allowlisted repair artifact named by the inner freeze manifest; neither the historical runner nor either historical freeze artifact entered R4 execution.

Only seven exact, independently reviewed files were selectively allowlisted under default deny: `__init__.py`, `model.py`, `harness.py`, `partitions.py`, `protocol_v2.json`, `claims_v2.json`, and `test_synthetic_v2.py`. The exact allowlist hash is `F0CCA4913FEB8167561665547F90D58B655CB96328ADFCC05D062EAB88B1672A`. File-level review found no empirical, protected-derived, sealed, Night-2, or outcome-dependent content in that subset.

The initial denylist paired the intended outer-manifest path with the inner-manifest hash. The outcome-blind import audit corrected the outer hash, rejected the bundle, and froze the seven-file allowlist before builder, raw, or outcome execution. This was classified nonmaterial with no scientific effect.
