# Environment Log

## NEUROLIB_ENV_V1 — 2026-08-16

- Official environment: `neurolib`
- Python: `3.10.20`
- neurolib: `0.6.1`
- Core simulator smoke test: `PASS`
- Exact interpreter: `C:\Users\YUS190\AppData\Local\anaconda3\envs\neurolib\python.exe`
- No package installation or environment mutation was performed.
- `brainmass` and `jax` are not installed in this environment. Any later dependency change requires a new environment version, renewed smoke tests, and compatibility adjudication before outcome-bearing use.

Two outcome-blind infrastructure repairs were used for the probe: invoking the exact Conda launcher after `conda` was absent from `PATH`, and replacing slow optional-module imports with distribution-metadata queries. No scientific outcome was visible.

## NEUROLIB_ENV_V2 — 2026-08-16

Independent compute audit failed V1 because the recorded command did not enforce the frozen thread variables and because total physical RAM was mislabeled as available RAM. V2 is an outcome-blind infrastructure-control repair:

- `PREFLIGHT/run_neurolib.ps1` sets and asserts `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, and `NUMEXPR_NUM_THREADS=4` for every official child process.
- The launcher accepts only current-campaign scientific scripts.
- The launcher applies memory admission thresholds for one, two, or three requested concurrent processes.
- Total and available RAM are separately labeled and timestamped.
- The neurolib smoke test passed with all four thread variables observed as `4`.
- No package or scientific-definition change occurred.

V2 did not pass independent re-audit: the frozen command was blocked by the host's effective Windows PowerShell script policy, and the launcher's raw prefix comparison did not separator-bound the campaign root or reject reparse points.

## NEUROLIB_ENV_V3 — 2026-08-16

V3 is the second and final bounded outcome-blind environment-control repair:

- The exact official command uses the absolute Windows PowerShell executable with process-scoped `-ExecutionPolicy Bypass`, `-NoProfile`, and `-NonInteractive`; machine and user policy are unchanged.
- Script containment now requires an existing `.py` leaf under a separator-bounded resolved campaign-root prefix.
- Every script-path component from the file through the campaign root is checked and rejected if it is a reparse point.
- The exact V3 probe command passed, the neurolib smoke output remained finite with shape `[1, 200]`, and all four thread variables were observed as `4`.
- No package, target, metric, threshold, intervention, mapping, comparator, or scientific definition changed.
- No further repair is permitted in this environment lineage if final independent re-audit fails.
