# Frozen clean-builder specification

Campaign: `COSTA_MVP_FRESH_R4_20260816_023933`  
Scope: `M1/P0/INDEPENDENT_REPRODUCTION`, optional `C1` only if M1 passes.

## Immutable scientific inputs

Use only the seven exact files copied into `CODE/MECHANISTIC_V2`, `CODE/FROZEN_V2`, and `CODE/TESTS`, whose identities are locked by `IMPORT_ALLOWLIST/V4D1_IMPORT_ALLOWLIST_V2.json` (`F0CCA491…B1672A`). Do not inspect any R3 path. Preserve the model, 16-candidate bank, synthetic worlds, thresholds, seeds, partition seed, operator, and claim definitions byte-identically.

## Fresh implementation boundary

Write new code only in `CODE/R4_EXECUTION/`. Do not read original data during construction or tests.

- Use MNE 1.9 only through `mne.read_annotations(path)` with no `verbose` argument and `mne.io.read_raw_edf(path, preload=True, verbose="ERROR")`.
- Never parse an EDF header directly.
- Accept only the exact Night-1 paths `data/sleep-edfx-cassette/SC4001E0-PSG.edf` and `data/sleep-edfx-cassette/SC4001EC-Hypnogram.edf`.
- Treat upstream `Sleep stage 3` and `Sleep stage 4` annotations as N3, split into 30-second epochs, then use the imported SHA-256 partition function with seed `26081601`, FIT fraction `0.5`, and signal/outcome-blind epoch IDs.
- Channel rule: first exact match in `EEG Fpz-Cz`, then `EEG Pz-Oz`; no outcome-guided fallback.
- Role isolation: the steward writes separate FIT and HELDOUT payloads. The selection stage may open FIT only, freezes the selected candidate and calibration, and exits. A distinct heldout stage verifies that freeze before opening HELDOUT.
- Comparison operator: detrend each signal, Welch PSD with a 4-second Hann window and 50% overlap, total power 0.5–20 Hz, and `log10(relative power)` for fixed bands SO 0.5–1, delta 1–4, theta 4–8, spindle 11–16 Hz. Apply identically to EEG volts and the model proxy. Output units are dimensionless and proxy-level; never relabel model units as voltage or claim a scalp forward model.
- Model prediction uses each frozen candidate under the frozen generic synthetic N3 drive and the imported state-to-proxy operator. Fit one scalar affine slope/intercept per candidate using FIT features only. C000 is the matched nonpersonalized comparator with the identical calibration/scoring rule. Candidate selection minimizes FIT NRMSE with candidate-ID tie break.
- G6 heldout rule: at least four eligible N3 epochs and at least two valid epochs in each role; personalized heldout relative NRMSE improvement over C000 at least `0.05`; 95% paired epoch-bootstrap CI (1,000 draws, seed `26081604`) for population-minus-personalized MSE has lower bound strictly above zero. Missingness excludes only nonfinite, flat, too-short, or out-of-range epochs and may not change role assignment.
- Empirical wrong-subject specificity is not estimable with one subject. M1-G3 must use the frozen matched-nonpersonalized plus frozen synthetic wrong-subject gates.
- M1-G4 combines the frozen synthetic uncertainty gate with the frozen empirical paired epoch-bootstrap rule.
- M1-G5 remains a proxy-level mapping claim.
- M1-G7 requires a fresh verifier to reproduce hashes, exact split, numeric results, baseline, threshold decisions, and access/open-order fidelity with `MATERIAL_DISAGREEMENT = FALSE`.

## Required modes and artifacts

Implement small auditable modules/scripts supporting:

1. data-free adapter unit/preflight tests using injected fake MNE objects (including exact call signatures), fake annotations/raw, path guard, N3 construction, split, feature operator, role separation, serialization, and failure cases;
2. data steward mode producing partition metadata, separate FIT/HELDOUT role payloads, and a receipt;
3. synthetic M1 mode executing the immutable harness twice and reporting G1–G5 plus synthetic G6-interface status;
4. FIT-only selection mode producing an immutable candidate/calibration freeze;
5. HELDOUT-only mode producing G6 metrics/uncertainty without changing the candidate;
6. deterministic canonical JSON and SHA-256 manifests for independent reproduction.

Every production entrypoint must fail closed on input/hash/schema mismatch. Do not silently enter raw-data mode from preflight. All output paths must be new children of the supplied output root.

## Runtime contract

Exact launcher: `C:\Users\YUS190\AppData\Local\anaconda3\condabin\conda.bat run --no-capture-output -n neurolib python`.

Use campaign-local `TEMP`, `TMP`, and `NUMBA_CACHE_DIR` because the default Numba temporary-cache creation path was demonstrably pathological. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `MNE_DONTWRITE_HOME=true`. This is an outcome-blind environment repair; it may not alter scientific logic.

For neurolib Git identity, call read-only Git with `-c safe.directory=D:/Year3_Mao_Projects/neurolib`. Required commit: `9b6b2b8f082c0cfa212f05576ead55bf23046d6f`. Tracked diff must be clean. `environment.yml` is permitted only if proven absent from source, configuration, module, and open-file closures; do not read or use its contents.
