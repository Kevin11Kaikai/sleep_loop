# Nonhistorical Repository Input Audit

Campaign: `COSTA_PHASE2_FRESH_20260816_105951`  
Auditor: `/root/repository_input_audit`  
Decision scope: local exact-file audit only; this document is not the official historical allowlist and does not release execution.

## Verdict

The exact-file audit is complete. All 54 files in the first five candidate groups were opened and SHA-256 hashed. No scientific file was imported or executed, no directory was enumerated, and no dependency was probed.

| Group | Total | Clean | Unresolved | Quarantined | Denied |
|---|---:|---:|---:|---:|---:|
| Documentation and environment | 2 | 1 | 0 | 1 | 0 |
| Model and control implementations | 23 | 2 | 2 | 18 | 1 |
| Tests and observation contracts | 9 | 2 | 0 | 6 | 1 |
| Detector and metric implementations | 9 | 1 | 0 | 7 | 1 |
| Historical control-code inventory only | 11 | 0 | 0 | 11 | 0 |
| **Total** | **54** | **6** | **2** | **43** | **3** |

No clean canonical personalized simulator was established, no scientific winner was selected, and no prior numerical result was imported.

## Audit boundary

The audit first read only these current controls:

- `GOVERNANCE/NONHISTORICAL_METADATA_RELEASE.json`
- `GOVERNANCE/PROTECTED_DENYLIST.json`
- `GOVERNANCE/SANITIZER_OUTPUT_AUDIT.json`
- `IMPORT_ALLOWLIST/NONHISTORICAL_METADATA_CANDIDATES.json`

It then read and hashed only the exact files listed in the first five candidate groups.

The audit did **not** open, hash, stat, preview, enumerate, or execute:

- any `DATA_AND_PARAMETER_METADATA_ONLY` candidate;
- any dependency merely referenced by reviewed code;
- any `docs`, `outputs`, `validation_outputs`, or historical campaign-tree content;
- any protected-denylist match;
- any manifest-resolved PSG or hypnogram;
- any scientific code.

No `STATE` file or official historical allowlist was changed.

## Classification semantics

- `CLEAN_ALLOWLISTED` means clean only for the stated narrow static role. It is not an execution release.
- `UNRESOLVED` means decisive outcome contamination was not found, but method, dependency, provenance, or import safety prevents release.
- `QUARANTINED` means the file contains prior results, outcome-guided choices, data-specific regression values, unsafe historical coupling, or equivalent contamination. It may be used only for governance/forensic inventory.
- `DENIED` means the file’s primary operation necessarily consumes excluded generated outputs, selected checkpoints, or mutable historical outcome state.

## Narrowly clean files

The six narrow clean decisions are:

- `requirements.txt`: package-name inventory only; it is unpinned and not a reproducible environment.
- `models/s3_sleep_kernel.py`: static cortical/thalamic topology and simulator contract only; top-level execution and fixed values remain prohibited.
- `scripts/hce_shift_finite_checks.py`: self-contained finite rational identity checks unrelated to sleep-model evidence.
- `tests/conftest.py`: test bootstrap and unavailable-data skip contract only.
- `tests/test_observation_plots.py`: engineering rendering/publication-safety contract only.
- `valid_scripts/validate_t6_mi_eeg_native.py`: deterministic synthetic signal-shape contract only; its implementation dependency was not reviewed.

Every clean decision remains subject to separate Root authorization and dependency audit before execution.

## Unresolved simulator candidates

Two early personalized simulator candidates contain no embedded fitted winner but remain unresolved:

- `models/s4_personalize.py`: early manifest-driven personalized-fit pipeline.
- `models/s4_personalize_fig7.py`: eight-parameter thalamocortical fit pipeline.

Both depend on an unaudited manifest and dynamic raw-data paths, have import-time optimization/write risks, and contain method or output-naming issues. Neither is canonical.

`models/s4_personalize_fig7_v8.py` is latest by reviewed filename lineage, but it is quarantined. It is outcome-guided, retains prior-derived values, and dynamically depends on an unreviewed `S4_v7_repair` module. “Latest” must not be interpreted as “approved” or “canonical.”

## Denied files

Three scripts are denied as scientific or executable inputs:

- `models/s8_sac_iterate.py`: loads a selected checkpoint and mutable outcome history, ranks checkpoints, and continues an outcome-conditioned policy.
- `tests/test_spindles.py`: loads generated output arrays at import scope and applies uncited pass thresholds.
- `valid_scripts/t5_deconfound.py`: performs import-time post-hoc rescoring of historical evolution outputs.

Only their hashes and provenance reasons may be carried forward.

## Quarantine findings

All 11 files in `HISTORICAL_CONTROL_CODE_GOVERNANCE_INVENTORY_ONLY` are quarantined. They encode outcome-selected points, grids, local-search neighborhoods, relaxed thresholds, rankings, correlations, prior-result narratives, or recommended next experiments.

The broader quarantined set also contains:

- successive version repairs explicitly motivated by earlier failures or loopholes;
- embedded prior winners and bounds narrowed around those winners;
- outcome-retuned T1–T13 thresholds, rewards, and search objectives;
- hard-coded subject/shape/count regression expectations;
- selected fitted-point comparisons described as canonical, confirmatory, or held out;
- raw-EEG comparison scripts whose polarity, ranking, or interpretation is outcome-guided;
- epoch concatenation that can create filter/event boundary artifacts;
- top-level simulation, optimization, output writes, and process exits;
- dynamic path injection, hard-coded machine paths, external library checkout assumptions, private FOOOF internals, broad exception suppression, and stale output-name contracts.

The per-file contamination and prohibition record is in `IMPORT_ALLOWLIST/REPOSITORY_CODE_AUDIT_DECISIONS.json`.

## Value-free capability inventory

The audit identified interfaces without selecting values or a scientific winner:

- simulator candidates expose thalamocortical builders, rate extraction, Welch spectra, periodic-spectrum comparison, fitness dictionaries, and seeded simulation hooks;
- observation contracts cover stage normalization, per-epoch validity masks, strict event containment, typed metadata roles, normalized PSD summaries, and publication-safe exports;
- detector concepts cover slow-oscillation intervals/waveforms, sigma-band RMS events, event density, cycle phase, spindle envelope modulation, Tort MI, preferred phase, concentration, and surrogate-shift nulls;
- historical T1–T6 functionality is present for lower/upper cortical-rate extremes, longest UP-state duration, slow-oscillation peak quality, peak width, and inter-burst-interval regularity;
- `SleepEnv` exposes a Gym-style reset/step interface, a spectrum/rate observation, and a scalar cortical-input intervention, but it is quarantined and rebuilds the simulation at each step;
- SAC training/continuation interfaces exist, but training is quarantined and selected-checkpoint continuation is denied;
- no PID gains, integral/derivative state, or PID update interface was found in the exact reviewed scope.

These are capability descriptions only. They do not validate the methods or authorize their use.

## Dependency disposition

The only exact data/config path explicitly referenced by a clean file is `data/manifest.csv`, through `tests/conftest.py`. It remains:

- `PENDING_INDEPENDENT_DATA_AUDIT`;
- unaccessed;
- without access inheritance;
- without cleanliness inheritance;
- unauthorized for scientific use.

Manifest-derived `psg_path` and `hypnogram_path` remain unresolved symbolic paths with the same status. Clean files also reference unreviewed code under `S4_sbi` and `S4_v7_repair`; those are pending independent code audit.

Every other data/config path mentioned by unresolved, quarantined, or denied code also remains `PENDING_INDEPENDENT_DATA_AUDIT`. A textual reference does not prove existence, cleanliness, independence, or suitability and does not authorize opening.

## Final disposition

This audit may support a later, separately authorized import decision for narrow clean contracts. It does not authorize simulation, fitting, data access, threshold reuse, parameter reuse, checkpoint reuse, scientific ranking, or claim transfer.

Machine-readable outputs:

- `IMPORT_ALLOWLIST/REPOSITORY_CODE_AUDIT_DECISIONS.json`
- `COMMON/VALUE_FREE_REPOSITORY_CAPABILITY_INVENTORY.json`

Status: `COMPLETE_NONHISTORICAL_REPOSITORY_INPUT_AUDIT_NO_EXECUTION_RELEASE`

