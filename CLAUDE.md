# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Research code that **reverse-engineers parameters of a thalamo-cortical mean-field model from real N3 sleep EEG**. Given one Sleep-EDF subject (default `SC4001`), the pipeline:

1. Extracts N3 epochs, computes the target Welch PSD, and runs FOOOF to isolate the periodic component.
2. Simulates cortex + thalamus firing rates with a `neurolib` MultiModel (`ALNNode` + `ThalamicNode`).
3. Scores simulations against the target using a weighted fitness of shape correlation, slow-oscillation / spindle power, and dynamics checks.
4. Drives a `scipy.optimize.differential_evolution` search over 8 parameters and writes the best params as JSON plus a full CSV of evaluations.

## Common commands

All commands are intended to be run **from the repository root** (scripts assume this and several of them `chdir` to it):

```bash
# Main V3 personalization run (long: ~hours; writes data/patient_params_fig7_v3_SC4001.json
# and outputs/evolution_fig7_v3_records.csv)
python models/s4_personalize_fig7_v3.py

# Single-evaluation smoke test + diagnostics for compute_fitness_v3
python tests/test_compute_fitness_v3.py                    # just prints T1–T5 + fitness components
python tests/test_compute_fitness_v3.py --plots            # adds outputs/fig7_v3_test_*.png
python tests/test_compute_fitness_v3.py --plots --out-v2-names  # write under fig7_v2_*.png names

# Fast plotting from an existing best-params JSON
python plot_scripts/plot_fig7_v2_fast.py
python plot_scripts/plot_fig7_v3_fast.py
```

There is no formal test runner, linter, or build step wired up — the `tests/` directory contains hands-on scripts, not `pytest` suites.

## Data prerequisites

Nothing in the main pipeline runs without local data. The scripts expect:

- `data/manifest.csv` with columns `subject_id,psg_path,hypnogram_path` (gitignored; contains local absolute/relative paths to Sleep-EDF files).
- Sleep-EDF files under `data/sleep-edfx-cassette/` (gitignored).
- Target subject from `SUBJECT_ID` constant (default `SC4001`) must be present in the manifest.

If `data/manifest.csv` is missing or the subject is absent, `load_target_psd` fails early.

## Architecture notes worth knowing before editing

### The `s4_personalize_fig7_v*` series is the live main line

`models/` holds many historical stages (`s1_*` through `s8_*`: early stage analysis, bifurcation, RL/SAC experiments). Of these, **only `s4_personalize_fig7_v3.py` is the current personalization main line**; `v1`, `v2`, and `v4*` variants are kept for comparison and should generally not be the starting point for new work. `README.md` and `docs/s4_personalize_fig7_v3_overview.md` are the authoritative walk-throughs.

### FOOOF is optional but changes results

`from fooof import FOOOF` is wrapped in `try/except`, setting `HAS_FOOOF`. When FOOOF is absent, `shape_r` silently falls back to a freq-weighted χ² surrogate that is **not** numerically equivalent to the FOOOF path. If comparing runs or reproducing figures, always note whether FOOOF was installed.

### `utils/02_preprocess_psd.py` is imported by path, not as a module

The filename begins with a digit, so it cannot be imported normally. Both `s4_personalize_fig7_v3.py` and the test script load it via `importlib.util.spec_from_file_location` and pull out `load_hypnogram`, `compute_epoch_psd`, `EPOCH_LEN_S`. Preserve these exported names when editing that file.

### NumPy alias shim for older neurolib

The v3 script patches deprecated aliases (`np.int`, `np.float`, etc.) onto NumPy before importing `neurolib`, because older `neurolib` releases still reference them. Removing the shim will break imports on current NumPy.

### Fitness and dynamics scoring

`compute_fitness_v3` returns `-fitness` for DE minimization. Fitness is
`0.35·shape_r + 0.15·SO + 0.15·spindle_eff + 0.35·dynamics` minus SO-overshoot and narrow-spindle penalties. `dynamics_score` is the weighted sum of 5 binary checks (T1–T5) on post-burn-in `r_ctx` / `r_thal`:

- T1 DOWN exists, T2 UP exists (> ~15 Hz), T3 UP sustained (avoids single-spike gaming of T2), T4 cortical SO peak in ~0.3–1.5 Hz, T5 thalamic spindle peak not too narrow.

Weights and thresholds are constants near the top of the module — changing them changes the entire optimization landscape.

### Module-level state used by DE

`_eval_count`, `_best_score`, `_best_params`, `_records`, `_t_start` are module globals mutated during `differential_evolution`. The test script (`tests/test_compute_fitness_v3.py`) calls `_reset_v3_globals()` before and after a one-shot evaluation; new drivers should do the same to keep logs clean.

### Output conventions

- Best params: `data/patient_params_fig7_{version}_{SUBJECT_ID}.json`
- Evolution CSV: `outputs/evolution_fig7_{version}_records.csv`
- Plot PNGs: `outputs/fig7_{version}_*.png`

`outputs/` is fully gitignored; Sleep-EDF raw data and `data/manifest.csv` are also gitignored.

### Plot scripts must match the fitness PSD+FOOOF pipeline

If a plotting script recomputes `shape_r` to compare against the JSON values, it must use the same order: Welch → interpolate to FOOOF's frequency axis → FOOOF → log-residual correlation. `docs/s4_personalize_fig7_v3_compute_fitness_v3.md` covers this in detail. Diverging from that order produces values that do not match the stored fitness.

## Key documentation in `docs/`

- `s4_personalize_fig7_v3_overview.md` — part-by-part tour of the V3 script.
- `s4_personalize_fig7_v3_compute_fitness_v3.md` — deep dive on `compute_fitness_v3`.
- `compute_fitness_v3_notes.md`, `compute_target_periodic_notes.md`, `load_target_psd_notes.md` — per-function notes.
- `0315_Progress.md`, `0404_Progress.md`, `0414_compute_fitness_v3.md` — dated progress logs.
