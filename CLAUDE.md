# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Research code that **reverse-engineers parameters of a thalamo-cortical mean-field model from real N3 sleep EEG**. Given one Sleep-EDF subject (default `SC4001`), the pipeline:

1. Extracts N3 epochs, computes the target Welch PSD, and runs FOOOF to isolate the periodic component.
2. Simulates cortex + thalamus firing rates with a `neurolib` MultiModel (`ALNNode` + `ThalamicNode`).
3. Scores simulations against the target using a weighted fitness of shape correlation, SO/spindle power, and dynamics checks (T1–T12).
4. Drives a `scipy.optimize.differential_evolution` search over 8 parameters and writes the best params as JSON plus a full CSV of evaluations.

## Common commands

All commands are run **from the repository root** (scripts `chdir` to it). Activate the environment first:

```bash
conda activate neurolib

# ── Stage 2: SBI (run in this order) ─────────────────────────────────────────
python S4_sbi/compute_xobs_from_eeg.py           # extract 8-dim x_obs from EEG
python S4_sbi/simulator_wrapper.py               # smoke test: 1 sim with Seed B
python S4_sbi/run_sbi.py --dry-run               # 50-sim pipeline test, no saves
python S4_sbi/run_sbi.py                         # full 5000-sim run (~10 h)

# ── Stage 1: V7 personalization ───────────────────────────────────────────────
# Main V7 personalization run (~3-4 hours; 60s sims, 160 individuals × 30 gens)
python models/s4_personalize_fig7_v7.py

# Warm-start DE with fixed PAC metrics (10 gens, ~1-2 hours; uses v7's top-K as seeds)
python S4_v7_repair/warm_start_de_with_fixed_pac.py
python S4_v7_repair/warm_start_de_with_fixed_pac.py --gens 5   # quick smoke test
python S4_v7_repair/warm_start_de_with_fixed_pac.py --top 20   # 20 seeds

# V3 personalization run (kept for baseline; 30s sims, ~hours)
python models/s4_personalize_fig7_v3.py

# Single-evaluation smoke test for V3 fitness
python tests/test_compute_fitness_v3.py                    # prints T1–T5 + fitness
python tests/test_compute_fitness_v3.py --plots            # adds outputs/fig7_v3_test_*.png

# Seed reproducibility + T1–T12 constraint diagnosis for V7 Pareto seeds
python tests/test_seed_reproducibility.py

# PAC fix validation (9 synthetic-signal experiments; outputs to validation_outputs/)
python valid_scripts/validate_compute_pac_metrics_fixed.py

# V7 PAC-phase diagnosis (4-layer cross-validation of preferred_phase ≈ 159° bug)
python S4_v7_repair/diagnose_v7_phase.py

# Fast plotting from an existing best-params JSON
python plot_scripts/plot_fig7_v3_fast.py
python plot_scripts/plot_fig7_compare_v7_vs_v8.py          # side-by-side V7 vs warm-start
```

There is no formal test runner, linter, or build step — `tests/` and `valid_scripts/` contain hands-on scripts, not pytest suites.

## Data prerequisites

Nothing in the main pipeline runs without local data. The scripts expect:

- `data/manifest.csv` with columns `subject_id,psg_path,hypnogram_path` (gitignored).
- Sleep-EDF files under `data/sleep-edfx-cassette/` (gitignored).
- Target subject from `SUBJECT_ID` constant (default `SC4001`) must be present in the manifest.

If `data/manifest.csv` is missing or the subject is absent, `load_target_psd` fails early.

## Architecture notes worth knowing before editing

### Current mainline: V7 (not V3)

`models/` holds historical stages `s1_*` through `s8_*`. The **active personalization mainline is `s4_personalize_fig7_v7.py`**; v3 is retained as a clean baseline. Key V7 changes from V3:

- `SIM_DUR_MS`: 30 000 → 60 000 ms (halves T6 IBI_CV sampling noise).
- `spindle_power` reward: FOOOF-based → event-based (`min(T12_n_verified / 15, 1.0)`).
- `so_power` reward: FOOOF-based → T4_q-derived (`clip((T4_q-1)/4, 0, 1)`).
- `c_th2ctx` bounds: `[0.05, 0.25]` → `[0.00, 0.05]` (wider range destroys cortex SO).
- T5 FWHM threshold: `> 2.0 Hz` → `> 0.2 Hz` (v3's 2 Hz rejected mechanistically valid narrow spindles).
- Dynamics checks extended to T1–T12 (v3 had T1–T5 only); adds PAC three-pack (T9–T11) and spindle event count (T12).

### PAC phase bug in V7 and its fix

V7's PAC computation (`compute_constraints_v7`) uses bandpass + Hilbert to extract the SO phase. For spike-like cortical firing-rate signals (long flat DOWN states followed by sharp UP peaks), this **systematically biases the preferred phase** away from the UP-state peak (~0°) toward ~159°. Root causes: (1) filter group delay shifts peak timing, (2) near-zero signal in DOWN states causes Hilbert phase instability.

The fix is in `S4_v7_repair/compute_pac_metrics_fixed.py` (also mirrored in `valid_scripts/`): uses **cycle-by-cycle phase** — `find_peaks` locates UP peaks directly in the time series and assigns phase by linear interpolation between peaks. This is convention-free and immune to waveform shape. Validation: `valid_scripts/validate_compute_pac_metrics_fixed.py` runs 9 synthetic experiments; all 9 pass. The warm-start DE run in `warm_start_de/` uses this fixed PAC.

### FOOOF is optional but changes results

`from fooof import FOOOF` is wrapped in `try/except`, setting `HAS_FOOOF`. When absent, `shape_r` silently falls back to a freq-weighted χ² surrogate that is **not** numerically equivalent to the FOOOF path. When comparing runs or reproducing figures, always note whether FOOOF was installed.

### `utils/02_preprocess_psd.py` is imported by path, not as a module

The filename begins with a digit so it cannot be imported normally. Both v3 and v7 scripts (and several diagnostic scripts) load it via `importlib.util.spec_from_file_location` and pull out `load_hypnogram`, `compute_epoch_psd`, `EPOCH_LEN_S`. Preserve these exported names when editing that file.

### NumPy alias shim for older neurolib

The scripts patch deprecated aliases (`np.int`, `np.float`, etc.) onto NumPy **before** importing `neurolib`, because older `neurolib` releases reference them. Removing the shim breaks imports on current NumPy. The shim must come before any `neurolib` import, including indirect ones.

### Local neurolib takes precedence over system install

The system-installed `neurolib` has a `compile_to_numba` bug (`co_consts[-3] IndexError`) on Python 3.11+. Several scripts (e.g. `plot_fig7_compare_v7_vs_v8.py`) prepend `D:\Year3_Mao_Projects\neurolib` to `sys.path` to use the locally patched copy. If adding new scripts that import `neurolib`, replicate this `sys.path` prepend.

### Fitness and dynamics scoring

`compute_fitness_v7` (and v3) returns `-fitness` for DE minimization. V7 fitness is:
`0.35·shape_r + 0.15·so_reward + 0.15·spindle_reward + 0.35·dynamics`
minus SO-overshoot and narrow-spindle penalties.

`dynamics_score` is the weighted sum of 5 continuous checks (T1–T8 + T12 mapped to feasibility scores). The 5 binary checks from v3 (T1–T5) are superseded in v7 by 12 checks covering UP/DOWN states, SO frequency, spindle FWHM, IBI regularity (T6), spindle waxing-waning (T7–T8), PAC three-pack (T9–T11), and event count (T12).

Weights and thresholds are constants near the top of each module — changing them changes the entire optimization landscape.

### Module-level state used by DE

`_eval_count`, `_best_score`, `_best_params`, `_records`, `_t_start` are module globals mutated during `differential_evolution`. The v3 test script calls `_reset_v3_globals()` before and after a one-shot evaluation; new drivers should do the same to keep logs clean. V7 test scripts import the module via `importlib` to get the same isolation.

### Numba seeding for reproducibility

V7 adds `seed_numba(seed)` (a `@numba.njit` function that calls `np.random.seed`) to make Ornstein-Uhlenbeck noise deterministic. Without seeding, stochastic inputs differ between runs and PAC metrics are not reproducible across calls with identical parameters. `tests/test_seed_reproducibility.py` verifies this property.

### Output conventions

- Best params: `data/patient_params_fig7_{version}_{SUBJECT_ID}.json`
- Evolution CSV: `outputs/evolution_fig7_{version}_records.csv`
- Plot PNGs: `outputs/fig7_{version}_*.png`
- Warm-start outputs: `warm_start_de/patient_params_warm_start.json`, `warm_start_de/warm_start_records.csv`

`outputs/` is fully gitignored; Sleep-EDF raw data and `data/manifest.csv` are also gitignored.

### Plot scripts must match the fitness PSD+FOOOF pipeline

If a plotting script recomputes `shape_r` to compare against JSON values, it must use the same order: Welch → interpolate to FOOOF's frequency axis → FOOOF → log-residual correlation. Diverging from that order produces values that don't match the stored fitness.

## Stage 2 SBI architecture (`S4_sbi/`)

Three files implement the SNPE-C inference pipeline. Run them in order.

**Free parameters**: `['g_h', 'g_LK', 'c_ctx2th', 'b']` — prior from V7 BOUNDS.  
**Fixed params** (Seed B): `mue, mui, tauA, c_th2ctx` — read from `S4_v7_repair/pareto_seeds_fresh_DE.json`.

- `compute_xobs_from_eeg.py` — extracts 8-dim `x_obs` from SC4001 N3 EEG via detrend→abs→50ms smooth→rescale proxy. Saves `S4_sbi/x_obs.npz`. Has hard-stop sanity checks; do not add fallbacks.
- `simulator_wrapper.py` — loads V7 via importlib at module init, loads target PSD once, exposes `simulator(theta_4d) → np.ndarray(8,)`. Returns `np.full(8, np.nan)` on failure.
- `run_sbi.py` — 4-round SNPE-C (sbi 0.26, NSF, `z_score_x='independent'`). Checkpoints after every round. 5 diagnostics: SBC (200 sims), PPC (100 sims), marginals, pairplot, Pareto overlay. Aborts if NaN rate > 30% or round produces < 200 valid sims. `--dry-run` flag runs 50 sims only.

**Summary stats** (SUMMARY_KEYS, same order in both files):  
`shape_r, T4_q, T4_freq, T6_ibi_cv, T8_n_sp_events, T11_lag_ms, T12_n_verified, MI`

**Important**: `pareto_seeds_fresh_DE.json` is the correct filename (with underscore). The spec had a typo (`pareto_seeds_freshDE.json`).

## Key documentation in `docs/`

- `s4_personalize_fig7_v3_overview.md` — part-by-part tour of the V3 script.
- `s4_personalize_fig7_v3_compute_fitness_v3.md` — deep dive on `compute_fitness_v3`.
- `T7_Spindle_Envelope_Burstiness_0501.md` — explains T7's `butter→sosfiltfilt→hilbert→abs` envelope CV signal chain.
- `PAC0502.md` — documents the V7 PAC phase-convention bug, the cycle-by-cycle fix, and the 9-experiment validation.
- `compute_fitness_v3_notes.md`, `compute_target_periodic_notes.md`, `load_target_psd_notes.md` — per-function notes.
- `Progress_0422.md`, `0414_compute_fitness_v3.md`, `0315_Progress.md`, `0404_Progress.md` — dated progress logs.
