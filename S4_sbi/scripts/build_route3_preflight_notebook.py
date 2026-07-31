"""Build the Route-3 synthetic recovery preflight notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "S4_sbi" / "notebooks" / "10_Route3_Synthetic_Recovery_Preflight.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        r"""
# Route-3 Synthetic Cortical-Observable Recovery Preflight

This notebook asks a deliberately narrow question: do eight model parameters

`mue, mui, b, tauA, g_LK, g_h, c_th2ctx, c_ctx2th`

produce numerically stable and locally distinguishable changes in explicitly
declared **synthetic cortical observables**?

## Scientific boundary

- `r_mean_EXC` and `r_mean_INH` are cortical population firing rates in Hz, not simulated EEG.
- ALN current/synaptic states are privileged model-internal observables, not scalp measurements.
- No 75-uV rule, EEG channel semantics, arbitrary Hz-to-uV scaling, real-EEG feature selection, posterior training, or real-EEG inference is used.
- A full-rank local Jacobian is necessary evidence for local identifiability at a center, but is not proof of global recovery, calibration, or a valid posterior.
- The 64-point Sobol set is a `diagnostic_microbank`, never an SNPE training bank.
"""
    ),
    code(
        r"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata as mdist
import json
import os
from pathlib import Path
import platform
import sys

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "S4_sbi" / "src").exists():
    for parent in Path.cwd().resolve().parents:
        if (parent / "S4_sbi" / "src").exists():
            PROJECT_ROOT = parent
            break
SRC = PROJECT_ROOT / "S4_sbi" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sleep_sbi.route3_synthetic_preflight import (
    AUGMENTED_FEATURES,
    HELD_OUT_DIAGNOSTICS,
    PARAMETER_NAMES,
    RATE_ONLY_FEATURES,
    RELATIVE_SINGULAR_VALUE_THRESHOLD,
    RESULTS_ROOT,
    SCHEMA_VERSION,
    analyze_local_sensitivity,
    export_core_contract,
    feature_dictionary,
    feature_redundancy,
    local_stability_gate,
    parameter_contract,
    parameter_table,
    run_diagnostic_microbank,
    run_local_sensitivity,
)

OUTPUT_DIR = RESULTS_ROOT
FIGURE_DIR = OUTPUT_DIR / "figures"
HTML_DIR = OUTPUT_DIR / "html"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
HTML_DIR.mkdir(parents=True, exist_ok=True)

packages = {}
for package in ("numpy", "scipy", "pandas", "matplotlib", "neurolib", "torch", "sbi", "nbformat"):
    try:
        packages[package] = mdist.version(package)
    except mdist.PackageNotFoundError:
        packages[package] = "not installed"

environment = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "working_directory": str(Path.cwd()),
    "project_root": str(PROJECT_ROOT),
    "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
    "sys_executable": sys.executable,
    "sys_prefix": sys.prefix,
    "python_version": platform.python_version(),
    "kernel_id": "neurolib",
    "kernel_display_name": "Python (neurolib)",
    "packages": packages,
}
assert Path(sys.executable).resolve() == Path(
    r"C:\Users\YUS190\AppData\Local\anaconda3\envs\neurolib\python.exe"
).resolve()
assert Path(sys.prefix).name.lower() == "neurolib"
(OUTPUT_DIR / "environment_report.json").write_text(
    json.dumps(environment, indent=2), encoding="utf-8"
)
pd.DataFrame(
    [{"item": key, "value": value} for key, value in environment.items() if key != "packages"]
)
"""
    ),
    md(
        r"""
## 1. Freeze the eight-parameter contract

The latest V8a uniform box is used for prior-wide diagnostics because its
`c_th2ctx` upper bound (0.075) contains V7, V8, and the V8a local-best center.
V7's historical upper bound was 0.05. The parameter order is never inferred
from dictionary iteration.

The simulator integrates at 0.1 ms, records at 1 ms (1000 Hz), runs for 60 s,
removes 5 s warm-up, analyzes the first complete native 30-s window, and leaves
25 s unused. Fixed simulator seed 42 supplies common random numbers across the
finite differences. Failed or non-finite runs retain theta and use invalid/NaN
features with an explicit reason; undefined values are never filled with zero.
"""
    ),
    code(
        r"""
contract_paths = export_core_contract()
contract = parameter_contract()
parameters = parameter_table()
assert tuple(contract["parameter_names"]) == PARAMETER_NAMES
assert len(PARAMETER_NAMES) == 8
assert all(center["inside_v8a_prior"] for center in contract["centers"])
display(parameters)
display(pd.DataFrame(contract["centers"]).drop(columns="theta"))
print("Contract hash:", contract["contract_hash"])
print("All V7/V8/V8a centers inside V8a prior:", all(c["inside_v8a_prior"] for c in contract["centers"]))
"""
    ),
    md(
        r"""
### Historical 4-parameter banks are audit evidence only

The old 5D and 7D archives contain four theta columns. They cannot be relabelled,
padded, or reused as an eight-parameter training bank. This cell checks their
stored shapes and names without loading them into the new diagnostic bank.
"""
    ),
    code(
        r"""
historical_paths = [
    PROJECT_ROOT / "S4_sbi" / "sbi_outputs" / "all_simulations.npz",
    PROJECT_ROOT / "S4_sbi" / "sbi_outputs_7dim_archive_20260507" / "all_simulations.npz",
]
historical_rows = []
for path in historical_paths:
    with np.load(path, allow_pickle=True) as archive:
        theta_key = "theta" if "theta" in archive.files else "thetas"
        x_key = "x" if "x" in archive.files else "xs"
        historical_rows.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "theta_shape": tuple(archive[theta_key].shape),
                "x_shape": tuple(archive[x_key].shape),
                "training_reuse_decision": "forbidden: historical 4-parameter bank",
            }
        )
historical_table = pd.DataFrame(historical_rows)
display(historical_table)
assert all(row[0][1] == 4 for row in zip(historical_table["theta_shape"]))
"""
    ),
    md(
        r"""
## 2. Two nested Route-3 schema candidates

### A. `cortex_rate_only_14d`

Seven summaries are applied separately to cortical excitatory and inhibitory
population rates: mean, standard deviation, relative SO-band power, relative
sigma-band power, slow-band peak frequency, sigma-band peak frequency, and
normalized spectral entropy. Welch uses a Hann window, 4-s segments, 50%
overlap, and 0.25-Hz resolution. These are model-rate statistics, not EEG
statistics.

### B. `cortex_state_augmented_24d`

The 14D prefix is preserved exactly, then ten summaries of `I_mu_EXC`,
`I_mu_INH`, adaptation `I_A`, effective drive, and two mean synaptic states are
added. This is a **privileged internal-state upper-bound experiment**: success
cannot establish recoverability from real scalp EEG.
"""
    ),
    code(
        r"""
feature_table = feature_dictionary()
rate_names = [spec.name for spec in RATE_ONLY_FEATURES]
augmented_names = [spec.name for spec in AUGMENTED_FEATURES]
assert len(rate_names) == 14
assert len(augmented_names) == 24
assert augmented_names[:14] == rate_names
assert len(set(rate_names)) == 14
assert len(set(augmented_names)) == 24
display_columns = [
    "index", "name", "source_signal", "formula", "unit", "role",
    "validity_rule", "redundancy",
]
display(feature_table[feature_table.schema == "cortex_rate_only_14d"][display_columns])
display(
    feature_table[
        (feature_table.schema == "cortex_state_augmented_24d")
        & (feature_table["index"] >= 14)
    ][display_columns]
)
"""
    ),
    md(
        r"""
### Held-out synthetic diagnostics

Inference features cannot also serve as independent predictive evidence in the
same experiment. Joint rate timing, distributional waveform shape, additional
synaptic-state variability, and thalamic mechanism metrics are therefore
reserved. Thalamic states remain mechanism diagnostics, not cortical or scalp
observables.
"""
    ),
    code(
        r"""
held_out_table = pd.DataFrame(HELD_OUT_DIAGNOSTICS)
display(held_out_table)
"""
    ),
    md(
        r"""
## 3. Local sensitivity experiment

At each V7, V8, and V8a center, every parameter is perturbed symmetrically by
2% of its V8a prior width. The standardized central-difference Jacobian is

\[
J_{ij} =
\frac{\Delta x_i / s_i}
     {\Delta \theta_j / (u_j-\ell_j)},
\]

where each feature scale \(s_i\) is fixed as the larger of the center magnitude
and a unit-aware numerical floor. The effective-rank rule was frozen before
viewing results:

\[
\sigma_k / \sigma_1 \ge 10^{-3}.
\]

Rank 8 is called *local full-rank* only. It does not prove global injectivity,
recoverability under noise, or calibrated SBI.
"""
    ),
    code(
        r"""
local_results = run_local_sensitivity(reuse_checkpoints=True)
local_analysis = analyze_local_sensitivity(local_results)
local_runs = pd.read_csv(OUTPUT_DIR / "local_sensitivity" / "simulation_runs.csv")
rank_table = local_analysis["tables"]["rank"]
sensitivity_table = local_analysis["tables"]["sensitivity"]
confounding_table = local_analysis["tables"]["confounding"]
direction_table = local_analysis["tables"]["direction_consistency"]

assert len(local_results) == 51
assert local_runs["rate_finite"].all()
assert local_runs["augmented_finite"].all()
assert local_runs["success"].all()
display(local_runs.groupby(["center", "success"], as_index=False).agg(
    simulations=("run_index", "count"), runtime_s=("runtime_s", "sum")
))
"""
    ),
    code(
        r"""
display(rank_table)
assert (rank_table["effective_rank"] <= 8).all()
print("Pre-frozen effective-rank threshold:", RELATIVE_SINGULAR_VALUE_THRESHOLD)
print("All six center/schema Jacobians locally full-rank:", rank_table["full_local_rank"].all())
"""
    ),
    md(
        r"""
### Jacobian structure

Rows are standardized features and columns are standardized parameter
perturbations. Large absolute color means a strong local response under the
declared feature scaling. Color sign is a local direction, not a causal effect
that must remain constant elsewhere in the prior.
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(3, 2, figsize=(15, 15), constrained_layout=True)
centers = ["v7_fitted", "v8_fitted", "v8a_local_best"]
schemas = ["cortex_rate_only_14d", "cortex_state_augmented_24d"]
for row_index, center in enumerate(centers):
    for column_index, schema in enumerate(schemas):
        matrix = local_analysis["matrices"][(center, schema)]
        limit = np.nanpercentile(np.abs(matrix), 98)
        image = axes[row_index, column_index].imshow(
            matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit
        )
        axes[row_index, column_index].set_title(f"{center} | {schema}")
        axes[row_index, column_index].set_xticks(range(8), PARAMETER_NAMES, rotation=45, ha="right")
        axes[row_index, column_index].set_ylabel("Ordered feature index")
        fig.colorbar(image, ax=axes[row_index, column_index], label="Standardized derivative")
fig.suptitle("Local standardized Jacobians (synthetic model-observable space)", fontsize=15)
fig.savefig(FIGURE_DIR / "local_jacobians.png", dpi=180)
fig.savefig(FIGURE_DIR / "local_jacobians.svg")
plt.show()
"""
    ),
    md(
        r"""
### Singular values and parameter sensitivity

The dashed line is the pre-frozen relative singular-value threshold. The
sensitivity norm summarizes the length of one Jacobian column; it depends on
the declared feature scaling and should be compared within this contract, not
read as a physical unit. A weak column or a pair of nearly parallel columns
signals a possible local ambiguity.
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for _, row in rank_table.iterrows():
    relative = [row[f"relative_singular_value_{i}"] for i in range(1, 9)]
    axes[0].semilogy(range(1, 9), relative, marker="o", label=f"{row['center']} | {row['schema'].replace('cortex_', '')}")
axes[0].axhline(RELATIVE_SINGULAR_VALUE_THRESHOLD, color="black", ls="--", label="rank threshold")
axes[0].set(xlabel="Singular-value index", ylabel="Relative singular value", title="Local Jacobian spectra")
axes[0].legend(fontsize=7)

rate_sensitivity = sensitivity_table[sensitivity_table["schema"] == "cortex_rate_only_14d"]
pivot = rate_sensitivity.pivot(index="parameter", columns="center", values="sensitivity_norm").loc[list(PARAMETER_NAMES)]
pivot.plot.bar(ax=axes[1])
axes[1].set(ylabel="Jacobian column norm", title="Rate-only local parameter sensitivity")
axes[1].tick_params(axis="x", rotation=45)
fig.savefig(FIGURE_DIR / "singular_values_and_sensitivity.png", dpi=180)
fig.savefig(FIGURE_DIR / "singular_values_and_sensitivity.svg")
plt.show()
"""
    ),
    md(
        r"""
### Targeted confounding and cross-center direction consistency

Cosine values near +1 or -1 indicate locally parallel or anti-parallel response
directions. This explicitly audits `g_LK` versus `g_h` and `c_th2ctx` versus
`c_ctx2th`. Negative cross-center cosines show that a parameter's local feature
direction can reverse across regimes, warning against extrapolating one local
Jacobian to the entire prior.
"""
    ),
    code(
        r"""
target_pairs = confounding_table[
    ((confounding_table["parameter_a"] == "g_LK") & (confounding_table["parameter_b"] == "g_h"))
    | ((confounding_table["parameter_a"] == "c_th2ctx") & (confounding_table["parameter_b"] == "c_ctx2th"))
].copy()
weakest = (
    sensitivity_table.sort_values(["center", "schema", "sensitivity_norm"])
    .groupby(["center", "schema"], as_index=False)
    .head(3)
)
inconsistent = direction_table.sort_values("sensitivity_direction_cosine").head(12)
display(target_pairs)
display(weakest)
display(inconsistent)
"""
    ),
    md(
        r"""
## 4. Checkpointed 64-point prior-wide diagnostic micro-bank

Because both schemas passed the basic numerical gate, a scrambled Sobol design
with fixed seed 20260727 is evaluated over the full V8a 8D box. Every completed
simulation is atomically checkpointed. The internal simulator seed remains 42.
All theta rows, failures, validity masks, runtimes, and reasons are retained.

This sample is too small to be a formal training bank and is not used to train
SNPE in this notebook.
"""
    ),
    code(
        r"""
rate_gate = local_stability_gate(local_results, "cortex_rate_only_14d")
augmented_gate = local_stability_gate(local_results, "cortex_state_augmented_24d")
assert rate_gate[0] and augmented_gate[0]
microbank = run_diagnostic_microbank(n_samples=64, reuse_checkpoint=True)

assert microbank["theta"].shape == (64, 8)
assert microbank["rate_x"].shape == (64, 14)
assert microbank["augmented_x"].shape == (64, 24)
assert np.allclose(microbank["augmented_x"][:, :14], microbank["rate_x"], rtol=0, atol=0, equal_nan=True)

microbank_summary = pd.DataFrame(
    [
        {
            "simulations": len(microbank["theta"]),
            "completed": int(microbank["completed"].sum()),
            "success": int(microbank["success"].sum()),
            "failed": int((~microbank["success"]).sum()),
            "rate_finite_rows": int(np.isfinite(microbank["rate_x"]).all(axis=1).sum()),
            "augmented_finite_rows": int(np.isfinite(microbank["augmented_x"]).all(axis=1).sum()),
            "runtime_s": float(np.nansum(microbank["runtime_s"])),
            "median_runtime_s": float(np.nanmedian(microbank["runtime_s"])),
        }
    ]
)
display(microbank_summary)
"""
    ),
    md(
        r"""
### Prior coverage and runtime/failure accounting

Each parameter panel shows normalized prior position, not a posterior sample.
The right panel reports runtime by Sobol index; failed rows would be retained
and marked rather than silently filtered.
"""
    ),
    code(
        r"""
bounds = np.asarray(contract["bounds"], dtype=float)
theta_unit = (microbank["theta"] - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
axes[0].boxplot([theta_unit[:, i] for i in range(8)], tick_labels=PARAMETER_NAMES, showfliers=True)
axes[0].set(ylabel="Normalized prior coordinate", title="64-point Sobol prior coverage", ylim=(-0.03, 1.03))
axes[0].tick_params(axis="x", rotation=45)
colors = np.where(microbank["success"], "#2c7a4b", "#b23a48")
axes[1].scatter(np.arange(64), microbank["runtime_s"], c=colors, s=28)
axes[1].set(xlabel="Sobol index", ylabel="Runtime (s)", title="Simulation runtime and failures")
fig.savefig(FIGURE_DIR / "microbank_coverage_and_runtime.png", dpi=180)
fig.savefig(FIGURE_DIR / "microbank_coverage_and_runtime.svg")
plt.show()
"""
    ),
    md(
        r"""
### Feature redundancy across the micro-bank

Spearman correlation detects monotonic co-variation but does not prove
deterministic equivalence. The 64-point matrix is a screening diagnostic:
correlation can reflect the Sobol design, nonlinear regimes, or common
parameter drivers. It does not justify deleting features without recovery and
held-out tests.
"""
    ),
    code(
        r"""
rate_corr, rate_pair_n = feature_redundancy(microbank["rate_x"], rate_names)
aug_corr, aug_pair_n = feature_redundancy(microbank["augmented_x"], augmented_names)
rate_corr.to_csv(OUTPUT_DIR / "diagnostic_microbank" / "rate_feature_spearman.csv")
aug_corr.to_csv(OUTPUT_DIR / "diagnostic_microbank" / "augmented_feature_spearman.csv")
rate_pair_n.to_csv(OUTPUT_DIR / "diagnostic_microbank" / "rate_valid_pair_counts.csv")
aug_pair_n.to_csv(OUTPUT_DIR / "diagnostic_microbank" / "augmented_valid_pair_counts.csv")

fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
for ax, corr, title in (
    (axes[0], rate_corr, "Rate-only 14D"),
    (axes[1], aug_corr, "Privileged-state 24D"),
):
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title(f"{title} Spearman correlation (N=64)")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Feature index")
    fig.colorbar(image, ax=ax, label="Spearman rho")
fig.savefig(FIGURE_DIR / "microbank_feature_redundancy.png", dpi=180)
fig.savefig(FIGURE_DIR / "microbank_feature_redundancy.svg")
plt.show()

high_pairs = []
for schema, corr in (("rate14", rate_corr), ("augmented24", aug_corr)):
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            rho = float(corr.iloc[i, j])
            if np.isfinite(rho) and abs(rho) >= 0.9:
                high_pairs.append(
                    {"schema": schema, "feature_a": corr.index[i], "feature_b": corr.columns[j], "spearman_rho": rho, "valid_pair_n": 64}
                )
high_pairs = pd.DataFrame(high_pairs).sort_values("spearman_rho", key=np.abs, ascending=False)
high_pairs.to_csv(OUTPUT_DIR / "diagnostic_microbank" / "high_correlation_pairs.csv", index=False)
display(high_pairs)
"""
    ),
    md(
        r"""
## 5. Preflight decision

Passing conditions observed here:

- exact 8D V8a prior contract; V7/V8/V8a centers are inside it;
- 51/51 local simulations and 64/64 Sobol diagnostics completed;
- fixed 14D and 24D vectors with no NaN/Inf;
- all three local Jacobians reach rank 8 under the pre-frozen threshold.

Limits that remain:

- local direction reversals and `g_LK`/`g_h` near-parallelism occur in important regimes;
- peak-frequency features are quantized at 0.25 Hz;
- one fixed stochastic seed tests common-random-number sensitivity, not robustness to simulator noise;
- 64 prior points cannot establish global injectivity or synthetic recovery;
- privileged internal states cannot justify real-EEG inference.

**Decision:** conditional GO for a staged, checkpointed **Route-3 diagnostic
bank expansion and synthetic recovery protocol**. It is still a NO-GO for
SNPE/NPE training until seed robustness, scaling on a training split, global
collision checks, held-out theta recovery, and validation budget are frozen.
"""
    ),
    code(
        r"""
decision_table = pd.DataFrame(
    [
        {
            "question": "Cortex-rate-only locally full-rank at V7/V8/V8a?",
            "result": bool(rank_table[rank_table.schema == "cortex_rate_only_14d"]["full_local_rank"].all()),
            "interpretation": "Necessary local evidence only; not global recoverability.",
        },
        {
            "question": "Did privileged internal states increase effective rank?",
            "result": False,
            "interpretation": "Both schemas are rank 8; internal states improve conditioning at V8/V8a but not V7.",
        },
        {
            "question": "64-point prior-wide numerical stability?",
            "result": bool(microbank["success"].all()),
            "interpretation": "64/64 finite under one fixed simulator seed.",
        },
        {
            "question": "Allow staged diagnostic bank expansion next?",
            "result": True,
            "interpretation": "Conditional GO with checkpointing and explicit recovery design.",
        },
        {
            "question": "Allow SNPE training now?",
            "result": False,
            "interpretation": "No: global recovery, seed robustness, scaling, split, and calibration protocol remain unfrozen.",
        },
    ]
)
display(decision_table)
"""
    ),
    md(
        r"""
## 6. Artifact reload and validation

The final cell reloads JSON, CSV, and NPZ without pickle, verifies shapes,
feature order, finite masks, schema version, and the absence of object arrays.
It also writes a machine-readable validation report. This validates artifact
integrity, not scientific identifiability.
"""
    ),
    code(
        r"""
contract_reload = json.loads(Path(contract_paths["contract"]).read_text(encoding="utf-8"))
features_reload = pd.read_csv(contract_paths["features"])
heldout_reload = pd.read_csv(contract_paths["held_out"])
bank_path = Path(microbank["path"])
with np.load(bank_path, allow_pickle=False) as bank_reload:
    object_arrays = [name for name in bank_reload.files if bank_reload[name].dtype == object]
    assert not object_arrays
    assert bank_reload["theta"].shape == (64, 8)
    assert bank_reload["rate_x"].shape == (64, 14)
    assert bank_reload["augmented_x"].shape == (64, 24)
    assert list(bank_reload["parameter_names"]) == list(PARAMETER_NAMES)
    assert list(bank_reload["rate_feature_names"]) == rate_names
    assert list(bank_reload["augmented_feature_names"]) == augmented_names
    assert str(bank_reload["schema_version"].item()) == SCHEMA_VERSION
    assert np.isfinite(bank_reload["theta"]).all()
    assert np.isfinite(bank_reload["rate_x"]).all()
    assert np.isfinite(bank_reload["augmented_x"]).all()

validation_report = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "schema_version": SCHEMA_VERSION,
    "contract_hash": contract_reload["contract_hash"],
    "parameter_dimension": 8,
    "rate_schema_dimension": 14,
    "augmented_schema_dimension": 24,
    "local_simulations": len(local_results),
    "local_failures": int((~local_runs["success"]).sum()),
    "local_effective_ranks": rank_table[["center", "schema", "effective_rank"]].to_dict("records"),
    "diagnostic_microbank_rows": 64,
    "diagnostic_microbank_failures": int((~microbank["success"]).sum()),
    "all_record_vectors_finite": bool(
        np.isfinite(microbank["rate_x"]).all() and np.isfinite(microbank["augmented_x"]).all()
    ),
    "npz_object_arrays": object_arrays,
    "snpe_training_authorized": False,
    "next_step_decision": "conditional GO for staged diagnostic bank expansion; SNPE remains NO-GO",
}
(OUTPUT_DIR / "validation_report.json").write_text(
    json.dumps(validation_report, indent=2), encoding="utf-8"
)
assert len(features_reload[features_reload.schema == "cortex_rate_only_14d"]) == 14
assert len(features_reload[features_reload.schema == "cortex_state_augmented_24d"]) == 24
assert len(heldout_reload) == len(HELD_OUT_DIAGNOSTICS)
print(json.dumps(validation_report, indent=2))
"""
    ),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {
        "display_name": "Python (neurolib)",
        "language": "python",
        "name": "neurolib",
    },
    "language_info": {
        "name": "python",
        "version": "3.10.20",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py",
    },
}
TARGET.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, TARGET)
print(TARGET)
