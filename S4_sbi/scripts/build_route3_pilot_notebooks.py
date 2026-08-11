"""Build Route-3 pilot notebooks 11-14 without embedding simulation logic."""

from __future__ import annotations

from pathlib import Path
import sys

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "S4_sbi" / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def write_notebook(name: str, cells: list) -> Path:
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
    path = NOTEBOOK_DIR / name
    nbf.write(notebook, path)
    return path


def build_11() -> Path:
    cells = [
        md(
            r"""
# Route-3 Global Robustness: Multi-seed, Multi-scale, and Collision Audit

This notebook tests whether the frozen Notebook-10 **synthetic cortical-rate
14D** contract remains numerically stable across simulator seeds and parameter
regimes. It does not train SNPE and does not use real EEG.

Pre-registered engineering gates:

- simulation failure rate at most 1%;
- at least 99% finite 14D vectors;
- invariant feature dimension and order;
- pickle-free artifacts with explicit simulator seeds.

Scientific warnings do not block the authorized exploratory pilot, but they
must propagate into held-out GO/NO-GO interpretation.
"""
        ),
        code(
            r"""
from __future__ import annotations
from datetime import datetime, timezone
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
    for parent in PROJECT_ROOT.parents:
        if (parent / "S4_sbi" / "src").exists():
            PROJECT_ROOT = parent
            break
SRC = PROJECT_ROOT / "S4_sbi" / "src"
for path in (SRC, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sleep_sbi.route3_global_robustness import (
    COLLISION_NOISE_QUANTILE,
    COLLISION_THETA_RMS_THRESHOLD,
    GLOBAL_ROOT,
    GLOBAL_SIMULATOR_SEEDS,
    LOCAL_SCALES,
    RESULTS_ROOT,
    analyze_local_multiscale,
    analyze_prior_multiseed,
    engineering_gate,
    rate_contract_hash,
    rate_feature_names,
    run_local_multiscale,
    run_prior_multiseed,
)
from sleep_sbi.route3_synthetic_preflight import PARAMETER_NAMES, parameter_contract

FIGURE_DIR = GLOBAL_ROOT / "figures"
HTML_DIR = RESULTS_ROOT / "html"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
HTML_DIR.mkdir(parents=True, exist_ok=True)
environment = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
    "sys_executable": sys.executable,
    "sys_prefix": sys.prefix,
    "python": platform.python_version(),
    "kernel_id": "neurolib",
    "sbi": mdist.version("sbi"),
    "torch": mdist.version("torch"),
    "neurolib": mdist.version("neurolib"),
    "cpu_count": os.cpu_count(),
}
assert Path(sys.prefix).name.lower() == "neurolib"
display(pd.DataFrame(environment.items(), columns=["item", "value"]))
"""
        ),
        md(
            r"""
## 1. Frozen schema and seed-control evidence

Feature definitions, order, units, Welch settings, warm-up, and 30-s window are
unchanged from Notebook 10. Every stochastic input seed and the numba RNG are
set to the recorded integer. A same-theta/same-seed repeat was exactly equal;
changing only the seed changed 12 of 14 summaries.
"""
        ),
        code(
            r"""
contract = parameter_contract()
feature_names = list(rate_feature_names())
assert len(feature_names) == 14 and len(set(feature_names)) == 14
seed_evidence = pd.DataFrame(
    [
        {"check": "same theta + same seed", "result": "max abs diff = 0"},
        {"check": "same theta + different seed", "result": "12/14 summaries changed"},
        {"check": "seeded inputs", "result": "ALN EXC, ALN INH, TCR, TRN plus numba RNG"},
    ]
)
display(pd.DataFrame({"index": range(14), "feature": feature_names}))
display(seed_evidence)
print("Rate contract hash:", rate_contract_hash())
"""
        ),
        md(
            r"""
## 2. 128 theta x 3 simulator seeds

The 128 scrambled Sobol theta values are identical across three replicates.
Replicates are retained separately; seed variation is never averaged away in
the stored bank. The analysis decomposes within-theta seed variance and
between-theta variance.
"""
        ),
        code(
            r"""
prior_run = run_prior_multiseed(max_workers=6)
prior = analyze_prior_multiseed(prior_run["path"])
prior_summary = prior["summary"]
feature_noise = prior["feature_metrics"]
display(pd.DataFrame([prior_summary]))
display(feature_noise)
assert prior_summary["attempted"] == 384
assert prior_summary["success"] == 384
"""
        ),
        md(
            r"""
### Seed-noise signal-to-noise and quantization

`between_to_within_snr` compares variation of theta-specific means against
replicate seed variance. It is not posterior information. Frequencies inherit
the frozen 0.25-Hz Welch grid, so low unique-value counts identify quantization.
"""
        ),
        code(
            r"""
fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
ordered = feature_noise.sort_values("between_to_within_snr")
axes[0].barh(ordered["feature"], ordered["between_to_within_snr"], color="#356a8a")
axes[0].set_xscale("log")
axes[0].axvline(1, color="black", ls="--")
axes[0].set(xlabel="Between-theta variance / within-theta seed variance", title="Feature signal-to-seed-noise")
axes[1].barh(ordered["feature"], ordered["unique_values_min_across_seeds"], color="#6a8a4d")
axes[1].axvline(16, color="black", ls="--", label="quantization warning rule")
axes[1].set(xlabel="Minimum unique values across three seeds", title="Feature value resolution")
axes[1].legend()
fig.savefig(FIGURE_DIR / "feature_seed_snr_and_quantization.png", dpi=180)
fig.savefig(FIGURE_DIR / "feature_seed_snr_and_quantization.svg")
plt.show()
"""
        ),
        md(
            r"""
### Nearest-neighbor stability

For each seed, nearest neighbors are computed after robust scaling on this
synthetic experiment. Low agreement means simulator noise and nonlinear feature
geometry can change which theta appears closest. This is not itself a collision
proof, but it is a warning for density estimation.
"""
        ),
        code(
            r"""
nearest = prior["nearest"]
agreement = np.mean(nearest[:, :, None] == nearest[:, None, :], axis=1)
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
matrix = np.array(
    [
        [1.0, prior_summary["nearest_neighbor_pair_agreement"]["seed_0_vs_1"], prior_summary["nearest_neighbor_pair_agreement"]["seed_0_vs_2"]],
        [prior_summary["nearest_neighbor_pair_agreement"]["seed_0_vs_1"], 1.0, prior_summary["nearest_neighbor_pair_agreement"]["seed_1_vs_2"]],
        [prior_summary["nearest_neighbor_pair_agreement"]["seed_0_vs_2"], prior_summary["nearest_neighbor_pair_agreement"]["seed_1_vs_2"], 1.0],
    ]
)
image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(3), GLOBAL_SIMULATOR_SEEDS)
ax.set_yticks(range(3), GLOBAL_SIMULATOR_SEEDS)
ax.set(xlabel="Simulator seed", ylabel="Simulator seed", title="Nearest-neighbor identity agreement")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", color="white" if matrix[i,j] < .6 else "black")
fig.colorbar(image, ax=ax, label="Agreement fraction")
fig.savefig(FIGURE_DIR / "nearest_neighbor_seed_agreement.png", dpi=180)
fig.savefig(FIGURE_DIR / "nearest_neighbor_seed_agreement.svg")
plt.show()
"""
        ),
        md(
            r"""
## 3. Multi-scale local Jacobians

At V7, V8, and V8a theta centers, central differences use 1%, 2%, and 5% of
prior width under three seeds. All runs use the same V8a dynamics contract;
the center labels refer to theta, not different extractor implementations.

Effective rank remains pre-frozen as
\(\sigma_k/\sigma_1 \ge 10^{-3}\).
"""
        ),
        code(
            r"""
local_run = run_local_multiscale(max_workers=6)
local = analyze_local_multiscale(local_run["path"])
rank = local["rank"]
target_cosines = local["target_cosines"]
direction = local["seed_direction_consistency"]
local_summary = local["summary"].iloc[0].to_dict()
display(rank.groupby(["center", "scale"]).agg(
    rank_min=("effective_rank", "min"),
    rank_max=("effective_rank", "max"),
    condition_median=("condition_number", "median"),
    condition_max=("condition_number", "max"),
))
assert len(rank) == 27
"""
        ),
        code(
            r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for center in rank["center"].unique():
    subset = rank[rank.center == center]
    for seed in GLOBAL_SIMULATOR_SEEDS:
        row = subset[subset.seed == seed]
        axes[0].plot(row.scale * 100, row.effective_rank, marker="o", label=f"{center} | {seed}")
        axes[1].plot(row.scale * 100, row.condition_number, marker="o", label=f"{center} | {seed}")
axes[0].set(xlabel="Perturbation (% prior width)", ylabel="Effective rank", title="Rank stability", ylim=(0, 8.5))
axes[1].set(xlabel="Perturbation (% prior width)", ylabel="Condition number", title="Condition-number sensitivity")
axes[1].legend(fontsize=7, ncol=2)
fig.savefig(FIGURE_DIR / "multiscale_rank_condition.png", dpi=180)
fig.savefig(FIGURE_DIR / "multiscale_rank_condition.svg")
plt.show()
"""
        ),
        md(
            r"""
### Targeted parameter confounding

Cosines near +1 or -1 indicate parallel or anti-parallel Jacobian columns.
Results are shown separately for each seed and scale; the wide variation is
scientifically relevant rather than averaged away.
"""
        ),
        code(
            r"""
display(target_cosines.groupby(["center", "scale", "parameter_a", "parameter_b"]).cosine.agg(["min", "median", "max"]))
worst_directions = direction.sort_values("direction_cosine").head(20)
display(worst_directions)
fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
for (center, pair), subset in target_cosines.assign(
    pair=target_cosines.parameter_a + "/" + target_cosines.parameter_b
).groupby(["center", "pair"]):
    ax.scatter(subset.scale * 100, subset.cosine, label=f"{center} | {pair}", alpha=.75)
ax.axhline(.9, color="black", ls="--")
ax.axhline(-.9, color="black", ls="--")
ax.set(xlabel="Perturbation (% prior width)", ylabel="Jacobian-column cosine", title="g_LK/g_h and bidirectional-coupling confounding")
ax.legend(fontsize=7, ncol=2)
fig.savefig(FIGURE_DIR / "target_parameter_confounds.png", dpi=180)
fig.savefig(FIGURE_DIR / "target_parameter_confounds.svg")
plt.show()
"""
        ),
        md(
            r"""
## 4. Global-collision audit

Pre-registered definitions:

- theta distance is Euclidean distance after prior-width scaling, divided by
  \(\sqrt{8}\);
- x uses median/IQR robust scaling fitted only on the 384 synthetic rows;
- the observable noise floor is the 95th percentile of same-theta replicate
  distances;
- a far pair requires theta RMS distance at least 0.25;
- `collision`: x distance at or below the noise floor;
- `near collision`: x distance between one and two noise floors.

Collisions indicate seed-induced overlap or global folding. They are distinct
from feature-feature correlation and do not contradict a locally full-rank
Jacobian.
"""
        ),
        code(
            r"""
collisions = prior["collisions"]
near_collisions = prior["near_collisions"]
display(pd.DataFrame([prior_summary]))
display(collisions.sort_values("severity_ratio_x_to_noise").head(20))

all_pairs = pd.concat(
    [
        collisions.assign(classification="collision"),
        near_collisions.assign(classification="near collision"),
    ],
    ignore_index=True,
)
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
for label, subset in all_pairs.groupby("classification"):
    ax.scatter(subset.theta_rms_distance, subset.x_distance, s=12, alpha=.45, label=label)
ax.axvline(COLLISION_THETA_RMS_THRESHOLD, color="black", ls="--", label="far-theta threshold")
ax.axhline(prior_summary["noise_floor_q95_robust_x_distance"], color="#b23a48", ls="--", label="seed-noise floor")
ax.axhline(2 * prior_summary["noise_floor_q95_robust_x_distance"], color="#b23a48", ls=":", label="2x noise floor")
ax.set(xlabel="Prior-normalized theta RMS distance", ylabel="Robust-scaled 14D distance", title="Far-theta observable overlaps")
ax.legend()
fig.savefig(FIGURE_DIR / "global_collision_audit.png", dpi=180)
fig.savefig(FIGURE_DIR / "global_collision_audit.svg")
plt.show()
"""
        ),
        md(
            r"""
## 5. Engineering gate and preregistered warnings

The engineering gate controls whether bank generation can start. Passing it
does not establish global identifiability. The collision count, low
nearest-neighbor stability, quantized peak frequencies, and seed/scale
sensitivity are carried forward unchanged into held-out validation.
"""
        ),
        code(
            r"""
gate = engineering_gate(prior_summary, local_summary)
warnings_table = pd.DataFrame(
    [
        {"warning": "far-theta collisions within seed-noise floor", "value": prior_summary["global_collision_count"]},
        {"warning": "near collisions within two noise floors", "value": prior_summary["near_collision_count"]},
        {"warning": "all-three-seed nearest-neighbor agreement", "value": prior_summary["nearest_neighbor_all_three_seed_agreement"]},
        {"warning": "minimum feature SNR", "value": feature_noise["between_to_within_snr"].min()},
        {"warning": "minimum local rank", "value": rank["effective_rank"].min()},
    ]
)
display(pd.DataFrame([gate]))
display(warnings_table)
assert gate["pass"]
"""
        ),
        md(
            r"""
## 6. Artifact reload and conclusion

**Engineering PASS:** 825/825 simulations completed with fixed 14D order and no
non-finite rows. This authorizes the pre-specified 2,048-point exploratory bank.

**Scientific warning:** local rank 8 is not global injectivity. Seed noise,
global collisions, quantized peak frequencies, and parameter-ridge behavior
must be adjudicated by held-out recovery, coverage, SBC, and PPC. No real-EEG
inference is unlocked.
"""
        ),
        code(
            r"""
prior_path = Path(prior_run["path"])
local_path = Path(local_run["path"])
for path, expected_shape in ((prior_path, (384, 14)), (local_path, (441, 14))):
    with np.load(path, allow_pickle=False) as data:
        assert data["x"].shape == expected_shape
        assert list(data["feature_names"]) == feature_names
        assert not [name for name in data.files if data[name].dtype == object]
        assert data["success"].all()
        assert np.isfinite(data["x"]).all()
report = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "engineering_gate": gate,
    "prior_multiseed": prior_summary,
    "local_multiscale": local_summary,
    "pre_registered_scientific_warnings": warnings_table.to_dict("records"),
    "decision": "Proceed to authorized 2048-point exploratory bank; identifiability not established.",
    "real_eeg_inference_unlocked": False,
}
(GLOBAL_ROOT / "notebook11_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
"""
        ),
    ]
    return write_notebook("11_Route3_Global_Robustness.ipynb", cells)


def build_12() -> Path:
    cells = [
        md(
            r"""
# Route-3 2,048-Point Cortex-Rate Simulation Bank

This notebook audits the authorized, resumable 2,048-row bank for the frozen
synthetic cortical-rate 14D schema. It does not train a posterior. Each theta
uses a distinct, deterministic simulator seed; all failures remain in the
manifest. Scaling is fit only on the fixed training split.
"""
        ),
        code(
            r"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "S4_sbi" / "src").exists():
    for parent in PROJECT_ROOT.parents:
        if (parent / "S4_sbi" / "src").exists():
            PROJECT_ROOT = parent
            break
for path in (PROJECT_ROOT / "S4_sbi" / "src", PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sleep_sbi.route3_global_robustness import (
    RESULTS_ROOT,
    TRAINING_SIMULATOR_SEED_BASE,
    TRAINING_SOBOL_SEED,
    TRAIN_VALIDATION_SPLIT_SEED,
    analyze_training_bank_collisions,
    rate_contract_hash,
    rate_feature_names,
    run_training_bank,
)
from sleep_sbi.route3_synthetic_preflight import PARAMETER_NAMES, parameter_contract

OUTPUT_DIR = RESULTS_ROOT / "simulation_bank_2048"
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
assert Path(sys.prefix).name.lower() == "neurolib"
print("Environment:", os.environ.get("CONDA_DEFAULT_ENV"))
print("Python:", sys.executable)
"""
        ),
        md(
            r"""
## 1. Bank contract and resume policy

- exactly 2,048 scrambled Sobol theta values (`sobol_seed=20260731`);
- V8a eight-parameter uniform box and Notebook-10 feature order;
- per-row simulator seed schedule, not a common seed;
- atomic per-simulation NPZ checkpoint;
- invalid rows are never silently deleted;
- train/validation split is fixed before training;
- x location/scale are computed from training rows only.
"""
        ),
        code(
            r"""
bank_run = run_training_bank(max_workers=6)
manifest = bank_run["manifest"]
display(pd.DataFrame([manifest]))
assert manifest["attempted"] == 2048
assert manifest["failed"] / manifest["attempted"] <= 0.01
"""
        ),
        md(
            r"""
## 2. Artifact integrity, feature order, and split

The validation split remains part of model training diagnostics, not the later
128-case held-out recovery set. The held-out Sobol sequence and simulator seeds
will be independent.
"""
        ),
        code(
            r"""
with np.load(bank_run["bank_path"], allow_pickle=False) as bank:
    theta = np.asarray(bank["theta"], dtype=float)
    x = np.asarray(bank["x"], dtype=float)
    simulator_seeds = np.asarray(bank["simulator_seed"], dtype=int)
    success = np.asarray(bank["success"], dtype=bool)
    runtime_s = np.asarray(bank["runtime_s"], dtype=float)
    assert not [name for name in bank.files if bank[name].dtype == object]
    assert list(bank["parameter_names"]) == list(PARAMETER_NAMES)
    assert list(bank["feature_names"]) == list(rate_feature_names())
with np.load(bank_run["split_path"], allow_pickle=False) as split:
    train_idx = np.asarray(split["training_indices"], dtype=int)
    validation_idx = np.asarray(split["validation_indices"], dtype=int)
    x_location = np.asarray(split["x_location"], dtype=float)
    x_scale = np.asarray(split["x_scale"], dtype=float)
    assert not [name for name in split.files if split[name].dtype == object]
assert theta.shape == (2048, 8)
assert x.shape == (2048, 14)
assert success.all() and np.isfinite(x).all()
assert len(np.unique(simulator_seeds)) == 2048
assert len(train_idx) == 1638 and len(validation_idx) == 410
assert not np.intersect1d(train_idx, validation_idx).size
display(pd.DataFrame(
    [{"rows": 2048, "train": len(train_idx), "validation": len(validation_idx),
      "unique_seeds": len(np.unique(simulator_seeds)), "runtime_sum_s": runtime_s.sum(),
      "runtime_median_s": np.median(runtime_s)}]
))
"""
        ),
        md(
            r"""
## 3. Prior coverage and feature distributions

Theta panels show design coverage, not posterior samples. Feature panels use
training-split robust scaling only for comparing relative variation; this does
not equalize physical meaning and is not refit on validation or held-out data.
"""
        ),
        code(
            r"""
bounds = np.asarray(parameter_contract()["bounds"], dtype=float)
theta_unit = (theta - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
x_scaled = (x - x_location) / x_scale
fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
axes[0].boxplot([theta_unit[:, i] for i in range(8)], tick_labels=PARAMETER_NAMES)
axes[0].set(ylabel="Normalized prior coordinate", title="2,048-point Sobol coverage", ylim=(-.03, 1.03))
axes[0].tick_params(axis="x", rotation=45)
axes[1].boxplot([x_scaled[:, i] for i in range(14)], tick_labels=range(14), showfliers=False)
axes[1].set(xlabel="Frozen feature index", ylabel="Training-scaled value", title="14D relative variation")
fig.savefig(FIGURE_DIR / "bank_prior_and_feature_coverage.png", dpi=180)
fig.savefig(FIGURE_DIR / "bank_prior_and_feature_coverage.svg")
plt.show()
"""
        ),
        md(
            r"""
## 4. Runtime and failure audit

Green points are successful simulations; failed rows would remain red with NaN
features and explicit reasons. No parameter region is removed post hoc.
"""
        ),
        code(
            r"""
status = pd.read_csv(OUTPUT_DIR / "bank_status.csv")
fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
ax.scatter(status.sample_id, status.runtime_s, c=np.where(status.success, "#2c7a4b", "#b23a48"), s=8)
ax.set(xlabel="Sample ID", ylabel="Runtime (s)", title="Per-simulation runtime and failure status")
fig.savefig(FIGURE_DIR / "bank_runtime_failure.png", dpi=180)
fig.savefig(FIGURE_DIR / "bank_runtime_failure.svg")
plt.show()
display(status.groupby(["split", "success"]).agg(rows=("sample_id", "count"), runtime_s=("runtime_s", "sum")))
"""
        ),
        md(
            r"""
## 5. Higher-resolution global-collision audit

The Stage-11 same-theta seed-noise floor remains frozen. Theta distance uses
prior-width scaling and x distance uses only training-split scaling. A large
collision fraction warns that multiple far theta values can be observationally
indistinguishable at the measured seed noise level, even though local
Jacobians were rank 8.
"""
        ),
        code(
            r"""
collision = analyze_training_bank_collisions(bank_run["bank_path"], bank_run["split_path"])
collision_summary = collision["summary"]
collision_table = collision["collisions"]
display(pd.DataFrame([collision_summary]))
display(collision_table.head(25))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
axes[0].hist(collision_table["theta_rms_distance"], bins=35, color="#356a8a")
axes[0].set(xlabel="Prior-normalized theta RMS distance", ylabel="Collision pairs", title="Theta separation among collisions")
axes[1].hist(collision_table["x_to_noise_ratio"], bins=35, color="#b06b4f")
axes[1].set(xlabel="14D distance / seed-noise floor", ylabel="Collision pairs", title="Collision severity")
fig.savefig(FIGURE_DIR / "bank_collision_distributions.png", dpi=180)
fig.savefig(FIGURE_DIR / "bank_collision_distributions.svg")
plt.show()
"""
        ),
        md(
            r"""
## 6. Training gate

The engineering gate passes because all 2,048 vectors are finite and failure
rate is zero. The high collision burden is a preregistered scientific warning,
not a reason to alter features after seeing results. It must be resolved by
held-out recovery, coverage, SBC, posterior ridge checks, and PPC.
"""
        ),
        code(
            r"""
training_gate = {
    "failure_rate_threshold": 0.01,
    "observed_failure_rate": manifest["failure_rate"],
    "schema_order_match": True,
    "fully_finite": bool(np.isfinite(x).all()),
    "training_rows": len(train_idx),
    "validation_rows": len(validation_idx),
    "scaling_fit_on_training_only": True,
    "collision_fraction_of_far_pairs": collision_summary["collision_fraction_of_far_pairs"],
    "pass": bool(manifest["failure_rate"] <= .01 and np.isfinite(x).all()),
    "scientific_identifiability_established": False,
}
(OUTPUT_DIR / "notebook12_validation.json").write_text(json.dumps(training_gate, indent=2), encoding="utf-8")
display(pd.DataFrame([training_gate]))
assert training_gate["pass"]
"""
        ),
        md(
            r"""
## Conclusion

The 2,048-point bank is technically valid for the authorized exploratory
single-round NPE training: 2,048/2,048 rows are finite, the split and scaling
are frozen, and artifacts are portable. It is not evidence that all eight
parameters are identifiable. Approximately the reported fraction of far-theta
pairs fall inside the empirical seed-noise floor, so false posterior
contraction is a primary held-out failure mode.
"""
        ),
    ]
    return write_notebook("12_Route3_2048_Simulation_Bank.ipynb", cells)


def build_13() -> Path:
    cells = [
        md(
            r"""
# Exploratory Route-3 Synthetic Cortical-Rate NPE Ensemble

Three independently initialized, single-round conditional density estimators
are trained on the same fixed 1,638/410 split. The proposal is the frozen prior;
there are no sequential rounds. The result is always called an **exploratory
Route-3 synthetic cortical-rate posterior**, never an EEG posterior.
"""
        ),
        code(
            r"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "S4_sbi" / "src").exists():
    for parent in PROJECT_ROOT.parents:
        if (parent / "S4_sbi" / "src").exists():
            PROJECT_ROOT = parent
            break
for path in (PROJECT_ROOT / "S4_sbi" / "src", PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sleep_sbi.route3_global_robustness import RESULTS_ROOT, rate_feature_names
from sleep_sbi.route3_pilot_snpe import (
    ARCHITECTURE,
    MODEL_SEEDS,
    TRAINING_POLICY,
    TRAINING_ROOT,
    exploratory_member_checks,
    load_ensemble_members,
    sample_equal_ensemble_normalized,
    train_ensemble,
    write_go_criteria,
)
from sleep_sbi.route3_synthetic_preflight import PARAMETER_NAMES

BANK_ROOT = RESULTS_ROOT / "simulation_bank_2048"
FIGURE_DIR = TRAINING_ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
assert Path(sys.prefix).name.lower() == "neurolib"
print("Environment:", os.environ.get("CONDA_DEFAULT_ENV"))
print("Python:", sys.executable)
"""
        ),
        md(
            r"""
## 1. Frozen training protocol

- MAF conditional density estimator, 64 hidden features, five transforms;
- explicit train-only x median/IQR scaling and normalized prior coordinates;
- batch size 128, Adam learning rate 5e-4, gradient clipping 5;
- best-validation checkpoint plus resumable last checkpoint;
- maximum 300 epochs, early-stopping patience 25;
- seeds 1301, 1302, 1303;
- equal ensemble weights, with no weighting fitted on held-out results.
"""
        ),
        code(
            r"""
training = train_ensemble(
    BANK_ROOT / "route3_cortex_rate_14d_bank_2048.npz",
    BANK_ROOT / "split_and_scaling.npz",
)
data = training["data"]
summary = training["summary"]
members = load_ensemble_members(data)
training_audit = json.loads(
    (TRAINING_ROOT / "training_execution_audit.json").read_text(encoding="utf-8")
)
member_frame = pd.DataFrame(summary["members"])
member_frame["initial_training_runtime_s"] = member_frame["seed"].astype(str).map(
    training_audit["member_runtime_s"]
)
display(
    member_frame[
        [
            "seed", "device", "epochs_completed", "best_epoch",
            "best_validation_loss", "final_train_loss",
            "final_validation_loss", "initial_training_runtime_s",
        ]
    ]
)
print("Initial ensemble training stage runtime (s):", training_audit["ensemble_stage_runtime_s"])
display(pd.DataFrame([{"architecture": ARCHITECTURE, "policy": TRAINING_POLICY}]))
assert len(members) == 3
"""
        ),
        md(
            r"""
## 2. Training and validation loss

Negative loss values are valid log-density objectives. Model selection uses
only the fixed validation split. Differences between member curves reflect
initialization and minibatch order, not different simulation data.
"""
        ),
        code(
            r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for seed in MODEL_SEEDS:
    history = pd.read_csv(TRAINING_ROOT / f"member_{seed}" / "training_history.csv")
    axes[0].plot(history.epoch, history.train_loss, label=str(seed), alpha=.8)
    axes[1].plot(history.epoch, history.validation_loss, label=str(seed), alpha=.8)
axes[0].set(xlabel="Epoch", ylabel="Negative log density", title="Training loss")
axes[1].set(xlabel="Epoch", ylabel="Negative log density", title="Validation loss")
axes[0].legend(title="Initialization seed")
axes[1].legend(title="Initialization seed")
fig.savefig(FIGURE_DIR / "ensemble_training_curves.png", dpi=180)
fig.savefig(FIGURE_DIR / "ensemble_training_curves.svg")
plt.show()
"""
        ),
        md(
            r"""
## 3. Pre-held-out member checks

Thirty-two positions from the internal validation split are used only for
engineering checks: finite samples, prior support, boundary accumulation,
posterior dispersion, and member disagreement. These are not the independent
128-case scientific held-out set.
"""
        ),
        code(
            r"""
checks = exploratory_member_checks(data, members)
checks.to_csv(TRAINING_ROOT / "member_checks.csv", index=False)
member_summary = checks.groupby("member").agg(
    finite_rate=("finite_rate", "mean"),
    in_prior_rate=("within_prior_rate", "mean"),
    median_error=("mean_normalized_median_error", "mean"),
    boundary_fraction=("boundary_sample_fraction", "mean"),
    posterior_std=("mean_posterior_std", "mean"),
)
display(member_summary)
assert (member_summary.finite_rate == 1).all()
assert (member_summary.in_prior_rate == 1).all()
"""
        ),
        md(
            r"""
## 4. Transparent equal-mixture sampling

For any observation, requested samples are divided as evenly as possible among
the three members, concatenated, and shuffled with a recorded seed. No member
receives a data-dependent weight.
"""
        ),
        code(
            r"""
case_index = len(data.x_validation) // 2
samples, labels = sample_equal_ensemble_normalized(
    members, data.x_validation[case_index], 4096, seed=20260801
)
assert samples.shape == (4096, 8)
assert np.isfinite(samples).all()
assert ((samples >= 0) & (samples <= 1)).all()
mixture_counts = pd.Series(labels).value_counts().sort_index()
display(pd.DataFrame({"member": range(3), "samples": mixture_counts.values, "weight": mixture_counts.values / 4096}))

fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
for parameter_index, ax in enumerate(axes.flat):
    for member_index in range(3):
        ax.hist(samples[labels == member_index, parameter_index], bins=35, histtype="step", density=True, label=f"member {member_index}")
    ax.axvline(data.theta_validation[case_index, parameter_index], color="black", ls="--")
    ax.set(title=PARAMETER_NAMES[parameter_index], xlabel="Normalized prior coordinate")
axes[0,0].legend(fontsize=7)
fig.suptitle("Exploratory member marginals for one internal-validation case")
fig.savefig(FIGURE_DIR / "representative_member_marginals.png", dpi=180)
fig.savefig(FIGURE_DIR / "representative_member_marginals.svg")
plt.show()
"""
        ),
        md(
            r"""
## 5. Ridge and boundary screening

The `g_LK/g_h` and bidirectional-coupling panels screen for the confounds
observed in Notebooks 10-12. A visible ridge is not automatically failure if
uncertainty and coverage are honest; a falsely narrow ridge is.
"""
        ),
        code(
            r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
axes[0].scatter(samples[:, 4], samples[:, 5], c=labels, s=5, alpha=.3, cmap="viridis")
axes[0].scatter(data.theta_validation[case_index, 4], data.theta_validation[case_index, 5], c="red", marker="x", s=80)
axes[0].set(xlabel="g_LK normalized", ylabel="g_h normalized", title="g_LK / g_h posterior geometry")
axes[1].scatter(samples[:, 6], samples[:, 7], c=labels, s=5, alpha=.3, cmap="viridis")
axes[1].scatter(data.theta_validation[case_index, 6], data.theta_validation[case_index, 7], c="red", marker="x", s=80)
axes[1].set(xlabel="c_th2ctx normalized", ylabel="c_ctx2th normalized", title="Bidirectional coupling geometry")
fig.savefig(FIGURE_DIR / "representative_ridges.png", dpi=180)
fig.savefig(FIGURE_DIR / "representative_ridges.svg")
plt.show()
"""
        ),
        md(
            r"""
## 6. GO criteria freeze

The operational GO/CONDITIONAL-GO/NO-GO thresholds are written before any
independent held-out theta is generated or opened. Subsequent notebooks may
evaluate but must not rewrite this artifact.
"""
        ),
        code(
            r"""
criteria_path = RESULTS_ROOT / "heldout_validation" / "go_criteria_frozen.json"
criteria = write_go_criteria(criteria_path)
display(pd.DataFrame(criteria["formal_go"].items(), columns=["formal GO criterion", "frozen value"]))
print("Criteria path:", criteria_path)
"""
        ),
        md(
            r"""
## Conclusion

All three models trained and return finite prior-supported samples. This is an
engineering prerequisite, not scientific validation. Independent held-out
recovery, nominal coverage, SBC, ensemble disagreement, and new-seed synthetic
PPC determine the formal decision.
"""
        ),
    ]
    return write_notebook("13_Route3_Exploratory_SNPE_Ensemble.ipynb", cells)


def build_14() -> Path:
    cells = [
        md(
            r"""
# Route-3 Held-out Recovery, Coverage, SBC, and Synthetic PPC

This notebook evaluates the exploratory three-member ensemble on 128
independent synthetic cortical-rate observations. The GO criteria artifact was
frozen before these theta values were generated. No threshold is changed after
viewing results.

Scope: recoverability of eight model parameters in synthetic cortical-rate
observable space only. This does not validate an EEG measurement model or
unlock real-EEG inference.
"""
        ),
        code(
            r"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "S4_sbi" / "src").exists():
    for parent in PROJECT_ROOT.parents:
        if (parent / "S4_sbi" / "src").exists():
            PROJECT_ROOT = parent
            break
for path in (PROJECT_ROOT / "S4_sbi" / "src", PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sleep_sbi.route3_global_robustness import RESULTS_ROOT, rate_feature_names
from sleep_sbi.route3_heldout_validation import (
    HELDOUT_ROOT,
    POSTERIOR_SAMPLES_PER_CASE,
    analyze_recovery_and_coverage,
    evaluate_go_decision,
    generate_heldout_posterior_samples,
    run_heldout_dataset,
    run_synthetic_ppc,
)
from sleep_sbi.route3_synthetic_preflight import PARAMETER_NAMES

BANK_ROOT = RESULTS_ROOT / "simulation_bank_2048"
FIGURE_DIR = HELDOUT_ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
assert Path(sys.prefix).name.lower() == "neurolib"
print("Environment:", os.environ.get("CONDA_DEFAULT_ENV"))
print("Python:", sys.executable)
"""
        ),
        md(
            r"""
## 1. Immutable GO criteria

Formal GO requires simulation reliability, almost entirely finite/in-support
posterior samples, nominally compatible 80% and 90% coverage for at least 7/8
parameters, no severe undercoverage, recovery improvement for at least 6/8,
honest contraction for at least 6/8, PPC improvement, and stable ensemble
conclusions. Conditional GO has weaker but still pre-specified limits.
"""
        ),
        code(
            r"""
criteria_path = HELDOUT_ROOT / "go_criteria_frozen.json"
criteria = json.loads(criteria_path.read_text(encoding="utf-8"))
display(pd.DataFrame(criteria["formal_go"].items(), columns=["criterion", "frozen value"]))
print("Frozen UTC:", criteria["frozen_utc"])
"""
        ),
        md(
            r"""
## 2. Independent 128-case held-out dataset

The scrambled Sobol sequence and simulator seed schedule differ from training.
Exact theta duplicates are rejected. Held-out observations never refit scaling,
features, architecture, or hyperparameters.
"""
        ),
        code(
            r"""
bank_path = BANK_ROOT / "route3_cortex_rate_14d_bank_2048.npz"
split_path = BANK_ROOT / "split_and_scaling.npz"
heldout = run_heldout_dataset(bank_path, max_workers=6)
display(pd.DataFrame([heldout["manifest"]]))
assert heldout["manifest"]["success"] == 128
assert heldout["manifest"]["exact_training_theta_duplicates"] == 0
"""
        ),
        md(
            r"""
## 3. Equal-ensemble posterior samples

Each case has exactly 4,096 prior-supported samples, split as evenly as possible
among three independently initialized members. Samples and member labels are
checkpointed per case. Posterior width is not itself a success metric.
"""
        ),
        code(
            r"""
posterior = generate_heldout_posterior_samples(
    heldout["path"], bank_path, split_path
)
assert posterior["samples"].shape == (128, POSTERIOR_SAMPLES_PER_CASE, 8)
assert np.isfinite(posterior["samples"]).all()
assert ((posterior["samples"] >= 0) & (posterior["samples"] <= 1)).all()
print("Posterior samples:", posterior["samples"].shape)
print("Sampling runtime sum (s):", posterior["runtime_s"].sum())
"""
        ),
        md(
            r"""
## 4. Parameter recovery against prior-median baseline

Axes use normalized prior coordinates. The dashed diagonal is perfect recovery.
Posterior medians are compared with the fixed prior median (0.5). Error bars
show 90% credible intervals.
"""
        ),
        code(
            r"""
recovery = analyze_recovery_and_coverage(posterior["path"])
recovery_table = recovery["recovery"]
display(recovery_table)

fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
for parameter_index, ax in enumerate(axes.flat):
    true = recovery["theta"][:, parameter_index]
    samples = recovery["samples"][:, :, parameter_index]
    median = np.median(samples, axis=1)
    low, high = np.quantile(samples, [0.05, 0.95], axis=1)
    ax.errorbar(true, median, yerr=[median-low, high-median], fmt=".", ms=3, alpha=.45)
    ax.plot([0,1], [0,1], color="black", ls="--")
    ax.set(xlabel="True normalized theta", ylabel="Posterior median", title=PARAMETER_NAMES[parameter_index], xlim=(0,1), ylim=(0,1))
fig.suptitle("Held-out posterior recovery with 90% intervals (N=128)")
fig.savefig(FIGURE_DIR / "heldout_recovery_scatter.png", dpi=180)
fig.savefig(FIGURE_DIR / "heldout_recovery_scatter.svg")
plt.show()
"""
        ),
        md(
            r"""
## 5. Coverage and binomial uncertainty

Points are empirical coverage and vertical bars are Wilson 95% binomial
intervals. A nominal level is compatible only if it lies inside the interval.
The decision uses 80% and 90% levels parameter by parameter.
"""
        ),
        code(
            r"""
coverage = recovery["coverage"]
ensemble_coverage = coverage[coverage.estimator == "ensemble"]
display(ensemble_coverage)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
for ax, level in zip(axes, (0.5, 0.8, 0.9)):
    subset = ensemble_coverage[ensemble_coverage.nominal_level == level].set_index("parameter").loc[list(PARAMETER_NAMES)]
    values = subset.empirical_coverage.to_numpy()
    lower = values - subset.wilson_95_low.to_numpy()
    upper = subset.wilson_95_high.to_numpy() - values
    ax.errorbar(range(8), values, yerr=[lower, upper], fmt="o", capsize=3)
    ax.axhline(level, color="black", ls="--")
    ax.set_xticks(range(8), PARAMETER_NAMES, rotation=45, ha="right")
    ax.set(ylabel="Empirical coverage", title=f"Nominal {int(level*100)}%", ylim=(0,1.03))
fig.savefig(FIGURE_DIR / "heldout_coverage.png", dpi=180)
fig.savefig(FIGURE_DIR / "heldout_coverage.svg")
plt.show()
"""
        ),
        md(
            r"""
## 6. SBC rank diagnostics

Ranks are normalized because ensemble and member sample counts differ slightly.
Ten-bin chi-square values are descriptive warnings, not standalone decisions.
Coverage, bias, contraction, and the 128-case sample size remain primary.
"""
        ),
        code(
            r"""
sbc = recovery["sbc"]
ensemble_sbc = sbc[sbc.estimator == "ensemble"].set_index("parameter").loc[list(PARAMETER_NAMES)]
fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
for parameter, ax in zip(PARAMETER_NAMES, axes.flat):
    row = ensemble_sbc.loc[parameter]
    counts = [row[f"bin_{i}"] for i in range(10)]
    ax.bar(np.arange(10), counts, color="#356a8a")
    ax.axhline(128/10, color="black", ls="--")
    ax.set(title=f"{parameter} | p={row.chi_square_p_value:.3g}", xlabel="Normalized rank bin")
fig.suptitle("Ensemble SBC ranks (N=128; p-values are descriptive)")
fig.savefig(FIGURE_DIR / "heldout_sbc_ranks.png", dpi=180)
fig.savefig(FIGURE_DIR / "heldout_sbc_ranks.svg")
plt.show()
display(ensemble_sbc[["rank_mean", "rank_std", "chi_square_p_value"]])
"""
        ),
        md(
            r"""
## 7. Ensemble disagreement and posterior ridges

Member-median ranges are measured in prior widths. Ridge correlations summarize
posterior dependence for `g_LK/g_h` and the two coupling parameters. Honest
ridge uncertainty is acceptable; false narrowness or undercoverage is not.
"""
        ),
        code(
            r"""
disagreement = recovery["disagreement"]
ridges = recovery["ridges"]
display(disagreement)
display(ridges.describe())
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
axes[0].boxplot([ridges.g_LK_g_h_spearman, ridges.c_th2ctx_c_ctx2th_spearman], tick_labels=["g_LK/g_h", "couplings"])
axes[0].set(ylabel="Within-posterior Spearman rho", title="Held-out posterior ridge correlations")
axes[1].bar(disagreement.parameter, disagreement.mean_member_median_range)
axes[1].axhline(.1, color="black", ls="--", label="frozen mean-disagreement gate")
axes[1].tick_params(axis="x", rotation=45)
axes[1].set(ylabel="Mean member-median range (prior width)", title="Ensemble member disagreement")
axes[1].legend()
fig.savefig(FIGURE_DIR / "heldout_ridges_disagreement.png", dpi=180)
fig.savefig(FIGURE_DIR / "heldout_ridges_disagreement.svg")
plt.show()
"""
        ),
        md(
            r"""
## 8. New-seed synthetic posterior predictive checks

Thirty-two cases are selected by a fixed random seed before looking at PPC.
Each receives 16 posterior theta draws and 16 independent prior-predictive
draws. Both are newly simulated with seeds unused by held-out observations.
Typical cases are closest to the median recovery score within the random set;
worst cases are the five largest recovery errors within that same set.
"""
        ),
        code(
            r"""
ppc = run_synthetic_ppc(
    recovery, heldout["path"], bank_path, split_path, max_workers=6
)
display(pd.DataFrame([ppc["summary"]]))
display(ppc["metrics"])
display(pd.DataFrame([ppc["selection"]]))

metrics = ppc["metrics"]
fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
positions = np.arange(14)
width = .38
ax.bar(positions-width/2, metrics.posterior_predictive_median_scaled_abs_error, width, label="Posterior predictive")
ax.bar(positions+width/2, metrics.prior_predictive_median_scaled_abs_error, width, label="Prior predictive")
ax.set_xticks(positions, range(14))
ax.set(xlabel="Frozen feature index", ylabel="Median scaled absolute error", title="Posterior vs prior predictive error (32 x 16 simulations)")
ax.legend()
fig.savefig(FIGURE_DIR / "synthetic_ppc_comparison.png", dpi=180)
fig.savefig(FIGURE_DIR / "synthetic_ppc_comparison.svg")
plt.show()
"""
        ),
        md(
            r"""
## 9. Frozen decision

The following result is computed directly from the immutable criteria. Training
success and a strong PPC do not override systematic calibration failure.
"""
        ),
        code(
            r"""
decision = evaluate_go_decision(
    criteria_path,
    BANK_ROOT / "bank_manifest.json",
    HELDOUT_ROOT / "dataset" / "heldout_manifest.json",
    recovery,
    ppc,
)
display(pd.DataFrame([{"check": key, "value": value} for key, value in decision["checks"].items()]))
print("FORMAL DECISION:", decision["decision"])
assert decision["decision"] in {"GO", "CONDITIONAL GO", "NO-GO"}
"""
        ),
        md(
            r"""
## Conclusion

The exploratory ensemble improves median recovery over the prior baseline for
seven parameters and strongly improves 14D synthetic PPC. Nevertheless,
nominal coverage is compatible for too few parameters, and `c_ctx2th` shows no
recovery improvement plus severe 80% undercoverage. Under the frozen rules the
decision is **NO-GO** for claiming validated eight-parameter Route-3 recovery.

Recommended next work is not an automatic 4K/8K expansion. First address the
weak `c_ctx2th` direction and calibration: consider a pre-registered seven-
parameter experiment with `c_ctx2th` fixed, reparameterization, feature
quantization changes decided independently of this held-out set, or
calibration-aware training. Real-EEG SNPE remains prohibited by the unresolved
measurement model.
"""
        ),
    ]
    return write_notebook("14_Route3_Heldout_Recovery_and_Coverage.ipynb", cells)


def main() -> None:
    requested = set(sys.argv[1:] or ["11"])
    paths = []
    if "11" in requested:
        paths.append(build_11())
    if "12" in requested:
        paths.append(build_12())
    if "13" in requested:
        paths.append(build_13())
    if "14" in requested:
        paths.append(build_14())
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
