"""Build notebooks 15-18 as thin, reproducible views over tested 7D modules."""

from __future__ import annotations

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "S4_sbi" / "notebooks"


def md(value: str):
    return nbf.v4.new_markdown_cell(value.strip())


def code(value: str):
    return nbf.v4.new_code_cell(value.strip())


def write(name: str, cells: list) -> Path:
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python (neurolib)",
            "language": "python",
            "name": "neurolib",
        },
        "language_info": {"name": "python", "version": "3.10.20"},
    }
    path = NOTEBOOK_DIR / name
    nbf.write(notebook, path)
    return path


ENV_CELL = r"""
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
for value in (PROJECT_ROOT / "S4_sbi" / "src", PROJECT_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

environment = {
    "executed_utc": datetime.now(timezone.utc).isoformat(),
    "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
    "sys.executable": sys.executable,
    "sys.prefix": sys.prefix,
    "python": platform.python_version(),
    "kernel_id": "neurolib",
    "kernel_display_name": "Python (neurolib)",
    "numpy": mdist.version("numpy"),
    "torch": mdist.version("torch"),
    "sbi": mdist.version("sbi"),
    "neurolib": mdist.version("neurolib"),
}
assert Path(sys.prefix).name.lower() == "neurolib"
display(pd.DataFrame(environment.items(), columns=["item", "value"]))
"""


def build15() -> Path:
    return write(
        "15_Route3_7D_Preregistration_and_Robustness.ipynb",
        [
            md(r"""
# Route-3 7D Preregistration and Independent Robustness

This notebook freezes and verifies the independent seven-free-parameter
experiment. `c_ctx2th` is fixed by the deterministic V8a-local-best rule before
new simulations. The observable remains the frozen 14D **cortical population
firing-rate** summary; it is not simulated EEG and does not resolve the Fpz-Cz
measurement-model blocker.

The engineering hard gate is evaluated before the 4,096-row training bank.
Scientific warnings such as collisions or ridges are retained without changing
the preregistered design.
"""),
            code(ENV_CELL),
            code(r"""
from sleep_sbi.route3_7d_experiment import *

prereg_hash = verify_preregistration()
prereg = read_preregistration()
print("Preregistration SHA-256:", prereg_hash)
display(pd.DataFrame({
    "parameter": prereg["parameter_order_7d"],
    "unit": prereg["parameter_units_7d"],
    "lower": [row[0] for row in prereg["prior_bounds_7d"]],
    "upper": [row[1] for row in prereg["prior_bounds_7d"]],
}))
display(pd.DataFrame([prereg["fixed_parameter"]]))
assert prereg["fixed_parameter"]["selection_rule"].startswith("Notebook 10")
assert prereg["seed_nonoverlap_audit"]["intersection_count"] == 0
"""),
            md(r"""
## Independent 128 x 3 prior-wide seed experiment

Each of 128 new Sobol parameter vectors is simulated with three new stochastic
seeds. Replicates are stored separately. The between-theta/within-theta variance
ratio measures parameter signal relative to simulator-seed noise; it does not
prove posterior identifiability.
"""),
            code(r"""
prior_run = run_preflight_multiseed(max_workers=6)
prior = analyze_preflight_multiseed(prior_run["path"])
display(pd.DataFrame([prior["summary"]]))
display(prior["features"])

fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
ordered = prior["features"].sort_values("between_to_within_snr")
axes[0].barh(ordered.feature, ordered.between_to_within_snr, color="#356a8a")
axes[0].set_xscale("log")
axes[0].axvline(1, color="black", ls="--")
axes[0].set(xlabel="Between-theta / within-theta variance", title="Parameter signal vs seed noise")
axes[1].barh(ordered.feature, ordered.unique_values_min_across_seeds, color="#708a45")
axes[1].axvline(16, color="black", ls="--")
axes[1].set(xlabel="Minimum unique values across seeds", title="Quantization / degeneracy screen")
figure_dir = PREFLIGHT_ROOT / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(figure_dir / "seed_noise_and_quantization.png", dpi=180)
fig.savefig(figure_dir / "seed_noise_and_quantization.svg")
plt.show()
"""),
            md(r"""
## Multi-scale local Jacobians

The seven free parameters are perturbed by 1%, 2%, and 5% of their unchanged
8D-prior marginal widths around V7, V8, and V8a centers under three seeds.
Effective rank uses the frozen rule \(\sigma_k/\sigma_1 \ge 10^{-3}\).
`c_ctx2th` is inserted at the same fixed value for every run.
"""),
            code(r"""
local_run = run_local_multiscale_7d(max_workers=6)
local = analyze_local_multiscale_7d(local_run["path"])
display(pd.DataFrame([local["summary"]]))
display(local["rank"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
for center, frame in local["rank"].groupby("center"):
    axes[0].plot(frame.scale, frame.effective_rank, "o", label=center, alpha=.7)
axes[0].axhline(7, color="black", ls="--")
axes[0].set(xlabel="Perturbation fraction of prior width", ylabel="Effective rank", title="7D local rank")
axes[0].legend()
pair = local["pair_cosines"]
target = pair[((pair.parameter_a == "g_LK") & (pair.parameter_b == "g_h"))]
for center, frame in target.groupby("center"):
    axes[1].scatter(frame.scale, frame.cosine, alpha=.7, label=center)
axes[1].axhline(0, color="black", lw=.8)
axes[1].set(xlabel="Perturbation fraction", ylabel="Jacobian direction cosine", title="g_LK / g_h local confounding")
axes[1].legend()
fig.savefig(figure_dir / "local_rank_and_gLK_gH.png", dpi=180)
fig.savefig(figure_dir / "local_rank_and_gLK_gH.svg")
plt.show()
"""),
            md(r"""
## Engineering hard gate

The next stage is permitted only when failures are at most 1%, at least 99% of
vectors are finite, feature order is invariant, seeds are reproducible, and
7-to-8 parameter insertion has no silent error. Local full rank is reported as
scientific evidence but is not substituted for held-out recovery.
"""),
            code(r"""
gate = evaluate_preflight_gate(prior, local)
display(pd.DataFrame([gate]))
assert gate["pass"]
assert local["summary"]["rank_max"] <= 7
print("Stage-15 hard gate: PASS")
"""),
        ],
    )


def build16() -> Path:
    return write(
        "16_Route3_7D_4096_Simulation_Bank.ipynb",
        [
            md(r"""
# Independent 4,096-point Route-3 7D Simulation Bank

This notebook generates the preregistered scrambled-Sobol bank with new theta
and simulator seeds. Each 7D vector is expanded to the simulator's 8D interface
by inserting the locked `c_ctx2th`. Checkpoints are atomic and manifest-driven.
The bank is solely for synthetic cortical-rate inference.
"""),
            code(ENV_CELL),
            code(r"""
from sleep_sbi.route3_7d_experiment import *

print("Preregistration SHA-256:", verify_preregistration())
gate = json.loads((PREFLIGHT_ROOT / "engineering_gate.json").read_text())
assert gate["pass"]
bank = run_training_bank_4096(max_workers=6)
display(pd.DataFrame([bank["manifest"]]))
"""),
            md(r"""
## Independent reload and fixed-insertion checks

No pickle/object arrays are allowed. The first seven columns are the free
parameters; the eighth simulator column must equal the preregistered fixed value
for all 4,096 rows. Scaling is fitted on the training split only.
"""),
            code(r"""
with np.load(bank["bank_path"], allow_pickle=False) as data:
    theta7 = data["theta"]; theta8 = data["theta_full_8d"]; x = data["x"]
    success = data["success"]; runtime = data["runtime_s"]
    assert theta7.shape == (4096, 7)
    assert theta8.shape == (4096, 8)
    assert x.shape == (4096, 14)
    assert np.all(theta8[:, :7] == theta7)
    assert np.all(theta8[:, 7] == fixed_c_ctx2th())
    assert np.isfinite(x[success]).all()
with np.load(bank["split_path"], allow_pickle=False) as split:
    assert np.intersect1d(split["training_indices"], split["validation_indices"]).size == 0
    print("train/validation:", len(split["training_indices"]), len(split["validation_indices"]))
"""),
            md(r"""
## Global-collision audit

Theta distance is normalized by the seven prior widths. Feature distance uses
the training-only robust scaler. The seed-noise floor is inherited from
Notebook 15, not tuned on this bank. A collision is a distant theta pair whose
14D distance falls within that noise floor.
"""),
            code(r"""
collision = analyze_bank_collisions(bank["bank_path"], bank["split_path"])
display(pd.DataFrame([collision]))

fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
axes[0].hist(runtime, bins=40, color="#356a8a")
axes[0].set(xlabel="Simulator runtime (s)", ylabel="Rows", title="Per-row runtime")
axes[1].bar(["valid", "failed"], [success.sum(), (~success).sum()], color=["#4c8a55", "#b74d4d"])
axes[1].set(ylabel="Scheduled rows", title="4,096-bank validity")
figure_dir = BANK_ROOT / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(figure_dir / "bank_runtime_and_validity.png", dpi=180)
fig.savefig(figure_dir / "bank_runtime_and_validity.svg")
plt.show()
"""),
        ],
    )


def build17() -> Path:
    return write(
        "17_Route3_7D_SNPE_Ensemble.ipynb",
        [
            md(r"""
# Preregistered Route-3 7D Single-round NPE Ensemble

Three independently initialized MAF density estimators use the same independent
4,096-row bank, fixed train/validation split, and training-only scaler.
Proposal is the frozen seven-dimensional prior. The equal-weight mixture is
transparent; there are no sequential rounds or held-out-driven adjustments.
"""),
            code(ENV_CELL),
            code(r"""
from sleep_sbi.route3_7d_experiment import *
from sleep_sbi.route3_7d_training import *
from sleep_sbi.route3_7d_validation import freeze_heldout_criteria

print("Preregistration SHA-256:", verify_preregistration())
training = train_ensemble_7d()
display(pd.DataFrame(training["summary"]["members"]))
"""),
            md(r"""
## Training histories and validation-only diagnostics

Validation loss, boundary accumulation, finite/support rate, and ensemble
disagreement are engineering diagnostics. They do not determine the formal
scientific verdict, which remains locked until independent held-out validation.
"""),
            code(r"""
checks = validation_member_checks()
display(checks.groupby("member").mean(numeric_only=True))

fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
for seed in model_seeds():
    history = pd.read_csv(TRAINING_ROOT / f"member_{seed}" / "training_history.csv")
    ax.plot(history.epoch, history.validation_loss, label=f"seed {seed}")
ax.set(xlabel="Epoch", ylabel="Validation negative log probability", title="Independent 7D NPE validation histories")
ax.legend()
figure_dir = TRAINING_ROOT / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(figure_dir / "validation_loss_curves.png", dpi=180)
fig.savefig(figure_dir / "validation_loss_curves.svg")
plt.show()
"""),
            md(r"""
## Held-out criteria lock

The numerical criteria were already part of the immutable preregistration.
This separate artifact is copied and hashed before held-out theta, observations,
posterior samples, recovery, or PPC results are opened.
"""),
            code(r"""
criteria = freeze_heldout_criteria()
print("Held-out criteria SHA-256:", criteria["sha256"])
display(pd.DataFrame([criteria["payload"]["criteria"]["formal_go"]]).T)
"""),
        ],
    )


def build18() -> Path:
    return write(
        "18_Route3_7D_Heldout_Recovery_Coverage_PPC.ipynb",
        [
            md(r"""
# Independent Route-3 7D Held-out Recovery, Coverage, SBC, and PPC

This notebook applies the immutable criteria to 256 new held-out seven-parameter
theta values and new simulator seeds. Every case receives 4,096 equal-mixture
posterior samples. Synthetic PPC uses 32 fixed-random and 32 disjoint
worst-recovery cases. `c_ctx2th` is fixed, not inferred, and receives no
posterior or recovery metric.

Even a GO would concern only conditional recovery in synthetic cortical-rate
space. Real-EEG inference remains NO-GO without an independently validated
cortical-source-to-Fpz-Cz measurement model.
"""),
            code(ENV_CELL),
            code(r"""
from sleep_sbi.route3_7d_experiment import *
from sleep_sbi.route3_7d_validation import *

print("Preregistration SHA-256:", verify_preregistration())
print("Held-out criteria SHA-256:", verify_heldout_criteria())
heldout = run_heldout_256(max_workers=6)
display(pd.DataFrame([heldout["manifest"]]))
independence = audit_independence_from_8d()
display(pd.DataFrame([independence["summary"]]))
"""),
            md(r"""
## Posterior sampling and parameter recovery

Errors are normalized by prior width. The prior-median baseline predicts 0.5
for every normalized parameter. Positive improvement means the ensemble median
beats this baseline. Rank correlation measures ordering, not calibration.
"""),
            code(r"""
posterior = sample_heldout_posteriors(heldout["path"])
recovery = analyze_recovery_coverage(posterior["path"])
display(recovery["recovery"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
table = recovery["recovery"]
axes[0].bar(table.parameter, table.prior_median_mae, label="Prior median", color="#999999")
axes[0].bar(table.parameter, table.posterior_median_mae, label="Posterior median", color="#356a8a")
axes[0].tick_params(axis="x", rotation=45)
axes[0].set(ylabel="Mean normalized absolute error", title="Held-out recovery (N=256)")
axes[0].legend()
axes[1].bar(table.parameter, table.rank_correlation, color="#4c8a55")
axes[1].tick_params(axis="x", rotation=45)
axes[1].set(ylabel="Spearman rank correlation", title="True vs posterior median")
figure_dir = HELDOUT_ROOT / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(figure_dir / "recovery_summary.png", dpi=180)
fig.savefig(figure_dir / "recovery_summary.svg")
plt.show()
"""),
            md(r"""
## Coverage and SBC

Coverage is accompanied by Wilson 95% intervals for 256 cases. Formal
compatibility requires nominal 80% and 90% levels to lie inside the interval.
SBC histograms are interpreted jointly with error, contraction, and coverage;
a single p-value is never used as the verdict.
"""),
            code(r"""
ensemble_coverage = recovery["coverage"].query("estimator == 'ensemble'")
display(ensemble_coverage)
display(recovery["sbc"].query("estimator == 'ensemble'"))

fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
for level, frame in ensemble_coverage.groupby("nominal_level"):
    axes[0].plot(frame.parameter, frame.empirical_coverage, "o-", label=f"{level:.0%}")
    axes[0].axhline(level, color="black", ls=":", alpha=.35)
axes[0].tick_params(axis="x", rotation=45)
axes[0].set(ylabel="Empirical coverage", title="Held-out credible-interval coverage")
axes[0].legend()
sbc = recovery["sbc"].query("estimator == 'ensemble'")
image = axes[1].imshow(sbc[[f"bin_{i}" for i in range(10)]], aspect="auto", cmap="viridis")
axes[1].set_yticks(range(7), sbc.parameter)
axes[1].set(xlabel="Normalized SBC rank bin", title="Ensemble SBC ranks (N=256)")
fig.colorbar(image, ax=axes[1], label="Cases")
fig.savefig(figure_dir / "coverage_and_sbc.png", dpi=180)
fig.savefig(figure_dir / "coverage_and_sbc.svg")
plt.show()
"""),
            md(r"""
## Preregistered 64-case synthetic PPC

For each selected held-out observation, 16 new posterior theta and 16 new prior
theta are simulated with independent seeds. Errors use the training-only
feature scale. PPC improvement does not override undercoverage or false
posterior contraction.
"""),
            code(r"""
ppc = run_ppc_64(recovery, heldout["path"], max_workers=6)
display(ppc["metrics"])
display(pd.DataFrame([ppc["summary"]]))

fig, (ax, label_ax) = plt.subplots(
    1, 2, figsize=(15, 7), constrained_layout=True,
    gridspec_kw={"width_ratios": [3, 2]},
)
ax.scatter(
    ppc["metrics"].prior_predictive_median_scaled_abs_error,
    ppc["metrics"].posterior_predictive_median_scaled_abs_error,
    color="#356a8a",
)
limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, limit], [0, limit], "k--", label="equal error")
for index, row in ppc["metrics"].reset_index(drop=True).iterrows():
    ax.annotate(
        str(index + 1),
        (row.prior_predictive_median_scaled_abs_error, row.posterior_predictive_median_scaled_abs_error),
        fontsize=9, xytext=(4, 3), textcoords="offset points",
    )
ax.set(xlabel="Prior-predictive median scaled error", ylabel="Posterior-predictive median scaled error", title="Synthetic PPC: 64 cases x 16 draws")
ax.legend()
label_ax.axis("off")
label_ax.set_title("Feature index", loc="left")
label_ax.text(
    0, 1,
    "\n".join(f"{index + 1:>2}. {name}" for index, name in enumerate(ppc["metrics"].feature)),
    va="top", ha="left", family="monospace", fontsize=10,
)
fig.savefig(figure_dir / "synthetic_ppc.png", dpi=180)
fig.savefig(figure_dir / "synthetic_ppc.svg")
plt.show()
"""),
            md(r"""
## Immutable automatic decision

No threshold, feature, fixed value, network, sample, or held-out case is changed
after these results. The output is exactly one of GO, CONDITIONAL GO, or NO-GO.
The decision never unlocks real-EEG inference.
"""),
            code(r"""
decision = evaluate_decision(recovery, ppc)
display(pd.DataFrame(decision["checks"].items(), columns=["check", "value"]))
print("FINAL ROUTE-3 7D DECISION:", decision["decision"])
assert decision["decision"] in {"GO", "CONDITIONAL GO", "NO-GO"}
assert not decision["real_eeg_inference_unlocked"]
"""),
        ],
    )


def main() -> None:
    paths = [build15(), build16(), build17(), build18()]
    for path in paths:
        notebook = nbf.read(path, as_version=4)
        nbf.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, f"{path.name}:cell", "exec")
        print(path)


if __name__ == "__main__":
    main()
