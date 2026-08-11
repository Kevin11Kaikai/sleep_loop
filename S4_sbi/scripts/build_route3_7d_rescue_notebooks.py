"""Build reporting notebooks 19--23 around the checkpointed rescue pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "S4_sbi" / "notebooks"


KERNEL = {
    "kernelspec": {
        "display_name": "neurolib",
        "language": "python",
        "name": "neurolib",
    },
    "language_info": {"name": "python", "version": "3.10.20"},
}

INIT = """\
from pathlib import Path
import json, os, sys
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown

PROJECT_ROOT = Path.cwd().resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "S4_sbi").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "S4_sbi").exists():
    raise RuntimeError("sleep_loop project root could not be resolved")
sys.path.insert(0, str(PROJECT_ROOT / "S4_sbi" / "src"))
RESULTS = PROJECT_ROOT / "S4_sbi" / "results" / "route3_7d_rescue"
print("CONDA_DEFAULT_ENV=", os.environ.get("CONDA_DEFAULT_ENV"))
print("sys.executable=", sys.executable)
print("sys.prefix=", sys.prefix)
print("PROJECT_ROOT=", PROJECT_ROOT)
assert "neurolib" in sys.executable.lower()
"""


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def notebook(cells):
    nb = nbf.v4.new_notebook(cells=cells, metadata=KERNEL)
    nbf.validate(nb)
    return nb


def write(name: str, cells) -> None:
    path = NOTEBOOKS / name
    nbf.write(notebook(cells), path)
    print(path)


def build() -> None:
    write(
        "19_Route3_7D_Coverage_Failure_Diagnosis.ipynb",
        [
            markdown(
                """
# Route-3 7D Coverage Failure Diagnosis

This notebook independently audits the already-inspected Notebook-18 evidence.
It does not tune against a final test set and does not alter Notebooks 15--18.
The scientific target is posterior calibration in the synthetic cortical-rate
observable space, not real EEG inference.
"""
            ),
            code(INIT),
            markdown("## Frozen evidence and independent recomputation"),
            code(
                """
from sleep_sbi.route3_7d_rescue import diagnose_original_failure
diagnosis = diagnose_original_failure()
display(pd.DataFrame([diagnosis["summary"]]).T.rename(columns={0: "value"}))
assert diagnosis["summary"]["verified_implementation_bug_found"] is False
assert diagnosis["summary"]["posterior_mc_halves_stable"] is True
"""
            ),
            markdown(
                """
## Per-network and ensemble coverage

Each curve uses central credible intervals. Similar undercoverage in the three
members and the equal mixture argues against accidental pooling or narrowing.
"""
            ),
            code(
                """
coverage = diagnosis["coverage"]
display(coverage[coverage.nominal_level.isin([0.5, 0.8, 0.9])].round(4))
"""
            ),
            markdown(
                """
## Bias, posterior width, and prior boundaries

Coverage near a bounded prior can fail even when point bias averaged over the
whole prior is small. The red/blue contrast is diagnostic, not a license to
remove boundary cases.
"""
            ),
            code(
                """
from sleep_sbi.route3_7d_rescue_reporting import plot_diagnosis
png, svg = plot_diagnosis()
display(Image(filename=str(png)))
display(diagnosis["decomposition"].round(4))
"""
            ),
            markdown(
                """
## Diagnosis

No verified transform, scaler, mixture-count, posterior-pooling, interval, NaN,
or Monte Carlo bug explains the result. The rescue therefore remains a genuine
training/calibration experiment. The old 256 cases are development evidence only.
"""
            ),
        ],
    )
    write(
        "20_Route3_7D_Rescue_Preregistration.ipynb",
        [
            markdown(
                """
# Route-3 7D Rescue Preregistration

This notebook displays the machine-locked rescue contract created before any new
scientific simulation. Parameters, prior, fixed `c_ctx2th`, 14D features,
simulator settings, numerical gates, and verdict logic remain unchanged.
"""
            ),
            code(INIT),
            markdown("## Lock verification"),
            code(
                """
from sleep_sbi.route3_7d_rescue import (
    read_rescue_preregistration, verify_rescue_preregistration,
    assert_seed_and_theta_disjointness,
)
rescue_hash = verify_rescue_preregistration()
cfg = read_rescue_preregistration()
print("locked rescue SHA-256:", rescue_hash)
display(pd.DataFrame({
    "parameter": cfg["frozen_contract"]["parameter_order"],
    "lower": np.asarray(cfg["frozen_contract"]["prior_bounds"])[:, 0],
    "upper": np.asarray(cfg["frozen_contract"]["prior_bounds"])[:, 1],
}))
assert cfg["frozen_contract"]["feature_dimension"] == 14
"""
            ),
            markdown("## Allowed bounded rescue and selection rule"),
            code(
                """
display(pd.json_normalize(cfg["allowed_rescue_configurations"]))
display(pd.Series(cfg["training_policy"], name="value").to_frame())
display(Markdown("\\n".join(f"{i+1}. {v}" for i, v in enumerate(cfg["selection_rule"]))))
"""
            ),
            markdown(
                """
## Independence and claim boundary

The additional training, development, and untouched final sequences must be
disjoint from the old bank/test and from one another. A calibrated pipeline is
separately labelled and is not eligible for Formal GO under the original contract.
"""
            ),
            code(
                """
independence = assert_seed_and_theta_disjointness()
display(pd.Series(independence["seed_intersections"], name="intersection_count").to_frame())
display(pd.Series(independence["exact_theta_overlaps"], name="exact_overlap_count").to_frame())
assert independence["pass"]
assert cfg["calibration"]["formal_go_eligibility"] is False
"""
            ),
        ],
    )
    write(
        "21_Route3_7D_Rescue_Training.ipynb",
        [
            markdown(
                """
# Route-3 7D Rescue Training and Development Selection

The production simulations and NPE training were executed by resumable scripts.
This notebook validates their manifests, displays all bounded candidates, and
documents the development-only primary-pipeline selection. It never opens the
fresh final outcomes.
"""
            ),
            code(INIT),
            markdown("## Combined stochastic bank and group-safe split"),
            code(
                """
from sleep_sbi.route3_7d_rescue import (
    BANK_ROOT, load_rescue_training_data, verify_rescue_preregistration,
)
verify_rescue_preregistration()
bank_manifest = json.loads((BANK_ROOT / "combined_bank_manifest.json").read_text())
display(pd.Series(bank_manifest, name="value").to_frame())
training_data = load_rescue_training_data()
assert bank_manifest["rows"] == 8192
assert bank_manifest["unique_theta_groups"] == 6144
assert bank_manifest["same_theta_cross_split_leakage"] is False
"""
            ),
            markdown(
                """
## Five-member ensembles

Both preregistered architectures use identical data, group split, optimizer
policy, and five independent initializations. Only hidden width/flow depth differ.
"""
            ),
            code(
                """
from sleep_sbi.route3_7d_rescue import TRAINING_ROOT, DEVELOPMENT_ROOT
training = []
for path in sorted(TRAINING_ROOT.glob("*/training_summary.csv")):
    frame = pd.read_csv(path)
    training.append(frame)
display(pd.concat(training, ignore_index=True).round(5))
"""
            ),
            markdown("## Development-only comparison and frozen selection"),
            code(
                """
from sleep_sbi.route3_7d_rescue_training import verify_primary_pipeline_lock
selection = verify_primary_pipeline_lock()
candidates = pd.read_csv(DEVELOPMENT_ROOT / "candidate_pipeline_metrics.csv")
display(candidates.round(5))
display(pd.Series(selection, name="value").to_frame())
"""
            ),
            markdown(
                """
## Training and calibration evidence

Coverage error is evaluated only on the independent 512-case development set.
Rank recalibration, if selected, is monotone and preserves within-case marginal
ordering, but it is reported separately from raw SNPE.
"""
            ),
            code(
                """
from sleep_sbi.route3_7d_rescue_reporting import plot_training_and_development
png, svg = plot_training_and_development()
display(Image(filename=str(png)))
"""
            ),
        ],
    )
    write(
        "22_Route3_7D_Fresh_Heldout_Validation.ipynb",
        [
            markdown(
                """
# Route-3 7D Fresh Independent Held-out Validation

This notebook reads the single, frozen primary decision on 1,024 new theta/seed
cases. No configuration, calibrator, threshold, feature, or prior is changed
after opening this final set. Raw and calibrated evidence remain distinct.
"""
            ),
            code(INIT),
            markdown("## Final dataset and posterior integrity"),
            code(
                """
from sleep_sbi.route3_7d_rescue import FINAL_ROOT
manifest = json.loads((FINAL_ROOT / "dataset" / "manifest.json").read_text())
decision = json.loads((FINAL_ROOT / "ROUTE3_7D_RESCUE_FINAL_DECISION.json").read_text())
display(pd.Series(manifest, name="value").to_frame())
display(pd.Series(decision["checks"], name="value").to_frame())
assert manifest["scheduled"] == 1024
assert manifest["valid"] + manifest["failed"] == 1024
"""
            ),
            markdown("## Parameter recovery, required coverage, and SBC"),
            code(
                """
method = decision["applies_to"]
recovery = pd.read_csv(FINAL_ROOT / "metrics" / "primary" / "parameter_recovery.csv")
coverage = pd.read_csv(FINAL_ROOT / "metrics" / "primary" / "coverage_curve.csv")
sbc = pd.read_csv(FINAL_ROOT / "metrics" / "primary" / "sbc_ranks.csv")
display(recovery.round(5))
display(coverage[(coverage.estimator == method) & coverage.nominal_level.isin([0.5, 0.8, 0.9])].round(5))
display(sbc[sbc.estimator == method].round(5))
"""
            ),
            markdown(
                """
## Fresh validation figures

Calibration curves show raw and frozen-primary results. Recovery errors use
normalized prior coordinates. SBC bars are diagnostic alongside coverage,
contraction, bias, and sample size, not a single-p-value pass/fail device.
"""
            ),
            code(
                """
from sleep_sbi.route3_7d_rescue_reporting import plot_final_validation
figures = plot_final_validation()
for png, svg in figures:
    display(Image(filename=str(png)))
"""
            ),
            markdown("## Synthetic 14D posterior predictive check"),
            code(
                """
ppc = json.loads((FINAL_ROOT / "ppc" / "ppc_summary.json").read_text())
display(pd.Series(ppc, name="value").to_frame())
assert ppc["posterior_predictive_attempted"] == 1024
assert ppc["prior_predictive_attempted"] == 1024
"""
            ),
            markdown(
                """
## Frozen decision

The verdict below is generated by the original numerical logic. If the primary
method is calibrated, the preregistration caps scientific status below Formal GO.
The result never unlocks real Fpz-Cz inference.
"""
            ),
            code(
                """
display(Markdown(f"# {decision['verdict']}"))
display(pd.Series({
    "applies_to": decision["applies_to"],
    "fresh_cases": decision["fresh_final_cases"],
    "coverage_pass_7": decision["parameters_passing_all_80_90_coverage_requirements"],
    "coverage_plus_contraction_7": decision["parameters_passing_coverage_plus_contraction"],
    "blockers": ", ".join(decision["blockers"]) or "none",
}).to_frame("value"))
"""
            ),
        ],
    )
    write(
        "23_Route3_7D_Rescue_Handoff.ipynb",
        [
            markdown(
                """
# Route-3 7D Rescue Handoff

This is the terminal, artifact-derived handoff. It does not recompute or modify
scientific results and does not reinterpret cortical population firing rate as EEG.
"""
            ),
            code(INIT),
            markdown("## Environment and artifact reload"),
            code(
                """
from sleep_sbi.route3_7d_rescue_reporting import (
    environment_report, validate_rescue_artifacts, FINAL_MD, FINAL_JSON,
)
environment = environment_report()
reload_report = validate_rescue_artifacts()
display(pd.Series(environment, name="value").to_frame())
display(pd.Series(reload_report, name="value").to_frame())
assert reload_report["pass"]
"""
            ),
            markdown("## Machine decision and parameter table"),
            code(
                """
decision = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
parameter_table = pd.DataFrame(decision["parameter_table"])
display(Markdown(f"# {decision['verdict']}"))
display(parameter_table.round(5))
"""
            ),
            markdown("## Original, development, and fresh-final comparison"),
            code(
                """
display(pd.DataFrame(decision["comparison"]).round(5))
print("Final Markdown:", FINAL_MD)
print("Final JSON:", FINAL_JSON)
"""
            ),
            markdown(
                """
## Claim boundary

Whatever the verdict, this experiment addresses only seven-parameter recovery in
the frozen synthetic cortical-rate observable space conditional on fixed
`c_ctx2th`. The cortical-source-to-Fpz-Cz measurement-model blocker remains, so
real EEG parameter inference and real-subject digital-twin claims are unsupported.
"""
            ),
        ],
    )


if __name__ == "__main__":
    build()
