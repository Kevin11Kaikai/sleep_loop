"""Figures, artifact verification, and terminal reports for the 7D rescue."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .route3_7d_experiment import PARAMETER_NAMES_7D, atomic_json, sha256_file
from .route3_7d_rescue import (
    DEVELOPMENT_ROOT,
    DIAGNOSIS_ROOT,
    FINAL_ROOT,
    HTML_ROOT,
    ORIGINAL_ROOT,
    RESCUE_ROOT,
    TRAINING_ROOT,
    PROJECT_ROOT,
    read_rescue_preregistration,
    verify_rescue_preregistration,
)
from .route3_7d_rescue_training import verify_primary_pipeline_lock


FIGURE_ROOT = RESCUE_ROOT / "figures"
FINAL_MD = PROJECT_ROOT / "ROUTE3_7D_RESCUE_FINAL_DECISION.md"
FINAL_JSON = PROJECT_ROOT / "ROUTE3_7D_RESCUE_FINAL_DECISION.json"


def environment_report() -> dict[str, Any]:
    import torch
    import sbi

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_cuda_available": torch.cuda.is_available(),
        "versions": {
            name: metadata.version(name)
            for name in (
                "neurolib",
                "sbi",
                "torch",
                "numpy",
                "scipy",
                "pandas",
                "matplotlib",
                "nbformat",
                "nbclient",
                "jupyter",
            )
        },
        "sbi_version_runtime": sbi.__version__,
        "rescue_preregistration_hash": verify_rescue_preregistration(),
    }
    atomic_json(RESCUE_ROOT / "environment_report.json", report)
    return report


def _save(fig: plt.Figure, name: str) -> tuple[Path, Path]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    png = FIGURE_ROOT / f"{name}.png"
    svg = FIGURE_ROOT / f"{name}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def plot_diagnosis() -> tuple[Path, Path]:
    coverage = pd.read_csv(DIAGNOSIS_ROOT / "original_coverage_curve.csv")
    decomposition = pd.read_csv(DIAGNOSIS_ROOT / "bias_width_boundary_audit.csv")
    ensemble = coverage[coverage.estimator == "ensemble"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for parameter in PARAMETER_NAMES_7D:
        sub = ensemble[ensemble.parameter == parameter]
        axes[0].plot(
            sub.nominal_level,
            sub.empirical_coverage,
            marker="o",
            linewidth=1.5,
            label=parameter,
        )
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal")
    axes[0].set(
        xlabel="Nominal central credible level",
        ylabel="Empirical coverage (N=256)",
        title="Original ensemble undercoverage",
        xlim=(0.08, 0.92),
        ylim=(0.02, 1.0),
    )
    axes[0].legend(fontsize=7, ncol=2)
    x = np.arange(7)
    axes[1].bar(
        x - 0.18,
        decomposition.coverage_90_boundary,
        width=0.36,
        label="within 0.1 of prior boundary",
        color="#C94C4C",
    )
    axes[1].bar(
        x + 0.18,
        decomposition.coverage_90_interior,
        width=0.36,
        label="interior",
        color="#3A7D8C",
    )
    axes[1].axhline(0.9, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, PARAMETER_NAMES_7D, rotation=35, ha="right")
    axes[1].set(
        ylabel="Empirical 90% coverage",
        title="Boundary-localized calibration failure",
        ylim=(0, 1.02),
    )
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "19_coverage_failure_diagnosis")


def plot_training_and_development() -> tuple[Path, Path]:
    candidate = pd.read_csv(DEVELOPMENT_ROOT / "candidate_pipeline_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for config in ("maf64_t5", "maf128_t8"):
        histories = []
        for path in sorted((TRAINING_ROOT / config).glob("member_*/training_history.csv")):
            frame = pd.read_csv(path)
            histories.append(frame.validation_loss.to_numpy())
            axes[0].plot(
                frame.epoch,
                frame.validation_loss,
                alpha=0.65,
                linewidth=1,
                label=f"{config}:{path.parent.name.replace('member_', '')}",
            )
    axes[0].set(
        xlabel="Epoch",
        ylabel="Validation negative log probability",
        title="Five independent members per bounded configuration",
    )
    axes[0].legend(fontsize=6, ncol=2)
    labels = candidate.config_id + "\n" + candidate.method
    colors = ["#4B78A8" if value == "raw" else "#E08B3E" for value in candidate.method]
    axes[1].bar(np.arange(len(candidate)), candidate.aggregate_coverage_error, color=colors)
    axes[1].set_xticks(np.arange(len(candidate)), labels, rotation=20, ha="right")
    axes[1].set(
        ylabel="Mean |empirical - nominal coverage|",
        title="Development-only pipeline comparison (N=512)",
    )
    fig.tight_layout()
    return _save(fig, "21_training_and_development_selection")


def plot_final_validation() -> list[tuple[Path, Path]]:
    selection = verify_primary_pipeline_lock()
    method = selection["selected_method"]
    raw_coverage = pd.read_csv(
        FINAL_ROOT / "metrics" / "raw" / "coverage_curve.csv"
    )
    primary_coverage = pd.read_csv(
        FINAL_ROOT / "metrics" / "primary" / "coverage_curve.csv"
    )
    raw = raw_coverage[raw_coverage.estimator == "raw"]
    primary = primary_coverage[primary_coverage.estimator == method]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for axis, frame, title in (
        (axes[0], raw, "Raw selected ensemble"),
        (axes[1], primary, f"Primary: {method}"),
    ):
        for parameter in PARAMETER_NAMES_7D:
            sub = frame[frame.parameter == parameter]
            axis.plot(
                sub.nominal_level,
                sub.empirical_coverage,
                marker="o",
                linewidth=1.4,
                label=parameter,
            )
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set(
            xlabel="Nominal central credible level",
            title=f"{title} (fresh N=1024)",
            xlim=(0.08, 0.92),
            ylim=(0.02, 1.0),
        )
    axes[0].set_ylabel("Empirical coverage")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    outputs = [_save(fig, "22_fresh_final_coverage_curves")]

    recovery = pd.read_csv(
        FINAL_ROOT / "metrics" / "primary" / "parameter_recovery.csv"
    )
    coverage = primary[primary.nominal_level.isin([0.8, 0.9])]
    c80 = coverage[np.isclose(coverage.nominal_level, 0.8)].set_index("parameter")
    c90 = coverage[np.isclose(coverage.nominal_level, 0.9)].set_index("parameter")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(7)
    axes[0].bar(
        x - 0.2,
        recovery.prior_median_mae,
        width=0.4,
        label="prior median",
        color="#A9A9A9",
    )
    axes[0].bar(
        x + 0.2,
        recovery.posterior_median_mae,
        width=0.4,
        label="posterior median",
        color="#3A7D8C",
    )
    axes[0].set_xticks(x, PARAMETER_NAMES_7D, rotation=35, ha="right")
    axes[0].set(
        ylabel="Normalized absolute error",
        title="Fresh held-out point recovery",
    )
    axes[0].legend()
    ordered_names = list(PARAMETER_NAMES_7D)
    axes[1].bar(
        x - 0.18,
        c80.loc[ordered_names].empirical_coverage,
        0.36,
        label="80%",
    )
    axes[1].bar(
        x + 0.18,
        c90.loc[ordered_names].empirical_coverage,
        0.36,
        label="90%",
    )
    axes[1].axhline(0.8, color="#4B78A8", linestyle="--", linewidth=1)
    axes[1].axhline(0.9, color="#E08B3E", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, PARAMETER_NAMES_7D, rotation=35, ha="right")
    axes[1].set(
        ylabel="Empirical coverage",
        title="Required coverage gates (fresh N=1024)",
        ylim=(0, 1.02),
    )
    axes[1].legend()
    fig.tight_layout()
    outputs.append(_save(fig, "22_recovery_and_required_coverage"))

    sbc = pd.read_csv(FINAL_ROOT / "metrics" / "primary" / "sbc_ranks.csv")
    sbc = sbc[sbc.estimator == method]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True, sharey=True)
    for axis, (_, row) in zip(axes.flat, sbc.iterrows()):
        counts = np.asarray([row[f"bin_{index}"] for index in range(10)])
        axis.bar(np.arange(10), counts, color="#4B78A8")
        axis.axhline(102.4, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(f"{row.parameter}; p={row.chi_square_p_value:.3g}", fontsize=9)
    axes.flat[-1].axis("off")
    for axis in axes[-1, :]:
        axis.set_xlabel("Normalized rank bin")
    for axis in axes[:, 0]:
        axis.set_ylabel("Count")
    fig.suptitle(f"Fresh final SBC ranks: {method} (N=1024)")
    fig.tight_layout()
    outputs.append(_save(fig, "22_fresh_final_sbc"))

    ppc = pd.read_csv(FINAL_ROOT / "ppc" / "ppc_feature_metrics.csv")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    order = np.argsort(ppc.improvement_fraction)
    ax.barh(
        np.arange(len(ppc)),
        ppc.improvement_fraction.to_numpy()[order],
        color=np.where(ppc.posterior_better.to_numpy()[order], "#3A7D8C", "#C94C4C"),
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(np.arange(len(ppc)), ppc.feature.to_numpy()[order], fontsize=8)
    ax.set(
        xlabel="Posterior vs prior predictive scaled-error improvement",
        title="Fresh 14D synthetic PPC: 32 fixed-random + 32 worst cases",
    )
    fig.tight_layout()
    outputs.append(_save(fig, "22_fresh_final_ppc"))
    return outputs


def validate_rescue_artifacts() -> dict[str, Any]:
    json_files = sorted(RESCUE_ROOT.rglob("*.json"))
    csv_files = sorted(RESCUE_ROOT.rglob("*.csv"))
    npz_files = sorted(RESCUE_ROOT.rglob("*.npz"))
    pt_files = sorted(RESCUE_ROOT.rglob("*.pt"))
    errors: list[str] = []
    object_arrays: list[str] = []
    for path in json_files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    for path in csv_files:
        try:
            pd.read_csv(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    for path in npz_files:
        try:
            with np.load(path, allow_pickle=False) as data:
                for key in data.files:
                    if data[key].dtype == object:
                        object_arrays.append(f"{path}:{key}")
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    for path in pt_files:
        if path.stat().st_size == 0:
            errors.append(f"{path}: empty checkpoint")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "json_count": len(json_files),
        "csv_count": len(csv_files),
        "npz_count": len(npz_files),
        "pt_count": len(pt_files),
        "object_array_count": len(object_arrays),
        "object_arrays": object_arrays,
        "errors": errors,
        "pass": not errors and not object_arrays,
        "rescue_preregistration_hash": verify_rescue_preregistration(),
    }
    atomic_json(RESCUE_ROOT / "artifact_reload_validation.json", report)
    return report


def write_final_reports() -> dict[str, Any]:
    decision_path = FINAL_ROOT / "ROUTE3_7D_RESCUE_FINAL_DECISION.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    selection = verify_primary_pipeline_lock()
    original_decision = json.loads(
        (
            ORIGINAL_ROOT
            / "heldout_validation"
            / "formal_route3_7d_decision.json"
        ).read_text(encoding="utf-8")
    )
    original_ppc = json.loads(
        (
            ORIGINAL_ROOT
            / "heldout_validation"
            / "ppc"
            / "ppc_summary.json"
        ).read_text(encoding="utf-8")
    )
    candidate = pd.read_csv(DEVELOPMENT_ROOT / "candidate_pipeline_metrics.csv")
    selected_dev = candidate[
        (candidate.config_id == selection["selected_config_id"])
        & (candidate.method == selection["selected_method"])
    ].iloc[0]
    recovery = pd.read_csv(
        FINAL_ROOT / "metrics" / "primary" / "parameter_recovery.csv"
    )
    coverage = pd.read_csv(
        FINAL_ROOT / "metrics" / "primary" / "coverage_curve.csv"
    )
    primary = coverage[coverage.estimator == decision["applies_to"]]
    c80 = primary[np.isclose(primary.nominal_level, 0.8)].set_index("parameter")
    c90 = primary[np.isclose(primary.nominal_level, 0.9)].set_index("parameter")
    parameter_table = recovery[
        [
            "parameter",
            "prior_median_mae",
            "posterior_median_mae",
            "mae_improvement_fraction",
            "rank_correlation",
            "median_90_ci_width",
        ]
    ].copy()
    parameter_table["coverage_80"] = [
        c80.loc[name].empirical_coverage for name in parameter_table.parameter
    ]
    parameter_table["coverage_90"] = [
        c90.loc[name].empirical_coverage for name in parameter_table.parameter
    ]
    parameter_table["coverage_80_90_compatible"] = [
        bool(
            c80.loc[name].nominal_inside_wilson
            and c90.loc[name].nominal_inside_wilson
        )
        for name in parameter_table.parameter
    ]
    parameter_table["contracted"] = parameter_table.median_90_ci_width < 0.9
    parameter_table.to_csv(FINAL_ROOT / "final_parameter_verdict_table.csv", index=False)

    ppc = json.loads((FINAL_ROOT / "ppc" / "ppc_summary.json").read_text(encoding="utf-8"))
    raw_recovery = pd.read_csv(
        FINAL_ROOT / "metrics" / "raw" / "parameter_recovery.csv"
    )
    raw_coverage = pd.read_csv(
        FINAL_ROOT / "metrics" / "raw" / "coverage_curve.csv"
    )
    raw_required = raw_coverage[
        (raw_coverage.estimator == "raw")
        & raw_coverage.nominal_level.isin([0.8, 0.9])
    ]
    raw_compatible = raw_required.groupby("parameter").nominal_inside_wilson.all()
    raw_case = pd.read_csv(FINAL_ROOT / "metrics" / "raw" / "case_recovery.csv")
    raw_improvement = float(
        1
        - raw_case.mean_parameter_median_abs_error.mean()
        / raw_case.mean_parameter_prior_median_abs_error.mean()
    )
    comparison = pd.DataFrame(
        [
            {
                "stage": "Original Route-3",
                "cases": 256,
                "method": "raw 3-member MAF64x5",
                "coverage_compatible_parameters": original_decision["checks"][
                    "coverage_compatible_parameter_count"
                ],
                "recovery_improvement_fraction": original_decision["checks"][
                    "overall_recovery_improvement_fraction"
                ],
                "ppc_improvement_fraction": original_ppc["overall_improvement_fraction"],
                "verdict": original_decision["decision"],
            },
            {
                "stage": "Best development rescue",
                "cases": 512,
                "method": f"{selection['selected_config_id']}:{selection['selected_method']}",
                "coverage_compatible_parameters": int(
                    selected_dev.coverage_compatible_parameter_count
                ),
                "recovery_improvement_fraction": float(
                    selected_dev.overall_recovery_improvement_fraction
                ),
                "ppc_improvement_fraction": np.nan,
                "verdict": "selection only; not final",
            },
            {
                "stage": "Fresh independent final (raw)",
                "cases": 1024,
                "method": f"{selection['selected_config_id']}:raw",
                "coverage_compatible_parameters": int(raw_compatible.sum()),
                "recovery_improvement_fraction": raw_improvement,
                "ppc_improvement_fraction": np.nan,
                "verdict": "diagnostic raw result; no separate PPC",
            },
            {
                "stage": "Fresh independent final",
                "cases": 1024,
                "method": decision["applies_to"],
                "coverage_compatible_parameters": decision[
                    "parameters_passing_all_80_90_coverage_requirements"
                ],
                "recovery_improvement_fraction": decision["checks"][
                    "overall_recovery_improvement_fraction"
                ],
                "ppc_improvement_fraction": ppc["overall_improvement_fraction"],
                "verdict": decision["verdict"],
            },
        ]
    )
    comparison.to_csv(FINAL_ROOT / "original_development_final_comparison.csv", index=False)
    blocker = (
        ", ".join(decision["blockers"]) if decision["blockers"] else "none"
    )
    md = [
        f"# {decision['verdict']}",
        "",
        f"1. Verdict applies to: **{decision['applies_to']}**.",
        f"2. Fresh independent final validation: **{decision['fresh_final_cases']} cases**.",
        "3. Parameters passing every 80% and 90% coverage requirement: "
        f"**{decision['parameters_passing_all_80_90_coverage_requirements']}/7**.",
        "4. Parameters passing simultaneous coverage plus contraction: "
        f"**{decision['parameters_passing_coverage_plus_contraction']}/7**.",
        "5. Posterior-median recovery improvement over prior median: "
        f"**{decision['checks']['overall_recovery_improvement_fraction']:.1%}**.",
        "6. Synthetic PPC improvement over prior predictive: "
        f"**{decision['checks']['overall_ppc_improvement_fraction']:.1%}**.",
        "7. Every original numerical hard gate passed: "
        f"**{decision['numeric_formal_gates_pass']}**.",
        f"8. Exact blocker if not Formal GO: **{blocker}**.",
        "",
        "## Original, development, and final comparison",
        "",
        comparison.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Fresh parameter evidence",
        "",
        parameter_table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Raw versus calibrated final evidence",
        "",
        (
            f"The selected raw ensemble passed both required coverage levels for "
            f"**{int(raw_compatible.sum())}/7** parameters and improved median "
            f"recovery by **{raw_improvement:.1%}**. The primary calibrated result "
            f"passed **{decision['parameters_passing_all_80_90_coverage_requirements']}/7** "
            "coverage requirements. Raw and calibrated coverage/SBC tables are "
            "stored separately; PPC was run only for the preregistered frozen primary."
        ),
        "",
        "## Scientific interpretation",
        "",
        (
            "The experiment concerns recoverability in a synthetic cortical population "
            "firing-rate observable space. It does not validate a cortical-source-to-"
            "Fpz-Cz measurement model and does not authorize real-EEG inference."
        ),
        "",
        (
            "Scientifically established: the complete statement is limited to the "
            "frozen 7D prior, fixed c_ctx2th, 14D cortical-rate schema, selected "
            "pipeline, and fresh independent simulator seeds reported here."
        ),
        "",
        (
            "Unsupported: real-subject parameters, a real EEG digital twin, "
            "thalamic-state observability from one scalp channel, or mechanism truth."
        ),
        "",
        (
            "Readiness: synthetic-only inference"
            + (
                " remains blocked by calibration."
                if decision["verdict"] == "NO-GO"
                else "; any extension beyond this synthetic scope remains blocked."
            )
        ),
        "",
        (
            "Single most important next action: "
            + (
                "resolve the remaining frozen coverage/calibration blocker with a new "
                "preregistered method and another untouched final set."
                if decision["verdict"] != "FORMAL GO"
                else "replicate the synthetic result before changing schema or prior."
            )
        ),
        "",
    ]
    FINAL_MD.write_text("\n".join(md), encoding="utf-8")
    final_payload = {
        **decision,
        "parameter_table": parameter_table.to_dict(orient="records"),
        "fresh_raw_summary": {
            "coverage_compatible_parameters": int(raw_compatible.sum()),
            "overall_recovery_improvement_fraction": raw_improvement,
            "ppc_run": False,
        },
        "comparison": comparison.replace({np.nan: None}).to_dict(orient="records"),
        "final_report_sha256": sha256_file(FINAL_MD),
    }
    atomic_json(FINAL_JSON, final_payload)
    atomic_json(FINAL_ROOT / "ROUTE3_7D_RESCUE_FINAL_DECISION.json", final_payload)
    return {
        "markdown": FINAL_MD,
        "json": FINAL_JSON,
        "decision": final_payload,
        "parameter_table": parameter_table,
        "comparison": comparison,
    }
