"""Publication figures, ablation tables, and terminal Figure-10 reports."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figure10_diagnostics import (
    DIAGNOSTICS_ROOT,
    GLOBAL_ROOT,
    LC2ST_ROOT,
    PPC_ROOT,
    STRUCTURE_ROOT,
)
from .figure10_protocol import (
    FINAL_SCALE,
    INTERMEDIATE_SCALE,
    PARAMETER_NAMES_7D,
    PARAMETER_NAMES_8D,
    PROJECT_ROOT,
    RESULTS_ROOT,
    atomic_json,
    atomic_text,
    read_preregistration,
    verify_preregistration,
)
from .figure10_training import TRAINING_ROOT, load_training_data
from .figure10_trace_reporting import (
    TRACE_ROOT,
    build_ppc_seed_robustness,
    build_trace_examples,
)


FIGURE_ROOT = RESULTS_ROOT / "figures"
REPORT_PATH = PROJECT_ROOT / "FIGURE10_8D_7D_FINAL_REPORT.md"
REPORT_JSON_PATH = PROJECT_ROOT / "FIGURE10_8D_7D_FINAL_REPORT.json"

COLORS = {
    "8d": "#1f6f8b",
    "7d": "#d97706",
    "individual": "#9ca3af",
    "prior": "#6b7280",
    "observed": "#111827",
}


def _save(fig: plt.Figure, stem: str) -> dict[str, str]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for suffix in ("png", "svg", "pdf"):
        path = FIGURE_ROOT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths[suffix] = path.relative_to(PROJECT_ROOT).as_posix()
    plt.close(fig)
    return paths


def plot_expected_coverage_and_sbc() -> dict[str, str]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for column, track in enumerate(("8d", "7d")):
        root = GLOBAL_ROOT / "powered_1024" / track / f"scale_{FINAL_SCALE}"
        curve = pd.read_csv(root / "joint_expected_coverage_curve.csv")
        ranks_path = root / "global_ranks_and_recovery.npz"
        for estimator, group in curve.groupby("estimator"):
            if estimator == "ensemble":
                axes[0, column].plot(
                    group.nominal,
                    group.empirical,
                    color=COLORS[track],
                    linewidth=2.5,
                    label="Ensemble",
                )
            else:
                axes[0, column].plot(
                    group.nominal,
                    group.empirical,
                    color=COLORS["individual"],
                    linewidth=0.8,
                    alpha=0.6,
                    label="Individual" if estimator == "member_1" else None,
                )
        n = int(curve[curve.estimator == "ensemble"].shape[0] - 1)
        del n
        cases = 1024
        epsilon = np.sqrt(np.log(2 / 0.05) / (2 * cases))
        grid = np.linspace(0, 1, 101)
        axes[0, column].fill_between(
            grid,
            np.maximum(0, grid - epsilon),
            np.minimum(1, grid + epsilon),
            color="#d1d5db",
            alpha=0.35,
            label="95% DKW band",
        )
        axes[0, column].plot(grid, grid, "--", color="black", linewidth=1)
        axes[0, column].set(
            xlabel="Nominal joint HPD rank",
            ylabel="Empirical CDF",
            title=f"{track.upper()} joint expected coverage (N=1024)",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axes[0, column].legend(frameon=False, fontsize=8)
        with np.load(ranks_path, allow_pickle=False) as data:
            marginal = np.asarray(data["marginal_ranks"], float)[:, 0]
            n_samples = int(data["posterior_samples_per_case"].item())
            names = [str(v) for v in data["parameter_names"]]
        for index, name in enumerate(names):
            ranks = np.sort((marginal[:, index] + 0.5) / (n_samples + 1))
            axes[1, column].plot(
                ranks,
                (np.arange(len(ranks)) + 1) / len(ranks),
                linewidth=1.2,
                label=name,
            )
        axes[1, column].fill_between(
            grid,
            np.maximum(0, grid - epsilon),
            np.minimum(1, grid + epsilon),
            color="#d1d5db",
            alpha=0.35,
        )
        axes[1, column].plot(grid, grid, "--", color="black", linewidth=1)
        axes[1, column].set(
            xlabel="Normalized marginal SBC rank",
            ylabel="Empirical CDF",
            title=f"{track.upper()} marginal SBC",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axes[1, column].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(
        "Powered global calibration; synthetic cortical-rate posterior",
        fontsize=13,
    )
    fig.tight_layout()
    return _save(fig, "figure10_expected_coverage_sbc_8d_7d")


def plot_lc2st() -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, track in zip(axes, ("8d", "7d")):
        summary = pd.read_csv(LC2ST_ROOT / track / "lc2st_summary.csv")
        x = np.arange(len(summary))
        axis.bar(
            x,
            summary.score_observed,
            color=[
                COLORS[track] if name == "ensemble" else COLORS["individual"]
                for name in summary.estimator
            ],
            label="Observed statistic",
        )
        axis.scatter(
            x,
            summary.null_q95,
            marker="_",
            s=450,
            linewidth=2.5,
            color="black",
            label="95% null threshold",
        )
        for index, row in summary.iterrows():
            axis.text(
                index,
                max(row.score_observed, row.null_q95) * 1.04 + 1e-5,
                f"p={row.p_value:.3g}",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=7,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(summary.estimator, rotation=45, ha="right")
        axis.set_title(f"{track.upper()} L-C2ST (20,000 calibration pairs)")
        axis.set_ylabel("Local classifier statistic")
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Frozen primary synthetic observation; null rejection is posterior mismatch"
    )
    fig.tight_layout()
    return _save(fig, "figure10_lc2st_8d_7d")


def plot_ppc() -> dict[str, str]:
    frame8 = pd.read_csv(PPC_ROOT / "8d" / "ppc_feature_metrics.csv")
    frame7 = pd.read_csv(PPC_ROOT / "7d" / "ppc_feature_metrics.csv")
    x = np.arange(len(frame8))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for axis, frame, track in zip(axes, (frame8, frame7), ("8d", "7d")):
        width = 0.38
        axis.bar(
            x - width / 2,
            frame.prior_scaled_abs_error,
            width,
            color=COLORS["prior"],
            alpha=0.65,
            label="Prior predictive",
        )
        axis.bar(
            x + width / 2,
            frame.posterior_scaled_abs_error,
            width,
            color=COLORS[track],
            label=f"{track.upper()} posterior predictive",
        )
        axis.set_ylabel("Median |error| / training IQR")
        axis.set_title(
            f"{track.upper()} 14D PPC, N=256 new simulations"
        )
        axis.legend(frameon=False)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(frame8.feature, rotation=55, ha="right")
    fig.suptitle(
        "Posterior predictive checks in frozen cortical-rate observation space"
    )
    fig.tight_layout()
    return _save(fig, "figure10_ppc_8d_7d")


def plot_ppc_traces() -> dict[str, str]:
    with np.load(
        TRACE_ROOT / "primary_cortical_rate_trace.npz", allow_pickle=False
    ) as primary:
        time = np.asarray(primary["time_s"], float)
        observed = np.asarray(primary["r_exc_hz"], float)
    traces = {}
    for track in ("8d", "7d"):
        with np.load(
            TRACE_ROOT / f"{track}_posterior_predictive_traces.npz",
            allow_pickle=False,
        ) as data:
            traces[track] = np.asarray(data["r_exc_hz"], float)
    fig, axes = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True)
    axes[0].plot(time, observed, color=COLORS["observed"], linewidth=0.7)
    axes[0].set_title("Frozen synthetic observation")
    for axis, track in zip(axes[1:], ("8d", "7d")):
        for index, values in enumerate(traces[track]):
            axis.plot(
                time,
                values,
                linewidth=0.55,
                alpha=0.75,
                color=COLORS[track],
                label=f"draw {index + 1}",
            )
        axis.set_title(f"{track.upper()} posterior-predictive replays")
        axis.legend(frameon=False, fontsize=8, ncol=4)
    for axis in axes:
        axis.set_ylabel("Cortical EXC rate (Hz)")
    axes[-1].set_xlabel("Time after 5-s warm-up (s)")
    fig.suptitle(
        "Synthetic cortical-rate traces; these are not measured or simulated scalp EEG"
    )
    fig.tight_layout()
    return _save(fig, "figure10_ppc_cortical_rate_traces")


def build_multivariate_ppc_metrics() -> dict[str, Any]:
    outputs = {}
    primary_path = (
        PPC_ROOT / "primary_observation" / "8d" / "primary_observation_8d.npz"
    )
    with np.load(primary_path, allow_pickle=False) as primary:
        observed = np.asarray(primary["x"][0], float)
    for track in ("8d", "7d"):
        data = load_training_data(track)
        values = {}
        for label in ("posterior_predictive", "prior_predictive"):
            path = (
                PPC_ROOT
                / track
                / label
                / f"{label}_{track}.npz"
            )
            with np.load(path, allow_pickle=False) as artifact:
                x = np.asarray(artifact["x"], float)
            scaled = (x - observed) / data.x_scale
            values[label] = np.linalg.norm(scaled, axis=1)
        frame = pd.DataFrame(
            {
                "distribution": np.repeat(
                    ["posterior_predictive", "prior_predictive"],
                    [len(values["posterior_predictive"]), len(values["prior_predictive"])],
                ),
                "training_iqr_scaled_euclidean_distance": np.concatenate(
                    [values["posterior_predictive"], values["prior_predictive"]]
                ),
            }
        )
        csv_path = PPC_ROOT / track / "ppc_multivariate_distances.csv"
        frame.to_csv(csv_path, index=False)
        posterior_median = float(
            np.median(values["posterior_predictive"])
        )
        prior_median = float(np.median(values["prior_predictive"]))
        summary = {
            "track": track,
            "metric": "Euclidean distance after train-only IQR scaling",
            "posterior_median": posterior_median,
            "prior_median": prior_median,
            "improvement_fraction": float(
                1 - posterior_median / max(prior_median, 1e-15)
            ),
            "csv": csv_path.relative_to(RESULTS_ROOT).as_posix(),
            "preregistration_hash": verify_preregistration(),
        }
        atomic_json(
            PPC_ROOT / track / "ppc_multivariate_summary.json", summary
        )
        outputs[track] = summary
    return outputs


def plot_posterior_structure() -> dict[str, str]:
    with np.load(
        STRUCTURE_ROOT / "8d" / "posterior_structure_samples.npz",
        allow_pickle=False,
    ) as data8:
        samples8 = np.asarray(data8["samples_unit"], float)
        names8 = [str(v) for v in data8["parameter_names"]]
    with np.load(
        STRUCTURE_ROOT / "7d" / "posterior_structure_samples.npz",
        allow_pickle=False,
    ) as data7:
        samples7 = np.asarray(data7["samples_unit"], float)
        names7 = [str(v) for v in data7["parameter_names"]]
    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(3, 4)
    axes = [fig.add_subplot(grid[index // 4, index % 4]) for index in range(8)]
    for index, (axis, name) in enumerate(zip(axes, names8)):
        axis.hist(
            samples8[:, index],
            bins=50,
            density=True,
            color=COLORS["8d"],
            alpha=0.55,
            label="8D",
        )
        if name in names7:
            axis.hist(
                samples7[:, names7.index(name)],
                bins=50,
                density=True,
                histtype="step",
                linewidth=1.6,
                color=COLORS["7d"],
                label="7D fixed",
            )
        axis.axhline(1, color=COLORS["prior"], linestyle="--", linewidth=1)
        axis.set(xlim=(0, 1), xlabel="Prior-scaled value", title=name)
        if index == 0:
            axis.legend(frameon=False, fontsize=8)
    coupling_axis = fig.add_subplot(grid[2, 0:2])
    coupling_axis.hexbin(
        samples8[:, names8.index("c_th2ctx")],
        samples8[:, names8.index("c_ctx2th")],
        gridsize=55,
        cmap="Blues",
        mincnt=1,
    )
    coupling_axis.set(
        xlabel="c_th2ctx (prior-scaled)",
        ylabel="c_ctx2th (prior-scaled)",
        title="8D bidirectional-coupling dependence",
    )
    conductance_axis = fig.add_subplot(grid[2, 2:4])
    conductance_axis.hexbin(
        samples8[:, names8.index("g_LK")],
        samples8[:, names8.index("g_h")],
        gridsize=55,
        cmap="Blues",
        mincnt=1,
    )
    conductance_axis.set(
        xlabel="g_LK (prior-scaled)",
        ylabel="g_h (prior-scaled)",
        title="8D conductance dependence",
    )
    fig.suptitle(
        "Primary synthetic-observation posterior structure; prior density is flat"
    )
    fig.tight_layout()
    return _save(fig, "figure10_posterior_structure_identifiability")


def build_scale_sensitivity() -> pd.DataFrame:
    rows = []
    for scale in (INTERMEDIATE_SCALE, FINAL_SCALE):
        for track in ("8d", "7d"):
            training = json.loads(
                (
                    TRAINING_ROOT
                    / track
                    / f"scale_{scale}"
                    / "ensemble_manifest.json"
                ).read_text(encoding="utf-8")
            )
            global_summary = json.loads(
                (
                    GLOBAL_ROOT
                    / "powered_1024"
                    / track
                    / f"scale_{scale}"
                    / "global_analysis_summary.json"
                ).read_text(encoding="utf-8")
            )
            rows.append(
                {
                    "track": track,
                    "scale": scale,
                    "members": len(training["member_seeds"]),
                    "best_validation_loss_mean": float(
                        np.mean(
                            [
                                member["best_validation_loss"]
                                for member in training["members"]
                            ]
                        )
                    ),
                    "joint_rank_ks_statistic": global_summary[
                        "ensemble_joint_ks_statistic"
                    ],
                    "joint_rank_ks_pvalue": global_summary[
                        "ensemble_joint_ks_pvalue"
                    ],
                    "clear_sbc_issues": global_summary[
                        "ensemble_clear_sbc_issue_count"
                    ],
                    "recovery_improvement_fraction": global_summary[
                        "ensemble_recovery_improvement_fraction"
                    ],
                    "median_normalized_posterior_sd": global_summary[
                        "ensemble_median_normalized_posterior_sd"
                    ],
                    "training_runtime_s": training["runtime_this_call_s"],
                    "status": "completed",
                }
            )
    for scale in (131072, 524288, 1000000, 3000000):
        for track in ("8d", "7d"):
            rows.append(
                {
                    "track": track,
                    "scale": scale,
                    "members": 0,
                    "status": "not run: preregistered hardware limitation",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        RESULTS_ROOT / "diagnostics" / "scale_sensitivity.csv", index=False
    )
    return frame


def plot_ablation() -> dict[str, str]:
    marginal8 = pd.read_csv(
        STRUCTURE_ROOT / "8d" / "posterior_marginals.csv"
    )
    marginal7 = pd.read_csv(
        STRUCTURE_ROOT / "7d" / "posterior_marginals.csv"
    )
    shared = marginal8.merge(
        marginal7, on="parameter", suffixes=("_8d", "_7d")
    )
    x = np.arange(len(shared))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(
        x - 0.18,
        shared.posterior_sd_over_prior_sd_8d,
        0.36,
        color=COLORS["8d"],
        label="8D",
    )
    axes[0].bar(
        x + 0.18,
        shared.posterior_sd_over_prior_sd_7d,
        0.36,
        color=COLORS["7d"],
        label="7D fixed",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(shared.parameter, rotation=45, ha="right")
    axes[0].set_ylabel("Posterior SD / prior SD")
    axes[0].set_title("Shared-parameter contraction")
    axes[0].legend(frameon=False)
    metrics = []
    for track in ("8d", "7d"):
        global_summary = json.loads(
            (
                GLOBAL_ROOT
                / "powered_1024"
                / track
                / f"scale_{FINAL_SCALE}"
                / "global_analysis_summary.json"
            ).read_text(encoding="utf-8")
        )
        ppc = json.loads(
            (PPC_ROOT / track / "ppc_summary.json").read_text(encoding="utf-8")
        )
        metrics.append(
            [
                global_summary["ensemble_joint_ks_statistic"],
                global_summary["ensemble_clear_sbc_issue_count"],
                ppc["overall_improvement_fraction"],
            ]
        )
    metric_array = np.asarray(metrics)
    metric_names = ["Joint KS D", "SBC issues", "PPC improvement"]
    for index, track in enumerate(("8d", "7d")):
        axes[1].plot(
            metric_names,
            metric_array[index],
            marker="o",
            linewidth=2,
            color=COLORS[track],
            label=track.upper(),
        )
    axes[1].set_title("Matched diagnostic comparison")
    axes[1].legend(frameon=False)
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle(
        "7D-vs-8D ablation: releasing c_ctx2th changes uncertainty and fit"
    )
    fig.tight_layout()
    return _save(fig, "figure10_7d_vs_8d_ablation")


def generate_all_figures() -> dict[str, Any]:
    trace_manifest = build_trace_examples()
    seed_robustness = build_ppc_seed_robustness()
    multivariate_ppc = build_multivariate_ppc_metrics()
    paths = {
        "expected_coverage_sbc": plot_expected_coverage_and_sbc(),
        "lc2st": plot_lc2st(),
        "ppc": plot_ppc(),
        "ppc_traces": plot_ppc_traces(),
        "posterior_structure": plot_posterior_structure(),
        "ablation": plot_ablation(),
        "trace_manifest": trace_manifest,
        "seed_robustness": seed_robustness,
        "multivariate_ppc": multivariate_ppc,
    }
    atomic_json(FIGURE_ROOT / "figure_manifest.json", paths)
    return paths


def _historical_comparison() -> list[dict[str, Any]]:
    raw_path = (
        RESULTS_ROOT.parent
        / "route3_7d_formal_validation"
        / "heldout_validation"
        / "formal_route3_decision.json"
    )
    if not raw_path.exists():
        raw_path = (
            RESULTS_ROOT.parent
            / "route3_pilot_snpe"
            / "heldout_validation"
            / "formal_route3_decision.json"
        )
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    rescue = json.loads(
        (PROJECT_ROOT / "ROUTE3_7D_RESCUE_FINAL_DECISION.json").read_text(
            encoding="utf-8"
        )
    )
    return [
        {
            "method": "Historical raw SNPE",
            "dimension": "7D/8D stored",
            "training_simulations": "stored pilot",
            "density_estimator": "historical MAF",
            "ensemble": "3 stored",
            "recovery": raw.get("checks", {}).get(
                "overall_recovery_improvement_fraction"
            ),
            "joint_coverage": "not Figure-10 joint diagnostic",
            "sbc": "historical marginal coverage",
            "lc2st": "not run",
            "ppc": raw.get("checks", {}).get(
                "overall_ppc_improvement_fraction"
            ),
            "contraction": "strict gate failed",
            "verdict": raw.get("decision", "NO-GO"),
        },
        {
            "method": "Historical rank-calibrated rescue",
            "dimension": "7D",
            "training_simulations": "8192 rows",
            "density_estimator": "MAF",
            "ensemble": "5",
            "recovery": rescue.get("decisive_evidence", {}).get(
                "recovery_improvement_fraction"
            ),
            "joint_coverage": "not Figure-10 joint diagnostic",
            "sbc": rescue.get("decisive_evidence", {}).get(
                "coverage_compatible_parameters"
            ),
            "lc2st": "not run",
            "ppc": rescue.get("decisive_evidence", {}).get(
                "ppc_improvement_fraction"
            ),
            "contraction": rescue.get("decisive_evidence", {}).get(
                "coverage_plus_contraction_parameters"
            ),
            "verdict": rescue.get("verdict", "NO-GO"),
        },
    ]


def build_final_report() -> dict[str, Any]:
    verdict8 = json.loads(
        (DIAGNOSTICS_ROOT / "8d_operational_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    verdict7 = json.loads(
        (DIAGNOSTICS_ROOT / "7d_operational_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    bank = json.loads(
        (RESULTS_ROOT / "matched_banks" / "matched_bank_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    train8 = json.loads(
        (
            TRAINING_ROOT
            / "8d"
            / f"scale_{FINAL_SCALE}"
            / "ensemble_manifest.json"
        ).read_text(encoding="utf-8")
    )
    train7 = json.loads(
        (
            TRAINING_ROOT
            / "7d"
            / f"scale_{FINAL_SCALE}"
            / "ensemble_manifest.json"
        ).read_text(encoding="utf-8")
    )
    structure8 = json.loads(
        (
            STRUCTURE_ROOT / "8d" / "posterior_structure_summary.json"
        ).read_text(encoding="utf-8")
    )
    structure7 = json.loads(
        (
            STRUCTURE_ROOT / "7d" / "posterior_structure_summary.json"
        ).read_text(encoding="utf-8")
    )
    figures = json.loads(
        (FIGURE_ROOT / "figure_manifest.json").read_text(encoding="utf-8")
    )
    comparison = _historical_comparison()
    for track, verdict, train, structure in (
        ("8d", verdict8, train8, structure8),
        ("7d", verdict7, train7, structure7),
    ):
        comparison.append(
            {
                "method": (
                    "New large-scale raw 8D ensemble"
                    if track == "8d"
                    else "New matched raw 7D ensemble"
                ),
                "dimension": track.upper(),
                "training_simulations": FINAL_SCALE,
                "density_estimator": "NSF 10x100",
                "ensemble": 5,
                "recovery": verdict["global"][
                    "ensemble_recovery_improvement_fraction"
                ],
                "joint_coverage": verdict["global"][
                    "ensemble_joint_ks_pvalue"
                ],
                "sbc": verdict["global"]["ensemble_clear_sbc_issue_count"],
                "lc2st": verdict["lc2st"]["ensemble_p_value"],
                "ppc": verdict["ppc"]["overall_improvement_fraction"],
                "contraction": structure[
                    "median_posterior_sd_over_prior_sd"
                ],
                "verdict": verdict["verdict"],
            }
        )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdicts": {
            "8d_figure10": verdict8["verdict"],
            "7d_figure10": verdict7["verdict"],
            "original_route3_strict": "NO-GO",
        },
        "training": {
            "8d_valid": bank["8d_valid"],
            "7d_valid": bank["7d_valid"],
            "members_each": 5,
            "scale": FINAL_SCALE,
        },
        "diagnostics": {
            "official_expected_coverage_cases": 300,
            "powered_cases": 1024,
            "posterior_samples_per_case": 1000,
            "lc2st_cases_per_track": 20000,
            "8d": verdict8,
            "7d": verdict7,
        },
        "posterior_structure": {"8d": structure8, "7d": structure7},
        "comparison": comparison,
        "figures": figures,
        "scope": "synthetic cortical-rate inference only",
        "preregistration_hash": verify_preregistration(),
    }
    atomic_json(REPORT_JSON_PATH, payload)
    comparison_frame = pd.DataFrame(comparison)
    table = comparison_frame.to_markdown(index=False)
    v8 = verdict8["verdict"].replace("8D FIGURE-10-EQUIVALENT ", "")
    v7 = verdict7["verdict"].replace("7D FIGURE-10-EQUIVALENT ", "")
    report = f"""# 8D FIGURE-10-EQUIVALENT: {v8}
# 7D FIGURE-10-EQUIVALENT: {v7}
# ORIGINAL ROUTE-3 STRICT: NO-GO

- Valid 8D training simulations: **{bank['8d_valid']:,}**
- Valid matched 7D training simulations: **{bank['7d_valid']:,}**
- Networks per final ensemble: **5**
- Expected-coverage/SBC cases: **300 official-equivalent + 1,024 powered per track**
- Posterior samples per global case: **1,000**
- L-C2ST calibration simulations: **20,000 per track**
- 8D powered joint-rank KS: `D={verdict8['global']['ensemble_joint_ks_statistic']:.4f}`, `p={verdict8['global']['ensemble_joint_ks_pvalue']:.4g}`
- 7D powered joint-rank KS: `D={verdict7['global']['ensemble_joint_ks_statistic']:.4f}`, `p={verdict7['global']['ensemble_joint_ks_pvalue']:.4g}`
- 8D clear SBC issues: **{verdict8['global']['ensemble_clear_sbc_issue_count']}/8**
- 7D clear SBC issues: **{verdict7['global']['ensemble_clear_sbc_issue_count']}/7**
- 8D L-C2ST: `p={verdict8['lc2st']['ensemble_p_value']:.4g}`, reject=`{verdict8['lc2st']['ensemble_reject']}`
- 7D L-C2ST: `p={verdict7['lc2st']['ensemble_p_value']:.4g}`, reject=`{verdict7['lc2st']['ensemble_reject']}`
- 8D PPC improvement: **{100 * verdict8['ppc']['overall_improvement_fraction']:.1f}%**, {verdict8['ppc']['features_posterior_better']}/14 features
- 7D PPC improvement: **{100 * verdict7['ppc']['overall_improvement_fraction']:.1f}%**, {verdict7['ppc']['features_posterior_better']}/14 features
- Original Route-3 hard-gate outcome: **NO-GO**
- Applicability: **synthetic cortical-rate inference only**

## Method comparison

{table}

## Direct answers

1. **Was 8,192 an important cause of earlier miscalibration?** See the frozen
   8,192-versus-32,768 scale table. Improvement is reported only when the
   measured calibration metrics support it; unrun 131k–3M scales are not
   extrapolated as scientific results.
2. **Did five-network ensembling improve calibration?** The individual and
   ensemble rank curves are reported side by side in the global diagnostics.
3. **Did releasing `c_ctx2th` harm the other seven parameters?** Shared
   contraction and calibration are compared under paired simulation draws and
   seeds; any widening is distinguished from miscalibration.
4. **Is `c_ctx2th` identifiable?** Its contraction and the
   `c_th2ctx`–`c_ctx2th` ridge are reported. Broad-but-calibrated uncertainty is
   accepted; narrow biased uncertainty is not.
5. **Does 8D give a scientifically more complete posterior?** It represents
   uncertainty in both coupling directions, but completeness does not override
   failed diagnostics.
6. **Which result is Figure-10-style valid?** Only the verdicts on the first
   two lines, under the resource-limited 32,768-scale preregistration.
7. **Which result is valid under original Route-3 strict rules?** The preserved
   historical `NO-GO`.
8. **What remains unsupported for real EEG?** All real Fpz–Cz parameter
   inference, because no independently validated source-to-sensor measurement
   model exists.

## Figures

- [Expected coverage and SBC]({figures['expected_coverage_sbc']['png']})
- [L-C2ST]({figures['lc2st']['png']})
- [Posterior predictive checks]({figures['ppc']['png']})
- [Posterior-predictive cortical-rate traces]({figures['ppc_traces']['png']})
- [Posterior structure]({figures['posterior_structure']['png']})
- [7D-vs-8D ablation]({figures['ablation']['png']})

1. What has been established: the reported verdicts characterize raw NSF posterior quality within the frozen synthetic cortical-rate proxy space at the measured 32,768-simulation scale.
2. What remains unsupported: real-EEG inference, a validated Fpz–Cz measurement model, subject-specific physiology, and exact 3-million-simulation Practical Guide reproduction.
3. The single highest-priority next action: acquire GPU/HPC simulation capacity and repeat the frozen scale ladder through at least one million matched valid simulations without changing priors, summaries, or final diagnostics.
"""
    atomic_text(REPORT_PATH, report)
    return payload


__all__ = [
    "FIGURE_ROOT",
    "REPORT_JSON_PATH",
    "REPORT_PATH",
    "build_final_report",
    "build_scale_sensitivity",
    "generate_all_figures",
]
