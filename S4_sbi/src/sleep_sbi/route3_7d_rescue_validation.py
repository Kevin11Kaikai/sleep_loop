"""Fresh, once-opened validation for the frozen Route-3 7D rescue pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .route3_7d_experiment import (
    PARAMETER_NAMES_7D,
    atomic_json,
    atomic_npz,
    insert_fixed_parameter,
    prior_bounds_7d,
    rate_feature_names,
    sha256_file,
    sobol_theta7,
)
from .route3_7d_rescue import (
    BANK_ROOT,
    FINAL_ROOT,
    ORIGINAL_ROOT,
    coverage_table,
    read_rescue_preregistration,
    sbc_table,
    verify_rescue_preregistration,
)
from .route3_7d_rescue_training import (
    apply_rank_calibrator,
    load_ensemble,
    load_rescue_training_data,
    sample_equal_ensemble,
    verify_primary_pipeline_lock,
)
from .route3_global_robustness import deterministic_seed_schedule, run_resumable_batch


def _run_theta7(
    theta7: np.ndarray,
    seeds: np.ndarray,
    output: Path,
    labels: list[str],
    max_workers: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    verify_rescue_preregistration()
    theta8 = insert_fixed_parameter(theta7)
    results = run_resumable_batch(
        theta8,
        seeds,
        output,
        model_version="v8a_t13",
        max_workers=max_workers,
        labels=labels,
    )
    return results, theta8


def _consolidate_results(
    path: Path,
    results: list[Mapping[str, Any]],
    theta7: np.ndarray,
    theta8: np.ndarray,
    **extra: Any,
) -> Path:
    arrays: dict[str, Any] = {
        "sample_id": np.arange(len(results), dtype=np.int64),
        "theta": np.asarray(theta7, np.float64),
        "theta_full_8d": np.asarray(theta8, np.float64),
        "x": np.vstack([row["x"] for row in results]).astype(np.float64),
        "validity": np.vstack([row["validity"] for row in results]).astype(bool),
        "simulator_seed": np.asarray(
            [row["simulator_seed"] for row in results], np.int64
        ),
        "success": np.asarray([row["success"] for row in results], bool),
        "runtime_s": np.asarray([row["runtime_s"] for row in results], float),
        "failure_reason": np.asarray(
            [row["failure_reason"] for row in results], dtype="<U1024"
        ),
        "parameter_names": np.asarray(PARAMETER_NAMES_7D, dtype="<U32"),
        "feature_names": np.asarray(rate_feature_names(), dtype="<U64"),
        "rescue_preregistration_hash": np.asarray(
            verify_rescue_preregistration(), dtype="<U64"
        ),
    }
    arrays.update(extra)
    atomic_npz(path, **arrays)
    return path


def run_fresh_final_1024(max_workers: int = 6) -> dict[str, Any]:
    """Generate the untouched final set only after the primary pipeline is locked."""

    selection = verify_primary_pipeline_lock()
    cfg = read_rescue_preregistration()
    theta7 = sobol_theta7(1024, int(cfg["seeds"]["final_sobol"]))
    seeds = deterministic_seed_schedule(
        1024, int(cfg["seeds"]["final_simulator_base"])
    )
    output = FINAL_ROOT / "dataset"
    opened_marker = FINAL_ROOT / "FINAL_SET_OPENED_ONCE.json"
    if not opened_marker.exists():
        atomic_json(
            opened_marker,
            {
                "opened_utc": datetime.now(timezone.utc).isoformat(),
                "primary_pipeline_sha256": sha256_file(
                    Path(selection_path())
                ),
                "purpose": "single frozen primary decision; no post-open tuning",
            },
        )
    results, theta8 = _run_theta7(
        theta7,
        seeds,
        output,
        [f"fresh_final_{index:04d}" for index in range(1024)],
        max_workers,
    )
    path = _consolidate_results(
        output / "route3_7d_rescue_fresh_final_1024.npz",
        results,
        theta7,
        theta8,
        dataset_role=np.asarray("fresh_independent_final", dtype="<U64"),
        primary_pipeline_sha256=np.asarray(
            sha256_file(Path(selection_path())), dtype="<U64"
        ),
    )
    success = np.asarray([row["success"] for row in results], bool)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scheduled": 1024,
        "valid": int(success.sum()),
        "failed": int((~success).sum()),
        "failure_rate": float((~success).mean()),
        "sum_simulator_runtime_s": float(
            np.nansum([row["runtime_s"] for row in results])
        ),
        "median_simulator_runtime_s": float(
            np.nanmedian([row["runtime_s"] for row in results])
        ),
        "artifact": path.as_posix(),
        "artifact_sha256": sha256_file(path),
        "simulator_seed_hash": sha256_file(path),
        "primary_pipeline_sha256": sha256_file(Path(selection_path())),
        "rescue_preregistration_hash": verify_rescue_preregistration(),
    }
    atomic_json(output / "manifest.json", manifest)
    return {"path": path, "manifest": manifest}


def selection_path() -> str:
    from .route3_7d_rescue import DEVELOPMENT_ROOT

    return str(DEVELOPMENT_ROOT / "primary_pipeline_locked.json")


def sample_fresh_final_posteriors() -> Path:
    selection = verify_primary_pipeline_lock()
    data = load_rescue_training_data()
    config_id = selection["selected_config_id"]
    method = selection["selected_method"]
    members = load_ensemble(config_id)
    dataset_path = FINAL_ROOT / "dataset" / "route3_7d_rescue_fresh_final_1024.npz"
    with np.load(dataset_path, allow_pickle=False) as dataset:
        theta = np.asarray(dataset["theta"], float)
        x = np.asarray(dataset["x"], float)
        success = np.asarray(dataset["success"], bool)
    if not success.all():
        raise RuntimeError("fresh final posterior sampling requires all cases valid")
    theta_unit = (theta - data.theta_lower) / (data.theta_upper - data.theta_lower)
    x_scaled = (x - data.x_location) / data.x_scale
    n_cases, n_samples = 1024, 4096
    output = FINAL_ROOT / "posterior_samples"
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    raw = np.empty((n_cases, n_samples, 7), np.float32)
    labels = np.empty((n_cases, n_samples), np.int8)
    runtime = np.empty(n_cases, float)
    base_seed = int(
        read_rescue_preregistration()["seeds"]["final_posterior_sampling"]
    )
    for case in range(n_cases):
        path = checkpoints / f"{case:04d}.npz"
        loaded = False
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as check:
                    loaded = (
                        str(check["rescue_preregistration_hash"].item())
                        == verify_rescue_preregistration()
                        and str(check["config_id"].item()) == config_id
                        and np.array_equal(check["theta_true"], theta_unit[case])
                    )
                    if loaded:
                        raw[case] = check["raw_samples"]
                        labels[case] = check["member_labels"]
                        runtime[case] = float(check["runtime_s"].item())
            except Exception:
                loaded = False
        if loaded:
            continue
        started = perf_counter()
        case_samples, case_labels = sample_equal_ensemble(
            members, x_scaled[case], n_samples, base_seed + case * 10_007
        )
        raw[case] = case_samples
        labels[case] = case_labels
        runtime[case] = perf_counter() - started
        atomic_npz(
            path,
            case=np.asarray(case, np.int64),
            theta_true=theta_unit[case],
            x_scaled=x_scaled[case],
            raw_samples=raw[case],
            member_labels=labels[case],
            runtime_s=np.asarray(runtime[case]),
            config_id=np.asarray(config_id, dtype="<U32"),
            rescue_preregistration_hash=np.asarray(
                verify_rescue_preregistration(), dtype="<U64"
            ),
        )
    if method == "empirical_rank_calibrated":
        from .route3_7d_rescue import DEVELOPMENT_ROOT

        with np.load(
            DEVELOPMENT_ROOT / "rank_calibrator.npz", allow_pickle=False
        ) as calibrator:
            sorted_ranks = np.asarray(
                calibrator["sorted_development_ranks"], float
            )
        primary = apply_rank_calibrator(raw, sorted_ranks).astype(np.float32)
    elif method == "raw":
        primary = raw.copy()
    else:
        raise RuntimeError(f"unsupported selected method {method}")
    path = output / "fresh_final_posterior_samples.npz"
    atomic_npz(
        path,
        theta_true_normalized=theta_unit,
        x_scaled=x_scaled,
        raw_samples=raw,
        primary_samples=primary,
        member_labels=labels,
        runtime_s=runtime,
        config_id=np.asarray(config_id, dtype="<U32"),
        primary_method=np.asarray(method, dtype="<U64"),
        parameter_names=np.asarray(PARAMETER_NAMES_7D, dtype="<U32"),
        primary_pipeline_sha256=np.asarray(
            sha256_file(Path(selection_path())), dtype="<U64"
        ),
        rescue_preregistration_hash=np.asarray(
            verify_rescue_preregistration(), dtype="<U64"
        ),
    )
    return path


def analyze_samples(
    theta: np.ndarray,
    samples: np.ndarray,
    labels: np.ndarray,
    estimator: str,
    output: Path,
    include_member_estimators: bool,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    posterior_median = np.median(samples, axis=1)
    posterior_mean = samples.mean(axis=1)
    quantiles = {
        q: np.quantile(samples, q, axis=1)
        for q in (0.05, 0.10, 0.25, 0.75, 0.90, 0.95)
    }
    prior_error = np.abs(theta - 0.5)
    median_error = np.abs(posterior_median - theta)
    recovery_rows = []
    for j, name in enumerate(PARAMETER_NAMES_7D):
        baseline = float(prior_error[:, j].mean())
        posterior = float(median_error[:, j].mean())
        recovery_rows.append(
            {
                "estimator": estimator,
                "parameter": name,
                "prior_median_mae": baseline,
                "posterior_median_mae": posterior,
                "posterior_median_rmse": float(
                    np.sqrt(np.mean((posterior_median[:, j] - theta[:, j]) ** 2))
                ),
                "posterior_mean_mae": float(
                    np.mean(np.abs(posterior_mean[:, j] - theta[:, j]))
                ),
                "mae_improvement_fraction": float(1 - posterior / baseline),
                "rank_correlation": float(
                    spearmanr(theta[:, j], posterior_median[:, j]).statistic
                ),
                "median_90_ci_width": float(
                    np.median(quantiles[0.95][:, j] - quantiles[0.05][:, j])
                ),
                "median_80_ci_width": float(
                    np.median(quantiles[0.90][:, j] - quantiles[0.10][:, j])
                ),
                "median_50_ci_width": float(
                    np.median(quantiles[0.75][:, j] - quantiles[0.25][:, j])
                ),
                "boundary_sample_fraction": float(
                    ((samples[:, :, j] < 0.01) | (samples[:, :, j] > 0.99)).mean()
                ),
            }
        )
    recovery = pd.DataFrame(recovery_rows)
    coverage_frames = [
        coverage_table(theta, samples, estimator, tuple(np.arange(0.1, 1, 0.1)))
    ]
    sbc_frames = [sbc_table(theta, samples, estimator)]
    if include_member_estimators:
        for member in sorted(np.unique(labels)):
            member_samples = np.stack(
                [samples[case, labels[case] == member] for case in range(len(theta))]
            )
            coverage_frames.append(
                coverage_table(
                    theta,
                    member_samples,
                    f"{estimator}_member_{member}",
                    tuple(np.arange(0.1, 1, 0.1)),
                )
            )
            sbc_frames.append(
                sbc_table(theta, member_samples, f"{estimator}_member_{member}")
            )
    coverage = pd.concat(coverage_frames, ignore_index=True)
    sbc = pd.concat(sbc_frames, ignore_index=True)
    case_metrics = pd.DataFrame(
        {
            "case": np.arange(len(theta)),
            "mean_parameter_median_abs_error": median_error.mean(axis=1),
            "mean_parameter_prior_median_abs_error": prior_error.mean(axis=1),
        }
    )
    disagreement_rows = []
    for j, name in enumerate(PARAMETER_NAMES_7D):
        member_medians = np.stack(
            [
                np.asarray(
                    [
                        np.median(samples[case, labels[case] == member, j])
                        for case in range(len(theta))
                    ]
                )
                for member in sorted(np.unique(labels))
            ],
            axis=1,
        )
        ranges = np.ptp(member_medians, axis=1)
        disagreement_rows.append(
            {
                "estimator": estimator,
                "parameter": name,
                "mean_member_median_range": float(ranges.mean()),
                "median_member_median_range": float(np.median(ranges)),
                "max_member_median_range": float(ranges.max()),
            }
        )
    disagreement = pd.DataFrame(disagreement_rows)
    ridges = pd.DataFrame(
        {
            "case": np.arange(len(theta)),
            "g_LK_g_h_spearman": [
                float(spearmanr(samples[case, :, 4], samples[case, :, 5]).statistic)
                for case in range(len(theta))
            ],
        }
    )
    recovery.to_csv(output / "parameter_recovery.csv", index=False)
    coverage.to_csv(output / "coverage_curve.csv", index=False)
    sbc.to_csv(output / "sbc_ranks.csv", index=False)
    case_metrics.to_csv(output / "case_recovery.csv", index=False)
    disagreement.to_csv(output / "ensemble_disagreement.csv", index=False)
    ridges.to_csv(output / "posterior_ridges.csv", index=False)
    return {
        "recovery": recovery,
        "coverage": coverage,
        "sbc": sbc,
        "case_metrics": case_metrics,
        "disagreement": disagreement,
        "ridges": ridges,
        "samples": samples,
        "theta": theta,
        "posterior_median": posterior_median,
    }


def analyze_fresh_final() -> dict[str, Any]:
    path = FINAL_ROOT / "posterior_samples" / "fresh_final_posterior_samples.npz"
    with np.load(path, allow_pickle=False) as data:
        theta = np.asarray(data["theta_true_normalized"], float)
        raw = np.asarray(data["raw_samples"], float)
        primary = np.asarray(data["primary_samples"], float)
        labels = np.asarray(data["member_labels"], int)
        method = str(data["primary_method"].item())
    raw_result = analyze_samples(
        theta, raw, labels, "raw", FINAL_ROOT / "metrics" / "raw", True
    )
    primary_result = analyze_samples(
        theta,
        primary,
        labels,
        method,
        FINAL_ROOT / "metrics" / "primary",
        False,
    )
    return {"raw": raw_result, "primary": primary_result, "method": method}


def run_final_ppc(
    primary_analysis: Mapping[str, Any],
    max_workers: int = 6,
) -> dict[str, Any]:
    cfg = read_rescue_preregistration()
    training = load_rescue_training_data()
    dataset_path = FINAL_ROOT / "dataset" / "route3_7d_rescue_fresh_final_1024.npz"
    with np.load(dataset_path, allow_pickle=False) as dataset:
        heldout_x = np.asarray(dataset["x"], float)
    seed = int(cfg["seeds"]["final_ppc_case_selection"])
    random_cases = np.sort(
        np.random.default_rng(seed).choice(1024, size=32, replace=False)
    )
    scores = primary_analysis["case_metrics"].set_index("case")[
        "mean_parameter_median_abs_error"
    ]
    worst_cases = np.sort(
        scores.drop(index=random_cases).nlargest(32).index.to_numpy()
    )
    case_ids = np.concatenate([random_cases, worst_cases])
    posterior_theta_unit = []
    case_rows = []
    for case in case_ids:
        indices = np.random.default_rng(seed + int(case)).choice(
            4096, size=16, replace=False
        )
        posterior_theta_unit.extend(primary_analysis["samples"][case, indices])
        case_rows.extend([case] * 16)
    posterior_theta_unit = np.asarray(posterior_theta_unit)
    posterior_theta = (
        training.theta_lower
        + posterior_theta_unit * (training.theta_upper - training.theta_lower)
    )
    posterior_seeds = deterministic_seed_schedule(
        1024, int(cfg["seeds"]["final_ppc_posterior_simulator_base"])
    )
    output = FINAL_ROOT / "ppc"
    post_results, post_theta8 = _run_theta7(
        posterior_theta,
        posterior_seeds,
        output / "posterior_predictive",
        [f"final_ppc_posterior_{index:04d}" for index in range(1024)],
        max_workers,
    )
    post_path = _consolidate_results(
        output / "posterior_predictive_1024.npz",
        post_results,
        posterior_theta,
        post_theta8,
        case_id=np.asarray(case_rows, np.int64),
    )
    prior_theta = sobol_theta7(1024, int(cfg["seeds"]["final_ppc_prior_sobol"]))
    prior_seeds = deterministic_seed_schedule(
        1024, int(cfg["seeds"]["final_ppc_prior_simulator_base"])
    )
    prior_results, prior_theta8 = _run_theta7(
        prior_theta,
        prior_seeds,
        output / "prior_predictive",
        [f"final_ppc_prior_{index:04d}" for index in range(1024)],
        max_workers,
    )
    prior_path = _consolidate_results(
        output / "prior_predictive_1024.npz",
        prior_results,
        prior_theta,
        prior_theta8,
        case_id=np.repeat(case_ids, 16).astype(np.int64),
    )
    posterior_x = np.vstack([row["x"] for row in post_results])
    prior_x = np.vstack([row["x"] for row in prior_results])
    rows = []
    for j, feature in enumerate(rate_feature_names()):
        posterior_error, prior_error = [], []
        for position, case in enumerate(case_ids):
            slc = slice(position * 16, (position + 1) * 16)
            observed = heldout_x[case, j]
            posterior_error.append(
                np.median(
                    np.abs(posterior_x[slc, j] - observed) / training.x_scale[j]
                )
            )
            prior_error.append(
                np.median(
                    np.abs(prior_x[slc, j] - observed) / training.x_scale[j]
                )
            )
        post_value = float(np.median(posterior_error))
        prior_value = float(np.median(prior_error))
        rows.append(
            {
                "feature": feature,
                "posterior_predictive_median_scaled_abs_error": post_value,
                "prior_predictive_median_scaled_abs_error": prior_value,
                "improvement_fraction": float(1 - post_value / prior_value),
                "posterior_better": post_value < prior_value,
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "ppc_feature_metrics.csv", index=False)
    post_success = np.asarray([row["success"] for row in post_results], bool)
    prior_success = np.asarray([row["success"] for row in prior_results], bool)
    summary = {
        "cases": 64,
        "random_cases": 32,
        "worst_cases": 32,
        "posterior_draws_per_case": 16,
        "posterior_predictive_attempted": 1024,
        "posterior_predictive_failed": int((~post_success).sum()),
        "prior_predictive_attempted": 1024,
        "prior_predictive_failed": int((~prior_success).sum()),
        "features_posterior_better": int(metrics.posterior_better.sum()),
        "overall_posterior_scaled_error": float(
            metrics.posterior_predictive_median_scaled_abs_error.median()
        ),
        "overall_prior_scaled_error": float(
            metrics.prior_predictive_median_scaled_abs_error.median()
        ),
        "overall_improvement_fraction": float(
            1
            - metrics.posterior_predictive_median_scaled_abs_error.median()
            / metrics.prior_predictive_median_scaled_abs_error.median()
        ),
        "posterior_artifact": post_path.as_posix(),
        "prior_artifact": prior_path.as_posix(),
    }
    atomic_json(
        output / "ppc_case_selection.json",
        {
            "random_case_ids": random_cases.tolist(),
            "worst_case_ids": worst_cases.tolist(),
            "selection_order_case_ids": case_ids.tolist(),
            "selection_rule": (
                "32 fixed-random plus 32 highest primary median recovery error "
                "after excluding random cases"
            ),
            "selection_seed": seed,
        },
    )
    atomic_json(output / "ppc_summary.json", summary)
    return {"metrics": metrics, "summary": summary}


def evaluate_final_decision(
    analysis: Mapping[str, Any], ppc: Mapping[str, Any]
) -> dict[str, Any]:
    selection = verify_primary_pipeline_lock()
    criteria = read_rescue_preregistration()["decision_criteria"]
    formal = criteria["formal_go"]
    conditional = criteria["conditional_go"]
    method = analysis["method"]
    primary = analysis["primary"]
    raw = analysis["raw"]
    final_manifest = json.loads(
        (FINAL_ROOT / "dataset" / "manifest.json").read_text(encoding="utf-8")
    )
    new_bank_manifest = json.loads(
        (
            BANK_ROOT / "additional_2048x2" / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    original_preflight = json.loads(
        (
            ORIGINAL_ROOT / "preflight" / "engineering_gate.json"
        ).read_text(encoding="utf-8")
    )
    coverage = primary["coverage"]
    estimator_coverage = coverage[coverage.estimator == method]
    c8090 = estimator_coverage[
        estimator_coverage.nominal_level.isin([0.8, 0.9])
    ]
    compatible = c8090.groupby("parameter").nominal_inside_wilson.all()
    severe = int(
        estimator_coverage.groupby("parameter").severe_undercoverage.any().sum()
    )
    recovery = primary["recovery"]
    better = int((recovery.mae_improvement_fraction > 0).sum())
    systematic_worse = int(
        (
            recovery.mae_improvement_fraction
            < formal["systematic_worsening_threshold"]
        ).sum()
    )
    prior_level = int(
        (
            (
                recovery.mae_improvement_fraction
                <= formal["prior_level_improvement_max"]
            )
            & (
                recovery.rank_correlation.abs()
                <= formal["prior_level_rank_correlation_max"]
            )
        ).sum()
    )
    case_metrics = primary["case_metrics"]
    overall_improvement = float(
        1
        - case_metrics.mean_parameter_median_abs_error.mean()
        / case_metrics.mean_parameter_prior_median_abs_error.mean()
    )
    contracted = set(
        recovery.loc[
            recovery.median_90_ci_width
            < formal["meaningful_contraction_width_max"],
            "parameter",
        ]
    )
    reasonable = set(compatible[compatible].index)
    contraction_coverage = len(contracted & reasonable)
    disagreement = raw["disagreement"]
    disagreement_mean = float(disagreement.mean_member_median_range.mean())
    disagreement_max = float(disagreement.mean_member_median_range.max())
    samples = primary["samples"]
    finite_support_rate = float(
        np.mean(
            np.isfinite(samples).all(axis=2)
            & ((samples >= 0) & (samples <= 1)).all(axis=2)
        )
    )
    checks = {
        "preflight_engineering_gate": bool(original_preflight["pass"]),
        "training_failure_gate": (
            new_bank_manifest["failure_rate"]
            <= formal["training_failure_rate_max"]
        ),
        "heldout_failure_gate": (
            final_manifest["failure_rate"]
            <= formal["heldout_failure_rate_max"]
        ),
        "posterior_finite_support_rate": finite_support_rate,
        "posterior_finite_support_gate": (
            finite_support_rate
            >= formal["posterior_finite_and_in_prior_rate_min"]
        ),
        "prior_level_unrecoverable_parameter_count": prior_level,
        "prior_level_unrecoverable_gate": (
            prior_level <= formal["prior_level_unrecoverable_parameters_max"]
        ),
        "systematically_worse_parameter_count": systematic_worse,
        "systematic_worsening_gate": (
            systematic_worse <= formal["systematically_worse_parameters_max"]
        ),
        "better_than_prior_parameter_count": better,
        "better_than_prior_gate": (
            better >= formal["median_error_better_than_prior_parameters_min"]
        ),
        "overall_recovery_improvement_fraction": overall_improvement,
        "overall_recovery_gate": (
            overall_improvement
            >= formal["overall_median_error_improvement_min_fraction"]
        ),
        "coverage_compatible_parameter_count": int(compatible.sum()),
        "coverage_compatible_gate": (
            int(compatible.sum())
            >= formal["coverage_80_or_90_wilson_contains_nominal_parameters_min"]
        ),
        "severe_undercoverage_parameter_count": severe,
        "severe_undercoverage_gate": (
            severe <= formal["severe_undercoverage_parameters_max"]
        ),
        "contraction_with_coverage_parameter_count": contraction_coverage,
        "contraction_with_coverage_gate": (
            contraction_coverage
            >= formal[
                "meaningful_contraction_with_reasonable_coverage_parameters_min"
            ]
        ),
        "ppc_better_feature_count": int(ppc["summary"]["features_posterior_better"]),
        "ppc_feature_gate": (
            int(ppc["summary"]["features_posterior_better"])
            >= formal["ppc_features_better_than_prior_min"]
        ),
        "overall_ppc_improvement_fraction": float(
            ppc["summary"]["overall_improvement_fraction"]
        ),
        "overall_ppc_gate": (
            float(ppc["summary"]["overall_improvement_fraction"])
            >= formal["overall_ppc_error_improvement_min_fraction"]
        ),
        "ensemble_disagreement_mean": disagreement_mean,
        "ensemble_disagreement_parameter_max": disagreement_max,
        "ensemble_disagreement_gate": (
            disagreement_mean
            <= formal[
                "ensemble_member_median_disagreement_mean_max_prior_width"
            ]
            and disagreement_max
            <= formal[
                "ensemble_member_disagreement_parameter_max_prior_width"
            ]
        ),
    }
    numeric_formal = all(
        value for key, value in checks.items() if key.endswith("_gate")
    )
    conditional_pass = (
        checks["preflight_engineering_gate"]
        and checks["training_failure_gate"]
        and checks["heldout_failure_gate"]
        and checks["posterior_finite_support_gate"]
        and severe <= conditional["severe_undercoverage_parameters_max"]
        and better >= conditional["median_error_better_than_prior_parameters_min"]
        and contraction_coverage
        >= conditional[
            "meaningful_contraction_with_reasonable_coverage_parameters_min"
        ]
        and ppc["summary"]["overall_improvement_fraction"] > 0
        and checks["ensemble_disagreement_gate"]
    )
    formal_eligible = bool(selection["formal_go_eligible"])
    if numeric_formal and formal_eligible:
        verdict = "FORMAL GO"
    elif conditional_pass:
        verdict = "CONDITIONAL GO"
    else:
        verdict = "NO-GO"
    blockers = [
        key for key, value in checks.items() if key.endswith("_gate") and not value
    ]
    if numeric_formal and not formal_eligible:
        blockers.append("calibrated_pipeline_not_formal_go_eligible")
    result = {
        "verdict": verdict,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "applies_to": method,
        "selected_config_id": selection["selected_config_id"],
        "formal_go_eligible": formal_eligible,
        "numeric_formal_gates_pass": numeric_formal,
        "conditional_gates_pass": conditional_pass,
        "fresh_final_cases": 1024,
        "parameters_passing_all_80_90_coverage_requirements": int(
            compatible.sum()
        ),
        "parameters_passing_coverage_plus_contraction": contraction_coverage,
        "checks": checks,
        "blockers": blockers,
        "scope": "synthetic cortical-rate observable inference only",
        "real_eeg_inference_unlocked": False,
        "rescue_preregistration_hash": verify_rescue_preregistration(),
        "primary_pipeline_sha256": sha256_file(Path(selection_path())),
    }
    atomic_json(FINAL_ROOT / "ROUTE3_7D_RESCUE_FINAL_DECISION.json", result)
    return result
