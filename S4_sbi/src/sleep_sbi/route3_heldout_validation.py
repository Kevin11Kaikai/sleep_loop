"""Independent held-out recovery, calibration, and PPC for Route-3."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chisquare, spearmanr

from .route3_global_robustness import (
    RESULTS_ROOT,
    deterministic_seed_schedule,
    rate_contract_hash,
    rate_feature_names,
    run_resumable_batch,
    sobol_theta,
    consolidate_batch,
)
from .route3_pilot_snpe import (
    ENSEMBLE_SEED,
    MODEL_SEEDS,
    TRAINING_ROOT,
    load_ensemble_members,
    load_training_data,
    sample_equal_ensemble_normalized,
)
from .route3_synthetic_preflight import PARAMETER_NAMES


HELDOUT_ROOT = RESULTS_ROOT / "heldout_validation"
HELDOUT_SOBOL_SEED = 20260802
HELDOUT_SIMULATOR_SEED_BASE = 910001
POSTERIOR_SAMPLES_PER_CASE = 4096
POSTERIOR_SAMPLE_SEED = 20260803
PPC_CASE_SELECTION_SEED = 20260804
PPC_CASES = 32
PPC_DRAWS_PER_CASE = 16
PPC_POSTERIOR_SIMULATOR_SEED_BASE = 920001
PPC_PRIOR_SOBOL_SEED = 20260805
PPC_PRIOR_SIMULATOR_SEED_BASE = 930001


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def run_heldout_dataset(
    training_bank_path: Path,
    output_dir: Path | None = None,
    max_workers: int = 6,
) -> dict[str, Any]:
    output_dir = Path(output_dir or HELDOUT_ROOT / "dataset")
    theta = sobol_theta(128, HELDOUT_SOBOL_SEED)
    with np.load(training_bank_path, allow_pickle=False) as training:
        training_theta = np.asarray(training["theta"], dtype=float)
    exact_duplicate = np.any(
        np.all(theta[:, None, :] == training_theta[None, :, :], axis=2)
    )
    if exact_duplicate:
        raise RuntimeError("held-out Sobol theta duplicates a training-bank row")
    seeds = deterministic_seed_schedule(128, HELDOUT_SIMULATOR_SEED_BASE)
    results = run_resumable_batch(
        theta,
        seeds,
        output_dir,
        max_workers=max_workers,
        labels=[f"heldout_{index:03d}" for index in range(128)],
    )
    path = consolidate_batch(
        results,
        output_dir / "route3_heldout_128.npz",
        {
            "sobol_seed": np.asarray(HELDOUT_SOBOL_SEED, dtype=np.int64),
            "seed_schedule_base": np.asarray(
                HELDOUT_SIMULATOR_SEED_BASE, dtype=np.int64
            ),
            "training_theta_exact_duplicate": np.asarray(False, dtype=bool),
            "dataset_role": np.asarray(
                "independent_heldout_synthetic_recovery", dtype="<U64"
            ),
        },
    )
    success = np.asarray([result["success"] for result in results], dtype=bool)
    manifest = {
        "attempted": 128,
        "success": int(success.sum()),
        "failed": int((~success).sum()),
        "failure_rate": float(1.0 - success.mean()),
        "sobol_seed": HELDOUT_SOBOL_SEED,
        "simulator_seed_base": HELDOUT_SIMULATOR_SEED_BASE,
        "exact_training_theta_duplicates": 0,
        "contract_hash": rate_contract_hash(),
    }
    _atomic_json(manifest, output_dir / "heldout_manifest.json")
    return {"path": path, "results": results, "manifest": manifest}


def generate_heldout_posterior_samples(
    heldout_path: Path,
    bank_path: Path,
    split_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir or HELDOUT_ROOT / "posterior_samples")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data = load_training_data(bank_path, split_path)
    posteriors = load_ensemble_members(data, TRAINING_ROOT)
    with np.load(heldout_path, allow_pickle=False) as heldout:
        theta = np.asarray(heldout["theta"], dtype=float)
        x = np.asarray(heldout["x"], dtype=float)
        success = np.asarray(heldout["success"], dtype=bool)
    if not success.all():
        raise RuntimeError("held-out dataset contains failed rows")
    theta_unit = (theta - data.theta_lower) / (data.theta_upper - data.theta_lower)
    x_scaled = (x - data.x_location) / data.x_scale
    samples = np.full(
        (128, POSTERIOR_SAMPLES_PER_CASE, 8), np.nan, dtype=np.float32
    )
    labels = np.full(
        (128, POSTERIOR_SAMPLES_PER_CASE), -1, dtype=np.int8
    )
    runtimes = np.full(128, np.nan)
    for case in range(128):
        path = checkpoint_dir / f"{case:03d}.npz"
        loaded = False
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as checkpoint:
                    if (
                        str(checkpoint["contract_hash"].item())
                        == rate_contract_hash()
                        and int(checkpoint["case"].item()) == case
                        and np.array_equal(checkpoint["theta_true"], theta_unit[case])
                    ):
                        samples[case] = checkpoint["samples"]
                        labels[case] = checkpoint["member_labels"]
                        runtimes[case] = checkpoint["runtime_s"].item()
                        loaded = True
            except Exception:
                loaded = False
        if loaded:
            continue
        started = perf_counter()
        case_samples, case_labels = sample_equal_ensemble_normalized(
            posteriors,
            x_scaled[case],
            POSTERIOR_SAMPLES_PER_CASE,
            seed=POSTERIOR_SAMPLE_SEED + case * 10_007,
        )
        samples[case] = case_samples.astype(np.float32)
        labels[case] = case_labels
        runtimes[case] = perf_counter() - started
        _atomic_npz(
            path,
            case=np.asarray(case, dtype=np.int64),
            theta_true=theta_unit[case].astype(np.float64),
            x_scaled=x_scaled[case].astype(np.float64),
            samples=samples[case],
            member_labels=labels[case],
            runtime_s=np.asarray(runtimes[case], dtype=np.float64),
            contract_hash=np.asarray(rate_contract_hash(), dtype="<U64"),
        )
    consolidated = output_dir / "heldout_ensemble_posterior_samples.npz"
    _atomic_npz(
        consolidated,
        theta_true_normalized=theta_unit.astype(np.float64),
        x_scaled=x_scaled.astype(np.float64),
        samples=samples,
        member_labels=labels,
        runtime_s=runtimes,
        parameter_names=np.asarray(PARAMETER_NAMES, dtype="<U32"),
        member_seeds=np.asarray(MODEL_SEEDS, dtype=np.int64),
        posterior_samples_per_case=np.asarray(
            POSTERIOR_SAMPLES_PER_CASE, dtype=np.int64
        ),
        contract_hash=np.asarray(rate_contract_hash(), dtype="<U64"),
    )
    return {
        "path": consolidated,
        "theta_true": theta_unit,
        "x_scaled": x_scaled,
        "samples": samples,
        "labels": labels,
        "runtime_s": runtimes,
        "data": data,
    }


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / n
    denominator = 1.0 + z**2 / n
    center = (proportion + z**2 / (2 * n)) / denominator
    radius = (
        z
        * np.sqrt(proportion * (1 - proportion) / n + z**2 / (4 * n**2))
        / denominator
    )
    return float(center - radius), float(center + radius)


def analyze_recovery_and_coverage(
    posterior_path: Path, output_dir: Path | None = None
) -> dict[str, Any]:
    output_dir = Path(output_dir or HELDOUT_ROOT / "metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(posterior_path, allow_pickle=False) as data:
        theta = np.asarray(data["theta_true_normalized"], dtype=float)
        samples = np.asarray(data["samples"], dtype=float)
        labels = np.asarray(data["member_labels"], dtype=int)
    posterior_mean = np.mean(samples, axis=1)
    posterior_median = np.median(samples, axis=1)
    quantiles = {
        value: np.quantile(samples, value, axis=1)
        for value in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    }
    prior_error = np.abs(theta - 0.5)
    median_error = np.abs(posterior_median - theta)
    mean_error = np.abs(posterior_mean - theta)
    recovery_rows = []
    for parameter_index, parameter in enumerate(PARAMETER_NAMES):
        rho = spearmanr(
            theta[:, parameter_index], posterior_median[:, parameter_index]
        ).statistic
        recovery_rows.append(
            {
                "parameter": parameter,
                "posterior_median_mae": float(
                    np.mean(median_error[:, parameter_index])
                ),
                "posterior_median_rmse": float(
                    np.sqrt(np.mean((posterior_median[:, parameter_index] - theta[:, parameter_index]) ** 2))
                ),
                "posterior_mean_mae": float(np.mean(mean_error[:, parameter_index])),
                "prior_median_mae": float(np.mean(prior_error[:, parameter_index])),
                "mae_improvement_fraction": float(
                    1
                    - np.mean(median_error[:, parameter_index])
                    / np.mean(prior_error[:, parameter_index])
                ),
                "rank_correlation": float(rho),
                "median_90_ci_width": float(
                    np.median(
                        quantiles[0.95][:, parameter_index]
                        - quantiles[0.05][:, parameter_index]
                    )
                ),
                "median_80_ci_width": float(
                    np.median(
                        quantiles[0.90][:, parameter_index]
                        - quantiles[0.10][:, parameter_index]
                    )
                ),
                "median_50_ci_width": float(
                    np.median(
                        quantiles[0.75][:, parameter_index]
                        - quantiles[0.25][:, parameter_index]
                    )
                ),
            }
        )
    recovery = pd.DataFrame(recovery_rows)
    recovery.to_csv(output_dir / "parameter_recovery.csv", index=False)
    case_metrics = pd.DataFrame(
        {
            "case": np.arange(128),
            "mean_parameter_median_abs_error": np.mean(median_error, axis=1),
            "mean_parameter_prior_median_abs_error": np.mean(prior_error, axis=1),
        }
    )
    case_metrics.to_csv(output_dir / "case_recovery.csv", index=False)

    estimator_sets: dict[str, list[np.ndarray]] = {"ensemble": [samples[case] for case in range(128)]}
    for member in range(3):
        estimator_sets[f"member_{member}"] = [
            samples[case, labels[case] == member] for case in range(128)
        ]
    coverage_rows = []
    sbc_rows = []
    for estimator, case_samples in estimator_sets.items():
        for parameter_index, parameter in enumerate(PARAMETER_NAMES):
            for level in (0.50, 0.80, 0.90):
                tail = (1.0 - level) / 2.0
                covered = []
                for case in range(128):
                    lower, upper = np.quantile(
                        case_samples[case][:, parameter_index], [tail, 1 - tail]
                    )
                    covered.append(lower <= theta[case, parameter_index] <= upper)
                successes = int(np.sum(covered))
                ci_low, ci_high = _wilson_interval(successes, 128)
                coverage_rows.append(
                    {
                        "estimator": estimator,
                        "parameter": parameter,
                        "nominal_level": level,
                        "covered": successes,
                        "n": 128,
                        "empirical_coverage": successes / 128,
                        "wilson_95_low": ci_low,
                        "wilson_95_high": ci_high,
                        "nominal_inside_wilson": ci_low <= level <= ci_high,
                        "severe_undercoverage": bool(
                            successes / 128 < level - 0.15 and ci_high < level
                        ),
                    }
                )
            ranks = np.asarray(
                [
                    np.sum(case_samples[case][:, parameter_index] < theta[case, parameter_index])
                    for case in range(128)
                ]
            )
            sample_counts = np.asarray(
                [len(case_samples[case]) for case in range(128)]
            )
            normalized_ranks = (ranks + 0.5) / (sample_counts + 1.0)
            histogram, _ = np.histogram(normalized_ranks, bins=np.linspace(0, 1, 11))
            statistic, p_value = chisquare(histogram)
            sbc_rows.append(
                {
                    "estimator": estimator,
                    "parameter": parameter,
                    "rank_mean": float(np.mean(normalized_ranks)),
                    "rank_std": float(np.std(normalized_ranks)),
                    "chi_square_10bin": float(statistic),
                    "chi_square_p_value": float(p_value),
                    **{f"bin_{index}": int(value) for index, value in enumerate(histogram)},
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    sbc = pd.DataFrame(sbc_rows)
    coverage.to_csv(output_dir / "coverage.csv", index=False)
    sbc.to_csv(output_dir / "sbc_ranks.csv", index=False)

    disagreement_rows = []
    for parameter_index, parameter in enumerate(PARAMETER_NAMES):
        member_medians = np.stack(
            [
                np.asarray(
                    [
                        np.median(
                            samples[case, labels[case] == member, parameter_index]
                        )
                        for case in range(128)
                    ]
                )
                for member in range(3)
            ],
            axis=1,
        )
        ranges = np.ptp(member_medians, axis=1)
        disagreement_rows.append(
            {
                "parameter": parameter,
                "mean_member_median_range": float(np.mean(ranges)),
                "median_member_median_range": float(np.median(ranges)),
                "max_member_median_range": float(np.max(ranges)),
            }
        )
    disagreement = pd.DataFrame(disagreement_rows)
    disagreement.to_csv(output_dir / "ensemble_disagreement.csv", index=False)

    ridge_rows = []
    for case in range(128):
        ridge_rows.append(
            {
                "case": case,
                "g_LK_g_h_spearman": float(
                    spearmanr(samples[case, :, 4], samples[case, :, 5]).statistic
                ),
                "c_th2ctx_c_ctx2th_spearman": float(
                    spearmanr(samples[case, :, 6], samples[case, :, 7]).statistic
                ),
            }
        )
    ridges = pd.DataFrame(ridge_rows)
    ridges.to_csv(output_dir / "posterior_ridges.csv", index=False)
    return {
        "recovery": recovery,
        "case_metrics": case_metrics,
        "coverage": coverage,
        "sbc": sbc,
        "disagreement": disagreement,
        "ridges": ridges,
        "theta": theta,
        "samples": samples,
        "labels": labels,
        "posterior_median": posterior_median,
    }


def run_synthetic_ppc(
    recovery: Mapping[str, Any],
    heldout_path: Path,
    bank_path: Path,
    split_path: Path,
    output_dir: Path | None = None,
    max_workers: int = 6,
) -> dict[str, Any]:
    output_dir = Path(output_dir or HELDOUT_ROOT / "ppc")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_training_data(bank_path, split_path)
    with np.load(heldout_path, allow_pickle=False) as heldout:
        heldout_x = np.asarray(heldout["x"], dtype=float)
    rng = np.random.default_rng(PPC_CASE_SELECTION_SEED)
    case_ids = np.sort(rng.choice(128, size=PPC_CASES, replace=False))
    posterior_theta_rows = []
    posterior_case_rows = []
    samples = recovery["samples"]
    for case in case_ids:
        case_rng = np.random.default_rng(PPC_CASE_SELECTION_SEED + int(case))
        indices = case_rng.choice(
            POSTERIOR_SAMPLES_PER_CASE, size=PPC_DRAWS_PER_CASE, replace=False
        )
        posterior_theta_rows.extend(samples[case, indices])
        posterior_case_rows.extend([case] * PPC_DRAWS_PER_CASE)
    posterior_theta_unit = np.asarray(posterior_theta_rows, dtype=float)
    posterior_theta = data.theta_lower + posterior_theta_unit * (
        data.theta_upper - data.theta_lower
    )
    posterior_seeds = deterministic_seed_schedule(
        len(posterior_theta), PPC_POSTERIOR_SIMULATOR_SEED_BASE
    )
    posterior_results = run_resumable_batch(
        posterior_theta,
        posterior_seeds,
        output_dir / "posterior_predictive",
        max_workers=max_workers,
        labels=[f"ppc_post_{index:04d}" for index in range(len(posterior_theta))],
    )
    posterior_path = consolidate_batch(
        posterior_results,
        output_dir / "posterior_predictive_512.npz",
        {"case_id": np.asarray(posterior_case_rows, dtype=np.int64)},
    )

    prior_theta = sobol_theta(PPC_CASES * PPC_DRAWS_PER_CASE, PPC_PRIOR_SOBOL_SEED)
    prior_case_rows = np.repeat(case_ids, PPC_DRAWS_PER_CASE)
    prior_seeds = deterministic_seed_schedule(
        len(prior_theta), PPC_PRIOR_SIMULATOR_SEED_BASE
    )
    prior_results = run_resumable_batch(
        prior_theta,
        prior_seeds,
        output_dir / "prior_predictive",
        max_workers=max_workers,
        labels=[f"ppc_prior_{index:04d}" for index in range(len(prior_theta))],
    )
    prior_path = consolidate_batch(
        prior_results,
        output_dir / "prior_predictive_512.npz",
        {"case_id": prior_case_rows.astype(np.int64)},
    )
    posterior_x = np.vstack([result["x"] for result in posterior_results])
    prior_x = np.vstack([result["x"] for result in prior_results])
    posterior_success = np.asarray(
        [result["success"] for result in posterior_results], dtype=bool
    )
    prior_success = np.asarray([result["success"] for result in prior_results], dtype=bool)
    rows = []
    for feature_index, feature in enumerate(rate_feature_names()):
        posterior_errors = []
        prior_errors = []
        for case_position, case in enumerate(case_ids):
            slc = slice(
                case_position * PPC_DRAWS_PER_CASE,
                (case_position + 1) * PPC_DRAWS_PER_CASE,
            )
            observed = heldout_x[case, feature_index]
            posterior_errors.append(
                np.median(
                    np.abs(posterior_x[slc, feature_index] - observed)
                    / data.x_scale[feature_index]
                )
            )
            prior_errors.append(
                np.median(
                    np.abs(prior_x[slc, feature_index] - observed)
                    / data.x_scale[feature_index]
                )
            )
        posterior_median_error = float(np.median(posterior_errors))
        prior_median_error = float(np.median(prior_errors))
        rows.append(
            {
                "feature": feature,
                "posterior_predictive_median_scaled_abs_error": posterior_median_error,
                "prior_predictive_median_scaled_abs_error": prior_median_error,
                "improvement_fraction": float(
                    1 - posterior_median_error / prior_median_error
                ),
                "posterior_better": posterior_median_error < prior_median_error,
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "ppc_feature_metrics.csv", index=False)
    case_scores = recovery["case_metrics"].set_index("case")[
        "mean_parameter_median_abs_error"
    ]
    selected_scores = case_scores.loc[case_ids]
    worst_cases = selected_scores.nlargest(5).index.to_numpy()
    median_score = float(selected_scores.median())
    typical_cases = (
        (selected_scores - median_score).abs().nsmallest(5).index.to_numpy()
    )
    selection = {
        "random_case_ids": case_ids.tolist(),
        "worst_recovery_cases_within_random_set": worst_cases.tolist(),
        "typical_cases_closest_to_random_set_median_recovery": typical_cases.tolist(),
        "case_selection_seed": PPC_CASE_SELECTION_SEED,
        "draws_per_case": PPC_DRAWS_PER_CASE,
    }
    _atomic_json(selection, output_dir / "ppc_case_selection.json")
    summary = {
        "cases": PPC_CASES,
        "posterior_draws_per_case": PPC_DRAWS_PER_CASE,
        "posterior_predictive_attempted": len(posterior_results),
        "posterior_predictive_failed": int((~posterior_success).sum()),
        "prior_predictive_attempted": len(prior_results),
        "prior_predictive_failed": int((~prior_success).sum()),
        "features_posterior_better": int(metrics["posterior_better"].sum()),
        "overall_posterior_scaled_error": float(
            metrics["posterior_predictive_median_scaled_abs_error"].median()
        ),
        "overall_prior_scaled_error": float(
            metrics["prior_predictive_median_scaled_abs_error"].median()
        ),
        "overall_improvement_fraction": float(
            1
            - metrics["posterior_predictive_median_scaled_abs_error"].median()
            / metrics["prior_predictive_median_scaled_abs_error"].median()
        ),
    }
    _atomic_json(summary, output_dir / "ppc_summary.json")
    return {
        "metrics": metrics,
        "summary": summary,
        "selection": selection,
        "posterior_path": posterior_path,
        "prior_path": prior_path,
        "case_ids": case_ids,
        "posterior_x": posterior_x,
        "prior_x": prior_x,
        "heldout_x": heldout_x,
    }


def evaluate_go_decision(
    criteria_path: Path,
    training_manifest_path: Path,
    heldout_manifest_path: Path,
    recovery: Mapping[str, Any],
    ppc: Mapping[str, Any],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir or HELDOUT_ROOT)
    criteria = json.loads(Path(criteria_path).read_text(encoding="utf-8"))
    training = json.loads(Path(training_manifest_path).read_text(encoding="utf-8"))
    heldout = json.loads(Path(heldout_manifest_path).read_text(encoding="utf-8"))
    formal = criteria["formal_go"]
    recovery_table = recovery["recovery"]
    coverage = recovery["coverage"]
    ensemble_coverage = coverage[coverage.estimator == "ensemble"]
    coverage_80_90 = ensemble_coverage[
        ensemble_coverage.nominal_level.isin([0.8, 0.9])
    ]
    compatible_by_parameter = (
        coverage_80_90.groupby("parameter")["nominal_inside_wilson"].all()
    )
    severe_parameters = int(
        ensemble_coverage.groupby("parameter")["severe_undercoverage"].any().sum()
    )
    better_parameters = int((recovery_table.mae_improvement_fraction > 0).sum())
    overall_prior_error = float(recovery["case_metrics"]["mean_parameter_prior_median_abs_error"].mean())
    overall_posterior_error = float(recovery["case_metrics"]["mean_parameter_median_abs_error"].mean())
    overall_recovery_improvement = 1 - overall_posterior_error / overall_prior_error
    reasonable_parameters = set(compatible_by_parameter[compatible_by_parameter].index)
    contracted_parameters = set(
        recovery_table.loc[
            recovery_table.median_90_ci_width < 0.90, "parameter"
        ]
    )
    contraction_coverage_count = len(reasonable_parameters & contracted_parameters)
    disagreement = recovery["disagreement"]
    disagreement_mean = float(disagreement.mean_member_median_range.mean())
    disagreement_parameter_max = float(
        disagreement.mean_member_median_range.max()
    )
    finite_support_rate = float(
        np.isfinite(recovery["samples"]).all(axis=2).mean()
        * (
            ((recovery["samples"] >= 0) & (recovery["samples"] <= 1))
            .all(axis=2)
            .mean()
        )
    )
    checks = {
        "training_failure_gate": training["failure_rate"]
        <= formal["training_failure_rate_max"],
        "heldout_failure_gate": heldout["failure_rate"]
        <= formal["heldout_failure_rate_max"],
        "posterior_finite_support_gate": finite_support_rate
        >= formal["posterior_finite_and_in_prior_rate_min"],
        "coverage_compatible_parameter_count": int(compatible_by_parameter.sum()),
        "coverage_compatible_gate": int(compatible_by_parameter.sum())
        >= formal["coverage_80_or_90_wilson_contains_nominal_parameters_min"],
        "severe_undercoverage_parameter_count": severe_parameters,
        "severe_undercoverage_gate": severe_parameters
        <= formal["severe_undercoverage_parameters_max"],
        "better_than_prior_parameter_count": better_parameters,
        "better_than_prior_gate": better_parameters
        >= formal["median_error_better_than_prior_parameters_min"],
        "overall_recovery_improvement_fraction": overall_recovery_improvement,
        "overall_recovery_gate": overall_recovery_improvement
        >= formal["overall_median_error_improvement_min_fraction"],
        "contraction_with_coverage_parameter_count": contraction_coverage_count,
        "contraction_with_coverage_gate": contraction_coverage_count
        >= formal[
            "meaningful_contraction_with_reasonable_coverage_parameters_min"
        ],
        "ppc_better_feature_count": ppc["summary"]["features_posterior_better"],
        "ppc_feature_gate": ppc["summary"]["features_posterior_better"]
        >= formal["ppc_features_better_than_prior_min"],
        "overall_ppc_improvement_fraction": ppc["summary"][
            "overall_improvement_fraction"
        ],
        "overall_ppc_gate": ppc["summary"]["overall_improvement_fraction"]
        >= formal["overall_ppc_error_improvement_min_fraction"],
        "ensemble_disagreement_mean": disagreement_mean,
        "ensemble_disagreement_parameter_max": disagreement_parameter_max,
        "ensemble_disagreement_gate": disagreement_mean
        <= formal["ensemble_member_median_disagreement_mean_max_prior_width"]
        and disagreement_parameter_max
        <= formal["ensemble_member_disagreement_parameter_max_prior_width"],
    }
    boolean_formal = [
        value for key, value in checks.items() if key.endswith("_gate")
    ]
    formal_go = all(boolean_formal)
    conditional = criteria["conditional_go"]
    conditional_go = (
        checks["training_failure_gate"]
        and checks["heldout_failure_gate"]
        and checks["posterior_finite_support_gate"]
        and severe_parameters
        <= conditional["severe_undercoverage_parameters_max"]
        and better_parameters
        >= conditional["median_error_better_than_prior_parameters_min"]
        and contraction_coverage_count
        >= conditional[
            "meaningful_contraction_with_reasonable_coverage_parameters_min"
        ]
        and ppc["summary"]["overall_improvement_fraction"] > 0
        and checks["ensemble_disagreement_gate"]
    )
    decision = "GO" if formal_go else "CONDITIONAL GO" if conditional_go else "NO-GO"
    result = {
        "criteria_version": criteria["version"],
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "checks": checks,
        "scope": "synthetic cortical-rate observable recovery only",
        "real_eeg_inference_unlocked": False,
    }
    _atomic_json(result, output_dir / "formal_route3_decision.json")
    return result


__all__ = [
    "HELDOUT_ROOT",
    "HELDOUT_SIMULATOR_SEED_BASE",
    "HELDOUT_SOBOL_SEED",
    "POSTERIOR_SAMPLES_PER_CASE",
    "PPC_CASES",
    "PPC_DRAWS_PER_CASE",
    "analyze_recovery_and_coverage",
    "evaluate_go_decision",
    "generate_heldout_posterior_samples",
    "run_heldout_dataset",
    "run_synthetic_ppc",
]
