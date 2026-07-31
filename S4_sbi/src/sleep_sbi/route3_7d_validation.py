"""Independent held-out validation for the preregistered Route-3 7D ensemble."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import chisquare, spearmanr

from .route3_7d_experiment import (
    BANK_ROOT,
    HELDOUT_ROOT,
    PARAMETER_NAMES_7D,
    PREFLIGHT_ROOT,
    TRAINING_ROOT,
    _consolidate_7d,
    _run_7d_batch,
    atomic_json,
    atomic_npz,
    deterministic_seed_schedule,
    fixed_c_ctx2th,
    prior_bounds_7d,
    rate_feature_names,
    read_preregistration,
    sha256_file,
    sobol_theta7,
    verify_preregistration,
)
from .route3_7d_training import (
    load_ensemble_7d,
    load_training_data_7d,
    model_seeds,
    sample_equal_ensemble_7d,
)


POSTERIOR_SAMPLES_PER_CASE = 4096
HELDOUT_CASES = 256
PPC_RANDOM_CASES = 32
PPC_WORST_CASES = 32
PPC_DRAWS_PER_CASE = 16


def _wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    radius = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return float(center - radius), float(center + radius)


def freeze_heldout_criteria() -> dict[str, Any]:
    """Copy the already locked preregistered criteria before held-out generation."""

    verify_preregistration()
    payload = {
        "experiment_id": read_preregistration()["experiment_id"],
        "criteria": read_preregistration()["decision_criteria"],
        "source_preregistration_hash": verify_preregistration(),
        "immutable_after_heldout_open": True,
    }
    path = HELDOUT_ROOT / "go_criteria_locked.json"
    digest_path = HELDOUT_ROOT / "go_criteria_locked.sha256"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing 7D held-out criteria differs from preregistration")
    else:
        atomic_json(path, payload)
    digest = sha256_file(path)
    if digest_path.exists() and digest_path.read_text(encoding="ascii").strip() != digest:
        raise RuntimeError("locked 7D held-out criteria hash changed")
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.write_text(digest + "\n", encoding="ascii")
    return {"path": path, "sha256": digest, "payload": payload}


def verify_heldout_criteria() -> str:
    frozen = freeze_heldout_criteria()
    if sha256_file(frozen["path"]) != frozen["sha256"]:
        raise RuntimeError("held-out criteria hash verification failed")
    return frozen["sha256"]


def run_heldout_256(max_workers: int = 6) -> dict[str, Any]:
    verify_preregistration()
    criteria_hash = verify_heldout_criteria()
    cfg = read_preregistration()["seeds"]
    theta = sobol_theta7(HELDOUT_CASES, int(cfg["heldout_sobol"]))
    with np.load(BANK_ROOT / "route3_7d_cortex_rate_14d_bank_4096.npz", allow_pickle=False) as bank:
        training_theta = np.asarray(bank["theta"], float)
        training_seeds = np.asarray(bank["simulator_seed"], np.int64)
    exact = bool(np.any(np.all(theta[:, None, :] == training_theta[None, :, :], axis=2)))
    if exact:
        raise RuntimeError("7D held-out theta exactly duplicates training theta")
    seeds = deterministic_seed_schedule(HELDOUT_CASES, int(cfg["heldout_simulator_base"]))
    if np.intersect1d(seeds, training_seeds).size:
        raise RuntimeError("7D held-out simulator seed overlaps training seed")
    out = HELDOUT_ROOT / "dataset"
    results = _run_7d_batch(
        theta, seeds, out,
        [f"heldout7d_{i:03d}" for i in range(HELDOUT_CASES)], max_workers,
    )
    path = _consolidate_7d(
        results, theta, out / "route3_7d_heldout_256.npz",
        {
            "dataset_role": np.asarray("independent_7d_heldout", dtype="<U64"),
            "heldout_criteria_hash": np.asarray(criteria_hash, dtype="<U64"),
            "training_theta_exact_duplicate": np.asarray(False, bool),
        },
    )
    success = np.asarray([row["success"] for row in results], bool)
    manifest = {
        "attempted": HELDOUT_CASES, "success": int(success.sum()),
        "failed": int((~success).sum()), "failure_rate": float(1 - success.mean()),
        "sum_simulator_runtime_s": float(np.nansum([row["runtime_s"] for row in results])),
        "median_runtime_s": float(np.nanmedian([row["runtime_s"] for row in results])),
        "exact_training_theta_duplicates": 0,
        "training_simulator_seed_overlaps": 0,
        "fixed_c_ctx2th": fixed_c_ctx2th(),
        "preregistration_hash": verify_preregistration(),
        "heldout_criteria_hash": criteria_hash,
    }
    atomic_json(out / "heldout_manifest.json", manifest)
    return {"path": path, "manifest": manifest}


def sample_heldout_posteriors(heldout_path: Path) -> dict[str, Any]:
    verify_preregistration()
    criteria_hash = verify_heldout_criteria()
    data = load_training_data_7d()
    members = load_ensemble_7d(data)
    with np.load(heldout_path, allow_pickle=False) as heldout:
        theta = np.asarray(heldout["theta"], float)
        x = np.asarray(heldout["x"], float)
        success = np.asarray(heldout["success"], bool)
    if not success.all():
        raise RuntimeError("held-out posterior sampling requires all 256 valid observations")
    theta_unit = (theta - data.theta_lower) / (data.theta_upper - data.theta_lower)
    x_scaled = (x - data.x_location) / data.x_scale
    if not np.isfinite(x_scaled).all():
        raise RuntimeError("held-out scaled observations contain nonfinite values")
    cfg = read_preregistration()["seeds"]
    base_seed = int(cfg["posterior_sampling"])
    checkpoint_dir = HELDOUT_ROOT / "posterior_samples" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples = np.empty((HELDOUT_CASES, POSTERIOR_SAMPLES_PER_CASE, 7), np.float32)
    labels = np.empty((HELDOUT_CASES, POSTERIOR_SAMPLES_PER_CASE), np.int8)
    runtimes = np.empty(HELDOUT_CASES, float)
    for case in range(HELDOUT_CASES):
        path = checkpoint_dir / f"{case:03d}.npz"
        loaded = False
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as check:
                    loaded = (
                        str(check["preregistration_hash"].item()) == verify_preregistration()
                        and str(check["heldout_criteria_hash"].item()) == criteria_hash
                        and int(check["case"].item()) == case
                        and np.array_equal(check["theta_true"], theta_unit[case])
                    )
                    if loaded:
                        samples[case] = check["samples"]
                        labels[case] = check["member_labels"]
                        runtimes[case] = float(check["runtime_s"].item())
            except Exception:
                loaded = False
        if loaded:
            continue
        started = perf_counter()
        case_samples, case_labels = sample_equal_ensemble_7d(
            members, x_scaled[case], POSTERIOR_SAMPLES_PER_CASE,
            base_seed + case * 10_007,
        )
        samples[case] = case_samples.astype(np.float32)
        labels[case] = case_labels
        runtimes[case] = perf_counter() - started
        atomic_npz(
            path,
            case=np.asarray(case, np.int64),
            theta_true=theta_unit[case].astype(np.float64),
            x_scaled=x_scaled[case].astype(np.float64),
            samples=samples[case], member_labels=labels[case],
            runtime_s=np.asarray(runtimes[case], float),
            preregistration_hash=np.asarray(verify_preregistration(), dtype="<U64"),
            heldout_criteria_hash=np.asarray(criteria_hash, dtype="<U64"),
        )
    consolidated = HELDOUT_ROOT / "posterior_samples" / "heldout_ensemble_samples.npz"
    atomic_npz(
        consolidated,
        theta_true_normalized=theta_unit.astype(np.float64),
        x_scaled=x_scaled.astype(np.float64),
        samples=samples, member_labels=labels, runtime_s=runtimes,
        parameter_names=np.asarray(PARAMETER_NAMES_7D, dtype="<U32"),
        member_seeds=np.asarray(model_seeds(), np.int64),
        posterior_samples_per_case=np.asarray(POSTERIOR_SAMPLES_PER_CASE, np.int64),
        preregistration_hash=np.asarray(verify_preregistration(), dtype="<U64"),
        heldout_criteria_hash=np.asarray(criteria_hash, dtype="<U64"),
    )
    return {"path": consolidated, "theta": theta_unit, "x_scaled": x_scaled}


def analyze_recovery_coverage(posterior_path: Path) -> dict[str, Any]:
    out = HELDOUT_ROOT / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    with np.load(posterior_path, allow_pickle=False) as data:
        theta = np.asarray(data["theta_true_normalized"], float)
        samples = np.asarray(data["samples"], float)
        labels = np.asarray(data["member_labels"], int)
    if theta.shape != (256, 7) or samples.shape != (256, 4096, 7):
        raise RuntimeError("held-out posterior shape drift")
    posterior_mean = samples.mean(axis=1)
    posterior_median = np.median(samples, axis=1)
    quantiles = {q: np.quantile(samples, q, axis=1) for q in (.05, .10, .25, .75, .90, .95)}
    prior_error = np.abs(theta - .5)
    median_error = np.abs(posterior_median - theta)
    recovery_rows = []
    for j, name in enumerate(PARAMETER_NAMES_7D):
        baseline = float(prior_error[:, j].mean())
        posterior = float(median_error[:, j].mean())
        recovery_rows.append({
            "parameter": name,
            "prior_median_mae": baseline,
            "posterior_median_mae": posterior,
            "posterior_median_rmse": float(np.sqrt(np.mean((posterior_median[:, j] - theta[:, j]) ** 2))),
            "posterior_mean_mae": float(np.mean(np.abs(posterior_mean[:, j] - theta[:, j]))),
            "mae_improvement_fraction": float(1 - posterior / baseline),
            "rank_correlation": float(spearmanr(theta[:, j], posterior_median[:, j]).statistic),
            "median_90_ci_width": float(np.median(quantiles[.95][:, j] - quantiles[.05][:, j])),
            "median_80_ci_width": float(np.median(quantiles[.90][:, j] - quantiles[.10][:, j])),
            "median_50_ci_width": float(np.median(quantiles[.75][:, j] - quantiles[.25][:, j])),
            "boundary_sample_fraction": float(((samples[:, :, j] < .01) | (samples[:, :, j] > .99)).mean()),
        })
    recovery = pd.DataFrame(recovery_rows)
    recovery.to_csv(out / "parameter_recovery.csv", index=False)
    case_metrics = pd.DataFrame({
        "case": np.arange(256),
        "mean_parameter_median_abs_error": median_error.mean(axis=1),
        "mean_parameter_prior_median_abs_error": prior_error.mean(axis=1),
    })
    case_metrics.to_csv(out / "case_recovery.csv", index=False)
    estimator_sets: dict[str, list[np.ndarray]] = {
        "ensemble": [samples[case] for case in range(256)]
    }
    for member in range(3):
        estimator_sets[f"member_{member}"] = [
            samples[case, labels[case] == member] for case in range(256)
        ]
    coverage_rows, sbc_rows = [], []
    for estimator, case_samples in estimator_sets.items():
        for j, name in enumerate(PARAMETER_NAMES_7D):
            for level in (.5, .8, .9):
                tail = (1 - level) / 2
                covered = [
                    np.quantile(case_samples[case][:, j], tail) <= theta[case, j]
                    <= np.quantile(case_samples[case][:, j], 1 - tail)
                    for case in range(256)
                ]
                successes = int(np.sum(covered))
                low, high = _wilson(successes, 256)
                coverage_rows.append({
                    "estimator": estimator, "parameter": name,
                    "nominal_level": level, "covered": successes, "n": 256,
                    "empirical_coverage": successes / 256,
                    "wilson_95_low": low, "wilson_95_high": high,
                    "nominal_inside_wilson": low <= level <= high,
                    "severe_undercoverage": bool(
                        successes / 256 < level - .15 and high < level
                    ),
                })
            ranks = np.asarray([
                np.sum(case_samples[case][:, j] < theta[case, j])
                for case in range(256)
            ])
            counts = np.asarray([len(case_samples[case]) for case in range(256)])
            normalized = (ranks + .5) / (counts + 1)
            histogram, _ = np.histogram(normalized, bins=np.linspace(0, 1, 11))
            statistic, p = chisquare(histogram)
            sbc_rows.append({
                "estimator": estimator, "parameter": name,
                "rank_mean": float(normalized.mean()), "rank_std": float(normalized.std()),
                "chi_square_10bin": float(statistic), "chi_square_p_value": float(p),
                **{f"bin_{k}": int(v) for k, v in enumerate(histogram)},
            })
    coverage = pd.DataFrame(coverage_rows)
    sbc = pd.DataFrame(sbc_rows)
    coverage.to_csv(out / "coverage.csv", index=False)
    sbc.to_csv(out / "sbc_ranks.csv", index=False)
    disagreement_rows = []
    for j, name in enumerate(PARAMETER_NAMES_7D):
        medians = np.stack([
            np.asarray([np.median(samples[case, labels[case] == member, j]) for case in range(256)])
            for member in range(3)
        ], axis=1)
        ranges = np.ptp(medians, axis=1)
        disagreement_rows.append({
            "parameter": name,
            "mean_member_median_range": float(ranges.mean()),
            "median_member_median_range": float(np.median(ranges)),
            "max_member_median_range": float(ranges.max()),
        })
    disagreement = pd.DataFrame(disagreement_rows)
    disagreement.to_csv(out / "ensemble_disagreement.csv", index=False)
    ridges = pd.DataFrame({
        "case": np.arange(256),
        "g_LK_g_h_spearman": [
            float(spearmanr(samples[case, :, 4], samples[case, :, 5]).statistic)
            for case in range(256)
        ],
    })
    ridges.to_csv(out / "posterior_ridges.csv", index=False)
    return {
        "recovery": recovery, "case_metrics": case_metrics,
        "coverage": coverage, "sbc": sbc,
        "disagreement": disagreement, "ridges": ridges,
        "theta": theta, "samples": samples, "labels": labels,
        "posterior_median": posterior_median,
    }


def run_ppc_64(
    recovery: Mapping[str, Any], heldout_path: Path, max_workers: int = 6
) -> dict[str, Any]:
    verify_preregistration()
    verify_heldout_criteria()
    cfg = read_preregistration()["seeds"]
    data = load_training_data_7d()
    with np.load(heldout_path, allow_pickle=False) as heldout:
        heldout_x = np.asarray(heldout["x"], float)
    rng = np.random.default_rng(int(cfg["ppc_case_selection"]))
    random_cases = np.sort(rng.choice(256, size=PPC_RANDOM_CASES, replace=False))
    scores = recovery["case_metrics"].set_index("case")["mean_parameter_median_abs_error"]
    remaining = scores.drop(index=random_cases)
    worst_cases = np.sort(remaining.nlargest(PPC_WORST_CASES).index.to_numpy())
    case_ids = np.concatenate([random_cases, worst_cases])
    case_role = np.asarray(["fixed_random"] * 32 + ["worst_recovery"] * 32, dtype="<U32")
    if len(np.unique(case_ids)) != 64:
        raise RuntimeError("PPC random and worst sets are not disjoint")
    posterior_theta_unit = []
    case_rows = []
    for case in case_ids:
        case_rng = np.random.default_rng(int(cfg["ppc_case_selection"]) + int(case))
        indices = case_rng.choice(4096, size=PPC_DRAWS_PER_CASE, replace=False)
        posterior_theta_unit.extend(recovery["samples"][case, indices])
        case_rows.extend([case] * PPC_DRAWS_PER_CASE)
    posterior_theta_unit = np.asarray(posterior_theta_unit)
    posterior_theta = data.theta_lower + posterior_theta_unit * (data.theta_upper - data.theta_lower)
    posterior_seeds = deterministic_seed_schedule(
        len(posterior_theta), int(cfg["ppc_posterior_simulator_base"])
    )
    out = HELDOUT_ROOT / "ppc"
    post_results = _run_7d_batch(
        posterior_theta, posterior_seeds, out / "posterior_predictive",
        [f"ppc7d_post_{i:04d}" for i in range(len(posterior_theta))], max_workers,
    )
    post_path = _consolidate_7d(
        post_results, posterior_theta, out / "posterior_predictive_1024.npz",
        {"case_id": np.asarray(case_rows, np.int64)},
    )
    prior_theta = sobol_theta7(1024, int(cfg["ppc_prior_sobol"]))
    prior_seeds = deterministic_seed_schedule(1024, int(cfg["ppc_prior_simulator_base"]))
    prior_results = _run_7d_batch(
        prior_theta, prior_seeds, out / "prior_predictive",
        [f"ppc7d_prior_{i:04d}" for i in range(1024)], max_workers,
    )
    prior_path = _consolidate_7d(
        prior_results, prior_theta, out / "prior_predictive_1024.npz",
        {"case_id": np.repeat(case_ids, PPC_DRAWS_PER_CASE).astype(np.int64)},
    )
    posterior_x = np.vstack([row["x"] for row in post_results])
    prior_x = np.vstack([row["x"] for row in prior_results])
    rows = []
    for j, feature in enumerate(rate_feature_names()):
        posterior_errors, prior_errors = [], []
        for position, case in enumerate(case_ids):
            slc = slice(position * 16, (position + 1) * 16)
            observed = heldout_x[case, j]
            posterior_errors.append(np.median(np.abs(posterior_x[slc, j] - observed) / data.x_scale[j]))
            prior_errors.append(np.median(np.abs(prior_x[slc, j] - observed) / data.x_scale[j]))
        post_error = float(np.median(posterior_errors))
        prior_error = float(np.median(prior_errors))
        rows.append({
            "feature": feature,
            "posterior_predictive_median_scaled_abs_error": post_error,
            "prior_predictive_median_scaled_abs_error": prior_error,
            "improvement_fraction": float(1 - post_error / prior_error),
            "posterior_better": post_error < prior_error,
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "ppc_feature_metrics.csv", index=False)
    selection = {
        "random_case_ids": random_cases.tolist(), "worst_case_ids": worst_cases.tolist(),
        "selection_order_case_ids": case_ids.tolist(), "selection_roles": case_role.tolist(),
        "random_selection_seed": int(cfg["ppc_case_selection"]),
        "worst_selection_rule": "top normalized posterior-median recovery score excluding random cases",
        "draws_per_case": 16,
    }
    atomic_json(out / "ppc_case_selection.json", selection)
    post_success = np.asarray([row["success"] for row in post_results], bool)
    prior_success = np.asarray([row["success"] for row in prior_results], bool)
    summary = {
        "cases": 64, "random_cases": 32, "worst_cases": 32,
        "posterior_draws_per_case": 16,
        "posterior_predictive_attempted": 1024,
        "posterior_predictive_failed": int((~post_success).sum()),
        "prior_predictive_attempted": 1024,
        "prior_predictive_failed": int((~prior_success).sum()),
        "features_posterior_better": int(metrics.posterior_better.sum()),
        "overall_posterior_scaled_error": float(metrics.posterior_predictive_median_scaled_abs_error.median()),
        "overall_prior_scaled_error": float(metrics.prior_predictive_median_scaled_abs_error.median()),
        "overall_improvement_fraction": float(
            1 - metrics.posterior_predictive_median_scaled_abs_error.median()
            / metrics.prior_predictive_median_scaled_abs_error.median()
        ),
    }
    atomic_json(out / "ppc_summary.json", summary)
    return {
        "metrics": metrics, "selection": selection, "summary": summary,
        "posterior_path": post_path, "prior_path": prior_path,
    }


def evaluate_decision(
    recovery: Mapping[str, Any], ppc: Mapping[str, Any]
) -> dict[str, Any]:
    criteria = read_preregistration()["decision_criteria"]
    formal = criteria["formal_go"]
    conditional = criteria["conditional_go"]
    training = json.loads((BANK_ROOT / "bank_manifest.json").read_text(encoding="utf-8"))
    preflight = json.loads((PREFLIGHT_ROOT / "engineering_gate.json").read_text(encoding="utf-8"))
    heldout = json.loads(
        (HELDOUT_ROOT / "dataset" / "heldout_manifest.json").read_text(encoding="utf-8")
    )
    coverage = recovery["coverage"]
    ensemble = coverage[coverage.estimator == "ensemble"]
    coverage_80_90 = ensemble[ensemble.nominal_level.isin([.8, .9])]
    compatible = coverage_80_90.groupby("parameter").nominal_inside_wilson.all()
    severe = int(ensemble.groupby("parameter").severe_undercoverage.any().sum())
    recovery_table = recovery["recovery"]
    better = int((recovery_table.mae_improvement_fraction > 0).sum())
    systematic_worse = int(
        (recovery_table.mae_improvement_fraction < formal["systematic_worsening_threshold"]).sum()
    )
    prior_level = int(
        (
            (recovery_table.mae_improvement_fraction <= formal["prior_level_improvement_max"])
            & (recovery_table.rank_correlation.abs() <= formal["prior_level_rank_correlation_max"])
        ).sum()
    )
    overall_prior = float(recovery["case_metrics"].mean_parameter_prior_median_abs_error.mean())
    overall_post = float(recovery["case_metrics"].mean_parameter_median_abs_error.mean())
    overall_improvement = 1 - overall_post / overall_prior
    contracted = set(
        recovery_table.loc[
            recovery_table.median_90_ci_width < formal["meaningful_contraction_width_max"],
            "parameter",
        ]
    )
    reasonable = set(compatible[compatible].index)
    contraction_coverage = len(contracted & reasonable)
    disagreement = recovery["disagreement"]
    disagreement_mean = float(disagreement.mean_member_median_range.mean())
    disagreement_max = float(disagreement.mean_member_median_range.max())
    finite_support_rate = float(
        np.mean(
            np.isfinite(recovery["samples"]).all(axis=2)
            & ((recovery["samples"] >= 0) & (recovery["samples"] <= 1)).all(axis=2)
        )
    )
    checks = {
        "preflight_engineering_gate": bool(preflight["pass"]),
        "training_failure_gate": training["failure_rate"] <= formal["training_failure_rate_max"],
        "heldout_failure_gate": heldout["failure_rate"] <= formal["heldout_failure_rate_max"],
        "posterior_finite_support_rate": finite_support_rate,
        "posterior_finite_support_gate": finite_support_rate >= formal["posterior_finite_and_in_prior_rate_min"],
        "prior_level_unrecoverable_parameter_count": prior_level,
        "prior_level_unrecoverable_gate": prior_level <= formal["prior_level_unrecoverable_parameters_max"],
        "systematically_worse_parameter_count": systematic_worse,
        "systematic_worsening_gate": systematic_worse <= formal["systematically_worse_parameters_max"],
        "better_than_prior_parameter_count": better,
        "better_than_prior_gate": better >= formal["median_error_better_than_prior_parameters_min"],
        "overall_recovery_improvement_fraction": overall_improvement,
        "overall_recovery_gate": overall_improvement >= formal["overall_median_error_improvement_min_fraction"],
        "coverage_compatible_parameter_count": int(compatible.sum()),
        "coverage_compatible_gate": int(compatible.sum()) >= formal["coverage_80_or_90_wilson_contains_nominal_parameters_min"],
        "severe_undercoverage_parameter_count": severe,
        "severe_undercoverage_gate": severe <= formal["severe_undercoverage_parameters_max"],
        "contraction_with_coverage_parameter_count": contraction_coverage,
        "contraction_with_coverage_gate": contraction_coverage >= formal["meaningful_contraction_with_reasonable_coverage_parameters_min"],
        "ppc_better_feature_count": int(ppc["summary"]["features_posterior_better"]),
        "ppc_feature_gate": int(ppc["summary"]["features_posterior_better"]) >= formal["ppc_features_better_than_prior_min"],
        "overall_ppc_improvement_fraction": float(ppc["summary"]["overall_improvement_fraction"]),
        "overall_ppc_gate": float(ppc["summary"]["overall_improvement_fraction"]) >= formal["overall_ppc_error_improvement_min_fraction"],
        "ensemble_disagreement_mean": disagreement_mean,
        "ensemble_disagreement_parameter_max": disagreement_max,
        "ensemble_disagreement_gate": (
            disagreement_mean <= formal["ensemble_member_median_disagreement_mean_max_prior_width"]
            and disagreement_max <= formal["ensemble_member_disagreement_parameter_max_prior_width"]
        ),
    }
    formal_go = all(value for key, value in checks.items() if key.endswith("_gate"))
    conditional_go = (
        checks["preflight_engineering_gate"]
        and checks["training_failure_gate"]
        and checks["heldout_failure_gate"]
        and checks["posterior_finite_support_gate"]
        and severe <= conditional["severe_undercoverage_parameters_max"]
        and better >= conditional["median_error_better_than_prior_parameters_min"]
        and contraction_coverage >= conditional["meaningful_contraction_with_reasonable_coverage_parameters_min"]
        and ppc["summary"]["overall_improvement_fraction"] > 0
        and checks["ensemble_disagreement_gate"]
    )
    decision = "GO" if formal_go else "CONDITIONAL GO" if conditional_go else "NO-GO"
    result = {
        "criteria_version": criteria["version"],
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision, "checks": checks,
        "scope": "seven free parameters conditional on fixed c_ctx2th in synthetic cortical-rate space",
        "fixed_c_ctx2th": fixed_c_ctx2th(),
        "real_eeg_inference_unlocked": False,
        "measurement_model_blocker_resolved": False,
        "preregistration_hash": verify_preregistration(),
        "heldout_criteria_hash": verify_heldout_criteria(),
    }
    atomic_json(HELDOUT_ROOT / "formal_route3_7d_decision.json", result)
    return result
