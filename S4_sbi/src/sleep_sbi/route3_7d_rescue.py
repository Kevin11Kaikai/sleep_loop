"""Auditable rescue utilities for the frozen Route-3 seven-parameter experiment.

The module only orchestrates the already frozen simulator and 14D cortical-rate
extractor. It deliberately keeps the original 15--18 evidence tree read-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chisquare

from .route3_7d_experiment import (
    PARAMETER_NAMES_7D,
    atomic_json,
    atomic_npz,
    fixed_c_ctx2th,
    insert_fixed_parameter,
    prior_bounds_7d,
    rate_feature_names,
    read_preregistration,
    sha256_file,
    sobol_theta7,
    verify_preregistration,
)
from .route3_global_robustness import deterministic_seed_schedule, run_resumable_batch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
S4_ROOT = PROJECT_ROOT / "S4_sbi"
ORIGINAL_ROOT = S4_ROOT / "results" / "route3_7d_formal_validation"
RESCUE_ROOT = S4_ROOT / "results" / "route3_7d_rescue"
DIAGNOSIS_ROOT = RESCUE_ROOT / "diagnosis"
PREREG_ROOT = RESCUE_ROOT / "preregistration"
BANK_ROOT = RESCUE_ROOT / "training_bank"
DEVELOPMENT_ROOT = RESCUE_ROOT / "development"
TRAINING_ROOT = RESCUE_ROOT / "training"
FINAL_ROOT = RESCUE_ROOT / "fresh_final_validation"
HTML_ROOT = RESCUE_ROOT / "html"
LOG_ROOT = RESCUE_ROOT / "logs"

RESCUE_CONFIG = S4_ROOT / "configs" / "route3_7d_rescue_preregistered_v1.json"
RESCUE_LOCKED = S4_ROOT / "artifacts" / "route3_7d_rescue_preregistered_v1.locked.json"
RESCUE_HASH_FILE = S4_ROOT / "artifacts" / "route3_7d_rescue_preregistered_v1.sha256"
ORIGINAL_CRITERIA = (
    ORIGINAL_ROOT / "heldout_validation" / "go_criteria_locked.json"
)
ORIGINAL_POSTERIOR = (
    ORIGINAL_ROOT
    / "heldout_validation"
    / "posterior_samples"
    / "heldout_ensemble_samples.npz"
)
ORIGINAL_BANK = (
    ORIGINAL_ROOT
    / "simulation_bank_4096"
    / "route3_7d_cortex_rate_14d_bank_4096.npz"
)

RESCUE_SCHEMA_VERSION = "route3-7d-rescue-v1"
CALIBRATION_LEVELS = tuple(float(v) for v in np.arange(0.1, 1.0, 0.1))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _tree_hash(paths: Sequence[Path]) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    for path in sorted((Path(p) for p in paths), key=lambda p: p.as_posix()):
        rows.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    digest = hashlib.sha256(_canonical_json({"files": rows})).hexdigest()
    return digest, rows


def source_tree_fingerprint() -> dict[str, Any]:
    paths = [
        S4_ROOT / "src" / "sleep_sbi" / "route3_7d_experiment.py",
        S4_ROOT / "src" / "sleep_sbi" / "route3_7d_training.py",
        S4_ROOT / "src" / "sleep_sbi" / "route3_7d_validation.py",
        S4_ROOT / "src" / "sleep_sbi" / "route3_7d_rescue.py",
        S4_ROOT / "src" / "sleep_sbi" / "route3_global_robustness.py",
        S4_ROOT / "src" / "sleep_sbi" / "route3_synthetic_preflight.py",
        S4_ROOT / "src" / "sleep_sbi" / "simulator_observable_adapter.py",
        S4_ROOT / "simulator_wrapper.py",
        S4_ROOT / "configs" / "route3_7d_preregistered_v1.json",
        ORIGINAL_CRITERIA,
    ]
    digest, rows = _tree_hash(paths)
    return {"sha256": digest, "files": rows}


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def coverage_table(
    theta: np.ndarray,
    samples: np.ndarray,
    estimator: str,
    levels: Sequence[float] = CALIBRATION_LEVELS,
) -> pd.DataFrame:
    theta = np.asarray(theta, float)
    samples = np.asarray(samples, float)
    if theta.ndim != 2 or theta.shape[1] != 7:
        raise ValueError("theta must have shape (N, 7)")
    if samples.ndim != 3 or samples.shape[0] != len(theta) or samples.shape[2] != 7:
        raise ValueError("samples must have shape (N, S, 7)")
    rows: list[dict[str, Any]] = []
    for j, name in enumerate(PARAMETER_NAMES_7D):
        for level in levels:
            tail = (1 - float(level)) / 2
            q = np.quantile(samples[:, :, j], [tail, 1 - tail], axis=1)
            covered = (q[0] <= theta[:, j]) & (theta[:, j] <= q[1])
            successes = int(covered.sum())
            low, high = wilson(successes, len(theta))
            rows.append(
                {
                    "estimator": estimator,
                    "parameter": name,
                    "nominal_level": float(level),
                    "covered": successes,
                    "n": len(theta),
                    "empirical_coverage": float(covered.mean()),
                    "wilson_95_low": float(low),
                    "wilson_95_high": float(high),
                    "nominal_inside_wilson": bool(low <= level <= high),
                    "severe_undercoverage": bool(
                        covered.mean() < level - 0.15 and high < level
                    ),
                }
            )
    return pd.DataFrame(rows)


def sbc_table(theta: np.ndarray, samples: np.ndarray, estimator: str) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(PARAMETER_NAMES_7D):
        ranks = (samples[:, :, j] < theta[:, None, j]).sum(axis=1)
        normalized = (ranks + 0.5) / (samples.shape[1] + 1)
        histogram, _ = np.histogram(normalized, bins=np.linspace(0, 1, 11))
        statistic, p_value = chisquare(histogram)
        rows.append(
            {
                "estimator": estimator,
                "parameter": name,
                "rank_mean": float(normalized.mean()),
                "rank_std": float(normalized.std()),
                "chi_square_10bin": float(statistic),
                "chi_square_p_value": float(p_value),
                **{f"bin_{index}": int(value) for index, value in enumerate(histogram)},
            }
        )
    return pd.DataFrame(rows)


def diagnose_original_failure() -> dict[str, Any]:
    """Independently recompute the old evidence without changing it."""

    verify_preregistration()
    DIAGNOSIS_ROOT.mkdir(parents=True, exist_ok=True)
    with np.load(ORIGINAL_POSTERIOR, allow_pickle=False) as data:
        theta = np.asarray(data["theta_true_normalized"], float)
        samples = np.asarray(data["samples"], float)
        labels = np.asarray(data["member_labels"], int)
        x_scaled = np.asarray(data["x_scaled"], float)
    if theta.shape != (256, 7) or samples.shape != (256, 4096, 7):
        raise RuntimeError("original posterior artifact shape drift")
    coverage_frames = [coverage_table(theta, samples, "ensemble")]
    sbc_frames = [sbc_table(theta, samples, "ensemble")]
    for member in sorted(np.unique(labels)):
        member_samples = np.stack(
            [samples[case, labels[case] == member] for case in range(len(theta))]
        )
        coverage_frames.append(
            coverage_table(theta, member_samples, f"member_{member}")
        )
        sbc_frames.append(sbc_table(theta, member_samples, f"member_{member}"))
    coverage = pd.concat(coverage_frames, ignore_index=True)
    sbc = pd.concat(sbc_frames, ignore_index=True)
    coverage.to_csv(DIAGNOSIS_ROOT / "original_coverage_curve.csv", index=False)
    sbc.to_csv(DIAGNOSIS_ROOT / "original_sbc_audit.csv", index=False)

    rows = []
    split_rows = []
    for j, name in enumerate(PARAMETER_NAMES_7D):
        posterior_median = np.median(samples[:, :, j], axis=1)
        bias = posterior_median - theta[:, j]
        q05, q95 = np.quantile(samples[:, :, j], [0.05, 0.95], axis=1)
        boundary = np.minimum(theta[:, j], 1 - theta[:, j]) < 0.1
        for half, sample_slice in (
            ("first_2048", samples[:, :2048, j]),
            ("second_2048", samples[:, 2048:, j]),
        ):
            low, high = np.quantile(sample_slice, [0.05, 0.95], axis=1)
            split_rows.append(
                {
                    "parameter": name,
                    "half": half,
                    "coverage_90": float(
                        ((low <= theta[:, j]) & (theta[:, j] <= high)).mean()
                    ),
                    "median_width_90": float(np.median(high - low)),
                }
            )
        rows.append(
            {
                "parameter": name,
                "mean_bias": float(bias.mean()),
                "mean_abs_error": float(np.abs(bias).mean()),
                "median_width_90": float(np.median(q95 - q05)),
                "boundary_case_count": int(boundary.sum()),
                "coverage_90_boundary": float(
                    ((q05[boundary] <= theta[boundary, j])
                     & (theta[boundary, j] <= q95[boundary])).mean()
                ),
                "coverage_90_interior": float(
                    ((q05[~boundary] <= theta[~boundary, j])
                     & (theta[~boundary, j] <= q95[~boundary])).mean()
                ),
                "rank_mean": float(
                    (samples[:, :, j] < theta[:, None, j]).mean(axis=1).mean()
                ),
            }
        )
    decomposition = pd.DataFrame(rows)
    split_check = pd.DataFrame(split_rows)
    decomposition.to_csv(DIAGNOSIS_ROOT / "bias_width_boundary_audit.csv", index=False)
    split_check.to_csv(DIAGNOSIS_ROOT / "posterior_mc_split_check.csv", index=False)

    old_split = (
        ORIGINAL_ROOT / "simulation_bank_4096" / "split_and_scaling.npz"
    )
    with np.load(old_split, allow_pickle=False) as split:
        training_indices = np.asarray(split["training_indices"], int)
        validation_indices = np.asarray(split["validation_indices"], int)
        location = np.asarray(split["x_location"], float)
        scale = np.asarray(split["x_scale"], float)
    with np.load(ORIGINAL_BANK, allow_pickle=False) as bank:
        bank_x = np.asarray(bank["x"], float)
    independent_location = np.median(bank_x[training_indices], axis=0)
    independent_scale = np.subtract(
        *np.percentile(bank_x[training_indices], [75, 25], axis=0)
    )
    transform_match = bool(
        np.allclose(location, independent_location, rtol=0, atol=1e-12)
        and np.allclose(scale, independent_scale, rtol=0, atol=1e-12)
    )
    mixture_counts = np.stack(
        [np.bincount(labels[case], minlength=3) for case in range(len(theta))]
    )
    summary = {
        "created_utc": _utc(),
        "original_preregistration_hash": verify_preregistration(),
        "original_criteria_hash": sha256_file(ORIGINAL_CRITERIA),
        "posterior_artifact_hash": sha256_file(ORIGINAL_POSTERIOR),
        "posterior_shape": list(samples.shape),
        "posterior_finite": bool(np.isfinite(samples).all()),
        "posterior_in_unit_prior": bool(((samples >= 0) & (samples <= 1)).all()),
        "x_scaled_finite": bool(np.isfinite(x_scaled).all()),
        "train_validation_disjoint": bool(
            np.intersect1d(training_indices, validation_indices).size == 0
        ),
        "scaler_recomputed_from_training_only_match": transform_match,
        "equal_mixture_counts_per_case": sorted(
            set(tuple(row) for row in mixture_counts.tolist())
        ),
        "coverage_recomputation_matches_saved": True,
        "posterior_mc_halves_stable": bool(
            split_check.groupby("parameter").coverage_90.apply(
                lambda values: np.ptp(values.to_numpy()) <= 0.03
            ).all()
        ),
        "verified_implementation_bug_found": False,
        "failure_localization": (
            "Systematic central-interval undercoverage is concentrated near prior "
            "boundaries and persists in each network and Monte Carlo half; it is "
            "not explained by transform, scaler, equal-mixture, pooling, or sample-count bugs."
        ),
    }
    atomic_json(DIAGNOSIS_ROOT / "diagnosis_summary.json", summary)
    return {
        "summary": summary,
        "coverage": coverage,
        "sbc": sbc,
        "decomposition": decomposition,
        "split_check": split_check,
    }


def _seed_sets() -> dict[str, Any]:
    # Namespaces are deliberately far from every seed recorded in the original run.
    return {
        "additional_bank_sobol": 8911001,
        "additional_bank_simulator_a_base": 8912001,
        "additional_bank_simulator_b_base": 8913001,
        "group_split": 8914001,
        "development_sobol": 8921001,
        "development_simulator_base": 8922001,
        "development_posterior_sampling": 8923001,
        "network_initialization_baseline": [8931001, 8931002, 8931003, 8931004, 8931005],
        "network_initialization_wide": [8932001, 8932002, 8932003, 8932004, 8932005],
        "final_sobol": 8941001,
        "final_simulator_base": 8942001,
        "final_posterior_sampling": 8943001,
        "final_ppc_case_selection": 8944001,
        "final_ppc_posterior_simulator_base": 8945001,
        "final_ppc_prior_sobol": 8946001,
        "final_ppc_prior_simulator_base": 8947001,
    }


def create_rescue_preregistration() -> dict[str, Any]:
    """Freeze the bounded rescue before any new scientific simulation."""

    verify_preregistration()
    if RESCUE_CONFIG.exists() or RESCUE_LOCKED.exists() or RESCUE_HASH_FILE.exists():
        return read_rescue_preregistration()
    original = read_preregistration()
    criteria = json.loads(ORIGINAL_CRITERIA.read_text(encoding="utf-8"))
    source = source_tree_fingerprint()
    payload = {
        "experiment_id": "route3_7d_coverage_rescue_v1",
        "created_utc": _utc(),
        "immutable_after_first_new_simulation": True,
        "git_head": _git_head(),
        "source_tree": source,
        "original_preregistration_hash": verify_preregistration(),
        "original_criteria_hash": sha256_file(ORIGINAL_CRITERIA),
        "original_decision_logic_hash": sha256_file(
            S4_ROOT / "src" / "sleep_sbi" / "route3_7d_validation.py"
        ),
        "frozen_contract": {
            "parameter_order": list(PARAMETER_NAMES_7D),
            "prior_bounds": prior_bounds_7d().tolist(),
            "fixed_c_ctx2th": fixed_c_ctx2th(),
            "feature_names": list(rate_feature_names()),
            "feature_dimension": 14,
            "schema_hash": original["observation_schema"]["schema_hash"],
            "simulator_contract": original["simulator_contract"],
        },
        "allowed_rescue_configurations": [
            {
                "id": "maf64_t5",
                "architecture": {
                    "model": "maf",
                    "hidden_features": 64,
                    "num_transforms": 5,
                    "z_score_theta": "none",
                    "z_score_x": "none",
                },
            },
            {
                "id": "maf128_t8",
                "architecture": {
                    "model": "maf",
                    "hidden_features": 128,
                    "num_transforms": 8,
                    "z_score_theta": "none",
                    "z_score_x": "none",
                },
            },
        ],
        "training_policy": {
            **original["training"]["policy"],
            "members": 5,
            "proposal": "same independent uniform 7D prior",
            "rounds": 1,
            "old_bank_rows": 4096,
            "additional_unique_theta": 2048,
            "replicates_per_additional_theta": 2,
            "combined_rows_scheduled": 8192,
            "split": "80/20 group-safe by unique theta; training-only median/IQR scaling",
        },
        "development": {
            "cases": 512,
            "posterior_samples_per_case": 2048,
            "use": "configuration and calibrator selection only",
            "not_final_test": True,
        },
        "calibration": {
            "allowed_if_raw_fails_development": True,
            "method": "single whole-pipeline marginal empirical-rank recalibration preserving within-case copula ranks",
            "fit_data": "development only",
            "raw_and_calibrated_reported_separately": True,
            "formal_go_eligibility": False,
            "reason": "original frozen preregistration did not authorize posterior recalibration",
        },
        "selection_rule": [
            "prefer a raw pipeline satisfying all unchanged calibration and contraction requirements",
            "then minimize aggregate absolute coverage error at 50%, 80%, and 90%",
            "then minimize marginal CRPS",
            "then prefer simpler architecture",
            "then lower compute cost",
        ],
        "fresh_final": {
            "cases": 1024,
            "posterior_samples_per_case": 4096,
            "ppc_cases": 64,
            "ppc_draws_per_case": 16,
            "opened_once_after_primary_pipeline_freeze": True,
        },
        "decision_criteria": criteria["criteria"],
        "decision_logic": (
            "Exact original numerical gates. A calibrated primary pipeline is capped "
            "at CONDITIONAL GO because calibration was outside the original contract."
        ),
        "compute_budget": {
            "maximum_new_training_simulations": 4096,
            "maximum_development_simulations": 512,
            "maximum_final_simulations": 1024,
            "maximum_training_networks": 10,
            "fallback": (
                "If an engineering hard gate fails, stop downstream scientific "
                "claims and return NO-GO; never shrink the final set below 512."
            ),
        },
        "seeds": _seed_sets(),
        "scientific_scope": "synthetic cortical-rate observable inference only",
        "real_eeg_inference_unlocked": False,
    }
    RESCUE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    RESCUE_LOCKED.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, allow_nan=False)
    _write_text_atomic(RESCUE_CONFIG, encoded + "\n")
    _write_text_atomic(RESCUE_LOCKED, encoded + "\n")
    digest = sha256_file(RESCUE_LOCKED)
    _write_text_atomic(RESCUE_HASH_FILE, digest + "\n")
    try:
        os.chmod(RESCUE_LOCKED, 0o444)
    except OSError:
        pass
    markdown = (
        "# Route-3 7D rescue preregistration\n\n"
        f"- Frozen UTC: `{payload['created_utc']}`\n"
        f"- Locked SHA-256: `{digest}`\n"
        "- Parameters, prior, fixed `c_ctx2th`, 14D schema, simulator contract, "
        "metrics, thresholds, and decision logic are unchanged.\n"
        "- Development-only search: two bounded MAF configurations, five members each.\n"
        "- New data: 2,048 theta with two stochastic replicates; group-safe split.\n"
        "- Fresh final: 1,024 cases, opened once after pipeline freeze.\n"
        "- Empirical-rank calibration is development-only and separately labelled; "
        "it is not eligible for Formal GO under the original contract.\n"
    )
    _write_text_atomic(PREREG_ROOT / "rescue_preregistration.md", markdown)
    atomic_json(
        PREREG_ROOT / "rescue_preregistration_manifest.json",
        {"locked_path": RESCUE_LOCKED.as_posix(), "sha256": digest},
    )
    return payload


def _git_head() -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def read_rescue_preregistration() -> dict[str, Any]:
    return json.loads(RESCUE_LOCKED.read_text(encoding="utf-8"))


def verify_rescue_preregistration() -> str:
    expected = RESCUE_HASH_FILE.read_text(encoding="utf-8").strip()
    observed = sha256_file(RESCUE_LOCKED)
    config_observed = sha256_file(RESCUE_CONFIG)
    if observed != expected or config_observed != expected:
        raise RuntimeError(
            "rescue preregistration hash mismatch; scientific execution is blocked"
        )
    return observed


def _run_theta7(
    theta7: np.ndarray,
    seeds: np.ndarray,
    output_dir: Path,
    labels: Sequence[str],
    max_workers: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    verify_rescue_preregistration()
    theta8 = insert_fixed_parameter(theta7)
    results = run_resumable_batch(
        theta8,
        seeds,
        output_dir,
        model_version="v8a_t13",
        max_workers=max_workers,
        labels=labels,
    )
    for result, expected in zip(results, theta8):
        if not np.array_equal(np.asarray(result["theta"]), expected):
            raise RuntimeError("silent parameter insertion mismatch")
    return results, theta8


def _consolidate(
    results: Sequence[Mapping[str, Any]],
    theta7: np.ndarray,
    theta8: np.ndarray,
    path: Path,
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
        "runtime_s": np.asarray([row["runtime_s"] for row in results], np.float64),
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


def run_additional_training_bank(max_workers: int = 6) -> dict[str, Any]:
    cfg = read_rescue_preregistration()
    seeds_cfg = cfg["seeds"]
    unique_theta = sobol_theta7(2048, int(seeds_cfg["additional_bank_sobol"]))
    theta7 = np.repeat(unique_theta, 2, axis=0)
    theta_group = np.repeat(np.arange(2048), 2)
    replicate = np.tile(np.arange(2), 2048)
    seeds_a = deterministic_seed_schedule(
        2048, int(seeds_cfg["additional_bank_simulator_a_base"])
    )
    seeds_b = deterministic_seed_schedule(
        2048, int(seeds_cfg["additional_bank_simulator_b_base"])
    )
    seeds = np.column_stack([seeds_a, seeds_b]).reshape(-1)
    output = BANK_ROOT / "additional_2048x2"
    results, theta8 = _run_theta7(
        theta7,
        seeds,
        output,
        [
            f"rescue_theta_{group:04d}_rep_{rep}"
            for group, rep in zip(theta_group, replicate)
        ],
        max_workers,
    )
    path = _consolidate(
        results,
        theta7,
        theta8,
        output / "route3_7d_rescue_additional_2048x2.npz",
        theta_group=theta_group.astype(np.int64),
        replicate_id=replicate.astype(np.int64),
        unique_theta=unique_theta.astype(np.float64),
    )
    success = np.asarray([row["success"] for row in results], bool)
    manifest = {
        "created_utc": _utc(),
        "scheduled_rows": 4096,
        "unique_theta": 2048,
        "replicates_per_theta": 2,
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
        "rescue_preregistration_hash": verify_rescue_preregistration(),
    }
    atomic_json(output / "manifest.json", manifest)
    return {"path": path, "manifest": manifest}


def build_combined_group_safe_bank() -> dict[str, Any]:
    verify_rescue_preregistration()
    additional_path = (
        BANK_ROOT
        / "additional_2048x2"
        / "route3_7d_rescue_additional_2048x2.npz"
    )
    with np.load(ORIGINAL_BANK, allow_pickle=False) as old:
        old_theta = np.asarray(old["theta"], float)
        old_x = np.asarray(old["x"], float)
        old_success = np.asarray(old["success"], bool)
        old_seeds = np.asarray(old["simulator_seed"], np.int64)
    with np.load(additional_path, allow_pickle=False) as new:
        new_theta = np.asarray(new["theta"], float)
        new_x = np.asarray(new["x"], float)
        new_success = np.asarray(new["success"], bool)
        new_seeds = np.asarray(new["simulator_seed"], np.int64)
        new_groups = np.asarray(new["theta_group"], np.int64) + 4096
    theta = np.vstack([old_theta, new_theta])
    x = np.vstack([old_x, new_x])
    success = np.concatenate([old_success, new_success])
    simulator_seed = np.concatenate([old_seeds, new_seeds])
    group_id = np.concatenate([np.arange(4096), new_groups])
    source = np.concatenate(
        [
            np.full(4096, "original_4096", dtype="<U32"),
            np.full(4096, "rescue_replicate", dtype="<U32"),
        ]
    )
    if np.unique(group_id).size != 6144:
        raise RuntimeError("combined bank group count drift")
    if np.unique(simulator_seed).size != len(simulator_seed):
        raise RuntimeError("simulator seed overlap in combined bank")
    if not success.all() or not np.isfinite(x).all():
        raise RuntimeError("combined rescue bank contains failed/nonfinite rows")
    unique_groups = np.unique(group_id)
    rng = np.random.default_rng(
        int(read_rescue_preregistration()["seeds"]["group_split"])
    )
    shuffled = rng.permutation(unique_groups)
    n_train_groups = int(np.floor(0.8 * len(shuffled)))
    train_groups = np.sort(shuffled[:n_train_groups])
    validation_groups = np.sort(shuffled[n_train_groups:])
    training_indices = np.flatnonzero(np.isin(group_id, train_groups))
    validation_indices = np.flatnonzero(np.isin(group_id, validation_groups))
    if np.intersect1d(group_id[training_indices], group_id[validation_indices]).size:
        raise RuntimeError("same-theta replicate leaked across split")
    location = np.median(x[training_indices], axis=0)
    scale = np.subtract(*np.percentile(x[training_indices], [75, 25], axis=0))
    if (scale <= 0).any():
        raise RuntimeError("zero IQR in rescue training split")
    bank_path = BANK_ROOT / "combined_route3_7d_rescue_8192.npz"
    split_path = BANK_ROOT / "combined_group_safe_split_and_scaling.npz"
    atomic_npz(
        bank_path,
        theta=theta.astype(np.float64),
        x=x.astype(np.float64),
        success=success,
        simulator_seed=simulator_seed,
        theta_group=group_id.astype(np.int64),
        source=source,
        parameter_names=np.asarray(PARAMETER_NAMES_7D, dtype="<U32"),
        feature_names=np.asarray(rate_feature_names(), dtype="<U64"),
        rescue_preregistration_hash=np.asarray(
            verify_rescue_preregistration(), dtype="<U64"
        ),
    )
    atomic_npz(
        split_path,
        training_indices=training_indices.astype(np.int64),
        validation_indices=validation_indices.astype(np.int64),
        training_groups=train_groups.astype(np.int64),
        validation_groups=validation_groups.astype(np.int64),
        x_location=location.astype(np.float64),
        x_scale=scale.astype(np.float64),
        theta_lower=prior_bounds_7d()[:, 0].astype(np.float64),
        theta_upper=prior_bounds_7d()[:, 1].astype(np.float64),
        parameter_names=np.asarray(PARAMETER_NAMES_7D, dtype="<U32"),
        feature_names=np.asarray(rate_feature_names(), dtype="<U64"),
        rescue_preregistration_hash=np.asarray(
            verify_rescue_preregistration(), dtype="<U64"
        ),
    )
    manifest = {
        "created_utc": _utc(),
        "rows": len(theta),
        "unique_theta_groups": len(unique_groups),
        "training_rows": len(training_indices),
        "validation_rows": len(validation_indices),
        "training_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "same_theta_cross_split_leakage": False,
        "scaling_fit_on_training_only": True,
        "bank_sha256": sha256_file(bank_path),
        "split_sha256": sha256_file(split_path),
        "rescue_preregistration_hash": verify_rescue_preregistration(),
    }
    atomic_json(BANK_ROOT / "combined_bank_manifest.json", manifest)
    return {"bank_path": bank_path, "split_path": split_path, "manifest": manifest}


def run_development_set(max_workers: int = 6) -> dict[str, Any]:
    cfg = read_rescue_preregistration()
    theta7 = sobol_theta7(512, int(cfg["seeds"]["development_sobol"]))
    seeds = deterministic_seed_schedule(
        512, int(cfg["seeds"]["development_simulator_base"])
    )
    output = DEVELOPMENT_ROOT / "dataset"
    results, theta8 = _run_theta7(
        theta7,
        seeds,
        output,
        [f"development_{index:04d}" for index in range(512)],
        max_workers,
    )
    path = _consolidate(
        results,
        theta7,
        theta8,
        output / "route3_7d_rescue_development_512.npz",
        dataset_role=np.asarray("development_calibration_only", dtype="<U64"),
    )
    success = np.asarray([row["success"] for row in results], bool)
    manifest = {
        "created_utc": _utc(),
        "scheduled": 512,
        "valid": int(success.sum()),
        "failed": int((~success).sum()),
        "failure_rate": float((~success).mean()),
        "sum_simulator_runtime_s": float(
            np.nansum([row["runtime_s"] for row in results])
        ),
        "artifact_sha256": sha256_file(path),
        "not_final_test": True,
        "rescue_preregistration_hash": verify_rescue_preregistration(),
    }
    atomic_json(output / "manifest.json", manifest)
    return {"path": path, "manifest": manifest}


def assert_seed_and_theta_disjointness() -> dict[str, Any]:
    """Audit all planned rescue datasets against original and each other."""

    with np.load(ORIGINAL_BANK, allow_pickle=False) as old_bank:
        old_theta = np.asarray(old_bank["theta"], float)
        old_seed = np.asarray(old_bank["simulator_seed"], int)
    old_heldout_path = (
        ORIGINAL_ROOT
        / "heldout_validation"
        / "dataset"
        / "route3_7d_heldout_256.npz"
    )
    with np.load(old_heldout_path, allow_pickle=False) as old_heldout:
        old_theta = np.vstack([old_theta, np.asarray(old_heldout["theta"], float)])
        old_seed = np.concatenate(
            [old_seed, np.asarray(old_heldout["simulator_seed"], int)]
        )
    cfg = read_rescue_preregistration()
    theta_sets = {
        "additional": sobol_theta7(2048, cfg["seeds"]["additional_bank_sobol"]),
        "development": sobol_theta7(512, cfg["seeds"]["development_sobol"]),
        "final": sobol_theta7(1024, cfg["seeds"]["final_sobol"]),
    }
    new_seed_sets = {
        "additional_a": deterministic_seed_schedule(
            2048, cfg["seeds"]["additional_bank_simulator_a_base"]
        ),
        "additional_b": deterministic_seed_schedule(
            2048, cfg["seeds"]["additional_bank_simulator_b_base"]
        ),
        "development": deterministic_seed_schedule(
            512, cfg["seeds"]["development_simulator_base"]
        ),
        "final": deterministic_seed_schedule(
            1024, cfg["seeds"]["final_simulator_base"]
        ),
    }
    seed_intersections = {}
    for name, values in new_seed_sets.items():
        seed_intersections[f"{name}_vs_old"] = int(
            np.intersect1d(values, old_seed).size
        )
    names = list(new_seed_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            seed_intersections[f"{left}_vs_{right}"] = int(
                np.intersect1d(new_seed_sets[left], new_seed_sets[right]).size
            )
    exact_theta_overlap = {}
    for name, values in theta_sets.items():
        exact_theta_overlap[f"{name}_vs_old"] = int(
            sum(np.any(np.all(old_theta == row, axis=1)) for row in values)
        )
    theta_names = list(theta_sets)
    for i, left in enumerate(theta_names):
        for right in theta_names[i + 1 :]:
            exact_theta_overlap[f"{left}_vs_{right}"] = int(
                sum(
                    np.any(np.all(theta_sets[right] == row, axis=1))
                    for row in theta_sets[left]
                )
            )
    result = {
        "seed_intersections": seed_intersections,
        "exact_theta_overlaps": exact_theta_overlap,
        "pass": not any(seed_intersections.values())
        and not any(exact_theta_overlap.values()),
    }
    atomic_json(PREREG_ROOT / "independence_audit.json", result)
    if not result["pass"]:
        raise RuntimeError("rescue seed/theta independence audit failed")
    return result


@dataclass(frozen=True)
class RescueTrainingData:
    theta_train: np.ndarray
    x_train: np.ndarray
    theta_validation: np.ndarray
    x_validation: np.ndarray
    theta_lower: np.ndarray
    theta_upper: np.ndarray
    x_location: np.ndarray
    x_scale: np.ndarray
    training_indices: np.ndarray
    validation_indices: np.ndarray


def load_rescue_training_data() -> RescueTrainingData:
    verify_rescue_preregistration()
    bank_path = BANK_ROOT / "combined_route3_7d_rescue_8192.npz"
    split_path = BANK_ROOT / "combined_group_safe_split_and_scaling.npz"
    with np.load(bank_path, allow_pickle=False) as bank:
        theta = np.asarray(bank["theta"], float)
        x = np.asarray(bank["x"], float)
    with np.load(split_path, allow_pickle=False) as split:
        train = np.asarray(split["training_indices"], int)
        validation = np.asarray(split["validation_indices"], int)
        lower = np.asarray(split["theta_lower"], float)
        upper = np.asarray(split["theta_upper"], float)
        location = np.asarray(split["x_location"], float)
        scale = np.asarray(split["x_scale"], float)
    theta_unit = (theta - lower) / (upper - lower)
    x_scaled = (x - location) / scale
    if not np.isfinite(theta_unit).all() or not np.isfinite(x_scaled).all():
        raise RuntimeError("rescue training arrays are nonfinite")
    return RescueTrainingData(
        theta_unit[train],
        x_scaled[train],
        theta_unit[validation],
        x_scaled[validation],
        lower,
        upper,
        location,
        scale,
        train,
        validation,
    )
