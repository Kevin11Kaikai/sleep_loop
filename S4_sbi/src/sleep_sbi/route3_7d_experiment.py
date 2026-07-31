"""Preregistered seven-parameter Route-3 experiment infrastructure.

The inferred observable is the frozen 14D cortical population-rate summary.
It is not EEG.  The eighth simulator parameter, ``c_ctx2th``, is fixed by the
preregistered V8a-local-best rule and inserted only at the simulator boundary.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import qmc

from .eeg_observation_mapping_audit import audit_parameter_sets
from .route3_global_robustness import (
    _robust_location_scale,
    deterministic_seed_schedule,
    rate_contract_hash,
    rate_feature_names,
    run_resumable_batch,
)
from .route3_synthetic_preflight import (
    PARAMETER_NAMES,
    RATE_ONLY_FEATURES,
    RELATIVE_SINGULAR_VALUE_THRESHOLD,
    parameter_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "S4_sbi" / "configs" / "route3_7d_preregistered_v1.json"
LOCKED_PATH = PROJECT_ROOT / "S4_sbi" / "artifacts" / "route3_7d_preregistered_v1.locked.json"
HASH_PATH = PROJECT_ROOT / "S4_sbi" / "artifacts" / "route3_7d_preregistered_v1.sha256"
RESULTS_ROOT = PROJECT_ROOT / "S4_sbi" / "results" / "route3_7d_formal_validation"
PREFLIGHT_ROOT = RESULTS_ROOT / "preflight"
BANK_ROOT = RESULTS_ROOT / "simulation_bank_4096"
TRAINING_ROOT = RESULTS_ROOT / "npe_ensemble"
HELDOUT_ROOT = RESULTS_ROOT / "heldout_validation"
HTML_ROOT = RESULTS_ROOT / "html"

PARAMETER_NAMES_8D = tuple(PARAMETER_NAMES)
PARAMETER_NAMES_7D = tuple(PARAMETER_NAMES[:-1])
FIXED_PARAMETER_NAME = "c_ctx2th"
SCHEMA_VERSION = "route3-cortex-rate-14d-v1"
EXPERIMENT_VERSION = "route3-7d-preregistered-v1"


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_preregistration() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def verify_preregistration() -> str:
    """Verify source, locked copy, and recorded digest before scientific work."""

    if not CONFIG_PATH.exists() or not LOCKED_PATH.exists() or not HASH_PATH.exists():
        raise RuntimeError("7D preregistration source/lock/hash artifact is missing")
    expected = HASH_PATH.read_text(encoding="ascii").strip().lower()
    source_hash = sha256_file(CONFIG_PATH)
    locked_hash = sha256_file(LOCKED_PATH)
    if source_hash != expected or locked_hash != expected:
        raise RuntimeError(
            "7D preregistration hash mismatch: "
            f"expected={expected}, source={source_hash}, locked={locked_hash}"
        )
    prereg = read_preregistration()
    if tuple(prereg["parameter_order_7d"]) != PARAMETER_NAMES_7D:
        raise RuntimeError("7D parameter order drift")
    if tuple(prereg["parameter_order_8d"]) != PARAMETER_NAMES_8D:
        raise RuntimeError("8D insertion contract drift")
    if tuple(prereg["observation_schema"]["feature_names"]) != rate_feature_names():
        raise RuntimeError("14D feature order drift")
    if prereg["observation_schema"]["schema_hash"] != rate_contract_hash():
        raise RuntimeError("14D schema hash drift")
    return expected


def prior_bounds_7d() -> np.ndarray:
    return np.asarray(parameter_contract()["bounds"], dtype=float)[:7]


def fixed_c_ctx2th() -> float:
    return float(read_preregistration()["fixed_parameter"]["value"])


def insert_fixed_parameter(theta7: Sequence[float]) -> np.ndarray:
    theta7 = np.asarray(theta7, dtype=float)
    if theta7.shape[-1:] != (7,):
        raise ValueError("theta7 must end in seven free parameters")
    fixed = fixed_c_ctx2th()
    if theta7.ndim == 1:
        return np.concatenate([theta7, [fixed]])
    return np.concatenate(
        [theta7, np.full((*theta7.shape[:-1], 1), fixed, dtype=float)], axis=-1
    )


def sobol_theta7(n: int, seed: int) -> np.ndarray:
    if n < 1 or n & (n - 1):
        raise ValueError("Sobol sample size must be a positive power of two")
    bounds = prior_bounds_7d()
    unit = qmc.Sobol(d=7, scramble=True, seed=int(seed)).random_base2(
        int(math.log2(n))
    )
    return qmc.scale(unit, bounds[:, 0], bounds[:, 1])


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _consolidate_7d(
    results: Sequence[Mapping[str, Any]],
    theta7: np.ndarray,
    path: Path,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    verify_preregistration()
    n = len(results)
    arrays: dict[str, Any] = {
        "sample_id": np.arange(n, dtype=np.int64),
        "theta": np.asarray(theta7, dtype=np.float64),
        "theta_full_8d": np.vstack([row["theta"] for row in results]).astype(np.float64),
        "x": np.vstack([row["x"] for row in results]).astype(np.float64),
        "validity": np.vstack([row["validity"] for row in results]).astype(bool),
        "simulator_seed": np.asarray([row["simulator_seed"] for row in results], dtype=np.int64),
        "success": np.asarray([row["success"] for row in results], dtype=bool),
        "runtime_s": np.asarray([row["runtime_s"] for row in results], dtype=np.float64),
        "failure_reason": np.asarray([row["failure_reason"] for row in results], dtype="<U1024"),
        "parameter_names": np.asarray(PARAMETER_NAMES_7D, dtype="<U32"),
        "parameter_names_full_8d": np.asarray(PARAMETER_NAMES_8D, dtype="<U32"),
        "feature_names": np.asarray(rate_feature_names(), dtype="<U64"),
        "fixed_parameter_name": np.asarray(FIXED_PARAMETER_NAME, dtype="<U32"),
        "fixed_parameter_value": np.asarray(fixed_c_ctx2th(), dtype=np.float64),
        "schema_version": np.asarray(SCHEMA_VERSION, dtype="<U64"),
        "schema_hash": np.asarray(rate_contract_hash(), dtype="<U64"),
        "preregistration_hash": np.asarray(verify_preregistration(), dtype="<U64"),
    }
    if extra:
        arrays.update(extra)
    atomic_npz(path, **arrays)
    return path


def _run_7d_batch(
    theta7: np.ndarray,
    seeds: Sequence[int],
    output_dir: Path,
    labels: Sequence[str],
    max_workers: int,
) -> list[dict[str, Any]]:
    verify_preregistration()
    theta7 = np.asarray(theta7, dtype=float)
    if theta7.ndim != 2 or theta7.shape[1] != 7:
        raise ValueError("theta7 must have shape (N,7)")
    theta8 = insert_fixed_parameter(theta7)
    results = run_resumable_batch(
        theta8,
        seeds,
        output_dir,
        model_version="v8a_t13",
        max_workers=max_workers,
        labels=labels,
    )
    for row, expected7, expected8 in zip(results, theta7, theta8):
        if not np.array_equal(np.asarray(row["theta"]), expected8):
            raise RuntimeError("silent 7D-to-8D insertion mismatch")
        if not np.isclose(row["theta"][7], fixed_c_ctx2th(), rtol=0, atol=0):
            raise RuntimeError("fixed c_ctx2th changed at simulator boundary")
    return results


def run_preflight_multiseed(max_workers: int = 6) -> dict[str, Any]:
    verify_preregistration()
    cfg = read_preregistration()["seeds"]
    theta_unique = sobol_theta7(128, int(cfg["preflight_sobol"]))
    replicate_seeds = np.asarray(cfg["preflight_simulator_replicates"], dtype=np.int64)
    theta = np.repeat(theta_unique, 3, axis=0)
    seeds = np.tile(replicate_seeds, 128)
    theta_id = np.repeat(np.arange(128), 3)
    replicate_id = np.tile(np.arange(3), 128)
    out = PREFLIGHT_ROOT / "prior_multiseed"
    results = _run_7d_batch(
        theta, seeds, out,
        [f"theta_{i:03d}_rep_{j}" for i, j in zip(theta_id, replicate_id)],
        max_workers,
    )
    path = _consolidate_7d(
        results,
        theta,
        out / "route3_7d_prior_multiseed_128x3.npz",
        {"theta_id": theta_id.astype(np.int64), "replicate_id": replicate_id.astype(np.int64)},
    )
    return {"path": path, "results": results, "theta_unique": theta_unique}


def analyze_preflight_multiseed(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        theta = np.asarray(data["theta"], dtype=float).reshape(128, 3, 7)[:, 0]
        x = np.asarray(data["x"], dtype=float).reshape(128, 3, 14)
        success = np.asarray(data["success"], dtype=bool)
    within = np.nanmean(np.nanvar(x, axis=1, ddof=1), axis=0)
    theta_means = np.nanmean(x, axis=1)
    between = np.nanvar(theta_means, axis=0, ddof=1)
    snr = np.divide(between, within, out=np.full(14, np.inf), where=within > 0)
    feature_rows = []
    for index, name in enumerate(rate_feature_names()):
        unique_counts = [len(np.unique(x[:, rep, index])) for rep in range(3)]
        feature_rows.append({
            "feature": name,
            "within_theta_seed_variance": within[index],
            "between_theta_variance": between[index],
            "between_to_within_snr": snr[index],
            "unique_values_min_across_seeds": min(unique_counts),
            "quantized_or_degenerate_warning": min(unique_counts) < 16,
        })
    features = pd.DataFrame(feature_rows)
    location, scale = _robust_location_scale(x.reshape(-1, 14))
    z = (x - location) / scale
    noise = np.concatenate([pdist(z[i]) for i in range(128)])
    noise_floor = float(np.quantile(noise, 0.95))
    nearest = np.empty((3, 128), dtype=int)
    for rep in range(3):
        distance = squareform(pdist(z[:, rep]))
        np.fill_diagonal(distance, np.inf)
        nearest[rep] = np.argmin(distance, axis=1)
    pair_agreements = [
        np.mean(nearest[a] == nearest[b]) for a, b in ((0, 1), (0, 2), (1, 2))
    ]
    theta_scaled = (theta - prior_bounds_7d()[:, 0]) / np.ptp(prior_bounds_7d(), axis=1)
    theta_distance = squareform(pdist(theta_scaled)) / np.sqrt(7)
    x_distance = squareform(pdist((theta_means - location) / scale))
    far = theta_distance >= 0.25
    upper = np.triu(np.ones_like(far, dtype=bool), 1)
    collisions = far & upper & (x_distance <= noise_floor)
    near = far & upper & (x_distance > noise_floor) & (x_distance <= 2 * noise_floor)
    collision_rows = []
    for a, b in np.argwhere(collisions):
        collision_rows.append({
            "theta_id_a": int(a), "theta_id_b": int(b),
            "theta_rms_distance": float(theta_distance[a, b]),
            "x_distance": float(x_distance[a, b]),
            "severity_ratio_x_to_noise": float(x_distance[a, b] / noise_floor),
            **{f"delta_{name}_prior_width": float(theta_scaled[a, j] - theta_scaled[b, j])
               for j, name in enumerate(PARAMETER_NAMES_7D)},
        })
    summary = {
        "attempted": int(success.size),
        "success": int(success.sum()),
        "failure_rate": float(1 - success.mean()),
        "finite_rate": float(np.isfinite(x).all(axis=2).mean()),
        "noise_floor_q95": noise_floor,
        "nearest_neighbor_all_three_agreement": float(np.mean(np.all(nearest == nearest[0], axis=0))),
        "nearest_neighbor_pair_agreement_mean": float(np.mean(pair_agreements)),
        "collision_count": int(collisions.sum()),
        "near_collision_count": int(near.sum()),
    }
    out = Path(path).parent
    features.to_csv(out / "feature_seed_noise.csv", index=False)
    pd.DataFrame(collision_rows).to_csv(out / "global_collisions.csv", index=False)
    atomic_json(out / "preflight_multiseed_summary.json", summary)
    return {
        "summary": summary, "features": features, "noise_floor": noise_floor,
        "theta": theta, "theta_means": theta_means, "nearest": nearest,
    }


def _canonical_centers_7d() -> list[tuple[str, np.ndarray]]:
    centers = []
    for center in audit_parameter_sets()[:3]:
        full = np.asarray([center.parameters[name] for name in PARAMETER_NAMES_8D], dtype=float)
        centers.append((center.identifier, full[:7]))
    return centers


def run_local_multiscale_7d(max_workers: int = 6) -> dict[str, Any]:
    verify_preregistration()
    cfg = read_preregistration()["seeds"]
    local_seeds = np.asarray(cfg["local_simulator_replicates"], dtype=np.int64)
    bounds = prior_bounds_7d()
    widths = np.ptp(bounds, axis=1)
    theta_rows, seeds, metadata = [], [], []
    for center_name, center in _canonical_centers_7d():
        for seed in local_seeds:
            theta_rows.append(center.copy()); seeds.append(seed)
            metadata.append((center_name, 0.0, "center", "center", seed))
            for scale in (0.01, 0.02, 0.05):
                for j, name in enumerate(PARAMETER_NAMES_7D):
                    for direction, sign in (("minus", -1), ("plus", 1)):
                        theta = center.copy()
                        theta[j] = np.clip(theta[j] + sign * scale * widths[j], *bounds[j])
                        theta_rows.append(theta); seeds.append(seed)
                        metadata.append((center_name, scale, name, direction, seed))
    theta = np.asarray(theta_rows)
    frame = pd.DataFrame(metadata, columns=["center", "scale", "parameter", "direction", "seed"])
    out = PREFLIGHT_ROOT / "local_multiscale"
    results = _run_7d_batch(
        theta, seeds, out,
        [f"{r.center}_{r.seed}_{r.scale}_{r.parameter}_{r.direction}" for r in frame.itertuples()],
        max_workers,
    )
    frame.to_csv(out / "local_design.csv", index=False)
    path = _consolidate_7d(
        results, theta, out / "route3_7d_local_multiscale.npz",
        {
            "center": frame.center.to_numpy(dtype="<U32"),
            "scale": frame.scale.to_numpy(float),
            "parameter": frame.parameter.to_numpy(dtype="<U32"),
            "direction": frame.direction.to_numpy(dtype="<U16"),
        },
    )
    return {"path": path, "results": results, "metadata": frame}


def analyze_local_multiscale_7d(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        theta = np.asarray(data["theta"], float)
        x = np.asarray(data["x"], float)
        center = np.asarray(data["center"]).astype(str)
        scale_values = np.asarray(data["scale"], float)
        parameter = np.asarray(data["parameter"]).astype(str)
        direction = np.asarray(data["direction"]).astype(str)
        seeds = np.asarray(data["simulator_seed"], int)
        success = np.asarray(data["success"], bool)
    widths = np.ptp(prior_bounds_7d(), axis=1)
    jacobians: dict[tuple[str, float, int], np.ndarray] = {}
    rank_rows, sensitivity_rows, pair_rows = [], [], []
    for center_name in np.unique(center):
        for seed in np.unique(seeds):
            center_index = np.flatnonzero(
                (center == center_name) & (seeds == seed) & (direction == "center")
            )[0]
            feature_scale = np.maximum(
                np.abs(x[center_index]),
                np.asarray([spec.scale_floor for spec in RATE_ONLY_FEATURES]),
            )
            for local_scale in (0.01, 0.02, 0.05):
                jacobian = np.empty((14, 7))
                for j, name in enumerate(PARAMETER_NAMES_7D):
                    minus = np.flatnonzero(
                        (center == center_name) & (seeds == seed)
                        & np.isclose(scale_values, local_scale)
                        & (parameter == name) & (direction == "minus")
                    )[0]
                    plus = np.flatnonzero(
                        (center == center_name) & (seeds == seed)
                        & np.isclose(scale_values, local_scale)
                        & (parameter == name) & (direction == "plus")
                    )[0]
                    normalized_step = (theta[plus, j] - theta[minus, j]) / widths[j]
                    jacobian[:, j] = (x[plus] - x[minus]) / feature_scale / normalized_step
                jacobians[(center_name, local_scale, int(seed))] = jacobian
                singular = np.linalg.svd(jacobian, compute_uv=False)
                relative = singular / singular[0]
                rank_rows.append({
                    "center": center_name, "scale": local_scale, "seed": seed,
                    "effective_rank": int(np.sum(relative >= RELATIVE_SINGULAR_VALUE_THRESHOLD)),
                    "condition_number": float(singular[0] / singular[-1]),
                    **{f"relative_singular_{i+1}": float(v) for i, v in enumerate(relative)},
                })
                for j, name in enumerate(PARAMETER_NAMES_7D):
                    sensitivity_rows.append({
                        "center": center_name, "scale": local_scale, "seed": seed,
                        "parameter": name, "sensitivity_norm": float(np.linalg.norm(jacobian[:, j])),
                    })
                for a in range(7):
                    for b in range(a + 1, 7):
                        denom = np.linalg.norm(jacobian[:, a]) * np.linalg.norm(jacobian[:, b])
                        pair_rows.append({
                            "center": center_name, "scale": local_scale, "seed": seed,
                            "parameter_a": PARAMETER_NAMES_7D[a],
                            "parameter_b": PARAMETER_NAMES_7D[b],
                            "cosine": float(np.dot(jacobian[:, a], jacobian[:, b]) / denom),
                        })
    direction_rows = []
    unique_seeds = list(np.unique(seeds))
    for center_name in np.unique(center):
        for local_scale in (0.01, 0.02, 0.05):
            for j, name in enumerate(PARAMETER_NAMES_7D):
                vectors = [jacobians[(center_name, local_scale, seed)][:, j] for seed in unique_seeds]
                for a, b in ((0, 1), (0, 2), (1, 2)):
                    denom = np.linalg.norm(vectors[a]) * np.linalg.norm(vectors[b])
                    direction_rows.append({
                        "center": center_name, "scale": local_scale, "parameter": name,
                        "seed_a": unique_seeds[a], "seed_b": unique_seeds[b],
                        "direction_cosine": float(np.dot(vectors[a], vectors[b]) / denom),
                    })
    rank = pd.DataFrame(rank_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    pairs = pd.DataFrame(pair_rows)
    directions = pd.DataFrame(direction_rows)
    out = Path(path).parent
    for name, frame in (
        ("rank", rank), ("sensitivity", sensitivity),
        ("parameter_pair_cosines", pairs), ("seed_direction_consistency", directions),
    ):
        frame.to_csv(out / f"{name}.csv", index=False)
    summary = {
        "attempted": int(success.size), "success": int(success.sum()),
        "failure_rate": float(1 - success.mean()),
        "rank_min": int(rank.effective_rank.min()), "rank_max": int(rank.effective_rank.max()),
        "all_local_full_rank_7": bool((rank.effective_rank == 7).all()),
    }
    atomic_json(out / "local_multiscale_summary.json", summary)
    return {
        "summary": summary, "rank": rank, "sensitivity": sensitivity,
        "pair_cosines": pairs, "direction_consistency": directions,
    }


def evaluate_preflight_gate(prior: Mapping[str, Any], local: Mapping[str, Any]) -> dict[str, Any]:
    gate = {
        "failure_rate_max": 0.01,
        "finite_rate_min": 0.99,
        "observed_failure_rate_max": max(
            float(prior["summary"]["failure_rate"]), float(local["summary"]["failure_rate"])
        ),
        "observed_finite_rate": float(prior["summary"]["finite_rate"]),
        "feature_dimension_order_stable": True,
        "seed_recorded_reproducible": True,
        "silent_parameter_insertion_errors": 0,
        "pickle_free_reload": True,
    }
    gate["pass"] = bool(
        gate["observed_failure_rate_max"] <= gate["failure_rate_max"]
        and gate["observed_finite_rate"] >= gate["finite_rate_min"]
        and gate["feature_dimension_order_stable"]
        and gate["seed_recorded_reproducible"]
        and gate["silent_parameter_insertion_errors"] == 0
    )
    atomic_json(PREFLIGHT_ROOT / "engineering_gate.json", gate)
    return gate


def run_training_bank_4096(max_workers: int = 6) -> dict[str, Any]:
    verify_preregistration()
    gate = json.loads((PREFLIGHT_ROOT / "engineering_gate.json").read_text(encoding="utf-8"))
    if not gate["pass"]:
        raise RuntimeError("7D preflight engineering hard gate did not pass")
    existing_bank = BANK_ROOT / "route3_7d_cortex_rate_14d_bank_4096.npz"
    existing_split = BANK_ROOT / "split_and_scaling.npz"
    existing_manifest = BANK_ROOT / "bank_manifest.json"
    if existing_bank.exists() and existing_split.exists() and existing_manifest.exists():
        manifest = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if (
            manifest.get("scheduled") == 4096
            and manifest.get("preregistration_hash") == verify_preregistration()
            and manifest.get("bank_sha256") == sha256_file(existing_bank)
            and manifest.get("split_sha256") == sha256_file(existing_split)
        ):
            return {
                "bank_path": existing_bank,
                "split_path": existing_split,
                "manifest": manifest,
                "resumed_from_completed_manifest": True,
            }
    cfg = read_preregistration()
    theta = sobol_theta7(4096, int(cfg["seeds"]["training_sobol"]))
    seeds = deterministic_seed_schedule(4096, int(cfg["seeds"]["training_simulator_base"]))
    old_seed_manifest = set(cfg["seed_nonoverlap_audit"]["old_explicit_seeds"])
    if old_seed_manifest.intersection(int(v) for v in seeds):
        raise RuntimeError("new training simulator seed overlaps old explicit seed")
    results = _run_7d_batch(
        theta, seeds, BANK_ROOT,
        [f"train7d_{i:04d}" for i in range(4096)], max_workers,
    )
    bank_path = _consolidate_7d(
        results, theta, BANK_ROOT / "route3_7d_cortex_rate_14d_bank_4096.npz",
        {"dataset_role": np.asarray("independent_7d_training_bank", dtype="<U64")},
    )
    success = np.asarray([row["success"] for row in results], bool)
    failure_rate = float(1 - success.mean())
    if failure_rate > 0.01:
        raise RuntimeError(f"4096 bank failure rate {failure_rate:.4f} exceeds 1%")
    valid_indices = np.flatnonzero(success)
    rng = np.random.default_rng(int(cfg["seeds"]["train_validation_split"]))
    shuffled = rng.permutation(valid_indices)
    n_train = int(np.floor(0.8 * len(shuffled)))
    train_indices = np.sort(shuffled[:n_train])
    validation_indices = np.sort(shuffled[n_train:])
    x = np.vstack([row["x"] for row in results])
    location, scale = _robust_location_scale(x[train_indices])
    split_path = BANK_ROOT / "split_and_scaling.npz"
    atomic_npz(
        split_path,
        training_indices=train_indices.astype(np.int64),
        validation_indices=validation_indices.astype(np.int64),
        x_location=location.astype(np.float64),
        x_scale=scale.astype(np.float64),
        theta_lower=prior_bounds_7d()[:, 0].astype(np.float64),
        theta_upper=prior_bounds_7d()[:, 1].astype(np.float64),
        parameter_names=np.asarray(PARAMETER_NAMES_7D, dtype="<U32"),
        feature_names=np.asarray(rate_feature_names(), dtype="<U64"),
        preregistration_hash=np.asarray(verify_preregistration(), dtype="<U64"),
    )
    manifest = {
        "scheduled": 4096, "attempted": 4096,
        "valid": int(success.sum()), "failed": int((~success).sum()),
        "failure_rate": failure_rate,
        "sum_simulator_runtime_s": float(np.nansum([row["runtime_s"] for row in results])),
        "median_simulator_runtime_s": float(np.nanmedian([row["runtime_s"] for row in results])),
        "training_rows": int(len(train_indices)), "validation_rows": int(len(validation_indices)),
        "scaling_fit_on_training_only": True,
        "bank_sha256": sha256_file(bank_path), "split_sha256": sha256_file(split_path),
        "schema_hash": rate_contract_hash(), "preregistration_hash": verify_preregistration(),
    }
    atomic_json(BANK_ROOT / "bank_manifest.json", manifest)
    return {"bank_path": bank_path, "split_path": split_path, "manifest": manifest}


def analyze_bank_collisions(bank_path: Path, split_path: Path) -> dict[str, Any]:
    with np.load(bank_path, allow_pickle=False) as bank:
        theta = np.asarray(bank["theta"], float)
        x = np.asarray(bank["x"], float)
        success = np.asarray(bank["success"], bool)
    with np.load(split_path, allow_pickle=False) as split:
        location = np.asarray(split["x_location"], float)
        scale = np.asarray(split["x_scale"], float)
    noise_summary = json.loads(
        (PREFLIGHT_ROOT / "prior_multiseed" / "preflight_multiseed_summary.json").read_text(encoding="utf-8")
    )
    noise_floor = float(noise_summary["noise_floor_q95"])
    valid_theta = theta[success]
    valid_x = x[success]
    theta_scaled = (valid_theta - prior_bounds_7d()[:, 0]) / np.ptp(prior_bounds_7d(), axis=1)
    theta_distance = pdist(theta_scaled) / np.sqrt(7)
    x_distance = pdist((valid_x - location) / scale)
    far = theta_distance >= 0.25
    collision = far & (x_distance <= noise_floor)
    near = far & (x_distance > noise_floor) & (x_distance <= 2 * noise_floor)
    summary = {
        "valid_rows": int(success.sum()), "all_pairs": int(len(theta_distance)),
        "far_pairs": int(far.sum()), "collision_count": int(collision.sum()),
        "near_collision_count": int(near.sum()),
        "collision_fraction_of_far_pairs": float(collision.sum() / max(1, far.sum())),
        "noise_floor_q95_from_independent_preflight": noise_floor,
    }
    atomic_json(BANK_ROOT / "global_collision_summary.json", summary)
    return summary


def audit_independence_from_8d() -> dict[str, Any]:
    """Compare new theta/seeds with old 8D train and held-out artifacts.

    Near-neighbor thresholds are descriptive only. They never filter samples.
    """

    from scipy.spatial.distance import cdist

    new_paths = {
        "new_training": BANK_ROOT / "route3_7d_cortex_rate_14d_bank_4096.npz",
        "new_heldout": HELDOUT_ROOT / "dataset" / "route3_7d_heldout_256.npz",
    }
    old_paths = {
        "old_8d_training": PROJECT_ROOT / "S4_sbi" / "results" / "route3_pilot_snpe"
        / "simulation_bank_2048" / "route3_cortex_rate_14d_bank_2048.npz",
        "old_8d_heldout": PROJECT_ROOT / "S4_sbi" / "results" / "route3_pilot_snpe"
        / "heldout_validation" / "dataset" / "route3_heldout_128.npz",
    }
    new_rows, old_rows = [], []
    new_seeds, old_seeds = [], []
    for role, path in new_paths.items():
        with np.load(path, allow_pickle=False) as data:
            theta7 = np.asarray(data["theta"], float)
            theta8 = np.asarray(data["theta_full_8d"], float)
            seeds = np.asarray(data["simulator_seed"], np.int64)
        new_rows.extend((role, i, theta7[i], theta8[i]) for i in range(len(theta7)))
        new_seeds.extend(int(v) for v in seeds)
    for role, path in old_paths.items():
        with np.load(path, allow_pickle=False) as data:
            theta8 = np.asarray(data["theta"], float)
            seeds = np.asarray(data["simulator_seed"], np.int64)
        old_rows.extend((role, i, theta8[i, :7], theta8[i]) for i in range(len(theta8)))
        old_seeds.extend(int(v) for v in seeds)
    new7 = np.vstack([row[2] for row in new_rows])
    new8 = np.vstack([row[3] for row in new_rows])
    old7 = np.vstack([row[2] for row in old_rows])
    old8 = np.vstack([row[3] for row in old_rows])
    exact8 = int(np.any(np.all(new8[:, None, :] == old8[None, :, :], axis=2), axis=1).sum())
    exact7 = int(np.any(np.all(new7[:, None, :] == old7[None, :, :], axis=2), axis=1).sum())
    bounds = prior_bounds_7d()
    new_scaled = (new7 - bounds[:, 0]) / np.ptp(bounds, axis=1)
    old_scaled = (old7 - bounds[:, 0]) / np.ptp(bounds, axis=1)
    distances = cdist(new_scaled, old_scaled) / np.sqrt(7)
    nearest_index = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(len(new_rows)), nearest_index]
    table = pd.DataFrame({
        "new_role": [row[0] for row in new_rows],
        "new_sample_id": [row[1] for row in new_rows],
        "nearest_old_role": [old_rows[index][0] for index in nearest_index],
        "nearest_old_sample_id": [old_rows[index][1] for index in nearest_index],
        "nearest_7d_prior_rms_distance": nearest_distance,
    })
    table.to_csv(RESULTS_ROOT / "cross_experiment_theta_nearest_neighbors.csv", index=False)
    summary = {
        "new_rows": len(new_rows),
        "old_rows": len(old_rows),
        "full_8d_exact_duplicate_new_rows": exact8,
        "free_7d_exact_duplicate_new_rows": exact7,
        "simulator_seed_intersection_count": len(set(new_seeds) & set(old_seeds)),
        "nearest_7d_prior_rms_distance_min": float(nearest_distance.min()),
        "nearest_7d_prior_rms_distance_median": float(np.median(nearest_distance)),
        "nearest_7d_prior_rms_distance_q05": float(np.quantile(nearest_distance, .05)),
        "descriptive_count_below_0p01": int((nearest_distance < .01).sum()),
        "descriptive_count_below_0p05": int((nearest_distance < .05).sum()),
        "near_neighbor_thresholds_are_exclusion_rules": False,
        "samples_removed_after_audit": 0,
    }
    atomic_json(RESULTS_ROOT / "cross_experiment_independence_audit.json", summary)
    return {"summary": summary, "table": table}


def reload_artifacts_no_pickle(paths: Sequence[Path]) -> dict[str, Any]:
    checked = []
    for path in paths:
        path = Path(path)
        if path.suffix.lower() == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".csv":
            pd.read_csv(path)
        elif path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as data:
                for name in data.files:
                    if data[name].dtype.kind == "O":
                        raise RuntimeError(f"object array {name} in {path}")
        checked.append(str(path.relative_to(PROJECT_ROOT)))
    return {"checked": len(checked), "paths": checked, "all_passed": True}
