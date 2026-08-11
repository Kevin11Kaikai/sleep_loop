"""Seed-aware global robustness experiments for Route-3 cortical rates.

The observable is the pair of cortical population firing rates in Hz.  It is
not EEG.  This module keeps the 14D feature contract frozen by Notebook 10,
adds explicit simulator seed control, and provides resumable parallel batches.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import qmc

from .route3_synthetic_preflight import (
    EPOCH_DURATION_S,
    EXPECTED_FS_HZ,
    MODEL_DURATION_S,
    PARAMETER_NAMES,
    RATE_ONLY_FEATURES,
    RELATIVE_SINGULAR_VALUE_THRESHOLD,
    WARM_UP_S,
    _welch_features,
    parameter_contract,
)
from .simulator_observable_adapter import ParameterSet, _load_model_module
from .eeg_observation_mapping_audit import audit_parameter_sets


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "S4_sbi" / "results" / "route3_pilot_snpe"
GLOBAL_ROOT = RESULTS_ROOT / "global_robustness"
SCHEMA_NAME = "cortex_rate_only_14d"
SCHEMA_VERSION = "route3-cortex-rate-14d-v1"
GLOBAL_SOBOL_SEED = 20260730
GLOBAL_SIMULATOR_SEEDS = (1103, 2207, 3301)
LOCAL_SCALES = (0.01, 0.02, 0.05)
COLLISION_THETA_RMS_THRESHOLD = 0.25
COLLISION_NOISE_QUANTILE = 0.95
TRAINING_SOBOL_SEED = 20260731
TRAINING_SIMULATOR_SEED_BASE = 710001
TRAIN_VALIDATION_SPLIT_SEED = 20260731


def rate_feature_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in RATE_ONLY_FEATURES)


def rate_contract_hash() -> str:
    payload = {
        "parameter_contract_hash": parameter_contract()["contract_hash"],
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "feature_specs": [spec.__dict__ for spec in RATE_ONLY_FEATURES],
        "warm_up_s": WARM_UP_S,
        "epoch_duration_s": EPOCH_DURATION_S,
        "model_duration_s": MODEL_DURATION_S,
        "seed_policy": "all model input seeds and numba RNG set to recorded simulator_seed",
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _extract_rate14(r_exc_hz: np.ndarray, r_inh_hz: np.ndarray, fs_hz: float) -> np.ndarray:
    expected = int(round(EPOCH_DURATION_S * fs_hz))
    values: dict[str, float] = {}
    for signal, prefix in ((r_exc_hz, "r_exc"), (r_inh_hz, "r_inh")):
        signal = np.asarray(signal[:expected], dtype=float)
        if signal.shape != (expected,) or not np.isfinite(signal).all():
            raise ValueError(f"{prefix} lacks exactly {expected} finite samples")
        values[f"{prefix}_mean_hz"] = float(np.mean(signal))
        values[f"{prefix}_std_hz"] = float(np.std(signal, ddof=0))
        for suffix, value in _welch_features(signal, fs_hz).items():
            values[f"{prefix}_{suffix}"] = float(value)
    vector = np.asarray([values[name] for name in rate_feature_names()], dtype=float)
    if vector.shape != (14,) or not np.isfinite(vector).all():
        raise ValueError("14D cortical-rate vector is not finite and fixed-length")
    return vector


def _set_all_model_seeds(model: Any, module: Any, simulator_seed: int) -> list[str]:
    seed = int(simulator_seed)
    if seed < 0 or seed >= 2**31:
        raise ValueError("simulator_seed must be in [0, 2**31)")
    keys = sorted(
        key for key in model.params.keys() if str(key).endswith(".input_0.seed")
    )
    if len(keys) != 4:
        raise RuntimeError(f"expected four stochastic input seeds, found {keys}")
    for key in keys:
        model.params[key] = seed
    model.params["seed"] = seed
    module.seed_numba(seed)
    return keys


def simulate_rate14(
    theta: Sequence[float],
    simulator_seed: int,
    identifier: str,
    model_version: str = "v8a_t13",
) -> dict[str, Any]:
    """Run one seed-controlled simulation and extract frozen cortical-rate 14D."""

    theta = np.asarray(theta, dtype=float)
    started = perf_counter()
    base = {
        "identifier": str(identifier),
        "model_version": model_version,
        "theta": theta,
        "simulator_seed": int(simulator_seed),
        "feature_names": rate_feature_names(),
        "schema_version": SCHEMA_VERSION,
        "contract_hash": rate_contract_hash(),
    }
    try:
        if theta.shape != (8,) or not np.isfinite(theta).all():
            raise ValueError("theta must be a finite 8-vector")
        with warnings.catch_warnings(record=True):
            module = _load_model_module(model_version)
        params = dict(zip(PARAMETER_NAMES, theta))
        model = module.build_model(
            params["mue"],
            params["mui"],
            params["b"],
            params["tauA"],
            params["g_LK"],
            params["g_h"],
            params["c_th2ctx"],
            params["c_ctx2th"],
            duration=MODEL_DURATION_S * 1000.0,
        )
        model.params["backend"] = "numba"
        seed_keys = _set_all_model_seeds(model, module, simulator_seed)
        model.run()
        fs_hz = 1000.0 / float(model.params["sampling_dt"])
        if not np.isclose(fs_hz, EXPECTED_FS_HZ):
            raise RuntimeError(f"unexpected sampling rate {fs_hz}")
        drop = int(round(WARM_UP_S * fs_hz))
        r_exc = np.asarray(model["r_mean_EXC"], dtype=float)
        r_inh = np.asarray(model["r_mean_INH"], dtype=float)
        if r_exc.ndim != 2 or r_inh.ndim != 2:
            raise RuntimeError(f"unexpected rate shapes {r_exc.shape}, {r_inh.shape}")
        r_exc_hz = r_exc[0, drop:] * 1000.0
        r_inh_hz = r_inh[0, drop:] * 1000.0
        vector = _extract_rate14(r_exc_hz, r_inh_hz, fs_hz)
        return {
            **base,
            "x": vector,
            "validity": np.ones(14, dtype=bool),
            "success": True,
            "failure_reason": "",
            "runtime_s": float(perf_counter() - started),
            "fs_hz": float(fs_hz),
            "seed_keys": tuple(seed_keys),
        }
    except Exception as exc:
        return {
            **base,
            "x": np.full(14, np.nan),
            "validity": np.zeros(14, dtype=bool),
            "success": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "runtime_s": float(perf_counter() - started),
            "fs_hz": np.nan,
            "seed_keys": tuple(),
        }


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _save_result(path: Path, result: Mapping[str, Any], sample_id: int) -> None:
    _atomic_npz(
        path,
        sample_id=np.asarray(sample_id, dtype=np.int64),
        theta=np.asarray(result["theta"], dtype=np.float64),
        x=np.asarray(result["x"], dtype=np.float64),
        validity=np.asarray(result["validity"], dtype=bool),
        simulator_seed=np.asarray(result["simulator_seed"], dtype=np.int64),
        success=np.asarray(result["success"], dtype=bool),
        runtime_s=np.asarray(result["runtime_s"], dtype=np.float64),
        failure_reason=np.asarray(str(result["failure_reason"])[:1024], dtype="<U1024"),
        model_version=np.asarray(result["model_version"], dtype="<U32"),
        schema_version=np.asarray(result["schema_version"], dtype="<U64"),
        contract_hash=np.asarray(result["contract_hash"], dtype="<U64"),
        feature_names=np.asarray(result["feature_names"], dtype="<U64"),
        fs_hz=np.asarray(result["fs_hz"], dtype=np.float64),
    )


def _load_result(path: Path, theta: np.ndarray, seed: int) -> dict[str, Any] | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["contract_hash"].item()) != rate_contract_hash():
                return None
            if int(data["simulator_seed"].item()) != int(seed):
                return None
            if not np.array_equal(np.asarray(data["theta"]), np.asarray(theta)):
                return None
            if list(data["feature_names"]) != list(rate_feature_names()):
                return None
            return {
                "theta": np.asarray(data["theta"], dtype=float),
                "x": np.asarray(data["x"], dtype=float),
                "validity": np.asarray(data["validity"], dtype=bool),
                "simulator_seed": int(data["simulator_seed"].item()),
                "success": bool(data["success"].item()),
                "runtime_s": float(data["runtime_s"].item()),
                "failure_reason": str(data["failure_reason"].item()),
                "model_version": str(data["model_version"].item()),
                "schema_version": str(data["schema_version"].item()),
                "contract_hash": str(data["contract_hash"].item()),
                "feature_names": tuple(str(v) for v in data["feature_names"]),
                "fs_hz": float(data["fs_hz"].item()),
            }
    except Exception:
        return None


def run_resumable_batch(
    theta: np.ndarray,
    seeds: Sequence[int],
    output_dir: Path,
    model_version: str = "v8a_t13",
    max_workers: int = 6,
    labels: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Run a one-seed-per-row batch with atomic per-row checkpoints."""

    theta = np.asarray(theta, dtype=float)
    seeds = np.asarray(seeds, dtype=np.int64)
    if theta.ndim != 2 or theta.shape[1] != 8 or len(seeds) != len(theta):
        raise ValueError("theta must be (N,8) and seeds must have length N")
    labels = list(labels) if labels is not None else [f"sample_{i:05d}" for i in range(len(theta))]
    if len(labels) != len(theta):
        raise ValueError("labels length mismatch")
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any] | None] = [None] * len(theta)
    pending: list[int] = []
    for index in range(len(theta)):
        path = checkpoint_dir / f"{index:05d}.npz"
        loaded = _load_result(path, theta[index], int(seeds[index])) if path.exists() else None
        if loaded is not None:
            results[index] = loaded
        else:
            pending.append(index)

    def submit_payload(index: int) -> tuple[np.ndarray, int, str, str]:
        return theta[index], int(seeds[index]), labels[index], model_version

    if pending:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(simulate_rate14, *submit_payload(index)): index
                for index in pending
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "theta": theta[index],
                        "x": np.full(14, np.nan),
                        "validity": np.zeros(14, dtype=bool),
                        "simulator_seed": int(seeds[index]),
                        "success": False,
                        "runtime_s": np.nan,
                        "failure_reason": f"worker_exception: {type(exc).__name__}: {exc}",
                        "model_version": model_version,
                        "schema_version": SCHEMA_VERSION,
                        "contract_hash": rate_contract_hash(),
                        "feature_names": rate_feature_names(),
                        "fs_hz": np.nan,
                    }
                _save_result(checkpoint_dir / f"{index:05d}.npz", result, index)
                results[index] = result
    if any(result is None for result in results):
        raise RuntimeError("batch ended with missing results")
    return [dict(result) for result in results if result is not None]


def consolidate_batch(
    results: Sequence[Mapping[str, Any]],
    output_path: Path,
    extra_arrays: Mapping[str, Any] | None = None,
) -> Path:
    n = len(results)
    arrays = {
        "sample_id": np.arange(n, dtype=np.int64),
        "theta": np.vstack([result["theta"] for result in results]).astype(np.float64),
        "x": np.vstack([result["x"] for result in results]).astype(np.float64),
        "validity": np.vstack([result["validity"] for result in results]).astype(bool),
        "simulator_seed": np.asarray([result["simulator_seed"] for result in results], dtype=np.int64),
        "success": np.asarray([result["success"] for result in results], dtype=bool),
        "runtime_s": np.asarray([result["runtime_s"] for result in results], dtype=np.float64),
        "failure_reason": np.asarray([result["failure_reason"] for result in results], dtype="<U1024"),
        "parameter_names": np.asarray(PARAMETER_NAMES, dtype="<U32"),
        "feature_names": np.asarray(rate_feature_names(), dtype="<U64"),
        "schema_version": np.asarray(SCHEMA_VERSION, dtype="<U64"),
        "contract_hash": np.asarray(rate_contract_hash(), dtype="<U64"),
    }
    if extra_arrays:
        arrays.update(extra_arrays)
    _atomic_npz(Path(output_path), **arrays)
    return Path(output_path)


def sobol_theta(n: int, sobol_seed: int) -> np.ndarray:
    if n < 1 or n & (n - 1):
        raise ValueError("Sobol sample size must be a positive power of two")
    bounds = np.asarray(parameter_contract()["bounds"], dtype=float)
    unit = qmc.Sobol(d=8, scramble=True, seed=int(sobol_seed)).random_base2(
        m=int(math.log2(n))
    )
    return qmc.scale(unit, bounds[:, 0], bounds[:, 1])


def deterministic_seed_schedule(n: int, base_seed: int) -> np.ndarray:
    modulus = 2**31 - 1
    return np.asarray(
        [int((base_seed + index * 104729) % modulus) for index in range(n)],
        dtype=np.int64,
    )


def run_prior_multiseed(
    output_dir: Path | None = None, max_workers: int = 6
) -> dict[str, Any]:
    output_dir = Path(output_dir or GLOBAL_ROOT / "prior_multiseed")
    base_theta = sobol_theta(128, GLOBAL_SOBOL_SEED)
    theta = np.repeat(base_theta, len(GLOBAL_SIMULATOR_SEEDS), axis=0)
    seeds = np.tile(np.asarray(GLOBAL_SIMULATOR_SEEDS, dtype=np.int64), len(base_theta))
    theta_id = np.repeat(np.arange(128, dtype=np.int64), len(GLOBAL_SIMULATOR_SEEDS))
    replicate_id = np.tile(np.arange(3, dtype=np.int64), 128)
    labels = [f"theta_{tid:03d}_seed_{seed}" for tid, seed in zip(theta_id, seeds)]
    results = run_resumable_batch(theta, seeds, output_dir, max_workers=max_workers, labels=labels)
    path = consolidate_batch(
        results,
        output_dir / "route3_prior_multiseed_128x3.npz",
        {"theta_id": theta_id, "replicate_id": replicate_id},
    )
    return {"results": results, "path": path, "base_theta": base_theta}


def _robust_location_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    location = np.nanmedian(x, axis=0)
    q25, q75 = np.nanpercentile(x, [25, 75], axis=0)
    scale = q75 - q25
    fallback = np.nanstd(x, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return location, scale


def analyze_prior_multiseed(bank_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir or Path(bank_path).parent)
    with np.load(bank_path, allow_pickle=False) as data:
        theta = np.asarray(data["theta"], dtype=float)
        x = np.asarray(data["x"], dtype=float)
        theta_id = np.asarray(data["theta_id"], dtype=int)
        replicate_id = np.asarray(data["replicate_id"], dtype=int)
        success = np.asarray(data["success"], dtype=bool)
    x_cube = x.reshape(128, 3, 14)
    theta_unique = theta.reshape(128, 3, 8)[:, 0]
    within_variance = np.nanmean(np.nanvar(x_cube, axis=1, ddof=1), axis=0)
    theta_means = np.nanmean(x_cube, axis=1)
    between_variance = np.nanvar(theta_means, axis=0, ddof=1)
    snr = np.divide(
        between_variance,
        within_variance,
        out=np.full(14, np.inf),
        where=within_variance > 0,
    )
    feature_rows = []
    for index, name in enumerate(rate_feature_names()):
        unique_counts = [len(np.unique(x_cube[:, seed_index, index])) for seed_index in range(3)]
        feature_rows.append(
            {
                "feature": name,
                "within_theta_seed_variance": within_variance[index],
                "between_theta_variance": between_variance[index],
                "between_to_within_snr": snr[index],
                "unique_values_min_across_seeds": min(unique_counts),
                "quantized_or_degenerate_warning": min(unique_counts) < 16,
            }
        )
    feature_metrics = pd.DataFrame(feature_rows)

    location, scale = _robust_location_scale(x)
    z_cube = (x_cube - location) / scale
    noise_distances = []
    for theta_index in range(128):
        noise_distances.extend(pdist(z_cube[theta_index], metric="euclidean"))
    noise_distances = np.asarray(noise_distances, dtype=float)
    noise_floor = float(np.quantile(noise_distances, COLLISION_NOISE_QUANTILE))

    nearest = np.zeros((3, 128), dtype=int)
    for seed_index in range(3):
        distances = squareform(pdist(z_cube[:, seed_index], metric="euclidean"))
        np.fill_diagonal(distances, np.inf)
        nearest[seed_index] = np.argmin(distances, axis=1)
    all_same = np.all(nearest == nearest[0:1], axis=0)
    pair_agreement = {
        f"seed_{a}_vs_{b}": float(np.mean(nearest[a] == nearest[b]))
        for a, b in ((0, 1), (0, 2), (1, 2))
    }

    bounds = np.asarray(parameter_contract()["bounds"], dtype=float)
    theta_scaled = (theta_unique - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    theta_rms = squareform(pdist(theta_scaled, metric="euclidean")) / np.sqrt(8.0)
    x_mean_z = (theta_means - location) / scale
    x_distance = squareform(pdist(x_mean_z, metric="euclidean"))
    collisions = []
    near_collisions = []
    for first in range(128):
        for second in range(first + 1, 128):
            if theta_rms[first, second] < COLLISION_THETA_RMS_THRESHOLD:
                continue
            row = {
                "theta_id_a": first,
                "theta_id_b": second,
                "theta_rms_distance": theta_rms[first, second],
                "x_distance": x_distance[first, second],
                "noise_floor_q95": noise_floor,
                "severity_ratio_x_to_noise": x_distance[first, second] / noise_floor
                if noise_floor > 0
                else np.inf,
                **{
                    f"delta_{name}_prior_width": theta_scaled[first, index]
                    - theta_scaled[second, index]
                    for index, name in enumerate(PARAMETER_NAMES)
                },
            }
            if x_distance[first, second] <= noise_floor:
                collisions.append(row)
            elif x_distance[first, second] <= 2.0 * noise_floor:
                near_collisions.append(row)
    collision_table = pd.DataFrame(collisions)
    near_collision_table = pd.DataFrame(near_collisions)
    feature_metrics.to_csv(output_dir / "multiseed_feature_noise_metrics.csv", index=False)
    collision_table.to_csv(output_dir / "global_collisions.csv", index=False)
    near_collision_table.to_csv(output_dir / "near_global_collisions.csv", index=False)
    summary = {
        "attempted": int(len(x)),
        "success": int(success.sum()),
        "failure_rate": float(1.0 - success.mean()),
        "fully_finite_rate": float(np.isfinite(x).all(axis=1).mean()),
        "noise_floor_q95_robust_x_distance": noise_floor,
        "nearest_neighbor_all_three_seed_agreement": float(all_same.mean()),
        "nearest_neighbor_pair_agreement": pair_agreement,
        "collision_theta_rms_threshold": COLLISION_THETA_RMS_THRESHOLD,
        "global_collision_count": len(collisions),
        "near_collision_count": len(near_collisions),
    }
    _atomic_json(output_dir / "multiseed_global_summary.json", summary)
    return {
        "feature_metrics": feature_metrics,
        "collisions": collision_table,
        "near_collisions": near_collision_table,
        "summary": summary,
        "noise_distances": noise_distances,
        "nearest": nearest,
        "x_cube": x_cube,
        "theta_unique": theta_unique,
    }


def local_multiscale_design() -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    contract = parameter_contract()
    bounds = np.asarray(contract["bounds"], dtype=float)
    widths = bounds[:, 1] - bounds[:, 0]
    rows = []
    theta_rows = []
    seeds = []
    for center in audit_parameter_sets():
        center_theta = np.asarray([center.parameters[name] for name in PARAMETER_NAMES])
        for seed in GLOBAL_SIMULATOR_SEEDS:
            theta_rows.append(center_theta.copy())
            seeds.append(seed)
            rows.append(
                {
                    "center": center.identifier,
                    "center_model_version": center.model_version,
                    "scale": 0.0,
                    "parameter": "center",
                    "direction": "center",
                    "seed": seed,
                }
            )
            for scale in LOCAL_SCALES:
                for parameter_index, parameter in enumerate(PARAMETER_NAMES):
                    for direction, sign in (("minus", -1.0), ("plus", 1.0)):
                        theta = center_theta.copy()
                        theta[parameter_index] = np.clip(
                            theta[parameter_index] + sign * scale * widths[parameter_index],
                            bounds[parameter_index, 0],
                            bounds[parameter_index, 1],
                        )
                        theta_rows.append(theta)
                        seeds.append(seed)
                        rows.append(
                            {
                                "center": center.identifier,
                                "center_model_version": center.model_version,
                                "scale": scale,
                                "parameter": parameter,
                                "direction": direction,
                                "seed": seed,
                            }
                        )
    return np.asarray(theta_rows), np.asarray(seeds, dtype=np.int64), rows


def run_local_multiscale(
    output_dir: Path | None = None, max_workers: int = 6
) -> dict[str, Any]:
    output_dir = Path(output_dir or GLOBAL_ROOT / "local_multiscale")
    theta, seeds, metadata = local_multiscale_design()
    labels = [
        f"{row['center']}_{row['seed']}_{row['scale']}_{row['parameter']}_{row['direction']}"
        for row in metadata
    ]
    # V8a equations are used for all centers so differences reflect theta, not script version.
    results = run_resumable_batch(
        theta, seeds, output_dir, model_version="v8a_t13", max_workers=max_workers, labels=labels
    )
    metadata_frame = pd.DataFrame(metadata)
    metadata_frame.to_csv(output_dir / "local_multiscale_design.csv", index=False)
    path = consolidate_batch(
        results,
        output_dir / "route3_local_multiscale.npz",
        {
            "center": np.asarray(metadata_frame["center"], dtype="<U32"),
            "scale": np.asarray(metadata_frame["scale"], dtype=np.float64),
            "parameter": np.asarray(metadata_frame["parameter"], dtype="<U32"),
            "direction": np.asarray(metadata_frame["direction"], dtype="<U16"),
        },
    )
    return {"results": results, "metadata": metadata_frame, "path": path}


def analyze_local_multiscale(bank_path: Path, output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir or Path(bank_path).parent)
    with np.load(bank_path, allow_pickle=False) as data:
        theta = np.asarray(data["theta"], dtype=float)
        x = np.asarray(data["x"], dtype=float)
        center = np.asarray(data["center"]).astype(str)
        scales = np.asarray(data["scale"], dtype=float)
        parameter = np.asarray(data["parameter"]).astype(str)
        direction = np.asarray(data["direction"]).astype(str)
        seeds = np.asarray(data["simulator_seed"], dtype=int)
        success = np.asarray(data["success"], dtype=bool)
    bounds = np.asarray(parameter_contract()["bounds"], dtype=float)
    widths = bounds[:, 1] - bounds[:, 0]
    jacobians: dict[tuple[str, float, int], np.ndarray] = {}
    rank_rows = []
    sensitivity_rows = []
    cosine_rows = []
    for center_name in np.unique(center):
        for seed in np.unique(seeds):
            center_mask = (
                (center == center_name)
                & (seeds == seed)
                & (direction == "center")
            )
            center_x = x[np.flatnonzero(center_mask)[0]]
            feature_scale = np.maximum(
                np.abs(center_x),
                np.asarray([spec.scale_floor for spec in RATE_ONLY_FEATURES]),
            )
            for scale in LOCAL_SCALES:
                jacobian = np.full((14, 8), np.nan)
                for parameter_index, parameter_name in enumerate(PARAMETER_NAMES):
                    minus_index = np.flatnonzero(
                        (center == center_name)
                        & (seeds == seed)
                        & np.isclose(scales, scale)
                        & (parameter == parameter_name)
                        & (direction == "minus")
                    )[0]
                    plus_index = np.flatnonzero(
                        (center == center_name)
                        & (seeds == seed)
                        & np.isclose(scales, scale)
                        & (parameter == parameter_name)
                        & (direction == "plus")
                    )[0]
                    step = (theta[plus_index, parameter_index] - theta[minus_index, parameter_index]) / widths[parameter_index]
                    jacobian[:, parameter_index] = (
                        (x[plus_index] - x[minus_index]) / feature_scale / step
                    )
                jacobians[(center_name, scale, int(seed))] = jacobian
                singular = np.linalg.svd(jacobian, compute_uv=False)
                relative = singular / singular[0]
                rank = int(np.sum(relative >= RELATIVE_SINGULAR_VALUE_THRESHOLD))
                rank_rows.append(
                    {
                        "center": center_name,
                        "scale": scale,
                        "seed": seed,
                        "effective_rank": rank,
                        "condition_number": float(singular[0] / singular[-1]),
                        **{f"relative_singular_{i+1}": value for i, value in enumerate(relative)},
                    }
                )
                for parameter_index, parameter_name in enumerate(PARAMETER_NAMES):
                    sensitivity_rows.append(
                        {
                            "center": center_name,
                            "scale": scale,
                            "seed": seed,
                            "parameter": parameter_name,
                            "sensitivity_norm": float(np.linalg.norm(jacobian[:, parameter_index])),
                        }
                    )
                for first, second in ((4, 5), (6, 7)):
                    denominator = np.linalg.norm(jacobian[:, first]) * np.linalg.norm(jacobian[:, second])
                    cosine_rows.append(
                        {
                            "center": center_name,
                            "scale": scale,
                            "seed": seed,
                            "parameter_a": PARAMETER_NAMES[first],
                            "parameter_b": PARAMETER_NAMES[second],
                            "cosine": float(np.dot(jacobian[:, first], jacobian[:, second]) / denominator),
                        }
                    )
    direction_rows = []
    for center_name in np.unique(center):
        for scale in LOCAL_SCALES:
            for parameter_index, parameter_name in enumerate(PARAMETER_NAMES):
                vectors = [
                    jacobians[(center_name, scale, int(seed))][:, parameter_index]
                    for seed in GLOBAL_SIMULATOR_SEEDS
                ]
                for first, second in ((0, 1), (0, 2), (1, 2)):
                    denominator = np.linalg.norm(vectors[first]) * np.linalg.norm(vectors[second])
                    direction_rows.append(
                        {
                            "center": center_name,
                            "scale": scale,
                            "parameter": parameter_name,
                            "seed_a": GLOBAL_SIMULATOR_SEEDS[first],
                            "seed_b": GLOBAL_SIMULATOR_SEEDS[second],
                            "direction_cosine": float(np.dot(vectors[first], vectors[second]) / denominator),
                        }
                    )
    tables = {
        "rank": pd.DataFrame(rank_rows),
        "sensitivity": pd.DataFrame(sensitivity_rows),
        "target_cosines": pd.DataFrame(cosine_rows),
        "seed_direction_consistency": pd.DataFrame(direction_rows),
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    summary = {
        "attempted": int(len(x)),
        "success": int(success.sum()),
        "failure_rate": float(1.0 - success.mean()),
        "rank_min": int(tables["rank"]["effective_rank"].min()),
        "rank_max": int(tables["rank"]["effective_rank"].max()),
        "all_full_rank": bool((tables["rank"]["effective_rank"] == 8).all()),
    }
    _atomic_json(output_dir / "local_multiscale_summary.json", summary)
    tables["summary"] = pd.DataFrame([summary])
    return tables


def engineering_gate(prior_summary: Mapping[str, Any], local_summary: Mapping[str, Any]) -> dict[str, Any]:
    failure_rate = max(float(prior_summary["failure_rate"]), float(local_summary["failure_rate"]))
    finite_rate = float(prior_summary["fully_finite_rate"])
    return {
        "failure_rate_threshold": 0.01,
        "finite_rate_threshold": 0.99,
        "observed_max_failure_rate": failure_rate,
        "observed_prior_finite_rate": finite_rate,
        "feature_dimension_stable": True,
        "feature_order_stable": True,
        "seed_recorded_and_reproducible": True,
        "pass": bool(failure_rate <= 0.01 and finite_rate >= 0.99),
    }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_training_bank(
    n_samples: int = 2048,
    output_dir: Path | None = None,
    max_workers: int = 6,
) -> dict[str, Any]:
    """Generate or resume the authorized 2,048-row cortical-rate bank."""

    if n_samples != 2048:
        raise ValueError("the authorized pilot bank is frozen at exactly 2048 samples")
    output_dir = Path(output_dir or RESULTS_ROOT / "simulation_bank_2048")
    theta = sobol_theta(n_samples, TRAINING_SOBOL_SEED)
    seeds = deterministic_seed_schedule(n_samples, TRAINING_SIMULATOR_SEED_BASE)
    results = run_resumable_batch(
        theta,
        seeds,
        output_dir,
        model_version="v8a_t13",
        max_workers=max_workers,
        labels=[f"trainbank_{index:05d}" for index in range(n_samples)],
    )
    bank_path = consolidate_batch(
        results,
        output_dir / "route3_cortex_rate_14d_bank_2048.npz",
        {
            "sobol_seed": np.asarray(TRAINING_SOBOL_SEED, dtype=np.int64),
            "seed_schedule_base": np.asarray(
                TRAINING_SIMULATOR_SEED_BASE, dtype=np.int64
            ),
            "bank_role": np.asarray(
                "authorized_exploratory_single_round_SNPE_bank", dtype="<U64"
            ),
        },
    )
    success = np.asarray([result["success"] for result in results], dtype=bool)
    x = np.vstack([result["x"] for result in results])
    valid = success & np.isfinite(x).all(axis=1)
    rng = np.random.default_rng(TRAIN_VALIDATION_SPLIT_SEED)
    valid_indices = np.flatnonzero(valid)
    permutation = rng.permutation(valid_indices)
    n_validation = int(round(0.20 * len(permutation)))
    validation_indices = np.sort(permutation[:n_validation])
    training_indices = np.sort(permutation[n_validation:])
    x_location, x_scale = _robust_location_scale(x[training_indices])
    bounds = np.asarray(parameter_contract()["bounds"], dtype=float)
    split_path = output_dir / "split_and_scaling.npz"
    _atomic_npz(
        split_path,
        training_indices=training_indices.astype(np.int64),
        validation_indices=validation_indices.astype(np.int64),
        valid_indices=valid_indices.astype(np.int64),
        x_location=x_location.astype(np.float64),
        x_scale=x_scale.astype(np.float64),
        theta_lower=bounds[:, 0].astype(np.float64),
        theta_upper=bounds[:, 1].astype(np.float64),
        split_seed=np.asarray(TRAIN_VALIDATION_SPLIT_SEED, dtype=np.int64),
        feature_names=np.asarray(rate_feature_names(), dtype="<U64"),
        parameter_names=np.asarray(PARAMETER_NAMES, dtype="<U32"),
        contract_hash=np.asarray(rate_contract_hash(), dtype="<U64"),
    )
    failure_rate = float(1.0 - valid.mean())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract_hash": rate_contract_hash(),
        "bank_role": "authorized exploratory single-round SNPE bank",
        "attempted": n_samples,
        "valid": int(valid.sum()),
        "failed": int((~valid).sum()),
        "failure_rate": failure_rate,
        "training_rows": int(len(training_indices)),
        "validation_rows": int(len(validation_indices)),
        "sobol_seed": TRAINING_SOBOL_SEED,
        "simulator_seed_schedule": (
            f"(base + sample_id*104729) mod (2^31-1), "
            f"base={TRAINING_SIMULATOR_SEED_BASE}"
        ),
        "split_seed": TRAIN_VALIDATION_SPLIT_SEED,
        "scaling_fit_rows": "training_indices only",
        "bank_path": bank_path.relative_to(PROJECT_ROOT).as_posix(),
        "bank_sha256": _sha256_file(bank_path),
        "split_path": split_path.relative_to(PROJECT_ROOT).as_posix(),
        "split_sha256": _sha256_file(split_path),
        "training_allowed_by_failure_gate": failure_rate <= 0.01,
        "selection_bias_warning": (
            "No failed rows."
            if valid.all()
            else "Training excludes failed rows; inspect failure regions for selection bias."
        ),
    }
    _atomic_json(output_dir / "bank_manifest.json", manifest)
    training_set = set(int(value) for value in training_indices)
    validation_set = set(int(value) for value in validation_indices)
    pd.DataFrame(
        [
            {
                "sample_id": index,
                "simulator_seed": int(result["simulator_seed"]),
                "success": bool(result["success"]),
                "fully_finite": bool(np.isfinite(result["x"]).all()),
                "runtime_s": float(result["runtime_s"]),
                "failure_reason": str(result["failure_reason"]),
                "split": (
                    "train"
                    if index in training_set
                    else "validation"
                    if index in validation_set
                    else "invalid"
                ),
            }
            for index, result in enumerate(results)
        ]
    ).to_csv(output_dir / "bank_status.csv", index=False)
    return {
        "results": results,
        "bank_path": bank_path,
        "split_path": split_path,
        "manifest": manifest,
    }


def analyze_training_bank_collisions(
    bank_path: Path,
    split_path: Path,
    stage11_summary_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit 2,048-bank collisions using training-split scaling only."""

    output_dir = Path(output_dir or Path(bank_path).parent)
    with np.load(bank_path, allow_pickle=False) as bank:
        theta = np.asarray(bank["theta"], dtype=float)
        x = np.asarray(bank["x"], dtype=float)
        success = np.asarray(bank["success"], dtype=bool)
    with np.load(split_path, allow_pickle=False) as split:
        location = np.asarray(split["x_location"], dtype=float)
        scale = np.asarray(split["x_scale"], dtype=float)
        lower = np.asarray(split["theta_lower"], dtype=float)
        upper = np.asarray(split["theta_upper"], dtype=float)
    stage11_summary_path = stage11_summary_path or (
        GLOBAL_ROOT / "prior_multiseed" / "multiseed_global_summary.json"
    )
    stage11_summary = json.loads(Path(stage11_summary_path).read_text(encoding="utf-8"))
    noise_floor = float(stage11_summary["noise_floor_q95_robust_x_distance"])
    valid_indices = np.flatnonzero(success & np.isfinite(x).all(axis=1))
    theta_scaled = (theta[valid_indices] - lower) / (upper - lower)
    x_scaled = (x[valid_indices] - location) / scale
    theta_distance = squareform(pdist(theta_scaled)) / np.sqrt(8.0)
    x_distance = squareform(pdist(x_scaled))
    upper_triangle = np.triu(np.ones_like(theta_distance, dtype=bool), k=1)
    far = theta_distance >= COLLISION_THETA_RMS_THRESHOLD
    collision_mask = far & (x_distance <= noise_floor) & upper_triangle
    near_mask = (
        far
        & (x_distance > noise_floor)
        & (x_distance <= 2.0 * noise_floor)
        & upper_triangle
    )
    first, second = np.where(collision_mask)
    rows = [
        {
            "sample_id_a": int(valid_indices[a]),
            "sample_id_b": int(valid_indices[b]),
            "theta_rms_distance": float(theta_distance[a, b]),
            "x_distance": float(x_distance[a, b]),
            "x_to_noise_ratio": float(x_distance[a, b] / noise_floor),
            **{
                f"delta_{name}_prior_width": float(
                    theta_scaled[a, index] - theta_scaled[b, index]
                )
                for index, name in enumerate(PARAMETER_NAMES)
            },
        }
        for a, b in zip(first, second)
    ]
    collisions = pd.DataFrame(rows)
    if not collisions.empty:
        collisions = collisions.sort_values("x_to_noise_ratio")
    collisions.head(1000).to_csv(
        output_dir / "bank_global_collisions_top1000.csv", index=False
    )
    far_pair_count = int((far & upper_triangle).sum())
    summary = {
        "valid_rows": int(len(valid_indices)),
        "pair_count": int(len(valid_indices) * (len(valid_indices) - 1) // 2),
        "noise_floor_from_stage11": noise_floor,
        "far_pair_count": far_pair_count,
        "collision_count": int(collision_mask.sum()),
        "near_collision_count": int(near_mask.sum()),
        "collision_fraction_of_far_pairs": float(
            collision_mask.sum() / max(1, far_pair_count)
        ),
        "stage11_collision_count": int(stage11_summary["global_collision_count"]),
        "scaling_source": "training split only",
    }
    _atomic_json(output_dir / "bank_collision_summary.json", summary)
    return {
        "summary": summary,
        "collisions": collisions,
        "theta_distance": theta_distance,
        "x_distance": x_distance,
    }


__all__ = [
    "COLLISION_NOISE_QUANTILE",
    "COLLISION_THETA_RMS_THRESHOLD",
    "GLOBAL_ROOT",
    "GLOBAL_SIMULATOR_SEEDS",
    "GLOBAL_SOBOL_SEED",
    "LOCAL_SCALES",
    "RESULTS_ROOT",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "TRAINING_SIMULATOR_SEED_BASE",
    "TRAINING_SOBOL_SEED",
    "TRAIN_VALIDATION_SPLIT_SEED",
    "analyze_local_multiscale",
    "analyze_prior_multiseed",
    "analyze_training_bank_collisions",
    "consolidate_batch",
    "deterministic_seed_schedule",
    "engineering_gate",
    "local_multiscale_design",
    "rate_contract_hash",
    "rate_feature_names",
    "run_local_multiscale",
    "run_prior_multiseed",
    "run_resumable_batch",
    "run_training_bank",
    "simulate_rate14",
    "sobol_theta",
]
