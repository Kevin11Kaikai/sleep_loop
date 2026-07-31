"""Resumable matched 8D/7D simulation banks for Figure-10 evaluation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import qmc

from .figure10_protocol import (
    FINAL_SCALE,
    INTERMEDIATE_SCALE,
    MAX_SIMULATION_WORKERS,
    PARAMETER_NAMES_7D,
    PARAMETER_NAMES_8D,
    RESULTS_ROOT,
    SEEDS,
    SHARD_SIZE,
    atomic_json,
    fixed_c_ctx2th,
    prior_bounds_8d,
    rate_contract_hash,
    rate_feature_names,
    sha256_file,
    verify_preregistration,
)
from .route3_global_robustness import (
    deterministic_seed_schedule,
    simulate_rate14,
)


BANK_ROOT = RESULTS_ROOT / "matched_banks"
POOL_SHARDS_PER_LIFETIME = 8


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def paired_theta(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return paired full 8D vectors; 7D track fixes the eighth coordinate."""

    if n < 1 or n & (n - 1):
        raise ValueError("matched training bank size must be a power of two")
    bounds = prior_bounds_8d()
    unit = qmc.Sobol(
        d=8, scramble=True, seed=int(SEEDS["training_sobol"])
    ).random_base2(int(math.log2(n)))
    theta8 = qmc.scale(unit, bounds[:, 0], bounds[:, 1])
    theta7_full = theta8.copy()
    theta7_full[:, 7] = fixed_c_ctx2th()
    return theta8, theta7_full


def _result_arrays(
    rows: Sequence[Mapping[str, Any]],
    sample_ids: np.ndarray,
    paired_ids: np.ndarray,
    track: str,
) -> dict[str, Any]:
    return {
        "sample_id": np.asarray(sample_ids, np.int64),
        "paired_row_id": np.asarray(paired_ids, np.int64),
        "theta_full_8d": np.vstack([row["theta"] for row in rows]).astype(np.float64),
        "x": np.vstack([row["x"] for row in rows]).astype(np.float64),
        "validity": np.vstack([row["validity"] for row in rows]).astype(bool),
        "simulator_seed": np.asarray(
            [row["simulator_seed"] for row in rows], np.int64
        ),
        "success": np.asarray([row["success"] for row in rows], bool),
        "runtime_s": np.asarray([row["runtime_s"] for row in rows], np.float64),
        "failure_reason": np.asarray(
            [str(row["failure_reason"])[:1024] for row in rows], dtype="<U1024"
        ),
        "feature_names": np.asarray(rate_feature_names(), dtype="<U64"),
        "parameter_names_full_8d": np.asarray(PARAMETER_NAMES_8D, dtype="<U32"),
        "track": np.asarray(track, dtype="<U16"),
        "schema_hash": np.asarray(rate_contract_hash(), dtype="<U64"),
        "preregistration_hash": np.asarray(
            verify_preregistration(), dtype="<U64"
        ),
    }


def _load_shard(
    path: Path,
    expected_theta: np.ndarray,
    expected_seeds: np.ndarray,
    track: str,
) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["preregistration_hash"].item()) != verify_preregistration():
                return None
            if str(data["schema_hash"].item()) != rate_contract_hash():
                return None
            if str(data["track"].item()) != track:
                return None
            if not np.array_equal(data["theta_full_8d"], expected_theta):
                return None
            if not np.array_equal(data["simulator_seed"], expected_seeds):
                return None
            return {key: np.asarray(data[key]) for key in data.files}
    except Exception:
        return None


def _retry_path(shard_dir: Path, original_path: Path) -> Path:
    return shard_dir.parent / "retries" / f"{original_path.stem}_retry.npz"


def _effective_shard(
    original_path: Path,
    expected_theta: np.ndarray,
    expected_seeds: np.ndarray,
    track: str,
) -> dict[str, np.ndarray] | None:
    """Load an immutable original shard plus an optional failed-row retry."""

    original = _load_shard(
        original_path, expected_theta, expected_seeds, track
    )
    if original is None:
        return None
    failed_positions = np.flatnonzero(~np.asarray(original["success"], bool))
    if not len(failed_positions):
        return original
    retry_path = _retry_path(original_path.parent, original_path)
    retry = _load_shard(
        retry_path,
        expected_theta[failed_positions],
        expected_seeds[failed_positions],
        track,
    )
    if retry is None or not np.asarray(retry["success"], bool).all():
        return original
    if not np.array_equal(
        np.asarray(retry["sample_id"], np.int64),
        np.asarray(original["sample_id"], np.int64)[failed_positions],
    ):
        raise RuntimeError(f"retry sample IDs mismatch: {retry_path}")
    effective = {
        key: np.asarray(value).copy() for key, value in original.items()
    }
    for key in (
        "theta_full_8d",
        "x",
        "validity",
        "simulator_seed",
        "success",
        "runtime_s",
        "failure_reason",
    ):
        effective[key][failed_positions] = retry[key]
    return effective


def _simulate_shard(
    theta: np.ndarray,
    seeds: np.ndarray,
    sample_ids: np.ndarray,
    paired_ids: np.ndarray,
    track: str,
    workers: int,
    executor: ProcessPoolExecutor | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any] | None] = [None] * len(theta)
    owns_executor = executor is None
    if executor is None:
        executor = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = {
            executor.submit(
                simulate_rate14,
                theta[index],
                int(seeds[index]),
                f"figure10_{track}_{int(sample_ids[index]):07d}",
                "v8a_t13",
            ): index
            for index in range(len(theta))
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                rows[index] = future.result()
            except Exception as exc:
                rows[index] = {
                    "theta": theta[index],
                    "x": np.full(14, np.nan),
                    "validity": np.zeros(14, bool),
                    "simulator_seed": int(seeds[index]),
                    "success": False,
                    "runtime_s": np.nan,
                    "failure_reason": (
                        f"worker_exception: {type(exc).__name__}: {exc}"
                    ),
                }
    finally:
        if owns_executor:
            executor.shutdown()
    if any(row is None for row in rows):
        raise RuntimeError("simulation shard ended with missing rows")
    return _result_arrays(
        [dict(row) for row in rows if row is not None],
        sample_ids,
        paired_ids,
        track,
    )


def _repair_failed_rows(
    original_path: Path,
    original: dict[str, np.ndarray],
    track: str,
    workers: int,
    executor: ProcessPoolExecutor,
) -> dict[str, np.ndarray]:
    """Retry only unsuccessful rows, preserving the immutable original shard."""

    failed_positions = np.flatnonzero(~np.asarray(original["success"], bool))
    if not len(failed_positions):
        return original
    retry_path = _retry_path(original_path.parent, original_path)
    retry_path.parent.mkdir(parents=True, exist_ok=True)
    expected_theta = np.asarray(original["theta_full_8d"], float)[
        failed_positions
    ]
    expected_seeds = np.asarray(original["simulator_seed"], np.int64)[
        failed_positions
    ]
    existing = _load_shard(
        retry_path, expected_theta, expected_seeds, track
    )
    if existing is None:
        arrays = _simulate_shard(
            expected_theta,
            expected_seeds,
            np.asarray(original["sample_id"], np.int64)[failed_positions],
            np.asarray(original["paired_row_id"], np.int64)[failed_positions],
            track,
            workers,
            executor=executor,
        )
        if not np.asarray(arrays["success"], bool).all():
            raise RuntimeError(
                f"failed-row retry did not resolve every row in {original_path}"
            )
        _atomic_npz(
            retry_path,
            **arrays,
            retry_scope=np.asarray(
                "same theta and simulator seed; original failure preserved",
                dtype="<U96",
            ),
        )
        existing = _load_shard(
            retry_path, expected_theta, expected_seeds, track
        )
    if existing is None or not np.asarray(existing["success"], bool).all():
        raise RuntimeError(f"failed-row retry reload failed: {retry_path}")
    effective = _effective_shard(
        original_path,
        np.asarray(original["theta_full_8d"], float),
        np.asarray(original["simulator_seed"], np.int64),
        track,
    )
    if effective is None or not np.asarray(effective["success"], bool).all():
        raise RuntimeError(f"effective retry merge failed: {original_path}")
    return effective


def _manifest(track: str, target: int, shard_dir: Path) -> dict[str, Any]:
    shard_files = sorted(shard_dir.glob("shard_*.npz"))
    valid = failed = completed = raw_failed = retry_resolved = 0
    simulator_runtime = 0.0
    checksums = []
    for path in shard_files:
        with np.load(path, allow_pickle=False) as data:
            theta = np.asarray(data["theta_full_8d"], float)
            seeds = np.asarray(data["simulator_seed"], np.int64)
            original_success = np.asarray(data["success"], bool)
        effective = _effective_shard(path, theta, seeds, track)
        if effective is None:
            raise RuntimeError(f"manifest could not reload shard: {path}")
        success = np.asarray(effective["success"], bool)
        completed += len(success)
        valid += int(success.sum())
        failed += int((~success).sum())
        raw_failed += int((~original_success).sum())
        retry_resolved += int(np.sum((~original_success) & success))
        simulator_runtime += float(np.nansum(effective["runtime_s"]))
        checksums.append(
            {
                "path": path.relative_to(RESULTS_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "track": track,
        "target": target,
        "completed": completed,
        "valid": valid,
        "failed": failed,
        "raw_original_failed": raw_failed,
        "retry_resolved": retry_resolved,
        "pending": target - completed,
        "failure_rate_completed": float(failed / completed) if completed else 0.0,
        "sum_simulator_runtime_s": simulator_runtime,
        "shard_size": SHARD_SIZE,
        "shards": checksums,
        "preregistration_hash": verify_preregistration(),
    }


def run_track(
    track: str,
    target: int = FINAL_SCALE,
    workers: int = MAX_SIMULATION_WORKERS,
) -> dict[str, Any]:
    """Generate one immutable-shard track, resuming completed valid shards."""

    verify_preregistration()
    if track not in {"8d", "7d"}:
        raise ValueError("track must be '8d' or '7d'")
    if target not in {INTERMEDIATE_SCALE, FINAL_SCALE}:
        raise ValueError("target is not a preregistered scale")
    theta8, theta7 = paired_theta(FINAL_SCALE)
    theta = theta8[:target] if track == "8d" else theta7[:target]
    seeds = deterministic_seed_schedule(
        FINAL_SCALE, int(SEEDS["training_simulator_base"])
    )[:target]
    shard_dir = BANK_ROOT / track / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = BANK_ROOT / track / "progress_manifest.json"
    wall_started = perf_counter()
    starts = list(range(0, target, SHARD_SIZE))
    for block in range(0, len(starts), POOL_SHARDS_PER_LIFETIME):
        block_starts = starts[block : block + POOL_SHARDS_PER_LIFETIME]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for start in block_starts:
                stop = min(start + SHARD_SIZE, target)
                path = shard_dir / f"shard_{start:07d}_{stop:07d}.npz"
                original = _load_shard(
                    path, theta[start:stop], seeds[start:stop], track
                )
                if original is None:
                    arrays = _simulate_shard(
                        theta[start:stop],
                        seeds[start:stop],
                        np.arange(start, stop, dtype=np.int64),
                        np.arange(start, stop, dtype=np.int64),
                        track,
                        workers,
                        executor=executor,
                    )
                    _atomic_npz(path, **arrays)
                    original = _load_shard(
                        path, theta[start:stop], seeds[start:stop], track
                    )
                    if original is None:
                        raise RuntimeError(
                            f"new shard failed independent reload: {path}"
                        )
                if not np.asarray(original["success"], bool).all():
                    _repair_failed_rows(
                        path, original, track, workers, executor
                    )
                manifest = _manifest(track, target, shard_dir)
                manifest["wall_time_this_call_s"] = float(
                    perf_counter() - wall_started
                )
                manifest["worker_recycle_shards"] = (
                    POOL_SHARDS_PER_LIFETIME
                )
                atomic_json(manifest_path, manifest)
                if manifest["completed"] % (SHARD_SIZE * 8) == 0:
                    print(
                        f"{track}: completed={manifest['completed']}/{target} "
                        f"failed={manifest['failed']} "
                        f"wall={manifest['wall_time_this_call_s'] / 3600:.2f} h",
                        flush=True,
                    )
    return _manifest(track, target, shard_dir)


def _combine_track(track: str, target: int) -> Path:
    theta8, theta7 = paired_theta(FINAL_SCALE)
    expected_theta = theta8[:target] if track == "8d" else theta7[:target]
    expected_seeds = deterministic_seed_schedule(
        FINAL_SCALE, int(SEEDS["training_simulator_base"])
    )[:target]
    shards = sorted((BANK_ROOT / track / "shards").glob("shard_*.npz"))
    arrays: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "sample_id",
            "paired_row_id",
            "theta_full_8d",
            "x",
            "validity",
            "simulator_seed",
            "success",
            "runtime_s",
            "failure_reason",
        )
    }
    for path in shards:
        with np.load(path, allow_pickle=False) as data:
            theta = np.asarray(data["theta_full_8d"], float)
            seeds = np.asarray(data["simulator_seed"], np.int64)
        effective = _effective_shard(path, theta, seeds, track)
        if effective is None:
            raise RuntimeError(f"could not load effective shard: {path}")
        if str(effective["track"].item()) != track:
            raise RuntimeError("track shard label drift")
        for key in arrays:
            arrays[key].append(np.asarray(effective[key]))
    combined = {
        key: np.concatenate(parts, axis=0)[:target] for key, parts in arrays.items()
    }
    if len(combined["sample_id"]) != target:
        raise RuntimeError(f"{track} bank has incomplete row count")
    if not np.array_equal(combined["theta_full_8d"], expected_theta):
        raise RuntimeError(f"{track} bank theta/order mismatch")
    if not np.array_equal(combined["simulator_seed"], expected_seeds):
        raise RuntimeError(f"{track} bank simulator seed mismatch")
    theta_inferred = (
        combined["theta_full_8d"]
        if track == "8d"
        else combined["theta_full_8d"][:, :7]
    )
    output = BANK_ROOT / track / f"figure10_{track}_bank_{target}.npz"
    _atomic_npz(
        output,
        **combined,
        theta=theta_inferred,
        parameter_names=np.asarray(
            PARAMETER_NAMES_8D if track == "8d" else PARAMETER_NAMES_7D,
            dtype="<U32",
        ),
        parameter_names_full_8d=np.asarray(PARAMETER_NAMES_8D, dtype="<U32"),
        feature_names=np.asarray(rate_feature_names(), dtype="<U64"),
        track=np.asarray(track, dtype="<U16"),
        schema_hash=np.asarray(rate_contract_hash(), dtype="<U64"),
        preregistration_hash=np.asarray(verify_preregistration(), dtype="<U64"),
    )
    with np.load(output, allow_pickle=False) as reloaded:
        if any(reloaded[key].dtype == object for key in reloaded.files):
            raise RuntimeError("combined bank contains object arrays")
        if reloaded["theta"].shape != (
            target,
            8 if track == "8d" else 7,
        ):
            raise RuntimeError("combined bank inferred theta shape mismatch")
    return output


def finalize_matched_banks(target: int = FINAL_SCALE) -> dict[str, Any]:
    """Consolidate tracks, build paired split, and fit train-only scalers."""

    verify_preregistration()
    paths = {track: _combine_track(track, target) for track in ("8d", "7d")}
    loaded = {}
    for track, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            loaded[track] = {key: np.asarray(data[key]) for key in data.files}
    paired_valid = loaded["8d"]["success"] & loaded["7d"]["success"]
    valid_ids = np.flatnonzero(paired_valid)
    rng = np.random.default_rng(int(SEEDS["train_validation_split"]))
    order = rng.permutation(valid_ids)
    split_at = int(np.floor(0.8 * len(order)))
    train_ids = np.sort(order[:split_at])
    validation_ids = np.sort(order[split_at:])
    if np.intersect1d(train_ids, validation_ids).size:
        raise RuntimeError("paired train/validation leakage")
    split_path = BANK_ROOT / "paired_split_and_scaling.npz"
    scalers = {}
    for track in ("8d", "7d"):
        x = loaded[track]["x"]
        location = np.median(x[train_ids], axis=0)
        q25, q75 = np.percentile(x[train_ids], [25, 75], axis=0)
        scale = q75 - q25
        fallback = np.std(x[train_ids], axis=0)
        scale = np.where(scale > 1e-12, scale, fallback)
        scale = np.where(scale > 1e-12, scale, 1.0)
        if not np.isfinite(location).all() or not np.isfinite(scale).all():
            raise RuntimeError(f"{track} training-only scaler is nonfinite")
        scalers[track] = (location, scale)
    _atomic_npz(
        split_path,
        train_paired_row_ids=train_ids.astype(np.int64),
        validation_paired_row_ids=validation_ids.astype(np.int64),
        paired_valid_mask=paired_valid.astype(bool),
        x_location_8d=scalers["8d"][0],
        x_scale_8d=scalers["8d"][1],
        x_location_7d=scalers["7d"][0],
        x_scale_7d=scalers["7d"][1],
        theta_lower_8d=prior_bounds_8d()[:, 0],
        theta_upper_8d=prior_bounds_8d()[:, 1],
        theta_lower_7d=prior_bounds_8d()[:7, 0],
        theta_upper_7d=prior_bounds_8d()[:7, 1],
        feature_names=np.asarray(rate_feature_names(), dtype="<U64"),
        preregistration_hash=np.asarray(verify_preregistration(), dtype="<U64"),
    )
    failure8 = 1 - loaded["8d"]["success"].mean()
    failure7 = 1 - loaded["7d"]["success"].mean()
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_per_track": target,
        "8d_valid": int(loaded["8d"]["success"].sum()),
        "8d_failed": int((~loaded["8d"]["success"]).sum()),
        "8d_failure_rate": float(failure8),
        "7d_valid": int(loaded["7d"]["success"].sum()),
        "7d_failed": int((~loaded["7d"]["success"]).sum()),
        "7d_failure_rate": float(failure7),
        "paired_valid": int(paired_valid.sum()),
        "train_rows": len(train_ids),
        "validation_rows": len(validation_ids),
        "bank_paths": {
            key: value.relative_to(RESULTS_ROOT).as_posix()
            for key, value in paths.items()
        },
        "bank_sha256": {key: sha256_file(value) for key, value in paths.items()},
        "split_path": split_path.relative_to(RESULTS_ROOT).as_posix(),
        "split_sha256": sha256_file(split_path),
        "training_allowed": bool(
            failure8 <= 0.01
            and failure7 <= 0.01
            and paired_valid.sum() >= 0.99 * target
        ),
        "selection_bias_warning": (
            "All failures remain in bank artifacts. Training uses only the "
            "paired-valid intersection."
        ),
        "preregistration_hash": verify_preregistration(),
    }
    atomic_json(BANK_ROOT / "matched_bank_manifest.json", summary)
    return summary


__all__ = [
    "BANK_ROOT",
    "finalize_matched_banks",
    "paired_theta",
    "run_custom_dataset",
    "run_track",
]


def run_custom_dataset(
    theta_full_8d: np.ndarray,
    simulator_seeds: Sequence[int],
    output_dir: Path,
    track: str,
    dataset_id: str,
    workers: int = MAX_SIMULATION_WORKERS,
) -> Path:
    """Run/resume an arbitrary preregistered diagnostic dataset in shards."""

    verify_preregistration()
    theta = np.asarray(theta_full_8d, float)
    seeds = np.asarray(simulator_seeds, np.int64)
    if theta.ndim != 2 or theta.shape[1] != 8 or len(theta) != len(seeds):
        raise ValueError("custom theta must be (N,8) with one seed per row")
    if track not in {"8d", "7d"}:
        raise ValueError("custom dataset track must be 8d or 7d")
    output_dir = Path(output_dir)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    starts = list(range(0, len(theta), SHARD_SIZE))
    for block in range(0, len(starts), POOL_SHARDS_PER_LIFETIME):
        block_starts = starts[block : block + POOL_SHARDS_PER_LIFETIME]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for start in block_starts:
                stop = min(start + SHARD_SIZE, len(theta))
                path = shard_dir / f"shard_{start:07d}_{stop:07d}.npz"
                loaded = _load_shard(
                    path, theta[start:stop], seeds[start:stop], track
                )
                if loaded is None:
                    arrays = _simulate_shard(
                        theta[start:stop],
                        seeds[start:stop],
                        np.arange(start, stop, dtype=np.int64),
                        np.arange(start, stop, dtype=np.int64),
                        track,
                        workers,
                        executor=executor,
                    )
                    _atomic_npz(
                        path,
                        **arrays,
                        dataset_id=np.asarray(dataset_id, dtype="<U96"),
                    )
                    loaded = _load_shard(
                        path, theta[start:stop], seeds[start:stop], track
                    )
                    if loaded is None:
                        raise RuntimeError(
                            f"custom shard failed reload: {path}"
                        )
                if not np.asarray(loaded["success"], bool).all():
                    _repair_failed_rows(
                        path, loaded, track, workers, executor
                    )
                completed = stop
                if (
                    completed % (SHARD_SIZE * 8) == 0
                    or completed == len(theta)
                ):
                    print(
                        f"{dataset_id}/{track}: "
                        f"completed={completed}/{len(theta)}",
                        flush=True,
                    )
    parts = sorted(shard_dir.glob("shard_*.npz"))
    collected: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "sample_id",
            "paired_row_id",
            "theta_full_8d",
            "x",
            "validity",
            "simulator_seed",
            "success",
            "runtime_s",
            "failure_reason",
        )
    }
    for path in parts:
        with np.load(path, allow_pickle=False) as data:
            shard_theta = np.asarray(data["theta_full_8d"], float)
            shard_seeds = np.asarray(data["simulator_seed"], np.int64)
        effective = _effective_shard(
            path, shard_theta, shard_seeds, track
        )
        if effective is None:
            raise RuntimeError(f"custom effective shard failed: {path}")
        for key in collected:
            collected[key].append(np.asarray(effective[key]))
    combined = {
        key: np.concatenate(value, axis=0)[: len(theta)]
        for key, value in collected.items()
    }
    if len(combined["sample_id"]) != len(theta):
        raise RuntimeError(f"{dataset_id}/{track} consolidation is incomplete")
    if not np.array_equal(combined["theta_full_8d"], theta):
        raise RuntimeError(f"{dataset_id}/{track} theta mismatch")
    if not np.array_equal(combined["simulator_seed"], seeds):
        raise RuntimeError(f"{dataset_id}/{track} seed mismatch")
    inferred_theta = theta if track == "8d" else theta[:, :7]
    output = output_dir / f"{dataset_id}_{track}.npz"
    _atomic_npz(
        output,
        **combined,
        theta=inferred_theta,
        parameter_names=np.asarray(
            PARAMETER_NAMES_8D if track == "8d" else PARAMETER_NAMES_7D,
            dtype="<U32",
        ),
        parameter_names_full_8d=np.asarray(PARAMETER_NAMES_8D, dtype="<U32"),
        feature_names=np.asarray(rate_feature_names(), dtype="<U64"),
        dataset_id=np.asarray(dataset_id, dtype="<U96"),
        track=np.asarray(track, dtype="<U16"),
        schema_hash=np.asarray(rate_contract_hash(), dtype="<U64"),
        preregistration_hash=np.asarray(verify_preregistration(), dtype="<U64"),
    )
    with np.load(output, allow_pickle=False) as data:
        if any(data[key].dtype == object for key in data.files):
            raise RuntimeError("custom dataset contains object arrays")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "track": track,
        "attempted": len(theta),
        "valid": int(combined["success"].sum()),
        "failed": int((~combined["success"]).sum()),
        "failure_rate": float(1 - combined["success"].mean()),
        "sum_simulator_runtime_s": float(np.nansum(combined["runtime_s"])),
        "path": output.relative_to(RESULTS_ROOT).as_posix(),
        "sha256": sha256_file(output),
        "preregistration_hash": verify_preregistration(),
    }
    atomic_json(output_dir / f"{dataset_id}_{track}_manifest.json", manifest)
    return output
