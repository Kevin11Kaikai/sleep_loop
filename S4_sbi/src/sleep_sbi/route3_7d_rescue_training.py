"""Bounded NPE training and development-only calibration for the 7D rescue."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader, TensorDataset
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

from .route3_7d_experiment import PARAMETER_NAMES_7D, atomic_json, atomic_npz
from .route3_7d_rescue import (
    DEVELOPMENT_ROOT,
    RESCUE_LOCKED,
    TRAINING_ROOT,
    coverage_table,
    load_rescue_training_data,
    read_rescue_preregistration,
    sbc_table,
    sha256_file,
    verify_rescue_preregistration,
)


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _architecture(config_id: str) -> dict[str, Any]:
    for entry in read_rescue_preregistration()["allowed_rescue_configurations"]:
        if entry["id"] == config_id:
            return dict(entry["architecture"])
    raise KeyError(f"unknown preregistered rescue configuration {config_id}")


def _model_seeds(config_id: str) -> tuple[int, ...]:
    key = (
        "network_initialization_baseline"
        if config_id == "maf64_t5"
        else "network_initialization_wide"
    )
    values = tuple(int(v) for v in read_rescue_preregistration()["seeds"][key])
    if len(values) != 5 or len(set(values)) != 5:
        raise RuntimeError("rescue ensemble must contain five independent seeds")
    return values


def _network(data, config_id: str, seed: int, device: torch.device):
    torch.manual_seed(int(seed))
    builder = posterior_nn(**_architecture(config_id))
    theta = torch.as_tensor(data.theta_train, dtype=torch.float32, device=device)
    x = torch.as_tensor(data.x_train, dtype=torch.float32, device=device)
    return builder(theta, x).to(device)


def _loss(network, loader, device, optimizer=None, clip_max_norm: float = 5.0) -> float:
    training = optimizer is not None
    network.train(training)
    total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for theta, x in loader:
            theta, x = theta.to(device), x.to(device)
            loss = network.loss(theta, condition=x).mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), clip_max_norm)
                optimizer.step()
            total += float(loss.detach().cpu()) * len(theta)
            count += len(theta)
    return total / count


def train_member(
    config_id: str, seed: int, device_name: str | None = None
) -> dict[str, Any]:
    rescue_hash = verify_rescue_preregistration()
    data = load_rescue_training_data()
    policy = dict(read_rescue_preregistration()["training_policy"])
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    network = _network(data, config_id, seed, device)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=float(policy["learning_rate"]),
        weight_decay=float(policy["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(int(seed) + 100_000)
    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(data.theta_train, dtype=torch.float32),
            torch.as_tensor(data.x_train, dtype=torch.float32),
        ),
        batch_size=int(policy["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(data.theta_validation, dtype=torch.float32),
            torch.as_tensor(data.x_validation, dtype=torch.float32),
        ),
        batch_size=int(policy["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    output_dir = TRAINING_ROOT / config_id / f"member_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last_checkpoint.pt"
    best_path = output_dir / "best_checkpoint.pt"
    history: list[dict[str, Any]] = []
    start_epoch, best_epoch, stale = 0, -1, 0
    best_validation = np.inf
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if (
            checkpoint.get("rescue_preregistration_hash") == rescue_hash
            and checkpoint.get("config_id") == config_id
            and int(checkpoint.get("seed")) == int(seed)
        ):
            network.load_state_dict(checkpoint["network_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            history = list(checkpoint["history"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_epoch = int(checkpoint["best_epoch"])
            best_validation = float(checkpoint["best_validation"])
            stale = int(checkpoint["stale_epochs"])
    started = perf_counter()
    already_stopped = (
        stale >= int(policy["early_stopping_patience"])
        or start_epoch >= int(policy["max_epochs"])
    )
    epochs = range(0) if already_stopped else range(start_epoch, int(policy["max_epochs"]))
    for epoch in epochs:
        train_loss = _loss(
            network, train_loader, device, optimizer, float(policy["clip_max_norm"])
        )
        validation_loss = _loss(
            network, validation_loader, device, None, float(policy["clip_max_norm"])
        )
        improved = validation_loss < best_validation - float(policy["minimum_improvement"])
        if improved:
            best_validation, best_epoch, stale = validation_loss, epoch, 0
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "improved": bool(improved),
            }
        )
        payload = {
            "config_id": config_id,
            "seed": int(seed),
            "epoch": epoch,
            "network_state": network.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_validation": best_validation,
            "best_epoch": best_epoch,
            "stale_epochs": stale,
            "architecture": _architecture(config_id),
            "training_policy": policy,
            "rescue_preregistration_hash": rescue_hash,
            "theta_dimension": 7,
            "x_dimension": 14,
        }
        _atomic_torch(last_path, payload)
        if improved:
            _atomic_torch(best_path, payload)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        if stale >= int(policy["early_stopping_patience"]):
            break
    runtime = perf_counter() - started
    if not best_path.exists():
        raise RuntimeError(f"rescue member {config_id}/{seed} produced no checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    summary = {
        "config_id": config_id,
        "seed": int(seed),
        "device": str(device),
        "status": "trained",
        "epochs_completed": len(best["history"]),
        "best_epoch": int(best["best_epoch"]),
        "best_validation_loss": float(best["best_validation"]),
        "final_train_loss": float(best["history"][-1]["train_loss"]),
        "runtime_this_call_s": float(runtime),
        "checkpoint": best_path.as_posix(),
        "checkpoint_sha256": sha256_file(best_path),
        "rescue_preregistration_hash": rescue_hash,
    }
    atomic_json(output_dir / "training_summary.json", summary)
    return summary


def train_configuration(config_id: str) -> dict[str, Any]:
    verify_rescue_preregistration()
    output = TRAINING_ROOT / config_id
    manifest_path = output / "ensemble_manifest.json"
    seeds = _model_seeds(config_id)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("rescue_preregistration_hash")
            == verify_rescue_preregistration()
            and manifest.get("member_seeds") == list(seeds)
            and all(
                (output / f"member_{seed}" / "best_checkpoint.pt").exists()
                for seed in seeds
            )
        ):
            return manifest
    started = perf_counter()
    members = [train_member(config_id, seed) for seed in seeds]
    data = load_rescue_training_data()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_id": config_id,
        "architecture": _architecture(config_id),
        "training_policy": read_rescue_preregistration()["training_policy"],
        "member_seeds": list(seeds),
        "weights": [0.2] * 5,
        "mixture_rule": "equal-count mixture; no learned or result-driven weights",
        "training_rows": len(data.training_indices),
        "validation_rows": len(data.validation_indices),
        "members": members,
        "runtime_this_call_s": float(perf_counter() - started),
        "rescue_preregistration_hash": verify_rescue_preregistration(),
        "packages": {
            name: metadata.version(name) for name in ("sbi", "torch", "numpy", "scipy")
        },
    }
    atomic_json(manifest_path, manifest)
    pd.DataFrame(members).to_csv(output / "training_summary.csv", index=False)
    return manifest


def load_member(config_id: str, seed: int, device_name: str = "cpu") -> DirectPosterior:
    data = load_rescue_training_data()
    device = torch.device(device_name)
    network = _network(data, config_id, seed, device)
    checkpoint_path = (
        TRAINING_ROOT / config_id / f"member_{seed}" / "best_checkpoint.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["rescue_preregistration_hash"] != verify_rescue_preregistration():
        raise RuntimeError("rescue checkpoint preregistration mismatch")
    network.load_state_dict(checkpoint["network_state"])
    network.eval()
    prior = BoxUniform(
        low=torch.zeros(7, device=device),
        high=torch.ones(7, device=device),
        device=device_name,
    )
    return DirectPosterior(posterior_estimator=network, prior=prior, device=device)


def load_ensemble(config_id: str) -> list[DirectPosterior]:
    return [load_member(config_id, seed) for seed in _model_seeds(config_id)]


def sample_member(
    posterior: DirectPosterior, x_scaled: np.ndarray, n: int, seed: int
) -> np.ndarray:
    torch.manual_seed(int(seed))
    result = posterior.sample(
        (int(n),),
        x=torch.as_tensor(x_scaled, dtype=torch.float32),
        show_progress_bars=False,
        reject_outside_prior=True,
    ).detach().cpu().numpy()
    if result.shape != (n, 7) or not np.isfinite(result).all():
        raise RuntimeError("rescue posterior returned nonfinite/wrong-shape samples")
    if ((result < 0) | (result > 1)).any():
        raise RuntimeError("rescue posterior samples escaped normalized prior")
    return result


def sample_equal_ensemble(
    members: Sequence[DirectPosterior],
    x_scaled: np.ndarray,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.full(len(members), n // len(members), dtype=int)
    counts[: n % len(members)] += 1
    values, labels = [], []
    for index, (member, count) in enumerate(zip(members, counts)):
        values.append(
            sample_member(member, x_scaled, int(count), seed + index * 100_003)
        )
        labels.append(np.full(count, index, dtype=np.int8))
    samples = np.vstack(values)
    member_labels = np.concatenate(labels)
    order = np.random.default_rng(seed + 900_001).permutation(n)
    return samples[order], member_labels[order]


def sample_development(config_id: str) -> Path:
    verify_rescue_preregistration()
    data = load_rescue_training_data()
    members = load_ensemble(config_id)
    dataset_path = (
        DEVELOPMENT_ROOT / "dataset" / "route3_7d_rescue_development_512.npz"
    )
    with np.load(dataset_path, allow_pickle=False) as dataset:
        theta = np.asarray(dataset["theta"], float)
        x = np.asarray(dataset["x"], float)
        success = np.asarray(dataset["success"], bool)
    if not success.all():
        raise RuntimeError("development set contains failed simulations")
    theta_unit = (theta - data.theta_lower) / (data.theta_upper - data.theta_lower)
    x_scaled = (x - data.x_location) / data.x_scale
    base_seed = int(
        read_rescue_preregistration()["seeds"]["development_posterior_sampling"]
    )
    output = DEVELOPMENT_ROOT / "posterior_samples" / config_id
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    n_cases, n_samples = 512, 2048
    samples = np.empty((n_cases, n_samples, 7), np.float32)
    labels = np.empty((n_cases, n_samples), np.int8)
    runtime = np.empty(n_cases, float)
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
                        samples[case] = check["samples"]
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
        samples[case] = case_samples
        labels[case] = case_labels
        runtime[case] = perf_counter() - started
        atomic_npz(
            path,
            case=np.asarray(case, np.int64),
            theta_true=theta_unit[case],
            x_scaled=x_scaled[case],
            samples=samples[case],
            member_labels=labels[case],
            runtime_s=np.asarray(runtime[case]),
            config_id=np.asarray(config_id, dtype="<U32"),
            rescue_preregistration_hash=np.asarray(
                verify_rescue_preregistration(), dtype="<U64"
            ),
        )
    consolidated = output / "development_samples.npz"
    atomic_npz(
        consolidated,
        theta_true_normalized=theta_unit,
        x_scaled=x_scaled,
        samples=samples,
        member_labels=labels,
        runtime_s=runtime,
        config_id=np.asarray(config_id, dtype="<U32"),
        parameter_names=np.asarray(PARAMETER_NAMES_7D, dtype="<U32"),
        rescue_preregistration_hash=np.asarray(
            verify_rescue_preregistration(), dtype="<U64"
        ),
    )
    return consolidated


def fit_rank_calibrator(theta: np.ndarray, raw_samples: np.ndarray) -> np.ndarray:
    """Fit marginal rank CDFs; the same algorithm is applied to all parameters."""

    ranks = np.empty((len(theta), 7), float)
    for j in range(7):
        counts = (raw_samples[:, :, j] < theta[:, None, j]).sum(axis=1)
        ranks[:, j] = (counts + 0.5) / (raw_samples.shape[1] + 1)
    return np.sort(ranks, axis=0)


def apply_rank_calibrator(raw_samples: np.ndarray, sorted_ranks: np.ndarray) -> np.ndarray:
    """Apply monotone marginal recalibration while preserving sample copula ranks."""

    raw_samples = np.asarray(raw_samples, float)
    n_cases, n_samples, n_parameters = raw_samples.shape
    if sorted_ranks.ndim != 2 or sorted_ranks.shape[1] != n_parameters:
        raise ValueError("calibrator shape mismatch")
    empirical_grid = (np.arange(len(sorted_ranks)) + 0.5) / len(sorted_ranks)
    sample_grid = (np.arange(n_samples) + 0.5) / n_samples
    transformed = np.empty_like(raw_samples)
    for case in range(n_cases):
        for j in range(n_parameters):
            target_probability = np.interp(
                sample_grid,
                empirical_grid,
                sorted_ranks[:, j],
                left=sorted_ranks[0, j],
                right=sorted_ranks[-1, j],
            )
            target_probability = np.clip(
                target_probability, 0.5 / n_samples, 1 - 0.5 / n_samples
            )
            target_values = np.quantile(
                raw_samples[case, :, j], target_probability
            )
            order = np.argsort(raw_samples[case, :, j], kind="mergesort")
            transformed[case, order, j] = target_values
    if not np.isfinite(transformed).all() or ((transformed < 0) | (transformed > 1)).any():
        raise RuntimeError("rank calibrator produced invalid samples")
    return transformed


def marginal_crps(theta: np.ndarray, samples: np.ndarray) -> float:
    """Deterministic sample CRPS estimate, averaged over cases and parameters."""

    first = np.mean(np.abs(samples - theta[:, None, :]), axis=1)
    # A deterministic half-vs-half pairing estimates E|X-X'| without O(S^2) memory.
    half = samples.shape[1] // 2
    second = np.mean(
        np.abs(samples[:, :half, :] - samples[:, -half:, :]), axis=1
    )
    return float(np.mean(first - 0.5 * second))


def _candidate_metrics(
    config_id: str,
    method: str,
    theta: np.ndarray,
    samples: np.ndarray,
) -> dict[str, Any]:
    coverage = coverage_table(theta, samples, f"{config_id}:{method}", (0.5, 0.8, 0.9))
    q05, q95 = np.quantile(samples, [0.05, 0.95], axis=1)
    width = np.median(q95 - q05, axis=0)
    c8090 = coverage[coverage.nominal_level.isin([0.8, 0.9])]
    compatible = c8090.groupby("parameter").nominal_inside_wilson.all()
    severe = int(coverage.groupby("parameter").severe_undercoverage.any().sum())
    contracted = {
        name
        for name, value in zip(PARAMETER_NAMES_7D, width)
        if value < 0.9
    }
    reasonable = set(compatible[compatible].index)
    median = np.median(samples, axis=1)
    prior_error = np.mean(np.abs(theta - 0.5))
    posterior_error = np.mean(np.abs(theta - median))
    return {
        "config_id": config_id,
        "method": method,
        "formal_go_eligible": method == "raw",
        "coverage_compatible_parameter_count": int(compatible.sum()),
        "severe_undercoverage_parameter_count": severe,
        "contraction_with_coverage_parameter_count": len(contracted & reasonable),
        "aggregate_coverage_error": float(
            np.mean(np.abs(coverage.empirical_coverage - coverage.nominal_level))
        ),
        "marginal_crps": marginal_crps(theta, samples),
        "overall_recovery_improvement_fraction": float(
            1 - posterior_error / prior_error
        ),
        "median_90_ci_width_mean": float(width.mean()),
        "all_original_calibration_contraction_requirements": bool(
            compatible.sum() >= 6 and severe == 0 and len(contracted & reasonable) >= 6
        ),
    }


def evaluate_and_select_development() -> dict[str, Any]:
    """Evaluate bounded candidates and freeze one pipeline before final data."""

    verify_rescue_preregistration()
    candidates: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], np.ndarray] = {}
    theta_reference = None
    calibrators: dict[str, np.ndarray] = {}
    for config_id in ("maf64_t5", "maf128_t8"):
        path = (
            DEVELOPMENT_ROOT
            / "posterior_samples"
            / config_id
            / "development_samples.npz"
        )
        with np.load(path, allow_pickle=False) as data:
            theta = np.asarray(data["theta_true_normalized"], float)
            raw = np.asarray(data["samples"], float)
        if theta_reference is None:
            theta_reference = theta
        elif not np.array_equal(theta_reference, theta):
            raise RuntimeError("development theta drift across configurations")
        sorted_ranks = fit_rank_calibrator(theta, raw)
        calibrated = apply_rank_calibrator(raw, sorted_ranks)
        calibrators[config_id] = sorted_ranks
        cache[(config_id, "raw")] = raw
        cache[(config_id, "empirical_rank_calibrated")] = calibrated
        for method, values in (
            ("raw", raw),
            ("empirical_rank_calibrated", calibrated),
        ):
            candidates.append(_candidate_metrics(config_id, method, theta, values))
            coverage_table(
                theta, values, f"{config_id}:{method}", (0.5, 0.8, 0.9)
            ).to_csv(
                DEVELOPMENT_ROOT / f"coverage_{config_id}_{method}.csv", index=False
            )
            sbc_table(theta, values, f"{config_id}:{method}").to_csv(
                DEVELOPMENT_ROOT / f"sbc_{config_id}_{method}.csv", index=False
            )
    frame = pd.DataFrame(candidates)
    frame.to_csv(DEVELOPMENT_ROOT / "candidate_pipeline_metrics.csv", index=False)
    raw_pass = frame[
        (frame.method == "raw")
        & frame.all_original_calibration_contraction_requirements
    ]
    if len(raw_pass):
        pool = raw_pass
    else:
        calibrated_pass = frame[
            (frame.method == "empirical_rank_calibrated")
            & frame.all_original_calibration_contraction_requirements
        ]
        pool = calibrated_pass if len(calibrated_pass) else frame
    complexity = {"maf64_t5": 0, "maf128_t8": 1}
    pool = pool.assign(
        complexity=pool.config_id.map(complexity),
        calibrated=(pool.method != "raw").astype(int),
    ).sort_values(
        [
            "aggregate_coverage_error",
            "marginal_crps",
            "calibrated",
            "complexity",
        ],
        kind="mergesort",
    )
    selected = pool.iloc[0].to_dict()
    config_id = str(selected["config_id"])
    method = str(selected["method"])
    calibrator_path = DEVELOPMENT_ROOT / "rank_calibrator.npz"
    if method == "empirical_rank_calibrated":
        atomic_npz(
            calibrator_path,
            sorted_development_ranks=calibrators[config_id],
            parameter_names=np.asarray(PARAMETER_NAMES_7D, dtype="<U32"),
            config_id=np.asarray(config_id, dtype="<U32"),
            method=np.asarray(method, dtype="<U64"),
            rescue_preregistration_hash=np.asarray(
                verify_rescue_preregistration(), dtype="<U64"
            ),
        )
    selection = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_config_id": config_id,
        "selected_method": method,
        "formal_go_eligible": method == "raw",
        "selection_metrics": {
            key: (
                bool(value)
                if isinstance(value, (np.bool_, bool))
                else int(value)
                if isinstance(value, (np.integer,))
                else float(value)
                if isinstance(value, (np.floating,))
                else value
            )
            for key, value in selected.items()
            if key not in ("complexity", "calibrated")
        },
        "selection_rule": read_rescue_preregistration()["selection_rule"],
        "calibrator": (
            {
                "method": "marginal empirical-rank recalibration",
                "artifact": calibrator_path.as_posix(),
                "artifact_sha256": sha256_file(calibrator_path),
                "fit_cases": 512,
                "fit_scope": "development only",
                "joint_dependence_policy": "within-case marginal sample ranks retained",
                "formal_go_eligibility": False,
            }
            if method != "raw"
            else None
        ),
        "rescue_preregistration_hash": verify_rescue_preregistration(),
        "rescue_locked_sha256": sha256_file(RESCUE_LOCKED),
    }
    locked = DEVELOPMENT_ROOT / "primary_pipeline_locked.json"
    if locked.exists():
        existing = json.loads(locked.read_text(encoding="utf-8"))
        if existing != selection:
            # Timestamp differences are not silently overwritten.
            existing_no_time = {k: v for k, v in existing.items() if k != "created_utc"}
            selected_no_time = {k: v for k, v in selection.items() if k != "created_utc"}
            if existing_no_time != selected_no_time:
                raise RuntimeError("primary pipeline lock already exists with different content")
            selection = existing
    else:
        atomic_json(locked, selection)
        (DEVELOPMENT_ROOT / "primary_pipeline_locked.sha256").write_text(
            sha256_file(locked) + "\n", encoding="utf-8"
        )
        try:
            os.chmod(locked, 0o444)
        except OSError:
            pass
    return selection


def verify_primary_pipeline_lock() -> dict[str, Any]:
    path = DEVELOPMENT_ROOT / "primary_pipeline_locked.json"
    expected = (
        DEVELOPMENT_ROOT / "primary_pipeline_locked.sha256"
    ).read_text(encoding="utf-8").strip()
    if sha256_file(path) != expected:
        raise RuntimeError("primary pipeline lock mismatch")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection["rescue_preregistration_hash"] != verify_rescue_preregistration():
        raise RuntimeError("primary pipeline references wrong rescue preregistration")
    return selection
