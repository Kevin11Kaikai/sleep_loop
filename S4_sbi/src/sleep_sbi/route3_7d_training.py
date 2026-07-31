"""Three-member single-round NPE ensemble for the preregistered 7D experiment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

from .route3_7d_experiment import (
    BANK_ROOT,
    PARAMETER_NAMES_7D,
    TRAINING_ROOT,
    prior_bounds_7d,
    rate_feature_names,
    read_preregistration,
    verify_preregistration,
)


ARCHITECTURE = {
    "model": "maf",
    "hidden_features": 64,
    "num_transforms": 5,
    "z_score_theta": "none",
    "z_score_x": "none",
}
TRAINING_POLICY = {
    "batch_size": 128,
    "learning_rate": 5e-4,
    "weight_decay": 1e-6,
    "max_epochs": 300,
    "early_stopping_patience": 25,
    "minimum_improvement": 1e-4,
    "clip_max_norm": 5.0,
}


@dataclass(frozen=True)
class TrainingData7D:
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


def model_seeds() -> tuple[int, int, int]:
    values = tuple(int(v) for v in read_preregistration()["seeds"]["network_initialization"])
    if len(values) != 3 or len(set(values)) != 3:
        raise RuntimeError("preregistered ensemble must have three distinct seeds")
    return values


def load_training_data_7d(
    bank_path: Path | None = None, split_path: Path | None = None
) -> TrainingData7D:
    verify_preregistration()
    bank_path = Path(bank_path or BANK_ROOT / "route3_7d_cortex_rate_14d_bank_4096.npz")
    split_path = Path(split_path or BANK_ROOT / "split_and_scaling.npz")
    with np.load(bank_path, allow_pickle=False) as bank:
        theta = np.asarray(bank["theta"], float)
        x = np.asarray(bank["x"], float)
        success = np.asarray(bank["success"], bool)
        if theta.shape != (4096, 7) or x.shape != (4096, 14):
            raise RuntimeError(f"unexpected 7D bank shapes {theta.shape}, {x.shape}")
        if list(bank["parameter_names"]) != list(PARAMETER_NAMES_7D):
            raise RuntimeError("7D bank parameter order drift")
        if list(bank["feature_names"]) != list(rate_feature_names()):
            raise RuntimeError("7D bank feature order drift")
        if str(bank["preregistration_hash"].item()) != verify_preregistration():
            raise RuntimeError("7D bank preregistration hash mismatch")
    with np.load(split_path, allow_pickle=False) as split:
        training_indices = np.asarray(split["training_indices"], np.int64)
        validation_indices = np.asarray(split["validation_indices"], np.int64)
        x_location = np.asarray(split["x_location"], float)
        x_scale = np.asarray(split["x_scale"], float)
        theta_lower = np.asarray(split["theta_lower"], float)
        theta_upper = np.asarray(split["theta_upper"], float)
    if np.intersect1d(training_indices, validation_indices).size:
        raise RuntimeError("train/validation index leakage")
    if not success[training_indices].all() or not success[validation_indices].all():
        raise RuntimeError("split contains failed simulations")
    theta_unit = (theta - theta_lower) / (theta_upper - theta_lower)
    x_scaled = (x - x_location) / x_scale
    if not np.isfinite(theta_unit).all() or not np.isfinite(x_scaled).all():
        raise RuntimeError("training arrays contain nonfinite values")
    return TrainingData7D(
        theta_train=theta_unit[training_indices],
        x_train=x_scaled[training_indices],
        theta_validation=theta_unit[validation_indices],
        x_validation=x_scaled[validation_indices],
        theta_lower=theta_lower,
        theta_upper=theta_upper,
        x_location=x_location,
        x_scale=x_scale,
        training_indices=training_indices,
        validation_indices=validation_indices,
    )


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _network(data: TrainingData7D, seed: int, device: torch.device):
    torch.manual_seed(int(seed))
    builder = posterior_nn(**ARCHITECTURE)
    theta = torch.as_tensor(data.theta_train, dtype=torch.float32, device=device)
    x = torch.as_tensor(data.x_train, dtype=torch.float32, device=device)
    return builder(theta, x).to(device)


def _loss(network, loader, device, optimizer=None) -> float:
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
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), TRAINING_POLICY["clip_max_norm"]
                )
                optimizer.step()
            total += float(loss.detach().cpu()) * len(theta)
            count += len(theta)
    return total / count


def train_member_7d(
    data: TrainingData7D, seed: int, output_dir: Path, device_name: str | None = None
) -> dict[str, Any]:
    verify_preregistration()
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    network = _network(data, seed, device)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=TRAINING_POLICY["learning_rate"],
        weight_decay=TRAINING_POLICY["weight_decay"],
    )
    generator = torch.Generator().manual_seed(int(seed) + 100_000)
    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(data.theta_train, dtype=torch.float32),
            torch.as_tensor(data.x_train, dtype=torch.float32),
        ),
        batch_size=TRAINING_POLICY["batch_size"], shuffle=True,
        generator=generator, num_workers=0,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(data.theta_validation, dtype=torch.float32),
            torch.as_tensor(data.x_validation, dtype=torch.float32),
        ),
        batch_size=TRAINING_POLICY["batch_size"], shuffle=False, num_workers=0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last_checkpoint.pt"
    best_path = output_dir / "best_checkpoint.pt"
    history: list[dict[str, Any]] = []
    start_epoch, best_epoch, stale = 0, -1, 0
    best_validation = np.inf
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if (
            checkpoint.get("preregistration_hash") == verify_preregistration()
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
        stale >= TRAINING_POLICY["early_stopping_patience"]
        or start_epoch >= TRAINING_POLICY["max_epochs"]
    )
    epochs = range(0) if already_stopped else range(start_epoch, TRAINING_POLICY["max_epochs"])
    for epoch in epochs:
        train_loss = _loss(network, train_loader, device, optimizer)
        validation_loss = _loss(network, validation_loader, device)
        improved = validation_loss < best_validation - TRAINING_POLICY["minimum_improvement"]
        if improved:
            best_validation, best_epoch, stale = validation_loss, epoch, 0
        else:
            stale += 1
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "validation_loss": validation_loss, "improved": bool(improved),
        })
        payload = {
            "seed": int(seed), "epoch": epoch,
            "network_state": network.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history, "best_validation": best_validation,
            "best_epoch": best_epoch, "stale_epochs": stale,
            "architecture": ARCHITECTURE, "training_policy": TRAINING_POLICY,
            "preregistration_hash": verify_preregistration(),
            "theta_dimension": 7, "x_dimension": 14,
        }
        _atomic_torch(last_path, payload)
        if improved:
            _atomic_torch(best_path, payload)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        if stale >= TRAINING_POLICY["early_stopping_patience"]:
            break
    runtime = perf_counter() - started
    if not best_path.exists():
        raise RuntimeError(f"7D member {seed} produced no best checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    summary = {
        "seed": int(seed), "device": str(device), "status": "trained",
        "epochs_completed": len(best["history"]),
        "best_epoch": int(best["best_epoch"]),
        "best_validation_loss": float(best["best_validation"]),
        "final_train_loss": float(best["history"][-1]["train_loss"]),
        "final_validation_loss": float(best["history"][-1]["validation_loss"]),
        "runtime_this_call_s": float(runtime),
        "checkpoint": best_path.as_posix(),
        "preregistration_hash": verify_preregistration(),
    }
    _atomic_json(output_dir / "training_summary.json", summary)
    return summary


def train_ensemble_7d() -> dict[str, Any]:
    data = load_training_data_7d()
    manifest_path = TRAINING_ROOT / "ensemble_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoints_exist = all(
            (TRAINING_ROOT / f"member_{seed}" / "best_checkpoint.pt").exists()
            for seed in model_seeds()
        )
        if (
            existing.get("status") == "trained"
            and existing.get("preregistration_hash") == verify_preregistration()
            and existing.get("member_seeds") == list(model_seeds())
            and checkpoints_exist
        ):
            return {
                "data": data,
                "summary": existing,
                "resumed_from_completed_manifest": True,
            }
    started = perf_counter()
    summaries = [
        train_member_7d(data, seed, TRAINING_ROOT / f"member_{seed}")
        for seed in model_seeds()
    ]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "trained", "members": summaries,
        "member_seeds": list(model_seeds()),
        "weights": [1 / 3] * 3,
        "mixture_rule": "equal-count sampling; no learned/result-driven weights",
        "architecture": ARCHITECTURE, "training_policy": TRAINING_POLICY,
        "training_rows": len(data.training_indices),
        "validation_rows": len(data.validation_indices),
        "runtime_this_call_s": float(perf_counter() - started),
        "scope": "preregistered Route-3 7D synthetic cortical-rate posterior",
        "real_eeg_inference": False,
        "preregistration_hash": verify_preregistration(),
        "package_versions": {
            name: metadata.version(name) for name in ("sbi", "torch", "numpy", "scipy")
        },
    }
    _atomic_json(TRAINING_ROOT / "ensemble_manifest.json", summary)
    pd.DataFrame(summaries).to_csv(TRAINING_ROOT / "training_summary.csv", index=False)
    return {"data": data, "summary": summary}


def load_member_7d(data: TrainingData7D, seed: int, device_name: str = "cpu") -> DirectPosterior:
    device = torch.device(device_name)
    network = _network(data, seed, device)
    checkpoint = torch.load(
        TRAINING_ROOT / f"member_{seed}" / "best_checkpoint.pt",
        map_location=device, weights_only=False,
    )
    if checkpoint["preregistration_hash"] != verify_preregistration():
        raise RuntimeError("7D checkpoint preregistration hash mismatch")
    network.load_state_dict(checkpoint["network_state"])
    network.eval()
    prior = BoxUniform(
        low=torch.zeros(7, device=device),
        high=torch.ones(7, device=device),
        device=device_name,
    )
    return DirectPosterior(posterior_estimator=network, prior=prior, device=device)


def load_ensemble_7d(
    data: TrainingData7D, device_name: str = "cpu"
) -> list[DirectPosterior]:
    return [load_member_7d(data, seed, device_name) for seed in model_seeds()]


def sample_member_7d(
    posterior: DirectPosterior, x_scaled: np.ndarray, n: int, seed: int
) -> np.ndarray:
    torch.manual_seed(int(seed))
    values = posterior.sample(
        (int(n),), x=torch.as_tensor(x_scaled, dtype=torch.float32),
        show_progress_bars=False, reject_outside_prior=True,
    ).detach().cpu().numpy()
    if values.shape != (n, 7) or not np.isfinite(values).all():
        raise RuntimeError("7D posterior returned invalid samples")
    if ((values < 0) | (values > 1)).any():
        raise RuntimeError("7D posterior sample escaped normalized prior")
    return values


def sample_equal_ensemble_7d(
    members: Sequence[DirectPosterior], x_scaled: np.ndarray, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.full(3, n // 3, dtype=int)
    counts[: n % 3] += 1
    values, labels = [], []
    for index, (member, count) in enumerate(zip(members, counts)):
        values.append(sample_member_7d(member, x_scaled, int(count), seed + index * 100_003))
        labels.append(np.full(count, index, dtype=np.int8))
    values = np.vstack(values)
    labels = np.concatenate(labels)
    order = np.random.default_rng(seed + 900_001).permutation(n)
    return values[order], labels[order]


def validation_member_checks(n_cases: int = 32, n_samples: int = 1024) -> pd.DataFrame:
    data = load_training_data_7d()
    members = load_ensemble_7d(data)
    base_seed = int(read_preregistration()["seeds"]["ensemble_validation_sampling"])
    indices = np.linspace(0, len(data.x_validation) - 1, n_cases, dtype=int)
    rows = []
    for case, position in enumerate(indices):
        for member_index, member in enumerate(members):
            samples = sample_member_7d(
                member, data.x_validation[position], n_samples,
                base_seed + case * 1009 + member_index,
            )
            median = np.median(samples, axis=0)
            rows.append({
                "case": case, "member": member_index,
                "finite_rate": float(np.isfinite(samples).all(axis=1).mean()),
                "within_prior_rate": float(((samples >= 0) & (samples <= 1)).all(axis=1).mean()),
                "mean_normalized_median_error": float(np.mean(np.abs(median - data.theta_validation[position]))),
                "boundary_sample_fraction": float(((samples < .01) | (samples > .99)).any(axis=1).mean()),
                "mean_posterior_std": float(np.mean(np.std(samples, axis=0))),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(TRAINING_ROOT / "validation_member_checks.csv", index=False)
    return frame
