"""Exploratory single-round NPE ensemble for Route-3 cortical-rate summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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

from .route3_global_robustness import RESULTS_ROOT, rate_contract_hash, rate_feature_names
from .route3_synthetic_preflight import PARAMETER_NAMES


TRAINING_ROOT = RESULTS_ROOT / "exploratory_npe_ensemble"
MODEL_SEEDS = (1301, 1302, 1303)
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
ENSEMBLE_SEED = 20260801


@dataclass(frozen=True)
class TrainingData:
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


def load_training_data(bank_path: Path, split_path: Path) -> TrainingData:
    with np.load(bank_path, allow_pickle=False) as bank:
        theta = np.asarray(bank["theta"], dtype=np.float64)
        x = np.asarray(bank["x"], dtype=np.float64)
        success = np.asarray(bank["success"], dtype=bool)
        if list(bank["feature_names"]) != list(rate_feature_names()):
            raise RuntimeError("bank feature order differs from frozen 14D contract")
    with np.load(split_path, allow_pickle=False) as split:
        training_indices = np.asarray(split["training_indices"], dtype=np.int64)
        validation_indices = np.asarray(split["validation_indices"], dtype=np.int64)
        x_location = np.asarray(split["x_location"], dtype=np.float64)
        x_scale = np.asarray(split["x_scale"], dtype=np.float64)
        theta_lower = np.asarray(split["theta_lower"], dtype=np.float64)
        theta_upper = np.asarray(split["theta_upper"], dtype=np.float64)
        if list(split["feature_names"]) != list(rate_feature_names()):
            raise RuntimeError("split feature order differs from frozen 14D contract")
    if not success[training_indices].all() or not success[validation_indices].all():
        raise RuntimeError("fixed split includes failed simulations")
    theta_unit = (theta - theta_lower) / (theta_upper - theta_lower)
    x_scaled = (x - x_location) / x_scale
    arrays = (theta_unit, x_scaled)
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("training arrays contain non-finite values")
    return TrainingData(
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


def _builder():
    return posterior_nn(**ARCHITECTURE)


def _build_network(data: TrainingData, seed: int, device: torch.device):
    torch.manual_seed(int(seed))
    theta = torch.as_tensor(data.theta_train, dtype=torch.float32, device=device)
    x = torch.as_tensor(data.x_train, dtype=torch.float32, device=device)
    return _builder()(theta, x).to(device)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _epoch_loss(network, loader, device, optimizer=None) -> float:
    training = optimizer is not None
    network.train(training)
    total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for theta_batch, x_batch in loader:
            theta_batch = theta_batch.to(device)
            x_batch = x_batch.to(device)
            loss = network.loss(theta_batch, condition=x_batch).mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), TRAINING_POLICY["clip_max_norm"]
                )
                optimizer.step()
            batch_size = len(theta_batch)
            total += float(loss.detach().cpu()) * batch_size
            count += batch_size
    return total / count


def train_member(
    data: TrainingData,
    seed: int,
    output_dir: Path | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir or TRAINING_ROOT / f"member_{seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    network = _build_network(data, seed, device)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=TRAINING_POLICY["learning_rate"],
        weight_decay=TRAINING_POLICY["weight_decay"],
    )
    theta_train = torch.as_tensor(data.theta_train, dtype=torch.float32)
    x_train = torch.as_tensor(data.x_train, dtype=torch.float32)
    theta_validation = torch.as_tensor(data.theta_validation, dtype=torch.float32)
    x_validation = torch.as_tensor(data.x_validation, dtype=torch.float32)
    train_dataset = TensorDataset(theta_train, x_train)
    validation_dataset = TensorDataset(theta_validation, x_validation)
    generator = torch.Generator().manual_seed(int(seed) + 100_000)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_POLICY["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=TRAINING_POLICY["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    last_path = output_dir / "last_checkpoint.pt"
    best_path = output_dir / "best_checkpoint.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_validation = np.inf
    best_epoch = -1
    stale_epochs = 0
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if (
            checkpoint.get("contract_hash") == rate_contract_hash()
            and int(checkpoint.get("seed")) == int(seed)
        ):
            network.load_state_dict(checkpoint["network_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            history = list(checkpoint["history"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_validation = float(checkpoint["best_validation"])
            best_epoch = int(checkpoint["best_epoch"])
            stale_epochs = int(checkpoint["stale_epochs"])
    started = perf_counter()
    already_stopped = (
        stale_epochs >= TRAINING_POLICY["early_stopping_patience"]
        or start_epoch >= TRAINING_POLICY["max_epochs"]
    )
    epoch_range = (
        range(0)
        if already_stopped
        else range(start_epoch, TRAINING_POLICY["max_epochs"])
    )
    for epoch in epoch_range:
        train_loss = _epoch_loss(network, train_loader, device, optimizer)
        validation_loss = _epoch_loss(network, validation_loader, device)
        improved = (
            validation_loss
            < best_validation - TRAINING_POLICY["minimum_improvement"]
        )
        if improved:
            best_validation = validation_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "improved": improved,
            }
        )
        payload = {
            "seed": int(seed),
            "epoch": epoch,
            "network_state": network.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_validation": best_validation,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "architecture": ARCHITECTURE,
            "training_policy": TRAINING_POLICY,
            "contract_hash": rate_contract_hash(),
        }
        _atomic_torch_save(payload, last_path)
        if improved:
            _atomic_torch_save(payload, best_path)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        if stale_epochs >= TRAINING_POLICY["early_stopping_patience"]:
            break
    runtime = perf_counter() - started
    if not best_path.exists():
        raise RuntimeError(f"member {seed} produced no best checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    network.load_state_dict(best["network_state"])
    network.eval()
    summary = {
        "seed": int(seed),
        "device": str(device),
        "status": "trained",
        "epochs_completed": len(history),
        "best_epoch": int(best["best_epoch"]),
        "best_validation_loss": float(best["best_validation"]),
        "final_train_loss": float(history[-1]["train_loss"]),
        "final_validation_loss": float(history[-1]["validation_loss"]),
        "runtime_this_call_s": float(runtime),
        "checkpoint": best_path.as_posix(),
        "contract_hash": rate_contract_hash(),
    }
    _atomic_json(summary, output_dir / "training_summary.json")
    return summary


def train_ensemble(
    bank_path: Path,
    split_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir or TRAINING_ROOT)
    data = load_training_data(bank_path, split_path)
    started = perf_counter()
    summaries = [
        train_member(data, seed, output_dir / f"member_{seed}") for seed in MODEL_SEEDS
    ]
    ensemble_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "trained",
        "members": summaries,
        "member_seeds": list(MODEL_SEEDS),
        "weights": [1.0 / len(MODEL_SEEDS)] * len(MODEL_SEEDS),
        "mixture_rule": "equal-count sampling from each member; no learned weights",
        "architecture": ARCHITECTURE,
        "training_policy": TRAINING_POLICY,
        "training_rows": len(data.training_indices),
        "validation_rows": len(data.validation_indices),
        "runtime_s": float(perf_counter() - started),
        "scope": "exploratory Route-3 synthetic cortical-rate posterior",
        "real_eeg_inference": False,
        "package_versions": {
            name: metadata.version(name)
            for name in ("sbi", "torch", "numpy", "scipy")
        },
    }
    _atomic_json(ensemble_summary, output_dir / "ensemble_manifest.json")
    pd.DataFrame(summaries).to_csv(output_dir / "training_summary.csv", index=False)
    return {"data": data, "summary": ensemble_summary}


def load_member(
    data: TrainingData, seed: int, checkpoint_path: Path, device_name: str = "cpu"
):
    device = torch.device(device_name)
    network = _build_network(data, seed, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["contract_hash"] != rate_contract_hash():
        raise RuntimeError("checkpoint contract hash mismatch")
    network.load_state_dict(checkpoint["network_state"])
    network.eval()
    prior = BoxUniform(
        low=torch.zeros(8, device=device),
        high=torch.ones(8, device=device),
        device=device_name,
    )
    return DirectPosterior(
        posterior_estimator=network,
        prior=prior,
        device=device,
    )


def load_ensemble_members(
    data: TrainingData, output_dir: Path | None = None, device_name: str = "cpu"
) -> list[DirectPosterior]:
    output_dir = Path(output_dir or TRAINING_ROOT)
    return [
        load_member(
            data,
            seed,
            output_dir / f"member_{seed}" / "best_checkpoint.pt",
            device_name=device_name,
        )
        for seed in MODEL_SEEDS
    ]


def sample_member_normalized(
    posterior: DirectPosterior,
    x_scaled: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(int(seed))
    samples = posterior.sample(
        (int(n_samples),),
        x=torch.as_tensor(x_scaled, dtype=torch.float32),
        show_progress_bars=False,
        reject_outside_prior=True,
    )
    result = samples.detach().cpu().numpy()
    if result.shape != (n_samples, 8) or not np.isfinite(result).all():
        raise RuntimeError("posterior returned invalid samples")
    if ((result < 0.0) | (result > 1.0)).any():
        raise RuntimeError("posterior samples escaped normalized prior support")
    return result


def sample_equal_ensemble_normalized(
    posteriors: Sequence[DirectPosterior],
    x_scaled: np.ndarray,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(posteriors) != 3:
        raise ValueError("the frozen exploratory ensemble has exactly three members")
    counts = np.full(3, n_samples // 3, dtype=int)
    counts[: n_samples % 3] += 1
    samples = []
    labels = []
    for member_index, (posterior, count) in enumerate(zip(posteriors, counts)):
        samples.append(
            sample_member_normalized(
                posterior, x_scaled, int(count), seed + member_index * 100_003
            )
        )
        labels.append(np.full(count, member_index, dtype=np.int8))
    combined = np.vstack(samples)
    member_labels = np.concatenate(labels)
    rng = np.random.default_rng(seed + 900_001)
    order = rng.permutation(n_samples)
    return combined[order], member_labels[order]


def exploratory_member_checks(
    data: TrainingData,
    posteriors: Sequence[DirectPosterior],
    n_cases: int = 32,
    n_samples: int = 1024,
) -> pd.DataFrame:
    indices = np.linspace(0, len(data.x_validation) - 1, n_cases, dtype=int)
    rows = []
    for case_position, validation_position in enumerate(indices):
        x = data.x_validation[validation_position]
        true_theta = data.theta_validation[validation_position]
        for member_index, posterior in enumerate(posteriors):
            samples = sample_member_normalized(
                posterior,
                x,
                n_samples,
                ENSEMBLE_SEED + case_position * 1009 + member_index,
            )
            median = np.median(samples, axis=0)
            rows.append(
                {
                    "case": case_position,
                    "member": member_index,
                    "finite_rate": float(np.isfinite(samples).all(axis=1).mean()),
                    "within_prior_rate": float(
                        ((samples >= 0) & (samples <= 1)).all(axis=1).mean()
                    ),
                    "mean_normalized_median_error": float(
                        np.mean(np.abs(median - true_theta))
                    ),
                    "boundary_sample_fraction": float(
                        ((samples < 0.01) | (samples > 0.99)).any(axis=1).mean()
                    ),
                    "mean_posterior_std": float(np.mean(np.std(samples, axis=0))),
                }
            )
    return pd.DataFrame(rows)


def write_go_criteria(path: Path) -> dict[str, Any]:
    """Freeze operational GO criteria before held-out results are opened."""

    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("version") != "route3-heldout-go-criteria-v1":
            raise RuntimeError("existing GO criteria artifact has an incompatible version")
        return existing
    criteria = {
        "version": "route3-heldout-go-criteria-v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_after_heldout_open": True,
        "formal_go": {
            "training_failure_rate_max": 0.01,
            "heldout_failure_rate_max": 0.01,
            "posterior_finite_and_in_prior_rate_min": 0.999,
            "coverage_80_or_90_wilson_contains_nominal_parameters_min": 7,
            "severe_undercoverage_definition": (
                "observed coverage < nominal-0.15 and Wilson upper bound < nominal"
            ),
            "severe_undercoverage_parameters_max": 0,
            "median_error_better_than_prior_parameters_min": 6,
            "overall_median_error_improvement_min_fraction": 0.10,
            "meaningful_contraction_definition": "median 90% CI width < 0.90 prior width",
            "meaningful_contraction_with_reasonable_coverage_parameters_min": 6,
            "ppc_features_better_than_prior_min": 10,
            "overall_ppc_error_improvement_min_fraction": 0.10,
            "ensemble_member_median_disagreement_mean_max_prior_width": 0.10,
            "ensemble_member_disagreement_parameter_max_prior_width": 0.20,
            "global_collision_requirement": (
                "coverage/contraction must remain honest despite collisions"
            ),
        },
        "conditional_go": {
            "engineering_failure_and_finite_gates_must_pass": True,
            "severe_undercoverage_parameters_max": 1,
            "median_error_better_than_prior_parameters_min": 5,
            "meaningful_contraction_with_reasonable_coverage_parameters_min": 5,
            "overall_ppc_must_improve": True,
            "ensemble_disagreement_gate_must_pass": True,
        },
        "no_go_if": [
            "conditional_go criteria fail",
            "multiple parameters do not beat prior baseline",
            "severe undercoverage or false contraction",
            "global collisions or seed noise dominate recovery",
            "ensemble scientific conclusions are unstable",
            "synthetic PPC does not improve over prior predictive",
        ],
        "scope": "synthetic cortical-rate observable recovery only",
        "does_not_unlock_real_eeg_inference": True,
    }
    _atomic_json(criteria, path)
    return criteria


__all__ = [
    "ARCHITECTURE",
    "ENSEMBLE_SEED",
    "MODEL_SEEDS",
    "TRAINING_POLICY",
    "TRAINING_ROOT",
    "TrainingData",
    "exploratory_member_checks",
    "load_ensemble_members",
    "load_training_data",
    "sample_equal_ensemble_normalized",
    "sample_member_normalized",
    "train_ensemble",
    "train_member",
    "write_go_criteria",
]
