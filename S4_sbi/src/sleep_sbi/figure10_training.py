"""NSF training and equal-mixture posterior utilities for Figure-10 tracks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import math
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

from .figure10_bank import BANK_ROOT
from .figure10_protocol import (
    FINAL_SCALE,
    INTERMEDIATE_SCALE,
    NSF_ARCHITECTURE,
    PARAMETER_NAMES_7D,
    PARAMETER_NAMES_8D,
    RESULTS_ROOT,
    SEEDS,
    TRAINING_POLICY,
    atomic_json,
    prior_bounds_7d,
    prior_bounds_8d,
    rate_feature_names,
    verify_preregistration,
)


TRAINING_ROOT = RESULTS_ROOT / "training"


@dataclass(frozen=True)
class TrainingData:
    track: str
    scale: int
    theta_train: np.ndarray
    x_train: np.ndarray
    theta_validation: np.ndarray
    x_validation: np.ndarray
    theta_lower: np.ndarray
    theta_upper: np.ndarray
    x_location: np.ndarray
    x_scale: np.ndarray
    train_ids: np.ndarray
    validation_ids: np.ndarray

    @property
    def theta_dimension(self) -> int:
        return self.theta_train.shape[1]


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def member_seeds(track: str, final: bool = True) -> tuple[int, ...]:
    if track not in {"8d", "7d"}:
        raise ValueError("track must be 8d or 7d")
    values = tuple(int(v) for v in SEEDS[f"network_{track}"])
    if len(values) != 5 or len(set(values)) != 5:
        raise RuntimeError("five distinct NSF member seeds are required")
    return values if final else values[:1]


def _bank_path(track: str) -> Path:
    return BANK_ROOT / track / f"figure10_{track}_bank_{FINAL_SCALE}.npz"


def load_training_data(track: str, scale: int = FINAL_SCALE) -> TrainingData:
    verify_preregistration()
    if track not in {"8d", "7d"}:
        raise ValueError("track must be 8d or 7d")
    if scale not in {INTERMEDIATE_SCALE, FINAL_SCALE}:
        raise ValueError("scale is not preregistered")
    with np.load(_bank_path(track), allow_pickle=False) as bank:
        theta = np.asarray(bank["theta"], float)[:scale]
        x = np.asarray(bank["x"], float)[:scale]
        success = np.asarray(bank["success"], bool)[:scale]
        feature_names = [str(value) for value in bank["feature_names"]]
        parameter_names = [str(value) for value in bank["parameter_names"]]
        preregistration_hash = str(bank["preregistration_hash"].item())
    expected_names = (
        list(PARAMETER_NAMES_8D) if track == "8d" else list(PARAMETER_NAMES_7D)
    )
    if parameter_names != expected_names:
        raise RuntimeError(f"{track} parameter order drift")
    if feature_names != list(rate_feature_names()):
        raise RuntimeError(f"{track} feature order drift")
    if preregistration_hash != verify_preregistration():
        raise RuntimeError(f"{track} bank preregistration mismatch")
    if not success.all():
        raise RuntimeError(
            f"{track} training loader requires all {scale} rows valid; "
            "failure audit remains in bank manifest"
        )
    if scale == FINAL_SCALE:
        with np.load(BANK_ROOT / "paired_split_and_scaling.npz", allow_pickle=False) as split:
            train_ids = np.asarray(split["train_paired_row_ids"], np.int64)
            validation_ids = np.asarray(
                split["validation_paired_row_ids"], np.int64
            )
        if train_ids.max(initial=-1) >= scale or validation_ids.max(initial=-1) >= scale:
            raise RuntimeError("final split index outside bank")
    else:
        rng = np.random.default_rng(int(SEEDS["train_validation_split"]) + scale)
        order = rng.permutation(scale)
        split_at = int(math.floor(0.8 * scale))
        train_ids = np.sort(order[:split_at])
        validation_ids = np.sort(order[split_at:])
    if np.intersect1d(train_ids, validation_ids).size:
        raise RuntimeError("train/validation leakage")
    if len(train_ids) + len(validation_ids) != scale:
        raise RuntimeError("split does not cover requested scale")
    bounds = prior_bounds_8d() if track == "8d" else prior_bounds_7d()
    theta_unit = (theta - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    x_location = np.median(x[train_ids], axis=0)
    q25, q75 = np.percentile(x[train_ids], [25, 75], axis=0)
    x_scale = q75 - q25
    fallback = np.std(x[train_ids], axis=0)
    x_scale = np.where(x_scale > 1e-12, x_scale, fallback)
    x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
    x_scaled = (x - x_location) / x_scale
    if not np.isfinite(theta_unit).all() or not np.isfinite(x_scaled).all():
        raise RuntimeError("training transforms contain nonfinite values")
    if ((theta_unit < -1e-10) | (theta_unit > 1 + 1e-10)).any():
        raise RuntimeError("normalized theta escaped unit prior")
    return TrainingData(
        track=track,
        scale=scale,
        theta_train=theta_unit[train_ids],
        x_train=x_scaled[train_ids],
        theta_validation=theta_unit[validation_ids],
        x_validation=x_scaled[validation_ids],
        theta_lower=bounds[:, 0],
        theta_upper=bounds[:, 1],
        x_location=x_location,
        x_scale=x_scale,
        train_ids=train_ids,
        validation_ids=validation_ids,
    )


def _network(data: TrainingData, seed: int, device: torch.device):
    torch.manual_seed(int(seed))
    builder = posterior_nn(**NSF_ARCHITECTURE)
    theta = torch.as_tensor(data.theta_train, dtype=torch.float32, device=device)
    x = torch.as_tensor(data.x_train, dtype=torch.float32, device=device)
    return builder(theta, x).to(device)


def _epoch_loss(
    network: Any,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    network.train(training)
    total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for theta, x in loader:
            theta = theta.to(device)
            x = x.to(device)
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
    if count == 0:
        raise RuntimeError("empty training or validation loader")
    return total / count


def train_member(
    track: str,
    scale: int,
    seed: int,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Train/resume one independently initialized raw NSF member."""

    preregistration_hash = verify_preregistration()
    data = load_training_data(track, scale)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    network = _network(data, seed, device)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=float(TRAINING_POLICY["learning_rate"]),
        weight_decay=float(TRAINING_POLICY["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(int(seed) + 100_003)
    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(data.theta_train, dtype=torch.float32),
            torch.as_tensor(data.x_train, dtype=torch.float32),
        ),
        batch_size=int(TRAINING_POLICY["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(data.theta_validation, dtype=torch.float32),
            torch.as_tensor(data.x_validation, dtype=torch.float32),
        ),
        batch_size=int(TRAINING_POLICY["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    output = TRAINING_ROOT / track / f"scale_{scale}" / f"member_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    last_path = output / "last_checkpoint.pt"
    best_path = output / "best_checkpoint.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_epoch = -1
    best_validation = np.inf
    stale = 0
    if last_path.exists():
        checkpoint = torch.load(
            last_path, map_location=device, weights_only=False
        )
        if (
            checkpoint.get("preregistration_hash") == preregistration_hash
            and checkpoint.get("track") == track
            and int(checkpoint.get("scale")) == scale
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
    already_complete = (
        stale >= int(TRAINING_POLICY["early_stopping_patience"])
        or start_epoch >= int(TRAINING_POLICY["max_epochs"])
    )
    epochs = (
        range(0)
        if already_complete
        else range(start_epoch, int(TRAINING_POLICY["max_epochs"]))
    )
    for epoch in epochs:
        train_loss = _epoch_loss(network, train_loader, device, optimizer)
        validation_loss = _epoch_loss(network, validation_loader, device)
        improved = (
            validation_loss
            < best_validation - float(TRAINING_POLICY["minimum_improvement"])
        )
        if improved:
            best_validation = validation_loss
            best_epoch = epoch
            stale = 0
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
            "track": track,
            "scale": scale,
            "seed": int(seed),
            "epoch": epoch,
            "network_state": network.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_validation": best_validation,
            "best_epoch": best_epoch,
            "stale_epochs": stale,
            "architecture": NSF_ARCHITECTURE,
            "training_policy": TRAINING_POLICY,
            "preregistration_hash": preregistration_hash,
            "theta_dimension": data.theta_dimension,
            "x_dimension": 14,
            "train_ids": data.train_ids,
            "validation_ids": data.validation_ids,
            "x_location": data.x_location,
            "x_scale": data.x_scale,
            "theta_lower": data.theta_lower,
            "theta_upper": data.theta_upper,
        }
        _atomic_torch(last_path, payload)
        if improved:
            _atomic_torch(best_path, payload)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(
            f"{track} scale={scale} member={seed} epoch={epoch} "
            f"train={train_loss:.5f} val={validation_loss:.5f} stale={stale}",
            flush=True,
        )
        if stale >= int(TRAINING_POLICY["early_stopping_patience"]):
            break
    if not best_path.exists():
        raise RuntimeError(f"{track} member {seed} produced no best checkpoint")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": track,
        "scale": scale,
        "seed": int(seed),
        "device": str(device),
        "status": "trained",
        "epochs_completed": len(best["history"]),
        "best_epoch": int(best["best_epoch"]),
        "best_validation_loss": float(best["best_validation"]),
        "last_train_loss": float(best["history"][-1]["train_loss"]),
        "last_validation_loss": float(best["history"][-1]["validation_loss"]),
        "runtime_this_call_s": float(perf_counter() - started),
        "checkpoint": best_path.relative_to(RESULTS_ROOT).as_posix(),
        "preregistration_hash": preregistration_hash,
    }
    atomic_json(output / "training_summary.json", summary)
    return summary


def train_ensemble(track: str, scale: int = FINAL_SCALE) -> dict[str, Any]:
    verify_preregistration()
    final = scale == FINAL_SCALE
    seeds = member_seeds(track, final=final)
    root = TRAINING_ROOT / track / f"scale_{scale}"
    manifest_path = root / "ensemble_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("preregistration_hash") == verify_preregistration()
            and manifest.get("member_seeds") == list(seeds)
            and all(
                (root / f"member_{seed}" / "best_checkpoint.pt").exists()
                for seed in seeds
            )
        ):
            return manifest
    started = perf_counter()
    members = [train_member(track, scale, seed) for seed in seeds]
    data = load_training_data(track, scale)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": track,
        "scale": scale,
        "status": "trained",
        "architecture": NSF_ARCHITECTURE,
        "training_policy": TRAINING_POLICY,
        "member_seeds": list(seeds),
        "weights": [1 / len(seeds)] * len(seeds),
        "mixture_rule": (
            "equal component probability; samples are pooled, never averaged; "
            "log_prob is logsumexp(member log_prob)-log(K)"
        ),
        "training_rows": len(data.train_ids),
        "validation_rows": len(data.validation_ids),
        "members": members,
        "runtime_this_call_s": float(perf_counter() - started),
        "preregistration_hash": verify_preregistration(),
        "packages": {
            name: metadata.version(name)
            for name in ("sbi", "torch", "numpy", "scipy")
        },
    }
    atomic_json(manifest_path, manifest)
    pd.DataFrame(members).to_csv(root / "training_summary.csv", index=False)
    return manifest


def _load_network(
    data: TrainingData, seed: int, device_name: str = "cpu"
) -> DirectPosterior:
    device = torch.device(device_name)
    network = _network(data, seed, device)
    checkpoint = torch.load(
        TRAINING_ROOT
        / data.track
        / f"scale_{data.scale}"
        / f"member_{seed}"
        / "best_checkpoint.pt",
        map_location=device,
        weights_only=False,
    )
    if checkpoint["preregistration_hash"] != verify_preregistration():
        raise RuntimeError("NSF checkpoint preregistration mismatch")
    network.load_state_dict(checkpoint["network_state"])
    network.eval()
    prior = BoxUniform(
        low=torch.zeros(data.theta_dimension, device=device),
        high=torch.ones(data.theta_dimension, device=device),
        device=device_name,
    )
    return DirectPosterior(
        posterior_estimator=network, prior=prior, device=device
    )


def load_members(
    track: str, scale: int = FINAL_SCALE, device_name: str = "cpu"
) -> tuple[TrainingData, list[DirectPosterior]]:
    data = load_training_data(track, scale)
    members = [
        _load_network(data, seed, device_name)
        for seed in member_seeds(track, final=scale == FINAL_SCALE)
    ]
    return data, members


def scale_observation(data: TrainingData, x: np.ndarray) -> np.ndarray:
    values = (np.asarray(x, float) - data.x_location) / data.x_scale
    if values.shape[-1] != 14 or not np.isfinite(values).all():
        raise RuntimeError("scaled observation is invalid")
    return values


def physical_to_unit(data: TrainingData, theta: np.ndarray) -> np.ndarray:
    values = (
        np.asarray(theta, float) - data.theta_lower
    ) / (data.theta_upper - data.theta_lower)
    if values.shape[-1] != data.theta_dimension:
        raise ValueError("physical theta dimension mismatch")
    return values


def unit_to_physical(data: TrainingData, theta: np.ndarray) -> np.ndarray:
    values = data.theta_lower + np.asarray(theta, float) * (
        data.theta_upper - data.theta_lower
    )
    if values.shape[-1] != data.theta_dimension:
        raise ValueError("unit theta dimension mismatch")
    return values


def sample_member(
    posterior: DirectPosterior,
    x_scaled: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(int(seed))
    # reject_outside_prior=False: leaking NPEs otherwise hang forever in
    # rejection sampling (0% accept). Clip back to the unit box afterwards.
    values = (
        posterior.sample(
            (int(n),),
            x=torch.as_tensor(x_scaled, dtype=torch.float32),
            show_progress_bars=False,
            reject_outside_prior=False,
        )
        .detach()
        .cpu()
        .numpy()
    )
    if values.shape[0] != n or not np.isfinite(values).all():
        raise RuntimeError("posterior returned invalid samples")
    return np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)


class EqualMixturePosterior:
    """Transparent equal-weight mixture of independently trained posteriors."""

    def __init__(self, members: Sequence[DirectPosterior]):
        if len(members) < 1:
            raise ValueError("mixture needs at least one member")
        self.members = list(members)

    def sample(
        self, x_scaled: np.ndarray, n: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(int(seed))
        labels = rng.integers(0, len(self.members), size=int(n), dtype=np.int16)
        samples = np.empty((n, self.dimension), dtype=np.float32)
        for index, member in enumerate(self.members):
            positions = np.flatnonzero(labels == index)
            if not len(positions):
                continue
            values = sample_member(
                member,
                x_scaled,
                len(positions),
                int(seed) + (index + 1) * 104729,
            )
            samples[positions] = values
        if not np.isfinite(samples).all():
            raise RuntimeError("mixture samples contain nonfinite values")
        return samples, labels

    @property
    def dimension(self) -> int:
        prior = self.members[0].prior
        if not prior.event_shape:
            raise RuntimeError("posterior prior has no event dimension")
        return int(prior.event_shape[0])

    def log_prob(
        self, theta_unit: np.ndarray, x_scaled: np.ndarray
    ) -> np.ndarray:
        theta_tensor = torch.as_tensor(theta_unit, dtype=torch.float32)
        x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32)
        logs = []
        with torch.no_grad():
            for member in self.members:
                # Skip leakage_correction; see _member_log_prob note.
                value = member.log_prob(
                    theta_tensor, x=x_tensor, norm_posterior=False
                )
                logs.append(value)
            stacked = torch.stack(logs, dim=0)
            mixed = torch.logsumexp(stacked, dim=0) - math.log(len(logs))
        values = mixed.detach().cpu().numpy()
        if not np.isfinite(values).all():
            raise RuntimeError("mixture log_prob is nonfinite")
        return values


__all__ = [
    "EqualMixturePosterior",
    "TRAINING_ROOT",
    "TrainingData",
    "load_members",
    "load_training_data",
    "member_seeds",
    "physical_to_unit",
    "sample_member",
    "scale_observation",
    "train_ensemble",
    "train_member",
    "unit_to_physical",
]
