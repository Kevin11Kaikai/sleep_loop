"""Reconstruct a few audited cortical-rate traces for Figure-10 PPC display.

This module is reporting-only. It replays already-recorded posterior-predictive
parameter vectors and simulator seeds, then verifies that the frozen 14D
extractor reproduces the saved summary vectors. The traces are cortical
population firing rates in Hz, not scalp EEG.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .figure10_bank import run_custom_dataset
from .figure10_diagnostics import PPC_ROOT
from .figure10_protocol import (
    PARAMETER_NAMES_8D,
    RESULTS_ROOT,
    SEEDS,
    atomic_json,
    rate_feature_names,
    sha256_file,
    verify_preregistration,
)
from .route3_global_robustness import (
    EXPECTED_FS_HZ,
    MODEL_DURATION_S,
    WARM_UP_S,
    _extract_rate14,
    _load_model_module,
    _set_all_model_seeds,
    deterministic_seed_schedule,
)


TRACE_ROOT = PPC_ROOT / "trace_examples"


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def simulate_cortical_rate_trace(
    theta_full_8d: np.ndarray,
    simulator_seed: int,
) -> dict[str, Any]:
    """Return the frozen 30-s cortical-rate segment and its 14D summaries."""

    theta = np.asarray(theta_full_8d, float)
    if theta.shape != (8,) or not np.isfinite(theta).all():
        raise ValueError("theta_full_8d must be a finite 8-vector")
    module = _load_model_module("v8a_t13")
    parameters = dict(zip(PARAMETER_NAMES_8D, theta))
    model = module.build_model(
        parameters["mue"],
        parameters["mui"],
        parameters["b"],
        parameters["tauA"],
        parameters["g_LK"],
        parameters["g_h"],
        parameters["c_th2ctx"],
        parameters["c_ctx2th"],
        duration=MODEL_DURATION_S * 1000.0,
    )
    model.params["backend"] = "numba"
    _set_all_model_seeds(model, module, int(simulator_seed))
    model.run()
    fs_hz = 1000.0 / float(model.params["sampling_dt"])
    if not np.isclose(fs_hz, EXPECTED_FS_HZ):
        raise RuntimeError(f"unexpected sampling rate {fs_hz}")
    start = int(round(WARM_UP_S * fs_hz))
    stop = start + int(round(30.0 * fs_hz))
    r_exc_hz = np.asarray(model["r_mean_EXC"], float)[0, start:stop] * 1000.0
    r_inh_hz = np.asarray(model["r_mean_INH"], float)[0, start:stop] * 1000.0
    if len(r_exc_hz) != int(round(30.0 * fs_hz)):
        raise RuntimeError("trace replay did not produce one complete 30-s epoch")
    summaries = _extract_rate14(r_exc_hz, r_inh_hz, fs_hz)
    return {
        "time_s": np.arange(len(r_exc_hz), dtype=float) / fs_hz,
        "r_exc_hz": r_exc_hz,
        "r_inh_hz": r_inh_hz,
        "summaries": summaries,
        "fs_hz": fs_hz,
    }


def build_trace_examples(examples_per_track: int = 4) -> dict[str, Any]:
    """Replay the primary observation and selected PPC examples."""

    verify_preregistration()
    primary_path = (
        PPC_ROOT / "primary_observation" / "8d" / "primary_observation_8d.npz"
    )
    with np.load(primary_path, allow_pickle=False) as data:
        primary_theta = np.asarray(data["theta_full_8d"][0], float)
        primary_seed = int(data["simulator_seed"][0])
        primary_x = np.asarray(data["x"][0], float)
    primary = simulate_cortical_rate_trace(primary_theta, primary_seed)
    if not np.allclose(
        primary["summaries"], primary_x, rtol=2e-6, atol=2e-7
    ):
        raise RuntimeError("primary trace replay summary mismatch")
    primary_output = TRACE_ROOT / "primary_cortical_rate_trace.npz"
    _atomic_npz(
        primary_output,
        time_s=primary["time_s"].astype(np.float32),
        r_exc_hz=primary["r_exc_hz"].astype(np.float32),
        r_inh_hz=primary["r_inh_hz"].astype(np.float32),
        summaries=primary["summaries"].astype(np.float64),
        theta_full_8d=primary_theta.astype(np.float64),
        simulator_seed=np.asarray(primary_seed, np.int64),
        fs_hz=np.asarray(primary["fs_hz"], np.float64),
        feature_names=np.asarray(rate_feature_names(), dtype="<U64"),
        signal_semantics=np.asarray(
            "synthetic cortical population firing rate; not scalp EEG",
            dtype="<U96",
        ),
        preregistration_hash=np.asarray(
            verify_preregistration(), dtype="<U64"
        ),
    )
    outputs: dict[str, Any] = {
        "primary": {
            "path": primary_output.relative_to(RESULTS_ROOT).as_posix(),
            "sha256": sha256_file(primary_output),
        }
    }
    for track in ("8d", "7d"):
        ppc_path = (
            PPC_ROOT
            / track
            / "posterior_predictive"
            / f"posterior_predictive_{track}.npz"
        )
        with np.load(ppc_path, allow_pickle=False) as data:
            theta = np.asarray(
                data["theta_full_8d"][:examples_per_track], float
            )
            seeds = np.asarray(
                data["simulator_seed"][:examples_per_track], np.int64
            )
            expected_x = np.asarray(data["x"][:examples_per_track], float)
        traces = [
            simulate_cortical_rate_trace(theta[index], int(seeds[index]))
            for index in range(examples_per_track)
        ]
        summaries = np.vstack([entry["summaries"] for entry in traces])
        if not np.allclose(
            summaries, expected_x, rtol=2e-6, atol=2e-7
        ):
            raise RuntimeError(f"{track} PPC trace replay summary mismatch")
        output = TRACE_ROOT / f"{track}_posterior_predictive_traces.npz"
        _atomic_npz(
            output,
            time_s=traces[0]["time_s"].astype(np.float32),
            r_exc_hz=np.vstack(
                [entry["r_exc_hz"] for entry in traces]
            ).astype(np.float32),
            r_inh_hz=np.vstack(
                [entry["r_inh_hz"] for entry in traces]
            ).astype(np.float32),
            summaries=summaries.astype(np.float64),
            theta_full_8d=theta.astype(np.float64),
            simulator_seed=seeds,
            fs_hz=np.asarray(traces[0]["fs_hz"], np.float64),
            feature_names=np.asarray(rate_feature_names(), dtype="<U64"),
            signal_semantics=np.asarray(
                "synthetic cortical population firing rate; not scalp EEG",
                dtype="<U96",
            ),
            track=np.asarray(track, dtype="<U8"),
            preregistration_hash=np.asarray(
                verify_preregistration(), dtype="<U64"
            ),
        )
        outputs[track] = {
            "path": output.relative_to(RESULTS_ROOT).as_posix(),
            "sha256": sha256_file(output),
            "examples": examples_per_track,
            "summary_replay_allclose": True,
        }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "reporting-only deterministic replay of frozen PPC rows",
        "outputs": outputs,
        "preregistration_hash": verify_preregistration(),
    }
    atomic_json(TRACE_ROOT / "trace_example_manifest.json", manifest)
    return manifest


def build_ppc_seed_robustness(
    parameter_cases: int = 16,
    seeds_per_case: int = 3,
) -> dict[str, Any]:
    """Quantify nuisance-seed variation for selected posterior draws.

    This secondary check does not enter the preregistered verdict. Selection is
    deterministic (the first rows in each frozen PPC manifest), and the seed
    namespace is the preregistered secondary-stress namespace.
    """

    outputs: dict[str, Any] = {}
    for track_index, track in enumerate(("8d", "7d")):
        source = (
            PPC_ROOT
            / track
            / "posterior_predictive"
            / f"posterior_predictive_{track}.npz"
        )
        with np.load(source, allow_pickle=False) as data:
            theta = np.asarray(
                data["theta_full_8d"][:parameter_cases], float
            )
        repeated_theta = np.repeat(theta, seeds_per_case, axis=0)
        seed_base = int(SEEDS["stress_observation_simulator"]) + (
            track_index * 100_000
        )
        seeds = deterministic_seed_schedule(len(repeated_theta), seed_base)
        path = run_custom_dataset(
            repeated_theta,
            seeds,
            PPC_ROOT / track / "seed_robustness",
            track,
            "ppc_seed_robustness",
        )
        with np.load(path, allow_pickle=False) as data:
            x = np.asarray(data["x"], float).reshape(
                parameter_cases, seeds_per_case, -1
            )
            success = np.asarray(data["success"], bool)
        if not success.all() or not np.isfinite(x).all():
            raise RuntimeError(f"{track} seed-robustness simulations failed")
        within = np.mean(np.var(x, axis=1, ddof=1), axis=0)
        between = np.var(np.mean(x, axis=1), axis=0, ddof=1)
        signal_to_seed = between / np.maximum(within, 1e-15)
        frame = pd.DataFrame(
            {
                "feature": rate_feature_names(),
                "mean_within_theta_seed_variance": within,
                "between_theta_variance": between,
                "between_to_seed_variance_ratio": signal_to_seed,
            }
        )
        csv_path = PPC_ROOT / track / "ppc_seed_robustness.csv"
        frame.to_csv(csv_path, index=False)
        summary = {
            "track": track,
            "parameter_cases": parameter_cases,
            "seeds_per_case": seeds_per_case,
            "attempted": len(repeated_theta),
            "failed": int((~success).sum()),
            "median_between_to_seed_variance_ratio": float(
                np.median(signal_to_seed)
            ),
            "minimum_between_to_seed_variance_ratio": float(
                np.min(signal_to_seed)
            ),
            "selection": "first frozen posterior-predictive parameter rows",
            "scope": "secondary robustness; excluded from operational verdict",
            "csv": csv_path.relative_to(RESULTS_ROOT).as_posix(),
            "preregistration_hash": verify_preregistration(),
        }
        atomic_json(PPC_ROOT / track / "ppc_seed_robustness.json", summary)
        outputs[track] = summary
    atomic_json(
        PPC_ROOT / "ppc_seed_robustness_manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "tracks": outputs,
            "preregistration_hash": verify_preregistration(),
        },
    )
    return outputs


__all__ = [
    "TRACE_ROOT",
    "build_ppc_seed_robustness",
    "build_trace_examples",
    "simulate_cortical_rate_trace",
]
