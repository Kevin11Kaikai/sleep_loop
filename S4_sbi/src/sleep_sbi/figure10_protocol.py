"""Frozen protocol for the matched 8D/7D Figure-10-equivalent experiment.

This module defines contracts and hashes only.  It deliberately does not run
simulations, train networks, or inspect final diagnostic outcomes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .route3_7d_experiment import fixed_c_ctx2th
from .route3_global_robustness import rate_contract_hash, rate_feature_names
from .route3_synthetic_preflight import PARAMETER_NAMES, parameter_contract


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "S4_sbi" / "results" / "figure10_8d_7d"
CONFIG_PATH = (
    PROJECT_ROOT / "S4_sbi" / "configs" / "figure10_8d_7d_preregistered_v1.json"
)
LOCKED_PATH = (
    PROJECT_ROOT / "S4_sbi" / "artifacts" / "figure10_8d_7d_preregistered_v1.locked.json"
)
HASH_PATH = (
    PROJECT_ROOT / "S4_sbi" / "artifacts" / "figure10_8d_7d_preregistered_v1.sha256"
)
HUMAN_PATH = RESULTS_ROOT / "preregistration" / "FIGURE10_8D_7D_PREREGISTRATION.md"
HUMAN_HASH_PATH = HUMAN_PATH.with_suffix(".md.sha256")
DEVIATION_PATH = RESULTS_ROOT / "preregistration" / "deviation_log.json"
BENCHMARK_PATH = RESULTS_ROOT / "resource_benchmark" / "resource_benchmark.json"
OFFICIAL_ROOT = (
    RESULTS_ROOT / "official_reference" / "sbi-practical-guide"
)

PARAMETER_NAMES_8D = tuple(PARAMETER_NAMES)
PARAMETER_NAMES_7D = tuple(PARAMETER_NAMES[:-1])
PARAMETER_UNITS_8D = (
    "mV/ms",
    "mV/ms",
    "pA",
    "ms",
    "mS/cm^2",
    "mS/cm^2",
    "dimensionless coupling",
    "dimensionless coupling",
)
SCHEMA_VERSION = "route3-cortex-rate-14d-v1"
FINAL_SCALE = 32_768
INTERMEDIATE_SCALE = 8_192
SHARD_SIZE = 64
MAX_SIMULATION_WORKERS = 8

OFFICIAL_COMMIT = "18852e1311880d5d053656984cf3fa7d7096a7f2"
OFFICIAL_SBI_VERSION = "0.25.0"
OFFICIAL_NOTEBOOKS = (
    "examples/4_3_pyloric/4_3_pyloric.ipynb",
    "paper/fig10_pyloric/notebooks/01_analysis_and_diagnostics.ipynb",
    "paper/fig10_pyloric/notebooks/02_assemble_figure.ipynb",
)

NSF_ARCHITECTURE = {
    "model": "nsf",
    "num_transforms": 10,
    "hidden_features": 100,
    "z_score_theta": "none",
    "z_score_x": "none",
}
TRAINING_POLICY = {
    "batch_size": 4096,
    "learning_rate": 5e-4,
    "weight_decay": 1e-6,
    "max_epochs": 400,
    "early_stopping_patience": 30,
    "minimum_improvement": 1e-4,
    "clip_max_norm": 5.0,
    "validation_fraction": 0.2,
}

SEEDS = {
    "training_sobol": 9530001,
    "training_simulator_base": 9531001,
    "train_validation_split": 9532001,
    "network_8d": [9550001, 9550002, 9550003, 9550004, 9550005],
    "network_7d": [9551001, 9551002, 9551003, 9551004, 9551005],
    "official_coverage_sobol": 9560001,
    "official_coverage_simulator_base": 9561001,
    "official_posterior_sampling_8d": 9562001,
    "official_posterior_sampling_7d": 9563001,
    "powered_coverage_sobol": 9570001,
    "powered_coverage_simulator_base": 9571001,
    "powered_posterior_sampling_8d": 9572001,
    "powered_posterior_sampling_7d": 9573001,
    "primary_observation_simulator": 9580001,
    "primary_posterior_sampling_8d": 9581001,
    "primary_posterior_sampling_7d": 9582001,
    "ppc_simulator_base_8d": 9583001,
    "ppc_simulator_base_7d": 9584001,
    "prior_ppc_sobol": 9585001,
    "prior_ppc_simulator_base": 9586001,
    "lc2st_sobol": 9590001,
    "lc2st_simulator_base": 9591001,
    "lc2st_posterior_8d": 9592001,
    "lc2st_posterior_7d": 9593001,
    "lc2st_classifier_8d": 9594001,
    "lc2st_classifier_7d": 9595001,
    "stress_observation_simulator": 9596001,
}


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    data = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(data).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def prior_bounds_8d() -> np.ndarray:
    contract = parameter_contract()
    if tuple(contract["parameter_names"]) != PARAMETER_NAMES_8D:
        raise RuntimeError("8D parameter order drift")
    bounds = np.asarray(contract["bounds"], dtype=float)
    if bounds.shape != (8, 2) or not np.all(bounds[:, 1] > bounds[:, 0]):
        raise RuntimeError("invalid 8D prior bounds")
    return bounds


def prior_bounds_7d() -> np.ndarray:
    return prior_bounds_8d()[:7].copy()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "neurolib",
        "sbi",
        "torch",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "nbformat",
        "nbclient",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "python": sys.version,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "packages": packages,
    }


def _source_hashes() -> list[dict[str, Any]]:
    paths = [
        "models/s4_personalize_fig7_v7.py",
        "models/s4_personalize_fig7_v8.py",
        "S4_sbi/src/sleep_sbi/route3_synthetic_preflight.py",
        "S4_sbi/src/sleep_sbi/route3_global_robustness.py",
        "S4_sbi/src/sleep_sbi/simulator_observable_adapter.py",
        "S4_sbi/src/sleep_sbi/figure10_protocol.py",
        "S4_sbi/src/sleep_sbi/figure10_bank.py",
        "S4_sbi/src/sleep_sbi/figure10_training.py",
        "S4_sbi/src/sleep_sbi/figure10_diagnostics.py",
    ]
    rows = []
    for relative in paths:
        path = PROJECT_ROOT / relative
        if not path.exists():
            raise FileNotFoundError(f"protocol source is missing: {relative}")
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _official_reference() -> dict[str, Any]:
    if not OFFICIAL_ROOT.exists():
        raise FileNotFoundError("pinned official reference clone is missing")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=OFFICIAL_ROOT, text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"official reference commit drift: {commit}")
    notebooks = []
    for relative in OFFICIAL_NOTEBOOKS:
        path = OFFICIAL_ROOT / relative
        notebooks.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "repository": "https://github.com/sbi-dev/sbi-practical-guide",
        "commit": commit,
        "paper": "Deistler, Boelts et al. (2025), arXiv:2508.12939",
        "sbi_version": OFFICIAL_SBI_VERSION,
        "notebooks": notebooks,
        "verified_protocol": {
            "training_simulations": 3_000_000,
            "ensemble_members": 5,
            "density_estimator": "NSF",
            "num_transforms": 10,
            "hidden_features": 100,
            "hidden_layers_per_transform": (
                "posterior_nn NSF default; not explicitly passed in train.py"
            ),
            "batch_size": 4096,
            "max_epochs": 400,
            "early_stopping_patience": 30,
            "official_equivalent_sbc_cases": 300,
            "posterior_samples_per_sbc_case": 1000,
            "lc2st_calibration_simulations": 20000,
        },
    }


def build_preregistration() -> dict[str, Any]:
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError("resource benchmark must precede preregistration")
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    bounds8 = prior_bounds_8d()
    bounds7 = bounds8[:7]
    fixed = fixed_c_ctx2th()
    if not bounds8[7, 0] <= fixed <= bounds8[7, 1]:
        raise RuntimeError("historical 7D fixed c_ctx2th lies outside 8D prior")
    feature_names = list(rate_feature_names())
    if len(feature_names) != 14 or len(set(feature_names)) != 14:
        raise RuntimeError("frozen observation schema is not unique 14D")
    scale_projections = benchmark["projections"]
    return {
        "experiment_id": "figure10_matched_8d_7d_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_after_first_scientific_simulation": True,
        "git_head_at_freeze": _git_head(),
        "scope": (
            "synthetic cortical excitatory/inhibitory population-rate posterior "
            "inference only; not scalp EEG inference"
        ),
        "contracts": {
            "primary_8d": {
                "parameter_names": list(PARAMETER_NAMES_8D),
                "parameter_units": list(PARAMETER_UNITS_8D),
                "prior_type": "independent uniform box",
                "prior_bounds": bounds8.tolist(),
            },
            "matched_7d": {
                "parameter_names": list(PARAMETER_NAMES_7D),
                "parameter_units": list(PARAMETER_UNITS_8D[:7]),
                "prior_type": "independent uniform box",
                "prior_bounds": bounds7.tolist(),
                "fixed_parameter": {
                    "name": "c_ctx2th",
                    "value": fixed,
                    "unit": "dimensionless coupling",
                    "source": (
                        "historical frozen Route-3 7D contract; Notebook 10 "
                        "canonical V8a local best"
                    ),
                    "source_artifact": (
                        "S4_sbi/configs/route3_7d_preregistered_v1.json"
                    ),
                },
            },
            "observation": {
                "name": "cortex-rate-only 14D",
                "schema_version": SCHEMA_VERSION,
                "schema_hash": rate_contract_hash(),
                "feature_names": feature_names,
                "dimension": 14,
                "source_signals": ["r_mean_EXC", "r_mean_INH"],
                "source_signal_units_after_conversion": "Hz",
                "not_eeg": True,
            },
            "simulator": {
                "model_version": "v8a_t13",
                "sampling_dt_ms": 1.0,
                "integration_dt_ms": 0.1,
                "duration_s": 60.0,
                "warm_up_s": 5.0,
                "analyzed_window_s": 30.0,
                "seed_policy": (
                    "all four stochastic inputs and numba RNG receive the "
                    "recorded per-row simulator seed"
                ),
                "failed_policy": (
                    "retain theta and seed; NaN x; explicit validity and reason"
                ),
                "undefined_policy": "never fill with zero",
            },
        },
        "official_reference": _official_reference(),
        "protocol_mapping": [
            {
                "official_component": "31D pyloric simulator and 18 summaries",
                "equivalent_8d": "8D thalamocortical model and cortical-rate 14D",
                "equivalent_7d": "same model with c_ctx2th fixed",
                "difference": "different simulator, parameter dimension, and observable",
            },
            {
                "official_component": "five independently initialized NSF NPEs",
                "equivalent_8d": "five raw NSF NPEs",
                "equivalent_7d": "five matched raw NSF NPEs",
                "difference": "CPU training and smaller measured feasible bank",
            },
            {
                "official_component": "joint expected coverage and marginal SBC",
                "equivalent_8d": "300 official-style plus 1024 powered cases",
                "equivalent_7d": "same paired design",
                "difference": "all 8/7 marginals shown rather than only three",
            },
            {
                "official_component": "L-C2ST",
                "equivalent_8d": "20,000 independent calibration simulations",
                "equivalent_7d": "separate matched 20,000 simulations",
                "difference": "local observation is synthetic cortical-rate 14D",
            },
            {
                "official_component": "PPC and posterior marginals/dependencies",
                "equivalent_8d": "new nuisance-seed simulations and all 8 parameters",
                "equivalent_7d": "matched fixed-c_ctx2th baseline",
                "difference": "synthetic observation; no real EEG inference",
            },
        ],
        "resource_decision": {
            "benchmark_path": BENCHMARK_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "benchmark_sha256": sha256_file(BENCHMARK_PATH),
            "workers": MAX_SIMULATION_WORKERS,
            "available_hardware": {
                "cpu": "Intel i5-13600KF, 14 cores / 20 logical",
                "ram_gib": 31.72,
                "cuda_available": False,
                "disk_free_gib_at_audit": 999.91,
            },
            "measured_tracks": benchmark["tracks"],
            "projections": scale_projections,
            "preferred_official_scale": 3_000_000,
            "minimum_requested_scale": 1_000_000,
            "frozen_final_scale_per_track": FINAL_SCALE,
            "frozen_intermediate_scale": INTERMEDIATE_SCALE,
            "hard_limitation": (
                "Measured paired 1M+1M simulation time is about "
                f"{scale_projections['1000000']['paired_tracks_wall_days']:.2f} "
                "days and 3M+3M about "
                f"{scale_projections['3000000']['paired_tracks_wall_days']:.2f} "
                "days on this CPU-only workstation, before training and diagnostics."
            ),
            "claim_consequence": (
                "This is a resource-limited Figure-10-equivalent evaluation, not "
                "an exact 3-million-simulation reproduction. An unqualified PASS "
                "requires all diagnostics to pass and must retain this scale caveat."
            ),
        },
        "simulation_bank": {
            "scales": [INTERMEDIATE_SCALE, FINAL_SCALE],
            "final_scale_per_track": FINAL_SCALE,
            "shard_size": SHARD_SIZE,
            "workers": MAX_SIMULATION_WORKERS,
            "pairing": (
                "same seven Sobol coordinates, paired row ID, and simulator seed; "
                "8D draws c_ctx2th and 7D fixes it"
            ),
            "training_validation_split": "80/20 paired-row split",
            "failed_rows": "retained in manifests; paired-valid intersection trains",
            "feature_scaling": "median/IQR fitted on each track training split only",
            "parameter_scaling": "affine prior-to-unit-box transform",
        },
        "training": {
            "architecture": NSF_ARCHITECTURE,
            "policy": TRAINING_POLICY,
            "members_per_track": 5,
            "independence": (
                "distinct initialization and dataloader seeds; same frozen bank"
            ),
            "ensemble": (
                "equal-weight mixture; component selected before sampling; "
                "mixture log_prob via log-sum-exp"
            ),
            "rounds": 1,
            "proposal": "corresponding frozen prior",
            "intermediate_scale": {
                "scale": INTERMEDIATE_SCALE,
                "8d_members": 1,
                "7d_members": 1,
            },
        },
        "diagnostics": {
            "official_expected_coverage_cases": 300,
            "powered_expected_coverage_cases": 1024,
            "posterior_samples_per_global_case": 1000,
            "sbc_parameters": {"8d": 8, "7d": 7},
            "lc2st_calibration_simulations_per_track": 20000,
            "lc2st_local_posterior_samples": 10000,
            "lc2st_alpha": 0.05,
            "primary_observation": (
                "canonical V8a local-best theta with c_ctx2th equal to the "
                "historical frozen 7D value; frozen before diagnostics"
            ),
            "ppc_posterior_draws_per_track": 256,
            "ppc_prior_draws": 256,
            "posterior_structure_samples_per_track": 50000,
        },
        "operational_verdict": {
            "status": "project reproducibility operationalization, not official threshold",
            "technical_gates": {
                "bank_failure_rate_max": 0.01,
                "diagnostic_failure_rate_max": 0.01,
                "posterior_finite_in_support_rate_min": 0.999,
                "feature_and_parameter_order_exact": True,
            },
            "pass": {
                "powered_joint_rank_ks_p_min": 0.05,
                "clear_sbc_issues_max": 0,
                "sbc_holm_family_alpha": 0.05,
                "ensemble_lc2st_reject": False,
                "ppc_features_better_than_prior_min": 10,
                "ppc_overall_improvement_min_fraction": 0.10,
            },
            "qualified_pass": {
                "powered_joint_rank_ks_p_min": 0.01,
                "clear_sbc_issues_max": 1,
                "ensemble_lc2st_reject": False,
                "ppc_features_better_than_prior_min": 10,
                "ppc_overall_improvement_min_fraction": 0.10,
                "requires_explicit_limitation": True,
            },
            "fail_if": [
                "powered joint-rank KS p < 0.01",
                "two or more Holm-corrected marginal SBC issues",
                "primary ensemble L-C2ST rejects at alpha 0.05",
                "PPC improvement gate fails",
                "technical/data-contract gate fails",
            ],
            "wide_prior_like_marginal": (
                "not a failure if joint/marginal calibration is adequate"
            ),
        },
        "historical_route3": {
            "verdict": "NO-GO",
            "frozen_logic_unchanged": True,
            "source": "ROUTE3_7D_RESCUE_FINAL_DECISION.json",
            "source_sha256": sha256_file(
                PROJECT_ROOT / "ROUTE3_7D_RESCUE_FINAL_DECISION.json"
            ),
            "reuse_final_cases": False,
            "note": (
                "Previously inspected Route-3 cases remain evidentiary history; "
                "the strict verdict is preserved, not retroactively recomputed."
            ),
        },
        "seeds": SEEDS,
        "dataset_disjointness": [
            "training/validation",
            "official 300 global diagnostics",
            "powered 1024 global diagnostics",
            "8D L-C2ST 20,000",
            "7D L-C2ST 20,000",
            "primary observation and PPC",
            "secondary stress observation",
            "historical Route-3 data",
        ],
        "allowed_claims": [
            "synthetic posterior inference in frozen cortical-rate proxy space",
            "matched scientific ablation of releasing c_ctx2th",
            "resource-limited Figure-10-equivalent diagnostics",
        ],
        "forbidden_claims": [
            "real Fpz-Cz EEG posterior inference",
            "simulated scalp EEG",
            "subject-specific physiological recovery",
            "real EEG digital twin",
            "measurement-model validation",
            "official 3M reproduction",
        ],
        "fallback": {
            "bank": (
                "Resume immutable shards. If failure exceeds 1%, stop training "
                "and report terminal technical FAIL."
            ),
            "training": (
                "Resume checkpoints. Terminal member failure remains visible; "
                "do not substitute sequential checkpoints as ensemble members."
            ),
            "diagnostics": (
                "Never tune on final sets. Missing required diagnostic yields FAIL."
            ),
        },
        "environment": _environment(),
        "source_files": _source_hashes(),
    }


def _human_markdown(payload: Mapping[str, Any], digest: str) -> str:
    bounds8 = payload["contracts"]["primary_8d"]["prior_bounds"]
    rows = "\n".join(
        f"| {i + 1} | `{name}` | {bounds8[i][0]:.12g} | {bounds8[i][1]:.12g} |"
        for i, name in enumerate(PARAMETER_NAMES_8D)
    )
    return f"""# Figure-10 8D/7D Preregistration

**Frozen JSON SHA-256:** `{digest}`

This protocol evaluates synthetic cortical population-rate inference. It does
not validate a scalp-EEG measurement model.

## Contracts

| Index | 8D parameter | Lower | Upper |
|---:|---|---:|---:|
{rows}

The matched 7D track uses the first seven rows and fixes `c_ctx2th` to
`{fixed_c_ctx2th():.16g}`. Both tracks use the frozen cortical-rate 14D schema
`{rate_contract_hash()}`.

## Resource-limited scale

The official example uses 3,000,000 simulations. This workstation benchmark
projects {payload['resource_decision']['projections']['1000000']['paired_tracks_wall_days']:.2f}
days for matched 1M+1M and
{payload['resource_decision']['projections']['3000000']['paired_tracks_wall_days']:.2f}
days for matched 3M+3M, before training/diagnostics. The frozen final scale is
{FINAL_SCALE:,} per track, with an {INTERMEDIATE_SCALE:,}-row sensitivity stage.

## Diagnostics and verdicts

The official procedures are reproduced first: five independent NSF NPEs,
equal-weight ensemble, joint expected coverage, marginal SBC, L-C2ST, PPC, and
posterior dependency analysis. Numerical PASS/QUALIFIED PASS/FAIL rules in the
JSON are a project reproducibility operationalization; they are not presented
as thresholds defined by the Practical Guide.

The historical Route-3 strict verdict remains `NO-GO` under its own unchanged
contract.
"""


def freeze_preregistration() -> dict[str, Any]:
    payload = build_preregistration()
    if CONFIG_PATH.exists() or LOCKED_PATH.exists():
        if not CONFIG_PATH.exists() or not LOCKED_PATH.exists():
            raise RuntimeError("partial preregistration freeze detected")
        configured = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        locked = json.loads(LOCKED_PATH.read_text(encoding="utf-8"))
        if configured != locked:
            raise RuntimeError("config and locked preregistration differ")
        digest = sha256_file(LOCKED_PATH)
        recorded = HASH_PATH.read_text(encoding="ascii").strip()
        if digest != recorded:
            raise RuntimeError("locked preregistration hash mismatch")
        return {"payload": locked, "sha256": digest, "resumed": True}
    atomic_json(CONFIG_PATH, payload)
    atomic_json(LOCKED_PATH, payload)
    digest = sha256_file(LOCKED_PATH)
    atomic_text(HASH_PATH, digest + "\n")
    atomic_text(HUMAN_PATH, _human_markdown(payload, digest))
    atomic_text(HUMAN_HASH_PATH, sha256_file(HUMAN_PATH) + "\n")
    atomic_json(
        DEVIATION_PATH,
        {
            "experiment_id": payload["experiment_id"],
            "preregistration_sha256": digest,
            "deviations": [],
        },
    )
    return {"payload": payload, "sha256": digest, "resumed": False}


def read_preregistration() -> dict[str, Any]:
    if not LOCKED_PATH.exists():
        raise FileNotFoundError("Figure-10 preregistration is not frozen")
    return json.loads(LOCKED_PATH.read_text(encoding="utf-8"))


def verify_preregistration() -> str:
    if not (CONFIG_PATH.exists() and LOCKED_PATH.exists() and HASH_PATH.exists()):
        raise FileNotFoundError("Figure-10 preregistration artifacts are incomplete")
    configured = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    locked = json.loads(LOCKED_PATH.read_text(encoding="utf-8"))
    if configured != locked:
        raise RuntimeError("Figure-10 preregistration config/locked mismatch")
    digest = sha256_file(LOCKED_PATH)
    if HASH_PATH.read_text(encoding="ascii").strip() != digest:
        raise RuntimeError("Figure-10 preregistration SHA-256 changed")
    return digest


__all__ = [
    "CONFIG_PATH",
    "FINAL_SCALE",
    "HASH_PATH",
    "HUMAN_PATH",
    "INTERMEDIATE_SCALE",
    "LOCKED_PATH",
    "MAX_SIMULATION_WORKERS",
    "NSF_ARCHITECTURE",
    "PARAMETER_NAMES_7D",
    "PARAMETER_NAMES_8D",
    "RESULTS_ROOT",
    "SEEDS",
    "SHARD_SIZE",
    "TRAINING_POLICY",
    "atomic_json",
    "atomic_text",
    "build_preregistration",
    "canonical_json_hash",
    "fixed_c_ctx2th",
    "freeze_preregistration",
    "prior_bounds_7d",
    "prior_bounds_8d",
    "rate_contract_hash",
    "rate_feature_names",
    "read_preregistration",
    "sha256_file",
    "verify_preregistration",
]
