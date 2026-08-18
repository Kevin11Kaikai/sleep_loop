"""FIT-only selection and distinct HELDOUT-only G6 evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from r4_adapter import BANDS, comparison_features
from r4_core import (
    FROZEN_FILES,
    activate_mechanistic_imports,
    finish_manifest,
    load_protocol_claims,
    new_output_child,
    read_json,
    sha256_file,
    write_canonical,
)


ROLE_PAYLOAD_KEYS = {
    "schema", "role", "subject_pseudonym", "night", "channel", "feature_order", "feature_units",
    "total_eligible_epochs", "assigned_epoch_count", "valid_epochs", "excluded_epochs", "partition_digest",
    "frozen_input_hashes",
}
FEATURE_ORDER = tuple(band[0] for band in BANDS)
SELECTION_FREEZE_KEYS = {
    "schema", "selection_source_role", "fit_payload_sha256", "fit_partition_digest", "fit_valid_epoch_count",
    "fit_epoch_ids", "selected_candidate_id", "selected_calibration", "population_candidate_id",
    "population_calibration", "candidate_fit_scores", "generic_drive", "feature_order", "frozen_input_hashes",
}
CALIBRATION_KEYS = {"slope", "intercept", "fit_nrmse"}


def _validate_payload(path: Path, expected_role: str, frozen_hashes: Mapping[str, str]) -> dict[str, object]:
    value = read_json(path)
    if not isinstance(value, dict) or set(value) != ROLE_PAYLOAD_KEYS:
        raise ValueError("role payload schema mismatch")
    if value["schema"] != "COSTA_R4_ROLE_PAYLOAD_V1" or value["role"] != expected_role:
        raise ValueError("role payload identity mismatch")
    if value["feature_order"] != list(FEATURE_ORDER) or value["feature_units"] != "dimensionless log10 relative power":
        raise ValueError("role payload feature contract mismatch")
    if value["frozen_input_hashes"] != dict(frozen_hashes):
        raise ValueError("role payload frozen-input closure mismatch")
    if int(value["total_eligible_epochs"]) < 4:
        raise ValueError("G6 requires at least four eligible N3 epochs")
    assigned = int(value["assigned_epoch_count"])
    valid = value["valid_epochs"]
    excluded = value["excluded_epochs"]
    if assigned != len(valid) + len(excluded):
        raise ValueError("assigned epoch accounting mismatch")
    ids = [str(row["epoch_id"]) for row in valid] + [str(row["epoch_id"]) for row in excluded]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate epoch in role payload")
    for row in valid:
        if set(row) != {"epoch_id", "stable_rank", "features"} or set(row["features"]) != set(FEATURE_ORDER):
            raise ValueError("valid epoch schema mismatch")
        if not all(math.isfinite(float(row["features"][name])) for name in FEATURE_ORDER):
            raise ValueError("nonfinite role feature")
    return value


def _model_feature_bank(protocol: Mapping[str, object]) -> dict[str, np.ndarray]:
    activate_mechanistic_imports()
    from harness import make_drive
    from model import candidates_from_protocol, simulate

    _time, drive_e, drive_t = make_drive(protocol, 0, 0)
    sfreq = 1000.0 / float(protocol["model"]["dt_ms"])
    result: dict[str, np.ndarray] = {}
    for candidate in candidates_from_protocol(protocol):
        simulation = simulate(candidate, drive_e, drive_t, protocol["model"], protocol["observation_mapping"])
        features = comparison_features(simulation.proxy, sfreq)
        result[candidate.candidate_id] = np.asarray([features[name] for name in FEATURE_ORDER], dtype=np.float64)
    return result


def _observed_matrix(payload: Mapping[str, object]) -> tuple[list[str], np.ndarray]:
    ordered = sorted(payload["valid_epochs"], key=lambda row: str(row["epoch_id"]))
    ids = [str(row["epoch_id"]) for row in ordered]
    matrix = np.asarray([[float(row["features"][name]) for name in FEATURE_ORDER] for row in ordered], dtype=np.float64)
    return ids, matrix


def _fit_affine(prediction: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    x = np.asarray(prediction, dtype=np.float64).reshape(-1)
    y = np.asarray(observed, dtype=np.float64).reshape(-1)
    centered = x - np.mean(x)
    denominator = float(np.dot(centered, centered))
    slope = 0.0 if denominator <= 1e-20 else float(np.dot(centered, y - np.mean(y)) / denominator)
    return slope, float(np.mean(y) - slope * np.mean(x))


def _nrmse(prediction: np.ndarray, observed: np.ndarray) -> float:
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(observed, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(residual))) / (float(np.std(observed, ddof=1)) + 1e-12))


def select_fit(fit_payload_path: Path, output_root: Path, output_child: str) -> Path:
    protocol, _claims, frozen_hashes = load_protocol_claims()
    payload = _validate_payload(fit_payload_path, "FIT", frozen_hashes)
    epoch_ids, observed = _observed_matrix(payload)
    if len(epoch_ids) < 2:
        raise RuntimeError("G6 requires at least two valid FIT epochs")
    bank = _model_feature_bank(protocol)
    scores: dict[str, dict[str, float]] = {}
    for candidate_id, vector in bank.items():
        prediction = np.tile(vector, (len(epoch_ids), 1))
        slope, intercept = _fit_affine(prediction, observed)
        scores[candidate_id] = {
            "slope": slope,
            "intercept": intercept,
            "fit_nrmse": _nrmse(slope * prediction + intercept, observed),
        }
    selected_id = min(scores, key=lambda candidate_id: (scores[candidate_id]["fit_nrmse"], candidate_id))
    if "C000" not in scores:
        raise RuntimeError("frozen matched nonpersonalized candidate C000 is absent")
    freeze = {
        "schema": "COSTA_R4_SELECTION_FREEZE_V1",
        "selection_source_role": "FIT",
        "fit_payload_sha256": sha256_file(fit_payload_path),
        "fit_partition_digest": payload["partition_digest"],
        "fit_valid_epoch_count": len(epoch_ids),
        "fit_epoch_ids": epoch_ids,
        "selected_candidate_id": selected_id,
        "selected_calibration": scores[selected_id],
        "population_candidate_id": "C000",
        "population_calibration": scores["C000"],
        "candidate_fit_scores": scores,
        "generic_drive": {"source": "frozen synthetic_fixture via imported make_drive", "subject_index": 0, "split_index": 0},
        "feature_order": list(FEATURE_ORDER),
        "frozen_input_hashes": frozen_hashes,
    }
    directory = new_output_child(output_root, output_child)
    artifacts = {"SELECTION_FREEZE.json": write_canonical(directory / "SELECTION_FREEZE.json", freeze)}
    finish_manifest(directory, "FIT_ONLY_SELECTION", artifacts, frozen_hashes)
    return directory


def _bootstrap_epoch_advantage(differences: np.ndarray, repetitions: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.integers(0, differences.size, size=differences.size)
        draws[index] = float(np.mean(differences[sampled]))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(lower), float(upper)


def evaluate_heldout(
    heldout_payload_path: Path,
    selection_freeze_path: Path,
    expected_freeze_sha256: str,
    output_root: Path,
    output_child: str,
) -> Path:
    protocol, _claims, frozen_hashes = load_protocol_claims()
    actual_freeze_hash = sha256_file(selection_freeze_path)
    if actual_freeze_hash != expected_freeze_sha256.upper():
        raise ValueError("selection freeze hash mismatch before HELDOUT open")
    freeze = read_json(selection_freeze_path)
    if not isinstance(freeze, dict) or set(freeze) != SELECTION_FREEZE_KEYS or freeze.get("schema") != "COSTA_R4_SELECTION_FREEZE_V1":
        raise ValueError("selection freeze schema mismatch")
    if freeze.get("frozen_input_hashes") != frozen_hashes or freeze.get("selection_source_role") != "FIT":
        raise ValueError("selection freeze closure mismatch")
    if int(freeze.get("fit_valid_epoch_count", 0)) < 2:
        raise ValueError("selection freeze records fewer than two valid FIT epochs")
    expected_candidate_ids = {str(row["candidate_id"]) for row in protocol["candidate_bank"]}
    if set(freeze["candidate_fit_scores"]) != expected_candidate_ids:
        raise ValueError("selection freeze candidate bank mismatch")
    if set(freeze["selected_calibration"]) != CALIBRATION_KEYS or set(freeze["population_calibration"]) != CALIBRATION_KEYS:
        raise ValueError("selection freeze calibration schema mismatch")
    if freeze["feature_order"] != list(FEATURE_ORDER):
        raise ValueError("selection freeze feature order mismatch")

    payload = _validate_payload(heldout_payload_path, "HELDOUT", frozen_hashes)
    epoch_ids, observed = _observed_matrix(payload)
    if len(epoch_ids) < 2:
        raise RuntimeError("G6 requires at least two valid HELDOUT epochs")
    if payload["partition_digest"] != freeze["fit_partition_digest"]:
        raise ValueError("FIT/HELDOUT partition digest mismatch")
    if set(epoch_ids) & set(freeze["fit_epoch_ids"]):
        raise ValueError("FIT/HELDOUT epoch leakage")

    bank = _model_feature_bank(protocol)
    selected_id = str(freeze["selected_candidate_id"])
    population_id = str(freeze["population_candidate_id"])
    if selected_id not in bank or population_id != "C000" or population_id not in bank:
        raise ValueError("frozen candidate identity mismatch")
    personal_raw = np.tile(bank[selected_id], (len(epoch_ids), 1))
    population_raw = np.tile(bank[population_id], (len(epoch_ids), 1))
    personal_cal = freeze["selected_calibration"]
    population_cal = freeze["population_calibration"]
    personal = float(personal_cal["slope"]) * personal_raw + float(personal_cal["intercept"])
    population = float(population_cal["slope"]) * population_raw + float(population_cal["intercept"])
    personal_nrmse = _nrmse(personal, observed)
    population_nrmse = _nrmse(population, observed)
    improvement = (population_nrmse - personal_nrmse) / max(population_nrmse, 1e-12)
    differences = np.mean(np.square(population - observed) - np.square(personal - observed), axis=1)
    repetitions = 1000
    seed = int(protocol["seeds"]["bootstrap_seed"])
    ci_lower, ci_upper = _bootstrap_epoch_advantage(differences, repetitions, seed)
    checks = {
        "eligible_n3_epochs_at_least_four": int(payload["total_eligible_epochs"]) >= 4,
        "valid_fit_epochs_at_least_two": int(freeze["fit_valid_epoch_count"]) >= 2,
        "valid_heldout_epochs_at_least_two": len(epoch_ids) >= 2,
        "relative_nrmse_improvement_at_least_0_05": improvement >= 0.05,
        "paired_epoch_bootstrap_ci_lower_strictly_positive": ci_lower > 0.0,
        "candidate_unchanged_after_fit_freeze": selected_id == str(freeze["selected_candidate_id"]),
    }
    result = {
        "schema": "COSTA_R4_HELDOUT_G6_V1",
        "heldout_source_role": "HELDOUT",
        "selection_freeze_sha256_verified_before_heldout_open": actual_freeze_hash,
        "selected_candidate_id": selected_id,
        "population_candidate_id": population_id,
        "heldout_epoch_ids": epoch_ids,
        "personalized_heldout_nrmse": personal_nrmse,
        "population_heldout_nrmse": population_nrmse,
        "relative_nrmse_improvement": improvement,
        "population_minus_personalized_epoch_mse": differences.tolist(),
        "paired_epoch_bootstrap": {
            "repetitions": repetitions,
            "seed": seed,
            "confidence_level": 0.95,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        },
        "checks": checks,
        "G6_pass": all(checks.values()),
        "units": "dimensionless log10-relative-power feature space; model proxy is not voltage",
        "limitations": "one subject; no population generalization and no empirical wrong-subject specificity",
        "frozen_input_hashes": frozen_hashes,
    }
    directory = new_output_child(output_root, output_child)
    artifacts = {"HELDOUT_G6_RESULT.json": write_canonical(directory / "HELDOUT_G6_RESULT.json", result)}
    finish_manifest(directory, "HELDOUT_ONLY_G6", artifacts, frozen_hashes)
    return directory
