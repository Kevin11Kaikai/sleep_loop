"""Synthetic preflight harness for the frozen clean-room v2 route."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from typing import Mapping, Sequence

import numpy as np

from model import Candidate, candidates_from_protocol, simulate
from partitions import build_single_subject_n3_split, n3_split_digest, stable_u64


def canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_by_id(candidates: Sequence[Candidate], candidate_id: str) -> Candidate:
    matches = [candidate for candidate in candidates if candidate.candidate_id == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"candidate {candidate_id!r} does not occur exactly once")
    return matches[0]


def _fixture_rng(protocol: Mapping[str, object], namespace: str, *parts: str) -> np.random.Generator:
    seed = int(protocol["seeds"]["noise_seed"])
    derived = stable_u64(seed, namespace, *parts) % (2**32)
    return np.random.default_rng(derived)


def make_drive(protocol: Mapping[str, object], subject_index: int, split_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixture = protocol["synthetic_fixture"]
    drive_spec = fixture["drive"]
    dt_s = float(protocol["model"]["dt_ms"]) / 1000.0
    sample_count = int(round(float(fixture["duration_s_per_split"]) / dt_s))
    time_s = np.arange(sample_count, dtype=np.float64) * dt_s
    phase = 0.37 * subject_index + 0.23 * split_index

    pulse_phase = np.mod(time_s + 0.071 * subject_index + 0.113 * split_index, float(drive_spec["e_pulse_period_s"]))
    e_pulse = (pulse_phase < float(drive_spec["e_pulse_width_s"])).astype(np.float64)
    drive_e = (
        float(drive_spec["e_bias"])
        + float(drive_spec["e_slow_amplitude"]) * np.sin(2.0 * np.pi * float(drive_spec["e_slow_hz"]) * time_s + phase)
        + float(drive_spec["e_pulse_amplitude"]) * e_pulse
    )

    burst_period = float(drive_spec["t_burst_period_s"])
    burst_width = float(drive_spec["t_burst_width_s"])
    burst_phase = np.mod(time_s + 0.053 * subject_index + 0.097 * split_index, burst_period)
    burst_window = np.zeros_like(time_s)
    inside = burst_phase < burst_width
    burst_window[inside] = 0.5 - 0.5 * np.cos(2.0 * np.pi * burst_phase[inside] / burst_width)
    drive_t = (
        float(drive_spec["t_bias"])
        + float(drive_spec["t_slow_amplitude"]) * np.sin(2.0 * np.pi * float(drive_spec["t_slow_hz"]) * time_s + 0.61 * phase)
        + float(drive_spec["t_burst_amplitude"])
        * burst_window
        * np.sin(2.0 * np.pi * float(drive_spec["t_burst_carrier_hz"]) * time_s + 0.29 * phase)
    )
    return time_s, drive_e, drive_t


def _fit_affine(prediction: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    x = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    x_centered = x - np.mean(x)
    denominator = float(np.dot(x_centered, x_centered))
    slope = 0.0 if denominator <= 1e-20 else float(np.dot(x_centered, y - np.mean(y)) / denominator)
    intercept = float(np.mean(y) - slope * np.mean(x))
    return slope, intercept


def _nrmse(prediction: np.ndarray, observed: np.ndarray) -> float:
    residual = np.asarray(prediction) - np.asarray(observed)
    denominator = float(np.std(observed, ddof=1)) + 1e-12
    return float(np.sqrt(np.mean(residual * residual)) / denominator)


def _calibrated_score(
    train_prediction: np.ndarray,
    train_observed: np.ndarray,
    test_prediction: np.ndarray,
    test_observed: np.ndarray,
) -> dict[str, object]:
    slope, intercept = _fit_affine(train_prediction, train_observed)
    fitted_train = slope * train_prediction + intercept
    fitted_test = slope * test_prediction + intercept
    return {
        "slope": slope,
        "intercept": intercept,
        "train_nrmse": _nrmse(fitted_train, train_observed),
        "heldout_nrmse": _nrmse(fitted_test, test_observed),
        "heldout_prediction": fitted_test,
        "heldout_error": fitted_test - test_observed,
    }


def _simulate_proxy(
    candidate: Candidate,
    drive_e: np.ndarray,
    drive_t: np.ndarray,
    protocol: Mapping[str, object],
    *,
    zero_coupling: bool = False,
) -> np.ndarray:
    simulation = simulate(
        candidate,
        drive_e,
        drive_t,
        protocol["model"],
        protocol["observation_mapping"],
        cortex_to_thalamus_override=0.0 if zero_coupling else None,
        thalamus_to_cortex_override=0.0 if zero_coupling else None,
    )
    return simulation.proxy


def _observed_truth(
    candidate: Candidate,
    drive_e: np.ndarray,
    drive_t: np.ndarray,
    protocol: Mapping[str, object],
    subject_id: str,
    split_name: str,
    *,
    zero_coupling: bool = False,
) -> np.ndarray:
    proxy = _simulate_proxy(candidate, drive_e, drive_t, protocol, zero_coupling=zero_coupling)
    rng = _fixture_rng(protocol, "observation-noise", subject_id, split_name, "zero" if zero_coupling else "truth")
    return proxy + rng.normal(0.0, float(protocol["synthetic_fixture"]["noise_sd_mau"]), size=proxy.size)


def _candidate_predictions(
    candidates: Sequence[Candidate],
    drives: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    protocol: Mapping[str, object],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        candidate.candidate_id: (
            _simulate_proxy(candidate, drives[0][1], drives[0][2], protocol),
            _simulate_proxy(candidate, drives[1][1], drives[1][2], protocol),
        )
        for candidate in candidates
    }


def _evaluate_synthetic_subjects(protocol: Mapping[str, object], candidates: Sequence[Candidate]) -> tuple[list[dict[str, object]], dict[str, dict[str, np.ndarray]]]:
    subjects = protocol["synthetic_fixture"]["subjects"]
    population_id = str(protocol["selection"]["population_candidate_id"])
    intermediate: dict[str, dict[str, object]] = {}

    for subject_index, subject in enumerate(subjects):
        subject_id = str(subject["subject_pseudonym"])
        truth_id = str(subject["truth_candidate_id"])
        truth_candidate = _candidate_by_id(candidates, truth_id)
        drives = [make_drive(protocol, subject_index, split_index) for split_index in (0, 1)]
        observed = (
            _observed_truth(truth_candidate, drives[0][1], drives[0][2], protocol, subject_id, "FIT"),
            _observed_truth(truth_candidate, drives[1][1], drives[1][2], protocol, subject_id, "HELDOUT"),
        )
        predictions = _candidate_predictions(candidates, drives, protocol)
        scores = {
            candidate.candidate_id: _calibrated_score(
                predictions[candidate.candidate_id][0], observed[0], predictions[candidate.candidate_id][1], observed[1]
            )
            for candidate in candidates
        }
        selected_id = min(scores, key=lambda item: (float(scores[item]["train_nrmse"]), item))
        selected_candidate = _candidate_by_id(candidates, selected_id)
        ablated = (
            _simulate_proxy(selected_candidate, drives[0][1], drives[0][2], protocol, zero_coupling=True),
            _simulate_proxy(selected_candidate, drives[1][1], drives[1][2], protocol, zero_coupling=True),
        )
        intermediate[subject_id] = {
            "truth_id": truth_id,
            "selected_id": selected_id,
            "drives": drives,
            "observed": observed,
            "predictions": predictions,
            "scores": scores,
            "ablated_score": _calibrated_score(ablated[0], observed[0], ablated[1], observed[1]),
            "population_score": scores[population_id],
        }

    summaries: list[dict[str, object]] = []
    errors: dict[str, dict[str, np.ndarray]] = {}
    wrong_cycle = protocol["synthetic_fixture"]["wrong_subject_cycle"]
    for subject in subjects:
        subject_id = str(subject["subject_pseudonym"])
        item = intermediate[subject_id]
        wrong_subject_id = str(wrong_cycle[subject_id])
        wrong_candidate_id = str(intermediate[wrong_subject_id]["selected_id"])
        wrong_predictions = item["predictions"][wrong_candidate_id]
        wrong_score = _calibrated_score(wrong_predictions[0], item["observed"][0], wrong_predictions[1], item["observed"][1])
        selected_score = item["scores"][item["selected_id"]]
        population_score = item["population_score"]
        ablated_score = item["ablated_score"]
        personalized_nrmse = float(selected_score["heldout_nrmse"])
        population_nrmse = float(population_score["heldout_nrmse"])
        wrong_nrmse = float(wrong_score["heldout_nrmse"])
        ablated_nrmse = float(ablated_score["heldout_nrmse"])
        summaries.append(
            {
                "subject_pseudonym": subject_id,
                "truth_candidate_id": item["truth_id"],
                "selected_candidate_id": item["selected_id"],
                "train_nrmse": float(selected_score["train_nrmse"]),
                "heldout_nrmse": personalized_nrmse,
                "population_candidate_id": population_id,
                "population_heldout_nrmse": population_nrmse,
                "wrong_synthetic_subject": wrong_subject_id,
                "wrong_candidate_id": wrong_candidate_id,
                "wrong_heldout_nrmse": wrong_nrmse,
                "zero_coupling_heldout_nrmse": ablated_nrmse,
                "relative_improvement_vs_population": (population_nrmse - personalized_nrmse) / max(population_nrmse, 1e-12),
                "relative_improvement_vs_wrong": (wrong_nrmse - personalized_nrmse) / max(wrong_nrmse, 1e-12),
                "relative_improvement_vs_zero_coupling": (ablated_nrmse - personalized_nrmse) / max(ablated_nrmse, 1e-12),
            }
        )
        errors[subject_id] = {
            "personalized": np.asarray(selected_score["heldout_error"], dtype=np.float64),
            "population": np.asarray(population_score["heldout_error"], dtype=np.float64),
        }
    return summaries, errors


def _evaluate_null_world(protocol: Mapping[str, object], candidates: Sequence[Candidate]) -> dict[str, object]:
    base_candidate = _candidate_by_id(candidates, str(protocol["selection"]["population_candidate_id"]))
    drives = [make_drive(protocol, 19, split_index) for split_index in (0, 1)]
    observed = (
        _observed_truth(base_candidate, drives[0][1], drives[0][2], protocol, "NULL_WORLD", "FIT", zero_coupling=True),
        _observed_truth(base_candidate, drives[1][1], drives[1][2], protocol, "NULL_WORLD", "HELDOUT", zero_coupling=True),
    )
    null_prediction = (
        _simulate_proxy(base_candidate, drives[0][1], drives[0][2], protocol, zero_coupling=True),
        _simulate_proxy(base_candidate, drives[1][1], drives[1][2], protocol, zero_coupling=True),
    )
    null_score = _calibrated_score(null_prediction[0], observed[0], null_prediction[1], observed[1])
    predictions = _candidate_predictions(candidates, drives, protocol)
    scores = {
        candidate.candidate_id: _calibrated_score(
            predictions[candidate.candidate_id][0], observed[0], predictions[candidate.candidate_id][1], observed[1]
        )
        for candidate in candidates
    }
    best_id = min(scores, key=lambda item: (float(scores[item]["train_nrmse"]), item))
    best_mechanistic = float(scores[best_id]["heldout_nrmse"])
    null_nrmse = float(null_score["heldout_nrmse"])
    return {
        "best_mechanistic_candidate_id": best_id,
        "best_mechanistic_heldout_nrmse": best_mechanistic,
        "registered_null_heldout_nrmse": null_nrmse,
        "registered_null_nrmse_advantage": best_mechanistic - null_nrmse,
    }


def _nonisomorphic_observation(time_s: np.ndarray, subject_index: int, split_index: int, protocol: Mapping[str, object]) -> np.ndarray:
    phase = 0.31 * subject_index + 0.47 * split_index
    envelope = 0.72 + 0.28 * np.sin(2.0 * np.pi * 0.19 * time_s + 0.2 * phase)
    oscillator = (
        3.2 * envelope * np.sin(2.0 * np.pi * 7.4 * time_s + phase)
        + 1.1 * np.sin(2.0 * np.pi * 3.1 * time_s + 0.7 * phase)
    )
    rng = _fixture_rng(protocol, "nonisomorphic-noise", str(subject_index), str(split_index))
    return oscillator + rng.normal(0.0, float(protocol["synthetic_fixture"]["noise_sd_mau"]), size=time_s.size)


def _evaluate_nonisomorphic_world(protocol: Mapping[str, object], candidates: Sequence[Candidate]) -> dict[str, object]:
    drives = [make_drive(protocol, 23, split_index) for split_index in (0, 1)]
    observed = (
        _nonisomorphic_observation(drives[0][0], 23, 0, protocol),
        _nonisomorphic_observation(drives[1][0], 23, 1, protocol),
    )
    predictions = _candidate_predictions(candidates, drives, protocol)
    scores = {
        candidate.candidate_id: _calibrated_score(
            predictions[candidate.candidate_id][0], observed[0], predictions[candidate.candidate_id][1], observed[1]
        )
        for candidate in candidates
    }
    best_id = min(scores, key=lambda item: (float(scores[item]["train_nrmse"]), item))
    return {
        "best_mechanistic_candidate_id": best_id,
        "best_mechanistic_train_nrmse": float(scores[best_id]["train_nrmse"]),
        "best_mechanistic_heldout_nrmse": float(scores[best_id]["heldout_nrmse"]),
        "latent_world": "external oscillator; no cortical E/I or thalamic relay/reticular states",
    }


def _block_bootstrap_advantage(
    errors: Mapping[str, Mapping[str, np.ndarray]], protocol: Mapping[str, object]
) -> dict[str, float | int]:
    gate = protocol["gates"]["G4"]
    repetitions = int(gate["bootstrap_repetitions"])
    block_samples = int(gate["block_samples"])
    rng = np.random.default_rng(int(protocol["seeds"]["bootstrap_seed"]))
    subject_ids = sorted(errors)
    draws = np.empty(repetitions, dtype=np.float64)
    observed_subject_means = []
    for subject_id in subject_ids:
        personal = np.square(errors[subject_id]["personalized"])
        population = np.square(errors[subject_id]["population"])
        observed_subject_means.append(float(np.mean(population - personal)))
    for repetition in range(repetitions):
        subject_draws = []
        for subject_id in subject_ids:
            personal = np.square(errors[subject_id]["personalized"])
            population = np.square(errors[subject_id]["population"])
            difference = population - personal
            blocks_needed = int(math.ceil(difference.size / block_samples))
            maximum_start = max(difference.size - block_samples + 1, 1)
            starts = rng.integers(0, maximum_start, size=blocks_needed)
            sampled = np.concatenate([difference[start : start + block_samples] for start in starts])[: difference.size]
            subject_draws.append(float(np.mean(sampled)))
        draws[repetition] = float(np.mean(subject_draws))
    alpha = 1.0 - float(gate["confidence_level"])
    lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "bootstrap_repetitions": repetitions,
        "block_samples": block_samples,
        "observed_mean_paired_mse_advantage": float(np.mean(observed_subject_means)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def _validate_g5(protocol: Mapping[str, object]) -> dict[str, object]:
    mapping = protocol["observation_mapping"]
    limits = [str(item) for item in mapping["limits"]]
    checks = {
        "four_source_weights": len(mapping["source_weights_E_I_T_R"]) == 4,
        "operator_chain_complete": len(mapping["operator_order"]) == 4,
        "arbitrary_model_unit_declared": "arbitrary model unit" in str(mapping["proxy_unit"]),
        "scalp_causality_limit_declared": any("scalp EEG causality" in item for item in limits),
        "physical_forward_model_absent_declared": any("volume conductor" in item for item in limits),
    }
    return {"checks": checks, "pass": all(checks.values()), "proxy_unit": mapping["proxy_unit"], "mapping_id": mapping["mapping_id"]}


def _validate_g6(protocol: Mapping[str, object]) -> dict[str, object]:
    expected_fields = set(protocol["interfaces"]["G6_epoch_metadata_exact_fields"])
    forbidden_fields = set(protocol["interfaces"]["G6_forbidden_fields"])
    epochs = [
        {"subject_pseudonym": "SYNTHETIC_INTERFACE_ONLY", "epoch_id": f"N3_EPOCH_{index:03d}", "stage": "N3", "night": "Night-1"}
        for index in range(12)
    ]
    schema_exact = all(set(epoch) == expected_fields and not (set(epoch) & forbidden_fields) for epoch in epochs)
    split_a = build_single_subject_n3_split(epochs, int(protocol["seeds"]["partition_seed"]), float(protocol["partitions"]["fit_fraction"]))
    split_b = build_single_subject_n3_split(list(reversed(epochs)), int(protocol["seeds"]["partition_seed"]), float(protocol["partitions"]["fit_fraction"]))
    digest_a = n3_split_digest(split_a)
    digest_b = n3_split_digest(split_b)
    fit_ids = {row.epoch_id for row in split_a if row.role == "FIT"}
    heldout_ids = {row.epoch_id for row in split_a if row.role == "HELDOUT"}
    checks = {
        "exact_metadata_schema": schema_exact,
        "input_order_invariant_digest": digest_a == digest_b,
        "fit_nonempty": bool(fit_ids),
        "heldout_nonempty": bool(heldout_ids),
        "fit_heldout_disjoint": fit_ids.isdisjoint(heldout_ids),
        "complete_once_only": len(fit_ids | heldout_ids) == len(epochs) == len(split_a),
        "single_subject": len({row.subject_pseudonym for row in split_a}) == 1,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "partition_digest": digest_a,
        "fit_count": len(fit_ids),
        "heldout_count": len(heldout_ids),
        "scope": "synthetic metadata exercise of future one-subject Night-1 N3 interface; no empirical metadata opened",
    }


def execute_core(protocol: Mapping[str, object]) -> dict[str, object]:
    """Execute all deterministic scientific and interface checks for G2--G6."""
    candidates = candidates_from_protocol(protocol)
    subject_summaries, errors = _evaluate_synthetic_subjects(protocol, candidates)
    null_world = _evaluate_null_world(protocol, candidates)
    nonisomorphic_world = _evaluate_nonisomorphic_world(protocol, candidates)
    bootstrap = _block_bootstrap_advantage(errors, protocol)
    g5 = _validate_g5(protocol)
    g6 = _validate_g6(protocol)

    g2_thresholds = protocol["gates"]["G2"]
    truth_recovery = float(np.mean([row["truth_candidate_id"] == row["selected_candidate_id"] for row in subject_summaries]))
    max_truth_nrmse = max(float(row["heldout_nrmse"]) for row in subject_summaries)
    min_ablation_improvement = min(float(row["relative_improvement_vs_zero_coupling"]) for row in subject_summaries)
    g2_checks = {
        "truth_recovery": truth_recovery >= float(g2_thresholds["truth_recovery_fraction_min"]),
        "truth_heldout_error": max_truth_nrmse <= float(g2_thresholds["truth_heldout_nrmse_max"]),
        "zero_coupling_ablation": min_ablation_improvement >= float(g2_thresholds["truth_vs_zero_coupling_relative_improvement_min"]),
        "registered_null_wins_null_world": float(null_world["registered_null_nrmse_advantage"]) >= float(g2_thresholds["null_model_nrmse_advantage_min"]),
        "nonisomorphic_world_rejected": float(nonisomorphic_world["best_mechanistic_heldout_nrmse"]) >= float(g2_thresholds["nonisomorphic_best_mechanistic_nrmse_min"]),
    }

    population_improvements = [float(row["relative_improvement_vs_population"]) for row in subject_summaries]
    wrong_improvements = [float(row["relative_improvement_vs_wrong"]) for row in subject_summaries]
    g3_thresholds = protocol["gates"]["G3"]
    g3_checks = {
        "every_synthetic_subject_beats_population": min(population_improvements) >= float(g3_thresholds["all_subject_personalized_improvement_min"]),
        "median_beats_population": float(np.median(population_improvements)) >= float(g3_thresholds["median_vs_population_improvement_min"]),
        "every_synthetic_subject_beats_wrong": min(wrong_improvements) >= float(g3_thresholds["all_subject_personalized_improvement_min"]),
        "median_beats_wrong": float(np.median(wrong_improvements)) >= float(g3_thresholds["median_vs_wrong_subject_improvement_min"]),
    }

    g4_thresholds = protocol["gates"]["G4"]
    g4_checks = {
        "paired_ci_lower_positive": float(bootstrap["ci_lower"]) > float(g4_thresholds["paired_mse_advantage_ci_lower_min"]),
        "truth_in_training_minimizer_set": all(row["truth_candidate_id"] == row["selected_candidate_id"] for row in subject_summaries),
        "finite_interval": bool(np.isfinite(float(bootstrap["ci_lower"])) and np.isfinite(float(bootstrap["ci_upper"]))),
    }

    gates = {
        "G2": {"pass": all(g2_checks.values()), "checks": g2_checks},
        "G3": {"pass": all(g3_checks.values()), "checks": g3_checks},
        "G4": {"pass": all(g4_checks.values()), "checks": g4_checks},
        "G5": {"pass": bool(g5["pass"]), "checks": g5["checks"]},
        "G6": {"pass": bool(g6["pass"]), "checks": g6["checks"]},
    }
    return {
        "analysis_scope": "synthetic/model-internal only; not empirical evidence",
        "candidate_bank_count": len(candidates),
        "candidate_bank": [asdict(candidate) for candidate in candidates],
        "synthetic_subjects": subject_summaries,
        "truth_recovery_fraction": truth_recovery,
        "maximum_truth_heldout_nrmse": max_truth_nrmse,
        "minimum_truth_vs_zero_coupling_relative_improvement": min_ablation_improvement,
        "null_world": null_world,
        "nonisomorphic_world": nonisomorphic_world,
        "uncertainty": bootstrap,
        "G5_mapping": g5,
        "G6_interface": g6,
        "gates": gates,
    }


def benchmark_model(protocol: Mapping[str, object]) -> dict[str, float | int | str]:
    """Benchmark a single 30-second epoch and scale only the one-subject workflow."""
    candidates = candidates_from_protocol(protocol)
    representative = candidates[len(candidates) // 2]
    dt_s = float(protocol["model"]["dt_ms"]) / 1000.0
    epoch_seconds = float(protocol["future_empirical_runtime_plan"]["epoch_duration_s"])
    sample_count = int(round(epoch_seconds / dt_s))
    time_s = np.arange(sample_count, dtype=np.float64) * dt_s
    drive_e = 0.34 + 0.18 * np.sin(2.0 * np.pi * 0.73 * time_s)
    drive_t = 0.23 + 0.13 * np.sin(2.0 * np.pi * 1.17 * time_s)
    start = time.perf_counter()
    _simulate_proxy(representative, drive_e, drive_t, protocol)
    elapsed = time.perf_counter() - start
    candidate_count = len(candidates)
    overhead = float(protocol["future_empirical_runtime_plan"]["serial_overhead_multiplier"])
    per_epoch_all_candidates = elapsed * candidate_count * overhead
    equivalent_epochs = int(protocol["future_empirical_runtime_plan"]["synthetic_verification_benchmark_equivalent_epochs"])
    return {
        "benchmark_epoch_seconds": epoch_seconds,
        "benchmark_single_candidate_wall_seconds": elapsed,
        "estimated_serial_seconds_per_n3_epoch_all_16_candidates_with_overhead": per_epoch_all_candidates,
        "frozen_formula": "estimated_total_seconds = observed_N_N3_epochs * estimated_serial_seconds_per_n3_epoch_all_16_candidates_with_overhead",
        "observed_N_N3_epochs": "UNKNOWN_PHASE1_NO_EMPIRICAL_METADATA_OPENED",
        "synthetic_eight_epoch_workflow_estimate_seconds": per_epoch_all_candidates * equivalent_epochs,
        "single_subjects": 1,
    }
