"""Eight-by-eight verifier primitives, aggregation order, and classifications."""

from __future__ import annotations

from collections import defaultdict
from math import pi
from typing import Any, Mapping, Sequence

import numpy as np

from .analysis import cell_score, phase_response, phase_score, wrap
from .contracts import ContractError
from .detectors import broadband_specificity_veto
from .run_registry import DRAWS, VERIFIER_SEEDS
from .statistics import enumerate_cross_target_family


TARGETS = tuple(f"T{i}" for i in range(1, 9))


def _metric(record: Mapping[str, Any], name: str) -> Any:
    return record["metrics"].get(name)


def _valid(record: Mapping[str, Any], target: str) -> bool:
    return bool(record["validity"].get("stable") and record["validity"].get(f"{target}_valid"))


def _primitive(target: str, active: Mapping[str, Any], sham: Mapping[str, Any]) -> float | None:
    if not _valid(active, target) or not _valid(sham, target):
        return None
    if target == "T1":
        fu, fs = _metric(active, "T1_dominant_so_peak_hz"), _metric(sham, "T1_dominant_so_peak_hz")
        if fu is None or fs is None: return None
        return -abs(float(fu) - float(fs)) / 0.05
    if target == "T2":
        au, a0 = _metric(active, "T2_so_median_peak_to_peak_m_u"), _metric(sham, "T2_so_median_peak_to_peak_m_u")
        if au is None or a0 is None or float(a0) <= 0.10: return None
        return (float(au) - float(a0)) / (0.05 * float(a0))
    if target == "T3":
        return (float(_metric(active, "T3_so_density_per_min")) - float(_metric(sham, "T3_so_density_per_min"))) / 1.5
    if target == "T4":
        return (float(_metric(active, "T4_spindle_density_per_min")) - float(_metric(sham, "T4_spindle_density_per_min"))) / 0.5
    if target == "T5":
        ru, r0 = _metric(active, "T5_spindle_event_rms_m_u"), _metric(sham, "T5_spindle_event_rms_m_u")
        if ru is None or r0 is None or float(r0) <= 1e-12: return None
        return (float(ru) - float(r0)) / (0.05 * float(r0))
    if target == "T6":
        bu, b0 = _metric(active, "T6_bias_corrected_mi"), _metric(sham, "T6_bias_corrected_mi")
        if bu is None or b0 is None: return None
        return (float(bu) - float(b0)) / 0.010
    if target == "T7":
        beta, beta0 = _metric(active, "T7_preferred_phase_radians"), _metric(sham, "T7_preferred_phase_radians")
        if beta is None or beta0 is None: return None
        return -abs(wrap(float(beta) - float(beta0))) / (pi / 6.0)
    if target == "T8":
        return (float(_metric(active, "T8_occupancy")) - float(_metric(sham, "T8_occupancy"))) / (1.0 / 6.0)
    raise ContractError(f"unknown cross-target column {target}")


def _condition_run_id(row_target: str, condition_index: int, draw: str, seed: int) -> str:
    return f"X3__{row_target}__K{condition_index:02d}__{draw}__S{seed}"


def _logical_condition_records(
    row_target: str,
    policies: Mapping[str, Mapping[str, Any]],
    lookup: Mapping[str, Mapping[str, Any]],
    block_keys: Sequence[tuple[str, int]],
) -> list[list[tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    conditions = policies[row_target]["conditions"]
    result: list[list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = []
    for kk, logical in enumerate(conditions):
        blocks: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for draw, seed in block_keys:
                if row_target == "T6":
                    matches = [
                        idx for idx, condition in enumerate(policies["T7"]["conditions"])
                        if condition["parameters"] == logical["parameters"]
                    ]
                    if len(matches) != 1: raise ContractError("T6 verifier reuse must resolve once")
                    run_id = _condition_run_id("T7", matches[0], draw, seed)
                else:
                    run_id = _condition_run_id(row_target, kk, draw, seed)
                sham_id = f"X3__SHAM__{draw}__S{seed}"
                if run_id not in lookup or sham_id not in lookup:
                    raise ContractError(f"missing verifier record {run_id} or {sham_id}")
                blocks.append((lookup[run_id], lookup[sham_id]))
        result.append(blocks)
    return result


def _denominator(blocks: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], target: str, *, bootstrap: bool) -> bool:
    complete = [(a, s) for a, s in blocks if _primitive(target, a, s) is not None]
    seeds = {int(a["seed"]) for a, _ in complete}
    draws = {str(a["parameter_draw_id"]) for a, _ in complete}
    if bootstrap:
        return len(complete) > 0
    return len(complete) >= 20 and len(seeds) >= 3 and len(draws) >= 5


def _diagonal_score(
    target: str,
    condition_blocks: Sequence[Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]],
    policies: Mapping[str, Mapping[str, Any]],
    t4_attribution: Mapping[str, Any] | None,
    *,
    bootstrap: bool,
) -> float | None:
    if target == "T1":
        scores = []
        for logical, blocks in zip(policies[target]["conditions"], condition_blocks):
            value = cell_score("T1", logical, blocks)
            if value is None: return None
            scores.append(value)
        return float(min(scores))
    if target == "T7":
        alpha = [float(condition["parameters"]["phi"]) for condition in policies[target]["conditions"]]
        statistics = []
        if bootstrap:
            lengths = {len(blocks) for blocks in condition_blocks}
            if len(lengths) != 1: return None
            for occurrence in range(next(iter(lengths))):
                beta = [_metric(blocks[occurrence][0], "T7_preferred_phase_radians") for blocks in condition_blocks]
                value = phase_response(alpha, beta)
                if value is not None: statistics.append(value)
        else:
            by_block: dict[tuple[str, int], list[float]] = defaultdict(list)
            for blocks in condition_blocks:
                for active, _sham in blocks:
                    beta = _metric(active, "T7_preferred_phase_radians")
                    if beta is not None:
                        by_block[(active["parameter_draw_id"], int(active["seed"]))].append(float(beta))
            for key in sorted(by_block):
                beta = by_block[key]
                if len(beta) != len(alpha): continue
                value = phase_response(alpha, beta)
                if value is not None: statistics.append(value)
        if len(statistics) < (1 if bootstrap else 20): return None
        aggregate = {
            "rho_c": float(np.median([row["rho_c"] for row in statistics])),
            "tracking_error": float(np.median([row["tracking_error"] for row in statistics])),
            "separation": float(np.median([row["separation"] for row in statistics])),
        }
        return phase_score(aggregate)
    if len(condition_blocks) != 1:
        raise ContractError(f"{target} diagonal expected one condition")
    attribution = t4_attribution if target == "T4" else None
    complete = [pair for pair in condition_blocks[0] if _primitive(target, pair[0], pair[1]) is not None]
    if len(complete) < (1 if bootstrap else 20):
        return None
    return cell_score(target, policies[target]["conditions"][0], complete, attribution=attribution)


def compute_cross_target_matrix(
    policies: Mapping[str, Mapping[str, Any]],
    verifier_records: Sequence[Mapping[str, Any]],
    *,
    selected_t4_attribution: Mapping[str, Any] | None,
    block_keys: Sequence[tuple[str, int]] | None = None,
    bootstrap: bool = False,
) -> dict[str, Any]:
    if tuple(sorted(policies)) != TARGETS:
        raise ContractError("cross-target computation requires T1-T8 policies")
    lookup = {record["run_id"]: record for record in verifier_records}
    if len(lookup) != 456:
        raise ContractError(f"verifier record count {len(lookup)}, expected 456")
    matrix: dict[str, Any] = {}
    family: dict[str, float | None] = {member: None for member in enumerate_cross_target_family()}
    if block_keys is None:
        block_keys = tuple((draw, seed) for draw in DRAWS for seed in VERIFIER_SEEDS)
    for row_target in TARGETS:
        condition_blocks = _logical_condition_records(row_target, policies, lookup, block_keys)
        row_cells: dict[str, Any] = {}
        diagonal = _diagonal_score(row_target, condition_blocks, policies, selected_t4_attribution, bootstrap=bootstrap)
        for column in TARGETS:
            condition_values: list[tuple[float, float]] = []
            missing = False
            for blocks in condition_blocks:
                if not _denominator(blocks, column, bootstrap=bootstrap):
                    missing = True; break
                primitives = [_primitive(column, active, sham) for active, sham in blocks]
                finite = [float(value) for value in primitives if value is not None and np.isfinite(value)]
                if len(finite) < (1 if bootstrap else 20):
                    missing = True; break
                median = float(np.median(np.asarray(finite, dtype=np.float64)))
                benefit = max(0.0, median); adverse = max(0.0, -median)
                if column == "T4" and row_target == "T4":
                    allowed_statuses = {"BOOTSTRAP_ESTIMABLE", "BOOTSTRAP_ZERO_ADDED"} if bootstrap else {"ESTIMABLE"}
                    if selected_t4_attribution is None or selected_t4_attribution.get("denominator_status") not in allowed_statuses:
                        missing = True; break
                    replica = selected_t4_attribution.get("F_replica")
                    if replica is None: missing = True; break
                    adverse = max(adverse, float(replica) / 0.50)
                condition_values.append((benefit, adverse))
            if missing:
                benefit = adverse = None
            else:
                benefit = min(value[0] for value in condition_values)
                adverse = max(value[1] for value in condition_values)
                if row_target == column:
                    if diagonal is None or not np.isfinite(diagonal):
                        benefit = adverse = None
                    else:
                        benefit = max(0.0, float(diagonal))
                        adverse = max(float(adverse), max(0.0, -float(diagonal)))
            i, j = int(row_target[1:]), int(column[1:])
            family[f"X3/T{i}/T{j}/BENEFIT"] = benefit
            family[f"X3/T{i}/T{j}/ADVERSE"] = adverse
            row_cells[column] = {"benefit": benefit, "adverse": adverse, "estimable": benefit is not None and adverse is not None}
        matrix[row_target] = row_cells
    return {"shape": [8, 8], "matrix": matrix, "family_members": family, "member_count": len(family)}


def classify_cross_target_matrix(
    point_matrix: Mapping[str, Any],
    simultaneous_bands: Mapping[str, Any],
    *,
    hard_vetoes: Mapping[tuple[str, str], bool] | None = None,
) -> dict[str, Any]:
    hard_vetoes = hard_vetoes or {}
    bands = simultaneous_bands.get("members", {})
    result: dict[str, Any] = {}
    for i in range(1, 9):
        row: dict[str, str] = {}
        for j in range(1, 9):
            benefit = bands.get(f"X3/T{i}/T{j}/BENEFIT")
            adverse = bands.get(f"X3/T{i}/T{j}/ADVERSE")
            if hard_vetoes.get((f"T{i}", f"T{j}"), False):
                label = "UNACCEPTABLE_COLLATERAL_EFFECT"
            elif benefit is None or adverse is None:
                label = "NOT_ESTIMABLE"
            elif i == j and benefit["lower"] > 1.0 and adverse["upper"] < 1.0:
                label = "INTENDED_IMPROVEMENT"
            elif i != j and benefit["lower"] > 1.0 and adverse["upper"] < 1.0:
                label = "BENEFICIAL_SECONDARY_EFFECT"
            elif benefit["upper"] <= 1.0 and adverse["upper"] <= 1.0:
                label = "NEGLIGIBLE"
            else:
                label = "TRADE_OFF"
            row[f"T{j}"] = label
        result[f"T{i}"] = row
    return {"classifications": result}


def cross_target_hard_vetoes(
    policies: Mapping[str, Mapping[str, Any]],
    verifier_records: Sequence[Mapping[str, Any]],
    simultaneous_bands: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], bool], dict[str, dict[str, list[str]]]]:
    """Evaluate registered hard vetoes before matrix label assignment."""
    lookup = {record["run_id"]: record for record in verifier_records}
    block_keys = tuple((draw, seed) for draw in DRAWS for seed in VERIFIER_SEEDS)
    bands = simultaneous_bands.get("members", {})
    vetoes: dict[tuple[str, str], bool] = {}
    reasons: dict[str, dict[str, list[str]]] = {}
    for row_target in TARGETS:
        condition_blocks = _logical_condition_records(row_target, policies, lookup, block_keys)
        row_reasons: set[str] = set()
        for blocks in condition_blocks:
            for active, sham in blocks:
                validity = active.get("validity", {})
                if not all(bool(validity.get(key)) for key in ("finite", "rate_bounds", "mean_rate_bounds", "extreme_fraction_ok", "paired_power_within_4x")):
                    row_reasons.add("HARD_STABILITY_OR_PLAUSIBILITY_VETO")
                if not bool(validity.get("paired_threshold_identity")) or active["metrics"].get("spindle_thresholds") != sham["metrics"].get("spindle_thresholds"):
                    row_reasons.add("DETECTOR_THRESHOLD_CHANGE_VETO")
                if broadband_specificity_veto(active["metrics"], sham["metrics"]):
                    row_reasons.add("BROADBAND_WITHOUT_TARGET_FRACTION_INCREASE_VETO")
                if not bool(validity.get("T8_valid")):
                    row_reasons.add("T8_REGIME_INVALIDITY_VETO")
        reasons[row_target] = {}
        for column in TARGETS:
            cell_reasons = set(row_reasons)
            if row_target != column:
                band = bands.get(f"X3/{row_target}/{column}/ADVERSE")
                if band is not None and float(band["lower"]) > 1.0:
                    cell_reasons.add("NON_TARGET_ADVERSE_SIMULTANEOUS_LCB_ABOVE_MARGIN")
            ordered = sorted(cell_reasons)
            reasons[row_target][column] = ordered
            vetoes[(row_target, column)] = bool(ordered)
    return vetoes, reasons
