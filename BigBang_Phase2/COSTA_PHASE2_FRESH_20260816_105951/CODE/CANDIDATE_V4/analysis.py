"""Target estimands, cell scores, exact policies, shape rules, and disposition."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, pi
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .contracts import ContractBundle, ContractError, FREEZE_HASH, PROTOCOL_AUDIT_HASH
from .run_registry import DRAWS, PHASES_A
from .t4_attribution import FORMAL_ROW_ID


TARGETS = tuple(f"T{i}" for i in range(1, 9))


def wrap(value: float) -> float:
    result = float(atan2(np.sin(value), np.cos(value)))
    return -pi if result == pi else result


def _finite(value: Any) -> bool:
    return value is not None and np.isscalar(value) and bool(np.isfinite(value))


def _median(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None
    return float(np.median(array))


def _mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None
    return float(np.mean(array, dtype=np.float64))


def fisher_lee(alpha: Sequence[float], beta: Sequence[float]) -> float | None:
    if len(alpha) != len(beta) or len(alpha) < 2:
        return None
    numerator = 0.0
    alpha_sq = 0.0
    beta_sq = 0.0
    for left in range(len(alpha)):
        for right in range(left + 1, len(alpha)):
            a = float(np.sin(float(alpha[left]) - float(alpha[right])))
            b = float(np.sin(float(beta[left]) - float(beta[right])))
            numerator += a * b
            alpha_sq += a * a
            beta_sq += b * b
    denominator = float(np.sqrt(alpha_sq * beta_sq))
    if denominator <= 1e-12 or not np.isfinite(denominator):
        return None
    return float(numerator / denominator)


def phase_response(alpha: Sequence[float], beta: Sequence[float]) -> dict[str, float] | None:
    if len(alpha) != len(beta) or not all(_finite(value) for value in beta):
        return None
    rho = fisher_lee(alpha, beta)
    if rho is None:
        return None
    errors = [abs(wrap(float(b) - float(a))) for a, b in zip(alpha, beta)]
    try:
        low = next(idx for idx, value in enumerate(alpha) if float(value) == -pi / 2.0)
        high = next(idx for idx, value in enumerate(alpha) if float(value) == pi / 2.0)
    except StopIteration:
        return None
    separation = abs(wrap(float(beta[high]) - float(beta[low])))
    return {
        "rho_c": rho,
        "tracking_error": float(np.median(np.asarray(errors, dtype=np.float64))),
        "separation": separation,
    }


def phase_score(statistics: Mapping[str, float]) -> float:
    return float(min(
        statistics["rho_c"] / 0.70,
        min(4.0, (pi / 6.0) / max(statistics["tracking_error"], 1e-12)),
        statistics["separation"] / (pi / 3.0),
    ))


def _metric(record: Mapping[str, Any], name: str) -> Any:
    return record["metrics"].get(name)


def _target_valid(record: Mapping[str, Any], target: str) -> bool:
    return bool(record["validity"].get("stable") and record["validity"].get(f"{target}_valid"))


def _param_signature(parameters: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(key), float(value)) for key, value in parameters.items()))


@dataclass
class BlockCurve:
    draw_id: str
    seed: int
    sham: Mapping[str, Any]
    conditions: Mapping[tuple[tuple[str, float], ...], Mapping[str, Any]]

    @property
    def block_id(self) -> str:
        return f"{self.draw_id}::S{self.seed}"


def build_level_a_curves(records: Sequence[Mapping[str, Any]], target: str) -> list[BlockCurve]:
    group = "T6_T7_SHARED" if target in {"T6", "T7"} else target
    selected = [record for record in records if record["stage"] == "LEVEL_A_WORKER" and record["target_id"] == group]
    by_run = {record["run_id"]: record for record in records}
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for record in selected:
        grouped.setdefault((record["parameter_draw_id"], int(record["seed"])), []).append(record)
    curves: list[BlockCurve] = []
    for (draw, seed), block_records in sorted(grouped.items()):
        sham_ids = {record["paired_sham_run_id"] for record in block_records}
        if len(sham_ids) != 1:
            raise ContractError(f"paired sham identity disagreement for {target}/{draw}/S{seed}")
        sham_id = next(iter(sham_ids))
        if sham_id not in by_run:
            raise ContractError(f"missing paired sham {sham_id}")
        sham = by_run[sham_id]
        conditions = {
            _param_signature(record["parameters"]): record
            for record in block_records if not record["is_sham"]
        }
        curves.append(BlockCurve(draw, seed, sham, conditions))
    return curves


def _find(curve: BlockCurve, **parameters: float) -> Mapping[str, Any] | None:
    wanted = _param_signature(parameters)
    return curve.conditions.get(wanted)


def _valid_curve(curve: BlockCurve, target: str, records: Sequence[Mapping[str, Any]]) -> bool:
    return _target_valid(curve.sham, target) and all(_target_valid(record, target) for record in records)


def target_block_rows(target: str, curves: Sequence[BlockCurve]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for curve in curves:
        row: dict[str, Any] = {"draw_id": curve.draw_id, "seed": curve.seed, "block_id": curve.block_id}
        sham = curve.sham
        required: list[Mapping[str, Any]] = []
        if target == "T1":
            low = _find(curve, A=0.8, f=0.55); high = _find(curve, A=0.8, f=1.15)
            if low is None or high is None: continue
            required = [low, high]
            if not _valid_curve(curve, target, required): continue
            f0, fl, fh = (_metric(r, "T1_dominant_so_peak_hz") for r in (sham, low, high))
            p_low = _metric(low, "T1_plv_by_reference").get("0.55")
            p_high = _metric(high, "T1_plv_by_reference").get("1.15")
            if not all(_finite(v) for v in (f0, fl, fh, p_low, p_high)): continue
            e_low = abs(f0 - 0.55) - abs(fl - 0.55)
            e_high = abs(f0 - 1.15) - abs(fh - 1.15)
            row.update(E1=0.5 * (e_low + e_high), e_low=e_low, e_high=e_high, plv_low=p_low, plv_high=p_high)
        elif target == "T2":
            negative = _find(curve, d=-1.5); positive = _find(curve, d=1.5)
            if negative is None or positive is None: continue
            required = [negative, positive]
            if not _valid_curve(curve, target, required): continue
            a0, a_neg, a_pos = (_metric(r, "T2_so_median_peak_to_peak_m_u") for r in (sham, negative, positive))
            if not all(_finite(v) for v in (a0, a_neg, a_pos)) or a0 <= 0.10: continue
            row.update(E2=(a_pos - a_neg) / a0, pos_gain=(a_pos - a0) / a0, neg_reduction=(a0 - a_neg) / a0)
        elif target == "T3":
            negative = _find(curve, d=-2.0, f_rep=0.85); positive = _find(curve, d=2.0, f_rep=0.85)
            if negative is None or positive is None: continue
            required = [negative, positive]
            if not _valid_curve(curve, target, required): continue
            d0, d_neg, d_pos = (_metric(r, "T3_so_density_per_min") for r in (sham, negative, positive))
            if not all(_finite(v) for v in (d0, d_neg, d_pos)): continue
            row.update(E3=d_pos - d_neg, pos_gain=d_pos - d0, neg_reduction=d0 - d_neg)
        elif target == "T4":
            negative = _find(curve, d=-1.5, f_c=13.0, f_rep=0.85); positive = _find(curve, d=1.5, f_c=13.0, f_rep=0.85)
            if negative is None or positive is None: continue
            required = [negative, positive]
            if not _valid_curve(curve, target, required): continue
            s0, s_neg, s_pos = (_metric(r, "T4_spindle_density_per_min") for r in (sham, negative, positive))
            if not all(_finite(v) for v in (s0, s_neg, s_pos)): continue
            row.update(E4=s_pos - s_neg, pos_gain=s_pos - s0, neg_reduction=s0 - s_neg)
        elif target == "T5":
            negative = _find(curve, d=-1.2, f_c=13.0); positive = _find(curve, d=1.2, f_c=13.0)
            if negative is None or positive is None: continue
            required = [negative, positive]
            if not _valid_curve(curve, target, required): continue
            r0, r_neg, r_pos = (_metric(r, "T5_spindle_event_rms_m_u") for r in (sham, negative, positive))
            if not all(_finite(v) for v in (r0, r_neg, r_pos)) or r0 <= 1e-12: continue
            row.update(E5=(r_pos - r_neg) / r0, pos_gain=(r_pos - r0) / r0, neg_reduction=(r0 - r_neg) / r0)
        elif target == "T6":
            phase_zero = _find(curve, phi=0.0, A=1.0, f_c=13.0)
            if phase_zero is None or not _valid_curve(curve, target, [phase_zero]): continue
            sham_mi, active_mi = (_metric(r, "T6_bias_corrected_mi") for r in (sham, phase_zero))
            if not all(_finite(v) for v in (sham_mi, active_mi)): continue
            row.update(E6=active_mi - sham_mi)
        elif target == "T7":
            records_by_phase = [_find(curve, phi=float(phi), A=1.0, f_c=13.0) for phi in PHASES_A]
            if any(record is None for record in records_by_phase): continue
            concrete = [record for record in records_by_phase if record is not None]
            if not _valid_curve(curve, target, concrete): continue
            beta = [_metric(record, "T7_preferred_phase_radians") for record in concrete]
            stats = phase_response(PHASES_A, beta)
            if stats is None: continue
            row.update(stats)
            row["alpha"] = list(PHASES_A); row["beta"] = list(beta); row["H"] = phase_score(stats)
        elif target == "T8":
            doses = (-2.0, -1.25, -0.5, 0.5, 1.25, 2.0)
            records_by_dose = [_find(curve, d=dose, fraction=1.0) for dose in doses]
            if any(record is None for record in records_by_dose): continue
            concrete = [record for record in records_by_dose if record is not None]
            if not _valid_curve(curve, target, concrete): continue
            sham_o = _metric(sham, "T8_occupancy")
            values = [_metric(record, "T8_occupancy") for record in concrete]
            if not all(_finite(v) for v in [sham_o, *values]): continue
            all_doses = (0.0, *doses)
            all_values = (float(sham_o), *(float(v) for v in values))
            row.update(occupancy_by_dose={str(d): v for d, v in zip(all_doses, all_values)}, E8=max(all_values) - min(all_values))
            row["persistence_by_dose"] = {str(d): bool(_metric(r, "T8_persistent")) for d, r in zip(doses, concrete)}
            row["persistence_by_dose"]["0.0"] = bool(_metric(sham, "T8_persistent"))
        else:
            raise ContractError(f"unknown target {target}")
        rows.append(row)
    return rows


def denominator_status(rows: Sequence[Mapping[str, Any]], *, minimum_blocks: int, minimum_seeds: int, minimum_draws: int) -> dict[str, Any]:
    seeds = {int(row["seed"]) for row in rows}
    draws = {str(row["draw_id"]) for row in rows}
    passed = len(rows) >= minimum_blocks and len(seeds) >= minimum_seeds and len(draws) >= minimum_draws
    return {
        "status": "ESTIMABLE" if passed else "NOT_ESTIMABLE",
        "complete_blocks": len(rows),
        "distinct_seeds": len(seeds),
        "distinct_draws": len(draws),
        "minimum_complete_blocks": minimum_blocks,
        "minimum_distinct_seeds": minimum_seeds,
        "minimum_distinct_draws": minimum_draws,
    }


def estimate_target_slacks(
    target: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    formal_t4_attribution: Mapping[str, Any] | None = None,
) -> dict[str, float] | None:
    if not rows:
        return None
    if target == "T1":
        values = {
            "E1-0.05": _median(row["E1"] for row in rows) - 0.05,
            "PLV_low-0.50": _median(row["plv_low"] for row in rows) - 0.50,
            "PLV_high-0.50": _median(row["plv_high"] for row in rows) - 0.50,
            "move_fraction_low-0.80": _mean(row["e_low"] > 0.0 for row in rows) - 0.80,
            "move_fraction_high-0.80": _mean(row["e_high"] > 0.0 for row in rows) - 0.80,
        }
    elif target == "T2":
        values = {"E2-0.15": _median(row["E2"] for row in rows) - 0.15, "fractional_positive_endpoint_gain-0.05": _median(row["pos_gain"] for row in rows) - 0.05, "fractional_negative_endpoint_reduction-0.05": _median(row["neg_reduction"] for row in rows) - 0.05}
    elif target == "T3":
        values = {"E3-3.0": _median(row["E3"] for row in rows) - 3.0, "positive_endpoint_gain_events_per_min-1.5": _median(row["pos_gain"] for row in rows) - 1.5, "negative_endpoint_reduction_events_per_min-1.5": _median(row["neg_reduction"] for row in rows) - 1.5}
    elif target == "T4":
        f_auto = None if formal_t4_attribution is None else formal_t4_attribution.get("F_auto")
        if not _finite(f_auto) or formal_t4_attribution.get("domain_row_id") != FORMAL_ROW_ID: return None
        values = {"E4-1.0": _median(row["E4"] for row in rows) - 1.0, "positive_endpoint_gain_spindles_per_min-0.5": _median(row["pos_gain"] for row in rows) - 0.5, "negative_endpoint_reduction_spindles_per_min-0.5": _median(row["neg_reduction"] for row in rows) - 0.5, "autonomous_added_event_fraction-0.50": float(f_auto) - 0.50}
    elif target == "T5":
        values = {"E5-0.15": _median(row["E5"] for row in rows) - 0.15, "fractional_positive_endpoint_gain-0.05": _median(row["pos_gain"] for row in rows) - 0.05, "fractional_negative_endpoint_reduction-0.05": _median(row["neg_reduction"] for row in rows) - 0.05}
    elif target == "T6":
        values = {"E6-0.010": _median(row["E6"] for row in rows) - 0.010, "positive_block_fraction-0.75": _mean(row["E6"] > 0.0 for row in rows) - 0.75}
    elif target == "T7":
        values = {"circular_correlation-0.70": _median(row["rho_c"] for row in rows) - 0.70, "pi_over_6-median_absolute_tracking_error": pi / 6.0 - _median(row["tracking_error"] for row in rows), "observed_phase_separation-pi_over_3": _median(row["separation"] for row in rows) - pi / 3.0}
    elif target == "T8":
        medians: dict[float, float] = {}
        for dose in (-2.0, -1.25, -0.5, 0.0, 0.5, 1.25, 2.0):
            medians[dose] = _median(row["occupancy_by_dose"][str(dose)] for row in rows)
        d_hi = sorted(medians, key=lambda d: (-medians[d], abs(d), d))[0]
        persistence = _mean(bool(row["persistence_by_dose"][str(d_hi)]) for row in rows)
        values = {"E8-one_third": _median(row["E8"] for row in rows) - 1.0 / 3.0, "persistent_block_fraction_at_d_hi-0.75": persistence - 0.75}
    else:
        raise ContractError(f"unknown target {target}")
    if any(value is None or not np.isfinite(value) for value in values.values()):
        return None
    return {key: float(value) for key, value in values.items()}


def cell_score(
    target: str,
    condition: Mapping[str, Any],
    paired_blocks: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    attribution: Mapping[str, Any] | None = None,
) -> float | None:
    params = {k: float(v) for k, v in condition["parameters"].items()}
    if not paired_blocks:
        return None
    if target == "T1":
        f, _a = params["f"], params["A"]
        improvements = []; plvs = []
        for active, sham in paired_blocks:
            fu, fs = _metric(active, "T1_dominant_so_peak_hz"), _metric(sham, "T1_dominant_so_peak_hz")
            plv = _metric(active, "T1_plv_by_reference").get(f"{f:.2f}")
            if not all(_finite(v) for v in (fu, fs, plv)): return None
            improvements.append(abs(fs - f) - abs(fu - f)); plvs.append(plv)
        return float(min(_median(improvements) / 0.05, _median(plvs) / 0.50, _mean(value > 0.0 for value in improvements) / 0.80))
    if target == "T2":
        if params["d"] == 0.0: return 0.0
        values = []
        for active, sham in paired_blocks:
            au, a0 = _metric(active, "T2_so_median_peak_to_peak_m_u"), _metric(sham, "T2_so_median_peak_to_peak_m_u")
            if not _finite(au) or not _finite(a0) or a0 <= 0.10: return None
            values.append(np.sign(params["d"]) * (au - a0) / (0.05 * a0))
        return _median(values)
    if target == "T3":
        if params["d"] == 0.0: return 0.0
        return _median(np.sign(params["d"]) * (_metric(a, "T3_so_density_per_min") - _metric(s, "T3_so_density_per_min")) / 1.5 for a, s in paired_blocks)
    if target == "T4":
        if params["d"] == 0.0: return 0.0
        density = _median(((_metric(a, "T4_spindle_density_per_min") - _metric(s, "T4_spindle_density_per_min")) / 0.5) if params["d"] > 0.0 else ((_metric(s, "T4_spindle_density_per_min") - _metric(a, "T4_spindle_density_per_min")) / 0.5) for a, s in paired_blocks)
        if params["d"] < 0.0: return density
        if attribution is None or not _finite(attribution.get("F_auto")): return None
        return float(min(density, float(attribution["F_auto"]) / 0.50))
    if target == "T5":
        if params["d"] == 0.0: return 0.0
        values = []
        for active, sham in paired_blocks:
            ru, r0 = _metric(active, "T5_spindle_event_rms_m_u"), _metric(sham, "T5_spindle_event_rms_m_u")
            if not _finite(ru) or not _finite(r0) or r0 <= 1e-12: return None
            values.append(np.sign(params["d"]) * (ru - r0) / (0.05 * r0))
        return _median(values)
    if target == "T6":
        deltas = [_metric(a, "T6_bias_corrected_mi") - _metric(s, "T6_bias_corrected_mi") for a, s in paired_blocks]
        if not all(_finite(v) for v in deltas): return None
        return float(min(_median(deltas) / 0.010, _mean(value > 0.0 for value in deltas) / 0.75))
    if target == "T7":
        values = []
        for active, _sham in paired_blocks:
            beta = _metric(active, "T7_preferred_phase_radians")
            if not _finite(beta): return None
            values.append(2.0 - abs(wrap(beta - params["phi"])) / (pi / 6.0))
        return _median(values)
    if target == "T8":
        if params["d"] == 0.0: return 0.0
        oriented = []; persistent = []
        for active, sham in paired_blocks:
            ou, o0 = _metric(active, "T8_occupancy"), _metric(sham, "T8_occupancy")
            if not _finite(ou) or not _finite(o0): return None
            oriented.append(np.sign(params["d"]) * (ou - o0) / (1.0 / 6.0)); persistent.append(bool(_metric(active, "T8_persistent")))
        return float(min(_median(oriented), _mean(persistent) / 0.75))
    raise ContractError(f"unknown target {target}")


def _policy_condition(record: Mapping[str, Any]) -> dict[str, Any]:
    return {"condition_id": record["condition_id"], "channel_id": record["channel_id"], "parameters": dict(record["parameters"])}


def select_level_a_policy(
    target: str,
    candidates: Sequence[Mapping[str, Any]],
    scores: Mapping[str, float | None],
    *,
    t8_eligible_condition_ids: set[str] | None = None,
    t4_domain_by_condition: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    estimable = [record for record in candidates if _finite(scores.get(record["condition_id"]))]
    if target == "T1":
        ordered = sorted(estimable, key=lambda r: float(r["parameters"]["f"]))
        if len(ordered) != 5: return {"target_id": target, "status": "NOT_ESTIMABLE", "conditions": []}
        score = min(float(scores[r["condition_id"]]) for r in ordered)
        return {"target_id": target, "status": "SELECTED", "policy_type": "MAPPING", "score": score, "conditions": [_policy_condition(r) for r in ordered]}
    if target == "T7":
        ordered = sorted(estimable, key=lambda r: PHASES_A.index(float(r["parameters"]["phi"])))
        if len(ordered) != 8: return {"target_id": target, "status": "NOT_ESTIMABLE", "conditions": []}
        return {"target_id": target, "status": "SELECTED", "policy_type": "MAPPING", "score": None, "conditions": [_policy_condition(r) for r in ordered]}
    if target == "T8":
        estimable = [r for r in estimable if t8_eligible_condition_ids and r["condition_id"] in t8_eligible_condition_ids]
    if target in {"T2", "T3"}:
        estimable = [r for r in estimable if float(r["parameters"]["d"]) != 0.0]
        key: Callable[[Mapping[str, Any]], Any] = lambda r: (-float(scores[r["condition_id"]]), abs(float(r["parameters"]["d"])), float(r["parameters"]["d"]))
    elif target in {"T4", "T5"}:
        estimable = [r for r in estimable if float(r["parameters"]["d"]) > 0.0]
        key = lambda r: (-float(scores[r["condition_id"]]), float(r["parameters"]["d"]))
    elif target == "T6":
        key = lambda r: (-float(scores[r["condition_id"]]), abs(wrap(float(r["parameters"]["phi"]))), float(r["parameters"]["phi"]))
    elif target == "T8":
        key = lambda r: (-float(scores[r["condition_id"]]), abs(float(r["parameters"]["d"])), float(r["parameters"]["d"]))
    else:
        raise ContractError(f"unsupported Level-A policy target {target}")
    if not estimable: return {"target_id": target, "status": "NOT_ESTIMABLE", "conditions": []}
    selected = sorted(estimable, key=key)[0]
    result = {"target_id": target, "status": "SELECTED", "policy_type": "SINGLE_CONDITION", "score": float(scores[selected["condition_id"]]), "conditions": [_policy_condition(selected)]}
    if target == "T4":
        if t4_domain_by_condition is None or selected["condition_id"] not in t4_domain_by_condition: raise ContractError("T4 policy missing verifier-domain mapping")
        result["attribution_domain_row_id"] = t4_domain_by_condition[selected["condition_id"]]
    return result


def select_level_b_policy(
    target: str,
    candidates: Sequence[Mapping[str, Any]],
    scores: Mapping[str, float | None],
    *,
    mapping_scores: Mapping[float, float | None] | None = None,
    t8_eligible_condition_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Apply the frozen Level-B mapping/single-condition maximum and tie rule."""
    if target == "T1":
        if mapping_scores is None: return {"target_id": target, "status": "NOT_ESTIMABLE", "conditions": []}
        eligible = [(float(amplitude), float(score)) for amplitude, score in mapping_scores.items() if _finite(score)]
        if not eligible: return {"target_id": target, "status": "NOT_ESTIMABLE", "conditions": []}
        amplitude, score = sorted(eligible, key=lambda item: (-item[1], item[0]))[0]
        mapped = sorted([r for r in candidates if float(r["parameters"]["A"]) == amplitude], key=lambda r: float(r["parameters"]["f"]))
        if len(mapped) != 5: raise ContractError("T1 Level-B mapping is incomplete")
        return {"target_id": target, "status": "SELECTED", "policy_type": "MAPPING", "score": score, "conditions": [_policy_condition(r) for r in mapped]}
    if target == "T7":
        if mapping_scores is None: return {"target_id": target, "status": "NOT_ESTIMABLE", "conditions": []}
        eligible = [(float(carrier), float(score)) for carrier, score in mapping_scores.items() if _finite(score)]
        if not eligible: return {"target_id": target, "status": "NOT_ESTIMABLE", "conditions": []}
        carrier, score = sorted(eligible, key=lambda item: (-item[1], abs(item[0] - 13.0), item[0]))[0]
        mapped = sorted([r for r in candidates if float(r["parameters"]["f_c"]) == carrier], key=lambda r: (-pi, -pi/2, 0.0, pi/2).index(float(r["parameters"]["phi"])))
        if len(mapped) != 4: raise ContractError("T7 Level-B mapping is incomplete")
        return {"target_id": target, "status": "SELECTED", "policy_type": "MAPPING", "score": score, "conditions": [_policy_condition(r) for r in mapped]}
    estimable = [r for r in candidates if _finite(scores.get(r["condition_id"]))]
    if target in {"T2", "T3"}:
        estimable = [r for r in estimable if float(r["parameters"]["d"]) != 0.0]
    elif target in {"T4", "T5"}:
        estimable = [r for r in estimable if float(r["parameters"]["d"]) > 0.0]
    elif target == "T6":
        estimable = [r for r in estimable if float(r["parameters"]["A"]) > 0.0]
    elif target == "T8":
        estimable = [r for r in estimable if t8_eligible_condition_ids and r["condition_id"] in t8_eligible_condition_ids]
    else:
        raise ContractError(f"unsupported Level-B policy target {target}")
    if not estimable: return {"target_id": target, "status": "NOT_ESTIMABLE", "conditions": []}
    def key(record: Mapping[str, Any]) -> tuple[float, float, int]:
        p = {name: float(value) for name, value in record["parameters"].items()}
        if target == "T2": cost = abs(p["d"]) / 1.5 + p["A"] / 1.2
        elif target == "T3": cost = abs(p["d"]) / 2.0 + abs(p["f_rep"] - 0.85) / 0.30
        elif target == "T4": cost = p["d"] / 1.5 + abs(p["f_c"] - 13.0) / 2.0
        elif target == "T5": cost = p["d"] / 1.2 + abs(p["f_c"] - 13.0) / 2.0
        elif target == "T6": cost = p["A"] + abs(wrap(p["phi"])) / pi
        else: cost = abs(p["d"]) / 2.0 + p["fraction"]
        return (-float(scores[record["condition_id"]]), cost, int(record["condition_index"]))
    selected = sorted(estimable, key=key)[0]
    return {"target_id": target, "status": "SELECTED", "policy_type": "SINGLE_CONDITION", "score": float(scores[selected["condition_id"]]), "conditions": [_policy_condition(selected)]}


def select_conditional_level_b_targets(level_a_summary: Mapping[str, Mapping[str, Any]]) -> list[str]:
    tiers = {"T1": 0, "T4": 0, "T6": 0, "T2": 1, "T3": 1, "T5": 1, "T7": 1, "T8": 2}
    eligible = [target for target in TARGETS if bool(level_a_summary.get(target, {}).get("level_b_eligible"))]
    eligible.sort(key=lambda target: (
        tiers[target],
        -float(level_a_summary[target]["normalized_minimum_lcb_slack"]),
        target,
    ))
    selected = eligible[:3]
    if len(selected) < 3 and "T8" in eligible and "T8" not in selected:
        selected.append("T8")
    return selected[:3]


def shape_consistency(target: str, curves: Sequence[BlockCurve]) -> dict[str, Any]:
    draw_rows: list[dict[str, Any]] = []
    for draw_id in DRAWS:
        selected = [curve for curve in curves if curve.draw_id == draw_id]
        row: dict[str, Any] = {"draw_id": draw_id, "pass": False, "reason": None}
        if not selected:
            row["reason"] = "MISSING_DRAW"
            draw_rows.append(row); continue
        try:
            if target == "T1":
                low = []; high = []; plv_low = []; plv_high = []
                for curve in selected:
                    lo = _find(curve, A=0.8, f=0.55); hi = _find(curve, A=0.8, f=1.15)
                    if lo is not None and _valid_curve(curve, target, [lo]):
                        f0 = _metric(curve.sham, "T1_dominant_so_peak_hz"); fu = _metric(lo, "T1_dominant_so_peak_hz"); plv = _metric(lo, "T1_plv_by_reference").get("0.55")
                        if all(_finite(v) for v in (f0, fu, plv)):
                            low.append(abs(f0 - 0.55) - abs(fu - 0.55)); plv_low.append(plv)
                    if hi is not None and _valid_curve(curve, target, [hi]):
                        f0 = _metric(curve.sham, "T1_dominant_so_peak_hz"); fu = _metric(hi, "T1_dominant_so_peak_hz"); plv = _metric(hi, "T1_plv_by_reference").get("1.15")
                        if all(_finite(v) for v in (f0, fu, plv)):
                            high.append(abs(f0 - 1.15) - abs(fu - 1.15)); plv_high.append(plv)
                if min(map(len, (low, high, plv_low, plv_high))) < 5: raise ValueError
                values = [_median(low), _median(high), _median(plv_low), _median(plv_high)]
                row.update(values=values, pass_=values[0] > 0 and values[1] > 0 and values[2] >= 0.50 and values[3] >= 0.50)
                row["pass"] = row.pop("pass_")
            elif target in {"T2", "T3", "T4", "T5"}:
                config = {
                    "T2": ((-1.5,-1,-0.5,0,0.5,1,1.5), "T2_so_median_peak_to_peak_m_u", lambda d: {"d": d}),
                    "T3": ((-2,-1.25,-0.5,0,0.5,1.25,2), "T3_so_density_per_min", lambda d: {"d": d,"f_rep":0.85}),
                    "T4": ((-1.5,-1,-0.5,0,0.5,1,1.5), "T4_spindle_density_per_min", lambda d: {"d":d,"f_c":13.0,"f_rep":0.85}),
                    "T5": ((-1.2,-0.8,-0.4,0,0.4,0.8,1.2), "T5_spindle_event_rms_m_u", lambda d: {"d":d,"f_c":13.0}),
                }[target]
                medians = []
                for dose in config[0]:
                    values = []
                    for curve in selected:
                        record = curve.sham if dose == 0 else _find(curve, **config[2](dose))
                        valid = record is not None and _target_valid(curve.sham, target) and _target_valid(record, target)
                        if valid:
                            value = _metric(record, config[1])
                            if _finite(value): values.append(value)
                    if len(values) < 5: raise ValueError
                    medians.append(_median(values))
                adjacent = [medians[idx+1]-medians[idx] for idx in range(6)]
                row.update(condition_medians=medians, adjacent_differences=adjacent, nonnegative_adjacent_count=sum(value >= 0 for value in adjacent), pass_=(medians[-1] > medians[0] and sum(value >= 0 for value in adjacent) >= 5))
                row["pass"] = row.pop("pass_")
            elif target == "T6":
                deltas = []
                for curve in selected:
                    active = _find(curve, phi=0.0,A=1.0,f_c=13.0)
                    if active is not None and _valid_curve(curve, target, [active]):
                        active_mi=_metric(active,"T6_bias_corrected_mi");sham_mi=_metric(curve.sham,"T6_bias_corrected_mi")
                        if _finite(active_mi) and _finite(sham_mi):deltas.append(active_mi-sham_mi)
                if len(deltas)<5:raise ValueError
                row.update(median_gain=_median(deltas), pass_=(_median(deltas)>0)); row["pass"] = row.pop("pass_")
            elif target == "T7":
                beta = []
                for phi in PHASES_A:
                    phases = []
                    for curve in selected:
                        active = _find(curve,phi=phi,A=1.0,f_c=13.0)
                        if active is not None and _valid_curve(curve, target, [active]):
                            value=_metric(active,"T7_preferred_phase_radians")
                            if _finite(value):phases.append(value)
                    if len(phases)<5:raise ValueError
                    resultant = np.sum(np.exp(1j*np.asarray(phases,dtype=np.float64)))
                    if abs(resultant)<=1e-12: raise ValueError
                    beta.append(float(atan2(resultant.imag,resultant.real)))
                stats=phase_response(PHASES_A,beta)
                if stats is None: raise ValueError
                row.update(statistics=stats,pass_=(stats["rho_c"]>0 and stats["separation"]>0)); row["pass"]=row.pop("pass_")
            elif target == "T8":
                doses=(-2.0,-1.25,-0.5,0.5,1.25,2.0); q={};occ_values=[]
                for curve in selected:
                    if _target_valid(curve.sham,target) and _finite(_metric(curve.sham,"T8_occupancy")):occ_values.append(_metric(curve.sham,"T8_occupancy"))
                if len(occ_values)<5:raise ValueError
                occ={0.0:_median(occ_values)}
                for dose in doses:
                    values=[]; oriented=[]
                    for curve in selected:
                        active=_find(curve,d=dose,fraction=1.0)
                        if active is not None and _valid_curve(curve,target,[active]):
                            o=_metric(active,"T8_occupancy");base=_metric(curve.sham,"T8_occupancy")
                            if _finite(o) and _finite(base):values.append(o);oriented.append(np.sign(dose)*(o-base))
                    if len(values)<5 or len(oriented)<5:raise ValueError
                    occ[dose]=_median(values); q[dose]=_median(oriented)
                pairs=((-2,-1.25),(-1.25,-0.5),(0.5,1.25),(1.25,2)); adjacency=any(q[a]>0 and q[b]>0 for a,b in pairs)
                d_hi=sorted(occ,key=lambda d:(-occ[d],abs(d),d))[0]
                persistent=[]
                for curve in selected:
                    active=curve.sham if d_hi==0 else _find(curve,d=d_hi,fraction=1.0)
                    if active is not None and _target_valid(active,target) and _target_valid(curve.sham,target):persistent.append(bool(_metric(active,"T8_persistent")))
                if len(persistent)<5:raise ValueError
                fraction=_mean(persistent); row.update(Q=q,d_hi=d_hi,persistence_fraction=fraction,pass_=(adjacency and fraction>=0.75)); row["pass"]=row.pop("pass_")
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            row["pass"] = False; row["reason"] = "MISSING_OR_INVALID_REQUIRED_CONDITION"
        draw_rows.append(row)
    count = sum(bool(row["pass"]) for row in draw_rows)
    return {"target_id": target, "draws": draw_rows, "pass_count": count, "required_pass_count": 4, "pass": count >= 4}


def phase2b_disposition(bundle: ContractBundle) -> dict[str, Any]:
    phase2b = bundle.protocol["phase2b"]
    decision = bundle.advancement["current_campaign_decision"]
    if phase2b["formal_status"] != "NOT_ESTIMABLE" or decision["advancing_targets"] != []:
        raise ContractError("frozen Phase2B disposition changed")
    return {
        "schema_version": "4.0.0",
        "artifact_type": "PHASE2B_DISPOSITION",
        "campaign_id": bundle.protocol["campaign_id"],
        "candidate_identity": bundle.protocol["candidate_identity"],
        "freeze_sha256": FREEZE_HASH,
        "protocol_audit_sha256": PROTOCOL_AUDIT_HASH,
        "status": "NOT_ESTIMABLE",
        "eligible_certified_subjects": 0,
        "minimum_certified_subjects": 3,
        "same_candidate_executable_identity_available": False,
        "advancing_targets": [],
        "synthetic_substitution": "PROHIBITED",
        "formal_execution": "PROHIBITED_UNDER_CANDIDATE_V4",
        "claim": "Formal Phase 2B and C2 are not estimable under Candidate V4.",
    }
