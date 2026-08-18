"""Frozen hierarchical bootstrap, randomization tests, and Holm correction."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .analysis import fisher_lee, phase_response, phase_score
from .contracts import ContractError
from .run_registry import DRAWS, LEVEL_B_DRAWS, LEVEL_B_SEEDS, VERIFIER_SEEDS, WORKER_SEEDS


BOOTSTRAP_REPLICATES = 10_000
MONTE_CARLO_REPLICATES = 100_000


@dataclass(frozen=True)
class HierarchicalIndices:
    draw_indices: np.ndarray
    seed_indices: np.ndarray
    draw_order: tuple[str, ...]
    seed_order: tuple[int, ...]
    seed: int


def hierarchical_indices(
    *,
    seed: int,
    draw_order: Sequence[str],
    seed_order: Sequence[int],
    replicates: int = BOOTSTRAP_REPLICATES,
) -> HierarchicalIndices:
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    n_draws, n_seeds = len(draw_order), len(seed_order)
    draws = rng.integers(0, n_draws, size=(replicates, n_draws), dtype=np.int16)
    seeds = rng.integers(0, n_seeds, size=(replicates, n_draws, n_seeds), dtype=np.int16)
    return HierarchicalIndices(draws, seeds, tuple(draw_order), tuple(int(v) for v in seed_order), int(seed))


def resample_block_rows(
    rows: Sequence[Mapping[str, Any]],
    indices: HierarchicalIndices,
    replicate: int,
) -> list[Mapping[str, Any]]:
    lookup = {(str(row["draw_id"]), int(row["seed"])): row for row in rows}
    selected: list[Mapping[str, Any]] = []
    for draw_occurrence, draw_index in enumerate(indices.draw_indices[replicate]):
        draw_id = indices.draw_order[int(draw_index)]
        for seed_index in indices.seed_indices[replicate, draw_occurrence]:
            seed = indices.seed_order[int(seed_index)]
            row = lookup.get((draw_id, seed))
            if row is not None:
                selected.append(row)
    return selected


def paired_percentile_interval(
    rows: Sequence[Mapping[str, Any]],
    estimator: Callable[[Sequence[Mapping[str, Any]]], float | None],
    *,
    seed: int = 860216,
) -> dict[str, Any]:
    point = estimator(rows)
    if point is None or not np.isfinite(point):
        return {"status": "NOT_ESTIMABLE"}
    indices = hierarchical_indices(seed=seed, draw_order=DRAWS, seed_order=WORKER_SEEDS)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        value = estimator(resample_block_rows(rows, indices, replicate))
        if value is None or not np.isfinite(value):
            return {"status": "NOT_ESTIMABLE", "reason": "NONFINITE_BOOTSTRAP_REPLICATE", "replicate_index": replicate}
        values[replicate] = value
    ordered = np.sort(values)
    lower = float(ordered[249])
    upper = float(ordered[9749])
    return {"status": "ESTIMABLE", "point": float(point), "lower_95": lower, "upper_95": upper, "replicates": BOOTSTRAP_REPLICATES, "seed": seed}


def simultaneous_max_bands(
    point_slacks: Mapping[str, float | None],
    bootstrap_slacks: Sequence[Mapping[str, float | None]],
    *,
    family_id: str,
    seed: int,
) -> dict[str, Any]:
    members = [member for member, value in point_slacks.items() if value is not None and np.isfinite(value)]
    omitted = [member for member, value in point_slacks.items() if value is None or not np.isfinite(value)]
    if not members:
        return {"family_id": family_id, "status": "NOT_ESTIMABLE", "omitted_members": omitted}
    if len(bootstrap_slacks) != BOOTSTRAP_REPLICATES:
        raise ContractError(f"{family_id} must contain exactly 10000 bootstrap replicates")
    deviations = np.empty((BOOTSTRAP_REPLICATES, len(members)), dtype=np.float64)
    for replicate, values in enumerate(bootstrap_slacks):
        for column, member in enumerate(members):
            value = values.get(member)
            if value is None or not np.isfinite(value):
                return {"family_id": family_id, "status": "NOT_ESTIMABLE", "reason": "NONFINITE_BOOTSTRAP_MEMBER", "member_id": member, "replicate_index": replicate, "omitted_members": omitted}
            deviations[replicate, column] = float(value) - float(point_slacks[member])
    point = np.asarray([float(point_slacks[member]) for member in members], dtype=np.float64)
    raw_scale = np.std(deviations, axis=0, ddof=1)
    floors = 1e-12 * np.maximum(1.0, np.abs(point))
    scales = np.maximum(raw_scale, floors)
    maxima = np.max(np.abs(deviations) / scales[None, :], axis=1)
    q = float(np.partition(maxima, 9500)[9500])
    bands: dict[str, Any] = {}
    for idx, member in enumerate(members):
        half = q * scales[idx]
        bands[member] = {
            "point": point[idx],
            "scale": float(scales[idx]),
            "lower": float(point[idx] - half),
            "upper": float(point[idx] + half),
            "pass_lower_strictly_positive": bool(point[idx] - half > 0.0),
        }
    return {
        "family_id": family_id,
        "status": "ESTIMABLE",
        "seed": seed,
        "replicates": BOOTSTRAP_REPLICATES,
        "order_statistic_one_based": 9501,
        "q": q,
        "members": bands,
        "omitted_members": omitted,
    }


def bootstrap_family(
    *,
    family_id: str,
    seed: int,
    draw_order: Sequence[str],
    seed_order: Sequence[int],
    point_estimator: Callable[[int | None, HierarchicalIndices | None], Mapping[str, float | None]],
) -> dict[str, Any]:
    point = dict(point_estimator(None, None))
    indices = hierarchical_indices(seed=seed, draw_order=draw_order, seed_order=seed_order)
    replicates: list[Mapping[str, float | None]] = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        replicates.append(dict(point_estimator(replicate, indices)))
    return simultaneous_max_bands(point, replicates, family_id=family_id, seed=seed)


SIGN_FLIP_SEEDS = {
    "T1": 860231,
    "T2": 860232,
    "T3": 860233,
    "T4": 860234,
    "T5": 860235,
    "T6": 860236,
    "T8": 860238,
}


def sign_flip_test(target: str, h_values: Sequence[float]) -> dict[str, Any]:
    if target not in SIGN_FLIP_SEEDS:
        raise ContractError(f"{target} does not use the sign-flip test")
    h = np.asarray(h_values, dtype=np.float64)
    if h.size < 30 or not np.all(np.isfinite(h)):
        return {"target_id": target, "status": "NOT_ESTIMABLE", "n_blocks": int(h.size)}
    observed = float(np.mean(h, dtype=np.float64))
    rng = np.random.Generator(np.random.PCG64(SIGN_FLIP_SEEDS[target]))
    bits = rng.integers(0, 2, size=(MONTE_CARLO_REPLICATES, h.size), dtype=np.int8)
    signs = 2 * bits - 1
    randomized = np.mean(signs * h[None, :], axis=1, dtype=np.float64)
    extreme = int(np.count_nonzero(randomized >= observed))
    p_value = (extreme + 1.0) / (MONTE_CARLO_REPLICATES + 1.0)
    return {
        "target_id": target,
        "status": "ESTIMABLE",
        "test": "ONE_SIDED_PAIRED_SIGN_FLIP",
        "seed": SIGN_FLIP_SEEDS[target],
        "n_blocks": int(h.size),
        "T_obs": observed,
        "random_rows": MONTE_CARLO_REPLICATES,
        "extreme_count_including_equal": extreme,
        "p_value": p_value,
    }


def h_values_for_target(target: str, rows: Sequence[Mapping[str, Any]]) -> list[float]:
    if target == "T1": return [float(row["E1"]) / 0.05 for row in rows]
    if target == "T2": return [float(row["E2"]) / 0.15 for row in rows]
    if target == "T3": return [float(row["E3"]) / 3.0 for row in rows]
    if target == "T4": return [float(row["E4"]) / 1.0 for row in rows]
    if target == "T5": return [float(row["E5"]) / 0.15 for row in rows]
    if target == "T6": return [float(row["E6"]) / 0.010 for row in rows]
    if target == "T8": return [float(row["E8"]) / (1.0 / 3.0) for row in rows]
    raise ContractError(f"no sign-flip h definition for {target}")


def _aggregate_t7(blocks: Sequence[Mapping[str, Any]], beta_key: str = "beta") -> float | None:
    rhos: list[float] = []; errors: list[float] = []; separations: list[float] = []
    for block in blocks:
        statistics = phase_response(block["alpha"], block[beta_key])
        if statistics is None: return None
        rhos.append(statistics["rho_c"]); errors.append(statistics["tracking_error"]); separations.append(statistics["separation"])
    if not blocks: return None
    aggregate = {"rho_c": float(np.median(rhos)), "tracking_error": float(np.median(errors)), "separation": float(np.median(separations))}
    return phase_score(aggregate)


def t7_circular_permutation(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(blocks) < 30:
        return {"target_id": "T7", "status": "NOT_ESTIMABLE", "n_blocks": len(blocks)}
    ordered = sorted(blocks, key=lambda row: (str(row["draw_id"]), int(row["seed"])))
    if len({int(row["seed"]) for row in ordered}) < 5 or len({str(row["draw_id"]) for row in ordered}) < 5:
        return {"target_id": "T7", "status": "NOT_ESTIMABLE", "reason": "SEED_OR_DRAW_DENOMINATOR"}
    observed = _aggregate_t7(ordered)
    if observed is None or not np.isfinite(observed):
        return {"target_id": "T7", "status": "NOT_ESTIMABLE", "reason": "OBSERVED_DEGENERACY"}
    rng = np.random.Generator(np.random.PCG64(860217))
    extreme = 0
    for _replicate in range(MONTE_CARLO_REPLICATES):
        permuted: list[dict[str, Any]] = []
        for block in ordered:
            permutation = rng.permutation(8)
            beta = np.asarray(block["beta"], dtype=np.float64)[permutation].tolist()
            copy = dict(block); copy["beta_permuted"] = beta; permuted.append(copy)
        value = _aggregate_t7(permuted, beta_key="beta_permuted")
        statistic = float("-inf") if value is None or not np.isfinite(value) else value
        if statistic >= observed:
            extreme += 1
    return {
        "target_id": "T7",
        "status": "ESTIMABLE",
        "test": "WITHIN_BLOCK_CIRCULAR_LABEL_PERMUTATION",
        "seed": 860217,
        "n_blocks": len(ordered),
        "T_obs": observed,
        "random_rows": MONTE_CARLO_REPLICATES,
        "extreme_count_including_equal": extreme,
        "p_value": (extreme + 1.0) / (MONTE_CARLO_REPLICATES + 1.0),
    }


def holm_adjust(
    raw_p: Mapping[str, float | None],
    family: Sequence[str],
    *,
    alpha: float = 0.05,
    gate_open: bool = True,
) -> dict[str, Any]:
    m = len(family)
    ordered = sorted(family, key=lambda target: (
        raw_p.get(target) is None or not np.isfinite(raw_p.get(target)),
        float("inf") if raw_p.get(target) is None else float(raw_p[target]),
        target,
    ))
    results: dict[str, Any] = {}
    prefix = 0.0
    for rank, target in enumerate(ordered, start=1):
        p_value = raw_p.get(target)
        if p_value is None or not np.isfinite(p_value):
            adjusted = None
            reject = False
        else:
            candidate = min(1.0, (m - rank + 1) * float(p_value))
            prefix = max(prefix, candidate)
            adjusted = prefix
            reject = bool(gate_open and adjusted < alpha)
        results[target] = {"raw_p": p_value, "adjusted_p": adjusted, "rank": rank, "reject": reject}
    return {"alpha": alpha, "gate_open": gate_open, "ordered_targets": ordered, "targets": results}


def apply_two_stage_holm(raw_p: Mapping[str, float | None], complete_primary_pass: Mapping[str, bool]) -> dict[str, Any]:
    primary = holm_adjust(raw_p, ("T1", "T4", "T6"), gate_open=True)
    primary_complete = any(
        primary["targets"][target]["reject"] and bool(complete_primary_pass.get(target))
        for target in ("T1", "T4", "T6")
    )
    secondary = holm_adjust(raw_p, ("T2", "T3", "T5", "T7"), gate_open=primary_complete)
    return {"primary": primary, "secondary": secondary, "secondary_gate_open": primary_complete, "T8": {"raw_p": raw_p.get("T8"), "descriptive_only": True}}


def enumerate_level_b_family(selected_targets: Sequence[str]) -> tuple[str, ...]:
    members: list[str] = []
    for target in selected_targets:
        cells = 20 if target in {"T6", "T7"} else 25
        columns = 5
        rows = cells // columns
        for rr in range(rows):
            for cc in range(columns):
                members.append(f"LB3/{target}/R{rr:02d}/C{cc:02d}/C_MINUS_1")
        if target == "T1":
            members.extend(f"LB3/T1/POLICY/A{cc:02d}/G_MINUS_1" for cc in range(5))
        if target == "T7":
            members.extend(f"LB3/T7/POLICY/FC{cc:02d}/G_MINUS_1" for cc in range(5))
    if len(members) > 80 or len(members) != len(set(members)):
        raise ContractError("F_LEVEL_B_SURFACES enumeration invalid")
    return tuple(members)


def enumerate_cross_target_family() -> tuple[str, ...]:
    members = tuple(
        f"X3/T{i}/T{j}/{kind}"
        for i in range(1, 9)
        for j in range(1, 9)
        for kind in ("BENEFIT", "ADVERSE")
    )
    if len(members) != 128 or len(set(members)) != 128:
        raise ContractError("F_CROSS_TARGET_8X8 enumeration invalid")
    return members


def phase2a_indices() -> HierarchicalIndices:
    return hierarchical_indices(seed=860218, draw_order=DRAWS, seed_order=WORKER_SEEDS)


def level_b_indices() -> HierarchicalIndices:
    return hierarchical_indices(seed=860220, draw_order=LEVEL_B_DRAWS, seed_order=LEVEL_B_SEEDS)


def verifier_indices() -> HierarchicalIndices:
    return hierarchical_indices(seed=860221, draw_order=DRAWS, seed_order=VERIFIER_SEEDS)

