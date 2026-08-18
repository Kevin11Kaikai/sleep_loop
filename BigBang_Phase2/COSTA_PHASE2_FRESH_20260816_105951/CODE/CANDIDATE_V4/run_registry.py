"""Mechanically enumerable Level-A, conditional Level-B, and verifier runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .contracts import ContractBundle, ContractError, FREEZE_HASH, object_sha256


WORKER_SEEDS = (11003, 22007, 33013, 44017, 55021, 66029)
LEVEL_B_SEEDS = (11003, 33013, 55021, 66029)
VERIFIER_SEEDS = (77041, 88069, 99079, 111091)
DRAWS = ("D00_NOMINAL", "D01", "D02", "D03", "D04", "D05")
LEVEL_B_DRAWS = ("D00_NOMINAL", "D02", "D04")
PHASES_A = (
    -3.141592653589793,
    -2.0943951023931953,
    -1.5707963267948966,
    -1.0471975511965976,
    0.0,
    1.0471975511965976,
    1.5707963267948966,
    2.0943951023931953,
)


@dataclass(frozen=True)
class Condition:
    stage: str
    target_id: str
    condition_id: str
    channel_id: str
    parameters: Mapping[str, Any]
    condition_index: int
    is_sham: bool = False

    def key(self) -> tuple[Any, ...]:
        return (
            self.stage,
            self.target_id,
            self.condition_id,
            self.channel_id,
            object_sha256(dict(self.parameters)),
        )


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    stage: str
    target_id: str
    condition_id: str
    channel_id: str
    parameters: Mapping[str, Any]
    parameter_draw_id: str
    seed: int
    paired_sham_run_id: str
    condition_index: int
    is_sham: bool
    registry_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "target_id": self.target_id,
            "condition_id": self.condition_id,
            "channel_id": self.channel_id,
            "parameters": dict(self.parameters),
            "parameter_draw_id": self.parameter_draw_id,
            "seed": self.seed,
            "paired_sham_run_id": self.paired_sham_run_id,
            "condition_index": self.condition_index,
            "is_sham": self.is_sham,
            "registry_index": self.registry_index,
            "freeze_sha256": FREEZE_HASH,
        }


@dataclass(frozen=True)
class RunRegistry:
    registry_id: str
    runs: tuple[RunSpec, ...]
    logical_reuse: tuple[Mapping[str, Any], ...] = ()
    inactive_conditions: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": "4.0.0",
            "registry_id": self.registry_id,
            "freeze_sha256": FREEZE_HASH,
            "run_count": len(self.runs),
            "runs": [run.as_dict() for run in self.runs],
            "logical_reuse": [dict(item) for item in self.logical_reuse],
            "inactive_conditions": [dict(item) for item in self.inactive_conditions],
        }
        body["registry_payload_sha256"] = object_sha256(body)
        return body


def _fmt_num(value: float) -> str:
    sign = "M" if value < 0 else "P"
    magnitude = f"{abs(float(value)):.8f}".rstrip("0").rstrip(".").replace(".", "P")
    return sign + magnitude


def _level_a_conditions() -> list[Condition]:
    """Return exactly 50 condition sets in frozen all-sham-then-target order."""
    conditions: list[Condition] = []
    # All seven explicit zero-forcing records precede intervention records.
    conditions.extend([
        Condition("LEVEL_A", "T1", "LEVEL_A_T1_SHAM", "SHAM", {}, 0, True),
        Condition("LEVEL_A", "T2", "LEVEL_A_T2_DOSE_INDEX_03", "GF_E_DC", {"d": 0.0}, 3, True),
        Condition("LEVEL_A", "T3", "LEVEL_A_T3_DOSE_INDEX_03", "GF_E_SO_PULSE", {"d": 0.0, "f_rep": 0.85}, 3, True),
        Condition("LEVEL_A", "T4", "LEVEL_A_SHARED_SHAM", "GF_TR_SIGMA_PACKET_BALANCE", {"d": 0.0, "f_c": 13.0, "f_rep": 0.85}, 3, True),
        Condition("LEVEL_A", "T5", "LEVEL_A_T5_DOSE_INDEX_03", "GF_TR_SIGMA_CONTINUOUS_BALANCE", {"d": 0.0, "f_c": 13.0}, 3, True),
        Condition("LEVEL_A", "T6_T7_SHARED", "LEVEL_A_PHASE_SHAM", "GF_T_PHASE_LOCKED_PACKET", {"phi": 0.0, "A": 0.0, "f_c": 13.0}, 0, True),
        Condition("LEVEL_A", "T8", "LEVEL_A_T8_DOSE_INDEX_03", "GF_BALANCED_DC", {"d": 0.0, "fraction": 1.0}, 3, True),
    ])
    for idx, f in enumerate((0.55, 0.70, 0.85, 1.00, 1.15)):
        conditions.append(Condition("LEVEL_A", "T1", f"LEVEL_A_T1_FREQUENCY_INDEX_{idx:02d}", "GF_E_SLOW_SINE", {"A": 0.8, "f": f}, idx))
    for target, channel, doses, fixed in (
        ("T2", "GF_E_DC", (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5), {}),
        ("T3", "GF_E_SO_PULSE", (-2.0, -1.25, -0.5, 0.0, 0.5, 1.25, 2.0), {"f_rep": 0.85}),
        ("T4", "GF_TR_SIGMA_PACKET_BALANCE", (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5), {"f_c": 13.0, "f_rep": 0.85}),
        ("T5", "GF_TR_SIGMA_CONTINUOUS_BALANCE", (-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2), {"f_c": 13.0}),
    ):
        for idx, dose in enumerate(doses):
            if dose == 0.0:
                continue
            params = {"d": dose, **fixed}
            conditions.append(Condition("LEVEL_A", target, f"LEVEL_A_{target}_DOSE_INDEX_{idx:02d}", channel, params, idx))
    for idx, phi in enumerate(PHASES_A):
        conditions.append(Condition("LEVEL_A", "T6_T7_SHARED", f"LEVEL_A_PHASE_INDEX_{idx:02d}", "GF_T_PHASE_LOCKED_PACKET", {"phi": phi, "A": 1.0, "f_c": 13.0}, idx))
    for idx, dose in enumerate((-2.0, -1.25, -0.5, 0.0, 0.5, 1.25, 2.0)):
        if dose == 0.0:
            continue
        conditions.append(Condition("LEVEL_A", "T8", f"LEVEL_A_T8_DOSE_INDEX_{idx:02d}", "GF_BALANCED_DC", {"d": dose, "fraction": 1.0}, idx))
    if len(conditions) != 50:
        raise ContractError(f"Level-A condition count is {len(conditions)}, expected 50")
    if len({c.key() for c in conditions}) != 50:
        raise ContractError("Level-A condition keys are not unique")
    return conditions


def _la_run_id(condition: Condition, draw: str, seed: int) -> str:
    if condition.condition_id == "LEVEL_A_SHARED_SHAM":
        return f"LEVEL_A_SHARED_SHAM::{draw}::S{seed}"
    if condition.is_sham:
        return f"LA__{condition.target_id}__SHAM__{draw}__S{seed}"
    return f"LA__{condition.target_id}__C{condition.condition_index:02d}__{draw}__S{seed}"


def _la_sham_id(condition: Condition, draw: str, seed: int) -> str:
    by_target = {
        "T1": f"LA__T1__SHAM__{draw}__S{seed}",
        "T2": f"LA__T2__SHAM__{draw}__S{seed}",
        "T3": f"LA__T3__SHAM__{draw}__S{seed}",
        "T4": f"LEVEL_A_SHARED_SHAM::{draw}::S{seed}",
        "T5": f"LA__T5__SHAM__{draw}__S{seed}",
        "T6_T7_SHARED": f"LA__T6_T7_SHARED__SHAM__{draw}__S{seed}",
        "T8": f"LA__T8__SHAM__{draw}__S{seed}",
    }
    return by_target[condition.target_id]


def build_level_a_registry(bundle: ContractBundle) -> RunRegistry:
    expected = bundle.protocol["phase2a"]["unique_simulation_count"]
    conditions = _level_a_conditions()
    runs: list[RunSpec] = []
    for condition in conditions:
        for draw in DRAWS:
            for seed in WORKER_SEEDS:
                run_id = _la_run_id(condition, draw, seed)
                sham_id = _la_sham_id(condition, draw, seed)
                runs.append(RunSpec(
                    run_id=run_id,
                    stage="LEVEL_A_WORKER",
                    target_id=condition.target_id,
                    condition_id=condition.condition_id,
                    channel_id=condition.channel_id,
                    parameters=condition.parameters,
                    parameter_draw_id=draw,
                    seed=seed,
                    paired_sham_run_id=sham_id,
                    condition_index=condition.condition_index,
                    is_sham=condition.is_sham,
                    registry_index=len(runs),
                ))
    if len(runs) != 1800 or len(runs) != expected["maximum_level_a_runs"]:
        raise ContractError(f"Level-A registry has {len(runs)} runs, expected 1800")
    if len({r.run_id for r in runs}) != len(runs):
        raise ContractError("duplicate Level-A run ID")
    return RunRegistry("COSTA_V4_LEVEL_A_1800", tuple(runs))


def _level_b_channel(target: str) -> str:
    return {
        "T1": "GF_E_SLOW_SINE",
        "T2": "GF_E_DC_PLUS_SLOW_SINE",
        "T3": "GF_E_SO_PULSE",
        "T4": "GF_TR_SIGMA_PACKET_BALANCE",
        "T5": "GF_TR_SIGMA_CONTINUOUS_BALANCE",
        "T6": "GF_T_PHASE_LOCKED_PACKET",
        "T7": "GF_T_PHASE_LOCKED_PACKET",
        "T8": "GF_BALANCED_DC",
    }[target]


def _level_b_params(target: str, row_value: float, column_value: float) -> dict[str, float]:
    if target == "T1": return {"f": row_value, "A": column_value}
    if target == "T2": return {"d": row_value, "A": column_value, "f": 0.85}
    if target == "T3": return {"d": row_value, "f_rep": column_value}
    if target == "T4": return {"d": row_value, "f_c": column_value, "f_rep": 0.85}
    if target == "T5": return {"d": row_value, "f_c": column_value}
    if target == "T6": return {"phi": row_value, "A": column_value, "f_c": 13.0}
    if target == "T7": return {"phi": row_value, "A": 1.0, "f_c": column_value}
    if target == "T8": return {"d": row_value, "fraction": column_value}
    raise ContractError(f"unknown Level-B target {target}")


def build_level_b_registry(bundle: ContractBundle, selected_targets: Sequence[str]) -> RunRegistry:
    selected = tuple(selected_targets)
    if len(selected) > 3 or len(set(selected)) != len(selected):
        raise ContractError("Level-B requires at most three distinct selected targets")
    if any(t not in {f"T{i}" for i in range(1, 9)} for t in selected):
        raise ContractError("invalid Level-B target")
    grids = bundle.protocol["phase2a"]["conditional_level_b"]["grids"]
    runs: list[RunSpec] = []
    reuse = tuple({
        "logical_run_id": f"LEVEL_B_SHARED_SHAM::{draw}::S{seed}",
        "reused_run_id": f"LEVEL_A_SHARED_SHAM::{draw}::S{seed}",
        "reason": "BYTE_IDENTICAL_LEVEL_A_ZERO_FORCING_SHAM",
    } for draw in LEVEL_B_DRAWS for seed in LEVEL_B_SEEDS)
    for target in selected:
        grid = grids[target]
        rows = tuple(grid["rows"])
        columns = tuple(grid["columns"])
        if len(rows) * len(columns) != grid["cells"]:
            raise ContractError(f"Level-B grid arithmetic failed for {target}")
        for rr, row_value in enumerate(rows):
            for cc, column_value in enumerate(columns):
                condition_id = f"LB3_{target}_R{rr:02d}_C{cc:02d}"
                params = _level_b_params(target, float(row_value), float(column_value))
                for draw in LEVEL_B_DRAWS:
                    for seed in LEVEL_B_SEEDS:
                        run_id = f"LB3__{target}__R{rr:02d}__C{cc:02d}__{draw}__S{seed}"
                        sham_id = f"LEVEL_B_SHARED_SHAM::{draw}::S{seed}"
                        runs.append(RunSpec(
                            run_id=run_id,
                            stage="LEVEL_B_SURFACE",
                            target_id=target,
                            condition_id=condition_id,
                            channel_id=_level_b_channel(target),
                            parameters=params,
                            parameter_draw_id=draw,
                            seed=seed,
                            paired_sham_run_id=sham_id,
                            condition_index=rr * len(columns) + cc,
                            # Every enumerated cell is an intervention identity,
                            # including exact-zero cells; its paired sham is the
                            # separately reused Level-A shared sham.
                            is_sham=False,
                            registry_index=len(runs),
                        ))
    n25 = sum(1 for t in selected if t in {"T1", "T2", "T3", "T4", "T5", "T8"})
    n20 = sum(1 for t in selected if t in {"T6", "T7"})
    expected = 12 * (25 * n25 + 20 * n20)
    if len(runs) != expected or len(runs) > 900:
        raise ContractError(f"Level-B registry arithmetic mismatch: {len(runs)} versus {expected}")
    if len({r.run_id for r in runs}) != len(runs):
        raise ContractError("duplicate Level-B run ID")
    return RunRegistry("COSTA_V4_LEVEL_B_CONDITIONAL", tuple(runs), reuse)


def _policy_conditions(policy: Mapping[str, Any], target: str) -> list[Mapping[str, Any]]:
    conditions = policy.get("conditions")
    if not isinstance(conditions, list):
        raise ContractError(f"selected policy {target} lacks conditions list")
    expected = 5 if target == "T1" else 8 if target == "T7" else 1
    if len(conditions) != expected:
        raise ContractError(f"policy {target} has {len(conditions)} conditions, expected {expected}")
    return conditions


def build_cross_target_registry(bundle: ContractBundle, policies: Mapping[str, Mapping[str, Any]]) -> RunRegistry:
    if tuple(sorted(policies)) != tuple(f"T{i}" for i in range(1, 9)):
        raise ContractError("verifier requires frozen policies for T1-T8")
    logical: dict[str, list[Mapping[str, Any]]] = {t: _policy_conditions(policies[t], t) for t in sorted(policies)}
    t6 = logical["T6"][0]
    matching_t7 = [idx for idx, c in enumerate(logical["T7"]) if c.get("parameters") == t6.get("parameters")]
    if len(matching_t7) != 1:
        raise ContractError("selected T6 tuple must match exactly one T7 mapping tuple")
    t6_reuse_index = matching_t7[0]
    runs: list[RunSpec] = []
    reuse: list[Mapping[str, Any]] = []
    inactive: list[Mapping[str, Any]] = []

    selected_t4_row = policies["T4"].get("attribution_domain_row_id")
    possible_t4 = tuple(bundle.protocol["observation_and_detector_contract"]["T4_added_event_attribution"]["candidate_domain_rows"]["cross_target_verifier_possible"])
    if selected_t4_row not in possible_t4:
        raise ContractError("selected T4 verifier domain row is not one of the three frozen rows")
    for row_id in possible_t4:
        if row_id != selected_t4_row:
            inactive.append({"domain_row_id": row_id, "status": "NOT_LAUNCHED_BY_POLICY"})

    for draw in DRAWS:
        for seed in VERIFIER_SEEDS:
            sham = f"X3__SHAM__{draw}__S{seed}"
            runs.append(RunSpec(sham, "CROSS_TARGET_VERIFIER", "SHAM", "X3_SHARED_SHAM", "SHAM", {}, draw, seed, sham, 0, True, len(runs)))
            for target in ("T1", "T2", "T3", "T4", "T5"):
                for kk, condition in enumerate(logical[target]):
                    params = dict(condition["parameters"])
                    run_id = f"X3__{target}__K{kk:02d}__{draw}__S{seed}"
                    runs.append(RunSpec(run_id, "CROSS_TARGET_VERIFIER", target, str(condition["condition_id"]), str(condition["channel_id"]), params, draw, seed, sham, kk, False, len(runs)))
            # T6 is a logical pointer, not a nineteenth intervention trajectory.
            t7_run = f"X3__T7__K{t6_reuse_index:02d}__{draw}__S{seed}"
            reuse.append({
                "logical_run_id": f"X3__T6__K00__{draw}__S{seed}",
                "reused_run_id": t7_run,
                "reason": "BYTE_IDENTICAL_T6_T7_PHASE_TUPLE",
            })
            for kk, condition in enumerate(logical["T7"]):
                params = dict(condition["parameters"])
                run_id = f"X3__T7__K{kk:02d}__{draw}__S{seed}"
                runs.append(RunSpec(run_id, "CROSS_TARGET_VERIFIER", "T7", str(condition["condition_id"]), str(condition["channel_id"]), params, draw, seed, sham, kk, False, len(runs)))
            condition = logical["T8"][0]
            runs.append(RunSpec(f"X3__T8__K00__{draw}__S{seed}", "CROSS_TARGET_VERIFIER", "T8", str(condition["condition_id"]), str(condition["channel_id"]), dict(condition["parameters"]), draw, seed, sham, 0, False, len(runs)))
    if len(runs) != 456:
        raise ContractError(f"cross-target registry has {len(runs)} runs, expected 456")
    if len(reuse) != 24:
        raise ContractError("cross-target T6 reuse count must be 24")
    if len({r.run_id for r in runs}) != 456:
        raise ContractError("duplicate cross-target run ID")
    return RunRegistry("COSTA_V4_CROSS_TARGET_456", tuple(runs), tuple(reuse), tuple(inactive))


def select_shard(runs: Sequence[RunSpec], shard_index: int, shard_count: int) -> tuple[RunSpec, ...]:
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ContractError("invalid shard index/count")
    draw_orders = {
        "LEVEL_A_WORKER": DRAWS,
        "LEVEL_B_SURFACE": LEVEL_B_DRAWS,
        "CROSS_TARGET_VERIFIER": DRAWS,
    }
    seed_orders = {
        "LEVEL_A_WORKER": WORKER_SEEDS,
        "LEVEL_B_SURFACE": LEVEL_B_SEEDS,
        "CROSS_TARGET_VERIFIER": VERIFIER_SEEDS,
    }
    def block_number(run: RunSpec) -> int:
        draws = draw_orders[run.stage]
        seeds = seed_orders[run.stage]
        return draws.index(run.parameter_draw_id) * len(seeds) + seeds.index(run.seed)
    return tuple(run for run in runs if block_number(run) % shard_count == shard_index)
