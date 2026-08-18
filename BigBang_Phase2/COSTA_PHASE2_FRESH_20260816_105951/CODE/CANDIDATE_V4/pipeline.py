"""Checkpointed workers and deterministic artifact generators for every CLI mode."""

from __future__ import annotations

from hashlib import sha256
import json
from math import pi
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .analysis import (
    TARGETS,
    build_level_a_curves,
    cell_score,
    denominator_status,
    estimate_target_slacks,
    phase_response,
    phase2b_disposition,
    phase_score,
    select_conditional_level_b_targets,
    select_level_a_policy,
    select_level_b_policy,
    shape_consistency,
    target_block_rows,
)
from .contracts import (
    ContractBundle,
    ContractError,
    FREEZE_HASH,
    PROTOCOL_AUDIT_HASH,
    assert_plain_descendant,
    atomic_write_json,
    load_json,
    object_sha256,
    sha256_file,
    write_checkpoint,
)
from .cross_target import classify_cross_target_matrix, compute_cross_target_matrix, cross_target_hard_vetoes
from .detectors import SpindleThresholds, analyze_simulation, broadband_specificity_veto, upward_zero_crossings
from .model import generate_parameter_draws, simulate
from .run_registry import (
    DRAWS,
    LEVEL_B_DRAWS,
    LEVEL_B_SEEDS,
    VERIFIER_SEEDS,
    WORKER_SEEDS,
    RunRegistry,
    RunSpec,
    build_cross_target_registry,
    build_level_a_registry,
    build_level_b_registry,
    select_shard,
)
from .statistics import (
    BOOTSTRAP_REPLICATES,
    apply_two_stage_holm,
    enumerate_level_b_family,
    h_values_for_target,
    hierarchical_indices,
    holm_adjust,
    resample_block_rows,
    sign_flip_test,
    simultaneous_max_bands,
    t7_circular_permutation,
    verifier_indices,
)
from .t4_attribution import classify_block, pool_row, validate_domain


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, complex): return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    return value


def prepare_output_root(bundle: ContractBundle, requested: Path) -> Path:
    target = assert_plain_descendant(requested, bundle.root, must_exist=False)
    relative = target.relative_to(bundle.root)
    if not relative.parts or relative.parts[0].upper() in {"CODE", "PREFLIGHT", "GOVERNANCE", "PROTOCOL", "STATE", "COMMON"}:
        raise ContractError("runtime output root must be a dedicated campaign descendant outside protected namespaces")
    target.mkdir(parents=True, exist_ok=True)
    if not target.is_dir(): raise ContractError("output root is not a directory")
    return target.resolve(strict=True)


def _record_path(output_root: Path, run_id: str) -> Path:
    digest = sha256(run_id.encode("utf-8")).hexdigest().upper()
    return output_root / "runs" / f"{digest}.json"


def _output_tree_gib(output_root: Path) -> float:
    """Return campaign-output bytes only; unrelated volume usage is out of scope."""
    total = 0
    for path in output_root.rglob("*"):
        if path.is_symlink():
            raise ContractError(f"symlink/reparse point forbidden in output tree: {path}")
        if path.is_file():
            total += path.stat().st_size
    return total / (1024.0 ** 3)


def _load_record(output_root: Path, run_id: str) -> Mapping[str, Any]:
    path = _record_path(output_root, run_id)
    if not path.is_file(): raise ContractError(f"required run record absent: {run_id}")
    body = load_json(path)
    if body.get("run_id") != run_id: raise ContractError(f"run-record identity mismatch for {run_id}")
    if body.get("freeze_sha256") != FREEZE_HASH or body.get("protocol_audit_sha256") != PROTOCOL_AUDIT_HASH:
        raise ContractError(f"run-record protocol binding mismatch for {run_id}")
    claimed = body.get("record_payload_sha256")
    payload = {key: value for key, value in body.items() if key != "record_payload_sha256"}
    if claimed != object_sha256(payload):
        raise ContractError(f"run-record payload hash mismatch for {run_id}")
    return body


def write_registry(registry: RunRegistry, output_root: Path) -> dict[str, Any]:
    body = registry.as_dict()
    path = output_root / "registries" / f"{registry.registry_id}.json"
    digest = atomic_write_json(path, body, output_root=output_root)
    return {"path": str(path), "sha256": digest, "run_count": len(registry.runs), "registry_payload_sha256": body["registry_payload_sha256"]}


def _resolve_alias(registry: RunRegistry, logical_run_id: str) -> str:
    matches = [str(row["reused_run_id"]) for row in registry.logical_reuse if row["logical_run_id"] == logical_run_id]
    if len(matches) > 1: raise ContractError(f"duplicate run alias {logical_run_id}")
    return matches[0] if matches else logical_run_id


def _thresholds(record: Mapping[str, Any]) -> SpindleThresholds:
    values = record["metrics"]["spindle_thresholds"]
    return SpindleThresholds(float(values["median"]), float(values["mad"]), float(values["high"]), float(values["low"]))


def _domain_context(stage: str) -> str:
    return {"LEVEL_A_WORKER": "LEVEL_A_WORKER", "LEVEL_B_SURFACE": "LEVEL_B_SURFACE", "CROSS_TARGET_VERIFIER": "CROSS_TARGET_VERIFIER"}[stage]


def _scientific_record(
    bundle: ContractBundle,
    registry: RunRegistry,
    run: RunSpec,
    output_root: Path,
    draws: Mapping[str, Any],
) -> dict[str, Any]:
    result = simulate(run, draws[run.parameter_draw_id])
    if run.is_sham:
        metric_result = analyze_simulation(result)
        sham_record = None
    else:
        actual_sham_id = _resolve_alias(registry, run.paired_sham_run_id)
        sham_record = _load_record(output_root, actual_sham_id)
        metric_result = analyze_simulation(result, sham_thresholds=_thresholds(sham_record), sham_metrics=sham_record["metrics"])
        if result.innovation_sha256 != sham_record["innovation_sha256"]:
            raise ContractError(f"OU pairing hash mismatch for {run.run_id}")
    artifacts = metric_result.artifacts
    crossings = upward_zero_crossings(artifacts.sigma_signal).tolist()
    t4 = None
    if (
        run.target_id == "T4"
        and not run.is_sham
        and float(run.parameters.get("d", 0.0)) > 0.0
        and run.channel_id == "GF_TR_SIGMA_PACKET_BALANCE"
    ):
        if sham_record is None: raise ContractError("T4 intervention lacks sham")
        domain = validate_domain(bundle)
        row = domain.resolve(_domain_context(run.stage), float(run.parameters["d"]), float(run.parameters["f_c"]), float(run.parameters.get("f_rep", 0.85)))
        classification = classify_block(
            sham_record["detector_artifacts"]["spindle_events"],
            [event.as_dict() for event in artifacts.spindle_events],
            result.packet_table,
            crossings,
        )
        t4 = {
            "domain_row_id": row["domain_row_id"],
            "execution_context": row["execution_context"],
            "valid_complete_pair": bool(
                metric_result.validity["stable"]
                and sham_record["validity"]["stable"]
                and metric_result.validity["T4_valid"]
                and sham_record["validity"]["T4_valid"]
            ),
            "draw_id": run.parameter_draw_id,
            "parameter_draw_id": run.parameter_draw_id,
            "seed": run.seed,
            "block_id": f"{run.parameter_draw_id}::S{run.seed}",
            "classification": classification,
        }
    record = {
        **run.as_dict(),
        "schema_version": "4.0.0",
        "artifact_type": "RUN_RESULT",
        "protocol_audit_sha256": PROTOCOL_AUDIT_HASH,
        "parameter_draw_sha256": result.parameter_draw.payload_sha256,
        "innovation_sha256": result.innovation_sha256,
        "state_sha256": result.state_sha256,
        "metrics": metric_result.metrics,
        "validity": metric_result.validity,
        "detector_artifacts": {
            "so_events": [event.as_dict() for event in artifacts.so_events],
            "spindle_events": [event.as_dict() for event in artifacts.spindle_events],
            "sigma_upward_crossings": crossings,
        },
        "scheduler_log": result.scheduler_log,
        "packet_table": result.packet_table,
        "t4_attribution": t4,
        "execution_claim": "RUN_EXECUTED_ONLY_WHEN_SEPARATELY_AUTHORIZED_BY_CLI_RECEIPT",
    }
    record["record_payload_sha256"] = object_sha256(_json_safe(record))
    return _json_safe(record)


def execute_registry_shard(
    bundle: ContractBundle,
    registry: RunRegistry,
    output_root: Path,
    *,
    shard_index: int,
    shard_count: int,
    cpu_hour_cap: float,
) -> dict[str, Any]:
    selected = select_shard(registry.runs, shard_index, shard_count)
    draws = generate_parameter_draws(bundle)
    start_cpu = time.process_time()
    entries: list[dict[str, Any]] = []
    block_requirements: dict[tuple[str, str, str, int], set[str]] = {}
    for item in selected:
        key = (item.stage, item.target_id, item.parameter_draw_id, item.seed)
        block_requirements.setdefault(key, set()).add(item.run_id)
    completed_ids: set[str] = set()
    checkpoint_entries: list[dict[str, Any]] = []
    checkpointed: set[tuple[str, str, str, int]] = set()

    def checkpoint_complete_blocks() -> None:
        for key, required_ids in block_requirements.items():
            if key in checkpointed or not required_ids.issubset(completed_ids):
                continue
            stage, target, draw, seed = key
            files = []
            for run_id in sorted(required_ids):
                run_path = _record_path(output_root, run_id)
                files.append({"run_id": run_id, "path": str(run_path), "sha256": sha256_file(run_path)})
            safe_target = target.replace("/", "_")
            name = f"BLOCK__{stage}__{safe_target}__{draw}__S{seed}.json"
            checkpoint_path = output_root / "checkpoints" / "blocks" / name
            payload = {
                "schema_version": "4.0.0",
                "artifact_type": "IMMUTABLE_TARGET_SEED_DRAW_BLOCK",
                "registry_id": registry.registry_id,
                "stage": stage,
                "target_id": target,
                "parameter_draw_id": draw,
                "seed": seed,
                "run_count": len(files),
                "files": files,
            }
            digest = write_checkpoint(checkpoint_path, payload, output_root=output_root)
            checkpoint_entries.append({"path": str(checkpoint_path), "sha256": digest, "stage": stage, "target_id": target, "parameter_draw_id": draw, "seed": seed})
            checkpointed.add(key)
    for run in selected:
        elapsed_hours = (time.process_time() - start_cpu) / 3600.0
        if elapsed_hours >= cpu_hour_cap:
            raise ContractError(f"worker CPU cap reached before {run.run_id}")
        if _output_tree_gib(output_root) > 45.0:
            raise ContractError("storage hard stop above 45 GiB")
        path = _record_path(output_root, run.run_id)
        if path.exists():
            existing = _load_record(output_root, run.run_id)
            if existing.get("record_payload_sha256") is None:
                raise ContractError(f"existing run record lacks payload hash: {run.run_id}")
            digest = sha256_file(path)
            entries.append({"run_id": run.run_id, "path": str(path), "sha256": digest, "status": "RESUMED_HASH_VERIFIED"})
            completed_ids.add(run.run_id)
            checkpoint_complete_blocks()
            continue
        if not run.is_sham:
            sham_id = _resolve_alias(registry, run.paired_sham_run_id)
            if not _record_path(output_root, sham_id).exists():
                raise ContractError(f"paired sham must complete before intervention: {sham_id}")
        record = _scientific_record(bundle, registry, run, output_root, draws)
        digest = atomic_write_json(path, record, output_root=output_root)
        entries.append({"run_id": run.run_id, "path": str(path), "sha256": digest, "status": "COMPLETE"})
        completed_ids.add(run.run_id)
        checkpoint_complete_blocks()
    manifest = {
        "schema_version": "4.0.0",
        "artifact_type": "WORKER_SHARD_MANIFEST",
        "registry_id": registry.registry_id,
        "registry_payload_sha256": registry.as_dict()["registry_payload_sha256"],
        "freeze_sha256": FREEZE_HASH,
        "protocol_audit_sha256": PROTOCOL_AUDIT_HASH,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_run_count": len(selected),
        "completed_run_count": len(entries),
        "files": entries,
        "immutable_block_checkpoints": checkpoint_entries,
        "cpu_hours_this_process": (time.process_time() - start_cpu) / 3600.0,
    }
    name = f"{registry.registry_id}__SHARD_{shard_index:03d}_OF_{shard_count:03d}.json"
    path = output_root / "checkpoints" / name
    digest = atomic_write_json(path, _json_safe(manifest), output_root=output_root)
    return {"path": str(path), "sha256": digest, **manifest}


def load_shard_records(output_root: Path, manifest_paths: Sequence[Path]) -> list[Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for manifest_path in manifest_paths:
        manifest_path = assert_plain_descendant(manifest_path, output_root, must_exist=True)
        manifest = load_json(manifest_path)
        if manifest.get("freeze_sha256") != FREEZE_HASH or manifest.get("protocol_audit_sha256") != PROTOCOL_AUDIT_HASH:
            raise ContractError(f"manifest binding mismatch: {manifest_path}")
        for item in manifest["files"]:
            path = assert_plain_descendant(Path(item["path"]), output_root, must_exist=True)
            if sha256_file(path) != item["sha256"]:
                raise ContractError(f"run file hash mismatch: {path}")
            record = load_json(path)
            if record["run_id"] != item["run_id"]:
                raise ContractError(f"manifest/run identity mismatch: {path}")
            if record["run_id"] in records and records[record["run_id"]] != record:
                raise ContractError(f"conflicting duplicate run record: {record['run_id']}")
            records[record["run_id"]] = record
    return list(records.values())


def _paired_candidates(records: Sequence[Mapping[str, Any]], target: str, stage: str) -> tuple[list[Mapping[str, Any]], dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]]]:
    group = "T6_T7_SHARED" if target in {"T6", "T7"} and stage == "LEVEL_A_WORKER" else target
    lookup = {r["run_id"]: r for r in records}
    active = [r for r in records if r["stage"] == stage and r["target_id"] == group and not r["is_sham"]]
    exemplars: dict[str, Mapping[str, Any]] = {}
    pairs: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for record in active:
        sham_id = record["paired_sham_run_id"]
        if sham_id.startswith("LEVEL_B_SHARED_SHAM"):
            suffix = sham_id[len("LEVEL_B_SHARED_SHAM") :]
            sham_id = "LEVEL_A_SHARED_SHAM" + suffix
        if sham_id not in lookup: continue
        if not (
            record["validity"].get("stable")
            and lookup[sham_id]["validity"].get("stable")
            and record["validity"].get(f"{target}_valid")
            and lookup[sham_id]["validity"].get(f"{target}_valid")
        ): continue
        exemplars.setdefault(record["condition_id"], record)
        pairs.setdefault(record["condition_id"], []).append((record, lookup[sham_id]))
    return list(exemplars.values()), pairs


def _attribution_pools(bundle: ContractBundle, records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, list[Mapping[str, Any]]]]:
    domain = validate_domain(bundle)
    blocks: dict[str, list[Mapping[str, Any]]] = {row["domain_row_id"]: [] for row in domain.rows}
    for record in records:
        value = record.get("t4_attribution")
        if value is not None:
            blocks[value["domain_row_id"]].append(value)
    pools = {row_id: pool_row(domain.by_id[row_id], values) for row_id, values in blocks.items() if values}
    return pools, blocks


def _level_a_t8_eligible(candidates: Sequence[Mapping[str, Any]], pairs: Mapping[str, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]]) -> set[str]:
    by_dose = {float(record["parameters"]["d"]): record for record in candidates}
    q: dict[float, float] = {}; persistence: dict[float, float] = {}
    for dose, record in by_dose.items():
        values = [np.sign(dose) * (float(a["metrics"]["T8_occupancy"]) - float(s["metrics"]["T8_occupancy"])) for a, s in pairs[record["condition_id"]]]
        q[dose] = float(np.median(values)); persistence[dose] = float(np.mean([bool(a["metrics"]["T8_persistent"]) for a, _ in pairs[record["condition_id"]]]))
    eligible: set[str] = set()
    for left, right in ((-2,-1.25),(-1.25,-0.5),(0.5,1.25),(1.25,2)):
        if left in q and right in q and q[left] > 0.0 and q[right] > 0.0:
            for dose in (left, right):
                if persistence[dose] >= 0.75: eligible.add(by_dose[dose]["condition_id"])
    return eligible


def _level_a_hard_vetoes(records: Sequence[Mapping[str, Any]], target: str) -> list[str]:
    group = "T6_T7_SHARED" if target in {"T6", "T7"} else target
    lookup = {record["run_id"]: record for record in records}
    active_records = [record for record in records if record["stage"] == "LEVEL_A_WORKER" and record["target_id"] == group and not record["is_sham"]]
    reasons: set[str] = set()
    for active in active_records:
        sham = lookup.get(active["paired_sham_run_id"])
        if sham is None:
            reasons.add("MISSING_PAIRED_SHAM_VETO")
            continue
        if active["metrics"].get("spindle_thresholds") != sham["metrics"].get("spindle_thresholds"):
            reasons.add("DETECTOR_THRESHOLD_CHANGE_VETO")
        if broadband_specificity_veto(active["metrics"], sham["metrics"]):
            reasons.add("BROADBAND_WITHOUT_TARGET_FRACTION_INCREASE_VETO")
        if not bool(active["validity"].get("T8_valid")):
            reasons.add("T8_REGIME_INVALIDITY_VETO")
    return sorted(reasons)


def analyze_level_a(bundle: ContractBundle, records: Sequence[Mapping[str, Any]], output_root: Path) -> dict[str, Any]:
    expected_registry = build_level_a_registry(bundle)
    expected_ids = {run.run_id for run in expected_registry.runs}
    observed_ids = {record["run_id"] for record in records if record["stage"] == "LEVEL_A_WORKER"}
    if observed_ids != expected_ids:
        raise ContractError(f"Level-A result identity set mismatch: missing={len(expected_ids-observed_ids)}, extra={len(observed_ids-expected_ids)}")
    pools, attr_blocks = _attribution_pools(bundle, records)
    domain = validate_domain(bundle)
    target_rows: dict[str, list[dict[str, Any]]] = {}
    point_slacks: dict[str, float | None] = {}
    shapes: dict[str, Any] = {}; denoms: dict[str, Any] = {}; policies: dict[str, Any] = {}; stability: dict[str, float] = {}; hard_vetoes: dict[str, list[str]] = {}
    for target in TARGETS:
        curves = build_level_a_curves(records, target)
        rows = target_block_rows(target, curves); target_rows[target] = rows
        denoms[target] = denominator_status(rows, minimum_blocks=30, minimum_seeds=5, minimum_draws=5)
        attr = pools.get("T4_LA_WORKER_D1P50_FC13") if target == "T4" else None
        slacks = estimate_target_slacks(target, rows, formal_t4_attribution=attr)
        members = bundle.protocol["uncertainty_and_statistics"]["simultaneous_families"]["F_TARGET_DECISIONS"]["family_members"][target]
        for member in members: point_slacks[f"{target}/{member}"] = None if slacks is None else slacks.get(member)
        shapes[target] = shape_consistency(target, curves)
        group = "T6_T7_SHARED" if target in {"T6", "T7"} else target
        target_records = [r for r in records if r["stage"] == "LEVEL_A_WORKER" and r["target_id"] == group]
        stability[target] = float(np.mean([bool(r["validity"]["stable"]) for r in target_records]))
        hard_vetoes[target] = _level_a_hard_vetoes(records, target)

        candidates, pairs = _paired_candidates(records, target, "LEVEL_A_WORKER")
        scores: dict[str, float | None] = {}
        for candidate in candidates:
            block_pairs = pairs[candidate["condition_id"]]
            denom = denominator_status([{"draw_id": a["parameter_draw_id"], "seed": a["seed"]} for a, _ in block_pairs], minimum_blocks=30, minimum_seeds=5, minimum_draws=5)
            attribution = None
            if target == "T4" and float(candidate["parameters"]["d"]) > 0:
                row = domain.resolve("LEVEL_A_WORKER", float(candidate["parameters"]["d"]), 13.0, 0.85); attribution = pools.get(row["domain_row_id"])
            scores[candidate["condition_id"]] = None if denom["status"] != "ESTIMABLE" else cell_score(target, candidate, block_pairs, attribution=attribution)
        t8_eligible = _level_a_t8_eligible(candidates, pairs) if target == "T8" else None
        verifier_by_condition = None
        if target == "T4":
            verifier_by_condition = {}
            for candidate in candidates:
                if float(candidate["parameters"].get("d", 0.0)) > 0.0:
                    xv = domain.resolve("CROSS_TARGET_VERIFIER", float(candidate["parameters"]["d"]), 13.0, 0.85)
                    verifier_by_condition[candidate["condition_id"]] = xv["domain_row_id"]
        policy = select_level_a_policy(target, candidates, scores, t8_eligible_condition_ids=t8_eligible, t4_domain_by_condition=verifier_by_condition)
        if target == "T7" and rows:
            aggregate = {key: float(np.median([row[key] for row in rows])) for key in ("rho_c","tracking_error","separation")}
            policy["score"] = phase_score(aggregate)
        policies[target] = policy

    indices = hierarchical_indices(seed=860218, draw_order=DRAWS, seed_order=WORKER_SEEDS)
    bootstrap_values: list[dict[str, float | None]] = []
    formal_row = domain.by_id["T4_LA_WORKER_D1P50_FC13"]
    for replicate in range(BOOTSTRAP_REPLICATES):
        values: dict[str, float | None] = {}
        for target in TARGETS:
            sampled = resample_block_rows(target_rows[target], indices, replicate)
            attribution = None
            if target == "T4":
                sampled_attr = resample_block_rows(attr_blocks.get("T4_LA_WORKER_D1P50_FC13", []), indices, replicate)
                attribution = pool_row(formal_row, sampled_attr, bootstrap=True, family_id="F_TARGET_DECISIONS", replicate_index=replicate)
            slacks = estimate_target_slacks(target, sampled, formal_t4_attribution=attribution)
            members = bundle.protocol["uncertainty_and_statistics"]["simultaneous_families"]["F_TARGET_DECISIONS"]["family_members"][target]
            for member in members: values[f"{target}/{member}"] = None if slacks is None else slacks.get(member)
        bootstrap_values.append(values)
    bands = simultaneous_max_bands(point_slacks, bootstrap_values, family_id="F_TARGET_DECISIONS", seed=860218)
    tests: dict[str, Any] = {}; raw_p: dict[str, float | None] = {}
    for target in ("T1","T2","T3","T4","T5","T6","T8"):
        tests[target] = sign_flip_test(target, h_values_for_target(target, target_rows[target])); raw_p[target] = tests[target].get("p_value")
    tests["T7"] = t7_circular_permutation(target_rows["T7"]); raw_p["T7"] = tests["T7"].get("p_value")
    pre_holm: dict[str, bool] = {}
    for target in TARGETS:
        member_ids = [key for key in bands.get("members", {}) if key.startswith(target + "/")]
        effect_pass = bool(member_ids) and all(bands["members"][key]["lower"] > 0.0 for key in member_ids)
        pre_holm[target] = bool(denoms[target]["status"] == "ESTIMABLE" and effect_pass and shapes[target]["pass"] and stability[target] >= 0.95 and not hard_vetoes[target])
    holm = apply_two_stage_holm(raw_p, pre_holm)
    summary: dict[str, Any] = {}
    for target in TARGETS:
        if target in {"T1","T4","T6"}: reject = holm["primary"]["targets"][target]["reject"]
        elif target in {"T2","T3","T5","T7"}: reject = holm["secondary"]["targets"][target]["reject"]
        else: reject = False
        target_band_values = [row["lower"] for key,row in bands.get("members",{}).items() if key.startswith(target+"/")]
        minimum_lcb = min(target_band_values) if target_band_values else None
        summary[target] = {
            "denominator": denoms[target], "shape_consistency": shapes[target], "stable_fraction": stability[target],
            "hard_vetoes": hard_vetoes[target],
            "effect_size_pass": pre_holm[target], "holm_reject": reject, "target_decision_pass": bool(pre_holm[target] and (reject or target=="T8")),
            "level_b_eligible": bool(pre_holm[target]), "normalized_minimum_lcb_slack": minimum_lcb,
            "selected_level_a_policy": policies[target], "grade": "PENDING_CROSS_TARGET_VERIFICATION" if pre_holm[target] else "D_OR_NOT_ESTIMABLE",
        }
    provisional_level_b_order = select_conditional_level_b_targets(summary)
    artifact = {
        "schema_version":"4.0.0","artifact_type":"LEVEL_A_ANALYSIS","freeze_sha256":FREEZE_HASH,"protocol_audit_sha256":PROTOCOL_AUDIT_HASH,
        "run_count":len(observed_ids),"target_summary":summary,"selected_level_a_policies":policies,"provisional_level_b_eligible_order_before_cross_target":provisional_level_b_order,
        "t4_attribution_pools":pools,"F_TARGET_DECISIONS":bands,"exact_tests":tests,"holm":holm,"phase2b":phase2b_disposition(bundle),
    }
    path=output_root/"analysis"/"LEVEL_A_ANALYSIS.json"; digest=atomic_write_json(path,_json_safe(artifact),output_root=output_root)
    policy_path=output_root/"analysis"/"LEVEL_A_FROZEN_POLICIES.json"; policy_digest=atomic_write_json(policy_path,_json_safe({"schema_version":"4.0.0","artifact_type":"LEVEL_A_FROZEN_POLICIES","freeze_sha256":FREEZE_HASH,"protocol_audit_sha256":PROTOCOL_AUDIT_HASH,"policy_source":"LEVEL_A_WORKER_ONLY_FROZEN_BEFORE_VERIFIER_AND_LEVEL_B","policies":policies}),output_root=output_root)
    return {"path":str(path),"sha256":digest,"policy_path":str(policy_path),"policy_sha256":policy_digest,**artifact}


def analyze_cross_target(
    bundle: ContractBundle,
    records: Sequence[Mapping[str, Any]],
    policies: Mapping[str, Mapping[str, Any]],
    level_a_analysis: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    if level_a_analysis.get("artifact_type") != "LEVEL_A_ANALYSIS" or level_a_analysis.get("freeze_sha256") != FREEZE_HASH or level_a_analysis.get("protocol_audit_sha256") != PROTOCOL_AUDIT_HASH:
        raise ContractError("cross-target finalization requires bound Level-A analysis")
    registry=build_cross_target_registry(bundle,policies); expected={r.run_id for r in registry.runs}; observed={r["run_id"] for r in records if r["stage"]=="CROSS_TARGET_VERIFIER"}
    if expected!=observed: raise ContractError("cross-target verifier identity set mismatch")
    cross_records=[record for record in records if record["stage"]=="CROSS_TARGET_VERIFIER"]
    pools,attr_blocks=_attribution_pools(bundle,cross_records); selected_row=policies["T4"].get("attribution_domain_row_id"); selected_attr=pools.get(selected_row)
    point=compute_cross_target_matrix(policies,cross_records,selected_t4_attribution=selected_attr)
    indices=verifier_indices(); domain=validate_domain(bundle); bootstrap=[]
    for replicate in range(BOOTSTRAP_REPLICATES):
        keys=[]
        for draw_occurrence,draw_index in enumerate(indices.draw_indices[replicate]):
            draw=indices.draw_order[int(draw_index)]
            for seed_index in indices.seed_indices[replicate,draw_occurrence]: keys.append((draw,indices.seed_order[int(seed_index)]))
        lookup={(b["parameter_draw_id"],int(b["seed"])):b for b in attr_blocks.get(selected_row,[])}
        sampled_attr=[lookup[key] for key in keys if key in lookup]
        attr=pool_row(domain.by_id[selected_row],sampled_attr,bootstrap=True,family_id="F_CROSS_TARGET_8X8",replicate_index=replicate)
        value=compute_cross_target_matrix(policies,cross_records,selected_t4_attribution=attr,block_keys=keys,bootstrap=True)
        bootstrap.append(value["family_members"])
    bands=simultaneous_max_bands(point["family_members"],bootstrap,family_id="F_CROSS_TARGET_8X8",seed=860221)
    vetoes,veto_reasons=cross_target_hard_vetoes(policies,cross_records,bands)
    classifications=classify_cross_target_matrix(point,bands,hard_vetoes=vetoes)
    finalized_summary:dict[str,Any]={};selection_input:dict[str,Any]={}
    for target in TARGETS:
        source=dict(level_a_analysis["target_summary"][target]);labels=classifications["classifications"][target]
        has_hard_veto=any(label=="UNACCEPTABLE_COLLATERAL_EFFECT" for label in labels.values())
        off_target_complete=all(label in {"BENEFICIAL_SECONDARY_EFFECT","NEGLIGIBLE"} for column,label in labels.items() if column!=target)
        diagonal_pass=labels[target]=="INTENDED_IMPROVEMENT"
        if has_hard_veto:grade="F"
        elif bool(source.get("target_decision_pass")) and diagonal_pass and off_target_complete:grade="C"
        else:grade="D"
        eligible=bool(source.get("level_b_eligible")) and not has_hard_veto
        finalized_summary[target]={**source,"cross_target_diagonal_label":labels[target],"cross_target_off_target_complete":off_target_complete,"cross_target_hard_veto":has_hard_veto,"final_grade":grade,"level_b_eligible_after_verifier":eligible}
        selection_input[target]={**source,"level_b_eligible":eligible}
    selected_targets=select_conditional_level_b_targets(selection_input)
    artifact={"schema_version":"4.0.0","artifact_type":"CROSS_TARGET_8X8_ANALYSIS","freeze_sha256":FREEZE_HASH,"protocol_audit_sha256":PROTOCOL_AUDIT_HASH,"unique_run_count":456,"point":point,"simultaneous_bands":bands,**classifications,"hard_veto_reasons":veto_reasons,"selected_t4_domain_row":selected_row,"inactive_t4_rows":list(registry.inactive_conditions),"finalized_target_summary":finalized_summary,"conditional_level_b_selected_targets":selected_targets,"grade_ceiling":"C"}
    path=output_root/"analysis"/"CROSS_TARGET_8X8_ANALYSIS.json"; digest=atomic_write_json(path,_json_safe(artifact),output_root=output_root)
    selection={"schema_version":"4.0.0","artifact_type":"CONDITIONAL_LEVEL_B_SELECTION","freeze_sha256":FREEZE_HASH,"protocol_audit_sha256":PROTOCOL_AUDIT_HASH,"source_artifact_type":"CROSS_TARGET_8X8_ANALYSIS","selection_cap":3,"selected_targets":selected_targets,"target_eligibility":{target:bool(selection_input[target]["level_b_eligible"]) for target in TARGETS},"level_b_independence":"DESCRIPTIVE_SURFACES_CANNOT_ALTER_LEVEL_A_POLICIES_OR_VERIFIER"}
    selection_path=output_root/"analysis"/"CONDITIONAL_LEVEL_B_SELECTION.json";selection_digest=atomic_write_json(selection_path,_json_safe(selection),output_root=output_root)
    return {"path":str(path),"sha256":digest,"selection_path":str(selection_path),"selection_sha256":selection_digest,**artifact}


def _resample_pairs(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    draw_indices: np.ndarray,
    seed_indices: np.ndarray,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    lookup={(str(active["parameter_draw_id"]),int(active["seed"])):(active,sham) for active,sham in pairs}
    selected=[]
    for occurrence,draw_index in enumerate(draw_indices):
        draw=LEVEL_B_DRAWS[int(draw_index)]
        for seed_index in seed_indices[occurrence]:
            key=(draw,LEVEL_B_SEEDS[int(seed_index)])
            if key in lookup: selected.append(lookup[key])
    return selected


def _t7_level_b_mapping_score(
    carrier: float,
    candidates: Sequence[Mapping[str, Any]],
    pairs: Mapping[str, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]],
    *,
    bootstrap: bool = False,
) -> float | None:
    phases=(-pi,-pi/2.0,0.0,pi/2.0)
    mapped=[]
    for phase in phases:
        matches=[record for record in candidates if float(record["parameters"]["f_c"])==carrier and float(record["parameters"]["phi"])==phase]
        if len(matches)!=1:return None
        mapped.append(matches[0])
    statistics=[]
    if bootstrap:
        block_lists=[pairs.get(record["condition_id"],[]) for record in mapped]
        lengths={len(items) for items in block_lists}
        if len(lengths)!=1:return None
        for occurrence in range(next(iter(lengths))):
            beta=[items[occurrence][0]["metrics"].get("T7_preferred_phase_radians") for items in block_lists]
            value=phase_response(phases,beta)
            if value is not None:statistics.append(value)
    else:
        by_block:dict[tuple[str,int],list[float]]={}
        for record in mapped:
            for active,_sham in pairs.get(record["condition_id"],[]):
                beta=active["metrics"].get("T7_preferred_phase_radians")
                if beta is not None:by_block.setdefault((active["parameter_draw_id"],int(active["seed"])),[]).append(float(beta))
        for key in sorted(by_block):
            if len(by_block[key])!=4:continue
            value=phase_response(phases,by_block[key])
            if value is not None:statistics.append(value)
    if len(statistics)<(1 if bootstrap else 10):return None
    aggregate={name:float(np.median([row[name] for row in statistics])) for name in ("rho_c","tracking_error","separation")}
    return phase_score(aggregate)


def _t7_level_b_bootstrap_score(
    carrier: float,
    candidates: Sequence[Mapping[str, Any]],
    pairs: Mapping[str, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]],
    draw_indices: np.ndarray,
    seed_indices: np.ndarray,
) -> float | None:
    phases=(-pi,-pi/2.0,0.0,pi/2.0)
    mapped=[]
    for phase in phases:
        matches=[record for record in candidates if float(record["parameters"]["f_c"])==carrier and float(record["parameters"]["phi"])==phase]
        if len(matches)!=1:return None
        mapped.append(matches[0])
    lookups=[]
    for record in mapped:
        lookups.append({(str(active["parameter_draw_id"]),int(active["seed"])):active for active,_sham in pairs.get(record["condition_id"],[])})
    statistics=[]
    for occurrence,draw_index in enumerate(draw_indices):
        draw=LEVEL_B_DRAWS[int(draw_index)]
        for seed_index in seed_indices[occurrence]:
            key=(draw,LEVEL_B_SEEDS[int(seed_index)])
            if not all(key in lookup for lookup in lookups):continue
            beta=[lookup[key]["metrics"].get("T7_preferred_phase_radians") for lookup in lookups]
            value=phase_response(phases,beta)
            if value is not None:statistics.append(value)
    if not statistics:return None
    aggregate={name:float(np.median([row[name] for row in statistics])) for name in ("rho_c","tracking_error","separation")}
    return phase_score(aggregate)


def _level_b_t8_eligible(
    candidates: Sequence[Mapping[str, Any]],
    pairs: Mapping[str, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]],
) -> set[str]:
    eligible:set[str]=set()
    for fraction in (0.2,0.4,0.6,0.8,1.0):
        q:dict[float,float]={};by_dose={float(r["parameters"]["d"]):r for r in candidates if float(r["parameters"]["fraction"])==fraction}
        for dose,record in by_dose.items():
            if dose==0.0:continue
            values=[np.sign(dose)*(float(a["metrics"]["T8_occupancy"])-float(s["metrics"]["T8_occupancy"])) for a,s in pairs.get(record["condition_id"],[])]
            if len(values)>=10:q[dose]=float(np.median(values))
        for left,right in ((-2.0,-1.0),(1.0,2.0)):
            if q.get(left,float("-inf"))>0.0 and q.get(right,float("-inf"))>0.0:
                eligible.add(by_dose[left]["condition_id"]);eligible.add(by_dose[right]["condition_id"])
    return eligible


def analyze_level_b(
    bundle: ContractBundle,
    records: Sequence[Mapping[str, Any]],
    selected_targets: Sequence[str],
    output_root: Path,
) -> dict[str, Any]:
    registry=build_level_b_registry(bundle,selected_targets)
    expected={run.run_id for run in registry.runs};observed={record["run_id"] for record in records if record["stage"]=="LEVEL_B_SURFACE"}
    if expected!=observed:raise ContractError("conditional Level-B result identity set mismatch")
    pools,attr_blocks=_attribution_pools(bundle,records);domain=validate_domain(bundle)
    target_data:dict[str,Any]={};point_members={member:None for member in enumerate_level_b_family(selected_targets)}
    for target in selected_targets:
        candidates,pairs=_paired_candidates(records,target,"LEVEL_B_SURFACE");scores={}
        for candidate in candidates:
            block_pairs=pairs.get(candidate["condition_id"],[])
            status=denominator_status([{"draw_id":a["parameter_draw_id"],"seed":a["seed"]} for a,_ in block_pairs],minimum_blocks=10,minimum_seeds=3,minimum_draws=3)
            attribution=None
            if target=="T4" and float(candidate["parameters"]["d"])>0:
                row=domain.resolve("LEVEL_B_SURFACE",float(candidate["parameters"]["d"]),float(candidate["parameters"]["f_c"]),0.85);attribution=pools.get(row["domain_row_id"])
            scores[candidate["condition_id"]]=None if status["status"]!="ESTIMABLE" else cell_score(target,candidate,block_pairs,attribution=attribution)
            rr=int(candidate["condition_id"].split("_R")[1].split("_")[0]);cc=int(candidate["condition_id"].split("_C")[1])
            point_members[f"LB3/{target}/R{rr:02d}/C{cc:02d}/C_MINUS_1"]=None if scores[candidate["condition_id"]] is None else float(scores[candidate["condition_id"]])-1.0
        mapping_scores=None
        if target=="T1":
            mapping_scores={}
            for cc,amplitude in enumerate((0.4,0.6,0.8,1.0,1.2)):
                mapped=[r for r in candidates if float(r["parameters"]["A"])==amplitude];values=[scores.get(r["condition_id"]) for r in mapped]
                value=None if len(mapped)!=5 or len(values)!=5 or any(v is None for v in values) else min(float(v) for v in values)
                mapping_scores[amplitude]=value;point_members[f"LB3/T1/POLICY/A{cc:02d}/G_MINUS_1"]=None if value is None else value-1.0
        elif target=="T7":
            mapping_scores={}
            for cc,carrier in enumerate((11.0,12.0,13.0,14.0,15.0)):
                value=_t7_level_b_mapping_score(carrier,candidates,pairs);mapping_scores[carrier]=value;point_members[f"LB3/T7/POLICY/FC{cc:02d}/G_MINUS_1"]=None if value is None else value-1.0
        t8eligible=_level_b_t8_eligible(candidates,pairs) if target=="T8" else None
        policy=select_level_b_policy(target,candidates,scores,mapping_scores=mapping_scores,t8_eligible_condition_ids=t8eligible)
        target_data[target]={"candidates":candidates,"pairs":pairs,"scores":scores,"mapping_scores":mapping_scores,"selected_policy":policy,"t8_eligible":t8eligible}
    indices=hierarchical_indices(seed=860220,draw_order=LEVEL_B_DRAWS,seed_order=LEVEL_B_SEEDS);boot=[]
    for replicate in range(BOOTSTRAP_REPLICATES):
        values={member:None for member in point_members}
        for target in selected_targets:
            data=target_data[target];resampled_pairs={cid:_resample_pairs(blocks,indices.draw_indices[replicate],indices.seed_indices[replicate]) for cid,blocks in data["pairs"].items()};scores={}
            for candidate in data["candidates"]:
                attribution=None
                if target=="T4" and float(candidate["parameters"]["d"])>0:
                    row=domain.resolve("LEVEL_B_SURFACE",float(candidate["parameters"]["d"]),float(candidate["parameters"]["f_c"]),0.85)
                    source={(b["parameter_draw_id"],int(b["seed"])):b for b in attr_blocks.get(row["domain_row_id"],[])};sampled=[]
                    for occurrence,di in enumerate(indices.draw_indices[replicate]):
                        draw=LEVEL_B_DRAWS[int(di)]
                        for si in indices.seed_indices[replicate,occurrence]:
                            key=(draw,LEVEL_B_SEEDS[int(si)])
                            if key in source:sampled.append(source[key])
                    attribution=pool_row(row,sampled,bootstrap=True,family_id="F_LEVEL_B_SURFACES",replicate_index=replicate)
                score=cell_score(target,candidate,resampled_pairs.get(candidate["condition_id"],[]),attribution=attribution);scores[candidate["condition_id"]]=score
                rr=int(candidate["condition_id"].split("_R")[1].split("_")[0]);cc=int(candidate["condition_id"].split("_C")[1]);values[f"LB3/{target}/R{rr:02d}/C{cc:02d}/C_MINUS_1"]=None if score is None else score-1.0
            if target=="T1":
                for cc,amplitude in enumerate((0.4,0.6,0.8,1.0,1.2)):
                    mapped=[r for r in data["candidates"] if float(r["parameters"]["A"])==amplitude];items=[scores.get(r["condition_id"]) for r in mapped];g=None if len(mapped)!=5 or len(items)!=5 or any(v is None for v in items) else min(float(v) for v in items);values[f"LB3/T1/POLICY/A{cc:02d}/G_MINUS_1"]=None if g is None else g-1.0
            if target=="T7":
                for cc,carrier in enumerate((11.0,12.0,13.0,14.0,15.0)):
                    g=_t7_level_b_bootstrap_score(carrier,data["candidates"],data["pairs"],indices.draw_indices[replicate],indices.seed_indices[replicate]);values[f"LB3/T7/POLICY/FC{cc:02d}/G_MINUS_1"]=None if g is None else g-1.0
        boot.append(values)
    bands=simultaneous_max_bands(point_members,boot,family_id="F_LEVEL_B_SURFACES",seed=860220)
    artifact={"schema_version":"4.0.0","artifact_type":"CONDITIONAL_LEVEL_B_ANALYSIS","freeze_sha256":FREEZE_HASH,"protocol_audit_sha256":PROTOCOL_AUDIT_HASH,"selected_targets":list(selected_targets),"run_count":len(expected),"selected_policies":{t:target_data[t]["selected_policy"] for t in selected_targets},"point_members":point_members,"simultaneous_bands":bands,"t4_attribution_pools":pools}
    path=output_root/"analysis"/"CONDITIONAL_LEVEL_B_ANALYSIS.json";digest=atomic_write_json(path,_json_safe(artifact),output_root=output_root)
    return {"path":str(path),"sha256":digest,**artifact}


def write_phase2b_disposition(bundle: ContractBundle, output_root: Path) -> dict[str, Any]:
    artifact=phase2b_disposition(bundle); path=output_root/"analysis"/"PHASE2B_NOT_ESTIMABLE.json"; digest=atomic_write_json(path,artifact,output_root=output_root)
    return {"path":str(path),"sha256":digest,**artifact}
