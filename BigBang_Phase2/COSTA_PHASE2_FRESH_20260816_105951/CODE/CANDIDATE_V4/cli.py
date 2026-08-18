"""Fail-closed direct-script CLI for Candidate V4.

The official PowerShell launcher accepts a Python script rather than ``-m``;
the small package bootstrap below therefore precedes all local imports.  Only
standard-library contract code is imported before authority and hashes pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from CANDIDATE_V4.contracts import (  # type: ignore[import-not-found]
        ContractError,
        FREEZE_HASH,
        PROTOCOL_AUDIT_HASH,
        assert_plain_descendant,
        campaign_root,
        load_contract,
        load_json,
        verify_hash,
        verify_implementation_manifest,
        verify_mode_authorization,
    )
else:
    from .contracts import (
        ContractError,
        FREEZE_HASH,
        PROTOCOL_AUDIT_HASH,
        assert_plain_descendant,
        campaign_root,
        load_contract,
        load_json,
        verify_hash,
        verify_implementation_manifest,
        verify_mode_authorization,
    )


MODES = (
    "implementation-contract",
    "data-free-preflight",
    "phase2a-worker",
    "phase2a-analyze",
    "cross-target-verifier",
    "cross-target-analyze",
    "phase2b-disposition",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="candidate-v4", allow_abbrev=False)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--protocol-audit-sha256", required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--release-revalidation-sha256", required=True)
    parser.add_argument("--implementation-manifest", required=True, type=Path)
    parser.add_argument("--implementation-manifest-sha256", required=True)
    parser.add_argument("--authorization-receipt", required=True, type=Path)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stage", choices=("level-a", "level-b"))
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--cpu-hour-cap", type=float)
    parser.add_argument("--manifest", action="append", default=[], metavar="PATH=SHA256")
    parser.add_argument("--policy-json", type=Path)
    parser.add_argument("--policy-sha256")
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--level-a-analysis-json", type=Path)
    parser.add_argument("--level-a-analysis-sha256")
    return parser


def _need(value: Any, name: str) -> Any:
    if value is None or value == []:
        raise ContractError(f"{name} is required for this mode")
    return value


def _verified_output_json(path: Path, digest: str, output_root: Path) -> Mapping[str, Any]:
    target = assert_plain_descendant(path, output_root, must_exist=True)
    verify_hash(target, digest.upper())
    body = load_json(target)
    if not isinstance(body, dict):
        raise ContractError(f"JSON artifact must be an object: {target}")
    if body.get("freeze_sha256") != FREEZE_HASH or body.get("protocol_audit_sha256") != PROTOCOL_AUDIT_HASH:
        raise ContractError(f"JSON artifact protocol binding mismatch: {target}")
    return body


def _parse_manifests(specifications: Sequence[str], output_root: Path) -> list[Path]:
    paths: list[Path] = []
    for specification in specifications:
        if "=" not in specification:
            raise ContractError("each --manifest must use PATH=SHA256")
        raw_path, digest = specification.rsplit("=", 1)
        if not raw_path or len(digest) != 64:
            raise ContractError(f"malformed manifest binding: {specification}")
        path = assert_plain_descendant(Path(raw_path), output_root, must_exist=True)
        verify_hash(path, digest.upper())
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ContractError("duplicate shard manifest argument")
    return paths


def _policies(body: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    if body.get("artifact_type") not in {"LEVEL_A_ANALYSIS", "LEVEL_A_FROZEN_POLICIES"}:
        raise ContractError("verifier policy provenance must be frozen Level-A worker analysis")
    value = body.get("policies", body.get("selected_level_a_policies"))
    if not isinstance(value, dict) or tuple(sorted(value)) != tuple(f"T{i}" for i in range(1, 9)):
        raise ContractError("policy artifact must contain exact T1-T8 policy objects")
    if any(row.get("status") != "SELECTED" for row in value.values()):
        raise ContractError("cross-target verifier cannot launch with a NOT_ESTIMABLE row policy")
    return value


def _selected_targets(body: Mapping[str, Any]) -> list[str]:
    if body.get("artifact_type") != "CONDITIONAL_LEVEL_B_SELECTION":
        raise ContractError("Level-B selection must come from post-verifier CONDITIONAL_LEVEL_B_SELECTION")
    value = body.get("selected_targets")
    if not isinstance(value, list) or len(value) > 3 or len(set(value)) != len(value):
        raise ContractError("selection artifact must contain at most three distinct targets")
    if any(target not in {f"T{i}" for i in range(1, 9)} for target in value):
        raise ContractError("selection artifact contains an unknown target")
    return [str(target) for target in value]


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))


def _dispatch(args: argparse.Namespace) -> Mapping[str, Any]:
    bundle = load_contract(
        explicit_freeze_hash=args.freeze_sha256,
        explicit_protocol_audit_hash=args.protocol_audit_sha256,
        explicit_release_hash=args.release_sha256,
        explicit_revalidation_hash=args.release_revalidation_sha256,
    )
    implementation = verify_implementation_manifest(args.implementation_manifest, args.implementation_manifest_sha256)
    authorization = verify_mode_authorization(args.mode, args.authorization_receipt, args.authorization_sha256)

    if args.mode == "implementation-contract":
        return {
            "schema_version": "4.0.0",
            "artifact_type": "IMPLEMENTATION_CONTRACT_CHECK",
            "status": "PASS",
            "scientific_execution": "NONE",
            "input_hashes": dict(bundle.input_hashes),
            "implementation_manifest_sha256": args.implementation_manifest_sha256.upper(),
            "implementation_set_sha256": implementation["implementation_set_sha256"],
            "authorization_effect": authorization["authorization_effect"],
        }

    # Heavy scientific dependencies are imported only after all authority,
    # immutable-input, and implementation-set checks pass.
    if __package__ in {None, ""}:
        from CANDIDATE_V4.pipeline import (  # type: ignore[import-not-found]
            analyze_cross_target,
            analyze_level_a,
            analyze_level_b,
            execute_registry_shard,
            load_shard_records,
            prepare_output_root,
            write_phase2b_disposition,
            write_registry,
        )
        from CANDIDATE_V4.preflight import run_data_free_preflight  # type: ignore[import-not-found]
        from CANDIDATE_V4.run_registry import build_cross_target_registry, build_level_a_registry, build_level_b_registry  # type: ignore[import-not-found]
    else:
        from .pipeline import (
            analyze_cross_target,
            analyze_level_a,
            analyze_level_b,
            execute_registry_shard,
            load_shard_records,
            prepare_output_root,
            write_phase2b_disposition,
            write_registry,
        )
        from .preflight import run_data_free_preflight
        from .run_registry import build_cross_target_registry, build_level_a_registry, build_level_b_registry

    requested_output = _need(args.output_root, "--output-root")
    output_root = prepare_output_root(bundle, requested_output)
    if args.mode == "data-free-preflight":
        return run_data_free_preflight(bundle, output_root)
    if args.mode == "phase2b-disposition":
        return write_phase2b_disposition(bundle, output_root)

    if args.mode == "phase2a-worker":
        stage = _need(args.stage, "--stage")
        shard_index = int(_need(args.shard_index, "--shard-index"))
        shard_count = int(_need(args.shard_count, "--shard-count"))
        cpu_cap = float(_need(args.cpu_hour_cap, "--cpu-hour-cap"))
        if not (0.0 < cpu_cap <= 60.0):
            raise ContractError("Phase2A shard CPU-hour cap must be in (0,60]")
        if stage == "level-a":
            if args.selection_json is not None or args.selection_sha256 is not None:
                raise ContractError("Level-A worker refuses Level-B selection arguments")
            registry = build_level_a_registry(bundle)
        else:
            selection_path = _need(args.selection_json, "--selection-json")
            selection_hash = _need(args.selection_sha256, "--selection-sha256")
            selection = _verified_output_json(selection_path, selection_hash, output_root)
            registry = build_level_b_registry(bundle, _selected_targets(selection))
        registry_receipt = write_registry(registry, output_root)
        shard = execute_registry_shard(bundle, registry, output_root, shard_index=shard_index, shard_count=shard_count, cpu_hour_cap=cpu_cap)
        return {"schema_version": "4.0.0", "artifact_type": "PHASE2A_WORKER_RESULT", "stage": stage, "registry": registry_receipt, "shard": shard}

    if args.mode == "phase2a-analyze":
        stage = _need(args.stage, "--stage")
        manifests = _parse_manifests(_need(args.manifest, "--manifest"), output_root)
        records = load_shard_records(output_root, manifests)
        if stage == "level-a":
            if args.selection_json is not None or args.selection_sha256 is not None:
                raise ContractError("Level-A analysis refuses Level-B selection arguments")
            return analyze_level_a(bundle, records, output_root)
        selection = _verified_output_json(_need(args.selection_json, "--selection-json"), _need(args.selection_sha256, "--selection-sha256"), output_root)
        return analyze_level_b(bundle, records, _selected_targets(selection), output_root)

    policy_body = _verified_output_json(_need(args.policy_json, "--policy-json"), _need(args.policy_sha256, "--policy-sha256"), output_root)
    policies = _policies(policy_body)
    if args.mode == "cross-target-verifier":
        shard_index = int(_need(args.shard_index, "--shard-index"))
        shard_count = int(_need(args.shard_count, "--shard-count"))
        cpu_cap = float(_need(args.cpu_hour_cap, "--cpu-hour-cap"))
        if not (0.0 < cpu_cap <= 50.0):
            raise ContractError("verifier shard CPU-hour cap must be in (0,50]")
        registry = build_cross_target_registry(bundle, policies)
        registry_receipt = write_registry(registry, output_root)
        shard = execute_registry_shard(bundle, registry, output_root, shard_index=shard_index, shard_count=shard_count, cpu_hour_cap=cpu_cap)
        return {"schema_version": "4.0.0", "artifact_type": "CROSS_TARGET_VERIFIER_RESULT", "registry": registry_receipt, "shard": shard}
    if args.mode == "cross-target-analyze":
        manifests = _parse_manifests(_need(args.manifest, "--manifest"), output_root)
        records = load_shard_records(output_root, manifests)
        level_a = _verified_output_json(
            _need(args.level_a_analysis_json, "--level-a-analysis-json"),
            _need(args.level_a_analysis_sha256, "--level-a-analysis-sha256"),
            output_root,
        )
        if level_a.get("artifact_type") != "LEVEL_A_ANALYSIS":
            raise ContractError("cross-target finalization requires LEVEL_A_ANALYSIS")
        return analyze_cross_target(bundle, records, policies, level_a, output_root)
    raise ContractError(f"unapproved mode escaped parser: {args.mode}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        _emit(_dispatch(args))
        return 0
    except (ContractError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"artifact_type": "CANDIDATE_V4_CLI_FAILURE", "status": "FAIL_CLOSED", "exception_type": type(exc).__name__, "message": str(exc)}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
