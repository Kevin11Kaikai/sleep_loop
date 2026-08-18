"""Immutable input bindings and fail-closed JSON/checkpoint I/O.

Only standard-library modules are used here so integrity checks happen before
NumPy/SciPy are imported by an execution mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


CAMPAIGN_ID = "COSTA_PHASE2_FRESH_20260816_105951"
CANDIDATE_ID = "PHASE2_CONTROL_BENCH_V1"
ROUTE_VERSION = "4.0.0"

RELEASE_HASH = "AA3DC83FFDCA4DD63031ABCAC524790D6DB2133509259E4AD1B641D6D59FC6EF"
RELEASE_REVALIDATION_HASH = "B3FBA1F63E2360A9C9D8CC9ADC7C18A1750EF6076EB4564F02D6FD884056B2C6"
INCIDENT_HASH = "3B6AD51C5C6D93453C38CD511D016257E8E9FD8B1FD686FBC08144B9B5B5904B"
FREEZE_HASH = "F002EE65EA450757E85C854FF9A6DB93518F39F2D1188CEEFF253D5AC2BEA996"
PROTOCOL_AUDIT_HASH = "D30035223430075381048426BE8E2C9AAD53E6165F4CB15CD2242E48189EF496"

BOUND_HASHES: dict[str, str] = {
    "GOVERNANCE/IMPLEMENTATION_RELEASE_V4_V2.json": RELEASE_HASH,
    "GOVERNANCE/IMPLEMENTATION_RELEASE_V4_V2_REVALIDATION.json": RELEASE_REVALIDATION_HASH,
    "GOVERNANCE/IMPLEMENTATION_RELEASE_BINDING_INCIDENT.json": INCIDENT_HASH,
    "PROTOCOL/PROTOCOL_FREEZE_V4.json": FREEZE_HASH,
    "GOVERNANCE/PROTOCOL_FREEZE_REAUDIT_V4.json": PROTOCOL_AUDIT_HASH,
    "PREFLIGHT/run_neurolib.ps1": "E47B363FFE3724CE7026E9618D5D79A034F28E0DB21C0AED4F3EC4761A734867",
    "STATE/ENVIRONMENT_SNAPSHOT.json": "9ACD57E9876C5FFCA20E7EAD233BE04BA1F90FBA1880ABAE9846114CEE4850D6",
    "COMMON/COMPUTE_BUDGET.json": "4C3C5961F5BE6EB134D7952FE74083C3B18303ABEE740027E8D9EE764F35ACBF",
    "GOVERNANCE/COMPUTE_BUDGET_REAUDIT_V3.json": "7CD0CD77D523AA1E1CC7B85FD5947FA939F6EEC3A08D131AF85D0D0C0266642A",
    "PROTOCOL/CANDIDATE_V4/control_target_registry.json": "E300F2B2184B1430428FD1E2DB0CF76D07DCD5B89FF78E287C06F5213629B058",
    "PROTOCOL/CANDIDATE_V4/mechanistic_parameter_map.json": "EB811129C358B7A25F78FE8B9EB9E253F79D155F548EDDF9D8919A5366FCD013",
    "PROTOCOL/CANDIDATE_V4/phase2_protocol.json": "AB4B5226DF13529BED15D910DF12401FF192110FC152489E153FF9E2641B3BE7",
    "PROTOCOL/CANDIDATE_V4/advancement_rule.json": "0F0EDE0523A40D59D598CEA18672F53D0780607999253B88CF93F201E028D086",
    "PROTOCOL/CANDIDATE_V4/c2_gate_protocol.json": "DB7A36F243BBB08885B122AC4353A3ACE94D066A677F27FA04A912B64942135D",
    "PROTOCOL/CANDIDATE_V4/claim_registry.json": "E792BEF42B7A2A279CF2F25658C3165FA1F3D3090D01D5A22A573394B393268F",
    "PROTOCOL/CANDIDATE_V4/PREOUTCOME_DESIGN_MEMO.md": "83EB7ADE9ADCFBA3DF371C3192EB904C6194B2AC2A79EE55B47237D42227735B",
    "PROTOCOL/CANDIDATE_V4/t4_attribution_fixture_v3.json": "30402053094C0C1BCDEF87989D1D7DE3DD4628D2C8D815757785BBA75D9DB4A9",
}

JSON_INPUTS = tuple(p for p in BOUND_HASHES if p.lower().endswith(".json"))
PROTOCOL_ARTIFACTS = (
    "PROTOCOL/CANDIDATE_V4/control_target_registry.json",
    "PROTOCOL/CANDIDATE_V4/mechanistic_parameter_map.json",
    "PROTOCOL/CANDIDATE_V4/phase2_protocol.json",
    "PROTOCOL/CANDIDATE_V4/advancement_rule.json",
    "PROTOCOL/CANDIDATE_V4/c2_gate_protocol.json",
    "PROTOCOL/CANDIDATE_V4/claim_registry.json",
    "PROTOCOL/CANDIDATE_V4/PREOUTCOME_DESIGN_MEMO.md",
    "PROTOCOL/CANDIDATE_V4/t4_attribution_fixture_v3.json",
)

CANDIDATE_SOURCE_PATHS = (
    "CODE/CANDIDATE_V4/__init__.py",
    "CODE/CANDIDATE_V4/analysis.py",
    "CODE/CANDIDATE_V4/cli.py",
    "CODE/CANDIDATE_V4/contracts.py",
    "CODE/CANDIDATE_V4/cross_target.py",
    "CODE/CANDIDATE_V4/detectors.py",
    "CODE/CANDIDATE_V4/model.py",
    "CODE/CANDIDATE_V4/pipeline.py",
    "CODE/CANDIDATE_V4/preflight.py",
    "CODE/CANDIDATE_V4/run_registry.py",
    "CODE/CANDIDATE_V4/statistics.py",
    "CODE/CANDIDATE_V4/t4_attribution.py",
)
EXPECTED_SCHEMA_PATH = "PREFLIGHT/CANDIDATE_V4_EXPECTED_SCHEMA.json"
IMPLEMENTATION_MANIFEST_PATH = "PREFLIGHT/CANDIDATE_V4_IMPLEMENTATION_MANIFEST.json"


class ContractError(RuntimeError):
    """An immutable binding, authorization, schema, or resume check failed."""


def campaign_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_reparse(path: Path) -> bool:
    st = path.lstat()
    attrs = getattr(st, "st_file_attributes", 0)
    flag = getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attrs & flag)


def assert_plain_descendant(path: Path, root: Path, *, must_exist: bool) -> Path:
    root = root.resolve(strict=True)
    candidate = path.resolve(strict=must_exist)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"path escapes campaign root: {candidate}") from exc
    cursor = candidate
    while not cursor.exists():
        if cursor == root or cursor.parent == cursor:
            raise ContractError(f"no existing ancestor inside campaign root: {candidate}")
        cursor = cursor.parent
    while True:
        if _is_reparse(cursor):
            raise ContractError(f"reparse point prohibited: {cursor}")
        if cursor == root:
            break
        if cursor.parent == cursor:
            raise ContractError(f"ancestry did not reach campaign root: {candidate}")
        cursor = cursor.parent
    return candidate


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest().upper()


def verify_hash(path: Path, expected: str) -> str:
    observed = sha256_file(path)
    if observed != expected.upper():
        raise ContractError(
            f"SHA256 mismatch for {path}: expected {expected.upper()}, observed {observed}"
        )
    return observed


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def canonical_json_bytes(payload: Any) -> bytes:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (text + "\n").encode("utf-8")


def object_sha256(payload: Any) -> str:
    return sha256(canonical_json_bytes(payload)).hexdigest().upper()


@dataclass(frozen=True)
class ContractBundle:
    root: Path
    release: Mapping[str, Any]
    revalidation: Mapping[str, Any]
    freeze: Mapping[str, Any]
    protocol_audit: Mapping[str, Any]
    target_registry: Mapping[str, Any]
    parameter_map: Mapping[str, Any]
    protocol: Mapping[str, Any]
    advancement: Mapping[str, Any]
    c2: Mapping[str, Any]
    claims: Mapping[str, Any]
    fixture: Mapping[str, Any]
    input_hashes: Mapping[str, str]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load_contract(
    *,
    explicit_freeze_hash: str,
    explicit_protocol_audit_hash: str,
    explicit_release_hash: str,
    explicit_revalidation_hash: str,
) -> ContractBundle:
    """Verify every frozen input before exposing any scientific configuration."""
    supplied = {
        "freeze": explicit_freeze_hash.upper(),
        "protocol_audit": explicit_protocol_audit_hash.upper(),
        "release": explicit_release_hash.upper(),
        "revalidation": explicit_revalidation_hash.upper(),
    }
    expected = {
        "freeze": FREEZE_HASH,
        "protocol_audit": PROTOCOL_AUDIT_HASH,
        "release": RELEASE_HASH,
        "revalidation": RELEASE_REVALIDATION_HASH,
    }
    _require(supplied == expected, f"explicit authority hashes do not equal frozen values: {supplied}")
    root = campaign_root()
    observed: dict[str, str] = {}
    docs: dict[str, Any] = {}
    for rel, digest in BOUND_HASHES.items():
        path = assert_plain_descendant(root / Path(rel), root, must_exist=True)
        _require(path.is_file(), f"bound input is not a file: {path}")
        observed[rel] = verify_hash(path, digest)
        if rel in JSON_INPUTS:
            docs[rel] = load_json(path)

    release = docs["GOVERNANCE/IMPLEMENTATION_RELEASE_V4_V2.json"]
    revalidation = docs["GOVERNANCE/IMPLEMENTATION_RELEASE_V4_V2_REVALIDATION.json"]
    freeze = docs["PROTOCOL/PROTOCOL_FREEZE_V4.json"]
    audit = docs["GOVERNANCE/PROTOCOL_FREEZE_REAUDIT_V4.json"]
    protocol = docs["PROTOCOL/CANDIDATE_V4/phase2_protocol.json"]
    targets = docs["PROTOCOL/CANDIDATE_V4/control_target_registry.json"]
    params = docs["PROTOCOL/CANDIDATE_V4/mechanistic_parameter_map.json"]
    advancement = docs["PROTOCOL/CANDIDATE_V4/advancement_rule.json"]
    c2 = docs["PROTOCOL/CANDIDATE_V4/c2_gate_protocol.json"]
    claims = docs["PROTOCOL/CANDIDATE_V4/claim_registry.json"]
    fixture = docs["PROTOCOL/CANDIDATE_V4/t4_attribution_fixture_v3.json"]

    _require(release["release_id"] == "OUTCOME_BLIND_IMPLEMENTATION_RELEASE_V4_V2", "wrong release")
    _require(revalidation["overall_verdict"] == "PASS", "release revalidation did not pass")
    _require(revalidation["authorization_effect"] == "CORRECTED_V2_RELEASE_VALID_FOR_STATIC_IMPLEMENTATION_ONLY", "wrong release scope")
    _require(freeze["freeze_id"] == "PHASE2_PROTOCOL_FREEZE_V4", "wrong freeze identity")
    _require(audit["overall_verdict"] == "PASS", "protocol audit did not pass")
    for doc in (freeze, protocol, targets, params, advancement, c2, claims, fixture):
        _require(doc["campaign_id"] == CAMPAIGN_ID, "campaign identity mismatch")
    _require(protocol["candidate_identity"] == CANDIDATE_ID, "candidate mismatch")
    _require(protocol["protocol_route_version"] == ROUTE_VERSION, "route mismatch")
    frozen_artifacts = {item["path"]: item["sha256"] for item in freeze["artifacts"]}
    _require(frozen_artifacts == {p: BOUND_HASHES[p] for p in PROTOCOL_ARTIFACTS}, "freeze artifact set mismatch")
    _require(len(targets["targets"]) == 8, "target count must be eight")
    _require(len(c2["gates"]) == 10, "C2 gate count must be ten")
    _require(len(claims["claims"]) == 8, "claim count must be eight")
    _require(protocol["phase2b"]["formal_status"] == "NOT_ESTIMABLE", "Phase2B must be NOT_ESTIMABLE")
    _require(protocol["phase2b"]["eligible_public_development_night1_subjects"] == 0, "Phase2B N must be zero")
    _require(protocol["phase2b"]["minimum_subjects"] == 3, "Phase2B minimum must be three")
    _require(protocol["phase2a"]["unique_simulation_count"]["maximum_level_a_runs"] == 1800, "Level-A count mismatch")
    _require(protocol["phase2a"]["conditional_level_b"]["exact_run_arithmetic"]["maximum_runs"] == 900, "Level-B count mismatch")
    _require(protocol["phase2a"]["cross_target_matrix"]["verifier_execution"]["unique_runs"] == "24*(18+1)=456", "verifier arithmetic mismatch")
    return ContractBundle(
        root=root,
        release=release,
        revalidation=revalidation,
        freeze=freeze,
        protocol_audit=audit,
        target_registry=targets,
        parameter_map=params,
        protocol=protocol,
        advancement=advancement,
        c2=c2,
        claims=claims,
        fixture=fixture,
        input_hashes=observed,
    )


MODE_AUTHORIZATION_EFFECTS: dict[str, set[str]] = {
    "implementation-contract": {"CORRECTED_V2_RELEASE_VALID_FOR_STATIC_IMPLEMENTATION_ONLY"},
    "data-free-preflight": {"DATA_FREE_PREFLIGHT_AUTHORIZED"},
    "phase2a-worker": {"PHASE2A_WORKER_EXECUTION_AUTHORIZED"},
    "phase2a-analyze": {"PHASE2A_ANALYSIS_AUTHORIZED", "PHASE2A_WORKER_EXECUTION_AUTHORIZED"},
    "cross-target-verifier": {"CROSS_TARGET_VERIFIER_EXECUTION_AUTHORIZED"},
    "cross-target-analyze": {"CROSS_TARGET_VERIFIER_ANALYSIS_AUTHORIZED", "CROSS_TARGET_VERIFIER_EXECUTION_AUTHORIZED"},
    "phase2b-disposition": {"CORRECTED_V2_RELEASE_VALID_FOR_STATIC_IMPLEMENTATION_ONLY", "PHASE2B_DISPOSITION_AUTHORIZED"},
}


def verify_mode_authorization(mode: str, receipt_path: Path, receipt_sha256: str) -> Mapping[str, Any]:
    if mode not in MODE_AUTHORIZATION_EFFECTS:
        raise ContractError(f"unapproved mode: {mode}")
    root = campaign_root()
    path = assert_plain_descendant(receipt_path, root, must_exist=True)
    verify_hash(path, receipt_sha256.upper())
    receipt = load_json(path)
    _require(receipt.get("campaign_id") == CAMPAIGN_ID, "authorization campaign mismatch")
    effect = receipt.get("authorization_effect")
    _require(effect in MODE_AUTHORIZATION_EFFECTS[mode], f"authorization effect {effect!r} does not authorize {mode}")
    _require(receipt.get("overall_verdict") == "PASS", "authorization verdict is not PASS")
    if mode not in {"implementation-contract", "phase2b-disposition"}:
        _require(receipt.get("scientific_outcomes", "NONE") == "NONE", "authorization receipt is not outcome blind")
    return receipt


def verify_implementation_manifest(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    """Verify the independently frozen source/schema inventory without importing it."""
    root = campaign_root()
    target = assert_plain_descendant(path, root, must_exist=True)
    required = (root / IMPLEMENTATION_MANIFEST_PATH).resolve(strict=True)
    _require(target == required, "implementation manifest path is not canonical")
    verify_hash(target, expected_sha256.upper())
    manifest = load_json(target)
    _require(manifest.get("artifact_type") == "CANDIDATE_V4_IMPLEMENTATION_MANIFEST", "wrong implementation manifest type")
    _require(manifest.get("campaign_id") == CAMPAIGN_ID, "implementation manifest campaign mismatch")
    _require(manifest.get("freeze_sha256") == FREEZE_HASH, "implementation manifest freeze mismatch")
    _require(manifest.get("protocol_audit_sha256") == PROTOCOL_AUDIT_HASH, "implementation manifest audit mismatch")
    entries = manifest.get("static_files")
    _require(isinstance(entries, list), "implementation manifest static_files missing")
    expected_paths = tuple(sorted(CANDIDATE_SOURCE_PATHS + (EXPECTED_SCHEMA_PATH,)))
    observed_paths = tuple(sorted(str(entry.get("path")) for entry in entries))
    _require(observed_paths == expected_paths, "implementation manifest file set mismatch")
    for entry in entries:
        rel = str(entry["path"])
        file_path = assert_plain_descendant(root / rel, root, must_exist=True)
        _require(file_path.is_file(), f"manifest member is not a file: {rel}")
        _require(file_path.stat().st_size == int(entry["bytes"]), f"manifest byte count mismatch: {rel}")
        verify_hash(file_path, str(entry["sha256"]))
    computed_set_hash = object_sha256(entries)
    _require(computed_set_hash == manifest.get("implementation_set_sha256"), "implementation set hash mismatch")
    _require(manifest.get("self_hash_policy") == "MANIFEST_SELF_HASH_EXCLUDED_TO_AVOID_CIRCULARITY", "manifest self-hash policy mismatch")
    return manifest


def atomic_write_json(path: Path, payload: Any, *, output_root: Path, immutable: bool = True) -> str:
    """Atomically write canonical JSON; identical existing files are reusable only."""
    output_root = output_root.resolve(strict=True)
    target = assert_plain_descendant(path, output_root, must_exist=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload)
    digest = sha256(data).hexdigest().upper()
    if target.exists():
        current = target.read_bytes()
        if current != data or immutable:
            if current == data:
                return digest
            raise ContractError(f"immutable artifact already exists with different bytes: {target}")
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    return digest


def write_checkpoint(path: Path, payload: Mapping[str, Any], *, output_root: Path) -> str:
    body = dict(payload)
    body["campaign_id"] = CAMPAIGN_ID
    body["candidate_identity"] = CANDIDATE_ID
    body["freeze_sha256"] = FREEZE_HASH
    body["protocol_audit_sha256"] = PROTOCOL_AUDIT_HASH
    body["checkpoint_payload_sha256"] = object_sha256(payload)
    return atomic_write_json(path, body, output_root=output_root, immutable=True)


def load_checkpoint(path: Path, *, output_root: Path, required_keys: Iterable[str]) -> Mapping[str, Any]:
    target = assert_plain_descendant(path, output_root.resolve(strict=True), must_exist=True)
    body = load_json(target)
    _require(body.get("campaign_id") == CAMPAIGN_ID, "checkpoint campaign mismatch")
    _require(body.get("candidate_identity") == CANDIDATE_ID, "checkpoint candidate mismatch")
    _require(body.get("freeze_sha256") == FREEZE_HASH, "checkpoint freeze mismatch")
    _require(body.get("protocol_audit_sha256") == PROTOCOL_AUDIT_HASH, "checkpoint audit mismatch")
    for key in required_keys:
        _require(key in body, f"checkpoint missing key {key}")
    payload = {k: v for k, v in body.items() if k not in {
        "campaign_id", "candidate_identity", "freeze_sha256", "protocol_audit_sha256", "checkpoint_payload_sha256"
    }}
    _require(object_sha256(payload) == body.get("checkpoint_payload_sha256"), "checkpoint payload hash mismatch")
    return body
