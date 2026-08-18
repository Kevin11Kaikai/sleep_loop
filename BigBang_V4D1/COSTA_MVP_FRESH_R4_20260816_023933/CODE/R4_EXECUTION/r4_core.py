"""Fail-closed serialization, frozen-input, and artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Mapping


EXECUTION_DIR = Path(__file__).resolve().parent
CODE_DIR = EXECUTION_DIR.parent
CAMPAIGN_ROOT = CODE_DIR.parent

FROZEN_FILES = {
    "MECHANISTIC_V2/__init__.py": "83CD4B0D88653F36C83F2707BD011D94F8D202956BFA299D44070BB3D9BE7BCA",
    "MECHANISTIC_V2/model.py": "775A3961093C386A1E7DF8F0D033963EAB684F2566AE206E5BB6DB4A6CD03869",
    "MECHANISTIC_V2/harness.py": "A8735BD770E4EBCB7657EDEA9ABE9B4831025000E08E00C8F6D696E5131D8F6B",
    "MECHANISTIC_V2/partitions.py": "21941B364BA56D6F447924B12C3ADD38F9D0700D21EDB2B1DFFA28EC53D9451B",
    "FROZEN_V2/protocol_v2.json": "338366593A2A4C8626349D70AA38306C5931F0FF4D22FD96DB546DD41505E3C3",
    "FROZEN_V2/claims_v2.json": "6E386873060858737DA2C8356BA59D291CA59BC914F76951AF49AE0FED1B4891",
    "TESTS/test_synthetic_v2.py": "445E4FBA4CA43B580377E8597DFE1EC63E412EF4661F8A544A50D220C5AEEA86",
}

EXPECTED_PROTOCOL_ID = "COSTA_CLEANROOM_THALAMOCORTICAL_V2_PHASE1"
EXPECTED_CLAIMS_ID = "COSTA_TC_V2_CLAIMS"


def activate_mechanistic_imports() -> None:
    path = str(CODE_DIR / "MECHANISTIC_V2")
    if path not in sys.path:
        sys.path.insert(0, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest().upper()


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_frozen_inputs() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in FROZEN_FILES.items():
        path = CODE_DIR / relative
        if not path.is_file():
            raise RuntimeError(f"frozen input missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"frozen input hash mismatch: {relative}")
        observed[relative] = actual
    return observed


def load_protocol_claims() -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
    hashes = validate_frozen_inputs()
    protocol = read_json(CODE_DIR / "FROZEN_V2" / "protocol_v2.json")
    claims = read_json(CODE_DIR / "FROZEN_V2" / "claims_v2.json")
    if not isinstance(protocol, dict) or protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise RuntimeError("frozen protocol schema/identity mismatch")
    if not isinstance(claims, dict) or claims.get("registry_id") != EXPECTED_CLAIMS_ID:
        raise RuntimeError("frozen claims schema/identity mismatch")
    if protocol.get("candidate_bank") is None or len(protocol["candidate_bank"]) != 16:
        raise RuntimeError("frozen candidate-bank schema mismatch")
    return protocol, claims, hashes


def new_output_child(output_root: Path, child_name: str) -> Path:
    root = output_root.resolve()
    if not child_name or child_name in {".", ".."} or Path(child_name).name != child_name:
        raise ValueError("output child must be one simple path component")
    root.mkdir(parents=True, exist_ok=True)
    child = root / child_name
    if child.exists():
        raise FileExistsError(f"output child already exists: {child}")
    child.mkdir()
    return child


def write_canonical(path: Path, value: object) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    payload = canonical_bytes(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest().upper()


def finish_manifest(directory: Path, mode: str, artifact_hashes: Mapping[str, str], frozen_hashes: Mapping[str, str]) -> str:
    manifest = {
        "schema": "COSTA_R4_MANIFEST_V1",
        "mode": mode,
        "canonical_json": True,
        "frozen_input_hashes": dict(sorted(frozen_hashes.items())),
        "artifacts": dict(sorted(artifact_hashes.items())),
    }
    return write_canonical(directory / "MANIFEST.json", manifest)


def require_exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{label} keys mismatch: expected {sorted(expected)}, got {sorted(actual)}")
