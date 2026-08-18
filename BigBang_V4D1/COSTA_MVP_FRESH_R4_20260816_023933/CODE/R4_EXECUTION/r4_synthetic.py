"""Synthetic-only M1 preflight and closed-runtime reporting."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

from r4_core import activate_mechanistic_imports, canonical_sha256, finish_manifest, load_protocol_claims, new_output_child, write_canonical


REQUIRED_PYTHON = Path(r"C:\Users\YUS190\AppData\Local\anaconda3\envs\neurolib\python.exe")
NEUROLIB_REPO = Path(r"D:\Year3_Mao_Projects\neurolib")
REQUIRED_COMMIT = "9b6b2b8f082c0cfa212f05576ead55bf23046d6f"
EXACT_LAUNCHER = r"C:\Users\YUS190\AppData\Local\anaconda3\condabin\conda.bat run --no-capture-output -n neurolib python"


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", "safe.directory=D:/Year3_Mao_Projects/neurolib", "-C", str(NEUROLIB_REPO), *arguments],
        check=False,
        text=True,
        capture_output=True,
    )


def runtime_checks() -> dict[str, object]:
    commit_result = _git("rev-parse", "HEAD")
    diff_result = _git("diff", "--quiet")
    cached_result = _git("diff", "--cached", "--quiet")
    status_result = _git("status", "--porcelain", "--untracked-files=all")
    status_lines = [line for line in status_result.stdout.splitlines() if line]
    disallowed_status = [line for line in status_lines if line != "?? environment.yml"]
    environment_yml_present = "?? environment.yml" in status_lines
    environment_mentions = [
        value for key, value in os.environ.items()
        if "environment.yml" in str(value).lower() and key not in {"TEMP", "TMP", "NUMBA_CACHE_DIR"}
    ]
    loaded_environment_modules = [
        str(getattr(module, "__file__", "")) for module in sys.modules.values()
        if "environment.yml" in str(getattr(module, "__file__", "")).lower()
    ]
    checks = {
        "exact_python": Path(sys.executable).resolve() == REQUIRED_PYTHON.resolve(),
        "neurolib_version_0_6_1": importlib.metadata.version("neurolib") == "0.6.1",
        "neurolib_commit": commit_result.returncode == 0 and commit_result.stdout.strip() == REQUIRED_COMMIT,
        "tracked_diff_clean": diff_result.returncode == 0 and cached_result.returncode == 0,
        "only_permitted_untracked_path": status_result.returncode == 0 and not disallowed_status,
        "environment_yml_absent_from_module_open_config_closure": not environment_mentions and not loaded_environment_modules,
        "single_thread_openblas": os.environ.get("OPENBLAS_NUM_THREADS") == "1",
        "single_thread_omp": os.environ.get("OMP_NUM_THREADS") == "1",
        "single_thread_mkl": os.environ.get("MKL_NUM_THREADS") == "1",
        "mne_home_writes_disabled": os.environ.get("MNE_DONTWRITE_HOME", "").lower() == "true",
        "campaign_local_temp": bool(os.environ.get("TEMP")) and "COSTA_MVP_FRESH_R4_20260816_023933" in os.environ.get("TEMP", ""),
        "campaign_local_tmp": bool(os.environ.get("TMP")) and "COSTA_MVP_FRESH_R4_20260816_023933" in os.environ.get("TMP", ""),
        "campaign_local_numba_cache": bool(os.environ.get("NUMBA_CACHE_DIR")) and "COSTA_MVP_FRESH_R4_20260816_023933" in os.environ.get("NUMBA_CACHE_DIR", ""),
    }
    return {
        "launcher": EXACT_LAUNCHER,
        "python": str(Path(sys.executable).resolve()),
        "neurolib_commit_observed": commit_result.stdout.strip(),
        "neurolib_status": status_lines,
        "environment_yml_present_untracked": environment_yml_present,
        "environment_yml_contents_read": False,
        "checks": checks,
        "pass": all(checks.values()),
    }


def run_synthetic(output_root: Path, output_child: str) -> Path:
    protocol, claims, frozen_hashes = load_protocol_claims()
    activate_mechanistic_imports()
    from harness import execute_core

    first = execute_core(protocol)
    second = execute_core(protocol)
    first_digest = canonical_sha256(first)
    second_digest = canonical_sha256(second)
    runtime = runtime_checks()
    g1_checks = dict(runtime["checks"])
    g1_checks["bitwise_core_repetition_digest"] = first_digest == second_digest
    gates = {"G1": {"pass": all(g1_checks.values()), "checks": g1_checks}}
    gates.update({name: first["gates"][name] for name in ("G2", "G3", "G4", "G5")})
    report = {
        "schema": "COSTA_R4_SYNTHETIC_M1_V1",
        "mode": "SYNTHETIC_ONLY_NO_EMPIRICAL_DATA",
        "claim_id": claims["primary"]["claim_id"],
        "core_repetitions": 2,
        "core_digest_first": first_digest,
        "core_digest_second": second_digest,
        "runtime": runtime,
        "gates_G1_to_G5": gates,
        "synthetic_G6_interface_status": first["gates"]["G6"],
        "scientific_core": first,
        "empirical_claim_status": "NOT_TESTED_EMPIRICALLY",
        "limitations": claims["primary"]["phase_1_interpretation"],
        "frozen_input_hashes": frozen_hashes,
    }
    directory = new_output_child(output_root, output_child)
    artifacts = {"SYNTHETIC_M1_REPORT.json": write_canonical(directory / "SYNTHETIC_M1_REPORT.json", report)}
    finish_manifest(directory, "SYNTHETIC_M1", artifacts, frozen_hashes)
    return directory

