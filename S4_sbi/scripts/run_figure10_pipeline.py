"""Resumable stage driver for the matched 8D/7D Figure-10 experiment."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "S4_sbi" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sleep_sbi.figure10_bank import finalize_matched_banks, run_track
from sleep_sbi.figure10_diagnostics import (
    analyze_global_ranks,
    analyze_posterior_structure,
    generate_global_datasets,
    generate_lc2st_datasets,
    generate_primary_observation,
    operational_verdict,
    run_global_diagnostic,
    run_lc2st,
    run_ppc,
)
from sleep_sbi.figure10_protocol import (
    FINAL_SCALE,
    INTERMEDIATE_SCALE,
    RESULTS_ROOT,
    atomic_json,
    verify_preregistration,
)
from sleep_sbi.figure10_training import train_ensemble
from sleep_sbi.figure10_reporting import (
    build_final_report,
    build_scale_sensitivity,
    generate_all_figures,
)


LOG_ROOT = RESULTS_ROOT / "logs"


def _prevent_sleep_while_running() -> None:
    if os.name == "nt":
        result = ctypes.windll.kernel32.SetThreadExecutionState(
            0x80000000 | 0x00000001
        )
        if not result:
            raise OSError("SetThreadExecutionState failed")


def _checkpoint(stage: str, status: str, details: dict | None = None) -> None:
    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        "pid": os.getpid(),
        "preregistration_hash": verify_preregistration(),
        "details": details or {},
    }
    atomic_json(LOG_ROOT / f"{stage}_checkpoint.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def stage_banks() -> None:
    _checkpoint("banks", "running")
    results = {}
    for track in ("8d", "7d"):
        results[track] = run_track(track, FINAL_SCALE)
    results["final"] = finalize_matched_banks(FINAL_SCALE)
    _checkpoint("banks", "complete", results)


def stage_training() -> None:
    _checkpoint("training", "running")
    results = {}
    for scale in (INTERMEDIATE_SCALE, FINAL_SCALE):
        for track in ("8d", "7d"):
            key = f"{track}_{scale}"
            results[key] = train_ensemble(track, scale)
    _checkpoint("training", "complete", results)


def stage_global() -> None:
    _checkpoint("global", "running")
    datasets = generate_global_datasets()
    results = {"datasets": datasets}
    for dataset_id in ("official_300", "powered_1024"):
        for track in ("8d", "7d"):
            path = run_global_diagnostic(track, dataset_id, FINAL_SCALE)
            results[f"{dataset_id}_{track}_{FINAL_SCALE}"] = analyze_global_ranks(
                path
            )
    for track in ("8d", "7d"):
        path = run_global_diagnostic(track, "powered_1024", INTERMEDIATE_SCALE)
        results[f"powered_1024_{track}_{INTERMEDIATE_SCALE}"] = (
            analyze_global_ranks(path)
        )
    _checkpoint("global", "complete", results)


def stage_lc2st() -> None:
    _checkpoint("lc2st", "running")
    primary = generate_primary_observation()
    datasets = generate_lc2st_datasets()
    results = {
        "primary_observation": primary.relative_to(RESULTS_ROOT).as_posix(),
        "datasets": datasets,
    }
    for track in ("8d", "7d"):
        results[track] = run_lc2st(track)
    _checkpoint("lc2st", "complete", results)


def stage_ppc_structure() -> None:
    _checkpoint("ppc_structure", "running")
    generate_primary_observation()
    results = {}
    for track in ("8d", "7d"):
        results[f"{track}_ppc"] = run_ppc(track)
        results[f"{track}_structure"] = analyze_posterior_structure(track)
    _checkpoint("ppc_structure", "complete", results)


def stage_verdict() -> None:
    _checkpoint("verdict", "running")
    results = {track: operational_verdict(track) for track in ("8d", "7d")}
    _checkpoint("verdict", "complete", results)


def stage_reporting() -> None:
    _checkpoint("reporting", "running")
    scale = build_scale_sensitivity()
    figures = generate_all_figures()
    report = build_final_report()
    _checkpoint(
        "reporting",
        "complete",
        {
            "scale_rows": len(scale),
            "figures": figures,
            "verdicts": report["verdicts"],
        },
    )


def main() -> None:
    _prevent_sleep_while_running()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "banks",
            "training",
            "global",
            "lc2st",
            "ppc",
            "verdict",
            "reporting",
        ),
    )
    args = parser.parse_args()
    {
        "banks": stage_banks,
        "training": stage_training,
        "global": stage_global,
        "lc2st": stage_lc2st,
        "ppc": stage_ppc_structure,
        "verdict": stage_verdict,
        "reporting": stage_reporting,
    }[args.stage]()


if __name__ == "__main__":
    main()
