"""Command-line driver for the resumable preregistered Route-3 7D stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT / "S4_sbi" / "src", ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))


def stage15(workers: int) -> None:
    from sleep_sbi.route3_7d_experiment import (
        analyze_local_multiscale_7d,
        analyze_preflight_multiseed,
        evaluate_preflight_gate,
        run_local_multiscale_7d,
        run_preflight_multiseed,
        verify_preregistration,
    )
    print(json.dumps({"stage": 15, "preregistration_hash": verify_preregistration()}), flush=True)
    prior_run = run_preflight_multiseed(workers)
    prior = analyze_preflight_multiseed(prior_run["path"])
    local_run = run_local_multiscale_7d(workers)
    local = analyze_local_multiscale_7d(local_run["path"])
    gate = evaluate_preflight_gate(prior, local)
    print(json.dumps({"prior": prior["summary"], "local": local["summary"], "gate": gate}, indent=2), flush=True)


def stage16(workers: int) -> None:
    from sleep_sbi.route3_7d_experiment import (
        analyze_bank_collisions,
        run_training_bank_4096,
        verify_preregistration,
    )
    print(json.dumps({"stage": 16, "preregistration_hash": verify_preregistration()}), flush=True)
    bank = run_training_bank_4096(workers)
    collisions = analyze_bank_collisions(bank["bank_path"], bank["split_path"])
    print(json.dumps({"manifest": bank["manifest"], "collisions": collisions}, indent=2), flush=True)


def stage17(_: int) -> None:
    from sleep_sbi.route3_7d_training import train_ensemble_7d, validation_member_checks
    from sleep_sbi.route3_7d_validation import freeze_heldout_criteria
    training = train_ensemble_7d()
    checks = validation_member_checks()
    criteria = freeze_heldout_criteria()
    print(json.dumps({
        "ensemble": training["summary"],
        "validation_checks": checks.groupby("member").mean(numeric_only=True).to_dict("index"),
        "heldout_criteria_sha256": criteria["sha256"],
    }, indent=2), flush=True)


def stage18(workers: int) -> None:
    from sleep_sbi.route3_7d_validation import (
        analyze_recovery_coverage,
        evaluate_decision,
        run_heldout_256,
        run_ppc_64,
        sample_heldout_posteriors,
        verify_heldout_criteria,
    )
    print(json.dumps({"stage": 18, "criteria_hash": verify_heldout_criteria()}), flush=True)
    heldout = run_heldout_256(workers)
    posterior = sample_heldout_posteriors(heldout["path"])
    recovery = analyze_recovery_coverage(posterior["path"])
    ppc = run_ppc_64(recovery, heldout["path"], workers)
    decision = evaluate_decision(recovery, ppc)
    print(json.dumps({
        "heldout": heldout["manifest"],
        "recovery": recovery["recovery"].to_dict("records"),
        "coverage": recovery["coverage"].query("estimator == 'ensemble'").to_dict("records"),
        "ppc": ppc["summary"],
        "decision": decision,
    }, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("stage15", "stage16", "stage17", "stage18"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    globals()[args.stage](args.workers)


if __name__ == "__main__":
    main()
