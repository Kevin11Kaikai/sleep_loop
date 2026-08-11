"""Command-line driver for resumable Route-3 pilot stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "S4_sbi" / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def run_stage11(workers: int) -> None:
    from sleep_sbi.route3_global_robustness import (
        GLOBAL_ROOT,
        analyze_local_multiscale,
        analyze_prior_multiseed,
        engineering_gate,
        run_local_multiscale,
        run_prior_multiseed,
    )

    prior = run_prior_multiseed(max_workers=workers)
    prior_analysis = analyze_prior_multiseed(prior["path"])
    local = run_local_multiscale(max_workers=workers)
    local_analysis = analyze_local_multiscale(local["path"])
    gate = engineering_gate(
        prior_analysis["summary"], local_analysis["summary"].iloc[0].to_dict()
    )
    path = GLOBAL_ROOT / "engineering_gate.json"
    path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps({"prior": prior_analysis["summary"], "local": local_analysis["summary"].iloc[0].to_dict(), "gate": gate}, indent=2))


def run_stage12(workers: int) -> None:
    from sleep_sbi.route3_global_robustness import (
        analyze_training_bank_collisions,
        run_training_bank,
    )

    bank = run_training_bank(max_workers=workers)
    collisions = analyze_training_bank_collisions(
        bank["bank_path"], bank["split_path"]
    )
    print(
        json.dumps(
            {"manifest": bank["manifest"], "collisions": collisions["summary"]},
            indent=2,
        )
    )


def run_stage13(_: int) -> None:
    from sleep_sbi.route3_global_robustness import RESULTS_ROOT
    from sleep_sbi.route3_pilot_snpe import (
        exploratory_member_checks,
        load_ensemble_members,
        train_ensemble,
        write_go_criteria,
    )

    bank_root = RESULTS_ROOT / "simulation_bank_2048"
    training = train_ensemble(
        bank_root / "route3_cortex_rate_14d_bank_2048.npz",
        bank_root / "split_and_scaling.npz",
    )
    members = load_ensemble_members(training["data"])
    checks = exploratory_member_checks(training["data"], members)
    checks.to_csv(
        RESULTS_ROOT / "exploratory_npe_ensemble" / "member_checks.csv", index=False
    )
    criteria = write_go_criteria(RESULTS_ROOT / "heldout_validation" / "go_criteria_frozen.json")
    print(
        json.dumps(
            {
                "ensemble": training["summary"],
                "member_checks": checks.groupby("member").mean(numeric_only=True).to_dict("index"),
                "go_criteria_version": criteria["version"],
            },
            indent=2,
        )
    )


def run_stage14(workers: int) -> None:
    from sleep_sbi.route3_global_robustness import RESULTS_ROOT
    from sleep_sbi.route3_heldout_validation import (
        HELDOUT_ROOT,
        analyze_recovery_and_coverage,
        evaluate_go_decision,
        generate_heldout_posterior_samples,
        run_heldout_dataset,
        run_synthetic_ppc,
    )

    bank_root = RESULTS_ROOT / "simulation_bank_2048"
    bank_path = bank_root / "route3_cortex_rate_14d_bank_2048.npz"
    split_path = bank_root / "split_and_scaling.npz"
    heldout = run_heldout_dataset(bank_path, max_workers=workers)
    posterior = generate_heldout_posterior_samples(
        heldout["path"], bank_path, split_path
    )
    recovery = analyze_recovery_and_coverage(posterior["path"])
    ppc = run_synthetic_ppc(
        recovery,
        heldout["path"],
        bank_path,
        split_path,
        max_workers=workers,
    )
    decision = evaluate_go_decision(
        HELDOUT_ROOT / "go_criteria_frozen.json",
        bank_root / "bank_manifest.json",
        HELDOUT_ROOT / "dataset" / "heldout_manifest.json",
        recovery,
        ppc,
    )
    print(
        json.dumps(
            {
                "heldout": heldout["manifest"],
                "recovery": recovery["recovery"].to_dict("records"),
                "coverage": recovery["coverage"][
                    recovery["coverage"]["estimator"] == "ensemble"
                ].to_dict("records"),
                "ppc": ppc["summary"],
                "decision": decision,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("stage11", "stage12", "stage13", "stage14"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.stage == "stage11":
        run_stage11(args.workers)
    elif args.stage == "stage12":
        run_stage12(args.workers)
    elif args.stage == "stage13":
        run_stage13(args.workers)
    elif args.stage == "stage14":
        run_stage14(args.workers)


if __name__ == "__main__":
    main()
