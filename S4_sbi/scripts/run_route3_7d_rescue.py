"""Command-line stages for the resumable Route-3 7D rescue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "S4_sbi" / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "diagnosis",
            "preregister",
            "additional-bank",
            "combined-bank",
            "development",
            "train-baseline",
            "train-wide",
            "sample-development",
            "select-primary",
            "fresh-final",
            "sample-final",
            "analyze-final",
            "ppc-final",
            "decide",
        ),
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    from sleep_sbi.route3_7d_rescue import (
        assert_seed_and_theta_disjointness,
        build_combined_group_safe_bank,
        create_rescue_preregistration,
        diagnose_original_failure,
        run_additional_training_bank,
        run_development_set,
        verify_rescue_preregistration,
    )

    if args.stage == "diagnosis":
        result = diagnose_original_failure()["summary"]
    elif args.stage == "preregister":
        result = {
            "payload": create_rescue_preregistration(),
            "hash": verify_rescue_preregistration(),
            "independence": assert_seed_and_theta_disjointness(),
        }
    elif args.stage == "additional-bank":
        result = run_additional_training_bank(args.workers)["manifest"]
    elif args.stage == "combined-bank":
        result = build_combined_group_safe_bank()["manifest"]
    elif args.stage == "development":
        result = run_development_set(args.workers)["manifest"]
    elif args.stage in ("train-baseline", "train-wide", "sample-development", "select-primary"):
        from sleep_sbi.route3_7d_rescue_training import (
            evaluate_and_select_development,
            sample_development,
            train_configuration,
        )

        if args.stage == "train-baseline":
            result = train_configuration("maf64_t5")
        elif args.stage == "train-wide":
            result = train_configuration("maf128_t8")
        elif args.stage == "sample-development":
            result = {
                config: str(sample_development(config))
                for config in ("maf64_t5", "maf128_t8")
            }
        else:
            result = evaluate_and_select_development()
    else:
        from sleep_sbi.route3_7d_rescue_validation import (
            analyze_fresh_final,
            evaluate_final_decision,
            run_final_ppc,
            run_fresh_final_1024,
            sample_fresh_final_posteriors,
        )

        if args.stage == "fresh-final":
            result = run_fresh_final_1024(args.workers)["manifest"]
        elif args.stage == "sample-final":
            result = {"path": str(sample_fresh_final_posteriors())}
        elif args.stage == "analyze-final":
            analysis = analyze_fresh_final()
            result = {
                "method": analysis["method"],
                "raw_recovery_rows": len(analysis["raw"]["recovery"]),
                "primary_recovery_rows": len(analysis["primary"]["recovery"]),
            }
        elif args.stage == "ppc-final":
            analysis = analyze_fresh_final()
            result = run_final_ppc(analysis["primary"], args.workers)["summary"]
        else:
            analysis = analyze_fresh_final()
            ppc_summary = json.loads(
                (
                    ROOT
                    / "S4_sbi"
                    / "results"
                    / "route3_7d_rescue"
                    / "fresh_final_validation"
                    / "ppc"
                    / "ppc_summary.json"
                ).read_text(encoding="utf-8")
            )
            result = evaluate_final_decision(
                analysis,
                {"summary": ppc_summary},
            )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
