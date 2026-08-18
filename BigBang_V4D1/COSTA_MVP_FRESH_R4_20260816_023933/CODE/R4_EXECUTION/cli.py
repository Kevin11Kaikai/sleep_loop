"""Command-line entrypoints for the fresh R4 execution route."""

from __future__ import annotations

import argparse
from pathlib import Path

from r4_adapter import steward
from r4_empirical import evaluate_heldout, select_fit
from r4_synthetic import run_synthetic


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="mode", required=True)

    steward_parser = subparsers.add_parser("steward")
    steward_parser.add_argument("--project-root", type=Path, required=True)
    steward_parser.add_argument("--output-root", type=Path, required=True)
    steward_parser.add_argument("--output-child", required=True)

    synthetic_parser = subparsers.add_parser("synthetic-m1")
    synthetic_parser.add_argument("--output-root", type=Path, required=True)
    synthetic_parser.add_argument("--output-child", required=True)

    fit_parser = subparsers.add_parser("select-fit")
    fit_parser.add_argument("--fit-payload", type=Path, required=True)
    fit_parser.add_argument("--output-root", type=Path, required=True)
    fit_parser.add_argument("--output-child", required=True)

    heldout_parser = subparsers.add_parser("evaluate-heldout")
    heldout_parser.add_argument("--heldout-payload", type=Path, required=True)
    heldout_parser.add_argument("--selection-freeze", type=Path, required=True)
    heldout_parser.add_argument("--expected-freeze-sha256", required=True)
    heldout_parser.add_argument("--output-root", type=Path, required=True)
    heldout_parser.add_argument("--output-child", required=True)
    return result


def main() -> int:
    arguments = parser().parse_args()
    if arguments.mode == "steward":
        directory = steward(arguments.project_root, arguments.output_root, arguments.output_child)
    elif arguments.mode == "synthetic-m1":
        directory = run_synthetic(arguments.output_root, arguments.output_child)
    elif arguments.mode == "select-fit":
        directory = select_fit(arguments.fit_payload, arguments.output_root, arguments.output_child)
    elif arguments.mode == "evaluate-heldout":
        directory = evaluate_heldout(
            arguments.heldout_payload,
            arguments.selection_freeze,
            arguments.expected_freeze_sha256,
            arguments.output_root,
            arguments.output_child,
        )
    else:
        raise AssertionError("unreachable mode")
    print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

