"""Terminal structural, provenance, and artifact validation for the 7D rescue."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "S4_sbi" / "src"))

from sleep_sbi.route3_7d_rescue import (  # noqa: E402
    FINAL_ROOT,
    ORIGINAL_ROOT,
    RESCUE_ROOT,
    verify_rescue_preregistration,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    notebooks = []
    for number in range(19, 24):
        path = next((ROOT / "S4_sbi" / "notebooks").glob(f"{number}_*.ipynb"))
        notebook = nbformat.read(path, 4)
        nbformat.validate(notebook)
        missing, errors = [], []
        streams = []
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            ast.parse(cell.source, filename=f"{path.name}:cell-{index}")
            if cell.execution_count is None:
                missing.append(index)
            for output in cell.get("outputs", []):
                if output.output_type == "error":
                    errors.append(
                        {"cell": index, "ename": output.ename, "evalue": output.evalue}
                    )
                if output.output_type == "stream":
                    streams.append(str(output.get("text", "")))
        stream_text = "\n".join(streams)
        notebooks.append(
            {
                "notebook": path.name,
                "sha256": sha(path),
                "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
                "missing_execution_counts": missing,
                "error_outputs": errors,
                "kernel": dict(notebook.metadata.kernelspec),
                "neurolib_executable_seen": (
                    "envs\\neurolib\\python.exe" in stream_text
                ),
                "conda_neurolib_seen": "CONDA_DEFAULT_ENV= neurolib" in stream_text,
            }
        )

    original_validation = json.loads(
        (ORIGINAL_ROOT / "validation_report.json").read_text(encoding="utf-8")
    )
    protected = []
    for row in original_validation["notebooks"]:
        path = ROOT / "S4_sbi" / "notebooks" / row["notebook"]
        protected.append(
            {
                "notebook": row["notebook"],
                "expected": row["sha256"],
                "observed": sha(path),
                "unchanged": sha(path) == row["sha256"],
            }
        )

    errors, objects = [], []
    json_files = sorted(RESCUE_ROOT.rglob("*.json"))
    csv_files = sorted(RESCUE_ROOT.rglob("*.csv"))
    npz_files = sorted(RESCUE_ROOT.rglob("*.npz"))
    pt_files = sorted(RESCUE_ROOT.rglob("*.pt"))
    for path in json_files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    for path in csv_files:
        try:
            pd.read_csv(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    for path in npz_files:
        try:
            with np.load(path, allow_pickle=False) as data:
                for key in data.files:
                    if data[key].dtype == object:
                        objects.append(f"{path}:{key}")
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    checkpoint_contract_errors = []
    for path in pt_files:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or "network_state" not in payload:
                checkpoint_contract_errors.append(str(path))
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    final_dataset = FINAL_ROOT / "dataset" / "route3_7d_rescue_fresh_final_1024.npz"
    posterior = FINAL_ROOT / "posterior_samples" / "fresh_final_posterior_samples.npz"
    with np.load(final_dataset, allow_pickle=False) as data:
        final_theta = np.asarray(data["theta"], float)
        final_x = np.asarray(data["x"], float)
        final_success = np.asarray(data["success"], bool)
        final_seeds = np.asarray(data["simulator_seed"], int)
    with np.load(posterior, allow_pickle=False) as data:
        raw = np.asarray(data["raw_samples"], float)
        primary = np.asarray(data["primary_samples"], float)

    seed_paths = {
        "additional": RESCUE_ROOT
        / "training_bank"
        / "additional_2048x2"
        / "route3_7d_rescue_additional_2048x2.npz",
        "development": RESCUE_ROOT
        / "development"
        / "dataset"
        / "route3_7d_rescue_development_512.npz",
        "final": final_dataset,
        "final_ppc_posterior": FINAL_ROOT / "ppc" / "posterior_predictive_1024.npz",
        "final_ppc_prior": FINAL_ROOT / "ppc" / "prior_predictive_1024.npz",
    }
    seed_sets = {}
    for name, path in seed_paths.items():
        with np.load(path, allow_pickle=False) as data:
            seed_sets[name] = np.asarray(data["simulator_seed"], int)
    intersections = {}
    names = list(seed_sets)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            intersections[f"{first}_vs_{second}"] = int(
                np.intersect1d(seed_sets[first], seed_sets[second]).size
            )

    html_rows = []
    for path in sorted((RESCUE_ROOT / "html").glob("*.html")):
        text = path.read_text(encoding="utf-8")
        html_rows.append(
            {
                "html": path.name,
                "size_bytes": path.stat().st_size,
                "contains_traceback": "Traceback (most recent call last)" in text,
                "contains_neurolib_executable": "envs\\neurolib\\python.exe" in text,
            }
        )
    figure_rows = []
    for path in sorted((RESCUE_ROOT / "figures").glob("*.png")):
        array = np.asarray(Image.open(path).convert("RGB"), dtype=float)
        figure_rows.append(
            {
                "figure": path.name,
                "width": int(array.shape[1]),
                "height": int(array.shape[0]),
                "pixel_std": float(array.std()),
                "nonblank": bool(array.std() > 5),
            }
        )

    decision_md = ROOT / "ROUTE3_7D_RESCUE_FINAL_DECISION.md"
    decision_json = ROOT / "ROUTE3_7D_RESCUE_FINAL_DECISION.json"
    decision = json.loads(decision_json.read_text(encoding="utf-8"))
    checks = {
        "notebooks_valid_executed_neurolib": all(
            not row["missing_execution_counts"]
            and not row["error_outputs"]
            and row["kernel"]["name"] == "neurolib"
            and row["neurolib_executable_seen"]
            and row["conda_neurolib_seen"]
            for row in notebooks
        ),
        "protected_15_18_unchanged": all(row["unchanged"] for row in protected),
        "artifact_reload": not errors and not objects and not checkpoint_contract_errors,
        "fresh_final_shapes": (
            final_theta.shape == (1024, 7)
            and final_x.shape == (1024, 14)
            and raw.shape == (1024, 4096, 7)
            and primary.shape == (1024, 4096, 7)
        ),
        "fresh_final_finite": bool(
            final_success.all()
            and np.isfinite(final_x).all()
            and np.isfinite(raw).all()
            and np.isfinite(primary).all()
            and ((raw >= 0) & (raw <= 1)).all()
            and ((primary >= 0) & (primary <= 1)).all()
        ),
        "seed_sets_disjoint": not any(intersections.values()),
        "html_exports_valid": (
            len(html_rows) == 5
            and all(
                row["size_bytes"] > 100_000
                and not row["contains_traceback"]
                and row["contains_neurolib_executable"]
                for row in html_rows
            )
        ),
        "figures_nonblank": len(figure_rows) >= 6
        and all(row["nonblank"] for row in figure_rows),
        "decision_markdown_starts_with_verdict": (
            decision_md.read_text(encoding="utf-8").splitlines()[0]
            == f"# {decision['verdict']}"
        ),
        "rescue_lock_valid": len(verify_rescue_preregistration()) == 64,
    }
    report = {
        "validated_utc": pd.Timestamp.utcnow().isoformat(),
        "environment": {
            "sys_executable": sys.executable,
            "sys_prefix": sys.prefix,
            "python": sys.version,
            "kernel_id": "neurolib",
        },
        "rescue_preregistration_hash": verify_rescue_preregistration(),
        "decision": decision["verdict"],
        "checks": checks,
        "notebooks": notebooks,
        "protected_notebooks_15_18": protected,
        "artifact_reload": {
            "json": len(json_files),
            "csv": len(csv_files),
            "npz": len(npz_files),
            "pt": len(pt_files),
            "errors": errors,
            "object_arrays": objects,
            "checkpoint_contract_errors": checkpoint_contract_errors,
        },
        "final_shapes": {
            "theta": list(final_theta.shape),
            "x": list(final_x.shape),
            "raw_posterior": list(raw.shape),
            "primary_posterior": list(primary.shape),
        },
        "final_simulator_seed_count": len(np.unique(final_seeds)),
        "cross_rescue_seed_intersections": intersections,
        "html": html_rows,
        "figures": figure_rows,
        "git_status_short": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).splitlines(),
        "pass": all(checks.values()),
    }
    path = RESCUE_ROOT / "terminal_validation_report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "checks": checks}, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
