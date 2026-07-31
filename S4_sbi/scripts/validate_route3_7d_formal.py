"""Independent structural, artifact, and checksum validation for Route-3 7D."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys

import nbformat
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT / "S4_sbi" / "src", ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from sleep_sbi.route3_7d_experiment import (
    BANK_ROOT,
    HELDOUT_ROOT,
    HTML_ROOT,
    PARAMETER_NAMES_7D,
    PREFLIGHT_ROOT,
    RESULTS_ROOT,
    TRAINING_ROOT,
    audit_independence_from_8d,
    rate_feature_names,
    sha256_file,
    verify_preregistration,
)
from sleep_sbi.route3_7d_validation import verify_heldout_criteria


NOTEBOOK_NAMES = [
    "15_Route3_7D_Preregistration_and_Robustness.ipynb",
    "16_Route3_7D_4096_Simulation_Bank.ipynb",
    "17_Route3_7D_SNPE_Ensemble.ipynb",
    "18_Route3_7D_Heldout_Recovery_Coverage_PPC.ipynb",
]


def main() -> None:
    notebook_report = []
    for name in NOTEBOOK_NAMES:
        path = ROOT / "S4_sbi" / "notebooks" / name
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        for cell in code_cells:
            compile(cell.source, f"{name}:cell", "exec")
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        missing_execution = [
            index for index, cell in enumerate(code_cells)
            if cell.get("execution_count") is None
        ]
        display_outputs = sum(
            output.get("output_type") in {"display_data", "execute_result"}
            for cell in code_cells for output in cell.get("outputs", [])
        )
        html = HTML_ROOT / name.replace(".ipynb", ".html")
        html_text = html.read_text(encoding="utf-8")
        notebook_report.append({
            "notebook": name,
            "sha256": sha256_file(path),
            "code_cells": len(code_cells),
            "executed_code_cells": len(code_cells) - len(missing_execution),
            "missing_execution_counts": missing_execution,
            "error_outputs": len(errors),
            "display_or_figure_outputs": int(display_outputs),
            "nbformat_valid": True,
            "static_compile_valid": True,
            "html_path": str(html.relative_to(ROOT)),
            "html_size_bytes": html.stat().st_size,
            "html_contains_title": "<h1" in html_text and "Route-3" in html_text,
        })
        if errors or missing_execution or display_outputs == 0 or html.stat().st_size < 100_000:
            raise RuntimeError(f"notebook output validation failed for {name}")

    checksum_paths = [
        ROOT / "S4_sbi" / "configs" / "route3_7d_preregistered_v1.json",
        ROOT / "S4_sbi" / "artifacts" / "route3_7d_preregistered_v1.locked.json",
        ROOT / "S4_sbi" / "artifacts" / "route3_7d_preregistered_v1.sha256",
        *(ROOT / "S4_sbi" / "notebooks" / name for name in NOTEBOOK_NAMES),
        BANK_ROOT / "route3_7d_cortex_rate_14d_bank_4096.npz",
        BANK_ROOT / "split_and_scaling.npz",
        HELDOUT_ROOT / "go_criteria_locked.json",
        HELDOUT_ROOT / "dataset" / "route3_7d_heldout_256.npz",
        HELDOUT_ROOT / "posterior_samples" / "heldout_ensemble_samples.npz",
        HELDOUT_ROOT / "ppc" / "posterior_predictive_1024.npz",
        HELDOUT_ROOT / "ppc" / "prior_predictive_1024.npz",
        HELDOUT_ROOT / "formal_route3_7d_decision.json",
        *(HTML_ROOT / name.replace(".ipynb", ".html") for name in NOTEBOOK_NAMES),
    ]
    pd.DataFrame([
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in checksum_paths
    ]).to_csv(RESULTS_ROOT / "artifact_checksums.csv", index=False)

    json_paths = list(RESULTS_ROOT.rglob("*.json"))
    csv_paths = list(RESULTS_ROOT.rglob("*.csv"))
    npz_paths = list(RESULTS_ROOT.rglob("*.npz"))
    pt_paths = list(RESULTS_ROOT.rglob("*.pt"))
    for path in json_paths:
        json.loads(path.read_text(encoding="utf-8"))
    for path in csv_paths:
        pd.read_csv(path)
    object_arrays = []
    for path in npz_paths:
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                if data[key].dtype.kind == "O":
                    object_arrays.append(f"{path}:{key}")
    if object_arrays:
        raise RuntimeError(f"unexpected object arrays: {object_arrays[:10]}")
    for path in pt_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("preregistration_hash") != verify_preregistration():
            raise RuntimeError(f"checkpoint preregistration hash mismatch: {path}")

    with np.load(BANK_ROOT / "route3_7d_cortex_rate_14d_bank_4096.npz", allow_pickle=False) as bank:
        bank_shapes = {"theta": list(bank["theta"].shape), "theta_full_8d": list(bank["theta_full_8d"].shape), "x": list(bank["x"].shape)}
        training_seeds = np.asarray(bank["simulator_seed"], np.int64)
        assert bank_shapes == {"theta": [4096, 7], "theta_full_8d": [4096, 8], "x": [4096, 14]}
        assert len(np.unique(training_seeds)) == 4096
    with np.load(HELDOUT_ROOT / "dataset" / "route3_7d_heldout_256.npz", allow_pickle=False) as heldout:
        heldout_shapes = {"theta": list(heldout["theta"].shape), "x": list(heldout["x"].shape)}
        heldout_seeds = np.asarray(heldout["simulator_seed"], np.int64)
        assert heldout_shapes == {"theta": [256, 7], "x": [256, 14]}
        assert np.intersect1d(training_seeds, heldout_seeds).size == 0
    with np.load(HELDOUT_ROOT / "posterior_samples" / "heldout_ensemble_samples.npz", allow_pickle=False) as posterior:
        posterior_shape = list(posterior["samples"].shape)
        samples = np.asarray(posterior["samples"], float)
        assert posterior_shape == [256, 4096, 7]
        assert np.isfinite(samples).all()
        assert ((samples >= 0) & (samples <= 1)).all()
    ppc_shapes = {}
    ppc_seed_sets = []
    for role in ("posterior_predictive", "prior_predictive"):
        path = HELDOUT_ROOT / "ppc" / f"{role}_1024.npz"
        with np.load(path, allow_pickle=False) as data:
            ppc_shapes[role] = {"theta": list(data["theta"].shape), "x": list(data["x"].shape)}
            ppc_seed_sets.append(np.asarray(data["simulator_seed"], np.int64))
            assert data["theta"].shape == (1024, 7)
            assert data["x"].shape == (1024, 14)
            assert np.isfinite(data["x"]).all()
    all_seed_sets = [training_seeds, heldout_seeds, *ppc_seed_sets]
    seed_intersections = {}
    for first in range(len(all_seed_sets)):
        for second in range(first + 1, len(all_seed_sets)):
            seed_intersections[f"{first}_{second}"] = int(
                np.intersect1d(all_seed_sets[first], all_seed_sets[second]).size
            )
    if any(seed_intersections.values()):
        raise RuntimeError(f"cross-stage simulator seed overlap: {seed_intersections}")

    protected_hashes = {}
    for index in range(15):
        matches = list((ROOT / "S4_sbi" / "notebooks").glob(f"{index:02d}_*.ipynb"))
        for path in matches:
            protected_hashes[path.name] = sha256_file(path)

    figures = [
        path for path in RESULTS_ROOT.rglob("*.png") if path.stat().st_size > 10_000
    ]
    if len(figures) < 7:
        raise RuntimeError("too few non-empty scientific figures")
    decision = json.loads(
        (HELDOUT_ROOT / "formal_route3_7d_decision.json").read_text(encoding="utf-8")
    )
    independence = audit_independence_from_8d()["summary"]
    if (
        independence["full_8d_exact_duplicate_new_rows"]
        or independence["free_7d_exact_duplicate_new_rows"]
        or independence["simulator_seed_intersection_count"]
    ):
        raise RuntimeError(f"cross-experiment independence failed: {independence}")
    report = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "sys_executable": sys.executable,
            "sys_prefix": sys.prefix,
            "python": platform.python_version(),
            "kernel_id": "neurolib",
            "kernel_display_name": "Python (neurolib)",
            "versions": {
                name: metadata.version(name)
                for name in ("neurolib", "sbi", "torch", "numpy", "scipy", "pandas")
            },
        },
        "preregistration_hash": verify_preregistration(),
        "heldout_criteria_hash": verify_heldout_criteria(),
        "notebooks": notebook_report,
        "artifact_reload": {
            "json": len(json_paths), "csv": len(csv_paths),
            "npz": len(npz_paths), "pt": len(pt_paths),
            "unexpected_object_arrays": len(object_arrays), "all_passed": True,
        },
        "bank_shapes": bank_shapes,
        "heldout_shapes": heldout_shapes,
        "posterior_shape": posterior_shape,
        "ppc_shapes": ppc_shapes,
        "cross_stage_seed_intersections": seed_intersections,
        "cross_experiment_independence": independence,
        "parameter_names": list(PARAMETER_NAMES_7D),
        "feature_names": list(rate_feature_names()),
        "nonempty_png_figures": len(figures),
        "decision": decision["decision"],
        "protected_notebook_hashes_00_14": protected_hashes,
        "all_passed": True,
    }
    output = RESULTS_ROOT / "validation_report.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    environment_path = RESULTS_ROOT / "environment_report.json"
    environment_path.write_text(json.dumps(report["environment"], indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
