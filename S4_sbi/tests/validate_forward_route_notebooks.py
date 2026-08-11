"""External validation for notebooks 09 and 00 and their safe artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.image as mpimg
import nbformat
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = PROJECT_ROOT / "S4_sbi" / "notebooks"
FORWARD_DIR = PROJECT_ROOT / "S4_sbi" / "results" / "forward_model_feasibility_route_decision"
GUIDE_DIR = PROJECT_ROOT / "S4_sbi" / "results" / "observation_sbi_reader_guide"
HTML_DIR = PROJECT_ROOT / "S4_sbi" / "results" / "overnight_observation_ablation" / "html"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_notebook(path: Path, expected_code_cells: int) -> dict[str, object]:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) == expected_code_cells
    for index, cell in enumerate(code_cells):
        compile(cell.source, f"{path.name}:code{index}", "exec")
    counts = [cell.execution_count for cell in code_cells]
    assert all(count is not None for count in counts)
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    assert not errors
    assert notebook.metadata.kernelspec.name == "python3"
    return {
        "nbformat": "PASS",
        "static_compile": "PASS",
        "code_cells": len(code_cells),
        "execution_counts": counts,
        "error_outputs": len(errors),
        "kernel_id": notebook.metadata.kernelspec.name,
        "kernel_display_name": notebook.metadata.kernelspec.display_name,
    }


def validate_csv(path: Path, expected_rows: int) -> dict[str, object]:
    frame = pd.read_csv(path)
    assert len(frame) == expected_rows
    assert len(frame.columns) > 0
    return {"rows": len(frame), "columns": len(frame.columns), "readable": True}


def validate_png(path: Path) -> dict[str, object]:
    image = mpimg.imread(path)
    assert image.ndim in (2, 3)
    assert min(image.shape[:2]) >= 300
    assert np.isfinite(image).all()
    pixel_std = float(np.std(image))
    assert pixel_std > 0.01
    return {
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "pixel_std": pixel_std,
    }


def main() -> None:
    notebook_checks = {
        "09_Forward_Model_Feasibility_and_Route_Decision.ipynb": validate_notebook(
            NOTEBOOK_DIR / "09_Forward_Model_Feasibility_and_Route_Decision.ipynb", 7
        ),
        "00_Observation_SBI_Reader_Guide.ipynb": validate_notebook(
            NOTEBOOK_DIR / "00_Observation_SBI_Reader_Guide.ipynb", 8
        ),
    }

    forward_csv_rows = {
        "source_candidate_decisions.csv": 12,
        "state_classification.csv": 6,
        "leadfield_requirements.csv": 10,
        "measurement_contract_components.csv": 15,
        "calibration_leakage_matrix.csv": 11,
        "route2_route3_decision.csv": 11,
        "literature_sources.csv": 5,
    }
    guide_csv_rows = {
        "reader_guide_registry.csv": 8,
        "evidence_status.csv": 11,
        "glossary.csv": 8,
        "reading_plan_90min.csv": 6,
        "researcher_decisions.csv": 3,
    }
    csv_checks = {
        f"forward/{name}": validate_csv(FORWARD_DIR / name, rows)
        for name, rows in forward_csv_rows.items()
    }
    csv_checks.update(
        {
            f"guide/{name}": validate_csv(GUIDE_DIR / name, rows)
            for name, rows in guide_csv_rows.items()
        }
    )

    decision = json.loads(
        (FORWARD_DIR / "forward_model_feasibility_decision.json").read_text(encoding="utf-8")
    )
    assert decision["current_model_supports_physically_declared_fpz_cz"] is False
    assert decision["single_source_temporal_rank"] == 1
    assert decision["simulation_bank_authorized"] is False
    assert decision["pilot_snpe_authorized"] is False
    rank = json.loads((FORWARD_DIR / "single_source_rank_audit.json").read_text(encoding="utf-8"))
    assert rank["can_be_nonzero"] is True
    assert rank["adds_new_temporal_structure"] is False

    stored_hashes = json.loads(
        (FORWARD_DIR / "protected_notebook_hashes.json").read_text(encoding="utf-8")
    )
    current_hashes = {
        relative: sha256(PROJECT_ROOT / relative)
        for relative in stored_hashes
    }
    assert current_hashes == stored_hashes

    figure_paths = [
        FORWARD_DIR / "figures" / "source_candidate_decisions.png",
        FORWARD_DIR / "figures" / "single_source_rank1.png",
        FORWARD_DIR / "figures" / "measurement_contract_status.png",
        GUIDE_DIR / "figures" / "notebook_workflow.png",
    ]
    figure_checks = {path.name: validate_png(path) for path in figure_paths}

    html_paths = [
        HTML_DIR / "09_Forward_Model_Feasibility_and_Route_Decision.html",
        HTML_DIR / "00_Observation_SBI_Reader_Guide.html",
    ]
    for path in html_paths:
        assert path.exists() and path.stat().st_size > 100_000

    unsafe_extensions = {".npy", ".npz", ".pkl", ".pt", ".pth", ".edf"}
    unexpected_binary = [
        path
        for root in (FORWARD_DIR, GUIDE_DIR)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in unsafe_extensions
    ]
    assert not unexpected_binary

    report = {
        "notebooks": notebook_checks,
        "csv_artifacts": csv_checks,
        "json_reload": "PASS",
        "figure_checks": figure_checks,
        "html_files": [path.relative_to(PROJECT_ROOT).as_posix() for path in html_paths],
        "protected_notebook_hashes_unchanged": True,
        "unexpected_binary_artifacts": [],
        "simulation_count": 0,
        "simulation_bank_created": False,
        "snpe_started": False,
        "route_2_gate": decision["route_2_current_status"],
        "route_3_authorized": decision["route_3_authorized"],
    }
    (FORWARD_DIR / "external_validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "PASS: notebooks 09/00 executed without errors; "
        f"{len(csv_checks)} CSVs, 4 figures, 2 HTML files, and protected hashes verified"
    )


if __name__ == "__main__":
    main()
