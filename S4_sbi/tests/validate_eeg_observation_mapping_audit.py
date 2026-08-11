"""Post-execution validation for notebook 08 and its aggregate artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "S4_sbi" / "notebooks" / "08_EEG_Observation_Mapping_Audit.ipynb"
OUT = ROOT / "S4_sbi" / "results" / "eeg_observation_mapping_audit"
HTML = (
    ROOT
    / "S4_sbi"
    / "results"
    / "overnight_observation_ablation"
    / "html"
    / "08_EEG_Observation_Mapping_Audit.html"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(isinstance(cell.execution_count, int) for cell in code_cells)
    error_outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    assert not error_outputs

    csv_names = [
        "code_evidence.csv",
        "parameter_sets.csv",
        "simulation_run_summary.csv",
        "simulator_state_variable_catalog.csv",
        "proxy_range_statistics.csv",
        "proxy_shape_metrics.csv",
        "extractor_screening.csv",
        "mapping_option_decisions.csv",
    ]
    reloaded = {name: pd.read_csv(OUT / name) for name in csv_names}
    assert len(reloaded["parameter_sets.csv"]) == 3
    assert len(reloaded["simulation_run_summary.csv"]) == 3
    assert len(reloaded["simulator_state_variable_catalog.csv"]) == 19
    assert len(reloaded["proxy_range_statistics.csv"]) == 57
    assert reloaded["proxy_range_statistics.csv"]["finite"].all()
    assert len(reloaded["mapping_option_decisions.csv"]) == 8

    route = json.loads((OUT / "route_decision.json").read_text(encoding="utf-8"))
    environment = json.loads(
        (OUT / "environment_report.json").read_text(encoding="utf-8")
    )
    assert route["recommended_route"] == "Route 2"
    assert route["simulation_bank_authorized"] is False
    assert route["pilot_snpe_authorized"] is False
    assert environment["conda_default_env"] == "neurolib"
    assert "neurolib" in environment["sys_executable"].lower()
    for relative, digest in environment["protected_notebook_sha256_start"].items():
        assert _sha256(ROOT / relative) == digest

    figure_results = {}
    for image_path in sorted((OUT / "figures").glob("*.png")):
        with Image.open(image_path) as image:
            array = np.asarray(image.convert("L"))
            assert image.width >= 1000 and image.height >= 500
            assert float(np.std(array)) > 5.0
            figure_results[image_path.name] = {
                "width": image.width,
                "height": image.height,
                "pixel_std": float(np.std(array)),
            }
    assert len(figure_results) == 4

    html_text = HTML.read_text(encoding="utf-8")
    assert len(html_text) > 500_000
    assert "Recommendation: Route 2" in html_text
    assert "Traceback (most recent call last)" not in html_text

    report_path = OUT / "validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        {
            "external_validation_utc": datetime.now(timezone.utc).isoformat(),
            "nbformat_validation": "PASS",
            "executed_code_cell_count": len(code_cells),
            "all_code_cells_have_execution_count": True,
            "error_output_count": 0,
            "artifact_reload_validation": "PASS",
            "html_export_validation": "PASS",
            "html_path": (
                "S4_sbi/results/overnight_observation_ablation/html/"
                "08_EEG_Observation_Mapping_Audit.html"
            ),
            "figure_file_validation": figure_results,
        }
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    print(
        "PASS:",
        f"{len(code_cells)} executed code cells;",
        f"{len(csv_names)} CSV artifacts;",
        f"{len(figure_results)} PNG figures;",
        "HTML and protected hashes verified",
    )


if __name__ == "__main__":
    main()
