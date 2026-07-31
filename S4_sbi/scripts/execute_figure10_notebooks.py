"""Execute, validate, and export the Figure-10 reader-facing notebooks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from traitlets.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "S4_sbi" / "notebooks"
RESULTS_ROOT = PROJECT_ROOT / "S4_sbi" / "results" / "figure10_8d_7d"
HTML_ROOT = RESULTS_ROOT / "html"


def _ordinary_python(source: str) -> bool:
    stripped = source.lstrip()
    return not stripped.startswith(("%", "!", "%%"))


def execute_one(path: Path) -> dict[str, object]:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code" and _ordinary_python(cell.source):
            compile(cell.source, f"{path.name}:cell-{index}", "exec")
    client = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="neurolib",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
        record_timing=True,
    )
    executed = client.execute()
    errors = []
    missing_counts = []
    figures = 0
    tables = 0
    for index, cell in enumerate(executed.cells):
        if cell.cell_type != "code":
            continue
        if cell.source.strip() and cell.execution_count is None:
            missing_counts.append(index)
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                errors.append(
                    {
                        "cell": index,
                        "ename": output.get("ename"),
                        "evalue": output.get("evalue"),
                    }
                )
            data = output.get("data", {})
            figures += int("image/png" in data or "image/svg+xml" in data)
            tables += int("text/html" in data)
    if errors or missing_counts:
        raise RuntimeError(
            f"{path.name}: errors={errors}, missing_counts={missing_counts}"
        )
    nbformat.validate(executed)
    nbformat.write(executed, path)
    HTML_ROOT.mkdir(parents=True, exist_ok=True)
    config = Config()
    config.HTMLExporter.exclude_input_prompt = False
    config.HTMLExporter.exclude_output_prompt = False
    exporter = HTMLExporter(config=config)
    body, _ = exporter.from_notebook_node(executed)
    html_path = HTML_ROOT / f"{path.stem}.html"
    html_path.write_text(body, encoding="utf-8")
    if html_path.stat().st_size < 10_000:
        raise RuntimeError(f"{html_path} is unexpectedly small")
    return {
        "notebook": path.relative_to(PROJECT_ROOT).as_posix(),
        "html": html_path.relative_to(PROJECT_ROOT).as_posix(),
        "code_cells": sum(cell.cell_type == "code" for cell in executed.cells),
        "markdown_cells": sum(
            cell.cell_type == "markdown" for cell in executed.cells
        ),
        "figures_in_outputs": figures,
        "html_tables_in_outputs": tables,
        "errors": 0,
        "missing_execution_counts": 0,
        "kernelspec": executed.metadata.get("kernelspec", {}),
    }


def main() -> None:
    if os.environ.get("CONDA_DEFAULT_ENV") != "neurolib":
        raise RuntimeError("must execute from the neurolib conda environment")
    if "neurolib" not in str(Path(sys.executable)).lower():
        raise RuntimeError(f"wrong Python executable: {sys.executable}")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "numbers",
        nargs="*",
        type=int,
        default=list(range(24, 33)),
    )
    args = parser.parse_args()
    rows = []
    for number in args.numbers:
        matches = sorted(NOTEBOOK_ROOT.glob(f"{number:02d}_Figure10*.ipynb"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one Figure10 notebook for {number}, found {matches}"
            )
        print(f"Executing {matches[0].name}", flush=True)
        rows.append(execute_one(matches[0]))
    report = {
        "environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "sys_executable": sys.executable,
        "notebooks": rows,
    }
    path = RESULTS_ROOT / "notebook_execution_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
