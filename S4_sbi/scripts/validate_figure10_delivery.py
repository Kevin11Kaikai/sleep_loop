"""Independent artifact, notebook, image, and link audit for Figure-10."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

import nbformat
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
S4_ROOT = PROJECT_ROOT / "S4_sbi"
NOTEBOOK_ROOT = S4_ROOT / "notebooks"
RESULTS_ROOT = S4_ROOT / "results" / "figure10_8d_7d"
ARTIFACT_ROOT = RESULTS_ROOT / "validation"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_protected_notebooks(path: Path) -> dict[str, Any]:
    rows = []
    for number in range(24):
        for notebook in sorted(NOTEBOOK_ROOT.glob(f"{number:02d}_*.ipynb")):
            rows.append(
                {
                    "path": notebook.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": sha256_file(notebook),
                    "size_bytes": notebook.stat().st_size,
                    "reader_guide_authorized_to_change": number == 0,
                }
            )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "notebooks": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _validate_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        objects = [key for key in data.files if data[key].dtype == object]
        shapes = {key: list(data[key].shape) for key in data.files}
    if objects:
        raise RuntimeError(f"{path} has object arrays: {objects}")
    return {"path": str(path), "arrays": len(shapes), "shapes": shapes}


def _validate_notebook(path: Path) -> dict[str, Any]:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    errors = []
    missing = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        stripped = cell.source.lstrip()
        if stripped and not stripped.startswith(("%", "!", "%%")):
            compile(cell.source, f"{path.name}:cell-{index}", "exec")
        if stripped and cell.execution_count is None:
            missing.append(index)
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                errors.append(
                    {
                        "cell": index,
                        "ename": output.get("ename"),
                        "evalue": output.get("evalue"),
                    }
                )
    if errors or missing:
        raise RuntimeError(
            f"{path.name}: errors={errors}, missing execution={missing}"
        )
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
        "markdown_cells": sum(
            c.cell_type == "markdown" for c in notebook.cells
        ),
        "kernel": notebook.metadata.get("kernelspec", {}),
    }


def _validate_relative_links(markdown: str, base: Path) -> list[str]:
    checked = []
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", markdown):
        if target.startswith(("http://", "https://", "#")):
            continue
        clean = target.split("#", 1)[0]
        if not clean:
            continue
        resolved = (base / clean).resolve()
        if not resolved.exists():
            raise RuntimeError(f"broken relative link: {target} -> {resolved}")
        checked.append(str(resolved))
    return checked


def final_audit() -> dict[str, Any]:
    before_path = ARTIFACT_ROOT / "protected_notebook_hashes_before.json"
    before = json.loads(before_path.read_text(encoding="utf-8"))
    protected_changes = []
    for row in before["notebooks"]:
        path = PROJECT_ROOT / row["path"]
        current = sha256_file(path)
        if (
            current != row["sha256"]
            and not row["reader_guide_authorized_to_change"]
        ):
            protected_changes.append(
                {
                    "path": row["path"],
                    "before": row["sha256"],
                    "after": current,
                }
            )
    if protected_changes:
        raise RuntimeError(
            f"protected evidentiary notebooks changed: {protected_changes}"
        )
    json_files = sorted(RESULTS_ROOT.rglob("*.json"))
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))
    csv_files = sorted(RESULTS_ROOT.rglob("*.csv"))
    for path in csv_files:
        pd.read_csv(path)
    npz_rows = [_validate_npz(path) for path in sorted(RESULTS_ROOT.rglob("*.npz"))]
    image_rows = []
    for path in sorted((RESULTS_ROOT / "figures").glob("*.png")):
        with Image.open(path) as image:
            extrema = image.convert("L").getextrema()
            if image.width < 500 or image.height < 300 or extrema[0] == extrema[1]:
                raise RuntimeError(f"blank or undersized figure: {path}")
            image_rows.append(
                {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "width": image.width,
                    "height": image.height,
                    "grayscale_extrema": list(extrema),
                }
            )
        for suffix in (".svg", ".pdf"):
            companion = path.with_suffix(suffix)
            if not companion.exists() or companion.stat().st_size < 1000:
                raise RuntimeError(f"missing figure companion: {companion}")
    notebook_rows = []
    for number in range(24, 33):
        matches = sorted(NOTEBOOK_ROOT.glob(f"{number:02d}_Figure10*.ipynb"))
        if len(matches) != 1:
            raise RuntimeError(f"notebook {number}: found {matches}")
        notebook_rows.append(_validate_notebook(matches[0]))
    guide = NOTEBOOK_ROOT / "00_Observation_SBI_Reader_Guide.ipynb"
    guide_row = _validate_notebook(guide)
    guide_nb = nbformat.read(guide, as_version=4)
    guide_markdown = "\n".join(
        cell.source for cell in guide_nb.cells if cell.cell_type == "markdown"
    )
    links = _validate_relative_links(guide_markdown, guide.parent)
    html_files = []
    for number in list(range(24, 33)) + [0]:
        prefix = f"{number:02d}_"
        matches = sorted((RESULTS_ROOT / "html").glob(f"{prefix}*.html"))
        if len(matches) != 1 or matches[0].stat().st_size < 10_000:
            raise RuntimeError(f"HTML export missing for notebook {number}")
        html_files.append(
            {
                "path": matches[0].relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": matches[0].stat().st_size,
            }
        )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "protected_notebook_changes": protected_changes,
        "json_files_reloaded": len(json_files),
        "csv_files_reloaded": len(csv_files),
        "npz_files_reloaded": len(npz_rows),
        "figures_checked": image_rows,
        "notebooks_checked": notebook_rows,
        "reader_guide": guide_row,
        "reader_guide_relative_links_checked": len(links),
        "html_checked": html_files,
        "status": "pass",
    }
    output = ARTIFACT_ROOT / "final_delivery_validation.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) != 2 or sys.argv[1] not in {"snapshot", "final"}:
        raise SystemExit("usage: validate_figure10_delivery.py snapshot|final")
    if sys.argv[1] == "snapshot":
        payload = snapshot_protected_notebooks(
            ARTIFACT_ROOT / "protected_notebook_hashes_before.json"
        )
    else:
        payload = final_audit()
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
