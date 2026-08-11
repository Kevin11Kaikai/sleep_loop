"""Build reader-facing notebooks for the matched Figure-10 experiment."""

from __future__ import annotations

from pathlib import Path
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "S4_sbi" / "notebooks"


def _metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": "neurolib",
            "language": "python",
            "name": "neurolib",
        },
        "language_info": {"name": "python", "version": "3.10.20"},
    }


def _write_notebook(name: str, cells: list) -> Path:
    path = NOTEBOOK_ROOT / name
    notebook = new_notebook(cells=cells, metadata=_metadata())
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    return path


def _prelude() -> str:
    return """from pathlib import Path
import json, os, sys
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

cwd = Path.cwd().resolve()
PROJECT_ROOT = cwd if (cwd / "S4_sbi").exists() else cwd.parent.parent
SRC = PROJECT_ROOT / "S4_sbi" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
RESULTS = PROJECT_ROOT / "S4_sbi" / "results" / "figure10_8d_7d"
assert os.environ.get("CONDA_DEFAULT_ENV") == "neurolib"
print("environment:", os.environ.get("CONDA_DEFAULT_ENV"))
print("python:", sys.executable)
print("results:", RESULTS)"""


def build_protocol_notebook() -> Path:
    path = (
        NOTEBOOK_ROOT
        / "24_Figure10_8D_7D_Protocol_Preregistration.ipynb"
    )
    cells = [
        new_markdown_cell(
            """# 8D-primary / matched-7D Figure-10 protocol

This notebook records the protocol **before** any new scientific training bank
or final diagnostic is generated. The primary track infers all eight
mechanistic parameters. The matched 7D track fixes `c_ctx2th` at the historical
Route-3 value and is an ablation, not a replacement for the 8D model.

The observable is the frozen 14D summary of synthetic cortical excitatory and
inhibitory population firing rates. It is **not** scalp EEG."""
        ),
        new_markdown_cell(
            """## Environment and immutable evidence

The following cell proves that execution uses the `neurolib` environment and
verifies the locked JSON SHA-256. A kernel label alone would not be sufficient
evidence."""
        ),
        new_code_cell(
            """from pathlib import Path
import json, os, platform, sys
import importlib.metadata as metadata

cwd = Path.cwd().resolve()
PROJECT_ROOT = cwd if (cwd / "S4_sbi").exists() else cwd.parent.parent
SRC = PROJECT_ROOT / "S4_sbi" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sleep_sbi.figure10_protocol import (
    CONFIG_PATH, HUMAN_PATH, LOCKED_PATH, verify_preregistration
)

environment = {
    "working_directory": str(cwd),
    "project_root": str(PROJECT_ROOT),
    "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
    "sys.executable": sys.executable,
    "sys.prefix": sys.prefix,
    "python": sys.version,
    "platform": platform.platform(),
    "versions": {
        name: metadata.version(name)
        for name in ("neurolib", "sbi", "torch", "numpy", "scipy", "pandas", "nbformat")
    },
    "locked_preregistration_sha256": verify_preregistration(),
}
environment"""
        ),
        new_markdown_cell(
            """## Four-contract audit

Historical 8D evidence, the frozen historical 7D strict experiment, the new
primary 8D evaluation, and the new matched 7D ablation remain separate. Array
shape alone never authorizes bank reuse."""
        ),
        new_code_cell(
            """import pandas as pd
payload = json.loads(LOCKED_PATH.read_text(encoding="utf-8"))
contract_audit = pd.DataFrame([
    {
        "Contract": "Historical 8D Route-3",
        "Dimension": 8,
        "Parameters": "stored exact 8D order",
        "Observation": "cortex-rate 14D",
        "Prior": "historical frozen 8D",
        "Status": "prior evidence",
    },
    {
        "Contract": "Historical 7D Route-3",
        "Dimension": 7,
        "Parameters": "c_ctx2th fixed",
        "Observation": "cortex-rate 14D",
        "Prior": "historical frozen 7D",
        "Status": "strict NO-GO",
    },
    {
        "Contract": "New 8D Figure-10",
        "Dimension": 8,
        "Parameters": ", ".join(payload["contracts"]["primary_8d"]["parameter_names"]),
        "Observation": "frozen cortex-rate 14D",
        "Prior": "current documented 8D box",
        "Status": "primary",
    },
    {
        "Contract": "New matched 7D Figure-10",
        "Dimension": 7,
        "Parameters": ", ".join(payload["contracts"]["matched_7d"]["parameter_names"]),
        "Observation": "same frozen cortex-rate 14D",
        "Prior": "same seven marginals",
        "Status": "ablation",
    },
])
contract_audit"""
        ),
        new_markdown_cell(
            """## Official reference mapping

The official Practical Guide repository is pinned to a commit and notebook
hashes. Its 3-million-simulation pyloric workflow supplies the diagnostic
organization, not our simulator or scientific semantics."""
        ),
        new_code_cell(
            """official = payload["official_reference"]
print("Official repository:", official["repository"])
print("Pinned commit:", official["commit"])
print("Official sbi:", official["sbi_version"], "| local sbi:", environment["versions"]["sbi"])
pd.DataFrame(payload["protocol_mapping"])"""
        ),
        new_markdown_cell(
            """## Measured resource boundary

The final scale is based on a paired throughput benchmark, not convenience.
The official 3M target is retained as a reference; the smaller local scale is
an explicit resource deviation and limits the strength of any PASS."""
        ),
        new_code_cell(
            """resource = payload["resource_decision"]
projection = pd.DataFrame(resource["projections"]).T
projection.index.name = "valid simulations per track"
display(projection)
print(resource["hard_limitation"])
print("Frozen final scale per track:", resource["frozen_final_scale_per_track"])
print("Frozen intermediate scale:", resource["frozen_intermediate_scale"])"""
        ),
        new_markdown_cell(
            """## Frozen model, training, and diagnostics

The 8D and 7D tracks use the same seven shared Sobol coordinates, simulator
seed schedule, simulator, 14D extractor, architecture family, and diagnostic
sizes. Only `c_ctx2th` changes from inferred to fixed."""
        ),
        new_code_cell(
            """display(pd.DataFrame({
    "8D parameter": payload["contracts"]["primary_8d"]["parameter_names"],
    "lower": [v[0] for v in payload["contracts"]["primary_8d"]["prior_bounds"]],
    "upper": [v[1] for v in payload["contracts"]["primary_8d"]["prior_bounds"]],
}))
print("Fixed 7D c_ctx2th:", payload["contracts"]["matched_7d"]["fixed_parameter"]["value"])
print("14D order:")
display(pd.DataFrame({"index": range(14), "feature": payload["contracts"]["observation"]["feature_names"]}))
print("NSF:", payload["training"]["architecture"])
print("Training:", payload["training"]["policy"])
print("Diagnostics:", payload["diagnostics"])"""
        ),
        new_markdown_cell(
            """## Three verdict systems

1. The 8D Figure-10-equivalent verdict evaluates the raw 8D NSF ensemble.
2. The matched 7D verdict uses the same philosophy but is conditional on fixed
   `c_ctx2th`.
3. The original Route-3 strict verdict remains `NO-GO`; it is not changed by a
   later Figure-10-style result.

The numerical operationalization is project-specific and frozen before final
results. A broad but calibrated marginal is not a failure."""
        ),
        new_code_cell(
            """display(pd.DataFrame([
    {"system": "8D Figure-10-equivalent", "possible": "PASS / QUALIFIED PASS / FAIL", "current": "not opened"},
    {"system": "7D Figure-10-equivalent", "possible": "PASS / QUALIFIED PASS / FAIL", "current": "not opened"},
    {"system": "Original Route-3 strict", "possible": "FORMAL GO / CONDITIONAL GO / NO-GO", "current": payload["historical_route3"]["verdict"]},
]))
payload["operational_verdict"]"""
        ),
        new_markdown_cell(
            """## Claim boundary

Even a successful terminal result establishes only synthetic posterior
inference in cortical-rate proxy space. It does not establish a cortical
source-to-Fpz–Cz forward model, real-EEG inference, subject-specific parameter
recovery, or a digital twin."""
        ),
    ]
    return _write_notebook(path.name, cells)


def build_bank_notebook() -> Path:
    cells = [
        new_markdown_cell(
            """# Matched 8D and 7D simulation banks

This notebook audits the immutable, paired simulation banks used by the new
Figure-10-equivalent experiment. The 8D and 7D rows share the first seven
Sobol coordinates and simulator-seed schedule; only `c_ctx2th` is inferred in
8D and fixed in 7D. The signal is a **synthetic cortical-rate observable**, not
scalp EEG."""
        ),
        new_code_cell(_prelude()),
        new_markdown_cell(
            """## Resource decision and checkpoint evidence

The official three-million scale could not be completed on this CPU within a
bounded reproducible run. The preregistered final scale is therefore 32,768
valid rows per track, supported by a measured throughput projection."""
        ),
        new_code_cell(
            """benchmark = json.loads((RESULTS / "resource_benchmark" / "resource_benchmark.json").read_text())
manifest = json.loads((RESULTS / "matched_banks" / "matched_bank_manifest.json").read_text())
display(pd.DataFrame(benchmark["projections"]).T)
display(pd.Series(manifest, name="matched bank manifest"))
assert manifest["8d_valid"] == manifest["target_per_track"]
assert manifest["7d_valid"] == manifest["target_per_track"]
assert manifest["paired_valid"] == manifest["target_per_track"]"""
        ),
        new_markdown_cell(
            """## Contract and paired-row checks

These assertions reject silent parameter-order drift, nonfinite summaries,
object arrays, unmatched shared coordinates, and reuse of a fixed nuisance
seed."""
        ),
        new_code_cell(
            """bank8_path = RESULTS / manifest["bank_paths"]["8d"]
bank7_path = RESULTS / manifest["bank_paths"]["7d"]
with np.load(bank8_path, allow_pickle=False) as b8, np.load(bank7_path, allow_pickle=False) as b7:
    audit = {
        "8D theta": b8["theta"].shape,
        "7D theta": b7["theta"].shape,
        "8D x": b8["x"].shape,
        "7D x": b7["x"].shape,
        "8D finite": bool(np.isfinite(b8["x"]).all()),
        "7D finite": bool(np.isfinite(b7["x"]).all()),
        "shared theta exact": bool(np.array_equal(b8["theta"][:, :7], b7["theta"])),
        "seed schedule exact": bool(np.array_equal(b8["simulator_seed"], b7["simulator_seed"])),
        "distinct simulator seeds": int(np.unique(b8["simulator_seed"]).size),
        "feature order match": bool(np.array_equal(b8["feature_names"], b7["feature_names"])),
        "object arrays": [key for key in b8.files if b8[key].dtype == object] + [key for key in b7.files if b7[key].dtype == object],
    }
display(pd.Series(audit))
assert all([audit["8D finite"], audit["7D finite"], audit["shared theta exact"], audit["seed schedule exact"], audit["feature order match"]])
assert not audit["object arrays"]"""
        ),
        new_markdown_cell(
            """## Interpretation

This establishes a matched engineering comparison. It does not establish
parameter identifiability, posterior calibration, or a source-to-Fpz-Cz
measurement model. Those claims require the later diagnostics."""
        ),
    ]
    return _write_notebook("25_Figure10_Matched_Simulation_Banks.ipynb", cells)


def _training_notebook(track: str, number: int, title: str) -> Path:
    cells = [
        new_markdown_cell(
            f"""# {title}

Five independently initialized neural spline-flow NPEs are trained on the same
frozen `{track.upper()}` bank. They are combined as an equal-weight mixture:
component samples are pooled, never averaged, and mixture log density uses
`logsumexp(log q_k) - log(5)`."""
        ),
        new_code_cell(_prelude()),
        new_markdown_cell(
            """## Independent members and training-only preprocessing

The following table verifies independent initialization seeds, the shared
train/validation contract, early stopping, and best validation losses. Feature
scaling was fitted on training rows only."""
        ),
        new_code_cell(
            f"""root = RESULTS / "training" / "{track}" / "scale_32768"
manifest = json.loads((root / "ensemble_manifest.json").read_text())
summary = pd.read_csv(root / "training_summary.csv")
display(summary)
assert len(manifest["member_seeds"]) == 5
assert len(set(manifest["member_seeds"])) == 5
assert manifest["weights"] == [0.2] * 5
assert all((root / f"member_{{seed}}" / "best_checkpoint.pt").exists() for seed in manifest["member_seeds"])
print("mixture rule:", manifest["mixture_rule"])
print("train/validation:", manifest["training_rows"], manifest["validation_rows"])"""
        ),
        new_markdown_cell(
            """## Learning curves

Validation loss selects checkpoints; no final diagnostic dataset is used for
training or model selection."""
        ),
        new_code_cell(
            """import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 4.5))
for seed in manifest["member_seeds"]:
    history = pd.read_csv(root / f"member_{seed}" / "training_history.csv")
    ax.plot(history.epoch, history.validation_loss, label=f"member {seed}")
ax.set(xlabel="Epoch", ylabel="Validation negative log density", title="Independent NSF validation histories")
ax.legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout()
display(fig)"""
        ),
        new_markdown_cell(
            """## Scope

A decreasing validation loss is an engineering result, not scientific
validation. Joint expected coverage, marginal SBC, L-C2ST, and new-simulation
PPC determine whether this posterior is usable."""
        ),
    ]
    return _write_notebook(
        f"{number:02d}_Figure10_{track.upper()}_Five_NSF_"
        + ("Ensemble.ipynb" if track == "8d" else "Matched_Baseline.ipynb"),
        cells,
    )


def build_global_notebook() -> Path:
    cells = [
        new_markdown_cell(
            """# Joint expected coverage and marginal SBC

The official-style 300-case dataset is kept separate from the powered
1,024-case sensitivity set. Joint HPD ranks use posterior density, while
marginal SBC ranks test each inferred coordinate. Individual networks and the
raw equal-weight ensemble are reported separately."""
        ),
        new_code_cell(_prelude()),
        new_markdown_cell(
            """## Machine-readable summaries

`holm_clear_issue` is a project reproducibility operationalization, not an
official Figure 10 threshold. Visual rank curves remain the primary
official-style presentation."""
        ),
        new_code_cell(
            """rows = []
sbc = []
for dataset_id in ("official_300", "powered_1024"):
    for track in ("8d", "7d"):
        root = RESULTS / "diagnostics" / "global" / dataset_id / track / "scale_32768"
        frame = pd.read_csv(root / "global_summary.csv")
        frame["analysis"] = dataset_id
        rows.append(frame)
        detail = pd.read_csv(root / "marginal_sbc_summary.csv")
        detail["analysis"] = dataset_id
        sbc.append(detail)
global_table = pd.concat(rows, ignore_index=True)
sbc_table = pd.concat(sbc, ignore_index=True)
display(global_table)
display(sbc_table[(sbc_table.analysis == "powered_1024") & (sbc_table.estimator == "ensemble")])"""
        ),
        new_markdown_cell(
            """## Figure

The diagonal is ideal calibration. Systematic curvature indicates
miscalibration; a broad marginal can still be calibrated. The shaded region is
the finite-sample uncertainty band for the powered analysis."""
        ),
        new_code_cell(
            """figure = RESULTS / "figures" / "figure10_expected_coverage_sbc_8d_7d.png"
assert figure.exists() and figure.stat().st_size > 1000
display(Image(filename=str(figure)))"""
        ),
    ]
    return _write_notebook("28_Figure10_Joint_Coverage_and_SBC.ipynb", cells)


def build_lc2st_notebook() -> Path:
    cells = [
        new_markdown_cell(
            """# Local L-C2ST at a frozen synthetic observation

The primary observation was fixed before the diagnostic and has
`c_ctx2th` equal to the matched 7D value, so both models are correctly
specified for this comparison. Each track uses 20,000 additional independent
prior-predictive simulations. Null rejection indicates a local posterior
mismatch; non-rejection is not proof of physiological validity."""
        ),
        new_code_cell(_prelude()),
        new_code_cell(
            """manifest = json.loads((RESULTS / "diagnostics" / "ppc" / "primary_observation_manifest.json").read_text())
display(pd.Series(manifest))
tables = []
for track in ("8d", "7d"):
    frame = pd.read_csv(RESULTS / "diagnostics" / "lc2st" / track / "lc2st_summary.csv")
    tables.append(frame)
display(pd.concat(tables, ignore_index=True))"""
        ),
        new_markdown_cell(
            """## Individual networks versus ensemble

Bars show observed classifier statistics; black markers show the 95% null
threshold. The ensemble must be interpreted as an equal mixture, not a product
of member densities."""
        ),
        new_code_cell(
            """figure = RESULTS / "figures" / "figure10_lc2st_8d_7d.png"
assert figure.exists() and figure.stat().st_size > 1000
display(Image(filename=str(figure)))"""
        ),
    ]
    return _write_notebook("29_Figure10_Local_LC2ST.ipynb", cells)


def build_ppc_structure_notebook() -> Path:
    cells = [
        new_markdown_cell(
            """# Posterior predictive checks and posterior structure

PPC uses new simulator seeds and compares the frozen 14D cortical-rate
summaries against prior predictive baselines. It checks predictive adequacy,
not calibration. Posterior structure then asks whether uncertainty is broad,
contracted, or organized into compensatory ridges."""
        ),
        new_code_cell(_prelude()),
        new_markdown_cell(
            """## Predictive summary audit

Errors are scaled by the training-set feature IQR. This is a numerical
comparison within synthetic cortical-rate space and does not map firing rate
to microvolts."""
        ),
        new_code_cell(
            """ppc_rows = []
for track in ("8d", "7d"):
    summary = json.loads((RESULTS / "diagnostics" / "ppc" / track / "ppc_summary.json").read_text())
    ppc_rows.append(summary)
display(pd.DataFrame(ppc_rows))
display(Image(filename=str(RESULTS / "figures" / "figure10_ppc_8d_7d.png")))
display(Image(filename=str(RESULTS / "figures" / "figure10_ppc_cortical_rate_traces.png")))"""
        ),
        new_markdown_cell(
            """## Marginals and mandatory ridges

The dashed prior density is flat in prior-scaled coordinates. A broad
`c_ctx2th` marginal is scientifically acceptable when calibrated. The
`c_th2ctx`-`c_ctx2th` and `g_LK`-`g_h` panels expose compensatory structure;
correlation alone is not causation."""
        ),
        new_code_cell(
            """structure = []
for track in ("8d", "7d"):
    payload = json.loads((RESULTS / "diagnostics" / "posterior_structure" / track / "posterior_structure_summary.json").read_text())
    structure.append(payload)
display(pd.DataFrame(structure))
display(Image(filename=str(RESULTS / "figures" / "figure10_posterior_structure_identifiability.png")))"""
        ),
    ]
    return _write_notebook(
        "30_Figure10_PPC_and_Posterior_Structure.ipynb", cells
    )


def build_ablation_notebook() -> Path:
    cells = [
        new_markdown_cell(
            """# Matched 7D-versus-8D ablation and scale sensitivity

This paired analysis isolates the effect of releasing `c_ctx2th`. Narrower 7D
marginals are not automatically better: fixing a difficult parameter may
remove an honest ridge. The ablation asks whether 8D adds mechanistic
completeness without unacceptable miscalibration."""
        ),
        new_code_cell(_prelude()),
        new_code_cell(
            """scale = pd.read_csv(RESULTS / "diagnostics" / "scale_sensitivity.csv")
display(scale)
display(Image(filename=str(RESULTS / "figures" / "figure10_7d_vs_8d_ablation.png")))"""
        ),
        new_markdown_cell(
            """## Interpretation boundary

Only 8,192 and 32,768 are measured scales. Rows for 131,072 through 3,000,000
are explicit `not run` records, not extrapolated scientific results. Therefore
the experiment can identify changes over this limited ladder but cannot claim
exact replication of the official three-million-simulation regime."""
        ),
    ]
    return _write_notebook("31_Figure10_7D_vs_8D_Ablation.ipynb", cells)


def build_handoff_notebook() -> Path:
    cells = [
        new_markdown_cell(
            """# Terminal 8D-primary / matched-7D handoff

This notebook reads the frozen reports after all diagnostics have completed.
It keeps three verdict systems separate: 8D Figure-10-equivalent, matched 7D
Figure-10-equivalent, and the preserved original Route-3 strict verdict."""
        ),
        new_code_cell(_prelude()),
        new_code_cell(
            """report_json = PROJECT_ROOT / "FIGURE10_8D_7D_FINAL_REPORT.json"
report_md = PROJECT_ROOT / "FIGURE10_8D_7D_FINAL_REPORT.md"
payload = json.loads(report_json.read_text())
display(pd.Series(payload["verdicts"]))
display(pd.DataFrame(payload["comparison"]))
display(Markdown(report_md.read_text(encoding="utf-8")))"""
        ),
        new_markdown_cell(
            """## Scientific boundary

All verdicts apply only to synthetic posterior inference in the frozen
cortical-rate proxy space. No result here validates real Fpz-Cz inference,
subject-specific recovery, or a source-to-sensor measurement model."""
        ),
    ]
    return _write_notebook("32_Figure10_8D_Final_Handoff.ipynb", cells)


def main() -> None:
    paths = [
        build_protocol_notebook(),
        build_bank_notebook(),
        _training_notebook("8d", 26, "Primary 8D five-NSF ensemble"),
        _training_notebook("7d", 27, "Matched 7D five-NSF baseline"),
        build_global_notebook(),
        build_lc2st_notebook(),
        build_ppc_structure_notebook(),
        build_ablation_notebook(),
        build_handoff_notebook(),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
