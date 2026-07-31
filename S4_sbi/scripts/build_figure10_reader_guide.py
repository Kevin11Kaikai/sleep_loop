"""Prepend the finalized Figure-10 teaching guide while preserving old cells."""

from __future__ import annotations

from pathlib import Path
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH = (
    PROJECT_ROOT
    / "S4_sbi"
    / "notebooks"
    / "00_Observation_SBI_Reader_Guide.ipynb"
)


def build() -> Path:
    notebook = nbformat.read(PATH, as_version=4)
    marker = "<!-- FIGURE10_8D_7D_FINAL_GUIDE_V1 -->"
    if any(
        marker in cell.source
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    ):
        raise RuntimeError("final Figure-10 reader guide section already exists")
    cells = [
        new_markdown_cell(
            f"""{marker}
# 00. Observation-SBI Reader Guide: final 8D/7D Figure-10 evaluation

This is the reader-facing entry point for the completed synthetic
cortical-thalamic SBI evaluation. It can be read in about 30-45 minutes.

**Claim boundary:** the posterior is conditioned on 14 summaries of synthetic
cortical excitatory/inhibitory population firing rates. It is not conditioned
on real Fpz-Cz EEG, and the cortical-rate signal is not simulated scalp EEG."""
        ),
        new_markdown_cell(
            """## The scientific question

Can the frozen 14D cortical-rate observation constrain the full eight
mechanistic parameters with an honest posterior, and what changes when
`c_ctx2th` is fixed in a matched seven-parameter baseline?

The experiment separates five objects:

1. **Real EEG reference:** used earlier to motivate observation design only.
2. **Synthetic simulator output:** cortical `r_mean_EXC` and `r_mean_INH` in Hz.
3. **Frozen summaries:** 14 deterministic time/frequency features.
4. **Posterior inference:** five independently initialized NSF-NPEs combined as
   an equal-weight mixture.
5. **Posterior diagnostics:** joint expected coverage, marginal SBC, L-C2ST,
   PPC, and posterior structure."""
        ),
        new_markdown_cell(
            r"""## Pipeline

$$
\theta_{8D}
\rightarrow \mathrm{simulator}
\rightarrow r_{\mathrm{exc}}(t),r_{\mathrm{inh}}(t)
\rightarrow 14D\ \mathrm{summaries}
\rightarrow 5\ \mathrm{NPEs}
\rightarrow \mathrm{mixture\ posterior}
\rightarrow \mathrm{diagnostics}.
$$

The matched 7D path uses the same seven shared coordinates, simulator seeds,
extractor, training scale, architecture, and diagnostics while fixing
`c_ctx2th` at the historical value."""
        ),
        new_markdown_cell(
            """## Why 8D is primary

All eight parameters are mechanistically part of the model. A difficult
parameter should retain broad uncertainty rather than be deleted merely to
obtain narrower plots. Lack of contraction means the data add little marginal
information; miscalibration means the posterior uncertainty is statistically
wrong. Only the latter is an automatic posterior-validity failure.

The 7D model remains valuable as a controlled ablation. It tests whether fixing
`c_ctx2th` artificially narrows shared parameters or removes a genuine
`c_th2ctx`-`c_ctx2th` compensatory ridge."""
        ),
        new_markdown_cell("## Final dashboard"),
        new_code_cell(
            """from pathlib import Path
import json, os, sys
import pandas as pd
from IPython.display import Image, Markdown, display

cwd = Path.cwd().resolve()
PROJECT_ROOT = cwd if (cwd / "S4_sbi").exists() else cwd.parent.parent
RESULTS = PROJECT_ROOT / "S4_sbi" / "results" / "figure10_8d_7d"
REPORT = PROJECT_ROOT / "FIGURE10_8D_7D_FINAL_REPORT.json"
assert os.environ.get("CONDA_DEFAULT_ENV") == "neurolib"
payload = json.loads(REPORT.read_text(encoding="utf-8"))
display(pd.Series(payload["verdicts"], name="terminal verdict"))
display(pd.DataFrame(payload["comparison"]))
print("Scope:", payload["scope"])
print("Python:", sys.executable)"""
        ),
        new_code_cell(
            """for filename in (
    "figure10_expected_coverage_sbc_8d_7d.png",
    "figure10_lc2st_8d_7d.png",
    "figure10_ppc_8d_7d.png",
    "figure10_posterior_structure_identifiability.png",
    "figure10_7d_vs_8d_ablation.png",
):
    path = RESULTS / "figures" / filename
    assert path.exists() and path.stat().st_size > 1000
    display(Image(filename=str(path)))"""
        ),
        new_markdown_cell(
            """## Recommended reading order

| Order | Notebook | Scientific question | First-reading focus | May initially skip |
|---:|---|---|---|---|
| 1 | [24 Protocol](24_Figure10_8D_7D_Protocol_Preregistration.ipynb) | What was frozen before results? | contracts, resource boundary, verdict rules | package details |
| 2 | [25 Banks](25_Figure10_Matched_Simulation_Banks.ipynb) | Are 8D and 7D data truly paired? | row/seed/order assertions | full manifest |
| 3 | [26 8D ensemble](26_Figure10_8D_Five_NSF_Ensemble.ipynb) | Were five independent 8D NPEs trained? | seeds and validation curves | epoch-by-epoch detail |
| 4 | [27 7D baseline](27_Figure10_7D_Five_NSF_Matched_Baseline.ipynb) | Is the baseline compute-matched? | same checks as 8D | repeated definitions |
| 5 | [28 Coverage/SBC](28_Figure10_Joint_Coverage_and_SBC.ipynb) | Is uncertainty globally calibrated? | ensemble curves and all marginals | individual numeric rows |
| 6 | [29 L-C2ST](29_Figure10_Local_LC2ST.ipynb) | Is the posterior locally correct at the frozen x? | ensemble test versus null | classifier internals |
| 7 | [30 PPC/structure](30_Figure10_PPC_and_Posterior_Structure.ipynb) | Does it predict x and express ridges honestly? | 14D PPC and mandatory 2D pairs | trace examples |
| 8 | [31 Ablation](31_Figure10_7D_vs_8D_Ablation.ipynb) | What changes when c_ctx2th is released? | shared contraction/calibration | unrun scale rows |
| 9 | [32 Handoff](32_Figure10_8D_Final_Handoff.ipynb) | What is the terminal verdict? | three separate verdicts | repeated tables |

Executed HTML versions are under
[`../results/figure10_8d_7d/html/`](../results/figure10_8d_7d/html/)."""
        ),
        new_markdown_cell(
            """## Official Figure 10 mapping

| Official panel | Official purpose | Our equivalent | Interpretation |
|---|---|---|---|
| a | Simulator | Notebooks 24-25 | Eight-parameter cortical-thalamic dynamics |
| b | Observation | Notebook 25/30 | Frozen synthetic cortical-rate 14D x |
| c | Five-NPE ensemble | Notebooks 26-27 | Raw equal-weight NSF mixtures |
| d | Global validation | Notebook 28 | Joint expected coverage and marginal SBC |
| e | Local validation | Notebook 29 | L-C2ST at the frozen paired observation |
| f | Posterior predictive | Notebook 30 | New-seed 14D PPC and trace examples |
| g | Posterior marginals | Notebook 30 | Prior references and contraction |
| h | Joint structure | Notebook 30 | Coupling and conductance ridges |

The mapping follows the pinned Practical Guide repository, but our simulator,
parameter dimension, observations, local compute scale, and installed `sbi`
version differ."""
        ),
        new_markdown_cell(
            """## Terms in plain language

- **NPE:** learns a conditional density \(q(\\theta\\mid x)\) from simulated
  parameter-observation pairs.
- **Neural spline flow (NSF):** an invertible, flexible density model used for
  each NPE member.
- **Five-model ensemble:** an equal mixture of five independently initialized
  posteriors; samples are pooled, not averaged.
- **Joint expected coverage:** ranks the true joint parameter under posterior
  density; calibrated ranks are uniform.
- **Marginal SBC:** checks randomized posterior ranks separately for every
  inferred coordinate.
- **L-C2ST:** a local classifier test asking whether posterior samples at the
  chosen observation behave like calibrated conditional samples.
- **PPC:** reruns the simulator at posterior draws to ask whether predicted
  summaries reproduce the observation.
- **Contraction:** posterior spread relative to prior spread. It measures
  information gain, not calibration by itself.
- **Identifiability:** whether different parameter values can be distinguished
  from the observation.
- **Posterior ridge:** a compensatory family of parameter combinations with
  similar posterior support.
- **Nuisance-seed marginalization:** varying simulator randomness across bank
  rows so the posterior represents stochastic variability."""
        ),
        new_markdown_cell(
            """## The c_ctx2th question

The historical 7D experiment fixed `c_ctx2th` after evidence that it was weakly
recoverable. The primary 8D experiment releases it because mechanistic
completeness requires representing that uncertainty. A broad, prior-like
`c_ctx2th` marginal can be correct when joint coverage and SBC are calibrated.
A narrow biased marginal is not correct.

The `c_th2ctx`-`c_ctx2th` plot should be read as a possible compensatory ridge:
one direction of coupling may trade against the other while producing similar
cortical-rate summaries. This is evidence about the synthetic observation
map, not causal proof about biological connectivity."""
        ),
        new_markdown_cell(
            """## How to read the verdicts

1. **8D Figure-10-equivalent verdict:** quality of the raw full-mechanism NSF
   posterior under the new official-style diagnostics.
2. **7D Figure-10-equivalent verdict:** the same diagnostic philosophy
   conditional on fixed `c_ctx2th`.
3. **Original Route-3 strict verdict:** remains the historical frozen
   `NO-GO`; it is not retroactively changed.
4. **Real-EEG readiness:** remains `NO-GO` because a validated cortical
   source-to-Fpz-Cz measurement model is absent.

An 8D PASS cannot be converted into a Route-3 Formal GO, and neither synthetic
verdict authorizes real-EEG parameter inference."""
        ),
        new_markdown_cell(
            """## What we can and cannot claim

**Potentially established by the terminal diagnostics:** calibrated posterior
inference and predictive adequacy within the frozen synthetic cortical-rate
proxy space, at the completed resource-limited scale.

**Not established:** real Fpz-Cz likelihood equivalence, subject-specific
physiology, cortical source localization, a real EEG digital twin, or exact
replication of the official three-million-simulation experiment."""
        ),
        new_markdown_cell(
            """---

## Preserved earlier Observation-SBI guide

The cells below are the pre-existing guide for Notebooks 02-23. They are
retained as historical context; the final Figure-10 dashboard above is the
current entry point."""
        ),
    ]
    notebook.cells = cells + notebook.cells
    notebook.metadata.kernelspec = {
        "display_name": "neurolib",
        "language": "python",
        "name": "neurolib",
    }
    nbformat.validate(notebook)
    nbformat.write(notebook, PATH)
    return PATH


if __name__ == "__main__":
    print(build())
