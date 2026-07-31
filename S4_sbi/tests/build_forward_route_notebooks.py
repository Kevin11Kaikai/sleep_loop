"""Build notebooks 09 and 00 without changing notebooks 01 through 08."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = PROJECT_ROOT / "S4_sbi" / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.20",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    }


def build_09() -> Path:
    path = NOTEBOOK_DIR / "09_Forward_Model_Feasibility_and_Route_Decision.ipynb"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing notebook: {path}")

    cells = [
        md(
            r"""
            # 09. Forward-Model Feasibility and Route Decision

            **Scientific question.** Can the current one-ALN-cortical-node plus
            one-thalamic-node model generate a physically declared `EEG Fpz-Cz`
            observable, or does Route 2 still require new spatial and source
            assumptions?

            This notebook is an audit, not a forward-model implementation. It
            performs no `Hz -> uV` scaling, no simulation-bank generation and no
            SNPE training. A high correlation after z-scoring or PSD
            normalization remains **shape-level screening**, not EEG measurement
            validation.

            The audit distinguishes:

            - a **model state**, such as population firing rate or adaptation;
            - a **primary EEG source**, normally requiring a physical
              current-dipole definition;
            - a **lead field**, which projects located and oriented sources
              through a volume conductor to sensor potentials;
            - the `Fpz-Cz` **bipolar reference**, applied only after separate
              sensor potentials exist.

            Primary references:

            - [Cakan & Obermayer (2020), ALN mean-field model](https://doi.org/10.1371/journal.pcbi.1007822)
            - [MNE `make_forward_solution`](https://mne.tools/stable/generated/mne.make_forward_solution.html)
            - [MNE head model and forward tutorial](https://mne.tools/stable/auto_tutorials/forward/30_forward.html)
            - [Hallez et al. (2007), EEG forward-problem review](https://pmc.ncbi.nlm.nih.gov/articles/PMC2234413/)
            - [Mosher et al. (1999), EEG/MEG forward solutions](https://pubmed.ncbi.nlm.nih.gov/10097460/)
            """
        ),
        md(
            """
            ## 1. Environment, immutable inputs, and audit boundary

            This block locates the repository without machine-specific paths,
            records the actual `neurolib` interpreter, hashes notebooks 01-08,
            and reloads the Route-2 NO-GO decision from notebook 08. No raw EEG or
            simulator arrays are loaded here.
            """
        ),
        code(
            """
            from __future__ import annotations

            import hashlib
            import importlib.metadata
            import json
            import os
            from pathlib import Path
            import platform
            import sys
            from datetime import datetime, timezone

            import matplotlib.pyplot as plt
            import nbformat
            import numpy as np
            import pandas as pd
            from IPython.display import display
            from jupyter_client.kernelspec import KernelSpecManager

            PROJECT_ROOT = Path.cwd().resolve()
            while not ((PROJECT_ROOT / ".git").exists() and (PROJECT_ROOT / "S4_sbi").exists()):
                if PROJECT_ROOT.parent == PROJECT_ROOT:
                    raise RuntimeError("Could not locate sleep_loop repository root")
                PROJECT_ROOT = PROJECT_ROOT.parent

            SRC_ROOT = PROJECT_ROOT / "S4_sbi" / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))

            from sleep_sbi.forward_model_feasibility import (
                SCHEMA_VERSION,
                MeasurementContractError,
                apply_measurement_model,
                calibration_leakage_matrix,
                citation_table,
                leadfield_requirement_matrix,
                measurement_contract_components,
                protected_notebook_paths,
                route_decision_matrix,
                single_source_rank_audit,
                source_candidate_decisions,
                validate_measurement_declaration,
            )

            OUTPUT_DIR = PROJECT_ROOT / "S4_sbi" / "results" / "forward_model_feasibility_route_decision"
            FIGURE_DIR = OUTPUT_DIR / "figures"
            HTML_DIR = PROJECT_ROOT / "S4_sbi" / "results" / "overnight_observation_ablation" / "html"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            HTML_DIR.mkdir(parents=True, exist_ok=True)
            NOTEBOOK_PATH = PROJECT_ROOT / "S4_sbi" / "notebooks" / "09_Forward_Model_Feasibility_and_Route_Decision.ipynb"

            def sha256(path: Path) -> str:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()

            protected_paths = protected_notebook_paths(PROJECT_ROOT)
            protected_hashes_start = {
                path.relative_to(PROJECT_ROOT).as_posix(): sha256(path)
                for path in protected_paths
            }

            ksm = KernelSpecManager()
            kernel_id = "python3"
            kernel_display_name = ksm.get_kernel_spec(kernel_id).display_name
            environment = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
                "sys_executable": sys.executable,
                "sys_prefix": sys.prefix,
                "python_version": platform.python_version(),
                "kernel_id": kernel_id,
                "kernel_display_name": kernel_display_name,
                "neurolib_version": importlib.metadata.version("neurolib"),
                "neurolib_source": str(Path(__import__("neurolib").__file__).resolve()),
                "schema_version": SCHEMA_VERSION,
            }
            assert environment["conda_environment"] == "neurolib"
            assert "neurolib" in environment["sys_executable"].lower()

            audit08_dir = PROJECT_ROOT / "S4_sbi" / "results" / "eeg_observation_mapping_audit"
            route08 = json.loads((audit08_dir / "route_decision.json").read_text(encoding="utf-8"))
            assert route08["recommended_route"] == "Route 2"
            assert route08["route_1_shared_schema_go"] is False
            display(pd.DataFrame([environment]))
            print(f"Protected notebooks hashed: {len(protected_hashes_start)}")
            print(f"Notebook-08 inherited gate: {route08['recommended_route']} / shared schema GO={route08['route_1_shared_schema_go']}")
            """
        ),
        md(
            """
            ## 2. Which ALN quantities are states, and which are intermediates?

            `r_mean`, `I_mu`, synaptic mean/variance variables and excitatory
            adaptation `I_A` are numerical states integrated by the ALN masses.
            The historical project network only asked neurolib to record
            `r_mean_EXC` and `r_mean_INH`; notebook 08 temporarily expanded
            `output_vars` for auditing without changing the equations.

            Population mean voltage is different: `voltage_lookup(...)` is
            evaluated inside the excitatory derivative to update adaptation. It
            is backed by the precomputed `V_mean_ss` transfer-function table but
            is not a saved state. It can be reconstructed in a future audit if
            `I_sigma` and all required coupling inputs are reconstructed. Even
            then, membrane voltage is not automatically a current dipole.
            """
        ),
        code(
            """
            state_classification = pd.DataFrame([
                {
                    "quantity": "r_mean_EXC / r_mean_INH",
                    "code_class": "integrated state",
                    "project_saved": True,
                    "native_unit": "1/ms; project converts to Hz",
                    "EEG_source_ready": False,
                    "evidence": "aln.py state_variable_names; V7 output_vars",
                },
                {
                    "quantity": "I_mu EXC/INH",
                    "code_class": "integrated state",
                    "project_saved": False,
                    "native_unit": "mV/ms",
                    "EEG_source_ready": False,
                    "evidence": "aln.py:414-421,577-606",
                },
                {
                    "quantity": "I_A EXC",
                    "code_class": "integrated adaptation state",
                    "project_saved": False,
                    "native_unit": "pA",
                    "EEG_source_ready": False,
                    "evidence": "aln.py:414-421,535-547",
                },
                {
                    "quantity": "synaptic mean/variance states",
                    "code_class": "integrated states",
                    "project_saved": False,
                    "native_unit": "dimensionless/model-native",
                    "EEG_source_ready": False,
                    "evidence": "aln.py:414-421,539-556",
                },
                {
                    "quantity": "population mean voltage",
                    "code_class": "transfer-function intermediate",
                    "project_saved": False,
                    "native_unit": "mV",
                    "EEG_source_ready": False,
                    "evidence": "V_mean_ss and voltage_lookup, aln.py:234-246,344-357,534-537",
                },
                {
                    "quantity": "TCR/TRN V",
                    "code_class": "integrated thalamic states",
                    "project_saved": False,
                    "native_unit": "mV",
                    "EEG_source_ready": False,
                    "evidence": "thalamus.py:125-135,304-311",
                },
            ])
            display(state_classification)

            source_candidates = source_candidate_decisions()
            assert source_candidates["candidate"].is_unique
            display(source_candidates)

            decision_category = []
            for decision in source_candidates["decision"]:
                lowered = decision.lower()
                if "preferred candidate" in lowered or "research candidate" in lowered or "research audit" in lowered:
                    decision_category.append("Research candidate")
                elif "route-3" in lowered:
                    decision_category.append("Route 3 only")
                else:
                    decision_category.append("Exclude/direct NO-GO")
            source_plot = source_candidates.assign(decision_category=decision_category)
            category_x = {"Exclude/direct NO-GO": 0, "Route 3 only": 1, "Research candidate": 2, "Forward-ready": 3}
            colors = {
                "Exclude/direct NO-GO": "#c44e52",
                "Route 3 only": "#4c72b0",
                "Research candidate": "#dd8452",
                "Forward-ready": "#55a868",
            }
            fig, ax = plt.subplots(figsize=(10.5, 6.4))
            y = np.arange(len(source_plot))
            for yi, (_, row) in zip(y, source_plot.iterrows()):
                ax.scatter(category_x[row["decision_category"]], yi, s=85, color=colors[row["decision_category"]], zorder=3)
            ax.set_yticks(y, source_plot["candidate"])
            ax.set_xticks(list(category_x.values()), list(category_x.keys()), rotation=12, ha="right")
            ax.set_xlim(-0.4, 3.4)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.25)
            ax.set_title("Cortical source candidates: none is forward-ready")
            ax.set_xlabel("Current evidence-based decision")
            fig.tight_layout()
            for suffix in ("png", "svg"):
                fig.savefig(FIGURE_DIR / f"source_candidate_decisions.{suffix}", dpi=180 if suffix == "png" else None, bbox_inches="tight")
            display(fig)
            plt.close(fig)
            """
        ),
        md(
            r"""
            ## 3. What neurolib's lead-field utility supplies, and what it assumes

            `neurolib/utils/leadfield.py` wraps MNE. It can construct a
            subject/template BEM, coregister sensor information, build a cortical
            surface source space, compute a sensor-by-dipole forward matrix, fix
            dipoles to surface-normal orientations and average columns into AAL2
            regions.

            It does **not** define which ALN variable is a current dipole, where
            the single ALN node is located, its cortical extent/orientation, its
            dipole-moment unit, or the Fpz-Cz reference/noise contract. The
            official MNE API requires sensor `Info`, a head-to-MRI transform,
            source space and BEM:
            [MNE API](https://mne.tools/stable/generated/mne.make_forward_solution.html).
            """
        ),
        code(
            """
            leadfield_requirements = leadfield_requirement_matrix()
            assert not leadfield_requirements["provided_by_current_two_node_model"].any()
            display(leadfield_requirements)

            local_leadfield = Path(__import__("neurolib").__file__).resolve().parent / "utils" / "leadfield.py"
            leadfield_code_check = {
                "local_file_exists": local_leadfield.exists(),
                "uses_mne_make_forward_solution": "mne.make_forward_solution" in local_leadfield.read_text(encoding="utf-8"),
                "uses_fixed_surface_orientation": "force_fixed=True" in local_leadfield.read_text(encoding="utf-8"),
                "implements_fpz_cz_reference": "Fpz-Cz" in local_leadfield.read_text(encoding="utf-8"),
                "implements_aln_state_to_dipole": "I_mu_EXC" in local_leadfield.read_text(encoding="utf-8"),
            }
            assert leadfield_code_check["local_file_exists"]
            assert leadfield_code_check["uses_mne_make_forward_solution"]
            assert not leadfield_code_check["implements_fpz_cz_reference"]
            assert not leadfield_code_check["implements_aln_state_to_dipole"]
            display(pd.DataFrame([leadfield_code_check]))
            """
        ),
        md(
            r"""
            ## 4. Single-source Fpz-Cz is temporally rank-1

            For one scalar source \(q(t)\) and a linear instantaneous lead field:

            \[
            V_{\mathrm{Fpz}}(t)=L_{\mathrm{Fpz}}q(t), \qquad
            V_{\mathrm{Cz}}(t)=L_{\mathrm{Cz}}q(t)
            \]

            \[
            V_{\mathrm{Fpz-Cz}}(t)
            =\left(L_{\mathrm{Fpz}}-L_{\mathrm{Cz}}\right)q(t).
            \]

            The bipolar signal may be non-zero when the two lead-field gains
            differ, but it remains a fixed gain/sign copy of the same source.
            Thus a fully declared single dipole could produce a physical
            Fpz-Cz voltage, but the current abstract node supplies none of the
            position, orientation or dipole-moment information needed to compute
            that gain. It also cannot express distributed source mixtures that
            alter channel-specific temporal or spectral structure.
            """
        ),
        code(
            """
            rank_audit = single_source_rank_audit()
            assert rank_audit["temporal_rank"] == 1
            assert rank_audit["adds_new_temporal_structure"] is False
            display(pd.DataFrame([rank_audit]))

            fig, ax = plt.subplots(figsize=(11, 3.8))
            ax.axis("off")
            boxes = [
                (0.05, 0.52, "one scalar source\\nq(t)\\n(no geometry)"),
                (0.39, 0.72, "Fpz potential\\nL_Fpz q(t)"),
                (0.39, 0.27, "Cz potential\\nL_Cz q(t)"),
                (0.72, 0.50, "bipolar output\\n(L_Fpz - L_Cz) q(t)\\nrank = 1"),
            ]
            for x, y0, label in boxes:
                ax.text(
                    x,
                    y0,
                    label,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f4f4f4", "edgecolor": "#444444"},
                )
            arrow = {"arrowstyle": "->", "color": "#555555", "lw": 1.8}
            ax.annotate("", xy=(0.31, 0.72), xytext=(0.15, 0.56), xycoords="axes fraction", arrowprops=arrow)
            ax.annotate("", xy=(0.31, 0.27), xytext=(0.15, 0.48), xycoords="axes fraction", arrowprops=arrow)
            ax.annotate("", xy=(0.63, 0.52), xytext=(0.49, 0.70), xycoords="axes fraction", arrowprops=arrow)
            ax.annotate("", xy=(0.63, 0.48), xytext=(0.49, 0.29), xycoords="axes fraction", arrowprops=arrow)
            ax.set_title("One cortical source can be non-zero at Fpz-Cz but adds no spatially mixed dynamics", pad=12)
            fig.tight_layout()
            for suffix in ("png", "svg"):
                fig.savefig(FIGURE_DIR / f"single_source_rank1.{suffix}", dpi=180 if suffix == "png" else None, bbox_inches="tight")
            display(fig)
            plt.close(fig)
            """
        ),
        md(
            """
            ## 5. Strict measurement contract and leakage gate

            The contract below follows the required chain
            `theta -> dynamics -> cortical source -> geometry/orientation ->
            lead field -> Fpz/Cz -> bipolar reference -> gain/noise -> 100 Hz ->
            30 s epochs -> EEG extractor`.

            Empty or arbitrary declarations are not repaired with defaults. In
            particular, source choice, location, orientation, gain or noise must
            not be selected by maximizing agreement with SC4001 and then reused
            to infer SC4001; that would leak the inference target into the
            measurement model.
            """
        ),
        code(
            """
            contract_components = measurement_contract_components()
            leakage = calibration_leakage_matrix()
            display(contract_components)
            display(leakage)

            empty_gate = validate_measurement_declaration({})
            assert empty_gate.status == "NO-GO"
            try:
                apply_measurement_model(np.ones(100), {})
            except MeasurementContractError as exc:
                refusal_message = str(exc)
            else:
                raise AssertionError("Incomplete measurement declaration was not refused")
            print("Strict empty-contract gate:", empty_gate.as_dict())
            print("Projection refusal:", refusal_message)

            gate_color = {"PASS": "#55a868", "BLOCKED": "#dd8452", "NO-GO": "#c44e52"}
            fig, ax = plt.subplots(figsize=(11, 7.5))
            y = np.arange(len(contract_components))
            for yi, gate in zip(y, contract_components["gate"]):
                ax.scatter(0, yi, s=130, marker="s", color=gate_color[gate])
            ax.set_yticks(y, [f"{i}. {name}" for i, name in zip(contract_components["stage_index"], contract_components["component"])])
            ax.set_xticks([])
            ax.set_xlim(-0.5, 0.5)
            ax.invert_yaxis()
            ax.set_title("Measurement contract status (green=available, orange=blocked, red=NO-GO)")
            ax.grid(axis="y", alpha=0.15)
            fig.tight_layout()
            for suffix in ("png", "svg"):
                fig.savefig(FIGURE_DIR / f"measurement_contract_status.{suffix}", dpi=180 if suffix == "png" else None, bbox_inches="tight")
            display(fig)
            plt.close(fig)
            """
        ),
        md(
            """
            ## 6. Route 2 versus Route 3

            **Route 2** remains scientifically possible in principle, but is not
            implementable from the current two-node state alone. It requires new,
            independently constrained source geometry and source-to-dipole
            assumptions plus held-out measurement validation.

            **Route 3** restricts inference to a declared cortical-observable
            simulator space. It can test synthetic parameter recovery and
            calibration, but cannot claim inference about real Fpz-Cz physiology.
            Recommending Route 3 for the near-term conference scope does not
            authorize a bank or SNPE run; those remain researcher decisions.
            """
        ),
        code(
            """
            routes = route_decision_matrix()
            citations = citation_table()
            display(routes)
            display(citations)

            decision = {
                "schema_version": SCHEMA_VERSION,
                "current_model_supports_physically_declared_fpz_cz": False,
                "single_source_can_be_nonzero": True,
                "single_source_temporal_rank": 1,
                "existing_validated_eeg_or_lfp_proxy": False,
                "existing_project_forward_model": False,
                "route_2_current_status": "NO-GO_PENDING_EXTERNAL_SOURCE_AND_SPATIAL_ASSUMPTIONS",
                "route_2_long_term_status": "scientifically_possible_if_independently_defined_and_validated",
                "near_term_recommendation": "Route 3 synthetic cortical-observable recovery, with real EEG limited to external shape-level validation",
                "route_3_authorized": False,
                "simulation_bank_authorized": False,
                "pilot_snpe_authorized": False,
                "conference_claim_boundary": (
                    "Report model dynamics, adapter engineering, shape-level external comparisons, and the explicit "
                    "measurement-model limitation. Do not claim a real-EEG posterior or measurement validation."
                ),
                "researcher_decisions_required": [
                    "Choose near-term Route 3 scope versus investing in a Route 2 source/forward program.",
                    "For Route 2, approve a biophysically defined source variable and independent calibration/validation datasets.",
                    "For Route 3, freeze the cortical observable, prior, extractor, diagnostics, and budget before any bank or SNPE run.",
                ],
            }
            display(pd.DataFrame([decision]))
            """
        ),
        md(
            """
            ## 7. Export, reload, and immutable-source verification

            Artifacts contain tables, decisions and provenance only. They do not
            contain raw EEG, simulator time series, a simulation bank or
            posterior samples. Every file is reloaded before the notebook ends,
            and notebooks 01-08 are re-hashed.
            """
        ),
        code(
            """
            tables = {
                "source_candidate_decisions.csv": source_candidates,
                "state_classification.csv": state_classification,
                "leadfield_requirements.csv": leadfield_requirements,
                "measurement_contract_components.csv": contract_components,
                "calibration_leakage_matrix.csv": leakage,
                "route2_route3_decision.csv": routes,
                "literature_sources.csv": citations,
            }
            for name, frame in tables.items():
                frame.to_csv(OUTPUT_DIR / name, index=False)

            json_artifacts = {
                "environment_report.json": environment,
                "single_source_rank_audit.json": rank_audit,
                "forward_model_feasibility_decision.json": decision,
                "protected_notebook_hashes.json": protected_hashes_start,
            }
            for name, payload in json_artifacts.items():
                (OUTPUT_DIR / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

            reload_checks = {}
            for name, frame in tables.items():
                loaded = pd.read_csv(OUTPUT_DIR / name)
                reload_checks[name] = {
                    "readable": True,
                    "rows": int(len(loaded)),
                    "columns": int(len(loaded.columns)),
                    "shape_matches": loaded.shape == frame.shape,
                }
                assert loaded.shape == frame.shape
            for name in json_artifacts:
                loaded = json.loads((OUTPUT_DIR / name).read_text(encoding="utf-8"))
                reload_checks[name] = {"readable": True, "top_level_keys": len(loaded)}

            protected_hashes_end = {
                path.relative_to(PROJECT_ROOT).as_posix(): sha256(path)
                for path in protected_paths
            }
            assert protected_hashes_end == protected_hashes_start
            nbformat.validate(nbformat.read(NOTEBOOK_PATH, as_version=4))

            validation = {
                "schema_version": SCHEMA_VERSION,
                "nbformat_validation": "PASS",
                "protected_notebook_hashes_unchanged": True,
                "artifact_reload": reload_checks,
                "route_2_gate": decision["route_2_current_status"],
                "simulation_run_count": 0,
                "simulation_bank_created": False,
                "snpe_started": False,
                "raw_eeg_saved": False,
                "raw_simulator_signal_saved": False,
            }
            (OUTPUT_DIR / "validation_report.json").write_text(
                json.dumps(validation, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print("Artifact reload: PASS")
            print("Protected notebooks 01-08 unchanged: PASS")
            print("Route 2 gate:", validation["route_2_gate"])
            """
        ),
        md(
            """
            ## 8. Decision summary

            1. The current single cortical node does not by itself support a
               spatially non-degenerate Fpz-Cz measurement model.
            2. A fully specified single dipole could yield a non-zero bipolar
               voltage, but only as a rank-1 scaled copy and only after external
               position, orientation, source-moment and head-model assumptions.
            3. Reconstructed ALN population mean voltage is the best-supported
               voltage-level research candidate, but it is not a primary-current
               dipole and remains insufficient.
            4. A derived transmembrane/synaptic-current dipole would be more
               relevant to Route 2, but it does not currently exist with
               traceable units, population geometry or validation.
            5. Near-term conference claims should use Route 3 scope or retain
               real EEG as external shape-level validation. No simulation bank
               or SNPE is authorized by this notebook.
            """
        ),
    ]

    notebook = nbf.v4.new_notebook(cells=cells, metadata=metadata())
    nbf.validate(notebook)
    nbf.write(notebook, path)
    return path


def build_00() -> Path:
    path = NOTEBOOK_DIR / "00_Observation_SBI_Reader_Guide.ipynb"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing notebook: {path}")

    cells = [
        md(
            """
            # 00. Observation and SBI Reader Guide

            这是 notebooks 02-09 的阅读导航，不是新的scientific experiment。
            它不复制大型输出、不重新运行simulation，也不产生posterior。

            当前主线是：

            `Observation schema -> schema freeze -> extractor parity -> training gate
            -> diagnostic availability -> engineering adapter -> mapping audit ->
            forward feasibility / Route decision`

            必须先记住三条边界：

            - 05和06没有实际posterior、coverage、recovery或PPC结果；
            - 07证明adapter工程路径可运行，但semantic parity失败；
            - 08只进行跨modality的shape-level screening，没有验证EEG measurement model。
            """
        ),
        md(
            """
            ## 1. Runtime and linked evidence

            此block只核对notebook和HTML是否存在、记录当前kernel，并加载09生成的
            registry。相对路径用于保证repo可迁移性。
            """
        ),
        code(
            """
            from __future__ import annotations

            import hashlib
            import json
            import os
            from pathlib import Path
            import platform
            import sys
            from datetime import datetime, timezone

            import matplotlib.pyplot as plt
            import nbformat
            import numpy as np
            import pandas as pd
            from IPython.display import display, Markdown
            from jupyter_client.kernelspec import KernelSpecManager

            PROJECT_ROOT = Path.cwd().resolve()
            while not ((PROJECT_ROOT / ".git").exists() and (PROJECT_ROOT / "S4_sbi").exists()):
                if PROJECT_ROOT.parent == PROJECT_ROOT:
                    raise RuntimeError("Could not locate sleep_loop repository root")
                PROJECT_ROOT = PROJECT_ROOT.parent
            SRC_ROOT = PROJECT_ROOT / "S4_sbi" / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))

            from sleep_sbi.forward_model_feasibility import (
                reader_guide_registry,
                route_decision_matrix,
            )

            OUTPUT_DIR = PROJECT_ROOT / "S4_sbi" / "results" / "observation_sbi_reader_guide"
            FIGURE_DIR = OUTPUT_DIR / "figures"
            HTML_DIR = PROJECT_ROOT / "S4_sbi" / "results" / "overnight_observation_ablation" / "html"
            NOTEBOOK_PATH = PROJECT_ROOT / "S4_sbi" / "notebooks" / "00_Observation_SBI_Reader_Guide.ipynb"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)

            environment = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
                "sys_executable": sys.executable,
                "sys_prefix": sys.prefix,
                "python_version": platform.python_version(),
                "kernel_id": "python3",
                "kernel_display_name": KernelSpecManager().get_kernel_spec("python3").display_name,
            }
            assert environment["conda_environment"] == "neurolib"
            assert "neurolib" in environment["sys_executable"].lower()
            display(pd.DataFrame([environment]))
            """
        ),
        md(
            """
            ## 2. Notebook 02-09 workflow

            图中的箭头表示证据依赖，不表示每一步都成功。红色节点是hard
            blocker或NO-GO；蓝色节点是已完成的审计/工程工作；绿色仅表示阅读入口，
            不代表SNPE已获批准。
            """
        ),
        code(
            """
            registry = reader_guide_registry()
            notebook_dir = PROJECT_ROOT / "S4_sbi" / "notebooks"
            notebook_map = {
                "02": "02_Observation_Schema_Comparison.ipynb",
                "03": "03_Observation_Schema_Freeze.ipynb",
                "04": "04_Real_Simulation_Extractor_Parity.ipynb",
                "05": "05_Pilot_SNPE_Ablation.ipynb",
                "06": "06_Ablation_Diagnostics.ipynb",
                "07": "07_Simulator_Observable_Adapter_Validation.ipynb",
                "08": "08_EEG_Observation_Mapping_Audit.ipynb",
                "09": "09_Forward_Model_Feasibility_and_Route_Decision.ipynb",
            }
            html_map = {
                "02": "S4_sbi/results/observation_schema_comparison/02_Observation_Schema_Comparison.html",
                "03": "S4_sbi/results/overnight_observation_ablation/html/03_Observation_Schema_Freeze.html",
                "04": "S4_sbi/results/overnight_observation_ablation/html/04_Real_Simulation_Extractor_Parity.html",
                "05": "S4_sbi/results/overnight_observation_ablation/html/05_Pilot_SNPE_Ablation.html",
                "06": "S4_sbi/results/overnight_observation_ablation/html/06_Ablation_Diagnostics.html",
                "07": "S4_sbi/results/simulator_observable_adapter_validation/html/07_Simulator_Observable_Adapter_Validation.html",
                "08": "S4_sbi/results/overnight_observation_ablation/html/08_EEG_Observation_Mapping_Audit.html",
                "09": "S4_sbi/results/overnight_observation_ablation/html/09_Forward_Model_Feasibility_and_Route_Decision.html",
            }
            registry["notebook_path"] = registry["notebook"].map(
                lambda key: f"S4_sbi/notebooks/{notebook_map[key]}"
            )
            registry["html_path"] = registry["notebook"].map(html_map)
            registry["notebook_exists"] = registry["notebook_path"].map(
                lambda value: (PROJECT_ROOT / value).exists()
            )
            registry["html_exists"] = registry["html_path"].map(
                lambda value: (PROJECT_ROOT / value).exists()
            )
            assert registry["notebook_exists"].all()
            display(registry)

            fig, ax = plt.subplots(figsize=(13, 3.7))
            ax.axis("off")
            labels = [
                ("02", "Schema\\n14D vs 23D", "#4c72b0"),
                ("03", "Freeze\\nNO-GO", "#c44e52"),
                ("04", "Parity\\nNO-GO", "#c44e52"),
                ("05", "SNPE gate\\nnot run", "#c44e52"),
                ("06", "Diagnostics\\nno results", "#c44e52"),
                ("07", "Adapter\\nengineering OK", "#4c72b0"),
                ("08", "Mapping\\nshape only", "#4c72b0"),
                ("09", "Route decision\\nRoute 2 blocked", "#dd8452"),
            ]
            xs = np.linspace(0.06, 0.94, len(labels))
            for i, (key, label, color) in enumerate(labels):
                ax.text(
                    xs[i],
                    0.52,
                    f"{key}\\n{label}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": "#333333"},
                )
                if i:
                    ax.annotate(
                        "",
                        xy=(xs[i] - 0.055, 0.52),
                        xytext=(xs[i - 1] + 0.055, 0.52),
                        xycoords="axes fraction",
                        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#555555"},
                    )
            ax.set_title("Observation/SBI evidence flow: completed audits do not imply posterior results", pad=20)
            fig.tight_layout()
            for suffix in ("png", "svg"):
                fig.savefig(FIGURE_DIR / f"notebook_workflow.{suffix}", dpi=180 if suffix == "png" else None, bbox_inches="tight")
            display(fig)
            plt.close(fig)
            """
        ),
        md(
            """
            ## 3. 每本notebook最值得看的1-3个位置

            - **02**：actual Baseline-14D证据、constructive prefix identity、SNPE readiness。
            - **03**：freeze decision、Go/No-Go表、role figure。
            - **04**：legacy bank资产审计、feature-level parity failures、hard gate。
            - **05**：training gate、`BLOCKED_BY_PARITY` manifest；没有posterior。
            - **06**：diagnostic availability与ablation summary；全部是insufficient evidence。
            - **07**：adapter contract、feature blocker matrix、NO-GO；工程成功不等于语义成功。
            - **08**：state catalog、standardized PSD/ACF、Route 2建议；仅shape screening。
            - **09**：source-candidate matrix、rank-1推导、measurement contract和Route 2/3表。
            """
        ),
        code(
            """
            priority_view = registry[
                [
                    "notebook",
                    "scientific_question",
                    "priority_sections_or_figures",
                    "current_conclusion",
                    "cannot_claim",
                    "notebook_path",
                    "html_path",
                ]
            ]
            display(priority_view)
            """
        ),
        md(
            """
            ## 4. 已成立、未成立与hard blockers

            `Established`只表示代码或审计证据成立；不等于生理机制已被识别。
            `Not established`不能被空表、旧bank或较窄posterior替代。
            """
        ),
        code(
            """
            evidence_status = pd.DataFrame([
                ("Actual executable baseline is 14D", "Established", "02"),
                ("Candidate augmented library is 23D", "Established", "02"),
                ("The augmented prefix is constructively the same baseline", "Established", "02"),
                ("A real-EEG schema has simulation semantic parity", "Not established", "03-04"),
                ("Engineering adapter runs V7/V8/V8a cortical rate outputs", "Established", "07"),
                ("Cortical firing rate is a validated EEG proxy", "Not established", "07-09"),
                ("Normalized model/EEG spectral shapes can be compared descriptively", "Established with caveat", "08"),
                ("A source-to-Fpz-Cz measurement model exists", "Not established", "08-09"),
                ("Pilot SNPE was trained", "False / not run", "05"),
                ("Posterior recovery, coverage, L-C2ST or PPC were obtained", "False / unavailable", "06"),
                ("Route 2 can be implemented from current node states alone", "False", "09"),
            ], columns=["statement", "status", "evidence_notebook"])
            display(evidence_status)
            """
        ),
        md(
            """
            ## 5. 术语速查

            - **Schema**：有固定名字、顺序、单位、aggregation和validity规则的向量定义。
            - **Extractor parity**：真实与模拟输入经过同语义算法后得到同名、同序、同单位的固定向量。
            - **Cortical observable**：模型内部可读取量，例如population firing rate；不能自动称EEG。
            - **Measurement model**：从物理source、geometry和volume conductor到sensor voltage/reference/noise的映射。
            - **Shape-level validation**：标准化后比较PSD或waveform形状；不检查绝对单位和sensor语义。
            - **Held-out validation**：未用于拟合、特征选择或calibration的独立证据。
            - **Synthetic recovery**：已知theta生成model observable，再测试inference能否恢复theta。
            - **PPC**：需要正式posterior estimator和posterior predictive simulations；当前05-06不具备。
            """
        ),
        code(
            """
            glossary = pd.DataFrame([
                ("observation schema", "Ordered feature contract with units, aggregation and validity", "Does not prove simulator parity"),
                ("extractor parity", "Same-semantic real/simulation feature extraction", "Not merely same Python function"),
                ("cortical observable", "Recorded or reconstructed model-internal signal", "Not automatically scalp EEG"),
                ("measurement model", "Source-to-sensor projection, reference, units and noise", "Currently missing"),
                ("shape-level comparison", "Normalized PSD/ACF/waveform comparison", "No raw-unit claim"),
                ("held-out validation", "Evidence untouched by fitting/calibration", "Cannot reuse inference inputs"),
                ("synthetic recovery", "Recover known simulated theta", "Does not establish real-data validity"),
                ("posterior predictive check", "Simulate from a trained posterior and compare held-out observables", "No result exists in 05-06"),
            ], columns=["term", "working_definition", "boundary"])
            display(glossary)
            """
        ),
        md(
            """
            ## 6. 90分钟推荐阅读路线

            1. **0-15 min：02 Sections 3/6/11**，确认14D、23D和prefix identity。
            2. **15-30 min：03 Go/No-Go + 04 hard gate**，理解为何现有bank不能改名复用。
            3. **30-40 min：05 gate + 06 availability**，确认没有实际posterior结果。
            4. **40-55 min：07 adapter contract/blocker matrix**，区分engineering与semantic parity。
            5. **55-70 min：08 state catalog和PSD/ACF**，把高shape correlation放回正确边界。
            6. **70-90 min：09 rank-1、contract与Route 2/3**，作研究范围决策。
            """
        ),
        code(
            """
            reading_plan = pd.DataFrame([
                ("00-15", "02", "Actual 14D, 23D crosswalk, constructive prefix", "Know what the observation vectors really are"),
                ("15-30", "03-04", "Freeze and semantic-parity gates", "Know why no schema entered training"),
                ("30-40", "05-06", "Training/diagnostic availability", "Verify there is no posterior result"),
                ("40-55", "07", "Adapter and blocker matrix", "Separate engineering success from scientific parity"),
                ("55-70", "08", "State catalog and normalized shape plots", "Understand the modality caveat"),
                ("70-90", "09", "Rank-1 source, measurement contract, Route 2/3", "Make the next research-scope decision"),
            ], columns=["minutes", "notebook", "focus", "reading_goal"])
            display(reading_plan)
            """
        ),
        md(
            """
            ## 7. Route decision and next human gate

            近期最稳妥的是把Route 3作为**可讨论但尚未授权**的synthetic
            recovery方向；Route 2保留为需要source derivation、geometry、BEM、
            calibration和held-out validation的长期工作。真实SC4001只能继续作为带
            modality caveat的external shape evidence。
            """
        ),
        code(
            """
            routes = route_decision_matrix()
            display(routes)

            human_gate = pd.DataFrame([
                ("Scope", "Choose near-term Route 3 synthetic recovery or fund a Route 2 source/forward program", "Researcher"),
                ("Route 2 source", "Approve source physics and independent calibration/validation datasets", "Researcher + domain expert"),
                ("Route 3 protocol", "Freeze observable, prior, extractor, budget, seeds and diagnostics before bank generation", "Researcher"),
            ], columns=["decision", "question", "owner"])
            display(human_gate)
            """
        ),
        md(
            """
            ## 8. Export and validation

            Reader-guide artifacts只保存索引、术语和验证状态。它们不包含原始EEG、
            simulation arrays、posterior或训练结果。
            """
        ),
        code(
            """
            registry.to_csv(OUTPUT_DIR / "reader_guide_registry.csv", index=False)
            evidence_status.to_csv(OUTPUT_DIR / "evidence_status.csv", index=False)
            glossary.to_csv(OUTPUT_DIR / "glossary.csv", index=False)
            reading_plan.to_csv(OUTPUT_DIR / "reading_plan_90min.csv", index=False)
            human_gate.to_csv(OUTPUT_DIR / "researcher_decisions.csv", index=False)
            (OUTPUT_DIR / "environment_report.json").write_text(
                json.dumps(environment, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            reload_shapes = {}
            for name in [
                "reader_guide_registry.csv",
                "evidence_status.csv",
                "glossary.csv",
                "reading_plan_90min.csv",
                "researcher_decisions.csv",
            ]:
                loaded = pd.read_csv(OUTPUT_DIR / name)
                assert len(loaded) > 0
                reload_shapes[name] = list(loaded.shape)
            nbformat.validate(nbformat.read(NOTEBOOK_PATH, as_version=4))
            validation = {
                "nbformat_validation": "PASS",
                "artifact_reload_shapes": reload_shapes,
                "notebooks_02_to_09_exist": bool(registry["notebook_exists"].all()),
                "html_available_before_external_export": int(registry["html_exists"].sum()),
                "simulation_count": 0,
                "snpe_started": False,
                "posterior_results_claimed": False,
            }
            (OUTPUT_DIR / "validation_report.json").write_text(
                json.dumps(validation, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print("Reader-guide artifact reload: PASS")
            print("No simulation or posterior computation was run.")
            """
        ),
        md(
            """
            ## 9. 阅读结束后的检查问题

            阅读者应能回答：

            1. Baseline为什么是14D而不是被强行补成18D？
            2. 为什么07的adapter success不等于EEG semantic parity？
            3. 为什么08的标准化PSD相似不能证明measurement model？
            4. 为什么单scalar source产生的Fpz-Cz仍然是rank-1？
            5. Route 2缺哪些外部假设，Route 3又不能声称什么？

            在这五个问题回答清楚前，不应生成simulation bank或启动pilot SNPE。
            """
        ),
    ]

    notebook = nbf.v4.new_notebook(cells=cells, metadata=metadata())
    nbf.validate(notebook)
    nbf.write(notebook, path)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=("09", "00", "both"))
    args = parser.parse_args()

    paths = []
    if args.target in ("09", "both"):
        paths.append(build_09())
    if args.target in ("00", "both"):
        paths.append(build_00())
    for path in paths:
        print(path)
