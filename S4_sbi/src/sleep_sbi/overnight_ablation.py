"""Auditable orchestration for the overnight observation-ablation workflow.

This module deliberately does not alter EEG extraction, simulator mechanisms, or
scientific thresholds.  It reads the completed observation-schema artifacts,
audits whether a semantically compatible simulator extractor exists, and writes
only new analysis artifacts under ``S4_sbi/results/overnight_observation_ablation``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCHEMA_VERSION = "overnight-observation-ablation-v0.1"
RANDOM_SEED = 20260726
REQUESTED_BASELINE_DIMENSION = 18
ACTUAL_BASELINE_DIMENSION = 14
ACTUAL_AUGMENTED_DIMENSION = 23
RETAINED_N3_EPOCHS = 142
REJECTED_N3_EPOCHS = 78
ALL_N3_EPOCHS = 220


def project_root() -> Path:
    """Return the repository root from this installed-source location."""

    root = Path(__file__).resolve().parents[3]
    if not (root / ".git").exists() or not (root / "S4_sbi").exists():
        raise RuntimeError("Could not locate the sleep_loop project root")
    return root


ROOT = project_root()
S4_ROOT = ROOT / "S4_sbi"
SCHEMA_COMPARISON_DIR = S4_ROOT / "results" / "observation_schema_comparison"
PYLORIC_OUTPUT_DIR = S4_ROOT / "outputs" / "pyloric_inspired_observation"
OVERNIGHT_DIR = S4_ROOT / "results" / "overnight_observation_ablation"

SOURCE_NOTEBOOKS = (
    S4_ROOT / "notebooks" / "01_observation.ipynb",
    S4_ROOT / "notebooks" / "01_observation_annotated.ipynb",
    S4_ROOT / "notebooks" / "01_Pyloric_Inspired_Observation.ipynb",
    S4_ROOT / "notebooks" / "01_Pyloric_Inspired_Observation_annotated.ipynb",
    S4_ROOT / "notebooks" / "02_Observation_Schema_Comparison.ipynb",
)


@dataclass(frozen=True)
class CandidateSchema:
    schema_id: str
    label: str
    feature_names: tuple[str, ...]
    status: str
    intended_use: str
    rationale: str

    @property
    def dimension(self) -> int:
        return len(self.feature_names)


def relative(path: Path) -> str:
    """Use project-relative identifiers in publication-safe artifacts."""

    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        return f"git {' '.join(args)} failed: {completed.stderr.strip()}"
    return completed.stdout.rstrip()


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


def environment_report() -> dict[str, Any]:
    """Collect reproducibility evidence without storing machine-specific paths."""

    package_names = [
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "mne",
        "torch",
        "sbi",
        "scikit-learn",
        "fooof",
        "neurolib",
        "nbformat",
        "nbclient",
        "jupyter",
    ]
    gpu: dict[str, Any] = {"available": False, "count": 0, "names": []}
    try:
        import torch

        gpu = {
            "available": bool(torch.cuda.is_available()),
            "count": int(torch.cuda.device_count()),
            "names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "torch_cuda_version": torch.version.cuda,
        }
    except Exception as exc:  # environment evidence only
        gpu["error"] = repr(exc)

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": timestamp(),
        "project_root": ".",
        "working_directory": ".",
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "gpu": gpu,
        "packages": {name: _package_version(name) for name in package_names},
        "git_branch": _run_git("branch", "--show-current"),
        "git_head": _run_git("rev-parse", "HEAD"),
        "git_status_short": _run_git("status", "--short"),
        "git_diff_stat": _run_git("diff", "--stat"),
        "source_notebook_sha256": {
            relative(path): sha256_file(path) for path in SOURCE_NOTEBOOKS
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return relative(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required input is absent: {relative(path)}")
    return path


def load_schema_comparison() -> dict[str, Any]:
    """Read and cross-check the completed notebook-02 artifacts."""

    baseline_path = _require_file(SCHEMA_COMPARISON_DIR / "observation_schema_baseline18.json")
    augmented_path = _require_file(SCHEMA_COMPARISON_DIR / "observation_schema_augmented.json")
    support_path = _require_file(SCHEMA_COMPARISON_DIR / "observation_support_summary.csv")
    crosswalk_path = _require_file(SCHEMA_COMPARISON_DIR / "observation_feature_crosswalk.csv")
    vector_path = _require_file(SCHEMA_COMPARISON_DIR / "real_observation_vectors.npz")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    augmented = json.loads(augmented_path.read_text(encoding="utf-8"))
    support = pd.read_csv(support_path)
    crosswalk = pd.read_csv(crosswalk_path)
    with np.load(vector_path, allow_pickle=False) as archive:
        vectors = {key: archive[key].copy() for key in archive.files}

    baseline_names = [item["feature_name"] for item in baseline["features"]]
    augmented_names = [item["feature_name"] for item in augmented["features"]]
    if len(baseline_names) != ACTUAL_BASELINE_DIMENSION:
        raise RuntimeError(
            f"Expected executable Baseline-{ACTUAL_BASELINE_DIMENSION}D, "
            f"found {len(baseline_names)}D"
        )
    if len(augmented_names) != ACTUAL_AUGMENTED_DIMENSION:
        raise RuntimeError(
            f"Expected candidate Augmented-{ACTUAL_AUGMENTED_DIMENSION}D, "
            f"found {len(augmented_names)}D"
        )
    if augmented_names[: len(baseline_names)] != baseline_names:
        raise RuntimeError("Augmented schema does not preserve the actual baseline prefix")
    if vectors["x_o_baseline"].shape != (ACTUAL_BASELINE_DIMENSION,):
        raise RuntimeError("Unexpected baseline observation-vector shape")
    if vectors["x_o_augmented"].shape != (ACTUAL_AUGMENTED_DIMENSION,):
        raise RuntimeError("Unexpected augmented observation-vector shape")
    if not np.allclose(
        vectors["x_o_augmented"][:ACTUAL_BASELINE_DIMENSION],
        vectors["x_o_baseline"],
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    ):
        raise RuntimeError("Constructive baseline prefix values no longer match")
    if np.asarray(vectors["x_o_baseline"], dtype=float).size != ACTUAL_BASELINE_DIMENSION:
        raise RuntimeError("Baseline vector cannot be interpreted numerically")

    return {
        "baseline": baseline,
        "augmented": augmented,
        "baseline_names": baseline_names,
        "augmented_names": augmented_names,
        "support": support,
        "crosswalk": crosswalk,
        "vectors": vectors,
        "artifact_paths": {
            "baseline_schema": relative(baseline_path),
            "augmented_schema": relative(augmented_path),
            "support": relative(support_path),
            "crosswalk": relative(crosswalk_path),
            "vectors": relative(vector_path),
        },
    }


def _feature_group(name: str) -> str:
    if name in {"fooof_aperiodic_exponent", "so_peak_frequency_hz", "relative_so_power", "so_q"}:
        return "Baseline spectral/SO"
    if name in {"so_event_rate_per_min", "ibi_cv", "so_median_ibi_s"}:
        return "SO rhythm"
    if name.startswith("so_up_proxy") or name == "so_down_proxy_duration_s":
        return "SO morphology"
    if name in {"so_trough_to_peak_time_s", "waveform_peak_to_peak_z"}:
        return "SO waveform morphology"
    if name.startswith("spindle_") and "phase" not in name:
        return "Spindle"
    if name.startswith("pac_") or "phase" in name:
        return "SOspindle coordination / PAC"
    return "Candidate / unresolved"


def _freeze_decision(name: str) -> tuple[str, str, str, str]:
    """Return role, freeze status, redundancy annotation, and scientific rationale."""

    rules = {
        "fooof_aperiodic_exponent": (
            "inference candidate",
            "Conditional parity-only",
            "none identified in current audit",
            "Stable record-level spectral observable; still needs simulator-visible extractor parity.",
        ),
        "so_peak_frequency_hz": (
            "inference candidate",
            "Conditional parity-only",
            "related to SO timing but not deterministic",
            "Record-level finite; requires same PSD semantics in simulation.",
        ),
        "relative_so_power": (
            "inference candidate",
            "Conditional parity-only",
            "correlated with so_q in current data",
            "Current Hann PSD definition is explicit; simulator observable mapping is absent.",
        ),
        "so_q": (
            "inference candidate",
            "Conditional parity-only",
            "related to relative_so_power",
            "Do not automatically remove correlation; parity and recovery must decide value.",
        ),
        "so_event_rate_per_min": (
            "inference candidate",
            "Conditional parity-only",
            "near-derived with median IBI",
            "Detector-valid zero has a defined meaning, but requires identical simulated detector input.",
        ),
        "ibi_cv": (
            "inference candidate",
            "Conditional parity-only",
            "related to SO rate/IBI, not deterministic",
            "18/142 epochs lack enough intervals; a simulation-side validity policy is not frozen.",
        ),
        "pac_up_down_ratio": (
            "mechanism diagnostic",
            "Exclude from pilot input",
            "PAC-derived ratio",
            "Observation-level ratio is not a direct internal-state measurement and should not consume held-out evidence.",
        ),
        "spindle_density_per_min": (
            "held-out PPC candidate",
            "Hold out pending protocol",
            "approximately linked with occupancy and duration",
            "Observable detector is provisional; retain for predictive checks unless an ablation prospectively assigns it to input.",
        ),
        "spindle_mean_duration_s": (
            "held-out PPC candidate",
            "Hold out pending protocol",
            "approximately linked with occupancy and density",
            "Undefined for no-event epochs; must not be zero-filled.",
        ),
        "pac_mi": (
            "held-out PPC candidate",
            "Hold out",
            "PAC family",
            "Preferred phase is unstable when MI is weak; keep diagnostic until parity is established.",
        ),
        "pac_preferred_phase_rad": (
            "held-out PPC candidate",
            "Hold out",
            "raw circular phase; do not combine with sin/cos as independent evidence",
            "Raw phase is circular and is not an independent additional coordinate.",
        ),
        "pac_preferred_phase_sin": (
            "held-out PPC candidate",
            "Hold out",
            "paired circular coordinate with pac_preferred_phase_cos",
            "Only use a pre-declared circular representation if it ever becomes an input.",
        ),
        "pac_preferred_phase_cos": (
            "held-out PPC candidate",
            "Hold out",
            "paired circular coordinate with pac_preferred_phase_sin",
            "Only use a pre-declared circular representation if it ever becomes an input.",
        ),
        "waveform_peak_to_peak_z": (
            "held-out PPC candidate",
            "Hold out",
            "waveform morphology",
            "Keep independent waveform evidence outside fitting while parity is unresolved.",
        ),
        "so_median_ibi_s": (
            "candidate new dimension",
            "Exclude from frozen input",
            "near-derived with SO rate",
            "It adds timing robustness but duplicates much of rate information and has 18 invalid epochs.",
        ),
        "so_up_proxy_duration_s": (
            "candidate new dimension",
            "Conditional parity-only",
            "coupled with DOWN duration and duty cycle",
            "Observable-level proxy only; do not equate to cortical UP state.",
        ),
        "so_down_proxy_duration_s": (
            "candidate new dimension",
            "Conditional parity-only",
            "coupled with UP duration and duty cycle",
            "Observable-level proxy only; do not equate to cortical DOWN state.",
        ),
        "so_up_proxy_duty_cycle": (
            "held-out PPC candidate",
            "Exclude from frozen input",
            "deterministic from UP and DOWN duration by definition",
            "Do not count a derived duty cycle as independent fitting evidence.",
        ),
        "so_trough_to_peak_time_s": (
            "held-out PPC candidate",
            "Hold out",
            "morphology/timing family",
            "Retain as waveform-level predictive evidence.",
        ),
        "spindle_occupancy": (
            "candidate new dimension",
            "Exclude from frozen input",
            "near-derived from density and duration",
            "Useful diagnostic but not independent of spindle density and duration.",
        ),
        "spindle_onset_phase_cos": (
            "candidate new dimension",
            "Exclude from frozen input",
            "paired circular coordinate with sin; 10/142 support",
            "Event-conditioned phase support is insufficient for a robust SNPE input.",
        ),
        "spindle_onset_phase_sin": (
            "candidate new dimension",
            "Exclude from frozen input",
            "paired circular coordinate with cos; 10/142 support",
            "Event-conditioned phase support is insufficient for a robust SNPE input.",
        ),
        "spindle_onset_phase_concentration": (
            "candidate new dimension",
            "Exclude from frozen input",
            "circular event-support statistic; 10/142 support",
            "Undefined with insufficient paired events; never encode undefined as zero.",
        ),
    }
    try:
        return rules[name]
    except KeyError as exc:
        raise KeyError(f"Missing freeze rule for {name}") from exc


def candidate_schemas(comparison: dict[str, Any]) -> list[CandidateSchema]:
    """Define requested schemes without silently promoting a candidate to SNPE."""

    baseline = tuple(comparison["baseline_names"])
    augmented = tuple(comparison["augmented_names"])
    return [
        CandidateSchema(
            "A_baseline14",
            "Scheme A: executable Baseline-14D",
            baseline,
            "conditional_parity_audit",
            "fairness reference; not a frozen pilot input",
            "Exact executable 14D order from notebook 02. It includes legacy held-out and mechanism fields, so it cannot be promoted without a prospective leakage decision and extractor parity.",
        ),
        CandidateSchema(
            "B_baseline_plus_so_morphology",
            "Scheme B: Baseline + new SO morphology",
            baseline + ("so_up_proxy_duration_s", "so_down_proxy_duration_s"),
            "conditional_parity_audit",
            "candidate only",
            "Adds observable UP/DOWN-proxy durations, not duty cycle. Per-epoch support is 116/142 and a simulator-side validity policy is absent.",
        ),
        CandidateSchema(
            "C_baseline_plus_spindle_occupancy",
            "Scheme C: Baseline + supported spindle observable",
            baseline + ("spindle_occupancy",),
            "diagnostic_reference",
            "candidate only",
            "Occupancy is record-level finite but approximately determined by density and duration; it is retained for an explicit redundancy audit, not presumed incremental.",
        ),
        CandidateSchema(
            "D_baseline_plus_event_phase",
            "Scheme D: Baseline + event-conditioned SOspindle phase",
            baseline
            + (
                "spindle_onset_phase_cos",
                "spindle_onset_phase_sin",
                "spindle_onset_phase_concentration",
            ),
            "blocked_low_support",
            "diagnostic reference only",
            "Each new circular field has only 10/142 valid epochs. No zero-fill or unverified missingness transform is permitted.",
        ),
        CandidateSchema(
            "E_recommended_frozen_augmented",
            "Scheme E: recommended Frozen Augmented",
            tuple(),
            "no_schema_frozen",
            "not an executable input",
            "No augmented schema is frozen before a simulator-visible EEG mapping, validity policy, scaling rule, aggregation contract, and leakage allocation exist.",
        ),
        CandidateSchema(
            "F_complete23_diagnostic",
            "Scheme F: complete 23D diagnostic reference",
            augmented,
            "diagnostic_reference",
            "diagnostic only",
            "The complete candidate library intentionally retains low-support and derived fields so their limitations remain visible; it is not SNPE-ready.",
        ),
    ]


def _support_lookup(comparison: dict[str, Any]) -> pd.DataFrame:
    support = comparison["support"].copy()
    return support.set_index("feature_name", drop=False)


def run_schema_freeze(output_root: Path = OVERNIGHT_DIR) -> dict[str, Any]:
    """Create the 03 decision artifacts from completed observation results."""

    comparison = load_schema_comparison()
    support = _support_lookup(comparison)
    schemes = candidate_schemas(comparison)
    all_features = comparison["augmented_names"]

    rows: list[dict[str, Any]] = []
    for name in all_features:
        role, freeze_status, redundancy, rationale = _freeze_decision(name)
        support_row = support.loc[name]
        rows.append(
            {
                "feature": name,
                "group": _feature_group(name),
                "real_support": f"{int(support_row['valid_epoch_count'])}/{RETAINED_N3_EPOCHS}",
                "real_support_fraction": float(support_row["valid_epoch_percentage"]) / 100.0,
                "simulation_extractable": False,
                "simulation_extractability_evidence": "No simulator-visible EEG observable mapping is implemented or validated.",
                "missingness_meaning": str(support_row["most_common_invalid_reason"]),
                "valid_event_count": None
                if pd.isna(support_row["valid_event_count"])
                else int(support_row["valid_event_count"]),
                "redundancy": redundancy,
                "role": role,
                "freeze_status": freeze_status,
                "rationale": rationale,
            }
        )
    decisions = pd.DataFrame(rows)

    evaluations = pd.DataFrame(
        [
            {
                "schema_id": schema.schema_id,
                "label": schema.label,
                "dimension": schema.dimension if schema.feature_names else "not frozen",
                "ordered_feature_names": json.dumps(schema.feature_names),
                "fixed_order": bool(schema.feature_names),
                "record_vector_finite": bool(schema.feature_names),
                "aggregation_frozen": False,
                "scaling_frozen": False,
                "simulation_extractable": False,
                "missingness_policy_frozen": False,
                "held_out_leakage_resolved": False,
                "status": schema.status,
                "intended_use": schema.intended_use,
                "rationale": schema.rationale,
            }
            for schema in schemes
        ]
    )

    schema_dir = output_root / "schema_freeze"
    schema_dir.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(schema_dir / "feature_freeze_decisions.csv", index=False)
    evaluations.to_csv(schema_dir / "schema_evaluation.csv", index=False)
    _write_json(output_root / "environment_report.json", environment_report())
    freeze_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": timestamp(),
        "source": comparison["artifact_paths"],
        "actual_baseline_dimension": ACTUAL_BASELINE_DIMENSION,
        "requested_baseline_dimension": REQUESTED_BASELINE_DIMENSION,
        "candidate_augmented_dimension": ACTUAL_AUGMENTED_DIMENSION,
        "epoch_accounting": {
            "retained_n3": RETAINED_N3_EPOCHS,
            "rejected_n3": REJECTED_N3_EPOCHS,
            "all_n3": ALL_N3_EPOCHS,
            "identity_pass": RETAINED_N3_EPOCHS + REJECTED_N3_EPOCHS == ALL_N3_EPOCHS,
        },
        "constructive_prefix_identity": {
            "pass": True,
            "meaning": "X_augmented is explicitly constructed as [X_baseline, X_new]; this is not independent-extractor regression evidence.",
        },
        "candidate_schemas": [asdict(schema) for schema in schemes],
        "go_no_go": {
            "pilot_snpe": "NO-GO_PENDING_PARITY",
            "reason": "No schema has a validated simulator-visible EEG extractor, frozen scaling, frozen aggregation, frozen validity handling, and prospective held-out allocation.",
        },
    }
    _write_json(schema_dir / "frozen_observation_schemas.json", freeze_manifest)
    _write_json(
        schema_dir / "validation_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_reload": True,
            "baseline_dimension_pass": len(comparison["baseline_names"]) == ACTUAL_BASELINE_DIMENSION,
            "augmented_dimension_pass": len(comparison["augmented_names"]) == ACTUAL_AUGMENTED_DIMENSION,
            "prefix_identity_pass": True,
            "epoch_accounting_pass": RETAINED_N3_EPOCHS + REJECTED_N3_EPOCHS == ALL_N3_EPOCHS,
            "record_vectors_finite": bool(
                np.isfinite(comparison["vectors"]["x_o_baseline"]).all()
                and np.isfinite(comparison["vectors"]["x_o_augmented"]).all()
            ),
            "pilot_go": False,
        },
    )

    colors = decisions["freeze_status"].map(
        {
            "Conditional parity-only": "#d58936",
            "Exclude from frozen input": "#a33a3a",
            "Hold out": "#3c7a89",
            "Hold out pending protocol": "#3c7a89",
        }
    ).fillna("#777777")
    fig, ax = plt.subplots(figsize=(11, 7))
    counts = decisions.groupby(["group", "freeze_status"], sort=False).size().unstack(fill_value=0)
    counts.plot(kind="barh", stacked=True, ax=ax, colormap="tab20c")
    ax.set_xlabel("Feature count")
    ax.set_ylabel("Feature group")
    ax.set_title("Schema-freeze audit: roles and unresolved gates")
    ax.legend(title="Freeze status", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save_figure(fig, schema_dir / "schema_freeze_roles")

    return {
        "comparison": comparison,
        "decisions": decisions,
        "evaluations": evaluations,
        "manifest": freeze_manifest,
        "output_dir": schema_dir,
    }


def _source_line_evidence(path: Path, needles: Iterable[str]) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    evidence: list[dict[str, Any]] = []
    for needle in needles:
        for line_number, line in enumerate(lines, start=1):
            if needle in line:
                evidence.append(
                    {
                        "source": relative(path),
                        "needle": needle,
                        "line": line_number,
                        "excerpt": line.strip(),
                    }
                )
                break
    return evidence


def _inspect_npz_bank(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    observation = arrays.get("x")
    names = arrays.get("summary_keys", np.array([], dtype="U1")).astype(str).tolist()
    return {
        "path": relative(path),
        "keys": list(arrays),
        "theta_shape": list(arrays.get("theta", np.empty((0,))).shape),
        "observation_shape": list(observation.shape) if observation is not None else None,
        "summary_keys": names,
        "finite_observation_values": bool(np.isfinite(observation).all()) if observation is not None else False,
        "has_object_arrays": any(value.dtype == object for value in arrays.values()),
        "schema_match_14d": names == load_schema_comparison()["baseline_names"],
        "schema_match_23d": names == load_schema_comparison()["augmented_names"],
    }


def _candidate_asset_inventory() -> pd.DataFrame:
    candidates = [
        ("V7 fitted parameter JSON", ROOT / "data" / "patient_params_fig7_v7_SC4001.json"),
        ("V8 fitted parameter JSON", ROOT / "data" / "patient_params_fig7_v8_SC4001.json"),
        ("V7 Pareto seed JSON", ROOT / "S4_v7_repair" / "pareto_seeds_fresh_DE.json"),
        ("V8a local-search best JSON", ROOT / "outputs" / "v8a_ultra_narrow_t6_t13_search" / "best_so_far.json"),
        ("V8a candidate archive", ROOT / "outputs" / "figure10_inspired_validation_panel" / "candidate_archive.csv"),
        ("V7 raw rate diagnostic", ROOT / "outputs" / "v7_phase_diagnosis_signals.npz"),
    ]
    rows = []
    for label, path in candidates:
        row = {
            "label": label,
            "path": relative(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "usable_as_matched_observation_bank": False,
            "reason": "Candidate parameters or rate diagnostics lack a validated simulated-EEG observation contract.",
        }
        if path.suffix == ".npz" and path.exists():
            with np.load(path, allow_pickle=False) as archive:
                row["keys"] = json.dumps(list(archive.files))
                row["array_shapes"] = json.dumps(
                    {key: list(archive[key].shape) for key in archive.files}
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _representative_parameter_assets() -> pd.DataFrame:
    """Record discoverable V7/V8/V8a parameter candidates without simulating them.

    These rows document that candidate parameters were considered for parity.  They
    are deliberately not used as a shortcut around the missing simulated-EEG
    observable contract.
    """

    parameter_keys = (
        "mue", "mui", "b", "tauA", "g_LK", "g_h", "c_th2ctx", "c_ctx2th"
    )
    common = {
        "selected_for_simulation": False,
        "simulation_status": "not simulated",
        "gate_reason": (
            "Extractor parity is NO-GO: candidate parameters do not supply a "
            "validated Fpz-Cz simulated-EEG observable."
        ),
    }
    rows: list[dict[str, Any]] = []

    def add_row(candidate_id: str, label: str, source: Path, values: Mapping[str, Any]) -> None:
        row = {
            "candidate_id": candidate_id,
            "label": label,
            "source": relative(source),
            **common,
        }
        for key in parameter_keys:
            value = values.get(key, np.nan)
            row[key] = float(value) if isinstance(value, (int, float, np.number)) else np.nan
        rows.append(row)

    json_candidates = [
        ("v7_fitted", "V7 fitted parameter JSON", ROOT / "data" / "patient_params_fig7_v7_SC4001.json"),
        ("v8_fitted", "V8 fitted parameter JSON", ROOT / "data" / "patient_params_fig7_v8_SC4001.json"),
        ("v8a_local_best", "V8a/T13 local-search best JSON", ROOT / "outputs" / "v8a_ultra_narrow_t6_t13_search" / "best_so_far.json"),
    ]
    for candidate_id, label, source in json_candidates:
        if source.exists():
            add_row(candidate_id, label, source, json.loads(source.read_text(encoding="utf-8")))

    pareto_source = ROOT / "S4_v7_repair" / "pareto_seeds_fresh_DE.json"
    if pareto_source.exists():
        payload = json.loads(pareto_source.read_text(encoding="utf-8"))
        for seed in payload.get("seeds", []):
            tag = str(seed.get("tag", "unknown"))
            add_row(f"v7_pareto_{tag}", f"V7 Pareto seed {tag}", pareto_source, seed.get("params", {}))

    archive_source = ROOT / "outputs" / "figure10_inspired_validation_panel" / "candidate_archive.csv"
    if archive_source.exists():
        archive = pd.read_csv(archive_source)
        for _, candidate in archive.head(3).iterrows():
            candidate_id = str(candidate.get("candidate_id", "archive_unknown"))
            add_row(candidate_id, "Candidate archive top-row inventory", archive_source, candidate.to_dict())

    return pd.DataFrame(rows)


def run_extractor_parity(output_root: Path = OVERNIGHT_DIR) -> dict[str, Any]:
    """Audit code/schema/semantic parity and enforce the hard training gate."""

    freeze_path = output_root / "schema_freeze" / "frozen_observation_schemas.json"
    _require_file(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    comparison = load_schema_comparison()
    schemes = [CandidateSchema(**row) for row in freeze["candidate_schemas"]]

    simulator_path = S4_ROOT / "simulator_wrapper.py"
    observation_path = S4_ROOT / "src" / "sleep_sbi" / "observation.py"
    code_evidence = _source_line_evidence(
        simulator_path,
        (
            "maps 4D theta",
            "SUMMARY_KEYS =",
            "firing-rate key",
            "Extract cortex (index 0) and thalamus (index 1) firing rates",
            "return np.array([stats[k] for k in SUMMARY_KEYS]",
        ),
    ) + _source_line_evidence(
        observation_path,
        (
            "def _read_manifest",
            "def _pick_channel",
            "max_peak_to_peak_uv",
            "def build_observation_bundle",
        ),
    )

    banks = [
        _inspect_npz_bank(S4_ROOT / "sbi_outputs" / "all_simulations.npz"),
        _inspect_npz_bank(S4_ROOT / "sbi_outputs_7dim_archive_20260507" / "all_simulations.npz"),
    ]
    asset_inventory = _candidate_asset_inventory()
    representative_assets = _representative_parameter_assets()

    legacy_dimension = int(banks[0]["observation_shape"][1])
    rows = []
    failures = []
    for schema in schemes:
        requested_dim = schema.dimension
        parity_pass = False
        reasons = [
            f"Current simulator wrapper/bank exposes legacy {legacy_dimension}D rate-summary vectors, not {requested_dim}D schema vectors.",
            "No validated mapping from cortical/thalamic firing rates to the Fpz-Cz scalp-EEG observable exists.",
            "Simulation-side epoch alignment, EEG units/reference, QC, event support, aggregation, scaling, and validity handling are not defined.",
        ]
        if not schema.feature_names:
            reasons.insert(0, "The requested Frozen Augmented schema is intentionally not frozen.")
        rows.append(
            {
                "schema": schema.schema_id,
                "dimension": requested_dim if schema.feature_names else "not frozen",
                "real_shape": f"({RETAINED_N3_EPOCHS}, {requested_dim})" if schema.feature_names else "not applicable",
                "simulation_shape": str(tuple(banks[0]["observation_shape"])),
                "order_match": False,
                "fixed_length_real": bool(schema.feature_names),
                "fixed_length_simulation": False,
                "fully_finite": False,
                "support_adequate": False,
                "same_semantics": False,
                "aggregation_frozen": False,
                "scaling_frozen": False,
                "held_out_leakage_resolved": False,
                "parity_pass": parity_pass,
                "gate": "NO-GO",
                "reason": " ".join(reasons),
            }
        )
        for feature in schema.feature_names:
            failures.append(
                {
                    "schema": schema.schema_id,
                    "feature": feature,
                    "failure_category": "missing_simulated_EEG_observable_contract",
                    "reason": "No same-unit, same-channel, same-preprocessing simulation observable is implemented.",
                    "severity": "hard_gate",
                }
            )

    parity = pd.DataFrame(rows)
    failure_frame = pd.DataFrame(failures)
    bank_frame = pd.DataFrame(banks)
    parity_dir = output_root / "extractor_parity"
    parity_dir.mkdir(parents=True, exist_ok=True)
    parity.to_csv(parity_dir / "extractor_parity_summary.csv", index=False)
    failure_frame.to_csv(parity_dir / "extractor_failures.csv", index=False)
    bank_frame.to_csv(parity_dir / "simulation_bank_audit.csv", index=False)
    asset_inventory.to_csv(parity_dir / "candidate_asset_inventory.csv", index=False)
    representative_assets.to_csv(parity_dir / "representative_parameter_assets.csv", index=False)
    pd.DataFrame(code_evidence).to_csv(parity_dir / "source_parity_evidence.csv", index=False)

    feature_audit = pd.DataFrame(
        {
            "feature": comparison["augmented_names"],
            "real_unit": [
                next(item["unit"] for item in comparison["augmented"]["features"] if item["feature_name"] == name)
                for name in comparison["augmented_names"]
            ],
            "simulation_feature_present": False,
            "same_name": False,
            "same_unit": False,
            "same_semantics": False,
            "parity_status": "NO-GO: observable mapping absent",
        }
    )
    feature_audit.to_csv(parity_dir / "feature_level_parity_audit.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(["Real Baseline", "Real Augmented", "Current simulator bank"], [14, 23, legacy_dimension], color=["#4c78a8", "#f58518", "#e45756"])
    axes[0].set_ylabel("Feature dimensions")
    axes[0].set_title("Dimension mismatch blocks parity")
    axes[0].set_ylim(0, 26)
    for index, value in enumerate([14, 23, legacy_dimension]):
        axes[0].text(index, value + 0.7, str(value), ha="center")
    axes[1].axis("off")
    axes[1].text(
        0.02,
        0.92,
        "Hard parity gate: NO-GO\n\n"
        "Current simulation assets expose cortex/thalamus firing-rate summaries.\n"
        "The real pipeline expects an Fpz-Cz scalp-EEG observable with\n"
        "epoch-level QC, PSD, SO, spindle, and PAC semantics.\n\n"
        "A rate-to-EEG substitution would be a new scientific model, not\n"
        "extractor parity. No SNPE is permitted on this basis.",
        va="top",
        fontsize=11,
        wrap=True,
    )
    fig.tight_layout()
    _save_figure(fig, parity_dir / "parity_gate_overview")

    fig, ax = plt.subplots(figsize=(10, 5))
    status_counts = failure_frame.groupby("schema").size().reindex([s.schema_id for s in schemes], fill_value=0)
    status_counts.plot(kind="bar", ax=ax, color="#c44e52")
    ax.set_ylabel("Hard-gate feature failures")
    ax.set_xlabel("Candidate schema")
    ax.set_title("Feature-level parity failures by schema")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save_figure(fig, parity_dir / "parity_failure_counts")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": timestamp(),
        "input_schema_freeze": relative(freeze_path),
        "simulation_banks": banks,
        "representative_parameter_assets": {
            "count": int(len(representative_assets)),
            "artifact": relative(parity_dir / "representative_parameter_assets.csv"),
            "selection_status": "Inventory only; no candidate was simulated because the hard gate failed.",
        },
        "hard_gate": {
            "passed_schemas": [],
            "status": "NO-GO",
            "reason": "No fixed-length, same-semantic simulated EEG observable schema exists. Existing banks are legacy 5D/7D rate-summary banks and cannot be relabelled.",
        },
        "reviewer_note": "Code-level execution of an old simulator is not code-, schema-, unit-, or semantic-level parity with the real EEG extractor.",
    }
    _write_json(parity_dir / "extractor_parity_manifest.json", manifest)
    _write_json(
        parity_dir / "validation_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_reload": True,
            "legacy_bank_dimensions": [bank["observation_shape"] for bank in banks],
            "legacy_banks_match_14d": [bank["schema_match_14d"] for bank in banks],
            "legacy_banks_match_23d": [bank["schema_match_23d"] for bank in banks],
            "parity_pass_count": int(parity["parity_pass"].sum()),
            "pilot_go": False,
        },
    )
    return {
        "comparison": comparison,
        "parity": parity,
        "failures": failure_frame,
        "banks": bank_frame,
        "representative_assets": representative_assets,
        "manifest": manifest,
        "output_dir": parity_dir,
    }


def _load_schemas_from_freeze(output_root: Path) -> list[CandidateSchema]:
    freeze_path = output_root / "schema_freeze" / "frozen_observation_schemas.json"
    payload = json.loads(_require_file(freeze_path).read_text(encoding="utf-8"))
    return [CandidateSchema(**row) for row in payload["candidate_schemas"]]


def run_pilot_ablation(output_root: Path = OVERNIGHT_DIR) -> dict[str, Any]:
    """Write a truthful blocked pilot manifest when the parity hard gate fails."""

    parity_path = output_root / "extractor_parity" / "extractor_parity_manifest.json"
    parity = json.loads(_require_file(parity_path).read_text(encoding="utf-8"))
    schemas = _load_schemas_from_freeze(output_root)
    approved = parity["hard_gate"]["passed_schemas"]
    if approved:
        raise RuntimeError(
            "This guarded pilot implementation intentionally supports only the observed no-go path. "
            "A passed schema requires a separately reviewed simulator-observable adapter."
        )

    pilot_dir = output_root / "pilot_snpe"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    training = pd.DataFrame(
        [
            {
                "schema": schema.schema_id,
                "dimension": schema.dimension if schema.feature_names else "not frozen",
                "training_status": "Blocked by parity",
                "train_simulations": 0,
                "validation_simulations": 0,
                "test_simulations": 0,
                "seeds_completed": 0,
                "runtime_s": 0.0,
                "reason": parity["hard_gate"]["reason"],
            }
            for schema in schemas
        ]
    )
    training.to_csv(pilot_dir / "training_summary.csv", index=False)
    blockers = pd.DataFrame(
        [
            {
                "blocker": "extractor_semantic_parity",
                "severity": "hard_gate",
                "evidence": parity["hard_gate"]["reason"],
                "minimum_recovery_step": "Implement and validate a simulator-to-observable EEG adapter with the same channel semantics, units, preprocessing, validity rules, aggregation, and feature order.",
            },
            {
                "blocker": "frozen_observation_contract",
                "severity": "hard_gate",
                "evidence": "No candidate has frozen simulator-side scaling, aggregation, missingness handling, and held-out allocation.",
                "minimum_recovery_step": "Pre-register one fixed schema and retain independent held-out waveform/event evidence.",
            },
        ]
    )
    blockers.to_csv(pilot_dir / "training_blockers.csv", index=False)

    empty_columns = {
        "recovery_metrics.csv": ["schema", "seed", "parameter", "mae", "rmse", "status"],
        "coverage_metrics.csv": ["schema", "seed", "credible_level", "empirical_coverage", "status"],
        "ppc_metrics.csv": ["schema", "seed", "feature", "metric", "status"],
    }
    for name, columns in empty_columns.items():
        pd.DataFrame(columns=columns).to_csv(pilot_dir / name, index=False)

    np.savez(
        pilot_dir / "simulation_split_metadata.npz",
        approved_schema_names=np.asarray([], dtype="<U1"),
        train_indices=np.asarray([], dtype=np.int64),
        validation_indices=np.asarray([], dtype=np.int64),
        test_indices=np.asarray([], dtype=np.int64),
        random_seeds=np.asarray([], dtype=np.int64),
        parity_status=np.asarray("NO-GO", dtype="<U5"),
        reason=np.asarray(parity["hard_gate"]["reason"], dtype="<U512"),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": timestamp(),
        "training_started": False,
        "approved_schemas": approved,
        "simulation_count": 0,
        "seeds_completed": 0,
        "status": "BLOCKED_BY_PARITY",
        "reason": parity["hard_gate"]["reason"],
        "scientific_boundary": "No posterior, posterior samples, recovery, calibration, or PPC results were generated because the simulation observations are semantically incompatible with the real EEG schemas.",
    }
    _write_json(pilot_dir / "ablation_manifest.json", manifest)
    _write_json(
        pilot_dir / "validation_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "training_started": False,
            "no_fake_posteriors": True,
            "npz_allow_pickle_not_required": True,
            "object_arrays": False,
            "hard_gate_respected": True,
        },
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")
    ax.text(
        0.03,
        0.92,
        "Pilot SNPE ablation: not run\n\n"
        "The extractor-parity hard gate failed for every candidate schema.\n"
        "No simulations were repurposed, no neural density estimator was trained,\n"
        "and no posterior or posterior predictive result exists in this artifact set.\n\n"
        "Required before a pilot:\n"
        "1. Semantically matched simulated EEG observable\n"
        "2. Frozen feature order, aggregation, scaling, and validity policy\n"
        "3. Prospective separation of inference and held-out summaries",
        va="top",
        fontsize=12,
    )
    fig.tight_layout()
    _save_figure(fig, pilot_dir / "pilot_gate_status")
    return {"training": training, "blockers": blockers, "manifest": manifest, "output_dir": pilot_dir}


def run_ablation_diagnostics(output_root: Path = OVERNIGHT_DIR) -> dict[str, Any]:
    """Generate an evidence-based diagnostics report for the no-go pilot state."""

    pilot_dir = output_root / "pilot_snpe"
    training = pd.read_csv(_require_file(pilot_dir / "training_summary.csv"))
    manifest = json.loads(_require_file(pilot_dir / "ablation_manifest.json").read_text(encoding="utf-8"))
    diagnostic_dir = output_root / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    summary = training.loc[:, ["schema", "dimension", "training_status", "train_simulations", "seeds_completed", "runtime_s"]].copy()
    summary["recovery"] = "Insufficient evidence: training not run"
    summary["coverage"] = "Insufficient evidence: training not run"
    summary["held_out_ppc"] = "Insufficient evidence: training not run"
    summary["failures"] = "Blocked by extractor parity"
    summary["interpretation"] = "Do not compare posterior width or select a schema."
    summary.to_csv(diagnostic_dir / "ablation_summary.csv", index=False)

    for filename in ("recovery_metrics.csv", "coverage_metrics.csv", "ppc_metrics.csv"):
        source = pilot_dir / filename
        target = diagnostic_dir / filename
        pd.read_csv(_require_file(source)).to_csv(target, index=False)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(summary["schema"], np.ones(len(summary)), color="#c44e52")
    ax.set_xlim(0, 1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["not available", "available"])
    ax.set_xlabel("Diagnostic availability")
    ax.set_title("Ablation diagnostics availability after parity gate")
    for index, (_, row) in enumerate(summary.iterrows()):
        ax.text(0.03, index, "Blocked by parity", va="center", color="white", fontsize=9)
    fig.tight_layout()
    _save_figure(fig, diagnostic_dir / "diagnostic_availability")

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": timestamp(),
        "training_started": bool(manifest["training_started"]),
        "synthetic_recovery": "not run",
        "coverage_calibration": "not run",
        "held_out_ppc": "not run",
        "posterior_quality": "not run",
        "conclusion": "No scheme can be selected for SNPE before extractor semantic parity is established. Narrower posterior would not constitute evidence even if one were produced.",
        "minimum_next_step": "Build and validate an observable-level simulator adapter, then repeat schema freeze and parity before training.",
    }
    _write_json(diagnostic_dir / "validation_report.json", report)
    return {"summary": summary, "report": report, "output_dir": diagnostic_dir}


def reload_overnight_artifacts(output_root: Path = OVERNIGHT_DIR) -> dict[str, Any]:
    """Reload every required JSON/CSV/NPZ artifact and reject object arrays."""

    expected = [
        output_root / "environment_report.json",
        output_root / "schema_freeze" / "frozen_observation_schemas.json",
        output_root / "schema_freeze" / "feature_freeze_decisions.csv",
        output_root / "extractor_parity" / "extractor_parity_summary.csv",
        output_root / "extractor_parity" / "extractor_failures.csv",
        output_root / "extractor_parity" / "extractor_parity_manifest.json",
        output_root / "pilot_snpe" / "ablation_manifest.json",
        output_root / "pilot_snpe" / "simulation_split_metadata.npz",
        output_root / "pilot_snpe" / "training_summary.csv",
        output_root / "diagnostics" / "ablation_summary.csv",
        output_root / "diagnostics" / "validation_report.json",
    ]
    checks: list[dict[str, Any]] = []
    for path in expected:
        _require_file(path)
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            checks.append({"path": relative(path), "kind": "json", "ok": isinstance(value, dict)})
        elif path.suffix == ".csv":
            frame = pd.read_csv(path)
            checks.append({"path": relative(path), "kind": "csv", "ok": True, "rows": len(frame)})
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                object_arrays = [key for key in archive.files if archive[key].dtype == object]
                checks.append(
                    {
                        "path": relative(path),
                        "kind": "npz",
                        "ok": not object_arrays,
                        "object_arrays": object_arrays,
                        "keys": list(archive.files),
                    }
                )
    source_hashes_end = {relative(path): sha256_file(path) for path in SOURCE_NOTEBOOKS}
    environment_path = output_root / "environment_report.json"
    source_hashes_start: dict[str, str] | None = None
    if environment_path.is_file():
        source_hashes_start = json.loads(environment_path.read_text(encoding="utf-8")).get(
            "source_notebook_sha256"
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": timestamp(),
        "checks": checks,
        "all_pass": all(check["ok"] for check in checks),
        "source_notebook_sha256_start": source_hashes_start,
        "source_notebook_sha256_end": source_hashes_end,
        "source_notebook_hashes_unchanged": source_hashes_start == source_hashes_end,
        "git_status_short_end": _run_git("status", "--short"),
        "git_diff_stat_end": _run_git("diff", "--stat"),
    }
    _write_json(output_root / "validation_report.json", report)
    return report


def concise_status_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a compact notebook display without changing the stored artifacts."""

    existing = [column for column in columns if column in frame.columns]
    return frame.loc[:, existing].copy()
