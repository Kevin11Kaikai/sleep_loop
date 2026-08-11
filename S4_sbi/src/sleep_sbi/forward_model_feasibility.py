"""Audit the feasibility of an EEG measurement model for the sleep-loop model.

This module deliberately does not implement a rate-to-voltage conversion.  It
records which model states exist, which spatial and biophysical declarations a
forward model would need, and exposes a strict gate that remains closed until
those declarations and independent validation evidence are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "forward-model-feasibility-v0.1"


class MeasurementContractError(RuntimeError):
    """Raised when a declared EEG measurement contract is incomplete."""


@dataclass(frozen=True)
class ContractGate:
    """Result of validating a proposed model-to-EEG measurement contract."""

    status: str
    missing_fields: tuple[str, ...]
    invalid_fields: tuple[str, ...]
    leakage_fields: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "GO"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "missing_fields": list(self.missing_fields),
            "invalid_fields": list(self.invalid_fields),
            "leakage_fields": list(self.leakage_fields),
            "warnings": list(self.warnings),
        }


REQUIRED_DECLARATION_FIELDS = (
    "source_variable",
    "source_physical_definition",
    "source_native_unit",
    "source_to_dipole_model",
    "dipole_moment_unit",
    "source_coordinates_mri_mm",
    "source_orientation_xyz",
    "source_extent_or_patch_area",
    "head_model_id",
    "head_to_mri_transform_id",
    "conductivity_model",
    "leadfield_id",
    "sensor_names",
    "reference_operator",
    "gain_calibration_id",
    "sensor_noise_model_id",
    "calibration_dataset_id",
    "validation_dataset_id",
    "calibration_independent_of_sc4001",
    "validation_independent_of_sc4001",
    "output_unit",
    "resample_hz",
    "epoch_duration_s",
)


def source_candidate_decisions() -> pd.DataFrame:
    """Return the code-grounded cortical source candidate decision matrix."""

    columns = [
        "candidate",
        "available_state",
        "physical_meaning",
        "units",
        "directionality",
        "spatial_location",
        "dipole_interpretation",
        "required_reconstruction",
        "identifiability_risk",
        "compatibility_with_lead_field",
        "scientific_support",
        "decision",
        "evidence",
    ]
    rows = [
        (
            "cortical_r_mean_EXC",
            "Yes; saved by the project",
            "Excitatory-population firing rate",
            "Hz after project kHz-to-Hz conversion",
            "None",
            "Abstract ALN cortical node; no coordinate or cortical patch",
            "No primary-current or dipole-moment definition",
            "None for rate; a separate source model would be required",
            "High: rate dynamics and measurement gain/source geometry can trade off",
            "No: a lead field multiplies a declared current-dipole source",
            "Strong as an ALN state; insufficient as scalp-EEG source",
            "Route-3 observable only; reject as direct Route-2 source",
            "models/s4_personalize_fig7_v7.py:226-227,913-924",
        ),
        (
            "cortical_r_mean_INH",
            "Yes; saved by the project",
            "Inhibitory-population firing rate",
            "Hz after project kHz-to-Hz conversion",
            "None",
            "Same abstract ALN cortical node",
            "No dipole moment; inhibitory populations do not define a sensor orientation",
            "Separate source model and population geometry",
            "High",
            "No",
            "Strong as an ALN state; insufficient as scalp-EEG source",
            "Route-3 diagnostic only",
            "neurolib/.../builder/aln.py:577-606,641-705",
        ),
        (
            "equal_weight_E_minus_I_rate",
            "Derived from two saved rates",
            "Arbitrary equal-weight rate contrast",
            "Hz",
            "Sign comes from an analyst-chosen subtraction",
            "Same abstract node",
            "No calibrated relationship to aligned pyramidal-cell current dipoles",
            "Physiological E/I weights and source geometry",
            "Very high: weights, gain, and theta can compensate one another",
            "No",
            "No project or literature calibration for equal weights",
            "Exclude from Route 2; shape-screening diagnostic only",
            "S4_sbi/src/sleep_sbi/eeg_observation_mapping_audit.py:319-337",
        ),
        (
            "equal_weight_E_plus_I_rate",
            "Derived from two saved rates",
            "Arbitrary equal-weight rate sum",
            "Hz",
            "None",
            "Same abstract node",
            "No dipole direction or physical source moment",
            "Physiological weights and source geometry",
            "Very high",
            "No",
            "No project calibration",
            "Exclude from Route 2",
            "S4_sbi/src/sleep_sbi/eeg_observation_mapping_audit.py:319-337",
        ),
        (
            "cortical_I_mu_EXC",
            "Yes; ALN state, audit-recorded",
            "Mean effective current/drive state of the excitatory mass",
            "mV/ms in neurolib convention",
            "None",
            "Abstract cortical population",
            "A temporal drive is not a vector current dipole",
            "Source-current density, patch geometry, and dipole orientation",
            "High",
            "No, not without a source-to-dipole definition",
            "Equation-level support as ALN dynamics",
            "Route-2 research candidate only; not forward-ready",
            "neurolib/.../builder/aln.py:414-421,511-558",
        ),
        (
            "cortical_I_mu_INH",
            "Yes; ALN state, audit-recorded",
            "Mean effective current/drive state of the inhibitory mass",
            "mV/ms in neurolib convention",
            "None",
            "Abstract cortical population",
            "Not a dipole source and lacks pyramidal-cell geometry",
            "Source-current density, patch geometry, and orientation",
            "High",
            "No",
            "Equation-level support as ALN dynamics",
            "Route-2 research diagnostic only",
            "neurolib/.../builder/aln.py:577-705",
        ),
        (
            "cortical_adaptation_I_A",
            "Yes; ALN excitatory state, audit-recorded",
            "Population adaptation-current state used in the excitatory transfer-function drive",
            "pA in neurolib parameterization",
            "No macroscopic orientation",
            "Abstract cortical population",
            "A cellular/population adaptation current is not a calibrated primary-current dipole",
            "Population size, current density, laminar geometry, and orientation",
            "High",
            "No",
            "Biophysically motivated internal current; no scalp mapping",
            "Mechanism diagnostic; do not project directly",
            "neurolib/.../builder/aln.py:414-421,535-547",
        ),
        (
            "cortical_effective_drive",
            "Derived from I_mu_EXC - I_A/C",
            "Input coordinate passed to ALN firing-rate, voltage, and tau lookup tables",
            "mV/ms",
            "None",
            "Abstract cortical population",
            "Transfer-function input, not an extracellular source",
            "A separate primary-current source model",
            "High",
            "No",
            "Exactly traceable in code but not an EEG generator",
            "Exclude as direct Route-2 source",
            "neurolib/.../builder/aln.py:534-544",
        ),
        (
            "individual_synaptic_mean_states",
            "Yes; four ALN state variables, audit-recorded",
            "Dimensionless excitatory/inhibitory synaptic mean states driving EXC and INH masses",
            "dimensionless state",
            "None",
            "Abstract cortical populations",
            "Not a current until combined with model couplings; still lacks dipole geometry",
            "Apply J terms, separate target populations, define source-current density and orientation",
            "Very high: multiple uncalibrated combinations produce similar signals",
            "No, not in native form",
            "Equation-level support as synaptic states",
            "Retain for source-model derivation audit; do not sum arbitrarily",
            "neurolib/.../builder/aln.py:414-421,539-556",
        ),
        (
            "reconstructed_population_mean_voltage",
            "Not saved; deterministically available from the ALN voltage lookup if I_sigma is reconstructed",
            "Lookup-table estimate of population mean membrane voltage",
            "mV",
            "No macroscopic dipole orientation",
            "Abstract cortical population",
            "Membrane voltage is not itself a primary-current dipole moment",
            "Reconstruct I_sigma/couplings and evaluate V_mean_ss without changing dynamics",
            "High: voltage, source moment, gain, and geometry remain separable",
            "Only after an independently justified voltage-to-dipole source model",
            "Best-supported voltage-level candidate in current ALN equations",
            "Preferred candidate for future source-model research, not approved for EEG inference",
            "neurolib/.../builder/aln.py:234-246,344-357,534-537",
        ),
        (
            "transmembrane_or_synaptic_current_dipole_proxy",
            "No complete state/output with traceable dipole units",
            "Desired primary-current or dipole-moment source for EEG forward modeling",
            "Would need A m, nA m, or current-density times geometry",
            "Required and currently absent",
            "Requires cortical patch position, extent, laminar axis, and population geometry",
            "Potentially compatible if derived and independently calibrated",
            "New source derivation from ALN currents plus population/laminar geometry",
            "High but scientifically addressable with independent constraints",
            "Potentially yes",
            "Forward-model literature supports current dipoles, not arbitrary firing-rate scaling",
            "Most relevant Route-2 target, but currently unavailable; NO-GO",
            "EEG forward-model literature; no implementation in sleep_loop",
        ),
        (
            "thalamic_TCR_or_TRN_voltage",
            "Yes; internal thalamic states",
            "Mean membrane voltage of TCR/TRN masses",
            "mV",
            "No declared deep-source orientation",
            "Abstract thalamic node, not assigned to cortical surface source space",
            "Deep internal voltage is not a scalp cortical dipole",
            "A validated deep-source geometry and full volume conductor",
            "Extreme",
            "Not with the current cortical AAL/surface lead-field path",
            "Valid thalamic mechanism state only",
            "Exclude from direct scalp projection; mechanism diagnostic",
            "neurolib/.../builder/thalamus.py:125-135,229-238,304-311",
        ),
    ]
    return pd.DataFrame(rows, columns=columns)


def measurement_contract_components() -> pd.DataFrame:
    """Describe every stage needed from model parameters to Fpz-Cz features."""

    columns = [
        "stage_index",
        "component",
        "required_definition",
        "current_status",
        "availability_class",
        "evidence_or_owner",
        "leakage_constraint",
        "gate",
    ]
    rows = [
        (1, "theta", "Eight model parameters with frozen bounds/order", "Available", "current_code", "V7/V8 parameter files", "Do not tune bounds on SC4001 fit outcomes", "PASS"),
        (2, "neurolib_dynamics", "Versioned ALN-cortex + thalamus equations, seeds, duration, warm-up", "Available", "current_code", "V7/V8 model modules and adapter", "Freeze before any bank", "PASS"),
        (3, "declared_cortical_source", "One explicitly defined primary source variable", "No source approved", "researcher_decision", "Candidate matrix", "Cannot select by best SC4001 spectral match", "NO-GO"),
        (4, "source_reconstruction", "Deterministic state-to-source equation with physical units", "Absent", "literature_and_implementation", "ALN exposes states but no dipole equation", "Cannot fit arbitrary weights to SC4001", "NO-GO"),
        (5, "source_geometry_orientation", "Coordinates, patch extent, cortical-normal orientation, source count", "Absent", "external_anatomy_and_researcher_decision", "Two-node model has no coordinates/orientation", "Do not choose location/orientation to maximize target match", "NO-GO"),
        (6, "head_volume_conductor", "MRI/template geometry, conductivities, BEM and head-MRI transform", "Generic utility only; no project instance", "external_data_and_code", "neurolib/utils/leadfield.py + MNE", "Template choice must be declared before target comparison", "NO-GO"),
        (7, "lead_field", "Versioned sensor-by-source gain matrix with fixed orientation", "Not generated for this model", "reconstructable_after_4_to_6", "MNE make_forward_solution", "No target-conditioned sensor/source selection", "NO-GO"),
        (8, "sensor_potentials", "Separate physical Fpz and Cz potentials before referencing", "Absent", "requires_lead_field", "V = L q", "Cannot estimate two gains from the same SC4001 target", "NO-GO"),
        (9, "Fpz_minus_Cz_reference", "Declared bipolar subtraction after sensor projection", "Operator known; inputs absent", "partly_available", "Observed channel label EEG Fpz-Cz", "Reference operation must not be replaced by a free scale", "NO-GO"),
        (10, "physical_gain_and_unit", "Dipole moment to volts/uV without arbitrary fitted constant", "Absent", "external_calibration", "No current project evidence", "SC4001 target cannot calibrate gain for its own inference", "NO-GO"),
        (11, "sensor_reference_noise", "Noise/artifact/reference model independent of theta", "Absent", "external_calibration", "No current project evidence", "Fit on independent recordings or pre-register", "NO-GO"),
        (12, "resampling_100_hz", "Anti-aliased conversion from simulated sensor voltage to 100 Hz", "Implementation available", "current_code", "scipy/MNE resampling path", "Freeze before bank", "PASS"),
        (13, "native_30_s_epochs", "Preserve independent 30-second windows; no concatenation", "Implementation available", "current_code", "Observation and simulator adapters", "Freeze before bank", "PASS"),
        (14, "EEG_feature_extractor", "Apply the unchanged uV/channel-semantic extractor", "Available but gated by missing sensor voltage", "current_code", "sleep_sbi.observation", "Do not apply uV thresholds to native model states", "BLOCKED"),
        (15, "independent_validation", "Validate amplitude, spectrum, SO, spindle and PAC on held-out data", "Absent", "external_data", "Required before real-EEG inference", "Do not validate on SC4001 inference target", "NO-GO"),
    ]
    return pd.DataFrame(rows, columns=columns)


def calibration_leakage_matrix() -> pd.DataFrame:
    """List which measurement choices require independent calibration."""

    columns = [
        "quantity",
        "may_be_fixed_from_code",
        "needs_literature",
        "needs_researcher_decision",
        "needs_external_calibration",
        "using_sc4001_target_would_leak",
        "acceptable_evidence",
    ]
    rows = [
        ("dynamics and theta order", True, False, True, False, True, "versioned model/config before inference"),
        ("cortical source variable", False, True, True, True, True, "biophysical derivation plus independent validation"),
        ("source coordinates/extent/orientation", False, True, True, True, True, "anatomical atlas/MRI independent of SC4001 outcome"),
        ("conductivity/BEM/template", False, True, True, False, False, "pre-registered MNE/fsaverage or subject MRI contract"),
        ("lead-field matrix", False, False, False, False, False, "deterministic output after geometry and sensor info are fixed"),
        ("Fpz-Cz reference", True, False, False, False, False, "explicit [1, -1] sensor operator"),
        ("source-to-dipole gain", False, True, True, True, True, "independent EEG/MEG/LFP calibration or validated literature"),
        ("sensor/reference noise", False, True, True, True, True, "independent N3 recordings or instrument model"),
        ("100 Hz resampling", True, False, True, False, False, "frozen anti-alias implementation"),
        ("30 s epoch policy", True, False, True, False, False, "same contract on real and simulated sensor voltage"),
        ("feature thresholds", True, False, False, False, True, "existing EEG extractor, unchanged after measurement validation"),
    ]
    return pd.DataFrame(rows, columns=columns)


def leadfield_requirement_matrix() -> pd.DataFrame:
    """Summarize inputs required by neurolib's MNE-based lead-field utility."""

    columns = ["requirement", "leadfield_utility_use", "provided_by_current_two_node_model", "status", "evidence"]
    rows = [
        ("EEG sensor info and locations", "MNE Raw/Info and montage", False, "missing_model_mapping", "leadfield.py:190-241"),
        ("head-to-MRI transform", "trans file for coregistration", False, "missing", "leadfield.py:98-116,244-263"),
        ("cortical surface source space", "dipole coordinates on cortical surface", False, "missing", "leadfield.py:158-188"),
        ("dipole orientation", "fixed surface-normal orientation after conversion", False, "missing", "leadfield.py:526-528"),
        ("BEM surfaces and conductivities", "three-layer EEG conductor model", False, "missing_instance", "leadfield.py:118-156"),
        ("forward solution", "sensor-by-dipole gain matrix", False, "missing", "leadfield.py:244-263"),
        ("atlas NIfTI/XML", "assign dipoles to AAL2 cortical regions", False, "missing_instance", "leadfield.py:488-553"),
        ("node-to-atlas mapping", "select/aggregate source columns for model nodes", False, "missing", "not defined in sleep_loop"),
        ("source time series in dipole-moment units", "multiply lead field by source amplitudes", False, "missing", "not implemented by leadfield.py"),
        ("Fpz-Cz bipolar operator", "subtract projected sensor potentials", False, "missing_project_contract", "not implemented by leadfield.py"),
    ]
    return pd.DataFrame(rows, columns=columns)


def single_source_rank_audit() -> dict[str, Any]:
    """Return the algebraic consequence of projecting one scalar source."""

    return {
        "source_count": 1,
        "source_model": "q(t) is one scalar cortical source",
        "sensor_model": "V_Fpz(t)=L_Fpz*q(t); V_Cz(t)=L_Cz*q(t)",
        "bipolar_model": "V_Fpz-Cz(t)=(L_Fpz-L_Cz)*q(t)",
        "temporal_rank": 1,
        "can_be_nonzero": True,
        "nonzero_condition": "L_Fpz != L_Cz and q(t) is nonzero",
        "adds_new_temporal_structure": False,
        "scientific_consequence": (
            "With one declared scalar source, Fpz-Cz is a fixed gain/sign copy. "
            "The gain can be physically computed only after source position, "
            "orientation, dipole moment and head model are fixed. The current "
            "two-node model supplies none of those spatial declarations."
        ),
    }


def route_decision_matrix() -> pd.DataFrame:
    """Compare Route 2 and Route 3 without authorizing either experiment."""

    columns = [
        "criterion",
        "route_2_measurement_model",
        "route_3_synthetic_observable",
        "current_assessment",
    ]
    rows = [
        ("scientific_claim", "Inference about theta conditional on a validated Fpz-Cz measurement model", "Parameter recovery for a declared model-internal cortical observable", "Route 2 is stronger but unsupported now"),
        ("required_implementation", "Source reconstruction, geometry, BEM/lead field, bipolar reference, gain/noise and EEG validation", "Freeze model observable, simulator extractor, prior, bank and synthetic diagnostics", "Route 3 is implementable sooner"),
        ("required_external_data", "Anatomy/template plus independent measurement-model calibration and validation", "None for synthetic recovery; real EEG only optional external shape check", "Route 2 lacks essential external evidence"),
        ("calibration_leakage_risk", "High if source/gain/noise are tuned on SC4001", "Low if real EEG is excluded from tuning", "Route 3 has cleaner immediate separation"),
        ("identifiability", "theta is confounded with source weights, geometry, gain and noise", "Identifiability of theta can be measured within the declared simulator observable", "Route 2 currently underidentified"),
        ("expected_time_cost", "High: source derivation, forward implementation and independent validation", "Moderate: honest simulator-bank and recovery study after researcher approval", "Conference horizon favors Route 3"),
        ("conference_suitability", "Only as a feasibility/limitations discussion until independently validated", "Suitable as a scoped synthetic-method result", "Recommend Route 3 for near-term claims"),
        ("can_claim", "Current work can claim a documented feasibility audit and missing contract", "Can later claim synthetic recovery/calibration for the chosen observable", "Neither route currently has posterior results"),
        ("cannot_claim", "No current real-EEG posterior or EEG measurement validation", "No inference about real Fpz-Cz physiology or patient parameters", "Keep modality boundary explicit"),
        ("minimum_go_no_go", "All contract stages pass and held-out measurement validation succeeds", "Frozen observable/schema, fixed finite extractor, prior and synthetic validation protocol", "Both remain researcher-gated"),
        ("current_decision", "NO-GO for real-EEG SBI; retain as longer research program", "Recommended near-term methodological route, not authorized to start", "Researcher must choose scope"),
    ]
    return pd.DataFrame(rows, columns=columns)


def citation_table() -> pd.DataFrame:
    """Return primary/official references used by the audit."""

    rows = [
        (
            "Cakan and Obermayer 2020",
            "Biophysically grounded mean-field models of neural populations under electrical stimulation",
            "ALN model meaning and transfer-function context",
            "https://doi.org/10.1371/journal.pcbi.1007822",
            "peer_reviewed_model_paper",
        ),
        (
            "MNE make_forward_solution",
            "mne.make_forward_solution official API",
            "Required sensor Info, transform, source space and BEM",
            "https://mne.tools/stable/generated/mne.make_forward_solution.html",
            "official_documentation",
        ),
        (
            "MNE forward tutorial",
            "Head model and forward computation",
            "Coregistration, source space, three-layer EEG BEM and gain matrix",
            "https://mne.tools/stable/auto_tutorials/forward/30_forward.html",
            "official_documentation",
        ),
        (
            "Hallez et al. 2007",
            "Review on solving the forward problem in EEG source analysis",
            "Dipole position/orientation, conductivity and scalp-potential forward problem",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC2234413/",
            "peer_reviewed_review",
        ),
        (
            "Mosher et al. 1999",
            "EEG and MEG: forward solutions for inverse methods",
            "Lead-field factorization and differential EEG measurements",
            "https://pubmed.ncbi.nlm.nih.gov/10097460/",
            "peer_reviewed_review",
        ),
    ]
    return pd.DataFrame(rows, columns=["citation", "title", "used_for", "url", "source_type"])


def reader_guide_registry() -> pd.DataFrame:
    """Return a concise scientific reading registry for notebooks 02 through 09."""

    rows = [
        ("02", "Observation schema comparison", "What are the executable Baseline and candidate Augmented schemas?", "01 observation artifacts", "Actual Baseline-14D, Augmented-23D, constructive prefix identity", "Sections 3, 6, 11", "Schema names are not evidence of simulator parity"),
        ("03", "Observation schema freeze", "Which real-EEG candidate schemas are finite and supportable?", "02 artifacts", "No schema approved because simulation extraction was not established", "Freeze decision; Go/No-Go; role figure", "A frozen real vector alone is insufficient"),
        ("04", "Real-simulation extractor parity", "Can existing simulation assets produce same-semantic vectors?", "03 schemas and legacy banks", "NO-GO: legacy banks are 5D/7D rate summaries", "Asset audit; feature failures; hard gate", "Code-level availability is not semantic parity"),
        ("05", "Pilot SNPE ablation", "Did any schema pass the training gate?", "04 gate", "Training not started; zero simulations and zero seeds", "Gate before training; manifest; blocker table", "Contains no posterior"),
        ("06", "Ablation diagnostics", "Which recovery, coverage or PPC results exist?", "05 artifacts", "All diagnostic entries are insufficient evidence", "Ablation summary; availability figure; conclusion", "Contains no recovery/coverage/PPC result"),
        ("07", "Simulator observable adapter", "Can existing V7/V8/V8a rate signals enter shared low-level code?", "Model parameter sets and EEG functions", "Engineering adapter works; semantic parity fails", "Adapter contract; blocker matrix; NO-GO gate", "Cortical firing rate is not simulated EEG"),
        ("08", "EEG observation mapping audit", "Do any recorded model states already constitute an EEG/LFP proxy?", "07 artifacts, ALN/thalamic states, real EEG", "Only standardized shape screening; no measurement model", "State catalog; PSD/ACF screening; Route decision", "High shape correlation is not EEG validation"),
        ("09", "Forward-model feasibility and route decision", "Can the single-node model support a defensible Fpz-Cz forward path?", "02-08 evidence, neurolib/MNE source", "Single source is rank-1; Route 2 needs external spatial/source assumptions", "Source matrix; contract gate; Route 2 vs 3", "Does not implement a forward model or authorize training"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "notebook",
            "title",
            "scientific_question",
            "inputs",
            "current_conclusion",
            "priority_sections_or_figures",
            "cannot_claim",
        ],
    )


def validate_measurement_declaration(declaration: Mapping[str, Any]) -> ContractGate:
    """Validate a proposed measurement declaration without filling defaults."""

    missing = tuple(field for field in REQUIRED_DECLARATION_FIELDS if field not in declaration)
    invalid: list[str] = []
    leakage: list[str] = []
    warnings: list[str] = []

    if declaration.get("calibration_independent_of_sc4001") is not True:
        leakage.append("calibration_independent_of_sc4001")
    if declaration.get("validation_independent_of_sc4001") is not True:
        leakage.append("validation_independent_of_sc4001")

    source_coords = declaration.get("source_coordinates_mri_mm")
    if source_coords is not None and (not isinstance(source_coords, Sequence) or len(source_coords) != 3):
        invalid.append("source_coordinates_mri_mm")

    orientation = declaration.get("source_orientation_xyz")
    if orientation is not None:
        try:
            orientation_array = np.asarray(orientation, dtype=float)
            if orientation_array.shape != (3,) or not np.all(np.isfinite(orientation_array)):
                invalid.append("source_orientation_xyz")
            elif not np.isclose(np.linalg.norm(orientation_array), 1.0, atol=1e-6):
                invalid.append("source_orientation_xyz")
        except (TypeError, ValueError):
            invalid.append("source_orientation_xyz")

    if declaration.get("sensor_names") not in (["Fpz", "Cz"], ("Fpz", "Cz")):
        if "sensor_names" in declaration:
            invalid.append("sensor_names")
    if declaration.get("reference_operator") != [1.0, -1.0]:
        if "reference_operator" in declaration:
            invalid.append("reference_operator")
    if declaration.get("output_unit") not in (None, "V", "uV"):
        invalid.append("output_unit")
    if declaration.get("resample_hz") not in (None, 100.0, 100):
        invalid.append("resample_hz")
    if declaration.get("epoch_duration_s") not in (None, 30.0, 30):
        invalid.append("epoch_duration_s")

    if declaration.get("source_count") == 1:
        warnings.append(
            "One scalar source yields a rank-1 sensor signal: Fpz-Cz is a fixed gain/sign copy."
        )

    status = "GO" if not missing and not invalid and not leakage else "NO-GO"
    return ContractGate(
        status=status,
        missing_fields=missing,
        invalid_fields=tuple(dict.fromkeys(invalid)),
        leakage_fields=tuple(dict.fromkeys(leakage)),
        warnings=tuple(warnings),
    )


def apply_measurement_model(
    source_signal: np.ndarray,
    declaration: Mapping[str, Any],
) -> np.ndarray:
    """Refuse projection unless a complete, validated implementation exists."""

    gate = validate_measurement_declaration(declaration)
    if not gate.passed:
        raise MeasurementContractError(
            "EEG measurement contract is NO-GO: "
            f"missing={list(gate.missing_fields)}, "
            f"invalid={list(gate.invalid_fields)}, "
            f"leakage={list(gate.leakage_fields)}"
        )
    raise NotImplementedError(
        "A declaration alone is insufficient. No validated source-to-Fpz-Cz "
        "forward implementation exists in this project."
    )


def protected_notebook_paths(project_root: Path) -> list[Path]:
    """Return notebooks 01 through 08 that this audit must not alter."""

    notebook_dir = project_root / "S4_sbi" / "notebooks"
    return sorted(path for path in notebook_dir.glob("*.ipynb") if path.name[:2].isdigit() and int(path.name[:2]) <= 8)
