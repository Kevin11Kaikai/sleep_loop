"""Auditable adapter from existing V7/V8 cortical rates to observation functions.

This module deliberately does *not* call the cortical firing rate ``simulated
EEG``.  Its only signal is the existing ALN cortical excitatory population
firing rate, reported in Hz after the project scripts' kHz-to-Hz conversion.
It is a cortical observable proxy.  Running the real-observation functions on
that proxy establishes code-level behaviour and exposes unit/semantic gaps; it
does not validate a scalp-EEG forward model.

No simulator constants, real-EEG thresholds, filters, or detector definitions
are changed here.  In particular, uV-dependent slow-oscillation detection is
blocked rather than numerically applied to Hz-valued firing rates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import observation as real_observation
from .schemas import ObservationConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CONFIG_PATH = "S4_sbi/configs/observation_sc4001.yaml"
RATE_PROXY_SIGNAL = "cortical excitatory population firing rate proxy"
RATE_PROXY_UNIT = "Hz"
WARM_UP_S = 5.0

PARAMETER_KEYS = (
    "mue",
    "mui",
    "b",
    "tauA",
    "g_LK",
    "g_h",
    "c_th2ctx",
    "c_ctx2th",
)

SPECTRAL_SO_PROXY_FEATURES = (
    "fooof_aperiodic_exponent",
    "so_peak_frequency_hz",
    "relative_so_power",
    "so_q",
)
SO_MORPHOLOGY_FEATURES = (
    "so_event_rate_per_min",
    "ibi_cv",
    "waveform_peak_to_peak_z",
)
SPINDLE_FEATURES = (
    "spindle_density_per_min",
    "spindle_mean_duration_s",
)
PAC_FEATURES = (
    "pac_up_down_ratio",
    "pac_mi",
    "pac_preferred_phase_rad",
    "pac_preferred_phase_sin",
    "pac_preferred_phase_cos",
)
BASELINE14_FEATURES = (
    "fooof_aperiodic_exponent",
    "so_peak_frequency_hz",
    "relative_so_power",
    "so_q",
    "so_event_rate_per_min",
    "ibi_cv",
    "pac_up_down_ratio",
    "spindle_density_per_min",
    "spindle_mean_duration_s",
    "pac_mi",
    "pac_preferred_phase_rad",
    "pac_preferred_phase_sin",
    "pac_preferred_phase_cos",
    "waveform_peak_to_peak_z",
)

SCHEMA_FEATURES = {
    "minimal_spectral_so_proxy4": SPECTRAL_SO_PROXY_FEATURES,
    "minimal_plus_so_morphology7": SPECTRAL_SO_PROXY_FEATURES + SO_MORPHOLOGY_FEATURES,
    "minimal_plus_spindle6": SPECTRAL_SO_PROXY_FEATURES + SPINDLE_FEATURES,
    "minimal_plus_pac9": SPECTRAL_SO_PROXY_FEATURES + PAC_FEATURES,
    "baseline14": BASELINE14_FEATURES,
}


@dataclass(frozen=True)
class ParameterSet:
    """A complete legacy model parameter vector with its source provenance."""

    identifier: str
    model_version: str
    source: str
    parameters: dict[str, float]


@dataclass
class AdapterResult:
    """Publication-safe output of one cortical-rate proxy simulation."""

    identifier: str
    model_version: str
    source: str
    success: bool
    runtime_s: float
    simulation_duration_s: float
    warm_up_removed_s: float
    active_duration_s: float
    discarded_tail_s: float
    fs_hz: float
    proxy_signal: str
    proxy_unit: str
    n_complete_30s_epochs: int
    feature_values: dict[str, float]
    feature_validity: dict[str, bool]
    feature_failure_reasons: dict[str, str | None]
    feature_semantic_status: dict[str, str]
    feature_support: dict[str, int]
    schema_status: dict[str, dict[str, Any]]
    failure_reason: str | None
    warnings: list[str]
    parameters: dict[str, float]

    def vector(self, schema_name: str) -> np.ndarray:
        """Return a fixed-order vector; undefined values remain ``NaN``."""
        names = SCHEMA_FEATURES[schema_name]
        return np.asarray([self.feature_values[name] for name in names], dtype=float)

    def validity_mask(self, schema_name: str) -> np.ndarray:
        names = SCHEMA_FEATURES[schema_name]
        return np.asarray([self.feature_validity[name] for name in names], dtype=bool)


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _to_float_parameters(payload: Mapping[str, Any]) -> dict[str, float]:
    """Normalize the project's ``g_LK`` spelling without changing values."""
    result: dict[str, float] = {}
    for key in PARAMETER_KEYS:
        source_key = "g_lk" if key == "g_LK" and "g_lk" in payload else key
        if source_key not in payload:
            raise KeyError(f"missing required model parameter: {key}")
        value = float(payload[source_key])
        if not np.isfinite(value):
            raise ValueError(f"non-finite parameter {key}")
        result[key] = value
    return result


@lru_cache(maxsize=2)
def _load_model_module(model_version: str):
    """Load the existing fitting script without executing its CLI entrypoint."""
    normalized = model_version.lower()
    if normalized == "v7":
        filename = "s4_personalize_fig7_v7.py"
        module_name = "sleep_sbi_adapter_v7"
    elif normalized in {"v8", "v8a", "v8a_t13"}:
        # The repository's V8 script contains the V8a/T13 cortical detector.
        filename = "s4_personalize_fig7_v8.py"
        module_name = "sleep_sbi_adapter_v8"
    else:
        raise ValueError(f"unsupported model version {model_version!r}")
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    # V7/V8 load their preprocessing helper via a project-root relative path.
    # This mirrors the existing simulator_wrapper setup and is required when a
    # notebook executor starts with S4_sbi/notebooks as its working directory.
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _run_existing_model(parameter_set: ParameterSet) -> tuple[np.ndarray, np.ndarray, float, float, float, list[str]]:
    """Run the current V7/V8 builder and return post-warm-up rate proxies.

    The calculation intentionally mirrors the existing V7/V8 fitting scripts:
    recorded output is multiplied by 1000 from kHz to Hz and the first five
    seconds are discarded.  No simulated signal is persisted by this function.
    """
    module = _load_model_module(parameter_set.model_version)
    params = parameter_set.parameters
    model = module.build_model(
        params["mue"],
        params["mui"],
        params["b"],
        params["tauA"],
        params["g_LK"],
        params["g_h"],
        params["c_th2ctx"],
        params["c_ctx2th"],
    )
    warnings: list[str] = []
    start = time.perf_counter()
    backend = "numba"
    try:
        module.seed_numba(42)
        model.run()
    except Exception as first_error:
        # This is the pre-existing V7/V8 fitting fallback, retained explicitly.
        backend = "jitcdde"
        warnings.append(f"numba_failed_then_jitcdde: {type(first_error).__name__}")
        model.params["backend"] = backend
        module.seed_numba(42)
        model.run()
    runtime_s = time.perf_counter() - start
    r_exc = np.asarray(model[f"r_mean_{module.EXC}"], dtype=float)
    if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
        cortical_rate_hz = r_exc[0] * 1000.0
        thalamic_rate_hz = r_exc[1] * 1000.0
    elif r_exc.ndim == 1:
        cortical_rate_hz = r_exc * 1000.0
        thalamic_rate_hz = np.zeros_like(cortical_rate_hz)
        warnings.append("thalamic_output_missing")
    else:
        raise RuntimeError(f"unexpected r_mean_EXC shape {r_exc.shape}")
    fs_hz = float(module.FS_SIM)
    n_drop = int(round(WARM_UP_S * fs_hz))
    cortical_rate_hz = cortical_rate_hz[n_drop:]
    thalamic_rate_hz = thalamic_rate_hz[n_drop:]
    if not len(cortical_rate_hz) or not np.isfinite(cortical_rate_hz).all():
        raise RuntimeError("post-warm-up cortical rate is absent or non-finite")
    if float(np.max(cortical_rate_hz)) < 0.1:
        raise RuntimeError("post-warm-up cortical rate is flat")
    warnings.append(f"backend={backend}")
    return cortical_rate_hz, thalamic_rate_hz, fs_hz, runtime_s, float(module.SIM_DUR_MS) / 1000.0, warnings


def _empty_feature_maps() -> tuple[dict[str, float], dict[str, bool], dict[str, str | None], dict[str, str], dict[str, int]]:
    names = BASELINE14_FEATURES
    values = {name: float("nan") for name in names}
    validity = {name: False for name in names}
    reasons = {name: "not_computed" for name in names}
    semantics = {name: "not_computed" for name in names}
    support = {name: 0 for name in names}
    return values, validity, reasons, semantics, support


def _mark(
    values: dict[str, float],
    validity: dict[str, bool],
    reasons: dict[str, str | None],
    semantics: dict[str, str],
    support: dict[str, int],
    name: str,
    value: float,
    valid: bool,
    reason: str | None,
    semantic_status: str,
    support_count: int,
) -> None:
    values[name] = float(value)
    validity[name] = bool(valid)
    reasons[name] = reason
    semantics[name] = semantic_status
    support[name] = int(support_count)


def _schema_status(
    values: Mapping[str, float],
    validity: Mapping[str, bool],
    semantics: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for schema_name, feature_names in SCHEMA_FEATURES.items():
        vector = np.asarray([values[name] for name in feature_names], dtype=float)
        valid = np.asarray([validity[name] for name in feature_names], dtype=bool)
        semantic_match = all(semantics[name] == "same_algorithm_proxy_only" for name in feature_names)
        result[schema_name] = {
            "feature_names": list(feature_names),
            "dimension": len(feature_names),
            "fixed_length": True,
            "record_vector_finite": bool(np.isfinite(vector).all()),
            "validity_mask_all_true": bool(valid.all()),
            "algorithm_level_available": bool(valid.all() and np.isfinite(vector).all()),
            "semantic_parity_pass": False,
            "semantic_status": (
                "NO-GO: cortical firing-rate proxy is not a validated Fpz-Cz scalp-EEG observable"
                if semantic_match
                else "NO-GO: includes unit-dependent or undefined proxy features"
            ),
            "scaling_status": "not frozen; no real-to-proxy observable mapping",
            "aggregation_status": "single complete 30-second post-warm-up epoch; 25-second tail excluded",
        }
    return result


def run_adapter(parameter_set: ParameterSet, config: ObservationConfig | None = None) -> AdapterResult:
    """Run one candidate through an auditable cortical-rate proxy adapter.

    It uses a single full 30-second post-warm-up epoch because legacy 60-second
    simulations leave 55 seconds after their established five-second burn-in.
    The remaining 25 seconds are excluded rather than concatenated across a
    fabricated boundary.  This preserves existing simulator duration and makes
    the duration mismatch explicit.
    """
    config = config or real_observation.load_observation_config(DEFAULT_CONFIG_PATH)
    values, validity, reasons, semantics, support = _empty_feature_maps()
    try:
        cortical, _thalamic, fs_hz, runtime_s, total_duration_s, warnings = _run_existing_model(parameter_set)
    except Exception as error:
        schema_status = _schema_status(values, validity, semantics)
        return AdapterResult(
            identifier=parameter_set.identifier,
            model_version=parameter_set.model_version,
            source=parameter_set.source,
            success=False,
            runtime_s=float("nan"),
            simulation_duration_s=float("nan"),
            warm_up_removed_s=WARM_UP_S,
            active_duration_s=float("nan"),
            discarded_tail_s=float("nan"),
            fs_hz=float("nan"),
            proxy_signal=RATE_PROXY_SIGNAL,
            proxy_unit=RATE_PROXY_UNIT,
            n_complete_30s_epochs=0,
            feature_values=values,
            feature_validity=validity,
            feature_failure_reasons=reasons,
            feature_semantic_status=semantics,
            feature_support=support,
            schema_status=schema_status,
            failure_reason=f"simulation_failed: {type(error).__name__}: {error}",
            warnings=[],
            parameters=parameter_set.parameters,
        )

    epoch_samples = int(round(config.epoch_duration_s * fs_hz))
    n_complete = len(cortical) // epoch_samples
    if n_complete < 1:
        raise RuntimeError("legacy post-warm-up run does not contain one full 30-second epoch")
    # Preserve original 60-second simulator duration: use one complete epoch only.
    segments = cortical[:epoch_samples].reshape(1, epoch_samples)
    discarded_tail_s = (len(cortical) - epoch_samples) / fs_hz
    retained_epoch_indices = np.asarray([0], dtype=int)
    warnings.extend(
        [
            "Cortical firing rate is a model-state observable proxy, not simulated EEG.",
            "No forward model, referencing model, or uV calibration to EEG Fpz-Cz is implemented.",
            "Real-EEG peak-to-peak QC is not applied because its threshold is defined in uV.",
            "SO event and waveform features are blocked: their detector contains a 75 uV half-wave threshold.",
            "Record aggregation uses one complete 30-second model epoch; the post-warm-up 25-second tail is excluded.",
        ]
    )

    # Exact real-observation spectral functions; PSD units are rate^2/Hz here.
    psd = real_observation._compute_psd(segments, fs_hz, config)
    spectral = real_observation._spectral_statistics(psd.frequencies_hz, psd.aggregate_hann_uv2_hz, config)
    fooof = real_observation._compute_fooof(psd, config)
    for name, value in (
        ("fooof_aperiodic_exponent", fooof["aperiodic_exponent"]),
        ("so_peak_frequency_hz", spectral["so_peak_frequency_hz"]),
        ("relative_so_power", spectral["relative_so_power"]),
        ("so_q", spectral["so_q"]),
    ):
        _mark(values, validity, reasons, semantics, support, name, value, np.isfinite(value), None if np.isfinite(value) else "non_finite_spectral_result", "same_algorithm_proxy_only", 1)

    # Do not apply the real 75 uV detector threshold to Hz-valued state data.
    for name in SO_MORPHOLOGY_FEATURES:
        _mark(values, validity, reasons, semantics, support, name, float("nan"), False, "blocked_unit_dependent_real_EEG_detector_threshold", "blocked_unit_mismatch", 0)

    # The real observable-channel spindle detector is amplitude-scale equivariant
    # (mean + SD RMS threshold).  Execute it for code-level support only; its
    # output remains a cortical-rate proxy diagnostic rather than scalp EEG.
    spindle = real_observation._compute_spindle_diagnostics(segments, retained_epoch_indices, fs_hz, config)
    spindle_epoch = spindle["per_epoch"][0]
    spindle_valid = bool(spindle_epoch["valid"])
    spindle_count = int(spindle_epoch["event_count"])
    for name, value in (
        ("spindle_density_per_min", spindle_epoch["density_per_min"]),
        ("spindle_mean_duration_s", spindle_epoch["mean_duration_s"]),
    ):
        event_dependent = name == "spindle_mean_duration_s"
        is_valid = spindle_valid and np.isfinite(value)
        reason = None if is_valid else ("no_detected_spindle_events" if event_dependent and spindle_valid else str(spindle_epoch["invalid_reason"]))
        _mark(values, validity, reasons, semantics, support, name, value, is_valid, reason, "same_algorithm_proxy_only" if is_valid else "proxy_event_support_insufficient", spindle_count)

    # PAC is dimensionless and scale-invariant to positive amplitude rescaling;
    # run the exact real function, but preserve the cortical-proxy semantic caveat.
    pac = real_observation._compute_pac_diagnostics(segments, retained_epoch_indices, fs_hz, config)
    for name, value in (
        ("pac_up_down_ratio", pac["pac_up_down_ratio"]),
        ("pac_mi", pac["mi"]),
        ("pac_preferred_phase_rad", pac["preferred_phase_rad"]),
        ("pac_preferred_phase_sin", pac["preferred_phase_sin"]),
        ("pac_preferred_phase_cos", pac["preferred_phase_cos"]),
    ):
        is_valid = bool(pac["valid"]) and np.isfinite(value)
        _mark(values, validity, reasons, semantics, support, name, value, is_valid, None if is_valid else "pac_detector_invalid", "same_algorithm_proxy_only" if is_valid else "proxy_event_support_insufficient", int(pac["valid_epoch_count"]))

    schema_status = _schema_status(values, validity, semantics)
    return AdapterResult(
        identifier=parameter_set.identifier,
        model_version=parameter_set.model_version,
        source=parameter_set.source,
        success=True,
        runtime_s=float(runtime_s),
        simulation_duration_s=float(total_duration_s),
        warm_up_removed_s=WARM_UP_S,
        active_duration_s=float(len(cortical) / fs_hz),
        discarded_tail_s=float(discarded_tail_s),
        fs_hz=float(fs_hz),
        proxy_signal=RATE_PROXY_SIGNAL,
        proxy_unit=RATE_PROXY_UNIT,
        n_complete_30s_epochs=1,
        feature_values=values,
        feature_validity=validity,
        feature_failure_reasons=reasons,
        feature_semantic_status=semantics,
        feature_support=support,
        schema_status=schema_status,
        failure_reason=None,
        warnings=warnings,
        parameters=parameter_set.parameters,
    )


def load_representative_parameter_sets() -> list[ParameterSet]:
    """Load V7, V8, V8a, and three archived V8a near-feasible candidates."""
    definitions = [
        ("v7_fitted", "v7", PROJECT_ROOT / "data" / "patient_params_fig7_v7_SC4001.json"),
        ("v8_fitted", "v8", PROJECT_ROOT / "data" / "patient_params_fig7_v8_SC4001.json"),
        ("v8a_local_best", "v8a_t13", PROJECT_ROOT / "outputs" / "v8a_ultra_narrow_t6_t13_search" / "best_so_far.json"),
    ]
    parameter_sets: list[ParameterSet] = []
    for identifier, version, path in definitions:
        payload = json.loads(path.read_text(encoding="utf-8"))
        parameter_sets.append(ParameterSet(identifier, version, _relative(path), _to_float_parameters(payload)))

    topk_path = PROJECT_ROOT / "outputs" / "v8a_coupling_sweep_long" / "selected_seed_candidates.csv"
    topk = pd.read_csv(topk_path)
    topk = topk.sort_values(["seed_n_passed", "seed_score"], ascending=[False, False], kind="stable").head(3)
    for _, row in topk.iterrows():
        identifier = f"v8a_near_feasible_{row['seed_id']}"
        parameter_sets.append(ParameterSet(identifier, "v8a_t13", _relative(topk_path), _to_float_parameters(row.to_dict())))
    if len(parameter_sets) != 6:
        raise RuntimeError("expected exactly six representative parameter sets")
    return parameter_sets


def result_rows(result: AdapterResult) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert a result into feature, schema, and run-level publication-safe tables."""
    feature_rows = []
    for name in BASELINE14_FEATURES:
        feature_rows.append(
            {
                "candidate_id": result.identifier,
                "model_version": result.model_version,
                "feature": name,
                "value": result.feature_values[name],
                "valid": result.feature_validity[name],
                "support": result.feature_support[name],
                "failure_reason": result.feature_failure_reasons[name],
                "semantic_status": result.feature_semantic_status[name],
                "proxy_signal": result.proxy_signal,
                "proxy_unit": result.proxy_unit,
            }
        )
    schema_rows = []
    for name, status in result.schema_status.items():
        schema_rows.append({"candidate_id": result.identifier, "model_version": result.model_version, "schema": name, **status})
    run_row = {
        key: value
        for key, value in asdict(result).items()
        if key not in {"feature_values", "feature_validity", "feature_failure_reasons", "feature_semantic_status", "feature_support", "schema_status", "parameters", "warnings"}
    }
    run_row["parameters_json"] = json.dumps(result.parameters, sort_keys=True)
    run_row["warnings_json"] = json.dumps(result.warnings)
    return pd.DataFrame(feature_rows), pd.DataFrame(schema_rows), pd.DataFrame([run_row])


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a small checkpoint table without retaining raw simulated signals."""
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def run_validation_batch(
    output_dir: str | Path,
    parameter_sets: list[ParameterSet] | None = None,
    completed_results: list[AdapterResult] | None = None,
    config: ObservationConfig | None = None,
) -> list[AdapterResult]:
    """Run an ordered representative batch and checkpoint after every candidate.

    The exported CSVs contain summaries, validity, support, and failures only.
    They intentionally omit complete firing-rate time series.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or real_observation.load_observation_config(DEFAULT_CONFIG_PATH)
    results = list(completed_results or [])
    already_done = {result.identifier for result in results}
    parameter_sets = parameter_sets or load_representative_parameter_sets()

    def checkpoint() -> None:
        feature_frames, schema_frames, run_frames = [], [], []
        for result in results:
            features, schemas, runs = result_rows(result)
            feature_frames.append(features)
            schema_frames.append(schemas)
            run_frames.append(runs)
        _atomic_csv(pd.concat(run_frames, ignore_index=True), output_dir / "adapter_run_summary.csv")
        _atomic_csv(pd.concat(feature_frames, ignore_index=True), output_dir / "adapter_feature_matrix.csv")
        _atomic_csv(pd.concat(schema_frames, ignore_index=True), output_dir / "adapter_schema_status.csv")
        _atomic_json(
            {
                "adapter_signal": RATE_PROXY_SIGNAL,
                "adapter_unit": RATE_PROXY_UNIT,
                "config": _relative(PROJECT_ROOT / DEFAULT_CONFIG_PATH),
                "model_duration_s": 60.0,
                "warm_up_removed_s": WARM_UP_S,
                "segment_policy": "first complete 30-second post-warm-up epoch; remaining 25 seconds excluded",
                "raw_signals_saved": False,
                "completed_candidate_ids": [result.identifier for result in results],
                "successful_candidate_count": int(sum(result.success for result in results)),
                "semantic_parity_pass_count": 0,
                "hard_gate": "NO-GO until a validated scalp-EEG observable mapping exists",
            },
            output_dir / "adapter_manifest.json",
        )

    if results:
        checkpoint()
    for parameter_set in parameter_sets:
        if parameter_set.identifier in already_done:
            continue
        result = run_adapter(parameter_set, config=config)
        results.append(result)
        checkpoint()
    return results
