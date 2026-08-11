"""Screen neurolib state variables as possible EEG measurement candidates.

This module is deliberately an audit helper, not a forward model.  It records
additional model states without changing the simulator dynamics, compares
shape-level statistics after explicit normalization, and keeps measurement
semantics separate from numerical extractor success.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr

from .observation import (
    _compute_pac_diagnostics,
    _compute_psd,
    _compute_so_diagnostics,
    _compute_spindle_diagnostics,
    _spectral_statistics,
    build_observation_bundle,
    load_observation_config,
)
from .simulator_observable_adapter import (
    ParameterSet,
    _load_model_module,
    load_representative_parameter_sets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "S4_sbi" / "configs" / "observation_sc4001.yaml"

MODEL_DURATION_S = 60.0
WARM_UP_S = 5.0
SCREENING_EPOCH_S = 30.0

ALN_OUTPUTS = [
    "r_mean_EXC",
    "r_mean_INH",
    "I_A_EXC",
    "I_mu_EXC",
    "I_mu_INH",
    "I_syn_mu_exc_EXC",
    "I_syn_mu_inh_EXC",
    "I_syn_sigma_exc_EXC",
    "I_syn_sigma_inh_EXC",
    "I_syn_mu_exc_INH",
    "I_syn_mu_inh_INH",
    "I_syn_sigma_exc_INH",
    "I_syn_sigma_inh_INH",
]
THALAMUS_OUTPUTS = ["r_mean_EXC", "r_mean_INH", "V_EXC", "V_INH"]
NETWORK_OUTPUTS = list(dict.fromkeys(ALN_OUTPUTS + THALAMUS_OUTPUTS))


@dataclass
class SimulationStateAudit:
    """One simulation and its post-warm-up state recordings."""

    parameter_set: ParameterSet
    fs_hz: float
    runtime_s: float
    duration_s: float
    warm_up_s: float
    signals: dict[str, np.ndarray]
    signal_metadata: dict[str, dict[str, str]]
    warnings: list[str]


@dataclass
class RealReference:
    """Publication-safe real-EEG reference plus in-memory retained epochs."""

    segments_uv: np.ndarray
    fs_hz: float
    retained_epoch_indices: np.ndarray
    rejected_epoch_indices: np.ndarray
    n3_epoch_indices: np.ndarray
    representative_epoch_index: int
    frequencies_hz: np.ndarray
    normalized_psd: np.ndarray
    acf_lags_s: np.ndarray
    mean_acf: np.ndarray
    metadata: dict[str, Any]


def audit_parameter_sets() -> list[ParameterSet]:
    """Return the V7, V8, and V8a parameter sets already used by notebook 07."""

    parameter_sets = load_representative_parameter_sets()
    requested = {"v7_fitted", "v8_fitted", "v8a_local_best"}
    selected = [item for item in parameter_sets if item.identifier in requested]
    if {item.identifier for item in selected} != requested:
        raise RuntimeError("V7/V8/V8a audit parameter sets are incomplete")
    return selected


def _safe_zscore(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    scale = float(np.std(signal))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("signal is constant or non-finite")
    return (signal - float(np.mean(signal))) / scale


def _mean_acf(
    segments: np.ndarray,
    fs_hz: float,
    *,
    max_lag_s: float = 5.0,
    lag_step_s: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean normalized autocorrelation without crossing epoch edges."""

    segments = np.atleast_2d(np.asarray(segments, dtype=float))
    lag_samples = np.unique(
        np.rint(np.arange(0.0, max_lag_s + lag_step_s / 2, lag_step_s) * fs_hz)
        .astype(int)
        .clip(0)
    )
    rows = []
    for segment in segments:
        z = _safe_zscore(segment)
        values = []
        for lag in lag_samples:
            if lag == 0:
                values.append(1.0)
            elif lag >= len(z):
                values.append(np.nan)
            else:
                values.append(float(np.mean(z[:-lag] * z[lag:])))
        rows.append(values)
    result = np.nanmean(np.asarray(rows, dtype=float), axis=0)
    return lag_samples / fs_hz, result


def _normalized_welch(
    segments: np.ndarray,
    fs_hz: float,
    *,
    band_hz: tuple[float, float] = (0.5, 20.0),
    segment_s: float = 4.0,
    overlap_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return arithmetic-mean Hann PSD normalized to unit band area."""

    segments = np.atleast_2d(np.asarray(segments, dtype=float))
    rows = []
    frequencies = None
    for segment in segments:
        frequencies, density = welch(
            segment,
            fs=fs_hz,
            window="hann",
            nperseg=int(round(segment_s * fs_hz)),
            noverlap=int(round(overlap_s * fs_hz)),
            detrend="constant",
            scaling="density",
        )
        rows.append(density)
    aggregate = np.mean(np.asarray(rows, dtype=float), axis=0)
    mask = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    area = float(np.trapezoid(aggregate[mask], frequencies[mask]))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("non-positive PSD area")
    return np.asarray(frequencies), aggregate / area


def load_real_reference() -> RealReference:
    """Load the existing SC4001 ObservationBundle without changing its rules."""

    config = load_observation_config(DEFAULT_CONFIG)
    bundle = build_observation_bundle(config)
    segments = np.asarray(bundle.segments, dtype=float)
    if segments.ndim != 2 or segments.shape[1] != int(config.epoch_duration_s * bundle.fs_hz):
        raise RuntimeError("real observation segments do not preserve 30-second epochs")

    peak_to_peak = np.ptp(segments, axis=1)
    representative_row = int(np.argsort(peak_to_peak)[len(peak_to_peak) // 2])
    representative_epoch = int(bundle.retained_epoch_indices[representative_row])
    frequencies, normalized_psd = _normalized_welch(
        segments,
        bundle.fs_hz,
        band_hz=config.psd_band_hz,
        segment_s=config.welch_segment_s,
        overlap_s=config.welch_overlap_s,
    )
    acf_lags, mean_acf = _mean_acf(segments, bundle.fs_hz)

    source = dict(bundle.provenance)
    safe_source = {
        "subject_id": bundle.subject_id,
        "channel": bundle.channel,
        "unit": source.get("analysis_unit", "uV"),
        "reference": source.get(
            "reference", "as-recorded Fpz-Cz derivation"
        ),
        "recording_duration_s": source.get("recording_duration_s"),
        "total_epoch_count": source.get("recording_complete_epochs"),
        "n3_epoch_count": int(len(bundle.n3_epoch_indices)),
        "retained_epoch_count": int(len(bundle.retained_epoch_indices)),
        "rejected_epoch_count": int(len(bundle.rejected_epoch_indices)),
        "epoch_duration_s": float(bundle.epoch_duration_s),
        "fs_hz": float(bundle.fs_hz),
    }
    return RealReference(
        segments_uv=segments,
        fs_hz=float(bundle.fs_hz),
        retained_epoch_indices=np.asarray(bundle.retained_epoch_indices, dtype=int),
        rejected_epoch_indices=np.asarray(bundle.rejected_epoch_indices, dtype=int),
        n3_epoch_indices=np.asarray(bundle.n3_epoch_indices, dtype=int),
        representative_epoch_index=representative_epoch,
        frequencies_hz=frequencies,
        normalized_psd=normalized_psd,
        acf_lags_s=acf_lags,
        mean_acf=mean_acf,
        metadata=safe_source,
    )


def _finite_node_row(array: np.ndarray, node_index: int, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    if array.ndim != 2 or node_index >= array.shape[0]:
        raise RuntimeError(f"{name} has unexpected shape {array.shape}")
    result = np.asarray(array[node_index], dtype=float)
    if not np.isfinite(result).all():
        raise RuntimeError(f"{name} node {node_index} contains non-finite values")
    return result


def run_state_audit(parameter_set: ParameterSet) -> SimulationStateAudit:
    """Run one existing model while recording additional audit-only states.

    Extending ``output_vars`` changes what is recorded, not the differential
    equations.  It does not establish any EEG measurement semantics.
    """

    module = _load_model_module(parameter_set.model_version)
    originals = {
        "aln": list(module.ALNNode.output_vars),
        "thalamus": list(module.ThalamicNode.output_vars),
        "network": list(module.ThalamoCorticalNetwork.output_vars),
    }
    started = perf_counter()
    try:
        module.ALNNode.output_vars = list(ALN_OUTPUTS)
        module.ThalamicNode.output_vars = list(THALAMUS_OUTPUTS)
        module.ThalamoCorticalNetwork.output_vars = list(NETWORK_OUTPUTS)

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
            duration=MODEL_DURATION_S * 1000.0,
        )
        model.params["backend"] = "numba"
        model.run()

        fs_hz = 1000.0 / float(model.params["sampling_dt"])
        drop = int(round(WARM_UP_S * fs_hz))
        duration_s = float(model.params["duration"]) / 1000.0
        raw = {name: np.asarray(model[name], dtype=float) for name in NETWORK_OUTPUTS}

        r_exc = _finite_node_row(raw["r_mean_EXC"], 0, "r_mean_EXC") * 1000.0
        r_inh = _finite_node_row(raw["r_mean_INH"], 0, "r_mean_INH") * 1000.0
        i_a = _finite_node_row(raw["I_A_EXC"], 0, "I_A_EXC")
        i_mu_exc = _finite_node_row(raw["I_mu_EXC"], 0, "I_mu_EXC")
        i_mu_inh = _finite_node_row(raw["I_mu_INH"], 0, "I_mu_INH")
        syn_exc = _finite_node_row(
            raw["I_syn_mu_exc_EXC"], 0, "I_syn_mu_exc_EXC"
        )
        syn_inh = _finite_node_row(
            raw["I_syn_mu_inh_EXC"], 0, "I_syn_mu_inh_EXC"
        )
        syn_sigma_exc_exc = _finite_node_row(
            raw["I_syn_sigma_exc_EXC"], 0, "I_syn_sigma_exc_EXC"
        )
        syn_sigma_inh_exc = _finite_node_row(
            raw["I_syn_sigma_inh_EXC"], 0, "I_syn_sigma_inh_EXC"
        )
        syn_exc_inh_mass = _finite_node_row(
            raw["I_syn_mu_exc_INH"], 0, "I_syn_mu_exc_INH"
        )
        syn_inh_inh_mass = _finite_node_row(
            raw["I_syn_mu_inh_INH"], 0, "I_syn_mu_inh_INH"
        )
        syn_sigma_exc_inh_mass = _finite_node_row(
            raw["I_syn_sigma_exc_INH"], 0, "I_syn_sigma_exc_INH"
        )
        syn_sigma_inh_inh_mass = _finite_node_row(
            raw["I_syn_sigma_inh_INH"], 0, "I_syn_sigma_inh_INH"
        )
        tcr_voltage = _finite_node_row(raw["V_EXC"], 1, "V_EXC")
        trn_voltage = _finite_node_row(raw["V_INH"], 1, "V_INH")

        capacitance_keys = [
            key
            for key in model.params.keys()
            if str(key).endswith("ALNMassEXC_0.C")
        ]
        if len(capacitance_keys) != 1:
            raise RuntimeError(
                f"expected one cortical excitatory capacitance, found {capacitance_keys}"
            )
        capacitance_pf = float(model.params[capacitance_keys[0]])
        effective_drive = i_mu_exc - i_a / capacitance_pf

        signals = {
            "cortex_r_exc": r_exc[drop:],
            "cortex_r_inh": r_inh[drop:],
            "cortex_rate_e_minus_i": (r_exc - r_inh)[drop:],
            "cortex_rate_e_plus_i": (r_exc + r_inh)[drop:],
            "cortex_I_mu_exc": i_mu_exc[drop:],
            "cortex_I_mu_inh": i_mu_inh[drop:],
            "cortex_adaptation_I_A": i_a[drop:],
            "cortex_effective_drive": effective_drive[drop:],
            "cortex_syn_mu_exc_on_exc": syn_exc[drop:],
            "cortex_syn_mu_inh_on_exc": syn_inh[drop:],
            "cortex_syn_sigma_exc_on_exc": syn_sigma_exc_exc[drop:],
            "cortex_syn_sigma_inh_on_exc": syn_sigma_inh_exc[drop:],
            "cortex_syn_mu_exc_on_inh": syn_exc_inh_mass[drop:],
            "cortex_syn_mu_inh_on_inh": syn_inh_inh_mass[drop:],
            "cortex_syn_sigma_exc_on_inh": syn_sigma_exc_inh_mass[drop:],
            "cortex_syn_sigma_inh_on_inh": syn_sigma_inh_inh_mass[drop:],
            "cortex_synaptic_state_balance": (syn_exc - syn_inh)[drop:],
            "thalamus_TCR_voltage": tcr_voltage[drop:],
            "thalamus_TRN_voltage": trn_voltage[drop:],
        }
        metadata = {
            "cortex_r_exc": {
                "unit": "Hz",
                "meaning": "cortical excitatory population firing rate",
                "location": "cortex",
                "availability": "project-recorded",
            },
            "cortex_r_inh": {
                "unit": "Hz",
                "meaning": "cortical inhibitory population firing rate",
                "location": "cortex",
                "availability": "project-recorded",
            },
            "cortex_rate_e_minus_i": {
                "unit": "Hz",
                "meaning": "equal-weight EXC minus INH firing-rate contrast",
                "location": "cortex",
                "availability": "audit-derived; weights not physiologically calibrated",
            },
            "cortex_rate_e_plus_i": {
                "unit": "Hz",
                "meaning": "equal-weight EXC plus INH firing-rate sum",
                "location": "cortex",
                "availability": "audit-derived; weights not physiologically calibrated",
            },
            "cortex_I_mu_exc": {
                "unit": "mV/ms",
                "meaning": "ALN excitatory-mass mean input/current drive state",
                "location": "cortex",
                "availability": "neurolib state; not recorded by project network",
            },
            "cortex_I_mu_inh": {
                "unit": "mV/ms",
                "meaning": "ALN inhibitory-mass mean input/current drive state",
                "location": "cortex",
                "availability": "neurolib state; not recorded by project network",
            },
            "cortex_adaptation_I_A": {
                "unit": "pA",
                "meaning": "ALN excitatory adaptation current",
                "location": "cortex",
                "availability": "neurolib ALNNode output; not project-recorded",
            },
            "cortex_effective_drive": {
                "unit": "mV/ms",
                "meaning": "I_mu_EXC - I_A_EXC/C, the ALN transfer-function drive",
                "location": "cortex",
                "availability": "audit-derived from model states",
            },
            "cortex_syn_mu_exc_on_exc": {
                "unit": "dimensionless synaptic state",
                "meaning": "mean excitatory synaptic state driving ALN EXC mass",
                "location": "cortex",
                "availability": "neurolib state; not project-recorded",
            },
            "cortex_syn_mu_inh_on_exc": {
                "unit": "dimensionless synaptic state",
                "meaning": "mean inhibitory synaptic state driving ALN EXC mass",
                "location": "cortex",
                "availability": "neurolib state; not project-recorded",
            },
            "cortex_syn_sigma_exc_on_exc": {
                "unit": "model-native synaptic variance state",
                "meaning": "excitatory synaptic variance state in ALN EXC mass",
                "location": "cortex",
                "availability": "neurolib state; not project-recorded",
            },
            "cortex_syn_sigma_inh_on_exc": {
                "unit": "model-native synaptic variance state",
                "meaning": "inhibitory synaptic variance state in ALN EXC mass",
                "location": "cortex",
                "availability": "neurolib state; not project-recorded",
            },
            "cortex_syn_mu_exc_on_inh": {
                "unit": "dimensionless synaptic state",
                "meaning": "mean excitatory synaptic state driving ALN INH mass",
                "location": "cortex",
                "availability": "neurolib state; not project-recorded",
            },
            "cortex_syn_mu_inh_on_inh": {
                "unit": "dimensionless synaptic state",
                "meaning": "mean inhibitory synaptic state driving ALN INH mass",
                "location": "cortex",
                "availability": "neurolib state; not project-recorded",
            },
            "cortex_syn_sigma_exc_on_inh": {
                "unit": "model-native synaptic variance state",
                "meaning": "excitatory synaptic variance state in ALN INH mass",
                "location": "cortex",
                "availability": "neurolib state; not project-recorded",
            },
            "cortex_syn_sigma_inh_on_inh": {
                "unit": "model-native synaptic variance state",
                "meaning": "inhibitory synaptic variance state in ALN INH mass",
                "location": "cortex",
                "availability": "neurolib state; not project-recorded",
            },
            "cortex_synaptic_state_balance": {
                "unit": "dimensionless state",
                "meaning": "EXC minus INH mean synaptic-state contrast",
                "location": "cortex",
                "availability": "audit-derived; not an electric potential",
            },
            "thalamus_TCR_voltage": {
                "unit": "mV",
                "meaning": "thalamocortical relay population membrane voltage",
                "location": "thalamus",
                "availability": "neurolib ThalamicNode output; internal state",
            },
            "thalamus_TRN_voltage": {
                "unit": "mV",
                "meaning": "thalamic reticular population membrane voltage",
                "location": "thalamus",
                "availability": "neurolib ThalamicNode output; internal state",
            },
        }
    finally:
        module.ALNNode.output_vars = originals["aln"]
        module.ThalamicNode.output_vars = originals["thalamus"]
        module.ThalamoCorticalNetwork.output_vars = originals["network"]

    expected_samples = int(round((duration_s - WARM_UP_S) * fs_hz))
    for name, signal in signals.items():
        if len(signal) != expected_samples or not np.isfinite(signal).all():
            raise RuntimeError(f"{name} failed post-warm-up length/finite validation")

    return SimulationStateAudit(
        parameter_set=parameter_set,
        fs_hz=fs_hz,
        runtime_s=perf_counter() - started,
        duration_s=duration_s,
        warm_up_s=WARM_UP_S,
        signals=signals,
        signal_metadata=metadata,
        warnings=[
            "Additional output recording does not change the simulator dynamics.",
            "None of these states is a validated Fpz-Cz scalp-EEG measurement.",
            "Equal-weight combinations are screening contrasts, not fitted lead fields.",
            "Only the first complete 30-second post-warm-up window enters extractor screening.",
        ],
    )


def state_catalog(audit: SimulationStateAudit) -> pd.DataFrame:
    """Describe every recorded/derived state without saving time series."""

    rows = []
    for name, signal in audit.signals.items():
        meta = audit.signal_metadata[name]
        rows.append(
            {
                "signal": name,
                **meta,
                "fs_hz": audit.fs_hz,
                "post_warm_up_duration_s": len(signal) / audit.fs_hz,
                "is_validated_scalp_eeg_proxy": False,
            }
        )
    return pd.DataFrame(rows)


def range_statistics(audits: list[SimulationStateAudit]) -> pd.DataFrame:
    """Return raw-unit statistics for each parameter set and signal."""

    rows = []
    for audit in audits:
        n_epoch = int(round(SCREENING_EPOCH_S * audit.fs_hz))
        for name, full_signal in audit.signals.items():
            signal = np.asarray(full_signal[:n_epoch], dtype=float)
            meta = audit.signal_metadata[name]
            rows.append(
                {
                    "parameter_set": audit.parameter_set.identifier,
                    "model_version": audit.parameter_set.model_version,
                    "signal": name,
                    "unit": meta["unit"],
                    "meaning": meta["meaning"],
                    "n_samples": len(signal),
                    "duration_s": len(signal) / audit.fs_hz,
                    "minimum": float(np.min(signal)),
                    "maximum": float(np.max(signal)),
                    "mean": float(np.mean(signal)),
                    "std": float(np.std(signal)),
                    "median": float(np.median(signal)),
                    "iqr": float(np.subtract(*np.percentile(signal, [75, 25]))),
                    "peak_to_peak": float(np.ptp(signal)),
                    "finite": bool(np.isfinite(signal).all()),
                }
            )
    return pd.DataFrame(rows)


def _band_fraction(
    frequencies: np.ndarray,
    normalized_psd: np.ndarray,
    band: tuple[float, float],
) -> float:
    mask = (frequencies >= band[0]) & (frequencies <= band[1])
    return float(np.trapezoid(normalized_psd[mask], frequencies[mask]))


def shape_metrics(
    audits: list[SimulationStateAudit],
    real: RealReference,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, np.ndarray]]]:
    """Compare normalized PSD and ACF shapes, never raw amplitudes."""

    rows = []
    curves: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    real_mask = (real.frequencies_hz >= 0.5) & (real.frequencies_hz <= 20.0)
    real_frequency_grid = real.frequencies_hz[real_mask]
    real_psd = real.normalized_psd[real_mask]
    for audit in audits:
        n_epoch = int(round(SCREENING_EPOCH_S * audit.fs_hz))
        for name, full_signal in audit.signals.items():
            signal = np.asarray(full_signal[:n_epoch], dtype=float)
            try:
                frequencies, normalized_psd = _normalized_welch(
                    signal,
                    audit.fs_hz,
                )
                sim_on_real = np.interp(real_frequency_grid, frequencies, normalized_psd)
                rho = float(
                    spearmanr(
                        np.log10(np.maximum(real_psd, 1e-30)),
                        np.log10(np.maximum(sim_on_real, 1e-30)),
                    ).statistic
                )
                lags, acf = _mean_acf(signal, audit.fs_hz)
                acf_on_real = np.interp(real.acf_lags_s, lags, acf)
                acf_rmse = float(np.sqrt(np.mean((acf_on_real - real.mean_acf) ** 2)))
                so_mask = (frequencies >= 0.5) & (frequencies <= 1.5)
                sigma_mask = (frequencies >= 11.0) & (frequencies <= 15.0)
                so_peak = float(frequencies[so_mask][np.argmax(normalized_psd[so_mask])])
                sigma_peak = float(
                    frequencies[sigma_mask][np.argmax(normalized_psd[sigma_mask])]
                )
                row = {
                    "parameter_set": audit.parameter_set.identifier,
                    "model_version": audit.parameter_set.model_version,
                    "signal": name,
                    "standardization": "PSD unit-area 0.5-20 Hz; ACF z-score per epoch",
                    "log_psd_spearman_vs_real": rho,
                    "acf_rmse_vs_real": acf_rmse,
                    "relative_so_power_0p5_1p5": _band_fraction(
                        frequencies, normalized_psd, (0.5, 1.5)
                    ),
                    "relative_sigma_power_11_15": _band_fraction(
                        frequencies, normalized_psd, (11.0, 15.0)
                    ),
                    "so_peak_frequency_hz": so_peak,
                    "sigma_peak_frequency_hz": sigma_peak,
                    "finite": True,
                    "measurement_parity": False,
                    "interpretation": "shape-level screening only",
                }
                curves[(audit.parameter_set.identifier, name)] = {
                    "frequencies_hz": frequencies,
                    "normalized_psd": normalized_psd,
                    "acf_lags_s": lags,
                    "acf": acf,
                }
            except (ValueError, FloatingPointError) as error:
                row = {
                    "parameter_set": audit.parameter_set.identifier,
                    "model_version": audit.parameter_set.model_version,
                    "signal": name,
                    "standardization": "failed",
                    "log_psd_spearman_vs_real": np.nan,
                    "acf_rmse_vs_real": np.nan,
                    "relative_so_power_0p5_1p5": np.nan,
                    "relative_sigma_power_11_15": np.nan,
                    "so_peak_frequency_hz": np.nan,
                    "sigma_peak_frequency_hz": np.nan,
                    "finite": False,
                    "measurement_parity": False,
                    "interpretation": f"screening failed: {error}",
                }
            rows.append(row)
    return pd.DataFrame(rows), curves


def extractor_screening(
    audits: list[SimulationStateAudit],
) -> pd.DataFrame:
    """Run existing EEG algorithms numerically, then audit their semantics."""

    config = load_observation_config(DEFAULT_CONFIG)
    rows = []
    for audit in audits:
        n_epoch = int(round(SCREENING_EPOCH_S * audit.fs_hz))
        epoch_ids = np.array([0], dtype=int)
        for name, full_signal in audit.signals.items():
            segment = np.asarray(full_signal[:n_epoch], dtype=float)
            segments = segment[np.newaxis, :]
            meta = audit.signal_metadata[name]
            row: dict[str, Any] = {
                "parameter_set": audit.parameter_set.identifier,
                "model_version": audit.parameter_set.model_version,
                "signal": name,
                "unit": meta["unit"],
                "numerically_finite_input": bool(np.isfinite(segment).all()),
                "global_qc_applicable": False,
                "global_qc_reason": "200 uV peak-to-peak threshold has no cross-modal unit mapping",
                "spectral_algorithm_runs": False,
                "spectral_semantics_shared": False,
                "so_algorithm_runs": False,
                "so_event_count": 0,
                "so_semantics_shared": False,
                "spindle_algorithm_runs": False,
                "spindle_event_count": 0,
                "spindle_semantics_shared": False,
                "pac_algorithm_runs": False,
                "pac_valid_epoch_count": 0,
                "pac_semantics_shared": False,
                "failure_reason": "",
            }
            failures = []
            try:
                psd = _compute_psd(segments, audit.fs_hz, config)
                spectral = _spectral_statistics(
                    psd.frequencies_hz, psd.aggregate_hann_uv2_hz, config
                )
                row["spectral_algorithm_runs"] = bool(
                    all(np.isfinite(value) for value in spectral.values())
                )
                row["spectral_failure_reason"] = (
                    "numeric success; PSD unit metadata would be false outside uV"
                )
            except (ValueError, RuntimeError, FloatingPointError) as error:
                row["spectral_failure_reason"] = str(error)
                failures.append(f"spectral:{error}")

            try:
                so = _compute_so_diagnostics(
                    segments, epoch_ids, audit.fs_hz, config
                )
                row["so_algorithm_runs"] = True
                row["so_event_count"] = int(len(so["events"]))
                row["so_valid_epoch_count"] = int(np.sum(so["validity_mask"]))
                row["so_failure_reason"] = (
                    "numeric result invalid for parity: detector applies 75 uV "
                    f"half-wave threshold to {meta['unit']}"
                )
            except (ValueError, RuntimeError, FloatingPointError) as error:
                row["so_failure_reason"] = str(error)
                failures.append(f"so:{error}")

            try:
                spindle = _compute_spindle_diagnostics(
                    segments, epoch_ids, audit.fs_hz, config
                )
                row["spindle_algorithm_runs"] = True
                row["spindle_event_count"] = int(len(spindle["events"]))
                row["spindle_valid_epoch_count"] = int(
                    np.sum(spindle["validity_mask"])
                )
                row["spindle_failure_reason"] = (
                    "adaptive threshold runs, but zero/nonzero events do not "
                    "establish scalp-spindle semantics for this state variable"
                )
            except (ValueError, RuntimeError, FloatingPointError) as error:
                row["spindle_failure_reason"] = str(error)
                failures.append(f"spindle:{error}")

            try:
                pac = _compute_pac_diagnostics(
                    segments, epoch_ids, audit.fs_hz, config
                )
                row["pac_algorithm_runs"] = True
                row["pac_valid_epoch_count"] = int(pac["valid_epoch_count"])
                row["pac_mi"] = float(pac["mi"])
                row["pac_failure_reason"] = (
                    "filter/Hilbert calculation runs, but phase-amplitude "
                    "meaning depends on the unvalidated measurement operator"
                )
            except (ValueError, RuntimeError, FloatingPointError) as error:
                row["pac_failure_reason"] = str(error)
                failures.append(f"pac:{error}")
            row["failure_reason"] = "; ".join(failures)
            rows.append(row)
    return pd.DataFrame(rows)


def mapping_option_decisions() -> pd.DataFrame:
    """Return the pre-registered A-H measurement-option decision table."""

    columns = [
        "option",
        "available_in_current_code",
        "physiological_meaning",
        "units",
        "can_use_eeg_extractor_unchanged",
        "required_assumptions",
        "identifiability_risk",
        "leakage_risk",
        "implementation_cost",
        "scientific_defensibility",
        "decision",
    ]
    rows = [
        [
            "A. Direct cortical firing rate",
            "Yes; Figure 5/7 and adapter use r_mean_EXC",
            "population firing rate, not an electric field",
            "Hz after kHz x 1000",
            "No",
            "rate spectrum stands in for bipolar scalp voltage",
            "High",
            "Medium if tuned on real EEG",
            "Low",
            "Shape screening only",
            "Reject for real-EEG inference",
        ],
        [
            "B. Center/standardize/linearly scale firing rate",
            "Mathematically available; z-scoring used in held-out morphology",
            "dimensionless rate shape or arbitrarily scaled rate",
            "z or arbitrary uV",
            "No",
            "linear calibration substitutes for lead field/reference",
            "High",
            "High if scale is fitted to SC4001 targets",
            "Low",
            "Does not repair channel semantics",
            "Use only for explicit shape-level external validation",
        ],
        [
            "C. EXC/INH firing-rate combination",
            "Both rates are recorded; weights are not defined",
            "population-rate contrast/sum",
            "Hz",
            "No",
            "unknown population and lead-field weights",
            "Very high",
            "High if weights selected using target",
            "Low to medium",
            "Uncalibrated screening contrast",
            "Do not freeze as EEG proxy",
        ],
        [
            "D. Membrane-potential or synaptic-current proxy",
            "Thalamic V and cortical current states exist; cortical V is lookup-only",
            "internal membrane/current states",
            "mV, mV/ms, pA, dimensionless states",
            "No",
            "internal state projects directly to Fpz-Cz",
            "High",
            "Medium",
            "Medium",
            "Closer biophysics, still no scalp measurement model",
            "Research candidate for a future forward model",
        ],
        [
            "E. Cortical mass/LFP-like observable",
            "No validated implementation found",
            "would approximate aggregate transmembrane/synaptic currents",
            "model dependent",
            "No",
            "geometry, source orientation, conductivity and reference",
            "High",
            "Medium",
            "High",
            "Potentially defensible after validation",
            "Not currently available",
        ],
        [
            "F. Explicit EEG forward/measurement model",
            "Generic neurolib leadfield utility exists; not integrated in project/model",
            "source-to-sensor projection with channel/reference contract",
            "uV after calibrated gain",
            "Only after validation",
            "define a dipole/source observable, map the two-node model to geometry, then specify reference and noise",
            "Medium to high",
            "Must separate calibration and validation data",
            "High",
            "Required path; generic geometry code alone is insufficient",
            "Recommended development route",
        ],
        [
            "G. Synthetic cortical-observable SBI only",
            "Yes",
            "parameters inferred from the same model observable",
            "native model units",
            "Use a model-observable extractor, not the EEG contract",
            "claims restricted to synthetic recovery",
            "Lower for synthetic task",
            "Low",
            "Medium",
            "Defensible with explicit scope",
            "Allowed interim route; not real-EEG inference",
        ],
        [
            "H. Pause SBI; real EEG as held-out external validation",
            "Yes",
            "qualitative/shape-level external comparison",
            "modality-specific",
            "No",
            "no parameter inference from EEG",
            "Low",
            "Low",
            "Low",
            "Conservative and auditable",
            "Use until a measurement model is validated",
        ],
    ]
    return pd.DataFrame(rows, columns=columns)
