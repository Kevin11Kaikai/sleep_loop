"""Pyloric-inspired feature engineering for the real SC4001 observation.

The existing :mod:`sleep_sbi.observation` module remains the authoritative
loader, epoch boundary policy, QC implementation, and detector definition.
This module adds auditable temporal and event-conditioned summaries without
changing those upstream definitions.

The full sample arrays stored in ``ObservationBundle`` are used only in
memory. Export helpers in this module serialize aggregate summaries, per-epoch
features, epoch accounting, and figures, never raw EEG samples.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import yaml

from .observation import (
    PROJECT_ROOT,
    _load_epoch_data,
    _require_fooof,
    _spectral_statistics,
    load_observation_config,
)
from .schemas import ObservationBundle, ObservationConfig


CORE_FIELDS: tuple[str, ...] = (
    "fooof_aperiodic_exponent",
    "relative_so_power",
    "so_peak_frequency_hz",
    "so_q",
    "so_event_rate_per_min",
    "so_median_ibi_s",
    "ibi_cv",
    "so_up_proxy_duration_s",
    "so_down_proxy_duration_s",
    "so_up_proxy_duty_cycle",
    "waveform_peak_to_peak_z",
    "so_trough_to_peak_time_s",
    "spindle_density_per_min",
    "spindle_mean_duration_s",
    "spindle_occupancy",
    "spindle_onset_phase_cos",
    "spindle_onset_phase_sin",
    "spindle_onset_phase_concentration",
)

INFERENCE_FIELDS = frozenset(
    {
        "fooof_aperiodic_exponent",
        "relative_so_power",
        "so_peak_frequency_hz",
        "so_q",
        "so_event_rate_per_min",
        "so_median_ibi_s",
        "ibi_cv",
        "so_up_proxy_duration_s",
        "so_down_proxy_duration_s",
    }
)

HELD_OUT_FIELDS = frozenset(set(CORE_FIELDS) - set(INFERENCE_FIELDS))


FEATURE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "field_name": "fooof_aperiodic_exponent",
        "domain": "spectral_structure",
        "pyloric_analogy": "baseline state/shape",
        "unit": "1",
        "frequency_band": "0.5-20 Hz",
        "algorithm": "FOOOF 1.1.1 fixed aperiodic fit",
        "aggregation_method": "fit to arithmetic-mean Hann epoch PSD",
        "intended_role": "inference_summary_candidate",
        "dependency_or_redundancy": "spectral baseline; may covary with relative SO power",
    },
    {
        "field_name": "relative_so_power",
        "domain": "spectral_structure",
        "pyloric_analogy": "relative state strength",
        "unit": "1",
        "frequency_band": "0.2-1.5 Hz / 0.5-20 Hz",
        "algorithm": "trapezoidal power ratio from Hann Welch PSD",
        "aggregation_method": "ratio from arithmetic-mean retained-epoch PSD",
        "intended_role": "inference_summary_candidate",
        "dependency_or_redundancy": "related to SO Q and aperiodic exponent",
    },
    {
        "field_name": "so_peak_frequency_hz",
        "domain": "spectral_structure",
        "pyloric_analogy": "rhythm speed",
        "unit": "Hz",
        "frequency_band": "0.2-1.5 Hz",
        "algorithm": "maximum of aggregate Hann Welch PSD in SO band",
        "aggregation_method": "arithmetic-mean retained-epoch PSD",
        "intended_role": "inference_summary_candidate",
        "dependency_or_redundancy": "frequency-bin resolution limited",
    },
    {
        "field_name": "so_q",
        "domain": "spectral_structure",
        "pyloric_analogy": "rhythm selectivity",
        "unit": "1",
        "frequency_band": "0.2-1.5 Hz",
        "algorithm": "SO peak PSD / adjacent-band mean PSD",
        "aggregation_method": "computed from aggregate Hann PSD",
        "intended_role": "inference_summary_candidate",
        "dependency_or_redundancy": "related to relative SO power",
    },
    {
        "field_name": "so_event_rate_per_min",
        "domain": "so_rhythm",
        "pyloric_analogy": "cycle frequency",
        "unit": "events/min",
        "frequency_band": "0.2-4 Hz detector",
        "algorithm": "existing DOWN-to-UP half-wave detector",
        "aggregation_method": "events / valid retained-N3 minutes",
        "intended_role": "inference_summary_candidate",
        "dependency_or_redundancy": "approximately 60 / median IBI for regular rhythms",
    },
    {
        "field_name": "so_median_ibi_s",
        "domain": "so_rhythm",
        "pyloric_analogy": "cycle period",
        "unit": "s",
        "frequency_band": "0.2-4 Hz detector",
        "algorithm": "within-epoch UP-event intervals",
        "aggregation_method": "median of per-epoch median IBI values",
        "intended_role": "inference_summary_candidate",
        "dependency_or_redundancy": "inverse relationship with SO event rate",
    },
    {
        "field_name": "ibi_cv",
        "domain": "so_rhythm",
        "pyloric_analogy": "cycle regularity",
        "unit": "1",
        "frequency_band": "0.2-4 Hz detector",
        "algorithm": "CV of within-epoch UP-event intervals",
        "aggregation_method": "all valid within-epoch intervals pooled; no boundary IBI",
        "intended_role": "inference_summary_candidate",
        "dependency_or_redundancy": "depends on event count and median IBI",
    },
    {
        "field_name": "so_up_proxy_duration_s",
        "domain": "so_temporal_morphology",
        "pyloric_analogy": "state duration",
        "unit": "s",
        "frequency_band": "0.2-4 Hz detector",
        "algorithm": "positive-phase zero-crossing duration around accepted SO event",
        "aggregation_method": "median of per-epoch medians over complete cycles",
        "intended_role": "inference_summary_candidate",
        "dependency_or_redundancy": "EEG observable proxy; not model cortical UP state",
    },
    {
        "field_name": "so_down_proxy_duration_s",
        "domain": "so_temporal_morphology",
        "pyloric_analogy": "state duration",
        "unit": "s",
        "frequency_band": "0.2-4 Hz detector",
        "algorithm": "negative-phase zero-crossing duration around accepted SO event",
        "aggregation_method": "median of per-epoch medians over complete cycles",
        "intended_role": "inference_summary_candidate",
        "dependency_or_redundancy": "EEG observable proxy; not model cortical DOWN state",
    },
    {
        "field_name": "so_up_proxy_duty_cycle",
        "domain": "so_temporal_morphology",
        "pyloric_analogy": "duty cycle",
        "unit": "1",
        "frequency_band": "0.2-4 Hz detector",
        "algorithm": "UP-proxy / (UP-proxy + DOWN-proxy)",
        "aggregation_method": "median of per-epoch medians over complete cycles",
        "intended_role": "held_out_ppc_candidate",
        "dependency_or_redundancy": "deterministic function of paired proxy durations",
    },
    {
        "field_name": "waveform_peak_to_peak_z",
        "domain": "so_temporal_morphology",
        "pyloric_analogy": "waveform shape",
        "unit": "z",
        "frequency_band": "0.2-4 Hz detector",
        "algorithm": "trough-locked mean of per-event z-scored SO waveforms",
        "aggregation_method": "peak-to-peak of aggregate aligned mean waveform",
        "intended_role": "held_out_ppc_candidate",
        "dependency_or_redundancy": "scale invariant but detector-conditioned",
    },
    {
        "field_name": "so_trough_to_peak_time_s",
        "domain": "so_temporal_morphology",
        "pyloric_analogy": "within-cycle relative timing",
        "unit": "s",
        "frequency_band": "0.2-4 Hz detector",
        "algorithm": "accepted trough-to-subsequent-peak latency",
        "aggregation_method": "median of per-epoch medians over complete cycles",
        "intended_role": "held_out_ppc_candidate",
        "dependency_or_redundancy": "component of complete-cycle proxy timing",
    },
    {
        "field_name": "spindle_density_per_min",
        "domain": "spindle",
        "pyloric_analogy": "event rate",
        "unit": "events/min",
        "frequency_band": "11-15 Hz",
        "algorithm": "existing per-epoch moving-RMS detector",
        "aggregation_method": "events / valid retained-N3 minutes",
        "intended_role": "held_out_ppc_candidate",
        "dependency_or_redundancy": "approximately occupancy / mean duration",
    },
    {
        "field_name": "spindle_mean_duration_s",
        "domain": "spindle",
        "pyloric_analogy": "event duration",
        "unit": "s",
        "frequency_band": "11-15 Hz",
        "algorithm": "existing per-epoch moving-RMS detector",
        "aggregation_method": "arithmetic mean across detected events",
        "intended_role": "held_out_ppc_candidate",
        "dependency_or_redundancy": "undefined when no events are detected",
    },
    {
        "field_name": "spindle_occupancy",
        "domain": "spindle",
        "pyloric_analogy": "state occupancy",
        "unit": "1",
        "frequency_band": "11-15 Hz",
        "algorithm": "union duration of accepted spindle intervals",
        "aggregation_method": "union duration / valid observation duration",
        "intended_role": "held_out_ppc_candidate",
        "dependency_or_redundancy": "approximately density * mean duration / 60",
    },
    {
        "field_name": "spindle_onset_phase_cos",
        "domain": "event_conditioned_coordination",
        "pyloric_analogy": "relative event phase",
        "unit": "1",
        "frequency_band": "SO 0.2-4 Hz; spindle 11-15 Hz",
        "algorithm": "cosine of trough-anchored circular mean onset phase",
        "aggregation_method": "all valid within-epoch onset-cycle pairs",
        "intended_role": "held_out_ppc_candidate",
        "dependency_or_redundancy": "paired with phase sine; interpret with concentration",
    },
    {
        "field_name": "spindle_onset_phase_sin",
        "domain": "event_conditioned_coordination",
        "pyloric_analogy": "relative event phase",
        "unit": "1",
        "frequency_band": "SO 0.2-4 Hz; spindle 11-15 Hz",
        "algorithm": "sine of trough-anchored circular mean onset phase",
        "aggregation_method": "all valid within-epoch onset-cycle pairs",
        "intended_role": "held_out_ppc_candidate",
        "dependency_or_redundancy": "paired with phase cosine; interpret with concentration",
    },
    {
        "field_name": "spindle_onset_phase_concentration",
        "domain": "event_conditioned_coordination",
        "pyloric_analogy": "phase consistency",
        "unit": "1",
        "frequency_band": "SO 0.2-4 Hz; spindle 11-15 Hz",
        "algorithm": "resultant vector length of trough-anchored onset phases",
        "aggregation_method": "all valid within-epoch onset-cycle pairs",
        "intended_role": "held_out_ppc_candidate",
        "dependency_or_redundancy": "low values make mean phase physiologically unstable",
    },
)


@dataclass(frozen=True)
class PyloricInspiredConfig:
    """Incremental settings that do not alter the upstream Observation detector."""

    base_observation_config: str
    random_seed: int
    cycle_duration_s: tuple[float, float]
    polarity_convention: str
    pairing_cycle_duration_s: tuple[float, float]
    minimum_global_paired_events: int
    minimum_per_epoch_paired_events: int
    phase_convention: str
    redundancy_abs_spearman: float
    minimum_pair_count: int
    near_constant_iqr: float
    artifact_output_dir: str
    artifact_dpi: int
    config_hash: str


@dataclass
class PyloricInspiredResult:
    """Publication-safe tables plus in-memory diagnostic tables."""

    summary: pd.DataFrame
    per_epoch_features: pd.DataFrame
    epoch_ledger: pd.DataFrame
    feature_dictionary: pd.DataFrame
    auxiliary_diagnostics: pd.DataFrame
    old_vs_new_regression: pd.DataFrame
    stability_audit: pd.DataFrame
    spearman_correlation: pd.DataFrame
    valid_pair_count: pd.DataFrame
    validation_checks: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict, repr=False)


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_pyloric_inspired_config(
    config_path: str | Path = (
        "S4_sbi/configs/pyloric_inspired_observation_sc4001.yaml"
    ),
) -> PyloricInspiredConfig:
    """Load and hash the incremental configuration."""
    path = _resolve_project_path(config_path)
    raw_bytes = path.read_bytes()
    payload = yaml.safe_load(raw_bytes)
    so = payload["slow_oscillation_proxy"]
    coordination = payload["event_conditioned_coordination"]
    stability = payload["stability"]
    artifacts = payload["artifacts"]
    return PyloricInspiredConfig(
        base_observation_config=str(payload["base_observation_config"]),
        random_seed=int(payload["reproducibility"]["random_seed"]),
        cycle_duration_s=tuple(map(float, so["complete_cycle_duration_s"])),
        polarity_convention=str(so["polarity_convention"]),
        pairing_cycle_duration_s=tuple(
            map(float, coordination["trough_to_trough_cycle_duration_s"])
        ),
        minimum_global_paired_events=int(
            coordination["minimum_global_paired_events"]
        ),
        minimum_per_epoch_paired_events=int(
            coordination["minimum_per_epoch_paired_events"]
        ),
        phase_convention=str(coordination["phase_convention"]),
        redundancy_abs_spearman=float(
            stability["redundancy_abs_spearman"]
        ),
        minimum_pair_count=int(stability["minimum_pair_count"]),
        near_constant_iqr=float(stability["near_constant_iqr"]),
        artifact_output_dir=str(artifacts["output_dir"]),
        artifact_dpi=int(artifacts["dpi"]),
        config_hash=sha256(raw_bytes).hexdigest(),
    )


def feature_dictionary() -> pd.DataFrame:
    """Return the stable 18-row feature definition table."""
    frame = pd.DataFrame(FEATURE_DEFINITIONS)
    if tuple(frame["field_name"]) != CORE_FIELDS:
        raise RuntimeError("feature definitions do not match the frozen 18D order")
    return frame


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _hierarchical_median(
    frame: pd.DataFrame, value_column: str
) -> tuple[float, int, int]:
    valid = frame.loc[np.isfinite(frame[value_column]), ["epoch_index", value_column]]
    if valid.empty:
        return float("nan"), 0, 0
    epoch_values = valid.groupby("epoch_index")[value_column].median()
    return (
        float(epoch_values.median()),
        int(epoch_values.size),
        int(valid.shape[0]),
    )


def _interpolated_crossing(
    signal_uv: np.ndarray, left_sample: int, right_sample: int
) -> float:
    """Linearly interpolate a zero crossing between adjacent samples."""
    left = float(signal_uv[left_sample])
    right = float(signal_uv[right_sample])
    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    denominator = right - left
    if abs(denominator) < 1e-15:
        return float(left_sample)
    fraction = -left / denominator
    return float(left_sample + np.clip(fraction, 0.0, 1.0))


def _find_previous_pos_to_neg(
    signal_uv: np.ndarray, before_sample: int
) -> float | None:
    for right in range(min(before_sample, len(signal_uv) - 1), 0, -1):
        left = right - 1
        if signal_uv[left] >= 0.0 and signal_uv[right] < 0.0:
            return _interpolated_crossing(signal_uv, left, right)
    return None


def _find_neg_to_pos(
    signal_uv: np.ndarray, start_sample: int, stop_sample: int
) -> float | None:
    stop = min(stop_sample, len(signal_uv) - 1)
    for right in range(max(1, start_sample + 1), stop + 1):
        left = right - 1
        if signal_uv[left] <= 0.0 and signal_uv[right] > 0.0:
            return _interpolated_crossing(signal_uv, left, right)
    return None


def _find_next_pos_to_neg(
    signal_uv: np.ndarray, after_sample: int
) -> float | None:
    for right in range(max(1, after_sample + 1), len(signal_uv)):
        left = right - 1
        if signal_uv[left] >= 0.0 and signal_uv[right] < 0.0:
            return _interpolated_crossing(signal_uv, left, right)
    return None


def extract_so_proxy_cycles(
    bundle: ObservationBundle,
    config: PyloricInspiredConfig,
) -> pd.DataFrame:
    """Extract complete EEG-positive/negative phase proxies per retained epoch.

    Every accepted upstream SO event is audited. Events lacking any required
    zero crossing, having an incompatible polarity, or producing a cycle
    outside the configured complete-cycle support remain in the returned table
    as invalid rows with a concrete reason. No search crosses a 30-second epoch.
    """
    so = bundle.diagnostics["slow_oscillation"]
    row_by_epoch = {
        int(epoch_index): row
        for row, epoch_index in enumerate(bundle.retained_epoch_indices)
    }
    records: list[dict[str, Any]] = []
    for event_index, event in enumerate(so["events"]):
        epoch_index = int(event["epoch_index"])
        row = row_by_epoch[epoch_index]
        signal_uv = np.asarray(so["filtered_uv"][row], dtype=float)
        trough = int(event["down_sample"])
        peak = int(event["up_sample"])
        record = {
            "epoch_index": epoch_index,
            "event_index": int(event_index),
            "trough_sample": trough,
            "peak_sample": peak,
            "negative_entry_sample": np.nan,
            "negative_to_positive_sample": np.nan,
            "positive_exit_sample": np.nan,
            "so_down_proxy_duration_s": np.nan,
            "so_up_proxy_duration_s": np.nan,
            "so_up_proxy_duty_cycle": np.nan,
            "so_trough_to_peak_time_s": np.nan,
            "valid": False,
            "invalid_reason": None,
        }
        if not np.isfinite(signal_uv).all():
            record["invalid_reason"] = "so_filter_non_finite"
            records.append(record)
            continue
        if not (0 <= trough < peak < len(signal_uv)):
            record["invalid_reason"] = "invalid_trough_peak_order"
            records.append(record)
            continue
        if signal_uv[trough] >= 0.0 or signal_uv[peak] <= 0.0:
            record["invalid_reason"] = "polarity_convention_not_satisfied"
            records.append(record)
            continue
        negative_entry = _find_previous_pos_to_neg(signal_uv, trough)
        negative_to_positive = _find_neg_to_pos(signal_uv, trough, peak)
        positive_exit = _find_next_pos_to_neg(
            signal_uv, int(np.floor(negative_to_positive))
        )
        if negative_entry is None:
            record["invalid_reason"] = "incomplete_cycle_at_left_epoch_boundary"
            records.append(record)
            continue
        if negative_to_positive is None:
            record["invalid_reason"] = "missing_negative_to_positive_crossing"
            records.append(record)
            continue
        if positive_exit is None:
            record["invalid_reason"] = "incomplete_cycle_at_right_epoch_boundary"
            records.append(record)
            continue
        if peak > positive_exit:
            record["invalid_reason"] = (
                "detected_up_peak_outside_immediate_positive_phase"
            )
            records.append(record)
            continue
        down_duration = (negative_to_positive - negative_entry) / bundle.fs_hz
        up_duration = (positive_exit - negative_to_positive) / bundle.fs_hz
        cycle_duration = down_duration + up_duration
        if not (
            config.cycle_duration_s[0]
            <= cycle_duration
            <= config.cycle_duration_s[1]
        ):
            record["invalid_reason"] = "complete_cycle_duration_out_of_range"
            records.append(record)
            continue
        if down_duration <= 0.0 or up_duration <= 0.0:
            record["invalid_reason"] = "non_positive_proxy_duration"
            records.append(record)
            continue
        record.update(
            {
                "negative_entry_sample": float(negative_entry),
                "negative_to_positive_sample": float(negative_to_positive),
                "positive_exit_sample": float(positive_exit),
                "so_down_proxy_duration_s": float(down_duration),
                "so_up_proxy_duration_s": float(up_duration),
                "so_up_proxy_duty_cycle": float(up_duration / cycle_duration),
                "so_trough_to_peak_time_s": float(
                    (peak - trough) / bundle.fs_hz
                ),
                "valid": True,
            }
        )
        records.append(record)
    columns = [
        "epoch_index",
        "event_index",
        "trough_sample",
        "peak_sample",
        "negative_entry_sample",
        "negative_to_positive_sample",
        "positive_exit_sample",
        "so_down_proxy_duration_s",
        "so_up_proxy_duration_s",
        "so_up_proxy_duty_cycle",
        "so_trough_to_peak_time_s",
        "valid",
        "invalid_reason",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def _compute_epoch_fooof_exponents(
    bundle: ObservationBundle, observation_config: ObservationConfig
) -> tuple[np.ndarray, list[str | None]]:
    """Fit the locked FOOOF implementation independently to each epoch PSD."""
    FOOOF, installed_version = _require_fooof(observation_config.fooof_version)
    if installed_version != "1.1.1":
        raise RuntimeError(
            f"FOOOF 1.1.1 is required, found {installed_version}; no fallback"
        )
    frequencies = bundle.psd.frequencies_hz
    mask = (
        (frequencies >= observation_config.psd_band_hz[0])
        & (frequencies <= observation_config.psd_band_hz[1])
    )
    values = np.full(len(bundle.retained_epoch_indices), np.nan, dtype=float)
    reasons: list[str | None] = [None] * len(values)
    for row, density in enumerate(bundle.psd.epoch_psd_hann_uv2_hz):
        try:
            model = FOOOF(
                **observation_config.fooof_parameters, verbose=False
            )
            model.fit(
                frequencies[mask],
                np.asarray(density, dtype=float)[mask],
                list(observation_config.psd_band_hz),
            )
            if model.aperiodic_params_.size < 2:
                reasons[row] = "fooof_missing_fixed_exponent"
            else:
                exponent = float(model.aperiodic_params_[1])
                if np.isfinite(exponent):
                    values[row] = exponent
                else:
                    reasons[row] = "fooof_non_finite_exponent"
        except (RuntimeError, ValueError, FloatingPointError) as error:
            reasons[row] = f"fooof_failed: {type(error).__name__}: {error}"
    return values, reasons


def _per_epoch_spectral_features(
    bundle: ObservationBundle, observation_config: ObservationConfig
) -> pd.DataFrame:
    exponents, reasons = _compute_epoch_fooof_exponents(
        bundle, observation_config
    )
    rows: list[dict[str, Any]] = []
    for row_index, (epoch_index, density) in enumerate(
        zip(
            bundle.retained_epoch_indices,
            bundle.psd.epoch_psd_hann_uv2_hz,
        )
    ):
        statistics = _spectral_statistics(
            bundle.psd.frequencies_hz,
            np.asarray(density, dtype=float),
            observation_config,
        )
        rows.append(
            {
                "epoch_index": int(epoch_index),
                "fooof_aperiodic_exponent": float(exponents[row_index]),
                "fooof_invalid_reason": reasons[row_index],
                **{key: float(value) for key, value in statistics.items()},
            }
        )
    return pd.DataFrame(rows)


def _per_epoch_waveform_peak_to_peak(
    bundle: ObservationBundle, observation_config: ObservationConfig
) -> pd.DataFrame:
    """Rebuild the existing trough-locked z-waveform statistic per epoch."""
    so = bundle.diagnostics["slow_oscillation"]
    row_by_epoch = {
        int(epoch_index): row
        for row, epoch_index in enumerate(bundle.retained_epoch_indices)
    }
    half_window = int(
        round(observation_config.waveform_half_window_s * bundle.fs_hz)
    )
    snippets_by_epoch: dict[int, list[np.ndarray]] = defaultdict(list)
    for event in so["events"]:
        if not event["waveform_valid"]:
            continue
        epoch_index = int(event["epoch_index"])
        signal_uv = np.asarray(
            so["filtered_uv"][row_by_epoch[epoch_index]], dtype=float
        )
        center = int(event["down_sample"])
        snippet = signal_uv[
            center - half_window : center + half_window + 1
        ]
        standard_deviation = float(np.std(snippet))
        if len(snippet) == 2 * half_window + 1 and standard_deviation > 1e-9:
            snippets_by_epoch[epoch_index].append(
                (snippet - np.mean(snippet)) / standard_deviation
            )
    rows = []
    for epoch_index in map(int, bundle.retained_epoch_indices):
        snippets = snippets_by_epoch.get(epoch_index, [])
        rows.append(
            {
                "epoch_index": epoch_index,
                "waveform_peak_to_peak_z": (
                    float(np.ptp(np.mean(np.vstack(snippets), axis=0)))
                    if snippets
                    else np.nan
                ),
                "waveform_event_count": int(len(snippets)),
                "waveform_invalid_reason": (
                    None if snippets else "no_complete_trough_locked_waveform"
                ),
            }
        )
    return pd.DataFrame(rows)


def _merge_intervals(
    intervals: Iterable[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Return a non-overlapping union of half-open sample intervals."""
    ordered = sorted(
        (int(start), int(stop))
        for start, stop in intervals
        if int(stop) > int(start)
    )
    merged: list[list[int]] = []
    for start, stop in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return [(start, stop) for start, stop in merged]


def spindle_epoch_features(
    bundle: ObservationBundle,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-epoch spindle occupancy using interval union duration.

    Detector-valid zero-event epochs have density and occupancy equal to zero,
    while their mean event duration remains undefined.
    """
    spindle = bundle.diagnostics["spindle"]
    events_by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in spindle["events"]:
        events_by_epoch[int(event["epoch_index"])].append(event)
    rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    samples_per_epoch = int(round(bundle.fs_hz * bundle.epoch_duration_s))
    for row_index, epoch_index_value in enumerate(
        bundle.retained_epoch_indices
    ):
        epoch_index = int(epoch_index_value)
        detector_valid = bool(spindle["validity_mask"][row_index])
        events = events_by_epoch.get(epoch_index, [])
        if not detector_valid:
            rows.append(
                {
                    "epoch_index": epoch_index,
                    "spindle_detector_valid": False,
                    "spindle_event_count": 0,
                    "spindle_density_per_min": np.nan,
                    "spindle_mean_duration_s": np.nan,
                    "spindle_occupancy": np.nan,
                    "spindle_invalid_reason": spindle[
                        "failure_reasons"
                    ].get(epoch_index, "spindle_detector_invalid"),
                }
            )
            continue
        merged = _merge_intervals(
            (event["start_sample"], event["stop_sample"])
            for event in events
        )
        union_samples = sum(stop - start for start, stop in merged)
        durations = [float(event["duration_s"]) for event in events]
        for union_index, (start, stop) in enumerate(merged):
            interval_rows.append(
                {
                    "epoch_index": epoch_index,
                    "union_interval_index": int(union_index),
                    "start_sample": int(start),
                    "stop_sample": int(stop),
                    "duration_s": float((stop - start) / bundle.fs_hz),
                }
            )
        rows.append(
            {
                "epoch_index": epoch_index,
                "spindle_detector_valid": True,
                "spindle_event_count": int(len(events)),
                "spindle_density_per_min": float(
                    len(events) / (bundle.epoch_duration_s / 60.0)
                ),
                "spindle_mean_duration_s": (
                    float(np.mean(durations)) if durations else np.nan
                ),
                "spindle_occupancy": float(
                    union_samples / samples_per_epoch
                ),
                "spindle_invalid_reason": (
                    None
                    if events
                    else "no_spindle_events_mean_duration_undefined"
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(interval_rows)


def _circular_statistics(phases_rad: Iterable[float]) -> dict[str, float]:
    phases = _finite(phases_rad)
    if not len(phases):
        return {
            "mean_angle_rad": np.nan,
            "mean_cos": np.nan,
            "mean_sin": np.nan,
            "concentration": np.nan,
        }
    complex_mean = np.mean(np.exp(1j * phases))
    angle = float(np.angle(complex_mean))
    return {
        "mean_angle_rad": angle,
        "mean_cos": float(np.cos(angle)),
        "mean_sin": float(np.sin(angle)),
        "concentration": float(np.abs(complex_mean)),
    }


def pair_spindle_onsets_to_so_cycles(
    bundle: ObservationBundle,
    config: PyloricInspiredConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map spindle onsets to trough-to-trough SO phase within each epoch.

    The current accepted SO trough defines phase 0 and the next accepted trough
    phase 2*pi. Only spindle onsets bracketed by a complete, duration-supported
    trough pair in the same 30-second epoch contribute.
    """
    troughs_by_epoch: dict[int, list[int]] = defaultdict(list)
    for event in bundle.diagnostics["slow_oscillation"]["events"]:
        troughs_by_epoch[int(event["epoch_index"])].append(
            int(event["down_sample"])
        )
    onsets_by_epoch: dict[int, list[int]] = defaultdict(list)
    for event in bundle.diagnostics["spindle"]["events"]:
        onsets_by_epoch[int(event["epoch_index"])].append(
            int(event["start_sample"])
        )
    pair_records: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []
    for epoch_index_value in bundle.retained_epoch_indices:
        epoch_index = int(epoch_index_value)
        troughs = np.asarray(
            sorted(set(troughs_by_epoch.get(epoch_index, []))), dtype=int
        )
        onsets = np.asarray(
            sorted(onsets_by_epoch.get(epoch_index, [])), dtype=int
        )
        used_onsets: set[int] = set()
        for cycle_index, (left, right) in enumerate(
            zip(troughs[:-1], troughs[1:])
        ):
            cycle_duration_s = (right - left) / bundle.fs_hz
            if not (
                config.pairing_cycle_duration_s[0]
                <= cycle_duration_s
                <= config.pairing_cycle_duration_s[1]
            ):
                continue
            for onset in onsets[(onsets >= left) & (onsets < right)]:
                onset_int = int(onset)
                if onset_int in used_onsets:
                    continue
                used_onsets.add(onset_int)
                phase = 2.0 * np.pi * (onset_int - left) / (right - left)
                pair_records.append(
                    {
                        "epoch_index": epoch_index,
                        "cycle_index": int(cycle_index),
                        "left_trough_sample": int(left),
                        "right_trough_sample": int(right),
                        "cycle_duration_s": float(cycle_duration_s),
                        "spindle_onset_sample": onset_int,
                        "phase_rad": float(phase),
                    }
                )
        epoch_phases = [
            row["phase_rad"]
            for row in pair_records
            if row["epoch_index"] == epoch_index
        ]
        enough = len(epoch_phases) >= config.minimum_per_epoch_paired_events
        circular = (
            _circular_statistics(epoch_phases)
            if enough
            else {
                "mean_angle_rad": np.nan,
                "mean_cos": np.nan,
                "mean_sin": np.nan,
                "concentration": np.nan,
            }
        )
        epoch_records.append(
            {
                "epoch_index": epoch_index,
                "paired_event_count": int(len(epoch_phases)),
                "spindle_onset_phase_cos": circular["mean_cos"],
                "spindle_onset_phase_sin": circular["mean_sin"],
                "spindle_onset_phase_concentration": circular[
                    "concentration"
                ],
                "pairing_valid": bool(enough),
                "pairing_invalid_reason": (
                    None
                    if enough
                    else "fewer_than_minimum_per_epoch_paired_events"
                ),
            }
        )
    return pd.DataFrame(pair_records), pd.DataFrame(epoch_records)


def _per_epoch_so_features(
    bundle: ObservationBundle, cycles: pd.DataFrame
) -> pd.DataFrame:
    so = bundle.diagnostics["slow_oscillation"]
    basic = pd.DataFrame(so["per_epoch"]).rename(
        columns={
            "event_count": "so_event_count",
            "event_rate_per_min": "so_event_rate_per_min",
            "median_ibi_s": "so_median_ibi_s",
            "ibi_cv": "ibi_cv",
            "ibi_count": "ibi_interval_count",
            "invalid_reason": "ibi_invalid_reason",
        }
    )
    valid_cycles = cycles.loc[cycles["valid"]].copy()
    aggregate_columns = [
        "so_up_proxy_duration_s",
        "so_down_proxy_duration_s",
        "so_up_proxy_duty_cycle",
        "so_trough_to_peak_time_s",
    ]
    if valid_cycles.empty:
        morphology = pd.DataFrame(
            {"epoch_index": bundle.retained_epoch_indices.astype(int)}
        )
        for column in aggregate_columns:
            morphology[column] = np.nan
        morphology["so_proxy_cycle_count"] = 0
    else:
        morphology = (
            valid_cycles.groupby("epoch_index")[aggregate_columns]
            .median()
            .reset_index()
        )
        counts = (
            valid_cycles.groupby("epoch_index")
            .size()
            .rename("so_proxy_cycle_count")
            .reset_index()
        )
        morphology = morphology.merge(counts, on="epoch_index", how="left")
    all_epochs = pd.DataFrame(
        {"epoch_index": bundle.retained_epoch_indices.astype(int)}
    )
    result = all_epochs.merge(basic, on="epoch_index", how="left").merge(
        morphology, on="epoch_index", how="left"
    )
    result["so_proxy_cycle_count"] = (
        result["so_proxy_cycle_count"].fillna(0).astype(int)
    )
    result["so_proxy_invalid_reason"] = np.where(
        result["so_proxy_cycle_count"] > 0,
        None,
        "no_complete_so_proxy_cycle",
    )
    return result


def build_per_epoch_features(
    bundle: ObservationBundle,
    observation_config: ObservationConfig,
    config: PyloricInspiredConfig,
    cycles: pd.DataFrame,
    spindle_features: pd.DataFrame,
    pairing_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build one auditable row per globally retained N3 epoch."""
    spectral = _per_epoch_spectral_features(bundle, observation_config)
    so = _per_epoch_so_features(bundle, cycles)
    waveforms = _per_epoch_waveform_peak_to_peak(bundle, observation_config)
    result = spectral.merge(so, on="epoch_index", how="outer")
    result = result.merge(waveforms, on="epoch_index", how="outer")
    result = result.merge(spindle_features, on="epoch_index", how="outer")
    result = result.merge(pairing_features, on="epoch_index", how="outer")
    result = result.sort_values("epoch_index").reset_index(drop=True)
    if len(result) != len(bundle.retained_epoch_indices):
        raise RuntimeError("per-epoch feature rows do not match retained epochs")
    if tuple(result["epoch_index"]) != tuple(
        map(int, bundle.retained_epoch_indices)
    ):
        raise RuntimeError("per-epoch feature ordering does not match bundle")
    return result


def build_epoch_ledger(
    bundle: ObservationBundle, per_epoch_features: pd.DataFrame
) -> pd.DataFrame:
    """Account for every aligned epoch and metric-specific support."""
    mapped_stages = np.asarray(bundle.diagnostics["mapped_stages"], dtype=object)
    raw_labels = np.asarray(
        bundle.diagnostics["raw_epoch_labels"], dtype=object
    )
    if len(mapped_stages) != len(raw_labels):
        raise RuntimeError("raw and normalized label arrays differ in length")
    retained = set(map(int, bundle.retained_epoch_indices))
    rejected = set(map(int, bundle.rejected_epoch_indices))
    n3 = set(map(int, bundle.n3_epoch_indices))
    feature_by_epoch = per_epoch_features.set_index("epoch_index")
    records = []
    for epoch_index, (raw_label, stage) in enumerate(
        zip(raw_labels, mapped_stages)
    ):
        is_n3 = epoch_index in n3
        if epoch_index in retained:
            qc_status = "retained"
        elif epoch_index in rejected:
            qc_status = "rejected"
        else:
            qc_status = "not_applicable_non_n3"
        row = (
            feature_by_epoch.loc[epoch_index]
            if epoch_index in feature_by_epoch.index
            else None
        )
        records.append(
            {
                "epoch_index": int(epoch_index),
                "raw_label": str(raw_label),
                "aasm_stage": str(stage),
                "is_n3": bool(is_n3),
                "global_qc_status": qc_status,
                "rejection_reasons": ";".join(
                    bundle.rejection_reasons.get(epoch_index, ())
                ),
                "psd_valid": bool(
                    row is not None
                    and np.isfinite(row["relative_so_power"])
                ),
                "fooof_valid": bool(
                    row is not None
                    and np.isfinite(row["fooof_aperiodic_exponent"])
                ),
                "so_detector_valid": bool(
                    row is not None and bool(row["detector_valid"])
                ),
                "ibi_valid": bool(
                    row is not None and bool(row["ibi_valid"])
                ),
                "so_proxy_valid": bool(
                    row is not None and int(row["so_proxy_cycle_count"]) > 0
                ),
                "waveform_valid": bool(
                    row is not None and int(row["waveform_event_count"]) > 0
                ),
                "spindle_detector_valid": bool(
                    row is not None and bool(row["spindle_detector_valid"])
                ),
                "spindle_duration_valid": bool(
                    row is not None
                    and np.isfinite(row["spindle_mean_duration_s"])
                ),
                "event_pairing_valid": bool(
                    row is not None and bool(row["pairing_valid"])
                ),
                "so_event_count": (
                    int(row["so_event_count"]) if row is not None else 0
                ),
                "ibi_interval_count": (
                    int(row["ibi_interval_count"]) if row is not None else 0
                ),
                "so_proxy_cycle_count": (
                    int(row["so_proxy_cycle_count"]) if row is not None else 0
                ),
                "spindle_event_count": (
                    int(row["spindle_event_count"]) if row is not None else 0
                ),
                "paired_event_count": (
                    int(row["paired_event_count"]) if row is not None else 0
                ),
            }
        )
    ledger = pd.DataFrame(records)
    if retained & rejected:
        raise RuntimeError("retained and rejected N3 sets overlap")
    if retained | rejected != n3:
        raise RuntimeError("retained + rejected does not equal all N3 epochs")
    rejected_ledger = ledger["global_qc_status"].eq("rejected")
    if (ledger.loc[rejected_ledger, "rejection_reasons"] == "").any():
        raise RuntimeError("a rejected N3 epoch lacks a rejection reason")
    if (
        ledger.loc[~ledger["is_n3"], "global_qc_status"]
        != "not_applicable_non_n3"
    ).any():
        raise RuntimeError("a non-N3 epoch was incorrectly treated as QC rejected")
    return ledger


def _summary_record(
    definition: Mapping[str, Any],
    *,
    value: float,
    valid_epoch_count: int,
    valid_event_count: int | None,
    validity_status: str,
    warnings: Iterable[str],
) -> dict[str, Any]:
    return {
        "field_name": definition["field_name"],
        "value": float(value),
        "unit": definition["unit"],
        "frequency_band": definition["frequency_band"],
        "algorithm": definition["algorithm"],
        "aggregation_method": definition["aggregation_method"],
        "valid_epoch_count": int(valid_epoch_count),
        "valid_event_count": (
            int(valid_event_count) if valid_event_count is not None else np.nan
        ),
        "validity_status": str(validity_status),
        "intended_role": definition["intended_role"],
        "dependency_or_redundancy": definition["dependency_or_redundancy"],
        "warnings": " | ".join(str(warning) for warning in warnings if warning),
    }


def _old_metric(bundle: ObservationBundle, name: str) -> float:
    metric = bundle.summaries.get(name)
    return float(metric.value) if metric is not None else float("nan")


def build_core_summary(
    bundle: ObservationBundle,
    per_epoch: pd.DataFrame,
    cycles: pd.DataFrame,
    paired_events: pd.DataFrame,
    config: PyloricInspiredConfig,
) -> pd.DataFrame:
    """Aggregate the frozen 18D feature library without filling invalid values."""
    definitions = {
        definition["field_name"]: definition
        for definition in FEATURE_DEFINITIONS
    }
    retained_count = len(bundle.retained_epoch_indices)
    so = bundle.diagnostics["slow_oscillation"]
    spindle = bundle.diagnostics["spindle"]
    valid_cycles = cycles.loc[cycles["valid"]]
    pairing_supported = (
        len(paired_events) >= config.minimum_global_paired_events
    )
    global_circular = (
        _circular_statistics(paired_events["phase_rad"])
        if pairing_supported
        else {
            "mean_angle_rad": np.nan,
            "mean_cos": np.nan,
            "mean_sin": np.nan,
            "concentration": np.nan,
        }
    )
    spindle_valid = per_epoch["spindle_detector_valid"].astype(bool)
    valid_spindle_duration = _finite(
        per_epoch.loc[spindle_valid, "spindle_mean_duration_s"]
    )
    total_spindle_union_s = float(
        np.nansum(
            per_epoch.loc[spindle_valid, "spindle_occupancy"]
            * bundle.epoch_duration_s
        )
    )
    total_spindle_valid_s = float(spindle_valid.sum() * bundle.epoch_duration_s)
    spindle_occupancy = (
        total_spindle_union_s / total_spindle_valid_s
        if total_spindle_valid_s > 0
        else np.nan
    )
    median_ibi = _hierarchical_median(per_epoch, "so_median_ibi_s")
    up_duration = _hierarchical_median(
        valid_cycles, "so_up_proxy_duration_s"
    )
    down_duration = _hierarchical_median(
        valid_cycles, "so_down_proxy_duration_s"
    )
    duty_cycle = _hierarchical_median(
        valid_cycles, "so_up_proxy_duty_cycle"
    )
    trough_to_peak = _hierarchical_median(
        valid_cycles, "so_trough_to_peak_time_s"
    )
    waveform_valid_count = int(
        np.isfinite(per_epoch["waveform_peak_to_peak_z"]).sum()
    )
    so_event_count = int(
        sum(int(row["event_count"]) for row in so["per_epoch"])
    )
    spindle_event_count = int(len(spindle["events"]))
    paired_epoch_count = int(paired_events["epoch_index"].nunique())
    inference_warning = "Candidate only - not yet frozen into fitting."
    held_warning = (
        "Held-out predictive-check candidate; no formal posterior estimator yet."
    )
    proxy_warning = (
        "Observable-level Fpz-Cz polarity proxy; not a model cortical "
        "membrane-potential UP/DOWN state."
    )
    spindle_warning = (
        "Provisional real-EEG observable detector; not V8a internal T13 and "
        "not yet cross-signal calibrated."
    )
    pairing_warning = (
        "Event-conditioned onset timing; distinct from continuous Tort PAC."
    )

    value_specs: dict[str, tuple[float, int, int | None, str, list[str]]] = {
        "fooof_aperiodic_exponent": (
            _old_metric(bundle, "fooof_aperiodic_exponent"),
            int(np.isfinite(per_epoch["fooof_aperiodic_exponent"]).sum()),
            None,
            "valid",
            [inference_warning],
        ),
        "relative_so_power": (
            _old_metric(bundle, "relative_so_power"),
            retained_count,
            None,
            "valid",
            [inference_warning],
        ),
        "so_peak_frequency_hz": (
            _old_metric(bundle, "so_peak_frequency_hz"),
            retained_count,
            None,
            "valid",
            [inference_warning, "Frequency-bin resolution is 0.25 Hz."],
        ),
        "so_q": (
            _old_metric(bundle, "so_q"),
            retained_count,
            None,
            "valid",
            [inference_warning],
        ),
        "so_event_rate_per_min": (
            _old_metric(bundle, "so_event_rate_per_min"),
            int(np.sum(so["validity_mask"])),
            so_event_count,
            "valid",
            [inference_warning],
        ),
        "so_median_ibi_s": (
            median_ibi[0],
            median_ibi[1],
            int(np.nansum(per_epoch["ibi_interval_count"])),
            "valid" if np.isfinite(median_ibi[0]) else "invalid",
            [
                inference_warning,
                "IBI is computed only within each 30-second epoch.",
            ],
        ),
        "ibi_cv": (
            _old_metric(bundle, "ibi_cv"),
            int(np.sum(so["ibi_validity_mask"])),
            int(len(so["ibi_s"])),
            "valid" if np.isfinite(_old_metric(bundle, "ibi_cv")) else "invalid",
            [
                inference_warning,
                "Requires at least three SO events per contributing epoch; "
                "within-epoch intervals are pooled only after that validity check.",
            ],
        ),
        "so_up_proxy_duration_s": (
            up_duration[0],
            up_duration[1],
            up_duration[2],
            "valid" if np.isfinite(up_duration[0]) else "invalid",
            [inference_warning, proxy_warning],
        ),
        "so_down_proxy_duration_s": (
            down_duration[0],
            down_duration[1],
            down_duration[2],
            "valid" if np.isfinite(down_duration[0]) else "invalid",
            [inference_warning, proxy_warning],
        ),
        "so_up_proxy_duty_cycle": (
            duty_cycle[0],
            duty_cycle[1],
            duty_cycle[2],
            "valid" if np.isfinite(duty_cycle[0]) else "invalid",
            [held_warning, proxy_warning, "Derived from the two proxy durations."],
        ),
        "waveform_peak_to_peak_z": (
            _old_metric(bundle, "waveform_peak_to_peak_z"),
            waveform_valid_count,
            int(np.nansum(per_epoch["waveform_event_count"])),
            (
                "valid"
                if np.isfinite(_old_metric(bundle, "waveform_peak_to_peak_z"))
                else "invalid"
            ),
            [held_warning],
        ),
        "so_trough_to_peak_time_s": (
            trough_to_peak[0],
            trough_to_peak[1],
            trough_to_peak[2],
            "valid" if np.isfinite(trough_to_peak[0]) else "invalid",
            [held_warning, proxy_warning],
        ),
        "spindle_density_per_min": (
            _old_metric(bundle, "spindle_density_per_min"),
            int(spindle_valid.sum()),
            spindle_event_count,
            (
                "provisional"
                if np.isfinite(_old_metric(bundle, "spindle_density_per_min"))
                else "invalid"
            ),
            [held_warning, spindle_warning],
        ),
        "spindle_mean_duration_s": (
            _old_metric(bundle, "spindle_mean_duration_s"),
            int(
                np.isfinite(per_epoch["spindle_mean_duration_s"]).sum()
            ),
            spindle_event_count,
            (
                "provisional"
                if len(valid_spindle_duration)
                else "invalid"
            ),
            [
                held_warning,
                spindle_warning,
                "Zero-event epochs have undefined duration, never duration zero.",
            ],
        ),
        "spindle_occupancy": (
            spindle_occupancy,
            int(spindle_valid.sum()),
            spindle_event_count,
            "provisional" if np.isfinite(spindle_occupancy) else "invalid",
            [
                held_warning,
                spindle_warning,
                "Computed from the union of intervals to prevent double counting.",
            ],
        ),
        "spindle_onset_phase_cos": (
            global_circular["mean_cos"],
            paired_epoch_count,
            len(paired_events),
            "provisional" if pairing_supported else "invalid",
            [held_warning, pairing_warning],
        ),
        "spindle_onset_phase_sin": (
            global_circular["mean_sin"],
            paired_epoch_count,
            len(paired_events),
            "provisional" if pairing_supported else "invalid",
            [held_warning, pairing_warning],
        ),
        "spindle_onset_phase_concentration": (
            global_circular["concentration"],
            paired_epoch_count,
            len(paired_events),
            "provisional" if pairing_supported else "invalid",
            [
                held_warning,
                pairing_warning,
                (
                    ""
                    if pairing_supported
                    else (
                        f"Fewer than {config.minimum_global_paired_events} "
                        "paired events; circular statistics are undefined."
                    )
                ),
            ],
        ),
    }
    rows = []
    for field_name in CORE_FIELDS:
        value, epochs, events, status, warnings = value_specs[field_name]
        if not np.isfinite(value) and status != "invalid":
            status = "invalid"
            warnings = [*warnings, "Aggregate value is undefined."]
        rows.append(
            _summary_record(
                definitions[field_name],
                value=value,
                valid_epoch_count=epochs,
                valid_event_count=events,
                validity_status=status,
                warnings=warnings,
            )
        )
    summary = pd.DataFrame(rows)
    if tuple(summary["field_name"]) != CORE_FIELDS:
        raise RuntimeError("core summary is not the frozen 18D schema")
    return summary


def build_auxiliary_diagnostics(
    bundle: ObservationBundle,
    paired_events: pd.DataFrame,
) -> pd.DataFrame:
    """Keep continuous PAC and simulator-internal quantities outside the 18D core."""
    circular = (
        _circular_statistics(paired_events["phase_rad"])
        if len(paired_events)
        else _circular_statistics([])
    )
    rows = [
        {
            "field_name": "continuous_pac_mi",
            "value": _old_metric(bundle, "pac_mi"),
            "unit": "1",
            "intended_role": "mechanism_diagnostic",
            "observability": "single-channel EEG auxiliary diagnostic",
            "warning": (
                "Continuous Tort PAC is not event-conditioned onset timing; "
                "weak MI makes preferred phase physiologically unstable."
            ),
        },
        {
            "field_name": "continuous_pac_preferred_phase_rad",
            "value": _old_metric(bundle, "pac_preferred_phase_rad"),
            "unit": "rad",
            "intended_role": "mechanism_diagnostic",
            "observability": "single-channel EEG auxiliary diagnostic",
            "warning": "Not included in the 18D core; sin/cos/onset phase are distinct.",
        },
        {
            "field_name": "continuous_pac_up_down_ratio",
            "value": _old_metric(bundle, "pac_up_down_ratio"),
            "unit": "1",
            "intended_role": "mechanism_diagnostic",
            "observability": "single-channel EEG auxiliary diagnostic",
            "warning": (
                "Semantic field name only. Legacy T11_lag_ms is a misnomer; "
                "do not interpret this ratio as a time lag."
            ),
        },
        {
            "field_name": "event_conditioned_mean_phase_rad",
            "value": circular["mean_angle_rad"],
            "unit": "rad",
            "intended_role": "mechanism_diagnostic",
            "observability": "single-channel EEG event-conditioned auxiliary",
            "warning": (
                "Auxiliary angle only; excluded from the 18D core to avoid "
                "duplicating its sine and cosine coordinates."
            ),
        },
    ]
    for name, explanation in (
        ("T9_model_internal_pac", "model-side coupling quantity"),
        ("T10_model_internal_phase", "model-side phase quantity"),
        ("T11_model_directional_lag", "model-side directional timing quantity"),
        ("T12_thalamic_spindle", "model-side thalamic spindle detector"),
        ("T13_cortex_visible_spindle", "V8a model audit detector"),
    ):
        rows.append(
            {
                "field_name": name,
                "value": np.nan,
                "unit": "not_observable",
                "intended_role": "mechanism_diagnostic",
                "observability": "not observable from single-channel Fpz-Cz EEG",
                "warning": (
                    f"{explanation}; no EEG value is fabricated and no direct "
                    "observable mapping is assumed."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_old_vs_new_regression(
    bundle: ObservationBundle,
    summary: pd.DataFrame,
    auxiliary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare all unchanged overlapping metrics to the existing Observation."""
    new_values = dict(zip(summary["field_name"], summary["value"]))
    new_values.update(dict(zip(auxiliary["field_name"], auxiliary["value"])))
    comparisons = (
        (
            "fooof_aperiodic_exponent",
            "fooof_aperiodic_exponent",
            1e-10,
            "exact upstream aggregate reuse",
        ),
        (
            "relative_so_power",
            "relative_so_power",
            1e-10,
            "exact upstream aggregate reuse",
        ),
        (
            "so_peak_frequency_hz",
            "so_peak_frequency_hz",
            1e-10,
            "exact upstream aggregate reuse",
        ),
        ("so_q", "so_q", 1e-10, "exact upstream aggregate reuse"),
        (
            "so_event_rate_per_min",
            "so_event_rate_per_min",
            1e-10,
            "exact upstream detector and aggregate reuse",
        ),
        (
            "ibi_cv",
            "ibi_cv",
            1e-10,
            "exact upstream pooled-within-epoch definition retained",
        ),
        (
            "waveform_peak_to_peak_z",
            "waveform_peak_to_peak_z",
            1e-10,
            "exact upstream aggregate waveform reuse",
        ),
        (
            "spindle_density_per_min",
            "spindle_density_per_min",
            1e-10,
            "exact upstream detector and aggregate reuse",
        ),
        (
            "spindle_mean_duration_s",
            "spindle_mean_duration_s",
            1e-10,
            "exact upstream event-duration aggregate reuse",
        ),
        (
            "pac_mi",
            "continuous_pac_mi",
            1e-10,
            "retained outside 18D as an unchanged auxiliary diagnostic",
        ),
        (
            "pac_preferred_phase_rad",
            "continuous_pac_preferred_phase_rad",
            1e-10,
            "retained outside 18D as an unchanged auxiliary diagnostic",
        ),
        (
            "pac_up_down_ratio",
            "continuous_pac_up_down_ratio",
            1e-10,
            "retained outside 18D; never interpreted as legacy lag",
        ),
    )
    rows = []
    for old_name, new_name, tolerance, reason in comparisons:
        old_value = _old_metric(bundle, old_name)
        new_value = float(new_values.get(new_name, np.nan))
        absolute = (
            abs(new_value - old_value)
            if np.isfinite(old_value) and np.isfinite(new_value)
            else np.nan
        )
        relative = (
            absolute / abs(old_value)
            if np.isfinite(absolute) and abs(old_value) > 1e-15
            else (0.0 if absolute == 0.0 else np.nan)
        )
        if np.isfinite(absolute):
            status = "pass" if absolute <= tolerance else "fail"
        elif np.isnan(old_value) and np.isnan(new_value):
            status = "warning"
        else:
            status = "fail"
        rows.append(
            {
                "old_field_name": old_name,
                "new_field_name": new_name,
                "old_value": old_value,
                "new_value": new_value,
                "absolute_difference": absolute,
                "relative_difference": relative,
                "tolerance": tolerance,
                "status": status,
                "difference_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _pairwise_valid_count(frame: pd.DataFrame) -> pd.DataFrame:
    valid = np.isfinite(frame.to_numpy(dtype=float)).astype(int)
    counts = valid.T @ valid
    return pd.DataFrame(counts, index=frame.columns, columns=frame.columns)


def build_stability_audit(
    per_epoch: pd.DataFrame,
    feature_definitions_frame: pd.DataFrame,
    config: PyloricInspiredConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quantify missingness, spread, outlier sensitivity, and redundancy."""
    features = per_epoch.loc[:, list(CORE_FIELDS)].astype(float)
    correlation = features.corr(method="spearman", min_periods=2)
    pair_counts = _pairwise_valid_count(features)
    roles = feature_definitions_frame.set_index("field_name")[
        "intended_role"
    ].to_dict()
    rows = []
    for field_name in CORE_FIELDS:
        values = _finite(features[field_name])
        missing_fraction = 1.0 - len(values) / len(features)
        if len(values):
            median = float(np.median(values))
            q25, q75 = np.percentile(values, [25.0, 75.0])
            iqr = float(q75 - q25)
            mean = float(np.mean(values))
            outlier_sensitivity = (
                abs(mean - median) / iqr
                if iqr > config.near_constant_iqr
                else np.nan
            )
        else:
            median = q25 = q75 = iqr = outlier_sensitivity = np.nan
        near_constant = bool(
            len(values) > 0
            and np.isfinite(iqr)
            and iqr <= config.near_constant_iqr
        )
        redundant_with: list[str] = []
        for other in CORE_FIELDS:
            if other == field_name:
                continue
            rho = correlation.loc[field_name, other]
            n_pair = int(pair_counts.loc[field_name, other])
            if (
                np.isfinite(rho)
                and abs(rho) >= config.redundancy_abs_spearman
                and n_pair >= config.minimum_pair_count
            ):
                redundant_with.append(f"{other} (rho={rho:.2f}, n={n_pair})")
        if missing_fraction > 0.5:
            recommendation = "insufficient event support"
        elif near_constant:
            recommendation = "unstable: near-constant across epochs"
        elif field_name.startswith("spindle_"):
            recommendation = "requires detector validation; keep held out"
        elif field_name.startswith("so_up_proxy") or field_name.startswith(
            "so_down_proxy"
        ):
            recommendation = (
                "suitable as observable proxy candidate; mapping not frozen"
            )
        elif redundant_with:
            recommendation = "redundant; do not treat as independent evidence"
        elif roles[field_name] == "inference_summary_candidate":
            recommendation = "suitable for inference candidacy audit"
        else:
            recommendation = "keep held out"
        rows.append(
            {
                "field_name": field_name,
                "finite_epoch_count": int(len(values)),
                "missing_fraction": float(missing_fraction),
                "median": median,
                "q25": q25,
                "q75": q75,
                "iqr": iqr,
                "robust_outlier_sensitivity": outlier_sensitivity,
                "near_constant": near_constant,
                "redundant_with": "; ".join(redundant_with),
                "recommendation": recommendation,
            }
        )
    return pd.DataFrame(rows), correlation, pair_counts


def build_validation_checks(
    bundle: ObservationBundle,
    summary: pd.DataFrame,
    per_epoch: pd.DataFrame,
    ledger: pd.DataFrame,
    regression: pd.DataFrame,
    config: PyloricInspiredConfig,
) -> pd.DataFrame:
    """Return explicit pass/fail checks used by tests and the notebook."""
    value_by_name = summary.set_index("field_name")["value"]
    finite_summary = summary["value"].replace([np.inf, -np.inf], np.nan)
    bounded_fields = {
        "so_up_proxy_duty_cycle": (0.0, 1.0),
        "spindle_occupancy": (0.0, 1.0),
        "spindle_onset_phase_concentration": (0.0, 1.0),
    }
    checks: list[tuple[str, bool, str]] = [
        (
            "schema_has_exactly_18_unique_fields",
            len(summary) == 18
            and summary["field_name"].is_unique
            and tuple(summary["field_name"]) == CORE_FIELDS,
            f"rows={len(summary)}, unique={summary['field_name'].nunique()}",
        ),
        (
            "raw_phase_not_in_core_18d",
            "spindle_onset_phase_rad" not in set(summary["field_name"]),
            "phase is represented by sin/cos; angle is auxiliary only",
        ),
        (
            "retained_plus_rejected_equals_all_n3",
            len(bundle.retained_epoch_indices)
            + len(bundle.rejected_epoch_indices)
            == len(bundle.n3_epoch_indices),
            (
                f"{len(bundle.retained_epoch_indices)} + "
                f"{len(bundle.rejected_epoch_indices)} = "
                f"{len(bundle.n3_epoch_indices)}"
            ),
        ),
        (
            "retained_and_rejected_are_disjoint",
            set(bundle.retained_epoch_indices).isdisjoint(
                set(bundle.rejected_epoch_indices)
            ),
            "N3 destination sets are mutually exclusive",
        ),
        (
            "every_rejected_epoch_has_reason",
            all(
                bool(bundle.rejection_reasons[int(epoch)])
                for epoch in bundle.rejected_epoch_indices
            ),
            f"rejected={len(bundle.rejected_epoch_indices)}",
        ),
        (
            "non_n3_not_counted_as_qc_rejection",
            (
                ledger.loc[~ledger["is_n3"], "global_qc_status"]
                == "not_applicable_non_n3"
            ).all(),
            "QC rejection is an N3-only decision",
        ),
        (
            "per_epoch_rows_equal_retained_n3",
            len(per_epoch) == len(bundle.retained_epoch_indices),
            f"rows={len(per_epoch)}",
        ),
        (
            "no_infinite_core_values",
            not np.isinf(summary["value"].to_numpy(dtype=float)).any(),
            f"undefined={int(finite_summary.isna().sum())}",
        ),
        (
            "overlap_regression_has_no_failures",
            not regression["status"].eq("fail").any(),
            regression["status"].value_counts().to_dict().__str__(),
        ),
        (
            "minimum_paired_event_rule_recorded",
            config.minimum_global_paired_events > 0
            and config.minimum_per_epoch_paired_events > 1,
            (
                f"global={config.minimum_global_paired_events}, "
                f"per_epoch={config.minimum_per_epoch_paired_events}"
            ),
        ),
    ]
    for field_name, (lower, upper) in bounded_fields.items():
        value = float(value_by_name[field_name])
        valid = np.isnan(value) or (lower <= value <= upper)
        checks.append(
            (
                f"{field_name}_within_0_1_or_undefined",
                bool(valid),
                f"value={value}",
            )
        )
    for field_name in (
        "so_up_proxy_duration_s",
        "so_down_proxy_duration_s",
        "so_trough_to_peak_time_s",
        "spindle_mean_duration_s",
    ):
        value = float(value_by_name[field_name])
        checks.append(
            (
                f"{field_name}_positive_or_undefined",
                bool(np.isnan(value) or value > 0.0),
                f"value={value}",
            )
        )
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


def build_pyloric_inspired_observation(
    bundle: ObservationBundle,
    observation_config: ObservationConfig | None = None,
    config: PyloricInspiredConfig | None = None,
) -> PyloricInspiredResult:
    """Build the full real-EEG Pyloric-inspired observation audit."""
    observation_config = observation_config or load_observation_config()
    config = config or load_pyloric_inspired_config()
    np.random.seed(config.random_seed)
    definitions = feature_dictionary()
    cycles = extract_so_proxy_cycles(bundle, config)
    spindle_features, spindle_union_intervals = spindle_epoch_features(bundle)
    paired_events, pairing_features = pair_spindle_onsets_to_so_cycles(
        bundle, config
    )
    per_epoch = build_per_epoch_features(
        bundle,
        observation_config,
        config,
        cycles,
        spindle_features,
        pairing_features,
    )
    summary = build_core_summary(
        bundle, per_epoch, cycles, paired_events, config
    )
    ledger = build_epoch_ledger(bundle, per_epoch)
    auxiliary = build_auxiliary_diagnostics(bundle, paired_events)
    regression = build_old_vs_new_regression(bundle, summary, auxiliary)
    stability, correlation, pair_count = build_stability_audit(
        per_epoch, definitions, config
    )
    checks = build_validation_checks(
        bundle,
        summary,
        per_epoch,
        ledger,
        regression,
        config,
    )
    result = PyloricInspiredResult(
        summary=summary,
        per_epoch_features=per_epoch,
        epoch_ledger=ledger,
        feature_dictionary=definitions,
        auxiliary_diagnostics=auxiliary,
        old_vs_new_regression=regression,
        stability_audit=stability,
        spearman_correlation=correlation,
        valid_pair_count=pair_count,
        validation_checks=checks,
        diagnostics={
            "so_proxy_cycles": cycles,
            "spindle_union_intervals": spindle_union_intervals,
            "paired_events": paired_events,
            "config": config,
        },
    )
    if not result.validation_checks["passed"].all():
        failed = result.validation_checks.loc[
            ~result.validation_checks["passed"], ["check", "detail"]
        ]
        raise RuntimeError(
            "Pyloric-inspired validation failed:\n"
            + failed.to_string(index=False)
        )
    return result


OBSERVATION_COLOR = "#2A6F97"
RETAINED_COLOR = "#2D936C"
REJECTED_COLOR = "#C14953"
SO_COLOR = "#6A4C93"
SPINDLE_COLOR = "#E09F3E"
NEUTRAL_COLOR = "#626262"
ROLE_COLORS = {
    "inference_summary_candidate": "#2A6F97",
    "held_out_ppc_candidate": "#E09F3E",
    "mechanism_diagnostic": "#777777",
}


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "legend.fontsize": 8,
            "figure.dpi": 110,
            "savefig.bbox": "tight",
        }
    )


def plot_concept_mapping(
    feature_definitions_frame: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Show the conceptual translation, not a literal pyloric 18D copy."""
    _plot_style()
    mapping = (
        feature_definitions_frame.groupby(["domain", "pyloric_analogy"])
        .agg(
            eeg_fields=("field_name", lambda values: "\n".join(values)),
            count=("field_name", "size"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.axis("off")
    table = ax.table(
        cellText=mapping[
            ["domain", "pyloric_analogy", "eeg_fields", "count"]
        ].values,
        colLabels=[
            "EEG feature domain",
            "Pyloric-inspired concept",
            "SC4001 observable fields",
            "n",
        ],
        colWidths=[0.19, 0.20, 0.52, 0.05],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.65)
    ax.set_title(
        "Pyloric-inspired observation engineering: concept mapping, not "
        "mechanical feature replication",
        pad=12,
    )
    fig.tight_layout()
    return fig, {
        "figure_id": "concept_mapping",
        "mapping_rows": mapping.to_dict(orient="records"),
        "warning": (
            "Pyloric timing concepts motivate EEG observables; simulator-internal "
            "states are not equated to scalp EEG proxies."
        ),
    }


def _representative_index(values: np.ndarray) -> int:
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        raise RuntimeError("no finite candidates for representative selection")
    median = float(np.median(values[finite]))
    return int(finite[np.argmin(np.abs(values[finite] - median))])


def plot_qc_audit(
    bundle: ObservationBundle,
    result: PyloricInspiredResult,
    observation_config: ObservationConfig,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot epoch accounting, PTP QC, reasons, waveforms, and metric support."""
    _plot_style()
    qc = pd.DataFrame(bundle.diagnostics["qc_rows"])
    reason_counts = Counter(
        reason
        for reasons in bundle.rejection_reasons.values()
        for reason in reasons
    )
    _, all_segments, _, _, _, _ = _load_epoch_data(observation_config)
    retained_qc = qc.loc[qc["retained"]].reset_index(drop=True)
    rejected_qc = qc.loc[~qc["retained"]].reset_index(drop=True)
    retained_pick = int(
        retained_qc.iloc[
            _representative_index(
                retained_qc["peak_to_peak_uv"].to_numpy(float)
            )
        ]["epoch_index"]
    )
    rejected_pick = (
        int(
            rejected_qc.iloc[
                _representative_index(
                    rejected_qc["peak_to_peak_uv"].to_numpy(float)
                )
            ]["epoch_index"]
        )
        if len(rejected_qc)
        else None
    )
    metric_columns = [
        "psd_valid",
        "fooof_valid",
        "so_detector_valid",
        "ibi_valid",
        "so_proxy_valid",
        "waveform_valid",
        "spindle_detector_valid",
        "spindle_duration_valid",
        "event_pairing_valid",
    ]
    support = result.epoch_ledger.loc[
        result.epoch_ledger["global_qc_status"].eq("retained"), metric_columns
    ].sum()
    time_s = np.arange(all_segments.shape[1]) / bundle.fs_hz
    fig = plt.figure(figsize=(12.0, 9.0))
    grid = fig.add_gridspec(3, 2)
    ax_count = fig.add_subplot(grid[0, 0])
    ax_ptp = fig.add_subplot(grid[0, 1])
    ax_reason = fig.add_subplot(grid[1, 0])
    ax_support = fig.add_subplot(grid[1, 1])
    ax_wave = fig.add_subplot(grid[2, :])
    count_labels = ["aligned", "N3", "retained", "rejected"]
    counts = [
        len(bundle.diagnostics["mapped_stages"]),
        len(bundle.n3_epoch_indices),
        len(bundle.retained_epoch_indices),
        len(bundle.rejected_epoch_indices),
    ]
    ax_count.bar(
        count_labels,
        counts,
        color=[NEUTRAL_COLOR, SO_COLOR, RETAINED_COLOR, REJECTED_COLOR],
    )
    ax_count.set(title="Epoch accounting", ylabel="30-second epochs")
    for x, count in enumerate(counts):
        ax_count.text(x, count, str(count), ha="center", va="bottom")
    ax_ptp.hist(
        retained_qc["peak_to_peak_uv"],
        bins=30,
        alpha=0.72,
        color=RETAINED_COLOR,
        label=f"retained (n={len(retained_qc)})",
    )
    if len(rejected_qc):
        ax_ptp.hist(
            rejected_qc["peak_to_peak_uv"],
            bins=30,
            alpha=0.65,
            color=REJECTED_COLOR,
            label=f"rejected (n={len(rejected_qc)})",
        )
    ax_ptp.axvline(
        observation_config.max_peak_to_peak_uv,
        color="black",
        linestyle="--",
        label=f"QC threshold={observation_config.max_peak_to_peak_uv:g} uV",
    )
    ax_ptp.set(
        title="N3 peak-to-peak QC",
        xlabel="Peak-to-peak amplitude (uV)",
        ylabel="Epoch count",
    )
    ax_ptp.legend()
    ax_reason.barh(
        list(reason_counts) or ["none"],
        list(reason_counts.values()) or [0],
        color=REJECTED_COLOR,
    )
    ax_reason.set(title="N3 rejection reasons", xlabel="Epoch count")
    ax_support.barh(
        [label.replace("_valid", "") for label in support.index],
        support.values,
        color=OBSERVATION_COLOR,
    )
    ax_support.axvline(
        len(bundle.retained_epoch_indices), color="black", linestyle=":"
    )
    ax_support.set(
        title="Metric-specific valid epoch support",
        xlabel="Retained N3 epochs",
    )
    ax_wave.plot(
        time_s,
        all_segments[retained_pick],
        color=RETAINED_COLOR,
        label=f"retained epoch {retained_pick}",
    )
    if rejected_pick is not None:
        ax_wave.plot(
            time_s,
            all_segments[rejected_pick],
            color=REJECTED_COLOR,
            alpha=0.8,
            label=f"rejected epoch {rejected_pick}",
        )
    ax_wave.set(
        title="Deterministic median-PTP retained/rejected N3 examples",
        xlabel="Time within native epoch (s)",
        ylabel=f"{bundle.channel} (uV)",
        xlim=(0.0, bundle.epoch_duration_s),
    )
    ax_wave.legend()
    fig.suptitle(
        f"{bundle.subject_id} observation QC | non-N3 epochs are not QC rejects",
        y=1.01,
    )
    fig.tight_layout()
    return fig, {
        "figure_id": "epoch_qc_audit",
        "epoch_counts": dict(zip(count_labels, map(int, counts))),
        "rejection_reason_counts": dict(reason_counts),
        "metric_support": {key: int(value) for key, value in support.items()},
        "representative_retained_epoch": retained_pick,
        "representative_rejected_epoch": rejected_pick,
        "qc_threshold_uv": observation_config.max_peak_to_peak_uv,
    }


def plot_feature_distributions(
    result: PyloricInspiredResult,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot all per-epoch feature distributions using native units."""
    _plot_style()
    fig, axes = plt.subplots(5, 4, figsize=(13.0, 13.5))
    units = result.feature_dictionary.set_index("field_name")["unit"]
    for axis, field_name in zip(axes.flat, CORE_FIELDS):
        values = _finite(result.per_epoch_features[field_name])
        if len(values):
            axis.hist(values, bins=min(24, max(6, int(np.sqrt(len(values))))),
                      color=OBSERVATION_COLOR, alpha=0.82)
            axis.axvline(np.median(values), color="black", linestyle="--")
        else:
            axis.text(0.5, 0.5, "undefined", ha="center", va="center")
        axis.set_title(field_name.replace("_", " "), fontsize=8)
        axis.set_xlabel(str(units[field_name]))
        axis.set_ylabel(f"epochs (finite n={len(values)})")
    for axis in axes.flat[len(CORE_FIELDS) :]:
        axis.axis("off")
    fig.suptitle(
        "Pyloric-inspired 18D per-epoch distributions in native units",
        y=1.0,
    )
    fig.tight_layout()
    return fig, {
        "figure_id": "feature_distributions",
        "finite_epoch_counts": {
            field: int(
                np.isfinite(result.per_epoch_features[field].to_numpy(float)).sum()
            )
            for field in CORE_FIELDS
        },
        "warning": "Panels use native units; distributions are not magnitude-normalized.",
    }


def plot_robust_variability(
    result: PyloricInspiredResult,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot robust-scaled cross-epoch variability without magnitude claims."""
    _plot_style()
    normalized: list[np.ndarray] = []
    labels: list[str] = []
    for field_name in CORE_FIELDS:
        values = _finite(result.per_epoch_features[field_name])
        if not len(values):
            continue
        median = float(np.median(values))
        iqr = float(np.subtract(*np.percentile(values, [75, 25])))
        if iqr <= 1e-12:
            continue
        normalized.append((values - median) / iqr)
        labels.append(field_name)
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.boxplot(
        normalized,
        tick_labels=labels,
        showfliers=False,
        orientation="vertical",
    )
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.tick_params(axis="x", rotation=72, labelsize=7)
    ax.set(
        title="Robust-normalized cross-epoch variability",
        ylabel="(value - median) / IQR",
    )
    ax.text(
        0.01,
        0.99,
        "Scale is feature-specific; do not compare absolute physiological strength.",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    return fig, {
        "figure_id": "robust_variability",
        "included_fields": labels,
        "normalization": "per feature: (value - median) / IQR",
        "warning": "Normalized magnitudes are not absolute physiological strengths.",
    }


def plot_redundancy_heatmap(
    result: PyloricInspiredResult,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Show Spearman rho and valid pair counts side by side."""
    _plot_style()
    correlation = result.spearman_correlation.loc[
        list(CORE_FIELDS), list(CORE_FIELDS)
    ]
    pair_count = result.valid_pair_count.loc[
        list(CORE_FIELDS), list(CORE_FIELDS)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.8))
    image = axes[0].imshow(
        correlation.to_numpy(float),
        cmap="coolwarm",
        norm=Normalize(vmin=-1.0, vmax=1.0),
        interpolation="nearest",
    )
    fig.colorbar(image, ax=axes[0], fraction=0.046, label="Spearman rho")
    count_image = axes[1].imshow(
        pair_count.to_numpy(float), cmap="viridis", interpolation="nearest"
    )
    fig.colorbar(
        count_image, ax=axes[1], fraction=0.046, label="valid epoch pairs"
    )
    short = [field.replace("spindle_", "sp_").replace("so_", "") for field in CORE_FIELDS]
    for axis, title in zip(
        axes, ["Feature redundancy", "Pairwise valid support"]
    ):
        axis.set_xticks(range(18), short, rotation=90, fontsize=6)
        axis.set_yticks(range(18), short, fontsize=6)
        axis.set_title(title)
    fig.suptitle(
        "18D dependence audit: different names do not imply independent evidence"
    )
    fig.tight_layout()
    return fig, {
        "figure_id": "redundancy_heatmap",
        "spearman_method": "pairwise complete Spearman correlation",
        "structural_relations": [
            "SO rate approximately 60 / IBI",
            "UP-proxy duty = UP / (UP + DOWN)",
            "spindle occupancy approximately density * duration / 60",
        ],
    }


def _representative_valid_cycle(cycles: pd.DataFrame) -> pd.Series:
    valid = cycles.loc[cycles["valid"]].copy()
    if valid.empty:
        raise RuntimeError("no valid SO proxy cycle is available for plotting")
    duration = (
        valid["so_up_proxy_duration_s"] + valid["so_down_proxy_duration_s"]
    )
    return valid.loc[(duration - duration.median()).abs().idxmin()]


def plot_so_proxy_morphology(
    bundle: ObservationBundle,
    result: PyloricInspiredResult,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot a deterministic complete SO proxy cycle and aggregate morphology."""
    _plot_style()
    cycles = result.diagnostics["so_proxy_cycles"]
    cycle = _representative_valid_cycle(cycles)
    epoch_index = int(cycle["epoch_index"])
    row = int(np.flatnonzero(bundle.retained_epoch_indices == epoch_index)[0])
    signal_uv = bundle.diagnostics["slow_oscillation"]["filtered_uv"][row]
    left = float(cycle["negative_entry_sample"])
    cross = float(cycle["negative_to_positive_sample"])
    right = float(cycle["positive_exit_sample"])
    padding = int(round(0.5 * bundle.fs_hz))
    start = max(0, int(np.floor(left)) - padding)
    stop = min(len(signal_uv), int(np.ceil(right)) + padding)
    time_s = np.arange(start, stop) / bundle.fs_hz
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    axes[0].plot(time_s, signal_uv[start:stop], color=SO_COLOR)
    axes[0].axhline(0.0, color="black", linewidth=0.6)
    axes[0].axvspan(
        left / bundle.fs_hz,
        cross / bundle.fs_hz,
        color=REJECTED_COLOR,
        alpha=0.16,
        label="negative / DOWN-proxy",
    )
    axes[0].axvspan(
        cross / bundle.fs_hz,
        right / bundle.fs_hz,
        color=RETAINED_COLOR,
        alpha=0.16,
        label="positive / UP-proxy",
    )
    axes[0].scatter(
        [
            cycle["trough_sample"] / bundle.fs_hz,
            cycle["peak_sample"] / bundle.fs_hz,
        ],
        [
            signal_uv[int(cycle["trough_sample"])],
            signal_uv[int(cycle["peak_sample"])],
        ],
        color=[REJECTED_COLOR, RETAINED_COLOR],
        zorder=4,
    )
    axes[0].set(
        title=f"Complete SO proxy cycle | epoch {epoch_index}",
        xlabel="Time within native 30-second epoch (s)",
        ylabel=f"0.2-4 Hz {bundle.channel} (uV)",
    )
    axes[0].legend()
    so = bundle.diagnostics["slow_oscillation"]
    axes[1].plot(
        so["waveform_time_s"],
        so["waveform_mean_z"],
        color=SO_COLOR,
        label=f"mean (events n={len(so['waveform_snippets_z'])})",
    )
    axes[1].fill_between(
        so["waveform_time_s"],
        so["waveform_mean_z"] - so["waveform_sem_z"],
        so["waveform_mean_z"] + so["waveform_sem_z"],
        color=SO_COLOR,
        alpha=0.2,
        label="SEM",
    )
    axes[1].axvline(0.0, color="black", linestyle="--", label="trough")
    axes[1].set(
        title="Trough-aligned observable waveform morphology",
        xlabel="Time from detected trough (s)",
        ylabel="Per-event z-scored EEG",
    )
    axes[1].legend()
    fig.suptitle(
        "EEG polarity is an observable proxy, not neuronal membrane-potential state"
    )
    fig.tight_layout()
    invalid_reasons = Counter(
        cycles.loc[~cycles["valid"], "invalid_reason"].dropna()
    )
    return fig, {
        "figure_id": "so_proxy_morphology",
        "representative_epoch": epoch_index,
        "valid_cycle_count": int(cycles["valid"].sum()),
        "invalid_cycle_count": int((~cycles["valid"]).sum()),
        "invalid_reason_counts": dict(invalid_reasons),
        "polarity_convention": result.diagnostics["config"].polarity_convention,
        "up_proxy_duration_s": float(cycle["so_up_proxy_duration_s"]),
        "down_proxy_duration_s": float(cycle["so_down_proxy_duration_s"]),
    }


def plot_spindle_detection(
    bundle: ObservationBundle,
    result: PyloricInspiredResult,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot a deterministic event-containing epoch and its spindle detections."""
    _plot_style()
    spindle = bundle.diagnostics["spindle"]
    candidates = result.per_epoch_features.loc[
        result.per_epoch_features["spindle_event_count"] > 0
    ]
    if candidates.empty:
        raise RuntimeError("no spindle event is available for plotting")
    event_counts = candidates["spindle_event_count"].to_numpy(float)
    selected = candidates.iloc[
        np.argmin(np.abs(event_counts - np.median(event_counts)))
    ]
    epoch_index = int(selected["epoch_index"])
    row = int(np.flatnonzero(bundle.retained_epoch_indices == epoch_index)[0])
    events = [
        event
        for event in spindle["events"]
        if int(event["epoch_index"]) == epoch_index
    ]
    time_s = np.arange(bundle.segments.shape[1]) / bundle.fs_hz
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 7.2), sharex=True)
    axes[0].plot(time_s, bundle.segments[row], color=NEUTRAL_COLOR)
    axes[0].set(ylabel=f"{bundle.channel} (uV)", title=f"Epoch {epoch_index}")
    axes[1].plot(time_s, spindle["filtered_uv"][row], color=SPINDLE_COLOR)
    axes[1].set(ylabel="11-15 Hz (uV)")
    axes[2].plot(
        time_s, spindle["envelope_uv"][row], color=OBSERVATION_COLOR
    )
    axes[2].axhline(
        spindle["threshold_uv"][row],
        color=REJECTED_COLOR,
        linestyle="--",
        label=f"threshold={spindle['threshold_uv'][row]:.2f} uV",
    )
    for axis in axes:
        for event in events:
            axis.axvspan(
                event["start_sample"] / bundle.fs_hz,
                event["stop_sample"] / bundle.fs_hz,
                color=SPINDLE_COLOR,
                alpha=0.18,
            )
    axes[2].set(
        xlabel="Time within native 30-second epoch (s)",
        ylabel="RMS envelope (uV)",
        xlim=(0.0, bundle.epoch_duration_s),
    )
    axes[2].legend()
    fig.suptitle(
        "Provisional real-EEG observable spindle detector; not V8a T13"
    )
    fig.tight_layout()
    return fig, {
        "figure_id": "spindle_detection",
        "representative_epoch": epoch_index,
        "event_count": int(len(events)),
        "threshold_uv": float(spindle["threshold_uv"][row]),
        "parameters": spindle["parameters"],
        "status": spindle["status"],
    }


def plot_event_pairing(
    bundle: ObservationBundle,
    result: PyloricInspiredResult,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Show within-epoch spindle-onset to SO-cycle pairing."""
    _plot_style()
    pairs = result.diagnostics["paired_events"]
    if pairs.empty:
        raise RuntimeError("no spindle-onset/SO-cycle pair is available")
    counts = pairs.groupby("epoch_index").size()
    epoch_index = int((counts - counts.median()).abs().idxmin())
    epoch_pairs = pairs.loc[pairs["epoch_index"] == epoch_index]
    row = int(np.flatnonzero(bundle.retained_epoch_indices == epoch_index)[0])
    so_signal = bundle.diagnostics["slow_oscillation"]["filtered_uv"][row]
    spindle_envelope = bundle.diagnostics["spindle"]["envelope_uv"][row]
    time_s = np.arange(len(so_signal)) / bundle.fs_hz
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 6.0), sharex=True)
    axes[0].plot(time_s, so_signal, color=SO_COLOR)
    axes[1].plot(time_s, spindle_envelope, color=SPINDLE_COLOR)
    for _, pair in epoch_pairs.iterrows():
        left = pair["left_trough_sample"] / bundle.fs_hz
        right = pair["right_trough_sample"] / bundle.fs_hz
        onset = pair["spindle_onset_sample"] / bundle.fs_hz
        axes[0].axvspan(left, right, color=SO_COLOR, alpha=0.08)
        axes[0].axvline(left, color=NEUTRAL_COLOR, linestyle=":")
        for axis in axes:
            axis.axvline(onset, color=REJECTED_COLOR, linestyle="--")
        axes[0].text(
            onset,
            np.nanmax(so_signal) * 0.8,
            f"{pair['phase_rad']:.2f} rad",
            rotation=90,
            fontsize=7,
            va="top",
        )
    axes[0].set(
        title=f"Trough-to-trough cycles and spindle onsets | epoch {epoch_index}",
        ylabel="0.2-4 Hz EEG (uV)",
    )
    axes[1].set(
        xlabel="Time within native 30-second epoch (s)",
        ylabel="11-15 Hz RMS (uV)",
        xlim=(0.0, bundle.epoch_duration_s),
    )
    fig.suptitle(
        "Trough-anchored event pairing is computed within epoch only"
    )
    fig.tight_layout()
    return fig, {
        "figure_id": "event_pairing",
        "representative_epoch": epoch_index,
        "paired_event_count": int(len(epoch_pairs)),
        "phase_convention": result.diagnostics["config"].phase_convention,
        "cycle_duration_support_s": list(
            result.diagnostics["config"].pairing_cycle_duration_s
        ),
    }


def plot_onset_phase_circular(
    result: PyloricInspiredResult,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot paired onset phases and their circular mean/resultant length."""
    _plot_style()
    phases = _finite(result.diagnostics["paired_events"]["phase_rad"])
    if not len(phases):
        raise RuntimeError("no paired phases are available for circular plotting")
    circular = _circular_statistics(phases)
    counts, edges = np.histogram(phases, bins=np.linspace(0, 2 * np.pi, 19))
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111, projection="polar")
    ax.bar(
        centers,
        counts,
        width=np.diff(edges),
        color=OBSERVATION_COLOR,
        alpha=0.72,
    )
    max_count = max(1, int(counts.max()))
    ax.annotate(
        "",
        xy=(
            circular["mean_angle_rad"],
            max_count * circular["concentration"],
        ),
        xytext=(circular["mean_angle_rad"], 0),
        arrowprops={"color": REJECTED_COLOR, "width": 2.0, "headwidth": 7},
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(
        "Spindle onset phase | "
        f"pairs n={len(phases)}, R={circular['concentration']:.3f}\n"
        "SO trough = 0 rad; next trough = 2*pi",
        pad=18,
    )
    fig.tight_layout()
    return fig, {
        "figure_id": "onset_phase_circular",
        "paired_event_count": int(len(phases)),
        **circular,
        "warning": (
            "Interpret mean phase with concentration; this is not continuous "
            "PAC preferred phase."
        ),
    }


def plot_support_and_missingness(
    result: PyloricInspiredResult,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot finite epoch and event support alongside missing fractions."""
    _plot_style()
    audit = result.stability_audit.set_index("field_name").loc[
        list(CORE_FIELDS)
    ]
    summary = result.summary.set_index("field_name").loc[list(CORE_FIELDS)]
    y = np.arange(len(CORE_FIELDS))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 7.2), sharey=True)
    axes[0].barh(y, audit["finite_epoch_count"], color=OBSERVATION_COLOR)
    axes[0].set(
        yticks=y,
        yticklabels=CORE_FIELDS,
        xlabel="Finite retained epochs",
        title="Metric-specific epoch support",
    )
    event_values = summary["valid_event_count"].fillna(0)
    axes[1].barh(y, audit["missing_fraction"], color=REJECTED_COLOR, alpha=0.7)
    for index, events in enumerate(event_values):
        if events > 0:
            axes[1].text(
                audit.iloc[index]["missing_fraction"] + 0.01,
                index,
                f"events={int(events)}",
                va="center",
                fontsize=7,
            )
    axes[1].set(
        xlabel="Missing fraction across retained epochs",
        title="Missingness and aggregate event support",
        xlim=(0.0, 1.25),
    )
    axes[0].invert_yaxis()
    fig.suptitle(
        "Metric invalidity is not global QC rejection; undefined is not zero"
    )
    fig.tight_layout()
    return fig, {
        "figure_id": "support_missingness",
        "finite_epoch_count": audit["finite_epoch_count"].astype(int).to_dict(),
        "missing_fraction": audit["missing_fraction"].to_dict(),
        "valid_event_count": event_values.astype(int).to_dict(),
    }


def plot_role_map(
    result: PyloricInspiredResult,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Show intended roles and stability recommendations without promotion."""
    _plot_style()
    merged = result.feature_dictionary.merge(
        result.stability_audit[
            ["field_name", "missing_fraction", "recommendation"]
        ],
        on="field_name",
    )
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    y = np.arange(len(merged))
    colors = [ROLE_COLORS[role] for role in merged["intended_role"]]
    ax.barh(y, 1.0 - merged["missing_fraction"], color=colors)
    ax.set(
        yticks=y,
        yticklabels=merged["field_name"],
        xlabel="Fraction of retained epochs with a finite value",
        title="18D role map and empirical support",
        xlim=(0.0, 1.02),
    )
    for index, recommendation in enumerate(merged["recommendation"]):
        ax.text(
            0.02,
            index,
            recommendation,
            va="center",
            fontsize=6.5,
            color=(
                "white"
                if merged.iloc[index]["missing_fraction"] < 0.6
                else "black"
            ),
        )
    ax.invert_yaxis()
    handles = [
        plt.Line2D([0], [0], color=color, lw=8, label=role)
        for role, color in ROLE_COLORS.items()
        if role != "mechanism_diagnostic"
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    return fig, {
        "figure_id": "role_map",
        "role_counts": merged["intended_role"].value_counts().to_dict(),
        "warning": "Candidate roles are not frozen fitting decisions.",
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_figure(
    figure: plt.Figure,
    output_directory: Path,
    figure_id: str,
    dpi: int,
) -> dict[str, str]:
    """Save one independently generated figure as PNG and PDF."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for extension in ("png", "pdf"):
        destination = output_directory / f"{figure_id}.{extension}"
        figure.savefig(destination, dpi=dpi)
        paths[extension] = destination.name
    return paths


def generate_and_save_figures(
    bundle: ObservationBundle,
    result: PyloricInspiredResult,
    observation_config: ObservationConfig,
    config: PyloricInspiredConfig,
) -> pd.DataFrame:
    """Generate every required figure independently and save PNG/PDF pairs."""
    output_directory = (
        _resolve_project_path(config.artifact_output_dir) / "figures"
    )
    plotters: list[
        tuple[str, Any, tuple[Any, ...]]
    ] = [
        (
            "01_concept_mapping",
            plot_concept_mapping,
            (result.feature_dictionary,),
        ),
        (
            "02_epoch_qc_audit",
            plot_qc_audit,
            (bundle, result, observation_config),
        ),
        (
            "03_feature_distributions",
            plot_feature_distributions,
            (result,),
        ),
        (
            "04_robust_variability",
            plot_robust_variability,
            (result,),
        ),
        (
            "05_redundancy_heatmap",
            plot_redundancy_heatmap,
            (result,),
        ),
        (
            "06_so_proxy_morphology",
            plot_so_proxy_morphology,
            (bundle, result),
        ),
        (
            "07_spindle_detection",
            plot_spindle_detection,
            (bundle, result),
        ),
        (
            "08_event_pairing",
            plot_event_pairing,
            (bundle, result),
        ),
        (
            "09_onset_phase_circular",
            plot_onset_phase_circular,
            (result,),
        ),
        (
            "10_support_missingness",
            plot_support_and_missingness,
            (result,),
        ),
        ("11_role_map", plot_role_map, (result,)),
    ]
    records = []
    for figure_id, plotter, arguments in plotters:
        figure, metadata = plotter(*arguments)
        paths = save_figure(
            figure, output_directory, figure_id, config.artifact_dpi
        )
        plt.close(figure)
        records.append(
            {
                "figure_id": figure_id,
                "png": f"figures/{paths['png']}",
                "pdf": f"figures/{paths['pdf']}",
                "metadata": _json_safe(metadata),
            }
        )
    return pd.DataFrame(records)


def _validation_report_markdown(
    bundle: ObservationBundle,
    result: PyloricInspiredResult,
    config: PyloricInspiredConfig,
    figure_manifest: pd.DataFrame | None,
) -> str:
    reason_counts = Counter(
        reason
        for reasons in bundle.rejection_reasons.values()
        for reason in reasons
    )
    summary_view = result.summary[
        [
            "field_name",
            "value",
            "unit",
            "valid_epoch_count",
            "valid_event_count",
            "validity_status",
            "intended_role",
        ]
    ]
    failed = result.validation_checks.loc[
        ~result.validation_checks["passed"]
    ]
    lines = [
        "# Pyloric-inspired SC4001 Observation validation",
        "",
        f"- Created (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Source identifier: {bundle.subject_id} / {bundle.channel}",
        f"- Configuration SHA-256: `{config.config_hash}`",
        f"- Aligned epochs: {len(bundle.diagnostics['mapped_stages'])}",
        f"- N3 epochs: {len(bundle.n3_epoch_indices)}",
        f"- Retained N3: {len(bundle.retained_epoch_indices)}",
        f"- Rejected N3: {len(bundle.rejected_epoch_indices)}",
        f"- Rejection reasons: {dict(reason_counts)}",
        (
            f"- Validation checks: {int(result.validation_checks['passed'].sum())}/"
            f"{len(result.validation_checks)} passed"
        ),
        f"- Regression failures: {int(result.old_vs_new_regression['status'].eq('fail').sum())}",
        "",
        "## Scientific boundaries",
        "",
        "- EEG-positive/negative phase durations are observable proxies, not model cortical UP/DOWN states.",
        "- The spindle detector is provisional and is not the V8a internal T13 detector.",
        "- Event-conditioned spindle-onset timing is distinct from continuous Tort PAC.",
        "- The 18D table is a feature library. Candidate roles are not frozen fitting decisions.",
        "- No NPE, posterior, prior predictive simulation, or simulator run is performed.",
        "",
        "## Core 18D summary",
        "",
        summary_view.to_markdown(index=False),
        "",
        "## Validation checks",
        "",
        result.validation_checks.to_markdown(index=False),
        "",
        "## Old-versus-new regression",
        "",
        result.old_vs_new_regression.to_markdown(index=False),
    ]
    if figure_manifest is not None:
        lines.extend(
            [
                "",
                "## Figures",
                "",
                figure_manifest[["figure_id", "png", "pdf"]].to_markdown(
                    index=False
                ),
            ]
        )
    if len(failed):
        lines.extend(
            [
                "",
                "## Failed checks",
                "",
                failed.to_markdown(index=False),
            ]
        )
    return "\n".join(lines) + "\n"


def write_publication_safe_artifacts(
    bundle: ObservationBundle,
    result: PyloricInspiredResult,
    config: PyloricInspiredConfig,
    figure_manifest: pd.DataFrame | None = None,
) -> Path:
    """Write tables and metadata, never full raw or filtered EEG arrays."""
    output_directory = _resolve_project_path(config.artifact_output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "pyloric_inspired_18d_summary.csv": result.summary,
        "pyloric_inspired_per_epoch_features.csv": result.per_epoch_features,
        "epoch_ledger.csv": result.epoch_ledger,
        "feature_dictionary.csv": result.feature_dictionary,
        "old_vs_new_regression.csv": result.old_vs_new_regression,
        "auxiliary_diagnostics.csv": result.auxiliary_diagnostics,
        "feature_stability_audit.csv": result.stability_audit,
        "spearman_correlation.csv": result.spearman_correlation,
        "valid_pair_count.csv": result.valid_pair_count,
        "validation_checks.csv": result.validation_checks,
    }
    for filename, frame in tables.items():
        frame.to_csv(output_directory / filename, index=not filename.startswith(
            ("spearman_", "valid_pair_")
        ))
    summary_json = {
        "schema": "PyloricInspiredObservation18D-v0.1-observation-only",
        "source": {
            "subject_id": bundle.subject_id,
            "channel": bundle.channel,
            "fs_hz": bundle.fs_hz,
            "epoch_duration_s": bundle.epoch_duration_s,
            "reference": bundle.provenance["reference"],
        },
        "config_sha256": config.config_hash,
        "epoch_accounting": {
            "aligned": len(bundle.diagnostics["mapped_stages"]),
            "n3": len(bundle.n3_epoch_indices),
            "retained": len(bundle.retained_epoch_indices),
            "rejected": len(bundle.rejected_epoch_indices),
        },
        "features": result.summary.to_dict(orient="records"),
        "warnings": [
            "Candidate only - not yet frozen into fitting.",
            "No raw or filtered EEG arrays are serialized.",
            "No posterior estimator exists at this stage.",
        ],
    }
    (output_directory / "pyloric_inspired_18d_summary.json").write_text(
        json.dumps(_json_safe(summary_json), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    if figure_manifest is not None:
        manifest_records = []
        for record in figure_manifest.to_dict(orient="records"):
            manifest_records.append(_json_safe(record))
        (output_directory / "figure_manifest.json").write_text(
            json.dumps(manifest_records, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    (output_directory / "validation_report.md").write_text(
        _validation_report_markdown(
            bundle, result, config, figure_manifest
        ),
        encoding="utf-8",
    )
    return output_directory
