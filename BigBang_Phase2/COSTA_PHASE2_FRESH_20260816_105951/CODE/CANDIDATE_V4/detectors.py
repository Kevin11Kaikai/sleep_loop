"""Frozen signal processing, event detectors, T1-T8 primitives, and validity."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, log, pi
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch

from .contracts import ContractError
from .model import ParameterDraw, SimulationResult, STORED_FS, STORED_SAMPLES


EDGE = 2_000
RETAINED_START = 2_000
RETAINED_STOP = 34_000
RETAINED_SECONDS = 160.0
T1_REFERENCES = (0.55, 0.70, 0.85, 1.00, 1.15)
T8_WINDOWS = (
    ("W1", 3_000, 8_000),
    ("W2", 8_000, 13_000),
    ("W3", 13_000, 18_000),
    ("W4", 18_000, 23_000),
    ("W5", 23_000, 28_000),
    ("W6", 28_000, 33_000),
)


@dataclass(frozen=True)
class SOEvent:
    event_id: str
    onset_sample: int
    offset_sample: int
    trough_sample: int
    peak_sample: int
    minimum: float
    maximum: float
    peak_to_peak: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class SpindleEvent:
    event_id: str
    onset_sample: int
    offset_sample: int
    peak_sample: int
    peak_envelope: float
    event_rms: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class SpindleThresholds:
    median: float
    mad: float
    high: float
    low: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


@dataclass
class DetectorArtifacts:
    centered: np.ndarray
    so_signal: np.ndarray
    sigma_signal: np.ndarray
    sigma_envelope: np.ndarray
    sigma_rms40: np.ndarray
    so_phase: np.ndarray
    so_events: list[SOEvent]
    spindle_events: list[SpindleEvent]
    thresholds: SpindleThresholds


@dataclass
class MetricResult:
    metrics: dict[str, Any]
    validity: dict[str, Any]
    artifacts: DetectorArtifacts

    def as_dict(self) -> dict[str, Any]:
        return {"metrics": self.metrics, "validity": self.validity}


def _bandpass(x: np.ndarray, lo: float, hi: float, *, fs: float = 200.0, order: int = 4) -> np.ndarray:
    sos = butter(order, [lo, hi], btype="bandpass", analog=False, output="sos", fs=fs)
    return sosfiltfilt(sos, x, padtype="odd", padlen=None).astype(np.float64, copy=False)


def preprocess(observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if observation.shape != (STORED_SAMPLES,):
        raise ContractError(f"observation shape {observation.shape}, expected {(STORED_SAMPLES,)}")
    centered = np.asarray(observation, dtype=np.float64) - np.mean(observation, dtype=np.float64)
    so = _bandpass(centered, 0.3, 1.5)
    sigma = _bandpass(centered, 11.0, 16.0)
    so_phase = np.angle(hilbert(so)).astype(np.float64, copy=False)
    sigma_envelope = np.abs(hilbert(sigma)).astype(np.float64, copy=False)
    rms = np.full(centered.shape, np.nan, dtype=np.float64)
    squared = sigma * sigma
    prefix = np.concatenate((np.array([0.0]), np.cumsum(squared, dtype=np.float64)))
    centers = np.arange(19, len(centered) - 20, dtype=np.int64)
    rms[centers] = np.sqrt((prefix[centers + 21] - prefix[centers - 19]) / 40.0)
    return centered, so, sigma, so_phase, sigma_envelope, rms


def upward_zero_crossings(signal: np.ndarray) -> np.ndarray:
    return np.flatnonzero((signal[:-1] < 0.0) & (signal[1:] >= 0.0)).astype(np.int64) + 1


def detect_so_events(so: np.ndarray, *, start: int = RETAINED_START, stop: int = RETAINED_STOP) -> list[SOEvent]:
    crossings = upward_zero_crossings(so)
    events: list[SOEvent] = []
    for left, right in zip(crossings[:-1], crossings[1:]):
        duration = (int(right) - int(left)) / STORED_FS
        if duration < 0.4 or duration > 2.5:
            continue
        segment = so[int(left):int(right)]
        if segment.size == 0:
            continue
        trough_rel = int(np.argmin(segment))
        peak_rel = int(np.argmax(segment))
        trough_sample = int(left) + trough_rel
        peak_sample = int(left) + peak_rel
        minimum = float(segment[trough_rel])
        maximum = float(segment[peak_rel])
        peak_to_peak = maximum - minimum
        if not (int(left) > start and int(right) < stop and trough_sample > start and trough_sample < stop and peak_sample > start and peak_sample < stop):
            continue
        if peak_to_peak < 0.10 or minimum > -0.04:
            continue
        events.append(SOEvent(
            event_id=f"SO{len(events):05d}",
            onset_sample=int(left),
            offset_sample=int(right),
            trough_sample=trough_sample,
            peak_sample=peak_sample,
            minimum=minimum,
            maximum=maximum,
            peak_to_peak=peak_to_peak,
        ))
    return events


def calibrate_spindle_thresholds(rms40: np.ndarray) -> SpindleThresholds:
    retained = rms40[RETAINED_START:RETAINED_STOP]
    if not np.all(np.isfinite(retained)):
        raise ContractError("nonfinite sham RMS threshold calibration interval")
    median = float(np.median(retained))
    mad = float(1.4826 * np.median(np.abs(retained - median)))
    return SpindleThresholds(median, mad, median + 3.0 * mad, median + 1.5 * mad)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate((np.array([False]), mask.astype(bool), np.array([False])))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def detect_spindle_events(
    sigma: np.ndarray,
    rms40: np.ndarray,
    thresholds: SpindleThresholds,
    *,
    start: int = RETAINED_START,
    stop: int = RETAINED_STOP,
) -> list[SpindleEvent]:
    high_runs = _true_runs(np.isfinite(rms40) & (rms40 >= thresholds.high))
    extended: list[tuple[int, int]] = []
    for high_start, high_stop in high_runs:
        left = high_start
        while left > 0 and np.isfinite(rms40[left - 1]) and rms40[left - 1] > thresholds.low:
            left -= 1
        right = high_stop
        while right < len(rms40) and np.isfinite(rms40[right]) and rms40[right] > thresholds.low:
            right += 1
        extended.append((left, right))
    extended.sort()
    merged: list[list[int]] = []
    for left, right in extended:
        if not merged or left - merged[-1][1] > 20:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    events: list[SpindleEvent] = []
    for left, right in merged:
        duration = (right - left) / STORED_FS
        if duration < 0.3 or duration > 2.0:
            continue
        if not (left > start and right < stop):
            continue
        envelope_segment = rms40[left:right]
        if envelope_segment.size == 0 or not np.all(np.isfinite(envelope_segment)):
            continue
        peak = left + int(np.argmax(envelope_segment))
        if not (peak > start and peak < stop):
            continue
        event_rms = float(np.sqrt(np.mean(sigma[left:right] ** 2, dtype=np.float64)))
        events.append(SpindleEvent(
            event_id=f"SP{len(events):05d}",
            onset_sample=left,
            offset_sample=right,
            peak_sample=peak,
            peak_envelope=float(rms40[peak]),
            event_rms=event_rms,
        ))
    return events


def _welch_peak(signal: np.ndarray, *, window_mode: bool = False) -> tuple[float | None, dict[str, Any]]:
    kwargs = {
        "fs": 200.0,
        "window": "hann",
        "nperseg": 5_000 if window_mode else 12_000,
        "noverlap": 0 if window_mode else 6_000,
        "nfft": 10_000 if window_mode else 24_000,
        "detrend": "constant",
        "scaling": "density",
    }
    frequencies, power = welch(signal, **kwargs)
    mask = (frequencies >= 0.4) & (frequencies <= 1.5)
    indices = np.flatnonzero(mask)
    if indices.size == 0 or not np.all(np.isfinite(power[indices])) or not np.any(power[indices] > 0.0):
        return None, {"status": "NOT_ESTIMABLE", "reason": "INVALID_WELCH_POWER"}
    local = int(np.argmax(power[indices]))
    index = int(indices[local])
    discrete = float(frequencies[index])
    boundary = index == indices[0] or index == indices[-1]
    refined = discrete
    if not boundary and power[index - 1] > 0.0 and power[index] > 0.0 and power[index + 1] > 0.0:
        y_m, y_0, y_p = np.log(power[index - 1:index + 2])
        denominator = y_m - 2.0 * y_0 + y_p
        if denominator != 0.0 and np.isfinite(denominator):
            delta = 0.5 * (y_m - y_p) / denominator
            refined = float(frequencies[index] + delta * (frequencies[1] - frequencies[0]))
    return refined, {
        "status": "ESTIMABLE",
        "discrete_peak_hz": discrete,
        "peak_power": float(power[index]),
        "boundary_peak": boundary,
        "welch_parameters": kwargs,
    }


def _band_power(signal: np.ndarray, lo: float, hi: float, *, window_mode: bool = False) -> float:
    nperseg = len(signal) if window_mode else 12_000
    nfft = 10_000 if window_mode else 24_000
    noverlap = 0 if window_mode else 6_000
    frequencies, power = welch(signal, fs=200.0, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend="constant", scaling="density")
    mask = (frequencies >= lo) & (frequencies <= hi)
    if not np.any(mask) or not np.all(np.isfinite(power[mask])):
        return float("nan")
    df = float(frequencies[1] - frequencies[0])
    return float(np.sum(power[mask], dtype=np.float64) * df)


def broadband_specificity_veto(active: Mapping[str, Any], sham: Mapping[str, Any]) -> bool:
    """Frozen >25% broadband-without-target-fraction-increase veto."""
    active_power = active.get("power_4_30")
    sham_power = sham.get("power_4_30")
    active_fraction = active.get("sigma_fraction_4_30")
    sham_fraction = sham.get("sigma_fraction_4_30")
    if any(value is None or not np.isfinite(value) for value in (active_power, sham_power, active_fraction, sham_fraction)):
        return True
    if float(sham_power) <= 1e-12:
        return float(active_power) > 1e-12
    return bool(float(active_power) > 1.25 * float(sham_power) and float(active_fraction) <= float(sham_fraction))


def mean_variance_rescale_event_veto(
    active_observation: np.ndarray,
    sham_observation: np.ndarray,
    sham_thresholds: SpindleThresholds,
    *,
    active_so_count: int,
    active_spindle_count: int,
    sham_so_count: int,
    sham_spindle_count: int,
) -> dict[str, Any]:
    """Operationalize the frozen rescale-only event veto on a paired trace.

    The active trace is affinely normalized to the paired sham mean/variance,
    then the registered filters and unchanged sham spindle thresholds are
    reapplied.  An excess event count that disappears after that sole affine
    normalization is a hard veto.  An unavailable diagnostic fails closed.
    """
    active = np.asarray(active_observation, dtype=np.float64)
    sham = np.asarray(sham_observation, dtype=np.float64)
    if active.shape != (STORED_SAMPLES,) or sham.shape != (STORED_SAMPLES,) or not np.all(np.isfinite(active)) or not np.all(np.isfinite(sham)):
        return {"status": "INVALID_FAIL_CLOSED", "veto": True, "reason": "NONFINITE_OR_WRONG_SHAPE_PAIRED_TRACE"}
    active_mean = float(np.mean(active, dtype=np.float64)); sham_mean = float(np.mean(sham, dtype=np.float64))
    active_std = float(np.std(active, dtype=np.float64)); sham_std = float(np.std(sham, dtype=np.float64))
    if active_std <= 1e-12 or sham_std <= 1e-12:
        return {"status": "INVALID_FAIL_CLOSED", "veto": True, "reason": "SILENT_OR_ZERO_VARIANCE_PAIRED_TRACE", "active_std": active_std, "sham_std": sham_std}
    normalized = (active - active_mean) * (sham_std / active_std) + sham_mean
    _centered, so, sigma, _phase, _envelope, rms = preprocess(normalized)
    normalized_so = len(detect_so_events(so))
    normalized_spindle = len(detect_spindle_events(sigma, rms, sham_thresholds))
    so_excess = int(active_so_count) > int(sham_so_count)
    spindle_excess = int(active_spindle_count) > int(sham_spindle_count)
    so_rescale_only = so_excess and normalized_so <= int(sham_so_count)
    spindle_rescale_only = spindle_excess and normalized_spindle <= int(sham_spindle_count)
    return {
        "status": "ESTIMABLE",
        "veto": bool(so_rescale_only or spindle_rescale_only),
        "active_mean": active_mean,
        "sham_mean": sham_mean,
        "active_std": active_std,
        "sham_std": sham_std,
        "active_so_count": int(active_so_count),
        "sham_so_count": int(sham_so_count),
        "normalized_so_count": normalized_so,
        "active_spindle_count": int(active_spindle_count),
        "sham_spindle_count": int(sham_spindle_count),
        "normalized_spindle_count": normalized_spindle,
        "so_rescale_only": bool(so_rescale_only),
        "spindle_rescale_only": bool(spindle_rescale_only),
        "rule": "PAIRED_AFFINE_NORMALIZATION_THEN_FROZEN_DETECTORS_UNCHANGED_SHAM_THRESHOLDS",
    }


def _mi(phase: np.ndarray, amplitude: np.ndarray) -> float | None:
    edges = np.linspace(-pi, pi, 19, dtype=np.float64)
    bins = np.digitize(phase, edges[1:-1], right=False)
    means = np.empty(18, dtype=np.float64)
    for idx in range(18):
        values = amplitude[bins == idx]
        if values.size == 0:
            return None
        means[idx] = np.mean(values, dtype=np.float64)
    total = float(np.sum(means, dtype=np.float64))
    if total <= 0.0 or not np.isfinite(total):
        return None
    probabilities = means / total
    terms = np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0)
    return float((log(18.0) + np.sum(terms, dtype=np.float64)) / log(18.0))


def corrected_mi(phase: np.ndarray, amplitude: np.ndarray, seed: int) -> tuple[float | None, dict[str, Any]]:
    retained_phase = phase[RETAINED_START:RETAINED_STOP]
    retained_amplitude = amplitude[RETAINED_START:RETAINED_STOP]
    observed = _mi(retained_phase, retained_amplitude)
    if observed is None:
        return None, {"status": "NOT_ESTIMABLE", "reason": "MI_BIN_OR_POWER_FAILURE"}
    lags = np.arange(2_000, 34_001, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(int(seed) + 600_000))
    selected = rng.choice(lags, size=200, replace=False)
    surrogates = np.empty(200, dtype=np.float64)
    for idx, lag in enumerate(selected):
        value = _mi(retained_phase, np.roll(amplitude, int(lag))[RETAINED_START:RETAINED_STOP])
        if value is None:
            return None, {"status": "NOT_ESTIMABLE", "reason": "SURROGATE_MI_FAILURE"}
        surrogates[idx] = value
    median_surrogate = float(np.median(surrogates))
    return observed - median_surrogate, {
        "status": "ESTIMABLE",
        "raw_mi": observed,
        "surrogate_median": median_surrogate,
        "surrogate_count": 200,
        "surrogate_lags": selected.tolist(),
    }


def _preferred_phase(phase: np.ndarray, sigma_envelope: np.ndarray, events: Sequence[SpindleEvent]) -> tuple[float | None, float | None, list[float]]:
    values: list[float] = []
    for event in events:
        segment = sigma_envelope[event.onset_sample:event.offset_sample]
        if segment.size == 0 or not np.all(np.isfinite(segment)):
            return None, None, values
        # np.argmax supplies the frozen earliest-sample tie rule.
        envelope_peak = event.onset_sample + int(np.argmax(segment))
        values.append(float(phase[envelope_peak]))
    if len(values) < 10:
        return None, None, values
    array = np.asarray(values, dtype=np.float64)
    resultant = np.mean(np.exp(1j * array))
    magnitude = float(abs(resultant))
    if magnitude <= 1e-12 or magnitude < 0.20:
        return None, magnitude, values
    angle = float(atan2(float(np.sum(np.sin(array))), float(np.sum(np.cos(array)))))
    if angle == pi:
        angle = -pi
    return angle, magnitude, values


def _contained(events: Sequence[Any], start: int, stop: int) -> list[Any]:
    result = []
    for event in events:
        extrema = [getattr(event, name) for name in ("trough_sample", "peak_sample") if hasattr(event, name)]
        if event.onset_sample > start and event.offset_sample < stop and all(start < value < stop for value in extrema):
            result.append(event)
    return result


def _t8_windows(
    rates: np.ndarray,
    centered: np.ndarray,
    so_events: Sequence[SOEvent],
    spindle_events: Sequence[SpindleEvent],
) -> tuple[list[dict[str, Any]], float, bool]:
    rows: list[dict[str, Any]] = []
    valid_flags: list[bool] = []
    for window_id, start, stop in T8_WINDOWS:
        so_local = _contained(so_events, start, stop)
        sp_local = _contained(spindle_events, start, stop)
        peak, peak_meta = _welch_peak(centered[start:stop], window_mode=True)
        so_amplitude = None if not so_local else float(np.median([event.peak_to_peak for event in so_local]))
        so_density = 60.0 * len(so_local) / 25.0
        spindle_density = 60.0 * len(sp_local) / 25.0
        sigma_power = _band_power(centered[start:stop], 11.0, 16.0, window_mode=True)
        broadband_power = _band_power(centered[start:stop], 4.0, 30.0, window_mode=True)
        sigma_fraction = None if broadband_power <= 0.0 or not np.isfinite(broadband_power) else sigma_power / broadband_power
        means = np.mean(rates[:, start:stop], axis=1, dtype=np.float64)
        valid = bool(
            np.all(np.isfinite(rates[:, start:stop]))
            and np.all((means >= 0.5) & (means <= 60.0))
            and peak is not None and 0.5 <= peak <= 1.25
            and so_amplitude is not None and 0.10 <= so_amplitude <= 2.00
            and 10.0 <= so_density <= 90.0
            and 0.5 <= spindle_density <= 20.0
            and sigma_fraction is not None and 0.02 <= sigma_fraction <= 0.60
        )
        valid_flags.append(valid)
        rows.append({
            "window_id": window_id,
            "sample_indices_half_open": [start, stop],
            "mean_rates_hz": means.tolist(),
            "dominant_so_peak_hz": peak,
            "peak_metadata": peak_meta,
            "so_event_count": len(so_local),
            "so_median_peak_to_peak_m_u": so_amplitude,
            "so_density_per_min": so_density,
            "spindle_event_count": len(sp_local),
            "spindle_density_per_min": spindle_density,
            "sigma_fraction_4_30": sigma_fraction,
            "valid_regime": valid,
        })
    occupancy = sum(valid_flags) / 6.0
    persistent = any(valid_flags[idx] and valid_flags[idx + 1] for idx in range(5))
    return rows, float(occupancy), bool(persistent)


def analyze_simulation(
    result: SimulationResult,
    *,
    sham_thresholds: SpindleThresholds | None = None,
    sham_metrics: Mapping[str, Any] | None = None,
) -> MetricResult:
    centered, so, sigma, so_phase, sigma_envelope, rms40 = preprocess(result.observation)
    thresholds = sham_thresholds if sham_thresholds is not None else calibrate_spindle_thresholds(rms40)
    so_events = detect_so_events(so)
    spindle_events = detect_spindle_events(sigma, rms40, thresholds)
    peak, peak_meta = _welch_peak(centered)
    times = np.arange(STORED_SAMPLES, dtype=np.float64) / STORED_FS
    plv: dict[str, float] = {}
    for reference in T1_REFERENCES:
        phase_difference = so_phase[RETAINED_START:RETAINED_STOP] - 2.0 * np.pi * reference * times[RETAINED_START:RETAINED_STOP]
        plv[f"{reference:.2f}"] = float(abs(np.mean(np.exp(1j * phase_difference))))
    t2 = None if len(so_events) < 30 else float(np.median([event.peak_to_peak for event in so_events]))
    t3 = 60.0 * len(so_events) / RETAINED_SECONDS
    t4 = 60.0 * len(spindle_events) / RETAINED_SECONDS
    t5 = None if len(spindle_events) < 10 else float(np.median([event.event_rms for event in spindle_events]))
    t6, t6_meta = corrected_mi(so_phase, sigma_envelope, result.run.seed)
    preferred_phase, concentration, event_phases = _preferred_phase(so_phase, sigma_envelope, spindle_events)
    t8_rows, t8_occupancy, t8_persistent = _t8_windows(result.rates, centered, so_events, spindle_events)
    power_4_30 = _band_power(centered, 4.0, 30.0)
    power_11_16 = _band_power(centered, 11.0, 16.0)
    sigma_fraction = None if power_4_30 <= 0.0 or not np.isfinite(power_4_30) else power_11_16 / power_4_30

    p = result.parameter_draw.values
    rmax = np.array([p["rmax_E"], p["rmax_I"], p["rmax_T"], p["rmax_R"]])[:, None]
    finite = bool(np.all(np.isfinite(result.rates)) and np.all(np.isfinite(result.adaptation)))
    rate_bounds = bool(np.all(result.rates >= -1e-6) and np.all(result.rates <= rmax + 1e-6))
    means = np.mean(result.rates, axis=1, dtype=np.float64)
    mean_bounds = bool(np.all((means >= 0.5) & (means <= 60.0)))
    extreme_fraction = np.mean((result.rates < 0.01) | (result.rates > 95.0), axis=1, dtype=np.float64)
    extreme_ok = bool(np.all(extreme_fraction <= 0.01))
    paired_power_ok = True
    threshold_identity = True
    if sham_metrics is not None:
        sham_power = sham_metrics.get("power_4_30")
        paired_power_ok = bool(sham_power is not None and sham_power > 0.0 and power_4_30 <= 4.0 * sham_power)
        threshold_identity = sham_metrics.get("spindle_thresholds") == thresholds.as_dict()
    stable = finite and rate_bounds and mean_bounds and extreme_ok and paired_power_ok and threshold_identity

    metrics: dict[str, Any] = {
        "T1_dominant_so_peak_hz": peak,
        "T1_peak_metadata": peak_meta,
        "T1_plv_by_reference": plv,
        "T2_so_median_peak_to_peak_m_u": t2,
        "T3_so_density_per_min": t3,
        "T4_spindle_density_per_min": t4,
        "T5_spindle_event_rms_m_u": t5,
        "T6_bias_corrected_mi": t6,
        "T6_metadata": t6_meta,
        "T7_preferred_phase_radians": preferred_phase,
        "T7_resultant_length": concentration,
        "T7_event_phases_radians": event_phases,
        "T8_windows": t8_rows,
        "T8_occupancy": t8_occupancy,
        "T8_persistent": t8_persistent,
        "so_event_count": len(so_events),
        "spindle_event_count": len(spindle_events),
        "power_4_30": power_4_30,
        "power_11_16": power_11_16,
        "sigma_fraction_4_30": sigma_fraction,
        "spindle_thresholds": thresholds.as_dict(),
        "population_mean_rates_hz": means.tolist(),
    }
    validity = {
        "finite": finite,
        "rate_bounds": rate_bounds,
        "mean_rate_bounds": mean_bounds,
        "extreme_fraction_by_population": extreme_fraction.tolist(),
        "extreme_fraction_ok": extreme_ok,
        "paired_power_within_4x": paired_power_ok,
        "paired_threshold_identity": threshold_identity,
        "detector_retained_fraction": RETAINED_SECONDS / 180.0,
        "detector_retained_fraction_ok": RETAINED_SECONDS / 180.0 >= 0.85,
        "stable": stable,
        "T1_valid": peak is not None and not peak_meta.get("boundary_peak", True) and len(so_events) >= 30,
        "T2_valid": t2 is not None,
        "T3_valid": True,
        "T4_valid": True,
        "T5_valid": t5 is not None,
        "T6_valid": t6 is not None and len(so_events) >= 30 and len(spindle_events) >= 10,
        "T7_valid": preferred_phase is not None and len(so_events) >= 30 and len(spindle_events) >= 10,
        "T8_valid": finite and rate_bounds,
    }
    artifacts = DetectorArtifacts(centered, so, sigma, sigma_envelope, rms40, so_phase, so_events, spindle_events, thresholds)
    return MetricResult(metrics, validity, artifacts)
