"""SC4001 observation loading and epoch-preserving EEG summaries."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable
import warnings as python_warnings

import mne
import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, detrend, find_peaks, hilbert, sosfiltfilt, welch

from .schemas import ObservationBundle, ObservationConfig, PSDResult, SummaryMetric


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_STAGE_TO_AASM = {
    "sleep stage w": "W",
    "sleep stage 1": "N1",
    "sleep stage 2": "N2",
    "sleep stage 3": "N3",
    "sleep stage 4": "N3",
    "sleep stage r": "REM",
    "sleep stage ?": "Unknown",
    "movement time": "Unknown",
    "w": "W",
    "1": "N1",
    "2": "N2",
    "3": "N3",
    "4": "N3",
    "r": "REM",
    "?": "Unknown",
    "mt": "Unknown",
    "unscored": "Unknown",
}


def map_raw_stage(raw_label: str) -> str:
    """Map R&K labels to AASM; short label ``3`` is explicitly N3."""
    return RAW_STAGE_TO_AASM.get(str(raw_label).strip().lower(), "Unknown")


def _pair(values: Iterable[float]) -> tuple[float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 2:
        raise ValueError(f"expected a two-element band, got {result}")
    return result


def load_observation_config(
    path: str | Path = "S4_sbi/configs/observation_sc4001.yaml",
) -> ObservationConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return ObservationConfig(
        subject_id=str(payload["subject_id"]),
        manifest_path=str(payload["manifest_path"]),
        channel_candidates=tuple(map(str, payload["channel_candidates"])),
        epoch_duration_s=float(payload["epoch"]["duration_s"]),
        random_seed=int(payload["reproducibility"]["random_seed"]),
        max_peak_to_peak_uv=float(payload["qc"]["max_peak_to_peak_uv"]),
        min_std_uv=float(payload["qc"]["min_std_uv"]),
        welch_window=str(payload["welch"]["primary_window"]),
        sensitivity_window=str(payload["welch"]["sensitivity_window"]),
        welch_segment_s=float(payload["welch"]["segment_s"]),
        welch_overlap_s=float(payload["welch"]["overlap_s"]),
        psd_band_hz=_pair(payload["welch"]["analysis_band_hz"]),
        so_psd_band_hz=_pair(payload["welch"]["so_band_hz"]),
        so_event_band_hz=_pair(payload["slow_oscillation"]["event_band_hz"]),
        so_half_wave_uv=float(payload["slow_oscillation"]["half_wave_uv"]),
        so_duration_s=_pair(payload["slow_oscillation"]["duration_s"]),
        so_min_peak_distance_s=float(
            payload["slow_oscillation"]["min_peak_distance_s"]
        ),
        waveform_half_window_s=float(
            payload["slow_oscillation"]["waveform_half_window_s"]
        ),
        spindle_band_hz=_pair(payload["spindle"]["detector_band_hz"]),
        spindle_rms_window_s=float(payload["spindle"]["rms_window_s"]),
        spindle_threshold_sd=float(payload["spindle"]["threshold_sd"]),
        spindle_duration_s=_pair(payload["spindle"]["duration_s"]),
        spindle_merge_gap_s=float(payload["spindle"]["merge_gap_s"]),
        pac_phase_band_hz=_pair(payload["pac"]["phase_band_hz"]),
        pac_amplitude_band_hz=_pair(payload["pac"]["amplitude_band_hz"]),
        pac_phase_bins=int(payload["pac"]["phase_bins"]),
        filter_order=int(payload["filters"]["order"]),
        filter_edge_trim_s=float(payload["filters"]["edge_trim_s"]),
        fooof_version=str(payload["fooof"]["required_version"]),
        fooof_parameters={
            "peak_width_limits": list(payload["fooof"]["peak_width_limits"]),
            "max_n_peaks": int(payload["fooof"]["max_n_peaks"]),
            "min_peak_height": float(payload["fooof"]["min_peak_height"]),
            "aperiodic_mode": str(payload["fooof"]["aperiodic_mode"]),
        },
    )


def _require_fooof(required_version: str):
    installed = metadata.version("fooof")
    if installed != required_version:
        raise RuntimeError(
            f"fooof=={required_version} is required; found {installed}. "
            "No fallback is permitted."
        )
    from fooof import FOOOF

    return FOOOF, installed


def _read_manifest(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-16")


def _resolve_source_paths(config: ObservationConfig) -> tuple[Path, Path, Path]:
    manifest_path = Path(config.manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest = _read_manifest(manifest_path)
    rows = manifest.loc[
        manifest["subject_id"].astype(str).eq(config.subject_id)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one manifest row for {config.subject_id}, found {len(rows)}"
        )
    row = rows.iloc[0]

    def resolve(value: Any) -> Path:
        source = Path(str(value))
        return source if source.is_absolute() else PROJECT_ROOT / source

    psg_path = resolve(row["psg_path"])
    hypnogram_path = resolve(row["hypnogram_path"])
    if not psg_path.is_file() or not hypnogram_path.is_file():
        raise FileNotFoundError(
            f"missing PSG or hypnogram for {config.subject_id}; "
            "check the ignored local manifest"
        )
    return manifest_path, psg_path, hypnogram_path


def _pick_channel(raw: mne.io.BaseRaw, candidates: tuple[str, ...]) -> str:
    upper_to_original = {name.upper(): name for name in raw.ch_names}
    for candidate in candidates:
        if candidate.upper() in upper_to_original:
            return upper_to_original[candidate.upper()]
    raise RuntimeError(
        "no configured Fpz-Cz channel was found; available channels were "
        + ", ".join(raw.ch_names)
    )


def _annotation_epoch_alignment(
    annotations: mne.Annotations,
    n_signal_epochs: int,
    epoch_duration_s: float,
) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
    raw_labels = np.full(n_signal_epochs, "Unscored", dtype=object)
    table: list[dict[str, Any]] = []
    warnings: list[str] = []
    grouped: dict[str, dict[str, float]] = {}

    for onset, duration, description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        label = str(description).strip()
        start_float = float(onset) / epoch_duration_s
        count_float = float(duration) / epoch_duration_s
        start_epoch = int(round(start_float))
        n_epochs = int(round(count_float))
        if not np.isclose(start_float, start_epoch, atol=1e-6):
            warnings.append(
                f"annotation onset {onset:.6f}s for {label!r} is off the "
                f"{epoch_duration_s:g}s epoch grid"
            )
        if not np.isclose(count_float, n_epochs, atol=1e-6):
            warnings.append(
                f"annotation duration {duration:.6f}s for {label!r} is not "
                f"an integer number of {epoch_duration_s:g}s epochs"
            )
        stop_epoch = start_epoch + n_epochs
        lo = max(0, start_epoch)
        hi = min(n_signal_epochs, stop_epoch)
        if lo < hi:
            raw_labels[lo:hi] = label
        record = grouped.setdefault(
            label, {"annotation_events": 0, "duration_s": 0.0, "aligned_epochs": 0}
        )
        record["annotation_events"] += 1
        record["duration_s"] += float(duration)
        record["aligned_epochs"] += max(0, hi - lo)

    for label in sorted(grouped):
        record = grouped[label]
        table.append(
            {
                "raw_label": label,
                "aasm_label": map_raw_stage(label),
                "annotation_events": int(record["annotation_events"]),
                "duration_s": float(record["duration_s"]),
                "duration_epochs_30s": int(
                    round(record["duration_s"] / epoch_duration_s)
                ),
                "aligned_recording_epochs": int(record["aligned_epochs"]),
            }
        )
    unscored = int(np.sum(raw_labels == "Unscored"))
    if unscored:
        warnings.append(
            f"{unscored} signal epochs were not covered by a hypnogram annotation"
        )
    annotated_stop = max(
        (
            float(onset) + float(duration)
            for onset, duration in zip(annotations.onset, annotations.duration)
        ),
        default=0.0,
    )
    signal_stop = n_signal_epochs * epoch_duration_s
    if annotated_stop > signal_stop + 1e-6:
        warnings.append(
            f"hypnogram extends {annotated_stop - signal_stop:.1f}s beyond the "
            "complete PSG epochs; alignment was clipped to the PSG"
        )
    return raw_labels.astype(str), table, warnings


def _load_epoch_data(
    config: ObservationConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[dict[str, Any]], list[str]]:
    manifest_path, psg_path, hypnogram_path = _resolve_source_paths(config)
    mne.set_log_level("ERROR")
    with python_warnings.catch_warnings():
        python_warnings.filterwarnings(
            "ignore", message="Channels contain different .* filters"
        )
        python_warnings.filterwarnings(
            "ignore", message="Highpass cutoff frequency .*"
        )
        raw = mne.io.read_raw_edf(str(psg_path), preload=False, verbose=False)
    channel = _pick_channel(raw, config.channel_candidates)
    fs_hz = float(raw.info["sfreq"])
    samples_per_epoch = int(round(config.epoch_duration_s * fs_hz))
    n_signal_epochs = raw.n_times // samples_per_epoch
    n_complete_samples = n_signal_epochs * samples_per_epoch
    data_uv = (
        raw.get_data(picks=[channel], start=0, stop=n_complete_samples)[0] * 1e6
    )
    segments_all = data_uv.reshape(n_signal_epochs, samples_per_epoch)
    annotations = mne.read_annotations(str(hypnogram_path))
    raw_epoch_labels, annotation_table, warnings = _annotation_epoch_alignment(
        annotations, n_signal_epochs, config.epoch_duration_s
    )
    mapped_stages = np.array([map_raw_stage(label) for label in raw_epoch_labels])
    raw_label_counts = Counter(map(str, annotations.description))
    raw.close()

    reference = (
        "bipolar derivation Fpz-Cz as recorded; no additional re-reference applied"
        if "FPZ-CZ" in channel.upper()
        else "as recorded; no additional re-reference applied"
    )
    source_metadata = {
        "manifest_path": str(Path(config.manifest_path).as_posix()),
        "psg_file": psg_path.name,
        "hypnogram_file": hypnogram_path.name,
        "channel": channel,
        "reference": reference,
        "native_storage_unit": "V (MNE internal)",
        "analysis_unit": "uV",
        "fs_hz": fs_hz,
        "recording_duration_s": float(n_complete_samples / fs_hz),
        "recording_complete_epochs": int(n_signal_epochs),
        "raw_label_counts": dict(sorted(raw_label_counts.items())),
        "aligned_raw_label_counts": dict(
            sorted(Counter(map(str, raw_epoch_labels)).items())
        ),
        "aligned_aasm_counts": dict(
            sorted(Counter(map(str, mapped_stages)).items())
        ),
        "manifest_file": manifest_path.name,
    }
    return (
        source_metadata,
        segments_all,
        mapped_stages,
        annotation_table,
        warnings,
    )


def _quality_control(
    segments_all: np.ndarray,
    mapped_stages: np.ndarray,
    config: ObservationConfig,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[str, ...]], list[dict[str, Any]]]:
    n3_indices = np.flatnonzero(mapped_stages == "N3")
    retained: list[int] = []
    rejected: list[int] = []
    reasons: dict[int, tuple[str, ...]] = {}
    qc_rows: list[dict[str, Any]] = []
    for epoch_index in n3_indices:
        segment = segments_all[epoch_index]
        epoch_reasons: list[str] = []
        finite = bool(np.isfinite(segment).all())
        peak_to_peak_uv = float(np.ptp(segment)) if finite else float("nan")
        std_uv = float(np.std(segment)) if finite else float("nan")
        if not finite:
            epoch_reasons.append("non_finite_samples")
        else:
            if peak_to_peak_uv > config.max_peak_to_peak_uv:
                epoch_reasons.append("peak_to_peak_above_threshold")
            if std_uv < config.min_std_uv:
                epoch_reasons.append("near_flat_signal")
        if epoch_reasons:
            rejected.append(int(epoch_index))
            reasons[int(epoch_index)] = tuple(epoch_reasons)
        else:
            retained.append(int(epoch_index))
        qc_rows.append(
            {
                "epoch_index": int(epoch_index),
                "retained": not epoch_reasons,
                "peak_to_peak_uv": peak_to_peak_uv,
                "std_uv": std_uv,
                "reasons": ";".join(epoch_reasons),
            }
        )
    return (
        np.asarray(retained, dtype=int),
        np.asarray(rejected, dtype=int),
        reasons,
        qc_rows,
    )


def _normalized_psd(
    frequencies_hz: np.ndarray,
    density: np.ndarray,
    band_hz: tuple[float, float],
) -> np.ndarray:
    mask = (frequencies_hz >= band_hz[0]) & (frequencies_hz <= band_hz[1])
    area = float(np.trapezoid(density[mask], frequencies_hz[mask]))
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError("PSD normalization area is non-positive")
    return density / area


def _compute_psd(
    segments: np.ndarray,
    fs_hz: float,
    config: ObservationConfig,
) -> PSDResult:
    nperseg = int(round(config.welch_segment_s * fs_hz))
    noverlap = int(round(config.welch_overlap_s * fs_hz))

    def compute(window: str) -> tuple[np.ndarray, np.ndarray]:
        rows = []
        frequencies = None
        for segment in segments:
            frequencies, density = welch(
                segment,
                fs=fs_hz,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
            )
            rows.append(density)
        return np.asarray(frequencies), np.vstack(rows)

    frequencies, hann_rows = compute(config.welch_window)
    hamming_frequencies, hamming_rows = compute(config.sensitivity_window)
    if not np.array_equal(frequencies, hamming_frequencies):
        raise RuntimeError("Hann and Hamming Welch grids differ")
    hann_mean = np.mean(hann_rows, axis=0)
    hamming_mean = np.mean(hamming_rows, axis=0)
    return PSDResult(
        frequencies_hz=frequencies,
        epoch_psd_hann_uv2_hz=hann_rows,
        aggregate_hann_uv2_hz=hann_mean,
        epoch_psd_hamming_uv2_hz=hamming_rows,
        aggregate_hamming_uv2_hz=hamming_mean,
        normalized_hann=_normalized_psd(
            frequencies, hann_mean, config.psd_band_hz
        ),
        normalized_hamming=_normalized_psd(
            frequencies, hamming_mean, config.psd_band_hz
        ),
        parameters={
            "algorithm": "scipy.signal.welch",
            "primary_window": config.welch_window,
            "sensitivity_window": config.sensitivity_window,
            "segment_s": config.welch_segment_s,
            "overlap_s": config.welch_overlap_s,
            "detrend": "constant",
            "scaling": "density",
            "epoch_aggregation": "arithmetic mean",
            "unit": "uV^2/Hz",
        },
    )


def _spectral_statistics(
    frequencies_hz: np.ndarray,
    density: np.ndarray,
    config: ObservationConfig,
) -> dict[str, float]:
    so_mask = (
        (frequencies_hz >= config.so_psd_band_hz[0])
        & (frequencies_hz <= config.so_psd_band_hz[1])
    )
    analysis_mask = (
        (frequencies_hz >= config.psd_band_hz[0])
        & (frequencies_hz <= config.psd_band_hz[1])
    )
    if not so_mask.any() or not analysis_mask.any():
        raise RuntimeError("configured SO or analysis band is absent from the PSD")
    peak_local = int(np.argmax(density[so_mask]))
    peak_frequency = float(frequencies_hz[so_mask][peak_local])
    peak_density = float(density[so_mask][peak_local])
    so_power = float(
        np.trapezoid(density[so_mask], frequencies_hz[so_mask])
    )
    total_power = float(
        np.trapezoid(density[analysis_mask], frequencies_hz[analysis_mask])
    )
    width = config.so_psd_band_hz[1] - config.so_psd_band_hz[0]
    neighbor_mask = (
        (
            (frequencies_hz >= max(0.1, config.so_psd_band_hz[0] - width))
            & (frequencies_hz < config.so_psd_band_hz[0])
        )
        | (
            (frequencies_hz > config.so_psd_band_hz[1])
            & (frequencies_hz <= config.so_psd_band_hz[1] + width)
        )
    )
    background = float(np.mean(density[neighbor_mask]))
    return {
        "so_peak_frequency_hz": peak_frequency,
        "relative_so_power": so_power / total_power,
        "so_q": peak_density / max(background, 1e-30),
    }


def _compute_fooof(
    psd: PSDResult,
    config: ObservationConfig,
) -> dict[str, Any]:
    FOOOF, installed_version = _require_fooof(config.fooof_version)
    mask = (
        (psd.frequencies_hz >= config.psd_band_hz[0])
        & (psd.frequencies_hz <= config.psd_band_hz[1])
    )
    model = FOOOF(**config.fooof_parameters, verbose=False)
    model.fit(
        psd.frequencies_hz[mask],
        psd.aggregate_hann_uv2_hz[mask],
        list(config.psd_band_hz),
    )
    if model.aperiodic_params_.size < 2:
        raise RuntimeError("FOOOF did not return a fixed-mode aperiodic exponent")
    return {
        "version": installed_version,
        "parameters": dict(config.fooof_parameters),
        "frequencies_hz": np.asarray(model.freqs, dtype=float),
        "aperiodic_fit_log10": np.asarray(model._ap_fit, dtype=float),
        "periodic_log10": np.asarray(model._peak_fit, dtype=float),
        "aperiodic_offset": float(model.aperiodic_params_[0]),
        "aperiodic_exponent": float(model.aperiodic_params_[1]),
        "peaks": np.asarray(model.peak_params_, dtype=float),
        "valid": True,
    }


def _filter_epoch(
    segment_uv: np.ndarray,
    fs_hz: float,
    band_hz: tuple[float, float],
    order: int,
) -> np.ndarray:
    sos = butter(order, band_hz, btype="band", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, detrend(segment_uv, type="constant"))


def _compute_so_diagnostics(
    segments: np.ndarray,
    retained_epoch_indices: np.ndarray,
    fs_hz: float,
    config: ObservationConfig,
) -> dict[str, Any]:
    filtered = np.full_like(segments, np.nan, dtype=float)
    valid_mask = np.zeros(len(segments), dtype=bool)
    failure_reasons: dict[int, str] = {}
    events: list[dict[str, Any]] = []
    ibis_s: list[float] = []
    waveform_snippets: list[np.ndarray] = []
    half_window = int(round(config.waveform_half_window_s * fs_hz))
    min_distance = int(round(config.so_min_peak_distance_s * fs_hz))

    for row, (epoch_index, segment) in enumerate(
        zip(retained_epoch_indices, segments)
    ):
        try:
            epoch_so = _filter_epoch(
                segment, fs_hz, config.so_event_band_hz, config.filter_order
            )
        except (ValueError, FloatingPointError) as error:
            failure_reasons[int(epoch_index)] = f"filter_failed: {error}"
            continue
        if not np.isfinite(epoch_so).all() or np.std(epoch_so) < 1e-9:
            failure_reasons[int(epoch_index)] = "filtered_signal_degenerate"
            continue
        filtered[row] = epoch_so
        valid_mask[row] = True
        down_peaks, _ = find_peaks(-epoch_so, distance=min_distance)
        up_peaks, _ = find_peaks(epoch_so, distance=min_distance)
        up_indices: list[int] = []
        up_cursor = 0
        for down_index in down_peaks:
            while (
                up_cursor < len(up_peaks)
                and int(up_peaks[up_cursor]) <= int(down_index)
            ):
                up_cursor += 1
            if up_cursor >= len(up_peaks):
                break
            up_index = int(up_peaks[up_cursor])
            up_cursor += 1
            duration_s = (up_index - int(down_index)) / fs_hz
            amplitude_uv = float(epoch_so[up_index] - epoch_so[down_index])
            if not (
                config.so_duration_s[0]
                <= duration_s
                <= config.so_duration_s[1]
            ):
                continue
            if amplitude_uv < config.so_half_wave_uv:
                continue
            up_indices.append(up_index)
            events.append(
                {
                    "epoch_index": int(epoch_index),
                    "down_sample": int(down_index),
                    "up_sample": up_index,
                    "duration_s": float(duration_s),
                    "half_wave_amplitude_uv": amplitude_uv,
                }
            )
            start = int(down_index) - half_window
            stop = int(down_index) + half_window + 1
            if start >= 0 and stop <= len(epoch_so):
                snippet = epoch_so[start:stop]
                snippet_std = float(np.std(snippet))
                if snippet_std > 1e-9:
                    waveform_snippets.append(
                        (snippet - np.mean(snippet)) / snippet_std
                    )
        if len(up_indices) >= 2:
            ibis_s.extend(np.diff(np.asarray(up_indices, dtype=float)) / fs_hz)

    waveform_time_s = (
        np.arange(2 * half_window + 1, dtype=float) - half_window
    ) / fs_hz
    if waveform_snippets:
        snippets_array = np.vstack(waveform_snippets)
        waveform_mean = np.mean(snippets_array, axis=0)
        waveform_sem = np.std(snippets_array, axis=0) / np.sqrt(
            len(snippets_array)
        )
    else:
        snippets_array = np.empty((0, len(waveform_time_s)))
        waveform_mean = np.full(len(waveform_time_s), np.nan)
        waveform_sem = np.full(len(waveform_time_s), np.nan)
    return {
        "parameters": {
            "filter": "scipy.signal.butter+sosfiltfilt, per epoch",
            "filter_order": config.filter_order,
            "band_hz": config.so_event_band_hz,
            "half_wave_uv": config.so_half_wave_uv,
            "down_to_up_duration_s": config.so_duration_s,
            "min_peak_distance_s": config.so_min_peak_distance_s,
            "waveform_half_window_s": config.waveform_half_window_s,
        },
        "filtered_uv": filtered,
        "validity_mask": valid_mask,
        "failure_reasons": failure_reasons,
        "events": events,
        "ibi_s": np.asarray(ibis_s, dtype=float),
        "waveform_snippets_z": snippets_array,
        "waveform_time_s": waveform_time_s,
        "waveform_mean_z": waveform_mean,
        "waveform_sem_z": waveform_sem,
    }


def _compute_spindle_diagnostics(
    segments: np.ndarray,
    retained_epoch_indices: np.ndarray,
    fs_hz: float,
    config: ObservationConfig,
) -> dict[str, Any]:
    filtered = np.full_like(segments, np.nan, dtype=float)
    envelopes = np.full_like(segments, np.nan, dtype=float)
    thresholds = np.full(len(segments), np.nan, dtype=float)
    valid_mask = np.zeros(len(segments), dtype=bool)
    failure_reasons: dict[int, str] = {}
    events: list[dict[str, Any]] = []
    rms_samples = max(1, int(round(config.spindle_rms_window_s * fs_hz)))
    kernel = np.ones(rms_samples, dtype=float) / rms_samples
    merge_gap_samples = int(round(config.spindle_merge_gap_s * fs_hz))

    for row, (epoch_index, segment) in enumerate(
        zip(retained_epoch_indices, segments)
    ):
        try:
            sigma = _filter_epoch(
                segment, fs_hz, config.spindle_band_hz, config.filter_order
            )
        except (ValueError, FloatingPointError) as error:
            failure_reasons[int(epoch_index)] = f"filter_failed: {error}"
            continue
        rms = np.sqrt(np.convolve(sigma**2, kernel, mode="same"))
        threshold = float(
            np.mean(rms) + config.spindle_threshold_sd * np.std(rms)
        )
        if not np.isfinite(rms).all() or not np.isfinite(threshold):
            failure_reasons[int(epoch_index)] = "envelope_non_finite"
            continue
        above = (rms > threshold).astype(np.int8)
        edges = np.diff(np.concatenate(([0], above, [0])))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        merged: list[list[int]] = []
        for start, stop in zip(starts, stops):
            if merged and int(start) - merged[-1][1] < merge_gap_samples:
                merged[-1][1] = int(stop)
            else:
                merged.append([int(start), int(stop)])
        for start, stop in merged:
            duration_s = (stop - start) / fs_hz
            if (
                config.spindle_duration_s[0]
                <= duration_s
                <= config.spindle_duration_s[1]
            ):
                events.append(
                    {
                        "epoch_index": int(epoch_index),
                        "start_sample": start,
                        "stop_sample": stop,
                        "duration_s": float(duration_s),
                        "peak_envelope_uv": float(np.max(rms[start:stop])),
                        "threshold_uv": threshold,
                    }
                )
        filtered[row] = sigma
        envelopes[row] = rms
        thresholds[row] = threshold
        valid_mask[row] = True
    return {
        "parameters": {
            "filter": "scipy.signal.butter+sosfiltfilt, per epoch",
            "filter_order": config.filter_order,
            "band_hz": config.spindle_band_hz,
            "envelope": "moving RMS",
            "rms_window_s": config.spindle_rms_window_s,
            "threshold": (
                f"per-epoch mean + {config.spindle_threshold_sd:g} SD"
            ),
            "duration_s": config.spindle_duration_s,
            "merge_gap_s": config.spindle_merge_gap_s,
        },
        "filtered_uv": filtered,
        "envelope_uv": envelopes,
        "threshold_uv": thresholds,
        "validity_mask": valid_mask,
        "failure_reasons": failure_reasons,
        "events": events,
    }


def _compute_pac_diagnostics(
    segments: np.ndarray,
    retained_epoch_indices: np.ndarray,
    fs_hz: float,
    config: ObservationConfig,
) -> dict[str, Any]:
    bin_edges = np.linspace(-np.pi, np.pi, config.pac_phase_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    amplitude_sums = np.zeros(config.pac_phase_bins, dtype=float)
    sample_counts = np.zeros(config.pac_phase_bins, dtype=int)
    valid_mask = np.zeros(len(segments), dtype=bool)
    failure_reasons: dict[int, str] = {}
    up_amplitude = 0.0
    down_amplitude = 0.0
    trim = int(round(config.filter_edge_trim_s * fs_hz))

    for row, (epoch_index, segment) in enumerate(
        zip(retained_epoch_indices, segments)
    ):
        try:
            phase_signal = _filter_epoch(
                segment, fs_hz, config.pac_phase_band_hz, config.filter_order
            )
            amplitude_signal = _filter_epoch(
                segment,
                fs_hz,
                config.pac_amplitude_band_hz,
                config.filter_order,
            )
            phase = np.angle(hilbert(phase_signal))
            amplitude = np.abs(hilbert(amplitude_signal))
        except (ValueError, FloatingPointError) as error:
            failure_reasons[int(epoch_index)] = f"filter_or_hilbert_failed: {error}"
            continue
        if 2 * trim >= len(phase):
            failure_reasons[int(epoch_index)] = "edge_trim_removes_entire_epoch"
            continue
        phase = phase[trim:-trim] if trim else phase
        amplitude = amplitude[trim:-trim] if trim else amplitude
        if not np.isfinite(phase).all() or not np.isfinite(amplitude).all():
            failure_reasons[int(epoch_index)] = "phase_or_amplitude_non_finite"
            continue
        for bin_index in range(config.pac_phase_bins):
            if bin_index == config.pac_phase_bins - 1:
                mask = (
                    (phase >= bin_edges[bin_index])
                    & (phase <= bin_edges[bin_index + 1])
                )
            else:
                mask = (
                    (phase >= bin_edges[bin_index])
                    & (phase < bin_edges[bin_index + 1])
                )
            amplitude_sums[bin_index] += float(np.sum(amplitude[mask]))
            sample_counts[bin_index] += int(np.sum(mask))
        up_mask = np.abs(phase) <= np.pi / 2
        up_amplitude += float(np.sum(amplitude[up_mask]))
        down_amplitude += float(np.sum(amplitude[~up_mask]))
        valid_mask[row] = True

    mean_amplitude = np.divide(
        amplitude_sums,
        sample_counts,
        out=np.zeros_like(amplitude_sums),
        where=sample_counts > 0,
    )
    total = float(np.sum(mean_amplitude))
    valid = bool(
        np.all(sample_counts > 0)
        and total > 1e-12
        and np.sum(valid_mask) > 0
    )
    if valid:
        probability = mean_amplitude / total
        entropy = float(-np.sum(probability * np.log(probability)))
        mi = float(
            (np.log(config.pac_phase_bins) - entropy)
            / np.log(config.pac_phase_bins)
        )
        preferred_phase = float(bin_centers[int(np.argmax(mean_amplitude))])
        up_down_ratio = (
            float(up_amplitude / down_amplitude)
            if down_amplitude > 1e-12
            else float("nan")
        )
    else:
        probability = np.full(config.pac_phase_bins, np.nan)
        mi = float("nan")
        preferred_phase = float("nan")
        up_down_ratio = float("nan")
    return {
        "parameters": {
            "algorithm": "Tort normalized-KL modulation index",
            "phase_filter": "scipy.signal.butter+sosfiltfilt, per epoch",
            "amplitude_filter": "scipy.signal.butter+sosfiltfilt, per epoch",
            "filter_order": config.filter_order,
            "phase_band_hz": config.pac_phase_band_hz,
            "amplitude_band_hz": config.pac_amplitude_band_hz,
            "n_phase_bins": config.pac_phase_bins,
            "edge_trim_s_per_epoch": config.filter_edge_trim_s,
            "aggregation": "phase-bin amplitude sums across valid epochs",
        },
        "valid": valid,
        "validity_mask": valid_mask,
        "failure_reasons": failure_reasons,
        "phase_bin_edges_rad": bin_edges,
        "phase_bin_centers_rad": bin_centers,
        "mean_amplitude_uv": mean_amplitude,
        "probability": probability,
        "mi": mi,
        "preferred_phase_rad": preferred_phase,
        "preferred_phase_sin": float(np.sin(preferred_phase)),
        "preferred_phase_cos": float(np.cos(preferred_phase)),
        "pac_up_down_ratio": up_down_ratio,
    }


def _summary(
    name: str,
    value: float,
    unit: str,
    band_hz: tuple[float, float] | None,
    algorithm: str,
    aggregation: str,
    source_signal: str,
    category: str,
    valid: bool = True,
    invalid_reason: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> SummaryMetric:
    return SummaryMetric(
        name=name,
        value=float(value),
        unit=unit,
        band_hz=band_hz,
        algorithm=algorithm,
        aggregation=aggregation,
        source_signal=source_signal,
        category=category,
        valid=valid,
        invalid_reason=invalid_reason,
        parameters={} if parameters is None else dict(parameters),
    )


def build_observation_bundle(
    config: ObservationConfig | None = None,
) -> ObservationBundle:
    """Load SC4001 and compute observation-only metrics without simulation."""
    config = config or load_observation_config()
    np.random.seed(config.random_seed)
    FOOOF, installed_fooof = _require_fooof(config.fooof_version)
    del FOOOF
    (
        source_metadata,
        all_segments,
        mapped_stages,
        annotation_table,
        warnings,
    ) = _load_epoch_data(config)
    (
        retained_indices,
        rejected_indices,
        rejection_reasons,
        qc_rows,
    ) = _quality_control(all_segments, mapped_stages, config)
    n3_indices = np.flatnonzero(mapped_stages == "N3")
    if not len(retained_indices):
        raise RuntimeError("no N3 epochs survived observation QC")
    segments = np.asarray(all_segments[retained_indices], dtype=float)
    psd = _compute_psd(segments, source_metadata["fs_hz"], config)
    hann_stats = _spectral_statistics(
        psd.frequencies_hz, psd.aggregate_hann_uv2_hz, config
    )
    hamming_stats = _spectral_statistics(
        psd.frequencies_hz, psd.aggregate_hamming_uv2_hz, config
    )
    fooof_result = _compute_fooof(psd, config)
    so = _compute_so_diagnostics(
        segments, retained_indices, source_metadata["fs_hz"], config
    )
    spindle = _compute_spindle_diagnostics(
        segments, retained_indices, source_metadata["fs_hz"], config
    )
    pac = _compute_pac_diagnostics(
        segments, retained_indices, source_metadata["fs_hz"], config
    )

    retained_minutes = (
        len(retained_indices) * config.epoch_duration_s / 60.0
    )
    so_event_rate = len(so["events"]) / retained_minutes
    ibi_valid = len(so["ibi_s"]) >= 2 and np.mean(so["ibi_s"]) > 0
    ibi_cv = (
        float(np.std(so["ibi_s"]) / np.mean(so["ibi_s"]))
        if ibi_valid
        else float("nan")
    )
    spindle_events = spindle["events"]
    spindle_density = len(spindle_events) / retained_minutes
    spindle_duration_valid = bool(spindle_events)
    spindle_mean_duration = (
        float(np.mean([event["duration_s"] for event in spindle_events]))
        if spindle_events
        else float("nan")
    )
    waveform_valid = bool(len(so["waveform_snippets_z"]))
    waveform_peak_to_peak = (
        float(np.ptp(so["waveform_mean_z"]))
        if waveform_valid
        else float("nan")
    )

    category_a = "inference-summary candidate"
    category_b = "mechanism diagnostic"
    category_c = "held-out PPC candidate"
    summaries = {
        "fooof_aperiodic_exponent": _summary(
            "fooof_aperiodic_exponent",
            fooof_result["aperiodic_exponent"],
            "1",
            config.psd_band_hz,
            "FOOOF fixed aperiodic fit",
            "fit to arithmetic-mean Hann epoch PSD",
            "EEG Fpz-Cz",
            category_a,
            parameters={
                "fooof_version": installed_fooof,
                **config.fooof_parameters,
            },
        ),
        "so_peak_frequency_hz": _summary(
            "so_peak_frequency_hz",
            hann_stats["so_peak_frequency_hz"],
            "Hz",
            config.so_psd_band_hz,
            "maximum of aggregate Welch PSD in SO band",
            "arithmetic mean across retained epoch PSDs",
            "EEG Fpz-Cz",
            category_a,
            parameters=psd.parameters,
        ),
        "relative_so_power": _summary(
            "relative_so_power",
            hann_stats["relative_so_power"],
            "1",
            config.so_psd_band_hz,
            "SO-band trapezoidal power / analysis-band power",
            "ratio from aggregate Hann PSD",
            "EEG Fpz-Cz",
            category_a,
            parameters={"denominator_band_hz": config.psd_band_hz},
        ),
        "so_q": _summary(
            "so_q",
            hann_stats["so_q"],
            "1",
            config.so_psd_band_hz,
            "SO peak PSD / mean adjacent-band PSD",
            "computed from aggregate Hann PSD",
            "EEG Fpz-Cz",
            category_a,
        ),
        "so_event_rate_per_min": _summary(
            "so_event_rate_per_min",
            so_event_rate,
            "events/min",
            config.so_event_band_hz,
            "AASM-style DOWN-to-UP half-wave detector",
            "events summed across epochs / retained N3 minutes",
            "EEG Fpz-Cz",
            category_a,
            parameters=so["parameters"],
        ),
        "ibi_cv": _summary(
            "ibi_cv",
            ibi_cv,
            "1",
            config.so_event_band_hz,
            "coefficient of variation of within-epoch SO UP-event intervals",
            "all within-epoch intervals pooled; no boundary interval",
            "EEG Fpz-Cz",
            category_a,
            valid=ibi_valid,
            invalid_reason=None if ibi_valid else "fewer than two valid intervals",
            parameters=so["parameters"],
        ),
        "pac_up_down_ratio": _summary(
            "pac_up_down_ratio",
            pac["pac_up_down_ratio"],
            "1",
            config.pac_phase_band_hz,
            "spindle-envelope amplitude at UP phases / DOWN phases",
            "sample amplitudes pooled after per-epoch filtering",
            "EEG Fpz-Cz",
            category_b,
            valid=pac["valid"],
            invalid_reason=None if pac["valid"] else "PAC detector invalid",
            parameters=pac["parameters"],
        ),
        "spindle_density_per_min": _summary(
            "spindle_density_per_min",
            spindle_density,
            "events/min",
            config.spindle_band_hz,
            "sigma-band moving-RMS detector",
            "events summed across epochs / retained N3 minutes",
            "EEG Fpz-Cz",
            category_c,
            parameters=spindle["parameters"],
        ),
        "spindle_mean_duration_s": _summary(
            "spindle_mean_duration_s",
            spindle_mean_duration,
            "s",
            config.spindle_band_hz,
            "sigma-band moving-RMS detector",
            "arithmetic mean across detected events",
            "EEG Fpz-Cz",
            category_c,
            valid=spindle_duration_valid,
            invalid_reason=(
                None if spindle_duration_valid else "no spindle events detected"
            ),
            parameters=spindle["parameters"],
        ),
        "pac_mi": _summary(
            "pac_mi",
            pac["mi"],
            "1",
            config.pac_phase_band_hz,
            "Tort normalized-KL modulation index",
            "phase-bin amplitude pooled after per-epoch filtering",
            "EEG Fpz-Cz",
            category_c,
            valid=pac["valid"],
            invalid_reason=None if pac["valid"] else "PAC detector invalid",
            parameters=pac["parameters"],
        ),
        "preferred_phase_sin": _summary(
            "preferred_phase_sin",
            pac["preferred_phase_sin"],
            "1",
            config.pac_phase_band_hz,
            "sine of maximum-amplitude PAC phase-bin center",
            "from aggregate PAC phase-amplitude distribution",
            "EEG Fpz-Cz",
            category_c,
            valid=pac["valid"],
            invalid_reason=None if pac["valid"] else "PAC detector invalid",
            parameters=pac["parameters"],
        ),
        "preferred_phase_cos": _summary(
            "preferred_phase_cos",
            pac["preferred_phase_cos"],
            "1",
            config.pac_phase_band_hz,
            "cosine of maximum-amplitude PAC phase-bin center",
            "from aggregate PAC phase-amplitude distribution",
            "EEG Fpz-Cz",
            category_c,
            valid=pac["valid"],
            invalid_reason=None if pac["valid"] else "PAC detector invalid",
            parameters=pac["parameters"],
        ),
        "waveform_peak_to_peak_z": _summary(
            "waveform_peak_to_peak_z",
            waveform_peak_to_peak,
            "z",
            config.so_event_band_hz,
            "trough-locked mean of per-event z-scored SO waveforms",
            "arithmetic mean across within-epoch waveform snippets",
            "EEG Fpz-Cz",
            category_c,
            valid=waveform_valid,
            invalid_reason=None if waveform_valid else "no complete SO snippets",
            parameters=so["parameters"],
        ),
    }
    rejection_reason_counts = Counter(
        reason
        for reason_tuple in rejection_reasons.values()
        for reason in reason_tuple
    )
    config_path = PROJECT_ROOT / "S4_sbi/configs/observation_sc4001.yaml"
    provenance = {
        **source_metadata,
        "schema": "ObservationBundle-v0.1-observation-only",
        "random_seed": config.random_seed,
        "epoch_boundary_policy": (
            "native 30-second PSG epochs; no concatenation before filtering, "
            "event detection, or PAC"
        ),
        "qc_parameters": {
            "max_peak_to_peak_uv": config.max_peak_to_peak_uv,
            "min_std_uv": config.min_std_uv,
        },
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "welch_primary": "hann",
        "welch_sensitivity": "hamming",
        "legacy_document_window": "hamming",
        "window_discrepancy_note": (
            "Legacy prose says Hamming, while the fitting target implementation "
            "uses Hann. Hann remains primary in this observation."
        ),
        "fooof_version": installed_fooof,
        "config_file": config_path.name,
        "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
        "full_raw_eeg_serialized": False,
    }
    for diagnostic_name, diagnostic in (
        ("SO", so),
        ("spindle", spindle),
        ("PAC", pac),
    ):
        failed = len(diagnostic["failure_reasons"])
        if failed:
            warnings.append(
                f"{diagnostic_name} detector failed in {failed} retained epochs; "
                "see its validity mask and failure reasons"
            )
    diagnostics = {
        "mapped_stages": mapped_stages,
        "annotation_table": annotation_table,
        "qc_rows": qc_rows,
        "spectral_hann": hann_stats,
        "spectral_hamming": hamming_stats,
        "fooof": fooof_result,
        "slow_oscillation": so,
        "spindle": spindle,
        "pac": pac,
    }
    bundle = ObservationBundle(
        subject_id=config.subject_id,
        channel=source_metadata["channel"],
        fs_hz=float(source_metadata["fs_hz"]),
        epoch_duration_s=config.epoch_duration_s,
        raw_label_counts=dict(source_metadata["raw_label_counts"]),
        n3_epoch_indices=n3_indices.astype(int),
        retained_epoch_indices=retained_indices,
        rejected_epoch_indices=rejected_indices,
        rejection_reasons=rejection_reasons,
        segments=segments,
        psd=psd,
        summaries=summaries,
        provenance=provenance,
        warnings=warnings,
        diagnostics=diagnostics,
    )
    bundle.validate()
    return bundle


def select_representative_epochs(
    bundle: ObservationBundle, n_epochs: int = 4
) -> np.ndarray:
    """Select reproducible low/median/high-amplitude retained epochs."""
    if n_epochs < 1:
        raise ValueError("n_epochs must be positive")
    qc_by_index = {
        int(row["epoch_index"]): row for row in bundle.diagnostics["qc_rows"]
    }
    indices = np.asarray(bundle.retained_epoch_indices, dtype=int)
    amplitudes = np.asarray(
        [qc_by_index[int(index)]["peak_to_peak_uv"] for index in indices]
    )
    order = np.argsort(amplitudes)
    quantile_positions = np.linspace(0.15, 0.85, min(n_epochs, len(indices)))
    selected_positions = {
        int(round(quantile * (len(order) - 1))) for quantile in quantile_positions
    }
    selected = [int(indices[order[position]]) for position in selected_positions]
    if len(selected) < min(n_epochs, len(indices)):
        rng = np.random.default_rng(int(bundle.provenance["random_seed"]))
        remaining = [int(index) for index in indices if int(index) not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: min(n_epochs, len(indices)) - len(selected)])
    return np.asarray(selected[:n_epochs], dtype=int)


def metric_classification_rows() -> list[dict[str, str]]:
    """Document the observation-side role of every planned metric."""
    return [
        {
            "class": "A. inference-summary candidates",
            "metric": "normalized_psd_0p5_20",
            "observation_status": "computed in PSDResult.normalized_hann",
            "note": "scale invariant; Hann is primary",
        },
        {
            "class": "A. inference-summary candidates",
            "metric": "so_peak_frequency_hz",
            "observation_status": "computed",
            "note": "aggregate Hann PSD",
        },
        {
            "class": "A. inference-summary candidates",
            "metric": "relative_so_power / so_q",
            "observation_status": "computed",
            "note": "dimensionless SO summaries",
        },
        {
            "class": "A. inference-summary candidates",
            "metric": "so_event_rate_per_min / ibi_cv",
            "observation_status": "computed",
            "note": "within-epoch detector only",
        },
        {
            "class": "B. mechanism diagnostics",
            "metric": "T8 / T12 thalamic spindle diagnostics",
            "observation_status": "not observable from scalp EEG",
            "note": "simulator-side mechanism only",
        },
        {
            "class": "B. mechanism diagnostics",
            "metric": "V8a internal T13",
            "observation_status": "not observable from scalp EEG",
            "note": "simulator-side mechanism only",
        },
        {
            "class": "B. mechanism diagnostics",
            "metric": "pac_up_down_ratio",
            "observation_status": "computed",
            "note": "new name; T11_lag_ms is a legacy misnomer",
        },
        {
            "class": "C. held-out PPC candidates",
            "metric": "spindle density / duration",
            "observation_status": "computed",
            "note": "observable-channel 11-15 Hz RMS detector",
        },
        {
            "class": "C. held-out PPC candidates",
            "metric": "PAC MI / preferred-phase sin and cos",
            "observation_status": "computed",
            "note": "single-channel per-epoch filtering",
        },
        {
            "class": "C. held-out PPC candidates",
            "metric": "SO waveform morphology",
            "observation_status": "computed",
            "note": "trough-locked z-scored template",
        },
    ]
