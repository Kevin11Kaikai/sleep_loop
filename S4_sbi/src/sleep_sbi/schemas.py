"""Structured, observation-side schemas.

The bundle may contain raw epoch arrays in memory. Only
``publication_safe_dict`` is intended for serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObservationConfig:
    subject_id: str
    manifest_path: str
    channel_candidates: tuple[str, ...]
    epoch_duration_s: float
    random_seed: int
    max_peak_to_peak_uv: float
    min_std_uv: float
    welch_window: str
    sensitivity_window: str
    welch_segment_s: float
    welch_overlap_s: float
    psd_band_hz: tuple[float, float]
    so_psd_band_hz: tuple[float, float]
    so_event_band_hz: tuple[float, float]
    so_half_wave_uv: float
    so_duration_s: tuple[float, float]
    so_min_peak_distance_s: float
    waveform_half_window_s: float
    spindle_band_hz: tuple[float, float]
    spindle_rms_window_s: float
    spindle_threshold_sd: float
    spindle_duration_s: tuple[float, float]
    spindle_merge_gap_s: float
    pac_phase_band_hz: tuple[float, float]
    pac_amplitude_band_hz: tuple[float, float]
    pac_phase_bins: int
    filter_order: int
    filter_edge_trim_s: float
    fooof_version: str
    fooof_parameters: dict[str, Any]


@dataclass(frozen=True)
class SummaryMetric:
    name: str
    value: float
    unit: str
    band_hz: tuple[float, float] | None
    algorithm: str
    aggregation: str
    source_signal: str
    category: str
    valid: bool
    invalid_reason: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": float(self.value),
            "unit": self.unit,
            "band_hz": list(self.band_hz) if self.band_hz is not None else None,
            "algorithm": self.algorithm,
            "aggregation": self.aggregation,
            "source_signal": self.source_signal,
            "category": self.category,
            "valid": bool(self.valid),
            "invalid_reason": self.invalid_reason,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class PSDResult:
    frequencies_hz: np.ndarray
    epoch_psd_hann_uv2_hz: np.ndarray
    aggregate_hann_uv2_hz: np.ndarray
    epoch_psd_hamming_uv2_hz: np.ndarray
    aggregate_hamming_uv2_hz: np.ndarray
    normalized_hann: np.ndarray
    normalized_hamming: np.ndarray
    parameters: dict[str, Any]

    def validate(self, n_epochs: int) -> None:
        n_freq = len(self.frequencies_hz)
        expected = (n_epochs, n_freq)
        if self.epoch_psd_hann_uv2_hz.shape != expected:
            raise ValueError("Hann epoch PSD shape does not match retained epochs")
        if self.epoch_psd_hamming_uv2_hz.shape != expected:
            raise ValueError("Hamming epoch PSD shape does not match retained epochs")
        for array in (
            self.frequencies_hz,
            self.epoch_psd_hann_uv2_hz,
            self.aggregate_hann_uv2_hz,
            self.epoch_psd_hamming_uv2_hz,
            self.aggregate_hamming_uv2_hz,
            self.normalized_hann,
            self.normalized_hamming,
        ):
            if not np.isfinite(array).all():
                raise ValueError("PSD result contains non-finite values")


@dataclass
class ObservationBundle:
    subject_id: str
    channel: str
    fs_hz: float
    epoch_duration_s: float
    raw_label_counts: dict[str, int]
    n3_epoch_indices: np.ndarray
    retained_epoch_indices: np.ndarray
    rejected_epoch_indices: np.ndarray
    rejection_reasons: dict[int, tuple[str, ...]]
    segments: np.ndarray
    psd: PSDResult
    summaries: dict[str, SummaryMetric]
    provenance: dict[str, Any]
    warnings: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict, repr=False)

    def validate(self) -> None:
        samples_per_epoch = int(round(self.fs_hz * self.epoch_duration_s))
        if self.segments.ndim != 2:
            raise ValueError("segments must retain an explicit epoch dimension")
        if self.segments.shape != (len(self.retained_epoch_indices), samples_per_epoch):
            raise ValueError("segments do not preserve exact 30-second boundaries")
        if not np.isfinite(self.segments).all():
            raise ValueError("segments contain non-finite samples")
        retained = set(map(int, self.retained_epoch_indices))
        rejected = set(map(int, self.rejected_epoch_indices))
        n3 = set(map(int, self.n3_epoch_indices))
        if retained & rejected:
            raise ValueError("retained and rejected epoch sets overlap")
        if retained | rejected != n3:
            raise ValueError("retained + rejected does not equal the N3 epoch set")
        if set(self.rejection_reasons) != rejected:
            raise ValueError("each rejected epoch must have a rejection reason")
        self.psd.validate(len(self.retained_epoch_indices))
        required_metadata = {
            "unit",
            "algorithm",
            "aggregation",
            "source_signal",
            "category",
        }
        for name, metric in self.summaries.items():
            if name != metric.name:
                raise ValueError(f"summary dictionary key mismatch for {name}")
            payload = metric.as_dict()
            missing = [key for key in required_metadata if not payload.get(key)]
            if missing:
                raise ValueError(f"summary {name} lacks metadata: {missing}")
            if metric.valid and not np.isfinite(metric.value):
                raise ValueError(f"valid summary {name} is non-finite")

    def summary_rows(self) -> list[dict[str, Any]]:
        return [self.summaries[name].as_dict() for name in sorted(self.summaries)]

    def publication_safe_dict(self) -> dict[str, Any]:
        """Return aggregate summaries and QC metadata, never raw sample arrays."""
        return {
            "schema": "ObservationBundle-v0.1-observation-only",
            "subject_id": self.subject_id,
            "channel": self.channel,
            "fs_hz": float(self.fs_hz),
            "epoch_duration_s": float(self.epoch_duration_s),
            "raw_label_counts": dict(self.raw_label_counts),
            "n_n3_epochs": int(len(self.n3_epoch_indices)),
            "n_retained_epochs": int(len(self.retained_epoch_indices)),
            "n_rejected_epochs": int(len(self.rejected_epoch_indices)),
            "rejection_reason_counts": dict(
                self.provenance.get("rejection_reason_counts", {})
            ),
            "summaries": self.summary_rows(),
            "provenance": dict(self.provenance),
            "warnings": list(self.warnings),
        }
