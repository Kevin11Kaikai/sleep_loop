"""Independent, publication-safe panels for the SC4001 observation notebook."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .observation import select_representative_epochs
from .schemas import ObservationBundle


COLORS = {
    "observation": "#235789",
    "observation_light": "#8FB8DE",
    "retained": "#238636",
    "rejected": "#C33C54",
    "so": "#E9C46A",
    "spindle": "#7A5195",
    "neutral": "#5C677D",
}

PANEL_SPECS = {
    "Obs-a": ("Recording context and N3 selection", "obs_a_recording_context"),
    "Obs-b": ("Representative 30-second N3 EEG", "obs_b_representative_eeg"),
    "Obs-c": ("Epoch-level and aggregate PSD", "obs_c_psd"),
    "Obs-d": ("Slow-oscillation waveform morphology", "obs_d_so_morphology"),
    "Obs-e": ("SO timing and regularity", "obs_e_so_regularity"),
    "Obs-f": ("Observable spindle activity", "obs_f_spindle_activity"),
    "Obs-g": ("SO-spindle coupling / PAC", "obs_g_pac"),
    "Obs-h": ("QC and Observation summary table", "obs_h_summary_qc"),
}

SUMMARY_COLUMNS = [
    "field_name",
    "value",
    "unit",
    "frequency_band",
    "algorithm",
    "aggregation_method",
    "valid_epoch_count",
    "validity_status",
    "intended_role",
    "warnings",
]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "svg.fonttype": "none",
        }
    )


def _finalize(fig: plt.Figure) -> plt.Figure:
    fig.align_labels()
    fig.tight_layout()
    return fig


def _row_for_epoch(bundle: ObservationBundle, epoch_index: int) -> int:
    matches = np.flatnonzero(bundle.retained_epoch_indices == epoch_index)
    if len(matches) != 1:
        raise ValueError(f"epoch {epoch_index} is not a unique retained epoch")
    return int(matches[0])


def _finite(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _reason_counts(reasons: dict[int, tuple[str, ...]]) -> dict[str, int]:
    counts = Counter(reason for values in reasons.values() for reason in values)
    return dict(sorted(counts.items()))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def plot_obs_a_recording_context(
    bundle: ObservationBundle,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot the full-night hypnogram and retained/rejected N3 positions."""
    _style()
    stages = np.asarray(bundle.diagnostics["mapped_stages"], dtype=object)
    hours = np.arange(len(stages)) * bundle.epoch_duration_s / 3600.0
    stage_codes = {"Unknown": -1, "REM": 0, "W": 1, "N1": 2, "N2": 3, "N3": 4}
    coded = np.asarray([stage_codes.get(str(stage), -1) for stage in stages])
    retained_hours = bundle.retained_epoch_indices * bundle.epoch_duration_s / 3600
    rejected_hours = bundle.rejected_epoch_indices * bundle.epoch_duration_s / 3600

    fig, axes = plt.subplots(
        2, 1, figsize=(10.2, 5.4), gridspec_kw={"height_ratios": [2.1, 1.0]}
    )
    axes[0].step(hours, coded, where="post", color=COLORS["observation"])
    axes[0].scatter(
        retained_hours,
        np.full(len(retained_hours), 4.15),
        s=10,
        color=COLORS["retained"],
        label=f"retained N3 (n={len(retained_hours)})",
        zorder=3,
    )
    axes[0].scatter(
        rejected_hours,
        np.full(len(rejected_hours), 4.15),
        s=12,
        marker="x",
        color=COLORS["rejected"],
        label=f"rejected N3 (n={len(rejected_hours)})",
        zorder=3,
    )
    axes[0].set(
        title=f"{bundle.subject_id} | {bundle.channel} | 30-s native epochs",
        xlabel="Time from PSG start (h)",
        ylabel="Sleep stage",
        yticks=[-1, 0, 1, 2, 3, 4],
        yticklabels=["Unknown", "REM", "W", "N1", "N2", "N3"],
        ylim=(4.5, -1.5),
    )
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].legend(ncol=2, loc="upper right")

    table = bundle.diagnostics["annotation_table"]
    labels = [row["raw_label"] for row in table]
    annotation_counts = [row["annotation_events"] for row in table]
    bar_colors = [
        COLORS["so"] if row["aasm_label"] == "N3" else COLORS["neutral"]
        for row in table
    ]
    axes[1].bar(labels, annotation_counts, color=bar_colors)
    axes[1].set(
        title="Raw annotation labels (event records; Stage 3 and Stage 4 map to N3)",
        ylabel="Annotation events",
    )
    axes[1].tick_params(axis="x", rotation=25)
    for label in axes[1].get_xticklabels():
        label.set_horizontalalignment("right")

    panel_data = {
        "panel_id": "Obs-a",
        "total_epoch_count": int(len(stages)),
        "n3_epoch_count": int(len(bundle.n3_epoch_indices)),
        "retained_epoch_count": int(len(bundle.retained_epoch_indices)),
        "rejected_epoch_count": int(len(bundle.rejected_epoch_indices)),
        "raw_label_counts": dict(bundle.raw_label_counts),
        "mapping": {
            "Sleep stage 3": "N3",
            "Sleep stage 4": "N3",
            "3": "N3",
        },
        "epoch_duration_s": float(bundle.epoch_duration_s),
        "boundary_policy": bundle.provenance["epoch_boundary_policy"],
        "valid_epoch_count": int(len(bundle.retained_epoch_indices)),
    }
    return _finalize(fig), panel_data


def plot_obs_b_representative_eeg(
    bundle: ObservationBundle, n_epochs: int = 4
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot deterministic peak-to-peak quantile representatives."""
    _style()
    selected = select_representative_epochs(bundle, n_epochs=n_epochs)
    time_s = np.arange(bundle.segments.shape[1]) / bundle.fs_hz
    fig, axes = plt.subplots(
        len(selected), 1, figsize=(10.2, 1.75 * len(selected)), sharex=True
    )
    axes = np.atleast_1d(axes)
    qc_by_index = {
        int(row["epoch_index"]): row for row in bundle.diagnostics["qc_rows"]
    }
    selections = []
    for axis, epoch_index in zip(axes, selected):
        row = _row_for_epoch(bundle, int(epoch_index))
        axis.plot(time_s, bundle.segments[row], color=COLORS["observation"], lw=0.7)
        qc = qc_by_index[int(epoch_index)]
        axis.set_ylabel("EEG (uV)")
        axis.set_title(
            f"epoch {int(epoch_index)} | retained | "
            f"peak-to-peak={qc['peak_to_peak_uv']:.1f} uV",
            loc="left",
        )
        axis.axhline(0, color="#B6B6B6", lw=0.6)
        selections.append(
            {
                "epoch_index": int(epoch_index),
                "selection_metric": "peak_to_peak_uv",
                "peak_to_peak_uv": float(qc["peak_to_peak_uv"]),
                "status": "retained",
            }
        )
    axes[-1].set_xlabel("Time within native 30-s epoch (s)")
    axes[-1].set_xlim(0, bundle.epoch_duration_s)
    fig.suptitle(
        f"{bundle.subject_id} representative N3 EEG | {bundle.channel}",
        y=1.01,
    )
    panel_data = {
        "panel_id": "Obs-b",
        "selection_rule": (
            "fixed 0.15-0.85 quantiles of retained-epoch peak-to-peak amplitude; "
            "seeded tie fill"
        ),
        "random_seed": int(bundle.provenance["random_seed"]),
        "selected_epochs": selections,
        "unit": "uV",
        "valid_epoch_count": int(len(selected)),
        "contains_raw_samples": False,
    }
    return _finalize(fig), panel_data


def plot_obs_c_psd(
    bundle: ObservationBundle,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot epoch variability, aggregate Hann PSD, and Hamming sensitivity."""
    _style()
    psd = bundle.psd
    freq = psd.frequencies_hz
    mask = (freq >= 0.2) & (freq <= 20.0)
    rows = psd.epoch_psd_hann_uv2_hz[:, mask]
    quantiles = np.quantile(rows, [0.05, 0.25, 0.75, 0.95], axis=0)
    selected_epoch = int(select_representative_epochs(bundle, 1)[0])
    selected_row = _row_for_epoch(bundle, selected_epoch)
    hann = psd.aggregate_hann_uv2_hz
    hamming = psd.aggregate_hamming_uv2_hz
    hann_stats = bundle.diagnostics["spectral_hann"]
    hamming_stats = bundle.diagnostics["spectral_hamming"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    axes[0].fill_between(
        freq[mask],
        quantiles[0],
        quantiles[3],
        color=COLORS["observation_light"],
        alpha=0.25,
        label="5-95% epochs",
    )
    axes[0].fill_between(
        freq[mask],
        quantiles[1],
        quantiles[2],
        color=COLORS["observation_light"],
        alpha=0.55,
        label="25-75% epochs",
    )
    axes[0].plot(
        freq[mask],
        psd.epoch_psd_hann_uv2_hz[selected_row, mask],
        color=COLORS["neutral"],
        alpha=0.8,
        label=f"epoch {selected_epoch}",
    )
    axes[0].plot(
        freq[mask], hann[mask], color=COLORS["observation"], lw=2, label="mean Hann"
    )
    axes[0].axvspan(0.2, 1.5, color=COLORS["so"], alpha=0.25, label="SO band")
    axes[0].axvline(
        hann_stats["so_peak_frequency_hz"],
        color=COLORS["rejected"],
        ls="--",
        label=f"SO peak {hann_stats['so_peak_frequency_hz']:.2f} Hz",
    )
    axes[0].set(
        title="Epoch-level and aggregate PSD",
        xlabel="Frequency (Hz)",
        ylabel="PSD (uV^2/Hz)",
        yscale="log",
        xlim=(0.2, 20),
    )
    axes[0].legend(ncol=2)

    axes[1].plot(
        freq[mask],
        psd.normalized_hann[mask],
        color=COLORS["observation"],
        lw=2,
        label="Hann (primary)",
    )
    axes[1].plot(
        freq[mask],
        psd.normalized_hamming[mask],
        color=COLORS["rejected"],
        ls="--",
        label="Hamming (sensitivity)",
    )
    axes[1].axvspan(0.2, 1.5, color=COLORS["so"], alpha=0.25)
    axes[1].set(
        title="Window sensitivity (area-normalized)",
        xlabel="Frequency (Hz)",
        ylabel="Normalized density (1/Hz)",
        xlim=(0.2, 20),
    )
    axes[1].legend()
    sensitivity_corr = float(
        np.corrcoef(psd.normalized_hann[mask], psd.normalized_hamming[mask])[0, 1]
    )
    axes[1].text(
        0.98,
        0.96,
        f"r={sensitivity_corr:.4f}\n"
        f"SO peak: {hann_stats['so_peak_frequency_hz']:.2f}/"
        f"{hamming_stats['so_peak_frequency_hz']:.2f} Hz",
        ha="right",
        va="top",
        transform=axes[1].transAxes,
    )
    frequency_resolution = float(np.median(np.diff(freq)))
    panel_data = {
        "panel_id": "Obs-c",
        "primary_window": "hann",
        "legacy_document_window": "hamming",
        "sensitivity_window": "hamming",
        "segment_length_s": float(psd.parameters["segment_s"]),
        "overlap_s": float(psd.parameters["overlap_s"]),
        "frequency_resolution_hz": frequency_resolution,
        "aggregation": psd.parameters["epoch_aggregation"],
        "analysis_band_hz": [0.5, 20.0],
        "so_band_hz": [0.2, 1.5],
        "unit": psd.parameters["unit"],
        "hann_statistics": hann_stats,
        "hamming_statistics": hamming_stats,
        "hann_hamming_correlation": sensitivity_corr,
        "valid_epoch_count": int(len(bundle.retained_epoch_indices)),
    }
    return _finalize(fig), panel_data


def plot_obs_d_so_morphology(
    bundle: ObservationBundle,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot within-epoch SO event detection and trough-centered morphology."""
    _style()
    so = bundle.diagnostics["slow_oscillation"]
    valid_events = [event for event in so["events"] if event["waveform_valid"]]
    if not valid_events:
        raise RuntimeError("Obs-d requires at least one complete SO waveform")
    amplitudes = np.asarray([event["half_wave_amplitude_uv"] for event in valid_events])
    example = valid_events[int(np.argsort(amplitudes)[len(amplitudes) // 2])]
    row = _row_for_epoch(bundle, int(example["epoch_index"]))
    time_s = np.arange(bundle.segments.shape[1]) / bundle.fs_hz
    snippets = so["waveform_snippets_z"]
    display_rows = np.linspace(
        0, len(snippets) - 1, min(40, len(snippets)), dtype=int
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    axes[0].plot(time_s, so["filtered_uv"][row], color=COLORS["observation"])
    axes[0].scatter(
        example["down_sample"] / bundle.fs_hz,
        so["filtered_uv"][row, example["down_sample"]],
        color=COLORS["rejected"],
        label="trough / alignment",
        zorder=3,
    )
    axes[0].scatter(
        example["up_sample"] / bundle.fs_hz,
        so["filtered_uv"][row, example["up_sample"]],
        color=COLORS["retained"],
        label="subsequent peak",
        zorder=3,
    )
    axes[0].set(
        title=f"SO detector example | epoch {example['epoch_index']}",
        xlabel="Time within epoch (s)",
        ylabel="0.2-4 Hz EEG (uV)",
        xlim=(0, bundle.epoch_duration_s),
    )
    axes[0].legend()
    axes[0].text(
        0.02,
        0.96,
        f"half-wave threshold >= {so['parameters']['half_wave_uv']:.0f} uV",
        transform=axes[0].transAxes,
        va="top",
    )

    for waveform in snippets[display_rows]:
        axes[1].plot(
            so["waveform_time_s"],
            waveform,
            color=COLORS["observation_light"],
            alpha=0.18,
            lw=0.7,
        )
    axes[1].fill_between(
        so["waveform_time_s"],
        so["waveform_mean_z"] - so["waveform_sem_z"],
        so["waveform_mean_z"] + so["waveform_sem_z"],
        color=COLORS["observation_light"],
        alpha=0.5,
        label="mean +/- SEM",
    )
    axes[1].plot(
        so["waveform_time_s"],
        so["waveform_mean_z"],
        color=COLORS["observation"],
        lw=2,
    )
    axes[1].axvline(0, color=COLORS["rejected"], ls="--")
    axes[1].set(
        title=f"Trough-centered SO morphology (n={len(snippets)} events)",
        xlabel="Time from trough (s)",
        ylabel="Within-event z-score",
    )
    axes[1].legend()
    panel_data = {
        "panel_id": "Obs-d",
        "alignment": "negative SO peak (trough), within native epoch",
        "detector_parameters": so["parameters"],
        "detector_valid_epoch_count": int(np.sum(so["validity_mask"])),
        "waveform_valid_epoch_count": int(np.sum(so["waveform_validity_mask"])),
        "detector_invalid_epoch_count": int(np.sum(~so["validity_mask"])),
        "valid_event_count": int(len(snippets)),
        "boundary_excluded_event_count": int(
            so["waveform_boundary_excluded_count"]
        ),
        "waveform_exclusion_reasons": so["waveform_exclusion_reasons"],
        "example_epoch_index": int(example["epoch_index"]),
        "valid_epoch_count": int(np.sum(so["waveform_validity_mask"])),
    }
    return _finalize(fig), panel_data


def plot_obs_e_so_regularity(
    bundle: ObservationBundle,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot strictly within-epoch SO intervals and regularity."""
    _style()
    so = bundle.diagnostics["slow_oscillation"]
    valid_rows = [row for row in so["per_epoch"] if row["ibi_valid"]]
    ibis = _finite(so["ibi_s"])
    if not len(ibis) or not valid_rows:
        raise RuntimeError("Obs-e requires valid within-epoch SO intervals")
    ibi_cvs = _finite([row["ibi_cv"] for row in valid_rows])
    event_rates = _finite(
        [row["event_rate_per_min"] for row in so["per_epoch"] if row["detector_valid"]]
    )
    aggregate_cv = float(np.std(ibis) / np.mean(ibis))

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    axes[0].hist(ibis, bins=24, color=COLORS["observation"], alpha=0.85)
    axes[0].axvline(np.mean(ibis), color=COLORS["rejected"], ls="--", label="mean")
    axes[0].axvline(np.median(ibis), color=COLORS["so"], ls="-.", label="median")
    axes[0].set(title="Within-epoch SO intervals", xlabel="IBI (s)", ylabel="Count")
    axes[0].legend()
    axes[1].hist(event_rates, bins=18, color=COLORS["retained"], alpha=0.85)
    axes[1].set(
        title="SO event-rate variability",
        xlabel="Events/min per epoch",
        ylabel="Epoch count",
    )
    axes[2].hist(ibi_cvs, bins=18, color=COLORS["so"], edgecolor="#7A6A31")
    axes[2].axvline(aggregate_cv, color=COLORS["rejected"], ls="--")
    axes[2].set(
        title=f"Regularity | pooled IBI_CV={aggregate_cv:.3f}",
        xlabel="Within-epoch IBI_CV",
        ylabel="Valid epoch count",
    )
    invalid_reasons = Counter(
        row["invalid_reason"] for row in so["per_epoch"] if not row["ibi_valid"]
    )
    panel_data = {
        "panel_id": "Obs-e",
        "ibi_policy": "intervals computed only within each native 30-s epoch",
        "so_event_rate_per_min": float(
            len(so["events"])
            / (len(bundle.retained_epoch_indices) * bundle.epoch_duration_s / 60)
        ),
        "mean_ibi_s": float(np.mean(ibis)),
        "median_ibi_s": float(np.median(ibis)),
        "ibi_cv": aggregate_cv,
        "valid_ibi_count": int(len(ibis)),
        "valid_epoch_count": int(len(valid_rows)),
        "invalid_epoch_count": int(len(so["per_epoch"]) - len(valid_rows)),
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
    }
    return _finalize(fig), panel_data


def plot_obs_f_spindle_activity(
    bundle: ObservationBundle,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot the provisional real-EEG observable-channel spindle detector."""
    _style()
    spindle = bundle.diagnostics["spindle"]
    events = spindle["events"]
    if not events:
        raise RuntimeError("Obs-f requires at least one detected spindle event")
    edge_samples = int(round(2.0 * bundle.fs_hz))
    interior_events = [
        event
        for event in events
        if event["start_sample"] >= edge_samples
        and event["stop_sample"] <= bundle.segments.shape[1] - edge_samples
    ]
    example_pool = interior_events or events
    example_pool = sorted(
        example_pool, key=lambda event: event["peak_envelope_uv"]
    )
    example = example_pool[len(example_pool) // 2]
    row = _row_for_epoch(bundle, int(example["epoch_index"]))
    time_s = np.arange(bundle.segments.shape[1]) / bundle.fs_hz
    durations = _finite([event["duration_s"] for event in events])
    densities = _finite(
        [item["density_per_min"] for item in spindle["per_epoch"] if item["valid"]]
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.0))
    axes[0, 0].plot(
        time_s, spindle["filtered_uv"][row], color=COLORS["spindle"], lw=0.75
    )
    axes[0, 0].axvspan(
        example["start_sample"] / bundle.fs_hz,
        example["stop_sample"] / bundle.fs_hz,
        color=COLORS["so"],
        alpha=0.35,
        label="detected event",
    )
    axes[0, 0].set(
        title=f"11-15 Hz observable EEG | epoch {example['epoch_index']}",
        xlabel="Time within epoch (s)",
        ylabel="Band-passed EEG (uV)",
        xlim=(0, bundle.epoch_duration_s),
    )
    axes[0, 0].legend()
    axes[0, 1].plot(
        time_s, spindle["envelope_uv"][row], color=COLORS["observation"]
    )
    axes[0, 1].axhline(
        spindle["threshold_uv"][row],
        color=COLORS["rejected"],
        ls="--",
        label="per-epoch threshold",
    )
    axes[0, 1].axvspan(
        example["start_sample"] / bundle.fs_hz,
        example["stop_sample"] / bundle.fs_hz,
        color=COLORS["so"],
        alpha=0.35,
    )
    axes[0, 1].set(
        title="Moving-RMS envelope",
        xlabel="Time within epoch (s)",
        ylabel="RMS amplitude (uV)",
        xlim=(0, bundle.epoch_duration_s),
    )
    axes[0, 1].legend()
    axes[1, 0].hist(durations, bins=18, color=COLORS["spindle"], alpha=0.85)
    axes[1, 0].set(
        title=f"Spindle duration (n={len(durations)})",
        xlabel="Duration (s)",
        ylabel="Event count",
    )
    axes[1, 1].hist(densities, bins=18, color=COLORS["observation_light"])
    axes[1, 1].set(
        title=f"Epoch spindle density | mean={np.mean(densities):.2f}/min",
        xlabel="Events/min per epoch",
        ylabel="Epoch count",
    )
    fig.suptitle(
        "Provisional real-EEG detector; not V8a T13 or thalamic T8/T12",
        y=1.01,
    )
    panel_data = {
        "panel_id": "Obs-f",
        "status": spindle["status"],
        "source": f"{bundle.subject_id}/{bundle.channel}",
        "detector_parameters": spindle["parameters"],
        "event_count": int(len(events)),
        "example_selection_rule": (
            "median peak envelope among events at least 2 s from an epoch edge"
        ),
        "spindle_density_per_min": float(
            len(events)
            / (len(bundle.retained_epoch_indices) * bundle.epoch_duration_s / 60)
        ),
        "mean_duration_s": float(np.mean(durations)),
        "valid_epoch_count": int(np.sum(spindle["validity_mask"])),
        "invalid_epoch_count": int(np.sum(~spindle["validity_mask"])),
        "failure_reasons": spindle["failure_reasons"],
        "warnings": [
            "held_out_ppc_candidate; detector is not cross-signal calibrated",
            "does not equal V8a internal T13 or thalamic T8/T12",
        ],
    }
    return _finalize(fig), panel_data


def plot_obs_g_pac(
    bundle: ObservationBundle,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot SO phase, sigma amplitude, and observational PAC support."""
    _style()
    pac = bundle.diagnostics["pac"]
    valid_rows = [item for item in pac["per_epoch"] if item["valid"]]
    if not pac["valid"] or not valid_rows:
        raise RuntimeError("Obs-g requires valid PAC support")
    ordered = sorted(valid_rows, key=lambda item: item["mi"])
    example = ordered[len(ordered) // 2]
    row = _row_for_epoch(bundle, int(example["epoch_index"]))
    time_s = np.arange(bundle.segments.shape[1]) / bundle.fs_hz
    centers = pac["phase_bin_centers_rad"]
    width = float(np.diff(pac["phase_bin_edges_rad"])[0])
    per_epoch_mi = _finite([item["mi"] for item in valid_rows])

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.0))
    axes[0, 0].plot(
        time_s, pac["phase_rad_by_epoch"][row], color=COLORS["observation"]
    )
    axes[0, 0].set(
        title=f"SO phase | epoch {example['epoch_index']}",
        xlabel="Time within epoch (s)",
        ylabel="Phase (rad)",
        xlim=(0, bundle.epoch_duration_s),
        ylim=(-np.pi, np.pi),
    )
    axes[0, 1].plot(
        time_s, pac["amplitude_uv_by_epoch"][row], color=COLORS["spindle"]
    )
    trim_s = float(pac["parameters"]["edge_trim_s_per_epoch"])
    for axis in axes[0]:
        axis.axvspan(0, trim_s, color=COLORS["neutral"], alpha=0.13)
        axis.axvspan(
            bundle.epoch_duration_s - trim_s,
            bundle.epoch_duration_s,
            color=COLORS["neutral"],
            alpha=0.13,
        )
    axes[0, 1].set(
        title="10-14 Hz Hilbert amplitude",
        xlabel="Time within epoch (s)",
        ylabel="Amplitude (uV)",
        xlim=(0, bundle.epoch_duration_s),
    )
    axes[1, 0].bar(
        centers,
        pac["probability"],
        width=0.9 * width,
        color=COLORS["observation"],
    )
    axes[1, 0].axvline(
        pac["preferred_phase_rad"],
        color=COLORS["rejected"],
        ls="--",
        label=f"preferred={pac['preferred_phase_rad']:.2f} rad",
    )
    axes[1, 0].set(
        title=f"Phase-amplitude distribution | MI={pac['mi']:.6f}",
        xlabel="SO phase (rad)",
        ylabel="Normalized amplitude",
        xlim=(-np.pi, np.pi),
    )
    axes[1, 0].legend()
    axes[1, 1].hist(per_epoch_mi, bins=18, color=COLORS["so"])
    axes[1, 1].axvline(pac["mi"], color=COLORS["rejected"], ls="--")
    axes[1, 1].set(
        title="PAC MI across valid epochs",
        xlabel="Tort modulation index",
        ylabel="Epoch count",
    )
    panel_data = {
        "panel_id": "Obs-g",
        "parameters": pac["parameters"],
        "pac_mi": float(pac["mi"]),
        "pac_preferred_phase_rad": float(pac["preferred_phase_rad"]),
        "pac_preferred_phase_sin": float(pac["preferred_phase_sin"]),
        "pac_preferred_phase_cos": float(pac["preferred_phase_cos"]),
        "pac_up_down_ratio": float(pac["pac_up_down_ratio"]),
        "valid_sample_count": int(pac["valid_sample_count"]),
        "valid_epoch_count": int(pac["valid_epoch_count"]),
        "invalid_epoch_count": int(np.sum(~pac["validity_mask"])),
        "failure_reasons": pac["failure_reasons"],
        "warnings": [
            "single-channel observational PAC; held_out_ppc_candidate",
            "T11_lag_ms is a legacy misnomer; do not use as lag",
        ],
    }
    return _finalize(fig), panel_data


def plot_obs_h_summary_qc(
    bundle: ObservationBundle,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot QC accounting, rejection reasons, and the final summary table."""
    _style()
    counts = [
        len(bundle.diagnostics["mapped_stages"]),
        len(bundle.n3_epoch_indices),
        len(bundle.retained_epoch_indices),
        len(bundle.rejected_epoch_indices),
    ]
    labels = ["total", "N3", "retained", "rejected"]
    colors = [
        COLORS["neutral"],
        COLORS["so"],
        COLORS["retained"],
        COLORS["rejected"],
    ]
    reasons = _reason_counts(bundle.rejection_reasons)
    so_rows = bundle.diagnostics["slow_oscillation"]["per_epoch"]
    spindle_rows = bundle.diagnostics["spindle"]["per_epoch"]
    pac_rows = bundle.diagnostics["pac"]["per_epoch"]
    distributions = {
        "SO rate (/min)": _finite(
            [row["event_rate_per_min"] for row in so_rows if row["detector_valid"]]
        ),
        "IBI_CV": _finite([row["ibi_cv"] for row in so_rows if row["ibi_valid"]]),
        "Spindle (/min)": _finite(
            [row["density_per_min"] for row in spindle_rows if row["valid"]]
        ),
        "PAC MI": _finite([row["mi"] for row in pac_rows if row["valid"]]),
    }

    fig = plt.figure(figsize=(11.4, 7.0))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.35])
    ax_counts = fig.add_subplot(grid[0, 0])
    ax_reasons = fig.add_subplot(grid[0, 1])
    ax_distribution = fig.add_subplot(grid[0, 2])
    ax_table = fig.add_subplot(grid[1, :])
    ax_counts.bar(labels, counts, color=colors)
    ax_counts.set(title="Epoch accounting", ylabel="Epoch count")
    for index, value in enumerate(counts):
        ax_counts.text(index, value, str(value), ha="center", va="bottom")
    ax_reasons.barh(
        list(reasons) or ["none"],
        list(reasons.values()) or [0],
        color=COLORS["rejected"],
    )
    ax_reasons.set(title="N3 rejection reasons", xlabel="Epoch count")
    nonempty = [(name, values) for name, values in distributions.items() if len(values)]
    standardized = []
    for _, values in nonempty:
        median = float(np.median(values))
        iqr = float(np.subtract(*np.percentile(values, [75, 25])))
        scale = iqr if iqr > 1e-12 else max(float(np.std(values)), 1.0)
        standardized.append((values - median) / scale)
    ax_distribution.boxplot(
        standardized,
        tick_labels=[name for name, _ in nonempty],
        showfliers=False,
    )
    ax_distribution.set_title("Cross-epoch variability")
    ax_distribution.set_ylabel("(value - median) / IQR")
    ax_distribution.axhline(0, color="#B6B6B6", lw=0.7)
    ax_distribution.tick_params(axis="x", rotation=25)

    summary_rows = [row for row in bundle.summary_rows()]
    table_rows = [
        [
            row["field_name"],
            f"{row['value']:.5g}" if row["value"] is not None else "invalid",
            row["unit"],
            str(row["valid_epoch_count"]),
            row["validity_status"],
            row["intended_role"],
        ]
        for row in summary_rows
    ]
    ax_table.axis("off")
    table = ax_table.table(
        cellText=table_rows,
        colLabels=[
            "field_name",
            "value",
            "unit",
            "valid epochs",
            "validity",
            "intended_role",
        ],
        colWidths=[0.22, 0.10, 0.09, 0.10, 0.10, 0.27],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.18)
    ax_table.set_title("Observation summary table", pad=8)
    panel_data = {
        "panel_id": "Obs-h",
        "epoch_counts": dict(zip(labels, map(int, counts))),
        "rejection_reason_counts": reasons,
        "distribution_scaling": "per metric: (value - median) / IQR",
        "summary_rows": [
            {key: row[key] for key in SUMMARY_COLUMNS} for row in summary_rows
        ],
        "valid_epoch_count": int(len(bundle.retained_epoch_indices)),
    }
    return _finalize(fig), panel_data


def save_panel(
    fig: plt.Figure,
    bundle: ObservationBundle,
    panel_id: str,
    panel_data: dict[str, Any],
    output_root: str | Path,
    dpi: int = 180,
) -> dict[str, Any]:
    """Save one independent panel and return its manifest record."""
    if panel_id not in PANEL_SPECS:
        raise ValueError(f"unknown observation panel {panel_id}")
    title, stem = PANEL_SPECS[panel_id]
    output_root = Path(output_root)
    panel_dir = output_root / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    svg_path = panel_dir / f"{stem}.svg"
    png_path = panel_dir / f"{stem}.png"
    fig.savefig(svg_path, bbox_inches="tight", transparent=False)
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi, facecolor="white")
    algorithms = {
        key: value
        for key, value in panel_data.items()
        if key in {"parameters", "detector_parameters"}
    }
    record = {
        "panel_id": panel_id,
        "title": title,
        "source_subject": bundle.subject_id,
        "source_channel": bundle.channel,
        "config_sha256": bundle.provenance["config_sha256"],
        "algorithm_parameters": algorithms,
        "input_epoch_count": int(len(bundle.n3_epoch_indices)),
        "valid_epoch_count": int(panel_data.get("valid_epoch_count", 0)),
        "artifacts": [
            f"panels/{stem}.svg",
            f"panels/{stem}.png",
        ],
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "warnings": list(panel_data.get("warnings", [])),
        "panel_data": _json_safe(panel_data),
    }
    return _json_safe(record)


def write_observation_artifacts(
    bundle: ObservationBundle,
    panel_records: list[dict[str, Any]],
    output_root: str | Path,
) -> dict[str, Path]:
    """Write aggregate summaries, QC, and the panel manifest without raw EEG."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "observation_summary.csv"
    qc_path = output_root / "qc_summary.json"
    manifest_path = output_root / "panel_manifest.json"
    summary_rows = [
        {key: row[key] for key in SUMMARY_COLUMNS} for row in bundle.summary_rows()
    ]
    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(
        summary_path, index=False
    )
    qc_payload = {
        "schema": bundle.provenance["schema"],
        "subject_id": bundle.subject_id,
        "channel": bundle.channel,
        "fs_hz": bundle.fs_hz,
        "epoch_duration_s": bundle.epoch_duration_s,
        "total_epoch_count": int(len(bundle.diagnostics["mapped_stages"])),
        "n3_epoch_count": int(len(bundle.n3_epoch_indices)),
        "retained_epoch_count": int(len(bundle.retained_epoch_indices)),
        "rejected_epoch_count": int(len(bundle.rejected_epoch_indices)),
        "rejection_reason_counts": _reason_counts(bundle.rejection_reasons),
        "raw_label_counts": dict(bundle.raw_label_counts),
        "config_sha256": bundle.provenance["config_sha256"],
        "warnings": list(bundle.warnings),
        "contains_raw_samples": False,
    }
    manifest_payload = {
        "schema": "ObservationPanelManifest-v1",
        "source_identifier": f"{bundle.subject_id}/{bundle.channel}",
        "config_sha256": bundle.provenance["config_sha256"],
        "panel_count": int(len(panel_records)),
        "panels": panel_records,
    }
    qc_path.write_text(
        json.dumps(_json_safe(qc_payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(_json_safe(manifest_payload), indent=2, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    return {
        "observation_summary": summary_path,
        "qc_summary": qc_path,
        "panel_manifest": manifest_path,
    }
