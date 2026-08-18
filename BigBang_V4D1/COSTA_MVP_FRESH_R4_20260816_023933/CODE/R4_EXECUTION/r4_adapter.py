"""Night-1 N3 MNE adapter and role-isolated data steward."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import numpy as np
from scipy.signal import detrend, welch

from r4_core import activate_mechanistic_imports, finish_manifest, load_protocol_claims, new_output_child, write_canonical


PSG_RELATIVE = "data/sleep-edfx-cassette/SC4001E0-PSG.edf"
ANNOTATION_RELATIVE = "data/sleep-edfx-cassette/SC4001EC-Hypnogram.edf"
SUBJECT_PSEUDONYM = "SC4001"
NIGHT = "Night-1"
EPOCH_SECONDS = 30.0
BANDS = (("SO", 0.5, 1.0), ("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("spindle", 11.0, 16.0))


def guarded_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    if value != normalized:
        raise ValueError("Night-1 path must use forward slashes exactly")
    if str(PurePosixPath(normalized)) not in {PSG_RELATIVE, ANNOTATION_RELATIVE}:
        raise ValueError("only the two exact frozen Night-1 paths are permitted")
    if normalized not in {PSG_RELATIVE, ANNOTATION_RELATIVE}:
        raise ValueError("Night-1 path must use its exact canonical spelling")
    return normalized


def build_n3_epoch_metadata(annotations: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for annotation_index, (onset, duration, description) in enumerate(
        zip(annotations.onset, annotations.duration, annotations.description)
    ):
        if str(description) not in {"Sleep stage 3", "Sleep stage 4"}:
            continue
        onset_f = float(onset)
        duration_f = float(duration)
        if not math.isfinite(onset_f) or not math.isfinite(duration_f) or onset_f < 0.0 or duration_f < EPOCH_SECONDS:
            continue
        whole_epochs = int(math.floor((duration_f + 1e-9) / EPOCH_SECONDS))
        for chunk_index in range(whole_epochs):
            epoch_onset = onset_f + EPOCH_SECONDS * chunk_index
            onset_us = int(round(epoch_onset * 1_000_000.0))
            epoch_id = f"SC4001E0_N3_{onset_us:016d}"
            if epoch_id in seen:
                raise ValueError(f"duplicate N3 epoch identifier from annotation {annotation_index}")
            seen.add(epoch_id)
            rows.append(
                {
                    "subject_pseudonym": SUBJECT_PSEUDONYM,
                    "epoch_id": epoch_id,
                    "stage": "N3",
                    "night": NIGHT,
                    "onset_s": epoch_onset,
                }
            )
    rows.sort(key=lambda row: (float(row["onset_s"]), str(row["epoch_id"])))
    return rows


def comparison_features(signal: Sequence[float], sfreq: float) -> dict[str, float]:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("signal must be one-dimensional and nonempty")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal contains nonfinite values")
    if not math.isfinite(float(sfreq)) or float(sfreq) <= 40.0:
        raise ValueError("sampling frequency cannot resolve the frozen spectrum")
    nperseg = int(round(4.0 * float(sfreq)))
    if values.size < nperseg:
        raise ValueError("signal is too short for a 4-second Welch window")
    centered = detrend(values, type="linear")
    if float(np.ptp(centered)) <= 1e-15:
        raise ValueError("signal is flat")
    frequencies, psd = welch(
        centered,
        fs=float(sfreq),
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend=False,
        scaling="density",
    )
    total_mask = (frequencies >= 0.5) & (frequencies <= 20.0)
    total = float(np.trapezoid(psd[total_mask], frequencies[total_mask]))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("total 0.5-20 Hz power is invalid")
    result: dict[str, float] = {}
    for name, low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        power = float(np.trapezoid(psd[mask], frequencies[mask]))
        if not math.isfinite(power) or power <= 0.0:
            raise ValueError(f"{name} band power is invalid")
        result[name] = float(np.log10(power / total))
    return result


def choose_channel(channel_names: Sequence[str]) -> str:
    for expected in ("EEG Fpz-Cz", "EEG Pz-Oz"):
        if expected in channel_names:
            return expected
    raise ValueError("neither frozen EEG channel is present")


def _extract_role_rows(raw: object, split_rows: Sequence[object], metadata: Mapping[str, Mapping[str, object]], channel: str) -> dict[str, list[dict[str, object]]]:
    sfreq = float(raw.info["sfreq"])
    expected_samples = int(round(EPOCH_SECONDS * sfreq))
    valid: dict[str, list[dict[str, object]]] = {"FIT": [], "HELDOUT": []}
    excluded: dict[str, list[dict[str, object]]] = {"FIT": [], "HELDOUT": []}
    for split in split_rows:
        item = metadata[split.epoch_id]
        start = int(round(float(item["onset_s"]) * sfreq))
        stop = start + expected_samples
        reason: str | None = None
        signal: np.ndarray | None = None
        if start < 0 or stop > int(raw.n_times):
            reason = "OUT_OF_RANGE"
        else:
            data = np.asarray(raw.get_data(picks=[channel], start=start, stop=stop), dtype=np.float64)
            if data.shape != (1, expected_samples):
                reason = "TOO_SHORT"
            else:
                signal = data[0]
                if not np.all(np.isfinite(signal)):
                    reason = "NONFINITE"
                elif float(np.ptp(signal)) <= 1e-15:
                    reason = "FLAT"
        common = {"epoch_id": split.epoch_id, "stable_rank": int(split.stable_rank)}
        if reason is not None:
            excluded[split.role].append({**common, "reason": reason})
            continue
        try:
            features = comparison_features(signal, sfreq)
        except ValueError as exc:
            excluded[split.role].append({**common, "reason": f"FEATURE_INVALID:{exc}"})
            continue
        valid[split.role].append({**common, "features": features})
    return {"valid": valid, "excluded": excluded}


def steward(project_root: Path, output_root: Path, output_child: str, mne_module: object | None = None) -> Path:
    protocol, _claims, frozen_hashes = load_protocol_claims()
    activate_mechanistic_imports()
    guarded_relative_path(PSG_RELATIVE)
    guarded_relative_path(ANNOTATION_RELATIVE)
    if mne_module is None:
        import mne as mne_module
    version_parts = str(getattr(mne_module, "__version__", "")).split(".")
    if version_parts[:2] != ["1", "9"]:
        raise RuntimeError("the empirical adapter requires MNE 1.9.x exactly")

    psg_path = project_root.resolve() / Path(PSG_RELATIVE)
    annotation_path = project_root.resolve() / Path(ANNOTATION_RELATIVE)
    annotations = mne_module.read_annotations(annotation_path)
    epoch_metadata = build_n3_epoch_metadata(annotations)
    if len(epoch_metadata) < int(protocol["partitions"]["minimum_n3_epochs"]):
        raise RuntimeError("fewer than four eligible upstream N3 epochs")

    from partitions import build_single_subject_n3_split, n3_split_digest

    split_input = [{key: row[key] for key in ("subject_pseudonym", "epoch_id", "stage", "night")} for row in epoch_metadata]
    split = build_single_subject_n3_split(
        split_input, int(protocol["seeds"]["partition_seed"]), float(protocol["partitions"]["fit_fraction"])
    )
    raw = mne_module.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
    channel = choose_channel(raw.ch_names)
    by_id = {str(row["epoch_id"]): row for row in epoch_metadata}
    extracted = _extract_role_rows(raw, split, by_id, channel)

    directory = new_output_child(output_root, output_child)
    artifacts: dict[str, str] = {}
    role_hashes: dict[str, str] = {}
    for role, filename in (("FIT", "FIT_PAYLOAD.json"), ("HELDOUT", "HELDOUT_PAYLOAD.json")):
        payload = {
            "schema": "COSTA_R4_ROLE_PAYLOAD_V1",
            "role": role,
            "subject_pseudonym": SUBJECT_PSEUDONYM,
            "night": NIGHT,
            "channel": channel,
            "feature_order": [band[0] for band in BANDS],
            "feature_units": "dimensionless log10 relative power",
            "total_eligible_epochs": len(split),
            "assigned_epoch_count": sum(row.role == role for row in split),
            "valid_epochs": extracted["valid"][role],
            "excluded_epochs": extracted["excluded"][role],
            "partition_digest": n3_split_digest(split),
            "frozen_input_hashes": frozen_hashes,
        }
        role_hashes[role] = artifacts[filename] = write_canonical(directory / filename, payload)
    receipt = {
        "schema": "COSTA_R4_STEWARD_RECEIPT_V1",
        "raw_paths": [PSG_RELATIVE, ANNOTATION_RELATIVE],
        "mne_calls": ["mne.read_annotations(path)", "mne.io.read_raw_edf(path, preload=True, verbose=\"ERROR\")"],
        "partition_before_missingness": True,
        "partition_seed": int(protocol["seeds"]["partition_seed"]),
        "fit_fraction": float(protocol["partitions"]["fit_fraction"]),
        "partition_digest": n3_split_digest(split),
        "split": [asdict(row) for row in split],
        "role_payload_hashes": role_hashes,
        "role_valid_counts": {role: len(extracted["valid"][role]) for role in ("FIT", "HELDOUT")},
        "role_excluded_counts": {role: len(extracted["excluded"][role]) for role in ("FIT", "HELDOUT")},
    }
    artifacts["STEWARD_RECEIPT.json"] = write_canonical(directory / "STEWARD_RECEIPT.json", receipt)
    finish_manifest(directory, "DATA_STEWARD", artifacts, frozen_hashes)
    return directory
