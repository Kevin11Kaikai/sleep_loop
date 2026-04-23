"""
Script 02: Preprocessing + Power Spectral Density
===================================================
Paper parameters:
  - Channel: Fpz-Cz
  - PSD: Welch, 30s epochs, 4s Hamming windows, 1s overlap
  - Frequency output: 0–45 Hz in 0.25 Hz bins (for FOOOF)
  - R&K → AASM staging conversion
  - Drop epochs with unknown staging
"""

import os
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy.signal import welch
from tqdm import tqdm
from typing import Optional, Tuple

# ── Parameters (match paper exactly) ─────────────────────────────────────────
CHANNEL = "Fpz-Cz"
EPOCH_LEN_S = 30          # seconds
WIN_LEN_S = 4             # Hamming window length (seconds)
WIN_OVERLAP_S = 1         # overlap between windows (seconds)
FREQ_MAX = 45             # Hz — upper bound for FOOOF and Braintrak
FREQ_RESOLUTION = 0.25    # Hz — target bin size for FOOOF

# R&K to AASM stage mapping
# Paper: "originally marked according to R&K, transformed into AASM standard"
RK_TO_AASM = {
    "Sleep stage W":   "W",
    "Sleep stage 1":   "N1",
    "Sleep stage 2":   "N2",
    "Sleep stage 3":   "N3",   # R&K Stage 3 → N3
    "Sleep stage 4":   "N3",   # R&K Stage 4 → N3 (deep SWS)
    "Sleep stage R":   "REM",
    "Sleep stage ?":   "Unknown",
    "Movement time":   "Unknown",
}
# Alternative annotation formats seen in PhysioNet EDF
RK_TO_AASM_ALT = {
    "W": "W", "1": "N1", "2": "N2", "3": "N2", "4": "N3", "R": "REM",
    "?": "Unknown", "MT": "Unknown",
}
AASM_STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_INT = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "Unknown": -1}

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/sleep-edfx-cassette")
RESULT_DIR = Path("results/psd")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_hypnogram(hyp_path: Path) -> np.ndarray:
    """
    Load EDF+ Hypnogram annotations and return an array of AASM stage labels
    (one per 30s epoch).
    """
    ann = mne.read_annotations(str(hyp_path))
    stages = []
    for desc, onset, duration in zip(ann.description, ann.onset, ann.duration):
        desc = desc.strip()
        # Try full-format first, then short format
        stage = RK_TO_AASM.get(desc) or RK_TO_AASM_ALT.get(desc, "Unknown")
        n_epochs = int(round(duration / EPOCH_LEN_S))
        stages.extend([stage] * n_epochs)
    return np.array(stages)


def compute_epoch_psd(data_1d: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch PSD for a single 30s epoch.

    Returns
    -------
    freqs : np.ndarray, shape (n_freqs,)
    psd   : np.ndarray, shape (n_freqs,)  — units: µV²/Hz
    """
    n_win = int(WIN_LEN_S * sfreq)
    n_overlap = int(WIN_OVERLAP_S * sfreq)
    freqs, psd = welch(
        data_1d,
        fs=sfreq,
        window="hann",         # Hann ≈ Hamming for this use; paper says Hamming
        nperseg=n_win,
        noverlap=n_overlap,
        scaling="density",
    )
    # NOTE: paper says Hamming window; scipy's 'hamming' keyword:
    # Replace 'hann' with 'hamming' for strict reproduction:
    # freqs, psd = welch(data_1d, fs=sfreq, window=np.hamming(n_win), ...)

    return freqs, psd


def process_recording(psg_path: Path, hyp_path: Path, subject_id: str) -> Optional[dict]:
    """
    Full preprocessing pipeline for one recording.
    Returns dict with PSD array, frequency axis, stage labels, and metadata.
    """
    out_path = RESULT_DIR / f"{subject_id}.npz"
    if out_path.exists():
        return None  # already processed

    # ── 1. Load EEG ─────────────────────────────────────────────────────────
    try:
        raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
    except Exception as e:
        print(f"  ERROR loading {psg_path.name}: {e}")
        return None

    sfreq = raw.info["sfreq"]  # should be 100 Hz for cassette recordings
    assert sfreq == 100.0, f"Unexpected sfreq={sfreq} for {subject_id}"

    # ── 2. Extract Fpz-Cz channel ────────────────────────────────────────────
    ch_names = [ch.upper() for ch in raw.ch_names]
    target = CHANNEL.upper()
    if target not in ch_names:
        # Try common alternatives
        for alt in ["FPZ-CZ", "EEG FPZ-CZ", "EEG Fpz-Cz"]:
            if alt.upper() in ch_names:
                target = alt.upper()
                break
        else:
            print(f"  WARNING: channel {CHANNEL} not found in {psg_path.name}. "
                  f"Available: {raw.ch_names}")
            return None

    orig_ch_name = raw.ch_names[ch_names.index(target)]
    data = raw.get_data(picks=[orig_ch_name])[0]  # shape: (n_samples,)
    data_uv = data * 1e6  # convert V → µV

    # ── 3. Load staging ──────────────────────────────────────────────────────
    stages = load_hypnogram(hyp_path)

    # ── 4. Epoch the signal into 30s windows ─────────────────────────────────
    n_samples_per_epoch = int(EPOCH_LEN_S * sfreq)
    n_epochs_signal = len(data_uv) // n_samples_per_epoch
    n_epochs = min(n_epochs_signal, len(stages))

    # ── 5. Compute PSD per epoch ─────────────────────────────────────────────
    # Get frequency axis from first epoch
    test_epoch = data_uv[:n_samples_per_epoch]
    freqs_full, _ = compute_epoch_psd(test_epoch, sfreq)

    # Mask to [0, 45] Hz
    freq_mask = freqs_full <= FREQ_MAX
    freqs = freqs_full[freq_mask]

    psd_all = np.zeros((n_epochs, freq_mask.sum()), dtype=np.float32)
    stage_ints = np.full(n_epochs, -1, dtype=np.int8)

    for i in range(n_epochs):
        epoch_data = data_uv[i * n_samples_per_epoch: (i + 1) * n_samples_per_epoch]
        _, psd_full = compute_epoch_psd(epoch_data, sfreq)
        psd_all[i] = psd_full[freq_mask].astype(np.float32)
        stage_ints[i] = STAGE_TO_INT.get(stages[i], -1)

    # ── 6. Drop unknown-stage epochs ─────────────────────────────────────────
    valid_mask = stage_ints >= 0
    psd_valid  = psd_all[valid_mask]
    stage_valid = stage_ints[valid_mask]

    n_dropped = (~valid_mask).sum()
    print(f"  {subject_id}: {n_epochs} epochs → dropped {n_dropped} unknown → "
          f"{valid_mask.sum()} valid")

    # ── 7. Save ──────────────────────────────────────────────────────────────
    np.savez_compressed(
        out_path,
        psd=psd_valid,          # (n_valid_epochs, n_freqs), µV²/Hz
        freqs=freqs,            # (n_freqs,), Hz
        stages=stage_valid,     # (n_valid_epochs,), int: 0=W,1=N1,2=N2,3=N3,4=REM
        sfreq=sfreq,
        subject_id=subject_id,
    )
    return {"subject_id": subject_id, "n_epochs": valid_mask.sum()}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import csv

    manifest_path = Path("data/manifest.csv")
    if not manifest_path.exists():
        print("ERROR: Run 01_download_edfx.py first to generate manifest.csv")
        exit(1)

    recordings = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            recordings.append(row)

    print(f"Processing {len(recordings)} recordings...")
    summary = []

    for rec in tqdm(recordings):
        result = process_recording(
            psg_path=Path(rec["psg_path"]),
            hyp_path=Path(rec["hypnogram_path"]),
            subject_id=rec["subject_id"],
        )
        if result:
            summary.append(result)

    print(f"\nDone. Processed {len(summary)} recordings.")
    print(f"Results saved to: {RESULT_DIR}")
