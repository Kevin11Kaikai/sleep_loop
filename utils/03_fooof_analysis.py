"""
Script 03: FOOOF Analysis — Aperiodic + Periodic Decomposition
==============================================================
Paper parameters (Section 2.2.2):
  - FOOOF v1.0.0 (do NOT use specparam)
  - Range: 0–45 Hz, 0.25 Hz bins
  - No knee (aperiodic_mode='fixed')
  - max_n_peaks=4
  - Peak bandwidth: 1–4 Hz
  - Min peak height: 1 (µV²/Hz units)
  - Output: exponent, offset, peak (freq, power, bandwidth) per epoch

Also computes AUC (trapezoidal) for each standard band.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from fooof import FOOOF, FOOOFGroup
from scipy.integrate import trapezoid
from tqdm import tqdm

# ── Band definitions (paper Section 1) ───────────────────────────────────────
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 7.5),
    "alpha": (7.5, 12.0),
    "beta":  (16.0, 30.0),
    "gamma": (30.0, 45.0),
}

# ── FOOOF parameters (paper Section 2.2.2) ────────────────────────────────────
FOOOF_PARAMS = dict(
    peak_width_limits=(1, 4),
    min_peak_height=1.0,        # in PSD units (µV²/Hz for this dataset)
    max_n_peaks=4,
    aperiodic_mode="fixed",     # no knee
    verbose=False,
)
FREQ_RANGE = [0.5, 45]          # fit range (avoid DC component)

# ── Paths ─────────────────────────────────────────────────────────────────────
PSD_DIR    = Path("results/psd")
FOOOF_DIR  = Path("results/fooof")
FOOOF_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_band_auc(freqs: np.ndarray, psd: np.ndarray) -> dict:
    """Trapezoidal AUC for each standard frequency band."""
    auc = {}
    for band, (flo, fhi) in BANDS.items():
        mask = (freqs >= flo) & (freqs <= fhi)
        if mask.sum() > 1:
            auc[band] = trapezoid(psd[mask], freqs[mask])
        else:
            auc[band] = np.nan
    return auc


def run_fooof_on_recording(subject_id: str) -> pd.DataFrame:
    """
    Run FOOOF on all epochs for one recording.

    Returns DataFrame with columns:
      epoch_idx, stage, aperiodic_exponent, aperiodic_offset,
      peak0_freq, peak0_power, peak0_bw, ... (up to 4 peaks),
      auc_delta, auc_theta, auc_alpha, auc_beta, auc_gamma
    """
    out_path = FOOOF_DIR / f"{subject_id}.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)

    psd_file = PSD_DIR / f"{subject_id}.npz"
    if not psd_file.exists():
        print(f"  WARNING: PSD file not found for {subject_id}")
        return pd.DataFrame()

    data = np.load(psd_file)
    psd_all   = data["psd"]     # (n_epochs, n_freqs)
    freqs     = data["freqs"]
    stages    = data["stages"]
    n_epochs  = psd_all.shape[0]

    # Initialise FOOOF model (reuse for speed)
    fm = FOOOF(**FOOOF_PARAMS)

    rows = []
    for i in range(n_epochs):
        psd = psd_all[i]

        # ── FOOOF fit ────────────────────────────────────────────────────────
        try:
            fm.fit(freqs, psd, FREQ_RANGE)
            exp    = fm.aperiodic_params_[1]    # exponent (slope)
            offset = fm.aperiodic_params_[0]    # offset
            peaks  = fm.peak_params_            # (n_peaks, 3): CF, PW, BW
        except Exception:
            exp, offset, peaks = np.nan, np.nan, np.zeros((0, 3))

        # ── Pack peak data (up to 4 peaks) ───────────────────────────────────
        peak_data = {}
        for p in range(4):
            if p < len(peaks):
                peak_data[f"peak{p}_freq"]  = peaks[p, 0]
                peak_data[f"peak{p}_power"] = peaks[p, 1]
                peak_data[f"peak{p}_bw"]    = peaks[p, 2]
            else:
                peak_data[f"peak{p}_freq"]  = np.nan
                peak_data[f"peak{p}_power"] = np.nan
                peak_data[f"peak{p}_bw"]    = np.nan

        # ── Band AUC ─────────────────────────────────────────────────────────
        auc = compute_band_auc(freqs, psd)

        rows.append({
            "subject_id":          subject_id,
            "epoch_idx":           i,
            "stage":               int(stages[i]),
            "aperiodic_exponent":  float(exp),
            "aperiodic_offset":    float(offset),
            **peak_data,
            **{f"auc_{k}": float(v) for k, v in auc.items()},
        })

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    return df


def compute_fooof_peak_band_power(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each epoch, find the dominant FOOOF peak within each frequency band.
    Adds columns: fooof_peak_delta, fooof_peak_theta, etc.
    """
    for band, (flo, fhi) in BANDS.items():
        col = f"fooof_peak_{band}"
        df[col] = np.nan
        for p in range(4):
            freq_col  = f"peak{p}_freq"
            power_col = f"peak{p}_power"
            if freq_col not in df.columns:
                continue
            in_band = (df[freq_col] >= flo) & (df[freq_col] < fhi)
            # Take max power peak in band per epoch
            current = df[col].copy()
            candidate = df[power_col].where(in_band)
            df[col] = np.fmax(current, candidate)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    psd_files = sorted(PSD_DIR.glob("*.npz"))
    print(f"Running FOOOF on {len(psd_files)} recordings...")

    all_dfs = []
    for f in tqdm(psd_files):
        subj = f.stem
        df = run_fooof_on_recording(subj)
        if not df.empty:
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = compute_fooof_peak_band_power(combined)
        combined.to_parquet(FOOOF_DIR / "all_subjects.parquet", index=False)
        print(f"\nCombined FOOOF results: {len(combined)} epochs")
        print(combined.groupby("stage")["aperiodic_exponent"].describe())
    else:
        print("No results to combine.")
