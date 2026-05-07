"""
compute_xobs_from_eeg.py
========================
Extract the len(SUMMARY_KEYS)-dim summary statistic vector x_obs from SC4001's real N3 EEG.

x_obs is the SBI inference target. The summaries mirror those computed by
V7's compute_constraints_v7, so the neural posterior maps simulated stats to
observed ones on a common scale.

FIRING-RATE PROXY
-----------------
V7 summaries are computed on r_ctx(t): firing rate >= 0, peaks ~60 Hz, narrow
UP pulses surrounded by near-zero DOWN baseline. Scalp EEG is bipolar (signed)
and in µV. The bridge:

    detrend → abs → 50ms Gaussian smooth → rescale to [0, 60]

This preserves the *envelope statistics* of the EEG (SO rate, IBI regularity,
spindle burstiness, SO-spindle coupling) without implying a physical
correspondence between EEG voltage and firing rate.

The SBI posterior should be interpreted as:
  "parameters whose simulated r_ctx envelope reproduces the same oscillatory
   statistics as the observed EEG envelope"
NOT as parameters that match the EEG voltage waveform directly.

SUMMARY_KEYS (same order as simulator_wrapper.SUMMARY_KEYS; currently 7-dim):
  0: shape_r        — hardcoded 1.0 (EEG matches itself by definition)
  1: T4_q           — SO peak Q-factor (peak/neighbor-band power ratio)
  2: T4_freq        — SO peak frequency [Hz]
  3: T6_ibi_cv      — UP-burst inter-event interval coefficient of variation
  4: T8_n_sp_events — spindle events per 60 s (normalized from proxy duration)
  5: T11_lag_ms     — up_down_ratio from PAC (named T11_lag_ms to match V7)
  6: MI             — PAC Modulation Index (cycle-by-cycle, fixed version)

NOTE on T11: V7 stores up_down_ratio under key "T11_lag_ms" in
compute_constraints_v7; we follow the same convention here.

NOTE on PAC for EEG: V7 computes PAC between r_ctx and r_thal. For scalp EEG,
thalamic activity is not directly observable. We pass r_proxy as both ctx and
thal signals. The resulting MI and up_down_ratio differ from simulation-based
values in scale; the SBI density estimator learns the mapping.

Usage (from project root):
    conda activate neurolib
    python S4_sbi/compute_xobs_from_eeg.py

Saves:
    S4_sbi/x_obs.npz  (keys: values [float32 array], keys [str list],
                              extraction_metadata [JSON string])
"""

import sys
import os
import json
import warnings
import importlib.util
from math import gcd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CWD must be project root ──────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
os.chdir(str(_ROOT))
sys.path.insert(0, str(_ROOT))

# ── NumPy alias shim — BEFORE any neurolib / mne import ──────────────────────
import numpy as np
import builtins as _builtins_mod
for _alias in ("int", "float", "bool", "object", "complex", "str"):
    if not hasattr(np, _alias):
        setattr(np, _alias, getattr(_builtins_mod, _alias))

# ── Local neurolib takes precedence over system install ───────────────────────
_LOCAL_NEUROLIB = Path(r"D:\Year3_Mao_Projects\neurolib")
if _LOCAL_NEUROLIB.is_dir() and str(_LOCAL_NEUROLIB) not in sys.path:
    sys.path.insert(0, str(_LOCAL_NEUROLIB))

# ── Standard imports (after shim) ────────────────────────────────────────────
import pandas as pd
from scipy.signal import welch, detrend, butter, sosfiltfilt, hilbert, resample_poly
from scipy.ndimage import gaussian_filter1d
import mne
mne.set_log_level("WARNING")

# ── Load 02_preprocess_psd via importlib (filename starts with digit) ─────────
_prep_spec = importlib.util.spec_from_file_location(
    "preprocess_psd", str(_ROOT / "utils" / "02_preprocess_psd.py")
)
_prep_mod = importlib.util.module_from_spec(_prep_spec)
_prep_spec.loader.exec_module(_prep_mod)
load_hypnogram   = _prep_mod.load_hypnogram
compute_epoch_psd = _prep_mod.compute_epoch_psd
EPOCH_LEN_S      = _prep_mod.EPOCH_LEN_S  # 30 s

# ── Load fixed PAC metrics via sys.path ───────────────────────────────────────
_repair_dir = _ROOT / "S4_v7_repair"
if str(_repair_dir) not in sys.path:
    sys.path.insert(0, str(_repair_dir))
from compute_pac_metrics_fixed import compute_pac_metrics  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# Config (mirrors s4_personalize_fig7_v7.py constants)
# ═══════════════════════════════════════════════════════════════════════════════
SUBJECT_ID      = "SC4001"
EEG_CHANNEL     = "EEG Fpz-Cz"          # same channel V7's load_target_psd uses
N3_LABELS       = ["N3"]                 # AASM label after R&K→AASM mapping
ARTIFACT_THRESH = 200e-6                 # V (200 µV peak-to-peak rejection)
FS_NATIVE       = 100.0                  # Sleep-EDF cassette native rate [Hz]
FS_SIM          = 1000.0                 # V7 simulation rate [Hz]

# V7 constraint thresholds (must match s4_personalize_fig7_v7.py exactly)
UP_THRESH_HZ         = 15.0
SO_FREQ_LO           = 0.2
SO_FREQ_HI           = 1.5
SPINDLE_LO           = 10.0
SPINDLE_HI           = 14.0
SPINDLE_DUR_LO_S     = 0.3
SPINDLE_DUR_HI_S     = 2.0
SPINDLE_EVT_PCTILE   = 75.0
SPINDLE_ENV_SMOOTH_MS = 200.0
T12_PEAK_INSIDE_RATIO = 1.5

SUMMARY_KEYS = [
    "shape_r", "T4_q", "T4_freq", "T6_ibi_cv",
    "T8_n_sp_events", "T11_lag_ms", "MI",
]

OUTPUT_PATH = _SCRIPT_DIR / "x_obs.npz"


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load N3 EEG
# ═══════════════════════════════════════════════════════════════════════════════

def load_n3_eeg():
    """
    Load and concatenate artifact-clean N3 epochs from SC4001.
    Returns: (eeg_uv, fs_native) where eeg_uv is 1D float64 array in µV.
    """
    try:
        manifest = pd.read_csv("data/manifest.csv", encoding="utf-8")
    except UnicodeDecodeError:
        manifest = pd.read_csv("data/manifest.csv", encoding="utf-16")

    subj_row = manifest[manifest["subject_id"] == SUBJECT_ID].iloc[0]

    raw = mne.io.read_raw_edf(
        subj_row["psg_path"], include=[EEG_CHANNEL], preload=True, verbose=False
    )
    fs = raw.info["sfreq"]

    stages = load_hypnogram(Path(subj_row["hypnogram_path"]))
    data_uv = raw.get_data()[0] * 1e6  # V → µV

    n_per_epoch = int(EPOCH_LEN_S * fs)
    n_epochs = min(len(stages), len(data_uv) // n_per_epoch)

    accepted, n_n3, n_rej = [], 0, 0
    for i in range(n_epochs):
        if stages[i] not in N3_LABELS:
            continue
        n_n3 += 1
        epoch = data_uv[i * n_per_epoch : (i + 1) * n_per_epoch]
        if np.ptp(epoch) > ARTIFACT_THRESH * 1e6:
            n_rej += 1
            continue
        accepted.append(epoch)

    if not accepted:
        raise RuntimeError(
            f"No N3 epochs passed artifact rejection for {SUBJECT_ID}"
        )

    print(f"  EEG: {n_n3} N3 epochs total, {n_rej} artifact-rejected, "
          f"{len(accepted)} accepted  ({sum(len(e) for e in accepted)/fs:.0f} s)")
    print(f"  Native fs = {fs} Hz")

    return np.concatenate(accepted), float(fs)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Build firing-rate proxy
# ═══════════════════════════════════════════════════════════════════════════════

def build_rate_proxy(eeg_uv, fs_from):
    """
    Map EEG voltage to a non-negative rate-like signal in [0, 60] at FS_SIM.

    Pipeline: resample → detrend → abs → 50ms Gaussian smooth → rescale.
    """
    # Resample to FS_SIM (100 → 1000 Hz)
    g = gcd(int(FS_SIM), int(fs_from))
    eeg_1k = resample_poly(eeg_uv, int(FS_SIM) // g, int(fs_from) // g)

    # Linear detrend
    eeg_det = detrend(eeg_1k, type="linear")

    # Rectify
    r = np.abs(eeg_det)

    # 50ms Gaussian smooth (σ = 50 samples at 1000 Hz).
    # Critical: without this, abs() produces V-shaped spikes at every
    # zero-crossing that corrupt SO UP-peak detection.
    r_smooth = gaussian_filter1d(r, sigma=50.0)

    # Rescale to [0, 60] (mirrors V7 UP-state firing rate range)
    r_proxy = r_smooth - r_smooth.min()
    p95 = np.percentile(r_proxy, 95)
    if p95 < 1e-9:
        raise RuntimeError("EEG proxy 95th-percentile ≈ 0 — check data quality")
    r_proxy = r_proxy / p95 * 60.0

    print(f"  r_proxy: {len(r_proxy)} samples  "
          f"mean={r_proxy.mean():.2f}  max={r_proxy.max():.2f}  "
          f"95pct={np.percentile(r_proxy, 95):.2f}  fs={FS_SIM} Hz")
    return r_proxy


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Compute the 8 summary statistics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_summaries(r_proxy, fs=FS_SIM):
    """
    Compute all 8 summary stats from r_proxy, reusing V7's constraint logic.
    """
    d = {}

    # ── shape_r = 1.0 ─────────────────────────────────────────────────────────
    # The EEG matches itself perfectly; this is the reference value.
    # In simulations, shape_r is computed as the FOOOF Pearson-r against the
    # target PSD. Setting x_obs[shape_r]=1.0 anchors the posterior to
    # "simulate something that looks like this spectrum."
    d["shape_r"] = 1.0

    # ── T4: SO peak frequency and Q-factor ────────────────────────────────────
    # Exact replica of compute_constraints_v7 T4 block.
    f_c, p_c = compute_epoch_psd(r_proxy, fs)
    so_mask     = (f_c >= SO_FREQ_LO) & (f_c <= SO_FREQ_HI)
    so_width    = SO_FREQ_HI - SO_FREQ_LO
    neigh_lo    = (f_c >= max(0.1, SO_FREQ_LO - so_width)) & (f_c < SO_FREQ_LO)
    neigh_hi    = (f_c > SO_FREQ_HI) & (f_c <= SO_FREQ_HI + so_width)
    so_peak_freq, so_q = 0.0, 0.0
    if so_mask.any():
        so_peak_freq = float(f_c[so_mask][np.argmax(p_c[so_mask])])
        so_peak_val  = float(p_c[so_mask].max())
        nbrs = np.concatenate([
            p_c[neigh_lo] if neigh_lo.any() else np.array([]),
            p_c[neigh_hi] if neigh_hi.any() else np.array([]),
        ])
        if len(nbrs) > 0 and nbrs.mean() > 0:
            so_q = float(so_peak_val / nbrs.mean())
    d["T4_q"]    = round(so_q, 3)
    d["T4_freq"] = round(so_peak_freq, 3)

    # ── T6: IBI coefficient of variation ──────────────────────────────────────
    # r_proxy has sub-second amplitude fluctuations (spindles, etc.) that create
    # spurious UP events if thresholded directly. Apply 500ms additional smoothing
    # to isolate SO-envelope timescale before UP-event detection. The median
    # threshold gives ~50% UP duty cycle — consistent with sleep SO physiology.
    r_for_ibi = gaussian_filter1d(r_proxy, sigma=500.0)
    up_thresh  = float(np.percentile(r_for_ibi, 50))
    print(f"  T6 SO-envelope UP threshold: {up_thresh:.2f} Hz (500ms smooth, 50th pct)")
    above  = (r_for_ibi > up_thresh).astype(np.int8)
    diff_  = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(diff_ == 1)[0]
    n_bursts = len(starts)
    ibi_cv = 999.0
    if n_bursts >= 3:
        intervals = np.diff(starts) / fs
        ibi_cv = float(intervals.std() / (intervals.mean() + 1e-12))
    d["T6_ibi_cv"] = round(ibi_cv, 3)

    # ── T8: Spindle event count (normalized to events per 60 s) ─────────────
    # V7 simulates exactly 60 s of signal (after burn-in). The EEG proxy covers
    # duration_s seconds. Normalize here so x_obs and simulated T8 are on the
    # same scale (events / 60 s). For a 60 s simulation the factor is 1.0.
    duration_s = float(len(r_proxy)) / fs
    n_sp_events = 0
    try:
        sos = butter(4, [SPINDLE_LO, SPINDLE_HI], btype="band", fs=fs, output="sos")
        filtered = sosfiltfilt(sos, r_proxy)
        envelope = np.abs(hilbert(filtered))
        sigma_samp = SPINDLE_ENV_SMOOTH_MS * fs / 1000.0
        env_sm = gaussian_filter1d(envelope, sigma=sigma_samp)
        thresh  = np.percentile(env_sm, SPINDLE_EVT_PCTILE)
        ab_sp   = (env_sm > thresh).astype(np.int8)
        diff_sp = np.diff(np.concatenate(([0], ab_sp, [0])))
        sp_st   = np.where(diff_sp == 1)[0]
        sp_en   = np.where(diff_sp == -1)[0]
        durs    = (sp_en - sp_st) / fs
        valid   = (durs >= SPINDLE_DUR_LO_S) & (durs <= SPINDLE_DUR_HI_S)
        n_sp_events = int(valid.sum())
    except Exception as exc:
        print(f"  [warn] Spindle detection error: {exc}")
    t8_normalized = n_sp_events * (60.0 / duration_s)
    print(f"  T8 raw={n_sp_events}  duration={duration_s:.0f}s  "
          f"normalized (per 60s)={t8_normalized:.2f}")
    d["T8_n_sp_events"] = round(t8_normalized, 3)

    # ── T9-T11: PAC (cycle-by-cycle, fixed version) ───────────────────────────
    # Pass r_proxy as BOTH ctx and thal. The PAC function uses r_ctx for SO
    # phase detection and r_thal for spindle envelope. Thalamic activity is not
    # directly observable in scalp EEG; this is a documented simplification.
    pac = compute_pac_metrics(r_proxy, r_proxy, fs=fs)
    d["T11_lag_ms"] = round(pac.get("up_down_ratio", 0.0), 3)  # V7 naming
    d["MI"]         = round(pac.get("mi", 0.0), 5)

    return d


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 — Sanity checks (HARD STOPS — do not add fallbacks)
# ═══════════════════════════════════════════════════════════════════════════════

def run_sanity_checks(d):
    """Return list of failure messages (empty = all passed)."""
    failures = []
    if not (0.5 <= d["T4_freq"] <= 1.5):
        failures.append(f"T4_freq = {d['T4_freq']:.3f} not in [0.5, 1.5] Hz")
    if not (0.1 <= d["T6_ibi_cv"] <= 0.8):
        failures.append(f"T6_ibi_cv = {d['T6_ibi_cv']:.3f} not in [0.1, 0.8]")
    if d["T11_lag_ms"] < 1.0:
        failures.append(
            f"T11_lag_ms (up_down_ratio) = {d['T11_lag_ms']:.3f} < 1.0"
        )
    if d["T8_n_sp_events"] <= 5:
        failures.append(f"T8_n_sp_events = {d['T8_n_sp_events']} <= 5")
    return failures


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("compute_xobs_from_eeg.py  —  x_obs extraction for SBI")
    print("=" * 64)

    # Step 1
    print("\n[Step 1]  Loading N3 EEG for SC4001 ...")
    eeg_uv, fs_native = load_n3_eeg()
    print(f"  Total N3 signal: {len(eeg_uv) / fs_native:.1f} s")

    # Step 2
    print("\n[Step 2]  Building firing-rate proxy ...")
    r_proxy = build_rate_proxy(eeg_uv, fs_native)

    # Step 3
    print("\n[Step 3]  Computing 7 summary statistics ...")
    d = compute_summaries(r_proxy)

    print("\n  x_obs values:")
    for k in SUMMARY_KEYS:
        print(f"    {k:20s} = {d[k]}")

    # Step 4
    print("\n[Step 4]  Sanity checks ...")
    failures = run_sanity_checks(d)
    if failures:
        print("\n  *** SANITY CHECK FAILED — DO NOT PROCEED ***")
        for msg in failures:
            print(f"    FAIL: {msg}")
        raise RuntimeError(
            "x_obs sanity checks failed. Review data / proxy parameters "
            "before launching SBI."
        )
    print("  All 4 sanity checks PASSED.")

    # Step 5
    x_obs_values = np.array([d[k] for k in SUMMARY_KEYS], dtype=np.float32)
    metadata = {
        "subject_id":    SUBJECT_ID,
        "eeg_channel":   EEG_CHANNEL,
        "n3_labels":     N3_LABELS,
        "proxy_method":  "detrend -> abs -> 50ms_gaussian_smooth -> rescale_to_[0,60]",
        "fs_native_hz":  float(fs_native),
        "fs_resampled_hz": FS_SIM,
        "n_samples_proxy": int(len(r_proxy)),
        "duration_s":    float(len(r_proxy) / FS_SIM),
        "artifact_thresh_uv": ARTIFACT_THRESH * 1e6,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(OUTPUT_PATH),
        values=x_obs_values,
        keys=SUMMARY_KEYS,
        extraction_metadata=json.dumps(metadata),
    )
    print(f"\n[Step 5]  Saved to {OUTPUT_PATH}")
    print(f"  x_obs  shape={x_obs_values.shape}  dtype={x_obs_values.dtype}")
    return x_obs_values


if __name__ == "__main__":
    main()
