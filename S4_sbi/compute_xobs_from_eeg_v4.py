"""
compute_xobs_from_eeg_v4.py
============================
7-dimensional x_obs extraction for SBI Stage 2 (Sprint 1 Phase 2).

SUMMARY_KEYS (7-dim, same order as simulator_wrapper.SUMMARY_KEYS post-Phase-2):
  0: shape_r        — fixed 1.0 (design sentinel; see sbi_report_0511.md §4.3)
  1: T4_q           — SO peak Q-factor on r_proxy Welch PSD
  2: T4_freq        — SO peak frequency [Hz] on r_proxy Welch PSD
  3: T6_ibi_cv      — EEG-native AASM SO UP IBI CV (NEW: bandpass + 75uV)
  4: T8_n_sp_events — spindle events per 60 s on eeg_raw
  5: T11_lag_ms     — up_down_ratio (EEG-native MI's UP/DOWN amp ratio)
  6: MI             — EEG-native Tort 2010 single-channel PAC (NEW)

Changes vs v3:
  - T6_ibi_cv reintroduced via compute_t6_eeg_aasm (was dropped from x_obs in v3)
  - MI reintroduced via compute_mi_eeg_native (was dropped from x_obs in v3)
  - T11_lag_ms switched from compute_pac_metrics_fixed.up_down_ratio
    (r_proxy phase × eeg_raw amp) to compute_mi_eeg_native.up_down_ratio
    (single-channel EEG-native, consistent with new MI)
  - shape_r / T4_q / T4_freq / T8_n_sp_events unchanged from v3

Rationale: see docs/sprint1_phase2_plan_v2.md §1-§2.

Usage (from project root):
    conda activate neurolib
    python S4_sbi/compute_xobs_from_eeg_v4.py
    python S4_sbi/compute_xobs_from_eeg_v4.py --output S4_sbi/x_obs_v4.npz
"""

import sys
import os
import json
import argparse
import warnings
import importlib.util
from math import gcd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CWD = project root ────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
os.chdir(str(_ROOT))
sys.path.insert(0, str(_ROOT))

# ── NumPy alias shim ─────────────────────────────────────────────────────────
import numpy as np
import builtins as _builtins_mod
for _alias in ("int", "float", "bool", "object", "complex", "str"):
    if not hasattr(np, _alias):
        setattr(np, _alias, getattr(_builtins_mod, _alias))

# ── Local neurolib ────────────────────────────────────────────────────────────
_LOCAL_NEUROLIB = Path(r"D:\Year3_Mao_Projects\neurolib")
if _LOCAL_NEUROLIB.is_dir() and str(_LOCAL_NEUROLIB) not in sys.path:
    sys.path.insert(0, str(_LOCAL_NEUROLIB))

import pandas as pd
from scipy.signal import detrend, resample_poly
from scipy.ndimage import gaussian_filter1d
import mne
mne.set_log_level("WARNING")

# ── 02_preprocess_psd via importlib ──────────────────────────────────────────
_prep_spec = importlib.util.spec_from_file_location(
    "preprocess_psd", str(_ROOT / "utils" / "02_preprocess_psd.py")
)
_prep_mod = importlib.util.module_from_spec(_prep_spec)
_prep_spec.loader.exec_module(_prep_mod)
load_hypnogram    = _prep_mod.load_hypnogram
compute_epoch_psd = _prep_mod.compute_epoch_psd
EPOCH_LEN_S       = _prep_mod.EPOCH_LEN_S

# ── EEG-native algorithms (new Phase 2 module) ────────────────────────────────
_repair_dir = _ROOT / "S4_v7_repair"
if str(_repair_dir) not in sys.path:
    sys.path.insert(0, str(_repair_dir))
from compute_pac_metrics_eeg_native import (
    compute_t6_eeg_aasm, compute_mi_eeg_native,
)

# Also import T8 spindle detection helpers (same logic as v3)
from scipy.signal import butter, sosfiltfilt, hilbert


# =============================================================================
# Configuration (must match s4_personalize_fig7_v7.py)
# =============================================================================
SUBJECT_ID            = "SC4001"
EEG_CHANNEL           = "EEG Fpz-Cz"
N3_LABELS             = ["N3"]
ARTIFACT_THRESH       = 200e-6
FS_NATIVE             = 100.0
FS_SIM                = 1000.0

SO_FREQ_LO            = 0.2
SO_FREQ_HI            = 1.5
SPINDLE_LO            = 10.0
SPINDLE_HI            = 14.0
SPINDLE_DUR_LO_S      = 0.3
SPINDLE_DUR_HI_S      = 2.0
SPINDLE_EVT_PCTILE    = 75.0
SPINDLE_ENV_SMOOTH_MS = 200.0

SUMMARY_KEYS = [
    "shape_r",
    "T4_q",
    "T4_freq",
    "T6_ibi_cv",
    "T8_n_sp_events",
    "T11_lag_ms",
    "MI",
]

DEFAULT_OUTPUT = str(_SCRIPT_DIR / "x_obs_v4.npz")


# =============================================================================
# Step 1 — Load N3 EEG (identical to v3)
# =============================================================================
def load_n3_eeg():
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
    data_uv = raw.get_data()[0] * 1e6

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
        raise RuntimeError(f"No N3 epochs passed artifact rejection for {SUBJECT_ID}")

    print(f"  EEG: {n_n3} N3 epochs total, {n_rej} rejected, "
          f"{len(accepted)} accepted  ({sum(len(e) for e in accepted)/fs:.0f} s)")
    print(f"  Native fs = {fs} Hz")
    return np.concatenate(accepted), float(fs)


# =============================================================================
# Step 2 — Build r_proxy + eeg_raw (identical to v3)
# =============================================================================
def build_rate_proxy(eeg_uv, fs_from):
    """
    Returns
    -------
    r_proxy : ndarray — abs->50ms-smooth->rescale [0,60], shape (N,)
              Used by T4 (Welch PSD on r_proxy).
    eeg_raw : ndarray — detrended EEG at FS_SIM [uV], shape (N,)
              Used by T6 (AASM), T8 (spindle), T11+MI (EEG-native PAC).
    """
    g = gcd(int(FS_SIM), int(fs_from))
    eeg_1k = resample_poly(eeg_uv, int(FS_SIM) // g, int(fs_from) // g)

    eeg_raw = detrend(eeg_1k, type="linear")

    r = np.abs(eeg_raw)
    r_smooth = gaussian_filter1d(r, sigma=50.0)
    r_proxy = r_smooth - r_smooth.min()
    p95 = np.percentile(r_proxy, 95)
    if p95 < 1e-9:
        raise RuntimeError("EEG proxy 95th-percentile ~0 — check data quality")
    r_proxy = r_proxy / p95 * 60.0

    print(f"  r_proxy: {len(r_proxy)} samples  "
          f"mean={r_proxy.mean():.2f}  max={r_proxy.max():.2f}  "
          f"95pct={np.percentile(r_proxy, 95):.2f}  fs={FS_SIM} Hz")
    print(f"  eeg_raw: {len(eeg_raw)} samples  "
          f"std={eeg_raw.std():.2f} uV  max={np.abs(eeg_raw).max():.2f} uV")
    return r_proxy, eeg_raw


# =============================================================================
# Step 3 — Compute 7 summary statistics
# =============================================================================
def compute_summaries(r_proxy, eeg_raw, fs=FS_SIM):
    d = {}

    # ── shape_r: fixed 1.0 (EEG vs EEG ref) ────────────────────────────────
    d["shape_r"] = 1.0

    # ── T4: SO peak freq + Q-factor on r_proxy Welch PSD ───────────────────
    f_c, p_c = compute_epoch_psd(r_proxy, fs)
    so_mask  = (f_c >= SO_FREQ_LO) & (f_c <= SO_FREQ_HI)
    so_width = SO_FREQ_HI - SO_FREQ_LO
    neigh_lo = (f_c >= max(0.1, SO_FREQ_LO - so_width)) & (f_c < SO_FREQ_LO)
    neigh_hi = (f_c > SO_FREQ_HI) & (f_c <= SO_FREQ_HI + so_width)
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

    # ── T6: EEG-native AASM SO UP IBI CV ──────────────────────────────────
    t6 = compute_t6_eeg_aasm(eeg_raw, fs)
    d["T6_ibi_cv"]   = round(t6["ibi_cv"], 4)
    d["_T6_n_neg"]   = t6["n_neg_peaks"]
    d["_T6_n_pos"]   = t6["n_pos_peaks"]
    d["_T6_n_up"]    = t6["n_up_events"]
    d["_T6_ok"]      = t6["ok"]
    d["_T6_mean_ibi_s"] = round(t6["mean_ibi_s"], 3)
    print(f"  T6 ibi_cv={d['T6_ibi_cv']}  n_up={t6['n_up_events']}  "
          f"n_neg={t6['n_neg_peaks']}  n_pos={t6['n_pos_peaks']}  "
          f"mean_ibi={t6['mean_ibi_s']:.2f}s  ok={t6['ok']}")

    # ── T8: spindle events on eeg_raw (same as v3) ────────────────────────
    duration_s = float(len(eeg_raw)) / fs
    n_sp_events = 0
    try:
        sos = butter(4, [SPINDLE_LO, SPINDLE_HI], btype="band", fs=fs, output="sos")
        filtered  = sosfiltfilt(sos, eeg_raw)
        envelope  = np.abs(hilbert(filtered))
        sigma_samp = SPINDLE_ENV_SMOOTH_MS * fs / 1000.0
        env_sm    = gaussian_filter1d(envelope, sigma=sigma_samp)
        thresh    = np.percentile(env_sm, SPINDLE_EVT_PCTILE)
        ab_sp     = (env_sm > thresh).astype(np.int8)
        diff_sp   = np.diff(np.concatenate(([0], ab_sp, [0])))
        sp_st     = np.where(diff_sp == 1)[0]
        sp_en     = np.where(diff_sp == -1)[0]
        durs      = (sp_en - sp_st) / fs
        valid     = (durs >= SPINDLE_DUR_LO_S) & (durs <= SPINDLE_DUR_HI_S)
        n_sp_events = int(valid.sum())
    except Exception as exc:
        print(f"  [warn] Spindle detection error: {exc}")
    t8_normalized = n_sp_events * (60.0 / duration_s)
    print(f"  T8 raw={n_sp_events}  duration={duration_s:.0f}s  "
          f"normalized={t8_normalized:.2f}/60s")
    d["T8_n_sp_events"] = round(t8_normalized, 3)

    # ── MI + T11 (up_down_ratio) from EEG-native single-channel PAC ───────
    mi_res = compute_mi_eeg_native(
        eeg_raw, fs,
        phase_band=(0.5, 1.5),
        amp_band=(SPINDLE_LO, SPINDLE_HI),
        n_phase_bins=18,
    )
    d["MI"]            = round(mi_res["mi"], 5)
    d["T11_lag_ms"]    = round(mi_res["up_down_ratio"], 4)
    d["_MI_pref_phase"] = round(mi_res["preferred_phase"], 3)
    d["_MI_ok"]         = mi_res["ok"]
    print(f"  MI mi={d['MI']}  up_down_ratio(T11)={d['T11_lag_ms']}  "
          f"pref_phase={d['_MI_pref_phase']} rad  ok={mi_res['ok']}")

    return d


# =============================================================================
# Step 4 — Sanity checks (per sprint1_phase2_plan_v2.md §4.3)
# =============================================================================
def run_sanity_checks(d):
    failures = []
    warnings_msgs = []
    if not (0.5 <= d["T4_freq"] <= 1.5):
        failures.append(f"T4_freq = {d['T4_freq']:.3f} not in [0.5, 1.5] Hz")
    if d["T11_lag_ms"] < 1.0:
        warnings_msgs.append(
            f"T11(up_down_ratio) = {d['T11_lag_ms']:.3f} < 1.0 — "
            "DOWN-locked? Verify before SBI."
        )
    if d["T8_n_sp_events"] <= 5:
        failures.append(f"T8_n_sp_events = {d['T8_n_sp_events']} <= 5")
    # T6 in physiological range [0.4, 0.55]
    if not d.get("_T6_ok", False):
        failures.append(f"T6 ok=False — EEG-native AASM detected no valid UP events")
    elif not (0.4 <= d["T6_ibi_cv"] <= 0.55):
        warnings_msgs.append(
            f"T6_ibi_cv = {d['T6_ibi_cv']:.4f} outside expected [0.40, 0.55] "
            "for healthy N3. Check t6 diagnostics."
        )
    # MI in physiological range [0.02, 0.05]
    if not d.get("_MI_ok", False):
        failures.append("MI ok=False")
    elif d["MI"] < 0.005:
        failures.append(
            f"MI = {d['MI']:.5f} below 0.005 — algorithm broken? "
            "Check Hilbert phase wrapping / bandpass design / phase anchor."
        )
    elif not (0.02 <= d["MI"] <= 0.05):
        warnings_msgs.append(
            f"MI = {d['MI']:.5f} outside expected [0.02, 0.05] for SC4001"
        )
    return failures, warnings_msgs


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output path for x_obs npz (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    print("=" * 64)
    print("compute_xobs_from_eeg_v4.py  --  7-dim x_obs extraction (EEG-native T6+MI)")
    print("Stats: shape_r | T4_q | T4_freq | T6_ibi_cv | T8 | T11(udr) | MI")
    print("=" * 64)

    print("\n[Step 1]  Loading N3 EEG for SC4001 ...")
    eeg_uv, fs_native = load_n3_eeg()
    print(f"  Total N3 signal: {len(eeg_uv) / fs_native:.1f} s")

    print("\n[Step 2]  Building r_proxy + eeg_raw ...")
    r_proxy, eeg_raw = build_rate_proxy(eeg_uv, fs_native)

    print("\n[Step 3]  Computing 7 summary statistics ...")
    d = compute_summaries(r_proxy, eeg_raw, fs=FS_SIM)

    print("\n  x_obs_v4 values:")
    for k in SUMMARY_KEYS:
        print(f"    {k:20s} = {d[k]}")

    print("\n[Step 4]  Sanity checks ...")
    failures, warnings_msgs = run_sanity_checks(d)
    for msg in warnings_msgs:
        print(f"    [warn] {msg}")
    if failures:
        print("\n  *** SANITY CHECK FAILED — DO NOT PROCEED ***")
        for msg in failures:
            print(f"    FAIL: {msg}")
        raise RuntimeError(
            "x_obs_v4 sanity checks failed. Review before launching SBI."
        )
    print("  All sanity checks PASSED.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x_obs_values = np.array([d[k] for k in SUMMARY_KEYS], dtype=np.float32)
    metadata = {
        "subject_id":       SUBJECT_ID,
        "eeg_channel":      EEG_CHANNEL,
        "version":          "v4",
        "n_dims":           7,
        "summary_keys":     SUMMARY_KEYS,
        "reintroduced":     ["T6_ibi_cv", "MI"],
        "reintroduction_method": (
            "T6: EEG-native AASM (bandpass 0.2-4 Hz, 75 µV half-wave, "
            "0.5-2.0 s duration); MI: Tort 2010 single-channel PAC on "
            "raw EEG (SO phase [0.5,1.5]Hz Hilbert, spindle amp [10,14]Hz "
            "Hilbert, 18 bins). See sprint1_phase2_plan_v2.md §1-2."
        ),
        "t6_diagnostics": {
            "n_neg_peaks":  d["_T6_n_neg"],
            "n_pos_peaks":  d["_T6_n_pos"],
            "n_up_events":  d["_T6_n_up"],
            "mean_ibi_s":   d["_T6_mean_ibi_s"],
            "ok":           d["_T6_ok"],
        },
        "mi_diagnostics": {
            "preferred_phase_rad": d["_MI_pref_phase"],
            "ok":                  d["_MI_ok"],
        },
        "fs_native_hz":     float(fs_native),
        "fs_resampled_hz":  FS_SIM,
        "n_samples_eeg_raw":int(len(eeg_raw)),
        "duration_s":       float(len(eeg_raw) / FS_SIM),
        "artifact_thresh_uv": ARTIFACT_THRESH * 1e6,
    }
    np.savez(
        str(out_path),
        values=x_obs_values,
        keys=SUMMARY_KEYS,
        extraction_metadata=json.dumps(metadata),
    )
    print(f"\n[Step 5]  Saved to {out_path}")
    print(f"  x_obs_v4  shape={x_obs_values.shape}  dtype={x_obs_values.dtype}")
    return x_obs_values


if __name__ == "__main__":
    main()
