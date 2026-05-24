"""
compute_xobs_from_eeg_v3.py
============================
5-dimensional x_obs extraction for SBI Stage 2.

SUMMARY_KEYS (5-dim, same order as simulator_wrapper.SUMMARY_KEYS):
  0: shape_r        — fixed 1.0 (design choice)
  1: T4_q           — SO peak Q-factor
  2: T4_freq        — SO peak frequency [Hz]
  3: T8_n_sp_events — spindle events per 60 s (from eeg_raw)
  4: T11_lag_ms     — up_down_ratio (PAC; SO-phase from r_proxy, spindle-amp from eeg_raw)

T6_ibi_cv and MI removed — diagnostic scan 2026-05-10.
Root cause: r_proxy envelope (abs -> 50ms Gaussian -> rescale) is structurally
incompatible with sharp UP/DOWN detection and cycle-by-cycle PAC phase interpolation.
No r_proxy threshold gives ibi_cv in [0.3, 0.6]; MI stays ~0.0001 regardless of
prominence_frac. Re-introduce only after Stage 3 EEG-native bandpass redesign.

Retained fixes from v2:
  - T8: spindle detection on eeg_raw (detrended 1000 Hz EEG, before abs/smooth)
  - T11: SO phase from r_proxy peaks; spindle amplitude from eeg_raw

Usage (from project root):
    conda activate neurolib
    python S4_sbi/compute_xobs_from_eeg_v3.py
    python S4_sbi/compute_xobs_from_eeg_v3.py --output S4_sbi/x_obs_v3.npz
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
from scipy.signal import welch, detrend, butter, sosfiltfilt, hilbert, resample_poly
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

# ── Fixed PAC metrics ─────────────────────────────────────────────────────────
_repair_dir = _ROOT / "S4_v7_repair"
if str(_repair_dir) not in sys.path:
    sys.path.insert(0, str(_repair_dir))
from compute_pac_metrics_fixed import compute_pac_metrics  # noqa: E402


# =============================================================================
# Configuration (must match s4_personalize_fig7_v7.py)
# =============================================================================
SUBJECT_ID            = "SC4001"
EEG_CHANNEL           = "EEG Fpz-Cz"
N3_LABELS             = ["N3"]
ARTIFACT_THRESH       = 200e-6          # V (200 uV peak-to-peak)
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

# 5-dim summary keys — T6_ibi_cv and MI dropped (see module docstring)
SUMMARY_KEYS = [
    "shape_r",
    "T4_q",
    "T4_freq",
    "T8_n_sp_events",
    "T11_lag_ms",
]

DEFAULT_OUTPUT = str(_SCRIPT_DIR / "x_obs_v3.npz")


# =============================================================================
# Step 1 — Load N3 EEG
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
# Step 2 — Build firing-rate proxy + preserve raw EEG
# =============================================================================
def build_rate_proxy(eeg_uv, fs_from):
    """
    Returns
    -------
    r_proxy : ndarray — abs->50ms-smooth->rescale [0,60], shape (N,)
    eeg_raw : ndarray — detrended EEG at FS_SIM [uV], shape (N,)
              Retains spindle-band content for T8 and T11 PAC.
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
# Step 3 — Compute 5 summary statistics
# =============================================================================
def compute_summaries(r_proxy, fs=FS_SIM, eeg_raw=None):
    """
    Compute 5 summary statistics from r_proxy (and eeg_raw for spindle stats).

    T6_ibi_cv and MI are intentionally absent — see module docstring for rationale.
    """
    d = {}

    # shape_r = 1.0 (EEG vs itself; anchors posterior toward high spectral fidelity)
    d["shape_r"] = 1.0

    # T4: SO peak frequency and Q-factor
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

    # T8: spindle events per 60 s — from eeg_raw (retains 10-14 Hz content)
    duration_s = float(len(r_proxy)) / fs
    n_sp_events = 0
    sp_signal = eeg_raw if eeg_raw is not None else r_proxy
    if eeg_raw is None:
        print("  [warn] eeg_raw not provided; T8 falls back to r_proxy (no spindle content)")
    try:
        sos = butter(4, [SPINDLE_LO, SPINDLE_HI], btype="band", fs=fs, output="sos")
        filtered  = sosfiltfilt(sos, sp_signal)
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
          f"normalized={t8_normalized:.2f}/60s  "
          f"[source: {'eeg_raw' if eeg_raw is not None else 'r_proxy (fallback)'}]")
    d["T8_n_sp_events"] = round(t8_normalized, 3)

    # T11: up_down_ratio — SO phase from r_proxy peaks, spindle amp from eeg_raw
    if eeg_raw is not None:
        pac = compute_pac_metrics(r_proxy, eeg_raw, fs=fs)
        print("  PAC: SO-phase from r_proxy, spindle-amp from eeg_raw")
    else:
        pac = compute_pac_metrics(r_proxy, r_proxy, fs=fs)
        print("  [warn] eeg_raw not provided; PAC uses r_proxy x r_proxy (v1 fallback)")
    d["T11_lag_ms"] = round(pac.get("up_down_ratio", 0.0), 3)
    print(f"  T11 up_down_ratio={d['T11_lag_ms']}  "
          f"n_so_cycles={pac.get('n_so_cycles', 0)}  ok={pac.get('ok')}")

    return d


# =============================================================================
# Step 4 — Sanity checks
# =============================================================================
def run_sanity_checks(d):
    failures = []
    if not (0.5 <= d["T4_freq"] <= 1.5):
        failures.append(f"T4_freq = {d['T4_freq']:.3f} not in [0.5, 1.5] Hz")
    if d["T11_lag_ms"] < 1.0:
        failures.append(f"T11_lag_ms (up_down_ratio) = {d['T11_lag_ms']:.3f} < 1.0")
    if d["T8_n_sp_events"] <= 5:
        failures.append(f"T8_n_sp_events = {d['T8_n_sp_events']} <= 5")
    return failures


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output path for x_obs npz (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    print("=" * 64)
    print("compute_xobs_from_eeg_v3.py  --  5-dim x_obs extraction")
    print("Stats: shape_r | T4_q | T4_freq | T8_n_sp_events | T11_lag_ms")
    print("Dropped: T6_ibi_cv (r_proxy incompatible), MI (r_proxy incompatible)")
    print("=" * 64)

    # Step 1
    print("\n[Step 1]  Loading N3 EEG for SC4001 ...")
    eeg_uv, fs_native = load_n3_eeg()
    print(f"  Total N3 signal: {len(eeg_uv) / fs_native:.1f} s")

    # Step 2
    print("\n[Step 2]  Building firing-rate proxy + eeg_raw ...")
    r_proxy, eeg_raw = build_rate_proxy(eeg_uv, fs_native)

    # Step 3
    print("\n[Step 3]  Computing 5 summary statistics ...")
    d = compute_summaries(r_proxy, fs=FS_SIM, eeg_raw=eeg_raw)

    print("\n  x_obs_v3 values:")
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
            "x_obs_v3 sanity checks failed. Review before launching SBI."
        )
    print("  All sanity checks PASSED.")

    # Step 5 — save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x_obs_values = np.array([d[k] for k in SUMMARY_KEYS], dtype=np.float32)
    metadata = {
        "subject_id":       SUBJECT_ID,
        "eeg_channel":      EEG_CHANNEL,
        "version":          "v3",
        "n_dims":           5,
        "summary_keys":     SUMMARY_KEYS,
        "dropped":          ["T6_ibi_cv", "MI"],
        "drop_reason":      (
            "r_proxy envelope structurally incompatible with UP/DOWN detection "
            "and cycle-by-cycle PAC phase interpolation. "
            "Diagnostic scan 2026-05-10: no threshold gives ibi_cv in [0.3,0.6]; "
            "MI stays ~0.0001 regardless of prominence_frac."
        ),
        "fixes_retained":   [
            "T8: spindle detection on eeg_raw (before abs/smooth/rescale)",
            "T11: PAC spindle amplitude from eeg_raw; SO phase from r_proxy peaks",
        ],
        "proxy_method":     "detrend -> abs -> 50ms_gaussian_smooth -> rescale_to_[0,60]",
        "fs_native_hz":     float(fs_native),
        "fs_resampled_hz":  FS_SIM,
        "n_samples_proxy":  int(len(r_proxy)),
        "duration_s":       float(len(r_proxy) / FS_SIM),
        "artifact_thresh_uv": ARTIFACT_THRESH * 1e6,
    }
    np.savez(
        str(out_path),
        values=x_obs_values,
        keys=SUMMARY_KEYS,
        extraction_metadata=json.dumps(metadata),
    )
    print(f"\n[Step 5]  Saved to {out_path}")
    print(f"  x_obs_v3  shape={x_obs_values.shape}  dtype={x_obs_values.dtype}")
    return x_obs_values


if __name__ == "__main__":
    main()
