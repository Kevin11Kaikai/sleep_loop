"""
compute_xobs_from_eeg_v2.py
============================

v2 修复版本（相对于 v1_buggy 的 3 处 bug 修复）：

BUG 1 (T8_n_sp_events): v1 对 r_proxy 做 10-14 Hz 带通检测纺锤波。
  r_proxy 已经过 50ms Gaussian 平滑（截止约 3 Hz），纺锤频段衰减 >99.9%。
  检测结果是噪声突发，不是真实纺锤波。
  修复：对 eeg_raw（去趋势后的 1000 Hz EEG，未经 abs/smooth）做纺锤检测。

BUG 2 (MI / T11_lag_ms): v1 的 PAC 计算中，纺锤振幅来自对 r_proxy 的带通。
  同上，r_proxy 无纺锤内容，PAC 测量的是噪声，导致 MI ≈ 0.0002 远低于仿真值。
  修复：SO 相位继续从 r_proxy 峰提取（保留完整 SO 结构），
        纺锤振幅改为对 eeg_raw 做 10-14 Hz 带通 + Hilbert envelope。

BUG 3 (T6_ibi_cv): v1 用 50th percentile 作 UP 阈值，
  导致 UP 占空比 ≈ 50%（由构造决定），与 V7 仿真侧的绝对阈值 15 Hz 不一致。
  修复：保留 500ms 额外平滑，阈值改为硬值 15.0
        （r_proxy 已 rescale 到 [0, 60]，V7 UP_THRESH_HZ=15 对应相同的相对位置）。

其余代码与 v1 完全相同。

SUMMARY_KEYS（顺序同 simulator_wrapper.SUMMARY_KEYS；7 维）:
  0: shape_r        — 固定为 1.0（设计选择，非 bug）
  1: T4_q           — SO 主峰 Q 值
  2: T4_freq        — SO 主峰频率 [Hz]
  3: T6_ibi_cv      — UP 脉冲间隔变异系数  ← BUG 3 修复
  4: T8_n_sp_events — 每 60 秒纺锤事件数   ← BUG 1 修复
  5: T11_lag_ms     — up_down_ratio         ← BUG 2 修复
  6: MI             — PAC 调制指数          ← BUG 2 修复

用法（在项目根目录下）:
    conda activate neurolib
    python S4_sbi/compute_xobs_from_eeg_v2.py
    python S4_sbi/compute_xobs_from_eeg_v2.py --output S4_sbi/x_obs_v2.npz
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


# ═══════════════════════════════════════════════════════════════════════════════
# 配置（与 s4_personalize_fig7_v7.py 保持一致）
# ═══════════════════════════════════════════════════════════════════════════════
SUBJECT_ID           = "SC4001"
EEG_CHANNEL          = "EEG Fpz-Cz"
N3_LABELS            = ["N3"]
ARTIFACT_THRESH      = 200e-6          # V (200µV peak-to-peak)
FS_NATIVE            = 100.0
FS_SIM               = 1000.0

# V7 constraint constants (must match s4_personalize_fig7_v7.py)
UP_THRESH_HZ         = 15.0           # V7 UP_THRESH_HZ — BUG 3 fix: use this as hard threshold
SO_FREQ_LO           = 0.2
SO_FREQ_HI           = 1.5
SPINDLE_LO           = 10.0
SPINDLE_HI           = 14.0
SPINDLE_DUR_LO_S     = 0.3
SPINDLE_DUR_HI_S     = 2.0
SPINDLE_EVT_PCTILE   = 75.0
SPINDLE_ENV_SMOOTH_MS = 200.0

SUMMARY_KEYS = [
    "shape_r", "T4_q", "T4_freq", "T6_ibi_cv",
    "T8_n_sp_events", "T11_lag_ms", "MI",
]

DEFAULT_OUTPUT = str(_SCRIPT_DIR / "x_obs_v2.npz")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load N3 EEG (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════════
def load_n3_eeg():
    """Load and concatenate artifact-free N3 epochs for SC4001."""
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

    print(f"  EEG: {n_n3} N3 epochs total, {n_rej} artifact-rejected, "
          f"{len(accepted)} accepted  ({sum(len(e) for e in accepted)/fs:.0f} s)")
    print(f"  Native fs = {fs} Hz")
    return np.concatenate(accepted), float(fs)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Build firing-rate proxy + raw EEG at FS_SIM
# ═══════════════════════════════════════════════════════════════════════════════
def build_rate_proxy(eeg_uv, fs_from):
    """
    Map EEG voltage to [0, 60] non-negative proxy and also return the
    resampled+detrended raw EEG at FS_SIM (before abs/smooth/rescale).

    Returns
    -------
    r_proxy : ndarray — firing-rate proxy, [0, 60], shape (N,)
    eeg_raw : ndarray — detrended EEG at FS_SIM [µV], shape (N,)
              Contains genuine spindle-band content for T8/PAC extraction.
    """
    g = gcd(int(FS_SIM), int(fs_from))
    eeg_1k = resample_poly(eeg_uv, int(FS_SIM) // g, int(fs_from) // g)

    # Detrended EEG at FS_SIM — preserved for spindle detection (BUG 1+2 fix)
    eeg_raw = detrend(eeg_1k, type="linear")

    # r_proxy pipeline (unchanged from v1)
    r = np.abs(eeg_raw)
    r_smooth = gaussian_filter1d(r, sigma=50.0)
    r_proxy = r_smooth - r_smooth.min()
    p95 = np.percentile(r_proxy, 95)
    if p95 < 1e-9:
        raise RuntimeError("EEG proxy 95th-percentile ≈ 0 — check data quality")
    r_proxy = r_proxy / p95 * 60.0

    print(f"  r_proxy: {len(r_proxy)} samples  "
          f"mean={r_proxy.mean():.2f}  max={r_proxy.max():.2f}  "
          f"95pct={np.percentile(r_proxy, 95):.2f}  fs={FS_SIM} Hz")
    print(f"  eeg_raw: {len(eeg_raw)} samples  "
          f"std={eeg_raw.std():.2f} µV  max={np.abs(eeg_raw).max():.2f} µV")
    return r_proxy, eeg_raw


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Compute 7 summary statistics (v2: 3 bugs fixed)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_summaries(r_proxy, fs=FS_SIM, eeg_raw=None):
    """
    Compute all 7 summary statistics from r_proxy (and eeg_raw for spindle stats).

    Parameters
    ----------
    r_proxy : ndarray — firing-rate proxy [0, 60]
    fs      : float   — sampling rate (FS_SIM = 1000 Hz)
    eeg_raw : ndarray — detrended raw EEG at FS_SIM [µV]
              Required for T8, MI, T11 (BUG 1+2 fix).
              If None, falls back to r_proxy (v1 behaviour, for testing).
    """
    d = {}

    # ── shape_r = 1.0 (design choice, not a bug) ──────────────────────────────
    d["shape_r"] = 1.0

    # ── T4: SO peak frequency and Q-factor (unchanged from v1) ───────────────
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

    # ── T6: IBI CV  ← BUG 3 FIX: hard threshold 15 instead of 50th pctile ───
    # Keep 500ms extra smooth (suppresses sub-second spindle/transient ripples).
    # Replace adaptive 50th-percentile threshold with hard value 15.0,
    # matching V7's UP_THRESH_HZ = 15 Hz. r_proxy is rescaled to [0, 60],
    # so threshold 15 = 25% of range, the same relative position as V7.
    r_for_ibi = gaussian_filter1d(r_proxy, sigma=500.0)
    up_thresh  = UP_THRESH_HZ   # = 15.0; aligned with V7 (BUG 3 fix)
    print(f"  T6 SO-envelope UP threshold: {up_thresh:.1f} (hard, aligned with V7 UP_THRESH_HZ)")
    above  = (r_for_ibi > up_thresh).astype(np.int8)
    diff_  = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(diff_ == 1)[0]
    n_bursts = len(starts)
    ibi_cv = 999.0
    if n_bursts >= 3:
        intervals = np.diff(starts) / fs
        ibi_cv = float(intervals.std() / (intervals.mean() + 1e-12))
    print(f"  T6 n_bursts={n_bursts}  ibi_cv={ibi_cv:.3f}")
    d["T6_ibi_cv"] = round(ibi_cv, 3)

    # ── T8: spindle events per 60 s  ← BUG 1 FIX: detect from eeg_raw ───────
    # v1 problem: r_proxy has no 10-14 Hz content (50ms smooth kills spindles).
    # v2 fix: use eeg_raw (detrended 1000 Hz EEG, before abs/smooth/rescale).
    # eeg_raw retains genuine spindle oscillations from thalamo-cortical loops.
    duration_s = float(len(r_proxy)) / fs
    n_sp_events = 0
    sp_signal = eeg_raw if eeg_raw is not None else r_proxy  # fallback for testing
    if eeg_raw is None:
        print("  [warn] eeg_raw not provided; T8 falls back to r_proxy (v1 behaviour)")
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
          f"normalized (per 60s)={t8_normalized:.2f}  "
          f"[source: {'eeg_raw' if eeg_raw is not None else 'r_proxy (fallback)'}]")
    d["T8_n_sp_events"] = round(t8_normalized, 3)

    # ── T9-T11: PAC  ← BUG 2 FIX: spindle amplitude from eeg_raw ────────────
    # v1: compute_pac_metrics(r_proxy, r_proxy, fs) — both signals are r_proxy.
    #   r_proxy has no 10-14 Hz content → PAC spindle amplitude = noise → MI ≈ 0.
    # v2: SO phase from r_proxy (has clear SO peaks → good cycle-by-cycle phase).
    #     Spindle amplitude from eeg_raw (genuine 10-14 Hz thalamo-cortical spindles).
    # The compute_pac_metrics_fixed function uses r_ctx for peak detection (SO phase)
    # and r_thal for bandpass spindle amplitude — a direct match for our hybrid use.
    if eeg_raw is not None:
        pac = compute_pac_metrics(r_proxy, eeg_raw, fs=fs)
        print("  PAC: SO-phase from r_proxy, spindle-amp from eeg_raw (BUG 2 fix)")
    else:
        pac = compute_pac_metrics(r_proxy, r_proxy, fs=fs)
        print("  [warn] eeg_raw not provided; PAC falls back to r_proxy×r_proxy (v1)")
    d["T11_lag_ms"] = round(pac.get("up_down_ratio", 0.0), 3)
    d["MI"]         = round(pac.get("mi", 0.0), 5)

    return d


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 — Sanity checks (same thresholds as v1)
# ═══════════════════════════════════════════════════════════════════════════════
def run_sanity_checks(d):
    failures = []
    if not (0.5 <= d["T4_freq"] <= 1.5):
        failures.append(f"T4_freq = {d['T4_freq']:.3f} not in [0.5, 1.5] Hz")
    if not (0.1 <= d["T6_ibi_cv"] <= 0.8):
        failures.append(f"T6_ibi_cv = {d['T6_ibi_cv']:.3f} not in [0.1, 0.8]")
    if d["T11_lag_ms"] < 1.0:
        failures.append(f"T11_lag_ms (up_down_ratio) = {d['T11_lag_ms']:.3f} < 1.0")
    if d["T8_n_sp_events"] <= 5:
        failures.append(f"T8_n_sp_events = {d['T8_n_sp_events']} <= 5")
    return failures


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output path for x_obs npz (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    print("=" * 64)
    print("compute_xobs_from_eeg_v2.py  —  x_obs extraction (3 bugs fixed)")
    print("=" * 64)
    print("Fixes: T8 from eeg_raw | PAC spindle-amp from eeg_raw | T6 hard thresh 15")

    # Step 1
    print("\n[Step 1]  Loading N3 EEG for SC4001 ...")
    eeg_uv, fs_native = load_n3_eeg()
    print(f"  Total N3 signal: {len(eeg_uv) / fs_native:.1f} s")

    # Step 2
    print("\n[Step 2]  Building firing-rate proxy + eeg_raw ...")
    r_proxy, eeg_raw = build_rate_proxy(eeg_uv, fs_native)

    # Step 3
    print("\n[Step 3]  Computing 7 summary statistics (v2) ...")
    d = compute_summaries(r_proxy, fs=FS_SIM, eeg_raw=eeg_raw)

    print("\n  x_obs_v2 values:")
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
            "x_obs_v2 sanity checks failed. Review fixes before launching SBI."
        )
    print("  All 4 sanity checks PASSED.")

    # Step 5 — save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x_obs_values = np.array([d[k] for k in SUMMARY_KEYS], dtype=np.float32)
    metadata = {
        "subject_id":       SUBJECT_ID,
        "eeg_channel":      EEG_CHANNEL,
        "version":          "v2",
        "fixes":            [
            "T8: spindle detection on eeg_raw (detrended 1000 Hz EEG before abs/smooth)",
            "MI/T11: PAC spindle amplitude from eeg_raw; SO phase from r_proxy peaks",
            "T6: UP threshold = 15.0 (hard, aligned with V7 UP_THRESH_HZ)",
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
    print(f"  x_obs_v2  shape={x_obs_values.shape}  dtype={x_obs_values.dtype}")
    return x_obs_values


if __name__ == "__main__":
    main()
