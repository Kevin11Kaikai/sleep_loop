"""
scan_xobs_params.py
===================
参数扫描脚本：确定 compute_xobs_from_eeg_v2.py 里两个关键参数的最优值。

任务 1 — T6 阈值扫描（硬阈值 on 500ms-smoothed r_proxy）：
    threshold ∈ [5, 8, 10, 12, 15]
    目标：T6_ibi_cv ∈ [0.3, 0.6]；n_bursts 接近生理合理范围

任务 2 — MI prominence 扫描（SO_PEAK_PROMINENCE_FRAC in compute_pac_metrics_fixed）：
    prominence_frac ∈ [0.05, 0.10, 0.15, 0.20, 0.30]
    目标：MI ∈ [0.01, 0.05]；up_down_ratio > 1.5

任务 3 — 用最优参数生成 x_obs_v3.npz（仅在两个扫描都有合理结果时执行）

STOP 条件（内置）：
    - T6：若无阈值使 ibi_cv ∈ [0.3, 0.6]，打印 STOP 并不生成 v3
    - MI ：若无 prominence 使 MI ∈ [0.01, 0.05]，打印 STOP 并不生成 v3
"""

import sys
import os
import json
import importlib.util
from math import gcd
from pathlib import Path

# Force UTF-8 output on Windows (avoids GBK UnicodeEncodeError for Chinese/symbols)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.signal import detrend, butter, sosfiltfilt, hilbert, resample_poly
from scipy.ndimage import gaussian_filter1d
import mne
mne.set_log_level("WARNING")

# ── 路径 ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
os.chdir(str(_ROOT))
sys.path.insert(0, str(_ROOT))

import builtins as _builtins_mod
for _alias in ("int", "float", "bool", "object", "complex", "str"):
    if not hasattr(np, _alias):
        setattr(np, _alias, getattr(_builtins_mod, _alias))

_LOCAL_NEUROLIB = Path(r"D:\Year3_Mao_Projects\neurolib")
if _LOCAL_NEUROLIB.is_dir() and str(_LOCAL_NEUROLIB) not in sys.path:
    sys.path.insert(0, str(_LOCAL_NEUROLIB))

_repair_dir = _ROOT / "S4_v7_repair"
if str(_repair_dir) not in sys.path:
    sys.path.insert(0, str(_repair_dir))
from compute_pac_metrics_fixed import compute_pac_metrics

_prep_spec = importlib.util.spec_from_file_location(
    "preprocess_psd", str(_ROOT / "utils" / "02_preprocess_psd.py")
)
_prep_mod = importlib.util.module_from_spec(_prep_spec)
_prep_spec.loader.exec_module(_prep_mod)
load_hypnogram    = _prep_mod.load_hypnogram
compute_epoch_psd = _prep_mod.compute_epoch_psd
EPOCH_LEN_S       = _prep_mod.EPOCH_LEN_S

# ── 常量 ──────────────────────────────────────────────────────────────────────
SUBJECT_ID           = "SC4001"
EEG_CHANNEL          = "EEG Fpz-Cz"
N3_LABELS            = ["N3"]
ARTIFACT_THRESH      = 200e-6
FS_NATIVE            = 100.0
FS_SIM               = 1000.0
SO_FREQ_LO, SO_FREQ_HI = 0.2, 1.5
SPINDLE_LO, SPINDLE_HI = 10.0, 14.0
SPINDLE_DUR_LO_S     = 0.3
SPINDLE_DUR_HI_S     = 2.0
SPINDLE_EVT_PCTILE   = 75.0
SPINDLE_ENV_SMOOTH_MS = 200.0
SUMMARY_KEYS = ["shape_r","T4_q","T4_freq","T6_ibi_cv","T8_n_sp_events","T11_lag_ms","MI"]


# ── Step 0: 加载 EEG（只做一次） ─────────────────────────────────────────────
def load_data():
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
    accepted = []
    for i in range(n_epochs):
        if stages[i] not in N3_LABELS:
            continue
        epoch = data_uv[i * n_per_epoch : (i + 1) * n_per_epoch]
        if np.ptp(epoch) <= ARTIFACT_THRESH * 1e6:
            accepted.append(epoch)
    eeg_uv = np.concatenate(accepted)

    # resample 100→1000 Hz, detrend
    g = gcd(int(FS_SIM), int(fs))
    eeg_1k = resample_poly(eeg_uv, int(FS_SIM) // g, int(fs) // g)
    eeg_raw = detrend(eeg_1k, type="linear")

    # r_proxy = abs → 50ms smooth → rescale [0,60]
    r = np.abs(eeg_raw)
    r_smooth = gaussian_filter1d(r, sigma=50.0)
    r_proxy = r_smooth - r_smooth.min()
    p95 = np.percentile(r_proxy, 95)
    r_proxy = r_proxy / p95 * 60.0

    print(f"  Loaded: {len(eeg_uv)/fs:.0f}s N3 EEG → eeg_raw {len(eeg_raw)} samples")
    print(f"  r_proxy: max={r_proxy.max():.1f}  mean={r_proxy.mean():.2f}")
    print(f"  eeg_raw: std={eeg_raw.std():.2f} uV  max_abs={np.abs(eeg_raw).max():.2f} uV")
    return r_proxy, eeg_raw, float(len(r_proxy)) / FS_SIM   # duration_s


# ── Task 1: T6 threshold scan ─────────────────────────────────────────────────
def scan_t6(r_proxy, thresholds):
    print("\n" + "=" * 60)
    print("Task 1 — T6_ibi_cv 阈值扫描")
    print("=" * 60)
    print(f"  {'threshold':>10}  {'n_bursts':>10}  {'ibi_cv':>10}  {'合理[0.3,0.6]':>14}")
    print("  " + "-" * 50)

    r_for_ibi = gaussian_filter1d(r_proxy, sigma=500.0)
    results = []
    for thr in thresholds:
        above  = (r_for_ibi > thr).astype(np.int8)
        diff_  = np.diff(np.concatenate(([0], above, [0])))
        starts = np.where(diff_ == 1)[0]
        n_bursts = len(starts)
        if n_bursts >= 3:
            intervals = np.diff(starts) / FS_SIM
            ibi_cv = float(intervals.std() / (intervals.mean() + 1e-12))
        else:
            ibi_cv = 999.0
        ok = "✓" if 0.3 <= ibi_cv <= 0.6 else "✗"
        print(f"  {thr:>10.1f}  {n_bursts:>10d}  {ibi_cv:>10.3f}  {ok:>14}")
        results.append({"threshold": thr, "n_bursts": n_bursts, "ibi_cv": ibi_cv, "ok": ok})

    # pick best: lowest threshold where ibi_cv in [0.3, 0.6]
    valid = [r for r in results if 0.3 <= r["ibi_cv"] <= 0.6]
    if not valid:
        print("\n  *** STOP: 无阈值使 ibi_cv ∈ [0.3, 0.6] ***")
        return None, results
    best = valid[0]  # lowest threshold that satisfies condition
    print(f"\n  → 选择 threshold = {best['threshold']} "
          f"(n_bursts={best['n_bursts']}, ibi_cv={best['ibi_cv']:.3f})")
    return best["threshold"], results


# ── Task 2: MI prominence scan ────────────────────────────────────────────────
def scan_mi(r_proxy, eeg_raw, prominence_fracs):
    print("\n" + "=" * 60)
    print("Task 2 — MI prominence_frac 扫描")
    print("=" * 60)
    print(f"  {'prom_frac':>10}  {'n_so_peaks':>12}  {'MI':>10}  {'udr':>8}  {'MI∈[0.01,0.05]':>16}")
    print("  " + "-" * 62)

    results = []
    for pf in prominence_fracs:
        pac = compute_pac_metrics(
            r_proxy, eeg_raw, fs=FS_SIM,
            SO_PEAK_PROMINENCE_FRAC=pf
        )
        mi  = pac.get("mi", 0.0)
        udr = pac.get("up_down_ratio", 0.0)
        n_peaks = pac.get("n_so_cycles", 0)
        ok_str  = "✓" if 0.01 <= mi <= 0.05 else "✗"
        print(f"  {pf:>10.2f}  {n_peaks:>12d}  {mi:>10.5f}  {udr:>8.3f}  {ok_str:>16}")
        results.append({"pf": pf, "n_peaks": n_peaks, "mi": mi, "udr": udr, "ok": ok_str})

    valid = [r for r in results if 0.01 <= r["mi"] <= 0.05]
    if not valid:
        print("\n  *** STOP: 无 prominence_frac 使 MI ∈ [0.01, 0.05] ***")
        return None, results
    # pick the one closest to target center 0.03
    best = min(valid, key=lambda r: abs(r["mi"] - 0.03))
    print(f"\n  → 选择 prominence_frac = {best['pf']} "
          f"(n_peaks={best['n_peaks']}, MI={best['mi']:.5f}, udr={best['udr']:.3f})")
    return best["pf"], results


# ── Task 3: generate x_obs_v3.npz with best params ───────────────────────────
def generate_v3(r_proxy, eeg_raw, duration_s, best_thresh, best_pf):
    print("\n" + "=" * 60)
    print("Task 3 — 生成 x_obs_v3.npz")
    print(f"  T6 threshold = {best_thresh}  |  MI prominence_frac = {best_pf}")
    print("=" * 60)

    d = {}
    d["shape_r"] = 1.0

    # T4 (unchanged)
    f_c, p_c = compute_epoch_psd(r_proxy, FS_SIM)
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

    # T6 — best_thresh
    r_for_ibi = gaussian_filter1d(r_proxy, sigma=500.0)
    above  = (r_for_ibi > best_thresh).astype(np.int8)
    diff_  = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(diff_ == 1)[0]
    n_bursts = len(starts)
    ibi_cv = 999.0
    if n_bursts >= 3:
        intervals = np.diff(starts) / FS_SIM
        ibi_cv = float(intervals.std() / (intervals.mean() + 1e-12))
    d["T6_ibi_cv"] = round(ibi_cv, 3)
    print(f"  T6: n_bursts={n_bursts}  ibi_cv={ibi_cv:.3f}")

    # T8 — eeg_raw (BUG 1 fix, unchanged from v2)
    n_sp_events = 0
    try:
        sos = butter(4, [SPINDLE_LO, SPINDLE_HI], btype="band", fs=FS_SIM, output="sos")
        filtered  = sosfiltfilt(sos, eeg_raw)
        envelope  = np.abs(hilbert(filtered))
        sigma_samp = SPINDLE_ENV_SMOOTH_MS * FS_SIM / 1000.0
        env_sm    = gaussian_filter1d(envelope, sigma=sigma_samp)
        thresh    = np.percentile(env_sm, SPINDLE_EVT_PCTILE)
        ab_sp     = (env_sm > thresh).astype(np.int8)
        diff_sp   = np.diff(np.concatenate(([0], ab_sp, [0])))
        sp_st     = np.where(diff_sp == 1)[0]
        sp_en     = np.where(diff_sp == -1)[0]
        durs      = (sp_en - sp_st) / FS_SIM
        valid_sp  = (durs >= SPINDLE_DUR_LO_S) & (durs <= SPINDLE_DUR_HI_S)
        n_sp_events = int(valid_sp.sum())
    except Exception as exc:
        print(f"  [warn] Spindle detection error: {exc}")
    t8_norm = n_sp_events * (60.0 / duration_s)
    d["T8_n_sp_events"] = round(t8_norm, 3)
    print(f"  T8: raw={n_sp_events}  normalized={t8_norm:.2f}")

    # PAC — best_pf prominence (BUG 2 fix)
    pac = compute_pac_metrics(r_proxy, eeg_raw, fs=FS_SIM,
                              SO_PEAK_PROMINENCE_FRAC=best_pf)
    d["T11_lag_ms"] = round(pac.get("up_down_ratio", 0.0), 3)
    d["MI"]         = round(pac.get("mi", 0.0), 5)
    print(f"  PAC: MI={d['MI']}  udr={d['T11_lag_ms']}  "
          f"n_so_cycles={pac.get('n_so_cycles', 0)}  ok={pac.get('ok')}")

    # sanity checks
    failures = []
    if not (0.5 <= d["T4_freq"] <= 1.5):
        failures.append(f"T4_freq={d['T4_freq']} not in [0.5,1.5]")
    if not (0.1 <= d["T6_ibi_cv"] <= 0.8):
        failures.append(f"T6_ibi_cv={d['T6_ibi_cv']} not in [0.1,0.8]")
    if d["T11_lag_ms"] < 1.0:
        failures.append(f"T11_lag_ms={d['T11_lag_ms']} < 1.0")
    if d["T8_n_sp_events"] <= 5:
        failures.append(f"T8={d['T8_n_sp_events']} <= 5")
    if failures:
        print("\n  *** SANITY CHECK FAILED ***")
        for m in failures:
            print(f"    FAIL: {m}")
        print("  x_obs_v3.npz NOT saved.")
        return d, False

    # save
    out_path = _SCRIPT_DIR / "x_obs_v3.npz"
    x_obs_values = np.array([d[k] for k in SUMMARY_KEYS], dtype=np.float32)
    metadata = {
        "subject_id": SUBJECT_ID, "version": "v3",
        "T6_threshold": best_thresh,
        "MI_prominence_frac": best_pf,
        "fixes": [
            "T8: spindle from eeg_raw",
            f"T6: hard threshold {best_thresh} on 500ms-smoothed r_proxy",
            f"MI/T11: SO phase from r_proxy peaks (prom_frac={best_pf}), spindle-amp from eeg_raw",
        ],
        "duration_s": duration_s,
    }
    np.savez(str(out_path), values=x_obs_values, keys=SUMMARY_KEYS,
             extraction_metadata=json.dumps(metadata))
    print(f"\n  Saved: {out_path}")
    return d, True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("scan_xobs_params.py — T6 & MI 参数扫描")
    print("=" * 60)

    print("\n[Loading EEG] ...")
    r_proxy, eeg_raw, duration_s = load_data()

    # Task 1
    best_thresh, t6_results = scan_t6(
        r_proxy,
        thresholds=[5, 8, 10, 12, 15]
    )

    # Task 2
    best_pf, mi_results = scan_mi(
        r_proxy, eeg_raw,
        prominence_fracs=[0.05, 0.10, 0.15, 0.20, 0.30]
    )

    # Task 3 — only if both found
    if best_thresh is None or best_pf is None:
        print("\n*** STOP: 参数扫描未找到合理值，不生成 x_obs_v3.npz ***")
        return

    d_v3, saved = generate_v3(r_proxy, eeg_raw, duration_s, best_thresh, best_pf)

    # comparison table
    v1 = {"shape_r": 1.0, "T4_q": 2.645, "T4_freq": 0.750,
          "T6_ibi_cv": 0.770, "T8_n_sp_events": 14.352,
          "T11_lag_ms": 1.319, "MI": 0.00023}
    v2 = {"shape_r": 1.0, "T4_q": 2.645, "T4_freq": 0.750,
          "T6_ibi_cv": 1.714, "T8_n_sp_events": 15.31,
          "T11_lag_ms": 1.280, "MI": 0.00012}
    ppc = {"shape_r": "1.0(设计)", "T4_q": "~2.4", "T4_freq": "~0.75",
           "T6_ibi_cv": "~0.36", "T8_n_sp_events": "~23",
           "T11_lag_ms": "~3.0", "MI": "~0.07"}

    print("\n" + "=" * 80)
    print("对比表: v1 (buggy) | v2 (partial) | v3 (final) | 仿真 PPC 中位数")
    print("=" * 80)
    print(f"  {'stat':20s}  {'v1':>10}  {'v2':>10}  {'v3':>10}  {'PPC median':>12}")
    print("  " + "-" * 68)
    for k in SUMMARY_KEYS:
        print(f"  {k:20s}  {str(v1[k]):>10}  {str(v2.get(k,'?')):>10}  "
              f"{str(round(d_v3[k],5)):>10}  {str(ppc.get(k,'?')):>12}")

    if saved:
        print("\n  x_obs_v3.npz 已生成。")
    else:
        print("\n  x_obs_v3.npz 未生成（sanity check 失败）。")


if __name__ == "__main__":
    main()
