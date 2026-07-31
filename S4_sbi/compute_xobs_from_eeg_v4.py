"""
compute_xobs_from_eeg_v4.py
============================
用于 睡眠孪生阶段 2 (SBI Stage 2 / Sprint 1 Phase 2) 的 7维观测特征向量 x_obs 提取脚本。

SUMMARY_KEYS (7维特征，与 Phase 2 后的 simulator_wrapper 输出顺序完全一致):
  0: shape_r        — 固定值 1.0 (设计哨兵值；详见 sbi_report_0511.md §4.3)
  1: T4_q           — r_proxy 慢波 Welch PSD 上的慢波峰 Q-factor (品质因数)
  2: T4_freq        — r_proxy 慢波 Welch PSD 上的慢波峰中心频率 [Hz]
  3: T6_ibi_cv      — 真实 EEG-raw 上 AASM 标准的慢波波谷间期变异系数 (新引入: 带通滤波 + 75uV 半波判定)
  4: T8_n_sp_events — 真实 eeg_raw 信号上每 60 秒的纺锤波事件数 (10-14 Hz 带通 + 包络阈值检测)
  5: T11_lag_ms     — up_down_ratio (EEG-native 慢波 UP/DOWN 振幅比，与新 MI 算法同源)
  6: MI             — 真实 EEG-raw 上基于单通道 Tort 2010 标准的 EEG-native 相位-振幅耦合指数 (新引入)

相对于 v3 的主要改进：
  - 重新引入了 `T6_ibi_cv` 特征 (在 v3 中曾被暂时移除，现采用 EEG-native 算法重构)
  - 重新引入了 `MI` 特征 (在 v3 中曾被暂时极简移除，现采用单通道 EEG-native PAC 重构)
  - `T11_lag_ms` 由原来的跨通道混合计算，改为与 `MI` 共享同一次 `compute_mi_eeg_native` 计算的 up_down_ratio 比值 (更具生理同构性)
  - `shape_r`, `T4_q`, `T4_freq`, `T8_n_sp_events` 的计算算法与 v3 保持完全一致

理论依据与计划背景：详见 docs/sprint1_phase2_plan_v2.md 第 1 和第 2 节。

运行方式 (在项目根目录下执行):
    conda activate neurolib
    python S4_sbi/compute_xobs_from_eeg_v4.py
    或者输出到指定位置：
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

# 过滤不必要的 Deprecation 警告等，保持终端输出整洁
warnings.filterwarnings("ignore")

# ── 自动定位工作目录为项目根目录 (必须在导入其它自定义模块之前完成) ──────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
os.chdir(str(_ROOT))
sys.path.insert(0, str(_ROOT))

# ── NumPy 别名修补 (因为旧版 neurolib 在有些环境下需要 np.int, np.float 等别名) ──
import numpy as np
import builtins as _builtins_mod
for _alias in ("int", "float", "bool", "object", "complex", "str"):
    if not hasattr(np, _alias):
        setattr(np, _alias, getattr(_builtins_mod, _alias))

# ── 优先导入本地定制的 neurolib 路径 ──────────────────────────────────────────
_LOCAL_NEUROLIB = Path(r"D:\Year3_Mao_Projects\neurolib")
if _LOCAL_NEUROLIB.is_dir() and str(_LOCAL_NEUROLIB) not in sys.path:
    sys.path.insert(0, str(_LOCAL_NEUROLIB))

import pandas as pd
from scipy.signal import detrend, resample_poly
from scipy.ndimage import gaussian_filter1d
import mne
mne.set_log_level("WARNING")  # 降低 MNE 的日志冗余

# ── 采用动态加载加载频域预处理工具 02_preprocess_psd ──
_prep_spec = importlib.util.spec_from_file_location(
    "preprocess_psd", str(_ROOT / "utils" / "02_preprocess_psd.py")
)
_prep_mod = importlib.util.module_from_spec(_prep_spec)
_prep_spec.loader.exec_module(_prep_mod)
load_hypnogram    = _prep_mod.load_hypnogram
compute_epoch_psd = _prep_mod.compute_epoch_psd
EPOCH_LEN_S       = _prep_mod.EPOCH_LEN_S

# ── 载入 Phase 2 重新打磨的 EEG-native 时域算法 ──
_repair_dir = _ROOT / "S4_v7_repair"
if str(_repair_dir) not in sys.path:
    sys.path.insert(0, str(_repair_dir))
from compute_pac_metrics_eeg_native import (
    compute_t6_eeg_aasm, compute_mi_eeg_native,
)

# 导入 T8 纺锤波时域检测所必须的 SciPy 滤波器与变换工具
from scipy.signal import butter, sosfiltfilt, hilbert


# =============================================================================
# 全局静态配置 (必须与 models/s4_personalize_fig7_v7.py 严格一致，确保双端同构)
# =============================================================================
SUBJECT_ID            = "SC4001"          # 默认受试者 ID
EEG_CHANNEL           = "EEG Fpz-Cz"      # 默认采集通道 (Sleep-EDF 数据集)
N3_LABELS             = ["N3"]            # 目标睡眠分期：慢波 N3 睡眠
ARTIFACT_THRESH       = 200e-6            # 伪迹剔除阈值：峰峰值 > 200 µV 则丢弃
FS_NATIVE             = 100.0             # Sleep-EDF 脑电原生采样率 100 Hz
FS_SIM                = 1000.0            # 物理仿真器标准采样率 1000 Hz

# 时频域特征计算常量限制
SO_FREQ_LO            = 0.2               # 慢波搜索区间下限
SO_FREQ_HI            = 1.5               # 慢波搜索区间上限
SPINDLE_LO            = 10.0              # 纺锤波频段下限 (慢纺锤偏低，此处取 10 Hz)
SPINDLE_HI            = 14.0              # 纺锤波频段上限
SPINDLE_DUR_LO_S      = 0.3               # 纺锤波事件最短持续时间 (秒)
SPINDLE_DUR_HI_S      = 2.0               # 纺锤波事件最长持续时间 (秒)
SPINDLE_EVT_PCTILE    = 75.0              # 纺锤波时域包络检测阈值分位数 (75% 分位)
SPINDLE_ENV_SMOOTH_MS = 200.0             # 提取包络时的高斯平滑 σ 大小 (毫秒)

# 7 维特征键名，神经网络根据此顺序接收输入
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
# Step 1 — 加载 N3 脑电时序列 (与 v3 流程完全同构)
# =============================================================================
def load_n3_eeg():
    """
    从 EDF 文件中读取指定病人的 Fpz-Cz 脑电信号，自动对齐催眠图，
    仅筛选无噪声伪迹的 N3 慢波睡眠片段，并在时域顺次拼装。
    """
    # 读入受试者配置对照清单 manifest
    try:
        manifest = pd.read_csv("data/manifest.csv", encoding="utf-8")
    except UnicodeDecodeError:
        manifest = pd.read_csv("data/manifest.csv", encoding="utf-16")

    # 取出当前受试者的对应数据行
    subj_row = manifest[manifest["subject_id"] == SUBJECT_ID].iloc[0]
    
    # 使用 MNE 以 preload 内存预加载方式载入原始 EDF 脑电通道
    raw = mne.io.read_raw_edf(
        subj_row["psg_path"], include=[EEG_CHANNEL], preload=True, verbose=False
    )
    fs = raw.info["sfreq"]  # 原生采样率，SC4001 为 100.0 Hz
    
    # 加载催眠图睡眠分期
    stages = load_hypnogram(Path(subj_row["hypnogram_path"]))
    
    # 提取电压数据并换算为微伏级 (MNE 内部默认是伏特 V，需要乘以 1e6 转换为 uV)
    data_uv = raw.get_data()[0] * 1e6

    n_per_epoch = int(EPOCH_LEN_S * fs)  # 每个 epoch 的采样点数 (30秒 * 100Hz = 3000点)
    n_epochs = min(len(stages), len(data_uv) // n_per_epoch)
    
    accepted, n_n3, n_rej = [], 0, 0
    # 遍历每个 epoch 进行睡眠阶段判定和噪声伪迹剔除
    for i in range(n_epochs):
        # 仅处理 N3 慢波睡眠
        if stages[i] not in N3_LABELS:
            continue
        n_n3 += 1
        epoch = data_uv[i * n_per_epoch : (i + 1) * n_per_epoch]
        
        # 质控：若当前 epoch 的 Peak-to-Peak 峰峰跨度超过 200 uV，视为包含噪声，予以剔除
        if np.ptp(epoch) > ARTIFACT_THRESH * 1e6:
            n_rej += 1
            continue
        accepted.append(epoch)

    if not accepted:
        raise RuntimeError(f"No N3 epochs passed artifact rejection for {SUBJECT_ID}")

    print(f"  EEG: {n_n3} N3 epochs total, {n_rej} rejected, "
          f"{len(accepted)} accepted  ({sum(len(e) for e in accepted)/fs:.0f} s)")
    print(f"  Native fs = {fs} Hz")
    
    # 将时域上所有合格的 epoch 首尾紧密拼接为一条超长时域信号
    return np.concatenate(accepted), float(fs)


# =============================================================================
# Step 2 — 降采样并构建放电率包络代理 r_proxy 与去趋势脑电 eeg_raw
# =============================================================================
def build_rate_proxy(eeg_uv, fs_from):
    """
    参数说明
    -------
    eeg_uv : ndarray，微伏级拼接脑电
    fs_from : float，原生采样率 (100 Hz)

    返回说明
    -------
    r_proxy : ndarray — 经过取绝对值->50ms高斯平滑->等比缩放到 [0, 60] Hz 范围的包络信号。
              代表慢波“皮层兴奋性包络代理”，专门用于 T4_freq 和 T4_q 频域指标的计算。
    eeg_raw : ndarray — 经 scipy 线性去趋势后并超采样到 1000 Hz 的时域脑电 [uV]。
              保留了完整的相位与高频纺锤特征，专门用于时域 T8, T6 和 EEG-native Tort PAC 耦合指标计算。
    """
    # 1. 采用多项式重采样 resample_poly 将原生脑电超采样到物理仿真的 1000 Hz 时钟
    g = gcd(int(FS_SIM), int(fs_from))
    eeg_1k = resample_poly(eeg_uv, int(FS_SIM) // g, int(fs_from) // g)

    # 2. 线性去趋势，消除时域信号基线慢速漂移
    eeg_raw = detrend(eeg_1k, type="linear")

    # 3. 提取包络模拟皮层发放率包络 (r_proxy)：
    r = np.abs(eeg_raw)                             # 取绝对值
    r_smooth = gaussian_filter1d(r, sigma=50.0)     # 50 ms 时间窗高斯滤波平滑
    r_proxy = r_smooth - r_smooth.min()             # 减去基底直流成分
    p95 = np.percentile(r_proxy, 95)                # 获取 95% 分位数作为分母缩放
    
    if p95 < 1e-9:
        raise RuntimeError("EEG proxy 95th-percentile ~0 — check data quality")
    
    # 将信号按比例映射缩放到 [0, 60] Hz 生理区间，对齐 V7 仿真中 r_ctx 的取值区间
    r_proxy = r_proxy / p95 * 60.0

    print(f"  r_proxy: {len(r_proxy)} samples  "
          f"mean={r_proxy.mean():.2f}  max={r_proxy.max():.2f}  "
          f"95pct={np.percentile(r_proxy, 95):.2f}  fs={FS_SIM} Hz")
    print(f"  eeg_raw: {len(eeg_raw)} samples  "
          f"std={eeg_raw.std():.2f} uV  max={np.abs(eeg_raw).max():.2f} uV")
    return r_proxy, eeg_raw


# =============================================================================
# Step 3 — 提取 7 个核心摘要统计量 (5 维基线 + T6、MI 修正重构)
# =============================================================================
def compute_summaries(r_proxy, eeg_raw, fs=FS_SIM):
    """
    提取真实数据的特征指纹 x_obs，送给神经网络进行对齐。
    """
    d = {}

    # ── shape_r: 固定 1.0 ────────────────────────────────────────────────
    # 因为真实脑电与自己算 FOOOF 频谱相关系数必然为 1.0，这在后验训练中作为完美目标（哨兵）。
    d["shape_r"] = 1.0

    # ── T4: 慢波段主峰频率与 Q-factor 估计 (作用于 r_proxy Welch PSD 频域) ──────
    f_c, p_c = compute_epoch_psd(r_proxy, fs)
    so_mask  = (f_c >= SO_FREQ_LO) & (f_c <= SO_FREQ_HI)       # 慢波带 [0.2, 1.5] Hz
    so_width = SO_FREQ_HI - SO_FREQ_LO
    neigh_lo = (f_c >= max(0.1, SO_FREQ_LO - so_width)) & (f_c < SO_FREQ_LO)  # 邻域低频段
    neigh_hi = (f_c > SO_FREQ_HI) & (f_c <= SO_FREQ_HI + so_width)           # 邻域高频段
    
    so_peak_freq, so_q = 0.0, 0.0
    if so_mask.any():
        # 获取慢波段内功率最大的谱线频率
        so_peak_freq = float(f_c[so_mask][np.argmax(p_c[so_mask])])
        so_peak_val  = float(p_c[so_mask].max())
        
        # 拼接邻近辅助背景频带，计算背景功率均值
        nbrs = np.concatenate([
            p_c[neigh_lo] if neigh_lo.any() else np.array([]),
            p_c[neigh_hi] if neigh_hi.any() else np.array([])
        ])
        if len(nbrs) > 0 and nbrs.mean() > 0:
            so_q = float(so_peak_val / nbrs.mean())   # Q-factor = 峰值功率 / 邻域平均背景功率

    d["T4_q"]    = round(so_q, 3)
    d["T4_freq"] = round(so_peak_freq, 3)

    # ── T6: EEG-native 时域慢波规律度 AASM SO UP IBI CV ───────────────────
    # 采用 Phase 2 改良的半波检测机制，获取慢波波谷起伏的规律度
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

    # ── T8: eeg_raw 上纺锤波事件的数目统计与 60s 归一化 (与 v3 完全一致) ─────────
    duration_s = float(len(eeg_raw)) / fs
    n_sp_events = 0
    try:
        # 设计 4 阶 Butterworth 带通滤波器，限制在 [10, 14] Hz
        sos = butter(4, [SPINDLE_LO, SPINDLE_HI], btype="band", fs=fs, output="sos")
        filtered  = sosfiltfilt(sos, eeg_raw)          # 双向零相位滤波
        envelope  = np.abs(hilbert(filtered))          # Hilbert 变换提取振幅包络
        sigma_samp = SPINDLE_ENV_SMOOTH_MS * fs / 1000.0
        env_sm    = gaussian_filter1d(envelope, sigma=sigma_samp) # 平滑包络
        thresh    = np.percentile(env_sm, SPINDLE_EVT_PCTILE)   # 取 75% 分位数作为事件检测线
        
        # 寻找越线段边界并提取持续时间
        ab_sp     = (env_sm > thresh).astype(np.int8)
        diff_sp   = np.diff(np.concatenate(([0], ab_sp, [0])))
        sp_st     = np.where(diff_sp == 1)[0]
        sp_en     = np.where(diff_sp == -1)[0]
        durs      = (sp_en - sp_st) / fs
        
        # 过滤处于 [0.3s, 2.0s] 范围内的合法生理纺锤波事件数
        valid     = (durs >= SPINDLE_DUR_LO_S) & (durs <= SPINDLE_DUR_HI_S)
        n_sp_events = int(valid.sum())
    except Exception as exc:
        print(f"  [warn] Spindle detection error: {exc}")
        
    # 规整到每 60 秒频次，作为 x_sim 相同的物理标度量
    t8_normalized = n_sp_events * (60.0 / duration_s)
    print(f"  T8 raw={n_sp_events}  duration={duration_s:.0f}s  "
          f"normalized={t8_normalized:.2f}/60s")
    d["T8_n_sp_events"] = round(t8_normalized, 3)

    # ── 基于单通道 EEG-native Tort PAC 耦合算法提取 MI 和 T11(up_down_ratio) ──
    # 在 eeg_raw 上算慢波相角 [0.5, 1.5] Hz 对纺锤波幅值 [10, 14] Hz 的调制指数
    mi_res = compute_mi_eeg_native(
        eeg_raw, fs,
        phase_band=(0.5, 1.5),
        amp_band=(SPINDLE_LO, SPINDLE_HI),
        n_phase_bins=18,
    )
    d["MI"]            = round(mi_res["mi"], 5)
    d["T11_lag_ms"]    = round(mi_res["up_down_ratio"], 4)   # up_down_ratio 成为新的 T11 表达
    d["_MI_pref_phase"] = round(mi_res["preferred_phase"], 3)
    d["_MI_ok"]         = mi_res["ok"]
    print(f"  MI mi={d['MI']}  up_down_ratio(T11)={d['T11_lag_ms']}  "
          f"pref_phase={d['_MI_pref_phase']} rad  ok={mi_res['ok']}")

    return d


# =============================================================================
# Step 4 — 生理学特征合理性预检检查 (防脏特征进入训练)
# =============================================================================
def run_sanity_checks(d):
    """
    根据 docs/sprint1_phase2_plan_v2.md §4.3 所定的生理健康边界进行判定。
    如果发现特征严重越界，说明算法/提取流程异常，将阻断正式训练。
    """
    failures = []
    warnings_msgs = []
    
    # 慢波主频必须在 [0.5, 1.5] Hz 生理区间
    if not (0.5 <= d["T4_freq"] <= 1.5):
        failures.append(f"T4_freq = {d['T4_freq']:.3f} not in [0.5, 1.5] Hz")
        
    # PAC up_down 比值如果小于 1.0 说明相位锁反（或锁定在 DOWN 态），予以警示
    if d["T11_lag_ms"] < 1.0:
        warnings_msgs.append(
            f"T11(up_down_ratio) = {d['T11_lag_ms']:.3f} < 1.0 — "
            "DOWN-locked? Verify before SBI."
        )
        
    # 纺锤事件过稀疏则报错
    if d["T8_n_sp_events"] <= 5:
        failures.append(f"T8_n_sp_events = {d['T8_n_sp_events']} <= 5")
        
    # T6 正确捕获并处于正常区间 [0.4, 0.55] 
    if not d.get("_T6_ok", False):
        failures.append(f"T6 ok=False — EEG-native AASM detected no valid UP events")
    elif not (0.4 <= d["T6_ibi_cv"] <= 0.55):
        warnings_msgs.append(
            f"T6_ibi_cv = {d['T6_ibi_cv']:.4f} outside expected [0.40, 0.55] "
            "for healthy N3. Check t6 diagnostics."
        )
        
    # MI 生理学调制指数强度检查
    if not d.get("_MI_ok", False):
        failures.append("MI ok=False")
    elif d["MI"] < 0.005:
        failures.append(
            f"MI = {d['MI']:.5f} below 0.005 — algorithm broken? "
            "Check Hilbert phase wrapping / bandpass design / phase anchor."
        )
    elif not (0.02 <= d["MI"] <= 0.05):
        # 如果超出 SC4001 典型范围，进行警示
        warnings_msgs.append(
            f"MI = {d['MI']:.5f} outside expected [0.02, 0.05] for SC4001"
        )
    return failures, warnings_msgs


# =============================================================================
# Step 5 — 主程序入口及落盘为 x_obs_v4.npz
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

    # 1. 载入原始脑电
    print("\n[Step 1]  Loading N3 EEG for SC4001 ...")
    eeg_uv, fs_native = load_n3_eeg()
    print(f"  Total N3 signal: {len(eeg_uv) / fs_native:.1f} s")

    # 2. 构建包络与去趋势波形
    print("\n[Step 2]  Building r_proxy + eeg_raw ...")
    r_proxy, eeg_raw = build_rate_proxy(eeg_uv, fs_native)

    # 3. 提取特征向量
    print("\n[Step 3]  Computing 7 summary statistics ...")
    d = compute_summaries(r_proxy, eeg_raw, fs=FS_SIM)

    print("\n  x_obs_v4 values:")
    for k in SUMMARY_KEYS:
        print(f"    {k:20s} = {d[k]}")

    # 4. 生理安全性检查
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

    # 5. 封装格式化元数据（Metadata）并落盘
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
