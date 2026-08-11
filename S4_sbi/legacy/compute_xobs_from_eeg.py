"""
compute_xobs_from_eeg.py
========================

从SC4001的真实N3 EEG中提取len(SUMMARY_KEYS)维的汇总统计向量 x_obs。

x_obs是SBI推断的目标。这些汇总统计量与V7中的compute_constraints_v7计算得到的结果保持一致，
因此神经后验会在相同的刻度下将模拟结果与观测结果进行映射。

发放率代理信号（FIRING-RATE PROXY）
-----------------
V7的统计量计算基于r_ctx(t): 发放率 >= 0，峰值约为60 Hz，窄的UP脉冲包裹在接近0的DOWN基线周围。
头皮EEG是双极（有正负号），单位为µV。两者之间的转换桥梁为：

    去趋势（detrend）→ 取绝对值（abs）→ 50ms高斯平滑 → 重标定到[0, 60]

这样可以保留EEG包络的统计特征（慢波速率、UP间隔规律性、纺锤爆发、慢波-纺锤耦合），
而无需假定EEG电压和神经发放率之间存在物理对应关系。

SBI后验应解释为：
  “那些其模拟r_ctx包络能复现与观测EEG包络相同振荡统计特征的参数”
而不是能够直接拟合EEG电压波形的参数。

SUMMARY_KEYS（顺序同simulator_wrapper.SUMMARY_KEYS；当前为7维）:
  0: shape_r        — 固定为1.0（EEG与自身匹配本就成立）
  1: T4_q           — SO主峰Q值（峰与邻域频段功率比）
  2: T4_freq        — SO主峰频率[Hz]
  3: T6_ibi_cv      — UP脉冲间隔的变异系数
  4: T8_n_sp_events — 每60秒检测到的纺锤事件数（代理信号长度归一化）
  5: T11_lag_ms     — PAC中的up_down_ratio（命名为T11_lag_ms以保持与V7一致）
  6: MI             — PAC调制指数（逐周期、fixed版本）

关于T11注释：V7在compute_constraints_v7中用“T11_lag_ms”这个key存储up_down_ratio；我们保持一致。

关于EEG的PAC说明：V7计算r_ctx和r_thal之间的PAC。但头皮EEG无法直接观察丘脑活动。
此处将r_proxy同时作为ctx和thal信号传入。因此，MI和up_down_ratio的尺度将与仿真得到的值不同；
SBI的密度估计器会学习这种映射关系。

用法（需在项目根目录下）:
    conda activate neurolib
    python S4_sbi/compute_xobs_from_eeg.py

保存：
    S4_sbi/x_obs.npz  （keys: values [float32数组], keys [字符串列表], 
                               extraction_metadata [JSON字符串]）
"""

import sys
import os
import json
import warnings
import importlib.util
from math import gcd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── 当前工作目录需为项目根目录 ──────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
os.chdir(str(_ROOT))
sys.path.insert(0, str(_ROOT))

# ── NumPy类型别名适配，需在任何neurolib/mne导入之前完成 ──────────────────────
import numpy as np
import builtins as _builtins_mod
for _alias in ("int", "float", "bool", "object", "complex", "str"):
    if not hasattr(np, _alias):
        setattr(np, _alias, getattr(_builtins_mod, _alias))

# ── 本地neurolib优先于系统安装的版本 ────────────────────────────────────────────
_LOCAL_NEUROLIB = Path(r"D:\Year3_Mao_Projects\neurolib")
if _LOCAL_NEUROLIB.is_dir() and str(_LOCAL_NEUROLIB) not in sys.path:
    sys.path.insert(0, str(_LOCAL_NEUROLIB))

# ── 标准库导入（需在NumPy shim之后） ────────────────────────────────────────────
import pandas as pd
from scipy.signal import welch, detrend, butter, sosfiltfilt, hilbert, resample_poly
from scipy.ndimage import gaussian_filter1d
import mne
mne.set_log_level("WARNING")

# ── 通过importlib加载02_preprocess_psd（因文件名以数字开头，不能直接import） ─────────
_prep_spec = importlib.util.spec_from_file_location(
    "preprocess_psd", str(_ROOT / "utils" / "02_preprocess_psd.py")
)
_prep_mod = importlib.util.module_from_spec(_prep_spec)
_prep_spec.loader.exec_module(_prep_mod)
load_hypnogram   = _prep_mod.load_hypnogram
compute_epoch_psd = _prep_mod.compute_epoch_psd
EPOCH_LEN_S      = _prep_mod.EPOCH_LEN_S  # 30秒

# ── 通过sys.path导入修正后的PAC指标计算模块 ───────────────────────────────────────
_repair_dir = _ROOT / "S4_v7_repair"
if str(_repair_dir) not in sys.path:
    sys.path.insert(0, str(_repair_dir))
from compute_pac_metrics_fixed import compute_pac_metrics  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# 配置参数（与s4_personalize_fig7_v7.py常量保持一致）
# ═══════════════════════════════════════════════════════════════════════════════
SUBJECT_ID      = "SC4001"
EEG_CHANNEL     = "EEG Fpz-Cz"          # 与V7的load_target_psd所用通道一致
N3_LABELS       = ["N3"]                 # R&K→AASM映射后的AASM标签
ARTIFACT_THRESH = 200e-6                 # V (200微伏峰峰值伪迹剔除)
FS_NATIVE       = 100.0                  # Sleep-EDF原始采样率 [Hz]
FS_SIM          = 1000.0                 # V7仿真采样率 [Hz]

# V7各项约束阈值（必须与s4_personalize_fig7_v7.py完全一致）
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
# 步骤1 — 加载N3期的EEG
# ═══════════════════════════════════════════════════════════════════════════════

def load_n3_eeg():
    """
    从SC4001中加载并拼接去除伪迹后的N3期数据。
    返回: (eeg_uv, fs_native)，其中eeg_uv是一维float64型µV数据。
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
    # data_uv是所有N3期的EEG数据，shape为(n_samples, 1)

    n_per_epoch = int(EPOCH_LEN_S * fs)
    # n_per_epoch是每个epoch的长度，单位为采样点
    n_epochs = min(len(stages), len(data_uv) // n_per_epoch)
    # n_epochs是N3期的epoch数
    accepted, n_n3, n_rej = [], 0, 0
    # accepted是接受的N3期的EEG数据，shape为(n_epochs, n_per_epoch)
    # n_n3是接受的N3期的epoch数
    # n_rej是拒绝的N3期的epoch数
    for i in range(n_epochs):
        if stages[i] not in N3_LABELS: # 如果当前epoch不是N3期，则跳过
            continue
        n_n3 += 1 # 接受的N3期的epoch数加1
        epoch = data_uv[i * n_per_epoch : (i + 1) * n_per_epoch] # 当前epoch的EEG数据，shape为(n_per_epoch, 1)
        if np.ptp(epoch) > ARTIFACT_THRESH * 1e6: # 如果当前epoch的EEG数据峰峰值大于阈值(200微伏峰峰值伪迹剔除) ，则跳过
            n_rej += 1 # 拒绝的N3期的epoch数加1
            continue
        accepted.append(epoch) # accepted是接受的N3期的EEG数据，shape为(n_epochs, n_per_epoch)

    if not accepted:
        raise RuntimeError(
            f"No N3 epochs passed artifact rejection for {SUBJECT_ID}"
        )# 如果没有接受的N3期的EEG数据，则抛出错误

    print(f"  EEG: {n_n3} N3 epochs total, {n_rej} artifact-rejected, "
          f"{len(accepted)} accepted  ({sum(len(e) for e in accepted)/fs:.0f} s)")# 打印接受的N3期的EEG数据的长度
    print(f"  Native fs = {fs} Hz")# 打印采样率

    return np.concatenate(accepted), float(fs)# 返回接受的N3期的EEG数据，shape为(n_epochs, n_per_epoch)，单位为采样点
    # 返回接受的N3期的EEG数据，shape为(n_epochs, n_per_epoch)，单位为采样点

# ═══════════════════════════════════════════════════════════════════════════════
# 步骤2 — 构建发放率代理信号
# ═══════════════════════════════════════════════════════════════════════════════

def build_rate_proxy(eeg_uv, fs_from):
    """
    将EEG电压映射为[0, 60]范围内的非负类似发放率的信号，采样率为FS_SIM。

    流程：重采样 → 去趋势 → 取绝对值 → 50ms高斯平滑 → 重标定。
    """
    # 重采样到FS_SIM (100 → 1000 Hz)
    g = gcd(int(FS_SIM), int(fs_from)) # 计算FS_SIM和fs_from的最大公约数
    eeg_1k = resample_poly(eeg_uv, int(FS_SIM) // g, int(fs_from) // g) # 重采样到FS_SIM
    # eeg_1k是重采样后的EEG数据，shape为(n_samples, 1), eeg_uv是原始EEG数据，shape为(n_samples, 1)
    # 线性去趋势
    eeg_det = detrend(eeg_1k, type="linear")
    # eeg_det是去趋势后的EEG数据，shape为(n_samples, 1)
    # 取绝对值
    r = np.abs(eeg_det)
    # r是取绝对值后的EEG数据，shape为(n_samples, 1)
    # 50ms高斯平滑（σ = 1000Hz下的50个采样点）。
    # 重要提示：如果没有这一步，abs()会在每个零交叉处产生V形尖峰，影响SO UP峰检测。
    r_smooth = gaussian_filter1d(r, sigma=50.0)
    # r_smooth是50ms高斯平滑后的EEG数据，shape为(n_samples, 1)
    # 重标定到[0, 60]（与V7的UP状态发放率范围一致）
    r_proxy = r_smooth - r_smooth.min()
    # r_proxy是重标定后的EEG数据，shape为(n_samples, 1)
    p95 = np.percentile(r_proxy, 95)
    # p95是r_proxy的95%分位数
    if p95 < 1e-9: # 如果p95小于1e-9，则抛出错误
        raise RuntimeError("EEG proxy 95th-percentile ≈ 0 — 检查原始数据质量")
    r_proxy = r_proxy / p95 * 60.0
    # r_proxy是重标定后的EEG数据，shape为(n_samples, 1), 换句话说，r_proxy是[0, 60]范围内的非负类似发放率的信号
    print(f"  r_proxy: {len(r_proxy)} samples  "
          f"mean={r_proxy.mean():.2f}  max={r_proxy.max():.2f}  "
          f"95pct={np.percentile(r_proxy, 95):.2f}  fs={FS_SIM} Hz")
    return r_proxy # 返回重标定后的EEG数据，shape为(n_samples, 1), 换句话说，r_proxy是[0, 60]范围内的非负类似发放率的信号


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤3 — 计算7维汇总统计量
# ═══════════════════════════════════════════════════════════════════════════════

def compute_summaries(r_proxy, fs=FS_SIM):
    """
    从r_proxy中计算全部7个汇总统计量，复用V7的约束逻辑。
    """
    d = {}

    # ── shape_r = 1.0 ─────────────────────────────────────────────────────────
    # EEG本身与自身完全吻合，作为基准值。
    # 仿真中，shape_r为FOOOF提取得到的PSD与目标PSD的Pearson相关系数。
    # 设定x_obs[shape_r]=1.0将后验锚定为“能重现这种频谱的模拟”。
    d["shape_r"] = 1.0

    # ── T4: SO主峰频率和Q值 ─────────────────────────────────────────────────────
    # 完全复刻compute_constraints_v7的T4代码块。
    f_c, p_c = compute_epoch_psd(r_proxy, fs) # 计算功率谱密度，f_c是频率，p_c是功率
    so_mask     = (f_c >= SO_FREQ_LO) & (f_c <= SO_FREQ_HI) # so_mask掩码，用于掩码SO主峰频率
    so_width    = SO_FREQ_HI - SO_FREQ_LO # so_width是SO主峰宽度
    neigh_lo    = (f_c >= max(0.1, SO_FREQ_LO - so_width)) & (f_c < SO_FREQ_LO) # neigh_lo是邻近频率的掩码
    neigh_hi    = (f_c > SO_FREQ_HI) & (f_c <= SO_FREQ_HI + so_width) # neigh_hi是邻近频率的掩码
    so_peak_freq, so_q = 0.0, 0.0 # so_peak_freq是SO主峰频率，so_q是SO主峰Q值
    if so_mask.any(): # 如果so_mask不为空则计算SO主峰频率和Q值
        so_peak_freq = float(f_c[so_mask][np.argmax(p_c[so_mask])]) # 计算SO主峰频率 so_peak_freq是SO主峰频率
        so_peak_val  = float(p_c[so_mask].max()) # so_peak_val是SO主峰功率
        nbrs = np.concatenate([
            p_c[neigh_lo] if neigh_lo.any() else np.array([]),
            p_c[neigh_hi] if neigh_hi.any() else np.array([]),
        ]) # nbrs是邻近功率的数组
        if len(nbrs) > 0 and nbrs.mean() > 0: # 如果邻近功率不为空且平均功率大于0则计算SO主峰Q值
            so_q = float(so_peak_val / nbrs.mean()) # so_q是SO主峰Q值，so_peak_val是SO主峰功率/nbrs.mean()是邻近功率的平均功率
            # 换句话说，so_q是SO主峰Q值，so_peak_val是SO主峰功率/邻近功率的平均功率
    d["T4_q"]    = round(so_q, 3) # 将SO主峰Q值四舍五入到3位小数,记录在d["T4_q"]中
    d["T4_freq"] = round(so_peak_freq, 3) # 将SO主峰频率四舍五入到3位小数，记录在d["T4_freq"]中

    # ── T6: IBI变异系数 ────────────────────────────────────────────────────────
    # r_proxy存在亚秒级幅度波动（纺锤等），直接阈值会误检UP事件。
    # 先作500ms额外平滑，用于分离SO包络时间尺度再检测UP事件。
    # 中位阈值约可得到~50%的UP占空比，符合睡眠SO生理特征。
    r_for_ibi = gaussian_filter1d(r_proxy, sigma=500.0) # r_for_ibi是500ms高斯平滑后的EEG数据，shape为(n_samples, 1)
    up_thresh  = float(np.percentile(r_for_ibi, 50)) # up_thresh是50%分位数，up_thresh是UP阈值
    print(f"  T6 SO-envelope UP threshold: {up_thresh:.2f} Hz (500ms smooth, 50th pct)")
    # 打印UP阈值
    above  = (r_for_ibi > up_thresh).astype(np.int8) # above是UP事件的掩码
    diff_  = np.diff(np.concatenate(([0], above, [0]))) # diff_是UP事件的差分
    starts = np.where(diff_ == 1)[0] # starts是UP事件的起始时间
    n_bursts = len(starts) # n_bursts是UP事件的数量
    ibi_cv = 999.0 # ibi_cv是IBI变异系数
    if n_bursts >= 3: # 如果UP事件的数量大于等于3则计算IBI变异系数，换句话说，如果UP事件的数量小于3则IBI变异系数为999.0
        intervals = np.diff(starts) / fs # intervals是UP事件的间隔时间
        ibi_cv = float(intervals.std() / (intervals.mean() + 1e-12)) # ibi_cv是IBI变异系数，
        # intervals.std()是间隔时间的标准差/intervals.mean()是间隔时间的平均值+1e-12是防止除以0
    d["T6_ibi_cv"] = round(ibi_cv, 3) # 将IBI变异系数四舍五入到3位小数，记录在d["T6_ibi_cv"]中

    # ── T8: 纺锤事件数（归一化为每60秒多少个事件） ─────────────
    # V7仿真信号长度正好为60秒（去除burn-in后），EEG代理信号覆盖duration_s秒。
    # 这里做归一化，使x_obs的T8和仿真输出处于相同单位（事件数/60秒）。
    # 若仿真为60秒归一化系数为1.0。
    duration_s = float(len(r_proxy)) / fs # duration_s是EEG代理信号的长度
    n_sp_events = 0 # n_sp_events是纺锤事件的数量
    try:
        sos = butter(4, [SPINDLE_LO, SPINDLE_HI], btype="band", fs=fs, output="sos") # sos是滤波器
        filtered = sosfiltfilt(sos, r_proxy) # filtered是滤波后的EEG数据，shape为(n_samples, 1)
        envelope = np.abs(hilbert(filtered)) # envelope是包络信号，shape为(n_samples, 1)
        sigma_samp = SPINDLE_ENV_SMOOTH_MS * fs / 1000.0 # sigma_samp是高斯平滑的参数，单位为采样点
        env_sm = gaussian_filter1d(envelope, sigma=sigma_samp) # env_sm是高斯平滑后的包络信号，shape为(n_samples, 1)
        thresh  = np.percentile(env_sm, SPINDLE_EVT_PCTILE) # thresh是阈值，单位为采样点
        ab_sp   = (env_sm > thresh).astype(np.int8) # ab_sp是掩码，用于掩码纺锤事件
        diff_sp = np.diff(np.concatenate(([0], ab_sp, [0]))) # diff_sp是纺锤事件的差分
        sp_st   = np.where(diff_sp == 1)[0] # sp_st是纺锤事件的起始时间
        sp_en   = np.where(diff_sp == -1)[0] # sp_en是纺锤事件的结束时间
        durs    = (sp_en - sp_st) / fs # durs是纺锤事件的持续时间，单位为秒
        valid   = (durs >= SPINDLE_DUR_LO_S) & (durs <= SPINDLE_DUR_HI_S) # valid是掩码，用于掩码纺锤事件
        n_sp_events = int(valid.sum()) # n_sp_events是纺锤事件的数量
    except Exception as exc: # 如果出现错误，则打印错误信息
        print(f"  [warn] Spindle detection error: {exc}") # 打印错误信息
    t8_normalized = n_sp_events * (60.0 / duration_s) # t8_normalized是纺锤事件的数量/60秒，
    # 换句话说，t8_normalized是纺锤事件的数量/60秒
    print(f"  T8 raw={n_sp_events}  duration={duration_s:.0f}s  "
          f"normalized (per 60s)={t8_normalized:.2f}")
    d["T8_n_sp_events"] = round(t8_normalized, 3) # 将纺锤事件的数量/60秒四舍五入到3位小数，记录在d["T8_n_sp_events"]中

    # ── T9-T11: PAC耦合指标（逐周期，fixed版本） ────────────────────────────────
    # 同时将r_proxy作为ctx和thal信号传入。
    # PAC函数用r_ctx检测SO相位，r_thal提取纺锤包络；
    # 丘脑活动在头皮EEG不直接可见，这是一种技术性简化。
    pac = compute_pac_metrics(r_proxy, r_proxy, fs=fs) # pac是PAC耦合指标
    d["T11_lag_ms"] = round(pac.get("up_down_ratio", 0.0), 3)  # 保持V7命名
    d["MI"]         = round(pac.get("mi", 0.0), 5) # 将PAC耦合指标四舍五入到5位小数，记录在d["MI"]中

    return d


# ═══════════════════════════════════════════════════════════════════════════════
# 步骤4 — 严格合理性检查（遇到失败立即终止，不要加兜底处理）
# ═══════════════════════════════════════════════════════════════════════════════

def run_sanity_checks(d):
    """返回所有未通过的检查说明（全通过返回空列表）。"""
    failures = []
    if not (0.5 <= d["T4_freq"] <= 1.5): # 如果SO主峰频率不在[0.5, 1.5] Hz之间，则添加失败信息
        failures.append(f"T4_freq = {d['T4_freq']:.3f} not in [0.5, 1.5] Hz")
    if not (0.1 <= d["T6_ibi_cv"] <= 0.8): # 如果IBI变异系数不在[0.1, 0.8]之间，则添加失败信息
        failures.append(f"T6_ibi_cv = {d['T6_ibi_cv']:.3f} not in [0.1, 0.8]")
    if d["T11_lag_ms"] < 1.0:
        failures.append(
            f"T11_lag_ms (up_down_ratio) = {d['T11_lag_ms']:.3f} < 1.0"
        ) # 如果PAC耦合指标小于1.0，则添加失败信息
    if d["T8_n_sp_events"] <= 5: # 如果纺锤事件的数量小于等于5，则添加失败信息
        failures.append(f"T8_n_sp_events = {d['T8_n_sp_events']} <= 5") # 添加失败信息
    return failures # 返回失败信息


# ═══════════════════════════════════════════════════════════════════════════════
# 主程序入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("compute_xobs_from_eeg.py  —  x_obs extraction for SBI")
    print("=" * 64)

    # 步骤1
    print("\n[Step 1]  Loading N3 EEG for SC4001 ...")
    eeg_uv, fs_native = load_n3_eeg()  # eeg_uv是N3期的EEG数据，shape为(n_samples, 1), fs_native是采样率
    print(f"  Total N3 signal: {len(eeg_uv) / fs_native:.1f} s") # 打印N3期的EEG数据的长度

    # 步骤2
    print("\n[Step 2]  Building firing-rate proxy ...")
    r_proxy = build_rate_proxy(eeg_uv, fs_native) # r_proxy是发放率代理信号，shape为(n_samples, 1)

    # 步骤3
    print("\n[Step 3]  Computing 7 summary statistics ...")
    d = compute_summaries(r_proxy) # d是汇总统计量，shape为(7, 1)
    # 打印汇总统计量
    print("\n  x_obs values:")
    for k in SUMMARY_KEYS:
        print(f"    {k:20s} = {d[k]}") # 打印汇总统计量

    # 步骤4
    print("\n[Step 4]  Sanity checks ...")
    failures = run_sanity_checks(d) # failures是失败信息，shape为(n_failures, 1)
    if failures:
        print("\n  *** SANITY CHECK FAILED — DO NOT PROCEED ***")
        for msg in failures:
            print(f"    FAIL: {msg}") 
        raise RuntimeError( # 如果失败信息不为空，则抛出错误
            "x_obs sanity checks failed. Review data / proxy parameters "
            "before launching SBI."
        ) # 如果失败信息不为空，则抛出错误
    print("  All 4 sanity checks PASSED.")

    # 步骤5
    x_obs_values = np.array([d[k] for k in SUMMARY_KEYS], dtype=np.float32)
    # x_obs_values是汇总统计量，shape为(7, 1)
    metadata = { # 打印元数据
        "subject_id":    SUBJECT_ID, # 被试ID
        "eeg_channel":   EEG_CHANNEL, # EEG通道
        "n3_labels":     N3_LABELS, # N3期标签
        "proxy_method":  "detrend -> abs -> 50ms_gaussian_smooth -> rescale_to_[0,60]", # 代理方法
        "fs_native_hz":  float(fs_native), # 原始采样率
        "fs_resampled_hz": FS_SIM, # 重采样后的采样率
        "n_samples_proxy": int(len(r_proxy)), # 代理信号的样本数
        "duration_s":    float(len(r_proxy) / FS_SIM), # 代理信号的持续时间
        "artifact_thresh_uv": ARTIFACT_THRESH * 1e6, # 伪迹阈值
    } # 元数据

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True) # 创建输出路径
    np.savez( # 保存汇总统计量
        str(OUTPUT_PATH), # 输出路径
        values=x_obs_values, # 汇总统计量
        keys=SUMMARY_KEYS, # 汇总统计量名称
        extraction_metadata=json.dumps(metadata), # 元数据
    ) # 保存汇总统计量
    print(f"\n[Step 5]  Saved to {OUTPUT_PATH}") # 打印输出路径
    print(f"  x_obs  shape={x_obs_values.shape}  dtype={x_obs_values.dtype}")
    return x_obs_values # 返回汇总统计量，shape为(7, 1)


if __name__ == "__main__":
    main()
