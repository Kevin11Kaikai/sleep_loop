"""
s4_personalize_fig7.py
=====================
Fig. 7–style personalised fitting of a thalamocortical model to a single
subject's N3 EEG, following the methodology of Cakan et al. (2023).

Key improvements over s4_personalize.py:
    1. FOOOF 1/f subtraction on BOTH target EEG and simulated PSD
    2. 8 free parameters (adds b, tauA, c_ctx→th, c_th→ctx)
    3. Shape-only fitness: corr(residuals)
    4. Larger search scale (popsize=15, 20 generations)
    5. Generates Fig. 7 (a)(b)(c)(d) panels

Requirements:
  pip install neurolib fooof scipy matplotlib mne pandas

Usage:
  python models/s4_personalize_fig7.py
"""

import os, sys, json, fnmatch, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.optimize import differential_evolution

from neurolib.models.multimodel import MultiModel, ALNNode, ThalamicNode
from neurolib.models.multimodel.builder.base.network import Network
from neurolib.models.multimodel.builder.base.constants import EXC, INH

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUBJECT_ID      = "SC4001"
EEG_CHANNELS    = ["EEG Fpz-Cz", "EEG Pz-Oz"]
N3_LABELS       = ["Sleep stage 3", "Sleep stage 4"]
ARTIFACT_THRESH = 200e-6 # 峰峰值超过 200 μV 的 epoch 被视为伪迹，跳过不纳入目标 PSD 计算；s4_personalize.py 中无此 QC 步骤
EPOCH_DURATION  = 30.0 # 每个 epoch 30 s，与 Fig. 7 / 论文展示一致；s4_personalize.py 中常截 20 s，二者勿混

# Evolution config — increase for better results (at cost of time)
DE_POPSIZE = 15     # DE_POPULATION_SIZE；每代评估 15 个个体，每个个体跑一次仿真；paper 用 20，但时间较长，20 可能更稳健但不太现实
N_GEN      = 20     # N_GEN；DE_GENERATIONS；迭代 20 代；paper 用 50，但时间较长，20 可能更稳健但不太现实
SIM_DUR_MS = 30_000 # 30 s per evaluation

# Frequency range for fitness
F_LO, F_HI = 0.5, 20.0   # paper uses 0-20 Hz for PSD correlation

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 1: Network definition（源自 s3_sleep_kernel，本脚本有两点扩展）
#   • 丘脑↔皮层连接强度 c_th2ctx / c_ctx2th 由构造参数传入，供 DE 作为自由参数搜索；
#   • b、tauA 不在此写死，由 build_model() 注入（s3/s4 基线版常固定为 Table 3）。
# 结构仍为 2 节点：节点 0 = ALNNode（皮层），节点 1 = ThalamicNode（丘脑）。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def set_params_glob(model, pattern, value):
    """按 glob 模式匹配 MultiModel.params 的长键名并批量赋值（与 s3/s4 相同）。"""
    for k in (k for k in model.params if fnmatch.fnmatch(k, pattern)):
        model.params[k] = value


class ThalamoCorticalNetwork(Network):
    """
    丘脑–皮层双节点 motif（Cakan 等多篇 neurolib 论文中的 TC 结构）。

    与 s3 相同之处：
      - ALNNode（皮层 E/I）+ ThalamicNode（TCR/TRN）；
      - 各 mass 需要的网络耦合符号见 sync_variables，在 _sync() 里接到
        network_exc_exc / network_exc_exc_sq / network_inh_exc；
      - _EXC_WITHIN_IDX：各节点内用于耦合的 r_mean_EXC 在局部状态向量中的下标
       （ALN 兴奋 mass=6，丘脑 TCR=9）。

    与 s3 不同之处：
      - __init__(c_th2ctx, c_ctx2th) 把两条有向边强度作为参数；矩阵为 [to, from]，
        C[0,1]=丘脑→皮层，C[1,0]=皮层→丘脑，对角为 0。
    """
    name  = "Thalamocortical Motif"
    label = "TCNet"
    # ALN 兴奋：network_exc_exc + network_exc_exc_sq；TCR：network_exc_exc；TRN：network_inh_exc
    sync_variables  = ["network_exc_exc", "network_exc_exc_sq", "network_inh_exc"]
    # sync_variables 定义了网络耦合符号；ALN 兴奋性 r_mean_EXC 既受 network_exc_exc（线性）又受 network_exc_exc_sq（非线性）调制；
    # TCR 兴奋性 r_mean_EXC 受 network_exc_exc 调制；TRN 兴奋性 r_mean_EXC 受 network_inh_exc 调制。
    default_output  = f"r_mean_{EXC}" # 默认输出皮层兴奋性 r_mean_EXC，供 DE 评估适应度时直接使用；
    # 也可在 build_model 后改为其他变量（如 TCR 兴奋性）
    output_vars     = [f"r_mean_{EXC}", f"r_mean_{INH}"]
    # output_vars 定义了可选的输出变量；默认输出是皮层兴奋性 r_mean_EXC，但也可以选择其他变量，如丘脑兴奋性 r_mean_EXC 或任意 r_mean_INH。
    _EXC_WITHIN_IDX = [6, 9]   # 节点内 r_mean_EXC 下标：皮层 / 丘脑

    def __init__(self, c_th2ctx=0.02, c_ctx2th=0.15):
        # 节点实例与全局状态布局（必须在 super() 前设好 noise_input_idx，见 s3 注释）
        aln = ALNNode(exc_seed=42, inh_seed=42)
        th  = ThalamicNode()
        aln.index = 0;  aln.idx_state_var = 0
        th.index  = 1;  th.idx_state_var  = aln.num_state_variables
        for i, node in enumerate([aln, th]):
            for mass in node:
                mass.noise_input_idx = [2 * i + mass.index]

        # 连接矩阵 [to, from]：非零元仅两条跨节点边，强度由优化变量给出
        connectivity = np.array([[0.0, c_th2ctx],
                                 [c_ctx2th, 0.0]])
        super().__init__(nodes=[aln, th],
                         connectivity_matrix=connectivity,
                         delay_matrix=np.zeros_like(connectivity))

    def _sync(self):
        # 先节点内部耦合，再用 Network._additive_coupling 把各节点 r_mean_EXC 按 C 加权
        # 接到三条网络符号；其中 network_exc_exc_sq 使用 C**2（供 ALN 适应项）
        couplings = sum([node._sync() for node in self], [])
        wi = self._EXC_WITHIN_IDX
        couplings += self._additive_coupling(wi, "network_exc_exc")
        # network_exc_exc 是线性耦合项，ALN 兴奋性 r_mean_EXC 受来自丘脑的线性调制；TCR 兴奋性 r_mean_EXC 受来自皮层的线性调制。
        couplings += self._additive_coupling(wi, "network_exc_exc_sq",
                                             connectivity=self.connectivity ** 2)
        # network_exc_exc_sq 是非线性耦合项，ALN 兴奋性 r_mean_EXC 受来自丘脑的非线性调制（C**2）；TCR 兴奋性 r_mean_EXC 不受非线性调制。
        couplings += self._additive_coupling(wi, "network_inh_exc")
        # network_inh_exc 是抑制性耦合项，ALN 兴奋性 r_mean_EXC 受来自丘脑的抑制调制；TCR 兴奋性 r_mean_EXC 不受抑制调制。
        return couplings


def build_model(mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th):
    """
    每次适应度评估新建一套 MultiModel（因 c_th2ctx/c_ctx2th 与 b/tauA 随个体变化）。

    流程：ThalamoCorticalNetwork(耦合) → MultiModel → 积分设置 → set_params_glob 写动力学参数。
    TRN 的 g_LK 仍固定 0.1；TCR 的 g_LK、g_h 与皮层 mue/mui、b、tauA 由参数传入。
    """
    net = ThalamoCorticalNetwork(c_th2ctx=c_th2ctx, c_ctx2th=c_ctx2th)
    m = MultiModel(net)
    m.params["backend"]     = "numba"
    m.params["dt"]          = 0.1
    m.params["sampling_dt"] = 1.0
    m.params["duration"]    = SIM_DUR_MS

    set_params_glob(m, "*ALNMassEXC*.input_0.mu", mue)
    set_params_glob(m, "*ALNMassINH*.input_0.mu", mui)
    set_params_glob(m, "*ALNMassEXC*.b",          b)
    set_params_glob(m, "*ALNMassEXC*.tauA",       tauA)
    set_params_glob(m, "*ALNMassEXC*.a",          0.0)
    set_params_glob(m, "*ALNMass*.input_0.sigma",  0.05)
    set_params_glob(m, "*TCR*.input_0.sigma",      0.005)
    set_params_glob(m, "*.input_0.tau",            5.0)
    set_params_glob(m, "*TRN*.g_LK",              0.1)
    set_params_glob(m, "*TCR*.g_LK",              g_lk)
    set_params_glob(m, "*TCR*.g_h",               g_h)
    return m


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 2：单人 N3 目标 PSD（与 s4_personalize.py / s1_all_stages 同一套路，频段见下）
#
#   目的：得到 target_psd、target_freqs，供 Part 3（FOOOF）与 Part 4（适应度）使用。
#   流程：manifest 取路径 → 读 PSG + Hyp → 仅 N3 标签生成 events → 30 s Epochs →
#         逐 epoch 伪迹 → 双通道平均 → Welch → 保留 [F_LO, F_HI]（此处 0.5–20 Hz，
#         对齐 Fig.7 / 本文 FSIM；s4 常截 0.5–30 Hz，二者勿混）。
#   聚合：仅通过 QC 的 epoch 的 PSD 在 epoch 维求平均 = 该受试者 N3 代表谱。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import mne
mne.set_log_level("WARNING")

manifest = pd.read_csv("data/manifest.csv")
subj_row = manifest[manifest["subject_id"] == SUBJECT_ID].iloc[0]

print(f"Building personal N3 PSD for {SUBJECT_ID}...")
# 仅 EEG 两路，preload 供 Epochs 使用
raw = mne.io.read_raw_edf(subj_row["psg_path"], include=EEG_CHANNELS,
                           preload=True, verbose=False)
fs_eeg = raw.info["sfreq"]
raw.set_annotations(mne.read_annotations(subj_row["hypnogram_path"]))

# N3 + N4 标签 → 与 s1 STAGE_MAP["n3"] 一致
event_id = {lbl: idx+1 for idx, lbl in enumerate(N3_LABELS)}
events, event_dict = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
# tmin=0：从分期事件起点对齐；无 baseline（与 s4 一致）
epochs_n3 = mne.Epochs(raw, events, event_id=event_dict,
                       tmin=0.0, tmax=EPOCH_DURATION,
                       baseline=None, preload=True, verbose=False)

psds_subject = []
for ep_idx in range(len(epochs_n3)):
    data = epochs_n3[ep_idx].get_data()[0]   # (channels, time)
    # 峰峰值伪迹，超过 ARTIFACT_THRESH 则跳过
    if np.any((data.max(axis=1) - data.min(axis=1)) > ARTIFACT_THRESH):
        continue
    mean_signal = data.mean(axis=0)   # 双通道平均
    nperseg = min(int(10.0 * fs_eeg), len(mean_signal))
    f_ep, p_ep = welch(mean_signal, fs=fs_eeg, nperseg=nperseg,
                       noverlap=nperseg//2, window="hann")
    # f_ep 是频率数组，p_ep 是对应的功率谱密度；保留 [F_LO, F_HI] 频段，供后续对齐与 FOOOF 使用
    freq_mask = (f_ep >= F_LO) & (f_ep <= F_HI)
    #freq_mask是一个布尔数组，标记了哪些频率在指定范围内；p_ep[freq_mask] 则提取了这些频率对应的功率谱密度值。
    psds_subject.append(p_ep[freq_mask])
    # psds_subject 是一个列表，收集了每个通过 QC 的 epoch 的功率谱密度；每个元素是一个数组，对应于 [F_LO, F_HI] 频段的功率谱密度值。

n_passed = len(psds_subject)
# n_passed 是通过 QC (QC指的是 Quality Control ) 的 epoch 数量；如果 n_passed 为 0，说明没有任何 epoch 通过 QC，无法计算目标 PSD，因此抛出错误。
print(f"  N3 epochs passed QC: {n_passed} / {len(epochs_n3)}")
# 输出通过 QC 的 epoch 数量与总 epoch 数量的比率，供用户了解数据质量和可用性。
if n_passed == 0:
    raise ValueError("No N3 epochs passed QC")

# 最后一个通过 QC 的 epoch 给出 f_ep/freq_mask；通常各 epoch 格点一致
target_psd   = np.mean(psds_subject, axis=0)
# target_psd 是通过 QC 的 epoch 的功率谱密度的平均值，代表了该受试者 N3 EEG 的典型 PSD；对齐到 [F_LO, F_HI] 频段。
target_freqs = f_ep[freq_mask]
# target_freqs 是对应于 target_psd 的频率数组，包含了 [F_LO, F_HI] 频段内的频率值；与 target_psd 的长度一致。
print(f"  Target PSD shape: {target_psd.shape}, freq: {target_freqs[0]:.1f}-{target_freqs[-1]:.1f} Hz")
# 输出 target_psd 的形状和对应的频率范围，供用户确认目标 PSD 的维度和频段是否正确。


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 3：对目标 EEG 功率谱做 FOOOF（论文式「去 1/f」）
#
#   动机：原始 PSD 含强 1/f 背景；直接和仿真发放率 PSD 比形状易被低频倾斜支配。
#   FOOOF 将 log10(PSD) 分解为「非周期项（aperiodic）+ 高斯峰（periodic）」；
#   本段用 周期残差 = log10(PSD) − aperiodic_fit（内部量 _ap_fit），突出振荡峰相对背景，
#   供 Part 4 与仿真侧同样处理后的谱做 Pearson 相关及 SO/纺锤峰功率。
#
#   若未安装 fooof：退化为「总功率归一化 + log10」，接近 s4 的归一化谱形状比较，但无显式 1/f 减除。
#
#   产出：target_periodic（与 target 对齐的残差序列）、fooof_freqs（与 FOOOF 网格一致，仿真插值后用）、
#         target_aperiodic（可选，用于 Part 8 图 d 画 1/f 拟合曲线）。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
try:
    from fooof import FOOOF
    HAS_FOOOF = True
except ImportError:
    print("  [warn] fooof not installed; falling back to normalised PSD")
    HAS_FOOOF = False

# 无 FOOOF 时 target_aperiodic 保持 None；target_periodic 在分支内赋值
target_periodic = None
target_aperiodic = None

if HAS_FOOOF:
    # peak_width_limits / max_n_peaks：约束慢波、纺锤等峰形；aperiodic_mode='fixed'：单指数 1/f
    fm_tgt = FOOOF(peak_width_limits=[1.0, 8.0], max_n_peaks=4,
                   min_peak_height=0.05, aperiodic_mode='fixed')
    # 直接 fit 0.5–20 Hz 的目标 PSD；FOOOF 内部会对数变换并拟合非周期项，得到残差谱（周期项）供后续使用
    fm_tgt.fit(target_freqs, target_psd, [F_LO, F_HI])
    # FOOOF 的 fit() 内部会对数变换 target_psd，并拟合出 aperiodic_fit（_ap_fit）和 periodic_peaks（peak_params_）；此处用 log10(PSD) − aperiodic_fit 近似周期残差，供后续与仿真对齐比较。

    # 在 log 域：全谱 = aperiodic_line + peaks；此处用「数据 log 谱 − 拟合的非周期线」近似周期残差
    target_log_psd = np.log10(target_psd + 1e-30)
    fooof_freqs = fm_tgt.freqs          # FOOOF 内部频率格，可能与 target_freqs 长度略有差异
    target_aperiodic = fm_tgt._ap_fit   # log10 域上的非周期拟合曲线（私有属性，与论文脚本一致用法）
    target_periodic = target_log_psd[: len(target_aperiodic)] - target_aperiodic
    #target_periodic是减去非周期成分target_log_psd与target_aperiodic得到的周期残差，
    # 供后续与仿真侧同样处理后的谱做 Pearson 相关及 SO/纺锤峰功率。

    print(f"  FOOOF target: aperiodic exponent={fm_tgt.aperiodic_params_[1]:.2f}")
    # FOOOF 的 aperiodic_params_ 包含了非周期项的参数；对于 'fixed' 模式，第二个参数是 1/f 的指数；输出供用户了解目标 PSD 的 1/f 特性。
    if fm_tgt.peak_params_ is not None and len(fm_tgt.peak_params_) > 0:
        for pk in fm_tgt.peak_params_: 
            print(f"    Peak: {pk[0]:.1f} Hz, power={pk[1]:.3f}, width={pk[2]:.1f}")
            # 输出 FOOOF 拟合出的每个峰的中心频率、功率和宽度，供用户了解目标 PSD 中显著的振荡成分。
else:
    # 无 FOOOF：用归一化功率的 log 谱代替周期残差，仍可与仿真归一化谱比形状，但无 1/f 显式剥离
    target_periodic = np.log10(target_psd / (target_psd.sum() + 1e-30) + 1e-30)
    fooof_freqs = target_freqs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 4：适应度 compute_fitness（Fig. 7；与 Part 5 的 _objective 合成标量 score）
#
#   输入：8 维参数 → build_model → 跑 SIM_DUR_MS（30 s）。
#   信号：仅皮层节点兴奋性 r_E（Hz），非 EEG；与 Part 3 的 EEG 目标在「各自去 1/f 后的残差」上比形状。
#   流程：丢 5 s → Welch → [F_LO,F_HI] → 插值到 fooof_freqs →（可选）FOOOF 得 sim_periodic。
#   返回：(shape_r, so_power, spindle_power)；当前仅保留 shape_r 计入 score。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# sampling_dt=1 ms → 1000 Hz，与 Welch、n_drop=5 s 的样本数一致
FS_SIM = 1000.0


def compute_fitness(mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th):
    """
    单次评估返回三元组；Part 5 中 score = shape_r（仅分量 1）。

    shape_r：target_periodic（Part 3）与 sim_periodic（本函数）对齐后的 Pearson r；
    so/spindle：保留字段但不再参与优化目标。
    """
    try:
        m = build_model(mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th)
        m.run()
    except Exception:
        try:
            m.params["backend"] = "jitcdde"
            m.run()
        except Exception:
            return -1.0, -1.0, -1.0

    # 皮层节点（index 0）兴奋性群体发放率；MultiModel 输出单位为 kHz
    r_exc = m[f"r_mean_{EXC}"]
    r_sim = (r_exc[0] if r_exc.ndim == 2 else r_exc) * 1000  # kHz → Hz

    n_drop = int(5.0 * FS_SIM)   # 去掉起始瞬态，与 s3/test_spindles 一致
    r_sim = r_sim[n_drop:]
    if r_sim.max() < 0.1:
        return -1.0, -1.0, -1.0

    nperseg = min(int(10.0 * FS_SIM), len(r_sim))
    f_s, p_s = welch(r_sim, fs=FS_SIM, nperseg=nperseg, noverlap=nperseg//2, window="hann")
    # f_s 是频率数组，p_s 是对应的功率谱密度；

    ms = (f_s >= F_LO) & (f_s <= F_HI)
    # ms 是一个布尔数组，标记了哪些频率在指定范围内；
    f_s, p_s = f_s[ms], p_s[ms]
    # f_s 和 p_s 现在仅包含了 [F_LO, F_HI] 频段内的频率和对应的功率谱密度值；与 Part 3 的 target_periodic 对齐。
    # f_s, p_s 是仿真侧的频率和功率谱密度；后续会对 p_s 进行 FOOOF 或归一化处理，以得到 sim_periodic 供与 target_periodic 比较。
    if len(f_s) < 10:
        return -1.0, -1.0, -1.0

    # 与 Part 3 的 fooof_freqs 对齐，便于 sp/tp 同维比较
    p_interp = interp1d(f_s, p_s, bounds_error=False,
                        fill_value=1e-30)(fooof_freqs)
    # p_interp 是通过线性插值将仿真侧的功率谱密度 p_s 从 f_s 频率点映射到 Part 3 的 fooof_freqs 频率点；
    # 这样 sim_periodic 就能与 target_periodic 在同一频率格点上进行比较。
    if HAS_FOOOF:
        try:
            fm_sim = FOOOF(peak_width_limits=[1.0, 8.0], max_n_peaks=4,
                           min_peak_height=0.05, aperiodic_mode='fixed')
            # fm_sim 是一个 FOOOF 对象，用于拟合仿真侧的功率谱密度；参数设置与 Part 3 的 fm_tgt 一致，确保拟合方式相同。
            fm_sim.fit(fooof_freqs, p_interp, [F_LO, F_HI])
            # 直接用 log10(PSD) − aperiodic_fit 近似周期残差，供后续与目标周期残差比形状及峰值；与 Part 3 一致用法。
            sim_log = np.log10(p_interp[:len(fm_sim._ap_fit)] + 1e-30)
            # sim_log 是仿真侧功率谱密度 p_interp 的 log10 变换；加上 1e-30 避免 log(0) 的数值问题；仅取与 fm_sim._ap_fit 长度一致的部分，供后续计算周期残差。
            sim_periodic = sim_log - fm_sim._ap_fit
            # sim_periodic 是仿真侧的周期残差，计算方式与 Part 3 的 target_periodic 一致；供后续与 target_periodic 比形状及峰值。
        except Exception:
            p_rel = p_interp / (p_interp.sum() + 1e-30)
            sim_periodic = np.log10(p_rel + 1e-30)
    else:
        p_rel = p_interp / (p_interp.sum() + 1e-30)
        sim_periodic = np.log10(p_rel + 1e-30)

    n = min(len(sim_periodic), len(target_periodic))
    # n 是 sim_periodic 和 target_periodic 的最小长度，确保后续比较时两者维度一致；通常两者长度应该相同，但以防万一做了保护。
    sp = sim_periodic[:n]
    # sp 是仿真侧的周期残差，仅取前 n 个元素，供后续与目标周期残差 tp 进行 Pearson 相关和峰值计算。
    tp = target_periodic[:n]
    # tp 是目标侧的周期残差，仅取前 n 个元素，与 sp 维度一致，供后续与仿真侧周期残差 sp 进行 Pearson 相关和峰值计算。
    ff = fooof_freqs[:n]
    # ff 是与 sp 和 tp 对齐的频率数组，仅取前 n 个元素，供后续在特定频段内计算峰值。

    if np.std(sp) < 1e-10:
        return -1.0, -1.0, -1.0

    # fitness分量 1：EEG 与仿真率 PSD 的「去 1/f 周期残差」逐频 Pearson 相关（形状）
    shape_r, _ = pearsonr(sp, tp)

    if np.isnan(shape_r):
        return -1.0, -1.0, -1.0

    # 分量 2/3 保留占位返回，避免改动下游记录字段
    so_power = 0.0
    spindle_power = 0.0

    return float(shape_r), float(so_power), float(spindle_power)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 5：差分进化（scipy.optimize.differential_evolution）
#
#   为何不用 neurolib.Evolution：依赖 pypet，与 NumPy 2.x 不兼容（同 s4_personalize.py）。
#   目标：最大化 score；DE 默认最小化，故 _objective 返回 -score。
#   搜索空间：见 BOUNDS（8 维）；种群规模见 Config 中 DE_POPSIZE、N_GEN。
#   scipy 约定：总种群 = popsize × len(x) = DE_POPSIZE × 8（每代个体数）。
#   workers=1：numba JIT 在多进程下不可靠，与 s4 一致。
#   记录：all_records 存每次评估的参数与 shape_r/so/spindle/score；_callback 每代打印最优与耗时。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BOUNDS = [
    (2.5,  4.5),      # mue  皮层兴奋背景输入
    (2.5,  5.0),      # mui  皮层抑制背景输入
    (0.0,  50.0),     # b    适应电流幅度（论文式宽范围）
    (500.0, 5000.0),  # tauA 适应时间常数 [ms]
    (0.02, 0.20),     # g_LK TCR K 漏电导
    (0.02, 0.20),     # g_h  TCR h 电流电导
    (0.001, 0.30),    # c_th2ctx 丘脑→皮层（connectivity[0,1]）
    (0.001, 0.30),    # c_ctx2th 皮层→丘脑（connectivity[1,0]）
]
PARAM_NAMES = ["mue", "mui", "b", "tauA", "g_LK", "g_h", "c_th2ctx", "c_ctx2th"]

all_records = []   # 每次 _objective 评估追加一行（含当前 _gen，用于粗分代）
_gen = [0]         # 由 _callback 递增，写入记录时作「代」标签（近似）
_evals = [0]       # 成功调用 compute_fitness 的次数
_t0 = [time.time()]


def _objective(x):
    """
    解包 8 维向量 → Part 4 compute_fitness → 仅使用 shape_r 作为 score。
    """
    mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th = x
    shape_r, so_pwr, sp_pwr = compute_fitness(
        mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th
    )
    _evals[0] += 1

    score = shape_r
    # score 仅由形状相关性决定（分量 1）。

    all_records.append({
        "mue": mue, "mui": mui, "b": b, "tauA": tauA,
        "g_LK": g_lk, "g_h": g_h,
        "c_th2ctx": c_th2ctx, "c_ctx2th": c_ctx2th,
        "shape_r": shape_r, "so_power": so_pwr, "spindle_power": sp_pwr,
        "score": score, "gen": _gen[0],
    })
    return -score


def _callback(xk, convergence):
    """每代结束调用：递增代计数，打印当前全局最优 score 与累计评估次数。"""
    _gen[0] += 1
    df = pd.DataFrame(all_records)
    best = df["score"].max()
    elapsed = time.time() - _t0[0]
    print(f"  Gen {_gen[0]:>3}/{N_GEN}  best={best:+.4f}  "
          f"evals={_evals[0]}  elapsed={elapsed:.0f}s")


print(f"\n{'='*60}")
print(f"Fig. 7 Evolution: 8 params, FOOOF={'ON' if HAS_FOOOF else 'OFF'}")
print(f"  popsize×n_params = {DE_POPSIZE}×8 = {DE_POPSIZE*8} individuals/gen")
print(f"  maxiter={N_GEN}, total evals ~ {DE_POPSIZE*8*(N_GEN+1)}")
print(f"{'='*60}\n")

_t0[0] = time.time()
result = differential_evolution(
    _objective,
    bounds=BOUNDS,
    strategy="best1bin",       # 经典 DE/rand/1/bin 变体之一
    maxiter=N_GEN,
    popsize=DE_POPSIZE,        # 乘以维数 8 为每代个体总数
    tol=1e-4,                  # 收敛容差（早停条件之一）
    mutation=(0.5, 1.0),       # 差分缩放因子范围
    recombination=0.7,         # 交叉概率
    seed=42,
    callback=_callback,
    polish=False,              # 不在最后用局部搜索抛光（省时间）
    init="latinhypercube",     # 初始种群空间填充
    workers=1,                 # 单进程，避免 numba 与多进程冲突
    updating="immediate",      # 逐个体更新最优（与 workers=1 搭配）
)

print(f"\nEvolution complete  (total evals = {_evals[0]})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 6：从进化记录中取出最优个体并落盘
#
#   准则：在 all_records 中按 **score** 最大选一行（加权适应度，见 Part 5）。
#   产出：
#     data/patient_params_fig7_<SUBJECT>.json — 最优 8 维参数 + score/shape_r/so/spindle
#               + subject_id、N3 epoch 数、是否使用 FOOOF（供复现与下游脚本读取）；
#     outputs/evolution_fig7_records.csv — 全部评估历史，便于画参数云图或复查。
#   注意：最优行对应某次评估时的三分量；若需与「再跑仿真」严格一致，需固定随机种子与实现版本。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

df = pd.DataFrame(all_records)
best_row = df.loc[df["score"].idxmax()]   # score 最大者；若有并列，idxmax 取首个

best_params = {name: float(best_row[name]) for name in PARAM_NAMES}
best_params["score"] = float(best_row["score"])
best_params["shape_r"] = float(best_row["shape_r"])
best_params["so_power"] = float(best_row["so_power"])
best_params["spindle_power"] = float(best_row["spindle_power"])
best_params["subject_id"] = SUBJECT_ID
best_params["n3_epochs"] = n_passed       # Part 2 通过 QC 的 N3 epoch 数
best_params["fooof"] = HAS_FOOOF          # 适应度是否走 FOOOF 分支

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

with open(f"data/patient_params_fig7_v1_0418_2_{SUBJECT_ID}.json", "w") as fh:
    json.dump(best_params, fh, indent=2)
df.to_csv("outputs/evolution_fig7_records.csv", index=False)

print(f"\nBest parameters (score={best_params['score']:.4f}):")
for k in PARAM_NAMES:
    print(f"  {k}: {best_params[k]:.4f}")
print(f"  shape_r: {best_params['shape_r']:.4f}") # 目标 EEG 与仿真周期残差的 Pearson r（形状相关性）
print(f"  so_power: {best_params['so_power']:.4f}") # 仿真周期残差在 0.2–1.5 Hz 频段的峰值（慢振荡功率）
print(f"  spindle_power: {best_params['spindle_power']:.4f}") # 仿真周期残差在 10–14 Hz 频段的峰值（纺锤体功率）


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 7：用最优参数再跑一段**更长**仿真，专供 Part 8 作图（非适应度路径）
#
#   进化阶段每次评估仅用 SIM_DUR_MS=30 s（Part 4）；此处改为 **60 s**，时间序列与 Welch
#   更平滑，便于 Fig.7(c)(d) 展示。
#   输出：皮层/丘脑 r_E（Hz）；t_s 统一为秒；f_sim/p_sim 为皮层 PSD；可选 FOOOF 得 1/f 线供 (d) 叠画。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\nRunning 60s simulation with best parameters...")
bm = build_model(
    best_params["mue"],
    best_params["mui"],
    best_params["b"],
    best_params["tauA"],
    best_params["g_LK"],
    best_params["g_h"],
    best_params["c_th2ctx"],
    best_params["c_ctx2th"],
)
bm.params["duration"] = 60_000   # 覆盖 build_model 内默认 30 s
bm.run()

r_exc_best = bm[f"r_mean_{EXC}"]
if r_exc_best.ndim == 2 and r_exc_best.shape[0] >= 2:
    rE_cortex   = r_exc_best[0, :] * 1000
    rE_thalamus = r_exc_best[1, :] * 1000
else:
    rE_cortex   = (r_exc_best[0] if r_exc_best.ndim == 2 else r_exc_best) * 1000
    rE_thalamus = np.zeros_like(rE_cortex)

t_s = bm["t"]
if t_s[-1] > 1000:   # MultiModel 可能给 ms，转成秒
    t_s = t_s / 1000.0

# 以下 Welch 仅针对皮层 r_E，供 (d) 下半「仿真谱」与 FOOOF 虚线
n_drop = int(5.0 * FS_SIM)
r_clean = rE_cortex[n_drop:]
nperseg = min(int(10.0 * FS_SIM), len(r_clean))
f_sim, p_sim = welch(r_clean, fs=FS_SIM, nperseg=nperseg,
                     noverlap=nperseg//2, window="hann")
mask_f = (f_sim >= F_LO) & (f_sim <= F_HI)
f_sim, p_sim = f_sim[mask_f], p_sim[mask_f]

sim_periodic_best = None
sim_aperiodic_best = None
if HAS_FOOOF:
    try:
        fm_best = FOOOF(peak_width_limits=[1.0, 8.0], max_n_peaks=4,
                        min_peak_height=0.05, aperiodic_mode="fixed")
        fm_best.fit(f_sim, p_sim, [F_LO, F_HI])
        sim_log_best = np.log10(p_sim[:len(fm_best._ap_fit)] + 1e-30)
        sim_periodic_best = sim_log_best - fm_best._ap_fit
        sim_aperiodic_best = fm_best._ap_fit
    except Exception:
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 8：Fig.7 四宫格 — outputs/fig7_personalized.png
#
#   (a) 示意图：皮层/丘脑方块 + 最优 c_th2ctx、c_ctx2th 标注（非数据驱动绘图）。
#   (b) 进化散点：mue–mui 平面，颜色=score，红星=best_params。
#   (c) 时间序列：16–24 s 窗口，皮层 r_E 与丘脑 TCR（与论文展示窗口类似，可改 t_start/t_end）。
#   (d) 谱对比：上=Part 2 目标 EEG PSD + FOOOF 1/f；下=Part 7 皮层仿真 PSD + 1/f；单位 V²/Hz vs Hz²/Hz。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig = plt.figure(figsize=(16, 12))
gs_top = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35,
                           height_ratios=[1, 1])

# (a) 网络示意图
ax_a = fig.add_subplot(gs_top[0, 0])
ax_a.set_xlim(0, 10); ax_a.set_ylim(0, 10)
ax_a.set_aspect('equal'); ax_a.axis('off')

# Cortex
ctx_box = plt.Rectangle((1, 5.5), 8, 3.5, facecolor='#E6F1FB',
                         edgecolor='#185FA5', lw=2)
ax_a.add_patch(ctx_box)
ax_a.text(5, 8.5, 'Cortex (ALNNode)', ha='center', fontsize=11,
          fontweight='bold', color='#0C447C')
for cx, cy, lbl, col in [(3, 6.8, 'EXC', '#d62728'), (7, 6.8, 'INH', '#1f77b4')]:
    ax_a.add_patch(plt.Circle((cx, cy), 0.8, color=col, alpha=0.7))
    ax_a.text(cx, cy, lbl, ha='center', va='center', fontsize=9,
              fontweight='bold', color='white')
ax_a.annotate('', xy=(6.2, 7.0), xytext=(3.8, 7.0),
              arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5))
ax_a.annotate('', xy=(3.8, 6.6), xytext=(6.2, 6.6),
              arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5))

# Thalamus
th_box = plt.Rectangle((1, 0.5), 8, 3.5, facecolor='#FAEEDA',
                        edgecolor='#854F0B', lw=2)
ax_a.add_patch(th_box)
ax_a.text(5, 3.5, 'Thalamus (ThalamicNode)', ha='center', fontsize=11,
          fontweight='bold', color='#633806')
for cx, cy, lbl, col in [(3, 1.8, 'TCR', '#d62728'), (7, 1.8, 'TRN', '#1f77b4')]:
    ax_a.add_patch(plt.Circle((cx, cy), 0.8, color=col, alpha=0.7))
    ax_a.text(cx, cy, lbl, ha='center', va='center', fontsize=9,
              fontweight='bold', color='white')

# Inter-node arrows with best-fit coupling strengths
c_th2 = best_params["c_th2ctx"]
c_ct2 = best_params["c_ctx2th"]
ax_a.annotate('', xy=(2.5, 5.5), xytext=(2.5, 4.0),
              arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax_a.text(1.0, 4.7, f'{c_th2:.3f}', fontsize=9, color='black')
ax_a.annotate('', xy=(7.5, 4.0), xytext=(7.5, 5.5),
              arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))
ax_a.text(8.2, 4.7, f'{c_ct2:.3f}', fontsize=9, color='gray')
ax_a.text(0.3, 9.5, '(a)', fontsize=16, fontweight='bold')


# (b) 皮层参数平面 mue–mui；灰点=低分，黄绿=高分分位
ax_b = fig.add_subplot(gs_top[0, 1])

valid_df = df[df["score"] > 0].copy()
threshold = valid_df["score"].quantile(0.7) if len(valid_df) > 10 else 0.3
good = valid_df[valid_df["score"] >= threshold]
bad  = valid_df[valid_df["score"] < threshold]

ax_b.scatter(bad["mue"], bad["mui"], c='gray', alpha=0.3, s=15, label='Below threshold')
sc = ax_b.scatter(good["mue"], good["mui"], c=good["score"],
                  cmap='YlGn', s=30, vmin=threshold, vmax=valid_df["score"].max(),
                  edgecolors='white', linewidths=0.5, label='Good fits')
ax_b.plot(best_params["mue"], best_params["mui"], '*', color='red',
          ms=15, markeredgecolor='black', markeredgewidth=1.5,
          zorder=10, label=f'Best (score={best_params["score"]:.3f})')
plt.colorbar(sc, ax=ax_b, shrink=0.8, label='Score')
ax_b.set_xlabel('Input to EXC ($\\mu_E$) [mV/ms]')
ax_b.set_ylabel('Input to INH ($\\mu_I$) [mV/ms]')
ax_b.set_title('(b) Cortex parameter space', fontweight='bold')
ax_b.legend(fontsize=7, loc='upper left')


# (c) 最优参数下 60 s 轨迹中的一段（默认 16–24 s）
gs_c = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_top[1, 0],
                                        hspace=0.4)
t_start, t_end = 16.0, 24.0
mask_t = (t_s >= t_start) & (t_s <= t_end)

ax_c1 = fig.add_subplot(gs_c[0])
ax_c1.plot(t_s[mask_t], rE_cortex[mask_t], '#534AB7', lw=0.6)
ax_c1.set_ylabel('$r_E$ [Hz]')
ax_c1.set_ylim(bottom=-1)
ax_c1.set_title('Cortex EXC', fontsize=10)
ax_c1.text(0.02, 0.88, '(c)', transform=ax_c1.transAxes,
           fontsize=16, fontweight='bold')

ax_c2 = fig.add_subplot(gs_c[1])
ax_c2.plot(t_s[mask_t], rE_thalamus[mask_t], '#1D9E75', lw=0.6)
ax_c2.set_xlabel('Time [s]')
ax_c2.set_ylabel('$r_{TCR}$ [Hz]')
ax_c2.set_ylim(bottom=-1)
ax_c2.set_title('Thalamus TCR', fontsize=10)


# (d) 上：观测 EEG；下：仿真皮层 PSD（与 Part 4 评价对象一致：皮层 r_E）
gs_d = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_top[1, 1],
                                        hspace=0.4)

ax_d1 = fig.add_subplot(gs_d[0])
ax_d1.semilogy(target_freqs, target_psd, 'k', lw=1.5, label=f'Target {SUBJECT_ID} N3')
if target_aperiodic is not None:
    ax_d1.semilogy(fooof_freqs, 10**target_aperiodic, 'b--', lw=1, label='1/f fit')
ax_d1.axvspan(0.2, 1.5, alpha=0.08, color='orange')
ax_d1.axvspan(10, 14, alpha=0.08, color='green')
ax_d1.set_ylabel('Power [V²/Hz]')
ax_d1.set_xlim(F_LO, F_HI)
ax_d1.legend(fontsize=7)
ax_d1.set_title('EEG', fontsize=10, loc='right')
ax_d1.text(0.02, 0.88, '(d)', transform=ax_d1.transAxes,
           fontsize=16, fontweight='bold')

ax_d2 = fig.add_subplot(gs_d[1])
ax_d2.semilogy(f_sim, p_sim, '#534AB7', lw=1.5, label='Cortex EXC (sim)')
if sim_aperiodic_best is not None:
    ax_d2.semilogy(fm_best.freqs, 10**sim_aperiodic_best, 'b--', lw=1, label='1/f fit')
ax_d2.axvspan(0.2, 1.5, alpha=0.08, color='orange', label='SO (0.2-1.5 Hz)')
ax_d2.axvspan(10, 14, alpha=0.08, color='green', label='Spindle (10-14 Hz)')
ax_d2.set_xlabel('Frequency [Hz]')
ax_d2.set_ylabel('Power [Hz²/Hz]')
ax_d2.set_xlim(F_LO, F_HI)
ax_d2.legend(fontsize=7)
ax_d2.set_title('Simulated firing rate', fontsize=10, loc='right')

# 角标：进化记录中的 shape_r（与 60 s 重跑谱非同一 Monte Carlo，仅作参考）
corr_str = f"shape_r = {best_params['shape_r']:.3f}"
if HAS_FOOOF:
    corr_str += " (1/f-subtracted)"
ax_d2.text(0.98, 0.05, corr_str, transform=ax_d2.transAxes,
           ha='right', fontsize=8, color='gray')

fig.suptitle(
    f"Fig. 7 reproduction — {SUBJECT_ID} "
    f"(score={best_params['score']:.3f}, "
    f"shape_r={best_params['shape_r']:.3f}, "
    f"N3 epochs={n_passed})\n"
    f"8 free params, FOOOF={'ON' if HAS_FOOOF else 'OFF'}, "
    f"{N_GEN} gen x {DE_POPSIZE*8} pop",
    fontsize=12, fontweight='bold',
)
plt.savefig("outputs/fig7_personalized_0417_v3.png", dpi=150, bbox_inches="tight")
print("\n✓ Saved: outputs/fig7_personalized_0417_v3.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 9：终端校验 — 根据最优个体的 shape_r 与 SO/纺锤分量打印通过/提示
#
#   阈值：>0.7 视为接近论文示例；0.5–0.7 可接受；否则建议增大 N_GEN 或 DE_POPSIZE。
#   最后列出 Part 6–8 主要输出路径，便于复制进实验记录。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print(f"\n{'='*50}")
print("Validation")
print(f"{'='*50}")
r = best_params["shape_r"]
if r > 0.7:
    print(f"✓ Shape r = {r:.3f} > 0.7  (matches paper threshold)")
elif r > 0.5:
    print(f"~ Shape r = {r:.3f} > 0.5  (acceptable)")
else:
    print(f"✗ Shape r = {r:.3f}  (poor — increase N_GEN or DE_POPSIZE)")

print(f"  SO power     : {best_params['so_power']:.3f}")
print(f"  Spindle power: {best_params['spindle_power']:.3f}")
print(f"\nSaved: data/patient_params_fig7_{SUBJECT_ID}_0417_v3.json")
print(f"Saved: outputs/evolution_fig7_records_0417_v3.csv")
print(f"Saved: outputs/fig7_personalized_0417_v3.png")
sys.exit(0)