"""
plot_fig7_v2_fast.py
====================
快速版：只生成 Fig. 7 (c)(d) + residual对比图，跳过耗时的 7(b) bifurcation scan。

输出（共3张图）：
  outputs/fig7_v2_timeseries.png     ← 7(c): 皮层 + 丘脑时间序列
  outputs/fig7_v2_spectra.png        ← 7(d): EEG vs 模拟 PSD（semilogy）
  outputs/fig7_v2_residuals.png      ← FOOOF 1/f去除后的residual对比

运行方式（从项目根目录）：
  python plot_scripts/plot_fig7_v2_fast.py

依赖：neurolib, mne, fooof, scipy, pandas, numpy, matplotlib
"""

import os
import sys
import json
import fnmatch

# 确保工作目录是项目根
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if os.getcwd() != _ROOT:
    os.chdir(_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import mne
mne.set_log_level("WARNING")

from neurolib.models.multimodel import MultiModel, ALNNode, ThalamicNode
from neurolib.models.multimodel.builder.base.network import Network
from neurolib.models.multimodel.builder.base.constants import EXC, INH

try:
    from fooof import FOOOF
    HAS_FOOOF = True
except ImportError:
    HAS_FOOOF = False
    print("[warn] fooof未安装，residual图将跳过")

# ─── 配置（与 s4_personalize_fig7_v2.py 保持一致）────────────────────────────
SUBJECT_ID      = "SC4001"
EEG_CHANNELS    = ["EEG Fpz-Cz", "EEG Pz-Oz"]
N3_LABELS       = ["Sleep stage 3", "Sleep stage 4"]
ARTIFACT_THRESH = 200e-6
EPOCH_DURATION  = 30.0
F_LO, F_HI     = 0.5, 20.0
FS_SIM          = 1000.0

# V2 文件路径
PARAMS_PATH  = f"data/patient_params_fig7_v2_{SUBJECT_ID}.json"

# 模拟时长：60秒，足够看清 UP/DOWN 结构，比 7(b) 的 scan 快得多
SIM_DUR_MS   = 60_000

# 时间序列展示窗口（秒）
T_SHOW_START = 16.0
T_SHOW_END   = 32.0   # 比论文长一点，能看到更多 SO 周期


# ─── 工具函数 ────────────────────────────────────────────────────────────────

def set_params_glob(model, pattern, value):
    for k in (k for k in model.params if fnmatch.fnmatch(k, pattern)):
        model.params[k] = value


class ThalamoCorticalNetwork(Network):
    name            = "Thalamocortical Motif"
    label           = "TCNet"
    sync_variables  = ["network_exc_exc", "network_exc_exc_sq", "network_inh_exc"]
    default_output  = f"r_mean_{EXC}"
    output_vars     = [f"r_mean_{EXC}", f"r_mean_{INH}"]
    _EXC_WITHIN_IDX = [6, 9]

    def __init__(self, c_th2ctx=0.02, c_ctx2th=0.15):
        aln = ALNNode(exc_seed=42, inh_seed=42)
        th  = ThalamicNode()
        aln.index = 0;  aln.idx_state_var = 0
        th.index  = 1;  th.idx_state_var  = aln.num_state_variables
        for i, node in enumerate([aln, th]):
            for mass in node:
                mass.noise_input_idx = [2 * i + mass.index]
        connectivity = np.array([[0.0, c_th2ctx], [c_ctx2th, 0.0]])
        super().__init__(
            nodes=[aln, th],
            connectivity_matrix=connectivity,
            delay_matrix=np.zeros_like(connectivity),
        )

    def _sync(self):
        couplings = sum([node._sync() for node in self], [])
        wi = self._EXC_WITHIN_IDX
        couplings += self._additive_coupling(wi, "network_exc_exc")
        couplings += self._additive_coupling(wi, "network_exc_exc_sq",
                                             connectivity=self.connectivity ** 2)
        couplings += self._additive_coupling(wi, "network_inh_exc")
        return couplings


def build_model(bp, duration=SIM_DUR_MS):
    net = ThalamoCorticalNetwork(
        c_th2ctx=bp["c_th2ctx"],
        c_ctx2th=bp["c_ctx2th"],
    )
    m = MultiModel(net)
    m.params["backend"]     = "numba"
    m.params["dt"]          = 0.1
    m.params["sampling_dt"] = 1.0
    m.params["duration"]    = duration
    set_params_glob(m, "*ALNMassEXC*.input_0.mu", bp["mue"])
    set_params_glob(m, "*ALNMassINH*.input_0.mu", bp["mui"])
    set_params_glob(m, "*ALNMassEXC*.b",          bp["b"])
    set_params_glob(m, "*ALNMassEXC*.tauA",       bp["tauA"])
    set_params_glob(m, "*ALNMassEXC*.a",          0.0)
    set_params_glob(m, "*ALNMass*.input_0.sigma",  0.05)
    set_params_glob(m, "*TCR*.input_0.sigma",      0.005)
    set_params_glob(m, "*.input_0.tau",            5.0)
    set_params_glob(m, "*TRN*.g_LK",              0.1)
    set_params_glob(m, "*TCR*.g_LK",              bp["g_LK"])
    set_params_glob(m, "*TCR*.g_h",               bp["g_h"])
    return m


def load_target_psd():
    """从 Sleep-EDF 加载 SC4001 的 N3 EEG，计算平均 PSD。"""
    manifest = pd.read_csv("data/manifest.csv")
    subj_row = manifest[manifest["subject_id"] == SUBJECT_ID].iloc[0]
    raw = mne.io.read_raw_edf(
        subj_row["psg_path"], include=EEG_CHANNELS, preload=True, verbose=False
    )
    fs_eeg = raw.info["sfreq"]
    raw.set_annotations(mne.read_annotations(subj_row["hypnogram_path"]))
    event_id = {lbl: idx + 1 for idx, lbl in enumerate(N3_LABELS)}
    events, event_dict = mne.events_from_annotations(
        raw, event_id=event_id, verbose=False
    )
    epochs_n3 = mne.Epochs(
        raw, events, event_id=event_dict,
        tmin=0.0, tmax=EPOCH_DURATION,
        baseline=None, preload=True, verbose=False,
    )
    psds, f_ep, freq_mask = [], None, None
    for ep_idx in range(len(epochs_n3)):
        data = epochs_n3[ep_idx].get_data()[0]
        if np.any((data.max(axis=1) - data.min(axis=1)) > ARTIFACT_THRESH):
            continue
        mean_sig = data.mean(axis=0)
        nperseg  = min(int(10.0 * fs_eeg), len(mean_sig))
        f_ep, p_ep = welch(mean_sig, fs=fs_eeg, nperseg=nperseg,
                           noverlap=nperseg // 2, window="hann")
        freq_mask = (f_ep >= F_LO) & (f_ep <= F_HI)
        psds.append(p_ep[freq_mask])

    if not psds:
        raise RuntimeError("没有通过质控的 N3 epoch")

    print(f"  EEG: {len(psds)} 个 epoch 通过质控")
    return np.mean(psds, axis=0), f_ep[freq_mask]


# ─── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    # 1. 加载 V2 最优参数
    if not os.path.isfile(PARAMS_PATH):
        print(f"[error] 找不到 {PARAMS_PATH}，请先运行 s4_personalize_fig7_v2.py")
        sys.exit(1)

    with open(PARAMS_PATH) as fh:
        bp = json.load(fh)

    print("=" * 55)
    print(f"V2 最优参数 (score={bp.get('score', 'N/A')})")
    print("=" * 55)
    for k, v in bp.items():
        print(f"  {k}: {v}")

    # 2. 跑模拟
    print(f"\n运行模拟（{SIM_DUR_MS/1000:.0f}s）...")
    model = build_model(bp, duration=SIM_DUR_MS)
    try:
        model.run()
    except Exception as e:
        print(f"  numba后端失败({e})，切换到 jitcdde")
        model.params["backend"] = "jitcdde"
        model.run()

    # 3. 提取时间序列
    n_total = int(SIM_DUR_MS / model.params["sampling_dt"])
    t_s = np.linspace(0, SIM_DUR_MS / 1000, n_total)

    r_exc_raw = model[f"r_mean_{EXC}"]
    rE_cortex = (r_exc_raw[0] if r_exc_raw.ndim == 2 else r_exc_raw) * 1000

    # 提取丘脑 TCR firing rate
    # ThalamicNode 的输出变量
    try:
        thal_key = [k for k in model.outputs.keys()
                    if "Thalamic" in k and EXC in k][0]
        r_thal_raw = model.outputs[thal_key]
        rE_thalamus = (r_thal_raw[0] if r_thal_raw.ndim == 2 else r_thal_raw) * 1000
    except (IndexError, KeyError):
        # 备用：尝试直接访问
        try:
            rE_thalamus = model.outputs[f"r_mean_{EXC}"][1] * 1000
        except Exception:
            print("  [warn] 无法提取丘脑时间序列，将用零填充")
            rE_thalamus = np.zeros_like(rE_cortex)

    # 截断到同样长度
    n_min = min(len(t_s), len(rE_cortex), len(rE_thalamus))
    t_s        = t_s[:n_min]
    rE_cortex  = rE_cortex[:n_min]
    rE_thalamus = rE_thalamus[:n_min]

    # burn-in 去掉前 5 秒
    n_burn = int(5.0 * FS_SIM)
    r_ctx_full = rE_cortex[n_burn:]

    # 4. 计算模拟 PSD（皮层）
    nperseg = min(int(10.0 * FS_SIM), len(r_ctx_full))
    f_sim, p_sim = welch(r_ctx_full, fs=FS_SIM, nperseg=nperseg,
                         noverlap=nperseg // 2, window="hann")
    mask_sim = (f_sim >= F_LO) & (f_sim <= F_HI)
    f_sim, p_sim = f_sim[mask_sim], p_sim[mask_sim]

    # 5. 加载目标 EEG PSD
    print("\n加载目标 EEG N3 PSD...")
    target_psd, target_freqs = load_target_psd()

    # 6. FOOOF residuals（如果可用）
    target_periodic = sim_periodic = fooof_freqs = None
    sim_aperiodic = tgt_aperiodic = None
    tgt_fooof_freqs = sim_fooof_freqs = None
    shape_r_recomputed = None

    if HAS_FOOOF:
        print("计算 FOOOF 1/f residuals...")
        # 目标 EEG
        fm_tgt = FOOOF(peak_width_limits=[1.0, 8.0], max_n_peaks=4,
                       min_peak_height=0.05, aperiodic_mode="fixed")
        fm_tgt.fit(target_freqs, target_psd, [F_LO, F_HI])
        tgt_fooof_freqs = fm_tgt.freqs
        tgt_aperiodic   = fm_tgt._ap_fit
        tgt_log         = np.log10(target_psd + 1e-30)
        target_periodic = tgt_log[:len(tgt_aperiodic)] - tgt_aperiodic

        # 模拟 PSD 插值到 FOOOF freqs
        p_interp = interp1d(f_sim, p_sim, bounds_error=False,
                            fill_value=1e-30)(tgt_fooof_freqs)
        fm_sim = FOOOF(peak_width_limits=[1.0, 8.0], max_n_peaks=4,
                       min_peak_height=0.05, aperiodic_mode="fixed")
        fm_sim.fit(tgt_fooof_freqs, p_interp, [F_LO, F_HI])
        sim_fooof_freqs = fm_sim.freqs
        sim_aperiodic   = fm_sim._ap_fit
        sim_log         = np.log10(p_interp[:len(sim_aperiodic)] + 1e-30)
        sim_periodic    = sim_log - sim_aperiodic
        fooof_freqs     = tgt_fooof_freqs

        n_r = min(len(sim_periodic), len(target_periodic))
        shape_r_recomputed, _ = pearsonr(sim_periodic[:n_r], target_periodic[:n_r])
        print(f"  shape_r (重新计算): {shape_r_recomputed:.4f}")
        print(f"  shape_r (JSON存储): {bp.get('shape_r', 'N/A')}")

    os.makedirs("outputs", exist_ok=True)

    # ────────────────────────────────────────────────────────────────────────
    # 图1: 7(c) 时间序列
    # ────────────────────────────────────────────────────────────────────────
    print("\n绘制 7(c) 时间序列...")
    mask_t = (t_s >= T_SHOW_START) & (t_s <= T_SHOW_END)

    fig_c, (ax_c1, ax_c2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig_c.suptitle(
        f"Fig. 7(c) V2 — 丘脑皮层时间序列\n"
        f"score={bp.get('score','?'):.4f}, shape_r={bp.get('shape_r','?'):.4f}, "
        f"dynamics={bp.get('dynamics_score','?'):.2f}",
        fontsize=12, fontweight="bold"
    )

    # 皮层
    ax_c1.plot(t_s[mask_t], rE_cortex[mask_t], color="#534AB7", lw=0.8)
    ax_c1.axhline(1.0, color="gray", lw=0.5, ls="--", alpha=0.5, label="DOWN阈值 (1 Hz)")
    ax_c1.set_ylabel("$r_E$ [Hz]", fontsize=11)
    ax_c1.set_title("皮层兴奋性 (Cortex EXC) — 慢振荡 (SO)", fontsize=10)
    ax_c1.set_ylim(bottom=-1)
    ax_c1.legend(fontsize=8, loc="upper right")
    # 标注 UP/DOWN 状态
    ax_c1.text(0.02, 0.90, "UP state: 高活动期", transform=ax_c1.transAxes,
               fontsize=8, color="#534AB7", alpha=0.8)
    ax_c1.text(0.02, 0.05, "DOWN state: 接近静默", transform=ax_c1.transAxes,
               fontsize=8, color="gray")

    # 丘脑
    ax_c2.plot(t_s[mask_t], rE_thalamus[mask_t], color="#1D9E75", lw=0.8)
    ax_c2.set_ylabel("$r_{TCR}$ [Hz]", fontsize=11)
    ax_c2.set_xlabel("Time [s]", fontsize=11)
    ax_c2.set_title("丘脑 TCR — 纺锤波 (Spindle)", fontsize=10)
    ax_c2.set_ylim(bottom=-1)

    # 标注 V2 最优参数
    param_txt = (
        f"mue={bp['mue']:.3f}  mui={bp['mui']:.3f}  "
        f"b={bp['b']:.1f}  tauA={bp['tauA']:.0f}\n"
        f"g_LK={bp['g_LK']:.4f}  g_h={bp['g_h']:.4f}  "
        f"c_th2ctx={bp['c_th2ctx']:.4f}  c_ctx2th={bp['c_ctx2th']:.4f}"
    )
    fig_c.text(0.5, 0.01, param_txt, ha="center", fontsize=8,
               color="gray", family="monospace")

    fig_c.tight_layout(rect=[0, 0.05, 1, 1])
    out_c = "outputs/fig7_v2_timeseries.png"
    fig_c.savefig(out_c, dpi=150, bbox_inches="tight")
    print(f"  保存: {out_c}")
    plt.close(fig_c)

    # ────────────────────────────────────────────────────────────────────────
    # 图2: 7(d) Power Spectra（semilogy）
    # ────────────────────────────────────────────────────────────────────────
    print("绘制 7(d) 功率谱...")
    fig_d, (ax_d1, ax_d2) = plt.subplots(2, 1, figsize=(8, 8))
    fig_d.suptitle(
        f"Fig. 7(d) V2 — 功率谱\n{SUBJECT_ID} N3 EEG vs 模拟皮层 firing rate",
        fontsize=12, fontweight="bold"
    )

    # 上图：EEG target
    ax_d1.semilogy(target_freqs, target_psd, "k", lw=1.8,
                   label=f"目标 EEG N3 ({SUBJECT_ID})")
    if tgt_aperiodic is not None:
        ax_d1.semilogy(tgt_fooof_freqs, 10 ** tgt_aperiodic, "b--", lw=1.2,
                       alpha=0.7, label="1/f 拟合")
    ax_d1.axvspan(0.2, 1.5, alpha=0.10, color="orange", label="SO 频段")
    ax_d1.axvspan(10.0, 14.0, alpha=0.10, color="green", label="Spindle 频段")
    ax_d1.set_xlim(F_LO, F_HI)
    ax_d1.set_ylabel("Power [V$^2$/Hz]", fontsize=10)
    ax_d1.set_title("EEG (ground truth)", fontsize=10, loc="right")
    ax_d1.legend(fontsize=8)

    # 下图：模拟 PSD
    ax_d2.semilogy(f_sim, p_sim, color="#534AB7", lw=1.8,
                   label="模拟皮层 EXC firing rate PSD")
    if sim_aperiodic is not None:
        ax_d2.semilogy(sim_fooof_freqs, 10 ** sim_aperiodic, "b--", lw=1.2,
                       alpha=0.7, label="1/f 拟合")
    ax_d2.axvspan(0.2, 1.5, alpha=0.10, color="orange", label="SO 频段")
    ax_d2.axvspan(10.0, 14.0, alpha=0.10, color="green", label="Spindle 频段")
    ax_d2.set_xlim(F_LO, F_HI)
    ax_d2.set_xlabel("Frequency [Hz]", fontsize=10)
    ax_d2.set_ylabel("Power [Hz$^2$/Hz]", fontsize=10)
    ax_d2.set_title("模拟 (V2)", fontsize=10, loc="right")
    ax_d2.legend(fontsize=8)

    score_txt = (f"shape_r={bp.get('shape_r','?'):.3f}  "
                 f"so_power={bp.get('so_power','?'):.3f}  "
                 f"spindle_power={bp.get('spindle_power','?'):.3f}")
    ax_d2.text(0.98, 0.04, score_txt, transform=ax_d2.transAxes,
               ha="right", fontsize=8, color="gray")

    fig_d.tight_layout()
    out_d = "outputs/fig7_v2_spectra.png"
    fig_d.savefig(out_d, dpi=150, bbox_inches="tight")
    print(f"  保存: {out_d}")
    plt.close(fig_d)

    # ────────────────────────────────────────────────────────────────────────
    # 图3: FOOOF Residual 对比
    # ────────────────────────────────────────────────────────────────────────
    if HAS_FOOOF and target_periodic is not None:
        print("绘制 FOOOF residual 对比...")
        n_r = min(len(sim_periodic), len(target_periodic))
        ff  = fooof_freqs[:n_r]
        tp  = target_periodic[:n_r]
        sp  = sim_periodic[:n_r]

        fig_r, ax_r = plt.subplots(figsize=(10, 5))
        ax_r.plot(ff, tp, "k-", lw=2.0,
                  label=f"EEG target N3 (1/f去除后)")
        ax_r.plot(ff, sp, color="#534AB7", lw=2.0, ls="--",
                  label=f"模拟皮层 r_E PSD (1/f去除后)")
        ax_r.axhline(0.0, color="gray", lw=0.5, alpha=0.5)
        ax_r.axvspan(0.2, 1.5, alpha=0.10, color="orange", label="SO 频段")
        ax_r.axvspan(10.0, 14.0, alpha=0.10, color="green", label="Spindle 频段")

        # 标注峰位置
        for freq_label, freq_val, col in [
            ("δ 2.4Hz", 2.4, "orange"),
            ("θ 6.2Hz", 6.2, "purple"),
            ("α 9.9Hz", 9.9, "blue"),
            ("σ 12.5Hz", 12.5, "green"),
        ]:
            ax_r.axvline(freq_val, color=col, lw=0.8, ls=":", alpha=0.6)
            ax_r.text(freq_val + 0.1, ax_r.get_ylim()[1] * 0.95 if ax_r.get_ylim()[1] > 0 else 0.3,
                      freq_label, fontsize=7, color=col, rotation=90, va="top")

        r_stored = bp.get("shape_r", "?")
        title_str = (
            f"FOOOF 1/f去除后 Residual 对比 | "
            f"Pearson r (重计算) = {shape_r_recomputed:.4f}  "
            f"(进化存储 = {r_stored:.4f})"
        ) if isinstance(r_stored, float) else (
            f"FOOOF 1/f去除后 Residual 对比 | "
            f"Pearson r = {shape_r_recomputed:.4f}"
        )
        ax_r.set_title(title_str, fontsize=11)
        ax_r.set_xlabel("Frequency [Hz]", fontsize=11)
        ax_r.set_ylabel("Log-domain residual (periodic component)", fontsize=10)
        ax_r.set_xlim(F_LO, F_HI)
        ax_r.legend(loc="upper right", fontsize=9)

        fig_r.tight_layout()
        out_r = "outputs/fig7_v2_residuals.png"
        fig_r.savefig(out_r, dpi=150, bbox_inches="tight")
        print(f"  保存: {out_r}")
        plt.close(fig_r)
    else:
        print("  跳过 residual 图（fooof 未安装或 target_periodic 为空）")

    # ─── 终端摘要 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"完成！V2 快速可视化结果摘要")
    print("=" * 55)
    print(f"  score          = {bp.get('score','N/A')}")
    print(f"  shape_r (JSON) = {bp.get('shape_r','N/A')}")
    if shape_r_recomputed is not None:
        print(f"  shape_r (重计) = {shape_r_recomputed:.4f}")
    print(f"  dynamics_score = {bp.get('dynamics_score','N/A')}")
    print(f"  so_power       = {bp.get('so_power','N/A')}")
    print(f"  spindle_power  = {bp.get('spindle_power','N/A')}")
    print(f"\n  皮层 DOWN state 检查：")
    print(f"    min r_E (全段) = {rE_cortex[n_burn:].min():.3f} Hz")
    print(f"    mean r_E       = {rE_cortex[n_burn:].mean():.3f} Hz")
    print(f"\n输出文件：")
    print(f"  outputs/fig7_v2_timeseries.png")
    print(f"  outputs/fig7_v2_spectra.png")
    print(f"  outputs/fig7_v2_residuals.png")


if __name__ == "__main__":
    main()