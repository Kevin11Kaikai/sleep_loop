"""
tests/test_compute_fitness_v3.py
================================
Hands-on check of compute_fitness_v3 (see docs/s4_personalize_fig7_v3_compute_fitness_v3.md).

Loads the same target EEG + FOOOF setup as evolution, runs ONE fitness evaluation with
parameters from data/patient_params_fig7_v3_SC4001.json (or built-in defaults), prints
fitness components and T1–T5 diagnostics.

Optional figures (same roles as plot_scripts/plot_fig7_v2_fast.py):
  - outputs/fig7_v3_test_timeseries.png  — Fig. 7(c)-style cortex + thalamus
  - outputs/fig7_v3_test_spectra.png     — Fig. 7(d)-style EEG vs sim PSD + 1/f
  - outputs/fig7_v3_test_residuals.png   — log-domain residuals + Pearson r

Use --out-v2-names to write fig7_v2_timeseries.png / fig7_v2_spectra.png / fig7_v2_residuals.png
(overwrites files from the V2 fast plot script if present).

Welch for spectra/residuals uses the same post-burn *length* as evolution (SIM_DUR_MS − 5 s),
taken from the start of a longer plot simulation so Fig. 7(c) can show 16–32 s.

Requirements:
  - Run from project root:  python tests/test_compute_fitness_v3.py
  - data/manifest.csv + Sleep-EDF paths for SC4001
  - neurolib, mne, scipy, numpy, matplotlib; fooof recommended for residual figure

This mutates module globals _eval_count / _records in s4_personalize_fig7_v3; we reset them
before/after so repeated runs stay readable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

# ── project root ─────────────────────────────────────────────────────────────
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_TESTS_DIR, ".."))
if os.getcwd() != _ROOT:
    os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── load s4_personalize_fig7_v3 as a standalone module (models/ is not a package) ─
_V3_PATH = os.path.join(_ROOT, "models", "s4_personalize_fig7_v3.py")
_spec = importlib.util.spec_from_file_location("s4_personalize_fig7_v3", _V3_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load spec for {_V3_PATH}")
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)

PARAM_NAMES = v3.PARAM_NAMES
DEFAULT_PARAMS_JSON = os.path.join(_ROOT, "data", "patient_params_fig7_v3_SC4001.json")
FALLBACK_PARAMS_JSON = os.path.join(_ROOT, "data", "patient_params_fig7_v2_SC4001.json")


def _params_vec_from_json(path: str) -> list[float]:
    with open(path, "r", encoding="utf-8") as fh:
        bp = json.load(fh)
    return [float(bp[k]) for k in PARAM_NAMES]


def _default_params_mid_bounds() -> list[float]:
    """Midpoint of BOUNDS if no JSON — may give poor fitness but should run."""
    return [0.5 * (lo + hi) for lo, hi in v3.BOUNDS]


def _reset_v3_globals() -> None:
    v3._eval_count = 0
    v3._best_score = float("-inf")
    v3._best_params = {}
    v3._records.clear()


def _fmt_rec_num(rec: dict, key: str, default: str = "?", fmt: str = ".4f") -> str:
    v = rec.get(key)
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return default
        return format(v, fmt)
    return default


def _generate_fig7_plots(
    params_vec: list[float],
    rec: dict,
    target_psd,
    target_freqs,
    target_periodic,
    fooof_freqs,
    *,
    plot_sim_ms: float,
    out_prefix: str,
) -> None:
    """Extra simulation (longer than evolution) + three PNGs; Welch slice matches evolution length."""
    # 中文画图流程总览（与 docs/compute_fitness_v3_notes.md 一致）：
    # 1) 单独再跑一轮仿真（时长 plot_sim_ms，常 ≥ SIM_DUR_MS），得到皮层/丘脑 r(t)，用于 Fig.7(c)。
    # 2) 从皮层序列取 burn-in 后、长度与进化一致的片段 r_welch，Welch 得 f_ctx/p_ctx（与 compute_fitness_v3 可比）。
    # 3) FOOOF：目标 EEG 画 1/f；仿真谱在原生 Welch 网格上画 1/f；另用「插值到 fooof_freqs + 与进化同配方」算 sim_periodic，与 target_periodic 算 Pearson 写入残差图标题。
    # 4) 输出三张 PNG：timeseries / spectra / residuals，文件名由 out_prefix 决定。
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.signal import welch
    from scipy.stats import pearsonr

    from neurolib.models.multimodel.builder.base.constants import EXC

    mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th = params_vec
    os.makedirs("outputs", exist_ok=True)

    print(f"\n[5] Fig. 7 plots — simulation {plot_sim_ms/1000:.0f} s (matplotlib)...")
    # 中文：此处仿真仅服务于画图，与前面 compute_fitness_v3 内的 30 s 仿真独立；时长用 plot_sim_ms（默认常 ≥30 s）以便时间轴能覆盖例如 16–32 s 窗口。
    m = v3.build_model(
        mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th,
        duration=plot_sim_ms,
    )
    try:
        m.run()
    except Exception as e:
        print(f"    numba failed ({e}); trying jitcdde")
        m.params["backend"] = "jitcdde"
        m.run()

    # 中文：按模型的 sampling_dt 生成与仿真长度一致的时间轴（秒），后续与 r 序列对齐截断。
    sampling_dt = float(m.params["sampling_dt"])
    n_total = int(plot_sim_ms / sampling_dt)
    t_s = np.linspace(0.0, plot_sim_ms / 1000.0, n_total)

    # 中文：r_mean_EXC 在 neurolib 中为 kHz；×1000 转为 Hz，与 fitness 中 r_E 单位一致。多节点时 [0] 皮层、[1] 丘脑。
    r_exc = m[f"r_mean_{EXC}"]
    if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
        rE_cortex = r_exc[0, :].astype(float) * 1000.0
        rE_thalamus = r_exc[1, :].astype(float) * 1000.0
    else:
        rE_cortex = (r_exc[0] if r_exc.ndim == 2 else r_exc).astype(float) * 1000.0
        rE_thalamus = np.zeros_like(rE_cortex)

    # 中文：若模型输出点数与 linspace 不完全一致，取最短长度，保证 t 与两条 r 一一对应。
    n_min = min(len(t_s), len(rE_cortex), len(rE_thalamus))
    t_s = t_s[:n_min]
    rE_cortex = rE_cortex[:n_min]
    rE_thalamus = rE_thalamus[:n_min]

    # 中文：与 compute_fitness_v3 一致——前 5 s 为 burn-in；之后只取与「进化仿真」等长的片段做 Welch（30 s 总时长 → 25 s @ FS_SIM）。
    n_burn = int(5.0 * v3.FS_SIM)
    # Same number of post-burn samples as evolution (30 s run → 25 s @ 1 kHz = 25000).
    n_evo_post = int((v3.SIM_DUR_MS / 1000.0 - 5.0) * v3.FS_SIM)
    r_ctx = rE_cortex[n_burn:]
    if len(r_ctx) < n_evo_post:
        print(f"    [warn] post-burn length {len(r_ctx)} < evolution window {n_evo_post}; using all")
        r_welch = r_ctx
    else:
        r_welch = r_ctx[:n_evo_post]

    # 中文：Welch 段长与 v3 中一致（10 s Hann，50% overlap）；再在 [F_LO,F_HI] 上裁频得到 f_ctx、p_ctx。
    nperseg = min(int(10.0 * v3.FS_SIM), len(r_welch))
    f_ctx, p_ctx = welch(
        r_welch,
        fs=v3.FS_SIM,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        window="hann",
    )
    mask = (f_ctx >= v3.F_LO) & (f_ctx <= v3.F_HI)
    f_ctx, p_ctx = f_ctx[mask], p_ctx[mask]

    tgt_ap = sim_ap = None
    tgt_ff = None
    shape_r_recomp = None
    sim_periodic_plot = None
    target_periodic_plot = None
    ff_plot = None

    # 中文：FOOOF 分三路——(A) 目标 EEG 得 tgt_ap/tgt_ff 供谱图上半 1/f 虚线；(B) 仿真 Welch 网格上得 sim_ap 供下半；(C) 将 p_ctx 插值到 fooof_freqs 后按进化配方算 periodic 残差与 Pearson，供残差图与标题。
    if v3.HAS_FOOOF:
        from fooof import FOOOF

        fm_tgt = FOOOF(
            peak_width_limits=[1.0, 8.0],
            max_n_peaks=4,
            min_peak_height=0.05,
            aperiodic_mode="fixed",
        )
        fm_tgt.fit(target_freqs, target_psd, [v3.F_LO, v3.F_HI])
        tgt_ff = fm_tgt.freqs
        tgt_ap = fm_tgt._ap_fit

        # 中文：(B) 在 Welch 原生频率上拟合 1/f，与下半图「原始仿真 PSD」同一网格。
        fm_native = FOOOF(
            peak_width_limits=[1.0, 8.0],
            max_n_peaks=4,
            min_peak_height=0.05,
            aperiodic_mode="fixed",
        )
        fm_native.fit(f_ctx, p_ctx, [v3.F_LO, v3.F_HI])
        sim_ff_native = fm_native.freqs
        sim_ap = fm_native._ap_fit

        # 中文：(C) 把仿真 PSD 插值到与 target_periodic 相同的 fooof_freqs，再 FOOOF 得 log10(P)-aperiodic = 周期性分量，与 target_periodic 对齐长度后算 Pearson → shape_r_recomp（与进化逻辑对齐，用于残差图说明）。
        if target_periodic is not None and fooof_freqs is not None:
            p_interp = interp1d(
                f_ctx, p_ctx, bounds_error=False, fill_value=1e-30,
            )(fooof_freqs)
            fm_evo = FOOOF(
                peak_width_limits=[1.0, 8.0],
                max_n_peaks=4,
                min_peak_height=0.05,
                aperiodic_mode="fixed",
            )
            fm_evo.fit(fooof_freqs, p_interp, [v3.F_LO, v3.F_HI])
            sim_log = np.log10(p_interp[: len(fm_evo._ap_fit)] + 1e-30)
            sim_periodic_evo = sim_log - fm_evo._ap_fit
            n_r = min(len(sim_periodic_evo), len(target_periodic))
            shape_r_recomp, _ = pearsonr(
                sim_periodic_evo[:n_r], target_periodic[:n_r]
            )
            sim_periodic_plot = sim_periodic_evo[:n_r]
            target_periodic_plot = target_periodic[:n_r]
            ff_plot = np.asarray(fooof_freqs)[:n_r]
    else:
        # 中文：无 FOOOF 时谱图下半仍用 Welch 频率；无 1/f 虚线、无残差第三张图。
        sim_ff_native = f_ctx

    # --- 7(c) time series -----------------------------------------------------
    # 中文：在整段仿真时间轴上取固定窗口 [16,32] s 画皮层/丘脑 r(t)；标题旁注来自 rec（与 [4] 同一次 fitness 的记录）。
    t0, t1 = 16.0, 32.0
    mask_t = (t_s >= t0) & (t_s <= t1)
    fig_c, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig_c.suptitle(
        f"Fig. 7(c) V3 test — thalamocortical time series\n"
        f"score={_fmt_rec_num(rec, 'score')}, shape_r={_fmt_rec_num(rec, 'shape_r')}, "
        f"dynamics={_fmt_rec_num(rec, 'dynamics_score', fmt='.2f')}",
        fontsize=12,
        fontweight="bold",
    )
    ax1.plot(t_s[mask_t], rE_cortex[mask_t], color="#534AB7", lw=0.8)
    ax1.axhline(1.0, color="gray", lw=0.5, ls="--", alpha=0.5, label="DOWN threshold (1 Hz)")
    ax1.set_ylabel("$r_E$ [Hz]", fontsize=11)
    ax1.set_title("Cortex EXC — slow oscillation (SO)", fontsize=10)
    ax1.set_ylim(bottom=-0.1)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.text(0.02, 0.90, "UP state: high activity", transform=ax1.transAxes, fontsize=8, color="#534AB7", alpha=0.8)
    ax1.text(0.02, 0.05, "DOWN state: near silence", transform=ax1.transAxes, fontsize=8, color="gray")

    ax2.plot(t_s[mask_t], rE_thalamus[mask_t], color="#1D9E75", lw=0.8)
    ax2.set_ylabel("$r_{TCR}$ [Hz]", fontsize=11)
    ax2.set_xlabel("Time [s]", fontsize=11)
    ax2.set_title("Thalamic TCR — spindle", fontsize=10)
    ax2.set_ylim(4, 8)

    param_txt = (
        f"mue={mue:.3f}  mui={mui:.3f}  b={b:.1f}  tauA={tauA:.0f}\n"
        f"g_LK={g_lk:.4f}  g_h={g_h:.4f}  "
        f"c_th2ctx={c_th2ctx:.4f}  c_ctx2th={c_ctx2th:.4f}"
    )
    fig_c.text(0.5, 0.01, param_txt, ha="center", fontsize=8, color="gray", family="monospace")
    fig_c.tight_layout(rect=[0, 0.05, 1, 1])
    p_ts = f"outputs/{out_prefix}_timeseries.png"
    fig_c.savefig(p_ts, dpi=150, bbox_inches="tight")
    plt.close(fig_c)
    print(f"    Saved: {p_ts}")

    # --- 7(d) spectra ---------------------------------------------------------
    # 中文：上图——目标 EEG 与 FOOOF 1/f（若有）；下图——仿真皮层 PSD 与原生网格 1/f；两侧纵轴单位不同（V²/Hz vs Hz²/Hz）；橙/绿带标 SO 与 spindle 频段。
    fig_d, (ax_d1, ax_d2) = plt.subplots(2, 1, figsize=(8, 8))
    fig_d.suptitle(
        f"Fig. 7(d) V3 test — power spectra\n{v3.SUBJECT_ID} N3 EEG vs simulated cortex "
        f"(Welch window = evolution post-burn length)",
        fontsize=12,
        fontweight="bold",
    )
    ax_d1.semilogy(target_freqs, target_psd, "k", lw=1.8, label=f"Target EEG N3 ({v3.SUBJECT_ID})")
    if tgt_ap is not None and tgt_ff is not None:
        ax_d1.semilogy(tgt_ff, 10 ** tgt_ap, "b--", lw=1.2, alpha=0.7, label="1/f fit")
    ax_d1.axvspan(0.2, 1.5, alpha=0.10, color="orange", label="SO band")
    ax_d1.axvspan(10.0, 14.0, alpha=0.10, color="green", label="Spindle band")
    ax_d1.set_xlim(v3.F_LO, v3.F_HI)
    ax_d1.set_ylabel("Power [V$^2$/Hz]", fontsize=10)
    ax_d1.set_title("EEG (ground truth)", fontsize=10, loc="right")
    ax_d1.legend(fontsize=8)

    ax_d2.semilogy(f_ctx, p_ctx, color="#534AB7", lw=1.5, label="Simulated cortex EXC firing-rate PSD")
    if sim_ap is not None:
        ax_d2.semilogy(sim_ff_native, 10 ** sim_ap, "b--", lw=1.0, alpha=0.7, label="1/f fit")
    ax_d2.axvspan(0.2, 1.5, alpha=0.10, color="orange")
    ax_d2.axvspan(10.0, 14.0, alpha=0.10, color="green")
    ax_d2.set_xlim(v3.F_LO, v3.F_HI)
    ax_d2.set_xlabel("Frequency [Hz]", fontsize=10)
    ax_d2.set_ylabel("Power [Hz$^2$/Hz]", fontsize=10)
    ax_d2.set_title("Simulation (V3 test)", fontsize=10, loc="right")
    ax_d2.legend(fontsize=8)
    score_txt = (
        f"shape_r={_fmt_rec_num(rec, 'shape_r', fmt='.3f')}  "
        f"so_power={_fmt_rec_num(rec, 'so_power', fmt='.3f')}  "
        f"spindle_power={_fmt_rec_num(rec, 'spindle_power', fmt='.3f')}"
    )
    ax_d2.text(0.98, 0.04, score_txt, transform=ax_d2.transAxes, ha="right", fontsize=8, color="gray")
    fig_d.tight_layout()
    p_sp = f"outputs/{out_prefix}_spectra.png"
    fig_d.savefig(p_sp, dpi=150, bbox_inches="tight")
    plt.close(fig_d)
    print(f"    Saved: {p_sp}")

    # --- residuals ------------------------------------------------------------
    # 中文：第三张——在 log 域去掉 1/f 后的「周期分量」曲线：黑=EEG 目标，紫虚线=仿真（与进化网格对齐）；竖虚线标典型 δ/θ/α/σ 频率；标题可展示重算的 Pearson 与 CSV 中 shape_r 对照。
    if (
        v3.HAS_FOOOF
        and target_periodic_plot is not None
        and sim_periodic_plot is not None
        and ff_plot is not None
    ):
        fig_r, ax_r = plt.subplots(figsize=(10, 5))
        ax_r.plot(ff_plot, target_periodic_plot, "k-", lw=2.0, label="EEG target N3 (1/f removed)")
        ax_r.plot(ff_plot, sim_periodic_plot, color="#534AB7", lw=2.0, ls="--",
                  label=r"Simulated cortex $r_E$ PSD (1/f removed)")
        ax_r.axhline(0.0, color="gray", lw=0.5, alpha=0.5)
        ax_r.axvspan(0.2, 1.5, alpha=0.10, color="orange", label="SO band")
        ax_r.axvspan(10.0, 14.0, alpha=0.10, color="green", label="Spindle band")
        for freq_label, freq_val, col in [
            (r"$\delta$ 2.4 Hz", 2.4, "orange"),
            (r"$\theta$ 6.2 Hz", 6.2, "purple"),
            (r"$\alpha$ 9.9 Hz", 9.9, "blue"),
            (r"$\sigma$ 12.5 Hz", 12.5, "green"),
        ]:
            ax_r.axvline(freq_val, color=col, lw=0.8, ls=":", alpha=0.6)
            y1 = ax_r.get_ylim()[1]
            ax_r.text(
                freq_val + 0.1,
                y1 * 0.95 if y1 > 0 else 0.3,
                freq_label,
                fontsize=7,
                color=col,
                rotation=90,
                va="top",
            )

        sr_stored = rec.get("shape_r")
        if shape_r_recomp is not None and isinstance(sr_stored, (int, float)) and not (
            isinstance(sr_stored, float) and math.isnan(sr_stored)
        ):
            title_str = (
                f"FOOOF residuals (1/f removed) | Pearson r (recomputed, evolution pipeline) = "
                f"{shape_r_recomp:.4f}  (CSV shape_r = {float(sr_stored):.4f})"
            )
        elif shape_r_recomp is not None:
            title_str = f"FOOOF residuals (1/f removed) | Pearson r (recomputed) = {shape_r_recomp:.4f}"
        else:
            title_str = "FOOOF residuals (1/f removed)"
        ax_r.set_title(title_str, fontsize=11)
        ax_r.set_xlabel("Frequency [Hz]", fontsize=11)
        ax_r.set_ylabel("Log-domain residual (periodic component)", fontsize=10)
        ax_r.set_xlim(v3.F_LO, v3.F_HI)
        ax_r.legend(loc="upper right", fontsize=9)
        fig_r.tight_layout()
        p_rs = f"outputs/{out_prefix}_residuals.png"
        fig_r.savefig(p_rs, dpi=150, bbox_inches="tight")
        plt.close(fig_r)
        print(f"    Saved: {p_rs}")
        if shape_r_recomp is not None:
            print(f"    Pearson r (recomputed, matches evolution FOOOF order) = {shape_r_recomp:.4f}")
    else:
        print("    Skipped residuals figure (need fooof + target_periodic + fooof_freqs)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Single compute_fitness_v3 check; optional Fig.7-style PNGs.")
    ap.add_argument(
        "--plots",
        action="store_true",
        default=True,
        help="After fitness, run a longer simulation and write outputs/fig7_v3_test_*.png "
        "(or fig7_v2_*.png with --out-v2-names).",
    )
    ap.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="Disable plotting (overrides default plotting-on behaviour).",
    )
    ap.add_argument(
        "--plot-sim-ms",
        type=float,
        default=60_000.0,
        help="Duration (ms) for the plotting-only simulation (default 60000).",
    )
    ap.add_argument(
        "--out-v2-names",
        action="store_true",
        help="Save as outputs/fig7_v2_timeseries.png, fig7_v2_spectra.png, fig7_v2_residuals.png.",
    )
    args = ap.parse_args()

    print("=" * 60)
    print("test_compute_fitness_v3 — single evaluation")
    print(f"  FOOOF available: {v3.HAS_FOOOF}")
    if args.plots:
        print(f"  --plots: will save Fig.7 PNGs (plot sim = {args.plot_sim_ms:.0f} ms)")
    print("=" * 60)

    if not os.path.isfile("data/manifest.csv"):
        print("[error] data/manifest.csv not found. Run from project root with data in place.")
        return 1

    # Target EEG + precomputed target_periodic / fooof_freqs (same as main())
    print("\n[1] load_target_psd() ...")
    target_psd, target_freqs = v3.load_target_psd()
    print(f"    target_psd shape: {target_psd.shape}, target_freqs: {target_freqs[0]:.2f}–{target_freqs[-1]:.2f} Hz")

    print("\n[2] compute_target_periodic() ...")
    target_periodic, fooof_freqs = v3.compute_target_periodic(target_psd, target_freqs)
    if target_periodic is None:
        print("    [warn] target_periodic is None (no FOOOF); shape_r will use chi2 fallback.")
    else:
        print(f"    fooof_freqs len={len(fooof_freqs)}, target_periodic len={len(target_periodic)}")

    # Parameters
    if os.path.isfile(DEFAULT_PARAMS_JSON):
        p_path = DEFAULT_PARAMS_JSON
        print(f"\n[3] Parameters from {p_path}")
        params_vec = _params_vec_from_json(p_path)
    elif os.path.isfile(FALLBACK_PARAMS_JSON):
        p_path = FALLBACK_PARAMS_JSON
        print(f"\n[3] v3 JSON missing; using {p_path} (v2 — still valid 8-vector)")
        params_vec = _params_vec_from_json(p_path)
    else:
        p_path = None
        print("\n[3] No patient_params JSON; using mid(BOUNDS) (exploratory only)")
        params_vec = _default_params_mid_bounds()

    for name, val in zip(PARAM_NAMES, params_vec):
        print(f"    {name:10s} = {val:.6g}")

    _reset_v3_globals()

    print("\n[4] compute_fitness_v3(...) — running simulation (~30 s eval, ~tens of seconds) ...")
    neg_fitness = v3.compute_fitness_v3(
        params_vec,
        target_psd,
        target_freqs,
        target_periodic,
        fooof_freqs,
    )
    fitness = -neg_fitness

    if not v3._records:
        print("[error] No record appended (unexpected).")
        return 1

    rec = v3._records[-1]
    print("\n" + "=" * 60)
    print("RESULT (same fields as evolution CSV / JSON)")
    print("=" * 60)
    print(f"  return value (for DE minimisation): {neg_fitness:.6f}")
    print(f"  fitness (maximise this):             {fitness:.6f}")
    print(f"  shape_r:         {rec.get('shape_r')}")
    print(f"  so_power:        {rec.get('so_power')}")
    print(f"  spindle_power:   {rec.get('spindle_power')}")
    print(f"  dynamics_score:  {rec.get('dynamics_score')}")
    print("\n  V3 sub-tests (1 = pass):")
    print(f"    T1 DOWN   : {rec.get('T1_down')}   (min r_E < {v3.DOWN_THRESH_HZ} Hz)")
    print(f"    T2 UP     : {rec.get('T2_up')}     (max r_E > {v3.UP_THRESH_HZ} Hz)  max_rE={rec.get('max_rE')} Hz")
    print(f"    T3 sustain: {rec.get('T3_sustained')}  (UP run ≥ {v3.UP_DURATION_MS} ms)  longest={rec.get('longest_up_ms')} ms")
    print(f"    T4 SO peak: {rec.get('T4_so_freq')}   in [{v3.SO_FREQ_LO},{v3.SO_FREQ_HI}] Hz  peak={rec.get('so_peak_hz')} Hz")
    print(f"    T5 spindle: {rec.get('T5_spindle')}   FWHM > {v3.SPINDLE_FWHM_MIN} Hz  FWHM={rec.get('spindle_fwhm')} Hz")

    print("\n  Decomposition check (should match score within rounding):")
    sr = float(rec.get("shape_r", 0.0))
    so = float(rec.get("so_power", 0.0))
    sp = float(rec.get("spindle_power", 0.0))
    dy = float(rec.get("dynamics_score", 0.0))
    recomputed = 0.35 * sr + 0.15 * so + 0.15 * sp + 0.35 * dy
    print(f"    0.35*shape_r + 0.15*so + 0.15*spindle + 0.35*dyn = {recomputed:.6f}")
    print(f"    recorded score                                      = {rec.get('score')}")

    # 中文：默认开启 --plots——在 [4] 结束后用同一组 params_vec 再跑 plot_sim_ms 的仿真，调用 _generate_fig7_plots 写 outputs/{out_prefix}_*.png；--no-plots 可跳过；--out-v2-names 改用 fig7_v2_* 前缀。
    if args.plots:
        out_prefix = "fig7_v2" if args.out_v2_names else "fig7_v3_test"
        try:
            _generate_fig7_plots(
                params_vec,
                rec,
                target_psd,
                target_freqs,
                target_periodic,
                fooof_freqs,
                plot_sim_ms=args.plot_sim_ms,
                out_prefix=out_prefix,
            )
        except Exception as e:
            print(f"\n[error] Figure generation failed: {e}")
            return 1

    print("\nDone. See docs/s4_personalize_fig7_v3_compute_fitness_v3.md for line-by-line meaning.")
    if not args.plots:
        print("Tip: pass --plots to write Fig.7-style spectra / residuals / timeseries PNGs under outputs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
