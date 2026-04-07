"""
s1_band_power_bars.py
从 s1_all_stages.py 已保存的 PSD 计算各频段功率，画柱状图（解决「全谱归一后线性轴看不清」的问题）。

前置：先运行  python models/s1_all_stages.py  生成
  data/target_freqs.npy, data/psd_wake.npy, data/psd_n1.npy, ...

输出：
  outputs/psd_band_power_bars.png  — 左：堆叠相对功率；右：分组柱状（相对功率）
  outputs/band_power_summary.csv   — 数值表
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 项目根目录（从 models/ 或根目录运行均可）
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

STAGES = ["wake", "n1", "n2", "n3", "rem"]
STAGE_DISPLAY = {
    "wake": "WAKE",
    "n1": "N1",
    "n2": "N2",
    "n3": "N3",
    "rem": "REM",
}

# 与 psd_all_stages 图一致：总功率定义在 0.5–30 Hz
F_TOTAL_LO, F_TOTAL_HI = 0.5, 30.0

# (列名, f_lo, f_hi, 颜色)
BANDS = [
    ("delta", 0.5, 4.0, "#6BAED6"),      # Delta — 与图中浅蓝带一致
    ("theta", 4.0, 8.0, "#74C476"),
    ("alpha_sigma", 8.0, 13.0, "#FD8D3C"),  # Alpha / Sigma
    ("beta", 13.0, 30.0, "#E377C2"),
]

BAND_LABELS = {
    "delta": "Delta\n0.5–4 Hz",
    "theta": "Theta\n4–8 Hz",
    "alpha_sigma": "Alpha/Sigma\n8–13 Hz",
    "beta": "Beta\n13–30 Hz",
}


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if len(y) < 2:
        return float(np.sum(y)) if len(y) == 1 else 0.0
    # NumPy 2.0+ 有 trapezoid；旧版用 trapz
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def band_power(freqs: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    """∫ PSD df 的梯形近似，单位与 Welch 一致（对 EEG 为 V² 量级）。"""
    m = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(m):
        return 0.0
    return _trapz(psd[m], freqs[m])


def load_psds():
    freq_path = os.path.join(ROOT, "data", "target_freqs.npy")
    if not os.path.isfile(freq_path):
        raise FileNotFoundError(
            f"未找到 {freq_path}。请先运行: python models/s1_all_stages.py"
        )
    freqs = np.load(freq_path)
    stage_psd = {}
    for st in STAGES:
        p = os.path.join(ROOT, "data", f"psd_{st}.npy")
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"未找到 {p}。请先运行: python models/s1_all_stages.py"
            )
        stage_psd[st] = np.load(p)
    return freqs, stage_psd


def main():
    freqs, stage_psd = load_psds()

    rows = []
    rel_matrix = []  # (n_stage, n_band) 相对 0.5–30Hz
    abs_matrix = []

    for st in STAGES:
        psd = stage_psd[st]
        total = band_power(freqs, psd, F_TOTAL_LO, F_TOTAL_HI)
        row = {"stage": st, "total_V2": total}
        rels = []
        abss = []
        for name, lo, hi, _ in BANDS:
            p = band_power(freqs, psd, lo, hi)
            rel = (p / total) if total > 1e-30 else 0.0
            row[f"{name}_abs"] = p
            row[f"{name}_rel"] = rel
            rels.append(rel)
            abss.append(p)
        rows.append(row)
        rel_matrix.append(rels)
        abs_matrix.append(abss)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    csv_path = os.path.join(ROOT, "outputs", "band_power_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    rel_matrix = np.array(rel_matrix)  # (5, 4)
    x = np.arange(len(STAGES))
    n_b = len(BANDS)

    # ── Figure: stacked + grouped ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左：堆叠柱状图（相对功率，百分比）
    ax = axes[0]
    bottom = np.zeros(len(STAGES))
    for j, (name, _lo, _hi, color) in enumerate(BANDS):
        h = rel_matrix[:, j] * 100.0
        ax.bar(
            x,
            h,
            bottom=bottom,
            label=BAND_LABELS[name],
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += h
    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_DISPLAY[s] for s in STAGES])
    ax.set_ylabel("Relative band power [%]")
    ax.set_title(
        f"Spectral budget (0.5–{F_TOTAL_HI:.0f} Hz)\n∫PSD df per band / total"
    )
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(100, color="k", lw=0.3, alpha=0.3)

    # 右：分组柱状图（同一数据，便于比「Delta 谁高」）
    ax = axes[1]
    width = 0.16
    for j, (name, _lo, _hi, color) in enumerate(BANDS):
        offset = (j - (n_b - 1) / 2) * width
        ax.bar(
            x + offset,
            rel_matrix[:, j] * 100.0,
            width,
            label=BAND_LABELS[name],
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_DISPLAY[s] for s in STAGES])
    ax.set_ylabel("Relative band power [%]")
    ax.set_title("Same data — grouped bars")
    ax.legend(loc="upper right", fontsize=7)

    plt.suptitle(
        "Sleep-EDF: band power (mean PSD per stage, trapezoid ∫PSD df)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    out_png = os.path.join(ROOT, "outputs", "psd_band_power_bars.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")

    # ── Optional: absolute power (log y) — second figure ───────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    abs_matrix = np.array(abs_matrix)
    width = 0.16
    for j, (name, _lo, _hi, color) in enumerate(BANDS):
        offset = (j - (n_b - 1) / 2) * width
        vals = np.maximum(abs_matrix[:, j], 1e-25)
        ax2.bar(x + offset, vals, width, label=BAND_LABELS[name], color=color, edgecolor="white", lw=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels([STAGE_DISPLAY[s] for s in STAGES])
    ax2.set_ylabel(r"Band power $\int$PSD d$f$ [V²]")
    ax2.set_yscale("log")
    ax2.set_title("Absolute band power (log scale)")
    ax2.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    out_png2 = os.path.join(ROOT, "outputs", "psd_band_power_bars_absolute_log.png")
    plt.savefig(out_png2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png2}")


if __name__ == "__main__":
    main()
