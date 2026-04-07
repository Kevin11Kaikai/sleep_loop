"""
s3_band_power_bars.py
对 s3_sleep_kernel 输出的皮层/丘脑发放率做 Welch PSD，再画与 s1_band_power_bars 同风格的柱状图。

前置：先运行  python models/s3_sleep_kernel.py  生成
  outputs/r_cortex.npy, outputs/r_thalamus.npy, outputs/t_ms.npy

输出：
  outputs/psd_sim_band_power_bars.png
      — 总功率 0.2–30 Hz，5 频段（含 0.2–1.5 Hz Slow-wave，与 psd_validation 橙带一致）
  outputs/psd_sim_band_power_bars_EEGlike.png
      — 总功率 0.5–30 Hz，4 频段（与 Sleep-EDF band 柱状图可直接对比）
  outputs/sim_band_power_summary.csv
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if len(y) < 2:
        return float(np.sum(y)) if len(y) == 1 else 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def band_power(freqs: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    m = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(m):
        return 0.0
    return _trapz(psd[m], freqs[m])


def load_rates_and_psd():
    rc = os.path.join(ROOT, "outputs", "r_cortex.npy")
    rt = os.path.join(ROOT, "outputs", "r_thalamus.npy")
    tt = os.path.join(ROOT, "outputs", "t_ms.npy")
    for p in (rc, rt, tt):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"未找到 {p}。请先运行: python models/s3_sleep_kernel.py")

    r_cortex = np.load(rc)
    r_thalamus = np.load(rt)
    t_ms = np.load(tt)

    dt_s = (t_ms[-1] - t_ms[0]) / (len(t_ms) - 1)
    fs = 1000.0 / dt_s if dt_s > 1.0 else 1.0 / dt_s

    n_drop = int(5.0 * fs)
    r_c = r_cortex[n_drop:]
    r_t = r_thalamus[n_drop:]

    nperseg = min(int(10.0 * fs), len(r_c))
    f_c, p_c = welch(r_c, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    f_t, p_t = welch(r_t, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    # 两条轴应一致（同一 fs / nperseg）
    if not np.allclose(f_c, f_t):
        raise RuntimeError("Welch 频率轴皮层/丘脑不一致")
    return f_c, p_c, p_t, fs


def plot_two_region_bars(
    freqs: np.ndarray,
    p_c: np.ndarray,
    p_t: np.ndarray,
    f_lo: float,
    f_hi: float,
    bands: list[tuple[str, str, float, float, str]],
    title_suffix: str,
    out_name: str,
    y_abs_label: str,
):
    """bands: (key, display_label, lo, hi, color)"""
    regions = [
        ("Cortex $r_E$", p_c),
        ("Thalamus $r_{TCR}$", p_t),
    ]

    rows = []
    rel_data = []  # (2, n_bands)
    abs_data = []

    for reg_name, psd in regions:
        total = band_power(freqs, psd, f_lo, f_hi)
        rels, abss = [], []
        for key, _lbl, lo, hi, _c in bands:
            p = band_power(freqs, psd, lo, hi)
            rel = (p / total) if total > 1e-30 else 0.0
            rows.append(
                {
                    "region": reg_name.replace("$", "").replace("_", ""),
                    "band": key,
                    "f_lo": lo,
                    "f_hi": hi,
                    "abs_integral": p,
                    "rel_frac": rel,
                    "total_f_lo": f_lo,
                    "total_f_hi": f_hi,
                }
            )
            rels.append(rel)
            abss.append(p)
        rel_data.append(rels)
        abs_data.append(abss)

    rel_data = np.array(rel_data)
    abs_data = np.array(abs_data)
    n_b = len(bands)

    # ── 2x2: 皮层堆叠/分组 + 丘脑堆叠/分组 ───────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    labels_short = ["Cortex", "Thalamus"]

    for row, (reg_name, _) in enumerate(regions):
        rel_row = rel_data[row]
        ax = axes[row, 0]
        bottom = 0.0
        for j, (_k, lbl, _lo, _hi, color) in enumerate(bands):
            h = rel_row[j] * 100.0
            ax.bar(
                [0],
                [h],
                bottom=[bottom],
                width=0.55,
                label=lbl,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
            bottom += h
        ax.set_xticks([0])
        ax.set_xticklabels([labels_short[row]])
        ax.set_ylabel("Relative band power [%]")
        ax.set_title(
            f"{reg_name} — spectral budget\n"
            f"({f_lo:.1f}–{f_hi:.0f} Hz total, ∫PSD df per band / total)"
        )
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right", fontsize=7)

        # 分组
        ax = axes[row, 1]
        width = min(0.15, 0.8 / max(n_b, 1))
        for j, (_k, lbl, _lo, _hi, color) in enumerate(bands):
            offset = (j - (n_b - 1) / 2) * width
            ax.bar(
                [0 + offset],
                [rel_row[j] * 100.0],
                width,
                label=lbl,
                color=color,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xticks([0])
        ax.set_xticklabels([labels_short[row]])
        ax.set_ylabel("Relative band power [%]")
        ax.set_title(f"{reg_name} — grouped")
        ax.legend(loc="upper right", fontsize=6)

    plt.suptitle(
        f"Thalamocortical simulation — band power ({title_suffix})",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    out_png = os.path.join(ROOT, "outputs", out_name)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")

    # ── 绝对功率 log：两区域并排对比（每组 n_b 根柱）──────────────────────
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    width = min(0.11, 0.35 / n_b)
    for ri, lab in enumerate(labels_short):
        base = ri * (n_b + 1)
        for j, (_k, lbl, _lo, _hi, color) in enumerate(bands):
            xpos = base + j
            val = max(abs_data[ri, j], 1e-25)
            ax2.bar(xpos, val, width=width * 0.95, color=color, edgecolor="white", lw=0.4)

    ax2.axvline(n_b + 0.5, color="k", lw=0.5, alpha=0.25)

    tick_pos = []
    tick_lbl = []
    for ri, lab in enumerate(labels_short):
        base = ri * (n_b + 1)
        tick_pos.append(base + (n_b - 1) / 2)
        tick_lbl.append(lab)
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_lbl)
    ax2.set_yscale("log")
    ax2.set_ylabel(y_abs_label)
    ax2.set_title(f"Absolute band power — log scale ({title_suffix})")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=b[4], edgecolor="white")
        for b in bands
    ]
    ax2.legend(handles, [b[1] for b in bands], loc="upper right", fontsize=7)
    plt.tight_layout()
    out_png2 = os.path.join(ROOT, "outputs", out_name.replace(".png", "_absolute_log.png"))
    plt.savefig(out_png2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png2}")

    return pd.DataFrame(rows)


def main():
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    freqs, p_c, p_t, fs = load_rates_and_psd()
    print(f"Inferred fs = {fs:.1f} Hz (Welch nperseg from test_spindles recipe)")

    y_label = r"$\int$PSD d$f$ [(Hz)$^2$]  (rate PSD integral)"

    # 5 带：与 psd_validation 橙/绿语义对齐
    bands_5 = [
        ("slow_wave", "Slow-wave\n0.2–1.5 Hz", 0.2, 1.5, "#FDB863"),
        ("delta_rest", "Delta\n1.5–4 Hz", 1.5, 4.0, "#6BAED6"),
        ("theta", "Theta\n4–8 Hz", 4.0, 8.0, "#74C476"),
        ("alpha_sigma", "Alpha/Sigma\n8–13 Hz", 8.0, 13.0, "#FD8D3C"),
        ("beta", "Beta\n13–30 Hz", 13.0, 30.0, "#E377C2"),
    ]
    df1 = plot_two_region_bars(
        freqs,
        p_c,
        p_t,
        0.2,
        30.0,
        bands_5,
        "0.2–30 Hz, incl. validation slow-wave band",
        "psd_sim_band_power_bars.png",
        y_label,
    )

    bands_4 = [
        ("delta", "Delta\n0.5–4 Hz", 0.5, 4.0, "#6BAED6"),
        ("theta", "Theta\n4–8 Hz", 4.0, 8.0, "#74C476"),
        ("alpha_sigma", "Alpha/Sigma\n8–13 Hz", 8.0, 13.0, "#FD8D3C"),
        ("beta", "Beta\n13–30 Hz", 13.0, 30.0, "#E377C2"),
    ]
    df2 = plot_two_region_bars(
        freqs,
        p_c,
        p_t,
        0.5,
        30.0,
        bands_4,
        "0.5–30 Hz (EEG-aligned, compare to psd_band_power_bars)",
        "psd_sim_band_power_bars_EEGlike.png",
        y_label,
    )

    df = pd.concat([df1.assign(schema="sim_0p2_30"), df2.assign(schema="sim_0p5_30")], ignore_index=True)
    csv_path = os.path.join(ROOT, "outputs", "sim_band_power_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
