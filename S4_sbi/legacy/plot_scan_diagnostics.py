"""
plot_scan_diagnostics.py
========================
Generates diagnostic figures from the 2026-05-10 parameter scan
(scan_xobs_params.py) for inclusion in progress reports.

Uses hardcoded scan results — no EEG reload needed.

Saves to S4_sbi/scan_diagnostics/:
  fig_t6_threshold_scan.png  — ibi_cv vs threshold, target band shaded
  fig_mi_prominence_scan.png — MI and n_peaks vs prominence_frac, target band shaded

Usage:
    conda activate neurolib
    python S4_sbi/plot_scan_diagnostics.py
"""

import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = _SCRIPT_DIR / "scan_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hardcoded scan results from 2026-05-10 run ───────────────────────────────

T6_RESULTS = [
    {"threshold": 5,  "n_bursts":   4, "ibi_cv": 0.631},
    {"threshold": 8,  "n_bursts":  23, "ibi_cv": 1.542},
    {"threshold": 10, "n_bursts":  88, "ibi_cv": 1.955},
    {"threshold": 12, "n_bursts": 196, "ibi_cv": 2.265},
    {"threshold": 15, "n_bursts": 368, "ibi_cv": 1.714},
]

MI_RESULTS = [
    {"pf": 0.05, "n_peaks": 3761, "mi": 0.00017, "udr": 1.232},
    {"pf": 0.10, "n_peaks": 3479, "mi": 0.00025, "udr": 1.219},
    {"pf": 0.15, "n_peaks": 3143, "mi": 0.00018, "udr": 1.219},
    {"pf": 0.20, "n_peaks": 2752, "mi": 0.00014, "udr": 1.244},
    {"pf": 0.30, "n_peaks": 1901, "mi": 0.00012, "udr": 1.280},
]

# Target bands
T6_LO, T6_HI = 0.3, 0.6
MI_LO, MI_HI = 0.01, 0.05


# ── Figure 1: T6 threshold scan ───────────────────────────────────────────────

def plot_t6():
    thresholds = [r["threshold"] for r in T6_RESULTS]
    ibi_cvs    = [r["ibi_cv"]    for r in T6_RESULTS]
    n_bursts   = [r["n_bursts"]  for r in T6_RESULTS]

    fig, ax1 = plt.subplots(figsize=(7, 5))

    # ibi_cv bars
    colors = ["#d62728" if not (T6_LO <= cv <= T6_HI) else "#2ca02c" for cv in ibi_cvs]
    bars = ax1.bar([str(t) for t in thresholds], ibi_cvs, color=colors,
                   alpha=0.85, zorder=3, width=0.5)
    ax1.set_xlabel("Hard threshold on 500ms-smoothed r_proxy", fontsize=11)
    ax1.set_ylabel("IBI CV (T6_ibi_cv)", fontsize=11)

    # target band
    ax1.axhspan(T6_LO, T6_HI, color="#2ca02c", alpha=0.12, zorder=1,
                label=f"Target band [{T6_LO}, {T6_HI}]")
    ax1.axhline(T6_LO, color="#2ca02c", ls="--", lw=1.2, zorder=2)
    ax1.axhline(T6_HI, color="#2ca02c", ls="--", lw=1.2, zorder=2)

    # annotate n_bursts
    for bar, n in zip(bars, n_bursts):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.04,
                 f"n={n}", ha="center", va="bottom", fontsize=9)

    ax1.set_ylim(0, max(ibi_cvs) * 1.18)
    ax1.set_title("T6 Threshold Scan\n"
                  "r_proxy envelope incompatible with sharp UP/DOWN transitions",
                  fontsize=11)

    red_patch  = mpatches.Patch(color="#d62728", alpha=0.85, label="FAIL (outside target)")
    green_band = mpatches.Patch(color="#2ca02c", alpha=0.12, label=f"Target [{T6_LO}, {T6_HI}]")
    ax1.legend(handles=[red_patch, green_band], fontsize=9, loc="upper right")

    ax1.grid(axis="y", lw=0.5, alpha=0.4, zorder=0)
    fig.tight_layout()

    path = OUT_DIR / "fig_t6_threshold_scan.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 2: MI prominence scan ─────────────────────────────────────────────

def plot_mi():
    pfs      = [r["pf"]      for r in MI_RESULTS]
    mis      = [r["mi"]      for r in MI_RESULTS]
    n_peaks  = [r["n_peaks"] for r in MI_RESULTS]
    udrs     = [r["udr"]     for r in MI_RESULTS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: MI vs prominence_frac ---
    pf_labels = [str(p) for p in pfs]
    colors_mi = ["#d62728" if not (MI_LO <= m <= MI_HI) else "#2ca02c" for m in mis]
    bars = ax1.bar(pf_labels, mis, color=colors_mi, alpha=0.85, zorder=3, width=0.06)
    ax1.axhspan(MI_LO, MI_HI, color="#2ca02c", alpha=0.12, zorder=1,
                label=f"Target band [{MI_LO}, {MI_HI}]")
    ax1.axhline(MI_LO, color="#2ca02c", ls="--", lw=1.2, zorder=2)
    ax1.axhline(MI_HI, color="#2ca02c", ls="--", lw=1.2, zorder=2)
    for bar, m in zip(bars, mis):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.000008,
                 f"{m:.5f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xlabel("SO_PEAK_PROMINENCE_FRAC", fontsize=11)
    ax1.set_ylabel("Modulation Index (MI)", fontsize=11)
    ax1.set_title("MI vs Prominence Fraction\n"
                  "MI ~10x below target regardless of parameter", fontsize=11)
    ax1.set_ylim(0, MI_HI * 0.6)  # show target band clearly even though MI is tiny
    red_patch  = mpatches.Patch(color="#d62728", alpha=0.85, label="FAIL (all points)")
    green_band = mpatches.Patch(color="#2ca02c", alpha=0.12,
                                label=f"Target [{MI_LO}, {MI_HI}]")
    ax1.legend(handles=[red_patch, green_band], fontsize=9)
    ax1.grid(axis="y", lw=0.5, alpha=0.4, zorder=0)

    # --- Right: n_so_peaks vs prominence_frac ---
    ax2.plot(pfs, n_peaks, "o-", color="steelblue", lw=2, ms=7, zorder=3)
    for pf, n in zip(pfs, n_peaks):
        ax2.annotate(f"  {n}", (pf, n), fontsize=9, va="center")

    # reference lines: expected SO count at 0.5 Hz and 1.0 Hz for 4260 s
    dur_s = 4260
    ax2.axhline(0.5 * dur_s, color="orange", ls="--", lw=1.2,
                label=f"Expected at 0.5 Hz ({int(0.5*dur_s)})")
    ax2.axhline(1.0 * dur_s, color="red",    ls="--", lw=1.2,
                label=f"Expected at 1.0 Hz ({int(1.0*dur_s)})")
    ax2.set_xlabel("SO_PEAK_PROMINENCE_FRAC", fontsize=11)
    ax2.set_ylabel("n_so_peaks detected", fontsize=11)
    ax2.set_title("Detected SO Peaks vs Prominence Fraction\n"
                  "Even with high prominence, peak count stays near physiological range;\n"
                  "MI still near zero => noise coupling, not real PAC", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(lw=0.5, alpha=0.4, zorder=0)

    fig.suptitle("MI Prominence Scan  (SC4001 N3 EEG, 4260 s)\n"
                 "Root cause: r_proxy has no 10-14 Hz content -> PAC amplitude = noise",
                 fontsize=11, y=1.01)
    fig.tight_layout()

    path = OUT_DIR / "fig_mi_prominence_scan.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating diagnostic plots from 2026-05-10 scan ...")
    plot_t6()
    plot_mi()
    print(f"Done. Figures in {OUT_DIR}")
