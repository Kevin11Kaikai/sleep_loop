"""
verify_pareto_seeds.py
======================
Stage 1 closure: physical verification of the 5 Pareto-front seeds.

Purpose
-------
The Pareto analysis (pareto_analysis.py) identified 5 representative
solutions on the Pareto front in (shape_r, PAC_compound) space. Those
are statistical claims: "these solutions are non-dominated under the
fitness function". This script answers the next question:

  Do these 5 seeds correspond to distinct PHYSIOLOGICAL regimes,
  or are they statistically distinguishable but physically similar?

If A (PAC-dominant) and E (shape-dominant) produce visibly different
time-domain dynamics — clearly different SO-spindle locking, different
event morphology, different polar PAC distributions — then the Pareto
front represents real scientific trade-offs, and SBI in Stage 2 will
have meaningful structure to recover.

If A through E look near-identical in the time domain, that would
suggest the fitness function is reading noise, not physiology, and
the Pareto front is an artifact.

This is the visual evidence supporting the methodological argument
in the progress report.

Outputs
-------
1. fig_pareto_seeds_verification.png — 5-column figure, one per seed.
   Each column shows: timeseries excerpt, event-locked spindle envelope,
   polar PAC histogram, summary metrics with auto-classified regime.

2. seeds_verification_metrics.csv — per-seed re-measured metrics
   (running each parameter set through a fresh 60s simulation), so
   you can compare the warm_start CSV's reported numbers against
   reproduced numbers. Discrepancies > 10% would be a concerning
   signal that warm_start metrics had run-to-run noise.

Usage
-----
  python verify_pareto_seeds.py \
      --seeds-json pareto_seeds.json \
      --plot-script /path/to/plot_fig7_compare_v7_vs_v8.py \
      --out-fig fig_pareto_seeds_verification.png \
      --out-csv seeds_verification_metrics.csv \
      --sim-dur-ms 60000

  # With sensible defaults (assumes both files in same dir):
  python verify_pareto_seeds.py
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# --------------------------------------------------------------------- #
# Import the existing comparison script as a module
# --------------------------------------------------------------------- #
def import_plot_module(plot_script_path):
    """
    Import plot_fig7_compare_v7_vs_v8.py as a module so we can reuse
    its simulation, event detection, and PAC computation functions.
    """
    plot_script_path = Path(plot_script_path).resolve()
    if not plot_script_path.exists():
        raise FileNotFoundError(f"Cannot find plot script: {plot_script_path}")

    print(f"Importing plot module from: {plot_script_path}")
    spec = importlib.util.spec_from_file_location(
        "plot_fig7_compare", str(plot_script_path)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["plot_fig7_compare"] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        # The plot script's main() may call sys.exit() when imported
        # without proper args; that's fine. We only need its functions.
        pass
    return module


# --------------------------------------------------------------------- #
# Regime classification
# --------------------------------------------------------------------- #
def classify_regime(seed_obj, all_seeds):
    """
    Auto-classify each seed as PAC-dominant / balanced / shape-dominant
    based on its position relative to the median across the 5 seeds.

    Returns a (label, color) tuple.
    """
    shape_rs = [s["objectives"]["shape_r"] for s in all_seeds]
    pacs = [s["objectives"]["PAC_compound"] for s in all_seeds]
    median_shape = np.median(shape_rs)
    median_pac = np.median(pacs)

    sr = seed_obj["objectives"]["shape_r"]
    pac = seed_obj["objectives"]["PAC_compound"]

    if pac >= median_pac and sr < median_shape:
        return ("PAC-dominant", "#1D9E75")
    if sr >= median_shape and pac < median_pac:
        return ("shape-dominant", "#D85A30")
    if sr >= median_shape and pac >= median_pac:
        return ("balanced (high)", "#534AB7")
    return ("balanced (low)", "#888780")


# --------------------------------------------------------------------- #
# Per-seed plotting
# --------------------------------------------------------------------- #
def plot_timeseries(ax, r_ctx, r_thal, ctx_peaks, sp_peaks, fs,
                     t_start_s=5.0, window_s=15.0):
    """Top panel: 15s of cortex r_E + thalamic r_TCR with markers."""
    i0 = int(t_start_s * fs)
    i1 = int((t_start_s + window_s) * fs)
    t = np.arange(i0, i1) / fs

    ax2 = ax.twinx()
    ax.plot(t, r_ctx[i0:i1], color='#185FA5', lw=0.8, label='r_E (cortex)')
    ax2.plot(t, r_thal[i0:i1], color='#0F6E56', lw=0.6, alpha=0.7,
              label='r_TCR (thalamus)')

    # Mark detected peaks within window
    in_win = (ctx_peaks >= i0) & (ctx_peaks < i1)
    if in_win.any():
        peaks_t = ctx_peaks[in_win] / fs
        ax.plot(peaks_t, r_ctx[ctx_peaks[in_win]], 'v', color='#A32D2D',
                 ms=4, mfc='#E24B4A', zorder=5)
    in_win_sp = (sp_peaks >= i0) & (sp_peaks < i1)
    if in_win_sp.any():
        sp_t = sp_peaks[in_win_sp] / fs
        ax2.plot(sp_t, r_thal[sp_peaks[in_win_sp]], '^',
                  color='#854F0B', ms=4, mfc='#EF9F27', zorder=5)

    ax.set_ylabel('r_E [Hz]', fontsize=8, color='#185FA5')
    ax.tick_params(axis='y', labelsize=7, colors='#185FA5')
    ax2.set_ylabel('r_TCR', fontsize=8, color='#0F6E56')
    ax2.tick_params(axis='y', labelsize=7, colors='#0F6E56')
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlabel('Time [s]', fontsize=8)
    ax.set_xlim(t_start_s, t_start_s + window_s)
    ax.grid(True, alpha=0.15, linewidth=0.4)
    ax.set_axisbelow(True)


def plot_event_locked(ax, r_thal, sp_envelope, ctx_peaks, fs,
                       window_s=0.7):
    """
    Middle panel: event-locked spindle envelope around UP peaks.
    Each light line is one cycle; bold line is the mean. The shape of
    this curve is the most direct visual evidence of SO-spindle locking.
    """
    n_win = int(window_s * fs)
    snippets = []
    for p in ctx_peaks:
        if p - n_win < 0 or p + n_win >= len(sp_envelope):
            continue
        snippets.append(sp_envelope[p - n_win: p + n_win])
    if not snippets:
        ax.text(0.5, 0.5, "no valid cycles",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=9, color='#888780')
        return

    arr = np.array(snippets)
    t = np.arange(-n_win, n_win) / fs
    mean = arr.mean(axis=0)
    sem = arr.std(axis=0) / np.sqrt(len(snippets))

    for s in arr:
        ax.plot(t, s, color='#888780', lw=0.3, alpha=0.18)
    ax.fill_between(t, mean - sem, mean + sem,
                     color='#0F6E56', alpha=0.25, linewidth=0)
    ax.plot(t, mean, color='#0F6E56', lw=2.0,
             label=f'mean (n={len(snippets)})')
    ax.axvline(0, color='#A32D2D', lw=0.8, ls='--', alpha=0.7)

    ax.set_xlabel('Time relative to UP peak [s]', fontsize=8)
    ax.set_ylabel('Spindle envelope', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.15, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)


def plot_polar_pac(ax, bin_centers, mean_amp, color='#534AB7'):
    """Bottom panel: polar histogram of PAC distribution (18 bins)."""
    if mean_amp.sum() <= 0:
        ax.text(0.5, 0.5, "no PAC", ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='#888780')
        return
    width = 2 * np.pi / len(bin_centers)
    p = mean_amp / mean_amp.sum()
    ax.bar(bin_centers, p, width=width, bottom=0,
            color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.set_xticks([0, np.pi / 2, np.pi, -np.pi / 2])
    ax.set_xticklabels(['0°\n(UP)', '+90°', '±180°\n(DOWN)', '−90°'],
                        fontsize=7)
    ax.tick_params(axis='y', labelsize=6)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.set_ylim(0, max(p.max() * 1.15, 0.001))


def write_metrics_box(ax, seed_obj, repro_metrics, regime_label,
                       regime_color):
    """
    Bottom row: text panel with original (pareto_seeds.json) metrics
    vs reproduced (this run) metrics, plus the regime classification.
    """
    ax.axis('off')

    orig = seed_obj["pac_metrics"]
    obj = seed_obj["objectives"]

    # Header pill with regime
    ax.add_patch(plt.Rectangle((0.05, 0.86), 0.9, 0.12,
                                transform=ax.transAxes,
                                facecolor=regime_color, alpha=0.18,
                                edgecolor=regime_color, linewidth=0.8))
    ax.text(0.5, 0.92, regime_label,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=10, fontweight='bold', color=regime_color)

    # Metrics table
    rows = [
        ("shape_r",    f"{obj['shape_r']:.4f}",     "—"),
        ("MI",         f"{orig['MI']:.4f}",
                       f"{repro_metrics['mi']:.4f}" if repro_metrics else "—"),
        ("up/down",    f"{orig['up_down_ratio']:.3f}",
                       f"{repro_metrics['up_down_ratio']:.3f}"
                          if repro_metrics else "—"),
        ("conc.",      f"{orig['concentration']:.4f}",
                       f"{repro_metrics['concentration']:.4f}"
                          if repro_metrics else "—"),
        ("φ argmax",   "—",
                       f"{repro_metrics['phase_argmax_deg']:+.0f}°"
                          if repro_metrics else "—"),
        ("# SO cyc.",  "—",
                       f"{repro_metrics['n_so_cycles']}"
                          if repro_metrics else "—"),
    ]

    y_top = 0.78
    line_h = 0.105
    ax.text(0.05, y_top + line_h * 0.4, "metric",
            transform=ax.transAxes, fontsize=8, color='#5F5E5A',
            fontweight='bold')
    ax.text(0.45, y_top + line_h * 0.4, "stage1",
            transform=ax.transAxes, fontsize=8, color='#5F5E5A',
            fontweight='bold')
    ax.text(0.75, y_top + line_h * 0.4, "repro",
            transform=ax.transAxes, fontsize=8, color='#5F5E5A',
            fontweight='bold')
    for k, (name, v_orig, v_repro) in enumerate(rows):
        y = y_top - (k + 1) * line_h
        ax.text(0.05, y, name, transform=ax.transAxes,
                fontsize=8, color='#2C2C2A')
        ax.text(0.45, y, v_orig, transform=ax.transAxes,
                fontsize=8, family='monospace', color='#444441')
        ax.text(0.75, y, v_repro, transform=ax.transAxes,
                fontsize=8, family='monospace', color='#0C447C')


# --------------------------------------------------------------------- #
# Main figure assembly
# --------------------------------------------------------------------- #
def make_verification_figure(seeds, results, out_path):
    """
    5 columns (one per seed) × 4 rows:
      Row 1: timeseries
      Row 2: event-locked envelope
      Row 3: polar PAC histogram
      Row 4: metrics + regime
    """
    n_seeds = len(seeds)
    fig = plt.figure(figsize=(4.0 * n_seeds, 13))
    gs = GridSpec(
        4, n_seeds,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.2, 0.95],
        hspace=0.42, wspace=0.32,
        top=0.94, bottom=0.05, left=0.06, right=0.97,
    )

    for i, (seed_obj, res) in enumerate(zip(seeds, results)):
        tag = seed_obj["tag"]
        regime_label, regime_color = classify_regime(seed_obj, seeds)

        # Column header — seed tag and shape_r/PAC summary
        fig.text(
            0.06 + (i + 0.5) * (0.91 / n_seeds), 0.965,
            f"Seed {tag}",
            ha='center', va='top',
            fontsize=14, fontweight='bold',
        )
        fig.text(
            0.06 + (i + 0.5) * (0.91 / n_seeds), 0.945,
            f"shape_r = {seed_obj['objectives']['shape_r']:.3f}   "
            f"PAC = {seed_obj['objectives']['PAC_compound']:.3f}",
            ha='center', va='top',
            fontsize=9, color='#5F5E5A',
        )

        # Row 1: timeseries
        ax1 = fig.add_subplot(gs[0, i])
        if res is not None:
            plot_timeseries(
                ax1, res["r_ctx"], res["r_thal"],
                res["ctx_peaks"], res["sp_peaks"], 1000.0,
            )
        else:
            ax1.text(0.5, 0.5, "simulation failed",
                     ha='center', va='center', transform=ax1.transAxes,
                     fontsize=9, color='#993C1D')

        # Row 2: event-locked envelope
        ax2 = fig.add_subplot(gs[1, i])
        if res is not None:
            plot_event_locked(
                ax2, res["r_thal"], res["sp_envelope"],
                res["ctx_peaks"], 1000.0,
            )
        else:
            ax2.axis('off')

        # Row 3: polar histogram
        ax3 = fig.add_subplot(gs[2, i], projection='polar')
        if res is not None:
            plot_polar_pac(ax3, res["bin_centers"], res["mean_amp"],
                            color=regime_color)
        else:
            ax3.axis('off')

        # Row 4: metrics + regime
        ax4 = fig.add_subplot(gs[3, i])
        write_metrics_box(
            ax4, seed_obj,
            res["metrics"] if (res is not None and res["metrics"]) else None,
            regime_label, regime_color,
        )

    # Row labels on the left edge
    row_labels = [
        "1.\nTimeseries\n(15 s)",
        "2.\nEvent-locked\nenvelope\n(±0.7 s)",
        "3.\nPolar PAC\nhistogram",
        "4.\nMetrics &\nregime",
    ]
    y_centers = [0.815, 0.612, 0.395, 0.155]
    for label, yc in zip(row_labels, y_centers):
        fig.text(0.012, yc, label, fontsize=9, ha='left', va='center',
                  color='#5F5E5A', fontweight='bold')

    fig.suptitle(
        "Stage 1 closure: physical verification of 5 Pareto-front seeds",
        fontsize=14, fontweight='bold', y=0.995,
    )

    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f"\n  Saved figure: {out_path}")
    plt.close(fig)


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds-json", type=str,
                        default="pareto_seeds.json")
    parser.add_argument("--plot-script", type=str,
                        default="plot_fig7_compare_v7_vs_v8.py",
                        help="Path to the existing plot_fig7_compare script "
                             "(its functions are reused).")
    parser.add_argument("--out-fig", type=str,
                        default="fig_pareto_seeds_verification.png")
    parser.add_argument("--out-csv", type=str,
                        default="seeds_verification_metrics.csv")
    parser.add_argument("--sim-dur-ms", type=int, default=60000,
                        help="Simulation duration per seed in ms.")
    args = parser.parse_args()

    print("=" * 60)
    print("Pareto seeds — physical verification")
    print("=" * 60)

    # Load seeds
    with open(args.seeds_json, "r", encoding="utf-8") as f:
        sj = json.load(f)
    seeds = sj["seeds"]
    print(f"  Loaded {len(seeds)} seeds from {args.seeds_json}")

    # Import plot module (gives us simulate, detect_*, analyze_one, etc.)
    plot_mod = import_plot_module(args.plot_script)

    # Run analyze_one for each seed
    results = []
    metrics_rows = []
    for seed_obj in seeds:
        tag = seed_obj["tag"]
        # Build the bp dict expected by analyze_one (includes _score)
        bp = dict(seed_obj["params"])
        bp["_score"] = seed_obj["objectives"]["score"]
        try:
            res = plot_mod.analyze_one(
                bp, args.sim_dur_ms, label=f"Seed {tag}"
            )
            results.append(res)
            m = res["metrics"] or {}
            orig = seed_obj["pac_metrics"]
            metrics_rows.append({
                "tag": tag,
                "shape_r": seed_obj["objectives"]["shape_r"],
                "PAC_compound": seed_obj["objectives"]["PAC_compound"],
                "MI_stage1": orig["MI"],
                "MI_repro": m.get("mi", np.nan),
                "udr_stage1": orig["up_down_ratio"],
                "udr_repro": m.get("up_down_ratio", np.nan),
                "conc_stage1": orig["concentration"],
                "conc_repro": m.get("concentration", np.nan),
                "phase_argmax_repro_deg": m.get("phase_argmax_deg", np.nan),
                "n_so_cycles_repro": m.get("n_so_cycles", np.nan),
            })
        except Exception as e:
            print(f"  [!] Seed {tag} failed: {e}")
            results.append(None)
            metrics_rows.append({
                "tag": tag, "shape_r": seed_obj["objectives"]["shape_r"],
                "PAC_compound": seed_obj["objectives"]["PAC_compound"],
                "MI_stage1": seed_obj["pac_metrics"]["MI"],
                "MI_repro": np.nan,
                "udr_stage1": seed_obj["pac_metrics"]["up_down_ratio"],
                "udr_repro": np.nan,
                "conc_stage1": seed_obj["pac_metrics"]["concentration"],
                "conc_repro": np.nan,
                "phase_argmax_repro_deg": np.nan,
                "n_so_cycles_repro": np.nan,
            })

    # CSV
    df = pd.DataFrame(metrics_rows)
    df.to_csv(args.out_csv, index=False, float_format="%.5f")
    print(f"\n  Saved CSV: {args.out_csv}")
    print(df.to_string(index=False))

    # Figure
    make_verification_figure(seeds, results, args.out_fig)

    # Summary check: does reproduction agree with stage 1 numbers?
    print("\n" + "=" * 60)
    print("Reproduction agreement check")
    print("=" * 60)
    for row in metrics_rows:
        if np.isnan(row["MI_repro"]):
            print(f"  Seed {row['tag']}: simulation failed.")
            continue
        d_mi = abs(row["MI_repro"] - row["MI_stage1"]) / max(row["MI_stage1"], 1e-6)
        d_udr = abs(row["udr_repro"] - row["udr_stage1"]) / row["udr_stage1"]
        flag = "OK" if (d_mi < 0.10 and d_udr < 0.10) else "DRIFT"
        print(f"  Seed {row['tag']}: ΔMI = {d_mi*100:5.1f}%  "
              f"Δudr = {d_udr*100:5.1f}%   [{flag}]")

    print("\nDone.")


if __name__ == "__main__":
    main()