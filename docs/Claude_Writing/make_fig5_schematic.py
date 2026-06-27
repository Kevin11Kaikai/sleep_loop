"""
Fig. 5 for the BMEiCON Stage 1 manuscript: residual-loophole schematic.

V7's audit (T8/T12) verifies that spindle EVENTS are real, but it inspects the
THALAMIC node, while the EEG-analog signal is the CORTICAL node. At V7's small
thalamo-cortical coupling (c_th2ctx ~= 0.0127), thalamic spindles do not
propagate to cortex, so cortical spindle density is ~0 even though the audit
passes. The audit's reality check and the held-out cortical measurement look at
different nodes -> a residual loophole.

This is a concept schematic (boxes + cartoon waveforms), not fitted data.

Run:  python docs/Claude_Writing/make_fig5_schematic.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[2] / "validation_outputs" / "fig_residual_loophole_schematic.png"

fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.4)
ax.axis("off")

GREEN, RED, GREY = "#2ca02c", "#d62728", "#555555"


def node_box(x, y, w, h, fc, ec, title):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                         linewidth=2, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top",
            fontsize=12, fontweight="bold", color=ec, zorder=3)


def mini_wave(xc, yc, xw, yamp, signal, color):
    """Draw a cartoon waveform in data coordinates."""
    xs = np.linspace(xc - xw / 2, xc + xw / 2, signal.size)
    ax.plot(xs, yc + yamp * signal, color=color, lw=1.1, zorder=3)


tt = np.linspace(0, 1, 600)
spindle = np.exp(-((tt - 0.5) ** 2) / (2 * 0.16 ** 2)) * np.sin(2 * np.pi * 11 * tt)
flat = 0.12 * np.sin(2 * np.pi * 1.0 * tt)

# ---- thalamic node (left) --------------------------------------------------
node_box(0.5, 1.3, 3.4, 3.4, "#eaf6ea", GREEN, "Thalamic node")
ax.text(2.2, 3.95, "spindles generated here", ha="center", va="center",
        fontsize=8.5, color=GREY)
mini_wave(2.2, 3.05, 2.7, 0.55, spindle, GREEN)
ax.text(2.2, 1.95, "T8: 25 events\nT12: 20 verified  ✓",
        ha="center", va="center", fontsize=9, color=GREEN,
        bbox=dict(boxstyle="round", fc="white", ec=GREEN, alpha=0.95))

# ---- cortical node (right) -------------------------------------------------
node_box(6.1, 1.3, 3.4, 3.4, "#fdecec", RED, "Cortical node")
ax.text(7.8, 3.95, "EEG-analog signal (held-out)", ha="center", va="center",
        fontsize=8.5, color=GREY)
mini_wave(7.8, 3.05, 2.7, 1.0, flat, RED)
ax.text(7.8, 1.95, "0 spindles / 300 s  ✗",
        ha="center", va="center", fontsize=9, color=RED,
        bbox=dict(boxstyle="round", fc="white", ec=RED, alpha=0.95))

# ---- weak coupling arrow ---------------------------------------------------
arr = FancyArrowPatch((3.95, 3.05), (6.05, 3.05), arrowstyle="-|>",
                      mutation_scale=18, lw=1.3, color=GREY,
                      linestyle=(0, (4, 3)), zorder=1)
ax.add_patch(arr)
ax.text(5.0, 3.65, "c_th2ctx ≈ 0.0127", ha="center", va="bottom",
        fontsize=10.5, fontweight="bold", color="#333333")
ax.text(5.0, 2.65, "weak coupling:\nspindles do not\npropagate", ha="center",
        va="top", fontsize=8.5, color=GREY)

# ---- node-role captions above the boxes ------------------------------------
ax.text(2.2, 5.05, "(audited node)", ha="center", va="bottom",
        fontsize=8.5, style="italic", color=GREEN)
ax.text(7.8, 5.05, "(held-out measurement)", ha="center", va="bottom",
        fontsize=8.5, style="italic", color=RED)

# ---- bottom takeaway -------------------------------------------------------
ax.text(5.0, 0.5,
        "Audit verifies spindle reality on the thalamic node; weak coupling "
        "suppresses cortical expression →\nthe reality check and the held-out "
        "cortical measurement inspect different nodes (residual loophole).",
        ha="center", va="center", fontsize=9,
        bbox=dict(boxstyle="round", fc="#fff3cd", ec="#d6b656"))

ax.set_title("Residual loophole: audited (thalamic) vs measured (cortical) spindles",
             fontsize=12.5, pad=14)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT}")
