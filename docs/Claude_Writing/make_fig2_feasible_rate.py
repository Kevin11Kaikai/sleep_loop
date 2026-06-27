"""
Fig. 2 for the BMEiCON Stage 1 manuscript: feasible-solution recovery V4-V7.

Source numbers are the manuscript feasibility table (Sec. V-B):
    V4  22/4703
    V5   5/4960
    V6   1/4960
    V7 364/4960   (156/4960 survive V6's strict T5 FWHM>2.0 Hz rule)

The V7 de-confounded count (156) is overlaid as a hatched bar so the figure
carries the same caveat as the text: ~half of V7's recovery is T5 relaxation.
Log y-scale is used so the 1 -> 364 order-of-magnitude shift is visible.

Run:  python docs/Claude_Writing/make_fig2_feasible_rate.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "validation_outputs" / "fig_feasible_rate_recovery.png"

versions = ["V4", "V5", "V6", "V7"]
feasible = [22, 5, 1, 364]
total = [4703, 4960, 4960, 4960]
v7_deconf = 156  # V7 points still feasible under V6's strict T5 (FWHM > 2.0 Hz)

x = range(len(versions))

fig, ax = plt.subplots(figsize=(6.5, 4.0))

bars = ax.bar(x, feasible, width=0.6, color="#1f77b4", zorder=3,
              label="feasible (V7 audit)")

# Overlay V7's de-confounded count as a hatched bar on the V7 column.
ax.bar(len(versions) - 1, v7_deconf, width=0.6, facecolor="none",
       edgecolor="#d62728", hatch="////", linewidth=1.3, zorder=4,
       label="V7 under V6 strict T5 (de-confounded)")

ax.set_yscale("log")
ax.set_ylim(0.7, 700)
ax.set_ylabel("feasible evaluations (count, log scale)")
ax.set_xticks(list(x))
ax.set_xticklabels(versions)
ax.set_title("Feasible-solution recovery (V4–V7)")
ax.grid(axis="y", which="both", linestyle=":", alpha=0.4, zorder=0)

# Annotate each bar with its n/N fraction.
for i, (n, N) in enumerate(zip(feasible, total)):
    ax.annotate(f"{n}/{N}", (i, n), textcoords="offset points",
                xytext=(0, 5), ha="center", fontsize=9)

# Annotate the de-confounded bar.
ax.annotate(f"{v7_deconf}/4960", (len(versions) - 1, v7_deconf),
            textcoords="offset points", xytext=(22, 0), ha="left",
            va="center", fontsize=8, color="#d62728")

ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"Saved: {OUT}")
