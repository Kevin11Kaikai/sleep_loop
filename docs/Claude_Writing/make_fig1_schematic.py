"""
Fig. 1 for the BMEiCON Stage 1 manuscript: schematic of the spectral-fit
loophole. Two models can share an almost identical power spectrum (what the
spectral objective sees) while having very different time-domain dynamics (what
it ignores): one with genuine UP/DOWN bistability + spindle bursts, one with
spiky transients and no sustained UP state or SO-spindle coupling.

These traces are illustrative cartoons, not fitted data; the figure is a
concept schematic.

Run:  python docs/Claude_Writing/make_fig1_schematic.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

OUT = Path(__file__).resolve().parents[2] / "validation_outputs" / "fig_loophole_schematic.png"

rng = np.random.default_rng(0)

# ---- synthetic PSDs (nearly identical) -------------------------------------
f = np.logspace(np.log10(0.3), np.log10(30), 400)


def psd(so_amp, sp_amp, jitter):
    aperiodic = 1.0 / (f ** 1.4)
    so = so_amp * np.exp(-((np.log10(f) - np.log10(1.0)) ** 2) / 0.02)
    sp = sp_amp * np.exp(-((np.log10(f) - np.log10(12.0)) ** 2) / 0.004)
    base = aperiodic + so + sp
    return base * (1 + jitter * 0.04 * np.sin(6 * np.log10(f)))


psd_valid = psd(0.9, 0.22, +1)
psd_invalid = psd(0.86, 0.205, -1)

# ---- illustrative time series ----------------------------------------------
t = np.linspace(0, 6, 3000)


def valid_trace():
    # bistable slow oscillation: smooth square-ish UP/DOWN
    so = np.tanh(3.0 * np.sin(2 * np.pi * 0.9 * t))
    up = (so > 0).astype(float)
    # spindles riding on UP states
    sp = 0.35 * np.sin(2 * np.pi * 12 * t) * up
    sp_env = up * (0.5 + 0.5 * np.sin(2 * np.pi * 0.9 * t - 0.5))
    y = 0.9 * so + sp * sp_env + 0.03 * rng.standard_normal(t.size)
    return y


def invalid_trace():
    # spiky transients: brief sharp peaks, no sustained UP, no spindle coupling
    y = np.full(t.size, -0.7)
    peak_times = np.arange(0.35, 6, 0.62)
    for pt in peak_times:
        y += 2.4 * np.exp(-((t - pt) ** 2) / (2 * 0.012 ** 2))
    y += 0.03 * rng.standard_normal(t.size)
    return y


# ============================================================================
fig = plt.figure(figsize=(8.2, 4.4))
gs = gridspec.GridSpec(2, 2, width_ratios=[1.05, 1.25], height_ratios=[1, 1],
                       hspace=0.55, wspace=0.32)

# Left: shared spectrum
axp = fig.add_subplot(gs[:, 0])
axp.loglog(f, psd_valid, color="#2ca02c", lw=2.2, label="Model A (valid)")
axp.loglog(f, psd_invalid, color="#d62728", lw=2.2, ls="--",
           label="Model B (invalid)")
axp.set_title("What the spectral objective sees", fontsize=11)
axp.set_xlabel("frequency (Hz)")
axp.set_ylabel("power (a.u.)")
axp.legend(fontsize=8, loc="lower left")
axp.annotate("SO peak", (1.0, psd_valid[np.argmin(abs(f - 1.0))]),
             xytext=(0.4, 4), textcoords="data", fontsize=8,
             arrowprops=dict(arrowstyle="->", lw=0.8))
axp.annotate("spindle\nbump", (12, psd_valid[np.argmin(abs(f - 12))]),
             xytext=(15, 0.06), textcoords="data", fontsize=8,
             arrowprops=dict(arrowstyle="->", lw=0.8))
axp.text(0.5, 0.93, "shape_r high\n→ spectra look matched",
         transform=axp.transAxes, ha="center", va="top", fontsize=9,
         bbox=dict(boxstyle="round", fc="#fff3cd", ec="#d6b656"))

# Right top: valid dynamics
axv = fig.add_subplot(gs[0, 1])
axv.plot(t, valid_trace(), color="#2ca02c", lw=0.9)
axv.set_title("What it misses: time-domain dynamics", fontsize=11)
axv.set_ylabel("A (valid)", fontsize=9)
axv.set_xticklabels([])
axv.text(0.99, 0.05, "sustained UP/DOWN + coupled spindles",
         transform=axv.transAxes, ha="right", va="bottom", fontsize=8,
         color="#2ca02c")
axv.set_ylim(-1.6, 1.8)

# Right bottom: invalid dynamics
axi = fig.add_subplot(gs[1, 1])
axi.plot(t, invalid_trace(), color="#d62728", lw=0.9)
axi.set_ylabel("B (invalid)", fontsize=9)
axi.set_xlabel("time (s)")
axi.text(0.99, 0.92, "spiky transients: no sustained UP, no SO-spindle coupling",
         transform=axi.transAxes, ha="right", va="top", fontsize=8,
         color="#d62728")
axi.set_ylim(-1.3, 2.2)

fig.suptitle("Spectral match ≠ dynamic match", fontsize=13, y=0.99)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT}")
