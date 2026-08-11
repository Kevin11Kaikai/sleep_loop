"""
plot_concat_vs_mean_psd.py
==========================
Visual comparison: long-concatenation Welch PSD (v4 x_obs T4 path) vs
per-epoch mean PSD (V7 load_target_psd path).

Outputs (S4_sbi/scan_diagnostics/):
  fig_concat_vs_mean_psd_overview.png  — 0–20 Hz, three methods overlaid
  fig_concat_vs_mean_psd_so_zoom.png   — 0.1–3 Hz SO band + T4 metrics table
  fig_concat_vs_mean_psd_timeline.png  — schematic + short r_proxy trace

Usage (repo root):
    conda activate neurolib
    python S4_sbi/plot_concat_vs_mean_psd.py
"""

import os
import sys
import importlib.util
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import detrend, resample_poly
import mne

mne.set_log_level("WARNING")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
OUT_DIR = _SCRIPT_DIR / "scan_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.chdir(str(_ROOT))
sys.path.insert(0, str(_ROOT))

# NumPy shim (neurolib not required here)
import builtins as _builtins_mod
for _alias in ("int", "float", "bool", "object", "complex", "str"):
    if not hasattr(np, _alias):
        setattr(np, _alias, getattr(_builtins_mod, _alias))

_prep_spec = importlib.util.spec_from_file_location(
    "preprocess_psd", str(_ROOT / "utils" / "02_preprocess_psd.py")
)
_prep_mod = importlib.util.module_from_spec(_prep_spec)
_prep_spec.loader.exec_module(_prep_mod)
load_hypnogram = _prep_mod.load_hypnogram
compute_epoch_psd = _prep_mod.compute_epoch_psd
EPOCH_LEN_S = _prep_mod.EPOCH_LEN_S

# Match v4 / V7
SUBJECT_ID = "SC4001"
EEG_CHANNEL = "EEG Fpz-Cz"
N3_LABELS = ["N3"]
ARTIFACT_THRESH = 200e-6
FS_SIM = 1000.0
SO_FREQ_LO = 0.2
SO_FREQ_HI = 1.5
F_LO_V7, F_HI_V7 = 0.5, 20.0


def load_accepted_epochs():
    try:
        manifest = pd.read_csv("data/manifest.csv", encoding="utf-8")
    except UnicodeDecodeError:
        manifest = pd.read_csv("data/manifest.csv", encoding="utf-16")

    subj_row = manifest[manifest["subject_id"] == SUBJECT_ID].iloc[0]
    raw = mne.io.read_raw_edf(
        subj_row["psg_path"], include=[EEG_CHANNEL], preload=True, verbose=False
    )
    fs = float(raw.info["sfreq"])
    stages = load_hypnogram(Path(subj_row["hypnogram_path"]))
    data_uv = raw.get_data()[0] * 1e6
    n_per = int(EPOCH_LEN_S * fs)

    accepted, n_n3, n_rej = [], 0, 0
    for i in range(min(len(stages), len(data_uv) // n_per)):
        if stages[i] not in N3_LABELS:
            continue
        n_n3 += 1
        ep = data_uv[i * n_per : (i + 1) * n_per]
        if np.ptp(ep) > ARTIFACT_THRESH * 1e6:
            n_rej += 1
            continue
        accepted.append(ep)

    if not accepted:
        raise RuntimeError("No accepted N3 epochs")
    return accepted, fs, n_n3, n_rej


def epoch_to_r_proxy(epoch_uv, fs_native, p95_global):
    g = gcd(int(FS_SIM), int(fs_native))
    eeg_1k = resample_poly(epoch_uv, int(FS_SIM) // g, int(fs_native) // g)
    eeg_raw = detrend(eeg_1k, type="linear")
    r = np.abs(eeg_raw)
    r_smooth = gaussian_filter1d(r, sigma=50.0)
    r_proxy = r_smooth - r_smooth.min()
    return r_proxy / p95_global * 60.0


def build_concat_r_proxy(accepted, fs_native):
    g = gcd(int(FS_SIM), int(fs_native))
    parts_raw = []
    for ep in accepted:
        eeg_1k = resample_poly(ep, int(FS_SIM) // g, int(fs_native) // g)
        parts_raw.append(detrend(eeg_1k, type="linear"))
    eeg_raw = np.concatenate(parts_raw)
    r = np.abs(eeg_raw)
    r_smooth = gaussian_filter1d(r, sigma=50.0)
    r_proxy = r_smooth - r_smooth.min()
    p95 = np.percentile(r_proxy, 95)
    r_proxy = r_proxy / p95 * 60.0
    return r_proxy, eeg_raw, p95


def t4_from_psd(f, p, lo=SO_FREQ_LO, hi=SO_FREQ_HI):
    so_mask = (f >= lo) & (f <= hi)
    width = hi - lo
    neigh_lo = (f >= max(0.1, lo - width)) & (f < lo)
    neigh_hi = (f > hi) & (f <= hi + width)
    peak_freq, q = 0.0, 0.0
    if so_mask.any():
        peak_freq = float(f[so_mask][np.argmax(p[so_mask])])
        peak_val = float(p[so_mask].max())
        nbrs = np.concatenate([
            p[neigh_lo] if neigh_lo.any() else np.array([]),
            p[neigh_hi] if neigh_hi.any() else np.array([]),
        ])
        if len(nbrs) > 0 and nbrs.mean() > 0:
            q = float(peak_val / nbrs.mean())
    return peak_freq, q


def main():
    print("Loading SC4001 N3 epochs ...")
    accepted, fs_native, n_n3, n_rej = load_accepted_epochs()
    duration_s = sum(len(e) for e in accepted) / fs_native
    print(f"  N3 total={n_n3}  rejected={n_rej}  accepted={len(accepted)}  "
          f"duration={duration_s:.0f}s")

    r_proxy_cat, _, p95_global = build_concat_r_proxy(accepted, fs_native)
    f_concat, p_concat = compute_epoch_psd(r_proxy_cat, FS_SIM)
    t4f_c, t4q_c = t4_from_psd(f_concat, p_concat)

    # V7: mean PSD on raw EEG (per 30s epoch @ native fs)
    psds_raw, f_ep = [], None
    for ep in accepted:
        f_ep, p_ep = compute_epoch_psd(ep, fs_native)
        mask = (f_ep >= F_LO_V7) & (f_ep <= F_HI_V7)
        psds_raw.append(p_ep[mask])
    f_raw_mean = f_ep[mask]
    p_raw_mean = np.mean(psds_raw, axis=0)
    t4f_raw, t4q_raw = t4_from_psd(f_raw_mean, p_raw_mean)

    # Per-epoch r_proxy PSD then mean (global p95 from concat)
    psds_rp, f_rp = [], None
    for ep in accepted:
        rp = epoch_to_r_proxy(ep, fs_native, p95_global)
        f_rp, p_rp = compute_epoch_psd(rp, FS_SIM)
        psds_rp.append(p_rp)
    f_rp_mean = f_rp
    p_rp_mean = np.mean(psds_rp, axis=0)
    t4f_rp, t4q_rp = t4_from_psd(f_rp_mean, p_rp_mean)

    # Individual epoch PSDs for spaghetti (r_proxy, global p95)
    psds_indiv_f = f_rp
    psds_indiv_p = np.array(psds_rp)

    print("\nT4 metrics (SO band 0.2–1.5 Hz):")
    print(f"  Long-concat r_proxy Welch:     T4_freq={t4f_c:.3f} Hz  T4_q={t4q_c:.3f}")
    print(f"  Mean per-epoch r_proxy PSD:    T4_freq={t4f_rp:.3f} Hz  T4_q={t4q_rp:.3f}")
    print(f"  Mean per-epoch raw EEG PSD:    T4_freq={t4f_raw:.3f} Hz  T4_q={t4q_raw:.3f}  (V7 target)")

    # ── Figure 1: overview 0–20 Hz ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    n_show = min(40, len(psds_rp))
    idx = np.linspace(0, len(psds_rp) - 1, n_show, dtype=int)
    for j in idx:
        ax.semilogy(
            psds_indiv_f, psds_indiv_p[j], color="0.75", lw=0.6, alpha=0.45, zorder=1
        )
    ax.semilogy(f_concat, p_concat, color="#d95f02", lw=2.5,
                label=f"Long concat r_proxy ({duration_s:.0f}s Welch)", zorder=4)
    ax.semilogy(f_rp_mean, p_rp_mean, color="#1b9e77", lw=2.5,
                label="Mean of per-epoch r_proxy PSDs", zorder=4)
    ax.semilogy(f_raw_mean, p_raw_mean, color="#7570b3", lw=2.5, ls="--",
                label="Mean of per-epoch raw EEG PSDs (V7)", zorder=4)
    ax.set_xlim(0, 20)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [a.u.]")
    ax.set_title(f"SC4001 N3 — PSD comparison ({len(accepted)} epochs, "
                 f"{n_rej} rejected)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = OUT_DIR / "fig_concat_vs_mean_psd_overview.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved {p1}")

    # ── Figure 2: SO zoom + metrics ─────────────────────────────────────
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    for j in idx:
        ax.semilogy(psds_indiv_f, psds_indiv_p[j], color="0.85", lw=0.5, alpha=0.4)
    ax.semilogy(f_concat, p_concat, color="#d95f02", lw=2.5, label="Long concat r_proxy")
    ax.semilogy(f_rp_mean, p_rp_mean, color="#1b9e77", lw=2.5, label="Mean epoch r_proxy")
    ax.semilogy(f_raw_mean, p_raw_mean, color="#7570b3", lw=2.5, ls="--",
                label="Mean epoch raw EEG (V7)")
    ax.axvspan(SO_FREQ_LO, SO_FREQ_HI, color="gold", alpha=0.15, label="SO band")
    ax.set_xlim(0.05, 3.0)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [a.u.]")
    ax.set_title("SO band zoom (0.2–1.5 Hz) — where T4_q and T4_freq are read")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    rows = [
        ("Long concat r_proxy (v4 T4)", t4f_c, t4q_c),
        ("Mean per-epoch r_proxy PSD", t4f_rp, t4q_rp),
        ("Mean per-epoch raw EEG (V7 target)", t4f_raw, t4q_raw),
    ]
    table = ax2.table(
        cellText=[[r[0], f"{r[1]:.3f}", f"{r[2]:.3f}"] for r in rows],
        colLabels=["Method", "T4_freq [Hz]", "T4_q"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    p2 = OUT_DIR / "fig_concat_vs_mean_psd_so_zoom.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p2}")

    # ── Figure 3: timeline schematic ────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), gridspec_kw={"height_ratios": [1, 1.2]})

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("Two ways to summarize the same N3 nights", fontsize=11)

    y_mean, y_concat = 2.0, 0.7
    n_ep_draw = 8
    w_ep = 0.85
    for k in range(n_ep_draw):
        rect = mpatches.FancyBboxPatch(
            (k * w_ep, y_mean - 0.25), w_ep * 0.92, 0.5,
            boxstyle="round,pad=0.02", fc="#c6dbef", ec="#3182bd", lw=1
        )
        ax.add_patch(rect)
        ax.text(k * w_ep + w_ep * 0.46, y_mean, f"E{k+1}", ha="center", va="center", fontsize=7)
    ax.annotate("", xy=(7.5, y_mean + 0.55), xytext=(0, y_mean + 0.55),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(3.8, y_mean + 0.75, "Welch each 30s epoch → PSDᵢ  →  meanᵢ PSD",
            fontsize=9, ha="center", color="#1b9e77")

    big = mpatches.FancyBboxPatch(
        (0, y_concat - 0.3), n_ep_draw * w_ep - 0.1, 0.6,
        boxstyle="round,pad=0.02", fc="#fdd0a2", ec="#d95f02", lw=1.5
    )
    ax.add_patch(big)
    ax.text(n_ep_draw * w_ep / 2 - 0.05, y_concat,
            f"concatenate {len(accepted)} epochs ({duration_s:.0f}s) → one r_proxy",
            ha="center", va="center", fontsize=9, color="#d95f02")
    ax.text(n_ep_draw * w_ep / 2 - 0.05, y_concat - 0.55,
            "→ single Welch PSD (v4 compute_xobs T4)",
            ha="center", va="center", fontsize=8, color="#d95f02")

    ax = axes[1]
    t_show = 25.0
    n_samp = int(t_show * FS_SIM)
    t = np.arange(n_samp) / FS_SIM
    ax.plot(t, r_proxy_cat[:n_samp], color="#d95f02", lw=0.8, alpha=0.9)
    ax.set_xlim(0, t_show)
    ax.set_xlabel("Time [s] (first 25 s of concatenated r_proxy)")
    ax.set_ylabel("r_proxy [0–60]")
    ax.set_title("Concatenated signal keeps order: events & phase matter for T6/T8/MI")
    ax.grid(True, alpha=0.3)

    p3 = OUT_DIR / "fig_concat_vs_mean_psd_timeline.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"Saved {p3}")


if __name__ == "__main__":
    main()
