"""
diagnose_v7_phase.py
====================
Diagnose why V7's best solution shows preferred_phase = 2.78 rad (≈159°).

Question: Is spindle amplitude really peaking near SO DOWN trough,
          or is this a phase-convention bug?

Strategy: 4-layer cross-validation on the SAME re-simulated signal.

  Layer 1 — SANITY CHECK on the phase convention itself:
            Build a known signal (pure cosine), pass it through the same
            bandpass + Hilbert pipeline, confirm phase=0 occurs at peaks.

  Layer 2 — RE-SIMULATE V7's best solution and recompute PAC, confirm
            we reproduce the JSON value (≈2.78 rad). If we cannot
            reproduce, the bug is in v7's pipeline upstream of PAC.

  Layer 3 — VISUAL inspection: plot r_ctx, SO-filtered, SO-phase,
            r_thal, spindle envelope on a shared time axis.
            Hand-mark spindle-burst centers. Read off their SO-phase.

  Layer 4 — KL MI INTROSPECTION: plot the 18-bin phase-binned mean
            amplitude (Tort 2010's canonical visualization). The bin
            with the largest amplitude tells us where spindle power
            concentrates, independent of the MVL angle calculation.

If layers 2, 3, 4 all converge on phase ≈ 159°  → it's REAL (spindle
   really does prefer SO DOWN trough). The model needs a phase fix.
If layer 1 fails or layers 2-4 disagree → it's a bug. We localize.

Usage (from project root):
    python diagnose_v7_phase.py

Saves:
    outputs/v7_phase_diagnosis_layer1.png   (cosine sanity check)
    outputs/v7_phase_diagnosis_layer2.txt   (PAC reproduction)
    outputs/v7_phase_diagnosis_layer3.png   (time-domain inspection)
    outputs/v7_phase_diagnosis_layer4.png   (phase-amplitude histogram)
    outputs/v7_phase_diagnosis_summary.txt  (final verdict)
"""

import os
import sys
import json
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from scipy.signal import butter, sosfiltfilt, hilbert, correlate

# numpy 1.20+ removed legacy aliases; v7 patches them up
for _attr in ("object", "bool", "int", "float", "complex", "str"):
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(__builtins__, _attr, object))

# ====================================================================
# Configuration — must match V7 exactly
# ====================================================================
FS_SIM = 1000.0
SIM_DUR_MS = 60_000

SO_FREQ_LO, SO_FREQ_HI = 0.5, 1.5
SPINDLE_LO, SPINDLE_HI = 10.0, 14.0
PAC_N_BINS = 18
PAC_MAX_LAG_S = 2.0

# V7's best feasible solution (score=0.7127, eval 1754)
V7_BEST = dict(
    mue=3.974470889224735,
    mui=3.0836402149600537,
    b=35.03834862856099,
    tauA=1777.4575077009965,
    g_LK=0.05416706933735575,
    g_h=0.06835808008530264,
    c_th2ctx=0.01057274126094118,
    c_ctx2th=0.13317940596325312,
)
V7_REPORTED_PHASE = 2.782   # rad, what JSON says we should reproduce
V7_REPORTED_MI = 0.00714
V7_REPORTED_LAG_MS = 458.0

OUTDIR = Path("outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ====================================================================
# Replicated PAC function (verbatim from V7)
# ====================================================================
def compute_pac_metrics(r_ctx, r_thal, fs=FS_SIM):
    """Verbatim copy of V7's compute_pac_metrics. Diagnostic must use SAME code."""
    out = {"mi": 0.0, "preferred_phase": np.pi, "lag_samples": 0,
           "lag_ms": 0.0, "ok": False,
           "so_filt": None, "so_phase": None, "sp_filt": None, "sp_amp": None,
           "mean_amp_per_bin": None, "bin_centers": None}
    if len(r_ctx) < int(2 * fs) or len(r_thal) < int(2 * fs):
        return out
    if r_ctx.std() < 1e-6 or r_thal.std() < 1e-6:
        return out

    sos_so = butter(4, [SO_FREQ_LO, SO_FREQ_HI], btype="band", fs=fs, output="sos")
    so_filt = sosfiltfilt(sos_so, r_ctx)
    so_analytic = hilbert(so_filt)
    so_phase = np.angle(so_analytic)

    sos_sp = butter(4, [SPINDLE_LO, SPINDLE_HI], btype="band", fs=fs, output="sos")
    sp_filt = sosfiltfilt(sos_sp, r_thal)
    sp_amp = np.abs(hilbert(sp_filt))

    edge = int(0.5 * fs)
    so_filt = so_filt[edge:-edge]
    so_phase = so_phase[edge:-edge]
    sp_filt = sp_filt[edge:-edge]
    sp_amp = sp_amp[edge:-edge]

    bin_edges = np.linspace(-np.pi, np.pi, PAC_N_BINS + 1)
    mean_amp = np.zeros(PAC_N_BINS)
    for i in range(PAC_N_BINS):
        mask = (so_phase >= bin_edges[i]) & (so_phase < bin_edges[i + 1])
        if mask.any():
            mean_amp[i] = sp_amp[mask].mean()
    total = mean_amp.sum()
    if total <= 0 or not np.isfinite(total):
        return out
    p = mean_amp / total
    p_safe = np.where(p > 0, p, 1.0)
    H = -np.sum(p * np.log(p_safe))
    mi = (np.log(PAC_N_BINS) - H) / np.log(PAC_N_BINS)
    mi = float(np.clip(mi, 0.0, 1.0))

    mvl = (sp_amp * np.exp(1j * so_phase)).mean()
    preferred_phase = float(np.angle(mvl))

    max_lag = int(PAC_MAX_LAG_S * fs)
    a = sp_amp - sp_amp.mean()
    b = so_filt - so_filt.mean()
    xc = correlate(a, b, mode="full")
    lags = np.arange(-(len(b) - 1), len(a))
    keep = (lags >= -max_lag) & (lags <= max_lag)
    xc_w = xc[keep]
    lags_w = lags[keep]
    peak_lag_samples = int(lags_w[np.argmax(xc_w)])

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    out.update({
        "mi": mi, "preferred_phase": preferred_phase,
        "lag_samples": peak_lag_samples,
        "lag_ms": peak_lag_samples / fs * 1000.0,
        "ok": True,
        "so_filt": so_filt, "so_phase": so_phase,
        "sp_filt": sp_filt, "sp_amp": sp_amp,
        "mean_amp_per_bin": mean_amp, "bin_centers": bin_centers,
    })
    return out


# ====================================================================
# LAYER 1 — Phase convention sanity check on a known signal
# ====================================================================
def layer1_sanity_check():
    """
    Build a pure cosine SO + amplitude-modulated spindle that we KNOW
    peaks at the SO UP peak. Run through the same pipeline. Verify
    preferred_phase ≈ 0.

    This tests *only* the PAC pipeline — bandpass, Hilbert, MVL angle.
    If this FAILS, the bug is in the pipeline itself, not in v7's
    simulation output.
    """
    print("\n=== LAYER 1: phase-convention sanity check ===")
    fs = FS_SIM
    T = 60.0  # seconds
    t = np.arange(int(T * fs)) / fs

    # SO at 1 Hz, cosine (peaks at t=0, 1, 2, ...)
    so_freq = 1.0
    so = np.cos(2 * np.pi * so_freq * t)

    # Spindle: 12 Hz oscillation amplitude-modulated by (1 + cos(2π·1·t))/2
    # → amplitude maximal at SO peak
    sp_freq = 12.0
    sp_carrier = np.cos(2 * np.pi * sp_freq * t)
    am_envelope = (1 + np.cos(2 * np.pi * so_freq * t)) / 2  # in [0,1]
    sp = am_envelope * sp_carrier

    # Inject as if r_ctx = SO-like, r_thal = spindle-like (no thalamic spike train)
    # Add tiny noise so std > 1e-6
    rng = np.random.default_rng(42)
    r_ctx = so + 0.01 * rng.standard_normal(len(t))
    r_thal = sp + 0.01 * rng.standard_normal(len(t))

    pac = compute_pac_metrics(r_ctx, r_thal, fs=fs)

    phase = pac["preferred_phase"]
    phase_deg = np.degrees(phase)
    mi = pac["mi"]
    lag_ms = pac["lag_ms"]

    print(f"  Synthetic ground-truth phase: 0.0 rad (0°)  [spindle peaks at SO peak]")
    print(f"  Pipeline reports preferred_phase: {phase:.3f} rad ({phase_deg:.1f}°)")
    print(f"  MI: {mi:.4f}")
    print(f"  Lag: {lag_ms:.1f} ms")
    if abs(phase) < 0.3:  # within ~17°
        verdict = "PASS — convention is correct (0 = UP peak)"
    elif abs(abs(phase) - np.pi) < 0.3:
        verdict = "FAIL — convention is INVERTED (π = UP peak in this pipeline!)"
    else:
        verdict = "AMBIGUOUS — phase neither 0 nor ±π; investigate further"
    print(f"  VERDICT: {verdict}")

    # Plot: synthetic signal + recovered phase mapping
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    n_show = int(5 * fs)
    tt = t[pac["so_phase"].size:][:0]  # just for sizing

    edge = int(0.5 * fs)
    show_t = (np.arange(pac["so_filt"].size) + edge) / fs
    show_idx = show_t < 5.0

    axes[0].plot(show_t[show_idx], pac["so_filt"][show_idx], "b", lw=1.5, label="SO-filtered r_ctx")
    axes[0].plot(t[:n_show], so[:n_show], "k--", lw=0.7, alpha=0.5, label="ground-truth SO (cosine)")
    axes[0].set_ylabel("SO")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title(f"LAYER 1 — sanity check: synthetic signal where spindle ⇿ SO peak.\n"
                      f"Recovered preferred_phase = {phase:.3f} rad ({phase_deg:.1f}°), "
                      f"expected ≈ 0°.  {verdict}")

    axes[1].plot(show_t[show_idx], pac["so_phase"][show_idx], "g", lw=1)
    axes[1].axhline(0, color="r", ls="--", lw=0.5)
    axes[1].set_ylabel("SO phase [rad]")
    axes[1].set_ylim([-np.pi - 0.2, np.pi + 0.2])

    axes[2].plot(show_t[show_idx], pac["sp_amp"][show_idx], "orange", lw=1.5, label="spindle envelope")
    axes[2].set_ylabel("Spindle amp")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTDIR / "v7_phase_diagnosis_layer1.png", dpi=130)
    plt.close()
    print(f"  Saved: {OUTDIR/'v7_phase_diagnosis_layer1.png'}")

    return {"phase": phase, "phase_deg": phase_deg, "mi": mi, "lag_ms": lag_ms, "verdict": verdict}


# ====================================================================
# LAYER 2 — Reproduce V7's PAC by re-simulating its best solution
# ====================================================================
def layer2_reproduce_pac():
    """
    Re-simulate V7's best solution and recompute PAC. Should give
    preferred_phase ≈ 2.78 rad if pipeline is deterministic.

    REQUIRES neurolib. Imports are local so layers 1 & 4 can run
    even without neurolib.
    """
    print("\n=== LAYER 2: reproduce V7's best PAC ===")

    try:
        from models.s4_personalize_fig7_v7 import build_model, EXC
    except ImportError:
        print("  SKIP: v7 model module or neurolib not installed in this environment.")
        return None

    # Build the exact v7 model rather than recreating the motif locally.
    p = V7_BEST
    model = build_model(
        p["mue"], p["mui"], p["b"], p["tauA"],
        p["g_LK"], p["g_h"], p["c_th2ctx"], p["c_ctx2th"],
        duration=SIM_DUR_MS,
    )

    print(f"  Re-simulating V7's best parameters (duration={SIM_DUR_MS} ms)...")
    try:
        model.run()
    except Exception as exc:
        print(f"  numba backend failed ({exc}); switching to numpy")
        model.params["backend"] = "numpy"
        model.run()

    r_exc_raw = model[f"r_mean_{EXC}"]
    if r_exc_raw.ndim == 2 and r_exc_raw.shape[0] >= 2:
        r_ctx = r_exc_raw[0, :] * 1000.0
        r_thal = r_exc_raw[1, :] * 1000.0
    else:
        r_ctx = (r_exc_raw[0] if r_exc_raw.ndim == 2 else r_exc_raw) * 1000.0
        r_thal = np.zeros_like(r_ctx)

    pac = compute_pac_metrics(r_ctx, r_thal, fs=FS_SIM)
    print(f"  Re-computed MI:               {pac['mi']:.5f}   (V7 reported: {V7_REPORTED_MI:.5f})")
    print(f"  Re-computed preferred_phase:  {pac['preferred_phase']:.3f} rad "
          f"({np.degrees(pac['preferred_phase']):.1f}°)   "
          f"(V7 reported: {V7_REPORTED_PHASE:.3f} rad / {np.degrees(V7_REPORTED_PHASE):.1f}°)")
    print(f"  Re-computed lag:              {pac['lag_ms']:.1f} ms   (V7 reported: {V7_REPORTED_LAG_MS:.1f} ms)")

    diff_phase = abs(pac["preferred_phase"] - V7_REPORTED_PHASE)
    diff_phase = min(diff_phase, 2 * np.pi - diff_phase)
    if diff_phase < 0.5:
        verdict = "REPRODUCED — V7's reported phase is real, not a stale-record bug"
    else:
        verdict = "MISMATCH — V7's reported phase does not reproduce; investigate"
    print(f"  VERDICT: {verdict}")

    # Save signals for layer 3 & 4
    np.savez(OUTDIR / "v7_phase_diagnosis_signals.npz",
             r_ctx=r_ctx, r_thal=r_thal,
             so_filt=pac["so_filt"], so_phase=pac["so_phase"],
             sp_filt=pac["sp_filt"], sp_amp=pac["sp_amp"],
             mean_amp_per_bin=pac["mean_amp_per_bin"],
             bin_centers=pac["bin_centers"])
    print(f"  Saved signals: {OUTDIR/'v7_phase_diagnosis_signals.npz'}")

    return {"pac": pac, "r_ctx": r_ctx, "r_thal": r_thal, "verdict": verdict}


# ====================================================================
# LAYER 3 — Time-domain inspection: hand-mark spindles, read SO phase
# ====================================================================
def layer3_visual_inspection(layer2_result):
    """
    Plot r_ctx, SO-filtered r_ctx, SO-phase, r_thal, spindle envelope
    on a shared 8 s window. Mark detected spindle envelope peaks and
    annotate the SO phase at each peak.
    """
    print("\n=== LAYER 3: visual time-domain inspection ===")

    if layer2_result is None:
        print("  SKIP: layer 2 did not produce signals.")
        return None

    pac = layer2_result["pac"]
    r_ctx = layer2_result["r_ctx"]
    r_thal = layer2_result["r_thal"]
    fs = FS_SIM

    # Use the post-edge-trimmed signals returned by compute_pac_metrics
    edge = int(0.5 * fs)
    t_full = (np.arange(pac["so_filt"].size) + edge) / fs

    # Inspection window: 8 to 24 s (matches v7 timeseries plot)
    t0, t1 = 8.0, 24.0
    idx = (t_full >= t0) & (t_full <= t1)
    t_show = t_full[idx]

    so_filt_show = pac["so_filt"][idx]
    so_phase_show = pac["so_phase"][idx]
    sp_amp_show = pac["sp_amp"][idx]
    sp_filt_show = pac["sp_filt"][idx]
    r_ctx_show = r_ctx[edge + np.where(idx)[0][0] : edge + np.where(idx)[0][-1] + 1]
    r_thal_show = r_thal[edge + np.where(idx)[0][0] : edge + np.where(idx)[0][-1] + 1]

    # Find top-N spindle envelope peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(sp_amp_show, distance=int(0.5 * fs),
                          height=np.percentile(sp_amp_show, 80))
    # Phase at each peak
    peak_phases = so_phase_show[peaks]
    peak_phases_deg = np.degrees(peak_phases)

    print(f"  Detected {len(peaks)} top-quintile spindle envelope peaks in window.")
    print(f"  SO-phase at peaks (degrees): {peak_phases_deg.round(1)}")
    print(f"  Mean phase at peaks: {np.degrees(np.angle(np.exp(1j * peak_phases).mean())):.1f}°")

    fig, axes = plt.subplots(5, 1, figsize=(15, 11), sharex=True)

    # 1. Raw cortex
    axes[0].plot(t_show, r_ctx_show, "indigo", lw=1, label="r_ctx (raw cortex EXC)")
    axes[0].set_ylabel("r_ctx [Hz]")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title(f"V7 best solution ({V7_REPORTED_PHASE:.2f} rad reported) — "
                      f"time-domain inspection.\n"
                      f"Mean phase at top spindle-envelope peaks: "
                      f"{np.degrees(np.angle(np.exp(1j * peak_phases).mean())):.1f}° "
                      f"(MVL says {np.degrees(pac['preferred_phase']):.1f}°)")

    # 2. SO-filtered with phase markers
    axes[1].plot(t_show, so_filt_show, "blue", lw=1.2, label="SO-filtered r_ctx")
    axes[1].axhline(0, color="gray", lw=0.4, alpha=0.6)
    # Mark UP peaks of SO (zero crossings of phase, downward)
    for pk in peaks:
        axes[1].axvline(t_show[pk], color="orange", lw=0.7, alpha=0.6, ls="--")
    axes[1].set_ylabel("SO [a.u.]")
    axes[1].legend(loc="upper right", fontsize=9)

    # 3. SO phase track
    axes[2].plot(t_show, so_phase_show, "darkgreen", lw=1)
    for pk, ph in zip(peaks, peak_phases_deg):
        axes[2].plot(t_show[pk], so_phase_show[pk], "ro", markersize=6)
        axes[2].annotate(f"{ph:.0f}°", (t_show[pk], so_phase_show[pk]),
                         textcoords="offset points", xytext=(0, 8),
                         fontsize=8, color="red", ha="center")
    axes[2].axhline(0, color="black", ls=":", lw=0.6, label="phase=0 (= SO UP peak by convention)")
    axes[2].axhline(np.pi, color="red", ls=":", lw=0.6, label="phase=±π (= SO trough)")
    axes[2].axhline(-np.pi, color="red", ls=":", lw=0.6)
    axes[2].set_ylabel("SO phase [rad]")
    axes[2].set_ylim([-np.pi - 0.3, np.pi + 0.3])
    axes[2].legend(loc="upper right", fontsize=8)

    # 4. Raw thalamus
    axes[3].plot(t_show, r_thal_show, "teal", lw=0.8, label="r_thal (raw TCR)")
    axes[3].set_ylabel("r_thal [Hz]")
    axes[3].legend(loc="upper right", fontsize=9)

    # 5. Spindle envelope with peaks
    axes[4].plot(t_show, sp_amp_show, "orange", lw=1.3, label="spindle envelope")
    axes[4].plot(t_show[peaks], sp_amp_show[peaks], "rv", markersize=8, label="detected peaks")
    axes[4].set_ylabel("Spindle amp")
    axes[4].set_xlabel("Time [s]")
    axes[4].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTDIR / "v7_phase_diagnosis_layer3.png", dpi=130)
    plt.close()
    print(f"  Saved: {OUTDIR/'v7_phase_diagnosis_layer3.png'}")

    return {"peak_phases_deg": peak_phases_deg.tolist(),
            "mean_peak_phase_deg": float(np.degrees(np.angle(np.exp(1j * peak_phases).mean())))}


# ====================================================================
# LAYER 4 — KL MI introspection: phase-binned amplitude histogram
# ====================================================================
def layer4_phase_amp_hist(layer2_result):
    """
    Plot the 18-bin phase-binned mean spindle amplitude. The bin with
    largest amplitude is where coupling concentrates. This is
    independent of MVL angle (which sums vectors).
    """
    print("\n=== LAYER 4: phase-binned amplitude histogram ===")

    if layer2_result is None:
        # Try loading saved signals
        path = OUTDIR / "v7_phase_diagnosis_signals.npz"
        if path.exists():
            d = np.load(path)
            mean_amp = d["mean_amp_per_bin"]
            bin_centers = d["bin_centers"]
            phase = float(np.angle((mean_amp * np.exp(1j * bin_centers)).sum()))
        else:
            print("  SKIP: no signals available.")
            return None
    else:
        mean_amp = layer2_result["pac"]["mean_amp_per_bin"]
        bin_centers = layer2_result["pac"]["bin_centers"]
        phase = layer2_result["pac"]["preferred_phase"]

    bin_centers_deg = np.degrees(bin_centers)
    peak_bin_idx = int(np.argmax(mean_amp))
    peak_bin_phase_deg = bin_centers_deg[peak_bin_idx]

    print(f"  Histogram peak bin: index {peak_bin_idx}, "
          f"center phase {peak_bin_phase_deg:.1f}°, "
          f"amplitude {mean_amp[peak_bin_idx]:.3f}")
    print(f"  MVL preferred phase: {np.degrees(phase):.1f}°")

    discrepancy = abs(peak_bin_phase_deg - np.degrees(phase))
    discrepancy = min(discrepancy, 360 - discrepancy)
    if discrepancy < 30:
        verdict = "CONSISTENT — histogram peak agrees with MVL angle"
    else:
        verdict = (f"DISAGREEMENT ({discrepancy:.0f}° apart) — "
                   "MVL angle may be biased by amplitude distribution; "
                   "histogram is more reliable")
    print(f"  VERDICT: {verdict}")

    # Plot bar chart + MVL marker
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(bin_centers_deg, mean_amp, width=20,
           color="steelblue", edgecolor="navy", alpha=0.7)
    ax.axvline(np.degrees(phase), color="red", lw=2,
               label=f"MVL preferred phase = {np.degrees(phase):.1f}°")
    ax.axvline(0, color="green", lw=2, ls="--",
               label="SO UP peak (= 0° by convention)")
    ax.axvline(180, color="orange", lw=2, ls="--",
               label="SO DOWN trough (= ±180°)")
    ax.axvline(-180, color="orange", lw=2, ls="--")
    ax.bar(bin_centers_deg[peak_bin_idx], mean_amp[peak_bin_idx],
           width=20, color="red", edgecolor="darkred", alpha=0.9, zorder=3,
           label=f"histogram peak bin = {peak_bin_phase_deg:.0f}°")
    ax.set_xlabel("SO phase bin [deg]  (0 = UP peak, ±180 = DOWN trough)")
    ax.set_ylabel("Mean spindle envelope amplitude")
    ax.set_title(f"LAYER 4 — phase-binned spindle amplitude (Tort 2010 view).  {verdict}")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xticks(np.arange(-180, 181, 45))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTDIR / "v7_phase_diagnosis_layer4.png", dpi=130)
    plt.close()
    print(f"  Saved: {OUTDIR/'v7_phase_diagnosis_layer4.png'}")

    return {"peak_bin_phase_deg": float(peak_bin_phase_deg),
            "mvl_phase_deg": float(np.degrees(phase)),
            "verdict": verdict}


# ====================================================================
# Final synthesis
# ====================================================================
def synthesize(l1, l2, l3, l4):
    print("\n" + "=" * 72)
    print("FINAL SYNTHESIS")
    print("=" * 72)

    lines = []
    lines.append(f"Layer 1 (sanity):        {l1['verdict']}")
    if l2 is None:
        lines.append("Layer 2 (reproduce):     SKIPPED (neurolib unavailable)")
    else:
        lines.append(f"Layer 2 (reproduce):     {l2['verdict']}")
    if l3 is None:
        lines.append("Layer 3 (visual):        SKIPPED")
    else:
        lines.append(f"Layer 3 (visual):        mean peak-phase = {l3['mean_peak_phase_deg']:.1f}°")
    if l4 is None:
        lines.append("Layer 4 (histogram):     SKIPPED")
    else:
        lines.append(f"Layer 4 (histogram):     {l4['verdict']}; "
                     f"peak bin = {l4['peak_bin_phase_deg']:.0f}°, "
                     f"MVL = {l4['mvl_phase_deg']:.1f}°")

    print("\n".join(lines))

    # Decision rule
    print("\nDECISION:")
    if "FAIL" in l1["verdict"] or "INVERTED" in l1["verdict"]:
        print("  → Layer 1 failed. The PAC pipeline ITSELF has a phase-convention bug.")
        print("    Fix the convention before trusting any v7 phase result.")
        print("    Likely cause: Hilbert phase definition or MVL angle sign.")
    elif l2 and "MISMATCH" in l2["verdict"]:
        print("  → Layer 2 mismatch. V7's stored result does not reproduce.")
        print("    Possible causes: stochastic seeding, neurolib version drift,")
        print("    parameter mapping in build_network differs.")
    elif l3 and l4 and abs(l3["mean_peak_phase_deg"] - l4["peak_bin_phase_deg"]) < 30:
        if abs(l3["mean_peak_phase_deg"]) < 50:
            print("  → Layers 3 & 4 agree, AND show phase near 0° (UP peak).")
            print("    V7's stored 159° was a stale or buggy value. The actual model")
            print("    is well-coupled to the UP peak. Proceed to control with confidence.")
        elif abs(abs(l3["mean_peak_phase_deg"]) - 180) < 50:
            print("  → Layers 3 & 4 agree on phase near ±180° (DOWN trough).")
            print("    V7's 159° is REAL: spindles really nest at SO trough, not peak.")
            print("    This is biologically interpretable (slow-frontal regime, Mölle 2011)")
            print("    BUT it is not the Helfrich healthy-young benchmark.")
            print("    Decision: either embrace as 'pathological-like baseline' (Route A),")
            print("    or add a T13 phase constraint and re-run (Route B).")
        else:
            print(f"  → Spindles nest at an intermediate phase. Investigate further.")
    else:
        print("  → Layers disagree. Need finer-grained analysis.")

    with open(OUTDIR / "v7_phase_diagnosis_summary.txt", "w") as f:
        f.write("V7 phase=159° diagnosis summary\n")
        f.write("=" * 72 + "\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary saved: {OUTDIR/'v7_phase_diagnosis_summary.txt'}")


# ====================================================================
if __name__ == "__main__":
    print("V7 phase diagnosis — 4-layer cross-validation")
    print("=" * 72)

    l1 = layer1_sanity_check()
    l2 = layer2_reproduce_pac()
    l3 = layer3_visual_inspection(l2) if l2 else None
    l4 = layer4_phase_amp_hist(l2)

    synthesize(l1, l2, l3, l4)
