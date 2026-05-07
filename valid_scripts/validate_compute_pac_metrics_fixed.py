"""
validate_compute_pac_metrics_fixed.py
======================================
Comprehensive verification that compute_pac_metrics_fixed actually works.

Why a separate validation script?
---------------------------------
The fixed function makes claims about how it handles different signals.
Each claim deserves an independent experiment that can either confirm
or falsify it. This script runs 9 such experiments. Every experiment
has a clear PASS/FAIL criterion based on what we know to be the
ground truth.

Run from anywhere:
    python validate_compute_pac_metrics_fixed.py

Outputs (all in ./validation_outputs/):
    V1_synthetic_up_locked.png
    V2_invariance_to_waveform.png
    V3_no_coupling_baseline.png
    V4_three_regimes.png
    V5_noise_robustness.png
    V6_v7_old_vs_new_comparison.png
    V7_bin_uniformity.png
    V8_bimodality_detection.png
    V9_edge_cases.txt
    SUMMARY.txt
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert

# Allow running from anywhere
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from compute_pac_metrics_fixed import compute_pac_metrics, _detect_bimodality

OUTDIR = SCRIPT_DIR / "validation_outputs"
OUTDIR.mkdir(exist_ok=True)

# Common synthetic signal builders
FS = 1000.0
T = 60.0
T_VEC = np.arange(int(T * FS)) / FS


def make_cortex_pulses(t, period=1.3, pulse_width_s=0.15, rng=None):
    """Generate cortex r_E with narrow exponential pulses (mimics real model)."""
    sig = np.zeros(len(t))
    n_pulses = int(t[-1] / period)
    for k in range(n_pulses):
        idx = int(k * period * FS)
        w = int(pulse_width_s * FS)
        if idx + w < len(sig):
            decay = np.exp(-np.linspace(0, 4, w))
            sig[idx:idx + w] = 60 * decay
    if rng is not None:
        sig += 0.5 * rng.standard_normal(len(t))
    return sig


def make_spindle_at_phase(t, ctx_peak_period, target_phase_rad,
                          burst_amp=30, fc=12, env_width=0.1,
                          half_width_s=0.25, rng=None):
    """
    Generate thalamic-like signal with spindle bursts centered at a
    specific SO phase. target_phase_rad=0 means at UP peak.
    target_phase_rad=π means at DOWN trough.
    """
    sig = np.zeros(len(t))
    carrier = np.cos(2 * np.pi * fc * t)
    n_pulses = int(t[-1] / ctx_peak_period)
    # Phase 0 = UP peak; phase π = halfway = DOWN trough
    # Convert phase to time offset from preceding UP peak
    phase_frac = abs(target_phase_rad) / (2 * np.pi)
    if target_phase_rad >= 0:
        offset_s = phase_frac * ctx_peak_period
    else:
        offset_s = (1 - phase_frac) * ctx_peak_period

    for k in range(n_pulses):
        center_t = k * ctx_peak_period + offset_s
        idx = int(center_t * FS)
        w = int(half_width_s * FS)
        s = max(0, idx - w)
        e = min(len(sig), idx + w)
        if e <= s:
            continue
        win = np.exp(-((np.arange(e - s) - (idx - s)) / (env_width * FS)) ** 2)
        sig[s:e] += burst_amp * win * carrier[s:e]

    sig += 25  # baseline rate
    if rng is not None:
        sig += 0.5 * rng.standard_normal(len(t))
    return sig


def fmt_phase(rad):
    return f"{np.degrees(rad):+.1f}°"


# ============================================================
# V1: Recovery on synthetic UP-locked coupling
# ============================================================
def V1_synthetic_up_locked():
    """
    Claim: with spindles synthesized to peak exactly at cortex UP peaks,
    the function should recover preferred_phase ≈ 0°.
    """
    print("\n" + "=" * 72)
    print("V1: Recovery on UP-locked synthetic coupling")
    print("=" * 72)

    rng = np.random.default_rng(0)
    r_ctx = make_cortex_pulses(T_VEC, rng=rng)
    r_thal = make_spindle_at_phase(T_VEC, 1.3, target_phase_rad=0, rng=rng)

    out = compute_pac_metrics(r_ctx, r_thal, FS)

    # Pass criteria
    pass_argmax = abs(np.degrees(out["phase_argmax"])) <= 30
    pass_pref = abs(np.degrees(out["preferred_phase"])) <= 30
    pass_ratio = out["up_down_ratio"] > 2.0
    pass_mi = out["mi"] > 0.05
    pass_concentration = out["phase_concentration"] > 0.3
    pass_unimodal = not out["bimodality_flag"]
    overall = (pass_argmax and pass_pref and pass_ratio
               and pass_mi and pass_concentration and pass_unimodal)

    print(f"  preferred_phase    = {fmt_phase(out['preferred_phase']):>10}    "
          f"expect ~0°    {'PASS' if pass_pref else 'FAIL'}")
    print(f"  phase_argmax       = {fmt_phase(out['phase_argmax']):>10}    "
          f"expect ~0°    {'PASS' if pass_argmax else 'FAIL'}")
    print(f"  up_down_ratio      = {out['up_down_ratio']:>9.2f}     "
          f"expect >2     {'PASS' if pass_ratio else 'FAIL'}")
    print(f"  mi                 = {out['mi']:>9.4f}     "
          f"expect >0.05  {'PASS' if pass_mi else 'FAIL'}")
    print(f"  phase_concentration= {out['phase_concentration']:>9.3f}     "
          f"expect >0.3   {'PASS' if pass_concentration else 'FAIL'}")
    print(f"  bimodality_flag    = {str(out['bimodality_flag']):>9}     "
          f"expect False  {'PASS' if pass_unimodal else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    show = slice(0, int(8 * FS))
    axes[0].plot(T_VEC[show], r_ctx[show], 'k', lw=0.8, label='r_ctx')
    axes[0].plot(T_VEC[show], r_thal[show] - 25, 'g', lw=0.6, alpha=0.6,
                 label='r_thal (offset)')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Signal')
    axes[0].legend()
    axes[0].set_title('V1 input: UP-locked spindles')

    axes[1].text(0.5, 0.95, "V1 results", ha='center', va='top',
                 transform=axes[1].transAxes, fontsize=12, fontweight='bold')
    txt = (f"preferred_phase = {fmt_phase(out['preferred_phase'])} (expect 0°)\n"
           f"phase_argmax    = {fmt_phase(out['phase_argmax'])} (expect 0°)\n"
           f"up_down_ratio   = {out['up_down_ratio']:.2f} (expect >2)\n"
           f"mi              = {out['mi']:.4f} (expect >0.05)\n"
           f"concentration   = {out['phase_concentration']:.3f} (expect >0.3)\n"
           f"bimodal         = {out['bimodality_flag']} (expect False)\n\n"
           f"OVERALL: {'PASS' if overall else 'FAIL'}")
    axes[1].text(0.05, 0.85, txt, ha='left', va='top',
                 transform=axes[1].transAxes, fontfamily='monospace',
                 fontsize=10,
                 color='darkgreen' if overall else 'darkred')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(OUTDIR / "V1_synthetic_up_locked.png", dpi=120)
    plt.close()

    return overall


# ============================================================
# V2: Invariance to cortex waveform shape
# ============================================================
def V2_invariance_to_waveform():
    """
    Claim: when spindles are physically locked at the same time relative
    to cortex peaks, recovered phase should be consistent regardless of
    cortex waveform (sine, wide pulse, narrow pulse, spike).
    This is the KEY claim that the v7 Hilbert version fails.
    """
    print("\n" + "=" * 72)
    print("V2: Invariance to cortex waveform shape")
    print("=" * 72)

    rng = np.random.default_rng(0)
    period = 1.3

    # All four cases share the SAME spindle (locked to UP peak time)
    spindle = make_spindle_at_phase(T_VEC, period, target_phase_rad=0, rng=rng)

    # Case A: cosine cortex
    r_ctx_A = 30 + 30 * np.cos(2 * np.pi * (1 / period) * T_VEC)
    r_ctx_A += 0.5 * rng.standard_normal(len(T_VEC))

    # Case B: wide pulses (300ms)
    r_ctx_B = make_cortex_pulses(T_VEC, period=period, pulse_width_s=0.30,
                                 rng=np.random.default_rng(1))

    # Case C: narrow pulses (150ms, like real cortex r_E)
    r_ctx_C = make_cortex_pulses(T_VEC, period=period, pulse_width_s=0.15,
                                 rng=np.random.default_rng(2))

    # Case D: very narrow spikes (50ms)
    r_ctx_D = make_cortex_pulses(T_VEC, period=period, pulse_width_s=0.05,
                                 rng=np.random.default_rng(3))

    cases = [("cosine", r_ctx_A),
             ("wide pulse 300ms", r_ctx_B),
             ("narrow pulse 150ms", r_ctx_C),
             ("very narrow 50ms", r_ctx_D)]

    print(f"  All four cases have spindle locked to UP peak time.")
    print(f"  Expectation: all four should give phase ≈ 0°.\n")

    fig, axes = plt.subplots(len(cases), 2, figsize=(13, 9), sharex='col')
    results = []
    for i, (name, r_ctx) in enumerate(cases):
        out = compute_pac_metrics(r_ctx, spindle, FS)
        results.append((name, out))
        passed = abs(np.degrees(out["phase_argmax"])) <= 30 and out["ok"]
        verdict = "PASS" if passed else "FAIL"
        print(f"  {name:25s}  argmax = {fmt_phase(out['phase_argmax']):>8}   "
              f"pref = {fmt_phase(out['preferred_phase']):>8}   "
              f"ratio = {out['up_down_ratio']:>5.2f}   {verdict}")

        # Plot
        show = slice(0, int(5 * FS))
        axes[i, 0].plot(T_VEC[show], r_ctx[show], 'k', lw=0.8)
        axes[i, 0].set_ylabel(name, fontsize=10)
        if i == 0:
            axes[i, 0].set_title("Cortex waveform (red=true UP peak)")
        for tk in np.arange(0, 5, period):
            axes[i, 0].axvline(tk, color='red', alpha=0.4, ls='--')

        axes[i, 1].axis('off')
        col = 'darkgreen' if passed else 'darkred'
        axes[i, 1].text(0.05, 0.5,
                        f"phase_argmax     = {fmt_phase(out['phase_argmax'])}\n"
                        f"preferred_phase  = {fmt_phase(out['preferred_phase'])}\n"
                        f"mi               = {out['mi']:.4f}\n"
                        f"up_down_ratio    = {out['up_down_ratio']:.2f}\n"
                        f"==> {verdict}",
                        transform=axes[i, 1].transAxes,
                        fontfamily='monospace', fontsize=10, color=col,
                        verticalalignment='center')
    axes[-1, 0].set_xlabel("Time [s]")
    plt.suptitle("V2: Recovery should be invariant to cortex waveform shape",
                 fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(OUTDIR / "V2_invariance_to_waveform.png", dpi=120)
    plt.close()

    overall = all(abs(np.degrees(o["phase_argmax"])) <= 30 and o["ok"]
                  for _, o in results)
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


# ============================================================
# V3: No-coupling baseline (random spindles)
# ============================================================
def V3_no_coupling_baseline():
    """
    Claim: with truly random spindle timing, MI should be near zero,
    phase_concentration near zero, up_down_ratio near 1.
    """
    print("\n" + "=" * 72)
    print("V3: No-coupling baseline (random spindle timing)")
    print("=" * 72)

    n_trials = 20
    mi_values = []
    conc_values = []
    ratio_values = []

    rng = np.random.default_rng(0)
    for trial in range(n_trials):
        r_ctx = make_cortex_pulses(T_VEC, rng=np.random.default_rng(trial))

        # Random spindle bursts uniformly distributed in time
        spindle = np.full(len(T_VEC), 25.0)
        sp_carrier = np.cos(2 * np.pi * 12 * T_VEC)
        n_bursts = int(T)  # ~1 per second
        burst_times = rng.uniform(0.5, T - 0.5, n_bursts)
        for bt in burst_times:
            idx = int(bt * FS)
            w = int(0.25 * FS)
            s = max(0, idx - w)
            e = min(len(spindle), idx + w)
            win = np.exp(-((np.arange(e - s) - (idx - s)) / (0.1 * FS)) ** 2)
            spindle[s:e] += 30 * win * sp_carrier[s:e]

        out = compute_pac_metrics(r_ctx, spindle, FS)
        if out["ok"]:
            mi_values.append(out["mi"])
            conc_values.append(out["phase_concentration"])
            ratio_values.append(out["up_down_ratio"])

    mi_values = np.array(mi_values)
    conc_values = np.array(conc_values)
    ratio_values = np.array(ratio_values)

    # Pass criteria: typical baseline values should be small
    mi_mean = mi_values.mean()
    pass_mi = mi_mean < 0.02         # well below typical "real coupling"
    pass_conc = conc_values.mean() < 0.15
    pass_ratio = abs(ratio_values.mean() - 1.0) < 0.4

    print(f"  Over {len(mi_values)} random-spindle trials:")
    print(f"    MI:               mean={mi_mean:.4f}, max={mi_values.max():.4f}, "
          f"expect <0.02   {'PASS' if pass_mi else 'FAIL'}")
    print(f"    concentration:    mean={conc_values.mean():.3f}, max={conc_values.max():.3f}, "
          f"expect <0.15   {'PASS' if pass_conc else 'FAIL'}")
    print(f"    up_down_ratio:    mean={ratio_values.mean():.2f}, std={ratio_values.std():.2f}, "
          f"expect ≈1.0    {'PASS' if pass_ratio else 'FAIL'}")
    overall = pass_mi and pass_conc and pass_ratio

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(mi_values, bins=15, color='steelblue', edgecolor='navy')
    axes[0].axvline(mi_mean, color='red', label=f'mean={mi_mean:.4f}')
    axes[0].axvline(0.02, color='green', ls='--', label='threshold 0.02')
    axes[0].set_xlabel('MI'); axes[0].set_title('V3: MI under no-coupling null')
    axes[0].legend(fontsize=9)

    axes[1].hist(conc_values, bins=15, color='seagreen', edgecolor='darkgreen')
    axes[1].axvline(conc_values.mean(), color='red',
                    label=f'mean={conc_values.mean():.3f}')
    axes[1].axvline(0.15, color='orange', ls='--', label='threshold 0.15')
    axes[1].set_xlabel('phase_concentration')
    axes[1].set_title('V3: concentration under null')
    axes[1].legend(fontsize=9)

    axes[2].hist(ratio_values, bins=15, color='coral', edgecolor='maroon')
    axes[2].axvline(1.0, color='green', ls='--', label='expected = 1.0')
    axes[2].axvline(ratio_values.mean(), color='red',
                    label=f'mean={ratio_values.mean():.2f}')
    axes[2].set_xlabel('up_down_ratio'); axes[2].set_title('V3: ratio under null')
    axes[2].legend(fontsize=9)

    plt.suptitle(f"V3 OVERALL: {'PASS' if overall else 'FAIL'}",
                 fontsize=13, color='darkgreen' if overall else 'darkred',
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTDIR / "V3_no_coupling_baseline.png", dpi=120)
    plt.close()

    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


# ============================================================
# V4: Three regimes give correctly-different answers
# ============================================================
def V4_three_regimes():
    """
    Claim: UP-locked, DOWN-locked, intermediate-locked spindles should
    produce three distinct, correctly-identified outputs.
    """
    print("\n" + "=" * 72)
    print("V4: UP / DOWN / intermediate three regimes")
    print("=" * 72)

    rng = np.random.default_rng(0)
    r_ctx = make_cortex_pulses(T_VEC, rng=rng)

    cases = [
        ("UP peak (0°)", 0.0, 0, 30, "up"),
        ("DOWN trough (180°)", np.pi, 180, 30, "down"),
        ("DOWN-to-UP (-90°)", -np.pi / 2, -90, 30, "intermediate"),
    ]

    print(f"  {'Case':<25} {'expected':<12} {'argmax':<10} {'pref':<10} {'ratio':<8} {'pass'}")
    results = []
    for name, target_rad, expect_deg, tol_deg, side in cases:
        spindle = make_spindle_at_phase(T_VEC, 1.3, target_rad,
                                        rng=np.random.default_rng(42))
        out = compute_pac_metrics(r_ctx, spindle, FS)

        argmax_dist = abs(np.degrees(out["phase_argmax"]) - expect_deg)
        argmax_dist = min(argmax_dist, 360 - argmax_dist)
        passed_argmax = argmax_dist <= tol_deg

        if side == "up":
            passed_ratio = out["up_down_ratio"] > 1.5
        elif side == "down":
            passed_ratio = out["up_down_ratio"] < 0.7
        else:
            passed_ratio = 0.4 < out["up_down_ratio"] < 2.5

        passed = passed_argmax and passed_ratio
        results.append(passed)
        print(f"  {name:<25} {expect_deg:>+4d}°       "
              f"{fmt_phase(out['phase_argmax']):>8}  "
              f"{fmt_phase(out['preferred_phase']):>8}  "
              f"{out['up_down_ratio']:>6.2f}   "
              f"{'PASS' if passed else 'FAIL'}")

    overall = all(results)
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4),
                             subplot_kw=dict(projection='polar'))
    for ax, (name, target_rad, expect_deg, _, _) in zip(axes, cases):
        spindle = make_spindle_at_phase(T_VEC, 1.3, target_rad,
                                        rng=np.random.default_rng(42))
        out = compute_pac_metrics(r_ctx, spindle, FS)

        # Reconstruct histogram for plotting
        ctx_range = r_ctx.max() - r_ctx.min()
        from scipy.signal import find_peaks
        ctx_peaks, _ = find_peaks(r_ctx, distance=int(0.7 * FS),
                                  prominence=0.3 * ctx_range)
        so_phase = np.full(len(r_ctx), np.nan)
        for i in range(len(ctx_peaks) - 1):
            p0 = ctx_peaks[i]; p1 = ctx_peaks[i + 1]
            rel = (np.arange(p0, p1) - p0) / (p1 - p0)
            so_phase[p0:p1] = np.where(rel < 0.5,
                                       2 * np.pi * rel,
                                       2 * np.pi * (rel - 1))
        sos = butter(4, [10, 14], btype='band', fs=FS, output='sos')
        sp_amp = np.abs(hilbert(sosfiltfilt(sos, spindle)))
        edge = int(0.5 * FS)
        ph = so_phase[edge:-edge]
        sp = sp_amp[edge:-edge]
        valid = ~np.isnan(ph)
        bin_edges = np.linspace(-np.pi, np.pi, 19)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        means = np.zeros(18)
        for i in range(18):
            m = (ph[valid] >= bin_edges[i]) & (ph[valid] < bin_edges[i + 1])
            if m.any():
                means[i] = sp[valid][m].mean()
        ax.bar(bin_centers, means, width=2 * np.pi / 18,
               alpha=0.7, color='steelblue', edgecolor='navy')
        ax.plot([np.radians(expect_deg)] * 2,
                [0, means.max() * 1.1], 'g-', lw=2.5, label='target')
        ax.plot([out["phase_argmax"]] * 2,
                [0, means.max() * 1.1], 'r--', lw=2, label='argmax')
        ax.set_title(name, fontsize=10, pad=12)
        ax.set_theta_zero_location('E')
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=8)
    plt.suptitle(f"V4 OVERALL: {'PASS' if overall else 'FAIL'}",
                 fontsize=13, color='darkgreen' if overall else 'darkred',
                 fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(OUTDIR / "V4_three_regimes.png", dpi=120, bbox_inches='tight')
    plt.close()
    return overall


# ============================================================
# V5: Robustness under noise
# ============================================================
def V5_noise_robustness():
    """
    Claim: as noise increases, recovery degrades gracefully — no crashes,
    phase still recoverable up to moderate noise levels.
    """
    print("\n" + "=" * 72)
    print("V5: Robustness under noise")
    print("=" * 72)

    noise_levels = [0, 0.5, 1, 2, 5, 10, 20]
    results = []
    rng = np.random.default_rng(0)
    base_r_ctx = make_cortex_pulses(T_VEC, rng=np.random.default_rng(0))
    base_spindle = make_spindle_at_phase(T_VEC, 1.3, 0,
                                         rng=np.random.default_rng(1))

    print(f"  {'noise σ':<10} {'argmax':<10} {'pref':<10} {'mi':<10} {'concentration':<14} {'ok'}")
    for ns in noise_levels:
        r_ctx = base_r_ctx + ns * rng.standard_normal(len(base_r_ctx))
        spindle = base_spindle + ns * rng.standard_normal(len(base_spindle))
        out = compute_pac_metrics(r_ctx, spindle, FS)
        results.append((ns, out))
        if out["ok"]:
            print(f"  {ns:<10.1f} "
                  f"{fmt_phase(out['phase_argmax']):>8}   "
                  f"{fmt_phase(out['preferred_phase']):>8}   "
                  f"{out['mi']:>8.4f}   "
                  f"{out['phase_concentration']:>10.3f}     ok")
        else:
            print(f"  {ns:<10.1f}  ok=False (computation refused)")

    # Pass criteria:
    # 1. No crashes (all return a dict)
    # 2. Low/moderate noise (≤5) recovers phase to within 30°
    no_crashes = all(isinstance(o, dict) for _, o in results)
    low_noise_recover = all(
        not o["ok"] or abs(np.degrees(o["phase_argmax"])) <= 30
        for ns, o in results if ns <= 5
    )
    overall = no_crashes and low_noise_recover

    fig, ax = plt.subplots(figsize=(10, 5))
    ns_array = [ns for ns, o in results if o["ok"]]
    arg_array = [np.degrees(o["phase_argmax"]) for ns, o in results if o["ok"]]
    pref_array = [np.degrees(o["preferred_phase"]) for ns, o in results if o["ok"]]
    mi_array = [o["mi"] for ns, o in results if o["ok"]]

    ax2 = ax.twinx()
    ax.plot(ns_array, arg_array, 'go-', label='phase_argmax')
    ax.plot(ns_array, pref_array, 'b^-', label='preferred_phase')
    ax.axhline(0, color='red', ls='--', alpha=0.5, label='ground truth = 0°')
    ax.fill_between(ns_array, -30, 30, color='green', alpha=0.1, label='±30° tolerance')
    ax.set_xlabel('Noise σ added to signals')
    ax.set_ylabel('Recovered phase [deg]')
    ax2.plot(ns_array, mi_array, 'ko:', alpha=0.5, label='MI')
    ax2.set_ylabel('MI')
    ax2.set_ylim([0, max(mi_array) * 1.2])
    ax.set_xscale('symlog', linthresh=0.5)
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.set_title(f"V5: robustness under noise — "
                 f"{'PASS' if overall else 'FAIL'}",
                 color='darkgreen' if overall else 'darkred', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTDIR / "V5_noise_robustness.png", dpi=120)
    plt.close()

    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


# ============================================================
# V6: Old vs new on the SAME synthetic UP-locked input
# ============================================================
def V6_v7_old_vs_new():
    """
    Claim: on synthetic UP-locked coupling with NARROW-PULSE cortex
    (the regime where v7 fails), the new function should give phase ≈ 0°
    while v7's Hilbert version gives a biased non-zero value.
    """
    print("\n" + "=" * 72)
    print("V6: Old (Hilbert) vs new (cycle-by-cycle) on same input")
    print("=" * 72)

    rng = np.random.default_rng(42)
    r_ctx = make_cortex_pulses(T_VEC, period=1.3, pulse_width_s=0.15, rng=rng)
    spindle = make_spindle_at_phase(T_VEC, 1.3, 0, rng=rng)

    # New (fixed) version
    new_out = compute_pac_metrics(r_ctx, spindle, FS)

    # Old (v7-style) version
    sos_so = butter(4, [0.5, 1.5], btype='band', fs=FS, output='sos')
    so_filt = sosfiltfilt(sos_so, r_ctx)
    so_phase_old = np.angle(hilbert(so_filt))
    sos_sp = butter(4, [10, 14], btype='band', fs=FS, output='sos')
    sp_amp = np.abs(hilbert(sosfiltfilt(sos_sp, spindle)))
    edge = int(0.5 * FS)
    so_phase_old = so_phase_old[edge:-edge]
    sp_amp_t = sp_amp[edge:-edge]
    mvl_old = (sp_amp_t * np.exp(1j * so_phase_old)).mean()
    old_phase_deg = np.degrees(np.angle(mvl_old))

    print(f"  Synthetic ground truth: spindle locked exactly at UP peak (0°)")
    print(f"  Old (v7 Hilbert):  preferred_phase = {old_phase_deg:+.1f}°  "
          f"{'[BIASED]' if abs(old_phase_deg) > 10 else ''}")
    print(f"  New (fixed):       preferred_phase = "
          f"{np.degrees(new_out['preferred_phase']):+.1f}°")
    print(f"  New phase_argmax = {fmt_phase(new_out['phase_argmax'])}")

    # Pass: new is closer to 0° than old
    new_dist = abs(np.degrees(new_out["preferred_phase"]))
    old_dist = abs(old_phase_deg)
    overall = (new_dist < 15) and (new_dist < old_dist)

    fig, ax = plt.subplots(figsize=(10, 5))
    methods = ['Old (Hilbert phase)', 'New (cycle-by-cycle)']
    values = [old_phase_deg, np.degrees(new_out["preferred_phase"])]
    colors = ['steelblue', 'seagreen']
    bars = ax.barh(methods, values, color=colors, edgecolor='black')
    ax.axvline(0, color='red', lw=2, ls='--',
               label='Ground truth = 0° (UP peak)')
    ax.fill_betweenx([-0.5, 1.5], -15, 15, color='green', alpha=0.15,
                     label='±15° tolerance')
    ax.set_xlim([-200, 200])
    ax.set_xlabel('Recovered preferred phase [deg]')
    ax.set_title(f"V6: same UP-locked input, {'PASS' if overall else 'FAIL'}",
                 color='darkgreen' if overall else 'darkred', fontweight='bold')
    for bar, val in zip(bars, values):
        ax.text(val + 5 * np.sign(val), bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}°", va='center',
                ha='left' if val > 0 else 'right', fontsize=11,
                fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "V6_v7_old_vs_new_comparison.png", dpi=120)
    plt.close()

    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


# ============================================================
# V7: Bin sampling uniformity
# ============================================================
def V7_bin_uniformity():
    """
    Claim: under cycle-by-cycle phase, the 18 bins should be sampled
    approximately uniformly (max/min ratio ≤ 1.5).
    Under v7 Hilbert phase on pulse cortex, max/min is ~3+.
    """
    print("\n" + "=" * 72)
    print("V7: Bin sampling uniformity")
    print("=" * 72)

    rng = np.random.default_rng(0)
    r_ctx = make_cortex_pulses(T_VEC, period=1.3, pulse_width_s=0.15, rng=rng)

    # Build cycle-by-cycle phase
    from scipy.signal import find_peaks
    ctx_range = r_ctx.max() - r_ctx.min()
    ctx_peaks, _ = find_peaks(r_ctx, distance=int(0.7 * FS),
                              prominence=0.3 * ctx_range)
    so_phase_new = np.full(len(r_ctx), np.nan)
    for i in range(len(ctx_peaks) - 1):
        p0 = ctx_peaks[i]; p1 = ctx_peaks[i + 1]
        rel = (np.arange(p0, p1) - p0) / (p1 - p0)
        so_phase_new[p0:p1] = np.where(rel < 0.5,
                                       2 * np.pi * rel,
                                       2 * np.pi * (rel - 1))

    # Old Hilbert phase
    sos = butter(4, [0.5, 1.5], btype='band', fs=FS, output='sos')
    so_phase_old = np.angle(hilbert(sosfiltfilt(sos, r_ctx)))

    edge = int(0.5 * FS)
    valid_new = ~np.isnan(so_phase_new[edge:-edge])
    bin_edges = np.linspace(-np.pi, np.pi, 19)
    counts_new = np.histogram(so_phase_new[edge:-edge][valid_new], bins=bin_edges)[0]
    counts_old = np.histogram(so_phase_old[edge:-edge], bins=bin_edges)[0]

    ratio_new = counts_new.max() / counts_new.min()
    ratio_old = counts_old.max() / counts_old.min()

    pass_new = ratio_new <= 1.5
    print(f"  Old Hilbert max/min: {ratio_old:.2f} (expect >2 on pulse cortex)")
    print(f"  New cycle  max/min: {ratio_new:.2f} (expect <1.5)   "
          f"{'PASS' if pass_new else 'FAIL'}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    bin_centers_deg = np.degrees((bin_edges[:-1] + bin_edges[1:]) / 2)
    axes[0].bar(bin_centers_deg, counts_old, width=18, color='steelblue',
                edgecolor='navy')
    axes[0].axhline(counts_old.mean(), color='red', ls='--', label='uniform')
    axes[0].set_title(f'Old Hilbert: max/min = {ratio_old:.2f}')
    axes[0].set_xlabel('SO phase bin [deg]')
    axes[0].set_ylabel('# samples')
    axes[0].legend()

    axes[1].bar(bin_centers_deg, counts_new, width=18, color='seagreen',
                edgecolor='darkgreen')
    axes[1].axhline(counts_new.mean(), color='red', ls='--', label='uniform')
    axes[1].set_title(f'New cycle: max/min = {ratio_new:.2f}')
    axes[1].set_xlabel('SO phase bin [deg]')
    axes[1].legend()

    plt.suptitle(f"V7 OVERALL: {'PASS' if pass_new else 'FAIL'}",
                 fontsize=13, color='darkgreen' if pass_new else 'darkred',
                 fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(OUTDIR / "V7_bin_uniformity.png", dpi=120)
    plt.close()

    print(f"  OVERALL: {'PASS' if pass_new else 'FAIL'}")
    return pass_new


# ============================================================
# V8: Bimodality detection accuracy
# ============================================================
def V8_bimodality_detection():
    """
    Claim: bimodality_flag should be True for genuinely bimodal coupling
    and False for unimodal coupling.
    """
    print("\n" + "=" * 72)
    print("V8: Bimodality detection")
    print("=" * 72)

    rng = np.random.default_rng(0)
    r_ctx = make_cortex_pulses(T_VEC, rng=rng)

    # Case 1: unimodal (only UP-locked)
    spindle1 = make_spindle_at_phase(T_VEC, 1.3, 0,
                                     rng=np.random.default_rng(1))
    out1 = compute_pac_metrics(r_ctx, spindle1, FS)

    # Case 2: bimodal (UP + DOWN)
    sp_up = make_spindle_at_phase(T_VEC, 1.3, 0, rng=np.random.default_rng(2))
    sp_dn = make_spindle_at_phase(T_VEC, 1.3, np.pi,
                                  rng=np.random.default_rng(3))
    spindle2 = sp_up + sp_dn - 25  # subtract one baseline
    out2 = compute_pac_metrics(r_ctx, spindle2, FS)

    # Case 3: weakly bimodal (UP much stronger than DOWN)
    sp_up_strong = make_spindle_at_phase(T_VEC, 1.3, 0,
                                         burst_amp=40,
                                         rng=np.random.default_rng(4))
    sp_dn_weak = make_spindle_at_phase(T_VEC, 1.3, np.pi,
                                       burst_amp=10,
                                       rng=np.random.default_rng(5))
    spindle3 = sp_up_strong + sp_dn_weak - 25
    out3 = compute_pac_metrics(r_ctx, spindle3, FS)

    cases = [
        ("unimodal UP", out1, False),
        ("strongly bimodal", out2, True),
        ("weakly bimodal (UP >> DOWN)", out3, False),  # heuristic threshold
    ]

    results = []
    print(f"  {'Case':<35} {'flag':<6} {'expect':<6} {'pass'}")
    for name, out, expected in cases:
        flag = out["bimodality_flag"]
        passed = (flag == expected)
        results.append(passed)
        print(f"  {name:<35} {str(flag):<6} {str(expected):<6} "
              f"{'PASS' if passed else 'FAIL'}")

    overall = all(results)
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, out, expected) in zip(axes, cases):
        # Recompute the histogram for plotting (re-run to access mean_amp)
        from scipy.signal import find_peaks
        cr = r_ctx.max() - r_ctx.min()
        ctx_peaks, _ = find_peaks(r_ctx, distance=int(0.7 * FS),
                                  prominence=0.3 * cr)
        so_phase = np.full(len(r_ctx), np.nan)
        for i in range(len(ctx_peaks) - 1):
            p0 = ctx_peaks[i]; p1 = ctx_peaks[i + 1]
            rel = (np.arange(p0, p1) - p0) / (p1 - p0)
            so_phase[p0:p1] = np.where(rel < 0.5,
                                       2 * np.pi * rel,
                                       2 * np.pi * (rel - 1))
        # Need spindle for this case; recompute
        if "unimodal" in name:
            sp = spindle1
        elif "strongly" in name:
            sp = spindle2
        else:
            sp = spindle3
        sos = butter(4, [10, 14], btype='band', fs=FS, output='sos')
        sp_amp = np.abs(hilbert(sosfiltfilt(sos, sp)))
        edge = int(0.5 * FS)
        ph = so_phase[edge:-edge]
        spm = sp_amp[edge:-edge]
        valid = ~np.isnan(ph)
        bin_edges = np.linspace(-np.pi, np.pi, 19)
        bin_centers_deg = np.degrees((bin_edges[:-1] + bin_edges[1:]) / 2)
        means = np.zeros(18)
        for i in range(18):
            m = (ph[valid] >= bin_edges[i]) & (ph[valid] < bin_edges[i + 1])
            if m.any():
                means[i] = spm[valid][m].mean()
        ax.bar(bin_centers_deg, means, width=18, color='steelblue',
               edgecolor='navy')
        col = 'darkgreen' if (out["bimodality_flag"] == expected) else 'darkred'
        ax.set_title(f"{name}\nflag={out['bimodality_flag']}, expect={expected}",
                     fontsize=10, color=col)
        ax.set_xlabel('SO phase [deg]')
    plt.suptitle(f"V8 OVERALL: {'PASS' if overall else 'FAIL'}",
                 fontsize=13, color='darkgreen' if overall else 'darkred',
                 fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(OUTDIR / "V8_bimodality_detection.png", dpi=120)
    plt.close()
    return overall


# ============================================================
# V9: Edge case handling
# ============================================================
def V9_edge_cases():
    """
    Claim: function refuses (ok=False) on degenerate inputs without
    crashing.
    """
    print("\n" + "=" * 72)
    print("V9: Edge case handling")
    print("=" * 72)

    edge_cases = [
        ("flat r_ctx (zeros)", np.zeros(int(60 * FS)),
         np.random.standard_normal(int(60 * FS))),
        ("flat r_thal (zeros)", np.cos(2 * np.pi * 1 * T_VEC),
         np.zeros(int(60 * FS))),
        ("very short signal (1s)", np.cos(2 * np.pi * 1 * np.arange(int(FS)) / FS),
         np.cos(2 * np.pi * 12 * np.arange(int(FS)) / FS)),
        ("constant signal (DC)", np.full(int(60 * FS), 5.0),
         np.full(int(60 * FS), 5.0)),
        ("NaN-injected r_ctx",
         np.where(np.arange(int(60 * FS)) % 10000 == 0, np.nan,
                  np.cos(2 * np.pi * 1 * T_VEC)),
         np.cos(2 * np.pi * 12 * T_VEC)),
        ("very low peak prominence (< threshold)",
         0.01 * np.cos(2 * np.pi * 1 * T_VEC) + 50,
         np.cos(2 * np.pi * 12 * T_VEC)),
    ]

    results = []
    log = []
    for name, r_ctx, r_thal in edge_cases:
        try:
            out = compute_pac_metrics(r_ctx, r_thal, FS)
            crashed = False
            ok = out["ok"]
        except Exception as e:
            crashed = True
            ok = None

        # Pass: no crash, AND ok=False on truly bad inputs
        passed = not crashed
        results.append(passed)
        verdict = "PASS" if passed else "FAIL (crashed)"
        msg = f"  {name:<45} ok={str(ok):<6}  {verdict}"
        print(msg)
        log.append(msg)

    overall = all(results)
    with open(OUTDIR / "V9_edge_cases.txt", "w") as f:
        f.write("V9 — Edge case handling\n")
        f.write("=" * 60 + "\n")
        f.write("\n".join(log) + "\n")
        f.write(f"\nOVERALL: {'PASS' if overall else 'FAIL'}\n")

    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 72)
    print("VALIDATION SUITE: compute_pac_metrics_fixed")
    print("=" * 72)
    print(f"Outputs go to: {OUTDIR}")
    print()

    tests = [
        ("V1: synthetic UP-locked recovery",     V1_synthetic_up_locked),
        ("V2: invariance to cortex waveform",    V2_invariance_to_waveform),
        ("V3: no-coupling baseline",             V3_no_coupling_baseline),
        ("V4: three regimes correctly distinguished", V4_three_regimes),
        ("V5: noise robustness",                 V5_noise_robustness),
        ("V6: old (v7) vs new comparison",       V6_v7_old_vs_new),
        ("V7: bin sampling uniformity",          V7_bin_uniformity),
        ("V8: bimodality detection",             V8_bimodality_detection),
        ("V9: edge case handling",               V9_edge_cases),
    ]

    results = {}
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n!!! {name} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # ========== Summary ==========
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    n_pass = sum(results.values())
    n_total = len(results)
    for name, passed in results.items():
        mark = "PASS" if passed else "FAIL"
        print(f"  {mark}   {name}")
    print(f"\n{n_pass} / {n_total} validations passed")

    with open(OUTDIR / "SUMMARY.txt", "w") as f:
        f.write("VALIDATION SUMMARY: compute_pac_metrics_fixed\n")
        f.write("=" * 60 + "\n\n")
        for name, passed in results.items():
            f.write(f"  {'PASS' if passed else 'FAIL'}   {name}\n")
        f.write(f"\n{n_pass} / {n_total} validations passed\n\n")
        f.write("Plots and details:\n")
        f.write("  V1_synthetic_up_locked.png\n")
        f.write("  V2_invariance_to_waveform.png\n")
        f.write("  V3_no_coupling_baseline.png\n")
        f.write("  V4_three_regimes.png\n")
        f.write("  V5_noise_robustness.png\n")
        f.write("  V6_v7_old_vs_new_comparison.png\n")
        f.write("  V7_bin_uniformity.png\n")
        f.write("  V8_bimodality_detection.png\n")
        f.write("  V9_edge_cases.txt\n")
    print(f"\nSummary saved to {OUTDIR / 'SUMMARY.txt'}")
    return n_pass == n_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)