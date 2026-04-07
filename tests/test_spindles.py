"""
tests/test_spindles.py
Validate thalamocortical simulation outputs from s3_sleep_kernel.py.

Checks:
  1. Thalamus TCR has a spindle-band peak (10–15 Hz) ≥ 2× background mean.
  2. Cortex has slow-wave band energy (0.2–1.5 Hz) ≥ 0.5× total mean power.
  3. Cortex is not completely silent (max firing > 0.01 Hz).

Run from the sleep_loop project root:
  python tests/test_spindles.py
"""

import os
import sys
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── load data ────────────────────────────────────────────────────────────────

r_cortex   = np.load("outputs/r_cortex.npy")    # Hz
r_thalamus = np.load("outputs/r_thalamus.npy")  # Hz
t_ms       = np.load("outputs/t_ms.npy")

# infer sampling rate from time axis
dt_s = (t_ms[-1] - t_ms[0]) / (len(t_ms) - 1)
if dt_s > 1.0:          # t is in ms
    fs = 1000.0 / dt_s
else:                    # t is already in seconds
    fs = 1.0 / dt_s
print(f"Inferred fs = {fs:.1f} Hz")

# discard first 5 s (transient)
n_drop     = int(5.0 * fs)
r_c_clean  = r_cortex[n_drop:]
r_th_clean = r_thalamus[n_drop:]

# ── Welch PSD (10-second window) ─────────────────────────────────────────────

nperseg = min(int(10.0 * fs), len(r_c_clean))
f_c,  p_c  = welch(r_c_clean,  fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
f_th, p_th = welch(r_th_clean, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

# ── helpers ──────────────────────────────────────────────────────────────────

def band_power(f, p, f_low, f_high):
    mask = (f >= f_low) & (f <= f_high)
    return float(p[mask].max()) if mask.any() else 0.0

def band_peak_freq(f, p, f_low, f_high):
    mask = (f >= f_low) & (f <= f_high)
    if not mask.any():
        return None
    return float(f[mask][np.argmax(p[mask])])

# ── band metrics ─────────────────────────────────────────────────────────────

# cortex: slow-wave
slow_pwr   = band_power(f_c, p_c, 0.2, 1.5)
slow_peak  = band_peak_freq(f_c, p_c, 0.2, 1.5)
total_pwr  = float(p_c[(f_c >= 0.1) & (f_c <= 30)].mean())

# thalamus: spindle
spin_pwr   = band_power(f_th, p_th, 10.0, 15.0)
spin_peak  = band_peak_freq(f_th, p_th, 10.0, 15.0)
spin_ratio = spin_pwr / (float(p_th[(f_th >= 0.5) & (f_th <= 30)].mean()) + 1e-10)
slow_ratio = slow_pwr / (total_pwr + 1e-10)

# ── diagnostics ──────────────────────────────────────────────────────────────

print("\n=== Power Spectrum Diagnostics ===")
print(f"Cortex   r_E  range  : {r_c_clean.min():.3f} – {r_c_clean.max():.3f} Hz")
print(f"Thalamus r_TCR range : {r_th_clean.min():.3f} – {r_th_clean.max():.3f} Hz")
print(f"Cortex  slow-wave  (0.2–1.5 Hz)  peak power : {slow_pwr:.4e}  at {slow_peak:.2f} Hz")
print(f"Thalamus spindle   (10–15 Hz)    peak power : {spin_pwr:.4e}  at {spin_peak:.2f} Hz")
print(f"Spindle-to-mean ratio (thalamus) : {spin_ratio:.2f}")
print(f"Slow-to-mean ratio   (cortex)    : {slow_ratio:.2f}")
print(f"Cortex total mean power (0.1–30 Hz) : {total_pwr:.4e}")

# ── plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Thalamocortical Simulation — PSD Validation", fontsize=13)

n_plot = int(20.0 * fs)
t_plot = np.arange(n_plot) / fs

axes[0, 0].plot(t_plot, r_c_clean[-n_plot:], color="#534AB7", lw=0.8)
axes[0, 0].set_title("Cortex $r_E$ — last 20 s")
axes[0, 0].set_xlabel("Time [s]")
axes[0, 0].set_ylabel("Rate [Hz]")

axes[0, 1].plot(t_plot, r_th_clean[-n_plot:], color="#1D9E75", lw=0.8)
axes[0, 1].set_title("Thalamus $r_{TCR}$ — last 20 s")
axes[0, 1].set_xlabel("Time [s]")
axes[0, 1].set_ylabel("Rate [Hz]")

freq_mask_c  = f_c  <= 30
freq_mask_th = f_th <= 30

axes[1, 0].semilogy(f_c[freq_mask_c], p_c[freq_mask_c], color="#534AB7", lw=1.5)
axes[1, 0].axvspan(0.2, 1.5, alpha=0.15, color="orange", label="Slow-wave (0.2–1.5 Hz)")
axes[1, 0].set_title("Cortex PSD")
axes[1, 0].set_xlabel("Frequency [Hz]")
axes[1, 0].set_ylabel("Power [(Hz)²/Hz]")
axes[1, 0].legend(fontsize=8)

axes[1, 1].semilogy(f_th[freq_mask_th], p_th[freq_mask_th], color="#1D9E75", lw=1.5)
axes[1, 1].axvspan(10, 15, alpha=0.15, color="green", label="Spindle (10–15 Hz)")
axes[1, 1].set_title("Thalamus PSD")
axes[1, 1].set_xlabel("Frequency [Hz]")
axes[1, 1].set_ylabel("Power [(Hz)²/Hz]")
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/psd_validation.png", dpi=150)
print("\nSaved: outputs/psd_validation.png")

# ── assertions ───────────────────────────────────────────────────────────────

PASS = True

# 1. thalamus spindle peak (≥ 2× background)
if spin_ratio >= 2.0:
    print(f"\n✓ Spindle peak detected in thalamus  (ratio = {spin_ratio:.1f} >= 2.0)")
else:
    print(f"\n✗ Spindle peak WEAK in thalamus      (ratio = {spin_ratio:.2f} < 2.0)")
    PASS = False

# 2. cortex slow-wave band energy (≥ 0.5× total mean)
if slow_ratio >= 0.5:
    print(f"✓ Slow-wave energy present in cortex  (ratio = {slow_ratio:.1f} >= 0.5)")
else:
    print(f"✗ Slow-wave energy WEAK in cortex     (ratio = {slow_ratio:.2f} < 0.5)")
    print("  Hint: cortex may be in down-state — check mue/mui parameters")
    PASS = False

# 3. cortex not completely silent
if r_c_clean.max() > 0.01:
    print(f"✓ Cortex is not silent                (max = {r_c_clean.max():.3f} Hz > 0.01)")
else:
    print(f"✗ Cortex appears completely SILENT    (max = {r_c_clean.max():.4f} Hz)")
    PASS = False

# ── verdict ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 45)
if PASS:
    print("ALL ASSERTIONS PASSED ✓")
else:
    print("SOME ASSERTIONS FAILED — see diagnostics above")
    print("Visual diagnosis: outputs/psd_validation.png")

sys.exit(0 if PASS else 1)
