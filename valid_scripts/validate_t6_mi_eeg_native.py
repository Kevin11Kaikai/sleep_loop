"""
validate_t6_mi_eeg_native.py
=============================
5-case synthetic validation of EEG-native T6 (AASM SO UP) and MI (Tort PAC).

Pass criteria (printed at the end as PASS/FAIL):
    Case 1  full coupling                T6 in [0.1, 0.8]   AND  MI >= 0.02
    Case 2  pure SO, no spindle          T6 in [0.1, 0.8]   AND  MI <= 0.005
    Case 3  pure spindle, no SO          T6 ok==False                 (no UP detected)
    Case 4  white noise                  T6 ok==False                 (no SO structure)
    Case 5  SO + spindle, DOWN-locked    T6 in [0.1, 0.8]   AND  MI >= 0.02
                                         AND |preferred_phase| > pi/2

These are signal-shape sanity checks, not exact-value checks. The point is
that the algorithm gives qualitatively correct outputs on signals where the
ground-truth structure is known.
"""

import sys, os
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "S4_v7_repair"))

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from compute_pac_metrics_eeg_native import (
    compute_t6_eeg_aasm, compute_mi_eeg_native,
)

FS   = 1000.0
DUR  = 60.0
T    = np.arange(0, DUR, 1 / FS)
RNG  = np.random.default_rng(42)


def make_so_signal(amp_uv=60.0, mean_freq=0.8, ibi_cv_target=0.35, seed=0):
    """
    Realistic SO: concatenate half-cycles with lognormally-jittered periods.
    Each cycle has independent period ~ LogNormal(mean=1/mean_freq, sigma=ibi_cv_target).
    Produces SO whose detected UP-event IBI CV is close to ibi_cv_target.
    """
    rng = np.random.default_rng(seed)
    mean_period = 1.0 / mean_freq
    n_target_samples = len(T)
    cycles = []
    t = 0.0
    while t < DUR + 5.0:
        period = float(rng.lognormal(mean=np.log(mean_period),
                                     sigma=ibi_cv_target))
        cycles.append(max(period, 0.4))  # floor to avoid sub-Nyquist cycles
        t += cycles[-1]
    parts = []
    for p in cycles:
        n = int(p * FS)
        if n < 4:
            continue
        phase = np.linspace(0, 2 * np.pi, n, endpoint=False)
        parts.append(np.sin(phase))
    sig = amp_uv * np.concatenate(parts)
    if len(sig) >= n_target_samples:
        sig = sig[:n_target_samples]
    else:
        sig = np.concatenate([sig,
                              np.zeros(n_target_samples - len(sig))])
    return sig


def make_spindle_burst(carrier_freq=12.0, peak_amp_uv=20.0, modulation=None):
    """Spindle carrier modulated by `modulation` (same length as T)."""
    carrier = np.sin(2 * np.pi * carrier_freq * T)
    if modulation is None:
        modulation = np.ones_like(T)
    m_norm = modulation / max(modulation.max(), 1e-9)
    return peak_amp_uv * carrier * m_norm


def so_phase_modulator(so_signal, fs, target_phase=0.0, sharpness=2.0):
    """
    Return a non-negative envelope that peaks when SO phase ≈ target_phase.
    target_phase=0 → UP-locked; target_phase=±π → DOWN-locked.
    """
    sos = butter(4, [0.5, 1.5], btype="band", fs=fs, output="sos")
    so_filt  = sosfiltfilt(sos, so_signal)
    so_phase = np.angle(hilbert(so_filt))
    return np.exp(sharpness * np.cos(so_phase - target_phase))


def evaluate(label, sig, expect):
    t6 = compute_t6_eeg_aasm(sig, FS)
    mi = compute_mi_eeg_native(sig, FS)
    print(f"\n--- {label} ---")
    print(f"  T6: ibi_cv={t6['ibi_cv']:.4f}  n_up={t6['n_up_events']}  "
          f"n_neg={t6['n_neg_peaks']}  n_pos={t6['n_pos_peaks']}  ok={t6['ok']}")
    print(f"  MI: mi={mi['mi']:.5f}  preferred_phase={mi['preferred_phase']:+.3f} rad  "
          f"ok={mi['ok']}")
    checks = []
    for name, cond in expect.items():
        ok = cond(t6, mi)
        checks.append((name, ok))
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    return checks


def main():
    print("=" * 60)
    print(f"Synthetic validation @ fs={FS} Hz, dur={DUR}s")
    print("=" * 60)

    all_checks = []

    # ── Case 1: full coupling, UP-locked spindle ─────────────────────────
    so   = make_so_signal(amp_uv=60.0, mean_freq=0.8, seed=1)
    mod1 = so_phase_modulator(so, FS, target_phase=0.0, sharpness=3.0)
    sp1  = make_spindle_burst(modulation=mod1)
    sig1 = so + sp1 + 3.0 * RNG.standard_normal(len(T))
    checks = evaluate("Case 1: full coupling (UP-locked)", sig1, {
        "T6 in [0.1, 0.8]": lambda t, m: t["ok"] and 0.1 <= t["ibi_cv"] <= 0.8,
        "MI >= 0.02":       lambda t, m: m["mi"] >= 0.02,
        "preferred near 0": lambda t, m: abs(m["preferred_phase"]) < np.pi / 2,
    })
    all_checks += [("C1::" + n, ok) for n, ok in checks]

    # ── Case 2: pure SO, no spindle ─────────────────────────────────────
    sig2 = make_so_signal(amp_uv=60.0, mean_freq=0.8, seed=1) \
         + 3.0 * RNG.standard_normal(len(T))
    checks = evaluate("Case 2: pure SO, no spindle", sig2, {
        "T6 in [0.1, 0.8]": lambda t, m: t["ok"] and 0.1 <= t["ibi_cv"] <= 0.8,
        "MI <= 0.005":      lambda t, m: m["mi"] <= 0.005,
    })
    all_checks += [("C2::" + n, ok) for n, ok in checks]

    # ── Case 3: pure spindle, no SO ─────────────────────────────────────
    sp3  = make_spindle_burst(modulation=np.ones_like(T))
    sig3 = sp3 + 3.0 * RNG.standard_normal(len(T))
    checks = evaluate("Case 3: pure spindle, no SO", sig3, {
        "T6 ok==False (no UP)": lambda t, m: t["ok"] is False,
    })
    all_checks += [("C3::" + n, ok) for n, ok in checks]

    # ── Case 4: white noise ─────────────────────────────────────────────
    sig4 = 10.0 * RNG.standard_normal(len(T))
    checks = evaluate("Case 4: white noise", sig4, {
        "T6 ok==False": lambda t, m: t["ok"] is False,
    })
    all_checks += [("C4::" + n, ok) for n, ok in checks]

    # ── Case 5: full coupling, DOWN-locked spindle ──────────────────────
    so   = make_so_signal(amp_uv=60.0, mean_freq=0.8, seed=1)
    mod5 = so_phase_modulator(so, FS, target_phase=np.pi, sharpness=3.0)
    sp5  = make_spindle_burst(modulation=mod5)
    sig5 = so + sp5 + 3.0 * RNG.standard_normal(len(T))
    checks = evaluate("Case 5: DOWN-locked spindle", sig5, {
        "T6 in [0.1, 0.8]":     lambda t, m: t["ok"] and 0.1 <= t["ibi_cv"] <= 0.8,
        "MI >= 0.02":           lambda t, m: m["mi"] >= 0.02,
        "preferred near ±π":    lambda t, m: abs(m["preferred_phase"]) > np.pi / 2,
    })
    all_checks += [("C5::" + n, ok) for n, ok in checks]

    # ── Summary ─────────────────────────────────────────────────────────
    n_pass = sum(1 for _, ok in all_checks if ok)
    n_tot  = len(all_checks)
    print()
    print("=" * 60)
    print(f"OVERALL: {n_pass}/{n_tot} checks PASSED")
    if n_pass < n_tot:
        print("FAILED checks:")
        for n, ok in all_checks:
            if not ok:
                print(f"  - {n}")
    print("=" * 60)
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    sys.exit(main())
