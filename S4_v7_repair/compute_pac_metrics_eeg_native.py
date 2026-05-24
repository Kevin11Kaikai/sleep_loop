"""
compute_pac_metrics_eeg_native.py
==================================
EEG-native T6 (SO IBI CV) and MI (SO–spindle PAC) for x_obs extraction.

Used ONLY on the EEG side of the pipeline (compute_xobs_from_eeg_v4.py).
The simulator side keeps V7's r_ctx run-length T6 and the legacy
cross-channel PAC (r_ctx phase × r_thal amp) — see sprint1_phase2_plan_v2.md §1.

Two functions:
    compute_t6_eeg_aasm(eeg_raw, fs) -> dict
        AASM-style SO UP event detection on raw EEG:
        bandpass 0.2-4 Hz -> find_peaks DOWN/UP -> 75 µV half-wave criterion
        -> 0.5-2.0 s duration criterion -> IBI CV.

    compute_mi_eeg_native(eeg_raw, fs) -> dict
        Tort 2010 single-channel PAC on raw EEG:
        SO phase from [0.5, 1.5] Hz bandpass + Hilbert;
        spindle amplitude from [10, 14] Hz bandpass + Hilbert;
        18 phase bins, KL-MI normalized to [0, 1].
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks


# ===========================================================================
# T6: AASM SO UP event detection on raw EEG
# ===========================================================================

def compute_t6_eeg_aasm(eeg_raw, fs,
                        so_band=(0.2, 4.0),
                        half_wave_uv=75.0,
                        dur_lo_s=0.5,
                        dur_hi_s=2.0,
                        min_peak_dist_s=0.4):
    """
    AASM-style slow oscillation UP event detection on raw EEG.

    Parameters
    ----------
    eeg_raw : 1-D ndarray
        Detrended raw EEG in µV at `fs` Hz.
    fs : float
        Sampling rate.
    so_band : tuple
        Butterworth bandpass for SO (Hz). 0.2-4.0 default; widened from
        AASM 0.5-2.0 to suppress filter group delay near band edges.
    half_wave_uv : float
        Min peak-to-trough amplitude per AASM 2017 §IV.D.2.
    dur_lo_s, dur_hi_s : float
        Min/max DOWN→UP transition duration.
    min_peak_dist_s : float
        Min spacing between consecutive extrema (find_peaks `distance`).

    Returns
    -------
    dict with keys:
        ibi_cv         : float, std / mean of UP-event IBIs (sec)
                         999.0 if fewer than 3 valid UP events
        n_up_events    : int, number of UP events passing all criteria
        n_neg_peaks    : int, number of DOWN extrema detected
        n_pos_peaks    : int, number of UP extrema detected
        mean_ibi_s     : float, mean inter-burst interval
        ok             : bool, True iff ibi_cv is a real number (not 999)
    """
    out = {
        "ibi_cv":      999.0,
        "n_up_events": 0,
        "n_neg_peaks": 0,
        "n_pos_peaks": 0,
        "mean_ibi_s":  0.0,
        "ok":          False,
    }
    if len(eeg_raw) < int(3 * fs):
        return out
    if np.std(eeg_raw) < 1e-6:
        return out

    sos = butter(4, so_band, btype="band", fs=fs, output="sos")
    eeg_so = sosfiltfilt(sos, eeg_raw)

    dist = int(min_peak_dist_s * fs)
    neg_peaks, _ = find_peaks(-eeg_so, distance=dist)
    pos_peaks, _ = find_peaks( eeg_so, distance=dist)
    out["n_neg_peaks"] = int(len(neg_peaks))
    out["n_pos_peaks"] = int(len(pos_peaks))

    if len(neg_peaks) < 2 or len(pos_peaks) < 2:
        return out

    # Pair each DOWN with the next UP, check duration + amplitude
    up_times = []
    j = 0
    for t_dn in neg_peaks:
        while j < len(pos_peaks) and pos_peaks[j] <= t_dn:
            j += 1
        if j >= len(pos_peaks):
            break
        t_up = pos_peaks[j]
        dur_s = (t_up - t_dn) / fs
        if dur_s < dur_lo_s or dur_s > dur_hi_s:
            continue
        amp = eeg_so[t_up] - eeg_so[t_dn]
        if amp < half_wave_uv:
            continue
        up_times.append(t_up / fs)

    up_times = np.array(up_times, dtype=float)
    out["n_up_events"] = int(len(up_times))

    if len(up_times) < 3:
        return out

    ibis = np.diff(up_times)
    if ibis.mean() < 1e-9:
        return out
    out["mean_ibi_s"] = float(ibis.mean())
    out["ibi_cv"]     = float(ibis.std() / ibis.mean())
    out["ok"]         = True
    return out


# ===========================================================================
# MI: Tort 2010 single-channel PAC on raw EEG
# ===========================================================================

def compute_mi_eeg_native(eeg_raw, fs,
                          phase_band=(0.5, 1.5),
                          amp_band=(10.0, 14.0),
                          n_phase_bins=18,
                          sig_for_amp=None):
    """
    Tort 2010 KL Modulation Index between SO phase and spindle amplitude,
    both derived from the same raw EEG channel.

    Parameters
    ----------
    eeg_raw : 1-D ndarray
        Raw EEG (µV) at `fs` Hz. Should be detrended; no further preprocessing.
    fs : float
        Sampling rate.
    phase_band : tuple
        SO bandpass for phase extraction. 0.5-1.5 Hz (Tort 2010 / Helfrich 2018).
    amp_band : tuple
        Spindle bandpass for amplitude extraction. 10-14 Hz; MUST match the
        simulator-side spindle band (V7 SPINDLE_LO/HI) for cross-comparison.
    n_phase_bins : int
        Tort 2010 standard: 18 bins from -π to π.

    Returns
    -------
    dict with keys:
        mi               : float, KL-MI in [0, 1] (normalized by log(n_bins))
        preferred_phase  : float, bin center with max mean amplitude (rad)
        up_down_ratio    : float, sum(amp at |phi|<=pi/2) / sum(amp at |phi|>pi/2)
                           >1 = UP-locked, <1 = DOWN-locked. Same convention
                           as compute_pac_metrics_fixed.up_down_ratio.
        n_samples        : int, number of valid (phase, amp) sample pairs
        mean_amp_per_bin : ndarray (n_phase_bins,), unnormalized
        ok               : bool, True iff MI computed without degenerate bin
    """
    out = {
        "mi":               0.0,
        "preferred_phase":  np.pi,
        "up_down_ratio":    0.0,
        "n_samples":        0,
        "mean_amp_per_bin": np.zeros(n_phase_bins),
        "ok":               False,
    }
    if len(eeg_raw) < int(3 * fs):
        return out
    if np.std(eeg_raw) < 1e-6:
        return out

    try:
        sos_p = butter(4, phase_band, btype="band", fs=fs, output="sos")
        sos_a = butter(4, amp_band,   btype="band", fs=fs, output="sos")
        eeg_for_amp = eeg_raw if sig_for_amp is None else sig_for_amp
        phase = np.angle(hilbert(sosfiltfilt(sos_p, eeg_raw)))
        amp   = np.abs  (hilbert(sosfiltfilt(sos_a, eeg_for_amp)))

        bins = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
        mean_amp_per_bin = np.zeros(n_phase_bins)
        for k in range(n_phase_bins):
            mask = (phase >= bins[k]) & (phase < bins[k + 1])
            if mask.any():
                mean_amp_per_bin[k] = amp[mask].mean()

        total = mean_amp_per_bin.sum()
        if total < 1e-12 or (mean_amp_per_bin > 0).sum() < 2:
            # Degenerate: all amp mass in one bin or essentially zero
            return out

        p = mean_amp_per_bin / total
        # Use only nonzero bins for entropy (log 0 guard)
        p_nz = p[p > 0]
        H    = -np.sum(p_nz * np.log(p_nz))
        mi   = (np.log(n_phase_bins) - H) / np.log(n_phase_bins)

        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # up_down_ratio: sample-level (not bin-averaged) to match
        # compute_pac_metrics_fixed semantics.
        up_mask   = np.abs(phase) <= np.pi / 2
        down_mask = ~up_mask
        up_amp   = amp[up_mask].sum()
        down_amp = amp[down_mask].sum()
        up_down_ratio = float(up_amp / down_amp) if down_amp > 1e-12 else 0.0

        out["mi"]               = float(mi)
        out["preferred_phase"]  = float(bin_centers[np.argmax(mean_amp_per_bin)])
        out["up_down_ratio"]    = up_down_ratio
        out["n_samples"]        = int(len(phase))
        out["mean_amp_per_bin"] = mean_amp_per_bin
        out["ok"]               = True
    except Exception:
        pass
    return out


# ===========================================================================
# Surrogate-normalized MI z-score (Tort 2010 canonical surrogate)
# ===========================================================================

def compute_mi_zscore(sig_phase, fs,
                      sig_amp=None,
                      n_surrogates=100,
                      phase_band=(0.5, 1.5),
                      amp_band=(10.0, 14.0),
                      n_phase_bins=18,
                      min_shift_s=10.0,
                      seed=42):
    """
    Surrogate-normalized PAC z-score using circular amplitude time-shift
    (Tort 2010 Methods §"statistical evaluation of MI").

    For each surrogate:
        amp_surr = np.roll(amp, random_shift)
        MI_surr  = Tort_KL(phase, amp_surr)

    The shift is uniform in [min_shift_s * fs, len - min_shift_s * fs] samples,
    large enough to fully break SO–spindle phase alignment (canonical PAC
    timescale ≈ 1 sec; shifting by ≥ 10 sec destroys any coherence).

    z = (MI_observed - mean(MI_null)) / std(MI_null)

    Under H0 (no coupling), MI_observed ≈ MI_null → z ≈ 0.
    Under H1 (real coupling), MI_observed > MI_null → z >> 0.
    Conventional threshold: |z| > 2 (one-sided p < 0.025) for significance.

    Parameters
    ----------
    sig_phase : 1-D ndarray  — signal providing SO phase
    fs        : float        — sampling rate
    sig_amp   : 1-D ndarray  — signal providing spindle amplitude.
                               None → same as sig_phase (single-channel PAC).
                               Different → cross-channel PAC (sim r_ctx × r_thal).
    n_surrogates : int       — number of surrogate iterations (Tort 2010: 200)
    min_shift_s  : float     — minimum |shift| in seconds to apply
    seed         : int       — RNG seed for reproducibility

    Returns
    -------
    dict with keys:
        mi_observed     : float, true Tort MI
        mi_null_mean    : float, mean of n_surrogates surrogate MIs
        mi_null_std     : float, std of surrogate MIs
        z               : float, (mi_observed - mi_null_mean) / mi_null_std
        n_surrogates    : int
        null_dist       : ndarray (n_surrogates,), all surrogate MIs
        ok              : bool, True iff observed and surrogates all computed
    """
    out = {
        "mi_observed":   0.0,
        "mi_null_mean":  0.0,
        "mi_null_std":   0.0,
        "z":             0.0,
        "n_surrogates":  0,
        "null_dist":     np.zeros(n_surrogates),
        "ok":            False,
    }

    if sig_amp is None:
        sig_amp = sig_phase
    if len(sig_phase) != len(sig_amp):
        return out
    if len(sig_phase) < int(3 * fs):
        return out

    sos_p = butter(4, phase_band, btype="band", fs=fs, output="sos")
    sos_a = butter(4, amp_band,   btype="band", fs=fs, output="sos")
    phase = np.angle(hilbert(sosfiltfilt(sos_p, sig_phase)))
    amp   = np.abs  (hilbert(sosfiltfilt(sos_a, sig_amp)))

    bins = np.linspace(-np.pi, np.pi, n_phase_bins + 1)

    def _mi_from_pa(phase_arr, amp_arr):
        mean_amp = np.zeros(n_phase_bins)
        for k in range(n_phase_bins):
            mask = (phase_arr >= bins[k]) & (phase_arr < bins[k + 1])
            if mask.any():
                mean_amp[k] = amp_arr[mask].mean()
        total = mean_amp.sum()
        if total < 1e-12 or (mean_amp > 0).sum() < 2:
            return 0.0
        p_arr = mean_amp / total
        p_nz  = p_arr[p_arr > 0]
        H     = -np.sum(p_nz * np.log(p_nz))
        return float((np.log(n_phase_bins) - H) / np.log(n_phase_bins))

    mi_obs = _mi_from_pa(phase, amp)
    out["mi_observed"] = mi_obs

    rng = np.random.default_rng(seed)
    min_shift = int(min_shift_s * fs)
    max_shift = len(amp) - min_shift
    if max_shift <= min_shift:
        # signal too short for valid surrogate
        return out

    null_vals = np.zeros(n_surrogates)
    for i in range(n_surrogates):
        shift = int(rng.integers(min_shift, max_shift))
        amp_shifted = np.roll(amp, shift)
        null_vals[i] = _mi_from_pa(phase, amp_shifted)

    out["null_dist"]    = null_vals
    out["mi_null_mean"] = float(null_vals.mean())
    out["mi_null_std"]  = float(null_vals.std())
    out["n_surrogates"] = n_surrogates
    if null_vals.std() > 1e-12:
        out["z"] = float((mi_obs - null_vals.mean()) / null_vals.std())
    out["ok"] = True
    return out


# ===========================================================================
# Smoke test (run as script): expect non-trivial outputs on coupled signal
# ===========================================================================

if __name__ == "__main__":
    np.random.seed(0)
    fs = 1000.0
    t  = np.arange(0, 30, 1 / fs)

    # Synthetic SO + spindle bursts at SO UP peaks
    so   = 60.0 * np.sin(2 * np.pi * 0.8 * t)
    so_phase = np.angle(hilbert(butter_filt := sosfiltfilt(
        butter(4, [0.5, 1.5], btype="band", fs=fs, output="sos"), so)))
    sp_carrier = np.sin(2 * np.pi * 12 * t)
    amp_env    = np.exp(2.0 * np.cos(so_phase))  # heavy UP-locked modulation
    spindle    = 8.0 * sp_carrier * amp_env / amp_env.max()
    sig        = so + spindle + 2.0 * np.random.randn(len(t))

    t6  = compute_t6_eeg_aasm(sig, fs)
    print(f"[smoke] T6  ibi_cv={t6['ibi_cv']:.3f}  "
          f"n_up={t6['n_up_events']}  ok={t6['ok']}")

    mi_res = compute_mi_eeg_native(sig, fs)
    print(f"[smoke] MI  mi={mi_res['mi']:.4f}  "
          f"preferred_phase={mi_res['preferred_phase']:.2f}  ok={mi_res['ok']}")
