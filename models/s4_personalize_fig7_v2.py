"""
s4_personalize_fig7_v3.py
=========================
Physics-constrained thalamocortical personalisation — Version 3.

Key changes vs v2:
  - dynamics_score redesigned with 4 sub-tests (was 3):
      T1  DOWN state exists:         min(r_E) < 1 Hz            (0.20)
      T2  UP state exists:           max(r_E) > 15 Hz           (0.25)  ← NEW (was absent)
      T3  UP state sustained:        longest run > 15 Hz > 100ms (0.25)  ← NEW
      T4  SO frequency in range:     PSD peak 0.3–1.5 Hz        (0.15)  (was 0.25)
      T5  Spindle width (waxing):    FWHM > 2 Hz                (0.15)  (was 0.25)
    Total still sums to 1.00; dynamics weight in fitness unchanged (0.35).
  - fitness = 0.35*shape_r + 0.15*so_power + 0.15*spindle_power + 0.35*dynamics
  - Parameter bounds: same as v2 (c_ctx2th ≤ 0.05 enforced).
  - Output files tagged _v3.

Why the UP-state tests matter:
  v2 could reach dynamics=1.0 by staying in a persistent DOWN state
  (min r_E ≈ 0 satisfies T1; SO "peak" from noise satisfies T4; thalamic
  limit-cycle satisfies T5). The new T2+T3 require genuine UP/DOWN
  alternation with cortical firing rates ≥ 15 Hz during UP states, matching
  the 20–30 Hz range visible in Fig. 7(c) of Cakan et al. 2023.

Usage (from project root):
  python models/s4_personalize_fig7_v3.py

Outputs:
  data/patient_params_fig7_v3_SC4001.json
  outputs/evolution_fig7_v3_records.csv
"""

import os
import sys
import json
import time
import fnmatch
import warnings
warnings.filterwarnings("ignore")

# ── NumPy ≥1.24 monkey-patch (must come BEFORE importing neurolib) ────────────
import numpy as np
for _attr in ("object", "bool", "int", "float", "complex", "str"):
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(__builtins__, _attr, object))

import pandas as pd
from scipy.signal import welch, find_peaks
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.optimize import differential_evolution
import mne
mne.set_log_level("WARNING")

from neurolib.models.multimodel import MultiModel, ALNNode, ThalamicNode
from neurolib.models.multimodel.builder.base.network import Network
from neurolib.models.multimodel.builder.base.constants import EXC, INH

try:
    from fooof import FOOOF
    HAS_FOOOF = True
except ImportError:
    HAS_FOOOF = False
    print("[warn] fooof not installed; shape_r will use freq-weighted chi2 fallback")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUBJECT_ID      = "SC4001"
EEG_CHANNELS    = ["EEG Fpz-Cz", "EEG Pz-Oz"]
N3_LABELS       = ["Sleep stage 3", "Sleep stage 4"]
ARTIFACT_THRESH = 200e-6
EPOCH_DURATION  = 30.0
F_LO, F_HI     = 0.5, 20.0
FS_SIM          = 1000.0
SIM_DUR_MS      = 30_000        # keep same as v2 for fair comparison
DE_POPSIZE      = 20            # 20 × 8 params = 160 individuals/gen
N_GEN           = 30            # total evals ≈ 4960

PARAMS_PATH  = f"data/patient_params_fig7_v3_{SUBJECT_ID}.json"
RECORDS_PATH = "outputs/evolution_fig7_v3_records.csv"

# ── Parameter bounds (same as v2) ─────────────────────────────────────────────
PARAM_NAMES = ["mue", "mui", "b", "tauA", "g_LK", "g_h", "c_th2ctx", "c_ctx2th"]
BOUNDS = [
    (2.5,  4.5),    # mue
    (2.5,  5.0),    # mui
    (10.0, 40.0),   # b   [pA]
    (800., 2000.),  # tauA [ms]
    (0.03, 0.15),   # g_LK [mS/cm²]
    (0.03, 0.15),   # g_h  [mS/cm²]
    (0.05, 0.25),   # c_th2ctx
    (0.001, 0.05),  # c_ctx2th  ← physics-constrained upper bound
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ThalamoCortical network (identical to v2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def set_params_glob(model, pattern, value):
    for k in (k for k in model.params if fnmatch.fnmatch(k, pattern)):
        model.params[k] = value


class ThalamoCorticalNetwork(Network):
    name            = "Thalamocortical Motif"
    label           = "TCNet"
    sync_variables  = ["network_exc_exc", "network_exc_exc_sq", "network_inh_exc"]
    default_output  = f"r_mean_{EXC}"
    output_vars     = [f"r_mean_{EXC}", f"r_mean_{INH}"]
    _EXC_WITHIN_IDX = [6, 9]

    def __init__(self, c_th2ctx=0.02, c_ctx2th=0.02):
        aln = ALNNode(exc_seed=42, inh_seed=42)
        th  = ThalamicNode()
        aln.index = 0;  aln.idx_state_var = 0
        th.index  = 1;  th.idx_state_var  = aln.num_state_variables
        for i, node in enumerate([aln, th]):
            for mass in node:
                mass.noise_input_idx = [2 * i + mass.index]
        connectivity = np.array([[0.0, c_th2ctx], [c_ctx2th, 0.0]])
        super().__init__(
            nodes=[aln, th],
            connectivity_matrix=connectivity,
            delay_matrix=np.zeros_like(connectivity),
        )

    def _sync(self):
        couplings = sum([node._sync() for node in self], [])
        wi = self._EXC_WITHIN_IDX
        couplings += self._additive_coupling(wi, "network_exc_exc")
        couplings += self._additive_coupling(wi, "network_exc_exc_sq",
                                             connectivity=self.connectivity ** 2)
        couplings += self._additive_coupling(wi, "network_inh_exc")
        return couplings


def build_model(mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th,
                duration=SIM_DUR_MS):
    net = ThalamoCorticalNetwork(c_th2ctx=c_th2ctx, c_ctx2th=c_ctx2th)
    m = MultiModel(net)
    m.params["backend"]     = "numba"
    m.params["dt"]          = 0.1
    m.params["sampling_dt"] = 1.0
    m.params["duration"]    = duration
    set_params_glob(m, "*ALNMassEXC*.input_0.mu", mue)
    set_params_glob(m, "*ALNMassINH*.input_0.mu", mui)
    set_params_glob(m, "*ALNMassEXC*.b",          b)
    set_params_glob(m, "*ALNMassEXC*.tauA",       tauA)
    set_params_glob(m, "*ALNMassEXC*.a",          0.0)
    set_params_glob(m, "*ALNMass*.input_0.sigma",  0.05)
    set_params_glob(m, "*TCR*.input_0.sigma",      0.005)
    set_params_glob(m, "*.input_0.tau",            5.0)
    set_params_glob(m, "*TRN*.g_LK",              0.1)
    set_params_glob(m, "*TCR*.g_LK",              g_lk)
    set_params_glob(m, "*TCR*.g_h",               g_h)
    return m


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# V3 dynamics score  ← CORE CHANGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sub-test weights must sum to 1.0
_T1_WEIGHT = 0.20   # DOWN state exists
_T2_WEIGHT = 0.25   # UP state exists (peak > 15 Hz)
_T3_WEIGHT = 0.25   # UP state sustained (longest run > 15 Hz ≥ 100 ms)
_T4_WEIGHT = 0.15   # SO peak frequency in [0.3, 1.5] Hz
_T5_WEIGHT = 0.15   # Spindle FWHM > 2 Hz (waxing-waning, not limit cycle)

# Thresholds
UP_THRESH_HZ     = 15.0    # Hz — minimum peak firing rate for a real UP state
UP_DURATION_MS   = 100.0   # ms  — minimum sustained UP state duration
DOWN_THRESH_HZ   = 1.0     # Hz — firing rate below this = DOWN state
SO_FREQ_LO       = 0.3     # Hz
SO_FREQ_HI       = 1.5     # Hz
SPINDLE_LO       = 8.0     # Hz
SPINDLE_HI       = 16.0    # Hz
SPINDLE_FWHM_MIN = 2.0     # Hz


def compute_dynamics_score_v3(r_ctx, r_thal, fs=FS_SIM):
    """
    Four-part dynamics check for genuine N3 slow oscillations.

    Parameters
    ----------
    r_ctx  : 1D array, cortex EXC firing rate [Hz], burn-in already removed
    r_thal : 1D array, thalamus TCR firing rate [Hz], same length as r_ctx
    fs     : float, sampling rate [Hz]

    Returns
    -------
    score        : float in [0, 1]
    details      : dict with per-test bool + diagnostic values
    """
    details = {}
    score   = 0.0

    # ── T1: DOWN state exists ─────────────────────────────────────────────────
    min_rE = float(r_ctx.min())
    t1     = min_rE < DOWN_THRESH_HZ
    details["T1_down_exists"]  = t1
    details["T1_min_rE_Hz"]    = min_rE
    if t1:
        score += _T1_WEIGHT

    # ── T2: UP state exists ───────────────────────────────────────────────────
    # Require at least one sample to exceed UP_THRESH_HZ.
    # This rejects persistent-DOWN-state solutions that v2 could not reject.
    max_rE = float(r_ctx.max())
    t2     = max_rE > UP_THRESH_HZ
    details["T2_up_exists"]    = t2
    details["T2_max_rE_Hz"]    = max_rE
    if t2:
        score += _T2_WEIGHT

    # ── T3: UP state sustained ────────────────────────────────────────────────
    # Require a continuous run of ≥ UP_DURATION_MS ms above UP_THRESH_HZ.
    # Prevents isolated noise spikes from passing T2.
    min_run_samples = int(UP_DURATION_MS * fs / 1000.0)
    above           = (r_ctx > UP_THRESH_HZ).astype(np.int8)
    # Vectorised run-length encoding
    diff   = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    max_run = int((ends - starts).max()) if len(starts) > 0 else 0
    t3      = max_run >= min_run_samples
    details["T3_up_sustained"]        = t3
    details["T3_longest_up_run_ms"]   = max_run * 1000.0 / fs
    if t3:
        score += _T3_WEIGHT

    # ── T4: SO frequency in range ─────────────────────────────────────────────
    nperseg = min(int(5 * fs), len(r_ctx))
    f_c, p_c = welch(r_ctx, fs=fs, nperseg=nperseg,
                     noverlap=nperseg // 2, window="hann")
    so_mask  = (f_c >= SO_FREQ_LO) & (f_c <= SO_FREQ_HI)
    if so_mask.any():
        so_peak_freq = float(f_c[so_mask][np.argmax(p_c[so_mask])])
    else:
        so_peak_freq = 0.0
    t4 = SO_FREQ_LO <= so_peak_freq <= SO_FREQ_HI
    details["T4_so_in_range"]      = t4
    details["T4_so_peak_freq_Hz"]  = so_peak_freq
    if t4:
        score += _T4_WEIGHT

    # ── T5: Thalamus spindle FWHM > 2 Hz (waxing-waning) ─────────────────────
    nperseg_th = min(int(5 * fs), len(r_thal))
    f_th, p_th = welch(r_thal, fs=fs, nperseg=nperseg_th,
                       noverlap=nperseg_th // 2, window="hann")
    sp_mask = (f_th >= SPINDLE_LO) & (f_th <= SPINDLE_HI)
    fwhm    = 0.0
    if sp_mask.any() and p_th[sp_mask].max() > 0:
        p_sp        = p_th[sp_mask]
        f_sp        = f_th[sp_mask]
        half_power  = p_sp.max() / 2.0
        above_half  = f_sp[p_sp >= half_power]
        if len(above_half) >= 2:
            fwhm = float(above_half[-1] - above_half[0])
    t5 = fwhm > SPINDLE_FWHM_MIN
    details["T5_spindle_waxing"]  = t5
    details["T5_spindle_fwhm_Hz"] = fwhm
    if t5:
        score += _T5_WEIGHT

    return round(score, 4), details


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 1: Load target EEG (runs once before evolution)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_target_psd():
    # manifest.csv is often saved by Excel as UTF-16 (BOM FF FE). Try UTF-8 first, then fall back.
    try:
        manifest = pd.read_csv("data/manifest.csv", encoding="utf-8")
    except UnicodeDecodeError:
        manifest = pd.read_csv("data/manifest.csv", encoding="utf-16")
    subj_row = manifest[manifest["subject_id"] == SUBJECT_ID].iloc[0]
    raw = mne.io.read_raw_edf(
        subj_row["psg_path"], include=EEG_CHANNELS, preload=True, verbose=False
    )
    fs_eeg = raw.info["sfreq"]
    raw.set_annotations(mne.read_annotations(subj_row["hypnogram_path"]))
    event_id  = {lbl: idx + 1 for idx, lbl in enumerate(N3_LABELS)}
    events, event_dict = mne.events_from_annotations(
        raw, event_id=event_id, verbose=False
    )
    epochs_n3 = mne.Epochs(
        raw, events, event_id=event_dict,
        tmin=0.0, tmax=EPOCH_DURATION,
        baseline=None, preload=True, verbose=False,
    )
    psds, f_ep, freq_mask = [], None, None
    for ep_idx in range(len(epochs_n3)):
        data = epochs_n3[ep_idx].get_data()[0]
        if np.any((data.max(axis=1) - data.min(axis=1)) > ARTIFACT_THRESH):
            continue
        mean_sig = data.mean(axis=0)
        nperseg  = min(int(10.0 * fs_eeg), len(mean_sig))
        f_ep, p_ep = welch(mean_sig, fs=fs_eeg, nperseg=nperseg,
                           noverlap=nperseg // 2, window="hann")
        freq_mask = (f_ep >= F_LO) & (f_ep <= F_HI)
        psds.append(p_ep[freq_mask])

    if not psds:
        raise RuntimeError("No N3 epochs passed QC for subject " + SUBJECT_ID)

    target_psd   = np.mean(psds, axis=0)
    target_freqs = f_ep[freq_mask]
    print(f"  Target EEG: {len(psds)} N3 epochs used")
    return target_psd, target_freqs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 2: Pre-compute FOOOF baseline on target (runs once)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_target_periodic(target_psd, target_freqs):
    if not HAS_FOOOF:
        return None, None

    fm = FOOOF(peak_width_limits=[1.0, 8.0], max_n_peaks=4,
               min_peak_height=0.05, aperiodic_mode="fixed")
    fm.fit(target_freqs, target_psd, [F_LO, F_HI])
    fooof_freqs     = fm.freqs
    target_log      = np.log10(target_psd + 1e-30)
    target_periodic = target_log[:len(fm._ap_fit)] - fm._ap_fit

    print(f"  FOOOF target peaks:")
    for pk in fm.peak_params_:
        print(f"    {pk[0]:.1f} Hz  power={pk[1]:.3f}  width={pk[2]:.1f}")
    print(f"  Aperiodic exponent: {fm.aperiodic_params_[1]:.2f}")

    return target_periodic, fooof_freqs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 3: Fitness function (called ~4960 times during evolution)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_eval_count  = 0
_best_score  = -np.inf
_best_params = {}
_records     = []
_t_start     = None


def compute_fitness_v3(params_vec,
                       target_psd, target_freqs,
                       target_periodic, fooof_freqs):
    """
    Returns negative fitness (scipy minimises).

    Fitness = 
    """
    global _eval_count, _best_score, _best_params, _records

    _eval_count += 1
    mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th = params_vec

    # ── Run simulation ────────────────────────────────────────────────────────
    try:
        m = build_model(mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th)
        m.run()
    except Exception:
        return 0.0   # return 0 so DE treats it as worst-case (we maximise by negating)

    # ── Extract cortex + thalamus firing rates ────────────────────────────────
    r_exc = m[f"r_mean_{EXC}"]
    if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
        r_ctx  = r_exc[0, :] * 1000.0
        r_thal = r_exc[1, :] * 1000.0
    else:
        r_ctx  = (r_exc[0] if r_exc.ndim == 2 else r_exc) * 1000.0
        r_thal = np.zeros_like(r_ctx)

    # Burn-in: discard first 5 s
    n_drop = int(5.0 * FS_SIM)
    r_ctx  = r_ctx[n_drop:]
    r_thal = r_thal[n_drop:]

    if r_ctx.max() < 0.1:
        return 0.0   # cortex silent after burn-in

    # ── Cortex PSD ───────────────────────────────────────────────────────────
    nperseg = min(int(10.0 * FS_SIM), len(r_ctx))
    f_ctx, p_ctx = welch(r_ctx, fs=FS_SIM, nperseg=nperseg,
                         noverlap=nperseg // 2, window="hann")
    mask = (f_ctx >= F_LO) & (f_ctx <= F_HI)
    f_ctx, p_ctx = f_ctx[mask], p_ctx[mask]

    # ── shape_r (FOOOF path identical to v2) ─────────────────────────────────
    shape_r = 0.0
    if HAS_FOOOF and target_periodic is not None:
        try:
            # Downsample to target FOOOF grid, then FOOOF
            p_interp = interp1d(f_ctx, p_ctx,
                                bounds_error=False, fill_value=1e-30)(fooof_freqs)
            fm_sim = FOOOF(peak_width_limits=[1.0, 8.0], max_n_peaks=4,
                           min_peak_height=0.05, aperiodic_mode="fixed")
            fm_sim.fit(fooof_freqs, p_interp, [F_LO, F_HI])
            sim_log      = np.log10(p_interp[:len(fm_sim._ap_fit)] + 1e-30)
            sim_periodic = sim_log - fm_sim._ap_fit
            n_r          = min(len(sim_periodic), len(target_periodic))
            shape_r, _   = pearsonr(sim_periodic[:n_r], target_periodic[:n_r])
            shape_r      = max(shape_r, 0.0)
        except Exception:
            shape_r = 0.0
    else:
        # Fallback: frequency-weighted chi-squared (same as v2 fallback)
        p_interp_fb = interp1d(f_ctx, p_ctx,
                               bounds_error=False, fill_value=1e-30)(target_freqs)
        weights  = target_freqs ** (-0.5)
        weights /= weights.sum()
        chi2     = np.sum(weights * (np.log10(p_interp_fb + 1e-30)
                                    - np.log10(target_psd + 1e-30)) ** 2)
        shape_r  = float(np.exp(-chi2))

    # ── so_power and spindle_power (same FOOOF logic as v2) ──────────────────
    so_power      = 0.0
    spindle_power = 0.0
    if HAS_FOOOF:
        try:
            p_interp2 = interp1d(f_ctx, p_ctx,
                                 bounds_error=False, fill_value=1e-30)(fooof_freqs)
            fm2 = FOOOF(peak_width_limits=[1.0, 8.0], max_n_peaks=4,
                        min_peak_height=0.05, aperiodic_mode="fixed")
            fm2.fit(fooof_freqs, p_interp2, [F_LO, F_HI])
            for pk in fm2.peak_params_:
                freq, power, _ = pk
                if SO_FREQ_LO <= freq <= SO_FREQ_HI:
                    so_power = max(so_power, float(power))
                if SPINDLE_LO <= freq <= SPINDLE_HI:
                    spindle_power = max(spindle_power, float(power))
        except Exception:
            pass

    # ── V3 dynamics score ─────────────────────────────────────────────────────
    dynamics_score, dyn_details = compute_dynamics_score_v3(r_ctx, r_thal)

    # ── Combined fitness ─────────────────────────────────────────────────────
    fitness = (0.35 * shape_r
               + 0.15 * so_power
               + 0.15 * spindle_power
               + 0.35 * dynamics_score)

    # ── Book-keeping ──────────────────────────────────────────────────────────
    record = dict(zip(PARAM_NAMES, params_vec))
    record.update({
        "score":          round(fitness, 6),
        "shape_r":        round(shape_r, 6),
        "so_power":       round(so_power, 6),
        "spindle_power":  round(spindle_power, 6),
        "dynamics_score": round(dynamics_score, 6),
        # Per-sub-test detail
        "T1_down":        int(dyn_details["T1_down_exists"]),
        "T2_up":          int(dyn_details["T2_up_exists"]),
        "T3_sustained":   int(dyn_details["T3_up_sustained"]),
        "T4_so_freq":     int(dyn_details["T4_so_in_range"]),
        "T5_spindle":     int(dyn_details["T5_spindle_waxing"]),
        "max_rE":         round(dyn_details["T2_max_rE_Hz"], 3),
        "longest_up_ms":  round(dyn_details["T3_longest_up_run_ms"], 1),
        "so_peak_hz":     round(dyn_details["T4_so_peak_freq_Hz"], 3),
        "spindle_fwhm":   round(dyn_details["T5_spindle_fwhm_Hz"], 3),
        "eval":           _eval_count,
    })
    _records.append(record)

    if fitness > _best_score:
        _best_score  = fitness
        _best_params = record.copy()

    return -fitness   # scipy minimises


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Evolution callback (printed after every generation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_gen = 0

def _callback(xk, convergence):
    global _gen
    _gen += 1
    elapsed = time.time() - _t_start
    bp = _best_params
    print(
        f"  Gen {_gen:2d}/{N_GEN}  best={_best_score:+.4f}"
        f"  shape_r={bp.get('shape_r', 0):.3f}"
        f"  dyn={bp.get('dynamics_score', 0):.2f}"
        f"  [T1={bp.get('T1_down',0)} T2={bp.get('T2_up',0)}"
        f"   T3={bp.get('T3_sustained',0)} T4={bp.get('T4_so_freq',0)}"
        f"   T5={bp.get('T5_spindle',0)}]"
        f"  max_rE={bp.get('max_rE', 0):.1f}Hz"
        f"  evals={_eval_count}"
        f"  elapsed={elapsed:.0f}s"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    global _t_start

    print("=" * 65)
    print("V3 Evolution: physics-constrained N3 fitting")
    print("  FOOOF=" + ("ON" if HAS_FOOOF else "OFF (fallback)"))
    print(f"  8 params, bounds constrained for bistable dynamics")
    print(f"  Key: c_ctx2th upper = 0.05 (was 0.30 in v1)")
    print(f"  NEW: T2+T3 UP-state tests (threshold = {UP_THRESH_HZ} Hz, "
          f"duration ≥ {UP_DURATION_MS} ms)")
    print(f"  Fitness: 0.35*shape_r + 0.15*SO + 0.15*spindle + 0.35*dynamics")
    print(f"  dynamics weights: T1={_T1_WEIGHT} T2={_T2_WEIGHT} "
          f"T3={_T3_WEIGHT} T4={_T4_WEIGHT} T5={_T5_WEIGHT}")
    print(f"  popsize={DE_POPSIZE} x {len(PARAM_NAMES)} = "
          f"{DE_POPSIZE*len(PARAM_NAMES)} individuals/gen")
    print(f"  maxiter={N_GEN}, total evals ~ "
          f"{DE_POPSIZE*len(PARAM_NAMES)*(N_GEN+1)}")
    print("=" * 65)

    # Load target
    print("\nLoading target EEG...")
    target_psd, target_freqs = load_target_psd()
    print("Computing FOOOF on target...")
    target_periodic, fooof_freqs = compute_target_periodic(target_psd, target_freqs)

    # Run evolution
    _t_start = time.time()
    print("\n")

    result = differential_evolution(
        compute_fitness_v3,
        bounds=BOUNDS,
        args=(target_psd, target_freqs, target_periodic, fooof_freqs),
        strategy="best1bin",
        maxiter=N_GEN,
        popsize=DE_POPSIZE,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        callback=_callback,
        polish=False,
        workers=1,
    )

    total_time = time.time() - _t_start
    print(f"\nEvolution complete in {total_time/3600:.2f} h")

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs("data",    exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Best params JSON
    best = _best_params.copy()
    with open(PARAMS_PATH, "w") as fh:
        json.dump(best, fh, indent=2)
    print(f"Saved: {PARAMS_PATH}")

    # Evolution records CSV
    df = pd.DataFrame(_records)
    df.to_csv(RECORDS_PATH, index=False)
    print(f"Saved: {RECORDS_PATH}")

    # ── Validation printout ───────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("V3 Validation")
    print(f"{'='*55}")
    print(f"  score         = {best.get('score'):.4f}")
    print(f"  shape_r       = {best.get('shape_r'):.4f}")
    print(f"  so_power      = {best.get('so_power'):.4f}")
    print(f"  spindle_power = {best.get('spindle_power'):.4f}")
    print(f"  dynamics      = {best.get('dynamics_score'):.4f}")
    print(f"")
    print(f"  Sub-tests:")
    print(f"    T1 DOWN state (min r_E < {DOWN_THRESH_HZ} Hz):  "
          f"{'✓' if best.get('T1_down') else '✗'}")
    print(f"    T2 UP state (max r_E > {UP_THRESH_HZ} Hz):      "
          f"{'✓' if best.get('T2_up') else '✗'}  "
          f"[max_rE={best.get('max_rE'):.1f} Hz]")
    print(f"    T3 UP sustained (≥ {UP_DURATION_MS} ms):         "
          f"{'✓' if best.get('T3_sustained') else '✗'}  "
          f"[longest_up={best.get('longest_up_ms'):.1f} ms]")
    print(f"    T4 SO freq [{SO_FREQ_LO}-{SO_FREQ_HI} Hz]:           "
          f"{'✓' if best.get('T4_so_freq') else '✗'}  "
          f"[peak={best.get('so_peak_hz'):.2f} Hz]")
    print(f"    T5 spindle FWHM > {SPINDLE_FWHM_MIN} Hz:            "
          f"{'✓' if best.get('T5_spindle') else '✗'}  "
          f"[FWHM={best.get('spindle_fwhm'):.2f} Hz]")
    print(f"")
    print(f"  Best parameters:")
    for k in PARAM_NAMES:
        print(f"    {k}: {best.get(k):.4f}")
    print(f"\nNext: run python plot_scripts/plot_fig7_v2_fast.py")
    print(f"  (update PARAMS_PATH to point to _v3 files)")


if __name__ == "__main__":
    main()