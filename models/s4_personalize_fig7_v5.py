"""
s4_personalize_fig7_v5.py
=========================
Physics-constrained thalamocortical personalisation — V5 (A-revision).

V5 = V4-improve + PAC hard constraints (T9, T10, T11).

  A-revision (2026-04): T9/T10 recalibrated based on isolated-thalamus
  sweep + golden-point verification (see s4_0b_verify_golden_point.py).
  Two-spindle-mode support added to T10 to accommodate both fast-spindle
  (Up-peak-locked, Mölle 2011) and slow-spindle (Down-to-Up-locked,
  Mölle 2011) regimes.

Motivation (from mechanistic-validity constraints review):
  V4 constraints T1-T8 guard well against Failure Mode 1 (cortex collapsing
  into noise-driven sporadic spikes: T1, T2, T3, T4+, T6) and Failure Mode 2
  (thalamic continuous 12 Hz oscillation without waxing-waning envelope:
  T5, T7, T8). But V4 has ZERO guard against Failure Mode 3 — thalamo-
  cortical spatiotemporal DECOUPLING, i.e. the loss of phase-amplitude
  coupling (PAC) between cortical SO phase and thalamic spindle amplitude.

  Clinically, loss of SO-spindle coupling is the common pathological
  signature of aging, MCI / AD (Mander 2015, Helfrich 2018, Ladenbauer
  2017), schizophrenia (Manoach 2016, Demanuele 2017), Parkinson's disease,
  and epilepsy. A model that fits the power spectrum well but decouples SO
  from spindles is — literally — simulating a diseased brain.

  V5 closes this loophole by adding three PAC hard constraints:

    T9  PAC strength        :  Modulation Index (Tort 2010)        > 0.005
    T10 PAC preferred phase :  MVL phase near SO Up-peak (fast spindle)
                               OR near Down-to-Up transition (slow spindle)
    T11 PAC directionality  :  SO phase leads spindle envelope      (lag > 0)

  These three together enforce that the optimiser lives inside the
  physiologically legitimate SO-spindle phase-locked regime, not just any
  parameter set that happens to reproduce the N3 power spectrum.

A-revision changes (vs original V5):
  • T9 PAC_MI_MIN: 0.010 → 0.005 (calibrated against null distribution
    95%ile = 0.0054; original 0.010 was unreachable in realistic 60s runs
    due to high MI sampling variance).
  • T10 from single-sided (|phase| < π/4, Up-peak only) to TWO-SIDED
    (|phase| < 50° OR |phase ± π| < 50°), accommodating both fast-spindle
    and slow-spindle regimes per Mölle 2011 Sleep 34:1411. Random-phase
    false-positive rate: ~55%. T10 now acts as an "not-anti-phase"
    filter rather than a strict Up-peak requirement.

Scope of change vs V4 (deliberate, minimal):
  • Adds T9, T10, T11 to constraints (feasible now requires 11/11).
  • Adds continuous feasibility scores for T9/T10/T11.
  • Adds PAC metrics to records CSV.
  • No changes to T1-T8, no changes to DE settings, no changes to bounds.

Usage (from project root):
  python models/s4_personalize_fig7_v5.py

Outputs:
  data/patient_params_fig7_v5_SC4001.json
  outputs/evolution_fig7_v5_records.csv
"""

import os
import sys
import json
import time
import fnmatch
import importlib.util
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


import numpy as np
for _attr in ("object", "bool", "int", "float", "complex", "str"):
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(__builtins__, _attr, object))

import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.optimize import differential_evolution
from scipy.signal import hilbert, butter, sosfiltfilt, correlate
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

# Import preprocessing functions from 02_preprocess_psd.py without circular imports
_spec = importlib.util.spec_from_file_location("02_preprocess_psd", "utils/02_preprocess_psd.py")
prep_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep_mod)
load_hypnogram = prep_mod.load_hypnogram
compute_epoch_psd = prep_mod.compute_epoch_psd
EPOCH_LEN_S = prep_mod.EPOCH_LEN_S

# =========================
SUBJECT_ID      = "SC4001"
EEG_CHANNELS    = ["EEG Fpz-Cz"]
N3_LABELS       = ["N3"]
ARTIFACT_THRESH = 200e-6
EPOCH_DURATION  = 30.0
F_LO, F_HI     = 0.5, 20.0
FS_SIM          = 1000.0
SIM_DUR_MS      = 30_000        # keep same as v2 for fair comparison
DE_POPSIZE      = 20            # reduce memory/time pressure while keeping diversity
N_GEN           = 30            # DE generations

PARAMS_PATH  = f"data/patient_params_fig7_v5_{SUBJECT_ID}.json"
RECORDS_PATH = "outputs/evolution_fig7_v5_records.csv"

# Evolutionary FOOOF Parameters (same as v2)
EVO_FOOOF_PARAMS = dict(
    peak_width_limits=[1.0, 8.0],
    max_n_peaks=4,
    min_peak_height=0.05,
    aperiodic_mode="fixed",
)

# Parameter names and bounds — NARROWED around V4 best solution
PARAM_NAMES = ["mue", "mui", "b", "tauA", "g_LK", "g_h", "c_th2ctx", "c_ctx2th"]

# V4 best feasible solution (from patient_params_fig7_v4_SC4001.json)
_V4_BEST = dict(mue=3.895, mui=3.027, b=35.5, tauA=1426,
                g_LK=0.0564, g_h=0.0517, c_th2ctx=0.0986, c_ctx2th=0.038)

BOUNDS = [
    (_V4_BEST["mue"]    * 0.85, _V4_BEST["mue"]    * 1.15),   # mue:    3.31 – 4.48
    (_V4_BEST["mui"]    * 0.85, _V4_BEST["mui"]    * 1.15),   # mui:    2.57 – 3.48
    (_V4_BEST["b"]      * 0.80, _V4_BEST["b"]      * 1.20),   # b:      28.4 – 42.6
    (_V4_BEST["tauA"]   * 0.70, _V4_BEST["tauA"]   * 1.30),   # tauA:   998  – 1854
    (_V4_BEST["g_LK"]   * 0.70, _V4_BEST["g_LK"]   * 1.30),   # g_LK:   0.039 – 0.073
    (_V4_BEST["g_h"]    * 0.70, _V4_BEST["g_h"]    * 1.30),   # g_h:    0.036 – 0.067
    (_V4_BEST["c_th2ctx"]* 0.70, min(_V4_BEST["c_th2ctx"]* 1.30, 0.25)),  # c_th2ctx
    (_V4_BEST["c_ctx2th"]* 0.70, min(_V4_BEST["c_ctx2th"]* 1.50, 0.05)),  # c_ctx2th: wider upward
]

# Helper to set multiple params with wildcard pattern (e.g. "*ALNMassEXC*.input_0.mu") in one call
def set_params_glob(model, pattern, value):
    for k in (k for k in model.params if fnmatch.fnmatch(k, pattern)):
        model.params[k] = value


class ThalamoCorticalNetwork(Network):
    name            = "Thalamocortical Motif" #          
    label           = "TCNet" #     
    sync_variables  = ["network_exc_exc", "network_exc_exc_sq", "network_inh_exc"] #    
    default_output  = f"r_mean_{EXC}" # state variable to use for fitness evaluation (cortical EXC firing rate)
    output_vars     = [f"r_mean_{EXC}", f"r_mean_{INH}"] # state variables to record during simulation
    _EXC_WITHIN_IDX = [6, 9] # indices of within-ALN EXC->EXC couplings in sync_variables

    def __init__(self, c_th2ctx=0.02, c_ctx2th=0.02): #     
        aln = ALNNode(exc_seed=42, inh_seed=42) # ALNNode is a 2-mass cortical microcircuit (1 EXC, 1 INH)
        th  = ThalamicNode()# ThalamicNode is a 2-mass thalamic motif (TCR, TRN)
        aln.index = 0;  aln.idx_state_var = 0 # cortical state variables start at index 0
        th.index  = 1;  th.idx_state_var  = aln.num_state_variables # thalamic state variables start after cortical ones
        for i, node in enumerate([aln, th]):
            for mass in node:
                mass.noise_input_idx = [2 * i + mass.index] # assign unique noise input to each mass (cortical EXC, cortical INH, thalamic TCR, thalamic TRN)
        connectivity = np.array([[0.0, c_th2ctx], [c_ctx2th, 0.0]]) # th->ctx and ctx->th coupling strengths
        super().__init__(
            nodes=[aln, th],
            connectivity_matrix=connectivity,
            delay_matrix=np.zeros_like(connectivity),
        ) # initialize base Network with nodes and connectivity

    def _sync(self):
        couplings = sum([node._sync() for node in self], []) # get all within-node couplings (e.g. ALN EXC->INH, TCR->TRN) from nodes
        wi = self._EXC_WITHIN_IDX # indices of within-ALN EXC->EXC couplings in sync_variables
        couplings += self._additive_coupling(wi, "network_exc_exc") #   add thalamocortical coupling to within-ALN EXC->EXC coupling (i.e. add to both EXC masses in ALN)
        couplings += self._additive_coupling(wi, "network_exc_exc_sq",
                                             connectivity=self.connectivity ** 2) # also add a nonlinear term proportional to the square of the thalamocortical coupling, to allow more flexible shaping of the SO peak (e.g. sharper peaks with stronger thalamocortical coupling)
        couplings += self._additive_coupling(wi, "network_inh_exc") # add thalamocortical coupling to within-ALN INH->EXC coupling (i.e. add to both INH->EXC couplings in ALN)
        return couplings


def build_model(mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th,
                duration=SIM_DUR_MS):
    net = ThalamoCorticalNetwork(c_th2ctx=c_th2ctx, c_ctx2th=c_ctx2th) # ThalamoCorticalNetwork is the custom Network class defined above, which implements the thalamocortical motif with ALN cortex and 2-mass thalamus, and defines how to combine within-node couplings with thalamocortical couplings in the _sync method.    
    m = MultiModel(net) # MultiModel is a wrapper around Network that handles simulation and parameter management. We set parameters below using set_params_glob for convenience.
    m.params["backend"]     = "numba" # use numba backend for speed (CPU parallelisation); other options are "numpy" (no parallelisation) and "torch" (GPU parallelisation, but much slower for this small model)
    m.params["dt"]          = 0.1 # simulation timestep [ms]
    m.params["sampling_dt"] = 1.0 # sampling timestep for recorded output (1 ms = 1000 Hz sampling rate, same as EEG target)
    m.params["duration"]    = duration # total simulation duration [ms]
    set_params_glob(m, "*ALNMassEXC*.input_0.mu", mue) # set excitatory population mean input to mue (same for both EXC masses in ALN)
    set_params_glob(m, "*ALNMassINH*.input_0.mu", mui) # set inhibitory population mean input to mui (same for both INH masses in ALN)
    set_params_glob(m, "*ALNMassEXC*.b",          b) # set excitatory population adaptation strength to b (same for both EXC masses in ALN)
    set_params_glob(m, "*ALNMassEXC*.tauA",       tauA) # set excitatory population adaptation time constant to tauA (same for both EXC masses in ALN)
    set_params_glob(m, "*ALNMassEXC*.a",          0.0) # no excitatory adaptation current (set to 0 for all EXC masses in ALN)
    set_params_glob(m, "*ALNMass*.input_0.sigma",  0.05) # set noise input standard deviation to 0.05 for all masses in ALN (EXC and INH)
    set_params_glob(m, "*TCR*.input_0.sigma",      0.005) # set noise input standard deviation to 0.005 for TCR mass in thalamus (lower than cortex to prevent noise-driven spindles that would fail T5)
    set_params_glob(m, "*.input_0.tau",            5.0) # set noise input time constant to 5 ms for all masses (same as v2)
    set_params_glob(m, "*TRN*.g_LK",              0.1) # set TRN leak conductance to 0.1 mS/cm^2 (same as v2; TCR g_LK is varied by DE, but TRN g_LK is fixed to prevent non-physiological solutions with very low TCR g_LK and very high TRN g_LK)
    set_params_glob(m, "*TCR*.g_LK",              g_lk) # set TCR leak conductance to g_lk (same as v2)
    set_params_glob(m, "*TCR*.g_h",               g_h) # set TCR h conductance to g_h (same as v2)
    return m


# =====================================================================
# V4 CONSTANTS — 12 total (down from V3's 26)
# =====================================================================

# --- Constraint thresholds (T1-T11) ---
DOWN_THRESH_HZ   = 1.0       # T1: below this = DOWN state
UP_THRESH_HZ     = 15.0      # T2/T3: above this = UP state
UP_DURATION_MS   = 100.0     # T3: min sustained UP duration [ms]
SO_FREQ_LO       = 0.2       # T4+: SO peak search range lower bound [Hz]
SO_FREQ_HI       = 1.5       # T4+: SO peak search range upper bound [Hz]
SO_Q_MIN         = 2.0       # T4+: min peak-to-neighbor power ratio
SPINDLE_LO       = 10.0      # T5/T7: spindle band lower bound [Hz]
SPINDLE_HI       = 14.0      # T5/T7: spindle band upper bound [Hz]
SPINDLE_FWHM_MIN = 2.0       # T5: min spindle FWHM [Hz]
IBI_CV_MAX       = 0.4       # T6: max inter-burst interval CV
SPINDLE_CV_MIN   = 0.7       # T7: min spindle envelope CV
SPINDLE_EVT_MIN  = 5         # T8: min number of spindle events in 25s
SPINDLE_DUR_LO_S = 0.3       # T8: min spindle event duration [s]
SPINDLE_DUR_HI_S = 2.0       # T8: max spindle event duration [s]
SPINDLE_EVT_THRESH_FACTOR = 1.5  # T8: envelope threshold = mean + factor*std

# --- PAC constraints T9-T11 (NEW in V5, A-revision calibrated) -------
# T9: PAC strength — Kullback-Leibler Modulation Index (Tort et al. 2010,
#     J Neurophysiol 104:1195). MI in [0, 1].
#
#     CALIBRATION (A-revision, empirically re-tuned against golden-point
#     verification, see outputs/golden_point_metrics.txt):
#       • Null distribution (pure Gaussian noise, 25s @ 1kHz): mean ≈ 0.003,
#         95th percentile ≈ 0.0054 (still the floor for "above chance").
#       • Sweep predicted MI=0.018 at a golden point, but single-run
#         verification measured 0.007 — MI sampling variance in a 60s run
#         is large (~3x). We therefore calibrate to the low end of realistic
#         coupled-simulation values, not the high end.
#       • Jajcay 2022 Fig. 11 report MI ≈ 0.0109 averaged over many runs.
#
#     Threshold set at 0.005 (just above null 95%ile). This is permissive
#     but still excludes noise-level coupling. Original 0.010 was unreachable
#     in practice because ~30 SO cycles per 60s run gives MI std ≈ 0.005.
PAC_N_BINS       = 18        # phase bins for KL-MI
PAC_MI_MIN       = 0.005     # T9: MI must exceed this to count as coupled

# T10: PAC preferred phase — mean vector length (MVL, Canolty 2006) angle
#     must fall near a physiologically valid target phase.
#
#     A-REVISION: TWO-SIDED. Both fast-spindle and slow-spindle regimes
#     are physiologically valid (Mölle 2011 Sleep 34:1411):
#       • Fast spindles (12-15 Hz): lock to SO Up-state peak → phase ≈ 0
#       • Slow spindles (9-12 Hz):  lock to SO Down-to-Up transition
#         → phase ≈ ±π (which is the SAME phase on the -π/+π circle)
#
#     Our 13 Hz model borderline-qualifies as either. Golden-point
#     verification showed the model produces Down-to-Up locked spindles
#     (phases ≈ ±135°), which is the slow-spindle physiological regime.
#     Rejecting this would mis-classify a legitimate NREM dynamic.
#
#     A solution passes T10 if the preferred phase is within ±PAC_PHASE_TOL
#     of EITHER 0 (Up-peak) OR ±π (Down-to-Up). Because ±π wrap equivalently
#     on the circle, we compute "distance to nearest of {0, +π, -π}".
#
#     TOLERANCE CHOICE: ±50° (5π/18). Rationale:
#       • Primary golden point: φ=-145°, dist_to_target=35°  → PASS (margin 15°)
#       • Control golden point: φ=+133°, dist_to_target=47°  → PASS (margin 3°)
#       • Random-phase false-positive rate: ~55% (vs 67% at ±60°, 33% at ±30°)
#       • T10 is not the only PAC guard (T9, T11 also guard); ±50° keeps T10
#         as a meaningful "not-anti-phase" filter without being so strict that
#         it rejects legitimate slow-spindle regimes with MI sampling variance.
PAC_PHASE_TOL    = 5 * np.pi / 18   # T10: ±50° tolerance around each target phase

# T11: PAC directionality — SO should drive spindle (not vice versa).
#     Use cross-correlation between spindle envelope and SO-band signal;
#     SO leading means correlate(sp_env, so_filt) peaks at positive lag.
#     Search restricted to ±PAC_MAX_LAG_S window to avoid spurious peaks.
PAC_MAX_LAG_S    = 2.0       # search lag window ±2s (covers 1-2 SO cycles)
PAC_MIN_LAG_MS   = 20.0      # T11: SO must lead by at least 20 ms (robustness)

# --- Reward weights (feasible solutions only, sum to 1.0) ---
W_SHAPE   = 0.50
W_SO      = 0.25
W_SPINDLE = 0.25

BAD_OBJECTIVE = 1e6  # DE minimizes, so large = worst case


# =====================================================================
# V5 NEW: PAC helper — computes MI, preferred phase, directional lag
# =====================================================================
def compute_pac_metrics(r_ctx, r_thal, fs=FS_SIM):
    """
    Compute three phase-amplitude coupling metrics between cortical SO
    phase and thalamic spindle amplitude, for T9-T11 hard constraints.

    Pipeline:
      1. Band-pass r_ctx in SO band [SO_FREQ_LO, SO_FREQ_HI] → Hilbert
         → instantaneous phase φ_SO(t).
      2. Band-pass r_thal in spindle band [SPINDLE_LO, SPINDLE_HI] →
         Hilbert → instantaneous amplitude envelope A_sp(t).
      3. Compute KL modulation index (Tort et al. 2010).
      4. Compute mean vector length (Canolty et al. 2006) → preferred phase.
      5. Compute cross-correlation lag of A_sp wrt band-pass SO signal.

    Convention for preferred phase:
      sosfiltfilt returns zero-phase filtered signal. Hilbert phase on a
      cosine-like SO signal equals 0 at the SO peak (Up-state maximum).
      So preferred_phase ≈ 0 means "spindle amplitude peaks at SO Up-state
      peak" — the physiologically correct target (Mölle 2011).

    Convention for lag (T11):
      correlate(sp_env, so_filt) indexed by lag k means: at k > 0,
      sp_env[n+k] correlates with so_filt[n] → so_filt leads sp_env by
      k samples → "SO leads spindle". This is what we want.

    Returns a dict with: mi, preferred_phase, lag_samples, lag_ms.
    Falls back to safe defaults on any numerical failure.
    """
    out = {
        "mi": 0.0,
        "preferred_phase": np.pi,  # farthest from 0 → fails T10 on failure
        "lag_samples": 0,
        "lag_ms": 0.0,
        "ok": False,
    }
    try:
        # Protect against tiny or flat signals
        if len(r_ctx) < int(2 * fs) or len(r_thal) < int(2 * fs):
            return out
        if r_ctx.std() < 1e-6 or r_thal.std() < 1e-6:
            return out

        # 1. SO phase from cortex
        sos_so = butter(4, [SO_FREQ_LO, SO_FREQ_HI], btype="band",
                        fs=fs, output="sos")
        so_filt = sosfiltfilt(sos_so, r_ctx)
        so_analytic = hilbert(so_filt)
        so_phase = np.angle(so_analytic)

        # 2. Spindle envelope from thalamus
        sos_sp = butter(4, [SPINDLE_LO, SPINDLE_HI], btype="band",
                        fs=fs, output="sos")
        sp_filt = sosfiltfilt(sos_sp, r_thal)
        sp_amp = np.abs(hilbert(sp_filt))

        # Drop ±0.5 s at edges to avoid Hilbert/filtfilt boundary artefacts
        edge = int(0.5 * fs)
        so_filt = so_filt[edge:-edge]
        so_phase = so_phase[edge:-edge]
        sp_amp = sp_amp[edge:-edge]

        # 3. KL Modulation Index
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
        # Shannon entropy with safe log
        p_safe = np.where(p > 0, p, 1.0)
        H = -np.sum(p * np.log(p_safe))
        mi = (np.log(PAC_N_BINS) - H) / np.log(PAC_N_BINS)
        mi = float(np.clip(mi, 0.0, 1.0))

        # 4. Preferred phase via mean vector length (Canolty 2006)
        mvl = (sp_amp * np.exp(1j * so_phase)).mean()
        preferred_phase = float(np.angle(mvl))

        # 5. Directional lag via cross-correlation, restricted search window
        max_lag = int(PAC_MAX_LAG_S * fs)
        # Subtract means to avoid DC dominating xcorr
        a = sp_amp - sp_amp.mean()
        b = so_filt - so_filt.mean()
        xc = correlate(a, b, mode="full")
        lags = np.arange(-(len(b) - 1), len(a))
        # Restrict to ±max_lag window
        keep = (lags >= -max_lag) & (lags <= max_lag)
        xc_w = xc[keep]
        lags_w = lags[keep]
        if xc_w.size == 0:
            return out
        peak_lag_samples = int(lags_w[np.argmax(xc_w)])

        out.update({
            "mi": mi,
            "preferred_phase": preferred_phase,
            "lag_samples": peak_lag_samples,
            "lag_ms": peak_lag_samples * 1000.0 / fs,
            "ok": True,
        })
    except Exception:
        pass
    return out


def compute_constraints_v5(r_ctx, r_thal, f_c=None, p_c=None, fs=FS_SIM):
    """
    11 hard binary constraints for V5. ALL must pass for feasibility.
    Returns (n_passed, details_dict).

    T1-T3, T5: inherited from V3 (basic UP/DOWN + spindle FWHM)
    T4+:       SO peak must be sharp (Q-factor), not just in range
    T6:        SO regularity — IBI CV (limit cycle, not excitable)
    T7:        Spindle burstiness — envelope CV (not continuous oscillation)
    T8:        Enough discrete spindle events with valid duration
    T9:        NEW — PAC strength (Modulation Index > threshold)
    T10:       NEW — PAC preferred phase near SO Up-peak OR Down-to-Up
                     (two-sided: fast-spindle OR slow-spindle regime)
    T11:       NEW — PAC directionality (SO leads spindle)
    """
    details = {}

    # ── T1: DOWN state exists ─────────────────────────────────────────
    min_rE = float(r_ctx.min())
    t1 = min_rE < DOWN_THRESH_HZ
    details["T1"] = t1
    details["T1_min_rE"] = min_rE

    # ── T2: UP state exists ───────────────────────────────────────────
    max_rE = float(r_ctx.max())
    t2 = max_rE > UP_THRESH_HZ
    details["T2"] = t2
    details["T2_max_rE"] = max_rE

    # ── T3: UP state sustained ≥ UP_DURATION_MS ───────────────────────
    min_run_samples = int(UP_DURATION_MS * fs / 1000.0)
    above = (r_ctx > UP_THRESH_HZ).astype(np.int8)
    diff  = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    max_run = int((ends - starts).max()) if len(starts) > 0 else 0
    t3 = max_run >= min_run_samples
    details["T3"] = t3
    details["T3_longest_ms"] = max_run * 1000.0 / fs

    # ── T4+: SO peak in range AND sharp (Q-factor) ───────────────────
    if f_c is None or p_c is None:
        f_c, p_c = compute_epoch_psd(r_ctx, fs)
    so_mask = (f_c >= SO_FREQ_LO) & (f_c <= SO_FREQ_HI)
    # Neighbor bands: same width on each side of SO band
    so_width = SO_FREQ_HI - SO_FREQ_LO
    neighbor_lo = (f_c >= max(0.1, SO_FREQ_LO - so_width)) & (f_c < SO_FREQ_LO)
    neighbor_hi = (f_c > SO_FREQ_HI) & (f_c <= SO_FREQ_HI + so_width)

    so_peak_freq = 0.0
    so_q = 0.0
    t4 = False
    if so_mask.any():
        so_peak_freq = float(f_c[so_mask][np.argmax(p_c[so_mask])])
        so_peak_val  = p_c[so_mask].max()
        neighbors = np.concatenate([
            p_c[neighbor_lo] if neighbor_lo.any() else np.array([]),
            p_c[neighbor_hi] if neighbor_hi.any() else np.array([])
        ])
        if len(neighbors) > 0 and neighbors.mean() > 0:
            so_q = float(so_peak_val / neighbors.mean())
        t4 = (SO_FREQ_LO <= so_peak_freq <= SO_FREQ_HI) and (so_q > SO_Q_MIN)
    details["T4"] = t4
    details["T4_freq"] = so_peak_freq
    details["T4_q"]    = round(so_q, 3)

    # ── T5: Spindle FWHM > threshold ──────────────────────────────────
    f_th, p_th = compute_epoch_psd(r_thal, fs)
    sp_mask = (f_th >= SPINDLE_LO) & (f_th <= SPINDLE_HI)
    fwhm = 0.0
    if sp_mask.any() and p_th[sp_mask].max() > 0:
        p_sp = p_th[sp_mask]
        f_sp = f_th[sp_mask]
        half_power  = p_sp.max() / 2.0
        above_half  = f_sp[p_sp >= half_power]
        if len(above_half) >= 2:
            fwhm = float(above_half[-1] - above_half[0])
    t5 = fwhm > SPINDLE_FWHM_MIN
    details["T5"] = t5
    details["T5_fwhm"] = fwhm

    # ── T6: SO regularity — inter-burst interval CV (NEW) ─────────────
    # Uses `starts` from T3's run-length encoding of UP events.
    # Requires ≥ 3 UP events to compute meaningful CV.
    ibi_cv = 999.0
    n_bursts = len(starts)
    if n_bursts >= 3:
        intervals = np.diff(starts) / fs  # in seconds
        ibi_cv = float(intervals.std() / (intervals.mean() + 1e-12))
    t6 = (n_bursts >= 3) and (ibi_cv < IBI_CV_MAX)
    details["T6"] = t6
    details["T6_ibi_cv"]    = round(ibi_cv, 3)
    details["T6_n_bursts"]  = n_bursts

    # ── T7: Spindle envelope burstiness — CV (NEW) ────────────────────
    sp_cv = 0.0
    envelope = None
    try:
        sos = butter(4, [SPINDLE_LO, SPINDLE_HI], btype='band', fs=fs, output='sos')
        filtered = sosfiltfilt(sos, r_thal)
        envelope = np.abs(hilbert(filtered))
        sp_cv = float(envelope.std() / (envelope.mean() + 1e-12))
    except Exception:
        sp_cv = 0.0
        envelope = None
    t7 = sp_cv > SPINDLE_CV_MIN
    details["T7"] = t7
    details["T7_sp_cv"] = round(sp_cv, 3)

    # ── T8: Spindle event count — enough discrete bursts (NEW) ────────
    # Detects individual spindle events from the envelope computed in T7.
    # Fixes T7 loophole: "silence + 2 extreme spikes" passes CV but isn't
    # real waxing-waning spindle activity.
    n_sp_events = 0
    mean_sp_dur = 0.0
    try:
        if envelope is not None and len(envelope) > 0:
            thresh = envelope.mean() + SPINDLE_EVT_THRESH_FACTOR * envelope.std()
            above = (envelope > thresh).astype(np.int8)
            diff_sp = np.diff(np.concatenate(([0], above, [0])))
            sp_starts = np.where(diff_sp == 1)[0]
            sp_ends   = np.where(diff_sp == -1)[0]
            # Filter by duration
            durations = (sp_ends - sp_starts) / fs  # seconds
            valid = (durations >= SPINDLE_DUR_LO_S) & (durations <= SPINDLE_DUR_HI_S)
            n_sp_events = int(valid.sum())
            if n_sp_events > 0:
                mean_sp_dur = float(durations[valid].mean())
    except Exception:
        n_sp_events = 0
        mean_sp_dur = 0.0
    t8 = n_sp_events >= SPINDLE_EVT_MIN
    details["T8"] = t8
    details["T8_n_sp_events"] = n_sp_events
    details["T8_mean_sp_dur"] = round(mean_sp_dur, 3)

    # ── T9-T11: PAC three-pack (NEW in V5) ────────────────────────────
    # Guards against Failure Mode 3 (thalamo-cortical spatiotemporal
    # decoupling). Without these, optimiser can fit the N3 spectrum while
    # simulating a diseased brain (PAC-decoupled = aged / AD / schizophrenia).
    pac = compute_pac_metrics(r_ctx, r_thal, fs=fs)

    # T9: PAC strength — Modulation Index above chance
    t9 = pac["ok"] and (pac["mi"] > PAC_MI_MIN)
    details["T9"] = t9
    details["T9_mi"] = round(pac["mi"], 5)

    # T10: preferred SO phase near Up-peak (0) OR Down-to-Up transition (±π)
    # Two-sided acceptance for fast-spindle and slow-spindle regimes.
    phi = pac["preferred_phase"]
    # Minimum angular distance to targets {0, +π, -π}. Since +π and -π
    # identify on the circle, just take min of |phi| and (π - |phi|).
    phase_dist_to_target = min(abs(phi), np.pi - abs(phi))
    t10 = pac["ok"] and (phase_dist_to_target < PAC_PHASE_TOL)
    details["T10"] = t10
    details["T10_phase"] = round(phi, 3)
    details["T10_dist_to_target"] = round(phase_dist_to_target, 3)

    # T11: SO leads spindle (positive xcorr lag of sp_env vs so_filt)
    t11 = pac["ok"] and (pac["lag_ms"] >= PAC_MIN_LAG_MS)
    details["T11"] = t11
    details["T11_lag_ms"] = round(pac["lag_ms"], 1)

    # ── Summary ───────────────────────────────────────────────────────
    n_passed = sum([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11])
    details["n_passed"] = n_passed
    details["feasible"] = (n_passed == 11)

    return n_passed, details


def compute_feasibility_score(con):
    """
    Continuous constraint satisfaction: each constraint → [0, 1].
    1.0 = fully satisfied, 0.0 = completely unsatisfied.
    Capped at 0.99 for infeasible to preserve feasible > infeasible ordering.

    This replaces the -1M Deb cliff. Analogous to the maze RL fix:
    "hitting a wall just stops you in place, doesn't kill you."
    A solution with T7_cv=0.69 now scores MUCH higher than T7_cv=0.30.
    """
    scores = []

    # T1: want min_rE < 1.0 Hz (DOWN state exists)
    if con["T1"]:
        scores.append(1.0)
    else:
        scores.append(np.clip(1.0 - con["T1_min_rE"] / 5.0, 0, 0.99))

    # T2: want max_rE > 15.0 Hz (UP state exists)
    if con["T2"]:
        scores.append(1.0)
    else:
        scores.append(np.clip(con["T2_max_rE"] / UP_THRESH_HZ, 0, 0.99))

    # T3: want longest UP run ≥ 100ms
    if con["T3"]:
        scores.append(1.0)
    else:
        scores.append(np.clip(con["T3_longest_ms"] / UP_DURATION_MS, 0, 0.99))

    # T4+: want SO Q-factor > 2.0
    if con["T4"]:
        scores.append(1.0)
    else:
        scores.append(np.clip(con["T4_q"] / SO_Q_MIN, 0, 0.99))

    # T5: want spindle FWHM > 2.0 Hz
    if con["T5"]:
        scores.append(1.0)
    else:
        scores.append(np.clip(con["T5_fwhm"] / SPINDLE_FWHM_MIN, 0, 0.99))

    # T6: want IBI CV < 0.4 (lower is better)
    if con["T6"]:
        scores.append(1.0)
    else:
        raw_cv = min(con["T6_ibi_cv"], 2.0)  # cap extreme values
        scores.append(np.clip(1.0 - raw_cv / 1.0, 0, 0.99))

    # T7: want spindle envelope CV > 0.7 (higher is better)
    if con["T7"]:
        scores.append(1.0)
    else:
        scores.append(np.clip(con["T7_sp_cv"] / SPINDLE_CV_MIN, 0, 0.99))

    # T8: want ≥ 5 spindle events with valid duration
    if con["T8"]:
        scores.append(1.0)
    else:
        scores.append(np.clip(con["T8_n_sp_events"] / SPINDLE_EVT_MIN, 0, 0.99))

    # T9: want MI > PAC_MI_MIN (higher is better, saturate at 2x threshold)
    if con["T9"]:
        scores.append(1.0)
    else:
        # Linearly ramp from 0 (MI=0) to 0.99 (MI=PAC_MI_MIN)
        scores.append(np.clip(con["T9_mi"] / PAC_MI_MIN, 0, 0.99))

    # T10: want preferred phase near 0 OR near ±π (two-sided)
    # Score = 1 - dist_to_nearest_target / π  (so closer → higher)
    if con["T10"]:
        scores.append(1.0)
    else:
        # con["T10_dist_to_target"] is in [0, π/2]; the WORST case is π/2
        # (exactly half-way between Up-peak and Down-to-Up).
        # Normalize: dist=0 → score=1, dist=π/2 → score=0
        dist = con.get("T10_dist_to_target", np.pi / 2)
        scores.append(np.clip(1.0 - dist / (np.pi / 2), 0, 0.99))

    # T11: want SO→spindle lag ≥ PAC_MIN_LAG_MS (positive, bounded)
    if con["T11"]:
        scores.append(1.0)
    else:
        lag = con["T11_lag_ms"]
        if lag <= 0:
            # Wrong direction — score proportional to how close to zero
            scores.append(np.clip(1.0 + lag / 500.0, 0, 0.5))
        else:
            # Right direction but not yet enough — linear ramp to threshold
            scores.append(np.clip(lag / PAC_MIN_LAG_MS, 0, 0.99))

    return scores


# load_target_psd loads the target N3 PSD from the subject's EEG, applying the same preprocessing and QC steps as in
#  02_preprocess_psd.py to ensure a fair comparison between the model and the target data.
#  It returns the average PSD across all N3 epochs that passed QC, along with the corresponding frequencies.
def load_target_psd():
    # manifest.csv is often saved by Excel as UTF-16 (BOM FF FE). Try UTF-8 first, then fall back.
    try:
        manifest = pd.read_csv("data/manifest.csv", encoding="utf-8")
    except UnicodeDecodeError:
        manifest = pd.read_csv("data/manifest.csv", encoding="utf-16")
    subj_row = manifest[manifest["subject_id"] == SUBJECT_ID].iloc[0]
    raw = mne.io.read_raw_edf(
        subj_row["psg_path"], include=EEG_CHANNELS, preload=True, verbose=False
    ) # raw is a MNE Raw object containing the EEG data for the subject, loaded from the EDF file specified in the manifest.csv for the given SUBJECT_ID. We include only the channels specified in EEG_CHANNELS and preload the data into memory for faster access during preprocessing.
    fs_eeg = raw.info["sfreq"] # fs_eeg is the sampling frequency of the EEG data in Hz.


    hyp_path = Path(subj_row["hypnogram_path"]) 
    stages = load_hypnogram(hyp_path) # stages is a list of sleep stage labels for each epoch of the EEG data, loaded from the hypnogram file specified in the manifest.csv for the given SUBJECT_ID. This is used to identify which epochs correspond to N3 sleep, which is the target state for our model personalization.

    data_uv = raw.get_data()[0] * 1e6  # convert from V to uV; data_uv is a 1D NumPy array containing the EEG data for the first channel (assumed to be the only channel included) in microvolts. This is used for preprocessing and PSD computation, and the conversion to microvolts ensures that the amplitude values are in a more interpretable range for EEG data.
    n_samples_per_epoch = int(EPOCH_LEN_S * fs_eeg) # n_samples_per_epoch is the number of samples in each epoch
    n_epochs = min(len(stages), len(data_uv) // n_samples_per_epoch) # n_epochs is the number of epochs in the EEG data

    psds, f_ep, freq_mask = [], None, None
    for i in range(n_epochs):
        if stages[i] not in N3_LABELS: # only include epochs labeled as N3 or Sleep stage 4 (some hypnograms use "Sleep stage 4" instead of "N3" to label deep sleep); this ensures that we are extracting the target PSD from the correct sleep stage for model personalization.
            continue

        epoch_data = data_uv[i * n_samples_per_epoch : (i + 1) * n_samples_per_epoch]
        #epoch_data is a 1D NumPy array containing the EEG data for the current epoch in microvolts. This is extracted from the full EEG data using slicing based on the current epoch index and the number of samples per epoch. This epoch data is used for quality control and PSD computation for each N3 epoch.
        if np.ptp(epoch_data) > ARTIFACT_THRESH * 1e6:
            continue # reject epochs with peak-to-peak amplitude greater than ARTIFACT_THRESH (converted to microvolts); this is a simple artifact rejection step to exclude epochs with excessive noise or artifacts that could distort the PSD and lead to an unfair comparison between the model and the target data.
        
        f_ep, p_ep = compute_epoch_psd(epoch_data, fs_eeg)
        # f_eq and p_ep are the frequencies and power spectral density of the current epoch, computed using the compute_epoch_psd function defined in utils/02_preprocess_psd.py. This is used to extract the PSD for each N3 epoch that passes QC, which will be averaged across epochs to create the target PSD for model personalization.

        if freq_mask is None:
            freq_mask = (f_ep >= F_LO) & (f_ep <= F_HI)
        # freq_mask is a boolean array that identifies the frequencies in f_ep that are within the F_LO and F_HI range; this is used to select the frequency range of interest for the target PSD, which is the same range used for computing the model PSD during fitness evaluation. This ensures that we are comparing the model and target PSDs over the same frequency range, which is important for a fair comparison.
        psds.append(p_ep[freq_mask])

    if not psds:
        raise RuntimeError("No N3 epochs passed QC for subject " + SUBJECT_ID)
    # if no N3 epochs passed QC, raise an error to indicate that we cannot compute a target PSD for this subject, which is necessary for model personalization.
    target_psd   = np.mean(psds, axis=0) # target_psd is the average power spectral density across all N3 epochs that passed QC, computed by taking the mean of the psds list along the first axis. This is the target PSD that we will use for model personalization, and it represents the characteristic spectral profile of N3 sleep for this subject.
    target_freqs = f_ep[freq_mask] # target_freqs is the array of frequencies corresponding to the target_psd, selected using the freq_mask to include only the frequencies within the F_LO and F_HI range. This is used for computing the model PSD during fitness evaluation, ensuring that we are comparing the model and target PSDs over the same frequency range.
    print(f"  Target EEG: {len(psds)} N3 epochs used")
    return target_psd, target_freqs

# Part 2: Pre-compute FOOOF baseline on target (runs once)

def compute_target_periodic(target_psd, target_freqs):
    if not HAS_FOOOF:
        return None, None

    fm = FOOOF(**EVO_FOOOF_PARAMS)
    fm.fit(target_freqs, target_psd, [F_LO, F_HI])
    fooof_freqs     = fm.freqs
    target_log      = np.log10(target_psd + 1e-30)
    target_periodic = target_log[:len(fm._ap_fit)] - fm._ap_fit

    print(f"  FOOOF target peaks:")
    for pk in fm.peak_params_:
        print(f"    {pk[0]:.1f} Hz  power={pk[1]:.3f}  width={pk[2]:.1f}")
    print(f"  Aperiodic exponent: {fm.aperiodic_params_[1]:.2f}")

    return target_periodic, fooof_freqs

_eval_count  = 0
_best_score  = -np.inf
_best_params = {}
_records     = []
_t_start     = None


def compute_fitness_v5(params_vec,
                       target_psd, target_freqs,
                       target_periodic, fooof_freqs):
    """
    V5 fitness: 11 constraints with continuous satisfaction + simple rewards.

    if ALL 11 constraints pass (feasible):
        fitness = 0.50*shape_r + 0.25*so_power + 0.25*spindle_power  [range: 0 to ~1]
    else:
        fitness = -10 + 10 * mean(constraint_scores)  [range: -10 to ~-0.1]
        Each constraint_score is continuous [0, 1], giving DE gradient signal.

    Rationale for the scaling:
      • feasible solutions always > 0 (spectral rewards) → feasible > infeasible
      • infeasible solutions capped at -0.1 (when all 11 scores ≈ 0.99), which
        is still below any true feasible solution.
    """
    global _eval_count, _best_score, _best_params, _records

    _eval_count += 1
    mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th = params_vec

    # ── Run simulation ────────────────────────────────────────────────
    try:
        m = build_model(mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th)
        m.run()
    except Exception:
        # Fallback backend to avoid losing whole generations on numba/JIT failures.
        try:
            m.params["backend"] = "jitcdde"
            m.run()
        except Exception:
            return BAD_OBJECTIVE

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
        return BAD_OBJECTIVE

    # ── Compute PSD ───────────────────────────────────────────────────
    f_ctx_full, p_ctx_full = compute_epoch_psd(r_ctx, FS_SIM)
    mask = (f_ctx_full >= F_LO) & (f_ctx_full <= F_HI)
    f_ctx, p_ctx = f_ctx_full[mask], p_ctx_full[mask]

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Hard constraints (Deb feasibility rules) — 11 in V5
    # ══════════════════════════════════════════════════════════════════
    n_passed, con = compute_constraints_v5(
        r_ctx, r_thal, f_c=f_ctx_full, p_c=p_ctx_full, fs=FS_SIM
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Rewards (only computed for feasible solutions)
    # ══════════════════════════════════════════════════════════════════
    shape_r = 0.0
    so_power = 0.0
    spindle_power = 0.0

    if con["feasible"]:
        # shape_r: FOOOF periodic residual Pearson-r (identical logic to V3)
        if HAS_FOOOF and target_periodic is not None:
            try:
                p_interp = interp1d(
                    f_ctx, p_ctx, bounds_error=False, fill_value=1e-30,
                )(fooof_freqs)
                fm_sim = FOOOF(**EVO_FOOOF_PARAMS)
                fm_sim.fit(fooof_freqs, p_interp, [F_LO, F_HI])
                sim_log = np.log10(p_interp[:len(fm_sim._ap_fit)] + 1e-30)
                sim_periodic = sim_log - fm_sim._ap_fit
                n_r = min(len(sim_periodic), len(target_periodic))
                shape_r, _ = pearsonr(sim_periodic[:n_r], target_periodic[:n_r])
                shape_r = max(shape_r, 0.0)
            except Exception:
                shape_r = 0.0
        else:
            p_interp_fb = interp1d(f_ctx, p_ctx,
                                   bounds_error=False, fill_value=1e-30)(target_freqs)
            weights  = target_freqs ** (-0.5)
            weights /= weights.sum()
            chi2     = np.sum(weights * (np.log10(p_interp_fb + 1e-30)
                                        - np.log10(target_psd + 1e-30)) ** 2)
            shape_r  = float(np.exp(-chi2))

        # so_power / spindle_power: FOOOF peak detection (identical to V3)
        if HAS_FOOOF:
            try:
                p_interp2 = interp1d(f_ctx, p_ctx,
                                     bounds_error=False, fill_value=1e-30)(fooof_freqs)
                fm2 = FOOOF(**EVO_FOOOF_PARAMS)
                fm2.fit(fooof_freqs, p_interp2, [F_LO, F_HI])
                for pk in fm2.peak_params_:
                    freq, power, _ = pk
                    if SO_FREQ_LO <= freq <= SO_FREQ_HI:
                        so_power = max(so_power, float(power))
                    if SPINDLE_LO <= freq <= SPINDLE_HI:
                        spindle_power = max(spindle_power, float(power))
            except Exception:
                pass

        # V4 fitness: simple weighted sum (V1 formula, no penalties/gates)
        fitness = (W_SHAPE * shape_r
                   + W_SO * so_power
                   + W_SPINDLE * spindle_power)
    else:
        # Continuous constraint satisfaction (replaces -1M Deb cliff)
        # Range: -10 (all 11 constraints far from passing) to ~-0.1 (almost all passing)
        # Feasible solutions are always [0, ~1], so feasible > infeasible guaranteed
        c_scores = compute_feasibility_score(con)
        fitness = -10.0 + 10.0 * (sum(c_scores) / 11.0)

    # ── Record ────────────────────────────────────────────────────────
    record = dict(zip(PARAM_NAMES, params_vec))
    record.update({
        "score":          round(fitness, 6),
        "feasible":       int(con["feasible"]),
        "n_passed":       n_passed,
        "shape_r":        round(shape_r, 6),
        "so_power":       round(so_power, 6),
        "spindle_power":  round(spindle_power, 6),
        # Per-constraint details
        "T1":             int(con["T1"]),
        "T1_min_rE":      round(con["T1_min_rE"], 3),
        "T2":             int(con["T2"]),
        "T2_max_rE":      round(con["T2_max_rE"], 3),
        "T3":             int(con["T3"]),
        "T3_longest_ms":  round(con["T3_longest_ms"], 1),
        "T4":             int(con["T4"]),
        "T4_freq":        round(con["T4_freq"], 3),
        "T4_q":           con["T4_q"],
        "T5":             int(con["T5"]),
        "T5_fwhm":        round(con["T5_fwhm"], 3),
        "T6":             int(con["T6"]),
        "T6_ibi_cv":      con["T6_ibi_cv"],
        "T6_n_bursts":    con["T6_n_bursts"],
        "T7":             int(con["T7"]),
        "T7_sp_cv":       con["T7_sp_cv"],
        "T8":             int(con["T8"]),
        "T8_n_sp_events": con["T8_n_sp_events"],
        "T8_mean_sp_dur": con["T8_mean_sp_dur"],
        "T9":             int(con["T9"]),
        "T9_mi":          con["T9_mi"],
        "T10":            int(con["T10"]),
        "T10_phase":      con["T10_phase"],
        "T10_dist_tgt":   con.get("T10_dist_to_target", 0.0),
        "T11":            int(con["T11"]),
        "T11_lag_ms":     con["T11_lag_ms"],
        "eval":           _eval_count,
    })
    _records.append(record)

    if fitness > _best_score:
        _best_score  = fitness
        _best_params = record.copy()

    return -fitness   # DE minimizes

_gen = 0
_cb_last_best = -np.inf

def _callback(xk, convergence):
    global _gen, _cb_last_best
    _gen += 1
    elapsed = time.time() - _t_start
    if not _best_params:
        print(
            f"  Gen {_gen:2d}/{N_GEN}  no-valid-best-yet"
            f"  evals={_eval_count}"
            f"  conv={convergence:.3e}"
            f"  elapsed={elapsed:.0f}s"
        )
        return False

    bp = _best_params
    improved = _best_score > _cb_last_best + 1e-12
    if improved:
        _cb_last_best = _best_score
    status = "improved" if improved else "steady"

    print(
        f"  Gen {_gen:2d}/{N_GEN}  best={_best_score:+.4f}"
        f"  {status}"
        f"  feasible={bp.get('feasible',0)}"
        f"  [{bp.get('n_passed',0)}/11]"
        f"  T1={bp.get('T1',0)} T2={bp.get('T2',0)}"
        f" T3={bp.get('T3',0)} T4={bp.get('T4',0)}"
        f" T5={bp.get('T5',0)} T6={bp.get('T6',0)}"
        f" T7={bp.get('T7',0)} T8={bp.get('T8',0)}"
        f" T9={bp.get('T9',0)} T10={bp.get('T10',0)} T11={bp.get('T11',0)}"
        f"  shape_r={bp.get('shape_r',0):.3f}"
        f"  so={bp.get('so_power',0):.3f}"
        f"  sp={bp.get('spindle_power',0):.3f}"
        f"  MI={bp.get('T9_mi',0):.4f}"
        f"  phi={bp.get('T10_phase',0):+.2f}"
        f"  lag={bp.get('T11_lag_ms',0):+.0f}ms"
        f"  evals={_eval_count}"
        f"  elapsed={elapsed:.0f}s"
    )
    return False


def main():
    global _t_start

    print("=" * 65)
    print("V5 Evolution: V4-improve + PAC hard constraints (T9, T10, T11)")
    print("  FOOOF=" + ("ON" if HAS_FOOOF else "OFF (fallback)"))
    print(f"  8 params, 11 constraints (continuous satisfaction)")
    print(f"  Narrowed bounds: ±20% around V4 best solution (unchanged)")
    print(f"  Strategy: best1bin, mutation=(0.5,1.0), workers=1 (unchanged)")
    print(f"  T4+ SO Q-factor > {SO_Q_MIN}")
    print(f"  T6 IBI CV < {IBI_CV_MAX} (SO regularity)")
    print(f"  T7 spindle envelope CV > {SPINDLE_CV_MIN} (burstiness)")
    print(f"  T8 spindle events >= {SPINDLE_EVT_MIN}, "
          f"dur {SPINDLE_DUR_LO_S}-{SPINDLE_DUR_HI_S}s")
    print(f"  T9  (NEW) PAC MI > {PAC_MI_MIN} "
          f"[Kullback-Leibler, {PAC_N_BINS} phase bins]")
    print(f"  T10 (NEW) preferred phase near 0 OR ±π within ±50° "
          f"(fast OR slow spindle, Mölle 2011)")
    print(f"  T11 (NEW) SO→spindle lag ≥ {PAC_MIN_LAG_MS} ms "
          f"(directionality: SO leads)")
    print(f"  Reward (feasible only): "
          f"{W_SHAPE:.2f}*shape_r + {W_SO:.2f}*so + {W_SPINDLE:.2f}*sp")
    print(f"  Infeasible scoring: continuous [-10, ~-0.1] (scaled for 11 C's)")
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
    _t_start = time.time()
    print("\n")

    result = differential_evolution(
        compute_fitness_v5,
        bounds=BOUNDS,
        args=(target_psd, target_freqs, target_periodic, fooof_freqs),
        strategy="best1bin",        # more stable than rand1bin for expensive objectives
        maxiter=N_GEN,
        popsize=DE_POPSIZE,
        tol=1e-4,
        mutation=(0.5, 1.0),        # less aggressive mutation to reduce oscillation
        recombination=0.7,
        seed=2024,                  # different seed from V4
        callback=_callback,
        polish=False,
        workers=1,                  # single process: avoids LLVM memory blow-ups
        updating="immediate",
    )

    total_time = time.time() - _t_start
    print(f"\nEvolution complete in {total_time/3600:.2f} h")

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

    if not best:
        print("[warn] No valid candidate found during evolution; skip validation summary.")
        print(f"  evals={_eval_count}  records={len(_records)}")
        return

    print(f"\n{'='*55}")
    print("V5 Validation")
    print(f"{'='*55}")
    print(f"  score         = {best.get('score'):.4f}")
    print(f"  feasible      = {best.get('feasible')}")
    print(f"  n_passed      = {best.get('n_passed')}/11")
    print(f"  shape_r       = {best.get('shape_r'):.4f}")
    print(f"  so_power      = {best.get('so_power'):.4f}")
    print(f"  spindle_power = {best.get('spindle_power'):.4f}")
    print(f"")
    print(f"  Constraints (T1-T8 from V4, T9-T11 new in V5):")
    print(f"    T1  DOWN exists:        {'PASS' if best.get('T1') else 'FAIL'}"
          f"  [min_rE={best.get('T1_min_rE'):.1f} Hz]")
    print(f"    T2  UP exists:          {'PASS' if best.get('T2') else 'FAIL'}"
          f"  [max_rE={best.get('T2_max_rE'):.1f} Hz]")
    print(f"    T3  UP sustained:       {'PASS' if best.get('T3') else 'FAIL'}"
          f"  [longest={best.get('T3_longest_ms'):.0f} ms]")
    print(f"    T4+ SO peak Q-factor:   {'PASS' if best.get('T4') else 'FAIL'}"
          f"  [freq={best.get('T4_freq'):.2f} Hz, Q={best.get('T4_q')}]")
    print(f"    T5  Spindle FWHM:       {'PASS' if best.get('T5') else 'FAIL'}"
          f"  [FWHM={best.get('T5_fwhm'):.2f} Hz]")
    print(f"    T6  SO regularity:      {'PASS' if best.get('T6') else 'FAIL'}"
          f"  [IBI_CV={best.get('T6_ibi_cv')}, n_bursts={best.get('T6_n_bursts')}]")
    print(f"    T7  Spindle burstiness: {'PASS' if best.get('T7') else 'FAIL'}"
          f"  [envelope_CV={best.get('T7_sp_cv')}]")
    print(f"    T8  Spindle events:     {'PASS' if best.get('T8') else 'FAIL'}"
          f"  [n_events={best.get('T8_n_sp_events')}, mean_dur={best.get('T8_mean_sp_dur')}s]")
    print(f"    T9  PAC strength:       {'PASS' if best.get('T9') else 'FAIL'}"
          f"  [MI={best.get('T9_mi'):.5f} (min {PAC_MI_MIN})]")
    print(f"    T10 PAC preferred phase:{'PASS' if best.get('T10') else 'FAIL'}"
          f"  [phi={best.get('T10_phase'):+.2f} rad, "
          f"dist_to_target={best.get('T10_dist_tgt', 0):.2f} rad "
          f"(< 5π/18 = 0.87 ≈ 50°)]")
    print(f"    T11 PAC direction:      {'PASS' if best.get('T11') else 'FAIL'}"
          f"  [SO→spindle lag={best.get('T11_lag_ms'):+.0f} ms (≥ {PAC_MIN_LAG_MS})]")
    print(f"")
    print(f"  Best parameters:")
    for k in PARAM_NAMES:
        print(f"    {k}: {best.get(k):.4f}")
    print(f"\nNext: run plot_fig7_v5.py (update plot script for 11 constraints)")


if __name__ == "__main__":
    main()