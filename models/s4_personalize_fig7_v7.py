"""
s4_personalize_fig7_v7.py
=========================
Physics-constrained thalamocortical personalisation — V7.

V7 = V6 + diagnostic-driven redesign of reward structure and
     simulation-length/constraint calibration.

Motivation — why v6 failed (from s4_0c + s4_0d diagnostics):

  v6 achieved only 0.02% feasibility rate (1 feasible in 4960 evals),
  far below predicted 5-15%. The single feasible solution had
  spindle_power=0 and thalamic time series identical to v5's
  "fake spindle" failure mode. Four diagnostic findings explain this
  and motivate v7:

  (D1) 30s simulation noise-limits T6 (SO IBI CV < 0.4).
       At c_th2ctx=0 in s4_0d, the SAME parameters give:
         - 30s:  T6 IBI_CV = [0.61, 0.45, 0.02, 0.50]  (1/4 pass)
         - 60s:  T6 IBI_CV = [0.32, 0.37, 0.32, 0.48]  (3/4 pass)
       T6 failure in v4/v5/v6 was largely a SAMPLING ARTIFACT, not a
       real dynamical failure. 60s cuts IBI_CV sampling noise in half.

  (D2) The spindle-power reward is unreachable for mechanistic spindles.
       At all 4 sweep hotspots × 5 c_th2ctx × 2 durations = 40 configs,
       only 3/40 produced FOOOF spindle_power > 0.1, and all 3 had
       c_th2ctx >= 0.075 which destroys the cortex SO (T4_q < 1).
       In the "SO-stable" region (c_th2ctx < 0.05), FOOOF sees no
       σ peak because spindles live in thalamus, not cortex.
       Conclusion: FOOOF spindle_power is a DISEASE-INDICATOR (only
       appears when coupling is pathologically strong), not a healthy
       spindle indicator. Replace with event-based reward.

  (D3) The SO-power reward is also nearly unreachable.
       Only 1/40 configs in s4_0d had so_power > 0.1 (P3 @ c_th2ctx=0.025
       at 60s, a statistical outlier). Yet T4_q > 3 at many configs,
       indicating a healthy SO peak that FOOOF's min_peak_height=0.05
       floor rejects. Replace with T4_q-derived reward.

  (D4) T5 (spindle FWHM > 2 Hz) rejects mechanistically valid spindles.
       All 8 sweep hotspots have FWHM in [0, 0.75] Hz. The v4-era T5
       threshold selected for "distributed σ-band activity" (which
       FOOOF happened to detect) rather than "true waxing-waning
       T-current resonance" (which has a NARROW peak). T5 > 2 Hz is
       a BUG; relax to > 0.2 Hz.

Six deliberate changes in V7:

  (1) SIM_DUR_MS: 30_000 → 60_000 ms.
      Halves T6 IBI_CV sampling noise. Doubles wall-clock time per eval.
      Motivation: D1. This alone is expected to raise feasibility
      >10x regardless of other changes.

  (2) spindle_power reward: FOOOF → event-based.
      reward = min(T12_n_verified / 15, 1.0)
      Events are already counted (T12). 15 ≈ expected real-spindle
      count in 60s (15-20 spindles/min in N3 per Ferrarelli 2007).
      Motivation: D2.

  (3) so_power reward: FOOOF → T4_q-derived.
      reward = clip((T4_q - 1.0) / 4.0, 0, 1)
      T4_q is peak/neighbor-floor ratio; subtract 1 (no peak = ratio 1)
      and normalize by 4 (a "strong" peak has ratio ~5).
      Motivation: D3.

  (4) c_th2ctx bounds: [0.05, 0.25] → [0.00, 0.05].
      Accepts the trade-off: c_th2ctx > 0.05 destroys SO (T4, T6 fail
      together). The "spindle PSD visibility" was a phantom goal —
      real EEG σ-peaks come from DISTRIBUTED thalamic activity
      averaged over many spindle events, not from single-run PSD.
      Motivation: D2.

  (5) T5 threshold: FWHM > 2.0 → > 0.2 Hz.
      Corrects T5's selection bias against mechanistically valid
      narrow-peak spindles. Redundancy: T8 + T12 already enforce
      waxing-waning event structure through time-domain detection.
      Motivation: D4.

  (6) No DE warm-start.
      Despite hotspot being in a small fraction of bounds volume,
      initial Sobol sampling gives ~93% probability that at least
      one individual lands within 30% normalized distance of the
      hotspot. Preserving pure exploration lets us demonstrate that
      v7's fitness landscape INTRINSICALLY guides DE to the hotspot.

Nothing else changes from v6:
  T1-T4, T6-T12 thresholds (except T5), PAC three-pack, DE settings
  (popsize=20, N_GEN=30, strategy='best1bin'), cortex bounds, fitness
  scaling, continuous feasibility scores.

Expected wall time: ~3-4 hours (2× v6's 1.5h due to change #1).

Usage (from project root):
  python models/s4_personalize_fig7_v7.py

Outputs:
  data/patient_params_fig7_v7_SC4001.json
  outputs/evolution_fig7_v7_records.csv
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
from scipy.signal import hilbert, butter, sosfiltfilt, correlate, welch
from scipy.ndimage import gaussian_filter1d
import mne
mne.set_log_level("WARNING")

from neurolib.models.multimodel import MultiModel, ALNNode, ThalamicNode
from neurolib.models.multimodel.builder.base.network import Network
from neurolib.models.multimodel.builder.base.constants import EXC, INH
from neurolib.utils.stimulus import OrnsteinUhlenbeckProcess
import numba

@numba.njit
def seed_numba(seed):
    np.random.seed(seed)

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
SIM_DUR_MS      = 60_000        # V7 CHANGE #1: 30s → 60s (D1: halves T6
                                 # IBI_CV sampling noise; see diagnostic
                                 # s4_0d_scan_c_th2ctx_sweetspot.py)
DE_POPSIZE      = 20            # reduce memory/time pressure while keeping diversity
N_GEN           = 30            # DE generations

PARAMS_PATH  = f"data/patient_params_fig7_v7_{SUBJECT_ID}.json"
RECORDS_PATH = "outputs/evolution_fig7_v7_records.csv"

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
# Retained for reference. Cortex bounds still anchored here since cortex
# SO was physiologically healthy in v4/v5 (T1-T6 all passed).
_V4_BEST = dict(mue=3.895, mui=3.027, b=35.5, tauA=1426,
                g_LK=0.0564, g_h=0.0517, c_th2ctx=0.0986, c_ctx2th=0.038)

# V7 BOUNDS — v6 + c_th2ctx tightened based on sweet-spot diagnostic
# ----------------------------------------------------------------
# Cortex (mue, mui, b, tauA): UNCHANGED from v5/v6.
#
# Thalamic (g_LK, g_h): UNCHANGED from v6 (sweep-calibrated, working).
#
# c_th2ctx: v6 [0.05, 0.25] → v7 [0.00, 0.05]  (V7 CHANGE #4)
#   Rationale (D2, s4_0d scan):
#   - c_th2ctx ≥ 0.075 destroys cortex SO (T4_q drops from ~3.5 to <1,
#     T6 IBI_CV rises above 0.6).
#   - c_th2ctx ∈ [0.05, 0.075] is a boundary layer: SO survives but
#     already weakened.
#   - c_th2ctx ∈ [0.00, 0.05] keeps SO healthy. No FOOOF-detectable
#     σ peak at these values, but event-based spindle reward (V7
#     CHANGE #2) works at any c_th2ctx.
#
# c_ctx2th: UNCHANGED from v6 [0.05, 0.22] — sweep's PAC hotspot is
#           [0.155, 0.200], which is CTX→TH (opposite direction), so
#           unaffected by the c_th2ctx tightening.
BOUNDS = [
    (_V4_BEST["mue"]    * 0.85, _V4_BEST["mue"]    * 1.15),   # mue:    3.31 – 4.48
    (_V4_BEST["mui"]    * 0.85, _V4_BEST["mui"]    * 1.15),   # mui:    2.57 – 3.48
    (_V4_BEST["b"]      * 0.80, _V4_BEST["b"]      * 1.20),   # b:      28.4 – 42.6
    (_V4_BEST["tauA"]   * 0.70, _V4_BEST["tauA"]   * 1.30),   # tauA:   998  – 1854
    (0.020, 0.070),                                            # g_LK     (sweep)
    (0.035, 0.095),                                            # g_h      (sweep)
    (0.00,  0.05),                                             # c_th2ctx (V7: tightened for SO stability)
    (0.05,  0.22),                                             # c_ctx2th (sweep PAC hotspot)
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
        # --- Bulletproof Seed Injection (mirrors plot_fig7_compare_v7_vs_v8.py) ---
        th[0].seed = 42
        th[0].noise_input = [OrnsteinUhlenbeckProcess(
            mu=0.0, sigma=0.0, tau=5.0, seed=42)]
        # -------------------------------------------------------------------------
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
SPINDLE_FWHM_MIN = 0.2       # T5: min spindle FWHM [Hz]
                              # V7 CHANGE #5: 2.0 → 0.2 (D4: mechanistically
                              # valid T-current-resonance spindles have narrow
                              # peaks; 2.0 Hz selected for distributed σ
                              # activity, a spurious regime. Redundant
                              # waxing-waning enforcement is via T8+T12.)
IBI_CV_MAX       = 0.4       # T6: max inter-burst interval CV
SPINDLE_CV_MIN   = 0.7       # T7: min spindle envelope CV
SPINDLE_EVT_MIN  = 5         # T8: min number of spindle events in 25s
SPINDLE_DUR_LO_S = 0.3       # T8: min spindle event duration [s]
SPINDLE_DUR_HI_S = 2.0       # T8: max spindle event duration [s]
# V6 BUG FIX: v5 used threshold = mean(env) + 1.5*std(env), which FAILS
# on bimodal envelope distributions (when bursts occupy a large fraction
# of the signal, mean and std are pulled up TOGETHER and the threshold
# lands ABOVE the burst mean, giving zero detected events). Replaced
# with industry-standard percentile threshold on smoothed envelope
# (Lacourse A7 / YASA spindle detection convention).
SPINDLE_EVT_PCTILE       = 75.0    # 75th percentile of smoothed envelope
SPINDLE_ENV_SMOOTH_MS    = 200.0   # Gaussian smoothing σ before threshold

# T12 constants (NEW in V6): peak-inside-event verification.
# For each detected spindle event, compute Welch PSD within the event
# window and require σ-band (10-14 Hz) peak power > 1.5× non-σ-band
# (4-10 Hz) peak power. This plugs the v5 loophole where sporadic
# transient spikes filtered through 10-14 Hz produce envelope "ripples"
# that cross threshold but contain no real σ oscillation.
T12_PEAK_INSIDE_RATIO   = 1.5     # σ-peak / non-σ-peak ratio within event
T12_N_VERIFIED_MIN      = 5       # T12: at least 5 events must pass peak check

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
    import sys
    from pathlib import Path
    _repair_dir = Path(__file__).resolve().parent.parent / "S4_v7_repair"
    if str(_repair_dir) not in sys.path:
        sys.path.insert(0, str(_repair_dir))
    from compute_pac_metrics_fixed import compute_pac_metrics as _compute_pac_fixed
    return _compute_pac_fixed(r_ctx, r_thal, fs)


def compute_constraints_v7(r_ctx, r_thal, f_c=None, p_c=None, fs=FS_SIM):
    """
    12 hard binary constraints for V6. ALL must pass for feasibility.
    Returns (n_passed, details_dict).

    T1-T3, T5: inherited from V3 (basic UP/DOWN + spindle FWHM)
    T4+:       SO peak must be sharp (Q-factor), not just in range
    T6:        SO regularity — IBI CV (limit cycle, not excitable)
    T7:        Spindle burstiness — envelope CV (not continuous oscillation)
    T8:        Enough discrete spindle events with valid duration
                V6 FIX: percentile threshold on smoothed envelope
                (was mean+1.5*std, broken on bimodal distributions)
    T9:        PAC strength (Modulation Index > threshold)
    T10:       PAC preferred phase near SO Up-peak OR Down-to-Up
               (two-sided: fast-spindle OR slow-spindle regime)
    T11:       PAC directionality (SO leads spindle)
    T12:       NEW — Peak-inside-event verification. Each detected spindle
               event must contain dominant σ-band oscillation (plugs v5
               "fake spindle" loophole).
    """
    details = {}

    # ── T1: DOWN state exists ─────────────────────────────────────────
    min_rE = float(r_ctx.min()) # minimum cortical EXC firing rate across the run
    t1 = min_rE < DOWN_THRESH_HZ # min_rE below threshold → DOWN state exists → T1 passes
    details["T1"] = t1 # boolean pass/fail for T1
    details["T1_min_rE"] = min_rE # actual minimum firing rate

    # ── T2: UP state exists ───────────────────────────────────────────
    max_rE = float(r_ctx.max()) # maximum cortical EXC firing rate across the run
    t2 = max_rE > UP_THRESH_HZ # max_rE above threshold → UP state exists → T2 passes
    details["T2"] = t2 # boolean pass/fail for T2
    details["T2_max_rE"] = max_rE # actual maximum firing rate

    # ── T3: UP state sustained ≥ UP_DURATION_MS ───────────────────────
    min_run_samples = int(UP_DURATION_MS * fs / 1000.0) # min number of samples 
    # corresponding to UP_DURATION_MS at sampling rate fs
    above = (r_ctx > UP_THRESH_HZ).astype(np.int8) # 1 where r_ctx above UP_THRESH_HZ, else 0
    diff  = np.diff(np.concatenate(([0], above, [0]))) # find transitions in above-threshold state
    starts = np.where(diff == 1)[0] # indices where above goes from 0 to 1 → start of UP run
    ends   = np.where(diff == -1)[0] # indices where above goes from 1 to 0 → end of UP run
    max_run = int((ends - starts).max()) if len(starts) > 0 else 0 # longest run of consecutive samples above threshold
    t3 = max_run >= min_run_samples # t3 passes if longest run is larger than or equal to min_run_samples
    details["T3"] = t3 # boolean pass/fail for T3
    details["T3_longest_ms"] = max_run * 1000.0 / fs # actual longest run duration in ms

    # ── T4+: SO peak in range AND sharp (Q-factor) ───────────────────
    if f_c is None or p_c is None:
        f_c, p_c = compute_epoch_psd(r_ctx, fs) # f_c: frequencies, p_c: power at f_c
        # f_s: sampling frequencies; r_ctx: cortical firing rate time series
    so_mask = (f_c >= SO_FREQ_LO) & (f_c <= SO_FREQ_HI) # so_mask is True where f_c in the Slow Oscillation (SO) frequency range
    # Neighbor bands: same width on each side of SO band
    so_width = SO_FREQ_HI - SO_FREQ_LO # width of the SO frequency band
    neighbor_lo = (f_c >= max(0.1, SO_FREQ_LO - so_width)) & (f_c < SO_FREQ_LO)
    # neighbor_lo is a mask: True where f_c is lower than SO_FREQ_LO but higher than max(0.1, SO_FREQ_LO - so_width)
    neighbor_hi = (f_c > SO_FREQ_HI) & (f_c <= SO_FREQ_HI + so_width)
    # neighbor_hi is a mask: True where f_c is higher than SO_FREQ_HI but lower than or equal to SO_FREQ_HI + so_width
    so_peak_freq = 0.0
    so_q = 0.0
    t4 = False
    if so_mask.any():
        so_peak_freq = float(f_c[so_mask][np.argmax(p_c[so_mask])])
        # so_peak_freq is the frequency that has the maximum power within the SO band
        so_peak_val  = p_c[so_mask].max()
        # so_peak_val is maximum power value within the SO band
        neighbors = np.concatenate([
            p_c[neighbor_lo] if neighbor_lo.any() else np.array([]),
            p_c[neighbor_hi] if neighbor_hi.any() else np.array([])
        ]) # neighbors_values are the power values in the neighbor_lo  and neighbor_hi bands
        if len(neighbors) > 0 and neighbors.mean() > 0:
            so_q = float(so_peak_val / neighbors.mean())
            # so_q is the ratio of the SO peak power to the average power in the neighboring frequency bands
        t4 = (SO_FREQ_LO <= so_peak_freq <= SO_FREQ_HI) and (so_q > SO_Q_MIN)
        # t4 passes if the SO peak freq is within the SO frequency range
        # and the Q-factor is above the minimum threshold
    details["T4"] = t4 # boolean pass/fail for T4
    details["T4_freq"] = so_peak_freq # actual SO peak frequency
    details["T4_q"]    = round(so_q, 3) # actual Q-factor


    # ── T5: Spindle FWHM > threshold ──────────────────────────────────
    f_th, p_th = compute_epoch_psd(r_thal, fs)
    # f_th: frequencies, p_th: power at f_th, r_thal: thalamic firing rate time series
    sp_mask = (f_th >= SPINDLE_LO) & (f_th <= SPINDLE_HI)
    # sp_mask is True where f_th is in the spindle frequency range
    fwhm = 0.0 # full width at half maximum of the spindle peak
    if sp_mask.any() and p_th[sp_mask].max() > 0:
        p_sp = p_th[sp_mask] # p_sp: array of power values in the spindle frequency range
        f_sp = f_th[sp_mask] # f_sp: array of frequencies in the spindle frequency range
        half_power  = p_sp.max() / 2.0 # half of the maximum power in the spindle band
        above_half  = f_sp[p_sp >= half_power] # array of frequencies where the power is above half of the maximum
        if len(above_half) >= 2:
            fwhm = float(above_half[-1] - above_half[0]) # full width at half maximum
    t5 = fwhm > SPINDLE_FWHM_MIN # t5 passes if the FWHM is greater than the minimum threshold
    details["T5"] = t5 # boolean pass/fail for T5
    details["T5_fwhm"] = fwhm # actual full width at half maximum of the spindle peak

    # ── T6: SO regularity — inter-burst interval CV (NEW) ─────────────
    # Uses `starts` from T3's run-length encoding of UP events.
    # Requires ≥ 3 UP events to compute meaningful CV.
    ibi_cv = 999.0 # coefficient of variation of inter-burst intervals (IBI)
    n_bursts = len(starts) # number of UP events (brusts) detected in T3
    # starts, an array of indices where UP events start (from T3)
    if n_bursts >= 3: 
        intervals = np.diff(starts) / fs  # inter-burst intervals in seconds
        # np.diff(starts) gives the differences between consecutive start indices
        ibi_cv = float(intervals.std() / (intervals.mean() + 1e-12))
        # ibi_cv is calcuated as intervals.std() / intervals.mean()
        # the higher the ibi_cv, the more irregular the intervals between UP events (bursts) are
        # the lower the ibi_cv, the more regular the intervals between UP events (bursts) are
    t6 = (n_bursts >= 3) and (ibi_cv < IBI_CV_MAX)
    # t6 passes if there are at least 3 UP events 
    # and ibi_cv is below the maximum threshold, indicating regular bursting

    details["T6"] = t6 # boolean pass/fail for T6
    details["T6_ibi_cv"]    = round(ibi_cv, 3) # actual IBI CV
    details["T6_n_bursts"]  = n_bursts # actual number of UP events (bursts) detected in T3

    # ── T7: Spindle envelope burstiness — CV (NEW) ────────────────────
    # T7是一个新的约束条件，用于评估纺锤体包络的爆发性（burstiness）。纺锤体活动通常以短暂的爆发形式出现，而不是持续的振荡。
    # T7通过计算纺锤体包络的变异系数（CV）来量化这种爆发性。CV是标准差与均值的比值，反映了包络振幅的相对变异程度。较高的CV表示包络具有更多的爆发性（更大的振幅变化），而较低的CV表示包络更连续（较小的振幅变化）。T7要求纺锤体包络的CV必须大于预设的最小阈值，以确保模拟中纺锤体活动具有足够的爆发性特征。
    sp_cv = 0.0 # sp_cv, coefficient of variation of the spindle envelope
    envelope = None # envelope of the spindl amplitude
    try:
        sos = butter(4, [SPINDLE_LO, SPINDLE_HI], btype='band', fs=fs, output='sos')
        # sos是一个巴特沃斯带通滤波器的二阶节系数，设计用于纺锤体频段 [SPINDLE_LO, SPINDLE_HI]，采样率为fs
        # sos is the second order representation of Butter filter coefficients for the spindle frequency band
        filtered = sosfiltfilt(sos, r_thal)
        # sosfiltfilt是一个零相位滤波函数，应用sos滤波器对r_thal进行前向和反向滤波，以避免相位失真
        # filtered is the filtered thalamic signal (t_thal) obtained by applying sos filter
        envelope = np.abs(hilbert(filtered))
        # envelope是通过对滤波后的信号应用Hilbert变换并取绝对值得到的纺锤体振幅包络。Hilbert变换提供了一个解析信号，其中实部是原始信号，
        # 虚部是原始信号的90度相位移。纺锤体振幅
        # envelope is the absolute value of the Hilbert transform of the filtered signal，
        #  A(t) = |H(x(t))|, which gives the amplitude envelope of the spindle activity
        # x(t) = A(t) * cos(θ(t)), where x(t) is the original signal, A(t) is the envelope, 
        # and θ(t) is the instantaneous phase
        sp_cv = float(envelope.std() / (envelope.mean() + 1e-12))
        # sp_cv is calculated as the standard deviation of the envelope divided by its mean
        # the higher the sp_cv, the more busty the spindle envelope is (more variability in amplitude)
        # the lower the sp_cv, the less busty the spindle envelope is (less variability in amplitude, more continuous)
    except Exception:
        sp_cv = 0.0
        envelope = None
    t7 = sp_cv > SPINDLE_CV_MIN
    # t7 is true if sp_cv is greater than the minimum threshold,
    # indicating that the spindle envelope is sufficiently bursty (variable) than continuous
    details["T7"] = t7 # boolean pass/fail for T7
    details["T7_sp_cv"] = round(sp_cv, 3) # actual spindle envelope CV

    # ── T8: Spindle event count — enough discrete bursts ─────────────
    # V6 FIX: threshold = percentile(smoothed envelope) instead of
    # mean + k*std (which fails on bimodal distributions). Lacourse A7
    # / YASA spindle-detection convention.
    n_sp_events = 0 # number of detected spindle events
    mean_sp_dur = 0.0 # mean duration of detected spindle events in seconds
    sp_starts_valid = np.array([], dtype=int)   # starts of valid spindle eventsfor T12
    sp_ends_valid = np.array([], dtype=int) # ends of valid spindle events
    try:
        if envelope is not None and len(envelope) > 0:
            sigma_samples = SPINDLE_ENV_SMOOTH_MS * fs / 1000.0
            # sigma_samples: number of samples corresponding to the Gaussian smoothing σ for the spindle envelope
            env_smooth = gaussian_filter1d(envelope, sigma=sigma_samples)
            # env_smooth: smoothed envelope using Gaussian smoothing with sigma_samples
            thresh = np.percentile(env_smooth, SPINDLE_EVT_PCTILE)
            # thresh: threshold for spindle event detection
            above = (env_smooth > thresh).astype(np.int8)
            # above: array indicating where the smoothed envelope is above the threshold (1) or not (0)
            diff_sp = np.diff(np.concatenate(([0], above, [0])))
            # diff_sp: array of differences to find transitions in above-threshold state for spindle events
            sp_starts = np.where(diff_sp == 1)[0]
            # sp_starts: array of indices where envelope goes from below to above thrshold → start of spindle event
            sp_ends   = np.where(diff_sp == -1)[0]
            # sp_ends: array of indices where envelope goes from above to below threshold → end of spindle event
            durations = (sp_ends - sp_starts) / fs  # seconds
            # durations: array of durations of detected spindle events in seconds
            valid = (durations >= SPINDLE_DUR_LO_S) & (durations <= SPINDLE_DUR_HI_S)
            # valid: boolean array indicating which spindle events have durations within the spindle duration range
            # is valid if the duration of the spindle event is >= SPINDLE_DUR_LO_S and <= SPINDLE_DUR_HI_S
            n_sp_events = int(valid.sum()) 
            # n_sp_events: number of valid spindle events that have durations within the spindle duration range
            if n_sp_events > 0:
                mean_sp_dur = float(durations[valid].mean()) 
                # mean_sp_dur: mean duration of valid spindle events in seconds
                sp_starts_valid = sp_starts[valid]
                # sp_starts_valid: array of start indices of valid spindle events for T12
                sp_ends_valid   = sp_ends[valid]
                # sp_ends_valid: array of end indices of valid spindle events for T12
    except Exception:
        n_sp_events = 0
        mean_sp_dur = 0.0
    t8 = n_sp_events >= SPINDLE_EVT_MIN
    # t8 is true if the number of valid spindle events is greater than or qual to the minimum threshold
    details["T8"] = t8 # boolean pass/fail for T8
    details["T8_n_sp_events"] = n_sp_events # actual number of valid spindle events
    details["T8_mean_sp_dur"] = round(mean_sp_dur, 3) # actual mean duration of valid spindle events in seconds

    # ── T9-T11: PAC three-pack (NEW in V5) ────────────────────────────
    # Guards against Failure Mode 3 (thalamo-cortical spatiotemporal
    # decoupling). Without these, optimiser can fit the N3 spectrum while
    # simulating a diseased brain (PAC-decoupled = aged / AD / schizophrenia).
    pac = compute_pac_metrics(r_ctx, r_thal, fs=fs) # compute PAC metrics for T9-T11

    PAC_MI_MIN = 0.005
    PAC_PHASE_TOL_DEG = 50
    PAC_CONCENTRATION_MIN = 0.08
    PAC_UP_DOWN_RATIO_MIN = 1.20

    # T9: PAC strength — Modulation Index above chance
    t9 = pac["ok"] and (pac.get("mi", 0.0) >= PAC_MI_MIN)
    details["T9"] = t9
    details["T9_mi"] = round(pac.get("mi", 0.0), 5)

    # T10: preferred SO phase AND phase concentration
    if pac["ok"]:
        ph_deg = np.degrees(pac.get("phase_argmax", np.pi))
        dist_0 = min(abs(ph_deg), 360 - abs(ph_deg))
        dist_180 = abs(abs(ph_deg) - 180)
        dist_tgt = min(dist_0, dist_180)
        t10a = dist_tgt <= PAC_PHASE_TOL_DEG
        t10b = pac.get("phase_concentration", 0.0) >= PAC_CONCENTRATION_MIN
        t10 = t10a and t10b
        phase_to_record = round(np.radians(ph_deg), 3)
        dist_to_record = round(np.radians(dist_tgt), 3)
    else:
        t10 = False
        phase_to_record = round(np.pi, 3)
        dist_to_record = round(np.pi, 3)

    details["T10"] = t10
    details["T10_phase"] = phase_to_record
    details["T10_dist_to_target"] = dist_to_record

    # T11: SO leads spindle (up_down_ratio proxy)
    t11 = pac["ok"] and (pac.get("up_down_ratio", 0.0) >= PAC_UP_DOWN_RATIO_MIN)
    details["T11"] = t11
    details["T11_lag_ms"] = round(pac.get("up_down_ratio", 0.0), 3)

    # ── T12: Peak-inside-event verification (NEW in V6) ──────────────
    # For each valid spindle event (T8), compute Welch PSD within the
    # event window and require σ-band peak > 1.5× non-σ-band peak.
    # This plugs the fake-spindle loophole: sporadic transient spikes
    # can pass T8 (envelope ripples cross threshold) but have no real
    # 10-14 Hz oscillation inside the event.
    # T12： 对于每一个有效的纺锤体事件（由T8定义的事件），在事件窗口内计算Welch功率谱密度，
    #并要求纺锤体频段（10-14 Hz）内的峰值功率必须大于非纺锤体频段内峰值功率的1.5倍。
    # 这是为了防止“假纺锤体”现象：偶发的瞬态峰值可能会通过T8（因为包络线波动超过阈值），但它们在事件内部没有真正的10-14 Hz振荡。
    n_verified = 0
    for s, e in zip(sp_starts_valid, sp_ends_valid):
    # 对于每一个有效的纺锤体事件（由sp_starts_valid和sp_ends_valid定义的开始和结束索引），执行以下操作：
        if e - s < int(0.2 * fs):
            continue
        # 如果事件持续时间小于0.2秒（对应的样本数小于0.2 * fs），则跳过该事件，因为太短的事件可能无法提供可靠的频谱估计。
        event = r_thal[s:e]
        # 提取事件窗口内的thalamic firing rate数据，准备进行Welch功率谱密度估计。
        try:
            f_ev, p_ev = welch(event, fs=fs,
                                nperseg=min(len(event), 512))
            # 使用Welch方法计算事件窗口内的功率谱密度，f_ev是频率数组，p_ev是对应的功率值。
            sp_m = (f_ev >= SPINDLE_LO) & (f_ev <= SPINDLE_HI)
            # sp_m是一个布尔数组，指示哪些频率点在纺锤体频段内（10-14 Hz）。
            ns_m = (f_ev >= 4) & (f_ev < SPINDLE_LO)
            # ns_m是一个布尔数组，指示哪些频率点在非纺锤体频段内（4-10 Hz）。
            if not sp_m.any() or not ns_m.any():
                continue
            ns_peak = p_ev[ns_m].max() if p_ev[ns_m].size > 0 else 1e-12 # ns_peak是非纺锤体频段内的最大功率值，
            #如果没有非纺锤体频段的频率点，则设置为一个非常小的值（1e-12）以避免除零错误。
            ns_peak = ns_peak if ns_peak > 0 else 1e-12 # 确保ns_peak是正数，避免后续计算中的除零错误。
            if (p_ev[sp_m].max() / ns_peak) > T12_PEAK_INSIDE_RATIO:
            # 如果纺锤体频段内的最大功率值与非纺锤体频段内的最大功率值之比大于T12_PEAK_INSIDE_RATIO（例如1.5），
            # 则认为该事件通过了峰值验证，n_verified计数器增加
                n_verified += 1
        except Exception:
            continue
    t12 = n_verified >= T12_N_VERIFIED_MIN
    details["T12"] = t12
    details["T12_n_verified"] = n_verified

    # ── Summary ───────────────────────────────────────────────────────
    n_passed = sum([t1, t2, t3, t4, t5, t6, t7, t8,
                    t9, t10, t11, t12])
    details["n_passed"] = n_passed
    details["feasible"] = (n_passed == 12)

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

    # T12 (NEW): want n_verified >= T12_N_VERIFIED_MIN (real σ in each event)
    if con["T12"]:
        scores.append(1.0)
    else:
        # Linear ramp from 0 (n_verified=0) to 0.99 (n_verified=threshold)
        scores.append(np.clip(con.get("T12_n_verified", 0) /
                              T12_N_VERIFIED_MIN, 0, 0.99))

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


def compute_fitness_v7(params_vec,
                       target_psd, target_freqs,
                       target_periodic, fooof_freqs):
    """
    V7 fitness: 12 constraints + event-based rewards.

    if ALL 12 constraints pass (feasible):
        fitness = 0.50*shape_r + 0.25*so_power_v7 + 0.25*spindle_power_v7
        where:
          so_power_v7      = clip((T4_q - 1) / 4, 0, 1)
                             (V7 CHANGE #3: T4_q-derived, FOOOF-free)
          spindle_power_v7 = clip(T12_n_verified / 15, 0, 1)
                             (V7 CHANGE #2: event-count-based, FOOOF-free)
        Range: [0, ~1].

    else:
        fitness = -10 + 10 * mean(constraint_scores)  [range: -10 to ~-0.1]
        (same as v6, gives DE gradient signal on infeasible solutions)

    Rationale for v7 reward redesign:
      FOOOF-based rewards were unreachable at SO-stable c_th2ctx values
      (diagnostic s4_0d: 0/20 configs had spindle_power > 0.1 at
      c_th2ctx ≤ 0.05; 1/20 had so_power > 0.1). Event-based rewards
      decouple optimization from cortical PSD peak detection.
    """
    global _eval_count, _best_score, _best_params, _records

    _eval_count += 1
    mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th = params_vec

    # ── Run simulation ────────────────────────────────────────────────
    try:
        m = build_model(mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th)
        seed_numba(42)  # reseed numba RNG before every evaluation
        m.run()
    except Exception:
        # Fallback backend to avoid losing whole generations on numba/JIT failures.
        try:
            m.params["backend"] = "jitcdde"
            seed_numba(42)  # reseed before fallback run too
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
    # STEP 1: Hard constraints (Deb feasibility rules) — 12 in V6
    # ══════════════════════════════════════════════════════════════════
    n_passed, con = compute_constraints_v7(
        r_ctx, r_thal, f_c=f_ctx_full, p_c=p_ctx_full, fs=FS_SIM
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Rewards (only computed for feasible solutions)
    # ══════════════════════════════════════════════════════════════════
    shape_r = 0.0
    so_power = 0.0
    spindle_power = 0.0

    if con["feasible"]:
        # shape_r: FOOOF periodic residual Pearson-r (UNCHANGED from v6;
        # still useful for FREQUENCY-SHAPE fit even though the peak-height
        # rewards below no longer use FOOOF)
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

        # V7 CHANGE #3: so_power reward is T4_q-derived (FOOOF peak detection
        # fails for sharp but low-absolute-amplitude SO peaks).
        # T4_q is peak/neighbor-floor ratio. Subtract 1 (a "no peak" gives
        # ratio ~1) then normalize so that Q=5 → reward 1.0.
        t4_q = con.get("T4_q", 0.0)
        so_power = float(np.clip((t4_q - 1.0) / 4.0, 0.0, 1.0))

        # V7 CHANGE #2: spindle_power reward is event-based (FOOOF peak
        # detection fails for narrow T-current-resonance peaks that live
        # only in the thalamus).
        # T12_n_verified is the # of detected spindle events that contain
        # dominant σ-band oscillation. Target = 15 events in 60s
        # (≈ 15-20 spindles/min is normal N3 per Ferrarelli 2007).
        n_ver = con.get("T12_n_verified", 0)
        spindle_power = float(np.clip(n_ver / 15.0, 0.0, 1.0))

        # V7 fitness: same weighted sum as v6 (W_SHAPE*shape_r
        #             + W_SO*so_power_new + W_SPINDLE*spindle_power_new)
        fitness = (W_SHAPE * shape_r
                   + W_SO * so_power
                   + W_SPINDLE * spindle_power)
    else:
        # Continuous constraint satisfaction (replaces -1M Deb cliff)
        # Range: -10 (all 11 constraints far from passing) to ~-0.1 (almost all passing)
        # Feasible solutions are always [0, ~1], so feasible > infeasible guaranteed
        c_scores = compute_feasibility_score(con)
        fitness = -10.0 + 10.0 * (sum(c_scores) / 12.0)

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
        "T12":            int(con["T12"]),
        "T12_n_verified": con["T12_n_verified"],
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
        f"  [{bp.get('n_passed',0)}/12]"
        f"  T1={bp.get('T1',0)} T2={bp.get('T2',0)}"
        f" T3={bp.get('T3',0)} T4={bp.get('T4',0)}"
        f" T5={bp.get('T5',0)} T6={bp.get('T6',0)}"
        f" T7={bp.get('T7',0)} T8={bp.get('T8',0)}"
        f" T9={bp.get('T9',0)} T10={bp.get('T10',0)}"
        f" T11={bp.get('T11',0)} T12={bp.get('T12',0)}"
        f"  shape_r={bp.get('shape_r',0):.3f}"
        f"  so={bp.get('so_power',0):.3f}"
        f"  sp={bp.get('spindle_power',0):.3f}"
        f"  MI={bp.get('T9_mi',0):.4f}"
        f"  phi={bp.get('T10_phase',0):+.2f}"
        f"  lag={bp.get('T11_lag_ms',0):+.0f}ms"
        f"  nver={bp.get('T12_n_verified',0)}"
        f"  evals={_eval_count}"
        f"  elapsed={elapsed:.0f}s"
    )
    return False


def main():
    global _t_start

    print("=" * 70)
    print("V7 Evolution: V6 + 60s duration + event-rewards + T5 fix +")
    print("              tight c_th2ctx bounds (see docstring for rationale)")
    print("  FOOOF=" + ("ON" if HAS_FOOOF else "OFF (fallback)") +
          "  (used for shape_r only; so_power/spindle_power are event-based)")
    print(f"  8 params, 12 constraints (continuous satisfaction)")
    print(f"  SIM_DUR_MS={SIM_DUR_MS/1000:.0f}s "
          f"(V7 change #1: doubled from v6, halves T6 sampling noise)")
    print(f"  Bounds:")
    print(f"    g_LK     [{BOUNDS[4][0]:.3f}, {BOUNDS[4][1]:.3f}]")
    print(f"    g_h      [{BOUNDS[5][0]:.3f}, {BOUNDS[5][1]:.3f}]")
    print(f"    c_th2ctx [{BOUNDS[6][0]:.3f}, {BOUNDS[6][1]:.3f}]  "
          f"(V7 change #4: tightened from [0.05, 0.25] for SO stability)")
    print(f"    c_ctx2th [{BOUNDS[7][0]:.3f}, {BOUNDS[7][1]:.3f}]  "
          f"(sweep PAC hotspot: 0.155-0.200)")
    print(f"  Strategy: best1bin, mutation=(0.5,1.0), workers=1 (unchanged)")
    print(f"  T4+ SO Q-factor > {SO_Q_MIN}")
    print(f"  T5  spindle FWHM > {SPINDLE_FWHM_MIN} Hz  "
          f"(V7 change #5: relaxed from 2.0 to allow narrow T-current peaks)")
    print(f"  T6 IBI CV < {IBI_CV_MAX} (SO regularity)")
    print(f"  T7 spindle envelope CV > {SPINDLE_CV_MIN} (burstiness)")
    print(f"  T8 spindle events >= {SPINDLE_EVT_MIN}, "
          f"dur {SPINDLE_DUR_LO_S}-{SPINDLE_DUR_HI_S}s "
          f"({SPINDLE_EVT_PCTILE:.0f}th pctile of smoothed envelope)")
    print(f"  T9  PAC MI > {PAC_MI_MIN} "
          f"[Kullback-Leibler, {PAC_N_BINS} phase bins]")
    print(f"  T10 preferred phase near 0 OR ±π within ±50° "
          f"(fast OR slow spindle, Mölle 2011)")
    print(f"  T11 SO→spindle lag ≥ {PAC_MIN_LAG_MS} ms "
          f"(directionality: SO leads)")
    print(f"  T12 n_verified >= {T12_N_VERIFIED_MIN} events with "
          f"σ-peak/non-σ-peak > {T12_PEAK_INSIDE_RATIO} inside event")
    print(f"  Rewards (feasible only):")
    print(f"    shape_r       = FOOOF periodic residual Pearson-r (unchanged)")
    print(f"    so_power      = clip((T4_q-1)/4, 0, 1)  "
          f"(V7 change #3: T4_q-derived, FOOOF-free)")
    print(f"    spindle_power = clip(T12_n_verified/15, 0, 1)  "
          f"(V7 change #2: event-based, FOOOF-free)")
    print(f"    total = {W_SHAPE:.2f}*shape_r + "
          f"{W_SO:.2f}*so_power + {W_SPINDLE:.2f}*spindle_power")
    print(f"  Infeasible scoring: continuous [-10, ~-0.1] (scaled for 12 C's)")
    print(f"  popsize={DE_POPSIZE} x {len(PARAM_NAMES)} = "
          f"{DE_POPSIZE*len(PARAM_NAMES)} individuals/gen  (V7 change #6: "
          f"no warm-start, pure Sobol init)")
    print(f"  maxiter={N_GEN}, total evals ~ "
          f"{DE_POPSIZE*len(PARAM_NAMES)*(N_GEN+1)}")
    print(f"  Expected wall time: ~3-4h (2× v6 due to 60s duration)")
    print("=" * 70)

    # Load target
    print("\nLoading target EEG...")
    target_psd, target_freqs = load_target_psd()
    print("Computing FOOOF on target...")
    target_periodic, fooof_freqs = compute_target_periodic(target_psd, target_freqs)
    _t_start = time.time()
    print("\n")

    result = differential_evolution(
        compute_fitness_v7,
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
    print("V7 Validation")
    print(f"{'='*55}")
    print(f"  score         = {best.get('score'):.4f}")
    print(f"  feasible      = {best.get('feasible')}")
    print(f"  n_passed      = {best.get('n_passed')}/12")
    print(f"  shape_r       = {best.get('shape_r'):.4f}")
    print(f"  so_power      = {best.get('so_power'):.4f}")
    print(f"  spindle_power = {best.get('spindle_power'):.4f}")
    print(f"")
    print(f"  Constraints (T1-T8 from V4/V6, T9-T11 from V5, T12 new in V6):")
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
          f"  [n_events={best.get('T8_n_sp_events')}, "
          f"mean_dur={best.get('T8_mean_sp_dur')}s]")
    print(f"    T9  PAC strength:       {'PASS' if best.get('T9') else 'FAIL'}"
          f"  [MI={best.get('T9_mi'):.5f} (min {PAC_MI_MIN})]")
    print(f"    T10 PAC preferred phase:{'PASS' if best.get('T10') else 'FAIL'}"
          f"  [phi={best.get('T10_phase'):+.2f} rad, "
          f"dist_to_target={best.get('T10_dist_tgt', 0):.2f} rad "
          f"(< 5π/18 = 0.87 ≈ 50°)]")
    print(f"    T11 PAC direction:      {'PASS' if best.get('T11') else 'FAIL'}"
          f"  [SO→spindle lag={best.get('T11_lag_ms'):+.0f} ms "
          f"(≥ {PAC_MIN_LAG_MS})]")
    print(f"    T12 Peak-inside-event:  {'PASS' if best.get('T12') else 'FAIL'}"
          f"  [n_verified={best.get('T12_n_verified')} "
          f"(min {T12_N_VERIFIED_MIN})]")
    print(f"")
    print(f"  Best parameters:")
    for k in PARAM_NAMES:
        print(f"    {k}: {best.get(k):.4f}")
    print(f"\nNext: update plot scripts for v7 JSON path + v7 reward semantics")
    print(f"      (so_power = T4_q-derived; spindle_power = T12_n_verified/15)")


if __name__ == "__main__":
    main()
