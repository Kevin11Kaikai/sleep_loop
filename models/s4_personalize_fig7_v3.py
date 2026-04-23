"""
s4_personalize_fig7_v3.py
=========================
Physics-constrained thalamocortical personalisation 鈥?Version 3.

Why the UP-state tests matter:
  v2 could reach dynamics=1.0 by staying in a persistent DOWN state
  (min r_E < 1 Hz satisfies T1; SO "peak" from noise satisfies T4; thalamic
  limit-cycle satisfies T5). The new T2+T3 require genuine UP/DOWN
  alternation with cortical firing rates > 15 Hz during UP states, matching
  the 20-30 Hz range visible in Fig. 7(c) of Cakan et al. 2023.

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
DE_POPSIZE      = 20            # DE population size (default 15*len(params)=120, but reduce for speed)
N_GEN           = 30            # DE generations (default 100, but reduce for speed)

PARAMS_PATH  = f"data/patient_params_fig7_v3_{SUBJECT_ID}.json"
RECORDS_PATH = "outputs/evolution_fig7_v3_records.csv"

# Evolutionary FOOOF Parameters (same as v2)
EVO_FOOOF_PARAMS = dict(
    peak_width_limits=[1.0, 8.0],
    max_n_peaks=4,
    min_peak_height=0.05,
    aperiodic_mode="fixed",
)

# Parameter names and bounds (same as v2)
PARAM_NAMES = ["mue", "mui", "b", "tauA", "g_LK", "g_h", "c_th2ctx", "c_ctx2th"]
BOUNDS = [
    (2.5,  4.5),    # mue
    (2.5,  5.0),    # mui
    (10.0, 40.0),   # b   [pA]
    (800., 2000.),  # tauA [ms]
    (0.03, 0.15),   # g_LK [mS/cm^2]
    (0.03, 0.15),   # g_h  [mS/cm^2
    (0.05, 0.25),   # c_th2ctx
    (0.001, 0.05),  # c_ctx2th  
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


_T1_WEIGHT = 0.20   # DOWN state exists
_T2_WEIGHT = 0.25   # UP state exists (peak > 15 Hz)
_T3_WEIGHT = 0.25   # UP state sustained (longest run > 15 Hz 鈮?100 ms)
_T4_WEIGHT = 0.15   # SO peak frequency in [0.3, 1.5] Hz
_T5_WEIGHT = 0.15   # Spindle FWHM > 2 Hz (waxing-waning, not limit cycle)

# Thresholds
UP_THRESH_HZ     = 15.0    # Hz  minimum firing rate to count as UP state (visible in Fig. 7(c) of Cakan et al. 2023 as the lower bound of the 20鈥?0 Hz range during UP states)
UP_DURATION_MS   = 100.0   # ms  minimum duration above UP_THRESH_HZ to count as sustained UP state (visible in Fig. 7(c) of Cakan et al. 2023 as the typical duration of UP states)
DOWN_THRESH_HZ   = 1.0     # Hz firing rate below this counts as DOWN state (visible in Fig. 7(c) of Cakan et al. 2023 as the upper bound of the <1 Hz range during DOWN states)
SO_FREQ_LO       = 0.2     # Hz minimum frequency for SO peak (visible in Fig. 7(c) of Cakan et al. 2023 as the lower bound of the 0.3鈥?.5 Hz range for the SO peak)
SO_FREQ_HI       = 1.5     # Hz maximum frequency for SO peak (visible in Fig. 7(c) of Cakan et al. 2023 as the upper bound of the 0.3鈥?.5 Hz range for the SO peak)
SPINDLE_LO       = 10.0     # Hz minimum frequency for spindle peak (visible in Fig. 7(c) of Cakan et al. 2023 as the lower bound of the 8鈥?12 Hz range for spindles)
SPINDLE_HI       = 14.0    # Hz maximum frequency for spindle peak (visible in Fig. 7(c) of Cakan et al. 2023 as the upper bound of the 8鈥?12 Hz range for spindles)
SPINDLE_FWHM_MIN = 2.0     # Hz minimum full width at half maximum for spindle peak (visible in Fig. 7(c) of Cakan et al. 2023 as the lower bound of the 2鈥?4 Hz range for spindle FWHM)
# Plan A: soft width gate for spindle reward
FWHM_SOFT_START  = 2.5     # Hz: score starts to grow from this width
FWHM_SOFT_SPAN   = 1.5     # Hz: reaches full score by start+span (i.e. 4 Hz)
SPINDLE_GATE_ALPHA = 0.10  # keeps 10% spindle reward even for narrow peaks

# Objective weights
W_SHAPE = 0.45
W_SO = 0.10
W_SP = 0.10
W_DYN = 0.35
# slow oscillation overshoot penalty 
SO_TARGET_MAX = 0.9
SO_OVERSHOOT_LAMBDA = 0.2
NARROW_PENALTY_FLOOR = 0.10
LAMBDA_NARROW = 0.12 # penalty for narrow spindle peaks (FWHM < 2 Hz), scaled by how much the spindle power exceeds the target (i.e. no penalty if spindle power is low, but increasing penalty for stronger but narrow spindle peaks that fail T5)
BAD_OBJECTIVE = 1e6 # DE minimizes objective, so a very large value reliably marks invalid simulations as worst-case.
def compute_dynamics_score_v3(r_ctx, r_thal, f_c=None, p_c=None, fs=FS_SIM):
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

    # T1: DOWN state exists
    min_rE = float(r_ctx.min())
    t1     = min_rE < DOWN_THRESH_HZ # require at least one sample below DOWN_THRESH_HZ; this rejects persistent-UP-state solutions that v2 could not reject.
    details["T1_down_exists"]  = t1 # visible in Fig. 7(c) of Cakan et al. 2023 as the presence of a <1 Hz range during DOWN states
    details["T1_min_rE_Hz"]    = min_rE # visible in Fig. 7(c) of Cakan et al. 2023 as the upper bound of the <1 Hz range during DOWN states
    if t1:
        score += _T1_WEIGHT # if the condition of t1 is satisfied, add the weight of T1 to the score. This rewards solutions that have at least some DOWN state activity below DOWN_THRESH_HZ, which is visible in Fig. 7(c) of Cakan et al. 2023 as the presence of a <1 Hz range during DOWN states.

    # T2: UP state exists
    max_rE = float(r_ctx.max())  
    t2     = max_rE > UP_THRESH_HZ # require at least one sample above UP_THRESH_HZ; this rejects persistent-DOWN-state solutions that v2 could not reject.
    details["T2_up_exists"]    = t2 
    details["T2_max_rE_Hz"]    = max_rE
    if t2:
        score += _T2_WEIGHT # if the condition of t2 is satisfied, add the weight of T2 to the score. This rewards solutions that have at least some UP state activity above UP_THRESH_HZ, which is visible in Fig. 7(c) of Cakan et al. 2023 as the presence of a 20鈥?0 Hz range during UP states.

    # T3: UP state sustained
    min_run_samples = int(UP_DURATION_MS * fs / 1000.0) # minimum number of consecutive samples above UP_THRESH_HZ to count as a sustained UP state; this rejects solutions with only brief noise-driven excursions above UP_THRESH_HZ that would fail to capture the sustained nature of UP states visible in Fig. 7(c) of Cakan et al. 2023.
    above           = (r_ctx > UP_THRESH_HZ).astype(np.int8) # above is defined as a binary array where samples above UP_THRESH_HZ are 1 and others are 0; this is used to identify runs of consecutive samples above UP_THRESH_HZ.
    # Vectorised run-length encoding
    diff   = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    max_run = int((ends - starts).max()) if len(starts) > 0 else 0
    t3      = max_run >= min_run_samples # t3 requires at least one run of consecutive samples above UP_THRESH_HZ that 
    #is long enough to count as a sustained UP state.
    details["T3_up_sustained"]        = t3
    details["T3_longest_up_run_ms"]   = max_run * 1000.0 / fs
    if t3:
        score += _T3_WEIGHT # if the condition of t3 is satisfied, add the weight of T3 to the score.

    # T4: SO frequency in range
    if f_c is None or p_c is None:
        f_c, p_c = compute_epoch_psd(r_ctx, fs) #f_c and p_c are the frequencies and power spectral density
        #of the cortical firing rate, computed using the compute_epoch_psd function defined in utils/02_preprocess_psd.py. This is used to identify the peak frequency of the slow oscillation in the cortical firing rate.
    so_mask  = (f_c >= SO_FREQ_LO) & (f_c <= SO_FREQ_HI) # so_mask is a boolean array that identifies the frequencies
    # in f_c that are within the SO_FREQ_LO and SO_FREQ_HI range; this is used to find the peak frequency of the slow oscillation within the expected range for N3 sleep, which is visible in Fig. 7(c) of Cakan et al. 2023 as the 0.3鈥?.5 Hz peak during N3 sleep.
    if so_mask.any():
        so_peak_freq = float(f_c[so_mask][np.argmax(p_c[so_mask])]) # so_peak_freq is the peak frequency
    else:
        so_peak_freq = 0.0
    t4 = SO_FREQ_LO <= so_peak_freq <= SO_FREQ_HI # t4 is true if the peak frequency of the slow oscillation
    # is within the expected range for N3. 
    details["T4_so_in_range"]      = t4
    details["T4_so_peak_freq_Hz"]  = so_peak_freq
    if t4:
        score += _T4_WEIGHT # if the condition of t4 is satisfied, add the weight of T4 to the score. This rewards solutions that have a slow oscillation peak frequency within the expected range for N3 sleep, which is visible in Fig. 7(c) of Cakan et al. 2023 as the 0.3鈥?.5 Hz peak during N3 sleep.

    # T5: Spindle width (waxing-waning, not limit cycle)
    f_th, p_th = compute_epoch_psd(r_thal, fs) 
    # f_th and p_th are the frequencies and power spectral density of the thalamus firing rate, r_thal, simulated firing rate
    # of thalamus, fs, sampling rate of the simulation
    sp_mask = (f_th >= SPINDLE_LO) & (f_th <= SPINDLE_HI) # sp_mask is a boolean array that identifies the frequencies
    fwhm    = 0.0
    if sp_mask.any() and p_th[sp_mask].max() > 0:
        p_sp        = p_th[sp_mask] # p_sp is the power spectral density values corresponding to the frequencies 
        # in f_th that are within the SPINDLE_LO and SPINDLE_HI range; this is used to find the peak frequency and width of the spindle oscillation within the expected range for spindles, which is visible in Fig. 7(c) of Cakan et al. 2023 as the 8鈥?12 Hz peak during N3 sleep.
        f_sp        = f_th[sp_mask] # f_sp is the frequencies corresponding to the power spectral density values in p_sp.
        half_power  = p_sp.max() / 2.0 # half_power is the half of the maximum power spectral density value 
        above_half  = f_sp[p_sp >= half_power] # above_half is an array of frequencies in f_sp where the power spectral density in p_sp is greater than or equal to half_power; this is used to calculate the full width at half maximum (FWHM) of the spindle peak, which is important for distinguishing genuine waxing-waning spindles from non-physiological limit-cycle oscillations that can arise with strong thalamocortical coupling. A genuine spindle should have a wider peak (FWHM > 2 Hz) compared to a narrow limit-cycle oscillation.
        if len(above_half) >= 2:
            fwhm = float(above_half[-1] - above_half[0]) # if the length of above_half is at least 2, 
            # calculate the full width at half maximum (FWHM) of the spindle peak as the difference
            # between the maximum and minimum frequencies in above_half;
            # this is used to determine if the spindle peak is wide enough to count as a genuine waxing-waning spindle rather than a narrow limit-cycle oscillation.
    t5 = fwhm > SPINDLE_FWHM_MIN # t5 is true if the FWHM of the spindle peak is greater than SPINDLE_FWHM_MIN, which is used to determine if the spindle peak is wide enough to count as a genuine waxing-waning spindle rather than a narrow limit-cycle oscillation. This is important for ensuring that the model captures the characteristic waxing-waning nature of spindles observed in Fig. 7(c) of Cakan et al. 2023, rather than non-physiological limit-cycle oscillations that can arise with strong thalamocortical coupling.
    t5_score_cont = float(np.clip((fwhm - FWHM_SOFT_START) / FWHM_SOFT_SPAN, 0.0, 1.0)) # t5_score_cont is a continuous score for the spindle width test, calculated as a linear function of the FWHM of the spindle peak, with a soft threshold defined by FWHM_SOFT_START and FWHM_SOFT_SPAN. This allows for partial credit for spindle peaks that are wider than FWHM_SOFT_START but do not fully reach SPINDLE_FWHM_MIN, which can help guide the evolutionary algorithm towards solutions with wider spindle peaks even if they do not fully satisfy the hard threshold of t5.
    details["T5_spindle_waxing"]  = t5 # T5 spindle waxing is true if the FWHM of the spindle peak is greater than SPINDLE_FWHM_MIN, which indicates that the spindle peak is wide enough to count as a genuine waxing-waning spindle rather than a narrow limit-cycle oscillation. This is important for ensuring that the model captures the characteristic waxing-waning nature of spindles observed in Fig. 7(c) of Cakan et al. 2023, rather than non-physiological limit-cycle oscillations that can arise with strong thalamocortical coupling.
    details["T5_spindle_fwhm_Hz"] = fwhm # T5 spindle FWHM in Hz is the full width at half maximum of the spindle peak, which is used to determine if the spindle peak is wide enough to count as a genuine waxing-waning spindle rather than a narrow limit-cycle oscillation. This is important for ensuring that the model captures the characteristic waxing-waning nature of spindles observed in Fig. 7(c) of Cakan et al. 2023, rather than non-physiological limit-cycle oscillations that can arise with strong thalamocortical coupling.
    details["T5_spindle_score_cont"] = t5_score_cont # T5 spindle score continuous is a continuous score for the spindle width test
    score += _T5_WEIGHT * t5_score_cont # if the condition of t5 is satisfied, add the weight of T5 to the score.

    return round(score, 4), details


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


def compute_fitness_v3(params_vec,
                       target_psd, target_freqs,
                       target_periodic, fooof_freqs):
    """
    Returns negative fitness (scipy minimises).

    fitness = (W_SHAPE * shape_r
    + W_SO * so_power
    + W_SP * spindle_power_eff
    + W_DYN * dynamics_score
    - so_overshoot_penalty- narrow_spindle_penalty)
    """
    global _eval_count, _best_score, _best_params, _records

    _eval_count += 1
    mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th = params_vec

    # build_model aims to construct and simulate the thalamocortical model with the given parameters, and return the MultiModel object after simulation. We wrap this in a try-except block to catch any exceptions that may arise during model construction or simulation (e.g. due to non-physiological parameter combinations leading to numerical instability), and return a fitness of 0.0 in those cases so that the evolutionary algorithm treats them as worst-case solutions. 
    try:
        m = build_model(mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th)
        m.run()
    except Exception:
        return BAD_OBJECTIVE

    # r_exc is the cortical excitatory firing rate time series extracted from the MultiModel object after simulation, using the key f"r_mean_{EXC}" which corresponds to the state variable we designated as the default output for fitness evaluation. This is the main variable we will analyze to compute the fitness components, as it represents the cortical activity that we want to match to the target N3 EEG PSD and dynamics.
    r_exc = m[f"r_mean_{EXC}"]
    
    # Depending on the DE strategy and the dimensionality of the output, r_exc may be a 1D array (if only one EXC mass in ALN) or a 2D array (if two EXC masses in ALN, with shape [2, n_samples]). We need to handle both cases to extract the cortical firing rate time series correctly. If r_exc is 2D with at least 2 rows, we take the first row as the cortical firing rate and ignore the second row (which would correspond to the second EXC mass in ALN if it exists). If r_exc is 1D, we take it as is. In either case, we multiply by 1000 to convert from kHz to Hz, since the model outputs firing rates in kHz but we want them in Hz for comparison with the target EEG data and for computing the dynamics score.
    if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
        r_ctx  = r_exc[0, :] * 1000.0 # r_ctx is the cortical firing rate time series in Hz, extracted from the first row of r_exc if it is 2D, or taken directly from r_exc if it is 1D. This is the main variable we will analyze to compute the fitness components related to the cortical activity, including the shape similarity to the target PSD and the presence of genuine N3-like dynamics.
        r_thal = r_exc[1, :] * 1000.0 # r_thal is the thalamic firing rate time series in Hz, extracted from the second row of r_exc if it is 2D, or set to zeros if r_exc is 1D. This is used for computing the spindle width test in the dynamics score, as genuine spindles should be visible in the thalamic firing rate. If r_exc is 1D, we set r_thal to zeros since we do not have a separate thalamic firing rate to analyze.
    else:
        r_ctx  = (r_exc[0] if r_exc.ndim == 2 else r_exc) * 1000.0
        r_thal = np.zeros_like(r_ctx)

    # Burn-in: discard first 5 s
    n_drop = int(5.0 * FS_SIM)
    r_ctx  = r_ctx[n_drop:]
    r_thal = r_thal[n_drop:]

    if r_ctx.max() < 0.1:
        return BAD_OBJECTIVE

    # f_ctx_full and p_ctx_full are the frequencies and power spectral density of the cortical firing rate time series,
    # computed using the compute_epoch_psd function defined in utils/02_preprocess_psd.py. This is used to analyze the spectral properties of the cortical activity and compare it to the target N3 EEG PSD, as well as to compute the shape similarity component of the fitness based on the periodic component extracted by FOOOF.
    f_ctx_full, p_ctx_full = compute_epoch_psd(r_ctx, FS_SIM) # f_ctx_full是 cortical firing rate time series的频率数组
    #，p_ctx_full是对应的功率谱密度数组，使用compute_epoch_psd函数计算得到。
    mask = (f_ctx_full >= F_LO) & (f_ctx_full <= F_HI)
    # mask的作用是选择f_ctx_full中在F_LO和F_HI范围内的频率索引
    f_ctx, p_ctx = f_ctx_full[mask], p_ctx_full[mask]
    # f_ctx和p_ctx是根据mask从f_ctx_full和p_ctx_full中选择出的频率和功率谱密度数组，分别对应于F_LO和F_HI范围内的频率。
    shape_r = 0.0
    if HAS_FOOOF and target_periodic is not None:
        try:


            p_interp = interp1d(
                f_ctx, p_ctx, bounds_error=False, fill_value=1e-30,
            )(fooof_freqs)
            # p_interp是将p_ctx插值到fooof_freqs对应的频率点上的功率谱密度值，使用scipy.interpolate.interp1d函数进行插值。
            # 这个插值是为了将模型的PSD与FOOOF提取的周期成分进行比较，因为FOOOF提取的周期成分是在fooof_freqs对应的频率点上定义的。
            fm_sim = FOOOF(**EVO_FOOOF_PARAMS)
            # fm_sim是一个FOOOF对象，使用与EVO_FOOOF_PARAMS相同的参数进行初始化。我们将使用这个对象来拟合模型的PSD，
            # 以提取模型的周期成分并与目标周期成分进行比较。
            fm_sim.fit(fooof_freqs, p_interp, [F_LO, F_HI])
            # fm_sim.fit方法用于拟合模型的PSD，输入是fooof_freqs对应的频率点、p_interp对应的功率谱密度值，
            # 以及拟合的频率范围[F_LO, F_HI]。拟合完成后，fm_sim对象将包含模型PSD的拟合结果，包括提取的周期成分和拟合的非周期成分。

            sim_log = np.log10(p_interp[: len(fm_sim._ap_fit)] + 1e-30)
            # sim_log是模型PSD的对数值，取p_interp中与fm_sim._ap_fit长度相同的部分，并加上一个小常数以避免对数零的情况。这个sim_log将用于计算模型的周期成分。
            sim_periodic = sim_log - fm_sim._ap_fit
            # sim_periodic是模型的周期成分，通过从模型PSD的对数值中减去拟合的非周期成分得到。
            n_r = min(len(sim_periodic), len(target_periodic))
            # n_r是sim_periodic和target_periodic中较短的长度，用于确保在计算相关系数时两者长度一致。
            shape_r, _ = pearsonr(sim_periodic[:n_r], target_periodic[:n_r])
            # shape_r是模型周期成分与目标周期成分之间的Pearson相关系数，计算时只使用前n_r个频率点，以确保两者长度一致。
            # 这个shape_r将作为形状相似度的指标，反映模型PSD的周期成分与目标N3 EEG PSD的周期成分之间的相似程度。
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


    so_power      = 0.0 # so_power是模型PSD中slow oscillation频段（SO_FREQ_LO到SO_FREQ_HI）内的最大功率值，
    # 用于评估模型在slow oscillation频段的表现。
    # 这个指标反映了模型是否成功地在slow oscillation频段产生了足够的功率，以匹配目标N3 EEG PSD中slow oscillation的特征。
    spindle_power = 0.0 # spindle_power是模型PSD中spindle频段（SPINDLE_LO到SPINDLE_HI）内的最大功率值，
    # 用于评估模型在spindle频段的表现。这个指标反映了模型是否成功地在spindle频段产生了足够的功率，以匹配目标N3 EEG PSD中spindle的特征。
    if HAS_FOOOF:
        try:
            p_interp2 = interp1d(f_ctx, p_ctx,
                                 bounds_error=False, fill_value=1e-30)(fooof_freqs)
            # p_interp2是将模型的PSD插值到fooof_freqs对应的频率点上的功率谱密度值，使用scipy.interpolate.interp1d函数进行插值。
            fm2 = FOOOF(**EVO_FOOOF_PARAMS)
            # fm2是一个FOOOF对象，使用与EVO_FOOOF_PARAMS相同的参数进行初始化。我们将使用这个对象来拟合模型的PSD，
            fm2.fit(fooof_freqs, p_interp2, [F_LO, F_HI])
            # 以提取模型的周期成分并与目标周期成分进行比较。
            for pk in fm2.peak_params_: # for循环遍历fm2.peak_params_中的每个峰参数，pk是一个包含峰频率、峰功率和峰宽度的数组。
                # 我们将检查这些峰是否位于slow oscillation频段或spindle频段内，并更新so_power和spindle_power以记录这些频段内的最大峰功率。
                freq, power, _ = pk # 从峰参数中提取峰频率和峰功率，忽略峰宽度（用_表示）。
                # 我们将使用freq和power来检查这个峰是否位于slow oscillation频段或spindle频段内，并更新相应的功率指标。
                if SO_FREQ_LO <= freq <= SO_FREQ_HI: # 如果峰频率位于slow oscillation频段内,
                    so_power = max(so_power, float(power)) # 更新so_power为当前so_power和这个峰功率的较大值，
                    # 以记录slow oscillation频段内的最大峰功率。
                    
                if SPINDLE_LO <= freq <= SPINDLE_HI: # 如果峰频率位于spindle频段内,
                    spindle_power = max(spindle_power, float(power))
                    # 更新spindle_power为当前spindle_power和这个峰功率的较大值，
                    # 以记录spindle频段内的最大峰功率。
                    
        except Exception:
            pass
    dynamics_score, dyn_details = compute_dynamics_score_v3(
        r_ctx, r_thal, f_c=f_ctx_full, p_c=p_ctx_full, fs=FS_SIM
    ) # dynamics_score is the score computed by the compute_dynamics_score_v3 function
    spindle_gate = SPINDLE_GATE_ALPHA + (1.0 - SPINDLE_GATE_ALPHA) * dyn_details["T5_spindle_score_cont"]
    spindle_power_eff = spindle_power * spindle_gate
    so_overshoot = max(so_power - SO_TARGET_MAX, 0.0)
   
    so_overshoot_penalty = SO_OVERSHOOT_LAMBDA * so_overshoot
    T5_score_cont = dyn_details["T5_spindle_score_cont"]
    narrow_spindle_penalty = LAMBDA_NARROW * (NARROW_PENALTY_FLOOR + (1.0 - NARROW_PENALTY_FLOOR) * (1.0 - T5_score_cont)) * spindle_power
    fitness = (W_SHAPE * shape_r
    + W_SO * so_power
    + W_SP * spindle_power_eff
    + W_DYN * dynamics_score
    - so_overshoot_penalty- narrow_spindle_penalty)
    # Plan A: gate spindle peak reward by continuous spindle-width quality.
  

    record = dict(zip(PARAM_NAMES, params_vec))
    record.update({
        "score":          round(fitness, 6),
        "shape_r":        round(shape_r, 6),
        "so_power":       round(so_power, 6),
        "spindle_power":  round(spindle_power, 6),
        "spindle_gate":   round(spindle_gate, 6),
        "spindle_power_eff": round(spindle_power_eff, 6),
        "so_overshoot": round(so_overshoot, 6),
        "narrow_weight": round(LAMBDA_NARROW, 6),
        "so_overshoot_penalty": round(so_overshoot_penalty, 6),
        "narrow_spindle_penalty": round(narrow_spindle_penalty, 6),
        "dynamics_score": round(dynamics_score, 6),
        # Per-sub-test detail
        "T1_down":        int(dyn_details["T1_down_exists"]),
        "T2_up":          int(dyn_details["T2_up_exists"]),
        "T3_sustained":   int(dyn_details["T3_up_sustained"]),
        "T4_so_freq":     int(dyn_details["T4_so_in_range"]),
        "T5_spindle":     int(dyn_details["T5_spindle_waxing"]),
        "T5_score_cont":  round(dyn_details["T5_spindle_score_cont"], 6),
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

    return -fitness   # return negative fitness because scipy minimizes and we want to maximize

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
    objective_val = -_best_score

    print(
        f"  Gen {_gen:2d}/{N_GEN}  best={_best_score:+.4f}"
        f"  obj={objective_val:+.4f}"
        f"  {status}"
        f"  conv={convergence:.3e}"
        f"  shape_r={bp.get('shape_r', 0):.3f}"
        f"  dyn={bp.get('dynamics_score', 0):.2f}"
        f"  [T1={bp.get('T1_down',0)} T2={bp.get('T2_up',0)}"
        f"   T3={bp.get('T3_sustained',0)} T4={bp.get('T4_so_freq',0)}"
        f"   T5={bp.get('T5_spindle',0)} T5c={bp.get('T5_score_cont',0):.2f}]"
        f"  Fitness: {W_SHAPE:.2f}*shape_r + {W_SO:.2f}*SO + {W_SP:.2f}*spindle_eff + {W_DYN:.2f}*dynamics - so_pen - narrow_pen"
        f"  SO penalty: lambda={SO_OVERSHOOT_LAMBDA:.3f}, target_max={SO_TARGET_MAX:.2f}"
        f" narrow_weight={LAMBDA_NARROW:.3f} narrow_floor={NARROW_PENALTY_FLOOR:.2f}"
        f"  sp_gate={bp.get('spindle_gate',0):.2f}"
        f"  so_over={bp.get('so_overshoot',0.0):.3f}"
        f"  so_pen={bp.get('so_overshoot_penalty',0.0):.3f}"
        f"  narrow_pen={bp.get('narrow_spindle_penalty',0.0):.3f}"
        f"  max_rE={bp.get('max_rE', 0):.1f}Hz"
        f"  evals={_eval_count}"
        f"  elapsed={elapsed:.0f}s"
    )
    return False


def main():
    global _t_start

    print("=" * 65)
    print("V3 Evolution: physics-constrained N3 fitting")
    print("  FOOOF=" + ("ON" if HAS_FOOOF else "OFF (fallback)"))
    print(f"  8 params, bounds constrained for bistable dynamics")
    print(f"  Key: c_ctx2th upper = 0.05 (was 0.30 in v1)")
    print(f"  NEW: T2+T3 UP-state tests (threshold = {UP_THRESH_HZ} Hz, "
            f"duration >= {UP_DURATION_MS} ms)")
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
    # target_psd is the average power spectral density of the target N3 EEG data for the subject,
    #  computed by averaging across all N3 epochs that passed quality control. 
    # target_freqs is the array of frequencies corresponding to the target_psd, selected to include only the frequencies
    # within the F_LO and F_HI range. 
    print("Computing FOOOF on target...")
    target_periodic, fooof_freqs = compute_target_periodic(target_psd, target_freqs)
    # target_periodic is the periodic component of the target PSD extracted by fitting a FOOOF model to the target PSD,
    # fooof_freqs is the array of frequencies corresponding to the FOOOF fit, which should be the same as target_freqs.
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
    ) # result is the output of the differential evolution optimization process, which includes the best parameter vector found, the corresponding fitness value, and other information about the optimization. We pass the compute_fitness_v3 function as the objective function to minimize, along with the bounds for each parameter, the target PSD and frequencies, and other parameters for the DE algorithm such as strategy, population size, number of generations, mutation and recombination rates, random seed for reproducibility, callback function for logging progress, and number of workers for parallel evaluation (set to 1 for simplicity).

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
    print("V3 Validation")
    print(f"{'='*55}")
    print(f"  score         = {best.get('score'):.4f}")
    print(f"  shape_r       = {best.get('shape_r'):.4f}")
    print(f"  so_power      = {best.get('so_power'):.4f}")
    print(f" so_overshoot = {best.get('so_overshoot', 0.0):.4f}")
    print(f" so_penalty = {best.get('so_overshoot_penalty', 0.0):.4f}")
    print(f" narrow_pen = {best.get('narrow_spindle_penalty', 0.0):.4f}")
    print(f"  spindle_power = {best.get('spindle_power'):.4f}")
    print(f"  spindle_gate  = {best.get('spindle_gate', 0.0):.4f}")
    print(f"  spindle_eff   = {best.get('spindle_power_eff', 0.0):.4f}")
    print(f"  dynamics      = {best.get('dynamics_score'):.4f}")
    print(f"")
    print(f"  Sub-tests:")
    print(f"    T1 DOWN state (min r_E < {DOWN_THRESH_HZ} Hz):  "
        f"{'PASS' if best.get('T1_down') else 'FAIL'}")
    print(f"    T2 UP state (max r_E > {UP_THRESH_HZ} Hz):      "
        f"{'PASS' if best.get('T2_up') else 'FAIL'}  "
        f"[max_rE={best.get('max_rE'):.1f} Hz]")
    print(f"    T3 UP sustained (>= {UP_DURATION_MS} ms):        "
        f"{'PASS' if best.get('T3_sustained') else 'FAIL'}  "
        f"[longest_up={best.get('longest_up_ms'):.1f} ms]")
    print(f"    T4 SO freq [{SO_FREQ_LO}-{SO_FREQ_HI} Hz]:       "
        f"{'PASS' if best.get('T4_so_freq') else 'FAIL'}  "
        f"[peak={best.get('so_peak_hz'):.2f} Hz]")
    print(f"    T5 spindle FWHM > {SPINDLE_FWHM_MIN} Hz:         "
        f"{'PASS' if best.get('T5_spindle') else 'FAIL'}  "
        f"[FWHM={best.get('spindle_fwhm'):.2f} Hz]")
    print(f"    T5 continuous score [0-1]:              "
        f"{best.get('T5_score_cont', 0.0):.3f}")
    print(f"")
    print(f"  Best parameters:")
    for k in PARAM_NAMES:
        print(f"    {k}: {best.get(k):.4f}")
    print(f"\nNext: run python plot_scripts/plot_fig7_v3_fast.py")
    print(f"  (update PARAMS_PATH to point to _v3 files)")


if __name__ == "__main__":
    main()
