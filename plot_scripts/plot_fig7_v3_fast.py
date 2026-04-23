"""
plot_fig7_v3_fast.py
====================
Fast visualization for v3 personalization results.

Design goal:
- Keep plotting lightweight (no bifurcation scan)
- Match v3 training logic for target PSD and shape_r as closely as possible

Outputs:
  outputs/fig7_v3_timeseries.png
  outputs/fig7_v3_spectra.png
  outputs/fig7_v3_residuals.png

Run from project root:
  python plot_scripts/plot_fig7_v3_fast.py

Examples:
    python plot_scripts/plot_fig7_v3_fast.py --sim-dur-ms 30000
    python plot_scripts/plot_fig7_v3_fast.py --sim-dur-ms 60000
"""

import os
import sys
import json
import fnmatch
import argparse
import importlib.util
import warnings

warnings.filterwarnings("ignore")

# Ensure working directory is project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if os.getcwd() != _ROOT:
    os.chdir(_ROOT)

# NumPy compatibility patch for neurolib stacks expecting deprecated aliases
import numpy as np
for _attr in ("object", "bool", "int", "float", "complex", "str"):
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(__builtins__, _attr, object))

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
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
    print("[warn] fooof not installed; residual and shape_r recomputation skipped")


# Reuse preprocessing bricks from utils/02_preprocess_psd.py
_spec = importlib.util.spec_from_file_location(
    "preprocess_psd_module", "utils/02_preprocess_psd.py"
)
prep_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep_mod)
load_hypnogram = prep_mod.load_hypnogram # import the load_hypnogram function from the preprocessing module
compute_epoch_psd = prep_mod.compute_epoch_psd # import the compute_epoch_psd function from the preprocessing module
EPOCH_LEN_S = prep_mod.EPOCH_LEN_S # epoch length in seconds used for PSD computation in the preprocessing module


# Config aligned with models/s4_personalize_fig7_v3.py
SUBJECT_ID = "SC4001"
EEG_CHANNELS = ["EEG Fpz-Cz"]
ARTIFACT_THRESH = 200e-6
F_LO, F_HI = 0.5, 20.0
FS_SIM = 1000.0

DEFAULT_PARAMS_PATH = f"data/patient_params_fig7_v3_{SUBJECT_ID}.json"

# Use 30s by default to mirror fitness evaluation conditions.
SIM_DUR_MS = 30_000

# Display window for time series
T_SHOW_START = 8.0
T_SHOW_END = 24.0

EVO_FOOOF_PARAMS = dict(
    peak_width_limits=[1.0, 8.0],
    max_n_peaks=4,
    min_peak_height=0.05,
    aperiodic_mode="fixed",
) # example FOOOF settings for the evo pipeline;


def _fmt_json_num(bp, key, default="?", fmt=".4f"):
    v = bp.get(key)
    if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
        return format(v, fmt)
    return default


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast v3 plotting with selectable simulation duration."
    )
    parser.add_argument(
        "--params-path",
        type=str,
        default=DEFAULT_PARAMS_PATH,
        help="Path to params JSON file.",
    )
    parser.add_argument(
        "--sim-dur-ms",
        type=int,
        choices=[30_000, 60_000],
        default=SIM_DUR_MS,
        help="Simulation duration in ms: 30000 (fitness-aligned) or 60000 (longer time-series).",
    ) # allow choice of 30s (to match fitness evaluation) or 60s (for a longer time series visualization), with 30s as the default
    parser.add_argument(
        "--out-timeseries",
        type=str,
        default="outputs/fig7_v3_timeseries.png",
        help="Output path for timeseries figure.",
    )
    parser.add_argument(
        "--out-spectra",
        type=str,
        default="outputs/fig7_v3_spectra.png",
        help="Output path for spectra figure.",
    )
    parser.add_argument(
        "--out-residuals",
        type=str,
        default="outputs/fig7_v3_residuals.png",
        help="Output path for residuals figure.",
    )
    return parser.parse_args()


def set_params_glob(model, pattern, value):
    for k in (k for k in model.params if fnmatch.fnmatch(k, pattern)):
        model.params[k] = value


class ThalamoCorticalNetwork(Network):
    name = "Thalamocortical Motif" # Thalamo-cortical motif with ALN cortex and thalamic relay population, matching the structure used in v3 personalization. The network includes additive coupling terms for excitatory-to-excitatory interactions within the cortex, as well as thalamocortical and corticothalamic interactions, to capture the dynamics relevant for N3 sleep.
    label = "TCNet"
    sync_variables = ["network_exc_exc", "network_exc_exc_sq", "network_inh_exc"] # Additive coupling variables for excitatory-to-excitatory interactions (linear and quadratic) and inhibitory-to-excitatory interactions within the cortex, which are used to implement the additional coupling terms in the _sync method.
    default_output = f"r_mean_{EXC}" # Default output variable for the network, set to the mean firing rate of the excitatory population, which is the primary variable of interest for comparing to EEG data and computing the fitness score based on spectral shape similarity.
    output_vars = [f"r_mean_{EXC}", f"r_mean_{INH}"] # Output variables for the network, including the mean firing rates of both the excitatory and inhibitory populations, which can be used for analysis and visualization of the network dynamics.
    _EXC_WITHIN_IDX = [6, 9]

    def __init__(self, c_th2ctx=0.02, c_ctx2th=0.02):
        aln = ALNNode(exc_seed=42, inh_seed=42) # ALN cortex node with fixed random seeds for reproducibility, which ensures that the intrinsic dynamics of the cortical node are consistent across runs and can be directly compared to the target EEG data. The ALN model captures the local cortical dynamics relevant for generating N3-like activity patterns, while the thalamic node captures the relay dynamics that interact with the cortex to produce the full thalamocortical motif.
        th = ThalamicNode() # Thalamic relay node
        aln.index = 0
        aln.idx_state_var = 0
        th.index = 1
        th.idx_state_var = aln.num_state_variables
        for i, node in enumerate([aln, th]):
            for mass in node:
                mass.noise_input_idx = [2 * i + mass.index]
        connectivity = np.array([[0.0, c_th2ctx], [c_ctx2th, 0.0]]) # define the connectivity matrix for the thalamocortical network, where c_th2ctx represents the strength of thalamocortical projections from the thalamic relay node to the cortical node, and c_ctx2th represents the strength of corticothalamic projections from the cortical node back to the thalamic relay node. These parameters are critical for shaping the interactions between the cortex and thalamus that give rise to N3-like dynamics, and they are included in the personalization process to allow tuning of these interactions based on the target EEG data.
        super().__init__(
            nodes=[aln, th],
            connectivity_matrix=connectivity,
            delay_matrix=np.zeros_like(connectivity),
        )

    def _sync(self):
        couplings = sum([node._sync() for node in self], []) # couplings from the individual nodes, which include the standard synaptic interactions defined in the ALN and thalamic node models. These couplings capture the baseline dynamics of the network, while the additional additive coupling terms implemented below specifically target the excitatory-to-excitatory interactions within the cortex, which are important for shaping the spectral characteristics of the cortical activity and improving the fit to the target EEG data.
        wi = self._EXC_WITHIN_IDX
        couplings += self._additive_coupling(wi, "network_exc_exc") # Additive coupling for excitatory-to-excitatory interactions within the cortex, which allows for additional modulation of the excitatory population's activity based on its own firing rate. This term can capture nonlinear effects and enhance the model's ability to reproduce the spectral shape of N3 sleep by providing a mechanism for self-excitation that can influence the generation of slow oscillations and spindles.
        couplings += self._additive_coupling(
            wi, "network_exc_exc_sq", connectivity=self.connectivity ** 2
        ) # Additive coupling for the squared excitatory-to-excitatory interactions within the cortex, which introduces a nonlinear component to the self-excitation mechanism. By using the squared connectivity, this term can capture more complex dynamics and interactions that may be present in the cortical circuitry, further enhancing the model's flexibility in fitting the target EEG data and reproducing the characteristic spectral features of N3 sleep.
        couplings += self._additive_coupling(wi, "network_inh_exc") # Additive coupling for inhibitory-to-excitatory interactions within the cortex
        return couplings


def build_model(bp, duration=SIM_DUR_MS):
    net = ThalamoCorticalNetwork(c_th2ctx=bp["c_th2ctx"], c_ctx2th=bp["c_ctx2th"])
    m = MultiModel(net)
    m.params["backend"] = "numba"
    m.params["dt"] = 0.1
    m.params["sampling_dt"] = 1.0
    m.params["duration"] = duration

    set_params_glob(m, "*ALNMassEXC*.input_0.mu", bp["mue"])
    set_params_glob(m, "*ALNMassINH*.input_0.mu", bp["mui"])
    set_params_glob(m, "*ALNMassEXC*.b", bp["b"])
    set_params_glob(m, "*ALNMassEXC*.tauA", bp["tauA"])
    set_params_glob(m, "*ALNMassEXC*.a", 0.0)
    set_params_glob(m, "*ALNMass*.input_0.sigma", 0.05)
    set_params_glob(m, "*TCR*.input_0.sigma", 0.005)
    set_params_glob(m, "*.input_0.tau", 5.0)
    set_params_glob(m, "*TRN*.g_LK", 0.1)
    set_params_glob(m, "*TCR*.g_LK", bp["g_LK"])
    set_params_glob(m, "*TCR*.g_h", bp["g_h"])
    return m


def load_target_psd_v3_aligned():
    """Match the target PSD pipeline used in v3 personalization."""
    try:
        manifest = pd.read_csv("data/manifest.csv", encoding="utf-8")
    except UnicodeDecodeError:
        manifest = pd.read_csv("data/manifest.csv", encoding="utf-16")

    subj_row = manifest[manifest["subject_id"] == SUBJECT_ID].iloc[0]
    raw = mne.io.read_raw_edf(
        subj_row["psg_path"], include=EEG_CHANNELS, preload=True, verbose=False
    )
    fs_eeg = raw.info["sfreq"]

    stages = load_hypnogram(subj_row["hypnogram_path"]) # load the hypnogram for the subject using the load_hypnogram function from the preprocessing module, which provides the sleep stage annotations needed to identify N3 epochs for PSD computation. This ensures that the target PSD is computed specifically from N3 sleep periods, which is critical for accurately capturing the spectral characteristics that the model is being personalized to reproduce. 
    data_uv = raw.get_data()[0] * 1e6 # convert from V to uV; the PSD computation and artifact rejection thresholds are based on microvolt units, so this conversion is necessary to ensure that the target PSD is computed correctly and that the same criteria for epoch selection and artifact rejection are applied as in the v3 personalization process.

    n_samples_per_epoch = int(EPOCH_LEN_S * fs_eeg) # number of samples in each epoch based on the defined epoch length and EEG sampling frequency, which is used to segment the continuous EEG data into epochs for PSD computation. This segmentation allows for the analysis of the spectral characteristics of the EEG during specific sleep stages (e.g., N3) and ensures that the PSD is computed in a manner consistent with the v3 personalization process, where epochs are evaluated for quality and only those meeting the criteria are included in the average PSD that serves as the target for model fitting.
    n_epochs = min(len(stages), len(data_uv) // n_samples_per_epoch)

    psds = []
    f_ep = None
    freq_mask = None

    for i in range(n_epochs):
        if stages[i] != "N3":
            continue

        epoch_data = data_uv[i * n_samples_per_epoch : (i + 1) * n_samples_per_epoch]
        # epoch_data is the segment of the EEG data corresponding to the current epoch, which is extracted based on the epoch index and the number of samples per epoch. This data is then evaluated for artifacts by checking the peak-to-peak amplitude against a predefined threshold, ensuring that only clean N3 epochs are included in the PSD computation. By applying this artifact rejection step, we ensure that the target PSD is representative of genuine N3 sleep activity and is not contaminated by noise or artifacts that could skew the model fitting process.
        if np.ptp(epoch_data) > ARTIFACT_THRESH * 1e6:
            continue

        f_ep, p_ep = compute_epoch_psd(epoch_data, fs_eeg)
        # f_ep and p_ep are the frequency bins and corresponding power spectral density values computed for the current epoch using the compute_epoch_psd function from the preprocessing module. This function applies the same PSD computation method used in the v3 personalization process, ensuring that the target PSD is computed in a consistent manner. The resulting PSD is then evaluated against the defined frequency range (F_LO to F_HI) to select the relevant portion of the spectrum for comparison with the model's simulated PSD, which is critical for accurately assessing the shape similarity and guiding the personalization process.
        if freq_mask is None:
            freq_mask = (f_ep >= F_LO) & (f_ep <= F_HI)
        psds.append(p_ep[freq_mask])

    if not psds:
        raise RuntimeError("No N3 epochs passed QC for subject " + SUBJECT_ID)

    print(f"  Target EEG: {len(psds)} N3 epochs used") # print the number of N3 epochs that passed the artifact rejection criteria and were included in the average PSD computation, which provides insight into the quality and quantity of data contributing to the target PSD. This information is important for interpreting the results of the model fitting, as a larger number of clean N3 epochs can lead to a more reliable target PSD and potentially a better fit, while a smaller number may indicate that the target PSD is based on limited data and should be interpreted with caution.
    return np.mean(psds, axis=0), f_ep[freq_mask]


def compute_target_periodic(target_psd, target_freqs):
# target_psd, target_freqs are the average power spectral density and corresponding frequency bins computed from the N3 epochs of the target EEG data, which serve as the reference for fitting the model's simulated PSD. This function applies FOOOF to decompose the target PSD into its aperiodic and periodic components, allowing for a more accurate assessment of the shape similarity between the model and the target by focusing on the periodic features (e.g., peaks corresponding to slow oscillations and spindles) that are critical for characterizing N3 sleep. The resulting target_periodic component is then used in the computation of the shape similarity score (shape_r) during model evaluation, ensuring that the personalization process is guided by the relevant spectral features of N3 sleep.
    if not HAS_FOOOF:
        return None, None, None

    fm = FOOOF(**EVO_FOOOF_PARAMS) # fm is an instance of the FOOOF class from the FOOOF library, initialized with specific parameters defined in EVO_FOOOF_PARAMS. These parameters configure the FOOOF algorithm to fit the target PSD by identifying peaks within a specified width range, limiting the maximum number of peaks, setting a minimum peak height, and using a fixed aperiodic mode. By applying FOOOF to the target PSD, we can extract the periodic components that correspond to the characteristic oscillatory activity of N3 sleep, which are essential for accurately assessing the shape similarity between the model and the target EEG data during personalization.
    fm.fit(target_freqs, target_psd, [F_LO, F_HI])
    # fm.fit applies the FOOOF fitting procedure to the target PSD using the specified frequency bins and fitting range. This process decomposes the target PSD into its aperiodic component (captured by fm._ap_fit) and its periodic components (captured by fm.peak_params_), allowing us to isolate the periodic features that are critical for characterizing N3 sleep. The resulting fit provides insights into the spectral characteristics of the target EEG data, including the presence and properties of peaks corresponding to slow oscillations and spindles, which are essential for guiding the personalization of the model to reproduce these features.
    target_log = np.log10(target_psd + 1e-30) # target_log is the logarithm of the target PSD, which is computed to facilitate the separation of the aperiodic and periodic components during the FOOOF fitting process. By taking the log of the PSD, we can more effectively model the aperiodic component as a linear function in log-log space, while the periodic peaks are represented as deviations from this aperiodic fit. The resulting target_log is then used to compute the target_periodic component by subtracting the aperiodic fit (fm._ap_fit) from the log-transformed PSD, isolating the periodic features that are critical for assessing shape similarity and guiding model personalization.
    target_periodic = target_log[: len(fm._ap_fit)] - fm._ap_fit
    # target_periodic is the periodic component of the target PSD, computed by subtracting the aperiodic fit (fm._ap_fit) from the log-transformed PSD
    print("  FOOOF target peaks:")
    for pk in fm.peak_params_:
        print(f"    {pk[0]:.1f} Hz  power={pk[1]:.3f}  width={pk[2]:.1f}")
    print(f"  Aperiodic exponent: {fm.aperiodic_params_[1]:.2f}")
    # The above print statements provide a summary of the FOOOF fitting results for the target PSD, including the identified peaks (with their frequencies, powers, and widths) and the aperiodic exponent. This information is valuable for understanding the spectral characteristics of the target EEG data, particularly the presence and properties of oscillatory features that are relevant for N3 sleep. By examining these FOOOF results, we can gain insights into the specific spectral features that the model will be personalized to reproduce, and it can also help in interpreting the shape similarity scores and the overall fit of the model to the target data.

    return target_periodic, fm.freqs, fm._ap_fit


def recompute_shape_and_peaks_v3_order(f_ctx, p_ctx, fooof_freqs, target_periodic):
    """Replicate v3 fitness order: interpolate power first, then FOOOF on common grid."""
    if not HAS_FOOOF or target_periodic is None:
        return None, None, None, None, None, None, None

    p_interp = interp1d(f_ctx, p_ctx, bounds_error=False, fill_value=1e-30)(fooof_freqs)
    # p_interp is the interpolated power spectral density of the model's simulated cortical activity, 
    # computed by applying a 1D interpolation function to the original frequency bins (f_ctx) and
    # corresponding PSD values (p_ctx) to evaluate the PSD at the frequency bins used by FOOOF (fooof_freqs). This interpolation step ensures that the model's PSD is evaluated on the same frequency grid as the target PSD used in the FOOOF fitting, allowing for a direct comparison of the periodic components and an accurate computation of the shape similarity score (shape_r) based on the periodic features extracted by FOOOF. By aligning the frequency grids in this way, we can ensure that the assessment of shape similarity and peak properties is consistent with the v3 personalization process, where both the model and target PSDs are analyzed using FOOOF on a common set of frequencies.
    fm = FOOOF(**EVO_FOOOF_PARAMS)
    # fm is a new instance of the FOOOF class, initialized with the same parameters as used for the target PSD fitting. This instance will be used to fit the interpolated model PSD (p_interp) on the same frequency grid (fooof_freqs) to extract the periodic components and aperiodic fit for the model's simulated data. By applying FOOOF to the model's PSD in this way, we can directly compare the periodic features of the model to those of the target, compute the shape similarity score based on the periodic components, and assess how well the model reproduces the characteristic spectral features of N3 sleep as captured in the target EEG data.
    fm.fit(fooof_freqs, p_interp, [F_LO, F_HI])
    # fm.fit applies the FOOOF fitting procedure to the interpolated model PSD (p_interp) using the specified frequency bins
    # (fooof_freqs) and fitting range. This process decomposes the model's PSD into its aperiodic component and periodic components, allowing us to extract the features that are relevant for characterizing N3 sleep and comparing them to the target. The resulting fit provides insights into how well the model's simulated activity captures the spectral characteristics of N3 sleep, including the presence and properties of peaks corresponding to slow oscillations and spindles, which are critical for guiding the personalization process and interpreting the shape similarity scores.
    sim_log = np.log10(p_interp[: len(fm._ap_fit)] + 1e-30)
    # sim_log is the logarithm of the interpolated model PSD (p_interp)
    sim_periodic = sim_log - fm._ap_fit
    # sim_periodic is the periodic component of the model's PSD
    n_r = min(len(sim_periodic), len(target_periodic))
    # n_r is the number of frequency bins to consider when computing the shape similarity score (shape_r), which is determined by taking the minimum length of the model's periodic component (sim_periodic) and the target's periodic component (target_periodic). This ensures that the Pearson correlation used to compute shape_r is based on a common set of frequency bins where both the model and target periodic components are defined, allowing for an accurate assessment of shape similarity that focuses on the relevant spectral features of N3 sleep.
    shape_r_raw, _ = pearsonr(sim_periodic[:n_r], target_periodic[:n_r])
    # shape_r_raw is the raw shape similarity score computed as the Pearson correlation coefficient between the model's
    # periodic component (sim_periodic) and the target's periodic component (target_periodic) 
    # over the common set of frequency bins defined by n_r. This score quantifies how well the model's simulated PSD captures
    # the shape of the target PSD in terms of its periodic features, which are critical for characterizing N3 sleep. A higher shape_r_raw indicates a better match between the model and target spectral shapes, while a lower score indicates a poorer fit. This raw score is then clipped to ensure it falls within a valid range for further analysis and interpretation.
    shape_r_clipped = max(float(shape_r_raw), 0.0)
    # shape_r_clipped is the shape similarity score after applying a clipping operation to ensure that it is non-negative. This is done because negative shape similarity scores may not be meaningful in the context of assessing the fit between the model and target PSDs, and clipping at 0.0 allows us to focus on cases where there is at least some degree of positive similarity in the spectral shapes. By using shape_r_clipped, we can provide a more interpretable measure of how well the model captures the relevant spectral features of N3 sleep as represented in the target EEG data.
    so_power = 0.0
    spindle_power = 0.0
    for pk in fm.peak_params_:
        freq, power, _ = pk
        # Keep the strongest FOOOF peak inside the SO band (0.2-1.5 Hz).
        # Using max(...) avoids double-counting when multiple peaks fall in-band.
        if 0.2 <= freq <= 1.5: 
            so_power = max(so_power, float(power))
        if 10.0 <= freq <= 14.0:
            spindle_power = max(spindle_power, float(power))

    return sim_periodic, fm._ap_fit, p_interp, shape_r_raw, shape_r_clipped, so_power, spindle_power


def main():
    args = parse_args()
    sim_dur_ms = args.sim_dur_ms

    if not os.path.isfile(args.params_path):
        print(f"[error] Missing {args.params_path}; run models/s4_personalize_fig7_v3.py first")
        sys.exit(1)

    with open(args.params_path, "r", encoding="utf-8") as fh:
        bp = json.load(fh)

    print("=" * 65)
    print("V3 fast visualization")
    print(f"  Subject: {SUBJECT_ID}")
    print(f"  Params:  {args.params_path}")
    print(f"  SimDur:  {sim_dur_ms/1000:.0f} s")
    print(f"  FOOOF:   {'ON' if HAS_FOOOF else 'OFF'}")
    print("=" * 65)

    print("\nRunning simulation...")
    model = build_model(bp, duration=sim_dur_ms)
    # Run the model using the numba backend for speed, but fall back to jitcdde if there are issues (e.g., due to numba version or environment). This allows for a more robust execution of the plotting script across different setups while still prioritizing performance when possible.
    try:
        model.run()
    except Exception as e:
        print(f"  numba backend failed ({e}); switching to jitcdde")
        model.params["backend"] = "jitcdde"
        model.run()

    n_total = int(sim_dur_ms / model.params["sampling_dt"])
    t_s = np.linspace(0, sim_dur_ms / 1000.0, n_total, endpoint=False)

    r_exc_raw = model[f"r_mean_{EXC}"]
    if r_exc_raw.ndim == 2 and r_exc_raw.shape[0] >= 2:
        rE_cortex = r_exc_raw[0, :] * 1000.0
        # rE_cortex and rE_thalamus are the firing rates of the cortical excitatory population and the thalamic relay population, respectively, extracted from the model's output. The code checks if r_exc_raw is a 2D array with at least two rows, which would indicate that it contains separate traces for the cortex and thalamus. If so, it assigns the first row to rE_cortex and the second row to rE_thalamus, converting from kHz to Hz by multiplying by 1000.0. If r_exc_raw does not have this structure, it assumes that it only contains cortical activity and sets rE_thalamus to zeros, while printing a warning message. This allows for flexibility in handling different output formats from the model while ensuring that the time series for both cortex and thalamus are available for plotting.
        rE_thalamus = r_exc_raw[1, :] * 1000.0
        # rE_thalamus is the firing rate of the thalamic relay population, extracted from the second row of r_exc_raw if it is a 2D array with at least two rows. This assumes that the model's output includes separate traces for the cortical excitatory population and the thalamic relay population, which is consistent with the structure of the ThalamoCorticalNetwork defined in this script. By extracting these traces, we can visualize the dynamics of both populations and compare them to the target EEG data, as well as compute the PSD and shape similarity scores based on the cortical activity.
    else:
        rE_cortex = (r_exc_raw[0] if r_exc_raw.ndim == 2 else r_exc_raw) * 1000.0
        rE_thalamus = np.zeros_like(rE_cortex)
        print("  [warn] r_mean_EXC is not 2-node; thalamus trace set to zeros")

    n_min = min(len(t_s), len(rE_cortex), len(rE_thalamus))
    t_s = t_s[:n_min]
    rE_cortex = rE_cortex[:n_min]
    rE_thalamus = rE_thalamus[:n_min]

    n_burn = int(5.0 * FS_SIM)
    r_ctx = rE_cortex[n_burn:]
    # r_ctx is from model simulation, representing the firing rate of the cortical excitatory population after discarding an initial burn-in period (n_burn) to allow the model dynamics to stabilize. This ensures that the PSD computation focuses on the steady-state activity of the model rather than transient dynamics that may occur at the beginning of the simulation. By using r_ctx for PSD computation, we can obtain a more accurate representation of the model's spectral characteristics during N3-like activity, which is critical for comparing to the target EEG data and assessing shape similarity.

    f_ctx_full, p_ctx_full = compute_epoch_psd(r_ctx, FS_SIM)
    # f_ctx_full and p_ctx_full are the frequency bins and corresponding power spectral density values 
    # computed from the simulated cortical firing rate.
    mask_ctx = (f_ctx_full >= F_LO) & (f_ctx_full <= F_HI)
    # mask_ctx is a boolean array that selects the frequency bins within the defined range (F_LO to F_HI) for the model's PSD, ensuring that the subsequent analysis and comparison to the target PSD focuses on the relevant portion of the spectrum that characterizes N3 sleep. By applying this mask, we can isolate the frequencies of interest and compute shape similarity scores and other metrics based on the spectral features that are most relevant for assessing the fit of the model to the target EEG data.
    f_ctx = f_ctx_full[mask_ctx]
    p_ctx = p_ctx_full[mask_ctx]

    print("\nLoading target EEG...")
    target_psd, target_freqs = load_target_psd_v3_aligned()
    target_periodic, fooof_freqs, tgt_aperiodic = compute_target_periodic(
        target_psd, target_freqs
    ) # load the target EEG data, compute the average PSD from N3 epochs, and decompose it into periodic and aperiodic components using FOOOF. This provides the reference spectral characteristics of N3 sleep that the model will be personalized to reproduce, and it allows for a more accurate assessment of shape similarity by focusing on the periodic features that are critical for characterizing N3 sleep. The resulting target_periodic component will be used in the computation of shape similarity scores during model evaluation, ensuring that the personalization process is guided by the relevant spectral features of the target EEG data.

    sim_periodic = None
    sim_aperiodic = None
    sim_interp_for_fooof = None
    shape_r_raw = None
    shape_r_clipped = None
    so_power_recomp = None
    spindle_power_recomp = None

    if HAS_FOOOF and target_periodic is not None:
        (
            sim_periodic,
            sim_aperiodic,
            sim_interp_for_fooof,
            shape_r_raw,
            shape_r_clipped,
            so_power_recomp,
            spindle_power_recomp,
        ) = recompute_shape_and_peaks_v3_order(f_ctx, p_ctx, fooof_freqs, target_periodic)
    # recompute_shape_and_peaks_v3_order is called to interpolate the model's PSD onto the FOOOF frequency grid,
    # apply FOOOF to extract the periodic and aperiodic components of the model's PSD, 
    # and compute the shape similarity scores (both raw and clipped) as well as the power of peaks in the slow oscillation and spindle bands. This step allows for a more accurate assessment of how well the model's simulated activity captures the spectral features of N3 sleep as represented in the target EEG data, and it provides insights into the specific characteristics of the model's fit to the target, including how closely it matches the periodic features that are critical for characterizing N3 sleep.
    os.makedirs("outputs", exist_ok=True)

    # Figure 7(c): time series
    print("\nPlotting time series...")
    mask_t = (t_s >= T_SHOW_START) & (t_s <= T_SHOW_END)

    fig_c, (ax_c1, ax_c2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig_c.suptitle(
        "Fig. 7(c) V3 - thalamocortical time series\n"
        f"score={_fmt_json_num(bp, 'score')}, shape_r={_fmt_json_num(bp, 'shape_r')}, "
        f"dynamics={_fmt_json_num(bp, 'dynamics_score', fmt='.2f')}",
        fontsize=12,
        fontweight="bold",
    )

    ax_c1.plot(t_s[mask_t], rE_cortex[mask_t], color="#534AB7", lw=0.8)
    # t_s[mask_t] and rE_cortex[mask_t] are the time points and corresponding cortical firing rates for the selected time
    #  window defined by T_SHOW_START and T_SHOW_END. This plot visualizes the dynamics of the cortical excitatory population
    #  during the specified time window, allowing us to assess the presence of N3-like activity patterns such as slow oscillations and spindles. By examining this time series, we can gain insights into how well the model captures the temporal dynamics of N3 sleep, which is an important aspect of the overall fit to the target EEG data.
    ax_c1.axhline(1.0, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax_c1.axhline(15.0, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax_c1.set_ylabel("r_E [Hz]")
    ax_c1.set_title("Cortex EXC")
    ax_c1.set_ylim(bottom=-0.1)

    ax_c2.plot(t_s[mask_t], rE_thalamus[mask_t], color="#1D9E75", lw=0.8)
    # t_s[mask_t] and rE_thalamus[mask_t] are the time points and corresponding thalamic firing rates for the selected time window defined by T_SHOW_START and T_SHOW_END. This plot visualizes the dynamics of the thalamic relay population during the specified time window, allowing us to assess how the thalamus interacts with the cortex to produce N3-like activity patterns. By examining this time series alongside the cortical activity, we can gain insights into the thalamocortical interactions that are critical for generating the characteristic spectral features of N3 sleep and evaluate how well the model captures these dynamics in comparison to the target EEG data.
    ax_c2.set_ylabel("r_TCR [Hz]")
    ax_c2.set_xlabel("Time [s]")
    ax_c2.set_title("Thalamic TCR")

    param_txt = (
        f"mue={bp['mue']:.3f}  mui={bp['mui']:.3f}  b={bp['b']:.1f}  tauA={bp['tauA']:.0f}\n"
        f"g_LK={bp['g_LK']:.4f}  g_h={bp['g_h']:.4f}  "
        f"c_th2ctx={bp['c_th2ctx']:.4f}  c_ctx2th={bp['c_ctx2th']:.4f}"
    )
    fig_c.text(0.5, 0.01, param_txt, ha="center", fontsize=8, color="gray", family="monospace")

    fig_c.tight_layout(rect=[0, 0.05, 1, 1])
    out_c = args.out_timeseries
    fig_c.savefig(out_c, dpi=150, bbox_inches="tight")
    plt.close(fig_c)
    print(f"  Saved: {out_c}")

    # Figure 7(d): spectra
    print("Plotting spectra...")
    fig_d, (ax_d1, ax_d2) = plt.subplots(2, 1, figsize=(8, 8))
    fig_d.suptitle(
        f"Fig. 7(d) V3 - power spectra\n{SUBJECT_ID} N3 EEG vs simulated cortex firing rate",
        fontsize=12,
        fontweight="bold",
    )

    ax_d1.semilogy(target_freqs, target_psd, "k", lw=1.8, label="Target EEG N3")
    # target_freqs and target_psd are the frequency bins and corresponding power spectral density values computed from 
    # the N3 epochs of the target EEG data, which are plotted on a semilogarithmic scale to visualize the 
    # spectral characteristics of the target EEG during N3 sleep. This plot serves as a reference for comparing the model's simulated PSD to the target, allowing us to visually assess how well the model captures the relevant spectral features such as slow oscillations and spindles that are characteristic of N3 sleep.
    if tgt_aperiodic is not None and fooof_freqs is not None:
        ax_d1.semilogy(fooof_freqs, 10 ** tgt_aperiodic, "b--", lw=1.2, alpha=0.7, label="1/f fit")
    ax_d1.axvspan(0.2, 1.5, alpha=0.10, color="orange", label="SO band")
    ax_d1.axvspan(10.0, 14.0, alpha=0.10, color="green", label="Spindle band")
    ax_d1.set_xlim(F_LO, F_HI)
    ax_d1.set_ylabel("Power")
    ax_d1.set_title("EEG target", loc="right")
    ax_d1.legend(fontsize=8)

    ax_d2.semilogy(f_ctx, p_ctx, color="#534AB7", lw=1.5, label="Simulated cortex PSD")
    # f_ctx and p_ctx are the frequency bins and corresponding power spectral density values computed from the simulated cortical firing rate, which are plotted on a semilogarithmic scale to visualize the spectral characteristics of the model's simulated activity. This plot allows us to visually compare the model's PSD to the target EEG PSD, assessing how well the model captures the relevant spectral features of N3 sleep such as slow oscillations and spindles. By examining this plot alongside the target EEG PSD, we can gain insights into the quality of the fit and the specific spectral features that are being reproduced by the model.
    if sim_aperiodic is not None and fooof_freqs is not None:
        ax_d2.semilogy(fooof_freqs, 10 ** sim_aperiodic, "b--", lw=1.0, alpha=0.7, label="1/f fit")
    ax_d2.axvspan(0.2, 1.5, alpha=0.10, color="orange", label="SO band")
    ax_d2.axvspan(10.0, 14.0, alpha=0.10, color="green", label="Spindle band")
    ax_d2.set_xlim(F_LO, F_HI)
    ax_d2.set_xlabel("Frequency [Hz]")
    ax_d2.set_ylabel("Power")
    ax_d2.set_title("Simulation (v3)", loc="right")
    ax_d2.legend(fontsize=8)

    score_txt = (
        f"shape_r JSON={_fmt_json_num(bp, 'shape_r', fmt='.3f')}  "
        f"shape_r recomputed={shape_r_clipped:.3f}" if shape_r_clipped is not None else "shape_r recomputation unavailable"
    )
    ax_d2.text(0.98, 0.04, score_txt, transform=ax_d2.transAxes, ha="right", fontsize=8, color="gray")

    fig_d.tight_layout()
    out_d = args.out_spectra
    fig_d.savefig(out_d, dpi=150, bbox_inches="tight")
    plt.close(fig_d)
    print(f"  Saved: {out_d}")

    # Residuals figure
    out_r = args.out_residuals
    residual_saved = False
    if HAS_FOOOF and target_periodic is not None and sim_periodic is not None and fooof_freqs is not None:
        print("Plotting residual comparison...")
        n_r = min(len(sim_periodic), len(target_periodic), len(fooof_freqs))
        ff = fooof_freqs[:n_r] # ff is the frequency bins of periodic components.
        tp = target_periodic[:n_r] # tp is the periodic component of the target PSD.
        sp = sim_periodic[:n_r] # sp is the periodic component of the simulated PSD.

        fig_r, ax_r = plt.subplots(figsize=(10, 5))
        ax_r.plot(ff, tp, "k-", lw=2.0, label="EEG target residual")
        # plot the peridoic component of the target psd against frequency bins of periodic components
        ax_r.plot(ff, sp, color="#534AB7", lw=2.0, ls="--", label="Simulation residual")
        # plot the periodic component of the simulated psd against frequency bins of periodic components
        ax_r.axhline(0.0, color="gray", lw=0.5, alpha=0.5)
        ax_r.axvspan(0.2, 1.5, alpha=0.10, color="orange", label="SO band")
        ax_r.axvspan(10.0, 14.0, alpha=0.10, color="green", label="Spindle band")

        title = (
            f"FOOOF residuals (v3 order) | r_raw={shape_r_raw:.4f}, "
            f"r_clipped={shape_r_clipped:.4f}, JSON={_fmt_json_num(bp, 'shape_r', fmt='.4f')}"
            if shape_r_raw is not None and shape_r_clipped is not None
            else "FOOOF residuals (v3 order)"
        )
        ax_r.set_title(title, fontsize=11)
        ax_r.set_xlabel("Frequency [Hz]")
        ax_r.set_ylabel("Log residual")
        ax_r.set_xlim(F_LO, F_HI)
        ax_r.legend(loc="upper right", fontsize=9)

        fig_r.tight_layout()
        fig_r.savefig(out_r, dpi=150, bbox_inches="tight")
        plt.close(fig_r)
        residual_saved = True
        print(f"  Saved: {out_r}")
    else:
        print("  Skipping residual figure (fooof missing or fit unavailable)")

    print("\n" + "=" * 65)
    print("V3 fast visualization summary")
    print("=" * 65)
    print(f"  score (JSON)              = {bp.get('score', 'N/A')}")
    print(f"  shape_r (JSON)            = {bp.get('shape_r', 'N/A')}")
    if shape_r_raw is not None and shape_r_clipped is not None:
        print(f"  shape_r recomputed raw    = {shape_r_raw:.4f}")
        print(f"  shape_r recomputed clip   = {shape_r_clipped:.4f}")
    if so_power_recomp is not None and spindle_power_recomp is not None:
        print(f"  so_power recomputed       = {so_power_recomp:.4f}")
        print(f"  spindle_power recomputed  = {spindle_power_recomp:.4f}")
    print(f"  dynamics_score (JSON)     = {bp.get('dynamics_score', 'N/A')}")
    print(f"  cortex min r_E post-burn  = {r_ctx.min():.3f} Hz")
    print(f"  cortex mean r_E post-burn = {r_ctx.mean():.3f} Hz")
    print("\nOutput files:")
    print(f"  {out_c}")
    print(f"  {out_d}")
    if residual_saved:
        print(f"  {out_r}")
    else:
        print(f"  {out_r} (not generated)")


if __name__ == "__main__":
    main()