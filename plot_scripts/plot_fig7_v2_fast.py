"""
plot_fig7_v2_fast.py
====================
Fast variant: only Fig. 7 (c)(d) + residual panel; skips the slow 7(b) bifurcation scan.

Outputs (3 figures):
  outputs/fig7_v2_timeseries.png     <- 7(c): cortex + thalamus time series
  outputs/fig7_v2_spectra.png        <- 7(d): EEG vs simulated PSD (semilogy)
  outputs/fig7_v2_residuals.png      <- FOOOF 1/f-removed residual comparison

Run from project root:
  python plot_scripts/plot_fig7_v2_fast.py

Note: simulation uses 60 s (longer than the 30 s fitness eval in s4_personalize_fig7_v2.py),
so recomputed shape_r may differ slightly from the JSON stored after evolution.

Requires: neurolib, mne, fooof, scipy, pandas, numpy, matplotlib
"""

import os
import sys
import json
import fnmatch

# Ensure working directory is project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if os.getcwd() != _ROOT:
    os.chdir(_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
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
    print("[warn] fooof not installed; residual figure will be skipped")

# --- Config (aligned with s4_personalize_fig7_v2.py) -------------------------
SUBJECT_ID      = "SC4001"
EEG_CHANNELS    = ["EEG Fpz-Cz", "EEG Pz-Oz"]
N3_LABELS       = ["Sleep stage 3", "Sleep stage 4"]
ARTIFACT_THRESH = 200e-6
EPOCH_DURATION  = 30.0
F_LO, F_HI     = 0.5, 20.0
FS_SIM          = 1000.0

# V2 params file
PARAMS_PATH  = f"data/patient_params_fig7_v2_{SUBJECT_ID}.json"

# Simulation length: 60 s, enough to see UP/DOWN; much faster than 7(b) scan
SIM_DUR_MS   = 60_000

# Time-series display window (seconds)
T_SHOW_START = 16.0
T_SHOW_END   = 32.0   # slightly longer than typical paper window to show more SO cycles


# --- Helpers ----------------------------------------------------------------

def _fmt_json_num(bp, key, default="?", fmt=".4f"):
    """Format numeric JSON field for labels; avoid :.3f on missing keys."""
    v = bp.get(key)
    if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
        return format(v, fmt)
    return default


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

    def __init__(self, c_th2ctx=0.02, c_ctx2th=0.15):
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


def build_model(bp, duration=SIM_DUR_MS):
    net = ThalamoCorticalNetwork(
        c_th2ctx=bp["c_th2ctx"],
        c_ctx2th=bp["c_ctx2th"],
    )
    m = MultiModel(net)
    m.params["backend"]     = "numba"
    m.params["dt"]          = 0.1
    m.params["sampling_dt"] = 1.0
    m.params["duration"]    = duration
    set_params_glob(m, "*ALNMassEXC*.input_0.mu", bp["mue"])
    set_params_glob(m, "*ALNMassINH*.input_0.mu", bp["mui"])
    set_params_glob(m, "*ALNMassEXC*.b",          bp["b"])
    set_params_glob(m, "*ALNMassEXC*.tauA",       bp["tauA"])
    set_params_glob(m, "*ALNMassEXC*.a",          0.0)
    set_params_glob(m, "*ALNMass*.input_0.sigma",  0.05)
    set_params_glob(m, "*TCR*.input_0.sigma",      0.005)
    set_params_glob(m, "*.input_0.tau",            5.0)
    set_params_glob(m, "*TRN*.g_LK",              0.1)
    set_params_glob(m, "*TCR*.g_LK",              bp["g_LK"])
    set_params_glob(m, "*TCR*.g_h",               bp["g_h"])
    return m


def load_target_psd():
    """Load SC4001 N3 EEG from Sleep-EDF and compute mean PSD."""
    manifest = pd.read_csv("data/manifest.csv")
    subj_row = manifest[manifest["subject_id"] == SUBJECT_ID].iloc[0]
    raw = mne.io.read_raw_edf(
        subj_row["psg_path"], include=EEG_CHANNELS, preload=True, verbose=False
    )
    fs_eeg = raw.info["sfreq"]
    raw.set_annotations(mne.read_annotations(subj_row["hypnogram_path"]))
    event_id = {lbl: idx + 1 for idx, lbl in enumerate(N3_LABELS)}
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
        raise RuntimeError("No N3 epochs passed QC")

    print(f"  EEG: {len(psds)} epochs passed QC")
    return np.mean(psds, axis=0), f_ep[freq_mask]


# --- Main -------------------------------------------------------------------

def main():
    """
    Pipeline: load params -> simulate -> signals & PSD -> FOOOF -> three figures -> print summary.

    Part 1  Load JSON from differential evolution (best V2 parameters).
    Part 2  Run thalamocortical MultiModel for SIM_DUR_MS (60 s; longer than the 30 s fitness eval).
    Part 3  Build time axis; extract cortex/thalamus r_E (Hz); Welch PSD on cortex after 5 s burn-in.
    Part 4  Load empirical N3 EEG mean PSD (same subject, Sleep-EDF) for comparison plots.
    Part 5  FOOOF: aperiodic fit + periodic residuals for EEG and sim; optional Pearson r vs JSON shape_r.
    Part 6  Plot Fig. 7(c): time series window [T_SHOW_START, T_SHOW_END] — cortex + thalamus.
    Part 7  Plot Fig. 7(d): semilogy EEG vs sim PSD + dashed 1/f fits + SO/spindle bands.
    Part 8  Plot residuals: log-domain periodic components on common freq axis (if fooof OK).
    Part 9  Print score / shape_r / dynamics and min/mean r_E for a quick DOWN-state sanity check.
    """
    # ─────────────────────────────────────────────────────────────────────────
    # Part 1 — Load best V2 parameters
    # ─────────────────────────────────────────────────────────────────────────
    if not os.path.isfile(PARAMS_PATH):
        print(f"[error] Missing {PARAMS_PATH}; run models/s4_personalize_fig7_v2.py first.")
        sys.exit(1)

    with open(PARAMS_PATH) as fh:
        bp = json.load(fh)

    print("=" * 55)
    print(f"Best V2 parameters (score={bp.get('score', 'N/A')})")
    print("=" * 55)
    for k, v in bp.items():
        print(f"  {k}: {v}")

    # ─────────────────────────────────────────────────────────────────────────
    # Part 2 — Run simulation (numba; fall back to jitcdde if integration fails)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\nRunning simulation ({SIM_DUR_MS/1000:.0f} s)...")
    model = build_model(bp, duration=SIM_DUR_MS)
    try:
        model.run()
    except Exception as e:
        print(f"  numba backend failed ({e}); switching to jitcdde")
        model.params["backend"] = "jitcdde"
        model.run()

    # ─────────────────────────────────────────────────────────────────────────
    # Part 3 — Time series + simulated cortex PSD (for spectra and FOOOF)
    #
    # Time vector: assumes uniform sampling_dt (ms) over duration (matches neurolib outputs).
    # r_mean_EXC: shape (2, T) for [cortex, thalamus] in kHz -> multiply by 1000 -> Hz.
    # Burn-in: first 5 s discarded for Welch (same habit as s4_personalize_fig7_v2).
    # ─────────────────────────────────────────────────────────────────────────
    n_total = int(SIM_DUR_MS / model.params["sampling_dt"])
    t_s = np.linspace(0, SIM_DUR_MS / 1000, n_total)

    # Same extraction as models/s4_personalize_fig7_v2.py: row 0 = cortex, row 1 = thalamus
    r_exc_raw = model[f"r_mean_{EXC}"]
    if r_exc_raw.ndim == 2 and r_exc_raw.shape[0] >= 2:
        rE_cortex = r_exc_raw[0, :] * 1000.0
        rE_thalamus = r_exc_raw[1, :] * 1000.0
    else:
        rE_cortex = (r_exc_raw[0] if r_exc_raw.ndim == 2 else r_exc_raw) * 1000.0
        rE_thalamus = np.zeros_like(rE_cortex)
        print("  [warn] r_mean_EXC is not 2-node; thalamus trace set to zeros")

    # Align lengths
    n_min = min(len(t_s), len(rE_cortex), len(rE_thalamus))
    t_s        = t_s[:n_min]
    rE_cortex  = rE_cortex[:n_min]
    rE_thalamus = rE_thalamus[:n_min]

    # Burn-in: drop first 5 s
    n_burn = int(5.0 * FS_SIM)
    r_ctx_full = rE_cortex[n_burn:]

    # 4. Simulated PSD (cortex)
    nperseg = min(int(10.0 * FS_SIM), len(r_ctx_full))
    f_sim, p_sim = welch(r_ctx_full, fs=FS_SIM, nperseg=nperseg,
                         noverlap=nperseg // 2, window="hann")
    mask_sim = (f_sim >= F_LO) & (f_sim <= F_HI)
    f_sim, p_sim = f_sim[mask_sim], p_sim[mask_sim]

    # ─────────────────────────────────────────────────────────────────────────
    # Part 4 — Target EEG PSD (empirical N3, subject SC4001)
    # ─────────────────────────────────────────────────────────────────────────
    print("\nLoading target EEG N3 PSD...")
    target_psd, target_freqs = load_target_psd()

    # ─────────────────────────────────────────────────────────────────────────
    # Part 5 — FOOOF: 1/f + periodic residuals; optional shape_r recomputation
    #
    # Target: fit aperiodic on log10(PSD), periodic = log10(PSD) - aperiodic (FOOOF internal freqs).
    # Sim:    fit on native Welch grid (f_sim, p_sim) so 7(d) 1/f line matches plot_fig7_v2.py.
    # Pearson r: interpolate sim periodic onto tgt_fooof_freqs so same x as EEG residual
    #             (approximation vs evolution, which fits on interpolated *power* — close intent).
    # ─────────────────────────────────────────────────────────────────────────
    target_periodic = sim_periodic = fooof_freqs = None
    sim_aperiodic = tgt_aperiodic = None
    tgt_fooof_freqs = sim_fooof_freqs = None
    shape_r_recomputed = None

    if HAS_FOOOF:
        print("Computing FOOOF 1/f residuals...")

        # Target EEG: defines tgt_fooof_freqs used as common x-axis for residual overlay + r
        fm_tgt = FOOOF(peak_width_limits=[1.0, 8.0], max_n_peaks=4,
                       min_peak_height=0.05, aperiodic_mode="fixed")
        fm_tgt.fit(target_freqs, target_psd, [F_LO, F_HI])
        tgt_fooof_freqs = fm_tgt.freqs
        tgt_aperiodic   = fm_tgt._ap_fit
        tgt_log         = np.log10(target_psd + 1e-30)
        target_periodic = tgt_log[:len(tgt_aperiodic)] - tgt_aperiodic

        # Sim: FOOOF directly on Welch PSD (same as plot_fig7_v2.py for 7(d) dashed 1/f)
        try:
            fm_sim = FOOOF(peak_width_limits=[1.0, 8.0], max_n_peaks=4,
                           min_peak_height=0.05, aperiodic_mode="fixed")
            fm_sim.fit(f_sim, p_sim, [F_LO, F_HI])
            sim_fooof_freqs = fm_sim.freqs
            sim_aperiodic   = fm_sim._ap_fit
            sim_log_full    = np.log10(p_sim + 1e-30)
            sim_periodic_own = sim_log_full[:len(sim_aperiodic)] - sim_aperiodic
        except Exception as e:
            print(f"  FOOOF sim failed: {e}")
            sim_fooof_freqs = f_sim
            sim_aperiodic   = None
            sim_periodic_own = None

        # Map sim periodic (native FOOOF freq grid) -> tgt_fooof_freqs for r and residual plot
        if sim_periodic_own is not None:
            sim_periodic_interp = interp1d(
                sim_fooof_freqs, sim_periodic_own,
                bounds_error=False, fill_value=0.0
            )(tgt_fooof_freqs)
            n_r = min(len(sim_periodic_interp), len(target_periodic))
            shape_r_recomputed, _ = pearsonr(
                sim_periodic_interp[:n_r], target_periodic[:n_r]
            )
            print(f"  shape_r (recomputed): {shape_r_recomputed:.4f}")
            print(f"  shape_r (stored in JSON): {bp.get('shape_r', 'N/A')}")
            # Residual figure: EEG and sim share tgt_fooof_freqs as x-axis
            sim_periodic = sim_periodic_interp
            fooof_freqs  = tgt_fooof_freqs
        else:
            sim_periodic = None
            fooof_freqs  = tgt_fooof_freqs

    os.makedirs("outputs", exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Part 6 — Figure: Fig. 7(c) time series -> fig7_v2_timeseries.png
    #
    # Two stacked axes, shared x (time). Only [T_SHOW_START, T_SHOW_END] shown (default 16–32 s).
    # Top: cortex r_E — DOWN threshold line at 1 Hz; bottom: thalamus TCR (often 4–8 Hz y-lim).
    # ─────────────────────────────────────────────────────────────────────────
    print("\nPlotting 7(c) time series...")
    mask_t = (t_s >= T_SHOW_START) & (t_s <= T_SHOW_END)

    fig_c, (ax_c1, ax_c2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig_c.suptitle(
        f"Fig. 7(c) V2 — thalamocortical time series\n"
        f"score={_fmt_json_num(bp, 'score')}, shape_r={_fmt_json_num(bp, 'shape_r')}, "
        f"dynamics={_fmt_json_num(bp, 'dynamics_score', fmt='.2f')}",
        fontsize=12, fontweight="bold"
    )

    # Cortex
    ax_c1.plot(t_s[mask_t], rE_cortex[mask_t], color="#534AB7", lw=0.8)
    ax_c1.axhline(1.0, color="gray", lw=0.5, ls="--", alpha=0.5, label="DOWN threshold (1 Hz)")
    ax_c1.set_ylabel("$r_E$ [Hz]", fontsize=11)
    ax_c1.set_title("Cortex EXC — slow oscillation (SO)", fontsize=10)
    # y-axis: auto upper limit (matches plot_fig7_v2.py behaviour: set_ylim(bottom=-1))
    ax_c1.set_ylim(bottom=-0.1)
    ax_c1.legend(fontsize=8, loc="upper right")
    ax_c1.text(0.02, 0.90, "UP state: high activity", transform=ax_c1.transAxes,
               fontsize=8, color="#534AB7", alpha=0.8)
    ax_c1.text(0.02, 0.05, "DOWN state: near silence", transform=ax_c1.transAxes,
               fontsize=8, color="gray")

    # Thalamus
    ax_c2.plot(t_s[mask_t], rE_thalamus[mask_t], color="#1D9E75", lw=0.8)
    ax_c2.set_ylabel("$r_{TCR}$ [Hz]", fontsize=11)
    ax_c2.set_xlabel("Time [s]", fontsize=11)
    ax_c2.set_title("Thalamic TCR — spindle", fontsize=10)
    # Tight y-range for spindle-band firing (reduces empty space below ~4 Hz)
    ax_c2.set_ylim(4, 8)

    # Best V2 parameters (caption)
    param_txt = (
        f"mue={bp['mue']:.3f}  mui={bp['mui']:.3f}  "
        f"b={bp['b']:.1f}  tauA={bp['tauA']:.0f}\n"
        f"g_LK={bp['g_LK']:.4f}  g_h={bp['g_h']:.4f}  "
        f"c_th2ctx={bp['c_th2ctx']:.4f}  c_ctx2th={bp['c_ctx2th']:.4f}"
    )
    fig_c.text(0.5, 0.01, param_txt, ha="center", fontsize=8,
               color="gray", family="monospace")

    fig_c.tight_layout(rect=[0, 0.05, 1, 1])
    out_c = "outputs/fig7_v2_timeseries.png"
    fig_c.savefig(out_c, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_c}")
    plt.close(fig_c)

    # ─────────────────────────────────────────────────────────────────────────
    # Part 7 — Figure: Fig. 7(d) spectra -> fig7_v2_spectra.png
    #
    # Top: empirical N3 EEG PSD + FOOOF 1/f (dashed) on semilogy.
    # Bottom: simulated cortex PSD (Welch) + sim FOOOF 1/f on *its* freq grid (sim_fooof_freqs).
    # Orange/green bands: SO (~0.2–1.5 Hz) and spindle (~10–14 Hz) for visual reference.
    # Grey text: shape_r / so_power / spindle_power from JSON (evolution metrics).
    # ─────────────────────────────────────────────────────────────────────────
    print("Plotting 7(d) power spectra...")
    fig_d, (ax_d1, ax_d2) = plt.subplots(2, 1, figsize=(8, 8))
    fig_d.suptitle(
        f"Fig. 7(d) V2 — power spectra\n{SUBJECT_ID} N3 EEG vs simulated cortex firing rate",
        fontsize=12, fontweight="bold"
    )

    # Top: EEG target
    ax_d1.semilogy(target_freqs, target_psd, "k", lw=1.8,
                   label=f"Target EEG N3 ({SUBJECT_ID})")
    if tgt_aperiodic is not None:
        ax_d1.semilogy(tgt_fooof_freqs, 10 ** tgt_aperiodic, "b--", lw=1.2,
                       alpha=0.7, label="1/f fit")
    ax_d1.axvspan(0.2, 1.5, alpha=0.10, color="orange", label="SO band")
    ax_d1.axvspan(10.0, 14.0, alpha=0.10, color="green", label="Spindle band")
    ax_d1.set_xlim(F_LO, F_HI)
    ax_d1.set_ylabel("Power [V$^2$/Hz]", fontsize=10)
    ax_d1.set_title("EEG (ground truth)", fontsize=10, loc="right")
    ax_d1.legend(fontsize=8)

    # Bottom: simulated PSD — use f_sim/p_sim directly (matches plot_fig7_v2.py)
    ax_d2.semilogy(f_sim, p_sim, color="#534AB7", lw=1.5,
                   label="Simulated cortex EXC firing-rate PSD")
    if sim_aperiodic is not None:
        # sim_fooof_freqs is FOOOF's own freq grid derived from f_sim
        ax_d2.semilogy(sim_fooof_freqs, 10 ** sim_aperiodic, "b--", lw=1.0,
                       alpha=0.7, label="1/f fit")
    ax_d2.axvspan(0.2, 1.5, alpha=0.10, color="orange", label="SO band")
    ax_d2.axvspan(10.0, 14.0, alpha=0.10, color="green", label="Spindle band")
    ax_d2.set_xlim(F_LO, F_HI)
    ax_d2.set_xlabel("Frequency [Hz]", fontsize=10)
    ax_d2.set_ylabel("Power [Hz$^2$/Hz]", fontsize=10)
    ax_d2.set_title("Simulation (V2)", fontsize=10, loc="right")
    ax_d2.legend(fontsize=8)

    score_txt = (
        f"shape_r={_fmt_json_num(bp, 'shape_r', fmt='.3f')}  "
        f"so_power={_fmt_json_num(bp, 'so_power', fmt='.3f')}  "
        f"spindle_power={_fmt_json_num(bp, 'spindle_power', fmt='.3f')}"
    )
    ax_d2.text(0.98, 0.04, score_txt, transform=ax_d2.transAxes,
               ha="right", fontsize=8, color="gray")

    fig_d.tight_layout()
    out_d = "outputs/fig7_v2_spectra.png"
    fig_d.savefig(out_d, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_d}")
    plt.close(fig_d)

    # ─────────────────────────────────────────────────────────────────────────
    # Part 8 — Figure: periodic residuals -> fig7_v2_residuals.png (requires fooof)
    #
    # Linear y: log10(PSD) minus FOOOF aperiodic (oscillatory "shape" after 1/f removal).
    # Black = EEG target; purple dashed = sim (interpolated to tgt_fooof_freqs in Part 5).
    # Title shows recomputed Pearson r vs JSON shape_r when available.
    # ─────────────────────────────────────────────────────────────────────────
    if HAS_FOOOF and target_periodic is not None:
        print("Plotting FOOOF residual comparison...")
        n_r = min(len(sim_periodic), len(target_periodic))
        ff  = fooof_freqs[:n_r]
        tp  = target_periodic[:n_r]
        sp  = sim_periodic[:n_r]

        fig_r, ax_r = plt.subplots(figsize=(10, 5))
        ax_r.plot(ff, tp, "k-", lw=2.0,
                  label="EEG target N3 (1/f removed)")
        ax_r.plot(ff, sp, color="#534AB7", lw=2.0, ls="--",
                  label="Simulated cortex $r_E$ PSD (1/f removed)")
        ax_r.axhline(0.0, color="gray", lw=0.5, alpha=0.5)
        ax_r.axvspan(0.2, 1.5, alpha=0.10, color="orange", label="SO band")
        ax_r.axvspan(10.0, 14.0, alpha=0.10, color="green", label="Spindle band")

        # Peak markers (Greek band labels)
        for freq_label, freq_val, col in [
            (r"$\delta$ 2.4 Hz", 2.4, "orange"),
            (r"$\theta$ 6.2 Hz", 6.2, "purple"),
            (r"$\alpha$ 9.9 Hz", 9.9, "blue"),
            (r"$\sigma$ 12.5 Hz", 12.5, "green"),
        ]:
            ax_r.axvline(freq_val, color=col, lw=0.8, ls=":", alpha=0.6)
            ax_r.text(freq_val + 0.1, ax_r.get_ylim()[1] * 0.95 if ax_r.get_ylim()[1] > 0 else 0.3,
                      freq_label, fontsize=7, color=col, rotation=90, va="top")

        r_stored = bp.get("shape_r", "?")
        title_str = (
            f"FOOOF residuals (1/f removed) | "
            f"Pearson r (recomputed) = {shape_r_recomputed:.4f}  "
            f"(evolution stored = {r_stored:.4f})"
        ) if isinstance(r_stored, float) else (
            f"FOOOF residuals (1/f removed) | "
            f"Pearson r = {shape_r_recomputed:.4f}"
        )
        ax_r.set_title(title_str, fontsize=11)
        ax_r.set_xlabel("Frequency [Hz]", fontsize=11)
        ax_r.set_ylabel("Log-domain residual (periodic component)", fontsize=10)
        ax_r.set_xlim(F_LO, F_HI)
        ax_r.legend(loc="upper right", fontsize=9)

        fig_r.tight_layout()
        out_r = "outputs/fig7_v2_residuals.png"
        fig_r.savefig(out_r, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_r}")
        plt.close(fig_r)
    else:
        print("  Skipping residual figure (fooof missing or target_periodic empty)")

    # ─────────────────────────────────────────────────────────────────────────
    # Part 9 — Console summary (no new figures)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Done — V2 fast visualization summary")
    print("=" * 55)
    print(f"  score            = {bp.get('score','N/A')}")
    print(f"  shape_r (JSON)   = {bp.get('shape_r','N/A')}")
    if shape_r_recomputed is not None:
        print(f"  shape_r (recomp) = {shape_r_recomputed:.4f}")
    print(f"  dynamics_score   = {bp.get('dynamics_score','N/A')}")
    print(f"  so_power         = {bp.get('so_power','N/A')}")
    print(f"  spindle_power    = {bp.get('spindle_power','N/A')}")
    print(f"\n  Cortex DOWN-state check:")
    print(f"    min r_E (post burn-in) = {rE_cortex[n_burn:].min():.3f} Hz")
    print(f"    mean r_E               = {rE_cortex[n_burn:].mean():.3f} Hz")
    print(f"\nOutput files:")
    print(f"  outputs/fig7_v2_timeseries.png")
    print(f"  outputs/fig7_v2_spectra.png")
    print(f"  outputs/fig7_v2_residuals.png")


if __name__ == "__main__":
    main()