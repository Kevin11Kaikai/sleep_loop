"""
s4_0b_verify_golden_point.py
=============================
Single-point verification of the sweep's identified spindle hotspot.

Purpose:
  Before writing v6, visually confirm that the "golden points" from the
  3D sweep (s4_0_isolated_thalamus_sweep.py) actually produce real
  waxing-waning spindle bursts with SO-spindle coupling — not just
  high metric scores.

  This is the critical "eyeball-the-signals" step that catches metric
  loopholes before they waste 15h of DE runtime.

Design:
  Run 60s simulations at two golden points identified by the sweep:

    Primary   : g_LK=0.043, g_h=0.066, c_ctx2th=0.155
                (33 events, MI=0.018, sp_log=2.24 — best "real spindle +
                real PAC" combination)

    Control   : g_LK=0.049, g_h=0.066, c_ctx2th=0.020
                (weak c_ctx2th — predicts real spindle BUT weak PAC;
                this isolates "which parameter controls PAC vs spindle")

  Produce a 6-panel diagnostic figure for each point, side-by-side:
    (1) Cortex r_E time series (with SO up-state shading)
    (2) Thalamus r_TCR time series (with detected spindle event boxes)
    (3) Thalamus power spectrum (Welch)
    (4) Spindle-band envelope (Hilbert, smoothed) + detection threshold
    (5) Zoom-in on one spindle (6s window) showing 13Hz oscillation + envelope
    (6) Phase-locked spindle amplitude curve (SO phase vs mean σ-amplitude)

Output:
  outputs/golden_point_verification.png   — side-by-side figure (2 points)
  outputs/golden_point_metrics.txt        — quantitative comparison table

Usage:
  python models/s4_0b_verify_golden_point.py
  (~2-3 minutes total: 60s sim × 2 points × ~30s wall-clock each)
"""

import os
import sys
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy.signal import hilbert, butter, sosfiltfilt, welch
from scipy.ndimage import gaussian_filter1d

from neurolib.models.multimodel import MultiModel, ALNNode, ThalamicNode
from neurolib.models.multimodel.builder.base.network import Network
from neurolib.models.multimodel.builder.base.constants import EXC, INH


# =====================================================================
# CONFIGURATION
# =====================================================================

FS_SIM     = 1000.0
SIM_DUR_MS = 60_000          # 60 s
BURN_IN_S  = 5.0

# Cortex fixed parameters (same as sweep script: v4b best feasible)
CORTEX_FIXED = dict(
    mue   = 3.895,
    mui   = 3.027,
    b     = 35.5,
    tauA  = 1426.0,
)

# The two points to verify
GOLDEN_POINTS = [
    dict(
        name   = "PRIMARY: real spindle + PAC",
        label  = "primary",
        g_LK   = 0.043,
        g_h    = 0.066,
        c_ctx2th = 0.155,
        # What the sweep predicted:
        predicted = dict(sp_log=2.24, n_ver=33, pac_mi=0.018),
    ),
    dict(
        name   = "CONTROL: real spindle WITHOUT strong PAC",
        label  = "control",
        g_LK   = 0.049,
        g_h    = 0.066,
        c_ctx2th = 0.020,
        predicted = dict(sp_log=1.44, n_ver=24, pac_mi=0.010),
    ),
]

# Spindle detection (same as sweep, v5-fixed thresholds)
SO_FREQ_LO, SO_FREQ_HI = 0.2, 1.5
SPINDLE_LO, SPINDLE_HI = 10.0, 14.0
SPINDLE_DUR_LO_S = 0.3
SPINDLE_DUR_HI_S = 2.0
SPINDLE_EVT_PCTILE     = 75.0
SPINDLE_ENV_SMOOTH_MS  = 200.0
PEAK_INSIDE_MIN_PWR_RATIO = 1.5
PAC_N_BINS = 18

OUT_PNG = "outputs/golden_point_verification.png"
OUT_TXT = "outputs/golden_point_metrics.txt"


# =====================================================================
# Model builder — same architecture as sweep script (isolated thalamus)
# =====================================================================

def set_params_glob(model, pattern, value):
    for k in (k for k in model.params if fnmatch.fnmatch(k, pattern)):
        model.params[k] = value


class ThalamoCorticalNetwork(Network):
    name            = "Thalamocortical Motif (verify)"
    label           = "TCNet_verify"
    sync_variables  = ["network_exc_exc", "network_exc_exc_sq", "network_inh_exc"]
    default_output  = f"r_mean_{EXC}"
    output_vars     = [f"r_mean_{EXC}", f"r_mean_{INH}"]
    _EXC_WITHIN_IDX = [6, 9]

    def __init__(self, c_th2ctx=0.0, c_ctx2th=0.04):
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
        couplings += self._additive_coupling(
            wi, "network_exc_exc_sq", connectivity=self.connectivity ** 2
        )
        couplings += self._additive_coupling(wi, "network_inh_exc")
        return couplings


def run_golden_point(g_lk, g_h, c_ctx2th):
    """Run one 60s simulation at the given parameters, return (r_ctx, r_thal)."""
    net = ThalamoCorticalNetwork(c_th2ctx=0.0, c_ctx2th=c_ctx2th)
    m = MultiModel(net)
    m.params["backend"]     = "numba"
    m.params["dt"]          = 0.1
    m.params["sampling_dt"] = 1.0
    m.params["duration"]    = SIM_DUR_MS
    set_params_glob(m, "*ALNMassEXC*.input_0.mu", CORTEX_FIXED["mue"])
    set_params_glob(m, "*ALNMassINH*.input_0.mu", CORTEX_FIXED["mui"])
    set_params_glob(m, "*ALNMassEXC*.b",          CORTEX_FIXED["b"])
    set_params_glob(m, "*ALNMassEXC*.tauA",       CORTEX_FIXED["tauA"])
    set_params_glob(m, "*ALNMassEXC*.a",          0.0)
    set_params_glob(m, "*ALNMass*.input_0.sigma", 0.05)
    set_params_glob(m, "*TCR*.input_0.sigma",     0.005)
    set_params_glob(m, "*.input_0.tau",           5.0)
    set_params_glob(m, "*TRN*.g_LK",              0.1)
    set_params_glob(m, "*TCR*.g_LK",              g_lk)
    set_params_glob(m, "*TCR*.g_h",               g_h)

    print(f"  Running simulation (60s, numba)...", flush=True)
    t0 = time.time()
    try:
        m.run()
    except Exception as e:
        print(f"  numba failed ({e}), falling back to jitcdde...")
        m.params["backend"] = "jitcdde"
        m.run()
    print(f"  Done in {time.time()-t0:.1f}s")

    r_exc = m[f"r_mean_{EXC}"]
    if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
        r_ctx  = r_exc[0, :] * 1000.0
        r_thal = r_exc[1, :] * 1000.0
    else:
        raise RuntimeError("Unexpected output shape from MultiModel")

    # Burn-in
    n_drop = int(BURN_IN_S * FS_SIM)
    return r_ctx[n_drop:], r_thal[n_drop:]


# =====================================================================
# Analysis functions
# =====================================================================

def detect_spindles(r_thal, fs=FS_SIM):
    """Returns: (starts, ends, env_smooth, threshold, verified_mask)."""
    sos = butter(4, [SPINDLE_LO, SPINDLE_HI], btype="band", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, r_thal)
    envelope = np.abs(hilbert(filtered))
    sigma_samples = SPINDLE_ENV_SMOOTH_MS * fs / 1000.0
    env_smooth = gaussian_filter1d(envelope, sigma=sigma_samples)
    thresh = np.percentile(env_smooth, SPINDLE_EVT_PCTILE)
    above = (env_smooth > thresh).astype(np.int8)
    diff = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    durs = (ends - starts) / fs
    valid_mask = (durs >= SPINDLE_DUR_LO_S) & (durs <= SPINDLE_DUR_HI_S)
    starts_v = starts[valid_mask]
    ends_v = ends[valid_mask]

    # Peak-inside-event verification
    verified = np.zeros(len(starts_v), dtype=bool)
    for i, (s, e) in enumerate(zip(starts_v, ends_v)):
        event = r_thal[s:e]
        if len(event) < int(0.2 * fs):
            continue
        f, p = welch(event, fs=fs, nperseg=min(len(event), 512))
        sp = (f >= SPINDLE_LO) & (f <= SPINDLE_HI)
        ns = (f >= 4) & (f < SPINDLE_LO)
        if not sp.any() or not ns.any():
            continue
        ratio = p[sp].max() / (p[ns].max() + 1e-12)
        verified[i] = (ratio > PEAK_INSIDE_MIN_PWR_RATIO)

    return starts_v, ends_v, env_smooth, thresh, verified


def compute_pac(r_ctx, r_thal, fs=FS_SIM):
    """Returns dict with mi, preferred_phase, phase_amplitude curve."""
    sos_so = butter(4, [SO_FREQ_LO, SO_FREQ_HI], btype="band", fs=fs, output="sos")
    so_filt = sosfiltfilt(sos_so, r_ctx)
    so_phase = np.angle(hilbert(so_filt))

    sos_sp = butter(4, [SPINDLE_LO, SPINDLE_HI], btype="band", fs=fs, output="sos")
    sp_filt = sosfiltfilt(sos_sp, r_thal)
    sp_amp = np.abs(hilbert(sp_filt))

    edge = int(0.5 * fs)
    so_phase = so_phase[edge:-edge]
    sp_amp = sp_amp[edge:-edge]

    bin_edges = np.linspace(-np.pi, np.pi, PAC_N_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mean_amp = np.zeros(PAC_N_BINS)
    for i in range(PAC_N_BINS):
        mask = (so_phase >= bin_edges[i]) & (so_phase < bin_edges[i + 1])
        if mask.any():
            mean_amp[i] = sp_amp[mask].mean()
    total = mean_amp.sum()
    p = mean_amp / (total + 1e-12)
    p_safe = np.where(p > 0, p, 1.0)
    H = -np.sum(p * np.log(p_safe))
    mi = (np.log(PAC_N_BINS) - H) / np.log(PAC_N_BINS)

    mvl = (sp_amp * np.exp(1j * so_phase)).mean()
    preferred_phase = np.angle(mvl)

    return dict(
        mi=float(mi),
        preferred_phase=float(preferred_phase),
        bin_centers=bin_centers,
        mean_amp=mean_amp,
    )


# =====================================================================
# Plotting — 6 panels per golden point
# =====================================================================

def plot_golden_point(axes_col, r_ctx, r_thal, point_info, fs=FS_SIM):
    """Fill one column of a 6-row figure with diagnostic panels."""
    t = np.arange(len(r_ctx)) / fs

    starts, ends, env_smooth, thresh, verified = detect_spindles(r_thal, fs)
    n_valid = len(starts)
    n_verified = int(verified.sum())
    pac = compute_pac(r_ctx, r_thal, fs)

    # σ peak power
    f, p = welch(r_thal, fs=fs, nperseg=int(4 * fs))
    sp_mask = (f >= SPINDLE_LO) & (f <= SPINDLE_HI)
    floor_mask = (f >= 4) & (f < SPINDLE_LO)
    if sp_mask.any() and floor_mask.any() and p[floor_mask].mean() > 0:
        sp_log = np.log10(p[sp_mask].max() / p[floor_mask].mean())
    else:
        sp_log = 0.0

    # Display window: a 16-second window centered mid-simulation
    win_start = 15.0
    win_end = 31.0
    win_mask = (t >= win_start) & (t < win_end)
    t_win = t[win_mask]

    # ----- Panel 1: Cortex r_E -----
    ax = axes_col[0]
    ax.plot(t_win, r_ctx[win_mask], color="steelblue", lw=0.8)
    ax.axhline(15, color='gray', ls=':', lw=0.5, alpha=0.5)
    ax.set_ylabel("r_E [Hz]")
    ax.set_title(
        f"{point_info['name']}\n"
        f"g_LK={point_info['g_LK']:.3f}  g_h={point_info['g_h']:.3f}  "
        f"c_ctx2th={point_info['c_ctx2th']:.3f}",
        fontsize=9,
    )
    ax.set_xlim(win_start, win_end)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, "Cortex EXC (SO generator)",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # ----- Panel 2: Thalamus r_TCR with detected events -----
    ax = axes_col[1]
    ax.plot(t_win, r_thal[win_mask], color="seagreen", lw=0.6)
    # Shade verified events
    y_max = r_thal[win_mask].max() * 1.05
    for i, (s, e) in enumerate(zip(starts, ends)):
        t_s = s / fs
        t_e = e / fs
        if t_s >= win_end or t_e < win_start:
            continue
        color = 'gold' if verified[i] else 'lightcoral'
        ax.axvspan(t_s, t_e, alpha=0.3, color=color,
                   label='_nolegend_')
    ax.set_ylabel("r_TCR [Hz]")
    ax.set_xlim(win_start, win_end)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95,
            f"Thalamus TCR — gold=verified ({n_verified}), "
            f"coral=unverified ({n_valid-n_verified})",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # ----- Panel 3: Thalamus PSD -----
    ax = axes_col[2]
    ax.loglog(f, p, color='seagreen', lw=1.0)
    ax.axvspan(SPINDLE_LO, SPINDLE_HI, alpha=0.2, color='lightgreen',
               label='σ band')
    ax.axvspan(SO_FREQ_LO, SO_FREQ_HI, alpha=0.2, color='orange',
               label='SO band')
    ax.set_xlim(0.3, 50)
    ax.set_xlabel("Freq [Hz]")
    ax.set_ylabel("Power")
    ax.grid(True, alpha=0.3, which='both')
    ax.text(0.02, 0.95,
            f"Thalamic PSD  |  sp_peak_log={sp_log:.2f}",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # ----- Panel 4: Spindle envelope + threshold -----
    ax = axes_col[3]
    ax.plot(t_win, env_smooth[win_mask], color='darkgreen', lw=0.8,
            label='Smoothed σ envelope')
    ax.axhline(thresh, color='red', ls='--', lw=0.8,
               label=f'75th pct threshold ({thresh:.2f})')
    for i, (s, e) in enumerate(zip(starts, ends)):
        t_s, t_e = s / fs, e / fs
        if t_s >= win_end or t_e < win_start:
            continue
        color = 'gold' if verified[i] else 'lightcoral'
        ax.axvspan(t_s, t_e, alpha=0.25, color=color)
    ax.set_ylabel("σ envelope")
    ax.set_xlim(win_start, win_end)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    # ----- Panel 5: Zoom on first verified spindle -----
    ax = axes_col[4]
    # Find a verified spindle in the display window for zoom
    zoom_start = None
    for i, (s, e) in enumerate(zip(starts, ends)):
        if verified[i]:
            t_s = s / fs
            if win_start <= t_s < win_end:
                zoom_start = t_s - 1.5  # 1.5s pre-context
                zoom_end = t_s + 2.5    # total 4s window
                break
    if zoom_start is None and len(starts) > 0:
        # fall back to any event
        zoom_start = starts[0] / fs - 1.5
        zoom_end = zoom_start + 4.0

    if zoom_start is not None:
        zoom_mask = (t >= zoom_start) & (t < zoom_end)
        t_zoom = t[zoom_mask]

        # Band-pass filtered σ signal for zoom (to see 13 Hz oscillation)
        sos_sp = butter(4, [SPINDLE_LO, SPINDLE_HI], btype="band",
                        fs=fs, output="sos")
        sp_filt = sosfiltfilt(sos_sp, r_thal)
        env_full = np.abs(hilbert(sp_filt))

        # Scale envelope to match σ signal range for overlay
        y = sp_filt[zoom_mask]
        e_y = env_full[zoom_mask]
        ax.plot(t_zoom, y, color='seagreen', lw=0.5, alpha=0.7,
                label='σ-band signal (10-14Hz)')
        ax.plot(t_zoom, e_y, color='black', lw=1.2, label='Envelope')
        ax.plot(t_zoom, -e_y, color='black', lw=1.2)
        ax.set_xlim(zoom_start, zoom_end)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("σ signal")
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.95, "Zoom: single spindle (4s window)",
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    else:
        ax.text(0.5, 0.5, "No spindle event to zoom",
                transform=ax.transAxes, ha='center', va='center')

    # ----- Panel 6: Phase-locked spindle amplitude histogram -----
    ax = axes_col[5]
    # Use the PAC computation's bin-averaged spindle amplitude
    bars = ax.bar(np.degrees(pac['bin_centers']),
                  pac['mean_amp'],
                  width=360 / PAC_N_BINS * 0.9,
                  color='slateblue', edgecolor='black', lw=0.3)
    # Mark preferred phase
    pref_deg = np.degrees(pac['preferred_phase'])
    ax.axvline(pref_deg, color='red', ls='--', lw=1.5,
               label=f'Preferred phase = {pref_deg:+.0f}°')
    ax.axvline(0, color='orange', ls=':', lw=1, alpha=0.6,
               label='SO Up-peak (0°)')
    ax.set_xlabel("SO phase [degrees]")
    ax.set_ylabel("Mean σ amplitude")
    ax.set_xlim(-180, 180)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95,
            f"PAC: MI={pac['mi']:.4f}  "
            f"φ_pref={pref_deg:+.0f}°",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Return metrics for text summary
    return dict(
        sp_log=sp_log,
        n_valid=n_valid,
        n_verified=n_verified,
        pac_mi=pac['mi'],
        pac_preferred_phase_deg=pref_deg,
    )


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("Golden Point Verification (before v6)")
    print("=" * 70)
    print(f"  Cortex fixed: {CORTEX_FIXED}")
    print(f"  Sim duration: {SIM_DUR_MS/1000:.0f}s (burn-in {BURN_IN_S}s)")
    print(f"  Verifying {len(GOLDEN_POINTS)} golden points from the sweep")
    print("=" * 70)

    os.makedirs("outputs", exist_ok=True)

    # Run simulations and collect data
    results = []
    for gp in GOLDEN_POINTS:
        print(f"\n[{gp['label'].upper()}] {gp['name']}")
        print(f"  g_LK={gp['g_LK']:.3f}  g_h={gp['g_h']:.3f}  "
              f"c_ctx2th={gp['c_ctx2th']:.3f}")
        r_ctx, r_thal = run_golden_point(gp['g_LK'], gp['g_h'],
                                          gp['c_ctx2th'])
        results.append(dict(point=gp, r_ctx=r_ctx, r_thal=r_thal))

    # Side-by-side figure, 6 rows × 2 columns
    n_cols = len(results)
    fig, axes = plt.subplots(6, n_cols, figsize=(7 * n_cols, 14), squeeze=False)

    metrics_list = []
    for col, res in enumerate(results):
        metrics = plot_golden_point(axes[:, col], res['r_ctx'], res['r_thal'],
                                    res['point'])
        metrics['point'] = res['point']
        metrics_list.append(metrics)

    fig.suptitle(
        "Golden Point Verification — Visual Confirmation of Spindle Activity\n"
        "Sweep-identified parameters should produce real waxing-waning spindles with SO-PAC",
        fontsize=11, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=140, bbox_inches='tight')
    print(f"\nSaved figure: {OUT_PNG}")
    plt.close()

    # Text summary comparing sweep prediction vs actual
    lines = []
    lines.append("=" * 78)
    lines.append("GOLDEN POINT VERIFICATION — Sweep prediction vs. Actual measurement")
    lines.append("=" * 78)
    for m in metrics_list:
        gp = m['point']
        pred = gp['predicted']
        lines.append(f"\n[{gp['label'].upper()}]  {gp['name']}")
        lines.append(f"  Params: g_LK={gp['g_LK']:.3f}  g_h={gp['g_h']:.3f}  "
                     f"c_ctx2th={gp['c_ctx2th']:.3f}")
        lines.append("")
        lines.append(f"  {'Metric':<25s} {'Sweep pred.':>14s} {'Actual':>14s} "
                     f"{'Δ':>10s}")
        lines.append(f"  {'-'*25} {'-'*14} {'-'*14} {'-'*10}")
        for key, label in [('sp_log',  'sp_peak_power_log'),
                           ('n_ver',   'n_events_verified'),
                           ('pac_mi',  'pac_mi')]:
            actual_key = {'sp_log': 'sp_log', 'n_ver': 'n_verified',
                          'pac_mi': 'pac_mi'}[key]
            p = pred[key]; a = m[actual_key]
            delta = a - p
            lines.append(f"  {label:<25s} {p:>14.3f} {a:>14.3f} "
                         f"{delta:>+10.3f}")
        # T10 two-sided check (A-revision)
        phi_rad = np.radians(m['pac_preferred_phase_deg'])
        dist_to_target = min(abs(phi_rad), np.pi - abs(phi_rad))
        PAC_PHASE_TOL_DEG = 50.0
        t10_pass = np.degrees(dist_to_target) < PAC_PHASE_TOL_DEG
        lines.append(f"  {'preferred_phase [deg]':<25s} {'-':>14s} "
                     f"{m['pac_preferred_phase_deg']:>+14.1f} {'-':>10s}")
        lines.append(f"  {'dist_to_target [deg]':<25s} {'-':>14s} "
                     f"{np.degrees(dist_to_target):>+14.1f} {'-':>10s}")
        lines.append(f"  {'T10 (two-sided ±50°)':<25s} {'-':>14s} "
                     f"{'PASS' if t10_pass else 'FAIL':>14s} {'-':>10s}")

    lines.append("\n" + "=" * 78)
    lines.append("INTERPRETATION CHECKLIST:")
    lines.append("=" * 78)
    lines.append("  [1] Cortex r_E shows clear Up/Down alternation around 0.8 Hz?")
    lines.append("      → required: yes for both points (cortex params fixed)")
    lines.append("  [2] Thalamic TCR shows waxing-waning bursts (not sporadic spikes)?")
    lines.append("      → required: YES for PRIMARY (this is the critical test)")
    lines.append("  [3] Thalamic PSD has a clear σ-band peak (10-14 Hz)?")
    lines.append("      → required: YES for PRIMARY, weak or absent for CONTROL OK")
    lines.append("  [4] Spindle events align with SO Up-state phase OR Down-to-Up transition?")
    lines.append("      → expected: preferred_phase near 0° (fast spindle)")
    lines.append("        OR near ±180° (slow spindle, Mölle 2011)")
    lines.append("      → T10 passes if within ±50° of either target")
    lines.append("  [5] PAC is stronger for PRIMARY than CONTROL?")
    lines.append("      → isolates effect of c_ctx2th on coupling")
    lines.append("")
    lines.append("If all 5 pass: proceed to write v6 with sweep-based bounds.")
    lines.append("If any fail:   metrics have a loophole — do NOT run v6 yet.")

    txt = "\n".join(lines)
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(f"Saved summary: {OUT_TXT}")
    print()
    print(txt)


if __name__ == "__main__":
    main()
