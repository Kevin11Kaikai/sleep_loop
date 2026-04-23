"""
s4_0_isolated_thalamus_sweep.py
================================
3D parameter sweep of the thalamic subsystem to map the spindle-generating
region, BEFORE running the full DE personalization (v6+).

Rationale (from v5 post-mortem):
  v5 produced only 2 evaluations (out of 9920 combined with v4b) in which
  the FOOOF detector found a real σ-band peak (spindle_power > 0.1).
  All other "T8-passing" solutions were sporadic spikes masquerading as
  spindle events. This indicates the v4 narrowed bounds do NOT contain
  a physiologically meaningful spindle-generating region.

  Following Jajcay et al. 2022 (Front Comput Neurosci, Fig. 3), we map
  the parameter landscape in 3D before further optimization.

Design:
  • Cortex is NOT removed — ALNNode still generates slow oscillations
    locally (adaptation-driven limit cycle, same mechanism as in v5).
  • The feedforward pathway thalamus → cortex is disabled (c_th2ctx = 0),
    isolating the THALAMUS's ability to respond. The cortex drives the
    thalamus via c_ctx2th as in the intact motif, but cannot be affected
    back. This isolates "what can the thalamus produce under realistic
    cortical SO drive?"

  • Swept dimensions (Q1=B):
      - g_LK     ∈ [0.02, 0.10] (TCR potassium leak)        [15 steps]
      - g_h      ∈ [0.02, 0.10] (TCR rectifying cation)     [15 steps]
      - c_ctx2th ∈ [0.02, 0.20] (cortex → thalamus drive)   [ 5 steps]
    Total: 1125 grid points.

  • At each point: simulate 60s, burn-in 5s, then compute 5 spindle-
    region indicators (from the v5 T5-T11 battery) on the thalamic
    TCR firing rate.

  • Cortex params are fixed at v4b best feasible values (the region v5
    also inhabits) — see CORTEX_FIXED below.

Output:
  data/thalamus_sweep_3d.csv     — full 1125-row grid with all metrics
  outputs/thalamus_sweep_3d.npz  — npz arrays for plotting heatmaps

Timeline expectation:
  Single-core, numba backend: 60s sim ≈ 10-20s wall-clock.
  1125 × 15s ≈ 4.7 hours. Overnight-safe.

Usage:
  python models/s4_0_isolated_thalamus_sweep.py
  # then: python plot_scripts/plot_thalamus_sweep_heatmaps.py
"""

import os
import sys
import time
import json
import fnmatch
import importlib.util
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
# numpy compatibility shim (same as v5)
for _attr in ("object", "bool", "int", "float", "complex", "str"):
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(__builtins__, _attr, object))

import pandas as pd
from scipy.signal import hilbert, butter, sosfiltfilt, correlate, welch

from neurolib.models.multimodel import MultiModel, ALNNode, ThalamicNode
from neurolib.models.multimodel.builder.base.network import Network
from neurolib.models.multimodel.builder.base.constants import EXC, INH


# Import preprocessing (used only if we later want to reuse compute_epoch_psd)
_spec = importlib.util.spec_from_file_location(
    "02_preprocess_psd", "utils/02_preprocess_psd.py"
)
prep_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep_mod)
compute_epoch_psd = prep_mod.compute_epoch_psd


# =====================================================================
# CONFIGURATION
# =====================================================================

FS_SIM       = 1000.0
SIM_DUR_MS   = 60_000      # 60 s per grid point (longer than v5's 30s
                           # to get more spindle events per eval)
BURN_IN_S    = 5.0         # discard first 5 s

OUT_CSV      = "data/thalamus_sweep_3d.csv"
OUT_NPZ      = "outputs/thalamus_sweep_3d.npz"

# ---------------------------------------------------------------------
# Swept parameters (Q4 = reviewer's recommendation)
# ---------------------------------------------------------------------
G_LK_GRID     = np.linspace(0.02, 0.10, 15)      # TCR potassium leak
G_H_GRID      = np.linspace(0.02, 0.10, 15)      # TCR h-current
C_CTX2TH_GRID = np.linspace(0.02, 0.20, 5)       # cortex → thalamus

N_POINTS = len(G_LK_GRID) * len(G_H_GRID) * len(C_CTX2TH_GRID)

# ---------------------------------------------------------------------
# Cortex parameters — FIXED at v4b best feasible
# These ensure the cortex produces realistic adaptation-driven SO so
# that the thalamus experiences physiological up/down-state drive.
# ---------------------------------------------------------------------
CORTEX_FIXED = dict(
    mue       = 3.895,
    mui       = 3.027,
    b         = 35.5,
    tauA      = 1426.0,
    c_th2ctx  = 0.0,     # << DISABLED: isolates thalamus (no feedback to cortex)
)

# ---------------------------------------------------------------------
# Spindle-region detection thresholds (inherited from v5)
# ---------------------------------------------------------------------
SO_FREQ_LO       = 0.2
SO_FREQ_HI       = 1.5
SPINDLE_LO       = 10.0
SPINDLE_HI       = 14.0

# T5-style: spindle FWHM on thalamic PSD
SPINDLE_FWHM_MIN = 2.0

# T7-style: envelope CV
SPINDLE_CV_MIN   = 0.7

# T8-style: discrete spindle events
SPINDLE_EVT_MIN  = 5
SPINDLE_DUR_LO_S = 0.3
SPINDLE_DUR_HI_S = 2.0
# ROBUSTNESS FIX (not in v5):
# v5 used threshold = mean(env) + 1.5*std(env). This FAILS on bimodal
# envelope distributions (long silences + long bursts), because mean
# and std are dominated by the burst mode, and the threshold lands ABOVE
# the burst mean — zero events detected.
# Fix: use a percentile-based threshold (robust, standard in Lacourse A7
# and YASA spindle detection). Also smooth envelope over 200ms before
# thresholding (standard practice in sleep-spindle literature).
SPINDLE_EVT_PCTILE       = 75.0    # use 75th percentile of smoothed envelope
SPINDLE_ENV_SMOOTH_MS    = 200.0   # Gaussian smoothing before thresholding

# T9-style: PAC MI
PAC_N_BINS       = 18

# NEW (v6-preview): peak-inside-event check. For each detected spindle
# event, verify that the dominant frequency within that event falls in
# the spindle band [10, 14] Hz. This plugs the T8 loophole v5 exposed.
PEAK_INSIDE_MIN_PWR_RATIO = 1.5   # σ-band power / non-σ-band power within event


# =====================================================================
# Model builder — reuses v5's ThalamoCorticalNetwork design
# =====================================================================

def set_params_glob(model, pattern, value):
    for k in (k for k in model.params if fnmatch.fnmatch(k, pattern)):
        model.params[k] = value


class ThalamoCorticalNetwork(Network):
    """Same as v5, but typically run with c_th2ctx=0 to isolate thalamus."""
    name            = "Thalamocortical Motif (sweep)"
    label           = "TCNet_sweep"
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


def build_sweep_model(g_lk, g_h, c_ctx2th, duration=SIM_DUR_MS):
    """Build model with fixed cortex params + swept thalamus params."""
    net = ThalamoCorticalNetwork(
        c_th2ctx=CORTEX_FIXED["c_th2ctx"],   # = 0 (isolated)
        c_ctx2th=c_ctx2th,
    )
    m = MultiModel(net)
    m.params["backend"]     = "numba"
    m.params["dt"]          = 0.1
    m.params["sampling_dt"] = 1.0
    m.params["duration"]    = duration
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
    return m


# =====================================================================
# Metric computation — spindle region indicators
# =====================================================================

def _fooof_spindle_power(f, p):
    """Cheap FOOOF-free spindle power: peak-to-floor ratio in σ band."""
    sp_mask = (f >= SPINDLE_LO) & (f <= SPINDLE_HI)
    floor_mask = ((f >= 4) & (f < SPINDLE_LO)) | ((f > SPINDLE_HI) & (f <= 20))
    if not sp_mask.any() or not floor_mask.any():
        return 0.0
    peak = p[sp_mask].max()
    floor = p[floor_mask].mean()
    if floor <= 0:
        return 0.0
    # Return log-ratio, capped at [0, 10]
    return float(np.clip(np.log10(peak / floor), 0, 10))


def detect_spindle_events(r_thal, fs):
    """Return list of (start, end) indices for σ-envelope-threshold events.

    Robust against bimodal envelope distributions (v5 latent bug fix):
    uses percentile threshold on Gaussian-smoothed envelope instead of
    mean+k*std, which fails when bursts occupy a large fraction of the
    signal (pulling mean and std upward together).
    """
    try:
        from scipy.ndimage import gaussian_filter1d
        sos = butter(4, [SPINDLE_LO, SPINDLE_HI], btype="band",
                     fs=fs, output="sos")
        filtered = sosfiltfilt(sos, r_thal)
        envelope = np.abs(hilbert(filtered))
        # Smooth envelope over ~200ms (standard for spindle detection)
        sigma_samples = SPINDLE_ENV_SMOOTH_MS * fs / 1000.0
        env_smooth = gaussian_filter1d(envelope, sigma=sigma_samples)
        # Percentile threshold (robust to bimodality)
        thresh = np.percentile(env_smooth, SPINDLE_EVT_PCTILE)
        above = (env_smooth > thresh).astype(np.int8)
        diff = np.diff(np.concatenate(([0], above, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        durations = (ends - starts) / fs
        valid = (durations >= SPINDLE_DUR_LO_S) & (durations <= SPINDLE_DUR_HI_S)
        return list(zip(starts[valid], ends[valid])), envelope
    except Exception:
        return [], None


def verify_peak_inside_event(r_thal, start, end, fs):
    """
    v6-preview check: within a detected event window, is σ-band power
    actually dominant over non-σ-band power?

    Returns True if the event contains a real spindle oscillation.
    """
    event = r_thal[start:end]
    if len(event) < int(0.2 * fs):     # too short for PSD
        return False
    nperseg = min(len(event), 512)
    f, p = welch(event, fs=fs, nperseg=nperseg)
    sp_mask = (f >= SPINDLE_LO) & (f <= SPINDLE_HI)
    nonsp_mask = (f >= 4) & (f < SPINDLE_LO)     # below σ, above δ
    if not sp_mask.any() or not nonsp_mask.any():
        return False
    sp_power = p[sp_mask].max()
    nonsp_power = p[nonsp_mask].max() if p[nonsp_mask].size > 0 else 1e-12
    if nonsp_power <= 0:
        return True     # no competing power → assume spindle
    return (sp_power / nonsp_power) > PEAK_INSIDE_MIN_PWR_RATIO


def compute_pac_mi(r_ctx, r_thal, fs):
    """Lean KL-MI calculation (copy of v5 PAC helper's MI portion)."""
    try:
        if len(r_ctx) < int(2 * fs) or r_ctx.std() < 1e-6:
            return 0.0
        sos_so = butter(4, [SO_FREQ_LO, SO_FREQ_HI], btype="band",
                        fs=fs, output="sos")
        so_filt = sosfiltfilt(sos_so, r_ctx)
        so_phase = np.angle(hilbert(so_filt))

        sos_sp = butter(4, [SPINDLE_LO, SPINDLE_HI], btype="band",
                        fs=fs, output="sos")
        sp_filt = sosfiltfilt(sos_sp, r_thal)
        sp_amp = np.abs(hilbert(sp_filt))

        edge = int(0.5 * fs)
        so_phase = so_phase[edge:-edge]
        sp_amp = sp_amp[edge:-edge]

        bin_edges = np.linspace(-np.pi, np.pi, PAC_N_BINS + 1)
        mean_amp = np.zeros(PAC_N_BINS)
        for i in range(PAC_N_BINS):
            mask = (so_phase >= bin_edges[i]) & (so_phase < bin_edges[i + 1])
            if mask.any():
                mean_amp[i] = sp_amp[mask].mean()
        total = mean_amp.sum()
        if total <= 0:
            return 0.0
        p = mean_amp / total
        p_safe = np.where(p > 0, p, 1.0)
        H = -np.sum(p * np.log(p_safe))
        return float(np.clip((np.log(PAC_N_BINS) - H) / np.log(PAC_N_BINS),
                             0.0, 1.0))
    except Exception:
        return 0.0


def score_grid_point(r_ctx, r_thal, fs=FS_SIM):
    """
    Compute all spindle-region indicators for a single grid point.
    Returns dict of scalar metrics.
    """
    out = dict(
        # Section 1: raw spindle power
        sp_peak_power_log=0.0,   # log10(σ-peak / floor), proxy for FOOOF peak
        # Section 2: event-based (T7 + T8 + v6 peak-inside-event)
        sp_env_cv=0.0,
        n_events_raw=0,          # all envelope-threshold events
        n_events_valid=0,        # duration in [0.3, 2.0] s
        n_events_verified=0,     # v6-preview: peak-inside-event passes
        mean_event_dur=0.0,
        # Section 3: PAC
        pac_mi=0.0,
        # Section 4: sanity on the cortex (did SO still happen?)
        ctx_min=0.0,
        ctx_max=0.0,
        ctx_ibi_cv=999.0,
        ctx_n_bursts=0,
        # Flag
        ok=True,
        fail_reason="",
    )

    # Sanity: did cortex produce activity?
    if r_ctx.max() < 1.0 or r_ctx.std() < 1e-3:
        out["ok"] = False
        out["fail_reason"] = "cortex_flat"
        return out

    out["ctx_min"] = float(r_ctx.min())
    out["ctx_max"] = float(r_ctx.max())

    # Cortex burst timing (T6-style)
    above = (r_ctx > 15.0).astype(np.int8)
    diff = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(diff == 1)[0]
    out["ctx_n_bursts"] = int(len(starts))
    if len(starts) >= 3:
        intervals = np.diff(starts) / fs
        out["ctx_ibi_cv"] = float(intervals.std() / (intervals.mean() + 1e-12))

    # Thalamic σ peak — quick & cheap spectrum
    try:
        f_th, p_th = welch(r_thal, fs=fs, nperseg=int(4 * fs))
        out["sp_peak_power_log"] = _fooof_spindle_power(f_th, p_th)
    except Exception:
        pass

    # Envelope + events
    events, envelope = detect_spindle_events(r_thal, fs)
    if envelope is not None and envelope.std() > 0:
        out["sp_env_cv"] = float(envelope.std() / (envelope.mean() + 1e-12))

    # Raw event count (all envelope-threshold crossings within duration range)
    out["n_events_valid"] = len(events)
    if len(events) > 0:
        durs = np.array([(e - s) / fs for s, e in events])
        out["mean_event_dur"] = float(durs.mean())

        # v6-preview: filter events by peak-inside-event check
        verified = [
            (s, e) for s, e in events
            if verify_peak_inside_event(r_thal, s, e, fs)
        ]
        out["n_events_verified"] = len(verified)
    out["n_events_raw"] = out["n_events_valid"]   # alias

    # PAC MI (cortex SO phase × thalamic spindle envelope)
    out["pac_mi"] = compute_pac_mi(r_ctx, r_thal, fs)

    return out


# =====================================================================
# Main sweep loop
# =====================================================================

def main():
    print("=" * 70)
    print("Isolated Thalamus 3D Sweep")
    print("=" * 70)
    print(f"  Grid: g_LK × g_h × c_ctx2th = "
          f"{len(G_LK_GRID)} × {len(G_H_GRID)} × {len(C_CTX2TH_GRID)} "
          f"= {N_POINTS} points")
    print(f"  g_LK     ∈ [{G_LK_GRID.min():.3f}, {G_LK_GRID.max():.3f}] "
          f"(v5 bounds [0.040, 0.073])")
    print(f"  g_h      ∈ [{G_H_GRID.min():.3f}, {G_H_GRID.max():.3f}] "
          f"(v5 bounds [0.036, 0.067])")
    print(f"  c_ctx2th ∈ [{C_CTX2TH_GRID.min():.3f}, "
          f"{C_CTX2TH_GRID.max():.3f}] "
          f"(v5 bounds [0.027, 0.050] — EXPANDED 4x upward)")
    print(f"  Cortex c_th2ctx = {CORTEX_FIXED['c_th2ctx']} "
          f"(DISABLED → thalamus isolated)")
    print(f"  Cortex fixed:  mue={CORTEX_FIXED['mue']}, "
          f"mui={CORTEX_FIXED['mui']}, b={CORTEX_FIXED['b']}, "
          f"tauA={CORTEX_FIXED['tauA']}")
    print(f"  Sim duration: {SIM_DUR_MS/1000:.0f}s (burn-in {BURN_IN_S}s)")
    print(f"  Expected wall-clock: {N_POINTS * 15 / 3600:.1f} h "
          f"(single core, numba backend)")
    print("=" * 70)

    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Prepare output arrays
    nlk, nh, nc = len(G_LK_GRID), len(G_H_GRID), len(C_CTX2TH_GRID)
    arrays = {
        "sp_peak_power_log":  np.full((nlk, nh, nc), np.nan),
        "sp_env_cv":          np.full((nlk, nh, nc), np.nan),
        "n_events_valid":     np.full((nlk, nh, nc), np.nan),
        "n_events_verified":  np.full((nlk, nh, nc), np.nan),
        "pac_mi":             np.full((nlk, nh, nc), np.nan),
        "ctx_ibi_cv":         np.full((nlk, nh, nc), np.nan),
    }

    rows = []
    t_start = time.time()
    pt_idx = 0

    for i, g_lk in enumerate(G_LK_GRID):
        for j, g_h in enumerate(G_H_GRID):
            for k, c_ctx2th in enumerate(C_CTX2TH_GRID):
                pt_idx += 1
                t_pt = time.time()

                # Build and run model
                try:
                    m = build_sweep_model(g_lk, g_h, c_ctx2th)
                    m.run()
                except Exception as e:
                    # Fall back to jitcdde on numba failure
                    try:
                        m.params["backend"] = "jitcdde"
                        m.run()
                    except Exception:
                        row = dict(
                            i=i, j=j, k=k,
                            g_LK=g_lk, g_h=g_h, c_ctx2th=c_ctx2th,
                            ok=False, fail_reason=f"sim_failed: {e}",
                        )
                        rows.append(row)
                        continue

                # Extract firing rates
                try:
                    r_exc = m[f"r_mean_{EXC}"]
                    if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
                        r_ctx  = r_exc[0, :] * 1000.0
                        r_thal = r_exc[1, :] * 1000.0
                    else:
                        row = dict(i=i, j=j, k=k, g_LK=g_lk, g_h=g_h,
                                   c_ctx2th=c_ctx2th, ok=False,
                                   fail_reason="output_shape")
                        rows.append(row)
                        continue
                except Exception as e:
                    row = dict(i=i, j=j, k=k, g_LK=g_lk, g_h=g_h,
                               c_ctx2th=c_ctx2th, ok=False,
                               fail_reason=f"extract_failed: {e}")
                    rows.append(row)
                    continue

                # Burn-in
                n_drop = int(BURN_IN_S * FS_SIM)
                r_ctx  = r_ctx[n_drop:]
                r_thal = r_thal[n_drop:]

                # Score
                metrics = score_grid_point(r_ctx, r_thal, fs=FS_SIM)

                # Fill arrays (only for the "ok" metrics)
                arrays["sp_peak_power_log"][i, j, k] = metrics["sp_peak_power_log"]
                arrays["sp_env_cv"][i, j, k]         = metrics["sp_env_cv"]
                arrays["n_events_valid"][i, j, k]    = metrics["n_events_valid"]
                arrays["n_events_verified"][i, j, k] = metrics["n_events_verified"]
                arrays["pac_mi"][i, j, k]            = metrics["pac_mi"]
                arrays["ctx_ibi_cv"][i, j, k]        = metrics["ctx_ibi_cv"]

                row = dict(
                    i=i, j=j, k=k,
                    g_LK=g_lk, g_h=g_h, c_ctx2th=c_ctx2th,
                    **metrics,
                )
                rows.append(row)

                dt_pt = time.time() - t_pt
                elapsed = time.time() - t_start
                remaining = (N_POINTS - pt_idx) * (elapsed / pt_idx)

                # Print every 10 points and on notable hits
                if pt_idx % 10 == 0 or metrics["n_events_verified"] >= 3:
                    print(f"  [{pt_idx:4d}/{N_POINTS}] "
                          f"g_LK={g_lk:.3f} g_h={g_h:.3f} c_ctx2th={c_ctx2th:.3f}  "
                          f"| sp_log={metrics['sp_peak_power_log']:.2f}  "
                          f"n_ev={metrics['n_events_valid']:2d}  "
                          f"n_ver={metrics['n_events_verified']:2d}  "
                          f"MI={metrics['pac_mi']:.4f}  "
                          f"| {dt_pt:.1f}s/pt  ETA {remaining/60:.0f}m")

    total_time = time.time() - t_start
    print(f"\nSweep complete in {total_time/3600:.2f} h")

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved CSV: {OUT_CSV} ({len(df)} rows)")

    # Save NPZ of arrays for plotting
    np.savez(OUT_NPZ,
             g_LK_grid=G_LK_GRID,
             g_h_grid=G_H_GRID,
             c_ctx2th_grid=C_CTX2TH_GRID,
             **arrays)
    print(f"Saved NPZ: {OUT_NPZ}")

    # Quick summary
    print("\n" + "=" * 70)
    print("Sweep summary")
    print("=" * 70)
    print(f"  Total points:           {N_POINTS}")
    print(f"  Simulation failures:    "
          f"{(df['ok']==False).sum() if 'ok' in df.columns else 0}")
    print(f"  Points with sp_peak_log > 0.5:  "
          f"{(df.get('sp_peak_power_log', pd.Series([])) > 0.5).sum()}")
    print(f"  Points with n_events_verified ≥ 3:  "
          f"{(df.get('n_events_verified', pd.Series([])) >= 3).sum()}")
    print(f"  Points with pac_mi > 0.01:  "
          f"{(df.get('pac_mi', pd.Series([])) > 0.01).sum()}")

    if 'n_events_verified' in df.columns:
        top = df.sort_values('n_events_verified', ascending=False).head(10)
        print(f"\n  Top 10 points by n_events_verified:")
        cols_print = ['g_LK','g_h','c_ctx2th','sp_peak_power_log',
                      'n_events_valid','n_events_verified','pac_mi']
        print(top[cols_print].to_string(index=False))

    print(f"\nNext: python plot_scripts/plot_thalamus_sweep_heatmaps.py")


if __name__ == "__main__":
    main()
