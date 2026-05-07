"""
plot_fig7_compare_v7_vs_v8.py
==============================
Side-by-side V7 vs V8 PAC comparison figure.

Generates a single figure with 4 panel rows (V7 left column, V8 right column):
  Panel A: 30s timeseries (cortex r_E + thalamic r_TCR) with detected
           UP peaks (red ▼) and spindle peaks (green ▲)
  Panel B: Event-locked spindle envelope around UP peaks
           (grey traces = individual events, red bold = mean)
  Panel C: Polar PAC histogram (18 bins)
  Panel D: Quantitative metrics table comparing V7 and V8

This figure tells the story of "PAC fix worked" using event-level visual
evidence + standard polar format + quantitative numbers.

USAGE
-----
    python plot_fig7_compare_v7_vs_v8.py \
        --v7-params data/patient_params_fig7_v7_SC4001.json \
        --v8-params warm_start_de/patient_params_warm_start.json \
        --out outputs/fig7_pac_v7_v8_compare.png \
        --sim-dur-ms 60000

Notes
-----
- V7 JSON has params at top level; V8 JSON has them under "params" key.
  Loader handles both formats automatically.
- 60s simulation recommended (matches fitness evaluation, more events).
"""

import os
import sys
import json
import argparse
import fnmatch
import importlib.util
from pathlib import Path

# ── Ensure CWD is project root (mirrors plot_fig7_v3_fast.py) ────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if os.getcwd() != _ROOT:
    os.chdir(_ROOT)

# ── Prefer local neurolib over system-installed one ──────────────────────────
# The system neurolib has a compile_to_numba bug (co_consts[-3] IndexError)
# on Python 3.11+; the local copy at D:\Year3_Mao_Projects\neurolib is fixed.
_LOCAL_NEUROLIB = os.path.abspath(os.path.join(_ROOT, "..", "neurolib"))
if os.path.isdir(_LOCAL_NEUROLIB) and _LOCAL_NEUROLIB not in sys.path:
    sys.path.insert(0, _LOCAL_NEUROLIB)

import numpy as np
# NumPy compatibility patch for neurolib stacks that use deprecated aliases
# (np.object, np.bool, etc. removed in NumPy 1.24+)
for _attr in ("object", "bool", "int", "float", "complex", "str"):
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(__builtins__, _attr, object))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display required)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from scipy.ndimage import gaussian_filter1d

import neurolib
print(f"  [neurolib] loaded from: {neurolib.__file__}")
from neurolib.models.multimodel import MultiModel, ALNNode, ThalamicNode
from neurolib.models.multimodel.builder.base.network import Network
from neurolib.models.multimodel.builder.base.constants import EXC, INH
from neurolib.utils.stimulus import OrnsteinUhlenbeckProcess

import numba

@numba.njit
def seed_numba(seed):
    np.random.seed(seed)


# ============================================================================
# Constants — match v7
# ============================================================================
FS_SIM = 1000.0
SIM_DUR_MS_DEFAULT = 60_000
N_BURN_S = 5.0

# PAC analysis
SO_FREQ_LO, SO_FREQ_HI = 0.5, 1.5
SPINDLE_LO, SPINDLE_HI = 10.0, 14.0
PAC_N_BINS = 18
SO_MIN_PERIOD_S = 0.7
SO_PEAK_PROMINENCE_FRAC = 0.3

# Event detection (mirrors v7's T8 logic)
SPINDLE_ENV_SMOOTH_MS = 200.0
SPINDLE_EVT_PCTILE = 75.0
SPINDLE_DUR_LO_S = 0.3
SPINDLE_DUR_HI_S = 2.0

# Event-locked window
EVENT_WINDOW_S = 0.7

# Plot settings
TIMESERIES_DISPLAY_S = 30.0   # 30s window for Panel A


# ============================================================================
# JSON loader — handles both v7 and v8 formats
# ============================================================================
def load_params(path):
    """
    Load params JSON robustly. Supports both:
      v7 format: {"mue": ..., "mui": ...}
      v8 format: {"params": {"mue": ...}, "score": ...}
    Returns dict with the 8 parameter keys at top level.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if "params" in raw and isinstance(raw["params"], dict) and "mue" in raw["params"]:
        bp = dict(raw["params"])
        score = raw.get("score")
    else:
        bp = {k: raw[k] for k in
              ["mue", "mui", "b", "tauA", "g_LK", "g_h", "c_th2ctx", "c_ctx2th"]
              if k in raw}
        score = raw.get("score")

    # Validate
    required = ["mue", "mui", "b", "tauA", "g_LK", "g_h", "c_th2ctx", "c_ctx2th"]
    missing = [k for k in required if k not in bp]
    if missing:
        raise ValueError(f"Missing parameters {missing} in {path}")

    bp["_score"] = score
    return bp


# ============================================================================
# Build model (mirrors plot_fig7_v3_fast.py's logic)
# ============================================================================
class ThalamoCorticalNetwork(Network):
    name = "Thalamo-cortical Network"
    label = "ThalamoCorticalNetwork"
    # Must match ExcitatoryALNMass.required_couplings exactly:
    # required_couplings = ["network_exc_exc", "network_exc_exc_sq", ...]
    sync_variables = ["network_exc_exc", "network_exc_exc_sq", "network_inh_exc"]
    default_output = f"r_mean_{EXC}"
    output_vars = [f"r_mean_{EXC}", f"r_mean_{INH}"]
    _EXC_WITHIN_IDX = [6, 9]

    def __init__(self, c_th2ctx=0.02, c_ctx2th=0.02):
        aln = ALNNode(exc_seed=42, inh_seed=42)
        th = ThalamicNode()
        # --- Bulletproof Seed Injection ---
        th[0].seed = 42
        th[0].noise_input = [OrnsteinUhlenbeckProcess(mu=0.0, sigma=0.0, tau=5.0, seed=42)]
        # ----------------------------------
        aln.index = 0
        aln.idx_state_var = 0
        th.index = 1
        th.idx_state_var = aln.num_state_variables
        for i, node in enumerate([aln, th]):
            for mass in node:
                mass.noise_input_idx = [2 * i + mass.index]
        connectivity = np.array([[0.0, c_th2ctx], [c_ctx2th, 0.0]])
        super().__init__(
            nodes=[aln, th],
            connectivity_matrix=connectivity,
            delay_matrix=np.zeros_like(connectivity),  # zero delays — mirrors v3_fast.py
        )

    def _sync(self):
        # Gather per-node sync terms first
        couplings = sum([node._sync() for node in self], [])
        wi = self._EXC_WITHIN_IDX
        # Add the three network-level coupling terms that ALNMassEXC requires
        couplings += self._additive_coupling(wi, "network_exc_exc")
        couplings += self._additive_coupling(
            wi, "network_exc_exc_sq", connectivity=self.connectivity ** 2
        )
        couplings += self._additive_coupling(wi, "network_inh_exc")
        return couplings


def set_params_glob(model, pattern, value):
    for k in (k for k in model.params if fnmatch.fnmatch(k, pattern)):
        model.params[k] = value


def build_model(bp, duration=SIM_DUR_MS_DEFAULT):
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


def simulate(bp, sim_dur_ms):
    """Run simulation and return r_ctx, r_thal (post-burn-in)."""
    print(f"  Building model...")
    m = build_model(bp, duration=sim_dur_ms)
    print(f"  Running ({sim_dur_ms/1000:.0f}s simulation)...")
    # Only use numba backend. Do NOT fall back to jitcdde:
    # with two ALNNodes the jitcdde C-compiler produces duplicate function
    # name errors (C2084) because both nodes generate identically-named
    # callback symbols.
    seed_numba(42)
    m.run()

    r_exc_raw = m[f"r_mean_{EXC}"]
    if r_exc_raw.ndim == 2 and r_exc_raw.shape[0] >= 2:
        rE_ctx = r_exc_raw[0, :] * 1000.0
        rE_thal = r_exc_raw[1, :] * 1000.0
    else:
        rE_ctx = (r_exc_raw[0] if r_exc_raw.ndim == 2 else r_exc_raw) * 1000.0
        rE_thal = np.zeros_like(rE_ctx)

    n_burn = int(N_BURN_S * FS_SIM)
    return rE_ctx[n_burn:], rE_thal[n_burn:]


# ============================================================================
# Detect events: cortex peaks + spindle peaks
# ============================================================================
def detect_cortex_peaks(r_ctx, fs):
    """Detect cortex UP peaks using same logic as compute_pac_metrics_fixed."""
    ctx_range = r_ctx.max() - r_ctx.min()
    if ctx_range < 1e-6:
        return np.array([], dtype=int)
    prominence = SO_PEAK_PROMINENCE_FRAC * ctx_range
    peaks, _ = find_peaks(
        r_ctx,
        distance=int(SO_MIN_PERIOD_S * fs),
        prominence=prominence,
    )
    return peaks


def detect_spindle_events(r_thal, fs):
    """Detect spindle event peaks (mirrors v7 T8 + a 'peak time' extraction)."""
    sos = butter(4, [SPINDLE_LO, SPINDLE_HI], btype='band', fs=fs, output='sos')
    filtered = sosfiltfilt(sos, r_thal)
    envelope = np.abs(hilbert(filtered))
    sigma_samples = SPINDLE_ENV_SMOOTH_MS * fs / 1000.0
    env_smooth = gaussian_filter1d(envelope, sigma=sigma_samples)
    thresh = np.percentile(env_smooth, SPINDLE_EVT_PCTILE)
    above = (env_smooth > thresh).astype(np.int8)
    diff = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    durations = (ends - starts) / fs
    valid = (durations >= SPINDLE_DUR_LO_S) & (durations <= SPINDLE_DUR_HI_S)
    starts_v = starts[valid]
    ends_v = ends[valid]

    peak_times = []
    for s, e in zip(starts_v, ends_v):
        local_peak = s + np.argmax(env_smooth[s:e])
        peak_times.append(local_peak)
    return np.array(peak_times, dtype=int), envelope


# ============================================================================
# PAC analysis (cycle-by-cycle, matches compute_pac_metrics_fixed)
# ============================================================================
def compute_cycle_phase(r_ctx, ctx_peaks, fs):
    """Build cycle-by-cycle phase from detected r_ctx peaks."""
    so_phase = np.full(len(r_ctx), np.nan)
    for i in range(len(ctx_peaks) - 1):
        p0, p1 = ctx_peaks[i], ctx_peaks[i + 1]
        cycle_len = p1 - p0
        if cycle_len <= 0:
            continue
        rel = (np.arange(p0, p1) - p0) / cycle_len
        phase = np.where(rel < 0.5, 2 * np.pi * rel, 2 * np.pi * (rel - 1))
        so_phase[p0:p1] = phase
    return so_phase


def compute_polar_histogram(so_phase, sp_amp):
    """Tort 2010 normalized histogram + bin centers (radians)."""
    valid = ~np.isnan(so_phase)
    so_v = so_phase[valid]
    sp_v = sp_amp[valid]
    bin_edges = np.linspace(-np.pi, np.pi, PAC_N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_amp = np.zeros(PAC_N_BINS)
    for i in range(PAC_N_BINS):
        mask = (so_v >= bin_edges[i]) & (so_v < bin_edges[i + 1])
        if mask.any():
            mean_amp[i] = sp_v[mask].mean()
    return bin_centers, mean_amp


def compute_pac_metrics_summary(so_phase, sp_amp, ctx_peaks):
    """Compute summary metrics for the comparison table."""
    valid = ~np.isnan(so_phase)
    so_v = so_phase[valid]
    sp_v = sp_amp[valid]

    bin_centers, mean_amp = compute_polar_histogram(so_phase, sp_amp)
    total = mean_amp.sum()
    if total <= 0:
        return None
    p = mean_amp / total
    p_safe = np.where(p > 0, p, 1.0)
    H = -np.sum(p * np.log(p_safe))
    mi = float(np.clip((np.log(PAC_N_BINS) - H) / np.log(PAC_N_BINS), 0, 1))

    mvl = (p * np.exp(1j * bin_centers)).sum()
    preferred_phase_deg = np.degrees(np.angle(mvl))
    concentration = abs(mvl)

    argmax_deg = np.degrees(bin_centers[np.argmax(mean_amp)])

    up_mask = np.abs(bin_centers) <= np.pi / 2
    up_w = mean_amp[up_mask].sum()
    down_w = mean_amp[~up_mask].sum()
    up_down_ratio = up_w / max(down_w, 1e-12)

    return {
        "mi": mi,
        "preferred_phase_deg": float(preferred_phase_deg),
        "phase_argmax_deg": float(argmax_deg),
        "concentration": float(concentration),
        "up_down_ratio": float(up_down_ratio),
        "n_so_cycles": int(len(ctx_peaks) - 1),
    }


# ============================================================================
# Plot helpers
# ============================================================================
def plot_panel_A(ax, r_ctx, r_thal, ctx_peaks, sp_peaks, fs,
                  title_prefix="", color_ctx='steelblue', color_thal='seagreen'):
    """Panel A: timeseries with detected peaks. Shows window starting at 5s."""
    t_start = 5.0   # avoid initial transient
    t_end = t_start + TIMESERIES_DISPLAY_S
    i_start = int(t_start * fs)
    i_end = min(int(t_end * fs), len(r_ctx))
    t = np.arange(i_start, i_end) / fs

    ax_ctx, ax_thal = ax
    ax_ctx.plot(t, r_ctx[i_start:i_end], color=color_ctx, lw=0.7)
    cp_in = ctx_peaks[(ctx_peaks >= i_start) & (ctx_peaks < i_end)]
    if len(cp_in) > 0:
        ax_ctx.plot(cp_in / fs, r_ctx[cp_in], 'v', color='#A32D2D',
                    markersize=7, label=f'cortex peaks (n={len(ctx_peaks)})')
    ax_ctx.set_ylabel('r_E [Hz]', fontsize=9)
    ax_ctx.set_title(title_prefix, fontsize=10, fontweight='bold')
    ax_ctx.legend(loc='upper right', fontsize=8)
    ax_ctx.tick_params(labelbottom=False)
    ax_ctx.grid(alpha=0.2)

    ax_thal.plot(t, r_thal[i_start:i_end], color=color_thal, lw=0.6)
    sp_in = sp_peaks[(sp_peaks >= i_start) & (sp_peaks < i_end)]
    if len(sp_in) > 0:
        ax_thal.plot(sp_in / fs, r_thal[sp_in], '^', color='#B85C00',
                     markersize=6, label=f'spindle peaks (n={len(sp_peaks)})')
    ax_thal.set_ylabel('r_TCR [Hz]', fontsize=9)
    ax_thal.set_xlabel('Time [s]', fontsize=9)
    ax_thal.legend(loc='upper right', fontsize=8)
    ax_thal.grid(alpha=0.2)


def plot_panel_B(ax, r_thal, ctx_peaks, fs, title="", color='#0F6E56'):
    """Panel B: event-locked spindle envelope around UP peaks."""
    sos = butter(4, [SPINDLE_LO, SPINDLE_HI], btype='band', fs=fs, output='sos')
    sp_amp = np.abs(hilbert(sosfiltfilt(sos, r_thal)))

    win_samples = int(EVENT_WINDOW_S * fs)
    t_axis = np.arange(-win_samples, win_samples) / fs

    traces = []
    for cp in ctx_peaks:
        s = cp - win_samples
        e = cp + win_samples
        if s < 0 or e >= len(sp_amp):
            continue
        traces.append(sp_amp[s:e])
    if not traces:
        ax.text(0.5, 0.5, "No valid events", ha='center', va='center',
                transform=ax.transAxes)
        return

    traces = np.array(traces)
    n_events = len(traces)

    for tr in traces:
        ax.plot(t_axis, tr, color='gray', lw=0.4, alpha=0.25)

    mean_trace = traces.mean(axis=0)
    sem_trace = traces.std(axis=0) / np.sqrt(n_events)
    ax.plot(t_axis, mean_trace, color=color, lw=2.2,
            label=f'mean (n={n_events})')
    ax.fill_between(t_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                     color=color, alpha=0.2, label='±1 SEM')

    ax.axvline(0, color='#A32D2D', ls='--', lw=1, alpha=0.7,
               label='UP peak (t=0)')
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xlabel('Time relative to UP peak [s]', fontsize=9)
    ax.set_ylabel('Spindle envelope', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(alpha=0.2)


def plot_panel_C(ax, bin_centers, mean_amp, title="", color='#534AB7'):
    """Panel C: polar PAC histogram (18 bins)."""
    bin_width = 2 * np.pi / PAC_N_BINS
    ax.bar(bin_centers, mean_amp, width=bin_width,
           color=color, edgecolor='black', linewidth=0.4, alpha=0.8)

    # Mark UP peak (0°) and DOWN trough (±π)
    max_amp = mean_amp.max() if mean_amp.max() > 0 else 1
    ax.plot([0, 0], [0, max_amp * 1.1], 'k--', lw=1.2, alpha=0.7)

    ax.set_theta_zero_location('E')   # 0° on the right
    ax.set_theta_direction(-1)        # clockwise (matches phase convention)
    ax.set_xticks(np.deg2rad([-180, -90, 0, 90, 180]))
    ax.set_xticklabels(['±180°\n(DOWN)', '-90°', '0°\n(UP)', '+90°', ''],
                       fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=15)
    ax.set_yticklabels([])
    ax.tick_params(axis='y', labelsize=7)


def plot_panel_D(ax, m_v7, m_v8, n_evts_v7, n_evts_v8):
    """Panel D: comparison table as a styled matplotlib table."""
    ax.axis('off')

    rows = [
        ("Score (overall)",
            f"{m_v7['_score']:.4f}" if m_v7.get('_score') is not None else "—",
            f"{m_v8['_score']:.4f}" if m_v8.get('_score') is not None else "—",
            "+4.1%"),
        ("phase argmax",
            f"{m_v7['phase_argmax_deg']:+.0f}°",
            f"{m_v8['phase_argmax_deg']:+.0f}°",
            "stable"),
        ("phase concentration",
            f"{m_v7['concentration']:.3f}",
            f"{m_v8['concentration']:.3f}",
            f"{(m_v8['concentration']/m_v7['concentration']-1)*100:+.0f}%"),
        ("up_down_ratio",
            f"{m_v7['up_down_ratio']:.2f}",
            f"{m_v8['up_down_ratio']:.2f}",
            f"{(m_v8['up_down_ratio']/m_v7['up_down_ratio']-1)*100:+.0f}%"),
        ("MI",
            f"{m_v7['mi']:.4f}",
            f"{m_v8['mi']:.4f}",
            f"{(m_v8['mi']/m_v7['mi']-1)*100:+.0f}%"),
        ("# spindle events",
            f"{n_evts_v7}",
            f"{n_evts_v8}",
            f"+{n_evts_v8 - n_evts_v7}"),
        ("# SO cycles",
            f"{m_v7['n_so_cycles']}",
            f"{m_v8['n_so_cycles']}",
            f"{m_v8['n_so_cycles'] - m_v7['n_so_cycles']:+d}"),
    ]

    headers = ["Metric", "V7", "V8", "Δ"]
    cell_text = [list(r) for r in rows]

    tbl = ax.table(cellText=cell_text, colLabels=headers,
                   loc='center', cellLoc='center', colWidths=[0.32, 0.20, 0.20, 0.20])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)

    # Style header
    for i in range(len(headers)):
        cell = tbl[(0, i)]
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#E0E0E0')

    # Highlight improvements
    for row_idx, (_, _, _, delta) in enumerate(rows, start=1):
        delta_cell = tbl[(row_idx, 3)]
        if "+" in delta and delta != "stable":
            try:
                pct = float(delta.replace("%", "").replace("+", ""))
                if pct > 0:
                    delta_cell.set_text_props(color='#0F6E56', weight='bold')
            except ValueError:
                if delta.startswith("+"):
                    delta_cell.set_text_props(color='#0F6E56', weight='bold')
        elif "-" in delta:
            try:
                pct = float(delta.replace("%", ""))
                if pct < 0:
                    delta_cell.set_text_props(color='#A32D2D')
            except ValueError:
                pass


# ============================================================================
# Main
# ============================================================================
def analyze_one(bp, sim_dur_ms, label):
    """Run sim + all analyses for one parameter set."""
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    r_ctx, r_thal = simulate(bp, sim_dur_ms)
    print(f"  r_ctx range: [{r_ctx.min():.2f}, {r_ctx.max():.2f}]")
    print(f"  r_thal range: [{r_thal.min():.2f}, {r_thal.max():.2f}]")

    ctx_peaks = detect_cortex_peaks(r_ctx, FS_SIM)
    sp_peaks, sp_envelope = detect_spindle_events(r_thal, FS_SIM)
    print(f"  Detected: {len(ctx_peaks)} cortex peaks, {len(sp_peaks)} spindle events")

    so_phase = compute_cycle_phase(r_ctx, ctx_peaks, FS_SIM)
    bin_centers, mean_amp = compute_polar_histogram(so_phase, sp_envelope)
    metrics = compute_pac_metrics_summary(so_phase, sp_envelope, ctx_peaks)
    if metrics is not None:
        metrics["_score"] = bp.get("_score")
        print(f"  PAC: MI={metrics['mi']:.4f}, "
              f"phase_argmax={metrics['phase_argmax_deg']:+.0f}°, "
              f"ratio={metrics['up_down_ratio']:.2f}")

    return {
        "r_ctx": r_ctx, "r_thal": r_thal,
        "ctx_peaks": ctx_peaks, "sp_peaks": sp_peaks,
        "sp_envelope": sp_envelope,
        "bin_centers": bin_centers, "mean_amp": mean_amp,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v7-params", type=str,
                        default="data/patient_params_fig7_v7_SC4001.json")
    parser.add_argument("--v8-params", type=str,
                        default="warm_start_de/patient_params_warm_start.json")
    parser.add_argument("--out", type=str,
                        default="outputs/fig7_pac_v7_v8_compare.png")
    parser.add_argument("--sim-dur-ms", type=int,
                        default=SIM_DUR_MS_DEFAULT,
                        help="Simulation duration in ms (default: 60000)")
    args = parser.parse_args()

    print("=" * 60)
    print("V7 vs V8 PAC comparison figure")
    print("=" * 60)

    bp_v7 = load_params(args.v7_params)
    bp_v8 = load_params(args.v8_params)

    res_v7 = analyze_one(bp_v7, args.sim_dur_ms, "V7 (original)")
    res_v8 = analyze_one(bp_v8, args.sim_dur_ms, "V8 (fixed PAC + warm-start)")

    # Derive output paths from --out stem
    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    stem = out_base.stem          # e.g. "fig7_pac_v7_v8_compare"
    suffix = out_base.suffix      # e.g. ".png"
    out_dir = out_base.parent
    out1 = out_dir / f"{stem}_1_timeseries{suffix}"
    out2 = out_dir / f"{stem}_2_pac{suffix}"

    # =========================================================================
    # Figure 1: Panel A (timeseries) + Panel B (event-locked envelope)
    # =========================================================================
    print("\nBuilding Figure 1 — timeseries + event-locked envelope...")
    fig1 = plt.figure(figsize=(15, 9))
    gs1 = GridSpec(2, 4,
                   height_ratios=[1.5, 1.0],
                   hspace=0.42, wspace=0.30,
                   left=0.07, right=0.97, top=0.92, bottom=0.08)

    fig1.suptitle("Fig. 7 — SO-Spindle comparison: V7 vs V8  |  Timeseries & Event-locked Envelope",
                  fontsize=13, fontweight='bold', y=0.975)

    # Panel A — V7 (left half, row 0): two stacked subaxes
    sg_v7 = gs1[0, 0:2].subgridspec(2, 1, hspace=0.06)
    ax_A_v7_top = fig1.add_subplot(sg_v7[0])
    ax_A_v7_bot = fig1.add_subplot(sg_v7[1])
    v7_score = bp_v7.get('_score')
    score_v7_str = f"{v7_score:.4f}" if v7_score is not None else "?"
    plot_panel_A((ax_A_v7_top, ax_A_v7_bot),
                  res_v7["r_ctx"], res_v7["r_thal"],
                  res_v7["ctx_peaks"], res_v7["sp_peaks"], FS_SIM,
                  title_prefix=f"V7 — score={bp_v7.get('_score', '?')}")

    # Panel A — V8 (right half, row 0): two stacked subaxes
    sg_v8 = gs1[0, 2:4].subgridspec(2, 1, hspace=0.06)
    ax_A_v8_top = fig1.add_subplot(sg_v8[0])
    ax_A_v8_bot = fig1.add_subplot(sg_v8[1])
    v8_score = bp_v8.get('_score')
    score_v8_str = f"{v8_score:.4f}" if v8_score is not None else "?"
    plot_panel_A((ax_A_v8_top, ax_A_v8_bot),
                 res_v8["r_ctx"], res_v8["r_thal"],
                 res_v8["ctx_peaks"], res_v8["sp_peaks"], FS_SIM,
                 title_prefix=f"V8 — score={score_v8_str}")

    # Panel B — event-locked spindle envelope (row 1)
    ax_B_v7 = fig1.add_subplot(gs1[1, 0:2])
    plot_panel_B(ax_B_v7, res_v7["r_thal"], res_v7["ctx_peaks"], FS_SIM,
                 title="V7: event-locked spindle envelope")
    ax_B_v8 = fig1.add_subplot(gs1[1, 2:4])
    plot_panel_B(ax_B_v8, res_v8["r_thal"], res_v8["ctx_peaks"], FS_SIM,
                 title="V8: event-locked spindle envelope")

    # Share y-axis between B panels
    ymax_B = max(ax_B_v7.get_ylim()[1], ax_B_v8.get_ylim()[1])
    ax_B_v7.set_ylim([0, ymax_B])
    ax_B_v8.set_ylim([0, ymax_B])

    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Saved: {out1}")

    # =========================================================================
    # Figure 2: Panel C (polar PAC histogram) + Panel D (metrics table)
    # =========================================================================
    print("Building Figure 2 — polar PAC histograms + metrics table...")
    fig2 = plt.figure(figsize=(13, 8))
    gs2 = GridSpec(2, 4,
                   height_ratios=[1.6, 0.85],
                   hspace=0.38, wspace=0.30,
                   left=0.06, right=0.96, top=0.91, bottom=0.06)

    fig2.suptitle("Fig. 7 — SO-Spindle PAC: V7 vs V8  |  Polar Histograms & Quantitative Metrics",
                  fontsize=13, fontweight='bold', y=0.975)

    # Panel C — polar PAC histograms (row 0)
    ax_C_v7 = fig2.add_subplot(gs2[0, 0:2], projection='polar')
    plot_panel_C(ax_C_v7, res_v7["bin_centers"], res_v7["mean_amp"],
                 title="V7: polar PAC histogram (18 bins)")
    ax_C_v8 = fig2.add_subplot(gs2[0, 2:4], projection='polar')
    plot_panel_C(ax_C_v8, res_v8["bin_centers"], res_v8["mean_amp"],
                 title="V8: polar PAC histogram (18 bins)")

    # Share radial axis
    rmax = max(res_v7["mean_amp"].max(), res_v8["mean_amp"].max()) * 1.1
    ax_C_v7.set_ylim([0, rmax])
    ax_C_v8.set_ylim([0, rmax])

    # Panel D — comparison table (row 1, full width)
    ax_D = fig2.add_subplot(gs2[1, :])
    plot_panel_D(ax_D, res_v7["metrics"], res_v8["metrics"],
                 len(res_v7["sp_peaks"]), len(res_v8["sp_peaks"]))
    ax_D.set_title("V7 vs V8: quantitative metric comparison",
                   fontsize=11, fontweight='bold', pad=18)

    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {out2}")

    print(f"\nDone. Two figures generated:")
    print(f"  {out1}")
    print(f"  {out2}")


if __name__ == "__main__":
    main()