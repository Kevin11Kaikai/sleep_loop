"""
s4_0d_scan_c_th2ctx_sweetspot.py
=================================
Find the c_th2ctx "sweet spot" and decide SIM_DUR_MS for v7.

Rationale (from v6 hotspot diagnosis, s4_0c):
  - Diagnosis revealed that c_th2ctx=0 (isolated) gives healthy cortex SO
    (T4_q ~3.5, T6 pass) BUT spindle_power=0 (FOOOF can't see spindle
    without thalamo-cortical feedback).
  - c_th2ctx=0.10 (coupled) gives spindle_power > 0.5 (FOOOF detects
    fed-back σ activity) BUT destroys cortex SO (T4_q < 1, T6 fail).
  - The two extremes frame a question: is there a value in between
    where both coexist?

  Separately, all 8 diagnostic points had so_power=0 despite T4_q > 3.
  Hypothesis: 30s simulation gives only ~24 SO cycles, too few for FOOOF
  to reliably fit the 1/f aperiodic baseline at low frequencies. Testing
  60s would answer this.

Design:
  4 spindle hotspot points × 5 c_th2ctx values × 2 simulation durations
  = 40 simulations (~10 minutes total).

  c_th2ctx grid: [0.00, 0.025, 0.05, 0.075, 0.10]
    - 0.00 and 0.10 reproduce s4_0c boundary conditions
    - 0.025, 0.05, 0.075 search for the sweet spot
  Duration grid: [30s, 60s]
    - 30s matches v6's SIM_DUR_MS
    - 60s matches verify script; tests FOOOF SO detection hypothesis

  Cortex params fixed at V4_best (same as s4_0c).

Sweet spot criteria (automatic report):
  ✓ T4 pass: SO Q-factor > 2
  ✓ T6 pass: SO IBI_CV < 0.4
  ✓ spindle_power > 0.1  (real σ peak detectable on cortex PSD)

  Bonus checks:
  * so_power > 0.1 (if yes, 60s helps FOOOF detect SO)
  * T5 can stay >0 but is no longer required (paper's T5 fix)

Output:
  outputs/c_th2ctx_scan.csv              — full results table
  outputs/c_th2ctx_scan_heatmap.png      — 4 heatmaps (T4_q, T6_CV, so_P, sp_P)
  outputs/c_th2ctx_scan_summary.txt      — text summary with sweet-spot report

Usage:
  python models/s4_0d_scan_c_th2ctx_sweetspot.py
"""

import os
import sys
import time
import fnmatch
import importlib.util
import warnings
warnings.filterwarnings("ignore")

import numpy as np
for _attr in ("object", "bool", "int", "float", "complex", "str"):
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(__builtins__, _attr, object))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import hilbert, butter, sosfiltfilt, welch
from scipy.ndimage import gaussian_filter1d

from neurolib.models.multimodel import MultiModel, ALNNode, ThalamicNode
from neurolib.models.multimodel.builder.base.network import Network
from neurolib.models.multimodel.builder.base.constants import EXC, INH

try:
    from fooof import FOOOF
    HAS_FOOOF = True
except ImportError:
    HAS_FOOOF = False
    print("[warn] fooof not installed — so_power/spindle_power unavailable")

# Load project's PSD utility
_spec = importlib.util.spec_from_file_location(
    "02_preprocess_psd", "utils/02_preprocess_psd.py"
)
prep_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep_mod)
compute_epoch_psd = prep_mod.compute_epoch_psd


# =====================================================================
# CONFIG
# =====================================================================

FS_SIM    = 1000.0
BURN_IN_S = 5.0
F_LO, F_HI = 0.5, 20.0

CORTEX_V4_BEST = dict(mue=3.895, mui=3.027, b=35.5, tauA=1426.0)

# 4 known-good spindle hotspots (from sweep + verify)
TEST_POINTS = [
    dict(label="P1_primary", g_LK=0.043, g_h=0.066, c_ctx2th=0.155),
    dict(label="P2_control", g_LK=0.049, g_h=0.066, c_ctx2th=0.020),
    dict(label="P3_top_sp",  g_LK=0.037, g_h=0.066, c_ctx2th=0.200),
    dict(label="P4_top_mi",  g_LK=0.043, g_h=0.066, c_ctx2th=0.200),
]

# c_th2ctx sweet-spot grid
C_TH2CTX_GRID = [0.00, 0.025, 0.05, 0.075, 0.10]

# Duration comparison
DUR_GRID_MS = [30_000, 60_000]

# Constraint thresholds (same as v6)
SO_FREQ_LO, SO_FREQ_HI = 0.2, 1.5
SO_Q_MIN = 2.0
SPINDLE_LO, SPINDLE_HI = 10.0, 14.0
IBI_CV_MAX = 0.4
UP_THRESH_HZ = 15.0

EVO_FOOOF_PARAMS = dict(
    peak_width_limits=[1.0, 8.0],
    max_n_peaks=4,
    min_peak_height=0.05,
    aperiodic_mode="fixed",
)

OUT_CSV  = "outputs/c_th2ctx_scan.csv"
OUT_PNG  = "outputs/c_th2ctx_scan_heatmap.png"
OUT_TXT  = "outputs/c_th2ctx_scan_summary.txt"


# =====================================================================
# Model (identical to v6)
# =====================================================================

def set_params_glob(model, pattern, value):
    for k in (k for k in model.params if fnmatch.fnmatch(k, pattern)):
        model.params[k] = value


class ThalamoCorticalNetwork(Network):
    name           = "Thalamocortical Motif (c_th2ctx scan)"
    label          = "TCNet_scan"
    sync_variables = ["network_exc_exc", "network_exc_exc_sq", "network_inh_exc"]
    default_output = f"r_mean_{EXC}"
    output_vars    = [f"r_mean_{EXC}", f"r_mean_{INH}"]
    _EXC_WITHIN_IDX = [6, 9]

    def __init__(self, c_th2ctx=0.0, c_ctx2th=0.04):
        aln = ALNNode(exc_seed=42, inh_seed=42)
        th  = ThalamicNode()
        aln.index = 0; aln.idx_state_var = 0
        th.index  = 1; th.idx_state_var  = aln.num_state_variables
        for i, node in enumerate([aln, th]):
            for mass in node:
                mass.noise_input_idx = [2 * i + mass.index]
        connectivity = np.array([[0.0, c_th2ctx], [c_ctx2th, 0.0]])
        super().__init__(nodes=[aln, th], connectivity_matrix=connectivity,
                         delay_matrix=np.zeros_like(connectivity))

    def _sync(self):
        couplings = sum([node._sync() for node in self], [])
        wi = self._EXC_WITHIN_IDX
        couplings += self._additive_coupling(wi, "network_exc_exc")
        couplings += self._additive_coupling(
            wi, "network_exc_exc_sq", connectivity=self.connectivity ** 2
        )
        couplings += self._additive_coupling(wi, "network_inh_exc")
        return couplings


def build_model(g_lk, g_h, c_th2ctx, c_ctx2th, dur_ms):
    net = ThalamoCorticalNetwork(c_th2ctx=c_th2ctx, c_ctx2th=c_ctx2th)
    m = MultiModel(net)
    m.params["backend"]     = "numba"
    m.params["dt"]          = 0.1
    m.params["sampling_dt"] = 1.0
    m.params["duration"]    = dur_ms
    set_params_glob(m, "*ALNMassEXC*.input_0.mu", CORTEX_V4_BEST["mue"])
    set_params_glob(m, "*ALNMassINH*.input_0.mu", CORTEX_V4_BEST["mui"])
    set_params_glob(m, "*ALNMassEXC*.b",          CORTEX_V4_BEST["b"])
    set_params_glob(m, "*ALNMassEXC*.tauA",       CORTEX_V4_BEST["tauA"])
    set_params_glob(m, "*ALNMassEXC*.a",          0.0)
    set_params_glob(m, "*ALNMass*.input_0.sigma", 0.05)
    set_params_glob(m, "*TCR*.input_0.sigma",     0.005)
    set_params_glob(m, "*.input_0.tau",           5.0)
    set_params_glob(m, "*TRN*.g_LK",              0.1)
    set_params_glob(m, "*TCR*.g_LK",              g_lk)
    set_params_glob(m, "*TCR*.g_h",               g_h)
    return m


# =====================================================================
# Lean metrics (just what we need for the sweet-spot decision)
# =====================================================================

def analyze_run(r_ctx, r_thal, fs=FS_SIM):
    """Extract T4_q, T6_ibi_cv, T4_freq, FOOOF so_power & spindle_power."""
    out = dict(T4_q=0.0, T4_freq=0.0, T6_ibi_cv=999.0, T6_n_bursts=0,
               so_power=0.0, spindle_power=0.0, fooof_ok=False)

    # T4+ SO Q-factor
    try:
        f_c, p_c = compute_epoch_psd(r_ctx, fs)
        so_mask = (f_c >= SO_FREQ_LO) & (f_c <= SO_FREQ_HI)
        so_width = SO_FREQ_HI - SO_FREQ_LO
        nlo = (f_c >= max(0.1, SO_FREQ_LO - so_width)) & (f_c < SO_FREQ_LO)
        nhi = (f_c > SO_FREQ_HI) & (f_c <= SO_FREQ_HI + so_width)
        if so_mask.any():
            out['T4_freq'] = float(f_c[so_mask][np.argmax(p_c[so_mask])])
            peak = p_c[so_mask].max()
            nbrs = np.concatenate([
                p_c[nlo] if nlo.any() else np.array([]),
                p_c[nhi] if nhi.any() else np.array([])])
            if len(nbrs) > 0 and nbrs.mean() > 0:
                out['T4_q'] = float(peak / nbrs.mean())
    except Exception:
        pass

    # T6 IBI_CV
    try:
        above = (r_ctx > UP_THRESH_HZ).astype(np.int8)
        diff = np.diff(np.concatenate(([0], above, [0])))
        starts = np.where(diff == 1)[0]
        n_bursts = len(starts)
        out['T6_n_bursts'] = n_bursts
        if n_bursts >= 3:
            intervals = np.diff(starts) / fs
            out['T6_ibi_cv'] = float(intervals.std() /
                                     (intervals.mean() + 1e-12))
    except Exception:
        pass

    # FOOOF: so_power & spindle_power
    if HAS_FOOOF:
        try:
            f_c, p_c = compute_epoch_psd(r_ctx, fs)
            mask = (f_c >= F_LO) & (f_c <= F_HI)
            fm = FOOOF(**EVO_FOOOF_PARAMS)
            fm.fit(f_c[mask], p_c[mask], [F_LO, F_HI])
            for pk in fm.peak_params_:
                freq, power, _ = pk
                if SO_FREQ_LO <= freq <= SO_FREQ_HI:
                    out['so_power'] = max(out['so_power'], float(power))
                if SPINDLE_LO <= freq <= SPINDLE_HI:
                    out['spindle_power'] = max(out['spindle_power'],
                                               float(power))
            out['fooof_ok'] = True
        except Exception:
            pass

    return out


def run_point(pt, c_th2ctx, dur_ms):
    """Run one configuration and return metrics dict."""
    m = build_model(pt['g_LK'], pt['g_h'], c_th2ctx, pt['c_ctx2th'], dur_ms)
    try:
        m.run()
    except Exception:
        try:
            m.params["backend"] = "jitcdde"
            m.run()
        except Exception:
            return None
    r_exc = m[f"r_mean_{EXC}"]
    if r_exc.ndim != 2 or r_exc.shape[0] < 2:
        return None
    r_ctx  = r_exc[0, :] * 1000.0
    r_thal = r_exc[1, :] * 1000.0
    n_drop = int(BURN_IN_S * FS_SIM)
    return analyze_run(r_ctx[n_drop:], r_thal[n_drop:], fs=FS_SIM)


# =====================================================================
# Plot heatmaps
# =====================================================================

def plot_heatmaps(df):
    """One figure: 2 rows (durations) × 4 cols (metrics), each a heatmap."""
    durations = sorted(df['dur_ms'].unique())
    metrics = [
        ('T4_q',          'SO Q-factor',          'T4 threshold >2 (brighter=healthier)',  'viridis', 0, 5),
        ('T6_ibi_cv',     'SO IBI CV',            'T6 threshold <0.4 (DARKER=healthier)',  'viridis_r', 0, 1.2),
        ('so_power',      'FOOOF SO power',       'reward target >0.1',                     'plasma',  0, 1.0),
        ('spindle_power', 'FOOOF spindle power',  'reward target >0.1',                     'plasma',  0, 1.0),
    ]
    fig, axes = plt.subplots(len(durations), len(metrics),
                              figsize=(5 * len(metrics), 4 * len(durations)),
                              squeeze=False)
    point_labels = [p['label'] for p in TEST_POINTS]
    for di, dur in enumerate(durations):
        for mi, (col, title, subtitle, cmap, vmin, vmax) in enumerate(metrics):
            ax = axes[di][mi]
            sub = df[df['dur_ms'] == dur]
            # Build grid: rows = points, cols = c_th2ctx
            grid = np.zeros((len(TEST_POINTS), len(C_TH2CTX_GRID)))
            for pi, pt in enumerate(TEST_POINTS):
                for ci, cv in enumerate(C_TH2CTX_GRID):
                    row = sub[(sub['label'] == pt['label']) &
                              (np.isclose(sub['c_th2ctx'], cv))]
                    if len(row) > 0:
                        grid[pi, ci] = row[col].iloc[0]
            im = ax.imshow(grid, aspect='auto', cmap=cmap,
                          vmin=vmin, vmax=vmax, origin='upper')
            ax.set_xticks(range(len(C_TH2CTX_GRID)))
            ax.set_xticklabels([f"{c:.3f}" for c in C_TH2CTX_GRID],
                              fontsize=9)
            ax.set_yticks(range(len(TEST_POINTS)))
            ax.set_yticklabels(point_labels, fontsize=9)
            ax.set_xlabel("c_th2ctx", fontsize=10)
            ax.set_title(f"{title} ({dur/1000:.0f}s)\n{subtitle}",
                        fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # Annotate values
            for pi in range(len(TEST_POINTS)):
                for ci in range(len(C_TH2CTX_GRID)):
                    val = grid[pi, ci]
                    ax.text(ci, pi, f"{val:.2f}",
                           ha='center', va='center',
                           fontsize=8, color='white' if val < (vmin+vmax)/2 else 'black')
    fig.suptitle(
        "c_th2ctx sweet-spot scan — 4 spindle hotspots × 5 c_th2ctx × 2 durations\n"
        "Goal: find c_th2ctx where T4_q>2 AND T6_CV<0.4 AND spindle_power>0.1",
        fontsize=11, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUT_PNG}")


# =====================================================================
# Sweet-spot analysis
# =====================================================================

def analyze_sweet_spot(df, out_fh):
    """Find configurations that satisfy all 3 criteria."""
    lines = []
    lines.append("=" * 78)
    lines.append("SWEET-SPOT REPORT")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Criteria (all three must hold):")
    lines.append("  [a] T4_q > 2          (SO peak sharp)")
    lines.append("  [b] T6_ibi_cv < 0.4   (SO regularity)")
    lines.append("  [c] spindle_power > 0.1 (FOOOF detects σ peak on cortex PSD)")
    lines.append("")

    df['sweet'] = (
        (df['T4_q'] > SO_Q_MIN) &
        (df['T6_ibi_cv'] < IBI_CV_MAX) &
        (df['spindle_power'] > 0.1)
    )

    for dur in sorted(df['dur_ms'].unique()):
        sub = df[df['dur_ms'] == dur]
        sweet = sub[sub['sweet']]
        lines.append("-" * 78)
        lines.append(f"Duration = {dur/1000:.0f}s: "
                     f"{len(sweet)}/{len(sub)} configurations are sweet")
        lines.append("-" * 78)
        if len(sweet) == 0:
            lines.append("  NO sweet-spot configurations at this duration!")
            # Per-c_th2ctx breakdown of which criteria fail
            for cv in C_TH2CTX_GRID:
                slc = sub[np.isclose(sub['c_th2ctx'], cv)]
                if len(slc) == 0: continue
                na = (slc['T4_q'] > SO_Q_MIN).sum()
                nb = (slc['T6_ibi_cv'] < IBI_CV_MAX).sum()
                nc = (slc['spindle_power'] > 0.1).sum()
                lines.append(f"  c_th2ctx={cv:.3f}:  "
                             f"T4_pass={na}/4  T6_pass={nb}/4  sp_P>0.1={nc}/4")
        else:
            lines.append(f"  {'label':<12s} {'c_th2ctx':>9s}  {'T4_q':>6s}  "
                         f"{'T6_CV':>6s}  {'sp_P':>6s}  {'so_P':>6s}")
            for _, r in sweet.iterrows():
                lines.append(f"  {r['label']:<12s} "
                             f"{r['c_th2ctx']:>9.3f}  "
                             f"{r['T4_q']:>6.2f}  "
                             f"{r['T6_ibi_cv']:>6.3f}  "
                             f"{r['spindle_power']:>6.3f}  "
                             f"{r['so_power']:>6.3f}")
        lines.append("")

    # Duration comparison: does 60s help FOOOF detect SO?
    lines.append("-" * 78)
    lines.append("DURATION COMPARISON (does 60s help FOOOF?)")
    lines.append("-" * 78)
    lines.append("")
    durations = sorted(df['dur_ms'].unique())
    if len(durations) == 2:
        d30, d60 = durations
        sp30 = df[df['dur_ms'] == d30].set_index(['label', 'c_th2ctx'])
        sp60 = df[df['dur_ms'] == d60].set_index(['label', 'c_th2ctx'])
        improved_so = 0; improved_sp = 0; total = 0
        for idx in sp30.index:
            if idx in sp60.index:
                total += 1
                if sp60.loc[idx, 'so_power'] > sp30.loc[idx, 'so_power'] + 0.05:
                    improved_so += 1
                if sp60.loc[idx, 'spindle_power'] > sp30.loc[idx, 'spindle_power'] + 0.05:
                    improved_sp += 1
        lines.append(f"  Out of {total} matched configurations:")
        lines.append(f"    so_power improved by >0.05:      {improved_so}/{total}")
        lines.append(f"    spindle_power improved by >0.05: {improved_sp}/{total}")

        n_so_30 = (df[df['dur_ms']==d30]['so_power'] > 0.1).sum()
        n_so_60 = (df[df['dur_ms']==d60]['so_power'] > 0.1).sum()
        lines.append(f"  Configs with so_power > 0.1:")
        lines.append(f"    at 30s: {n_so_30}/{len(sp30)}")
        lines.append(f"    at 60s: {n_so_60}/{len(sp60)}")
        lines.append("")
        if n_so_60 > n_so_30:
            lines.append("  → RECOMMENDATION: use 60s in v7 (FOOOF SO detection improved)")
        elif n_so_30 == n_so_60 == 0:
            lines.append("  → NEITHER duration detects SO via FOOOF. Reward must be replaced.")
        else:
            lines.append("  → 30s sufficient; FOOOF SO detection is not duration-limited")

    lines.append("")
    lines.append("=" * 78)
    lines.append("V7 DECISION FRAMEWORK")
    lines.append("=" * 78)
    sweet_60 = df[(df['dur_ms'] == DUR_GRID_MS[-1]) & (df['sweet'])]
    if len(sweet_60) >= 3:
        lines.append(f"  ✓ Sweet spot exists ({len(sweet_60)} configs at 60s)")
        min_c = sweet_60['c_th2ctx'].min()
        max_c = sweet_60['c_th2ctx'].max()
        lines.append(f"  → v7 c_th2ctx bounds: [{max(0.01, min_c-0.01):.3f}, "
                     f"{min(0.25, max_c+0.01):.3f}]")
        lines.append(f"  → Keep FOOOF spindle_power reward (it works here)")
    elif len(sweet_60) >= 1:
        lines.append(f"  ⚠ Sweet spot narrow ({len(sweet_60)} configs)")
        lines.append(f"  → v7 should use the sweet c_th2ctx values as tight bounds")
        lines.append(f"  → Consider replacing spindle_power reward with event-based")
    else:
        lines.append("  ✗ NO sweet spot found at either duration")
        lines.append("  → v7 must replace FOOOF spindle_power reward "
                     "with event-based metric (n_events_verified)")
        lines.append("  → c_th2ctx bounds can remain [0.00, 0.10] for PAC")
        lines.append("  → Accept that FOOOF σ peak is an unreachable target "
                     "for isolated-thalamus-optimal points")

    txt = "\n".join(lines)
    out_fh.write(txt + "\n")
    print(txt)


# =====================================================================
# Main
# =====================================================================

def main():
    n_sims = len(TEST_POINTS) * len(C_TH2CTX_GRID) * len(DUR_GRID_MS)
    print("=" * 78)
    print("c_th2ctx sweet-spot scan")
    print("=" * 78)
    print(f"  {len(TEST_POINTS)} points × {len(C_TH2CTX_GRID)} c_th2ctx "
          f"× {len(DUR_GRID_MS)} durations = {n_sims} simulations")
    print(f"  Expected wall time: ~{n_sims * 15 / 60:.1f} min")
    print(f"  Cortex: {CORTEX_V4_BEST}")
    print("=" * 78)

    os.makedirs("outputs", exist_ok=True)

    rows = []
    t_total = time.time()
    for pi, pt in enumerate(TEST_POINTS):
        for ci, cv in enumerate(C_TH2CTX_GRID):
            for di, dur in enumerate(DUR_GRID_MS):
                idx = (pi * len(C_TH2CTX_GRID) * len(DUR_GRID_MS)
                       + ci * len(DUR_GRID_MS) + di + 1)
                t0 = time.time()
                print(f"[{idx:2d}/{n_sims}] {pt['label']:<12s} "
                      f"c_th2ctx={cv:.3f}  dur={dur/1000:.0f}s ... ",
                      end="", flush=True)
                res = run_point(pt, cv, dur)
                dt = time.time() - t0
                if res is None:
                    print(f"FAILED ({dt:.1f}s)")
                    continue
                row = dict(label=pt['label'],
                           g_LK=pt['g_LK'], g_h=pt['g_h'],
                           c_ctx2th=pt['c_ctx2th'],
                           c_th2ctx=cv, dur_ms=dur,
                           wall_s=round(dt, 1),
                           **res)
                rows.append(row)
                # Quick read-out
                t4_ok = "✓" if res['T4_q'] > SO_Q_MIN else "✗"
                t6_ok = "✓" if res['T6_ibi_cv'] < IBI_CV_MAX else "✗"
                sp_ok = "✓" if res['spindle_power'] > 0.1 else "✗"
                sweet = (res['T4_q'] > SO_Q_MIN and
                         res['T6_ibi_cv'] < IBI_CV_MAX and
                         res['spindle_power'] > 0.1)
                marker = " ★" if sweet else ""
                print(f"T4{t4_ok}(q={res['T4_q']:.2f}) "
                      f"T6{t6_ok}(cv={res['T6_ibi_cv']:.2f}) "
                      f"sp{sp_ok}(P={res['spindle_power']:.2f}) "
                      f"so_P={res['so_power']:.2f}  "
                      f"[{dt:.1f}s]{marker}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nTotal wall time: {(time.time()-t_total)/60:.1f} min")
    print(f"Saved: {OUT_CSV}")

    # Plot heatmaps
    plot_heatmaps(df)

    # Sweet-spot report
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        analyze_sweet_spot(df, fh)
    print(f"Saved: {OUT_TXT}")


if __name__ == "__main__":
    main()
