"""
pareto_analysis.py
==================
Pareto front analysis of warm_start DE feasible solutions.

Purpose
-------
Stage 1 closure deliverable. Demonstrates that the 302 feasible solutions
found by warm_start DE form a Pareto front in (shape_r, PAC) space, not
a single optimum. This is the visual evidence that justifies switching
to SBI in Stage 2 — because no scalar weighted fitness can pick a single
"correct" point from this front; only a posterior distribution can
represent the full trade-off structure.

Two outputs
-----------
1. fig_pareto_2d.png — main figure for progress report. Scatter of all
   302 feasible solutions in (shape_r, PAC_compound) space, with
   Pareto front highlighted, V8 saved-best marked, and 5 representative
   solutions extracted as SBI prior seeds.

2. pareto_seeds.json — the 5 representative solutions on the Pareto
   front, ready to be loaded as informative prior centers for SBI.

PAC_compound is a normalised geometric mean of the three PAC dimensions
(MI, up_down_ratio, concentration). Geometric mean is used instead of
arithmetic mean because it penalises solutions that are weak on any
single PAC dimension — a solution must be "all-around good on PAC",
not just good on one metric.

Usage
-----
  python pareto_analysis.py \
      --records warm_start_records.csv \
      --out-fig fig_pareto_2d.png \
      --out-json pareto_seeds.json

  # Quick run with defaults:
  python pareto_analysis.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# --------------------------------------------------------------------- #
# Pareto front computation
# --------------------------------------------------------------------- #
def compute_pareto_front(points):
    """
    Compute the Pareto-optimal subset of `points` for MAXIMISING all
    objectives.

    Args
    ----
    points : (N, K) array — N candidate solutions, each with K objectives
             (all to be maximised).

    Returns
    -------
    is_pareto : (N,) boolean — True for Pareto-optimal points.

    Algorithm: O(N^2) pairwise dominance check. Fine for N=302.
    """
    n = points.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is >= in all dims and > in at least one
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                is_pareto[i] = False
                break
    return is_pareto


def pac_compound(df):
    """
    Geometric mean of three PAC dimensions, after each is min-max
    normalised across the feasible population.

    Geometric mean (rather than arithmetic) is used because it goes to
    zero if ANY dimension is zero — i.e. it penalises solutions that
    are weak on any single PAC dimension.
    """
    norm = pd.DataFrame()
    
    # T9_mi: Higher is better
    c = 'T9_mi'
    lo, hi = df[c].min(), df[c].max()
    norm[c] = (df[c] - lo) / (hi - lo + 1e-12)

    # T11_lag_ms (which holds up_down_ratio in V7): Higher is better
    c = 'T11_lag_ms'
    if c in df.columns:
        lo, hi = df[c].min(), df[c].max()
        norm[c] = (df[c] - lo) / (hi - lo + 1e-12)
        
    # T10_dist_tgt: LOWER is better (distance to target phase)
    c = 'T10_dist_tgt'
    if c in df.columns:
        lo, hi = df[c].min(), df[c].max()
        norm[c] = (hi - df[c]) / (hi - lo + 1e-12)

    # Geometric mean. Add tiny offset so zeros don't kill the mean.
    return np.exp(np.log(norm + 1e-6).mean(axis=1)).values


# --------------------------------------------------------------------- #
# Representative solution selection
# --------------------------------------------------------------------- #
def select_representatives(df, pareto_mask, n_seeds=5):
    """
    Pick n_seeds representative points along the Pareto front, evenly
    spaced by their position along the (shape_r, PAC_compound) curve.

    Strategy: sort Pareto-optimal points by shape_r, then pick n_seeds
    points evenly distributed across that ordering. This guarantees
    coverage from "shape_r-dominant" to "PAC-dominant" extremes and
    a few balanced solutions in between.
    """
    pareto_df = df[pareto_mask].copy()
    pareto_df = pareto_df.sort_values('shape_r').reset_index(drop=True)
    n_pareto = len(pareto_df)
    if n_pareto <= n_seeds:
        return pareto_df, list(range(n_pareto))
    # Even index spacing along the sorted front
    idx = np.round(np.linspace(0, n_pareto - 1, n_seeds)).astype(int)
    return pareto_df.iloc[idx].reset_index(drop=True), idx.tolist()


# --------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------- #
def make_figure(df, pareto_mask, reps, out_path):
    """
    Two-panel figure:
      Left:  (shape_r, PAC_compound) scatter with Pareto front + reps.
      Right: 3D-ish view via small multiples — shape_r vs each individual
             PAC dimension, with Pareto-optimal points highlighted in each.
    """
    fig = plt.figure(figsize=(14, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.28)

    # ---------------- Panel A: main 2D Pareto plot ---------------- #
    ax = fig.add_subplot(gs[0, 0])

    pac = df['PAC_compound'].values
    sr = df['shape_r'].values

    # Background: all feasible
    ax.scatter(sr[~pareto_mask], pac[~pareto_mask],
               s=22, c='#888780', alpha=0.45,
               edgecolors='none', label=f'Feasible (n={(~pareto_mask).sum()})')

    # Pareto-optimal points
    ax.scatter(sr[pareto_mask], pac[pareto_mask],
               s=46, c='#534AB7', alpha=0.85,
               edgecolors='#26215C', linewidths=0.6,
               label=f'Pareto-optimal (n={pareto_mask.sum()})', zorder=3)

    # Connect Pareto front with a line for visual clarity
    pareto_pts = np.column_stack([sr[pareto_mask], pac[pareto_mask]])
    pareto_pts = pareto_pts[pareto_pts[:, 0].argsort()]
    ax.plot(pareto_pts[:, 0], pareto_pts[:, 1],
            color='#534AB7', alpha=0.4, lw=1.2, zorder=2)

    # Mark V8 saved-best (highest score)
    v8_idx = df['score'].idxmax()
    ax.scatter([df.loc[v8_idx, 'shape_r']],
               [df.loc[v8_idx, 'PAC_compound']],
               s=180, marker='D', c='#D85A30',
               edgecolors='#4A1B0C', linewidths=1.0,
               label=f'V8 saved-best (score={df.loc[v8_idx, "score"]:.4f})',
               zorder=5)

    # Mark the representative seeds
    ax.scatter(reps['shape_r'], reps['PAC_compound'],
               s=120, marker='*', c='#1D9E75',
               edgecolors='#04342C', linewidths=0.8,
               label=f'SBI prior seeds (n={len(reps)})', zorder=6)

    # Annotate seeds with letters
    for i, (_, r) in enumerate(reps.iterrows()):
        ax.annotate(chr(65 + i),  # A, B, C, ...
                    (r['shape_r'], r['PAC_compound']),
                    xytext=(7, 7), textcoords='offset points',
                    fontsize=11, fontweight='bold', color='#04342C')

    ax.set_xlabel('Spectral shape similarity  (shape_r, FOOOF Pearson r)',
                  fontsize=11)
    ax.set_ylabel('PAC compound score  (geometric mean of MI, udr, concentration)',
                  fontsize=11)
    ax.set_title('A. Pareto front in (shape_r, PAC) space — '
                 '302 feasible solutions',
                 fontsize=12, loc='left', pad=10)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.18, linewidth=0.5)
    ax.set_axisbelow(True)

    # ---------------- Panel B: per-dimension small multiples ---------------- #
    pac_dims = [
        ('T9_mi',                'PAC strength (MI)',           '#534AB7'),
        ('T11_lag_ms',           'Up-down ratio',               '#1D9E75'),
        ('T10_dist_tgt',         'Distance to Target Phase',    '#D4537E'),
    ]

    # Stack three sub-panels vertically inside the right column
    inner = gs[0, 1].subgridspec(3, 1, hspace=0.55)
    for k, (col, label, color) in enumerate(pac_dims):
        ax_k = fig.add_subplot(inner[k, 0])
        ax_k.scatter(df['shape_r'], df[col],
                     s=14, c='#888780', alpha=0.4, edgecolors='none')
        ax_k.scatter(df.loc[pareto_mask, 'shape_r'],
                     df.loc[pareto_mask, col],
                     s=22, c=color, alpha=0.8, edgecolors='none')
        ax_k.scatter([df.loc[v8_idx, 'shape_r']],
                     [df.loc[v8_idx, col]],
                     s=80, marker='D', c='#D85A30',
                     edgecolors='#4A1B0C', linewidths=0.8)
        # Compute Pearson r between shape_r and this PAC dim
        r = np.corrcoef(df['shape_r'], df[col])[0, 1]
        ax_k.set_xlabel('shape_r', fontsize=9)
        ax_k.set_ylabel(label, fontsize=9)
        ax_k.tick_params(labelsize=8)
        ax_k.grid(True, alpha=0.18, linewidth=0.5)
        ax_k.set_axisbelow(True)
        ax_k.text(0.97, 0.95, f'r = {r:+.2f}',
                  transform=ax_k.transAxes,
                  fontsize=9, ha='right', va='top',
                  bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white', edgecolor='#888780',
                            alpha=0.9, linewidth=0.5))

    fig.text(0.685, 0.93,
             'B. Per-dimension breakdown',
             fontsize=12, fontweight='bold', ha='left')
    fig.text(0.685, 0.91,
             'shape_r vs each PAC dimension (gray = all feasible, '
             'colored = Pareto-optimal)',
             fontsize=9, ha='left', color='#444441')

    fig.suptitle(
        'Stage 1 closure: Pareto front of 302 feasible warm_start solutions',
        fontsize=13, fontweight='bold', y=0.995)

    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f"  Saved figure: {out_path}")
    plt.close(fig)


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _HERE = Path(__file__).resolve().parent
    _ROOT = _HERE.parent
    parser.add_argument('--records', type=str,
                        default=str(_ROOT / 'warm_start_de' / 'warm_start_records.csv'),
                        help='Path to warm_start_records.csv')
    parser.add_argument('--out-fig', type=str,
                        default=str(_HERE / 'fig_pareto_2d.png'),
                        help='Output path for the Pareto figure')
    parser.add_argument('--out-json', type=str,
                        default=str(_HERE / 'pareto_seeds.json'),
                        help='Output path for the SBI prior seeds JSON')
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='Number of representative seeds to extract')
    args = parser.parse_args()

    print('=' * 60)
    print('Pareto front analysis')
    print('=' * 60)

    # Load
    df = pd.read_csv(args.records)
    print(f'  Total records: {len(df)}')
    feas = df[df['feasible'] == 1].copy().reset_index(drop=True)
    print(f'  Feasible:      {len(feas)}')

    # Compute compound PAC score
    feas['PAC_compound'] = pac_compound(feas)

    # 2D Pareto front in (shape_r, PAC_compound) space
    objectives = feas[['shape_r', 'PAC_compound']].values
    pareto_mask = compute_pareto_front(objectives)
    print(f'  Pareto-optimal: {pareto_mask.sum()} / {len(feas)} '
          f'({100 * pareto_mask.sum() / len(feas):.1f}%)')

    # Select representatives
    reps, _ = select_representatives(feas, pareto_mask, n_seeds=args.n_seeds)
    print(f'  Selected {len(reps)} representative seeds.')
    print()

    # Print representative table
    PARAM_COLS = ['mue', 'mui', 'b', 'tauA',
                  'g_LK', 'g_h', 'c_th2ctx', 'c_ctx2th']
    print('Representative seeds (Pareto front, sorted by shape_r):')
    print('-' * 60)
    for i, (_, r) in enumerate(reps.iterrows()):
        tag = chr(65 + i)
        print(f'  Seed {tag}:  shape_r={r["shape_r"]:.4f}  '
              f'PAC_cmp={r["PAC_compound"]:.4f}  score={r["score"]:.4f}')
        print(f'           MI={r["T9_mi"]:.5f}  '
              f'udr={r["T11_lag_ms"]:.3f}  '
              f'dist={r["T10_dist_tgt"]:.4f}  '
              f'T4_q={r["T4_q"]:.3f}')

    print()

    # Make figure
    print('Building figure...')
    make_figure(feas, pareto_mask, reps, args.out_fig)

    # Save seeds JSON
    seeds_out = {
        'description': (
            'Stage 1 closure: 5 representative solutions from the Pareto '
            'front of 302 feasible warm_start_DE solutions. Sorted by '
            'shape_r ascending (i.e., seed A is most PAC-dominant, '
            f'seed {chr(65 + len(reps) - 1)} is most shape_r-dominant). '
            'Use these as informative prior centers for Stage 2 SBI.'
        ),
        'n_feasible': int(len(feas)),
        'n_pareto_optimal': int(pareto_mask.sum()),
        'n_seeds': int(len(reps)),
        'seeds': [],
    }
    for i, (_, r) in enumerate(reps.iterrows()):
        tag = chr(65 + i)
        seeds_out['seeds'].append({
            'tag': tag,
            'params': {c: float(r[c]) for c in PARAM_COLS},
            'objectives': {
                'shape_r': float(r['shape_r']),
                'PAC_compound': float(r['PAC_compound']),
                'score': float(r['score']),
                'T4_q': float(r['T4_q']),
                'T12_n_verified': int(r['T12_n_verified']),
            },
            'pac_metrics': {
                'MI': float(r['T9_mi']),
                'up_down_ratio': float(r['T11_lag_ms']),
                'distance_to_target': float(r['T10_dist_tgt']),
            },
        })
    with open(args.out_json, 'w') as f:
        json.dump(seeds_out, f, indent=2)
    print(f'  Saved seeds JSON: {args.out_json}')
    print()
    print('Done.')


if __name__ == '__main__':
    main()