"""
plot_thalamus_sweep_heatmaps.py
================================
Visualize the 3D isolated thalamus sweep as heatmap slices.

For each value of c_ctx2th (the 3rd swept dimension), plot a grid of
heatmaps over (g_LK, g_h) showing:

  Row 1: sp_peak_power_log  (spindle peak vs floor, in dB-ish)
  Row 2: n_events_verified  (v6-preview: peak-inside-event events)
  Row 3: pac_mi             (Kullback-Leibler modulation index)

This identifies the "spindle hotspot" — the (g_LK, g_h) region where
the thalamus genuinely produces waxing-waning σ-band bursts, as a
function of cortex → thalamus coupling strength.

Usage:
  python plot_scripts/plot_thalamus_sweep_heatmaps.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

IN_NPZ  = "outputs/thalamus_sweep_3d.npz"
OUT_PNG = "outputs/thalamus_sweep_heatmaps.png"

# V5 bounds (for overlay)
V5_GLK_BOUNDS  = (0.040, 0.073)
V5_GH_BOUNDS   = (0.036, 0.067)


def plot_heatmap(ax, data, x_grid, y_grid, title, cmap='viridis',
                 vmin=None, vmax=None, show_v5_box=True, log_norm=False):
    """One heatmap panel."""
    data = np.nan_to_num(data, nan=0.0)
    if log_norm and data.max() > 0:
        norm = LogNorm(vmin=max(data[data>0].min(), 1e-6) if (data>0).any() else 1e-6,
                       vmax=max(data.max(), 1e-5))
        im = ax.imshow(data.T, origin='lower',
                       extent=[x_grid.min(), x_grid.max(),
                               y_grid.min(), y_grid.max()],
                       aspect='auto', cmap=cmap, norm=norm)
    else:
        im = ax.imshow(data.T, origin='lower',
                       extent=[x_grid.min(), x_grid.max(),
                               y_grid.min(), y_grid.max()],
                       aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    if show_v5_box:
        # Draw v5 bounds rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (V5_GLK_BOUNDS[0], V5_GH_BOUNDS[0]),
            V5_GLK_BOUNDS[1] - V5_GLK_BOUNDS[0],
            V5_GH_BOUNDS[1] - V5_GH_BOUNDS[0],
            linewidth=1.5, edgecolor='white', facecolor='none',
            linestyle='--', alpha=0.8, label='v5 bounds',
        )
        ax.add_patch(rect)

    ax.set_xlabel('g_LK [mS/cm²]')
    ax.set_ylabel('g_h [mS/cm²]')
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    if not os.path.exists(IN_NPZ):
        print(f"[error] {IN_NPZ} not found. Run s4_0_isolated_thalamus_sweep.py first.")
        return

    data = np.load(IN_NPZ)
    g_LK     = data['g_LK_grid']
    g_h      = data['g_h_grid']
    c_ctx2th = data['c_ctx2th_grid']

    sp_peak = data['sp_peak_power_log']     # shape (nlk, nh, nc)
    n_ver   = data['n_events_verified']
    pac_mi  = data['pac_mi']

    nc = len(c_ctx2th)
    fig, axes = plt.subplots(3, nc, figsize=(4 * nc, 10), squeeze=False)

    # Unified color scales across c_ctx2th slices
    vmax_sp = np.nanmax(sp_peak) if np.any(np.isfinite(sp_peak)) else 1.0
    vmax_nv = np.nanmax(n_ver) if np.any(np.isfinite(n_ver)) else 1.0
    vmax_mi = np.nanmax(pac_mi) if np.any(np.isfinite(pac_mi)) else 0.01

    for k, c_val in enumerate(c_ctx2th):
        # Row 1: spindle peak power
        plot_heatmap(
            axes[0, k], sp_peak[:, :, k], g_LK, g_h,
            f'sp_peak_log\nc_ctx2th={c_val:.3f}',
            cmap='viridis', vmin=0, vmax=vmax_sp,
        )
        # Row 2: verified event count
        plot_heatmap(
            axes[1, k], n_ver[:, :, k], g_LK, g_h,
            f'n_events_verified\nc_ctx2th={c_val:.3f}',
            cmap='hot', vmin=0, vmax=vmax_nv,
        )
        # Row 3: PAC MI
        plot_heatmap(
            axes[2, k], pac_mi[:, :, k], g_LK, g_h,
            f'PAC MI\nc_ctx2th={c_val:.3f}',
            cmap='plasma', vmin=0, vmax=vmax_mi,
        )

    # Global title
    fig.suptitle(
        'Isolated thalamus sweep — spindle-region landscape\n'
        'Dashed white rectangle = v5 DE bounds (g_LK × g_h plane)',
        fontsize=12, y=1.01,
    )

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUT_PNG}")

    # Also identify and print the "hotspot" — highest n_events_verified point
    print("\n" + "=" * 60)
    print("Spindle hotspots (top 10 points by n_events_verified):")
    print("=" * 60)
    flat_idx = np.argsort(n_ver.ravel())[::-1][:10]
    for rank, idx in enumerate(flat_idx, 1):
        i, j, k = np.unravel_index(idx, n_ver.shape)
        print(f"  #{rank:2d}  g_LK={g_LK[i]:.3f}  g_h={g_h[j]:.3f}  "
              f"c_ctx2th={c_ctx2th[k]:.3f}  |  "
              f"n_ver={int(n_ver[i,j,k]):2d}  "
              f"sp_log={sp_peak[i,j,k]:.2f}  "
              f"MI={pac_mi[i,j,k]:.4f}")

    # Recommend new v6 bounds from hotspot envelope
    print("\n" + "=" * 60)
    print("Recommended v6 bounds (envelope of top 10% hotspot points):")
    print("=" * 60)
    threshold = np.nanpercentile(n_ver, 90)
    if threshold > 0:
        mask = n_ver >= threshold
        ii, jj, kk = np.where(mask)
        if len(ii) > 0:
            print(f"  g_LK     ∈ [{g_LK[ii].min():.4f}, {g_LK[ii].max():.4f}]  "
                  f"(v5: [0.040, 0.073])")
            print(f"  g_h      ∈ [{g_h[jj].min():.4f}, {g_h[jj].max():.4f}]  "
                  f"(v5: [0.036, 0.067])")
            print(f"  c_ctx2th ∈ [{c_ctx2th[kk].min():.4f}, "
                  f"{c_ctx2th[kk].max():.4f}]  (v5: [0.027, 0.050])")


if __name__ == "__main__":
    main()
