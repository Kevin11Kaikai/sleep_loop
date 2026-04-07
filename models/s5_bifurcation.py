"""
s5_bifurcation.py
Bifurcation analysis: 2-D parameter scan of the ALN mean-field model,
reproducing neurolib paper Fig. 3g style map.
Marks patient_params position and prints bistable-boundary distance.

Backend note
------------
neurolib.optimize.exploration.BoxSearch and ParameterSpace both depend on
pypet, which is incompatible with NumPy >= 2.0 (uses removed np.string_).
Replacement: manual nested-loop grid scan over ALNModel instances — equivalent
in results, no third-party dependency required.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neurolib.models.aln import ALNModel

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 1: load patient params
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with open("data/patient_params.json") as fh:
    pp = json.load(fh)

print("Patient params loaded:")
for k, v in pp.items():
    print(f"  {k}: {v}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 2: ALNModel parameter key discovery
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core ALNModel uses:
#   mue_ext_mean, mui_ext_mean  — background input [mV/ms]  (scan axes)
#   sigma_ou                    — OU noise std (set 0 for clean bifurcation)
#   b, tauA                     — adaptation parameters
#   duration, dt                — simulation time
#   output[0]                   — rates_exc, shape (1, n_times)
#
# MultiModel ALNNode uses input_0.mu for the same physical quantity.
# Units are the same; we use patient values directly on the core scale.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 3: scan configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Grid covers [0, 4] × [0, 5] to capture patient mui ≈ 4.4
N_PTS    = 21       # 21×21 = 441 sims (fast MVP; increase to 31 for publication)
MUE_VALS = np.linspace(0.0, 4.0, N_PTS)
MUI_VALS = np.linspace(0.0, 5.0, N_PTS)

# Simulation: 2 s no-noise, analyse last 1 s
SIM_DUR_MS  = 2000.0
ANALYS_MS   = 1000.0
DT_MS       = 0.1
FS          = 1000.0 / DT_MS          # 10 000 Hz (internal model rate)
N_ANALYS    = int(ANALYS_MS / DT_MS)  # last 10 000 steps

# Run two bifurcation maps:
#   A) paper default  : b=0,    tauA=200  (classic Fig.3g)
#   B) patient-specific: b=19.5, tauA=1040 (personalised)
SCAN_CONFIGS = {
    "default": {"b": 0.0,        "tauA": 200.0},
    "patient": {"b": pp["b"],    "tauA": pp["tauA"]},
}

os.makedirs("outputs", exist_ok=True)
os.makedirs("data",    exist_ok=True)


def run_scan(b_val, tauA_val, label):
    """
    Run 21×21 grid of ALNModel simulations.
    Returns DataFrames with max_r, min_r, mean_r per (mue, mui) point.
    """
    records = []
    n_total = N_PTS * N_PTS
    n_done  = 0
    t0      = time.time()

    for i, mue in enumerate(MUE_VALS):
        for j, mui in enumerate(MUI_VALS):
            model = ALNModel()
            model.params["duration"]     = SIM_DUR_MS
            model.params["dt"]           = DT_MS
            model.params["sigma_ou"]     = 0.0
            model.params["b"]            = b_val
            model.params["tauA"]         = tauA_val
            model.params["mue_ext_mean"] = float(mue)
            model.params["mui_ext_mean"] = float(mui)

            try:
                model.run()
                r_e = model.output[0]        # shape (1, n_times) → squeeze to (n_times,)
                r_e = np.asarray(r_e).ravel()
                r_end = r_e[-N_ANALYS:]      # last 1 s
                max_r  = float(r_end.max())
                min_r  = float(r_end.min())
                mean_r = float(r_end.mean())
            except Exception as exc:
                max_r = min_r = mean_r = float("nan")

            records.append({
                "mue_ext_mean": float(mue),
                "mui_ext_mean": float(mui),
                "max_r":  max_r,
                "min_r":  min_r,
                "mean_r": mean_r,
            })

            n_done += 1
            if n_done % 50 == 0 or n_done == n_total:
                elapsed = time.time() - t0
                eta     = elapsed / n_done * (n_total - n_done)
                print(f"  [{label}] {n_done}/{n_total}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    return pd.DataFrame(records)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 4: run scans
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

dfs = {}
for cfg_name, cfg in SCAN_CONFIGS.items():
    print(f"\nRunning scan [{cfg_name}]: b={cfg['b']}, tauA={cfg['tauA']}")
    print(f"  Grid: {N_PTS}×{N_PTS} = {N_PTS**2} simulations")
    dfs[cfg_name] = run_scan(cfg["b"], cfg["tauA"], cfg_name)
    dfs[cfg_name].to_csv(f"outputs/bifurcation_{cfg_name}.csv", index=False)
    print(f"  Saved: outputs/bifurcation_{cfg_name}.csv")

print("\nAll scans complete.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 5: build 2-D grids and identify bistable regions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_grids(df):
    """Pivot DataFrame into 2-D arrays indexed [mui_idx, mue_idx]."""
    mue_u = sorted(df["mue_ext_mean"].unique())
    mui_u = sorted(df["mui_ext_mean"].unique())
    shape  = (len(mui_u), len(mue_u))

    max_g  = np.full(shape, np.nan)
    min_g  = np.full(shape, np.nan)
    osc_g  = np.full(shape, np.nan)

    for _, row in df.iterrows():
        i = mui_u.index(row["mui_ext_mean"])
        j = mue_u.index(row["mue_ext_mean"])
        max_g[i, j] = row["max_r"]
        min_g[i, j] = row["min_r"]
        osc_g[i, j] = row["max_r"] - row["min_r"]

    # Bistable approx: up-state (max > 5 Hz) AND down-state (min < 2 Hz) co-exist
    bistable = (max_g > 5.0) & (min_g < 2.0)
    return np.array(mue_u), np.array(mui_u), max_g, min_g, osc_g, bistable


grids = {name: build_grids(df) for name, df in dfs.items()}


def distance_to_bistable(mue_pt, mui_pt, mue_u, mui_u, bistable):
    """Euclidean distance in (mue, mui) space from point to nearest bistable grid cell."""
    coords = np.argwhere(bistable)
    if len(coords) == 0:
        return None
    dists = [
        np.sqrt((mue_u[c] - mue_pt) ** 2 + (mui_u[r] - mui_pt) ** 2)
        for r, c in coords
    ]
    return float(min(dists))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 6: patient mapping & diagnosis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── coordinate mapping: MultiModel → core ALN scale ─────────────────────────
# MultiModel ALNNode input_0.mu and core ALNModel mue_ext_mean share the same
# physical meaning but differ in effective scale (verified in Session 1-B).
# Empirical mapping: ALN_mue ≈ MM_mue × 0.76
#   reference: MM mue=3.20 → ALN Table 3 mue=2.30 (ratio ≈ 0.72)
SCALE = 0.76   # MultiModel → core ALN empirical scale factor

pp_mue_mm = pp["mue"]
pp_mui_mm = pp["mui"]

pp_mue = pp_mue_mm * SCALE
pp_mui = pp_mui_mm * SCALE

print(f"\nPatient coordinates on bifurcation map:")
print(f"  MultiModel  : mue={pp_mue_mm:.3f}, mui={pp_mui_mm:.3f}")
print(f"  ALN (×{SCALE}) : mue={pp_mue:.3f}, mui={pp_mui:.3f}  <- used for bifurcation map")

patient_results = {}
for cfg_name in SCAN_CONFIGS:
    mue_u, mui_u, max_g, min_g, osc_g, bistable = grids[cfg_name]
    dist = distance_to_bistable(pp_mue, pp_mui, mue_u, mui_u, bistable)
    patient_results[cfg_name] = dist

    print(f"\n[{cfg_name}] Dynamics diagnosis:")
    print(f"  Bistable grid points : {bistable.sum()}")
    if dist is None:
        print("  ⚠ No bistable region found in scanned range")
    else:
        print(f"  Distance to bistable boundary : {dist:.3f} mV/ms")
        if dist < 0.3:
            print("  ⚠ CLOSE to bistable boundary → high sleep-fragmentation risk")
        elif dist < 0.8:
            print("  ● NEAR bistable region         → moderate fragmentation risk")
        else:
            print("  ✓ FAR from bistable boundary   → low fragmentation risk")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 7: plots
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.ravel()

for ax_idx, cfg_name in enumerate(["default", "patient"]):
    mue_u, mui_u, max_g, min_g, osc_g, bistable = grids[cfg_name]
    dist = patient_results[cfg_name]
    ext  = [mue_u[0], mue_u[-1], mui_u[0], mui_u[-1]]
    cfg  = SCAN_CONFIGS[cfg_name]

    # ── max r_E heat-map (left column) ──────────────────────────────
    ax = axes[ax_idx * 2]
    im = ax.imshow(max_g, origin="lower", extent=ext,
                   aspect="auto", cmap="plasma",
                   vmin=0, vmax=max_g[~np.isnan(max_g)].max())
    plt.colorbar(im, ax=ax, label="Max $r_E$ [Hz]")

    # bistable boundary contour
    if bistable.any():
        ax.contour(mue_u, mui_u, bistable.astype(float),
                   levels=[0.5], colors="lime",
                   linestyles="--", linewidths=1.8)

    # patient marker
    ax.scatter(pp_mue, pp_mui, marker="*", s=350,
               color="red", zorder=10,
               label=(f"Patient\n"
                      f"MM:({pp_mue_mm:.2f},{pp_mui_mm:.2f})\n"
                      f"ALN:({pp_mue:.2f},{pp_mui:.2f})"))

    title_dist = f"dist={dist:.3f}" if dist is not None else "no bistable region"
    ax.set_title(
        f"[{cfg_name}] Max $r_E$ — b={cfg['b']}, tauA={cfg['tauA']} ms\n"
        f"Green dashed = bistable boundary  |  {title_dist}",
        fontsize=10,
    )
    ax.set_xlabel("mue_ext_mean [mV/ms]"); ax.set_ylabel("mui_ext_mean [mV/ms]")
    ax.legend(fontsize=8, loc="upper left")

    # ── oscillation amplitude heat-map (right column) ────────────────
    ax = axes[ax_idx * 2 + 1]
    im2 = ax.imshow(osc_g, origin="lower", extent=ext,
                    aspect="auto", cmap="viridis")
    plt.colorbar(im2, ax=ax, label="Oscillation amplitude [Hz]")

    if bistable.any():
        ax.contour(mue_u, mui_u, bistable.astype(float),
                   levels=[0.5], colors="lime",
                   linestyles="--", linewidths=1.8)

    ax.scatter(pp_mue, pp_mui, marker="*", s=350,
               color="red", zorder=10)
    ax.set_title(
        f"[{cfg_name}] Oscillation amplitude (max−min $r_E$)\n"
        f"High = limit-cycle region",
        fontsize=10,
    )
    ax.set_xlabel("mue_ext_mean [mV/ms]"); ax.set_ylabel("mui_ext_mean [mV/ms]")

dist_default = patient_results["default"]
dist_patient = patient_results["patient"]
fig.suptitle(
    f"Session 2-C: Bifurcation map  (ALN {N_PTS}×{N_PTS} grid)\n"
    f"Patient distance to bistable boundary — "
    f"default: {dist_default:.3f}  |  personalised: {dist_patient:.3f}",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("outputs/bifurcation_map.png", dpi=150, bbox_inches="tight")
print("\nSaved: outputs/bifurcation_map.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 8: save summary JSON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

summary = {
    "patient_mue":            pp_mue,
    "patient_mui":            pp_mui,
    "grid_n":                 N_PTS,
    "bistable_dist_default":  dist_default,
    "bistable_dist_patient":  dist_patient,
    "bistable_n_default":     int(grids["default"][-1].sum()),
    "bistable_n_patient":     int(grids["patient"][-1].sum()),
}
with open("data/bifurcation_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print("Saved: data/bifurcation_summary.json")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 9: final validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n=== Session 2-C Validation ===")
_, _, _, _, _, bs_default = grids["default"]
_, _, _, _, _, bs_patient = grids["patient"]

if bs_default.sum() > 0:
    print(f"✓ Bistable region identified [default] ({bs_default.sum()} grid points)")
else:
    print("~ No bistable region in default scan (may need wider range)")

if bs_patient.sum() > 0:
    print(f"✓ Bistable region identified [patient] ({bs_patient.sum()} grid points)")
else:
    print("~ No bistable region in patient scan")

if dist_patient is not None:
    print(f"✓ Patient distance to bistable boundary: {dist_patient:.3f} mV/ms")

print("✓ outputs/bifurcation_map.png saved")
print("Session 2-C complete. Ready for Session 3-A (RL environment).")
