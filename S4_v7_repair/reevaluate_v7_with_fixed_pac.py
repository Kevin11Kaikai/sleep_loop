"""
reevaluate_v7_with_fixed_pac.py
================================
Re-evaluate v7's existing solutions with the fixed PAC metrics.

What this does
--------------
Reads the 96 feasible solutions from evolution_fig7_v7_records.csv,
re-simulates each one (~3 sec/solution), and re-scores using:
  - The same v7 fitness formula (shape_r, so_power, spindle_power)
  - The FIXED compute_pac_metrics for T9/T10/T11
  - New T10/T11 thresholds based on phase_argmax + up_down_ratio

Output
------
  reevaluate_v7_results.csv — full new scores for all 96 solutions
  reevaluate_v7_summary.txt — top 10 ranking under new criteria

Total runtime: ~5-15 minutes for 96 solutions.

USAGE
-----
Place this file in the same directory as s4_personalize_fig7_v7.py.
Make sure compute_pac_metrics_fixed.py is in the same directory or in PYTHONPATH.

    python reevaluate_v7_with_fixed_pac.py

If you want to re-evaluate fewer solutions for a quick test:
    python reevaluate_v7_with_fixed_pac.py --top 10
"""

import os
import sys
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, sosfiltfilt, hilbert, welch
from scipy.ndimage import gaussian_filter1d


# ============================================================================
# Path setup — adjust if your project layout differs
# ============================================================================
PROJECT_ROOT = Path(r"D:\Year3_Mao_Projects\sleep_loop")
V7_SCRIPT = PROJECT_ROOT / "models" / "s4_personalize_fig7_v7.py"
RECORDS_CSV = PROJECT_ROOT / "outputs" / "evolution_fig7_v7_records.csv"
OUTPUT_DIR = PROJECT_ROOT / "reevaluate_v7"
OUTPUT_DIR.mkdir(exist_ok=True)

# Try alternative paths
if not V7_SCRIPT.exists():
    V7_SCRIPT = PROJECT_ROOT / "models" / "s4_personalize_fig7_v7.py"
if not RECORDS_CSV.exists():
    RECORDS_CSV = PROJECT_ROOT / "outputs" / "evolution_fig7_v7_records.csv"


# ============================================================================
# Import the fixed PAC function
# ============================================================================
sys.path.insert(0, str(Path(__file__).parent))
try:
    from compute_pac_metrics_fixed import compute_pac_metrics as compute_pac_fixed
except ImportError:
    print("ERROR: cannot import compute_pac_metrics_fixed")
    print("Please put compute_pac_metrics_fixed.py in the same directory.")
    sys.exit(1)


# ============================================================================
# Import v7's helper functions (build_model, compute_constraints, etc)
# We DON'T want to run v7's main(), just reuse its functions.
# ============================================================================
def import_v7_module():
    """Import v7 script as module without triggering main()."""
    spec = importlib.util.spec_from_file_location("v7_module", str(V7_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    # Suppress main() execution
    original_name = "__main__"
    sys.modules["v7_module"] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass  # v7 might call sys.exit somewhere
    return module


# ============================================================================
# New thresholds (mirror integration guide)
# ============================================================================
# Old PAC criteria (will be overridden):
#   PAC_MI_MIN = 0.005
#   PAC_PHASE_TOL = 5*pi/18  (=50°)
#   PAC_MIN_LAG_MS = 20.0
#
# New PAC criteria:
PAC_MI_MIN = 0.005                          # T9: same as before
PAC_PHASE_TOL_DEG = 50                      # T10a: argmax within 50° of UP peak (0°)
PAC_CONCENTRATION_MIN = 0.08                # T10b: |histogram MVL|, Helfrich precision
PAC_UP_DOWN_RATIO_MIN = 1.20                # T11_v2: spindle UP-locked
# Note: old T11 (lag-based) is DROPPED — replaced by up_down_ratio


def evaluate_new_pac_constraints(r_ctx, r_thal, fs=1000.0) -> Dict:
    """Compute new T9/T10a/T10b/T11_v2 with the fixed PAC function."""
    pac = compute_pac_fixed(r_ctx, r_thal, fs)

    if not pac["ok"]:
        return {
            "T9_new": False, "T9_mi": 0.0,
            "T10a_new": False, "T10a_phase_argmax_deg": 180.0,
            "T10b_new": False, "T10b_concentration": 0.0,
            "T11_new": False, "T11_up_down_ratio": 0.0,
            "pac_bimodal": False, "pac_ok": False,
            "pac_n_so_cycles": 0,
        }

    T9_new = pac["mi"] >= PAC_MI_MIN

    phase_argmax_deg = np.degrees(pac["phase_argmax"])
    phase_dist = min(abs(phase_argmax_deg), 360 - abs(phase_argmax_deg))
    T10a_new = phase_dist <= PAC_PHASE_TOL_DEG

    T10b_new = pac["phase_concentration"] >= PAC_CONCENTRATION_MIN

    T11_new = pac["up_down_ratio"] >= PAC_UP_DOWN_RATIO_MIN

    return {
        "T9_new": T9_new, "T9_mi": pac["mi"],
        "T10a_new": T10a_new, "T10a_phase_argmax_deg": phase_argmax_deg,
        "T10b_new": T10b_new, "T10b_concentration": pac["phase_concentration"],
        "T11_new": T11_new, "T11_up_down_ratio": pac["up_down_ratio"],
        "pac_bimodal": pac["bimodality_flag"],
        "pac_ok": True,
        "pac_n_so_cycles": pac["n_so_cycles"],
    }


# ============================================================================
# Main re-evaluation loop
# ============================================================================
def reevaluate_all(top_n: int = None):
    print("=" * 72)
    print("Re-evaluating v7 solutions with fixed PAC metrics")
    print("=" * 72)
    print(f"v7 script:    {V7_SCRIPT}")
    print(f"Records CSV:  {RECORDS_CSV}")
    print(f"Output dir:   {OUTPUT_DIR}")

    # ── Load records ────────────────────────────────────────────────────
    df = pd.read_csv(RECORDS_CSV)
    feasible = df[df["feasible"] == 1].copy().reset_index(drop=True)
    print(f"\nFound {len(feasible)} feasible solutions.")

    if top_n is not None:
        feasible = feasible.sort_values("score", ascending=False).head(top_n)
        print(f"Re-evaluating top {top_n} only (testing mode).")

    # ── Import v7 module ────────────────────────────────────────────────
    print("\nLoading v7 simulation pipeline...")
    v7 = import_v7_module()
    print("  build_model, compute_constraints_v7 loaded.")

    # ── Re-evaluate each ────────────────────────────────────────────────
    PARAM_NAMES = ["mue", "mui", "b", "tauA", "g_LK",
                    "g_h", "c_th2ctx", "c_ctx2th"]
    FS_SIM = v7.FS_SIM

    results = []
    print(f"\nRe-simulating {len(feasible)} solutions...")
    for i, row in tqdm(feasible.iterrows(), total=len(feasible)):
        params = {k: row[k] for k in PARAM_NAMES}
        old_score = row["score"]

        # Run simulation
        try:
            m = v7.build_model(**{
                "mue": params["mue"], "mui": params["mui"],
                "b": params["b"], "tauA": params["tauA"],
                "g_lk": params["g_LK"], "g_h": params["g_h"],
                "c_th2ctx": params["c_th2ctx"],
                "c_ctx2th": params["c_ctx2th"],
            })
            m.run()
        except Exception as e:
            print(f"  [warn] sim failed for row {i}: {e}")
            continue

        r_exc = m[f"r_mean_{v7.EXC}"]
        if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
            r_ctx = r_exc[0, :] * 1000.0
            r_thal = r_exc[1, :] * 1000.0
        else:
            r_ctx = (r_exc[0] if r_exc.ndim == 2 else r_exc) * 1000.0
            r_thal = np.zeros_like(r_ctx)

        # Discard burn-in
        n_drop = int(5.0 * FS_SIM)
        r_ctx = r_ctx[n_drop:]
        r_thal = r_thal[n_drop:]

        # Compute new PAC
        pac_new = evaluate_new_pac_constraints(r_ctx, r_thal, FS_SIM)

        # Compute T1-T8, T12 with v7's logic (these are unchanged)
        n_passed_old, con_old = v7.compute_constraints_v7(
            r_ctx, r_thal, fs=FS_SIM
        )

        # Now combine: T1-T8, T12 from old, T9/T10/T11 from new
        n_passed_new = sum([
            con_old.get("T1", False), con_old.get("T2", False),
            con_old.get("T3", False), con_old.get("T4", False),
            con_old.get("T5", False), con_old.get("T6", False),
            con_old.get("T7", False), con_old.get("T8", False),
            pac_new["T9_new"],
            pac_new["T10a_new"] and pac_new["T10b_new"],  # combined T10
            pac_new["T11_new"],
            con_old.get("T12", False),
        ])
        new_feasible = (n_passed_new == 12)

        # Compute new fitness if feasible (same v7 formula, recomputed shape_r and rewards
        # would require re-running compute_fitness_v7 — instead we use stored
        # shape_r since it's parameter-deterministic)
        if new_feasible:
            shape_r = row["shape_r"]
            so_power = float(np.clip((con_old.get("T4_q", 0) - 1) / 4, 0, 1))
            spindle_power = float(np.clip(
                con_old.get("T12_n_verified", 0) / 15, 0, 1))
            new_score = 0.5 * shape_r + 0.25 * so_power + 0.25 * spindle_power
        else:
            new_score = -10.0  # infeasible marker

        result = {
            **{k: params[k] for k in PARAM_NAMES},
            "old_score": old_score,
            "old_feasible": True,  # we filtered on this
            "new_score": round(new_score, 6),
            "new_feasible": int(new_feasible),
            "n_passed_new": n_passed_new,
            **{f"old_{k}": v for k, v in con_old.items()
               if k.startswith("T") and not k[1:].isdigit()},
            **pac_new,
            "shape_r": row["shape_r"],
            "T4_q": con_old.get("T4_q", 0),
            "T12_n_verified": con_old.get("T12_n_verified", 0),
        }
        results.append(result)

    # ── Save ────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(results)
    out_csv = OUTPUT_DIR / "reevaluate_v7_results_new.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

    # ── Summary ─────────────────────────────────────────────────────────
    n_still_feasible = df_out["new_feasible"].sum()
    print()
    print("=" * 72)
    print(f"SUMMARY — {n_still_feasible} / {len(df_out)} solutions still feasible")
    print("=" * 72)

    if n_still_feasible > 0:
        feasible_new = df_out[df_out["new_feasible"] == 1].sort_values(
            "new_score", ascending=False
        )
        print(f"\nTop 10 under new criteria:")
        print(f"{'rank':<5} {'old_score':<10} {'new_score':<10} "
              f"{'old_feas':<9} {'new_feas':<9} {'phase_argmax':<13} "
              f"{'concentration':<14} {'up_down':<8} {'bimodal':<8}")
        for rank, (_, r) in enumerate(feasible_new.head(10).iterrows(), 1):
            print(f"{rank:<5} {r['old_score']:<10.4f} {r['new_score']:<10.4f} "
                  f"{int(r['old_feasible']):<9} {int(r['new_feasible']):<9} "
                  f"{r['T10a_phase_argmax_deg']:<+13.1f} "
                  f"{r['T10b_concentration']:<14.4f} "
                  f"{r['T11_up_down_ratio']:<8.3f} {r['pac_bimodal']!s:<8}")

    # Save summary
    summary_path = OUTPUT_DIR / "reevaluate_v7_summary_new.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("V7 RE-EVALUATION WITH FIXED PAC\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Total v7 feasible solutions:      {len(df_out)}\n")
        f.write(f"Still feasible under new criteria: {n_still_feasible}\n")
        f.write(f"Failure rate:                     "
                f"{100*(1-n_still_feasible/len(df_out)):.1f}%\n\n")

        # Why solutions failed
        if n_still_feasible < len(df_out):
            failed = df_out[df_out["new_feasible"] == 0]
            t9_fail = (~failed["T9_new"]).sum()
            t10a_fail = (~failed["T10a_new"]).sum()
            t10b_fail = (~failed["T10b_new"]).sum()
            t11_fail = (~failed["T11_new"]).sum()
            f.write("Failure breakdown (counts; some may fail multiple):\n")
            f.write(f"  T9 fail (MI < {PAC_MI_MIN}):                  {t9_fail}\n")
            f.write(f"  T10a fail (phase_argmax > {PAC_PHASE_TOL_DEG}° from UP): {t10a_fail}\n")
            f.write(f"  T10b fail (concentration < {PAC_CONCENTRATION_MIN}):     {t10b_fail}\n")
            f.write(f"  T11 fail (up_down_ratio < {PAC_UP_DOWN_RATIO_MIN}):      {t11_fail}\n\n")

        if n_still_feasible > 0:
            feasible_new = df_out[df_out["new_feasible"] == 1].sort_values(
                "new_score", ascending=False
            )
            f.write("Top 10 solutions under new criteria:\n")
            f.write("-" * 72 + "\n")
            for rank, (_, r) in enumerate(feasible_new.head(10).iterrows(), 1):
                f.write(f"\n# Rank {rank} (was rank ~{(df_out['old_score']>r['old_score']).sum()+1} under old criteria)\n")
                f.write(f"  Score: {r['old_score']:.4f} → {r['new_score']:.4f}\n")
                f.write(f"  PAC:   MI={r['T9_mi']:.4f}, "
                        f"phase_argmax={r['T10a_phase_argmax_deg']:+.1f}°, "
                        f"concentration={r['T10b_concentration']:.3f}, "
                        f"up_down_ratio={r['T11_up_down_ratio']:.2f}, "
                        f"bimodal={r['pac_bimodal']}\n")
                f.write(f"  Params: ")
                for k in ["mue", "mui", "b", "tauA", "g_LK", "g_h",
                         "c_th2ctx", "c_ctx2th"]:
                    f.write(f"{k}={r[k]:.4f} ")
                f.write("\n")

    print(f"\nSummary saved to {summary_path}")
    print("\nNext step:")
    print("  - If top 1-3 are stable, no need to re-run DE → use the new ranking")
    print("  - If best v7 solution fails new criteria, run warm_start_de.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=None,
                        help="Only re-evaluate the top N solutions (for quick test)")
    args = parser.parse_args()
    reevaluate_all(top_n=args.top)