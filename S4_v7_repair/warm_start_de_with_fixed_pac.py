"""
warm_start_de_with_fixed_pac.py
================================
Warm-start DE: re-run optimization for 10 generations starting from
v7's top-K solutions, using the FIXED PAC metrics.

Why warm-start instead of full re-run?
---------------------------------------
v7 converged at gen 10 (best score plateaued). The fitness landscape's
optimal region is already known. With new PAC criteria, the optimum may
shift slightly but won't be in a totally different region.

Warm-start strategy:
  - Initialize DE population from v7's top-K solutions + small perturbations
  - Run 10 gens (vs v7's 30) — enough for fine-tuning
  - Total: ~10 × 160 = 1600 evals ≈ 1-2 hours

Compare to full re-run:
  - Full 20-gen rerun = 3200 evals ≈ 3-5 hours
  - Higher risk of not finishing overnight

USAGE
-----
    python warm_start_de_with_fixed_pac.py            # 10 gens warm-start
    python warm_start_de_with_fixed_pac.py --gens 5   # quick 5-gen test
    python warm_start_de_with_fixed_pac.py --top 20   # use top-20 as seeds
"""

import os
import sys
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import numba
from neurolib.utils.stimulus import OrnsteinUhlenbeckProcess

@numba.njit
def seed_numba(seed):
    np.random.seed(seed)


# ============================================================================
# Path setup
# ============================================================================
PROJECT_ROOT = Path(r"D:\Year3_Mao_Projects\sleep_loop")
V7_SCRIPT = PROJECT_ROOT / "models" / "s4_personalize_fig7_v7.py"
RECORDS_CSV = PROJECT_ROOT / "outputs" / "evolution_fig7_v7_records.csv"
OUTPUT_DIR = PROJECT_ROOT / "warm_start_de"
OUTPUT_DIR.mkdir(exist_ok=True)

if not V7_SCRIPT.exists():
    V7_SCRIPT = PROJECT_ROOT / "models" / "s4_personalize_fig7_v7.py"
if not RECORDS_CSV.exists():
    RECORDS_CSV = PROJECT_ROOT / "outputs" / "evolution_fig7_v7_records.csv"


# ============================================================================
# Import fixed PAC
# ============================================================================
sys.path.insert(0, str(Path(__file__).parent))
try:
    from compute_pac_metrics_fixed import compute_pac_metrics as compute_pac_fixed
except ImportError:
    print("ERROR: cannot import compute_pac_metrics_fixed")
    sys.exit(1)


# ============================================================================
# New PAC thresholds (must match reevaluate_v7_with_fixed_pac.py)
# ============================================================================
PAC_MI_MIN = 0.005
PAC_PHASE_TOL_DEG = 50
PAC_CONCENTRATION_MIN = 0.08
PAC_UP_DOWN_RATIO_MIN = 1.20


# ============================================================================
# Load v7 module (needed for build_model, compute_constraints, etc.)
# ============================================================================
def import_v7_module():
    spec = importlib.util.spec_from_file_location("v7_module", str(V7_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    sys.modules["v7_module"] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    return module


print("Loading v7 module...")
v7 = import_v7_module()


# ============================================================================
# New fitness function with fixed PAC
# ============================================================================
def evaluate_new_pac_constraints(r_ctx, r_thal, fs):
    """Compute T9/T10a/T10b/T11 with fixed PAC."""
    pac = compute_pac_fixed(r_ctx, r_thal, fs)
    if not pac["ok"]:
        return False, False, False, False, pac
    T9 = pac["mi"] >= PAC_MI_MIN
    phase_argmax_deg = np.degrees(pac["phase_argmax"])
    phase_dist = min(abs(phase_argmax_deg), 360 - abs(phase_argmax_deg))
    T10a = phase_dist <= PAC_PHASE_TOL_DEG
    T10b = pac["phase_concentration"] >= PAC_CONCENTRATION_MIN
    T11 = pac["up_down_ratio"] >= PAC_UP_DOWN_RATIO_MIN
    return T9, T10a, T10b, T11, pac


# Track records globally
_records = []
_eval_count = [0]


def fitness_with_fixed_pac(params_vec, target_psd, target_freqs,
                            target_periodic, fooof_freqs):
    """
    V7's fitness, but with T9/T10/T11 replaced by fixed PAC criteria.
    Same structure: 12 hard constraints + event-based rewards.
    """
    _eval_count[0] += 1
    mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th = params_vec

    # ── Run simulation ────────────────────────────────────────────────
    try:
        m = v7.build_model(mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th)
        
        # --- Bulletproof Seed Injection ---
        th = m.model_instance.nodes[1]
        th[0].seed = 42
        th[0].noise_input = [OrnsteinUhlenbeckProcess(mu=0.0, sigma=0.0, tau=5.0, seed=42)]
        seed_numba(42)
        # ----------------------------------
        
        m.run()
    except Exception:
        try:
            m.params["backend"] = "jitcdde"
            seed_numba(42)  # reseed before fallback run too
            m.run()
        except Exception:
            return v7.BAD_OBJECTIVE

    r_exc = m[f"r_mean_{v7.EXC}"]
    if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
        r_ctx = r_exc[0, :] * 1000.0
        r_thal = r_exc[1, :] * 1000.0
    else:
        r_ctx = (r_exc[0] if r_exc.ndim == 2 else r_exc) * 1000.0
        r_thal = np.zeros_like(r_ctx)

    n_drop = int(5.0 * v7.FS_SIM)
    r_ctx = r_ctx[n_drop:]
    r_thal = r_thal[n_drop:]

    if r_ctx.max() < 0.1:
        return v7.BAD_OBJECTIVE

    # ── Compute v7 constraints (T1-T8, T12) ──────────────────────────
    f_ctx_full, p_ctx_full = v7.compute_epoch_psd(r_ctx, v7.FS_SIM)
    n_passed_v7, con = v7.compute_constraints_v7(
        r_ctx, r_thal, f_c=f_ctx_full, p_c=p_ctx_full, fs=v7.FS_SIM
    )

    # ── Compute new PAC (T9/T10a/T10b/T11) ───────────────────────────
    T9_new, T10a_new, T10b_new, T11_new, pac_new = (
        evaluate_new_pac_constraints(r_ctx, r_thal, v7.FS_SIM)
    )

    # ── Combine: T1-T8 + T12 from v7, T9-T11 from new ────────────────
    n_passed_new = sum([
        con.get("T1", False), con.get("T2", False),
        con.get("T3", False), con.get("T4", False),
        con.get("T5", False), con.get("T6", False),
        con.get("T7", False), con.get("T8", False),
        T9_new,
        T10a_new and T10b_new,
        T11_new,
        con.get("T12", False),
    ])
    feasible = (n_passed_new == 12)

    # ── Compute fitness ───────────────────────────────────────────────
    if feasible:
        # Same shape_r as v7 (FOOOF-based or chi2 fallback)
        if v7.HAS_FOOOF and target_periodic is not None:
            try:
                from scipy.interpolate import interp1d
                from fooof import FOOOF
                from scipy.stats import pearsonr
                F_LO, F_HI = v7.F_LO, v7.F_HI
                mask = (f_ctx_full >= F_LO) & (f_ctx_full <= F_HI)
                f_ctx, p_ctx = f_ctx_full[mask], p_ctx_full[mask]
                p_interp = interp1d(f_ctx, p_ctx, bounds_error=False,
                                     fill_value=1e-30)(fooof_freqs)
                fm_sim = FOOOF(**v7.EVO_FOOOF_PARAMS)
                fm_sim.fit(fooof_freqs, p_interp, [F_LO, F_HI])
                sim_log = np.log10(p_interp[:len(fm_sim._ap_fit)] + 1e-30)
                sim_periodic = sim_log - fm_sim._ap_fit
                n_r = min(len(sim_periodic), len(target_periodic))
                shape_r, _ = pearsonr(sim_periodic[:n_r], target_periodic[:n_r])
                shape_r = max(shape_r, 0.0)
            except Exception:
                shape_r = 0.0
        else:
            shape_r = 0.0

        t4_q = con.get("T4_q", 0.0)
        so_power = float(np.clip((t4_q - 1.0) / 4.0, 0.0, 1.0))
        n_ver = con.get("T12_n_verified", 0)
        spindle_power = float(np.clip(n_ver / 15.0, 0.0, 1.0))

        fitness = (v7.W_SHAPE * shape_r
                   + v7.W_SO * so_power
                   + v7.W_SPINDLE * spindle_power)
    else:
        # Soft penalty (continuous gradient on infeasible)
        c_scores = v7.compute_feasibility_score(con)
        # Override the PAC scores with new ones
        # (otherwise we'd use the old broken PAC scores)
        n_pac_new = sum([T9_new, T10a_new and T10b_new, T11_new])
        n_other = sum([con.get(f"T{i}", False) for i in [1,2,3,4,5,6,7,8,12]])
        fitness = -10.0 + 10.0 * (n_other + n_pac_new) / 12.0

    # ── Record ────────────────────────────────────────────────────────
    record = {
        "mue": mue, "mui": mui, "b": b, "tauA": tauA,
        "g_LK": g_lk, "g_h": g_h,
        "c_th2ctx": c_th2ctx, "c_ctx2th": c_ctx2th,
        "score": round(fitness, 6),
        "feasible": int(feasible),
        "n_passed": n_passed_new,
        "T9_new": int(T9_new),
        "T10a_new": int(T10a_new),
        "T10b_new": int(T10b_new),
        "T11_new": int(T11_new),
        "T9_mi": pac_new.get("mi", 0.0),
        "T10a_phase_argmax_deg": np.degrees(pac_new.get("phase_argmax", np.pi))
                                  if pac_new.get("ok") else 180.0,
        "T10b_concentration": pac_new.get("phase_concentration", 0.0),
        "T11_up_down_ratio": pac_new.get("up_down_ratio", 0.0),
        "pac_bimodal": int(pac_new.get("bimodality_flag", False))
                       if pac_new.get("ok") else 0,
        "shape_r": shape_r if feasible else 0.0,
        "T4_q": con.get("T4_q", 0.0),
        "T12_n_verified": con.get("T12_n_verified", 0),
        "eval": _eval_count[0],
    }
    _records.append(record)

    # Save records every 50 evals (in case crash)
    if _eval_count[0] % 50 == 0:
        pd.DataFrame(_records).to_csv(
            OUTPUT_DIR / "warm_start_records.csv", index=False
        )

    # DE minimizes; we want to maximize, so return -fitness
    return -fitness


# ============================================================================
# Build initial population from v7's top-K
# ============================================================================
def build_warm_start_init(top_k=10, popsize=20, n_params=8, seed=42):
    """
    Construct DE initial population:
      - First top_k members: v7's top-K solutions exactly
      - Remaining popsize*n_params - top_k members: v7's top solutions
        + small Gaussian perturbations
    """
    df = pd.read_csv(RECORDS_CSV)
    feasible = df[df["feasible"] == 1].sort_values("score", ascending=False)
    print(f"v7 records: {len(df)} total, {len(feasible)} feasible")

    PARAM_NAMES = ["mue", "mui", "b", "tauA", "g_LK", "g_h",
                    "c_th2ctx", "c_ctx2th"]
    seeds = feasible[PARAM_NAMES].head(top_k).values
    print(f"Using top {top_k} v7 solutions as seeds (scores: "
          f"{feasible['score'].head(top_k).tolist()})")

    n_total = popsize * n_params  # DE init size
    n_seeds_used = min(top_k, n_total)
    n_perturb = n_total - n_seeds_used

    init = np.zeros((n_total, n_params))
    init[:n_seeds_used] = seeds[:n_seeds_used]

    # Perturb the best solution to fill remaining slots
    rng = np.random.default_rng(seed)
    bounds_arr = np.array(v7.BOUNDS)
    bounds_lo = bounds_arr[:, 0]
    bounds_hi = bounds_arr[:, 1]
    bounds_range = bounds_hi - bounds_lo

    base = seeds[0]  # best v7 solution
    for i in range(n_perturb):
        # 2% perturbation of bound range
        perturb = 0.02 * bounds_range * rng.standard_normal(n_params)
        new_pt = base + perturb
        new_pt = np.clip(new_pt, bounds_lo, bounds_hi)
        init[n_seeds_used + i] = new_pt

    return init


# ============================================================================
# Main warm-start optimization
# ============================================================================
def main(args):
    print("=" * 72)
    print("WARM-START DE with fixed PAC")
    print("=" * 72)
    print(f"v7 script:    {V7_SCRIPT}")
    print(f"Records CSV:  {RECORDS_CSV}")
    print(f"Output dir:   {OUTPUT_DIR}")
    print(f"\nGenerations:  {args.gens}")
    print(f"Top-K seeds:  {args.top}")
    print(f"DE popsize:   {args.popsize} × {len(v7.PARAM_NAMES)} params = "
          f"{args.popsize * len(v7.PARAM_NAMES)} individuals/gen")
    print(f"Total evals:  ~{args.popsize * len(v7.PARAM_NAMES) * (args.gens+1)}")

    # Load target PSD (FOOOF target)
    print("\nLoading target PSD...")
    target_psd, target_freqs = v7.load_target_psd()
    target_periodic, fooof_freqs = v7.compute_target_periodic(
        target_psd, target_freqs)

    # Build warm-start init
    print("\nBuilding warm-start initial population...")
    init = build_warm_start_init(
        top_k=args.top,
        popsize=args.popsize,
        n_params=len(v7.PARAM_NAMES),
        seed=42,
    )

    print(f"\nStarting DE for {args.gens} generations...")
    print("=" * 72)

    result = differential_evolution(
        fitness_with_fixed_pac,
        bounds=v7.BOUNDS,
        args=(target_psd, target_freqs, target_periodic, fooof_freqs),
        init=init,
        maxiter=args.gens,
        popsize=args.popsize,
        strategy="best1bin",
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=0.0001,
        seed=42,
        polish=False,
        workers=1,
        disp=True,
    )

    print("\n" + "=" * 72)
    print("DE COMPLETE")
    print("=" * 72)
    print(f"Best score (negated): {-result.fun:.4f}")
    print(f"Best params:")
    for k, v in zip(v7.PARAM_NAMES, result.x):
        print(f"  {k} = {v:.4f}")

    # Final save
    df_records = pd.DataFrame(_records)
    final_csv = OUTPUT_DIR / "warm_start_records.csv"
    df_records.to_csv(final_csv, index=False)
    print(f"\nFull records: {final_csv}")

    # Best feasible
    feas = df_records[df_records["feasible"] == 1]
    if len(feas) > 0:
        feas_sorted = feas.sort_values("score", ascending=False)
        print(f"\nTop 5 feasible solutions found by warm-start DE:")
        for rank, (_, r) in enumerate(feas_sorted.head(5).iterrows(), 1):
            print(f"  #{rank} score={r['score']:.4f} "
                  f"phase_argmax={r['T10a_phase_argmax_deg']:+.1f}° "
                  f"up_down_ratio={r['T11_up_down_ratio']:.2f} "
                  f"bimodal={bool(r['pac_bimodal'])}")

        # Save best params
        best = feas_sorted.iloc[0]
        params_path = OUTPUT_DIR / "patient_params_warm_start.json"
        import json
        with open(params_path, "w") as f:
            json.dump({
                "params": {k: float(best[k]) for k in v7.PARAM_NAMES},
                "score": float(best["score"]),
                "shape_r": float(best["shape_r"]),
                "T4_q": float(best["T4_q"]),
                "T12_n_verified": int(best["T12_n_verified"]),
                "PAC": {
                    "mi": float(best["T9_mi"]),
                    "phase_argmax_deg": float(best["T10a_phase_argmax_deg"]),
                    "concentration": float(best["T10b_concentration"]),
                    "up_down_ratio": float(best["T11_up_down_ratio"]),
                    "bimodal": bool(best["pac_bimodal"]),
                },
            }, f, indent=2)
        print(f"\nBest params saved to: {params_path}")
    else:
        print("\n[!] No feasible solutions found — try more gens or relax thresholds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=10,
                        help="Number of DE generations (default: 10)")
    parser.add_argument("--top", type=int, default=10,
                        help="Use top-K v7 solutions as seeds (default: 10)")
    parser.add_argument("--popsize", type=int, default=20,
                        help="DE popsize multiplier (default: 20, same as v7)")
    args = parser.parse_args()
    main(args)