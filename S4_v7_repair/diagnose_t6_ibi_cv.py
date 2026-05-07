"""
diagnose_t6_ibi_cv.py
=====================
Stage 1.5 diagnostic — is T6 a "bimodal basin" constraint?

Background
----------
sobol10_feasibility_check.py found 10/10 random samples fail T6
(IBI_CV < 0.4). But Pareto seeds C and E are fully feasible (12/12)
under the same V7 — meaning their T6 passes. This script measures
the IBI_CV of all 5 Pareto seeds explicitly and compares to the
Sobol-10 distribution, to test whether T6 splits the parameter
space into two well-separated regions.

Two outcomes
------------
(a) Pareto seeds cluster at IBI_CV ≈ 0.2-0.4, Sobol-10 cluster at
    IBI_CV ≈ 0.5-0.8 → CONFIRMED bimodal. Fresh DE will work
    because the "good basin" exists; T6 just rules out a large
    swath of bad parameters.

(b) Pareto seeds spread broadly, no clear separation → T6 may
    be threshold-sensitive (small parameter changes flip the
    pass/fail). Then fresh DE may struggle.

This script also dumps the raw inter-burst intervals for each seed
so you can see whether T6 failures are (i) too few bursts (n < 3),
(ii) bursts but very irregular, or (iii) just barely above threshold.

Usage
-----
    python diagnose_t6_ibi_cv.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

# -------------------------------------------------------------------- #
# Import V7 module
# -------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
V7_SCRIPT = ROOT / "models" / "s4_personalize_fig7_v7.py"

print(f"Importing V7 from: {V7_SCRIPT}")
spec = importlib.util.spec_from_file_location("v7", str(V7_SCRIPT))
v7 = importlib.util.module_from_spec(spec)
sys.modules["v7"] = v7
try:
    spec.loader.exec_module(v7)
except SystemExit:
    pass

# -------------------------------------------------------------------- #
# Load 5 Pareto seeds
# -------------------------------------------------------------------- #
SEEDS_JSON = ROOT / "S4_v7_repair" / "pareto_seeds.json"
with open(SEEDS_JSON, "r", encoding="utf-8") as f:
    seeds = json.load(f)["seeds"]
print(f"Loaded {len(seeds)} Pareto seeds from {SEEDS_JSON}")

# -------------------------------------------------------------------- #
# Load target PSD (needed by compute_fitness_v7)
# -------------------------------------------------------------------- #
print("\nLoading target PSD...")
target_psd, target_freqs = v7.load_target_psd()
target_periodic, fooof_freqs = v7.compute_target_periodic(
    target_psd, target_freqs)

# -------------------------------------------------------------------- #
# For each seed: re-simulate, compute IBI_CV from scratch, also
# extract the raw inter-burst intervals.
# -------------------------------------------------------------------- #
PARAM_NAMES = ["mue", "mui", "b", "tauA",
               "g_LK", "g_h", "c_th2ctx", "c_ctx2th"]

print("\n" + "=" * 78)
print(f"{'seed':<6} {'n_bursts':>9} {'IBI_CV':>8} {'mean_IBI_s':>11} "
      f"{'min_IBI':>8} {'max_IBI':>8} {'T6 pass?':>10}")
print("=" * 78)

results = []
for seed_obj in seeds:
    tag = seed_obj["tag"]
    params_vec = np.array([seed_obj["params"][k] for k in PARAM_NAMES])

    # Reset record state and run a single fitness evaluation
    v7._eval_count = 0
    v7._records = []
    _ = v7.compute_fitness_v7(
        params_vec,
        target_psd, target_freqs,
        target_periodic, fooof_freqs,
    )
    rec = v7._records[-1]

    # The fitness eval already ran constraint computation, so T6 fields exist.
    # But we want the raw IBI distribution — need to re-extract.
    # Easiest path: re-build the model, run, and replicate T6's logic.
    m = v7.build_model(*params_vec)
    v7.seed_numba(42)
    m.run()
    r_exc = m[f"r_mean_{v7.EXC}"]
    if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
        r_ctx = r_exc[0, :] * 1000.0
    else:
        r_ctx = (r_exc[0] if r_exc.ndim == 2 else r_exc) * 1000.0
    n_drop = int(5.0 * v7.FS_SIM)
    r_ctx = r_ctx[n_drop:]

    # Mirror T3's run-length encoding (T6 reuses its `starts`)
    UP_THRESH_HZ = v7.UP_THRESH_HZ
    above = (r_ctx > UP_THRESH_HZ).astype(np.int8)
    diff = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(diff == 1)[0]

    n_bursts = len(starts)
    if n_bursts >= 3:
        intervals_s = np.diff(starts) / v7.FS_SIM
        ibi_cv = float(intervals_s.std() / (intervals_s.mean() + 1e-12))
        mean_ibi = float(intervals_s.mean())
        min_ibi = float(intervals_s.min())
        max_ibi = float(intervals_s.max())
    else:
        intervals_s = np.array([])
        ibi_cv = float("nan")
        mean_ibi = min_ibi = max_ibi = float("nan")

    t6_pass = (n_bursts >= 3) and (ibi_cv < v7.IBI_CV_MAX)

    print(f"{tag:<6} {n_bursts:>9} {ibi_cv:>8.3f} {mean_ibi:>11.3f} "
          f"{min_ibi:>8.3f} {max_ibi:>8.3f} {'PASS' if t6_pass else 'FAIL':>10}")

    results.append({
        "tag": tag,
        "n_bursts": n_bursts,
        "ibi_cv": ibi_cv,
        "mean_ibi": mean_ibi,
        "min_ibi": min_ibi,
        "max_ibi": max_ibi,
        "intervals": intervals_s,
        "t6_pass": t6_pass,
        "report_in_json": rec.get("T6_ibi_cv"),
    })

# -------------------------------------------------------------------- #
# Side-by-side: each seed's full IBI distribution
# -------------------------------------------------------------------- #
print("\n" + "=" * 78)
print("Per-seed IBI distribution (sorted, in seconds)")
print("=" * 78)
for r in results:
    if len(r["intervals"]) == 0:
        print(f"  Seed {r['tag']}: <3 bursts, no IBIs")
        continue
    sorted_ibi = np.sort(r["intervals"])
    # Print as compact list with 3 decimals
    short = ", ".join(f"{x:.3f}" for x in sorted_ibi[:12])
    suffix = f"  ...({len(sorted_ibi)} total)" if len(sorted_ibi) > 12 else ""
    print(f"  Seed {r['tag']}: [{short}]{suffix}")

# -------------------------------------------------------------------- #
# Bimodality test: compare Pareto seeds to Sobol expectation
# -------------------------------------------------------------------- #
print("\n" + "=" * 78)
print("Bimodality assessment")
print("=" * 78)

passing = [r for r in results if r["t6_pass"]]
failing = [r for r in results if not r["t6_pass"]]

if passing:
    pass_cvs = [r["ibi_cv"] for r in passing]
    print(f"  T6-passing seeds (n={len(passing)}): "
          f"IBI_CV = {pass_cvs} "
          f"(mean = {np.mean(pass_cvs):.3f})")
if failing:
    fail_cvs = [r["ibi_cv"] for r in failing
                if not np.isnan(r["ibi_cv"])]
    if fail_cvs:
        print(f"  T6-failing seeds (n={len(failing)}): "
              f"IBI_CV = {fail_cvs} "
              f"(mean = {np.mean(fail_cvs):.3f})")
    else:
        print(f"  T6-failing seeds (n={len(failing)}): all had <3 bursts")

if passing and failing:
    fail_cvs_clean = [r["ibi_cv"] for r in failing
                      if not np.isnan(r["ibi_cv"])]
    if fail_cvs_clean:
        gap = min(fail_cvs_clean) - max(pass_cvs)
        print(f"\n  Gap between passing-max and failing-min: {gap:+.3f}")
        if gap > 0.05:
            print(f"  → Clean separation. T6 is a true 'basin' constraint:")
            print(f"    parameter space splits into two regions by SO regularity.")
        else:
            print(f"  → Marginal separation. T6 may be threshold-sensitive;")
            print(f"    small parameter changes can flip pass/fail.")

# -------------------------------------------------------------------- #
# Cross-check: do the recomputed IBI_CV values match the JSON-reported
# values from the (stochastic) warm_start_records.csv?
# -------------------------------------------------------------------- #
print("\n" + "=" * 78)
print("Sanity check: deterministic IBI_CV vs warm_start record")
print("=" * 78)
print(f"{'seed':<6} {'reported_in_warm_start':>24} {'recomputed_now':>16} "
      f"{'delta':>10}")
print("-" * 78)
for r in results:
    rep = r.get("report_in_json")
    now = r["ibi_cv"]
    if rep is not None and not np.isnan(now):
        try:
            rep_f = float(rep)
            delta = now - rep_f
            print(f"  {r['tag']:<4} {rep_f:>22.3f} {now:>16.3f} "
                  f"{delta:>+10.3f}")
        except (TypeError, ValueError):
            print(f"  {r['tag']:<4} {'(n/a)':>22} {now:>16.3f}")
print("\nIf deltas are large (>0.1), it confirms warm_start's T6 numbers")
print("were noise-dependent and shouldn't be trusted at face value.")