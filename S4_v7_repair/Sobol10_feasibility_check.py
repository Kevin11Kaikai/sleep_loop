"""
sobol10_feasibility_check.py
============================
Stage 1.5 sanity check — fast feasibility probe of seed-locked V7.

Purpose
-------
Before committing 1-2 hours to re-running warm_start_de, do a 1-2 minute
sanity check: sample 10 Sobol-distributed points from the V7 parameter
bounds, evaluate each through compute_fitness_v7, and count how many
satisfy the 12 deterministic constraints.

Decision rule
-------------
  ≥ 2 feasible / 10  →  GO    (thresholds reasonable, run warm_start)
  0 - 1 feasible / 10 →  STOP  (thresholds too strict OR landscape changed,
                                investigate specific failing constraints
                                before re-running)

Why Sobol and not random
------------------------
Sobol gives near-uniform coverage of an 8D box with N=10 — a random sample
of size 10 in 8D is heavily clumpy. The 10 Sobol points are guaranteed
to span all corners of the parameter box.

Why 10 points and not 100
-------------------------
This is a Go/No-Go signal, not a population-level estimate. If the
ground-truth feasibility rate is ≥ 10%, you should see ≥ 1 feasible in
10 samples with > 65% probability. If it's ≥ 30%, you should see ≥ 2
feasible with > 85% probability. 10 samples is enough to distinguish
"healthy DE landscape" from "thresholds too strict".

Usage
-----
    python sobol10_feasibility_check.py

Time
----
~1-2 minutes (10 evaluations × ~6-8s each, including numba JIT warm-up).
"""

import sys
import importlib.util
from pathlib import Path

import numpy as np
from scipy.stats import qmc

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
# Generate Sobol samples in V7's parameter bounds
# -------------------------------------------------------------------- #
N_SAMPLES = 10
PARAM_NAMES = ["mue", "mui", "b", "tauA",
               "g_LK", "g_h", "c_th2ctx", "c_ctx2th"]

bounds_lo = np.array([b[0] for b in v7.BOUNDS])
bounds_hi = np.array([b[1] for b in v7.BOUNDS])

# Use Sobol sequence for near-uniform coverage in 8D box
sampler = qmc.Sobol(d=8, seed=2024)
samples_unit = sampler.random(n=N_SAMPLES)
samples = qmc.scale(samples_unit, bounds_lo, bounds_hi)

print(f"\nGenerated {N_SAMPLES} Sobol samples in 8D parameter box")
print(f"Bounds:")
for name, lo, hi in zip(PARAM_NAMES, bounds_lo, bounds_hi):
    print(f"  {name:10s} [{lo:.4f}, {hi:.4f}]")

# -------------------------------------------------------------------- #
# Load target PSD (needed by compute_fitness_v7)
# -------------------------------------------------------------------- #
print("\nLoading target PSD...")
target_psd, target_freqs = v7.load_target_psd()
target_periodic, fooof_freqs = v7.compute_target_periodic(
    target_psd, target_freqs)

# -------------------------------------------------------------------- #
# Evaluate each sample
# -------------------------------------------------------------------- #
print("\n" + "=" * 78)
print(f"{'#':>3} {'feas':>5} {'n/12':>5} {'T1-T12':<14} {'failing':<22} "
      f"{'MI':>8} {'udr':>6} {'shape_r':>8}")
print("=" * 78)

results = []
for i in range(N_SAMPLES):
    params_vec = samples[i]
    # Reset global record state so we can read this evaluation's record
    v7._eval_count = 0
    v7._records = []

    score = v7.compute_fitness_v7(
        params_vec,
        target_psd, target_freqs,
        target_periodic, fooof_freqs,
    )

    rec = v7._records[-1] if v7._records else {}

    n_passed = rec.get("n_passed", 0)
    feasible = rec.get("feasible", False)
    t_str = "".join("1" if rec.get(f"T{k}", False) else "0"
                    for k in range(1, 13))
    failing = [f"T{k}" for k in range(1, 13)
               if not rec.get(f"T{k}", False)]
    failing_str = ",".join(failing) if failing else "—"

    mi = rec.get("T9_mi", 0.0)
    udr = rec.get("T11_lag_ms", 0.0)  # T11_lag_ms now stores up_down_ratio
    sr = rec.get("shape_r", 0.0)

    print(f"{i:>3} {'YES' if feasible else 'no':>5} {n_passed:>2}/12 "
          f"[{t_str}] {failing_str:<22} "
          f"{mi:>8.5f} {udr:>6.3f} {sr:>8.4f}")

    results.append({
        "feasible": feasible,
        "n_passed": n_passed,
        "failing": failing,
        "mi": mi,
        "udr": udr,
        "shape_r": sr,
    })

# -------------------------------------------------------------------- #
# Summary statistics
# -------------------------------------------------------------------- #
print("\n" + "=" * 78)
print("Summary")
print("=" * 78)

n_feasible = sum(r["feasible"] for r in results)
n_passes = [r["n_passed"] for r in results]
print(f"Feasible solutions:  {n_feasible} / {N_SAMPLES}")
print(f"n_passed distribution: "
      f"min={min(n_passes)}, median={int(np.median(n_passes))}, "
      f"max={max(n_passes)}, mean={np.mean(n_passes):.1f}")

# Per-constraint failure rate
print(f"\nPer-constraint failure count (out of {N_SAMPLES}):")
for k in range(1, 13):
    n_fail = sum(1 for r in results if f"T{k}" in r["failing"])
    bar = "█" * n_fail + "·" * (N_SAMPLES - n_fail)
    print(f"  T{k:<2}: {bar}  ({n_fail}/{N_SAMPLES})")

# -------------------------------------------------------------------- #
# Go / No-Go decision
# -------------------------------------------------------------------- #
print("\n" + "=" * 78)
print("Decision")
print("=" * 78)

if n_feasible >= 2:
    print(f"  GO — {n_feasible}/10 feasible suggests landscape is healthy.")
    print(f"  Recommend: re-run warm_start_de_with_fixed_pac.py with "
          f"the now-deterministic V7.")
elif n_feasible == 1:
    print(f"  MARGINAL — only 1/10 feasible. DE should still find more,")
    print(f"  but convergence may be slow. Investigate top failing "
          f"constraints before re-running.")
    most_common_fail = max(range(1, 13),
                            key=lambda k: sum(1 for r in results
                                              if f"T{k}" in r["failing"]))
    print(f"  Most common failure: T{most_common_fail}")
else:
    print(f"  STOP — 0/10 feasible. Threshold(s) likely too strict for ")
    print(f"  seed-locked V7. Do NOT re-run warm_start yet.")
    # Identify the dominating failure
    fail_counts = [(k, sum(1 for r in results if f"T{k}" in r["failing"]))
                   for k in range(1, 13)]
    fail_counts.sort(key=lambda x: -x[1])
    print(f"  Dominant failures:")
    for k, c in fail_counts[:3]:
        if c > 0:
            print(f"    T{k}: {c}/10 samples fail")
    print(f"  Suggest: relax the dominant-failing constraint's threshold,")
    print(f"  or analyze why it became hard under deterministic noise.")

print("=" * 78)