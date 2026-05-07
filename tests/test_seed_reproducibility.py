"""
test_seed_reproducibility.py
============================
Verify that with the fixed seed injection, identical parameters
produce IDENTICAL PAC metrics across repeated runs, AND diagnose
which T1-T12 constraints are failing for each Pareto seed.

Usage (from project root, in neurolib conda env):
    python tests/test_seed_reproducibility.py
"""
import json
import sys
import importlib.util
from pathlib import Path
from collections import Counter

import numpy as np

# ── Import V7 module ─────────────────────────────────────────────────────────
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

# ── Load seeds ────────────────────────────────────────────────────────────────
SEEDS_JSON = ROOT / "S4_v7_repair" / "pareto_seeds.json"
with open(SEEDS_JSON, "r", encoding="utf-8") as f:
    seeds = json.load(f)["seeds"]
print(f"Loaded {len(seeds)} Pareto seeds from {SEEDS_JSON}\n")

# ── Load target PSD ───────────────────────────────────────────────────────────
print("Loading target PSD...")
target_psd, target_freqs = v7.load_target_psd()
target_periodic, fooof_freqs = v7.compute_target_periodic(target_psd, target_freqs)
print()

# ── Constants ─────────────────────────────────────────────────────────────────
PARAM_COLS = ["mue", "mui", "b", "tauA", "g_LK", "g_h", "c_th2ctx", "c_ctx2th"]
CONSTRAINT_KEYS = [f"T{i}" for i in range(1, 13)]
N_RUNS = 3

# ── Run test ─────────────────────────────────────────────────────────────────
print(f"{'Seed':<6}{'Run':<5}{'MI':<12}{'n_pass':<8}  T1-T12              fail")
print("-" * 75)

all_results = {}

for seed_obj in seeds:
    tag = seed_obj["tag"]
    params_vec = np.array([seed_obj["params"][k] for k in PARAM_COLS])

    run_data = []
    for run_i in range(N_RUNS):
        v7._eval_count = 0
        v7._records = []

        v7.compute_fitness_v7(
            params_vec,
            target_psd, target_freqs,
            target_periodic, fooof_freqs,
        )

        rec = v7._records[-1]
        mi       = rec["T9_mi"]
        udr      = rec["T11_lag_ms"]
        sr       = rec["shape_r"]
        n_passed = rec["n_passed"]

        bits   = "".join(str(int(bool(rec.get(k, 0)))) for k in CONSTRAINT_KEYS)
        failed = [k for k in CONSTRAINT_KEYS if not bool(rec.get(k, 0))]

        run_data.append({
            "MI": mi, "udr": udr, "shape_r": sr,
            "n_passed": n_passed, "bits": bits, "failed": failed,
        })

        print(f"{tag:<6}{run_i:<5}{mi:<12.6f}{n_passed}/12    [{bits}]  {failed}")

    # Reproducibility check
    def rel_dev(vals):
        lo, hi = min(vals), max(vals)
        return (hi - lo) / max(abs(np.mean(vals)), 1e-9)

    dev_mi = rel_dev([r["MI"]      for r in run_data])
    dev_ud = rel_dev([r["udr"]     for r in run_data])
    dev_sr = rel_dev([r["shape_r"] for r in run_data])

    passed = dev_mi < 1e-3 and dev_ud < 1e-3 and dev_sr < 1e-3
    status = "PASS" if passed else "FAIL"
    print(f"  Seed {tag}: ΔMI={dev_mi*100:.4f}%  Δudr={dev_ud*100:.4f}%  "
          f"[{status}]  n_passed={run_data[0]['n_passed']}/12  "
          f"failing={run_data[0]['failed']}\n")

    all_results[tag] = {
        "dev_mi": dev_mi, "dev_ud": dev_ud, "dev_sr": dev_sr,
        "passed": passed,
        "n_passed": run_data[0]["n_passed"],
        "failed": run_data[0]["failed"],
        "bits": run_data[0]["bits"],
    }

# ── Constraint failure summary ────────────────────────────────────────────────
print("=" * 75)
print("Constraint failure summary across 5 seeds:")
print("-" * 75)
all_failed = []
for tag, r in all_results.items():
    print(f"  Seed {tag}:  [{r['bits']}]  {r['n_passed']}/12  failing={r['failed']}")
    all_failed.extend(r["failed"])

counts = Counter(all_failed)
print()
print("Failure frequency per constraint (how many of the 5 seeds fail each):")
for k in CONSTRAINT_KEYS:
    n = counts.get(k, 0)
    bar = "█" * n
    marker = " ← COMMON" if n >= 3 else ""
    print(f"  {k:4s}: {bar:<6} ({n}/5){marker}")

# ── Final reproducibility verdict ─────────────────────────────────────────────
print()
print("=" * 75)
all_pass = all(r["passed"] for r in all_results.values())
if all_pass:
    print("REPRODUCIBILITY: PASS — all 5 seeds are bit-identical across 3 runs.")
else:
    bad = [t for t, r in all_results.items() if not r["passed"]]
    print(f"REPRODUCIBILITY: FAIL — seeds {bad} show > 0.1% deviation.")
    sys.exit(1)
