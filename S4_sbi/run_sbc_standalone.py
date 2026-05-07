"""
run_sbc_standalone.py
=====================
Standalone SBC diagnostic for Stage 2 SBI.

Loads round3_posterior.pkl (the final posterior) and runs 200-sample SBC
using sbi.diagnostics.sbc.run_sbc (correct path for sbi 0.26.1).

Reports KS p-values per parameter and saves fig_sbc.png.
"""

import sys
import os
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CWD = project root ────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
os.chdir(str(_ROOT))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

# ── NumPy alias shim ─────────────────────────────────────────────────────────
import numpy as np
import builtins as _builtins_mod
for _alias in ("int", "float", "bool", "object", "complex", "str"):
    if not hasattr(np, _alias):
        setattr(np, _alias, getattr(_builtins_mod, _alias))

# ── Local neurolib ────────────────────────────────────────────────────────────
_LOCAL_NEUROLIB = Path(r"D:\Year3_Mao_Projects\neurolib")
if _LOCAL_NEUROLIB.is_dir() and str(_LOCAL_NEUROLIB) not in sys.path:
    sys.path.insert(0, str(_LOCAL_NEUROLIB))

print("[sbc] Importing simulator_wrapper (loads target PSD) ...")
from simulator_wrapper import simulator, SUMMARY_KEYS

import torch
from torch import tensor
from sbi.utils import BoxUniform
from sbi.diagnostics.sbc import run_sbc
from sbi.analysis.plot import sbc_rank_plot
from scipy.stats import kstest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
PARAMS      = ["g_h", "g_LK", "c_ctx2th", "b"]
PRIOR_LOW   = tensor([0.035, 0.020, 0.05, 28.4])
PRIOR_HIGH  = tensor([0.095, 0.070, 0.22, 42.6])
N_SBC       = 200
N_POST_SAMP = 1000
OUTPUT_DIR  = _SCRIPT_DIR / "sbi_outputs"
POST_PKL    = OUTPUT_DIR / "round3_posterior.pkl"
SBC_FIG     = OUTPUT_DIR / "fig_sbc.png"

prior = BoxUniform(low=PRIOR_LOW, high=PRIOR_HIGH)

# ── Load posterior ────────────────────────────────────────────────────────────
print(f"[sbc] Loading posterior from {POST_PKL} ...")
with open(str(POST_PKL), "rb") as f:
    posterior = pickle.load(f)
print(f"[sbc] Posterior type: {type(posterior).__name__}")

# Load x_obs so posterior can be conditioned
x_obs_data = np.load(str(_SCRIPT_DIR / "x_obs.npz"), allow_pickle=True)
x_obs_vals = x_obs_data["values"].astype(np.float32)
x_obs_t    = tensor(x_obs_vals).unsqueeze(0)
print(f"[sbc] x_obs loaded: {x_obs_vals.tolist()}")
posterior.set_default_x(x_obs_t)

# ── Step 1: Sample 200 theta from prior, simulate ────────────────────────────
print(f"\n[sbc] Sampling {N_SBC} theta from prior and simulating ...")
thetas_sbc = prior.sample((N_SBC,))
xs_list    = []
n_nan      = 0

import time
t0 = time.time()
for i in range(N_SBC):
    x = simulator(thetas_sbc[i].numpy())
    xs_list.append(x)
    if np.isnan(x).any():
        n_nan += 1
    if (i + 1) % 50 == 0:
        elapsed = (time.time() - t0) / 60
        eta     = elapsed / (i + 1) * (N_SBC - i - 1)
        print(f"  {i+1}/{N_SBC}  NaN={n_nan}  elapsed={elapsed:.1f}min  ETA={eta:.1f}min")

xs_sbc   = torch.tensor(np.stack(xs_list), dtype=torch.float32)
valid    = ~torch.isnan(xs_sbc).any(dim=1)
n_valid  = int(valid.sum())
print(f"\n[sbc] Valid: {n_valid}/{N_SBC}  NaN discarded: {N_SBC - n_valid}")
thetas_sbc = thetas_sbc[valid]
xs_sbc     = xs_sbc[valid]

# ── Step 2: Run SBC ───────────────────────────────────────────────────────────
print(f"\n[sbc] Running run_sbc ({N_POST_SAMP} posterior samples per theta) ...")
ranks, dap_samples = run_sbc(
    thetas_sbc, xs_sbc, posterior,
    num_posterior_samples=N_POST_SAMP,
    num_workers=1,
    show_progress_bar=True,
)
print(f"[sbc] ranks shape: {ranks.shape}")

# ── Step 3: KS test ───────────────────────────────────────────────────────────
print("\n[sbc] KS test results (uniform null — p > 0.05 = PASS):")
print(f"  {'Parameter':<12}  {'KS stat':>8}  {'p-value':>10}  {'Result':>6}")
print("  " + "-" * 44)
ks_results = {}
for i, param in enumerate(PARAMS):
    r = ranks[:, i].float().numpy()
    r_norm = r / (N_POST_SAMP + 1)          # normalise to [0,1]
    stat, pval = kstest(r_norm, "uniform")
    result = "PASS" if pval >= 0.05 else "FAIL"
    ks_results[param] = {"stat": stat, "pval": pval, "result": result}
    print(f"  {param:<12}  {stat:8.4f}  {pval:10.4f}  {result:>6}")

# ── Step 4: Rank histogram shape ─────────────────────────────────────────────
print("\n[sbc] Rank histogram shape assessment:")
for param in PARAMS:
    r = ranks[:, PARAMS.index(param)].float().numpy()
    n_bins = 10
    counts, _ = np.histogram(r, bins=n_bins, range=(0, N_POST_SAMP))
    expected  = n_valid / n_bins
    lo_frac   = counts[:2].mean() / expected   # first 2 bins (low ranks)
    hi_frac   = counts[-2:].mean() / expected  # last 2 bins (high ranks)
    mid_frac  = counts[4:6].mean() / expected  # middle 2 bins
    if lo_frac > 1.4 and hi_frac > 1.4:
        shape = "U-shape  (over-confident / posterior too narrow)"
    elif mid_frac > 1.4:
        shape = "inv-U    (under-confident / posterior too wide)"
    elif ks_results[param]["result"] == "PASS":
        shape = "uniform  (well-calibrated)"
    else:
        shape = "non-uniform (inspect fig_sbc.png)"
    print(f"  {param:<12}  {shape}")

# ── Step 5: Plot and save ────────────────────────────────────────────────────
print(f"\n[sbc] Saving rank-histogram plot to {SBC_FIG} ...")
try:
    fig, axes = sbc_rank_plot(
        ranks,
        num_posterior_samples=N_POST_SAMP,
        parameter_labels=PARAMS,
        num_cols=2,
    )
    fig.suptitle("SBC rank histograms — Stage 2 posterior (R3)", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(SBC_FIG), dpi=120, bbox_inches="tight")
    plt.close(fig)
    if SBC_FIG.exists():
        print(f"  fig_sbc.png saved ({SBC_FIG.stat().st_size // 1024} KB) ✓")
    else:
        print("  ERROR: fig_sbc.png was not written despite no exception")
except Exception as e:
    print(f"  [warn] sbc_rank_plot failed ({e}); saving manual histogram instead")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for i, (ax, param) in enumerate(zip(axes, PARAMS)):
        ax.hist(ranks[:, i].numpy(), bins=20, color="steelblue", edgecolor="white")
        ax.axhline(n_valid / 20, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{param}\np={ks_results[param]['pval']:.3f}")
        ax.set_xlabel("rank")
    fig.suptitle("SBC rank histograms — Stage 2 posterior (R3)")
    fig.tight_layout()
    fig.savefig(str(SBC_FIG), dpi=120, bbox_inches="tight")
    plt.close(fig)
    if SBC_FIG.exists():
        print(f"  fig_sbc.png saved ({SBC_FIG.stat().st_size // 1024} KB) ✓")

print("\n[sbc] Done.")
