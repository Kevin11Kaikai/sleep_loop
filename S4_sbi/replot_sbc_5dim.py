"""
replot_sbc_5dim.py
==================
Standalone fix for fig_sbc.png — re-runs SBC ranks and the rank plot
with the correct sbc_rank_plot(ranks, num_posterior_samples) signature.

Does NOT retrain. Reuses round4_posterior.pkl + re-simulates 200 SBC pairs.
"""
import sys, os, warnings, pickle
from pathlib import Path
warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(_ROOT))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "S4_sbi"))

import numpy as np
import builtins as _b
for _alias in ("int", "float", "bool", "object", "complex", "str"):
    if not hasattr(np, _alias):
        setattr(np, _alias, getattr(_b, _alias))

_LOCAL_NEUROLIB = Path(r"D:\Year3_Mao_Projects\neurolib")
if _LOCAL_NEUROLIB.is_dir() and str(_LOCAL_NEUROLIB) not in sys.path:
    sys.path.insert(0, str(_LOCAL_NEUROLIB))

import time
import torch
from torch import tensor
from sbi.utils import BoxUniform
from sbi.diagnostics.sbc import run_sbc
from sbi.analysis.plot import sbc_rank_plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulator_wrapper import simulator

PARAMS     = ["g_h", "g_LK", "c_ctx2th", "b"]
PRIOR_LOW  = tensor([0.035, 0.020, 0.05, 28.4], dtype=torch.float32)
PRIOR_HIGH = tensor([0.095, 0.070, 0.22, 80.0], dtype=torch.float32)
N_SBC      = 200
N_POST     = 1000

OUT_DIR = _ROOT / "S4_sbi" / "sbi_outputs"

prior = BoxUniform(low=PRIOR_LOW, high=PRIOR_HIGH)

with open(OUT_DIR / "round4_posterior.pkl", "rb") as f:
    posterior = pickle.load(f)

print(f"[replot_sbc] Simulating {N_SBC} SBC pairs from prior ...")
thetas_sbc = prior.sample((N_SBC,))
xs_list, n_nan = [], 0
t0 = time.time()
for i, th in enumerate(thetas_sbc):
    x = simulator(th.numpy())
    xs_list.append(x)
    if np.isnan(x).any():
        n_nan += 1
    if (i + 1) % 25 == 0:
        print(f"  SBC {i+1}/{N_SBC}  NaN={n_nan}  elapsed={(time.time()-t0)/60:.1f} min")

xs_sbc = torch.tensor(np.stack(xs_list), dtype=torch.float32)
valid = ~torch.isnan(xs_sbc).any(dim=1)
print(f"[replot_sbc] SBC valid: {valid.sum().item()}/{N_SBC}")
thetas_sbc = thetas_sbc[valid]
xs_sbc     = xs_sbc[valid]

print(f"[replot_sbc] Computing ranks via run_sbc (num_posterior_samples={N_POST}) ...")
try:
    ranks, _ = run_sbc(thetas_sbc, xs_sbc, posterior,
                        num_posterior_samples=N_POST,
                        show_progress_bar=True)
except TypeError:
    ranks = run_sbc(thetas_sbc, xs_sbc, posterior, num_posterior_samples=N_POST)

print(f"[replot_sbc] ranks shape: {tuple(ranks.shape)}")

# KS check (echo numbers for sanity)
from scipy.stats import kstest
print("[replot_sbc] SBC KS p-values (>0.05 = PASS):")
for i, name in enumerate(PARAMS):
    ranks_i = ranks[:, i].float().numpy() / N_POST
    ks_stat, p_val = kstest(ranks_i, "uniform")
    print(f"  {name:12s}  KS={ks_stat:.4f}  p={p_val:.4f}")

# Correct call
fig, _ = sbc_rank_plot(ranks, num_posterior_samples=N_POST,
                       parameter_labels=PARAMS, plot_type="cdf")
out_path = OUT_DIR / "fig_sbc.png"
fig.savefig(str(out_path), dpi=150)
plt.close(fig)
print(f"[replot_sbc] Saved {out_path}")
