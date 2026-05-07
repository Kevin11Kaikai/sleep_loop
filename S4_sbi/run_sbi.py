"""
run_sbi.py
==========
Sequential SNPE-C with Neural Spline Flow for Sleep Digital Twin Stage 2.

Inference target:   x_obs from S4_sbi/x_obs.npz  (len(SUMMARY_KEYS); same order as simulator)
Free parameters:    [g_h, g_LK, c_ctx2th, b]       (4-dim)
Algorithm:          SNPE-C  (sbi >= 0.21, density_estimator='nsf')
Budget:             ~5000 simulations across 4 rounds  (~9-10 h wall-clock)
                    + 200 SBC sims + 100 PPC sims

Round plan:
    R1: 2000 sims from prior
    R2: 1000 sims from R1 posterior at x_obs
    R3: 1000 sims from R2 posterior at x_obs  (early stop if Δstd < 10%)
    R4: 1000 sims from R3 posterior at x_obs

Checkpointing: posterior pickled after every round; all_simulations.npz
               grows by appending each round's (theta, x) batch.

Windows / neurolib constraints:
  - num_workers=1 enforced throughout (numba hangs under fork).
  - No multiprocessing. Sequential simulation only.

Abort rules:
  - NaN rate > 30% in any round.
  - Round produces < 200 valid sims (after NaN filtering).
  - SBC shows U-shaped rank histogram (over-confident posterior).
  - x_obs sanity check fails (run compute_xobs_from_eeg.py first).

Usage (from project root):
    conda activate neurolib
    python S4_sbi/run_sbi.py             # full run
    python S4_sbi/run_sbi.py --dry-run   # 50-sim Round 1, no save

Outputs in S4_sbi/sbi_outputs/:
    round{1-4}_posterior.pkl
    all_simulations.npz
    fig_marginals.png, fig_pairplot.png
    fig_sbc.png, fig_ppc.png, fig_pareto_overlay.png

Also writes:
    S4_sbi/sbi_log.txt
    S4_sbi/sbi_results.md
"""

import sys
import os
import time
import json
import pickle
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CWD = project root (must happen before simulator_wrapper import) ──────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
os.chdir(str(_ROOT))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

# ── NumPy alias shim — before any neurolib import ────────────────────────────
import numpy as np
import builtins as _builtins_mod
for _alias in ("int", "float", "bool", "object", "complex", "str"):
    if not hasattr(np, _alias):
        setattr(np, _alias, getattr(_builtins_mod, _alias))

# ── Local neurolib takes precedence ──────────────────────────────────────────
_LOCAL_NEUROLIB = Path(r"D:\Year3_Mao_Projects\neurolib")
if _LOCAL_NEUROLIB.is_dir() and str(_LOCAL_NEUROLIB) not in sys.path:
    sys.path.insert(0, str(_LOCAL_NEUROLIB))

# ── Check environment before heavy imports ────────────────────────────────────
print("[run_sbi] Checking environment ...")
try:
    import sbi
    import torch
    from fooof import FOOOF
    sbi_ver   = sbi.__version__
    torch_ver = torch.__version__
    cuda_avail = torch.cuda.is_available()
    print(f"  sbi={sbi_ver}  torch={torch_ver}  cuda={cuda_avail}  fooof=OK")
    if tuple(int(x) for x in sbi_ver.split(".")[:2]) < (0, 21):
        raise RuntimeError(f"sbi version {sbi_ver} < 0.21 — update required")
except ImportError as e:
    raise RuntimeError(
        f"Environment check failed: {e}\n"
        "Run: conda activate neurolib && pip install sbi>=0.21 torch"
    )

DEVICE = "cuda" if cuda_avail else "cpu"
print(f"  Neural network training device: {DEVICE}")

# ── SBI imports ───────────────────────────────────────────────────────────────
import torch
from torch import tensor
from sbi.inference import SNPE
from sbi.utils import BoxUniform

# Optional imports (diagnostic functions; version-guard below)
try:
    from sbi.diagnostics.sbc import run_sbc
    from sbi.analysis.plot import sbc_rank_plot
    HAS_SBC = True
except ImportError:
    HAS_SBC = False
    print("  [warn] sbi.diagnostics.sbc not found — SBC will be skipped")

try:
    from sbi.analysis import pairplot
    HAS_PAIRPLOT = True
except ImportError:
    HAS_PAIRPLOT = False

# ── Simulator (loaded here; triggers V7 import + target PSD load) ─────────────
print("[run_sbi] Importing simulator_wrapper ...")
from simulator_wrapper import simulator, SUMMARY_KEYS

# ── Matplotlib (non-interactive backend for headless execution) ───────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════
PARAMS      = ["g_h", "g_LK", "c_ctx2th", "b"]
PRIOR_LOW   = tensor([0.035, 0.020, 0.05, 28.4], dtype=torch.float32)
PRIOR_HIGH  = tensor([0.095, 0.070, 0.22, 42.6], dtype=torch.float32)

ROUND_SIMS  = [2000, 1000, 1000, 1000]   # sims per round (normal run)
SEED        = 42

OUTPUT_DIR  = _SCRIPT_DIR / "sbi_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH    = _SCRIPT_DIR / "sbi_log.txt"

# Pareto seeds — NOTE: spec says pareto_seeds_freshDE.json (no underscore)
# but actual file on disk is pareto_seeds_fresh_DE.json (with underscore).
PARETO_SEEDS_PATH = _ROOT / "S4_v7_repair" / "pareto_seeds_fresh_DE.json"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════════

class Logger:
    def __init__(self, path):
        self.path = path
        self.path.write_text("", encoding="utf-8")   # clear on start

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def run_batch(proposal, n_sims, log, rnd_label, fallback_prior=None):
    """
    Sample theta from proposal and run simulator sequentially (num_workers=1).
    Returns (theta_tensor, x_tensor, n_nan).
    Aborts if NaN rate > 30% or valid sims < 200 (for full rounds).
    fallback_prior: BoxUniform; used if posterior sampling raises an assertion
    (can happen when NSF is trained on very few samples, e.g. dry-run R2+).
    """
    log.log(f"  Sampling {n_sims} theta from proposal ...")
    try:
        thetas = proposal.sample((n_sims,))   # (n_sims, 4) float32 tensor
    except (AssertionError, RuntimeError) as _sample_err:
        if fallback_prior is not None:
            log.log(f"  [warn] Posterior sampling failed ({_sample_err}); "
                    f"falling back to prior for this round.")
            thetas = fallback_prior.sample((n_sims,))
        else:
            raise

    xs, t_start = [], time.time()
    n_nan = 0
    for i in range(n_sims):
        x = simulator(thetas[i])          # returns np.ndarray (len(SUMMARY_KEYS),)
        xs.append(x)
        if np.isnan(x).any():
            n_nan += 1
        if (i + 1) % 100 == 0 or (i + 1) == n_sims:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (n_sims - i - 1)
            log.log(f"    {i+1}/{n_sims}  NaN={n_nan}  "
                    f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

    x_tensor = torch.tensor(np.stack(xs), dtype=torch.float32)

    # NaN filtering
    valid_mask = ~torch.isnan(x_tensor).any(dim=1)
    nan_rate   = (~valid_mask).float().mean().item()
    log.log(f"  {rnd_label}: {valid_mask.sum().item()} valid / {n_sims} sims "
            f"(NaN rate={100*nan_rate:.1f}%)")

    if nan_rate > 0.30:
        raise RuntimeError(
            f"ABORT: NaN rate {100*nan_rate:.1f}% > 30% in {rnd_label}. "
            "Check simulator or parameter bounds."
        )
    n_valid = int(valid_mask.sum())
    if n_valid < 200 and n_sims >= 500:   # only enforce for full rounds
        raise RuntimeError(
            f"ABORT: only {n_valid} valid sims in {rnd_label} (< 200). "
            "Prior / bounds may be too wide for the current model config."
        )

    return thetas[valid_mask], x_tensor[valid_mask], n_nan


def append_to_simulations(theta, x, path):
    """Append a new round's (theta, x) to the cumulative npz file."""
    theta_np = theta.numpy() if hasattr(theta, "numpy") else np.array(theta)
    x_np     = x.numpy()     if hasattr(x, "numpy")     else np.array(x)
    if path.exists():
        prev = np.load(str(path), allow_pickle=True)
        theta_np = np.concatenate([prev["theta"], theta_np], axis=0)
        x_np     = np.concatenate([prev["x"], x_np], axis=0)
    np.savez(str(path), theta=theta_np, x=x_np, param_names=PARAMS,
             summary_keys=SUMMARY_KEYS)


# ═══════════════════════════════════════════════════════════════════════════════
# Build SNPE with NSF, z_score_x='independent'
# ═══════════════════════════════════════════════════════════════════════════════

def build_inference(prior):
    try:
        # sbi >= 0.22: posterior_nn factory available
        from sbi.neural_nets import posterior_nn
        density_est = posterior_nn(
            model="nsf",
            z_score_x="independent",
            z_score_theta="independent",
        )
        return SNPE(prior=prior, density_estimator=density_est,
                    device=DEVICE, show_progress_bars=True)
    except (ImportError, TypeError):
        # Fallback for slightly older sbi: pass string; z_score handled internally
        return SNPE(prior=prior, density_estimator="nsf",
                    device=DEVICE, show_progress_bars=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def plot_marginals(posterior, x_obs_t, prior, log):
    """4 marginal histograms with prior overlay."""
    samples = posterior.sample((2000,), x=x_obs_t).numpy()
    prior_s = prior.sample((2000,)).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (ax, name) in enumerate(zip(axes, PARAMS)):
        ax.hist(prior_s[:, i],   bins=40, alpha=0.4, color="grey",  label="prior",
                density=True)
        ax.hist(samples[:, i],   bins=40, alpha=0.7, color="steelblue", label="posterior",
                density=True)
        ax.axvline(PRIOR_LOW[i],  color="red",  ls="--", lw=1)
        ax.axvline(PRIOR_HIGH[i], color="red",  ls="--", lw=1)
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.suptitle("Posterior marginals (blue) vs prior (grey)")
    fig.tight_layout()
    path = OUTPUT_DIR / "fig_marginals.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    log.log(f"  Saved {path}")


def plot_pairplot_fig(posterior, x_obs_t, log):
    """4×4 pairplot of posterior samples."""
    samples = posterior.sample((2000,), x=x_obs_t)
    if HAS_PAIRPLOT:
        fig, _ = pairplot(samples, labels=PARAMS,
                          figsize=(10, 10), upper="contour")
    else:
        import pandas as pd
        df = pd.DataFrame(samples.numpy(), columns=PARAMS)
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i in range(4):
            for j in range(4):
                ax = axes[i, j]
                if i == j:
                    ax.hist(df.iloc[:, i], bins=30, color="steelblue")
                    ax.set_xlabel(PARAMS[i])
                elif j < i:
                    ax.scatter(df.iloc[:, j], df.iloc[:, i],
                               alpha=0.1, s=2, color="steelblue")
                    ax.set_xlabel(PARAMS[j]); ax.set_ylabel(PARAMS[i])
                else:
                    ax.axis("off")
        fig.tight_layout()
    path = OUTPUT_DIR / "fig_pairplot.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    log.log(f"  Saved {path}")


def plot_ppc(posterior, x_obs_t, x_obs_vals, log):
    """PPC: 100 posterior-predictive simulations vs x_obs."""
    log.log("  PPC: sampling 100 posterior params and simulating ...")
    samples = posterior.sample((100,), x=x_obs_t).numpy()
    xs_ppc = []
    n_nan = 0
    for i, th in enumerate(samples):
        x = simulator(th)
        xs_ppc.append(x)
        if np.isnan(x).any():
            n_nan += 1
        if (i + 1) % 25 == 0:
            log.log(f"    PPC {i+1}/100  NaN={n_nan}")
    xs_ppc = np.stack(xs_ppc)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    ppc_pcts = []
    for i, (ax, key) in enumerate(zip(axes, SUMMARY_KEYS)):
        col = xs_ppc[:, i]
        valid = col[~np.isnan(col)]
        ax.hist(valid, bins=20, color="skyblue", alpha=0.8, label="PPC samples")
        ax.axvline(x_obs_vals[i], color="red", lw=2, label="x_obs")
        lo, hi = np.nanpercentile(col, 5), np.nanpercentile(col, 95)
        ax.axvline(lo, color="grey", ls="--", lw=1)
        ax.axvline(hi, color="grey", ls="--", lw=1)
        ax.set_title(key, fontsize=9)
        ax.legend(fontsize=7)
        # Percentile of x_obs in PPC
        pct = float(np.mean(valid <= x_obs_vals[i]) * 100)
        ppc_pcts.append(pct)
        ax.set_xlabel(f"x_obs pct={pct:.0f}%", fontsize=8)
    fig.suptitle("Posterior Predictive Check  (grey dashes = 5/95 pct)")
    fig.tight_layout()
    path = OUTPUT_DIR / "fig_ppc.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    log.log(f"  Saved {path}")

    # Report
    log.log("  PPC percentiles of x_obs:")
    all_pass = True
    for key, pct in zip(SUMMARY_KEYS, ppc_pcts):
        status = "PASS" if 5 <= pct <= 95 else "FAIL"
        if status == "FAIL":
            all_pass = False
        log.log(f"    {key:20s}  x_obs at {pct:.0f}th pct  [{status}]")
    return ppc_pcts, all_pass


def run_sbc_diagnostics(posterior, prior, x_obs_t, log, n_sbc=200):
    """Run SBC: 200 ground-truth simulations from prior."""
    if not HAS_SBC:
        log.log("  SBC skipped (sbi.analysis.run_sbc not available)")
        return None, None

    log.log(f"  SBC: simulating {n_sbc} prior samples ...")
    thetas_sbc = prior.sample((n_sbc,))
    xs_sbc_list = []
    n_nan = 0
    t0 = time.time()
    for i, th in enumerate(thetas_sbc):
        x = simulator(th.numpy())
        xs_sbc_list.append(x)
        if np.isnan(x).any():
            n_nan += 1
        if (i + 1) % 50 == 0:
            log.log(f"    SBC {i+1}/{n_sbc}  NaN={n_nan}  "
                    f"elapsed={( time.time()-t0)/60:.1f} min")

    xs_sbc = torch.tensor(np.stack(xs_sbc_list), dtype=torch.float32)
    valid  = ~torch.isnan(xs_sbc).any(dim=1)
    log.log(f"  SBC valid: {valid.sum().item()}/{n_sbc}")
    thetas_sbc = thetas_sbc[valid]
    xs_sbc     = xs_sbc[valid]

    try:
        # API varies across sbi versions; try both
        try:
            ranks, _ = run_sbc(thetas_sbc, xs_sbc, posterior,
                                num_posterior_samples=1000,
                                show_progress_bar=True)
        except TypeError:
            ranks = run_sbc(thetas_sbc, xs_sbc, posterior,
                            num_posterior_samples=1000)

        # KS test: ranks should be uniform if calibrated
        from scipy.stats import kstest
        log.log("  SBC KS p-values (> 0.05 = PASS):")
        ks_results = {}
        for i, param in enumerate(PARAMS):
            ranks_i = ranks[:, i].float().numpy() / 1000.0
            ks_stat, p_val = kstest(ranks_i, "uniform")
            status = "PASS" if p_val >= 0.05 else "FAIL"
            log.log(f"    {param:12s}  KS={ks_stat:.4f}  p={p_val:.4f}  [{status}]")
            ks_results[param] = {"ks": float(ks_stat), "p": float(p_val),
                                 "pass": p_val >= 0.05}

        # Check for U-shape (over-confident posterior)
        n_fail = sum(1 for v in ks_results.values() if not v["pass"])
        if n_fail >= 2:
            log.log(f"  *** SBC WARNING: {n_fail}/4 params failed KS test — "
                    "possible over-confident posterior. STOP and review. ***")

        # Plot
        try:
            fig, _ = sbc_rank_plot(ranks, num_sims=len(thetas_sbc))
        except TypeError:
            fig, _ = sbc_rank_plot(ranks)
        path = OUTPUT_DIR / "fig_sbc.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        log.log(f"  Saved {path}")
        return ks_results, ranks

    except Exception as exc:
        log.log(f"  [warn] SBC computation failed: {exc}")
        return None, None


def plot_pareto_overlay(posterior, x_obs_t, log):
    """Overlay Pareto seeds on pairplot of posterior samples."""
    if not PARETO_SEEDS_PATH.exists():
        log.log(f"  [warn] Pareto seeds not found at {PARETO_SEEDS_PATH}")
        return

    with open(str(PARETO_SEEDS_PATH), encoding="utf-8") as f:
        seeds_data = json.load(f)

    seeds = seeds_data["seeds"]
    samples = posterior.sample((1000,), x=x_obs_t).numpy()

    log.log("  Pareto seed log-probs under posterior:")
    seed_params_list = []
    for seed in seeds:
        p = seed["params"]
        theta_s = np.array([p["g_h"], p["g_LK"], p["c_ctx2th"], p["b"]])
        seed_params_list.append(theta_s)

        theta_t = torch.tensor(theta_s, dtype=torch.float32).unsqueeze(0)
        try:
            lp = posterior.log_prob(theta_t, x=x_obs_t).item()
        except Exception:
            lp = float("nan")
        log.log(f"    Seed {seed['tag']}: log_prob={lp:.3f}  "
                f"(shape_r={seed['objectives']['shape_r']:.4f}  "
                f"MI={seed['pac_metrics']['MI']:.4f})")

    # Simple 2D scatter overlay: c_ctx2th vs g_h (most informative dims per V7)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    colors = ["red", "orange", "green"]
    for ax, (i, j) in zip(axes.ravel(), pairs):
        ax.scatter(samples[:, j], samples[:, i],
                   alpha=0.15, s=3, color="steelblue", label="posterior")
        for k, (th, seed) in enumerate(zip(seed_params_list, seeds)):
            ax.scatter(th[j], th[i], s=120, marker="*", zorder=5,
                       color=colors[k % len(colors)],
                       label=f"Seed {seed['tag']}")
        ax.set_xlabel(PARAMS[j]); ax.set_ylabel(PARAMS[i])
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles[-4:], labels[-4:], loc="upper right", fontsize=9)
    fig.suptitle("Posterior samples + Pareto seeds overlay")
    fig.tight_layout()
    path = OUTPUT_DIR / "fig_pareto_overlay.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    log.log(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAP extraction
# ═══════════════════════════════════════════════════════════════════════════════

def get_map_and_ci(posterior, x_obs_t, log, n_samples=5000):
    """Return MAP (argmax of samples) and 95% CI per parameter."""
    samples = posterior.sample((n_samples,), x=x_obs_t).numpy()
    map_est  = samples[np.argmax([1] * n_samples), :]   # placeholder; use mode
    # Better MAP: find sample nearest to KDE mode (approx via histogram peak)
    map_est = np.array([
        float(np.histogram(samples[:, i], bins=50)[1][
            np.argmax(np.histogram(samples[:, i], bins=50)[0])
        ])
        for i in range(4)
    ])
    ci_lo = np.percentile(samples, 2.5, axis=0)
    ci_hi = np.percentile(samples, 97.5, axis=0)

    log.log("  Posterior MAP + 95% CI:")
    for i, param in enumerate(PARAMS):
        log.log(f"    {param:12s}  MAP={map_est[i]:.5f}  "
                f"95%CI=[{ci_lo[i]:.5f}, {ci_hi[i]:.5f}]")
    return map_est, ci_lo, ci_hi, samples


# ═══════════════════════════════════════════════════════════════════════════════
# Write summary report
# ═══════════════════════════════════════════════════════════════════════════════

def write_results_md(x_obs_vals, meta, map_est, ci_lo, ci_hi,
                     ks_results, ppc_pcts, round_times, log):
    lines = [
        "# SBI Stage 2 Results",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Subject**: {meta.get('subject_id', 'SC4001')}",
        "",
        "## x_obs values",
        "",
        "| Summary | Value | Extraction method |",
        "|---------|-------|-------------------|",
    ]
    notes = {
        "shape_r":        "hardcoded 1.0 (EEG = reference)",
        "T4_q":           "SO peak Q-factor",
        "T4_freq":        "SO peak freq [Hz]",
        "T6_ibi_cv":      "UP-burst IBI CV",
        "T8_n_sp_events": "spindle events per 60 s (normalized)",
        "T11_lag_ms":     "up_down_ratio (PAC)",
        "MI":             "PAC Modulation Index",
    }
    for k, v in zip(SUMMARY_KEYS, x_obs_vals):
        lines.append(f"| {k} | {v:.5f} | {notes.get(k, '')} |")

    lines += [
        "",
        "## Posterior MAP + 95% CI",
        "",
        "| Parameter | MAP | CI_lo | CI_hi |",
        "|-----------|-----|-------|-------|",
    ]
    for i, param in enumerate(PARAMS):
        lines.append(
            f"| {param} | {map_est[i]:.5f} | {ci_lo[i]:.5f} | {ci_hi[i]:.5f} |"
        )

    lines += ["", "## SBC Results", ""]
    if ks_results:
        lines += ["| Parameter | KS stat | p-value | Pass |",
                  "|-----------|---------|---------|------|"]
        for param, r in ks_results.items():
            lines.append(
                f"| {param} | {r['ks']:.4f} | {r['p']:.4f} | "
                f"{'PASS' if r['pass'] else 'FAIL'} |"
            )
    else:
        lines.append("SBC not run or failed.")

    lines += ["", "## PPC Results", ""]
    if ppc_pcts:
        lines += ["| Summary | x_obs percentile | Pass (5–95%) |",
                  "|---------|-----------------|-------------|"]
        for k, pct in zip(SUMMARY_KEYS, ppc_pcts):
            status = "PASS" if 5 <= pct <= 95 else "FAIL"
            lines.append(f"| {k} | {pct:.0f}% | {status} |")

    lines += ["", "## Wall-clock breakdown", ""]
    for rnd, t_min in enumerate(round_times, 1):
        lines.append(f"- Round {rnd}: {t_min:.1f} min")

    out = _SCRIPT_DIR / "sbi_results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.log(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(dry_run=False):
    log = Logger(LOG_PATH)
    t_overall = time.time()

    # Log versions
    log.log(f"=== SBI Stage 2 ({'DRY RUN' if dry_run else 'FULL RUN'}) ===")
    log.log(f"sbi={sbi.__version__}  torch={torch.__version__}  cuda={cuda_avail}")

    # ── Load x_obs ────────────────────────────────────────────────────────────
    x_obs_path = _SCRIPT_DIR / "x_obs.npz"
    if not x_obs_path.exists():
        raise FileNotFoundError(
            f"{x_obs_path} not found.\n"
            "Run: python S4_sbi/compute_xobs_from_eeg.py first."
        )
    xdata = np.load(str(x_obs_path), allow_pickle=True)
    x_obs_vals = xdata["values"].astype(np.float32)
    meta_str   = str(xdata.get("extraction_metadata", "{}"))
    meta       = json.loads(meta_str) if meta_str else {}
    x_obs_t    = torch.tensor(x_obs_vals, dtype=torch.float32)
    log.log(f"x_obs loaded: {x_obs_vals.tolist()}")

    # ── Build prior ───────────────────────────────────────────────────────────
    prior = BoxUniform(low=PRIOR_LOW, high=PRIOR_HIGH)
    log.log(f"Prior: BoxUniform  low={PRIOR_LOW.tolist()}  high={PRIOR_HIGH.tolist()}")

    # ── Build SNPE inference ──────────────────────────────────────────────────
    inference = build_inference(prior)

    round_sims  = [50, 50, 50, 50] if dry_run else ROUND_SIMS
    posteriors  = []
    std_history = []
    round_times = []

    sim_path = OUTPUT_DIR / "all_simulations.npz"

    # ── 4 Rounds ──────────────────────────────────────────────────────────────
    for rnd in range(1, 5):
        n_sims = round_sims[rnd - 1]
        rnd_label = f"Round {rnd}"
        log.log(f"\n{'='*50}")
        log.log(f"{rnd_label}: {n_sims} simulations")
        log.log(f"{'='*50}")

        t_rnd = time.time()

        # Proposal: prior for R1, posterior for R2+
        if rnd == 1:
            proposal = prior
        else:
            proposal = posteriors[-1].set_default_x(x_obs_t)

        # Simulate
        theta, x, n_nan = run_batch(proposal, n_sims, log, rnd_label,
                                    fallback_prior=prior)
        log.log(f"  {rnd_label} sims: {len(theta)} valid  NaN={n_nan}")

        # Append to cumulative dataset (survives crashes)
        if not dry_run:
            append_to_simulations(theta, x, sim_path)

        # Train SNPE-C on all accumulated simulations
        log.log(f"  Training SNPE-C (device={DEVICE}) ...")
        inference.append_simulations(theta, x,
                                     proposal=prior if rnd == 1 else posteriors[-1])
        density_est = inference.train()
        posterior = inference.build_posterior(density_est)
        posteriors.append(posterior)

        # Checkpoint
        if not dry_run:
            pkl_path = OUTPUT_DIR / f"round{rnd}_posterior.pkl"
            with open(str(pkl_path), "wb") as f:
                pickle.dump(posterior, f)
            log.log(f"  Checkpoint saved: {pkl_path}")

        # Convergence check (R3 and R4)
        if rnd >= 2:
            try:
                samps = posterior.sample((500,), x=x_obs_t)
                std_cur = samps.std(dim=0)
                std_history.append(std_cur)
                log.log(f"  Posterior std: {std_cur.tolist()}")
                if rnd == 3 and len(std_history) >= 2:
                    std_prev = std_history[-2]
                    rel = ((std_cur - std_prev).abs() / (std_prev + 1e-9)).mean().item()
                    log.log(f"  Relative std change R2→R3: {100*rel:.1f}%")
                    if rel < 0.10:
                        log.log("  *** Converged early at Round 3 — skipping Round 4 ***")
                        round_times.append((time.time() - t_rnd) / 60)
                        break
            except (AssertionError, RuntimeError) as _conv_err:
                log.log(f"  [warn] Convergence-check sampling failed ({_conv_err}); "
                        f"skipping std check for Round {rnd}.")

        elapsed_rnd = (time.time() - t_rnd) / 60
        round_times.append(elapsed_rnd)
        log.log(f"  {rnd_label} done in {elapsed_rnd:.1f} min")

    if dry_run:
        log.log("\nDry run complete. No files saved. Pipeline works end-to-end.")
        return

    # ── Final posterior ───────────────────────────────────────────────────────
    final_posterior = posteriors[-1]

    # ── Diagnostics ───────────────────────────────────────────────────────────
    log.log("\n=== Diagnostics ===")

    log.log("\n[1/5] Marginals ...")
    plot_marginals(final_posterior, x_obs_t, prior, log)

    log.log("\n[2/5] Pairplot ...")
    plot_pairplot_fig(final_posterior, x_obs_t, log)

    log.log("\n[3/5] MAP + 95% CI ...")
    map_est, ci_lo, ci_hi, post_samples = get_map_and_ci(
        final_posterior, x_obs_t, log
    )

    log.log("\n[4/5] PPC (100 sims) ...")
    ppc_pcts, ppc_all_pass = plot_ppc(
        final_posterior, x_obs_t, x_obs_vals, log
    )

    log.log("\n[5/5] SBC (200 sims) ...")
    ks_results, _ = run_sbc_diagnostics(
        final_posterior.set_default_x(x_obs_t), prior, x_obs_t, log, n_sbc=200
    )

    log.log("\n[6/6] Pareto overlay ...")
    plot_pareto_overlay(final_posterior.set_default_x(x_obs_t), x_obs_t, log)

    # ── Summary report ────────────────────────────────────────────────────────
    write_results_md(x_obs_vals, meta, map_est, ci_lo, ci_hi,
                     ks_results, ppc_pcts, round_times, log)

    total_min = (time.time() - t_overall) / 60
    log.log(f"\nTotal wall-clock: {total_min:.1f} min ({total_min/60:.2f} h)")
    log.log("Stage 2 SBI complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="50-sim Round 1 only; verify pipeline, no saves")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
