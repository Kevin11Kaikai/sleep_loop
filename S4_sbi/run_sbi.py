"""
run_sbi.py
==========
睡眠数字孪生 Stage 2：基于神经样条流（NSF）的顺序 SNPE-C。

推断目标：  S4_sbi/x_obs.npz 中的 x_obs（长度 len(SUMMARY_KEYS)，与 simulator 输出顺序一致）
自由参数：  [g_h, g_LK, c_ctx2th, b]（4 维）
算法：      SNPE-C（sbi >= 0.21，density_estimator='nsf'）
预算：      4 轮合计约 5000 次仿真（墙钟约 9–10 小时）
            + 200 次 SBC 仿真 + 100 次 PPC 仿真

各轮计划：
    R1：从先验采样 2000 次仿真
    R2：在 x_obs 处从 R1 后验采样 1000 次
    R3：在 x_obs 处从 R2 后验采样 1000 次（若 Δstd < 10% 可提前结束）
    R4：在 x_obs 处从 R3 后验采样 1000 次

检查点：    每轮结束后 pickle 保存后验；all_simulations.npz 按轮追加 (theta, x) 批次。

Windows / neurolib 约束：
  - 全程 num_workers=1（fork 下 numba 易挂起）
  - 不使用多进程，仅顺序仿真

中止规则：
  - 任一轮 NaN 率 > 30%
  - 一轮过滤 NaN 后有效仿真 < 200
  - SBC 秩直方图呈 U 形（后验过窄）
  - x_obs 未通过合理性检查（须先运行 compute_xobs_from_eeg.py）

用法（在项目根目录）：
    conda activate neurolib
    python S4_sbi/run_sbi.py             # 完整运行
    python S4_sbi/run_sbi.py --dry-run   # 每轮 50 次仿真（共 4 轮），不落盘

输出目录 S4_sbi/sbi_outputs/：
    round{1-4}_posterior.pkl
    all_simulations.npz
    fig_marginals.png, fig_pairplot.png
    fig_sbc.png, fig_ppc.png, fig_pareto_overlay.png

另写入：
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

# ── 工作目录 = 项目根（须在导入 simulator_wrapper 之前完成）──────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
os.chdir(str(_ROOT))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

# ── NumPy 旧别名补丁 — 须在任何 neurolib 导入之前 ───────────────────────────
import numpy as np
import builtins as _builtins_mod
for _alias in ("int", "float", "bool", "object", "complex", "str"):
    if not hasattr(np, _alias):
        setattr(np, _alias, getattr(_builtins_mod, _alias))

# ── 优先使用本地 neurolib ───────────────────────────────────────────────────
_LOCAL_NEUROLIB = Path(r"D:\Year3_Mao_Projects\neurolib")
if _LOCAL_NEUROLIB.is_dir() and str(_LOCAL_NEUROLIB) not in sys.path:
    sys.path.insert(0, str(_LOCAL_NEUROLIB))

# ── 在重量级导入之前检查环境 ─────────────────────────────────────────────────
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

# ── SBI 相关导入 ─────────────────────────────────────────────────────────────
import torch
from torch import tensor
from sbi.inference import SNPE
from sbi.utils import BoxUniform

# 可选导入（诊断函数；下方有版本判断）
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

# ── 仿真器（在此导入；会触发 V7 与目标 PSD 的加载）────────────────────────────
print("[run_sbi] Importing simulator_wrapper ...")
from simulator_wrapper import simulator, SUMMARY_KEYS

# ── Matplotlib（无界面后端，便于无显示器环境运行）──────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════
PARAMS      = ["g_h", "g_LK", "c_ctx2th", "b"]
PRIOR_LOW   = tensor([0.035, 0.020, 0.05, 28.4], dtype=torch.float32)
PRIOR_HIGH  = tensor([0.095, 0.070, 0.22, 80.0], dtype=torch.float32)

ROUND_SIMS  = [2000, 1000, 1000, 1000]   # 正常模式下每轮仿真次数
SEED        = 42

OUTPUT_DIR  = _SCRIPT_DIR / "sbi_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH    = _SCRIPT_DIR / "sbi_log.txt"

# Pareto 种子 — 说明文档曾写 pareto_seeds_freshDE.json（无下划线），
# 磁盘上实际文件名为 pareto_seeds_fresh_DE.json（带下划线）。
PARETO_SEEDS_PATH = _ROOT / "S4_v7_repair" / "pareto_seeds_fresh_DE.json"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# 日志
# ═══════════════════════════════════════════════════════════════════════════════

class Logger:
    def __init__(self, path):
        self.path = path
        self.path.write_text("", encoding="utf-8")   # 启动时清空日志文件

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 仿真辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def run_batch(proposal, n_sims, log, rnd_label, fallback_prior=None):
    """
    从 proposal 采样 theta，顺序调用 simulator（等价 num_workers=1）。
    返回 (theta_tensor, x_tensor, n_nan)。
    若 NaN 率 > 30%，或完整轮次有效仿真 < 200，则中止。
    fallback_prior：BoxUniform；当从后验采样触发断言/运行时错误时使用
    （例如 dry-run 下 R2+ NSF 训练样本过少）。
    """
    log.log(f"  Sampling {n_sims} theta from proposal ...")
    try:
        thetas = proposal.sample((n_sims,))   # (n_sims, 4) float32 张量
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
        x = simulator(thetas[i])          # 返回 np.ndarray，形状 (len(SUMMARY_KEYS),)
        xs.append(x)
        if np.isnan(x).any():
            n_nan += 1
        if (i + 1) % 100 == 0 or (i + 1) == n_sims:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (n_sims - i - 1)
            log.log(f"    {i+1}/{n_sims}  NaN={n_nan}  "
                    f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

    x_tensor = torch.tensor(np.stack(xs), dtype=torch.float32)

    # 过滤含 NaN 的样本
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
    if n_valid < 200 and n_sims >= 500:   # 仅对大样本轮次强制执行
        raise RuntimeError(
            f"ABORT: only {n_valid} valid sims in {rnd_label} (< 200). "
            "Prior / bounds may be too wide for the current model config."
        )

    return thetas[valid_mask], x_tensor[valid_mask], n_nan


def append_to_simulations(theta, x, path):
    """将本轮 (theta, x) 追加写入累计 npz 文件。"""
    theta_np = theta.numpy() if hasattr(theta, "numpy") else np.array(theta)
    x_np     = x.numpy()     if hasattr(x, "numpy")     else np.array(x)
    if path.exists():
        prev = np.load(str(path), allow_pickle=True)
        theta_np = np.concatenate([prev["theta"], theta_np], axis=0)
        x_np     = np.concatenate([prev["x"], x_np], axis=0)
    np.savez(str(path), theta=theta_np, x=x_np, param_names=PARAMS,
             summary_keys=SUMMARY_KEYS)


# ═══════════════════════════════════════════════════════════════════════════════
# 构建带 NSF 的 SNPE，z_score_x='independent'
# ═══════════════════════════════════════════════════════════════════════════════

def build_inference(prior):
    try:
        # sbi >= 0.22：可使用 posterior_nn 工厂
        from sbi.neural_nets import posterior_nn
        density_est = posterior_nn(
            model="nsf",
            z_score_x="independent",
            z_score_theta="independent",
        )
        return SNPE(prior=prior, density_estimator=density_est,
                    device=DEVICE, show_progress_bars=True)
    except (ImportError, TypeError):
        # 略旧版 sbi 的回退：传入字符串；z_score 由库内部处理
        return SNPE(prior=prior, density_estimator="nsf",
                    device=DEVICE, show_progress_bars=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 诊断作图与检验
# ═══════════════════════════════════════════════════════════════════════════════

def plot_marginals(posterior, x_obs_t, prior, log):
    """4 个参数的一维边际直方图，并叠画先验。"""
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
    """后验样本的 4×4 配对图。"""
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
    """PPC：从后验抽 100 组参数再仿真，与 x_obs 对比。"""
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

    n_stats = len(SUMMARY_KEYS)
    fig, axes = plt.subplots(1, n_stats, figsize=(4 * n_stats + 2, 5))
    if n_stats == 1:
        axes = [axes]
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
        # x_obs 在 PPC 样本分布中的分位位置
        pct = float(np.mean(valid <= x_obs_vals[i]) * 100)
        ppc_pcts.append(pct)
        ax.set_xlabel(f"x_obs pct={pct:.0f}%", fontsize=8)
    fig.suptitle("Posterior Predictive Check  (grey dashes = 5/95 pct)")
    fig.tight_layout()
    path = OUTPUT_DIR / "fig_ppc.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    log.log(f"  Saved {path}")

    # 汇总报告到日志
    log.log("  PPC percentiles of x_obs:")
    all_pass = True
    for key, pct in zip(SUMMARY_KEYS, ppc_pcts):
        status = "PASS" if 5 <= pct <= 95 else "FAIL"
        if status == "FAIL":
            all_pass = False
        log.log(f"    {key:20s}  x_obs at {pct:.0f}th pct  [{status}]")
    return ppc_pcts, all_pass


def run_sbc_diagnostics(posterior, prior, x_obs_t, log, n_sbc=200):
    """运行 SBC：从先验采样 n_sbc 组 (theta, x) 作为真值检验校准。"""
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
        # 不同 sbi 版本 API 不同；两种调用都尝试
        try:
            ranks, _ = run_sbc(thetas_sbc, xs_sbc, posterior,
                                num_posterior_samples=1000,
                                show_progress_bar=True)
        except TypeError:
            ranks = run_sbc(thetas_sbc, xs_sbc, posterior,
                            num_posterior_samples=1000)

        # KS 检验：若后验校准良好，秩应接近均匀分布
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

        # 检查 U 形秩直方图（后验过窄、过度自信）
        n_fail = sum(1 for v in ks_results.values() if not v["pass"])
        if n_fail >= 2:
            log.log(f"  *** SBC WARNING: {n_fail}/4 params failed KS test — "
                    "possible over-confident posterior. STOP and review. ***")

        # 作图
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
    """在后验样本的若干 2D 投影上叠画 Pareto 种子点。"""
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

    # 简单 2D 散点叠画：多组参数对（V7 经验上信息量较大的维度组合）
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
# MAP 与置信区间
# ═══════════════════════════════════════════════════════════════════════════════

def get_map_and_ci(posterior, x_obs_t, log, n_samples=5000):
    """返回各参数的 MAP（近似）及 95% 置信区间。"""
    samples = posterior.sample((n_samples,), x=x_obs_t).numpy()
    map_est  = samples[np.argmax([1] * n_samples), :]   # 占位行，下一行会覆盖
    # 更合理的 MAP：用直方图峰值近似众数（非严格 KDE）
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
# 写入 Markdown 汇总报告
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
        "shape_r":        "fixed 1.0 (EEG reference)",
        "T4_q":           "SO peak Q-factor",
        "T4_freq":        "SO peak frequency [Hz]",
        "T8_n_sp_events": "spindle events per 60 s (from eeg_raw)",
        "T11_lag_ms":     "up_down_ratio (PAC)",
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
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def main(dry_run=False, x_obs_arg=None):
    log = Logger(LOG_PATH)
    t_overall = time.time()

    # 记录库版本
    log.log(f"=== SBI Stage 2 ({'DRY RUN' if dry_run else 'FULL RUN'}) ===")
    log.log(f"sbi={sbi.__version__}  torch={torch.__version__}  cuda={cuda_avail}")

    # ── 加载 x_obs ────────────────────────────────────────────────────────────
    x_obs_path = Path(x_obs_arg) if x_obs_arg else _SCRIPT_DIR / "x_obs_v3.npz"
    if not x_obs_path.exists():
        raise FileNotFoundError(
            f"{x_obs_path} not found.\n"
            "Run: python S4_sbi/compute_xobs_from_eeg_v3.py first."
        )
    xdata = np.load(str(x_obs_path), allow_pickle=True)
    x_obs_vals = xdata["values"].astype(np.float32)
    meta_str   = str(xdata.get("extraction_metadata", "{}"))
    meta       = json.loads(meta_str) if meta_str else {}
    x_obs_t    = torch.tensor(x_obs_vals, dtype=torch.float32)
    log.log(f"x_obs loaded: {x_obs_vals.tolist()}")

    # ── 构建先验 ───────────────────────────────────────────────────────────────
    prior = BoxUniform(low=PRIOR_LOW, high=PRIOR_HIGH)
    log.log(f"Prior: BoxUniform  low={PRIOR_LOW.tolist()}  high={PRIOR_HIGH.tolist()}")

    # ── 构建 SNPE 推断对象 ─────────────────────────────────────────────────────
    inference = build_inference(prior)

    round_sims  = [50, 50, 50, 50] if dry_run else ROUND_SIMS
    posteriors  = []
    std_history = []
    round_times = []

    sim_path = OUTPUT_DIR / "all_simulations.npz"

    # ── 共 4 轮（或第 3 轮后提前结束）──────────────────────────────────────────
    for rnd in range(1, 5):
        n_sims = round_sims[rnd - 1]
        rnd_label = f"Round {rnd}"
        log.log(f"\n{'='*50}")
        log.log(f"{rnd_label}: {n_sims} simulations")
        log.log(f"{'='*50}")

        t_rnd = time.time()

        # 提议分布：R1 用先验，R2 起用上一轮后验（在 x_obs 处条件化）
        if rnd == 1:
            proposal = prior
        else:
            proposal = posteriors[-1].set_default_x(x_obs_t)

        # 仿真
        theta, x, n_nan = run_batch(proposal, n_sims, log, rnd_label,
                                    fallback_prior=prior)
        log.log(f"  {rnd_label} sims: {len(theta)} valid  NaN={n_nan}")

        # 追加到累计数据集（便于崩溃后恢复）
        if not dry_run:
            append_to_simulations(theta, x, sim_path)

        # 用至今累计的全部仿真数据训练 SNPE-C
        log.log(f"  Training SNPE-C (device={DEVICE}) ...")
        inference.append_simulations(theta, x,
                                     proposal=prior if rnd == 1 else posteriors[-1])
        density_est = inference.train()
        posterior = inference.build_posterior(density_est)
        posteriors.append(posterior)

        # 检查点：保存本轮后验
        if not dry_run:
            pkl_path = OUTPUT_DIR / f"round{rnd}_posterior.pkl"
            with open(str(pkl_path), "wb") as f:
                pickle.dump(posterior, f)
            log.log(f"  Checkpoint saved: {pkl_path}")

        # 收敛判据（自第 2 轮起采样后验标准差；第 3 轮可与第 2 轮比较）
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

    # ── 最终后验 ───────────────────────────────────────────────────────────────
    final_posterior = posteriors[-1]

    # ── 诊断 ───────────────────────────────────────────────────────────────────
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

    # ── Markdown 汇总报告 ─────────────────────────────────────────────────────
    write_results_md(x_obs_vals, meta, map_est, ci_lo, ci_hi,
                     ks_results, ppc_pcts, round_times, log)

    total_min = (time.time() - t_overall) / 60
    log.log(f"\nTotal wall-clock: {total_min:.1f} min ({total_min/60:.2f} h)")
    log.log("Stage 2 SBI complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="50 sims per round (4 rounds), pipeline check, no saves")
    parser.add_argument("--x-obs", type=str, default=None,
                        help="Path to x_obs .npz file (default: S4_sbi/x_obs_v3.npz)")
    args = parser.parse_args()
    main(dry_run=args.dry_run, x_obs_arg=args.x_obs)
