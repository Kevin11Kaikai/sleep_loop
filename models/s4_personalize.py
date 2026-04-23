"""
s4_personalize.py — Session 2-B：丘脑–皮层参数个体化（差分进化）

用仿真皮层 PSD 逼近指定受试者（默认 SC4001）的 N3 EEG 目标谱；最优参数写入
data/patient_params.json，供 s5 等下游使用。

正文阅读顺序（搜索「# ── 步骤」跳转）：
  步骤 1 — ThalamoCorticalNetwork + set_params_glob（与 s3 同构）
  步骤 2 — 从 manifest 读 PSG/Hyp，筛 N3 epoch，QC + Welch → target_psd / target_freqs，
           保存 data/target_psd_<SUBJECT>.npy（非群体平均 PSD）
  步骤 3 — 目标谱频段占比 → TARGET_DELTA_RATIO、TARGET_SIGMA_RATIO（供适应度 3a/3b）
  步骤 4 — 单例 MultiModel（30 s/eval）+ _psd_corr：归一化 → log 域 r_shape →
           delta_bonus → sigma_term；score = 0.6×① + 0.3×② + 0.1×③
  步骤 5 — _objective（最小化 -score）、all_records、_callback 代际打印
  步骤 6 — scipy.optimize.differential_evolution（workers=1，numba 不宜多进程）
  步骤 7 — 取最优行 → patient_params.json / patient_params_<SUBJECT>.json + evolution_records.csv
  步骤 8 — evolution_result.png（曲线/分布/归一化 Sim vs Target）+ 终端 Session 2-B 校验

Backend
-------
neurolib.optimize.evolution.Evolution 依赖 pypet，与 NumPy ≥ 2.0 不兼容。
改用 scipy.optimize.differential_evolution（单目标全局搜索）。

搜索空间（4 维）：
  mue  : 皮层兴奋背景 [2.5, 4.5]   |  mui : 皮层抑制背景 [2.5, 5.0]
  g_LK : TCR K 漏电导 [0.02, 0.20] |  g_h : TCR h 电流 [0.02, 0.20]
"""

import os
import sys
import json
import fnmatch
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.optimize import differential_evolution

from neurolib.models.multimodel import MultiModel, ALNNode, ThalamicNode
from neurolib.models.multimodel.builder.base.network import Network
from neurolib.models.multimodel.builder.base.constants import EXC, INH

# ── 步骤 1：网络定义（与 s3_sleep_kernel.py 一致的双节点丘脑–皮层）──────────────

def set_params_glob(model, pattern, value):
    for k in (k for k in model.params if fnmatch.fnmatch(k, pattern)):
        model.params[k] = value


class ThalamoCorticalNetwork(Network):
    name  = "Thalamocortical Motif"
    label = "TCNet"
    sync_variables  = ["network_exc_exc", "network_exc_exc_sq", "network_inh_exc"]
    default_output  = f"r_mean_{EXC}"
    output_vars     = [f"r_mean_{EXC}", f"r_mean_{INH}"]
    _EXC_WITHIN_IDX = [6, 9]   # r_mean_EXC index within ALNNode / ThalamicNode

    def __init__(self):
        aln = ALNNode(exc_seed=42, inh_seed=42)
        th  = ThalamicNode()
        aln.index = 0;  aln.idx_state_var = 0
        th.index  = 1;  th.idx_state_var  = aln.num_state_variables
        for i, node in enumerate([aln, th]):
            for mass in node:
                mass.noise_input_idx = [2 * i + mass.index]
        connectivity = np.array([[0.0, 0.02], [0.15, 0.0]])
        super().__init__(nodes=[aln, th],
                         connectivity_matrix=connectivity,
                         delay_matrix=np.zeros_like(connectivity))

    def _sync(self):
        couplings = sum([node._sync() for node in self], [])
        wi = self._EXC_WITHIN_IDX
        couplings += self._additive_coupling(wi, "network_exc_exc")
        couplings += self._additive_coupling(wi, "network_exc_exc_sq",
                                             connectivity=self.connectivity ** 2)
        couplings += self._additive_coupling(wi, "network_inh_exc")
        return couplings


# ── 步骤 2：从 EEG 构建个人 N3 目标 PSD（非 s1 群体平均）──────────────────────
import mne
mne.set_log_level("WARNING")

SUBJECT_ID      = "SC4001"
EEG_CHANNELS    = ["EEG Fpz-Cz", "EEG Pz-Oz"]
N3_LABELS       = ["Sleep stage 3", "Sleep stage 4"]
ARTIFACT_THRESH = 200e-6      # V, peak-to-peak limit
EPOCH_DURATION  = 30.0        # seconds

manifest = pd.read_csv("data/manifest.csv")
subj_row = manifest[manifest["subject_id"] == SUBJECT_ID]
if len(subj_row) == 0:
    raise ValueError(f"Subject {SUBJECT_ID} not found in manifest.csv")
subj_row = subj_row.iloc[0]

psg_path = subj_row["psg_path"]
hyp_path = subj_row["hypnogram_path"]
print(f"Building personal N3 PSD for {SUBJECT_ID}...")
print(f"  PSG : {psg_path}")
print(f"  Hyp : {hyp_path}")

raw = mne.io.read_raw_edf(
    psg_path,
    include=EEG_CHANNELS,
    preload=True,
    verbose=False,
)
fs_eeg = raw.info["sfreq"]

annotations = mne.read_annotations(hyp_path)
raw.set_annotations(annotations)

event_id = {lbl: idx + 1 for idx, lbl in enumerate(N3_LABELS)}
events, event_dict = mne.events_from_annotations(
    raw, event_id=event_id, verbose=False
)
print(f"  N3 events found: {len(events)}")

epochs_n3 = mne.Epochs(
    raw, events,
    event_id=event_dict,
    tmin=0.0, tmax=EPOCH_DURATION,
    baseline=None,
    preload=True,
    verbose=False,
)

psds_subject = []
n_total  = len(epochs_n3)
n_passed = 0

for ep_idx in range(n_total):
    data = epochs_n3[ep_idx].get_data()[0]  # (n_channels, n_times)

    pp = data.max(axis=1) - data.min(axis=1)
    if np.any(pp > ARTIFACT_THRESH):
        continue

    mean_signal = data.mean(axis=0)

    nperseg = min(int(10.0 * fs_eeg), len(mean_signal))
    f_ep, p_ep = welch(
        mean_signal, fs=fs_eeg,
        nperseg=nperseg, noverlap=nperseg // 2,
        window="hann",
    )

    freq_mask = (f_ep >= 0.5) & (f_ep <= 30.0)
    psds_subject.append(p_ep[freq_mask])
    n_passed += 1

print(f"  N3 epochs passed QC: {n_passed} / {n_total}")

if n_passed == 0:
    raise ValueError(
        f"No N3 epochs passed QC for {SUBJECT_ID}. "
        "Check ARTIFACT_THRESH or EEG channel names."
    )

psds_arr     = np.array(psds_subject)    # (n_passed, n_freqs)
target_psd   = psds_arr.mean(axis=0)    # (n_freqs,)
target_freqs = f_ep[freq_mask]           # (n_freqs,)

np.save(f"data/target_psd_{SUBJECT_ID}.npy",   target_psd)
np.save(f"data/target_freqs_{SUBJECT_ID}.npy", target_freqs)

print(f"  Personal target PSD : shape={target_psd.shape}, "
      f"freq={target_freqs[0]:.1f}–{target_freqs[-1]:.1f} Hz")
print(f"  Saved: data/target_psd_{SUBJECT_ID}.npy")

delta_mask  = (target_freqs >= 0.5) & (target_freqs <= 4.0)
total_mask  = (target_freqs >= 0.5) & (target_freqs <= 30.0)
delta_ratio = target_psd[delta_mask].sum() / (target_psd[total_mask].sum() + 1e-10)
print(f"  Delta ratio (0.5–4 Hz): {delta_ratio:.2f}  "
      f"{'✓ N3-like' if delta_ratio > 0.5 else '⚠ check data'}")

# ── 步骤 3：目标谱 delta/sigma/beta 占比 → 适应度里 TARGET_* 阈值与 sigma 分支 ──
diag_delta = (target_freqs >= 0.5)  & (target_freqs <= 4.0)
diag_sigma = (target_freqs >= 10.0) & (target_freqs <= 15.0)
diag_total = (target_freqs >= 0.5)  & (target_freqs <= 30.0)
_total = target_psd[diag_total].sum() + 1e-30
_delta_ratio = target_psd[diag_delta].sum() / _total
_sigma_ratio = target_psd[diag_sigma].sum() / _total
_beta_ratio  = target_psd[(target_freqs >= 15.0) & (target_freqs <= 30.0)].sum() / _total

print(f"\n=== Target PSD Diagnostics (SC4001 N3) ===")
print(f"  delta (0.5–4 Hz)  占比: {_delta_ratio:.4f}  ({_delta_ratio*100:.1f}%)")
print(f"  sigma (10–15 Hz)  占比: {_sigma_ratio:.4f}  ({_sigma_ratio*100:.1f}%)")
print(f"  beta  (15–30 Hz)  占比: {_beta_ratio:.4f}  ({_beta_ratio*100:.1f}%)")
print(f"  delta/sigma 比值: {_delta_ratio/(_sigma_ratio+1e-10):.1f}x")

TARGET_DELTA_RATIO = float(_delta_ratio)
TARGET_SIGMA_RATIO = float(_sigma_ratio)
print(f"\n  → 适应度函数将使用:")
print(f"     TARGET_DELTA_RATIO = {TARGET_DELTA_RATIO:.4f}")
print(f"     TARGET_SIGMA_RATIO = {TARGET_SIGMA_RATIO:.4f}")
print("=" * 45)

# ── 步骤 4：共享 MultiModel（避免重复 JIT）+ 适应度 _psd_corr ──────────────────

_model = MultiModel(ThalamoCorticalNetwork())
_model.params["backend"]     = "numba"
_model.params["dt"]          = 0.1
_model.params["sampling_dt"] = 1.0
_model.params["duration"]    = 30 * 1000   # 30 s per eval

# fixed Table-3 parameters
set_params_glob(_model, "*ALNMassEXC*.tauA",        1040.0)
set_params_glob(_model, "*ALNMassEXC*.b",             19.5)
set_params_glob(_model, "*ALNMassEXC*.a",              0.0)
set_params_glob(_model, "*ALNMass*.input_0.sigma",    0.05)
set_params_glob(_model, "*TCR*.input_0.sigma",       0.005)
set_params_glob(_model, "*.input_0.tau",               5.0)
set_params_glob(_model, "*TRN*.g_LK",                  0.1)

FS_SIM = 1000.0 / _model.params["sampling_dt"]   # 1000 Hz


def _psd_corr(mue, mui, g_lk, g_h):
    """
    Improved three-step fitness (uses TARGET_DELTA_RATIO / TARGET_SIGMA_RATIO
    computed from the actual subject PSD at load time).

    Step 1: normalise both PSDs by total power → dimensionless relative spectra.
    Step 2: Pearson r on log10(normalised PSD) → spectral shape similarity.
    Step 3a: delta-ratio bonus   — reward matching N3 slow-wave dominance.
    Step 3b: sigma term          — penalise or reward sigma depending on target.
    Combined: 0.6 × r_shape + 0.3 × delta_bonus + 0.1 × sigma_term  ∈ [-1, 1]
    """
    set_params_glob(_model, "*ALNMassEXC*.input_0.mu", mue)
    set_params_glob(_model, "*ALNMassINH*.input_0.mu", mui)
    set_params_glob(_model, "*TCR*.g_LK",              g_lk)
    set_params_glob(_model, "*TCR*.g_h",               g_h)

    try:
        _model.run()
    except Exception:
        try:
            _model.params["backend"] = "jitcdde"
            _model.run()
            _model.params["backend"] = "numba"
        except Exception:
            return -1.0

    r_exc = _model[f"r_mean_{EXC}"]
    r_sim = (r_exc[0] if r_exc.ndim == 2 else r_exc) * 1000   # kHz → Hz

    n_drop = int(5.0 * FS_SIM)
    r_sim  = r_sim[n_drop:]
    if r_sim.max() < 0.1:
        return -1.0    # cortex silent

    nperseg  = min(int(10.0 * FS_SIM), len(r_sim))
    f_s, p_s = welch(r_sim, fs=FS_SIM,
                     nperseg=nperseg, noverlap=nperseg // 2,
                     window="hann")

    f_lo = max(f_s[1], target_freqs[0])
    f_hi = min(f_s[-1], target_freqs[-1])
    mt   = (target_freqs >= f_lo) & (target_freqs <= f_hi)
    ms   = (f_s >= f_lo) & (f_s <= f_hi)
    if ms.sum() < 10 or mt.sum() < 10:
        return -1.0

    p_interp = interp1d(f_s[ms], p_s[ms],
                        bounds_error=False,
                        fill_value=1e-30)(target_freqs[mt])
    t_target  = target_psd[mt]
    freqs_mt  = target_freqs[mt]

    # ── Step 1: normalise (remove 8-order-of-magnitude unit gap) ─────────────
    p_sim_rel = p_interp / (p_interp.sum() + 1e-30)
    p_tgt_rel = t_target / (t_target.sum() + 1e-30)

    # ── Step 2: log-domain Pearson r on normalised spectra ───────────────────
    log_s = np.log10(p_sim_rel + 1e-30)
    log_t = np.log10(p_tgt_rel + 1e-30)
    if np.std(log_s) < 1e-10:
        return -1.0
    r_shape, _ = pearsonr(log_s, log_t)
    if np.isnan(r_shape):
        return -1.0

    # ── Step 3a: delta-ratio bonus ───────────────────────────────────────────
    delta_mask = (freqs_mt >= 0.5) & (freqs_mt <= 4.0)
    total_mask = (freqs_mt >= 0.5) & (freqs_mt <= 30.0)
    if delta_mask.sum() > 0 and total_mask.sum() > 0:
        sim_delta = p_sim_rel[delta_mask].sum() / (p_sim_rel[total_mask].sum() + 1e-30)
        delta_bonus = max(0.0, 1.0 - abs(sim_delta - TARGET_DELTA_RATIO)
                          / (TARGET_DELTA_RATIO + 1e-10))
    else:
        delta_bonus = 0.0

    # ── Step 3b: sigma term (penalty or reward based on target) ──────────────
    sigma_mask = (freqs_mt >= 10.0) & (freqs_mt <= 15.0)
    if sigma_mask.sum() > 0 and total_mask.sum() > 0:
        sim_sigma = p_sim_rel[sigma_mask].sum() / (p_sim_rel[total_mask].sum() + 1e-30)
        if TARGET_SIGMA_RATIO < 0.05:
            # target has almost no sigma → penalise excess spindle activity
            sigma_term = -max(0.0, sim_sigma - TARGET_SIGMA_RATIO * 2.0) * 5.0
        else:
            # target has measurable sigma → reward closeness
            sigma_term = max(0.0, 1.0 - abs(sim_sigma - TARGET_SIGMA_RATIO)
                             / (TARGET_SIGMA_RATIO + 1e-10)) * 0.5
    else:
        sigma_term = 0.0

    score = 0.6 * r_shape + 0.3 * delta_bonus + 0.1 * sigma_term
    return float(np.clip(score, -1.0, 1.0))


# ── 步骤 5：差分进化簿记 — _objective 最小化 -score，记录 all_records，callback ─

POP_INIT = 40    # initial population  (popsize × n_params = 10 × 4)
POP_SIZE = 20    # popsize per generation (5 × 4 = 20)
N_GEN    = 10    # max generations
# differential_evolution: popsize argument = pop_total / n_params
_N_PARAMS = 4
_DE_POPSIZE = max(5, POP_SIZE // _N_PARAMS)   # = 5 → 20 individuals/gen

all_records  = []   # list of dicts: {mue, mui, g_LK, g_h, psd_corr, gen}
_gen_counter = [0]
_eval_counter = [0]
_t0 = [time.time()]


def _objective(x):
    """Scipy DE objective: minimise –r (so maximise r)."""
    mue, mui, g_lk, g_h = x
    r = _psd_corr(mue, mui, g_lk, g_h)
    _eval_counter[0] += 1
    all_records.append({
        "mue": mue, "mui": mui, "g_LK": g_lk, "g_h": g_h,
        "psd_corr": r, "gen": _gen_counter[0],
    })
    return -r   # minimise


def _callback(xk, convergence):
    """Called after each generation."""
    _gen_counter[0] += 1
    gen_df = pd.DataFrame(all_records)
    best_r = gen_df["psd_corr"].max()
    elapsed = time.time() - _t0[0]
    print(f"  Gen {_gen_counter[0]:>3}/{N_GEN}  "
          f"best_r={best_r:+.4f}  "
          f"evals={_eval_counter[0]}  "
          f"elapsed={elapsed:.0f}s")


# ── 步骤 6：运行 differential_evolution（bounds、workers=1）────────────────────

BOUNDS = [
    (2.5, 4.5),    # mue
    (2.5, 5.0),    # mui
    (0.02, 0.20),  # g_LK
    (0.02, 0.20),  # g_h
]

print(f"\nEvolution (scipy differential_evolution, GA-style, NSGA-II equivalent)")
print(f"  popsize × n_params = {_DE_POPSIZE} × {_N_PARAMS} = {_DE_POPSIZE * _N_PARAMS} individuals/gen")
print(f"  maxiter = {N_GEN}  |  total evals ≈ {_DE_POPSIZE * _N_PARAMS * (N_GEN + 1)}")
print(f"  init population = {POP_INIT}  (via init='latinhypercube')\n")

_t0[0] = time.time()

result = differential_evolution(
    _objective,
    bounds=BOUNDS,
    strategy="best1bin",
    maxiter=N_GEN,
    popsize=_DE_POPSIZE,
    tol=1e-4,
    mutation=(0.5, 1.0),
    recombination=0.7,
    seed=42,
    callback=_callback,
    polish=False,
    init="latinhypercube",
    workers=1,          # must be 1: numba JIT not fork-safe
    updating="immediate",
)

print(f"\nEvolution complete  (total evals = {_eval_counter[0]})")

# ── 步骤 7：最优参数 → patient_params*.json + evolution_records.csv ────────────

df = pd.DataFrame(all_records)
print(f"\nTotal individuals recorded : {len(df)}")
print(f"Best  psd_corr : {df['psd_corr'].max():+.4f}")
print(f"Mean  psd_corr : {df['psd_corr'].mean():+.4f}")

best_row = df.loc[df["psd_corr"].idxmax()]

patient_params = {
    "mue":      float(best_row["mue"]),
    "mui":      float(best_row["mui"]),
    "g_LK":     float(best_row["g_LK"]),
    "g_h":      float(best_row["g_h"]),
    "b":        19.5,
    "tauA":     1040.0,
    "psd_corr": float(best_row["psd_corr"]),
    "n_gen":    N_GEN,
}

os.makedirs("data",    exist_ok=True)
os.makedirs("outputs", exist_ok=True)

params_path = f"data/patient_params_{SUBJECT_ID}.json"
patient_params["subject_id"]      = SUBJECT_ID
patient_params["n3_epochs_used"]  = n_passed

with open(params_path, "w") as fh:
    json.dump(patient_params, fh, indent=2)

with open("data/patient_params.json", "w") as fh:
    json.dump(patient_params, fh, indent=2)

print(f"\nBest parameters:")
for k, v in patient_params.items():
    print(f"  {k}: {v}")
print(f"Saved: {params_path}")
print(f"Saved: data/patient_params.json  (symlink to {SUBJECT_ID})")

# save full results table
df.to_csv("outputs/evolution_records.csv", index=False)
print("Saved: outputs/evolution_records.csv")

# ── 步骤 8：evolution_result.png（进化曲线 / 适应度分布 / 归一化 Sim vs Target）──
#           文末为 Session 2-B 终端校验

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# left: evolution curve (best r per generation)
valid_df = df[df["psd_corr"] > -0.5].copy()
gen_best = valid_df.groupby("gen")["psd_corr"].max()
if len(gen_best) > 0:
    axes[0].plot(gen_best.index, gen_best.values, "o-",
                 color="#534AB7", lw=2)
axes[0].set_xlabel("Generation")
axes[0].set_ylabel("Best PSD Pearson r")
axes[0].set_title("Evolution curve")
axes[0].set_ylim(-0.1, 1.0)

# middle: fitness distribution
axes[1].hist(valid_df["psd_corr"], bins=20,
             color="#1D9E75", edgecolor="white")
axes[1].axvline(df["psd_corr"].max(), color="red", ls="--",
                label=f"Best = {df['psd_corr'].max():.3f}")
axes[1].set_xlabel("PSD Pearson r")
axes[1].set_ylabel("Count")
axes[1].set_title("Fitness distribution")
axes[1].legend()

# right: best-params sim PSD vs target
print("\nRunning best-parameter simulation for final plot...")
bm = MultiModel(ThalamoCorticalNetwork())
bm.params["backend"]     = "numba"
bm.params["dt"]          = 0.1
bm.params["sampling_dt"] = 1.0
bm.params["duration"]    = 30 * 1000
set_params_glob(bm, "*ALNMassEXC*.tauA",        1040.0)
set_params_glob(bm, "*ALNMassEXC*.b",             19.5)
set_params_glob(bm, "*ALNMassEXC*.a",              0.0)
set_params_glob(bm, "*ALNMass*.input_0.sigma",    0.05)
set_params_glob(bm, "*TCR*.input_0.sigma",       0.005)
set_params_glob(bm, "*.input_0.tau",               5.0)
set_params_glob(bm, "*TRN*.g_LK",                  0.1)
set_params_glob(bm, "*ALNMassEXC*.input_0.mu", patient_params["mue"])
set_params_glob(bm, "*ALNMassINH*.input_0.mu", patient_params["mui"])
set_params_glob(bm, "*TCR*.g_LK",              patient_params["g_LK"])
set_params_glob(bm, "*TCR*.g_h",               patient_params["g_h"])
bm.run()

r_b  = bm[f"r_mean_{EXC}"]
r_b  = (r_b[0] if r_b.ndim == 2 else r_b) * 1000
r_b  = r_b[int(5 * FS_SIM):]
np_b = min(int(10 * FS_SIM), len(r_b))
f_b, p_b = welch(r_b, fs=FS_SIM, nperseg=np_b, noverlap=np_b // 2)
fm   = (f_b >= 0.5) & (f_b <= 30)

# normalise to match fitness function (removes unit gap for visual comparison)
p_b_rel = p_b[fm] / (p_b[fm].sum() + 1e-30)
t_rel   = target_psd / (target_psd.sum() + 1e-30)

axes[2].semilogy(target_freqs, t_rel,
                 "k-", lw=2,
                 label=(f"Target SC4001 N3 (normalised)\n"
                        f"delta={TARGET_DELTA_RATIO:.2f}, "
                        f"sigma={TARGET_SIGMA_RATIO:.3f}"))
axes[2].semilogy(f_b[fm], p_b_rel,
                 color="#534AB7", lw=2, ls="--",
                 label=f"Simulation (score={patient_params['psd_corr']:.3f})")
axes[2].axvspan(0.5,  4.0,  alpha=0.12, color="orange", label="Delta (0.5–4 Hz)")
axes[2].axvspan(10.0, 15.0, alpha=0.12, color="green",  label="Sigma (10–15 Hz)")
axes[2].set_xlabel("Frequency [Hz]")
axes[2].set_ylabel("Normalised power (a.u.)")
axes[2].set_xlim(0.5, 30)
axes[2].legend(fontsize=8)

bm_fm_freqs  = f_b[fm]
bm_delta_m   = (bm_fm_freqs >= 0.5) & (bm_fm_freqs <= 4.0)
bm_total_m   = (bm_fm_freqs >= 0.5) & (bm_fm_freqs <= 30.0)
if bm_delta_m.sum() > 0:
    sim_delta_final = p_b_rel[bm_delta_m].sum() / (p_b_rel[bm_total_m].sum() + 1e-30)
    axes[2].set_title(
        f"Best fit (normalised)\n"
        f"sim delta={sim_delta_final:.2f} | tgt delta={TARGET_DELTA_RATIO:.2f}",
        fontsize=10,
    )
else:
    axes[2].set_title("Best fit: Sim vs Target (normalised)", fontsize=10)

plt.suptitle(
    f"Session 2-B: Evolutionary fitting — {SUBJECT_ID}  "
    f"(score={patient_params['psd_corr']:.3f}, "
    f"N3 epochs={n_passed})\n"
    f"Fitness = 0.6×shape_r + 0.3×delta_bonus + 0.1×sigma_term",
    fontsize=11,
)
plt.tight_layout()
plt.savefig("outputs/evolution_result_0417_v0.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/evolution_result_0417_v0.png")

# ── 步骤 8（续）：最终校验（与上图同一步输出）──────────────────────────────────

print("\n=== Session 2-B Validation ===")
r = patient_params["psd_corr"]
if r > 0.5:
    print(f"✓ PSD Pearson r = {r:.3f} > 0.5  (good fit)")
elif r > 0.3:
    print(f"~ PSD Pearson r = {r:.3f}  (acceptable for MVP)")
    print("  To improve: increase _DE_POPSIZE=10, N_GEN=20")
else:
    print(f"✗ PSD Pearson r = {r:.3f}  (poor fit)")
    print("  Check: cortex not silent, mue search range correct")

print("data/patient_params_0417_v0.json ready for Session 2-C ✓")
sys.exit(0)
