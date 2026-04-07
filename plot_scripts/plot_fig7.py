"""
plot_fig7_v2.py
===============
Standalone Fig. 7 plotting for V2 results.

Changes vs plot_fig7.py (v1):
  - Reads _v2 JSON and CSV files
  - Green dots filtered by shape_r > 0.7 AND dynamics_score >= 0.67
  - Cortex contour uses DOWN-state + delta-ratio criterion (not raw amplitude)
  - Thalamus contour uses spindle FWHM > 2 Hz criterion
  - All output files tagged _v2

Usage:
  python models/plot_fig7_v2.py
"""

import os, json, fnmatch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch
import mne
mne.set_log_level("WARNING")

from neurolib.models.multimodel import MultiModel, ALNNode, ThalamicNode
from neurolib.models.multimodel.builder.base.network import Network
from neurolib.models.multimodel.builder.base.constants import EXC, INH

try:
    from fooof import FOOOF
    HAS_FOOOF = True
except ImportError:
    HAS_FOOOF = False
    print("[warn] fooof not installed, 1/f lines will be omitted")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUBJECT_ID   = "SC4001"
PARAMS_PATH  = f"data/patient_params_fig7_v2_{SUBJECT_ID}.json"
RECORDS_PATH = "outputs/evolution_fig7_v2_records.csv"
F_LO, F_HI  = 0.5, 20.0
FS_SIM       = 1000.0
SCAN_N       = 20

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Load best parameters & evolution records
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with open(PARAMS_PATH, "r") as fh:
    best_params = json.load(fh)

print(f"Loaded V2 parameters from {PARAMS_PATH}")
for k, v in best_params.items():
    print(f"  {k}: {v}")

df = None
if os.path.exists(RECORDS_PATH):
    df = pd.read_csv(RECORDS_PATH)
    print(f"Loaded {len(df)} evolution records")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Network definition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def set_params_glob(model, pattern, value):
    for k in (k for k in model.params if fnmatch.fnmatch(k, pattern)):
        model.params[k] = value


class ThalamoCorticalNetwork(Network):
    name  = "Thalamocortical Motif"
    label = "TCNet"
    sync_variables  = ["network_exc_exc", "network_exc_exc_sq", "network_inh_exc"]
    default_output  = f"r_mean_{EXC}"
    output_vars     = [f"r_mean_{EXC}", f"r_mean_{INH}"]
    _EXC_WITHIN_IDX = [6, 9]

    def __init__(self, c_th2ctx=0.02, c_ctx2th=0.15):
        aln = ALNNode(exc_seed=42, inh_seed=42)
        th  = ThalamicNode()
        aln.index = 0;  aln.idx_state_var = 0
        th.index  = 1;  th.idx_state_var  = aln.num_state_variables
        for i, node in enumerate([aln, th]):
            for mass in node:
                mass.noise_input_idx = [2 * i + mass.index]
        connectivity = np.array([[0.0, c_th2ctx], [c_ctx2th, 0.0]])
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


def build_model(mue, mui, b, tauA, g_lk, g_h, c_th2ctx, c_ctx2th, duration=60000):
    net = ThalamoCorticalNetwork(c_th2ctx=c_th2ctx, c_ctx2th=c_ctx2th)
    m = MultiModel(net)
    m.params["backend"]     = "numba"
    m.params["dt"]          = 0.1
    m.params["sampling_dt"] = 1.0
    m.params["duration"]    = duration
    set_params_glob(m, "*ALNMassEXC*.input_0.mu", mue)
    set_params_glob(m, "*ALNMassINH*.input_0.mu", mui)
    set_params_glob(m, "*ALNMassEXC*.b",          b)
    set_params_glob(m, "*ALNMassEXC*.tauA",       tauA)
    set_params_glob(m, "*ALNMassEXC*.a",          0.0)
    set_params_glob(m, "*ALNMass*.input_0.sigma",  0.05)
    set_params_glob(m, "*TCR*.input_0.sigma",      0.005)
    set_params_glob(m, "*.input_0.tau",            5.0)
    set_params_glob(m, "*TRN*.g_LK",              0.1)
    set_params_glob(m, "*TCR*.g_LK",              g_lk)
    set_params_glob(m, "*TCR*.g_h",               g_h)
    return m


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Bifurcation scan — cortex (FIXED criterion)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v1 bug: used raw amplitude > 10 Hz → almost everything passed
# v2 fix: requires BOTH min(r_E) < 1 Hz (true DOWN state)
#         AND delta_ratio > 0.3 (delta power dominates)

SCAN_CACHE_CTX = "outputs/_scan_cortex_v2.npz"
SCAN_CACHE_TH  = "outputs/_scan_thalamus_v2.npz"

mue_range = np.linspace(2.5, 4.5, SCAN_N)
mui_range = np.linspace(2.5, 5.0, SCAN_N)

if os.path.exists(SCAN_CACHE_CTX):
    print(f"\nLoading cached cortex scan")
    _d = np.load(SCAN_CACHE_CTX)
    so_map = _d["so_map"]
    mue_range = _d["mue_range"]
    mui_range = _d["mui_range"]
else:
    print(f"\nRunning cortex bifurcation scan ({SCAN_N}x{SCAN_N})...")
    so_map = np.zeros((SCAN_N, SCAN_N))

    for i, mue_val in enumerate(mue_range):
        for j, mui_val in enumerate(mui_range):
            try:
                m = build_model(
                    mue=mue_val, mui=mui_val,
                    b=best_params["b"], tauA=best_params["tauA"],
                    g_lk=best_params["g_LK"], g_h=best_params["g_h"],
                    c_th2ctx=best_params["c_th2ctx"],
                    c_ctx2th=best_params["c_ctx2th"],
                    duration=10_000,
                )
                m.run()
                r_exc = m[f"r_mean_{EXC}"]
                r_ctx = (r_exc[0] if r_exc.ndim == 2 else r_exc) * 1000
                r_ctx = r_ctx[int(2.0 * FS_SIM):]

                # FIXED: check for real DOWN state + delta dominance
                has_down = r_ctx.min() < 1.0

                f_c, p_c = welch(r_ctx, fs=FS_SIM,
                                 nperseg=min(int(5*FS_SIM), len(r_ctx)),
                                 noverlap=min(int(2.5*FS_SIM), len(r_ctx)//2),
                                 window="hann")
                delta_m = (f_c >= 0.2) & (f_c <= 1.5)
                total_m = (f_c >= 0.2) & (f_c <= 30.0)
                if delta_m.any() and total_m.any():
                    d_ratio = p_c[delta_m].sum() / (p_c[total_m].sum() + 1e-30)
                else:
                    d_ratio = 0.0

                so_map[j, i] = d_ratio if has_down else 0.0
            except Exception:
                so_map[j, i] = 0.0

        pct = (i + 1) / SCAN_N * 100
        print(f"  Cortex scan: {pct:.0f}%")

    os.makedirs("outputs", exist_ok=True)
    np.savez(SCAN_CACHE_CTX, so_map=so_map, mue_range=mue_range, mui_range=mui_range)
    print(f"  Cached")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Bifurcation scan — thalamus (FIXED criterion)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v1: used spindle peak/background ratio
# v2: uses spindle FWHM > 2 Hz (distinguishes waxing-waning from limit cycle)

glk_range = np.linspace(0.03, 0.15, SCAN_N)
gh_range  = np.linspace(0.03, 0.15, SCAN_N)

if os.path.exists(SCAN_CACHE_TH):
    print(f"Loading cached thalamus scan")
    _d = np.load(SCAN_CACHE_TH)
    sp_map = _d["sp_map"]
    glk_range = _d["glk_range"]
    gh_range  = _d["gh_range"]
else:
    print(f"\nRunning thalamus bifurcation scan ({SCAN_N}x{SCAN_N})...")
    sp_map = np.zeros((SCAN_N, SCAN_N))

    for i, glk_val in enumerate(glk_range):
        for j, gh_val in enumerate(gh_range):
            try:
                m = build_model(
                    mue=best_params["mue"], mui=best_params["mui"],
                    b=best_params["b"], tauA=best_params["tauA"],
                    g_lk=glk_val, g_h=gh_val,
                    c_th2ctx=best_params["c_th2ctx"],
                    c_ctx2th=best_params["c_ctx2th"],
                    duration=10_000,
                )
                m.run()
                r_exc = m[f"r_mean_{EXC}"]
                if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
                    r_th = r_exc[1, :] * 1000
                else:
                    r_th = np.zeros(1000)

                r_th = r_th[int(2.0 * FS_SIM):]
                if r_th.max() < 0.01:
                    sp_map[j, i] = 0.0
                    continue

                # FIXED: measure spindle FWHM instead of peak/background ratio
                nperseg_th = min(int(5.0 * FS_SIM), len(r_th))
                f_th, p_th = welch(r_th, fs=FS_SIM, nperseg=nperseg_th,
                                   noverlap=nperseg_th//2, window="hann")
                sp_band = (f_th >= 8.0) & (f_th <= 16.0)
                if sp_band.any() and p_th[sp_band].max() > 0:
                    p_sp = p_th[sp_band]
                    f_sp = f_th[sp_band]
                    half_power = p_sp.max() / 2.0
                    above_half = f_sp[p_sp >= half_power]
                    if len(above_half) >= 2:
                        sp_map[j, i] = above_half[-1] - above_half[0]  # FWHM in Hz
                    else:
                        sp_map[j, i] = 0.0
                else:
                    sp_map[j, i] = 0.0
            except Exception:
                sp_map[j, i] = 0.0

        pct = (i + 1) / SCAN_N * 100
        print(f"  Thalamus scan: {pct:.0f}%")

    os.makedirs("outputs", exist_ok=True)
    np.savez(SCAN_CACHE_TH, sp_map=sp_map, glk_range=glk_range, gh_range=gh_range)
    print(f"  Cached")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Run 60s best-params simulation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\nRunning 60s simulation with V2 best parameters...")
bm = build_model(
    mue=best_params["mue"], mui=best_params["mui"],
    b=best_params["b"], tauA=best_params["tauA"],
    g_lk=best_params["g_LK"], g_h=best_params["g_h"],
    c_th2ctx=best_params["c_th2ctx"], c_ctx2th=best_params["c_ctx2th"],
    duration=60_000,
)
bm.run()
print("Simulation complete")

r_exc_out = bm[f"r_mean_{EXC}"]
if r_exc_out.ndim == 2 and r_exc_out.shape[0] >= 2:
    rE_cortex   = r_exc_out[0, :] * 1000
    rE_thalamus = r_exc_out[1, :] * 1000
else:
    rE_cortex   = (r_exc_out[0] if r_exc_out.ndim == 2 else r_exc_out) * 1000
    rE_thalamus = np.zeros_like(rE_cortex)

t_s = bm['t']
if t_s[-1] > 1000:
    t_s = t_s / 1000.0

print(f"  Cortex  : [{rE_cortex.min():.2f}, {rE_cortex.max():.2f}] Hz")
print(f"  Thalamus: [{rE_thalamus.min():.2f}, {rE_thalamus.max():.2f}] Hz")

# Cortex PSD
n_drop = int(5.0 * FS_SIM)
r_clean = rE_cortex[n_drop:]
nperseg = min(int(10.0 * FS_SIM), len(r_clean))
f_sim, p_sim = welch(r_clean, fs=FS_SIM, nperseg=nperseg,
                     noverlap=nperseg//2, window="hann")
mf = (f_sim >= F_LO) & (f_sim <= F_HI)
f_sim, p_sim = f_sim[mf], p_sim[mf]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Load target EEG PSD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EEG_CHANNELS    = ["EEG Fpz-Cz", "EEG Pz-Oz"]
N3_LABELS       = ["Sleep stage 3", "Sleep stage 4"]
ARTIFACT_THRESH = 200e-6

manifest = pd.read_csv("data/manifest.csv")
subj_row = manifest[manifest["subject_id"] == SUBJECT_ID].iloc[0]

raw = mne.io.read_raw_edf(subj_row["psg_path"], include=EEG_CHANNELS,
                           preload=True, verbose=False)
fs_eeg = raw.info["sfreq"]
raw.set_annotations(mne.read_annotations(subj_row["hypnogram_path"]))

event_id = {lbl: idx+1 for idx, lbl in enumerate(N3_LABELS)}
events, event_dict = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
epochs_n3 = mne.Epochs(raw, events, event_id=event_dict,
                       tmin=0.0, tmax=30.0, baseline=None,
                       preload=True, verbose=False)

psds_subject = []
for ep_idx in range(len(epochs_n3)):
    data = epochs_n3[ep_idx].get_data()[0]
    if np.any((data.max(axis=1) - data.min(axis=1)) > ARTIFACT_THRESH):
        continue
    mean_signal = data.mean(axis=0)
    nps = min(int(10.0 * fs_eeg), len(mean_signal))
    f_ep, p_ep = welch(mean_signal, fs=fs_eeg, nperseg=nps,
                       noverlap=nps//2, window="hann")
    freq_mask = (f_ep >= F_LO) & (f_ep <= F_HI)
    psds_subject.append(p_ep[freq_mask])

target_psd   = np.mean(psds_subject, axis=0)
target_freqs = f_ep[freq_mask]
n_passed     = len(psds_subject)
print(f"\nTarget EEG: {n_passed} N3 epochs")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. FOOOF on target and simulation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tgt_aperiodic = None
sim_aperiodic = None
tgt_fooof_freqs = target_freqs
sim_fooof_freqs = f_sim

if HAS_FOOOF:
    fm_tgt = FOOOF(peak_width_limits=[1.0, 8.0], max_n_peaks=4,
                   min_peak_height=0.05, aperiodic_mode='fixed')
    fm_tgt.fit(target_freqs, target_psd, [F_LO, F_HI])
    tgt_aperiodic = fm_tgt._ap_fit
    tgt_fooof_freqs = fm_tgt.freqs

    try:
        fm_sim = FOOOF(peak_width_limits=[1.0, 8.0], max_n_peaks=4,
                       min_peak_height=0.05, aperiodic_mode='fixed')
        fm_sim.fit(f_sim, p_sim, [F_LO, F_HI])
        sim_aperiodic = fm_sim._ap_fit
        sim_fooof_freqs = fm_sim.freqs
    except Exception as e:
        print(f"  FOOOF sim failed: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. PLOT — Fig. 7 V2 (a)(b)(c)(d)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\nGenerating Fig. 7 V2...")
fig = plt.figure(figsize=(17, 14))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35,
                       width_ratios=[0.7, 1.15, 1.15],
                       height_ratios=[1, 1])


# ══════════════ (a) Network schematic ══════════════
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_xlim(0, 10); ax_a.set_ylim(0, 10)
ax_a.set_aspect('equal'); ax_a.axis('off')

ax_a.add_patch(plt.Rectangle((1, 5.5), 8, 3.5, facecolor='#E6F1FB',
                               edgecolor='#185FA5', lw=2))
ax_a.text(5, 8.5, 'Cortex', ha='center', fontsize=11, fontweight='bold', color='#0C447C')
for cx, cy, lbl, col in [(3, 6.8, 'EXC', '#d62728'), (7, 6.8, 'INH', '#1f77b4')]:
    ax_a.add_patch(plt.Circle((cx, cy), 0.8, color=col, alpha=0.7))
    ax_a.text(cx, cy, lbl, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
ax_a.annotate('', xy=(6.2, 7.0), xytext=(3.8, 7.0),
              arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5))
ax_a.annotate('', xy=(3.8, 6.6), xytext=(6.2, 6.6),
              arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5))

ax_a.add_patch(plt.Rectangle((1, 0.5), 8, 3.5, facecolor='#FAEEDA',
                               edgecolor='#854F0B', lw=2))
ax_a.text(5, 3.5, 'Thalamus', ha='center', fontsize=11, fontweight='bold', color='#633806')
for cx, cy, lbl, col in [(3, 1.8, 'TCR', '#d62728'), (7, 1.8, 'TRN', '#1f77b4')]:
    ax_a.add_patch(plt.Circle((cx, cy), 0.8, color=col, alpha=0.7))
    ax_a.text(cx, cy, lbl, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

c1 = best_params["c_th2ctx"]
c2 = best_params["c_ctx2th"]
ax_a.annotate('', xy=(2.5, 5.5), xytext=(2.5, 4.0),
              arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax_a.text(0.5, 4.7, f'{c1:.3f}', fontsize=9, color='black')
ax_a.annotate('', xy=(7.5, 4.0), xytext=(7.5, 5.5),
              arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))
ax_a.text(8.2, 4.7, f'{c2:.3f}', fontsize=9, color='gray')

params_txt = (f"$\\mu_E$={best_params['mue']:.2f}  $\\mu_I$={best_params['mui']:.2f}\n"
              f"b={best_params['b']:.1f}  $\\tau_A$={best_params['tauA']:.0f}\n"
              f"$g_{{LK}}$={best_params['g_LK']:.3f}  $g_h$={best_params['g_h']:.3f}")
ax_a.text(5, 9.8, params_txt, ha='center', fontsize=8, color='gray', va='top', family='monospace')
ax_a.text(0.3, 9.5, '(a)', fontsize=16, fontweight='bold')


# ══════════════ (b) LEFT: Cortex parameter space ══════════════
ax_b1 = fig.add_subplot(gs[0, 1])

# Blue contour: slow oscillation region (delta_ratio > 0.3 with DOWN state)
SO_THRESH = 0.3
if so_map.max() > SO_THRESH:
    ax_b1.contour(mue_range, mui_range, so_map,
                  levels=[SO_THRESH], colors=['#185FA5'], linewidths=1.5)
    ax_b1.contourf(mue_range, mui_range, so_map,
                   levels=[SO_THRESH, so_map.max() + 0.01],
                   colors=['#185FA5'], alpha=0.08)

# Green dots: good fits with correct dynamics
if df is not None and len(df) > 0:
    has_dyn = "dynamics_score" in df.columns
    if has_dyn:
        good = df[(df["shape_r"] > 0.7) & (df["dynamics_score"] >= 0.67)]
    else:
        good = df[df["shape_r"] > 0.7]
    bad = df[~df.index.isin(good.index)]
    bad = bad[bad["score"] > 0]

    ax_b1.scatter(bad["mue"], bad["mui"], c='gray', alpha=0.15, s=8, zorder=2)
    if len(good) > 0:
        ax_b1.scatter(good["mue"], good["mui"], c='#2ca02c', s=20,
                      edgecolors='white', linewidths=0.3, alpha=0.7, zorder=3,
                      label=f'Good fit + dynamics (n={len(good)})')

ax_b1.plot(best_params["mue"], best_params["mui"], '*', color='red',
           ms=14, markeredgecolor='black', markeredgewidth=1.2, zorder=10,
           label='Best')
ax_b1.set_xlabel('Input to EXC [mV/ms]')
ax_b1.set_ylabel('Input to INH [mV/ms]')
ax_b1.set_title('Cortex', fontweight='bold')
ax_b1.legend(fontsize=7, loc='upper left')
ax_b1.text(0.02, 0.02, 'Blue = bistable SO region',
           transform=ax_b1.transAxes, fontsize=7, color='#185FA5')
ax_b1.text(-0.12, 1.05, '(b)', transform=ax_b1.transAxes, fontsize=16, fontweight='bold')


# ══════════════ (b) RIGHT: Thalamus parameter space ══════════════
ax_b2 = fig.add_subplot(gs[0, 2])

# Blue contour: waxing-waning spindle region (FWHM > 2 Hz)
SP_THRESH = 2.0
if sp_map.max() > SP_THRESH:
    ax_b2.contour(glk_range, gh_range, sp_map,
                  levels=[SP_THRESH], colors=['#185FA5'], linewidths=1.5)
    ax_b2.contourf(glk_range, gh_range, sp_map,
                   levels=[SP_THRESH, sp_map.max() + 0.01],
                   colors=['#185FA5'], alpha=0.08)

if df is not None and len(df) > 0:
    ax_b2.scatter(bad["g_LK"], bad["g_h"], c='gray', alpha=0.15, s=8, zorder=2)
    if len(good) > 0:
        ax_b2.scatter(good["g_LK"], good["g_h"], c='#2ca02c', s=20,
                      edgecolors='white', linewidths=0.3, alpha=0.7, zorder=3,
                      label=f'Good fit + dynamics (n={len(good)})')

ax_b2.plot(best_params["g_LK"], best_params["g_h"], '*', color='red',
           ms=14, markeredgecolor='black', markeredgewidth=1.2, zorder=10,
           label='Best')
ax_b2.set_xlabel('Thalamus $g_{LK}$ [mS/cm$^2$]')
ax_b2.set_ylabel('Thalamus $g_h$ [mS/cm$^2$]')
ax_b2.set_title('Thalamus', fontweight='bold')
ax_b2.legend(fontsize=7, loc='upper left')
ax_b2.text(0.02, 0.02, 'Blue = waxing-waning spindle region',
           transform=ax_b2.transAxes, fontsize=7, color='#185FA5')


# ══════════════ (c) Time series (16-24 s) ══════════════
gs_c = gridspec.GridSpecFromSubplotSpec(2, 1,
        subplot_spec=gs[1, 0:2], hspace=0.4)

t_start, t_end = 16.0, 24.0
mask_t = (t_s >= t_start) & (t_s <= t_end)

ax_c1 = fig.add_subplot(gs_c[0])
ax_c1.plot(t_s[mask_t], rE_cortex[mask_t], '#534AB7', lw=0.6)
ax_c1.set_ylabel('$r_E$ [Hz]')
ax_c1.set_ylim(bottom=-1)
ax_c1.set_title('Cortex EXC — slow oscillations (V2)', fontsize=10)
ax_c1.text(0.02, 0.88, '(c)', transform=ax_c1.transAxes, fontsize=16, fontweight='bold')

ax_c2 = fig.add_subplot(gs_c[1])
ax_c2.plot(t_s[mask_t], rE_thalamus[mask_t], '#1D9E75', lw=0.6)
ax_c2.set_xlabel('Time [s]')
ax_c2.set_ylabel('$r_{TCR}$ [Hz]')
ax_c2.set_ylim(bottom=-1)
ax_c2.set_title('Thalamus TCR — spindle oscillations (V2)', fontsize=10)


# ══════════════ (d) Power spectra ══════════════
gs_d = gridspec.GridSpecFromSubplotSpec(2, 1,
        subplot_spec=gs[1, 2], hspace=0.4)

ax_d1 = fig.add_subplot(gs_d[0])
ax_d1.semilogy(target_freqs, target_psd, 'k', lw=1.5,
               label=f'Target {SUBJECT_ID} N3 (n={n_passed})')
if tgt_aperiodic is not None:
    ax_d1.semilogy(tgt_fooof_freqs, 10**tgt_aperiodic, 'b--', lw=1,
                   alpha=0.7, label='1/f fit')
ax_d1.axvspan(0.2, 1.5, alpha=0.08, color='orange')
ax_d1.axvspan(10, 14, alpha=0.08, color='green')
ax_d1.set_ylabel('Power [V$^2$/Hz]')
ax_d1.set_xlim(F_LO, F_HI)
ax_d1.legend(fontsize=7)
ax_d1.set_title('EEG', fontsize=10, loc='right')
ax_d1.text(0.02, 0.88, '(d)', transform=ax_d1.transAxes, fontsize=16, fontweight='bold')

ax_d2 = fig.add_subplot(gs_d[1])
ax_d2.semilogy(f_sim, p_sim, '#534AB7', lw=1.5, label='Cortex EXC (sim)')
if sim_aperiodic is not None:
    ax_d2.semilogy(sim_fooof_freqs, 10**sim_aperiodic, 'b--', lw=1,
                   alpha=0.7, label='1/f fit')
ax_d2.axvspan(0.2, 1.5, alpha=0.08, color='orange', label='SO')
ax_d2.axvspan(10, 14, alpha=0.08, color='green', label='Spindle')
ax_d2.set_xlabel('Frequency [Hz]')
ax_d2.set_ylabel('Power [Hz$^2$/Hz]')
ax_d2.set_xlim(F_LO, F_HI)
ax_d2.legend(fontsize=7)
ax_d2.set_title('Simulated firing rate (V2)', fontsize=10, loc='right')

# Score annotation with dynamics info
dyn_str = f"dyn={best_params.get('dynamics_score', 'N/A')}"
ax_d2.text(0.98, 0.05,
           f"shape_r={best_params['shape_r']:.3f}  {dyn_str}",
           transform=ax_d2.transAxes, ha='right', fontsize=8, color='gray')


# ══════════════ Title and save ══════════════
fig.suptitle(
    f"Fig. 7 V2: Thalamocortical motif — {SUBJECT_ID} "
    f"(physics-constrained)\n"
    f"score={best_params['score']:.3f}, shape_r={best_params['shape_r']:.3f}, "
    f"dynamics={best_params.get('dynamics_score', 'N/A')}, "
    f"SO={best_params['so_power']:.2f}, spindle={best_params['spindle_power']:.2f}",
    fontsize=12, fontweight='bold',
)

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/fig7_personalized_v2.png", dpi=150, bbox_inches="tight")
print("\n✓ Saved: outputs/fig7_personalized_v2.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'='*55}")
print(f"Fig. 7 V2 Summary for {SUBJECT_ID}")
print(f"{'='*55}")
print(f"  score={best_params['score']:.4f}  shape_r={best_params['shape_r']:.4f}")
print(f"  dynamics={best_params.get('dynamics_score', 'N/A')}")
print(f"  SO={best_params['so_power']:.4f}  spindle={best_params['spindle_power']:.4f}")
print(f"  Cortex:   mue={best_params['mue']:.3f}  mui={best_params['mui']:.3f}  "
      f"b={best_params['b']:.1f}  tauA={best_params['tauA']:.0f}")
print(f"  Thalamus: g_LK={best_params['g_LK']:.4f}  g_h={best_params['g_h']:.4f}")
print(f"  Coupling: th->ctx={best_params['c_th2ctx']:.4f}  ctx->th={best_params['c_ctx2th']:.4f}")
print(f"\nScan caches: {SCAN_CACHE_CTX}, {SCAN_CACHE_TH}")
print(f"Delete them to force rescan.")