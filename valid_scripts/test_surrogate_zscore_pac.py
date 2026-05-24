"""
test_surrogate_zscore_pac.py
=============================
Run surrogate-normalized Tort MI on:
    (1) SC4001 raw EEG: full 4260 s concat AND longest 480 s contig
    (2) Seed B simulator: r_ctx phase × r_thal amp (cross-channel)
    (3) SC4001 raw EEG vs synthetic null (paranoia check):
        pure white-noise EEG-shape signal should give z ≈ 0

Reports MI_obs, MI_null_mean, MI_null_std, z for each.
Decision criterion: z > 2 means PAC signal exists above null.
"""

import sys, os, warnings, importlib.util, time
from pathlib import Path
warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(_ROOT))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "S4_v7_repair"))

import numpy as np
import builtins as _b
for _alias in ("int", "float", "bool", "object", "complex", "str"):
    if not hasattr(np, _alias):
        setattr(np, _alias, getattr(_b, _alias))

_LOCAL_NEUROLIB = Path(r"D:\Year3_Mao_Projects\neurolib")
if _LOCAL_NEUROLIB.is_dir() and str(_LOCAL_NEUROLIB) not in sys.path:
    sys.path.insert(0, str(_LOCAL_NEUROLIB))

import pandas as pd
from scipy.signal import detrend, resample_poly
from math import gcd
import mne
mne.set_log_level("WARNING")

from compute_pac_metrics_eeg_native import compute_mi_zscore

# ---- helper ----------------------------------------------------------------
def print_result(label, r):
    print(f"  {label}")
    print(f"    MI_obs       = {r['mi_observed']:.6f}")
    print(f"    null_mean    = {r['mi_null_mean']:.6f}")
    print(f"    null_std     = {r['mi_null_std']:.6f}")
    print(f"    z            = {r['z']:+.3f}")
    print(f"    n_surrogates = {r['n_surrogates']}")
    print(f"    ok           = {r['ok']}")


# ---- Part 1: SC4001 EEG ----------------------------------------------------
FS_SIM = 1000.0

_prep_spec = importlib.util.spec_from_file_location(
    "preprocess_psd", str(_ROOT / "utils" / "02_preprocess_psd.py"))
_prep_mod = importlib.util.module_from_spec(_prep_spec)
_prep_spec.loader.exec_module(_prep_mod)

print("=" * 64)
print("Part 1: SC4001 raw EEG surrogate test")
print("=" * 64)

try:
    manifest = pd.read_csv("data/manifest.csv", encoding="utf-8")
except UnicodeDecodeError:
    manifest = pd.read_csv("data/manifest.csv", encoding="utf-16")

row = manifest[manifest["subject_id"] == "SC4001"].iloc[0]
raw = mne.io.read_raw_edf(row["psg_path"], include=["EEG Fpz-Cz"],
                          preload=True, verbose=False)
fs_native = raw.info["sfreq"]
stages = _prep_mod.load_hypnogram(Path(row["hypnogram_path"]))
data_uv = raw.get_data()[0] * 1e6
EPOCH = int(_prep_mod.EPOCH_LEN_S * fs_native)
ART = 200.0  # µV ptp threshold

accepted_idx, accepted_ep = [], []
for i in range(min(len(stages), len(data_uv) // EPOCH)):
    if stages[i] != "N3":
        continue
    ep = data_uv[i * EPOCH:(i + 1) * EPOCH]
    if np.ptp(ep) > ART:
        continue
    accepted_idx.append(i)
    accepted_ep.append(ep)

# Resample each epoch separately, then concat for "concat mode"
g = gcd(int(FS_SIM), int(fs_native))
def to_1k_detrend(epoch):
    e1k = resample_poly(epoch, int(FS_SIM) // g, int(fs_native) // g)
    return detrend(e1k, type="linear")

print(f"\nAccepted: {len(accepted_ep)} N3 epochs")

# Mode X: full concat
eeg_concat = np.concatenate([to_1k_detrend(e) for e in accepted_ep])
print(f"Mode X (concat 4260s): {len(eeg_concat)/FS_SIM:.0f}s, std={eeg_concat.std():.1f} µV")
t0 = time.time()
res_x = compute_mi_zscore(eeg_concat, FS_SIM, n_surrogates=100, seed=42)
print(f"  [{time.time()-t0:.1f}s]")
print_result("Mode X: SC4001 raw EEG, full 4260s concat", res_x)

# Mode Y: longest contiguous run
contig, cur = [], [accepted_idx[0]]
for k in accepted_idx[1:]:
    if k == cur[-1] + 1:
        cur.append(k)
    else:
        contig.append(cur); cur = [k]
contig.append(cur)
longest = max(contig, key=len)
print(f"\nLongest contig: {len(longest)} epochs = {len(longest)*30}s")
eeg_contig = np.concatenate([
    to_1k_detrend(data_uv[i * EPOCH:(i + 1) * EPOCH]) for i in longest
])
t0 = time.time()
res_y = compute_mi_zscore(eeg_contig, FS_SIM, n_surrogates=100, seed=42)
print(f"  [{time.time()-t0:.1f}s]")
print_result(f"Mode Y: SC4001 longest contig {len(longest)*30}s", res_y)

# Paranoia check: white-noise EEG-shape signal
print("\n--- Paranoia: white noise (should give z ≈ 0) ---")
rng = np.random.default_rng(0)
noise = 28.0 * rng.standard_normal(len(eeg_contig))
res_noise = compute_mi_zscore(noise, FS_SIM, n_surrogates=100, seed=42)
print_result("Paranoia: white-noise null", res_noise)

# ---- Part 2: Seed B simulator ---------------------------------------------
print("\n" + "=" * 64)
print("Part 2: Seed B simulator (r_ctx phase × r_thal amp)")
print("=" * 64)

# Reuse simulator_wrapper machinery — but call v7 directly to get r_ctx + r_thal
# (simulator_wrapper returns the 5-dim summary array, hides intermediates)
sys.path.insert(0, str(_ROOT / "S4_sbi"))

_v7_path = _ROOT / "models" / "s4_personalize_fig7_v7.py"
_spec_v7 = importlib.util.spec_from_file_location("v7", str(_v7_path))
v7 = importlib.util.module_from_spec(_spec_v7)
sys.modules["v7"] = v7
try:
    _spec_v7.loader.exec_module(v7)
except SystemExit:
    pass

print("\nBuilding model with Seed B params ...")
theta_b = dict(
    g_h=0.0550313855044075,
    g_lk=0.0523567827437,
    c_ctx2th=0.0997838197248941,
    b=41.839010370803365,
    mue=3.3406859406304865,
    mui=3.2758268081375705,
    tauA=1257.4091819444602,
    c_th2ctx=0.0329531573906836,
)
m = v7.build_model(**theta_b)
v7.seed_numba(42)
t0 = time.time()
m.run()
print(f"Simulation: {time.time()-t0:.1f}s")

r_exc = m[f"r_mean_{v7.EXC}"]
if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
    r_ctx  = r_exc[0, :] * 1000.0
    r_thal = r_exc[1, :] * 1000.0
else:
    r_ctx  = (r_exc[0] if r_exc.ndim == 2 else r_exc) * 1000.0
    r_thal = np.zeros_like(r_ctx)

# discard 5s burn-in
n_drop = int(5.0 * v7.FS_SIM)
r_ctx  = r_ctx[n_drop:]
r_thal = r_thal[n_drop:]
print(f"r_ctx: {len(r_ctx)} samples = {len(r_ctx)/v7.FS_SIM:.0f}s, "
      f"std={r_ctx.std():.2f} Hz, max={r_ctx.max():.2f} Hz")
print(f"r_thal: std={r_thal.std():.2f} Hz, max={r_thal.max():.2f} Hz")

# Cross-channel: phase from r_ctx, amp from r_thal
t0 = time.time()
res_sim = compute_mi_zscore(r_ctx, v7.FS_SIM, sig_amp=r_thal,
                             n_surrogates=100, seed=42)
print(f"  [{time.time()-t0:.1f}s]")
print_result("Seed B cross-channel: r_ctx phase × r_thal amp", res_sim)

# Also single-channel on r_ctx (for symmetry with EEG side, if we go that route)
t0 = time.time()
res_sim_sc = compute_mi_zscore(r_ctx, v7.FS_SIM,
                                n_surrogates=100, seed=42)
print(f"  [{time.time()-t0:.1f}s]")
print_result("Seed B single-channel: r_ctx phase × r_ctx amp", res_sim_sc)

# ---- Summary --------------------------------------------------------------
print("\n" + "=" * 64)
print("VERDICT TABLE")
print("=" * 64)
fmt = "{:42s}  MI_obs={:.5f}  μ={:.5f}  σ={:.5f}  z={:+.3f}"
print(fmt.format("EEG Mode X (concat 4260s)",
                 res_x["mi_observed"], res_x["mi_null_mean"],
                 res_x["mi_null_std"], res_x["z"]))
print(fmt.format(f"EEG Mode Y (contig {len(longest)*30}s)",
                 res_y["mi_observed"], res_y["mi_null_mean"],
                 res_y["mi_null_std"], res_y["z"]))
print(fmt.format("White-noise paranoia",
                 res_noise["mi_observed"], res_noise["mi_null_mean"],
                 res_noise["mi_null_std"], res_noise["z"]))
print(fmt.format("Seed B cross-channel (r_ctx × r_thal)",
                 res_sim["mi_observed"], res_sim["mi_null_mean"],
                 res_sim["mi_null_std"], res_sim["z"]))
print(fmt.format("Seed B single-channel (r_ctx × r_ctx)",
                 res_sim_sc["mi_observed"], res_sim_sc["mi_null_mean"],
                 res_sim_sc["mi_null_std"], res_sim_sc["z"]))
