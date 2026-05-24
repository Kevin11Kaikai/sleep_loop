"""
simulator_wrapper.py
====================
SBI simulator: maps 4D theta -> len(SUMMARY_KEYS) summary statistics.

Free parameters (theta order matches prior in run_sbi.py):
    theta = [g_h, g_LK, c_ctx2th, b]

Fixed parameters (Seed B from S4_v7_repair/pareto_seeds_fresh_DE.json):
    mue      = 3.3406859406304865
    mui      = 3.2758268081375705
    tauA     = 1257.4091819444602
    c_th2ctx = 0.0329531573906836

Seed B reference metrics (for smoke-test comparison):
    shape_r ~ 0.679, MI ~ 0.07 (full V7 run also reports T12_n_verified in constraints)

Returns: np.ndarray of shape (len(SUMMARY_KEYS),) in SUMMARY_KEYS order.
On failure: np.full(len(SUMMARY_KEYS), np.nan).

SUMMARY_KEYS (must match compute_xobs_from_eeg_v3.py):
    [shape_r, T4_q, T4_freq, T8_n_sp_events, T11_lag_ms]

T6_ibi_cv and MI dropped — r_proxy envelope is structurally incompatible with
UP/DOWN detection and cycle-by-cycle PAC phase interpolation (diagnostic 2026-05-10).

Windows-specific constraints (non-negotiable):
  - num_workers=1 in all simulate_for_sbi calls (neurolib/numba hangs in fork)
  - No multiprocessing; sequential simulation is the only safe mode.
  - NumPy alias shim runs BEFORE any neurolib import.
  - Local neurolib at D:/Year3_Mao_Projects/neurolib takes precedence.
"""

import sys
import os
import warnings
import importlib.util
from pathlib import Path

warnings.filterwarnings("ignore")

# ── 1. NumPy alias shim — MUST come before any neurolib import ────────────────
import numpy as np
import builtins as _builtins_mod
for _alias in ("int", "float", "bool", "object", "complex", "str"):
    if not hasattr(np, _alias):
        setattr(np, _alias, getattr(_builtins_mod, _alias))

# ── 2. Local neurolib takes precedence over system install ────────────────────
# System neurolib has a compile_to_numba bug (co_consts[-3] IndexError) on
# Python 3.11+. The patched copy at this path fixes it.
_LOCAL_NEUROLIB = Path(r"D:\Year3_Mao_Projects\neurolib")
if _LOCAL_NEUROLIB.is_dir() and str(_LOCAL_NEUROLIB) not in sys.path:
    sys.path.insert(0, str(_LOCAL_NEUROLIB))

# ── 3. Project root must be cwd (V7's importlib uses relative paths) ──────────
_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(_ROOT))
sys.path.insert(0, str(_ROOT))

# ── 4. Import V7 via importlib (guards against triggering main()) ─────────────
_v7_path = _ROOT / "models" / "s4_personalize_fig7_v7.py"
if not _v7_path.exists():
    raise FileNotFoundError(f"V7 script not found: {_v7_path}")
_spec_v7 = importlib.util.spec_from_file_location("v7", str(_v7_path))
v7 = importlib.util.module_from_spec(_spec_v7)
sys.modules["v7"] = v7
try:
    _spec_v7.loader.exec_module(v7)
except SystemExit:
    pass  # v7's main() is guarded by __name__=="__main__", but just in case

# ── 5. Add S4_v7_repair to sys.path (needed by v7.compute_pac_metrics) ────────
_repair_dir = _ROOT / "S4_v7_repair"
if str(_repair_dir) not in sys.path:
    sys.path.insert(0, str(_repair_dir))

# ── 6. Load target PSD once at module init ────────────────────────────────────
print("[simulator_wrapper] Loading target PSD ...")
_target_psd, _target_freqs = v7.load_target_psd()
_target_periodic, _fooof_freqs = v7.compute_target_periodic(
    _target_psd, _target_freqs
)
print(f"[simulator_wrapper] Target PSD loaded: {len(_target_psd)} bins, "
      f"FOOOF={'OK' if _target_periodic is not None else 'unavailable'}")

# ── Constants ─────────────────────────────────────────────────────────────────
SUMMARY_KEYS = [
    "shape_r", "T4_q", "T4_freq",
    "T8_n_sp_events", "T11_lag_ms",
]

# Seed B fixed parameters (non-free dims; kept constant across SBI runs)
_FIXED = {
    "mue":      3.3406859406304865,
    "mui":      3.2758268081375705,
    "tauA":     1257.4091819444602,
    "c_th2ctx": 0.0329531573906836,
}

# Convenience aliases into v7 namespace
_FS       = v7.FS_SIM     # 1000.0 Hz
_F_LO     = v7.F_LO       # 0.5 Hz
_F_HI     = v7.F_HI       # 20.0 Hz
_EXC      = v7.EXC        # "EXC" firing-rate key

# EVO_FOOOF_PARAMS is a module-level dict in v7 (passed to FOOOF constructor)
_EVO_FOOOF = v7.EVO_FOOOF_PARAMS


def _extract_summaries(r_ctx, r_thal):
    """
    Run compute_constraints_v7 + FOOOF shape_r.
    Returns dict with keys matching SUMMARY_KEYS.
    """
    # Compute PSD (full frequency axis; v7 applies a mask internally too)
    f_full, p_full = v7.compute_epoch_psd(r_ctx, _FS)

    # 12 constraint checks (T1–T12, including PAC via fixed compute_pac_metrics)
    _, con = v7.compute_constraints_v7(
        r_ctx, r_thal, f_c=f_full, p_c=p_full, fs=_FS
    )

    # shape_r: FOOOF Pearson-r between simulated and target periodic components.
    # Computed for ALL simulations (feasible or not) so the 8-dim vector is
    # always complete. For infeasible sims the correlation is typically low.
    shape_r = 0.0
    if v7.HAS_FOOOF and _target_periodic is not None:
        try:
            from scipy.interpolate import interp1d
            from scipy.stats import pearsonr
            from fooof import FOOOF

            mask = (f_full >= _F_LO) & (f_full <= _F_HI)
            f_ctx = f_full[mask]
            p_ctx = p_full[mask]
            p_interp = interp1d(
                f_ctx, p_ctx, bounds_error=False, fill_value=1e-30
            )(_fooof_freqs)
            fm_sim = FOOOF(**_EVO_FOOOF)
            fm_sim.fit(_fooof_freqs, p_interp, [_F_LO, _F_HI])
            sim_log = np.log10(p_interp[: len(fm_sim._ap_fit)] + 1e-30)
            sim_periodic = sim_log - fm_sim._ap_fit
            n_r = min(len(sim_periodic), len(_target_periodic))
            r_val, _ = pearsonr(sim_periodic[:n_r], _target_periodic[:n_r])
            shape_r = float(max(r_val, 0.0))
        except Exception:
            shape_r = 0.0

    # T8 is normalized to events per 60 s. The model's active signal is exactly
    # 60 s (after 5 s burn-in discard), so the factor is 1.0 — stated explicitly
    # to match the convention used in compute_xobs_from_eeg.py.
    _sim_dur_s = (len(r_ctx) / _FS)   # seconds of active (post-burn-in) signal
    t8_raw = float(con.get("T8_n_sp_events", 0))
    t8_norm = t8_raw * (60.0 / _sim_dur_s)

    return {
        "shape_r":         shape_r,
        "T4_q":            float(con.get("T4_q", 0.0)),
        "T4_freq":         float(con.get("T4_freq", 0.0)),
        "T8_n_sp_events":  t8_norm,
        "T11_lag_ms":      float(con.get("T11_lag_ms", 0.0)),   # = up_down_ratio
    }


def simulator(theta):
    """
    Simulate the thalamocortical model and return 8 summary statistics.

    Parameters
    ----------
    theta : array-like, shape (4,) — [g_h, g_LK, c_ctx2th, b]
        Can be torch.Tensor, np.ndarray, or list.

    Returns
    -------
    np.ndarray of shape (len(SUMMARY_KEYS),) — summary stats in SUMMARY_KEYS order.
    Returns np.full(len(SUMMARY_KEYS), np.nan) on any failure.
    """
    try:
        # Convert to plain numpy (handles torch.Tensor input from sbi)
        if hasattr(theta, "detach"):
            theta = theta.detach().cpu().numpy()
        theta = np.asarray(theta, dtype=float).ravel()

        g_h, g_lk, c_ctx2th, b = theta

        # Build model with free params + Seed B fixed params
        m = v7.build_model(
            mue      = _FIXED["mue"],
            mui      = _FIXED["mui"],
            b        = b,
            tauA     = _FIXED["tauA"],
            g_lk     = g_lk,
            g_h      = g_h,
            c_th2ctx = _FIXED["c_th2ctx"],
            c_ctx2th = c_ctx2th,
        )

        # Seed numba RNG deterministically before every run
        v7.seed_numba(42)
        m.run()

        # Extract cortex (index 0) and thalamus (index 1) firing rates
        r_exc = m[f"r_mean_{_EXC}"]
        if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
            r_ctx  = r_exc[0, :] * 1000.0   # kHz → Hz
            r_thal = r_exc[1, :] * 1000.0
        else:
            r_ctx  = (r_exc[0] if r_exc.ndim == 2 else r_exc) * 1000.0
            r_thal = np.zeros_like(r_ctx)

        # Discard 5 s burn-in (mirrors compute_fitness_v7)
        n_drop = int(5.0 * _FS)
        r_ctx  = r_ctx[n_drop:]
        r_thal = r_thal[n_drop:]

        if r_ctx.max() < 0.1:
            # All-zeros / flat — simulation failed silently
            return np.full(len(SUMMARY_KEYS), np.nan)

        stats = _extract_summaries(r_ctx, r_thal)
        return np.array([stats[k] for k in SUMMARY_KEYS], dtype=np.float64)

    except Exception as exc:
        print(f"  [Sim failed] {exc}")
        return np.full(len(SUMMARY_KEYS), np.nan)


# ─── Smoke-test when run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Smoke test: single simulation with Seed B params")
    print("Expected: shape_r~0.68, MI~0.07")
    print("=" * 60)

    # Seed B free params
    theta_b = np.array([
        0.0550313855044075,    # g_h
        0.0523567827437,       # g_LK
        0.0997838197248941,    # c_ctx2th
        41.839010370803365,    # b
    ])

    import time
    t0 = time.time()
    x = simulator(theta_b)
    elapsed = time.time() - t0

    print(f"\nResult ({elapsed:.1f} s):")
    for k, v_val in zip(SUMMARY_KEYS, x):
        print(f"  {k:20s} = {v_val:.5f}")

    nan_count = int(np.isnan(x).sum())
    print(f"\nNaN count: {nan_count}/5")
    if nan_count == 0:
        print("SMOKE TEST PASSED")
    else:
        print("SMOKE TEST FAILED — check error messages above")
