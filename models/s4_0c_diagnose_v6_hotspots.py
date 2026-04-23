"""
s4_0c_diagnose_v6_hotspots.py
==============================
Diagnostic: why does v6 DE fail to find sweep-identified hotspots?

Rationale:
  v6 DE produced only 1 feasible solution (0.02% rate), with parameters
  FAR from the sweep-identified spindle hotspot. The single feasible
  solution had spindle_power=0 (FOOOF could not detect a σ peak) and
  thalamic TCR timeseries identical to v5's "fake spindle" failure mode.

  Two competing explanations:
    (H1) SEARCH PROBLEM: v6 bounds are correct, sweep hotspots would
         pass all 12 v6 constraints, but DE is poorly initialized and
         converges to a local trap. Fix: better DE seeding or tighter
         c_th2ctx bounds.
    (H2) CONSTRAINT PROBLEM: the v6 constraints (T6, T12 in particular)
         have wrong thresholds — even the sweep hotspots fail them.
         Fix: recalibrate T6 or T12 thresholds.

  This script directly tests which hypothesis is correct by running the
  FULL v6 fitness (constraints + FOOOF rewards) on 4 known-good spindle
  hotspot parameter sets × 2 c_th2ctx regimes:

    Regime A (isolated, c_th2ctx=0):    reproduces sweep conditions.
                                          Must pass if sweep was valid.
    Regime B (coupled, c_th2ctx=0.10):   full bidirectional coupling
                                          (no feedforward disconnection).
                                          Tests whether v6's cortex SO
                                          survives with real thalamic
                                          feedback.

Test points:
  (1) Primary golden point (verified in s4_0b_verify_golden_point.py)
  (2) Control golden point (verified — weak PAC regime)
  (3) Sweep top #1 by sp_peak_log
  (4) Sweep top #2 by PAC MI

Output:
  outputs/v6_hotspot_diagnosis.txt  — per-point per-constraint pass table
  outputs/v6_hotspot_diagnosis.csv  — machine-readable version

Usage:
  python models/s4_0c_diagnose_v6_hotspots.py
  # ~8 simulations × ~15s = 2-3 minutes
"""

import os
import sys
import time
import fnmatch
import importlib.util
import warnings
warnings.filterwarnings("ignore")

import numpy as np
for _attr in ("object", "bool", "int", "float", "complex", "str"):
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(__builtins__, _attr, object))

import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.signal import hilbert, butter, sosfiltfilt, correlate, welch
from scipy.ndimage import gaussian_filter1d

from neurolib.models.multimodel import MultiModel, ALNNode, ThalamicNode
from neurolib.models.multimodel.builder.base.network import Network
from neurolib.models.multimodel.builder.base.constants import EXC, INH

try:
    from fooof import FOOOF
    HAS_FOOOF = True
except ImportError:
    HAS_FOOOF = False
    print("[warn] fooof not installed; so_power/spindle_power via FOOOF unavailable")

# Import preprocessing module (for EEG loading — not strictly needed here but keeps parity with v6)
_spec = importlib.util.spec_from_file_location(
    "02_preprocess_psd", "utils/02_preprocess_psd.py"
)
prep_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep_mod)
compute_epoch_psd = prep_mod.compute_epoch_psd


# =====================================================================
# CONFIG — mirror v6 EXACTLY
# =====================================================================

FS_SIM     = 1000.0
SIM_DUR_MS = 30_000       # match v6's fitness simulation length
BURN_IN_S  = 5.0
F_LO, F_HI = 0.5, 20.0

# V4 best cortex params (sweep was done with these — must reproduce)
CORTEX_V4_BEST = dict(
    mue   = 3.895,
    mui   = 3.027,
    b     = 35.5,
    tauA  = 1426.0,
)

# Test points from sweep analysis
TEST_POINTS = [
    dict(
        label      = "P1_primary_golden",
        desc       = "Primary golden point (verified: real spindle + PAC)",
        g_LK       = 0.043,
        g_h        = 0.066,
        c_ctx2th   = 0.155,
    ),
    dict(
        label      = "P2_control_golden",
        desc       = "Control golden point (verified: spindle without strong PAC)",
        g_LK       = 0.049,
        g_h        = 0.066,
        c_ctx2th   = 0.020,
    ),
    dict(
        label      = "P3_sweep_top_splog",
        desc       = "Sweep top by sp_peak_log (strongest σ peak)",
        g_LK       = 0.037,
        g_h        = 0.066,
        c_ctx2th   = 0.200,
    ),
    dict(
        label      = "P4_sweep_top_mi",
        desc       = "Sweep top by PAC MI (strongest coupling)",
        g_LK       = 0.043,
        g_h        = 0.066,
        c_ctx2th   = 0.200,
    ),
]

# Two c_th2ctx regimes to test
REGIMES = [
    dict(label="A_isolated", c_th2ctx=0.0,
         desc="Isolated thalamus (reproduces sweep conditions)"),
    dict(label="B_coupled",  c_th2ctx=0.10,
         desc="Bidirectional coupling (v4_best c_th2ctx)"),
]

# Constraint thresholds — COPY FROM v6 (keep in sync!)
DOWN_THRESH_HZ   = 1.0
UP_THRESH_HZ     = 15.0
UP_DURATION_MS   = 100.0
SO_FREQ_LO       = 0.2
SO_FREQ_HI       = 1.5
SO_Q_MIN         = 2.0
SPINDLE_LO       = 10.0
SPINDLE_HI       = 14.0
SPINDLE_FWHM_MIN = 2.0
IBI_CV_MAX       = 0.4
SPINDLE_CV_MIN   = 0.7
SPINDLE_EVT_MIN  = 5
SPINDLE_DUR_LO_S = 0.3
SPINDLE_DUR_HI_S = 2.0
SPINDLE_EVT_PCTILE       = 75.0
SPINDLE_ENV_SMOOTH_MS    = 200.0

T12_PEAK_INSIDE_RATIO   = 1.5
T12_N_VERIFIED_MIN      = 5

PAC_N_BINS       = 18
PAC_MI_MIN       = 0.005
PAC_PHASE_TOL    = 5 * np.pi / 18   # ±50°
PAC_MAX_LAG_S    = 2.0
PAC_MIN_LAG_MS   = 20.0

EVO_FOOOF_PARAMS = dict(
    peak_width_limits=[1.0, 8.0],
    max_n_peaks=4,
    min_peak_height=0.05,
    aperiodic_mode="fixed",
)

OUT_TXT = "outputs/v6_hotspot_diagnosis.txt"
OUT_CSV = "outputs/v6_hotspot_diagnosis.csv"


# =====================================================================
# Model (identical to v6's ThalamoCorticalNetwork)
# =====================================================================

def set_params_glob(model, pattern, value):
    for k in (k for k in model.params if fnmatch.fnmatch(k, pattern)):
        model.params[k] = value


class ThalamoCorticalNetwork(Network):
    name            = "Thalamocortical Motif (diagnostic)"
    label           = "TCNet_diag"
    sync_variables  = ["network_exc_exc", "network_exc_exc_sq", "network_inh_exc"]
    default_output  = f"r_mean_{EXC}"
    output_vars     = [f"r_mean_{EXC}", f"r_mean_{INH}"]
    _EXC_WITHIN_IDX = [6, 9]

    def __init__(self, c_th2ctx=0.0, c_ctx2th=0.04):
        aln = ALNNode(exc_seed=42, inh_seed=42)
        th  = ThalamicNode()
        aln.index = 0; aln.idx_state_var = 0
        th.index  = 1; th.idx_state_var  = aln.num_state_variables
        for i, node in enumerate([aln, th]):
            for mass in node:
                mass.noise_input_idx = [2 * i + mass.index]
        connectivity = np.array([[0.0, c_th2ctx], [c_ctx2th, 0.0]])
        super().__init__(nodes=[aln, th], connectivity_matrix=connectivity,
                         delay_matrix=np.zeros_like(connectivity))

    def _sync(self):
        couplings = sum([node._sync() for node in self], [])
        wi = self._EXC_WITHIN_IDX
        couplings += self._additive_coupling(wi, "network_exc_exc")
        couplings += self._additive_coupling(
            wi, "network_exc_exc_sq", connectivity=self.connectivity ** 2
        )
        couplings += self._additive_coupling(wi, "network_inh_exc")
        return couplings


def build_model(g_lk, g_h, c_th2ctx, c_ctx2th):
    net = ThalamoCorticalNetwork(c_th2ctx=c_th2ctx, c_ctx2th=c_ctx2th)
    m = MultiModel(net)
    m.params["backend"]     = "numba"
    m.params["dt"]          = 0.1
    m.params["sampling_dt"] = 1.0
    m.params["duration"]    = SIM_DUR_MS
    set_params_glob(m, "*ALNMassEXC*.input_0.mu", CORTEX_V4_BEST["mue"])
    set_params_glob(m, "*ALNMassINH*.input_0.mu", CORTEX_V4_BEST["mui"])
    set_params_glob(m, "*ALNMassEXC*.b",          CORTEX_V4_BEST["b"])
    set_params_glob(m, "*ALNMassEXC*.tauA",       CORTEX_V4_BEST["tauA"])
    set_params_glob(m, "*ALNMassEXC*.a",          0.0)
    set_params_glob(m, "*ALNMass*.input_0.sigma", 0.05)
    set_params_glob(m, "*TCR*.input_0.sigma",     0.005)
    set_params_glob(m, "*.input_0.tau",           5.0)
    set_params_glob(m, "*TRN*.g_LK",              0.1)
    set_params_glob(m, "*TCR*.g_LK",              g_lk)
    set_params_glob(m, "*TCR*.g_h",               g_h)
    return m


# =====================================================================
# Constraint battery (copied verbatim from v6)
# =====================================================================

def compute_pac_metrics(r_ctx, r_thal, fs=FS_SIM):
    out = {"mi": 0.0, "preferred_phase": np.pi, "lag_samples": 0,
           "lag_ms": 0.0, "ok": False}
    try:
        if len(r_ctx) < int(2 * fs) or r_ctx.std() < 1e-6:
            return out
        if r_thal.std() < 1e-6:
            return out
        sos_so = butter(4, [SO_FREQ_LO, SO_FREQ_HI], btype="band",
                        fs=fs, output="sos")
        so_filt = sosfiltfilt(sos_so, r_ctx)
        so_phase = np.angle(hilbert(so_filt))
        sos_sp = butter(4, [SPINDLE_LO, SPINDLE_HI], btype="band",
                        fs=fs, output="sos")
        sp_filt = sosfiltfilt(sos_sp, r_thal)
        sp_amp = np.abs(hilbert(sp_filt))
        edge = int(0.5 * fs)
        so_filt = so_filt[edge:-edge]
        so_phase = so_phase[edge:-edge]
        sp_amp = sp_amp[edge:-edge]
        bin_edges = np.linspace(-np.pi, np.pi, PAC_N_BINS + 1)
        mean_amp = np.zeros(PAC_N_BINS)
        for i in range(PAC_N_BINS):
            mask = (so_phase >= bin_edges[i]) & (so_phase < bin_edges[i + 1])
            if mask.any():
                mean_amp[i] = sp_amp[mask].mean()
        total = mean_amp.sum()
        if total <= 0 or not np.isfinite(total):
            return out
        p = mean_amp / total
        p_safe = np.where(p > 0, p, 1.0)
        H = -np.sum(p * np.log(p_safe))
        mi = float(np.clip((np.log(PAC_N_BINS) - H) / np.log(PAC_N_BINS),
                           0.0, 1.0))
        mvl = (sp_amp * np.exp(1j * so_phase)).mean()
        preferred_phase = float(np.angle(mvl))
        max_lag = int(PAC_MAX_LAG_S * fs)
        a = sp_amp - sp_amp.mean()
        b = so_filt - so_filt.mean()
        xc = correlate(a, b, mode="full")
        lags = np.arange(-(len(b) - 1), len(a))
        keep = (lags >= -max_lag) & (lags <= max_lag)
        xc_w = xc[keep]; lags_w = lags[keep]
        if xc_w.size == 0:
            return out
        peak_lag_samples = int(lags_w[np.argmax(xc_w)])
        out.update({"mi": mi, "preferred_phase": preferred_phase,
                    "lag_samples": peak_lag_samples,
                    "lag_ms": peak_lag_samples * 1000.0 / fs, "ok": True})
    except Exception:
        pass
    return out


def compute_constraints_v6(r_ctx, r_thal, f_c=None, p_c=None, fs=FS_SIM):
    """Exact copy of v6's compute_constraints_v6. All 12 constraints."""
    details = {}

    # T1
    min_rE = float(r_ctx.min())
    t1 = min_rE < DOWN_THRESH_HZ
    details["T1"] = t1; details["T1_min_rE"] = min_rE
    # T2
    max_rE = float(r_ctx.max())
    t2 = max_rE > UP_THRESH_HZ
    details["T2"] = t2; details["T2_max_rE"] = max_rE
    # T3
    min_run_samples = int(UP_DURATION_MS * fs / 1000.0)
    above = (r_ctx > UP_THRESH_HZ).astype(np.int8)
    diff = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    max_run = int((ends - starts).max()) if len(starts) > 0 else 0
    t3 = max_run >= min_run_samples
    details["T3"] = t3; details["T3_longest_ms"] = max_run * 1000.0 / fs
    # T4
    if f_c is None or p_c is None:
        f_c, p_c = compute_epoch_psd(r_ctx, fs)
    so_mask = (f_c >= SO_FREQ_LO) & (f_c <= SO_FREQ_HI)
    so_width = SO_FREQ_HI - SO_FREQ_LO
    nlo = (f_c >= max(0.1, SO_FREQ_LO - so_width)) & (f_c < SO_FREQ_LO)
    nhi = (f_c > SO_FREQ_HI) & (f_c <= SO_FREQ_HI + so_width)
    so_peak_freq = 0.0; so_q = 0.0; t4 = False
    if so_mask.any():
        so_peak_freq = float(f_c[so_mask][np.argmax(p_c[so_mask])])
        so_peak_val = p_c[so_mask].max()
        neighbors = np.concatenate([
            p_c[nlo] if nlo.any() else np.array([]),
            p_c[nhi] if nhi.any() else np.array([])])
        if len(neighbors) > 0 and neighbors.mean() > 0:
            so_q = float(so_peak_val / neighbors.mean())
        t4 = (SO_FREQ_LO <= so_peak_freq <= SO_FREQ_HI) and (so_q > SO_Q_MIN)
    details["T4"] = t4; details["T4_freq"] = so_peak_freq
    details["T4_q"] = round(so_q, 3)
    # T5
    f_th, p_th = compute_epoch_psd(r_thal, fs)
    sp_mask = (f_th >= SPINDLE_LO) & (f_th <= SPINDLE_HI)
    fwhm = 0.0
    if sp_mask.any() and p_th[sp_mask].max() > 0:
        p_sp = p_th[sp_mask]; f_sp = f_th[sp_mask]
        hp = p_sp.max() / 2.0
        ah = f_sp[p_sp >= hp]
        if len(ah) >= 2:
            fwhm = float(ah[-1] - ah[0])
    t5 = fwhm > SPINDLE_FWHM_MIN
    details["T5"] = t5; details["T5_fwhm"] = fwhm
    # T6
    ibi_cv = 999.0; n_bursts = len(starts)
    if n_bursts >= 3:
        intervals = np.diff(starts) / fs
        ibi_cv = float(intervals.std() / (intervals.mean() + 1e-12))
    t6 = (n_bursts >= 3) and (ibi_cv < IBI_CV_MAX)
    details["T6"] = t6; details["T6_ibi_cv"] = round(ibi_cv, 3)
    details["T6_n_bursts"] = n_bursts
    # T7
    sp_cv = 0.0; envelope = None
    try:
        sos = butter(4, [SPINDLE_LO, SPINDLE_HI], btype='band', fs=fs, output='sos')
        filtered = sosfiltfilt(sos, r_thal)
        envelope = np.abs(hilbert(filtered))
        sp_cv = float(envelope.std() / (envelope.mean() + 1e-12))
    except Exception:
        sp_cv = 0.0; envelope = None
    t7 = sp_cv > SPINDLE_CV_MIN
    details["T7"] = t7; details["T7_sp_cv"] = round(sp_cv, 3)
    # T8 (v6 fix: percentile of smoothed envelope)
    n_sp_events = 0; mean_sp_dur = 0.0
    sp_starts_valid = np.array([], dtype=int)
    sp_ends_valid = np.array([], dtype=int)
    try:
        if envelope is not None and len(envelope) > 0:
            sigma_samples = SPINDLE_ENV_SMOOTH_MS * fs / 1000.0
            env_smooth = gaussian_filter1d(envelope, sigma=sigma_samples)
            thresh = np.percentile(env_smooth, SPINDLE_EVT_PCTILE)
            above = (env_smooth > thresh).astype(np.int8)
            diff_sp = np.diff(np.concatenate(([0], above, [0])))
            sp_starts = np.where(diff_sp == 1)[0]
            sp_ends = np.where(diff_sp == -1)[0]
            durations = (sp_ends - sp_starts) / fs
            valid = (durations >= SPINDLE_DUR_LO_S) & (durations <= SPINDLE_DUR_HI_S)
            n_sp_events = int(valid.sum())
            if n_sp_events > 0:
                mean_sp_dur = float(durations[valid].mean())
                sp_starts_valid = sp_starts[valid]
                sp_ends_valid = sp_ends[valid]
    except Exception:
        pass
    t8 = n_sp_events >= SPINDLE_EVT_MIN
    details["T8"] = t8; details["T8_n_sp_events"] = n_sp_events
    details["T8_mean_sp_dur"] = round(mean_sp_dur, 3)
    # T9-T11: PAC
    pac = compute_pac_metrics(r_ctx, r_thal, fs=fs)
    t9 = pac["ok"] and (pac["mi"] > PAC_MI_MIN)
    details["T9"] = t9; details["T9_mi"] = round(pac["mi"], 5)
    phi = pac["preferred_phase"]
    dist = min(abs(phi), np.pi - abs(phi))
    t10 = pac["ok"] and (dist < PAC_PHASE_TOL)
    details["T10"] = t10; details["T10_phase"] = round(phi, 3)
    details["T10_dist_tgt"] = round(dist, 3)
    t11 = pac["ok"] and (pac["lag_ms"] >= PAC_MIN_LAG_MS)
    details["T11"] = t11; details["T11_lag_ms"] = round(pac["lag_ms"], 1)
    # T12: peak-inside-event
    n_verified = 0
    for s, e in zip(sp_starts_valid, sp_ends_valid):
        if e - s < int(0.2 * fs):
            continue
        event = r_thal[s:e]
        try:
            f_ev, p_ev = welch(event, fs=fs, nperseg=min(len(event), 512))
            sp_m = (f_ev >= SPINDLE_LO) & (f_ev <= SPINDLE_HI)
            ns_m = (f_ev >= 4) & (f_ev < SPINDLE_LO)
            if not sp_m.any() or not ns_m.any():
                continue
            ns_peak = p_ev[ns_m].max() if p_ev[ns_m].size > 0 else 1e-12
            ns_peak = ns_peak if ns_peak > 0 else 1e-12
            ratio = p_ev[sp_m].max() / ns_peak
            if ratio > T12_PEAK_INSIDE_RATIO:
                n_verified += 1
        except Exception:
            continue
    t12 = n_verified >= T12_N_VERIFIED_MIN
    details["T12"] = t12; details["T12_n_verified"] = n_verified
    # Summary
    n_passed = sum([t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12])
    details["n_passed"] = n_passed
    details["feasible"] = (n_passed == 12)
    return n_passed, details


# =====================================================================
# FOOOF rewards (computed regardless of feasibility, for diagnostic)
# =====================================================================

def compute_fooof_rewards(r_ctx, fs=FS_SIM):
    """Compute so_power and spindle_power via FOOOF, bypassing feasibility gate.

    Returns dict with so_power, spindle_power, fooof_ok.
    """
    out = {"so_power": 0.0, "spindle_power": 0.0, "fooof_ok": False}
    if not HAS_FOOOF:
        return out
    try:
        f, p = compute_epoch_psd(r_ctx, fs)
        mask = (f >= F_LO) & (f <= F_HI)
        f = f[mask]; p = p[mask]
        fm = FOOOF(**EVO_FOOOF_PARAMS)
        fm.fit(f, p, [F_LO, F_HI])
        for pk in fm.peak_params_:
            freq, power, _ = pk
            if SO_FREQ_LO <= freq <= SO_FREQ_HI:
                out["so_power"] = max(out["so_power"], float(power))
            if SPINDLE_LO <= freq <= SPINDLE_HI:
                out["spindle_power"] = max(out["spindle_power"], float(power))
        out["fooof_ok"] = True
    except Exception as e:
        out["fooof_err"] = str(e)
    return out


# =====================================================================
# Main
# =====================================================================

def run_point(g_lk, g_h, c_th2ctx, c_ctx2th):
    """Run one simulation and compute all diagnostics."""
    m = build_model(g_lk, g_h, c_th2ctx, c_ctx2th)
    t0 = time.time()
    try:
        m.run()
    except Exception as e:
        try:
            m.params["backend"] = "jitcdde"
            m.run()
        except Exception:
            return None, time.time() - t0
    r_exc = m[f"r_mean_{EXC}"]
    if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
        r_ctx  = r_exc[0, :] * 1000.0
        r_thal = r_exc[1, :] * 1000.0
    else:
        return None, time.time() - t0
    n_drop = int(BURN_IN_S * FS_SIM)
    r_ctx = r_ctx[n_drop:]; r_thal = r_thal[n_drop:]

    n_passed, con = compute_constraints_v6(r_ctx, r_thal, fs=FS_SIM)
    foof = compute_fooof_rewards(r_ctx, fs=FS_SIM)
    return dict(con=con, fooof=foof, n_passed=n_passed,
                feasible=con["feasible"]), time.time() - t0


def fmt_pass(b):
    return "✓" if b else "✗"


def main():
    print("=" * 80)
    print("V6 Hotspot Diagnostic")
    print("=" * 80)
    print(f"  {len(TEST_POINTS)} test points × {len(REGIMES)} regimes = "
          f"{len(TEST_POINTS) * len(REGIMES)} simulations")
    print(f"  Sim duration: {SIM_DUR_MS/1000:.0f}s each "
          f"(~15s wall-clock → total ~{len(TEST_POINTS)*len(REGIMES)*15/60:.1f} min)")
    print(f"  Cortex fixed at V4_best: {CORTEX_V4_BEST}")
    print("=" * 80)

    os.makedirs("outputs", exist_ok=True)

    results = []
    for pt in TEST_POINTS:
        for rg in REGIMES:
            tag = f"{pt['label']}_{rg['label']}"
            print(f"\n[{tag}]")
            print(f"  Params: g_LK={pt['g_LK']}  g_h={pt['g_h']}  "
                  f"c_th2ctx={rg['c_th2ctx']}  c_ctx2th={pt['c_ctx2th']}")
            print(f"  Running...", end=" ", flush=True)
            res, dt = run_point(pt['g_LK'], pt['g_h'],
                                rg['c_th2ctx'], pt['c_ctx2th'])
            print(f"done in {dt:.1f}s")
            if res is None:
                print(f"  [FAILED to simulate]")
                results.append(dict(tag=tag, point=pt, regime=rg,
                                    failed=True))
                continue

            con = res['con']; foof = res['fooof']
            print(f"  n_passed:  {res['n_passed']}/12  "
                  f"feasible={res['feasible']}")
            line = "  "
            for c in ['T1','T2','T3','T4','T5','T6','T7','T8',
                      'T9','T10','T11','T12']:
                line += f"{c}={fmt_pass(con[c])} "
            print(line)
            print(f"  Key metrics: T4_q={con['T4_q']}  "
                  f"T5_fwhm={con['T5_fwhm']:.2f}  "
                  f"T6_ibi_cv={con['T6_ibi_cv']}  "
                  f"T6_n_bursts={con['T6_n_bursts']}")
            print(f"               T7_cv={con['T7_sp_cv']}  "
                  f"T8_n={con['T8_n_sp_events']}  "
                  f"T9_mi={con['T9_mi']}  "
                  f"T10_phase={con['T10_phase']}  "
                  f"T11_lag={con['T11_lag_ms']}ms  "
                  f"T12_nver={con['T12_n_verified']}")
            print(f"  FOOOF:     so_power={foof['so_power']:.3f}  "
                  f"spindle_power={foof['spindle_power']:.3f}  "
                  f"(ok={foof['fooof_ok']})")

            results.append(dict(
                tag=tag, point=pt, regime=rg, failed=False,
                n_passed=res['n_passed'], feasible=res['feasible'],
                constraints=con, fooof=foof,
            ))

    # ======== Summary matrix ========
    print("\n" + "=" * 80)
    print("SUMMARY — pass/fail matrix")
    print("=" * 80)
    header = f"{'TAG':<32s} {'n/12':>5s}  "
    for c in ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12']:
        header += f"{c:>4s} "
    header += f"{'so_P':>6s} {'sp_P':>6s}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r.get('failed'):
            print(f"{r['tag']:<32s} [sim failed]")
            continue
        con = r['constraints']
        row = f"{r['tag']:<32s} {r['n_passed']:>2d}/12  "
        for c in ['T1','T2','T3','T4','T5','T6','T7','T8',
                  'T9','T10','T11','T12']:
            row += f"{fmt_pass(con[c]):>4s} "
        row += f"{r['fooof']['so_power']:>6.3f} {r['fooof']['spindle_power']:>6.3f}"
        print(row)

    # ======== Save TXT / CSV ========
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(header + "\n" + "-" * len(header) + "\n")
        for r in results:
            if r.get('failed'):
                fh.write(f"{r['tag']:<32s} [sim failed]\n")
                continue
            con = r['constraints']
            row = f"{r['tag']:<32s} {r['n_passed']:>2d}/12  "
            for c in ['T1','T2','T3','T4','T5','T6','T7','T8',
                      'T9','T10','T11','T12']:
                row += f"{fmt_pass(con[c]):>4s} "
            row += (f"{r['fooof']['so_power']:>6.3f} "
                    f"{r['fooof']['spindle_power']:>6.3f}\n")
            fh.write(row)
        # Also write diagnostic interpretation
        fh.write("\n\n" + "=" * 80 + "\n")
        fh.write("INTERPRETATION GUIDE\n")
        fh.write("=" * 80 + "\n")
        fh.write(
            "Look at the 4 PRIMARY points × 2 regimes = 8 rows:\n"
            "\n"
            "IF all 4 Regime_A (isolated) points are FEASIBLE (12/12):\n"
            "  → Sweep conditions reproduce correctly. v6 constraints are valid.\n"
            "  → Problem is DE SEARCH: bounds or initialization issue.\n"
            "  → Fix in v7: seed DE at hotspot, tighten c_th2ctx bounds.\n"
            "\n"
            "IF Regime_A points FAIL T12 (even with spindle_power > 0.1):\n"
            "  → T12 threshold (ratio=1.5) is mis-calibrated.\n"
            "  → Real spindles in this model produce lower σ/non-σ ratios.\n"
            "  → Fix in v7: relax T12 ratio or replace with different gate.\n"
            "\n"
            "IF Regime_B (c_th2ctx=0.10) points FAIL T6 while Regime_A pass:\n"
            "  → Bidirectional coupling destabilizes cortical SO regularity.\n"
            "  → v6 c_th2ctx bound upper 0.25 is too permissive.\n"
            "  → Fix in v7: tighten c_th2ctx to e.g. [0.08, 0.15].\n"
            "\n"
            "IF spindle_power is 0 even at sweep hotspots:\n"
            "  → FOOOF cannot detect σ peak even though events exist.\n"
            "  → The 'spindles' are too brief/weak for FOOOF's peak fitting.\n"
            "  → Fix: replace spindle_power reward with event-based metric.\n"
        )
    print(f"\nSaved: {OUT_TXT}")

    # CSV
    rows_csv = []
    for r in results:
        if r.get('failed'):
            continue
        con = r['constraints']
        base = dict(tag=r['tag'], **r['point'], **{
            'regime': r['regime']['label'],
            'c_th2ctx': r['regime']['c_th2ctx'],
            'n_passed': r['n_passed'],
            'feasible': r['feasible'],
            'so_power': r['fooof']['so_power'],
            'spindle_power': r['fooof']['spindle_power'],
        })
        # Add key constraint values
        for k in ['T4_q','T5_fwhm','T6_ibi_cv','T6_n_bursts',
                  'T7_sp_cv','T8_n_sp_events','T9_mi','T10_phase',
                  'T10_dist_tgt','T11_lag_ms','T12_n_verified']:
            base[k] = con.get(k)
        for c in ['T1','T2','T3','T4','T5','T6','T7','T8',
                  'T9','T10','T11','T12']:
            base[c] = int(con[c])
        rows_csv.append(base)
    pd.DataFrame(rows_csv).to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
