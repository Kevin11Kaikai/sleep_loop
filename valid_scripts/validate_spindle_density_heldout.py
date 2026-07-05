"""
Held-out spindle-density check for the BMEiCON V7 paper (panel Item 2B).

Second, event-level held-out statistic, disjoint from T1-T12: empirical spindle
density (events per minute) measured with a standard Moelle/Ferrarelli sigma-band
RMS detector applied identically to real SC4001 N3 EEG and to simulated CORTICAL
firing rate (the EEG analog) for V1/V3/V7.

Why this is "held-out" / disjoint from the fitness audit:
  - operates on the cortical / scalp signal (T8/T12 operate on the THALAMIC signal),
  - uses a moving-RMS envelope + mean+1.5*SD threshold (T8 uses a 75th-percentile
    Gaussian-smoothed Hilbert envelope; T12 adds Welch peak-inside verification),
  - uses standard 0.5-3.0 s spindle-duration bounds (T8 uses 0.3-2.0 s),
  - never calls compute_constraints_v7 and reuses no T1-T12 / shape_r fields.

Run from repo root:
    python valid_scripts/validate_spindle_density_heldout.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if Path.cwd() != ROOT:
    os.chdir(ROOT)

# Local patched neurolib takes precedence (mirrors the audit harness)
LOCAL_NEUROLIB = r"D:\Year3_Mao_Projects\neurolib"
if os.path.isdir(LOCAL_NEUROLIB):
    sys.path.insert(0, LOCAL_NEUROLIB)

import numpy as np

for _attr in ("object", "bool", "int", "float", "complex", "str"):
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(__builtins__, _attr, object))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import pandas as pd
from scipy.signal import butter, detrend, sosfiltfilt

mne.set_log_level("WARNING")

SUBJECT_ID = "SC4001"
PARAM_COLS = ["mue", "mui", "b", "tauA", "g_LK", "g_h", "c_th2ctx", "c_ctx2th"]
OUT_DIR = ROOT / "validation_outputs"

# Detector parameters (standard sigma-RMS spindle detector)
SIGMA_BAND   = (11.0, 15.0)   # Hz
RMS_WIN_S    = 0.20           # moving-RMS window
THRESH_K     = 1.5            # threshold = mean(RMS) + K*SD(RMS)
DUR_LO_S     = 0.5            # min spindle duration
DUR_HI_S     = 3.0           # max spindle duration
MERGE_GAP_S  = 0.10          # merge supra-threshold runs closer than this

BURN_IN_S          = 5.0
SIM_DUR_MS_DENSITY = 120_000  # 120 s sim -> ~115 s after burn-in for a stable rate
ARTIFACT_THRESH_UV = 200.0

# Canonical V1 is the *_v1_0418_2_* (shape_r=0.8586) point, per the audit.
VERSIONS = [
    ("V1 canonical (shape_r=0.8586)", ROOT / "data" / "patient_params_fig7_v1_0418_2_SC4001.json"),
    ("V1 taskfile (shape_r=0.8226)", ROOT / "data" / "patient_params_fig7_SC4001.json"),
    ("V3 dynamics", ROOT / "data" / "patient_params_fig7_v3_SC4001.json"),
    ("V7 event-constrained", ROOT / "data" / "patient_params_fig7_v7_SC4001.json"),
]


def import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_manifest():
    p = ROOT / "data" / "manifest.csv"
    try:
        return pd.read_csv(p, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="utf-16")


def pick_eeg_channel(raw):
    preferred = ["EEG Fpz-Cz", "Fpz-Cz", "EEG FPZ-CZ", "FPZ-CZ"]
    upper = {n.upper(): n for n in raw.ch_names}
    for c in preferred:
        if c.upper() in upper:
            return upper[c.upper()]
    raise RuntimeError(f"No Fpz-Cz-like channel. Available: {raw.ch_names}")


def load_real_n3(prep_mod):
    manifest = read_manifest()
    rows = manifest[manifest["subject_id"] == SUBJECT_ID]
    if rows.empty:
        raise RuntimeError(f"{SUBJECT_ID} not in manifest")
    row = rows.iloc[0]
    raw = mne.io.read_raw_edf(str(row["psg_path"]), preload=True, verbose=False)
    ch = pick_eeg_channel(raw)
    fs = float(raw.info["sfreq"])
    data_uv = raw.get_data(picks=[ch])[0] * 1e6
    stages = prep_mod.load_hypnogram(Path(row["hypnogram_path"]))
    nspe = int(prep_mod.EPOCH_LEN_S * fs)
    n_ep = min(len(stages), len(data_uv) // nspe)
    n3 = {"N3", "Sleep stage 3", "Sleep stage 4"}
    epochs = []
    n_artifact = 0
    for i in range(n_ep):
        if stages[i] not in n3:
            continue
        seg = data_uv[i * nspe:(i + 1) * nspe].astype(float)
        if np.ptp(seg) > ARTIFACT_THRESH_UV:
            n_artifact += 1
            continue
        epochs.append(detrend(seg, type="constant"))
    if not epochs:
        raise RuntimeError("No clean N3 epochs")
    meta = {"channel": ch, "fs": fs, "n_clean_n3": len(epochs),
            "n_artifact": n_artifact, "epoch_len_s": float(prep_mod.EPOCH_LEN_S)}
    return np.concatenate(epochs), fs, meta


def simulate_ctx(v7, params_path, duration_ms):
    with open(params_path, encoding="utf-8") as fh:
        bp = json.load(fh)
    m = v7.build_model(bp["mue"], bp["mui"], bp["b"], bp["tauA"],
                       bp["g_LK"], bp["g_h"], bp["c_th2ctx"], bp["c_ctx2th"],
                       duration=duration_ms)
    try:
        v7.seed_numba(42)
        m.run()
    except Exception:
        m.params["backend"] = "jitcdde"
        v7.seed_numba(42)
        m.run()
    r_exc = m[f"r_mean_{v7.EXC}"]
    if r_exc.ndim == 2 and r_exc.shape[0] >= 1:
        r_ctx = r_exc[0, :] * 1000.0
    else:
        r_ctx = np.asarray(r_exc).reshape(-1) * 1000.0
    n_drop = int(BURN_IN_S * v7.FS_SIM)
    return detrend(np.asarray(r_ctx[n_drop:], dtype=float), type="constant"), float(v7.FS_SIM)


def detect_spindles(x, fs):
    """Standard sigma-RMS detector. Returns (events[(start,stop)], rms, thr, xf)."""
    sos = butter(4, SIGMA_BAND, btype="band", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x)
    win = max(1, int(round(RMS_WIN_S * fs)))
    kernel = np.ones(win) / win
    rms = np.sqrt(np.convolve(xf ** 2, kernel, mode="same"))
    thr = float(rms.mean() + THRESH_K * rms.std())
    above = (rms > thr).astype(np.int8)
    edges = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]
    merged = []
    for s, e in zip(starts, stops):
        if merged and (s - merged[-1][1]) < MERGE_GAP_S * fs:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    lo, hi = DUR_LO_S * fs, DUR_HI_S * fs
    events = [(s, e) for s, e in merged if lo <= (e - s) <= hi]
    return events, rms, thr, xf


def density_per_min(events, n_samples, fs):
    minutes = n_samples / fs / 60.0
    return len(events) / minutes, minutes


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prep_mod = import_module_from_path("sd_prep", ROOT / "utils" / "02_preprocess_psd.py")
    v7 = import_module_from_path("sd_v7", ROOT / "models" / "s4_personalize_fig7_v7.py")

    print(f"Detector: sigma={SIGMA_BAND} Hz, RMS win={RMS_WIN_S}s, thr=mean+{THRESH_K}*SD, "
          f"dur=[{DUR_LO_S},{DUR_HI_S}]s")
    print("Loading real SC4001 N3 EEG ...")
    real_sig, fs_real, meta = load_real_n3(prep_mod)
    print(f"  {meta['n_clean_n3']} clean N3 epochs, {meta['channel']} @ {fs_real:.0f} Hz")

    rows = []
    traces = {}  # label -> (rms, thr, fs, events)

    # Real
    ev, rms, thr, xf = detect_spindles(real_sig, fs_real)
    dens, mins = density_per_min(ev, len(real_sig), fs_real)
    rows.append({"label": "Real SC4001 N3 EEG", "n_events": len(ev),
                 "minutes": round(mins, 2), "density_per_min": round(dens, 3),
                 "fs": fs_real, "signal": meta["channel"]})
    traces["Real SC4001 N3 EEG"] = (rms, thr, fs_real, ev)
    print(f"  Real: {len(ev)} events / {mins:.1f} min = {dens:.2f}/min")

    # Sims
    for label, path in VERSIONS:
        print(f"Simulating {label} ({SIM_DUR_MS_DENSITY/1000:.0f}s) ...", flush=True)
        r_ctx, fs_sim = simulate_ctx(v7, path, SIM_DUR_MS_DENSITY)
        ev, rms, thr, xf = detect_spindles(r_ctx, fs_sim)
        dens, mins = density_per_min(ev, len(r_ctx), fs_sim)
        rows.append({"label": label, "n_events": len(ev),
                     "minutes": round(mins, 2), "density_per_min": round(dens, 3),
                     "fs": fs_sim, "signal": "cortical r_ctx"})
        traces[label] = (rms, thr, fs_sim, ev)
        print(f"  {label}: {len(ev)} events / {mins:.1f} min = {dens:.2f}/min")

    real_dens = rows[0]["density_per_min"]

    # ── CSV ────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "spindle_density_heldout.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + ["abs_diff_vs_real"])
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["abs_diff_vs_real"] = round(abs(r["density_per_min"] - real_dens), 3)
            w.writerow(r2)

    # ── Summary ────────────────────────────────────────────────────────
    sim_rows = rows[1:]
    v1c = next(r for r in sim_rows if r["label"].startswith("V1 canonical"))
    v7r = next(r for r in sim_rows if r["label"].startswith("V7"))
    d_v1 = abs(v1c["density_per_min"] - real_dens)
    d_v7 = abs(v7r["density_per_min"] - real_dens)
    favored = "V7" if d_v7 < d_v1 else ("V1" if d_v1 < d_v7 else "tie")

    sum_path = OUT_DIR / "spindle_density_heldout_summary.txt"
    with open(sum_path, "w", encoding="utf-8") as fh:
        fh.write("Held-out spindle-density check (Item 2B)\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"Subject: {SUBJECT_ID}   real channel: {meta['channel']} @ {fs_real:.0f} Hz\n")
        fh.write(f"Detector: sigma={SIGMA_BAND} Hz, RMS win={RMS_WIN_S}s, "
                 f"thr=mean+{THRESH_K}*SD, dur=[{DUR_LO_S},{DUR_HI_S}]s, merge<{MERGE_GAP_S}s\n")
        fh.write(f"Sim duration: {SIM_DUR_MS_DENSITY/1000:.0f}s, burn-in {BURN_IN_S}s, "
                 f"detector on cortical r_ctx. Disjoint from T1-T12.\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"{'signal':34s} {'events':>7s} {'min':>7s} {'/min':>7s} {'|Δ real|':>9s}\n")
        for r in rows:
            d = abs(r["density_per_min"] - real_dens)
            fh.write(f"{r['label']:34s} {r['n_events']:>7d} {r['minutes']:>7.1f} "
                     f"{r['density_per_min']:>7.2f} {d:>9.2f}\n")
        fh.write("\n")
        fh.write(f"Real spindle density           : {real_dens:.2f} /min\n")
        fh.write(f"|V1 canonical - real|          : {d_v1:.2f}\n")
        fh.write(f"|V7 - real|                    : {d_v7:.2f}\n")
        fh.write(f"=> Held-out spindle density FAVORS: {favored}\n\n")
        fh.write("Interpretation: if this favors V7, the held-out evidence is MIXED\n")
        fh.write("(SO morphology favored V1; spindle density favors V7) rather than\n")
        fh.write("uniformly pro-V1, which licenses a directional event-level reading.\n")
        fh.write("If it also favors V1, the conservative 'invalid-under-audit' claim stands.\n")

    # ── Figure: density bars + example envelope/threshold traces ────────
    fig, (axb, axt) = plt.subplots(2, 1, figsize=(10, 8),
                                   gridspec_kw={"height_ratios": [1, 1.6]})
    labels = [r["label"] for r in rows]
    dens_vals = [r["density_per_min"] for r in rows]
    colors = ["black"] + ["tab:blue", "tab:cyan", "tab:orange", "tab:red"][:len(rows) - 1]
    axb.bar(range(len(rows)), dens_vals, color=colors)
    axb.axhline(real_dens, color="black", ls="--", lw=1, label=f"real = {real_dens:.2f}/min")
    axb.set_xticks(range(len(rows)))
    axb.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    axb.set_ylabel("spindle density (events/min)")
    axb.set_title("Held-out spindle density: real vs V1/V3/V7 (sigma-RMS detector)")
    axb.legend(fontsize=8)

    # Example 15 s envelope window for real, V1 canonical, V7
    show = ["Real SC4001 N3 EEG", "V1 canonical (shape_r=0.8586)", "V7 event-constrained"]
    offset = 0.0
    for label in show:
        if label not in traces:
            continue
        rms, thr, fs, ev = traces[label]
        win = int(15.0 * fs)
        seg = rms[:win]
        seg_n = seg / (seg.max() + 1e-12)
        t = np.arange(len(seg_n)) / fs
        axt.plot(t, seg_n + offset, lw=0.7, label=label)
        thr_n = thr / (seg.max() + 1e-12)
        axt.axhline(thr_n + offset, color="0.6", ls=":", lw=0.6)
        for s, e in ev:
            if s < win:
                axt.axvspan(s / fs, min(e, win) / fs, ymin=0, ymax=1, alpha=0.06, color="red")
        offset += 1.2
    axt.set_xlabel("time (s)  [first 15 s; envelopes peak-normalized, stacked]")
    axt.set_ylabel("normalized sigma-RMS envelope")
    axt.set_title("Example sigma-RMS envelopes with detection threshold (dotted)")
    axt.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_spindle_density_heldout.png", dpi=140)

    print(f"\nReal={real_dens:.2f}/min  |V1c-real|={d_v1:.2f}  |V7-real|={d_v7:.2f}  => FAVORS {favored}")
    print(f"wrote {csv_path}")
    print(f"wrote {sum_path}")
    print(f"wrote {OUT_DIR / 'fig_spindle_density_heldout.png'}")


if __name__ == "__main__":
    main()
