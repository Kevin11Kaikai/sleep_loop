"""
Confirmatory held-out spindle-density re-run (panel Item 2B robustness).

Tests whether V7's near-zero cortical spindle density is real or an artifact of
(a) short simulation or (b) sigma-band choice. Same detector as
validate_spindle_density_heldout.py, but:
  - longer simulation: 300 s
  - two sigma bands compared: 10-14 Hz (model T-current resonance) and 11-15 Hz (EEG sigma)
  - only V1 canonical (shape_r=0.8586) and V7 event-constrained (+ real EEG reference)

Does NOT edit the manuscript. Outputs to validation_outputs/.
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
OUT_DIR = ROOT / "validation_outputs"

# Detector parameters (identical to the original spindle-density script except band)
RMS_WIN_S   = 0.20
THRESH_K    = 1.5
DUR_LO_S    = 0.5
DUR_HI_S    = 3.0
MERGE_GAP_S = 0.10

BANDS = [(10.0, 14.0), (11.0, 15.0)]

BURN_IN_S          = 5.0
SIM_DUR_MS         = 300_000      # 300 s (confirmatory; ~295 s after burn-in)
ARTIFACT_THRESH_UV = 200.0

VERSIONS = [
    ("V1 canonical (0.8586)", ROOT / "data" / "patient_params_fig7_v1_0418_2_SC4001.json"),
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
    for c in ["EEG Fpz-Cz", "Fpz-Cz", "EEG FPZ-CZ", "FPZ-CZ"]:
        upper = {n.upper(): n for n in raw.ch_names}
        if c.upper() in upper:
            return upper[c.upper()]
    raise RuntimeError(f"No Fpz-Cz-like channel. Available: {raw.ch_names}")


def load_real_n3(prep_mod):
    manifest = read_manifest()
    rows = manifest[manifest["subject_id"] == SUBJECT_ID]
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
    for i in range(n_ep):
        if stages[i] not in n3:
            continue
        seg = data_uv[i * nspe:(i + 1) * nspe].astype(float)
        if np.ptp(seg) > ARTIFACT_THRESH_UV:
            continue
        epochs.append(detrend(seg, type="constant"))
    return np.concatenate(epochs), fs, {"channel": ch, "n_clean_n3": len(epochs)}


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
    r_ctx = (r_exc[0, :] if (r_exc.ndim == 2 and r_exc.shape[0] >= 1)
             else np.asarray(r_exc).reshape(-1)) * 1000.0
    n_drop = int(BURN_IN_S * v7.FS_SIM)
    return detrend(np.asarray(r_ctx[n_drop:], dtype=float), type="constant"), float(v7.FS_SIM)


def detect_spindles(x, fs, band):
    sos = butter(4, band, btype="band", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x)
    win = max(1, int(round(RMS_WIN_S * fs)))
    rms = np.sqrt(np.convolve(xf ** 2, np.ones(win) / win, mode="same"))
    thr = float(rms.mean() + THRESH_K * rms.std())
    above = (rms > thr).astype(np.int8)
    edges = np.diff(np.concatenate(([0], above, [0])))
    starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
    merged = []
    for s, e in zip(starts, stops):
        if merged and (s - merged[-1][1]) < MERGE_GAP_S * fs:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    lo, hi = DUR_LO_S * fs, DUR_HI_S * fs
    return [(s, e) for s, e in merged if lo <= (e - s) <= hi]


def density(events, n, fs):
    minutes = n / fs / 60.0
    return len(events) / minutes, minutes


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prep_mod = import_module_from_path("sdc_prep", ROOT / "utils" / "02_preprocess_psd.py")
    v7 = import_module_from_path("sdc_v7", ROOT / "models" / "s4_personalize_fig7_v7.py")

    print(f"Confirmatory run: sim {SIM_DUR_MS/1000:.0f}s, bands {BANDS}, "
          f"thr=mean+{THRESH_K}*SD, dur=[{DUR_LO_S},{DUR_HI_S}]s")
    real_sig, fs_real, meta = load_real_n3(prep_mod)
    print(f"Real: {meta['n_clean_n3']} clean N3 epochs, {meta['channel']} @ {fs_real:.0f} Hz")

    # Simulate once per model (reuse the same long trace across bands)
    sims = {}
    for label, path in VERSIONS:
        print(f"Simulating {label} ({SIM_DUR_MS/1000:.0f}s) ...", flush=True)
        r_ctx, fs_sim = simulate_ctx(v7, path, SIM_DUR_MS)
        sims[label] = (r_ctx, fs_sim)

    rows = []
    for band in BANDS:
        bstr = f"{band[0]:.0f}-{band[1]:.0f}Hz"
        ev = detect_spindles(real_sig, fs_real, band)
        d_real, m_real = density(ev, len(real_sig), fs_real)
        rows.append({"signal": "Real SC4001 N3 EEG", "band": bstr, "n_events": len(ev),
                     "minutes": round(m_real, 1), "density_per_min": round(d_real, 3)})
        print(f"[{bstr}] Real: {d_real:.2f}/min ({len(ev)} ev / {m_real:.1f} min)")
        for label, (r_ctx, fs_sim) in sims.items():
            evs = detect_spindles(r_ctx, fs_sim, band)
            d, mn = density(evs, len(r_ctx), fs_sim)
            rows.append({"signal": label, "band": bstr, "n_events": len(evs),
                         "minutes": round(mn, 1), "density_per_min": round(d, 3)})
            print(f"[{bstr}] {label}: {d:.2f}/min ({len(evs)} ev / {mn:.1f} min)")

    # ── CSV ──
    csv_path = OUT_DIR / "spindle_density_confirm.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["signal", "band", "n_events", "minutes", "density_per_min"])
        w.writeheader()
        w.writerows(rows)

    # ── Summary ──
    sum_path = OUT_DIR / "spindle_density_confirm_summary.txt"
    with open(sum_path, "w", encoding="utf-8") as fh:
        fh.write("Confirmatory held-out spindle-density re-run (Item 2B robustness)\n")
        fh.write("=" * 64 + "\n")
        fh.write(f"Subject {SUBJECT_ID}; real {meta['channel']} @ {fs_real:.0f} Hz, "
                 f"{meta['n_clean_n3']} clean N3 epochs\n")
        fh.write(f"Sim {SIM_DUR_MS/1000:.0f}s, burn-in {BURN_IN_S}s, detector on cortical r_ctx, "
                 f"thr=mean+{THRESH_K}*SD, dur=[{DUR_LO_S},{DUR_HI_S}]s\n")
        fh.write("=" * 64 + "\n\n")
        for band in BANDS:
            bstr = f"{band[0]:.0f}-{band[1]:.0f}Hz"
            fh.write(f"sigma band {bstr}:\n")
            br = [r for r in rows if r["band"] == bstr]
            real_d = next(r["density_per_min"] for r in br if r["signal"].startswith("Real"))
            for r in br:
                d = abs(r["density_per_min"] - real_d)
                fh.write(f"  {r['signal']:26s} {r['density_per_min']:6.2f}/min "
                         f"({r['n_events']:>3d} ev / {r['minutes']:.0f} min)  |Δreal|={d:.2f}\n")
            v1 = next(r["density_per_min"] for r in br if r["signal"].startswith("V1"))
            v7d = next(r["density_per_min"] for r in br if r["signal"].startswith("V7"))
            fav = "V7" if abs(v7d - real_d) < abs(v1 - real_d) else ("V1" if abs(v1 - real_d) < abs(v7d - real_d) else "tie")
            fh.write(f"  -> favors {fav}\n\n")
        fh.write("Conclusion:\n")
        v7_all = [r["density_per_min"] for r in rows if r["signal"].startswith("V7")]
        fh.write(f"  V7 cortical spindle density across bands: {v7_all} /min\n")
        if max(v7_all) < 0.5:
            fh.write("  V7 is near-zero in BOTH bands at 300 s -> the zero is robust, not a\n")
            fh.write("  short-duration or band artifact. Thalamic spindles (T8/T12) do not\n")
            fh.write("  propagate to the EEG-analog cortical signal at V7's small c_th2ctx.\n")
        else:
            fh.write("  V7 shows non-trivial cortical spindles in at least one band -> the\n")
            fh.write("  earlier zero was partly a duration/band artifact; revise interpretation.\n")
    # ── Figure ──
    fig, ax = plt.subplots(figsize=(8.5, 5))
    signals = ["Real SC4001 N3 EEG", "V1 canonical (0.8586)", "V7 event-constrained"]
    x = np.arange(len(signals))
    width = 0.38
    for j, band in enumerate(BANDS):
        bstr = f"{band[0]:.0f}-{band[1]:.0f}Hz"
        vals = []
        for s in signals:
            match = [r for r in rows if r["signal"] == s and r["band"] == bstr]
            vals.append(match[0]["density_per_min"] if match else 0.0)
        ax.bar(x + (j - 0.5) * width, vals, width, label=f"sigma {bstr}")
    ax.set_xticks(x)
    ax.set_xticklabels(signals, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("spindle density (events/min)")
    ax.set_title(f"Confirmatory held-out spindle density ({SIM_DUR_MS/1000:.0f}s sims, 2 bands)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_spindle_density_confirm.png", dpi=140)

    print(f"\nwrote {csv_path}")
    print(f"wrote {sum_path}")
    print(f"wrote {OUT_DIR / 'fig_spindle_density_confirm.png'}")


if __name__ == "__main__":
    main()
