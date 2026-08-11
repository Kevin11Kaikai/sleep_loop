"""Held-out cortical spindle-density figure for V8a 0712 best0417.

This is a visualization/diagnostic script. It does not modify V8a logic,
does not run DE, and does not run 300 s validation. It mirrors the existing
sigma-RMS held-out detector from validate_spindle_density_heldout.py and adds
the V8a best0417 candidate to the Real/V1/V7 comparison.
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

ROOT = Path(__file__).resolve().parents[1]
if Path.cwd() != ROOT:
    os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, sosfiltfilt

mne.set_log_level("WARNING")

SUBJECT_ID = "SC4001"
OUT_DIR = ROOT / "outputs" / "v8a_0712_best0417_figures"
RELAXED_CSV = ROOT / "outputs" / "v8a_relaxed_t6_sensitivity" / "relaxed_t6_summary.csv"

SIGMA_BAND = (11.0, 15.0)
RMS_WIN_S = 0.20
THRESH_K = 1.5
DUR_LO_S = 0.5
DUR_HI_S = 3.0
MERGE_GAP_S = 0.10
BURN_IN_S = 5.0
SIM_DUR_MS_DENSITY = 120_000
ARTIFACT_THRESH_UV = 200.0

PARAM_NAMES = ["mue", "mui", "b", "tauA", "g_LK", "g_h", "c_th2ctx", "c_ctx2th"]


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_manifest() -> pd.DataFrame:
    p = ROOT / "data" / "manifest.csv"
    try:
        return pd.read_csv(p, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="utf-16")


def pick_eeg_channel(raw) -> str:
    preferred = ["EEG Fpz-Cz", "Fpz-Cz", "EEG FPZ-CZ", "FPZ-CZ"]
    upper = {n.upper(): n for n in raw.ch_names}
    for channel in preferred:
        if channel.upper() in upper:
            return upper[channel.upper()]
    raise RuntimeError(f"No Fpz-Cz-like channel. Available: {raw.ch_names}")


def load_real_n3(prep_mod):
    manifest = read_manifest()
    rows = manifest[manifest["subject_id"] == SUBJECT_ID]
    if rows.empty:
        raise RuntimeError(f"{SUBJECT_ID} not in manifest")
    row = rows.iloc[0]
    raw = mne.io.read_raw_edf(str(row["psg_path"]), preload=True, verbose=False)
    channel = pick_eeg_channel(raw)
    fs = float(raw.info["sfreq"])
    data_uv = raw.get_data(picks=[channel])[0] * 1e6
    stages = prep_mod.load_hypnogram(Path(row["hypnogram_path"]))
    samples_per_epoch = int(prep_mod.EPOCH_LEN_S * fs)
    n_epoch = min(len(stages), len(data_uv) // samples_per_epoch)
    n3_labels = {"N3", "Sleep stage 3", "Sleep stage 4"}
    epochs = []
    n_artifact = 0
    for i in range(n_epoch):
        if stages[i] not in n3_labels:
            continue
        seg = data_uv[i * samples_per_epoch : (i + 1) * samples_per_epoch].astype(float)
        if np.ptp(seg) > ARTIFACT_THRESH_UV:
            n_artifact += 1
            continue
        epochs.append(detrend(seg, type="constant"))
    if not epochs:
        raise RuntimeError("No clean N3 epochs")
    meta = {
        "channel": channel,
        "fs": fs,
        "n_clean_n3": len(epochs),
        "n_artifact": n_artifact,
        "epoch_len_s": float(prep_mod.EPOCH_LEN_S),
    }
    return np.concatenate(epochs), fs, meta


def load_best0417_params() -> dict[str, float]:
    relaxed = pd.read_csv(RELAXED_CSV)
    row = relaxed[relaxed["threshold"].astype(float) == 0.42].iloc[0]
    return {name: float(row[name]) for name in PARAM_NAMES}


def load_params_json(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return {name: float(data[name]) for name in PARAM_NAMES}


def simulate_ctx(module, params: dict[str, float], duration_ms: int):
    model = module.build_model(
        params["mue"],
        params["mui"],
        params["b"],
        params["tauA"],
        params["g_LK"],
        params["g_h"],
        params["c_th2ctx"],
        params["c_ctx2th"],
        duration=duration_ms,
    )
    try:
        if hasattr(module, "seed_numba"):
            module.seed_numba(42)
        model.run()
    except Exception:
        model.params["backend"] = "jitcdde"
        if hasattr(module, "seed_numba"):
            module.seed_numba(42)
        model.run()
    r_exc = model[f"r_mean_{module.EXC}"]
    if r_exc.ndim == 2 and r_exc.shape[0] >= 1:
        r_ctx = r_exc[0, :] * 1000.0
    else:
        r_ctx = np.asarray(r_exc).reshape(-1) * 1000.0
    n_drop = int(BURN_IN_S * module.FS_SIM)
    return detrend(np.asarray(r_ctx[n_drop:], dtype=float), type="constant"), float(module.FS_SIM)


def detect_spindles(x: np.ndarray, fs: float):
    sos = butter(4, SIGMA_BAND, btype="band", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x)
    win = max(1, int(round(RMS_WIN_S * fs)))
    kernel = np.ones(win) / win
    rms = np.sqrt(np.convolve(xf**2, kernel, mode="same"))
    thr = float(rms.mean() + THRESH_K * rms.std())
    above = (rms > thr).astype(np.int8)
    edges = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]
    merged: list[list[int]] = []
    for start, stop in zip(starts, stops):
        if merged and (start - merged[-1][1]) < MERGE_GAP_S * fs:
            merged[-1][1] = int(stop)
        else:
            merged.append([int(start), int(stop)])
    lo = DUR_LO_S * fs
    hi = DUR_HI_S * fs
    events = [(start, stop) for start, stop in merged if lo <= (stop - start) <= hi]
    return events, rms, thr, xf


def density_per_min(events, n_samples: int, fs: float):
    minutes = n_samples / fs / 60.0
    return len(events) / minutes, minutes


def add_row(rows, traces, label: str, signal: np.ndarray, fs: float, source: str):
    events, rms, thr, _xf = detect_spindles(signal, fs)
    density, minutes = density_per_min(events, len(signal), fs)
    rows.append(
        {
            "label": label,
            "n_events": len(events),
            "minutes": round(minutes, 3),
            "density_per_min": round(density, 3),
            "fs": fs,
            "signal": source,
        }
    )
    traces[label] = (rms, thr, fs, events)
    print(f"  {label}: {len(events)} events / {minutes:.2f} min = {density:.3f}/min", flush=True)


def write_outputs(rows: list[dict], traces: dict, meta: dict):
    real_density = rows[0]["density_per_min"]
    csv_path = OUT_DIR / "fig_v8a_0712_heldout_spindle_density.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(rows[0].keys()) + ["abs_diff_vs_real"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["abs_diff_vs_real"] = round(abs(row["density_per_min"] - real_density), 3)
            writer.writerow(out)

    summary_path = OUT_DIR / "fig_v8a_0712_heldout_spindle_density_summary.txt"
    v1 = next(row for row in rows if row["label"].startswith("V1"))
    v7 = next(row for row in rows if row["label"].startswith("V7"))
    v8 = next(row for row in rows if row["label"].startswith("V8a"))
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write("V8a 0712 held-out cortical spindle-density visualization\n")
        fh.write("=" * 72 + "\n")
        fh.write(f"Subject: {SUBJECT_ID}; real channel: {meta['channel']} @ {meta['fs']:.0f} Hz\n")
        fh.write(
            f"Detector: sigma={SIGMA_BAND[0]:.1f}-{SIGMA_BAND[1]:.1f} Hz, "
            f"RMS={RMS_WIN_S:.2f}s, threshold=mean+{THRESH_K:.1f}SD, "
            f"duration=[{DUR_LO_S:.1f},{DUR_HI_S:.1f}]s\n"
        )
        fh.write(f"Model duration: {SIM_DUR_MS_DENSITY/1000:.0f}s; burn-in={BURN_IN_S:.0f}s\n\n")
        fh.write(f"{'label':42s} {'events':>7s} {'min':>7s} {'/min':>8s} {'|real|':>8s}\n")
        for row in rows:
            diff = abs(row["density_per_min"] - real_density)
            fh.write(
                f"{row['label']:42s} {row['n_events']:7d} {row['minutes']:7.2f} "
                f"{row['density_per_min']:8.3f} {diff:8.3f}\n"
            )
        fh.write("\n")
        fh.write("Interpretation:\n")
        fh.write("- V8a 0712 is shown as a controlled audit-patch diagnostic, not final correctness.\n")
        fh.write("- Compare absolute distance to the real EEG density for V1, V7, and V8a.\n")
        fh.write(
            f"- |V1-real|={abs(v1['density_per_min'] - real_density):.3f}, "
            f"|V7-real|={abs(v7['density_per_min'] - real_density):.3f}, "
            f"|V8a-real|={abs(v8['density_per_min'] - real_density):.3f} events/min.\n"
        )

    fig, (axb, axt) = plt.subplots(
        2,
        1,
        figsize=(11.5, 8.8),
        gridspec_kw={"height_ratios": [1, 1.65]},
        dpi=150,
    )
    labels = [row["label"] for row in rows]
    density_values = [row["density_per_min"] for row in rows]
    colors = ["black", "tab:blue", "tab:red", "tab:green"]
    axb.bar(range(len(rows)), density_values, color=colors[: len(rows)])
    axb.axhline(real_density, color="black", linestyle="--", linewidth=1.0, label=f"real = {real_density:.2f}/min")
    axb.set_xticks(range(len(rows)))
    axb.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    axb.set_ylabel("spindle density (events/min)")
    axb.set_title("V8a 0712 held-out cortical spindle density: real vs V1/V7/V8a")
    axb.legend(fontsize=8)
    axb.grid(True, axis="y", alpha=0.2)

    show_labels = [
        "Real SC4001 N3 EEG",
        "V1 canonical (shape_r=0.8586)",
        "V7 event-constrained",
        "V8a 0712 best0417",
    ]
    offset = 0.0
    for label, color in zip(show_labels, colors):
        rms, thr, fs, events = traces[label]
        win = int(15.0 * fs)
        seg = rms[:win]
        denom = seg.max() + 1e-12
        seg_n = seg / denom
        t = np.arange(len(seg_n)) / fs
        axt.plot(t, seg_n + offset, linewidth=0.75, color=color, label=label)
        axt.axhline(thr / denom + offset, color="0.6", linestyle=":", linewidth=0.7)
        for start, stop in events:
            if start < win:
                axt.axvspan(start / fs, min(stop, win) / fs, alpha=0.055, color="red")
        offset += 1.15
    axt.set_xlabel("time (s) [first 15 s; envelopes peak-normalized, stacked]")
    axt.set_ylabel("normalized sigma-RMS envelope")
    axt.set_title("Example sigma-RMS envelopes with detection threshold (dotted)")
    axt.legend(fontsize=8, loc="upper right")
    axt.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig_path = OUT_DIR / "fig_v8a_0712_heldout_spindle_density.png"
    fig.savefig(fig_path)
    plt.close(fig)

    print(f"wrote {fig_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prep_mod = import_module_from_path("sd_prep_0712", ROOT / "utils" / "02_preprocess_psd.py")
    v7 = import_module_from_path("sd_v7_0712", ROOT / "models" / "s4_personalize_fig7_v7.py")
    v8 = import_module_from_path("sd_v8_0712", ROOT / "models" / "s4_personalize_fig7_v8.py")

    print(
        f"Detector: sigma={SIGMA_BAND} Hz, RMS win={RMS_WIN_S}s, "
        f"thr=mean+{THRESH_K}*SD, dur=[{DUR_LO_S},{DUR_HI_S}]s"
    )
    print("Loading real SC4001 N3 EEG ...")
    real_sig, fs_real, meta = load_real_n3(prep_mod)
    print(f"  {meta['n_clean_n3']} clean N3 epochs, {meta['channel']} @ {fs_real:.0f} Hz")

    rows: list[dict] = []
    traces: dict = {}
    add_row(rows, traces, "Real SC4001 N3 EEG", real_sig, fs_real, meta["channel"])

    versions = [
        (
            "V1 canonical (shape_r=0.8586)",
            v7,
            load_params_json(ROOT / "data" / "patient_params_fig7_v1_0418_2_SC4001.json"),
        ),
        (
            "V7 event-constrained",
            v7,
            load_params_json(ROOT / "data" / "patient_params_fig7_v7_SC4001.json"),
        ),
        ("V8a 0712 best0417", v8, load_best0417_params()),
    ]
    for label, module, params in versions:
        print(f"Simulating {label} ({SIM_DUR_MS_DENSITY/1000:.0f}s) ...", flush=True)
        r_ctx, fs = simulate_ctx(module, params, SIM_DUR_MS_DENSITY)
        add_row(rows, traces, label, r_ctx, fs, "cortical r_ctx")

    write_outputs(rows, traces, meta)


if __name__ == "__main__":
    main()
