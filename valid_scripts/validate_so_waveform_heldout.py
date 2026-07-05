"""
BMEiCON V7 论文的留出慢波（slow-oscillation, SO）波形验证。

本脚本将真实的 SC4001 N3 EEG 数据和 V1/V3/V7 仿真模拟生成的平均 SO 周期波形形态进行比较。
该度量方法有意与 shape_r，so_power/T4，spindle_power/T12 以及 T1-T12 可行性标签相独立。

请在项目根目录下运行：
    python valid_scripts/validate_so_waveform_heldout.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
# import python标准库
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if Path.cwd() != ROOT:
    os.chdir(ROOT)
#找到项目根目录
import numpy as np

for _attr in ("object", "bool", "int", "float", "complex", "str"):
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(__builtins__, _attr, object))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import pandas as pd
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt
from scipy.stats import pearsonr

mne.set_log_level("WARNING")
#导入信号处理和科学计算库

SUBJECT_ID = "SC4001"
PARAM_COLS = ["mue", "mui", "b", "tauA", "g_LK", "g_h", "c_th2ctx", "c_ctx2th"]

OUT_DIR = ROOT / "validation_outputs"
METRICS_CSV = OUT_DIR / "so_waveform_heldout_metrics.csv"
SUMMARY_TXT = OUT_DIR / "so_waveform_heldout_summary.txt"
FIG_TEMPLATES = OUT_DIR / "fig_so_waveform_templates.png"
FIG_DISTANCES = OUT_DIR / "fig_so_waveform_distances.png"

SIM_DUR_MS = 60_000
BURN_IN_S = 5.0
ARTIFACT_THRESH_UV = 200.0
SO_BAND = (0.3, 1.5)
TEMPLATE_HALF_WINDOW_S = 1.0
MIN_PEAK_DISTANCE_S = 0.65
MIN_SNIPPETS = 5
# 全局参数和脚本配置区

VERSIONS = [
    ("V1 spectral-only", ROOT / "data" / "patient_params_fig7_SC4001.json"),
    ("V3 dynamics", ROOT / "data" / "patient_params_fig7_v3_SC4001.json"),
    ("V7 event-constrained", ROOT / "data" / "patient_params_fig7_v7_SC4001.json"),
]


@dataclass
class TemplateResult:
    label: str
    template: np.ndarray
    time_s: np.ndarray
    n_snippets: int
    anchor_mode: str
    fs: float


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_v7_module():
    return import_module_from_path("heldout_v7", ROOT / "models" / "s4_personalize_fig7_v7.py")


def load_preprocess_module():
    return import_module_from_path("heldout_preprocess", ROOT / "utils" / "02_preprocess_psd.py")


def read_manifest() -> pd.DataFrame:
    manifest_path = ROOT / "data" / "manifest.csv"
    try:
        return pd.read_csv(manifest_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(manifest_path, encoding="utf-16")


def pick_eeg_channel(raw: mne.io.BaseRaw) -> str:
    preferred = ["EEG Fpz-Cz", "Fpz-Cz", "EEG FPZ-CZ", "FPZ-CZ"]
    upper_to_name = {name.upper(): name for name in raw.ch_names}
    for candidate in preferred:
        if candidate.upper() in upper_to_name:
            return upper_to_name[candidate.upper()]
    raise RuntimeError(f"No Fpz-Cz-like EEG channel found. Available: {raw.ch_names}")


def load_real_n3_epochs(prep_mod) -> tuple[list[np.ndarray], float, dict]:
    manifest = read_manifest()
    subj_rows = manifest[manifest["subject_id"] == SUBJECT_ID]
    if subj_rows.empty:
        raise RuntimeError(f"{SUBJECT_ID} not found in data/manifest.csv")

    row = subj_rows.iloc[0]
    raw = mne.io.read_raw_edf(str(row["psg_path"]), preload=True, verbose=False)
    channel = pick_eeg_channel(raw)
    fs = float(raw.info["sfreq"])
    data_uv = raw.get_data(picks=[channel])[0] * 1e6

    stages = prep_mod.load_hypnogram(Path(row["hypnogram_path"]))
    n_samples_per_epoch = int(prep_mod.EPOCH_LEN_S * fs)
    n_epochs = min(len(stages), len(data_uv) // n_samples_per_epoch)

    n3_labels = {"N3", "Sleep stage 3", "Sleep stage 4"}
    epochs: list[np.ndarray] = []
    n_n3_total = 0
    n_artifact = 0

    for idx in range(n_epochs):
        if stages[idx] not in n3_labels:
            continue
        n_n3_total += 1
        start = idx * n_samples_per_epoch
        stop = start + n_samples_per_epoch
        epoch = data_uv[start:stop].astype(float)
        if np.ptp(epoch) > ARTIFACT_THRESH_UV:
            n_artifact += 1
            continue
        epochs.append(detrend(epoch, type="constant"))

    if not epochs:
        raise RuntimeError("No real N3 EEG epochs survived artifact rejection")

    meta = {
        "channel": channel,
        "fs": fs,
        "n_epochs_total": int(n_epochs),
        "n_n3_total": int(n_n3_total),
        "n_artifact_rejected": int(n_artifact),
        "n_clean_n3": int(len(epochs)),
    }
    return epochs, fs, meta


def load_params(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def simulate_version(v7, label: str, params_path: Path) -> np.ndarray:
    bp = load_params(params_path)
    missing = [key for key in PARAM_COLS if key not in bp]
    if missing:
        raise RuntimeError(f"{params_path} missing parameter keys: {missing}")

    m = v7.build_model(
        bp["mue"],
        bp["mui"],
        bp["b"],
        bp["tauA"],
        bp["g_LK"],
        bp["g_h"],
        bp["c_th2ctx"],
        bp["c_ctx2th"],
        duration=SIM_DUR_MS,
    )
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
    r_ctx = np.asarray(r_ctx[n_drop:], dtype=float)
    if not np.all(np.isfinite(r_ctx)):
        raise RuntimeError(f"{label} simulation produced non-finite r_ctx values")
    return detrend(r_ctx, type="constant")


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = float(np.std(x))
    if sd < 1e-12:
        return x - float(np.mean(x))
    return (x - float(np.mean(x))) / sd


def bandpass_so(x: np.ndarray, fs: float) -> np.ndarray:
    sos = butter(4, SO_BAND, btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def snippets_from_segment(
    segment: np.ndarray,
    fs: float,
    anchor_mode: str,
    half_window_s: float,
) -> list[np.ndarray]:
    x = zscore(segment)
    x_so = bandpass_so(x, fs)

    anchor_signal = x_so if anchor_mode == "peak" else -x_so
    min_distance = int(MIN_PEAK_DISTANCE_S * fs)
    prominence = max(0.15 * np.nanstd(anchor_signal), 1e-6)
    peaks, _ = find_peaks(anchor_signal, distance=min_distance, prominence=prominence)

    half_n = int(round(half_window_s * fs))
    snippets: list[np.ndarray] = []
    for pk in peaks:
        start = pk - half_n
        stop = pk + half_n + 1
        if start < 0 or stop > len(x_so):
            continue
        snippet = x_so[start:stop]
        amp = np.ptp(snippet)
        if amp < 1e-9:
            continue
        snippets.append(zscore(snippet))
    return snippets


def build_template(
    label: str,
    segments: Iterable[np.ndarray],
    fs: float,
    anchor_mode: str,
) -> TemplateResult:
    all_snippets: list[np.ndarray] = []
    for segment in segments:
        if len(segment) < int(2.5 * fs):
            continue
        all_snippets.extend(
            snippets_from_segment(segment, fs, anchor_mode, TEMPLATE_HALF_WINDOW_S)
        )

    if len(all_snippets) < MIN_SNIPPETS:
        raise RuntimeError(
            f"{label} produced only {len(all_snippets)} snippets "
            f"with {anchor_mode} anchors"
        )

    snippets = np.vstack(all_snippets)
    template = np.mean(snippets, axis=0)
    template = zscore(template)
    n = len(template)
    time_s = (np.arange(n) - n // 2) / fs
    return TemplateResult(
        label=label,
        template=template,
        time_s=time_s,
        n_snippets=len(all_snippets),
        anchor_mode=anchor_mode,
        fs=fs,
    )


def resample_template(result: TemplateResult, target_time: np.ndarray) -> np.ndarray:
    return np.interp(target_time, result.time_s, result.template)


def compute_distances(real: TemplateResult, sim: TemplateResult) -> dict:
    sim_template = resample_template(sim, real.time_s)
    rmse = float(np.sqrt(np.mean((sim_template - real.template) ** 2)))
    corr = float(pearsonr(real.template, sim_template)[0])
    return {
        "label": sim.label,
        "template_rmse": rmse,
        "template_corr": corr,
        "template_1_minus_corr": float(1.0 - corr),
        "n_snippets": sim.n_snippets,
        "anchor_mode": sim.anchor_mode,
    }


def choose_real_anchor(
    real_epochs: list[np.ndarray],
    fs_real: float,
    v7_template: TemplateResult,
) -> TemplateResult:
    candidates = [
        build_template("Real SC4001 N3 EEG", real_epochs, fs_real, "peak"),
        build_template("Real SC4001 N3 EEG", real_epochs, fs_real, "trough"),
    ]
    best = None
    best_corr = -np.inf
    for candidate in candidates:
        v7_on_real_grid = resample_template(v7_template, candidate.time_s)
        corr = float(pearsonr(candidate.template, v7_on_real_grid)[0])
        if corr > best_corr:
            best_corr = corr
            best = candidate
    return best


def write_metrics(rows: list[dict], real_result: TemplateResult, real_meta: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "template_rmse",
        "template_corr",
        "template_1_minus_corr",
        "n_snippets",
        "anchor_mode",
        "real_anchor_mode",
        "real_n_snippets",
        "real_clean_n3_epochs",
        "real_channel",
    ]
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            enriched = dict(row)
            enriched["real_anchor_mode"] = real_result.anchor_mode
            enriched["real_n_snippets"] = real_result.n_snippets
            enriched["real_clean_n3_epochs"] = real_meta["n_clean_n3"]
            enriched["real_channel"] = real_meta["channel"]
            writer.writerow(enriched)


def plot_templates(real: TemplateResult, sims: list[TemplateResult]) -> None:
    plt.figure(figsize=(8.0, 4.8))
    plt.plot(real.time_s, real.template, color="black", linewidth=2.5, label="Real EEG")
    for sim in sims:
        plt.plot(
            real.time_s,
            resample_template(sim, real.time_s),
            linewidth=1.8,
            label=f"{sim.label} (n={sim.n_snippets})",
        )
    plt.axvline(0, color="0.5", linestyle="--", linewidth=1)
    plt.xlabel("Time from SO anchor (s)")
    plt.ylabel("Normalized SO waveform")
    plt.title("Held-out SO average-cycle waveform templates")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_TEMPLATES, dpi=180)
    plt.close()


def plot_distances(rows: list[dict]) -> None:
    labels = [row["label"] for row in rows]
    rmse = [row["template_rmse"] for row in rows]
    one_minus_corr = [row["template_1_minus_corr"] for row in rows]

    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(8.0, 4.8))
    plt.bar(x - width / 2, rmse, width, label="RMSE")
    plt.bar(x + width / 2, one_minus_corr, width, label="1 - corr")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Distance to real EEG template")
    plt.title("Held-out SO waveform morphology distance")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DISTANCES, dpi=180)
    plt.close()


def write_summary(rows: list[dict], real_result: TemplateResult, real_meta: dict) -> None:
    best_rmse = min(rows, key=lambda row: row["template_rmse"])
    best_corr = max(rows, key=lambda row: row["template_corr"])
    lines = [
        "Held-out SO waveform validation",
        "=" * 40,
        f"Subject: {SUBJECT_ID}",
        f"Real EEG channel: {real_meta['channel']}",
        f"Clean real N3 epochs: {real_meta['n_clean_n3']} "
        f"(artifact rejected: {real_meta['n_artifact_rejected']})",
        f"Real anchor mode selected: {real_result.anchor_mode}",
        f"Real snippets: {real_result.n_snippets}",
        "",
        "Distances to real template:",
    ]
    for row in rows:
        lines.append(
            f"  {row['label']}: RMSE={row['template_rmse']:.4f}, "
            f"corr={row['template_corr']:.4f}, "
            f"1-corr={row['template_1_minus_corr']:.4f}, "
            f"snippets={row['n_snippets']}"
        )
    lines.extend(
        [
            "",
            f"Best by RMSE: {best_rmse['label']}",
            f"Best by correlation: {best_corr['label']}",
            "",
            "Note: This metric uses SO average-cycle waveform morphology only. "
            "It does not call compute_constraints_v7 or reuse T1-T12 fields.",
        ]
    )
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prep_mod = load_preprocess_module()
    v7 = load_v7_module()

    print("Loading real SC4001 N3 EEG epochs...")
    real_epochs, fs_real, real_meta = load_real_n3_epochs(prep_mod)
    print(
        f"  {real_meta['n_clean_n3']} clean N3 epochs "
        f"from {real_meta['channel']} at {fs_real:.1f} Hz"
    )

    print("Simulating version best points...")
    sim_results: list[TemplateResult] = []
    for label, params_path in VERSIONS:
        print(f"  {label}: {params_path}")
        r_ctx = simulate_version(v7, label, params_path)
        sim_result = build_template(label, [r_ctx], v7.FS_SIM, "peak")
        sim_results.append(sim_result)
        print(f"    snippets={sim_result.n_snippets}")

    print("Selecting real EEG polarity/anchor mode...")
    v7_template = next(result for result in sim_results if result.label.startswith("V7"))
    real_result = choose_real_anchor(real_epochs, fs_real, v7_template)
    print(
        f"  selected {real_result.anchor_mode} anchors, "
        f"snippets={real_result.n_snippets}"
    )

    rows = [compute_distances(real_result, sim) for sim in sim_results]
    rows = sorted(rows, key=lambda row: row["template_rmse"])

    write_metrics(rows, real_result, real_meta)
    plot_templates(real_result, sim_results)
    plot_distances(rows)
    write_summary(rows, real_result, real_meta)

    print("\nSaved:")
    print(f"  {METRICS_CSV}")
    print(f"  {FIG_TEMPLATES}")
    print(f"  {FIG_DISTANCES}")
    print(f"  {SUMMARY_TXT}")
    print("\nDistances:")
    for row in rows:
        print(
            f"  {row['label']}: RMSE={row['template_rmse']:.4f}, "
            f"corr={row['template_corr']:.4f}"
        )


if __name__ == "__main__":
    main()
