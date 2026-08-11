"""Figure-10-inspired validation panel for the sleep neural mass fitting project.

This is an audit/visualization script, not SBI posterior inference. It builds a
fitted-candidate ensemble from existing JSON/CSV archives, runs short V8a
simulations, computes PPC-like predictive diagnostics, and performs a cheap
archive-based synthetic parameter-recovery diagnostic.

Terminology is intentional:
- fitted-candidate ensemble / candidate archive, not posterior samples
- PPC-like predictive diagnostics, not calibrated posterior predictive checks
- synthetic parameter-recovery diagnostic, not SBC
- true SBI diagnostics require q_phi(theta | x)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
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
from scipy.interpolate import interp1d
from scipy.signal import butter, detrend, sosfiltfilt
from scipy.stats import pearsonr

mne.set_log_level("WARNING")

from models.s4_personalize_fig7_v8 import (  # noqa: E402
    BOUNDS,
    EXC,
    F_HI,
    F_LO,
    FS_SIM,
    HAS_FOOOF,
    PARAM_NAMES,
    SUBJECT_ID,
    build_model,
    compute_constraints_v8,
    compute_epoch_psd,
    compute_target_periodic,
    load_target_psd,
    seed_numba,
)

OUTDIR = ROOT / "outputs" / "figure10_inspired_validation_panel"
SIM_DUR_MS = 60_000
BURN_IN_S = 5.0
DEFAULT_SEEDS = [11, 22, 33]
TOP_K_PER_VERSION = 10

SIGMA_BAND = (11.0, 15.0)
RMS_WIN_S = 0.20
THRESH_K = 1.5
DUR_LO_S = 0.5
DUR_HI_S = 3.0
MERGE_GAP_S = 0.10
ARTIFACT_THRESH_UV = 200.0

SUMMARY_FEATURES = [
    "shape_r",
    "T4_q",
    "T6_ibi_cv",
    "T8_n_sp_events",
    "T12_n_verified",
    "T13_n_ctx_verified",
    "T13_ctx_verified_density_per_min",
    "heldout_rms_spindle_density",
    "T9_mi",
    "T10_phase",
    "T11_lag_ms",
]


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_csv_numeric(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() == df[col].notna().sum():
            df[col] = converted
    return df


def candidate_json_sources() -> list[tuple[str, Path]]:
    sources = []
    aliases = {
        "V1": [
            ROOT / "data" / "patient_params_fig7_v1_SC4001.json",
            ROOT / "data" / "patient_params_fig7_v1_0418_2_SC4001.json",
            ROOT / "data" / "patient_params_fig7_SC4001.json",
        ],
        "V2": [ROOT / "data" / "patient_params_fig7_v2_SC4001.json"],
        "V3": [ROOT / "data" / "patient_params_fig7_v3_SC4001.json"],
        "V4": [ROOT / "data" / "patient_params_fig7_v4_SC4001.json"],
        "V5": [ROOT / "data" / "patient_params_fig7_v5_SC4001.json"],
        "V6": [ROOT / "data" / "patient_params_fig7_v6_SC4001.json"],
        "V7": [ROOT / "data" / "patient_params_fig7_v7_SC4001.json"],
        "V8a": [ROOT / "data" / "patient_params_fig7_v8_SC4001.json"],
    }
    for version, paths in aliases.items():
        seen = set()
        for path in paths:
            if path.exists() and path not in seen:
                sources.append((version, path))
                seen.add(path)
    return sources


def records_sources() -> list[tuple[str, Path]]:
    out = []
    for version in range(1, 9):
        path = ROOT / "outputs" / f"evolution_fig7_v{version}_records.csv"
        if path.exists():
            out.append((f"V{version}" if version < 8 else "V8a", path))
    return out


def diagnostic_sources() -> list[tuple[str, Path]]:
    return [
        ("V8a-relaxed", ROOT / "outputs" / "v8a_relaxed_t6_sensitivity" / "relaxed_t6_best_candidates.csv"),
        ("V8a-coupling", ROOT / "outputs" / "v8a_t6_t13_coupling_sweep.csv"),
        ("V8a-ultra", ROOT / "outputs" / "v8a_ultra_narrow_t6_t13_search" / "ultra_narrow_records.csv"),
        ("V8a-local-long", ROOT / "outputs" / "v8a_local_de_t6_rescue_long" / "local_de_records.csv"),
        ("V8a-local-narrow", ROOT / "outputs" / "v8a_local_de_t6_rescue_narrow" / "narrow_local_de_records.csv"),
    ]


def row_has_params(row: pd.Series) -> bool:
    return all(name in row and pd.notna(row[name]) for name in PARAM_NAMES)


def normalize_candidate(row: dict, *, version: str, source: str, source_file: Path, source_row_index: int) -> dict:
    cand = {name: float(row.get(name, np.nan)) for name in PARAM_NAMES}
    cand.update(
        {
            "candidate_source": source,
            "version": version,
            "source_file": str(source_file.relative_to(ROOT)),
            "source_row_index": int(source_row_index),
            "score": float(row.get("score", np.nan)) if pd.notna(row.get("score", np.nan)) else np.nan,
            "n_passed": float(row.get("n_passed", np.nan)) if pd.notna(row.get("n_passed", np.nan)) else np.nan,
            "failed_constraints": str(row.get("failed_constraints", "")),
        }
    )
    for key in ["T4", "T6", "T8", "T12", "T13", "T4_q", "T6_ibi_cv", "T13_ctx_verified_density_per_min"]:
        if key in row and pd.notna(row[key]):
            cand[key] = row[key]
    return cand


def select_top_k_diverse(df: pd.DataFrame, k: int) -> pd.DataFrame:
    sort_cols = [col for col in ["n_passed", "score"] if col in df.columns]
    ascending = [False] * len(sort_cols)
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending, na_position="last")
    if "c_th2ctx" not in df.columns or len(df) <= k:
        return df.head(k)
    selected = []
    seen_bins = set()
    for idx, row in df.iterrows():
        c_val = row.get("c_th2ctx", np.nan)
        bin_id = int(np.floor(float(c_val) / 0.01)) if pd.notna(c_val) else None
        if bin_id not in seen_bins:
            selected.append(idx)
            seen_bins.add(bin_id)
        if len(selected) >= k:
            break
    if len(selected) < k:
        for idx in df.index:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= k:
                break
    return df.loc[selected]


def build_candidate_archive(max_candidates: int | None) -> tuple[pd.DataFrame, list[str], list[str]]:
    rows = []
    used, missing = [], []
    for version, path in candidate_json_sources():
        used.append(str(path.relative_to(ROOT)))
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if all(name in data for name in PARAM_NAMES):
            rows.append(normalize_candidate(data, version=version, source="best_json", source_file=path, source_row_index=0))

    expected_jsons = [ROOT / "data" / f"patient_params_fig7_v{i}_SC4001.json" for i in range(1, 9)]
    for path in expected_jsons:
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))

    for version, path in records_sources():
        used.append(str(path.relative_to(ROOT)))
        df = read_csv_numeric(path)
        if not set(PARAM_NAMES).issubset(df.columns):
            continue
        top = select_top_k_diverse(df, TOP_K_PER_VERSION)
        for idx, row in top.iterrows():
            rows.append(
                normalize_candidate(
                    row.to_dict(),
                    version=version,
                    source="evolution_records_topK",
                    source_file=path,
                    source_row_index=int(idx),
                )
            )

    for version, path in diagnostic_sources():
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue
        used.append(str(path.relative_to(ROOT)))
        df = read_csv_numeric(path)
        if not set(PARAM_NAMES).issubset(df.columns):
            continue
        top = select_top_k_diverse(df, TOP_K_PER_VERSION)
        for idx, row in top.iterrows():
            rows.append(
                normalize_candidate(
                    row.to_dict(),
                    version=version,
                    source="diagnostic_archive_topK",
                    source_file=path,
                    source_row_index=int(idx),
                )
            )

    archive = pd.DataFrame(rows)
    if archive.empty:
        raise RuntimeError("No candidates collected.")
    key = archive[PARAM_NAMES].round(12).astype(str).agg("|".join, axis=1)
    archive = archive.loc[~key.duplicated()].reset_index(drop=True)
    archive.insert(0, "candidate_id", [f"cand_{i:04d}" for i in range(len(archive))])
    if max_candidates is not None and len(archive) > max_candidates:
        best_json = archive[archive["candidate_source"].eq("best_json")].copy()
        remaining = archive[~archive["candidate_id"].isin(best_json["candidate_id"])].copy()
        n_remaining = max(0, max_candidates - len(best_json))
        archive = pd.concat(
            [best_json, select_top_k_diverse(remaining, n_remaining)],
            ignore_index=True,
        ).head(max_candidates)
        archive["candidate_id"] = [f"cand_{i:04d}" for i in range(len(archive))]
    return archive, used, sorted(set(missing))


def read_manifest() -> pd.DataFrame:
    path = ROOT / "data" / "manifest.csv"
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-16")


def pick_eeg_channel(raw) -> str:
    preferred = ["EEG Fpz-Cz", "Fpz-Cz", "EEG FPZ-CZ", "FPZ-CZ"]
    upper = {name.upper(): name for name in raw.ch_names}
    for channel in preferred:
        if channel.upper() in upper:
            return upper[channel.upper()]
    raise RuntimeError(f"No Fpz-Cz-like EEG channel found: {raw.ch_names}")


def load_real_n3() -> tuple[np.ndarray, float, dict]:
    prep_mod = import_module_from_path("fig10_prep", ROOT / "utils" / "02_preprocess_psd.py")
    manifest = read_manifest()
    row = manifest[manifest["subject_id"] == SUBJECT_ID].iloc[0]
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
        raise RuntimeError("No clean N3 EEG epochs.")
    meta = {"channel": channel, "fs": fs, "n_clean_n3": len(epochs), "n_artifact": n_artifact}
    return np.concatenate(epochs), fs, meta


def detect_rms_spindles(x: np.ndarray, fs: float):
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
    events = [(s, e) for s, e in merged if lo <= (e - s) <= hi]
    density = len(events) / (len(x) / fs / 60.0)
    return events, density


def compute_real_summaries() -> tuple[pd.DataFrame, dict]:
    real_signal, fs_real, meta = load_real_n3()
    target_psd, target_freqs = load_target_psd()
    events, density = detect_rms_spindles(real_signal, fs_real)
    real = {
        "subject_id": SUBJECT_ID,
        "channel": meta["channel"],
        "n_clean_n3_epochs": meta["n_clean_n3"],
        "psd_delta_mean": float(np.nanmean(target_psd[(target_freqs >= 0.5) & (target_freqs < 4.0)])),
        "psd_sigma_mean": float(np.nanmean(target_psd[(target_freqs >= 11.0) & (target_freqs <= 15.0)])),
        "heldout_rms_spindle_density": float(density),
        "heldout_rms_spindle_events": int(len(events)),
        "T4_q": np.nan,
        "T6_ibi_cv": np.nan,
        "T9_mi": np.nan,
        "T10_phase": np.nan,
        "T11_lag_ms": np.nan,
    }
    psd_df = pd.DataFrame({"freq": target_freqs, "psd": target_psd})
    return pd.DataFrame([real]), {"psd": psd_df, "signal_meta": meta}


def simulate_candidate(params: dict, seed: int):
    model = build_model(
        params["mue"],
        params["mui"],
        params["b"],
        params["tauA"],
        params["g_LK"],
        params["g_h"],
        params["c_th2ctx"],
        params["c_ctx2th"],
        duration=SIM_DUR_MS,
    )
    try:
        seed_numba(seed)
        model.run()
    except Exception:
        model.params["backend"] = "jitcdde"
        seed_numba(seed)
        model.run()
    r_exc = model[f"r_mean_{EXC}"]
    if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
        r_ctx = r_exc[0, :] * 1000.0
        r_thal = r_exc[1, :] * 1000.0
    else:
        r_ctx = np.asarray(r_exc).reshape(-1) * 1000.0
        r_thal = np.zeros_like(r_ctx)
    n_drop = int(BURN_IN_S * FS_SIM)
    return np.asarray(r_ctx[n_drop:], dtype=float), np.asarray(r_thal[n_drop:], dtype=float)


def compute_shape_r(f_ctx, p_ctx, target_periodic, fooof_freqs) -> float:
    if not HAS_FOOOF or target_periodic is None or fooof_freqs is None:
        return np.nan
    try:
        from fooof import FOOOF

        p_interp = interp1d(f_ctx, p_ctx, bounds_error=False, fill_value=1e-30)(fooof_freqs)
        fm = FOOOF(peak_width_limits=[1.0, 8.0], max_n_peaks=4, min_peak_height=0.05, aperiodic_mode="fixed")
        fm.fit(fooof_freqs, p_interp, [F_LO, F_HI])
        sim_log = np.log10(p_interp[: len(fm._ap_fit)] + 1e-30)
        sim_periodic = sim_log - fm._ap_fit
        n = min(len(sim_periodic), len(target_periodic))
        r, _ = pearsonr(sim_periodic[:n], target_periodic[:n])
        return float(r)
    except Exception:
        return np.nan


def failed_constraints(details: dict) -> str:
    return ",".join([f"T{i}" for i in range(1, 14) if not bool(details.get(f"T{i}", False))])


def run_predictive_summaries(archive: pd.DataFrame, n_reps: int) -> pd.DataFrame:
    target_psd, target_freqs = load_target_psd()
    target_periodic, fooof_freqs = compute_target_periodic(target_psd, target_freqs)
    seeds = DEFAULT_SEEDS[:n_reps]
    if n_reps > len(seeds):
        seeds = DEFAULT_SEEDS + list(range(100, 100 + n_reps - len(DEFAULT_SEEDS)))
    rows = []
    for _, cand in archive.iterrows():
        params = {name: float(cand[name]) for name in PARAM_NAMES}
        for rep_idx, seed in enumerate(seeds):
            print(f"Simulating {cand['candidate_id']} rep {rep_idx + 1}/{len(seeds)} seed={seed}", flush=True)
            try:
                r_ctx, r_thal = simulate_candidate(params, seed)
                f_ctx, p_ctx = compute_epoch_psd(r_ctx, FS_SIM)
                n_passed, details = compute_constraints_v8(r_ctx, r_thal, f_c=f_ctx, p_c=p_ctx, fs=FS_SIM)
                _events, rms_density = detect_rms_spindles(detrend(r_ctx, type="constant"), FS_SIM)
                row = {
                    "candidate_id": cand["candidate_id"],
                    "replicate": rep_idx,
                    "seed": seed,
                    "version": cand["version"],
                    "candidate_source": cand["candidate_source"],
                    "source_file": cand["source_file"],
                    "source_row_index": cand["source_row_index"],
                    **{name: params[name] for name in PARAM_NAMES},
                    "simulation_ok": 1,
                    "shape_r": compute_shape_r(f_ctx, p_ctx, target_periodic, fooof_freqs),
                    "n_passed": int(n_passed),
                    "failed_constraints": failed_constraints(details),
                    "heldout_rms_spindle_density": float(rms_density),
                }
                for key in [
                    "T4_q",
                    "T6_ibi_cv",
                    "T8_n_sp_events",
                    "T12_n_verified",
                    "T13_n_ctx_verified",
                    "T13_ctx_verified_density_per_min",
                    "T9_mi",
                    "T10_phase",
                    "T11_lag_ms",
                ]:
                    row[key] = details.get(key, np.nan)
                for t in range(1, 14):
                    row[f"T{t}"] = int(bool(details.get(f"T{t}", False)))
            except Exception as exc:
                row = {
                    "candidate_id": cand["candidate_id"],
                    "replicate": rep_idx,
                    "seed": seed,
                    "version": cand["version"],
                    "candidate_source": cand["candidate_source"],
                    "source_file": cand["source_file"],
                    "source_row_index": cand["source_row_index"],
                    **{name: params[name] for name in PARAM_NAMES},
                    "simulation_ok": 0,
                    "error": repr(exc),
                }
            rows.append(row)
    return pd.DataFrame(rows)


def archive_feature_table(summaries: pd.DataFrame) -> pd.DataFrame:
    ok = summaries[summaries["simulation_ok"] == 1].copy()
    agg_cols = [col for col in SUMMARY_FEATURES if col in ok.columns]
    base_cols = ["candidate_id", "version", "candidate_source", "source_file", "source_row_index"] + PARAM_NAMES
    return ok.groupby(base_cols, dropna=False)[agg_cols].mean().reset_index()


def normalized_distance_matrix(features: pd.DataFrame, query: pd.Series, feature_cols: list[str]) -> pd.Series:
    vals = features[feature_cols].astype(float)
    q = query[feature_cols].astype(float)
    means = vals.mean(skipna=True)
    scales = vals.std(skipna=True).replace(0, np.nan)
    z = (vals - means) / scales
    qz = (q - means) / scales
    diff = z.subtract(qz, axis=1)
    distances = np.sqrt(np.nanmean(diff.to_numpy() ** 2, axis=1))
    return pd.Series(distances, index=features.index)


def pick_synthetic_truths(archive: pd.DataFrame) -> pd.DataFrame:
    picks = []
    def add_first(mask):
        sub = archive[mask]
        if not sub.empty:
            picks.append(sub.iloc[0])

    add_first(archive["version"].astype(str).str.startswith("V1"))
    add_first(archive["version"].astype(str).str.startswith("V7"))
    add_first(archive["version"].astype(str).str.startswith("V8"))
    add_first(archive["source_file"].astype(str).str.contains("relaxed_t6", case=False, na=False))
    if "c_th2ctx" in archive:
        picks.append(archive.sort_values("c_th2ctx", ascending=False).iloc[0])
    if not picks:
        return archive.head(0)
    out = pd.DataFrame(picks)
    key = out[PARAM_NAMES].round(12).astype(str).agg("|".join, axis=1)
    return out.loc[~key.duplicated()].head(5).reset_index(drop=True)


def run_synthetic_recovery(archive: pd.DataFrame, feature_table: pd.DataFrame) -> pd.DataFrame:
    truths = pick_synthetic_truths(archive)
    rows = []
    feature_cols = [col for col in SUMMARY_FEATURES if col in feature_table.columns and feature_table[col].notna().any()]
    if truths.empty or not feature_cols:
        return pd.DataFrame()
    bounds = dict(zip(PARAM_NAMES, BOUNDS))
    for i, truth in truths.iterrows():
        params = {name: float(truth[name]) for name in PARAM_NAMES}
        try:
            r_ctx, r_thal = simulate_candidate(params, seed=501 + i)
            f_ctx, p_ctx = compute_epoch_psd(r_ctx, FS_SIM)
            n_passed, details = compute_constraints_v8(r_ctx, r_thal, f_c=f_ctx, p_c=p_ctx, fs=FS_SIM)
            _events, rms_density = detect_rms_spindles(detrend(r_ctx, type="constant"), FS_SIM)
            query = pd.Series({col: np.nan for col in feature_cols})
            target_psd, target_freqs = load_target_psd()
            target_periodic, fooof_freqs = compute_target_periodic(target_psd, target_freqs)
            query["shape_r"] = compute_shape_r(f_ctx, p_ctx, target_periodic, fooof_freqs)
            query["heldout_rms_spindle_density"] = float(rms_density)
            for key in feature_cols:
                if key in details:
                    query[key] = details[key]
            distances = normalized_distance_matrix(feature_table, query, feature_cols)
            best_idx = int(distances.idxmin())
            recovered = feature_table.loc[best_idx]
            row = {
                "synthetic_id": f"synth_{i:02d}",
                "theta_true_candidate_id": truth["candidate_id"],
                "theta_true_version": truth["version"],
                "theta_recovered_candidate_id": recovered["candidate_id"],
                "theta_recovered_version": recovered["version"],
                "same_version": int(str(truth["version"]) == str(recovered["version"])),
                "summary_distance": float(distances.loc[best_idx]),
                "synthetic_n_passed": int(n_passed),
            }
            rel_errors = []
            for name in PARAM_NAMES:
                true_val = float(truth[name])
                rec_val = float(recovered[name])
                lo, hi = bounds[name]
                abs_err = abs(rec_val - true_val)
                rel_err = abs_err / (hi - lo) if hi > lo else np.nan
                row[f"{name}_true"] = true_val
                row[f"{name}_recovered"] = rec_val
                row[f"{name}_abs_error"] = abs_err
                row[f"{name}_rel_error"] = rel_err
                rel_errors.append(rel_err)
            row["mean_relative_error"] = float(np.nanmean(rel_errors))
        except Exception as exc:
            row = {
                "synthetic_id": f"synth_{i:02d}",
                "theta_true_candidate_id": truth["candidate_id"],
                "theta_true_version": truth["version"],
                "error": repr(exc),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def scatter_versions(ax, archive: pd.DataFrame):
    versions = sorted(archive["version"].astype(str).unique())
    cmap = plt.get_cmap("tab10")
    for i, version in enumerate(versions):
        sub = archive[archive["version"].astype(str) == version]
        size = 35
        if "n_passed" in sub.columns:
            size = 20 + 8 * pd.to_numeric(sub["n_passed"], errors="coerce").fillna(3)
        ax.scatter(
            sub["c_th2ctx"],
            sub["c_ctx2th"],
            s=size,
            alpha=0.70,
            color=cmap(i % 10),
            label=version,
            edgecolors="none",
        )
    for label, mask in [
        ("V7 best", archive["version"].astype(str).str.startswith("V7") & archive["candidate_source"].eq("best_json")),
        ("V8a best", archive["version"].astype(str).str.startswith("V8") & archive["candidate_source"].eq("best_json")),
        ("best0417", archive["source_file"].astype(str).str.contains("relaxed_t6", case=False, na=False)),
    ]:
        sub = archive[mask]
        if not sub.empty:
            row = sub.iloc[0]
            ax.scatter([row["c_th2ctx"]], [row["c_ctx2th"]], marker="*", s=180, color="black", zorder=10)
            ax.text(row["c_th2ctx"], row["c_ctx2th"], f" {label}", fontsize=7, va="center")
    ax.set_xlabel("c_th2ctx")
    ax.set_ylabel("c_ctx2th")
    ax.set_title("C. fitted-candidate ensemble")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    ax.grid(True, alpha=0.2)


def make_figure(archive: pd.DataFrame, summaries: pd.DataFrame, real: pd.DataFrame, recovery: pd.DataFrame):
    fig = plt.figure(figsize=(16, 12), dpi=150)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.25, 1.0], hspace=0.42, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    ax_a.axis("off")
    text_a = (
        "A. 8-parameter cortical-thalamic neural mass model\n\n"
        "Parameters:\n"
        "mue, mui, b, tauA\n"
        "g_LK, g_h\n"
        "c_th2ctx, c_ctx2th\n\n"
        "Validation object:\n"
        "fitted-candidate ensemble / candidate archive\n"
        "not posterior samples"
    )
    ax_a.text(0.02, 0.98, text_a, va="top", fontsize=11, family="monospace")

    ax_b.set_title("B. real SC4001 N3 EEG summaries")
    real_row = real.iloc[0]
    labels = ["delta PSD", "sigma PSD", "RMS density"]
    values = [real_row["psd_delta_mean"], real_row["psd_sigma_mean"], real_row["heldout_rms_spindle_density"]]
    ax_b.bar(labels, values, color=["#4e79a7", "#f28e2b", "#59a14f"])
    ax_b.set_ylabel("summary value")
    ax_b.text(
        0.02,
        0.95,
        f"channel={real_row['channel']}\nN3 epochs={int(real_row['n_clean_n3_epochs'])}\nPAC/SO morphology: NaN if utility unavailable",
        transform=ax_b.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"),
    )
    ax_b.grid(True, axis="y", alpha=0.2)

    scatter_versions(ax_c, archive)

    ax_d.set_title("D. PPC-like predictive diagnostics over fitted-candidate ensemble")
    ok = summaries[summaries["simulation_ok"] == 1].copy()
    metrics = ["T6_ibi_cv", "T13_ctx_verified_density_per_min", "heldout_rms_spindle_density", "T4_q"]
    data = [pd.to_numeric(ok[m], errors="coerce").dropna().values for m in metrics]
    ax_d.boxplot(data, labels=["T6 IBI-CV", "T13 dens", "RMS dens", "T4_q"], showfliers=False)
    for i, vals in enumerate(data, start=1):
        if len(vals):
            rng = np.random.default_rng(123 + i)
            x = i + rng.uniform(-0.10, 0.10, size=len(vals))
            ax_d.scatter(x, vals, s=8, alpha=0.25, color="black", linewidths=0)
    ax_d.axhline(0.40, color="#d62728", linestyle="--", linewidth=1.0, label="strict T6 threshold")
    ax_d.axhline(float(real_row["heldout_rms_spindle_density"]), color="#59a14f", linestyle=":", linewidth=1.0,
                 label="real RMS density")
    ax_d.legend(fontsize=7, frameon=False)
    ax_d.grid(True, axis="y", alpha=0.2)

    ax_e.set_title("E. synthetic parameter-recovery diagnostic")
    if recovery.empty or "mean_relative_error" not in recovery:
        ax_e.text(0.05, 0.8, "Recovery diagnostic unavailable", fontsize=11)
    else:
        rec_ok = recovery[pd.to_numeric(recovery.get("mean_relative_error", np.nan), errors="coerce").notna()]
        ax_e.bar(rec_ok["synthetic_id"], rec_ok["mean_relative_error"], color="#9c755f")
        ax_e.set_ylabel("mean relative parameter error")
        ax_e.set_xlabel("synthetic theta")
        ax_e.text(0.02, 0.95, "archive-based nearest-neighbor recovery\nSBC-inspired, not SBC",
                  transform=ax_e.transAxes, va="top", fontsize=8,
                  bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"))
    ax_e.grid(True, axis="y", alpha=0.2)

    ax_f.axis("off")
    text_f = (
        "F. future SBI posterior diagnostics\n\n"
        "Not performed here.\n\n"
        "Full posterior diagnostics require\n"
        "q_phi(theta | x) from an NPE/NLE/NRE\n"
        "posterior estimator.\n\n"
        "Future SBI extension:\n"
        "NPE/NLE/NRE posterior estimator ->\n"
        "posterior predictive checks ->\n"
        "expected coverage / SBC / TARP ->\n"
        "local diagnostics such as L-C2ST.\n\n"
        "Current figure: PPC-like diagnostics\n"
        "over a fitted-candidate ensemble."
    )
    ax_f.text(0.02, 0.98, text_f, va="top", fontsize=11, family="monospace",
              bbox=dict(facecolor="#f7f7f7", edgecolor="0.75", boxstyle="round,pad=0.5"))

    fig.suptitle("Figure-10-inspired validation panel: sleep neural mass fitted-candidate ensemble", fontsize=15)
    png = OUTDIR / "figure10_inspired_validation_panel.png"
    pdf = OUTDIR / "figure10_inspired_validation_panel.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def write_summary(archive, summaries, real, recovery, used, missing, png, pdf, elapsed):
    ok_sims = int((summaries.get("simulation_ok", pd.Series(dtype=int)) == 1).sum()) if not summaries.empty else 0
    versions = sorted(archive["version"].astype(str).unique())
    lines = [
        "# Figure-10-inspired Validation Panel Summary",
        "",
        "This figure is inspired by SBI Figure 10, but no posterior estimator is available.",
        "Candidate rows are a fitted-candidate ensemble / candidate archive, not posterior samples.",
        "",
        "## Files Used",
        "",
    ]
    lines.extend([f"- `{p}`" for p in used])
    lines.extend(["", "## Missing / Skipped Files", ""])
    lines.extend([f"- `{p}`" for p in missing] if missing else ["- none"])
    lines.extend(
        [
            "",
            "## Run Summary",
            "",
            f"- candidates collected after dedup/truncation: `{len(archive)}`",
            f"- versions represented: `{', '.join(versions)}`",
            f"- simulations completed: `{ok_sims}`",
            f"- wall time seconds: `{elapsed:.1f}`",
            f"- real observation summaries computed: `{not real.empty}`",
            f"- PNG: `{png.relative_to(ROOT)}`",
            f"- PDF: `{pdf.relative_to(ROOT)}`",
            "",
            "## Wording Guardrails",
            "",
            "- PPC-like predictive diagnostics, not calibrated posterior predictive checks.",
            "- Synthetic parameter-recovery diagnostic, not SBC.",
            "- Expected coverage / SBC / TARP / L-C2ST are not performed here.",
            "- Future SBI diagnostics require q_phi(theta|x).",
        ]
    )
    if not recovery.empty:
        lines.extend(["", "## Synthetic Recovery", ""])
        cols = [c for c in ["synthetic_id", "theta_true_version", "theta_recovered_version", "same_version", "summary_distance", "mean_relative_error"] if c in recovery.columns]
        lines.append(recovery[cols].to_markdown(index=False))
    (OUTDIR / "figure10_inspired_validation_panel_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--n-reps", type=int, default=3)
    args = parser.parse_args()

    start = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    archive, used, missing = build_candidate_archive(args.max_candidates)
    archive_path = OUTDIR / "candidate_archive.csv"
    archive.to_csv(archive_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Candidate archive: {len(archive)} rows -> {archive_path.relative_to(ROOT)}")

    real, _real_extra = compute_real_summaries()
    real_path = OUTDIR / "real_observation_summaries.csv"
    real.to_csv(real_path, index=False)
    print(f"Real observation summaries -> {real_path.relative_to(ROOT)}")

    summaries_path = OUTDIR / "candidate_predictive_summaries.csv"
    reuse_summaries = False
    if summaries_path.exists():
        try:
            existing = pd.read_csv(summaries_path)
            expected_rows = len(archive) * args.n_reps
            same_ids = set(existing.get("candidate_id", [])) == set(archive["candidate_id"])
            reuse_summaries = len(existing) == expected_rows and same_ids
        except Exception:
            reuse_summaries = False
    if reuse_summaries:
        summaries = pd.read_csv(summaries_path)
        print(f"Reusing predictive summaries -> {summaries_path.relative_to(ROOT)}")
    else:
        summaries = run_predictive_summaries(archive, args.n_reps)
        summaries.to_csv(summaries_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Predictive summaries -> {summaries_path.relative_to(ROOT)}")

    feature_table = archive_feature_table(summaries)
    recovery = run_synthetic_recovery(archive, feature_table)
    recovery_path = OUTDIR / "synthetic_recovery_results.csv"
    recovery.to_csv(recovery_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Synthetic recovery -> {recovery_path.relative_to(ROOT)}")

    png, pdf = make_figure(archive, summaries, real, recovery)
    elapsed = time.time() - start
    write_summary(archive, summaries, real, recovery, used, missing, png, pdf, elapsed)

    print("\nFigure-10-inspired validation panel complete")
    print(f"Candidates collected: {len(archive)}")
    print(f"Versions represented: {', '.join(sorted(archive['version'].astype(str).unique()))}")
    print(f"Simulations completed: {(summaries['simulation_ok'] == 1).sum() if 'simulation_ok' in summaries else 0}")
    print(f"Real summaries computed: {not real.empty}")
    print(f"PNG: {png.relative_to(ROOT)}")
    print(f"PDF: {pdf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
