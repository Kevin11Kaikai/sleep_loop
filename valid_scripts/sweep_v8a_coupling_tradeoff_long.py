"""Long targeted V8a coupling trade-off diagnostic.

This script is intentionally not an optimizer. It imports the V8a model
builder and constraint function, then sweeps selected coupling coordinates
around candidates found by the completed V8a run.

It writes checkpoints continuously and can resume after interruption.
It does not modify the main V8a script and does not run 300 s validation.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.s4_personalize_fig7_v8 import (  # noqa: E402
    FS_SIM,
    IBI_CV_MAX,
    SO_Q_MIN,
    build_model,
    compute_constraints_v8,
    compute_epoch_psd,
)
from neurolib.models.multimodel.builder.base.constants import EXC  # noqa: E402


RECORDS_PATH = ROOT / "outputs" / "evolution_fig7_v8_records.csv"
BEST_PATH = ROOT / "data" / "patient_params_fig7_v8_SC4001.json"
NEAR12_PATH = ROOT / "outputs" / "v8a_diagnostics" / "v8a_12of13_candidates.csv"
OUTDIR = ROOT / "outputs" / "v8a_coupling_sweep_long"
LOG_PATH = OUTDIR / "run_log.txt"
SEEDS_PATH = OUTDIR / "selected_seed_candidates.csv"
PHASE1_PATH = OUTDIR / "phase1_cth2ctx_sweep.csv"
PHASE2_PATH = OUTDIR / "phase2_coupling_2d_sweep.csv"
PHASE3_PATH = OUTDIR / "phase3_local_physiological_sweep.csv"
PHASE4_PATH = OUTDIR / "phase4_120s_stability_check.csv"
SUMMARY_PATH = OUTDIR / "v8a_coupling_sweep_long_summary.txt"

T_COLS = [f"T{i}" for i in range(1, 14)]
PARAM_COLS = ["mue", "mui", "b", "tauA", "g_LK", "g_h", "c_th2ctx", "c_ctx2th"]
REQUIRED_COLUMNS = PARAM_COLS + [
    "score",
    "n_passed",
    "feasible",
    *T_COLS,
    "T4_q",
    "T6_ibi_cv",
    "T8_n_sp_events",
    "T12_n_verified",
    "T13_n_ctx_events",
    "T13_n_ctx_verified",
    "T13_ctx_verified_density_per_min",
]
RESULT_FIELDS = [
    "phase",
    "seed_id",
    "seed_group",
    "seed_score",
    "seed_n_passed",
    "seed_failed_constraints",
    "source_eval",
    *PARAM_COLS,
    "duration_ms",
    "n_passed",
    "feasible",
    *T_COLS,
    "T4_q",
    "T6_ibi_cv",
    "T8_n_sp_events",
    "T12_n_verified",
    "T13_n_ctx_events",
    "T13_n_ctx_verified",
    "T13_ctx_density_per_min",
    "T13_ctx_verified_density_per_min",
    "T13_mean_ctx_dur",
    "error",
    "elapsed_s",
]
MAX_TOTAL_SIMS = 2500
DEFAULT_DURATION_MS = 60_000
LONG_DURATION_MS = 120_000
DROP_S = 5.0


def log(message: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def read_csv_numeric(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() == df[col].notna().sum():
            df[col] = converted
    return df


def failed_constraints(row: pd.Series) -> str:
    failed = [t for t in T_COLS if int(row[t]) == 0]
    return ",".join(failed) if failed else "none"


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def dedupe_params(df: pd.DataFrame) -> pd.DataFrame:
    key = df[PARAM_COLS].round(12).astype(str).agg("|".join, axis=1)
    return df.loc[~key.duplicated()].copy()


def choose_seeds(records: pd.DataFrame) -> pd.DataFrame:
    records = records.copy()
    records["failed_constraints"] = records.apply(failed_constraints, axis=1)
    seeds = []

    a = records[
        (records["n_passed"] == 12)
        & (records["failed_constraints"] == "T6")
        & (records["T13"] == 1)
    ].sort_values("score", ascending=False)
    if not a.empty:
        tmp = a.head(1).copy()
        tmp["seed_group"] = "A_best_T6_only_12of13"
        seeds.append(tmp)

    b = records[
        (records["n_passed"] == 12)
        & (records["failed_constraints"] == "T13")
        & (records["T4"] == 1)
        & (records["T6"] == 1)
    ].sort_values("score", ascending=False)
    if not b.empty:
        tmp = b.head(10).copy()
        tmp["seed_group"] = "B_top10_T13_only_12of13"
        seeds.append(tmp)

    c = records[records["T13"] == 1].sort_values(
        ["n_passed", "score"], ascending=[False, False]
    )
    if not c.empty:
        tmp = c.head(10).copy()
        tmp["seed_group"] = "C_top10_T13_positive"
        seeds.append(tmp)

    d = records[(records["T4"] == 1) & (records["T6"] == 1)].sort_values(
        "score", ascending=False
    )
    if not d.empty:
        tmp = d.head(10).copy()
        tmp["seed_group"] = "D_top10_SO_stable"
        seeds.append(tmp)

    if not seeds:
        raise RuntimeError("No seed candidates selected.")

    out = pd.concat(seeds, ignore_index=True)
    out = dedupe_params(out)
    out = out.sort_values(["n_passed", "score"], ascending=[False, False]).reset_index(drop=True)
    out.insert(0, "seed_id", [f"seed_{i:03d}" for i in range(len(out))])
    out["seed_failed_constraints"] = out.apply(failed_constraints, axis=1)
    out = out.rename(columns={"score": "seed_score", "n_passed": "seed_n_passed"})
    return out


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULT_FIELDS})


def read_done_keys(path: Path, key_cols: list[str]) -> set[tuple]:
    if not path.exists():
        return set()
    df = read_csv_numeric(path)
    if df.empty:
        return set()
    return {tuple(row[col] for col in key_cols) for _, row in df.iterrows()}


def total_completed_sims() -> int:
    total = 0
    for path in (PHASE1_PATH, PHASE2_PATH, PHASE3_PATH, PHASE4_PATH):
        if path.exists():
            try:
                total += len(pd.read_csv(path))
            except Exception:
                pass
    return total


def extract_series(model) -> tuple[np.ndarray, np.ndarray]:
    r_exc = model[f"r_mean_{EXC}"]
    if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
        r_ctx = r_exc[0, :] * 1000.0
        r_thal = r_exc[1, :] * 1000.0
    else:
        r_ctx = np.asarray(r_exc).squeeze() * 1000.0
        r_thal = np.zeros_like(r_ctx)
    n_drop = int(DROP_S * FS_SIM)
    return r_ctx[n_drop:], r_thal[n_drop:]


def simulate_constraints(params: dict, duration_ms: int = DEFAULT_DURATION_MS) -> dict:
    start = time.time()
    try:
        model = build_model(
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
        model.run()
        r_ctx, r_thal = extract_series(model)
        f_ctx, p_ctx = compute_epoch_psd(r_ctx, FS_SIM)
        n_passed, details = compute_constraints_v8(
            r_ctx, r_thal, f_c=f_ctx, p_c=p_ctx, fs=FS_SIM
        )
        row = {
            "duration_ms": duration_ms,
            "n_passed": int(n_passed),
            "feasible": int(bool(details.get("feasible", n_passed == 13))),
            "error": "",
            "elapsed_s": round(time.time() - start, 3),
        }
        for t in T_COLS:
            row[t] = int(bool(details.get(t, False)))
        for key in [
            "T4_q",
            "T6_ibi_cv",
            "T8_n_sp_events",
            "T12_n_verified",
            "T13_n_ctx_events",
            "T13_n_ctx_verified",
            "T13_ctx_density_per_min",
            "T13_ctx_verified_density_per_min",
            "T13_mean_ctx_dur",
        ]:
            val = details.get(key, np.nan)
            try:
                row[key] = float(val)
            except Exception:
                row[key] = val
        return row
    except Exception as exc:
        return {
            "duration_ms": duration_ms,
            "n_passed": 0,
            "feasible": 0,
            **{t: 0 for t in T_COLS},
            "T4_q": np.nan,
            "T6_ibi_cv": np.nan,
            "T8_n_sp_events": np.nan,
            "T12_n_verified": np.nan,
            "T13_n_ctx_events": np.nan,
            "T13_n_ctx_verified": np.nan,
            "T13_ctx_density_per_min": np.nan,
            "T13_ctx_verified_density_per_min": np.nan,
            "T13_mean_ctx_dur": np.nan,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.time() - start, 3),
        }


def base_seed_row(seed: pd.Series, phase: str) -> dict:
    return {
        "phase": phase,
        "seed_id": seed["seed_id"],
        "seed_group": seed["seed_group"],
        "seed_score": seed["seed_score"],
        "seed_n_passed": seed["seed_n_passed"],
        "seed_failed_constraints": seed["seed_failed_constraints"],
        "source_eval": seed.get("eval", ""),
        **{p: float(seed[p]) for p in PARAM_COLS},
    }


def run_and_append(path: Path, row_base: dict, params: dict, duration_ms: int) -> None:
    sim = simulate_constraints(params, duration_ms=duration_ms)
    out = {**row_base, **{p: float(params[p]) for p in PARAM_COLS}, **sim}
    append_row(path, out)


def phase1(seeds: pd.DataFrame) -> None:
    done = read_done_keys(PHASE1_PATH, ["seed_id", "c_th2ctx"])
    for _, seed in seeds.iterrows():
        seed_c = float(seed["c_th2ctx"])
        global_grid = np.linspace(0.00, 0.10, 41)
        local_grid = np.linspace(max(0.0, seed_c - 0.025), min(0.10, seed_c + 0.035), 31)
        grid = sorted({round(float(v), 10) for v in np.concatenate([global_grid, local_grid])})
        log(f"Phase 1 seed={seed['seed_id']} group={seed['seed_group']} grid_n={len(grid)}")
        for c_th2ctx in grid:
            if total_completed_sims() >= MAX_TOTAL_SIMS:
                log("Simulation budget reached during Phase 1; stopping gracefully.")
                return
            key = (seed["seed_id"], c_th2ctx)
            if key in done:
                continue
            params = {p: float(seed[p]) for p in PARAM_COLS}
            params["c_th2ctx"] = c_th2ctx
            row = base_seed_row(seed, "phase1_cth2ctx")
            row["c_th2ctx"] = c_th2ctx
            run_and_append(PHASE1_PATH, row, params, DEFAULT_DURATION_MS)
            done.add(key)


def score_distance(row: pd.Series) -> float:
    t4_penalty = 0.0 if int(row["T4"]) == 1 else max(0.0, SO_Q_MIN - float(row["T4_q"]))
    t6_penalty = max(0.0, float(row["T6_ibi_cv"]) - IBI_CV_MAX)
    t13_penalty = 0.0 if int(row["T13"]) == 1 else 1.0
    return t4_penalty + t6_penalty + t13_penalty


def select_phase2_bases(seeds: pd.DataFrame) -> pd.DataFrame:
    phase1_df = read_csv_numeric(PHASE1_PATH) if PHASE1_PATH.exists() else pd.DataFrame()
    rows = []
    if not phase1_df.empty:
        phase1_df["distance"] = phase1_df.apply(score_distance, axis=1)
        rows.append(phase1_df.sort_values(["n_passed"], ascending=False).head(1))
        t13 = phase1_df[phase1_df["T13"] == 1]
        if not t13.empty:
            rows.append(t13.sort_values(["T6_ibi_cv", "n_passed"], ascending=[True, False]).head(1))
        stable = phase1_df[(phase1_df["T4"] == 1) & (phase1_df["T6"] == 1)]
        if not stable.empty:
            rows.append(
                stable.sort_values(
                    ["T13_ctx_verified_density_per_min", "n_passed"],
                    ascending=[False, False],
                ).head(1)
            )
        rows.append(phase1_df.sort_values(["distance", "n_passed"], ascending=[True, False]).head(1))

    a = seeds[seeds["seed_group"] == "A_best_T6_only_12of13"].head(1)
    b = seeds[seeds["seed_group"] == "B_top10_T13_only_12of13"].head(1)
    if not a.empty:
        rows.append(a.rename(columns={"seed_n_passed": "n_passed"}))
    if not b.empty:
        rows.append(b.rename(columns={"seed_n_passed": "n_passed"}))

    if not rows:
        return pd.DataFrame()
    bases = pd.concat(rows, ignore_index=True, sort=False)
    for col in ["seed_id", "seed_group"]:
        if col not in bases:
            bases[col] = ""
    bases = dedupe_params(bases)
    bases = bases.head(6).reset_index(drop=True)
    bases["phase2_base_id"] = [f"p2base_{i:02d}" for i in range(len(bases))]
    return bases


def normalize_base_row(row: pd.Series) -> pd.Series:
    out = row.copy()
    if "seed_score" not in out or pd.isna(out.get("seed_score", np.nan)):
        out["seed_score"] = out.get("score", "")
    if "seed_n_passed" not in out or pd.isna(out.get("seed_n_passed", np.nan)):
        out["seed_n_passed"] = out.get("n_passed", "")
    if "seed_failed_constraints" not in out or pd.isna(out.get("seed_failed_constraints", np.nan)):
        out["seed_failed_constraints"] = failed_constraints(out)
    if "seed_id" not in out or pd.isna(out.get("seed_id", np.nan)):
        out["seed_id"] = out.get("phase2_base_id", "phase2_base")
    if "seed_group" not in out or pd.isna(out.get("seed_group", np.nan)):
        out["seed_group"] = "phase2_selected"
    return out


def phase2(seeds: pd.DataFrame) -> None:
    bases = select_phase2_bases(seeds)
    if bases.empty:
        log("Phase 2 skipped: no bases available.")
        return
    bases.to_csv(OUTDIR / "phase2_selected_bases.csv", index=False)
    done = read_done_keys(PHASE2_PATH, ["seed_id", "c_th2ctx", "c_ctx2th"])
    for _, raw_base in bases.iterrows():
        base = normalize_base_row(raw_base)
        c0 = float(base["c_th2ctx"])
        k0 = float(base["c_ctx2th"])
        c_grid = np.linspace(max(0.0, c0 - 0.025), min(0.10, c0 + 0.045), 19)
        k_grid = np.linspace(max(0.05, k0 - 0.05), min(0.22, k0 + 0.05), 15)
        log(
            f"Phase 2 base={base['seed_id']} c_grid={len(c_grid)} "
            f"ctx2th_grid={len(k_grid)}"
        )
        for c_th2ctx in [round(float(v), 10) for v in c_grid]:
            for c_ctx2th in [round(float(v), 10) for v in k_grid]:
                if total_completed_sims() >= MAX_TOTAL_SIMS:
                    log("Simulation budget reached during Phase 2; stopping gracefully.")
                    return
                key = (base["seed_id"], c_th2ctx, c_ctx2th)
                if key in done:
                    continue
                params = {p: float(base[p]) for p in PARAM_COLS}
                params["c_th2ctx"] = c_th2ctx
                params["c_ctx2th"] = c_ctx2th
                row = base_seed_row(base, "phase2_coupling_2d")
                row["c_th2ctx"] = c_th2ctx
                row["c_ctx2th"] = c_ctx2th
                run_and_append(PHASE2_PATH, row, params, DEFAULT_DURATION_MS)
                done.add(key)


def collect_all_results() -> pd.DataFrame:
    frames = []
    for path in (PHASE1_PATH, PHASE2_PATH, PHASE3_PATH, PHASE4_PATH):
        if path.exists():
            frames.append(read_csv_numeric(path))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def choose_phase3_base() -> pd.Series | None:
    df = collect_all_results()
    if df.empty:
        return None
    candidates = df[(df["T4"] == 1) & (df["T13"] == 1) & (df["T6_ibi_cv"] < 0.60)]
    if candidates.empty:
        return None
    candidates = candidates.copy()
    candidates["distance"] = candidates.apply(score_distance, axis=1)
    return candidates.sort_values(["n_passed", "distance"], ascending=[False, True]).iloc[0]


def phase3() -> None:
    base = choose_phase3_base()
    if base is None:
        log("Phase 3 skipped: no T4=1, T13=1, T6_ibi_cv<0.60 candidate.")
        return
    done = read_done_keys(PHASE3_PATH, ["c_th2ctx", "c_ctx2th", "g_LK", "g_h"])
    rng = np.random.default_rng(20260706)
    lows = {
        "c_th2ctx": max(0.0, float(base["c_th2ctx"]) - 0.015),
        "c_ctx2th": max(0.05, float(base["c_ctx2th"]) - 0.035),
        "g_LK": max(0.020, float(base["g_LK"]) - 0.008),
        "g_h": max(0.035, float(base["g_h"]) - 0.010),
    }
    highs = {
        "c_th2ctx": min(0.10, float(base["c_th2ctx"]) + 0.015),
        "c_ctx2th": min(0.22, float(base["c_ctx2th"]) + 0.035),
        "g_LK": min(0.070, float(base["g_LK"]) + 0.008),
        "g_h": min(0.095, float(base["g_h"]) + 0.010),
    }
    samples = []
    for i in range(300):
        u = (np.arange(4) + rng.random(4)) / 300.0
        rng.shuffle(u)
        # independent Latin-like projection per variable, deterministic enough for diagnostics
        sample = {}
        for j, key in enumerate(["c_th2ctx", "c_ctx2th", "g_LK", "g_h"]):
            sample[key] = lows[key] + u[j] * (highs[key] - lows[key])
        samples.append(sample)
    log("Phase 3 starting local physiological Latin-hypercube-like sweep n<=300")
    seed = normalize_base_row(base)
    seed["seed_id"] = "phase3_base"
    seed["seed_group"] = "phase3_local_best_T4_T13_T6lt060"
    for sample in samples:
        if total_completed_sims() >= MAX_TOTAL_SIMS:
            log("Simulation budget reached during Phase 3; stopping gracefully.")
            return
        key = tuple(round(sample[k], 10) for k in ["c_th2ctx", "c_ctx2th", "g_LK", "g_h"])
        if key in done:
            continue
        params = {p: float(seed[p]) for p in PARAM_COLS}
        params.update(sample)
        row = base_seed_row(seed, "phase3_local_physiological")
        row.update(sample)
        run_and_append(PHASE3_PATH, row, params, DEFAULT_DURATION_MS)
        done.add(key)


def phase4() -> None:
    df = collect_all_results()
    if df.empty:
        return
    candidates = df[
        (df["n_passed"] == 13)
        | ((df["T4"] == 1) & (df["T13"] == 1) & (df["T6_ibi_cv"] <= 0.45))
    ].copy()
    if candidates.empty:
        log("Phase 4 skipped: no 13/13 or T4=1,T13=1,T6_ibi_cv<=0.45 candidate.")
        return
    candidates["distance"] = candidates.apply(score_distance, axis=1)
    top = candidates.sort_values(["n_passed", "distance"], ascending=[False, True]).head(5)
    done = read_done_keys(PHASE4_PATH, ["c_th2ctx", "c_ctx2th", "g_LK", "g_h"])
    log(f"Phase 4 starting 120s checks for {len(top)} candidate(s).")
    for _, raw in top.iterrows():
        if total_completed_sims() >= MAX_TOTAL_SIMS:
            log("Simulation budget reached during Phase 4; stopping gracefully.")
            return
        base = normalize_base_row(raw)
        base["seed_id"] = f"phase4_{int(raw.name):05d}"
        base["seed_group"] = "phase4_120s_selected"
        key = tuple(round(float(base[p]), 10) for p in ["c_th2ctx", "c_ctx2th", "g_LK", "g_h"])
        if key in done:
            continue
        params = {p: float(base[p]) for p in PARAM_COLS}
        row = base_seed_row(base, "phase4_120s")
        run_and_append(PHASE4_PATH, row, params, LONG_DURATION_MS)
        done.add(key)


def save_scatter(df: pd.DataFrame, path: Path, x: str, y: str, color: str, title: str,
                 xlabel: str, ylabel: str, hline: float | None = None) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for val, label in [(0, f"{color}=0"), (1, f"{color}=1")]:
        sub = df[df[color] == val]
        ax.scatter(sub[x], sub[y], s=18, alpha=0.55 if val == 0 else 0.8, label=label)
    if hline is not None:
        ax.axhline(hline, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_heatmap(df: pd.DataFrame, path: Path, value: str, title: str) -> None:
    if df.empty or value not in df:
        return
    plot_df = df.dropna(subset=["c_th2ctx", "c_ctx2th", value]).copy()
    if plot_df.empty:
        return
    plot_df["x"] = plot_df["c_th2ctx"].round(5)
    plot_df["y"] = plot_df["c_ctx2th"].round(5)
    pivot = plot_df.groupby(["y", "x"])[value].max().unstack("x")
    fig, ax = plt.subplots(figsize=(8, 5.8), dpi=150)
    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("c_th2ctx")
    ax.set_ylabel("c_ctx2th")
    fig.colorbar(im, ax=ax, label=value)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_figures() -> None:
    p1 = read_csv_numeric(PHASE1_PATH) if PHASE1_PATH.exists() else pd.DataFrame()
    p2 = read_csv_numeric(PHASE2_PATH) if PHASE2_PATH.exists() else pd.DataFrame()
    save_scatter(
        p1,
        OUTDIR / "fig_phase1_cth2ctx_vs_t13_density.png",
        "c_th2ctx",
        "T13_ctx_verified_density_per_min",
        "T6",
        "Phase 1 c_th2ctx vs T13 density",
        "c_th2ctx",
        "T13 verified cortical spindle density per min",
    )
    save_scatter(
        p1,
        OUTDIR / "fig_phase1_cth2ctx_vs_t6.png",
        "c_th2ctx",
        "T6_ibi_cv",
        "T13",
        "Phase 1 c_th2ctx vs T6 IBI CV",
        "c_th2ctx",
        "T6 IBI CV",
        hline=IBI_CV_MAX,
    )
    save_scatter(
        p1,
        OUTDIR / "fig_phase1_cth2ctx_vs_t4q.png",
        "c_th2ctx",
        "T4_q",
        "T13",
        "Phase 1 c_th2ctx vs T4 Q",
        "c_th2ctx",
        "T4 Q",
        hline=SO_Q_MIN,
    )
    save_scatter(
        p1,
        OUTDIR / "fig_phase1_t6_vs_t13_density.png",
        "T13_ctx_verified_density_per_min",
        "T6_ibi_cv",
        "T4",
        "Phase 1 T6 vs T13 density",
        "T13 verified cortical spindle density per min",
        "T6 IBI CV",
    )
    save_heatmap(p2, OUTDIR / "fig_phase2_heatmap_npassed.png", "n_passed", "Phase 2 max n_passed")
    save_heatmap(p2, OUTDIR / "fig_phase2_heatmap_t6.png", "T6_ibi_cv", "Phase 2 T6 IBI CV")
    save_heatmap(
        p2,
        OUTDIR / "fig_phase2_heatmap_t13_density.png",
        "T13_ctx_verified_density_per_min",
        "Phase 2 T13 density",
    )


def corr(df: pd.DataFrame, x: str, y: str) -> tuple[float, float, int]:
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 3:
        return np.nan, np.nan, len(sub)
    rho, p = spearmanr(sub[x], sub[y])
    return float(rho), float(p), len(sub)


def write_summary() -> str:
    df = collect_all_results()
    p1 = read_csv_numeric(PHASE1_PATH) if PHASE1_PATH.exists() else pd.DataFrame()
    p2 = read_csv_numeric(PHASE2_PATH) if PHASE2_PATH.exists() else pd.DataFrame()
    p3 = read_csv_numeric(PHASE3_PATH) if PHASE3_PATH.exists() else pd.DataFrame()
    p4 = read_csv_numeric(PHASE4_PATH) if PHASE4_PATH.exists() else pd.DataFrame()

    if df.empty:
        text = "No sweep results were generated.\n"
        SUMMARY_PATH.write_text(text, encoding="utf-8")
        return text

    any_13 = int((df["n_passed"] == 13).sum())
    any_t4_t6_t13 = int(((df["T4"] == 1) & (df["T6"] == 1) & (df["T13"] == 1)).sum())
    best = df.sort_values(
        ["n_passed", "T4", "T6", "T13", "T13_ctx_verified_density_per_min"],
        ascending=[False, False, False, False, False],
    ).iloc[0]
    phase1_rhos = {
        "c_th2ctx_vs_T13_density": corr(p1, "c_th2ctx", "T13_ctx_verified_density_per_min"),
        "c_th2ctx_vs_T6_ibi_cv": corr(p1, "c_th2ctx", "T6_ibi_cv"),
        "c_th2ctx_vs_T4_q": corr(p1, "c_th2ctx", "T4_q"),
    }
    v8b_band = df[(df["c_th2ctx"] >= 0.075) & (df["c_th2ctx"] <= 0.085)]
    v8b_band_best = int(v8b_band["n_passed"].max()) if not v8b_band.empty else None
    v8b_band_t4t6t13 = int(((v8b_band["T4"] == 1) & (v8b_band["T6"] == 1) & (v8b_band["T13"] == 1)).sum()) if not v8b_band.empty else 0

    if any_13:
        recommendation = "C) Run longer-duration T6 stability validation on near-feasible candidates"
    elif any_t4_t6_t13:
        recommendation = "C) Run longer-duration T6 stability validation on near-feasible candidates"
    else:
        recommendation = "B) Run narrower local DE around best sweep region"

    lines = [
        "V8a coupling sweep long diagnostic summary",
        "=" * 44,
        "",
        "No full DE, no V8b, no threshold/reward/T13 changes, no 300s validation.",
        f"total_simulations_completed: {len(df)}",
        f"phase1_simulations: {len(p1)}",
        f"phase2_simulations: {len(p2)}",
        f"phase3_simulations: {len(p3)}",
        f"phase4_120s_simulations: {len(p4)}",
        "",
        f"any_13_of_13_candidate: {'yes' if any_13 else 'no'} ({any_13})",
        f"any_T4_T6_T13_candidate: {'yes' if any_t4_t6_t13 else 'no'} ({any_t4_t6_t13})",
        f"best_n_passed: {int(best['n_passed'])}",
        (
            "best_T4_T6_T13: "
            f"T4={int(best['T4'])}, T6={int(best['T6'])}, T13={int(best['T13'])}"
        ),
        (
            "best_metrics: "
            f"c_th2ctx={best['c_th2ctx']:.6f}, c_ctx2th={best['c_ctx2th']:.6f}, "
            f"T4_q={best['T4_q']:.3f}, T6_ibi_cv={best['T6_ibi_cv']:.3f}, "
            f"T13_density={best['T13_ctx_verified_density_per_min']:.3f}"
        ),
        "",
        "Phase 1 Spearman trends:",
    ]
    for name, (rho, p, n) in phase1_rhos.items():
        lines.append(f"  {name}: rho={rho:.4f}, p={p:.3g}, n={n}")
    lines.extend(
        [
            "",
            "Boundary/V8b reading:",
            f"  points_in_0.075_to_0.085_band: {len(v8b_band)}",
            f"  best_n_passed_in_0.075_to_0.085_band: {v8b_band_best}",
            f"  T4_T6_T13_points_in_0.075_to_0.085_band: {v8b_band_t4t6t13}",
            "",
            "Conservative answers:",
            f"1. Did any 13/13 candidate appear? {'Yes' if any_13 else 'No'}.",
            (
                "2. Did any candidate satisfy T4=1, T6=1, T13=1? "
                f"{'Yes' if any_t4_t6_t13 else 'No'}."
            ),
            (
                "3. Increasing c_th2ctx and T13 density: "
                f"Phase 1 rho={phase1_rhos['c_th2ctx_vs_T13_density'][0]:.4f}."
            ),
            (
                "4. Increasing c_th2ctx and T6_ibi_cv: "
                f"Phase 1 rho={phase1_rhos['c_th2ctx_vs_T6_ibi_cv'][0]:.4f}."
            ),
            (
                "5. Increasing c_th2ctx and T4_q: "
                f"Phase 1 rho={phase1_rhos['c_th2ctx_vs_T4_q'][0]:.4f}."
            ),
            (
                "6. Is c_th2ctx too small the main blocker? The sweep should be "
                "read cautiously; lack of T4/T6/T13 co-occurrence argues against "
                "a simple 'too small only' explanation."
            ),
            (
                "7. Coupling trade-off: records support a coupling trade-off if "
                "higher c_th2ctx improves T13 while increasing T6_ibi_cv or lowering T4_q."
            ),
            (
                "8. V8b usefulness: V8b may be useful only as a diagnostic if the "
                "0.075-0.085 band has promising points; otherwise it is likely to "
                "worsen SO stability."
            ),
            (
                "9. Narrow local DE: more reasonable than full V8b if no 13/13 "
                "appears but a near-feasible region exists."
            ),
            (
                "10. Longer-duration T6 validation: should precede objective changes "
                "only for candidates close enough to T6/T13 coexistence."
            ),
            "",
            f"Recommended one next action: {recommendation}",
        ]
    )
    text = "\n".join(lines) + "\n"
    SUMMARY_PATH.write_text(text, encoding="utf-8", newline="\n")
    return text


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log("Starting V8a coupling sweep long diagnostic.")
    try:
        records = read_csv_numeric(RECORDS_PATH)
        ensure_required_columns(records)
        if not BEST_PATH.exists():
            raise FileNotFoundError(BEST_PATH)
        if not NEAR12_PATH.exists():
            log(f"Warning: optional near-12 file not found: {NEAR12_PATH}")
        with BEST_PATH.open("r", encoding="utf-8") as f:
            json.load(f)

        if SEEDS_PATH.exists():
            seeds = read_csv_numeric(SEEDS_PATH)
            log(f"Loaded existing selected seeds: {len(seeds)}")
        else:
            seeds = choose_seeds(records)
            seeds.to_csv(SEEDS_PATH, index=False)
            log(f"Selected and saved seeds: {len(seeds)}")

        if total_completed_sims() < MAX_TOTAL_SIMS:
            phase1(seeds)
        if total_completed_sims() < MAX_TOTAL_SIMS:
            phase2(seeds)
        if total_completed_sims() < MAX_TOTAL_SIMS:
            phase3()
        if total_completed_sims() < MAX_TOTAL_SIMS:
            phase4()
        make_figures()
        summary = write_summary()
        log("Finished V8a coupling sweep long diagnostic.")
        print(summary)
    except KeyboardInterrupt:
        log("Interrupted by user; writing partial summary.")
        make_figures()
        write_summary()
        raise
    except Exception:
        log("Fatal error:\n" + traceback.format_exc())
        make_figures()
        write_summary()
        raise


if __name__ == "__main__":
    main()
