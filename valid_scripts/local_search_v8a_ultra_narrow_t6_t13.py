"""Final ultra-narrow V8a T6-vs-T13 local diagnostic search.

This script starts from the best T13-preserved point in
outputs/v8a_t6_t13_coupling_sweep.csv, then performs a small local DE within a
very narrow neighborhood. It is not V8b, not a full DE, and does not change any
V8a hard constraint or detector logic.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.s4_personalize_fig7_v8 import (  # noqa: E402
    FS_SIM,
    IBI_CV_MAX,
    build_model,
    compute_constraints_v8,
    compute_epoch_psd,
)
from neurolib.models.multimodel.builder.base.constants import EXC  # noqa: E402


INPUT_SWEEP = ROOT / "outputs" / "v8a_t6_t13_coupling_sweep.csv"
OUTDIR = ROOT / "outputs" / "v8a_ultra_narrow_t6_t13_search"
LOG_PATH = OUTDIR / "run_log.txt"
BASE_PATH = OUTDIR / "selected_base_point.json"
RECORDS_PATH = OUTDIR / "ultra_narrow_records.csv"
BEST_PATH = OUTDIR / "best_so_far.json"
SUMMARY_PATH = OUTDIR / "summary.txt"

PARAM_NAMES = ["mue", "mui", "b", "tauA", "g_LK", "g_h", "c_th2ctx", "c_ctx2th"]
T_COLS = [f"T{i}" for i in range(1, 14)]
METRIC_COLS = [
    "T4_q",
    "T6_ibi_cv",
    "T8_n_sp_events",
    "T12_n_verified",
    "T13_n_ctx_verified",
    "T13_ctx_verified_density_per_min",
]
RECORD_FIELDS = [
    "eval",
    "phase",
    "de_seed",
    "objective",
    *PARAM_NAMES,
    "n_passed",
    "feasible",
    "failed_constraints",
    "T4",
    "T4_q",
    "T6",
    "T6_ibi_cv",
    "T8",
    "T8_n_sp_events",
    "T12",
    "T12_n_verified",
    "T13",
    "T13_n_ctx_verified",
    "T13_ctx_verified_density_per_min",
    "core_T4_T6_T13",
    "confirmed_13of13",
    "error",
    "elapsed_s",
    "wall_elapsed_s",
]

SIM_DURATION_MS = 60_000
BURN_IN_S = 5.0
MAX_TOTAL_SIMS = 1600
WALL_TIME_LIMIT_S = 8.0 * 3600.0
DE_SEEDS = [20260711, 20260712]
DE_MAXITER = 11
DE_POPSIZE = 8

STATE = {
    "start_time": None,
    "eval_count": 0,
    "best": None,
    "found_confirmed_13": False,
}


class StopSearch(Exception):
    pass


def log(message: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
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


def failed_constraints(row) -> str:
    failed = []
    for t in T_COLS:
        value = row.get(t, np.nan)
        try:
            if pd.isna(value) or int(float(value)) == 0:
                failed.append(t)
        except Exception:
            failed.append(t)
    return ",".join(failed) if failed else "none"


def select_base_point() -> dict:
    if not INPUT_SWEEP.exists():
        raise FileNotFoundError(INPUT_SWEEP)
    df = read_csv_numeric(INPUT_SWEEP)
    required = PARAM_NAMES + ["T4", "T13", "T6_ibi_cv"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in coupling sweep CSV: {missing}")
    candidates = df[(df["T4"] == 1) & (df["T13"] == 1)].copy()
    if candidates.empty:
        raise RuntimeError("No T4=1,T13=1 point exists in coupling sweep CSV.")
    candidates = candidates.sort_values(
        ["T6_ibi_cv", "n_passed", "T13_ctx_verified_density_per_min"],
        ascending=[True, False, False],
    )
    row = candidates.iloc[0].to_dict()
    BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASE_PATH.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def append_record(row: dict) -> None:
    exists = RECORDS_PATH.exists()
    with RECORDS_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RECORD_FIELDS})
        f.flush()


def load_eval_count() -> int:
    if not RECORDS_PATH.exists():
        return 0
    try:
        return len(pd.read_csv(RECORDS_PATH))
    except Exception:
        return 0


def load_best() -> dict | None:
    if not BEST_PATH.exists():
        return None
    try:
        return json.loads(BEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_best(row: dict) -> None:
    BEST_PATH.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_series(model):
    r_exc = model[f"r_mean_{EXC}"]
    if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
        r_ctx = r_exc[0, :] * 1000.0
        r_thal = r_exc[1, :] * 1000.0
    else:
        r_ctx = np.asarray(r_exc).squeeze() * 1000.0
        r_thal = np.zeros_like(r_ctx)
    n_drop = int(BURN_IN_S * FS_SIM)
    return r_ctx[n_drop:], r_thal[n_drop:]


def diagnostic_score(row: dict) -> float:
    n_passed = int(float(row.get("n_passed", 0)))
    t4 = int(float(row.get("T4", 0)))
    t6 = int(float(row.get("T6", 0)))
    t13 = int(float(row.get("T13", 0)))
    t4_q = float(row.get("T4_q", 0.0) or 0.0)
    t6_cv = float(row.get("T6_ibi_cv", 9.0) or 9.0)
    t8 = float(row.get("T8_n_sp_events", 0.0) or 0.0)
    t12 = float(row.get("T12_n_verified", 0.0) or 0.0)
    t13_ver = float(row.get("T13_n_ctx_verified", 0.0) or 0.0)
    t13_den = float(row.get("T13_ctx_verified_density_per_min", 0.0) or 0.0)

    if n_passed == 13:
        rank = 1_500_000.0
    elif t4 and t6 and t13:
        rank = 1_000_000.0
    elif t4 and t13:
        rank = 700_000.0
    elif t4 and t6:
        rank = 450_000.0
    else:
        rank = 120_000.0

    soft = (
        16_000.0 * n_passed
        + 20_000.0 * t4
        + 24_000.0 * t13
        + 26_000.0 * t6
        - 95_000.0 * max(0.0, t6_cv - IBI_CV_MAX)
        - 2_000.0 * abs(t6_cv - 0.395)
        + 1_000.0 * min(t4_q, 8.0)
        + 5_000.0 * min(t13_den, 4.0)
        + 2_000.0 * min(t13_ver, 4.0)
        + 100.0 * min(t8, 30.0)
        + 120.0 * min(t12, 30.0)
    )
    if not t4:
        soft -= 100_000.0
    if not t13:
        soft -= 140_000.0
    return rank + soft


def simulate(params: dict, phase: str, de_seed: int | str) -> dict:
    start = time.time()
    row = {
        "phase": phase,
        "de_seed": de_seed,
        **{p: float(params[p]) for p in PARAM_NAMES},
        "confirmed_13of13": 0,
    }
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
            duration=SIM_DURATION_MS,
        )
        model.run()
        r_ctx, r_thal = extract_series(model)
        f_ctx, p_ctx = compute_epoch_psd(r_ctx, FS_SIM)
        n_passed, details = compute_constraints_v8(
            r_ctx, r_thal, f_c=f_ctx, p_c=p_ctx, fs=FS_SIM
        )
        row["n_passed"] = int(n_passed)
        row["feasible"] = int(bool(details.get("feasible", n_passed == 13)))
        for t in T_COLS:
            row[t] = int(bool(details.get(t, False)))
        for key in METRIC_COLS:
            row[key] = float(details.get(key, np.nan))
        row["failed_constraints"] = failed_constraints(row)
        row["core_T4_T6_T13"] = int(row["T4"] == 1 and row["T6"] == 1 and row["T13"] == 1)
        row["error"] = ""
    except Exception as exc:
        row.update(
            {
                "n_passed": 0,
                "feasible": 0,
                "failed_constraints": "simulation_error",
                "T4": 0,
                "T4_q": np.nan,
                "T6": 0,
                "T6_ibi_cv": np.nan,
                "T8": 0,
                "T8_n_sp_events": np.nan,
                "T12": 0,
                "T12_n_verified": np.nan,
                "T13": 0,
                "T13_n_ctx_verified": np.nan,
                "T13_ctx_verified_density_per_min": np.nan,
                "core_T4_T6_T13": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    row["elapsed_s"] = round(time.time() - start, 3)
    row["wall_elapsed_s"] = round(time.time() - STATE["start_time"], 3)
    row["objective"] = diagnostic_score(row)
    return row


def update_best(row: dict) -> None:
    if STATE["best"] is None or row["objective"] > STATE["best"].get("objective", -math.inf):
        STATE["best"] = row.copy()
        save_best(row)
        log(
            "New best: "
            f"obj={row['objective']:.3f} n={row['n_passed']} "
            f"T4/T6/T13={row['T4']}/{row['T6']}/{row['T13']} "
            f"T4q={row['T4_q']:.3f} T6cv={row['T6_ibi_cv']:.3f} "
            f"T13dens={row['T13_ctx_verified_density_per_min']:.3f} "
            f"c_th2ctx={row['c_th2ctx']:.6f} c_ctx2th={row['c_ctx2th']:.6f}"
        )


def should_stop() -> bool:
    if STATE["eval_count"] >= MAX_TOTAL_SIMS:
        log("Stopping: simulation budget reached.")
        return True
    if time.time() - STATE["start_time"] >= WALL_TIME_LIMIT_S:
        log("Stopping: wall-time limit reached.")
        return True
    if STATE["found_confirmed_13"]:
        log("Stopping: confirmed 13/13 candidate found.")
        return True
    return False


def confirm_13(row: dict, de_seed: int | str) -> bool:
    params = {p: float(row[p]) for p in PARAM_NAMES}
    confirm = simulate(params, "confirm_13of13_same_params", de_seed)
    STATE["eval_count"] += 1
    confirm["eval"] = STATE["eval_count"]
    confirm["confirmed_13of13"] = int(confirm["n_passed"] == 13)
    append_record(confirm)
    update_best(confirm)
    return confirm["n_passed"] == 13


def objective_factory(de_seed: int):
    def objective(x):
        if should_stop():
            raise StopSearch()
        params = {p: float(v) for p, v in zip(PARAM_NAMES, x)}
        row = simulate(params, "ultra_narrow_local_de", de_seed)
        STATE["eval_count"] += 1
        row["eval"] = STATE["eval_count"]
        append_record(row)
        update_best(row)
        if row["n_passed"] == 13:
            log("13/13 candidate found; confirming same params once.")
            if confirm_13(row, de_seed):
                STATE["found_confirmed_13"] = True
                raise StopSearch()
        return -float(row["objective"])

    return objective


def local_bounds(base: dict) -> list[tuple[float, float]]:
    v = {p: float(base[p]) for p in PARAM_NAMES}
    return [
        (v["mue"] * 0.97, v["mue"] * 1.03),
        (v["mui"] * 0.97, v["mui"] * 1.03),
        (v["b"] * 0.96, v["b"] * 1.04),
        (v["tauA"] * 0.96, v["tauA"] * 1.04),
        (max(0.020, v["g_LK"] - 0.004), min(0.070, v["g_LK"] + 0.004)),
        (max(0.035, v["g_h"] - 0.004), min(0.095, v["g_h"] + 0.004)),
        (0.055, 0.075),
        (0.105, 0.135),
    ]


def write_summary() -> str:
    if not RECORDS_PATH.exists():
        text = "No ultra-narrow records were generated.\n"
        SUMMARY_PATH.write_text(text, encoding="utf-8")
        return text
    df = read_csv_numeric(RECORDS_PATH)
    if df.empty:
        text = "Ultra-narrow records file is empty.\n"
        SUMMARY_PATH.write_text(text, encoding="utf-8")
        return text
    best = df.sort_values("objective", ascending=False).iloc[0]
    any_13 = int((df["n_passed"] == 13).sum())
    core = int(((df["T4"] == 1) & (df["T6"] == 1) & (df["T13"] == 1)).sum())
    t13 = df[df["T13"] == 1]
    t6 = df[df["T6"] == 1]
    min_t6_t13 = t13["T6_ibi_cv"].min() if not t13.empty else np.nan
    max_t13_t6 = t6["T13_ctx_verified_density_per_min"].max() if not t6.empty else np.nan
    wall_h = (time.time() - STATE["start_time"]) / 3600.0 if STATE["start_time"] else np.nan

    if any_13 or core:
        recommendation = "repeat candidate and run longer-duration stability diagnostics"
        interpretation = (
            "A T4/T6/T13-compatible point appeared and should be treated only as a "
            "diagnostic candidate until repeated and stress-tested."
        )
    else:
        recommendation = "stop searching for 13/13 and move to paper framing plus diagnostic figures"
        interpretation = (
            "No T4=1,T6=1,T13=1 point appeared. This final ultra-narrow search does "
            "not change the T6-vs-T13 trade-off conclusion under the current V8a "
            "detector/objective and local parameter neighborhood."
        )

    lines = [
        "V8a ultra-narrow T6-vs-T13 local search summary",
        "=" * 55,
        "",
        "No V8b, no full DE, no threshold/T13 changes, no long validation.",
        f"total_simulations: {len(df)}",
        f"wall_time_h: {wall_h:.3f}",
        f"13_of_13_candidates: {any_13}",
        f"T4_T6_T13_candidates: {core}",
        f"best_n_passed: {int(best['n_passed'])}",
        f"best_failed_constraints: {best['failed_constraints']}",
        f"best_T4_q: {float(best['T4_q']):.3f}",
        f"best_T6_ibi_cv: {float(best['T6_ibi_cv']):.3f}",
        f"best_T13_n_ctx_verified: {float(best['T13_n_ctx_verified']):.0f}",
        f"best_T13_ctx_verified_density_per_min: {float(best['T13_ctx_verified_density_per_min']):.3f}",
        f"best_T8_n_sp_events: {float(best['T8_n_sp_events']):.0f}",
        f"best_T12_n_verified: {float(best['T12_n_verified']):.0f}",
        f"best_c_th2ctx: {float(best['c_th2ctx']):.6f}",
        f"best_c_ctx2th: {float(best['c_ctx2th']):.6f}",
        "",
        f"lowest_T6_ibi_cv_while_T13=1: {min_t6_t13:.3f}",
        f"highest_T13_density_while_T6=1: {max_t13_t6:.3f}",
        "",
        interpretation,
        "",
        f"Recommended next step: {recommendation}",
        "",
    ]
    text = "\n".join(lines)
    SUMMARY_PATH.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    STATE["start_time"] = time.time()
    STATE["eval_count"] = load_eval_count()
    STATE["best"] = load_best()
    log("Starting final ultra-narrow V8a T6-vs-T13 local search.")
    log(f"Existing records: {STATE['eval_count']}")
    try:
        base = select_base_point()
        log(
            "Selected base: "
            f"c_th2ctx={float(base['c_th2ctx']):.6f} "
            f"c_ctx2th={float(base['c_ctx2th']):.6f} "
            f"T6cv={float(base['T6_ibi_cv']):.3f} "
            f"T13dens={float(base['T13_ctx_verified_density_per_min']):.3f}"
        )
        bounds = local_bounds(base)
        for seed in DE_SEEDS:
            if should_stop():
                raise StopSearch()
            log(f"Starting ultra-narrow local DE seed={seed}")
            try:
                differential_evolution(
                    objective_factory(seed),
                    bounds=bounds,
                    strategy="best1bin",
                    maxiter=DE_MAXITER,
                    popsize=DE_POPSIZE,
                    mutation=(0.30, 0.70),
                    recombination=0.65,
                    seed=seed,
                    polish=False,
                    workers=1,
                    updating="immediate",
                    tol=1e-4,
                )
            except StopSearch:
                raise
            except Exception:
                log("Ultra-narrow local DE block error:\n" + traceback.format_exc())
        log("All ultra-narrow local DE blocks completed.")
    except StopSearch:
        log("Ultra-narrow search stopped by planned stop condition.")
    except KeyboardInterrupt:
        log("Interrupted by user.")
        raise
    except Exception:
        log("Fatal error:\n" + traceback.format_exc())
        raise
    finally:
        summary = write_summary()
        log("Summary written.")
        print(summary, flush=True)


if __name__ == "__main__":
    main()
