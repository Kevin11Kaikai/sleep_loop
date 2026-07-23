"""Narrow local V8a T6-rescue diagnostic search.

This follow-up focuses only on the best regions from the previous local
diagnostic:

- best 12/13 T13-preserved point from local_de_v8a_t6_rescue_long
- lowest T6_ibi_cv among T13-preserved points
- best points near c_th2ctx ~= 0.079, c_ctx2th ~= 0.137

It is not V8b, not full DE, and does not change V8a thresholds or T13 logic.
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


LONG_RECORDS = ROOT / "outputs" / "v8a_local_de_t6_rescue_long" / "local_de_records.csv"
LONG_BEST = ROOT / "outputs" / "v8a_local_de_t6_rescue_long" / "best_so_far.json"
COUPLING_PHASE1 = ROOT / "outputs" / "v8a_coupling_sweep_long" / "phase1_cth2ctx_sweep.csv"
COUPLING_PHASE2 = ROOT / "outputs" / "v8a_coupling_sweep_long" / "phase2_coupling_2d_sweep.csv"
OUTDIR = ROOT / "outputs" / "v8a_local_de_t6_rescue_narrow"
LOG_PATH = OUTDIR / "run_log.txt"
BASES_PATH = OUTDIR / "selected_base_candidates.csv"
RECORDS_PATH = OUTDIR / "narrow_local_de_records.csv"
BEST_PATH = OUTDIR / "best_so_far.json"
SUMMARY_PATH = OUTDIR / "summary.txt"

PARAM_NAMES = ["mue", "mui", "b", "tauA", "g_LK", "g_h", "c_th2ctx", "c_ctx2th"]
T_COLS = [f"T{i}" for i in range(1, 14)]
METRIC_COLS = [
    "T4_q",
    "T6_ibi_cv",
    "T8_n_sp_events",
    "T12_n_verified",
    "T13_n_ctx_events",
    "T13_n_ctx_verified",
    "T13_ctx_density_per_min",
    "T13_ctx_verified_density_per_min",
    "T13_mean_ctx_dur",
]
RECORD_FIELDS = [
    "eval",
    "phase",
    "base_id",
    "base_source",
    "de_seed",
    "objective",
    *PARAM_NAMES,
    "n_passed",
    "feasible",
    *T_COLS,
    *METRIC_COLS,
    "failed_constraints",
    "core_T4_T6_T13",
    "confirmed_13of13",
    "error",
    "elapsed_s",
    "wall_elapsed_s",
]

MAX_TOTAL_SIMS = 4500
WALL_TIME_LIMIT_S = 10.5 * 3600.0
SIM_DURATION_MS = 60_000
BURN_IN_S = 5.0
DE_SEEDS = [20260710, 20260711, 20260712, 20260713]
DE_MAXITER = 18
DE_POPSIZE = 6
MAX_BASES = 6

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
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() == df[col].notna().sum():
            df[col] = converted
    return df


def failed_constraints(row) -> str:
    failed = [t for t in T_COLS if int(float(row.get(t, 0))) == 0]
    return ",".join(failed) if failed else "none"


def append_record(row: dict) -> None:
    exists = RECORDS_PATH.exists()
    with RECORDS_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RECORD_FIELDS})
        f.flush()


def load_existing_eval_count() -> int:
    if not RECORDS_PATH.exists():
        return 0
    try:
        return len(pd.read_csv(RECORDS_PATH))
    except Exception:
        return 0


def load_best() -> dict | None:
    if BEST_PATH.exists():
        try:
            return json.loads(BEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
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


def diagnostic_score(row) -> float:
    n_passed = int(float(row.get("n_passed", 0)))
    t4 = int(float(row.get("T4", 0)))
    t6 = int(float(row.get("T6", 0)))
    t13 = int(float(row.get("T13", 0)))
    t4_q = float(row.get("T4_q", 0.0) or 0.0)
    t6_cv = float(row.get("T6_ibi_cv", 9.0) or 9.0)
    t12 = float(row.get("T12_n_verified", 0.0) or 0.0)
    t13_den = float(row.get("T13_ctx_verified_density_per_min", 0.0) or 0.0)
    t13_ver = float(row.get("T13_n_ctx_verified", 0.0) or 0.0)

    if n_passed == 13:
        rank = 1_200_000.0
    elif t4 and t6 and t13:
        rank = 900_000.0
    elif t4 and t13:
        rank = 650_000.0
    elif t4 and t6:
        rank = 420_000.0
    else:
        rank = 100_000.0

    # Strongly favor crossing T6 while preserving T13, but do not relax any hard threshold.
    soft = (
        18_000.0 * n_passed
        + 14_000.0 * t4
        + 18_000.0 * t13
        + 18_000.0 * t6
        - 55_000.0 * max(0.0, t6_cv - IBI_CV_MAX)
        - 3_000.0 * abs(t6_cv - 0.39)
        + 1_200.0 * min(t4_q, 8.0)
        + 4_500.0 * min(t13_den, 4.0)
        + 1_500.0 * min(t13_ver, 4.0)
        + 120.0 * min(t12, 30.0)
    )
    if not t4:
        soft -= 80_000.0
    if not t13:
        soft -= 90_000.0
    return rank + soft


def simulate(params: dict, phase: str, base, de_seed) -> dict:
    start = time.time()
    row = {
        "phase": phase,
        "base_id": base.get("base_id", ""),
        "base_source": base.get("base_source", ""),
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
            value = details.get(key, np.nan)
            row[key] = float(value) if value is not None else np.nan
        row["failed_constraints"] = failed_constraints(row)
        row["core_T4_T6_T13"] = int(row["T4"] == 1 and row["T6"] == 1 and row["T13"] == 1)
        row["error"] = ""
    except Exception as exc:
        row.update(
            {
                "n_passed": 0,
                "feasible": 0,
                **{t: 0 for t in T_COLS},
                **{m: np.nan for m in METRIC_COLS},
                "failed_constraints": "simulation_error",
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
            f"c_th2ctx={row['c_th2ctx']:.5f} c_ctx2th={row['c_ctx2th']:.5f}"
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


def confirm_13(row: dict, base, de_seed) -> bool:
    params = {p: float(row[p]) for p in PARAM_NAMES}
    confirm = simulate(params, "confirm_13of13_same_params", base, de_seed)
    STATE["eval_count"] += 1
    confirm["eval"] = STATE["eval_count"]
    confirm["confirmed_13of13"] = int(confirm["n_passed"] == 13)
    append_record(confirm)
    update_best(confirm)
    return confirm["n_passed"] == 13


def objective_factory(base, de_seed):
    def objective(x):
        if should_stop():
            raise StopSearch()
        params = {p: float(v) for p, v in zip(PARAM_NAMES, x)}
        row = simulate(params, "narrow_local_de", base, de_seed)
        STATE["eval_count"] += 1
        row["eval"] = STATE["eval_count"]
        append_record(row)
        update_best(row)
        if row["n_passed"] == 13:
            log("13/13 candidate found; confirming same params once.")
            if confirm_13(row, base, de_seed):
                STATE["found_confirmed_13"] = True
                raise StopSearch()
        return -float(row["objective"])

    return objective


def local_bounds(base) -> list[tuple[float, float]]:
    v = {p: float(base[p]) for p in PARAM_NAMES}
    return [
        (v["mue"] * 0.98, v["mue"] * 1.02),
        (v["mui"] * 0.98, v["mui"] * 1.02),
        (v["b"] * 0.95, v["b"] * 1.05),
        (v["tauA"] * 0.95, v["tauA"] * 1.05),
        (max(0.020, v["g_LK"] - 0.005), min(0.070, v["g_LK"] + 0.005)),
        (max(0.035, v["g_h"] - 0.006), min(0.095, v["g_h"] + 0.006)),
        (max(0.000, v["c_th2ctx"] - 0.008), min(0.110, v["c_th2ctx"] + 0.008)),
        (max(0.050, v["c_ctx2th"] - 0.012), min(0.220, v["c_ctx2th"] + 0.012)),
    ]


def standardize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["base_source"] = source
    if "objective" not in out:
        out["objective"] = 0.0
    return out


def select_bases() -> pd.DataFrame:
    if BASES_PATH.exists():
        return read_csv_numeric(BASES_PATH)

    frames = [
        standardize(read_csv_numeric(LONG_RECORDS), "long_local_de"),
        standardize(read_csv_numeric(COUPLING_PHASE1), "phase1_coupling_sweep"),
        standardize(read_csv_numeric(COUPLING_PHASE2), "phase2_coupling_sweep"),
    ]
    data = pd.concat([f for f in frames if not f.empty], ignore_index=True, sort=False)
    if data.empty:
        raise RuntimeError("No records found for narrow base selection.")
    for col in PARAM_NAMES + T_COLS + METRIC_COLS + ["n_passed", "objective"]:
        if col not in data:
            data[col] = np.nan

    data["failed_constraints"] = data.apply(failed_constraints, axis=1)
    data["base_rank"] = (
        50.0 * data["n_passed"].fillna(0)
        - 80.0 * data["T6_ibi_cv"].fillna(9.0)
        + 10.0 * data["T4_q"].fillna(0)
        + 25.0 * data["T13_n_ctx_verified"].fillna(0)
        + 8.0 * data["T13_ctx_verified_density_per_min"].fillna(0)
        + 0.001 * data["objective"].fillna(0)
    )

    selected = []
    # Best overall from previous local DE.
    if LONG_BEST.exists():
        best = json.loads(LONG_BEST.read_text(encoding="utf-8"))
        selected.append(pd.DataFrame([{**best, "base_source": "long_best_so_far"}]))

    # Lowest T6 CV while T13 is preserved.
    t13 = data[(data["T4"] == 1) & (data["T13"] == 1) & (data["T6"] == 0)]
    if not t13.empty:
        selected.append(t13.sort_values(["T6_ibi_cv", "n_passed"], ascending=[True, False]).head(3))

    # Best 12/13 T13-preserved T6-only candidates.
    n12 = data[
        (data["n_passed"] == 12)
        & (data["T4"] == 1)
        & (data["T13"] == 1)
        & (data["T6"] == 0)
    ]
    if not n12.empty:
        selected.append(n12.sort_values(["T6_ibi_cv", "base_rank"], ascending=[True, False]).head(3))

    # Focused geometric region around c_th2ctx ~= 0.079 / c_ctx2th ~= 0.137.
    region = data[
        data["c_th2ctx"].between(0.070, 0.088)
        & data["c_ctx2th"].between(0.125, 0.149)
        & (data["T4"] == 1)
    ]
    if not region.empty:
        selected.append(region.sort_values(["base_rank"], ascending=False).head(4))

    bases = pd.concat(selected, ignore_index=True, sort=False)
    bases = bases.dropna(subset=PARAM_NAMES)
    key = bases[PARAM_NAMES].round(10).astype(str).agg("|".join, axis=1)
    bases = bases.loc[~key.duplicated()].head(MAX_BASES).reset_index(drop=True)
    if "base_id" in bases.columns:
        bases = bases.drop(columns=["base_id"])
    bases.insert(0, "base_id", [f"narrow_base_{i:03d}" for i in range(len(bases))])
    bases.to_csv(BASES_PATH, index=False)
    return bases


def write_summary() -> str:
    records = read_csv_numeric(RECORDS_PATH)
    wall = time.time() - STATE["start_time"] if STATE["start_time"] else 0.0
    if records.empty:
        text = "No narrow local DE records were generated.\n"
        SUMMARY_PATH.write_text(text, encoding="utf-8")
        return text
    best = records.sort_values("objective", ascending=False).iloc[0]
    any13 = int((records["n_passed"] == 13).sum())
    core = int(((records["T4"] == 1) & (records["T6"] == 1) & (records["T13"] == 1)).sum())
    t13_preserved = records[(records["T13"] == 1) & (records["T4"] == 1)]
    t6_pass = records[records["T6"] == 1]
    min_cv_t13 = t13_preserved["T6_ibi_cv"].min() if not t13_preserved.empty else np.nan
    max_t13_den_t6 = (
        t6_pass["T13_ctx_verified_density_per_min"].max() if not t6_pass.empty else np.nan
    )
    recommendation = (
        "C) Run longer-duration T6 stability validation on near-feasible candidates"
        if any13 or core
        else "D) Redesign cortical spindle detector/audit"
    )

    lines = [
        "V8a narrow local DE T6 rescue summary",
        "=" * 43,
        "",
        "No V8b, no full DE over broad V8a bounds, no threshold/T13 changes, no long validation.",
        f"total_simulations: {len(records)}",
        f"wall_time_h: {wall / 3600.0:.3f}",
        f"13_of_13_candidates: {any13}",
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
        f"lowest_T6_ibi_cv_with_T4_and_T13_preserved: {min_cv_t13:.3f}",
        f"highest_T13_density_when_T6_passed: {max_t13_den_t6:.3f}",
        "",
        "Conservative interpretation:",
    ]
    if any13 or core:
        lines.append(
            "A T4/T6/T13-compatible point appeared. It should be treated as a diagnostic "
            "candidate requiring repeat and longer-duration stability validation."
        )
    else:
        lines.append(
            "No T4=1,T6=1,T13=1 point appeared. This strengthens the local evidence "
            "for a T6-vs-T13 trade-off under the current V8a detector/objective, "
            "without implying model incapacity."
        )
    lines.extend(["", f"Recommended one next action: {recommendation}", ""])
    text = "\n".join(lines)
    SUMMARY_PATH.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    STATE["start_time"] = time.time()
    STATE["eval_count"] = load_existing_eval_count()
    STATE["best"] = load_best()
    log("Starting narrow local V8a T6 rescue diagnostic.")
    log(f"Existing records: {STATE['eval_count']}")
    try:
        bases = select_bases()
        log(f"Selected/loaded bases: {len(bases)}")
        for _, base in bases.iterrows():
            for de_seed in DE_SEEDS:
                if should_stop():
                    raise StopSearch()
                bounds = local_bounds(base)
                log(
                    f"Starting narrow local DE base={base['base_id']} "
                    f"source={base.get('base_source', '')} seed={de_seed}"
                )
                try:
                    differential_evolution(
                        objective_factory(base, de_seed),
                        bounds=bounds,
                        strategy="best1bin",
                        maxiter=DE_MAXITER,
                        popsize=DE_POPSIZE,
                        mutation=(0.35, 0.75),
                        recombination=0.65,
                        seed=de_seed,
                        polish=False,
                        workers=1,
                        updating="immediate",
                        tol=1e-4,
                    )
                except StopSearch:
                    raise
                except Exception:
                    log("Narrow local DE block error:\n" + traceback.format_exc())
        log("All narrow local DE blocks completed.")
    except StopSearch:
        log("Narrow local DE stopped by planned stop condition.")
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
