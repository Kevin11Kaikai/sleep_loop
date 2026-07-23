"""Local V8a T6-rescue diagnostic search.

This is not V8b and not a full DE over the original broad bounds. It runs
narrow local differential-evolution searches around existing V8a near-feasible
regions to test whether T6 can be rescued while preserving T4 and T13.

Outputs are checkpointed continuously under:
    outputs/v8a_local_de_t6_rescue_long/
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


EVOLUTION_RECORDS = ROOT / "outputs" / "evolution_fig7_v8_records.csv"
PHASE1_SWEEP = ROOT / "outputs" / "v8a_coupling_sweep_long" / "phase1_cth2ctx_sweep.csv"
PHASE2_SWEEP = ROOT / "outputs" / "v8a_coupling_sweep_long" / "phase2_coupling_2d_sweep.csv"
OUTDIR = ROOT / "outputs" / "v8a_local_de_t6_rescue_long"
LOG_PATH = OUTDIR / "run_log.txt"
BASES_PATH = OUTDIR / "selected_base_candidates.csv"
RECORDS_PATH = OUTDIR / "local_de_records.csv"
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

MAX_TOTAL_SIMS = 6000
WALL_TIME_LIMIT_S = 11.5 * 3600.0
SIM_DURATION_MS = 60_000
BURN_IN_S = 5.0

# Keep local DE deliberately narrow. Several bases and seeds provide diversity.
MAX_BASES = 10
DE_SEEDS = [20260706, 20260707, 20260708]
DE_MAXITER = 14
DE_POPSIZE = 7


class StopSearch(Exception):
    pass


STATE = {
    "start_time": None,
    "eval_count": 0,
    "best": None,
    "found_confirmed_13": False,
}


def log(message: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
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


def failed_constraints(row: pd.Series | dict) -> str:
    failed = [t for t in T_COLS if int(float(row.get(t, 0))) == 0]
    return ",".join(failed) if failed else "none"


def output_record(row: dict) -> None:
    exists = RECORDS_PATH.exists()
    with RECORDS_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RECORD_FIELDS})
        f.flush()


def load_best_so_far() -> dict | None:
    if BEST_PATH.exists():
        try:
            return json.loads(BEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_best(row: dict) -> None:
    BEST_PATH.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")


def load_existing_eval_count() -> int:
    if not RECORDS_PATH.exists():
        return 0
    try:
        return len(pd.read_csv(RECORDS_PATH))
    except Exception:
        return 0


def extract_series(model) -> tuple[np.ndarray, np.ndarray]:
    r_exc = model[f"r_mean_{EXC}"]
    if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
        r_ctx = r_exc[0, :] * 1000.0
        r_thal = r_exc[1, :] * 1000.0
    else:
        r_ctx = np.asarray(r_exc).squeeze() * 1000.0
        r_thal = np.zeros_like(r_ctx)
    n_drop = int(BURN_IN_S * FS_SIM)
    return r_ctx[n_drop:], r_thal[n_drop:]


def simulate(params: dict, phase: str, base: pd.Series | dict, de_seed: int | str) -> dict:
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


def diagnostic_score(row: dict | pd.Series) -> float:
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
        rank = 1_000_000.0
    elif t4 and t6 and t13:
        rank = 700_000.0
    elif t4 and t13:
        rank = 500_000.0
    elif t4 and t6:
        rank = 350_000.0
    else:
        rank = 100_000.0

    soft = (
        20_000.0 * n_passed
        + 12_000.0 * t4
        + 12_000.0 * t13
        + 10_000.0 * t6
        - 25_000.0 * max(0.0, t6_cv - IBI_CV_MAX)
        - 8_000.0 * max(0.0, 0.30 - t6_cv)
        + 1_500.0 * min(t4_q, 8.0)
        + 3_000.0 * min(t13_den, 4.0)
        + 1_000.0 * min(t13_ver, 4.0)
        + 100.0 * min(t12, 30.0)
    )
    if not t4:
        soft -= 60_000.0
    if not t13:
        soft -= 70_000.0
    return rank + soft


def update_best(row: dict) -> None:
    best = STATE["best"]
    if best is None or row["objective"] > best.get("objective", -math.inf):
        STATE["best"] = row.copy()
        save_best(STATE["best"])
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


def confirm_13(row: dict, base: pd.Series | dict, de_seed: int | str) -> bool:
    params = {p: float(row[p]) for p in PARAM_NAMES}
    confirm = simulate(params, "confirm_13of13_same_params", base, de_seed)
    STATE["eval_count"] += 1
    confirm["eval"] = STATE["eval_count"]
    confirm["confirmed_13of13"] = int(confirm["n_passed"] == 13)
    output_record(confirm)
    update_best(confirm)
    return confirm["n_passed"] == 13


def objective_factory(base: pd.Series, bounds: list[tuple[float, float]], de_seed: int):
    def objective(x: np.ndarray) -> float:
        if should_stop():
            raise StopSearch()
        params = {name: float(value) for name, value in zip(PARAM_NAMES, x)}
        row = simulate(params, "local_de", base, de_seed)
        STATE["eval_count"] += 1
        row["eval"] = STATE["eval_count"]
        output_record(row)
        update_best(row)
        if row["n_passed"] == 13:
            log("13/13 candidate found; rerunning same parameters once for confirmation.")
            if confirm_13(row, base, de_seed):
                STATE["found_confirmed_13"] = True
                raise StopSearch()
        return -float(row["objective"])

    return objective


def local_bounds(base: pd.Series) -> list[tuple[float, float]]:
    vals = {p: float(base[p]) for p in PARAM_NAMES}
    return [
        (vals["mue"] * 0.95, vals["mue"] * 1.05),
        (vals["mui"] * 0.95, vals["mui"] * 1.05),
        (vals["b"] * 0.90, vals["b"] * 1.10),
        (vals["tauA"] * 0.90, vals["tauA"] * 1.10),
        (max(0.020, vals["g_LK"] - 0.010), min(0.070, vals["g_LK"] + 0.010)),
        (max(0.035, vals["g_h"] - 0.010), min(0.095, vals["g_h"] + 0.010)),
        (max(0.000, vals["c_th2ctx"] - 0.015), min(0.110, vals["c_th2ctx"] + 0.015)),
        (max(0.050, vals["c_ctx2th"] - 0.020), min(0.220, vals["c_ctx2th"] + 0.020)),
    ]


def standardize_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["base_source"] = source
    if "score" not in out:
        out["score"] = out.get("objective", 0.0)
    return out


def select_bases() -> pd.DataFrame:
    if BASES_PATH.exists():
        return read_csv_numeric(BASES_PATH)

    frames = [
        standardize_source(read_csv_numeric(PHASE1_SWEEP), "phase1_coupling_sweep"),
        standardize_source(read_csv_numeric(PHASE2_SWEEP), "phase2_coupling_sweep"),
        standardize_source(read_csv_numeric(EVOLUTION_RECORDS), "v8a_evolution_records"),
    ]
    data = pd.concat([f for f in frames if not f.empty], ignore_index=True, sort=False)
    if data.empty:
        raise RuntimeError("No input records found for local DE base selection.")

    for col in PARAM_NAMES + T_COLS + METRIC_COLS + ["n_passed", "score"]:
        if col not in data:
            data[col] = np.nan

    data["failed_constraints"] = data.apply(failed_constraints, axis=1)
    data["priority_score"] = (
        -data["T6_ibi_cv"].fillna(9.0)
        + 0.20 * data["T4_q"].fillna(0.0)
        + 0.50 * data["T13_n_ctx_verified"].fillna(0.0)
        + 0.30 * data["T13_ctx_verified_density_per_min"].fillna(0.0)
        + 0.05 * data["n_passed"].fillna(0.0)
    )

    candidates = data[
        (data["T4"] == 1)
        & (data["T13"] == 1)
        & (data["T6"] == 0)
        & (data["T6_ibi_cv"] < 0.65)
        & (data["T4_q"] > 2.0)
        & (data["T13_n_ctx_verified"] >= 1)
        & (data["T13_ctx_verified_density_per_min"] > 0)
    ].copy()
    candidates = candidates.sort_values(
        ["priority_score", "n_passed", "score"], ascending=[False, False, False]
    ).head(8)

    # Explicitly include known near-feasible neighborhoods if present.
    known = data[
        (
            (data["c_th2ctx"].between(0.0775, 0.0875))
            & (data["c_ctx2th"].between(0.118, 0.128))
        )
        | (
            (data["c_th2ctx"].between(0.090, 0.100))
            & (data["c_ctx2th"].between(0.124, 0.134))
        )
    ].copy()
    known = known[
        (known["T4"] == 1)
        & (known["T13"] == 1)
        & (known["T6_ibi_cv"] < 0.65)
    ].sort_values(["n_passed", "priority_score"], ascending=[False, False]).head(4)

    v8a_best = data[
        (data["n_passed"] == 12)
        & (data["failed_constraints"] == "T6")
        & (data["T13"] == 1)
    ].sort_values("score", ascending=False).head(1)

    bases = pd.concat([candidates, known, v8a_best], ignore_index=True, sort=False)
    if bases.empty:
        raise RuntimeError("No valid local DE bases selected.")
    dedupe_key = bases[PARAM_NAMES].round(10).astype(str).agg("|".join, axis=1)
    bases = bases.loc[~dedupe_key.duplicated()].copy().head(MAX_BASES)
    bases = bases.reset_index(drop=True)
    bases.insert(0, "base_id", [f"base_{i:03d}" for i in range(len(bases))])
    bases.to_csv(BASES_PATH, index=False)
    return bases


def write_summary() -> str:
    records = read_csv_numeric(RECORDS_PATH)
    wall = time.time() - STATE["start_time"] if STATE["start_time"] else 0.0
    if records.empty:
        text = "No local DE records were generated.\n"
        SUMMARY_PATH.write_text(text, encoding="utf-8")
        return text

    records["objective"] = pd.to_numeric(records["objective"], errors="coerce")
    best = records.sort_values("objective", ascending=False).iloc[0]
    any_13 = int((records["n_passed"] == 13).sum())
    core = int(((records["T4"] == 1) & (records["T6"] == 1) & (records["T13"] == 1)).sum())
    t13_preserved_t6_fail = records[(records["T13"] == 1) & (records["T6"] == 0)]
    t6_rescued_t13_lost = records[(records["T6"] == 1) & (records["T13"] == 0)]

    lines = [
        "V8a local DE T6 rescue diagnostic summary",
        "=" * 48,
        "",
        "No full DE, no V8b, no V8a threshold/reward/T13 changes, no long validation.",
        f"total_simulations: {len(records)}",
        f"wall_time_h: {wall / 3600.0:.3f}",
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
        f"T13_preserved_but_T6_failed_count: {len(t13_preserved_t6_fail)}",
        f"T6_rescued_but_T13_lost_count: {len(t6_rescued_t13_lost)}",
    ]
    if not t13_preserved_t6_fail.empty:
        min_cv = t13_preserved_t6_fail["T6_ibi_cv"].min()
        lines.append(f"lowest_T6_ibi_cv_while_T13_preserved: {min_cv:.3f}")
    if not t6_rescued_t13_lost.empty:
        max_den = t6_rescued_t13_lost["T13_ctx_verified_density_per_min"].max()
        lines.append(f"highest_T13_density_when_T6_rescued_but_T13_lost: {max_den:.3f}")

    if any_13 or core:
        interpretation = (
            "A local core-compatible candidate appeared. The next action should be "
            "longer-duration T6 stability validation before any objective change."
        )
        recommendation = "C) Run longer-duration T6 stability validation on near-feasible candidates"
    else:
        interpretation = (
            "No 13/13 or T4=1,T6=1,T13=1 candidate appeared in this local diagnostic. "
            "This is evidence for a local T6-vs-T13 trade-off under the current V8a "
            "detector/objective and parameter neighborhood, not evidence of model failure."
        )
        recommendation = "B) Run narrower local DE around best sweep region"

    lines.extend(["", interpretation, "", f"Recommended one next action: {recommendation}", ""])
    text = "\n".join(lines)
    SUMMARY_PATH.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    STATE["start_time"] = time.time()
    STATE["eval_count"] = load_existing_eval_count()
    STATE["best"] = load_best_so_far()
    log("Starting local DE V8a T6 rescue diagnostic.")
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
                    f"Starting local DE base={base['base_id']} "
                    f"source={base.get('base_source', '')} seed={de_seed}"
                )
                try:
                    differential_evolution(
                        objective_factory(base, bounds, de_seed),
                        bounds=bounds,
                        strategy="best1bin",
                        maxiter=DE_MAXITER,
                        popsize=DE_POPSIZE,
                        mutation=(0.45, 0.9),
                        recombination=0.7,
                        seed=de_seed,
                        polish=False,
                        workers=1,
                        updating="immediate",
                        tol=1e-4,
                    )
                except StopSearch:
                    raise
                except Exception:
                    log("Local DE block error:\n" + traceback.format_exc())
        log("All local DE blocks completed.")
    except StopSearch:
        log("Local DE stopped by planned stop condition.")
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
