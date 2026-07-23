"""Diagnostic V8a T6-vs-T13 coupling sweep.

This script maps c_th2ctx/c_ctx2th coupling effects around representative
parameter sets. It is not an optimizer and does not modify V8a. It imports the
approved V8a build_model and compute_constraints_v8 functions, runs 60 s
simulations with the same 5 s burn-in convention, and writes flat CSV/summary
outputs.

Outputs:
    outputs/v8a_t6_t13_coupling_sweep.csv
    outputs/v8a_t6_t13_coupling_sweep_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.s4_personalize_fig7_v8 import (  # noqa: E402
    FS_SIM,
    build_model,
    compute_constraints_v8,
    compute_epoch_psd,
)
from neurolib.models.multimodel.builder.base.constants import EXC  # noqa: E402


V7_BEST_PATH = ROOT / "data" / "patient_params_fig7_v7_SC4001.json"
V8A_BEST_PATH = ROOT / "data" / "patient_params_fig7_v8_SC4001.json"
NARROW_BEST_PATH = ROOT / "outputs" / "v8a_local_de_t6_rescue_narrow" / "best_so_far.json"
NARROW_RECORDS_PATH = (
    ROOT / "outputs" / "v8a_local_de_t6_rescue_narrow" / "narrow_local_de_records.csv"
)
OUT_CSV = ROOT / "outputs" / "v8a_t6_t13_coupling_sweep.csv"
OUT_SUMMARY = ROOT / "outputs" / "v8a_t6_t13_coupling_sweep_summary.txt"

PARAM_NAMES = ["mue", "mui", "b", "tauA", "g_LK", "g_h", "c_th2ctx", "c_ctx2th"]
T_COLS = [f"T{i}" for i in range(1, 14)]
RESULT_FIELDS = [
    "base_id",
    "base_source",
    "base_n_passed",
    "base_failed_constraints",
    *PARAM_NAMES,
    "n_passed",
    "feasible",
    "T4",
    "T4_q",
    "T6",
    "T6_ibi_cv",
    "T8",
    "T8_n_sp_events",
    "T12",
    "T12_n_verified",
    "T13",
    "T13_n_ctx_events",
    "T13_n_ctx_verified",
    "T13_ctx_verified_density_per_min",
    "failed_constraints",
    "error",
    "elapsed_s",
]

SIM_DURATION_MS = 60_000
BURN_IN_S = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cth2ctx-n", type=int, default=31)
    parser.add_argument("--cctx2th-n", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-sims", type=int, default=None)
    return parser.parse_args()


def read_json(path: Path, source: str) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        row = json.load(f)
    row["base_source"] = source
    return row


def failed_constraints(row: dict | pd.Series) -> str:
    failed = []
    unknown = []
    for t in T_COLS:
        value = row.get(t, np.nan)
        try:
            if pd.isna(value):
                unknown.append(t)
            elif int(float(value)) == 0:
                failed.append(t)
        except Exception:
            unknown.append(t)
    if unknown and not failed:
        return "unknown"
    return ",".join(failed) if failed else "none"


def read_csv_numeric(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() == df[col].notna().sum():
            df[col] = converted
    return df


def select_bases() -> pd.DataFrame:
    bases: list[dict] = []
    for source, path in [
        ("v7_best", V7_BEST_PATH),
        ("v8a_full_de_best", V8A_BEST_PATH),
        ("narrow_local_de_best", NARROW_BEST_PATH),
    ]:
        row = read_json(path, source)
        if row is not None:
            bases.append(row)

    narrow = read_csv_numeric(NARROW_RECORDS_PATH)
    if not narrow.empty:
        for col in PARAM_NAMES + ["n_passed"]:
            if col not in narrow:
                narrow[col] = np.nan
        top = narrow[narrow["n_passed"] == 12].copy()
        if not top.empty:
            top["failed_constraints"] = top.apply(failed_constraints, axis=1)
            top = top.sort_values(
                ["T4", "T13", "T6_ibi_cv", "T13_ctx_verified_density_per_min"],
                ascending=[False, False, True, False],
            ).head(3)
            for _, row in top.iterrows():
                item = row.to_dict()
                item["base_source"] = "narrow_top3_npassed12"
                bases.append(item)

    if not bases:
        raise RuntimeError("No representative bases found.")

    df = pd.DataFrame(bases)
    missing_param = [p for p in PARAM_NAMES if p not in df.columns]
    if missing_param:
        raise ValueError(f"Missing parameter columns in bases: {missing_param}")
    df = df.dropna(subset=PARAM_NAMES).copy()
    key = df[PARAM_NAMES].round(10).astype(str).agg("|".join, axis=1)
    df = df.loc[~key.duplicated()].reset_index(drop=True)
    if "base_id" in df.columns:
        df = df.drop(columns=["base_id"])
    df.insert(0, "base_id", [f"base_{i:02d}" for i in range(len(df))])
    df["base_n_passed"] = df.get("n_passed", np.nan)
    df["base_failed_constraints"] = df.apply(failed_constraints, axis=1)
    return df


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULT_FIELDS})
        f.flush()


def done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    df = read_csv_numeric(path)
    if df.empty:
        return set()
    return {
        (str(r["base_id"]), round(float(r["c_th2ctx"]), 10), round(float(r["c_ctx2th"]), 10))
        for _, r in df.iterrows()
    }


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


def simulate(params: dict) -> dict:
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
            duration=SIM_DURATION_MS,
        )
        model.run()
        r_ctx, r_thal = extract_series(model)
        f_ctx, p_ctx = compute_epoch_psd(r_ctx, FS_SIM)
        n_passed, details = compute_constraints_v8(
            r_ctx, r_thal, f_c=f_ctx, p_c=p_ctx, fs=FS_SIM
        )
        row = {
            "n_passed": int(n_passed),
            "feasible": int(bool(details.get("feasible", n_passed == 13))),
            "error": "",
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
            "T13_ctx_verified_density_per_min",
        ]:
            row[key] = float(details.get(key, np.nan))
        row["failed_constraints"] = failed_constraints(row)
    except Exception as exc:
        row = {
            "n_passed": 0,
            "feasible": 0,
            "T4": 0,
            "T4_q": np.nan,
            "T6": 0,
            "T6_ibi_cv": np.nan,
            "T8": 0,
            "T8_n_sp_events": np.nan,
            "T12": 0,
            "T12_n_verified": np.nan,
            "T13": 0,
            "T13_n_ctx_events": np.nan,
            "T13_n_ctx_verified": np.nan,
            "T13_ctx_verified_density_per_min": np.nan,
            "failed_constraints": "simulation_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    row["elapsed_s"] = round(time.time() - start, 3)
    return row


def run_sweep(args: argparse.Namespace) -> None:
    bases = select_bases()
    bases_path = ROOT / "outputs" / "v8a_t6_t13_coupling_sweep_bases.csv"
    bases.to_csv(bases_path, index=False)

    if OUT_CSV.exists() and not args.resume:
        OUT_CSV.unlink()
    done = done_keys(OUT_CSV) if args.resume else set()

    cth_grid = np.linspace(0.00, 0.10, args.cth2ctx_n)
    ctx_grid = np.linspace(0.08, 0.20, args.cctx2th_n)
    sim_count = 0
    for _, base in bases.iterrows():
        for c_th2ctx in cth_grid:
            for c_ctx2th in ctx_grid:
                key = (str(base["base_id"]), round(float(c_th2ctx), 10), round(float(c_ctx2th), 10))
                if key in done:
                    continue
                if args.max_sims is not None and sim_count >= args.max_sims:
                    write_summary()
                    return
                params = {p: float(base[p]) for p in PARAM_NAMES}
                params["c_th2ctx"] = float(c_th2ctx)
                params["c_ctx2th"] = float(c_ctx2th)
                sim = simulate(params)
                row = {
                    "base_id": base["base_id"],
                    "base_source": base["base_source"],
                    "base_n_passed": base["base_n_passed"],
                    "base_failed_constraints": base["base_failed_constraints"],
                    **params,
                    **sim,
                }
                append_row(OUT_CSV, row)
                sim_count += 1
                print(
                    f"{sim_count:05d} {base['base_id']} c_th2ctx={c_th2ctx:.4f} "
                    f"c_ctx2th={c_ctx2th:.4f} n={sim['n_passed']} "
                    f"T4/T6/T13={sim['T4']}/{sim['T6']}/{sim['T13']}",
                    flush=True,
                )
    write_summary()


def monotonic_summary(df: pd.DataFrame) -> tuple[str, str, str]:
    messages = []
    for base_id, group in df.groupby("base_id"):
        agg = group.groupby("c_th2ctx").agg(
            T13_density=("T13_ctx_verified_density_per_min", "mean"),
            T6_ibi_cv=("T6_ibi_cv", "mean"),
            T4_q=("T4_q", "mean"),
        )
        if len(agg) < 3:
            continue
        rho_t13, _ = spearmanr(agg.index, agg["T13_density"])
        rho_t6, _ = spearmanr(agg.index, agg["T6_ibi_cv"])
        rho_t4, _ = spearmanr(agg.index, agg["T4_q"])
        messages.append((base_id, rho_t13, rho_t6, rho_t4))
    if not messages:
        return "unknown", "unknown", "unknown"
    t13_help = np.nanmean([m[1] for m in messages])
    t6_hurt = np.nanmean([m[2] for m in messages])
    t4_hurt = np.nanmean([m[3] for m in messages])
    return (
        f"mean Spearman rho c_th2ctx->T13 density = {t13_help:.3f}",
        f"mean Spearman rho c_th2ctx->T6 IBI CV = {t6_hurt:.3f}",
        f"mean Spearman rho c_th2ctx->T4_q = {t4_hurt:.3f}",
    )


def write_summary() -> None:
    if not OUT_CSV.exists():
        OUT_SUMMARY.write_text("No sweep CSV found.\n", encoding="utf-8")
        return
    df = read_csv_numeric(OUT_CSV)
    if df.empty:
        OUT_SUMMARY.write_text("Sweep CSV is empty.\n", encoding="utf-8")
        return
    core = df[(df["T4"] == 1) & (df["T6"] == 1) & (df["T13"] == 1)]
    t4_t13 = df[(df["T4"] == 1) & (df["T13"] == 1)]
    t4_t6 = df[(df["T4"] == 1) & (df["T6"] == 1)]
    lowest_t6 = t4_t13["T6_ibi_cv"].min() if not t4_t13.empty else np.nan
    highest_t13 = (
        t4_t6["T13_ctx_verified_density_per_min"].max() if not t4_t6.empty else np.nan
    )
    t13_msg, t6_msg, t4_msg = monotonic_summary(df)
    if not core.empty:
        recommendation = "local DE around the T4/T6/T13-compatible grid region"
    elif np.isfinite(lowest_t6) and lowest_t6 <= 0.43:
        recommendation = "one final ultra-narrow local search around the best T13-preserved T6_ibi_cv point"
    else:
        recommendation = "stop searching for 13/13 and move to paper framing / post-hoc diagnostic figures"

    lines = [
        "V8a T6-vs-T13 coupling sweep summary",
        "=" * 42,
        "",
        "This is a diagnostic sweep only; no V8a constraints or T13 logic were changed.",
        f"total_grid_points: {len(df)}",
        f"base_count: {df['base_id'].nunique()}",
        f"T4=1,T6=1,T13=1 points: {len(core)}",
        f"lowest T6_ibi_cv among T4=1,T13=1 points: {lowest_t6:.3f}",
        f"highest T13 density among T4=1,T6=1 points: {highest_t13:.3f}",
        "",
        "c_th2ctx trend checks:",
        f"  {t13_msg}",
        f"  {t6_msg}",
        f"  {t4_msg}",
        "",
        f"Recommended next step: {recommendation}",
        "",
    ]
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        run_sweep(args)
    except KeyboardInterrupt:
        write_summary()
        raise
    except Exception:
        traceback.print_exc()
        write_summary()
        raise


if __name__ == "__main__":
    main()
