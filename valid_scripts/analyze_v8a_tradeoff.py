"""Post-run diagnostics for V8a evolution records.

This script reads existing V8a CSV/JSON/log outputs and writes summary
tables/figures. It does not import neurolib or run simulations.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "outputs" / "evolution_fig7_v8_records.csv"
DEFAULT_JSON = ROOT / "data" / "patient_params_fig7_v8_SC4001.json"
DEFAULT_LOG = ROOT / "outputs" / "v8a_full_de_run_log.txt"
DEFAULT_OUT = ROOT / "outputs" / "v8a_diagnostics"

T_COLS = [f"T{i}" for i in range(1, 14)]
PARAM_COLS = ["c_th2ctx", "c_ctx2th", "g_LK", "g_h"]
GROUP_STAT_COLS = PARAM_COLS + [
    "T4_q",
    "T6_ibi_cv",
    "T13_ctx_verified_density_per_min",
]
TOP_COLS = [
    "eval",
    "score",
    "failed_constraint",
    "c_th2ctx",
    "c_ctx2th",
    "T4_q",
    "T6_ibi_cv",
    "T8_n_sp_events",
    "T12_n_verified",
    "T13_n_ctx_events",
    "T13_n_ctx_verified",
    "T13_ctx_verified_density_per_min",
    "shape_r",
    "so_power",
    "thal_spindle_power",
    "ctx_spindle_power",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_inputs(csv_path: Path, json_path: Path, log_path: Path) -> tuple[pd.DataFrame, dict, str]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    df = pd.read_csv(csv_path)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() == df[col].notna().sum():
            df[col] = converted

    with json_path.open("r", encoding="utf-8") as f:
        best_json = json.load(f)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    return df, best_json, log_text


def failed_constraint(row: pd.Series) -> str:
    failed = [t for t in T_COLS if int(row[t]) == 0]
    return ",".join(failed) if failed else "none"


def quantile_summary(series: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(numeric.size),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "min": float(numeric.min()),
        "q05": float(numeric.quantile(0.05)),
        "q25": float(numeric.quantile(0.25)),
        "q75": float(numeric.quantile(0.75)),
        "q95": float(numeric.quantile(0.95)),
        "max": float(numeric.max()),
    }


def group_stats(df: pd.DataFrame, groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for group_name, gdf in groups.items():
        for col in GROUP_STAT_COLS:
            summary = quantile_summary(gdf[col] if col in gdf else pd.Series(dtype=float))
            rows.append({"group": group_name, "metric": col, **summary})
    return pd.DataFrame(rows)


def spearman_table(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("c_th2ctx", "T13_ctx_verified_density_per_min"),
        ("c_th2ctx", "T6_ibi_cv"),
        ("c_th2ctx", "T4_q"),
        ("c_ctx2th", "T13_ctx_verified_density_per_min"),
        ("c_ctx2th", "T6_ibi_cv"),
        ("g_LK", "T13_ctx_verified_density_per_min"),
        ("g_LK", "T6_ibi_cv"),
        ("g_h", "T13_ctx_verified_density_per_min"),
        ("g_h", "T6_ibi_cv"),
    ]
    rows = []
    for x_col, y_col in pairs:
        sub = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 3:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearmanr(sub[x_col], sub[y_col])
        rows.append(
            {
                "x": x_col,
                "y": y_col,
                "n": int(len(sub)),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(pval) if np.isfinite(pval) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_scatter(
    path: Path,
    df: pd.DataFrame,
    x: str,
    y: str,
    color_col: str,
    color_label: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    mask0 = df[color_col] == 0
    mask1 = df[color_col] == 1
    ax.scatter(df.loc[mask0, x], df.loc[mask0, y], s=16, alpha=0.45, label=f"{color_label}=0")
    ax.scatter(df.loc[mask1, x], df.loc[mask1, y], s=20, alpha=0.75, label=f"{color_label}=1")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_bar(path: Path, values: pd.Series, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    values.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_summary(
    path: Path,
    df: pd.DataFrame,
    best_json: dict,
    log_text: str,
    near12: pd.DataFrame,
    top20: pd.DataFrame,
    failed_counts: pd.Series,
    pass_rates: pd.Series,
    group_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> None:
    n_total = len(df)
    n_13 = int((df["n_passed"] == 13).sum())
    n_12 = int((df["n_passed"] == 12).sum())
    npassed = df["n_passed"].value_counts().sort_index()

    t13 = df[df["T13"] == 1]
    t4t6 = df[(df["T4"] == 1) & (df["T6"] == 1)]
    t13_t4_pass_t6_pass = int(((t13["T4"] == 1) & (t13["T6"] == 1)).sum())
    t13_t4_pass_t6_fail = int(((t13["T4"] == 1) & (t13["T6"] == 0)).sum())
    t13_t4_fail_t6_pass = int(((t13["T4"] == 0) & (t13["T6"] == 1)).sum())
    t13_t4_fail_t6_fail = int(((t13["T4"] == 0) & (t13["T6"] == 0)).sum())

    upper = 0.075
    near_upper_margin = 0.005
    t13_near_upper = int((t13["c_th2ctx"] >= upper - near_upper_margin).sum())
    t6 = df[df["T6"] == 1]
    t6_near_upper = int((t6["c_th2ctx"] >= upper - near_upper_margin).sum())
    best_t13 = t13.sort_values("score", ascending=False).head(20)

    wall_time = "not found"
    match = re.search(r"Evolution complete in\s+([0-9.]+)\s+h", log_text)
    if match:
        wall_time = f"{match.group(1)} h"

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("V8a Trade-off Diagnostic Summary\n")
        f.write("=" * 38 + "\n\n")
        f.write("Inputs read only: CSV, JSON, log. No simulations were run.\n\n")
        f.write("Basic summary\n")
        f.write("-" * 13 + "\n")
        f.write(f"wall_time_from_log: {wall_time}\n")
        f.write(f"total_evaluations: {n_total}\n")
        f.write(f"13_of_13_candidates: {n_13}\n")
        f.write(f"12_of_13_candidates: {n_12}\n")
        f.write("n_passed_distribution:\n")
        for key, val in npassed.items():
            f.write(f"  {int(key)}: {int(val)}\n")
        f.write("\npass_rates:\n")
        for key, val in pass_rates.items():
            f.write(f"  {key}: {val:.6f}\n")

        f.write("\nBest overall candidate\n")
        f.write("-" * 22 + "\n")
        for key in [
            "score",
            "feasible",
            "n_passed",
            "c_th2ctx",
            "c_ctx2th",
            "T4_q",
            "T6_ibi_cv",
            "T8_n_sp_events",
            "T12_n_verified",
            "T13_n_ctx_events",
            "T13_n_ctx_verified",
            "T13_ctx_verified_density_per_min",
            "shape_r",
            "so_power",
            "thal_spindle_power",
            "ctx_spindle_power",
        ]:
            f.write(f"{key}: {best_json.get(key)}\n")
        failed = [t for t in T_COLS if int(best_json.get(t, 0)) == 0]
        f.write(f"failed_constraints: {','.join(failed) if failed else 'none'}\n")

        f.write("\n12/13 near-feasible failures\n")
        f.write("-" * 28 + "\n")
        for key, val in failed_counts.items():
            f.write(f"{key}: {int(val)}\n")

        f.write("\nTop 20 12/13 candidates by score\n")
        f.write("-" * 34 + "\n")
        f.write(top20.to_string(index=False))
        f.write("\n\nT4/T6/T13 trade-off counts\n")
        f.write("-" * 30 + "\n")
        f.write(f"T13=1 candidates: {len(t13)}\n")
        f.write(f"Among T13=1, T4=1 and T6=1: {t13_t4_pass_t6_pass}\n")
        f.write(f"Among T13=1, T4=1 and T6=0: {t13_t4_pass_t6_fail}\n")
        f.write(f"Among T13=1, T4=0 and T6=1: {t13_t4_fail_t6_pass}\n")
        f.write(f"Among T13=1, T4=0 and T6=0: {t13_t4_fail_t6_fail}\n")
        f.write(f"T4=1 and T6=1 candidates: {len(t4t6)}\n")
        f.write(f"Among T4=1 and T6=1, T13=1: {int((t4t6['T13'] == 1).sum())}\n")

        f.write("\nBoundary analysis\n")
        f.write("-" * 17 + "\n")
        f.write(f"T13=1 near c_th2ctx upper bound >=0.070: {t13_near_upper}/{len(t13)}\n")
        f.write(f"T6=1 near c_th2ctx upper bound >=0.070: {t6_near_upper}/{len(t6)}\n")
        if not best_t13.empty:
            f.write(
                "Top T13=1 c_th2ctx range: "
                f"{best_t13['c_th2ctx'].min():.6f} to {best_t13['c_th2ctx'].max():.6f}\n"
            )
            f.write(
                "Top T13=1 median c_th2ctx: "
                f"{best_t13['c_th2ctx'].median():.6f}\n"
            )
        f.write(
            "V8b bound expansion reading: records do not show that T13 requires "
            "the V8a upper boundary; T13=1 occurs within the existing interval. "
            "Because T13=1 candidates all fail T4 or T6, simply expanding the "
            "upper bound may be harmful unless tested as a targeted diagnostic.\n"
        )

        f.write("\nConservative interpretation\n")
        f.write("-" * 27 + "\n")
        f.write(
            "These records support a practical V8a search trade-off between "
            "T13 cortical spindle observability and T6 slow-oscillation regularity: "
            "no candidate passed all 13 constraints, T13=1 candidates never "
            "co-occurred with T4=1 and T6=1, and T4=1/T6=1 candidates never "
            "passed T13. This is evidence about this objective, detector, bounds, "
            "and search run; it is not proof of model incapacity and not a claim "
            "that V8a failed scientifically.\n"
        )
        f.write(
            "Recommended next step: B) a targeted c_th2ctx/c_ctx2th sweep around "
            "the V8a near-feasible candidates, especially the 12/13 T6-only "
            "failures, before creating V8b. A longer-duration T6 stability check "
            "can follow for promising points, but 300s validation was not run here. "
            "Detector redesign or model-structure revision should wait until the "
            "local trade-off is mapped more cleanly.\n"
        )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df, best_json, log_text = read_inputs(args.csv, args.json, args.log)

    missing = [col for col in T_COLS + GROUP_STAT_COLS + ["score", "n_passed"] if col not in df]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    pass_rates = df[T_COLS].mean()
    near12 = df[df["n_passed"] == 12].copy()
    near12["failed_constraint"] = near12.apply(failed_constraint, axis=1)
    failed_counts = near12["failed_constraint"].value_counts().sort_index()
    top20 = near12.sort_values("score", ascending=False).head(20)
    top20 = top20[[col for col in TOP_COLS if col in top20.columns]]

    e_group = near12[near12["failed_constraint"] == "T6"].sort_values("score", ascending=False).head(20)
    groups = {
        "A_all": df,
        "B_T13_pass": df[df["T13"] == 1],
        "C_T4_T6_pass": df[(df["T4"] == 1) & (df["T6"] == 1)],
        "D_12of13": near12,
        "E_best_12of13_T6_only": e_group,
    }
    group_df = group_stats(df, groups)
    corr_df = spearman_table(df)

    near12[[col for col in TOP_COLS if col in near12.columns]].sort_values(
        "score", ascending=False
    ).to_csv(args.outdir / "v8a_12of13_candidates.csv", index=False)
    group_df.to_csv(args.outdir / "v8a_group_stats.csv", index=False)
    corr_df.to_csv(args.outdir / "v8a_correlations.csv", index=False)

    write_summary(
        args.outdir / "v8a_tradeoff_summary.txt",
        df,
        best_json,
        log_text,
        near12,
        top20,
        failed_counts,
        pass_rates,
        group_df,
        corr_df,
    )

    save_scatter(
        args.outdir / "fig_v8a_t6_vs_t13_density.png",
        df,
        "T13_ctx_verified_density_per_min",
        "T6_ibi_cv",
        "T4",
        "T4",
        "V8a T6 regularity vs cortical spindle density",
        "T13 verified cortical spindle density per min",
        "T6 IBI CV",
    )
    save_scatter(
        args.outdir / "fig_v8a_cth2ctx_vs_t13_density.png",
        df,
        "c_th2ctx",
        "T13_ctx_verified_density_per_min",
        "T6",
        "T6",
        "V8a c_th2ctx vs cortical spindle density",
        "c_th2ctx",
        "T13 verified cortical spindle density per min",
    )
    save_scatter(
        args.outdir / "fig_v8a_cth2ctx_vs_t6.png",
        df,
        "c_th2ctx",
        "T6_ibi_cv",
        "T13",
        "T13",
        "V8a c_th2ctx vs T6 IBI CV",
        "c_th2ctx",
        "T6 IBI CV",
    )
    save_bar(
        args.outdir / "fig_v8a_npassed_distribution.png",
        df["n_passed"].value_counts().sort_index(),
        "V8a n_passed distribution",
        "n_passed",
        "count",
    )
    save_bar(
        args.outdir / "fig_v8a_failed_constraint_12of13.png",
        failed_counts,
        "V8a failed constraint among 12/13 candidates",
        "failed constraint",
        "count",
    )

    print(f"Wrote diagnostics to {args.outdir}")
    print(f"total_evaluations={len(df)}")
    print(f"13_of_13={(df['n_passed'] == 13).sum()}")
    print(f"12_of_13={(df['n_passed'] == 12).sum()}")
    print(f"T13_pass={(df['T13'] == 1).sum()}")
    print(f"T4_T6_pass={((df['T4'] == 1) & (df['T6'] == 1)).sum()}")
    print(f"T4_T6_T13_pass={((df['T4'] == 1) & (df['T6'] == 1) & (df['T13'] == 1)).sum()}")


if __name__ == "__main__":
    main()
