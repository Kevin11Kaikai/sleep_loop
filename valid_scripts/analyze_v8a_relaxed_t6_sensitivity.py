"""Post-hoc relaxed-T6 sensitivity analysis for V8a records.

This script does not run simulations. It reads existing V8a diagnostic CSVs and
asks whether near-miss candidates would become feasible if only the T6 IBI-CV
threshold were relaxed post hoc.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILES = [
    ROOT / "outputs" / "evolution_fig7_v8_records.csv",
    ROOT / "outputs" / "v8a_t6_t13_coupling_sweep.csv",
    ROOT / "outputs" / "v8a_ultra_narrow_t6_t13_search" / "ultra_narrow_records.csv",
    ROOT / "outputs" / "v8a_local_de_t6_rescue_long" / "local_de_records.csv",
    ROOT / "outputs" / "v8a_local_de_t6_rescue_narrow" / "narrow_local_de_records.csv",
    ROOT / "outputs" / "v8a_coupling_sweep_long" / "phase1_cth2ctx_sweep.csv",
    ROOT / "outputs" / "v8a_coupling_sweep_long" / "phase2_coupling_2d_sweep.csv",
]
OUTDIR = ROOT / "outputs" / "v8a_relaxed_t6_sensitivity"
SUMMARY_CSV = OUTDIR / "relaxed_t6_summary.csv"
BEST_CSV = OUTDIR / "relaxed_t6_best_candidates.csv"
INTERPRETATION_MD = OUTDIR / "relaxed_t6_interpretation.md"

THRESHOLDS = [0.40, 0.42, 0.45, 0.50]
T_COLS = [f"T{i}" for i in range(1, 14)]
REQUIRED = [
    "n_passed",
    "failed_constraints",
    "T4",
    "T4_q",
    "T6",
    "T6_ibi_cv",
    "T13",
    "T13_n_ctx_verified",
    "T13_ctx_verified_density_per_min",
    "T8_n_sp_events",
    "T12_n_verified",
    "c_th2ctx",
    "c_ctx2th",
    "mue",
    "mui",
    "b",
    "tauA",
    "g_LK",
    "g_h",
]
SUMMARY_FIELDS = [
    "threshold",
    "relaxed_13of13_count",
    "T4_T13_relaxedT6_count",
    "best_status",
    "best_source_file",
    "best_source_row_index",
    "original_n_passed",
    "original_failed_constraints",
    "T4",
    "T4_q",
    "original_T6",
    "T6_ibi_cv",
    "threshold_margin",
    "T13",
    "T13_n_ctx_verified",
    "T13_ctx_verified_density_per_min",
    "T8_n_sp_events",
    "T12_n_verified",
    "c_th2ctx",
    "c_ctx2th",
    "mue",
    "mui",
    "b",
    "tauA",
    "g_LK",
    "g_h",
]


def read_csv_numeric(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() == df[col].notna().sum():
            df[col] = converted
    return df


def load_all() -> tuple[pd.DataFrame, list[Path], list[Path]]:
    frames = []
    used = []
    missing = []
    for path in INPUT_FILES:
        if not path.exists():
            missing.append(path)
            continue
        df = read_csv_numeric(path)
        df["source_file"] = str(path.relative_to(ROOT))
        df["source_row_index"] = np.arange(len(df))
        frames.append(df)
        used.append(path)
    if not frames:
        raise RuntimeError("No input CSV files found.")
    data = pd.concat(frames, ignore_index=True, sort=False)
    for col in T_COLS + REQUIRED:
        if col not in data.columns:
            data[col] = np.nan
    for col in T_COLS + ["T4", "T6", "T13"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ["n_passed", "T4_q", "T6_ibi_cv", "T13_ctx_verified_density_per_min"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data, used, missing


def all_available_except_t6_pass(row: pd.Series) -> bool:
    for col in T_COLS:
        if col == "T6":
            continue
        value = row.get(col, np.nan)
        if pd.isna(value):
            continue
        if int(float(value)) != 1:
            return False
    return True


def add_relaxed_columns(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = data.copy()
    out["relaxed_T6"] = out["T6_ibi_cv"] < threshold
    out["all_available_except_T6"] = out.apply(all_available_except_t6_pass, axis=1)
    out["relaxed_13of13"] = (
        (out["T4"] == 1)
        & (out["T13"] == 1)
        & out["all_available_except_T6"]
        & out["relaxed_T6"]
    )
    out["T4_T13_relaxedT6"] = (
        (out["T4"] == 1)
        & (out["T13"] == 1)
        & out["relaxed_T6"]
    )
    out["threshold"] = threshold
    out["threshold_margin"] = threshold - out["T6_ibi_cv"]
    return out


def choose_best(df: pd.DataFrame, threshold: float) -> tuple[str, pd.Series | None]:
    feasible = df[df["relaxed_13of13"]].copy()
    if not feasible.empty:
        feasible = feasible.sort_values(
            [
                "n_passed",
                "T6_ibi_cv",
                "T13_ctx_verified_density_per_min",
                "T4_q",
                "T12_n_verified",
                "T8_n_sp_events",
            ],
            ascending=[False, True, False, False, False, False],
        )
        return "relaxed_feasible", feasible.iloc[0]

    near = df[(df["T4"] == 1) & (df["T13"] == 1)].copy()
    if near.empty:
        return "closest_near_miss", None
    near = near.sort_values(
        [
            "T6_ibi_cv",
            "n_passed",
            "T13_ctx_verified_density_per_min",
            "T4_q",
            "T12_n_verified",
            "T8_n_sp_events",
        ],
        ascending=[True, False, False, False, False, False],
    )
    return "closest_near_miss", near.iloc[0]


def summary_row(threshold: float, df: pd.DataFrame) -> tuple[dict, pd.Series | None]:
    status, best = choose_best(df, threshold)
    row = {
        "threshold": threshold,
        "relaxed_13of13_count": int(df["relaxed_13of13"].sum()),
        "T4_T13_relaxedT6_count": int(df["T4_T13_relaxedT6"].sum()),
        "best_status": status,
    }
    if best is None:
        for key in SUMMARY_FIELDS:
            row.setdefault(key, "")
        return row, None
    row.update(
        {
            "best_source_file": best.get("source_file", ""),
            "best_source_row_index": best.get("source_row_index", ""),
            "original_n_passed": best.get("n_passed", ""),
            "original_failed_constraints": best.get("failed_constraints", ""),
            "T4": best.get("T4", ""),
            "T4_q": best.get("T4_q", ""),
            "original_T6": best.get("T6", ""),
            "T6_ibi_cv": best.get("T6_ibi_cv", ""),
            "threshold_margin": threshold - best.get("T6_ibi_cv", np.nan),
            "T13": best.get("T13", ""),
            "T13_n_ctx_verified": best.get("T13_n_ctx_verified", ""),
            "T13_ctx_verified_density_per_min": best.get(
                "T13_ctx_verified_density_per_min", ""
            ),
            "T8_n_sp_events": best.get("T8_n_sp_events", ""),
            "T12_n_verified": best.get("T12_n_verified", ""),
            "c_th2ctx": best.get("c_th2ctx", ""),
            "c_ctx2th": best.get("c_ctx2th", ""),
            "mue": best.get("mue", ""),
            "mui": best.get("mui", ""),
            "b": best.get("b", ""),
            "tauA": best.get("tauA", ""),
            "g_LK": best.get("g_LK", ""),
            "g_h": best.get("g_h", ""),
        }
    )
    return row, best


def write_interpretation(
    used: list[Path],
    missing: list[Path],
    summary: pd.DataFrame,
    best_rows: pd.DataFrame,
) -> None:
    lines = [
        "# V8a Relaxed-T6 Sensitivity Analysis",
        "",
        "This is a read-only post-hoc sensitivity check. No simulations, DE, V8b, threshold changes, or T13 detector changes were run.",
        "",
        "Strict V8a uses `T6_ibi_cv < 0.40`. The relaxed thresholds tested here are post-hoc sensitivity checks only:",
        "",
        "- `T6_ibi_cv < 0.40`",
        "- `T6_ibi_cv < 0.42`",
        "- `T6_ibi_cv < 0.45`",
        "- `T6_ibi_cv < 0.50`",
        "",
        "A relaxed-feasible candidate must still pass T4, T13, and every available T constraint except T6. Only T6 is relaxed.",
        "",
        "## Input Files Used",
        "",
    ]
    for path in used:
        lines.append(f"- `{path.relative_to(ROOT)}`")
    lines.extend(["", "## Missing Input Files", ""])
    if missing:
        for path in missing:
            lines.append(f"- `{path.relative_to(ROOT)}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Summary", ""])
    for _, row in summary.iterrows():
        threshold = row["threshold"]
        count = int(row["relaxed_13of13_count"])
        status = row["best_status"]
        t6_cv = row["T6_ibi_cv"]
        margin = row["threshold_margin"]
        lines.append(
            f"- Threshold `{threshold:.2f}`: relaxed 13/13 count = `{count}`, "
            f"selected status = `{status}`, selected T6_ibi_cv = `{t6_cv:.3f}`, "
            f"margin = `{margin:.3f}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If a threshold such as `0.42` produces a relaxed-feasible candidate, it should be described as a relaxed-T6 sensitivity result, not strict V8a success.",
            "",
            "This analysis does not claim that the model is fully validated. It also does not claim that V8a succeeded under the original strict criteria.",
            "",
            "The main V8a T6 threshold should not be changed based on this table alone. Any future threshold change would require explicit scientific justification and separate approval.",
            "",
        ]
    )
    INTERPRETATION_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data, used, missing = load_all()
    summary_rows = []
    best_candidates = []
    for threshold in THRESHOLDS:
        relaxed = add_relaxed_columns(data, threshold)
        row, best = summary_row(threshold, relaxed)
        summary_rows.append(row)
        if best is not None:
            best_dict = best.to_dict()
            best_dict["threshold"] = threshold
            best_dict["best_status"] = row["best_status"]
            best_dict["threshold_margin"] = row["threshold_margin"]
            best_candidates.append(best_dict)
    summary = pd.DataFrame(summary_rows)
    summary = summary[[c for c in SUMMARY_FIELDS if c in summary.columns]]
    best_df = pd.DataFrame(best_candidates)

    summary.to_csv(SUMMARY_CSV, index=False)
    best_df.to_csv(BEST_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    write_interpretation(used, missing, summary, best_df)

    print("Used input files:")
    for path in used:
        print(f"  {path.relative_to(ROOT)}")
    print("Missing input files:")
    if missing:
        for path in missing:
            print(f"  {path.relative_to(ROOT)}")
    else:
        print("  none")
    print("\nSummary table:")
    print(summary.to_string(index=False))
    print("\nMarkdown interpretation:")
    print(INTERPRETATION_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
