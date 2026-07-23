"""Create V8a 0712 presentation figures for the best T6 near-miss candidate.

Figures:
1. T6 IBI-CV vs T13 cortical density with best0417 highlighted.
2. Coupling heatmap triptych around the representative base.
3. Relaxed-T6 sensitivity ladder.
4. Candidate-level V8a record verification panel.

This script does not change V8a logic and does not run DE or long validation.
It uses existing V8a diagnostic records for the candidate-level verification figure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.signal import butter, hilbert, sosfiltfilt, welch
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.s4_personalize_fig7_v8 import (  # noqa: E402
    FS_SIM,
    F_HI,
    F_LO,
    HAS_FOOOF,
    SPINDLE_DUR_HI_S,
    SPINDLE_DUR_LO_S,
    SPINDLE_ENV_SMOOTH_MS,
    SPINDLE_EVT_PCTILE,
    SPINDLE_HI,
    SPINDLE_LO,
    SUBJECT_ID,
    T12_PEAK_INSIDE_RATIO,
    T13_CTX_DUR_HI_S,
    T13_CTX_DUR_LO_S,
    T13_PEAK_INSIDE_RATIO,
    build_model,
    compute_constraints_v8,
    compute_epoch_psd,
    compute_target_periodic,
    load_target_psd,
    seed_numba,
)
from neurolib.models.multimodel.builder.base.constants import EXC  # noqa: E402


OUTDIR = ROOT / "outputs" / "v8a_0712_best0417_figures"
COUPLING_CSV = ROOT / "outputs" / "v8a_t6_t13_coupling_sweep.csv"
RELAXED_CSV = ROOT / "outputs" / "v8a_relaxed_t6_sensitivity" / "relaxed_t6_summary.csv"

PARAM_NAMES = ["mue", "mui", "b", "tauA", "g_LK", "g_h", "c_th2ctx", "c_ctx2th"]


def read_numeric(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() == df[col].notna().sum():
            df[col] = converted
    return df


def select_candidate() -> dict:
    relaxed = read_numeric(RELAXED_CSV)
    row = relaxed[relaxed["threshold"] == 0.42].iloc[0].to_dict()
    source = read_numeric(ROOT / row["best_source_file"])
    source_row = source.iloc[int(row["best_source_row_index"])].to_dict()
    failed = {
        item.strip()
        for item in str(source_row.get("failed_constraints", row.get("original_failed_constraints", ""))).split(",")
        if item.strip()
    }
    inferred_constraints = {
        f"T{i}": int(source_row[f"T{i}"])
        if f"T{i}" in source_row and pd.notna(source_row[f"T{i}"])
        else int(f"T{i}" not in failed)
        for i in range(1, 14)
    }
    return {p: float(row[p]) for p in PARAM_NAMES} | {
        "source_file": row["best_source_file"],
        "source_row_index": int(row["best_source_row_index"]),
        "n_passed": int(source_row["n_passed"]),
        "failed_constraints": str(source_row["failed_constraints"]),
        **inferred_constraints,
        "T4_q": float(row["T4_q"]),
        "T6_ibi_cv": float(row["T6_ibi_cv"]),
        "T13_density": float(row["T13_ctx_verified_density_per_min"]),
        "T13_verified": float(row["T13_n_ctx_verified"]),
        "T8_events": float(row["T8_n_sp_events"]),
        "T12_verified": float(row["T12_n_verified"]),
    }


def save_scatter_tradeoff(candidate: dict) -> Path:
    df = read_numeric(COUPLING_CSV)
    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=170)
    for t4_value, color, label, alpha in [
        (0, "#9aa0a6", "T4 fail", 0.30),
        (1, "#2c7fb8", "T4 pass", 0.45),
    ]:
        sub = df[df["T4"] == t4_value]
        ax.scatter(
            sub["T13_ctx_verified_density_per_min"],
            sub["T6_ibi_cv"],
            s=16,
            c=color,
            alpha=alpha,
            linewidths=0,
            label=label,
        )
    ax.scatter(
        [candidate["T13_density"]],
        [candidate["T6_ibi_cv"]],
        s=180,
        c="#d62728",
        edgecolors="black",
        linewidths=1.2,
        marker="*",
        label="V8a 0712 best T13-preserved near-miss",
        zorder=10,
    )
    ax.axhline(0.40, color="black", linestyle="-", linewidth=1.2, label="strict T6 threshold 0.40")
    ax.axhline(0.42, color="#d62728", linestyle="--", linewidth=1.1, label="post-hoc relaxed T6 0.42")
    ax.set_xlabel("T13 verified cortical spindle density / min")
    ax.set_ylabel("T6 IBI-CV")
    ax.set_title("V8a 0712: T6 regularity vs cortical spindle observability")
    ax.text(
        candidate["T13_density"] + 0.12,
        candidate["T6_ibi_cv"] + 0.01,
        "T6=0.417\nT13=1.091/min",
        fontsize=9,
        va="bottom",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = OUTDIR / "fig_v8a_0712_t6_vs_t13_density_highlight_best0417.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def pivot_for_base(df: pd.DataFrame, base_id: str, value: str) -> pd.DataFrame:
    sub = df[df["base_id"] == base_id].copy()
    sub["x"] = sub["c_th2ctx"].round(6)
    sub["y"] = sub["c_ctx2th"].round(6)
    return sub.groupby(["y", "x"])[value].max().unstack("x")


def save_heatmap_triptych(candidate: dict) -> Path:
    df = read_numeric(COUPLING_CSV)
    base_id = "base_02"
    values = [
        ("T6_ibi_cv", "T6 IBI-CV", "viridis"),
        ("T13_ctx_verified_density_per_min", "T13 verified density / min", "magma"),
        ("n_passed", "n_passed", "cividis"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=170, constrained_layout=True)
    for ax, (value, title, cmap) in zip(axes, values):
        piv = pivot_for_base(df, base_id, value)
        im = ax.imshow(
            piv.values,
            origin="lower",
            aspect="auto",
            extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()],
            cmap=cmap,
            interpolation="nearest",
        )
        if value == "T6_ibi_cv":
            xg, yg = np.meshgrid(piv.columns.values, piv.index.values)
            ax.contour(xg, yg, piv.values, levels=[0.40], colors="white", linewidths=1.1)
            ax.contour(xg, yg, piv.values, levels=[0.42], colors="#ffdd57", linewidths=1.1, linestyles="--")
        ax.scatter(
            [candidate["c_th2ctx"]],
            [candidate["c_ctx2th"]],
            s=95,
            marker="*",
            c="#d62728",
            edgecolors="black",
            linewidths=1.0,
            zorder=10,
        )
        ax.set_title(title)
        ax.set_xlabel("c_th2ctx")
        ax.set_ylabel("c_ctx2th")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("V8a 0712: coupling scan around best0417 near-miss", fontsize=13)
    path = OUTDIR / "fig_v8a_0712_coupling_heatmap_triptych_best0417.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_relaxed_ladder() -> Path:
    relaxed = read_numeric(RELAXED_CSV)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=170)
    x = np.arange(len(relaxed))
    width = 0.36
    ax.bar(
        x - width / 2,
        relaxed["relaxed_13of13_count"],
        width,
        label="relaxed 13/13 count",
        color="#2c7fb8",
    )
    ax.bar(
        x + width / 2,
        relaxed["T4_T13_relaxedT6_count"],
        width,
        label="T4=1, T13=1, relaxed T6 count",
        color="#f28e2b",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in relaxed["threshold"]])
    ax.set_xlabel("Post-hoc T6 IBI-CV threshold")
    ax.set_ylabel("candidate count")
    ax.set_title("V8a 0712: relaxed-T6 sensitivity (post-hoc only)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for xi, row in zip(x, relaxed.itertuples(index=False)):
        ax.text(xi - width / 2, row.relaxed_13of13_count + 0.12, str(int(row.relaxed_13of13_count)),
                ha="center", va="bottom", fontsize=9)
        ax.text(xi + width / 2, row.T4_T13_relaxedT6_count + 0.12, str(int(row.T4_T13_relaxedT6_count)),
                ha="center", va="bottom", fontsize=9)
    ax.text(
        0.02,
        0.95,
        "Strict V8a remains T6 < 0.40\n0.42 is sensitivity, not strict success",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    fig.tight_layout()
    path = OUTDIR / "fig_v8a_0712_relaxed_t6_sensitivity_ladder.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def detect_sigma_events(signal: np.ndarray, fs: float, dur_lo: float, dur_hi: float, ratio: float):
    sos = butter(4, [SPINDLE_LO, SPINDLE_HI], btype="band", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, signal)
    envelope = np.abs(hilbert(filtered))
    env_smooth = gaussian_filter1d(envelope, sigma=SPINDLE_ENV_SMOOTH_MS * fs / 1000.0)
    thresh = np.percentile(env_smooth, SPINDLE_EVT_PCTILE)
    above = (env_smooth > thresh).astype(np.int8)
    diff = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    durations = (ends - starts) / fs
    valid = (durations >= dur_lo) & (durations <= dur_hi)
    events = []
    verified = []
    for s, e, d in zip(starts[valid], ends[valid], durations[valid]):
        events.append((int(s), int(e), float(d)))
        segment = signal[s:e]
        if len(segment) < int(0.2 * fs):
            continue
        f_ev, p_ev = welch(segment, fs=fs, nperseg=min(len(segment), 512))
        sp_m = (f_ev >= SPINDLE_LO) & (f_ev <= SPINDLE_HI)
        ns_m = (f_ev >= 4.0) & (f_ev < SPINDLE_LO)
        if sp_m.any() and ns_m.any():
            if float(p_ev[sp_m].max()) > ratio * float(p_ev[ns_m].max()):
                verified.append((int(s), int(e), float(d)))
    return filtered, env_smooth, thresh, events, verified


def simulate_candidate(candidate: dict):
    model = build_model(
        candidate["mue"],
        candidate["mui"],
        candidate["b"],
        candidate["tauA"],
        candidate["g_LK"],
        candidate["g_h"],
        candidate["c_th2ctx"],
        candidate["c_ctx2th"],
        duration=60_000,
    )
    seed_numba(42)
    model.run()
    r_exc = model[f"r_mean_{EXC}"]
    if r_exc.ndim == 2 and r_exc.shape[0] >= 2:
        r_ctx = r_exc[0, :] * 1000.0
        r_thal = r_exc[1, :] * 1000.0
    else:
        r_ctx = np.asarray(r_exc).squeeze() * 1000.0
        r_thal = np.zeros_like(r_ctx)
    n_drop = int(5 * FS_SIM)
    r_ctx = r_ctx[n_drop:]
    r_thal = r_thal[n_drop:]
    f_ctx, p_ctx = compute_epoch_psd(r_ctx, FS_SIM)
    n_passed, details = compute_constraints_v8(r_ctx, r_thal, f_c=f_ctx, p_c=p_ctx, fs=FS_SIM)
    return r_ctx, r_thal, f_ctx, p_ctx, n_passed, details


def save_fig7_style_standard_plots(candidate: dict) -> list[Path]:
    target_psd, target_freqs = load_target_psd()
    target_periodic, fooof_freqs = compute_target_periodic(target_psd, target_freqs)
    r_ctx, r_thal, f_ctx, p_ctx, _n_passed, _details = simulate_candidate(candidate)
    t = np.arange(len(r_ctx)) / FS_SIM

    paths: list[Path] = []

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), dpi=170, sharex=True)
    t0, t1 = 10.0, 26.0
    mask_t = (t >= t0) & (t <= t1)
    axes[0].plot(t[mask_t], r_ctx[mask_t], color="#534AB7", linewidth=0.8)
    axes[0].axhline(1.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
    axes[0].set_ylabel("r_ctx (Hz)")
    axes[0].set_title("Cortex EXC slow oscillation")
    axes[0].grid(True, alpha=0.2)
    axes[1].plot(t[mask_t], r_thal[mask_t], color="#1D9E75", linewidth=0.8)
    axes[1].set_ylabel("r_thal (Hz)")
    axes[1].set_xlabel("Time after 5 s burn-in (s)")
    axes[1].set_title("Thalamic TCR spindle activity")
    axes[1].grid(True, alpha=0.2)
    fig.suptitle(
        "Fig7-style V8a 0712 best0417 timeseries visualization\n"
        f"T4=1, T13=1, strict T6 near-miss: T6 IBI-CV={candidate['T6_ibi_cv']:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUTDIR / "fig7_v8a_0712_best0417_timeseries.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8), dpi=170, sharex=True)
    axes[0].semilogy(target_freqs, target_psd, color="black", linewidth=1.6, label=f"Target EEG N3 ({SUBJECT_ID})")
    axes[0].axvspan(0.2, 1.5, alpha=0.12, color="orange", label="SO band")
    axes[0].axvspan(SPINDLE_LO, SPINDLE_HI, alpha=0.12, color="green", label="sigma band")
    axes[0].set_ylabel("Power")
    axes[0].set_title("Target EEG spectrum")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, alpha=0.2)
    mask = (f_ctx >= F_LO) & (f_ctx <= F_HI)
    axes[1].semilogy(f_ctx[mask], p_ctx[mask], color="#534AB7", linewidth=1.4, label="Simulated cortex PSD")
    axes[1].axvspan(0.2, 1.5, alpha=0.12, color="orange")
    axes[1].axvspan(SPINDLE_LO, SPINDLE_HI, alpha=0.12, color="green")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power")
    axes[1].set_xlim(F_LO, F_HI)
    axes[1].set_title("V8a 0712 best0417 visualization-run cortex spectrum")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, alpha=0.2)
    fig.suptitle("Fig7-style V8a 0712 spectra", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = OUTDIR / "fig7_v8a_0712_best0417_spectra.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    if HAS_FOOOF and target_periodic is not None and fooof_freqs is not None:
        from fooof import FOOOF

        p_interp = interp1d(f_ctx, p_ctx, bounds_error=False, fill_value=1e-30)(fooof_freqs)
        fm_sim = FOOOF(
            peak_width_limits=[1.0, 8.0],
            max_n_peaks=4,
            min_peak_height=0.05,
            aperiodic_mode="fixed",
        )
        fm_sim.fit(fooof_freqs, p_interp, [F_LO, F_HI])
        sim_log = np.log10(p_interp[: len(fm_sim._ap_fit)] + 1e-30)
        sim_periodic = sim_log - fm_sim._ap_fit
        n_r = min(len(sim_periodic), len(target_periodic), len(fooof_freqs))
        residual_r, _ = pearsonr(sim_periodic[:n_r], target_periodic[:n_r])

        fig, ax = plt.subplots(figsize=(10, 5.2), dpi=170)
        ax.plot(fooof_freqs[:n_r], target_periodic[:n_r], color="black", linewidth=1.8,
                label="Target EEG periodic residual")
        ax.plot(fooof_freqs[:n_r], sim_periodic[:n_r], color="#534AB7", linewidth=1.8, linestyle="--",
                label="Simulated cortex periodic residual")
        ax.axhline(0.0, color="gray", linewidth=0.7, alpha=0.6)
        ax.axvspan(0.2, 1.5, alpha=0.12, color="orange", label="SO band")
        ax.axvspan(SPINDLE_LO, SPINDLE_HI, alpha=0.12, color="green", label="sigma band")
        for label, freq, color in [
            ("delta", 2.4, "orange"),
            ("theta", 6.2, "purple"),
            ("alpha", 9.9, "blue"),
            ("sigma", 12.5, "green"),
        ]:
            ax.axvline(freq, color=color, linewidth=0.8, linestyle=":", alpha=0.7)
            ax.text(freq + 0.1, ax.get_ylim()[1] * 0.88, label, fontsize=8, color=color, rotation=90)
        ax.set_xlim(F_LO, F_HI)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Log-domain periodic residual")
        ax.set_title(
            "Fig7-style V8a 0712 FOOOF residuals\n"
            f"visualization-run Pearson r={residual_r:.3f}; strict result remains 12/13 failed T6"
        )
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        path = OUTDIR / "fig7_v8a_0712_best0417_residuals.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

        residual_metrics = {
            "source": "visualization_run_not_de",
            "pearson_r": float(residual_r),
            "note": "This residual plot is for Fig7-style visualization only; candidate-level pass/fail uses existing coupling sweep record.",
        }
        (OUTDIR / "fig7_v8a_0712_best0417_residual_metrics.json").write_text(
            json.dumps(residual_metrics, indent=2), encoding="utf-8"
        )

    return paths


def save_candidate_verification(candidate: dict) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=170)
    axes = axes.ravel()

    ax = axes[0]
    labels = [f"T{i}" for i in range(1, 14)]
    values = [int(candidate.get(label, 0)) for label in labels]
    colors = ["#2ca02c" if v == 1 else "#d62728" for v in values]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("pass")
    ax.set_title("V8a 0712 candidate-level verification: 12/13, failed T6")
    ax.set_yticks([0, 1])
    for i, (label, value) in enumerate(zip(labels, values)):
        ax.text(i, value + 0.04, "PASS" if value else "FAIL", ha="center", fontsize=7, rotation=90)
    ax.grid(True, axis="y", alpha=0.2)

    ax = axes[1]
    metric_names = ["T4_q", "T6_ibi_cv", "T13 density/min"]
    metric_values = [candidate["T4_q"], candidate["T6_ibi_cv"], candidate["T13_density"]]
    thresholds = [2.0, 0.40, 1.0]
    metric_colors = ["#2ca02c", "#d62728", "#2ca02c"]
    ax.bar(metric_names, metric_values, color=metric_colors, alpha=0.85)
    ax.scatter(metric_names, thresholds, color="black", marker="_", s=650, label="strict threshold/reference")
    ax.axhline(0.42, color="#d62728", linestyle="--", linewidth=1.0, label="post-hoc T6=0.42")
    ax.set_title("Core V8a audit metrics")
    ax.set_ylabel("value")
    ax.text(1, candidate["T6_ibi_cv"] + 0.05, "0.417\nstrict fail\n0.42 sensitivity pass",
            ha="center", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)

    ax = axes[2]
    event_labels = ["T8 thal events", "T12 thal verified", "T13 ctx verified"]
    event_values = [candidate["T8_events"], candidate["T12_verified"], candidate["T13_verified"]]
    ax.bar(event_labels, event_values, color=["#9467bd", "#2ca02c", "#ff7f0e"])
    ax.set_title("Spindle event reality and cortical observability")
    ax.set_ylabel("count")
    for i, v in enumerate(event_values):
        ax.text(i, v + 0.15, f"{v:.0f}", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)

    ax = axes[3]
    ax.axis("off")
    text = (
        "V8a 0712 best T13-preserved near-miss\n"
        f"source: {candidate['source_file']} row {candidate['source_row_index']}\n\n"
        f"n_passed = {candidate['n_passed']}/13\n"
        f"failed_constraints = {candidate['failed_constraints']}\n\n"
        f"c_th2ctx = {candidate['c_th2ctx']:.6f}\n"
        f"c_ctx2th = {candidate['c_ctx2th']:.3f}\n"
        f"T4_q = {candidate['T4_q']:.3f}\n"
        f"T6_ibi_cv = {candidate['T6_ibi_cv']:.3f}\n"
        f"T13 verified density = {candidate['T13_density']:.3f}/min\n\n"
        "T1-T13 bars use recorded failed_constraints where\n"
        "the sweep CSV did not store every binary T column.\n\n"
        "Interpretation: strict V8a near-miss; only T6 fails.\n"
        "The 0.42 line is post-hoc sensitivity, not strict V8a success."
    )
    ax.text(0.02, 0.98, text, va="top", fontsize=10, family="monospace")

    fig.tight_layout()
    path = OUTDIR / "fig_v8a_0712_best0417_candidate_verification.png"
    fig.savefig(path)
    plt.close(fig)

    metrics = {
        "source": "existing_v8a_coupling_sweep_record",
        "n_passed": int(candidate["n_passed"]),
        "failed_constraints": candidate["failed_constraints"],
        "T_constraints": {f"T{i}": int(candidate.get(f"T{i}", 0)) for i in range(1, 14)},
        "T4_q": candidate["T4_q"],
        "T6_ibi_cv": candidate["T6_ibi_cv"],
        "T8_n_sp_events": candidate["T8_events"],
        "T12_n_verified": candidate["T12_verified"],
        "T13_n_ctx_verified": candidate["T13_verified"],
        "T13_ctx_verified_density_per_min": candidate["T13_density"],
        "c_th2ctx": candidate["c_th2ctx"],
        "c_ctx2th": candidate["c_ctx2th"],
    }
    (OUTDIR / "v8a_0712_best0417_candidate_verification_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    candidate = select_candidate()
    paths = [
        save_scatter_tradeoff(candidate),
        save_heatmap_triptych(candidate),
        save_relaxed_ladder(),
        save_candidate_verification(candidate),
    ]
    paths.extend(save_fig7_style_standard_plots(candidate))
    manifest = {
        "label": "V8a 0712 best0417 presentation figures",
        "candidate": candidate,
        "figures": [str(p.relative_to(ROOT)) for p in paths],
    }
    (OUTDIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Generated figures:")
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
