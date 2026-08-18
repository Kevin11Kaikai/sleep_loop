from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt, welch


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
SOURCE = WORKSPACE / "S4_sbi" / "DE_holdout_audit" / "scripts" / "run_frozen_validation.py"
SEEDS = (11003, 22007, 33013)
DOSES = (-0.0500, -0.0350, -0.0200, -0.0075, 0.0, 0.0075, 0.0200, 0.0350, 0.0500)
ACTIVE_DOSES = tuple(d for d in DOSES if d != 0.0)
TARGETS = (
    ("T1", "so_peak_frequency_hz", "Oscillation frequency / entrainment"),
    ("T2", "so_rms_amplitude", "Slow-oscillation amplitude"),
    ("T3", "so_density_per_min", "Slow-oscillation density"),
    ("T4", "spindle_density_per_min", "Spindle density"),
    ("T5", "sigma_power", "Spindle amplitude / power"),
    ("T6", "coupling_strength", "SO-spindle coupling strength"),
    ("T7", "preferred_phase_rad", "Preferred phase / timing"),
    ("T8", "state_transitions_per_min", "Dynamical regime / state transition"),
)
PARAMS = {
    "mue": 2.5543399414050265,
    "mui": 4.638376924416662,
    "b": 39.7650205359397,
    "tauA": 4305.440791463454,
    "g_LK": 0.051187832154447256,
    "g_h": 0.15302235700635436,
    "c_th2ctx": 0.09928570402727765,
    "c_ctx2th": 0.05873748566621345,
}
PROTOCOL = {
    "simulation": {
        "backend_primary": "numba",
        "integration_dt_ms": 0.1,
        "sampling_dt_ms": 1.0,
    }
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    with (ROOT / "RUN_LOG.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def load_science_module():
    spec = importlib.util.spec_from_file_location("costa_public_dev_runner", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import scientific runner: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def set_external_current(model, dose: float) -> None:
    keys = [key for key in model.params if key.endswith("ALNMassEXC_0.ext_exc_current")]
    if len(keys) != 1:
        raise RuntimeError(f"Expected one cortical external-current key, got {keys}")
    model.params[keys[0]] = float(dose)
    model._update_model_params()


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def metrics(x: np.ndarray, fs: float = 1000.0) -> dict:
    x = np.asarray(x, dtype=float)
    centered = x - np.mean(x)
    f, p = welch(centered, fs=fs, window="hann", nperseg=min(len(x), int(8 * fs)), noverlap=int(4 * fs))
    so_mask = (f >= 0.2) & (f <= 1.5)
    sigma_mask = (f >= 11.0) & (f <= 15.0)
    so = sosfiltfilt(butter(4, [0.2, 1.5], btype="bandpass", fs=fs, output="sos"), centered)
    sigma = sosfiltfilt(butter(4, [11.0, 15.0], btype="bandpass", fs=fs, output="sos"), centered)
    phase = np.angle(hilbert(so))
    env = np.abs(hilbert(sigma))
    vector = np.sum(env * np.exp(1j * phase)) / max(float(np.sum(env)), 1e-12)
    duration_min = len(x) / fs / 60.0
    troughs, _ = find_peaks(-so, distance=max(1, int(0.5 * fs)), prominence=max(0.25 * np.std(so), 1e-12))
    rms = np.sqrt(np.convolve(sigma * sigma, np.ones(int(0.2 * fs)) / int(0.2 * fs), mode="same"))
    threshold = float(np.mean(rms) + 1.5 * np.std(rms))
    spindle_count = sum(1 for a, b in runs(rms > threshold) if 0.5 <= (b - a) / fs <= 3.0)
    smooth = np.convolve(centered, np.ones(int(0.25 * fs)) / int(0.25 * fs), mode="same")
    low, high = np.quantile(smooth, [0.35, 0.65])
    state = 0
    transitions = 0
    for value in smooth:
        new_state = 1 if value > high else (-1 if value < low else state)
        if state and new_state != state:
            transitions += 1
        state = new_state
    return {
        "so_peak_frequency_hz": float(f[so_mask][np.argmax(p[so_mask])]),
        "so_rms_amplitude": float(np.sqrt(np.mean(so * so))),
        "so_density_per_min": float(len(troughs) / duration_min),
        "spindle_density_per_min": float(spindle_count / duration_min),
        "sigma_power": float(np.trapezoid(p[sigma_mask], f[sigma_mask])),
        "coupling_strength": float(abs(vector)),
        "preferred_phase_rad": float(np.angle(vector)),
        "state_transitions_per_min": float(transitions / duration_min),
    }


def simulate(science, seed: int, dose: float, duration_ms: int, burn_s: float) -> dict:
    np.random.seed(seed)
    science.seed_numba(seed)
    model = science.build_model(PARAMS, seed, duration_ms, PROTOCOL)
    set_external_current(model, dose)
    started = time.time()
    model.run()
    r_exc = np.asarray(model["r_mean_EXC"], dtype=float)
    if r_exc.ndim != 2 or r_exc.shape[0] < 2:
        raise RuntimeError(f"Unexpected EXC shape {r_exc.shape}")
    drop = int(round(burn_s * 1000.0))
    cortex = r_exc[0, drop:] * 1000.0
    thalamus = r_exc[1, drop:] * 1000.0
    finite = bool(np.isfinite(cortex).all() and np.isfinite(thalamus).all())
    max_abs = float(max(np.max(np.abs(cortex)), np.max(np.abs(thalamus)))) if finite else float("inf")
    stable = bool(finite and max_abs < 500.0 and np.ptp(cortex) > 1e-8)
    if not stable:
        raise RuntimeError(f"Numerical stability failure: finite={finite}, max_abs={max_abs}")
    return {
        "seed": seed,
        "dose_mV_per_ms": dose,
        "condition": "SHAM" if dose == 0.0 else "ACTIVE",
        "numerically_stable": stable,
        "cortical_min_hz": float(np.min(cortex)),
        "cortical_max_hz": float(np.max(cortex)),
        "thalamic_min_hz": float(np.min(thalamus)),
        "thalamic_max_hz": float(np.max(thalamus)),
        "elapsed_s": float(time.time() - started),
        **metrics(cortex),
    }


def write_rows(rows: list[dict]) -> None:
    path = ROOT / "mvp_phase2a_simulations.csv"
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def circular_mean(values: list[float]) -> float:
    return float(np.angle(np.mean(np.exp(1j * np.asarray(values)))))


def build_response(rows: list[dict]) -> tuple[list[dict], np.ndarray]:
    matrix = np.full((8, 8), np.nan)
    output = []
    sham_rows = [r for r in rows if r["dose_mV_per_ms"] == 0.0]
    for i, (target, metric, description) in enumerate(TARGETS):
        sham_values = [float(r[metric]) for r in sham_rows]
        sham = circular_mean(sham_values) if target == "T7" else float(np.mean(sham_values))
        item = {"target": target, "metric": metric, "description": description, "sham_mean": sham}
        for j, dose in enumerate(ACTIVE_DOSES):
            values = [float(r[metric]) for r in rows if r["dose_mV_per_ms"] == dose]
            if len(values) != len(SEEDS):
                continue
            active = circular_mean(values) if target == "T7" else float(np.mean(values))
            if target == "T7":
                response = float(math.atan2(math.sin(active - sham), math.cos(active - sham)))
            else:
                response = float((active - sham) / max(abs(sham), 1e-12))
            matrix[i, j] = response
            item[f"dose_{dose:+.4f}"] = response
        output.append(item)
    return output, matrix


def write_quicklook(response_rows: list[dict], matrix: np.ndarray) -> None:
    with (ROOT / "mvp_phase2a_quicklook.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(response_rows[0]))
        writer.writeheader()
        writer.writerows(response_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6.5))
    finite = np.abs(matrix[np.isfinite(matrix)])
    vmax = max(float(np.quantile(finite, 0.95)) if finite.size else 1.0, 1e-6)
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(8), [f"{d:+.4f}" for d in ACTIVE_DOSES], rotation=35, ha="right")
    ax.set_yticks(range(8), [t[0] for t in TARGETS])
    ax.set_xlabel("Native cortical external-current dose (mV/ms)")
    ax.set_ylabel("Target metric")
    ax.set_title("COSTA Phase 2A exploratory response screen\nrelative change vs sham; T7 is wrapped phase shift (rad)")
    fig.colorbar(im, ax=ax, label="response vs sham")
    fig.tight_layout()
    fig.savefig(ROOT / "mvp_phase2a_quicklook.png", dpi=170)
    plt.close(fig)


def write_summary(rows: list[dict], matrix: np.ndarray, elapsed: float) -> None:
    strongest = []
    for i, (target, metric, description) in enumerate(TARGETS):
        j = int(np.nanargmax(np.abs(matrix[i])))
        strongest.append(f"- {target} — {description}: largest coarse response {matrix[i, j]:+.4g} at dose {ACTIVE_DOSES[j]:+.4f} mV/ms")
    text = f"""# COSTA Phase 2A Fast Exploratory MVP

**Evidence ceiling:** EXPLORATORY_ONLY. These are model-internal, preliminary screening results. They do not support C1, C2, clinical, confirmatory, dissertation-final, efficacy, safety, or prescribing claims.

## Execution

- Candidate: accessible frozen development candidate `V1`; no claim that it is the prohibited stopped-lineage M1 judgment.
- Model: neurolib thalamocortical model, native cortical `ext_exc_current` control channel.
- Grid: sham plus 8 prospectively fixed active doses.
- Seeds: {list(SEEDS)}.
- Planned/completed simulations: {len(SEEDS) * len(DOSES)} / {len(rows)}.
- Analysis: 5 s burn-in plus 30 s exploratory window per simulation.
- Numerical stability: {sum(bool(r['numerically_stable']) for r in rows)} / {len(rows)} passed finite/range/non-flat checks.
- Wall time: {elapsed:.1f} s.

## First-pass response screen

Responses in `mvp_phase2a_quicklook.csv` are relative changes from the three-seed sham mean, except T7, which is a wrapped phase shift in radians.

{chr(10).join(strongest)}

## Interpretation limits

The grid is intentionally coarse, the analysis windows are short, and no protected, Night-2, Sealed Bank, fresh-final, or confirmatory payload was used. Phase 2B was not executed. Any apparent dose-response pattern is hypothesis-generating only and requires independent development-only replication before stronger interpretation.
"""
    (ROOT / "MVP_SUMMARY.md").write_text(text, encoding="utf-8")


def preflight() -> int:
    science = load_science_module()
    assert SOURCE.is_file()
    assert len(TARGETS) == 8 and len(ACTIVE_DOSES) == 8
    assert len(SEEDS) >= 3 and len(PARAMS) == 8
    assert all(np.isfinite(list(PARAMS.values())))
    assert callable(science.build_model) and callable(science.seed_numba)
    print(json.dumps({"PREFLIGHT_OK": True, "planned_simulations": len(SEEDS) * len(DOSES), "table_shape": [8, 8]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--duration-ms", type=int, default=35000)
    parser.add_argument("--burn-s", type=float, default=5.0)
    args = parser.parse_args()
    if args.preflight:
        return preflight()
    started = time.time()
    status = {
        "schema_version": "1.0",
        "evidence_ceiling": "EXPLORATORY_ONLY",
        "phase": "2A",
        "phase2b_executed": False,
        "execution_started": True,
        "status": "RUNNING",
        "pid": os.getpid(),
        "started_utc": utc_now(),
        "planned_simulations": len(SEEDS) * len(DOSES),
        "completed_simulations": 0,
        "candidate": "V1_ACCESSIBLE_FROZEN_DEVELOPMENT_CANDIDATE_NOT_ASSERTED_AS_M1",
        "prohibited_payloads_used": False,
        "log_path": str(ROOT / "RUN_LOG.txt"),
    }
    atomic_json(ROOT / "mvp_phase2a_status.json", status)
    log(f"EXECUTION_STARTED=true PID={os.getpid()} planned={status['planned_simulations']}")
    rows = []
    try:
        science = load_science_module()
        # Sham first, then increasing active magnitude so an interpretable contrast arrives early.
        order = (0.0, -0.0075, 0.0075, -0.0200, 0.0200, -0.0350, 0.0350, -0.0500, 0.0500)
        for dose in order:
            for seed in SEEDS:
                log(f"SIM_START seed={seed} dose={dose:+.4f}")
                row = simulate(science, seed, dose, args.duration_ms, args.burn_s)
                rows.append(row)
                write_rows(rows)
                status["completed_simulations"] = len(rows)
                status["last_completed"] = {"seed": seed, "dose_mV_per_ms": dose, "utc": utc_now()}
                if len(rows) >= 6 and len(rows) % 3 == 0:
                    response, matrix = build_response(rows)
                    # Partial tables are allowed to contain blank cells in CSV, but PNG waits for completion.
                    with (ROOT / "mvp_phase2a_quicklook.csv").open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(response[0]))
                        writer.writeheader(); writer.writerows(response)
                atomic_json(ROOT / "mvp_phase2a_status.json", status)
                log(f"SIM_DONE seed={seed} dose={dose:+.4f} elapsed_s={row['elapsed_s']:.2f}")
        response, matrix = build_response(rows)
        write_quicklook(response, matrix)
        write_summary(rows, matrix, time.time() - started)
        status.update({"status": "COMPLETE", "completed_utc": utc_now(), "completed_simulations": len(rows), "numerically_stable_simulations": sum(bool(r["numerically_stable"]) for r in rows)})
        atomic_json(ROOT / "mvp_phase2a_status.json", status)
        log("RUN_COMPLETE quicklook and summary written")
        return 0
    except Exception as exc:
        status.update({"status": "FAILED", "failed_utc": utc_now(), "error": repr(exc), "completed_simulations": len(rows)})
        atomic_json(ROOT / "mvp_phase2a_status.json", status)
        log("RUN_FAILED " + repr(exc))
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
