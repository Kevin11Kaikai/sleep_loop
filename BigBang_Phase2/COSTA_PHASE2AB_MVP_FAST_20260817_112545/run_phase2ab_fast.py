from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import neurolib
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt, welch


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
SCIENCE_SOURCE = WORKSPACE / "S4_sbi" / "DE_holdout_audit" / "scripts" / "run_frozen_validation.py"
SEEDS = (44017, 55021, 66029)
DOSES = (0.0, -0.035, 0.035)
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
PROTOCOL = {"simulation": {"backend_primary": "numba", "integration_dt_ms": 0.1, "sampling_dt_ms": 1.0}}
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    with (ROOT / "RUN_LOG.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_deadline(deadline: float, where: str) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError(f"900_SECOND_HARD_STOP_AT_{where}")


def load_science_module():
    spec = importlib.util.spec_from_file_location("costa_phase2ab_dev_runner", SCIENCE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runner {SCIENCE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def version(package: str, module) -> str:
    value = getattr(module, "__version__", None)
    if value:
        return str(value)
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def environment_manifest() -> dict:
    return {
        "created_utc": utc_now(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "versions": {
            "neurolib": version("neurolib", neurolib),
            "numpy": version("numpy", np),
            "scipy": version("scipy", scipy),
            "pandas": version("pandas", pd),
            "matplotlib": version("matplotlib", matplotlib),
            "numba": version("numba", numba),
        },
        "science_source": str(SCIENCE_SOURCE),
        "evidence_ceiling": "EXPLORATORY_ONLY",
    }


def set_external_current(model, dose: float) -> None:
    keys = [key for key in model.params if key.endswith("ALNMassEXC_0.ext_exc_current")]
    if len(keys) != 1:
        raise RuntimeError(f"Expected one cortical current key, got {keys}")
    model.params[keys[0]] = float(dose)
    model._update_model_params()


def mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    diff = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)))


def compute_metrics(x: np.ndarray, fs: float = 1000.0) -> dict:
    x = np.asarray(x, dtype=float)
    centered = x - float(np.mean(x))
    nperseg = min(len(x), int(8 * fs))
    f, p = welch(centered, fs=fs, window="hann", nperseg=nperseg, noverlap=min(int(4 * fs), nperseg - 1))
    so_mask = (f >= 0.2) & (f <= 1.5)
    sigma_mask = (f >= 11.0) & (f <= 15.0)
    so = sosfiltfilt(butter(4, [0.2, 1.5], btype="bandpass", fs=fs, output="sos"), centered)
    sigma = sosfiltfilt(butter(4, [11.0, 15.0], btype="bandpass", fs=fs, output="sos"), centered)
    phase = np.angle(hilbert(so))
    env = np.abs(hilbert(sigma))
    vector = np.sum(env * np.exp(1j * phase)) / max(float(np.sum(env)), 1e-12)
    duration_min = len(x) / fs / 60.0
    troughs, _ = find_peaks(-so, distance=max(1, int(0.5 * fs)), prominence=max(0.25 * float(np.std(so)), 1e-12))
    rms = np.sqrt(uniform_filter1d(sigma * sigma, size=max(1, int(0.2 * fs)), mode="nearest"))
    threshold = float(np.mean(rms) + 1.5 * np.std(rms))
    spindle_count = sum(1 for a, b in mask_runs(rms > threshold) if 0.5 <= (b - a) / fs <= 3.0)
    smooth = uniform_filter1d(centered, size=max(1, int(0.25 * fs)), mode="nearest")
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


def stable_signals(cortex: np.ndarray, thalamus: np.ndarray) -> tuple[bool, float]:
    finite = bool(np.isfinite(cortex).all() and np.isfinite(thalamus).all())
    max_abs = float(max(np.max(np.abs(cortex)), np.max(np.abs(thalamus)))) if finite else float("inf")
    stable = bool(finite and max_abs < 500.0 and np.ptp(cortex) > 1e-8)
    return stable, max_abs


def simulate_static(science, seed: int, dose: float, duration_ms: int, burn_s: float) -> dict:
    np.random.seed(seed)
    science.seed_numba(seed)
    model = science.build_model(PARAMS, seed, duration_ms, PROTOCOL)
    set_external_current(model, dose)
    started = time.monotonic()
    model.run()
    r_exc = np.asarray(model["r_mean_EXC"], dtype=float)
    if r_exc.ndim != 2 or r_exc.shape[0] < 2:
        raise RuntimeError(f"Unexpected EXC output shape {r_exc.shape}")
    drop = int(round(burn_s * 1000.0))
    cortex = r_exc[0, drop:] * 1000.0
    thalamus = r_exc[1, drop:] * 1000.0
    stable, max_abs = stable_signals(cortex, thalamus)
    if not stable:
        raise RuntimeError(f"Numerical stability failure seed={seed} dose={dose} max_abs={max_abs}")
    return {
        "seed": int(seed),
        "dose_mV_per_ms": float(dose),
        "condition": "SHAM" if dose == 0.0 else ("NEGATIVE" if dose < 0 else "POSITIVE"),
        "numerically_stable": stable,
        "max_abs_rate_hz": max_abs,
        "elapsed_s": float(time.monotonic() - started),
        **compute_metrics(cortex),
    }


def paired_effects(rows: list[dict], metric: str, dose: float) -> list[float]:
    values = []
    for seed in SEEDS:
        sham = next(float(r[metric]) for r in rows if r["seed"] == seed and r["dose_mV_per_ms"] == 0.0)
        active = next(float(r[metric]) for r in rows if r["seed"] == seed and r["dose_mV_per_ms"] == dose)
        values.append((active - sham) / max(abs(sham), 1e-12))
    return values


def phase2a_decision(rows: list[dict]) -> tuple[dict, list[dict]]:
    evaluation = []
    for target, metric, description in TARGETS:
        neg = paired_effects(rows, metric, -0.035)
        pos = paired_effects(rows, metric, 0.035)
        evaluation.append({
            "target": target,
            "metric": metric,
            "description": description,
            "negative_effect_seed_44017": neg[0],
            "negative_effect_seed_55021": neg[1],
            "negative_effect_seed_66029": neg[2],
            "negative_effect_median": float(np.median(neg)),
            "positive_effect_seed_44017": pos[0],
            "positive_effect_seed_55021": pos[1],
            "positive_effect_seed_66029": pos[2],
            "positive_effect_median": float(np.median(pos)),
        })
    expected = {"T5": -1, "T6": -1, "T8": 1}
    core = {}
    for target, direction in expected.items():
        item = next(r for r in evaluation if r["target"] == target)
        effects = [item[f"negative_effect_seed_{s}"] for s in SEEDS]
        consistent = sum(1 for x in effects if direction * x > 0)
        magnitude = abs(float(np.median(effects)))
        core[target] = {
            "consistent_seeds": consistent,
            "median_relative_effect": float(np.median(effects)),
            "passed": bool(consistent >= 2 and magnitude >= 0.20),
        }
    stable_count = sum(bool(r["numerically_stable"]) for r in rows)
    passed_core = sum(int(v["passed"]) for v in core.values())
    proceed = bool(stable_count == 9 and passed_core >= 2)
    target = None
    target_detail = {}
    if proceed:
        t2 = paired_effects(rows, "so_rms_amplitude", 0.035)
        t2_ok = sum(x > 0 for x in t2) >= 2 and float(np.median(t2)) >= 0.10
        t6 = paired_effects(rows, "coupling_strength", 0.035)
        t6_ok = sum(x > 0 for x in t6) >= 2 and float(np.median(t6)) >= 0.10
        if t2_ok:
            target = "T2"
            target_detail = {"effects": t2, "median": float(np.median(t2))}
        elif t6_ok:
            target = "T6"
            target_detail = {"effects": t6, "median": float(np.median(t6))}
    decision = {
        "schema_version": "1.0",
        "evidence_ceiling": "EXPLORATORY_ONLY",
        "decision": "PROCEED_TO_EXPLORATORY_PHASE2B" if proceed and target else "DO_NOT_PROCEED",
        "numerically_stable_simulations": stable_count,
        "core_targets_passed": passed_core,
        "core_gate": core,
        "selected_pid_target": target,
        "selected_target_detail": target_detail,
        "phase2b_formal_or_confirmatory_authorized": False,
        "prohibited_payloads_used": False,
        "decided_utc": utc_now(),
    }
    if proceed and not target:
        decision["single_blocker"] = "NO_T2_OR_T6_POSITIVE_DOSE_TARGET_QUALIFIED"
    elif not proceed:
        decision["single_blocker"] = "PHASE2A_MINIMUM_EXIT_GATE_FAILED"
    return decision, evaluation


def rate_indices(model) -> tuple[int, int]:
    names = [name for node in model.model_instance.state_variable_names for name in node]
    exc = [i for i, name in enumerate(names) if name == "r_mean_EXC"]
    if len(exc) != 2:
        raise RuntimeError(f"Unexpected rate-state layout: {names}")
    return exc[0], exc[1]


def feedback_metric(signal: np.ndarray, target: str) -> float:
    values = compute_metrics(signal)
    return float(values["so_rms_amplitude"] if target == "T2" else values["coupling_strength"])


def simulate_pid(science, seed: int, target: str, sham_setpoint_base: float, deadline: float) -> tuple[dict, list[dict]]:
    duration_ms = 65000
    burn_ms = 5000
    update_ms = 2000
    dt = 0.1
    sample_dt = 1.0
    np.random.seed(seed)
    science.seed_numba(seed)
    model = science.build_model(PARAMS, seed, duration_ms, PROTOCOL)
    model.params["duration"] = duration_ms
    model._update_model_params()
    full_noise = np.asarray(model._init_noise_inputs("numba"), dtype=float)
    expected_steps = int(round(duration_ms / dt))
    if full_noise.ndim != 2 or full_noise.shape[1] != expected_steps:
        raise RuntimeError(f"Unexpected noise shape {full_noise.shape}")
    ctx_idx, th_idx = rate_indices(model)
    segment_steps = int(round(update_ms / dt))
    sample_stride = int(round(sample_dt / dt))
    max_delay_steps = int(round(model.model_instance.max_delay / dt))
    n_segments = int(math.ceil(expected_steps / segment_steps))
    state_history = None
    cortical_chunks: list[np.ndarray] = []
    thalamic_chunks: list[np.ndarray] = []
    audit: list[dict] = []
    current_command = 0.0
    integral = 0.0
    previous_error = 0.0
    derivative_smooth = 0.0
    setpoint = 1.20 * sham_setpoint_base
    for segment in range(n_segments):
        check_deadline(deadline, f"PID_SEED_{seed}_SEGMENT_{segment}")
        start = segment * segment_steps
        end = min((segment + 1) * segment_steps, expected_steps)
        if start < int(round(burn_ms / dt)):
            current_command = 0.0
        set_external_current(model, current_command)
        if state_history is not None:
            model.model_instance.initial_state = state_history
        _, result = model.model_instance.run(
            duration=(end - start) * dt,
            dt=dt,
            noise_input=full_noise[:, start:end],
            backend="numba",
            return_xarray=False,
        )
        state_history = result[:, -max_delay_steps - 1:].copy()
        sample_idx = np.arange(sample_stride - 1, result.shape[1], sample_stride)
        cortex = result[ctx_idx, sample_idx] * 1000.0
        thalamus = result[th_idx, sample_idx] * 1000.0
        cortical_chunks.append(cortex)
        thalamic_chunks.append(thalamus)
        accumulated = np.concatenate(cortical_chunks)
        post = accumulated[int(round(burn_ms / sample_dt)):]
        measurement = None
        normalized_error = None
        if end * dt > burn_ms and len(post) >= 5000:
            window = post[-min(len(post), 10000):]
            measurement = feedback_metric(window, target)
            normalized_error = (setpoint - measurement) / max(abs(setpoint), 1e-12)
            integral = float(np.clip(integral + normalized_error * update_ms / 1000.0, -4.0, 4.0))
            raw_derivative = (normalized_error - previous_error) / (update_ms / 1000.0)
            derivative_smooth = 0.8 * derivative_smooth + 0.2 * raw_derivative
            previous_error = normalized_error
            current_command = float(np.clip(0.05 * normalized_error + 0.005 * integral + 0.001 * derivative_smooth, 0.0, 0.05))
        audit.append({
            "seed": seed,
            "segment": segment,
            "time_end_s": end * dt / 1000.0,
            "applied_command_mV_per_ms": float(0.0 if start < int(round(burn_ms / dt)) else current_command),
            "measurement": measurement,
            "setpoint": setpoint,
            "normalized_error": normalized_error,
        })
    cortex_all = np.concatenate(cortical_chunks)[int(round(burn_ms / sample_dt)):]
    thalamus_all = np.concatenate(thalamic_chunks)[int(round(burn_ms / sample_dt)):]
    stable, max_abs = stable_signals(cortex_all, thalamus_all)
    if not stable:
        raise RuntimeError(f"PID numerical stability failure seed={seed} max_abs={max_abs}")
    full = compute_metrics(cortex_all)
    final = compute_metrics(cortex_all[-20000:])
    valid_audit = [r for r in audit if r["normalized_error"] is not None]
    first_errors = [abs(float(r["normalized_error"])) for r in valid_audit if float(r["time_end_s"]) <= 25.0]
    last_errors = [abs(float(r["normalized_error"])) for r in valid_audit if float(r["time_end_s"]) >= 45.0]
    commands = [float(r["applied_command_mV_per_ms"]) for r in valid_audit]
    result_row = {
        "seed": seed,
        "target": target,
        "setpoint": setpoint,
        "numerically_stable": stable,
        "max_abs_rate_hz": max_abs,
        "final_target_value": final["so_rms_amplitude" if target == "T2" else "coupling_strength"],
        "initial_mean_abs_error": float(np.mean(first_errors)),
        "final_mean_abs_error": float(np.mean(last_errors)),
        "command_min": float(np.min(commands)),
        "command_max": float(np.max(commands)),
        "command_range": float(np.ptp(commands)),
        "saturation_fraction": float(np.mean([(u <= 1e-9) or (u >= 0.05 - 1e-9) for u in commands])),
        **full,
    }
    return result_row, audit


def pid_decision(pid_rows: list[dict], phase2a_rows: list[dict], target: str) -> dict:
    target_metric = "so_rms_amplitude" if target == "T2" else "coupling_strength"
    improvements = []
    error_improved = []
    dynamic = []
    for row in pid_rows:
        sham = next(r for r in phase2a_rows if r["seed"] == row["seed"] and r["dose_mV_per_ms"] == 0.0)
        improvements.append((float(row["final_target_value"]) - float(sham[target_metric])) / max(abs(float(sham[target_metric])), 1e-12))
        error_improved.append(float(row["final_mean_abs_error"]) < float(row["initial_mean_abs_error"]))
        dynamic.append(float(row["command_range"]) >= 0.005)
    t5_changes, t6_changes, t8_ratios = [], [], []
    for row in pid_rows:
        sham = next(r for r in phase2a_rows if r["seed"] == row["seed"] and r["dose_mV_per_ms"] == 0.0)
        t5_changes.append((row["sigma_power"] - sham["sigma_power"]) / max(abs(sham["sigma_power"]), 1e-12))
        t6_changes.append((row["coupling_strength"] - sham["coupling_strength"]) / max(abs(sham["coupling_strength"]), 1e-12))
        t8_ratios.append(row["state_transitions_per_min"] / max(sham["state_transitions_per_min"], 1e-12))
    gates = {
        "numerical_stability_3_of_3": sum(bool(r["numerically_stable"]) for r in pid_rows) == 3,
        "target_improvement_2_of_3": sum(x >= 0.10 for x in improvements) >= 2,
        "tracking_error_improved_2_of_3": sum(error_improved) >= 2,
        "dynamic_command_2_of_3": sum(dynamic) >= 2,
        "median_saturation_below_50pct": float(np.median([r["saturation_fraction"] for r in pid_rows])) < 0.50,
        "t5_median_decline_within_50pct": float(np.median(t5_changes)) >= -0.50,
        "t6_median_decline_within_20pct": float(np.median(t6_changes)) >= -0.20,
        "t8_median_ratio_below_2": float(np.median(t8_ratios)) < 2.0,
    }
    passed = all(gates.values())
    return {
        "schema_version": "1.0",
        "evidence_ceiling": "EXPLORATORY_ONLY",
        "decision": "PID_MVP_DEMONSTRATED" if passed else "NO_DEMONSTRATED_CONTROL",
        "selected_target": target,
        "target_improvements_vs_sham": improvements,
        "median_target_improvement": float(np.median(improvements)),
        "gates": gates,
        "formal_or_clinical_claim_authorized": False,
        "decided_utc": utc_now(),
    }


def plot_pid(pid_rows: list[dict], traces: list[dict], phase2a_rows: list[dict], target: str) -> None:
    metric = "so_rms_amplitude" if target == "T2" else "coupling_strength"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(SEEDS))
    sham = [next(r[metric] for r in phase2a_rows if r["seed"] == s and r["dose_mV_per_ms"] == 0.0) for s in SEEDS]
    fixed = [next(r[metric] for r in phase2a_rows if r["seed"] == s and r["dose_mV_per_ms"] == 0.035) for s in SEEDS]
    pid = [next(r[metric] for r in pid_rows if r["seed"] == s) for s in SEEDS]
    width = 0.25
    axes[0].bar(x - width, sham, width, label="sham")
    axes[0].bar(x, fixed, width, label="fixed +0.035")
    axes[0].bar(x + width, pid, width, label="PID")
    axes[0].set_xticks(x, [str(s) for s in SEEDS])
    axes[0].set_title(f"{target} model-internal response")
    axes[0].set_xlabel("seed")
    axes[0].set_ylabel(metric)
    axes[0].legend()
    for seed in SEEDS:
        rows = [r for r in traces if r["seed"] == seed and r["measurement"] is not None]
        axes[1].plot([r["time_end_s"] for r in rows], [r["applied_command_mV_per_ms"] for r in rows], label=str(seed))
    axes[1].set_title("Bounded PID command")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("ext_exc_current (mV/ms)")
    axes[1].legend(title="seed")
    fig.suptitle("COSTA exploratory Phase 2B PID MVP — EXPLORATORY_ONLY")
    fig.tight_layout()
    fig.savefig(ROOT / "pid_quicklook.png", dpi=170)
    plt.close(fig)


def summary_text(closeout: dict, pid_result: dict | None, elapsed: float) -> str:
    if pid_result is None:
        outcome = f"Phase 2B was not launched. Single blocker: `{closeout.get('single_blocker', 'UNKNOWN')}`."
    elif pid_result["decision"] == "PID_MVP_DEMONSTRATED":
        outcome = f"A bounded model-internal PID MVP was demonstrated for {pid_result['selected_target']} with median target change {pid_result['median_target_improvement']:+.1%} versus paired sham."
    else:
        failed = [k for k, v in pid_result["gates"].items() if not v]
        outcome = f"Exploratory Phase 2B executed, but control was not demonstrated. Failed gates: {failed}. No additional experiments were launched."
    return f"""# COSTA Phase 2A Closeout / Exploratory Phase 2B MVP

**Evidence ceiling: EXPLORATORY_ONLY.** This is a model-internal feasibility result, not clinical, confirmatory, C1/C2, efficacy, safety, treatment, or dissertation-final evidence.

## Phase 2A closeout

- Decision: `{closeout['decision']}`
- Stable simulations: {closeout['numerically_stable_simulations']}/9
- Core gates passed: {closeout['core_targets_passed']}/3
- Selected PID target: `{closeout.get('selected_pid_target')}`

## Exploratory Phase 2B

{outcome}

## Resource and data boundary

- Total elapsed time: {elapsed:.1f} seconds.
- Execution environment: activated Conda environment `neurolib`.
- No Night-2, Sealed Bank, protected-derived, fresh-final, or confirmatory payload was used.
- No gain sweep or recursive repair loop was performed.
"""


def preflight() -> int:
    science = load_science_module()
    manifest = environment_manifest()
    assert os.environ.get("CONDA_DEFAULT_ENV") == "neurolib", manifest
    assert len(SEEDS) == 3 and len(DOSES) == 3 and len(TARGETS) == 8 and len(PARAMS) == 8
    assert all(np.isfinite(list(PARAMS.values())))
    assert callable(science.build_model) and callable(science.seed_numba)
    print(json.dumps({"PREFLIGHT_OK": True, "environment": manifest, "planned_phase2a_simulations": 9, "maximum_pid_simulations": 3}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--mode", default="closeout_then_pid", choices=["closeout_then_pid"])
    parser.add_argument("--hard-stop-min", type=float, default=15.0)
    args = parser.parse_args()
    if args.preflight:
        return preflight()
    started = time.monotonic()
    deadline = started + args.hard_stop_min * 60.0
    manifest = environment_manifest()
    if manifest["conda_default_env"] != "neurolib":
        raise RuntimeError(f"WRONG_CONDA_ENVIRONMENT: {manifest['conda_default_env']}")
    atomic_json(ROOT / "environment_manifest.json", manifest)
    status = {
        "schema_version": "1.0",
        "status": "RUNNING_PHASE2A_CLOSEOUT",
        "evidence_ceiling": "EXPLORATORY_ONLY",
        "pid": os.getpid(),
        "started_utc": utc_now(),
        "hard_stop_minutes": args.hard_stop_min,
        "phase2a_planned": 9,
        "phase2a_completed": 0,
        "phase2b_planned_maximum": 3,
        "phase2b_completed": 0,
        "prohibited_payloads_used": False,
    }
    atomic_json(ROOT / "phase2ab_status.json", status)
    log(f"EXECUTION_STARTED=true PID={os.getpid()} env={manifest['conda_default_env']} python={manifest['python_executable']}")
    log("VERSIONS " + json.dumps(manifest["versions"], sort_keys=True))
    phase2a_rows: list[dict] = []
    try:
        science = load_science_module()
        for dose in DOSES:
            for seed in SEEDS:
                check_deadline(deadline, f"PHASE2A_SEED_{seed}_DOSE_{dose}")
                log(f"PHASE2A_SIM_START seed={seed} dose={dose:+.3f}")
                row = simulate_static(science, seed, dose, 65000, 5.0)
                phase2a_rows.append(row)
                write_csv(ROOT / "phase2a_replication.csv", phase2a_rows)
                status["phase2a_completed"] = len(phase2a_rows)
                atomic_json(ROOT / "phase2ab_status.json", status)
                log(f"PHASE2A_SIM_DONE seed={seed} dose={dose:+.3f} elapsed_s={row['elapsed_s']:.2f}")
        closeout, evaluation = phase2a_decision(phase2a_rows)
        write_csv(ROOT / "phase2a_8target_evaluation.csv", evaluation)
        atomic_json(ROOT / "PHASE2A_CLOSEOUT_DECISION.json", closeout)
        log(f"PHASE2A_DECISION={closeout['decision']} core_passed={closeout['core_targets_passed']} target={closeout['selected_pid_target']}")
        if closeout["decision"] != "PROCEED_TO_EXPLORATORY_PHASE2B":
            status.update({"status": "STOPPED_AT_PHASE2A_GATE", "completed_utc": utc_now(), "single_blocker": closeout.get("single_blocker")})
            atomic_json(ROOT / "phase2ab_status.json", status)
            (ROOT / "PHASE2B_MVP_SUMMARY.md").write_text(summary_text(closeout, None, time.monotonic() - started), encoding="utf-8")
            return 2
        status["status"] = "RUNNING_EXPLORATORY_PHASE2B_PID"
        status["selected_pid_target"] = closeout["selected_pid_target"]
        atomic_json(ROOT / "phase2ab_status.json", status)
        target = str(closeout["selected_pid_target"])
        target_metric = "so_rms_amplitude" if target == "T2" else "coupling_strength"
        pid_rows: list[dict] = []
        traces: list[dict] = []
        for seed in SEEDS:
            check_deadline(deadline, f"PHASE2B_PID_SEED_{seed}")
            sham = next(r for r in phase2a_rows if r["seed"] == seed and r["dose_mV_per_ms"] == 0.0)
            log(f"PID_SIM_START seed={seed} target={target}")
            row, audit = simulate_pid(science, seed, target, float(sham[target_metric]), deadline)
            pid_rows.append(row)
            traces.extend(audit)
            write_csv(ROOT / "pid_summary.csv", pid_rows)
            write_csv(ROOT / "pid_traces.csv", traces)
            status["phase2b_completed"] = len(pid_rows)
            atomic_json(ROOT / "phase2ab_status.json", status)
            log(f"PID_SIM_DONE seed={seed} target={target} command_range={row['command_range']:.5f}")
        pid_result = pid_decision(pid_rows, phase2a_rows, target)
        atomic_json(ROOT / "PHASE2B_PID_DECISION.json", pid_result)
        plot_pid(pid_rows, traces, phase2a_rows, target)
        (ROOT / "PHASE2B_MVP_SUMMARY.md").write_text(summary_text(closeout, pid_result, time.monotonic() - started), encoding="utf-8")
        status.update({
            "status": "COMPLETE",
            "completed_utc": utc_now(),
            "pid_decision": pid_result["decision"],
            "elapsed_seconds": float(time.monotonic() - started),
        })
        atomic_json(ROOT / "phase2ab_status.json", status)
        log(f"RUN_COMPLETE pid_decision={pid_result['decision']} elapsed_s={status['elapsed_seconds']:.1f}")
        return 0 if pid_result["decision"] == "PID_MVP_DEMONSTRATED" else 3
    except Exception as exc:
        status.update({"status": "FAILED", "failed_utc": utc_now(), "error": repr(exc), "single_blocker": str(exc)})
        atomic_json(ROOT / "phase2ab_status.json", status)
        log("RUN_FAILED " + repr(exc))
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
