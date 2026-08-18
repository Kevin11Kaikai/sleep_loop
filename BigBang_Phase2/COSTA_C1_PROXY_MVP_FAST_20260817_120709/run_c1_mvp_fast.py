from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurolib
import numba
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt, welch


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
SCIENCE_SOURCE = WORKSPACE / "S4_sbi" / "DE_holdout_audit" / "scripts" / "run_frozen_validation.py"
TARGET_PSD_PATH = WORKSPACE / "data" / "target_psd_SC4001.npy"
TARGET_FREQS_PATH = WORKSPACE / "data" / "target_freqs_SC4001.npy"

DEVELOPMENT_SEEDS = (44017, 55021, 66029)
EVALUATION_SEEDS = (77131, 88141, 99149)
PATIENT_SETPOINT = 1.1310055082
PLANNED_SIMULATIONS = 18
IMPLEMENTATION_REPAIR = "flatten_neurolib_nested_state_variable_names_after_preflight_shape_failure"

PATIENT_PARAMS = {
    "mue": 2.5543399414050265,
    "mui": 4.638376924416662,
    "b": 39.7650205359397,
    "tauA": 4305.440791463454,
    "g_LK": 0.051187832154447256,
    "g_h": 0.15302235700635436,
    "c_th2ctx": 0.09928570402727765,
    "c_ctx2th": 0.05873748566621345,
}

GENERIC_PARAMS = {
    "mue": 3.20,
    "mui": 3.50,
    "b": 19.5,
    "tauA": 1040.0,
    "g_LK": 0.1,
    "g_h": 0.1,
    "c_th2ctx": 0.02,
    "c_ctx2th": 0.15,
}

PROTOCOL = {
    "simulation": {
        "backend_primary": "numba",
        "integration_dt_ms": 0.1,
        "sampling_dt_ms": 1.0,
    }
}

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

BANDS = {
    "so_0p5_1p25_hz": (0.5, 1.25),
    "delta_0p5_4_hz": (0.5, 4.0),
    "sigma_10_15_hz": (10.0, 15.0),
}


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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def check_deadline(deadline: float, where: str) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError(f"900_SECOND_HARD_STOP_AT_{where}")


def load_science_module():
    spec = importlib.util.spec_from_file_location("costa_c1_proxy_science", SCIENCE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import scientific runner: {SCIENCE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def package_version(package: str, module) -> str:
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
        "where_python": shutil.which("python"),
        "python_version": platform.python_version(),
        "versions": {
            "neurolib": package_version("neurolib", neurolib),
            "numpy": package_version("numpy", np),
            "scipy": package_version("scipy", scipy),
            "pandas": package_version("pandas", pd),
            "matplotlib": package_version("matplotlib", matplotlib),
            "numba": package_version("numba", numba),
        },
        "science_source": str(SCIENCE_SOURCE),
        "evidence_ceiling": "EXPLORATORY_ONLY",
        "claim_level": "EXPLORATORY_C1_PROXY_ONLY",
    }


def set_external_current(model, dose: float) -> None:
    keys = [key for key in model.params if key.endswith("ALNMassEXC_0.ext_exc_current")]
    if len(keys) != 1:
        raise RuntimeError(f"Expected exactly one cortical current parameter, got: {keys}")
    model.params[keys[0]] = float(dose)
    model._update_model_params()


def mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    diff = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)))


def compute_metrics(x: np.ndarray, fs: float = 1000.0) -> dict:
    x = np.asarray(x, dtype=float)
    centered = x - float(np.mean(x))
    nperseg = min(len(x), int(8 * fs))
    noverlap = min(int(4 * fs), nperseg - 1)
    freq, power = welch(centered, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)

    so_band = (freq >= 0.5) & (freq <= 1.25)
    sigma_band = (freq >= 10.0) & (freq <= 15.0)
    if not np.any(so_band) or not np.any(sigma_band):
        raise RuntimeError("PSD frequency grid does not cover required bands")
    so_freq = float(freq[so_band][int(np.argmax(power[so_band]))])
    sigma_power = float(np.trapezoid(power[sigma_band], freq[sigma_band]))

    so_sos = butter(4, [0.5, 1.25], btype="bandpass", fs=fs, output="sos")
    sigma_sos = butter(4, [10.0, 15.0], btype="bandpass", fs=fs, output="sos")
    so = sosfiltfilt(so_sos, centered)
    sigma = sosfiltfilt(sigma_sos, centered)
    so_rms = float(np.sqrt(np.mean(so * so)))

    duration_min = len(x) / fs / 60.0
    troughs, _ = find_peaks(-so, distance=max(1, int(0.5 * fs)), prominence=max(1e-12, 0.5 * np.std(so)))
    sigma_env = np.abs(hilbert(sigma))
    spindle_threshold = float(np.mean(sigma_env) + 2.0 * np.std(sigma_env))
    spindle_runs = [
        (a, b)
        for a, b in mask_runs(sigma_env > spindle_threshold)
        if int(0.3 * fs) <= (b - a) <= int(3.0 * fs)
    ]

    slow_phase = np.angle(hilbert(so))
    weights = sigma_env / max(float(np.mean(sigma_env)), 1e-12)
    phase_vector = np.mean(weights * np.exp(1j * slow_phase))
    coupling_strength = float(np.abs(phase_vector))
    preferred_phase = float(np.angle(phase_vector))

    smooth = uniform_filter1d(x, size=max(1, int(0.25 * fs)))
    threshold = float(np.median(smooth))
    transitions = int(np.count_nonzero(np.diff(smooth > threshold)))
    return {
        "so_peak_frequency_hz": so_freq,
        "so_rms_amplitude": so_rms,
        "so_density_per_min": float(len(troughs) / duration_min),
        "spindle_density_per_min": float(len(spindle_runs) / duration_min),
        "sigma_power": sigma_power,
        "coupling_strength": coupling_strength,
        "preferred_phase_rad": preferred_phase,
        "state_transitions_per_min": float(transitions / duration_min),
    }


def stable_signals(cortex: np.ndarray, thalamus: np.ndarray) -> tuple[bool, float]:
    arrays = [np.asarray(cortex, dtype=float), np.asarray(thalamus, dtype=float)]
    finite = all(np.isfinite(a).all() for a in arrays)
    max_abs = max(float(np.max(np.abs(a))) for a in arrays)
    nonflat = all(float(np.ptp(a)) > 1e-8 for a in arrays)
    return bool(finite and nonflat and max_abs < 1000.0), max_abs


def model_rate_indices(model) -> tuple[int, int]:
    names = [name for node_names in model.model_instance.state_variable_names for name in node_names]
    exc = [idx for idx, name in enumerate(names) if name == "r_mean_EXC"]
    if len(exc) != 2:
        raise RuntimeError(f"Unexpected state layout; EXC rate indices={exc}, names={names}")
    return exc[0], exc[1]


def simulate_static(science, params: dict, seed: int, dose: float, condition: str) -> tuple[dict, np.ndarray]:
    duration_ms = 65000
    burn_samples = 5000
    np.random.seed(seed)
    science.seed_numba(seed)
    started = time.monotonic()
    model = science.build_model(params, seed, duration_ms, PROTOCOL)
    set_external_current(model, dose)
    model.run()
    rates = np.asarray(model["r_mean_EXC"], dtype=float) * 1000.0
    if rates.ndim != 2 or rates.shape[0] < 2:
        raise RuntimeError(f"Unexpected static output shape {rates.shape}")
    cortex = rates[0, burn_samples:]
    thalamus = rates[1, burn_samples:]
    stable, max_abs = stable_signals(cortex, thalamus)
    if not stable:
        raise RuntimeError(f"Numerical stability failure condition={condition} seed={seed} max_abs={max_abs}")
    full = compute_metrics(cortex)
    final = compute_metrics(cortex[-20000:])
    row = {
        "seed": int(seed),
        "condition": condition,
        "dose_mV_per_ms": float(dose),
        "numerically_stable": True,
        "max_abs_rate_hz": max_abs,
        "final_T2_so_rms_amplitude": float(final["so_rms_amplitude"]),
        "elapsed_s": float(time.monotonic() - started),
        **full,
    }
    return row, cortex


def normalized_psd(freq: np.ndarray, power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freq = np.asarray(freq, dtype=float).reshape(-1)
    power = np.asarray(power, dtype=float).reshape(-1)
    if len(freq) != len(power):
        raise RuntimeError(f"PSD/frequency length mismatch: {len(power)} vs {len(freq)}")
    keep = np.isfinite(freq) & np.isfinite(power) & (freq >= 0.5) & (freq <= 20.0)
    freq = freq[keep]
    power = np.maximum(power[keep], 0.0)
    order = np.argsort(freq)
    freq, power = freq[order], power[order]
    if len(freq) < 10 or np.any(np.diff(freq) <= 0):
        raise RuntimeError("Insufficient or non-increasing PSD frequency grid")
    area = float(np.trapezoid(power, freq))
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError("PSD has non-positive normalization area")
    return freq, power / area


def anchor_errors(signal: np.ndarray, target_freq: np.ndarray, target_power: np.ndarray) -> dict:
    sim_freq, sim_power = welch(
        signal - float(np.mean(signal)),
        fs=1000.0,
        window="hann",
        nperseg=min(len(signal), 8000),
        noverlap=min(4000, min(len(signal), 8000) - 1),
    )
    sim_freq, sim_power = normalized_psd(sim_freq, sim_power)
    target_freq, target_power = normalized_psd(target_freq, target_power)
    sim_interp = np.interp(target_freq, sim_freq, sim_power)
    epsilon = max(float(np.max(target_power)), float(np.max(sim_interp))) * 1e-12
    result = {}
    for name, (low, high) in BANDS.items():
        mask = (target_freq >= low) & (target_freq <= high)
        if np.count_nonzero(mask) < 2:
            raise RuntimeError(f"Target frequency grid lacks samples for {name}")
        result[f"{name}_log_psd_mae"] = float(
            np.mean(np.abs(np.log10(sim_interp[mask] + epsilon) - np.log10(target_power[mask] + epsilon)))
        )
    result["composite_log_psd_mae"] = float(
        np.mean([result[f"{name}_log_psd_mae"] for name in BANDS])
    )
    return result


def evaluate_anchor(rows: list[dict]) -> dict:
    band_results = {}
    bands_better = 0
    for name in BANDS:
        key = f"{name}_log_psd_mae"
        patient = float(np.median([float(r[key]) for r in rows if r["model"] == "patient_v1"]))
        generic = float(np.median([float(r[key]) for r in rows if r["model"] == "generic_table3"]))
        better = patient < generic
        bands_better += int(better)
        band_results[name] = {
            "patient_median_log_psd_mae": patient,
            "generic_median_log_psd_mae": generic,
            "patient_better": better,
        }
    patient_composite = float(np.median([float(r["composite_log_psd_mae"]) for r in rows if r["model"] == "patient_v1"]))
    generic_composite = float(np.median([float(r["composite_log_psd_mae"]) for r in rows if r["model"] == "generic_table3"]))
    reduction = (generic_composite - patient_composite) / max(abs(generic_composite), 1e-12)
    gates = {
        "numerical_stability_6_of_6": sum(bool(r["numerically_stable"]) for r in rows) == 6,
        "patient_better_in_at_least_2_of_3_bands": bands_better >= 2,
        "composite_error_reduction_at_least_10pct": reduction >= 0.10,
    }
    passed = all(gates.values())
    return {
        "schema_version": "1.0",
        "evidence_ceiling": "EXPLORATORY_ONLY",
        "decision": "PERSONALIZATION_ANCHOR_MET" if passed else "NO_PERSONALIZATION_ANCHOR",
        "gates": gates,
        "bands_better": bands_better,
        "composite_error_reduction": reduction,
        "patient_composite_log_psd_mae": patient_composite,
        "generic_composite_log_psd_mae": generic_composite,
        "band_results": band_results,
        "formal_c1_authorized": False,
        "decided_utc": utc_now(),
    }


def feedback_t2(signal: np.ndarray) -> float:
    return float(compute_metrics(signal)["so_rms_amplitude"])


def simulate_pid(
    science,
    seed: int,
    condition: str,
    controller_setpoint: float,
    patient_setpoint: float,
    deadline: float,
) -> tuple[dict, list[dict]]:
    duration_ms = 65000
    burn_ms = 5000
    update_ms = 2000
    dt = 0.1
    sample_dt = 1.0
    np.random.seed(seed)
    science.seed_numba(seed)
    started = time.monotonic()
    model = science.build_model(PATIENT_PARAMS, seed, duration_ms, PROTOCOL)
    model.params["duration"] = duration_ms
    model._update_model_params()
    full_noise = np.asarray(model._init_noise_inputs("numba"), dtype=float)
    expected_steps = int(round(duration_ms / dt))
    if full_noise.ndim != 2 or full_noise.shape[1] != expected_steps:
        raise RuntimeError(f"Unexpected noise shape {full_noise.shape}")
    ctx_idx, th_idx = model_rate_indices(model)
    segment_steps = int(round(update_ms / dt))
    sample_stride = int(round(sample_dt / dt))
    max_delay_steps = int(round(model.model_instance.max_delay / dt))
    n_segments = int(math.ceil(expected_steps / segment_steps))

    state_history = None
    cortical_chunks: list[np.ndarray] = []
    thalamic_chunks: list[np.ndarray] = []
    audit: list[dict] = []
    next_command = 0.0
    integral = 0.0
    previous_error = 0.0
    derivative_smooth = 0.0

    for segment in range(n_segments):
        check_deadline(deadline, f"{condition}_SEED_{seed}_SEGMENT_{segment}")
        start = segment * segment_steps
        end = min((segment + 1) * segment_steps, expected_steps)
        applied_command = 0.0 if start < int(round(burn_ms / dt)) else next_command
        set_external_current(model, applied_command)
        if state_history is not None:
            model.model_instance.initial_state = state_history
        _, result = model.model_instance.run(
            duration=(end - start) * dt,
            dt=dt,
            noise_input=full_noise[:, start:end],
            backend="numba",
            return_xarray=False,
        )
        state_history = result[:, -max_delay_steps - 1 :].copy()
        sample_idx = np.arange(sample_stride - 1, result.shape[1], sample_stride)
        cortex = result[ctx_idx, sample_idx] * 1000.0
        thalamus = result[th_idx, sample_idx] * 1000.0
        cortical_chunks.append(cortex)
        thalamic_chunks.append(thalamus)

        accumulated = np.concatenate(cortical_chunks)
        post = accumulated[int(round(burn_ms / sample_dt)) :]
        measurement = None
        normalized_error = None
        patient_reference_error = None
        if end * dt > burn_ms and len(post) >= 5000:
            window = post[-min(len(post), 10000) :]
            measurement = feedback_t2(window)
            normalized_error = (controller_setpoint - measurement) / max(abs(controller_setpoint), 1e-12)
            patient_reference_error = abs(patient_setpoint - measurement) / max(abs(patient_setpoint), 1e-12)
            integral = float(np.clip(integral + normalized_error * update_ms / 1000.0, -4.0, 4.0))
            raw_derivative = (normalized_error - previous_error) / (update_ms / 1000.0)
            derivative_smooth = 0.8 * derivative_smooth + 0.2 * raw_derivative
            previous_error = normalized_error
            next_command = float(
                np.clip(0.05 * normalized_error + 0.005 * integral + 0.001 * derivative_smooth, 0.0, 0.05)
            )
        audit.append(
            {
                "seed": seed,
                "condition": condition,
                "segment": segment,
                "time_end_s": end * dt / 1000.0,
                "applied_command_mV_per_ms": float(applied_command),
                "measurement_T2": measurement,
                "controller_setpoint": controller_setpoint,
                "patient_setpoint": patient_setpoint,
                "controller_normalized_error": normalized_error,
                "patient_reference_abs_error": patient_reference_error,
            }
        )

    cortex_all = np.concatenate(cortical_chunks)[int(round(burn_ms / sample_dt)) :]
    thalamus_all = np.concatenate(thalamic_chunks)[int(round(burn_ms / sample_dt)) :]
    stable, max_abs = stable_signals(cortex_all, thalamus_all)
    if not stable:
        raise RuntimeError(f"PID stability failure condition={condition} seed={seed} max_abs={max_abs}")
    full = compute_metrics(cortex_all)
    final = compute_metrics(cortex_all[-20000:])
    valid = [r for r in audit if r["measurement_T2"] is not None]
    late = [r for r in valid if float(r["time_end_s"]) >= 45.0]
    commands = [float(r["applied_command_mV_per_ms"]) for r in valid]
    row = {
        "seed": seed,
        "condition": condition,
        "dose_mV_per_ms": None,
        "controller_setpoint": controller_setpoint,
        "patient_setpoint": patient_setpoint,
        "numerically_stable": True,
        "max_abs_rate_hz": max_abs,
        "final_T2_so_rms_amplitude": float(final["so_rms_amplitude"]),
        "final_patient_target_error": float(np.mean([float(r["patient_reference_abs_error"]) for r in late])),
        "command_min": float(np.min(commands)),
        "command_max": float(np.max(commands)),
        "command_range": float(np.ptp(commands)),
        "saturation_fraction": float(np.mean([(u <= 1e-9) or (u >= 0.05 - 1e-9) for u in commands])),
        "elapsed_s": float(time.monotonic() - started),
        **full,
    }
    return row, audit


def relative_change(value: float, reference: float) -> float:
    return (float(value) - float(reference)) / max(abs(float(reference)), 1e-12)


def evaluate_control(rows: list[dict]) -> tuple[dict, list[dict]]:
    evaluation = []
    personalized_improvements = []
    fixed_improvements = []
    personalization_error_advantages = []
    command_dynamic = []
    saturation_ok = []
    t5_changes, t6_changes, t8_ratios = [], [], []

    for seed in EVALUATION_SEEDS:
        sham = next(r for r in rows if r["seed"] == seed and r["condition"] == "sham")
        fixed = next(r for r in rows if r["seed"] == seed and r["condition"] == "fixed_plus_0p035")
        generic = next(r for r in rows if r["seed"] == seed and r["condition"] == "generic_pid")
        patient = next(r for r in rows if r["seed"] == seed and r["condition"] == "personalized_pid")
        p_gain = relative_change(patient["final_T2_so_rms_amplitude"], sham["final_T2_so_rms_amplitude"])
        f_gain = relative_change(fixed["final_T2_so_rms_amplitude"], sham["final_T2_so_rms_amplitude"])
        error_advantage = (
            float(generic["final_patient_target_error"]) - float(patient["final_patient_target_error"])
        ) / max(abs(float(generic["final_patient_target_error"])), 1e-12)
        personalized_improvements.append(p_gain)
        fixed_improvements.append(f_gain)
        personalization_error_advantages.append(error_advantage)
        command_dynamic.append(float(patient["command_range"]) >= 0.005)
        saturation_ok.append(float(patient["saturation_fraction"]) < 0.50)
        t5_changes.append(relative_change(patient["sigma_power"], sham["sigma_power"]))
        t6_changes.append(relative_change(patient["coupling_strength"], sham["coupling_strength"]))
        t8_ratios.append(float(patient["state_transitions_per_min"]) / max(float(sham["state_transitions_per_min"]), 1e-12))
        evaluation.append(
            {
                "seed": seed,
                "personalized_T2_change_vs_sham": p_gain,
                "fixed_T2_change_vs_sham": f_gain,
                "personalized_minus_fixed_percentage_points": p_gain - f_gain,
                "generic_pid_patient_target_error": generic["final_patient_target_error"],
                "personalized_pid_patient_target_error": patient["final_patient_target_error"],
                "personalized_error_advantage_vs_generic": error_advantage,
                "personalized_command_range": patient["command_range"],
                "personalized_saturation_fraction": patient["saturation_fraction"],
                "T5_change_vs_sham": t5_changes[-1],
                "T6_change_vs_sham": t6_changes[-1],
                "T8_ratio_vs_sham": t8_ratios[-1],
            }
        )

    median_personalized = float(np.median(personalized_improvements))
    median_fixed = float(np.median(fixed_improvements))
    gates = {
        "numerical_stability_12_of_12": sum(bool(r["numerically_stable"]) for r in rows) == 12,
        "personalized_T2_gain_at_least_10pct_in_2_of_3": sum(x >= 0.10 for x in personalized_improvements) >= 2,
        "personalized_median_T2_gain_at_least_10pct": median_personalized >= 0.10,
        "personalized_median_exceeds_fixed_by_5pp": (median_personalized - median_fixed) >= 0.05,
        "personalized_error_10pct_better_than_generic_in_2_of_3": sum(x >= 0.10 for x in personalization_error_advantages) >= 2,
        "personalized_command_range_at_least_0p005_in_2_of_3": sum(command_dynamic) >= 2,
        "personalized_saturation_below_50pct_in_2_of_3": sum(saturation_ok) >= 2,
        "T5_median_decline_within_50pct": float(np.median(t5_changes)) >= -0.50,
        "T6_median_decline_within_20pct": float(np.median(t6_changes)) >= -0.20,
        "T8_median_ratio_below_2": float(np.median(t8_ratios)) < 2.0,
    }
    passed = all(gates.values())
    decision = {
        "schema_version": "1.0",
        "evidence_ceiling": "EXPLORATORY_ONLY",
        "claim_level": "EXPLORATORY_C1_PROXY_ONLY",
        "decision": "EXPLORATORY_C1_PROXY_MVP_MET" if passed else "NO_C1_PROXY_CONTROL_SIGNAL",
        "gates": gates,
        "personalized_T2_changes_vs_sham": personalized_improvements,
        "fixed_T2_changes_vs_sham": fixed_improvements,
        "personalization_error_advantages_vs_generic": personalization_error_advantages,
        "median_personalized_T2_change": median_personalized,
        "median_fixed_T2_change": median_fixed,
        "median_personalized_minus_fixed": median_personalized - median_fixed,
        "median_T5_change": float(np.median(t5_changes)),
        "median_T6_change": float(np.median(t6_changes)),
        "median_T8_ratio": float(np.median(t8_ratios)),
        "formal_c1_authorized": False,
        "clinical_claim_authorized": False,
        "actual_patient_sleep_improvement_demonstrated": False,
        "decided_utc": utc_now(),
    }
    return decision, evaluation


def plot_quicklook(anchor_rows: list[dict], control_rows: list[dict], traces: list[dict], decision: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    band_names = list(BANDS)
    x = np.arange(len(band_names))
    patient_errors = [
        np.median([r[f"{name}_log_psd_mae"] for r in anchor_rows if r["model"] == "patient_v1"])
        for name in band_names
    ]
    generic_errors = [
        np.median([r[f"{name}_log_psd_mae"] for r in anchor_rows if r["model"] == "generic_table3"])
        for name in band_names
    ]
    axes[0, 0].bar(x - 0.18, patient_errors, 0.36, label="SC4001/V1")
    axes[0, 0].bar(x + 0.18, generic_errors, 0.36, label="Generic Table-3")
    axes[0, 0].set_xticks(x, ["SO", "Delta", "Sigma"])
    axes[0, 0].set_ylabel("normalized log-PSD MAE")
    axes[0, 0].set_title("Personalization anchor (lower is better)")
    axes[0, 0].legend()

    if control_rows:
        conditions = ["fixed_plus_0p035", "generic_pid", "personalized_pid"]
        labels = ["Fixed", "Generic PID", "Personalized PID"]
        for idx, seed in enumerate(EVALUATION_SEEDS):
            sham = next(r for r in control_rows if r["seed"] == seed and r["condition"] == "sham")
            values = []
            for condition in conditions:
                row = next(r for r in control_rows if r["seed"] == seed and r["condition"] == condition)
                values.append(100.0 * relative_change(row["final_T2_so_rms_amplitude"], sham["final_T2_so_rms_amplitude"]))
            axes[0, 1].plot(labels, values, marker="o", label=str(seed))
        axes[0, 1].axhline(10.0, color="gray", linestyle="--", linewidth=1)
        axes[0, 1].set_ylabel("T2 change vs sham (%)")
        axes[0, 1].set_title("Predicted slow-wave amplitude response")
        axes[0, 1].legend(title="seed")

        for condition, style in [("generic_pid", "--"), ("personalized_pid", "-")]:
            for seed in EVALUATION_SEEDS:
                subset = [
                    r for r in traces
                    if r["condition"] == condition and r["seed"] == seed and r["measurement_T2"] is not None
                ]
                axes[1, 0].plot(
                    [r["time_end_s"] for r in subset],
                    [r["applied_command_mV_per_ms"] for r in subset],
                    linestyle=style,
                    alpha=0.8,
                    label=f"{condition}:{seed}",
                )
        axes[1, 0].set_xlabel("time (s)")
        axes[1, 0].set_ylabel("ext_exc_current (mV/ms)")
        axes[1, 0].set_title("Bounded PID commands")
        axes[1, 0].legend(fontsize=7, ncol=2)

        safety_labels = ["T5 change", "T6 change", "T8 ratio-1"]
        safety_values = [
            100.0 * decision.get("median_T5_change", 0.0),
            100.0 * decision.get("median_T6_change", 0.0),
            100.0 * (decision.get("median_T8_ratio", 1.0) - 1.0),
        ]
        axes[1, 1].bar(safety_labels, safety_values, color=["#4c78a8", "#72b7b2", "#f58518"])
        axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
        axes[1, 1].set_ylabel("median relative change (%)")
        axes[1, 1].set_title("Physiology guardrails vs sham")
    else:
        axes[0, 1].axis("off")
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "Control phase not executed", ha="center", va="center")

    fig.suptitle(
        f"COSTA exploratory C1 proxy MVP — {decision.get('decision', 'ANCHOR_ONLY')}\n"
        "EXPLORATORY_ONLY; no formal C1 or clinical claim",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(ROOT / "c1_quicklook.png", dpi=170)
    plt.close(fig)


def write_summary(anchor: dict, decision: dict, elapsed_s: float) -> None:
    if decision["decision"] == "EXPLORATORY_C1_PROXY_MVP_MET":
        outcome = (
            "In one public-development participant-specific computational model, a pre-frozen bounded PID "
            "predicted improved slow-oscillation amplitude and showed preliminary advantage over sham, fixed "
            "dose, and a generic-reference controller. This supports exploratory C1-proxy feasibility only."
        )
    elif decision["decision"] == "NO_PERSONALIZATION_ANCHOR":
        outcome = "The personalized model did not meet the pre-specified baseline anchor; the control phase was not executed."
    else:
        outcome = "The bounded experiment did not meet every pre-specified exploratory C1-proxy control gate."
    text = f"""# COSTA Exploratory C1 Proxy MVP Summary

## Decision

- Evidence ceiling: `EXPLORATORY_ONLY`
- Personalization anchor: `{anchor['decision']}`
- C1 proxy decision: `{decision['decision']}`
- Formal C1 authorized: `false`
- Clinical or real-patient sleep-improvement claim authorized: `false`
- Total elapsed time: `{elapsed_s:.1f} s`

## Main result

{outcome}

## Key numbers

- Patient-vs-generic baseline composite PSD-error reduction: `{anchor.get('composite_error_reduction', float('nan')):+.1%}`
- Personalized PID median T2 change vs sham: `{decision.get('median_personalized_T2_change', float('nan')):+.1%}`
- Fixed-dose median T2 change vs sham: `{decision.get('median_fixed_T2_change', float('nan')):+.1%}`
- Personalized-minus-fixed median difference: `{decision.get('median_personalized_minus_fixed', float('nan')):+.1%}`
- Personalized PID median T5 change: `{decision.get('median_T5_change', float('nan')):+.1%}`
- Personalized PID median T6 change: `{decision.get('median_T6_change', float('nan')):+.1%}`
- Personalized PID median T8/sham ratio: `{decision.get('median_T8_ratio', float('nan')):.3f}`

## Interpretation boundary

This is a single-participant, model-internal, hypothesis-generating proxy. It does not contain an observed intervention-response label and therefore does not demonstrate that a real patient slept better. Formal C1 requires independent real intervention-response data.
"""
    (ROOT / "C1_MVP_SUMMARY.md").write_text(text, encoding="utf-8")


def ensure_placeholder_outputs() -> None:
    control_fields = [
        "seed", "condition", "dose_mV_per_ms", "controller_setpoint", "patient_setpoint",
        "numerically_stable", "max_abs_rate_hz", "final_T2_so_rms_amplitude",
        "final_patient_target_error", "command_min", "command_max", "command_range",
        "saturation_fraction", "elapsed_s",
    ] + [metric for _, metric, _ in TARGETS]
    trace_fields = [
        "seed", "condition", "segment", "time_end_s", "applied_command_mV_per_ms",
        "measurement_T2", "controller_setpoint", "patient_setpoint",
        "controller_normalized_error", "patient_reference_abs_error",
    ]
    evaluation_fields = [
        "seed", "personalized_T2_change_vs_sham", "fixed_T2_change_vs_sham",
        "personalized_minus_fixed_percentage_points", "generic_pid_patient_target_error",
        "personalized_pid_patient_target_error", "personalized_error_advantage_vs_generic",
        "personalized_command_range", "personalized_saturation_fraction",
        "T5_change_vs_sham", "T6_change_vs_sham", "T8_ratio_vs_sham",
    ]
    for path, fields in [
        (ROOT / "c1_control_runs.csv", control_fields),
        (ROOT / "pid_traces.csv", trace_fields),
        (ROOT / "c1_target_evaluation.csv", evaluation_fields),
    ]:
        if not path.exists():
            write_csv(path, [], fields)


def run_preflight(science) -> None:
    if os.environ.get("CONDA_DEFAULT_ENV") != "neurolib":
        raise RuntimeError(f"Expected active conda env neurolib, got {os.environ.get('CONDA_DEFAULT_ENV')!r}")
    if not SCIENCE_SOURCE.is_file():
        raise FileNotFoundError(SCIENCE_SOURCE)
    if not TARGET_PSD_PATH.is_file() or not TARGET_FREQS_PATH.is_file():
        raise FileNotFoundError("Required public/development SC4001 PSD summary files are absent")
    if len(DEVELOPMENT_SEEDS) != 3 or len(EVALUATION_SEEDS) != 3:
        raise RuntimeError("Expected exactly three development and three evaluation seeds")
    if set(DEVELOPMENT_SEEDS) & set(EVALUATION_SEEDS):
        raise RuntimeError("Development and evaluation seeds overlap")
    if PLANNED_SIMULATIONS != 6 + 4 * 3:
        raise RuntimeError("Planned simulation count is inconsistent")
    patient_model = science.build_model(PATIENT_PARAMS, DEVELOPMENT_SEEDS[0], 1000, PROTOCOL)
    generic_model = science.build_model(GENERIC_PARAMS, DEVELOPMENT_SEEDS[0], 1000, PROTOCOL)
    for label, model in [("patient", patient_model), ("generic", generic_model)]:
        keys = [key for key in model.params if key.endswith("ALNMassEXC_0.ext_exc_current")]
        if len(keys) != 1:
            raise RuntimeError(f"{label} model has unexpected cortical-control keys: {keys}")
        model_rate_indices(model)
    atomic_json(
        ROOT / "preflight_result.json",
        {
            "passed": True,
            "data_loaded": False,
            "scientific_simulation_executed": False,
            "planned_simulations": PLANNED_SIMULATIONS,
            "patient_model_constructed": True,
            "generic_model_constructed": True,
            "created_utc": utc_now(),
        },
    )
    log(f"PREFLIGHT_PASS data_loaded=false planned_simulations={PLANNED_SIMULATIONS}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="anchor_then_control", choices=["anchor_then_control"])
    parser.add_argument("--hard-stop-min", type=float, default=15.0)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    manifest = environment_manifest()
    atomic_json(ROOT / "environment_manifest.json", manifest)
    science = load_science_module()
    if args.preflight:
        run_preflight(science)
        return 0

    started = time.monotonic()
    deadline = started + args.hard_stop_min * 60.0
    ensure_placeholder_outputs()
    log(
        f"EXECUTION_STARTED=true PID={os.getpid()} process_status=RUNNING planned_simulations={PLANNED_SIMULATIONS} "
        f"estimated_first_result_min=1 log_path={ROOT / 'RUN_LOG.txt'}"
    )
    log(
        f"ENV conda={manifest['conda_default_env']} python={manifest['python_executable']} "
        f"python_version={manifest['python_version']} where_python={manifest['where_python']} versions={manifest['versions']}"
    )

    input_manifest = {
        "schema_version": "1.0",
        "created_utc": utc_now(),
        "evidence_ceiling": "EXPLORATORY_ONLY",
        "claim_level": "EXPLORATORY_C1_PROXY_ONLY",
        "participant": "SC4001_PUBLIC_DEVELOPMENT",
        "patient_params": PATIENT_PARAMS,
        "generic_params": GENERIC_PARAMS,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "patient_T2_setpoint": PATIENT_SETPOINT,
        "generic_T2_setpoint": None,
        "planned_simulations": PLANNED_SIMULATIONS,
        "bounded_implementation_repair": IMPLEMENTATION_REPAIR,
        "target_psd_path": str(TARGET_PSD_PATH),
        "target_freqs_path": str(TARGET_FREQS_PATH),
        "pid": {
            "Kp": 0.05, "Ki": 0.005, "Kd": 0.001, "command_bounds": [0.0, 0.05],
            "update_s": 2.0, "feedback_window_s": 10.0, "integral_bounds": [-4.0, 4.0],
            "derivative_smoothing_alpha": 0.8,
        },
        "prohibited_payloads_used": False,
        "formal_c1_authorized": False,
    }
    atomic_json(ROOT / "c1_input_manifest.json", input_manifest)

    anchor_rows: list[dict] = []
    control_rows: list[dict] = []
    traces: list[dict] = []
    anchor_decision: dict | None = None
    final_decision: dict | None = None
    try:
        target_power = np.asarray(np.load(TARGET_PSD_PATH, allow_pickle=False), dtype=float).squeeze()
        target_freq = np.asarray(np.load(TARGET_FREQS_PATH, allow_pickle=False), dtype=float).squeeze()
        normalized_psd(target_freq, target_power)
        log(f"SCIENTIFIC_DATA_LOAD target_psd_shape={target_power.shape} target_freq_shape={target_freq.shape}")

        for model_name, params in [("patient_v1", PATIENT_PARAMS), ("generic_table3", GENERIC_PARAMS)]:
            for seed in DEVELOPMENT_SEEDS:
                check_deadline(deadline, f"ANCHOR_{model_name}_{seed}")
                log(f"ANCHOR_SIM_START model={model_name} seed={seed}")
                row, signal = simulate_static(science, params, seed, 0.0, f"anchor_{model_name}")
                errors = anchor_errors(signal, target_freq, target_power)
                anchor_row = {"model": model_name, **row, **errors}
                anchor_rows.append(anchor_row)
                write_csv(ROOT / "personalization_anchor.csv", anchor_rows)
                log(
                    f"ANCHOR_SIM_DONE model={model_name} seed={seed} composite_log_psd_mae="
                    f"{errors['composite_log_psd_mae']:.6f} elapsed_s={row['elapsed_s']:.2f}"
                )

        anchor_decision = evaluate_anchor(anchor_rows)
        atomic_json(ROOT / "PERSONALIZATION_ANCHOR_DECISION.json", anchor_decision)
        generic_setpoint = 1.20 * float(
            np.median([r["so_rms_amplitude"] for r in anchor_rows if r["model"] == "generic_table3"])
        )
        input_manifest["generic_T2_setpoint"] = generic_setpoint
        input_manifest["generic_setpoint_frozen_utc"] = utc_now()
        atomic_json(ROOT / "c1_input_manifest.json", input_manifest)
        log(
            f"ANCHOR_DECISION decision={anchor_decision['decision']} bands_better={anchor_decision['bands_better']} "
            f"composite_reduction={anchor_decision['composite_error_reduction']:+.3%} generic_setpoint={generic_setpoint:.10f}"
        )

        if anchor_decision["decision"] != "PERSONALIZATION_ANCHOR_MET":
            final_decision = {
                "schema_version": "1.0",
                "evidence_ceiling": "EXPLORATORY_ONLY",
                "claim_level": "EXPLORATORY_C1_PROXY_ONLY",
                "decision": "NO_PERSONALIZATION_ANCHOR",
                "control_phase_executed": False,
                "formal_c1_authorized": False,
                "clinical_claim_authorized": False,
                "decided_utc": utc_now(),
            }
            atomic_json(ROOT / "C1_MVP_DECISION.json", final_decision)
            plot_quicklook(anchor_rows, [], [], final_decision)
            write_summary(anchor_decision, final_decision, time.monotonic() - started)
            log("STOP single_blocker=NO_PERSONALIZATION_ANCHOR no_additional_experiments=true")
            return 2

        for seed in EVALUATION_SEEDS:
            for condition, dose in [("sham", 0.0), ("fixed_plus_0p035", 0.035)]:
                check_deadline(deadline, f"CONTROL_{condition}_{seed}")
                log(f"CONTROL_SIM_START condition={condition} seed={seed}")
                row, _ = simulate_static(science, PATIENT_PARAMS, seed, dose, condition)
                control_rows.append(row)
                write_csv(ROOT / "c1_control_runs.csv", control_rows)
                log(
                    f"CONTROL_SIM_DONE condition={condition} seed={seed} "
                    f"final_T2={row['final_T2_so_rms_amplitude']:.6f} elapsed_s={row['elapsed_s']:.2f}"
                )

        for condition, setpoint in [("generic_pid", generic_setpoint), ("personalized_pid", PATIENT_SETPOINT)]:
            for seed in EVALUATION_SEEDS:
                check_deadline(deadline, f"CONTROL_{condition}_{seed}")
                log(f"CONTROL_SIM_START condition={condition} seed={seed} setpoint={setpoint:.10f}")
                row, audit = simulate_pid(science, seed, condition, setpoint, PATIENT_SETPOINT, deadline)
                control_rows.append(row)
                traces.extend(audit)
                write_csv(ROOT / "c1_control_runs.csv", control_rows)
                write_csv(ROOT / "pid_traces.csv", traces)
                log(
                    f"CONTROL_SIM_DONE condition={condition} seed={seed} final_T2={row['final_T2_so_rms_amplitude']:.6f} "
                    f"patient_target_error={row['final_patient_target_error']:.6f} command_range={row['command_range']:.6f} "
                    f"saturation={row['saturation_fraction']:.3f} elapsed_s={row['elapsed_s']:.2f}"
                )

        final_decision, evaluation = evaluate_control(control_rows)
        write_csv(ROOT / "c1_target_evaluation.csv", evaluation)
        atomic_json(ROOT / "C1_MVP_DECISION.json", final_decision)
        plot_quicklook(anchor_rows, control_rows, traces, final_decision)
        write_summary(anchor_decision, final_decision, time.monotonic() - started)
        log(
            f"C1_PROXY_DECISION decision={final_decision['decision']} "
            f"median_personalized_T2={final_decision['median_personalized_T2_change']:+.3%} "
            f"median_fixed_T2={final_decision['median_fixed_T2_change']:+.3%} gates={final_decision['gates']}"
        )
        log(f"RUN_COMPLETE elapsed_s={time.monotonic() - started:.2f} formal_c1=false clinical_claim=false")
        return 0 if final_decision["decision"] == "EXPLORATORY_C1_PROXY_MVP_MET" else 3
    except Exception as exc:
        log(f"RUN_FAILED type={type(exc).__name__} blocker={exc}")
        log(traceback.format_exc())
        failure = {
            "schema_version": "1.0",
            "evidence_ceiling": "EXPLORATORY_ONLY",
            "claim_level": "EXPLORATORY_C1_PROXY_ONLY",
            "decision": "EXECUTION_FAILED",
            "single_blocker": f"{type(exc).__name__}: {exc}",
            "formal_c1_authorized": False,
            "clinical_claim_authorized": False,
            "decided_utc": utc_now(),
        }
        atomic_json(ROOT / "C1_MVP_DECISION.json", failure)
        if anchor_decision is None:
            anchor_decision = {
                "decision": "ANCHOR_NOT_COMPLETED",
                "composite_error_reduction": float("nan"),
            }
        if anchor_rows:
            plot_quicklook(anchor_rows, control_rows, traces, failure)
        write_summary(anchor_decision, failure, time.monotonic() - started)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
