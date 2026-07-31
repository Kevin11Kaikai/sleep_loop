"""Route-3 synthetic cortical-observable recovery preflight.

This module intentionally works in model-observable space.  Cortical firing
rates and internal ALN states are not labelled as EEG and are not converted to
microvolts.  The routines provide deterministic feature contracts, local
sensitivity diagnostics, and a checkpointed prior-wide diagnostic micro-bank.
They do not train a posterior estimator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import qmc, spearmanr

from .eeg_observation_mapping_audit import audit_parameter_sets, run_state_audit
from .simulator_observable_adapter import ParameterSet, _load_model_module


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "S4_sbi" / "results" / "route3_synthetic_preflight"
PARAMETER_NAMES = (
    "mue",
    "mui",
    "b",
    "tauA",
    "g_LK",
    "g_h",
    "c_th2ctx",
    "c_ctx2th",
)
PARAMETER_UNITS = (
    "mV/ms",
    "mV/ms",
    "pA",
    "ms",
    "mS/cm^2",
    "mS/cm^2",
    "dimensionless coupling",
    "dimensionless coupling",
)
PERTURBATION_FRACTION = 0.02
RELATIVE_SINGULAR_VALUE_THRESHOLD = 1e-3
SIMULATION_SEED = 42
SOBOL_SEED = 20260727
SCHEMA_VERSION = "route3-preflight-v1"
EPOCH_DURATION_S = 30.0
WARM_UP_S = 5.0
MODEL_DURATION_S = 60.0
EXPECTED_FS_HZ = 1000.0
PSD_WINDOW_S = 4.0
PSD_OVERLAP_FRACTION = 0.5
PSD_TOTAL_BAND_HZ = (0.5, 20.0)
SO_BAND_HZ = (0.5, 1.5)
SIGMA_BAND_HZ = (11.0, 15.0)


@dataclass(frozen=True)
class FeatureSpec:
    """Definition of one fixed-order model-observable feature."""

    name: str
    source_signal: str
    formula: str
    unit: str
    aggregation: str
    validity_rule: str
    redundancy: str
    role: str
    scientific_interpretation: str
    scale_floor: float


def _rate_specs(signal: str, prefix: str, label: str) -> tuple[FeatureSpec, ...]:
    common = {
        "source_signal": signal,
        "aggregation": "first complete 30-s post-warm-up model window",
        "validity_rule": (
            "30,000 finite samples at 1000 Hz; PSD features additionally require "
            "positive finite 0.5-20 Hz Welch power"
        ),
        "role": "inference_candidate",
    }
    return (
        FeatureSpec(
            f"{prefix}_mean_hz",
            formula="arithmetic mean",
            unit="Hz",
            redundancy="paired with SD but not deterministically duplicated",
            scientific_interpretation=f"Mean {label} population firing rate.",
            scale_floor=1.0,
            **common,
        ),
        FeatureSpec(
            f"{prefix}_std_hz",
            formula="population standard deviation (ddof=0)",
            unit="Hz",
            redundancy="scale statistic; no coefficient-of-variation duplicate included",
            scientific_interpretation=f"Temporal variability of {label} firing rate.",
            scale_floor=1.0,
            **common,
        ),
        FeatureSpec(
            f"{prefix}_relative_so_power",
            formula="Welch power 0.5-1.5 Hz / Welch power 0.5-20 Hz",
            unit="dimensionless",
            redundancy="shares the same denominator with relative sigma power",
            scientific_interpretation=f"Slow-band spectral allocation in {label} rate.",
            scale_floor=0.01,
            **common,
        ),
        FeatureSpec(
            f"{prefix}_relative_sigma_power",
            formula="Welch power 11-15 Hz / Welch power 0.5-20 Hz",
            unit="dimensionless",
            redundancy="shares the same denominator with relative SO power",
            scientific_interpretation=f"Sigma-band spectral allocation in {label} rate.",
            scale_floor=0.01,
            **common,
        ),
        FeatureSpec(
            f"{prefix}_so_peak_frequency_hz",
            formula="argmax Welch PSD within 0.5-1.5 Hz",
            unit="Hz",
            redundancy="discrete at the 0.25-Hz Welch resolution",
            scientific_interpretation=f"Dominant slow-band frequency in {label} rate.",
            scale_floor=0.25,
            **common,
        ),
        FeatureSpec(
            f"{prefix}_sigma_peak_frequency_hz",
            formula="argmax Welch PSD within 11-15 Hz",
            unit="Hz",
            redundancy="discrete at the 0.25-Hz Welch resolution",
            scientific_interpretation=f"Dominant sigma-band frequency in {label} rate.",
            scale_floor=0.25,
            **common,
        ),
        FeatureSpec(
            f"{prefix}_spectral_entropy_0p5_20",
            formula="Shannon entropy of normalized 0.5-20 Hz Welch bins / log(N bins)",
            unit="dimensionless",
            redundancy="depends on the same normalized spectrum as relative powers",
            scientific_interpretation=f"Broadband spectral concentration of {label} rate.",
            scale_floor=0.01,
            **common,
        ),
    )


RATE_ONLY_FEATURES = (
    *_rate_specs("cortex_r_exc", "r_exc", "excitatory"),
    *_rate_specs("cortex_r_inh", "r_inh", "inhibitory"),
)


def _state_spec(
    name: str,
    signal: str,
    statistic: str,
    unit: str,
    interpretation: str,
    scale_floor: float,
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        source_signal=signal,
        formula=f"{statistic} over first complete 30-s post-warm-up model window",
        unit=unit,
        aggregation="first complete 30-s post-warm-up model window",
        validity_rule="30,000 finite samples at 1000 Hz",
        redundancy="mean and SD describe distinct location and scale aspects",
        role="privileged_internal_state_inference_candidate",
        scientific_interpretation=interpretation,
        scale_floor=scale_floor,
    )


PRIVILEGED_STATE_FEATURES = (
    _state_spec(
        "I_mu_exc_mean_mV_per_ms",
        "cortex_I_mu_exc",
        "arithmetic mean",
        "mV/ms",
        "Mean ALN excitatory-mass input/current-drive state.",
        0.01,
    ),
    _state_spec(
        "I_mu_exc_std_mV_per_ms",
        "cortex_I_mu_exc",
        "population standard deviation (ddof=0)",
        "mV/ms",
        "Variability of ALN excitatory-mass input/current-drive state.",
        0.01,
    ),
    _state_spec(
        "I_mu_inh_mean_mV_per_ms",
        "cortex_I_mu_inh",
        "arithmetic mean",
        "mV/ms",
        "Mean ALN inhibitory-mass input/current-drive state.",
        0.01,
    ),
    _state_spec(
        "I_mu_inh_std_mV_per_ms",
        "cortex_I_mu_inh",
        "population standard deviation (ddof=0)",
        "mV/ms",
        "Variability of ALN inhibitory-mass input/current-drive state.",
        0.01,
    ),
    _state_spec(
        "I_A_mean_pA",
        "cortex_adaptation_I_A",
        "arithmetic mean",
        "pA",
        "Mean excitatory adaptation current; privileged internal state.",
        1.0,
    ),
    _state_spec(
        "I_A_std_pA",
        "cortex_adaptation_I_A",
        "population standard deviation (ddof=0)",
        "pA",
        "Variability of excitatory adaptation current.",
        1.0,
    ),
    _state_spec(
        "effective_drive_mean_mV_per_ms",
        "cortex_effective_drive",
        "arithmetic mean",
        "mV/ms",
        "Mean I_mu_EXC - I_A/C transfer-function drive.",
        0.01,
    ),
    _state_spec(
        "effective_drive_std_mV_per_ms",
        "cortex_effective_drive",
        "population standard deviation (ddof=0)",
        "mV/ms",
        "Variability of I_mu_EXC - I_A/C transfer-function drive.",
        0.01,
    ),
    _state_spec(
        "syn_mu_exc_on_exc_mean",
        "cortex_syn_mu_exc_on_exc",
        "arithmetic mean",
        "dimensionless synaptic state",
        "Mean excitatory synaptic state driving the cortical EXC mass.",
        0.001,
    ),
    _state_spec(
        "syn_mu_inh_on_exc_mean",
        "cortex_syn_mu_inh_on_exc",
        "arithmetic mean",
        "dimensionless synaptic state",
        "Mean inhibitory synaptic state driving the cortical EXC mass.",
        0.001,
    ),
)

AUGMENTED_FEATURES = RATE_ONLY_FEATURES + PRIVILEGED_STATE_FEATURES

HELD_OUT_DIAGNOSTICS = (
    {
        "name": "rate_exc_inh_zero_lag_correlation",
        "source_signal": "cortex_r_exc + cortex_r_inh",
        "role": "synthetic_held_out_ppc",
        "reason": "Preserves a joint temporal diagnostic outside inference schemas.",
    },
    {
        "name": "rate_waveform_quantiles",
        "source_signal": "cortex_r_exc + cortex_r_inh",
        "role": "synthetic_held_out_ppc",
        "reason": "Distribution shape is held out from mean/SD and spectral summaries.",
    },
    {
        "name": "individual_synaptic_state_variability",
        "source_signal": "cortical ALN synaptic states",
        "role": "synthetic_held_out_ppc",
        "reason": "Internal-state variability is not included in the privileged schema.",
    },
    {
        "name": "thalamic_spindle_mechanism_metrics",
        "source_signal": "thalamic internal states",
        "role": "mechanism_diagnostic",
        "reason": "Not a cortical observable and not a scalp-EEG measurement.",
    },
)


def schema_specs(schema_name: str) -> tuple[FeatureSpec, ...]:
    """Return the frozen ordered feature specification for a schema."""

    if schema_name == "cortex_rate_only_14d":
        return RATE_ONLY_FEATURES
    if schema_name == "cortex_state_augmented_24d":
        return AUGMENTED_FEATURES
    raise KeyError(f"unknown Route-3 schema: {schema_name}")


def feature_dictionary() -> pd.DataFrame:
    """Return both schemas' auditable feature dictionary."""

    rows: list[dict[str, Any]] = []
    for schema_name in ("cortex_rate_only_14d", "cortex_state_augmented_24d"):
        for index, spec in enumerate(schema_specs(schema_name)):
            row = asdict(spec)
            row.update(
                {
                    "schema": schema_name,
                    "index": index,
                    "observable_scope": (
                        "synthetic cortical-rate observable"
                        if schema_name == "cortex_rate_only_14d"
                        else "privileged model-internal upper-bound experiment"
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _parameter_contract_raw() -> dict[str, Any]:
    # fooof 1.1 installs an "always" filter during import; recording prevents
    # that known package-level deprecation notice from polluting notebook output.
    with warnings.catch_warnings(record=True):
        module = _load_model_module("v8a_t13")
    names = tuple(module.PARAM_NAMES)
    bounds = tuple(tuple(map(float, pair)) for pair in module.BOUNDS)
    if names != PARAMETER_NAMES or len(bounds) != len(PARAMETER_NAMES):
        raise RuntimeError("V8a parameter contract does not match the frozen 8D order")
    return {
        "schema_version": SCHEMA_VERSION,
        "prior_name": "v8a_uniform_box_8d",
        "prior_type": "independent uniform box",
        "parameter_names": list(names),
        "parameter_units": list(PARAMETER_UNITS),
        "bounds": [list(pair) for pair in bounds],
        "simulator_version_for_prior_wide_bank": "v8a_t13",
        "sampling_dt_ms": 1.0,
        "fs_hz": EXPECTED_FS_HZ,
        "model_integration_dt_ms": 0.1,
        "duration_s": MODEL_DURATION_S,
        "warm_up_s": WARM_UP_S,
        "analyzed_window_s": EPOCH_DURATION_S,
        "discarded_post_window_s": MODEL_DURATION_S - WARM_UP_S - EPOCH_DURATION_S,
        "simulation_seed": SIMULATION_SEED,
        "local_perturbation_fraction_of_prior_width": PERTURBATION_FRACTION,
        "effective_rank_relative_singular_value_threshold": RELATIVE_SINGULAR_VALUE_THRESHOLD,
        "failure_policy": "retain theta; store NaN x, validity false, and explicit reason",
        "nonfinite_policy": "invalid; never fill with zero",
        "scientific_scope": "synthetic model-observable recovery only; not EEG inference",
    }


def contract_hash(contract: Mapping[str, Any] | None = None) -> str:
    """Return a stable SHA-256 for the parameter and schema contract."""

    payload = dict(_parameter_contract_raw() if contract is None else contract)
    payload.pop("contract_hash", None)
    payload["rate_features"] = [asdict(spec) for spec in RATE_ONLY_FEATURES]
    payload["augmented_features"] = [asdict(spec) for spec in AUGMENTED_FEATURES]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def parameter_contract() -> dict[str, Any]:
    """Return the frozen 8D contract plus center-in-prior audit."""

    contract = _parameter_contract_raw()
    lows = np.asarray(contract["bounds"], dtype=float)[:, 0]
    highs = np.asarray(contract["bounds"], dtype=float)[:, 1]
    centers = []
    for center in audit_parameter_sets():
        theta = np.asarray([center.parameters[name] for name in PARAMETER_NAMES])
        inside = (theta >= lows) & (theta <= highs)
        centers.append(
            {
                "identifier": center.identifier,
                "model_version": center.model_version,
                "source": center.source,
                "inside_v8a_prior": bool(inside.all()),
                "outside_parameters": [
                    PARAMETER_NAMES[i] for i, value in enumerate(inside) if not value
                ],
                "theta": theta.tolist(),
            }
        )
    contract["centers"] = centers
    contract["contract_hash"] = contract_hash(contract)
    return contract


def parameter_table() -> pd.DataFrame:
    """Return parameter bounds, units, widths, and all three centers."""

    contract = parameter_contract()
    bounds = np.asarray(contract["bounds"], dtype=float)
    rows = []
    centers = {row["identifier"]: row["theta"] for row in contract["centers"]}
    for index, (name, unit) in enumerate(zip(PARAMETER_NAMES, PARAMETER_UNITS)):
        row = {
            "index": index,
            "parameter": name,
            "unit": unit,
            "prior_type": "uniform",
            "lower": bounds[index, 0],
            "upper": bounds[index, 1],
            "width": bounds[index, 1] - bounds[index, 0],
        }
        for center_name, theta in centers.items():
            row[center_name] = theta[index]
        rows.append(row)
    return pd.DataFrame(rows)


def _welch_features(signal: np.ndarray, fs_hz: float) -> dict[str, float]:
    nperseg = int(round(PSD_WINDOW_S * fs_hz))
    noverlap = int(round(PSD_OVERLAP_FRACTION * nperseg))
    frequencies, power = welch(
        signal,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    total_mask = (frequencies >= PSD_TOTAL_BAND_HZ[0]) & (
        frequencies <= PSD_TOTAL_BAND_HZ[1]
    )
    so_mask = (frequencies >= SO_BAND_HZ[0]) & (frequencies <= SO_BAND_HZ[1])
    sigma_mask = (frequencies >= SIGMA_BAND_HZ[0]) & (
        frequencies <= SIGMA_BAND_HZ[1]
    )
    selected = np.asarray(power[total_mask], dtype=float)
    total = float(selected.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("non-positive or non-finite 0.5-20 Hz Welch power")
    probabilities = selected / total
    positive = probabilities > 0.0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    entropy /= float(np.log(len(probabilities)))
    return {
        "relative_so_power": float(power[so_mask].sum() / total),
        "relative_sigma_power": float(power[sigma_mask].sum() / total),
        "so_peak_frequency_hz": float(
            frequencies[so_mask][int(np.argmax(power[so_mask]))]
        ),
        "sigma_peak_frequency_hz": float(
            frequencies[sigma_mask][int(np.argmax(power[sigma_mask]))]
        ),
        "spectral_entropy_0p5_20": entropy,
    }


def _extract_from_audit(
    audit: Any, schema_name: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    specs = schema_specs(schema_name)
    expected_samples = int(round(EPOCH_DURATION_S * audit.fs_hz))
    values: dict[str, float] = {}
    reasons: list[str] = []
    for signal_name, prefix in (
        ("cortex_r_exc", "r_exc"),
        ("cortex_r_inh", "r_inh"),
    ):
        signal = np.asarray(audit.signals[signal_name][:expected_samples], dtype=float)
        if signal.shape != (expected_samples,) or not np.isfinite(signal).all():
            reasons.append(f"{signal_name}: invalid length or non-finite samples")
            continue
        values[f"{prefix}_mean_hz"] = float(np.mean(signal))
        values[f"{prefix}_std_hz"] = float(np.std(signal, ddof=0))
        try:
            spectral = _welch_features(signal, audit.fs_hz)
        except Exception as exc:
            reasons.append(f"{signal_name}: {type(exc).__name__}: {exc}")
        else:
            for suffix, value in spectral.items():
                values[f"{prefix}_{suffix}"] = value

    if schema_name == "cortex_state_augmented_24d":
        state_pairs = (
            ("cortex_I_mu_exc", "I_mu_exc_mean_mV_per_ms", "I_mu_exc_std_mV_per_ms"),
            ("cortex_I_mu_inh", "I_mu_inh_mean_mV_per_ms", "I_mu_inh_std_mV_per_ms"),
            ("cortex_adaptation_I_A", "I_A_mean_pA", "I_A_std_pA"),
            (
                "cortex_effective_drive",
                "effective_drive_mean_mV_per_ms",
                "effective_drive_std_mV_per_ms",
            ),
        )
        for signal_name, mean_name, std_name in state_pairs:
            signal = np.asarray(
                audit.signals[signal_name][:expected_samples], dtype=float
            )
            if signal.shape != (expected_samples,) or not np.isfinite(signal).all():
                reasons.append(f"{signal_name}: invalid length or non-finite samples")
                continue
            values[mean_name] = float(np.mean(signal))
            values[std_name] = float(np.std(signal, ddof=0))
        for signal_name, feature_name in (
            ("cortex_syn_mu_exc_on_exc", "syn_mu_exc_on_exc_mean"),
            ("cortex_syn_mu_inh_on_exc", "syn_mu_inh_on_exc_mean"),
        ):
            signal = np.asarray(
                audit.signals[signal_name][:expected_samples], dtype=float
            )
            if signal.shape != (expected_samples,) or not np.isfinite(signal).all():
                reasons.append(f"{signal_name}: invalid length or non-finite samples")
                continue
            values[feature_name] = float(np.mean(signal))

    vector = np.asarray([values.get(spec.name, np.nan) for spec in specs], dtype=float)
    validity = np.isfinite(vector)
    missing = [
        f"{spec.name}: undefined after extraction"
        for spec, is_valid in zip(specs, validity)
        if not is_valid
    ]
    return vector, validity, reasons + missing


def simulate_features(
    parameter_set: ParameterSet,
) -> dict[str, Any]:
    """Run one simulator and extract both nested Route-3 schemas."""

    started = perf_counter()
    try:
        audit = run_state_audit(parameter_set)
        rate_vector, rate_validity, rate_reasons = _extract_from_audit(
            audit, "cortex_rate_only_14d"
        )
        aug_vector, aug_validity, aug_reasons = _extract_from_audit(
            audit, "cortex_state_augmented_24d"
        )
        if not np.allclose(
            aug_vector[: len(rate_vector)],
            rate_vector,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            raise RuntimeError("augmented schema does not preserve the rate-only prefix")
        return {
            "success": bool(rate_validity.all() and aug_validity.all()),
            "identifier": parameter_set.identifier,
            "model_version": parameter_set.model_version,
            "source": parameter_set.source,
            "theta": [parameter_set.parameters[name] for name in PARAMETER_NAMES],
            "rate_vector": rate_vector,
            "augmented_vector": aug_vector,
            "rate_validity": rate_validity,
            "augmented_validity": aug_validity,
            "failure_reason": "; ".join(rate_reasons + aug_reasons),
            "runtime_s": float(audit.runtime_s),
            "fs_hz": float(audit.fs_hz),
            "post_warm_up_duration_s": float(audit.duration_s - audit.warm_up_s),
            "analyzed_duration_s": EPOCH_DURATION_S,
        }
    except Exception as exc:
        return {
            "success": False,
            "identifier": parameter_set.identifier,
            "model_version": parameter_set.model_version,
            "source": parameter_set.source,
            "theta": [parameter_set.parameters[name] for name in PARAMETER_NAMES],
            "rate_vector": np.full(len(RATE_ONLY_FEATURES), np.nan),
            "augmented_vector": np.full(len(AUGMENTED_FEATURES), np.nan),
            "rate_validity": np.zeros(len(RATE_ONLY_FEATURES), dtype=bool),
            "augmented_validity": np.zeros(len(AUGMENTED_FEATURES), dtype=bool),
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "runtime_s": float(perf_counter() - started),
            "fs_hz": np.nan,
            "post_warm_up_duration_s": np.nan,
            "analyzed_duration_s": EPOCH_DURATION_S,
        }


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _jsonable_result(result: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    for key in ("rate_vector", "augmented_vector"):
        payload[key] = [
            None if not np.isfinite(value) else float(value) for value in payload[key]
        ]
    for key in ("rate_validity", "augmented_validity"):
        payload[key] = [bool(value) for value in payload[key]]
    return payload


def _result_from_json(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    for key in ("rate_vector", "augmented_vector"):
        result[key] = np.asarray(
            [np.nan if value is None else value for value in result[key]], dtype=float
        )
    for key in ("rate_validity", "augmented_validity"):
        result[key] = np.asarray(result[key], dtype=bool)
    return result


def _local_design() -> list[tuple[str, str, str, ParameterSet]]:
    contract = parameter_contract()
    bounds = np.asarray(contract["bounds"], dtype=float)
    design: list[tuple[str, str, str, ParameterSet]] = []
    for center in audit_parameter_sets():
        design.append(
            (
                center.identifier,
                "center",
                "center",
                ParameterSet(
                    center.identifier,
                    center.model_version,
                    center.source,
                    dict(center.parameters),
                ),
            )
        )
        center_theta = np.asarray(
            [center.parameters[name] for name in PARAMETER_NAMES], dtype=float
        )
        for parameter_index, parameter_name in enumerate(PARAMETER_NAMES):
            delta = PERTURBATION_FRACTION * (
                bounds[parameter_index, 1] - bounds[parameter_index, 0]
            )
            for direction, sign in (("minus", -1.0), ("plus", 1.0)):
                theta = center_theta.copy()
                theta[parameter_index] = np.clip(
                    theta[parameter_index] + sign * delta,
                    bounds[parameter_index, 0],
                    bounds[parameter_index, 1],
                )
                identifier = f"{center.identifier}__{parameter_name}__{direction}"
                design.append(
                    (
                        center.identifier,
                        parameter_name,
                        direction,
                        ParameterSet(
                            identifier,
                            center.model_version,
                            center.source,
                            dict(zip(PARAMETER_NAMES, theta)),
                        ),
                    )
                )
    return design


def run_local_sensitivity(
    output_dir: Path | None = None, reuse_checkpoints: bool = True
) -> list[dict[str, Any]]:
    """Run or resume the 51 simulations required by central differences."""

    output_dir = Path(output_dir or RESULTS_ROOT / "local_sensitivity")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    expected_hash = contract_hash(parameter_contract())
    results: list[dict[str, Any]] = []
    design = _local_design()
    for run_index, (center_name, parameter_name, direction, parameter_set) in enumerate(
        design
    ):
        path = checkpoint_dir / f"{run_index:03d}_{parameter_set.identifier}.json"
        if reuse_checkpoints and path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("contract_hash") == expected_hash:
                result = _result_from_json(payload["result"])
                result.update(
                    {
                        "run_index": run_index,
                        "center": center_name,
                        "perturbed_parameter": parameter_name,
                        "direction": direction,
                    }
                )
                results.append(result)
                continue
        result = simulate_features(parameter_set)
        result.update(
            {
                "run_index": run_index,
                "center": center_name,
                "perturbed_parameter": parameter_name,
                "direction": direction,
            }
        )
        _atomic_json(
            {
                "contract_hash": expected_hash,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "result": _jsonable_result(result),
            },
            path,
        )
        results.append(result)
    return results


def _safe_cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 0.0 else np.nan


def analyze_local_sensitivity(
    results: Sequence[Mapping[str, Any]], output_dir: Path | None = None
) -> dict[str, Any]:
    """Compute standardized Jacobians and pre-frozen rank diagnostics."""

    output_dir = Path(output_dir or RESULTS_ROOT / "local_sensitivity")
    bounds = np.asarray(parameter_contract()["bounds"], dtype=float)
    widths = bounds[:, 1] - bounds[:, 0]
    centers = sorted({str(row["center"]) for row in results})
    matrices: dict[tuple[str, str], np.ndarray] = {}
    jacobian_rows = []
    rank_rows = []
    sensitivity_rows = []
    confounding_rows = []
    for center in centers:
        center_result = next(
            row
            for row in results
            if row["center"] == center and row["direction"] == "center"
        )
        for schema_name, vector_key, specs in (
            ("cortex_rate_only_14d", "rate_vector", RATE_ONLY_FEATURES),
            (
                "cortex_state_augmented_24d",
                "augmented_vector",
                AUGMENTED_FEATURES,
            ),
        ):
            center_vector = np.asarray(center_result[vector_key], dtype=float)
            scales = np.maximum(
                np.abs(center_vector),
                np.asarray([spec.scale_floor for spec in specs], dtype=float),
            )
            jacobian = np.full((len(specs), len(PARAMETER_NAMES)), np.nan)
            for parameter_index, parameter_name in enumerate(PARAMETER_NAMES):
                minus = next(
                    row
                    for row in results
                    if row["center"] == center
                    and row["perturbed_parameter"] == parameter_name
                    and row["direction"] == "minus"
                )
                plus = next(
                    row
                    for row in results
                    if row["center"] == center
                    and row["perturbed_parameter"] == parameter_name
                    and row["direction"] == "plus"
                )
                minus_theta = float(minus["theta"][parameter_index])
                plus_theta = float(plus["theta"][parameter_index])
                standardized_step = (plus_theta - minus_theta) / widths[parameter_index]
                if standardized_step <= 0.0:
                    continue
                derivative = (
                    np.asarray(plus[vector_key], dtype=float)
                    - np.asarray(minus[vector_key], dtype=float)
                ) / scales
                jacobian[:, parameter_index] = derivative / standardized_step
            matrices[(center, schema_name)] = jacobian
            finite_matrix = bool(np.isfinite(jacobian).all())
            if finite_matrix:
                singular_values = np.linalg.svd(jacobian, compute_uv=False)
                relative = singular_values / singular_values[0]
                effective_rank = int(
                    np.sum(relative >= RELATIVE_SINGULAR_VALUE_THRESHOLD)
                )
                condition = (
                    float(singular_values[0] / singular_values[-1])
                    if singular_values[-1] > 0.0
                    else np.inf
                )
            else:
                singular_values = np.full(len(PARAMETER_NAMES), np.nan)
                relative = singular_values.copy()
                effective_rank = 0
                condition = np.inf
            rank_rows.append(
                {
                    "center": center,
                    "schema": schema_name,
                    "n_features": len(specs),
                    "finite_jacobian": finite_matrix,
                    "effective_rank": effective_rank,
                    "full_local_rank": effective_rank == len(PARAMETER_NAMES),
                    "rank_threshold_relative": RELATIVE_SINGULAR_VALUE_THRESHOLD,
                    "condition_number": condition,
                    **{
                        f"singular_value_{index + 1}": value
                        for index, value in enumerate(singular_values)
                    },
                    **{
                        f"relative_singular_value_{index + 1}": value
                        for index, value in enumerate(relative)
                    },
                }
            )
            for parameter_index, parameter_name in enumerate(PARAMETER_NAMES):
                sensitivity_rows.append(
                    {
                        "center": center,
                        "schema": schema_name,
                        "parameter": parameter_name,
                        "sensitivity_norm": float(
                            np.linalg.norm(jacobian[:, parameter_index])
                        )
                        if finite_matrix
                        else np.nan,
                    }
                )
            for first_index in range(len(PARAMETER_NAMES)):
                for second_index in range(first_index + 1, len(PARAMETER_NAMES)):
                    confounding_rows.append(
                        {
                            "center": center,
                            "schema": schema_name,
                            "parameter_a": PARAMETER_NAMES[first_index],
                            "parameter_b": PARAMETER_NAMES[second_index],
                            "jacobian_column_cosine": _safe_cosine(
                                jacobian[:, first_index], jacobian[:, second_index]
                            )
                            if finite_matrix
                            else np.nan,
                        }
                    )
            for feature_index, spec in enumerate(specs):
                for parameter_index, parameter_name in enumerate(PARAMETER_NAMES):
                    jacobian_rows.append(
                        {
                            "center": center,
                            "schema": schema_name,
                            "feature": spec.name,
                            "parameter": parameter_name,
                            "standardized_derivative": jacobian[
                                feature_index, parameter_index
                            ],
                        }
                    )

    direction_rows = []
    for schema_name in ("cortex_rate_only_14d", "cortex_state_augmented_24d"):
        for parameter_index, parameter_name in enumerate(PARAMETER_NAMES):
            for first_index in range(len(centers)):
                for second_index in range(first_index + 1, len(centers)):
                    first_center = centers[first_index]
                    second_center = centers[second_index]
                    direction_rows.append(
                        {
                            "schema": schema_name,
                            "parameter": parameter_name,
                            "center_a": first_center,
                            "center_b": second_center,
                            "sensitivity_direction_cosine": _safe_cosine(
                                matrices[(first_center, schema_name)][
                                    :, parameter_index
                                ],
                                matrices[(second_center, schema_name)][
                                    :, parameter_index
                                ],
                            ),
                        }
                    )

    tables = {
        "jacobian": pd.DataFrame(jacobian_rows),
        "rank": pd.DataFrame(rank_rows),
        "sensitivity": pd.DataFrame(sensitivity_rows),
        "confounding": pd.DataFrame(confounding_rows),
        "direction_consistency": pd.DataFrame(direction_rows),
    }
    for name, frame in tables.items():
        _atomic_csv(frame, output_dir / f"{name}.csv")
    run_rows = []
    for row in results:
        run_rows.append(
            {
                "run_index": row["run_index"],
                "center": row["center"],
                "perturbed_parameter": row["perturbed_parameter"],
                "direction": row["direction"],
                "identifier": row["identifier"],
                "model_version": row["model_version"],
                "success": row["success"],
                "runtime_s": row["runtime_s"],
                "failure_reason": row["failure_reason"],
                "rate_finite": bool(np.isfinite(row["rate_vector"]).all()),
                "augmented_finite": bool(
                    np.isfinite(row["augmented_vector"]).all()
                ),
            }
        )
    _atomic_csv(pd.DataFrame(run_rows), output_dir / "simulation_runs.csv")
    return {"tables": tables, "matrices": matrices}


def local_stability_gate(
    results: Sequence[Mapping[str, Any]], schema_name: str
) -> tuple[bool, str]:
    """Apply the numerical gate for diagnostic prior-wide sampling."""

    key = (
        "rate_vector"
        if schema_name == "cortex_rate_only_14d"
        else "augmented_vector"
    )
    if not results:
        return False, "no local simulations"
    failed = [row for row in results if not np.isfinite(row[key]).all()]
    if failed:
        return False, f"{len(failed)} local simulations have non-finite {schema_name}"
    expected = len(schema_specs(schema_name))
    if any(len(row[key]) != expected for row in results):
        return False, "feature length changed across local simulations"
    return True, "fixed order, fixed length, finite local simulations"


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def run_diagnostic_microbank(
    n_samples: int = 64,
    output_dir: Path | None = None,
    reuse_checkpoint: bool = True,
) -> dict[str, Any]:
    """Run or resume a Sobol diagnostic micro-bank, checkpointing each row."""

    if n_samples < 1 or n_samples > 64:
        raise ValueError("diagnostic micro-bank must contain between 1 and 64 rows")
    output_dir = Path(output_dir or RESULTS_ROOT / "diagnostic_microbank")
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = parameter_contract()
    expected_hash = contract_hash(contract)
    bounds = np.asarray(contract["bounds"], dtype=float)
    if n_samples == 64:
        unit = qmc.Sobol(d=len(PARAMETER_NAMES), scramble=True, seed=SOBOL_SEED).random_base2(
            m=6
        )
    else:
        unit = qmc.Sobol(d=len(PARAMETER_NAMES), scramble=True, seed=SOBOL_SEED).random(
            n=n_samples
        )
    theta = qmc.scale(unit, bounds[:, 0], bounds[:, 1])
    rate_x = np.full((n_samples, len(RATE_ONLY_FEATURES)), np.nan)
    augmented_x = np.full((n_samples, len(AUGMENTED_FEATURES)), np.nan)
    rate_validity = np.zeros_like(rate_x, dtype=bool)
    augmented_validity = np.zeros_like(augmented_x, dtype=bool)
    runtimes = np.full(n_samples, np.nan)
    completed = np.zeros(n_samples, dtype=bool)
    success = np.zeros(n_samples, dtype=bool)
    failure_reason = np.full(n_samples, "", dtype="<U1024")
    checkpoint_path = output_dir / f"diagnostic_microbank_{n_samples}.npz"

    if reuse_checkpoint and checkpoint_path.exists():
        loaded = np.load(checkpoint_path, allow_pickle=False)
        loaded_hash = str(loaded["contract_hash"].item())
        if loaded_hash != expected_hash:
            raise RuntimeError("existing diagnostic micro-bank contract hash mismatch")
        if not np.allclose(loaded["theta"], theta, rtol=0.0, atol=0.0):
            raise RuntimeError("existing diagnostic micro-bank Sobol theta mismatch")
        for name, target in (
            ("rate_x", rate_x),
            ("augmented_x", augmented_x),
            ("rate_validity", rate_validity),
            ("augmented_validity", augmented_validity),
            ("runtime_s", runtimes),
            ("completed", completed),
            ("success", success),
            ("failure_reason", failure_reason),
        ):
            target[...] = loaded[name]

    for index in range(n_samples):
        if completed[index]:
            continue
        parameter_set = ParameterSet(
            identifier=f"sobol_{index:03d}",
            model_version="v8a_t13",
            source="generated: v8a_uniform_box_8d Sobol diagnostic sample",
            parameters=dict(zip(PARAMETER_NAMES, theta[index])),
        )
        result = simulate_features(parameter_set)
        rate_x[index] = result["rate_vector"]
        augmented_x[index] = result["augmented_vector"]
        rate_validity[index] = result["rate_validity"]
        augmented_validity[index] = result["augmented_validity"]
        runtimes[index] = result["runtime_s"]
        success[index] = bool(result["success"])
        failure_reason[index] = str(result["failure_reason"])[:1024]
        completed[index] = True
        _atomic_npz(
            checkpoint_path,
            theta=theta,
            rate_x=rate_x,
            augmented_x=augmented_x,
            rate_validity=rate_validity,
            augmented_validity=augmented_validity,
            runtime_s=runtimes,
            completed=completed,
            success=success,
            failure_reason=failure_reason,
            parameter_names=np.asarray(PARAMETER_NAMES, dtype="<U32"),
            rate_feature_names=np.asarray(
                [spec.name for spec in RATE_ONLY_FEATURES], dtype="<U64"
            ),
            augmented_feature_names=np.asarray(
                [spec.name for spec in AUGMENTED_FEATURES], dtype="<U64"
            ),
            simulation_seed=np.asarray(SIMULATION_SEED, dtype=np.int64),
            sobol_seed=np.asarray(SOBOL_SEED, dtype=np.int64),
            sobol_index=np.arange(n_samples, dtype=np.int64),
            schema_version=np.asarray(SCHEMA_VERSION, dtype="<U64"),
            contract_hash=np.asarray(expected_hash, dtype="<U64"),
            bank_role=np.asarray(
                "diagnostic_microbank_not_for_SNPE_training", dtype="<U64"
            ),
        )

    rows = []
    for index in range(n_samples):
        rows.append(
            {
                "sobol_index": index,
                "completed": bool(completed[index]),
                "success": bool(success[index]),
                "runtime_s": runtimes[index],
                "rate_finite": bool(np.isfinite(rate_x[index]).all()),
                "augmented_finite": bool(np.isfinite(augmented_x[index]).all()),
                "failure_reason": failure_reason[index],
            }
        )
    _atomic_csv(pd.DataFrame(rows), output_dir / "diagnostic_microbank_status.csv")
    return {
        "path": checkpoint_path,
        "theta": theta,
        "rate_x": rate_x,
        "augmented_x": augmented_x,
        "rate_validity": rate_validity,
        "augmented_validity": augmented_validity,
        "runtime_s": runtimes,
        "completed": completed,
        "success": success,
        "failure_reason": failure_reason,
    }


def feature_redundancy(
    matrix: np.ndarray, feature_names: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return Spearman correlations and pair counts without imputation."""

    matrix = np.asarray(matrix, dtype=float)
    n_features = matrix.shape[1]
    correlations = np.full((n_features, n_features), np.nan)
    counts = np.zeros((n_features, n_features), dtype=int)
    for first in range(n_features):
        for second in range(n_features):
            valid = np.isfinite(matrix[:, first]) & np.isfinite(matrix[:, second])
            counts[first, second] = int(valid.sum())
            if valid.sum() >= 3:
                correlations[first, second] = float(
                    spearmanr(matrix[valid, first], matrix[valid, second]).statistic
                )
    return (
        pd.DataFrame(correlations, index=feature_names, columns=feature_names),
        pd.DataFrame(counts, index=feature_names, columns=feature_names),
    )


def export_core_contract(output_dir: Path | None = None) -> dict[str, Path]:
    """Export the parameter/schema contract and feature dictionaries."""

    output_dir = Path(output_dir or RESULTS_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = parameter_contract()
    contract_path = output_dir / "route3_parameter_contract.json"
    feature_path = output_dir / "route3_feature_dictionary.csv"
    heldout_path = output_dir / "route3_held_out_diagnostics.csv"
    _atomic_json(contract, contract_path)
    _atomic_csv(feature_dictionary(), feature_path)
    _atomic_csv(pd.DataFrame(HELD_OUT_DIAGNOSTICS), heldout_path)
    return {
        "contract": contract_path,
        "features": feature_path,
        "held_out": heldout_path,
    }


__all__ = [
    "AUGMENTED_FEATURES",
    "EXPECTED_FS_HZ",
    "HELD_OUT_DIAGNOSTICS",
    "PARAMETER_NAMES",
    "RATE_ONLY_FEATURES",
    "RELATIVE_SINGULAR_VALUE_THRESHOLD",
    "RESULTS_ROOT",
    "SCHEMA_VERSION",
    "analyze_local_sensitivity",
    "contract_hash",
    "export_core_contract",
    "feature_dictionary",
    "feature_redundancy",
    "local_stability_gate",
    "parameter_contract",
    "parameter_table",
    "run_diagnostic_microbank",
    "run_local_sensitivity",
    "schema_specs",
    "simulate_features",
]
