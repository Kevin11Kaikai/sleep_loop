"""Frozen PHASE2_CONTROL_BENCH_V1 executable implementation.

This module is a fresh, current-campaign clean-room implementation.  It contains
no repository model, fitted parameter, subject datum, or scientific outcome.
Every applied input is represented as GENERIC EXTERNAL FORCING.

The public API is intentionally split into deterministic simulation/detection,
registered estimands/statistics, execution planning/checkpointing, and
publication-safe plotting.  Importing this module never launches a simulation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy import signal, stats
from scipy.stats import qmc

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - frozen environment contains numba
    NUMBA_AVAILABLE = False

    def njit(*args: Any, **kwargs: Any) -> Callable[..., Any]:
        def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorate


SCHEMA_VERSION = "1.0.0"
CAMPAIGN_ID = "COSTA_PHASE2_FRESH_20260816_105951"
BENCH_ID = "PHASE2_CONTROL_BENCH_V1"
PROTOCOL_FREEZE_SHA256 = (
    "1A27094D05DD24848E7F8D9643936D69078464B864EFA18621E022A04410F3F0"
)
ENVIRONMENT_VERSION = "NEUROLIB_ENV_V3"
GENERIC_FORCING_NAME = "GENERIC EXTERNAL FORCING"
INTERNAL_DT_SECONDS = 0.001
STORED_FS_HZ = 200.0
WARMUP_SECONDS = 30.0
ANALYSIS_SECONDS = 180.0
EDGE_SECONDS = 10.0
INITIAL_STATE = np.array([8.0, 10.0, 6.0, 8.0, 8.0, 0.0, 0.0, 0.0, 0.0])
WORKER_SEEDS = (11003, 22007, 33013, 44017, 55021, 66029)
VERIFIER_SEEDS = (77041, 88069, 99079, 111091)
DRAW_IDS = ("D00_NOMINAL", "D01", "D02", "D03", "D04", "D05")
TARGET_IDS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
PRIMARY_TARGETS = ("T1", "T4", "T6")
SECONDARY_TARGETS = ("T2", "T3", "T5", "T7")
EXPLORATORY_TARGETS = ("T8",)
BOOTSTRAP_SEED = 860216
TEST_SEED = 860217
PHASE2B_BOOTSTRAP_SEED = 860219
BOOTSTRAP_RESAMPLES = 10_000
SIGN_FLIP_DRAWS = 100_000
SURROGATE_COUNT = 200


PARAMETER_DEFAULTS: dict[str, float] = {
    "tau_E": 0.020,
    "tau_I": 0.010,
    "tau_T": 0.020,
    "tau_R": 0.010,
    "tau_a": 1.000,
    "tau_eta": 0.050,
    "rmax_E": 100.0,
    "rmax_I": 100.0,
    "rmax_T": 100.0,
    "rmax_R": 100.0,
    "beta": 0.18,
    "theta": 15.0,
    "b_E": 12.0,
    "b_I": 8.0,
    "b_T": 10.0,
    "b_R": 8.0,
    "g_a": 0.30,
    "w_EE": 0.22,
    "w_EI": 0.35,
    "w_ET": 0.16,
    "w_IE": 0.30,
    "w_II": 0.10,
    "w_IT": 0.10,
    "w_TE": 0.12,
    "w_TR": 0.40,
    "w_RE": 0.08,
    "w_RT": 0.35,
    "w_RR": 0.08,
    "sigma_E": 0.25,
    "sigma_I": 0.20,
    "sigma_T": 0.20,
    "sigma_R": 0.20,
}

PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "tau_E": (0.015, 0.030),
    "tau_I": (0.0075, 0.015),
    "tau_T": (0.015, 0.030),
    "tau_R": (0.0075, 0.015),
    "tau_a": (0.600, 1.800),
    "tau_eta": (0.025, 0.100),
    "rmax_E": (80.0, 120.0),
    "rmax_I": (80.0, 120.0),
    "rmax_T": (80.0, 120.0),
    "rmax_R": (80.0, 120.0),
    "beta": (0.12, 0.24),
    "theta": (12.0, 18.0),
    "b_E": (8.0, 16.0),
    "b_I": (5.0, 12.0),
    "b_T": (6.0, 14.0),
    "b_R": (5.0, 12.0),
    "g_a": (0.15, 0.45),
    "w_EE": (0.12, 0.32),
    "w_EI": (0.22, 0.50),
    "w_ET": (0.08, 0.25),
    "w_IE": (0.18, 0.42),
    "w_II": (0.04, 0.18),
    "w_IT": (0.04, 0.18),
    "w_TE": (0.05, 0.22),
    "w_TR": (0.25, 0.55),
    "w_RE": (0.03, 0.16),
    "w_RT": (0.22, 0.50),
    "w_RR": (0.03, 0.16),
    "sigma_E": (0.10, 0.50),
    "sigma_I": (0.08, 0.40),
    "sigma_T": (0.08, 0.40),
    "sigma_R": (0.08, 0.40),
}

VARIED_PARAMETERS = (
    "tau_E",
    "tau_I",
    "tau_T",
    "tau_R",
    "tau_a",
    "beta",
    "theta",
    "b_E",
    "b_T",
    "g_a",
    "w_EE",
    "w_EI",
    "w_ET",
    "w_TE",
    "w_TR",
    "w_RT",
    "w_RR",
)
ADDITIVE_BIASES = frozenset(("b_E", "b_T"))


CHANNEL_CODES = {
    "SHAM": 0,
    "GF_E_DC": 1,
    "GF_E_SLOW_SINE": 2,
    "GF_E_SO_PULSE": 3,
    "GF_TR_SIGMA_PACKET_BALANCE": 4,
    "GF_TR_SIGMA_CONTINUOUS_BALANCE": 5,
    "GF_T_PHASE_LOCKED_PACKET": 6,
    "GF_BALANCED_DC": 7,
    "GF_E_DC_X_GF_E_SLOW_SINE": 8,
}

LEVEL_A_GRIDS: dict[str, dict[str, Any]] = {
    "T1": {
        "channel": "GF_E_SLOW_SINE",
        "values": [0.55, 0.70, 0.85, 1.00, 1.15],
        "coordinate": "drive_frequency_hz",
        "amplitude": 0.8,
    },
    "T2": {
        "channel": "GF_E_DC",
        "values": [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        "coordinate": "dose",
    },
    "T3": {
        "channel": "GF_E_SO_PULSE",
        "values": [-2.0, -1.25, -0.5, 0.0, 0.5, 1.25, 2.0],
        "coordinate": "dose",
        "repetition_hz": 0.85,
    },
    "T4": {
        "channel": "GF_TR_SIGMA_PACKET_BALANCE",
        "values": [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        "coordinate": "dose",
        "carrier_hz": 13.0,
    },
    "T5": {
        "channel": "GF_TR_SIGMA_CONTINUOUS_BALANCE",
        "values": [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2],
        "coordinate": "dose",
        "carrier_hz": 13.0,
    },
    "T6_T7_SHARED": {
        "channel": "GF_T_PHASE_LOCKED_PACKET",
        "values": [
            -math.pi,
            -2.0 * math.pi / 3.0,
            -math.pi / 3.0,
            0.0,
            math.pi / 3.0,
            2.0 * math.pi / 3.0,
            math.pi,
        ],
        "coordinate": "phase_offset_radians",
        "amplitude": 1.0,
        "carrier_hz": 13.0,
    },
    "T8": {
        "channel": "GF_BALANCED_DC",
        "values": [-2.0, -1.25, -0.5, 0.0, 0.5, 1.25, 2.0],
        "coordinate": "dose",
    },
}

PRACTICAL_MARGINS: dict[str, float] = {
    "T1": 0.05,
    "T2": 0.15,
    "T3": 3.0,
    "T4": 1.0,
    "T5": 0.15,
    "T6": 0.010,
    "T7": math.pi / 3.0,
    "T8": 1.0 / 3.0,
}


@dataclass(frozen=True)
class ForcingConfig:
    """One frozen GENERIC EXTERNAL FORCING condition."""

    channel: str = "SHAM"
    dose: float = 0.0
    frequency_hz: float = 0.85
    amplitude: float = 0.0
    phase_offset_radians: float = 0.0
    carrier_hz: float = 13.0
    repetition_hz: float = 0.85
    exposure_fraction: float = 1.0
    secondary_amplitude: float = 0.0

    def normalized(self) -> "ForcingConfig":
        if self.channel not in CHANNEL_CODES:
            raise ValueError(f"Unknown frozen forcing channel: {self.channel}")
        if not (0.0 <= self.exposure_fraction <= 1.0):
            raise ValueError("exposure_fraction must be in [0,1]")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": GENERIC_FORCING_NAME,
            "channel": self.channel,
            "dose": float(self.dose),
            "frequency_hz": float(self.frequency_hz),
            "amplitude": float(self.amplitude),
            "phase_offset_radians": float(self.phase_offset_radians),
            "carrier_hz": float(self.carrier_hz),
            "repetition_hz": float(self.repetition_hz),
            "exposure_fraction": float(self.exposure_fraction),
            "secondary_amplitude": float(self.secondary_amplitude),
        }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def generate_parameter_draws() -> dict[str, dict[str, float]]:
    """Generate the nominal and five frozen synthetic-only uncertainty draws."""

    draws: dict[str, dict[str, float]] = {"D00_NOMINAL": dict(PARAMETER_DEFAULTS)}
    sampler = qmc.LatinHypercube(d=17, scramble=True, seed=271828)
    unit = sampler.random(n=5)
    z_values = stats.norm.ppf(unit)
    for row_index, draw_id in enumerate(DRAW_IDS[1:]):
        params = dict(PARAMETER_DEFAULTS)
        for col_index, name in enumerate(VARIED_PARAMETERS):
            lower, upper = PARAMETER_BOUNDS[name]
            z = float(z_values[row_index, col_index])
            if name in ADDITIVE_BIASES:
                proposed = PARAMETER_DEFAULTS[name] + 0.05 * (upper - lower) * z
            else:
                proposed = PARAMETER_DEFAULTS[name] * math.exp(0.05 * z)
            params[name] = float(np.clip(proposed, lower, upper))
        draws[draw_id] = params
    return draws


def parameter_draw_manifest() -> dict[str, Any]:
    draws = generate_parameter_draws()
    return {
        "status": "SYNTHETIC_ONLY_NOT_PERSONALIZATION",
        "generator": "scipy.stats.qmc.LatinHypercube",
        "qmc_seed": 271828,
        "draw_ids": list(DRAW_IDS),
        "varied_parameters": list(VARIED_PARAMETERS),
        "draws": draws,
        "sha256": sha256_bytes(canonical_json_bytes(draws)),
    }


PARAMETER_ORDER = tuple(PARAMETER_DEFAULTS)


def _parameter_vector(parameters: Mapping[str, float]) -> np.ndarray:
    missing = [name for name in PARAMETER_ORDER if name not in parameters]
    if missing:
        raise ValueError(f"Missing bench parameters: {missing}")
    vector = np.array([float(parameters[name]) for name in PARAMETER_ORDER], dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError("All parameters must be finite")
    return vector


def generate_ou_innovations(
    seed: int,
    warmup_seconds: float = WARMUP_SECONDS,
    analysis_seconds: float = ANALYSIS_SECONDS,
    dt: float = INTERNAL_DT_SECONDS,
) -> np.ndarray:
    steps = int(round((warmup_seconds + analysis_seconds) / dt))
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    innovations = rng.standard_normal((steps, 4), dtype=np.float64)
    return np.ascontiguousarray(innovations)


def innovation_sha256(innovations: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(innovations, dtype=np.float64)
    return sha256_bytes(memoryview(contiguous).cast("B"))


def _causal_filter_contract(dt: float) -> tuple[np.ndarray, float]:
    fs = 1.0 / dt
    sos = signal.butter(4, (0.3, 1.5), btype="bandpass", fs=fs, output="sos")
    # Frozen calibration convention: phase delay of the registered causal SOS at
    # the trigger's nominal 0.85 Hz, mapped to the positive sub-cycle delay.
    _, response = signal.sosfreqz(sos, worN=np.array([0.85]), fs=fs)
    phase = float(np.angle(response[0]))
    delay = ((-phase) % (2.0 * math.pi)) / (2.0 * math.pi * 0.85)
    delay %= 1.0 / 0.85
    return np.ascontiguousarray(sos, dtype=np.float64), float(delay)


@njit(cache=True)
def _stable_sigmoid_input(h: float, beta: float, theta: float, rmax: float) -> float:
    z = beta * (h - theta)
    if z >= 0.0:
        return rmax / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return rmax * ez / (1.0 + ez)


@njit(cache=True)
def _apply_sos_sample(x: float, sos: np.ndarray, zi: np.ndarray) -> float:
    y = x
    for section in range(sos.shape[0]):
        b0 = sos[section, 0]
        b1 = sos[section, 1]
        b2 = sos[section, 2]
        a1 = sos[section, 4]
        a2 = sos[section, 5]
        out = b0 * y + zi[section, 0]
        zi[section, 0] = b1 * y - a1 * out + zi[section, 1]
        zi[section, 1] = b2 * y - a2 * out
        y = out
    return y


@njit(cache=True)
def _forcing_values(
    code: int,
    analysis_t: float,
    dose: float,
    frequency: float,
    amplitude: float,
    carrier: float,
    repetition: float,
    exposure_fraction: float,
    secondary_amplitude: float,
    packet_start_t: float,
) -> tuple[float, float, float, float]:
    u_e = 0.0
    u_i = 0.0
    u_t = 0.0
    u_r = 0.0
    if analysis_t < 0.0:
        return u_e, u_i, u_t, u_r

    if code == 1:  # GF_E_DC
        u_e = dose
    elif code == 2:  # GF_E_SLOW_SINE
        u_e = amplitude * math.sin(2.0 * math.pi * frequency * analysis_t)
    elif code == 3:  # GF_E_SO_PULSE
        phase = analysis_t % (1.0 / repetition)
        if phase < 0.080:
            u_e = dose
    elif code == 4:  # GF_TR_SIGMA_PACKET_BALANCE
        age = analysis_t % (1.0 / repetition)
        if age < 0.500:
            hann = 0.5 * (1.0 - math.cos(2.0 * math.pi * age / 0.500))
            packet = hann * 0.5 * (1.0 + math.sin(2.0 * math.pi * carrier * age))
            if dose >= 0.0:
                u_t = dose * packet
            else:
                u_r = -dose * packet
    elif code == 5:  # GF_TR_SIGMA_CONTINUOUS_BALANCE
        wave = 0.5 * (1.0 + math.sin(2.0 * math.pi * carrier * analysis_t))
        if dose >= 0.0:
            u_t = dose * wave
        else:
            u_r = -dose * wave
    elif code == 6:  # GF_T_PHASE_LOCKED_PACKET
        age = analysis_t - packet_start_t
        if age >= 0.0 and age < 0.500:
            hann = 0.5 * (1.0 - math.cos(2.0 * math.pi * age / 0.500))
            packet = hann * 0.5 * (1.0 + math.sin(2.0 * math.pi * carrier * age))
            u_t = amplitude * packet
    elif code == 7:  # GF_BALANCED_DC
        within_window = analysis_t % 30.0
        if within_window < 30.0 * exposure_fraction:
            if dose >= 0.0:
                u_e = dose
                u_t = 0.5 * dose
            else:
                u_i = -dose
                u_r = -0.5 * dose
    elif code == 8:  # conditional T2 surface, registered channel crossing
        u_e = dose + secondary_amplitude * math.sin(2.0 * math.pi * 0.85 * analysis_t)
    return u_e, u_i, u_t, u_r


@njit(cache=True)
def _simulate_kernel(
    params: np.ndarray,
    innovations: np.ndarray,
    initial_state: np.ndarray,
    dt: float,
    warmup_steps: int,
    store_stride: int,
    channel_code: int,
    dose: float,
    frequency: float,
    amplitude: float,
    phase_offset: float,
    carrier: float,
    repetition: float,
    exposure_fraction: float,
    secondary_amplitude: float,
    causal_sos: np.ndarray,
    causal_delay_seconds: float,
    keep_internal: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    # PARAMETER_ORDER positional unpacking.
    tau_e, tau_i, tau_t, tau_r, tau_a, tau_eta = params[0:6]
    rmax_e, rmax_i, rmax_t, rmax_r = params[6:10]
    beta, theta, b_e, b_i, b_t, b_r, g_a = params[10:17]
    w_ee, w_ei, w_et, w_ie, w_ii, w_it = params[17:23]
    w_te, w_tr, w_re, w_rt, w_rr = params[23:28]
    sigma_e, sigma_i, sigma_t, sigma_r = params[28:32]

    total_steps = innovations.shape[0]
    analysis_steps = total_steps - warmup_steps
    stored_count = analysis_steps // store_stride
    stored = np.empty((stored_count, 5), dtype=np.float64)
    internal = np.empty((analysis_steps + 1, 5), dtype=np.float64) if keep_internal else np.empty((0, 5))
    state = initial_state.copy()
    rho = math.exp(-dt / tau_eta)
    ou_scale = math.sqrt(1.0 - rho * rho)
    sigmas = np.array([sigma_e, sigma_i, sigma_t, sigma_r])
    zi = np.zeros((causal_sos.shape[0], 2), dtype=np.float64)
    previous_filtered = 0.0
    last_cross_t = -1.0e30
    estimated_period = 1.0 / 0.85
    packet_start_t = -1.0e30
    trigger_count = 0
    stored_index = 0
    internal_index = 0

    for step in range(total_steps):
        t = step * dt
        analysis_t = t - warmup_steps * dt
        r_e, r_i, r_t, r_r, adapt = state[0], state[1], state[2], state[3], state[4]
        eta_e, eta_i, eta_t, eta_r = state[5], state[6], state[7], state[8]

        observation = (r_e + 0.25 * r_t) / 10.0
        causal_value = _apply_sos_sample(observation, causal_sos, zi)
        if channel_code == 6 and analysis_t >= 0.0 and previous_filtered <= 0.0 and causal_value > 0.0:
            if last_cross_t > -1.0e20:
                candidate_period = analysis_t - last_cross_t
                if candidate_period >= 0.4 and candidate_period <= 2.5:
                    estimated_period = candidate_period
            wrapped_phase = phase_offset % (2.0 * math.pi)
            requested_delay = wrapped_phase * estimated_period / (2.0 * math.pi)
            corrected_delay = requested_delay - causal_delay_seconds
            while corrected_delay < 0.0:
                corrected_delay += estimated_period
            packet_start_t = analysis_t + corrected_delay
            last_cross_t = analysis_t
            trigger_count += 1
        previous_filtered = causal_value

        if step == warmup_steps:
            if keep_internal:
                internal[0, 0:5] = state[0:5]
                internal_index = 1
            if stored_count > 0:
                stored[0, 0:5] = state[0:5]
                stored_index = 1

        eta_next_e = rho * eta_e + sigma_e * ou_scale * innovations[step, 0]
        eta_next_i = rho * eta_i + sigma_i * ou_scale * innovations[step, 1]
        eta_next_t = rho * eta_t + sigma_t * ou_scale * innovations[step, 2]
        eta_next_r = rho * eta_r + sigma_r * ou_scale * innovations[step, 3]

        u_e, u_i, u_t, u_r = _forcing_values(
            channel_code,
            analysis_t,
            dose,
            frequency,
            amplitude,
            carrier,
            repetition,
            exposure_fraction,
            secondary_amplitude,
            packet_start_t,
        )
        next_t = analysis_t + dt
        nu_e, nu_i, nu_t, nu_r = _forcing_values(
            channel_code,
            next_t,
            dose,
            frequency,
            amplitude,
            carrier,
            repetition,
            exposure_fraction,
            secondary_amplitude,
            packet_start_t,
        )

        h_e = b_e + w_ee * r_e - w_ei * r_i + w_et * r_t - g_a * adapt + eta_e + u_e
        h_i = b_i + w_ie * r_e - w_ii * r_i + w_it * r_t + eta_i + u_i
        h_t = b_t + w_te * r_e - w_tr * r_r + eta_t + u_t
        h_r = b_r + w_re * r_e + w_rt * r_t - w_rr * r_r + eta_r + u_r
        k_e = (-r_e + _stable_sigmoid_input(h_e, beta, theta, rmax_e)) / tau_e
        k_i = (-r_i + _stable_sigmoid_input(h_i, beta, theta, rmax_i)) / tau_i
        k_t = (-r_t + _stable_sigmoid_input(h_t, beta, theta, rmax_t)) / tau_t
        k_r = (-r_r + _stable_sigmoid_input(h_r, beta, theta, rmax_r)) / tau_r
        k_a = (-adapt + r_e) / tau_a

        pred_e = r_e + dt * k_e
        pred_i = r_i + dt * k_i
        pred_t = r_t + dt * k_t
        pred_r = r_r + dt * k_r
        pred_a = adapt + dt * k_a
        hp_e = b_e + w_ee * pred_e - w_ei * pred_i + w_et * pred_t - g_a * pred_a + eta_next_e + nu_e
        hp_i = b_i + w_ie * pred_e - w_ii * pred_i + w_it * pred_t + eta_next_i + nu_i
        hp_t = b_t + w_te * pred_e - w_tr * pred_r + eta_next_t + nu_t
        hp_r = b_r + w_re * pred_e + w_rt * pred_t - w_rr * pred_r + eta_next_r + nu_r
        q_e = (-pred_e + _stable_sigmoid_input(hp_e, beta, theta, rmax_e)) / tau_e
        q_i = (-pred_i + _stable_sigmoid_input(hp_i, beta, theta, rmax_i)) / tau_i
        q_t = (-pred_t + _stable_sigmoid_input(hp_t, beta, theta, rmax_t)) / tau_t
        q_r = (-pred_r + _stable_sigmoid_input(hp_r, beta, theta, rmax_r)) / tau_r
        q_a = (-pred_a + pred_e) / tau_a

        state[0] = r_e + 0.5 * dt * (k_e + q_e)
        state[1] = r_i + 0.5 * dt * (k_i + q_i)
        state[2] = r_t + 0.5 * dt * (k_t + q_t)
        state[3] = r_r + 0.5 * dt * (k_r + q_r)
        state[4] = adapt + 0.5 * dt * (k_a + q_a)
        state[5] = eta_next_e
        state[6] = eta_next_i
        state[7] = eta_next_t
        state[8] = eta_next_r

        completed = step + 1
        if completed > warmup_steps:
            if keep_internal and internal_index < internal.shape[0]:
                internal[internal_index, 0:5] = state[0:5]
                internal_index += 1
            if (completed - warmup_steps) % store_stride == 0 and stored_index < stored_count:
                stored[stored_index, 0:5] = state[0:5]
                stored_index += 1

    return stored, internal, trigger_count


def simulate_bench(
    parameters: Mapping[str, float] | None = None,
    seed: int = 123457,
    forcing: ForcingConfig | None = None,
    warmup_seconds: float = WARMUP_SECONDS,
    analysis_seconds: float = ANALYSIS_SECONDS,
    innovations: np.ndarray | None = None,
    return_internal: bool = False,
) -> dict[str, Any]:
    """Run the fixed-step stochastic-Heun clean-room bench.

    The caller may supply an innovations array so paired conditions reuse the
    exact same C-order E/I/T/R stream.  No clipping is performed.
    """

    params = dict(PARAMETER_DEFAULTS if parameters is None else parameters)
    config = (forcing or ForcingConfig()).normalized()
    warmup_steps = int(round(float(warmup_seconds) / INTERNAL_DT_SECONDS))
    analysis_steps = int(round(float(analysis_seconds) / INTERNAL_DT_SECONDS))
    expected_steps = warmup_steps + analysis_steps
    if analysis_steps <= 0 or analysis_steps % 5:
        raise ValueError("analysis duration must contain a positive multiple of five 1 ms steps")
    if innovations is None:
        innovations = generate_ou_innovations(
            seed, warmup_seconds, analysis_seconds, INTERNAL_DT_SECONDS
        )
    innovations = np.ascontiguousarray(innovations, dtype=np.float64)
    if innovations.shape != (expected_steps, 4):
        raise ValueError(
            f"innovation shape {innovations.shape} does not match {(expected_steps, 4)}"
        )
    causal_sos, causal_delay = _causal_filter_contract(INTERNAL_DT_SECONDS)
    stored, internal, trigger_count = _simulate_kernel(
        _parameter_vector(params),
        innovations,
        INITIAL_STATE.astype(np.float64),
        INTERNAL_DT_SECONDS,
        warmup_steps,
        5,
        CHANNEL_CODES[config.channel],
        float(config.dose),
        float(config.frequency_hz),
        float(config.amplitude),
        float(config.phase_offset_radians),
        float(config.carrier_hz),
        float(config.repetition_hz),
        float(config.exposure_fraction),
        float(config.secondary_amplitude),
        causal_sos,
        causal_delay,
        bool(return_internal),
    )
    observation = (stored[:, 0] + 0.25 * stored[:, 2]) / 10.0
    return {
        "bench_id": BENCH_ID,
        "seed": int(seed),
        "forcing": config.as_dict(),
        "states": stored,
        "observation": observation,
        "internal_states": internal if return_internal else None,
        "innovation_sha256": innovation_sha256(innovations),
        "causal_filter_delay_seconds": causal_delay,
        "causal_trigger_count": int(trigger_count),
        "numba_enabled": bool(NUMBA_AVAILABLE),
    }

