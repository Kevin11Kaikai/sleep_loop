"""Explicit four-population thalamocortical rate model.

This module was written from first principles for the v2 clean-room route.  It
uses only NumPy.  State order is always cortical E, cortical I, thalamic relay,
and thalamic reticular.  The proxy is a declared model-space observation; it is
not a head model and is not expressed in scalp-voltage units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


STATE_NAMES = ("cortical_excitatory", "cortical_inhibitory", "thalamic_relay", "thalamic_reticular")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    cortical_gain: float
    cortex_to_thalamus: float
    thalamus_to_cortex: float
    relay_reticular_inhibition: float


@dataclass(frozen=True)
class Simulation:
    states: np.ndarray
    source_current: np.ndarray
    proxy: np.ndarray


def candidates_from_protocol(protocol: Mapping[str, object]) -> tuple[Candidate, ...]:
    """Return the prospectively enumerated candidate bank in frozen order."""
    rows = protocol["candidate_bank"]
    candidates = tuple(
        Candidate(
            candidate_id=str(row["candidate_id"]),
            cortical_gain=float(row["cortical_gain"]),
            cortex_to_thalamus=float(row["cortex_to_thalamus"]),
            thalamus_to_cortex=float(row["thalamus_to_cortex"]),
            relay_reticular_inhibition=float(row["relay_reticular_inhibition"]),
        )
        for row in rows
    )
    ids = [candidate.candidate_id for candidate in candidates]
    if len(candidates) == 0 or len(candidates) > int(protocol["selection"]["maximum_candidates"]):
        raise ValueError("candidate bank is empty or exceeds its frozen bound")
    if len(ids) != len(set(ids)):
        raise ValueError("candidate identifiers must be unique")
    return candidates


def _sigmoid(value: np.ndarray | float, slope: float) -> np.ndarray:
    clipped = np.clip(slope * np.asarray(value, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def state_to_proxy(states: np.ndarray, dt_ms: float, mapping: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen state -> source operator -> model proxy mapping."""
    weights = np.asarray(mapping["source_weights_E_I_T_R"], dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 4 or weights.shape != (4,):
        raise ValueError("expected an n-by-4 state matrix and four source weights")
    source = states @ weights
    tau_ms = float(mapping["baseline_tau_ms"])
    alpha = float(dt_ms) / tau_ms
    if not (0.0 < alpha <= 1.0):
        raise ValueError("invalid baseline filter discretization")
    baseline = np.empty_like(source)
    baseline[0] = source[0]
    for index in range(1, source.size):
        baseline[index] = baseline[index - 1] + alpha * (source[index] - baseline[index - 1])
    proxy = float(mapping["proxy_scale_model_units"]) * (source - baseline)
    return source, proxy


def simulate(
    candidate: Candidate,
    drive_e: Sequence[float],
    drive_t: Sequence[float],
    model_spec: Mapping[str, object],
    mapping: Mapping[str, object],
    *,
    cortex_to_thalamus_override: float | None = None,
    thalamus_to_cortex_override: float | None = None,
) -> Simulation:
    """Integrate the explicit E/I/relay/reticular equations by frozen Euler steps."""
    drive_e_array = np.asarray(drive_e, dtype=np.float64)
    drive_t_array = np.asarray(drive_t, dtype=np.float64)
    if drive_e_array.ndim != 1 or drive_e_array.shape != drive_t_array.shape or drive_e_array.size < 2:
        raise ValueError("drives must be same-length one-dimensional arrays")

    dt_ms = float(model_spec["dt_ms"])
    taus = np.asarray(model_spec["tau_ms_E_I_T_R"], dtype=np.float64)
    initial = np.asarray(model_spec["initial_state_E_I_T_R"], dtype=np.float64)
    slope = float(model_spec["sigmoid_slope"])
    fixed = model_spec["fixed_weights"]
    if taus.shape != (4,) or initial.shape != (4,) or np.any(taus <= dt_ms):
        raise ValueError("invalid state constants")

    g_ct = candidate.cortex_to_thalamus if cortex_to_thalamus_override is None else float(cortex_to_thalamus_override)
    g_tc = candidate.thalamus_to_cortex if thalamus_to_cortex_override is None else float(thalamus_to_cortex_override)
    states = np.empty((drive_e_array.size, 4), dtype=np.float64)
    states[0] = initial

    for index in range(1, drive_e_array.size):
        e_state, i_state, relay, reticular = states[index - 1]
        cortical_scale = candidate.cortical_gain
        e_input = (
            cortical_scale * (float(fixed["E_from_E"]) * e_state - float(fixed["E_from_I"]) * i_state)
            + g_tc * float(fixed["E_from_T"]) * relay
            + drive_e_array[index - 1]
            - float(fixed["E_threshold"])
        )
        i_input = (
            cortical_scale * (float(fixed["I_from_E"]) * e_state - float(fixed["I_from_I"]) * i_state)
            + g_tc * float(fixed["I_from_T"]) * relay
            + float(fixed["I_drive_fraction"]) * drive_e_array[index - 1]
            - float(fixed["I_threshold"])
        )
        relay_input = (
            g_ct * float(fixed["T_from_E"]) * e_state
            - candidate.relay_reticular_inhibition * reticular
            + drive_t_array[index - 1]
            - float(fixed["T_threshold"])
        )
        reticular_input = (
            g_ct * float(fixed["R_from_E"]) * e_state
            + float(fixed["R_from_T"]) * relay
            - float(fixed["R_self_inhibition"]) * reticular
            + float(fixed["R_drive_fraction"]) * drive_t_array[index - 1]
            - float(fixed["R_threshold"])
        )
        target = _sigmoid(np.array([e_input, i_input, relay_input, reticular_input]), slope)
        next_state = states[index - 1] + (dt_ms / taus) * (target - states[index - 1])
        states[index] = np.clip(next_state, 0.0, 1.0)

    source, proxy = state_to_proxy(states, dt_ms, mapping)
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(proxy)):
        raise FloatingPointError("non-finite model output")
    return Simulation(states=states, source_current=source, proxy=proxy)
