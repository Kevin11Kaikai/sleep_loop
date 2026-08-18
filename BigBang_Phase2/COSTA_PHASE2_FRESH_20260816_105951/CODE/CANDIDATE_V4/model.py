"""Exact clean-room rate model, forcing channels, OU streams, and scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import cos, exp, floor, pi, sin
from typing import Any, Mapping

import numpy as np
from scipy.special import expit
from scipy.stats import norm, qmc

from .contracts import ContractBundle, ContractError, object_sha256
from .run_registry import DRAWS, RunSpec


INTERNAL_FS = 1000
STORED_FS = 200
TOTAL_STEPS = 210_000
WARMUP_STEPS = 30_000
ANALYSIS_STEPS = 180_000
STORED_SAMPLES = 36_000
STATE_ORDER = ("r_E", "r_I", "r_T", "r_R", "a")
POPULATION_ORDER = ("E", "I", "T", "R")


@dataclass(frozen=True)
class ParameterDraw:
    draw_id: str
    values: Mapping[str, float]
    payload_sha256: str


@dataclass
class Packet:
    packet_id: str
    start_scheduler_sample: int
    stop_scheduler_sample: int
    start_internal_step: int
    stop_internal_step: int
    amplitude: float
    carrier_hz: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "packet_start_sample": self.start_scheduler_sample,
            "packet_stop_sample": self.stop_scheduler_sample,
            "packet_start_internal_step": self.start_internal_step,
            "packet_stop_internal_step": self.stop_internal_step,
            "A_hz_equivalent": self.amplitude,
            "f_c_hz": self.carrier_hz,
        }


@dataclass
class SimulationResult:
    run: RunSpec
    parameter_draw: ParameterDraw
    rates: np.ndarray
    adaptation: np.ndarray
    observation: np.ndarray
    innovation_sha256: str
    state_sha256: str
    scheduler_log: list[dict[str, Any]] = field(default_factory=list)
    packet_table: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class PrefixResult:
    """State trace used only by the separately authorized solver preflight."""

    state_trace: np.ndarray
    innovation_sha256: str
    state_sha256: str


def _bounds(parameter_map: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    return {
        row["parameter"]: (float(row["bound"][0]), float(row["bound"][1]))
        for row in parameter_map["parameter_defaults_and_bounds"]
    }


def generate_parameter_draws(bundle: ContractBundle) -> dict[str, ParameterDraw]:
    rule = bundle.parameter_map["synthetic_uncertainty_rule"]
    defaults = {k: float(v) for k, v in bundle.protocol["bench"]["parameter_values"].items()}
    bounds = _bounds(bundle.parameter_map)
    names = tuple(rule["varied_parameters_in_coordinate_order"])
    if len(names) != 17 or tuple(rule["draw_ids"]) != DRAWS:
        raise ContractError("parameter draw contract mismatch")
    sampler = qmc.LatinHypercube(d=17, scramble=True, seed=271828)
    uniforms = sampler.random(n=5)
    z_values = norm.ppf(uniforms)
    draws: dict[str, ParameterDraw] = {}
    nominal = dict(defaults)
    draws[DRAWS[0]] = ParameterDraw(DRAWS[0], nominal, object_sha256(nominal))
    additive_biases = {"b_E", "b_T"}
    for draw_id, z_row in zip(DRAWS[1:], z_values):
        values = dict(defaults)
        for name, z in zip(names, z_row):
            lo, hi = bounds[name]
            if name in additive_biases:
                candidate = defaults[name] + 0.05 * (hi - lo) * float(z)
            else:
                candidate = defaults[name] * exp(0.05 * float(z))
            values[name] = float(np.clip(candidate, lo, hi))
        draws[draw_id] = ParameterDraw(draw_id, values, object_sha256(values))
    return draws


def generate_innovations(seed: int) -> tuple[np.ndarray, str]:
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    values = rng.normal(0.0, 1.0, size=(TOTAL_STEPS, 4)).astype(np.float64, copy=False)
    if not values.flags.c_contiguous:
        values = np.ascontiguousarray(values)
    digest = sha256(values.tobytes(order="C")).hexdigest().upper()
    return values, digest


def _round_half_up_nonnegative(value: float) -> int:
    if value < 0.0 or not np.isfinite(value):
        raise ContractError("round-half-up input must be finite and nonnegative")
    return int(floor(value + 0.5))


def packet_waveform(amplitude: float, carrier_hz: float) -> np.ndarray:
    k = np.arange(500, dtype=np.float64)
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * k / 499.0))
    carrier = 0.5 * (1.0 + np.sin(2.0 * np.pi * float(carrier_hz) * k / 1000.0))
    return float(amplitude) * window * carrier


class CausalPhaseScheduler:
    """Online 200 Hz phase scheduler with exact queue and logging semantics."""

    ALPHA_HP = 0.990663219125838
    ALPHA_LP = 0.04500316558786029

    def __init__(self, phi: float, amplitude: float, carrier_hz: float) -> None:
        phi = float(phi)
        if phi == pi or not (-pi <= phi < pi):
            raise ContractError("phase must be canonical [-pi,pi), with +pi prohibited")
        if amplitude not in {0.0, 0.25, 0.5, 0.75, 1.0}:
            raise ContractError("unregistered phase-packet amplitude")
        if carrier_hz not in {11.0, 12.0, 13.0, 14.0, 15.0}:
            raise ContractError("unregistered phase-packet carrier")
        self.phi = phi
        self.amplitude = float(amplitude)
        self.carrier_hz = float(carrier_hz)
        self.x_prev = 0.0
        self.hp = 0.0
        self.bp = 0.0
        self.initialized = False
        self.last_anchor: int | None = None
        self.periods: list[int] = []
        self.last_bp = 0.0
        self.packets: list[Packet] = []
        self.logs: list[dict[str, Any]] = []
        self._waveform = packet_waveform(self.amplitude, self.carrier_hz)

    def _base_log(self, n: int, interval: int | None) -> dict[str, Any]:
        return {
            "crossing_sample": n,
            "interval_samples": interval,
            "history_samples": list(self.periods),
            "P_use": None,
            "phi_c": self.phi,
            "q": None,
            "d_real": None,
            "d_round": None,
            "A_hz_equivalent": self.amplitude,
            "f_c_hz": self.carrier_hz,
            "scheduled_start": None,
            "scheduled_stop": None,
            "decision_code": None,
        }

    def update(self, n: int, x_value: float) -> None:
        x_value = float(x_value)
        if not self.initialized:
            self.x_prev = x_value
            self.last_bp = 0.0
            self.initialized = True
            return
        self.hp = self.ALPHA_HP * (self.hp + x_value - self.x_prev)
        self.bp = self.bp + self.ALPHA_LP * (self.hp - self.bp)
        self.x_prev = x_value
        crossing = self.last_bp < 0.0 and self.bp >= 0.0
        self.last_bp = self.bp
        if not crossing:
            return
        self._process_crossing(n)

    def _process_crossing(self, n: int) -> None:
        """Apply the frozen crossing decision tree (also used by queue preflight)."""
        if n < 4000:
            row = self._base_log(n, None)
            row["decision_code"] = "WARMUP_ONLY"
            self.logs.append(row)
            return
        if self.last_anchor is None:
            self.last_anchor = n
            self.periods.clear()
            row = self._base_log(n, None)
            row["decision_code"] = "FIRST_ANCHOR"
            self.logs.append(row)
            return
        interval = n - self.last_anchor
        if interval < 134 or interval > 400:
            self.last_anchor = n
            self.periods.clear()
            row = self._base_log(n, interval)
            row["decision_code"] = "PERIOD_OUT_OF_RANGE"
            self.logs.append(row)
            return
        self.periods.append(interval)
        self.periods[:] = self.periods[-3:]
        self.last_anchor = n
        row = self._base_log(n, interval)
        if n < 6000:
            row["decision_code"] = "WARMUP_ONLY"
            self.logs.append(row)
            return
        if len(self.periods) < 3:
            row["decision_code"] = "INSUFFICIENT_PERIOD_HISTORY"
            self.logs.append(row)
            return
        p_use = int(sorted(self.periods)[1])
        p_use = min(400, max(134, p_use))
        phi_c = ((self.phi + pi) % (2.0 * pi)) - pi
        if phi_c == pi:
            phi_c = -pi
        q_value = ((phi_c % (2.0 * pi)) + 2.0 * pi) % (2.0 * pi)
        d_real = q_value * p_use / (2.0 * pi)
        d_round = _round_half_up_nonnegative(d_real)
        delay = max(1, d_round)
        start = n + delay
        stop = start + 100
        row.update({
            "P_use": p_use,
            "phi_c": phi_c,
            "q": q_value,
            "d_real": d_real,
            "d_round": d_round,
            "scheduled_start": start,
            "scheduled_stop": stop,
        })
        if self.amplitude == 0.0:
            row["decision_code"] = "ZERO_AMPLITUDE_SHAM"
            self.logs.append(row)
            return
        start_internal = 5 * start
        stop_internal = start_internal + 500
        if start_internal < WARMUP_STEPS or stop_internal > TOTAL_STEPS:
            row["decision_code"] = "OUTSIDE_ANALYSIS_SKIP"
            self.logs.append(row)
            return
        active_or_queued = [p for p in self.packets if p.stop_scheduler_sample > n]
        overlap = any(max(start, p.start_scheduler_sample) < min(stop, p.stop_scheduler_sample) for p in active_or_queued)
        if overlap:
            row["decision_code"] = "OVERLAP_SKIP"
            self.logs.append(row)
            return
        future_count = sum(1 for p in active_or_queued if p.start_scheduler_sample > n)
        if future_count >= 2:
            row["decision_code"] = "QUEUE_FULL"
            self.logs.append(row)
            return
        packet = Packet(
            packet_id=f"PKT{len(self.packets):05d}",
            start_scheduler_sample=start,
            stop_scheduler_sample=stop,
            start_internal_step=start_internal,
            stop_internal_step=stop_internal,
            amplitude=self.amplitude,
            carrier_hz=self.carrier_hz,
        )
        self.packets.append(packet)
        self.packets.sort(key=lambda p: p.start_internal_step)
        row["decision_code"] = "SCHEDULED"
        self.logs.append(row)

    def value(self, internal_step: int) -> np.ndarray:
        result = np.zeros(4, dtype=np.float64)
        for packet in self.packets:
            if packet.start_internal_step <= internal_step < packet.stop_internal_step:
                result[2] = self._waveform[internal_step - packet.start_internal_step]
                break
        return result


class ForcingProgram:
    def __init__(self, run: RunSpec) -> None:
        self.run = run
        self.params = {k: float(v) for k, v in run.parameters.items()}
        self.channel = run.channel_id
        self.scheduler: CausalPhaseScheduler | None = None
        self.packet_starts: list[int] = []
        self.packet_values: np.ndarray | None = None
        self.t8_intervals: list[tuple[int, int]] = []
        if self.channel == "GF_T_PHASE_LOCKED_PACKET":
            self.scheduler = CausalPhaseScheduler(self.params["phi"], self.params["A"], self.params["f_c"])
        elif self.channel == "GF_TR_SIGMA_PACKET_BALANCE":
            self.packet_values = packet_waveform(abs(self.params["d"]), self.params["f_c"])
            p = 0
            while True:
                stored_start = _round_half_up_nonnegative(p * STORED_FS / self.params.get("f_rep", 0.85))
                relative_start = 5 * stored_start
                if relative_start + 500 > ANALYSIS_STEPS:
                    break
                self.packet_starts.append(WARMUP_STEPS + relative_start)
                p += 1
        elif self.channel == "GF_E_SO_PULSE":
            p = 0
            while True:
                relative_start = _round_half_up_nonnegative(p * INTERNAL_FS / self.params["f_rep"])
                if relative_start + 80 > ANALYSIS_STEPS:
                    break
                self.packet_starts.append(WARMUP_STEPS + relative_start)
                p += 1
        elif self.channel == "GF_BALANCED_DC" and self.run.stage == "LEVEL_B_SURFACE":
            fraction = self.params["fraction"]
            length = _round_half_up_nonnegative(25_000 * fraction)
            for a0 in (15_000, 40_000, 65_000, 90_000, 115_000, 140_000):
                start = a0 + floor((25_000 - length) / 2)
                self.t8_intervals.append((WARMUP_STEPS + start, WARMUP_STEPS + start + length))

    def observe_scheduler(self, internal_step: int, observation: float) -> None:
        if self.scheduler is not None:
            if internal_step % 5 != 0:
                raise ContractError("scheduler observation is not on stored-sample grid")
            self.scheduler.update(internal_step // 5, observation)

    def _inside_analysis(self, m: int) -> bool:
        return WARMUP_STEPS <= m < TOTAL_STEPS

    def value(self, m: int) -> np.ndarray:
        u = np.zeros(4, dtype=np.float64)
        if not self._inside_analysis(m) or self.channel == "SHAM":
            return u
        a = m - WARMUP_STEPS
        if self.channel == "GF_E_DC":
            u[0] = self.params["d"]
        elif self.channel == "GF_E_SLOW_SINE":
            u[0] = self.params["A"] * sin(2.0 * pi * self.params["f"] * a / 1000.0)
        elif self.channel == "GF_E_DC_PLUS_SLOW_SINE":
            u[0] = self.params["d"] + self.params["A"] * sin(2.0 * pi * self.params["f"] * a / 1000.0)
        elif self.channel == "GF_E_SO_PULSE":
            if any(start <= m < start + 80 for start in self.packet_starts):
                u[0] = self.params["d"]
        elif self.channel == "GF_TR_SIGMA_PACKET_BALANCE":
            for start in self.packet_starts:
                if start <= m < start + 500:
                    value = float(self.packet_values[m - start])
                    if self.params["d"] >= 0.0:
                        u[2] = value
                    else:
                        u[3] = value
                    break
        elif self.channel == "GF_TR_SIGMA_CONTINUOUS_BALANCE":
            q_value = 0.5 * (1.0 + sin(2.0 * pi * self.params["f_c"] * a / 1000.0))
            if self.params["d"] >= 0.0:
                u[2] = self.params["d"] * q_value
            else:
                u[3] = -self.params["d"] * q_value
        elif self.channel == "GF_T_PHASE_LOCKED_PACKET":
            if self.scheduler is None:
                raise ContractError("phase scheduler missing")
            u = self.scheduler.value(m)
        elif self.channel == "GF_BALANCED_DC":
            active = self.run.stage != "LEVEL_B_SURFACE" or any(start <= m < stop for start, stop in self.t8_intervals)
            if active:
                d = self.params["d"]
                if d >= 0.0:
                    u[0], u[2] = d, 0.5 * d
                else:
                    u[1], u[3] = -d, -0.5 * d
        else:
            raise ContractError(f"unsupported forcing channel {self.channel}")
        return u

    def packet_table(self) -> list[dict[str, Any]]:
        if self.scheduler is not None:
            return [packet.as_dict() for packet in self.scheduler.packets]
        if self.channel == "GF_TR_SIGMA_PACKET_BALANCE":
            result = []
            for idx, start in enumerate(self.packet_starts):
                result.append({
                    "packet_id": f"PKT{idx:05d}",
                    "packet_start_sample": (start - WARMUP_STEPS) // 5,
                    "packet_stop_sample": (start - WARMUP_STEPS) // 5 + 100,
                    "packet_start_internal_step": start,
                    "packet_stop_internal_step": start + 500,
                    "dose_hz_equivalent": self.params["d"],
                    "carrier_hz": self.params["f_c"],
                })
            return result
        return []


def _drift(state: np.ndarray, eta: np.ndarray, u: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    r_e, r_i, r_t, r_r, adaptation = state
    h_e = p["b_E"] + p["w_EE"] * r_e - p["w_EI"] * r_i + p["w_ET"] * r_t - p["g_a"] * adaptation + eta[0] + u[0]
    h_i = p["b_I"] + p["w_IE"] * r_e - p["w_II"] * r_i + p["w_IT"] * r_t + eta[1] + u[1]
    h_t = p["b_T"] + p["w_TE"] * r_e - p["w_TR"] * r_r + eta[2] + u[2]
    h_r = p["b_R"] + p["w_RE"] * r_e + p["w_RT"] * r_t - p["w_RR"] * r_r + eta[3] + u[3]
    h = np.array([h_e, h_i, h_t, h_r], dtype=np.float64)
    beta = p["beta"]
    theta = p["theta"]
    targets = np.array([p["rmax_E"], p["rmax_I"], p["rmax_T"], p["rmax_R"]]) * expit(beta * (h - theta))
    rates = state[:4]
    taus = np.array([p["tau_E"], p["tau_I"], p["tau_T"], p["tau_R"]], dtype=np.float64)
    rate_drift = (-rates + targets) / taus
    adaptation_drift = (-adaptation + r_e) / p["tau_a"]
    return np.concatenate((rate_drift, np.array([adaptation_drift], dtype=np.float64)))


def simulate(run: RunSpec, draw: ParameterDraw) -> SimulationResult:
    """Run exactly 210,000 fixed intervals; this function is never called by import."""
    if run.parameter_draw_id != draw.draw_id:
        raise ContractError("run/draw identity mismatch")
    innovations, innovation_hash = generate_innovations(run.seed)
    p = draw.values
    state = np.array([8.0, 10.0, 6.0, 8.0, 8.0], dtype=np.float64)
    eta = np.zeros(4, dtype=np.float64)
    rates = np.empty((4, STORED_SAMPLES), dtype=np.float64)
    adaptation = np.empty(STORED_SAMPLES, dtype=np.float64)
    observation = np.empty(STORED_SAMPLES, dtype=np.float64)
    forcing = ForcingProgram(run)
    rho = exp(-0.001 / p["tau_eta"])
    sigmas = np.array([p["sigma_E"], p["sigma_I"], p["sigma_T"], p["sigma_R"]], dtype=np.float64)
    ou_scale = sigmas * np.sqrt(1.0 - rho * rho)
    stored_index = 0
    for m in range(TOTAL_STEPS):
        x_now = (state[0] + 0.25 * state[2]) / 10.0
        if m % 5 == 0:
            forcing.observe_scheduler(m, x_now)
            if m >= WARMUP_STEPS:
                rates[:, stored_index] = state[:4]
                adaptation[stored_index] = state[4]
                observation[stored_index] = x_now
                stored_index += 1
        u = forcing.value(m)
        eta_next = rho * eta + ou_scale * innovations[m]
        drift_now = _drift(state, eta, u, p)
        predictor = state + 0.001 * drift_now
        drift_predictor = _drift(predictor, eta_next, u, p)
        state = state + 0.0005 * (drift_now + drift_predictor)
        eta = eta_next
    if stored_index != STORED_SAMPLES:
        raise ContractError(f"stored sample count {stored_index}, expected {STORED_SAMPLES}")
    digest = sha256()
    digest.update(rates.tobytes(order="C"))
    digest.update(adaptation.tobytes(order="C"))
    digest.update(observation.tobytes(order="C"))
    scheduler_log = [] if forcing.scheduler is None else list(forcing.scheduler.logs)
    return SimulationResult(
        run=run,
        parameter_draw=draw,
        rates=rates,
        adaptation=adaptation,
        observation=observation,
        innovation_sha256=innovation_hash,
        state_sha256=digest.hexdigest().upper(),
        scheduler_log=scheduler_log,
        packet_table=forcing.packet_table(),
    )


def simulate_nominal_prefix(draw: ParameterDraw, *, seed: int = 123457, steps: int = 10_000) -> PrefixResult:
    """Integrate an unforced prefix with the identical OU/Heun ordering.

    This is intentionally separate from :func:`simulate`: it is callable only
    by the data-free preflight mode and cannot create a scientific run record.
    """
    if draw.draw_id != "D00_NOMINAL":
        raise ContractError("solver-repeat preflight requires D00_NOMINAL")
    if steps != 10_000:
        raise ContractError("solver-repeat preflight prefix must be exactly 10 s")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    innovations = rng.normal(0.0, 1.0, size=(steps, 4)).astype(np.float64, copy=False)
    innovation_hash = sha256(innovations.tobytes(order="C")).hexdigest().upper()
    p = draw.values
    state = np.array([8.0, 10.0, 6.0, 8.0, 8.0], dtype=np.float64)
    eta = np.zeros(4, dtype=np.float64)
    trace = np.empty((steps + 1, 5), dtype=np.float64)
    trace[0] = state
    rho = exp(-0.001 / p["tau_eta"])
    sigmas = np.array([p["sigma_E"], p["sigma_I"], p["sigma_T"], p["sigma_R"]], dtype=np.float64)
    ou_scale = sigmas * np.sqrt(1.0 - rho * rho)
    zero = np.zeros(4, dtype=np.float64)
    for m in range(steps):
        eta_next = rho * eta + ou_scale * innovations[m]
        drift_now = _drift(state, eta, zero, p)
        predictor = state + 0.001 * drift_now
        drift_predictor = _drift(predictor, eta_next, zero, p)
        state = state + 0.0005 * (drift_now + drift_predictor)
        eta = eta_next
        trace[m + 1] = state
    state_hash = sha256(trace.tobytes(order="C")).hexdigest().upper()
    return PrefixResult(trace, innovation_hash, state_hash)
