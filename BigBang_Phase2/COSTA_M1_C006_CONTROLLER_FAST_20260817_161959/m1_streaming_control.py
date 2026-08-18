from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt, welch


DT = 0.002
FS = 1.0 / DT
STATE0 = np.array([0.18, 0.16, 0.14, 0.12], dtype=float)
TAUS = np.array([0.018, 0.012, 0.028, 0.016], dtype=float)
WEIGHTS = np.array([1.0, -0.55, 0.35, -0.2], dtype=float)
C006 = dict(cortical_gain=0.82, cortex_to_thalamus=0.95,
            thalamus_to_cortex=1.10, relay_reticular_inhibition=1.25)


def make_drive(seed: int, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Frozen synthetic drive form with seed-fixed prospective phase offsets."""
    n = int(round(duration_s / DT))
    t = np.arange(n) * DT
    rng = np.random.default_rng(seed)
    pe, pt, pc = rng.uniform(0, 2 * np.pi, 3)
    e = 0.34 + 0.18 * np.sin(2 * np.pi * 0.73 * t + pe)
    ep = np.mod(t + pe / (2 * np.pi * 0.73), 0.91) < 0.11
    e = e + 0.32 * ep
    tg = np.mod(t + pt / (2 * np.pi * 1.17), 1.37) < 0.42
    th = 0.23 + 0.13 * np.sin(2 * np.pi * 1.17 * t + pt)
    th = th + 0.20 * np.sin(2 * np.pi * 11.0 * t + pc) * tg
    return e.astype(float), th.astype(float)


class CausalPLL:
    """Small normalized second-order digital PLL tracking a sinusoidal proxy."""
    def __init__(self, nominal_hz: float = 0.73):
        self.phase = 0.0
        self.omega = 2 * np.pi * nominal_hz
        self.amp = 1e-3
        self.kp = 0.035
        self.ki = 0.00008

    def update(self, x: float) -> float:
        self.amp = 0.999 * self.amp + 0.001 * abs(x)
        xn = float(np.clip(x / max(2.0 * self.amp, 1e-6), -2.0, 2.0))
        error = xn * math.cos(self.phase)
        self.omega = float(np.clip(self.omega + self.ki * error, 2*np.pi*0.5, 2*np.pi*1.25))
        self.phase = (self.phase + self.omega * DT + self.kp * error + np.pi) % (2*np.pi) - np.pi
        return self.phase


@dataclass
class RunResult:
    seed: int
    condition: str
    states: np.ndarray
    proxy: np.ndarray
    observed_proxy: np.ndarray
    command: np.ndarray
    pulse_starts: list[int]
    pulse_amplitudes: list[float]
    baseline_metrics: dict
    control_metrics: dict
    stable: bool
    saturation_fraction: float


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(3.0 * x, -60.0, 60.0)))


def simulate_open(drive_e: np.ndarray, drive_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-control streaming core used for bit-level adapter equivalence."""
    n = len(drive_e); states = np.empty((n, 4)); states[0] = STATE0
    source = np.empty(n); source[0] = states[0] @ WEIGHTS
    base = source[0]; proxy = np.zeros(n)
    for i in range(1, n):
        e, inh, th, ret = states[i-1]; ue = drive_e[i-1]; ut = drive_t[i-1]
        inputs = np.array([.82*(2.8*e-2.2*inh)+1.10*1.8*th+ue-.8,
                           .82*(2.2*e-.6*inh)+1.10*.7*th+.7*ue-.65,
                           .95*1.6*e-1.25*ret+ut-.4,
                           .95*e+1.8*th-.3*ret+.5*ut-.55])
        states[i] = np.clip(states[i-1] + DT/TAUS*(sigmoid(inputs)-states[i-1]), 0, 1)
        source[i] = states[i] @ WEIGHTS
        base += (DT/.160)*(source[i]-base)
        proxy[i] = 100*(source[i]-base)
    return states, proxy


def metrics(proxy: np.ndarray, cortical_e: np.ndarray) -> dict:
    x = np.asarray(proxy, float)
    centered = x - np.mean(x)
    nperseg = min(len(x), int(8 * FS))
    f, p = welch(centered, fs=FS, window="hann", nperseg=nperseg,
                 noverlap=min(int(4*FS), nperseg-1))
    so_mask = (f >= 0.5) & (f <= 1.25)
    sigma_mask = (f >= 10.0) & (f <= 15.0)
    so = sosfiltfilt(butter(4, [0.5, 1.25], btype="bandpass", fs=FS, output="sos"), centered)
    sigma = sosfiltfilt(butter(4, [10.0, 15.0], btype="bandpass", fs=FS, output="sos"), centered)
    phase = np.angle(hilbert(so)); env = np.abs(hilbert(sigma))
    vec = np.sum(env * np.exp(1j*phase)) / max(float(np.sum(env)), 1e-12)
    duration_min = len(x) / FS / 60.0
    troughs, _ = find_peaks(-so, distance=int(0.5*FS), prominence=max(.25*np.std(so), 1e-12))
    rms = np.sqrt(uniform_filter1d(sigma*sigma, size=int(.2*FS), mode="nearest"))
    threshold = np.mean(rms) + 1.5*np.std(rms)
    changes = np.diff(np.r_[False, rms > threshold, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    spindles = sum(.5 <= (b-a)/FS <= 3.0 for a, b in zip(starts, ends))
    smooth = uniform_filter1d(np.asarray(cortical_e, float), size=max(1, int(.25*FS)), mode="nearest")
    lo, hi = np.quantile(smooth, [.35, .65]); state = 0; crossings = 0
    for val in smooth:
        nxt = 1 if val > hi else (-1 if val < lo else state)
        if state and nxt != state: crossings += 1
        state = nxt
    return {
        "T1": float(f[so_mask][np.argmax(p[so_mask])]),
        "T2": float(np.sqrt(np.mean(so*so))),
        "T3": float(len(troughs)/duration_min),
        "T4": float(spindles/duration_min),
        "T5": float(np.trapezoid(p[sigma_mask], f[sigma_mask])),
        "T6": float(abs(vec)),
        "T7": float(np.angle(vec)),
        "T8_MVP_cortical_state_crossing_rate": float(crossings/duration_min),
    }


def score(active: dict, sham: dict) -> tuple[float, dict]:
    eps = {"T2": 1e-9, "T5": 1e-12, "T6": 1e-4}
    d = {k: (active[k]-sham[k])/max(abs(sham[k]), eps[k]) for k in eps}
    return float(.50*d["T2"] + .25*d["T5"] + .25*d["T6"]), d


def _feedback(window: np.ndarray, e_window: np.ndarray) -> tuple[float, float]:
    m = metrics(window, e_window)
    return m["T2"], m["T5"]


def simulate(seed: int, condition: str, *, phase_target: float = 0.0,
             pid: dict | None = None, generic_target: float | None = None,
             replay: tuple[list[int], list[float]] | None = None,
             deadline: Callable[[], None] | None = None) -> RunResult:
    burn_s, baseline_s, control_s = 5.0, 10.0, 60.0
    total_s = burn_s + baseline_s + control_s
    de, dt = make_drive(seed, total_s); n = len(de)
    states = np.empty((n, 4)); states[0] = STATE0
    source = np.empty(n); source[0] = states[0] @ WEIGHTS
    baseline = source[0]; proxy = np.zeros(n); obs = np.zeros(n); command = np.zeros(n)
    rng = np.random.default_rng(seed + 26081603)
    noise = rng.normal(0.0, .0015, n)
    pll = CausalPLL(); previous_phase = pll.phase
    start_control = int((burn_s + baseline_s)/DT); baseline_start = int(burn_s/DT)
    width = int(round(.11/DT)); refractory = int(round(.8/DT)); last_pulse = -refractory
    pulse_starts: list[int] = []; pulse_amplitudes: list[float] = []
    current_amp = .035; active_pulse_amp = .035; pulse_remaining = 0
    integral = derivative_smooth = previous_error = 0.0
    pid_command = 0.0
    replay_map = {}
    if replay:
        shift = int(round((.5/.73)/DT))
        control_len = n - start_control
        shifted = [start_control + ((i - start_control + shift) % control_len) for i in replay[0]]
        shifted = [i if i + width <= n else i - control_len for i in shifted]
        replay_map = {i: a for i, a in zip(shifted, replay[1])}
        if len(replay_map) != len(replay[0]):
            raise RuntimeError("PHASE_CONTROL_MATCH_FAILED: shifted pulse collision")
    for i in range(1, n):
        if deadline and i % 5000 == 0: deadline()
        phase = pll.update(obs[i-1])
        if i >= start_control:
            if condition == "FIXED": pid_command = .035
            elif condition in ("GENERIC_PID", "BASELINE_ADAPTIVE_PID") and i % int(2/DT) == 0:
                w0 = max(baseline_start, i-int(10/DT)); t2, _ = _feedback(obs[w0:i], states[w0:i,0])
                target = float(generic_target) if condition == "GENERIC_PID" else 1.2 * metrics(proxy[baseline_start:start_control], states[baseline_start:start_control,0])["T2"]
                err = (target-t2)/max(abs(target), 1e-9); integral = np.clip(integral+err*2, -4, 4)
                deriv = (err-previous_error)/2; derivative_smooth = .8*derivative_smooth+.2*deriv
                pid_command = float(np.clip(pid["Kp"]*err+pid["Ki"]*integral+pid["Kd"]*derivative_smooth, 0, .05))
                previous_error = err
            elif condition == "PHASE_SHIFTED":
                if i in replay_map:
                    current_amp = replay_map[i]; active_pulse_amp = current_amp; pulse_remaining = width
                    pulse_starts.append(i); pulse_amplitudes.append(active_pulse_amp)
            elif condition.startswith("PHASE_LOCKED"):
                crossed = ((previous_phase-phase_target+2*np.pi)%(2*np.pi)) > ((phase-phase_target+2*np.pi)%(2*np.pi))
                if crossed and i-last_pulse >= refractory:
                    active_pulse_amp = current_amp; pulse_remaining = width; last_pulse = i
                    pulse_starts.append(i); pulse_amplitudes.append(active_pulse_amp)
                if condition == "PHASE_LOCKED_ADAPTIVE" and i % int(10/DT) == 0:
                    w0 = max(start_control, i-int(10/DT)); t2, t5 = _feedback(obs[w0:i], states[w0:i,0])
                    bm = metrics(proxy[baseline_start:start_control], states[baseline_start:start_control,0])
                    e2 = (1.2*bm["T2"]-t2)/max(1.2*bm["T2"],1e-9)
                    e5 = (1.15*bm["T5"]-t5)/max(1.15*bm["T5"],1e-12)
                    err=.7*e2+.3*e5; integral=np.clip(integral+err*10,-4,4)
                    current_amp=float(np.clip(current_amp+.01*err+.002*integral,0,.05))
            if condition.startswith("PHASE_LOCKED") or condition == "PHASE_SHIFTED":
                pid_command = active_pulse_amp if pulse_remaining > 0 else 0.0
                pulse_remaining = max(0, pulse_remaining-1)
        command[i-1] = np.clip(pid_command, 0, .05)
        e, inh, th, ret = states[i-1]
        ue = de[i-1] + command[i-1]; ut = dt[i-1]
        inputs = np.array([
            .82*(2.8*e-2.2*inh)+1.10*1.8*th+ue-.8,
            .82*(2.2*e-.6*inh)+1.10*.7*th+.7*ue-.65,
            .95*1.6*e-1.25*ret+ut-.4,
            .95*e+1.8*th-.3*ret+.5*ut-.55])
        states[i] = np.clip(states[i-1] + DT/TAUS*(sigmoid(inputs)-states[i-1]), 0, 1)
        source[i] = states[i] @ WEIGHTS
        baseline += (DT/.160)*(source[i]-baseline)
        proxy[i] = 100*(source[i]-baseline); obs[i] = proxy[i] + noise[i]
        previous_phase = phase
    command[-1] = command[-2]
    b = metrics(proxy[baseline_start:start_control], states[baseline_start:start_control,0])
    c = metrics(proxy[start_control:], states[start_control:,0])
    stable = bool(np.isfinite(states).all() and np.isfinite(proxy).all() and states.min() >= 0 and states.max() <= 1)
    if condition.startswith("PHASE_LOCKED") or condition == "PHASE_SHIFTED":
        sat = float(np.mean([(a <= 1e-12 or a >= .05-1e-12) for a in pulse_amplitudes])) if pulse_amplitudes else 1.0
    else:
        sat = float(np.mean((command[start_control:] <= 1e-12) | (command[start_control:] >= .05-1e-12)))
    return RunResult(seed, condition, states, proxy, obs, command, pulse_starts,
                     pulse_amplitudes, b, c, stable, sat)
