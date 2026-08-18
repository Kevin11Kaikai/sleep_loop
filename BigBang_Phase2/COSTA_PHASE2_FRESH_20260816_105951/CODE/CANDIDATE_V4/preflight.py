"""Separately authorized, data-free implementation preflight.

Nothing in this module runs on import.  The sole public runner constructs only
analytic fixtures plus the frozen raw T4 fixture; it never opens scientific
data and never creates a Phase2A result.
"""

from __future__ import annotations

from math import pi
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.signal import decimate, hilbert

from .contracts import (
    ContractBundle,
    ContractError,
    FREEZE_HASH,
    PROTOCOL_AUDIT_HASH,
    atomic_write_json,
)
from .detectors import (
    RETAINED_START,
    RETAINED_STOP,
    SOEvent,
    SpindleEvent,
    SpindleThresholds,
    T8_WINDOWS,
    _band_power,
    _bandpass,
    _contained,
    _preferred_phase,
    _t8_windows,
    _welch_peak,
    broadband_specificity_veto,
    calibrate_spindle_thresholds,
    corrected_mi,
    detect_so_events,
    detect_spindle_events,
    preprocess,
)
from .model import (
    CausalPhaseScheduler,
    Packet,
    STORED_FS,
    STORED_SAMPLES,
    TOTAL_STEPS,
    generate_parameter_draws,
    simulate_nominal_prefix,
)
from .run_registry import PHASES_A
from .t4_attribution import compare_fixture, validate_domain


def _pass(test_id: str, observed: Mapping[str, Any]) -> dict[str, Any]:
    return {"test_id": test_id, "status": "PASS", "observed": dict(observed)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _numeric_zero(_bundle: ContractBundle) -> dict[str, Any]:
    zero = np.zeros(STORED_SAMPLES, dtype=np.float64)
    _centered, so, sigma, phase, envelope, rms = preprocess(zero)
    thresholds = calibrate_spindle_thresholds(rms)
    so_events = detect_so_events(so)
    spindles = detect_spindle_events(sigma, rms, thresholds)
    mi, _metadata = corrected_mi(phase, envelope, 123457)
    preferred, _r, _values = _preferred_phase(phase, envelope, spindles)
    t2 = None if len(so_events) < 30 else float(np.median([event.peak_to_peak for event in so_events]))
    t3 = 60.0 * len(so_events) / 160.0
    t4 = 60.0 * len(spindles) / 160.0
    t5 = None if len(spindles) < 10 else float(np.median([event.event_rms for event in spindles]))
    _require(t3 == 0.0 and t4 == 0.0, "zero fixture produced an event")
    _require(t2 is None and t5 is None and mi is None and preferred is None, "zero fixture missingness mismatch")
    return _pass("PF_NUMERIC_ZERO", {"T2": t2, "T3": t3, "T4": t4, "T5": t5, "T6": mi, "T7": preferred})


def _frequency(_bundle: ContractBundle) -> dict[str, Any]:
    time = np.arange(STORED_SAMPLES, dtype=np.float64) / STORED_FS
    signal = np.sin(2.0 * pi * 0.80 * time)
    peak, metadata = _welch_peak(signal)
    slow = _bandpass(signal, 0.3, 1.5)
    phase = np.angle(hilbert(slow))
    retained = slice(RETAINED_START, RETAINED_STOP)
    plv = float(abs(np.mean(np.exp(1j * (phase[retained] - 2.0 * pi * 0.80 * time[retained])))))
    _require(peak is not None and abs(peak - 0.80) <= 0.02 and plv >= 0.99, "T1 analytic frequency fixture failed")

    # Frozen stride storage must agree spectrally with the required FIR check.
    fine_time = np.arange(180_000, dtype=np.float64) / 1000.0
    fine = np.sin(2.0 * pi * 0.80 * fine_time)
    stride = fine[::5]
    fir = decimate(fine, 5, ftype="fir", zero_phase=True)
    stride_power = _band_power(stride, 0.4, 1.5)
    fir_power = _band_power(fir, 0.4, 1.5)
    spectral_fraction = abs(stride_power - fir_power) / max(abs(fir_power), 1e-12)
    _require(spectral_fraction <= 0.01, "stored-stride/FIR preflight spectrum differs by more than one percent")
    return _pass("PF_T1_FREQUENCY", {"peak_hz": peak, "plv": plv, "peak_metadata": metadata, "fir_spectral_fraction": spectral_fraction})


def _so_events(_bundle: ContractBundle) -> dict[str, Any]:
    time = np.arange(STORED_SAMPLES, dtype=np.float64) / STORED_FS
    signal = 0.20 * np.sin(2.0 * pi * 0.80 * time)
    _centered, so, _sigma, _phase, _envelope, _rms = preprocess(signal)
    events = detect_so_events(so)
    density = 60.0 * len(events) / 160.0
    amplitude = None if len(events) < 30 else float(np.median([event.peak_to_peak for event in events]))
    _require(abs(density - 48.0) <= 1.0, "SO fixture density outside tolerance")
    _require(amplitude is not None and abs(amplitude - 0.40) <= 0.01, "SO fixture amplitude outside tolerance")
    return _pass("PF_SO_EVENTS", {"event_count": len(events), "density_per_min": density, "median_peak_to_peak": amplitude})


def _spindle_events(_bundle: ContractBundle) -> dict[str, Any]:
    sigma = np.zeros(STORED_SAMPLES, dtype=np.float64)
    rms = np.zeros(STORED_SAMPLES, dtype=np.float64)
    starts = [3_000 + 2_300 * idx for idx in range(13)]
    k = np.arange(200, dtype=np.float64)
    burst = np.sin(2.0 * pi * 13.0 * k / STORED_FS) * np.hanning(200)
    for start in starts:
        sigma[start:start + 200] = burst
        rms[start:start + 200] = 0.10
    thresholds = SpindleThresholds(0.0, 0.0, 0.08, 0.04)
    events = detect_spindle_events(sigma, rms, thresholds)
    durations = [(event.offset_sample - event.onset_sample) / STORED_FS for event in events]
    median_duration = None if not durations else float(np.median(durations))
    _require(len(events) == 13, "spindle fixture did not retain exactly thirteen events")
    _require(median_duration is not None and abs(median_duration - 1.0) <= 0.05, "spindle fixture duration mismatch")
    return _pass("PF_SPINDLE_EVENTS", {"event_count": len(events), "median_duration_seconds": median_duration})


def _nonstationary_phase() -> np.ndarray:
    time = np.arange(STORED_SAMPLES, dtype=np.float64) / STORED_FS
    instantaneous = 0.80 + 0.06 * np.sin(2.0 * pi * 0.037 * time)
    return np.mod(2.0 * pi * np.cumsum(instantaneous) / STORED_FS + pi, 2.0 * pi) - pi


def _events_at_phase_zero(phase: np.ndarray) -> list[SpindleEvent]:
    crossings = np.flatnonzero((phase[:-1] < 0.0) & (phase[1:] >= 0.0)) + 1
    result = []
    for center in crossings:
        if RETAINED_START + 60 < center < RETAINED_STOP - 60:
            result.append(SpindleEvent(f"SP{len(result):05d}", int(center - 40), int(center + 40), int(center), 1.0, 1.0))
    return result


def _coupling_positive(_bundle: ContractBundle) -> dict[str, Any]:
    phase = _nonstationary_phase()
    amplitude = np.exp(3.0 * np.cos(phase)).astype(np.float64)
    mi, metadata = corrected_mi(phase, amplitude, 123457)
    preferred, resultant, _values = _preferred_phase(phase, amplitude, _events_at_phase_zero(phase))
    error = None if preferred is None else abs(float(np.arctan2(np.sin(preferred), np.cos(preferred))))
    _require(mi is not None and mi > 0.01, "positive coupling MI did not exceed 0.01")
    _require(error is not None and error <= pi / 12.0, "positive coupling phase error exceeded pi/12")
    return _pass("PF_COUPLING_POSITIVE", {"bias_corrected_mi": mi, "preferred_phase": preferred, "resultant_length": resultant, "absolute_phase_error": error, "metadata": metadata})


def _coupling_null(_bundle: ContractBundle) -> dict[str, Any]:
    phase = _nonstationary_phase()
    rng = np.random.Generator(np.random.PCG64(900001))
    cycle = np.floor(np.unwrap(phase) / (2.0 * pi)).astype(np.int64)
    unique = np.unique(cycle)
    offsets = {int(value): float(rng.uniform(-pi, pi)) for value in unique}
    shifted = np.asarray([phase[idx] - offsets[int(cycle[idx])] for idx in range(len(phase))], dtype=np.float64)
    amplitude = np.exp(3.0 * np.cos(shifted))
    mi, metadata = corrected_mi(phase, amplitude, 123457)
    _require(mi is not None and abs(mi) <= 0.01, "null coupling absolute corrected MI exceeded 0.01")
    return _pass("PF_COUPLING_NULL", {"bias_corrected_mi": mi, "metadata": metadata, "rng_seed": 900001})


def _edge_containment(_bundle: ContractBundle) -> dict[str, Any]:
    so = [
        SOEvent("SO_LEFT", RETAINED_START - 20, RETAINED_START + 20, RETAINED_START - 5, RETAINED_START + 5, -0.1, 0.1, 0.2),
        SOEvent("SO_IN", 10_000, 10_200, 10_050, 10_150, -0.1, 0.1, 0.2),
        SOEvent("SO_RIGHT", RETAINED_STOP - 20, RETAINED_STOP + 20, RETAINED_STOP - 5, RETAINED_STOP + 5, -0.1, 0.1, 0.2),
    ]
    spindle = [
        SpindleEvent("SP_LEFT", RETAINED_START, RETAINED_START + 100, RETAINED_START + 50, 1.0, 1.0),
        SpindleEvent("SP_IN", 12_000, 12_200, 12_100, 1.0, 1.0),
        SpindleEvent("SP_RIGHT", RETAINED_STOP - 100, RETAINED_STOP, RETAINED_STOP - 50, 1.0, 1.0),
    ]
    kept_so = _contained(so, RETAINED_START, RETAINED_STOP)
    kept_spindle = _contained(spindle, RETAINED_START, RETAINED_STOP)
    _require([event.event_id for event in kept_so] == ["SO_IN"], "SO strict-edge containment failed")
    _require([event.event_id for event in kept_spindle] == ["SP_IN"], "spindle strict-edge containment failed")
    return _pass("PF_EDGE_CONTAINMENT", {"retained_so_ids": [e.event_id for e in kept_so], "retained_spindle_ids": [e.event_id for e in kept_spindle]})


def _broadband_veto(_bundle: ContractBundle) -> dict[str, Any]:
    sham = {"power_4_30": 1.0, "sigma_fraction_4_30": 0.20}
    active = {"power_4_30": 2.0, "sigma_fraction_4_30": 0.20}
    veto = broadband_specificity_veto(active, sham)
    _require(veto, "broadband specificity fixture did not trigger veto")
    return _pass("PF_BROADBAND_VETO", {"veto": veto, "broadband_ratio": 2.0, "sigma_fraction_change": 0.0})


def _solver_repeat(bundle: ContractBundle) -> dict[str, Any]:
    nominal = generate_parameter_draws(bundle)["D00_NOMINAL"]
    first = simulate_nominal_prefix(nominal, seed=123457, steps=10_000)
    second = simulate_nominal_prefix(nominal, seed=123457, steps=10_000)
    identical = bool(first.innovation_sha256 == second.innovation_sha256 and first.state_sha256 == second.state_sha256 and first.state_trace.tobytes(order="C") == second.state_trace.tobytes(order="C"))
    finite = bool(np.all(np.isfinite(first.state_trace)))
    rmax = np.asarray([nominal.values[f"rmax_{name}"] for name in ("E", "I", "T", "R")], dtype=np.float64)
    bounds = bool(np.all(first.state_trace[:, :4] >= -1e-6) and np.all(first.state_trace[:, :4] <= rmax[None, :] + 1e-6))
    _require(identical and finite and bounds, "solver-repeat byte identity, finiteness, or bounds failed")
    return _pass("PF_SOLVER_REPEAT", {"seed": 123457, "steps": 10_000, "byte_identical": identical, "finite": finite, "unclipped_bounds": bounds, "innovation_sha256": first.innovation_sha256, "state_sha256": first.state_sha256})


def _causal_scheduler(_bundle: ContractBundle) -> dict[str, Any]:
    expected = {0.0: 1, pi / 2.0: 63, -pi / 2.0: 188, -pi: 125}
    observed: dict[str, Any] = {}
    sample = np.arange(8_400, dtype=np.float64)
    signal = np.sin(2.0 * pi * 0.80 * sample / STORED_FS)
    for phase in PHASES_A:
        scheduler = CausalPhaseScheduler(float(phase), 1.0, 13.0)
        for n, value in enumerate(signal):
            scheduler.update(n, float(value))
        scheduled = [row for row in scheduler.logs if row["decision_code"] == "SCHEDULED"]
        periods = [row["P_use"] for row in scheduled]
        _require(scheduled and all(value == 250 for value in periods), f"scheduler period history failed for phase {phase}")
        if float(phase) in expected:
            delays = [row["scheduled_start"] - row["crossing_sample"] for row in scheduled]
            _require(all(value == expected[float(phase)] for value in delays), f"scheduler delay failed for phase {phase}")
        observed[str(phase)] = {"crossing_logs": len(scheduler.logs), "scheduled": len(scheduled), "P_use_values": sorted(set(periods))}
    rejected = False
    try:
        CausalPhaseScheduler(pi, 1.0, 13.0)
    except ContractError:
        rejected = True
    _require(rejected, "+pi scheduler input was not rejected")
    observed["plus_pi_rejected"] = rejected
    return _pass("PF_CAUSAL_PHASE_SCHEDULER", observed)


def _seed_scheduler(n: int, phase: float = 0.0) -> CausalPhaseScheduler:
    scheduler = CausalPhaseScheduler(phase, 1.0, 13.0)
    scheduler.last_anchor = n - 134
    scheduler.periods[:] = [134, 134, 134]
    return scheduler


def _causal_queue(_bundle: ContractBundle) -> dict[str, Any]:
    # Overlap: an active interval covers the proposed phi=0 start.
    overlap = _seed_scheduler(7_000)
    overlap.packets.append(Packet("EXISTING", 6_950, 7_050, 34_750, 35_250, 1.0, 13.0))
    overlap._process_crossing(7_000)
    _require(overlap.logs[-1]["decision_code"] == "OVERLAP_SKIP", "overlap proposal was not skipped")

    # Half-open exact touching: existing stop equals proposed start n+1.
    touching = _seed_scheduler(7_000)
    touching.packets.append(Packet("EXISTING", 6_901, 7_001, 34_505, 35_005, 1.0, 13.0))
    touching._process_crossing(7_000)
    _require(touching.logs[-1]["decision_code"] == "SCHEDULED", "half-open exact touching was not accepted")

    # Two nonoverlapping future packets fill the queue before a third proposal.
    full = _seed_scheduler(7_000)
    full.packets.extend([
        Packet("FUTURE1", 7_200, 7_300, 36_000, 36_500, 1.0, 13.0),
        Packet("FUTURE2", 7_400, 7_500, 37_000, 37_500, 1.0, 13.0),
    ])
    full._process_crossing(7_000)
    _require(full.logs[-1]["decision_code"] == "QUEUE_FULL", "third future packet did not receive QUEUE_FULL")

    outside = _seed_scheduler(41_950)
    outside._process_crossing(41_950)
    _require(outside.logs[-1]["decision_code"] == "OUTSIDE_ANALYSIS_SKIP", "boundary-crossing packet was not skipped")
    _require(all(packet.stop_internal_step - packet.start_internal_step == 500 for packet in touching.packets), "packet was truncated")
    return _pass("PF_CAUSAL_QUEUE", {"overlap": overlap.logs[-1]["decision_code"], "touching": touching.logs[-1]["decision_code"], "queue": full.logs[-1]["decision_code"], "boundary": outside.logs[-1]["decision_code"], "no_sum_or_truncation": True})


def _t8_windows_fixture(_bundle: ContractBundle) -> dict[str, Any]:
    time = np.arange(STORED_SAMPLES, dtype=np.float64) / STORED_FS
    centered = 0.20 * np.sin(2.0 * pi * 0.80 * time)
    rates = np.full((4, STORED_SAMPLES), 5.0, dtype=np.float64)
    so_events: list[SOEvent] = []
    spindle_events: list[SpindleEvent] = []
    for idx, (_window, start, stop) in enumerate(T8_WINDOWS):
        so_events.extend([
            SOEvent(f"SO_B{idx}", start, start + 200, start + 50, start + 150, -0.1, 0.1, 0.2),
            SOEvent(f"SO_I{idx}", start + 500, start + 700, start + 550, start + 650, -0.1, 0.1, 0.2),
        ])
        spindle_events.extend([
            SpindleEvent(f"SP_B{idx}", stop - 100, stop, stop - 50, 1.0, 1.0),
            SpindleEvent(f"SP_I{idx}", start + 900, start + 1_100, start + 1_000, 1.0, 1.0),
        ])
    rows, _occupancy, _persistent = _t8_windows(rates, centered, so_events, spindle_events)
    pairs = [row["sample_indices_half_open"] for row in rows]
    expected = [[start, stop] for _window, start, stop in T8_WINDOWS]
    peaks_ok = all(row["dominant_so_peak_hz"] is not None and abs(row["dominant_so_peak_hz"] - 0.80) <= 0.02 for row in rows)
    boundary_rejected = all(row["so_event_count"] == 1 and row["spindle_event_count"] == 1 for row in rows)
    welch_ok = all(row["peak_metadata"]["welch_parameters"]["nperseg"] == 5_000 for row in rows)
    _require(pairs == expected and peaks_ok and boundary_rejected and welch_ok, "T8 window fixture failed")
    return _pass("PF_T8_WINDOWS", {"sample_pairs": pairs, "boundary_touching_rejected": boundary_rejected, "peaks_within_0p02": peaks_ok, "nperseg": 5_000})


def _t4_fixture(bundle: ContractBundle) -> dict[str, Any]:
    validate_domain(bundle)
    result = compare_fixture(bundle)
    _require(result["status"] == "PASS" and result["raw_row_consumption"]["complete"], "PF_T4_EVENT_ATTRIBUTION_FAIL")
    return result


def _detector_perturb(_bundle: ContractBundle) -> dict[str, Any]:
    time = np.arange(STORED_SAMPLES, dtype=np.float64) / STORED_FS
    signal = 0.20 * np.sin(2.0 * pi * 0.80 * time)
    for start in [3_000 + 2_300 * idx for idx in range(13)]:
        k = np.arange(200, dtype=np.float64)
        signal[start:start + 200] += 0.20 * np.hanning(200) * np.sin(2.0 * pi * 13.0 * k / STORED_FS)
    results: dict[str, dict[str, float]] = {}
    for fs in (199.999, 200.0, 200.001):
        for order in (3, 4, 5):
            so = _bandpass(signal - np.mean(signal), 0.3, 1.5, fs=fs, order=order)
            sigma = _bandpass(signal - np.mean(signal), 11.0, 16.0, fs=fs, order=order)
            squared = sigma * sigma
            prefix = np.concatenate((np.array([0.0]), np.cumsum(squared, dtype=np.float64)))
            rms = np.full(STORED_SAMPLES, np.nan, dtype=np.float64)
            centers = np.arange(19, STORED_SAMPLES - 20, dtype=np.int64)
            rms[centers] = np.sqrt((prefix[centers + 21] - prefix[centers - 19]) / 40.0)
            thresholds = calibrate_spindle_thresholds(rms)
            so_events = detect_so_events(so)
            spindle_events = detect_spindle_events(sigma, rms, thresholds)
            amplitude = float(np.median([event.peak_to_peak for event in so_events]))
            results[f"fs={fs:.3f}/order={order}"] = {"so_count": float(len(so_events)), "spindle_count": float(len(spindle_events)), "so_amplitude": amplitude}
    official = results["fs=200.000/order=4"]
    for values in results.values():
        _require(abs(values["so_count"] - official["so_count"]) <= 1 and abs(values["spindle_count"] - official["spindle_count"]) <= 1, "detector perturbation event count changed by more than one")
        _require(abs(values["so_amplitude"] - official["so_amplitude"]) / max(abs(official["so_amplitude"]), 1e-12) <= 0.05, "detector perturbation continuous metric changed by more than five percent")
    return _pass("PF_DETECTOR_PERTURB", {"registered": official, "sensitivity": results, "official_filter_order": 4})


TESTS: tuple[tuple[str, Callable[[ContractBundle], dict[str, Any]]], ...] = (
    ("PF_NUMERIC_ZERO", _numeric_zero),
    ("PF_T1_FREQUENCY", _frequency),
    ("PF_SO_EVENTS", _so_events),
    ("PF_SPINDLE_EVENTS", _spindle_events),
    ("PF_COUPLING_POSITIVE", _coupling_positive),
    ("PF_COUPLING_NULL", _coupling_null),
    ("PF_EDGE_CONTAINMENT", _edge_containment),
    ("PF_BROADBAND_VETO", _broadband_veto),
    ("PF_SOLVER_REPEAT", _solver_repeat),
    ("PF_CAUSAL_PHASE_SCHEDULER", _causal_scheduler),
    ("PF_CAUSAL_QUEUE", _causal_queue),
    ("PF_T8_WINDOWS", _t8_windows_fixture),
    ("PF_T4_EVENT_ATTRIBUTION", _t4_fixture),
    ("PF_DETECTOR_PERTURB", _detector_perturb),
)


def run_data_free_preflight(bundle: ContractBundle, output_root: Path) -> dict[str, Any]:
    """Run all and only the fourteen frozen analytic/fixture checks."""
    frozen_ids = tuple(row["test_id"] for row in bundle.protocol["detector_preflight"]["tests"])
    implemented_ids = tuple(test_id for test_id, _function in TESTS)
    if frozen_ids != implemented_ids:
        raise ContractError(f"preflight test order/set mismatch: frozen={frozen_ids}, implemented={implemented_ids}")
    results: list[dict[str, Any]] = []
    for test_id, function in TESTS:
        try:
            value = function(bundle)
            if value.get("test_id") != test_id:
                raise ContractError(f"preflight returned wrong test id for {test_id}")
            results.append(value)
        except Exception as exc:  # the artifact must retain a path to the failed boundary
            results.append({"test_id": test_id, "status": "FAIL", "exception_type": type(exc).__name__, "message": str(exc)})
    passed = sum(row["status"] == "PASS" for row in results)
    artifact = {
        "schema_version": "4.0.0",
        "artifact_type": "DATA_FREE_PREFLIGHT",
        "freeze_sha256": FREEZE_HASH,
        "protocol_audit_sha256": PROTOCOL_AUDIT_HASH,
        "test_count": len(results),
        "passed_count": passed,
        "overall_status": "PASS_ALL" if passed == len(TESTS) else "FAIL",
        "scientific_outcomes": "NONE",
        "tests": results,
    }
    path = output_root / "preflight" / "CANDIDATE_V4_DATA_FREE_PREFLIGHT.json"
    digest = atomic_write_json(path, artifact, output_root=output_root)
    return {"path": str(path), "sha256": digest, **artifact}
