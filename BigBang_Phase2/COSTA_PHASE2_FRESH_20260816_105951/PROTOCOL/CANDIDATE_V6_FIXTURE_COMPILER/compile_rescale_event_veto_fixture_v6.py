#!/usr/bin/env python
"""Compile the outcome-blind, data-free V6 rescale-veto oracle.

This file is a prospective protocol artifact.  It imports no campaign code,
model, repository module, preflight implementation, data, R4 material, or
outcome.  Execution is denied unless Root separately freezes and releases the
exact compiler, manifest, schema, launcher, environment, arguments, and output
pre-state hashes.
"""

import hashlib
import json
import sys

import numpy as np
import scipy
from scipy import signal


SCHEMA_VERSION = "6.0.0"
FIXTURE_ID = "PF_RESCALE_EVENT_VETO_FIXTURE_V6"
VETO_ID = "RESCALE_ONLY_EVENT_VETO_V6"
BLOCK_SCHEMA_ID = "RESCALE_EVENT_VETO_BLOCK_RECORD_V6"
CONDITION_SCHEMA_ID = "RESCALE_EVENT_VETO_CONDITION_RECORD_V6"
CANONICALIZATION_ID = "PY_JSON_CANONICAL_ASCII_V1"
HASH_ENVELOPE_ID = "CANONICAL_JSON_SHA256_ENVELOPE_V6"
PAIR_REGISTRY_ID = "PAIRED_SHAM_IDENTITY_REGISTRY_V6"
OBSERVATION_FORMULA_ID = "XRAW_RE_PLUS_0P25_RT_DIV10_FLOAT64_V1"
AFFINE_EVALUATION_ID = "AFFINE_DIRECT_MUL_THEN_ADD_FLOAT64_V6"
CAMPAIGN_ID = "COSTA_PHASE2_FRESH_20260816_105951"
CANDIDATE_IDENTITY = "PHASE2_CONTROL_BENCH_V1"
FS_HZ = 200
N_SAMPLES = 36000
RETAINED_START = 2000
RETAINED_STOP = 34000
MATCH_TOLERANCE = 50
CONSUMERS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
STAGES = [
    "INTERVENTION_TO_SHAM",
    "RESCALED_TO_SHAM",
    "ORIGINAL_ADDED_TO_RESCALED_ADDED",
]


def canonical_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest().upper()


def sha256_file(path):
    with open(path, "rb") as handle:
        return sha256_bytes(handle.read())


def f64(value):
    return np.float64(value)


def fhex(value):
    value = f64(value)
    if not bool(np.isfinite(value)):
        raise ValueError("nonfinite value cannot be serialized as float hex")
    return float(value).hex()


def require(condition, code):
    if not bool(condition):
        raise RuntimeError(code)


def make_hash_envelope(record, excluded_field):
    require(excluded_field not in record, "HASH_DOMAIN_RECURSION")
    payload = canonical_bytes(record)
    return {
        "schema_id": HASH_ENVELOPE_ID,
        "algorithm": "SHA-256",
        "canonicalization_id": CANONICALIZATION_ID,
        "excluded_field": excluded_field,
        "payload_byte_length": len(payload),
        "payload_sha256": sha256_bytes(payload),
    }


def add_array_blob(blob_map, array, semantic_role):
    arr = np.asarray(array)
    require(arr.dtype == np.dtype("<f8"), "ARRAY_DTYPE_NOT_LITTLE_ENDIAN_FLOAT64")
    require(arr.flags.c_contiguous, "ARRAY_NOT_C_CONTIGUOUS")
    raw = arr.tobytes(order="C")
    digest = sha256_bytes(raw)
    if digest not in blob_map:
        blob_map[digest] = {
            "sha256": digest,
            "dtype": "<f8",
            "shape": [int(x) for x in arr.shape],
            "byte_count": len(raw),
            "encoding": "LOWERCASE_HEXADECIMAL_OF_EXACT_C_CONTIGUOUS_BYTES",
            "data_hex": raw.hex(),
            "semantic_roles": [semantic_role],
        }
    elif semantic_role not in blob_map[digest]["semantic_roles"]:
        blob_map[digest]["semantic_roles"].append(semantic_role)
    return digest


def sequential_moments(x):
    require(x.dtype == np.dtype("<f8"), "MOMENT_DTYPE")
    total = f64(0.0)
    for index in range(int(x.size)):
        total = f64(total + x[index])
        require(np.isfinite(total), "INVALID_NUMERIC_MEAN_SUM")
    mu = f64(total / f64(x.size))
    require(np.isfinite(mu), "INVALID_NUMERIC_MEAN")
    total_square = f64(0.0)
    for index in range(int(x.size)):
        deviation = f64(x[index] - mu)
        square = f64(deviation * deviation)
        require(np.isfinite(square), "INVALID_NUMERIC_SQUARE")
        total_square = f64(total_square + square)
        require(np.isfinite(total_square), "INVALID_NUMERIC_VARIANCE_SUM")
    variance = f64(total_square / f64(x.size))
    require(np.isfinite(variance), "INVALID_NUMERIC_VARIANCE")
    require(variance >= f64(0.0), "INVALID_NUMERIC_NEGATIVE_VARIANCE")
    sigma = f64(np.sqrt(variance))
    require(np.isfinite(sigma), "INVALID_NUMERIC_SIGMA")
    return mu, variance, sigma


def construct_observation(r_e, r_t):
    require(r_e.dtype == np.dtype("<f8"), "RE_DTYPE")
    require(r_t.dtype == np.dtype("<f8"), "RT_DTYPE")
    require(r_e.shape == (N_SAMPLES,), "RE_SHAPE")
    require(r_t.shape == (N_SAMPLES,), "RT_SHAPE")
    require(r_e.flags.c_contiguous and r_t.flags.c_contiguous, "SOURCE_CONTIGUITY")
    x = np.empty(N_SAMPLES, dtype="<f8")
    for index in range(N_SAMPLES):
        thalamic = f64(f64(0.25) * r_t[index])
        numerator = f64(r_e[index] + thalamic)
        x[index] = f64(numerator / f64(10.0))
    return x


def first_nonfinite(x):
    for index in range(int(x.size)):
        if not bool(np.isfinite(x[index])):
            return index
    return None


def affine_rescale(x_sham, x_intervention):
    mu_s, variance_s, sigma_s = sequential_moments(x_sham)
    mu_i, variance_i, sigma_i = sequential_moments(x_intervention)
    if sigma_s == f64(0.0) and sigma_i == f64(0.0):
        return {
            "status": "NOT_ESTIMABLE_BOTH_ZERO_VARIANCE",
            "mu_s": mu_s,
            "variance_s": variance_s,
            "sigma_s": sigma_s,
            "mu_i": mu_i,
            "variance_i": variance_i,
            "sigma_i": sigma_i,
            "a": None,
            "b": None,
            "x_affine": None,
        }
    if sigma_i == f64(0.0):
        return {
            "status": "NOT_ESTIMABLE_ZERO_INTERVENTION_VARIANCE",
            "mu_s": mu_s,
            "variance_s": variance_s,
            "sigma_s": sigma_s,
            "mu_i": mu_i,
            "variance_i": variance_i,
            "sigma_i": sigma_i,
            "a": None,
            "b": None,
            "x_affine": None,
        }
    if sigma_s == f64(0.0):
        return {
            "status": "NOT_ESTIMABLE_ZERO_SHAM_VARIANCE",
            "mu_s": mu_s,
            "variance_s": variance_s,
            "sigma_s": sigma_s,
            "mu_i": mu_i,
            "variance_i": variance_i,
            "sigma_i": sigma_i,
            "a": None,
            "b": None,
            "x_affine": None,
        }
    a = f64(sigma_s / sigma_i)
    require(np.isfinite(a) and a > f64(0.0), "INVALID_NUMERIC_A")
    a_times_mu_i = f64(a * mu_i)
    b = f64(mu_s - a_times_mu_i)
    require(np.isfinite(b), "INVALID_NUMERIC_B")
    x_affine = np.empty(N_SAMPLES, dtype="<f8")
    for index in range(N_SAMPLES):
        product = f64(a * x_intervention[index])
        x_affine[index] = f64(product + b)
        require(np.isfinite(x_affine[index]), "INVALID_NUMERIC_AFFINE_SAMPLE")
    return {
        "status": "VALID",
        "mu_s": mu_s,
        "variance_s": variance_s,
        "sigma_s": sigma_s,
        "mu_i": mu_i,
        "variance_i": variance_i,
        "sigma_i": sigma_i,
        "a": a,
        "b": b,
        "x_affine": x_affine,
    }


def preprocessing(x, sos):
    centered = np.asarray(x - np.mean(x, dtype=np.float64), dtype="<f8", order="C")
    filtered = np.asarray(
        signal.sosfiltfilt(sos, centered, padtype="odd", padlen=None),
        dtype="<f8",
        order="C",
    )
    require(filtered.shape == (N_SAMPLES,), "FILTER_SHAPE")
    require(np.all(np.isfinite(filtered)), "INVALID_DETECTOR_NONFINITE_FILTER")
    return centered, filtered


def spindle_rms(y):
    rms = np.full(N_SAMPLES, np.nan, dtype="<f8")
    for center in range(19, N_SAMPLES - 20):
        total = f64(0.0)
        for sample_index in range(center - 19, center + 21):
            square = f64(y[sample_index] * y[sample_index])
            total = f64(total + square)
        rms[center] = f64(np.sqrt(f64(total / f64(40.0))))
    return rms


def sham_spindle_thresholds(rms):
    retained = np.asarray(rms[RETAINED_START:RETAINED_STOP], dtype="<f8")
    require(np.all(np.isfinite(retained)), "INVALID_THRESHOLD_SUPPORT")
    median = f64(np.median(retained))
    absolute = np.asarray(np.abs(retained - median), dtype="<f8")
    mad_unscaled = f64(np.median(absolute))
    mad = f64(f64(1.4826) * mad_unscaled)
    high = f64(median + f64(f64(3.0) * mad))
    low = f64(median + f64(f64(1.5) * mad))
    require(np.isfinite(high) and np.isfinite(low), "INVALID_THRESHOLD_NONFINITE")
    return median, mad, high, low


def detect_so(y):
    crossings = []
    for index in range(1, N_SAMPLES):
        if y[index - 1] < f64(0.0) and y[index] >= f64(0.0):
            crossings.append(index)
    rows = []
    rejected = []
    for pair_index in range(len(crossings) - 1):
        left = crossings[pair_index]
        right = crossings[pair_index + 1]
        duration = right - left
        segment = y[left : right + 1]
        minimum_offset = int(np.argmin(segment))
        maximum_offset = int(np.argmax(segment))
        minimum_sample = left + minimum_offset
        maximum_sample = left + maximum_offset
        minimum = f64(segment[minimum_offset])
        maximum = f64(segment[maximum_offset])
        peak_to_peak = f64(maximum - minimum)
        reasons = []
        if not (80 <= duration <= 500):
            reasons.append("DURATION")
        if not (peak_to_peak >= f64(0.10)):
            reasons.append("PEAK_TO_PEAK")
        if not (minimum <= f64(-0.04)):
            reasons.append("TROUGH")
        if not (
            RETAINED_START < left < right < RETAINED_STOP
            and RETAINED_START < minimum_sample < RETAINED_STOP
            and RETAINED_START < maximum_sample < RETAINED_STOP
        ):
            reasons.append("STRICT_EDGE")
        primitive = {
            "left_upcross_sample": left,
            "right_upcross_sample": right,
            "minimum_sample": minimum_sample,
            "maximum_sample": maximum_sample,
            "minimum_m_u_float64_hex": fhex(minimum),
            "maximum_m_u_float64_hex": fhex(maximum),
            "peak_to_peak_m_u_float64_hex": fhex(peak_to_peak),
            "duration_samples": duration,
        }
        if reasons:
            primitive["rejection_reasons"] = reasons
            rejected.append(primitive)
        else:
            rows.append(primitive)
    rows.sort(
        key=lambda row: (
            row["left_upcross_sample"],
            row["right_upcross_sample"],
            row["minimum_sample"],
        )
    )
    events = []
    for index, row in enumerate(rows, start=1):
        events.append(
            {
                "event_id": "SO" + str(index).zfill(6),
                "event_type": "SO",
                "left_upcross_sample": row["left_upcross_sample"],
                "right_upcross_sample": row["right_upcross_sample"],
                "anchor_sample": row["minimum_sample"],
                "amplitude_float64_hex": row["peak_to_peak_m_u_float64_hex"],
            }
        )
    return events, rejected


def detect_spindle(rms, high, low):
    intervals = []
    for index in range(20, N_SAMPLES - 20):
        if rms[index - 1] < high and rms[index] >= high:
            left = index
            while left > 19 and np.isfinite(rms[left - 1]) and rms[left - 1] >= low:
                left -= 1
            right = index + 1
            while right < N_SAMPLES - 20 and np.isfinite(rms[right]) and rms[right] >= low:
                right += 1
            interval = (left, right)
            if interval not in intervals:
                intervals.append(interval)
    intervals.sort(key=lambda pair: (pair[0], pair[1]))
    merged = []
    for left, right in intervals:
        if not merged or left - merged[-1][1] > 20:
            merged.append([left, right])
        else:
            if right > merged[-1][1]:
                merged[-1][1] = right
    accepted_primitives = []
    rejected = []
    for left, right in merged:
        duration = right - left
        segment = rms[left:right]
        maximum_offset = int(np.argmax(segment))
        anchor = left + maximum_offset
        maximum = f64(segment[maximum_offset])
        reasons = []
        if not (60 <= duration <= 400):
            reasons.append("DURATION")
        if not (RETAINED_START < left < right < RETAINED_STOP):
            reasons.append("STRICT_EDGE")
        primitive = {
            "onset_sample": left,
            "offset_sample_exclusive": right,
            "anchor_sample": anchor,
            "amplitude_float64_hex": fhex(maximum),
            "duration_samples": duration,
        }
        if reasons:
            primitive["rejection_reasons"] = reasons
            rejected.append(primitive)
        else:
            accepted_primitives.append(primitive)
    accepted_primitives.sort(
        key=lambda row: (
            row["onset_sample"],
            row["offset_sample_exclusive"],
            row["anchor_sample"],
        )
    )
    events = []
    for index, row in enumerate(accepted_primitives, start=1):
        events.append(
            {
                "event_id": "SP" + str(index).zfill(6),
                "event_type": "SPINDLE",
                "onset_sample": row["onset_sample"],
                "offset_sample_exclusive": row["offset_sample_exclusive"],
                "anchor_sample": row["anchor_sample"],
                "amplitude_float64_hex": row["amplitude_float64_hex"],
            }
        )
    return events, rejected


def enumerate_and_match(left_events, right_events, stage):
    candidates = []
    for left in left_events:
        for right in right_events:
            distance = abs(int(left["anchor_sample"]) - int(right["anchor_sample"]))
            if distance <= MATCH_TOLERANCE:
                candidates.append(
                    {
                        "left_event_id": left["event_id"],
                        "right_event_id": right["event_id"],
                        "left_anchor_sample": int(left["anchor_sample"]),
                        "right_anchor_sample": int(right["anchor_sample"]),
                        "absolute_anchor_difference_samples": distance,
                    }
                )
    candidates.sort(
        key=lambda row: (
            row["absolute_anchor_difference_samples"],
            row["left_anchor_sample"],
            row["right_anchor_sample"],
            row["left_event_id"],
            row["right_event_id"],
        )
    )
    left_used = set()
    right_used = set()
    candidate_rows = []
    accepted_rows = []
    for candidate_rank, row in enumerate(candidates, start=1):
        left_before = row["left_event_id"] in left_used
        right_before = row["right_event_id"] in right_used
        if not left_before and not right_before:
            disposition = "ACCEPTED"
            left_used.add(row["left_event_id"])
            right_used.add(row["right_event_id"])
            accepted_rows.append(
                {
                    "accepted_rank": len(accepted_rows) + 1,
                    "candidate_rank": candidate_rank,
                    **row,
                }
            )
        elif left_before and right_before:
            disposition = "REJECTED_BOTH_ALREADY_CONSUMED"
        elif left_before:
            disposition = "REJECTED_LEFT_ALREADY_CONSUMED"
        else:
            disposition = "REJECTED_RIGHT_ALREADY_CONSUMED"
        candidate_rows.append(
            {
                "candidate_rank": candidate_rank,
                **row,
                "left_consumed_before": left_before,
                "right_consumed_before": right_before,
                "disposition": disposition,
            }
        )
    unmatched_left = [
        row["event_id"] for row in left_events if row["event_id"] not in left_used
    ]
    unmatched_right = [
        row["event_id"] for row in right_events if row["event_id"] not in right_used
    ]
    return {
        "match_stage": stage,
        "candidate_schema_id": "RESCALE_EVENT_MATCH_CANDIDATE_LEDGER_V6",
        "accepted_schema_id": "RESCALE_EVENT_MATCH_ACCEPTED_LEDGER_V6",
        "candidate_rows": candidate_rows,
        "accepted_rows": accepted_rows,
        "unmatched_left_event_ids": unmatched_left,
        "unmatched_right_event_ids": unmatched_right,
    }


def rank_by_event(match_result, side):
    key = "left_event_id" if side == "left" else "right_event_id"
    return {row[key]: row["accepted_rank"] for row in match_result["accepted_rows"]}


def classify_and_consume(e_s, e_i, e_a, i_to_s, a_to_s, added_to_added):
    i_s_rank = rank_by_event(i_to_s, "left")
    s_i_rank = rank_by_event(i_to_s, "right")
    a_s_rank = rank_by_event(a_to_s, "left")
    s_a_rank = rank_by_event(a_to_s, "right")
    i_a_rank = rank_by_event(added_to_added, "left")
    a_i_rank = rank_by_event(added_to_added, "right")
    classes = []
    consumption = []
    invalid = False

    for source_set, events in (("E_S", e_s), ("E_I", e_i), ("E_A", e_a)):
        seen = set()
        for row_index, event in enumerate(events):
            event_id = event["event_id"]
            duplicate = event_id in seen
            seen.add(event_id)
            reason = None
            if duplicate:
                event_class = "UNMATCHED_OR_AMBIGUOUS_INVALID"
                disposition = "UNMATCHED_OR_AMBIGUOUS_INVALID"
                reason = "DUPLICATE_EVENT_ID"
                valid_once = False
                invalid = True
            elif source_set == "E_S":
                matched_i = event_id in s_i_rank
                matched_a = event_id in s_a_rank
                if matched_i or matched_a:
                    event_class = "PRE_EXISTING"
                    disposition = "MATCHED_PRE_EXISTING"
                else:
                    event_class = "LOST"
                    disposition = "UNMATCHED_LOST"
                valid_once = True
            elif source_set == "E_I":
                if event_id in i_s_rank:
                    event_class = "PRE_EXISTING"
                    disposition = "MATCHED_PRE_EXISTING"
                elif event_id in i_a_rank:
                    event_class = "PERSISTENT"
                    disposition = "MATCHED_PERSISTENT"
                else:
                    event_class = "RESCALE_ONLY"
                    disposition = "UNMATCHED_RESCALE_ONLY"
                valid_once = True
            else:
                if event_id in a_s_rank:
                    event_class = "PRE_EXISTING"
                    disposition = "MATCHED_PRE_EXISTING"
                    valid_once = True
                elif event_id in a_i_rank:
                    event_class = "PERSISTENT"
                    disposition = "MATCHED_PERSISTENT"
                    valid_once = True
                else:
                    event_class = "UNMATCHED_OR_AMBIGUOUS_INVALID"
                    disposition = "UNMATCHED_OR_AMBIGUOUS_INVALID"
                    reason = "AFFINE_ADDED_UNMATCHED"
                    valid_once = True
                    invalid = True
            class_row = {
                "source_set": source_set,
                "source_row_index": row_index,
                "event_id": event_id,
                "event_class": event_class,
                "intervention_to_sham_accepted_rank_or_null": (
                    i_s_rank.get(event_id)
                    if source_set == "E_I"
                    else s_i_rank.get(event_id) if source_set == "E_S" else None
                ),
                "rescaled_to_sham_accepted_rank_or_null": (
                    a_s_rank.get(event_id)
                    if source_set == "E_A"
                    else s_a_rank.get(event_id) if source_set == "E_S" else None
                ),
                "original_added_to_rescaled_added_accepted_rank_or_null": (
                    i_a_rank.get(event_id)
                    if source_set == "E_I"
                    else a_i_rank.get(event_id) if source_set == "E_A" else None
                ),
                "reason_code_or_null": reason,
            }
            classes.append(class_row)
            consumption.append(
                {
                    "source_set": source_set,
                    "source_row_index": row_index,
                    "event_id": event_id,
                    "intervention_to_sham_accepted_rank_or_null": class_row[
                        "intervention_to_sham_accepted_rank_or_null"
                    ],
                    "rescaled_to_sham_accepted_rank_or_null": class_row[
                        "rescaled_to_sham_accepted_rank_or_null"
                    ],
                    "original_added_to_rescaled_added_accepted_rank_or_null": class_row[
                        "original_added_to_rescaled_added_accepted_rank_or_null"
                    ],
                    "consumption_disposition": disposition,
                    "consumed_exactly_once_boolean": valid_once,
                }
            )
    expected_rows = len(e_s) + len(e_i) + len(e_a)
    require(len(classes) == expected_rows, "CLASS_LEDGER_NOT_EXHAUSTIVE")
    require(len(consumption) == expected_rows, "CONSUMPTION_LEDGER_NOT_EXHAUSTIVE")
    return classes, consumption, invalid


def match_and_count(e_s, e_i, e_a):
    i_to_s = enumerate_and_match(e_i, e_s, "INTERVENTION_TO_SHAM")
    a_to_s = enumerate_and_match(e_a, e_s, "RESCALED_TO_SHAM")
    i_added = [row for row in e_i if row["event_id"] in i_to_s["unmatched_left_event_ids"]]
    a_added = [row for row in e_a if row["event_id"] in a_to_s["unmatched_left_event_ids"]]
    added_to_added = enumerate_and_match(
        i_added, a_added, "ORIGINAL_ADDED_TO_RESCALED_ADDED"
    )
    classes, consumption, ledger_invalid = classify_and_consume(
        e_s, e_i, e_a, i_to_s, a_to_s, added_to_added
    )
    n_added_original = len(i_to_s["unmatched_left_event_ids"])
    n_added_rescaled = len(a_to_s["unmatched_left_event_ids"])
    n_persistent = len(added_to_added["accepted_rows"])
    counts = {
        "N_sham": len(e_s),
        "N_intervention": len(e_i),
        "N_rescaled": len(e_a),
        "N_added_original": n_added_original,
        "N_added_rescaled": n_added_rescaled,
        "N_persistent": n_persistent,
        "N_rescale_only": n_added_original - n_persistent,
    }
    require(counts["N_rescale_only"] >= 0, "NEGATIVE_RESCALE_ONLY_COUNT")
    block_boolean = (
        n_added_original > 0 and n_added_rescaled == 0 and n_persistent == 0
    )
    fraction = None if n_added_original == 0 else f64(counts["N_rescale_only"] / n_added_original)
    return {
        "matches": [i_to_s, a_to_s, added_to_added],
        "event_class_ledger": classes,
        "row_consumption_ledger": consumption,
        "ledger_invalid": ledger_invalid,
        "counts": counts,
        "F_rescale_only": fraction,
        "block_rescale_only": block_boolean,
    }


def detector_bundle(x_s, x_i, x_a, blob_map, case_id):
    so_sos = np.asarray(
        signal.butter(
            4, [0.3, 1.5], btype="bandpass", analog=False, output="sos", fs=200.0
        ),
        dtype="<f8",
        order="C",
    )
    spindle_sos = np.asarray(
        signal.butter(
            4, [11.0, 16.0], btype="bandpass", analog=False, output="sos", fs=200.0
        ),
        dtype="<f8",
        order="C",
    )
    roles = (("SHAM", x_s), ("INTERVENTION", x_i), ("AFFINE", x_a))
    processed = {}
    for role, x in roles:
        centered_so, y_so = preprocessing(x, so_sos)
        centered_spindle, y_spindle = preprocessing(x, spindle_sos)
        require(
            np.array_equal(centered_so, centered_spindle),
            "CENTERING_NOT_IDENTICAL_ACROSS_BANDS",
        )
        rms = spindle_rms(y_spindle)
        processed[role] = {
            "centered": centered_so,
            "so_filtered": y_so,
            "spindle_filtered": y_spindle,
            "spindle_rms": rms,
            "array_sha256": {
                "centered": add_array_blob(blob_map, centered_so, case_id + "/" + role + "/centered"),
                "so_filtered": add_array_blob(blob_map, y_so, case_id + "/" + role + "/so_filtered"),
                "spindle_filtered": add_array_blob(blob_map, y_spindle, case_id + "/" + role + "/spindle_filtered"),
                "spindle_rms": add_array_blob(blob_map, rms, case_id + "/" + role + "/spindle_rms"),
            },
        }
    median, mad, high, low = sham_spindle_thresholds(processed["SHAM"]["spindle_rms"])
    event_tables = {}
    rejected_tables = {}
    for role, _ in roles:
        so_events, so_rejected = detect_so(processed[role]["so_filtered"])
        spindle_events, spindle_rejected = detect_spindle(
            processed[role]["spindle_rms"], high, low
        )
        event_tables[role] = {"SO": so_events, "SPINDLE": spindle_events}
        rejected_tables[role] = {"SO": so_rejected, "SPINDLE": spindle_rejected}
    sos_so_hex = [[fhex(value) for value in row] for row in so_sos.tolist()]
    sos_spindle_hex = [[fhex(value) for value in row] for row in spindle_sos.tolist()]
    provenance = {
        "filter_implementation": "scipy.signal.butter(4,band,btype='bandpass',analog=False,output='sos',fs=200.0) then scipy.signal.sosfiltfilt(sos,centered,padtype='odd',padlen=None)",
        "so_sos_coefficients_float64_hex": sos_so_hex,
        "spindle_sos_coefficients_float64_hex": sos_spindle_hex,
        "spindle_sham_median_float64_hex": fhex(median),
        "spindle_sham_mad_float64_hex": fhex(mad),
        "spindle_high_threshold_float64_hex": fhex(high),
        "spindle_low_threshold_float64_hex": fhex(low),
        "so_peak_to_peak_threshold_float64_hex": fhex(f64(0.10)),
        "so_negative_trough_threshold_float64_hex": fhex(f64(-0.04)),
        "duration_bounds_samples": {"SO": [80, 500], "SPINDLE": [60, 400]},
        "retained_bounds_samples": [RETAINED_START, RETAINED_STOP],
        "comparison_semantics": "threshold equality passes; retained boundaries are strict",
    }
    return {
        "processed": processed,
        "event_tables": event_tables,
        "rejected_event_candidates": rejected_tables,
        "threshold_provenance": provenance,
        "threshold_provenance_sha256": sha256_bytes(canonical_bytes(provenance)),
    }


def raised_cosine_burst(sample_axis, center_sample, half_width_samples, frequency_hz, amplitude):
    result = np.zeros(N_SAMPLES, dtype="<f8")
    start = max(0, int(center_sample - half_width_samples))
    stop = min(N_SAMPLES, int(center_sample + half_width_samples + 1))
    local = sample_axis[start:stop] - f64(center_sample)
    envelope = f64(0.5) * (
        f64(1.0) + np.cos(np.pi * local / f64(half_width_samples + 1))
    )
    carrier = np.sin(f64(2.0 * np.pi * frequency_hz / FS_HZ) * sample_axis[start:stop])
    result[start:stop] = np.asarray(f64(amplitude) * envelope * carrier, dtype="<f8")
    return result


def base_observation(sample_axis):
    slow = f64(0.12) * np.sin(f64(2.0 * np.pi * 0.85 / FS_HZ) * sample_axis)
    background_amplitude = f64(0.004) * (
        f64(1.0)
        + f64(0.45) * np.sin(f64(2.0 * np.pi * 0.20 / FS_HZ) * sample_axis)
    )
    background_sigma = background_amplitude * np.sin(
        f64(2.0 * np.pi * 13.0 / FS_HZ) * sample_axis
    )
    result = np.asarray(slow + background_sigma, dtype="<f8", order="C")
    for center in (6000, 12000, 18000, 24000, 30000):
        result = np.asarray(
            result + raised_cosine_burst(sample_axis, center, 110, 13.0, 0.055),
            dtype="<f8",
            order="C",
        )
    return result


def observation_to_sources(x):
    r_e = np.asarray(f64(10.0) * x, dtype="<f8", order="C")
    r_t = np.zeros(N_SAMPLES, dtype="<f8")
    return r_e, r_t


def synthesize_so_threshold_equality(sample_axis):
    """Return one fixed 36,000-sample trace with a production SO equality witness.

    The finite search order is prospective and deterministic.  It searches only
    adjacent binary64 amplitudes around the analytic linear-filter scale; the
    emitted fixture contains the selected raw bytes, so replay needs no search.
    """
    unit = np.asarray(
        np.sin(f64(2.0 * np.pi * 0.85 / FS_HZ) * sample_axis),
        dtype="<f8",
        order="C",
    )
    sos = np.asarray(
        signal.butter(
            4, [0.3, 1.5], btype="bandpass", analog=False, output="sos", fs=200.0
        ),
        dtype="<f8",
        order="C",
    )
    _, filtered_unit = preprocessing(unit, sos)
    unit_events, unit_rejected = detect_so(filtered_unit)
    primitives = []
    primitives.extend(unit_rejected)
    for event in unit_events:
        left = event["left_upcross_sample"]
        right = event["right_upcross_sample"]
        segment = filtered_unit[left : right + 1]
        minimum = f64(np.min(segment))
        maximum = f64(np.max(segment))
        primitives.append(
            {
                "left_upcross_sample": left,
                "right_upcross_sample": right,
                "minimum_m_u_float64_hex": fhex(minimum),
                "peak_to_peak_m_u_float64_hex": fhex(f64(maximum - minimum)),
            }
        )
    interior = [
        row
        for row in primitives
        if 5000 < row["left_upcross_sample"] < row["right_upcross_sample"] < 30000
    ]
    require(len(interior) > 0, "THRESHOLD_SYNTHESIS_NO_UNIT_CYCLE")
    unit_peak_to_peak = f64(float.fromhex(interior[0]["peak_to_peak_m_u_float64_hex"]))
    center_scale = f64(f64(0.10) / unit_peak_to_peak)
    candidates = [center_scale]
    lower = center_scale
    upper = center_scale
    for _ in range(1, 1025):
        lower = f64(np.nextafter(lower, f64(-np.inf)))
        upper = f64(np.nextafter(upper, f64(np.inf)))
        candidates.extend([lower, upper])
    threshold_hex = fhex(f64(0.10))
    for candidate_rank, scale in enumerate(candidates, start=1):
        trace = np.asarray(scale * unit, dtype="<f8", order="C")
        _, filtered = preprocessing(trace, sos)
        accepted, rejected = detect_so(filtered)
        replay = list(rejected)
        for event in accepted:
            left = event["left_upcross_sample"]
            right = event["right_upcross_sample"]
            segment = filtered[left : right + 1]
            minimum_offset = int(np.argmin(segment))
            maximum_offset = int(np.argmax(segment))
            replay.append(
                {
                    "left_upcross_sample": left,
                    "right_upcross_sample": right,
                    "minimum_sample": left + minimum_offset,
                    "maximum_sample": left + maximum_offset,
                    "minimum_m_u_float64_hex": fhex(segment[minimum_offset]),
                    "maximum_m_u_float64_hex": fhex(segment[maximum_offset]),
                    "peak_to_peak_m_u_float64_hex": fhex(
                        f64(segment[maximum_offset] - segment[minimum_offset])
                    ),
                    "duration_samples": right - left,
                }
            )
        for row in replay:
            if (
                row["peak_to_peak_m_u_float64_hex"] == threshold_hex
                and float.fromhex(row["minimum_m_u_float64_hex"]) <= -0.04
                and 80 <= row["duration_samples"] <= 500
                and RETAINED_START < row["left_upcross_sample"] < row["right_upcross_sample"] < RETAINED_STOP
            ):
                return trace, {
                    "candidate_rank": candidate_rank,
                    "scale_float64_hex": fhex(scale),
                    "equality_field": "peak_to_peak_m_u",
                    "equality_value_float64_hex": threshold_hex,
                    "cycle": row,
                }
    raise RuntimeError("THRESHOLD_SYNTHESIS_EXHAUSTED")


def fixed_source_cases():
    sample_axis = np.arange(N_SAMPLES, dtype="<f8")
    base = base_observation(sample_axis)
    persistent_burst = raised_cosine_burst(sample_axis, 21000, 110, 13.0, 0.080)
    vanish_burst = raised_cosine_burst(sample_axis, 21000, 110, 13.0, 0.080)
    five_hz = f64(10.0) * np.sin(f64(2.0 * np.pi * 5.0 / FS_HZ) * sample_axis)
    tie_sham = np.asarray(
        f64(0.12) * np.sin(f64(2.0 * np.pi * 0.85 / FS_HZ) * sample_axis)
        + f64(0.004) * np.sin(f64(2.0 * np.pi * 13.0 / FS_HZ) * sample_axis)
        + raised_cosine_burst(sample_axis, 20000, 110, 13.0, 0.060),
        dtype="<f8",
        order="C",
    )
    tie_intervention = np.asarray(
        f64(0.12) * np.sin(f64(2.0 * np.pi * 0.85 / FS_HZ) * sample_axis)
        + f64(0.004) * np.sin(f64(2.0 * np.pi * 13.0 / FS_HZ) * sample_axis)
        + raised_cosine_burst(sample_axis, 19950, 110, 13.0, 0.060)
        + raised_cosine_burst(sample_axis, 20050, 110, 13.0, 0.060)
        + raised_cosine_burst(sample_axis, 27000, 110, 13.0, 0.060),
        dtype="<f8",
        order="C",
    )
    boundary_sham = np.asarray(base, dtype="<f8", order="C")
    boundary_intervention = np.asarray(
        base
        + raised_cosine_burst(sample_axis, 8000, 180, 13.0, 0.080)
        + raised_cosine_burst(sample_axis, 2000, 180, 13.0, 0.080),
        dtype="<f8",
        order="C",
    )
    normal = np.asarray(base, dtype="<f8", order="C")
    constant_three = np.full(N_SAMPLES, f64(3.0), dtype="<f8")
    constant_minus_two = np.full(N_SAMPLES, f64(-2.0), dtype="<f8")
    nonfinite = np.asarray(base + f64(4.0), dtype="<f8", order="C")
    nonfinite[17001] = np.nan
    threshold_trace, threshold_witness = synthesize_so_threshold_equality(sample_axis)
    return [
        {"case_id":"E2E_MEAN_SHIFT","coverage":["PURE_MEAN_SHIFT"],"x_sham":normal,"x_intervention":np.asarray(base+f64(4.0),dtype="<f8",order="C")},
        {"case_id":"E2E_VARIANCE_SCALE","coverage":["PURE_VARIANCE_RESCALE"],"x_sham":normal,"x_intervention":np.asarray(f64(2.0)*base,dtype="<f8",order="C")},
        {"case_id":"E2E_GENUINE_EVENT_PERSISTENCE","coverage":["GENUINE_EVENT_PERSISTENCE","NONZERO_EVENT_SURVIVAL","UNMATCHED_EVENT"],"x_sham":normal,"x_intervention":np.asarray(base+persistent_burst,dtype="<f8",order="C")},
        {"case_id":"E2E_ALL_ADDED_EVENTS_VANISH","coverage":["ALL_ADDED_EVENTS_VANISH_VETO"],"x_sham":normal,"x_intervention":np.asarray(base+five_hz+vanish_burst,dtype="<f8",order="C")},
        {"case_id":"E2E_MATCH_TIE_AND_UNMATCHED","coverage":["EXACT_MATCHER_TIE","UNMATCHED_EVENT"],"x_sham":tie_sham,"x_intervention":tie_intervention},
        {"case_id":"E2E_EDGE_AND_T8_WINDOW_BOUNDARY","coverage":["STRICT_RETAINED_EDGE","T8_WINDOW_BOUNDARY_TOUCH","CROSS_WINDOW_REJECTION"],"x_sham":boundary_sham,"x_intervention":boundary_intervention},
        {"case_id":"E2E_ZERO_INTERVENTION_VARIANCE","coverage":["ZERO_INTERVENTION_VARIANCE"],"x_sham":normal,"x_intervention":constant_three},
        {"case_id":"E2E_ZERO_SHAM_VARIANCE","coverage":["ZERO_SHAM_VARIANCE"],"x_sham":constant_minus_two,"x_intervention":normal},
        {"case_id":"E2E_BOTH_ZERO_VARIANCE","coverage":["BOTH_ZERO_VARIANCE"],"x_sham":constant_minus_two,"x_intervention":constant_three},
        {"case_id":"E2E_NONFINITE_INTERVENTION","coverage":["NONFINITE_INPUT"],"x_sham":normal,"x_intervention":nonfinite},
        {"case_id":"E2E_SO_THRESHOLD_EQUALITY","coverage":["EXACT_THRESHOLD_EQUALITY"],"x_sham":normal,"x_intervention":threshold_trace,"design_witness":threshold_witness},
    ]


def event_window_membership(event, event_type):
    windows = [
        ("W1", 3000, 8000),
        ("W2", 8000, 13000),
        ("W3", 13000, 18000),
        ("W4", 18000, 23000),
        ("W5", 23000, 28000),
        ("W6", 28000, 33000),
    ]
    if event_type == "SO":
        onset = event["left_upcross_sample"]
        offset = event["right_upcross_sample"]
        extrema = [event["anchor_sample"]]
    else:
        onset = event["onset_sample"]
        offset = event["offset_sample_exclusive"]
        extrema = [event["anchor_sample"]]
    accepted = []
    rejected = []
    for window_id, start, stop in windows:
        strict = start < onset and offset < stop and all(start < x < stop for x in extrema)
        touches_or_crosses = onset <= start <= offset or onset <= stop <= offset
        if strict:
            accepted.append(window_id)
        elif touches_or_crosses:
            rejected.append(
                {
                    "window_id": window_id,
                    "window_start_sample": start,
                    "window_stop_sample": stop,
                    "reason": "T8_BOUNDARY_REJECT",
                }
            )
    return {"accepted_window_ids": accepted, "boundary_rejections": rejected}


def null_block_tail(record):
    nullable = [
        "mu_sham_float64_hex_or_null",
        "v_sham_float64_hex_or_null",
        "sigma_sham_float64_hex_or_null",
        "mu_intervention_float64_hex_or_null",
        "v_intervention_float64_hex_or_null",
        "sigma_intervention_float64_hex_or_null",
        "affine_a_float64_hex_or_null",
        "affine_b_float64_hex_or_null",
        "affine_evaluation_id_or_null",
        "rescaled_raw_sha256_or_null",
        "mu_rescaled_float64_hex_or_null",
        "v_rescaled_float64_hex_or_null",
        "threshold_provenance_object_or_null",
        "threshold_provenance_sha256_or_null",
        "sham_event_table_or_null",
        "intervention_event_table_or_null",
        "rescaled_event_table_or_null",
        "candidate_match_ledgers_or_null",
        "accepted_match_ledgers_or_null",
        "event_class_ledger_or_null",
        "row_consumption_ledger_or_null",
        "unmatched_event_id_arrays_or_null",
        "integer_counts_or_null",
        "density_contrasts_events_per_min_or_null",
        "F_rescale_only_or_null",
        "F_rescale_only_null_reason_or_null",
        "block_rescale_only_or_null",
    ]
    for field in nullable:
        record[field] = None


def base_block_record(case_id, event_type):
    condition = "LEVEL_A_T4_DOSE_INDEX_06"
    draw = "D00_NOMINAL"
    seed = 11003
    context = "LEVEL_A_WORKER"
    target = "T4"
    return {
        "schema_version": SCHEMA_VERSION,
        "canonical_block_schema_id": BLOCK_SCHEMA_ID,
        "veto_id": VETO_ID,
        "block_id": "RV6B::" + context + "::" + target + "::" + condition + "::" + event_type + "::" + draw + "::S" + str(seed),
        "raw_pair_key": "RV6P::" + context + "::" + target + "::" + condition + "::" + draw + "::S" + str(seed),
        "run_id_intervention": "LA__LEVEL_A_T4_DOSE_INDEX_06__D00_NOMINAL__S11003",
        "run_id_sham": "LA__LEVEL_A_T4_DOSE_INDEX_03__D00_NOMINAL__S11003",
        "sham_logical_alias_or_null": "LEVEL_A_SHARED_SHAM::D00_NOMINAL::S11003",
        "paired_sham_registry_id": PAIR_REGISTRY_ID,
        "self_pair_boolean": False,
        "execution_context": context,
        "target_id": target,
        "condition_id": condition,
        "downstream_consumer_ids": list(CONSUMERS),
        "event_type": event_type,
        "parameter_draw_id": draw,
        "seed": seed,
        "intervention_input_validity_status": "VALID_RUN",
        "sham_input_validity_status": "VALID_RUN",
        "intervention_finite_status": "NOT_SCANNED_DUE_PRIOR_STATUS",
        "sham_finite_status": "NOT_SCANNED_DUE_PRIOR_STATUS",
        "sham_forcing_status": "ALL_FOUR_CHANNELS_EXACT_ZERO",
        "first_nonfinite_role_or_null": None,
        "first_nonfinite_index_or_null": None,
        "raw_shape": [N_SAMPLES],
        "raw_dtype": "<f8",
        "raw_endianness": "LITTLE",
        "raw_c_contiguous": True,
        "sample_count": N_SAMPLES,
        "sample_domain_half_open": [0, N_SAMPLES],
        "observation_formula_id": OBSERVATION_FORMULA_ID,
        "sham_raw_sha256": None,
        "intervention_raw_sha256": None,
    }


def finalize_block_record(record):
    record_without_envelope = dict(record)
    record["block_record_hash_envelope"] = make_hash_envelope(
        record_without_envelope, "block_record_hash_envelope"
    )
    return record


def compile_raw_case(case, blob_map):
    case_id = case["case_id"]
    sham_r_e, sham_r_t = observation_to_sources(case["x_sham"])
    intervention_r_e, intervention_r_t = observation_to_sources(case["x_intervention"])
    source_hashes = {
        "sham_r_E_stored": add_array_blob(blob_map, sham_r_e, case_id + "/sham/r_E_stored"),
        "sham_r_T_stored": add_array_blob(blob_map, sham_r_t, case_id + "/sham/r_T_stored"),
        "intervention_r_E_stored": add_array_blob(blob_map, intervention_r_e, case_id + "/intervention/r_E_stored"),
        "intervention_r_T_stored": add_array_blob(blob_map, intervention_r_t, case_id + "/intervention/r_T_stored"),
    }
    x_s = construct_observation(sham_r_e, sham_r_t)
    x_i = construct_observation(intervention_r_e, intervention_r_t)
    x_s_hash = add_array_blob(blob_map, x_s, case_id + "/sham/x_raw")
    x_i_hash = add_array_blob(blob_map, x_i, case_id + "/intervention/x_raw")
    sham_nonfinite = first_nonfinite(x_s)
    intervention_nonfinite = first_nonfinite(x_i)
    result = {
        "case_id": case_id,
        "coverage_tags": list(case["coverage"]),
        "design_witness_or_null": case.get("design_witness"),
        "source_array_sha256": source_hashes,
        "observation_array_sha256": {"sham_x_raw": x_s_hash, "intervention_x_raw": x_i_hash},
        "observation_equation": "For n=0..35999: q=float64(0.25*r_T[n]); p=float64(r_E[n]+q); x_raw[n]=float64(p/10.0).",
        "raw_pair_validity": {
            "raw_pair_key": "RV6P::LEVEL_A_WORKER::T4::LEVEL_A_T4_DOSE_INDEX_06::D00_NOMINAL::S11003",
            "sham_forcing_status": "ALL_FOUR_CHANNELS_EXACT_ZERO",
            "sham_input_validity_status": "VALID_RUN",
            "intervention_input_validity_status": "VALID_RUN",
            "sham_finite_status": "FINITE_ALL_36000" if sham_nonfinite is None else "NONFINITE_AT_RECORDED_INDEX",
            "intervention_finite_status": "FINITE_ALL_36000" if intervention_nonfinite is None else "NONFINITE_AT_RECORDED_INDEX",
            "first_nonfinite_role_or_null": "SHAM" if sham_nonfinite is not None else "INTERVENTION" if intervention_nonfinite is not None else None,
            "first_nonfinite_index_or_null": sham_nonfinite if sham_nonfinite is not None else intervention_nonfinite,
        },
        "moment_and_affine_expected": None,
        "detector_expected": None,
        "event_matching_expected": None,
        "t8_boundary_expected": None,
        "canonical_block_records": [],
    }
    if sham_nonfinite is not None or intervention_nonfinite is not None:
        for event_type in ("SO", "SPINDLE"):
            block = base_block_record(case_id, event_type)
            block["sham_raw_sha256"] = x_s_hash
            block["intervention_raw_sha256"] = x_i_hash
            block["sham_finite_status"] = result["raw_pair_validity"]["sham_finite_status"]
            block["intervention_finite_status"] = result["raw_pair_validity"]["intervention_finite_status"]
            block["first_nonfinite_role_or_null"] = result["raw_pair_validity"]["first_nonfinite_role_or_null"]
            block["first_nonfinite_index_or_null"] = result["raw_pair_validity"]["first_nonfinite_index_or_null"]
            null_block_tail(block)
            block["status"] = "INVALID_NONFINITE_INPUT"
            block["reason_code_or_null"] = "INVALID_NONFINITE_INPUT"
            block["first_failing_field_or_null"] = result["raw_pair_validity"]["first_nonfinite_role_or_null"] + "[" + str(result["raw_pair_validity"]["first_nonfinite_index_or_null"]) + "]"
            result["canonical_block_records"].append(finalize_block_record(block))
        result["expected_terminal_status"] = "INVALID_NONFINITE_INPUT"
        return result

    affine = affine_rescale(x_s, x_i)
    result["moment_and_affine_expected"] = {
        "mu_S_float64_hex": fhex(affine["mu_s"]),
        "v_S_float64_hex": fhex(affine["variance_s"]),
        "sigma_S_float64_hex": fhex(affine["sigma_s"]),
        "mu_I_float64_hex": fhex(affine["mu_i"]),
        "v_I_float64_hex": fhex(affine["variance_i"]),
        "diagnostic_sigma_I_float64_hex": fhex(affine["sigma_i"]),
        "a_float64_hex_or_null": None if affine["a"] is None else fhex(affine["a"]),
        "b_float64_hex_or_null": None if affine["b"] is None else fhex(affine["b"]),
        "affine_evaluation_id_or_null": None if affine["x_affine"] is None else AFFINE_EVALUATION_ID,
        "x_A_sha256_or_null": None,
        "mu_A_float64_hex_or_null": None,
        "v_A_float64_hex_or_null": None,
        "status": affine["status"],
    }
    if affine["status"] != "VALID":
        for event_type in ("SO", "SPINDLE"):
            block = base_block_record(case_id, event_type)
            block["sham_raw_sha256"] = x_s_hash
            block["intervention_raw_sha256"] = x_i_hash
            block["sham_finite_status"] = "FINITE_ALL_36000"
            block["intervention_finite_status"] = "FINITE_ALL_36000"
            null_block_tail(block)
            block["mu_sham_float64_hex_or_null"] = fhex(affine["mu_s"])
            block["v_sham_float64_hex_or_null"] = fhex(affine["variance_s"])
            block["sigma_sham_float64_hex_or_null"] = fhex(affine["sigma_s"])
            block["mu_intervention_float64_hex_or_null"] = fhex(affine["mu_i"])
            block["v_intervention_float64_hex_or_null"] = fhex(affine["variance_i"])
            block["sigma_intervention_float64_hex_or_null"] = fhex(affine["sigma_i"])
            block["status"] = affine["status"]
            block["reason_code_or_null"] = affine["status"]
            block["first_failing_field_or_null"] = (
                "sigma_intervention_float64_hex_or_null"
                if affine["sigma_i"] == f64(0.0) and affine["sigma_s"] != f64(0.0)
                else "sigma_sham_float64_hex_or_null"
                if affine["sigma_s"] == f64(0.0) and affine["sigma_i"] != f64(0.0)
                else "sigma_sham_float64_hex_or_null+sigma_intervention_float64_hex_or_null"
            )
            result["canonical_block_records"].append(finalize_block_record(block))
        result["expected_terminal_status"] = affine["status"]
        return result

    x_a = affine["x_affine"]
    x_a_hash = add_array_blob(blob_map, x_a, case_id + "/affine/x_raw")
    mu_a, variance_a, _ = sequential_moments(x_a)
    result["moment_and_affine_expected"]["x_A_sha256_or_null"] = x_a_hash
    result["moment_and_affine_expected"]["mu_A_float64_hex_or_null"] = fhex(mu_a)
    result["moment_and_affine_expected"]["v_A_float64_hex_or_null"] = fhex(variance_a)
    detectors = detector_bundle(x_s, x_i, x_a, blob_map, case_id)
    result["detector_expected"] = {
        "threshold_provenance": detectors["threshold_provenance"],
        "threshold_provenance_sha256": detectors["threshold_provenance_sha256"],
        "official_sham_event_tables": detectors["event_tables"]["SHAM"],
        "official_intervention_event_tables": detectors["event_tables"]["INTERVENTION"],
        "affine_diagnostic_event_tables": detectors["event_tables"]["AFFINE"],
        "rejected_event_candidates": detectors["rejected_event_candidates"],
        "processed_array_sha256": {
            role: detectors["processed"][role]["array_sha256"]
            for role in ("SHAM", "INTERVENTION", "AFFINE")
        },
        "official_table_rerun_deep_equality_required": True,
    }
    matching_by_type = {}
    boundary_by_type = {}
    for event_type in ("SO", "SPINDLE"):
        matched = match_and_count(
            detectors["event_tables"]["SHAM"][event_type],
            detectors["event_tables"]["INTERVENTION"][event_type],
            detectors["event_tables"]["AFFINE"][event_type],
        )
        matching_by_type[event_type] = matched
        boundary_by_type[event_type] = {
            role: [
                {"event_id": event["event_id"], **event_window_membership(event, event_type)}
                for event in detectors["event_tables"][role][event_type]
            ]
            for role in ("SHAM", "INTERVENTION", "AFFINE")
        }
        block = base_block_record(case_id, event_type)
        block["sham_raw_sha256"] = x_s_hash
        block["intervention_raw_sha256"] = x_i_hash
        block["sham_finite_status"] = "FINITE_ALL_36000"
        block["intervention_finite_status"] = "FINITE_ALL_36000"
        block["mu_sham_float64_hex_or_null"] = fhex(affine["mu_s"])
        block["v_sham_float64_hex_or_null"] = fhex(affine["variance_s"])
        block["sigma_sham_float64_hex_or_null"] = fhex(affine["sigma_s"])
        block["mu_intervention_float64_hex_or_null"] = fhex(affine["mu_i"])
        block["v_intervention_float64_hex_or_null"] = fhex(affine["variance_i"])
        block["sigma_intervention_float64_hex_or_null"] = fhex(affine["sigma_i"])
        block["affine_a_float64_hex_or_null"] = fhex(affine["a"])
        block["affine_b_float64_hex_or_null"] = fhex(affine["b"])
        block["affine_evaluation_id_or_null"] = AFFINE_EVALUATION_ID
        block["rescaled_raw_sha256_or_null"] = x_a_hash
        block["mu_rescaled_float64_hex_or_null"] = fhex(mu_a)
        block["v_rescaled_float64_hex_or_null"] = fhex(variance_a)
        block["threshold_provenance_object_or_null"] = detectors["threshold_provenance"]
        block["threshold_provenance_sha256_or_null"] = detectors["threshold_provenance_sha256"]
        block["sham_event_table_or_null"] = detectors["event_tables"]["SHAM"][event_type]
        block["intervention_event_table_or_null"] = detectors["event_tables"]["INTERVENTION"][event_type]
        block["rescaled_event_table_or_null"] = detectors["event_tables"]["AFFINE"][event_type]
        block["candidate_match_ledgers_or_null"] = [
            {"match_stage": item["match_stage"], "rows": item["candidate_rows"]}
            for item in matched["matches"]
        ]
        block["accepted_match_ledgers_or_null"] = [
            {"match_stage": item["match_stage"], "rows": item["accepted_rows"]}
            for item in matched["matches"]
        ]
        block["event_class_ledger_or_null"] = matched["event_class_ledger"]
        block["row_consumption_ledger_or_null"] = matched["row_consumption_ledger"]
        block["unmatched_event_id_arrays_or_null"] = {
            "intervention_to_sham_left_intervention_ids": matched["matches"][0]["unmatched_left_event_ids"],
            "intervention_to_sham_right_sham_ids": matched["matches"][0]["unmatched_right_event_ids"],
            "rescaled_to_sham_left_rescaled_ids": matched["matches"][1]["unmatched_left_event_ids"],
            "rescaled_to_sham_right_sham_ids": matched["matches"][1]["unmatched_right_event_ids"],
            "original_added_to_rescaled_added_left_original_added_ids": matched["matches"][2]["unmatched_left_event_ids"],
            "original_added_to_rescaled_added_right_rescaled_added_ids": matched["matches"][2]["unmatched_right_event_ids"],
        }
        block["integer_counts_or_null"] = matched["counts"]
        block["density_contrasts_events_per_min_or_null"] = {
            "Delta_original_float64_hex": fhex(f64(60.0 * (matched["counts"]["N_intervention"] - matched["counts"]["N_sham"]) / 160.0)),
            "Delta_rescaled_float64_hex": fhex(f64(60.0 * (matched["counts"]["N_rescaled"] - matched["counts"]["N_sham"]) / 160.0)),
        }
        block["F_rescale_only_or_null"] = None if matched["F_rescale_only"] is None else fhex(matched["F_rescale_only"])
        block["F_rescale_only_null_reason_or_null"] = "NO_ADDED_EVENTS" if matched["F_rescale_only"] is None else None
        block["block_rescale_only_or_null"] = matched["block_rescale_only"]
        block["status"] = "INVALID_MATCH_LEDGER" if matched["ledger_invalid"] else "VALID"
        block["reason_code_or_null"] = "INVALID_MATCH_LEDGER" if matched["ledger_invalid"] else None
        block["first_failing_field_or_null"] = "event_class_ledger_or_null" if matched["ledger_invalid"] else None
        result["canonical_block_records"].append(finalize_block_record(block))
    result["event_matching_expected"] = matching_by_type
    result["t8_boundary_expected"] = boundary_by_type
    result["expected_terminal_status"] = "VALID"
    return result


def clone_block_for_identity(prototype, context, target, condition, draw_id, seed, event_type):
    clone = json.loads(json.dumps(prototype, allow_nan=False))
    clone.pop("block_record_hash_envelope", None)
    clone["execution_context"] = context
    clone["target_id"] = target
    clone["condition_id"] = condition
    clone["parameter_draw_id"] = draw_id
    clone["seed"] = int(seed)
    clone["event_type"] = event_type
    clone["block_id"] = "RV6B::" + context + "::" + target + "::" + condition + "::" + event_type + "::" + draw_id + "::S" + str(seed)
    clone["raw_pair_key"] = "RV6P::" + context + "::" + target + "::" + condition + "::" + draw_id + "::S" + str(seed)
    if context == "LEVEL_A_WORKER":
        clone["run_id_intervention"] = "LA__" + condition + "__" + draw_id + "__S" + str(seed)
        clone["run_id_sham"] = "LA__LEVEL_A_T4_DOSE_INDEX_03__" + draw_id + "__S" + str(seed)
        clone["sham_logical_alias_or_null"] = "LEVEL_A_SHARED_SHAM::" + draw_id + "::S" + str(seed)
    elif context == "LEVEL_B_SURFACE":
        clone["run_id_intervention"] = condition + "__" + draw_id + "__S" + str(seed)
        clone["run_id_sham"] = "LA__LEVEL_A_T4_DOSE_INDEX_03__" + draw_id + "__S" + str(seed)
        clone["sham_logical_alias_or_null"] = "LEVEL_B_SHARED_SHAM::" + draw_id + "::S" + str(seed)
    else:
        clone["run_id_intervention"] = condition + "__" + draw_id + "__S" + str(seed)
        clone["run_id_sham"] = "X3__SHAM__" + draw_id + "__S" + str(seed)
        clone["sham_logical_alias_or_null"] = None
    return finalize_block_record(clone)


def missing_block_from_identity(prototype, context, target, condition, draw_id, seed, event_type, status):
    record = clone_block_for_identity(
        prototype, context, target, condition, draw_id, seed, event_type
    )
    record.pop("block_record_hash_envelope")
    null_block_tail(record)
    record["status"] = status
    record["reason_code_or_null"] = status
    record["first_failing_field_or_null"] = "registered_raw_pair"
    if status == "NOT_ESTIMABLE_INPUT":
        record["intervention_input_validity_status"] = "NOT_ESTIMABLE_INPUT"
        record["intervention_finite_status"] = "NOT_SCANNED_DUE_PRIOR_STATUS"
    return finalize_block_record(record)


def denominator_rule(context):
    if context == "LEVEL_A_WORKER":
        return {"min_complete_blocks": 30, "min_distinct_seeds": 5, "min_distinct_draws": 5}
    if context == "LEVEL_B_SURFACE":
        return {"min_complete_blocks": 10, "min_distinct_seeds": 3, "min_distinct_draws": 3}
    if context == "CROSS_TARGET_VERIFIER":
        return {"min_complete_blocks": 20, "min_distinct_seeds": 3, "min_distinct_draws": 5}
    raise ValueError("unknown context")


def pool_condition(context, target, condition, event_type, blocks):
    ordered = [row["block_id"] for row in blocks]
    require(len(ordered) == len(set(ordered)), "DUPLICATE_BLOCK_ID")
    invalid = []
    missing = []
    valid = []
    for block in blocks:
        status = block["status"]
        if status == "VALID":
            valid.append(block)
        elif status.startswith("NOT_ESTIMABLE_"):
            missing.append({"block_id": block["block_id"], "reason_code": status})
        else:
            invalid.append({"block_id": block["block_id"], "reason_code": status})
    rules = denominator_rule(context)
    distinct_seeds = len(set(row["seed"] for row in valid))
    distinct_draws = len(set(row["parameter_draw_id"] for row in valid))
    minimum_pass = (
        len(valid) >= rules["min_complete_blocks"]
        and distinct_seeds >= rules["min_distinct_seeds"]
        and distinct_draws >= rules["min_distinct_draws"]
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "canonical_condition_schema_id": CONDITION_SCHEMA_ID,
        "veto_id": VETO_ID,
        "condition_record_id": "RV6C::" + context + "::" + target + "::" + condition + "::" + event_type,
        "execution_context": context,
        "target_id": target,
        "condition_id": condition,
        "event_type": event_type,
        "ordered_block_ids": ordered,
        "valid_complete_blocks": len(valid),
        "distinct_seed_count": distinct_seeds,
        "distinct_draw_count": distinct_draws,
        "missing_block_ids_and_reasons": missing,
        "invalid_block_ids_and_reasons": invalid,
        "pooled_integer_counts_or_null": None,
        "pooled_F_rescale_only_or_null": None,
        "pooled_F_rescale_only_null_reason_or_null": None,
        "block_rescale_only_fraction_or_null": None,
        "minimum_denominator_rule": rules,
        "minimum_denominator_pass": minimum_pass,
        "status": None,
        "hard_veto_boolean_or_null": None,
        "downstream_consumer_ids": list(CONSUMERS),
    }
    if invalid:
        first_invalid_id = invalid[0]["block_id"]
        first = next(row for row in blocks if row["block_id"] == first_invalid_id)
        record["status"] = first["status"]
    elif not minimum_pass:
        record["status"] = "NOT_ESTIMABLE_INSUFFICIENT_BLOCKS"
    else:
        names = [
            "N_sham",
            "N_intervention",
            "N_rescaled",
            "N_added_original",
            "N_added_rescaled",
            "N_persistent",
            "N_rescale_only",
        ]
        pooled = {
            name: sum(row["integer_counts_or_null"][name] for row in valid)
            for name in names
        }
        record["pooled_integer_counts_or_null"] = pooled
        original = pooled["N_added_original"]
        if original == 0:
            record["pooled_F_rescale_only_or_null"] = None
            record["pooled_F_rescale_only_null_reason_or_null"] = "NO_ADDED_EVENTS"
        else:
            record["pooled_F_rescale_only_or_null"] = fhex(
                f64((original - pooled["N_persistent"]) / original)
            )
            record["pooled_F_rescale_only_null_reason_or_null"] = None
        record["block_rescale_only_fraction_or_null"] = fhex(
            f64(sum(1 for row in valid if row["block_rescale_only_or_null"]) / len(valid))
        )
        hard = (
            original > 0
            and pooled["N_added_rescaled"] == 0
            and pooled["N_persistent"] == 0
            and pooled["N_rescale_only"] == original
        )
        if hard:
            record["status"] = "HARD_VETO_RESCALE_ONLY"
            record["hard_veto_boolean_or_null"] = True
        elif original == 0:
            record["status"] = "NO_VETO_NO_ADDED_EVENTS"
            record["hard_veto_boolean_or_null"] = False
        else:
            record["status"] = "NO_VETO_PERSISTENT_OR_MIXED"
            record["hard_veto_boolean_or_null"] = False
    record["condition_record_hash_envelope"] = make_hash_envelope(
        dict(record), "condition_record_hash_envelope"
    )
    return record


def expected_identity_grid(context):
    if context == "LEVEL_A_WORKER":
        draws = ["D00_NOMINAL", "D01", "D02", "D03", "D04", "D05"]
        seeds = [11003, 22007, 33013, 44017, 55021, 66029]
    elif context == "LEVEL_B_SURFACE":
        draws = ["D00_NOMINAL", "D02", "D04"]
        seeds = [11003, 33013, 55021, 66029]
    else:
        draws = ["D00_NOMINAL", "D01", "D02", "D03", "D04", "D05"]
        seeds = [77041, 88069, 99079, 111091]
    return [(draw, seed) for draw in draws for seed in seeds]


def condition_oracles(raw_oracles):
    by_id = {row["case_id"]: row for row in raw_oracles}
    prototypes = {}
    for case_id, oracle in by_id.items():
        for record in oracle["canonical_block_records"]:
            prototypes[(case_id, record["event_type"])] = record
    outputs = []

    definitions = [
        ("POOL_LEVEL_A_ALL_ADDED_VANISH", "LEVEL_A_WORKER", "T4", "LEVEL_A_T4_DOSE_INDEX_06", "SPINDLE", "E2E_ALL_ADDED_EVENTS_VANISH", 36),
        ("POOL_LEVEL_A_NONZERO_SURVIVAL", "LEVEL_A_WORKER", "T4", "LEVEL_A_T4_DOSE_INDEX_06", "SPINDLE", "E2E_GENUINE_EVENT_PERSISTENCE", 36),
        ("POOL_ZERO_INTERVENTION_VARIANCE", "LEVEL_A_WORKER", "T4", "LEVEL_A_T4_DOSE_INDEX_06", "SPINDLE", "E2E_ZERO_INTERVENTION_VARIANCE", 36),
        ("POOL_ZERO_SHAM_VARIANCE", "LEVEL_A_WORKER", "T4", "LEVEL_A_T4_DOSE_INDEX_06", "SPINDLE", "E2E_ZERO_SHAM_VARIANCE", 36),
        ("POOL_BOTH_ZERO_VARIANCE", "LEVEL_A_WORKER", "T4", "LEVEL_A_T4_DOSE_INDEX_06", "SPINDLE", "E2E_BOTH_ZERO_VARIANCE", 36),
        ("POOL_NONFINITE_FAIL_CLOSED", "LEVEL_A_WORKER", "T4", "LEVEL_A_T4_DOSE_INDEX_06", "SPINDLE", "E2E_NONFINITE_INTERVENTION", 36),
    ]
    for oracle_id, context, target, condition, event_type, source_case, count in definitions:
        identities = expected_identity_grid(context)
        require(len(identities) >= count, "IDENTITY_GRID_SHORT")
        blocks = [
            clone_block_for_identity(
                prototypes[(source_case, event_type)],
                context,
                target,
                condition,
                draw,
                seed,
                event_type,
            )
            for draw, seed in identities[:count]
        ]
        outputs.append(
            {
                "oracle_id": oracle_id,
                "source_raw_case_id": source_case,
                "complete_block_universe": len(identities),
                "blocks": blocks,
                "expected_condition_record": pool_condition(
                    context, target, condition, event_type, blocks
                ),
            }
        )

    persistent = prototypes[("E2E_GENUINE_EVENT_PERSISTENCE", "SPINDLE")]
    level_a_identities = expected_identity_grid("LEVEL_A_WORKER")
    level_b_identities = expected_identity_grid("LEVEL_B_SURFACE")
    la_blocks = []
    for index, (draw, seed) in enumerate(level_a_identities):
        if index < 29:
            la_blocks.append(
                clone_block_for_identity(
                    persistent,
                    "LEVEL_A_WORKER",
                    "T4",
                    "LEVEL_A_T4_DOSE_INDEX_06",
                    draw,
                    seed,
                    "SPINDLE",
                )
            )
        else:
            la_blocks.append(
                missing_block_from_identity(
                    persistent,
                    "LEVEL_A_WORKER",
                    "T4",
                    "LEVEL_A_T4_DOSE_INDEX_06",
                    draw,
                    seed,
                    "SPINDLE",
                    "NOT_ESTIMABLE_INPUT",
                )
            )
    lb_blocks = []
    for index, (draw, seed) in enumerate(level_b_identities):
        if index < 9:
            lb_blocks.append(
                clone_block_for_identity(
                    persistent,
                    "LEVEL_B_SURFACE",
                    "T4",
                    "LB3_T4_R04_C02",
                    draw,
                    seed,
                    "SPINDLE",
                )
            )
        else:
            lb_blocks.append(
                missing_block_from_identity(
                    persistent,
                    "LEVEL_B_SURFACE",
                    "T4",
                    "LB3_T4_R04_C02",
                    draw,
                    seed,
                    "SPINDLE",
                    "NOT_ESTIMABLE_INPUT",
                )
            )
    la_condition = pool_condition(
        "LEVEL_A_WORKER", "T4", "LEVEL_A_T4_DOSE_INDEX_06", "SPINDLE", la_blocks
    )
    lb_condition = pool_condition(
        "LEVEL_B_SURFACE", "T4", "LB3_T4_R04_C02", "SPINDLE", lb_blocks
    )
    require(la_condition["status"] == "NOT_ESTIMABLE_INSUFFICIENT_BLOCKS", "CROSS_CONTEXT_LA_NOT_FAIL_CLOSED")
    require(lb_condition["status"] == "NOT_ESTIMABLE_INSUFFICIENT_BLOCKS", "CROSS_CONTEXT_LB_NOT_FAIL_CLOSED")
    outputs.append(
        {
            "oracle_id": "POOL_CROSS_CONTEXT_BORROWING_TRAP",
            "source_raw_case_id": "E2E_GENUINE_EVENT_PERSISTENCE",
            "level_a_blocks": la_blocks,
            "level_b_blocks": lb_blocks,
            "expected_level_a_condition_record": la_condition,
            "expected_level_b_condition_record": lb_condition,
            "forbidden_pooled_valid_block_count": 38,
            "forbidden_combined_clearance": True,
            "required_result": "BOTH_CONTEXTS_REMAIN_NOT_ESTIMABLE_NO_BORROWING",
        }
    )
    return outputs


def bootstrap_missingness_oracles():
    return [
        {
            "oracle_id": "BOOT_COMPONENT_ONE_OF_10000_MISSING",
            "point_component_status": "ELIGIBLE_FINITE",
            "replicate_array_contract": {"registered_length":10000,"finite_indices_half_open":[0,9999],"missing_indices":[9999]},
            "expected_component_status": "NOT_ESTIMABLE_INCOMPLETE_BOOTSTRAP_ARRAY",
            "numeric_interval": None,
            "skip_redraw_impute_or_shorten": "PROHIBITED",
        },
        {
            "oracle_id": "BOOT_COMPONENT_ONE_OF_10000_NONFINITE",
            "point_component_status": "ELIGIBLE_FINITE",
            "replicate_array_contract": {"registered_length":10000,"finite_indices_half_open":[0,4321],"nonfinite_indices":[4321],"finite_suffix_half_open":[4322,10000]},
            "expected_component_status": "NOT_ESTIMABLE_INCOMPLETE_BOOTSTRAP_ARRAY",
            "numeric_interval": None,
            "skip_redraw_impute_or_shorten": "PROHIBITED",
        },
        {
            "oracle_id": "BOOT_FAMILY_EMPTY_REQUIRED_M_B",
            "family_id": "F_TARGET_DECISIONS",
            "registered_replicates":10000,
            "first_empty_estimable_member_set_replicate_index":731,
            "expected_family_status":"NOT_ESTIMABLE_EMPTY_REQUIRED_M_B",
            "numeric_M_b_array":None,
            "numeric_quantile":None,
            "registered_replicate_identities_retained":10000,
            "skip_redraw_or_family_shrink":"PROHIBITED",
        },
        {
            "oracle_id": "BOOT_COMPLETE_FINITE_CONSTANT_ARRAY",
            "point_component_status":"ELIGIBLE_FINITE",
            "replicate_array_contract":{"registered_length":10000,"constant_float64_hex":"0x0.0p+0"},
            "expected_component_status":"ESTIMABLE_USE_REGISTERED_SCALE_FLOOR",
            "scale_floor_formula":"1e-12*max(1,abs(z_hat))",
        },
        {
            "oracle_id":"BOOT_T4_ZERO_ADDED_SOLE_EXCEPTION",
            "precondition":"T4 point-estimate attribution denominator passed",
            "replicate_N_added":0,
            "expected_F_auto_float64_hex":"0x0.0p+0",
            "expected_F_replica_float64_hex":"0x1.0000000000000p+0",
            "required_log":"ZERO_ADDED_BOOTSTRAP",
            "generalization_to_other_missingness":"PROHIBITED",
        },
    ]


def validate_record_schema(record, schema_fields, envelope_field):
    expected = [row["name"] for row in schema_fields]
    require(len(expected) == len(set(expected)), "SCHEMA_DUPLICATE_FIELD")
    require(set(record.keys()) == set(expected), "RECORD_FIELD_SET_MISMATCH")
    envelope = record[envelope_field]
    require(envelope["excluded_field"] == envelope_field, "ENVELOPE_EXCLUDED_FIELD")
    payload = dict(record)
    payload.pop(envelope_field)
    expected_envelope = make_hash_envelope(payload, envelope_field)
    require(envelope == expected_envelope, "RECORD_HASH_ENVELOPE_MISMATCH")


def any_match_tie(raw_oracles):
    for oracle in raw_oracles:
        matching = oracle.get("event_matching_expected")
        if matching is None:
            continue
        for event_type in ("SO", "SPINDLE"):
            for match in matching[event_type]["matches"]:
                rows = match["candidate_rows"]
                distances = [row["absolute_anchor_difference_samples"] for row in rows]
                if len(distances) != len(set(distances)):
                    return True
    return False


def any_boundary_rejection(raw_oracles, exact_boundary=None):
    for oracle in raw_oracles:
        detector = oracle.get("detector_expected")
        if detector is None:
            continue
        rejected = detector["rejected_event_candidates"]
        for role in ("SHAM", "INTERVENTION", "AFFINE"):
            for event_type in ("SO", "SPINDLE"):
                for row in rejected[role][event_type]:
                    if "STRICT_EDGE" in row["rejection_reasons"]:
                        if exact_boundary is None:
                            return True
                        values = [
                            value
                            for key, value in row.items()
                            if key.endswith("_sample") or key.endswith("_sample_exclusive")
                        ]
                        if exact_boundary in values:
                            return True
    return False


def any_t8_boundary_rejection(raw_oracles):
    for oracle in raw_oracles:
        boundary = oracle.get("t8_boundary_expected")
        if boundary is None:
            continue
        for event_type in ("SO", "SPINDLE"):
            for role in ("SHAM", "INTERVENTION", "AFFINE"):
                for row in boundary[event_type][role]:
                    if row["boundary_rejections"]:
                        return True
    return False


def validate_fixture(fixture, schema_bundle):
    required_top = schema_bundle["fixture_output_schema"]["required_top_level_fields"]
    require(set(fixture.keys()) == set(required_top), "FIXTURE_TOP_LEVEL_FIELD_SET")
    block_fields = schema_bundle["record_schemas"][BLOCK_SCHEMA_ID]["fields"]
    condition_fields = schema_bundle["record_schemas"][CONDITION_SCHEMA_ID]["fields"]
    for oracle in fixture["raw_end_to_end_oracles"]:
        for block in oracle["canonical_block_records"]:
            validate_record_schema(block, block_fields, "block_record_hash_envelope")
    for oracle in fixture["condition_pooling_oracles"]:
        for key, value in oracle.items():
            if key == "expected_condition_record" or key.startswith("expected_") and key.endswith("_condition_record"):
                validate_record_schema(
                    value, condition_fields, "condition_record_hash_envelope"
                )
        for key in ("blocks", "level_a_blocks", "level_b_blocks"):
            if key in oracle:
                for block in oracle[key]:
                    validate_record_schema(
                        block, block_fields, "block_record_hash_envelope"
                    )
    for digest, blob in fixture["array_blobs_by_sha256"].items():
        require(digest == blob["sha256"], "ARRAY_BLOB_KEY_HASH_MISMATCH")
        raw = bytes.fromhex(blob["data_hex"])
        require(len(raw) == blob["byte_count"], "ARRAY_BLOB_BYTE_COUNT")
        require(sha256_bytes(raw) == digest, "ARRAY_BLOB_SHA256")
        count = 1
        for length in blob["shape"]:
            count *= int(length)
        require(blob["byte_count"] == 8 * count, "ARRAY_BLOB_SHAPE_BYTE_COUNT")
    required_coverage = {
        "PURE_MEAN_SHIFT",
        "PURE_VARIANCE_RESCALE",
        "GENUINE_EVENT_PERSISTENCE",
        "NONZERO_EVENT_SURVIVAL",
        "ALL_ADDED_EVENTS_VANISH_VETO",
        "ZERO_INTERVENTION_VARIANCE",
        "ZERO_SHAM_VARIANCE",
        "BOTH_ZERO_VARIANCE",
        "NONFINITE_INPUT",
        "EXACT_MATCHER_TIE",
        "EXACT_THRESHOLD_EQUALITY",
        "STRICT_RETAINED_EDGE",
        "T8_WINDOW_BOUNDARY_TOUCH",
        "CROSS_WINDOW_REJECTION",
        "UNMATCHED_EVENT",
        "INSUFFICIENT_BLOCK_DENOMINATOR",
        "CROSS_CONTEXT_POOLING_TRAP",
    }
    coverage = set(fixture["coverage_matrix"]["covered_case_tags"])
    require(coverage == required_coverage, "FIXTURE_COVERAGE_SET_MISMATCH")
    by_id = {row["case_id"]: row for row in fixture["raw_end_to_end_oracles"]}
    require(by_id["E2E_ZERO_INTERVENTION_VARIANCE"]["expected_terminal_status"] == "NOT_ESTIMABLE_ZERO_INTERVENTION_VARIANCE", "ZERO_I_STATUS")
    require(by_id["E2E_ZERO_SHAM_VARIANCE"]["expected_terminal_status"] == "NOT_ESTIMABLE_ZERO_SHAM_VARIANCE", "ZERO_S_STATUS")
    require(by_id["E2E_BOTH_ZERO_VARIANCE"]["expected_terminal_status"] == "NOT_ESTIMABLE_BOTH_ZERO_VARIANCE", "BOTH_ZERO_STATUS")
    require(by_id["E2E_NONFINITE_INTERVENTION"]["expected_terminal_status"] == "INVALID_NONFINITE_INPUT", "NONFINITE_STATUS")
    persistent = by_id["E2E_GENUINE_EVENT_PERSISTENCE"]["event_matching_expected"]
    require(any(persistent[event_type]["counts"]["N_persistent"] > 0 for event_type in ("SO", "SPINDLE")), "NO_PERSISTENT_EVENT_WITNESS")
    vanished = by_id["E2E_ALL_ADDED_EVENTS_VANISH"]["event_matching_expected"]
    require(any(vanished[event_type]["block_rescale_only"] for event_type in ("SO", "SPINDLE")), "NO_ALL_ADDED_VANISH_WITNESS")
    require(any_match_tie(fixture["raw_end_to_end_oracles"]), "NO_MATCH_TIE_WITNESS")
    require(any_boundary_rejection(fixture["raw_end_to_end_oracles"]), "NO_STRICT_EDGE_REJECTION_WITNESS")
    require(any_t8_boundary_rejection(fixture["raw_end_to_end_oracles"]), "NO_T8_BOUNDARY_REJECTION_WITNESS")
    threshold = by_id["E2E_SO_THRESHOLD_EQUALITY"]["design_witness_or_null"]
    require(threshold is not None and threshold["equality_value_float64_hex"] == fhex(f64(0.10)), "NO_THRESHOLD_EQUALITY_WITNESS")
    pooling_by_id = {row["oracle_id"]: row for row in fixture["condition_pooling_oracles"]}
    trap = pooling_by_id["POOL_CROSS_CONTEXT_BORROWING_TRAP"]
    require(trap["expected_level_a_condition_record"]["status"] == "NOT_ESTIMABLE_INSUFFICIENT_BLOCKS", "LA_POOLING_TRAP_STATUS")
    require(trap["expected_level_b_condition_record"]["status"] == "NOT_ESTIMABLE_INSUFFICIENT_BLOCKS", "LB_POOLING_TRAP_STATUS")
    require(trap["forbidden_pooled_valid_block_count"] >= 30, "POOLING_TRAP_NOT_POTENT")
    fixture_without_envelope = dict(fixture)
    envelope = fixture_without_envelope.pop("fixture_payload_hash_envelope")
    require(
        envelope == make_hash_envelope(fixture_without_envelope, "fixture_payload_hash_envelope"),
        "FIXTURE_HASH_ENVELOPE_MISMATCH",
    )


def build_fixture(manifest, manifest_sha256, schema_bundle, schema_sha256, compiler_sha256):
    blob_map = {}
    raw_oracles = [compile_raw_case(case, blob_map) for case in fixed_source_cases()]
    pooling_oracles = condition_oracles(raw_oracles)
    coverage_tags = sorted(
        {
            tag
            for oracle in raw_oracles
            for tag in oracle["coverage_tags"]
        }
        | {"INSUFFICIENT_BLOCK_DENOMINATOR", "CROSS_CONTEXT_POOLING_TRAP"}
    )
    record_hashes = []
    for oracle in raw_oracles:
        for record in oracle["canonical_block_records"]:
            record_hashes.append(
                {
                    "record_id": record["block_id"],
                    "payload_sha256": record["block_record_hash_envelope"]["payload_sha256"],
                }
            )
    for oracle in pooling_oracles:
        for key, value in oracle.items():
            if key == "expected_condition_record" or key.startswith("expected_") and key.endswith("_condition_record"):
                record_hashes.append(
                    {
                        "record_id": value["condition_record_id"],
                        "payload_sha256": value["condition_record_hash_envelope"]["payload_sha256"],
                    }
                )
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": FIXTURE_ID,
        "campaign_id": CAMPAIGN_ID,
        "candidate_identity": CANDIDATE_IDENTITY,
        "scientific_repair_lineage": {
            "finding_id": "PV4-EDGE-B001",
            "protocol_route_version": "6.0.0",
            "v4_parent_freeze": {"path":"PROTOCOL/PROTOCOL_FREEZE_V4.json","sha256":"F002EE65EA450757E85C854FF9A6DB93518F39F2D1188CEEFF253D5AC2BEA996"},
            "edge_adjudication": {"path":"GOVERNANCE/V4_EDGE_SEMANTICS_ADJUDICATION.json","sha256":"227E0CF0E8865A3101A3E293758483B5ADF450AB6486D955E6BE334DAC50277A"},
            "v5_parent_freeze": {"path":"PROTOCOL/PROTOCOL_FREEZE_V5.json","sha256":"4C270020736BAA2D7C0B024019D38B7AE8364B4502407CD7069CAE26820B3DAA"},
            "v5_fail_receipt": {"path":"GOVERNANCE/PROTOCOL_FREEZE_REAUDIT_V5.json","sha256":"485E664B663BBC2E3A13A5FE05C86684B1295490FEB703EFFE77AB5CEB7EA1CA"},
            "v5_prospective_plan": {"path":"GOVERNANCE/PROTOCOL_V5_REAUDIT_PLAN.json","sha256":"F84EBEAD585F0A773A092D3912413FE8319818777262024510FB49D6594C9EC6"},
            "v5_stale_declaration": {"path":"GOVERNANCE/PROTOCOL_V5_STALE_DECLARATION.json","sha256":"3CB110644DDE677E57F67D0B4D769BA0F9C7FEB7CBC76755B721138A7C24D503"},
            "governance_blockers_repaired": ["PV5-B001","PV5-B002","PV5-B003","PV5-B004","PV5-B005"],
            "unaffected_science_changed": False,
        },
        "compiler_binding": {
            "compiler_id": manifest["compiler_id"],
            "compiler_path": manifest["compiler_path"],
            "compiler_sha256": compiler_sha256,
            "manifest_path": manifest["manifest_path"],
            "manifest_sha256": manifest_sha256,
            "record_schema_path": manifest["record_schema_path"],
            "record_schema_sha256": schema_sha256,
            "launcher_path": manifest["launcher_path"],
            "launcher_sha256": manifest["launcher_sha256"],
            "environment_versions": manifest["environment_versions"],
            "execution_release_required": True,
        },
        "production_contract_binding": {
            "veto_id": VETO_ID,
            "canonical_block_schema_id": BLOCK_SCHEMA_ID,
            "canonical_condition_schema_id": CONDITION_SCHEMA_ID,
            "candidate_ledger_schema_id": "RESCALE_EVENT_MATCH_CANDIDATE_LEDGER_V6",
            "accepted_ledger_schema_id": "RESCALE_EVENT_MATCH_ACCEPTED_LEDGER_V6",
            "class_ledger_schema_id": "RESCALE_EVENT_CLASS_LEDGER_V6",
            "consumption_ledger_schema_id": "RESCALE_EVENT_ROW_CONSUMPTION_LEDGER_V6",
            "affine_evaluation_id": AFFINE_EVALUATION_ID,
            "sample_rate_hz": FS_HZ,
            "sample_domain_half_open": [0, N_SAMPLES],
            "retained_domain_half_open": [RETAINED_START, RETAINED_STOP],
            "event_match_tolerance_samples_inclusive": MATCH_TOLERANCE,
            "no_projection_or_reduced_schema_bypass": True,
        },
        "array_blob_codec": schema_bundle["fixture_output_schema"]["array_blob_codec"],
        "array_blobs_by_sha256": {key: blob_map[key] for key in sorted(blob_map)},
        "raw_end_to_end_oracles": raw_oracles,
        "condition_pooling_oracles": pooling_oracles,
        "bootstrap_missingness_oracles": bootstrap_missingness_oracles(),
        "coverage_matrix": {
            "covered_case_tags": coverage_tags,
            "noncompensatory": True,
            "production_route_required": True,
            "external_arrays_or_hidden_generation": "NONE_AFTER_COMPILATION",
        },
        "intermediate_hash_ledger": {
            "array_blob_sha256_in_lexical_order": sorted(blob_map),
            "record_payload_hashes_in_construction_order": record_hashes,
            "lineage_direction_fixture_to_compiler": "compiler_binding",
            "lineage_direction_compiler_to_fixture": manifest["designated_output_binding"],
        },
        "comparison_contract": {
            "materialization": "Decode every array blob data_hex to exact little-endian float64 bytes and verify shape/hash before use.",
            "execution_order": ["raw_pair_validation","observation","sequential_moments","direct_affine","common_preprocessing","sham_thresholds","three_detector_reruns","official_table_deep_equality","candidate_traversal","accepted_matches","class_and_consumption_ledgers","block_records","condition_pooling","bootstrap_missingness"],
            "equality": "Exact JSON type/order/value and exact binary64-byte equality; no tolerance unless the production comparator itself defines a threshold inequality.",
            "full_row_consumption": "Every E_S,E_I,E_A source row appears exactly once in both class and consumption ledgers; all eligible candidates appear exactly once in traversal order.",
            "failure": "Any missing/extra field or case, hash mismatch, nonfinite JSON number, candidate/order difference, unconsumed row, external dependency, environment drift, or status difference emits PF_RESCALE_EVENT_VETO_FAIL before science.",
            "evidence_role": "IMPLEMENTATION_PREFLIGHT_ONLY_NOT_SCIENTIFIC_EVIDENCE",
            "scientific_use": "PROHIBITED",
        },
    }
    fixture["fixture_payload_hash_envelope"] = make_hash_envelope(
        dict(fixture), "fixture_payload_hash_envelope"
    )
    validate_fixture(fixture, schema_bundle)
    return fixture


def parse_exact_arguments(argv):
    expected_flags = ["--manifest", "--schema", "--output"]
    require(len(argv) == 7, "CLI_ARGUMENT_COUNT")
    values = {}
    for offset, flag in enumerate(expected_flags):
        index = 1 + 2 * offset
        require(argv[index] == flag, "CLI_FLAG_ORDER_" + flag)
        values[flag[2:]] = argv[index + 1]
    return values


def main(argv):
    args = parse_exact_arguments(argv)
    with open(args["manifest"], "rb") as handle:
        manifest_raw = handle.read()
    manifest_sha256 = sha256_bytes(manifest_raw)
    manifest = json.loads(manifest_raw.decode("utf-8"))
    require(manifest["schema_version"] == SCHEMA_VERSION, "MANIFEST_SCHEMA_VERSION")
    require(args["manifest"] == manifest["manifest_path"], "MANIFEST_PATH_MISMATCH")
    require(args["schema"] == manifest["record_schema_path"], "SCHEMA_PATH_MISMATCH")
    require(args["output"] == manifest["designated_output_binding"]["path"], "OUTPUT_PATH_MISMATCH")
    require(__file__ == manifest["compiler_path"], "COMPILER_PATH_MISMATCH")
    compiler_sha256 = sha256_file(__file__)
    require(compiler_sha256 == manifest["compiler_sha256"], "COMPILER_SHA256_MISMATCH")
    schema_sha256 = sha256_file(args["schema"])
    require(schema_sha256 == manifest["record_schema_sha256"], "SCHEMA_SHA256_MISMATCH")
    require(tuple(sys.version_info[:3]) == (3, 10, 20), "PYTHON_VERSION_MISMATCH")
    require(np.__version__ == "2.2.6", "NUMPY_VERSION_MISMATCH")
    require(scipy.__version__ == "1.15.2", "SCIPY_VERSION_MISMATCH")
    require(manifest["requested_concurrent_processes"] == 1, "CONCURRENCY_NOT_ONE")
    require(manifest["execution_authorized_by_manifest"] is False, "MANIFEST_SELF_AUTHORIZATION_PROHIBITED")
    with open(args["schema"], "rb") as handle:
        schema_bundle = json.loads(handle.read().decode("utf-8"))
    require(schema_bundle["schema_bundle_id"] == "RESCALE_EVENT_VETO_TYPED_SCHEMA_BUNDLE_V6", "SCHEMA_BUNDLE_ID")
    with open(args["output"], "rb") as handle:
        output_prestate = handle.read()
    require(
        sha256_bytes(output_prestate) == manifest["designated_output_binding"]["required_prestate_sha256"],
        "OUTPUT_PRESTATE_SHA256_MISMATCH",
    )
    fixture = build_fixture(
        manifest, manifest_sha256, schema_bundle, schema_sha256, compiler_sha256
    )
    output_bytes = canonical_bytes(fixture)
    with open(args["output"], "wb") as handle:
        written = handle.write(output_bytes)
        handle.flush()
    require(written == len(output_bytes), "OUTPUT_SHORT_WRITE")
    output_sha256 = sha256_file(args["output"])
    require(output_sha256 == sha256_bytes(output_bytes), "OUTPUT_POSTWRITE_SHA256")
    receipt = {
        "status": "PASS_DATA_FREE_FIXTURE_COMPILATION",
        "fixture_id": FIXTURE_ID,
        "output_path": args["output"],
        "output_byte_count": len(output_bytes),
        "output_sha256": output_sha256,
        "compiler_sha256": compiler_sha256,
        "manifest_sha256": manifest_sha256,
        "record_schema_sha256": schema_sha256,
        "scientific_execution": False,
        "model_or_repository_import": False,
    }
    sys.stdout.write(canonical_bytes(receipt).decode("ascii") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
