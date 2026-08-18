"""T4 context-domain validation, event matching, fixture oracle, and pooling."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .contracts import ContractBundle, ContractError


FORMAL_ROW_ID = "T4_LA_WORKER_D1P50_FC13"
ROW_CONTRACT_ID = "T4_ATTRIBUTION_ROW_CONTRACT_V4"


def _event_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, Mapping):
        source = event
    elif hasattr(event, "as_dict"):
        source = event.as_dict()
    else:
        source = event.__dict__
    return {
        "event_id": str(source["event_id"]),
        "onset_sample": int(source["onset_sample"]),
        "offset_sample": int(source["offset_sample"]),
        "peak_sample": int(source["peak_sample"]),
    }


def _packet_dict(packet: Any) -> dict[str, Any]:
    if isinstance(packet, Mapping):
        source = packet
    elif hasattr(packet, "as_dict"):
        source = packet.as_dict()
    else:
        source = packet.__dict__
    return {
        "packet_id": str(source["packet_id"]),
        "packet_start_sample": int(source["packet_start_sample"]),
        "packet_stop_sample": int(source["packet_stop_sample"]),
    }


def _crossing_sample(crossing: Any) -> int:
    if isinstance(crossing, Mapping):
        return int(crossing["sample"])
    return int(crossing)


def _validate_unique_ids(rows: Sequence[Mapping[str, Any]], id_name: str) -> None:
    ids = [row[id_name] for row in rows]
    if len(ids) != len(set(ids)):
        raise ContractError(f"duplicate {id_name} in T4 attribution input")


def classify_block(
    sham_events: Iterable[Any],
    intervention_events: Iterable[Any],
    packets: Iterable[Any],
    sigma_upward_crossings: Iterable[Any],
) -> dict[str, Any]:
    sham = sorted((_event_dict(e) for e in sham_events), key=lambda e: (e["onset_sample"], e["offset_sample"], e["event_id"]))
    intervention = sorted((_event_dict(e) for e in intervention_events), key=lambda e: (e["onset_sample"], e["offset_sample"], e["event_id"]))
    packet_rows = sorted((_packet_dict(p) for p in packets), key=lambda p: (p["packet_start_sample"], p["packet_stop_sample"], p["packet_id"]))
    crossing_samples = sorted(_crossing_sample(z) for z in sigma_upward_crossings)
    _validate_unique_ids(sham, "event_id")
    _validate_unique_ids(intervention, "event_id")
    _validate_unique_ids(packet_rows, "packet_id")
    if len(crossing_samples) != len(set(crossing_samples)):
        raise ContractError("duplicate T4 sigma crossing sample")

    candidates: list[dict[str, Any]] = []
    lookup_i = {row["event_id"]: row for row in intervention}
    lookup_s = {row["event_id"]: row for row in sham}
    for i_event in intervention:
        for s_event in sham:
            difference = abs(i_event["peak_sample"] - s_event["peak_sample"])
            if difference <= 50:
                candidates.append({
                    "intervention_event_id": i_event["event_id"],
                    "sham_event_id": s_event["event_id"],
                    "absolute_peak_difference": difference,
                })
    candidates.sort(key=lambda row: (
        row["absolute_peak_difference"],
        lookup_i[row["intervention_event_id"]]["peak_sample"],
        lookup_s[row["sham_event_id"]]["peak_sample"],
        row["intervention_event_id"],
        row["sham_event_id"],
    ))
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    matched_i: set[str] = set()
    matched_s: set[str] = set()
    for row in candidates:
        i_used = row["intervention_event_id"] in matched_i
        s_used = row["sham_event_id"] in matched_s
        if not i_used and not s_used:
            accepted.append(dict(row))
            matched_i.add(row["intervention_event_id"])
            matched_s.add(row["sham_event_id"])
        else:
            reason = "BOTH_ALREADY_MATCHED" if i_used and s_used else "INTERVENTION_ALREADY_MATCHED" if i_used else "SHAM_ALREADY_MATCHED"
            rejected.append({
                "intervention_event_id": row["intervention_event_id"],
                "sham_event_id": row["sham_event_id"],
                "reason": reason,
            })
    paired_ids = [row["event_id"] for row in intervention if row["event_id"] in matched_i]
    added_ids = [row["event_id"] for row in intervention if row["event_id"] not in matched_i]
    lost_ids = [row["event_id"] for row in sham if row["event_id"] not in matched_s]

    classification: list[dict[str, Any]] = []
    for event_id in added_ids:
        event = lookup_i[event_id]
        qualifying = [
            packet for packet in packet_rows
            if packet["packet_start_sample"] <= event["peak_sample"] < packet["packet_stop_sample"] + 50
        ]
        qualifying.sort(key=lambda packet: (
            abs(event["peak_sample"] - packet["packet_stop_sample"]),
            packet["packet_start_sample"],
            packet["packet_id"],
        ))
        row: dict[str, Any] = {
            "event_id": event_id,
            "selected_packet_id": None,
            "qualifying_packet_ids": [packet["packet_id"] for packet in qualifying],
            "post_packet_successive_crossing_pairs_considered": [],
        }
        if not qualifying:
            row["class"] = "UNASSOCIATED_ADDED"
            classification.append(row)
            continue
        selected = qualifying[0]
        row["selected_packet_id"] = selected["packet_id"]
        if len(qualifying) >= 2:
            distances = [abs(event["peak_sample"] - packet["packet_stop_sample"]) for packet in qualifying]
            if distances[0] == distances[1]:
                row["packet_distance_tie_samples"] = distances[0]
        inside = [z for z in crossing_samples if selected["packet_stop_sample"] < z < event["offset_sample"]]
        pairs = [[a, b] for a, b in zip(inside[:-1], inside[1:])]
        row["post_packet_successive_crossing_pairs_considered"] = pairs
        qualifying_cycle = next((pair for pair in pairs if 13 <= pair[1] - pair[0] <= 19), None)
        if qualifying_cycle is None:
            row["class"] = "PACKET_REPLICA_ADDED"
        else:
            row["qualifying_autonomous_crossing_pair"] = qualifying_cycle
            row["class"] = "AUTONOMOUS_PACKET_ASSOCIATED_ADDED"
        classification.append(row)

    n_auto = sum(row["class"] == "AUTONOMOUS_PACKET_ASSOCIATED_ADDED" for row in classification)
    n_replica = sum(row["class"] == "PACKET_REPLICA_ADDED" for row in classification)
    n_unassociated = sum(row["class"] == "UNASSOCIATED_ADDED" for row in classification)
    n_added = len(added_ids)
    counts = {
        "N_candidate_pairs": len(candidates),
        "N_accepted_pairs": len(accepted),
        "N_paired_existing": len(paired_ids),
        "N_added": n_added,
        "N_lost": len(lost_ids),
        "N_autonomous_packet_associated_added": n_auto,
        "N_packet_replica_added": n_replica,
        "N_unassociated_added": n_unassociated,
        "N_blocks_with_added": int(n_added > 0),
    }
    return {
        "candidate_pairs_sorted": candidates,
        "accepted_pairs_in_traversal_order": accepted,
        "rejected_candidates": rejected,
        "paired_existing_intervention_event_ids": paired_ids,
        "added_intervention_event_ids": added_ids,
        "lost_sham_event_ids": lost_ids,
        "added_event_classification": classification,
        "counts": counts,
        "F_auto": {"numerator": n_auto, "denominator": n_added},
        "F_replica": {"numerator": n_replica, "denominator": n_added},
        "scientific_denominator_status": "ESTIMABLE" if n_added >= 30 else "NOT_ESTIMABLE",
    }


def _deep_diff(expected: Any, observed: Any, path: str = "expected") -> list[dict[str, Any]]:
    if type(expected) is not type(observed):
        return [{"path": path, "expected": expected, "observed": observed, "reason": "TYPE_MISMATCH"}]
    if isinstance(expected, dict):
        diffs: list[dict[str, Any]] = []
        if list(expected) != list(observed):
            diffs.append({"path": path, "expected_keys": list(expected), "observed_keys": list(observed), "reason": "KEY_OR_ORDER_MISMATCH"})
        for key in expected:
            if key in observed:
                diffs.extend(_deep_diff(expected[key], observed[key], f"{path}.{key}"))
        return diffs
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return [{"path": path, "expected_length": len(expected), "observed_length": len(observed), "reason": "LENGTH_MISMATCH"}]
        diffs = []
        for idx, (left, right) in enumerate(zip(expected, observed)):
            diffs.extend(_deep_diff(left, right, f"{path}[{idx}]"))
        return diffs
    if expected != observed:
        return [{"path": path, "expected": expected, "observed": observed, "reason": "VALUE_MISMATCH"}]
    return []


def compare_fixture(bundle: ContractBundle) -> dict[str, Any]:
    fixture = bundle.fixture
    if fixture["fixture_id"] != "PF_T4_EVENT_ATTRIBUTION_FIXTURE_V3":
        raise ContractError("wrong T4 fixture identity")
    if len(fixture["blocks"]) != 1:
        raise ContractError("T4 fixture must contain exactly one block")
    block = fixture["blocks"][0]
    observed = classify_block(
        block["sham_events"],
        block["intervention_events"],
        block["packets"],
        block["sigma_upward_crossings"],
    )
    fields = fixture["comparison_contract"]["exact_fields"]
    selected_expected: dict[str, Any] = {}
    selected_observed: dict[str, Any] = {}
    for field in fields:
        prefix = "expected."
        if not field.startswith(prefix):
            raise ContractError(f"unsupported fixture comparison path {field}")
        key = field[len(prefix):]
        selected_expected[key] = deepcopy(fixture["expected"][key])
        selected_observed[key] = deepcopy(observed[key])
    diffs = _deep_diff(selected_expected, selected_observed)
    status = "PASS" if not diffs else "PF_T4_EVENT_ATTRIBUTION_FAIL"
    return {
        "test_id": "PF_T4_EVENT_ATTRIBUTION",
        "fixture_id": fixture["fixture_id"],
        "status": status,
        "exact_fields": list(fields),
        "diffs": diffs,
        "raw_row_consumption": {
            "sham_events": len(block["sham_events"]),
            "intervention_events": len(block["intervention_events"]),
            "packets": len(block["packets"]),
            "sigma_upward_crossings": len(block["sigma_upward_crossings"]),
            "complete": True,
        },
        "evidence_role": "IMPLEMENTATION_PREFLIGHT_ONLY_NOT_SCIENTIFIC_EVIDENCE",
    }


@dataclass(frozen=True)
class AttributionDomain:
    rows: tuple[Mapping[str, Any], ...]
    by_id: Mapping[str, Mapping[str, Any]]
    by_join: Mapping[tuple[str, float, float, float], Mapping[str, Any]]

    def resolve(self, context: str, dose: float, carrier: float, repetition: float) -> Mapping[str, Any]:
        key = (context, float(dose), float(carrier), float(repetition))
        if key not in self.by_join:
            raise ContractError(f"PF_T4_ATTRIBUTION_DOMAIN_FAIL: unresolved {key}")
        return self.by_join[key]


def validate_domain(bundle: ContractBundle) -> AttributionDomain:
    section = bundle.protocol["observation_and_detector_contract"]["T4_added_event_attribution"]
    rows = tuple(section["attribution_domain_table"])
    if len(rows) != 16:
        raise ContractError("PF_T4_ATTRIBUTION_DOMAIN_FAIL: row count")
    by_id: dict[str, Mapping[str, Any]] = {}
    by_join: dict[tuple[str, float, float, float], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("row_contract_id") != ROW_CONTRACT_ID:
            raise ContractError("PF_T4_ATTRIBUTION_DOMAIN_FAIL: row contract")
        row_id = row["domain_row_id"]
        join = (
            row["execution_context"],
            float(row["dose_hz_equivalent"]),
            float(row["carrier_hz"]),
            float(row["repetition_hz"]),
        )
        if row_id in by_id or join in by_join:
            raise ContractError("PF_T4_ATTRIBUTION_DOMAIN_FAIL: duplicate ID or join key")
        by_id[row_id] = row
        by_join[join] = row
    context_map = {
        "LEVEL_A_WORKER": "level_a_worker",
        "LEVEL_B_SURFACE": "level_b_policy",
        "CROSS_TARGET_VERIFIER": "cross_target_verifier_possible",
    }
    for context, candidate_key in context_map.items():
        projection = sorted(row["domain_row_id"] for row in rows if row["execution_context"] == context)
        expected = sorted(section["candidate_domain_rows"][candidate_key])
        if projection != expected:
            raise ContractError(f"PF_T4_ATTRIBUTION_DOMAIN_FAIL: context projection {context}")
    formal = [row["domain_row_id"] for row in rows if row["formal_target_decision_member"]]
    if formal != [FORMAL_ROW_ID]:
        raise ContractError("PF_T4_ATTRIBUTION_DOMAIN_FAIL: formal row")
    return AttributionDomain(rows, by_id, by_join)


def pool_row(
    row: Mapping[str, Any],
    block_results: Sequence[Mapping[str, Any]],
    *,
    bootstrap: bool = False,
    family_id: str | None = None,
    replicate_index: int | None = None,
) -> dict[str, Any]:
    valid = [block for block in block_results if block.get("valid_complete_pair") is True]
    seeds = {int(block["seed"]) for block in valid}
    draws = {str(block["parameter_draw_id"]) for block in valid}
    n_added = sum(int(block["classification"]["counts"]["N_added"]) for block in valid)
    n_auto = sum(int(block["classification"]["counts"]["N_autonomous_packet_associated_added"]) for block in valid)
    n_replica = sum(int(block["classification"]["counts"]["N_packet_replica_added"]) for block in valid)
    contributing = sum(int(block["classification"]["counts"]["N_added"]) > 0 for block in valid)
    point_denominator = (
        len(valid) >= int(row["minimum_complete_paired_blocks"])
        and len(seeds) >= int(row["minimum_distinct_seeds"])
        and len(draws) >= int(row["minimum_distinct_draws"])
        and n_added >= int(row["N_added_minimum"])
        and contributing >= int(row["minimum_contributing_blocks"])
    )
    zero_record = None
    if bootstrap and n_added == 0:
        f_auto, f_replica = 0.0, 1.0
        zero_record = {
            "code": "ZERO_ADDED_BOOTSTRAP",
            "domain_row_id": row["domain_row_id"],
            "family_id": family_id,
            "replicate_index": replicate_index,
        }
        status = "BOOTSTRAP_ZERO_ADDED"
    elif bootstrap:
        f_auto = n_auto / n_added
        f_replica = n_replica / n_added
        status = "BOOTSTRAP_ESTIMABLE"
    elif not point_denominator:
        f_auto = f_replica = None
        status = "NOT_ESTIMABLE"
    else:
        f_auto = n_auto / n_added
        f_replica = n_replica / n_added
        status = "ESTIMABLE"
    return {
        "domain_row_id": row["domain_row_id"],
        "execution_context": row["execution_context"],
        "complete_paired_blocks": len(valid),
        "distinct_seeds": len(seeds),
        "distinct_draws": len(draws),
        "N_added": n_added,
        "distinct_contributing_blocks": contributing,
        "N_autonomous": n_auto,
        "N_packet_replica": n_replica,
        "F_auto": f_auto,
        "F_replica": f_replica,
        "denominator_status": status,
        "bootstrap_zero_added_record": zero_record,
    }
