"""Deterministic, leakage-resistant partition builder for the v2 route."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PartitionRow:
    subject_pseudonym: str
    record_id: str
    session_id: str
    role: str
    fold: int


@dataclass(frozen=True)
class N3EpochSplitRow:
    subject_pseudonym: str
    epoch_id: str
    role: str
    stable_rank: int


def stable_u64(seed: int, namespace: str, *parts: str) -> int:
    payload = "\x1f".join([str(seed), namespace, *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def build_subject_folds(subject_ids: Sequence[str], seed: int, n_folds: int) -> dict[str, int]:
    """Assign sorted pseudonyms round-robin after a stable SHA-256 ordering."""
    normalized = sorted({str(subject_id) for subject_id in subject_ids})
    if len(normalized) != len(subject_ids):
        raise ValueError("subject pseudonyms must be unique")
    if n_folds < 2 or len(normalized) < n_folds:
        raise ValueError("need at least one subject per fold")
    ordered = sorted(normalized, key=lambda item: (stable_u64(seed, "subject-fold", item), item))
    return {subject_id: index % n_folds for index, subject_id in enumerate(ordered)}


def build_record_partitions(
    records: Iterable[Mapping[str, str]],
    seed: int,
    n_folds: int,
) -> tuple[PartitionRow, ...]:
    """Create subject-grouped outer folds and session-grouped train/calibration/test roles.

    Sessions, rather than individual samples, are assigned roles.  This prevents
    samples from one session crossing a partition boundary.
    """
    rows = [
        {
            "subject_pseudonym": str(record["subject_pseudonym"]),
            "record_id": str(record["record_id"]),
            "session_id": str(record["session_id"]),
        }
        for record in records
    ]
    if not rows:
        raise ValueError("records cannot be empty")
    record_ids = [row["record_id"] for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("record identifiers must be globally unique")
    subjects = sorted({row["subject_pseudonym"] for row in rows})
    folds = build_subject_folds(subjects, seed, n_folds)

    assignments: list[PartitionRow] = []
    for subject in subjects:
        sessions = sorted({row["session_id"] for row in rows if row["subject_pseudonym"] == subject})
        if len(sessions) < 3:
            raise ValueError("each subject needs at least three sessions")
        ordered_sessions = sorted(
            sessions,
            key=lambda session: (stable_u64(seed, "session-role", subject, session), session),
        )
        calibration_session = ordered_sessions[-2]
        test_session = ordered_sessions[-1]
        for row in rows:
            if row["subject_pseudonym"] != subject:
                continue
            role = "test" if row["session_id"] == test_session else "calibration" if row["session_id"] == calibration_session else "train"
            assignments.append(
                PartitionRow(
                    subject_pseudonym=subject,
                    record_id=row["record_id"],
                    session_id=row["session_id"],
                    role=role,
                    fold=folds[subject],
                )
            )
    result = tuple(sorted(assignments, key=lambda row: (row.subject_pseudonym, row.session_id, row.record_id)))
    validate_partitions(result, n_folds)
    return result


def validate_partitions(rows: Sequence[PartitionRow], n_folds: int) -> None:
    if len({row.record_id for row in rows}) != len(rows):
        raise ValueError("a record occurs in more than one partition")
    valid_roles = {"train", "calibration", "test"}
    if any(row.role not in valid_roles for row in rows):
        raise ValueError("unknown partition role")
    by_subject: dict[str, list[PartitionRow]] = {}
    for row in rows:
        by_subject.setdefault(row.subject_pseudonym, []).append(row)
    for subject_rows in by_subject.values():
        if {row.role for row in subject_rows} != valid_roles:
            raise ValueError("every subject must have train, calibration, and test records")
        role_by_session: dict[str, set[str]] = {}
        for row in subject_rows:
            role_by_session.setdefault(row.session_id, set()).add(row.role)
        if any(len(roles) != 1 for roles in role_by_session.values()):
            raise ValueError("session leakage across roles")
        if len({row.fold for row in subject_rows}) != 1:
            raise ValueError("subject leakage across outer folds")
    if any(not (0 <= row.fold < n_folds) for row in rows):
        raise ValueError("fold out of range")


def partition_digest(rows: Sequence[PartitionRow]) -> str:
    canonical = json.dumps([asdict(row) for row in rows], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_single_subject_n3_split(
    epochs: Iterable[Mapping[str, object]],
    seed: int,
    fit_fraction: float,
) -> tuple[N3EpochSplitRow, ...]:
    """Build the frozen one-night, one-subject N3 FIT/HELDOUT interface.

    Only pseudonymous epoch metadata is accepted.  Epochs are ordered by a
    SHA-256 key and split once; no feature or outcome value participates.
    """
    rows = [
        {
            "subject_pseudonym": str(epoch["subject_pseudonym"]),
            "epoch_id": str(epoch["epoch_id"]),
            "stage": str(epoch["stage"]),
            "night": str(epoch["night"]),
        }
        for epoch in epochs
    ]
    if len(rows) < 4:
        raise ValueError("at least four N3 epochs are required for FIT/HELDOUT")
    subjects = {row["subject_pseudonym"] for row in rows}
    nights = {row["night"] for row in rows}
    epoch_ids = [row["epoch_id"] for row in rows]
    if len(subjects) != 1 or len(nights) != 1:
        raise ValueError("G6 empirical interface permits exactly one subject and one night")
    if any(row["stage"] != "N3" for row in rows):
        raise ValueError("only epochs already labeled N3 by the upstream public annotation are eligible")
    if len(epoch_ids) != len(set(epoch_ids)):
        raise ValueError("epoch identifiers must be unique")
    if not (0.25 <= float(fit_fraction) <= 0.75):
        raise ValueError("fit_fraction outside frozen safety range")

    ordered = sorted(
        rows,
        key=lambda row: (
            stable_u64(seed, "single-subject-n3-fit-heldout", row["subject_pseudonym"], row["night"], row["epoch_id"]),
            row["epoch_id"],
        ),
    )
    fit_count = int(round(len(ordered) * float(fit_fraction)))
    fit_count = min(max(fit_count, 1), len(ordered) - 1)
    result = tuple(
        sorted(
            (
                N3EpochSplitRow(
                    subject_pseudonym=row["subject_pseudonym"],
                    epoch_id=row["epoch_id"],
                    role="FIT" if rank < fit_count else "HELDOUT",
                    stable_rank=rank,
                )
                for rank, row in enumerate(ordered)
            ),
            key=lambda row: row.epoch_id,
        )
    )
    if {row.role for row in result} != {"FIT", "HELDOUT"}:
        raise ValueError("both FIT and HELDOUT must be nonempty")
    if len({row.epoch_id for row in result}) != len(result):
        raise ValueError("epoch leakage across FIT and HELDOUT")
    return result


def n3_split_digest(rows: Sequence[N3EpochSplitRow]) -> str:
    canonical = json.dumps([asdict(row) for row in rows], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
