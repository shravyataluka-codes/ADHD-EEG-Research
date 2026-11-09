"""Deterministic, synthetic-only participant assignment for evaluation splits."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from .classifier import CLASS_ORDER
from .experiment_data import (
    ExperimentSplitRow,
    _validate_source_file_hash_consistency,
)


SPLIT_ALGORITHM_VERSION = "participant-random-v1"
AUDIT_SCHEMA_VERSION = "participant-split-audit-v2"
PartitionClassCounts: TypeAlias = Mapping[str, int]


@dataclass(frozen=True)
class SplitPartition:
    """Immutable trial and participant assignments for one partition."""

    rows: tuple[ExperimentSplitRow, ...]
    participant_ids: tuple[str, ...]
    trial_class_counts: PartitionClassCounts
    participant_class_counts: PartitionClassCounts

    @property
    def dataset_indices(self) -> tuple[int, ...]:
        return tuple(row.dataset_index for row in self.rows)

    @property
    def trial_ids(self) -> tuple[str, ...]:
        return tuple(row.trial_id for row in self.rows)


@dataclass(frozen=True)
class ParticipantSplit:
    """Reproducible three-way participant split and its provenance."""

    train: SplitPartition
    validation: SplitPartition
    test: SplitPartition
    random_seed: int
    source_snapshot_id: str
    protocol_id: str
    algorithm_version: str = SPLIT_ALGORITHM_VERSION

    def to_audit_dict(self) -> dict[str, object]:
        """Return a JSON-compatible audit record without exposing mutable state."""
        partitions: dict[str, object] = {}
        for name, partition in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            assignments = []
            for row in partition.rows:
                record = row.record
                participant_claim = record.mapping_provenance.participant
                diagnosis_claim = record.mapping_provenance.diagnosis
                assignments.append(
                    {
                        "dataset_index": row.dataset_index,
                        "trial_id": row.trial_id,
                        "participant_id": record.participant_id,
                        "diagnosis": record.diagnosis,
                        "source_file_sha256": record.source_file_sha256,
                        "locator": {
                            "source_filename": record.locator.source_filename,
                            "within_file_trial_index": (
                                record.locator.within_file_trial_index
                            ),
                        },
                        "evidence_kind": record.evidence_kind,
                        "mapping_provenance": {
                            "participant": {
                                "status": participant_claim.status,
                                "source_reference": participant_claim.source_reference,
                                "rationale": participant_claim.rationale,
                            },
                            "diagnosis": {
                                "status": diagnosis_claim.status,
                                "source_reference": diagnosis_claim.source_reference,
                                "rationale": diagnosis_claim.rationale,
                            },
                        },
                    }
                )
            partitions[name] = {
                "participant_ids": list(partition.participant_ids),
                "trial_class_counts": dict(partition.trial_class_counts),
                "participant_class_counts": dict(
                    partition.participant_class_counts
                ),
                "assignments": assignments,
            }

        return {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "split_algorithm_version": self.algorithm_version,
            "random_seed": self.random_seed,
            "source_snapshot_id": self.source_snapshot_id,
            "protocol_id": self.protocol_id,
            "partitions": partitions,
        }

    def to_audit_json(self) -> str:
        """Return deterministic, human-readable JSON for this split audit."""
        return json.dumps(
            self.to_audit_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def _validate_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validated_class_minimums(
    class_minimums: Mapping[str, int] | None,
) -> dict[str, int]:
    if class_minimums is None:
        return {}
    if not isinstance(class_minimums, Mapping):
        raise TypeError("class_minimums must be a mapping from diagnosis to count")

    validated: dict[str, int] = {}
    for diagnosis, minimum in class_minimums.items():
        if diagnosis not in CLASS_ORDER:
            raise ValueError(
                f"class_minimums contains unknown diagnosis {diagnosis!r}; "
                f"expected one of {CLASS_ORDER!r}"
            )
        if isinstance(minimum, bool) or not isinstance(minimum, int):
            raise TypeError(
                f"class_minimums[{diagnosis!r}] must be an integer"
            )
        if minimum < 0:
            raise ValueError(
                f"class_minimums[{diagnosis!r}] must be non-negative"
            )
        validated[diagnosis] = minimum
    return validated


def _make_partition(
    rows: tuple[ExperimentSplitRow, ...],
    participant_ids: tuple[str, ...],
    diagnoses_by_participant: Mapping[str, str],
) -> SplitPartition:
    trial_counts = {
        diagnosis: sum(row.record.diagnosis == diagnosis for row in rows)
        for diagnosis in CLASS_ORDER
    }
    participant_counts = {
        diagnosis: sum(
            diagnoses_by_participant[participant_id] == diagnosis
            for participant_id in participant_ids
        )
        for diagnosis in CLASS_ORDER
    }
    return SplitPartition(
        rows=rows,
        participant_ids=participant_ids,
        trial_class_counts=MappingProxyType(trial_counts),
        participant_class_counts=MappingProxyType(participant_counts),
    )


def split_participants(
    rows: Iterable[ExperimentSplitRow],
    *,
    train_participant_count: int,
    validation_participant_count: int,
    test_participant_count: int,
    random_seed: int,
    class_minimums: Mapping[str, int] | None = None,
) -> ParticipantSplit:
    """Assign all trials for each synthetic participant to one partition.

    Participants are lexicographically ordered before a local seeded shuffle;
    the requested train, validation, and test participant counts are then
    consumed in that order. Class minimums apply to distinct participants in
    each realized partition and never trigger retries or rebalancing.
    """
    try:
        immutable_rows = tuple(rows)
    except TypeError as error:
        raise TypeError("rows must be an iterable of ExperimentSplitRow values") from error
    if not immutable_rows:
        raise ValueError("rows must contain at least one ExperimentSplitRow")

    train_count = _validate_count(
        train_participant_count, "train_participant_count"
    )
    validation_count = _validate_count(
        validation_participant_count, "validation_participant_count"
    )
    test_count = _validate_count(test_participant_count, "test_participant_count")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("random_seed must be an integer")
    minimums = _validated_class_minimums(class_minimums)

    dataset_indices: set[int] = set()
    trial_ids: set[str] = set()
    locators = set()
    diagnoses_by_participant: dict[str, str] = {}
    rows_by_participant: dict[str, list[ExperimentSplitRow]] = {}
    snapshot_ids: set[str] = set()
    protocol_ids: set[str] = set()

    for row in immutable_rows:
        if not isinstance(row, ExperimentSplitRow):
            raise TypeError("rows must contain only ExperimentSplitRow values")

        if row.dataset_index in dataset_indices:
            raise ValueError(f"Duplicate dataset_index: {row.dataset_index}")
        dataset_indices.add(row.dataset_index)
        if row.trial_id in trial_ids:
            raise ValueError(f"Duplicate trial_id: {row.trial_id!r}")
        trial_ids.add(row.trial_id)

        record = row.record
        if record.locator in locators:
            raise ValueError(f"Duplicate trial locator: {record.locator!r}")
        locators.add(record.locator)

        if record.evidence_kind != "synthetic_fixture":
            raise ValueError(
                "Only records with evidence_kind 'synthetic_fixture' may be split"
            )
        provenance = record.mapping_provenance
        if (
            provenance.participant.status != "synthetic_fixture"
            or provenance.diagnosis.status != "synthetic_fixture"
        ):
            raise ValueError(
                "Synthetic participant and diagnosis mapping provenance is required"
            )

        target = row.target
        if record.diagnosis not in CLASS_ORDER or target != CLASS_ORDER.index(
            record.diagnosis
        ):
            raise ValueError(f"Invalid diagnosis/target mapping for {row.trial_id!r}")

        previous_diagnosis = diagnoses_by_participant.setdefault(
            record.participant_id, record.diagnosis
        )
        if previous_diagnosis != record.diagnosis:
            raise ValueError(
                f"Participant {record.participant_id!r} has inconsistent "
                "diagnosis/target mappings"
            )
        rows_by_participant.setdefault(record.participant_id, []).append(row)
        snapshot_ids.add(record.source_snapshot_id)
        protocol_ids.add(record.protocol_id)

    if len(snapshot_ids) != 1:
        raise ValueError("All records must use a consistent source_snapshot_id")
    if len(protocol_ids) != 1:
        raise ValueError("All records must use a consistent protocol_id")
    _validate_source_file_hash_consistency(row.record for row in immutable_rows)

    participant_count = len(diagnoses_by_participant)
    if train_count + validation_count + test_count != participant_count:
        raise ValueError(
            "train, validation, and test participant counts must sum exactly "
            f"to the {participant_count} unique participants"
        )

    participant_ids = sorted(diagnoses_by_participant)
    random.Random(random_seed).shuffle(participant_ids)
    train_end = train_count
    validation_end = train_end + validation_count
    partition_participants = (
        ("train", tuple(participant_ids[:train_end])),
        ("validation", tuple(participant_ids[train_end:validation_end])),
        ("test", tuple(participant_ids[validation_end:])),
    )

    partitions: dict[str, SplitPartition] = {}
    for name, assigned_participants in partition_participants:
        assigned_rows = tuple(
            sorted(
                (
                    row
                    for participant_id in assigned_participants
                    for row in rows_by_participant[participant_id]
                ),
                key=lambda row: row.dataset_index,
            )
        )
        partition = _make_partition(
            assigned_rows, assigned_participants, diagnoses_by_participant
        )
        for diagnosis, minimum in minimums.items():
            observed = partition.participant_class_counts[diagnosis]
            if observed < minimum:
                raise ValueError(
                    f"{name} partition has {observed} distinct participant(s) "
                    f"for class {diagnosis!r}; minimum is {minimum}"
                )
        partitions[name] = partition

    result = ParticipantSplit(
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
        random_seed=random_seed,
        source_snapshot_id=next(iter(snapshot_ids)),
        protocol_id=next(iter(protocol_ids)),
    )

    assigned_rows = result.train.rows + result.validation.rows + result.test.rows
    if (
        len(assigned_rows) != len(immutable_rows)
        or {row.dataset_index for row in assigned_rows} != dataset_indices
        or {row.trial_id for row in assigned_rows} != trial_ids
    ):
        raise RuntimeError("Split assignment failed complete trial coverage")

    participant_sets = tuple(
        set(partition.participant_ids)
        for partition in (result.train, result.validation, result.test)
    )
    if (
        participant_sets[0] & participant_sets[1]
        or participant_sets[0] & participant_sets[2]
        or participant_sets[1] & participant_sets[2]
    ):
        raise RuntimeError("Split assignment failed participant disjointness")

    return result


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "SPLIT_ALGORITHM_VERSION",
    "ParticipantSplit",
    "SplitPartition",
    "split_participants",
]
