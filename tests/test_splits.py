"""Synthetic-only tests for deterministic participant assignment."""

import json
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adhd_eeg.classifier import CLASS_ORDER
from adhd_eeg.experiment_data import (
    ExperimentSplitRow,
    ExperimentTrialRecord,
    MappingClaim,
    MappingProvenance,
    TrialLocator,
)
from adhd_eeg.splits import (
    AUDIT_SCHEMA_VERSION,
    SPLIT_ALGORITHM_VERSION,
    split_participants,
)


class ParticipantSplitTests(unittest.TestCase):
    def _record(
        self,
        participant_id: str,
        diagnosis: str,
        index: int,
        *,
        evidence_kind: str = "synthetic_fixture",
        snapshot_id: str = "synthetic-fixture:snapshot-001",
        source_file_sha256: str = "f" * 64,
        protocol_id: str = "synthetic-fixture:protocol-001",
    ) -> ExperimentTrialRecord:
        status = (
            "verified_source"
            if evidence_kind == "verified_real"
            else "synthetic_fixture"
        )
        return ExperimentTrialRecord(
            locator=TrialLocator("synthetic-fixture.mat", index),
            participant_id=participant_id,
            diagnosis=diagnosis,
            mapping_provenance=MappingProvenance(
                participant=MappingClaim(
                    status=status,
                    source_reference=f"synthetic-fixture:participant-map-{index}",
                    rationale="Fabricated test fixture only.",
                ),
                diagnosis=MappingClaim(
                    status=status,
                    source_reference=f"synthetic-fixture:diagnosis-map-{index}",
                    rationale="Fabricated test fixture only.",
                ),
            ),
            source_snapshot_id=snapshot_id,
            source_file_sha256=source_file_sha256,
            protocol_id=protocol_id,
            evidence_kind=evidence_kind,
        )

    def _rows(self) -> tuple[ExperimentSplitRow, ...]:
        participant_diagnoses = (
            ("synthetic-fixture:p1", "HC"),
            ("synthetic-fixture:p2", "HC"),
            ("synthetic-fixture:p3", "ADD"),
            ("synthetic-fixture:p4", "ADD"),
            ("synthetic-fixture:p5", "ADHD"),
            ("synthetic-fixture:p6", "ADHD"),
        )
        rows = []
        index = 0
        for participant_index, (participant_id, diagnosis) in enumerate(
            participant_diagnoses
        ):
            trial_count = 2 if participant_index in (0, 3) else 1
            for _ in range(trial_count):
                record = self._record(participant_id, diagnosis, index)
                rows.append(
                    ExperimentSplitRow(
                        dataset_index=index,
                        trial_id=f"synthetic-fixture.mat:{index}",
                        record=record,
                    )
                )
                index += 1
        return tuple(rows)

    def _split(
        self,
        rows: tuple[ExperimentSplitRow, ...],
        *,
        seed: int = 17,
        class_minimums: dict[str, int] | None = None,
        counts: tuple[int, int, int] = (2, 2, 2),
    ):
        return split_participants(
            rows,
            train_participant_count=counts[0],
            validation_participant_count=counts[1],
            test_participant_count=counts[2],
            random_seed=seed,
            class_minimums=class_minimums,
        )

    def test_same_seed_produces_identical_assignments(self) -> None:
        rows = self._rows()

        first = self._split(rows)
        second = self._split(rows)

        self.assertEqual(first, second)
        self.assertEqual(first.algorithm_version, SPLIT_ALGORITHM_VERSION)
        self.assertEqual(first.algorithm_version, "participant-random-v1")
        self.assertEqual(first.random_seed, 17)
        self.assertEqual(first.source_snapshot_id, "synthetic-fixture:snapshot-001")
        self.assertEqual(first.protocol_id, "synthetic-fixture:protocol-001")

    def test_rejects_boolean_and_non_integer_seeds(self) -> None:
        for seed in (True, 17.0, "17"):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(
                    TypeError, "random_seed must be an integer"
                ):
                    self._split(self._rows(), seed=seed)

    def test_rejects_boolean_and_non_integer_participant_counts(self) -> None:
        count_names = (
            "train_participant_count",
            "validation_participant_count",
            "test_participant_count",
        )
        for count_name in count_names:
            for value in (True, 2.0, "2"):
                with self.subTest(count_name=count_name, value=value):
                    counts = {
                        "train_participant_count": 2,
                        "validation_participant_count": 2,
                        "test_participant_count": 2,
                    }
                    counts[count_name] = value
                    with self.assertRaisesRegex(
                        TypeError, f"{count_name} must be an integer"
                    ):
                        split_participants(
                            self._rows(),
                            **counts,
                            random_seed=17,
                        )

    def test_different_seeds_produce_different_assignments(self) -> None:
        rows = self._rows()

        first = self._split(rows, seed=1)
        second = self._split(rows, seed=2)

        first_assignments = tuple(
            partition.participant_ids
            for partition in (first.train, first.validation, first.test)
        )
        second_assignments = tuple(
            partition.participant_ids
            for partition in (second.train, second.validation, second.test)
        )
        self.assertNotEqual(first_assignments, second_assignments)

    def test_result_assignments_and_counts_are_immutable(self) -> None:
        split = self._split(self._rows())

        with self.assertRaises(FrozenInstanceError):
            split.train.rows = ()
        with self.assertRaises(TypeError):
            split.train.trial_class_counts["HC"] = 100

    def test_audit_schema_and_assignment_coverage(self) -> None:
        rows = self._rows()
        split = self._split(rows)
        audit = split.to_audit_dict()

        self.assertEqual(
            set(audit),
            {
                "audit_schema_version",
                "split_algorithm_version",
                "random_seed",
                "source_snapshot_id",
                "protocol_id",
                "partitions",
            },
        )
        self.assertEqual(audit["audit_schema_version"], AUDIT_SCHEMA_VERSION)
        self.assertEqual(
            audit["split_algorithm_version"], SPLIT_ALGORITHM_VERSION
        )
        self.assertNotEqual(AUDIT_SCHEMA_VERSION, SPLIT_ALGORITHM_VERSION)
        self.assertEqual(audit["random_seed"], 17)
        self.assertEqual(
            audit["source_snapshot_id"], "synthetic-fixture:snapshot-001"
        )
        self.assertEqual(audit["protocol_id"], "synthetic-fixture:protocol-001")

        partitions = audit["partitions"]
        self.assertEqual(set(partitions), {"train", "validation", "test"})
        for name, split_partition in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        ):
            exported_partition = partitions[name]
            self.assertEqual(
                set(exported_partition),
                {
                    "participant_ids",
                    "trial_class_counts",
                    "participant_class_counts",
                    "assignments",
                },
            )
            self.assertEqual(
                exported_partition["participant_ids"],
                list(split_partition.participant_ids),
            )
            expected_assignments = []
            for assignment in exported_partition["assignments"]:
                self.assertEqual(
                    set(assignment),
                    {
                        "dataset_index",
                        "trial_id",
                        "participant_id",
                        "diagnosis",
                        "source_file_sha256",
                        "locator",
                        "evidence_kind",
                        "mapping_provenance",
                    },
                )
                self.assertEqual(
                    set(assignment["locator"]),
                    {"source_filename", "within_file_trial_index"},
                )
                self.assertEqual(
                    assignment["evidence_kind"], "synthetic_fixture"
                )
                self.assertEqual(
                    set(assignment["mapping_provenance"]),
                    {"participant", "diagnosis"},
                )
                for claim in assignment["mapping_provenance"].values():
                    self.assertEqual(
                        set(claim), {"status", "source_reference", "rationale"}
                    )
            for row in split_partition.rows:
                record = row.record
                participant_claim = record.mapping_provenance.participant
                diagnosis_claim = record.mapping_provenance.diagnosis
                expected_assignments.append(
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
            self.assertEqual(
                exported_partition["assignments"], expected_assignments
            )

    def test_audit_class_counts_match_exported_assignments(self) -> None:
        audit = self._split(self._rows()).to_audit_dict()

        for partition in audit["partitions"].values():
            assignments = partition["assignments"]
            diagnoses_by_participant = {
                assignment["participant_id"]: assignment["diagnosis"]
                for assignment in assignments
            }
            for diagnosis in CLASS_ORDER:
                self.assertEqual(
                    partition["trial_class_counts"][diagnosis],
                    sum(
                        assignment["diagnosis"] == diagnosis
                        for assignment in assignments
                    ),
                )
                self.assertEqual(
                    partition["participant_class_counts"][diagnosis],
                    sum(
                        diagnoses_by_participant[participant_id] == diagnosis
                        for participant_id in partition["participant_ids"]
                    ),
                )

    def test_audit_json_is_deterministic(self) -> None:
        rows = self._rows()
        split = self._split(rows)

        first = split.to_audit_json()
        second = self._split(tuple(reversed(rows))).to_audit_json()

        self.assertEqual(first, second)
        self.assertEqual(json.loads(first), split.to_audit_dict())

    def test_audit_export_does_not_mutate_split(self) -> None:
        split = self._split(self._rows())
        original_rows = tuple(
            row for partition in (split.train, split.validation, split.test)
            for row in partition.rows
        )
        original_participant_ids = tuple(
            partition.participant_ids
            for partition in (split.train, split.validation, split.test)
        )
        original_counts = tuple(
            (
                dict(partition.trial_class_counts),
                dict(partition.participant_class_counts),
            )
            for partition in (split.train, split.validation, split.test)
        )

        audit = split.to_audit_dict()
        audit["partitions"]["train"]["participant_ids"].append(
            "synthetic-fixture:mutation"
        )
        audit["partitions"]["train"]["trial_class_counts"]["HC"] = -1

        self.assertEqual(
            tuple(
                row for partition in (split.train, split.validation, split.test)
                for row in partition.rows
            ),
            original_rows,
        )
        self.assertEqual(
            tuple(
                partition.participant_ids
                for partition in (split.train, split.validation, split.test)
            ),
            original_participant_ids,
        )
        self.assertEqual(
            tuple(
                (
                    dict(partition.trial_class_counts),
                    dict(partition.participant_class_counts),
                )
                for partition in (split.train, split.validation, split.test)
            ),
            original_counts,
        )

    def test_assignment_is_independent_of_input_row_order(self) -> None:
        rows = self._rows()

        first = self._split(rows)
        second = self._split(tuple(reversed(rows)))

        self.assertEqual(first, second)

    def test_participant_isolation_and_complete_trial_coverage(self) -> None:
        rows = self._rows()
        split = self._split(rows)
        partitions = (split.train, split.validation, split.test)

        for partition in partitions:
            for row in partition.rows:
                self.assertIn(row.record.participant_id, partition.participant_ids)
        self.assertFalse(set(split.train.participant_ids) & set(split.validation.participant_ids))
        self.assertFalse(set(split.train.participant_ids) & set(split.test.participant_ids))
        self.assertFalse(set(split.validation.participant_ids) & set(split.test.participant_ids))
        self.assertCountEqual(
            (row.dataset_index for partition in partitions for row in partition.rows),
            (row.dataset_index for row in rows),
        )
        self.assertCountEqual(
            (row.trial_id for partition in partitions for row in partition.rows),
            (row.trial_id for row in rows),
        )
        self.assertEqual(
            sum(len(partition.participant_ids) for partition in partitions), 6
        )

    def test_partition_class_counts_include_trials_and_distinct_participants(self) -> None:
        split = self._split(self._rows())

        for partition in (split.train, split.validation, split.test):
            self.assertEqual(tuple(partition.trial_class_counts), CLASS_ORDER)
            self.assertEqual(tuple(partition.participant_class_counts), CLASS_ORDER)
            self.assertEqual(
                sum(partition.trial_class_counts.values()), len(partition.rows)
            )
            self.assertEqual(
                sum(partition.participant_class_counts.values()),
                len(partition.participant_ids),
            )
            for diagnosis in CLASS_ORDER:
                self.assertEqual(
                    partition.trial_class_counts[diagnosis],
                    sum(row.record.diagnosis == diagnosis for row in partition.rows),
                )
                self.assertEqual(
                    partition.participant_class_counts[diagnosis],
                    sum(
                        participant_diagnosis == diagnosis
                        for participant_diagnosis in {
                            row.record.participant_id: row.record.diagnosis
                            for row in partition.rows
                        }.values()
                    ),
                )

    def test_class_minimums_succeed_for_realized_assignment(self) -> None:
        rows = tuple(
            ExperimentSplitRow(
                dataset_index=index,
                trial_id=f"synthetic-fixture.mat:{index}",
                record=self._record(
                    f"synthetic-fixture:p{index}",
                    "HC" if index < 3 else "ADD",
                    index,
                ),
            )
            for index in range(6)
        )
        result = self._split(
            rows, seed=1, class_minimums={"HC": 1, "ADD": 1}
        )

        for partition in (result.train, result.validation, result.test):
            self.assertEqual(partition.participant_class_counts["HC"], 1)
            self.assertEqual(partition.participant_class_counts["ADD"], 1)

    def test_class_minimum_failure_does_not_retry_or_rebalance(self) -> None:
        rows = self._rows()
        expected = self._split(rows, seed=17)
        failing_class = next(
            diagnosis
            for diagnosis in CLASS_ORDER
            if expected.validation.participant_class_counts[diagnosis] == 0
        )
        with self.assertRaisesRegex(
            ValueError, f"validation partition.*{failing_class!r}.*minimum is 1"
        ):
            self._split(rows, seed=17, class_minimums={failing_class: 1})

        repeated = self._split(rows, seed=17)
        self.assertEqual(repeated, expected)

    def test_rejects_empty_or_invalid_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            self._split(())
        with self.assertRaisesRegex(TypeError, "only ExperimentSplitRow"):
            self._split((object(),))

    def test_rejects_duplicate_indices_trial_ids_and_locators(self) -> None:
        rows = self._rows()
        duplicate_index = ExperimentSplitRow(
            dataset_index=rows[0].dataset_index,
            trial_id="synthetic-fixture.mat:100",
            record=self._record("synthetic-fixture:other", "HC", 100),
        )
        with self.assertRaisesRegex(ValueError, "Duplicate dataset_index"):
            self._split((rows[0], duplicate_index))

        duplicate_trial_id = object.__new__(ExperimentSplitRow)
        object.__setattr__(duplicate_trial_id, "dataset_index", 100)
        object.__setattr__(duplicate_trial_id, "trial_id", rows[0].trial_id)
        object.__setattr__(
            duplicate_trial_id,
            "record",
            self._record("synthetic-fixture:other", "HC", 100),
        )
        with self.assertRaisesRegex(ValueError, "Duplicate trial_id"):
            self._split((rows[0], duplicate_trial_id))

        duplicate_locator = object.__new__(ExperimentSplitRow)
        object.__setattr__(duplicate_locator, "dataset_index", 100)
        object.__setattr__(duplicate_locator, "trial_id", "synthetic-fixture.mat:100")
        object.__setattr__(
            duplicate_locator,
            "record",
            self._record(rows[0].record.participant_id, "HC", 0),
        )
        with self.assertRaisesRegex(ValueError, "Duplicate trial locator"):
            self._split((rows[0], duplicate_locator))

    def test_rejects_verified_real_records(self) -> None:
        verified_record = self._record(
            "fixture-only:not-a-real-participant",
            "HC",
            0,
            evidence_kind="verified_real",
        )
        verified_row = ExperimentSplitRow(0, "synthetic-fixture.mat:0", verified_record)

        with self.assertRaisesRegex(ValueError, "Only records.*synthetic_fixture"):
            self._split((verified_row,))

    def test_rejects_inconsistent_provenance_snapshot_protocol_and_diagnosis(self) -> None:
        row = self._rows()[0]
        record_with_provenance = self._record(
            "synthetic-fixture:p20", "HC", 20
        )
        object.__setattr__(
            record_with_provenance,
            "mapping_provenance",
            MappingProvenance(
                participant=MappingClaim(
                    "verified_source",
                    "fixture-only:participant-reference",
                    "Fabricated rejection fixture only.",
                ),
                diagnosis=MappingClaim(
                    "synthetic_fixture",
                    "synthetic-fixture:diagnosis-reference",
                    "Fabricated rejection fixture only.",
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "mapping provenance is required"):
            self._split(
                (
                    row,
                    ExperimentSplitRow(20, "synthetic-fixture.mat:20", record_with_provenance),
                )
            )

        changed_snapshot = self._record(
            "synthetic-fixture:p20",
            "HC",
            20,
            snapshot_id="synthetic-fixture:snapshot-002",
        )
        with self.assertRaisesRegex(ValueError, "consistent source_snapshot_id"):
            self._split(
                (row, ExperimentSplitRow(20, "synthetic-fixture.mat:20", changed_snapshot))
            )

        changed_protocol = self._record(
            "synthetic-fixture:p20",
            "HC",
            20,
            protocol_id="synthetic-fixture:protocol-002",
        )
        with self.assertRaisesRegex(ValueError, "consistent protocol_id"):
            self._split(
                (row, ExperimentSplitRow(20, "synthetic-fixture.mat:20", changed_protocol))
            )

        changed_source_hash = self._record(
            "synthetic-fixture:p20",
            "HC",
            20,
            source_file_sha256="e" * 64,
        )
        with self.assertRaisesRegex(ValueError, "consistent source_file_sha256"):
            self._split(
                (
                    row,
                    ExperimentSplitRow(
                        20, "synthetic-fixture.mat:20", changed_source_hash
                    ),
                )
            )

        changed_diagnosis = self._record(
            row.record.participant_id, "ADD", 20
        )
        with self.assertRaisesRegex(ValueError, "inconsistent diagnosis/target"):
            self._split(
                (
                    row,
                    ExperimentSplitRow(20, "synthetic-fixture.mat:20", changed_diagnosis),
                )
            )

    def test_requires_positive_counts_that_sum_to_unique_participants(self) -> None:
        rows = self._rows()
        for counts, error in (
            ((0, 3, 3), "train_participant_count must be positive"),
            ((2, 2, 1), "sum exactly"),
        ):
            with self.subTest(counts=counts):
                with self.assertRaisesRegex(ValueError, error):
                    split_participants(
                        rows,
                        train_participant_count=counts[0],
                        validation_participant_count=counts[1],
                        test_participant_count=counts[2],
                        random_seed=1,
                    )

    def test_validates_class_minimums(self) -> None:
        rows = self._rows()
        for minimums, error in (
            ({"UNKNOWN": 1}, "unknown diagnosis"),
            ({"HC": -1}, "non-negative"),
            ({"HC": True}, "must be an integer"),
            (["HC"], "must be a mapping"),
        ):
            with self.subTest(minimums=minimums):
                with self.assertRaisesRegex((TypeError, ValueError), error):
                    self._split(rows, class_minimums=minimums)


if __name__ == "__main__":
    unittest.main()
