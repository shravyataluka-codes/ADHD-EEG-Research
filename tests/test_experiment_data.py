"""Synthetic-only tests for the Sprint 6 experiment-data contract."""

import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    import h5py
    import numpy as np
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if torch is not None:
    from adhd_eeg.classifier import CLASS_ORDER
    from adhd_eeg.data_access import TrialManifest
    from adhd_eeg.dataset import EEGTrialDataset
    from adhd_eeg.experiment_data import (
        AmbiguousTrialLocatorError,
        ExperimentSplitRow,
        ExperimentTrialDataset,
        ExperimentTrialRecord,
        MappingClaim,
        MappingProvenance,
        TrialLocator,
    )


@unittest.skipIf(torch is None, "Install the optional 'pytorch' extra to run Dataset tests")
class ExperimentTrialDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name)
        self._write_synthetic_eeg_dataset()

        with patch(
            "adhd_eeg.data_access._default_dataset_dir",
            side_effect=AssertionError("tests must not resolve the real dataset directory"),
        ) as default_dataset_dir:
            self.manifest = TrialManifest(
                self.dataset_dir,
                file_trial_counts=(2, 2),
                group_ranges=(("HC", 0, 2), ("ADD", 2, 4)),
            )
            self.eeg_dataset = EEGTrialDataset(self.manifest)
            default_dataset_dir.assert_not_called()

        self.snapshot_id = "synthetic-fixture:snapshot-001"
        self.protocol_id = "synthetic-fixture:protocol-001"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_synthetic_eeg_dataset(self) -> None:
        y_stim = np.ones((4, 4), dtype=np.float64)
        with h5py.File(self.dataset_dir / "y_stim.mat", "w") as metadata_file:
            metadata_file.create_dataset("y_stim", data=y_stim)

        for file_number in (1, 2):
            trials = np.empty((385, 56, 2), dtype=np.float32)
            for within_file_index in range(2):
                trials[:, :, within_file_index] = (
                    (file_number - 1) * 2 + within_file_index
                )
            with h5py.File(self.dataset_dir / f"d{file_number}.mat", "w") as eeg_file:
                eeg_file.create_dataset(
                    f"d{file_number}", data=trials, dtype=np.float32
                )

    def _claim(self, field: str) -> MappingClaim:
        return MappingClaim(
            status="synthetic_fixture",
            source_reference=f"synthetic-fixture:{field}-mapping",
            rationale="Fabricated test fixture; not real participant or diagnosis evidence.",
        )

    def _record(
        self,
        filename: str,
        trial_index: int,
        participant_id: str,
        diagnosis: str,
        **overrides: object,
    ) -> ExperimentTrialRecord:
        values: dict[str, object] = {
            "locator": TrialLocator(filename, trial_index),
            "participant_id": participant_id,
            "diagnosis": diagnosis,
            "mapping_provenance": MappingProvenance(
                participant=self._claim("participant"),
                diagnosis=self._claim("diagnosis"),
            ),
            "source_snapshot_id": self.snapshot_id,
            "source_file_sha256": self._source_file_sha256(filename),
            "protocol_id": self.protocol_id,
            "evidence_kind": "synthetic_fixture",
        }
        values.update(overrides)
        return ExperimentTrialRecord(**values)

    def _record_mapping(self) -> dict[str, object]:
        return {
            "locator": {
                "source_filename": "d1.mat",
                "within_file_trial_index": 0,
            },
            "participant_id": "synthetic-fixture:participant-001",
            "diagnosis": "HC",
            "mapping_provenance": {
                "participant": {
                    "status": "synthetic_fixture",
                    "source_reference": "synthetic-fixture:participant-mapping",
                    "rationale": "Fabricated fixture only.",
                },
                "diagnosis": {
                    "status": "synthetic_fixture",
                    "source_reference": "synthetic-fixture:diagnosis-mapping",
                    "rationale": "Fabricated fixture only.",
                },
            },
            "source_snapshot_id": self.snapshot_id,
            "source_file_sha256": self._source_file_sha256("d1.mat"),
            "protocol_id": self.protocol_id,
            "evidence_kind": "synthetic_fixture",
        }

    @staticmethod
    def _source_file_sha256(filename: str) -> str:
        return {
            "d1.mat": "a" * 64,
            "d2.mat": "b" * 64,
        }.get(filename, "c" * 64)

    def _make_dataset(
        self, records: list[ExperimentTrialRecord], *, class_order=CLASS_ORDER
    ) -> ExperimentTrialDataset:
        return ExperimentTrialDataset(
            self.eeg_dataset,
            records,
            class_order=class_order,
        )

    def test_synthetic_record_construction_and_strict_parsing(self) -> None:
        parsed = ExperimentTrialRecord.from_mapping(self._record_mapping())

        self.assertEqual(parsed.locator, TrialLocator("d1.mat", 0))
        self.assertEqual(parsed.participant_id, "synthetic-fixture:participant-001")
        self.assertEqual(parsed.diagnosis, "HC")
        self.assertEqual(parsed.evidence_kind, "synthetic_fixture")
        self.assertEqual(parsed.source_file_sha256, "a" * 64)
        self.assertEqual(parsed.mapping_provenance.participant.status, "synthetic_fixture")
        self.assertEqual(parsed.mapping_provenance.diagnosis.status, "synthetic_fixture")

    def test_parser_rejects_unknown_candidate_and_y_stim_fields(self) -> None:
        for field in (
            "unexpected",
            "candidate_participant_id",
            "candidate_diagnostic_group",
            "y_stim",
            "y_stim_target",
        ):
            with self.subTest(field=field):
                values = self._record_mapping()
                values[field] = "not accepted"
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    ExperimentTrialRecord.from_mapping(values)

    def test_parser_rejects_missing_or_empty_required_fields(self) -> None:
        for field in (
            "locator",
            "participant_id",
            "diagnosis",
            "mapping_provenance",
            "source_snapshot_id",
            "source_file_sha256",
            "protocol_id",
            "evidence_kind",
        ):
            with self.subTest(field=field):
                values = self._record_mapping()
                del values[field]
                with self.assertRaisesRegex(ValueError, "Missing"):
                    ExperimentTrialRecord.from_mapping(values)

        empty_cases = (
            ("participant_id", ""),
            ("diagnosis", ""),
            ("source_snapshot_id", ""),
            ("protocol_id", ""),
        )
        for field, empty_value in empty_cases:
            with self.subTest(field=field):
                values = self._record_mapping()
                values[field] = empty_value
                with self.assertRaises(ValueError):
                    ExperimentTrialRecord.from_mapping(values)

        for invalid_hash in ("A" * 64, "a" * 63, "g" * 64):
            with self.subTest(invalid_hash=invalid_hash):
                values = self._record_mapping()
                values["source_file_sha256"] = invalid_hash
                with self.assertRaisesRegex(ValueError, "source_file_sha256"):
                    ExperimentTrialRecord.from_mapping(values)

        values = self._record_mapping()
        values["source_file_sha256"] = 123
        with self.assertRaisesRegex(TypeError, "source_file_sha256"):
            ExperimentTrialRecord.from_mapping(values)

        values = self._record_mapping()
        values["locator"] = {"source_filename": "", "within_file_trial_index": 0}
        with self.assertRaises(ValueError):
            ExperimentTrialRecord.from_mapping(values)

        values = self._record_mapping()
        del values["participant_id"]
        values["locator"]["source_filename"] = "HC_01.mat"
        with self.assertRaisesRegex(ValueError, "Missing"):
            ExperimentTrialRecord.from_mapping(values)

        values = self._record_mapping()
        provenance = values["mapping_provenance"]
        provenance["participant"]["source_reference"] = ""
        with self.assertRaises(ValueError):
            ExperimentTrialRecord.from_mapping(values)

    def test_parser_rejects_missing_provenance_and_invalid_status(self) -> None:
        values = self._record_mapping()
        values["mapping_provenance"] = None
        with self.assertRaises(TypeError):
            ExperimentTrialRecord.from_mapping(values)

        values = self._record_mapping()
        provenance = values["mapping_provenance"]
        provenance["diagnosis"]["status"] = "candidate"
        with self.assertRaisesRegex(ValueError, "Unsupported evidence status"):
            ExperimentTrialRecord.from_mapping(values)

    def test_evidence_and_provenance_must_match_in_both_directions(self) -> None:
        from adhd_eeg.experiment_data import (
            _require_consistent_value,
            _validate_evidence_consistency,
        )

        with self.assertRaisesRegex(ValueError, "synthetic_fixture.*require"):
            _validate_evidence_consistency(
                "synthetic_fixture", ("verified_source", "verified_source")
            )

        with self.assertRaisesRegex(ValueError, "verified_real.*require"):
            _validate_evidence_consistency(
                "verified_real", ("synthetic_fixture", "synthetic_fixture")
            )
        with self.assertRaisesRegex(ValueError, "consistent evidence_kind"):
            _require_consistent_value(
                {"synthetic_fixture", "verified_real"}, "evidence_kind"
            )
        with self.assertRaisesRegex(ValueError, "consistent provenance statuses"):
            _require_consistent_value(
                {
                    ("synthetic_fixture", "synthetic_fixture"),
                    ("verified_source", "verified_source"),
                },
                "provenance statuses",
            )

    def test_invalid_diagnosis_and_class_order_are_rejected(self) -> None:
        record = self._record("d1.mat", 0, "synthetic-fixture:p1", "UNKNOWN")
        with self.assertRaisesRegex(ValueError, "Invalid diagnosis"):
            self._make_dataset([record])
        with self.assertRaisesRegex(ValueError, "Invalid diagnosis"):
            ExperimentSplitRow(0, "d1.mat:0", record).target

        record = self._record("d1.mat", 0, "synthetic-fixture:p1", "HC")
        with self.assertRaisesRegex(ValueError, "class_order"):
            self._make_dataset([record], class_order=("ADD", "HC", "ADHD"))

    def test_duplicate_sidecar_locators_are_rejected(self) -> None:
        first = self._record("d1.mat", 0, "synthetic-fixture:p1", "HC")
        duplicate = self._record("d1.mat", 0, "synthetic-fixture:p2", "ADD")

        with self.assertRaisesRegex(ValueError, "Duplicate sidecar"):
            self._make_dataset([first, duplicate])

    def test_participant_cannot_have_inconsistent_diagnoses(self) -> None:
        first = self._record("d1.mat", 0, "synthetic-fixture:p1", "HC")
        second = self._record("d1.mat", 1, "synthetic-fixture:p1", "ADD")

        with self.assertRaisesRegex(ValueError, "inconsistent diagnoses"):
            self._make_dataset([first, second])

    def test_records_must_agree_on_snapshot_and_protocol(self) -> None:
        base = self._record("d1.mat", 0, "synthetic-fixture:p1", "HC")
        changed_snapshot = self._record(
            "d1.mat",
            1,
            "synthetic-fixture:p2",
            "ADD",
            source_snapshot_id="synthetic-fixture:snapshot-002",
        )
        with self.assertRaisesRegex(ValueError, "source_snapshot_id"):
            self._make_dataset([base, changed_snapshot])

        changed_protocol = self._record(
            "d1.mat",
            1,
            "synthetic-fixture:p2",
            "ADD",
            protocol_id="synthetic-fixture:protocol-002",
        )
        with self.assertRaisesRegex(ValueError, "protocol_id"):
            self._make_dataset([base, changed_protocol])

        changed_hash = self._record(
            "d1.mat",
            1,
            "synthetic-fixture:p2",
            "ADD",
            source_file_sha256="b" * 64,
        )
        with self.assertRaisesRegex(ValueError, "consistent source_file_sha256"):
            self._make_dataset([base, changed_hash])

    def test_sidecar_lookup_is_exact_and_independent_of_record_order(self) -> None:
        records = [
            self._record("d2.mat", 1, "synthetic-fixture:p3", "ADHD"),
            self._record("d2.mat", 0, "synthetic-fixture:p2", "ADD"),
            self._record("d1.mat", 1, "synthetic-fixture:p1", "HC"),
            self._record("d1.mat", 0, "synthetic-fixture:p1", "HC"),
        ]
        experiment_dataset = self._make_dataset(records)

        signal, target, metadata = experiment_dataset[2]
        self.assertEqual(target, 1)
        self.assertTrue(torch.equal(signal, torch.full((385, 56), 2.0)))
        self.assertEqual(metadata["source_filename"], "d2.mat")
        self.assertEqual(metadata["within_file_trial_index"], 0)
        self.assertEqual(metadata["participant_id"], "synthetic-fixture:p2")

    def test_split_rows_have_exact_coverage_and_dataset_index_order(self) -> None:
        records = [
            self._record("d2.mat", 1, "synthetic-fixture:p3", "ADHD"),
            self._record("d2.mat", 0, "synthetic-fixture:p2", "ADD"),
            self._record("d1.mat", 1, "synthetic-fixture:p1", "HC"),
            self._record("d1.mat", 0, "synthetic-fixture:p1", "HC"),
        ]
        experiment_dataset = self._make_dataset(records)

        rows = experiment_dataset.split_rows()

        self.assertEqual(
            [row.dataset_index for row in rows],
            list(range(len(self.eeg_dataset))),
        )
        self.assertEqual(
            [row.trial_id for row in rows],
            ["d1.mat:0", "d1.mat:1", "d2.mat:0", "d2.mat:1"],
        )
        self.assertEqual(
            [row.record for row in rows],
            [
                next(record for record in records if record.locator == TrialLocator("d1.mat", 0)),
                next(record for record in records if record.locator == TrialLocator("d1.mat", 1)),
                next(record for record in records if record.locator == TrialLocator("d2.mat", 0)),
                next(record for record in records if record.locator == TrialLocator("d2.mat", 1)),
            ],
        )
        self.assertEqual(
            [row.target for row in rows],
            [CLASS_ORDER.index(diagnosis) for diagnosis in ("HC", "HC", "ADD", "ADHD")],
        )

    def test_split_rows_reject_missing_and_extra_records(self) -> None:
        incomplete = self._make_dataset(
            [self._record("d1.mat", 0, "synthetic-fixture:p1", "HC")]
        )
        with self.assertRaisesRegex(ValueError, "Missing experiment metadata"):
            incomplete.split_rows()

        complete_records = [
            self._record("d1.mat", 0, "synthetic-fixture:p1", "HC"),
            self._record("d1.mat", 1, "synthetic-fixture:p1", "HC"),
            self._record("d2.mat", 0, "synthetic-fixture:p2", "ADD"),
            self._record("d2.mat", 1, "synthetic-fixture:p3", "ADHD"),
            self._record("d3.mat", 0, "synthetic-fixture:p4", "HC"),
        ]
        with_extra = self._make_dataset(complete_records)
        with self.assertRaisesRegex(ValueError, "outside the base dataset"):
            with_extra.split_rows()

    def test_split_rows_do_not_load_eeg_signals(self) -> None:
        records = [
            self._record("d1.mat", 0, "synthetic-fixture:p1", "HC"),
            self._record("d1.mat", 1, "synthetic-fixture:p1", "HC"),
            self._record("d2.mat", 0, "synthetic-fixture:p2", "ADD"),
            self._record("d2.mat", 1, "synthetic-fixture:p3", "ADHD"),
        ]
        experiment_dataset = self._make_dataset(records)

        with patch.object(
            EEGTrialDataset,
            "__getitem__",
            side_effect=AssertionError("split_rows must not load EEG signals"),
        ):
            rows = experiment_dataset.split_rows()

        self.assertEqual(len(rows), len(self.eeg_dataset))

    def test_missing_sidecar_mapping_raises_on_access(self) -> None:
        experiment_dataset = self._make_dataset(
            [self._record("d1.mat", 0, "synthetic-fixture:p1", "HC")]
        )

        with self.assertRaisesRegex(KeyError, "No experiment metadata"):
            experiment_dataset[1]

    def test_duplicate_locator_in_base_dataset_raises_ambiguity_error(self) -> None:
        class DuplicateLocatorDataset(EEGTrialDataset):
            def __init__(self) -> None:
                def metadata_for(index: int) -> SimpleNamespace:
                    if index not in (0, 1):
                        raise IndexError(index)
                    return SimpleNamespace(
                        source_file="d1.mat",
                        within_file_trial_index=0,
                    )

                self._manifest = SimpleNamespace(
                    metadata_for=metadata_for
                )

            def __len__(self) -> int:
                return 2

            def __getitem__(self, index: int):
                return torch.full((385, 56), float(index)), {
                    "trial_id": "d1.mat:0",
                    "source_filename": "d1.mat",
                    "within_file_trial_index": 0,
                }

        duplicate_base = DuplicateLocatorDataset()
        record = self._record("d1.mat", 0, "synthetic-fixture:p1", "HC")
        with self.assertRaises(AmbiguousTrialLocatorError):
            ExperimentTrialDataset(duplicate_base, [record])

    def test_target_and_metadata_are_separate_and_metadata_hides_diagnosis(self) -> None:
        record = self._record("d1.mat", 0, "synthetic-fixture:p1", "ADHD")
        experiment_dataset = self._make_dataset([record])

        signal, target, metadata = experiment_dataset[0]

        self.assertEqual(target, CLASS_ORDER.index("ADHD"))
        self.assertIsInstance(signal, torch.Tensor)
        self.assertNotIn("diagnosis", metadata)
        self.assertNotIn("ADHD", metadata.values())
        self.assertEqual(metadata["diagnosis_mapping_status"], "synthetic_fixture")
        self.assertEqual(metadata["evidence_kind"], "synthetic_fixture")

    def test_eeg_dataset_passes_only_signal_tensor_to_model(self) -> None:
        class SpyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: list[torch.Tensor] = []

            def forward(self, model_input: torch.Tensor) -> torch.Tensor:
                self.inputs.append(model_input)
                return model_input

        signal, source_metadata = self.eeg_dataset[0]
        model = SpyModel()
        model_input = signal.unsqueeze(0)

        model(model_input)

        self.assertEqual(len(model.inputs), 1)
        self.assertIsInstance(model.inputs[0], torch.Tensor)
        self.assertTrue(torch.equal(model.inputs[0], model_input))
        self.assertEqual(source_metadata["source_filename"], "d1.mat")
        self.assertNotIn("participant_id", source_metadata)
        self.assertFalse(
            any(key.startswith("candidate_") for key in source_metadata)
        )

    def test_participant_provenance_details_stay_in_record_not_sample_metadata(self) -> None:
        rationale = "Synthetic participant mapping rationale."
        participant_claim = MappingClaim(
            status="synthetic_fixture",
            source_reference="synthetic-fixture:participant-reference",
            rationale=rationale,
        )
        record = self._record(
            "d1.mat",
            0,
            "synthetic-fixture:p1",
            "HC",
            mapping_provenance=MappingProvenance(
                participant=participant_claim,
                diagnosis=self._claim("diagnosis"),
            ),
        )
        experiment_dataset = self._make_dataset([record])

        _, _, metadata = experiment_dataset[0]

        self.assertEqual(metadata["participant_mapping_status"], "synthetic_fixture")
        for field in ("participant_mapping_source", "participant_mapping_rationale"):
            self.assertNotIn(field, metadata)
        self.assertNotIn(participant_claim.source_reference, metadata.values())
        self.assertNotIn(participant_claim.rationale, metadata.values())
        self.assertEqual(
            experiment_dataset._records[0].mapping_provenance.participant,
            participant_claim,
        )

    def test_records_are_immutable_and_copied_at_construction(self) -> None:
        record = self._record("d1.mat", 0, "synthetic-fixture:p1", "HC")
        records = [record]
        experiment_dataset = self._make_dataset(records)
        records.clear()

        with self.assertRaises(FrozenInstanceError):
            record.participant_id = "synthetic-fixture:changed"
        self.assertEqual(len(experiment_dataset), len(self.eeg_dataset))
        self.assertEqual(experiment_dataset[0][1], 0)

    def test_dataset_remains_picklable_for_worker_processes(self) -> None:
        record = self._record("d1.mat", 0, "synthetic-fixture:p1", "HC")
        experiment_dataset = self._make_dataset([record])

        restored = pickle.loads(pickle.dumps(experiment_dataset))

        self.assertEqual(restored[0][1], 0)

    def test_original_dataset_still_returns_exactly_two_items(self) -> None:
        sample = self.eeg_dataset[0]

        self.assertEqual(len(sample), 2)
        signal, metadata = sample
        self.assertIsInstance(signal, torch.Tensor)
        self.assertEqual(
            set(metadata),
            {"trial_id", "source_filename", "within_file_trial_index"},
        )
        self.assertNotIn("participant_id", metadata)
        self.assertNotIn("diagnosis", metadata)

    def test_fixture_does_not_resolve_default_dataset_directory(self) -> None:
        with patch(
            "adhd_eeg.data_access._default_dataset_dir",
            side_effect=AssertionError("real dataset directory must not be resolved"),
        ) as default_dataset_dir:
            self.eeg_dataset[0]
            default_dataset_dir.assert_not_called()

    def test_diagnosis_provenance_details_are_not_exposed_in_metadata(self) -> None:
        revealing_claim = MappingClaim(
            status="synthetic_fixture",
            source_reference="synthetic-fixture:ADHD-clinical-report",
            rationale="This synthetic participant has an ADHD diagnosis.",
        )

        record = self._record(
            "d1.mat",
            0,
            "synthetic-fixture:p1",
            "ADHD",
            mapping_provenance=MappingProvenance(
                participant=self._claim("participant"),
                diagnosis=revealing_claim,
            ),
        )

        experiment_dataset = self._make_dataset([record])
        _, _, metadata = experiment_dataset[0]

        self.assertNotIn("diagnosis_mapping_source", metadata)
        self.assertNotIn("diagnosis_mapping_rationale", metadata)
        self.assertNotIn("ADHD", str(metadata))

    def test_record_rejects_mismatched_evidence_and_provenance(self) -> None:
        synthetic_provenance = MappingProvenance(
            participant=self._claim("participant"),
            diagnosis=self._claim("diagnosis"),
        )

        verified_provenance = MappingProvenance(
            participant=MappingClaim(
                status="verified_source",
                source_reference="source:participant",
                rationale="Evidence reference for testing only.",
            ),
            diagnosis=MappingClaim(
                status="verified_source",
                source_reference="source:diagnosis",
                rationale="Evidence reference for testing only.",
            ),
        )

        with self.assertRaisesRegex(ValueError, "verified_real.*require"):
            ExperimentTrialRecord(
                locator=TrialLocator("d1.mat", 0),
                participant_id="synthetic-fixture:p1",
                diagnosis="HC",
                mapping_provenance=synthetic_provenance,
                source_snapshot_id=self.snapshot_id,
                source_file_sha256=self._source_file_sha256("d1.mat"),
                protocol_id=self.protocol_id,
                evidence_kind="verified_real",
            )

        with self.assertRaisesRegex(ValueError, "synthetic_fixture.*require"):
            ExperimentTrialRecord(
                locator=TrialLocator("d1.mat", 0),
                participant_id="synthetic-fixture:p1",
                diagnosis="HC",
                mapping_provenance=verified_provenance,
                source_snapshot_id=self.snapshot_id,
                source_file_sha256=self._source_file_sha256("d1.mat"),
                protocol_id=self.protocol_id,
                evidence_kind="synthetic_fixture",
            )

if __name__ == "__main__":
    unittest.main()
