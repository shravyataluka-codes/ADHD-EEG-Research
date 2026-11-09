"""Synthetic-testable, provenance-gated labels for EEG trial experiments."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from torch import Tensor
from torch.utils.data import Dataset

from .classifier import CLASS_ORDER
from .dataset import EEGTrialDataset


EvidenceStatus: TypeAlias = Literal["verified_source", "synthetic_fixture"]
EvidenceKind: TypeAlias = Literal["verified_real", "synthetic_fixture"]
Diagnosis: TypeAlias = Literal["HC", "ADD", "ADHD"]
ExperimentMetadata: TypeAlias = dict[str, str | int]
ExperimentSample: TypeAlias = tuple[Tensor, int, ExperimentMetadata]

_EVIDENCE_STATUSES = frozenset(("verified_source", "synthetic_fixture"))
_EVIDENCE_KINDS = frozenset(("verified_real", "synthetic_fixture"))
_RECORD_FIELDS = frozenset(
    (
        "locator",
        "participant_id",
        "diagnosis",
        "mapping_provenance",
        "source_snapshot_id",
        "source_file_sha256",
        "protocol_id",
        "evidence_kind",
    )
)
_LOCATOR_FIELDS = frozenset(("source_filename", "within_file_trial_index"))
_CLAIM_FIELDS = frozenset(("status", "source_reference", "rationale"))
_PROVENANCE_FIELDS = frozenset(("participant", "diagnosis"))


class AmbiguousTrialLocatorError(ValueError):
    """Raised when multiple base-dataset indices identify the same trial."""


def _require_non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a non-empty, trimmed string")
    return value


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a 64-character lowercase SHA-256 digest")
    return value


def _require_exact_fields(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    keys = set(value)
    unsupported = keys - expected
    if unsupported:
        names = ", ".join(sorted(repr(key) for key in unsupported))
        raise ValueError(f"Unsupported {context} field(s): {names}")
    missing = expected - keys
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Missing {context} field(s): {names}")


def _validate_evidence_consistency(
    evidence_kind: str, provenance_statuses: tuple[str, str]
) -> None:
    if evidence_kind not in _EVIDENCE_KINDS:
        raise ValueError(f"Unsupported evidence kind: {evidence_kind!r}")
    required_status = (
        "synthetic_fixture"
        if evidence_kind == "synthetic_fixture"
        else "verified_source"
    )
    if provenance_statuses != (required_status, required_status):
        raise ValueError(
            f"{evidence_kind!r} records require both mapping claims "
            f"to use {required_status!r} provenance"
        )


def _require_consistent_value(values: set[object], field: str) -> None:
    if len(values) != 1:
        raise ValueError(f"All records must use a consistent {field}")


@dataclass(frozen=True)
class TrialLocator:
    """Exact source-local identity for one stored EEG trial."""

    source_filename: str
    within_file_trial_index: int

    def __post_init__(self) -> None:
        source_filename = _require_non_empty_string(
            self.source_filename, "source_filename"
        )
        if (
            Path(source_filename).name != source_filename
            or "/" in source_filename
            or "\\" in source_filename
        ):
            raise ValueError("source_filename must be a basename")
        if isinstance(self.within_file_trial_index, bool) or not isinstance(
            self.within_file_trial_index, int
        ):
            raise TypeError("within_file_trial_index must be an integer")
        if self.within_file_trial_index < 0:
            raise ValueError("within_file_trial_index must be non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TrialLocator:
        if not isinstance(value, Mapping):
            raise TypeError("locator must be a TrialLocator or mapping")
        _require_exact_fields(value, _LOCATOR_FIELDS, "locator")
        return cls(
            source_filename=value["source_filename"],
            within_file_trial_index=value["within_file_trial_index"],
        )


@dataclass(frozen=True)
class MappingClaim:
    """Evidence citation for one participant or diagnosis mapping."""

    status: EvidenceStatus
    source_reference: str
    rationale: str

    def __post_init__(self) -> None:
        if self.status not in _EVIDENCE_STATUSES:
            raise ValueError(f"Unsupported evidence status: {self.status!r}")
        _require_non_empty_string(self.source_reference, "source_reference")
        _require_non_empty_string(self.rationale, "rationale")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> MappingClaim:
        if not isinstance(value, Mapping):
            raise TypeError("mapping claim must be a MappingClaim or mapping")
        _require_exact_fields(value, _CLAIM_FIELDS, "mapping claim")
        return cls(
            status=value["status"],
            source_reference=value["source_reference"],
            rationale=value["rationale"],
        )


@dataclass(frozen=True)
class MappingProvenance:
    """Independent provenance claims for participant identity and diagnosis."""

    participant: MappingClaim
    diagnosis: MappingClaim

    def __post_init__(self) -> None:
        if not isinstance(self.participant, MappingClaim):
            raise TypeError("participant provenance must be a MappingClaim")
        if not isinstance(self.diagnosis, MappingClaim):
            raise TypeError("diagnosis provenance must be a MappingClaim")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> MappingProvenance:
        if not isinstance(value, Mapping):
            raise TypeError("mapping_provenance must be a MappingProvenance or mapping")
        _require_exact_fields(value, _PROVENANCE_FIELDS, "mapping provenance")
        participant = value["participant"]
        diagnosis = value["diagnosis"]
        return cls(
            participant=(
                participant
                if isinstance(participant, MappingClaim)
                else MappingClaim.from_mapping(participant)
            ),
            diagnosis=(
                diagnosis
                if isinstance(diagnosis, MappingClaim)
                else MappingClaim.from_mapping(diagnosis)
            ),
        )


@dataclass(frozen=True)
class ExperimentTrialRecord:
    """One explicit trial-to-participant/diagnosis sidecar record."""

    locator: TrialLocator
    participant_id: str
    diagnosis: str
    mapping_provenance: MappingProvenance
    source_snapshot_id: str
    source_file_sha256: str
    protocol_id: str
    evidence_kind: EvidenceKind

    def __post_init__(self) -> None:
        if not isinstance(self.locator, TrialLocator):
            raise TypeError("locator must be a TrialLocator")
        _require_non_empty_string(self.participant_id, "participant_id")
        _require_non_empty_string(self.diagnosis, "diagnosis")
        if not isinstance(self.mapping_provenance, MappingProvenance):
            raise TypeError("mapping_provenance must be a MappingProvenance")
        _require_non_empty_string(self.source_snapshot_id, "source_snapshot_id")
        _require_sha256(self.source_file_sha256, "source_file_sha256")
        _require_non_empty_string(self.protocol_id, "protocol_id")
        statuses = (
            self.mapping_provenance.participant.status,
            self.mapping_provenance.diagnosis.status,
        )
        _validate_evidence_consistency(self.evidence_kind, statuses)
        if self.evidence_kind == "synthetic_fixture":
            synthetic_values = (
                ("participant_id", self.participant_id),
                ("source_snapshot_id", self.source_snapshot_id),
                ("protocol_id", self.protocol_id),
                (
                    "participant mapping source_reference",
                    self.mapping_provenance.participant.source_reference,
                ),
                (
                    "diagnosis mapping source_reference",
                    self.mapping_provenance.diagnosis.source_reference,
                ),
            )
            for field, value in synthetic_values:
                if not value.startswith("synthetic-fixture:"):
                    raise ValueError(
                        f"Synthetic fixture {field} must be explicitly tagged"
                    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ExperimentTrialRecord:
        """Parse a strict record mapping without ignoring unsupported fields."""
        if not isinstance(value, Mapping):
            raise TypeError("experiment trial record must be a mapping")
        _require_exact_fields(value, _RECORD_FIELDS, "experiment trial record")

        locator = value["locator"]
        provenance = value["mapping_provenance"]
        return cls(
            locator=(
                locator
                if isinstance(locator, TrialLocator)
                else TrialLocator.from_mapping(locator)
            ),
            participant_id=value["participant_id"],
            diagnosis=value["diagnosis"],
            mapping_provenance=(
                provenance
                if isinstance(provenance, MappingProvenance)
                else MappingProvenance.from_mapping(provenance)
            ),
            source_snapshot_id=value["source_snapshot_id"],
            source_file_sha256=value["source_file_sha256"],
            protocol_id=value["protocol_id"],
            evidence_kind=value["evidence_kind"],
        )


def _validate_source_file_hash_consistency(
    records: Iterable[ExperimentTrialRecord],
) -> None:
    hashes_by_filename: dict[str, str] = {}
    for record in records:
        filename = record.locator.source_filename
        previous_hash = hashes_by_filename.setdefault(
            filename, record.source_file_sha256
        )
        if previous_hash != record.source_file_sha256:
            raise ValueError(
                f"Records for source file {filename!r} must use a consistent "
                "source_file_sha256"
            )


@dataclass(frozen=True)
class ExperimentSplitRow:
    """Metadata-only row for constructing participant-aware splits."""

    dataset_index: int
    trial_id: str
    record: ExperimentTrialRecord

    def __post_init__(self) -> None:
        if isinstance(self.dataset_index, bool) or not isinstance(
            self.dataset_index, int
        ):
            raise TypeError("dataset_index must be an integer")
        if self.dataset_index < 0:
            raise ValueError("dataset_index must be non-negative")
        _require_non_empty_string(self.trial_id, "trial_id")
        if not isinstance(self.record, ExperimentTrialRecord):
            raise TypeError("record must be an ExperimentTrialRecord")
        expected_trial_id = (
            f"{self.record.locator.source_filename}:"
            f"{self.record.locator.within_file_trial_index}"
        )
        if self.trial_id != expected_trial_id:
            raise ValueError("trial_id must match the record locator")

    @property
    def target(self) -> int:
        """Return the class index for the record diagnosis."""
        try:
            return CLASS_ORDER.index(self.record.diagnosis)
        except ValueError as error:
            raise ValueError(
                f"Invalid diagnosis {self.record.diagnosis!r}; "
                f"expected one of {CLASS_ORDER!r}"
            ) from error


class ExperimentTrialDataset(Dataset[ExperimentSample]):
    """Compose verified or explicitly synthetic labels with EEGTrialDataset.

    Source samples retain their existing two-item contract. This wrapper uses
    only the returned source filename and within-file index to resolve a
    sidecar record; it does not infer a participant or diagnosis.
    """

    def __init__(
        self,
        eeg_dataset: EEGTrialDataset,
        records: Iterable[ExperimentTrialRecord],
        *,
        class_order: Sequence[str] = CLASS_ORDER,
    ) -> None:
        if not isinstance(eeg_dataset, EEGTrialDataset):
            raise TypeError("eeg_dataset must be an EEGTrialDataset")
        if tuple(class_order) != CLASS_ORDER:
            raise ValueError(f"class_order must be exactly {CLASS_ORDER!r}")

        immutable_records = tuple(records)
        if not immutable_records:
            raise ValueError("records must contain at least one experiment record")

        snapshots: set[str] = set()
        protocols: set[str] = set()
        evidence_kinds: set[str] = set()
        provenance_statuses: set[tuple[str, str]] = set()
        locator_records: dict[TrialLocator, ExperimentTrialRecord] = {}
        participant_diagnoses: dict[str, str] = {}

        for record in immutable_records:
            if not isinstance(record, ExperimentTrialRecord):
                raise TypeError("records must contain ExperimentTrialRecord values")
            if record.diagnosis not in CLASS_ORDER:
                raise ValueError(
                    f"Invalid diagnosis {record.diagnosis!r}; expected one of {CLASS_ORDER!r}"
                )
            if record.locator in locator_records:
                raise ValueError(f"Duplicate sidecar trial locator: {record.locator!r}")
            locator_records[record.locator] = record

            previous_diagnosis = participant_diagnoses.setdefault(
                record.participant_id, record.diagnosis
            )
            if previous_diagnosis != record.diagnosis:
                raise ValueError(
                    f"Participant {record.participant_id!r} has inconsistent diagnoses"
                )

            snapshots.add(record.source_snapshot_id)
            protocols.add(record.protocol_id)
            evidence_kinds.add(record.evidence_kind)
            provenance_statuses.add(
                (
                    record.mapping_provenance.participant.status,
                    record.mapping_provenance.diagnosis.status,
                )
            )

        _require_consistent_value(snapshots, "source_snapshot_id")
        _require_consistent_value(protocols, "protocol_id")
        _require_consistent_value(evidence_kinds, "evidence_kind")
        _require_consistent_value(provenance_statuses, "provenance statuses")
        _validate_source_file_hash_consistency(immutable_records)

        base_locator_indices: dict[TrialLocator, int] = {}
        manifest = getattr(eeg_dataset, "_manifest", None)
        metadata_for = getattr(manifest, "metadata_for", None)
        if not callable(metadata_for):
            raise TypeError(
                "EEGTrialDataset must expose its existing metadata-only manifest"
            )
        for index in range(len(eeg_dataset)):
            source_row = metadata_for(index)
            locator = TrialLocator(
                source_filename=source_row.source_file,
                within_file_trial_index=source_row.within_file_trial_index,
            )
            previous_index = base_locator_indices.get(locator)
            if previous_index is not None:
                raise AmbiguousTrialLocatorError(
                    f"Base-dataset indices {previous_index} and {index} share "
                    f"the locator {locator!r}"
                )
            base_locator_indices[locator] = index

        self._eeg_dataset = eeg_dataset
        self._records = immutable_records
        self._records_by_locator = dict(locator_records)
        self._base_locator_indices = dict(base_locator_indices)
        self._targets_by_diagnosis = {
            diagnosis: index for index, diagnosis in enumerate(CLASS_ORDER)
        }

    def __len__(self) -> int:
        return len(self._eeg_dataset)

    def split_rows(self) -> tuple[ExperimentSplitRow, ...]:
        """Return exact-coverage split metadata without reading EEG samples."""
        base_locators = set(self._base_locator_indices)
        record_locators = set(self._records_by_locator)
        missing_locators = base_locators - record_locators
        if missing_locators:
            ordered_missing = sorted(
                missing_locators,
                key=lambda locator: (
                    locator.source_filename,
                    locator.within_file_trial_index,
                ),
            )
            raise ValueError(
                "Missing experiment metadata for base-dataset locator(s): "
                f"{ordered_missing!r}"
            )

        extra_locators = record_locators - base_locators
        if extra_locators:
            ordered_extra = sorted(
                extra_locators,
                key=lambda locator: (
                    locator.source_filename,
                    locator.within_file_trial_index,
                ),
            )
            raise ValueError(
                "Experiment metadata contains locator(s) outside the base dataset: "
                f"{ordered_extra!r}"
            )

        locators_by_index = {
            index: locator for locator, index in self._base_locator_indices.items()
        }
        rows: list[ExperimentSplitRow] = []
        for index in range(len(self._eeg_dataset)):
            locator = locators_by_index[index]
            record = self._records_by_locator[locator]
            row = ExperimentSplitRow(
                dataset_index=index,
                trial_id=f"{locator.source_filename}:{locator.within_file_trial_index}",
                record=record,
            )
            _ = row.target
            rows.append(row)
        return tuple(rows)

    def __getitem__(self, index: int) -> ExperimentSample:
        signal, source_metadata = self._eeg_dataset[index]
        if not isinstance(source_metadata, Mapping):
            raise TypeError("EEGTrialDataset source metadata must be a mapping")
        try:
            locator = TrialLocator(
                source_filename=source_metadata["source_filename"],
                within_file_trial_index=source_metadata["within_file_trial_index"],
            )
        except KeyError as error:
            raise ValueError(
                "EEGTrialDataset source metadata is missing an exact locator field"
            ) from error
        expected_locator_index = self._base_locator_indices.get(locator)
        if expected_locator_index != index:
            raise ValueError(
                "EEGTrialDataset returned source metadata inconsistent with its manifest"
            )

        if "trial_id" in source_metadata:
            expected_trial_id = (
                f"{locator.source_filename}:{locator.within_file_trial_index}"
            )
            if source_metadata["trial_id"] != expected_trial_id:
                raise ValueError("EEGTrialDataset trial_id disagrees with its locator")

        try:
            record = self._records_by_locator[locator]
        except KeyError as error:
            raise KeyError(
                f"No experiment metadata for source-local trial {locator!r}"
            ) from error

        provenance = record.mapping_provenance
        metadata: ExperimentMetadata = {
            "trial_id": f"{locator.source_filename}:{locator.within_file_trial_index}",
            "source_filename": locator.source_filename,
            "within_file_trial_index": locator.within_file_trial_index,
            "participant_id": record.participant_id,
            "participant_mapping_status": provenance.participant.status,
            "diagnosis_mapping_status": provenance.diagnosis.status,
            "source_snapshot_id": record.source_snapshot_id,
            "source_file_sha256": record.source_file_sha256,
            "protocol_id": record.protocol_id,
            "evidence_kind": record.evidence_kind,
        }
        return signal, self._targets_by_diagnosis[record.diagnosis], metadata


__all__ = [
    "AmbiguousTrialLocatorError",
    "Diagnosis",
    "EvidenceKind",
    "EvidenceStatus",
    "ExperimentMetadata",
    "ExperimentSample",
    "ExperimentSplitRow",
    "ExperimentTrialDataset",
    "ExperimentTrialRecord",
    "MappingClaim",
    "MappingProvenance",
    "TrialLocator",
]
