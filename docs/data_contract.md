# ADHD EEG Data Contract

**Status:** Proposed for review; documentation only
**Scope:** Raw EEG sample and metadata representation for research data access
**Authority:** This contract documents observed data structure and unresolved provenance. It does not approve preprocessing, establish diagnostic labels, or authorize model training.

## 1. Principles

1. Preserve the supplied EEG representation and source files.
2. Keep raw source facts separate from interpretations and joins.
3. Store participant identity, diagnostic group, and stimulus/task information in separate fields.
4. A candidate value must never be serialized or exposed as verified.
5. Unknown or unverified metadata is nullable; do not infer a value to make a record complete.
6. Provenance and verification status apply to each mapping claim, not just to a whole record.

## 2. Evidence status vocabulary

Use these statuses for individual fields and relationships:

| Status | Meaning |
|---|---|
| `verified_source` | Directly observed in the named source, for the precise fact stated. |
| `supported_indirectly` | Consistent with structural evidence but not directly keyed or defined by the source. |
| `candidate` | A proposed interpretation or join that has not been independently confirmed. |
| `unresolved` | Evidence is insufficient to assign a value or establish a relationship. |
| `not_applicable` | No value or relationship is expected for this record. |

A value can be directly observed in metadata without its meaning or linkage to an EEG sample being verified. For example, a `y_stim` row-0 value is observed in a metadata column; associating it with a particular EEG trial requires a separate join claim.

## 3. EEG sample schema

### Observed representation

| Field | Contract |
|---|---|
| Trial identifier | Stable source locator: source filename plus zero-based within-file trial index. This identifies a stored slice, not a participant or a verified metadata join. |
| Source filename | Required basename such as `d1.mat`; the file contains the correspondingly named HDF5 dataset. |
| Within-file trial index | Required zero-based integer in the source dataset's trial axis. |
| Signal shape | Required `(385, 56)`: 385 observed sample positions by 56 anonymous channel columns. Preserve this order. |
| Data type | Required `float32`; do not silently cast. |
| Sampling frequency | `256 Hz` is reported by the cited cohort paper, not encoded in the inspected EEG HDF5 dataset. Store as a paper-reported value with provenance, not as a directly verified array property. |
| Sample count | `385`, directly observed. Do not crop to 384 or assign a physical interpretation to position 385. |
| Channel count | `56`, directly observed. Do not assign electrode identities or map columns to `chan.mat` without evidence. |

`d1.mat`–`d6.mat` have 5,000 stored trials each and `d7.mat` has 3,902. The cumulative global index produced by concatenating files numerically is a **candidate locator**, not proof that EEG trial order matches `y_stim` column order.

### Proposed Python representation

The following is a schema proposal, not implemented code. Counts and dimensions may be derived from `signal`, but if serialized explicitly they must equal the observed array properties.

```python
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

EvidenceStatus = Literal[
    "verified_source",
    "supported_indirectly",
    "candidate",
    "unresolved",
    "not_applicable",
]


@dataclass(frozen=True)
class FieldProvenance:
    field: str
    source: str
    status: EvidenceStatus
    note: str


@dataclass(frozen=True)
class EEGSample:
    trial_id: str  # e.g. "d1.mat:0"; source-local locator only
    source_filename: str
    within_file_trial_index: int
    signal: npt.NDArray[np.float32]  # exactly (385, 56), stored axis order
    sampling_frequency_hz: float | None  # 256.0 only with paper provenance
    sample_count: int  # exactly 385
    channel_count: int  # exactly 56
    provenance: tuple[FieldProvenance, ...]


@dataclass(frozen=True)
class TrialMetadata:
    trial_id: str
    metadata_column_index: int | None
    participant_source_index: int | None
    participant_id: str | None  # independently verified identity only
    candidate_participant_id: str | None
    diagnostic_group: Literal["HC", "ADD", "ADHD"] | None
    candidate_diagnostic_group: Literal["HC", "ADD", "ADHD"] | None
    stimulus_indicator: tuple[int, int, int] | None
    stimulus_task_category: str | None
    mapping_provenance: tuple[FieldProvenance, ...]
```

For any non-null participant, diagnostic, stimulus-category, or metadata-column value, `mapping_provenance` must include the source, evidence status, and rationale for that field. In a production schema, field-specific status may instead be represented directly alongside each value; it must not be collapsed into one record-level status.

## 4. Metadata schema and current values

| Metadata field | Type | Required? | Current status and rule |
|---|---|---:|---|
| `trial_id` | String | Yes | Identifies the source slice. Must agree with `source_filename` and `within_file_trial_index`; does not assert metadata alignment. |
| `metadata_column_index` | Integer or null | No | Null by default. If populated using concatenated file offsets, mark the EEG-to-`y_stim` join `candidate` until independently confirmed. |
| `participant_source_index` | Positive integer or null | No | `y_stim` row 0 is observed as integer values with three reset blocks. Its participant-index interpretation is supported indirectly. If attached to an EEG sample via the candidate join, that association remains candidate. |
| `participant_id` | String or null | No | Reserved for an identity verified against an authoritative participant key. **Null for current EEG-to-participant records.** |
| `candidate_participant_id` | String or null | No | Optional composite block/index identifier, clearly marked candidate and never presented as a verified identity. It is not a substitute for an externally keyed participant identity. |
| `diagnostic_group` | `HC`, `ADD`, `ADHD`, or null | No | **Null until the participant diagnosis and EEG join are independently verified.** Do not derive it from `y_stim` indicator rows. |
| `candidate_diagnostic_group` | `HC`, `ADD`, `ADHD`, or null | No | Optional proposed interpretation of a metadata block, with `candidate` status and explicit provenance. Must not be used as a training target. |
| `stimulus_indicator` | Three binary values or null | No | May preserve the raw three-row one-hot value for a verified metadata column. It describes source indicators only; do not rename components as clinical diagnoses or infer their physical task meaning. |
| `stimulus_task_category` | String or null | No | Null unless category semantics are independently documented. Keep separate from diagnosis. |
| `mapping_provenance` | Sequence of provenance records | Yes | Must record evidence status and source for each populated or asserted mapping. For unverified joins, record the unresolved/candidate state rather than omitting it. |

The meanings of rows 1–3 of `y_stim` are unresolved. Their binary one-hot structure and correspondence to metadata blocks are directly observed, but their physical stimulus/task names are not established by that structure alone.

## 5. Validation rules

### Required fields

- Every EEG record has a non-empty `trial_id`, `source_filename`, and `within_file_trial_index`, plus its signal.
- Every metadata record has a `trial_id` and field-level provenance, including explicit unresolved entries where a requested mapping cannot be established.
- A `trial_id` is unique within the dataset snapshot and must resolve to exactly one source slice.

### Optional fields and nullability

- `sampling_frequency_hz` may be null if the paper-reported value is not being attached; if set to `256.0`, provenance must identify it as paper-reported.
- Metadata column, participant index/identity, candidate values, diagnostic group, and task-category name may be null when unverified or unavailable.
- Null means unknown/not established, not “negative,” “control,” or a default class.
- `diagnostic_group` must be null in the current dataset contract until both diagnosis provenance and the EEG-to-metadata mapping are independently established.
- `stimulus_indicator` may be null when no metadata column has been linked; if present, it must be exactly three binary integers and must not be used as a diagnostic target.

### Types and signal shape

- `within_file_trial_index`, metadata column indices, and source participant indices must be integers; reject booleans and non-integral numeric values.
- `source_filename` is a basename matching the documented source file naming convention; do not accept paths that resolve outside the configured dataset directory.
- `signal` must be a NumPy array with shape `(385, 56)` and dtype `float32`.
- `sample_count` and `channel_count`, if stored, must equal `385` and `56`, respectively, and match the array dimensions.
- Sampling frequency, if provided, must be finite and positive; `256 Hz` carries paper provenance.
- Any transpose, crop, resampling, cast, filtering, channel mapping, or other transformation creates a different representation and requires its own approved contract and recorded transformation provenance. No such transformation is authorized here.

### Metadata consistency

- `trial_id` must agree with the source filename and local trial index. This is a source locator check, not a cross-file metadata join check.
- A metadata column index must be within `[0, 33902)`. Its association with a specific EEG sample must be marked candidate unless independently verified.
- A source participant index must be a positive integer present in the associated `y_stim` metadata. Do not infer a globally unique identity from the repeated row-0 values alone.
- A verified `participant_id` requires an authoritative identifier crosswalk. A candidate composite key such as `HC_01` belongs only in `candidate_participant_id`.
- A non-null `diagnostic_group` requires independent diagnostic provenance keyed to the participant and a verified trial-to-participant join. Filename prefixes, group counts, one-hot indicators, matching paper totals, or a proposed boundary are insufficient alone.
- A non-null `stimulus_task_category` requires authoritative semantic documentation for the source indicator. Preserve raw indicator values separately.
- Contradictory provenance, multiple distinct identities for one trial, invalid indicator values, shape/dtype mismatch, or a verified status unsupported by cited evidence must fail validation rather than be silently repaired.

### Provenance requirements

For each claim, record:

1. The source artifact or authoritative documentation (file, table/field, paper section, or versioned manifest).
2. The exact relationship being claimed (for example, EEG slice → metadata column, metadata index → participant ID, or participant ID → diagnosis).
3. Evidence status and a concise rationale, including known limitations or conflicting evidence.
4. The method and version used to derive a candidate value, where applicable.

Count equality, sequential filenames, file timestamps, and signal similarity may be recorded as supporting evidence. They must not be described as a direct identity key.

## 6. Example records

### Valid: raw trial with unresolved participant and diagnosis

```python
EEGSample(
    trial_id="d1.mat:0",
    source_filename="d1.mat",
    within_file_trial_index=0,
    signal=signal,  # validated shape (385, 56), dtype float32
    sampling_frequency_hz=256.0,
    sample_count=385,
    channel_count=56,
    provenance=(
        FieldProvenance(
            field="signal",
            source="dataset/d1.mat, dataset d1, trial-axis index 0",
            status="verified_source",
            note="Observed shape (385, 56) and dtype float32.",
        ),
        FieldProvenance(
            field="sampling_frequency_hz",
            source="He et al. (2023), as cited in docs/base-paper-analysis.md",
            status="verified_source",
            note="Paper-reported 256 Hz; not encoded in the EEG array.",
        ),
    ),
)

TrialMetadata(
    trial_id="d1.mat:0",
    metadata_column_index=None,
    participant_source_index=None,
    participant_id=None,
    candidate_participant_id=None,
    diagnostic_group=None,
    candidate_diagnostic_group=None,
    stimulus_indicator=None,
    stimulus_task_category=None,
    mapping_provenance=(
        FieldProvenance(
            field="trial_id",
            source="dataset/d1.mat, dataset d1, trial-axis index 0",
            status="verified_source",
            note="Identifies the stored source slice; no metadata join asserted.",
        ),
        FieldProvenance(
            field="EEG_to_y_stim_join",
            source="dataset/d1.mat and dataset/y_stim.mat",
            status="unresolved",
            note="No independent trial identifier or keyed crosswalk was found.",
        ),
        FieldProvenance(
            field="diagnostic_group",
            source="dataset metadata and cohort reports",
            status="unresolved",
            note="No verified participant diagnosis joined to this EEG trial.",
        ),
    ),
)
```

This is valid for read-only access and signal inspection. It is not a labeled supervised-learning example.

### Valid: metadata-only observation

A metadata record may preserve a source column's row-0 integer and one-hot indicator after direct inspection, provided it identifies the metadata column and does not claim that the column belongs to a particular EEG sample. Its participant identity and diagnostic group remain null unless separately verified.

### Invalid: inferred diagnosis treated as verified

```python
TrialMetadata(
    trial_id="d1.mat:0",
    metadata_column_index=0,
    participant_source_index=1,
    participant_id="controls2MSCT_Time_Estimation_NF_VP_02C.mat",
    candidate_participant_id="HC_01",
    diagnostic_group="HC",
    candidate_diagnostic_group="HC",
    stimulus_indicator=(1, 0, 0),
    stimulus_task_category="healthy_control",
    mapping_provenance=(),  # invalid: no provenance and unsupported verified claims
)
```

This record is invalid because it promotes an assumed sequential join, filename ordinal, block interpretation, and indicator meaning to verified identity/diagnosis without independent evidence. The stimulus indicator cannot establish diagnosis.

## 7. Supervised-learning eligibility

A sample may be used as a supervised-learning example **only when all** of the following hold:

1. The signal passes the shape, dtype, finite-value, and source-locator checks.
2. Its EEG-to-metadata association has an independently verified mapping (or a researcher-approved, authoritative mapping protocol with explicit evidence and limitations).
3. The participant identity is sufficiently verified to enforce the declared evaluation unit and prevent participant overlap when participant-independent generalization is claimed.
4. The diagnostic group is independently verified for that participant and the target's provenance is recorded.
5. The target is sourced from the verified diagnostic metadata, not from `y_stim` rows 1–3 or an inferred filename/group boundary.

Until these requirements are met, records are eligible for structural inspection and unlabeled signal handling only; they are not eligible for training, validation, testing, or performance claims about ADHD/ADD/HC classification.

## 8. Future PyTorch Dataset integration rules

No PyTorch `Dataset` is implemented by this contract. A future adapter must:

- Read through the bounded, read-only HDF5 data-access layer; never load the full EEG corpus just to serve an item or batch.
- Return signal tensors with per-item shape `(385, 56)` and `torch.float32`; a batch has shape `(batch, 385, 56)`. Preserve sample/channel order and do not silently transpose or crop.
- Keep trial identity and metadata separate from model input. Do not pass participant IDs, source filenames, metadata indices, stimulus indicators, or provenance flags as features unless a separately approved study explicitly requires them.
- Return a target only for records that pass the supervised-learning eligibility rules. For unresolved labels, return no target or fail closed for supervised mode; never synthesize a default class.
- Carry provenance/status alongside records or in an auditable sidecar so batching cannot erase candidate/unresolved distinctions.
- Keep every verified participant's trials within a single partition for participant-independent evaluation. Do not claim this guarantee from candidate subject keys.
- Open HDF5 handles in a worker-safe, read-only manner; do not share live handles across multiprocessing workers without an explicitly tested lifecycle.
- Record any approved transformation and its configuration; this contract alone authorizes none.

## 9. Current gate

Current directly observed facts include the seven EEG array shapes/counts/dtypes, `y_stim` dimensions and values, and the three filename-block sizes and contents. The global EEG-to-`y_stim` column join, filename-ordinal-to-row-0 identity join, and per-participant diagnostic labels remain unverified. Accordingly, the data contract defaults those joined fields to null or explicitly candidate status and blocks supervised-learning use until the provenance gate is passed.

Related evidence: [trial/subject mapping audit](./trial_subject_mapping_validation.md), [Sprint 2 preprocessing and data contract](./SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md), [dataset metadata investigation](../dataset_metadata_investigation.md), [preprocessing implementation](../src/adhd_eeg/preprocessing.py), and [data-access implementation](../src/adhd_eeg/data_access.py).
