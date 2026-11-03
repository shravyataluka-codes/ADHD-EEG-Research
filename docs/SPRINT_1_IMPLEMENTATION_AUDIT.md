# Sprint 1 Implementation Audit

**Status:** Engineering audit prepared for review; no Sprint 1 commit created
**Scope:** HDF5-to-PyTorch data access, synthetic validation, and provenance controls
**Pipeline reviewed:** HDF5 files → `TrialManifest` → `EEGTrialStore` → `EEGTrialDataset` → PyTorch `DataLoader`

## 1. Executive summary

Sprint 1 delivered a read-only HDF5 trial access path, expanded synthetic tests, documented unresolved trial/participant/diagnosis provenance, proposed a data contract, and added a PyTorch adapter with DataLoader integration.

The engineering path from a stored HDF5 trial to a batched PyTorch tensor works in synthetic tests. This is **not scientific validation** of the sample-to-metadata join, participant identities, or diagnostic targets. The source-local trial locator emitted by the Dataset must not be interpreted as proof of that join. Clinical supervised learning remains blocked pending independent provenance and protocol approval.

## 2. Task 1 — HDF5 data-access validation

### Direct observations

The seven local EEG HDF5 files were inspected read-only. Their observed arrays have shape `(385, 56, trials)` and `float32` dtype:

| Files | Trials per file |
|---|---:|
| `d1.mat`–`d6.mat` | 5,000 each |
| `d7.mat` | 3,902 |
| **Total** | **33,902** |

The trial total matches the number of columns in `y_stim.mat`. This count match does not establish that the EEG trial order matches the metadata column order.

`EEGTrialStore` opens files in read-only mode. Its uncached path reads only the requested trial slice and closes the file after the access; its context-managed path caches handles until `close()` or context exit. Synthetic tests and read-only probes covered individual trial retrieval, bounded batches, returned shape/dtype, first/last positions, and file transitions. The synthetic read-only access tests verified that trial retrieval did not modify the source data. Specifically, `test_source_files_are_unchanged_and_returned_trials_are_detached` in `tests/test_data_access.py` computes SHA-256 hashes of the three synthetic EEG files before and after retrieval and asserts that they match.

**Conclusion:** HDF5 indexing and retrieval behavior were verified against the observed data layout and synthetic fixtures. The scientific correctness of metadata joins was not established by this task.

## 3. Task 2 — Automated test strengthening

`tests/test_data_access.py` now covers first and last trials in every synthetic file; all file transitions; batches crossing transitions with value checks; invalid indices through store APIs; empty index sequences and batch-size validation; malformed dimensions and source dtype; non-integral file trial counts and group-range boundaries; and source immutability.

`src/adhd_eeg/data_access.py` received a narrow validation fix: non-integral file trial counts and range boundaries are rejected rather than silently truncated by `int()`. Participant mapping behavior and group boundaries were not changed.

### Recorded test counts

| Test scope | Result |
|---|---:|
| Data-access tests | **14 passed** |
| Dataset tests | **7 passed** |
| Combined data-access and Dataset tests | **21 passed** |
| Full unittest discovery | **Blocked**: preprocessing tests could not import because `mne` is unavailable |

The full suite did **not** pass as a complete run: unittest discovery reported an import error for `test_preprocessing` (`ModuleNotFoundError: No module named 'mne'`). Dependencies were not installed.

## 4. Task 3 — Trial-to-subject provenance audit

### Evidence inventory and classification

| Observation or relationship | Classification | Evidence and limitation |
|---|---|---|
| EEG file shapes, dtype, and per-file trial counts | **Directly observed** | HDF5 inspection of `d1.mat`–`d7.mat`: `(385, 56, trials)`, `float32`; 5,000 trials in `d1`–`d6`, 3,902 in `d7`. |
| `y_stim.mat` shape and dtype | **Directly observed** | HDF5 metadata has shape `(4, 33902)` and dtype `float64`. |
| `y_stim` row 0 values and runs | **Directly observed** | Integer-valued 1–52, forming 144 contiguous runs, with resets at run positions 0, 44, and 96. Interpreting these as within-group participant indices is **indirectly supported**, not independently keyed. |
| `y_stim` rows 1–3 | **Directly observed** | Three binary, one-hot combinations are present, constant within the corresponding metadata blocks; counts are 10,129, 13,031, and 10,742. Their physical task/stimulus meanings are **unresolved**. |
| `sub_name_stim.mat` | **Directly observed** | Three ordered cell blocks with 44, 52, and 48 unique filenames, with `controls`, `subtype1`, and `subtype2` prefixes. |
| Filename ordinal *i* equals row-0 participant index *i* | **Candidate** | Block sizes, ordering, and timestamps are consistent with ordinal pairing, but there is no independent filename-to-index key. |
| Concatenated EEG trial order equals `y_stim` column order | **Unresolved** | Counts agree, but inspected EEG datasets contain no trial IDs or keyed metadata references. File timestamps establish creation order, not the EEG-to-metadata join. |
| Blocks correspond to HC, ADD, ADHD | **Indirectly supported / candidate interpretation** | Filename prefixes and cohort/trial totals align with published group descriptions. No local participant diagnosis table independently links a verified diagnosis to every EEG trial. |

### Candidate mapping assumptions

The current `TrialManifest` default crosswalk assumes:

1. `d1`–`d7` are concatenated in numeric order using their observed trial counts.
2. The resulting global index equals the `y_stim` column index.
3. Row 0 is a within-block participant index.
4. The three contiguous metadata blocks correspond to the configured group ranges.
5. Filename list order matches participant-index order within each block.

Those rules produce a deterministic candidate mapping. Deterministic indexing does not make the identity join verified.

### Why the existing mapping validation is not independent proof

The validation report tests boundary correlations by selecting proposed “straddling subjects” and signal slices using the candidate `y_stim` alignment. The result is corroborating structural evidence, but it does not supply an independently keyed subject or trial identity and does not rule out alternate joins. Likewise, exact count matches, chronological file creation times, sorted filenames, and matching timestamps do not establish which EEG trial corresponds to which metadata column or filename.

The `y_stim` one-hot indicators must remain separate from any diagnostic target. Their block-wise coincidence with the candidate group blocks is not diagnostic provenance. The audit’s detailed evidence and its conflicting earlier verdicts are documented in [trial_subject_mapping_validation.md](./trial_subject_mapping_validation.md), [ADHD_EEG_PREPROCESSING_AUDIT.md](./ADHD_EEG_PREPROCESSING_AUDIT.md), and [SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md](./SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md).

**Conclusion:** The EEG-to-`y_stim` join, filename-to-index link, participant identity mapping, and diagnostic label provenance remain unverified. `y_stim` stimulus indicators are not diagnostic labels. Clinical supervised learning is blocked until these mappings are independently established and approved.

## 5. Task 4 — Data contract

[data_contract.md](./data_contract.md) records the proposed raw signal representation, metadata separation, provenance vocabulary, nullability rules, examples, validation expectations, and supervised-learning eligibility gate. It keeps participant identity, diagnostic group, and stimulus/task indicators in separate fields; candidate identities and groups are not represented as verified values.

The proposed contract requires `(385, 56)` `float32` signals, provenance for asserted mappings, finite signal values for eligible samples, and independently verified diagnostic labels and trial-to-participant joins before supervised use. Participant-disjoint evaluation requires a verified grouping identity; the currently proposed block/index key is not enough while its join remains unverified.

The contract is an engineering/data-governance proposal, **not scientific validation or approval of a research protocol**. The Dataset adapter does not independently scan tensors for finite values; finiteness must be checked by a separate validation gate before any downstream eligibility decision.

## 6. Task 5 — PyTorch Dataset/DataLoader integration

### Implementation

`src/adhd_eeg/dataset.py` provides `EEGTrialDataset(manifest)`. Each indexed sample returns:

- A `torch.float32` tensor with shape `(385, 56)`.
- A separate metadata dictionary containing exactly `trial_id`, `source_filename`, and `within_file_trial_index`.

Here, `trial_id` is a repository-defined, source-local locator. It does not assert an EEG-to-metadata join. The Dataset omits candidate subject IDs, clinical groups, diagnostic labels, and unresolved metadata associations.

`tests/test_dataset.py` contains seven synthetic tests for length, values, tensor shape/dtype, metadata separation, invalid indices, default DataLoader collation, and HDF5 cleanup after both successful reads and a validation exception. The cleanup tests confirm files can be renamed after each operation.

`pyproject.toml` declares PyTorch as the optional `pytorch = ["torch"]` extra. HDF5-only functionality remains importable without PyTorch; `adhd_eeg.dataset` itself requires the optional dependency.

### HDF5 lifecycle and DataLoader behavior

The Dataset does not enter the store’s handle-caching context. Each uncached `EEGTrialStore` read opens the HDF5 file in read-only mode inside a context manager and closes it before returning or propagating an exception. The Dataset therefore retains no live HDF5 handles between accesses.

Default DataLoader collation was tested: signals batch as `(batch, 385, 56)` tensors, and the source metadata keys collate separately. A synthetic `DataLoader(num_workers=2)` test passed on Windows using a guarded temporary Python script. An initial attempt to launch multiprocessing from standard input failed because Windows spawn could not reload `<stdin>`; rerunning from a script file passed. This is a narrow synthetic Windows check, not a guarantee across operating systems, worker start methods, or production workload.

## 7. Verification record

Recorded commands and outcomes:

- `python -m unittest discover -s tests -p test_data_access.py -v` — **14 passed**.
- `python -m unittest discover -s tests -p test_dataset.py -v` — **7 passed**.
- `python -m unittest discover -s tests -v` — **21 tests passed; preprocessing test import failed** because `mne` is unavailable. The complete suite is therefore **not verified as passing**.
- Synthetic `DataLoader(num_workers=2)` probe from a guarded script — **passed** with expected values and source-local identifiers.
- Optional-dependency isolation probe with PyTorch imports blocked — **passed** for `adhd_eeg` and HDF5 data-access imports; the Dataset module correctly requires PyTorch.
- `git diff --check` — **passed** for the Sprint 1 implementation changes before this audit document was added. The new audit document receives a fresh diff check as part of this review.

No claim of scientific validity, verified participant mapping, or reproduced model performance is made.

## 8. Remaining risks and blockers

- **EEG-to-`y_stim` trial alignment:** unresolved; no independent per-trial join key was found.
- **Participant identity:** filename ordinal and EEG-to-participant mapping are not independently verified.
- **Diagnostic labels:** participant-level diagnostic provenance and the relationship to the EEG trials remain unverified.
- **385th sample:** its physical/time-coordinate meaning remains unresolved; preserve all 385 positions.
- **Channel identity:** EEG arrays have 56 columns while `chan.mat` has 60 channel entries; identities of the four unmatched channels and column mapping remain unresolved.
- **Signal finiteness:** required by the proposed contract for eligible samples, but not checked by `EEGTrialDataset` itself.
- **Preprocessing:** no preprocessing choice is approved by this audit. Any fitted preprocessing statistics must be confined to training participants after a valid split and an approved protocol.
- **Evaluation design:** exact folds, split implementation, validation, and other protocol details remain to be independently established and approved.
- **Modeling/results:** no Transformer training, classification, or scientific performance claims have been made.
- **Environment:** the preprocessing test module remains untested because `mne` is unavailable; multi-worker validation was limited to a small synthetic Windows run.

## 9. Sprint 2 entry criteria

Sprint 2 must begin with **participant-aware evaluation design and leakage prevention**. Before any split manifest or training work is authorized:

1. Independently establish the EEG trial-to-metadata and trial-to-participant mappings.
2. Verify diagnostic label provenance at the participant level and its association with the EEG trials.
3. Define and approve the evaluation protocol, including the participant grouping unit, split strategy, validation/model-selection boundaries, and leakage controls.
4. Reconcile conflicting mapping conclusions in existing project records and record evidence limitations.

Until these criteria are met and approved, the pipeline is limited to structural inspection and unlabeled data access. This audit does not authorize model training.

## 10. Change and approval status

This document records engineering work and remaining research gates. It does not stage or commit files. Review and explicit approval are required before creating the Sprint 1 commit or proceeding to the next task.
