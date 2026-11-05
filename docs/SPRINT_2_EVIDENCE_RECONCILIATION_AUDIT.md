# Sprint 2 — Evidence Reconciliation Audit

**Scope:** Read-only reconciliation of repository evidence on dataset structure, trial/participant joins, and diagnostic labels
**Status:** Documentation audit; no mapping is promoted to verified
**Branch at inspection:** `ml/transformer-development`

## 1. Executive findings

The inspected repository supports a clear distinction between structural observations and identity provenance:

- The project records direct inspection of seven EEG files with 33,902 total trial positions, `y_stim` with 33,902 columns, 144 contiguous row-0 runs, three filename blocks, and a 60-entry channel metadata structure. These observations establish structure and counts, not the join between a particular EEG trial and a metadata column or person.
- The candidate crosswalk is explicit in [`src/adhd_eeg/data_access.py`](../src/adhd_eeg/data_access.py): concatenate `d1.mat`–`d7.mat` in numeric order, use the same zero-based global index for `candidate_y_stim_column`, assign the configured contiguous block as `candidate_diagnostic_group`, and combine that block with row-0 value for `candidate_participant_id`. This is a deterministic implementation of assumptions, not independent evidence for them.
- `y_stim` row 0 has values and runs that are consistent with within-block participant indices. The values do not identify people by themselves. `sub_name_stim` has ordered filename blocks with matching counts and prefixes, but no authoritative key links a filename ordinal to a row-0 run.
- The paper reports aggregate HC/ADD/ADHD counts and says participants were clinically classified. The repository has no identified participant-level clinical diagnosis table keyed to stable identities and linked to EEG trial locators. Aggregate count agreement and filename prefixes do not establish diagnosis for each local participant.
- Rows 1–3 of `y_stim` are directly described as binary one-hot patterns constant across the three metadata blocks. The repository alternately calls them stimulus/task categories and group/diagnostic indicators. Their physical semantics are unresolved; their collinearity with the candidate blocks does not make them an independent diagnostic source. Do not use them as clinical targets.
- The six boundary correlations in the mapping-validation script are selected and named using the candidate `y_stim` alignment. They may be consistent with continuity, but are not an independent trial key and do not prove every trial’s identity. The script’s final “validated” verdict is stronger than its evidence.
- The 385-position sample axis and 56 EEG channels versus 60 `chan.mat` entries are observed dimensions; the endpoint meaning and exact channel-column correspondence remain unresolved.

**Overall conclusion:** Structural facts are substantially documented; EEG-to-metadata order, filename-to-participant identity, and participant-level diagnoses remain unresolved. No current evidence justifies describing the crosswalk or diagnostic labels as verified.

## 2. Files inspected

All six specifically requested documents and both implementation/validation files were present:

- [`docs/dataset-investigation.md`](dataset-investigation.md)
- [`docs/dataset-preprocessing-audit.md`](dataset-preprocessing-audit.md)
- [`docs/ADHD_EEG_PREPROCESSING_AUDIT.md`](ADHD_EEG_PREPROCESSING_AUDIT.md)
- [`docs/domain-understanding.md`](domain-understanding.md)
- [`docs/SPRINT_2_EVALUATION_READINESS_AUDIT.md`](SPRINT_2_EVALUATION_READINESS_AUDIT.md)
- [`docs/SPRINT_2_PROVENANCE_EVIDENCE_INVESTIGATION.md`](SPRINT_2_PROVENANCE_EVIDENCE_INVESTIGATION.md)
- [`src/adhd_eeg/data_access.py`](../src/adhd_eeg/data_access.py), especially `TrialManifest`, `DEFAULT_FILE_TRIAL_COUNTS`, `DEFAULT_GROUP_RANGES`, and `metadata_for()`
- [`scripts/validate_trial_subject_mapping.py`](../scripts/validate_trial_subject_mapping.py), especially `validate_eeg_files()`, `validate_y_stim()`, `validate_sub_name_stim()`, and `validate_boundaries_and_ordering()`

No requested file was missing. This was a review of the listed documents and code, not a fresh binary-data revalidation or an external OSF lookup.

## 3. Evidence classifications

| Classification | Meaning in this reconciliation |
|---|---|
| **Verified source evidence** | A primary source establishes the limited statement attributed to it (for example, the paper reports a cohort count). |
| **Verified binary observation** | Repository records of direct file inspection establish a structural property of the local MAT data. This does not establish a semantic interpretation or cross-file identity join. |
| **Supported indirectly** | Multiple observations are consistent with an interpretation, but an authoritative keyed source is absent. |
| **Candidate** | A proposed join or interpretation encoded in code or documentation, without independent confirmation. |
| **Unresolved** | The inspected evidence cannot establish the asserted fact or relationship. |

“Verified” is scoped to the precise observation: verified counts do not verify identity; verified strings do not verify the person or diagnosis represented by a string.

**Current terminology boundary:** `(source_filename, within_file_trial_index)` is a source-local trial locator. `candidate_participant_id` and `candidate_diagnostic_group` are candidate mappings, not verified identities or clinical labels. A verified clinical diagnosis label must come from an authoritative participant-level diagnosis source linked to the source-local trial; no such linkage is established here. `y_stim` rows 1–3 are not accepted as clinical targets.

## 4. Evidence reconciliation table

| Topic and original claim | Supporting source or observation | Classification | Later confirmation, qualification, or contradiction | Final defensible conclusion |
|---|---|---|---|---|
| **1. Seven EEG files and trial counts.** The dataset is split into `d1.mat`–`d7.mat`; `d1`–`d6` have 5,000 trials each and `d7` has 3,902. Earlier [`dataset-investigation.md`](dataset-investigation.md), §§7–8 and 13, initially treats some counts as strongly supported from sizes and total counts. | [`scripts/validate_trial_subject_mapping.py`](../scripts/validate_trial_subject_mapping.py), `validate_eeg_files()`, checks expected shapes `(385, 56, 5000)` for `d1`–`d6` and `(385, 56, 3902)` for `d7` and asserts total 33,902 (lines 40–94). [`docs/ADHD_EEG_PREPROCESSING_AUDIT.md`](ADHD_EEG_PREPROCESSING_AUDIT.md), §§3–4, reports direct inspection of all seven shapes and counts. | **Verified binary observation** as documented and checked by the script’s assertions. Total-count equality with `y_stim` is a separate structural observation. | The later audit raises the count status from earlier size-based inference by reporting inspection of all seven binaries. It explicitly warns that equal total counts do not establish trial order. | The file inventory and trial counts are adequately supported as binary structure. They establish the number of array positions and metadata columns, not which array position corresponds to which column. This audit did not rerun the binary scans. |
| **2. Candidate EEG-file-to-`y_stim` column mapping.** Older [`dataset-investigation.md`](dataset-investigation.md), §9 and §17 Decision 1, says `d1` maps to columns `[0,5000)` is “strongly supported,” while acknowledging it is not directly documented. | [`src/adhd_eeg/data_access.py`](../src/adhd_eeg/data_access.py), file counts and `metadata_for()`, assigns the file by numeric-order cumulative offsets and sets `candidate_y_stim_column=index`. The current evaluation-readiness audit §4 and provenance investigation §5 explicitly call this candidate. | **Candidate / unresolved.** Count equality and numeric order support a hypothesis, not a keyed join. | [`docs/dataset-preprocessing-audit.md`](dataset-preprocessing-audit.md), §§2.1–2.3, uses “strongly supported” sequential chunking, but the stricter [`ADHD_EEG_PREPROCESSING_AUDIT.md`](ADHD_EEG_PREPROCESSING_AUDIT.md), §4, states there is no trial ID or checksum cross-reference and no independent proof. The provenance investigation finds no source crosswalk in the repository/history. | The numeric-order offsets in the manifest are a deterministic candidate only. No trial’s `y_stim` column is verified against its EEG locator by the inspected evidence. |
| **3. `y_stim` row 0 and participant indexing.** Older [`dataset-investigation.md`](dataset-investigation.md), §4, and [`domain-understanding.md`](domain-understanding.md), §§2.4, 2.6, and 15–16, call row 0 a within-group subject index and at points describe that identity as directly established. | [`scripts/validate_trial_subject_mapping.py`](../scripts/validate_trial_subject_mapping.py), `validate_y_stim()`, tests integer run structure, 144 runs, resets `[0,44,96]`, and ascending values within 44/52/48 blocks (lines 96–185). The reported direct observation is row-0 value patterns and contiguous runs. | **Verified binary observation** for values/runs; **supported indirectly** for “within-block subject index”; **unresolved** for stable person identity. | [`ADHD_EEG_PREPROCESSING_AUDIT.md`](ADHD_EEG_PREPROCESSING_AUDIT.md), §5, explicitly distinguishes observed row-0 runs from an independently keyed participant identity. The provenance investigation §6 makes the same distinction. | Row 0 has contiguous index-like runs. Interpreting a run as one participant is plausible and consistent with metadata structure, but the integer is not itself a verified identifier and does not prove which EEG trials belong to that person. |
| **4. `sub_name_stim` block ordering and filename correspondence.** Earlier [`dataset-investigation.md`](dataset-investigation.md), §5, describes the file as participant identifiers/original filenames and relates each block to a group. [`domain-understanding.md`](domain-understanding.md), §15, calls filenames links to original source data. The mapping report proposes filename ordinal *i* = row-0 index *i*. | The mapping script’s `validate_sub_name_stim()` checks 44/52/48 strings, prefixes, uniqueness, and whether stored strings are lexically sorted (lines 188–254). [`docs/trial_subject_mapping_validation.md`](trial_subject_mapping_validation.md), §§6 and 8, reports those observations and the ordinal hypothesis. | **Verified binary observation** as reported for block structure/strings/order; **candidate** for filename-to-index/person correspondence. | The mapping report itself (§§8 and 14) acknowledges no individual source files were available to verify ordinal pairing. [`ADHD_EEG_PREPROCESSING_AUDIT.md`](ADHD_EEG_PREPROCESSING_AUDIT.md), §5, and the provenance investigation §6 state that sorted order and matching block lengths are not an authoritative key. | Stored filename order is an observable property; the original ordering procedure and its correspondence to row-0 runs are unresolved. Do not assign a filename or identity to a candidate index as verified. |
| **5. HC, ADD, ADHD group assignment.** Older [`dataset-investigation.md`](dataset-investigation.md), §§2 and 17 Decision 4, and [`domain-understanding.md`](domain-understanding.md), §§2 and 15–16, state the filename blocks correspond to HC/ADD/ADHD, sometimes with a “directly established” label. | [`docs/base-paper-analysis.md`](base-paper-analysis.md), §4, reports 44 HC, 52 ADD, 48 ADHD participants and group trial counts; the filenames’ `controls`, `subtype1`, and `subtype2` prefixes and block sizes match those totals. | **Verified source evidence** for the paper’s aggregate group counts; **verified binary observation** for block sizes/prefixes; **supported indirectly / candidate** for mapping each local block or participant to a diagnosis. | The provenance and evaluation audits qualify the previous labels: the MAT files do not contain a participant-keyed diagnostic field, and exact count matching is not individual diagnosis provenance. | The candidate block interpretation controls→HC, subtype1→ADD, subtype2→ADHD is consistent with the paper and names, but no individual diagnosis in the local trial table is thereby verified. |
| **6. Participant identity and diagnostic-label provenance.** The paper reports professional clinical assessment; some older project summaries say the file strings are participant identities and treat the group assignment as validated. | [`docs/base-paper-analysis.md`](base-paper-analysis.md), §4, is the source for what the paper reports about clinical assessment and aggregate cohort. [`docs/data_contract.md`](data_contract.md), §§3–5 and 7 (cited by the evaluation audit), requires an authoritative participant key and diagnosis provenance. | **Verified source evidence** for the paper’s description of diagnosis procedures; **unresolved** for participant-level local labels and trial linkage. | [`SPRINT_2_EVALUATION_READINESS_AUDIT.md`](SPRINT_2_EVALUATION_READINESS_AUDIT.md), §§4–5 and 9–10, and [`SPRINT_2_PROVENANCE_EVIDENCE_INVESTIGATION.md`](SPRINT_2_PROVENANCE_EVIDENCE_INVESTIGATION.md), §§6 and 9, confirm no participant-level, identity-keyed diagnosis record/crosswalk was found. | No verified participant identity or per-participant clinical label is available in the inspected repository. Aggregate counts and filename prefixes cannot substitute for that missing record. |
| **7. Stimulus categories versus clinical labels.** [`dataset-investigation.md`](dataset-investigation.md), §4, calls rows 1–3 stimulus/task categories and says they are not diagnostic labels; [`domain-understanding.md`](domain-understanding.md), §§12, 15, and 17, repeats the distinction. The later mapping-validation script/report calls the rows block/group indicators and uses them to support group/diagnosis claims. | `validate_y_stim()` verifies binary one-hot patterns and that each pattern is constant within its row-0 block; its own output says they are “100% collinear” with block identities (lines 96–185). The paper’s aggregate trial counts happen to match row indicator totals. | **Verified binary observation** for one-hot/block-constant structure; **unresolved** for the rows’ semantic names and intended role. | The direct structure contradicts an interpretation as trial-varying stimulus labels, but does not independently establish a clinical meaning. The mapping report’s inference from collinearity to diagnosis overstates the evidence. Later data contracts and both Sprint 2 audits prohibit use as diagnostic targets. | Keep rows 1–3 as uninterpreted source indicators unless an authoritative data dictionary resolves their meaning. They are not independent diagnostic evidence and must not be used as HC/ADD/ADHD labels. |
| **8a. 385-sample discrepancy.** Older documents describe a 385-position sample/time axis and suggest an inclusive endpoint or other possibilities. | The mapping script checks first dimension `385`; the paper analysis reports 256 Hz, nominal 1.5-second trials, and an architecture input length of 385. | **Verified binary observation** for dimension; **verified source evidence** for paper-reported parameters; **unresolved** for the extra position’s physical meaning. | [`domain-understanding.md`](domain-understanding.md), §§5–6, explicitly says the physical meaning is unknown. The evaluation and provenance audits preserve the limitation. | Keep 385 as an observed dimension. No endpoint, padding, baseline, or time-coordinate interpretation is established. |
| **8b. 56-versus-60 channel discrepancy.** Earlier documents call 56 the EEG data dimension and 60 the `chan.mat` entry count. | The script’s EEG assertions use 56 columns; [`ADHD_EEG_PREPROCESSING_AUDIT.md`](ADHD_EEG_PREPROCESSING_AUDIT.md), §§3–4, and [`domain-understanding.md`](domain-understanding.md), §4, report 56 data columns and 60 channel records. | **Verified binary observation** for the two counts; **unresolved** for column labels and the four unmatched entries. | Later audits consistently warn not to infer the missing channels’ identities or their causes. | The mismatch is real as a count comparison; there is no verified 56-to-60 mapping. |
| **9. Does mapping validation independently establish the joins?** [`trial_subject_mapping_validation.md`](trial_subject_mapping_validation.md), §§7 and 16, claims boundary correlations prove sequential continuity and declares the composite key validated/safe. | In `validate_boundaries_and_ordering()`, the six `boundary_cases` encode candidate global ranges and claimed participant identities; code then uses offsets based on 5,000 trials, reads those slices, and compares channel-variance profiles (lines 256–316). `print_trial_subject_crosswalk_sample()` also combines `runs`, numeric file boundaries, and filename ordinal using the assumed crosswalk (lines 318–353). | **Indirect/corroborative** signal consistency only; **candidate-dependent test**, not independent validation. | [`SPRINT_1_IMPLEMENTATION_AUDIT.md`](SPRINT_1_IMPLEMENTATION_AUDIT.md), §4, and the two Sprint 2 audits explicitly explain that the ranges are selected from the candidate alignment. The validation report’s conclusion is contradicted by that methodological limitation and by its own filename caveat. | The tests are not independent of the proposed join: the labels/ranges being “validated” are used to select the slices and assign names before calculating correlations. Correlation is not a unique participant identifier, and six boundary checks cannot establish all 33,902 trial mappings. The “validated/safe” conclusion is unsupported. |

## 5. Contradictions and overstatements in older documentation

1. **“Directly established subject index” versus observed index-like values.** The values, resets, and runs are directly observable. Their semantics as a within-group participant index are plausible but not independently keyed. Statements in [`dataset-investigation.md`](dataset-investigation.md) and [`domain-understanding.md`](domain-understanding.md) should be read with this narrower distinction.
2. **Filename blocks versus clinical group truth.** The files contain names/prefixes and block counts. Using those as individual clinical labels overstates the evidence. The paper’s aggregate counts support the block interpretation indirectly; they do not supply a diagnosis per local identity.
3. **Rows 1–3 described as stimulus categories and also as diagnostic/block categories.** Binary one-hot structure and its block-constant relation are verified observations; intended semantics are not. Neither the word `stim` nor count collinearity settles whether the fields are stimulus/task labels, group encodings, or another source convention. Treating them as clinical targets is not defensible.
4. **“Strongly supported” or “validated” file order from counts and timing.** Equal totals, chunk sizes, creation times, and numeric filenames are consistent with numeric concatenation, but none keys a particular trial. The later preprocessing audit, implementation audit, and Sprint 2 audits properly qualify these statements.
5. **Boundary-correlation “proof.”** The six slices are selected according to the proposed mapping; their correlation is not an independent source of participant identity. Describing the comparison control slice as a different participant also relies on the candidate grouping unless supported by an independent key.
6. **Historical “safe composite subject key” claim.** `group_block + row0` may be mathematically unique for the configured ranges and row values. Uniqueness of a constructed candidate-participant key is not proof that all trials grouped under it belong to one verified person or that the candidate group is correct.
7. **“Participant information” in `sub_name_stim`.** Filename strings may be participant source filenames, but their ordinal linkage to row-0 runs and individual diagnoses is still candidate. Do not equate an observed filename with an independently verified participant ID.

## 6. Remaining provenance blockers

- **Authoritative trial order:** Source procedure or independent key mapping every `(source file, within-file trial index)` to a `y_stim` column.
- **Participant key:** Stable participant identifiers linked to every metadata run and trial; if filenames are to be used, a source-backed filename-to-identifier/order crosswalk.
- **Clinical diagnosis:** Participant-level diagnosis records linked to that stable identifier, with coding definitions and treatment of exclusions/ambiguous labels.
- **Field semantics:** Authoritative documentation of `y_stim` row meanings and the relation, if any, between the one-hot fields and clinical group.
- **Source-version provenance:** OSF README/file manifest and source hashes/versions tying the local binaries to a recoverable source snapshot; original chunk-generation script or equivalent custodian statement.

Until these are supplied, every trial-to-column, trial-to-participant, and trial-to-diagnosis assignment remains candidate or unresolved.

## 7. What can safely be implemented now

Within the current evidence and approved scope, the following are safe:

- Read-only inspection and reporting of file presence, shape, dtype, counts, source hashes, and metadata structure, clearly separating binary observations from joins.
- Source-local trial access using `(source_filename, within_file_trial_index)` as a locator, without attaching a verified participant or diagnosis.
- Synthetic-only testing of generic validation behavior using fabricated participant IDs/labels, provided those tests are not presented as real dataset splits or evidence.
- Documentation and provenance work: preserve an OSF source snapshot, seek original construction code, request authoritative crosswalks, and reconcile evidence wording.

These are not permission to change source code or data in this audit; the current deliverable is documentation only.

## 8. What must remain blocked

- Treating `TrialManifest.candidate_participant_id`, `candidate_diagnostic_group`, or `candidate_y_stim_column` as verified research labels or identity associations.
- Generating or using real train/validation/test partitions, including a split-generator implementation intended for this dataset.
- Three-class supervised training, validation, testing, or performance claims.
- Assigning participant names or diagnostic labels from filename ordinal, row-0 values, one-hot indicators, matching totals, or correlations.
- Claiming that the current mapping-validation script proves the joins.

No model training, preprocessing, or split generation is authorized by this audit.

## 9. Recommendation for the next Sprint 2 task

**Recover and review primary provenance artifacts before any split work.** Obtain the exact OSF README and versioned file listing, original MATLAB/Octave construction code (or an authoritative custodian account of its behavior), an all-trial EEG-locator-to-metadata-column key, a stable participant/filename crosswalk, and a participant-level diagnosis record linked by that key. Record source versions and checksums, then update the evidence inventory only after independently checking coverage and uniqueness. If the public project lacks these artifacts, request them from the OSF maintainer/data custodian or original research team; do not infer them from the existing files.

After primary evidence is reviewed, reconcile the mapping report’s “validated” verdict against the source-backed joins. Only a later, explicit approval should open split implementation.

## 10. Review boundary and worktree preservation

This audit adds no source-code, dataset, manifest, split-logic, test, or existing-audit changes. No files were staged or committed. At inspection start, `git status --short` showed pre-existing changes:

```text
 M docs/README.md
 M src/adhd_eeg/preprocessing.py
?? docs/SPRINT_2_EVALUATION_READINESS_AUDIT.md
?? docs/SPRINT_2_PROVENANCE_EVIDENCE_INVESTIGATION.md
?? docs/literature-review/ADHD_EEG_Preprocessing_Literature_Synthesis.md
```

Those changes were preserved.
