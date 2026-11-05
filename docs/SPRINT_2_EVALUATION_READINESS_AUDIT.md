# Sprint 2 — Evaluation Readiness Audit

**Branch:** `ml/transformer-development`
**Scope:** Participant-aware evaluation readiness and leakage controls
**Status:** Repository evidence audit; documentation only
**Decision:** Evaluation partitions and supervised classification are blocked pending independent mapping and diagnosis evidence.

## 1. Executive summary

The repository contains a working, read-only path for loading source-local EEG trials, plus a deterministic `TrialManifest` crosswalk that attaches each global trial index to a file/index, `y_stim` column, candidate participant key, and configured group. The implementation and its synthetic tests establish software behavior; they do not establish that the candidate crosswalk corresponds to the real participant/trial identities.

The critical evidence is still missing: an authoritative per-trial link from the seven EEG files to `y_stim` columns, an independently keyed link from the row-0 subject runs to participant identities, and participant-level diagnostic records linked to those identities. Matching counts, contiguous runs, filenames, file timestamps, and deterministic boundary calculations do not independently establish these links. The boundary-correlation procedure in the mapping-validation script selects its “same-subject” slices using the candidate alignment, so it is corroborative structural evidence, not an independent identity key. The mapping report’s final `VALIDATED` verdict conflicts with its own limitations and with the more conservative Sprint 1 implementation audit.

**Current readiness:** **Blocked.** The dataset is not ready for participant-independent supervised evaluation. A split generator must not be used to create research partitions from the current candidate IDs or candidate class labels. Three-class diagnosis cannot be safely assigned to EEG trials from the current evidence.

If authoritative mapping and diagnosis evidence is obtained, keep every verified participant wholly within one partition. A stratified grouped cross-validation design is a reasonable **conditional candidate** because the repository reports a candidate 44/52/48 participants across three blocks; exact fold count, held-out test design, validation/model-selection boundaries, and metrics still require a prespecified decision. This recommendation does not promote the current candidate identities or labels to verified status.

This audit examines repository code and documentation. It does not re-read or independently re-validate the local EEG binary files, contact data custodians, generate splits, or validate scientific results.

## 2. Status vocabulary and scope

Conclusions use the requested classifications:

| Classification | Meaning in this audit |
|---|---|
| **Verified** | Directly established for the narrowly stated repository/source fact. |
| **Supported indirectly** | Consistent structural or external-description evidence exists, but no authoritative key establishes the relevant identity or join. |
| **Candidate** | A plausible, deterministic interpretation used by current code or reports, not independently confirmed. |
| **Unresolved** | Evidence inspected here cannot establish the fact or relationship. |
| **Blocked** | The missing evidence prevents the proposed research operation or claim. |

This audit is limited to Sprint 2 evaluation readiness. It does not change the roadmap, authorize preprocessing, or claim scientific validation.

## 3. Repository investigation

### 3.1 Data access and metadata surfaces

- [`src/adhd_eeg/data_access.py`](../src/adhd_eeg/data_access.py), `TrialManifest`: defaults to seven file trial counts and three contiguous `group_ranges`; loads `y_stim[0, :]` after checking its dimensions and integer-valued contents. `metadata_for()` returns source-local file/index locators plus the candidate `candidate_y_stim_column`, `candidate_participant_id`, and `candidate_diagnostic_group` fields. These operations validate internal structure and deterministic indexing, not the truth of the EEG-to-metadata, participant, or diagnostic joins.
- [`src/adhd_eeg/data_access.py`](../src/adhd_eeg/data_access.py), `EEGTrialStore`: validates the expected HDF5 dataset name, shape, dtype, and requested slice. It reads the trial locator supplied by the manifest; it does not independently identify a participant or confirm the metadata column associated with the signal.
- [`src/adhd_eeg/dataset.py`](../src/adhd_eeg/dataset.py), `EEGTrialDataset`: returns a tensor and source-local identifiers (`trial_id`, `source_filename`, and `within_file_trial_index`). It returns no participant ID, clinical group, stimulus indicator, or target. This is appropriate for unlabeled access and does not provide split-ready labels.
- [`docs/data_contract.md`](data_contract.md), §§2–5, 7–9: distinguishes source-local locators from candidate mappings, requires provenance for asserted mappings, keeps diagnosis separate from `y_stim` indicators, and blocks supervised eligibility absent independently verified trial-to-participant and diagnostic mappings.
- Metadata inspection scripts [`scripts/inspect_ystim_relations.py`](../scripts/inspect_ystim_relations.py), [`scripts/inspect_sub.py`](../scripts/inspect_sub.py), and [`scripts/inspect_deep.py`](../scripts/inspect_deep.py) examine `y_stim` or `sub_name_stim` structure. They do not establish the EEG-to-metadata join. They also contain machine-specific absolute paths and should not be treated as a reproducible mapping authority.

**Terminology boundary:** `(source_filename, within_file_trial_index)` is a source-local trial locator, not a participant ID. `candidate_participant_id` and `candidate_diagnostic_group` are deterministic candidate mappings only. Verified clinical diagnosis labels require a separate participant-keyed source and verified linkage to the EEG locators; none is identified here. `y_stim` rows 1–3 are not clinical targets.

### 3.2 Mapping validation and conflicting conclusions

[`scripts/validate_trial_subject_mapping.py`](../scripts/validate_trial_subject_mapping.py) checks file dimensions/counts, `y_stim` runs and indicator patterns, and `sub_name_stim` block sizes, prefixes, and ordering. Its boundary test (function `validate_boundaries_and_ordering`) selects boundary trial ranges and labels them as straddling participants from the proposed `y_stim` alignment, then compares channel-variance profiles. The final printed verdict calls the composite key and group assignment validated. The selected slices therefore test whether signals are consistent with a preselected candidate join; they do not independently identify those trials or exclude alternate joins.

The report [`docs/trial_subject_mapping_validation.md`](trial_subject_mapping_validation.md) §§7–9 and 12–16 elevates count agreement, header timestamps, filename ordering, and the boundary correlations to “validated” sequential ordering, clinical diagnosis, and a “safe” split key. However, §§8 and 14 acknowledge that filename ordinal is not independently verified; §16’s verdict is stronger than that caveat permits. Most importantly, neither matching counts nor a signal statistic computed from ranges selected by the candidate alignment supplies an independent trial/participant key.

The later [`docs/SPRINT_1_IMPLEMENTATION_AUDIT.md`](SPRINT_1_IMPLEMENTATION_AUDIT.md), §§4–5 and 8–9, explicitly treats those joins and diagnosis provenance as unverified, identifies the boundary correlations as corroborative rather than independent proof, and says the candidate key is not authorized as a safe split key. [`docs/ADHD_EEG_PREPROCESSING_AUDIT.md`](ADHD_EEG_PREPROCESSING_AUDIT.md), §§4–6, 9, 14, preserves the same conservative limitations. For evaluation readiness, this audit follows the limitation-aware evidence and does not treat the earlier validation verdict as proof.

### 3.3 Existing evaluation and preprocessing decisions

- [`docs/SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md`](SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md), §§3–6, requires verified participant identity as the split unit; its historical summary of the Sprint 1 report's stronger claim is explicitly not accepted as verification. The current candidate composite key is not eligible for real-data partitions. The exact method, fold count, held-out test, seed, balance method, and validation strategy also remain undecided.
- [`docs/SPRINT_2_DECISION_ANALYSIS.md`](SPRINT_2_DECISION_ANALYSIS.md), items 11–17, identifies the subject grouping invariant and unresolved protocol choices. It does not establish that the current key is verified.
- [`docs/ADHD_EEG_PREPROCESSING_AUDIT.md`](ADHD_EEG_PREPROCESSING_AUDIT.md), §§8–10, describes participant grouping and training-only fitting as future requirements, not completed operations.
- [`docs/ML_EXECUTION_PLAN.md`](ML_EXECUTION_PLAN.md), §§1 and 4 (Gate 3), labels candidate split approaches as proposed and requires a frozen strategy before training. The plan’s proposed stratified grouped folds are not evidence that current participant labels are verified.
- [`docs/base-paper-analysis.md`](base-paper-analysis.md), §§2.1 and 8, reports cohort-level diagnoses and trial counts and describes the paper’s trial-level train/test/validation quantities. It does not document a subject-disjoint partition or provide a local participant-keyed diagnostic table.

## 4. Evidence inventory

| Claim or relationship | Classification | Evidence and limitation |
|---|---|---|
| Seven EEG arrays have reported shape `(385, 56, trials)`, `float32`, with 5,000 trials in `d1`–`d6` and 3,902 in `d7`; total 33,902. | **Verified** (as repository-reported structural observations) | [`docs/SPRINT_1_IMPLEMENTATION_AUDIT.md`](SPRINT_1_IMPLEMENTATION_AUDIT.md), §§2 and 4; [`docs/ADHD_EEG_PREPROCESSING_AUDIT.md`](ADHD_EEG_PREPROCESSING_AUDIT.md), §§3–4. This verifies dimensions/counts, not an ordering join. This audit did not rescan the binaries. |
| `y_stim` has 33,902 columns; row 0 contains 144 contiguous runs in three resetting blocks; rows 1–3 are one-hot and block-constant in those columns. | **Verified** (as repository-reported metadata structure) | [`docs/trial_subject_mapping_validation.md`](trial_subject_mapping_validation.md), §§5 and 9; [`scripts/validate_trial_subject_mapping.py`](../scripts/validate_trial_subject_mapping.py), `validate_y_stim()`. Structure does not establish that a column belongs to a particular EEG trial or define the one-hot semantics. |
| Three `sub_name_stim` blocks contain 44, 52, and 48 unique filenames, with `controls`, `subtype1`, and `subtype2` prefixes. | **Verified** (as repository-reported filename structure) | [`docs/trial_subject_mapping_validation.md`](trial_subject_mapping_validation.md), §6; validation script `validate_sub_name_stim()`. A filename block’s ordinal is not an authoritative key to row-0 participant index. |
| Numeric concatenation of `d1`–`d7` corresponds to `y_stim` column order for every trial. | **Candidate / Unresolved** | Equal counts and chronology are consistent with this join; no trial ID, keyed metadata reference, or independent per-trial crosswalk is documented. See [`docs/ADHD_EEG_PREPROCESSING_AUDIT.md`](ADHD_EEG_PREPROCESSING_AUDIT.md), §4; [`docs/SPRINT_1_IMPLEMENTATION_AUDIT.md`](SPRINT_1_IMPLEMENTATION_AUDIT.md), §4. |
| Row-0 runs correspond to real participants, and each filename ordinal identifies its corresponding run. | **Supported indirectly / Unresolved** | Run structure is observed; filename block lengths/order are consistent with the proposed pairing. No independent participant key links the filename at ordinal *i* to row-0 index *i*. See [`docs/trial_subject_mapping_validation.md`](trial_subject_mapping_validation.md), §§5–6 and 8; [`docs/SPRINT_1_IMPLEMENTATION_AUDIT.md`](SPRINT_1_IMPLEMENTATION_AUDIT.md), §4. |
| Every EEG trial can currently be assigned to a **verified** participant. | **Blocked** | The manifest assigns each trial a deterministic composite key, but its EEG-to-column and column-to-identity joins are not independently confirmed. No candidate ID should be described as a verified participant identifier. |
| Paper cohort contains 44 HC, 52 ADD, and 48 ADHD participants; the paper reports diagnosis/classification by clinical professionals and matching trial totals. | **Supported indirectly** for local block interpretation; **Verified** only as a statement in the paper record | [`docs/base-paper-analysis.md`](base-paper-analysis.md), §2.1; [`docs/trial_subject_mapping_validation.md`](trial_subject_mapping_validation.md), §§5 and 9. Agreement of aggregate counts does not attach a diagnosis to an identified local participant. |
| Participant-level diagnostic labels exist locally and can be authoritatively joined to every EEG trial. | **Unresolved / Blocked** | No participant-level, identity-keyed diagnostic table or authoritative trial-to-diagnosis crosswalk is identified in the inspected repository. The `y_stim` rows are not a substitute. |
| `y_stim` rows 1–3 are clinical diagnoses. | **Not supported; must not be assumed** | They are observed one-hot indicators aligned to metadata blocks; their task meaning is unresolved. The data contract explicitly prohibits using them as diagnostic targets. See [`docs/data_contract.md`](data_contract.md), §§4 and 7. |
| Candidate participant counts by block are 44/52/48; candidate trial counts are 10,129/13,031/10,742; candidate trial counts per subject range 46–295. | **Verified** as metadata run/block counts; **Candidate** as clinical class distribution | [`docs/trial_subject_mapping_validation.md`](trial_subject_mapping_validation.md), §§5 and 9–10. Counts can inform conditional design only; they do not verify class labels or trial membership. |

### 4.1 Exact independent evidence still missing

1. **EEG trial ↔ metadata column:** An authoritative source record or reproducible original construction procedure establishing the order of every trial in `d1`–`d7` relative to every `y_stim` column. A statement that large arrays were split into files or equal total counts is insufficient unless it also defines the exact order and offset.
2. **Metadata run ↔ participant identity:** A participant-level key joining each row-0 run to a stable participant identifier and, if filename identities are required, linking that identifier to the corresponding `sub_name_stim` filename. Sorted filenames and matching block sizes do not prove ordinal identity.
3. **Participant ↔ diagnosis:** An authoritative clinical/source record or data dictionary with diagnosis per stable participant identifier, diagnostic-group definitions, and the key connecting those records to the candidate participant identities. Aggregate paper counts are not enough.
4. **Joined trial eligibility:** Evidence that the verified participant key and verified diagnostic label attach to the same EEG trial locator through the authoritative trial join, with coverage, uniqueness, and consistency checks over all eligible trials.
5. **Protocol decisions:** Researcher approval of the estimand, exact grouped split design, fold/holdout and validation/model-selection boundaries, seed/repeat policy, balance/weighting approach, evaluation unit, and metrics. These are design decisions rather than facts that can be inferred from current files.

## 5. Diagnostic-label readiness

The paper record reports three groups and their aggregate sizes, and describes clinical diagnosis/classification in the cohort ([`docs/base-paper-analysis.md`](base-paper-analysis.md), §2.1). Filename prefixes and exact aggregate cohort/trial count agreement support an interpretation of the three metadata blocks as controls, subtype1/ADD, and subtype2/ADHD. This is indirect cohort-level consistency, not a participant-level label source.

The local `y_stim` matrix is not a diagnostic-label table: row 0 is an observed index-like value and rows 1–3 are one-hot patterns constant across the three blocks. No trial-wise stimulus category changes are observed in those indicator rows, and their physical category semantics are not documented. Even if those rows are collinear with candidate diagnostic blocks, using them as diagnosis would circularly elevate the block interpretation to a target without verifying individual identity or diagnosis.

**Conclusion:** Three-class supervised classification is **not currently defensible**. It becomes eligible for consideration only after the participant diagnosis and all EEG/metadata/participant joins are independently established and the diagnosis target is explicitly sourced from that authoritative record. The experiment must also freeze whether the target is three-class HC/ADD/ADHD and how any exclusions or ambiguous diagnoses are handled.

## 6. Participant-aware evaluation protocol comparison

The required invariant for an unseen-participant claim is that no verified participant contributes trials to more than one of train, validation, or test in a given evaluation. This invariant cannot be enforced reliably until the trial-to-participant mapping is verified.

| Protocol | Independence and balance | Advantages | Limitations and required metadata | Readiness here |
|---|---|---|---|---|
| **Leave-One-Subject-Out (LOSO)** | Hold out all trials for one verified participant per outer fold. Independence is direct at participant level. Each test fold contains one participant and therefore only one diagnostic class if diagnosis is participant-level. | Uses each participant for testing once; no arbitrary participant holdout fraction; class-wise out-of-fold predictions can be aggregated across held-out participants. | Requires verified participant IDs, one verified diagnosis per participant for supervised evaluation, a prespecified inner validation/tuning plan, and reporting that respects participant-level clustering. 144 outer fits may be costly; single-subject folds are noisy and cannot each estimate three-class balance. Test folds must not be used to tune. | **Blocked now.** Conditionally feasible if the 144-subject identity and class records are verified. |
| **Grouped K-Fold / GroupKFold** | Assign whole participant groups to folds; class proportions are not guaranteed. | Simple subject-disjoint folds; supports out-of-fold prediction for all participants and is less computationally intensive than LOSO. | Requires verified grouping IDs. With unequal class counts or trial counts, folds may be imbalanced; the grouping constraint alone does not stratify. Fold count and any fixed final test strategy require approval. | **Blocked now.** Conditionally feasible, but not the strongest balance option if verified class labels allow stratification. |
| **Stratified Group K-Fold / StratifiedGroupKFold** | Assign whole verified participants to folds while attempting to preserve class proportions. Stratification should represent participants, not treat each trial as an independent class-bearing unit. It cannot guarantee exact balance, especially for trial counts or other covariates. | Best match among these candidates to a three-class, participant-independent design if each participant has one verified class and class counts support the chosen fold count. Candidate participant counts 44/52/48 suggest approximate balance may be feasible after verification. | Requires verified participant IDs and per-participant diagnostic labels; fold count must be compatible with the smallest class and the chosen validation/test architecture. It may trade exact stratification for non-overlap; check realized folds before training. The project dependency list does not declare scikit-learn, so implementation availability/version must be checked when implementation is approved. | **Blocked now. Conditional preferred candidate** if independent mapping and labels confirm the reported class sizes and the research team approves the fold design. |

### Recommendation, conditional on resolving evidence gaps

Prefer **participant-level stratified grouped cross-validation** as the initial candidate because the stated research objective is generalization to unseen participants and the apparent participant-level class sizes are reasonably similar. That recommendation is conditional on verifying identity, diagnosis, and one diagnosis per participant; confirming the actual class distribution; and prespecifying the outer evaluation and inner validation/model-selection scheme. Stratification must target the intended balance: if folds are stratified using one row per trial with repeated subject labels, variable trial counts can balance trial totals rather than participant counts. If balancing participant counts is intended, assign folds from a verified one-row-per-participant table (or use an explicitly justified group-count objective) and propagate each assignment to all of that participant’s trials. It is not approval of a particular fold count or of `StratifiedGroupKFold` as an implementation.

LOSO remains a defensible alternative if exhaustive per-participant out-of-fold evaluation is the desired estimand and compute/model-selection can be handled without using each outer test participant for tuning. GroupKFold remains viable if simpler group-only partitioning is preferred and class-balance variation is acceptable and reported. The available evidence does not justify choosing among exact test architectures or freezing a seed today.

## 7. Leakage threat analysis and correct operation order

| Threat | Risk | Required control |
|---|---|---|
| Trial-level random split | Repeated trials from one participant can appear in both train and test, allowing participant-specific signals to masquerade as generalization. | Reject for participant-independent claims. Split on verified participant IDs, then assign every trial from that participant together. |
| Unverified participant mapping | A splitter can enforce separation only for the supplied key; a wrong/candidate key gives a false guarantee. | Do not use current `TrialManifest.candidate_participant_id` for research partitions until its joins are independently verified and the contradictory mapping verdicts are reconciled. |
| Preprocessing fitted before splitting | Whole-dataset statistics or learned transforms can carry held-out information into training. | Establish outer groups first. Fit every data-dependent operation within training data for that fold; apply only its fitted parameters to the fold’s validation/test subjects. |
| ICA and normalization | ICA component selection, scaling parameters, clipping thresholds, or other adaptations may use held-out signal distributions or labels. | Do not perform these under the current audit. If later approved, define whether and how each operation is fitted per training fold; keep test labels/outcomes unavailable. Any transductive unsupervised treatment of held-out signals must be explicitly justified and predeclared rather than silently assumed safe. |
| Feature selection and derived features | Choosing channels, bands, features, or extraction settings after seeing held-out outcomes biases performance. | Fix deterministic definitions in advance; fit/select data-driven features using only training folds and nested validation. Do not assign channel identities without resolving the 56-to-60 mapping. |
| Hyperparameter/model selection | Repeated decisions based on validation/test performance leak evaluation information. | Use only the prespecified inner train/validation procedure for tuning and stopping. Keep outer test subjects untouched until the complete model and analysis plan are frozen. |
| Validation/test contamination | Reusing validation/test subjects for training, preprocessing decisions, threshold selection, early stopping outside the declared design, or repeated design changes invalidates the intended separation. | Record distinct participant sets and permitted decisions per split. Evaluate the final held-out set only after choices are frozen; do not feed test results back into design. |
| Duplicate/correlated trials and repeated sessions | Exact/near duplicates or correlated trials may cross partitions if grouping is trial-based; within-participant dependence also makes trial counts an unreliable independent sample size. | Keep all trials and sessions for a participant together. Before a future run, document whether duplicate/overlap screening is needed, how it is performed, and any cross-participant duplicates found; never use trial-level randomization to break participant dependence. |
| Unequal trials per participant and class | Trial-weighted loss/metrics can let participants with more trials dominate; candidate counts vary substantially by participant. | Prespecify the metric unit and any class/subject weighting. Estimate weights/sampling from training data only; report participant- and class-aware results and trial counts. |

**Correct order for a future participant-independent experiment:**

1. Freeze the authoritative source snapshot and its provenance; validate complete trial locators and their one-to-one metadata/participant/diagnosis joins.
2. Freeze the research question, target, unit of evaluation, split protocol, metrics, and permitted preprocessing/model-selection decisions.
3. Construct participant groups and assign whole participants to outer train/validation/test folds. Verify no participant overlap and record the assignment before training.
4. Within each outer training set, create any inner participant-grouped validation/tuning splits. Keep outer test participants and labels out of all fitting and selection.
5. Fit any approved adaptive preprocessing, normalization, artifact thresholds, feature selection, weights, or other learned parameters using only the applicable training partition. Apply those fixed choices to validation/test data without refitting on held-out outcomes.
6. Select model and stopping/hyperparameters only under the prespecified inner procedure. Freeze all choices before evaluating the outer test fold.
7. Evaluate once per outer test fold, preserve subject-level grouping in analysis, and report the prespecified trial- and/or participant-level metrics, uncertainty, realized class balance, and exact fold manifest.

This order is a future control specification, not authorization to transform data or train a model now.

## 8. Proposed split-manifest design

The manifest should be an auditable sidecar keyed by the immutable trial locator. Store source snapshot identity and provenance so assignments can be reproduced without embedding EEG signals in the manifest.

| Field | Requirement | Use and status rule |
|---|---|---|
| `trial_id` | **Required, verified source locator** | Stable unique locator such as source filename plus zero-based within-file index. Identifies a stored slice only; does not itself verify metadata alignment. |
| `source_filename`, `within_file_trial_index` | **Required, verified source locator fields** | Must resolve to exactly one trial in the recorded source snapshot. |
| `source_snapshot_id` / source hashes | **Required for a generated research manifest** | Identifies the immutable input version used. Hashes and file inventory must be captured before a future split run. |
| `verified_participant_id` | **Required and verified before participant-aware splitting** | Stable identity from an authoritative participant key. Candidate block/index values must not populate this field. |
| `verified_diagnostic_label` | **Required and verified for supervised diagnostic classification** | Participant-level label from the authoritative diagnosis source, linked through verified identity. Null/not applicable for unlabeled structural splits. Never source from `y_stim` one-hot indicators. |
| `fold_id` | **Required after protocol assignment** | Outer fold identifier; unassigned until generator run. For a fixed holdout scheme, identify the protocol-defined fold/replicate as applicable. |
| `split_assignment` | **Required after protocol assignment** | Enumerated role such as `train`, `validation`, or `test`; validation may require separate inner-fold records if nested. |
| `mapping_provenance_id` and status | **Required** | Reference the evidence record for EEG-to-metadata, metadata-to-participant, and participant-to-diagnosis joins; each relationship needs its own source and verification status. |
| `protocol_version` | **Required** | Version/hash of the approved protocol describing split unit, algorithm, fold count, balance objective, evaluation unit, and exclusion rules. |
| `random_seed` | **Required for randomized assignment** | Record the actual seed and relevant software/library versions. A seed cannot compensate for unverified labels or groups. |
| `candidate_participant_id`, `candidate_diagnostic_label`, `candidate_metadata_column` | **Optional, clearly marked candidate-only** | May be retained in a separate investigation artifact to document hypotheses. Must not drive assignment or target creation and must never be serialized as verified fields. |

No split generator or manifest file is implemented by this audit.

## 9. Blockers, dependencies, and Sprint 2 entry criteria

### Current blockers

1. **Trial alignment:** No authoritative evidence links all EEG trial locators to `y_stim` columns in the asserted order.
2. **Participant grouping:** No independent key proves that each candidate row-0 run maps to a real participant and, where needed, a specific filename. Thus not every EEG trial is currently assignable to a **verified** participant.
3. **Diagnosis:** No participant-level, identity-keyed clinical diagnosis source is identified in the inspected repository, nor a verified link from that source to EEG trials. Three-class supervised classification is blocked.
4. **Conflicting records:** The earlier mapping-validation report and script overstate validation; the Sprint 1 implementation audit and preprocessing audit preserve unresolved mapping/diagnosis limitations. Reconcile this in the reviewed evidence record before implementation.
5. **Protocol:** Exact outer split/test design, validation and model-selection boundaries, fold count, seed/repeat policy, weighting/balancing, duplicate handling, and reporting metrics are not frozen or approved.

### Precise entry criteria before Sprint 2 split implementation

All of the following must be met and reviewed before implementing or running a research split generator:

1. An authoritative, reproducible EEG-file/trial-to-metadata mapping establishes exact coverage and offsets for every included trial; independent checks show no missing, duplicate, or ambiguous joins.
2. An authoritative participant key links each included trial to a verified stable participant identity. The split grouping key is unique per participant across all records and cannot be inferred solely from deterministic index construction.
3. For supervised evaluation, a source-backed participant-level diagnosis table defines HC/ADD/ADHD labels and links every included participant to one verified target (or specifies documented exclusions/ambiguous cases). The target is not taken from `y_stim` rows 1–3.
4. The mapping report, data contract, and Sprint 2 documents use consistent evidence statuses and explicitly resolve the previous unsupported “validated/safe split key” conclusions.
5. Researchers approve and version the exact evaluation design: estimand; outer grouped protocol and fold/holdout structure; inner validation/model-selection design; stratification target; seed and repeats; inclusion/exclusion rules; class/participant balancing; metrics and evaluation unit; and restrictions on test-set use.
6. The implementation contract defines source snapshot/provenance, manifest schema, failure behavior for null/duplicate/unverified keys, group-disjointness assertions, class-distribution reporting, and reproducibility/version recording.
7. The approved plan has synthetic tests for coverage, disjointness, reproducibility, label/group consistency, and invalid or missing metadata, plus a review of the realized folds before any model training.

These are entry conditions for implementation, not evidence that Sprint 2 is complete and not approval for Sprint 3 model work.

### Work that can proceed without verified identities or labels

- Read-only data shape, dtype, count, and source-integrity checks; preserve source-local trial locators.
- Unlabeled bounded EEG loading and synthetic tests of the existing data-access path.
- Provenance investigation with data custodians/original source scripts and reconciliation of the conflicting mapping documents.
- A written, reviewable split/metric/leakage specification and a proposed schema such as the one above.
- Generic validation concepts and synthetic-only test cases that use fabricated participants/classes, provided they are not represented as real-dataset folds or performance evidence.

Do not assign real participants/groups, generate real train/validation/test partitions, manufacture labels, preprocess signals, or train/evaluate a classifier until the applicable entry criteria are approved.

## 10. Required conclusions

| Question | Conclusion | Classification |
|---|---|---|
| Is the dataset currently ready for participant-independent evaluation? | **No.** The current participant mapping is deterministic but not independently verified, and diagnostic links are absent/unverified. | **Blocked** |
| Can we safely implement a split generator now? | **No, not for research partitions on this dataset.** The splitter would otherwise enforce separation only on candidate identifiers and labels. Synthetic-only generic tests/specification work may proceed. | **Blocked** |
| What exact evidence is missing? | Authoritative all-trial EEG-to-metadata ordering; verified participant key and row-run/filename link; participant-level diagnosis linked to that key and each EEG trial; reconciliation of conflicting audit verdicts; and approved protocol decisions. | **Unresolved / Blocked** |
| Which evaluation protocol is most appropriate, conditional on resolving missing evidence? | Participant-level stratified grouped cross-validation is the preferred candidate if verified participant labels and actual class counts support it. LOSO and unstratified GroupKFold remain viable alternatives for different prespecified estimands. Exact folds/test design remain open. | **Candidate** |
| What are the precise Sprint 2 implementation entry criteria? | Satisfy all seven criteria in §9, including verified joins, authoritative participant diagnoses for supervised work, reconciled evidence statuses, and approved/versioned evaluation design. | **Blocked until criteria met** |
| What work can proceed without diagnostic labels or verified participant mapping? | Unlabeled structural/integrity inspection, source-local access/testing, provenance research, protocol documentation, and synthetic-only validation design. | **Verified as allowed by current scope** |

## 11. Recommended next action

Do not implement the split generator yet. Obtain the original data-custodian documentation or source construction code that defines the EEG/metadata ordering and participant key, together with a participant-keyed diagnosis record and explicit join procedure. Then update and reconcile the mapping evidence record, validate complete joined coverage, and seek review/approval of the exact participant-aware evaluation protocol. Only after those gates pass should Sprint 2 implementation begin.
