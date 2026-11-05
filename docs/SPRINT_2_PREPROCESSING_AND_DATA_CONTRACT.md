# Sprint 2 — Preprocessing, Data Contract, and Split Strategy

**Branch:** `ml/transformer-development`  
**Sprint:** 2 — Decision/freeze only  
**Status:** Decision record; no preprocessing or derived EEG data created  
**Gate:** Gate 2

## Purpose and scope

This document records what repository evidence supports, what is permitted as a no-op data contract, and what still requires a research decision. A proposal in a roadmap or guide is not treated as methodological evidence. Nothing here authorizes model training or creation of processed EEG tensors.

Evidence sources are the Sprint 1 report, [trial_subject_mapping_validation.md](trial_subject_mapping_validation.md); [ADHD_EEG_PREPROCESSING_AUDIT.md](ADHD_EEG_PREPROCESSING_AUDIT.md); [dataset-preprocessing-audit.md](dataset-preprocessing-audit.md); [ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md](ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md); [base-paper-analysis.md](base-paper-analysis.md); [ML_EXECUTION_PLAN.md](ML_EXECUTION_PLAN.md); and [PROJECT-SPRINT-MAP.md](PROJECT-SPRINT-MAP.md).

## 1. Raw data contract evidence

| Contract item | Classification | Evidence and limit |
|---|---|---|
| Trial array shape | **VALIDATED** | All seven EEG arrays are reported as `(385, 56, trials)` in the direct binary inspection summarized in Sprint 1, §4, and the preprocessing audit, §3–4. A trial is therefore represented by the observed `(385, 56)` slice. |
| Stored dtype | **VALIDATED** | EEG arrays are reported as MATLAB `single` / `float32`; see Sprint 1, §4, and the preprocessing audit, §3. |
| Sample count | **VALIDATED** | 385 stored positions per trial in all seven arrays; this validates count, not the time coordinate or endpoint meaning. |
| Channel count | **VALIDATED** | 56 data columns in every EEG array; see Sprint 1, §4. |
| Sampling frequency | **STRONGLY SUPPORTED** | 256 Hz is explicitly reported by He et al. and recorded in [base-paper-analysis.md](base-paper-analysis.md), §4. It is not encoded in the inspected EEG array headers. |
| Duration | **STRONGLY SUPPORTED** | The paper reports 1.5-second experiments/trials. The duration represented by the stored 385 positions is not independently established: 1.5 × 256 implies 384 sample intervals, while the array contains 385 positions. |
| Axis interpretation | **STRONGLY SUPPORTED** | The raw dimensions and paper's channel-by-sampling-point description and `(385, 56)` model input support axis 0 = sample/time position and axis 1 = channel. The binary headers do not name axes; preserve the stored order and avoid stronger physical claims. |
| Amplitude units and reference | **UNRESOLVED** | The audits state that units and reference scheme are not established. A value range labelled µV in the Sprint 1 report does not establish acquisition units or reference provenance. `chan.mat` metadata alone does not establish how the 56 array columns were acquired. |
| Meaning of position 385 | **UNRESOLVED** | The paper's nominal duration/rate and reported model length conflict by one position; neither paper nor data headers explains whether it is an endpoint or another convention. Do not crop or reinterpret it. |
| 56-channel identity/mapping | **UNRESOLVED** | `chan.mat` has 60 channel entries and EEG arrays have 56 columns. The correspondence and four absent entries are unknown; do not infer their identities. |

**Sprint 1 evidence reconciliation:** The Sprint 1 mapping report historically called the sequential EEG-file-to-column and trial-to-subject/group mappings validated, citing boundary correlations, contiguous row-0 runs, and aggregate count agreement. The later evaluation-readiness, evidence-reconciliation, provenance, and external-provenance audits find that these observations do not establish an independent join. This contract does **not** accept the earlier validated-key conclusion: the composite key remains a candidate participant mapping, the EEG-locator-to-`y_stim` correspondence remains unresolved, and the candidate diagnostic-group mapping is not a verified clinical diagnosis. The original report's stronger verdict is retained here as a historical finding, not an accepted status.

### Terminology boundary

- **Source-local trial identifier:** `(source_filename, within_file_trial_index)` locates an array slice in the supplied files. It does not identify a participant or diagnosis.
- **Candidate participant mapping:** The deterministic composite key formed from a configured group block and `y_stim` row-0 value. Neither the row-0 interpretation nor its correspondence to an EEG locator is independently verified.
- **Candidate diagnostic-group mapping:** A configured block name such as `HC`, `ADD`, or `ADHD`; it is not an established clinical label for any local trial or participant.
- **Verified clinical diagnosis label:** No participant-level, identity-keyed diagnosis record linked to the local trial locators is currently available in the inspected evidence. `y_stim` rows 1–3 are not clinical targets.

## 2. Preprocessing decisions

Statuses describe the decision for the next implementation stage, not a claim that an operation has been run. **APPROVED** means the stated handling is allowed; **REJECTED** means the proposed transformation is not allowed under this contract; **DEFERRED** means no implementation until additional evidence and an explicit research decision.

| Operation | Decision | Evidence-based rationale |
|---|---|---|
| Epoch/segmentation handling | **APPROVED: preserve existing trial slices unchanged.** Re-epoching or changing boundaries is not approved. | Arrays are already trial-shaped; the paper reports 1.5-second trials but does not provide a reconstructable segmentation rule. Sprint 1 establishes an index mapping, not new epoch boundaries. See preprocessing audit, §§4.1, 4.7–8; base-paper analysis, §5. |
| Crop 385 → 384 | **REJECTED.** Preserve all 385 positions. | The extra position's meaning is unresolved, and the paper itself reports an input length of 385. No evidence justifies discarding a sample. See preprocessing audit, §§4.2, 8; base-paper analysis, §§4, 7.1. |
| Filtering | **DEFERRED.** No bandpass or other filter is approved. | The guide's 0.5–45 Hz proposal lacks source evidence and exact design/edge parameters. The paper does not provide a reproducible filter specification. See preprocessing audit, §§4.3–4.4; base-paper analysis, §5. |
| Notch filtering | **DEFERRED.** | No evidence establishes line frequency or mains contamination; 50/60 Hz proposals are not a justified default. See preprocessing audit, §4.5; base-paper analysis, §5. |
| ICA | **DEFERRED.** Do not rerun ICA. | The paper says source data underwent ICA but does not report its algorithm or component rules; local-file preprocessing state is unknown. Reprocessing could duplicate cleaning. See preprocessing audit, §4.6; base-paper analysis, §§5, 7. |
| Baseline correction | **DEFERRED.** | No baseline interval or source procedure is documented. See preprocessing audit, §4.10; base-paper analysis, §5. |
| Amplitude normalization | **DEFERRED.** No full-dataset normalization is permitted. | Scaling is not specified by the paper; per-trial scaling may alter amplitude information. Any later fitted statistics must be learned from training subjects only. See preprocessing audit, §§4.10, 9–10; methodology audit, §9. |
| Artifact rejection/repair | **DEFERRED.** | No quality labels, thresholds, or rejection rules are established. See preprocessing audit, §4 and §11; base-paper analysis, §§5, 7. |
| Channel mapping | **DEFERRED.** Preserve the 56 columns in stored order; do not assign electrode identities or spatial locations. | `chan.mat` has 60 entries versus 56 data columns, and the mapping is unresolved. See preprocessing audit, §§4.2, 6, 8; Sprint 1, §§4, 12, 14. |
| Spectral feature extraction (PSD/TBR) | **DEFERRED.** | Welch PSD bands and TBR are proposals, not established requirements; settings and scientific rationale are absent. See preprocessing audit, §§4.8–4.9; methodology audit, §§7, 12. |

No signal-changing transformation is approved. A future change to a deferred item requires a documented research rationale, complete parameters, and leakage-safe fitting/evaluation rules before implementation.

## 3. Data split contract

| Question | Evidence-supported decision |
|---|---|
| Split unit | For a future claim about generalization to unseen participants, the required unit is the **verified participant identity**, with every trial for that identity in one partition. The current composite candidate key cannot establish this requirement or be used as if it were verified. |
| GroupKFold / StratifiedGroupKFold | Grouped splitting is required in principle. Neither `GroupKFold` nor `StratifiedGroupKFold` is selected as the exact method. The 5-fold stratified proposal in [ML_EXECUTION_PLAN.md](ML_EXECUTION_PLAN.md), §1D, and the Sprint Map is explicitly proposed, not frozen. |
| Number of folds | **UNRESOLVED / DECISION REQUIRED.** Five folds is a proposal, not an evidence-backed frozen value. |
| Held-out test set | **UNRESOLVED / DECISION REQUIRED.** The paper analysis records that He et al. describe 6,000 test trials and five-fold validation over the remainder, but do not establish subject grouping. This paper procedure does not determine our test design. See [base-paper-analysis.md](base-paper-analysis.md), §8, and [ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md](ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md), §10. |
| Random seed | **UNRESOLVED / DECISION REQUIRED.** No seed is justified or frozen in the inspected evidence. |
| Class balancing | **UNRESOLVED / DECISION REQUIRED.** Subject counts and trial counts differ by class, and per-subject trial counts vary. Decide whether to weight, resample, or use subject-level aggregation; any estimated weights/sampling parameters must use training data only. See Sprint 1, §§9–10; preprocessing audit, §§8–10. |
| Validation strategy | **UNRESOLVED / DECISION REQUIRED.** Choose how validation is nested or otherwise separated from training and any final test set. Validation subjects cannot also be used for fitting or final model selection outside the prespecified procedure. |

The paper's split description does **not** prove participant overlap, trial-level leakage, or absence of leakage. Repository audits say the split unit is unspecified; therefore this document makes no claim that the original study leaked. See [base-paper-analysis.md](base-paper-analysis.md), §§8, 10–12, and [ML_EXECUTION_PLAN.md](ML_EXECUTION_PLAN.md), §§1B, 1D.

## 4. Required leakage controls

- **Participant assignment:** Only after an authoritative participant crosswalk is obtained and the evaluation protocol is approved, keep all trials for each verified participant in one partition. The current candidate composite key may be exercised only in synthetic tests; it must not be presented as a verified identity or used to create research partitions.
- **Preprocessing statistics:** Fit any data-dependent statistic or transform separately within each training partition/fold. Do not calculate parameters over all subjects before splitting.
- **Normalization:** If later approved, fit means, variances, clipping limits, or other scaling parameters using training subjects only, then apply those fixed parameters to that fold's validation/test subjects. No whole-dataset normalization.
- **Feature extraction:** Fix deterministic feature definitions and settings before held-out evaluation. Any data-driven band/channel/feature selection or learned transform must be fitted only within training data; do not inspect held-out labels or outcomes to choose features.
- **Model selection:** Use only the prespecified training/validation procedure. Do not use a final test set to choose preprocessing, architecture, stopping rules, or the reported model.
- **Hyperparameter tuning:** Restrict tuning to training/validation data under the approved grouped procedure. If a held-out test set is chosen, do not tune against it.
- **Test evaluation:** Freeze the full protocol and model-selection decisions before using test subjects. Evaluate the test set once for the final report; do not feed test results back into design choices. The exact test design remains unresolved.

These are controls required for a future leakage-safe experiment; they do not authorize any fitting, feature extraction, or training in Sprint 2.

## 5. Proposed next-stage data contract

This is a **no-transformation input contract**, not authorization to create a derived dataset.

| Field | Contract |
|---|---|
| Raw source | Immutable `dataset/d1.mat`–`dataset/d7.mat`; do not rewrite or replace source files. |
| Trial input | Observed shape `(385, 56)` in stored axis order: sample/time positions × anonymous channel columns. Preserve all 385 positions and all 56 columns. |
| Dtype | Preserve source `float32`; do not silently cast or scale. |
| Source-local trial identifier | `(source_filename, within_file_trial_index)` locates a trial slice; `trial_id` is its serialized form. These identifiers do not encode participant identity. |
| Candidate participant mapping | `candidate_participant_id`, currently derived from configured block and row-0 value. Candidate only; correspondence to each EEG locator and verified identity remain unresolved. |
| Candidate diagnostic-group mapping | `candidate_diagnostic_group`, currently the configured block name. This is not a verified clinical label. |
| Verified clinical diagnosis label | Not available from the inspected evidence. Do not assign diagnoses from candidate groups, filenames, aggregate counts, or `y_stim` rows 1–3. |
| Split unit | Verified participant identity, once an authoritative crosswalk and diagnosis evidence are available and the protocol is approved. No current real-data partition is authorized. |
| Preprocessing status | Raw trial values unchanged. No preprocessing has been run or is approved by this document. |
| Allowed before Gate 2 approval | Read-only structural/integrity checks, bounded loading for inspection, and non-mutating attachment of explicitly candidate Sprint 1 mappings for contract validation. No persisted processed EEG tensors. |
| Prohibited | Filtering, notch filtering, ICA, cropping/resampling/re-epoching, baseline correction, normalization, artifact rejection/repair, channel reordering/mapping, PSD/TBR or other derived features, full-dataset statistics, model construction, and training/evaluation. |

Still unresolved before implementation: physical interpretation of the 385th position; amplitude units/reference; channel identity map; downloaded-file ICA state; rationale and parameters for any signal-changing operation; whether normalization or spectral features are needed; filename ordinal confirmation; fold count and exact grouped splitter; held-out test design; random seed; class-balancing method; and validation/model-selection design. The split and preprocessing decisions must be completed and reviewed before derived-data construction or training.

## API naming and migration note

The public metadata names were changed to make their candidate status explicit: `subject_id` → `candidate_participant_id`, `clinical_group` → `candidate_diagnostic_group`, `y_stim_column` → `candidate_y_stim_column`, `iter_subject()` → `iter_candidate_participant()`, and `iter_group()` → `iter_candidate_diagnostic_group()`. No compatibility aliases are provided: the old unqualified names can be mistaken for verified identities, clinical diagnoses, or metadata correspondence. Callers must migrate to the candidate names and retain the provenance caveat.

The preprocessing review schema follows the same rule: `SubjectICAReview` is replaced by `CandidateParticipantICAReview`, and the JSON mapping key `subjects` is replaced by `candidate_participants`. Preprocessing metadata uses `candidate_participant_count` and `ica.candidate_participants`; the output explicitly labels the grouping key as candidate. These are API/schema changes, not claims that ICA behavior or grouping has been scientifically validated.

## 6. Gate 2 verdict

Gate 2 requires an explicit, evidence-based decision for every preprocessing choice before transformation. This record freezes preservation/no-crop handling and rejects unsupported default transformations, but multiple operations remain deferred and require research decisions; the exact split and validation protocol also remain undecided. A decision record is not team sign-off, and unresolved items cannot be promoted to approved methodology by documenting them.

Decisions/evidence still required:

- Determine whether any filtering, notch filtering, ICA, baseline correction, artifact handling, normalization, or spectral-feature experiment is scientifically justified; if approved, specify exact parameters and fitting scope.
- Establish source preprocessing state, signal units/reference, 385th-position meaning, and whether channel mapping is needed for the planned experiment.
- Resolve/reconcile the older audit's mapping caveat with the Sprint 1 validation record, especially before using filename identities.
- Freeze the exact subject-grouped split method, number of folds, held-out test design, seed, class-balancing approach, and validation/model-selection protocol; obtain required team/researcher approval.

GATE 2 NOT PASSED