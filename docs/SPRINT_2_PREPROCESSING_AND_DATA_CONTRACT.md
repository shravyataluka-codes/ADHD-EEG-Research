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

**Sprint 1 evidence note:** The Sprint 1 report (§§7–9, 12, 15–16) declares the sequential EEG-file-to-column mapping and trial-to-subject/group mapping validated, citing all six boundary correlations, contiguous subject runs, and exact cohort-count agreement. It classifies filename ordinal-to-index linkage as strongly supported, not validated. Earlier audit text still calls the trial join unproven ([ADHD_EEG_PREPROCESSING_AUDIT.md](ADHD_EEG_PREPROCESSING_AUDIT.md), §§4–5, 13; [ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md](ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md), §§5–6). For this contract, use the later, dedicated Sprint 1 report's validated composite key while retaining the filename-linkage caveat and recording the contradictory audit text for reconciliation before implementation.

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
| Split unit | **Subject is the required grouping unit** for any claim about generalization to unseen participants. Keep every trial assigned to a subject in one partition. This is a methodological requirement, not a claim about the base paper. The Sprint 1 report supplies the composite key `(<clinical-group block>, <within-block subject index>)`; its filename ordinal mapping remains strongly supported only. |
| GroupKFold / StratifiedGroupKFold | Grouped splitting is required in principle. Neither `GroupKFold` nor `StratifiedGroupKFold` is selected as the exact method. The 5-fold stratified proposal in [ML_EXECUTION_PLAN.md](ML_EXECUTION_PLAN.md), §1D, and the Sprint Map is explicitly proposed, not frozen. |
| Number of folds | **UNRESOLVED / DECISION REQUIRED.** Five folds is a proposal, not an evidence-backed frozen value. |
| Held-out test set | **UNRESOLVED / DECISION REQUIRED.** The paper analysis records that He et al. describe 6,000 test trials and five-fold validation over the remainder, but do not establish subject grouping. This paper procedure does not determine our test design. See [base-paper-analysis.md](base-paper-analysis.md), §8, and [ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md](ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md), §10. |
| Random seed | **UNRESOLVED / DECISION REQUIRED.** No seed is justified or frozen in the inspected evidence. |
| Class balancing | **UNRESOLVED / DECISION REQUIRED.** Subject counts and trial counts differ by class, and per-subject trial counts vary. Decide whether to weight, resample, or use subject-level aggregation; any estimated weights/sampling parameters must use training data only. See Sprint 1, §§9–10; preprocessing audit, §§8–10. |
| Validation strategy | **UNRESOLVED / DECISION REQUIRED.** Choose how validation is nested or otherwise separated from training and any final test set. Validation subjects cannot also be used for fitting or final model selection outside the prespecified procedure. |

The paper's split description does **not** prove participant overlap, trial-level leakage, or absence of leakage. Repository audits say the split unit is unspecified; therefore this document makes no claim that the original study leaked. See [base-paper-analysis.md](base-paper-analysis.md), §§8, 10–12, and [ML_EXECUTION_PLAN.md](ML_EXECUTION_PLAN.md), §§1B, 1D.

## 4. Required leakage controls

- **Subject assignment:** Attach the Sprint 1 composite subject key and group before partition assignment. Use the composite key, not row-0 index alone. Keep all trials for a subject in one partition; do not split individual trials across partitions. Resolve the documented filename-order caveat before relying on filename identities.
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
| Subject metadata | Globally unique key composed from group block and within-block subject index, as documented in Sprint 1. Filename identity is optional and remains strongly supported, not independently verified. |
| Clinical group metadata | `HC`, `ADD`, or `ADHD` from the Sprint 1 block mapping. Keep this metadata separate from EEG input; `y_stim` rows 1–3 are not to be used as diagnostic targets. |
| Split unit | Subject key. No subject may occur in multiple partitions. Exact splitter, folds, seed, test design, and validation protocol remain unresolved. |
| Preprocessing status | Raw trial values unchanged. No preprocessing has been run or is approved by this document. |
| Allowed before Gate 2 approval | Read-only structural/integrity checks, bounded loading for inspection, and non-mutating attachment of Sprint 1 metadata for contract validation. No persisted processed EEG tensors. |
| Prohibited | Filtering, notch filtering, ICA, cropping/resampling/re-epoching, baseline correction, normalization, artifact rejection/repair, channel reordering/mapping, PSD/TBR or other derived features, full-dataset statistics, model construction, and training/evaluation. |

Still unresolved before implementation: physical interpretation of the 385th position; amplitude units/reference; channel identity map; downloaded-file ICA state; rationale and parameters for any signal-changing operation; whether normalization or spectral features are needed; filename ordinal confirmation; fold count and exact grouped splitter; held-out test design; random seed; class-balancing method; and validation/model-selection design. The split and preprocessing decisions must be completed and reviewed before derived-data construction or training.

## 6. Gate 2 verdict

Gate 2 requires an explicit, evidence-based decision for every preprocessing choice before transformation. This record freezes preservation/no-crop handling and rejects unsupported default transformations, but multiple operations remain deferred and require research decisions; the exact split and validation protocol also remain undecided. A decision record is not team sign-off, and unresolved items cannot be promoted to approved methodology by documenting them.

Decisions/evidence still required:

- Determine whether any filtering, notch filtering, ICA, baseline correction, artifact handling, normalization, or spectral-feature experiment is scientifically justified; if approved, specify exact parameters and fitting scope.
- Establish source preprocessing state, signal units/reference, 385th-position meaning, and whether channel mapping is needed for the planned experiment.
- Resolve/reconcile the older audit's mapping caveat with the Sprint 1 validation record, especially before using filename identities.
- Freeze the exact subject-grouped split method, number of folds, held-out test design, seed, class-balancing approach, and validation/model-selection protocol; obtain required team/researcher approval.

GATE 2 NOT PASSED