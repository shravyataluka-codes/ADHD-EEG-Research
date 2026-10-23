# ML Execution Plan — ADHD-EEG Research
## Branch: `ml/transformer-development`

> **Document status:** Planning only. No model code, preprocessing code, or training has been performed.
> No existing research documents were modified to produce this plan.
>
> **Evidence conventions used throughout:**
> - `VALIDATED` — established by direct binary-data inspection, cross-referenced metadata, or the base paper.
> - `STRONGLY SUPPORTED` — multiple independent observations align; not yet independently proven.
> - `PROPOSED — NOT YET FROZEN` — appears in a teammate roadmap or master guide but lacks validated evidence.
> - `UNRESOLVED — EVIDENCE REQUIRED` — cannot be established from any currently available source.

---

## Files Inspected During This Audit

| File | Type | Notes |
|---|---|---|
| `docs/dataset-investigation.md` | Research doc | Full dataset structure audit; provides validated facts about shapes, dtypes, metadata |
| `docs/base-paper-analysis.md` | Research doc | 1,147-line deep-dive of He et al. (2023); catalogs 10 internal inconsistencies |
| `docs/ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md` | Research doc | ML pipeline reconstruction, leakage analysis, candidate research questions |
| `docs/ADHD_EEG_PREPROCESSING_AUDIT.md` | Research doc | Strictest current audit; most conservative evidence labeling; 12-item approval checklist |
| `docs/dataset-preprocessing-audit.md` | Research doc | Earlier preprocessing audit; step-by-step audit of proposed pipeline decisions |
| `docs/PROJECT-SPRINT-MAP.md` | Planning doc | Phase A–G sprint hierarchy; Phase A and B described in detail |
| `docs/sprint_1_research_landscape.md` | Research doc | Literature comparison matrix; identifies Vahid et al. (2019) as seminal paper; notes trial-level leakage in base paper |
| `docs/domain-understanding.md` | Terminology doc | Project-specific EEG glossary anchored to paper and dataset observations |
| `adhd-eeg-transformer-master-guide.md` | Teammate guide | High-level proposal for a Hybrid Spatial-Spectral Conformer; **NOT a frozen protocol** |
| `dataset_metadata_investigation.md` | Investigation | Earliest independent script-based inspection; confirms y_stim structure |
| `scripts/inspect_mat_files.py` | Script | h5py + scipy inspection of y_stim, sub_name_stim, chan.mat |
| `scripts/inspect_deep.py` | Script | Additional deep inspection script |
| `scripts/inspect_sub.py` | Script | Subject metadata inspection |
| `scripts/inspect_ystim_relations.py` | Script | y_stim relationship investigation |
| `scripts/dataset-investigation/*.py` | Scripts | Dataset-investigation suite (inspect_mat, inspect_metadata, check_chan, view_eeg, etc.) |
| `.gitignore` | Config | Confirms dataset/ is excluded from Git |
| `README.md` | Root readme | Project overview |

---

## Section 1 — Current ML State

### A. Completed and Validated

| Item | Evidence | Source |
|---|---|---|
| Raw dataset fully acquired locally | All 10 files present; byte sizes match OSF listing | `ADHD_EEG_PREPROCESSING_AUDIT.md` |
| d1–d6 shape: `(385, 56, 5000)` | Direct HDF5 header inspection, float32, gzip-chunked | Binary data — `VALIDATED` |
| d7 shape: `(385, 56, 3902)` | Direct HDF5 header inspection | Binary data — `VALIDATED` |
| Total trial count: 33,902 | d1–d6 × 5,000 + d7 × 3,902 = 33,902; equals y_stim column count | `VALIDATED` |
| y_stim shape: `(4, 33902)`, dtype float64 | Direct inspection | `VALIDATED` |
| y_stim row 0: within-block subject index | Values 1–52, reset per block; 144 contiguous runs, no gaps | `VALIDATED` |
| y_stim rows 1–3: mutually exclusive binary one-hot | Exactly one active per column; column sums = 10,129 / 13,031 / 10,742 | `VALIDATED` |
| sub_name_stim.mat: three blocks of 44 / 52 / 48 unique filenames | Direct HDF5 cell-array inspection | `VALIDATED` |
| Filename tokens: block 1 = `controls`, block 2 = `subtype1`, block 3 = `subtype2` | All filenames in each block share the respective prefix | `VALIDATED` |
| chan.mat: 60 channel entries with standard labels | Direct inspection (Cz, FCz, Fz, Fp1, Fp2, Oz, Iz, etc.) | `VALIDATED` |
| Channel count mismatch: chan.mat = 60, EEG arrays = 56 | Direct count comparison | `VALIDATED` |
| Dataset entirely excluded from Git | `.gitignore` contains `dataset/` | `VALIDATED` |
| Base paper is He et al. (2023), Journal of Neural Engineering, DOI 10.1088/1741-2552/acf7f5 | Paper confirmed | `VALIDATED` |
| Paper reports 144 subjects: 44 HC, 52 ADD, 48 ADHD | Paper section 2.1 | `VALIDATED` |
| Paper reports 33,902 trials; 10,129 HC / 13,031 ADD / 10,742 ADHD | Paper section 2.1 | `VALIDATED` |
| Paper's Transformer input shape reported as `(None, 385, 56)` | Table 1 | `VALIDATED` |
| Paper reports ICA as the only stated preprocessing step | Paper section 2.1 | `VALIDATED` |
| No model code has been implemented | Codebase review | `VALIDATED` |
| No preprocessing has been executed | Audit conclusion in ADHD_EEG_PREPROCESSING_AUDIT.md | `VALIDATED` |

### B. Completed but Requiring Final Validation

| Item | Current status | What must be confirmed |
|---|---|---|
| Trial → file → y_stim column join | `STRONGLY SUPPORTED` by exact count arithmetic | No independent trial-ID or checksum exists in EEG file headers; ordering inference not independently proven |
| Group label assignment: block 1 = HC, block 2 = ADD, block 3 = ADHD | `STRONGLY SUPPORTED` by filename tokens and matching paper counts | No explicit diagnosis field keyed to each trial exists in the MAT files |
| sub_name_stim filename ordinal ↔ row-0 index mapping | `STRONGLY SUPPORTED`; assumed linear ordering | Filename ordinal-to-row-0 join not independently proven |
| Seminal earlier work is Vahid et al. (2019) | Identified in sprint_1_research_landscape.md | Dataset lineage not fully traced through all published papers on this cohort |
| Paper used trial-level (not subject-level) splitting | `STRONGLY SUPPORTED` by paper description; paper does not state subject-aware splitting | Cannot be proven without the original paper's code |

### C. Unresolved

| Item | Classification |
|---|---|
| Physical meaning of the 385th sample (384 = 256 Hz × 1.5 s) | `UNRESOLVED — EVIDENCE REQUIRED` |
| Exact identities of the 4 channels absent from EEG arrays (chan.mat has 60, arrays have 56) | `UNRESOLVED — EVIDENCE REQUIRED` |
| Exact physical meanings of y_stim rows 1–3 (stimulus categories, not diagnoses) | `UNRESOLVED — EVIDENCE REQUIRED` |
| Whether downloaded EEG files are already ICA-cleaned or represent raw continuous data | `UNRESOLVED — EVIDENCE REQUIRED` |
| Whether the file-level trial ordering matches y_stim column ordering | `UNRESOLVED — EVIDENCE REQUIRED` (only inferred from count arithmetic) |
| Whether sub_name_stim block entries are in same ordinal order as y_stim row-0 indices | `UNRESOLVED — EVIDENCE REQUIRED` |
| Paper's true train/test/validation splitting unit (trial-level vs. subject-level) | `UNRESOLVED — EVIDENCE REQUIRED` (paper does not state) |
| EEG amplitude units, reference electrode scheme | `UNRESOLVED — EVIDENCE REQUIRED` |
| Whether trials overlap in the original segmentation | `UNRESOLVED — EVIDENCE REQUIRED` |
| Number of trials per subject (highly variable; ranges from 46 to 295 within groups) | `VALIDATED` as variable; implications for class weighting and evaluation `UNRESOLVED` |
| The paper's internal inconsistencies (144 vs 300 subjects, parameter count conflicts, ablation value conflicts) | `UNRESOLVED — EVIDENCE REQUIRED` |

### D. Proposed Future Work (Not Yet Frozen)

The following are proposals from the teammate master guide (`adhd-eeg-transformer-master-guide.md`) or sprint landscape report. They are **not frozen methodological decisions**.

| Proposal | Source | Status |
|---|---|---|
| 0.5–45 Hz Butterworth bandpass filter | Master guide | `PROPOSED — NOT YET FROZEN` |
| 50/60 Hz notch filter | Master guide | `PROPOSED — NOT YET FROZEN` |
| Repeat ICA for artifact removal | Master guide | `PROPOSED — NOT YET FROZEN` |
| Per-channel theta/beta ratio (TBR) feature | Master guide | `PROPOSED — NOT YET FROZEN` |
| Welch PSD spectral band extraction (5 bands) | Master guide | `PROPOSED — NOT YET FROZEN` |
| Hybrid Spatial-Spectral Conformer architecture | Master guide | `PROPOSED — NOT YET FROZEN` |
| Depthwise spatial convolution layer | Master guide | `PROPOSED — NOT YET FROZEN` |
| StratifiedGroupKFold(n_splits=5) subject-level split | Both documents | `PROPOSED — NOT YET FROZEN` (approach is directionally correct; exact parameters not frozen) |
| AdamW optimizer with cosine annealing | Master guide | `PROPOSED — NOT YET FROZEN` |
| MNE-Python preprocessing stack | Master guide | `PROPOSED — NOT YET FROZEN` |
| Scalp topomap visualization | Master guide | `PROPOSED — NOT YET FROZEN` |
| "Trial-level leakage in the base paper" as a confirmed research gap | Sprint landscape doc | `PROPOSED — NOT YET FROZEN` (assessment is methodologically sound but requires formal verification) |
| "80–90% accuracy improvement" or any performance claim | — | **EXPLICITLY PROHIBITED — DO NOT PREDICT** |

---

## Section 2 — ML Pipeline Dependency Graph

```
[Raw EEG: d1.mat – d7.mat]
         │
         │  PREREQUISITE: Establish that all 7 files are intact,
         │  share the same (385, 56) trial shape and float32 dtype.
         ▼
[Metadata Integration]
  y_stim.mat + sub_name_stim.mat + chan.mat
         │
         │  PREREQUISITE: Formal validation of the file→y_stim column
         │  join and the sub_name_stim filename→row-0 index join.
         ▼
[Deterministic Trial → Subject → Clinical Group Mapping]
  Global trial index | source file | within-file index
  | subject ID (globally unique composite key) | clinical group
         │
         │  PREREQUISITE: Validated mapping with no ambiguity;
         │  approval of the composite key and label construction rule.
         │  GATE 1 must pass before this can proceed.
         ▼
[Preprocessing Protocol Decision]
  Decide: filter? normalize? handle 385th sample? channel mapping?
         │
         │  PREREQUISITE: Mapping validated; preprocessing decisions
         │  documented and frozen BEFORE any code is written.
         │  GATE 2 must pass before this can proceed.
         ▼
[Derived Dataset Construction]
  Preprocessed trials with validated metadata attached;
  subject-level splits defined; leakage controls documented.
         │
         │  PREREQUISITE: Protocol frozen; no test-set information
         │  used in preprocessing or split parameter selection.
         │  GATE 3 must pass before split strategy is locked.
         ▼
[Data Split]
  Subject-level train / validation / test partition;
  globally unique subject key; frozen fold seed.
         │
         │  PREREQUISITE: Split must be frozen before any model sees data.
         ▼
[Baseline Models]
  Simple statistical classifier + paper's CNN baselines
  (EEGNet, ShallowConvNet) under the same subject-level split.
         │
         │  PREREQUISITE: Baselines must be trained and evaluated
         │  before any Transformer claim is made.
         │  GATE 4 must pass before Transformer training proceeds.
         ▼
[Transformer / Candidate Architecture]
  Architecture justified by evidence, not copied arbitrarily.
         │
         │  PREREQUISITE: Architecture decision documented and frozen;
         │  hyperparameters pre-specified; GATE 5 must pass.
         ▼
[Training]
  Reproducible training loop; seeds recorded; training-only
  fitting of any learned preprocessing parameters.
         │
         │  PREREQUISITE: Evaluation protocol frozen BEFORE training begins.
         │  GATE 6 must pass before training loop runs.
         ▼
[Subject-Level Evaluation]
  Primary metric: subject-level generalization.
  Evaluation unit: held-out subjects never seen during training.
         │
         │  PREREQUISITE: Test set never inspected before this step.
         ▼
[Explainability and Research Analysis]
  Attribution methods; attention analysis; channel/time saliency.
         │
         │  PREREQUISITE: Model evaluation is complete; claims are bounded
         │  by what attribution methods can legitimately establish.
         │  GATE 7 must pass before interpretability claims are published.
         ▼
[Research Report / Paper]
```

**Key rule:** No stage may begin until all prerequisites of the immediately preceding stage are met.
If a prerequisite cannot be established, the stage must not proceed and the blocker must be documented.

---

## Section 3 — Sprint Plan

> Each sprint ends with a Git commit boundary. No sprint may proceed until its predecessor's acceptance criteria are met.
> Sprints are listed in dependency order, not calendar order.

---

### Sprint 0 — ML Planning and Data Contract Definition

**Objective:** Establish the ML execution plan, freeze unresolved items as explicit gates, and define the data contract between stages before any code is written.

**Prerequisites:** Repository audit complete (all docs read); no new code needed.

**Inputs:**
- `docs/dataset-investigation.md`
- `docs/base-paper-analysis.md`
- `docs/ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md`
- `docs/ADHD_EEG_PREPROCESSING_AUDIT.md`
- `docs/dataset-preprocessing-audit.md`
- `docs/PROJECT-SPRINT-MAP.md`
- `docs/sprint_1_research_landscape.md`
- `adhd-eeg-transformer-master-guide.md`

**Outputs:**
- `docs/ML_EXECUTION_PLAN.md` ← **this document**

**Implementation tasks:**
- Audit all existing research documents (read-only)
- Synthesize validated facts vs. proposed-but-not-frozen items
- Define sprint structure and dependency order
- Define decision gates
- Define data contract

**Validation tasks:**
- Review each item in Section 1 against source documents for correct evidence labeling
- Confirm no new assumptions were introduced

**Audit questions:**
- Does every "VALIDATED" item have a direct source document with matching evidence?
- Does every "PROPOSED" item trace to a specific teammate document rather than to validated evidence?
- Are all 7 open gates correctly identified?

**Acceptance criteria:**
- Document created and reviewed by team
- No "VALIDATED" items were inferred; every one traces to binary data, paper, or script output
- Gates clearly block implementation if not passed

**Documentation required:** This document (`docs/ML_EXECUTION_PLAN.md`)

**Git commit boundary:** `docs: add ML execution plan (Sprint 0)`

---

### Sprint 1 — Trial → Subject → Clinical Group Mapping Validation

**Objective:** Produce a deterministic, independently verifiable crosswalk from every EEG trial to a globally unique subject identifier and clinical group label. This is GATE 1.

**Prerequisites:** Sprint 0 complete; no preprocessing yet.

**Inputs:**
- `dataset/d1.mat` – `dataset/d7.mat` (read-only, HDF5 headers + row-0 runs only)
- `dataset/y_stim.mat`
- `dataset/sub_name_stim.mat`
- Existing inspection scripts in `scripts/`

**Outputs:**
- `docs/trial_subject_mapping_validation.md` — formal evidence document
- A reproducible, version-controlled script that derives (but does NOT write out) the mapping for validation purposes only
- Mapping validation report: verdict of VALIDATED or UNRESOLVED with evidence

**Implementation tasks:**
1. Verify that all 7 EEG files have the expected internal variable names, shapes, and dtypes
2. Formally verify the sequential file-to-column mapping: confirm that d1 trial 1 corresponds to y_stim column 0 (or present evidence that it does not)
3. Formally verify the sub_name_stim filename ordinal ordering matches row-0 index ordering within each block
4. Construct the composite subject key: `(block_index, within_block_subject_index)` → `(group_label, filename)`
5. Produce per-subject trial counts and verify consistency with dataset-investigation.md totals
6. Attempt to locate any independent ordering evidence in the EEG file headers or OSF metadata

**Validation tasks:**
- Cross-validate total trial counts per group against paper-reported counts (10,129 / 13,031 / 10,742)
- Verify no subject index is missing or duplicated within any block
- Verify the 144 contiguous row-0 runs span all 33,902 columns with no gaps
- Attempt to find a counter-example to the sequential ordering assumption
- Document any evidence (positive or negative) about the y_stim row-1/2/3 physical meanings

**Audit questions:**
- Is there any independent cross-reference (hash, checksum, embedded trial ID) that would confirm or deny the sequential ordering?
- Do the EEG file headers contain any metadata that identifies trial origin?
- Is the sub_name_stim block entry order explicitly documented anywhere?
- Are the y_stim rows 1–3 category counts (10,129 / 13,031 / 10,742) the same as the group trial counts — or are they something else (stimulus categories)?

**Acceptance criteria:**
- The trial → subject → group mapping is either:
  - FORMALLY VALIDATED: evidence establishes the ordering independently, OR
  - FORMALLY UNRESOLVED: the limitation is precisely documented and subsequent sprints are blocked until resolved
- A globally unique composite subject key is proposed and validated
- The script that derives the mapping is committed and readable
- The 56 vs. 60 channel discrepancy is documented with any additional evidence found

**Documentation required:** `docs/trial_subject_mapping_validation.md`

**Git commit boundary:** `data: validate trial-subject-group mapping (Sprint 1)`

---

### Sprint 2 — Preprocessing Protocol Decision and Freeze

**Objective:** Make and document an explicit, evidence-based decision for every preprocessing choice before implementing any transformation. Produce a frozen preprocessing specification. This is GATE 2.

**Prerequisites:** Sprint 1 mapping validated (GATE 1 passed); mapping script committed.

**Inputs:**
- `docs/ADHD_EEG_PREPROCESSING_AUDIT.md` (the current most complete audit)
- `docs/dataset-preprocessing-audit.md`
- Literature: Vahid et al. 2019, He et al. 2023
- Validated mapping from Sprint 1

**Outputs:**
- `docs/preprocessing_protocol.md` — the frozen preprocessing specification
  - Each operation: APPROVED / REJECTED / DEFERRED with written justification
  - Filter parameters (if approved): exact cutoffs, family, order, phase mode, edge handling
  - ICA decision: what is the preprocessing state of the downloaded files?
  - 385-sample handling: preserve or crop, and why
  - Channel handling: retain 56 raw columns; spatial mapping deferred pending 56/60 resolution
  - Normalization: scope and fitting rule
  - Label construction rule: exact composite key used as target

**Implementation tasks:**
1. Determine the preprocessing state of downloaded files (already ICA-cleaned by OSF source, or raw?)
2. For each item in the 12-item checklist in `ADHD_EEG_PREPROCESSING_AUDIT.md` Section 12: produce a team decision with written justification
3. Resolve the 385-sample issue: if evidence cannot be found, formally document "preserve all 385 positions" as the default
4. Decide whether filtering is justified (requires evidence of what frequency content is present and whether it has been previously cleaned)
5. Decide whether any normalization is needed and where statistics may be fitted
6. Decide whether the 56-to-60 channel mapping is needed for the primary experiment or can be deferred

**Validation tasks:**
- Confirm every decision in the frozen protocol has a documented evidence basis
- Confirm that no operation's parameters were chosen by inspecting held-out data
- Confirm that the protocol does not require information unavailable without running training (circular dependency check)
- Confirm the protocol specifies training-only fitting for every operation that requires learned parameters

**Audit questions:**
- What is the ground truth regarding the preprocessing state of the OSF dataset? (ICA-cleaned already, or raw?)
- Is bandpass filtering justified by visual inspection of a small number of trials, without using label information?
- Does the 385th sample represent an inclusive endpoint, zero-padding, or something else?
- If filtering is approved, what frequency response and phase characteristics are appropriate for 1.5-second epochs at 256 Hz?

**Acceptance criteria:**
- Every preprocessing operation is either APPROVED with exact parameters, REJECTED with written justification, or DEFERRED with explicit conditions for future reconsideration
- The protocol is signed off by the team before any code is written
- No training-data-dependent parameter decisions are left unspecified
- The 385-sample handling is explicitly decided

**Documentation required:** `docs/preprocessing_protocol.md`

**Git commit boundary:** `docs: freeze preprocessing protocol (Sprint 2)`

---

### Sprint 3 — Preprocessing Implementation and Integrity Audit

**Objective:** Implement the frozen preprocessing protocol in code. Apply it only to the training data within each fold. Produce a derived dataset with complete audit trail. This is NOT GATE 3 (that is the split strategy); this sprint delivers the pipeline artifact.

**Prerequisites:** Sprint 2 protocol frozen (GATE 2 passed). Split strategy from Sprint 4 must also be available before this can fully execute on all folds; the pipeline code can be written in Sprint 3 and executed after Sprint 4.

**Inputs:**
- `docs/preprocessing_protocol.md` (frozen)
- Validated mapping from Sprint 1
- Raw EEG files (read-only)

**Outputs:**
- `src/preprocessing/pipeline.py` — reproducible, auditable preprocessing code
- `src/preprocessing/integrity_checks.py` — post-preprocessing verification suite
- Per-fold derived data manifests (output paths, source hashes, parameter log)

**Implementation tasks:**
1. Implement file-by-file bounded reading (avoid full concatenation in RAM)
2. Implement trial-shape verification before any transformation (expected: `(385, 56)`)
3. Implement approved transformations in exact sequence specified in the frozen protocol
4. Attach validated metadata (subject ID, group label) to each trial
5. Implement output writing to a separate `data/derived/` directory (never overwriting raw)
6. Record source file hash, software versions, parameters, and timestamp for every derived output

**Validation tasks (future acceptance criteria to define now):**
- Shape preservation: every output trial is exactly `(T, 56)` where T is the approved sample count
- dtype: output is float32 or explicitly documented float64
- Finite-value check: no NaN, no Inf in any output trial
- Sample count: total output trials = 33,902 (no duplications or deletions)
- Channel count: 56 channels preserved exactly
- Subject identity: every trial has a validated, unique composite subject key
- Label integrity: HC=0, ADD=1, ADHD=2 (or team-approved encoding); no trial has an ambiguous label
- Leakage check: no test-subject data was accessed during any training-fold processing step
- Reproducibility: re-running the script with the same seed produces bit-identical output
- Transformation correctness: every approved step matches the protocol specification

**Audit questions:**
- Does the output trial count exactly equal 33,902?
- Does each derived file's hash match what the script recorded?
- Are there any trials where the composite subject key is ambiguous or missing?

**Acceptance criteria:**
- All integrity checks pass
- Derived data is in a separate, versioned directory
- Manifest file records all parameters and hashes
- Raw data files are unmodified (hashes unchanged)

**Documentation required:** Manifest files + inline code documentation + `docs/preprocessing_run_log.md`

**Git commit boundary:** `feat: preprocessing pipeline and integrity checks (Sprint 3)`

---

### Sprint 4 — Subject-Level Split Strategy Freeze

**Objective:** Define, justify, and freeze the exact train/validation/test split strategy. This is GATE 3. No model trains until this is frozen.

**Prerequisites:** Sprint 1 mapping validated (GATE 1); Sprint 2 protocol frozen (GATE 2).

**Inputs:**
- Validated subject mapping from Sprint 1
- Subject-level trial counts (highly variable: 46–295 trials/subject)
- Group counts: 44 HC / 52 ADD / 48 ADHD

**Outputs:**
- `docs/split_strategy.md` — exact split specification
  - Primary evaluation: subject-level grouped cross-validation
  - Fold count: to be decided (must be justified relative to 144 subjects)
  - Fixed test set vs. nested cross-validation: to be decided
  - Random seed: must be recorded and frozen
  - Stratification target: class label at subject level
  - Class imbalance handling: specify how unequal trials-per-subject is addressed

**Implementation tasks:**
1. Implement `StratifiedGroupKFold` or equivalent with frozen group/label mapping
2. Verify that no subject appears in more than one partition in any fold
3. Verify that the composite key (not row-0 alone) is used as the grouping variable
4. Document the expected subject counts per fold and verify class distribution
5. Generate and record all fold assignments before any training begins

**Validation tasks:**
- No subject ID appears in both train and test partitions for any fold
- Trial count per fold is documented
- Class distribution per fold is documented (imbalance expected and acceptable if documented)
- Fold assignments are deterministic given the seed

**Audit questions:**
- How many subjects per fold? Are any folds severely imbalanced at the subject level?
- Should a fixed held-out test set be reserved, or is all 144-subject data used in the cross-validation?
- How should trials-per-subject imbalance (46 vs. 295) be handled in loss and metric computation?
- Should class weights be computed from training subjects only, separately per fold?

**Acceptance criteria:**
- Split strategy document is approved by all team members
- No subject appears across partitions in any fold (automated check passes)
- Fold assignments are committed as a reproducible artifact
- Class weights (if used) are computed from training data only

**Documentation required:** `docs/split_strategy.md`

**Git commit boundary:** `docs: freeze subject-level split strategy (Sprint 4)`

---

### Sprint 5 — Baseline Models

**Objective:** Train and evaluate at least two baseline models under the frozen subject-level evaluation protocol before claiming any benefit from the Transformer architecture. This is GATE 4.

**Prerequisites:** Sprints 1–4 all complete; derived dataset available; split strategy frozen.

**Inputs:**
- Derived preprocessed trials (from Sprint 3)
- Frozen split assignments (from Sprint 4)

**Outputs:**
- `src/baselines/` — implementation of at least two baseline models
- `results/baselines/` — evaluation results per fold
- `docs/baseline_results.md` — quantitative summary

**Baseline candidates (to be decided during this sprint, not assumed):**
- A linear classifier on raw trial features (reference floor)
- EEGNet (from He et al., 2023 paper; exact architecture to be confirmed from source)
- ShallowConvNet (from He et al., 2023 paper)
- Statistical features (e.g., band power) + SVM/LDA (as a classical EEG reference)

**Implementation tasks:**
1. Select and justify the exact set of baselines
2. Implement each baseline identically under the frozen subject-level split
3. Record all hyperparameters before training; do not tune on test data
4. Record per-fold and aggregate metrics

**Validation tasks:**
- All baselines use identical splits and evaluation code
- No test-set data was used for model selection or hyperparameter tuning
- Per-class metrics are reported (accuracy alone is insufficient for 3-class imbalanced evaluation)
- Macro-F1 and AUC are reported alongside accuracy

**Audit questions:**
- Does the linear baseline perform above chance (33%)? If not, is there a data issue?
- Do EEGNet and ShallowConvNet under subject-level splitting reproduce approximately the paper's general performance range, or do results differ substantially?
- Are there obvious class-specific failure modes visible from confusion matrices?

**Acceptance criteria:**
- At least two baselines are evaluated and results are documented
- Results are substantially lower than the paper's reported 95% because we are using a stricter evaluation (expected; this is the point)
- No metric was computed using test-set information before evaluation

**Documentation required:** `docs/baseline_results.md`

**Git commit boundary:** `feat: baseline models and subject-level evaluation (Sprint 5)`

---

### Sprint 6 — Transformer Architecture Justification and Implementation

**Objective:** Justify the chosen Transformer architecture from evidence rather than by copying the base paper, and implement it. This is GATE 5.

**Prerequisites:** GATE 4 passed (baselines evaluated). Architecture decision documented before code is written.

**Inputs:**
- Base paper architecture (He et al. 2023)
- Baseline results from Sprint 5
- Literature from sprint_1_research_landscape.md

**Outputs:**
- `docs/architecture_decision.md` — justification of architectural choices
- `src/models/transformer.py` — Transformer implementation
- Parameter count verification against expected values

**Implementation tasks:**
1. Document the exact architectural choices and their justifications (not assumed from master guide)
2. Resolve the missing implementation details from the base paper (dropout rate, d_model, FFN hidden dim, loss function)
3. Implement the chosen architecture with explicit hyperparameter documentation
4. Verify parameter counts match documented expectations
5. Implement a forward-pass unit test on a synthetic batch before training

**Audit questions:**
- Is the `(None, 385, 56)` input orientation correct, or should it be `(None, 56, 385)`?
- What is the correct d_model value — is it 56 (equal to channels) or something else?
- What dropout rate is appropriate?
- Is Global Max Pooling justified over Global Average Pooling for this data?
- Is a Conformer (conv + attention) justified over a pure Transformer given the evidence from baselines?

**Acceptance criteria:**
- Architecture is documented and justified before implementation starts
- All undeclared hyperparameters (dropout, FFN dim, etc.) are explicitly chosen and recorded
- Forward-pass unit test passes on synthetic data
- The architecture is not an arbitrary copy of the master guide's proposal without justification

**Documentation required:** `docs/architecture_decision.md`

**Git commit boundary:** `feat: Transformer architecture implementation (Sprint 6)`

---

### Sprint 7 — Training and Experiment Management

**Objective:** Train the Transformer under the frozen protocol with reproducible experiment management. This is GATE 6.

**Prerequisites:** GATE 5 passed (architecture justified and implemented); split strategy frozen; preprocessing complete; baseline results documented.

**Inputs:**
- Derived dataset (from Sprint 3)
- Frozen split assignments (from Sprint 4)
- Frozen architecture (from Sprint 6)

**Outputs:**
- `src/training/train.py` — training loop
- `experiments/` — per-fold checkpoints, logs, loss curves
- `docs/training_log.md` — training configuration, seeds, hardware, software versions

**Implementation tasks:**
1. Record all seeds (Python, NumPy, framework) before training begins
2. Implement early stopping monitored on subject-level validation loss
3. Implement model checkpointing
4. Implement learning-rate scheduling (if approved in protocol)
5. Implement class-weighting if approved
6. Record training curves per fold

**Audit questions:**
- Are training curves stable, or are there signs of instability suggesting preprocessing or architecture issues?
- Do any folds converge to degenerate solutions (all-one-class predictions)?
- Is training time feasible on available hardware?

**Acceptance criteria:**
- All training runs are reproducible given the recorded seed
- No test-set data was touched during training
- Training configuration is fully documented

**Documentation required:** `docs/training_log.md`

**Git commit boundary:** `feat: training pipeline and experiment management (Sprint 7)`

---

### Sprint 8 — Subject-Level Evaluation

**Objective:** Run final evaluation on held-out subjects; report subject-level generalization metrics.

**Prerequisites:** GATE 6 passed; training complete; test set has never been inspected.

**Inputs:**
- Trained models from Sprint 7
- Held-out test partition from Sprint 4
- Derived test-set trials from Sprint 3 (test-fold only)

**Outputs:**
- `results/evaluation/` — per-fold and aggregate results
- `docs/evaluation_report.md` — primary results document

**Metrics to report (minimum):**
- Subject-level accuracy (macro-averaged)
- Class-wise accuracy (HC / ADD / ADHD separately)
- Macro-F1 score
- Per-class AUC (one-vs-rest)
- Confusion matrix
- Per-fold statistics + mean ± standard deviation across folds
- Confidence intervals (bootstrap or analytical)

**Audit questions:**
- Does performance substantially differ from the paper's 95%? (Expected, due to stricter evaluation)
- Is the performance gap large or small, and what does this imply for the paper's evaluation protocol?
- Is any class (HC / ADD / ADHD) systematically misclassified?
- Are results stable across folds, or is there high variance suggesting small effective test set?

**Acceptance criteria:**
- Evaluation uses only test-fold data, never training folds
- Results are reported per class and overall
- Statistical uncertainty is quantified

**Documentation required:** `docs/evaluation_report.md`

**Git commit boundary:** `results: subject-level evaluation report (Sprint 8)`

---

### Sprint 9 — Explainability and Research Analysis

**Objective:** Apply attribution methods to interpret model decisions; answer the research questions established by the methodology audit. This is GATE 7.

**Prerequisites:** Sprint 8 complete; performance results interpreted.

**Important constraint:** Attention weights ≠ clinical explanation. All attribution claims must be bounded by what the method can legitimately establish.

**Outputs:**
- `src/explainability/` — attribution code
- `docs/explainability_report.md` — findings and their limitations

**Methods to consider (not final):**
- Attention weight visualization (temporal and channel axes)
- Integrated gradients
- Occlusion / perturbation analysis
- Per-class saliency maps

**Audit questions:**
- Do attention patterns differ between HC, ADD, and ADHD groups?
- Are high-saliency time windows or channels consistent across subjects?
- Do any channel attributions align with established neurophysiology (frontal theta, etc.)?
- Are attributions stable across folds, or highly variable?

**Acceptance criteria:**
- No attribution method is described as proving a clinical mechanism
- Attribution findings are stated as model behavior, not as established biomarkers
- Channel attributions are bounded by the unresolved 56-to-60 mapping issue if spatial claims are made

**Documentation required:** `docs/explainability_report.md`

**Git commit boundary:** `analysis: explainability and research analysis (Sprint 9)`

---

## Section 4 — Layer-by-Layer Audit System

Every future implementation layer must be audited using the following cycle before the next layer begins:

```
PLAN
  └─ Define exactly what will be built before any code is written.
     Document: inputs, outputs, parameters, assumptions, limitations.
     ↓
IMPLEMENT
  └─ Write code against the frozen plan only.
     No decisions made during implementation without updating the plan.
     ↓
UNIT / INTEGRITY CHECK
  └─ Automated checks that verify the implementation matches the plan.
     Run independently before any downstream code uses the output.
     ↓
RESEARCH AUDIT
  └─ Review the implementation for methodological correctness.
     Could this introduce leakage? Does it match the frozen protocol?
     Are any assumptions implicit?
     ↓
VALIDATION
  └─ Confirm the output is correct on known test cases.
     For data: check against expected counts, shapes, dtypes.
     For models: check forward pass on synthetic inputs.
     ↓
DOCUMENT
  └─ Record what was done, what was observed, what remains unresolved.
     ↓
COMMIT
  └─ Git commit with a descriptive message referencing the sprint.
```

### Preprocessing Audit Checklist (future implementation criteria — do not implement now)

Every preprocessing output should eventually pass all of the following checks:

| Check | Description |
|---|---|
| Shape preservation | Every output trial has shape `(T, 56)` where T is the approved sample count |
| dtype | Output dtype is float32 (or explicitly documented deviation) |
| Finite values | No NaN or Inf in any trial after any transformation |
| Sample count | Total output trials = 33,902 |
| Channel count | 56 channels preserved exactly |
| Subject identity | Every trial carries a validated, globally unique composite subject key |
| Label integrity | Every trial carries a validated group label (HC / ADD / ADHD); y_stim rows 1–3 are NOT used as labels |
| Leakage | No test-subject data was accessed during any training-fold transformation |
| Reproducibility | Identical output produced by re-running with the same seed and software versions |
| Transformation correctness | Every approved operation matches the frozen protocol specification |
| Parameter recording | All transformation parameters, software versions, and source hashes are logged |
| Raw preservation | Source .mat files are unmodified after any derived output is written |

### Model Audit Checklist (future implementation criteria — do not implement now)

| Check | Description |
|---|---|
| Forward pass | Model accepts expected input shape without error |
| Parameter count | Recorded parameter count matches documented expectation |
| Output shape | Softmax output has 3 classes |
| Gradient flow | Gradients are non-zero for all trainable parameters on a synthetic batch |
| Seed reproducibility | Given identical seed, forward pass produces identical outputs |
| No test leakage | Test-partition data is never accessed before final evaluation |
| Architecture justification | Every non-obvious architecture choice has a documented rationale |

---

## Section 5 — Data Contract

The following defines the expected data contract between pipeline stages.

### 5.1 Raw Trial (VALIDATED as existing)

```
Shape:  (385, 56)          — VALIDATED (from HDF5 inspection)
dtype:  float32            — VALIDATED (from HDF5 inspection)
Units:  UNRESOLVED — EVIDENCE REQUIRED (microvolts assumed; not confirmed)
Axes:   dim 0 = time/sample positions (385 positions, physical meaning UNRESOLVED)
        dim 1 = channels (56; physical channel mapping UNRESOLVED)
```

### 5.2 Trial Metadata (PARTIALLY VALIDATED)

| Field | Status | Source |
|---|---|---|
| `global_trial_index` | STRONGLY SUPPORTED (sequential file arithmetic) | y_stim column ordering |
| `source_file` | VALIDATED (d1–d7) | file identity |
| `within_file_trial_index` | STRONGLY SUPPORTED (sequential within each file) | file ordering inference |
| `block_index` (0, 1, or 2) | VALIDATED (y_stim one-hot rows 1–3 → block) | binary data inspection |
| `within_block_subject_index` (1-based) | VALIDATED (y_stim row 0 contiguous runs) | binary data inspection |
| `composite_subject_key` = `(block_index, within_block_subject_index)` | STRONGLY SUPPORTED; requires Sprint 1 validation | proposed in ADHD_EEG_PREPROCESSING_AUDIT.md |
| `subject_filename` | STRONGLY SUPPORTED (from sub_name_stim) | sub_name_stim ordinal assumed |
| `clinical_group` = HC / ADD / ADHD | STRONGLY SUPPORTED (filename tokens + paper counts) | Sprint 1 must validate |
| `y_stim_row1`, `y_stim_row2`, `y_stim_row3` | VALIDATED as binary indicators; physical meaning UNRESOLVED | binary inspection |

### 5.3 Distinction: What y_stim IS and IS NOT

`VALIDATED:` y_stim rows 1–3 are mutually exclusive binary indicators.
`VALIDATED:` They are NOT the HC / ADD / ADHD diagnostic labels.
`STRONGLY SUPPORTED:` They represent stimulus/task categories (consistent with "stim" name).
`UNRESOLVED:` Their exact physical category names (e.g., target, non-target, distractor) are unknown.

**Implication for modeling:** The target label for classification must come from the subject-group metadata (block identity), NOT from y_stim rows 1–3.

### 5.4 Post-Preprocessing Trial (PROPOSED — NOT YET FROZEN)

```
Shape:  (T, 56)   where T = 385 (unless 385-sample issue resolves differently)
dtype:  float32
Label:  integer ∈ {0, 1, 2} = {HC, ADD, ADHD}
Key:    composite_subject_key (globally unique)
```

This contract is conditional on: Sprint 1 mapping validation, Sprint 2 protocol freeze.

### 5.5 Channel Contract

```
Axis 1 dimension: 56     — VALIDATED
Channel names:    UNRESOLVED (chan.mat has 60 entries; mapping to 56 array columns is unknown)
Spatial positions: UNRESOLVED (do not use for topographic features until resolved)
```

---

## Section 6 — Research Decision Gates

The following gates must be explicitly evaluated and passed before the corresponding sprint may proceed. If a gate fails, the next sprint must NOT begin until the gate issue is resolved.

---

### GATE 1 — Trial → Subject → Clinical Group Mapping

**Question:** Can every EEG trial in d1–d7 be deterministically mapped to a globally unique subject and clinical group with validated evidence?

**Current status:** `UNRESOLVED — EVIDENCE REQUIRED`

**Blocking:** Sprints 2, 3, 4, 5, 6, 7, 8, 9 (all ML implementation sprints)

**What must be demonstrated:**
- Independent evidence that d1 trial ordering matches y_stim columns 0–4,999
- Independent evidence that sub_name_stim block entries are in the same ordinal order as row-0 indices
- OR: explicit documentation of what cannot be proven, with team decision on whether the strongly-supported mapping is sufficient to proceed with stated limitations

**If gate fails:** Document the limitation; decide whether to proceed with a "strongly supported but not independently proven" mapping as a stated limitation, or to delay until external confirmation is obtained.

---

### GATE 2 — Preprocessing Protocol is Frozen

**Question:** Is the preprocessing protocol fully specified, evidence-supported, and team-approved before any code is written?

**Current status:** `UNRESOLVED — EVIDENCE REQUIRED`

**Blocking:** Sprint 3 (implementation), all downstream sprints

**What must be demonstrated:**
- Every preprocessing operation has a documented decision (APPROVED / REJECTED / DEFERRED)
- No operation is approved without an evidence basis
- 385-sample handling is explicitly decided
- All learned parameters will be fitted on training data only
- The frozen protocol document is signed off by the team

---

### GATE 3 — Subject-Level Split Strategy is Frozen

**Question:** Is the exact evaluation split strategy (fold count, seed, test-set design, class-weighting rule) documented and frozen before any model trains?

**Current status:** `UNRESOLVED — EVIDENCE REQUIRED`

**Blocking:** Sprints 5, 6, 7, 8, 9

**What must be demonstrated:**
- Fold count and test-set design are documented
- A random seed is recorded
- No test-set data will be inspected until Sprint 8
- Class imbalance handling is specified

---

### GATE 4 — Baselines Established Before Claiming Transformer Benefit

**Question:** Have baseline models (including at least one from the paper's comparison set) been evaluated under the frozen subject-level protocol before any Transformer results are interpreted?

**Current status:** `UNRESOLVED — EVIDENCE REQUIRED`

**Blocking:** Sprint 6 (Transformer implementation may begin; claiming benefit is blocked)

**What must be demonstrated:**
- At least two baseline models evaluated
- Results documented with per-class and overall metrics
- Baselines use identical splits and evaluation code as the future Transformer

---

### GATE 5 — Transformer Architecture is Justified, Not Assumed

**Question:** Is the Transformer architecture decision documented with explicit justification for each non-trivial architectural choice?

**Current status:** `UNRESOLVED — EVIDENCE REQUIRED`

**Blocking:** Sprint 7 (training) and Sprint 8 (evaluation)

**What must be demonstrated:**
- Input orientation, d_model, number of heads, number of blocks, FFN dimension, dropout rate, pooling type, loss function are all explicitly chosen and justified
- The choice to use a Transformer (vs. a simpler baseline) is justified by what baselines showed
- The architecture is not an uncritical copy of the master guide's Conformer proposal

---

### GATE 6 — Evaluation Metrics and Evaluation Unit are Frozen Before Training

**Question:** Are the primary evaluation metric, evaluation unit (trial-level vs. subject-level), and reporting convention frozen before training begins?

**Current status:** `UNRESOLVED — EVIDENCE REQUIRED`

**Blocking:** Sprint 7 (training loop must not start until this is frozen)

**What must be demonstrated:**
- Primary metric: subject-level generalization accuracy (macro-averaged)
- Secondary metrics: macro-F1, per-class AUC, confusion matrix
- The test set is used only once (no repeated evaluation with different thresholds)
- Mean and standard deviation are reported across folds, not just overall

---

### GATE 7 — Explainability Claims are Methodologically Bounded

**Question:** Are all explainability and interpretability claims bounded by what the attribution method can legitimately establish, and explicitly NOT presented as validated clinical biomarkers?

**Current status:** `UNRESOLVED — EVIDENCE REQUIRED`

**Blocking:** Sprint 9 claims and any publication

**What must be demonstrated:**
- No attention weight visualization is described as "proving" which brain region causes ADHD
- Attribution claims are stated as "model sensitivity to this feature" not "this feature is a clinical biomarker"
- Channel attributions are bounded by the unresolved 56-to-60 mapping limitation
- All explainability findings are labeled as descriptive, not causal

---

## Section 7 — What We Must NOT Do Yet

The following operations must NOT occur at the current project stage.

| Prohibited action | Reason |
|---|---|
| Preprocessing the entire dataset | Sprint 1 (GATE 1) and Sprint 2 (GATE 2) are not yet complete |
| Training any Transformer or model | Sprints 1–5 (GATES 1–4) are not yet complete |
| Applying 0.5–45 Hz bandpass filter | No evidence basis; preprocessing protocol not frozen |
| Applying 50/60 Hz notch filter | Line frequency not established; preprocessing protocol not frozen |
| Running ICA | Unknown processing state of downloaded files; no approved protocol |
| Cropping 385 → 384 samples | Physical meaning of 385th sample is unresolved; no evidence justifies removal |
| Random trial-level train/test splitting | Would allow same-subject trials across partitions; explicitly prohibited for the primary evaluation |
| Using y_stim rows 1–3 as HC/ADD/ADHD labels | These are stimulus indicators, not diagnostic labels — validated distinction |
| Computing normalization statistics over the full dataset | Normalization must be fitted on training subjects only |
| Assuming the 56-to-60 channel mapping | Identity of 4 absent channels is unknown; spatial features are blocked |
| Building topographic / scalp topomap features | Channel spatial mapping is unresolved |
| Claiming a specific research gap as established | Research gap hypotheses require literature evidence, not just a teammate's proposal |
| Predicting or reporting expected model accuracy | No performance predictions (e.g., "80–90% accuracy") |
| Claiming trial-level leakage in the base paper as a confirmed fact | It is strongly supported but the paper's code is unavailable for verification |
| Claiming the Hybrid Conformer as the approved architecture | It is a proposal; no architecture decision has been made |
| Reporting TBR as a validated ADHD biomarker in this dataset | TBR is proposed; not verified in this specific dataset |
| Starting Sprint N+1 if Sprint N's gate has not passed | The dependency order is mandatory |

---

## Section 8 — First Implementation Target

The single next implementation target, given the current state of the repository, is:

### SPRINT 1: Formal Validation of the Trial → Subject → Clinical Group Mapping

**Why this is the correct first target:**

The entire ML pipeline depends on the ability to create leakage-safe subject-level splits. Subject-level splitting requires knowing which trials belong to which subject. That knowledge depends on the trial → y_stim column mapping and the sub_name_stim ordinal mapping. Neither of these has been independently validated — both are "strongly supported" inferences.

Before preprocessing, before splitting, before any model code, before any normalization, the research team must answer:

> *"Can we produce a formal, defensible, independently-verified mapping from every EEG trial in d1.mat–d7.mat to a globally unique subject identifier and clinical group label?"*

**If the answer is YES** (with documented evidence):
- GATE 1 is passed
- Sprint 2 (preprocessing protocol) can begin

**If the answer is NO** (the ordering cannot be independently proven):
- The limitation must be formally documented
- The team must decide: proceed with the "strongly supported" mapping as an explicit stated limitation, OR obtain external evidence
- No model training may proceed under a mapping that is asserted without evidence

**What Sprint 1 consists of practically:**
1. Write a script that inspects all 7 EEG file headers and verifies expected variable names, shapes, dtypes, and chunking
2. Write a script that derives the sequential file→column mapping and computes per-subject trial counts
3. Search for any independent ordering evidence (embedded trial IDs, cross-reference file, OSF readme content, original data collection documentation)
4. Produce `docs/trial_subject_mapping_validation.md` with a formal verdict

This is the correct first implementation target.

---

## End-of-Plan Summary

### Files Inspected
- `docs/dataset-investigation.md` (199 lines)
- `docs/base-paper-analysis.md` (1,147 lines)
- `docs/ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md` (927 lines)
- `docs/ADHD_EEG_PREPROCESSING_AUDIT.md` (233 lines)
- `docs/dataset-preprocessing-audit.md` (367 lines)
- `docs/PROJECT-SPRINT-MAP.md` (246 lines)
- `docs/sprint_1_research_landscape.md` (73 lines)
- `docs/domain-understanding.md` (partial, 100 lines reviewed)
- `adhd-eeg-transformer-master-guide.md` (304 lines)
- `dataset_metadata_investigation.md` (57 lines)
- `scripts/inspect_mat_files.py` (114 lines)
- `scripts/inspect_deep.py`, `inspect_sub.py`, `inspect_ystim_relations.py` (reviewed)
- `scripts/dataset-investigation/` (8 files reviewed)
- `.gitignore` (6 lines)

### Important Findings

1. **The dataset is fully acquired and structurally validated.** Shapes, dtypes, trial counts, and metadata structure are confirmed by direct binary inspection.

2. **The trial→subject→group mapping is strongly supported but NOT independently proven.** This is the most critical unresolved issue. The sequential file-to-column ordering is arithmetically consistent but has no independent cross-reference.

3. **The base paper contains at least 10 internal inconsistencies.** These must be preserved as documented uncertainties, not silently resolved.

4. **The master guide (`adhd-eeg-transformer-master-guide.md`) is a teammate proposal, not a validated protocol.** Every preprocessing step in it (0.5–45 Hz filter, notch, ICA, TBR, spectral bands, Conformer architecture) is explicitly `PROPOSED — NOT YET FROZEN`.

5. **y_stim rows 1–3 are stimulus category indicators, not HC/ADD/ADHD labels.** Using them as diagnostic targets would be a critical error.

6. **The 385-sample discrepancy is unresolved.** The 385th position must be preserved until evidence establishes its meaning.

7. **The channel mapping (56 vs. 60) is unresolved.** Spatial/topographic features are blocked until this is resolved.

8. **The preprocessing audit (`ADHD_EEG_PREPROCESSING_AUDIT.md`) has a 12-item checklist of which only 1 item is checked.** All 11 remaining items are prerequisites to Sprint 2.

### Unresolved Questions (ordered by criticality)

1. Can the trial → y_stim column join be independently verified? (GATE 1)
2. Can the sub_name_stim filename ordinal → row-0 index mapping be independently verified? (GATE 1)
3. Are the downloaded EEG files already ICA-cleaned, or are they raw? (GATE 2)
4. What is the physical meaning of the 385th sample? (GATE 2)
5. Which 4 channels are absent between chan.mat (60) and EEG arrays (56)? (blocks spatial features)
6. What are the physical meanings of y_stim rows 1–3 categories? (affects experimental design interpretation)
7. Was the base paper's splitting performed at trial level or subject level? (affects interpretation of our results vs. theirs)
8. What exact preprocessing does the paper report that was applied before OSF publication? (GATE 2)

### Proposed Sprint Sequence

```
Sprint 0: ML Planning (THIS DOCUMENT) ← done
Sprint 1: Trial → Subject → Group Mapping Validation [GATE 1]
Sprint 2: Preprocessing Protocol Freeze [GATE 2]
Sprint 3: Preprocessing Implementation + Integrity Checks
Sprint 4: Subject-Level Split Strategy Freeze [GATE 3]
Sprint 5: Baseline Models [GATE 4]
Sprint 6: Transformer Architecture Justification + Implementation [GATE 5]
Sprint 7: Training + Experiment Management [GATE 6]
Sprint 8: Subject-Level Evaluation
Sprint 9: Explainability and Research Analysis [GATE 7]
```

### First Implementation Target

> **Sprint 1: Formal validation of the trial → subject → clinical group mapping.**
>
> Deliverable: `docs/trial_subject_mapping_validation.md` with a formal verdict of VALIDATED or UNRESOLVED.
> Deliverable: A committed inspection script that derives and documents the mapping.
>
> No preprocessing code, model code, or training should begin until this sprint is complete and its acceptance criteria are met.

---

*Document created: 2026-10-01. Branch: ml/transformer-development. Author: ML audit pass (Sprint 0).*
*No source code was modified. No data was transformed. No models were trained.*
