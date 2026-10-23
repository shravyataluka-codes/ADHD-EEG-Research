# 01 — Project State Audit

**Date:** 2026-10-06  
**Purpose:** Record the actual current state of the project before architecture decisions are made.  
**Audience:** CS student with basic programming knowledge (Java / C / Python / DSA), new to EEG, ML, and Transformers.  
**Scope:** What exists RIGHT NOW. No architecture decisions. No implementation decisions.

---

## Quick Orientation — Key Terms Used in This Document

| Term | Plain meaning |
|---|---|
| **EEG** | Electroencephalogram — a recording of the brain's electrical activity picked up by electrodes placed on the scalp. It looks like a grid of wavy lines over time. |
| **Trial / epoch** | One short segment of EEG recording, here roughly 1.5 seconds long. The dataset contains 33,902 of these. |
| **Channel** | One electrode on the scalp. This dataset has 56 channels per trial. |
| **Sample / time step** | One measurement at one point in time, across all channels. Each trial has 385 of these. |
| **Trial shape (385, 56)** | A 2D table: 385 rows (one per time step) × 56 columns (one per electrode). |
| **HC / ADD / ADHD** | The three subject groups: Healthy Control, Attention Deficit Disorder, Attention Deficit Hyperactivity Disorder. |
| **Subject-level split** | Keeping all trials from the same person in the same data partition. This is required to fairly test on people the model has never seen. |
| **Label / class** | The answer the model should predict: HC, ADD, or ADHD. |
| **Transformer** | A type of machine learning model originally built for text. It works by finding relationships between different positions in a sequence — here, potentially between different time steps or channels in an EEG. |
| **Gate** | A checkpoint the project must pass before moving to the next step. |
| **Sprint** | A defined unit of work with specific deliverables. |

---

## Summary — Project Objective, Research Objective, Prototype Objective, Intended User, Dataset

### Project Objective

Take an EEG recording as input and predict whether it belongs to an HC, ADD, or ADHD subject.  
The system must also explain *why* it made its prediction and show how confident it is.

### Research Objective

Study whether EEG recordings can reliably distinguish between three groups (HC, ADD, ADHD) in people the model has never seen before.  
Investigate which parts of the EEG the model uses to make its decisions.  
Honestly report what the model gets right and what it fails at — not just the final accuracy number.

### Prototype Objective

Build a simple, usable application where a user uploads an EEG recording and receives:
1. A predicted class (HC / ADD / ADHD)
2. A confidence score for each class
3. Visualizations showing what the model paid attention to

> [!IMPORTANT]
> The prototype is **not intended to replace a clinical diagnosis**. It is a research-support tool for trained clinicians, EEG technicians, or researchers.

### Intended User

People who already work with EEG data — such as clinicians, EEG technicians, or academic researchers.  
The user provides the EEG recording. The system analyses it and shows results.  
The person whose brain was recorded is the *subject* — different from the *user* who operates the tool.

### Current Research Dataset

| Item | Value |
|---|---|
| Source | He et al. (2023), *Journal of Neural Engineering*, DOI 10.1088/1741-2552/acf7f5 |
| Full dataset name | ADHD EEG Dataset, OSF project `6594x` |
| Subjects | 144 children total: 44 HC, 52 ADD, 48 ADHD |
| Total trials | 33,902 |
| Trial shape | (385 time steps, 56 channels) |
| File format | MATLAB `.mat` files (HDF5 format internally) |
| Files | `d1.mat` – `d7.mat` (EEG data), `y_stim.mat` (metadata), `sub_name_stim.mat` (subject names), `chan.mat` (channel info) |
| Local status | **All files present** on disk. Dataset is excluded from Git (too large). |

---

## 1. VERIFIED / ACTUALLY IMPLEMENTED

These things are confirmed by either direct file inspection, running code, or executed tests.

### 1.1 Dataset — Confirmed by Binary File Inspection

- All 7 EEG files (`d1.mat` – `d7.mat`) are present and fully intact.
- Every EEG file stores data in the shape **(385 time steps × 56 channels × N trials)**.
  - `d1` through `d6`: 5,000 trials each.
  - `d7`: 3,902 trials.
  - Total: **33,902 trials** — exactly matches the paper's reported count.
- Data type: `float32` (a standard number format for ML; values like 3.14 stored in 32 bits).
- Values range roughly −500 to +500 (likely in microvolts — but the unit is not officially confirmed; see Section 4).
- **Zero missing data**: all 33,902 trials across all 56 channels and 385 time steps contain valid numbers. No corrupted entries (no `NaN` or `Inf` anywhere).
- The 7 EEG files are physically one continuous recording split into chunks to comply with a 5 GB file-size limit on OSF (the storage platform). The OSF readme explicitly states this.
- File creation timestamps in the binary headers confirm `d1` through `d7` were written **sequentially** within a 2-minute window on March 15, 2019.
- `y_stim.mat` contains a matrix of shape **(4 rows × 33,902 columns)** in `float64` format.
  - Row 0: subject index numbers (integers 1–44, 1–52, 1–48 — one block per group).
  - Rows 1–3: one-hot group indicators (**not stimulus categories** — an earlier misconception was resolved during Sprint 1; see below).
- `sub_name_stim.mat`: 144 subject filenames, alphabetically sorted within three blocks of 44, 52, and 48 filenames each.
- `chan.mat`: 60 channel-metadata entries (name, coordinates). **Mismatch with EEG arrays which have 56 channels** — 4 are absent; identities unknown.

### 1.2 Trial-to-Subject-to-Group Mapping — GATE 1 PASSED (Sprint 1 Complete)

This was the most critical open question before any ML work could begin.

**The question:** Can we reliably say "trial #5,137 belongs to subject ADD_02"?

**The answer (from Sprint 1 validation):** **Yes — with one minor remaining caveat.**

Evidence collected:
- The sequential file ordering (`d1` → `d7`) was confirmed by **empirical signal tests**: at every file boundary, the same person's EEG signal appears on both sides (correlation r >= 0.90), while a different person's signal gives low correlation (r < 0.57). This was tested at all 6 file boundaries.
- The 144 subject runs in `y_stim` row 0 are perfectly contiguous — no gaps, no duplicates.
- Subject-to-group assignment (which block = HC / ADD / ADHD) matches the paper's numbers exactly.
- A safe, unique subject ID was established: `HC_01` through `HC_44`, `ADD_01` through `ADD_52`, `ADHD_01` through `ADHD_48`.

**Minor remaining caveat:** The link between a subject's ID number and their specific filename in `sub_name_stim.mat` is "strongly supported" but not independently proven (the original individual files are not on OSF). This does **not** affect safety of machine learning evaluation.

> [!NOTE]
> **Also resolved in Sprint 1:** `y_stim` rows 1–3 were previously suspected to be stimulus category indicators. Direct inspection proved they are actually the three group block indicators (HC / ADD / ADHD), one-hot encoded. So rows 1–3 carry the same information as knowing which group the subject belongs to — they are not separate stimulus types.

### 1.3 Implemented Code — `src/adhd_eeg/data_access.py`

This is the **only Python source code module** in the project. It has two main classes:

**`TrialManifest`** — Reads the metadata (not the EEG) and provides a lookup table.
- Loads `y_stim.mat` (metadata only — never reads EEG data).
- Produces metadata for any trial: which file it's in, which subject it belongs to, which group.
- Can iterate over all trials, or filter by subject ID or group.
- Can export the metadata table as a CSV file.

**`EEGTrialStore`** — Reads actual EEG data safely.
- Uses `TrialManifest` to know where to find a trial.
- Reads one trial at a time (shape `(385, 56)`) from the correct `.mat` file.
- Can read a batch of trials at once.
- Can iterate through all trials in configurable batch sizes.
- Opens files read-only. Never modifies the source files.
- Validates every trial's shape and data type before returning it.
- Returns a **copy** of the data — modifying the returned array does not change the file.

**What this code does NOT do:**
- It does not preprocess, filter, or transform the EEG signal in any way.
- It does not split data into training/validation/test sets.
- It does not normalize the data.
- It does not provide labels for ML training (it provides metadata, but the ML label encoding is not yet decided).
- It does not contain any model, any training loop, or any feature extraction.

### 1.4 Tests — All Passing

**File:** `tests/test_data_access.py`  
**Result: 7 / 7 tests passing** (verified by running `pytest` — exit code 0, 0.63s)

| Test name | What it checks |
|---|---|
| `test_trial_retrieval_shape_and_dtype` | Reading a single trial returns shape `(385, 56)` in `float32`. |
| `test_file_boundary_indices_and_metadata_alignment` | Trials at file boundaries map to the correct file and within-file index. Metadata (subject ID, group) is correct. |
| `test_subject_and_group_lookup` | Filtering by subject ID and by group returns the correct trials. |
| `test_batches_are_bounded_and_preserve_request_order` | Batch reading respects the batch size limit and preserves order. |
| `test_manifest_csv_contains_metadata_only` | CSV export contains metadata fields, no EEG values. |
| `test_source_files_are_unchanged_and_returned_trials_are_detached` | Modifying the returned array does not alter the source file (confirmed by SHA-256 hash). |
| `test_invalid_indices_and_batch_sizes_are_rejected` | Out-of-range indices and invalid batch sizes raise correct errors. |

All tests use **synthetic data** (fake EEG files created in a temporary folder) — they do not require the real dataset files.

---

## 2. VERIFIED BUT NOT IMPLEMENTED

These things are confirmed to be true, but **no code has been written** to use them in the ML pipeline yet.

- **Subject-level split is required**: All 144 subjects are identified. Keeping each subject's trials together in one partition is a confirmed requirement. The exact method (which subjects go into which fold, how many folds, what random seed) is NOT yet decided.
- **Class imbalance exists**: Trials per subject range from 46 to 295. ADD subjects have more trials on average (250.6 +/- 31) than HC (230.2 +/- 56.6) or ADHD (223.8 +/- 57.3). How to handle this imbalance is not yet decided.
- **y_stim rows 1–3 identity** was resolved during Sprint 1 (they are group block indicators, not stimulus categories). But the code has not been updated to *use* them differently — they are still just metadata.
- **Gate 1 (mapping)** passed — but no ML pipeline exists yet that *consumes* this mapping.

---

## 3. PROPOSED / SUGGESTED

These appear in documents written by a teammate or in planning files. They are **not frozen decisions**. They have not been validated, approved, or implemented.

> [!WARNING]
> Do not treat anything in this section as a confirmed design decision. These are starting points for discussion, not instructions for implementation.

| Proposal | Source |
|---|---|
| 0.5–45 Hz bandpass filter (remove very slow and very fast noise from the signal) | Teammate master guide |
| 50 Hz or 60 Hz notch filter (remove electrical power-line interference) | Teammate master guide |
| Re-running ICA (a technique to remove eye-blink and muscle noise from EEG) | Teammate master guide |
| Theta/Beta Ratio (TBR) as a feature (ratio of two frequency bands, proposed as ADHD biomarker) | Teammate master guide |
| Spectral band power features using Welch PSD (breaking the signal into frequency components) | Teammate master guide |
| Hybrid Spatial-Spectral Conformer architecture (a specific model combining convolution and attention) | Teammate master guide |
| Depthwise spatial convolution layer | Teammate master guide |
| StratifiedGroupKFold with 5 folds (a specific data-splitting approach) | Teammate master guide + sprint plan |
| AdamW optimizer with cosine annealing (specific training settings) | Teammate master guide |
| MNE-Python for preprocessing (a popular EEG processing library) | Teammate master guide |
| Scalp topomap visualization (map of which electrodes are most active) | Teammate master guide |

---

## 4. UNKNOWN / UNRESOLVED

These questions have not been answered and must be resolved (or explicitly deferred) before certain decisions can be made.

### Dataset unknowns

| Unknown | Why it matters |
|---|---|
| **Physical meaning of sample 385** | At 256 Hz for 1.5 seconds, you expect 384 samples (256 × 1.5 = 384). But the files have 385. No documentation explains the extra sample. Until this is resolved, all 385 must be preserved — none can be discarded. |
| **Which 4 channels are missing** | `chan.mat` has 60 channel entries; the EEG arrays have 56 columns. Which 4 are absent is unknown. This blocks any experiment that needs to know which brain location each column represents. |
| **EEG amplitude units and reference** | The values in the data are floating-point numbers. They are likely in microvolts (based on the value ranges), but this is not officially established. The reference electrode (where "zero" is defined) is also unknown. |
| **ICA state of the downloaded files** | The paper says ICA was applied before releasing the data. But it is not confirmed whether the OSF files are already ICA-cleaned or are still raw. This matters if anyone considers running ICA again. |
| **Whether trials overlap** | Were the 1.5-second windows extracted with any overlap between consecutive trials? Unknown. |
| **Exact physical meaning of y_stim rows 1–3** | Confirmed to be group block indicators, but the naming ("stim" = stimulus) still creates ambiguity. Using them directly as ML labels is not approved — labels must come from the block identity. |
| **The paper's internal inconsistencies** | The base paper (He et al. 2023) contains at least 10 internal inconsistencies (e.g., 144 subjects in one place, 300 in another; parameter count conflicts). These are unresolved. |

### ML / evaluation unknowns

| Unknown | Why it matters |
|---|---|
| **Exact data split method** | Must be decided before training. Must be subject-grouped (all trials from one person in one partition). Exact algorithm, number of folds, and random seed not yet chosen. |
| **Held-out test set design** | Whether to keep a final test set separate from cross-validation, or use all subjects in cross-validation. Not decided. |
| **Random seed** | Any randomized partition requires a seed to be reproducible. Not recorded. |
| **Class balancing** | How to handle the fact that some subjects have many more trials than others. Not decided. |
| **Validation strategy** | How to select the best model during training without accidentally using test data. Not decided. |
| **Evaluation unit** | Whether the model's performance is measured trial-by-trial or subject-by-subject (aggregating all trials from one person into one final prediction). Not decided. |

---

## 5. DECISIONS ALREADY FROZEN

These items have been formally decided and must not be changed without documented justification.

| Decision | What is frozen |
|---|---|
| **Preserve all 385 time steps** | Do not crop or discard any sample. The 385th sample's meaning is unknown, but it must be kept. |
| **Preserve all 56 channel columns** | Do not drop or reorder channels. Channel identities are unknown; no spatial assumptions are allowed. |
| **Keep data as float32** | Do not silently convert to another number format. |
| **Do not touch the source files** | `d1.mat` through `d7.mat` are read-only raw data. No code may write to or overwrite them. |
| **Subject is the split unit** | All trials from one person must go into the same partition. Trial-level splitting is explicitly prohibited. |
| **No preprocessing without explicit approval** | Filtering, ICA, normalization, baseline correction, artifact rejection, and spectral features are all currently DEFERRED. None may be implemented until a documented research decision with evidence and parameters is made. |
| **Do not use y_stim rows 1–3 as classification labels** | The block identity (HC / ADD / ADHD) must be derived from the block position (0, 1, or 2), not from y_stim rows 1–3 directly. |
| **No normalization over the full dataset** | If normalization is ever approved, statistics (mean, standard deviation, etc.) may only be calculated from training data — never from the full dataset before splitting. |
| **The base paper's architecture is not automatically adopted** | The Transformer architecture from He et al. (2023) is a reference, not an automatic choice. Any architecture must be justified. |
| **No performance predictions** | Statements like "we expect 80–90% accuracy" are explicitly prohibited. |
| **Gate sequence is mandatory** | No sprint may begin until the previous sprint's gate has passed. |

**Current gate status:**

| Gate | Status | What it unlocks |
|---|---|---|
| Gate 1 — Trial/subject/group mapping | **PASSED** | Preprocessing decisions can now begin |
| Gate 2 — Preprocessing protocol frozen | **NOT PASSED** | Preprocessing code, all downstream steps |
| Gate 3 — Split strategy frozen | **NOT PASSED** | Model training, evaluation |
| Gate 4 — Baselines evaluated | **NOT PASSED** | Claiming any benefit from Transformer |
| Gate 5 — Architecture justified | **NOT PASSED** | Training |
| Gate 6 — Evaluation protocol frozen | **NOT PASSED** | Running the training loop |
| Gate 7 — Explainability claims bounded | **NOT PASSED** | Publishing interpretability results |

---

## Repository File Overview

```
ADHD-EEG-Research/
│
├── dataset/                        <- Raw data files (excluded from Git)
│   ├── d1.mat – d7.mat             <- EEG data; 33,902 trials total
│   ├── y_stim.mat                  <- Subject metadata (4 rows x 33,902 cols)
│   ├── sub_name_stim.mat           <- 144 subject filenames
│   └── chan.mat                    <- 60 channel entries (mismatch with 56)
│
├── src/adhd_eeg/
│   └── data_access.py              <- ONLY implemented code module
│                                      TrialManifest + EEGTrialStore
│
├── tests/
│   └── test_data_access.py         <- 7 tests, all passing
│
├── scripts/                        <- One-off inspection scripts (not part of pipeline)
│   ├── validate_trial_subject_mapping.py   <- Sprint 1 validation script
│   ├── inspect_mat_files.py
│   ├── inspect_deep.py
│   ├── inspect_sub.py
│   ├── inspect_ystim_relations.py
│   └── dataset-investigation/      <- Early exploration scripts
│
├── docs/
│   ├── architecture-preparation/
│   │   └── 01_PROJECT_STATE_AUDIT.md  <- This document
│   ├── trial_subject_mapping_validation.md   <- Sprint 1 report (Gate 1 passed)
│   ├── SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md   <- Gate 2 status: NOT PASSED
│   ├── SPRINT_2_DECISION_ANALYSIS.md
│   ├── ML_EXECUTION_PLAN.md        <- Full sprint plan (Sprint 0 output)
│   ├── PROJECT_SCOPE_AND_SYSTEM_SPECIFICATION.md
│   ├── PROJECT-SPRINT-MAP.md
│   ├── ADHD_EEG_PREPROCESSING_AUDIT.md
│   ├── ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md
│   ├── base-paper-analysis.md      <- Deep analysis of He et al. (2023)
│   ├── dataset-investigation.md
│   ├── dataset-preprocessing-audit.md
│   ├── domain-understanding.md
│   └── sprint_1_research_landscape.md
│
└── papers/
    └── base-paper.pdf              <- He et al. (2023)
```

---

## A. What We Definitely Know

1. The dataset has **33,902 trials** from **144 subjects** (44 HC, 52 ADD, 48 ADHD).
2. Every trial is a **2D array of shape (385, 56)** — 385 time steps × 56 channels — stored as `float32`.
3. All data files are present locally, intact, and contain zero missing values.
4. The dataset was split across 7 files only because of a storage size limit; they are one continuous dataset.
5. Every trial can be traced to a specific subject and group using the mapping: `HC_01 – HC_44`, `ADD_01 – ADD_52`, `ADHD_01 – ADHD_48`.
6. The mapping was validated by empirical signal-correlation tests at all 6 file boundaries (Gate 1 passed).
7. `y_stim` rows 1–3 are group block indicators (not stimulus categories). Clinical labels must come from block position, not rows 1–3 directly.
8. Trial counts per subject range from **46 to 295**. This imbalance must be handled in ML evaluation.
9. `chan.mat` has 60 channel entries; EEG arrays have 56 columns. The 4 missing channels are unknown.
10. The base paper (He et al. 2023) uses a Transformer with input shape `(385, 56)`, but contains at least 10 internal inconsistencies that are unresolved.
11. The paper used ICA for preprocessing, but whether the OSF files are already ICA-cleaned is unknown.
12. **No preprocessing has been run. No model exists. No training has occurred.**

---

## B. What We Definitely Have Implemented

1. **One Python package** (`src/adhd_eeg/`), containing one module (`data_access.py`).
2. **`TrialManifest`**: reads subject/group metadata from `y_stim.mat`; can look up, iterate, and filter metadata by subject or group; can export to CSV. Does not read EEG samples.
3. **`EEGTrialStore`**: reads individual trials and batches from the raw `.mat` files; validates shapes and types; never modifies source files; returns copies.
4. **7 automated tests**, all passing, covering: trial reading, boundary correctness, subject/group lookup, batch iteration, CSV export, file immutability, and input validation.
5. **One Sprint 1 validation script** (`validate_trial_subject_mapping.py`) and its formal report — this is what passed Gate 1.

That is everything. There is no preprocessing code. There is no model code. There is no training code. There is no evaluation code.

---

## C. What Remains Undecided

In order of what must be decided first:

1. **Preprocessing protocol** (Gate 2 — currently blocking all downstream work)
   - Will any signal filtering be applied? If yes: which type, which frequency range, which exact settings?
   - Will normalization be applied? If yes: how, and fitted on which data?
   - How will the 56-to-60 channel mismatch be handled?
   - Will any trials be excluded as artifacts?

2. **Data splitting strategy** (Gate 3)
   - How many folds in cross-validation?
   - Is there a final held-out test set, or is all data used in cross-validation?
   - Which random seed?
   - How is class/subject imbalance handled?

3. **Evaluation design** (Gate 6)
   - Is the primary metric trial-level or subject-level?
   - What metric is used (accuracy, F1, AUC)?

4. **Model architecture** (Gate 5)
   - What kind of model is used?
   - What are its specific settings?
   - This decision must come *after* baselines are evaluated (Gate 4).

5. **Explainability method** (Gate 7)
   - Which method will be used to explain the model's decisions?
   - What claims can be made from it?

---

## D. What Must Be Known Before Designing the Architecture

These are blockers. The architecture cannot be responsibly designed until these are answered.

1. **What does the model's input look like?**
   - Is it the raw `(385, 56)` trial? Or a transformed version?
   - If transformed (filtered, normalized, frequency-domain): what is the output shape?
   - This requires the preprocessing protocol to be frozen first (Gate 2).

2. **What is the evaluation protocol?**
   - Are we measuring trial-level accuracy or subject-level accuracy?
   - How many training subjects vs. validation subjects vs. test subjects?
   - This requires the split strategy to be frozen (Gate 3).

3. **What do the baselines show?**
   - The architecture must be justified by showing it improves on simpler models.
   - Baselines must be trained and evaluated first (Gate 4).

4. **Which channels can be used spatially?**
   - The 56-to-60 channel mapping is unresolved.
   - Any architecture that assigns meaning to specific electrode positions requires this to be resolved first.

5. **Is there a class imbalance problem?**
   - Trial counts vary widely per subject. If this is not handled in the training objective, the model may be biased.
   - Must be decided as part of the evaluation design.

> [!IMPORTANT]
> The architecture cannot be designed in isolation. It depends on the input format (from preprocessing decisions) and the evaluation setup (from split decisions). Those decisions are made by the researcher — not by an AI agent. This document captures what is known so those decisions can be made with full information.

---

*Audit produced by: Antigravity AI (inspection only — no architecture decisions, no code generation)*  
*Sources inspected: all files in `src/`, `tests/`, `scripts/`, and `docs/` as of 2026-10-06*
