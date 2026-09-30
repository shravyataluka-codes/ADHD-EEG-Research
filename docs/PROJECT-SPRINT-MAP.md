# ADHD-EEG Research Project: Sprint Map

> **Important Principles**
> * Sprints evolve from evidence. This map is a living document; if dataset investigation reveals a sprint is unnecessary, it will be combined or removed. If a new investigation becomes necessary, it will be added.
> * No preprocessing, modeling, or research question selection until Phase A and Phase B are definitively complete.
> * **The first major milestone:** *"Can we confidently explain exactly what our dataset contains, how its files relate to each other, how trials relate to subjects/groups/stimuli, and what remains unresolved?"*

---

## Phase A: Dataset Understanding
*Objective: Understand the dataset completely enough to freeze its structure and limitations.*

### Sprint A1 — Dataset Inventory
* **Objective:** Establish a comprehensive file-level understanding of the dataset.
* **Questions to answer:** What files exist? What are their sizes, formats, and relationships? Which are raw EEG vs. metadata? How is the OSF structured?
* **Main tasks:** Map all files (d1-d7, mat files, readme), check file sizes/types, document available documentation.
* **Expected evidence/output:** Reliable dataset inventory document.
* **Completion criteria:** All files in the OSF distribution are accounted for and categorized.
* **Dependencies:** None.
* **What must NOT be done yet:** Opening raw data to analyze the actual signals; writing preprocessing scripts.
* **Suggested team ownership:** User (Integration/Documentation)

### Sprint A2 — Raw EEG Structure
* **Objective:** Determine the dimensional and technical structure of the raw EEG files.
* **Questions to answer:** What are the shapes, dimensions, dtypes, compression, and chunking of d1-d7? How many trials are per file? Do d1-d6 have the same structure? What is the exact structure of d7?
* **Main tasks:** Load each data file, inspect headers/metadata, check tensor shapes, verify compression details.
* **Expected evidence/output:** Document detailing raw-data structure with exact shape evidence for all files.
* **Completion criteria:** The technical structure (shape, type, dimension meaning) of every raw file is established.
* **Dependencies:** Sprint A1.
* **What must NOT be done yet:** Any form of preprocessing, filtering, or cross-referencing with clinical groups.
* **Suggested team ownership:** Person 2

### Sprint A3 — Subject Metadata
* **Objective:** Establish the structure and meaning of subject groupings.
* **Questions to answer:** What are the subject identifiers? How are HC/ADD/ADHD groups defined? What is the group ordering? How many subjects are in each group? Do subject names map to raw data ordering?
* **Main tasks:** Analyze `sub_name_stim.mat`, count unique subjects, extract group associations.
* **Expected evidence/output:** Subject/group structure mapping.
* **Completion criteria:** Every subject is identified and assigned to a group (HC/ADD/ADHD) with exact counts matching existing knowledge (44 HC, 52 ADD, 48 ADHD).
* **Dependencies:** Sprint A1.
* **What must NOT be done yet:** Linking subjects to raw trial data (this comes later).
* **Suggested team ownership:** Person 1

### Sprint A4 — Stimulus Metadata
* **Objective:** Document the semantics and structure of the stimulus metadata.
* **Questions to answer:** What is the meaning of `y_stim.mat` rows (especially rows 1-3)? What is the trial ordering and stimulus categorization? How do subject indices relate to stimulus entries?
* **Main tasks:** Inspect `y_stim.mat`, analyze the distributions of the 3 mutually exclusive categories.
* **Expected evidence/output:** Documented stimulus metadata semantics.
* **Completion criteria:** The structure of `y_stim.mat` is completely explained (excluding clinical interpretations if not explicitly stated).
* **Dependencies:** Sprint A1.
* **What must NOT be done yet:** Assigning HC/ADD/ADHD labels to stimulus categories unless the original source strictly defines them as such.
* **Suggested team ownership:** Person 1

### Sprint A5 — Channel Metadata
* **Objective:** Understand the channel configuration and resolve discrepancies.
* **Questions to answer:** What are the channel names and coordinates in `chan.mat`? How do the 60 metadata channels map to the 56 EEG channels in `d1.mat`?
* **Main tasks:** Extract channel names/coordinates from `chan.mat`, compare against expected 10-20 system or similar standard, attempt to find the missing 4 channels.
* **Expected evidence/output:** Channel structure document.
* **Completion criteria:** 56 vs 60 channel discrepancy is resolved, OR officially documented as an unresolved limitation.
* **Dependencies:** Sprint A1, A2.
* **What must NOT be done yet:** Dropping channels in the raw data.
* **Suggested team ownership:** Person 1 or 2

### Sprint A6 — Trial / Subject / File Mapping
* **Objective:** Establish the grand mapping across all dataset dimensions.
* **Questions to answer:** How do trials order and map to subjects/groups? How do trials span d1-d7? Does d1 contain the first 5000 metadata entries? Does d2 continue from d1? What is the exact final trial count?
* **Main tasks:** Cross-reference `y_stim.mat` lengths with trial counts in d1-d7. Map index ranges across files.
* **Expected evidence/output:** A defensible index mapping: RAW EEG ↔ TRIAL ↔ SUBJECT ↔ GROUP ↔ STIMULUS.
* **Completion criteria:** Every single trial in d1-d7 can be deterministically linked to a subject and stimulus category.
* **Dependencies:** Sprints A2, A3, A4.
* **What must NOT be done yet:** Creating combined tensors or restructuring the data on disk.
* **Suggested team ownership:** User / All (Integration)

### Sprint A7 — EEG Terminology & Experimental Structure
* **Objective:** Anchor domain terminology to this specific dataset.
* **Questions to answer:** How are terms like trial, epoch, channel, sample, stimulus, and recording strictly defined for this specific dataset and the base paper?
* **Main tasks:** Review base paper and OSF docs to create a glossary mapping terms to tensor dimensions.
* **Expected evidence/output:** Domain terminology reference document.
* **Completion criteria:** Glossary is completed and used universally by the team.
* **Dependencies:** None.
* **What must NOT be done yet:** Assuming standard MNE/EEGLAB definitions without verifying they apply here.
* **Suggested team ownership:** User

### Sprint A8 — Dataset ↔ Base Paper Alignment
* **Objective:** Audit the actual dataset against the He et al. (2023) paper.
* **Questions to answer:** Does the dataset match the paper's claims regarding subjects, sampling rate, trial count, preprocessing state, etc.?
* **Main tasks:** Line-by-line comparison of paper's data section vs. our findings from A1-A6.
* **Expected evidence/output:** Base-paper ↔ actual-dataset alignment audit (documenting every discrepancy).
* **Completion criteria:** All discrepancies between paper claims and actual OSF files are identified and documented.
* **Dependencies:** Sprints A1-A7.
* **What must NOT be done yet:** Assuming the paper is right and the data is wrong.
* **Suggested team ownership:** User

### Sprint A9 — Raw Dataset Visualization
* **Objective:** Visually confirm the raw data structure and signal integrity.
* **Questions to answer:** What do the raw signals look like? Is the data already filtered/referenced? What does channel x time look like for a single trial?
* **Main tasks:** Plot a single channel, multiple channels, and a complete trial. Check for obvious artifacts or pre-applied filtering.
* **Expected evidence/output:** Basic raw-data visualization plots and interpretation notes.
* **Completion criteria:** Code exists to plot any given trial/channel directly from the raw `.mat` files.
* **Dependencies:** Sprint A2, A6.
* **What must NOT be done yet:** Automated artifact rejection, bandpass filtering, or feature extraction.
* **Suggested team ownership:** Person 2

### Sprint A10 — Dataset Audit / Freeze
* **Objective:** Finalize Phase A and declare the dataset structure understood.
* **Questions to answer:** What is established? What is strongly supported? What is unresolved? What are the limitations? What can this dataset reasonably support?
* **Main tasks:** Compile findings from A1-A9 into a master audit document.
* **Expected evidence/output:** The Final Dataset Audit (Established, Supported, Unresolved, Limitations, Definition, Implications).
* **Completion criteria:** Audit is reviewed and accepted by all team members. **MILESTONE 1 ACHIEVED.**
* **Dependencies:** Sprints A1-A9.
* **What must NOT be done yet:** Moving to Phase B before the audit is signed off.
* **Suggested team ownership:** User

---

## Phase B: Literature & Research Landscape
*Objective: Understand how this dataset and similar ones are used, identifying genuine, evidence-backed research gaps.*

### Sprint B1 — Dataset Lineage
* **Objective:** Trace the origin and history of this exact dataset cohort.
* **Expected evidence/output:** Document detailing the original dataset paper, releases, and related papers using the same cohort.
* **Dependencies:** Phase A complete.
* **Suggested team ownership:** Person 1

### Sprint B2 — Same-Dataset Literature
* **Objective:** Review papers that have used this specific dataset.
* **Expected evidence/output:** Matrix of previous work (preprocessing, target, model, split strategy, metrics, limitations).
* **Dependencies:** Sprint B1.
* **Suggested team ownership:** Person 1 or 2

### Sprint B3 — Methodological Investigation
* **Objective:** Investigate critical methodological considerations in EEG ML.
* **Expected evidence/output:** Report on leakage risks, repeated-measures issues, splitting strategies (subject vs trial), and domain approaches.
* **Dependencies:** Phase A complete.
* **Suggested team ownership:** User

### Sprint B4 — Research Gap Discovery
* **Objective:** Identify genuine, evidence-backed gaps in the existing literature.
* **Expected evidence/output:** List of candidate gaps supported by evidence (not just "nobody used model X").
* **Dependencies:** Sprints B2, B3.
* **What must NOT be done yet:** Selecting the final gap.
* **Suggested team ownership:** All

### Sprint B5 — Dataset ↔ Gap Compatibility
* **Objective:** Filter research gaps based on what our dataset actually supports.
* **Expected evidence/output:** Feasibility analysis for each candidate gap (baselines, evidence needed).
* **Dependencies:** Sprint B4.
* **Suggested team ownership:** User

---

## Phase C: Research Question & Experiment Design
*Objective: Formulate the research question and design rigorous experiments.*

### Sprint C1 — Candidate Research Questions
* **Objective:** Generate multiple evidence-backed questions based on Phase B.
* **Dependencies:** Phase B complete.
* **Suggested team ownership:** All

### Sprint C2 — Research Question Selection
* **Objective:** Select ONE research question based on significance, feasibility, and methodological rigor.
* **Dependencies:** Sprint C1.
* **Suggested team ownership:** User

### Sprint C3 — Experimental Design
* **Objective:** Define the exact methodology to answer the research question.
* **Expected evidence/output:** Document specifying splits, preprocessing, representations, baselines, metrics, and leakage controls.
* **Dependencies:** Sprint C2.
* **Suggested team ownership:** User

### Sprint C4 — Experiment Protocol
* **Objective:** Freeze the protocol before any large-scale implementation.
* **Expected evidence/output:** Finalized Experiment Protocol document.
* **Dependencies:** Sprint C3.

---

## Phase D: Data Preparation
*Objective: Preprocess and structure the data according to the frozen experimental design.*

### Sprint D1 — Preprocessing Reconstruction
* **Objective:** Reconstruct necessary preprocessing pipelines from literature.
* **Dependencies:** Phase C complete.

### Sprint D2 — Preprocessing Implementation
* **Objective:** Write reproducible preprocessing code.
* **Dependencies:** Sprint D1.

### Sprint D3 — Quality Validation
* **Objective:** Verify signal integrity, lack of leakage, and label correctness after preprocessing.
* **Dependencies:** Sprint D2.

### Sprint D4 — Experimental Dataset Construction
* **Objective:** Create versioned, ready-to-train dataset splits.
* **Dependencies:** Sprint D3.

---

## Phase E: Experiments
*Objective: Run baselines and proposed methods to gather empirical evidence.*

### Sprint E1 — Baseline
* **Objective:** Implement the simplest scientifically justified baseline (e.g., simple linear model on raw features).

### Sprint E2 — Literature Baseline
* **Objective:** Reproduce a relevant existing method if feasible.

### Sprint E3 — Proposed Method
* **Objective:** Implement the method specifically justified by the research question.

### Sprint E4 — Main Experiments
* **Objective:** Run controlled experiments comparing proposed method vs baselines.

### Sprint E5 — Ablation / Sensitivity
* **Objective:** Test component importance and hyperparameter sensitivity.

---

## Phase F: Validation & Analysis
*Objective: Rigorously evaluate the empirical results.*

### Sprint F1 — Statistical Evaluation
* **Objective:** Perform statistical significance testing on the results.

### Sprint F2 — Subject-Level Generalization
* **Objective:** Validate performance on unseen subjects (if applicable to split strategy).

### Sprint F3 — Robustness & F4: Error Analysis
* **Objective:** Analyze failure modes, model robustness, and systematic errors.

### Sprint F5 — Explainability (If relevant)
* **Objective:** Interpret model decisions in physiological terms.

### Sprint F6 — Final Results Audit
* **Objective:** Consolidate all evidence into a final review before writing.

---

## Phase G: Paper & Reproducibility
*Objective: Convert evidence into a publishable paper and public repository.*

### Sprint G1 — Results Consolidation
### Sprint G2 — Research Paper Drafting
### Sprint G3 — Figures/Tables Generation
### Sprint G4 — Reproducibility Audit
### Sprint G5 — Repository Finalization
