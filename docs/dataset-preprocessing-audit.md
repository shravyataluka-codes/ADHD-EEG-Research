# ADHD EEG Dataset — Preprocessing Audit

Status: PREPROCESSING NOT YET EXECUTED  
Raw data: PRESERVED  
Purpose: Research methodology record

This document records a read-only methodology audit of the proposed preprocessing and modeling pipeline. It does not authorize preprocessing and does not replace an approved experiment protocol.

## 1. Dataset acquisition status

The remaining raw EEG files were acquired from the official OSF project associated with the base paper, node `6594x`.

| File | Size | Status |
|---|---:|---|
| `d1.mat` | 402,642,207 bytes | Existing local file; preserved |
| `d2.mat` | 402,606,807 bytes | Downloaded and size-verified |
| `d3.mat` | 402,837,633 bytes | Downloaded and size-verified |
| `d4.mat` | 402,887,010 bytes | Downloaded and size-verified |
| `d5.mat` | 402,691,834 bytes | Downloaded from the official OSF URL and size-verified |
| `d6.mat` | 402,722,744 bytes | Downloaded and size-verified |
| `d7.mat` | 314,191,152 bytes | Downloaded and size-verified |

The six missing files were downloaded from official OSF download links. Their local byte sizes match the sizes reported by the OSF listing. `d1.mat` was not replaced or rewritten. Its recorded SHA-256 after acquisition was:

`F45742C8B2BC58E473527A983139B003EF586B88AEC9E94E5A494AD098BB732C`

No preprocessing or processed-data generation was performed during acquisition or this audit.

## 2. Verified dataset structure

### 2.1 EEG files

`d1.mat` was directly inspected as an HDF5-based MATLAB file.

- Observed array shape: `(385, 56, 5000)`.
- Observed numeric type: `float32` / MATLAB `single`.
- Strongly supported interpretation: `(sample/time positions, channels, trials)`.
- The third dimension is consistent with the 5,000-trial file chunking.
- The seven-file organization is documented by the OSF dataset.
- `d1.mat` through `d6.mat` are strongly supported to contain 5,000 trials each.
- `d7.mat` is strongly supported to contain the remaining 3,902 trials.
- The total is `33,902` trials, matching `y_stim.mat` and the base paper.

Evidence classification:

- Shape, dtype, and local file facts: **VERIFIED FROM BINARY DATA**.
- Sampling rate of 256 Hz and nominal 1.5-second trials: **VERIFIED FROM PAPER**.
- Axis interpretation and sequential file chunking: **STRONGLY SUPPORTED**.
- Exact axis semantics are not explicitly stated in the OSF metadata: **UNKNOWN**.

### 2.2 Metadata files

`y_stim.mat`:

- Shape: `(4, 33902)`.
- Row 0 contains within-group subject indices and resets between subject groups.
- Rows 1-3 are mutually exclusive binary indicators.
- The three rows have counts of 10,129, 13,031, and 10,742.
- These rows represent three stimulus/task categories in structure, but their physical names are unresolved.
- They are not the HC, ADD, and ADHD diagnostic labels.

Evidence classification:

- Shape, binary structure, exclusivity, and counts: **VERIFIED FROM BINARY DATA**.
- Stimulus-category interpretation: **STRONGLY SUPPORTED**.
- Exact physical meaning of each category: **UNKNOWN**.

`sub_name_stim.mat`:

- Contains three subject blocks with 44, 52, and 48 entries.
- The total is 144 subjects.
- Filename patterns support the mapping `controls` to HC, `subtype1` to ADD, and `subtype2` to ADHD.

Evidence classification:

- Block counts and total subject entries: **VERIFIED FROM BINARY DATA**.
- Clinical-group mapping from filename patterns and matching paper counts: **STRONGLY SUPPORTED**.
- A fully explicit dataset-provided diagnostic-label mapping: **UNKNOWN**.

`chan.mat`:

- Contains 60 channel entries and standard channel names such as `Fp1`, `Fp2`, `Cz`, `FCz`, and `Oz`.
- The EEG arrays contain 56 channels.
- The identities of the four channels absent from the EEG arrays are not established.
- The absent channels must not be assumed to be EOG, ECG, reference, mastoid, or other auxiliary channels.

Evidence classification:

- 60 metadata entries: **VERIFIED FROM BINARY DATA**.
- 56 channels in the EEG array: **VERIFIED FROM BINARY DATA**.
- The exact 56-to-60 channel mapping: **UNKNOWN**.

### 2.3 Known uncertainties

1. The paper reports `256 Hz x 1.5 seconds = 384` samples, but the observed trial dimension is 385.
2. The physical meaning of the 385th sample is unknown.
3. The exact file-to-column mapping from `d1.mat` through `d7.mat` to `y_stim.mat` is strongly supported by sequential chunking but not explicitly documented.
4. The exact meanings of the three stimulus categories are unknown.
5. The diagnostic mapping from subject metadata to trial metadata requires deterministic validation.
6. The exact four channels missing between `chan.mat` and the EEG arrays are unknown.
7. The source documentation does not fully specify reference, filtering, trial rejection, or ICA implementation details.
8. The exact shapes and headers of all seven files should be independently validated before any transformation, even though their sizes and expected chunking are known.

## 3. Proposed preprocessing pipeline

The project guide proposed the following operations:

1. Load `d1.mat` through `d7.mat`.
2. Interpret each trial as a `(385, 56)` matrix.
3. Apply a 0.5-45 Hz fourth-order Butterworth bandpass filter.
4. Apply a 50/60 Hz notch filter.
5. Apply ICA to remove ocular and cardiac artifacts.
6. Reconstruct or slice 1.5-second epochs.
7. Extract delta, theta, alpha, beta, and gamma band powers.
8. Calculate per-channel theta/beta ratios.
9. Normalize or standardize the input/features.
10. Construct HC, ADD, and ADHD labels.
11. Split with subject-level `StratifiedGroupKFold`.
12. Train a spatial-spectral Conformer or Transformer.

The final two items are experimental design and model decisions rather than preprocessing operations, but they are included because they affect leakage and preprocessing requirements.

## 4. Audit of each preprocessing step

### 4.1 Load the seven EEG files

- **Decision:** Keep, with bounded or file-by-file reads.
- **Evidence:** The OSF readme states that the data is split because it is too large; file sizes are approximately 300-403 MB.
- **Justification:** Loading all seven arrays into RAM is unnecessary and conflicts with the memory-safe research requirement.
- **Risk:** A careless concatenation step could create a second large raw-data copy.
- **Approval:** Approved as an implementation constraint; exact block size remains a researcher/engineering decision.
- **Evidence classification:** **VERIFIED FROM BINARY DATA**, **VERIFIED FROM CODE**.

### 4.2 Preserve `(385, 56)` trials

- **Decision:** Keep provisionally; validate all seven headers first.
- **Evidence:** `d1.mat` directly has shape `(385, 56, 5000)`. The paper reports 56 channels and an architecture input length of 385.
- **Justification:** The extra sample is unresolved. Cropping it would change the dataset without evidence.
- **Risk:** Treating 385 as exactly 1.5 seconds may misstate the time axis.
- **Approval:** Representation is recommended, but the physical interpretation of the final sample remains unresolved.
- **Evidence classification:** **VERIFIED FROM BINARY DATA**, **VERIFIED FROM PAPER**, **STRONGLY SUPPORTED**.

### 4.3 Apply a 0.5-45 Hz bandpass filter

- **Decision:** Remove from the default pipeline.
- **Evidence:** This parameter appears in the project guide but not in the paper or available OSF documentation.
- **Justification:** A common EEG cutoff is not automatically appropriate for this dataset. The source data may already have been processed.
- **Risk:** It could remove scientifically meaningful low- or high-frequency information and make reproduction impossible.
- **Approval:** Unresolved; requires a separate approved ablation with an explicitly justified filter design.
- **Evidence classification:** **UNKNOWN**, **INFERENCE**.

### 4.4 Apply a fourth-order Butterworth filter

- **Decision:** Remove from the default pipeline.
- **Evidence:** No source specifies filter family, order, phase mode, or edge handling.
- **Justification:** Filter order and phase behavior materially affect EEG signals.
- **Risk:** Phase distortion, edge artifacts, and unreproducible results.
- **Approval:** Unresolved.
- **Evidence classification:** **UNKNOWN**.

### 4.5 Apply a 50/60 Hz notch filter

- **Decision:** Remove from the default pipeline.
- **Evidence:** Neither the paper nor the dataset documentation establishes mains contamination or the correct line frequency.
- **Justification:** A notch filter should respond to measured contamination and a known acquisition frequency, not location alone.
- **Risk:** It may remove real signal content or introduce ringing.
- **Approval:** Unresolved; requires inspection and an explicit frequency decision.
- **Evidence classification:** **UNKNOWN**, **INFERENCE**.

### 4.6 Run ICA for ocular and cardiac artifact removal

- **Decision:** Do not rerun ICA by default.
- **Evidence:** The paper states that the original EEG data were preprocessed with ICA for noise removal, but gives no algorithm, component count, rejection criteria, or implementation details.
- **Justification:** Repeating an undocumented adaptive transform may double-clean the signal and diverge from the source data.
- **Risk:** Signal distortion and adaptive leakage if ICA is fitted using validation/test data.
- **Approval:** Unresolved; requires evidence that the downloaded files are uncleaned and a fully specified ICA protocol.
- **Evidence classification:** **VERIFIED FROM PAPER**, **UNKNOWN**.

### 4.7 Reconstruct or re-slice 1.5-second epochs

- **Decision:** Remove.
- **Evidence:** The downloaded files already contain trial-shaped arrays and the paper describes 33,902 divided trials.
- **Justification:** There is no documented continuous recording available from which to reconstruct boundaries.
- **Risk:** Incorrect boundaries, duplicated samples, and loss of the unresolved 385th position.
- **Approval:** Not approved.
- **Evidence classification:** **VERIFIED FROM BINARY DATA**, **VERIFIED FROM PAPER**.

### 4.8 Extract five spectral bands with Welch PSD

- **Decision:** Do not include in the mandatory baseline.
- **Evidence:** The project guide proposes this feature branch, but the paper does not require it and the exact PSD settings are unspecified.
- **Justification:** It is a model feature experiment, not a necessary preprocessing step.
- **Risk:** Choice of windows, overlap, detrending, and bands can change the scientific result. Adaptive feature decisions can also leak test information.
- **Approval:** Requires a separate researcher-approved experiment and train-only fitting for any learned feature selection.
- **Evidence classification:** **MODEL REQUIREMENT**, **UNKNOWN**.

### 4.9 Calculate theta/beta ratio

- **Decision:** Do not include in the mandatory baseline.
- **Evidence:** TBR is proposed in the guide but is not established as a valid label-linked biomarker for this exact dataset in the available evidence.
- **Justification:** TBR may be useful, but it must be tested as a hypothesis rather than assumed to be clinically explanatory.
- **Risk:** Division instability, age/task effects, reference effects, and confirmation bias in interpretation.
- **Approval:** Requires explicit approval, exact band definitions, denominator handling, and a preregistered analysis.
- **Evidence classification:** **UNKNOWN**, **RESEARCHER DECISION REQUIRED**.

### 4.10 Normalize or standardize

- **Decision:** No global normalization. Training-only standardization is optional.
- **Evidence:** The paper does not specify normalization or scaling.
- **Justification:** Global statistics would expose validation/test distributions. Per-trial scaling could remove amplitude information.
- **Risk:** Leakage and alteration of biologically meaningful amplitude differences.
- **Approval:** Training-only standardization requires approval and a frozen statistic-fitting protocol.
- **Evidence classification:** **UNKNOWN**, **MODEL REQUIREMENT**.

### 4.11 Construct diagnostic labels

- **Decision:** Use only after deterministic subject/trial mapping is validated.
- **Evidence:** The paper reports HC, ADD, and ADHD groups. `sub_name_stim.mat` contains blocks of 44, 52, and 48 subjects. `y_stim.mat` rows 1-3 are stimulus indicators, not diagnostic labels.
- **Justification:** Diagnostic labels must come from participant-group metadata, not stimulus rows.
- **Risk:** Mislabeling trials if file order or group-block boundaries are assumed incorrectly.
- **Approval:** Requires explicit validation and approval of the mapping rule.
- **Evidence classification:** **VERIFIED FROM PAPER**, **VERIFIED FROM BINARY DATA**, **STRONGLY SUPPORTED**.

### 4.12 Use spatial channel mapping or scalp topomaps

- **Decision:** Remove from the initial pipeline.
- **Evidence:** `chan.mat` has 60 channels while the EEG arrays have 56; the missing four are unresolved.
- **Justification:** Spatial models require a verified mapping from each EEG column to a physical channel.
- **Risk:** Incorrect topographies and false physiological conclusions.
- **Approval:** Requires resolution of the 56-to-60 mapping.
- **Evidence classification:** **VERIFIED FROM BINARY DATA**, **UNKNOWN**.

### 4.13 Use subject-level `StratifiedGroupKFold`

- **Decision:** Keep provisionally.
- **Evidence:** Each subject contributes multiple trials, and the project objective is eventual ML/Transformer evaluation on clinical groups.
- **Justification:** Subject-level grouping prevents the same child from appearing in training and validation/test partitions.
- **Risk:** Class proportions may vary across folds because there are only 144 subjects.
- **Approval:** Requires a frozen fold count, seed, and test protocol.
- **Evidence classification:** **MODEL REQUIREMENT**, **RESEARCHER DECISION REQUIRED**.

## 5. Data leakage risks

### Trial-level leakage

Randomly splitting the 33,902 trials can place trials from the same child in training and test sets. The model may learn subject-specific signatures rather than diagnostic generalization. The paper does not establish that its split was subject-independent.

**Classification:** **STRONGLY SUPPORTED**, **UNKNOWN** regarding the paper's exact implementation.

### Subject-level splitting

All trials from a subject must remain in one partition for the primary generalization evaluation. Group identifiers must be globally unique because row 0 of `y_stim.mat` resets within diagnostic groups.

**Classification:** **MODEL REQUIREMENT**, **STRONGLY SUPPORTED**.

### Normalization leakage

Means, standard deviations, clipping thresholds, feature-selection thresholds, and class weights must not be calculated using validation or test subjects. Any transformation with fitted parameters must be fitted on training subjects only.

**Classification:** **MODEL REQUIREMENT**.

### Adaptive preprocessing leakage

ICA, artifact-rejection thresholds, learned filters, feature selection, and data-driven channel decisions can leak information if fitted before splitting. If ICA is ever repeated, its fitting scope and component-selection rule must be frozen and training-only where applicable.

**Classification:** **MODEL REQUIREMENT**, **RESEARCHER DECISION REQUIRED**.

### Test-set contamination

The test set must not be used to choose filters, normalization, labels, epoch handling, model architecture, hyperparameters, stopping criteria, or explanatory biomarkers. Test data should be read only for final evaluation.

**Classification:** **MODEL REQUIREMENT**.

## 6. Current recommended baseline

The minimally transformed baseline is:

1. Read one source file or bounded trial block at a time.
2. Validate file headers and expected shapes before transformation.
3. Preserve each trial as `(385, 56)`.
4. Preserve the original numeric values and `float32` dtype where possible.
5. Do not crop the 385th sample.
6. Do not re-epoch, baseline-correct, filter, notch-filter, or rerun ICA.
7. Attach only validated subject and diagnostic-group metadata.
8. Keep `y_stim` stimulus categories as separate metadata, not target labels.
9. Split strictly by globally unique subject ID.
10. Fit any optional standardization on training subjects only.
11. Write derived data only to a new, versioned output location after protocol approval.

This baseline is intended to establish a leakage-controlled reference before adding transformations or spectral/spatial feature branches.

## 7. Unresolved questions

The following questions must be answered before preprocessing begins:

1. Do all seven MAT files have the expected variable names, shapes, dtype, and axis order?
2. Is the sequential mapping from file-local trial index to `y_stim` column explicitly confirmed, or only inferred?
3. How are the 385 sample positions defined, especially the final position?
4. Are the downloaded EEG files already ICA-cleaned, and what exact source preprocessing produced them?
5. What reference scheme was used for the 56 EEG channels?
6. What are the exact identities of the four `chan.mat` entries absent from the EEG arrays?
7. What are the physical meanings of the three stimulus categories in `y_stim.mat`?
8. What deterministic rule maps each trial to a globally unique subject and clinical group?
9. Is the primary objective faithful reproduction of the base paper, leakage-controlled evaluation, or both?
10. Are filtering and notch filtering permitted in the primary experiment?
11. If filtering is approved, what frequency response, order, phase mode, and edge policy are justified?
12. Is ICA permitted, and if so, what algorithm, components, rejection criteria, and fitting scope are required?
13. Is baseline correction scientifically justified, and what documented baseline interval would be used?
14. Is normalization required by the model, and must its statistics be training-only?
15. Are spectral bands and TBR primary features, ablations, or excluded?
16. What exact fold count, random seed, test design, class weighting, and evaluation metrics will be frozen?
17. Is spatial modeling blocked until the 56-to-60 channel mapping is resolved?
18. What output format and storage budget are approved for derived data?

## 8. Approval checklist

### Dataset and labels

- [ ] Approve the expected `(385, 56, trials)` representation for all seven files.
- [ ] Approve the validated file-to-`y_stim` trial ordering.
- [ ] Approve the diagnostic label mapping from subject metadata.
- [ ] Confirm that `y_stim` stimulus rows will not be used as HC/ADD/ADHD labels.
- [ ] Approve globally unique subject IDs combining group block and within-group index.

### Signal processing

- [ ] Approve no cropping of the 385th sample.
- [ ] Approve no re-epoching.
- [ ] Decide whether any bandpass filter is allowed.
- [ ] If filtering is allowed, approve exact cutoffs, filter family, order, phase mode, and edge handling.
- [ ] Decide whether notch filtering is allowed and identify the justified line frequency.
- [ ] Decide whether the source data is already sufficiently ICA-cleaned.
- [ ] If ICA is repeated, approve the complete ICA and component-rejection protocol.
- [ ] Decide whether baseline correction is allowed and document the baseline interval.
- [ ] Decide whether normalization is required.
- [ ] If normalization is used, approve training-only statistic fitting.

### Features and model

- [ ] Decide whether spectral power features are primary features or an ablation.
- [ ] Approve exact spectral bands, PSD settings, and TBR handling if used.
- [ ] Resolve or explicitly defer the 56-to-60 channel mapping.
- [ ] Approve or defer spatial convolutions and scalp topomaps.
- [ ] Approve the Transformer/Conformer architecture and its input representation.

### Evaluation and reproducibility

- [ ] Approve subject-level grouped splitting as the primary evaluation rule.
- [ ] Approve the fold count, random seed, and fixed test protocol.
- [ ] Approve training-only fitting for all adaptive preprocessing and feature operations.
- [ ] Approve class weighting, if used, based only on training subjects.
- [ ] Freeze the preprocessing and evaluation protocol before creating derived data.
- [ ] Approve the output directory and versioning scheme for processed data.

## 9. Evidence classification

This document uses the following classifications:

- **VERIFIED FROM BINARY DATA**: Directly observed from local MAT/HDF5 contents or file metadata.
- **VERIFIED FROM PAPER**: Explicitly stated in the base research paper.
- **VERIFIED FROM CODE**: Directly established by existing repository scripts or code behavior.
- **STRONGLY SUPPORTED**: Supported by multiple observations or consistent evidence, but not explicitly documented.
- **INFERENCE**: A reasoned interpretation that is not directly established.
- **UNKNOWN**: Not established by the available evidence.

The existence of a project-guide proposal is not treated as evidence that a preprocessing step is scientifically justified. Any proposal not supported by binary data, the paper, or verified code remains an explicit researcher decision or an unresolved assumption.
