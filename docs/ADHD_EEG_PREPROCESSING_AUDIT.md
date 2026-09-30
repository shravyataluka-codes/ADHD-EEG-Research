# ADHD EEG — Dataset & Preprocessing Audit

Status: AUDIT IN PROGRESS / PREPROCESSING NOT AUTHORIZED

This is a mandatory research checkpoint. It records a read-only structural audit and a methodology review; it is not permission to transform or write EEG data. The previous audit at [dataset-preprocessing-audit.md](dataset-preprocessing-audit.md) is preserved and remains part of the project history. Where that document described sequential file ordering as strongly supported, this audit states the limitation more strictly: the EEG blocks contain no embedded trial identifiers, so the file-to-metadata join has not been independently proven.

Evidence labels used throughout are defined in Section 14. Statements about expected joins are marked as conditional/inference rather than as binary facts.

## 1. Research objective

Investigate three-class EEG classification for ADHD, ADD, and healthy controls (HC), with a future Transformer-family model as a possible experiment. The research objective includes evaluating generalization to unseen participants, not merely classifying additional trials from known participants. The base paper reports an EEG-Transformer; faithful reproduction and a leakage-controlled evaluation are separate goals that still require a researcher decision. [VERIFIED FROM PAPER]

No model or preprocessing workflow was run for this audit. [VERIFIED FROM CODE / audit activity]

## 2. Dataset provenance

- **OSF source:** OSF project `6594x`, <https://osf.io/6594x/>; the official storage API lists the `data` folder and the files inventoried below. [VERIFIED FROM OSF DOCUMENTATION]
- **Base paper:** Yuchao He et al., “Classification of attention deficit/hyperactivity disorder based on EEG signals using a EEG-Transformer model,” *Journal of Neural Engineering* 20 (2023) 056013, DOI <https://doi.org/10.1088/1741-2552/acf7f5>. It reports 144 children (48 ADHD, 52 ADD, 44 HC), 33,902 trials, 56 channels, 256 Hz, and 1.5-second experiments. [VERIFIED FROM PAPER]
- **OSF documentation:** Existing project investigation records the OSF `readme.txt` statement that the large sample data is split across files. OSF's file listing supplies names, byte sizes, and hashes; it does not establish the per-trial join between `d1`–`d7` and `y_stim`. [VERIFIED FROM CODE / UNKNOWN for the join]
- **Local acquisition:** All ten requested files are present locally. The raw EEG file byte sizes match the recorded OSF listing sizes. No source file was written or transformed during this read-only audit. [VERIFIED FROM BINARY DATA]
- **Local supporting records:** [base-paper-analysis.md](base-paper-analysis.md), [dataset-investigation.md](dataset-investigation.md), and [dataset-preprocessing-audit.md](dataset-preprocessing-audit.md) contain prior research observations. These are evidence records, not substitutes for missing source provenance.

## 3. Raw dataset inventory

All `.mat` sizes below are local byte counts. The `d*` files are MATLAB v7.3/HDF5 files containing one numeric dataset named for the file. Their local dataset headers report `float32` and MATLAB class `single`. [VERIFIED FROM BINARY DATA]

| File | Bytes | Verified structure |
|---|---:|---|
| `d1.mat` | 402,642,207 | HDF5 dataset `d1`, shape `(385, 56, 5000)`, `float32`, gzip |
| `d2.mat` | 402,606,807 | HDF5 dataset `d2`, shape `(385, 56, 5000)`, `float32`, gzip |
| `d3.mat` | 402,837,633 | HDF5 dataset `d3`, shape `(385, 56, 5000)`, `float32`, gzip |
| `d4.mat` | 402,887,010 | HDF5 dataset `d4`, shape `(385, 56, 5000)`, `float32`, gzip |
| `d5.mat` | 402,691,834 | HDF5 dataset `d5`, shape `(385, 56, 5000)`, `float32`, gzip |
| `d6.mat` | 402,722,744 | HDF5 dataset `d6`, shape `(385, 56, 5000)`, `float32`, gzip |
| `d7.mat` | 314,191,152 | HDF5 dataset `d7`, shape `(385, 56, 3902)`, `float32`, gzip |
| `y_stim.mat` | 1,153,360 | HDF5 dataset `y_stim`, shape `(4, 33902)`, `float64` |
| `sub_name_stim.mat` | 70,128 | HDF5 cell array `sub_name_stim`, shape `(1, 3)`; referenced blocks contain 44, 52, and 48 unique filename strings |
| `chan.mat` | 3,135 | MATLAB struct `chan`, shape `(1, 60)`, with channel metadata fields including `labels`, `unit`, coordinate fields, `type`, and `ref` |

The EEG datasets are chunked `(192, 56, 1)` and gzip-compressed. Their stored HDF5 shape and dtype were inspected without reading or rewriting EEG samples. [VERIFIED FROM BINARY DATA]

## 4. Dataset structural validation

### EEG dimensions and concatenation

- Exact observed shape for `d1`–`d6`: `(385, 56, 5000)` each. Exact observed shape for `d7`: `(385, 56, 3902)`. All seven dimensions agree in their first two axes and dtype; only the trial dimension differs. [VERIFIED FROM BINARY DATA]
- Per-file trial counts: `5000, 5000, 5000, 5000, 5000, 5000, 3902`. Their sum is exactly `33,902`, equal to the number of `y_stim` columns. [VERIFIED FROM BINARY DATA]
- The observed channel axis is 56. The observed first axis is 385 positions. Treating these axes as sample/time positions and channels respectively is consistent with the paper/model input `(385, 56)`, but the exact physical interpretation and endpoint convention are not declared in the binary headers. [VERIFIED FROM BINARY DATA for dimensions; VERIFIED FROM PAPER / STRONGLY SUPPORTED for axis meaning]
- The paper reports 256 Hz and 1.5 seconds. A nominal 1.5 seconds at 256 Hz gives 384 intervals/samples depending on endpoint convention; it does not explain the extra observed position. The 385th position is unresolved and must not be cropped, padded, or assigned a meaning without evidence. [VERIFIED FROM PAPER / UNKNOWN]
- Summing the files in numeric order gives a deterministic *candidate* global index: `d1` trials 1–5,000, `d2` 5,001–10,000, `d3` 10,001–15,000, `d4` 15,001–20,000, `d5` 20,001–25,000, `d6` 25,001–30,000, `d7` 30,001–33,902. The corresponding candidate `y_stim` columns use those same 1-based positions (or zero-based half-open ranges `[0:5000)`, `[5000:10000)`, `[10000:15000)`, `[15000:20000)`, `[20000:25000)`, `[25000:30000)`, `[30000:33902)`). [VERIFIED FROM BINARY DATA for file sizes/counts and y columns; INFERENCE for the join]
- Numeric file order plus exact counts produces 33,902 positions with no arithmetic gap or excess. This does **not** establish that the EEG trials inside each file were written in that same order as the metadata columns. No trial ID or checksum cross-reference exists in the inspected EEG dataset headers. [VERIFIED FROM BINARY DATA / UNKNOWN for independent ordering evidence]

### Sampling and channel information

- Sampling rate 256 Hz and trial duration 1.5 seconds are paper-reported, not encoded in the EEG array headers. [VERIFIED FROM PAPER]
- `chan.mat` has 60 named channel entries; examples include `Cz`, `FCz`, `Fz`, `Fp1`, `Fp2`, `Oz`, and `Iz`. The EEG arrays have 56 channels and contain no channel-name array in the inspected dataset. The exact 56-to-60 column mapping and the identity of the four unrepresented metadata entries are unknown. [VERIFIED FROM BINARY DATA / UNKNOWN]
- Do not infer that the four entries are EOG, ECG, reference, mastoid, or otherwise auxiliary; that is not established by the inspected evidence. [UNKNOWN]

## 5. Trial → subject → clinical-group mapping

### Exact candidate methodology

1. Interpret `y_stim` row 0 as the within-block subject index: observed values are integer-valued and reset to 1 at the starts of the three blocks. It produces exactly 144 contiguous subject runs: indices 1–44, then 1–52, then 1–48, with no missing, invalid, or non-contiguous index run inside a block. [VERIFIED FROM BINARY DATA]
2. Interpret the three `sub_name_stim` referenced cell blocks in order. Their lengths are 44, 52, and 48; every filename is unique both within and across the three blocks. All filenames in block 1 contain `controls`, block 2 contain `subtype1`, and block 3 contain `subtype2`. [VERIFIED FROM BINARY DATA]
3. Candidate identity for a trial is therefore `(metadata block, row-0 subject index)`. If the cell-array order corresponds to row-0 index order, metadata entry `i` in that block supplies its participant filename. A globally unique split group key must retain both the block/group and the within-block index; row 0 by itself repeats across blocks. [STRONGLY SUPPORTED for the block/index relationship; INFERENCE for filename ordinal linkage]
4. Candidate global trial interval for a subject is the contiguous run of that subject's row-0 value inside its block. The per-subject intervals are exactly reconstructible from the starting global position below plus the cumulative sum of earlier subject counts in the ordered count vector. [VERIFIED FROM BINARY DATA for row runs; INFERENCE for assigning those runs to EEG files]
5. Candidate clinical group assignment is block 1 = HC/controls, block 2 = ADD/subtype1, block 3 = ADHD/subtype2. This interpretation is strongly supported by filename tokens and the paper's subject and trial counts, but the MAT files do not provide an explicit diagnosis field keyed to each participant. [STRONGLY SUPPORTED, not VERIFIED FROM BINARY DATA]

### Group boundaries and counts

| Candidate block | Subject indices | Candidate global trial range, 1-based | Subjects | Trials | Basis |
|---|---:|---:|---:|---:|---|
| controls / HC | 1–44 | 1–10,129 | 44 | 10,129 | Metadata block order and row-0 runs; group label inferred from names and paper |
| subtype1 / ADD | 1–52 | 10,130–23,160 | 52 | 13,031 | Metadata block order and row-0 runs; group label inferred from names and paper |
| subtype2 / ADHD | 1–48 | 23,161–33,902 | 48 | 10,742 | Metadata block order and row-0 runs; group label inferred from names and paper |

The three trial totals match the paper's reported HC, ADD, and ADHD trial counts exactly. [VERIFIED FROM BINARY DATA for row-run totals; VERIFIED FROM PAPER for published totals; STRONGLY SUPPORTED for the group alignment]

### Trial count per subject

Within each line, the first number is the count for subject index 1, the second for index 2, and so on, in the corresponding `sub_name_stim` block order. These counts are direct lengths of the 144 contiguous `y_stim` row-0 runs. To obtain each candidate subject's exact global trial interval, cumulatively sum counts from the start of that group's range in the preceding table. This representation avoids silently asserting that the EEG trial order has been joined to those metadata runs. [VERIFIED FROM BINARY DATA for counts/order; INFERENCE for name-to-index linkage]

- **controls / candidate HC, subject indices 1–44:** 281, 262, 295, 229, 255, 135, 232, 254, 159, 272, 279, 168, 137, 237, 205, 115, 195, 229, 222, 209, 160, 229, 203, 99, 229, 241, 79, 287, 102, 279, 278, 276, 280, 241, 288, 273, 267, 259, 273, 279, 292, 290, 277, 278.
- **subtype1 / candidate ADD, subject indices 1–52:** 266, 249, 220, 276, 265, 224, 269, 276, 252, 272, 264, 263, 255, 231, 264, 206, 199, 239, 191, 248, 255, 172, 262, 269, 226, 199, 267, 268, 267, 282, 270, 275, 219, 278, 290, 245, 281, 249, 253, 294, 289, 290, 283, 263, 197, 251, 257, 258, 171, 259, 184, 279.
- **subtype2 / candidate ADHD, subject indices 1–48:** 257, 279, 46, 259, 205, 262, 214, 262, 261, 265, 285, 255, 213, 238, 168, 125, 239, 195, 286, 197, 234, 203, 223, 265, 255, 226, 264, 259, 260, 290, 287, 229, 172, 219, 217, 129, 187, 51, 269, 234, 124, 204, 262, 254, 142, 270, 219, 283.

### Ambiguity checks and split safety

- No invalid subject index, missing subject run, duplicate subject filename, or non-one-hot stimulus column was found. The row-0 runs cover all 33,902 columns, have positive lengths, and partition into 44/52/48 subjects. [VERIFIED FROM BINARY DATA]
- `y_stim` shape is `(4, 33902)`. Rows 1–3 are binary and exactly one of the three indicators is active for every column. Their column sums are 10,129, 13,031, and 10,742. These are stimulus/task indicators by structure and paper-count alignment, not clinical diagnostic labels. Their physical category names remain unknown. [VERIFIED FROM BINARY DATA for structure; STRONGLY SUPPORTED for stimulus interpretation; UNKNOWN for category meanings]
- **Mapping result:** Metadata-based subject grouping is deterministic if block order and within-block index semantics are accepted. The candidate sequential d-file-to-column map is also deterministic arithmetically. However, no independent trial identifier proves the d-file order-to-`y_stim` column join or the `sub_name_stim` filename ordinal-to-row-0 index join. Therefore the full trial-to-person crosswalk is **not yet validated**, and the candidate key is **not yet authorized as a safe split key**. Obtain authoritative ordering confirmation or other independent evidence before treating it as validated. [VERIFIED FROM BINARY DATA for internal consistency; UNKNOWN for cross-file provenance]

## 6. Metadata interpretation

| Metadata | Verified observation | Interpretation and limit |
|---|---|---|
| `y_stim` | `(4, 33902)`; row 0 has three sequential index blocks; rows 1–3 are exhaustive one-hot indicators | Row 0 is a within-block participant index [STRONGLY SUPPORTED]. Rows 1–3 are distinct trial categories [STRONGLY SUPPORTED]; exact physical names are UNKNOWN. They must not be used as ADHD/ADD/HC targets. |
| `sub_name_stim` | Three ordered referenced blocks with 44, 52, and 48 unique filename strings | Filename tokens are `controls`, `subtype1`, `subtype2` respectively [VERIFIED FROM BINARY DATA]. Interpreting them as HC, ADD, ADHD is STRONGLY SUPPORTED by paper counts, but is not an explicit participant diagnosis field. |
| `chan` | MATLAB struct with 60 labels and associated channel fields | EEG data has 56 columns. Exact mapping and four absent entries are UNKNOWN; no spatial location mapping is approved. |

## 7. Proposed preprocessing pipeline

These are candidates to audit, not instructions to execute:

1. Load or stream the seven raw EEG arrays and attach trial metadata.
2. Preserve or reinterpret the observed `(385, 56)` trial representation.
3. Filter (the project guide previously proposed 0.5–45 Hz, fourth-order Butterworth).
4. Apply 50/60 Hz notch filtering.
5. Run ICA or other adaptive artifact cleaning.
6. Reject or repair artifact trials/channels.
7. Baseline-correct trials.
8. Re-epoch, crop, or otherwise handle 385 positions as nominal 1.5-second epochs.
9. Normalize or standardize raw samples/features.
10. Derive spectral powers (delta/theta/alpha/beta/gamma), theta/beta ratios, or other features.
11. Construct clinical labels from subject/group metadata while retaining `y_stim` separately.
12. Map channels to physical locations or build spatial/topographic representations.
13. Split train/validation/test data and select class weighting or sampling.
14. Write a derived representation for a future Transformer/Conformer model.
15. Choose a bounded-memory loading and transformation strategy.

## 8. Methodological audit

| Step | Proposed operation | Evidence | Risk | Leakage risk | Decision |
|---|---|---|---|---|---|
| Trial-file-to-metadata crosswalk | Join each EEG trial to `y_stim` and participant metadata | File counts and metadata totals align, but EEG blocks have no embedded trial identifiers [VERIFIED FROM BINARY DATA / UNKNOWN] | A wrong ordering creates systematically incorrect subjects and targets | Invalid groups can make a subject split appear safe while leaking participants | UNRESOLVED |
| Sample preservation | Keep all 385 positions and raw `float32` values unchanged pending clarification | Binary shape; paper model input uses 385 positions [VERIFIED FROM BINARY DATA / VERIFIED FROM PAPER] | Physical meaning of final position is unknown | Low if unchanged; high if data-driven crop chosen using held-out results | CONDITIONAL: preserve; no crop/pad approval |
| Channel handling | Retain 56 anonymous channel columns; do not map to `chan` or spatial positions yet | 56 data columns vs 60 channel labels [VERIFIED FROM BINARY DATA] | Incorrect mapping can corrupt spatial claims | Channel selection/map chosen using full data risks test contamination | CONDITIONAL: retain raw order only; physical mapping requires evidence and researcher approval |
| Bandpass filtering | Proposed 0.5–45 Hz, fourth-order Butterworth | Only prior project-guide proposal; no justified source parameter found [UNKNOWN] | Removes signal content; phase/edge behavior unspecified | Filter choice/tuning on all participants contaminates test decisions | REJECTED as default; any alternative needs a prespecified, approved experiment |
| Notch filtering | Proposed 50 or 60 Hz | No line frequency or contamination evidence established [UNKNOWN] | Removes nearby content and can ring | Data-driven decision on all splits can contaminate held-out set | REJECTED as default; researcher decision required for a justified ablation |
| ICA | Re-run ICA to remove ocular/cardiac artifacts | Paper says original data underwent ICA but reports no algorithm, components, or rejection rules [VERIFIED FROM PAPER / UNKNOWN] | May double-clean already cleaned data; no auxiliary channel map | Adaptive fit/component selection can leak if fitted across partitions | REJECTED as default; establish source state and protocol before reconsideration |
| Artifact handling | Reject, interpolate, or repair trials/channels | No thresholds, labels, or supported rejection rule in available evidence [UNKNOWN] | Selective data loss and differential group bias | Global thresholds or review can use test distribution/labels | RESEARCHER DECISION REQUIRED; do not reject/repair by assumption |
| Baseline correction | Subtract a baseline interval | Baseline interval and applicability are undocumented [UNKNOWN] | May alter amplitudes and cannot be specified from current epoch metadata | Baseline learned/selected across trials can contaminate held-out evaluation | REJECTED as default |
| Epoch handling | Re-epoch or force 1.5 s/384 samples | Arrays are already trial-shaped with 385 positions [VERIFIED FROM BINARY DATA]; source paper reports 1.5 s [VERIFIED FROM PAPER] | Boundaries and extra sample unresolved; crop would discard data | Boundary selection based on model/test outcomes contaminates evaluation | REJECTED; preserve existing trial boundaries and all positions |
| Label construction | Use subject group as clinical target; retain `y_stim` separately | Group filenames/counts align with paper; no explicit diagnosis field [STRONGLY SUPPORTED] | Wrong block/order join gives systematic mislabels | Label-informed trial or subject mapping could contaminate evaluation | CONDITIONAL: only after authoritative crosswalk validation |
| Normalization | Per-sample, per-channel, or feature scaling | Paper does not specify it [UNKNOWN]; models may require numerical scaling [INFERENCE] | Can erase biologically meaningful amplitude differences | Global or per-dataset fitted statistics leak held-out information | CONDITIONAL: compare prespecified choices; fit statistics on training subjects only |
| Spectral features / TBR | Welch/band power or theta/beta ratio | Proposed by project guide; not established as required in paper [UNKNOWN] | PSD/band settings and biomarker interpretation are unspecified | Feature/band selection on full data leaks test information | RESEARCHER DECISION REQUIRED; separate experiment, not mandatory preprocessing |
| Subject-level split | Keep all trials for a participant in one fold | Multiple trials per participant; 144 participants [VERIFIED FROM BINARY DATA / VERIFIED FROM PAPER] | Only 144 groups; group/class balance may vary | Prevents direct subject overlap if crosswalk is correct | CONDITIONAL: required primary strategy after crosswalk is proven; freeze folds and seed |
| Trial-level split | Randomly split individual trials | Repeated trials per participant [VERIFIED FROM BINARY DATA] | Inflated performance from participant-specific signatures | Direct subject leakage | REJECTED for the primary unseen-subject claim |
| Class imbalance | Weight, resample, or report class-aware metrics | Candidate trial totals 10,129/13,031/10,742; participant counts 44/52/48 [VERIFIED FROM BINARY DATA / STRONGLY SUPPORTED group labels] | Unequal trials per participant (observed counts range 46–295) can let high-trial participants dominate | Weights/sampling calculated using validation/test labels leak | RESEARCHER DECISION REQUIRED; report class-wise metrics; fit any weights on training data only |
| Adaptive preprocessing | Learn filters, ICA, thresholds, channel/feature choices | Such operations require choices/fitted parameters [INFERENCE] | Unspecified transforms may change signal meaning | High unless fitted/selected within training folds only | CONDITIONAL: training-fold-only pipeline; no full-dataset fitting |
| Feature selection | Select bands/channels/features | No feature-selection protocol established [UNKNOWN] | Instability and biased feature claims | High if selection precedes split or inspects test results | CONDITIONAL: training-fold only; otherwise reject |
| Output representation | Keep `(trials, 385, 56)` values with separate validated subject, group, and stimulus metadata | Raw shape `385 x 56`; paper model input reports 385 x 56 [VERIFIED FROM BINARY DATA / VERIFIED FROM PAPER] | Spatial semantics, axis convention, label join, and units remain partly unresolved | Metadata leakage if target/group fields enter inputs | CONDITIONAL: candidate only after mapping and model-contract approval; raw source stays separate |
| Memory strategy | Read filewise or in bounded blocks; avoid full concatenation/copies | Seven raw files total about 2.73 GB; HDF5 datasets are chunked [VERIFIED FROM BINARY DATA] | Whole-array copies increase RAM use and accidental mutation risk | Memory handling itself is low risk; accidental global-stat scans can leak | APPROVED as a future engineering constraint only; it does not authorize processing now |

## 9. Data leakage audit

- **Trial-level splitting:** Rejected for the primary unseen-participant evaluation. Trials from one child must not cross partitions. [STRONGLY SUPPORTED]
- **Subject leakage:** The candidate key is `(group block, within-block subject index)`, not row 0 alone. Because EEG-to-metadata ordering and filename-to-index ordering remain unproven, no split may rely on the candidate mapping until corroborated. [INFERENCE / UNKNOWN]
- **Normalization leakage:** Fit means, variances, clipping limits, imputation values, and any other learned transform using training participants only, separately inside each training fold. Do not calculate them once over the whole dataset. [MODEL REQUIREMENT]
- **Adaptive preprocessing leakage:** ICA, learned filters, artifact thresholds, channel selection, feature selection, and data-driven preprocessing decisions must be fit/selected inside training folds. Any permitted per-trial deterministic operation must be specified before inspecting held-out outcomes. [MODEL REQUIREMENT]
- **Feature-selection leakage:** Select features/bands/channels only within the training partition and nested validation procedure; the final test set cannot influence the selection. [MODEL REQUIREMENT]
- **Test-set contamination:** Reserve test participants and labels from choices about preprocessing, epoch/sample handling, labels, features, architecture, hyperparameters, stopping, and interpretation. Use the test set only for the frozen final evaluation. [MODEL REQUIREMENT]
- **Class weights and sampling:** Estimate any weights or resampling plan from training participants only. Trial count imbalance and unequal trials per participant must be reported; do not silently treat all trials as independent people. [VERIFIED FROM BINARY DATA / MODEL REQUIREMENT]

## 10. Reproducibility audit

- **Parameters justified by evidence:** EEG array dimensions and `float32` dtype; per-file trial counts; paper-reported 56 channels, 256 Hz and 1.5 seconds; paper model input length 385. These facts do not justify a filter, reference, baseline, artifact threshold, unit, or the semantics of position 385. [VERIFIED FROM BINARY DATA / VERIFIED FROM PAPER]
- **Parameters not justified:** Filter cutoffs/order/phase/edge handling; line frequency; ICA algorithm and rejection rules; artifact thresholds; baseline window; sample-385 meaning; channel-to-label map/reference; amplitude units; normalization scope; PSD window/overlap/bands; feature selection; fold count; validation/test proportions; model hyperparameters; output storage format. [UNKNOWN]
- **Random seeds:** No seed was justified or selected. A seed must be recorded when a split/model experiment is approved; deterministic subject groups remain a prerequisite. [UNKNOWN]
- **Fold strategy:** Subject-grouped evaluation is required for the primary unseen-subject claim; `StratifiedGroupKFold` is a candidate, not yet frozen. Fold count, fixed test protocol, stratification target, and repeated-run policy require researcher approval. [MODEL REQUIREMENT / RESEARCHER DECISION REQUIRED]
- **Software versions available in the inspection environment:** Python 3.13.2, NumPy 2.3.5, SciPy 1.17.0, h5py 3.15.1. These identify the environment used for header/metadata inspection only; they are not a frozen preprocessing environment. [VERIFIED FROM CODE]
- **Raw preservation:** Treat all original `.mat` files as immutable inputs. Do not overwrite them, use a separate versioned output location for any future authorized derivatives, and record source hashes plus software/configuration/parameters before and after a future approved run. This audit did not write derived EEG data. [MODEL REQUIREMENT / VERIFIED FROM BINARY DATA for this audit activity]

## 11. Final recommended preprocessing protocol

This is a conservative recommendation for a future protocol review, not authorization to execute it.

### A. Required operations

- Preserve raw `.mat` files bit-for-bit and keep raw and derived data in separate locations.
- Before any transformation, obtain independent confirmation of the EEG-file trial order against `y_stim`, and confirm the `sub_name_stim` filename/index convention and clinical diagnosis mapping.
- Keep the observed trial shape and all 385 positions until the sample endpoint and axis semantics are resolved; do not infer a channel-name mapping.
- For a primary unseen-subject analysis, keep every participant's trials in exactly one partition and use globally unique subject keys. Fit learned preprocessing and model parameters on training data only.
- After explicit authorization, process in bounded blocks and preserve an auditable manifest of source hashes, parameters, versions, split IDs, and output paths.

### B. Optional approved experiments

- None are approved at this checkpoint. A prespecified comparison of normalization or spectral representations may be proposed later, with all learned choices fit within training folds and a held-out test protocol frozen in advance.

### C. Rejected operations

- Do not randomly split trials for the primary unseen-subject claim.
- Do not crop the 385th position, rebuild epochs, assume the missing four channels, or assign physical meanings to `y_stim` categories.
- Do not apply the guide's 0.5–45 Hz Butterworth filter, a 50/60 Hz notch, repeat ICA, artifact rejection, or baseline correction as an undocumented/default pipeline.
- Do not globally normalize, select features, set weights, or tune preprocessing using all trials or test participants.

### D. Decisions still requiring researcher approval

- Authoritative per-trial crosswalk and clinical diagnosis mapping.
- Meaning of all channel columns, channel reference/units, and the 56-to-60 metadata discrepancy.
- Meaning of the 385th sample and whether to preserve it in the model contract.
- Whether the goal is paper reproduction, leakage-controlled evaluation, or both.
- Any filtering, notch, ICA, artifact, baseline, normalization, or feature-engineering experiment and its exact parameters.
- Class imbalance handling, participant weighting, fold count/seed, validation and test design, metrics, model input/output contract, software environment, storage format, and output location.

## 12. Approval checklist

- [x] Dataset structure validated (exact local shapes, dtypes, and trial counts checked)
- [ ] Trial mapping validated (candidate sequential mapping is not independently proven)
- [ ] Subject mapping validated (metadata index runs checked; EEG and filename joins remain conditional)
- [ ] Clinical labels validated (filename/paper alignment is strongly supported, not an explicit participant diagnosis map)
- [ ] Channel handling approved
- [ ] Sample handling approved
- [ ] Filtering decision approved
- [ ] ICA decision approved
- [ ] Artifact handling approved
- [ ] Normalization strategy approved
- [ ] Leakage controls approved
- [ ] Train/validation/test strategy approved
- [ ] Output format approved
- [ ] Preprocessing protocol frozen

## 13. Audit conclusion

The binary structure, exact per-file dimensions/counts, metadata index runs, uniqueness checks, and candidate group trial totals have been validated. The full cross-file trial-to-subject-to-clinical-group mapping is not independently proven, several signal-processing meanings remain unknown, and researcher approvals and a frozen evaluation protocol are still missing. The audit is therefore **INCOMPLETE**. Preprocessing is not authorized.

"RAW DATA MUST REMAIN IMMUTABLE UNTIL THIS AUDIT IS COMPLETE AND THE PREPROCESSING PROTOCOL HAS BEEN FROZEN."

"NO PREPROCESSING HAS BEEN EXECUTED."

## 14. Evidence classification

Every material finding and methodological judgment above is tagged or explicitly tied to one of these evidence classes:

- **VERIFIED FROM BINARY DATA**: Directly observed local file sizes, HDF5/MAT structures, shapes, dtypes, values, counts, or metadata contents.
- **VERIFIED FROM PAPER**: Explicitly reported by He et al. in the cited base paper.
- **VERIFIED FROM CODE**: Directly recorded by inspected project documentation/scripts or the inspection environment. Existing repository statements about OSF readme content are identified as such; they are not treated as proof of trial ordering.
- **STRONGLY SUPPORTED**: Multiple independent observations align, but the claim is not explicit or directly linked by the dataset.
- **INFERENCE**: A plausible interpretation or mapping rule that requires confirmation before it is treated as fact.
- **UNKNOWN**: Not established by the available binary data, paper, OSF listing/readme record, or project code.

A paper-reported statement that original data underwent ICA does not establish the exact ICA procedure or prove the processing history of every local array. An exact count match does not, by itself, prove a file-to-label ordering. Neither inference is silently promoted to a verified fact.
