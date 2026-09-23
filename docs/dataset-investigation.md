# ADHD-EEG Dataset Investigation Audit

This document serves as a research audit of the ADHD-EEG dataset. It records what was investigated, directly observed, inferred, and what remains unknown. It also documents the decisions made and the rationale behind them.

## 1. Dataset Source

- **Source**: OSF project associated with the base research paper.
- **Available Dataset Files**: The OSF repository contains metadata files (`y_stim.mat`, `Sub_name_stim.mat`, `chan.mat`) and raw EEG data files.
- **Raw EEG Data Split**: **DIRECTLY ESTABLISHED** via OSF project inspection that the raw EEG data is split across seven files: `d1.mat` through `d7.mat`.
- **OSF Readme Statement**: **DIRECTLY ESTABLISHED** from the OSF `readme.txt` that the sample data is split because it is too large.
- **Distinction**: The base research paper describes the complete dataset and experiment. The OSF readme explains the physical file splitting. Our independent investigation confirms the physical file structure and metadata contents.

## 2. Participants / Groups

- **Total Subjects**: 144 subjects. (**PAPER-REPORTED** and **STRONGLY SUPPORTED** by dataset observation)
- **Groups**:
  - 44 Healthy Controls (HC)
  - 52 ADD
  - 48 ADHD
- **Evidence**:
  - **DIRECTLY ESTABLISHED**: `Sub_name_stim.mat` contains exactly three groups/blocks of subject entries, containing 44, 52, and 48 entries respectively. 44 + 52 + 48 = 144 total subjects.
  - **STRONGLY SUPPORTED**: Filename patterns within these entries support the mapping: `controls` corresponds to Healthy Controls (44), `subtype1` corresponds to ADD (52), and `subtype2` corresponds to ADHD (48).
- **Note**: These numbers describe the COMPLETE dataset of 144 subjects, not merely the contents of `d1.mat`.

## 3. Total Trial/Epoch Count

- **Total Trials/Epochs**: 33,902 (**DIRECTLY ESTABLISHED** for the dataset)
- **Evidence**: `y_stim.mat` has a shape of `(4, 33902)`. This provides four rows of metadata across exactly 33,902 column entries, representing the total number of epochs in the entire dataset. This aligns with the expected scale from the paper.

## 4. `y_stim.mat`

- **Description**: Contains the trial-level metadata and labels for all 33,902 epochs in the dataset.
- **Shape**: `(4, 33902)` (**DIRECTLY ESTABLISHED**)
- **Data Type**: numeric arrays (0/1 and integer indices).
- **Row 0**: **DIRECTLY ESTABLISHED** to contain values ranging from 1 to 52. The values reset sequentially when moving between subject groups. This represents the subject index WITHIN each diagnostic group.
- **Rows 1-3**:
  - **DIRECTLY ESTABLISHED** to contain 0/1 binary values.
  - **DIRECTLY ESTABLISHED** that exactly one of the three rows has a value of 1 for every trial. They are mutually exclusive.
  - **Counts**:
    - Row 1: 10,129
    - Row 2: 13,031
    - Row 3: 10,742
  - **Interpretation**: **STRONGLY SUPPORTED** that Rows 1-3 represent mutually exclusive stimulus/task categories.
- **IMPORTANT LIMITATION**: The physical meanings of these three categories (e.g., target, non-target, distractor) are **UNKNOWN / UNRESOLVED** based purely on the dataset files. Explicit physical names should not be assigned without further evidence. Furthermore, it is **DIRECTLY ESTABLISHED** that these rows do NOT represent the ADHD/ADD/HC diagnosis labels.

## 5. `Sub_name_stim.mat`

- **Description**: This file is essentially the dataset's list of participant identifiers and original source file names, organized into three distinct groups.
- **Contents**: **DIRECTLY ESTABLISHED** to contain three groups of entries: 44, 52, and 48.
- **Total**: 44 + 52 + 48 = 144 subjects.
- **Evidence**: Representative filename evidence (e.g., paths containing `controls`, `subtype1`, `subtype2`) connects these blocks to the participant groups (HC, ADD, ADHD respectively).

## 6. `chan.mat`

- **Description**: Describes the locations and names of the EEG measurement channels.
- **Contents**: **DIRECTLY ESTABLISHED** to contain 60 channel entries with standard channel names (e.g., Fp1, Fp2).
- **Discrepancy**: 
  - `chan.mat` lists 60 channels.
  - `d1.mat` contains 56 channels in its data array.
- **Status**: The exact four-channel difference is **UNKNOWN / UNRESOLVED**. It is not known which specific channels are excluded from the raw data. They cannot be assumed to be EOG, reference, or mastoid channels without explicit evidence.

## 7. `d1.mat`

- **Observation**:
  - **Shape**: `(385, 56, 5000)` (**DIRECTLY ESTABLISHED**)
  - **Data Type**: `float32` / MATLAB class: `single` (**DIRECTLY ESTABLISHED**)
- **Interpretation**:
  - The `5000` dimension is **STRONGLY SUPPORTED** to correspond to the chunk size used for splitting the EEG files.
  - The `56` dimension is **STRONGLY SUPPORTED** to correspond to the number of channels in the actual EEG data.
  - The `385` dimension is **STRONGLY SUPPORTED** to be associated with the within-epoch samples/time dimension.
- **Limitation**: The exact physical meaning of the 385th sample is **UNKNOWN / UNRESOLVED**. It should not be labeled as a baseline, padding, or zero point without further evidence.

## 8. Seven-File Organization

- **Observations**:
  - `d1.mat` – `d6.mat`: Approximately 402.6–402.8 MB each. (**DIRECTLY ESTABLISHED** via OSF)
  - `d7.mat`: Approximately 314.1 MB. (**DIRECTLY ESTABLISHED** via OSF)
  - Total epochs from `y_stim`: 33,902.
- **Analysis**:
  - If `d1` – `d6` each contain 5,000 epochs: 6 × 5,000 = 30,000 epochs.
  - Remaining epochs: 33,902 − 30,000 = 3,902 epochs.
  - The file size ratio of `d7` to `d1` (~314.1 / 402.7) is approximately consistent with the epoch ratio (3,902 / 5,000).
- **Conclusion**: It is **STRONGLY SUPPORTED** that `d1`–`d6` contain 5,000 epochs each, and `d7` contains the remaining 3,902 epochs.

## 9. `d1` ↔ `y_stim` Relationship

- **Evidence**:
  - `d1` contains 5,000 epochs.
  - `y_stim` contains 33,902 metadata columns.
  - Sequential file naming (`d1` to `d7`) and the exact chunk sizes.
- **Interpretation**: It is **STRONGLY SUPPORTED**, BUT NOT DIRECTLY DOCUMENTED, that `d1` corresponds exactly to `y_stim[:, 0:5000]`.
- **Limitation**: There is no explicit OSF metadata statement directly proving this mapping. This limitation must be preserved in future research stages.

## 10. Subjects Represented in `d1`

- **Finding**: Under the **STRONGLY SUPPORTED** sequential-chunk interpretation (that `d1` corresponds to the first 5,000 `y_stim` columns):
  - The first 5,000 trials fall entirely within the Healthy Control (HC) block.
  - They correspond to Healthy Control subject indices 1 through 24.
  - The 5,000th trial does not cross into the ADD group.
- **Limitation**: This conclusion is dependent on the `d1` ↔ `y_stim[:, 0:5000]` mapping, which is strongly supported but not directly established.

## 11. Dimension Interpretation

- **Observed Dimensions for `d1`**: `(385, 56, 5000)`
- **Strongly Supported Interpretation**: `(time/sample positions × channels × epochs)`
- **Evidence**: The base paper reports a sampling frequency of 256 Hz and an epoch duration of 1.5 seconds.
- **Limitation**: The exact dimension semantics are not directly documented in the available OSF metadata. The paper's parameters serve as supporting evidence.

## 12. 385-Sample Issue

- **Paper Reports**: 256 Hz sampling rate, 1.5-second experiment duration.
- **Calculation**: 256 samples/second × 1.5 seconds = 384 samples.
- **Actual Data**: 385 samples observed in the first dimension of `d1`.
- **Conclusion**: The additional sample is **UNKNOWN / UNRESOLVED**. It could be an inclusive boundary, a zero-padding artifact, or something else, but its physical meaning remains unresolved.

## 13. What We Know

| Item | Finding | Evidence Level |
|---|---|---|
| Total Subjects | 144 | **PAPER-REPORTED** / **STRONGLY SUPPORTED** |
| Subject Groups | 44 HC, 52 ADD, 48 ADHD | **DIRECTLY ESTABLISHED** (counts) / **STRONGLY SUPPORTED** (labels) |
| Total Trials/Epochs | 33,902 | **DIRECTLY ESTABLISHED** |
| `d1.mat` Shape | `(385, 56, 5000)` | **DIRECTLY ESTABLISHED** |
| Actual EEG Channels | 56 | **DIRECTLY ESTABLISHED** |
| `chan.mat` Channels | 60 | **DIRECTLY ESTABLISHED** |
| Metadata Structure | `y_stim` holds trial info; `Sub_name_stim` holds subject info | **DIRECTLY ESTABLISHED** |
| Seven-File Org. | Dataset split into `d1`-`d7` | **DIRECTLY ESTABLISHED** |
| `d1`-`d6` Size | ~5000 epochs each | **STRONGLY SUPPORTED** |
| Stimulus Categories | 3 mutually exclusive categories | **DIRECTLY ESTABLISHED** (structure) |
| Subject Indexing | Row 0 of `y_stim` represents subject index within group | **DIRECTLY ESTABLISHED** |

## 14. What We Do Not Know

1. **Exact physical meaning of the three stimulus categories** in `y_stim` Rows 1-3.
2. **Exact mapping of `d1`** to the first 5,000 columns of `y_stim`.
3. **Exact meaning of the 385th sample** in the time dimension.
4. **Exact four channels missing** between `chan.mat` (60) and `d1.mat` (56).
5. **Mapping of specific physical labels** (Target, Non-Target, etc.) to the 0/1 indicators.

## 15. Dataset Decision

The evidence gathered is sufficient to understand the general physical layout and logical structure of the dataset.

**Decision**: The dataset structure is sufficiently understood to document and freeze the current dataset definition, while strictly preserving the unresolved limitations detailed above.

**Action**: We will NOT begin preprocessing at this stage.

## 16. Dataset Definition

**Current Dataset Definition:**

- **Source**: OSF dataset associated with the base paper.
- **Participants**: 144 (COMPLETE dataset)
- **Groups**: 44 HC / 52 ADD / 48 ADHD
- **Total trials**: 33,902
- **EEG data**: Split across `d1.mat`–`d7.mat`
- **Channels in actual EEG array**: 56
- **Metadata**: `y_stim.mat` + participant information (`Sub_name_stim.mat`) + channel information (`chan.mat`)
- **Sampling frequency**: 256 Hz (**PAPER-REPORTED**)
- **Epoch duration**: 1.5 s (**PAPER-REPORTED**)
- **Within-epoch array dimension**: 385 (Observed; exact interpretation **UNRESOLVED**)

## 17. Research Decisions

**Decision 1: File chunking interpretation**
- **Question**: How are the 33,902 trials distributed across the 7 `.mat` files?
- **Evidence**: `d1` contains 5,000 epochs. File sizes for `d1`-`d6` are identical (~402 MB). `d7` is smaller (~314 MB).
- **Interpretation**: `d1`-`d6` each hold 5,000 epochs (30,000 total). `d7` holds the remaining 3,902 epochs.
- **Decision**: Treat `d1.mat` as containing the first 5,000 epochs of the dataset for structural understanding.
- **Remaining uncertainty**: The exact mapping to `y_stim` columns 0:5000 is STRONGLY SUPPORTED but not directly proven by explicit metadata.

**Decision 2: Handling the 56 vs 60 channel discrepancy**
- **Question**: Why does `d1` have 56 channels while `chan.mat` lists 60?
- **Evidence**: Direct observation of `d1` shape (56) and `chan.mat` array size (60).
- **Interpretation**: Four channels were dropped or excluded from the raw arrays.
- **Decision**: Document the raw data as having 56 channels and do not attempt to guess or map the missing 4 channels to EOG/mastoid without further evidence.
- **Remaining uncertainty**: The exact identities of the 4 missing channels.

**Decision 3: Handling the 385-sample dimension**
- **Question**: What does the 385 dimension represent, given 256 Hz * 1.5s = 384?
- **Evidence**: Direct observation of `d1` shape. Base paper parameters.
- **Interpretation**: It represents the time dimension, but contains an extra unexplained sample.
- **Decision**: Record the dimension as 385 and do not discard or label the extra sample.
- **Remaining uncertainty**: The physical origin or mathematical necessity of the 385th sample.

**Decision 4: Group categorization mapping**
- **Question**: Which blocks in `Sub_name_stim` correspond to which clinical groups?
- **Evidence**: 44 `controls`, 52 `subtype1`, 48 `subtype2` paths. Paper reports 44 HC, 52 ADD, 48 ADHD.
- **Interpretation**: Controls = HC, Subtype1 = ADD, Subtype2 = ADHD.
- **Decision**: Use this mapping for understanding the high-level grouping structure.
- **Remaining uncertainty**: Minor, as the counts align perfectly, but relies on path string inferences.

**Decision 5: Readiness for next stage**
- **Question**: Is the dataset understood enough to proceed?
- **Evidence**: We have mapped the sizes, dimensions, block structures, and primary metadata links.
- **Interpretation**: We have reached the limit of structural investigation without making unsupported domain assumptions.
- **Decision**: Freeze the dataset definition and proceed to domain/terminology auditing next (but not preprocessing).
- **Remaining uncertainty**: Documented in "What We Do Not Know" section.
