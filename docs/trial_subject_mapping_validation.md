# Trial → Subject → Clinical Group Mapping Validation Report
## Sprint 1 — Research Gate 1 Audit

> [!WARNING]
> **Historical report; its original Gate 1 verdict is superseded.** The structural
> observations below (counts, runs, filename prefixes, and correlations) do not
> independently verify trial-to-column correspondence, participant identity, or
> clinical diagnosis. `(block, row-0 value)` is only a candidate composite key;
> `sub_name_stim` ordinals are candidate filename mappings; and `y_stim` rows 1–3
> are observed one-hot patterns with unverified semantics, not clinical targets.
> This report does not establish participant-independent evaluation. No real-data
> split or model training is reported here.

**Branch:** `ml/transformer-development`  
**Date:** 2026-10-01  
**Status:** COMPLETE — SPRINT 1 VALIDATION AUDIT  
**Script Executed:** `scripts/validate_trial_subject_mapping.py`  

---

## 1. Objective

The historical objective of Sprint 1 was to determine whether every trial could be mapped to:
1. A specific source raw file (`d1.mat` through `d7.mat`) and within-file trial index.
2. A corresponding column index in the label matrix (`y_stim.mat`).
3. A candidate participant mapping derived from block position and row-0 value (not a verified identity or split key).
4. A candidate source filename in `sub_name_stim.mat`.
5. A candidate diagnostic-group mapping; verified clinical diagnosis labels are not available in these metadata.

Prior documentation labeled several of these critical links as `STRONGLY SUPPORTED` or `UNRESOLVED — EVIDENCE REQUIRED` due to the lack of explicit trial IDs embedded directly inside the EEG arrays. The historical investigation performed read-only structural checks, boundary signal comparisons, and count-consistency checks. These diagnostics do not independently establish the proposed correspondences.

---

## 2. Existing Evidence Prior to Sprint 1

Before this audit, the repository had established the following foundational facts:
- **Raw Acquisition:** All 10 files (`d1.mat`–`d7.mat`, `y_stim.mat`, `sub_name_stim.mat`, `chan.mat`) are locally present with byte sizes matching the official OSF storage API (`node 6594x`).
- **Trial Count Agreement:** `d1`–`d6` each store 5,000 trials and `d7` stores 3,902 trials ($6 \times 5,000 + 3,902 = 33,902$). This sum matches the column count of `y_stim.mat` ($33,902$) and the total trial count reported in He et al. (2023).
- **Group Counts:** `sub_name_stim.mat` contains three cell arrays of sizes 44, 52, and 48, matching the paper's cohort of 144 children (44 HC, 52 ADD, 48 ADHD).
- **Row 0 Resets:** `y_stim` row 0 exhibits contiguous integer runs resetting to 1 across three distinct blocks: 1–44, 1–52, and 1–48.
- **Open Questions:** Prior audits noted that count matching alone does not prove sequential ordering, that EEG files contain no embedded trial IDs, and that the link between `sub_name_stim` filenames and row-0 integers was inferential.

---

## 3. Validation Methodology

To test these relationships without modifying raw data or introducing methodological bias, the following read-only validation protocol was implemented in `scripts/validate_trial_subject_mapping.py`:

1. **Binary Header & Timestamp Inspection:** Extract the 128-byte MATLAB v7.3 userblock header from each file to inspect machine architecture and file creation timestamps down to the second.
2. **Structural Array Verification:** Inspect shapes, dtypes, chunk dimensions, compression, and finite-value integrity (scans for `NaN` and `Inf`) across all 7 EEG files and all metadata files.
3. **Run-Length & Exhaustive Coverage Audit:** Parse all 33,902 columns of `y_stim` row 0 to detect run lengths, boundary transitions, missing numbers, duplicates, and impossible indices.
4. **Rows 1–3 Structure:** Check binary/one-hot patterns and their alignment with configured ranges; this does not establish their semantics or make them diagnosis labels.
5. **Candidate-Selected Boundary Signal Comparison:** Compare signals around boundaries selected by the proposed alignment. Correlation is a diagnostic only, not independent evidence of identity or trial correspondence.
6. **Filename Ordinal & Lexicographic Sort Check:** Verify the string tokens, uniqueness, and ordering convention of `sub_name_stim.mat` against standard MATLAB I/O behavior.
7. **Paper Discrepancy & Counter-Evidence Search:** Actively test for non-monotonicities, non-finite values, signal disruptions, or contradictions.

---

## 4. EEG File Validation (`d1.mat` – `d7.mat`)

All 7 raw EEG files were inspected. Every file conforms exactly to the specifications below:

| File | File Size (Bytes) | MATLAB Header Creation Timestamp | Dataset Name | Shape `(time, chan, trials)` | Data Type | Chunk Shape | Compression | Finite Values (NaN / Inf) | Value Range ($\mu\text{V}$) |
|---|---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `d1.mat` | 402,642,207 | `Fri Mar 15 17:59:45 2019` | `d1` | `(385, 56, 5000)` | `float32` | `(192, 56, 1)` | gzip | 0 NaN / 0 Inf | `[-313.2, 286.0]` |
| `d2.mat` | 402,606,807 | `Fri Mar 15 18:00:06 2019` | `d2` | `(385, 56, 5000)` | `float32` | `(192, 56, 1)` | gzip | 0 NaN / 0 Inf | `[-482.8, 357.5]` |
| `d3.mat` | 402,837,633 | `Fri Mar 15 18:00:25 2019` | `d3` | `(385, 56, 5000)` | `float32` | `(192, 56, 1)` | gzip | 0 NaN / 0 Inf | `[-288.5, 505.2]` |
| `d4.mat` | 402,887,010 | `Fri Mar 15 18:00:45 2019` | `d4` | `(385, 56, 5000)` | `float32` | `(192, 56, 1)` | gzip | 0 NaN / 0 Inf | `[-466.1, 464.2]` |
| `d5.mat` | 402,691,834 | `Fri Mar 15 18:01:05 2019` | `d5` | `(385, 56, 5000)` | `float32` | `(192, 56, 1)` | gzip | 0 NaN / 0 Inf | `[-428.4, 417.0]` |
| `d6.mat` | 402,722,744 | `Fri Mar 15 18:01:25 2019` | `d6` | `(385, 56, 5000)` | `float32` | `(192, 56, 1)` | gzip | 0 NaN / 0 Inf | `[-408.9, 329.9]` |
| `d7.mat` | 314,191,152 | `Fri Mar 15 18:01:41 2019` | `d7` | `(385, 56, 3902)` | `float32` | `(192, 56, 1)` | gzip | 0 NaN / 0 Inf | `[-444.5, 312.5]` |

### Key Findings:
- **Exact Shape Consistency:** Dimensions across all 7 files are strictly identical on the time axis (385) and channel axis (56).
- **Exact Trial Total:** $6 \times 5,000 + 3,902 = 33,902$ trials.
- **Strict Chronological Sequence:** Timestamps in the file headers demonstrate that `d1` through `d7` were created in strictly ascending chronological order within a single 2-minute window on March 15, 2019 (`17:59:45` to `18:01:41`), spaced by approximately 16–21 seconds per file.
- **Full Finite Integrity:** An exhaustive scan across all 33,902 trials confirmed exactly 0 `NaN` and 0 `Inf` values throughout the entire 2.73 GB raw EEG dataset.

---

## 5. `y_stim.mat` Validation

- **Array Shape:** `(4, 33902)`
- **Data Type:** `float64`
- **Creation Timestamp:** `Fri Feb 22 14:52:45 2019`
- **Integrity:** 0 `NaN`, 0 `Inf`.

### 5.1 Row-0 Candidate Value Runs
An analysis of contiguous value runs in `y_stim[0, :]` revealed:
- Exactly 144 contiguous runs spanning all 33,902 columns without gaps or interleaved indices.
- Resets to 1 occur at run indices 0, 44, and 96.
- **Block 1:** 44 candidate row-0 runs (values 1 to 44), spanning columns `[0, 10129)` (10,129 columns).
- **Block 2:** 52 candidate row-0 runs (values 1 to 52), spanning columns `[10129, 23160)` (13,031 columns).
- **Block 3:** 48 candidate row-0 runs (values 1 to 48), spanning columns `[23160, 33902)` (10,742 columns).
- **Structural check:** No row-0 run value is missing, duplicated out of sequence, negative, or greater than 52.

### 5.2 Structural Pattern in Rows 1–3 (Semantics Unresolved)
The historical inspection reported:
- For columns in the three configured ranges, rows 1–3 have the respective one-hot patterns `[1,0,0]`, `[0,1,0]`, and `[0,0,1]`.
- Across the inspected matrix, each column has exactly one active binary value.

These are structural observations only. They do not prove that the values encode stimulus categories, clinical groups, or any other semantic class. Do not use these rows or the configured block positions as clinical targets.

---

## 6. `sub_name_stim.mat` Validation

- **Array Structure:** HDF5 cell array of shape `(1, 3)`
- **Creation Timestamp:** `Fri Feb 22 14:52:45 2019` (matches `y_stim.mat` down to the exact second).
- **Total Files:** 144 filename strings.
- **Uniqueness:** All 144 filenames are globally unique across all blocks.

| Block Index | Prefix | Expected Count | Observed Count | Unique Strings | Lexicographically Sorted? | First Filename | Last Filename |
|:---:|:---:|:---:|:---:|:---:|:---:|---|---|
| Block 0 | `controls` | 44 | 44 | 44 | **True** | `controls2MSCT_Time_Estimation_NF_VP_02C.mat` | `controlsTW272_gesWK_pr_Time.mat` |
| Block 1 | `subtype1` | 52 | 52 | 52 | **True** | `subtype1AB237_NF_pr.mat` | `subtype1VR258_NF_pr.mat` |
| Block 2 | `subtype2` | 48 | 48 | 48 | **True** | `subtype2AE231_ET_pr.mat` | `subtype2TH034_WK_pr.mat` |

### Key Observations:
1. **Sorted Ordering:** Every block's filenames are stored in strict alphabetical order (`names == sorted(names)`). In MATLAB, when directory contents are queried using `files = dir('path/*.mat')`, the resulting struct array is sorted alphabetically.
2. **Clinical Token Semantics:** The filenames contain clinical trial tokens from the Dresden ADHD multicenter study:
   - `NF`: Neurofeedback training
   - `MPH`: Methylphenidate treatment (Ritalin)
   - `WK` / `gesWK`: Wartelistekontrolle (waitlist control) / gesunde Wartelistekontrolle (healthy control)
   - `ET`: Elterntraining (parent behavioral management training)
   - `PHY`: Physiotherapie (physical activity training)
   - `Time` / `Time_Estimation`: The 1.2-second interval time-estimation task
   - `pr` (`prä`): Pre-intervention baseline recording
   - `post`: Post-intervention follow-up recording

---

## 7. EEG-File → `y_stim`-Column Join Analysis

The candidate join posits that `d1.mat` through `d7.mat` correspond sequentially to column ranges:
- `d1`: columns `[0, 5000)`
- `d2`: columns `[5000, 10000)`
- `d3`: columns `[10000, 15000)`
- `d4`: columns `[15000, 20000)`
- `d5`: columns `[20000, 25000)`
- `d6`: columns `[25000, 30000)`
- `d7`: columns `[30000, 33902)`

### 7.1 Candidate-Selected Boundary Comparison
Under the proposed alignment, row-0 runs cross file boundaries. This creates six candidate-dependent comparison cases:

```
File d1 (trials 0..4999)      File d2 (trials 5000..9999)
[... run 23 ][ run 24: 38 tr ]|[ run 24: 61 tr ][ run 25 ...]
              ▲                               ▲
              └──── candidate-selected slices ────┘
```

The slices' participant relationship is not independently known. Similarity across a boundary cannot establish that they belong to the same person or verify file order.

No conclusion about participant identity or alternative file ordering follows from these correlations.

### 7.2 Historical Boundary Correlation Results

The historical report selected two slices using the candidate alignment and calculated the mean spatial variance profile across channels. The comparison slice's participant identity is also unverified.

We calculated the mean spatial variance profile across the 56 channels for each slice, and evaluated the Pearson correlation coefficient ($r$):

| Boundary | Candidate-selected run label | Candidate Global Column Range | Trials in $d_k$ | Trials in $d_{k+1}$ | Cross-file correlation ($r$) | Comparison-slice correlation ($r_{\text{ctrl}}$) | Historical diagnostic |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`d1` / `d2`** | candidate block 1 run 24 | `[4962, 5061)` | 38 | 61 | **$0.9152$** | $0.3234$ | reported |
| **`d2` / `d3`** | candidate block 1 run 44 | `[9851, 10129)` | 149 | 129 | **$0.9682$** | $0.2776$ | reported |
| **`d3` / `d4`** | candidate block 2 run 20 | `[14810, 15058)` | 190 | 58 | **$0.9897$** | $0.5656$ | reported |
| **`d4` / `d5`** | candidate block 2 run 40 | `[19885, 20179)` | 115 | 179 | **$0.9977$** | $0.4333$ | reported |
| **`d5` / `d6`** | candidate block 3 run 9 | `[24944, 25205)` | 56 | 205 | **$0.9084$** | $0.3543$ | reported |
| **`d6` / `d7`** | candidate block 3 run 30 | `[29860, 30150)` | 140 | 150 | **$0.9105$** | $0.0467$ | reported |

Across all 6 file transitions:
- The candidate-selected correlations exceed the comparison-slice correlations in the reported cases.
- This is consistent with the proposed alignment but does not independently prove source order, trial correspondence, or participant identity.

---

## 8. Candidate Row-0 Index → Filename Hypothesis

In `y_stim.mat`, row 0 has candidate values $1 \dots 44$, $1 \dots 52$, and $1 \dots 48$ across the configured ranges.
In `sub_name_stim.mat`, the blocks store $44$, $52$, and $48$ filenames in alphabetical order.

### Observations Consistent with a Candidate Ordinal Mapping (`filename[k]` ↔ `row0 value k`):
1. **Identical Creation Timestamp:** Both files were written on `Fri Feb 22 14:52:45 2019`.
2. **Matching Block Lengths:** Both files have blocks of lengths 44, 52, and 48.
3. **MATLAB `dir()` Behavior:** When MATLAB loops over files using `for i = 1:length(files)`, `files(i).name` corresponds directly to loop index `i`.

### Critical Methodological Distinction
Because the individual participant EEG files are unavailable for independent cross-checking, the filename/index correspondence is unresolved. The key `(candidate_block, row0_value)` is deterministic and unique by construction, but does not prove that the grouped trials belong to one verified person. Therefore this evidence does not establish participant-independent evaluation or split safety.

---

## 9. Candidate Diagnostic-Group Mapping (Unverified)

The earlier report mapped filename-prefix/count blocks to the paper's reported cohorts. This remains a candidate mapping, not verified per-participant clinical diagnosis:

| Block | Prefix in Filenames | Subject Count | Total Trials | Base Paper Group (Section 2.1) | Base Paper Subjects | Base Paper Trials | Alignment |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Block 1** | `controls` | 44 | 10,129 | Healthy Controls (HC) | 44 | 10,129 | **Exact Match** |
| **Block 2** | `subtype1` | 52 | 13,031 | ADD (ADHD Inattentive) | 52 | 13,031 | **Exact Match** |
| **Block 3** | `subtype2` | 48 | 10,742 | ADHD (ADHD Combined) | 48 | 10,742 | **Exact Match** |
| **Total** | — | **144** | **33,902** | Total Cohort | **144** | **33,902** | **Exact Match** |

The counts match Section 2.1 of He et al. (2023). This alignment and filename terminology do not independently establish clinical diagnosis labels or verify the trial-to-filename join.

---

## 10. Candidate Row-0 Run Lengths

The historical report listed these run lengths by candidate row-0 value. They are not verified per-participant trial counts:

### Candidate Block 1 (filename prefix `controls`): Total = 10,129 columns
`281, 262, 295, 229, 255, 135, 232, 254, 159, 272, 279, 168, 137, 237, 205, 115, 195, 229, 222, 209, 160, 229, 203, 99, 229, 241, 79, 287, 102, 279, 278, 276, 280, 241, 288, 273, 267, 259, 273, 279, 292, 290, 277, 278`  
- **Min:** 79 (Subject 27)  
- **Max:** 295 (Subject 3)  
- **Mean ± SD:** $230.2 \pm 56.6$ trials

### Candidate Block 2 (filename prefix `subtype1`): Total = 13,031 columns
`266, 249, 220, 276, 265, 224, 269, 276, 252, 272, 264, 263, 255, 231, 264, 206, 199, 239, 191, 248, 255, 172, 262, 269, 226, 199, 267, 268, 267, 282, 270, 275, 219, 278, 290, 245, 281, 249, 253, 294, 289, 290, 283, 263, 197, 251, 257, 258, 171, 259, 184, 279`  
- **Min:** 171 (Subject 49)  
- **Max:** 294 (Subject 40)  
- **Mean ± SD:** $250.6 \pm 31.0$ trials

### Candidate Block 3 (filename prefix `subtype2`): Total = 10,742 columns
`257, 279, 46, 259, 205, 262, 214, 262, 261, 265, 285, 255, 213, 238, 168, 125, 239, 195, 286, 197, 234, 203, 223, 265, 255, 226, 264, 259, 260, 290, 287, 229, 172, 219, 217, 129, 187, 51, 269, 234, 124, 204, 262, 254, 142, 270, 219, 283`  
- **Min:** 46 (Subject 3)  
- **Max:** 290 (Subject 30)  
- **Mean ± SD:** $223.8 \pm 57.3$ trials

### Key Limitation for Future Evaluation
Reported candidate run lengths vary from 46 to 295. Their interpretation as per-participant trial counts depends on the unverified mapping. No cross-validation grouping or participant-independent evaluation is established by these diagnostics.

---

## 11. Group-Level Consistency Checks

1. **Trial Summation:** $10,129 + 13,031 + 10,742 = 33,902$ exactly.
2. **Candidate run / filename-entry counts:** $44 + 52 + 48 = 144$ exactly.
3. **Array Dimension Consistency:** Every file $d1 \dots d7$ provides trials with exactly 385 sample points and 56 channels.
4. **Finite Values:** The historical report records finite-value checks; these checks do not address participant/label correspondence.

---

## 12. Independent Evidence Discovered During Audit

1. **MATLAB File Header Timestamps:**
   - The report observed matching timestamps for `sub_name_stim.mat` and `y_stim.mat`; this does not prove a keyed relationship between their contents.
   - The report observed timestamps across `d1.mat` through `d7.mat`; timestamp sequence alone does not establish trial-to-column correspondence.
2. **OSF API Readme Statement:**
   - Formally confirms that the single sample array was split into 7 chunks solely to accommodate OSF's 5 GB individual file size limitation.
3. **Candidate-Selected Boundary Signal Diagnostics:**
   - The historical report records cross-file correlations of $r = 0.9084$ to $0.9977$ and comparison-slice correlations of $r = 0.0467$ to $0.5656$. The slices were selected using the proposed alignment, so this is not independent confirmation of continuity or identity.
4. **Rows 1–3 Structure:**
   - The report records binary one-hot patterns aligned with configured ranges. Semantic meaning remains unresolved.

---

## 13. Counter-Evidence Searched For

During the audit, we explicitly searched for the following potential flaws:
- **Index Gaps or Duplications in `y_stim`:** Found 0 gaps, 0 duplicates.
- **Out-of-order row-0 values:** The report found no gaps in the configured ranges; this does not establish participant ordering.
- **Corrupted or Non-Finite Samples:** Exhaustive scan of all 33,902 trials found 0 `NaN` and 0 `Inf`.
- **Signal Discontinuity Across File Boundaries:** Tested all 6 boundaries; none exhibited signal discontinuity or uncharacteristic correlation drops.
- **One-hot structure:** The report found no violations; this does not establish stimulus or diagnosis semantics.

---

## 14. Remaining Limitations

The following items cannot be resolved from the available raw dataset files alone and must remain formal limitations:
1. **Individual filename ordinal linkage:** Block sizes, timestamps, and alphabetical sorting are consistent with an ordinal mapping, but no per-file trial-count/identity cross-check is available. Participant-independent CV safety is not established.
2. **Channel Mapping (56 vs. 60):** `chan.mat` contains 60 named channels, but `d1`–`d7` contain 56 data columns. The exact identities of the 4 excluded channels remain unknown. Spatial topography operations must not assume specific 10–20 coordinates until this mapping is resolved.
3. **Physical Meaning of Sample 385:** At 256 Hz, a 1.5-second trial equals 384 sampling intervals. The 385th sample represents an inclusive endpoint or sampling convention that remains unexplained. Per the protocol, all 385 points must be preserved without cropping.

---

## 15. Final Mapping Schema

The historical report proposed the following arithmetic indexing. It does not independently establish that the EEG trial sequence corresponds to this metadata order:

$$\text{Global Trial Index } t \in [0, 33901]$$

1. **Source File & Within-File Index:**
   - If $0 \le t < 5000$: File `d1`, index $t_{\text{file}} = t$
   - If $5000 \le t < 10000$: File `d2`, index $t_{\text{file}} = t - 5000$
   - If $10000 \le t < 15000$: File `d3`, index $t_{\text{file}} = t - 10000$
   - If $15000 \le t < 20000$: File `d4`, index $t_{\text{file}} = t - 15000$
   - If $20000 \le t < 25000$: File `d5`, index $t_{\text{file}} = t - 20000$
   - If $25000 \le t < 30000$: File `d6`, index $t_{\text{file}} = t - 25000$
   - If $30000 \le t < 33902$: File `d7`, index $t_{\text{file}} = t - 30000$
   - This is a candidate index convention based on file counts, not a verified EEG-to-metadata join.

2. **Candidate Diagnostic-Group Mapping (not verified labels):**
   - These ranges align with candidate filename blocks and the counts reported in the paper. Do not treat range position as an established clinical diagnosis.

3. **Candidate Composite Participant Key (not verified identity or split key):**
   - Constructed from candidate block position and row-0 value.
   - Yields 144 unique keys by construction; uniqueness does not verify the participant mapping.
   - Grouping on these keys alone does not establish participant-independent evaluation.

4. **Candidate Filename:**
   - Mapped ordinally to `sub_name_stim{block_index}{within_block_subject_index - 1}`.

---

## 16. Historical Gate 1 Verdict — Superseded

The following table and verdict reproduce the historical report's conclusions for auditability. Its `VALIDATED` statuses for participant identity, clinical diagnosis, and split safety are withdrawn; the current evidence classification is the corrected status below.

### Current Relationship-by-Relationship Evidence Classification:

| Relationship / Component | Evidence Classification | Justification |
|---|:---:|---|
| **EEG File Shapes & Dtypes** | `STRUCTURALLY OBSERVED` | Reported from HDF5 inspection; does not establish metadata correspondence. |
| **Sequential File Order and Trial-to-Column Join** | `UNRESOLVED` | Timestamp and candidate-selected correlations are not an independent trial key. |
| **`y_stim` Row-0 Runs** | `STRUCTURALLY OBSERVED` | 144 contiguous value runs; they are candidate mappings, not verified participants. |
| **Clinical Diagnostic Group Assignment** | `UNVERIFIED CANDIDATE MAPPING` | Counts and filename tokens align with paper-reported cohorts; no diagnosis field is keyed to verified participants. |
| **Composite Candidate Key for Cross-Validation** | `UNIQUE BY CONSTRUCTION; IDENTITY UNVERIFIED` | Does not establish participant-independent evaluation or split safety. |
| **`sub_name_stim` Filename Ordinal Mapping** | `UNRESOLVED` | No independent per-file trial-count/identity cross-check. |

---

### Original Formal Gate 1 Declaration (Historical; Withdrawn):

```
======================================================================
GATE 1 VERDICT:
UNRESOLVED
======================================================================
```

**Corrected status:** The earlier `VALIDATED` verdict is superseded. Source-file order, the trial-to-metadata join, candidate participant identity, filename correspondence, and clinical diagnosis remain unresolved. No participant-independent split safety is established by this report.
