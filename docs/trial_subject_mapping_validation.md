# Trial → Subject → Clinical Group Mapping Validation Report
## Sprint 1 — Research Gate 1 Audit
**Branch:** `ml/transformer-development`  
**Date:** 2026-10-01  
**Status:** COMPLETE — SPRINT 1 VALIDATION AUDIT  
**Script Executed:** `scripts/validate_trial_subject_mapping.py`  

---

## 1. Objective

The primary objective of Sprint 1 is to determine whether every single trial among the 33,902 raw EEG epochs can be defensibly, deterministically, and reproducibly mapped to:
1. A specific source raw file (`d1.mat` through `d7.mat`) and within-file trial index.
2. A corresponding column index in the label matrix (`y_stim.mat`).
3. A globally unique participant identifier (subject ID) to ensure strictly leak-free subject-level evaluation.
4. A candidate subject source filename in `sub_name_stim.mat`.
5. A verified clinical diagnostic group (`HC`, `ADD`, `ADHD`).

Prior documentation labeled several of these critical links as `STRONGLY SUPPORTED` or `UNRESOLVED — EVIDENCE REQUIRED` due to the lack of explicit trial IDs embedded directly inside the EEG arrays. This investigation subjected these relationships to direct, read-only empirical tests, chronological metadata audits, boundary signal correlations, and consistency checks across all 33,902 trials.

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
4. **Isomorphism of Rows 1–3:** Formally test whether rows 1–3 represent within-subject stimulus conditions or block-level one-hot diagnostic indicators.
5. **Cross-Boundary Empirical Signal Correlation:** At every file boundary ($d1/d2, d2/d3, d3/d4, d4/d5, d5/d6, d6/d7$), identify the subject whose recording straddles the boundary according to the candidate sequential join. Compute the cross-channel variance profile (anatomical EEG fingerprint) before and after the file split, and measure the Pearson correlation coefficient ($r$) against within-file control comparisons.
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

### 5.1 Row 0 Subject Index Runs
An analysis of contiguous value runs in `y_stim[0, :]` revealed:
- Exactly 144 contiguous runs spanning all 33,902 columns without gaps or interleaved indices.
- Resets to 1 occur at run indices 0, 44, and 96.
- **Block 1:** 44 subjects (indices 1 to 44 strictly ascending), spanning columns `[0, 10129)` (total 10,129 trials).
- **Block 2:** 52 subjects (indices 1 to 52 strictly ascending), spanning columns `[10129, 23160)` (total 13,031 trials).
- **Block 3:** 48 subjects (indices 1 to 48 strictly ascending), spanning columns `[23160, 33902)` (total 10,742 trials).
- **Anomaly Check:** No subject index is missing, duplicated out of sequence, negative, or greater than 52.

### 5.2 Resolution of Rows 1–3: Block/Class Identity vs. Stimulus Categories
Prior audit documents hypothesized that rows 1–3 might represent within-subject stimulus task conditions (e.g., target, non-target, distractor) and cautioned against using them as diagnostic labels.

Our direct validation discovered that:
- For all 10,129 columns of Block 1 (Controls / HC), `y_stim[1:4, :]` is strictly $\begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$.
- For all 13,031 columns of Block 2 (Subtype 1 / ADD), `y_stim[1:4, :]` is strictly $\begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$.
- For all 10,742 columns of Block 3 (Subtype 2 / ADHD), `y_stim[1:4, :]` is strictly $\begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$.
- Rows 1–3 **never change within a subject** and **never change within a block**.
- Across all 33,902 trials, column sums of rows 1–3 are identically 1.0.

> [!IMPORTANT]
> **Resolution on Rows 1–3:** Rows 1–3 are mathematically isomorphic to the three block groups. However, to maintain strict compliance with the ML Execution Plan data contract (Section 5.3) and prevent reliance on ambiguous naming, the clinical target label should be derived directly from the block identity:
> - Block 0 (`controls`) $\rightarrow$ `HC` (label 0)
> - Block 1 (`subtype1`) $\rightarrow$ `ADD` (label 1)
> - Block 2 (`subtype2`) $\rightarrow$ `ADHD` (label 2)

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

### 7.1 The Straddling Subject Test
Because 5,000 does not divide the cumulative subject trial counts evenly, exactly one participant straddles every boundary between consecutive files. This creates 6 natural empirical test cases:

```
File d1 (trials 0..4999)      File d2 (trials 5000..9999)
[... Sub 23 ][ Sub 24: 38 tr ]|[ Sub 24: 61 tr ][ Sub 25 ...]
              ▲                               ▲
              └──────── SAME PARTICIPANT ─────┘
```

If the candidate sequential file ordering is correct, the EEG trials immediately preceding the boundary and immediately following the boundary belong to the **same human participant**, and must share that individual's unique anatomical EEG signature (spatial variance across electrodes).

If the ordering were non-sequential, random, or transposed, the trials on either side would belong to different children, exhibiting low spatial correlation.

### 7.2 Empirical Boundary Correlation Results

For each boundary, we extracted:
- Slice A: the trials of the straddling subject in file $d_k$
- Slice B: the trials of the straddling subject in file $d_{k+1}$
- Control Slice: trials of an unrelated subject from file $d_k$

We calculated the mean spatial variance profile across the 56 channels for each slice, and evaluated the Pearson correlation coefficient ($r$):

| Boundary | Straddling Subject Identity | Candidate Global Column Range | Trials in $d_k$ | Trials in $d_{k+1}$ | Same-Subject Cross-File Correlation ($r_{\text{same}}$) | Different-Subject Control Correlation ($r_{\text{ctrl}}$) | Verdict |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`d1` / `d2`** | Block 1 (HC) Subject 24 | `[4962, 5061)` | 38 | 61 | **$0.9152$** | $0.3234$ | **CORROBORATED** |
| **`d2` / `d3`** | Block 1 (HC) Subject 44 | `[9851, 10129)` | 149 | 129 | **$0.9682$** | $0.2776$ | **CORROBORATED** |
| **`d3` / `d4`** | Block 2 (ADD) Subject 20 | `[14810, 15058)` | 190 | 58 | **$0.9897$** | $0.5656$ | **CORROBORATED** |
| **`d4` / `d5`** | Block 2 (ADD) Subject 40 | `[19885, 20179)` | 115 | 179 | **$0.9977$** | $0.4333$ | **CORROBORATED** |
| **`d5` / `d6`** | Block 3 (ADHD) Subject 9 | `[24944, 25205)` | 56 | 205 | **$0.9084$** | $0.3543$ | **CORROBORATED** |
| **`d6` / `d7`** | Block 3 (ADHD) Subject 30 | `[29860, 30150)` | 140 | 150 | **$0.9105$** | $0.0467$ | **CORROBORATED** |

Across all 6 file transitions:
- $r_{\text{same}}$ exceeds $0.90$ in every single transition, reaching as high as $0.9977$.
- In every case, $r_{\text{same}}$ dramatically exceeds the unrelated control baseline ($r = 0.0467$ to $0.5656$).
- This provides direct, signal-level empirical proof that `d1.mat` through `d7.mat` form a continuous, contiguous concatenation that matches the `y_stim` column layout.

---

## 8. Subject-Index → Filename Analysis

In `y_stim.mat`, row 0 indexes subjects $1 \dots 44$, $1 \dots 52$, and $1 \dots 48$.  
In `sub_name_stim.mat`, the blocks store $44$, $52$, and $48$ filenames in alphabetical order.

### Evidence Supporting 1-to-1 Ordinal Mapping (`filename[k]` $\leftrightarrow$ `index k`):
1. **Identical Creation Timestamp:** Both files were written on `Fri Feb 22 14:52:45 2019`.
2. **Matching Block Lengths:** Both files have blocks of lengths 44, 52, and 48.
3. **MATLAB `dir()` Behavior:** When MATLAB loops over files using `for i = 1:length(files)`, `files(i).name` corresponds directly to loop index `i`.

### Critical Methodological Distinction:
Because the raw individual subject files (e.g., `controls2MSCT_..._02C.mat`) are not provided on OSF, we cannot independently verify whether file #1 had exactly 281 trials.

**Does this ambiguity impact ML evaluation? NO.**  
For subject-level cross-validation (`GroupKFold`), the required grouping key is the participant identity. Whether subject `HC_01` was originally named `VP_02C` or `VP_03C` has zero impact on model training or evaluation safety, because:
- Every trial of `HC_01` is contiguous and uniquely indexed in `y_stim`.
- All 281 trials of `HC_01` are kept strictly within the same CV fold.
- No trials from `HC_01` can ever leak into another fold.

The grouping key `(group_block, within_block_index)` is 100% deterministic, mathematically sound, and fully leak-free.

---

## 9. Clinical-Group Analysis

The mapping of blocks to clinical diagnostic groups is established as follows:

| Block | Prefix in Filenames | Subject Count | Total Trials | Base Paper Group (Section 2.1) | Base Paper Subjects | Base Paper Trials | Alignment |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Block 1** | `controls` | 44 | 10,129 | Healthy Controls (HC) | 44 | 10,129 | **Exact Match** |
| **Block 2** | `subtype1` | 52 | 13,031 | ADD (ADHD Inattentive) | 52 | 13,031 | **Exact Match** |
| **Block 3** | `subtype2` | 48 | 10,742 | ADHD (ADHD Combined) | 48 | 10,742 | **Exact Match** |
| **Total** | — | **144** | **33,902** | Total Cohort | **144** | **33,902** | **Exact Match** |

Every participant count (44, 52, 48) and trial count (10,129, 13,031, 10,742) matches Section 2.1 of He et al. (2023) down to the single integer. Clinical DSM subtype terminology (`subtype1` = Inattentive/ADD; `subtype2` = Combined/ADHD) is fully corroborated by the literature lineage (Vahid et al., 2019; Beste et al.).

---

## 10. Per-Subject Trial Counts

The exact trial counts for all 144 subjects, in order of within-block subject index $1 \dots N$, are:

### Healthy Controls (`HC`, Block 1, Subjects 1–44): Total = 10,129 trials
`281, 262, 295, 229, 255, 135, 232, 254, 159, 272, 279, 168, 137, 237, 205, 115, 195, 229, 222, 209, 160, 229, 203, 99, 229, 241, 79, 287, 102, 279, 278, 276, 280, 241, 288, 273, 267, 259, 273, 279, 292, 290, 277, 278`  
- **Min:** 79 (Subject 27)  
- **Max:** 295 (Subject 3)  
- **Mean ± SD:** $230.2 \pm 56.6$ trials

### Attention Deficit Disorder (`ADD` / Subtype 1, Block 2, Subjects 1–52): Total = 13,031 trials
`266, 249, 220, 276, 265, 224, 269, 276, 252, 272, 264, 263, 255, 231, 264, 206, 199, 239, 191, 248, 255, 172, 262, 269, 226, 199, 267, 268, 267, 282, 270, 275, 219, 278, 290, 245, 281, 249, 253, 294, 289, 290, 283, 263, 197, 251, 257, 258, 171, 259, 184, 279`  
- **Min:** 171 (Subject 49)  
- **Max:** 294 (Subject 40)  
- **Mean ± SD:** $250.6 \pm 31.0$ trials

### ADHD Combined Type (`ADHD` / Subtype 2, Block 3, Subjects 1–48): Total = 10,742 trials
`257, 279, 46, 259, 205, 262, 214, 262, 261, 265, 285, 255, 213, 238, 168, 125, 239, 195, 286, 197, 234, 203, 223, 265, 255, 226, 264, 259, 260, 290, 287, 229, 172, 219, 217, 129, 187, 51, 269, 234, 124, 204, 262, 254, 142, 270, 219, 283`  
- **Min:** 46 (Subject 3)  
- **Max:** 290 (Subject 30)  
- **Mean ± SD:** $223.8 \pm 57.3$ trials

### Key Implication for Cross-Validation:
Trial counts vary from 46 to 295 across subjects. When designing cross-validation in Sprint 4, evaluation must be grouped by subject, and metrics must be calculated per-subject or class-balanced so that high-trial participants do not dominate loss and accuracy metrics.

---

## 11. Group-Level Consistency Checks

1. **Trial Summation:** $10,129 + 13,031 + 10,742 = 33,902$ exactly.
2. **Subject Summation:** $44 + 52 + 48 = 144$ exactly.
3. **Array Dimension Consistency:** Every file $d1 \dots d7$ provides trials with exactly 385 sample points and 56 channels.
4. **Finite Values:** All 33,902 trials across all 56 channels and 385 timepoints contain valid finite real numbers with zero missing data.

---

## 12. Independent Evidence Discovered During Audit

1. **MATLAB File Header Timestamps:**
   - Proves that `sub_name_stim.mat` and `y_stim.mat` were compiled simultaneously in the same MATLAB workspace at `2019-02-22 14:52:45`.
   - Proves that `d1.mat` through `d7.mat` were split sequentially in order between `17:59:45` and `18:01:41` on `2019-03-15`.
2. **OSF API Readme Statement:**
   - Formally confirms that the single sample array was split into 7 chunks solely to accommodate OSF's 5 GB individual file size limitation.
3. **Cross-Boundary Empirical Signal Fingerprinting:**
   - Straddling subjects at all 6 file boundaries demonstrate cross-boundary spatial variance correlations of $r = 0.9084$ to $0.9977$, contrasted with control correlations of $r = 0.0467$ to $0.5656$. This independently confirms the sequential physical continuity of `d1` to `d7`.
4. **Exhaustive Collinearity of Rows 1–3:**
   - Direct verification proved that rows 1–3 of `y_stim` are the block indicators, resolving the earlier misconception that they represented dynamic within-subject stimulus task conditions.

---

## 13. Counter-Evidence Searched For

During the audit, we explicitly searched for the following potential flaws:
- **Index Gaps or Duplications in `y_stim`:** Found 0 gaps, 0 duplicates.
- **Out-of-Order or Non-Contiguous Subjects:** Found 0; all 144 subjects appear in strict contiguous runs.
- **Corrupted or Non-Finite Samples:** Exhaustive scan of all 33,902 trials found 0 `NaN` and 0 `Inf`.
- **Signal Discontinuity Across File Boundaries:** Tested all 6 boundaries; none exhibited signal discontinuity or uncharacteristic correlation drops.
- **Stimulus Indicator Contradictions:** Tested all 33,902 columns; none violated the one-hot block assignment.

---

## 14. Remaining Limitations

The following items cannot be resolved from the available raw dataset files alone and must remain formal limitations:
1. **Individual Filename Ordinal Linkage:** While matching block sizes (44/52/48), identical timestamps, and MATLAB `dir()` alphabetical sorting strongly support ordinal pairing ($1 \leftrightarrow 1$), the absence of raw individual participant `.mat` files means this link cannot be independently verified by checking individual trial counts. (Note: This does not affect subject-level CV safety).
2. **Channel Mapping (56 vs. 60):** `chan.mat` contains 60 named channels, but `d1`–`d7` contain 56 data columns. The exact identities of the 4 excluded channels remain unknown. Spatial topography operations must not assume specific 10–20 coordinates until this mapping is resolved.
3. **Physical Meaning of Sample 385:** At 256 Hz, a 1.5-second trial equals 384 sampling intervals. The 385th sample represents an inclusive endpoint or sampling convention that remains unexplained. Per the protocol, all 385 points must be preserved without cropping.

---

## 15. Final Mapping Schema

For all 33,902 trials, the deterministic data crosswalk is formally defined as:

$$\text{Global Trial Index } t \in [0, 33901]$$

1. **Source File & Within-File Index:**
   - If $0 \le t < 5000$: File `d1`, index $t_{\text{file}} = t$
   - If $5000 \le t < 10000$: File `d2`, index $t_{\text{file}} = t - 5000$
   - If $10000 \le t < 15000$: File `d3`, index $t_{\text{file}} = t - 10000$
   - If $15000 \le t < 20000$: File `d4`, index $t_{\text{file}} = t - 15000$
   - If $20000 \le t < 25000$: File `d5`, index $t_{\text{file}} = t - 20000$
   - If $25000 \le t < 30000$: File `d6`, index $t_{\text{file}} = t - 25000$
   - If $30000 \le t < 33902$: File `d7`, index $t_{\text{file}} = t - 30000$

2. **Clinical Group Assignment:**
   - If $0 \le t < 10129$: Block 0 $\rightarrow$ `HC` (label 0)
   - If $10129 \le t < 23160$: Block 1 $\rightarrow$ `ADD` (label 1)
   - If $23160 \le t < 33902$: Block 2 $\rightarrow$ `ADHD` (label 2)

3. **Composite Subject Identifier (Safe Split Key):**
   - Constructed as `f"{clinical_group}_{within_block_subject_index:02d}"`
   - Yields exactly 144 unique identifiers: `HC_01` to `HC_44`, `ADD_01` to `ADD_52`, `ADHD_01` to `ADHD_48`.
   - Every trial carries this composite key, guaranteeing strict zero-leakage subject partitioning.

4. **Candidate Filename:**
   - Mapped ordinally to `sub_name_stim{block_index}{within_block_subject_index - 1}`.

---

## 16. Gate 1 Verdict

### Relationship-by-Relationship Evidence Classification:

| Relationship / Component | Evidence Classification | Justification |
|---|:---:|---|
| **EEG File Shapes & Dtypes** | `VALIDATED` | Direct HDF5 header inspection; all 7 files verified; zero NaNs/Infs. |
| **Sequential File Ordering (`d1` $\rightarrow$ `d7`)** | `VALIDATED` | Ascending chronological timestamps + empirical boundary signal correlations ($r \ge 0.9084$ across all 6 boundaries). |
| **`y_stim` Row 0 Subject Runs** | `VALIDATED` | Exactly 144 contiguous runs; 0 missing, duplicate, or out-of-order indices. |
| **Clinical Diagnostic Group Assignment** | `VALIDATED` | Exact match with He et al. (2023) Section 2.1 on subject counts (44/52/48) and trial counts (10,129 / 13,031 / 10,742). |
| **Composite Subject Key for Cross-Validation** | `VALIDATED` | 144 mutually exclusive, non-overlapping participant groups; leak-free. |
| **`sub_name_stim` Filename Ordinal Mapping** | `STRONGLY SUPPORTED` | Matches timestamps, block counts, and MATLAB `dir()` sort order, but lack of individual MAT files prevents direct per-file trial count verification. |

---

### Formal Gate 1 Declaration:

```
======================================================================
GATE 1 VERDICT:
VALIDATED
======================================================================
```

**Verdict Details:**
- The deterministic mapping of **EEG trial $\rightarrow$ source file $\rightarrow$ within-file index $\rightarrow$ global column $\rightarrow$ composite subject ID $\rightarrow$ clinical group** is **`VALIDATED`**.
- The candidate mapping of **composite subject ID $\rightarrow$ specific filename string** is **`STRONGLY SUPPORTED`**. Because this remaining inference has no impact on subject-level cross-validation safety, Gate 1 passes without blocking downstream ML modeling.
