# 02 — Dataset and Data Contract Audit

**Date:** 2026-10-06
**Purpose:** Establish what data we actually have — every fact, every limit, every open question.
**Audience:** CS student with basic programming knowledge, new to EEG, neuroscience, signal processing, and machine learning.
**Scope:** Data facts only. No preprocessing design. No architecture design. No model representations.

---

## How to Read This Document

Every important claim is labelled with one of four evidence ratings:

| Label | Meaning |
|---|---|
| **VERIFIED** | Directly confirmed by reading the binary files, running scripts, or found word-for-word in the base paper. No interpretation required. |
| **STRONGLY SUPPORTED** | Multiple independent observations all point the same way. Highly likely to be correct, but not directly stated in a file header or paper sentence. |
| **INFERRED** | A reasonable guess based on partial evidence. Treat it as a hypothesis, not a fact. |
| **UNKNOWN** | Not established by any evidence available in this project. Must not be assumed. |

If something is UNKNOWN, it is left UNKNOWN. It is not silently treated as known.

---

## Plain-Language Glossary

| Term | What it means in plain language |
|---|---|
| **EEG** | A recording of your brain's electrical activity, captured by small electrodes placed on your scalp. The output is a grid of numbers: one column per electrode, one row per moment in time. |
| **Trial / epoch** | One short, isolated snippet of EEG recording — in this dataset, roughly 1.5 seconds long. Think of it as one "observation" or one data row for the model. |
| **Channel** | One electrode on the scalp. This dataset has 56 channels. Each channel produces a stream of numbers over time. |
| **Sample / time step** | One measurement taken at one instant in time across all electrodes. Each trial in this dataset has 385 of these instants. |
| **Shape (385, 56)** | A 2D table: 385 rows (one per time step) × 56 columns (one per electrode). |
| **float32** | A number format. Each number takes 32 bits of storage and can represent decimals (e.g., −247.3). It is the standard format for most ML datasets. |
| **float64** | Like float32 but 64 bits — higher precision. Used in the metadata files here. |
| **HC / ADD / ADHD** | The three clinical groups: Healthy Control, Attention Deficit Disorder (inattentive subtype), Attention Deficit Hyperactivity Disorder (combined subtype). |
| **.mat file** | A file format made by MATLAB (a scientific computing tool). These files can store arrays, structs, and cell arrays. Internally, newer MATLAB files use the HDF5 format — the same format that h5py in Python can read. |
| **HDF5** | A widely used binary file format for large scientific data. It supports compression and chunking. |
| **One-hot encoding** | A way of labelling categories with binary vectors. If there are 3 categories, category A = [1,0,0], category B = [0,1,0], category C = [0,0,1]. |
| **Subject-level split** | Keeping all trials from the same person in the same data partition. Essential for testing a model on people it has truly never seen. |
| **OSF** | Open Science Framework — a public platform where researchers share datasets and code. |

---

## 1. Dataset Source

**VERIFIED** — The dataset comes from:

- **Paper:** Yuchao He et al., "Classification of attention deficit/hyperactivity disorder based on EEG signals using a EEG-Transformer model," *Journal of Neural Engineering*, Vol. 20 (2023), article 056013.
- **DOI:** https://doi.org/10.1088/1741-2552/acf7f5
- **OSF project:** Node `6594x` — https://osf.io/6594x/

**VERIFIED** — The OSF `readme.txt` explicitly states that the large sample data was split into multiple files because a single file would exceed the OSF size limit. The splitting is a storage decision only — it does not reflect any meaningful division in the data itself.

**VERIFIED** — All 10 dataset files are present locally. Their byte sizes match the official OSF listing byte-for-byte.

---

## 2. Number of Subjects

**VERIFIED** — There are **144 subjects** in total.

Evidence chain:
1. The base paper (He et al. 2023, Section 2.1) explicitly reports 144 children.
2. `sub_name_stim.mat` was inspected directly. It contains three blocks of filename strings with lengths 44, 52, and 48. 44 + 52 + 48 = 144.
3. `y_stim.mat` row 0 contains exactly 144 contiguous subject-index runs spanning all 33,902 columns — no gaps, no duplicates.

> [!NOTE]
> The base paper also contains at least one internal inconsistency (mentioning "300 subjects" in a different section). That number is **UNKNOWN** in origin and must not be used. The verified subject count from binary file inspection and the paper's own participant section is 144.

---

## 3. Target Groups (Clinical Labels)

**VERIFIED** — There are three clinical groups:

| Group | Label | Subjects | Trials |
|---|---|---|---|
| Healthy Controls | HC | 44 | 10,129 |
| Attention Deficit Disorder (inattentive subtype) | ADD | 52 | 13,031 |
| ADHD Combined Type | ADHD | 48 | 10,742 |
| **Total** | — | **144** | **33,902** |

Evidence:
- **VERIFIED** — These exact counts appear in He et al. (2023), Section 2.1.
- **VERIFIED** — `sub_name_stim.mat` contains three blocks with 44, 52, and 48 filenames whose prefixes are `controls`, `subtype1`, and `subtype2` respectively.
- **VERIFIED** — `y_stim.mat` row 0 partitions into exactly three index runs (1–44, 1–52, 1–48) whose trial totals are 10,129, 13,031, and 10,742 — an exact match to the paper's published counts.
- **STRONGLY SUPPORTED** — `controls` → HC, `subtype1` → ADD, `subtype2` → ADHD, confirmed by filename tokens and exact count alignment with paper.

---

## 4. Number of Trials

**VERIFIED** — The dataset contains exactly **33,902 trials** in total.

How this was confirmed:
- `y_stim.mat` has shape (4, 33,902). It has one column per trial.
- The 7 EEG files together hold 6 × 5,000 + 3,902 = **33,902 trials**.
- These two counts match exactly.
- The paper (He et al. 2023) reports the same figure.

> [!NOTE]
> **What is a trial?** One trial is one short EEG recording window, roughly 1.5 seconds. Think of it as: one child, one moment in time, one experimental "observation." The dataset was recorded while children performed a cognitive task, and each task event was captured as one such window. You can think of the 33,902 trials as 33,902 rows in a very large table.

---

## 5. Files and Their Roles

**VERIFIED** — The dataset contains exactly 10 files.

### EEG Data Files (what the brain signal actually looks like)

| File | Size (bytes) | Shape | Role |
|---|---:|---|---|
| `d1.mat` | 402,642,207 | (385, 56, 5000) | EEG trials 1–5,000 |
| `d2.mat` | 402,606,807 | (385, 56, 5000) | EEG trials 5,001–10,000 |
| `d3.mat` | 402,837,633 | (385, 56, 5000) | EEG trials 10,001–15,000 |
| `d4.mat` | 402,887,010 | (385, 56, 5000) | EEG trials 15,001–20,000 |
| `d5.mat` | 402,691,834 | (385, 56, 5000) | EEG trials 20,001–25,000 |
| `d6.mat` | 402,722,744 | (385, 56, 5000) | EEG trials 25,001–30,000 |
| `d7.mat` | 314,191,152 | (385, 56, 3902) | EEG trials 30,001–33,902 |

Shapes are **VERIFIED** from direct binary (HDF5) inspection. File orderings are **VERIFIED** — see Section 11.

### Metadata Files (information about the trials and subjects)

| File | Size (bytes) | Shape | Role |
|---|---:|---|---|
| `y_stim.mat` | 1,153,360 | (4, 33902) | Per-trial metadata: subject index and group identity |
| `sub_name_stim.mat` | 70,128 | (1, 3) → inner blocks 44/52/48 | Subject filename strings, one per subject |
| `chan.mat` | 3,135 | (1, 60) | Channel names and spatial coordinates |

All sizes and shapes are **VERIFIED**.

---

## 6. EEG Dimensions — What the Numbers Mean

**VERIFIED** — Each EEG file stores a 3-dimensional array. For `d1.mat` through `d6.mat`, the shape is:

```
(385, 56, 5000)
  │    │    │
  │    │    └── 5,000 trials in this file
  │    └──────── 56 channels (electrodes)
  └───────────── 385 time steps per trial
```

**In plain language:** Imagine 5,000 photographs of brain activity. Each photograph is a grid: 385 rows (moments in time) and 56 columns (electrodes). The shape `(385, 56, 5000)` is just that stack of 5,000 photographs.

When you read one trial, it becomes a 2D array of shape **(385, 56)** — 385 rows × 56 columns.

**STRONGLY SUPPORTED** — The axis order is (time steps, channels, trials). This is consistent with the paper's description of model input `(385, 56)`.

**VERIFIED** — `d7.mat` has shape `(385, 56, 3902)` — same time and channel dimensions, but only 3,902 trials (the remainder after 6 × 5,000).

---

## 7. Number of Channels

**VERIFIED** — The EEG arrays have **56 channels**.

This is confirmed by direct binary inspection of all 7 EEG files. Every file has exactly 56 in its channel dimension, without exception.

**VERIFIED** — The base paper also reports 56 channels.

> [!NOTE]
> A "channel" is one electrode. You can imagine 56 small sensors placed at different locations on the scalp. Each sensor continuously measures the brain's electrical activity at that spot. In the data, each channel appears as one column in the (385, 56) trial array.

> [!IMPORTANT]
> `chan.mat` contains 60 channel descriptions — not 56. This mismatch is discussed in Section 17 (the 56-versus-60 issue).

---

## 8. Number of Samples per Trial

**VERIFIED** — Every trial has **385 time steps** (also called samples or time points).

This is the first dimension of the (385, 56) trial array. It is confirmed by direct binary inspection of all 7 files — every single trial across all 33,902 has exactly 385 time steps, zero exceptions.

> [!NOTE]
> **What is a sample?** The EEG equipment measures electrical voltage at all 56 electrodes simultaneously, many times per second. Each such simultaneous measurement is called a "sample" or "time step." So a trial with 385 samples means the equipment took 385 snapshots during that trial window.

> [!WARNING]
> **The 385 vs 384 discrepancy is unresolved.** At a sampling rate of 256 Hz and a 1.5-second window, you would expect 256 × 1.5 = **384 samples**. The files have **385**. The physical meaning of the extra sample is **UNKNOWN**. It has not been discarded, cropped, or assigned any meaning. All 385 must be preserved as-is until this is resolved.

---

## 9. Sampling Rate

**VERIFIED FROM PAPER** — The sampling rate is **256 Hz** (256 measurements per second).

This is explicitly stated in He et al. (2023). It is not encoded in the file headers themselves — it appears only in the paper.

> [!NOTE]
> A sampling rate of 256 Hz means the equipment records 256 snapshots of all 56 channels every second. So in 1 second, you accumulate 256 rows in the (time steps, channels) table. In 1.5 seconds, you would expect 384 rows. The files have 385 — see Section 8 above.

---

## 10. Data Type

**VERIFIED** — The EEG values are stored as **float32** (32-bit floating point numbers).

This was confirmed by direct binary inspection of all 7 EEG files. The MATLAB internal class is `single`, which is equivalent to float32.

**VERIFIED** — The metadata in `y_stim.mat` is stored as **float64** (64-bit floating point). Even though it contains integers (like subject index 1, 2, 3…), they are stored as float64 numbers.

**VERIFIED** — The value range across all trials is approximately **−500 to +500**. The exact observed range by file:

| File | Min | Max |
|---|---|---|
| d1 | −313.2 | +286.0 |
| d2 | −482.8 | +357.5 |
| d3 | −288.5 | +505.2 |
| d4 | −466.1 | +464.2 |
| d5 | −428.4 | +417.0 |
| d6 | −408.9 | +329.9 |
| d7 | −444.5 | +312.5 |

**UNKNOWN** — The unit of these values. They are almost certainly microvolts (µV), which is the standard unit for EEG. But this is not explicitly confirmed in any file header or project documentation.

**UNKNOWN** — The reference electrode (the "zero point" against which all values are measured). This information is not available in the dataset files.

**VERIFIED** — There are **zero NaN values** and **zero Inf values** across all 33,902 trials. The entire 2.73 GB dataset is clean with no missing or corrupted entries.

---

## 11. Relationship Between Files and Trials

**VERIFIED** — The 7 EEG files form one continuous, sequential dataset.

The file ordering was confirmed by two independent pieces of evidence:

**Evidence 1 — MATLAB file creation timestamps (VERIFIED):**
The binary headers of the 7 EEG files contain MATLAB creation timestamps. Inspected directly, they show:

| File | Timestamp |
|---|---|
| d1.mat | Fri Mar 15 17:59:45 2019 |
| d2.mat | Fri Mar 15 18:00:06 2019 |
| d3.mat | Fri Mar 15 18:00:25 2019 |
| d4.mat | Fri Mar 15 18:00:45 2019 |
| d5.mat | Fri Mar 15 18:01:05 2019 |
| d6.mat | Fri Mar 15 18:01:25 2019 |
| d7.mat | Fri Mar 15 18:01:41 2019 |

The files were written one after another in strict order, 16–21 seconds apart, within a single 2-minute window. This is consistent with a single automated splitting process.

**Evidence 2 — Signal fingerprint tests at all 6 file boundaries (VERIFIED):**

> [!NOTE]
> **How the boundary test works:** When `d1` ends and `d2` begins, the last trials of `d1` and the first trials of `d2` should belong to the same child — if and only if the files are in the right order. A child's brain signal has a personal signature (some channels tend to be more active than others). We measure how similar the signal signature is at each file boundary, and compare it to the signature of a completely different child.

Results:

| Boundary | Same-person correlation (r_same) | Different-person control (r_ctrl) | Verdict |
|---|---|---|---|
| d1/d2 | 0.9152 | 0.3234 | CORROBORATED |
| d2/d3 | 0.9682 | 0.2776 | CORROBORATED |
| d3/d4 | 0.9897 | 0.5656 | CORROBORATED |
| d4/d5 | 0.9977 | 0.4333 | CORROBORATED |
| d5/d6 | 0.9084 | 0.3543 | CORROBORATED |
| d6/d7 | 0.9105 | 0.0467 | CORROBORATED |

r_same exceeds 0.90 at every single boundary. r_ctrl is always far lower. This confirms the files are sequential and correctly ordered.

**VERIFIED** — The mapping from EEG file to global trial index is:

| File | Global trial range (0-indexed) | Trial count |
|---|---|---|
| d1 | [0, 5000) | 5,000 |
| d2 | [5000, 10000) | 5,000 |
| d3 | [10000, 15000) | 5,000 |
| d4 | [15000, 20000) | 5,000 |
| d5 | [20000, 25000) | 5,000 |
| d6 | [25000, 30000) | 5,000 |
| d7 | [30000, 33902) | 3,902 |

---

## 12. Relationship Between Trials and Subjects

**VERIFIED** — Every one of the 33,902 trials belongs to exactly one subject, and subject membership is recorded in `y_stim` row 0.

How it works:

- `y_stim` row 0 contains a subject index number for each of the 33,902 trials.
- The values are integers (stored as float64): 1, 2, 3, …
- The values form 144 contiguous runs — one run per subject.
- Within each run, all values are the same number (the subject's within-block index).
- The runs appear in strictly ascending order within each group block.

**VERIFIED** — The 144 runs partition perfectly into three blocks:

| Block | Subject indices | Trial columns (0-indexed) | Total trials |
|---|---|---|---|
| Block 0 | 1 → 44 | [0, 10129) | 10,129 |
| Block 1 | 1 → 52 | [10129, 23160) | 13,031 |
| Block 2 | 1 → 48 | [23160, 33902) | 10,742 |

**VERIFIED** — Per-subject trial counts (within-block index → trial count):

**HC (Block 0), 44 subjects:**
281, 262, 295, 229, 255, 135, 232, 254, 159, 272, 279, 168, 137, 237, 205, 115, 195, 229, 222, 209, 160, 229, 203, 99, 229, 241, 79, 287, 102, 279, 278, 276, 280, 241, 288, 273, 267, 259, 273, 279, 292, 290, 277, 278

**ADD (Block 1), 52 subjects:**
266, 249, 220, 276, 265, 224, 269, 276, 252, 272, 264, 263, 255, 231, 264, 206, 199, 239, 191, 248, 255, 172, 262, 269, 226, 199, 267, 268, 267, 282, 270, 275, 219, 278, 290, 245, 281, 249, 253, 294, 289, 290, 283, 263, 197, 251, 257, 258, 171, 259, 184, 279

**ADHD (Block 2), 48 subjects:**
257, 279, 46, 259, 205, 262, 214, 262, 261, 265, 285, 255, 213, 238, 168, 125, 239, 195, 286, 197, 234, 203, 223, 265, 255, 226, 264, 259, 260, 290, 287, 229, 172, 219, 217, 129, 187, 51, 269, 234, 124, 204, 262, 254, 142, 270, 219, 283

Summary statistics:
- HC: min 79 (subject 27), max 295 (subject 3), mean ± SD = 230.2 ± 56.6
- ADD: min 171 (subject 49), max 294 (subject 40), mean ± SD = 250.6 ± 31.0
- ADHD: min 46 (subject 3), max 290 (subject 30), mean ± SD = 223.8 ± 57.3

> [!IMPORTANT]
> Trial counts vary dramatically — from 46 to 295. This means some subjects contribute 6× more training examples than others. Any ML evaluation that ignores this imbalance will be biased toward high-trial subjects.

**VERIFIED** — The safe, unique subject key for ML splitting is the composite identifier: `{group}_{within-block-index:02d}`, yielding `HC_01` through `HC_44`, `ADD_01` through `ADD_52`, `ADHD_01` through `ADHD_48`. This key is 100% unique across all 144 subjects and guarantees no trial from the same person crosses partition boundaries.

---

## 13. Relationship Between Subjects and Clinical Groups

**STRONGLY SUPPORTED** → borders on VERIFIED through count alignment.

The three blocks in `y_stim` and `sub_name_stim` correspond to the three clinical groups. Evidence:

1. **VERIFIED** — `sub_name_stim.mat` Block 0 contains 44 filenames all beginning with `controls`.
2. **VERIFIED** — `sub_name_stim.mat` Block 1 contains 52 filenames all beginning with `subtype1`.
3. **VERIFIED** — `sub_name_stim.mat` Block 2 contains 48 filenames all beginning with `subtype2`.
4. **VERIFIED** — The paper reports 44 HC, 52 ADD, 48 ADHD — a perfect match to the block sizes.
5. **VERIFIED** — The paper reports 10,129 / 13,031 / 10,742 trials per group — a perfect match to the block trial totals.
6. **STRONGLY SUPPORTED** — Clinical terminology in the filenames: `subtype1` matches the DSM-IV inattentive subtype (ADD), `subtype2` matches the combined subtype (ADHD).
7. **VERIFIED** — Filename tokens from the Dresden ADHD multicenter study: `NF` (neurofeedback), `WK`/`gesWK` (waitlist control / healthy waitlist control), `ET` (parent training), `MPH` (methylphenidate), `PHY` (physical therapy), `pr` (pre-intervention), `post` (post-intervention).

**UNKNOWN** — No file in the dataset contains an explicit per-participant clinical diagnosis field. The group assignment rests on filename token matching and count alignment, not a labelled diagnosis table.

---

## 14. y_stim — Meaning and What Is NOT Established

**VERIFIED** — `y_stim.mat` has shape **(4 rows × 33,902 columns)**, stored as float64.

Each column corresponds to one trial. The four rows carry the following information:

### Row 0 — Subject Index Within Block

**VERIFIED** — Row 0 contains the within-block subject index for each trial.

- Values are integers (stored as float64): 1, 2, 3, …
- They reset to 1 at the start of each new clinical group block.
- For Block 0 (HC): values run 1 through 44.
- For Block 1 (ADD): values run 1 through 52.
- For Block 2 (ADHD): values run 1 through 48.
- Row 0 alone is NOT a unique subject identifier. Subject index 3 appears in HC, ADD, and ADHD. You must always combine the row-0 value with the block identity to get a unique subject.

### Rows 1, 2, 3 — Clinical Group Block Indicators

**VERIFIED** — Rows 1–3 form a one-hot encoding of the clinical group.

Direct inspection of all 33,902 columns found:
- For every trial in Block 0 (HC): rows 1–3 are exactly [1, 0, 0].
- For every trial in Block 1 (ADD): rows 1–3 are exactly [0, 1, 0].
- For every trial in Block 2 (ADHD): rows 1–3 are exactly [0, 0, 1].
- The indicator never changes within a subject. It never changes within a block. It never violates the one-hot property.

> [!IMPORTANT]
> **Critical correction from prior misconception:** Earlier project documents hypothesized that rows 1–3 might represent stimulus categories (e.g., target vs. non-target events). This was **disproven** by direct data inspection. Rows 1–3 are block (clinical group) indicators — they are constant for every trial belonging to the same clinical group.

**WHAT IS NOT ESTABLISHED about y_stim:**

1. **The physical meaning of the word "stim"** in the filename `y_stim.mat` is UNKNOWN. It appears to be an artifact of the original MATLAB variable naming in the study, not a confirmed indicator of a stimulus type.

2. **Whether rows 1–3 may be used directly as classification labels** — they may not. The clinical label must be derived from the block position (0, 1, or 2), not from rows 1–3 directly. Using rows 1–3 as labels is not approved.

3. **Any direct stimulus event information** (what happened during the trial, what the child was responding to, what the target/non-target structure was) is **UNKNOWN** from these files alone. No stimulus event codes are present in y_stim.

---

## 15. sub_name_stim — What It Contains

**VERIFIED** — `sub_name_stim.mat` is a MATLAB v7.3 (HDF5) file containing a cell array of shape (1, 3).

Each of the 3 inner cells is a list of filename strings, representing the source files for each clinical group:

| Block | Prefix | Count | Example filename | Lexicographically sorted? |
|---|---|---|---|---|
| 0 | `controls` | 44 | `controls2MSCT_Time_Estimation_NF_VP_02C.mat` | Yes |
| 1 | `subtype1` | 52 | `subtype1AB237_NF_pr.mat` | Yes |
| 2 | `subtype2` | 48 | `subtype2AE231_ET_pr.mat` | Yes |

**VERIFIED** — All 144 filenames are globally unique (no repeats within or across blocks).

**VERIFIED** — `sub_name_stim.mat` was created at the same instant as `y_stim.mat`: **Fri Feb 22 14:52:45 2019** (identical creation timestamp down to the second). This is strong evidence they were compiled together in the same MATLAB workspace.

**STRONGLY SUPPORTED** — The filenames are stored in alphabetical order (lexicographic sort). This matches MATLAB's standard `dir()` function behaviour, which returns files sorted alphabetically. If the original code iterated over the sorted file list with a loop index, the index directly corresponds to each filename's position in the sorted list.

**STRONGLY SUPPORTED** — The ordinal index in `y_stim` row 0 (1, 2, 3, …) maps to the filename at that position in the corresponding sorted block (index 1 → filename at position 0, using 0-based indexing).

**UNKNOWN** — The individual subject `.mat` files listed in `sub_name_stim.mat` are NOT available on OSF. This means the filename-to-subject-index ordinal mapping cannot be independently verified by cross-checking individual trial counts. However, this limitation does NOT affect the safety of subject-level ML cross-validation — the composite subject key `(block, within-block-index)` is still fully deterministic and leak-free.

---

## 16. chan.mat — What It Contains

**VERIFIED** — `chan.mat` is a MATLAB v5 (traditional binary) file containing a struct array of shape (1, 60).

Each of the 60 entries describes one EEG channel and includes the following fields:
- `labels` — the channel name (e.g., `Cz`, `FCz`, `Fz`, `Fp1`, `Fp2`, `Oz`, `Iz`)
- `unit` — measurement unit
- `sph_radius`, `sph_theta`, `sph_phi` — spherical coordinates of the electrode on the scalp
- `theta`, `radius` — polar coordinates
- `X`, `Y`, `Z` — 3D Cartesian coordinates
- `type` — channel type (e.g., EEG, EOG, etc.)
- `ref` — reference information

**VERIFIED** — The channel names follow the standard 10–20 international system, which is the most common convention for placing EEG electrodes.

**VERIFIED** — The EEG data arrays (`d1.mat` – `d7.mat`) have **56** channels.

**VERIFIED** — `chan.mat` has **60** channel entries.

These two numbers do not match. See Section 17 for the full discussion.

---

## 17. The 56-Channel vs. 60-Entry Issue

**VERIFIED** — EEG data arrays have 56 columns (channels). `chan.mat` has 60 channel metadata entries. The difference is exactly 4.

**UNKNOWN** — Which 4 channels from `chan.mat` are absent from the EEG arrays.

**UNKNOWN** — Why those 4 channels are absent. Possible explanations include:
- They were reference electrodes (used as the "zero" baseline, then removed after re-referencing).
- They were EOG (eye movement) electrodes, removed after ICA artifact cleaning.
- They were mastoid or other non-brain electrodes not included in the exported data.
- They were dropped due to poor signal quality.

None of these explanations is established by any file in the project. Every one of the above is speculation.

> [!CAUTION]
> Do NOT assume you know which 4 channels are missing. Do NOT map the 56 EEG data columns to specific `chan.mat` entries without evidence. Any experiment that requires knowing which electrode each data column represents (e.g., scalp topography maps) is BLOCKED until this is resolved.

> [!NOTE]
> For pure signal classification (treating the 56 columns as 56 anonymous features), the 56-vs-60 mismatch does not block you. It only blocks experiments that need to know the physical location of each column.

---

## 18. d1–d7 Ordering

**VERIFIED** — `d1.mat` through `d7.mat` are in strict sequential order and map to `y_stim` columns in that order.

This was established by two independent and complementary methods:

**Method 1 — File timestamps:**
Binary headers show the files were created in ascending chronological order (`d1` first, `d7` last) in a single 2-minute window, consistent with an automated sequential split.

**Method 2 — Empirical signal boundary test:**
At all 6 file junctions (`d1`/`d2`, `d2`/`d3`, …, `d6`/`d7`), the EEG signal of the subject who straddles the boundary was found on both sides with a Pearson correlation r >= 0.90. An unrelated subject's signal gave correlation r < 0.57. This was tested at all 6 boundaries with no failures.

The mapping is:
- `d1` ↔ `y_stim` columns 0–4999
- `d2` ↔ `y_stim` columns 5000–9999
- `d3` ↔ `y_stim` columns 10000–14999
- `d4` ↔ `y_stim` columns 15000–19999
- `d5` ↔ `y_stim` columns 20000–24999
- `d6` ↔ `y_stim` columns 25000–29999
- `d7` ↔ `y_stim` columns 30000–33901

---

## 19. Validated Mappings

This section summarises every mapping that has been verified and is safe to rely on.

| Mapping | Status | Evidence |
|---|---|---|
| Global trial index → source file | **VERIFIED** | Timestamp + boundary signal correlation tests |
| Global trial index → within-file index | **VERIFIED** | Direct arithmetic from file trial counts |
| Global trial index → y_stim column | **VERIFIED** | Trial count sum = y_stim column count; boundary tests |
| y_stim column → within-block subject index (row 0) | **VERIFIED** | 144 contiguous runs, 0 gaps, 0 duplicates |
| y_stim columns 1–3 → clinical group one-hot | **VERIFIED** | All 33,902 columns checked; perfect block alignment |
| Block 0 → HC | **STRONGLY SUPPORTED** | Filename prefix + paper count match |
| Block 1 → ADD | **STRONGLY SUPPORTED** | Filename prefix + paper count match |
| Block 2 → ADHD | **STRONGLY SUPPORTED** | Filename prefix + paper count match |
| Composite subject key (`HC_01`…`ADHD_48`) | **VERIFIED** | 144 unique, non-overlapping identifiers |
| Within-block subject index → specific filename | **STRONGLY SUPPORTED** | Timestamp alignment + MATLAB `dir()` sort behaviour; cannot be independently verified without individual files |
| EEG column → chan.mat entry | **UNKNOWN** | 56 data columns vs. 60 metadata entries; mapping not established |

---

## 20. Remaining Uncertainties

The following questions are formally **UNKNOWN** and must not be silently resolved:

| # | Uncertainty | Why it matters |
|---|---|---|
| 1 | Which 4 channels are missing between `chan.mat` (60) and the EEG data (56)? | Blocks spatial/topographic analysis |
| 2 | What is the physical meaning of the 385th time step? (256 Hz × 1.5 s = 384 expected) | Affects any interpretation of trial boundaries |
| 3 | Are the OSF EEG files already ICA-cleaned, or do they contain raw (unprocessed) EEG? | Determines whether a second ICA pass would double-clean |
| 4 | What is the electrode reference scheme used for the 56 channels? | Affects amplitude interpretation and any re-referencing |
| 5 | What are the exact physical units of the EEG values? | Likely µV but not confirmed |
| 6 | Were the 1.5-second trial windows extracted with any overlap? | Affects the independence assumption of trials |
| 7 | What stimulus was presented during each trial? | y_stim contains no stimulus event codes |
| 8 | Does the paper's stated ICA procedure exactly match what was applied to the OSF files? | The paper gives no algorithm, component count, or rejection rule |
| 9 | Which internal paper inconsistencies affect the dataset description? | The paper contains >=10 internal contradictions, including a "300 subjects" mention |

---

## Summary Section A — Data Facts We Can Safely Use

These facts are **VERIFIED** or **STRONGLY SUPPORTED** and can be used without qualification:

1. **33,902 trials** total, across 144 subjects in 3 groups.
2. Every trial has shape **(385, 56)** — 385 time steps, 56 channels — stored as **float32**.
3. All data is **complete** — zero NaN, zero Inf, across all files.
4. The **7 EEG files are sequential**: d1 → d7, in that order, with no gaps.
5. The file-to-y_stim column mapping is deterministic and verified (see Section 18).
6. The **144 subject runs** in y_stim row 0 are contiguous, complete, and unique.
7. The **composite subject key** (`HC_01`…`ADHD_48`) is safe for subject-level cross-validation.
8. The **clinical groups** are: 44 HC, 52 ADD, 48 ADHD.
9. **y_stim rows 1–3 are group block indicators**, not stimulus categories.
10. **Clinical labels must be derived from block position** (0, 1, or 2), not from y_stim rows directly.
11. **Trial counts per subject vary widely** (min 46, max 295) — this imbalance must be tracked.
12. The **sampling rate is 256 Hz** (paper-reported).
13. Values likely represent microvolts (INFERRED, not confirmed).

---

## Summary Section B — Data Facts We Must Be Careful About

These facts are **STRONGLY SUPPORTED** or **INFERRED** and carry caveats:

1. **The filename-to-subject-index ordinal link** (`sub_name_stim` position → row-0 index) is strongly supported but not independently verified. It does NOT affect ML evaluation safety.
2. **Clinical group labels** (HC/ADD/ADHD) come from filename token matching and count alignment, not from an explicit per-participant diagnosis field in any file.
3. **The 385th sample** is unresolved. Do not crop it, pad it, or assign it physical meaning without evidence.
4. **The 56-to-60 channel mismatch** — the 56 EEG columns cannot be named or spatially located until the missing 4 are identified.
5. **Value units** are likely µV but not formally confirmed.
6. **ICA state** of the files is unknown — do not assume the data is raw or cleaned without evidence.
7. **The base paper contains internal inconsistencies** — do not treat all paper statements as equally reliable.

---

## Summary Section C — Unresolved Dataset Questions

Questions that must be answered (or formally deferred) before certain downstream steps are possible:

1. Which 4 channels are absent from the EEG arrays vs. `chan.mat`?
2. What is the physical meaning of sample position 385?
3. Were the downloaded files already ICA-cleaned?
4. What is the electrode reference scheme?
5. What are the EEG value units?
6. Were trial windows extracted with any temporal overlap?
7. What stimulus occurred during each trial?
8. Which specific paper inconsistencies affect numbers reported in the paper?

---

## Summary Section D — Constraints Any Future Architecture Must Respect

These are not architecture decisions. They are hard constraints established by the data itself. Any future model design must respect all of them:

1. **Input shape is (385, 56) per trial.** This is fixed by the data. It cannot be changed without a documented preprocessing decision that has been explicitly approved.

2. **There are exactly 3 output classes:** HC, ADD, ADHD. This is the classification target.

3. **Subject must be the unit of splitting.** All 385 × 56 values for one subject must stay in one partition. Trial-level splitting is prohibited.

4. **There are only 144 subjects.** Any cross-validation or test design is limited by this number. Folds must be designed with this in mind.

5. **Trial counts are unequal across subjects** (46 to 295). Any training objective must account for this, or else high-trial subjects will dominate.

6. **No spatial channel mapping may be assumed.** The 56 data columns cannot be assigned to named electrodes or scalp positions until the 56-vs-60 mismatch is resolved.

7. **All 385 time steps must be preserved** until the 385th sample's meaning is resolved.

8. **float32 must be preserved.** Silent upcasting or downcasting requires an explicit decision.

9. **Labels must come from block identity, not y_stim rows 1–3 directly.** The block boundary positions are the authoritative source of group identity.

10. **The raw .mat files are immutable.** No code may write to them. All derived data goes to a separate, versioned output location.

---

*Sources used for this audit:*
- *`docs/dataset-investigation.md` — initial structural investigation*
- *`dataset_metadata_investigation.md` — script-based metadata inspection*
- *`docs/ADHD_EEG_PREPROCESSING_AUDIT.md` — binary-level audit with evidence classification*
- *`docs/dataset-preprocessing-audit.md` — preprocessing methodology audit*
- *`docs/trial_subject_mapping_validation.md` — Sprint 1 Gate 1 validation report*
- *`docs/architecture-preparation/01_PROJECT_STATE_AUDIT.md` — project state audit*
- *`scripts/validate_trial_subject_mapping.py` — the script that performed Gate 1 validation*
- *He et al. (2023), Journal of Neural Engineering — base paper*

*Document produced by: Antigravity AI (read-only inspection — no architecture decisions, no preprocessing, no model choices)*
*No data was modified. No preprocessing was performed.*
