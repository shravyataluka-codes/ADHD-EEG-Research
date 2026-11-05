# Historical Dataset Metadata Investigation Report

> [!WARNING]
> **Historical initial interpretation; identity and diagnosis conclusions below
> are superseded.** The inspected arrays show shapes, counts, row-0 value runs,
> filename prefixes, and one-hot patterns. They do not independently verify the
> trial-to-column or row-0-to-filename correspondence. Row 0 is a candidate
> within-block index, filename blocks are candidate participant mappings, and
> rows 1–3 have unverified semantics. None of these is a verified participant
> identity or clinical diagnosis label. Do not use `y_stim` rows 1–3 as clinical
> targets.

This historical report records observations from an earlier script-based inspection of the raw `.mat` files using Python (`h5py` and `scipy.io`); inferred correspondence and semantics are not verified.

### 1. `sub_name_stim.mat`
- **File Format:** MATLAB v7.3 (HDF5-based).
- **Internal Structure:** A cell array containing 3 nested cell arrays of strings. 
- **Shape/Dimensions:** The top-level array shape is `(1, 3)`. The inner arrays have lengths of `(1, 44)`, `(1, 52)`, and `(1, 48)`.
- **Data Types:** HDF5 Object References pointing to 16-bit character arrays (strings).
- **What Each Dimension Represents:**
  - **Dimension 1 (Length 3):** Contains 3 filename blocks.
  - **Dimension 2:** Contains filename strings. Their mapping to verified participants is not established.
- **Observed Values by Candidate Range:**
  - **Group 1:** Contains 44 filenames. All filenames begin with the prefix `'controls'` (e.g., `'controls2MSCT_Time_Estimation_NF_VP_02C.mat'`).
  - **Group 2:** Contains 52 filenames. All filenames begin with the prefix `'subtype1'` (e.g., `'subtype1AB237_NF_pr.mat'`).
  - **Group 3:** Contains 48 filenames. All filenames begin with the prefix `'subtype2'` (e.g., `'subtype2AE231_ET_pr.mat'`).
  - Total filename entries across the three blocks: **144** (not 144 verified identities).

### 2. `y_stim.mat`
- **File Format:** MATLAB v7.3 (HDF5-based).
- **Internal Structure:** A single 2D numeric matrix named `y_stim`.
- **Shape/Dimensions:** `(4, 33902)`.
- **Data Types:** `float64`.
- **What Each Dimension Represents:**
  - **Columns (33,902):** Correspond to individual observations/events—almost certainly EEG trials or epochs.
  - **Rows (4):** Contain metadata for each trial. 
    - **Row 0:** Contains integer values ranging from 1.0 to 52.0, organized into contiguous runs. These are candidate within-block indices; participant identity is unverified.
    - **Rows 1, 2, 3:** Contain strictly binary values (`0.0` or `1.0`).
- **Unique Values and Group Info:** 
  - An analysis of Rows 1-3 reveals exactly three mutually exclusive combinations (one-hot encoding):
    - `[1, 0, 0]` appears in **10,129** columns. The associated Row 0 values range from 1 to 44, matching the size of candidate filename block 1.
    - `[0, 1, 0]` appears in **13,031** columns. The associated Row 0 values range from 1 to 52, matching the size of candidate filename block 2.
    - `[0, 0, 1]` appears in **10,742** columns. The associated Row 0 values range from 1 to 48, matching the size of candidate filename block 3.

### 3. `chan.mat`
- **File Format:** MATLAB v5.
- **Internal Structure:** A cell array (list) of MATLAB structures (dictionaries).
- **Shape/Dimensions:** Length 60.
- **Data Types:** Dictionaries containing mixed string/numeric fields (`labels`, `unit`, `sph_radius`, `sph_theta`, `sph_phi`, `theta`, `radius`, `X`, `Y`, `Z`, etc.).
- **Channel Information:** Contains 60 unique channel definitions. A sample reveals standard 10-20 system labels (e.g., 'Cz', 'FCz', 'FC1', 'CP1', 'Fz') along with their 3D spatial coordinates.

---

### Candidate Relationships Between the Files (Not Independently Verified)
- The one-hot patterns in `y_stim.mat` rows 1–3 align structurally with three configured ranges whose counts match the three filename blocks. This does not establish that the rows encode diagnosis or prove a trial-to-file join.
- The hypothesis that row-0 values index filename entries in `sub_name_stim.mat` is a candidate mapping only. The available metadata does not independently verify which participant, if any, each row/run represents.

### Candidate Interpretations; Semantics Unresolved
- **Filename blocks:** Prefixes are `controls`, `subtype1`, and `subtype2`; mapping these to participant diagnoses remains unverified.
- **Rows 1–3:** Their binary one-hot structure and alignment with configured ranges are observable; whether they are stimulus categories, diagnostic labels, or other metadata is unknown. They are not verified clinical targets.

### Unresolved Questions & Needed Evidence
1. **Stimulus/Task Information:** Despite the word "stim" appearing in both `y_stim.mat` and `sub_name_stim.mat`, there is **no data** in these files indicating what stimulus was presented during the 33,902 trials, nor the behavioral responses.
   - *Evidence Needed:* We need to locate another array (perhaps inside the individual `.mat` files listed in `sub_name_stim.mat` or an `X_stim.mat` file) to find stimulus codes.
2. **Channel Usage:** While `chan.mat` defines 60 standard channels, we cannot verify if the actual raw EEG recordings contain exactly 60 channels or if some were dropped. 
   - *Evidence Needed:* We need to check the dimensions of the raw EEG arrays to confirm they align with a `60 x time` shape.
