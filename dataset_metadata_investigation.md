# Dataset Metadata Investigation Report

Based on an independent script-based inspection of the raw `.mat` files using Python (`h5py` and `scipy.io`), here is the factual investigation report of the dataset's metadata.

### 1. `sub_name_stim.mat`
- **File Format:** MATLAB v7.3 (HDF5-based).
- **Internal Structure:** A cell array containing 3 nested cell arrays of strings. 
- **Shape/Dimensions:** The top-level array shape is `(1, 3)`. The inner arrays have lengths of `(1, 44)`, `(1, 52)`, and `(1, 48)`.
- **Data Types:** HDF5 Object References pointing to 16-bit character arrays (strings).
- **What Each Dimension Represents:**
  - **Dimension 1 (Length 3):** Represents 3 distinct subject groups.
  - **Dimension 2:** Represents individual strings containing the original file names for the subjects in that group.
- **Unique Values and Subject/Group Information:** 
  - **Group 1:** Contains 44 filenames. All filenames begin with the prefix `'controls'` (e.g., `'controls2MSCT_Time_Estimation_NF_VP_02C.mat'`).
  - **Group 2:** Contains 52 filenames. All filenames begin with the prefix `'subtype1'` (e.g., `'subtype1AB237_NF_pr.mat'`).
  - **Group 3:** Contains 48 filenames. All filenames begin with the prefix `'subtype2'` (e.g., `'subtype2AE231_ET_pr.mat'`).
  - Total verified subjects across all groups: **144**.

### 2. `y_stim.mat`
- **File Format:** MATLAB v7.3 (HDF5-based).
- **Internal Structure:** A single 2D numeric matrix named `y_stim`.
- **Shape/Dimensions:** `(4, 33902)`.
- **Data Types:** `float64`.
- **What Each Dimension Represents:**
  - **Columns (33,902):** Correspond to individual observations/events—almost certainly EEG trials or epochs.
  - **Rows (4):** Contain metadata for each trial. 
    - **Row 0:** Contains integer values ranging from 1.0 to 52.0. This represents a 1-based **subject index** relative to the subject's group.
    - **Rows 1, 2, 3:** Contain strictly binary values (`0.0` or `1.0`).
- **Unique Values and Group Info:** 
  - An analysis of Rows 1-3 reveals exactly three mutually exclusive combinations (one-hot encoding):
    - `[1, 0, 0]` appears in **10,129** trials. The associated Row 0 values range from 1 to 44 (exactly matching the 44 subjects in Group 1).
    - `[0, 1, 0]` appears in **13,031** trials. The associated Row 0 values range from 1 to 52 (exactly matching the 52 subjects in Group 2).
    - `[0, 0, 1]` appears in **10,742** trials. The associated Row 0 values range from 1 to 48 (exactly matching the 48 subjects in Group 3).

### 3. `chan.mat`
- **File Format:** MATLAB v5.
- **Internal Structure:** A cell array (list) of MATLAB structures (dictionaries).
- **Shape/Dimensions:** Length 60.
- **Data Types:** Dictionaries containing mixed string/numeric fields (`labels`, `unit`, `sph_radius`, `sph_theta`, `sph_phi`, `theta`, `radius`, `X`, `Y`, `Z`, etc.).
- **Channel Information:** Contains 60 unique channel definitions. A sample reveals standard 10-20 system labels (e.g., 'Cz', 'FCz', 'FC1', 'CP1', 'Fz') along with their 3D spatial coordinates.

---

### Verified Relationships Between the Files
- `sub_name_stim.mat` and `y_stim.mat` are inextricably linked. The one-hot encoded rows (Rows 1-3) in `y_stim.mat` perfectly map to the 3 cell arrays in `sub_name_stim.mat`.
- The values in `y_stim.mat` Row 0 act as array indices (1-based) pointing to the specific filename stored in `sub_name_stim.mat`. To determine which subject a trial belongs to, you look at which row (1-3) is active to determine the group, and then use Row 0 to index into that group's filename array.

### Strongly Supported Interpretations
- **Groups:** Based solely on the strings inside `sub_name_stim.mat`, the three groups represent "Controls", "Subtype 1", and "Subtype 2".
- **Labels:** The dataset labels are structured at the *group* level rather than the *stimulus* level. The target variable `y` designates which clinical group the trial originated from.

### Unresolved Questions & Needed Evidence
1. **Stimulus/Task Information:** Despite the word "stim" appearing in both `y_stim.mat` and `sub_name_stim.mat`, there is **no data** in these files indicating what stimulus was presented during the 33,902 trials, nor the behavioral responses.
   - *Evidence Needed:* We need to locate another array (perhaps inside the individual `.mat` files listed in `sub_name_stim.mat` or an `X_stim.mat` file) to find stimulus codes.
2. **Channel Usage:** While `chan.mat` defines 60 standard channels, we cannot verify if the actual raw EEG recordings contain exactly 60 channels or if some were dropped. 
   - *Evidence Needed:* We need to check the dimensions of the raw EEG arrays to confirm they align with a `60 x time` shape.
