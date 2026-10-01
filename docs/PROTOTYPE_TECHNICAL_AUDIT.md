# Technical Audit: ADHD-EEG Research Prototype

**Audit Date:** October 1, 2026  
**Auditor:** Antigravity AI (Google DeepMind)  
**Git Branch:** `prototype/dummy-ml-integration`  
**Repository:** `ADHD-EEG-Research`  
**Scope:** Read-Only Technical Audit of Prototype & Dummy ML Integration  

---

## Executive Summary

This technical audit evaluates the prototype codebase on branch `prototype/dummy-ml-integration` of the `ADHD-EEG-Research` repository. The audit was conducted in a strictly **read-only** mode: no source files were modified, no dependencies were installed, no applications were launched, and no models were trained.

### Core Verdict
| Audit Item | Result | Details |
| :--- | :--- | :--- |
| **Actual Technology Stack** | **Streamlit / Python 3** | Pure Python prototype; no React, Node.js, or Vite present. |
| **Real Dataset Usage** | **NO** | Zero `.mat`, OSF, or real patient EEG data files are used by the prototype. |
| **Real Model Usage** | **NO** | Inference runs purely via deterministic mock heuristics; placeholder raises `NotImplementedError`. |
| **ML Contract Shape** | **`[385, 56]`, float32** | Input tensor shape and dtype are strictly validated. |
| **Output Schema Contract** | **VERIFIED** | Emits `class`, `class_index`, `probabilities`, and `confidence` alongside safety demo metadata. |
| **Unit Test Suite** | **PASS (9/9)** | `tests/test_model_interface.py` passes all contract validation tests. |
| **Safe to Run as Demo/Mock** | **YES** | Safe for local execution and presentation; no clinical leakage risk. |

---

## A. Frontend Stack

The inspection of repository configuration files confirms that **no JavaScript, TypeScript, Node.js, React, or Vite components exist** on this branch. The entire UI layer is implemented in Python using the Streamlit framework.

* **Framework:** [Streamlit](https://streamlit.io/) (v1.57.0 installed; specification requires `>=1.30.0` in `prototype/requirements.txt`).
* **Language:** Python 3 (Python 3.13.2 64-bit via Windows `py.exe`).
* **Build Tool:** None / Interpreted Python runtime. The Streamlit server manages the WebSocket communication and live reload mechanism without an ahead-of-time bundler.
* **Package Manager:** `pip` (Python package installer). Dependencies are enumerated in `prototype/requirements.txt`. No `package.json`, `package-lock.json`, `pnpm-lock.yaml`, or `yarn.lock` exists.
* **UI & Component Libraries:**
  * Streamlit built-in widgets: `st.file_uploader`, `st.sidebar.radio`, `st.selectbox`, `st.button`, `st.metric`, `st.dataframe`, `st.columns`, `st.info`, `st.warning`, `st.error`.
  * Plotly Graph Objects (`plotly.graph_objects`) for interactive signal time-series, multi-channel stacked montages, and probability gauges.
* **Styling Approach:** Streamlit native light theme with custom CSS injected via `st.markdown(..., unsafe_allow_html=True)`. Custom CSS classes include:
  * `.metric-card`: Structured stat boxes for shape, dtype, and architecture metrics.
  * `.disclaimer-banner`: Prominent amber warning banners for medical disclaimers.
  * `.demo-badge`: Amber badge tagging mock/demo inference status.
  * `.active-badge`: Green badge reserved for active real Transformer mode.
  * `.contract-box`: Blue callout box detailing parallel integration boundaries.
* **Routing:** Single-Page Application (SPA) state-based routing via `st.sidebar.radio` with 6 views:
  1. `1. Dashboard`
  2. `2. EEG Upload & Demo`
  3. `3. Signal Inspection`
  4. `4. Model Prediction`
  5. `5. Model Information`
  6. `6. About Project`
* **State Management:** Streamlit `st.session_state` storing session variables:
  * `eeg_dataframe`: Active EEG DataFrame (synthetic or uploaded).
  * `data_source`: Description of data provenance (e.g., `"Synthetic Demo Trial (Seed=42)"`).
  * `prediction_result`: Dictionary containing active classification output.
  * `input_validation_status`: Status string indicating contract compliance.
* **Dependencies & Versions (`prototype/requirements.txt` vs Environment):**
  | Dependency | Specified Version | Environment Version | Status |
  | :--- | :--- | :--- | :--- |
  | `streamlit` | `>=1.30.0` | `1.57.0` | Satisfied |
  | `numpy` | `>=1.24.0` | `2.2.6` | Satisfied |
  | `pandas` | `>=2.0.0` | `2.2.3` | Satisfied |
  | `plotly` | `>=5.18.0` | `6.7.0` | Satisfied |

---

## B. Backend / ML Stack

* **Backend Architecture:** There is **no separate backend microservice** (no FastAPI, Flask, Django, or Express). The prototype operates as a self-contained, in-process architecture where the Streamlit UI directly invokes Python library functions within the same memory space.
* **Language:** Python 3.
* **ML Framework / Model Implementation:**
  * Model interface abstraction: `ModelAdapter` class in `prototype/model/model_interface.py`.
  * Active model service: `MockEEGTransformerModel` in `prototype/model/mock_model.py`.
  * Future model connector: `trained_model_placeholder.py` (stub functions raising `NotImplementedError`).
* **Active ML Libraries:** `numpy` and `pandas` only. PyTorch, TensorFlow, Scikit-learn, ONNX, and MNE are **not imported or executed** in the prediction path.
* **API Communication:** In-process Python method dispatch:
  ```python
  ModelAdapter.predict_safe(eeg_input) -> Dict[str, Any]
  ```
* **Execution Status:** **Only returning mock/synthetic data.** The global switch `USE_REAL_MODEL` in `prototype/model/model_interface.py` (line 40) is set to `False`.

---

## C. Mock ML Implementation & Flow Trace

### Complete Inference Flow Trace

```
User Action / UI Trigger
   │
   ├─► Option A: Click "Reload Synthetic Demo Trial" / "Seed 42" / "Seed 100"
   ├─► Option B: Upload custom CSV trial via st.file_uploader
   └─► Option C: Click "Execute Model Prediction"
   │
   ▼
[prototype/app.py] load_demo_trial(seed) or direct event handler
   │
   ▼
[prototype/data/dummy_generator.py] generate_synthetic_eeg_dataframe(seed)
   │  - Synthesizes 385 timepoints × 56 channels via sinusoidal harmonics + normal noise
   │  - Returns pandas DataFrame with columns: ['timestamp', 'ch_1', ..., 'ch_56']
   │
   ▼
[prototype/model/model_interface.py] ModelAdapter.predict_safe(eeg_input)
   │  - Safe wrapper catching validation errors
   │  - Calls ModelAdapter.predict(eeg_input)
   │
   ▼
[prototype/model/model_interface.py] ModelAdapter.extract_eeg_array(eeg_input)
   │  - Strips non-signal columns ('timestamp', 'time')
   │  - Enforces 2D matrix shape
   │  - Casts array to float32
   │
   ▼
[prototype/model/model_interface.py] Shape Contract Validation
   │  - Validates arr.shape == (385, 56)
   │  - Raises ValueError on shape mismatch
   │  - Evaluates USE_REAL_MODEL (False) -> Dispatches to _MOCK_INSTANCE
   │
   ▼
[prototype/model/mock_model.py] MockEEGTransformerModel.predict(trial_data)
   │  - Calculates signal mean and standard deviation:
   │      signal_mean = np.mean(trial_data)
   │      signal_std  = np.std(trial_data)
   │  - Applies deterministic heuristic branch:
   │      std > 0.35   ──► ADHD Profile: {HC: 0.05, ADD: 0.10, ADHD: 0.85}
   │      mean < -0.05 ──► ADD Profile:  {HC: 0.15, ADD: 0.70, ADHD: 0.15}
   │      Default      ──► HC Profile:   {HC: 0.80, ADD: 0.12, ADHD: 0.08}
   │  - Normalizes probabilities: round(v / sum(probs), 4)
   │  - Synthesizes 56 optional channel importance weights
   │  - Formats output dictionary adhering to ML Contract
   │
   ▼
[prototype/app.py] View 4: UI Display
   │  - Large bold color-coded Class Card (ADHD: Red, HC: Green, ADD: Amber)
   │  - Model Confidence Percentage and Class Index
   │  - Status Badge: "DEMO / MOCK PREDICTION"
   │  - Disclaimer Text: "This is a synthetic mock prediction..."
   │  - Plotly Probability Distribution Horizontal Bar Gauge
   └─► Plotly 56-Channel Interpretability Bar Chart
```

### Verification of Input Contract
* **Target Input Shape:** `[385, 56]` (385 temporal samples $\times$ 56 electrode channels).
  * **Verified:** `prototype/model/mock_model.py` (line 24) defines `EXPECTED_SHAPE = (385, 56)` and raises `ValueError` if `trial_data.shape != EXPECTED_SHAPE`.
* **Target Data Type:** `float32`.
  * **Verified:** `prototype/model/mock_model.py` (line 25) defines `EXPECTED_DTYPE = np.float32`. Any input array is automatically cast to `np.float32` by `ModelAdapter.extract_eeg_array`.

### Verification of Output Contract Schema
The audit compared the requested schema against the actual returned payload:

```json
/* User Request Expected Schema */
{
  "class": "ADHD",
  "class_index": 2,
  "probabilities": {
    "HC": 0.05,
    "ADD": 0.10,
    "ADHD": 0.85
  },
  "confidence": 0.85
}
```

```json
/* Actual Implementation Output Schema (from ModelAdapter.predict_safe) */
{
  "class": "ADHD",
  "class_index": 2,
  "probabilities": {
    "HC": 0.05,
    "ADD": 0.10,
    "ADHD": 0.85
  },
  "confidence": 0.85,
  "is_demo": true,
  "status_label": "DEMO / MOCK PREDICTION",
  "disclaimer": "This is a synthetic mock prediction for prototype validation. Not a medical diagnosis.",
  "explanation": {
    "type": "Optional Synthetic Attention & Channel Importance",
    "channel_importance": [0.1, 0.45, 0.82, "...", 0.31],
    "note": "Optional demo field. Real Transformer may provide attention topomaps."
  },
  "status": "success"
}
```

**Verdict:** The actual output schema strictly supersets the required fields (`class`, `class_index`, `probabilities`, `confidence`), adding explicit safety flags (`is_demo`, `status_label`, `disclaimer`) and accommodating the optional explainability specification.

---

## D. Real Dataset Safety Audit

A systematic grep and semantic audit of all files in `prototype/`, `app.py`, `tests/`, and prototype documentation was conducted to inspect for unauthorized dataset references.

| Target Pattern / Item | Found in Prototype Code? | Reference Details |
| :--- | :--- | :--- |
| `.mat` files | **NO** | No `.mat` files are opened, loaded, or referenced. |
| `d1.mat` through `d7.mat` | **NO** | No raw trial batch files are accessed. |
| `y_stim` | **NO** | No stimulus onset markers are used. |
| `sub_name_stim` | **NO** | No subject identifiers are used. |
| `chan.mat` | **NO** | No channel metadata files are loaded. |
| OSF dataset | **NO** | No OSF client, API, or downloaded raw data used. |
| EEG raw files | **NO** | No BDF, EDF, or raw binary signals loaded. |
| NumPy loading of real EEG | **NO** | Only synthetic numpy tensors generated in memory. |
| MATLAB / HDF5 loading | **NO** | Neither `scipy.io` nor `h5py` is imported. |
| `h5py` | **NO** | Not imported anywhere in `prototype/`. |
| `scipy.io` | **NO** | Not imported anywhere in `prototype/`. |
| `mne` dataset loading | **NO** | MNE is not imported or required. |
| Real trained model / weights | **NO** | `trained_model_placeholder.py` raises `NotImplementedError`. |
| Real clinical diagnostic labels | **NO** | Only synthetic strings (`"HC"`, `"ADD"`, `"ADHD"`). |

> [!NOTE]
> In `prototype/model/trained_model_placeholder.py` (line 12), the string `eeg_transformer_best.pt` appears inside a docstring comment as an example for the ML team. No file exists with that name.
> In the repository root, `papers/d1.mat` exists as an untracked file from previous dataset exploration scripts (`scripts/dataset-investigation/`). The prototype code has **zero linkages, imports, or dependencies** on this file or directory.

### Audit Result
$$\mathbf{REAL\ DATASET\ USED = NO}$$

---

## E. Dummy Data & Mock Behavior Audit

1. **Generation Location:**
   * `prototype/data/dummy_generator.py`: Contains `generate_synthetic_eeg_trial()` and `generate_synthetic_eeg_dataframe()`.
   * `prototype/data/sample_eeg.csv`: A static pre-generated CSV containing 385 rows and 57 columns (`timestamp` + 56 channels) matching the synthetic generator's output with `seed=42`.
2. **Random vs. Deterministic:**
   * **100% Deterministic:** Signal generation uses `np.random.default_rng(seed)`. When invoked with fixed seeds (`seed=42` or `seed=100`), identical waveforms and numerical values are produced every time.
3. **Hardcoded vs. Algorithmic Predictions:**
   * Predictions are **not** static constants. They are generated via deterministic rule-based heuristics calculated from the input signal's summary statistics:
     * Standard deviation $> 0.35 \implies$ Class: `ADHD`, Probabilities: `[HC: 0.05, ADD: 0.10, ADHD: 0.85]`.
     * Mean $< -0.05 \implies$ Class: `ADD`, Probabilities: `[HC: 0.15, ADD: 0.70, ADHD: 0.15]`.
     * Otherwise $\implies$ Class: `HC`, Probabilities: `[HC: 0.80, ADD: 0.12, ADHD: 0.08]`.
   * Probabilities are normalized via `round(v / sum(probs), 4)`.
4. **UI Labeling & Clinical Safeguards:**
   * **Persistent Disclaimers:** Every page features a prominent warning banner stating that the system is a research prototype and outputs are not clinical diagnoses.
   * **Visual Badging:** A yellow badge labeled `DEMO / MOCK PREDICTION` is permanently displayed in the sidebar and directly beneath the predicted class box.
   * **Clinical Confusion Risk:** Minimal. The mock status is clear across all six views.

---

## F. Testing Audit

The test suite was inspected and executed.

* **Test File:** `tests/test_model_interface.py`
* **Framework:** Standard Library `unittest`
* **Execution Status:** **PASS** (Ran 9 tests in 0.018s with exit code 0).
* **Detailed Test Breakdown:**

| Test Method | Target Contract Verification | Result |
| :--- | :--- | :--- |
| `test_1_valid_contract_input_accepted` | Verifies `[385, 56]` `float32` synthetic array executes successfully. | **PASS** |
| `test_2_wrong_shape_rejected` | Tests bad shapes (`[100, 56]`, `[385, 16]`, `[385, 64]`, `[385,]`, `[1, 385, 56]`) and verifies `ValueError`. | **PASS** |
| `test_3_wrong_dtype_handled_and_cast` | Verifies `float64` input is properly cast to `float32` without crashing. | **PASS** |
| `test_4_probabilities_are_valid_range` | Verifies all class probabilities lie in the valid interval $[0.0, 1.0]$. | **PASS** |
| `test_5_probabilities_sum_to_approximately_one` | Verifies $\sum P_i \approx 1.0 \pm 0.01$. | **PASS** |
| `test_6_predicted_class_matches_highest_probability` | Verifies `response["class"] == argmax(probabilities)`. | **PASS** |
| `test_7_demo_flag_is_present` | Verifies `is_demo == True` and status label contains `"DEMO"`. | **PASS** |
| `test_8_missing_optional_explanation_does_not_break` | Verifies inference succeeds and maintains schema when explainability is disabled. | **PASS** |
| `test_9_safe_wrapper_handles_errors_gracefully` | Verifies `ModelAdapter.predict_safe` returns structured error dict without unhandled exceptions. | **PASS** |

---

## G. Git Audit

* **Current Branch:** `prototype/dummy-ml-integration`
* **Parent / Base Branch:** `main` (branch point commit: `deaea30` *"Add base-paper-analysis"*).
* **Commits on this Branch:**
  1. `d658832`: *"Define ML model integration contract"*
  2. `47501ac`: *"Build prototype with dummy EEG workflow"*
* **Files Added Relative to `main` (15 files):**
  * `app.py` (Root launcher)
  * `docs/PROTOTYPE_ML_INTERFACE.md` (Integration contract)
  * `prototype/app.py` (Streamlit multi-view application)
  * `prototype/assets/logo.png` (Static asset)
  * `prototype/data/__init__.py`
  * `prototype/data/dummy_generator.py` (Synthetic EEG trial generator)
  * `prototype/data/sample_eeg.csv` (Pre-generated synthetic demo trial)
  * `prototype/model/__init__.py`
  * `prototype/model/mock_model.py` (Mock model engine)
  * `prototype/model/model_interface.py` (Model adapter boundary)
  * `prototype/model/trained_model_placeholder.py` (Real model stub)
  * `prototype/requirements.txt` (Prototype dependencies)
  * `prototype/utils/__init__.py`
  * `prototype/utils/visualization.py` (Plotly visualizers)
  * `tests/test_model_interface.py` (Contract integration test suite)
* **Files Modified Relative to `main`:** 0 committed files modified. (1 uncommitted file in working copy: `docs/README.md` adding links).
* **Files Deleted Relative to `main`:** 0.
* **Accidental Dataset Commits:** **NONE.** No `.mat`, `.pt`, `.h5`, or patient files are committed.

---

## H. Running Requirements

To run this prototype locally, follow the steps below based on the project structure and installed environment:

### 1. Prerequisites
* Python 3.10, 3.11, 3.12, or 3.13.
* Existing virtual environment or system Python with required packages.

### 2. Dependency Installation Command
```powershell
pip install -r prototype/requirements.txt
```
*(On Windows using the Python Launcher: `py -m pip install -r prototype/requirements.txt`)*

### 3. Application Launch Commands
The application can be launched from the repository root using either of the following commands:

* **Recommended (via Root Launcher):**
  ```powershell
  streamlit run app.py
  ```
  *(Or: `py -m streamlit run app.py`)*

* **Direct (Targeting Prototype Directory):**
  ```powershell
  streamlit run prototype/app.py
  ```
  *(Or: `py -m streamlit run prototype/app.py`)*

### 4. Test Suite Execution Command
```powershell
py -m unittest tests/test_model_interface.py
```

---

## I. Audit Verdict

1. **Actual Technical Stack:** Python 3 + Streamlit + Plotly + Pandas + NumPy. No Node/React stack.
2. **Prototype Architecture:** Self-contained, modular in-process Python application decoupled through a clean adapter boundary (`ModelAdapter`).
3. **Files Involved:** 15 tracked prototype files across `prototype/`, `app.py`, `tests/`, and `docs/`.
4. **Mock ML Flow:** User input $\to$ `ModelAdapter.predict_safe()` $\to$ `MockEEGTransformerModel.predict()` $\to$ deterministic rule-based output $\to$ Streamlit UI with Plotly charts.
5. **Real Dataset Usage:** **NO**
6. **Real Model Usage:** **NO**
7. **Existing Tests:** 9 unit tests in `tests/test_model_interface.py` (All passing).
8. **Required Run Commands:** `streamlit run app.py` or `py -m streamlit run app.py`.
9. **Risks / Issues Found:**
   * **Minor UX Consideration:** In View 2 (*EEG Upload & Demo*), if a user uploads an arbitrary CSV trial matching `[385, 56]`, the mock heuristics will compute a synthetic prediction. While disclaimers and badges are visible, users should be reminded in documentation that uploaded files are evaluated by mock statistics, not a clinical model.
   * **Python PATH Configuration:** On some Windows command shells, `python` may not be directly aliased in PATH; using `py` (the official Windows Python launcher) avoids invocation errors.
10. **Safety Assessment:** **SAFE TO RUN AS A MOCK / DEMO.** The prototype satisfies all clinical isolation safeguards, contains no proprietary or sensitive EEG recordings, and strictly enforces the interface contract.
11. **Prerequisites Before Running:** None. All required dependencies (`streamlit`, `numpy`, `pandas`, `plotly`) are already installed and functional.
