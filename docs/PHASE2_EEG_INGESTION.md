# Phase 2: EEG Ingestion and Visualization Backend

**Drushti Detect — ADHD-EEG-Research**
**Branch:** `feature/drushti-detect-platform`
**Status:** Implemented and verified

---

## Overview

Phase 2 adds format-agnostic EEG file ingestion, a canonical recording
representation, signal quality inspection, and EEG visualization data
generation to the existing FastAPI backend.  Phase 1 model inference is
completely unchanged.

All Phase 2 code lives in `api/app/ingestion/`.  New endpoints are mounted
under `/api/v2/recordings`.

---

## Architecture

```
api/
└── app/
    └── ingestion/
        ├── __init__.py
        ├── recording.py          # EEGRecording canonical representation
        ├── csv_reader.py         # CSV ingestion
        ├── mat_reader.py         # MAT v5 / v7.3 HDF5 ingestion
        ├── inspection.py         # Signal quality inspection
        ├── visualization.py      # Waveform, PSD, and stats data generation
        ├── schemas_ingestion.py  # Pydantic response schemas
        └── routes.py             # FastAPI router

api/tests/ingestion/
├── __init__.py
├── conftest.py                   # Shared synthetic fixtures
├── test_csv_reader.py
├── test_mat_reader.py
├── test_inspection.py
├── test_visualization.py
└── test_routes_ingestion.py
```

---

## Canonical Recording Representation

All readers produce an `EEGRecording` dataclass with:

| Field | Description |
|-------|-------------|
| `recording_id` | UUID assigned at ingestion |
| `source_format` | `"csv"` or `"mat"` |
| `provenance` | `"csv"`, `"mat_v5"`, or `"mat_v7.3_hdf5"` |
| `array` | `[n_samples, n_channels]` float64 — canonical 2D form |
| `array_3d` | `[n_trials, n_samples, n_channels]` — full 3D for epoched data |
| `n_trials` | `None` for continuous; integer for epoched recordings |
| `sampling_rate_hz` | Hz if known; `None` otherwise |
| `channel_names` | Extracted from CSV header; `None` for MAT |
| `signal_units_known` | Always `False` in Phase 2 (not in source files) |
| `observed_time_axis` | Axis index (0 or 1) or `"unknown"` |
| `orientation_note` | Human-readable explanation of orientation decision |
| `source_metadata` | All file-level metadata, unmodified |
| `import_status` | `"ok"`, `"warnings"`, or `"rejected"` |
| `warnings` | Non-fatal issues |
| `errors` | Fatal issues (only when `import_status == "rejected"`) |

### Key invariants

- `array` is always `[n_samples, n_channels]`
- For 3D recordings, `array` is a **view** of `array_3d[0]`
- Use `EEGRecording.get_trial_view(trial_index)` to access any trial safely
- Signal data is **never modified** — no filtering, resampling, or normalisation

---

## Supported File Formats

### CSV

- UTF-8 (with or without BOM)
- Optional single non-numeric header row (auto-detected)
- One row per time sample; one column per EEG channel
- Minimum 2 data rows required
- All values must be finite real numbers; NaN / Inf → error
- Sampling rate supplied as an explicit parameter (not in file)
- Units are always unknown (`signal_units_known = False`)

**Limitations:**
- No time-column support (first numeric column treated as EEG data)
- No automatic resampling

### MATLAB .mat v5

- Loaded via `scipy.io.loadmat`
- Format detected from file magic bytes, not filename
- Variable auto-selection when exactly one large numeric array is present
- Ambiguous (multiple large numeric arrays) → `AmbiguousStructureError`; user must supply `variable_name`
- 1D arrays reshaped to `[n_samples, 1]` with warning
- 3D arrays preserved fully in `array_3d`; orientation inferred heuristically
- Cell arrays, nested structs, function handles: not supported as the primary EEG variable

### MATLAB .mat v7.3 (HDF5)

- Loaded via `h5py`
- Format detected from HDF5 magic bytes (`\x89HDF\r\n\x1a\n`)
- **HDF5 / MATLAB axis convention documented explicitly:**
  - h5py reads arrays in C order (row-major)
  - MATLAB writes in Fortran/column-major order
  - Axis order is **reversed** between MATLAB and h5py
  - Both shapes (`h5py_shape` and `matlab_inferred_shape`) are recorded in `source_metadata`
  - **No blind transposition** — orientation is flagged as unknown when it cannot be established confidently
  - User can supply `time_axis` and `channel_axis` explicitly to override heuristics

#### Heuristic orientation rules (documented assumptions, not facts)

| Array rank | Rule |
|------------|------|
| 2D, rows > cols | Axis 0 = samples, Axis 1 = channels |
| 2D, cols > rows | Axis 1 = samples, Axis 0 = channels |
| 2D, square | Ambiguous → `observed_time_axis = "unknown"` |
| 3D | Smallest dim (≤ 512) = channels; largest remaining = trials; middle = samples |
| 3D, all dims > 512 | Ambiguous → `observed_time_axis = "unknown"` |

All heuristics emit a warning and are documented in `orientation_note`.

---

## API Endpoints

All endpoints are under `/api/v2/recordings`.

### Upload

#### `POST /api/v2/recordings/upload`

Upload a CSV or MAT EEG recording.

**Form parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | ✅ | `.csv` or `.mat` file |
| `sampling_rate_hz` | float | ❌ | Sampling rate in Hz. Required for time-axis calculations. |
| `variable_name` | string | ❌ | MAT variable to use. Required if file has multiple candidates. |
| `time_axis` | int | ❌ | Explicit time axis index. Overrides heuristic. |
| `channel_axis` | int | ❌ | Explicit channel axis index. Overrides heuristic. |

**Response:** `RecordingMetadataResponse` (200)

**Errors:**
- `413` — file exceeds `INGESTION_MAX_UPLOAD_SIZE_BYTES` limit
- `415` — unsupported file type or invalid MAT magic bytes
- `422` — parse error, non-numeric data, ambiguous MAT structure
- `507` — in-memory recording count or size limit reached

---

#### `POST /api/v2/recordings/inspect-mat`

Inspect a MAT file structure without storing a recording. Use to determine which `variable_name` to pass to the upload endpoint.

**Response:** `MatInspectResponse` — lists all variables and numeric candidates.

---

### Retrieval

#### `GET /api/v2/recordings`

List all recordings held in memory.

**Response:** `RecordingListResponse`

---

#### `GET /api/v2/recordings/{id}`

Full metadata for a specific recording.

**Response:** `RecordingMetadataResponse`

---

#### `GET /api/v2/recordings/{id}/validation`

Import status, warnings, and errors for a recording.

**Response:** `ValidationResponse`

---

#### `GET /api/v2/recordings/{id}/quality`

Per-channel signal quality statistics.

**Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trial_index` | int | `0` | Trial to inspect (3D recordings only) |

**Response:** `SignalQualityResponse`

---

#### `GET /api/v2/recordings/{id}/waveform`

Waveform display data. Returns actual uploaded samples (optionally downsampled for display).

**Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `channels` | string | all | Comma-separated 0-based channel indices, e.g. `"0,2,4"` |
| `start_s` | float | `0` | Start time in seconds (requires known sampling rate) |
| `end_s` | float | duration | End time in seconds |
| `start_sample` | int | `0` | Start sample index (alternative to seconds) |
| `end_sample` | int | n_samples | End sample index |
| `max_points` | int | `2000` | Maximum display points per channel |
| `trial_index` | int | `0` | Trial to visualize (3D recordings only) |

**Response:** `WaveformDataResponse`

**Downsampling note:** When the requested window exceeds `max_points`, data is
stride-subsampled for display efficiency.  The original signal is preserved
unchanged.  The stride and original sample count are reported in the response.

**Time axis:** seconds when `sampling_rate_hz` is known; sample indices otherwise.
Both cases are explicitly documented in `time_axis_unit` and `time_axis_note`.

---

#### `GET /api/v2/recordings/{id}/stats`

Per-channel amplitude statistics (min, max, mean, std, flatline flag).

**Query parameters:** `trial_index` (default `0`)

**Response:** `ChannelStatsSummaryResponse`

---

#### `GET /api/v2/recordings/{id}/psd`

Welch power spectral density. Requires a known sampling rate.

**Query parameters:** `channels`, `trial_index`

**Response:** `PSDDataResponse` — includes Welch parameters (`nperseg`, `noverlap`, `window`) and
a list of explicit computational assumptions.

---

## Signal Quality Inspection

Computed from actual uploaded samples without any modification:

- Per-channel: min, max, range, mean, std
- Non-finite count (NaN, ±Inf) per channel
- Flatline detection (peak-to-peak range ≤ 1e-10)
- Duration in seconds (only when sampling rate is known)

Flatline channels are **not automatically rejected**. The report is informational; the Phase 3 model adapter will apply its own preprocessing contract.

---

## Configuration

All limits are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `INGESTION_MAX_UPLOAD_SIZE_BYTES` | 100 MB | Per-upload file size limit |
| `MAX_RECORDINGS_IN_MEMORY` | 10 | Maximum concurrent recordings |
| `MAX_TOTAL_RECORDING_MEMORY_BYTES` | 512 MB | Total in-memory footprint limit |
| `MAX_WAVEFORM_DISPLAY_POINTS` | 2000 | Max display points per channel in waveform responses |

---

## In-Memory Storage

> **This storage is temporary, process-local, and non-persistent.**
>
> - Recordings are lost when the server process restarts.
> - There is no user isolation — all recordings in a process are accessible to any client.
> - Maximum recordings and total memory usage are configurable.
> - Phase 4 will replace this with MongoDB + object storage.

---

## Phase 3 Handoff Contract

The `EEGRecording` dataclass is designed to be consumed directly by the Phase 3 model adapter:

- `array` → `[n_samples, n_channels]` float64 array for model preprocessing
- `sampling_rate_hz` → may be `None`; the adapter must validate this
- `channel_names` → may be `None`; the adapter must not assume any ordering
- `n_trials` → `None` for continuous; adapter must handle both
- `import_status` → adapter should reject `"rejected"` recordings
- `warnings` → adapter may add additional warnings

The Phase 3 adapter must **not** import any module from `api.app.ingestion` directly into `src/inference.py` or `src/preprocessing.py`.  The adapter should live in a new `api/app/model_adapter/` package.

---

## Verification

### Test results

```
154 passed, 0 failed — api/tests/ingestion/
15 passed, 0 failed  — api/tests/ (Phase 1 regression, all existing tests)
Total: 169 tests, 0 failures
```

### Test coverage areas

| Module | Tests | Synthetic data only |
|--------|-------|---------------------|
| `csv_reader.py` | 26 | ✅ |
| `mat_reader.py` | 38 | ✅ |
| `inspection.py` | 24 | ✅ |
| `visualization.py` | 32 | ✅ |
| `routes.py` (integration) | 34 | ✅ |
| Phase 1 regression | 15 | — |

---

## Constraints Honoured

- ✅ All Phase 2 code is inside `api/app/ingestion/`
- ✅ No imports from `src.inference`, `src.preprocessing`, or `src.ui_service`
- ✅ Existing `/api/v1/analyze` endpoint unchanged and passing all tests
- ✅ Actual uploaded signal samples used for visualization
- ✅ No automatic filtering, resampling, normalisation, or artifact removal
- ✅ Source metadata and raw data preserved
- ✅ Synthetic data only in automated tests
- ✅ MAT 3D arrays preserved in full (`array_3d`)
- ✅ HDF5 orientation not blindly transposed — documented in `source_metadata`
- ✅ When orientation is unknown, `observed_time_axis = "unknown"` is set
- ✅ In-memory storage limits configurable via environment variables
