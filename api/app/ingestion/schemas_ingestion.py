"""Pydantic response schemas for the EEG ingestion API (Phase 2).

These schemas are completely independent of the Phase 1 model-inference
schemas in ``api/app/schemas.py``.  They describe the structure of responses
from the ``/api/v2/recordings`` endpoints.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class IngestionAPIModel(BaseModel):
    """Base model for all Phase 2 ingestion API responses."""

    model_config = ConfigDict(allow_inf_nan=False)


# ---------------------------------------------------------------------------
# Recording metadata
# ---------------------------------------------------------------------------


class RecordingMetadataResponse(IngestionAPIModel):
    """Metadata for an ingested EEG recording."""

    recording_id: str
    original_filename: str
    source_format: Literal["csv", "mat"]
    provenance: str
    n_channels: int
    n_samples: int
    n_trials: int | None
    sampling_rate_hz: float | None
    channel_names: list[str] | None
    channel_units: list[str] | None
    signal_units_known: bool
    observed_time_axis: str  # "0", "1", or "unknown"
    orientation_note: str
    data_shape: list[int]
    duration_seconds: float | None
    import_status: Literal["ok", "warnings", "rejected"]
    warnings: list[str]
    errors: list[str]
    memory_bytes: int
    storage_note: str = (
        "In-memory storage is temporary, process-local, and non-persistent.  "
        "Recordings are lost on server restart.  No user isolation is provided."
    )


class RecordingListItem(IngestionAPIModel):
    """Summary of one recording in the listing response."""

    recording_id: str
    original_filename: str
    source_format: Literal["csv", "mat"]
    provenance: str
    n_channels: int
    n_samples: int
    n_trials: int | None
    sampling_rate_hz: float | None
    import_status: Literal["ok", "warnings", "rejected"]
    memory_bytes: int


class RecordingListResponse(IngestionAPIModel):
    """Response for the recording list endpoint."""

    recordings: list[RecordingListItem]
    total_count: int
    total_memory_bytes: int
    storage_note: str = (
        "In-memory storage is temporary, process-local, and non-persistent."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ValidationResponse(IngestionAPIModel):
    """Validation status and issues for a recording."""

    recording_id: str
    import_status: Literal["ok", "warnings", "rejected"]
    warnings: list[str]
    errors: list[str]
    provenance: str
    source_metadata_keys: list[str]


# ---------------------------------------------------------------------------
# Signal quality
# ---------------------------------------------------------------------------


class ChannelStatsItem(IngestionAPIModel):
    """Per-channel quality statistics."""

    channel_index: int
    channel_name: str | None
    n_samples: int
    n_nonfinite: int
    signal_min: float | None
    signal_max: float | None
    signal_range: float | None
    signal_mean: float | None
    signal_std: float | None
    is_flatline: bool
    has_nonfinite: bool


class SignalQualityResponse(IngestionAPIModel):
    """Signal quality report for a recording."""

    recording_id: str
    trial_index: int | None
    n_channels: int
    n_samples: int
    sampling_rate_hz: float | None
    duration_seconds: float | None
    total_nonfinite_count: int
    flatline_channel_indices: list[int]
    channel_stats: list[ChannelStatsItem]
    warnings: list[str]


# ---------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------


class WaveformChannelResponse(IngestionAPIModel):
    """Waveform data for one channel."""

    channel_index: int
    channel_name: str | None
    time_values: list[float]
    amplitude_values: list[float]
    n_original_samples: int
    stride: int


class WaveformDataResponse(IngestionAPIModel):
    """Waveform visualization data for a recording."""

    recording_id: str
    trial_index: int | None
    time_axis_unit: Literal["seconds", "sample_index"]
    sampling_rate_hz: float | None
    time_axis_note: str
    downsampling_note: str
    channels: list[WaveformChannelResponse]


# ---------------------------------------------------------------------------
# PSD
# ---------------------------------------------------------------------------


class PSDChannelResponse(IngestionAPIModel):
    """PSD data for one channel."""

    channel_index: int
    channel_name: str | None
    frequencies_hz: list[float]
    power_density: list[float]
    welch_nperseg: int
    welch_noverlap: int
    welch_window: str


class PSDDataResponse(IngestionAPIModel):
    """Power spectral density response."""

    recording_id: str
    trial_index: int | None
    sampling_rate_hz: float
    method: str
    channels: list[PSDChannelResponse]
    assumptions: list[str]


# ---------------------------------------------------------------------------
# Channel statistics
# ---------------------------------------------------------------------------


class ChannelStatsSummaryResponse(IngestionAPIModel):
    """Per-channel amplitude summary for visualization."""

    recording_id: str
    n_channels: int
    n_samples: int
    sampling_rate_hz: float | None
    duration_seconds: float | None
    channel_names: list[str | None]
    signal_min: list[float | None]
    signal_max: list[float | None]
    signal_mean: list[float | None]
    signal_std: list[float | None]
    is_flatline: list[bool]


# ---------------------------------------------------------------------------
# MAT inspection
# ---------------------------------------------------------------------------


class MatInspectResponse(IngestionAPIModel):
    """MAT file structure inspection (pre-upload inspection)."""

    format: str
    variables: list[dict[str, Any]]
    numeric_candidates: list[str]
    note: str = (
        "Provide variable_name when uploading to select the EEG variable.  "
        "If numeric_candidates contains exactly one entry it will be "
        "auto-selected with a warning."
    )


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class IngestionErrorResponse(IngestionAPIModel):
    """Standard error response body for ingestion API errors."""

    detail: str
    error_type: str
