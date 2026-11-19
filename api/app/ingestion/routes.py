"""FastAPI router for EEG ingestion and visualization — Phase 2.

All endpoints are mounted under ``/api/v2/recordings``.

Endpoints
---------
POST   /api/v2/recordings/upload          Upload CSV or MAT recording
POST   /api/v2/recordings/inspect-mat     Inspect MAT structure (pre-upload)
GET    /api/v2/recordings                 List all in-memory recordings
GET    /api/v2/recordings/{id}            Get recording metadata
GET    /api/v2/recordings/{id}/validation Validation status and warnings
GET    /api/v2/recordings/{id}/quality    Signal quality report
GET    /api/v2/recordings/{id}/waveform   Waveform data
GET    /api/v2/recordings/{id}/stats      Per-channel statistics
GET    /api/v2/recordings/{id}/psd        Power spectral density

Storage note
------------
Recordings are stored in ``app.state.recordings`` (a plain Python dict).
This is temporary, process-local, and non-persistent storage intended only
for Phase 2.  Phase 4 will replace this with MongoDB + object storage.
No user isolation is provided.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from .csv_reader import read_csv
from .inspection import inspect_recording
from .mat_reader import (
    AmbiguousStructureError,
    IngestionError,
    UnsupportedFormatError,
    inspect_mat_structure,
    read_mat,
)
from .recording import EEGRecording
from .schemas_ingestion import (
    ChannelStatsSummaryResponse,
    IngestionErrorResponse,
    MatInspectResponse,
    PSDChannelResponse,
    PSDDataResponse,
    RecordingListItem,
    RecordingListResponse,
    RecordingMetadataResponse,
    SignalQualityResponse,
    ChannelStatsItem,
    ValidationResponse,
    WaveformChannelResponse,
    WaveformDataResponse,
)
from .visualization import get_channel_stats, get_psd, get_waveform

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/recordings", tags=["EEG Ingestion (Phase 2)"])

# ---------------------------------------------------------------------------
# Supported upload extensions (validated in addition to content inspection).
# ---------------------------------------------------------------------------
_ALLOWED_EXTENSIONS = {".csv", ".mat"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_recording(request: Request, recording_id: str) -> EEGRecording:
    """Retrieve a recording by ID or raise 404."""
    store: dict = getattr(request.app.state, "recordings", {})
    recording = store.get(recording_id)
    if recording is None:
        raise HTTPException(
            status_code=404,
            detail=f"Recording '{recording_id}' not found.",
        )
    return recording


def _recording_to_metadata(rec: EEGRecording) -> RecordingMetadataResponse:
    """Convert an EEGRecording to its metadata response schema."""
    return RecordingMetadataResponse(
        recording_id=rec.recording_id,
        original_filename=rec.original_filename,
        source_format=rec.source_format,
        provenance=rec.provenance,
        n_channels=rec.n_channels,
        n_samples=rec.n_samples,
        n_trials=rec.n_trials,
        sampling_rate_hz=rec.sampling_rate_hz,
        channel_names=rec.channel_names,
        channel_units=rec.channel_units,
        signal_units_known=rec.signal_units_known,
        observed_time_axis=str(rec.observed_time_axis),
        orientation_note=rec.orientation_note,
        data_shape=list(rec.data_shape),
        duration_seconds=rec.duration_seconds,
        import_status=rec.import_status,
        warnings=rec.warnings,
        errors=rec.errors,
        memory_bytes=rec.memory_bytes,
    )


def _check_store_limits(request: Request, incoming_bytes: int) -> None:
    """Raise 507 if adding this recording would exceed configured limits."""
    settings = getattr(request.app.state, "settings", None)
    store: dict = getattr(request.app.state, "recordings", {})

    if settings is not None:
        max_count = getattr(settings, "max_recordings_in_memory", 10)
        max_total_bytes = getattr(settings, "max_total_recording_memory_bytes", 512 * 1024 * 1024)

        if len(store) >= max_count:
            raise HTTPException(
                status_code=507,
                detail=(
                    f"Maximum number of in-memory recordings ({max_count}) has been "
                    "reached.  No further uploads are accepted until the server "
                    "is restarted.  Phase 4 will add persistent storage."
                ),
            )
        current_bytes = sum(r.memory_bytes for r in store.values())
        if current_bytes + incoming_bytes > max_total_bytes:
            raise HTTPException(
                status_code=507,
                detail=(
                    f"Storing this recording would exceed the total in-memory limit "
                    f"({max_total_bytes // (1024 * 1024)} MB).  "
                    "Phase 4 will add persistent storage."
                ),
            )


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/upload",
    response_model=RecordingMetadataResponse,
    summary="Upload a CSV or MAT EEG recording",
    responses={
        413: {"description": "File too large"},
        415: {"description": "Unsupported file type"},
        422: {"description": "Ingestion error (invalid data, ambiguous structure, etc.)"},
        507: {"description": "In-memory storage limit reached"},
    },
)
async def upload_recording(
    request: Request,
    file: UploadFile = File(..., description="CSV or MAT EEG file"),
    sampling_rate_hz: float | None = Form(
        None,
        description="Sampling rate in Hz.  Required for time-axis calculations.",
    ),
    variable_name: str | None = Form(
        None,
        description="MAT variable name to use as EEG data (MAT files only).  "
                    "Required when the MAT file contains multiple candidate variables.",
    ),
    time_axis: int | None = Form(
        None,
        description="Explicit time axis index (0-based) for MAT files.  "
                    "When omitted, orientation is inferred heuristically.",
    ),
    channel_axis: int | None = Form(
        None,
        description="Explicit channel axis index (0-based) for MAT files.",
    ),
):
    """Upload a CSV or MAT EEG recording.

    The recording is validated, ingested into the canonical EEGRecording
    representation, and stored in process memory.

    Returns recording metadata including import status, warnings, and
    orientation notes.  Signal data can be retrieved via the waveform,
    quality, and stats endpoints.

    **Storage note**: In-memory storage is temporary, process-local, and
    non-persistent.  Recordings are lost on server restart.
    """
    settings = getattr(request.app.state, "settings", None)
    max_upload = getattr(settings, "ingestion_max_upload_size_bytes", 100 * 1024 * 1024)

    # Validate filename extension before reading bytes.
    raw_name = file.filename or ""
    suffix = Path(raw_name).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type {suffix!r}.  "
                f"Supported types: {sorted(_ALLOWED_EXTENSIONS)}"
            ),
        )

    # Read upload with size enforcement.
    content = await file.read(max_upload + 1)
    if len(content) > max_upload:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Upload exceeds the configured limit of "
                f"{max_upload // (1024 * 1024)} MB."
            ),
        )
    await file.close()

    
   

    # Parse in a thread pool to avoid blocking the event loop.
    try:
        if suffix == ".csv":
            recording = await run_in_threadpool(
                read_csv,
                content,
                raw_name,
                sampling_rate_hz,
            )
        else:
            recording = await run_in_threadpool(
                read_mat,
                content,
                raw_name,
                sampling_rate_hz,
                variable_name,
                time_axis,
                channel_axis,
            )
    except AmbiguousStructureError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.error("Unexpected ingestion error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during ingestion.",
        ) from exc

    # Store recording.
    # Validate actual decoded memory and store atomically.
    store: dict = request.app.state.recordings
    store_lock = request.app.state.recordings_lock
    async with store_lock:
        _check_store_limits(request, recording.memory_bytes)
        store[recording.recording_id] = recording
    LOGGER.info(
        "Recording stored: id=%s, file=%s, status=%s",
        recording.recording_id,
        recording.original_filename,
        recording.import_status,
    )

    return _recording_to_metadata(recording)


# ---------------------------------------------------------------------------
# MAT inspection endpoint (pre-upload)
# ---------------------------------------------------------------------------


@router.post(
    "/inspect-mat",
    response_model=MatInspectResponse,
    summary="Inspect a MAT file structure without committing an upload",
)
async def inspect_mat(
    request: Request,
    file: UploadFile = File(..., description="MATLAB .mat file to inspect"),
):
    """Return a summary of MAT file variables and numeric candidates.

    Use this endpoint to determine which variable to pass as ``variable_name``
    to the upload endpoint when the file contains multiple numeric arrays.

    Does NOT store the recording.
    """
    settings = getattr(request.app.state, "settings", None)
    max_upload = getattr(settings, "ingestion_max_upload_size_bytes", 100 * 1024 * 1024)

    raw_name = file.filename or ""
    suffix = Path(raw_name).suffix.lower()
    if suffix != ".mat":
        raise HTTPException(
            status_code=415,
            detail="This endpoint only accepts .mat files.",
        )

    content = await file.read(max_upload + 1)
    if len(content) > max_upload:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the configured limit of {max_upload // (1024 * 1024)} MB.",
        )
    await file.close()

    try:
        result = await run_in_threadpool(inspect_mat_structure, content)
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.error("MAT inspection error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during MAT inspection.",
        ) from exc

    return MatInspectResponse(**result)


# ---------------------------------------------------------------------------
# Listing endpoint
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=RecordingListResponse,
    summary="List all in-memory recordings",
)
async def list_recordings(request: Request):
    """Return a summary list of all recordings currently held in memory."""
    store: dict = getattr(request.app.state, "recordings", {})
    items = [
        RecordingListItem(
            recording_id=r.recording_id,
            original_filename=r.original_filename,
            source_format=r.source_format,
            provenance=r.provenance,
            n_channels=r.n_channels,
            n_samples=r.n_samples,
            n_trials=r.n_trials,
            sampling_rate_hz=r.sampling_rate_hz,
            import_status=r.import_status,
            memory_bytes=r.memory_bytes,
        )
        for r in store.values()
    ]
    return RecordingListResponse(
        recordings=items,
        total_count=len(items),
        total_memory_bytes=sum(i.memory_bytes for i in items),
    )


# ---------------------------------------------------------------------------
# Recording metadata endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/{recording_id}",
    response_model=RecordingMetadataResponse,
    summary="Get recording metadata",
)
async def get_recording_metadata(request: Request, recording_id: str):
    """Return full metadata for a specific recording."""
    return _recording_to_metadata(_get_recording(request, recording_id))


# ---------------------------------------------------------------------------
# Validation endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/{recording_id}/validation",
    response_model=ValidationResponse,
    summary="Get validation results for a recording",
)
async def get_validation(request: Request, recording_id: str):
    """Return import status, warnings, and errors for a recording."""
    rec = _get_recording(request, recording_id)
    return ValidationResponse(
        recording_id=rec.recording_id,
        import_status=rec.import_status,
        warnings=rec.warnings,
        errors=rec.errors,
        provenance=rec.provenance,
        source_metadata_keys=list(rec.source_metadata.keys()),
    )


# ---------------------------------------------------------------------------
# Signal quality endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/{recording_id}/quality",
    response_model=SignalQualityResponse,
    summary="Get signal quality report",
)
async def get_quality(
    request: Request,
    recording_id: str,
    trial_index: Annotated[int, Query(ge=0)] = 0,
):
    """Return per-channel signal quality statistics.

    For 3D (epoched) recordings, specify ``trial_index`` to select a trial.
    """
    rec = _get_recording(request, recording_id)
    try:
        report = await run_in_threadpool(inspect_recording, rec, trial_index)
    except IndexError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return SignalQualityResponse(
        recording_id=report.recording_id,
        trial_index=report.trial_index,
        n_channels=report.n_channels,
        n_samples=report.n_samples,
        sampling_rate_hz=report.sampling_rate_hz,
        duration_seconds=report.duration_seconds,
        total_nonfinite_count=report.total_nonfinite_count,
        flatline_channel_indices=report.flatline_channel_indices,
        channel_stats=[
            ChannelStatsItem(
                channel_index=s.channel_index,
                channel_name=s.channel_name,
                n_samples=s.n_samples,
                n_nonfinite=s.n_nonfinite,
                signal_min=s.signal_min,
                signal_max=s.signal_max,
                signal_range=s.signal_range,
                signal_mean=s.signal_mean,
                signal_std=s.signal_std,
                is_flatline=s.is_flatline,
                has_nonfinite=s.has_nonfinite,
            )
            for s in report.channel_stats
        ],
        warnings=report.warnings,
    )


# ---------------------------------------------------------------------------
# Waveform endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/{recording_id}/waveform",
    response_model=WaveformDataResponse,
    summary="Get waveform data for selected channels and time window",
)
async def get_waveform_data(
    request: Request,
    recording_id: str,
    channels: Annotated[
        str | None,
        Query(
            description=(
                "Comma-separated 0-based channel indices, e.g. '0,1,2'.  "
                "Defaults to all channels."
            )
        ),
    ] = None,
    start_s: Annotated[
        float | None,
        Query(description="Start time in seconds (requires known sampling rate)."),
    ] = None,
    end_s: Annotated[
        float | None,
        Query(description="End time in seconds (requires known sampling rate)."),
    ] = None,
    start_sample: Annotated[
        int | None,
        Query(ge=0, description="Start sample index (0-based, inclusive)."),
    ] = None,
    end_sample: Annotated[
        int | None,
        Query(ge=1, description="End sample index (0-based, exclusive)."),
    ] = None,
    max_points: Annotated[
        int,
        Query(ge=10, le=2000, description="Maximum display points per channel."),
    ] = 2000,
    trial_index: Annotated[int, Query(ge=0)] = 0,
):
    """Return waveform display data for the specified channels and time window.

    Actual uploaded samples are returned (or display-downsampled if the window
    exceeds ``max_points``).  The original signal is never modified.

    Second-based parameters require a known sampling rate.
    """
    rec = _get_recording(request, recording_id)

    # Parse comma-separated channel list.
    channel_indices: list[int] | None = None
    if channels is not None:
        try:
            channel_indices = [int(c.strip()) for c in channels.split(",") if c.strip()]
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="Invalid 'channels' parameter.  Provide comma-separated integers.",
            )
        if not channel_indices:
            raise HTTPException(
                status_code=422,
                detail="'channels' parameter must not be empty.",
            )

    try:
        waveform = await run_in_threadpool(
            get_waveform,
            rec,
            channel_indices,
            start_s,
            end_s,
            start_sample,
            end_sample,
            max_points,
            trial_index,
        )
    except IndexError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return WaveformDataResponse(
        recording_id=waveform.recording_id,
        trial_index=waveform.trial_index,
        time_axis_unit=waveform.time_axis_unit,
        sampling_rate_hz=waveform.sampling_rate_hz,
        time_axis_note=waveform.time_axis_note,
        downsampling_note=waveform.downsampling_note,
        channels=[
            WaveformChannelResponse(
                channel_index=ch.channel_index,
                channel_name=ch.channel_name,
                time_values=ch.time_values,
                amplitude_values=ch.amplitude_values,
                n_original_samples=ch.n_original_samples,
                stride=ch.stride,
            )
            for ch in waveform.channels
        ],
    )


# ---------------------------------------------------------------------------
# Channel statistics endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/{recording_id}/stats",
    response_model=ChannelStatsSummaryResponse,
    summary="Get per-channel amplitude statistics",
)
async def get_stats(
    request: Request,
    recording_id: str,
    trial_index: Annotated[int, Query(ge=0)] = 0,
):
    """Return per-channel amplitude statistics for a recording."""
    rec = _get_recording(request, recording_id)
    try:
        summary = await run_in_threadpool(get_channel_stats, rec, trial_index)
    except IndexError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ChannelStatsSummaryResponse(
        recording_id=summary.recording_id,
        n_channels=summary.n_channels,
        n_samples=summary.n_samples,
        sampling_rate_hz=summary.sampling_rate_hz,
        duration_seconds=summary.duration_seconds,
        channel_names=summary.channel_names,
        signal_min=summary.signal_min,
        signal_max=summary.signal_max,
        signal_mean=summary.signal_mean,
        signal_std=summary.signal_std,
        is_flatline=summary.is_flatline,
    )


# ---------------------------------------------------------------------------
# PSD endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/{recording_id}/psd",
    response_model=PSDDataResponse,
    summary="Get power spectral density (requires known sampling rate)",
)
async def get_psd_data(
    request: Request,
    recording_id: str,
    channels: Annotated[
        str | None,
        Query(description="Comma-separated 0-based channel indices.  Defaults to all."),
    ] = None,
    trial_index: Annotated[int, Query(ge=0)] = 0,
):
    """Return Welch PSD computed from actual recording samples.

    Requires a known sampling rate.  Returns frequencies in Hz and power
    spectral density with documented Welch parameters.
    """
    rec = _get_recording(request, recording_id)

    channel_indices: list[int] | None = None
    if channels is not None:
        try:
            channel_indices = [int(c.strip()) for c in channels.split(",") if c.strip()]
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="Invalid 'channels' parameter.",
            )

    try:
        psd = await run_in_threadpool(get_psd, rec, channel_indices, trial_index)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IndexError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PSDDataResponse(
        recording_id=psd.recording_id,
        trial_index=psd.trial_index,
        sampling_rate_hz=psd.sampling_rate_hz,
        method=psd.method,
        channels=[
            PSDChannelResponse(
                channel_index=ch.channel_index,
                channel_name=ch.channel_name,
                frequencies_hz=ch.frequencies_hz,
                power_density=ch.power_density,
                welch_nperseg=ch.welch_nperseg,
                welch_noverlap=ch.welch_noverlap,
                welch_window=ch.welch_window,
            )
            for ch in psd.channels
        ],
        assumptions=psd.assumptions,
    )
