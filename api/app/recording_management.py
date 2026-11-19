"""Authenticated persistent recording operations (Phase 4.3)."""
import io
import logging
import uuid
from urllib.parse import quote
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from api.app.auth_routes import get_authenticated_user
from api.app.domain.entities import Recording
from api.app.ingestion.csv_reader import read_csv
from api.app.ingestion.mat_reader import AmbiguousStructureError, IngestionError, UnsupportedFormatError, read_mat
from api.app.infrastructure.storage.interfaces import StorageConflict, StorageError, StorageUnavailable

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v2/recordings", tags=["Persistent recordings (Phase 4.3)"])


def _safe_name(name: str | None) -> str:
    # Store a basename for metadata only; it is never used to form a storage path.
    return Path((name or "upload").replace("\\", "/")).name[:255] or "upload"


def _metadata(rec: Recording):
    public = dict(rec.metadata)
    public.pop("storage_key", None)
    return {"recording_id": rec.id, **public}


def _dependencies(request: Request, user):
    if not user.application_user_id:
        raise HTTPException(503, "Persistent recording service unavailable: internal user identity is unavailable.")
    repo = getattr(request.app.state, "recording_repository", None)
    storage = getattr(request.app.state, "file_storage", None)
    if repo is None or storage is None or getattr(request.app.state, "database_error", None):
        raise HTTPException(503, "Persistent recording service unavailable.")
    return repo, storage


@router.post("/persistent/upload", status_code=201)
async def upload_persistent_recording(
    request: Request,
    file: UploadFile = File(...),
    sampling_rate_hz: float | None = Form(None),
    variable_name: str | None = Form(None),
    time_axis: int | None = Form(None),
    channel_axis: int | None = Form(None),
    user=Depends(get_authenticated_user),
):
    repo, storage = _dependencies(request, user)
    original = _safe_name(file.filename)
    suffix = Path(original).suffix.lower()
    if suffix not in {".csv", ".mat"}:
        raise HTTPException(415, "Supported file types are .csv and .mat.")
    max_bytes = request.app.state.settings.ingestion_max_upload_size_bytes
    chunks, size = [], 0
    while True:
        chunk = await file.read(min(1024 * 1024, max_bytes + 1 - size))
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            await file.close()
            raise HTTPException(413, "Upload exceeds the configured size limit.")
        chunks.append(chunk)
    await file.close()
    content = b"".join(chunks)
    try:
        if suffix == ".csv":
            eeg = await run_in_threadpool(read_csv, content, original, sampling_rate_hz)
        else:
            eeg = await run_in_threadpool(read_mat, content, original, sampling_rate_hz, variable_name, time_axis, channel_axis)
    except UnsupportedFormatError as exc:
        raise HTTPException(415, str(exc)) from exc
    except (AmbiguousStructureError, IngestionError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("EEG import failed")
        raise HTTPException(422, "EEG file could not be imported.") from exc

    now = datetime.now(timezone.utc)
    recording_id = str(uuid.uuid4())
    key = f"{uuid.uuid4().hex}{suffix}"
    storage_meta = {
        "original_filename": original, "storage_key": storage.reference(key),
        "file_format": suffix[1:], "file_size": size,
        "sampling_rate_hz": eeg.sampling_rate_hz, "n_channels": eeg.n_channels,
        "channel_names": eeg.channel_names, "n_samples": eeg.n_samples,
        "n_trials": eeg.n_trials, "dimensions": list(eeg.data_shape),
        "duration_seconds": eeg.duration_seconds, "import_status": eeg.import_status,
        "validation": {"warnings": eeg.warnings, "errors": eeg.errors},
        "quality": {}, "provenance": {"source": eeg.provenance, "source_metadata": eeg.source_metadata},
        "created_at": now, "updated_at": now,
    }
    try:
        await run_in_threadpool(storage.save, key, io.BytesIO(content), max_bytes=max_bytes)
    except StorageConflict as exc:
        raise HTTPException(503, "Storage key collision; retry the upload.") from exc
    except StorageError as exc:
        logger.exception("Recording file storage failed")
        raise HTTPException(503, "Recording storage is unavailable.") from exc
    try:
        created = await repo.create(Recording(id=recording_id, owner_id=user.application_user_id, created_at=now, metadata=storage_meta))
    except Exception as exc:
        try:
            await run_in_threadpool(storage.delete, key)
        except Exception:
            logger.exception("Compensating file cleanup failed for recording %s", recording_id)
        logger.exception("Recording metadata persistence failed")
        raise HTTPException(503, "Recording metadata could not be persisted.") from exc
    # Keep the Phase 2 canonical representation available for this process's analysis endpoints.
    eeg.recording_id = recording_id
    request.app.state.recordings[recording_id] = eeg
    return _metadata(created)


@router.get("/persistent")
async def list_persistent_recordings(request: Request, user=Depends(get_authenticated_user), limit: int = 100, skip: int = 0):
    repo, _ = _dependencies(request, user)
    rows = await repo.list_owned(user.application_user_id, limit=max(1, min(limit, 100)), skip=max(skip, 0))
    return {"recordings": [_metadata(row) for row in rows], "total_count": len(rows)}


async def _owned(request, recording_id, user):
    repo, storage = _dependencies(request, user)
    row = await repo.get_owned(recording_id, user.application_user_id)
    if row is None:
        raise HTTPException(404, "Recording not found.")
    return row, storage


@router.get("/persistent/{recording_id}")
async def get_persistent_recording(request: Request, recording_id: str, user=Depends(get_authenticated_user)):
    row, _ = await _owned(request, recording_id, user)
    return _metadata(row)


@router.get("/persistent/{recording_id}/download")
async def download_persistent_recording(request: Request, recording_id: str, user=Depends(get_authenticated_user)):
    row, storage = await _owned(request, recording_id, user)
    key = row.metadata.get("storage_key")
    try:
        stream = await run_in_threadpool(storage.retrieve, key)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Stored recording file is missing.") from exc
    except StorageError as exc:
        raise HTTPException(503, "Recording storage is unavailable.") from exc
    filename = quote(row.metadata.get("original_filename", "recording"), safe="")
    return StreamingResponse(stream, media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"})


@router.post("/persistent/{recording_id}/load")
async def load_persistent_recording(request: Request, recording_id: str, user=Depends(get_authenticated_user)):
    """Rehydrate the canonical Phase 2 EEGRecording representation on demand."""
    row, storage = await _owned(request, recording_id, user)
    key = row.metadata.get("storage_key")
    try:
        stream = await run_in_threadpool(storage.retrieve, key)
        try:
            content = await run_in_threadpool(stream.read)
        finally:
            stream.close()
    except FileNotFoundError as exc:
        raise HTTPException(404, "Stored recording file is missing.") from exc
    except StorageError as exc:
        raise HTTPException(503, "Recording storage is unavailable.") from exc
    metadata = row.metadata
    try:
        if metadata["file_format"] == "csv":
            eeg = await run_in_threadpool(read_csv, content, metadata["original_filename"], metadata.get("sampling_rate_hz"))
        else:
            provenance = metadata.get("provenance", {})
            source = provenance.get("source_metadata", {})
            eeg = await run_in_threadpool(read_mat, content, metadata["original_filename"], metadata.get("sampling_rate_hz"), source.get("variable_name"), source.get("time_axis"), source.get("channel_axis"))
    except Exception as exc:
        logger.exception("Persisted recording could not be reloaded")
        raise HTTPException(422, "Stored recording could not be imported.") from exc
    eeg.recording_id = recording_id
    request.app.state.recordings[recording_id] = eeg
    return {"recording_id": recording_id, "loaded": True}


@router.delete("/persistent/{recording_id}", status_code=204)
async def delete_persistent_recording(request: Request, recording_id: str, user=Depends(get_authenticated_user)):
    row, storage = await _owned(request, recording_id, user)
    key = row.metadata.get("storage_key")
    try:
        await run_in_threadpool(storage.delete, key)
    except StorageError as exc:
        logger.exception("Recording file deletion failed")
        raise HTTPException(503, "Recording storage is unavailable.") from exc
    if not await request.app.state.recording_repository.delete(recording_id):
        raise HTTPException(503, "Recording metadata could not be deleted.")
    request.app.state.recordings.pop(recording_id, None)
