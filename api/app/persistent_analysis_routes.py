"""Authenticated analysis of owned Phase 4.3 recordings."""
import io
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool

from api.app.auth_routes import get_authenticated_user
from api.app.domain.entities import Analysis, Explanation
from api.app.ingestion.csv_reader import read_csv
from api.app.ingestion.mat_reader import read_mat
from api.app.infrastructure.storage.interfaces import StorageError
from api.app.model_adapter.schemas import AnalysisRequest, AnalysisResponse, AnalysisListResponse
from api.app.model_adapter.service import AnalysisError, AnalysisService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v2/analyses/persistent", tags=["Persistent analyses (Phase 4.4)"])


def _deps(request, user):
    if not user.application_user_id:
        raise HTTPException(503, "Internal user identity is unavailable.")
    state = request.app.state
    if state.database_error or state.recording_repository is None or state.analysis_repository is None or state.explanation_repository is None or state.file_storage is None:
        raise HTTPException(503, "Persistent analysis service unavailable.")
    return state


async def _load_recording(state, recording_id, owner_id):
    row = await state.recording_repository.get_owned(recording_id, owner_id)
    if row is None:
        raise HTTPException(404, "Recording not found.")
    metadata = row.metadata
    try:
        stream = await run_in_threadpool(state.file_storage.retrieve, metadata.get("storage_key"))
        try:
            content = await run_in_threadpool(stream.read)
        finally:
            stream.close()
    except FileNotFoundError as exc:
        raise HTTPException(404, "Stored recording file is missing.") from exc
    except StorageError as exc:
        raise HTTPException(503, "Recording storage is unavailable.") from exc
    try:
        if metadata.get("file_format") == "csv":
            eeg = await run_in_threadpool(read_csv, content, metadata["original_filename"], metadata.get("sampling_rate_hz"))
        elif metadata.get("file_format") == "mat":
            provenance = metadata.get("provenance", {}).get("source_metadata", {})
            eeg = await run_in_threadpool(read_mat, content, metadata["original_filename"], metadata.get("sampling_rate_hz"), provenance.get("variable_name"), provenance.get("time_axis"), provenance.get("channel_axis"))
        else:
            raise HTTPException(422, "Stored recording format is unsupported.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Persisted recording could not be reloaded")
        raise HTTPException(422, "Stored recording could not be imported.") from exc
    eeg.recording_id = recording_id
    return eeg


def _response(entity):
    return AnalysisResponse.model_validate(entity.result)


@router.post("/recordings/{recording_id}", response_model=AnalysisResponse, status_code=201)
async def analyze_persistent_recording(recording_id: str, body: AnalysisRequest, request: Request, user=Depends(get_authenticated_user)):
    state = _deps(request, user)
    owner_id = user.application_user_id
    # Ownership is checked before touching file storage.
    eeg = await _load_recording(state, recording_id, owner_id)
    if state.predictor is None:
        raise HTTPException(503, detail={"detail": "The inference model is not ready.", "error_type": "model_unavailable"})
    state.recordings[recording_id] = eeg
    try:
        result = await AnalysisService(request.app).analyze(recording_id, body.include_explanation)
    except AnalysisError as exc:
        failed_at = datetime.now(timezone.utc)
        failure = Analysis(id=str(uuid.uuid4()), owner_id=owner_id, recording_id=recording_id,
                           created_at=failed_at, result={"recording_id": recording_id},
                           status="failed", completed_at=failed_at,
                           error={"error_type": exc.error_type, "detail": exc.detail,
                                  "http_status": exc.status_code})
        try:
            await state.analysis_repository.create(failure)
        except Exception as persist_exc:
            logger.exception("Failed analysis status could not be persisted")
            raise HTTPException(503, "Analysis status could not be persisted.") from persist_exc
        raise HTTPException(exc.status_code, detail={"detail": exc.detail, "error_type": exc.error_type}) from exc

    now = datetime.now(timezone.utc)
    result_dict = result.model_dump(mode="json")
    explanation_trials = result_dict.get("explanation", {}).get("trials", [])
    if result.explanation.status == "available":
        result_dict["explanation"]["trials"] = []
    entity = Analysis(id=result.analysis_id, owner_id=owner_id, recording_id=recording_id,
                      created_at=result.created_at, result=result_dict, status=result.status,
                      completed_at=now)
    try:
        await state.analysis_repository.create(entity)
    except Exception as exc:
        logger.exception("Analysis persistence failed")
        raise HTTPException(503, "Analysis result could not be persisted.") from exc

    explanation = result.explanation
    if explanation.status == "available":
        exp = Explanation(id=str(uuid.uuid4()), owner_id=owner_id, analysis_id=result.analysis_id,
                          created_at=now, content={"recording_id": recording_id,
                          "method": explanation.method, "integration_steps": explanation.integration_steps,
                          "status": explanation.status, "metadata": {"trial_count": len(explanation.trials)},
                          "trials": explanation_trials}, recording_id=recording_id)
        try:
            await state.explanation_repository.create(exp)
        except Exception:
            # Keep the successful inference and mark the explanation unavailable.
            logger.exception("Explanation persistence failed for analysis %s", result.analysis_id)
            result_dict["explanation"] = {"status": "unavailable", "method": explanation.method,
                "integration_steps": explanation.integration_steps, "trials": [],
                "failure_reason": "Explanation persistence failed."}
            try:
                await state.analysis_repository.update(entity.id, {"result": result_dict})
            except Exception:
                logger.exception("Analysis explanation status update failed")
            result = AnalysisResponse.model_validate(result_dict)
    return result


@router.get("", response_model=AnalysisListResponse)
async def list_persistent_analyses(request: Request, user=Depends(get_authenticated_user), recording_id: str | None = Query(None), limit: int = 100, skip: int = 0):
    state = _deps(request, user)
    owner = user.application_user_id
    try:
        rows = (await state.analysis_repository.list_for_recording(recording_id, owner, limit=max(1, min(limit, 100)), skip=max(skip, 0))
                if recording_id else await state.analysis_repository.list_owned(owner, limit=max(1, min(limit, 100)), skip=max(skip, 0)))
    except Exception as exc:
        logger.exception("Analysis listing failed")
        raise HTTPException(503, "Analysis storage is unavailable.") from exc
    responses = [_response(row) for row in rows]
    return AnalysisListResponse(analyses=responses, total_count=len(responses))


@router.get("/{analysis_id}/explanation")
async def get_persistent_explanation(analysis_id: str, request: Request, user=Depends(get_authenticated_user)):
    state = _deps(request, user)
    analysis = await state.analysis_repository.get_owned(analysis_id, user.application_user_id)
    if analysis is None:
        raise HTTPException(404, "Analysis not found.")
    try:
        rows = await state.explanation_repository.list_for_analysis(analysis_id, user.application_user_id, limit=1)
    except Exception as exc:
        logger.exception("Explanation retrieval failed")
        raise HTTPException(503, "Explanation storage is unavailable.") from exc
    if not rows:
        raise HTTPException(404, "Explanation not found.")
    return rows[0].content


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_persistent_analysis(analysis_id: str, request: Request, user=Depends(get_authenticated_user)):
    state = _deps(request, user)
    try:
        analysis = await state.analysis_repository.get_owned(analysis_id, user.application_user_id)
    except Exception as exc:
        logger.exception("Analysis retrieval failed")
        raise HTTPException(503, "Analysis storage is unavailable.") from exc
    if analysis is None:
        raise HTTPException(404, "Analysis not found.")
    result = analysis.result
    if result.get("explanation", {}).get("status") == "available":
        try:
            rows = await state.explanation_repository.list_for_analysis(analysis_id, user.application_user_id, limit=1)
        except Exception as exc:
            logger.exception("Historical explanation retrieval failed")
            raise HTTPException(503, "Explanation storage is unavailable.") from exc
        if rows:
            result = dict(result)
            result["explanation"] = dict(result["explanation"])
            result["explanation"]["trials"] = rows[0].content.get("trials", [])
    return AnalysisResponse.model_validate(result)
