"""Phase 3 analysis API routes."""

from fastapi import APIRouter, HTTPException, Query, Request

from .compatibility import validate_recording_compatibility
from .config import ARTIFACT_TYPE, ADHD_THRESHOLD, CHANNEL_COUNT, EPOCH_SAMPLES, FEATURE_COUNT, LIMITATIONS, SAMPLING_RATE_HZ
from .schemas import AnalysisListResponse, AnalysisRequest, AnalysisResponse, CompatibilityResponse, ModelStatusResponse
from .service import AnalysisError, AnalysisService

router = APIRouter(prefix="/api/v2/analyses", tags=["Model Analysis (Phase 3)"])


def _raise(exc: AnalysisError):
    raise HTTPException(status_code=exc.status_code, detail={"detail": exc.detail, "error_type": exc.error_type})


@router.get("/model-status", response_model=ModelStatusResponse)
async def model_status(request: Request):
    return ModelStatusResponse(available=request.app.state.predictor is not None, artifact_type=ARTIFACT_TYPE,
        sampling_rate_hz=SAMPLING_RATE_HZ, channels=CHANNEL_COUNT, epoch_samples=EPOCH_SAMPLES,
        feature_count=FEATURE_COUNT, class_mapping={"Control": 0, "ADHD": 1}, threshold=ADHD_THRESHOLD,
        scientific_limitations=LIMITATIONS)


@router.get("/{recording_id}/compatibility", response_model=CompatibilityResponse)
async def compatibility(recording_id: str, request: Request):
    recording = request.app.state.recordings.get(recording_id)
    if recording is None:
        raise HTTPException(404, detail={"detail": "Recording not found.", "error_type": "recording_not_found"})
    errors = validate_recording_compatibility(recording)
    return CompatibilityResponse(recording_id=recording_id, compatible=not errors, errors=errors, scientific_limitations=LIMITATIONS)


@router.post("/{recording_id}", response_model=AnalysisResponse)
async def analyze(recording_id: str, body: AnalysisRequest, request: Request):
    try:
        return await AnalysisService(request.app).analyze(recording_id, body.include_explanation)
    except AnalysisError as exc:
        _raise(exc)


@router.get("", response_model=AnalysisListResponse)
async def list_analyses(request: Request, recording_id: str | None = Query(default=None)):
    async with request.app.state.analyses_lock:
        values = list(request.app.state.analyses.values())
    if recording_id is not None:
        values = [x for x in values if x.recording_id == recording_id]
    return AnalysisListResponse(analyses=values, total_count=len(values))


@router.get("/{analysis_id}/explanation")
async def get_explanation(analysis_id: str, request: Request, recording_id: str | None = Query(default=None)):
    try:
        result = await AnalysisService(request.app).get(analysis_id, recording_id)
    except AnalysisError as exc:
        _raise(exc)
    return result.explanation


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str, request: Request, recording_id: str | None = Query(default=None)):
    try:
        return await AnalysisService(request.app).get(analysis_id, recording_id)
    except AnalysisError as exc:
        _raise(exc)
