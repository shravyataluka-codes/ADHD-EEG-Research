"""Async-safe application service for model analyses."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from fastapi.concurrency import run_in_threadpool

from . import explainability, inference
from .compatibility import validate_recording_compatibility
from .config import ARTIFACT_TYPE, ADHD_THRESHOLD, INTEGRATION_STEPS, LIMITATIONS
from .preprocessing import canonical_trials
from .schemas import AnalysisResponse, ExplanationResponse

LOGGER = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    """Convert Phase 1 dataclasses and NumPy values to response-safe values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class AnalysisError(Exception):
    def __init__(self, status_code: int, error_type: str, detail: str):
        self.status_code, self.error_type, self.detail = status_code, error_type, detail
        super().__init__(detail)


class AnalysisService:
    def __init__(self, app):
        self.app = app

    async def analyze(self, recording_id: str, include_explanation: bool) -> AnalysisResponse:
        recording = self.app.state.recordings.get(recording_id)
        if recording is None:
            raise AnalysisError(404, "recording_not_found", "Recording not found.")
        errors = validate_recording_compatibility(recording)
        if errors:
            raise AnalysisError(422, "incompatible_recording", "; ".join(errors))
        predictor = self.app.state.predictor
        if predictor is None:
            raise AnalysisError(503, "model_unavailable", "The inference model is not ready.")
        trials = await run_in_threadpool(canonical_trials, recording)
        outputs: list[dict[str, Any]] = []
        explanations: list[dict[str, Any]] = []
        explanation_failed = False
        for index, trial in enumerate(trials):
            try:
                result = await run_in_threadpool(inference.predict_recording, predictor, trial, recording.sampling_rate_hz)
            except Exception as exc:
                LOGGER.error("Inference failed (%s).", type(exc).__name__)
                raise AnalysisError(500, "inference_failed", "Inference failed.") from None
            outputs.append({"trial_index": index, "status": result.status, "summary": result.summary,
                            "epoch_predictions": [item.__dict__ for item in result.epoch_predictions]})
            if include_explanation:
                try:
                    explanation = await run_in_threadpool(explainability.explain_recording, predictor, trial, recording.sampling_rate_hz, INTEGRATION_STEPS)
                    explanations.append({"trial_index": index, "result": _jsonable(explanation)})
                except Exception as exc:
                    explanation_failed = True
                    LOGGER.warning("Explanation unavailable (%s).", type(exc).__name__)
        probabilities = [x["summary"]["mean_probabilities"] for x in outputs if x["summary"].get("mean_probabilities")]
        # A top-level probability is returned only for one continuous recording
        # or one trial. Averaging independent trials would create an untrained
        # recording-level score, so multi-trial results stay trial-scoped.
        adhd = probabilities[0]["ADHD"] if len(probabilities) == 1 else None
        control = probabilities[0]["Control"] if len(probabilities) == 1 else None
        predicted = ("ADHD" if adhd >= ADHD_THRESHOLD else "Control") if adhd is not None else None
        exp_available = bool(explanations) and not explanation_failed
        exp = ExplanationResponse(status="available" if exp_available else "unavailable", method="joint_integrated_gradients" if explanations else None,
                                  integration_steps=INTEGRATION_STEPS if explanations else None, trials=explanations,
                                  failure_reason=None if exp_available else ("Not requested." if not include_explanation else "Explanation generation failed for one or more trials."))
        model_metadata = {"artifact_type": ARTIFACT_TYPE, "training_epoch_count": 553,
                          "class_mapping": dict(getattr(predictor, "class_mapping", {"Control": 0, "ADHD": 1}))}
        response = AnalysisResponse(analysis_id=str(uuid.uuid4()), recording_id=recording_id,
            status=("ok" if any(item["status"] == "ok" for item in outputs) else "no_valid_epochs"),
            predicted_class=predicted, adhd_probability=adhd,
            control_probability=control, threshold=ADHD_THRESHOLD, model_metadata=model_metadata,
            processing_metadata={"sampling_rate_hz": recording.sampling_rate_hz, "trial_count": len(trials), "provenance": recording.provenance},
            created_at=datetime.now(timezone.utc), scientific_limitations=LIMITATIONS, explanation=exp, trials=outputs)
        async with self.app.state.analyses_lock:
            self.app.state.analyses[response.analysis_id] = response
        return response

    async def get(self, analysis_id: str, recording_id: str | None = None):
        async with self.app.state.analyses_lock:
            result = self.app.state.analyses.get(analysis_id)
        if result is None or (recording_id is not None and result.recording_id != recording_id):
            raise AnalysisError(404, "analysis_not_found", "Analysis not found.")
        return result
