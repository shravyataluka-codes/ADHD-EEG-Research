"""Pydantic v2 request and response schemas for model adapter endpoints."""

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class AdapterModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class AnalysisRequest(AdapterModel):
    include_explanation: bool = False


class CompatibilityResponse(AdapterModel):
    recording_id: str
    compatible: bool
    errors: list[str]
    scientific_limitations: list[str]


class ExplanationResponse(AdapterModel):
    status: Literal["available", "unavailable"]
    method: Literal["joint_integrated_gradients"] | None = None
    integration_steps: int | None = None
    trials: list[dict[str, Any]] = Field(default_factory=list)
    failure_reason: str | None = None


class AnalysisResponse(AdapterModel):
    analysis_id: str
    recording_id: str
    status: str
    predicted_class: str | None
    adhd_probability: float | None
    control_probability: float | None
    threshold: float
    model_metadata: dict[str, Any]
    processing_metadata: dict[str, Any]
    created_at: datetime
    scientific_limitations: list[str]
    explanation: ExplanationResponse
    trials: list[dict[str, Any]]


class AnalysisListResponse(AdapterModel):
    analyses: list[AnalysisResponse]
    total_count: int


class ModelStatusResponse(AdapterModel):
    available: bool
    artifact_type: str
    sampling_rate_hz: int
    channels: int
    epoch_samples: int
    feature_count: int
    class_mapping: dict[str, int]
    threshold: float
    scientific_limitations: list[str]


class ErrorResponse(AdapterModel):
    detail: str
    error_type: str
