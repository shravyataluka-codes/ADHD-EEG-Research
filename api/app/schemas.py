"""Typed JSON response models for recording analysis."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class HealthResponse(APIModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(APIModel):
    status: Literal["ready"] = "ready"


class ModelProbabilities(APIModel):
    control: float
    adhd: float


class RejectedEpoch(APIModel):
    epoch_index: int
    start_sample: int | None = None
    stop_sample: int | None = None
    reason: str
    channel_slots: list[int] = Field(default_factory=list)


class QualityControl(APIModel):
    candidate_epoch_count: int
    accepted_epoch_count: int
    rejected_epoch_count: int
    rejected_epochs: list[RejectedEpoch]
    edge_trim_samples_per_side: int
    epoch_samples: int
    discarded_tail_samples: int | None


class RecordingPrediction(APIModel):
    majority_vote_class: Literal["Control", "ADHD"] | None
    majority_vote_label: int | None
    predicted_epoch_counts: dict[str, int]
    mean_model_output_probabilities: ModelProbabilities | None
    probabilities_are_calibrated_confidence: Literal[False] = False


class FeatureAttribution(APIModel):
    feature_name: str
    signed_contribution: float
    absolute_magnitude: float


class ChannelSlotAttributions(APIModel):
    channel_slot_1: list[float]
    channel_slot_2: list[float]


class EpochAnalysis(APIModel):
    epoch_index: int
    start_sample: int
    stop_sample: int
    predicted_class: Literal["Control", "ADHD"]
    model_output_probabilities: ModelProbabilities
    adhd_minus_control_logit: float
    completeness_residual: float
    feature_attributions: list[FeatureAttribution]
    raw_attributions: ChannelSlotAttributions
    raw_attribution_magnitudes: ChannelSlotAttributions


class RecordingExplanation(APIModel):
    method: Literal["joint_integrated_gradients"] = "joint_integrated_gradients"
    target: str
    integration_steps: int
    baseline_description: str
    feature_attributions: list[FeatureAttribution]
    mean_signed_raw_attributions: ChannelSlotAttributions | None
    mean_absolute_raw_attributions: ChannelSlotAttributions | None
    epoch_completeness_residuals: list[float]


class AnalysisResponse(APIModel):
    status: Literal["ok", "no_valid_epochs"]
    sampling_rate_hz: float
    prediction: RecordingPrediction
    quality_control: QualityControl
    epoch_analyses: list[EpochAnalysis]
    recording_explanation: RecordingExplanation
    research_use_disclaimer: str
    scientific_limitations: list[str]
