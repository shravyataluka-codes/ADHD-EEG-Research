"""CSV upload validation and serialization around the existing EEG service."""

from pathlib import Path

import numpy as np

from src.inference import RecordingInferenceResult
from src.ui_service import (
    UIAnalysisResult,
    UIInputError,
    analyze_recording,
    parse_sampling_rate,
    parse_uploaded_recording,
    validate_recording_for_model,
)

from .schemas import (
    AnalysisResponse,
    ChannelSlotAttributions,
    EpochAnalysis,
    FeatureAttribution,
    ModelProbabilities,
    QualityControl,
    RecordingExplanation as RecordingExplanationResponse,
    RecordingPrediction,
    RejectedEpoch,
)
from .settings import Settings


INTEGRATION_STEPS = 128
RESEARCH_USE_DISCLAIMER = (
    "Research use only. This exploratory tool is not a clinical diagnostic "
    "system. Model output probabilities are not calibrated confidence estimates."
)
SCIENTIFIC_LIMITATIONS = [
    "Channel slots 1 and 2 are array positions; electrode identities are unverified.",
    "Participant identities are not verified, and subject-independent "
    "generalization has not been established.",
    "The final checkpoint was fit using all eligible Cell 1 epochs; outputs "
    "for recordings from that set are in-sample, not independent test results.",
    "Integrated Gradients are exploratory attributions, not causal or diagnostic evidence.",
]


class UploadTooLargeError(ValueError):
    """The uploaded file exceeds the configured byte or duration limit."""


def _finite_array(value, shape, field_name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(
            f"Backend returned an unexpected {field_name} shape."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"Backend returned non-finite {field_name}.")
    return array


def _finite_scalar(value, field_name):
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Backend returned non-finite {field_name}.")
    return result


def _feature_attributions(names, signed_values, absolute_values, field_name):
    if len(names) != 40:
        raise ValueError("Backend must return 40 checkpoint feature names.")
    signed = _finite_array(signed_values, (40,), f"{field_name} signed features")
    absolute = _finite_array(
        absolute_values,
        (40,),
        f"{field_name} absolute features",
    )
    return [
        FeatureAttribution(
            feature_name=name,
            signed_contribution=float(signed[index]),
            absolute_magnitude=float(absolute[index]),
        )
        for index, name in enumerate(names)
    ]


def _channel_attributions(values, field_name):
    array = _finite_array(values, (2, 1024), field_name)
    return ChannelSlotAttributions(
        channel_slot_1=array[0].tolist(),
        channel_slot_2=array[1].tolist(),
    )


def _rejected_epochs(values):
    result = []
    for item in values:
        result.append(
            RejectedEpoch(
                epoch_index=int(item["epoch_index"]),
                start_sample=(
                    int(item["start_sample"])
                    if item.get("start_sample") is not None
                    else None
                ),
                stop_sample=(
                    int(item["stop_sample"])
                    if item.get("stop_sample") is not None
                    else None
                ),
                reason=str(item["reason"]),
                channel_slots=[
                    int(slot) for slot in item.get("channel_slots", [])
                ],
            )
        )
    return result


def serialize_analysis(analysis: UIAnalysisResult):
    """Convert existing prediction/XAI dataclasses to finite JSON fields."""
    inference: RecordingInferenceResult = analysis.inference
    explanation = analysis.explanation
    if len(inference.epoch_predictions) != len(explanation.epoch_explanations):
        raise ValueError("Prediction and explanation epoch counts do not match.")

    feature_names = tuple(explanation.feature_names)
    if len(feature_names) != 40 or len(set(feature_names)) != 40:
        raise ValueError("Backend must return 40 unique feature names.")

    epoch_analyses = []
    residuals = []
    for prediction, attribution in zip(
        inference.epoch_predictions,
        explanation.epoch_explanations,
    ):
        if prediction.epoch_index != attribution.epoch_index:
            raise ValueError("Prediction and explanation epoch indices do not match.")
        probabilities = _finite_array(
            attribution.probabilities,
            (2,),
            "epoch probabilities",
        )
        feature_attributions = _feature_attributions(
            feature_names,
            attribution.feature_attributions,
            attribution.feature_attribution_magnitudes,
            "epoch",
        )
        raw_attributions = _channel_attributions(
            attribution.raw_attributions,
            "epoch raw attributions",
        )
        raw_magnitudes = _channel_attributions(
            attribution.raw_attribution_magnitudes,
            "epoch raw attribution magnitudes",
        )
        residual = _finite_scalar(
            attribution.completeness_residual,
            "completeness residual",
        )
        residuals.append(residual)
        epoch_analyses.append(
            EpochAnalysis(
                epoch_index=int(prediction.epoch_index),
                start_sample=int(prediction.start_sample),
                stop_sample=int(prediction.stop_sample),
                predicted_class=prediction.predicted_class,
                model_output_probabilities=ModelProbabilities(
                    control=float(probabilities[0]),
                    adhd=float(probabilities[1]),
                ),
                adhd_minus_control_logit=_finite_scalar(
                    attribution.logit_margin,
                    "logit margin",
                ),
                completeness_residual=residual,
                feature_attributions=feature_attributions,
                raw_attributions=raw_attributions,
                raw_attribution_magnitudes=raw_magnitudes,
            )
        )

    summary = inference.summary
    mean_probabilities = summary.get("mean_probabilities")
    if mean_probabilities is None:
        serialized_probabilities = None
    else:
        serialized_probabilities = ModelProbabilities(
            control=_finite_scalar(
                mean_probabilities["Control"],
                "mean Control probability",
            ),
            adhd=_finite_scalar(
                mean_probabilities["ADHD"],
                "mean ADHD probability",
            ),
        )

    if explanation.accepted_epoch_count:
        mean_feature_attributions = _feature_attributions(
            feature_names,
            explanation.mean_signed_feature_attribution,
            explanation.mean_absolute_feature_attribution,
            "recording",
        )
        mean_raw_signed = _channel_attributions(
            explanation.mean_signed_raw_attribution,
            "recording mean signed raw attributions",
        )
        mean_raw_absolute = _channel_attributions(
            explanation.mean_absolute_raw_attribution,
            "recording mean absolute raw attributions",
        )
    else:
        mean_feature_attributions = []
        mean_raw_signed = None
        mean_raw_absolute = None

    return AnalysisResponse(
        status=inference.status,
        sampling_rate_hz=_finite_scalar(
            analysis.sampling_rate_hz,
            "sampling rate",
        ),
        prediction=RecordingPrediction(
            majority_vote_class=summary["majority_vote_class"],
            majority_vote_label=summary["majority_vote_label"],
            predicted_epoch_counts={
                "Control": int(summary["predicted_epoch_counts"]["Control"]),
                "ADHD": int(summary["predicted_epoch_counts"]["ADHD"]),
            },
            mean_model_output_probabilities=serialized_probabilities,
        ),
        quality_control=QualityControl(
            candidate_epoch_count=int(inference.candidate_epoch_count),
            accepted_epoch_count=int(explanation.accepted_epoch_count),
            rejected_epoch_count=int(explanation.rejected_epoch_count),
            rejected_epochs=_rejected_epochs(inference.rejected_epochs),
            edge_trim_samples_per_side=int(
                inference.edge_trim_samples_per_side
            ),
            epoch_samples=int(inference.epoch_samples),
            discarded_tail_samples=(
                int(inference.discarded_tail_samples)
                if inference.discarded_tail_samples is not None
                else None
            ),
        ),
        epoch_analyses=epoch_analyses,
        recording_explanation=RecordingExplanationResponse(
            target="ADHD logit minus Control logit",
            integration_steps=int(explanation.integration_steps),
            baseline_description=explanation.baseline_description,
            feature_attributions=mean_feature_attributions,
            mean_signed_raw_attributions=mean_raw_signed,
            mean_absolute_raw_attributions=mean_raw_absolute,
            epoch_completeness_residuals=residuals,
        ),
        research_use_disclaimer=RESEARCH_USE_DISCLAIMER,
        scientific_limitations=list(SCIENTIFIC_LIMITATIONS),
    )


def analyze_csv_upload(predictor, filename, content, sampling_rate, settings):
    """Validate a bounded CSV upload, then call the existing analysis service."""
    if len(content) > settings.max_upload_size_bytes:
        raise UploadTooLargeError("Upload exceeds the configured file-size limit.")
    if Path(filename or "").suffix.lower() != ".csv":
        raise UIInputError("Upload a CSV file containing exactly two numeric channels.")

    rate = parse_sampling_rate(sampling_rate)
    recording = parse_uploaded_recording(filename, content)
    validate_recording_for_model(recording, rate, predictor)
    duration_seconds = recording.shape[0] / rate
    if duration_seconds > settings.max_recording_duration_seconds:
        raise UploadTooLargeError(
            "Recording exceeds the configured maximum duration of "
            f"{settings.max_recording_duration_seconds:g} seconds."
        )

    analysis = analyze_recording(
        predictor,
        recording,
        sampling_rate_hz=rate,
        integration_steps=INTEGRATION_STEPS,
    )
    return serialize_analysis(analysis)
