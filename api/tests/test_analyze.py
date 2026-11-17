from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from api.app.main import create_app
from api.app.settings import Settings
from src.explainability import EpochExplanation, RecordingExplanation
from src.inference import EpochPrediction, RecordingInferenceResult
from src.ui_service import UIAnalysisResult


def _settings():
    return Settings(
        checkpoint_path=Path("unused.pt"),
        max_upload_size_bytes=2 * 1024 * 1024,
        max_request_size_bytes=3 * 1024 * 1024,
        max_recording_duration_seconds=60,
        allowed_origins=("http://localhost:3000",),
        max_concurrent_inference=1,
        inference_wait_timeout_seconds=0.1,
    )


def _recording(seconds=6):
    samples = seconds * 256
    time = np.arange(samples, dtype=np.float64) / 256
    return np.column_stack(
        (np.sin(2 * np.pi * 10 * time), np.sin(2 * np.pi * 17 * time))
    )


def _csv(recording):
    header = "channel_slot_1,channel_slot_2\n"
    body = "\n".join(f"{row[0]:.8f},{row[1]:.8f}" for row in recording)
    return (header + body + "\n").encode("utf-8")


def _analysis():
    names = tuple(f"feature_{index}" for index in range(40))
    raw = np.zeros((2, 1024), dtype=np.float64)
    feature_values = np.arange(40, dtype=np.float64) / 100
    epoch_explanation = EpochExplanation(
        epoch_index=1,
        start_sample=256,
        stop_sample=1280,
        probabilities=np.asarray([0.25, 0.75]),
        predicted_label=1,
        predicted_class="ADHD",
        logit_margin=1.0,
        raw_attributions=raw,
        raw_attribution_magnitudes=np.abs(raw),
        feature_attributions=feature_values,
        feature_attribution_magnitudes=np.abs(feature_values),
        completeness_residual=0.001,
    )
    inference = RecordingInferenceResult(
        status="ok",
        epoch_predictions=(
            EpochPrediction(
                epoch_index=1,
                start_sample=256,
                stop_sample=1280,
                probability_control=0.25,
                probability_adhd=0.75,
                predicted_label=1,
                predicted_class="ADHD",
            ),
        ),
        summary={
            "valid_epoch_count": 1,
            "rejected_epoch_count": 0,
            "predicted_epoch_counts": {"Control": 0, "ADHD": 1},
            "majority_vote_label": 1,
            "majority_vote_class": "ADHD",
            "mean_probabilities": {"Control": 0.25, "ADHD": 0.75},
        },
        candidate_epoch_count=1,
        rejected_epochs=(),
        edge_trim_samples_per_side=256,
        epoch_samples=1024,
        discarded_tail_samples=0,
    )
    explanation = RecordingExplanation(
        status="ok",
        feature_names=names,
        epoch_explanations=(epoch_explanation,),
        mean_signed_feature_attribution=feature_values,
        mean_absolute_feature_attribution=np.abs(feature_values),
        mean_signed_raw_attribution=raw,
        mean_absolute_raw_attribution=np.abs(raw),
        candidate_epoch_count=1,
        accepted_epoch_count=1,
        rejected_epoch_count=0,
        rejected_epochs=(),
        edge_trim_samples_per_side=256,
        epoch_samples=1024,
        discarded_tail_samples=0,
        integration_steps=128,
        baseline_description="Zero-valued normalized inputs.",
    )
    return UIAnalysisResult(
        inference=inference,
        explanation=explanation,
        sampling_rate_hz=256.0,
    )


def _client(predictor=None):
    if predictor is None:
        predictor = SimpleNamespace(
            preprocessing_configuration={
                "sampling_rate_hz": 256,
                "edge_trim_seconds": 1,
                "epoch_duration_seconds": 4,
            }
        )
    return TestClient(
        create_app(
            _settings(),
            predictor_loader=lambda *_args, **_kwargs: predictor,
        )
    )


def test_valid_upload_returns_predictions_attributions_and_finite_json():
    recording = _recording()
    with patch("api.app.service.analyze_recording", return_value=_analysis()) as run:
        with _client() as client:
            response = client.post(
                "/api/v1/analyze",
                files={"file": ("sample.csv", _csv(recording), "text/csv")},
                data={"sampling_rate": "256"},
            )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["prediction"]["majority_vote_class"] == "ADHD"
    assert payload["prediction"]["mean_model_output_probabilities"] == {
        "control": 0.25,
        "adhd": 0.75,
    }
    assert payload["quality_control"]["accepted_epoch_count"] == 1
    assert len(payload["epoch_analyses"]) == 1
    epoch = payload["epoch_analyses"][0]
    assert len(epoch["feature_attributions"]) == 40
    assert len(epoch["raw_attributions"]["channel_slot_1"]) == 1024
    assert len(epoch["raw_attributions"]["channel_slot_2"]) == 1024
    assert epoch["completeness_residual"] == 0.001
    assert payload["recording_explanation"]["integration_steps"] == 128
    assert "not a clinical diagnostic system" in payload["research_use_disclaimer"]
    assert "calibrated confidence" in payload["research_use_disclaimer"]
    assert run.call_count == 1
    assert run.call_args.kwargs["integration_steps"] == 128


def test_malformed_csv_is_rejected_without_running_inference():
    with patch("api.app.service.analyze_recording") as run:
        with _client() as client:
            response = client.post(
                "/api/v1/analyze",
                files={"file": ("bad.csv", b"a,b\n1,not-a-number\n", "text/csv")},
                data={"sampling_rate": "256"},
            )

    assert response.status_code == 422
    assert "numeric" in response.json()["detail"]
    run.assert_not_called()


def test_wrong_channel_count_is_rejected_without_running_inference():
    with patch("api.app.service.analyze_recording") as run:
        with _client() as client:
            response = client.post(
                "/api/v1/analyze",
                files={"file": ("bad.csv", b"a,b,c\n1,2,3\n", "text/csv")},
                data={"sampling_rate": "256"},
            )

    assert response.status_code == 422
    assert "two columns" in response.json()["detail"]
    run.assert_not_called()


def test_incorrect_sampling_rate_is_rejected():
    with patch("api.app.service.analyze_recording") as run:
        with _client() as client:
            response = client.post(
                "/api/v1/analyze",
                files={"file": ("sample.csv", _csv(_recording()), "text/csv")},
                data={"sampling_rate": "128"},
            )

    assert response.status_code == 422
    assert "256 Hz" in response.json()["detail"]
    run.assert_not_called()


def test_recording_too_short_is_rejected():
    with patch("api.app.service.analyze_recording") as run:
        with _client() as client:
            response = client.post(
                "/api/v1/analyze",
                files={
                    "file": (
                        "short.csv",
                        _csv(_recording(seconds=5))[:-1],
                        "text/csv",
                    )
                },
                data={"sampling_rate": "256"},
            )

    assert response.status_code == 422
    assert "too short" in response.json()["detail"].lower()
    run.assert_not_called()
