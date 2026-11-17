from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from api.app.main import create_app
from api.app.settings import Settings


def _settings(
    max_upload_size_bytes=2 * 1024 * 1024,
    max_request_size_bytes=3 * 1024 * 1024,
    max_recording_duration_seconds=60,
):
    return Settings(
        checkpoint_path=Path("unused.pt"),
        max_upload_size_bytes=max_upload_size_bytes,
        max_request_size_bytes=max_request_size_bytes,
        max_recording_duration_seconds=max_recording_duration_seconds,
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
    lines = ["channel_slot_1,channel_slot_2"]
    lines.extend(f"{row[0]:.8f},{row[1]:.8f}" for row in recording)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _client(settings=None):
    predictor = SimpleNamespace(
        preprocessing_configuration={
            "sampling_rate_hz": 256,
            "edge_trim_seconds": 1,
            "epoch_duration_seconds": 4,
        }
    )
    return TestClient(
        create_app(
            settings or _settings(),
            predictor_loader=lambda *_args, **_kwargs: predictor,
        )
    )


def test_oversized_upload_is_rejected_before_inference():
    settings = _settings(
        max_upload_size_bytes=64,
        max_request_size_bytes=1024 * 1024,
    )
    with patch("api.app.service.analyze_recording") as run:
        with _client(settings) as client:
            response = client.post(
                "/api/v1/analyze",
                files={"file": ("large.csv", _csv(_recording()), "text/csv")},
                data={"sampling_rate": "256"},
            )

    assert response.status_code == 413
    run.assert_not_called()


def test_request_content_length_limit_is_enforced():
    settings = _settings(
        max_upload_size_bytes=64,
        max_request_size_bytes=256,
    )
    with _client(settings) as client:
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("sample.csv", b"a,b\n1,2\n", "text/csv")},
            data={"sampling_rate": "256"},
        )

    assert response.status_code == 413
    assert "size limit" in response.json()["detail"]


def test_streamed_request_body_limit_is_enforced():
    settings = _settings(
        max_upload_size_bytes=64,
        max_request_size_bytes=256,
    )
    boundary = "streamed-body"
    body = (
        f'--{boundary}\r\n'
        'Content-Disposition: form-data; name="sampling_rate"\r\n\r\n'
        '256\r\n'
        f'--{boundary}\r\n'
        'Content-Disposition: form-data; name="file"; filename="sample.csv"\r\n'
        'Content-Type: text/csv\r\n\r\n'
    ).encode() + b"a,b\n" + (b"1,2\n" * 200) + f"\r\n--{boundary}--\r\n".encode()

    def chunks():
        yield body[:100]
        yield body[100:]

    with patch("api.app.service.analyze_recording") as run:
        with _client(settings) as client:
            response = client.post(
                "/api/v1/analyze",
                content=chunks(),
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )

    assert response.status_code == 413
    run.assert_not_called()


def test_recording_duration_limit_is_enforced_before_inference():
    with patch("api.app.service.analyze_recording") as run:
        with _client(
            _settings(max_recording_duration_seconds=5)
        ) as client:
            response = client.post(
                "/api/v1/analyze",
                files={"file": ("long.csv", _csv(_recording()), "text/csv")},
                data={"sampling_rate": "256"},
            )

    assert response.status_code == 413
    assert "maximum duration" in response.json()["detail"]
    run.assert_not_called()


def test_internal_inference_error_is_sanitized():
    with patch(
        "api.app.service.analyze_recording",
        side_effect=RuntimeError("C:\\private\\recording.csv and stack details"),
    ):
        with _client() as client:
            response = client.post(
                "/api/v1/analyze",
                files={"file": ("sample.csv", _csv(_recording()), "text/csv")},
                data={"sampling_rate": "256"},
            )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Analysis failed due to an internal inference error."
    )
    assert "C:\\private" not in response.text
    assert "stack details" not in response.text
