from pathlib import Path
import pytest

from fastapi.testclient import TestClient

from api.app.main import create_app
from api.app.settings import Settings


def _settings():
    return Settings(
        checkpoint_path=Path("C:/private/final_fusion_checkpoint.pt"),
        max_upload_size_bytes=1024 * 1024,
        max_request_size_bytes=2 * 1024 * 1024,
        max_recording_duration_seconds=60,
        allowed_origins=("http://localhost:3000",),
        max_concurrent_inference=1,
        inference_wait_timeout_seconds=0.1,
    )


def test_health_readiness_and_checkpoint_loader_runs_once():
    calls = []

    def loader(path, device):
        calls.append((path, device))
        return object()

    app = create_app(_settings(), predictor_loader=loader)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").json() == {"status": "ready"}
        assert client.get("/ready").status_code == 200
        rejected = client.post(
            "/api/v1/analyze",
            files={"file": ("unsupported.txt", b"not a CSV", "text/plain")},
            data={"sampling_rate": "256"},
        )
        assert rejected.status_code == 422

    assert len(calls) == 1
    assert calls[0][1] == "cpu"


def test_openapi_metadata_uses_approved_application_name():
    app = create_app(_settings(), predictor_loader=lambda *_args, **_kwargs: object())
    with TestClient(app) as client:
        info = client.get("/openapi.json").json()["info"]

    assert info["title"] == "Drushti Detect API"
    assert info["description"].startswith("Drushti Detect")


def test_readiness_fails_safely_when_checkpoint_cannot_load():
    def loader(_path, _device):
        raise FileNotFoundError("C:\\private\\checkpoint.pt")

    app = create_app(_settings(), predictor_loader=loader)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/ready")

    assert response.status_code == 503
    assert "C:\\private" not in response.text
    assert "FileNotFoundError" not in response.text


def test_cors_allows_configured_origin_only():
    app = create_app(_settings(), predictor_loader=lambda *_args, **_kwargs: object())
    with TestClient(app) as client:
        allowed = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )
        denied = client.get(
            "/health",
            headers={"Origin": "https://unapproved.example"},
        )

    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "access-control-allow-origin" not in denied.headers


def test_production_requires_explicit_non_wildcard_cors_origins(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    with pytest.raises(ValueError, match="explicitly configured"):
        Settings.from_env()

    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
    with pytest.raises(ValueError, match="explicit HTTP"):
        Settings.from_env()
