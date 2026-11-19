from fastapi.testclient import TestClient

from api.app.main import create_app


def test_model_status_and_missing_recording():
    app = create_app(predictor_loader=lambda *_args, **_kwargs: object())
    with TestClient(app) as client:
        status = client.get("/api/v2/analyses/model-status")
        assert status.status_code == 200
        assert len(status.json()["scientific_limitations"]) >= 7
        assert client.get("/api/v2/analyses/missing/compatibility").status_code == 404
        assert client.post("/api/v2/analyses/missing", json={}).status_code == 404


def test_checkpoint_loader_is_startup_only():
    calls = []
    app = create_app(predictor_loader=lambda *args, **kwargs: calls.append((args, kwargs)) or object())
    with TestClient(app) as client:
        assert client.get("/api/v2/analyses/model-status").json()["available"]
        assert len(calls) == 1
