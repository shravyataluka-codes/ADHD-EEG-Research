from fastapi.testclient import TestClient

from api.app.main import create_app


def test_predictor_loader_called_once():
    calls = []
    app = create_app(predictor_loader=lambda *a, **k: calls.append(1) or object())
    with TestClient(app) as client:
        client.get("/api/v2/analyses/model-status")
        client.get("/api/v2/analyses/model-status")
    assert len(calls) == 1
