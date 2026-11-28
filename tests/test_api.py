# NeuroForge AI
# Maintainer: Shahab Baloochi

from fastapi.testclient import TestClient

from neuroforge import api


client = TestClient(api.app)


def test_health_without_model(monkeypatch):
    monkeypatch.delenv("NEUROFORGE_ADAPTER", raising=False)
    assert client.get("/health").json()["status"] == "missing_adapter"


def test_mocked_inference(monkeypatch):
    monkeypatch.setattr(api, "get_runtime", lambda: ("tokenizer", "model"))
    monkeypatch.setattr(api, "predict", lambda *args: ("positive", "positive"))
    response = client.post("/analyze", json={"text": "Loved the film."})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"


def test_invalid_request():
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 422
