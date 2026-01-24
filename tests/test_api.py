from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)

def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data

def test_predict_validation():
    r = client.post("/predict", json={"text": ""})
    assert r.status_code in (422, 400)
