from fastapi.testclient import TestClient

from src.api import app
from src.leak_analysis import analyze_leak

client = TestClient(app)

# 3 hours of readings while supply is off
NORMAL_NIGHT = [16.3, 13.4, 19.0, 13.8, 19.7, 13.3, 19.1, 15.2, 17.1, 17.7, 21.4, 17.5]
LEAK_NIGHT = [47.9, 49.2, 46.5, 44.0, 43.0, 48.4, 48.1, 44.3, 44.6, 46.9, 42.4, 45.7]


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict(monkeypatch):
    monkeypatch.setattr("src.api.predict_leak", lambda history, valve_open, minutes: 0.12)

    response = client.post("/predict", json={
        "zone_id": "zone-01",
        "inflow_history": NORMAL_NIGHT,
        "valve_open": False,
        "valve_minutes": 255
    })

    assert response.status_code == 200
    assert response.json()["leak_probability"] == 0.12
    assert response.json()["leak_detected"] is False
    assert response.json()["zone_id"] == "zone-01"


def test_analyze_no_leak(monkeypatch):
    monkeypatch.setattr("src.api.predict_leak", lambda history, valve_open, minutes: 0.05)

    response = client.post("/analyze", json={
        "zone_id": "zone-normal",
        "inflow_history": NORMAL_NIGHT,
        "valve_open": False,
        "valve_minutes": 255
    })

    assert response.status_code == 200
    assert response.json()["status"] == "NO_LEAK"


def test_analyze_critical_leak(monkeypatch):
    monkeypatch.setattr("src.api.predict_leak", lambda history, valve_open, minutes: 0.97)

    response = client.post("/analyze", json={
        "zone_id": "zone-leak",
        "inflow_history": LEAK_NIGHT,
        "valve_open": False,
        "valve_minutes": 285
    })

    assert response.status_code == 200
    assert response.json()["status"] == "CRITICAL_LEAK"
    # about 30 L/s above the normal 16 L/s while supply is off
    assert 25 < response.json()["water_loss"]["estimated_loss_lps"] < 35


def test_real_model_finds_the_leak():
    normal = client.post("/predict", json={
        "inflow_history": NORMAL_NIGHT, "valve_open": False, "valve_minutes": 255
    }).json()
    leak = client.post("/predict", json={
        "inflow_history": LEAK_NIGHT, "valve_open": False, "valve_minutes": 285
    }).json()

    assert normal["leak_detected"] is False
    assert leak["leak_detected"] is True


def test_no_water_loss_estimate_during_supply():
    result = analyze_leak(0.9, LEAK_NIGHT, valve_open=True)
    assert result["water_loss"] is None


def test_invalid_history():
    response = client.post("/predict", json={
        "zone_id": "zone-invalid",
        "inflow_history": [10, 20, 30],
        "valve_open": False,
        "valve_minutes": 60
    })

    assert response.status_code == 422


def test_invalid_inflow_values():
    response = client.post("/predict", json={
        "zone_id": "zone-invalid",
        "inflow_history": [10, 20, 30, 40, 50, 60,
                           70, 80, 90, 100, -5, 9000],
        "valve_open": False,
        "valve_minutes": 60
    })

    assert response.status_code == 422
