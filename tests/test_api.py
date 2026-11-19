"""API tests with a model trained on the test data (MLflow is not needed)."""
import json

import joblib
import pytest
from fastapi.testclient import TestClient

import api.main as main
from api.models import ModelStore

OPERATOR = {"X-API-Key": "operator-dev-key"}
VIEWER = {"X-API-Key": "viewer-dev-key"}


@pytest.fixture(scope="module")
def client(tmp_path_factory, model):
    folder = tmp_path_factory.mktemp("models")
    (folder / "production").mkdir()
    joblib.dump(model, folder / "production" / "model.joblib")
    (folder / "production" / "info.json").write_text(json.dumps({"version": 7}))
    main.store = ModelStore(folder)
    return TestClient(main.app)


def body(rows):
    return {"readings": [{"timestamp": str(r["timestamp"]), "valve_open": bool(r["valve_open"]),
                          "valve_minutes": int(r["valve_minutes"]),
                          "sensors": {s: float(r[s]) for s in main.SENSORS}} for _, r in rows.iterrows()]}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": 7}


def test_website_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "AquaSentinel" in r.text


def test_no_key_gives_401(client, sim_data):
    assert client.post("/predict", json=body(sim_data.head(3))).status_code == 401


def test_viewer_cannot_predict_or_rollback(client, sim_data):
    assert client.post("/predict", json=body(sim_data.head(3)), headers=VIEWER).status_code == 403
    assert client.post("/model/rollback", headers=VIEWER).status_code == 403


def test_predict_returns_one_result_per_reading(client, sim_data):
    out = client.post("/predict", json=body(sim_data.head(12)), headers=OPERATOR).json()
    assert out["model_version"] == 7 and len(out["results"]) == 12
    assert {"anomaly_score", "is_anomaly", "leak_alert", "suspected_node"} <= set(out["results"][0])


def test_leak_raises_an_alert(client, sim_data):
    alerted = False
    for _, g in sim_data[sim_data["leak_node"] != ""].groupby("scenario_id"):
        start = int(g["leak"].values.argmax())
        window = g.iloc[start + 4:start + 20]   # some time after the leak started
        alerted |= client.post("/predict", json=body(window), headers=OPERATOR).json()["leak_alert"]
    assert alerted


def test_missing_sensor_gives_422(client, sim_data):
    b = body(sim_data.head(2))
    del b["readings"][0]["sensors"]["P_15"]
    assert client.post("/predict", json=b, headers=OPERATOR).status_code == 422


def test_network_map_and_metrics(client):
    net = client.get("/api/network").json()
    assert len(net["nodes"]) == 97 and len(net["pressure_sensors"]) == 8
    assert "aqua_model_version 7.0" in client.get("/metrics").text
