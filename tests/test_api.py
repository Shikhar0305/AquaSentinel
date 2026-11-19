"""API tests with a model trained on the test fixture data (no MLflow needed)."""
import json

import joblib
import pytest
from fastapi.testclient import TestClient

import api.main as main
from api.model_service import ModelService

OPERATOR = {"X-API-Key": "operator-dev-key"}
VIEWER = {"X-API-Key": "viewer-dev-key"}


@pytest.fixture(scope="module")
def client(tmp_path_factory, trained_model):
    models_dir = tmp_path_factory.mktemp("models")
    prod = models_dir / "production"
    prod.mkdir()
    joblib.dump(trained_model, prod / "model.joblib")
    (prod / "meta.json").write_text(json.dumps({"name": "test", "version": 7, "run_id": "abc"}))
    main.service = ModelService(models_dir)
    return TestClient(main.app)


def body(df, sensors):
    return {"readings": [{"timestamp": str(r["timestamp"]), "sensors": {s: float(r[s]) for s in sensors}}
                         for _, r in df.iterrows()]}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_version": 7}


def test_predict_requires_api_key(client, sim_data):
    assert client.post("/predict", json=body(sim_data.head(3), main.SENSORS)).status_code == 401


def test_viewer_cannot_predict(client, sim_data):
    r = client.post("/predict", json=body(sim_data.head(3), main.SENSORS), headers=VIEWER)
    assert r.status_code == 403


def test_viewer_cannot_rollback(client):
    assert client.post("/model/rollback", headers=VIEWER).status_code == 403


def test_predict_returns_one_result_per_reading(client, sim_data):
    rows = sim_data.head(12)
    r = client.post("/predict", json=body(rows, main.SENSORS), headers=OPERATOR)
    assert r.status_code == 200
    out = r.json()
    assert out["model_version"] == 7 and out["served_by"] == "champion"
    assert len(out["results"]) == 12
    assert {"anomaly_score", "is_anomaly", "leak_alert", "suspected_area"} <= set(out["results"][0])


def test_leak_scenario_raises_alert(client, sim_data):
    leak_ids = sim_data.loc[sim_data["leak"] == 1, "scenario_id"].unique()
    alerted = False
    for sid in leak_ids:
        g = sim_data[sim_data["scenario_id"] == sid]
        start = int(g["leak"].values.argmax())
        window = g.iloc[start + 4: start + 20]  # some time after the leak started
        alerted |= client.post("/predict", json=body(window, main.SENSORS), headers=OPERATOR).json()["leak_alert"]
    assert alerted


def test_missing_sensor_gives_422(client, sim_data):
    b = body(sim_data.head(2), main.SENSORS)
    del b["readings"][0]["sensors"]["P_15"]
    assert client.post("/predict", json=b, headers=OPERATOR).status_code == 422


def test_model_info_and_metrics(client):
    info = client.get("/model/info", headers=VIEWER).json()
    assert info["champion"]["version"] == 7
    metrics = client.get("/metrics").text
    assert "aqua_predictions_total" in metrics and 'aqua_model_version{role="champion"} 7.0' in metrics
