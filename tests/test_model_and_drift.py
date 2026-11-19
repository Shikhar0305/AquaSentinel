import numpy as np

from src.drift import check_drift
from src.evaluate import evaluate
from src.utils import sensor_columns


def test_leak_readings_score_higher_than_normal(trained_model, sim_data):
    scores = trained_model.anomaly_score(sim_data)
    leak = sim_data["leak"].values == 1
    assert scores[leak].mean() > scores[~leak].mean()


def test_model_detects_most_leaks_with_few_false_alarms(trained_model, sim_data):
    m = evaluate(trained_model, sim_data)
    assert m["roc_auc"] > 0.8
    assert m["detection_rate"] >= 0.75
    assert m["false_alarm_rate"] < 0.05


def test_alert_needs_consecutive_anomalies(trained_model):
    preds = np.array([1, 0, 1, 1, 0, 1, 1, 1, 1])
    alerts = trained_model.alerts(preds)
    # consecutive_for_alert = 3: first alert at index 7
    assert list(alerts) == [0, 0, 0, 0, 0, 0, 0, 1, 1]


def test_suspected_area_names_a_pressure_sensor(trained_model, sim_data, params):
    areas = trained_model.suspected_area(sim_data.head(10))
    valid = {f"junction {n}" for n in params["sensors"]["pressure"]}
    assert set(areas) <= valid


def test_no_drift_on_same_distribution(sim_data, params):
    cols = sensor_columns(params)
    normal = sim_data[sim_data["leak"] == 0]
    a = normal[normal["scenario_id"].isin(["test_000", "test_001"])]
    b = normal[normal["scenario_id"].isin(["test_002", "test_003"])]
    assert not check_drift(a, b, cols, threshold=0.5, psi_threshold=0.5)["drift_detected"]


def test_drift_when_demand_changes(sim_data, params):
    cols = sensor_columns(params)
    ref = sim_data[sim_data["leak"] == 0]
    summer = ref.copy()
    summer[cols] = summer[cols] * 1.3   # crude stand-in for a +30% demand season
    assert check_drift(ref, summer, cols, threshold=0.5, psi_threshold=0.5)["drift_detected"]
