import numpy as np
import pandas as pd

from src.config import sensor_columns
from src.data import validate
from src.monitor import check_drift
from src.train import evaluate, split_by_scenario


# ---------------- data
def test_simulation_has_every_sensor_and_no_gaps(params, sim_data):
    cols = sensor_columns(params)
    assert set(cols + ["timestamp", "scenario_id", "leak"]) <= set(sim_data.columns)
    assert not sim_data[cols].isna().any().any()


def test_first_day_is_dropped(sim_data):
    assert pd.to_datetime(sim_data["timestamp"]).min() == pd.Timestamp("2026-01-02")


def test_leak_label_switches_on_once_and_stays_on(sim_data):
    for _, g in sim_data.groupby("scenario_id"):
        labels = g["leak"].values
        if g["leak_node"].iloc[0]:
            start = labels.argmax()
            assert labels[:start].sum() == 0 and labels[start:].all()
        else:
            assert labels.sum() == 0


def test_validation_fixes_broken_sensor_values(params, sim_data):
    batch = sim_data[sim_data["scenario_id"] == "test_000"].copy()
    batch.iloc[5, batch.columns.get_loc("P_15")] = 9999     # impossible pressure
    batch.iloc[6, batch.columns.get_loc("F_60")] = np.nan   # lost reading
    cleaned, bad = validate(batch, params)
    assert bad == 1
    assert cleaned["P_15"].max() < 200 and not cleaned[sensor_columns(params)].isna().any().any()


def test_split_never_cuts_a_scenario_in_half(sim_data):
    train, test = split_by_scenario(sim_data)
    assert set(train["scenario_id"]).isdisjoint(test["scenario_id"])
    assert test["leak"].max() == 1


# ---------------- model
def test_leaks_score_higher_than_normal_readings(model, sim_data):
    scores = model.anomaly_score(sim_data)
    leak = sim_data["leak"].values == 1
    assert scores[leak].mean() > scores[~leak].mean()


def test_model_finds_leaks_without_false_alarms(model, sim_data):
    m = evaluate(model, sim_data)
    assert m["roc_auc"] > 0.8
    assert m["leaks_detected"] >= 0.75
    assert m["false_alarm_rate"] < 0.05


def test_alert_needs_three_anomalies_in_a_row(model, sim_data):
    out = model.predict_details(sim_data)
    for _, g in out.groupby(sim_data["scenario_id"]):
        flags, alerts = g["is_anomaly"].values, g["leak_alert"].values
        for i in np.where(alerts == 1)[0]:
            assert flags[i - 2:i + 1].all()


def test_less_water_than_normal_is_never_a_leak(model, sim_data, params):
    low = sim_data[sim_data["leak"] == 0].copy()
    for p in params["sensors"]["inflow"]:
        low[f"F_{p}"] *= 0.7     # 30% less inflow everywhere
    assert model.predict_details(low)["is_anomaly"].sum() == 0


def test_suspected_node_is_a_pressure_sensor(model, sim_data, params):
    nodes = set(model.predict_details(sim_data.head(20))["suspected_node"])
    assert nodes <= set(params["sensors"]["pressure"])


# ---------------- drift
def test_no_drift_between_normal_days(sim_data, params):
    normal = sim_data[sim_data["leak"] == 0]
    a = normal[normal["scenario_id"].isin(["test_000", "test_001"])]
    b = normal[normal["scenario_id"].isin(["test_002", "test_003"])]
    assert not check_drift(a, b, sensor_columns(params))["drift_detected"]


def test_drift_when_demand_goes_up(sim_data, params):
    cols = sensor_columns(params)
    ref = sim_data[sim_data["leak"] == 0]
    summer = ref.copy()
    summer[cols] = summer[cols] * 1.3
    assert check_drift(ref, summer, cols)["drift_detected"]
