import numpy as np
import pandas as pd

from src.evaluate import split_by_scenario
from src.features import add_time_slot, add_water_balance, fit_baseline, make_features
from src.ingest import validate
from src.utils import sensor_columns


def test_simulation_has_all_sensor_columns(params, sim_data):
    for col in sensor_columns(params) + ["timestamp", "scenario_id", "leak"]:
        assert col in sim_data.columns
    assert not sim_data[sensor_columns(params)].isna().any().any()


def test_warmup_day_is_dropped(sim_data):
    first = pd.to_datetime(sim_data["timestamp"]).min()
    assert first == pd.Timestamp("2026-01-02")


def test_leak_labels_only_after_leak_starts(sim_data):
    for sid, g in sim_data.groupby("scenario_id"):
        if g["leak_node"].iloc[0]:
            labels = g["leak"].values
            start = labels.argmax()
            assert labels[:start].sum() == 0 and labels[start:].all()
        else:
            assert g["leak"].sum() == 0


def test_water_balance_is_sum_of_inflow_pipes(params):
    df = pd.DataFrame({f"F_{p}": [1.0, 2.0] for p in params["sensors"]["inflow"]})
    out = add_water_balance(df, params["sensors"]["inflow"])
    assert list(out["net_inflow"]) == [5.0, 10.0]


def test_baseline_covers_every_15_min_slot(sim_data, params):
    df = add_water_balance(sim_data[sim_data["leak"] == 0], params["sensors"]["inflow"])
    base = fit_baseline(df, ["net_inflow"])
    assert len(base) == 96
    assert add_time_slot(pd.DataFrame({"timestamp": ["2026-01-01 23:45"]})).iloc[0] == 95


def test_rolling_features_do_not_mix_scenarios():
    base = pd.DataFrame({"net_inflow": np.zeros(96)})
    df = pd.DataFrame({
        "timestamp": ["2026-01-01 00:00", "2026-01-01 00:15"] * 2,
        "scenario_id": ["a", "a", "b", "b"],
        "net_inflow": [100.0, 100.0, 0.0, 0.0],
    })
    feats = make_features(df, base, windows=(4,))
    # scenario b must not inherit the 100s from scenario a
    assert feats.iloc[2, 0] == 0.0 and feats.iloc[3, 0] == 0.0


def test_split_keeps_whole_scenarios_apart(sim_data):
    train, test = split_by_scenario(sim_data, test_size=0.3, seed=0)
    assert set(train["scenario_id"]).isdisjoint(set(test["scenario_id"]))
    assert test["leak"].max() == 1 and (test.groupby("scenario_id")["leak"].max() == 0).any()


def test_validation_repairs_bad_sensor_values(params, sim_data):
    batch = sim_data[sim_data["scenario_id"] == "test_000"].copy()
    batch.iloc[5, batch.columns.get_loc("P_15")] = 9999   # broken sensor
    batch.iloc[6, batch.columns.get_loc("F_60")] = np.nan  # lost packet
    cleaned, report = validate(batch, params)
    assert report["out_of_range"] == 1
    assert not cleaned[sensor_columns(params)].isna().any().any()
    assert cleaned["P_15"].max() < 200
