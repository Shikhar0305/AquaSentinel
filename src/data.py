"""
Everything about data: simulation, validation, ingestion and the feature table.

Real sensor data from water utilities is not public, so we simulate a supply zone
(the EPANET Net3 network) with WNTR, the way most Indian cities run it:
water is supplied only a few hours a day, households fill their storage tanks when
the valve opens, supply timings slip, and power cuts leave gaps in the data.
Random leaks are added and every reading after the leak starts is labelled leak = 1.

    python -m src.data simulate    # winter + summer data   -> data/raw/
    python -m src.data ingest      # next 4 summer scenarios -> data/incoming/recent_data.csv
    python -m src.data features    # feature table for Feast -> data/features/
    python -m src.data reset       # forget the ingested data
"""
import sys

import numpy as np
import pandas as pd
import wntr

from src.config import DATA, RECENT_FILE, SUMMER_FILE, TRAIN_FILE, get_logger, load_params, sensor_columns

log = get_logger("data")


# ------------------------------------------------------------------ simulation
def supply_schedule(sim, windows, days, rng):
    """
    Intermittent supply: the zone valve opens only in a few windows a day.
    Returns the demand multiplier and the valve status for every 15-minute step.
    - while the valve is open, households fill their tanks and sumps, so demand
      is high at the start and falls as the tanks fill up
    - while it is closed, only a little water is used
    - supply rarely starts exactly on time
    """
    demand = np.zeros(days * 96)
    valve = np.zeros(days * 96, dtype=int)
    shift = sim["supply_shift_min"] // 15
    for day in range(days):
        off_level = sim["off_supply_use"] * rng.uniform(0.7, 1.3)
        demand[day * 96:(day + 1) * 96] = off_level * rng.uniform(0.9, 1.1, 96)
        for start_h, end_h in windows:
            start = day * 96 + int(start_h * 4) + int(rng.integers(-shift, shift + 1))
            steps = int((end_h - start_h) * 4)
            fill = np.linspace(*sim["storage_fill"], steps) * rng.uniform(1 - sim["fill_variation"],
                                                                          1 + sim["fill_variation"])
            demand[start:start + steps] = fill * rng.uniform(0.95, 1.05, steps)
            valve[start:start + steps] = 1
    return demand, valve


def minutes_since_switch(valve):
    """How long the valve has been in its current state (open or closed), in minutes."""
    out = np.zeros(len(valve), dtype=int)
    for i in range(1, len(valve)):
        out[i] = out[i - 1] + 15 if valve[i] == valve[i - 1] else 0
    return out


def run_scenario(params, scenario_id, has_leak, demand_multiplier, windows, rng):
    sim = params["simulation"]
    days = sim["days"]
    wn = wntr.network.WaterNetworkModel(sim["network"])
    wn.options.time.duration = days * 24 * 3600
    wn.options.time.hydraulic_timestep = 900     # 15 minutes
    wn.options.time.report_timestep = 900
    wn.options.time.pattern_timestep = 900
    wn.options.hydraulic.demand_model = "PDD"    # needed for realistic leaks

    demand, valve = supply_schedule(sim, windows, days, rng)
    wn.add_pattern("supply", list(demand))
    for _, junction in wn.junctions():
        for d in junction.demand_timeseries_list:
            d.base_value *= demand_multiplier
            d.pattern_name = "supply"

    leak_node, leak_area, leak_start = "", 0.0, None
    if has_leak:
        leak_node = str(rng.choice(wn.junction_name_list))
        leak_area = float(rng.uniform(*sim["leak_area"]))
        leak_start = int(rng.integers(*sim["leak_start_hour"])) * 3600
        wn.get_node(leak_node).add_leak(wn, area=leak_area, start_time=leak_start)

    results = wntr.sim.WNTRSimulator(wn).run_sim()
    pressure = results.node["pressure"][params["sensors"]["pressure"]]
    flow = results.link["flowrate"][params["sensors"]["flow"]] * 1000   # m3/s -> L/s

    df = pd.DataFrame(index=pressure.index)
    for col in pressure.columns:
        df[f"P_{col}"] = (pressure[col] + rng.normal(0, sim["noise_pressure"], len(df))).round(3)
    for col in flow.columns:
        df[f"F_{col}"] = (flow[col] + rng.normal(0, sim["noise_flow"], len(df))).round(3)

    step = np.minimum(df.index.values // 900, len(valve) - 1)   # which 15-minute step each row is
    df["valve_open"] = valve[step]
    df["valve_minutes"] = minutes_since_switch(valve)[step]
    df["timestamp"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(df.index, unit="s")
    df["scenario_id"] = scenario_id
    df["leak_node"] = leak_node
    df["leak_area"] = leak_area
    df["leak"] = 0 if leak_start is None else (df.index >= leak_start).astype(int)

    df = df[df.index >= 24 * 3600].reset_index(drop=True)   # drop day 1: tanks are still settling

    # power cut: the sensors' data logger stops for 1-3 hours
    if rng.random() < sim["power_cut_chance"]:
        start = int(rng.integers(0, len(df) - 12))
        cols = [c for c in df.columns if c.startswith(("P_", "F_"))]
        df.loc[start:start + int(rng.integers(4, 13)), cols] = np.nan
    return df


def simulate(params, n_normal, n_leak, demand_multiplier, seed, prefix, windows=None):
    rng = np.random.default_rng(seed)
    windows = windows or params["simulation"]["supply_windows"]
    frames = []
    for i, has_leak in enumerate([False] * n_normal + [True] * n_leak):
        frames.append(run_scenario(params, f"{prefix}_{i:03d}", has_leak, demand_multiplier, windows, rng))
        log.info("scenario %s_%03d done%s", prefix, i, " (leak)" if has_leak else "")
    data = pd.concat(frames, ignore_index=True)
    first = ["timestamp", "scenario_id"]
    return data[first + [c for c in data.columns if c not in first]]


def simulate_all():
    p = load_params()
    s, w = p["simulation"], p["summer"]
    winter = simulate(p, s["normal_scenarios"], s["leak_scenarios"], 1.0, s["seed"], "winter")
    winter.to_csv(TRAIN_FILE, index=False)
    summer = simulate(p, w["normal_scenarios"], w["leak_scenarios"], w["demand_multiplier"], w["seed"], "summer",
                      windows=w["supply_windows"])
    summer.to_csv(SUMMER_FILE, index=False)
    log.info("saved %d winter rows and %d summer rows", len(winter), len(summer))


# ------------------------------------------------------------------ validation + ingestion
def validate(df, params):
    """Check columns, remove impossible values and fill short gaps."""
    cols = sensor_columns(params)
    missing = [c for c in ["timestamp", "scenario_id"] + cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df.copy()
    bad = 0
    for c in cols:
        low, high = (-20, 200) if c.startswith("P_") else (-5000, 5000)
        wrong = (df[c] < low) | (df[c] > high)
        bad += int(wrong.sum())
        df.loc[wrong, c] = np.nan
    df[cols] = df.groupby("scenario_id")[cols].transform(lambda s: s.interpolate(limit_direction="both"))
    if df[cols].isna().any().any():
        raise ValueError("data still has missing values after cleaning")
    return df, bad


def ingest(batch_size=4, source=SUMMER_FILE):
    """Simulates new data arriving: moves the next few summer scenarios into recent_data.csv."""
    params = load_params()
    data = pd.read_csv(source)
    done = set(pd.read_csv(RECENT_FILE, usecols=["scenario_id"])["scenario_id"]) if RECENT_FILE.exists() else set()
    todo = [s for s in data["scenario_id"].unique() if s not in done][:batch_size]
    if not todo:
        log.info("no new data left to ingest")
        return []
    batch, bad = validate(data[data["scenario_id"].isin(todo)], params)
    batch.to_csv(RECENT_FILE, mode="a", header=not RECENT_FILE.exists(), index=False)
    log.info("ingested %s (%d rows, %d bad values fixed)", todo, len(batch), bad)
    return todo


# ------------------------------------------------------------------ feature table (for Feast)
def build_feature_table():
    params = load_params()
    df, _ = validate(pd.read_csv(TRAIN_FILE, parse_dates=["timestamp"]), params)
    pcols = [f"P_{n}" for n in params["sensors"]["pressure"]]
    out = pd.DataFrame({
        "scenario_id": df["scenario_id"],
        "event_timestamp": df["timestamp"].dt.tz_localize("UTC"),
        "net_inflow": df[[f"F_{p}" for p in params["sensors"]["inflow"]]].sum(axis=1),
        "pressure_mean": df[pcols].mean(axis=1),
        "pressure_min": df[pcols].min(axis=1),
    })
    out["created"] = pd.Timestamp.now(tz="UTC")
    path = DATA / "features" / "sensor_features.parquet"
    path.parent.mkdir(exist_ok=True)
    out.to_parquet(path, index=False)
    log.info("feature table saved to %s", path)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "simulate"
    if command == "simulate":
        simulate_all()
    elif command == "ingest":
        ingest()
    elif command == "features":
        build_feature_table()
    elif command == "reset":
        RECENT_FILE.unlink(missing_ok=True)
        log.info("ingested data removed")
    else:
        print("usage: python -m src.data [simulate|ingest|features|reset]")
