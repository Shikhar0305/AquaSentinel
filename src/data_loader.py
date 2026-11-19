
from pathlib import Path

import numpy as np
import pandas as pd
import wntr


ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

RAW_FILE = RAW_DIR / "wntr_sensor_readings.csv"
PROCESSED_FILE = PROCESSED_DIR / "zone_inflow.csv"

# EPANET Net3 is used as one supply zone (DMA)
NETWORK = "Net3"

# Pipes leaving the 2 sources and the 3 service reservoirs.
# Their total flow = all water entering the zone.
INFLOW_PIPES = ["10", "60", "20", "40", "50"]

# Intermittent supply, as in most Indian cities:
# the zone valve is open 5-8 am and 5-7 pm
SUPPLY_WINDOWS = [(5, 8), (17, 19)]

DAYS = 3                 # the first day is dropped (tanks still settling)
STEP_SECONDS = 900       # one reading every 15 minutes


def supply_schedule(rng):
    """
    Demand multiplier and valve status for every 15-minute step.

    - Supply rarely starts on time (up to 45 minutes early or late).
    - When the valve opens, households fill sumps and overhead tanks,
      so demand is high at first and falls as the tanks fill.
    - While the valve is closed, very little water is used.
    """
    steps_per_day = 96
    demand = np.zeros(DAYS * steps_per_day)
    valve = np.zeros(DAYS * steps_per_day, dtype=int)

    for day in range(DAYS):
        day_start = day * steps_per_day
        off_supply_use = 0.08 * rng.uniform(0.7, 1.3)
        demand[day_start:day_start + steps_per_day] = off_supply_use * rng.uniform(0.9, 1.1, steps_per_day)

        for start_hour, end_hour in SUPPLY_WINDOWS:
            start = day_start + start_hour * 4 + int(rng.integers(-3, 4))
            length = (end_hour - start_hour) * 4
            tank_filling = np.linspace(3.0, 1.6, length) * rng.uniform(0.85, 1.15)

            demand[start:start + length] = tank_filling
            valve[start:start + length] = 1

    return demand, valve


def minutes_since_switch(valve):
    """How long the valve has been open (or closed), in minutes."""
    minutes = np.zeros(len(valve), dtype=int)

    for i in range(1, len(valve)):
        if valve[i] == valve[i - 1]:
            minutes[i] = minutes[i - 1] + 15

    return minutes


def simulate_scenario(scenario_id, has_leak, rng):
    wn = wntr.network.WaterNetworkModel(NETWORK)

    wn.options.time.duration = DAYS * 24 * 3600
    wn.options.time.hydraulic_timestep = STEP_SECONDS
    wn.options.time.report_timestep = STEP_SECONDS
    wn.options.time.pattern_timestep = STEP_SECONDS
    wn.options.hydraulic.demand_model = "PDD"

    demand, valve = supply_schedule(rng)
    wn.add_pattern("supply", list(demand))

    for _, junction in wn.junctions():
        for d in junction.demand_timeseries_list:
            d.pattern_name = "supply"

    leak_node, leak_area, leak_start = "", 0.0, None

    if has_leak:
        leak_node = str(rng.choice(wn.junction_name_list))
        leak_area = float(rng.uniform(0.0005, 0.003))        # hole size in m^2
        leak_start = int(rng.integers(30, 54)) * 3600        # during day 2
        wn.get_node(leak_node).add_leak(wn, area=leak_area, start_time=leak_start)

    results = wntr.sim.WNTRSimulator(wn).run_sim()

    flows = results.link["flowrate"][INFLOW_PIPES] * 1000    # m3/s -> L/s
    flows = flows + rng.normal(0, 1.0, flows.shape)          # flow meter noise

    data = pd.DataFrame({
        "scenario_id": scenario_id,
        "timestamp": flows.index.astype(int),
    })

    for pipe in INFLOW_PIPES:
        data[f"flow_pipe_{pipe}"] = flows[pipe].values.round(3)

    step = np.minimum(data["timestamp"] // STEP_SECONDS, len(valve) - 1)
    data["valve_open"] = valve[step]
    data["valve_minutes"] = minutes_since_switch(valve)[step]

    data["leak"] = 0
    if has_leak:
        data.loc[data["timestamp"] >= leak_start, "leak"] = 1

    data["leak_node"] = leak_node
    data["leak_area"] = leak_area

    # drop day 1 while the tanks settle
    return data[data["timestamp"] >= 24 * 3600].reset_index(drop=True)


def generate_raw_data(num_normal=12, num_leak=24, seed=42):
    rng = np.random.default_rng(seed)
    frames = []

    plan = [False] * num_normal + [True] * num_leak

    for i, has_leak in enumerate(plan):
        scenario_id = f"zone-{i:03d}"
        frames.append(simulate_scenario(scenario_id, has_leak, rng))
        print(f"Simulated {scenario_id} ({'leak' if has_leak else 'normal'})")

    raw = pd.concat(frames, ignore_index=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_FILE, index=False)

    print(f"Raw data saved to: {RAW_FILE}")

    return raw


def load_dataset(save=True):
    if RAW_FILE.exists():
        raw = pd.read_csv(RAW_FILE, dtype={"leak_node": str})
    else:
        raw = generate_raw_data()

    dataset = raw[["scenario_id", "timestamp", "valve_open", "valve_minutes"]].copy()

    # water balance: total water entering the zone
    dataset["inflow_lps"] = raw[[f"flow_pipe_{p}" for p in INFLOW_PIPES]].sum(axis=1).round(3)

    dataset["leak"] = raw["leak"]
    dataset["leak_node"] = raw["leak_node"].fillna("")
    dataset["leak_area"] = raw["leak_area"]

    dataset = dataset.sort_values(["scenario_id", "timestamp"]).reset_index(drop=True)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(PROCESSED_FILE, index=False)
        print(f"Processed dataset saved to: {PROCESSED_FILE}")

    return dataset


if __name__ == "__main__":
    data = load_dataset()

    print("\nDataset shape:", data.shape)
    print("\nColumns:", data.columns.tolist())
    print("\nScenario count:", data["scenario_id"].nunique())
    print("\nPreview:")
    print(data.head())
