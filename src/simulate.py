"""
Step 1 - Generate labelled sensor data with WNTR.

Real utility sensor data is not public, so we simulate the EPANET Net3 network.
Each scenario runs for a few days with 15-minute readings. In leak scenarios we
add a leak at a random junction on day 2 and label every reading after the leak
starts as 1.

Usage:
    python -m src.simulate --mode train   -> data/raw/sensor_data.csv
    python -m src.simulate --mode drift   -> data/raw/drift_data.csv  (summer demand)
"""
import argparse

import numpy as np
import pandas as pd
import wntr

from src.utils import DATA_DIR, get_logger, load_params

log = get_logger("simulate")
START_DATE = pd.Timestamp("2026-01-01")


def build_network(sim, demand_multiplier, rng):
    wn = wntr.network.WaterNetworkModel(sim["network"])
    step = sim["timestep_min"] * 60
    wn.options.time.duration = sim["days"] * 24 * 3600
    wn.options.time.hydraulic_timestep = step
    wn.options.time.report_timestep = step
    wn.options.time.pattern_timestep = 3600
    # pressure dependent demand: needed so that leaks behave realistically
    wn.options.hydraulic.demand_model = "PDD"

    # every scenario gets a slightly different demand level
    factor = demand_multiplier * (1 + rng.uniform(-sim["demand_variation"], sim["demand_variation"]))
    for _, junction in wn.junctions():
        for demand in junction.demand_timeseries_list:
            demand.base_value *= factor
    return wn, factor


def run_scenario(params, scenario_id, has_leak, demand_multiplier, rng):
    sim = params["simulation"]
    wn, factor = build_network(sim, demand_multiplier, rng)

    leak_node, leak_area, leak_start = None, 0.0, None
    if has_leak:
        leak_node = str(rng.choice(wn.junction_name_list))
        leak_area = float(rng.uniform(sim["leak_area_min"], sim["leak_area_max"]))
        hour = int(rng.integers(sim["leak_start_hour_min"], sim["leak_start_hour_max"]))
        leak_start = hour * 3600
        wn.get_node(leak_node).add_leak(wn, area=leak_area, start_time=leak_start)

    results = wntr.sim.WNTRSimulator(wn).run_sim()

    pressure = results.node["pressure"][params["sensors"]["pressure"]]
    flow = results.link["flowrate"][params["sensors"]["flow"]] * 1000  # m3/s -> L/s

    df = pd.DataFrame(index=pressure.index)
    for col in pressure.columns:
        df[f"P_{col}"] = pressure[col] + rng.normal(0, sim["noise_pressure_m"], len(df))
    for col in flow.columns:
        df[f"F_{col}"] = flow[col] + rng.normal(0, sim["noise_flow_lps"], len(df))

    df["timestamp"] = START_DATE + pd.to_timedelta(df.index, unit="s")
    df["scenario_id"] = scenario_id
    df["demand_factor"] = round(factor, 3)
    df["leak_node"] = leak_node or ""
    df["leak_area"] = leak_area
    df["leak"] = 0
    if has_leak:
        df.loc[df.index >= leak_start, "leak"] = 1
    sensor_cols = [c for c in df.columns if c.startswith(("P_", "F_"))]
    df[sensor_cols] = df[sensor_cols].round(3)

    # drop the warm-up period (tank levels are still settling)
    df = df[df.index >= sim["warmup_hours"] * 3600]
    return df.reset_index(drop=True)


def simulate(params, n_normal, n_leak, demand_multiplier, seed, prefix):
    rng = np.random.default_rng(seed)
    frames = []
    plan = [False] * n_normal + [True] * n_leak
    for i, has_leak in enumerate(plan):
        sid = f"{prefix}_{i:03d}"
        df = run_scenario(params, sid, has_leak, demand_multiplier, rng)
        frames.append(df)
        info = f"leak at {df['leak_node'].iloc[0]}" if has_leak else "normal"
        log.info("scenario %s done (%s)", sid, info)
    data = pd.concat(frames, ignore_index=True)
    first = ["timestamp", "scenario_id"]
    return data[first + [c for c in data.columns if c not in first]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "drift"], default="train")
    args = parser.parse_args()
    params = load_params()

    if args.mode == "train":
        s = params["simulation"]
        data = simulate(params, s["n_normal_scenarios"], s["n_leak_scenarios"], 1.0, s["seed"], "base")
        out = DATA_DIR / "raw" / "sensor_data.csv"
    else:
        d = params["drift"]
        data = simulate(params, d["n_normal_scenarios"], d["n_leak_scenarios"],
                        d["demand_multiplier"], d["seed"], "summer")
        out = DATA_DIR / "raw" / "drift_data.csv"

    data.to_csv(out, index=False)
    log.info("saved %d rows (%d leak readings) to %s", len(data), data["leak"].sum(), out)


if __name__ == "__main__":
    main()
