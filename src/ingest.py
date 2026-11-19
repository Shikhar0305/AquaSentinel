"""
Data ingestion: picks up the next batch of sensor readings, validates it and
appends it to data/processed/recent_data.csv.

In a real utility the batch would come from SCADA / an MQTT or Kafka topic.
Here we "stream" a simulated file (default: the summer drift data) batch by batch,
remembering how far we got in data/processed/ingest_state.json.

Usage:
    python -m src.ingest                       # next batch of 4 scenarios
    python -m src.ingest --all                 # everything that is left
    python -m src.ingest --reset               # start again from the beginning
"""
import argparse
import json

import pandas as pd

from src.utils import DATA_DIR, get_logger, load_params, sensor_columns

log = get_logger("ingest")
STATE_FILE = DATA_DIR / "processed" / "ingest_state.json"
RECENT_FILE = DATA_DIR / "processed" / "recent_data.csv"

# simple physical limits - anything outside is a broken sensor, not a leak
LIMITS = {"P_": (-20, 200), "F_": (-5000, 5000)}


def validate(df, params):
    cols = sensor_columns(params)
    missing = [c for c in ["timestamp", "scenario_id"] + cols if c not in df.columns]
    if missing:
        raise ValueError(f"batch is missing columns: {missing}")

    report = {"rows": len(df), "nan_filled": int(df[cols].isna().sum().sum()), "out_of_range": 0}
    for prefix, (lo, hi) in LIMITS.items():
        for c in [c for c in cols if c.startswith(prefix)]:
            bad = (df[c] < lo) | (df[c] > hi)
            report["out_of_range"] += int(bad.sum())
            df.loc[bad, c] = None
    # short gaps (lost packets) -> interpolate within each scenario
    df[cols] = df.groupby("scenario_id")[cols].transform(lambda s: s.interpolate(limit_direction="both"))
    if df[cols].isna().any().any():
        raise ValueError("batch still has missing values after cleaning")
    return df, report


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"source": None, "done_scenarios": []}


def ingest(source=DATA_DIR / "raw" / "drift_data.csv", n_scenarios=4, take_all=False):
    params = load_params()
    state = load_state()
    if state["source"] != str(source):
        state = {"source": str(source), "done_scenarios": []}
        RECENT_FILE.unlink(missing_ok=True)

    data = pd.read_csv(source)
    todo = [s for s in data["scenario_id"].unique() if s not in state["done_scenarios"]]
    if not todo:
        log.info("nothing new to ingest")
        return None
    batch_ids = todo if take_all else todo[:n_scenarios]
    batch = data[data["scenario_id"].isin(batch_ids)].copy()

    batch, report = validate(batch, params)
    batch.to_csv(RECENT_FILE, mode="a", header=not RECENT_FILE.exists(), index=False)

    state["done_scenarios"] += list(batch_ids)
    STATE_FILE.write_text(json.dumps(state, indent=2))
    log.info("ingested %d scenarios (%d rows), report=%s", len(batch_ids), len(batch), report)
    return {"scenarios": list(batch_ids), **report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DATA_DIR / "raw" / "drift_data.csv"))
    parser.add_argument("--n", type=int, default=4, help="scenarios per batch")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    if args.reset:
        STATE_FILE.unlink(missing_ok=True)
        RECENT_FILE.unlink(missing_ok=True)
    ingest(args.source, args.n, args.all)


if __name__ == "__main__":
    main()
