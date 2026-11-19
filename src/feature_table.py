"""
Builds the feature table that the Feast feature store serves
(data/features/sensor_features.parquet).

One row per (scenario_id, event_timestamp). In a real deployment scenario_id would be
the district / zone id. The heavy part of the features (baseline residuals) belongs
to the model, so the store keeps the cleaned sensor signals the model is built from.

Usage:  python -m src.feature_table
Then:   cd feature_repo && feast apply && feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
"""
import pandas as pd

from src.features import add_water_balance
from src.utils import DATA_DIR, get_logger, load_params

log = get_logger("feature_table")
OUT = DATA_DIR / "features" / "sensor_features.parquet"


def build(source=DATA_DIR / "raw" / "sensor_data.csv"):
    params = load_params()
    df = pd.read_csv(source, parse_dates=["timestamp"])
    df = add_water_balance(df, params["sensors"]["inflow"])
    pcols = [f"P_{n}" for n in params["sensors"]["pressure"]]
    df["pressure_mean"] = df[pcols].mean(axis=1)
    df["pressure_min"] = df[pcols].min(axis=1)

    out = df[["scenario_id", "timestamp", "net_inflow", "pressure_mean", "pressure_min"] + pcols].copy()
    out = out.rename(columns={"timestamp": "event_timestamp"})
    out["event_timestamp"] = pd.to_datetime(out["event_timestamp"]).dt.tz_localize("UTC")
    out["created"] = pd.Timestamp.now(tz="UTC")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    log.info("wrote %d rows to %s", len(out), OUT)
    return out


if __name__ == "__main__":
    build()
