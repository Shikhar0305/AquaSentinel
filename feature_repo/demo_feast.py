"""
Shows the two ways the feature store is used.

1. Offline (training): point-in-time correct features for a list of
   (zone, timestamp) rows - the same call works for any past date.
2. Online (serving): latest features for a zone in milliseconds.

Run from the project root after `feast apply` and `feast materialize`:
    python feature_repo/demo_feast.py
"""
from pathlib import Path

import pandas as pd
from feast import FeatureStore

REPO = Path(__file__).resolve().parent
store = FeatureStore(repo_path=str(REPO))
FEATURES = ["sensor_features:net_inflow", "sensor_features:pressure_mean", "sensor_features:pressure_min"]

# --- offline: build a small training set
entity_df = pd.DataFrame({
    "scenario_id": ["winter_000", "winter_000", "winter_020"],
    "event_timestamp": pd.to_datetime(["2026-01-02 06:00", "2026-01-02 18:00", "2026-01-03 03:00"], utc=True),
})
training = store.get_historical_features(entity_df=entity_df, features=FEATURES).to_df()
print("Offline features (training):")
print(training.to_string(index=False))

# --- online: what the API would see right now
online = store.get_online_features(features=FEATURES,
                                   entity_rows=[{"scenario_id": "winter_000"}, {"scenario_id": "winter_020"}]).to_df()
print("\nOnline features (serving):")
print(online.to_string(index=False))
