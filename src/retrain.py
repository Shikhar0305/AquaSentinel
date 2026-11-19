"""
Monitoring cycle: ingest -> drift check -> retrain (if needed) -> promote or keep.

This is what the Airflow DAG `aquasentinel_drift_retrain` runs every few hours.
It can also be run by hand:

    python -m src.retrain            # one cycle (ingests the next batch)
    python -m src.retrain --all      # ingest all remaining data, then check

Labels for the new data: in a real utility, leak labels come from field crews
confirming a leak (repair tickets) - that is our feedback loop. The model itself
trains only on normal readings; labels are used to compare the new model with
the current champion before promotion.
"""
import argparse
import shutil
from datetime import datetime

import pandas as pd

from src.drift import run_drift_check
from src.ingest import RECENT_FILE, ingest
from src.train import train
from src.utils import DATA_DIR, METRICS_DIR, get_logger, load_params, save_json

log = get_logger("retrain")
SNAPSHOT_DIR = DATA_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def should_retrain(drift_result, recent_path=RECENT_FILE):
    params = load_params()
    if not drift_result["drift_detected"]:
        return False, "no drift"
    n = pd.read_csv(recent_path, usecols=["scenario_id"])["scenario_id"].nunique()
    need = params["drift"]["min_scenarios_for_retrain"]
    if n < need:
        return False, f"drift detected but only {n}/{need} scenarios of new data - waiting for more"
    return True, f"drift share {drift_result['drift_share']:.2f}"


def run_cycle(take_all=False):
    batch = ingest(take_all=take_all)
    if batch is None and not RECENT_FILE.exists():
        log.info("no data yet")
        return {"status": "no data"}

    drift = run_drift_check()
    retrain, reason = should_retrain(drift)
    result = {"drift_share": drift["drift_share"], "drift_detected": drift["drift_detected"],
              "retrained": retrain, "reason": reason}

    if retrain:
        log.info("retraining: %s", reason)
        # freeze a copy of the data so the new model always points to the exact rows it saw
        snapshot = SNAPSHOT_DIR / f"retrain_{datetime.now():%Y%m%d_%H%M%S}.csv"
        shutil.copy(RECENT_FILE, snapshot)
        train_result = train(snapshot, run_name="retrain-drift", user="drift-monitor")
        result.update({"new_version": train_result["version"], "decision": train_result["decision"],
                       "new_f1": train_result["f1"]})
    else:
        log.info("no retraining: %s", reason)

    save_json(result, METRICS_DIR / "last_cycle.json")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="ingest all remaining data first")
    args = parser.parse_args()
    print(run_cycle(args.all))


if __name__ == "__main__":
    main()
