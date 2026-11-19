"""
Monitoring cycle: ingest new data -> check drift -> retrain if needed.

Drift check (Evidently): we compare the new data with the data the production
model was trained on, using PSI (population stability index), for every sensor
and for the supply timetable (valve open/closed and for how long).

Retraining is needed when
  - the supply timetable changed (e.g. summer rationing: one supply a day instead of two), or
  - at least half of the sensors drifted (e.g. a new pipeline or pump).

    python -m src.monitor          # run one cycle (Airflow runs this every hour)

In a real utility the leak labels for new data come from repair tickets
(field crews confirming leaks). That is our feedback loop.
"""
import json
import shutil
from datetime import datetime

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import DATA, METRICS, MODELS, RECENT_FILE, REPORTS, ROOT, TRAIN_FILE
from src.config import get_logger, load_params, relative, save_json, sensor_columns
from src.data import ingest, validate

log = get_logger("monitor")


SCHEDULE = ["valve_open", "valve_minutes"]


def check_drift(reference, current, sensors, psi_limit=0.25, share_limit=0.5, html=None):
    columns = sensors + [c for c in SCHEDULE if c in reference and c in current]
    ref = reference[columns].sample(min(1000, len(reference)), random_state=0)
    cur = current[columns].sample(min(1000, len(current)), random_state=0)
    snapshot = Report([DataDriftPreset(method="psi", threshold=psi_limit)]).run(cur, ref)
    psi = {m["config"]["column"]: round(float(m["value"]), 3)
           for m in snapshot.dict()["metrics"] if m["config"]["type"].endswith("ValueDrift")}
    drifted = [c for c in sensors if psi[c] > psi_limit]
    schedule_changed = any(psi[c] > psi_limit for c in SCHEDULE if c in psi)
    share = len(drifted) / len(sensors)
    if html:
        snapshot.save_html(str(html))
    return {"psi": psi, "drifted": drifted, "drift_share": round(share, 3), "schedule_changed": schedule_changed,
            "drift_detected": schedule_changed or share >= share_limit}


def production_training_file():
    info = MODELS / "production" / "info.json"
    if info.exists():
        return ROOT / json.loads(info.read_text())["data_file"]
    return TRAIN_FILE


def run_cycle():
    params = load_params()
    m = params["monitoring"]
    ingest()
    if not RECENT_FILE.exists():
        return {"status": "no data"}

    reference, _ = validate(pd.read_csv(production_training_file()), params)
    reference = reference[reference["leak"] == 0]          # normal behaviour the model knows
    recent = pd.read_csv(RECENT_FILE)                        # everything that came in (already cleaned)
    result = check_drift(reference, recent, sensor_columns(params), m["psi_limit"], m["drift_share"],
                         html=REPORTS / "drift_report.html")
    result.update({"time": datetime.now().isoformat(timespec="seconds"),
                   "reference": relative(production_training_file()), "current": relative(RECENT_FILE)})
    log.info("sensor drift %.2f, timetable changed: %s -> drift: %s",
             result["drift_share"], result["schedule_changed"], result["drift_detected"])

    scenarios = recent["scenario_id"].nunique()
    if not result["drift_detected"]:
        result["action"] = "no retraining needed"
    elif scenarios < m["min_scenarios"]:
        result["action"] = f"waiting for more data ({scenarios}/{m['min_scenarios']} scenarios)"
    else:
        # freeze a copy so the new model always points to the exact data it saw
        snapshot = DATA / "snapshots" / f"retrain_{datetime.now():%Y%m%d_%H%M%S}.csv"
        snapshot.parent.mkdir(exist_ok=True)
        shutil.copy(RECENT_FILE, snapshot)
        from src.train import train
        trained = train(snapshot, user="drift-monitor")
        result["action"] = f"retrained v{trained['version']}: {trained['decision']}"

    log.info(result["action"])
    save_json(result, METRICS / "drift.json")
    return result


if __name__ == "__main__":
    run_cycle()
