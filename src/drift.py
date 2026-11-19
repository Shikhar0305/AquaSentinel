"""
Data drift detection with Evidently.

Reference data = the data the current production model was trained on
(we know which file from the model lineage / meta.json).
Current data   = the most recent ingested readings.

Evidently computes the PSI (population stability index) for every sensor column.
A column with PSI above the limit counts as drifted. If the share of drifted
columns is above the threshold in params.yaml, the model should be retrained.
(We first tried the default K-S test, but with thousands of time-correlated
readings it flagged almost every column even for normal data.)

Usage:
    python -m src.drift
"""
import json

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.ingest import RECENT_FILE
from src.utils import METRICS_DIR, MODELS_DIR, REPORTS_DIR, ROOT, get_logger, load_params, save_json, sensor_columns

log = get_logger("drift")
SAMPLE_SIZE = 1000  # enough for the tests, keeps the report fast


def reference_file():
    """Training data of the model currently in production (stored in its meta.json)."""
    meta_file = MODELS_DIR / "production" / "meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        if meta.get("data_file"):
            return ROOT / meta["data_file"]
    return ROOT / "data" / "raw" / "sensor_data.csv"


def check_drift(reference, current, columns, threshold, psi_threshold=0.5, html_path=None):
    ref = reference[columns].sample(min(SAMPLE_SIZE, len(reference)), random_state=0)
    cur = current[columns].sample(min(SAMPLE_SIZE, len(current)), random_state=0)
    preset = DataDriftPreset(method="psi", threshold=psi_threshold)
    snapshot = Report([preset]).run(current_data=cur, reference_data=ref)

    per_column = {}
    for m in snapshot.dict()["metrics"]:
        cfg = m["config"]
        if cfg["type"].endswith("ValueDrift"):
            per_column[cfg["column"]] = round(float(m["value"]), 3)
    drifted = sum(v > psi_threshold for v in per_column.values())
    share = drifted / len(columns)

    if html_path:
        snapshot.save_html(str(html_path))
    return {
        "drift_share": round(share, 3),
        "drifted_columns": int(drifted),
        "total_columns": len(columns),
        "drift_detected": bool(share >= threshold),
        "psi": per_column,
    }


def run_drift_check(current_path=RECENT_FILE):
    params = load_params()
    # reference = normal readings the model learned from.
    # current = everything that came in (in real life we don't know which rows are leaks)
    reference = pd.read_csv(reference_file())
    reference = reference[reference["leak"] == 0]
    current = pd.read_csv(current_path)

    result = check_drift(reference, current, sensor_columns(params),
                         params["drift"]["drift_share_threshold"],
                         params["drift"]["psi_threshold"],
                         html_path=REPORTS_DIR / "drift_report.html")
    result["reference"] = str(reference_file().relative_to(ROOT))
    result["current"] = str(current_path)
    save_json(result, METRICS_DIR / "drift.json")
    log.info("drift share %.2f (%d/%d columns) -> drift_detected=%s",
             result["drift_share"], result["drifted_columns"], result["total_columns"], result["drift_detected"])
    return result


if __name__ == "__main__":
    run_drift_check()
