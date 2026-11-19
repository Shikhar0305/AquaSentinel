"""Small helpers shared by all the scripts."""
import hashlib
import json
import logging
import os
from pathlib import Path

import yaml

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
METRICS_DIR = ROOT / "metrics"
LOGS_DIR = ROOT / "logs"

for d in (DATA_DIR / "raw", DATA_DIR / "processed", MODELS_DIR, REPORTS_DIR, METRICS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_params(path=ROOT / "params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def sensor_columns(params):
    """Column names used for sensors, e.g. P_123 (pressure, m) and F_20 (flow, L/s)."""
    p = [f"P_{n}" for n in params["sensors"]["pressure"]]
    f = [f"F_{n}" for n in params["sensors"]["flow"]]
    return p + f


def file_md5(path):
    """Hash of a data file - stored with every model so we know which data trained it."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def get_tracking_uri():
    # local sqlite by default, http://mlflow:5000 inside docker compose
    return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
