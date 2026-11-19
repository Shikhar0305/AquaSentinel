"""Paths, settings and small helpers used by every other file."""
import hashlib
import json
import logging
import os
from pathlib import Path

import yaml

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
METRICS = ROOT / "metrics"
LOGS = ROOT / "logs"
REPORTS = ROOT / "reports"

TRAIN_FILE = DATA / "raw" / "sensor_data.csv"     # winter data
SUMMER_FILE = DATA / "raw" / "summer_data.csv"    # data with higher demand
RECENT_FILE = DATA / "incoming" / "recent_data.csv"  # data that arrived after deployment

for folder in (DATA / "raw", DATA / "incoming", MODELS, METRICS, LOGS, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)


def load_params():
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def sensor_columns(params):
    """P_<junction> = pressure in m, F_<pipe> = flow in L/s."""
    return ([f"P_{n}" for n in params["sensors"]["pressure"]]
            + [f"F_{n}" for n in params["sensors"]["flow"]])


def tracking_uri():
    # local file database by default, the MLflow server inside docker compose
    return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")


def file_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def relative(path):
    """Paths are stored relative to the project so they work on any computer."""
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def get_logger(name):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
    return logging.getLogger(name)
