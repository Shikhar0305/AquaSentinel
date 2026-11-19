"""
Airflow DAGs for AquaSentinel.

1. aquasentinel_training   (manual or weekly)
       simulate_data -> validate_data -> train_and_register

2. aquasentinel_monitoring (every hour)
       ingest_batch -> check_drift -> [retrain_model | no_retrain] -> reload_api

Setup: point Airflow's dags_folder to this folder (or copy this file there) and set
AQUA_HOME to the project root, e.g.  export AQUA_HOME=/opt/aquasentinel
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

AQUA_HOME = os.getenv("AQUA_HOME", str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, AQUA_HOME)

try:  # Airflow 3
    from airflow.providers.standard.operators.empty import EmptyOperator
    from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator
    from airflow.sdk import DAG
except ImportError:  # Airflow 2.x
    from airflow import DAG
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.python import BranchPythonOperator, PythonOperator

default_args = {
    "owner": "aquasentinel",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


# ---------------- task functions (imports inside so the scheduler parses the file fast)
def simulate_data():
    import subprocess
    subprocess.run([sys.executable, "-m", "src.simulate", "--mode", "train"], cwd=AQUA_HOME, check=True)


def validate_data():
    import pandas as pd

    from src.ingest import validate
    from src.utils import DATA_DIR, load_params
    df = pd.read_csv(DATA_DIR / "raw" / "sensor_data.csv")
    _, report = validate(df, load_params())
    if report["out_of_range"] > 0.01 * report["rows"]:
        raise ValueError(f"too many bad sensor values: {report}")
    return report


def train_and_register():
    from src.train import train
    from src.utils import DATA_DIR
    return train(DATA_DIR / "raw" / "sensor_data.csv", run_name="airflow-train", user="airflow")


def ingest_batch():
    from src.ingest import ingest
    return ingest()


def check_drift(**context):
    from src.drift import run_drift_check
    from src.ingest import RECENT_FILE
    from src.retrain import should_retrain
    if not RECENT_FILE.exists():
        return "no_retrain"
    result = run_drift_check()
    retrain, reason = should_retrain(result)
    context["ti"].xcom_push(key="drift", value={"share": result["drift_share"], "reason": reason})
    return "retrain_model" if retrain else "no_retrain"


def retrain_model():
    import shutil

    from src.ingest import RECENT_FILE
    from src.retrain import SNAPSHOT_DIR
    from src.train import train
    snapshot = SNAPSHOT_DIR / f"retrain_{datetime.now():%Y%m%d_%H%M%S}.csv"
    shutil.copy(RECENT_FILE, snapshot)
    return train(snapshot, run_name="airflow-retrain", user="airflow-drift")


def reload_api():
    import requests
    url = os.getenv("API_URL", "http://localhost:8000")
    key = os.getenv("AQUA_ADMIN_KEY", "admin-dev-key")
    try:
        r = requests.post(f"{url}/model/reload", headers={"X-API-Key": key}, timeout=10)
        return r.json()
    except requests.RequestException as e:
        # the API also reloads by itself when models/production changes, so this is not fatal
        return {"warning": str(e)}


# ---------------- DAG 1: training
with DAG(
    dag_id="aquasentinel_training",
    description="Simulate data, validate it, train and register the leak detector",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    default_args=default_args,
    tags=["aquasentinel", "training"],
) as training_dag:
    t1 = PythonOperator(task_id="simulate_data", python_callable=simulate_data)
    t2 = PythonOperator(task_id="validate_data", python_callable=validate_data)
    t3 = PythonOperator(task_id="train_and_register", python_callable=train_and_register)
    t4 = PythonOperator(task_id="reload_api", python_callable=reload_api)
    t1 >> t2 >> t3 >> t4


# ---------------- DAG 2: monitoring + auto retraining
with DAG(
    dag_id="aquasentinel_monitoring",
    description="Ingest new sensor data, check drift, retrain when needed",
    start_date=datetime(2026, 1, 1),
    schedule="@hourly",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["aquasentinel", "monitoring"],
) as monitoring_dag:
    ingest = PythonOperator(task_id="ingest_batch", python_callable=ingest_batch)
    drift = BranchPythonOperator(task_id="check_drift", python_callable=check_drift)
    retrain = PythonOperator(task_id="retrain_model", python_callable=retrain_model)
    skip = EmptyOperator(task_id="no_retrain")
    reload_ = PythonOperator(task_id="reload_api", python_callable=reload_api,
                             trigger_rule="none_failed_min_one_success")
    ingest >> drift >> [retrain, skip]
    retrain >> reload_
