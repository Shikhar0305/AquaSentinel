"""
Airflow DAGs.

aquasentinel_training   (weekly):  simulate_data -> train_model
aquasentinel_monitoring (hourly):  monitoring_cycle  (ingest -> drift check -> retrain if needed)

Set AQUA_HOME to the project folder and point Airflow's dags_folder here.
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.getenv("AQUA_HOME", str(Path(__file__).resolve().parent.parent)))

try:  # Airflow 3
    from airflow.providers.standard.operators.python import PythonOperator
    from airflow.sdk import DAG
except ImportError:  # Airflow 2
    from airflow import DAG
    from airflow.operators.python import PythonOperator


def simulate_data():
    from src.data import simulate_all
    simulate_all()


def train_model():
    from src.train import train
    return train(user="airflow")


def monitoring_cycle():
    from src.monitor import run_cycle
    return run_cycle()   # the API notices a new production model by itself


defaults = {"owner": "aquasentinel", "retries": 1, "retry_delay": timedelta(minutes=2)}

with DAG("aquasentinel_training", start_date=datetime(2026, 1, 1), schedule="@weekly",
         catchup=False, default_args=defaults, tags=["aquasentinel"]):
    PythonOperator(task_id="simulate_data", python_callable=simulate_data) >> \
        PythonOperator(task_id="train_model", python_callable=train_model)

with DAG("aquasentinel_monitoring", start_date=datetime(2026, 1, 1), schedule="@hourly",
         catchup=False, max_active_runs=1, default_args=defaults, tags=["aquasentinel"]):
    PythonOperator(task_id="monitoring_cycle", python_callable=monitoring_cycle)
