"""Evaluate a champion model, retrain a candidate, and record a quality decision."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.preprocessing import FEATURES, TARGET, create_features, validate_data


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT_DIR / "data" / "processed" / "zone_inflow.csv"
DEFAULT_INCOMING = ROOT_DIR / "data" / "incoming" / "new_batch.csv"
DEFAULT_CHAMPION = ROOT_DIR / "models" / "champion.joblib"
DEFAULT_CANDIDATE = ROOT_DIR / "models" / "candidate.joblib"
DEFAULT_REPORT = ROOT_DIR / "reports" / "evaluation_report.json"
DEFAULT_DECISION = ROOT_DIR / "reports" / "quality_gate.json"

REQUIRED_COLUMNS = [
    "scenario_id",
    "timestamp",
    "valve_open",
    "valve_minutes",
    "inflow_lps",
    TARGET,
]


def validate_incoming_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate a labelled incoming batch before feature engineering."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Missing values found in required columns")

    if data.duplicated(["scenario_id", "timestamp"]).any():
        raise ValueError("Duplicate scenario/timestamp records found")

    return validate_data(data.copy())


def _select_demo_scenarios(data: pd.DataFrame) -> list[str]:
    scenario_labels = data.groupby("scenario_id")[TARGET].max()
    selected = []
    for label in (0, 1):
        ids = sorted(scenario_labels[scenario_labels == label].index)
        selected.extend(ids[: max(1, round(len(ids) * 0.2))])
    return selected


def load_incoming_batch(
    source: Path = DEFAULT_DATA,
    incoming_path: Path = DEFAULT_INCOMING,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load an incoming batch, creating a reproducible demo batch if needed."""
    full_data = pd.read_csv(source)
    if incoming_path.exists():
        incoming = pd.read_csv(incoming_path)
    else:
        selected = _select_demo_scenarios(full_data)
        incoming = full_data[full_data["scenario_id"].isin(selected)].copy()
        incoming_path.parent.mkdir(parents=True, exist_ok=True)
        incoming.to_csv(incoming_path, index=False)
    return validate_incoming_data(incoming), full_data


def evaluate_model(model, featured_data: pd.DataFrame) -> dict[str, float]:
    predictions = model.predict(featured_data[FEATURES])
    labels = featured_data[TARGET]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def _artifact_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def track_candidate_with_mlflow(model, metrics: dict[str, float], rows: int) -> str:
    """Track the candidate when MLflow is installed; otherwise report that it is unavailable."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        return "unavailable: install requirements.txt to enable MLflow tracking"

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("AquaSentinel-Leak-Detection")
    with mlflow.start_run(run_name="monitor-candidate") as run:
        mlflow.log_param("training_rows", rows)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            name="candidate_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        return run.info.run_id


def run_monitoring(
    incoming_path: Path = DEFAULT_INCOMING,
    data_path: Path = DEFAULT_DATA,
    champion_path: Path = DEFAULT_CHAMPION,
    candidate_path: Path = DEFAULT_CANDIDATE,
    report_path: Path = DEFAULT_REPORT,
    decision_path: Path = DEFAULT_DECISION,
) -> dict:
    incoming, full_data = load_incoming_batch(data_path, incoming_path)
    incoming_features = create_features(incoming)
    training_data = full_data[
        ~full_data["scenario_id"].isin(incoming["scenario_id"].unique())
    ]
    combined_features = create_features(
        pd.concat([training_data, incoming], ignore_index=True)
    )

    champion = joblib.load(champion_path)
    champion_metrics = evaluate_model(champion, incoming_features)

    candidate = clone(champion)
    candidate.fit(combined_features[FEATURES], combined_features[TARGET])
    candidate_metrics = evaluate_model(candidate, incoming_features)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, candidate_path)

    eligible = candidate_metrics["f1"] >= champion_metrics["f1"]
    decision = {
        "decision": "eligible_for_promotion" if eligible else "retain_champion",
        "metric": "f1",
        "rule": "candidate F1 must be greater than or equal to champion F1",
        "champion_model": _artifact_label(champion_path),
        "candidate_model": _artifact_label(candidate_path),
        "champion_metrics": champion_metrics,
        "candidate_metrics": candidate_metrics,
        "champion_overwritten": False,
    }
    report = {
        "incoming_batch": {
            "path": _artifact_label(incoming_path),
            "rows": int(len(incoming)),
            "scenarios": int(incoming["scenario_id"].nunique()),
            "target_distribution": {
                str(int(label)): int(count)
                for label, count in incoming[TARGET].value_counts().sort_index().items()
            },
        },
        "validation": "passed",
        "evaluation": {
            "accuracy": champion_metrics["accuracy"],
            "precision": champion_metrics["precision"],
            "recall": champion_metrics["recall"],
            "f1": champion_metrics["f1"],
        },
        "retraining": {
            "combined_rows": int(len(combined_features)),
            "candidate_artifact": _artifact_label(candidate_path),
            "mlflow_run_id": track_candidate_with_mlflow(
                candidate, candidate_metrics, len(combined_features)
            ),
        },
        "quality_gate": decision,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming", type=Path, default=DEFAULT_INCOMING)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    run_monitoring(incoming_path=args.incoming, data_path=args.data)


if __name__ == "__main__":
    main()
