"""
Train, evaluate and register the leak detector with MLflow.

1. Split whole scenarios into train and test.
2. Fit the model on NORMAL readings of the train scenarios.
3. Evaluate on the test scenarios.
4. Log everything to MLflow and register a new model version.
5. Make it the "champion" (production model) if it beats the current one.

    python -m src.train
"""
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src import registry
from src.config import METRICS, TRAIN_FILE, file_md5, get_logger, load_params, relative, save_json, tracking_uri
from src.data import validate
from src.model import build_model

log = get_logger("train")


def split_by_scenario(df, test_size=0.3, seed=42):
    """Whole scenarios go to train or test, never half and half."""
    rng = np.random.default_rng(seed)
    leak_ids = df.loc[df["leak"] == 1, "scenario_id"].unique()
    normal_ids = np.setdiff1d(df["scenario_id"].unique(), leak_ids)
    test_ids = []
    for ids in (normal_ids, leak_ids):   # keep both kinds in the test set
        ids = rng.permutation(ids)
        test_ids += list(ids[:max(1, round(len(ids) * test_size))])
    is_test = df["scenario_id"].isin(test_ids)
    return df[~is_test], df[is_test]


def evaluate(model, df):
    out = model.predict_details(df)
    y = df["leak"].values
    detected, delays = [], []
    for _, g in df.assign(alert=out["leak_alert"].values).groupby("scenario_id"):
        if g["leak"].any():
            labels, alerts = g["leak"].values, g["alert"].values
            start = labels.argmax()
            hits = np.where((labels == 1) & (alerts == 1))[0]
            detected.append(len(hits) > 0)
            if len(hits):
                delays.append((hits[0] - start) * 15)   # minutes
    return {
        "precision": round(precision_score(y, out["is_anomaly"], zero_division=0), 4),
        "recall": round(recall_score(y, out["is_anomaly"], zero_division=0), 4),
        "f1": round(f1_score(y, out["is_anomaly"], zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, out["anomaly_score"]), 4),
        "leaks_detected": round(float(np.mean(detected)), 4),
        "detection_delay_min": float(np.median(delays)) if delays else -1.0,
        "false_alarm_rate": round(float(out["leak_alert"].values[y == 0].mean()), 4),
    }


def score_for_promotion(metrics, params):
    """ROC-AUC, but a model that raises too many false alarms is useless (score 0)."""
    if metrics["false_alarm_rate"] > params["monitoring"]["max_false_alarms"]:
        return 0.0
    return metrics["roc_auc"]


def train(data_file=TRAIN_FILE, user="pipeline"):
    params = load_params()
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("aquasentinel")

    df, _ = validate(pd.read_csv(data_file), params)   # fills gaps from power cuts
    train_df, test_df = split_by_scenario(df, params["model"]["test_size"], params["model"]["seed"])
    model = build_model(params).fit(train_df[train_df["leak"] == 0])
    metrics = evaluate(model, test_df)
    log.info("test metrics: %s", metrics)

    data_tags = {"data_file": relative(data_file), "data_md5": file_md5(data_file), "trained_by": user}
    with mlflow.start_run(run_name=f"train-{user}"):
        mlflow.log_params(params["model"])
        mlflow.log_metrics(metrics)
        mlflow.set_tags(data_tags)
        # cloudpickle because LeakDetector is our own class
        info = mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle",
                                        registered_model_name=params["model"]["name"])
    version = int(info.registered_model_version)
    registry.tag_version(version, {**data_tags, **metrics})
    registry.log_event("registered", version, user)

    # champion vs challenger on the same test data
    champion = registry.get_alias("champion")
    if champion is None:
        decision = "promoted (first model)"
        registry.promote(version, user, "first model")
    else:
        old = evaluate(registry.load(champion), test_df)
        new_score, old_score = score_for_promotion(metrics, params), score_for_promotion(old, params)
        reason = f"score v{champion}={old_score:.3f}, v{version}={new_score:.3f}"
        if new_score > 0 and new_score >= old_score - 0.01:
            decision = f"promoted (replaced v{champion})"
            registry.promote(version, user, reason)
        else:
            decision = "kept as challenger"
            registry.set_challenger(version)
            registry.log_event("rejected", version, user, reason=reason)

    result = {"version": version, "decision": decision, **metrics}
    save_json(result, METRICS / "train_metrics.json")
    log.info("v%d: %s", version, decision)
    return result


if __name__ == "__main__":
    train(sys.argv[1] if len(sys.argv) > 1 else TRAIN_FILE)
