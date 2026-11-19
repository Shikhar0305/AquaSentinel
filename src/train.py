"""
Step 2 - Train, evaluate and register the leak detector.

1. Split scenarios into train / test.
2. Fit the LeakDetector on NORMAL readings from the train scenarios only.
3. Evaluate on the unseen test scenarios.
4. Log params, metrics, plots and the model to MLflow and register a new version.
5. Promote it to "champion" if it is at least as good as the current champion,
   otherwise keep it as "challenger".

Usage:
    python -m src.train                                   # uses data/raw/sensor_data.csv
    python -m src.train --data data/raw/drift_data.csv    # used by retraining
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402

from src import registry  # noqa: E402
from src.evaluate import evaluate, normal_rows, split_by_scenario  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import (DATA_DIR, METRICS_DIR, REPORTS_DIR, ROOT, file_md5, get_logger,  # noqa: E402
                       get_tracking_uri, load_params, save_json)

log = get_logger("train")


def plot_scenario(model, df, path):
    """Anomaly score of one leak scenario - quick visual sanity check."""
    leak_ids = df.loc[df["leak"] == 1, "scenario_id"].unique()
    g = df[df["scenario_id"] == leak_ids[0]]
    scores = model.anomaly_score(g)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(pd.to_datetime(g["timestamp"]), scores, label="anomaly score", color="#1f77b4")
    ax.axhline(model.threshold_, color="red", ls="--", label="threshold")
    start = pd.to_datetime(g.loc[g["leak"] == 1, "timestamp"].iloc[0])
    ax.axvline(start, color="orange", label="leak starts")
    ax.set_title(f"Scenario {leak_ids[0]} (leak at junction {g['leak_node'].iloc[0]})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def relative_path(path):
    """Store paths relative to the project so lineage works on any machine."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def train(data_path, run_name="train", user="pipeline"):
    params = load_params()
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment("aquasentinel")

    df = pd.read_csv(data_path)
    data_path = relative_path(data_path)
    train_df, test_df = split_by_scenario(df, params["model"]["test_size"], params["model"]["random_state"])
    log.info("train scenarios: %d, test scenarios: %d",
             train_df["scenario_id"].nunique(), test_df["scenario_id"].nunique())

    with mlflow.start_run(run_name=run_name) as run:
        model = build_model(params)
        model.fit(normal_rows(train_df))
        metrics = evaluate(model, test_df)
        log.info("test metrics: %s", metrics)

        mlflow.log_params({f"model.{k}": v for k, v in params["model"].items()})
        mlflow.log_params({"data_file": str(data_path), "train_rows": len(normal_rows(train_df)),
                           "network": params["simulation"]["network"]})
        mlflow.log_metrics(metrics)
        mlflow.log_metric("threshold", model.threshold_)
        mlflow.set_tags({"data_md5": file_md5(ROOT / data_path), "git_commit": registry.git_commit(),
                         "triggered_by": user})

        plot_path = REPORTS_DIR / "leak_scenario_example.png"
        plot_scenario(model, test_df, plot_path)
        mlflow.log_artifact(str(plot_path))

        # cloudpickle because LeakDetector is our own class (skops only allows known types)
        info = mlflow.sklearn.log_model(model, name="model",
                                        serialization_format="cloudpickle",
                                        registered_model_name=params["model"]["name"])
        version = int(info.registered_model_version)

    # tag the version so lineage can be traced from the registry alone
    client = registry.get_client()
    for k, v in {"data_md5": file_md5(ROOT / data_path), "data_file": str(data_path),
                 "git_commit": registry.git_commit(), "test_f1": metrics["f1"],
                 "test_roc_auc": metrics["roc_auc"]}.items():
        client.set_model_version_tag(params["model"]["name"], str(version), k, str(v))
    registry.record_lineage("registered", version, user, metrics=metrics)

    decision = decide_promotion(version, metrics, test_df, params, user)
    result = {"version": version, "run_id": run.info.run_id, "decision": decision, **metrics}
    save_json(result, METRICS_DIR / "train_metrics.json")
    return result


def promotion_score(metrics, params):
    p = params["promotion"]
    if metrics["false_alarm_rate"] > p["max_false_alarm_rate"]:
        return 0.0
    return metrics[p["metric"]]


def decide_promotion(version, metrics, test_df, params, user):
    """Champion/challenger: compare the new model with production on the same test data."""
    champion = registry.get_alias_version("champion")
    if champion is None:
        registry.promote(version, user, reason="first model")
        return "promoted (first model)"

    champ_model = mlflow.sklearn.load_model(f"models:/{params['model']['name']}@champion")
    champ_metrics = evaluate(champ_model, test_df)
    new_score = promotion_score(metrics, params)
    old_score = promotion_score(champ_metrics, params)
    log.info("challenger v%s score=%.3f vs champion v%s score=%.3f (champion false alarms %.1f%%)",
             version, new_score, champion, old_score, 100 * champ_metrics["false_alarm_rate"])

    reason = f"score {old_score:.3f} (v{champion}) -> {new_score:.3f} (v{version})"
    if new_score > 0 and new_score - old_score >= params["promotion"]["min_improvement"]:
        registry.promote(version, user, reason=reason)
        return f"promoted (replaced v{champion})"

    registry.set_challenger(version)
    registry.record_lineage("rejected", version, user, reason=reason)
    return f"kept as challenger (champion v{champion} is better)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR / "raw" / "sensor_data.csv"))
    parser.add_argument("--run-name", default="train")
    args = parser.parse_args()
    result = train(args.data, args.run_name)
    print(result)


if __name__ == "__main__":
    main()
