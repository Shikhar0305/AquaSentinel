"""
Evaluation metrics.

Point metrics (per reading) are standard ML metrics. For operations we also care about:
- detection rate: % of leak scenarios where an alert was raised at all
- detection delay: how long after the leak started the first alert came
- false alarm rate: % of normal readings that raised an alert
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def evaluate(model, df):
    scores = model.anomaly_score(df)
    preds = (scores > model.threshold_).astype(int)
    alerts = model.alerts(preds, df["scenario_id"])
    y = df["leak"].values

    res = df[["scenario_id", "leak"]].copy()
    res["alert"] = alerts
    res["step"] = res.groupby("scenario_id").cumcount()

    leak_scen = res[res["scenario_id"].isin(res.loc[res["leak"] == 1, "scenario_id"].unique())]
    detected, delays = [], []
    for _, g in leak_scen.groupby("scenario_id"):
        start = g.loc[g["leak"] == 1, "step"].min()
        hits = g.loc[(g["leak"] == 1) & (g["alert"] == 1), "step"]
        detected.append(len(hits) > 0)
        if len(hits):
            delays.append((hits.min() - start) * 15)

    metrics = {
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, scores) if len(np.unique(y)) > 1 else float("nan"),
        "alert_f1": f1_score(y, alerts, zero_division=0),
        "detection_rate": float(np.mean(detected)) if detected else float("nan"),
        "median_detection_delay_min": float(np.median(delays)) if delays else float("nan"),
        "false_alarm_rate": float(alerts[y == 0].mean()) if (y == 0).any() else 0.0,
    }
    return {k: round(float(v), 4) for k, v in metrics.items()}


def split_by_scenario(df, test_size=0.3, seed=42):
    """Split whole scenarios (not single rows) so test data is truly unseen."""
    leak_ids = df.loc[df["leak"] == 1, "scenario_id"].unique()
    normal_ids = np.setdiff1d(df["scenario_id"].unique(), leak_ids)
    rng = np.random.default_rng(seed)
    test_ids = []
    for ids in (normal_ids, leak_ids):  # stratified: keep both types in test
        ids = rng.permutation(ids)
        test_ids += list(ids[: max(1, int(round(len(ids) * test_size)))])
    is_test = df["scenario_id"].isin(test_ids)
    return df[~is_test].copy(), df[is_test].copy()


def normal_rows(df):
    return df[df["leak"] == 0]


def summary_table(metrics_by_model):
    return pd.DataFrame(metrics_by_model).T
