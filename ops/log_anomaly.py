"""
Log anomaly detector for the API itself (AIOps, Unit 1).

Reads logs/api.log, groups requests per minute and builds 4 features:
    request count, error rate (4xx/5xx), mean latency, p95 latency
An Isolation Forest finds unusual minutes (traffic spikes, error bursts, slow responses).
The result is saved as reports/log_anomalies.png and reports/log_anomalies.csv.

Usage:
    python -m ops.log_anomaly
    python -m ops.log_anomaly --generate-demo   # write a synthetic log with a few incidents first
"""
import argparse
import json
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402

from src.utils import LOGS_DIR, REPORTS_DIR, get_logger  # noqa: E402

log = get_logger("log_anomaly")


def load_log(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    df["error"] = df["status"] >= 400
    return df


def per_minute(df):
    g = df.set_index("time").resample("1min")
    feats = pd.DataFrame({
        "requests": g["status"].count(),
        "error_rate": g["error"].mean(),
        "latency_mean": g["latency_ms"].mean(),
        "latency_p95": g["latency_ms"].quantile(0.95),
    }).fillna(0)
    return feats[feats["requests"] > 0]


def detect(feats, contamination=0.05):
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=0)
    feats = feats.copy()
    feats["anomaly"] = model.fit_predict(feats[["requests", "error_rate", "latency_mean", "latency_p95"]]) == -1
    return feats


def plot(feats, path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    bad = feats[feats["anomaly"]]
    for ax, col, title in zip(axes, ["requests", "error_rate", "latency_p95"],
                              ["Requests per minute", "Error rate", "p95 latency (ms)"]):
        ax.plot(feats.index, feats[col], color="#1f77b4", lw=1.2)
        ax.scatter(bad.index, bad[col], color="red", zorder=3, s=25, label="anomaly")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    fig.suptitle("API log anomalies (Isolation Forest)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def generate_demo_log(path, minutes=240, seed=1):
    """Synthetic normal traffic with an error burst, a latency spike and a traffic surge."""
    rng = np.random.default_rng(seed)
    start = datetime.now().replace(second=0, microsecond=0) - timedelta(minutes=minutes)
    with open(path, "w") as f:
        for m in range(minutes):
            n = rng.poisson(30)
            err_p, lat = 0.01, 25
            if 80 <= m < 85:
                err_p = 0.4            # error burst
            if 150 <= m < 154:
                lat = 400              # slow model / disk
            if 200 <= m < 203:
                n = rng.poisson(250)   # traffic surge
            for _ in range(n):
                t = start + timedelta(minutes=m, seconds=float(rng.uniform(0, 60)))
                status = 500 if rng.random() < err_p else 200
                f.write(json.dumps({"time": t.isoformat(timespec="seconds"), "method": "POST",
                                    "path": "/predict", "status": status,
                                    "latency_ms": round(float(rng.gamma(4, lat / 4)), 2)}) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=str(LOGS_DIR / "api.log"))
    parser.add_argument("--generate-demo", action="store_true")
    args = parser.parse_args()
    path = args.log
    if args.generate_demo:
        path = str(LOGS_DIR / "api_demo.log")
        generate_demo_log(path)
    feats = detect(per_minute(load_log(path)))
    feats.to_csv(REPORTS_DIR / "log_anomalies.csv")
    plot(feats, REPORTS_DIR / "log_anomalies.png")
    log.info("%d of %d minutes look anomalous -> reports/log_anomalies.png", feats["anomaly"].sum(), len(feats))
    print(feats[feats["anomaly"]].round(3).to_string())


if __name__ == "__main__":
    main()
