"""
The leak detection model.

Idea - the WATER BALANCE:
    water going into the network (2 sources + 3 tanks) = water used + leaks
So a leak shows up as extra inflow compared with a normal day at the same time.

Features: extra inflow (actual - normal for that 15-min slot of the day),
averaged over 15 min, 1 h and 3 h. A short spike and a slow, steady increase
(a leak) look different across these windows.

Model: Isolation Forest trained only on normal readings. Unusual points are
easy to isolate with random splits, so they get a higher anomaly score.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def time_slot(df):
    """0..95 = which 15-minute slot of the day the reading belongs to."""
    ts = pd.to_datetime(df["timestamp"])
    return (ts.dt.hour * 4 + ts.dt.minute // 15).values


class LeakDetector(BaseEstimator):
    def __init__(self, pressure_sensors, inflow_pipes, windows=(1, 4, 12), n_estimators=200,
                 threshold_percentile=98, alert_after=3, seed=42):
        self.pressure_sensors = pressure_sensors
        self.inflow_pipes = inflow_pipes
        self.windows = windows
        self.n_estimators = n_estimators
        self.threshold_percentile = threshold_percentile
        self.alert_after = alert_after
        self.seed = seed

    # ---------------- features
    def _prepare(self, df):
        out = pd.DataFrame(index=df.index)
        out["net_inflow"] = df[[f"F_{p}" for p in self.inflow_pipes]].sum(axis=1)
        for n in self.pressure_sensors:
            out[f"P_{n}"] = df[f"P_{n}"]
        out["group"] = df["scenario_id"] if "scenario_id" in df else 0   # live data = one stream
        out["slot"] = time_slot(df)
        return out

    def _features(self, prep):
        extra = prep["net_inflow"] - self.normal_.loc[prep["slot"], "net_inflow"].values
        feats = {}
        for w in self.windows:
            feats[f"extra_inflow_{w * 15}min"] = extra.groupby(prep["group"]).transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean())
        return pd.DataFrame(feats)

    # ---------------- training
    def fit(self, normal_df):
        prep = self._prepare(normal_df)
        # normal value of every column for each time slot of the day
        self.normal_ = prep.drop(columns="group").groupby("slot").median().reindex(range(96)).interpolate()
        X = self._features(prep)
        self.scaler_ = StandardScaler().fit(X)
        self.forest_ = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed)
        self.forest_.fit(self.scaler_.transform(X))
        self.threshold_ = float(np.percentile(self._score(X), self.threshold_percentile))
        # how much each pressure sensor normally varies (used for the suspected area)
        pcols = [f"P_{n}" for n in self.pressure_sensors]
        self.pressure_std_ = (prep[pcols] - self.normal_.loc[prep["slot"], pcols].values).std()
        return self

    # ---------------- prediction
    def _score(self, X):
        return -self.forest_.score_samples(self.scaler_.transform(X))

    def predict_details(self, df):
        """Everything the API and evaluation need, in one pass."""
        prep = self._prepare(df)
        X = self._features(prep)
        scores = self._score(X)
        extra = X.iloc[:, min(1, X.shape[1] - 1)].values     # 1-hour extra inflow
        # a leak can only ADD water, so less inflow than normal is never a leak
        anomaly = ((scores > self.threshold_) & (extra > 0)).astype(int)
        # alert = N anomalous readings in a row
        run = pd.Series(anomaly, index=df.index).groupby(prep["group"]).transform(
            lambda s: s.rolling(self.alert_after, min_periods=self.alert_after).sum())
        alert = (run >= self.alert_after).astype(int).values
        # suspected area = pressure sensor with the biggest unusual drop
        pcols = [f"P_{n}" for n in self.pressure_sensors]
        drop = (prep[pcols] - self.normal_.loc[prep["slot"], pcols].values) / self.pressure_std_
        area = drop.rolling(4, min_periods=1).mean().idxmin(axis=1).str[2:].values
        return pd.DataFrame({
            "anomaly_score": scores,
            "is_anomaly": anomaly,
            "leak_alert": alert,
            "extra_inflow": extra,
            "suspected_node": area,
        }, index=df.index)

    def anomaly_score(self, df):
        return self.predict_details(df)["anomaly_score"].values


def build_model(params):
    m = params["model"]
    return LeakDetector(params["sensors"]["pressure"], params["sensors"]["inflow"], tuple(m["windows"]),
                        m["n_estimators"], m["threshold_percentile"], m["alert_after"], m["seed"])
