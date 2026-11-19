"""
The leak detection model, built for intermittent supply (water for a few hours a day).

Idea - the WATER BALANCE:
    water going into the zone (2 sources + 3 tanks) = water used + leaks
So a leak shows up as extra inflow compared with normal.

What is "normal" depends on the supply phase, not the clock: supply timings slip
from day to day, and right after the valve opens households fill their tanks, so
inflow is very high and varies a lot. When the valve is closed almost no water is
used, so a leak stands out clearly. (Indian utilities use the same idea when they
check minimum flow while supply is off.)

So for every phase step - "valve open for 0-15 min", "open 15-30 min", ...
"closed for 2 h" - we learn the normal inflow and how much it varies, and turn
each reading into a z-score: how many standard deviations above normal it is.

Features: that z-score averaged over 15 min, 1 h and 3 h.
Model: Isolation Forest trained only on normal readings.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


MAX_STEPS = 48   # phases longer than 12 hours share the last slot


def phase_slot(df):
    """Slot number for the supply phase: 0-47 = valve closed, 100-147 = valve open."""
    steps = np.minimum(df["valve_minutes"].values // 15, MAX_STEPS - 1)
    return df["valve_open"].values * 100 + steps


def all_slots():
    return list(range(MAX_STEPS)) + list(range(100, 100 + MAX_STEPS))


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
        out["slot"] = phase_slot(df)
        return out

    def _smooth(self, series, groups, w):
        return series.groupby(groups).transform(lambda s: s.rolling(w, min_periods=1).mean())

    def _features(self, prep):
        normal = self.normal_.loc[prep["slot"]]
        z = (prep["net_inflow"] - normal["net_inflow"].values) / self.spread_.loc[prep["slot"], "net_inflow"].values
        return pd.DataFrame({f"inflow_z_{w * 15}min": self._smooth(z, prep["group"], w) for w in self.windows})

    # ---------------- training
    def fit(self, normal_df):
        prep = self._prepare(normal_df)
        # normal value and normal spread of every column for each supply phase slot
        by_slot = prep.drop(columns="group").groupby("slot")
        self.normal_ = self._fill(by_slot.median())
        self.spread_ = self._fill(by_slot.std()).clip(lower=0.1)
        X = self._features(prep)
        self.scaler_ = StandardScaler().fit(X)
        self.forest_ = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed)
        self.forest_.fit(self.scaler_.transform(X))
        self.threshold_ = float(np.percentile(self._score(X), self.threshold_percentile))
        return self

    @staticmethod
    def _fill(table):
        """Slots never seen in training borrow the nearest slot with the same valve state."""
        table = table.reindex(all_slots())
        closed, opened = table.loc[:MAX_STEPS - 1].ffill().bfill(), table.loc[100:].ffill().bfill()
        return pd.concat([closed, opened]).fillna(table.mean())

    # ---------------- prediction
    def _score(self, X):
        return -self.forest_.score_samples(self.scaler_.transform(X))

    def predict_details(self, df):
        """Everything the API and evaluation need, in one pass."""
        prep = self._prepare(df)
        X = self._features(prep)
        scores = self._score(X)
        # a leak can only ADD water, so less inflow than normal is never a leak
        above_normal = X.iloc[:, min(1, X.shape[1] - 1)].values > 0      # 1-hour average above normal
        anomaly = ((scores > self.threshold_) & above_normal).astype(int)
        extra = self._smooth(prep["net_inflow"] - self.normal_.loc[prep["slot"], "net_inflow"].values,
                             prep["group"], 4).values    # extra L/s, 1-hour average
        # alert = N anomalous readings in a row
        run = pd.Series(anomaly, index=df.index).groupby(prep["group"]).transform(
            lambda s: s.rolling(self.alert_after, min_periods=self.alert_after).sum())
        alert = (run >= self.alert_after).astype(int).values
        # suspected area = pressure sensor with the biggest unusual drop
        pcols = [f"P_{n}" for n in self.pressure_sensors]
        normal, spread = self.normal_.loc[prep["slot"], pcols].values, self.spread_.loc[prep["slot"], pcols].values
        drop = (prep[pcols] - normal) / spread
        area = drop.rolling(4, min_periods=1).mean().idxmin(axis=1).str[2:].values
        return pd.DataFrame({
            "anomaly_score": scores,
            "is_anomaly": anomaly,
            "leak_alert": alert,
            "extra_inflow": extra,
            "valve_open": df["valve_open"].values,
            "suspected_node": area,
        }, index=df.index)

    def predict(self, df):
        """1 = leak alert, 0 = no alert (lets MLflow serve the model as a normal sklearn model)."""
        return self.predict_details(df)["leak_alert"].values

    def anomaly_score(self, df):
        return self.predict_details(df)["anomaly_score"].values


def build_model(params):
    m = params["model"]
    return LeakDetector(params["sensors"]["pressure"], params["sensors"]["inflow"], tuple(m["windows"]),
                        m["n_estimators"], m["threshold_percentile"], m["alert_after"], m["seed"])
