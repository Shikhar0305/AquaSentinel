"""
LeakDetector = feature engineering + StandardScaler + Isolation Forest in one object,
so the API, dashboard and training all use exactly the same steps.

Isolation Forest is trained only on normal readings. It isolates points with random
splits; unusual points get isolated quickly and receive a high anomaly score.
The alarm threshold is a percentile of the scores seen on normal training data,
so no leak labels are needed for training (labels are only used for evaluation).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.features import add_water_balance, fit_baseline, make_features, residuals


class LeakDetector(BaseEstimator):
    def __init__(self, pressure_sensors, inflow_pipes, windows=(1, 4, 12), n_estimators=200,
                 max_samples=256, threshold_percentile=98.0, consecutive_for_alert=3,
                 timestep_min=15, random_state=42):
        self.pressure_sensors = pressure_sensors
        self.inflow_pipes = inflow_pipes
        self.windows = windows
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.threshold_percentile = threshold_percentile
        self.consecutive_for_alert = consecutive_for_alert
        self.timestep_min = timestep_min
        self.random_state = random_state

    def _features(self, df):
        df = add_water_balance(df, self.inflow_pipes)
        return make_features(df, self.baseline_, self.windows, self.timestep_min)

    def fit(self, df_normal):
        """df_normal: readings with no leak."""
        self.pressure_cols_ = [f"P_{n}" for n in self.pressure_sensors]
        df_normal = add_water_balance(df_normal, self.inflow_pipes)
        self.baseline_ = fit_baseline(df_normal, ["net_inflow"] + self.pressure_cols_, self.timestep_min)

        X = make_features(df_normal, self.baseline_, self.windows, self.timestep_min)
        self.scaler_ = StandardScaler().fit(X)
        self.forest_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
        ).fit(self.scaler_.transform(X))

        train_scores = -self.forest_.score_samples(self.scaler_.transform(X))
        self.threshold_ = float(np.percentile(train_scores, self.threshold_percentile))
        self.feature_names_ = list(X.columns)

        # normal spread of each pressure sensor, used by explain()
        p_res = residuals(df_normal, self.baseline_, self.pressure_cols_, self.timestep_min)
        self.pressure_std_ = p_res.std().replace(0, 1.0)
        return self

    def anomaly_score(self, df):
        """Higher = more unusual."""
        X = self._features(df)
        return -self.forest_.score_samples(self.scaler_.transform(X))

    def predict(self, df):
        """1 = reading looks like a leak, 0 = normal."""
        return (self.anomaly_score(df) > self.threshold_).astype(int)

    def alerts(self, predictions, groups=None):
        """Raise an alert only after N anomalous readings in a row (avoids one-off false alarms)."""
        p = pd.Series(predictions)
        groups = pd.Series(0 if groups is None else np.asarray(groups), index=p.index)
        n = self.consecutive_for_alert
        run = p.groupby(groups).transform(lambda s: s.rolling(n, min_periods=n).sum())
        return (run >= n).astype(int).values

    def extra_inflow(self, df):
        """Estimated leak size in L/s = inflow above normal, smoothed over 1 hour."""
        X = self._features(df)
        col = [c for c in X.columns if c.endswith(f"_{4 * self.timestep_min}min")]
        return X[col[0] if col else X.columns[-1]].values

    def suspected_area(self, df):
        """
        Root-cause hint: the pressure sensor with the biggest unusual DROP.
        The leak is most likely in the part of the network near that junction.
        """
        p_res = residuals(df, self.baseline_, self.pressure_cols_, self.timestep_min)
        z = p_res / self.pressure_std_
        z = z.rolling(4, min_periods=1).mean()
        return z.idxmin(axis=1).str.replace("P_", "junction ", regex=False).values


def build_model(params):
    m = params["model"]
    return LeakDetector(
        pressure_sensors=params["sensors"]["pressure"],
        inflow_pipes=params["sensors"]["inflow"],
        windows=tuple(m["windows"]),
        n_estimators=m["n_estimators"],
        max_samples=m["max_samples"],
        threshold_percentile=m["threshold_percentile"],
        consecutive_for_alert=m["consecutive_for_alert"],
        timestep_min=params["simulation"]["timestep_min"],
        random_state=m["random_state"],
    )
