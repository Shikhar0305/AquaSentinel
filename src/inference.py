
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "champion.joblib"

_model = None


def get_model():
    global _model

    if _model is None:
        _model = joblib.load(MODEL_PATH)

    return _model


FEATURES = [
    "inflow_lps",
    "valve_open",
    "valve_minutes",
    "inflow_mean_4",
    "inflow_mean_12",
    "inflow_min_12",
    "inflow_change_4",
]


def create_input_features(inflow_history: list[float], valve_open: bool, valve_minutes: int) -> pd.DataFrame:
    if len(inflow_history) < 12:
        raise ValueError("At least 12 inflow readings (3 hours) are required.")

    if any(not np.isfinite(x) or x < 0 or x > 5000 for x in inflow_history):
        raise ValueError("Inflow readings must be between 0 and 5000 L/s.")

    history = pd.Series(inflow_history, dtype=float)

    features = {
        "inflow_lps": history.iloc[-1],
        "valve_open": int(valve_open),
        "valve_minutes": valve_minutes,
        "inflow_mean_4": history.iloc[-4:].mean(),
        "inflow_mean_12": history.iloc[-12:].mean(),
        "inflow_min_12": history.iloc[-12:].min(),
        "inflow_change_4": history.iloc[-1] - history.iloc[-5],
    }

    return pd.DataFrame([features])[FEATURES]


def predict_leak(inflow_history: list[float], valve_open: bool, valve_minutes: int) -> float:
    """Probability (0-1) that the zone has a leak right now."""
    features = create_input_features(inflow_history, valve_open, valve_minutes)
    probability = get_model().predict_proba(features)[0, 1]

    return float(np.clip(probability, 0, 1))
