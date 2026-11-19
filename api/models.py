"""
Keeps the production model (champion) and the waiting model (challenger) in memory.

Deployment strategies, chosen with environment variables:
    SHADOW_MODE=true   the challenger also scores every request, but only the champion's
                       answer is returned. Disagreements are counted.
    CANARY_PERCENT=10  10% of requests are answered by the challenger.

The previous production model is also kept. It scores every request in the
background, so the watchdog can see when a newly deployed model raises far
more alarms than the last good one (and roll it back).

If models/production/ changes on disk (retraining promoted a new version), the
model is reloaded automatically on the next request.
"""
import json
import os
import random
from collections import deque

import joblib

from src.config import MODELS


class LoadedModel:
    def __init__(self, folder):
        self.model = joblib.load(folder / "model.joblib")
        self.info = json.loads((folder / "info.json").read_text())
        self.version = self.info["version"]
        self.stamp = (folder / "info.json").stat().st_mtime


class ModelStore:
    def __init__(self, folder=MODELS):
        self.folder = folder
        self.shadow = os.getenv("SHADOW_MODE", "false").lower() == "true"
        self.canary = float(os.getenv("CANARY_PERCENT", "0"))
        self.recent = deque(maxlen=500)            # last 500 anomaly flags of the champion
        self.recent_previous = deque(maxlen=500)   # same readings scored by the previous model
        self.champion = self.challenger = self.previous = None
        self.reload()

    def _load(self, name):
        folder = self.folder / name
        return LoadedModel(folder) if (folder / "model.joblib").exists() else None

    def reload(self):
        self.champion = self._load("production")
        self.challenger = self._load("challenger")
        self.previous = self._load("previous")
        version = self.champion.version if self.champion else None
        if self.challenger and self.challenger.version == version:
            self.challenger = None
        if self.previous and self.previous.version == version:
            self.previous = None
        self.recent.clear()
        self.recent_previous.clear()

    def reload_if_changed(self):
        info = self.folder / "production" / "info.json"
        if info.exists() and (self.champion is None or info.stat().st_mtime != self.champion.stamp):
            self.reload()

    def pick(self):
        if self.challenger and random.uniform(0, 100) < self.canary:
            return self.challenger, "challenger (canary)"
        return self.champion, "champion"

    @staticmethod
    def rate(flags):
        return sum(flags) / len(flags) if flags else 0.0

    def anomaly_rate(self):
        return self.rate(self.recent)
