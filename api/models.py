"""
Keeps the production model (champion) and the waiting model (challenger) in memory.

Deployment strategies, chosen with environment variables:
    SHADOW_MODE=true   the challenger also scores every request, but only the champion's
                       answer is returned. Disagreements are counted.
    CANARY_PERCENT=10  10% of requests are answered by the challenger.

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
        self.recent = deque(maxlen=500)   # last 500 anomaly flags, watched by the watchdog
        self.champion = self.challenger = None
        self.reload()

    def _load(self, name):
        folder = self.folder / name
        return LoadedModel(folder) if (folder / "model.joblib").exists() else None

    def reload(self):
        self.champion = self._load("production")
        self.challenger = self._load("challenger")
        if self.challenger and self.champion and self.challenger.version == self.champion.version:
            self.challenger = None
        self.recent.clear()

    def reload_if_changed(self):
        info = self.folder / "production" / "info.json"
        if info.exists() and (self.champion is None or info.stat().st_mtime != self.champion.stamp):
            self.reload()

    def pick(self):
        if self.challenger and random.uniform(0, 100) < self.canary:
            return self.challenger, "challenger (canary)"
        return self.champion, "champion"

    def anomaly_rate(self):
        return sum(self.recent) / len(self.recent) if self.recent else 0.0
