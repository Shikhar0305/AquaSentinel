"""
Loads the production (champion) model and, optionally, the challenger.

Deployment strategies (set with environment variables):
    SHADOW_MODE=true     challenger scores every request too, but only the champion's
                         answer is returned. Disagreements are counted in Prometheus.
    CANARY_PERCENT=10    10% of requests are answered by the challenger.

The service checks models/production/meta.json on every request, so when the
retraining pipeline promotes a new version the API picks it up automatically.
"""
import json
import os
import random
import threading
from collections import deque
from datetime import datetime, timezone

import joblib

from src.utils import MODELS_DIR, get_logger

log = get_logger("model_service")


class LoadedModel:
    def __init__(self, folder):
        self.folder = folder
        self.model = joblib.load(folder / "model.joblib")
        self.meta = json.loads((folder / "meta.json").read_text())
        self.mtime = (folder / "meta.json").stat().st_mtime

    @property
    def version(self):
        return self.meta["version"]


class ModelService:
    def __init__(self, models_dir=MODELS_DIR):
        self.prod_dir = models_dir / "production"
        self.chal_dir = models_dir / "challenger"
        self.shadow_mode = os.getenv("SHADOW_MODE", "false").lower() == "true"
        self.canary_percent = float(os.getenv("CANARY_PERCENT", "0"))
        self.champion = None
        self.challenger = None
        self.recent_flags = deque(maxlen=500)  # last 500 is_anomaly values, for the watchdog
        self.loaded_at = None
        self._lock = threading.Lock()
        self.reload()

    def reload(self):
        with self._lock:
            if not (self.prod_dir / "model.joblib").exists():
                log.error("no production model found in %s - run training first", self.prod_dir)
                self.champion = None
                return
            self.champion = LoadedModel(self.prod_dir)
            self.challenger = None
            if (self.chal_dir / "model.joblib").exists():
                chal = LoadedModel(self.chal_dir)
                if chal.version != self.champion.version:
                    self.challenger = chal
            self.recent_flags.clear()
            self.loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            log.info("loaded champion v%s (challenger: %s)", self.champion.version,
                     self.challenger.version if self.challenger else "none")

    def reload_if_changed(self):
        meta = self.prod_dir / "meta.json"
        if meta.exists() and (self.champion is None or meta.stat().st_mtime != self.champion.mtime):
            log.info("production model changed on disk - reloading")
            self.reload()

    def pick_model(self):
        """Canary routing: send a small % of traffic to the challenger."""
        if self.challenger and self.canary_percent > 0 and random.uniform(0, 100) < self.canary_percent:
            return self.challenger, "challenger (canary)"
        return self.champion, "champion"

    @staticmethod
    def score(loaded, df):
        m = loaded.model
        scores = m.anomaly_score(df)
        flags = (scores > m.threshold_).astype(int)
        alerts = m.alerts(flags)
        return {
            "scores": scores,
            "flags": flags,
            "alerts": alerts,
            "extra_inflow": m.extra_inflow(df),
            "area": m.suspected_area(df),
            "threshold": m.threshold_,
        }

    def anomaly_rate(self):
        if not self.recent_flags:
            return 0.0
        return sum(self.recent_flags) / len(self.recent_flags)
