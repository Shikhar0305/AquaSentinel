"""
AIOps watchdog - automatic remediation for the AquaSentinel API.

Every few seconds it:
1. Calls /health. After N failures in a row it RESTARTS the service.
2. Reads /model/info. If the share of readings flagged as anomalous is far too
   high, the model is probably broken (or the data changed), so it ROLLS BACK
   to the previous model version through the admin API.
Every action is written to logs/incidents.jsonl so the website can show it.

Two ways to run it:
    # watchdog starts and supervises the API itself (local development)
    python -m ops.watchdog --start-cmd "uvicorn api.main:app --port 8000"

    # API runs in Docker, watchdog restarts the container
    python -m ops.watchdog --restart-cmd "docker restart aquasentinel-api"
"""
import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import datetime

import requests

from src.config import LOGS, get_logger

log = get_logger("watchdog")
INCIDENTS = LOGS / "incidents.jsonl"


def record(kind, detail):
    entry = {"time": datetime.now().isoformat(timespec="seconds"), "incident": kind, "detail": detail}
    with open(INCIDENTS, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log.warning("%s: %s", kind, detail)


class Watchdog:
    def __init__(self, url, admin_key, start_cmd=None, restart_cmd=None, max_failures=3,
                 max_anomaly_rate=0.5, min_predictions=100, cooldown=300):
        self.url = url.rstrip("/")
        self.admin_key = admin_key
        self.start_cmd = start_cmd
        self.restart_cmd = restart_cmd
        self.max_failures = max_failures
        self.max_anomaly_rate = max_anomaly_rate
        self.min_predictions = min_predictions
        self.cooldown = cooldown          # seconds to wait after a rollback before checking again
        self.failures = 0
        self.proc = None
        self.last_rollback = 0

    # ---- service management
    def start(self):
        if self.start_cmd:
            log.info("starting service: %s", self.start_cmd)
            self.proc = subprocess.Popen(shlex.split(self.start_cmd))

    def restart(self):
        if self.start_cmd:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            self.start()
        elif self.restart_cmd:
            subprocess.run(shlex.split(self.restart_cmd), check=False)
        else:
            log.error("no start/restart command configured - cannot restart")
            return
        record("service_restarted", f"after {self.failures} failed health checks")
        self.failures = 0

    # ---- checks
    def check_health(self):
        try:
            r = requests.get(f"{self.url}/health", timeout=3)
            healthy = r.ok and r.json().get("status") == "ok"
        except requests.RequestException:
            healthy = False
        self.failures = 0 if healthy else self.failures + 1
        if not healthy:
            log.info("health check failed (%d/%d)", self.failures, self.max_failures)
        if self.failures >= self.max_failures:
            self.restart()
        return healthy

    def check_model(self):
        """Rule: too many anomalies after enough predictions -> bad model -> roll back."""
        if time.time() - self.last_rollback < self.cooldown:
            return
        headers = {"X-API-Key": self.admin_key}
        try:
            info = requests.get(f"{self.url}/model/info", headers=headers, timeout=3).json()
        except (requests.RequestException, ValueError):
            return
        rate, n = info.get("recent_anomaly_rate", 0), info.get("recent_predictions", 0)
        if n >= self.min_predictions and rate > self.max_anomaly_rate:
            version = info["champion"]["version"]
            reason = f"auto-rollback: v{version} flagged {rate:.0%} of last {n} readings"
            r = requests.post(f"{self.url}/model/rollback", params={"reason": reason}, headers=headers, timeout=30)
            if r.ok:
                record("model_rolled_back", f"{reason} -> now v{r.json()['rolled_back_to']}")
            else:
                record("rollback_failed", f"{reason}: {r.text}")
            self.last_rollback = time.time()

    def run(self, interval=5):
        self.start()
        time.sleep(3 if self.start_cmd else 0)
        log.info("watching %s every %ss", self.url, interval)
        while True:
            if self.check_health():
                self.check_model()
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("API_URL", "http://localhost:8000"))
    parser.add_argument("--start-cmd", help="command that starts the API (watchdog supervises it)")
    parser.add_argument("--restart-cmd", help="command that restarts an API running elsewhere")
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--max-anomaly-rate", type=float, default=0.5)
    args = parser.parse_args()
    Watchdog(args.url, os.getenv("AQUA_ADMIN_KEY", "admin-dev-key"), args.start_cmd, args.restart_cmd,
             max_anomaly_rate=args.max_anomaly_rate).run(args.interval)


if __name__ == "__main__":
    main()
