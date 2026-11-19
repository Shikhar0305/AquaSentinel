"""
AquaSentinel REST API (FastAPI).

Run locally:   uvicorn api.main:app --reload --port 8000
Docs:          http://localhost:8000/docs

Every request is logged to logs/api.log (JSON lines). The watchdog and the log
anomaly detector in ops/ read that file. Prometheus scrapes /metrics.
"""
import json
import os
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from api.model_service import ModelService
from api.security import audit, require_role
from src.utils import LOGS_DIR, METRICS_DIR, get_logger, load_params, sensor_columns

log = get_logger("api")
PARAMS = load_params()
SENSORS = sensor_columns(PARAMS)

app = FastAPI(title="AquaSentinel API", version="1.0",
              description="Leak detection for water pipelines from pressure and flow sensors.")
service = ModelService()

# ---------------- Prometheus metrics ----------------
REQUESTS = Counter("aqua_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram("aqua_request_latency_seconds", "Request latency", ["path"],
                    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("aqua_predictions_total", "Readings scored", ["result"])
LEAK_ALERTS = Counter("aqua_leak_alerts_total", "Leak alerts raised")
SCORE = Histogram("aqua_anomaly_score", "Anomaly score of the latest reading",
                  buckets=(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8))
SHADOW_DISAGREE = Counter("aqua_shadow_disagreements_total", "Champion and challenger disagree")
MODEL_VERSION = Gauge("aqua_model_version", "Model version loaded", ["role"])
ANOMALY_RATE = Gauge("aqua_recent_anomaly_rate", "Share of anomalous readings in the last 500")
DRIFT_SHARE = Gauge("aqua_drift_share", "Share of sensor columns with drift (last check)")
DRIFT_FLAG = Gauge("aqua_drift_detected", "1 if the last drift check found drift")


# ---------------- request / response models ----------------
class Reading(BaseModel):
    timestamp: datetime
    sensors: Dict[str, float] = Field(..., examples=[{"P_15": 27.5, "F_60": 830.2}])


class PredictRequest(BaseModel):
    readings: List[Reading] = Field(..., min_length=1, max_length=2000,
                                    description="Recent readings in time order. Send at least the "
                                                "last 12 (3 hours) so the smoothing windows work.")


class ReadingResult(BaseModel):
    timestamp: datetime
    anomaly_score: float
    is_anomaly: bool
    leak_alert: bool
    extra_inflow_lps: float
    suspected_area: str


class PredictResponse(BaseModel):
    model_version: int
    served_by: str
    threshold: float
    leak_alert: bool
    results: List[ReadingResult]


# ---------------- logging middleware ----------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path
        if path != "/metrics":
            REQUESTS.labels(path, str(status)).inc()
            LATENCY.labels(path).observe(elapsed)
            with open(LOGS_DIR / "api.log", "a") as f:
                f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                                    "method": request.method, "path": path, "status": status,
                                    "latency_ms": round(elapsed * 1000, 2)}) + "\n")


def to_dataframe(readings):
    rows = []
    for r in readings:
        missing = [s for s in SENSORS if s not in r.sensors]
        if missing:
            raise HTTPException(status_code=422, detail=f"missing sensors: {missing}")
        rows.append({"timestamp": r.timestamp, **{s: r.sensors[s] for s in SENSORS}})
    return pd.DataFrame(rows)


# ---------------- endpoints ----------------
@app.get("/health")
def health():
    ok = service.champion is not None
    return {"status": "ok" if ok else "no_model",
            "model_version": service.champion.version if ok else None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, role: str = Depends(require_role("operator"))):
    service.reload_if_changed()
    if service.champion is None:
        raise HTTPException(status_code=503, detail="no model loaded")

    df = to_dataframe(req.readings)
    chosen, served_by = service.pick_model()
    out = service.score(chosen, df)

    # shadow deployment: run the challenger silently and compare
    if service.shadow_mode and service.challenger is not None and chosen is service.champion:
        shadow = service.score(service.challenger, df)
        if shadow["flags"][-1] != out["flags"][-1]:
            SHADOW_DISAGREE.inc()

    last_flag = int(out["flags"][-1])
    service.recent_flags.append(last_flag)
    PREDICTIONS.labels("anomaly" if last_flag else "normal").inc()
    SCORE.observe(float(out["scores"][-1]))
    leak_alert = bool(out["alerts"][-1])
    if leak_alert:
        LEAK_ALERTS.inc()

    results = [
        ReadingResult(timestamp=ts, anomaly_score=round(float(s), 4), is_anomaly=bool(f),
                      leak_alert=bool(a), extra_inflow_lps=round(float(e), 2), suspected_area=str(ar))
        for ts, s, f, a, e, ar in zip(df["timestamp"], out["scores"], out["flags"],
                                      out["alerts"], out["extra_inflow"], out["area"])
    ]
    return PredictResponse(model_version=chosen.version, served_by=served_by,
                           threshold=round(out["threshold"], 4), leak_alert=leak_alert, results=results)


@app.get("/model/info")
def model_info(role: str = Depends(require_role("viewer"))):
    service.reload_if_changed()
    return {
        "champion": service.champion.meta if service.champion else None,
        "challenger": service.challenger.meta if service.challenger else None,
        "shadow_mode": service.shadow_mode,
        "canary_percent": service.canary_percent,
        "loaded_at": service.loaded_at,
        "recent_anomaly_rate": round(service.anomaly_rate(), 4),
        "recent_predictions": len(service.recent_flags),
    }


@app.get("/model/lineage")
def model_lineage(role: str = Depends(require_role("viewer"))):
    from src.registry import read_lineage
    return read_lineage()


@app.get("/monitoring/drift")
def drift_status(role: str = Depends(require_role("viewer"))):
    path = METRICS_DIR / "drift.json"
    if not path.exists():
        return {"status": "no drift check has run yet"}
    return json.loads(path.read_text())


@app.post("/model/reload")
def reload_model(role: str = Depends(require_role("admin"))):
    service.reload()
    audit(role, "reload", f"v{service.champion.version if service.champion else None}")
    return {"loaded_version": service.champion.version if service.champion else None}


@app.post("/model/rollback")
def rollback_model(reason: str = "manual rollback from API", role: str = Depends(require_role("admin"))):
    from src.registry import rollback
    try:
        version = rollback(user=f"api:{role}", reason=reason)
    except Exception as e:
        raise HTTPException(status_code=409, detail=str(e))
    service.reload()
    audit(role, "rollback", f"to v{version}: {reason}")
    return {"rolled_back_to": version}


@app.post("/model/promote/{version}")
def promote_model(version: int, role: str = Depends(require_role("admin"))):
    from src.registry import promote
    promote(version, user=f"api:{role}", reason="manual promote from API")
    service.reload()
    audit(role, "promote", f"v{version}")
    return {"champion": version}


@app.post("/admin/crash", include_in_schema=os.getenv("ENABLE_CHAOS", "false") == "true")
def crash(role: str = Depends(require_role("admin"))):
    """Demo only: kills the API process so the watchdog can show auto-restart."""
    if os.getenv("ENABLE_CHAOS", "false") != "true":
        raise HTTPException(status_code=404)
    audit(role, "crash", "chaos test")
    os._exit(1)


@app.get("/metrics")
def metrics():
    if service.champion:
        MODEL_VERSION.labels("champion").set(service.champion.version)
    if service.challenger:
        MODEL_VERSION.labels("challenger").set(service.challenger.version)
    ANOMALY_RATE.set(service.anomaly_rate())
    drift_file = METRICS_DIR / "drift.json"
    if drift_file.exists():
        d = json.loads(drift_file.read_text())
        DRIFT_SHARE.set(d.get("drift_share", 0))
        DRIFT_FLAG.set(1 if d.get("drift_detected") else 0)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
