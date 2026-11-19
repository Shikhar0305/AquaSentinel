"""
AquaSentinel API + website.

    uvicorn api.main:app --port 8000

    http://localhost:8000        website (control room)
    http://localhost:8000/docs   API documentation
    http://localhost:8000/metrics  Prometheus metrics

Every request is written to logs/api.log; the watchdog and the log anomaly
detector in ops/ read it.
"""
import json
import os
import time
from datetime import datetime
from functools import lru_cache
from typing import Dict, List

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from api.auth import audit, current_role, require
from api.models import ModelStore
from src import registry
from src.config import LOGS, METRICS, ROOT, SUMMER_FILE, TRAIN_FILE, load_params, sensor_columns
from src.data import validate

PARAMS = load_params()
SENSORS = sensor_columns(PARAMS)

app = FastAPI(title="AquaSentinel API", version="2.0",
              description="Leak detection for water pipelines from pressure and flow sensors.")
store = ModelStore()

# ------------------------------------------------------------ Prometheus metrics
REQUESTS = Counter("aqua_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram("aqua_request_latency_seconds", "Request latency", ["path"])
PREDICTIONS = Counter("aqua_predictions_total", "Readings scored", ["result"])
ALERTS = Counter("aqua_leak_alerts_total", "Leak alerts raised")
SHADOW_DIFF = Counter("aqua_shadow_disagreements_total", "Champion and challenger disagree")
VERSION = Gauge("aqua_model_version", "Model version in production")
ANOMALY_RATE = Gauge("aqua_recent_anomaly_rate", "Share of anomalous readings in the last 500")
DRIFT_SHARE = Gauge("aqua_drift_share", "Share of sensors that drifted in the last check")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    seconds = time.perf_counter() - start
    path = request.url.path
    if path.startswith(("/predict", "/model", "/monitoring", "/api", "/health")):
        REQUESTS.labels(path, str(response.status_code)).inc()
        LATENCY.labels(path).observe(seconds)
        with open(LOGS / "api.log", "a") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "method": request.method,
                                "path": path, "status": response.status_code,
                                "latency_ms": round(seconds * 1000, 2)}) + "\n")
    return response


# ------------------------------------------------------------ prediction
class Reading(BaseModel):
    timestamp: datetime
    valve_open: bool = Field(..., description="Is the zone supply valve open (from SCADA)?")
    valve_minutes: int = Field(..., ge=0, description="Minutes since the valve last opened or closed")
    sensors: Dict[str, float] = Field(..., examples=[{"P_15": 27.5, "F_60": 830.2}])


class PredictRequest(BaseModel):
    readings: List[Reading] = Field(..., min_length=1, max_length=2000,
                                    description="Recent readings in time order (send the last 12 or more).")


@app.get("/health")
def health():
    store.reload_if_changed()
    return {"status": "ok" if store.champion else "no_model",
            "model_version": store.champion.version if store.champion else None}


@app.post("/predict")
def predict(req: PredictRequest, role: str = Depends(require("operator"))):
    store.reload_if_changed()
    if store.champion is None:
        raise HTTPException(503, "No model in production yet. Run: python -m src.train")
    rows = []
    for r in req.readings:
        missing = [s for s in SENSORS if s not in r.sensors]
        if missing:
            raise HTTPException(422, f"missing sensors: {missing}")
        rows.append({"timestamp": r.timestamp, "valve_open": int(r.valve_open),
                     "valve_minutes": r.valve_minutes, **r.sensors})
    df = pd.DataFrame(rows)

    chosen, served_by = store.pick()
    out = chosen.model.predict_details(df)
    if store.shadow and store.challenger and chosen is store.champion:
        shadow = store.challenger.model.predict_details(df)
        if shadow["is_anomaly"].iloc[-1] != out["is_anomaly"].iloc[-1]:
            SHADOW_DIFF.inc()

    if store.previous:   # last good model scores the same readings in the background
        store.recent_previous.append(int(store.previous.model.predict_details(df)["is_anomaly"].iloc[-1]))

    last = out.iloc[-1]
    store.recent.append(int(last["is_anomaly"]))
    PREDICTIONS.labels("anomaly" if last["is_anomaly"] else "normal").inc()
    if last["leak_alert"]:
        ALERTS.inc()

    out["extra_inflow"] = out["extra_inflow"].round(2)
    out["anomaly_score"] = out["anomaly_score"].round(4)
    out.insert(0, "timestamp", df["timestamp"].astype(str))
    return {"model_version": chosen.version, "served_by": served_by,
            "threshold": round(chosen.model.threshold_, 4), "leak_alert": bool(last["leak_alert"]),
            "results": out.to_dict(orient="records")}


# ------------------------------------------------------------ models and monitoring
@app.get("/model/info")
def model_info(role: str = Depends(require("viewer"))):
    store.reload_if_changed()
    return {"champion": store.champion.info if store.champion else None,
            "challenger": store.challenger.info if store.challenger else None,
            "shadow_mode": store.shadow, "canary_percent": store.canary,
            "recent_anomaly_rate": round(store.anomaly_rate(), 4), "recent_predictions": len(store.recent),
            "previous_version": store.previous.version if store.previous else None,
            "previous_anomaly_rate": round(store.rate(store.recent_previous), 4) if store.previous else None}


@app.get("/model/lineage")
def model_lineage(role: str = Depends(require("viewer"))):
    return registry.read_lineage()


@app.get("/monitoring/drift")
def drift(role: str = Depends(require("viewer"))):
    path = METRICS / "drift.json"
    return json.loads(path.read_text()) if path.exists() else {}


@app.post("/monitoring/run")
def run_monitoring(role: str = Depends(require("admin"))):
    """Ingest the next batch of data, check drift and retrain if needed (same as python -m src.monitor)."""
    from src.monitor import run_cycle
    result = run_cycle()
    store.reload_if_changed()
    audit(role, "monitoring cycle", result.get("action", ""))
    return result


@app.post("/model/promote/{version}")
def promote(version: int, role: str = Depends(require("admin"))):
    registry.promote(version, f"api:{role}", "manual promote")
    store.reload()
    audit(role, "promote", f"v{version}")
    return {"champion": version}


@app.post("/model/rollback")
def rollback(reason: str = "manual rollback", role: str = Depends(require("admin"))):
    try:
        version = registry.rollback(f"api:{role}", reason)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    store.reload()
    audit(role, "rollback", f"to v{version}: {reason}")
    return {"rolled_back_to": version}


@app.post("/admin/crash", include_in_schema=False)
def crash(role: str = Depends(require("admin"))):
    """Demo only (ENABLE_CHAOS=true): kills the API so the watchdog can restart it."""
    if os.getenv("ENABLE_CHAOS") != "true":
        raise HTTPException(404)
    audit(role, "crash test")
    os._exit(1)


@app.get("/metrics")
def metrics():
    VERSION.set(store.champion.version if store.champion else 0)
    ANOMALY_RATE.set(store.anomaly_rate())
    drift_file = METRICS / "drift.json"
    if drift_file.exists():
        DRIFT_SHARE.set(json.loads(drift_file.read_text()).get("drift_share", 0))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ------------------------------------------------------------ data for the website
@lru_cache
def load_season(season):
    path = TRAIN_FILE if season == "winter" else SUMMER_FILE
    if not path.exists():
        raise HTTPException(404, "No data yet. Run: python -m src.data simulate")
    return validate(pd.read_csv(path), PARAMS)[0]   # fills gaps left by power cuts


@app.get("/api/me")
def me(role: str = Depends(current_role)):
    return {"role": role}


@app.get("/api/network")
@lru_cache
def network():
    import wntr
    wn = wntr.network.WaterNetworkModel(PARAMS["simulation"]["network"])
    return {"nodes": {n: wn.get_node(n).coordinates for n in wn.node_name_list},
            "pipes": [[link.start_node_name, link.end_node_name] for _, link in wn.links()],
            "pressure_sensors": PARAMS["sensors"]["pressure"],
            "sources": wn.reservoir_name_list, "tanks": wn.tank_name_list}


@app.get("/api/scenarios")
def scenarios(season: str = "winter", role: str = Depends(require("viewer"))):
    df = load_season(season)
    info = df.groupby("scenario_id").agg(leak=("leak", "max"), node=("leak_node", "first"),
                                         area=("leak_area", "first")).reset_index()
    info["node"] = info["node"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    return info.to_dict(orient="records")


@app.get("/api/scenarios/{scenario_id}")
def scenario(scenario_id: str, season: str = "winter", role: str = Depends(require("viewer"))):
    df = load_season(season)
    rows = df[df["scenario_id"] == scenario_id]
    if rows.empty:
        raise HTTPException(404, "unknown scenario")
    return {"timestamps": rows["timestamp"].tolist(), "leak": rows["leak"].tolist(),
            "valve_open": rows["valve_open"].tolist(), "valve_minutes": rows["valve_minutes"].tolist(),
            "sensors": {s: rows[s].round(3).tolist() for s in SENSORS}}


@app.get("/reports/drift", include_in_schema=False)
def drift_report():
    """Full Evidently report (opened in a new tab, so no API key header)."""
    path = ROOT / "reports" / "drift_report.html"
    if not path.exists():
        raise HTTPException(404, "No drift report yet. Run a monitoring cycle first.")
    return FileResponse(path)


@app.get("/api/operations")
def operations(role: str = Depends(require("viewer"))):
    def read(path, last=200):
        if not path.exists():
            return []
        return [json.loads(x) for x in path.read_text().splitlines()[-last:] if x.strip()]
    requests_log = read(LOGS / "api.log", 3000)
    per_minute = {}
    for r in requests_log:
        minute = r["time"][:16]
        m = per_minute.setdefault(minute, {"minute": minute, "requests": 0, "errors": 0, "latency": 0.0})
        m["requests"] += 1
        m["errors"] += r["status"] >= 400
        m["latency"] += r["latency_ms"]
    traffic = [{**m, "latency": round(m["latency"] / m["requests"], 1)} for m in per_minute.values()][-60:]
    return {"incidents": read(LOGS / "incidents.jsonl")[::-1], "audit": read(LOGS / "audit.log", 50)[::-1],
            "traffic": traffic}


# the website itself (must be mounted last so it doesn't hide the API routes)
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
