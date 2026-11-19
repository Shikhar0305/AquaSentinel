
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from src.inference import predict_leak
from src.leak_analysis import analyze_leak

app = FastAPI(
    title="AquaSentinel",
    description="Leak detection API for water supply zones with intermittent supply",
    version="1.0.0"
)


class LeakRequest(BaseModel):
    zone_id: str = Field(default="zone-01", min_length=1)
    inflow_history: list[float] = Field(
        min_length=12,
        description="Total inflow into the zone in L/s, one reading every 15 minutes, oldest first"
    )
    valve_open: bool = Field(description="Is the zone supply valve open right now?")
    valve_minutes: int = Field(ge=0, description="Minutes since the valve last opened or closed")

    @field_validator("inflow_history")
    @classmethod
    def validate_inflow_history(cls, values):
        if any(not 0 <= value <= 5000 for value in values):
            raise ValueError("Inflow values must be between 0 and 5000 L/s")

        return values


@app.get("/")
def root():
    return {
        "application": "AquaSentinel",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "AquaSentinel_Leak_Detector",
        "alias": "champion"
    }


@app.post("/predict")
def predict(request: LeakRequest):
    probability = predict_leak(
        request.inflow_history,
        request.valve_open,
        request.valve_minutes
    )

    return {
        "zone_id": request.zone_id,
        "leak_probability": round(probability, 3),
        "leak_detected": probability >= 0.5
    }


@app.post("/analyze")
def analyze(request: LeakRequest):
    probability = predict_leak(
        request.inflow_history,
        request.valve_open,
        request.valve_minutes
    )

    result = analyze_leak(
        probability,
        request.inflow_history,
        request.valve_open
    )

    return {
        "zone_id": request.zone_id,
        **result
    }
