"""
Role-based access control with API keys, plus an audit log.

    viewer   - can look at everything (website, models, drift)
    operator - viewer + can send sensor data for predictions
    admin    - operator + can promote / roll back / reload models

Keys come from environment variables. The defaults are only for local testing.
"""
import json
import os
from datetime import datetime

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from src.config import LOGS

LEVEL = {"viewer": 1, "operator": 2, "admin": 3}
KEYS = {os.getenv(f"AQUA_{role.upper()}_KEY", f"{role}-dev-key"): role for role in LEVEL}

header = APIKeyHeader(name="X-API-Key", auto_error=False)


def audit(role, action, detail=""):
    entry = {"time": datetime.now().isoformat(timespec="seconds"), "role": role, "action": action, "detail": detail}
    with open(LOGS / "audit.log", "a") as f:
        f.write(json.dumps(entry) + "\n")


def current_role(key: str = Security(header)):
    if key not in KEYS:
        raise HTTPException(401, "Missing or wrong API key (send it in the X-API-Key header)")
    return KEYS[key]


def require(min_role):
    def check(role: str = Depends(current_role)):
        if LEVEL[role] < LEVEL[min_role]:
            audit(role, "denied", f"needs {min_role}")
            raise HTTPException(403, f"The {role} role cannot do this. It needs {min_role}.")
        return role
    return check
