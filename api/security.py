"""
Role-based access control with API keys + audit log.

Roles (each role can do everything the roles below it can):
    admin    - reload, promote, roll back models
    operator - send sensor data and get predictions
    viewer   - read model info, lineage, drift status

Keys come from environment variables. The defaults are ONLY for local development.
"""
import json
import os
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from src.utils import LOGS_DIR, get_logger

log = get_logger("security")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

ROLE_LEVEL = {"viewer": 1, "operator": 2, "admin": 3}
DEV_KEYS = {"admin": "admin-dev-key", "operator": "operator-dev-key", "viewer": "viewer-dev-key"}


def load_keys():
    keys = {}
    for role, default in DEV_KEYS.items():
        key = os.getenv(f"AQUA_{role.upper()}_KEY", default)
        if key == default:
            log.warning("using default dev key for role '%s' - set AQUA_%s_KEY in production", role, role.upper())
        keys[key] = role
    return keys


API_KEYS = load_keys()


def audit(user_role, action, detail=""):
    entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "role": user_role, "action": action, "detail": detail}
    with open(LOGS_DIR / "audit.log", "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_role(api_key: str = Security(api_key_header)):
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Missing or invalid API key (header X-API-Key)")
    return API_KEYS[api_key]


def require_role(min_role):
    def checker(role: str = Depends(get_role)):
        if ROLE_LEVEL[role] < ROLE_LEVEL[min_role]:
            audit(role, "denied", f"needs {min_role}")
            raise HTTPException(status_code=403, detail=f"Role '{role}' is not allowed, needs '{min_role}'")
        return role
    return checker
