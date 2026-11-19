"""
Model registry (MLflow) and lineage log.

- Each training run creates a new version of the model.
- Alias "champion"   = the version used in production.
- Alias "challenger" = a newer version that is waiting (used for shadow/canary).
- The champion is also copied to models/production/ so the API can load it fast.
- Every promote / rollback is written to models/lineage.jsonl (who, when, which data).

    python -m src.registry list
    python -m src.registry promote 2
    python -m src.registry rollback
"""
import json
import shutil
import sys
from datetime import datetime

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from src.config import MODELS, load_params, tracking_uri

LINEAGE = MODELS / "lineage.jsonl"


def client():
    mlflow.set_tracking_uri(tracking_uri())
    return MlflowClient()


def name():
    return load_params()["model"]["name"]


def get_alias(alias):
    try:
        return int(client().get_model_version_by_alias(name(), alias).version)
    except Exception:
        return None


def tag_version(version, tags):
    c = client()
    for k, v in tags.items():
        c.set_model_version_tag(name(), str(version), k, str(v))


def load(version):
    client()
    return mlflow.sklearn.load_model(f"models:/{name()}/{version}")


def export(version, folder):
    """Save the model + a small info file where the API can find it."""
    tags = client().get_model_version(name(), str(version)).tags
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    joblib.dump(load(version), folder / "model.joblib")
    info = {"version": version, "exported_at": datetime.now().isoformat(timespec="seconds"), **tags}
    (folder / "info.json").write_text(json.dumps(info, indent=2))


def log_event(event, version, user, **extra):
    tags = client().get_model_version(name(), str(version)).tags
    entry = {"time": datetime.now().isoformat(timespec="seconds"), "event": event, "version": version,
             "user": user, "data_file": tags.get("data_file"), "data_md5": tags.get("data_md5"),
             "roc_auc": tags.get("roc_auc"), **extra}
    with open(LINEAGE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_lineage():
    if not LINEAGE.exists():
        return []
    return [json.loads(line) for line in LINEAGE.read_text().splitlines() if line.strip()]


def promote(version, user="admin", reason=""):
    previous = get_alias("champion")
    client().set_registered_model_alias(name(), "champion", str(version))
    export(version, MODELS / "production")
    log_event("promoted", version, user, previous=previous, reason=reason)


def set_challenger(version):
    client().set_registered_model_alias(name(), "challenger", str(version))
    export(version, MODELS / "challenger")


def rollback(user="admin", reason="manual rollback"):
    """Go back to the version that was champion before the current one."""
    current = get_alias("champion")
    for e in reversed(read_lineage()):
        if e["event"] in ("promoted", "rolled_back") and e["version"] == current and e.get("previous"):
            target = e["previous"]
            client().set_registered_model_alias(name(), "champion", str(target))
            export(target, MODELS / "production")
            log_event("rolled_back", target, user, previous=current, reason=reason)
            return target
    raise RuntimeError("there is no earlier version to roll back to")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "list"
    if action == "promote":
        promote(int(sys.argv[2]), "cli", "manual promote")
    elif action == "rollback":
        print("now using version", rollback("cli"))
    else:
        for mv in client().search_model_versions(f"name='{name()}'"):
            mv = client().get_model_version(name(), mv.version)
            print(f"v{mv.version}  aliases={mv.aliases}  roc_auc={mv.tags.get('roc_auc')}  "
                  f"data={mv.tags.get('data_file')}")
