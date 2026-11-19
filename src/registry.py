"""
Model registry helpers (MLflow) + model lineage log.

- Every trained model is registered as a new version of `aquasentinel-leak-detector`.
- Alias "champion"   = model in production.
- Alias "challenger" = newest model, can be run in shadow / canary mode by the API.
- On promotion the model is exported to models/production/ so the API can load it
  without talking to MLflow on every start.
- Every promotion / rollback / rejection is appended to models/lineage.jsonl
  (who, when, which version, trained on which data) for governance.

CLI:
    python -m src.registry list
    python -m src.registry promote --version 3 --user admin
    python -m src.registry rollback --user admin
    python -m src.registry lineage
"""
import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from src.utils import MODELS_DIR, get_logger, get_tracking_uri, load_params, save_json

log = get_logger("registry")
LINEAGE_FILE = MODELS_DIR / "lineage.jsonl"
PRODUCTION_DIR = MODELS_DIR / "production"
CHALLENGER_DIR = MODELS_DIR / "challenger"


def get_client():
    mlflow.set_tracking_uri(get_tracking_uri())
    return MlflowClient()


def model_name():
    return load_params()["model"]["name"]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def get_alias_version(alias):
    client = get_client()
    try:
        return int(client.get_model_version_by_alias(model_name(), alias).version)
    except Exception:
        return None


def record_lineage(event, version, user="system", **extra):
    client = get_client()
    entry = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        "model": model_name(),
        "version": version,
        "user": user,
    }
    if version is not None:
        mv = client.get_model_version(model_name(), str(version))
        entry["run_id"] = mv.run_id
        entry["data_md5"] = mv.tags.get("data_md5")
        entry["data_file"] = mv.tags.get("data_file")
        entry["git_commit"] = mv.tags.get("git_commit")
        entry["test_f1"] = mv.tags.get("test_f1")
        entry["test_roc_auc"] = mv.tags.get("test_roc_auc")
    entry.update(extra)
    with open(LINEAGE_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log.info("lineage: %s v%s by %s", event, version, user)
    return entry


def read_lineage():
    if not LINEAGE_FILE.exists():
        return []
    with open(LINEAGE_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def export_model(version, target_dir):
    """Download a registered model version and save it as a joblib bundle for the API."""
    get_client()
    model = mlflow.sklearn.load_model(f"models:/{model_name()}/{version}")
    mv = MlflowClient().get_model_version(model_name(), str(version))
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    joblib.dump(model, target_dir / "model.joblib")
    save_json({
        "name": model_name(),
        "version": int(version),
        "run_id": mv.run_id,
        "data_md5": mv.tags.get("data_md5"),
        "data_file": mv.tags.get("data_file"),
        "test_f1": mv.tags.get("test_f1"),
        "test_roc_auc": mv.tags.get("test_roc_auc"),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, target_dir / "meta.json")
    return model


def promote(version, user="system", reason=""):
    client = get_client()
    previous = get_alias_version("champion")
    client.set_registered_model_alias(model_name(), "champion", str(version))
    export_model(version, PRODUCTION_DIR)
    record_lineage("promoted", version, user, previous_version=previous, reason=reason)
    return version


def set_challenger(version):
    client = get_client()
    client.set_registered_model_alias(model_name(), "challenger", str(version))
    export_model(version, CHALLENGER_DIR)


def rollback(user="system", reason="manual rollback"):
    """Go back to the model that was champion before the current one."""
    current = get_alias_version("champion")
    history = [e for e in read_lineage() if e["event"] in ("promoted", "rolled_back")]
    # walk back through promotions to find the last version different from current
    candidates = [e.get("previous_version") for e in reversed(history)
                  if e["version"] == current and e.get("previous_version")]
    if not candidates:
        raise RuntimeError("No previous model version to roll back to")
    target = candidates[0]
    client = get_client()
    client.set_registered_model_alias(model_name(), "champion", str(target))
    export_model(target, PRODUCTION_DIR)
    record_lineage("rolled_back", target, user, previous_version=current, reason=reason)
    return target


def list_versions():
    client = get_client()
    rows = []
    for found in client.search_model_versions(f"name='{model_name()}'"):
        mv = client.get_model_version(model_name(), found.version)  # includes aliases
        aliases = mv.aliases or []
        rows.append({"version": int(mv.version), "aliases": ",".join(aliases),
                     "run_id": mv.run_id[:8], "test_f1": mv.tags.get("test_f1"),
                     "test_roc_auc": mv.tags.get("test_roc_auc"),
                     "data_md5": (mv.tags.get("data_md5") or "")[:8]})
    return sorted(rows, key=lambda r: r["version"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["list", "promote", "rollback", "lineage"])
    parser.add_argument("--version", type=int)
    parser.add_argument("--user", default="cli")
    args = parser.parse_args()

    if args.action == "list":
        for r in list_versions():
            print(r)
    elif args.action == "promote":
        promote(args.version, args.user, reason="manual promote")
    elif args.action == "rollback":
        print("rolled back to version", rollback(args.user))
    else:
        for e in read_lineage():
            print(e)


if __name__ == "__main__":
    main()
