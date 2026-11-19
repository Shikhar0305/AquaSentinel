
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from mlflow import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


ROOT_DIR = Path(__file__).resolve().parent

MLFLOW_DB = ROOT_DIR / "mlflow.db"
TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"

MODEL_PATH = ROOT_DIR / "models" / "champion.joblib"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

mlflow.set_experiment("AquaSentinel-Leak-Detection")

MODEL_REGISTRY_NAME = "AquaSentinel_Leak_Detector"

# MLflow saves scikit-learn models with skops, which only loads types it trusts.
# Tree models store their nodes in this type; we trust it because we trained the models ourselves.
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def get_models():
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000)
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),

        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),
    }


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
    }


def train_models(X_train, X_test, y_train, y_test):
    results = []
    model_uris = {}
    trained_models = {}

    for model_name, model in get_models().items():

        with mlflow.start_run(run_name=model_name) as run:

            model.fit(X_train, y_train)

            metrics = evaluate_model(
                model,
                X_test,
                y_test
            )

            mlflow.log_params({
                "model_name": model_name,
                "number_of_features": X_train.shape[1],
                "training_rows": X_train.shape[0],
                "testing_rows": X_test.shape[0],
            })

            mlflow.log_metrics(metrics)

            signature = infer_signature(
                X_train,
                model.predict(X_train.head(5))
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=X_train.head(5),
                skops_trusted_types=TRUSTED_TYPES
            )

            model_uri = f"runs:/{run.info.run_id}/model"

            model_uris[model_name] = model_uri
            trained_models[model_name] = model

            results.append({
                "Model": model_name,
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "ROC_AUC": metrics["ROC_AUC"],
                "Run_ID": run.info.run_id,
            })

            print(f"\n{model_name}")
            print("-" * 30)

            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

    return results, model_uris, trained_models


def register_best_model(results, model_uris, trained_models):
    best_model = max(
        results,
        key=lambda result: result["F1"]
    )

    best_name = best_model["Model"]
    best_uri = model_uris[best_name]

    print("\nBest model:", best_name)
    print("Best F1:", round(best_model["F1"], 4))

    registered_model = mlflow.register_model(
        model_uri=best_uri,
        name=MODEL_REGISTRY_NAME
    )

    client = MlflowClient()

    client.set_registered_model_alias(
        name=MODEL_REGISTRY_NAME,
        alias="champion",
        version=registered_model.version
    )

    # export the champion so the API can load it without MLflow
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models[best_name], MODEL_PATH)

    print("\nModel registered successfully!")
    print("Registered model:", registered_model.name)
    print("Model version:", registered_model.version)
    print("Champion alias assigned to version:", registered_model.version)
    print("Champion exported to:", MODEL_PATH)

    return registered_model


def print_results(results):
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    sorted_results = sorted(
        results,
        key=lambda result: result["F1"],
        reverse=True
    )

    for result in sorted_results:
        print(
            f"\n{result['Model']}\n"
            f"Precision: {result['Precision']:.4f}\n"
            f"Recall:    {result['Recall']:.4f}\n"
            f"F1:        {result['F1']:.4f}\n"
            f"ROC AUC:   {result['ROC_AUC']:.4f}"
        )


def run_training(X_train, X_test, y_train, y_test):
    results, model_uris, trained_models = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print_results(results)

    registered_model = register_best_model(
        results,
        model_uris,
        trained_models
    )

    return results, registered_model
