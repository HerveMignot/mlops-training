"""Variante de train_flow.py utilisant l'autologging MLflow pour scikit-learn.

Différence avec train_flow.py : le run MLflow est ouvert dans la tâche `fit`,
et c'est `mlflow.sklearn.autolog()` qui enregistre les hyperparamètres, les
métriques d'entraînement et le modèle. Seules les métriques de test et
l'inscription au registre restent explicites.

Lancement (serveur MLflow démarré) :
    uv run python -m training.train_flow_autolog
"""

import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from prefect import flow, get_run_logger, task
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from training.data import load_data
from training.train import build_pipeline

ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "employee_attrition_prefect"
MODEL_NAME = "employee-attrition"
TARGET = "LeaveOrNot"


@task
def load(data_path: Path) -> pd.DataFrame:
    df = load_data(data_path)
    get_run_logger().info("%d lignes chargées depuis %s", len(df), data_path)
    return df


@task
def split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=42
    )
    return train_df, test_df


@task
def fit(
    train_df: pd.DataFrame, n_estimators: int, max_depth: int
) -> tuple[Pipeline, str]:
    """Entraîne le modèle ; l'autologging trace le run pendant le fit."""
    X = train_df.drop(columns=[TARGET])
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical = X.select_dtypes(include=["number"]).columns.tolist()
    model = build_pipeline(categorical, numerical)
    model.set_params(
        classifier__n_estimators=n_estimators, classifier__max_depth=max_depth
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    # log_post_training_metrics=False : sinon tout appel ultérieur à sklearn.metrics
    # ajoute au run des métriques aux noms automatiques, en doublon des nôtres
    mlflow.sklearn.autolog(log_post_training_metrics=False)
    with mlflow.start_run(run_name="autolog") as run:
        model.fit(X, train_df[TARGET])
    return model, run.info.run_id


@task
def evaluate(model: Pipeline, test_df: pd.DataFrame, run_id: str) -> dict[str, float]:
    """Calcule les métriques de test et les ajoute au run ouvert par `fit`."""
    X = test_df.drop(columns=[TARGET])
    metrics = {
        "test_auc": roc_auc_score(test_df[TARGET], model.predict_proba(X)[:, 1]),
        "test_f1": f1_score(test_df[TARGET], model.predict(X)),
    }
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
    get_run_logger().info("Métriques de test : %s", metrics)
    return metrics


@task
def register(run_id: str, promote: bool) -> str:
    """Inscrit au registre le modèle que l'autologging a enregistré dans le run."""
    version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME).version
    if promote:
        # L'alias "champion" désigne la version que le flux de scoring utilisera
        MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)
    get_run_logger().info("Modèle %s enregistré en version %s", MODEL_NAME, version)
    return str(version)


@flow(name="entrainement-attrition-autolog", log_prints=True)
def train_flow_autolog(
    data_path: Path = ROOT / "data" / "employees.csv",
    test_size: float = 0.2,
    n_estimators: int = 200,
    max_depth: int = 8,
    promote: bool = True,
) -> str:
    df = load(data_path)
    train_df, test_df = split(df, test_size)
    model, run_id = fit(train_df, n_estimators, max_depth)
    evaluate(model, test_df, run_id)
    return register(run_id, promote)


if __name__ == "__main__":
    train_flow_autolog()
