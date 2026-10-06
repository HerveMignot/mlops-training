"""Flux Prefect d'entraînement du modèle d'attrition, tracé dans MLflow.

Lancement (serveur MLflow démarré) :
    uv run python -m training.train_flow
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
def fit(train_df: pd.DataFrame, n_estimators: int, max_depth: int) -> Pipeline:
    X = train_df.drop(columns=[TARGET])
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical = X.select_dtypes(include=["number"]).columns.tolist()
    model = build_pipeline(categorical, numerical)
    model.set_params(
        classifier__n_estimators=n_estimators, classifier__max_depth=max_depth
    )
    return model.fit(X, train_df[TARGET])


@task
def evaluate(model: Pipeline, test_df: pd.DataFrame) -> dict[str, float]:
    X = test_df.drop(columns=[TARGET])
    metrics = {
        "test_auc": roc_auc_score(test_df[TARGET], model.predict_proba(X)[:, 1]),
        "test_f1": f1_score(test_df[TARGET], model.predict(X)),
    }
    get_run_logger().info("Métriques de test : %s", metrics)
    return metrics


@task
def log_and_register(
    model: Pipeline,
    params: dict,
    metrics: dict[str, float],
    example: pd.DataFrame,
    promote: bool,
) -> str:
    """Trace le run dans MLflow et enregistre le modèle dans le registre."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=example,
        )
    version = info.registered_model_version
    if promote:
        # L'alias "champion" désigne la version que le flux de scoring utilisera
        MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)
    get_run_logger().info("Modèle %s enregistré en version %s", MODEL_NAME, version)
    return str(version)


@flow(name="entrainement-attrition", log_prints=True)
def train_flow(
    data_path: Path = ROOT / "data" / "employees.csv",
    test_size: float = 0.2,
    n_estimators: int = 200,
    max_depth: int = 8,
    promote: bool = True,
) -> str:
    df = load(data_path)
    train_df, test_df = split(df, test_size)
    model = fit(train_df, n_estimators, max_depth)
    metrics = evaluate(model, test_df)
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "test_size": test_size,
        "n_train": len(train_df),
    }
    example = train_df.drop(columns=[TARGET]).head(3)
    return log_and_register(model, params, metrics, example, promote)


if __name__ == "__main__":
    train_flow()
