"""Flux Prefect de scoring d'un lot avec le modèle du registre MLflow.

Lancement (serveur MLflow démarré, modèle déjà enregistré par train_flow) :
    uv run python -m training.score_flow
"""

import os
from pathlib import Path

import mlflow
import pandas as pd
from prefect import flow, get_run_logger, task
from sklearn.pipeline import Pipeline

from training.data import load_data

ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = "models:/employee-attrition@champion"
TARGET = "LeaveOrNot"


@task
def load_batch(input_path: Path) -> pd.DataFrame:
    batch = load_data(input_path)
    get_run_logger().info("%d lignes à scorer depuis %s", len(batch), input_path)
    # En production la cible n'est pas connue : on l'écarte si elle est présente
    return batch.drop(columns=[TARGET], errors="ignore")


@task
def load_model(model_uri: str) -> Pipeline:
    mlflow.set_tracking_uri(TRACKING_URI)
    get_run_logger().info("Chargement du modèle %s", model_uri)
    return mlflow.sklearn.load_model(model_uri)


@task
def score(model: Pipeline, batch: pd.DataFrame) -> pd.DataFrame:
    scored = batch.copy()
    scored["proba_depart"] = model.predict_proba(batch)[:, 1]
    scored["prediction"] = model.predict(batch)
    return scored


@task
def save(scored: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    get_run_logger().info(
        "%d prédictions écrites dans %s (%.1f %% de départs prédits)",
        len(scored),
        output_path,
        100 * scored["prediction"].mean(),
    )
    return output_path


@flow(name="scoring-attrition", log_prints=True)
def score_flow(
    input_path: Path = ROOT / "data" / "employees.csv",
    output_path: Path = ROOT / "data" / "predictions.csv",
    model_uri: str = MODEL_URI,
) -> Path:
    batch = load_batch(input_path)
    model = load_model(model_uri)
    scored = score(model, batch)
    return save(scored, output_path)


if __name__ == "__main__":
    score_flow()
