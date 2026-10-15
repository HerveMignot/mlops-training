"""Entraînement du modèle d'attrition : étape B2 du TP.

Produit dans le dossier de sortie :
  - model.joblib   le pipeline scikit-learn entraîné (encodage + classifieur) ;
  - metrics.json   les métriques sur le jeu de test, avec quelques métadonnées.

Le run est aussi journalisé dans MLflow (paramètres, métriques, tags du commit).
Le backend est piloté par des variables d'environnement, ce qui permet d'utiliser
le même script en local et en CI :
  MLFLOW_TRACKING_URI      défaut : sqlite:///mlflow.db
  MLFLOW_EXPERIMENT_NAME   défaut : attrition

Usage :
    python src/train.py --data data/employees.csv --params params.yaml --out artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# `python src/train.py` place src/ (et non la racine) dans sys.path : on ajoute la
# racine du dépôt pour pouvoir écrire `from src.features import ...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import TARGET, add_features, build_model  # noqa: E402

CI_TAGS = {
    "git.commit": "CI_COMMIT_SHA",
    "git.branch": "CI_COMMIT_REF_NAME",
    "ci.pipeline_id": "CI_PIPELINE_ID",
    "ci.job_url": "CI_JOB_URL",
    "ci.project_url": "CI_PROJECT_URL",
}


def compute_metrics(y_true: pd.Series, proba: pd.Series, threshold: float = 0.5) -> dict:
    """Métriques de la classe positive (1 = le collaborateur part)."""
    pred = (proba >= threshold).astype(int)
    values = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }
    return {name: round(float(value), 4) for name, value in values.items()}


def log_to_mlflow(
    params: dict, metrics: dict, meta: dict, out_dir: Path, params_path: Path
) -> None:
    import mlflow  # import tardif : lourd, inutile avec --no-mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "attrition"))

    kind = params["model"]["type"]
    logged_params = {
        "model.type": kind,
        "split.test_size": params["split"]["test_size"],
        "split.seed": params["split"]["seed"],
        "data.reference_year": params["data"]["reference_year"],
        **{f"model.{k}": v for k, v in params["model"][kind].items()},
    }
    tags = {tag: os.environ[var] for tag, var in CI_TAGS.items() if var in os.environ}

    with mlflow.start_run(run_name=meta["commit"]):
        mlflow.log_params(logged_params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        mlflow.log_artifact(str(out_dir / "model.joblib"))
        mlflow.log_artifact(str(params_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entraîne le modèle d'attrition.")
    parser.add_argument("--data", required=True, help="CSV des collaborateurs")
    parser.add_argument("--params", required=True, help="fichier params.yaml")
    parser.add_argument("--out", required=True, help="dossier de sortie (créé si besoin)")
    parser.add_argument("--no-mlflow", action="store_true", help="ne pas journaliser dans MLflow")
    args = parser.parse_args(argv)

    params_path = Path(args.params)
    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    seed = params["split"]["seed"]

    df = pd.read_csv(args.data)
    features = add_features(df, params["data"]["reference_year"])
    target = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=params["split"]["test_size"],
        stratify=target,  # conserve ~34 % de départs dans chaque jeu
        random_state=seed,
    )

    model = build_model(params["model"], seed)
    model.fit(x_train, y_train)
    metrics = compute_metrics(y_test, model.predict_proba(x_test)[:, 1])

    meta = {
        "commit": os.environ.get("CI_COMMIT_SHORT_SHA", "local"),
        "model_type": params["model"]["type"],
        "seed": seed,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "positive_rate": round(float(target.mean()), 4),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "meta": meta}, indent=2) + "\n", encoding="utf-8"
    )

    if not args.no_mlflow:
        log_to_mlflow(params, metrics, meta, out_dir, params_path)

    print(f"Modèle {meta['model_type']} entraîné sur {meta['n_train']} lignes.")
    for name, value in metrics.items():
        print(f"  {name:10} {value:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
