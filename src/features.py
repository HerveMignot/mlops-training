"""Préparation des variables et construction du modèle.

Ce module est la SEULE définition de la préparation des données : il est importé
par l'entraînement, par les tests et par l'API d'inférence. Cela évite le décalage
entre entraînement et inférence (training/serving skew).

Choix délibéré : l'ancienneté est calculée par rapport à une année de référence
fixe (paramètre), jamais par rapport à la date du jour. Sinon le même code
donnerait des résultats différents d'une année sur l'autre.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "LeaveOrNot"

RAW_COLUMNS = [
    "Education",
    "JoiningYear",
    "City",
    "PaymentTier",
    "Age",
    "Gender",
    "EverBenched",
    "ExperienceInCurrentDomain",
]

CATEGORICAL = ["Education", "City", "Gender"]
NUMERIC = ["PaymentTier", "Age", "ExperienceInCurrentDomain", "Tenure", "EverBenched"]


def add_features(df: pd.DataFrame, reference_year: int = 2018) -> pd.DataFrame:
    """Transforme les colonnes brutes en variables du modèle.

    - ``Tenure`` : ancienneté = année de référence - année d'arrivée ;
    - ``EverBenched`` : "Yes"/"No" -> 1/0 (toute autre valeur donne 0, d'où
      l'importance de la validation des données en amont).

    La cible et les colonnes inutiles sont écartées : la sortie contient
    exactement les colonnes attendues par le modèle, dans un ordre fixe.
    """
    out = df.copy()
    out["Tenure"] = reference_year - out["JoiningYear"]
    is_benched = out["EverBenched"].astype(str).str.strip().str.lower() == "yes"
    out["EverBenched"] = is_benched.astype(int)
    return out[CATEGORICAL + NUMERIC]


def build_model(model_params: dict, seed: int) -> Pipeline:
    """Construit le pipeline scikit-learn (encodage + classifieur).

    ``model_params`` est la section ``model`` de params.yaml.
    """
    kind = model_params["type"]
    if kind == "random_forest":
        clf = RandomForestClassifier(random_state=seed, n_jobs=1, **model_params["random_forest"])
    elif kind == "logistic_regression":
        clf = LogisticRegression(random_state=seed, **model_params["logistic_regression"])
    else:
        raise ValueError(f"Type de modèle inconnu : {kind!r}")

    preprocessing = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    return Pipeline([("prep", preprocessing), ("clf", clf)])
