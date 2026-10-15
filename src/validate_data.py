"""Validation du jeu de données : étape B1 du TP.

Le schéma ci-dessous est le contrat entre les données et le reste du pipeline.
Si une donnée sort du contrat, le job échoue (code retour 1) AVANT l'entraînement :
on ne gaspille ni minutes de calcul ni confiance dans un modèle entraîné sur des
données corrompues.

Usage :
    python src/validate_data.py data/employees.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:  # pandera récent : chemin d'import dédié à pandas
    import pandera.pandas as pa
except ImportError:  # anciennes versions
    import pandera as pa

from pandera.errors import SchemaErrors

REFERENCE_YEAR = 2018  # dernière année d'arrivée plausible
MIN_ROWS = 1000  # en dessous, le jeu est probablement tronqué
LEAVE_RATE_RANGE = (0.20, 0.50)  # taux de départs attendu (~34 %)
MAX_EXAMPLES = 3


def build_schema() -> pa.DataFrameSchema:
    """Schéma des colonnes (types, domaines, bornes) et règles globales."""
    return pa.DataFrameSchema(
        {
            "Education": pa.Column(str, pa.Check.isin(["Bachelors", "Masters", "PHD"])),
            "JoiningYear": pa.Column(int, pa.Check.in_range(2000, REFERENCE_YEAR)),
            "City": pa.Column(str, pa.Check.isin(["Bangalore", "Pune", "New Delhi"])),
            "PaymentTier": pa.Column(int, pa.Check.isin([1, 2, 3])),
            "Age": pa.Column(int, pa.Check.in_range(18, 65)),
            "Gender": pa.Column(str, pa.Check.isin(["Male", "Female"])),
            "EverBenched": pa.Column(str, pa.Check.isin(["Yes", "No"])),
            "ExperienceInCurrentDomain": pa.Column(int, pa.Check.in_range(0, 40)),
            "LeaveOrNot": pa.Column(int, pa.Check.isin([0, 1])),
        },
        checks=[
            pa.Check(
                lambda df: len(df) >= MIN_ROWS,
                error=f"le jeu de données contient moins de {MIN_ROWS} lignes",
            ),
            pa.Check(
                lambda df: LEAVE_RATE_RANGE[0] <= df["LeaveOrNot"].mean() <= LEAVE_RATE_RANGE[1],
                error=f"taux de départs hors de la plage attendue {LEAVE_RATE_RANGE}",
            ),
            pa.Check(
                lambda df: df["ExperienceInCurrentDomain"] <= df["Age"],
                error="expérience supérieure à l'âge (colonnes probablement mélangées)",
            ),
        ],
        strict=True,  # une colonne en trop ou manquante est une rupture de contrat
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Valide le CSV des collaborateurs.")
    parser.add_argument("path", help="chemin du fichier CSV, ex. data/employees.csv")
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.is_file():
        print(f"ÉCHEC : fichier introuvable : {path}")
        return 1

    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        print(f"ÉCHEC : CSV illisible : {exc}")
        return 1

    try:
        build_schema().validate(df, lazy=True)  # lazy : remonte TOUTES les violations
    except SchemaErrors as exc:
        cases = exc.failure_cases
        print(f"ÉCHEC : {cases['check'].nunique()} règle(s) violée(s) dans {path}")
        for check, group in cases.groupby("check", dropna=False):
            columns = sorted({str(c) for c in group["column"].dropna()})
            n_rows = int(group["index"].nunique()) if group["index"].notna().any() else len(group)
            where = f"colonne={columns[0]}" if len(columns) == 1 else "règle globale"
            line = f"  - {where} | règle={check} | {n_rows} ligne(s)"
            if len(columns) == 1:  # les exemples n'ont de sens que pour une règle de colonne
                examples = ", ".join(str(v) for v in group["failure_case"].head(MAX_EXAMPLES))
                line += f" | ex. : {examples}"
            print(line)
        return 1

    duplicates = int(df.duplicated().sum())
    if duplicates:
        print(f"info : {duplicates} lignes dupliquées (non bloquant, à discuter)")
    print(f"OK : {len(df)} lignes, {df['LeaveOrNot'].mean():.1%} de départs, schéma conforme.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
