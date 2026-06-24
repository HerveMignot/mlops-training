# Fiches de TP — Formation MLOps

Fil rouge unique : un modèle de *credit scoring* (classification *good / bad payer*) ou modèle de rétention de collaborateurs (classification *va rester / va partir*).


> Jeu de données : **German Credit** (UCI), récupérable sans authentification via `scikit-learn` :
> `fetch_openml("credit-g", version=1, as_frame=True)` → 1000 lignes, cible `class` ∈ {good, bad}.


---

# TP 1 (Jour 2) — Projet Python ML moderne + outillage + MLflow

**Durée indicative : 3h30–4h.** Objectif : transformer un script de notebook en **projet reproductible, versionné et traçable**.

## TP 1.0 — Prérequis (à vérifier en début de séance)
```bash
python --version        # 3.11, 3.12 ou 3.13
git --version
code --version          # VS Code + extensions Python & Jupyter
pip install uv          # si uv n'est pas déjà installé
```
*Point important : rappeler pourquoi un environnement isolé (reproductibilité, sécurité, conflits de versions).*

## TP 1.1 — Initialiser le projet et le dépôt Git
```bash
uv init scoring --python 3.12
cd scoring
git init
```
Créer l'arborescence cible :
```
scoring/
├── data/                # données (versionnées via DVC, pas Git)
├── src/scoring/         # code source (modulaire)
│   ├── __init__.py
│   ├── data.py          # chargement / préparation
│   └── train.py         # entraînement + tracking
├── tests/
├── pyproject.toml
└── .gitignore
```
**Livrable intermédiaire :** premier commit « squelette du projet ».

## TP 1.2 — Gérer les dépendances avec `uv`
```bash
uv add scikit-learn pandas mlflow
uv add --dev pytest pre-commit ruff mypy
uv sync                 # synchronise l'environnement avec pyproject.toml
uv tree                 # visualiser l'arbre de dépendances
```
*Point important : toutes les dépendances vivent dans `pyproject.toml` (et `uv.lock`), jamais en `pip install` sauvage.*

Remarque : mlflow 3.14 présente un bogue à date, forcer l'utilisation de la version 3.13 dans ce cas.
```bash
uv add mlflow==3.13
uv sync
```


## TP 1.3 — Qualité de code avec `pre-commit`
Créer `.pre-commit-config.yaml` :
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.18
    hooks:
      - id: ruff            # lint + corrections
        args: [--fix]
      - id: ruff-format     # formatage (alternative à black)
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v2.1.0
    hooks:
      - id: mypy
```
Activer et tester :
```bash
uv run pre-commit install
uv run pre-commit run --all-files
```
**Exercice :** introduire volontairement un import inutilisé + une variable non typée, committer, observer le hook bloquer le commit, corriger.
*Notions : PEP 8, docstrings, typage statique, « tester tôt » (shift-left).*

## TP 1.4 — Versionner les données avec DVC
```bash
uv add dvc
dvc init
git commit -m "Initialize DVC"
```
Script de récupération `src/scoring/data.py` :
```python
from sklearn.datasets import fetch_openml

def load_raw(path="data/credit-g.csv"):
    """Télécharge le jeu German Credit et l'écrit en CSV."""
    df = fetch_openml("credit-g", version=1, as_frame=True).frame
    df.to_csv(path, index=False)
    return df

if __name__ == "__main__":
    load_raw()
```
Générer puis suivre la donnée avec DVC :
```bash
uv run python -m scoring.data
dvc add data/credit-g.csv
git add data/credit-g.csv.dvc data/.gitignore
git commit -m "data: Track raw data with DVC"
```
**Option (niveau 1 de maturité) — pipeline DVC** `dvc.yaml` :
```yaml
stages:
  prepare:
    cmd: python -m scoring.data
    outs:
      - data/credit-g.csv
  train:
    cmd: python -m scoring.train
    deps:
      - data/credit-g.csv
      - src/scoring/train.py
    outs:
      - models/model.joblib
```
```bash
dvc repro     # rejoue tout le pipeline ; dvc.lock fige les hash
```
*Point important : reproductibilité = code (Git) + données (DVC) + environnement (uv).*

## TP 1.5 — Suivi d'expériences avec MLflow (cœur du TP)
Lancer le serveur dans un terminal dédié :
```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```
Script d'entraînement `src/scoring/train.py` :
```python
import mlflow, joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("credit-scoring")

def main(n_estimators: int = 200, max_depth: int = 8) -> None:
    df = pd.read_csv("data/credit-g.csv")
    y = (df.pop("class") == "good").astype(int)
    X = df
    cat = X.select_dtypes("object").columns.tolist()
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat)],
        remainder="passthrough",
    )
    model = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42)),
    ])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    mlflow.sklearn.autolog()          # log auto des params/métriques/modèle
    with mlflow.start_run():
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        mlflow.log_metric("test_f1", f1_score(y_te, model.predict(X_te)))
        mlflow.log_metric("test_auc", roc_auc_score(y_te, proba))
        joblib.dump(model, "models/model.joblib")

if __name__ == "__main__":
    main()
```
Lancer plusieurs runs avec des hyperparamètres différents :
```bash
mkdir -p models
uv run python -m scoring.train      # run 1
# modifier n_estimators / max_depth, relancer pour générer d'autres runs
```
**Exercice :** dans l'UI MLflow (http://127.0.0.1:8080), comparer les runs, trier par `test_auc`, identifier la meilleure configuration.

## TP 1.6 — Model Registry
Dans l'UI : ouvrir le meilleur run → **Register Model** → nom `credit-scoring` → version 1.
Ou en code :
```python
import mlflow
result = mlflow.register_model(
    model_uri="runs:/<RUN_ID>/model", name="credit-scoring")
```
Affecter un alias (ex. `champion`) à la meilleure version dans l'UI.
*Notions : versions, alias, tags, traçabilité run ↔ modèle ↔ données.*

## TP 1.7 — Servir le modèle en local et le tester

S'assurer que les variables d'environnement sont correctement initialisées :

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
# ou
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
```
puis :

```bash
uv run mlflow models serve -m "models:/credit-scoring/1" -p 5001 --no-conda
```
Tester avec une requête :
```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

## Récapitulatif TP 1 — livrables & pièges

**Livrables :** dépôt Git propre (uv + pre-commit + DVC), runs MLflow comparés, 1 modèle dans le registre avec alias, modèle servi et testé en local.
**Pièges fréquents :** oublier `dvc add` avant le commit (données poussées dans Git) ; lancer le script sans le serveur MLflow démarré ; ne pas fixer `random_state` (résultats non reproductibles) ; committer `.venv/` (à mettre dans `.gitignore`).

---

# TP 2 — Industrialiser dans le cloud (GCP)

**Durée indicative : 3h30–4h.** Objectif : pousser le modèle du TP 1 vers **Vertex AI** / **Agent Platform**, le déployer et le surveiller. On reprend l'illustration « boucle Dev / boucle Ops » vue en cours.

> ⚠️ **Coûts** : un *endpoint* Vertex AI facture tant qu'il tourne. Prévoir le **nettoyage en fin de séance** (section finale). Utiliser le crédit d'essai GCP ou un projet pédagogique dédié.

## TP 2.0 — Prérequis
- Un projet GCP avec **facturation activée** (`PROJECT_ID`).
- **gcloud CLI** installé localement.
- Le fichier `models/model.joblib` produit au TP 1.

## TP 2.1 — Authentification et configuration
```bash
gcloud auth login
gcloud auth application-default login      # pour les SDK (DVC, aiplatform)
gcloud config set project PROJECT_ID
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com
```

## TP 2.2 — Stockage : bucket GCS + remote DVC
```bash
export REGION=europe-west1
export BUCKET=gs://${PROJECT_ID}-mlops
gcloud storage buckets create $BUCKET --location=$REGION
```
Brancher le remote DVC sur le bucket (les données suivent le modèle dans le cloud) :
```bash
uv add "dvc[gs]"
dvc remote add -d gcs ${BUCKET}/dvc
dvc push          # envoie les données versionnées vers GCS
```
*Notion : même donnée, même hash, accessible depuis n'importe quel environnement → symétrie dev/prod.*

## TP 2.3 — (Option) Préparer les données dans BigQuery
Charger le CSV dans une table et faire une requête d'exploration :
```bash
bq mk --dataset ${PROJECT_ID}:scoring
bq load --autodetect --source_format=CSV \
  scoring.credit data/credit-g.csv
```
*Montre l'usage de BigQuery comme source de données dans la boucle Dev.*

## TP 2.4 — Suivi d'expériences côté cloud
Deux options à présenter :
1. **Vertex AI Experiments** (natif GCP) :
```python
from google.cloud import aiplatform
aiplatform.init(project="PROJECT_ID", location="europe-west1",
                experiment="credit-scoring")
```
2. **MLflow distant** (continuité avec le TP 1) : pointer `set_tracking_uri` vers un serveur MLflow hébergé (Cloud Run).

## TP 2.5 — Enregistrer le modèle dans Vertex AI Model Registry
Téléverser l'artefact puis l'enregistrer avec un conteneur de service pré-construit scikit-learn :
```bash
# Vertex attend un fichier nommé model.joblib dans un dossier GCS
gcloud storage cp models/model.joblib ${BUCKET}/model/model.joblib
```
```python
from google.cloud import aiplatform
aiplatform.init(project="PROJECT_ID", location="europe-west1",
                staging_bucket="gs://PROJECT_ID-mlops")

model = aiplatform.Model.upload(
    display_name="credit-scoring",
    artifact_uri="gs://PROJECT_ID-mlops/model",
    serving_container_image_uri=(
        "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
    ),
)
print(model.resource_name)
```
*Vérifier la version exacte du conteneur dans la doc : les URIs évoluent.*

## TP 2.6 — Déployer sur un Endpoint et tester
```python
endpoint = model.deploy(
    machine_type="n1-standard-2",
    min_replica_count=1, max_replica_count=1,
)
# une instance = une ligne de features (mêmes colonnes que l'entraînement)
print(endpoint.predict(instances=[[...]]))
```
**Exercice :** envoyer 2–3 profils (un « bon » et un « mauvais » payeur) et interpréter la prédiction.

## TP 2.7 — (Option) Prédiction par lot (*batch*)
Préparer un fichier d'entrées dans GCS, lancer un `BatchPredictionJob`, récupérer les sorties.
*Illustre la différence endpoint temps réel vs batch (cf. cours).*

## TP 2.8 — Monitoring de base
Activer le **Model Monitoring** sur l'endpoint (détection de dérive / *skew* entre données d'entraînement et de production), définir un seuil et une fréquence d'échantillonnage.
*Relier à la notion de data drift / concept drift du Jour 1.*

## TP 2.9 — (Avancé) Amorcer un CI/CD
Créer un `cloudbuild.yaml` minimal qui : (1) lance les tests + pre-commit, (2) construit/téléverse l'artefact, (3) ré-enregistre le modèle. Connecter un *trigger* Cloud Build au dépôt Git.
*C'est le passage du niveau 1 (pipeline d'entraînement) vers le niveau 2 (CI/CD) — boucle Ops de l'illustration GCP.*

## TP 2.10 — ⚠️ Nettoyage (obligatoire)
```python
endpoint.undeploy_all()
endpoint.delete()
model.delete()
```
```bash
gcloud storage rm -r ${BUCKET}     # si le bucket n'est plus utile
```
*Vérifier dans la console qu'aucun endpoint ne reste actif (coûts).*

---

## Variante sans cloud (plan B si comptes GCP indisponibles)
Reproduire la logique d'industrialisation sans GCP :
- **DagsHub** comme **serveur MLflow distant** + remote DVC → même expérience « cloud » (tracking + données versionnées partagés).
- Déploiement local du modèle via **FastAPI** + **Docker** (image conteneurisée), test par `curl`.
- CI avec **GitHub Actions** au lieu de Cloud Build.
Les concepts (registry, serving, conteneurisation, CI/CD, monitoring) restent identiques ; seul le fournisseur change.

---

## Grille d'évaluation suggérée (sur les 2 TP)
| Critère | Indicateur observable |
|---|---|
| Projet reproductible | uv + `pyproject.toml`/`uv.lock`, `.gitignore` correct |
| Qualité de code | pre-commit passe sur tout le dépôt |
| Versionnage données | données suivies par DVC (et poussées sur remote) |
| Traçabilité | ≥ 3 runs MLflow comparés, modèle enregistré + alias |
| Déploiement | endpoint (ou conteneur) fonctionnel et testé |
| Industrialisation | monitoring activé / amorce CI/CD |
| Hygiène cloud | ressources nettoyées en fin de séance |