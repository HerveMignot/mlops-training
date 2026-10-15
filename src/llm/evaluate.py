"""TP 2 — Évaluation : comparer deux versions de prompt avec des scorers.

On évalue la même application avec le prompt @production (v1) puis @staging (v2),
sur un petit jeu de données, avec :
  - des scorers "code" (déterministes, gratuits) : réponse courte, en français
  - des juges LLM (LLM-as-a-judge) : Correctness et Guidelines  (désactivés en MOCK)

Lancement :  python tp2_evaluation.py          (MOCK=1 : scorers code uniquement)
Résultat : UI -> expérience "LLMOps-TP" -> onglet "Evaluation runs" -> comparer les 2 runs.
"""
import os

import mlflow
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines

from llm.llm import JUDGE_MODEL, MOCK, setup_tracing
from llm.trace_prompts import PROMPT_NAME, answer  # réutilise l'app du TP1  -> envisager une factorisation

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("LLMOps-TP")
setup_tracing()

# ---- Jeu de données d'évaluation --------------------------------------------------
eval_dataset = [
    {
        "inputs": {"question": "Qu'est-ce que MLflow ?"},
        "expectations": {"expected_response": "Une plateforme open source pour le cycle de vie ML et GenAI."},
    },
    {
        "inputs": {"question": "À quoi sert le tracing dans une application LLM ?"},
        "expectations": {"expected_response": "À observer les étapes d'exécution : prompts, appels, latence, coûts."},
    },
    {
        "inputs": {"question": "Pourquoi versionner ses prompts ?"},
        "expectations": {"expected_response": "Pour la reproductibilité, la comparaison et le retour arrière."},
    },
]


# ---- Scorers "code" -----------------------------------------------------------------
@scorer
def est_court(outputs: str) -> bool:
    """Vrai si la réponse fait au plus 25 mots."""
    return len(outputs.split()) <= 25


@scorer
def est_en_francais(outputs: str) -> bool:
    """Heuristique volontairement simple : présence de mots-outils français."""
    mots = {"le", "la", "les", "une", "un", "des", "est", "pour", "et", "de", "du"}
    return len(mots & set(outputs.lower().replace(".", " ").split())) >= 2


scorers = [est_court, est_en_francais]
if not MOCK:
    scorers += [
        Correctness(model=JUDGE_MODEL),
        Guidelines(
            name="ton_professionnel",
            guidelines="La réponse doit être factuelle, sans formule de politesse ni familiarité.",
            model=JUDGE_MODEL,
        ),
    ]


# ---- Une évaluation par alias de prompt --------------------------------------------
for alias in ("production", "staging"):
    print(f"\n=== Évaluation du prompt @{alias} ===")

    def predict_fn(question: str, _alias=alias) -> str:
        return answer(question, alias=_alias)

    with mlflow.start_run(run_name=f"eval-{PROMPT_NAME}@{alias}"):
        mlflow.log_param("prompt_alias", alias)
        results = mlflow.genai.evaluate(
            data=eval_dataset,
            predict_fn=predict_fn,
            scorers=scorers,
        )
        print(results.metrics)

print("\nOuvrez l'UI et comparez les deux runs d'évaluation.")
