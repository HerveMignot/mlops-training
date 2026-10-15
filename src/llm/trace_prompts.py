"""TP 1 — Observabilité (tracing) + Prompt Registry.

Objectifs :
  1. Voir une application GenAI "à l'intérieur" : arbre de spans, entrées/sorties, latence.
  2. Versionner un prompt comme un artefact (v1 -> v2) et le référencer par alias.

Lancement :  python tp1_trace_prompts.py          (MOCK=1 pour tester sans clé)
Puis ouvrir l'UI : http://127.0.0.1:5000  ->  expérience "LLMOps-TP"
"""
import mlflow

from llm.llm import call_llm, setup_tracing

PROMPT_NAME = "qa-assistant"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("LLMOps-TP")
setup_tracing()


# ---- Application tracée : chaque étape devient un span ---------------------------
@mlflow.trace(span_type="RETRIEVER", name="retrieve_context")
def retrieve(question: str) -> str:
    # Faux retriever : dans un vrai RAG, ici une recherche vectorielle.
    return "MLflow est un projet open source de la Linux Foundation."


@mlflow.trace(name="answer_question")
def answer(question: str, alias: str = "production") -> str:
    prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@{alias}")
    context = retrieve(question)
    messages = prompt.format(question=f"{question}\n\nContexte : {context}")
    return call_llm(messages[0]["content"], messages[1]["content"])


# ---- Registre de prompts : v1 puis v2 (les versions sont immuables) --------------
def register_prompts():
    v1 = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=[
            {"role": "system", "content": "Tu es un assistant utile. Réponds à la question."},
            {"role": "user", "content": "{{ question }}"},
        ],
        commit_message="v1 : prompt minimal",
    )
    v2 = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant technique. Réponds en français, en une seule phrase "
                    "de moins de 25 mots, sans introduction."
                ),
            },
            {"role": "user", "content": "{{ question }}"},
        ],
        commit_message="v2 : contraintes de forme (langue, longueur)",
    )
    print(f"Prompt '{PROMPT_NAME}' : versions {v1.version} et {v2.version}")

    # Alias = pointeur mobile (comme @production dans un model registry)
    mlflow.genai.set_prompt_alias(PROMPT_NAME, alias="production", version=v1.version)
    mlflow.genai.set_prompt_alias(PROMPT_NAME, alias="staging", version=v2.version)


if __name__ == "__main__":
    register_prompts()
    q = "Qu'est-ce que MLflow ?"
    for alias in ("production", "staging"):
        print(f"\n--- alias @{alias} ---")
        print(answer(q, alias=alias))
    print("\nOuvrez l'UI : onglet Traces (arbre de spans) et menu Prompts (diff v1/v2).")
