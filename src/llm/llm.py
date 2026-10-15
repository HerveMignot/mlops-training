"""Petit client LLM commun au TP.

Trois modes, choisis par variables d'environnement :
  - MOCK=1                  -> aucun appel réseau (réponses simulées), pour tester sans clé
  - PROVIDER=openai         -> OpenAI ou tout serveur compatible (OPENAI_BASE_URL possible)
  - PROVIDER=anthropic      -> API Anthropic

Le modèle se règle avec MODEL (défaut adapté au provider).
Le tracing MLflow est activé par autolog : chaque appel devient une trace.
"""
import os

import mlflow

MOCK = os.getenv("MOCK", "1") == "1"
PROVIDER = os.getenv("PROVIDER", "openai")
MODEL = os.getenv("MODEL") or ("gpt-4o-mini" if PROVIDER == "openai" else "claude-haiku-5-5")

# URI de modèle utilisée par les juges LLM de MLflow (Correctness, Guidelines, ...)
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or f"{PROVIDER}:/{MODEL}"


def setup_tracing():
    """Active l'auto-tracing du SDK utilisé (aucun effet en mode MOCK)."""
    if MOCK:
        return
    if PROVIDER == "anthropic":
        mlflow.anthropic.autolog()
    else:
        mlflow.openai.autolog()


@mlflow.trace(span_type="LLM", name="llm_call")
def _mock_call(system: str, user: str) -> str:
    # Simule un modèle sensible à la consigne du prompt (pour que la démo ait du sens).
    if "une seule phrase" in system.lower() or "one sentence" in system.lower():
        return "MLflow est une plateforme open source de gestion du cycle de vie ML et GenAI."
    return (
        "MLflow est une plateforme open source qui couvre le suivi d'expériences, "
        "le registre de modèles et, depuis la version 3, le tracing, l'évaluation "
        "et la gestion de prompts pour les applications GenAI. Elle est vendor-neutral "
        "et compatible OpenTelemetry. Elle peut être auto-hébergée."
    )


def call_llm(system: str, user: str) -> str:
    if MOCK:
        return _mock_call(system, user)
    if PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        r = client.messages.create(
            model=MODEL,
            max_tokens=500,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return r.content[0].text

    # Cas général : utilisation d'un serveur compatible OpenAI
    from openai import OpenAI

    client = OpenAI()  # lit OPENAI_API_KEY (et OPENAI_BASE_URL si défini)
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return r.choices[0].message.content
