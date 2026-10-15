# Fiche de TP — Formation MLOps - LLMOps avec MLflow (≈ 45 min)

Testé avec MLflow 3.17.1 (Python 3.12). Fonctionne avec toute version 3.x récente.

## 0. Installation (5 min)

Reprendre le même environnement que pour la partie MLOps, et ajouter les packages nécessaires aux appels des modèles de langage (LLM) qui seront utilisés comme juges dans la partie Evaluation.

Si l'environnement n'a pas été créé, procéder comme suit.

### Sans UV et organisation simple

```bash
python -m venv .venv && source .venv/bin/activate
pip install mlflow openai anthropic     # openai OU anthropic selon votre clé
```

### Avec UV et organisation du repo comme pour la partie MLOps

Se reporter à la fiche de TP pour la partie MLOps, section 1.0, 1.1 et 1.2.

### Lancement de MLflow

Dans un terminal, lancer le serveur MLflow (laisser tourner) :

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```
*Illustration des options de MLflow, `mlflow server` suffit.*

Pour accéder à l'UI : http://127.0.0.1:5000

## 1. Configurer le LLM (2 min) — ou rester en mode sans clé

| Cas | Variables d'environnement |
|---|---|
| Sans clé (simulation) | `export MOCK=1` |
| OpenAI | `export PROVIDER=openai OPENAI_API_KEY=...` (`MODEL=gpt-4o-mini` par défaut) |
| Serveur compatible OpenAI (vLLM, Ollama, passerelle…) | idem + `OPENAI_BASE_URL=http://.../v1` et `MODEL=<nom>` |
| Anthropic | `export PROVIDER=anthropic ANTHROPIC_API_KEY=... MODEL=<id du modèle>` |

En mode `MOCK=1`, seuls les scorers "code" tournent ; les juges LLM sont désactivés.

## LLM TP1 — Tracing + Prompt Registry (15 min)

```bash
uv run -m llm.trace_prompts
```

À observer dans l'interface de MLflow :
1. **Traces** : ouvrir une trace `answer_question` → arbre de spans (retriever, appel LLM), entrées/sorties, latence, tokens (hors MOCK).
2. **Prompts → qa-assistant** : versions 1 et 2, bouton de comparaison (diff), alias `production` / `staging`.

Questions : que se passe-t-il si on déplace l'alias `production` sur la v2 ? Faut-il redéployer l'application ?

## LLM TP2 — Évaluation (20 min)

```bash
uv run -m llm.evaluate
```

À observer : **Evaluation runs** -> sélectionner les deux runs -> comparer. Le prompt v2 doit gagner sur `est_court`.
Cliquer sur une ligne : la trace de chaque prédiction et la justification des juges LLM sont liées au score.

Exercices :
- Ajouter une question au jeu de données, relancer, comparer.
- Écrire un scorer `contient_une_source` (code) ou une `Guidelines` (juge LLM) de votre choix.
- Créer une v3 du prompt qui casse volontairement une contrainte, et vérifier que l'évaluation la détecte (= test de non-régression).

## 4. Pour aller plus loin (démo formateur)

- **AI Gateway** : centraliser les clés et exposer un endpoint unique (routage, fallback, budgets, guardrails).
- **Optimisation de prompt** : `mlflow.genai.optimize_prompt` (algorithmes dont GEPA) à partir d'un jeu annoté.
- **Détection automatique de problèmes** sur les traces ; **MLflow Assistant**.
- Démo publique sans installation : https://demo.mlflow.org ou `uvx mlflow demo`.

## Dépannage

- `Connection refused` : le serveur du terminal 1 n'est pas lancé.
- Erreur de clé pendant l'évaluation : vérifier que `PROVIDER` correspond bien à la clé ; les juges utilisent `JUDGE_MODEL` (défaut `<provider>:/<model>`).
- Repartir de zéro : arrêter le serveur et supprimer `mlflow.db`.
