"""Publie (ou met à jour) le rapport d'évaluation en commentaire de merge request.

Étape B4 du TP. Bibliothèque standard uniquement : le script tourne dans n'importe
quelle image Python, sans installation.

Un commentaire déjà publié par ce script est retrouvé grâce à un marqueur HTML
invisible et mis à jour, pour ne pas ajouter un commentaire par pipeline.

Variables d'environnement :
  MR_COMMENT_TOKEN       jeton personnel, scope "api" (variable CI/CD masquée)
  CI_API_V4_URL, CI_PROJECT_ID, CI_MERGE_REQUEST_IID   fournies par GitLab CI

Usage :
    python scripts/post_mr_comment.py artifacts/report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

MARKER = "<!-- mlops-tp:evaluation-report -->"
PAGE_SIZE = 100
MAX_PAGES = 5
TIMEOUT_S = 20


def call_api(method: str, url: str, token: str, payload: dict | None = None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"PRIVATE-TOKEN": token, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8") or "null")


def find_existing_note(notes_url: str, token: str) -> int | None:
    """Retourne l'identifiant du commentaire portant le marqueur, s'il existe."""
    for page in range(1, MAX_PAGES + 1):
        notes = call_api(
            "GET",
            f"{notes_url}?per_page={PAGE_SIZE}&page={page}&order_by=updated_at&sort=desc",
            token,
        )
        for note in notes:
            if not note.get("system") and MARKER in note.get("body", ""):
                return note["id"]
        if len(notes) < PAGE_SIZE:
            break
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Commente la MR avec le rapport d'évaluation.")
    parser.add_argument("report", help="rapport Markdown, ex. artifacts/report.md")
    args = parser.parse_args(argv)

    required = ["MR_COMMENT_TOKEN", "CI_API_V4_URL", "CI_PROJECT_ID", "CI_MERGE_REQUEST_IID"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        print(f"Variable(s) absente(s) : {', '.join(missing)}")
        if "MR_COMMENT_TOKEN" in missing:
            print(
                "Créez un jeton personnel (scope api) puis ajoutez-le dans "
                "Settings > CI/CD > Variables sous le nom MR_COMMENT_TOKEN (masquée)."
            )
        if "CI_MERGE_REQUEST_IID" in missing:
            print("Ce script ne s'exécute que dans un pipeline de merge request.")
        return 1

    report_path = Path(args.report)
    if not report_path.is_file():
        print(f"Rapport introuvable : {report_path}")
        return 1

    token = os.environ["MR_COMMENT_TOKEN"]
    notes_url = (
        f"{os.environ['CI_API_V4_URL']}/projects/{os.environ['CI_PROJECT_ID']}"
        f"/merge_requests/{os.environ['CI_MERGE_REQUEST_IID']}/notes"
    )
    body = f"{MARKER}\n{report_path.read_text(encoding='utf-8')}"

    try:
        note_id = find_existing_note(notes_url, token)
        if note_id is None:
            call_api("POST", notes_url, token, {"body": body})
            print("Commentaire créé dans la merge request.")
        else:
            call_api("PUT", f"{notes_url}/{note_id}", token, {"body": body})
            print(f"Commentaire {note_id} mis à jour.")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        print(f"Erreur API GitLab : HTTP {exc.code} {exc.reason} : {detail}")
        if exc.code == 401:
            print("Jeton refusé : vérifiez MR_COMMENT_TOKEN (scope api, non expiré).")
        return 1
    except urllib.error.URLError as exc:
        print(f"API GitLab injoignable : {exc.reason}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
