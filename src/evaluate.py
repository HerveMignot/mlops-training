"""Quality gate du modèle candidat : étape B3 du TP.

Deux familles de gates, toutes deux pilotées par params.yaml / baseline.json :
  1. gate ABSOLU : chaque métrique suivie doit dépasser son seuil minimal ;
  2. gate RELATIF : aucune métrique suivie ne doit baisser de plus de
     ``max_regression`` par rapport à baseline.json (le modèle de référence).

Les gates portent sur f1, recall et roc_auc, pas sur l'accuracy : avec ~34 % de
départs, un modèle qui prédirait "reste" pour tout le monde aurait déjà ~66 %
d'accuracy, sans aucune valeur.

Le rapport Markdown est TOUJOURS écrit, même en cas d'échec : c'est lui que le job
mr-report publie dans la merge request.

Codes retour : 0 gates passés ; 1 au moins un gate en échec ; 2 entrées inutilisables.

baseline.json a le même format que metrics.json. Pour promouvoir un nouveau modèle de
référence, c'est un commit volontaire : cp artifacts/metrics.json baseline.json

Usage :
    python src/evaluate.py --metrics artifacts/metrics.json --params params.yaml \
        --baseline baseline.json --report artifacts/report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

GATED_METRICS = ("f1", "recall", "roc_auc")
INFO_METRICS = ("accuracy", "precision")


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def fmt_delta(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def check_gates(
    metrics: dict, gates: dict, baseline: dict | None
) -> tuple[dict[str, list[str]], list[str]]:
    """Retourne (échecs par métrique, liste plate des messages d'échec)."""
    failures: dict[str, list[str]] = {m: [] for m in GATED_METRICS}
    tolerance = gates["max_regression"]
    for name in GATED_METRICS:
        value = metrics[name]
        minimum = gates[f"min_{name}"]
        if value < minimum:
            failures[name].append(f"`{name}` = {value:.4f} est sous le seuil minimal {minimum:.4f}")
        if baseline is not None:
            reference = baseline[name]
            if value < reference - tolerance:
                failures[name].append(
                    f"`{name}` = {value:.4f} régresse de {reference - value:.4f} par rapport à "
                    f"la baseline ({reference:.4f}), tolérance {tolerance:.4f}"
                )
    flat = [message for messages in failures.values() for message in messages]
    return failures, flat


def render_report(
    metrics: dict,
    meta: dict,
    gates: dict,
    baseline: dict | None,
    failures: dict[str, list[str]],
    flat_failures: list[str],
) -> str:
    lines = [
        "## Évaluation du modèle candidat",
        "",
        f"Commit `{meta.get('commit', '?')}` · modèle `{meta.get('model_type', '?')}` · "
        f"{meta.get('n_test', '?')} lignes de test · {meta.get('positive_rate', 0):.1%} de départs",
        "",
        "| Métrique | Candidat | Baseline | Écart | Seuil minimal | Statut |",
        "|---|---|---|---|---|---|",
    ]
    for name in (*GATED_METRICS, *INFO_METRICS):
        value = metrics[name]
        reference = baseline[name] if baseline is not None else None
        delta = None if reference is None else value - reference
        if name in GATED_METRICS:
            minimum = fmt(gates[f"min_{name}"])
            status = "❌ échec" if failures[name] else "✅ ok"
        else:
            minimum, status = "n/a", "info"
        cells = [f"`{name}`", fmt(value), fmt(reference), fmt_delta(delta), minimum, status]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    if baseline is None:
        lines += ["> Aucune baseline trouvée : le gate de non-régression est ignoré.", ""]

    if flat_failures:
        lines.append(f"**Verdict : ❌ {len(flat_failures)} gate(s) en échec**")
        lines.append("")
        lines += [f"- {message}" for message in flat_failures]
    else:
        lines.append("**Verdict : ✅ tous les gates sont passés**")
    return "\n".join(lines) + "\n"


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Applique les quality gates au modèle candidat.")
    parser.add_argument("--metrics", required=True, help="metrics.json du candidat")
    parser.add_argument("--params", required=True, help="params.yaml (section gates)")
    parser.add_argument("--baseline", default="baseline.json", help="metrics.json de référence")
    parser.add_argument("--report", required=True, help="rapport Markdown à écrire")
    args = parser.parse_args(argv)
    report_path = Path(args.report)

    try:
        candidate = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
        gates = yaml.safe_load(Path(args.params).read_text(encoding="utf-8"))["gates"]
        metrics = candidate["metrics"]
        meta = candidate.get("meta", {})
        for name in (*GATED_METRICS, *INFO_METRICS):
            metrics[name]
        for key in ("max_regression", *(f"min_{m}" for m in GATED_METRICS)):
            gates[key]
        baseline_path = Path(args.baseline)
        baseline = (
            json.loads(baseline_path.read_text(encoding="utf-8"))["metrics"]
            if baseline_path.is_file()
            else None
        )
    except (OSError, KeyError, ValueError, yaml.YAMLError) as exc:
        write_report(
            report_path,
            f"## Évaluation impossible\n\n❌ Entrée manquante ou invalide : "
            f"`{type(exc).__name__}: {exc}`\n",
        )
        return 2

    failures, flat_failures = check_gates(metrics, gates, baseline)
    report = render_report(metrics, meta, gates, baseline, failures, flat_failures)
    if "CI_PIPELINE_URL" in os.environ:
        report += f"\n[Pipeline]({os.environ['CI_PIPELINE_URL']})\n"
    write_report(report_path, report)
    return 1 if flat_failures else 0


if __name__ == "__main__":
    sys.exit(main())
