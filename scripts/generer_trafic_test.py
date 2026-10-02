"""Génère un trafic synthétique et l'envoie à l'API réellement déployée.

Ce n'est PAS du trafic de production organique — c'est un générateur
qui fabrique des payloads, envoyés via HTTP à l'API
`trajectoire_emploi` réellement démarrée (`docker compose up`). L'API et
le pipeline de scoring sont réels ; les données entrantes sont fabriquées.

Chaque payload envoyé est aussi écrit dans un CSV local, pour permettre
une comparaison a posteriori avec `data/reference_set.csv` via
`scripts/comparer_derive_production.py`.

Usage :
    docker compose up --build -d   # l'API doit tourner
    python scripts/generer_trafic_test.py [--n 80] [--sortie mon_trafic.csv]
"""

from __future__ import annotations

import argparse
import csv
import random
import sys

import httpx

ROME_CODES = ["D1503", "G1302", "M1705", "A1203", "N1301"]
DIPLOMES = ["Sans diplôme", "Bac", "Bac+2", "Bac+5"]
COMMUNES = ["18273", "07240", "01405", "75056", "97228"]


def generer_payload() -> dict:
    return {
        "age": random.randint(20, 60),
        "anciennete_poste_ans": round(random.uniform(0, 15), 1),
        "niveau_diplome": random.choice(DIPLOMES),
        "code_rome_vise": random.choice(ROME_CODES),
        "code_insee_commune": random.choice(COMMUNES),
        "est_allocataire": random.choice([True, False]),
        "synthese_entretien": "test trafic",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=80, help="Nombre de requêtes à envoyer")
    parser.add_argument("--sortie", default="mon_trafic_envoye.csv", help="CSV de trace")
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    champs = [
        "age", "anciennete_poste_ans", "niveau_diplome", "code_rome_vise",
        "code_insee_commune", "est_allocataire", "synthese_entretien",
    ]

    try:
        httpx.get(f"{args.api_url}/health", timeout=5).raise_for_status()
    except httpx.HTTPError as exc:
        print(f"ERREUR : API injoignable sur {args.api_url} ({exc}).", file=sys.stderr)
        print("Lancez d'abord : docker compose up --build -d", file=sys.stderr)
        return 1

    with open(args.sortie, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=champs)
        writer.writeheader()

        for _ in range(args.n):
            payload = generer_payload()
            httpx.post(f"{args.api_url}/predict", json=payload, timeout=10)
            writer.writerow(payload)

    print(f"{args.n} requêtes envoyées à {args.api_url}, trace dans {args.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
