"""Compare un lot de trafic (généré ou réel) à `data/reference_set.csv`.

Complète la démonstration du notebook (train vs test comme proxy) par
un test sur des données réellement envoyées à l'API déployée (via
`scripts/generer_trafic_test.py). 

Lecture des résultats : un signal fort ici peut venir d'une vraie
dérive, **ou** d'un générateur de trafic non représentatif.

Usage :
    python scripts/comparer_derive_production.py [--trafic mon_trafic_envoye.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectoire_emploi.drift import (  # noqa: E402
    calculer_chi2,
    calculer_ks,
    psi,
    verdict_psi,
)

REFERENCE_SET_PATH = ROOT / "data" / "reference_set.csv"

FEATURES_NUMERIQUES = ["age", "anciennete_poste_ans"]
FEATURES_CATEGORIELLES = ["niveau_diplome", "code_rome_vise"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trafic", default="mon_trafic_envoye.csv")
    args = parser.parse_args()

    trafic_path = Path(args.trafic)
    if not trafic_path.exists():
        print(f"ERREUR : {trafic_path} introuvable.", file=sys.stderr)
        print("Lancez d'abord : python scripts/generer_trafic_test.py", file=sys.stderr)
        return 1

    reference = pd.read_csv(REFERENCE_SET_PATH)
    trafic = pd.read_csv(trafic_path)

    print("=== Features numériques (PSI + KS) ===")
    for feature in FEATURES_NUMERIQUES:
        valeur_psi = psi(reference[feature], trafic[feature])
        _, p_ks = calculer_ks(reference[feature], trafic[feature])
        print(
            f"{feature}: PSI={valeur_psi:.3f} ({verdict_psi(valeur_psi)}), "
            f"p-value KS={p_ks:.4f}"
        )

    print()
    print("=== Features catégorielles (Chi²) ===")
    for feature in FEATURES_CATEGORIELLES:
        _, p_chi2 = calculer_chi2(reference[feature], trafic[feature])
        verdict = "derive detectable" if p_chi2 < 0.05 else "pas de signal"
        print(f"{feature}: p-value Chi2={p_chi2:.4f} ({verdict})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
