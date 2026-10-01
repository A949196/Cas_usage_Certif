"""Garde-fou d'évaluation continue.

⚠️ **Ce script ne mesure PAS la performance de généralisation du modèle.**
Le modèle de production (`models/trajectoire_emploi_v1.joblib`) a été réentraîné sur
train+test combinés (§8.1) — il a donc **déjà vu** `data/reference_set.csv`
(qui est l'ancien `X_test`/`y_test`) pendant son entraînement. Évaluer ce
modèle sur ce jeu donne des métriques **en partie en-échantillon**,
artificiellement hautes (F1 macro ≈ 0,94 au lieu de 0,72)

**Ce que ce script détecte réellement** : une **régression de code** —
feature engineering cassé, ordre des colonnes changé, dépendance qui
change silencieusement le comportement de `predict_proba`, seuil de
décision mal branché. Sur un modèle et un code inchangés, l'écart au
golden run doit être **exactement nul**, pas « petit ».

Usage :
    # Geler le golden run (une fois, après avoir vérifié le résultat) :
    python scripts/evaluate_model.py --freeze-baseline

    # Vérifier une release (ce que fait la CI) :
    python scripts/evaluate_model.py

    # Prouver que le garde-fou bloque réellement :
    python scripts/evaluate_model.py --degrade
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectoire_emploi.decision import decision_avec_abstention  # noqa: E402
from trajectoire_emploi.evaluation import evaluer  # noqa: E402
from trajectoire_emploi.features import (  # noqa: E402
    extraire_departement,
    nettoyer_anciennete_incoherente,
)
from trajectoire_emploi.persistence import charger_modele  # noqa: E402

REFERENCE_SET_PATH = ROOT / "data" / "reference_set.csv"
BASELINE_PATH = ROOT / "data" / "reference_baseline.json"
MODEL_PATH = ROOT / "models" / "trajectoire_emploi_v1.joblib"
TAILLE_MIN_REFERENCE = 100

# Seuils — justifiés par la variance mesurée.
#
# f1_macro : bootstrap (500 ré-échantillonnages sur reference_set.csv) 
# → sigma ≈ 0,012 → tolérance relative fixée à 0,05 (> 2 sigma = 0,025, marge de sécurité).
#
# taux_erreur_2_vers_0 : le bootstrap EN-ÉCHANTILLON est dégénéré ici
# (sigma = 0 exactement) — le modèle de production a déjà vu
# `reference_set.csv` à l'entraînement (§8.1). 
# On utilise donc la meilleure variance RÉELLE disponible :
# l'écart-type inter-folds mesuré à l'Étape 5 :
# sigma = 0,043 → tolérance = 2 x 0,043 = 0,086 (arrondi à 0,09).
# Écart assumé : cette variance vient d'une règle de décision légèrement
# différente (argmax, pas coût minimal + abstention).
THRESHOLDS: dict[str, dict[str, float]] = {
    "f1_macro": {"absolute_min": 0.55, "max_drop_vs_golden": 0.05},
    "taux_erreur_2_vers_0": {"absolute_max": 0.10, "max_increase_vs_golden": 0.09},
}



def construire_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applique le même feature engineering que l'API (`app/main.py`).
    """
    df = df.copy()
    df["departement"] = extraire_departement(df["code_insee_commune"].astype("string"))
    df["anciennete_poste_ans"], df["anciennete_incoherente"] = nettoyer_anciennete_incoherente(
        df["age"], df["anciennete_poste_ans"]
    )
    return df


def charger_reference_set(degrade: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    if not REFERENCE_SET_PATH.exists():
        print(f"ERREUR : {REFERENCE_SET_PATH} introuvable.", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(REFERENCE_SET_PATH, dtype={"code_insee_commune": "string"})
    if len(df) < TAILLE_MIN_REFERENCE:
        print(
            f"ERREUR : reference_set trop petit ({len(df)} < {TAILLE_MIN_REFERENCE}).",
            file=sys.stderr,
        )
        sys.exit(2)
    if df["classe_retour_emploi"].nunique() < 2:
        print("ERREUR : reference_set mono-classe.", file=sys.stderr)
        sys.exit(2)

    y = df["classe_retour_emploi"].to_numpy()
    X = df.drop(columns=["classe_retour_emploi"])

    if degrade:
        # Chemin rouge volontaire (fiche 517, exercice guidé point 2) :
        # désaligne X et y pour simuler une régression de pipeline.
        rng = np.random.default_rng(0)
        y = rng.permutation(y)
        print("[--degrade] y permuté volontairement pour tester le chemin rouge.")

    return X, y


def calculer_metriques(X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    pipeline, metadata = charger_modele(MODEL_PATH)
    X_prepare = construire_dataframe(X)

    probas = pipeline.predict_proba(X_prepare)
    seuil = metadata.get("seuil_abstention", 0.7)
    resultat = decision_avec_abstention(probas, seuil)
    masque_garde = resultat != "revue_humaine"

    metriques = evaluer(y[masque_garde], resultat[masque_garde].astype(int))
    metriques["taux_revue_humaine"] = float((~masque_garde).mean())
    return metriques


def verifier_seuils(
    metriques: dict[str, float], golden: dict[str, float]
) -> list[str]:
    violations = []
    for nom, regle in THRESHOLDS.items():
        valeur = metriques[nom]
        valeur_golden = golden.get(nom)

        if "absolute_min" in regle and valeur < regle["absolute_min"]:
            violations.append(f"{nom}={valeur:.4f} < minimum absolu {regle['absolute_min']}")
        if "absolute_max" in regle and valeur > regle["absolute_max"]:
            violations.append(f"{nom}={valeur:.4f} > maximum absolu {regle['absolute_max']}")

        if valeur_golden is not None:
            if "max_drop_vs_golden" in regle and (valeur_golden - valeur) > regle["max_drop_vs_golden"]:
                violations.append(
                    f"{nom} a chuté de {valeur_golden:.4f} à {valeur:.4f} "
                    f"(> {regle['max_drop_vs_golden']} vs golden run)"
                )
            if "max_increase_vs_golden" in regle and (valeur - valeur_golden) > regle["max_increase_vs_golden"]:
                violations.append(
                    f"{nom} a augmenté de {valeur_golden:.4f} à {valeur:.4f} "
                    f"(> {regle['max_increase_vs_golden']} vs golden run)"
                )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze-baseline",
        action="store_true",
        help="Geler le résultat actuel comme golden run (écrase reference_baseline.json)",
    )
    parser.add_argument(
        "--degrade",
        action="store_true",
        help="Désaligne volontairement X/y pour tester le chemin rouge (fiche 517)",
    )
    args = parser.parse_args()

    X, y = charger_reference_set(degrade=args.degrade)
    metriques = calculer_metriques(X, y)

    print("Métriques calculées sur reference_set.csv :")
    for cle, valeur in metriques.items():
        print(f"  {cle}: {valeur:.4f}")

    if args.freeze_baseline:
        BASELINE_PATH.write_text(
            json.dumps(metriques, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nGolden run gelé dans {BASELINE_PATH}")
        return 0

    if not BASELINE_PATH.exists():
        print(
            f"\nERREUR : {BASELINE_PATH} introuvable — lancez d'abord "
            "--freeze-baseline.",
            file=sys.stderr,
        )
        return 2

    golden = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    violations = verifier_seuils(metriques, golden)

    print()
    if violations:
        print("VIOLATIONS DE SEUIL DÉTECTÉES :")
        for v in violations:
            print(f"  - {v}")
        print("\nRelease BLOQUÉE.")
        return 1

    print("Aucune violation — release autorisée.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
