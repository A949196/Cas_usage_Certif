"""Métriques métier pour la cible ordinale à 3 classes.
Les métriques « recall classe 2 », « taux d'erreur critique 2→0 » et
« coût moyen » reprennent les définitions posées dans le cadrage (§1.4 du
notebook).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score

# Matrice de coûts — hypothèse de travail posée en cadrage §1.4 (ligne = réel,
# colonne = prédit). Arbitraire dans ses valeurs : seule la structure compte
# (2→0 très supérieur au reste).
MATRICE_COUTS = np.array(
    [
        [0, 1, 2],
        [2, 0, 1],
        [10, 3, 0],
    ]
)


def recall_classe_2(y_true, y_pred) -> float:
    """Part des usagers réellement à risque (classe 2) correctement détectés."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    masque = y_true == 2
    if masque.sum() == 0:
        return float("nan")
    return float((y_pred[masque] == 2).mean())


def taux_erreur_critique(y_true, y_pred) -> float:
    """Part des usagers à risque (classe 2) classés à tort « retour rapide » (0).

    C'est l'erreur la plus grave selon le sujet : elle prive l'usager d'un
    accompagnement renforcé indispensable.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    masque = y_true == 2
    if masque.sum() == 0:
        return float("nan")
    return float((y_pred[masque] == 0).mean())


def kappa_pondere(y_true, y_pred) -> float:
    """Kappa pondéré quadratique : tient compte de l'ordre des classes."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def cout_moyen(y_true, y_pred, matrice: np.ndarray = MATRICE_COUTS) -> float:
    """Coût moyen par usager sous la matrice de coûts du cadrage."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    couts = matrice[y_true, y_pred]
    return float(couts.mean())


def evaluer(y_true, y_pred) -> dict[str, float]:
    """Synthétise toutes les métriques retenues pour un couple (y_true, y_pred)."""
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "recall_classe_2": recall_classe_2(y_true, y_pred),
        "kappa_pondere": kappa_pondere(y_true, y_pred),
        "taux_erreur_2_vers_0": taux_erreur_critique(y_true, y_pred),
        "cout_moyen": cout_moyen(y_true, y_pred),
    }


def rapport_classification(y_true, y_pred) -> str:
    """`classification_report` avec les libellés de classe du cadrage."""
    noms = ["Rapide (0)", "Moyen (1)", "Longue durée (2)"]
    return classification_report(y_true, y_pred, target_names=noms)
