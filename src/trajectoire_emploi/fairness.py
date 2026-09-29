"""Calcul du disparate impact (règle des 4/5).

Réutilisé aux Étapes 2 (premier diagnostic), 3 (bilan éthique n°2) et 6
(audit d'équité de la décision finale) du plan — c'est ce qui justifie de
sortir cette fonction du notebook plutôt que de la dupliquer trois fois.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

SEUIL_DI_ALERTE = 0.8
SEUIL_DI_INVERSE = 1.25
SUPPORT_MIN_FIABLE = 30


def taux_de_selection(
    df: pd.DataFrame,
    groupe: str,
    cible: str,
    est_positif: Callable[[pd.Series], pd.Series],
) -> pd.Series:
    """Taux de sélection (P(outcome positif | groupe)) par modalité de `groupe`.

    `est_positif` reçoit la colonne `cible` et renvoie un masque booléen
    (ex. `lambda s: s == 0` pour « retour rapide »).
    """
    return df.groupby(groupe)[cible].apply(lambda s: est_positif(s).mean())


def disparate_impact(
    df: pd.DataFrame,
    groupe: str,
    cible: str,
    est_positif: Callable[[pd.Series], pd.Series],
) -> tuple[float, pd.Series]:
    """Calcule le DI (règle des 4/5) entre les modalités de `groupe`.

    Retourne `(di, taux_de_selection_par_groupe)`. Ne conclut pas seul à une
    discrimination : DI < 0.8 est un *signal* à documenter, pas une preuve
    (cf. fiche 202, avertissement final).
    """
    sr = taux_de_selection(df, groupe, cible, est_positif)
    di = sr.min() / sr.max()
    return float(di), sr


def verdict_4_5(di: float) -> str:
    """Traduit un DI en verdict lisible selon la règle des 4/5."""
    if di < SEUIL_DI_ALERTE:
        return "⚠️ signal (DI < 0.8) — groupe minoritaire désavantagé"
    if di > SEUIL_DI_INVERSE:
        return "⚠️ signal inversé (DI > 1.25) — groupe majoritaire désavantagé"
    return "✅ pas de signal au sens de la règle 4/5"


def supports_fiables(effectifs: pd.Series, seuil: int = SUPPORT_MIN_FIABLE) -> pd.Series:
    """Modalités dont l'effectif est en dessous du seuil de fiabilité statistique."""
    return effectifs[effectifs < seuil]
