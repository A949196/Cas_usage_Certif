"""Feature engineering déterministe sur les identifiants géographiques.

Règle appliquée (confirmée sur les données réelles en Étape 2/3) :
- Corse : codes `2A...`/`2B...` → département = 2 premiers caractères (`2A`, `2B`).
- DOM : codes commençant par `97` → département sur 3 chiffres (`971`...`976`).
- Reste de la France métropolitaine : département = 2 premiers chiffres.
"""

from __future__ import annotations

import pandas as pd


def extraire_departement(code_insee: pd.Series) -> pd.Series:
    """Extrait le département à partir d'un code INSEE commune (5 caractères).

    `code_insee` est de type chaîne (`string`/`str`)
    """
    code = code_insee.astype("string")
    est_dom = code.str.startswith("97")
    departement = code.str.slice(0, 2)
    departement_dom = code.str.slice(0, 3)
    return departement.mask(est_dom, departement_dom)


def nettoyer_anciennete_incoherente(
    age: pd.Series, anciennete_poste_ans: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Neutralise les valeurs d'ancienneté incohérentes avec l'âge.

    Utilisée à la fois par le notebook (entraînement) et par l'API (inférence)
    pour garantir une parité stricte du feature engineering entre les deux
    contextes.

    Retourne `(anciennete_poste_ans_nettoyee, anciennete_incoherente)`
    """
    masque_incoherent = anciennete_poste_ans > (age - 15)
    anciennete_nettoyee = anciennete_poste_ans.mask(masque_incoherent, pd.NA)
    anciennete_incoherente = masque_incoherent.astype("Int64")
    return anciennete_nettoyee, anciennete_incoherente
