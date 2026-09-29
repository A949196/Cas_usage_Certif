"""Feature engineering déterministe sur les identifiants géographiques.

Règle appliquée (confirmée sur les données réelles en Étape 2/3) :
- Corse : codes `2A...`/`2B...` → département = 2 premiers caractères (`2A`, `2B`).
- DOM : codes commençant par `97` → département sur 3 chiffres (`971`...`976`).
- Reste de la France métropolitaine : département = 2 premiers chiffres.

Fonction déterministe (pas de paramètre appris) : appliquée avant le split,
conformément à la convention déjà en place dans le dépôt (cf. `AGENTS.md` §4).
"""

from __future__ import annotations

import pandas as pd


def extraire_departement(code_insee: pd.Series) -> pd.Series:
    """Extrait le département à partir d'un code INSEE commune (5 caractères).

    `code_insee` doit être de type chaîne (`string`/`str`), pas un entier
    (sinon les zéros initiaux et les codes Corse `2A`/`2B` sont perdus).
    """
    code = code_insee.astype("string")
    est_dom = code.str.startswith("97")
    departement = code.str.slice(0, 2)
    departement_dom = code.str.slice(0, 3)
    return departement.mask(est_dom, departement_dom)
