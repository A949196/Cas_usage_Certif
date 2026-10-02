"""Construction du préprocesseur scikit-learn par scénario.

- **S1 (complet)** : « intégralité des variables (tabulaires + texte
  vectorisé) ».
- **S2 (sans variables sensibles)** : S1 privé de `nationalite_hors_ue`
  (décision D2, seule variable sensible **directe** retirée).
- **S3 (texte seul)** : « exclusivement la synthèse écrite ».
- **S4 (tabulaire pur)** : « uniquement l'âge, les diplômes, l'ancienneté et
  la géographie » — **volontairement plus restreint que S1 sans texte** :
  ni `code_rome_vise`, ni `est_allocataire`, ni `nationalite_hors_ue`.

`anciennete_incoherente` suit partout où `anciennete_poste_ans` est présente : 
c'est une feature dérivée de la même variable, pas une variable indépendante du sujet.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.feature_extraction.text import TfidfVectorizer

ORDRE_DIPLOME = ["Sans diplôme", "Bac", "Bac+2", "Bac+5"]

SCENARIOS = ("S1", "S2", "S3", "S4")


def _pipeline_numerique(avec_scaler: bool) -> Pipeline:
    etapes: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if avec_scaler:
        etapes.append(("scaler", StandardScaler()))
    return Pipeline(etapes)


def _pipeline_ordinal() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(categories=[ORDRE_DIPLOME])),
        ]
    )


def _pipeline_categoriel() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def _pipeline_binaire() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])


def _remplir_texte_manquant(colonne: pd.Series) -> pd.Series:
    """Remplace les synthèses manquantes par une chaîne vide avant le TF-IDF."""
    return pd.Series(colonne).fillna("")


def _pipeline_texte() -> Pipeline:
    return Pipeline(
        [
            ("remplissage", FunctionTransformer(_remplir_texte_manquant)),
            ("tfidf", TfidfVectorizer()),
        ]
    )


def construire_preprocesseur(scenario: str, avec_scaler: bool = False) -> ColumnTransformer:
    """Construit le `ColumnTransformer` du scénario demandé.

    `avec_scaler` ne doit être `True` que pour un modèle à distance/gradient
    (régression logistique) — jamais pour les modèles à base d'arbres
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario inconnu : {scenario!r} (attendu : {SCENARIOS})")

    numerique = _pipeline_numerique(avec_scaler)
    ordinal = _pipeline_ordinal()
    categoriel = _pipeline_categoriel()
    binaire = _pipeline_binaire()
    texte = _pipeline_texte()

    if scenario == "S1":
        return ColumnTransformer(
            transformers=[
                ("num", numerique, ["age", "anciennete_poste_ans"]),
                ("ord", ordinal, ["niveau_diplome"]),
                ("cat", categoriel, ["code_rome_vise", "departement"]),
                ("bin", binaire, ["est_allocataire", "nationalite_hors_ue", "anciennete_incoherente"]),
                ("texte", texte, "synthese_entretien"),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    if scenario == "S2":
        return ColumnTransformer(
            transformers=[
                ("num", numerique, ["age", "anciennete_poste_ans"]),
                ("ord", ordinal, ["niveau_diplome"]),
                ("cat", categoriel, ["code_rome_vise", "departement"]),
                ("bin", binaire, ["est_allocataire", "anciennete_incoherente"]),
                ("texte", texte, "synthese_entretien"),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    if scenario == "S3":
        return ColumnTransformer(
            transformers=[("texte", texte, "synthese_entretien")],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    # S4 — restreint au mot près du sujet : âge, diplômes, ancienneté, géographie
    return ColumnTransformer(
        transformers=[
            ("num", numerique, ["age", "anciennete_poste_ans"]),
            ("ord", ordinal, ["niveau_diplome"]),
            ("cat", categoriel, ["departement"]),
            ("bin", binaire, ["anciennete_incoherente"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
