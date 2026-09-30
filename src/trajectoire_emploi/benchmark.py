"""Boucle de benchmark scénarios × modèles, mêmes folds pour tous.
Règle d'or de comparabilité (même split, mêmes métriques, même prétraitement, mêmes
hyperparamètres par défaut sauf variante annoncée) et les 5 familles de
critères (précision, vitesse d'entraînement, vitesse d'inférence, mémoire,
explicabilité). 
Les métriques de précision viennent de `evaluation.py` (F1 macro, recall classe 2, kappa pondéré, 
taux d'erreur 2→0, coût moyen).

Réglage des hyperparamètres volontairement **limité** : pas de recherche
automatique (`GridSearchCV`), seulement quelques variantes manuelles
choisies à l'avance, toujours évaluées **dans la validation croisée**
(jamais sur le jeu de test), conformément à la règle anti-fuite.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from trajectoire_emploi.evaluation import cout_moyen, kappa_pondere, recall_classe_2, taux_erreur_critique
from trajectoire_emploi.pipeline import construire_preprocesseur


@dataclass(frozen=True)
class ConfigModele:
    """Un modèle à comparer, avec l'information « a-t-il besoin d'un scaler ? »."""

    modele: BaseEstimator
    avec_scaler: bool = False


SCORING = {
    "f1_macro": make_scorer(f1_score, average="macro"),
    "recall_classe_2": make_scorer(recall_classe_2),
    "kappa_pondere": make_scorer(kappa_pondere),
    # greater_is_better=True partout : on n'utilise pas ces scorers pour une
    # sélection automatique (pas de GridSearchCV), seulement pour peupler le
    # tableau. « Plus bas = mieux » pour ces deux-là, à lire ainsi dans le
    # tableau final (pas d'inversion de signe pour rester lisible).
    "taux_erreur_2_vers_0": make_scorer(taux_erreur_critique),
    "cout_moyen": make_scorer(cout_moyen),
}

N_ECHANTILLONS_INFERENCE = 1000


def _construire_pipeline_complet(scenario: str, config: ConfigModele) -> Pipeline:
    preprocesseur = construire_preprocesseur(scenario, avec_scaler=config.avec_scaler)
    return Pipeline([("preprocesseur", preprocesseur), ("modele", config.modele)])


def executer_benchmark(
    X: pd.DataFrame,
    y: pd.Series,
    scenarios: list[str],
    modeles: dict[str, ConfigModele],
    cv,
) -> pd.DataFrame:
    """Exécute le benchmark scénarios × modèles sur **les mêmes folds** `cv`.

    Pour chaque combinaison : métriques de précision en validation croisée
    (moyenne + écart-type sur les répétitions), puis un unique `fit`/`predict` sur tout `X`/`y` **seulement pour chronométrer**
    """
    lignes = []
    for nom_scenario in scenarios:
        for nom_modele, config in modeles.items():
            pipe = _construire_pipeline_complet(nom_scenario, config)

            # n_jobs=None (séquentiel) : évite le sur-abonnement de processus
            # quand un modèle (RandomForest) parallélise déjà en interne
            # (n_jobs=-1) — la parallélisation imbriquée avec joblib/loky
            # peut sinon ralentir fortement, voire quasi bloquer, l'exécution.
            resultats_cv = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=None)

            ligne: dict[str, float | str] = {"scenario": nom_scenario, "modele": nom_modele}
            for cle in SCORING:
                scores = resultats_cv[f"test_{cle}"]
                ligne[f"{cle}_moyenne"] = float(scores.mean())
                ligne[f"{cle}_ecart_type"] = float(scores.std())

            pipe_chrono = _construire_pipeline_complet(nom_scenario, config)
            t0 = time.perf_counter()
            pipe_chrono.fit(X, y)
            ligne["temps_entrainement_s"] = time.perf_counter() - t0

            n = min(N_ECHANTILLONS_INFERENCE, len(X))
            t0 = time.perf_counter()
            pipe_chrono.predict(X.iloc[:n])
            ligne["temps_inference_ms_pour_1k"] = (time.perf_counter() - t0) * 1000 * (1000 / n)

            lignes.append(ligne)

    return pd.DataFrame(lignes)
