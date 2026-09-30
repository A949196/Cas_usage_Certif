"""Tests de `benchmark.py` — données synthétiques, CV réduite pour rapidité."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from trajectoire_emploi.benchmark import ConfigModele, executer_benchmark


def _jeu_synthetique(n: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "age": rng.integers(18, 63, size=n).astype(float),
            "anciennete_poste_ans": rng.uniform(0, 20, size=n),
            "niveau_diplome": rng.choice(["Sans diplôme", "Bac", "Bac+2", "Bac+5"], size=n),
            "code_rome_vise": rng.choice(["A1101", "K2101", "M1705"], size=n),
            "departement": rng.choice(["18", "07", "2A"], size=n),
            "est_allocataire": rng.integers(0, 2, size=n).astype(float),
            "nationalite_hors_ue": rng.integers(0, 2, size=n),
            "anciennete_incoherente": rng.integers(0, 2, size=n),
            "synthese_entretien": rng.choice(
                ["Profil autonome, projet clair.", "Freins périphériques majeurs."], size=n
            ),
        }
    )
    y = pd.Series(rng.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2]))
    return X, y


def test_executer_benchmark_produit_une_ligne_par_combinaison() -> None:
    X, y = _jeu_synthetique()
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    modeles = {
        "baseline": ConfigModele(DummyClassifier(strategy="stratified", random_state=42)),
        "logistique": ConfigModele(LogisticRegression(max_iter=200), avec_scaler=True),
    }

    resultats = executer_benchmark(X, y, ["S1", "S3"], modeles, cv)

    assert len(resultats) == 4  # 2 scénarios x 2 modèles
    assert {"S1", "S3"} == set(resultats["scenario"])
    assert {"baseline", "logistique"} == set(resultats["modele"])


def test_executer_benchmark_contient_les_colonnes_attendues() -> None:
    X, y = _jeu_synthetique()
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    modeles = {"baseline": ConfigModele(DummyClassifier(strategy="stratified", random_state=42))}

    resultats = executer_benchmark(X, y, ["S4"], modeles, cv)

    for metrique in ["f1_macro", "recall_classe_2", "kappa_pondere", "taux_erreur_2_vers_0", "cout_moyen"]:
        assert f"{metrique}_moyenne" in resultats.columns
        assert f"{metrique}_ecart_type" in resultats.columns
    assert "temps_entrainement_s" in resultats.columns
    assert "temps_inference_ms_pour_1k" in resultats.columns


def test_executer_benchmark_valeurs_dans_les_bornes_attendues() -> None:
    X, y = _jeu_synthetique()
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    modeles = {"baseline": ConfigModele(DummyClassifier(strategy="stratified", random_state=42))}

    resultats = executer_benchmark(X, y, ["S1"], modeles, cv)
    ligne = resultats.iloc[0]

    assert 0.0 <= ligne["f1_macro_moyenne"] <= 1.0
    assert 0.0 <= ligne["recall_classe_2_moyenne"] <= 1.0
    assert ligne["temps_entrainement_s"] >= 0.0
