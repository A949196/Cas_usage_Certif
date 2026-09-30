"""Tests de `evaluation.py` — données synthétiques uniquement."""

from __future__ import annotations

import numpy as np

from trajectoire_emploi.evaluation import (
    cout_moyen,
    evaluer,
    kappa_pondere,
    recall_classe_2,
    taux_erreur_critique,
)


def test_recall_classe_2_parfait() -> None:
    y_true = np.array([2, 2, 2, 0, 1])
    y_pred = np.array([2, 2, 2, 0, 1])

    assert recall_classe_2(y_true, y_pred) == 1.0


def test_recall_classe_2_rate_tout() -> None:
    y_true = np.array([2, 2, 0, 1])
    y_pred = np.array([0, 1, 0, 1])

    assert recall_classe_2(y_true, y_pred) == 0.0


def test_taux_erreur_critique_detecte_les_2_vers_0() -> None:
    # 2 classes 2 réelles, une prédite 0 (erreur critique), une prédite 1 (moins grave)
    y_true = np.array([2, 2, 0, 1])
    y_pred = np.array([0, 1, 0, 1])

    assert taux_erreur_critique(y_true, y_pred) == 0.5


def test_taux_erreur_critique_nan_si_pas_de_classe_2() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])

    resultat = taux_erreur_critique(y_true, y_pred)

    assert np.isnan(resultat)


def test_cout_moyen_zero_si_predictions_parfaites() -> None:
    y_true = np.array([0, 1, 2, 2, 0])
    y_pred = np.array([0, 1, 2, 2, 0])

    assert cout_moyen(y_true, y_pred) == 0.0


def test_cout_moyen_penalise_fortement_erreur_2_vers_0() -> None:
    # Une seule erreur 2→0 (coût 10) vs une seule erreur 0→1 (coût 1)
    cout_2_vers_0 = cout_moyen(np.array([2]), np.array([0]))
    cout_0_vers_1 = cout_moyen(np.array([0]), np.array([1]))

    assert cout_2_vers_0 == 10.0
    assert cout_0_vers_1 == 1.0
    assert cout_2_vers_0 > cout_0_vers_1


def test_kappa_pondere_parfait() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])

    assert kappa_pondere(y_true, y_pred) == 1.0


def test_kappa_pondere_penalise_plus_les_erreurs_eloignees() -> None:
    # Même nombre d'erreurs, mais l'une est adjacente (2→1) et l'autre distante (2→0)
    y_true = np.array([2, 2, 0, 1])
    kappa_adjacente = kappa_pondere(y_true, np.array([1, 2, 0, 1]))
    kappa_distante = kappa_pondere(y_true, np.array([0, 2, 0, 1]))

    assert kappa_distante < kappa_adjacente


def test_evaluer_retourne_toutes_les_cles() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 0])

    resultat = evaluer(y_true, y_pred)

    assert set(resultat) == {
        "f1_macro",
        "recall_classe_2",
        "kappa_pondere",
        "taux_erreur_2_vers_0",
        "cout_moyen",
    }
