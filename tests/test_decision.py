"""Tests de `decision.py` — données synthétiques uniquement."""

from __future__ import annotations

import numpy as np

from trajectoire_emploi.decision import (
    cout_attendu_minimal,
    decision_avec_abstention,
    decision_cout_minimal,
)
from trajectoire_emploi.evaluation import MATRICE_COUTS


def test_decision_cout_minimal_avec_matrice_uniforme_egale_argmax() -> None:
    # Matrice uniforme (toutes les erreurs valent 1, diagonale 0) : la
    # décision à coût minimal doit redevenir l'argmax classique.
    matrice_uniforme = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
    )
    probas = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
        ]
    )

    resultat = decision_cout_minimal(probas, matrice_uniforme)

    np.testing.assert_array_equal(resultat, probas.argmax(axis=1))


def test_decision_cout_minimal_bascule_vers_classe_2_si_ambigu() -> None:
    # Cas ambigu : P(0)=0.45, P(2)=0.40 — l'argmax choisirait 0, mais le
    # coût d'un vrai 2→0 (10) est tellement supérieur au coût d'un vrai
    # 0→2 (2) que la décision à coût minimal doit basculer vers 2.
    probas = np.array([[0.45, 0.15, 0.40]])

    resultat = decision_cout_minimal(probas, MATRICE_COUTS)

    assert probas.argmax(axis=1)[0] == 0  # confirme que l'argmax dirait 0
    assert resultat[0] == 2  # la décision à coût minimal corrige


def test_decision_cout_minimal_cas_certain() -> None:
    probas = np.array([[0.99, 0.005, 0.005]])

    resultat = decision_cout_minimal(probas, MATRICE_COUTS)

    assert resultat[0] == 0


def test_cout_attendu_minimal_bas_si_cas_certain() -> None:
    probas = np.array([[0.99, 0.005, 0.005]])

    valeur = cout_attendu_minimal(probas, MATRICE_COUTS)

    assert valeur[0] < 0.1


def test_cout_attendu_minimal_eleve_si_cas_ambigu() -> None:
    probas = np.array([[0.45, 0.15, 0.40]])

    valeur = cout_attendu_minimal(probas, MATRICE_COUTS)

    assert valeur[0] > 1.0


def test_decision_avec_abstention_garde_les_cas_certains() -> None:
    probas = np.array([[0.99, 0.005, 0.005], [0.02, 0.03, 0.95]])

    resultat = decision_avec_abstention(probas, seuil_abstention=0.5)

    assert list(resultat) == [0, 2]  # aucune abstention, cas certains


def test_decision_avec_abstention_route_les_cas_ambigus() -> None:
    probas = np.array([[0.45, 0.15, 0.40], [0.99, 0.005, 0.005]])

    resultat = decision_avec_abstention(probas, seuil_abstention=0.5)

    assert resultat[0] == "revue_humaine"  # cas ambigu, coût attendu > seuil
    assert resultat[1] == 0  # cas certain, gardé


def test_decision_avec_abstention_seuil_large_ne_rejette_rien() -> None:
    probas = np.array([[0.45, 0.15, 0.40]])

    resultat = decision_avec_abstention(probas, seuil_abstention=5.0)

    assert resultat[0] != "revue_humaine"
