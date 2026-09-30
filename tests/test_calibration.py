"""Tests de `calibration.py` — données synthétiques uniquement."""

from __future__ import annotations

import numpy as np

from trajectoire_emploi.calibration import ece, reliability_diagram_data


def test_ece_parfaitement_calibre_proche_de_zero() -> None:
    rng = np.random.default_rng(42)
    n = 2000
    proba = rng.uniform(0, 1, size=n)
    # y tiré selon exactement la proba annoncée : modèle parfaitement calibré
    y = (rng.uniform(0, 1, size=n) < proba).astype(int)

    valeur = ece(proba, y, n_bins=10)

    assert valeur < 0.05  # tolérance liée au bruit d'échantillonnage


def test_ece_sur_confiance_donne_une_valeur_elevee() -> None:
    # Le modèle annonce toujours ~0.9 mais la vraie fréquence est ~0.3
    rng = np.random.default_rng(42)
    n = 1000
    proba = np.full(n, 0.9)
    y = (rng.uniform(0, 1, size=n) < 0.3).astype(int)

    valeur = ece(proba, y, n_bins=10)

    assert valeur > 0.4


def test_reliability_diagram_data_colonnes_attendues() -> None:
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])

    resultat = reliability_diagram_data(proba, y, n_bins=5)

    assert {"bin", "n", "confiance", "observe"} <= set(resultat.columns)
    assert resultat["n"].sum() == 4
