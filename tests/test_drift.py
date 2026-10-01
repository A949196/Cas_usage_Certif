"""Tests de `drift.py` — PSI / KS / Chi² / diagnostic data vs concept drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trajectoire_emploi.drift import (
    diagnostic_data_vs_concept_drift,
    psi,
    calculer_chi2,
    calculer_ks,
    verdict_psi,
)


def test_psi_distributions_identiques_proche_de_zero() -> None:
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(0, 1, 2000))
    courant = pd.Series(rng.normal(0, 1, 2000))

    valeur = psi(reference, courant, n_bins=10)

    assert valeur < 0.10


def test_psi_distributions_decalees_signal_fort() -> None:
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(0, 1, 2000))
    courant = pd.Series(rng.normal(2.0, 1, 2000))  # décalage important

    valeur = psi(reference, courant, n_bins=10)

    assert valeur > 0.25


def test_psi_gere_les_bins_vides_sans_inf() -> None:
    reference = pd.Series([1.0] * 100 + [2.0] * 100)
    courant = pd.Series([1.0] * 200)  # un bin de la référence devient vide

    valeur = psi(reference, courant, n_bins=4)

    assert np.isfinite(valeur)


def test_verdict_psi_trois_paliers() -> None:
    assert verdict_psi(0.05) == "stable (signal faible)"
    assert verdict_psi(0.15) == "a investiguer"
    assert verdict_psi(0.30) == "derive forte"


def test_ks_distributions_identiques_p_value_elevee() -> None:
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(0, 1, 500))
    courant = pd.Series(rng.normal(0, 1, 500))

    _, p_value = calculer_ks(reference, courant)

    assert p_value > 0.05


def test_ks_distributions_differentes_p_value_faible() -> None:
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(0, 1, 500))
    courant = pd.Series(rng.normal(3.0, 1, 500))

    _, p_value = calculer_ks(reference, courant)

    assert p_value < 0.05


def test_chi2_memes_proportions_p_value_elevee() -> None:
    rng = np.random.default_rng(42)
    modalites = ["A", "B", "C"]
    reference = pd.Series(rng.choice(modalites, size=1000, p=[0.5, 0.3, 0.2]))
    courant = pd.Series(rng.choice(modalites, size=1000, p=[0.5, 0.3, 0.2]))

    _, p_value = calculer_chi2(reference, courant)

    assert p_value > 0.05


def test_chi2_proportions_redistribuees_p_value_faible() -> None:
    rng = np.random.default_rng(42)
    modalites = ["A", "B", "C"]
    reference = pd.Series(rng.choice(modalites, size=1000, p=[0.8, 0.1, 0.1]))
    courant = pd.Series(rng.choice(modalites, size=1000, p=[0.1, 0.1, 0.8]))

    _, p_value = calculer_chi2(reference, courant)

    assert p_value < 0.05


def test_chi2_aligne_les_modalites_absentes_dans_un_echantillon() -> None:
    reference = pd.Series(["A", "A", "B"])
    courant = pd.Series(["A", "C", "C"])  # "C" absent de reference, "B" absent de courant

    # Ne doit pas lever d'exception de dimension
    statistique, p_value = calculer_chi2(reference, courant)

    assert np.isfinite(statistique)
    assert 0.0 <= p_value <= 1.0


def test_diagnostic_auc_stable_compatible_data_drift() -> None:
    diagnostic = diagnostic_data_vs_concept_drift(auc_reference=0.75, auc_courant=0.74)
    assert "data drift" in diagnostic
    assert "stable" in diagnostic


def test_diagnostic_auc_degradee_signale_investigation() -> None:
    diagnostic = diagnostic_data_vs_concept_drift(auc_reference=0.75, auc_courant=0.60)
    assert "investiguer" in diagnostic
