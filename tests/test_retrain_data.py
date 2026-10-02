"""Tests de la reconstruction déterministe du split train/test original."""

from __future__ import annotations

from pathlib import Path

import pytest

from trajectoire_emploi.retrain_data import (
    CIBLE,
    reconstruire_split,
    verifier_coherence_reference_set,
)

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "dataset_trajectoire_emploi_Sujet.csv"
REFERENCE_SET_PATH = ROOT / "data" / "reference_set.csv"

pytestmark = pytest.mark.skipif(
    not DATASET_PATH.exists(),
    reason="dataset brut absent (fichier non versionné, nécessaire uniquement en local)",
)


def test_reconstruire_split_tailles_attendues() -> None:
    X_train, X_test, y_train, y_test = reconstruire_split(DATASET_PATH)
    assert len(X_train) == 2000
    assert len(X_test) == 500
    assert len(y_train) == 2000
    assert len(y_test) == 500


def test_reconstruire_split_est_deterministe() -> None:
    _, X_test_1, _, _ = reconstruire_split(DATASET_PATH)
    _, X_test_2, _, _ = reconstruire_split(DATASET_PATH)
    assert X_test_1.equals(X_test_2)


def test_reconstruire_split_stratifie() -> None:
    _, _, y_train, y_test = reconstruire_split(DATASET_PATH)
    proportions_train = y_train.value_counts(normalize=True).sort_index()
    proportions_test = y_test.value_counts(normalize=True).sort_index()
    for classe in proportions_train.index:
        assert abs(proportions_train[classe] - proportions_test[classe]) < 0.02


def test_verifier_coherence_reference_set_ne_leve_pas() -> None:
    # Garde-fou anti-dérive : doit passer silencieusement si reference_set.csv
    # n'a pas dérivé du split original.
    verifier_coherence_reference_set(DATASET_PATH, REFERENCE_SET_PATH)


def test_verifier_coherence_reference_set_detecte_une_derive(tmp_path: Path) -> None:
    import pandas as pd

    reference = pd.read_csv(REFERENCE_SET_PATH)
    reference_alteree = reference.copy()
    reference_alteree.loc[0, "age"] = 999  # altération volontaire
    chemin_altere = tmp_path / "reference_alteree.csv"
    reference_alteree.to_csv(chemin_altere, index=False)

    with pytest.raises(AssertionError):
        verifier_coherence_reference_set(DATASET_PATH, chemin_altere)


def test_reconstruire_split_cible_absente_de_X() -> None:
    X_train, X_test, _, _ = reconstruire_split(DATASET_PATH)
    assert CIBLE not in X_train.columns
    assert CIBLE not in X_test.columns
    assert "usager_id" not in X_train.columns
