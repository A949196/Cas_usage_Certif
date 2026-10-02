"""Reconstruction déterministe du split train/test original.

Ce module reconstruit `X_train`/`y_train` à l'identique à partir du
jeu brut complet (`data/dataset_trajectoire_emploi_Sujet.csv`).

`verifier_coherence_reference_set` est le garde-fou anti-dérive : si cette
fonction échoue, soit le jeu brut a changé, soit le split a dérivé.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2
COLONNES_EXCLUES_DU_SPLIT = ("usager_id",)
CIBLE = "classe_retour_emploi"


def charger_dataset_brut(dataset_path: Path) -> pd.DataFrame:
    """Charge le CSV brut complet (train+test combinés, 2500 lignes)."""
    return pd.read_csv(dataset_path, dtype={"code_insee_commune": "string"})


def reconstruire_split(
    dataset_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reconstruit `(X_train, X_test, y_train, y_test)` à l'identique du notebook.

    Colonnes brutes - à appliquer ensuite via `trajectoire_emploi.features`, comme le fait l'API.
    """
    df = charger_dataset_brut(dataset_path)
    y = df[CIBLE]
    X = df.drop(columns=[*COLONNES_EXCLUES_DU_SPLIT, CIBLE])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED,
    )
    return X_train, X_test, y_train, y_test


def verifier_coherence_reference_set(
    dataset_path: Path, reference_set_path: Path
) -> None:
    """Lève `AssertionError` si le `X_test` reconstruit ne correspond pas
    exactement à `reference_set.csv` (garde-fou anti-dérive)."""
    _, X_test, _, y_test = reconstruire_split(dataset_path)

    reconstruit = X_test.copy()
    reconstruit[CIBLE] = y_test
    reconstruit = reconstruit.reset_index(drop=True)

    reference = pd.read_csv(reference_set_path, dtype={"code_insee_commune": "string"})
    reference = reference[reconstruit.columns].reset_index(drop=True)

    pd.testing.assert_frame_equal(reconstruit, reference, check_dtype=False)
