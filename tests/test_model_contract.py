"""Contract test du modèle packagé — filet en amont des tests d'API.

(`contract_test_model`). Si ce test échoue, ne pas chercher ailleurs : le
`.joblib` packagé (§8.2 du notebook) a dérivé ou n'a pas la bonne signature.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trajectoire_emploi.features import extraire_departement, nettoyer_anciennete_incoherente
from trajectoire_emploi.persistence import charger_modele

MODEL_PATH = Path(__file__).parent.parent / "models" / "trajectoire_emploi_v1.joblib"


@pytest.fixture(scope="module")
def modele_charge():
    if not MODEL_PATH.exists():
        pytest.skip(f"Modèle non packagé : {MODEL_PATH} absent (exécuter §8.2 du notebook)")
    return charger_modele(MODEL_PATH)


def _df_depuis_payload(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    df["departement"] = extraire_departement(df["code_insee_commune"].astype("string"))
    df["anciennete_poste_ans"], df["anciennete_incoherente"] = nettoyer_anciennete_incoherente(
        df["age"], df["anciennete_poste_ans"]
    )
    return df


def test_modele_contrat_shape_et_probabilites(modele_charge, valid_payload: dict) -> None:
    pipeline, metadata = modele_charge
    df = _df_depuis_payload(valid_payload)

    prediction = pipeline.predict(df)
    proba = pipeline.predict_proba(df)

    assert prediction.shape == (1,), f"shape predict={prediction.shape}, attendu (1,)"
    assert proba.shape == (1, 3), f"shape predict_proba={proba.shape}, attendu (1, 3)"
    assert (proba >= 0).all() and (proba <= 1).all(), "probabilités hors [0, 1]"
    assert abs(proba.sum() - 1.0) < 1e-6, "les probabilités ne somment pas à 1"
    assert set(prediction.tolist()) <= {0, 1, 2}, f"classe inattendue : {prediction}"


def test_metadata_contient_les_cles_obligatoires(modele_charge) -> None:
    _, metadata = modele_charge
    for cle in ("model_version", "created_at", "sklearn_version", "dataset_sha256", "metrics_holdout"):
        assert cle in metadata
        assert metadata[cle], f"clé {cle} vide"
