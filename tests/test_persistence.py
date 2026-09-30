"""Tests de `persistence.py` — packaging modèle + métadonnées"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trajectoire_emploi.persistence import (
    CLES_OBLIGATOIRES,
    charger_modele,
    construire_metadata,
    persister_modele,
)


@pytest.fixture
def dataset_factice(tmp_path):
    chemin = tmp_path / "dataset.csv"
    chemin.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    return chemin


@pytest.fixture
def pipeline_factice():
    import numpy as np

    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y = np.array([0, 1, 0, 1])
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit(X, y)
    return pipe


def test_construire_metadata_contient_les_cles_obligatoires(dataset_factice) -> None:
    metadata = construire_metadata(
        model_name="test_model",
        model_version="v1.0.0",
        dataset_path=dataset_factice,
        feature_columns=["x"],
        target_mapping={"classe_0": 0, "classe_1": 1},
        metrics_holdout={"f1_macro": 0.7},
    )

    for cle in CLES_OBLIGATOIRES:
        assert cle in metadata
        assert metadata[cle], f"clé {cle} vide"


def test_construire_metadata_leve_si_metrics_holdout_vide(dataset_factice) -> None:
    with pytest.raises(ValueError, match="manquantes"):
        construire_metadata(
            model_name="test_model",
            model_version="v1.0.0",
            dataset_path=dataset_factice,
            feature_columns=["x"],
            target_mapping={"classe_0": 0},
            metrics_holdout={},
        )


def test_persister_puis_charger_modele_roundtrip(
    tmp_path, dataset_factice, pipeline_factice
) -> None:
    metadata = construire_metadata(
        model_name="test_model",
        model_version="v1.0.0",
        dataset_path=dataset_factice,
        feature_columns=["x"],
        target_mapping={"classe_0": 0, "classe_1": 1},
        metrics_holdout={"f1_macro": 0.7},
    )

    model_path, meta_path = persister_modele(
        pipeline_factice, metadata, tmp_path, "test_model"
    )

    assert model_path.exists()
    assert meta_path.exists()

    pipeline_recharge, metadata_rechargee = charger_modele(model_path)

    assert metadata_rechargee["model_version"] == "v1.0.0"
    assert metadata_rechargee["metrics_holdout"] == {"f1_macro": 0.7}

    X_test = pd.DataFrame({"x": [1.0, 2.0]})
    # Le pipeline rechargé doit prédire à l'identique du pipeline d'origine.
    assert list(pipeline_recharge.predict(X_test)) == list(
        pipeline_factice.predict(X_test)
    )


def test_persister_modele_ecrit_un_json_valide(tmp_path, dataset_factice, pipeline_factice) -> None:
    metadata = construire_metadata(
        model_name="test_model",
        model_version="v1.0.0",
        dataset_path=dataset_factice,
        feature_columns=["x"],
        target_mapping={"classe_0": 0},
        metrics_holdout={"f1_macro": 0.7},
    )
    _, meta_path = persister_modele(pipeline_factice, metadata, tmp_path, "test_model")

    contenu = json.loads(meta_path.read_text(encoding="utf-8"))
    assert contenu["model_name"] == "test_model"
