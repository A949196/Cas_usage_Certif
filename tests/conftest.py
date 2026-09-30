"""Fixtures partagées pour les tests d'API"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """TestClient avec lifespan déclenché (modèle chargé)."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload() -> dict:
    """Payload valide, aligné sur `DemandeurInput` (feature_columns du modèle)."""
    return {
        "age": 35,
        "anciennete_poste_ans": 3.0,
        "niveau_diplome": "Bac+2",
        "code_rome_vise": "D1503",
        "code_insee_commune": "18273",
        "est_allocataire": True,
        "synthese_entretien": "Candidat motivé avec expérience en logistique.",
    }
