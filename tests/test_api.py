"""Tests de l'API

Couverture minimale : routes publiques (`/health`, `/info`, `/predict`),
chemin d'erreur (422 sur entrée invalide)
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_info_expose_les_cles_obligatoires(client: TestClient) -> None:
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    for cle in (
        "api_version",
        "model_name",
        "model_version",
        "model_created_at",
        "sklearn_version",
        "dataset_sha256",
        "metrics_holdout",
        "seuil_abstention",
    ):
        assert cle in data
        assert data[cle] not in (None, "", {})


def test_predict_payload_valide(client: TestClient, valid_payload: dict) -> None:
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["decision"] in {
        "retour_rapide",
        "retour_moyen",
        "risque_longue_duree",
        "revue_humaine",
    }
    assert set(data["probabilites"].keys()) == {
        "retour_rapide",
        "retour_moyen",
        "risque_longue_duree",
    }
    assert abs(sum(data["probabilites"].values()) - 1.0) < 1e-6
    assert "request_id" in data
    assert "model_version" in data
    if data["decision"] == "revue_humaine":
        assert data["prediction"] is None
    else:
        assert data["prediction"] in (0, 1, 2)


def test_predict_champ_manquant_retourne_422(client: TestClient, valid_payload: dict) -> None:
    invalide = {k: v for k, v in valid_payload.items() if k != "code_rome_vise"}
    response = client.post("/predict", json=invalide)
    assert response.status_code == 422
    assert "code_rome_vise" in response.text


def test_predict_age_hors_bornes_retourne_422(client: TestClient, valid_payload: dict) -> None:
    invalide = {**valid_payload, "age": 150}
    response = client.post("/predict", json=invalide)
    assert response.status_code == 422


def test_predict_niveau_diplome_invalide_retourne_422(client: TestClient, valid_payload: dict) -> None:
    invalide = {**valid_payload, "niveau_diplome": "Doctorat"}
    response = client.post("/predict", json=invalide)
    assert response.status_code == 422


def test_predict_est_deterministe(client: TestClient, valid_payload: dict) -> None:
    r1 = client.post("/predict", json=valid_payload).json()
    r2 = client.post("/predict", json=valid_payload).json()
    assert r1["decision"] == r2["decision"]
    assert r1["prediction"] == r2["prediction"]
    for classe in r1["probabilites"]:
        assert abs(r1["probabilites"][classe] - r2["probabilites"][classe]) < 1e-9


def test_predict_request_id_dans_header(client: TestClient, valid_payload: dict) -> None:
    response = client.post("/predict", json=valid_payload)
    assert "x-request-id" in response.headers
