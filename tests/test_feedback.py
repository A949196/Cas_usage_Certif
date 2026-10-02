"""Tests de la boucle de feedback (`POST /feedback`, `GET /feedback/count`).

Couvre 4 cas : 
- nouveau (201) idempotent (201 sans doublon), 
- `request_id` inconnu (404), 
- label hors bornes (422),
- contradiction (409).
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def _scorer_un_dossier(client: TestClient, valid_payload: dict) -> str:
    """Appelle /predict et renvoie le request_id obtenu (prérequis au feedback)."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    return response.json()["request_id"]


def test_predict_persiste_la_prediction_pour_le_feedback(
    client: TestClient, valid_payload: dict
) -> None:
    request_id = _scorer_un_dossier(client, valid_payload)
    # Un feedback valide sur ce request_id ne doit pas être rejeté en 404.
    response = client.post(
        "/feedback", json={"request_id": request_id, "true_label": 1}
    )
    assert response.status_code == 201


def test_feedback_request_id_inconnu_retourne_404(client: TestClient) -> None:
    response = client.post(
        "/feedback", json={"request_id": "INCONNU-123", "true_label": 1}
    )
    assert response.status_code == 404


def test_feedback_label_hors_bornes_retourne_422(
    client: TestClient, valid_payload: dict
) -> None:
    request_id = _scorer_un_dossier(client, valid_payload)
    response = client.post(
        "/feedback", json={"request_id": request_id, "true_label": 5}
    )
    assert response.status_code == 422


def test_feedback_meme_label_est_idempotent(
    client: TestClient, valid_payload: dict
) -> None:
    request_id = _scorer_un_dossier(client, valid_payload)
    r1 = client.post("/feedback", json={"request_id": request_id, "true_label": 0})
    r2 = client.post("/feedback", json={"request_id": request_id, "true_label": 0})
    assert r1.status_code == 201
    assert r1.json()["status"] == "cree"
    assert r2.status_code == 201
    assert r2.json()["status"] == "idempotent"


def test_feedback_label_different_retourne_409(
    client: TestClient, valid_payload: dict
) -> None:
    request_id = _scorer_un_dossier(client, valid_payload)
    r1 = client.post("/feedback", json={"request_id": request_id, "true_label": 0})
    r2 = client.post("/feedback", json={"request_id": request_id, "true_label": 2})
    assert r1.status_code == 201
    assert r2.status_code == 409


def test_feedback_count_reflete_les_insertions(
    client: TestClient, valid_payload: dict
) -> None:
    avant = client.get("/feedback/count").json()
    request_id = _scorer_un_dossier(client, valid_payload)
    client.post("/feedback", json={"request_id": request_id, "true_label": 1})
    apres = client.get("/feedback/count").json()
    assert apres["total"] == avant["total"] + 1
    assert apres["new"] == avant["new"] + 1


def test_feedback_comments_optionnel(client: TestClient, valid_payload: dict) -> None:
    request_id = _scorer_un_dossier(client, valid_payload)
    response = client.post(
        "/feedback", json={"request_id": request_id, "true_label": 1}
    )
    assert response.status_code == 201
