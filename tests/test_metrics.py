"""Tests de l'observabilité Prometheus.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_metrics_endpoint_repond_au_format_prometheus(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    # Métriques HTTP automatiques (instrumentator) présentes
    assert "http_requests_total" in response.text


def test_metrics_absent_du_schema_openapi(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/metrics" not in schema["paths"]


def test_predict_incremente_le_compteur_metier(client: TestClient, valid_payload: dict) -> None:
    avant = client.get("/metrics").text
    client.post("/predict", json=valid_payload)
    apres = client.get("/metrics").text

    assert "trajectoire_emploi_decisions_total" in apres
    # Au moins une des 4 décisions possibles doit être présente avec un compteur > 0
    assert avant != apres
