"""Tests du store SQLite predictions/feedbacks (`feedback_store.py`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from trajectoire_emploi.feedback_store import (
    FeedbackConflictError,
    compter_feedbacks,
    enregistrer_feedback,
    enregistrer_prediction,
    initialiser_base,
    request_id_connu,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "feedback.db"
    initialiser_base(path)
    return path


def _enregistrer_prediction_test(db_path: Path, request_id: str = "REQ-1") -> None:
    enregistrer_prediction(
        db_path,
        request_id=request_id,
        classe_predite=2,
        decision="risque_longue_duree",
        probabilites={0: 0.1, 1: 0.2, 2: 0.7},
    )


def test_initialiser_base_est_idempotente(db_path: Path) -> None:
    initialiser_base(db_path)  # ne doit pas lever
    initialiser_base(db_path)


def test_request_id_inconnu(db_path: Path) -> None:
    assert request_id_connu(db_path, "INCONNU") is False


def test_request_id_connu_apres_enregistrement(db_path: Path) -> None:
    _enregistrer_prediction_test(db_path)
    assert request_id_connu(db_path, "REQ-1") is True


def test_enregistrer_feedback_nouveau(db_path: Path) -> None:
    _enregistrer_prediction_test(db_path)
    statut = enregistrer_feedback(
        db_path, request_id="REQ-1", true_label=2, comments="ok"
    )
    assert statut == "cree"
    assert compter_feedbacks(db_path) == {"total": 1, "new": 1}


def test_enregistrer_feedback_idempotent(db_path: Path) -> None:
    _enregistrer_prediction_test(db_path)
    enregistrer_feedback(db_path, request_id="REQ-1", true_label=2, comments=None)
    statut = enregistrer_feedback(
        db_path, request_id="REQ-1", true_label=2, comments="rejoue"
    )
    assert statut == "idempotent"
    assert compter_feedbacks(db_path) == {"total": 1, "new": 1}


def test_enregistrer_feedback_contradiction_leve_conflit(db_path: Path) -> None:
    _enregistrer_prediction_test(db_path)
    enregistrer_feedback(db_path, request_id="REQ-1", true_label=2, comments=None)
    with pytest.raises(FeedbackConflictError):
        enregistrer_feedback(db_path, request_id="REQ-1", true_label=0, comments=None)
    # La première vérité terrain n'a pas été écrasée.
    assert compter_feedbacks(db_path) == {"total": 1, "new": 1}


def test_compter_feedbacks_vide(db_path: Path) -> None:
    assert compter_feedbacks(db_path) == {"total": 0, "new": 0}


def test_compter_feedbacks_plusieurs_request_id(db_path: Path) -> None:
    for i in range(3):
        _enregistrer_prediction_test(db_path, request_id=f"REQ-{i}")
        enregistrer_feedback(
            db_path, request_id=f"REQ-{i}", true_label=i % 3, comments=None
        )
    assert compter_feedbacks(db_path) == {"total": 3, "new": 3}


def test_enregistrer_prediction_abstention_classe_predite_none(db_path: Path) -> None:
    enregistrer_prediction(
        db_path,
        request_id="REQ-ABSTENTION",
        classe_predite=None,
        decision="revue_humaine",
        probabilites={0: 0.3, 1: 0.35, 2: 0.35},
    )
    assert request_id_connu(db_path, "REQ-ABSTENTION") is True
