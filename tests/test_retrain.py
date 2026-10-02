"""Tests de `scripts/retrain.py`.

Mélange tests unitaires rapides et un test d'intégration réel via subprocess 
(environnement entièrement isolé, aucune interférence avec `data/runtime/feedback.db`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "dataset_trajectoire_emploi_Sujet.csv"
REFERENCE_SET_PATH = ROOT / "data" / "reference_set.csv"
PRODUCTION_MODEL_PATH = ROOT / "models" / "trajectoire_emploi_v1.joblib"

pytestmark = pytest.mark.skipif(
    not DATASET_PATH.exists(),
    reason="dataset brut absent (fichier non versionné, nécessaire uniquement en local)",
)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from retrain import construire_jeu_entrainement, verifier_contrat_candidat  # noqa: E402
from trajectoire_emploi.feedback_store import (  # noqa: E402
    enregistrer_feedback,
    enregistrer_prediction,
    initialiser_base,
)

FEATURES_TEST = {
    "age": 35,
    "anciennete_poste_ans": 3.0,
    "niveau_diplome": "Bac+2",
    "code_rome_vise": "D1503",
    "code_insee_commune": "18273",
    "est_allocataire": True,
    "synthese_entretien": "Candidat motivé.",
}


class _PipelineFactice:
    """Double de test : simule `predict_proba` sans entraînement réel."""

    def __init__(self, probas: np.ndarray) -> None:
        self._probas = probas

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._probas


def test_construire_jeu_entrainement_sans_feedback(tmp_path: Path) -> None:
    db_path = tmp_path / "feedback.db"
    initialiser_base(db_path)

    X, y, request_ids = construire_jeu_entrainement(DATASET_PATH, db_path)

    assert len(X) == 2000
    assert len(y) == 2000
    assert request_ids == []


def test_construire_jeu_entrainement_avec_feedback(tmp_path: Path) -> None:
    db_path = tmp_path / "feedback.db"
    initialiser_base(db_path)
    enregistrer_prediction(
        db_path,
        request_id="REQ-1",
        classe_predite=1,
        decision="retour_moyen",
        probabilites={0: 0.3, 1: 0.4, 2: 0.3},
        features=FEATURES_TEST,
    )
    enregistrer_feedback(db_path, request_id="REQ-1", true_label=1, comments=None)

    X, y, request_ids = construire_jeu_entrainement(DATASET_PATH, db_path)

    assert len(X) == 2001
    assert len(y) == 2001
    assert request_ids == ["REQ-1"]
    assert X.iloc[-1]["code_rome_vise"] == "D1503"


def test_verifier_contrat_candidat_valide() -> None:
    probas_valides = np.array([[0.2, 0.3, 0.5]])
    pipeline = _PipelineFactice(probas_valides)
    verifier_contrat_candidat(pipeline, pd.DataFrame([{"x": 1}]))  # ne doit pas lever


def test_verifier_contrat_candidat_proba_hors_bornes() -> None:
    probas_invalides = np.array([[0.2, 0.3, 1.5]])
    pipeline = _PipelineFactice(probas_invalides)
    with pytest.raises(AssertionError):
        verifier_contrat_candidat(pipeline, pd.DataFrame([{"x": 1}]))


def test_verifier_contrat_candidat_ne_somme_pas_a_un() -> None:
    probas_invalides = np.array([[0.2, 0.2, 0.2]])
    pipeline = _PipelineFactice(probas_invalides)
    with pytest.raises(AssertionError):
        verifier_contrat_candidat(pipeline, pd.DataFrame([{"x": 1}]))


def _env_isole(tmp_path: Path) -> dict:
    return {
        **os.environ,
        "FEEDBACK_DB_PATH": str(tmp_path / "feedback.db"),
        "RETRAIN_MODELS_DIR": str(tmp_path / "models"),
        "RETRAIN_LOG_PATH": str(tmp_path / "retrain_log.jsonl"),
    }


def test_cli_skip_sous_le_seuil(tmp_path: Path) -> None:
    db_path = tmp_path / "feedback.db"
    initialiser_base(db_path)
    enregistrer_prediction(
        db_path, request_id="R1", classe_predite=1, decision="retour_moyen",
        probabilites={0: 0.3, 1: 0.4, 2: 0.3}, features=FEATURES_TEST,
    )
    enregistrer_feedback(db_path, request_id="R1", true_label=1, comments=None)

    resultat = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "retrain.py"), "--min-feedback", "200"],
        env=_env_isole(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert resultat.returncode == 0
    assert '"action": "skip"' in resultat.stdout


def test_cli_pas_de_store_renvoie_zero(tmp_path: Path) -> None:
    resultat = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "retrain.py")],
        env=_env_isole(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resultat.returncode == 0
    assert "rien à faire" in resultat.stderr


@pytest.mark.skipif(
    not PRODUCTION_MODEL_PATH.exists(), reason="modèle de production non packagé"
)
def test_cli_force_entraine_et_journalise_une_decision(tmp_path: Path) -> None:
    """Test end-to-end réel : entraîne un vrai candidat (quelques secondes)."""
    db_path = tmp_path / "feedback.db"
    initialiser_base(db_path)
    enregistrer_prediction(
        db_path, request_id="R1", classe_predite=1, decision="retour_moyen",
        probabilites={0: 0.3, 1: 0.4, 2: 0.3}, features=FEATURES_TEST,
    )
    enregistrer_feedback(db_path, request_id="R1", true_label=1, comments=None)

    env = _env_isole(tmp_path)
    resultat = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "retrain.py"), "--force"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert resultat.returncode == 0, resultat.stderr
    assert "Décision :" in resultat.stdout

    log_path = tmp_path / "retrain_log.jsonl"
    assert log_path.exists()
    lignes = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lignes) == 1
    entree = json.loads(lignes[0])
    assert entree["n_feedbacks_inclus"] == 1
    assert "promote" in entree

    candidat_path = tmp_path / "models" / "trajectoire_emploi_candidate.joblib"
    assert candidat_path.exists()

    # Le feedback utilisé doit être marqué consommé.
    import sqlite3

    con = sqlite3.connect(db_path)
    row = con.execute(
        "SELECT used_for_training FROM feedbacks WHERE request_id = 'R1'"
    ).fetchone()
    con.close()
    assert row == (1,)
