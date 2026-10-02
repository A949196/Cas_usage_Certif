"""Tests de la politique de promotion (`scripts/promotion.py`).

Fonction pure testée sur des métriques mockées
"""

from __future__ import annotations

from promotion import decide_promotion


def _metriques(f1_macro: float, recall_classe_2: float, taux_erreur_2_vers_0: float) -> dict:
    return {
        "f1_macro": f1_macro,
        "recall_classe_2": recall_classe_2,
        "taux_erreur_2_vers_0": taux_erreur_2_vers_0,
    }


def test_gain_net_promu() -> None:
    """Candidat moins bon en f1_macro (dans la tolérance) mais meilleur sur
    la métrique critique : doit être promu."""
    production = _metriques(f1_macro=0.70, recall_classe_2=0.70, taux_erreur_2_vers_0=0.05)
    candidat = _metriques(f1_macro=0.69, recall_classe_2=0.715, taux_erreur_2_vers_0=0.05)

    decision = decide_promotion(candidat, production)

    assert decision.promote is True
    assert "recall_classe_2" in decision.reason


def test_regression_critique_rejetee() -> None:
    """Le candidat dépasse le plancher absolu de taux_erreur_2_vers_0 : rejeté,
    même s'il gagne par ailleurs sur recall_classe_2."""
    production = _metriques(f1_macro=0.70, recall_classe_2=0.70, taux_erreur_2_vers_0=0.05)
    candidat = _metriques(f1_macro=0.70, recall_classe_2=0.75, taux_erreur_2_vers_0=0.15)

    decision = decide_promotion(candidat, production)

    assert decision.promote is False
    assert "plancher" in decision.reason


def test_candidat_identique_rejete() -> None:
    """Un candidat rigoureusement équivalent n'achète rien — rejeté (pas de
    gain minimum sur la métrique critique)."""
    production = _metriques(f1_macro=0.70, recall_classe_2=0.70, taux_erreur_2_vers_0=0.05)
    candidat = _metriques(f1_macro=0.70, recall_classe_2=0.70, taux_erreur_2_vers_0=0.05)

    decision = decide_promotion(candidat, production)

    assert decision.promote is False
    assert "Gain insuffisant" in decision.reason


def test_sous_le_plancher_rejete() -> None:
    """Un candidat sous le plancher absolu de f1_macro est rejeté d'emblée,
    même s'il améliore le recall de la classe critique."""
    production = _metriques(f1_macro=0.70, recall_classe_2=0.70, taux_erreur_2_vers_0=0.05)
    candidat = _metriques(f1_macro=0.40, recall_classe_2=0.90, taux_erreur_2_vers_0=0.05)

    decision = decide_promotion(candidat, production)

    assert decision.promote is False
    assert "plancher" in decision.reason


def test_recul_f1_hors_tolerance_rejete() -> None:
    production = _metriques(f1_macro=0.70, recall_classe_2=0.70, taux_erreur_2_vers_0=0.05)
    candidat = _metriques(f1_macro=0.60, recall_classe_2=0.80, taux_erreur_2_vers_0=0.05)

    decision = decide_promotion(candidat, production)

    assert decision.promote is False
    assert "f1_macro recule" in decision.reason


def test_hausse_erreur_critique_hors_tolerance_rejetee() -> None:
    production = _metriques(f1_macro=0.70, recall_classe_2=0.70, taux_erreur_2_vers_0=0.005)
    candidat = _metriques(f1_macro=0.70, recall_classe_2=0.75, taux_erreur_2_vers_0=0.10)

    decision = decide_promotion(candidat, production)

    assert decision.promote is False
    assert "taux_erreur_2_vers_0 augmente" in decision.reason


def test_reason_toujours_non_vide() -> None:
    production = _metriques(f1_macro=0.70, recall_classe_2=0.70, taux_erreur_2_vers_0=0.05)
    candidat = _metriques(f1_macro=0.70, recall_classe_2=0.70, taux_erreur_2_vers_0=0.05)

    decision = decide_promotion(candidat, production)

    assert decision.reason.strip() != ""
