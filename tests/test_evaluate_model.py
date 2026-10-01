"""Tests de la logique de seuils du garde-fou d'évaluation continue (fiche 517).

On ne teste pas ici `calculer_metriques` (qui charge le vrai modèle — c'est
le rôle du contract test, `tests/test_model_contract.py`), seulement la
fonction pure `verifier_seuils`, indépendante du modèle.
"""

from __future__ import annotations

from evaluate_model import THRESHOLDS, verifier_seuils


def test_aucune_violation_sur_metriques_identiques_au_golden() -> None:
    golden = {"f1_macro": 0.72, "taux_erreur_2_vers_0": 0.03}
    violations = verifier_seuils(golden, golden)
    assert violations == []


def test_violation_f1_macro_sous_le_minimum_absolu() -> None:
    golden = {"f1_macro": 0.72, "taux_erreur_2_vers_0": 0.03}
    metriques = {"f1_macro": 0.40, "taux_erreur_2_vers_0": 0.03}
    violations = verifier_seuils(metriques, golden)
    assert any("minimum absolu" in v for v in violations)


def test_violation_taux_erreur_2_vers_0_au_dessus_du_maximum_absolu() -> None:
    golden = {"f1_macro": 0.72, "taux_erreur_2_vers_0": 0.03}
    metriques = {"f1_macro": 0.72, "taux_erreur_2_vers_0": 0.50}
    violations = verifier_seuils(metriques, golden)
    assert any("maximum absolu" in v for v in violations)


def test_violation_chute_relative_vs_golden_meme_au_dessus_du_minimum_absolu() -> None:
    golden = {"f1_macro": 0.90, "taux_erreur_2_vers_0": 0.0}
    # 0.70 reste au-dessus du minimum absolu (0.55) mais chute de 0.20 vs golden
    metriques = {"f1_macro": 0.70, "taux_erreur_2_vers_0": 0.0}
    violations = verifier_seuils(metriques, golden)
    assert any("a chuté" in v for v in violations)


def test_pas_de_violation_si_ecart_sous_la_tolerance() -> None:
    golden = {"f1_macro": 0.72, "taux_erreur_2_vers_0": 0.03}
    metriques = {"f1_macro": 0.715, "taux_erreur_2_vers_0": 0.035}
    violations = verifier_seuils(metriques, golden)
    assert violations == []


def test_thresholds_couvre_les_deux_metriques_attendues() -> None:
    assert "f1_macro" in THRESHOLDS
    assert "taux_erreur_2_vers_0" in THRESHOLDS
