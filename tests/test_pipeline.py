"""Tests du préprocesseur (`pipeline.py`) — données synthétiques uniquement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trajectoire_emploi.pipeline import construire_preprocesseur


def _jeu_synthetique(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 63, size=n).astype(float),
            "anciennete_poste_ans": rng.uniform(0, 20, size=n),
            "niveau_diplome": rng.choice(
                ["Sans diplôme", "Bac", "Bac+2", "Bac+5"], size=n
            ),
            "code_rome_vise": rng.choice(["A1101", "K2101", "M1705"], size=n),
            "departement": rng.choice(["18", "07", "2A"], size=n),
            "est_allocataire": rng.integers(0, 2, size=n).astype(float),
            "nationalite_hors_ue": rng.integers(0, 2, size=n),
            "anciennete_incoherente": rng.integers(0, 2, size=n),
            "synthese_entretien": rng.choice(
                ["Profil autonome, projet clair.", "Freins périphériques majeurs."],
                size=n,
            ),
        }
    )


def test_scenario_inconnu_leve_une_erreur() -> None:
    with pytest.raises(ValueError):
        construire_preprocesseur("S5")


def test_s1_produit_une_matrice_sans_nan() -> None:
    df = _jeu_synthetique()
    preprocesseur = construire_preprocesseur("S1")

    X = preprocesseur.fit_transform(df)

    assert X.shape[0] == len(df)
    assert not np.isnan(X).any()


def test_s2_ignore_la_nationalite() -> None:
    df = _jeu_synthetique()
    df_modifiee = df.copy()
    df_modifiee["nationalite_hors_ue"] = 1 - df_modifiee["nationalite_hors_ue"]

    preprocesseur = construire_preprocesseur("S2")
    X1 = preprocesseur.fit_transform(df)
    X2 = construire_preprocesseur("S2").fit_transform(df_modifiee)

    np.testing.assert_array_equal(X1, X2)


def test_s3_utilise_uniquement_le_texte() -> None:
    df = _jeu_synthetique()
    df_modifiee = df.copy()
    df_modifiee["age"] = df_modifiee["age"] + 1000  # ne doit avoir aucun effet

    preprocesseur = construire_preprocesseur("S3")
    X1 = preprocesseur.fit_transform(df)
    X2 = construire_preprocesseur("S3").fit_transform(df_modifiee)

    np.testing.assert_array_equal(X1, X2)


def test_s4_exclut_rome_allocataire_nationalite_et_texte() -> None:
    df = _jeu_synthetique()
    df_modifiee = df.copy()
    df_modifiee["code_rome_vise"] = "AUTRE"
    df_modifiee["est_allocataire"] = 1 - df_modifiee["est_allocataire"]
    df_modifiee["nationalite_hors_ue"] = 1 - df_modifiee["nationalite_hors_ue"]
    df_modifiee["synthese_entretien"] = "texte totalement différent"

    preprocesseur = construire_preprocesseur("S4")
    X1 = preprocesseur.fit_transform(df)
    X2 = construire_preprocesseur("S4").fit_transform(df_modifiee)

    np.testing.assert_array_equal(X1, X2)


def test_gestion_modalite_inconnue_au_transform_sans_crash() -> None:
    df = _jeu_synthetique()
    preprocesseur = construire_preprocesseur("S1")
    preprocesseur.fit(df)

    nouvelle_ligne = df.iloc[[0]].copy()
    nouvelle_ligne["code_rome_vise"] = "CODE_JAMAIS_VU"
    nouvelle_ligne["departement"] = "99"

    # ne doit pas lever d'exception (handle_unknown="ignore")
    resultat = preprocesseur.transform(nouvelle_ligne)
    assert resultat.shape[0] == 1


def test_scaler_present_uniquement_si_demande() -> None:
    sans_scaler = construire_preprocesseur("S1", avec_scaler=False)
    avec_scaler = construire_preprocesseur("S1", avec_scaler=True)

    noms_sans = [nom for nom, _ in sans_scaler.transformers[0][1].steps]
    noms_avec = [nom for nom, _ in avec_scaler.transformers[0][1].steps]

    assert "scaler" not in noms_sans
    assert "scaler" in noms_avec
