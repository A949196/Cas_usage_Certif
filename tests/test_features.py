"""Tests de `features.extraire_departement` — données synthétiques uniquement."""

from __future__ import annotations

import pandas as pd

from trajectoire_emploi.features import extraire_departement


def test_extraire_departement_metropole_standard() -> None:
    codes = pd.Series(["18273", "07240", "01405", "63359", "21302"], dtype="string")

    resultat = extraire_departement(codes)

    assert resultat.tolist() == ["18", "07", "01", "63", "21"]


def test_extraire_departement_corse() -> None:
    codes = pd.Series(["2A061", "2B304"], dtype="string")

    resultat = extraire_departement(codes)

    assert resultat.tolist() == ["2A", "2B"]


def test_extraire_departement_dom() -> None:
    codes = pd.Series(["97228", "97601", "97116"], dtype="string")

    resultat = extraire_departement(codes)

    assert resultat.tolist() == ["972", "976", "971"]


def test_extraire_departement_conserve_les_manquants() -> None:
    codes = pd.Series(["18273", pd.NA], dtype="string")

    resultat = extraire_departement(codes)

    assert resultat.iloc[0] == "18"
    assert pd.isna(resultat.iloc[1])
