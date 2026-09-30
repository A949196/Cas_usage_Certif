"""Tests de `features.extraire_departement` — données synthétiques uniquement."""

from __future__ import annotations

import pandas as pd

from trajectoire_emploi.features import (
    extraire_departement,
    nettoyer_anciennete_incoherente,
)


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


def test_nettoyer_anciennete_incoherente_neutralise_et_flague() -> None:
    age = pd.Series([40, 25, 50])
    anciennete = pd.Series([10, 20, 5], dtype="float64")  # ligne 1 incohérente (20 > 25-15=10)

    anciennete_nettoyee, flag = nettoyer_anciennete_incoherente(age, anciennete)

    assert pd.isna(anciennete_nettoyee.iloc[1])
    assert anciennete_nettoyee.iloc[0] == 10
    assert anciennete_nettoyee.iloc[2] == 5
    assert flag.tolist() == [0, 1, 0]


def test_nettoyer_anciennete_incoherente_aucune_ligne_incoherente() -> None:
    age = pd.Series([40, 50])
    anciennete = pd.Series([5, 10], dtype="float64")

    anciennete_nettoyee, flag = nettoyer_anciennete_incoherente(age, anciennete)

    assert anciennete_nettoyee.tolist() == [5, 10]
    assert flag.tolist() == [0, 0]


def test_nettoyer_anciennete_incoherente_conserve_les_manquants() -> None:
    age = pd.Series([40, 30])
    anciennete = pd.Series([float("nan"), 5], dtype="float64")

    anciennete_nettoyee, flag = nettoyer_anciennete_incoherente(age, anciennete)

    assert pd.isna(anciennete_nettoyee.iloc[0])
    assert flag.iloc[0] == 0
