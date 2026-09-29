"""Tests du module `fairness` — données synthétiques uniquement (jamais `data/*.csv`).
"""

from __future__ import annotations

import pandas as pd

from trajectoire_emploi.fairness import (
    disparate_impact,
    supports_fiables,
    verdict_4_5,
)


def test_disparate_impact_detecte_un_signal_clair() -> None:
    # Groupe minoritaire clairement désavantagé (SR très inférieur).
    df = pd.DataFrame(
        {
            "groupe": ["a"] * 10 + ["b"] * 10,
            "cible": [0] * 8 + [1] * 2 + [0] * 2 + [1] * 8,
        }
    )
    di, sr = disparate_impact(df, "groupe", "cible", est_positif=lambda s: s == 0)

    assert round(di, 3) == 0.25  # SR(b)=0.2 / SR(a)=0.8
    assert set(sr.index) == {"a", "b"}
    assert "signal" in verdict_4_5(di)
    assert "⚠️" in verdict_4_5(di)


def test_disparate_impact_pas_de_signal_si_taux_proches() -> None:
    df = pd.DataFrame(
        {
            "groupe": ["a"] * 10 + ["b"] * 10,
            "cible": [0] * 9 + [1] * 1 + [0] * 8 + [1] * 2,
        }
    )
    di, _ = disparate_impact(df, "groupe", "cible", est_positif=lambda s: s == 0)

    assert di >= 0.8
    assert verdict_4_5(di) == "✅ pas de signal au sens de la règle 4/5"


def test_supports_fiables_signale_les_petits_effectifs() -> None:
    effectifs = pd.Series({"a_x": 500, "b_y": 12, "c_z": 45})

    fragiles = supports_fiables(effectifs)

    assert list(fragiles.index) == ["b_y"]


def test_disparate_impact_avec_cible_ordinale_a_trois_classes() -> None:
    # Reflète notre cas réel : cible ordinale (0/1/2), positif = classe 0.
    df = pd.DataFrame(
        {
            "nationalite_hors_ue": [0] * 6 + [1] * 6,
            "classe_retour_emploi": [0, 0, 0, 1, 1, 2] + [0, 1, 1, 2, 2, 2],
        }
    )
    di, sr = disparate_impact(
        df,
        "nationalite_hors_ue",
        "classe_retour_emploi",
        est_positif=lambda s: s == 0,
    )

    assert round(sr[0], 3) == 0.5
    assert round(sr[1], 3) == round(1 / 6, 3)
    assert round(di, 3) == round((1 / 6) / 0.5, 3)
