"""Calibration du modèle — reliability diagram et ECE.

Calibration en **conception** (choisir/valider un modèle avant mise en
production), sur des probabilités **out-of-fold** obtenues par validation
croisée sur le train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_diagram_data(proba_classe: np.ndarray, y_est_classe: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Données du reliability diagram : confiance moyenne vs taux observé, par bin.

    `proba_classe` : probabilité prédite de la classe d'intérêt (ex. classe 2).
    `y_est_classe` : booléen, `y_true == classe d'intérêt`.
    """
    df = pd.DataFrame({"p": np.asarray(proba_classe), "y": np.asarray(y_est_classe).astype(int)})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, n_bins + 1), include_lowest=True, labels=False)
    g = df.groupby("bin").agg(n=("y", "size"), confiance=("p", "mean"), observe=("y", "mean"))
    return g.reset_index()


def ece(proba_classe: np.ndarray, y_est_classe: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (formule de la fiche 613, telle quelle)."""
    g = reliability_diagram_data(proba_classe, y_est_classe, n_bins)
    poids = g["n"] / g["n"].sum()
    return float((poids * (g["confiance"] - g["observe"]).abs()).sum())
