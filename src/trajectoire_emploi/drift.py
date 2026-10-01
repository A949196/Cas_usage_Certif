"""Détection de dérive — PSI / KS / Chi² et diagnostic data vs concept drift.

Utilisé en §9 du notebook — **en mode batch**, jamais exposé à Grafana;

Limite assumée de cette démonstration (documentée en §9 du notebook) :
faute de trafic de production réel, les fonctions ci-dessous sont illustrées
en comparant **train vs test** comme proxy de « référence vs nouvelle
population » — ce n'est pas une vraie mesure de dérive temporelle de
production, seulement une démonstration de la méthode avec des données
réelles du projet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp


def psi(
    reference: pd.Series,
    courant: pd.Series,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index entre deux échantillons numériques.

    Repères conventionnels (heuristiques, pas des lois statistiques) :
    < 0.10 signal faible, 0.10-0.25 à investiguer, > 0.25 signal fort.
    """
    reference = pd.Series(reference).dropna()
    courant = pd.Series(courant).dropna()

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    p_ref = np.histogram(reference, edges)[0] / len(reference)
    p_cur = np.histogram(courant, edges)[0] / len(courant)

    p_ref, p_cur = p_ref + eps, p_cur + eps
    p_ref, p_cur = p_ref / p_ref.sum(), p_cur / p_cur.sum()

    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def verdict_psi(valeur_psi: float) -> str:
    """Traduit un PSI en verdict lisible (repères conventionnels)."""
    if valeur_psi < 0.10:
        return "stable (signal faible)"
    if valeur_psi < 0.25:
        return "a investiguer"
    return "derive forte"


def calculer_ks(reference: pd.Series, courant: pd.Series) -> tuple[float, float]:
    """Test de Kolmogorov-Smirnov entre deux échantillons numériques.

    Retourne `(statistique, p_value)`. p < 0.05 ⇒ assez d'évidence contre
    l'égalité des distributions — **très sensible sur gros échantillons**
    """
    resultat = ks_2samp(
        pd.Series(reference).dropna(), pd.Series(courant).dropna()
    )
    return float(resultat.statistic), float(resultat.pvalue)


def calculer_chi2(reference: pd.Series, courant: pd.Series) -> tuple[float, float]:
    """Test du Chi² entre deux échantillons catégoriels (table de contingence).

    Retourne `(statistique, p_value)`. Les modalités absentes d'un des
    deux échantillons sont alignées (comptées à 0);
    """
    reference = pd.Series(reference).dropna()
    courant = pd.Series(courant).dropna()

    modalites = sorted(set(reference.unique()) | set(courant.unique()))
    effectifs_ref = reference.value_counts().reindex(modalites, fill_value=0)
    effectifs_cur = courant.value_counts().reindex(modalites, fill_value=0)

    table = np.array([effectifs_ref.to_numpy(), effectifs_cur.to_numpy()])
    statistique, p_value, _, _ = chi2_contingency(table)
    return float(statistique), float(p_value)


def diagnostic_data_vs_concept_drift(
    auc_reference: float, auc_courant: float, seuil_delta: float = 0.03
) -> str:
    """Diagnostic data drift vs concept drift à partir du delta d'AUC.

    L'AUC mesure le pouvoir de tri indépendamment du seuil. Une AUC stable
    est **compatible** avec un data drift (le modèle trie encore bien) —
    ça ne **prouve** pas que la relation X→Y est intacte.
    """
    delta = auc_courant - auc_reference
    if abs(delta) < seuil_delta:
        return (
            f"AUC stable (Δ={delta:+.3f}) → compatible data drift ; "
            "a croiser avec PSI/KS/Chi2 et la calibration"
        )
    return (
        f"AUC dégradée (Δ={delta:+.3f}) → a investiguer : concept drift ? "
        "qualité des données ? changement de population ?"
    )
