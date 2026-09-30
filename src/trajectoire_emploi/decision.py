"""Décision à coût minimal (au lieu de l'argmax de probabilité).
Répond explicitement à la demande du sujet : « trouver quels paramètres ou
quelles métriques favoriser lors de l'apprentissage (poids des classes,
seuils de décision) ».

Principe (théorie de la décision bayésienne, il découle directement de la matrice de
coûts déjà justifiée) : au lieu de prédire `argmax(P(classe))`, on prédit
la classe qui **minimise le coût attendu** :

    coût_attendu(c) = Σ_k P(classe=k) * matrice[k][c]

et on choisit `c* = argmin_c coût_attendu(c)`. Si la matrice était uniforme
(toutes les erreurs égales), ceci redevient strictement l'argmax classique
— la matrice de coûts est donc une **généralisation**, pas un remplacement.
"""

from __future__ import annotations

import numpy as np

from trajectoire_emploi.evaluation import MATRICE_COUTS


def decision_cout_minimal(probas: np.ndarray, matrice: np.ndarray = MATRICE_COUTS) -> np.ndarray:
    """Prédit la classe qui minimise le coût attendu sous `matrice`.

    `probas` : matrice (n_echantillons, n_classes) de probabilités prédites
    (ex. sortie de `predict_proba`). `matrice[k, c]` = coût de prédire `c`
    quand la réalité est `k`.
    """
    probas = np.asarray(probas)
    # coûts_attendus[i, c] = somme_k probas[i, k] * matrice[k, c]
    couts_attendus = probas @ matrice
    return couts_attendus.argmin(axis=1)
