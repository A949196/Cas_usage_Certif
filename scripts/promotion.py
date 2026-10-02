"""Politique de promotion — candidat vs production (fiche 625).

Une politique, pas un seuil isolé : un **plancher** absolu (le candidat
n'est pas cassé), une **non-régression** (il ne dégrade rien d'important au-
delà du bruit mesuré), un **gain minimum** sur la métrique critique du
projet (le candidat doit apporter assez pour justifier un redéploiement).

Métrique critique = `recall_classe_2` : dans ce projet, rater la classe 2
(risque de longue durée) prive l'usager d'un accompagnement renforcé —
l'erreur la plus coûteuse (cf. matrice de coûts, cadrage §1.4). Un candidat
qui améliore tout sauf `recall_classe_2` n'apporte rien sur ce qui compte le
plus métier.

Tolérances reprises de `scripts/evaluate_model.py` (mêmes justifications
bootstrap/inter-folds, fiche 517) — mêmes seuils, nouveau contexte de
comparaison (candidat vs production réellement déployée, pas golden run).

Fonction pure, testée sur des dictionnaires de métriques mockés (fiche 625 :
*« aucune dépendance à scikit-learn »*) — jamais sur un entraînement réel,
pour rester stable et rapide.
"""

from __future__ import annotations

from dataclasses import dataclass

# Plancher absolu (fiche 517, mêmes seuils que evaluate_model.py).
PLANCHER_F1_MACRO_MIN = 0.55
PLANCHER_TAUX_ERREUR_2_VERS_0_MAX = 0.10

# Non-régression vs production (tolérance = bruit d'échantillonnage mesuré,
# cf. scripts/evaluate_model.py pour la justification détaillée).
TOLERANCE_F1_MACRO_RECUL = 0.05
TOLERANCE_TAUX_ERREUR_2_VERS_0_HAUSSE = 0.09

# Gain minimum exigé sur la métrique critique pour justifier un redéploiement.
GAIN_MINIMUM_RECALL_CLASSE_2 = 0.01


@dataclass(frozen=True)
class PromotionDecision:
    """Décision de promotion, motivée — `reason` doit se comprendre sans le
    code sous les yeux (fiche 625 : *« la reason fait partie du livrable »*)."""

    promote: bool
    reason: str


def decide_promotion(
    candidate: dict[str, float], production: dict[str, float]
) -> PromotionDecision:
    """Décide si `candidate` doit remplacer `production`.

    Ordre des vérifications (fiche 624/625) :
    1. Plancher absolu — le candidat n'est pas cassé.
    2. Non-régression — il ne dégrade rien d'important au-delà du bruit.
    3. Gain minimum — il apporte assez sur la métrique critique pour
       justifier le risque d'un redéploiement.
    """
    f1_candidat = candidate["f1_macro"]
    f1_prod = production["f1_macro"]
    erreur_candidat = candidate["taux_erreur_2_vers_0"]
    erreur_prod = production["taux_erreur_2_vers_0"]
    recall_candidat = candidate["recall_classe_2"]
    recall_prod = production["recall_classe_2"]

    if f1_candidat < PLANCHER_F1_MACRO_MIN:
        return PromotionDecision(
            promote=False,
            reason=(
                f"f1_macro={f1_candidat:.4f} sous le plancher absolu "
                f"{PLANCHER_F1_MACRO_MIN} — candidat cassé."
            ),
        )
    if erreur_candidat > PLANCHER_TAUX_ERREUR_2_VERS_0_MAX:
        return PromotionDecision(
            promote=False,
            reason=(
                f"taux_erreur_2_vers_0={erreur_candidat:.4f} au-dessus du "
                f"plancher absolu {PLANCHER_TAUX_ERREUR_2_VERS_0_MAX} — "
                "candidat cassé sur l'erreur critique."
            ),
        )

    recul_f1 = f1_prod - f1_candidat
    if recul_f1 > TOLERANCE_F1_MACRO_RECUL:
        return PromotionDecision(
            promote=False,
            reason=(
                f"f1_macro recule de {f1_prod:.4f} à {f1_candidat:.4f} "
                f"(-{recul_f1:.4f}, > tolérance {TOLERANCE_F1_MACRO_RECUL})."
            ),
        )

    hausse_erreur = erreur_candidat - erreur_prod
    if hausse_erreur > TOLERANCE_TAUX_ERREUR_2_VERS_0_HAUSSE:
        return PromotionDecision(
            promote=False,
            reason=(
                f"taux_erreur_2_vers_0 augmente de {erreur_prod:.4f} à "
                f"{erreur_candidat:.4f} (+{hausse_erreur:.4f}, > tolérance "
                f"{TOLERANCE_TAUX_ERREUR_2_VERS_0_HAUSSE})."
            ),
        )

    gain_recall = recall_candidat - recall_prod
    if gain_recall < GAIN_MINIMUM_RECALL_CLASSE_2:
        return PromotionDecision(
            promote=False,
            reason=(
                f"Gain insuffisant sur recall_classe_2 : {recall_prod:.4f} → "
                f"{recall_candidat:.4f} ({gain_recall:+.4f}, < gain minimum "
                f"{GAIN_MINIMUM_RECALL_CLASSE_2}) — candidat rigoureusement "
                "équivalent, un redéploiement n'achète rien."
            ),
        )

    return PromotionDecision(
        promote=True,
        reason=(
            f"Gain net : recall_classe_2 {recall_prod:.4f} → {recall_candidat:.4f} "
            f"({gain_recall:+.4f}). Arbitrage assumé : f1_macro "
            f"{f1_prod:.4f} → {f1_candidat:.4f} ({-recul_f1:+.4f}), "
            f"taux_erreur_2_vers_0 {erreur_prod:.4f} → {erreur_candidat:.4f} "
            f"({hausse_erreur:+.4f}), les deux dans la tolérance."
        ),
    )
