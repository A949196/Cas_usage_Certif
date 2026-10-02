"""Schémas Pydantic — contrat d'entrée/sortie de l'API `trajectoire_emploi`.
`nationalite_hors_ue` est délibérément **absente** de ce schéma : le modèle
servi (S2) ne l'utilise pas.

Bornes issues de l'EDA (Étape 2) : `age` observé dans [18, 63], `anciennete
_poste_ans` observée dans [0, 22.6]. Marge de tolérance ajoutée pour ne pas
être trop restrictif, mais bornée pour éviter l'extrapolation silencieuse.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

NiveauDiplome = Literal["Sans diplôme", "Bac", "Bac+2", "Bac+5"]


class DemandeurInput(BaseModel):
    """Entrée brute d'un usager, telle que fournie par le SI de l'agence."""

    age: int = Field(..., ge=16, le=70, description="Âge en années")
    anciennete_poste_ans: float | None = Field(
        None, ge=0, le=50, description="Ancienneté dans le dernier poste, en années"
    )
    niveau_diplome: NiveauDiplome | None = Field(
        None, description="Niveau de diplôme le plus élevé obtenu"
    )
    code_rome_vise: str = Field(
        ..., min_length=5, max_length=5, description="Code ROME du métier visé"
    )
    code_insee_commune: str = Field(
        ..., min_length=5, max_length=5, description="Code INSEE de la commune de résidence"
    )
    est_allocataire: bool | None = Field(
        None, description="L'usager perçoit-il une allocation chômage ?"
    )
    synthese_entretien: str = Field(
        "", max_length=5000, description="Synthèse libre du premier entretien"
    )


class PredictionResponse(BaseModel):
    """Sortie de `/predict` — décision à coût minimal, avec abstention possible."""

    prediction: int | None = Field(
        None, description="Classe prédite (0/1/2), null si abstention (revue_humaine)"
    )
    decision: Literal[
        "retour_rapide", "retour_moyen", "risque_longue_duree", "revue_humaine"
    ]
    probabilites: dict[str, float] = Field(
        ..., description="Probabilités par classe, clés = libellés métier"
    )
    cout_attendu: float = Field(..., description="Coût attendu de la décision retenue")
    model_version: str
    request_id: str


class InfoResponse(BaseModel):
    """Sortie de `/info` — métadonnées du modèle servi."""

    api_version: str
    model_name: str
    model_version: str
    model_created_at: str
    sklearn_version: str
    dataset_sha256: str
    metrics_holdout: dict
    seuil_abstention: float


class HealthResponse(BaseModel):
    status: Literal["ok"]


class FeedbackInput(BaseModel):
    """Vérité terrain remontée par un conseiller sur un dossier déjà scoré.

    `request_id` doit correspondre à une prédiction réellement servie par
    `/predict` (sinon 404). `true_label` ∈ {0, 1, 2} (sinon 422 automatique).
    """

    request_id: str = Field(..., description="request_id renvoyé par /predict")
    true_label: int = Field(
        ..., ge=0, le=2, description="Vraie classe observée (0/1/2)"
    )
    comments: str | None = Field(None, max_length=2000)


class FeedbackResponse(BaseModel):
    """Sortie de `POST /feedback`."""

    status: Literal["cree", "idempotent"]


class FeedbackCountResponse(BaseModel):
    """Sortie de `GET /feedback/count` — `new` pilote un futur trigger de
    réentraînement, jamais `total`."""

    total: int
    new: int
