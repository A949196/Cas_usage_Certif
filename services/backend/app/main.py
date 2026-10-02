"""API FastAPI — service de scoring `trajectoire_emploi`.

Le modèle servi est (`models/trajectoire_emploi_v1.joblib` + `.json`) : pipeline S2-LightGBM
complet (préprocesseur + classifieur).

Le feature engineering appliqué ici (département, ancienneté incohérente)
réutilise **exactement** les mêmes fonctions que le notebook (`trajectoire_emploi.features`).

Observabilité :
`/metrics` expose les métriques HTTP automatiques (latence, volume, codes
retour) via `prometheus-fastapi-instrumentator`.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from loguru import logger
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectoire_emploi.decision import ( 
    cout_attendu_minimal,
    decision_avec_abstention,
)
from trajectoire_emploi.feedback_store import (
    FeedbackConflictError,
    compter_feedbacks,
    enregistrer_feedback,
    enregistrer_prediction,
    initialiser_base,
    request_id_connu,
)
from trajectoire_emploi.features import (  
    extraire_departement,
    nettoyer_anciennete_incoherente,
)
from trajectoire_emploi.persistence import charger_modele  

from app.middleware import LoggingMiddleware
from app.schemas import (
    DemandeurInput,
    FeedbackCountResponse,
    FeedbackInput,
    FeedbackResponse,
    HealthResponse,
    InfoResponse,
    PredictionResponse,
)

MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "trajectoire_emploi_v1.joblib"
)

# Store SQLite des prédictions servies + feedbacks conseillers.
# Chemin surchargeable par variable d'environnement (tests, déploiements).
FEEDBACK_DB_PATH = Path(
    os.environ.get(
        "FEEDBACK_DB_PATH",
        str(Path(__file__).resolve().parents[3] / "data" / "runtime" / "feedback.db"),
    )
)

LIBELLE_CLASSE = {
    0: "retour_rapide",
    1: "retour_moyen",
    2: "risque_longue_duree",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model, app.state.metadata = charger_modele(MODEL_PATH)
        logger.info(
            f"Modèle chargé : {app.state.metadata['model_name']} "
            f"{app.state.metadata['model_version']}"
        )
    except FileNotFoundError:
        app.state.model = None
        app.state.metadata = None
        logger.error(f"Modèle introuvable à {MODEL_PATH} — l'API démarre en mode dégradé")
    initialiser_base(FEEDBACK_DB_PATH)
    yield
    app.state.model = None


app = FastAPI(
    title="Trajectoire Emploi — API de scoring",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(LoggingMiddleware)

# Observabilité : /metrics (HTTP auto) + métrique métier.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

DECISIONS = Counter(
    "trajectoire_emploi_decisions_total",
    "Nombre de décisions rendues par /predict, par type de décision",
    ["decision"],
)


def _construire_dataframe(item: DemandeurInput) -> pd.DataFrame:
    """Applique le feature engineering d'entraînement à une entrée unique.
    """
    df = pd.DataFrame([item.model_dump()])
    df["departement"] = extraire_departement(df["code_insee_commune"].astype("string"))
    df["anciennete_poste_ans"], df["anciennete_incoherente"] = nettoyer_anciennete_incoherente(
        df["age"], df["anciennete_poste_ans"]
    )
    return df


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if app.state.model is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Modèle non chargé")
    return HealthResponse(status="ok")


@app.get("/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    if app.state.metadata is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Modèle non chargé")
    meta: dict[str, Any] = app.state.metadata
    return InfoResponse(
        api_version=app.version,
        model_name=meta["model_name"],
        model_version=meta["model_version"],
        model_created_at=meta["created_at"],
        sklearn_version=meta["sklearn_version"],
        dataset_sha256=meta["dataset_sha256"],
        metrics_holdout=meta["metrics_holdout"],
        seuil_abstention=meta.get("seuil_abstention", 0.7),
    )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(item: DemandeurInput, request: Request) -> PredictionResponse:
    # Même request_id que celui posé dans le header par le middleware (M5-B1) :
    # un seul identifiant pour corréler logs, réponse et feedback ultérieur.
    request_id = request.state.request_id

    if app.state.model is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Modèle non chargé")

    try:
        df = _construire_dataframe(item)
        probas = app.state.model.predict_proba(df)
    except Exception as exc: 
        logger.bind(request_id=request_id).exception("Échec de la prédiction")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Échec de la prédiction : {exc}"
        ) from exc

    seuil_abstention = app.state.metadata.get("seuil_abstention", 0.7)
    resultat = decision_avec_abstention(probas, seuil_abstention)[0]
    cout = float(cout_attendu_minimal(probas)[0])

    classes = app.state.model.classes_
    probas_par_classe = {int(c): float(p) for c, p in zip(classes, probas[0])}
    probabilites = {
        LIBELLE_CLASSE[c]: p for c, p in probas_par_classe.items()
    }

    if resultat == "revue_humaine":
        prediction = None
        decision_label = "revue_humaine"
    else:
        prediction = int(resultat)
        decision_label = LIBELLE_CLASSE[prediction]

    DECISIONS.labels(decision=decision_label).inc()

    try:
        enregistrer_prediction(
            FEEDBACK_DB_PATH,
            request_id=request_id,
            classe_predite=prediction,
            decision=decision_label,
            probabilites=probas_par_classe,
        )
    except Exception:
        # Ne bloque jamais /predict : un feedback ultérieur sur ce
        # request_id sera simplement rejeté en 404 (journalisation prioritaire
        # sur la disponibilité du store de feedback).
        logger.bind(request_id=request_id).exception(
            "Échec de la journalisation de la prédiction (store feedback)"
        )

    return PredictionResponse(
        prediction=prediction,
        decision=decision_label,
        probabilites=probabilites,
        cout_attendu=cout,
        model_version=app.state.metadata["model_version"],
        request_id=request_id,
    )


@app.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def post_feedback(item: FeedbackInput) -> FeedbackResponse:
    """Enregistre la vraie classe d'un dossier déjà scoré.

    404 si `request_id` ne correspond à aucune prédiction réellement servie ;
    409 si un label différent a déjà été enregistré pour ce `request_id`
    (contradiction — arbitrage humain requis, jamais d'écrasement silencieux).
    """
    if not request_id_connu(FEEDBACK_DB_PATH, item.request_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "request_id inconnu")

    try:
        statut = enregistrer_feedback(
            FEEDBACK_DB_PATH,
            request_id=item.request_id,
            true_label=item.true_label,
            comments=item.comments,
        )
    except FeedbackConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    return FeedbackResponse(status=statut)


@app.get("/feedback/count", response_model=FeedbackCountResponse)
async def feedback_count() -> FeedbackCountResponse:
    """Total de feedbacks stockés et nombre de non-consommés (`new`)."""
    return FeedbackCountResponse(**compter_feedbacks(FEEDBACK_DB_PATH))
