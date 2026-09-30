"""API FastAPI — service de scoring `trajectoire_emploi` (Étape 8, Lot 1).

Le modèle servi est celui packagé en §8.2 du notebook
(`models/trajectoire_emploi_v1.joblib` + `.json`) : pipeline S2-LightGBM
complet (préprocesseur + classifieur).

Le feature engineering appliqué ici (département, ancienneté incohérente)
réutilise **exactement** les mêmes fonctions que le notebook
(`trajectoire_emploi.features`).
"""

from __future__ import annotations

import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from loguru import logger

SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectoire_emploi.decision import (  # noqa: E402
    cout_attendu_minimal,
    decision_avec_abstention,
)
from trajectoire_emploi.features import (  # noqa: E402
    extraire_departement,
    nettoyer_anciennete_incoherente,
)
from trajectoire_emploi.persistence import charger_modele  # noqa: E402

from app.middleware import LoggingMiddleware  # noqa: E402
from app.schemas import (  # noqa: E402
    DemandeurInput,
    HealthResponse,
    InfoResponse,
    PredictionResponse,
)

MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "trajectoire_emploi_v1.joblib"
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
    yield
    app.state.model = None


app = FastAPI(
    title="Trajectoire Emploi — API de scoring",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(LoggingMiddleware)


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
async def predict(item: DemandeurInput) -> PredictionResponse:
    request_id = str(uuid.uuid4())

    if app.state.model is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Modèle non chargé")

    try:
        df = _construire_dataframe(item)
        probas = app.state.model.predict_proba(df)
    except Exception as exc:  # modèle cassé, pas un problème d'entrée (déjà validée par Pydantic)
        logger.bind(request_id=request_id).exception("Échec de la prédiction")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Échec de la prédiction : {exc}"
        ) from exc

    seuil_abstention = app.state.metadata.get("seuil_abstention", 0.7)
    resultat = decision_avec_abstention(probas, seuil_abstention)[0]
    cout = float(cout_attendu_minimal(probas)[0])

    classes = app.state.model.classes_
    probabilites = {
        LIBELLE_CLASSE[int(c)]: float(p) for c, p in zip(classes, probas[0])
    }

    if resultat == "revue_humaine":
        prediction = None
        decision_label = "revue_humaine"
    else:
        prediction = int(resultat)
        decision_label = LIBELLE_CLASSE[prediction]

    return PredictionResponse(
        prediction=prediction,
        decision=decision_label,
        probabilites=probabilites,
        cout_attendu=cout,
        model_version=app.state.metadata["model_version"],
        request_id=request_id,
    )
