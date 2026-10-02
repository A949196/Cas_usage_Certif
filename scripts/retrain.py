"""Réentraînement gardé — candidat vs production.

Même recette que la production
(`Pipeline(S2 + LGBMClassifier(random_state=42, verbosity=-1))`,
`decision_avec_abstention(seuil=0.7)`).

Contrairement au modèle de production actuel (réentraîné sur train+test
combinés, le candidat est entraîné **uniquement** sur `X_train` reconstruit (2000 lignes)
+ les feedbacks non consommés. Conséquence : les métriques du candidat sur
`reference_set.csv` sont une mesure de généralisation **authentique** — celles
de la production ne le sont **pas** (elle a déjà vu ce jeu). Cette asymétrie
est structurelle : documentée plutôt que masquée.

**Pas de déploiement automatique** : une promotion ne fait que journaliser
la décision et conserver le candidat sous
`models/trajectoire_emploi_candidate.joblib`. 

Usage :
    python scripts/retrain.py                       # garde-seuil normal (200)
    python scripts/retrain.py --min-feedback 5       # seuil abaissé (démo)
    python scripts/retrain.py --force                # ignore le garde-seuil
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectoire_emploi.decision import decision_avec_abstention  
from trajectoire_emploi.evaluation import evaluer  
from trajectoire_emploi.feedback_store import (  
    charger_feedbacks_enrichis,
    compter_feedbacks,
    marquer_feedbacks_consommes,
)
from trajectoire_emploi.persistence import charger_modele, construire_metadata, persister_modele  
from trajectoire_emploi.pipeline import construire_preprocesseur  
from trajectoire_emploi.retrain_data import (  
    reconstruire_split,
    verifier_coherence_reference_set,
)

# scripts/ est déjà sur sys.path (répertoire du script courant) à l'exécution
# directe ; ajouté explicitement pour les imports via pytest (pyproject.toml).
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_model import construire_dataframe  
from promotion import decide_promotion  

DATASET_PATH = Path(os.environ.get("RETRAIN_DATASET_PATH", str(ROOT / "data" / "dataset_trajectoire_emploi_Sujet.csv")))
REFERENCE_SET_PATH = Path(os.environ.get("RETRAIN_REFERENCE_SET_PATH", str(ROOT / "data" / "reference_set.csv")))
PRODUCTION_MODEL_PATH = Path(os.environ.get("RETRAIN_PRODUCTION_MODEL_PATH", str(ROOT / "models" / "trajectoire_emploi_v1.joblib")))
CANDIDATE_MODEL_NAME = "trajectoire_emploi_candidate"
MODELS_DIR = Path(os.environ.get("RETRAIN_MODELS_DIR", str(ROOT / "models")))
FEEDBACK_DB_PATH = Path(os.environ.get("FEEDBACK_DB_PATH", str(ROOT / "data" / "runtime" / "feedback.db")))
RETRAIN_LOG_PATH = Path(os.environ.get("RETRAIN_LOG_PATH", str(ROOT / "data" / "runtime" / "retrain_log.jsonl")))

SEUIL_ABSTENTION = 0.7
SEED = 42
MIN_FEEDBACK_DEFAUT = 200


def verifier_contrat_candidat(pipeline: Pipeline, exemple: pd.DataFrame) -> None:
    """Contract test minimal : schéma accepté, probas valides.

    Lève `AssertionError` si le candidat ne respecte pas le contrat.
    """
    proba = pipeline.predict_proba(exemple)
    assert proba.shape[1] == 3, f"attendu 3 classes, obtenu {proba.shape[1]}"
    assert (proba >= 0).all() and (proba <= 1).all(), "probabilités hors [0, 1]"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), "les probabilités ne somment pas à 1"


def construire_jeu_entrainement(
    dataset_path: Path, db_path: Path
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Construit X_train/y_train enrichis des feedbacks non consommés.

    Retourne aussi la liste des `request_id` inclus — pour les marquer
    consommés après un réentraînement abouti.
    """
    X_train, _, y_train, _ = reconstruire_split(dataset_path)

    df_feedback = charger_feedbacks_enrichis(db_path, non_consommes_uniquement=True)
    request_ids = df_feedback["request_id"].tolist() if not df_feedback.empty else []

    if df_feedback.empty:
        return X_train.reset_index(drop=True), y_train.reset_index(drop=True), request_ids

    X_feedback = df_feedback.drop(columns=["request_id", "classe_retour_emploi"])
    y_feedback = df_feedback["classe_retour_emploi"]

    X_combine = pd.concat([X_train.reset_index(drop=True), X_feedback], ignore_index=True)
    y_combine = pd.concat([y_train.reset_index(drop=True), y_feedback], ignore_index=True)
    return X_combine, y_combine, request_ids


def evaluer_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    """Évalue un pipeline (candidat ou production) sur `reference_set.csv`,
    même code pour les deux."""
    X_prepare = construire_dataframe(X)
    probas = pipeline.predict_proba(X_prepare)
    resultat = decision_avec_abstention(probas, SEUIL_ABSTENTION)
    masque_garde = resultat != "revue_humaine"

    metriques = evaluer(y[masque_garde], resultat[masque_garde].astype(int))
    metriques["taux_revue_humaine"] = float((~masque_garde).mean())
    return metriques


def entrainer_candidat(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Recette identique à la production"""
    X_prepare = construire_dataframe(X_train)
    pipeline = Pipeline(
        [
            ("pre", construire_preprocesseur("S2")),
            ("m", LGBMClassifier(random_state=SEED, verbosity=-1)),
        ]
    )
    pipeline.fit(X_prepare, y_train)
    return pipeline


def journaliser_decision(log_path: Path, entree: dict) -> None:
    """Journalise chaque exécution"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entree, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-feedback", type=int, default=MIN_FEEDBACK_DEFAUT)
    parser.add_argument(
        "--force", action="store_true", help="Ignore le garde-seuil (tests/démo)"
    )
    args = parser.parse_args()

    if not FEEDBACK_DB_PATH.exists():
        print(
            f"Pas de store de feedback ({FEEDBACK_DB_PATH}) — rien à faire.",
            file=sys.stderr,
        )
        return 0

    # Compte les NON CONSOMMÉS, jamais le total.
    # Volontairement vérifié AVANT le dataset brut : le mécanisme de trigger
    # doit pouvoir être démontré (CI) sans disposer du dataset réel
    # tant qu'on reste sous le seuil.
    compte = compter_feedbacks(FEEDBACK_DB_PATH)
    if compte["new"] < args.min_feedback and not args.force:
        print(
            json.dumps(
                {"action": "skip", "reason": f"{compte['new']} nouveaux < {args.min_feedback}"}
            )
        )
        return 0

    if not DATASET_PATH.exists():
        print(f"ERREUR : {DATASET_PATH} introuvable.", file=sys.stderr)
        return 2

    # Garde-fou anti-dérive : reference_set.csv doit encore correspondre au
    # split original, sinon le candidat serait comparé à un arbitre corrompu.
    try:
        verifier_coherence_reference_set(DATASET_PATH, REFERENCE_SET_PATH)
    except AssertionError as exc:
        print(f"ERREUR : incohérence reference_set.csv — {exc}", file=sys.stderr)
        return 2

    if not PRODUCTION_MODEL_PATH.exists():
        print(f"ERREUR : modèle de production introuvable ({PRODUCTION_MODEL_PATH}).", file=sys.stderr)
        return 2
    pipeline_prod, metadata_prod = charger_modele(PRODUCTION_MODEL_PATH)

    X_train, y_train, request_ids = construire_jeu_entrainement(DATASET_PATH, FEEDBACK_DB_PATH)
    print(f"Jeu d'entraînement : {len(X_train)} lignes ({len(request_ids)} feedbacks inclus).")

    candidat = entrainer_candidat(X_train, y_train)

    # Contract test — une vraie erreur technique arrête tout avant d'évaluer.
    exemple = construire_dataframe(X_train.iloc[[0]])
    try:
        verifier_contrat_candidat(candidat, exemple)
    except AssertionError as exc:
        print(f"ERREUR : contract test du candidat échoué — {exc}", file=sys.stderr)
        return 1

    reference = pd.read_csv(REFERENCE_SET_PATH, dtype={"code_insee_commune": "string"})
    y_ref = reference["classe_retour_emploi"].to_numpy()
    X_ref = reference.drop(columns=["classe_retour_emploi"])

    metriques_candidat = evaluer_pipeline(candidat, X_ref, y_ref)
    metriques_prod = evaluer_pipeline(pipeline_prod, X_ref, y_ref)

    print("\nMétriques candidat (généralisation authentique, n'a jamais vu reference_set.csv) :")
    for cle, valeur in metriques_candidat.items():
        print(f"  {cle}: {valeur:.4f}")
    print("\nMétriques production (ATTENTION : inflatées — a déjà vu reference_set.csv à l'entraînement) :")
    for cle, valeur in metriques_prod.items():
        print(f"  {cle}: {valeur:.4f}")

    decision = decide_promotion(metriques_candidat, metriques_prod)
    print(f"\nDécision : {'PROMU' if decision.promote else 'REJETÉ'} — {decision.reason}")

    # Candidat toujours packagé (inspectable), jamais nommé version officielle.
    metadata_candidat = construire_metadata(
        model_name=CANDIDATE_MODEL_NAME,
        model_version="candidate",
        dataset_path=DATASET_PATH,
        feature_columns=list(X_train.columns),
        target_mapping={"retour_rapide": 0, "retour_moyen": 1, "risque_longue_duree": 2},
        metrics_holdout=metriques_candidat,
        hyperparameters={"random_state": SEED, "verbosity": -1},
        seuil_abstention=SEUIL_ABSTENTION,
    )
    metadata_candidat["n_feedbacks_inclus"] = len(request_ids)
    metadata_candidat["production_model_version"] = metadata_prod.get("model_version")
    metadata_candidat["decision_promotion"] = decision.promote
    metadata_candidat["decision_reason"] = decision.reason
    persister_modele(candidat, metadata_candidat, MODELS_DIR, CANDIDATE_MODEL_NAME)

    journaliser_decision(
        RETRAIN_LOG_PATH,
        {
            "horodatage": datetime.now(timezone.utc).isoformat(),
            "n_feedbacks_inclus": len(request_ids),
            "metriques_candidat": metriques_candidat,
            "metriques_production": metriques_prod,
            "promote": decision.promote,
            "reason": decision.reason,
        },
    )

    # Réentraînement abouti (promu ou rejeté) : on consomme les feedbacks
    # utilisés, qu'ils aient ou non mené à une promotion.
    marquer_feedbacks_consommes(FEEDBACK_DB_PATH, request_ids)

    if decision.promote:
        print(
            f"\nPour déployer : renommer {MODELS_DIR / (CANDIDATE_MODEL_NAME + '.joblib')} "
            f"en trajectoire_emploi_v1.joblib (geste humain délibéré, jamais automatique)."
        )

    return 0 


if __name__ == "__main__":
    sys.exit(main())
