"""Packaging du modèle final : persistance joblib + métadonnées JSON.

Le `.joblib` produit contient le **pipeline scikit-learn complet**
(préprocesseur + classifieur), jamais le classifieur seul: sinon l'API 
devrait ré-implémenter le préprocessing à la main, source de dérive silencieuse.

`metrics_holdout` n'est **jamais recalculé**.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

CLES_OBLIGATOIRES = (
    "model_version",
    "created_at",
    "sklearn_version",
    "dataset_sha256",
    "metrics_holdout",
)


def construire_metadata(
    *,
    model_name: str,
    model_version: str,
    dataset_path: Path,
    feature_columns: list[str],
    target_mapping: dict[str, int],
    metrics_holdout: dict[str, float],
    hyperparameters: dict[str, Any] | None = None,
    seuil_abstention: float | None = None,
) -> dict[str, Any]:
    """Construit le dict de métadonnées respectant le contrat des 5 clés obligatoires.

    `metrics_holdout` doit provenir du verdict du test scellé (§7.4)
    """
    metadata: dict[str, Any] = {
        "model_name": model_name,
        "model_version": model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "dataset_sha256": sha256(dataset_path.read_bytes()).hexdigest(),
        "feature_columns": feature_columns,
        "target_mapping": target_mapping,
        "metrics_holdout": metrics_holdout,
    }
    if hyperparameters is not None:
        metadata["hyperparameters"] = hyperparameters
    if seuil_abstention is not None:
        metadata["seuil_abstention"] = seuil_abstention

    manquantes = [cle for cle in CLES_OBLIGATOIRES if not metadata.get(cle)]
    if manquantes:
        raise ValueError(f"clés obligatoires manquantes ou vides : {manquantes}")

    return metadata


def persister_modele(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    output_dir: Path,
    model_name: str,
) -> tuple[Path, Path]:
    """Sérialise le pipeline complet (`.joblib`) et écrit les métadonnées (`.json`).

    Retourne `(model_path, meta_path)`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}.joblib"
    meta_path = output_dir / f"{model_name}.json"

    joblib.dump(pipeline, model_path, compress=3)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return model_path, meta_path


def charger_modele(model_path: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Recharge le pipeline persisté et ses métadonnées adjacentes (même stem)."""
    meta_path = model_path.with_suffix(".json")
    pipeline = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return pipeline, metadata
