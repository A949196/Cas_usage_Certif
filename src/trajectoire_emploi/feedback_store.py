"""Store SQLite — prédictions servies et feedbacks des conseillers.

Deux tables :
- `predictions` : chaque décision rendue par `/predict` (request_id, classe
  prédite, probabilités). Aucune feature, aucune PII.
- `feedbacks` : vérité terrain remontée par un conseiller, liée par
  `request_id`. Schéma minimal : `request_id` (PK),`true_label`, `comments`, `created_at`,
  `used_for_training` (défaut 0,consommé lors d'un futur réentraînement).

Politique de doublon : on lit AVANT d'écrire.
- Même `request_id`, même label → idempotent, pas de doublon.
- Même `request_id`, label différent → `FeedbackConflictError` (409), jamais
  d'écrasement silencieux.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


class FeedbackConflictError(Exception):
    """Levée quand un feedback contradictoire est soumis sur un `request_id`
    déjà annoté (même id, label différent) — à arbitrer par un humain."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    classe_predite INTEGER,
    decision TEXT NOT NULL,
    proba_0 REAL NOT NULL,
    proba_1 REAL NOT NULL,
    proba_2 REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS feedbacks (
    request_id TEXT PRIMARY KEY,
    true_label INTEGER NOT NULL,
    comments TEXT,
    created_at TEXT NOT NULL,
    used_for_training INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (request_id) REFERENCES predictions(request_id)
);
"""


def initialiser_base(db_path: Path) -> None:
    """Crée le fichier et les tables si nécessaire (idempotent)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(_SCHEMA)


@contextmanager
def _connexion(db_path: Path) -> Iterator[sqlite3.Connection]:
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
    finally:
        con.close()


def enregistrer_prediction(
    db_path: Path,
    *,
    request_id: str,
    classe_predite: int | None,
    decision: str,
    probabilites: dict[int, float],
) -> None:
    """Journalise une prédiction servie.

    `INSERT OR IGNORE` par défense (un `request_id` dupliqué ne doit jamais
    faire échouer `/predict`, même si en pratique l'UUID est unique).
    """
    with _connexion(db_path) as con:
        con.execute(
            "INSERT OR IGNORE INTO predictions "
            "(request_id, created_at, classe_predite, decision, proba_0, proba_1, proba_2) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                request_id,
                datetime.now(timezone.utc).isoformat(),
                classe_predite,
                decision,
                probabilites.get(0, 0.0),
                probabilites.get(1, 0.0),
                probabilites.get(2, 0.0),
            ),
        )
        con.commit()


def request_id_connu(db_path: Path, request_id: str) -> bool:
    """True si `request_id` correspond à une prédiction réellement servie."""
    with _connexion(db_path) as con:
        row = con.execute(
            "SELECT 1 FROM predictions WHERE request_id = ?", (request_id,)
        ).fetchone()
        return row is not None


def enregistrer_feedback(
    db_path: Path,
    *,
    request_id: str,
    true_label: int,
    comments: str | None,
) -> str:
    """Insère un feedback. Retourne `"cree"` ou `"idempotent"`.

    Lève `FeedbackConflictError` si un label différent existe déjà pour ce `request_id`.
    """
    with _connexion(db_path) as con:
        row = con.execute(
            "SELECT true_label FROM feedbacks WHERE request_id = ?", (request_id,)
        ).fetchone()
        if row is not None:
            if row[0] == true_label:
                return "idempotent"
            raise FeedbackConflictError(
                f"Feedback contradictoire pour request_id={request_id} : "
                f"déjà enregistré avec le label {row[0]}, reçu {true_label} — "
                "arbitrage humain requis."
            )
        con.execute(
            "INSERT INTO feedbacks (request_id, true_label, comments, created_at) "
            "VALUES (?, ?, ?, ?)",
            (request_id, true_label, comments, datetime.now(timezone.utc).isoformat()),
        )
        con.commit()
        return "cree"


def compter_feedbacks(db_path: Path) -> dict[str, int]:
    """Retourne `{"total": N, "new": N_non_consommes}`.

    C'est `new` (`used_for_training = 0`) qui doit piloter un futur trigger de
    réentraînement — jamais `total`.
    """
    with _connexion(db_path) as con:
        total = con.execute("SELECT COUNT(*) FROM feedbacks").fetchone()[0]
        new = con.execute(
            "SELECT COUNT(*) FROM feedbacks WHERE used_for_training = 0"
        ).fetchone()[0]
    return {"total": total, "new": new}
