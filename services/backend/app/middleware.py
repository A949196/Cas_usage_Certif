"""Middleware de logging

Log d'accès structuré (JSON), avec `request_id` corrélable, **sans aucune
PII** : ni le body de la requête (qui contient `synthese_entretien`, texte
libre potentiellement identifiant), ni le résultat de prédiction détaillé.
Seuls method/path/status/latence/request_id sont journalisés.
"""

from __future__ import annotations

import time
import uuid

from fastapi import Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    """Journalise chaque requête : request_id, method, path, status, latence."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            logger.bind(request_id=request_id).exception(
                "Exception non gérée durant la requête"
            )
            raise

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        if status_code < 400:
            log_level = "INFO"
        elif status_code < 500:
            log_level = "WARNING"
        else:
            log_level = "ERROR"

        logger.bind(request_id=request_id).log(
            log_level,
            "{method} {path} {status} {latency_ms}ms",
            method=request.method,
            path=request.url.path,
            status=status_code,
            latency_ms=latency_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response
