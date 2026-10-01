# Dockerfile du service backend (API de scoring trajectoire_emploi)
# (base slim, user non-root, cache des layers, healthcheck).
#
# Contexte de build = RACINE DU DÉPÔT (pas ce dossier) : ce service dépend
# de `src/trajectoire_emploi/` (partagé avec le notebook) et de `models/`
# (artefact produit par le notebook) qui vivent hors de `services/backend/`
#
# Build :   docker build -f services/backend/Dockerfile -t trajectoire-emploi-api:v0.1.0-api .
# Run :     docker run -d -p 8000:8000 --name trajectoire-emploi-api trajectoire-emploi-api:v0.1.0-api

FROM python:3.11-slim

# 1. Utilisateur non-root
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# 1bis. libgomp1 : requis par LightGBM au runtime (OpenMP), absent de
# l'image python:3.11-slim par défaut — sans lui, le chargement du
# pipeline packagé échoue au démarrage
RUN apt-get update && \
    apt-get install --no-install-recommends -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/appuser/app

# 2. Dépendances d'abord
COPY --chown=appuser:appuser services/backend/requirements.txt ./services/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r services/backend/requirements.txt

# 3. Code applicatif — src/ (partagé) et models/ (artefact) copiés depuis la
# racine du contexte de build
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/trajectoire_emploi_v1.joblib models/trajectoire_emploi_v1.json ./models/
COPY --chown=appuser:appuser services/backend/app/ ./services/backend/app/

# 4. Permissions
USER appuser

# 5. Port exposé (documentaire — `-p` reste obligatoire au `docker run`)
EXPOSE 8000

# 6. Healthcheck — start_period généreux pour laisser le modèle se charger
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 7. Démarrage — --app-dir résout `app.main:app` depuis services/backend/
# sans transformer services/ en package Python (même mécanisme qu'en local)
CMD ["uvicorn", "app.main:app", "--app-dir", "services/backend", "--host", "0.0.0.0", "--port", "8000"]
