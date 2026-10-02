# Frontend — Interface conseiller

UI **Streamlit** qui consomme l'API du backend via HTTP (client `httpx`) :

- **Aucune dépendance** à `src/trajectoire_emploi/` ni à `models/` — ce
  service est un pur client HTTP, contrairement à `services/backend/`.
- Communication avec le backend via le nom de service Docker (`http://backend:8000`),
  jamais `localhost` (cf. fiche `011_DockerCompose_essentiel.md`).
- Deux onglets :
  1. **Scorer un dossier** : formulaire `DemandeurInput`, affichage de la
     décision et des probabilités, `request_id` affiché pour la suite.
  2. **Remonter un feedback** : conseiller saisit le `request_id` + la vraie
     classe observée → `POST /feedback`. Gère les 3 cas métier (201 créé/idempotent,
     404 `request_id` inconnu, 409 contradiction à arbitrer humainement).

## Lancement

```bash
docker compose up --build -d
# UI sur http://localhost:8501
```

## Structure

```
services/frontend/
├── app.py             # script Streamlit principal
├── requirements.txt   # streamlit, httpx — rien d'autre
└── Dockerfile
```

