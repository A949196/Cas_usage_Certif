# Frontend — Interface conseiller (réservé, Lot 3)

Ce dossier est **réservé** pour l'interface conseiller.
Il est vide intentionnellement — pas de code, pas de
`Dockerfile` avant le démarrage effectif du Lot 3.

## Ce qui est prévu ici

Une UI **Streamlit** qui consomme l'API du backend via HTTP (client `httpx`) :

- **Aucune dépendance** à `src/trajectoire_emploi/` ni à `models/` — ce
  service est un pur client HTTP, contrairement à `services/backend/`.
- Communication avec le backend via le nom de service Docker (`http://backend:8000`),
  jamais `localhost` (cf. fiche `011_DockerCompose_essentiel.md`).
- Formulaire de saisie des champs `DemandeurInput` (cf.
  `services/backend/app/schemas.py`), affichage de la décision
  (`retour_rapide` / `retour_moyen` / `risque_longue_duree` /
  `revue_humaine`) et des probabilités.

## Structure anticipée (au démarrage du Lot 3)

```
services/frontend/
├── app.py             # script Streamlit principal
├── requirements.txt   # streamlit, httpx — rien d'autre
└── Dockerfile
```
