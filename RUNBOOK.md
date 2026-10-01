# Runbook d'astreinte — API `trajectoire_emploi`

Fiche appliquée : `515_Runbook_astreinte_essentiel.md`. Document court,
orienté action — se lit en 30 secondes sous stress. Pas de la doc
exhaustive : pour chaque incident type, quoi regarder, quoi faire, qui
appeler, et surtout **ce qu'il ne faut PAS faire**.

Dashboard de référence : `observability/grafana/provisioning/dashboards/trajectoire_emploi_prod.json`
(panels Vie / Vitesse / Comportement, cf. README § Docker).

---

## 1. Service KO (le backend ne répond plus)

**Déclenchement** : panel « Vie » → `up{job="backend"}` passe à `0`, ou
`docker compose ps` montre `backend` en `Exited`/`unhealthy`.

**Actions** :
1. `docker compose logs --tail=100 backend` — lire l'erreur en premier,
   ne rien redémarrer à l'aveugle.
2. `docker compose restart backend`.
3. Si toujours KO après 2 tentatives : `docker compose ps` pour vérifier
   que `prometheus`/`grafana` sont eux-mêmes sains (dépendance
   `service_healthy`) ; si l'un d'eux est cassé, le backend ne redémarre
   jamais proprement.
4. Si KO persistant : escalade.

**Qui appeler** : le mainteneur du projet (cf. §2.5 section 7 de la
datasheet, `notebook/use_case.ipynb`).

**On NE fait PAS** : `docker compose down -v` (détruit les volumes —
aucun volume de données persistantes ici, mais réflexe à proscrire
systématiquement) ; redéployer une image non testée en urgence.

---

## 2. Latence dégradée (p95 anormalement haute)

**Déclenchement** : panel « Vitesse » — p95 sur `/predict` dépasse
**300 ms** pendant plus de 5 minutes (baseline mesurée : ≈ 95 ms en charge
légère, cf. §13.3 — une marge ×3 avant d'alerter).

**Actions** :
1. `docker stats` — vérifier que le conteneur `backend` n'est pas
   CPU-bound (modèle LightGBM : l'inférence est normalement < 30 ms/1000
   prédictions, cf. `experiments.md`).
2. Vérifier le panel « Comportement » : un pic de `taux_revue_humaine`
   anormal peut indiquer un afflux de requêtes ambiguës, pas un problème
   d'infra.
3. Vérifier qu'aucun déploiement récent n'a changé le modèle (comparer
   `/info` → `model_version` à la dernière release connue).

**Qui appeler** : mainteneur du projet.

**On NE fait PAS** : augmenter les ressources à l'aveugle sans avoir lu
les logs ; redéployer une version non testée pour « voir si ça va plus
vite ».

---

## 3. Métrique modèle dégradée (le garde-fou CI a bloqué une release)

**Déclenchement** : le job `evaluate-model` de la CI (`.github/workflows/ci.yml`)
sort en `exit 1` — une métrique a violé un seuil (`scripts/evaluate_model.py`,
fiche 517).

**Actions** :
1. Lire la sortie du job : elle liste explicitement quelle métrique a
   violé quel seuil (ex. `f1_macro a chuté de 0.94 à 0.40`).
2. **Ne pas forcer le merge.** Reproduire en local :
   `python scripts/evaluate_model.py` et comparer à
   `data/reference_baseline.json`.
3. Chercher ce qui a changé dans le code de service (`services/backend/app/`,
   `src/trajectoire_emploi/features.py`, `src/trajectoire_emploi/decision.py`)
   depuis le dernier golden run vert — **jamais** dans le `.joblib` lui-même
   (il n'a pas bougé si personne n'a relancé le packaging du notebook).
4. Corriger le code, relancer `pytest` puis `evaluate_model.py` en local
   avant de re-pousser.

**Qui appeler** : mainteneur du projet — ce garde-fou ne doit **jamais**
être contourné silencieusement (ex. augmenter les seuils pour faire
passer la CI sans comprendre la cause).

**On NE fait PAS** : relancer `--freeze-baseline` pour « faire passer » le
garde-fou sans avoir compris la régression — ça gèle une régression comme
nouveau golden run, le garde-fou devient aveugle pour toujours.

---

## 4. Rollback (revenir à la version stable précédente)

**Déclenchement** : n'importe lequel des 3 incidents ci-dessus, non résolu
après la procédure associée, ou dégradation confirmée en production.

**Actions** :
1. Identifier le dernier tag Docker sain : `docker images trajectoire-emploi-api`.
2. `docker compose down`, puis relancer avec l'image taguée précédente
   (`docker compose up -d` après avoir ajusté `image:` dans
   `docker-compose.yml` ou via `docker run` direct sur l'ancien tag).
3. Vérifier `/health` et `/info` → `model_version` correspond bien à la
   version attendue avant de considérer l'incident clos.

**Qui appeler** : mainteneur du projet.

**On NE fait PAS** : corriger en urgence sur l'image en cours d'exécution
(hotfix non testé en prod) — toujours préférer revenir à une version
**déjà validée** plutôt que de rustiner l'actuelle.
