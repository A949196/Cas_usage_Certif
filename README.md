# Orientation et tri multimodal des demandeurs d'emploi

Projet de certification : classification multiclasse ordonnée du délai de
retour à l'emploi (0 = rapide < 6 mois, 1 = moyen 6-12 mois, 2 = risque de
longue durée > 12 mois), à partir de données hybrides (tabulaires + texte
libre) issues d'une agence nationale de l'emploi.

## Sujet

Concevoir un système d'aide à la décision (pas de décision automatique)
pour repérer, à l'issue du premier entretien, les usagers à risque de
chômage de longue durée. Contraintes imposées par le sujet :
- Entraîner et comparer plusieurs modèles (Random Forest, LightGBM,
  XGBoost…) avec validation croisée stratifiée.
- Nettoyer/imputer les données, extraire le département depuis le code
  INSEE (feature engineering haute cardinalité).
- Réduire spécifiquement les erreurs critiques **2 → 0** (usager à risque
  classé « retour rapide »), plus graves qu'une erreur adjacente.
- Comparer 4 scénarios : multimodal complet (S1), sans variables sensibles
  (S2), texte seul (S3), tabulaire seul (S4).
- Respecter le cadre RGPD / non-discrimination / responsabilité juridique.
- Concevoir, en seconde partie, une architecture d'industrialisation
  (API, Docker, CI/CD, MLflow, monitoring) intégrée au SI de l'agence.

## Matrice de coûts — pourquoi une métrique ne suffit pas

Le F1 macro (ou l'accuracy) traite toutes les erreurs de la même façon.
Le sujet impose l'inverse : une erreur **2 → 0** (usager à risque classé
« retour rapide ») est bien plus grave qu'une erreur adjacente (1 → 0 ou
2 → 1), car elle prive l'usager d'un accompagnement renforcé nécessaire.

La matrice de coûts (implémentée dans `src/trajectoire_emploi/evaluation.py::cout_moyen`) pondère chaque
type d'erreur selon sa gravité réelle (2 → 0 pèse 10× plus qu'une erreur
mineure). Elle sert à :

- **départager des modèles à F1 macro identique** :
  LightGBM et XGBoost ont le même F1 macro sur S1, mais pas le même coût
  moyen — ils ne commettent pas les mêmes erreurs ;
- **révéler des compromis invisibles à l'accuracy** : un scénario peut avoir
  un bon recall sur la classe à risque tout en ayant un coût moyen plus
  élevé ;
- **guider l'ajustement d'un seuil de décision** : comme demandé
  explicitement par le sujet (« favoriser la diminution des erreurs
  critiques ») ;
- **traduire un score technique en langage métier** : « coût moyen par
  usager » est plus parlant pour un décideur qu'un F1 macro.

Limite assumée : les valeurs de la matrice (0, 1, 2, 10, 3, 0) sont une
**hypothèse de travail arbitraire** — seule la structure (2→0 très
supérieur au reste) compte. La sensibilité à ces valeurs est testée à
l'Étape 6 (décision D3, jamais formellement validée avec le métier).

## Axes de développement (feuille de route en 8 phases)

| Phase | Contenu | Partie |
|---|---|---|
| 1. Cadrer | besoin, tâche ML, risques éthiques | A |
| 2. Explorer | EDA, qualité, biais | A |
| 3. Préparer | pipeline sans fuite, scénarios | A |
| 4. Modéliser | benchmark multi-modèles | A |
| 5. Arbitrer | verdict multicritères, équité, coûts | A |
| 6. Industrialiser | API, Docker, CI/CD, MLflow | B |
| 7. Surveiller | monitoring, dérive, réentraînement | B |
| 8. Architecturer & défendre | archi cible, dossier, soutenance | B |

## État d'avancement

- [x] **Étape 0 — Inventaire** : dépôt création.
- [x] **Étape 1 — Revue du cadrage** : §1 du notebook confronté à 9 fiches
      de méthodologie (entretien client, cartographie source, risques RGPD
      de croisement, datasheet Gebru, grille de décision, traçage
      d'expériences).
- [x] **Étape 2 — Exploration** : audit qualité, vérification des
      incohérences (74 lignes, code ROME `W1401` non conforme à la
      nomenclature), absence de PII dans le texte libre, premier
      disparate impact, datasheet Gebru (7 sections) et dictionnaire de variables,
      4 visualisations (distribution cible, âge, boxplot ancienneté,
      heatmap cible × nationalité). **`experiments.md`** : 5 runs
      documentés, de la baseline au modèle de production.
- [x] **Étape 3 — Qualité et biais** : crosstab cible × nationalité sur
      les 3 classes ; décisions tranchées sur les 74 lignes incohérentes
      (valeur neutralisée + flag `anciennete_incoherente`) et sur le code
      `W1401` (conservé, effectif au seuil de fiabilité) ; bilan éthique n°2.
- [x] **Étape 4 — Préparation** : `features.py` (extraction département) 
      et `pipeline.py` (4 `ColumnTransformer`, un par scénario) créés avec 11 tests ; 
      split stratifié 2000/500.
- [x] **Étape 5 — Benchmark** : `evaluation.py` (métriques métier) et
      `benchmark.py` (boucle scénarios × modèles, mêmes folds) créés avec
      12 tests ; 4 scénarios × 6 modèles comparés en validation croisée
      répétée. Meilleur compromis provisoire : **LightGBM/XGBoost sur
      S1** (F1 macro ≈ 0,695). Découverte : sur S3 (texte seul), tous les
      modèles convergent vers un score identique — confirme le risque de contamination du texte. 
      Test de robustesse (OOD) : extrapolation silencieuse détectée sur un âge hors plage d'entraînement.
- [x] **Étape 6 — Arbitrage** : `decision.py` (décision à coût minimal) et `calibration.py` (ECE
      reliability diagram) créés avec 6 tests. Calibration bonne (ECE ≈ 0,044-0,049). 
      **Audit d'équité entièrement refait par groupe** (`nationalite_hors_ue`), sous la
      décision **réellement déployée** (coût minimal, pas argmax) : sous S1, le groupe hors-UE
      est **mieux protégé** contre l'erreur critique 2→0 que le groupe
      majoritaire (recall classe 2 = 0,804 vs 0,667 ; taux 2→0 = 0,011 vs
      0,063). 
      Sous S2 (sans cette variable), ce renversement s'inverse : recall du groupe hors-UE 
      tombe à 0,620 (sous le groupe majoritaire) et son taux 2→0 monte à **0,076, 7 fois
      supérieur à S1**. Le DI agrégé sous la bonne règle (0,722, pas 0,838).
      **Choix final maintenu :S2-LightGBM avec décision à coût minimal**, mais l'argumentation
      est révisée : retenu pour le risque légal de l'usage direct d'une
      variable protégée.
- [x] **Étape 7 — Communication** : analyse d'erreurs (bloc principal = 
      confusion 0↔1, pas 2→0), seuil d'abstention à 0,7 (~19 % de revue
      humaine), note de recommandation client rédigée. **Verdict final sur
      le test scellé (exécuté une seule fois)** : F1 macro = 0,722, taux
      d'erreur critique 2→0 = **0,027**, DI sur les prédictions = **0,808**
      — confirme sur données jamais vues les propriétés mesurées en validation croisée.
- [x] **Étape 8 — API** : `features.py` enrichi de `nettoyer_anciennete_incoherente` ;
      `persistence.py` créé (packaging `.joblib` + métadonnées JSON,
      5 clés obligatoires, 4 tests) ; pipeline final **réentraîné sur
      train+test combinés** (2500 lignes, même architecture S2-LightGBM),
      packagé dans `models/trajectoire_emploi_v1.joblib`/`.json`
      (`metrics_holdout` = verdict du test scellé, jamais recalculé) ;
      API FastAPI (`services/backend/app/`) avec `/health`, `/info`,
      `/predict` (décision à coût minimal avec abstention, seuil 0,7) ;
      logs structurés sans PII (`request_id`) ; 10 tests API + contract
      test du modèle. 
- [x] **Étape 9 — Docker** : `services/backend/Dockerfile` (base
      `python:3.11-slim`, user non-root, `libgomp1` ajouté — requis par
      LightGBM au runtime, absent de l'image slim par défaut, détecté en
      testant réellement le conteneur) ; `docker-compose.yml` (service
      `backend`, healthcheck). **Modèle packagé versionné dans Git**
      (`models/trajectoire_emploi_v1.joblib`, 404 Ko — exception ciblée
      au `.gitignore`, nécessaire pour que Docker/CI fonctionnent sans
      dépendre d'un run notebook complet). Image testée de bout en bout,
      `docker exec whoami` → `appuser`,`docker compose ps` → `healthy`, 
      `/health` `/info` `/predict`(200 et 422) vérifiés sur le conteneur réel.
- [x] **Étape 10 — CI/CD** : `.github/workflows/ci.yml` — job `test`
      (pytest sur la suite complète) → job `build` (`needs: test`,
      `docker compose build` + vérification `/health` sur le conteneur
      démarré). Portée volontairement limitée au gate test→build (pas de
      push GHCR, ce dépôt n'a pas encore de remote GitHub actif) —
- [x] **Étape 11 — MLflow** : `scripts/log_experiments_mlflow.py`
      reloggue les 5 runs d'`experiments.md` vers MLflow (params +
      métriques + tag verdict), capitalisant sur le traçage existant sans
      le remplacer. `mlruns/` ajouté au `.gitignore` (artefact local).
      Vérifié : serveur `mlflow ui` réellement lancé, API interrogée —
      5 runs comparables dans l'expérience `trajectoire_emploi`.
- [x] **Étape 13.1 — Instrumentation Prometheus** :
      `prometheus-fastapi-instrumentator` sur `services/backend/app/main.py`
      — `/metrics` (métriques HTTP auto : latence, volume, codes retour) ; `Counter`
      métier `trajectoire_emploi_decisions_total` labellé par décision
      (4 valeurs bornées, jamais de `request_id` en label). 3 tests
      ajoutés. Vérifié réellement : `/metrics` interrogé dans le vrai
      conteneur Docker après un `/predict`, compteur incrémenté, healthcheck
      toujours `healthy`.
- [x] **Étape 13.2 — Stack Prometheus + Grafana** : `observability/`
      (config Prometheus + provisioning Grafana datasource/dashboards) ; 
      `docker-compose.yml` étendu (3 services). 
      Testé réellement, stack complète lancée : les 3 services démarrent
      dans l'ordre et passent `healthy` ; Prometheus scrape `backend:8000`
      ; Grafana interroge Prometheus via la datasource provisionnée ; 
      Test de la panne muette : `docker compose stop backend` → `up` bascule
      à `0` en ~15s, confirmant que c'est le seul signal fiable de panne.
- [x] **Étape 13.3 — Dashboard Grafana** :
      `trajectoire_emploi_prod.json`— 3 panels provisionnés : Vie (`up{job="backend"}`),
      Vitesse (p95 par route via `histogram_quantile` sur
      `http_request_duration_seconds_bucket`), Comportement
      (répartition des décisions prédites, `rate(...)` par label). Vérifié
      réellement : dashboard présent dans `GET /api/search` sans import
      manuel ; trafic généré sur l'API réelle.
- [x] **Étape 13.4 — Dérive (PSI/KS/Chi²) en notebook** :
      `src/trajectoire_emploi/drift.py` (PSI, KS, Chi², diagnostic data vs
      concept drift) créé avec 11 tests. Démonstration
      train vs test comme proxy (pas de vrai trafic de production à
      ce stade, limite assumée explicitement) — aucun signal de dérive
      détecté (attendu sur un split stratifié propre), AUC stable
      (0,838 → 0,856). Test complémentaire : `generer_trafic_test.py`
      + `comparer_derive_production.py` — envoient un trafic
      synthétique à l'API, puis comparent à `data/reference_set.csv`. 
      Résultat obtenu : signal fort sur `anciennete_poste_ans` (PSI≈1,2-2,4) et `code_rome_vise`
      (Chi² p≈0). Utile pour vérifier que le mécanisme réagit, pas pour diagnostiquer une
      dérive réelle.
- [x] **Étape 13.5 — Runbook + évaluation continue** :
      `RUNBOOK.md` (4 procédures : Service KO, Latence dégradée, Métrique
      modèle dégradée, Rollback) ; `scripts/evaluate_model.py` : garde-fou CI sur
      seuils bloquants. `data/reference_set.csv` (= ancien `X_test`,
      figé, réutilisation explicitement actée) + `data/reference_baseline.json`
      (golden run gelé). Limite assumée et documentée dans le script :
      le modèle de production a été réentraîné sur train+test combinés
      — il a donc déjà vu ce jeu, les métriques du golden run sont
      artificiellement hautes (F1 macro ≈ 0,94) ; ce script détecte des
      régressions de code. Seuils de tolérance justifiés par la variance mesurée : 
      `f1_macro` par bootstrap (500 ré-échantillonnages,
      sigma ≈ 0,012, tolérance 0,05 > 2σ) ; `taux_erreur_2_vers_0` par
      l'écart-type inter-folds de l'Étape 5 :
      sigma ≈ 0,043 sur 15 folds, tolérance 0,09 > 2σ. 6 tests unitaires sur la
      logique de seuils. Job CI `evaluate-model`
      (`needs: test`) branché dans `.github/workflows/ci.yml`.
- [x] **Étape 12 — Boucle de feedback (interface conseiller)** :
      prérequis non couvert par les fiches : store SQLite `predictions`
      (journalise chaque décision rendue par `/predict`, sans PII ni
      features) — nécessaire pour valider les `request_id` entrants côté
      feedback. `POST /feedback` (adapté multiclasse 0/1/2) :
      404 `request_id` inconnu, 422 label hors bornes (Pydantic),
      201 nouveau/idempotent, 409 contradiction (jamais d'écrasement
      silencieux). 
      Stockage SQLite (préféré au CSV versionné pour l'intégrité en écriture concurrente) : table `feedbacks`(`request_id` PK, `true_label`, `comments`, `created_at`,
      `used_for_training` défaut 0 — pilotera un futur trigger de réentraînement, jamais le total). 
      Interface conseiller Streamlit : `services/frontend/app.py`, 2 onglets (scorer un
      dossier / remonter un feedback), 4ᵉ service Docker. 
      Correction de bord : le `request_id` du body `/predict` était généré
      indépendamment de celui posé en header par le middleware (M5-B1) —
      désormais unifié (un seul id pour logs, réponse et feedback).
      Vérifié réellement : stack 4 services `healthy`, les 5 cas testés
      sur le vrai conteneur (201 nouveau, 201 idempotent, 404, 409, 422),
      fichier `feedback.db` persisté côté hôte (`data/runtime/`, non
      versionné), UI Streamlit répond (`/_stcore/health` → ok). 
      16 tests unitaires ajoutés (store + API).

**Décisions encore ouvertes** :
- Base légale d'usage de `nationalite_hors_ue` pour l'audit d'équité.
- Valeur définitive du coût de l'erreur 2→0 dans la matrice de coûts
  (décision D3 — sensibilité testée en Étape 6, jamais formellement
  validée avec le métier).
- **D6** : le coût mesuré de S2 pour le groupe hors-UE (taux
  2→0 multiplié par 7 vs S1) est-il acceptable en
  production tel quel, ou faut-il explorer un seuil de décision différencié
  par groupe (piste non implémentée) ? Question posée au métier/DPO, en
  lien avec D5.
- **D7** : durée de rétention des `predictions`/`feedbacks`
  stockés (*« décider combien de temps on garde les
  feedbacks, et pourquoi »*) — non encore tranchée, à documenter avant
  mise en production réelle.

## Structure du dépôt

```
data/                          CSV du sujet (non versionné)
experiments.md                 traçage des runs de modèle (5 runs, baseline → production)
notebook/
  use_case.ipynb                notebook de travail (cadrage → exploration → …)
  journal-de-bord.ipynb         journal de bord (jour par jour)
src/trajectoire_emploi/         code réutilisable, PARTAGÉ entre le notebook
                                 (entraînement) et services/backend/ (API) —
                                 reste à la racine, n'appartient à aucun
                                 service Docker en particulier :
                                 fairness.py (disparate impact),
                                 features.py (extraction département,
                                   nettoyage ancienneté incohérente),
                                 pipeline.py (préprocesseur par scénario),
                                 evaluation.py (métriques métier),
                                 benchmark.py (comparaison scénarios × modèles),
                                 decision.py (décision à coût minimal),
                                 calibration.py (ECE, reliability diagram),
                                  persistence.py (packaging modèle .joblib+.json),
                                  drift.py (PSI, KS, Chi², diagnostic drift),
                                  feedback_store.py (store SQLite predictions/
                                    feedbacks conseillers)
models/                         modèle packagé — **`.joblib` ET `.json`
                                  versionnés** (exception ciblée au
                                  `.gitignore`, 404 Ko, nécessaire pour que
                                  Docker/CI fonctionnent sans dépendre d'un
                                  run notebook complet) — artefact produit
                                  par le notebook, consommé par
                                  services/backend/
services/
  backend/
    app/                         API FastAPI (main.py, schemas.py, middleware.py)
    requirements.txt              dépendances runtime allégées (pas de jupyter/
                                   xgboost/pytest, image < 1 Go)
    Dockerfile                    image du service (python:3.11-slim, user
                                   non-root, libgomp1 pour LightGBM, healthcheck)
  frontend/
    app.py                       interface conseiller Streamlit :
                                   scorer un dossier + remonter un feedback
    requirements.txt              streamlit + httpx uniquement
    Dockerfile                    image du service (python:3.11-slim, user non-root)
docker-compose.yml              orchestration locale (backend + frontend + prometheus + grafana)
observability/
  prometheus/prometheus.yml       config de scrape (cible : backend:8000/metrics)
  grafana/provisioning/
    datasources/datasource.yml      datasource Prometheus (uid fixe)
    dashboards/dashboards.yml       provider (charge tout JSON de ce dossier)
    dashboards/trajectoire_emploi_prod.json   dashboard (vie/vitesse/comportement)
.github/workflows/ci.yml        CI : pytest → evaluate-model → build Docker
scripts/
  log_experiments_mlflow.py       reloggue experiments.md vers MLflow
  evaluate_model.py                garde-fou CI sur seuils bloquants 
  generer_trafic_test.py            envoie un trafic synthetique a l'API reelle
  comparer_derive_production.py     compare ce trafic a data/reference_set.csv
RUNBOOK.md                       4 procédures d'astreinte
data/
  reference_set.csv                jeu de référence figé (= ancien X_test, versionné)
  reference_baseline.json          golden run gelé (métriques de référence)
  runtime/                        store SQLite predictions/feedbacks,
                                   jamais versionné (généré à l'exécution)
tests/                          tests pytest (unitaires + contract test + API),
                                 centralisés (tests src/ ET services/backend/app)
.dockerignore                   exclusions du contexte de build Docker (racine)
pyproject.toml                  config pytest (pythonpath src/ + services/backend + scripts)
requirements.txt                dépendances Python dev complet (3.11+) : notebook,
                                 tests, ET service — pour l'environnement local
```

## Docker

```bash
docker compose up --build        # backend + frontend + prometheus + grafana
docker compose ps                 # les 4 services doivent passer "healthy"
curl http://localhost:8000/health
curl http://localhost:8000/metrics   # métriques Prometheus (HTTP + métier)
curl http://localhost:9090/api/v1/targets   # vérifie que Prometheus scrape le backend
# Grafana : http://localhost:3001 (admin/admin), datasource Prometheus préconfigurée
# Interface conseiller : http://localhost:8501
docker compose down
```

## CI/CD

`.github/workflows/ci.yml` : job `test` (pytest) → jobs `evaluate-model`
(seuils bloquants) et `build` (Docker + `/health`), tous deux
`needs: test`. Portée limitée au gate — pas de push vers un registre.

```bash
# Reproduire le gate d'évaluation continue en local :
.venv/bin/python scripts/evaluate_model.py              # chemin vert attendu
.venv/bin/python scripts/evaluate_model.py --degrade     # chemin rouge (exit 1)
.venv/bin/python scripts/evaluate_model.py --freeze-baseline   # regeler le golden run
```

## Runbook

`RUNBOOK.md` : 4 procédures d'astreinte (Service KO, Latence dégradée,
Métrique modèle dégradée, Rollback), seuils reliés aux panels Grafana
réels.

## Tester la dérive sur un trafic envoyé à l'API réelle

```bash
docker compose up --build -d
.venv/bin/python scripts/generer_trafic_test.py --n 80
.venv/bin/python scripts/comparer_derive_production.py
```

Il s'agit d'un trafic synthétique (fabriqué par le script) à l'API
réellement déployée.

## MLflow (traçage des expériences)

```bash
.venv/bin/python scripts/log_experiments_mlflow.py   # reloggue experiments.md
.venv/bin/mlflow ui                                   # http://localhost:5000
```

Capitalise sur `experiments.md` (qui reste la source de vérité lisible
humainement) : les 5 mêmes runs sont reloggués vers MLflow pour la
comparaison outillée dans le temps (params, métriques, tag verdict).
`mlruns/` est local, non versionné.

## Installation

```bash
uv venv .venv --python 3.11
uv pip install -r requirements.txt --python .venv/bin/python
```

## Tests

```bash
.venv/bin/python -m pytest
```

## Lancer l'API (en local, hors Docker)

```bash
uv run --python .venv/bin/python uvicorn app.main:app --app-dir services/backend --reload
```

`--app-dir services/backend` indique à uvicorn où résoudre `app.main:app`
sans avoir à transformer `services/` en package Python. Documentation
interactive : `http://localhost:8000/docs`.
