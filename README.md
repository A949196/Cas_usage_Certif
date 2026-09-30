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

Texte intégral du sujet : `Sujet.docx` (hors dépôt).

## Matrice de coûts — pourquoi une métrique ne suffit pas

Le F1 macro (ou l'accuracy) traite toutes les erreurs de la même façon.
Le sujet impose l'inverse : une erreur **2 → 0** (usager à risque classé
« retour rapide ») est bien plus grave qu'une erreur adjacente (0 → 1 ou
1 → 2), car elle prive l'usager d'un accompagnement renforcé nécessaire.

La matrice de coûts (posée en hypothèse au §1.4 du notebook, implémentée
dans `src/trajectoire_emploi/evaluation.py::cout_moyen`) pondère chaque
type d'erreur selon sa gravité réelle (2 → 0 pèse 10× plus qu'une erreur
mineure). Elle sert à :

- **départager des modèles à F1 macro identique** (cf. benchmark Étape 5 :
  LightGBM et XGBoost ont le même F1 macro sur S1, mais pas le même coût
  moyen — ils ne commettent pas les mêmes erreurs) ;
- **révéler des compromis invisibles à l'accuracy** (un scénario peut avoir
  un bon recall sur la classe à risque tout en ayant un coût moyen plus
  élevé — cas mesuré du scénario texte seul, §5.2 du notebook) ;
- **guider l'ajustement d'un seuil de décision** à l'Étape 6, comme demandé
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

**Partie A** (modélisation) en cours. **Partie B** (industrialisation) non
démarrée — débute uniquement sur décision explicite, une fois la Partie A
validée et le test scellé exécuté.

## État d'avancement

- [x] **Étape 0 — Inventaire** : dépôt création.
- [x] **Étape 1 — Revue du cadrage** : §1 du notebook confronté à 9 fiches
      de méthodologie (entretien client, cartographie source, risques RGPD
      de croisement, datasheet Gebru, grille de décision, traçage
      d'expériences).
- [x] **Étape 2 — Exploration** : audit qualité, vérification des
      incohérences (74 lignes, code ROME `W1401` non conforme à la
      nomenclature), absence de PII dans le texte libre, premier
      disparate impact (`nationalite_hors_ue`, DI ≈ 0,342 — signal
      majeur), brouillon de datasheet et dictionnaire de variables,
      4 visualisations (distribution cible, âge, boxplot ancienneté,
      heatmap cible × nationalité).
- [x] **Étape 3 — Qualité et biais** : crosstab cible × nationalité sur
      les 3 classes ; décisions tranchées sur les 74 lignes incohérentes
      (valeur neutralisée + flag `anciennete_incoherente`) et sur le code
      `W1401` (conservé, effectif au seuil de fiabilité) ; mise à l'échelle
      actée pour l'Étape 4 ; bilan éthique n°2 en brouillon.
- [x] **Étape 4 — Préparation** : `features.py` (extraction département,
      hors ontologie) et `pipeline.py` (4 `ColumnTransformer`, un par
      scénario) créés avec 11 tests ; split stratifié 2000/500 ; S4 relu
      strictement contre le sujet et corrigé (plus restreint que « S1 sans
      texte » : ni `code_rome_vise`, ni `est_allocataire`, ni
      `nationalite_hors_ue`).
- [x] **Étape 5 — Benchmark** : `evaluation.py` (métriques métier) et
      `benchmark.py` (boucle scénarios × modèles, mêmes folds) créés avec
      12 tests ; 4 scénarios × 6 modèles comparés en validation croisée
      répétée. Meilleur compromis provisoire : **LightGBM/XGBoost sur
      S1** (F1 macro ≈ 0,695). Découverte : sur S3 (texte seul), tous les
      modèles convergent vers un score identique (seulement 10 vecteurs
      TF-IDF distincts) — confirme le risque de contamination du texte
      noté au cadrage. Test de robustesse (OOD) : extrapolation
      silencieuse détectée sur un âge hors plage d'entraînement.
- [x] **Étape 6 — Arbitrage** : `decision.py` (décision à coût minimal) et
      `calibration.py` (ECE, reliability diagram) créés avec 6 tests.
      Calibration bonne (ECE ≈ 0,04-0,05). Découverte majeure : l'audit
      d'équité refait sur les **prédictions** (pas l'étiquette) montre que
      le modèle **amplifie** le biais sur S1 (DI ≈ 0,26-0,28 contre 0,342
      sur l'étiquette), mais **repasse au-dessus du seuil d'alerte 4/5**
      sur S2 (DI = 0,838) en retirant seulement `nationalite_hors_ue`.
      **Choix final retenu : S2-LightGBM avec décision à coût minimal**
      (divise par plus de deux le taux d'erreur critique 2→0 pour un coût
      modeste en F1 macro).
- [x] **Étape 7 — Communication** : analyse d'erreurs (bloc principal = 
      confusion 0↔1, pas 2→0), seuil d'abstention à 0,7 (~19 % de revue
      humaine), note de recommandation client rédigée. **Verdict final sur
      le test scellé (exécuté une seule fois)** : F1 macro = 0,722, taux
      d'erreur critique 2→0 = **0,027**, DI sur les prédictions = **0,808**
      (toujours au-dessus du seuil d'alerte) — confirme sur données jamais
      vues les propriétés mesurées en validation croisée. **Partie A
      (Modélisation) close.**
- [ ] **Partie B (Industrialisation)** : pas encore démarrée techniquement ;
      découpage validé en 4 lots (Lot 1 : API/Docker/CI-CD — Lot 2 :
      MLflow/Monitoring — Lot 3 : Interface conseiller/Réentraînement —
      Lot 4 : Architecture/Dossier). Prochaine étape : Étape 8 (API).
  - [x] **Étape 8 — API** : `features.py` enrichi de
        `nettoyer_anciennete_incoherente` (refactor, code désormais partagé
        entre notebook et API — évite la dérive train/serve) ;
        `persistence.py` créé (packaging `.joblib` + métadonnées JSON,
        5 clés obligatoires, 4 tests) ; pipeline final **réentraîné sur
        train+test combinés** (2500 lignes, même architecture S2-LightGBM),
        packagé dans `models/trajectoire_emploi_v1.joblib`/`.json`
        (`metrics_holdout` = verdict du test scellé §7.4, jamais recalculé) ;
        API FastAPI (`services/backend/app/`) avec `/health`, `/info`,
        `/predict` (décision à coût minimal avec abstention, seuil 0,7) ;
        logs structurés sans PII (`request_id`) ; 10 tests API + contract
        test du modèle. **Réorganisation `services/backend` /
        `services/frontend`** anticipant la dockerisation (Étape 9) :
        `src/` et `models/` restent partagés à la racine (utilisés par le
        notebook ET le backend), `services/backend/requirements.txt`
        allégé pour une image de prod plus légère (pas de jupyter/xgboost),
        `services/frontend/` réservé au Lot 3 (interface conseiller).

**Décisions encore ouvertes** (voir `notebook/use_case.ipynb`, §1.5) :
- Base légale d'usage de `nationalite_hors_ue` pour l'audit d'équité.
- Valeur définitive du coût de l'erreur 2→0 dans la matrice de coûts
  (décision D3 — sensibilité testée en Étape 6, jamais formellement
  validée avec le métier).

**Point de vigilance** : `outputs/split_scelle.json` (trace de l'unique
exécution du test scellé) n'est actuellement pas exclu par `.gitignore` —
à vérifier avant un commit si vous souhaitez qu'il reste local uniquement.

## Structure du dépôt

```
data/                          CSV du sujet (non versionné)
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
                                 persistence.py (packaging modèle .joblib+.json)
models/                         modèle packagé (.joblib non versionné, .json
                                 versionné) — artefact partagé, produit par le
                                 notebook, consommé par services/backend/
services/
  backend/
    app/                         API FastAPI (main.py, schemas.py, middleware.py)
    requirements.txt              dépendances runtime allégées (pas de jupyter/
                                   xgboost/pytest, image < 1 Go)
  frontend/
    README.md                    réservé au Lot 3 (interface conseiller Streamlit)
tests/                          tests pytest (unitaires + contract test + API),
                                 centralisés (teste src/ ET services/backend/app)
.dockerignore                   exclusions du contexte de build Docker (racine)
pyproject.toml                  config pytest (pythonpath src/ + services/backend)
requirements.txt                dépendances Python dev complet (3.11+) : notebook,
                                 tests, ET service — pour l'environnement local
```

**Pourquoi `src/` et `models/` restent à la racine** (pas sous
`services/backend/`) : ce sont des artefacts **partagés** avec le notebook
d'entraînement (Partie A). Les déplacer sous `services/backend/`
suggérerait à tort qu'ils appartiennent exclusivement à l'API. Le
`Dockerfile` du backend (Étape 9) les copiera explicitement depuis la
racine (contexte de build = racine du dépôt).

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
