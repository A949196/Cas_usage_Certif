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
- [ ] **Étape 5 — Benchmark** : comparaison des modèles, mêmes folds.
- [ ] **Étape 6 — Arbitrage** : décision à coût minimal, audit d'équité.
- [ ] **Étape 7 — Communication** : note de recommandation client.
- [ ] **Partie B (Industrialisation)** : non démarrée.

**Décisions encore ouvertes** (voir `notebook/use_case.ipynb`, §1.5) :
- Base légale d'usage de `nationalite_hors_ue` pour l'audit d'équité.
- Définition retenue de l'« outcome positif » pour le disparate impact.

## Structure du dépôt

```
data/                        CSV du sujet (non versionné)
notebook/
  use_case.ipynb              notebook de travail (cadrage → exploration → …)
  journal-de-bord.ipynb       journal de bord (jour par jour)
src/trajectoire_emploi/       code réutilisable (créé au fil du besoin,
                               pas de structure anticipée) — actuellement :
                               fairness.py (disparate impact),
                               features.py (extraction département),
                               pipeline.py (préprocesseur par scénario)
tests/                        tests pytest sur données synthétiques
pyproject.toml                config pytest (pythonpath src/)
requirements.txt              dépendances Python (3.11+)
```

## Installation

```bash
uv venv .venv --python 3.11
uv pip install -r requirements.txt --python .venv/bin/python
```

## Tests

```bash
.venv/bin/python -m pytest
```
