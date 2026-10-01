# Expériences — Orientation et tri des demandeurs d'emploi

2 à 5 runs documentés,pas une liste exhaustive de toutes les combinaisons testées 
(mais seuls les runs qui ont pesé sur une décision sont retracés ici).

**Dataset** : `data/dataset_trajectoire_emploi_Sujet.csv`
(sha256 `2b9cb8c813b5f97540f0130b5e1cae2bad7ce5cbc1f57ffce51f54b498a7b34e`,
n=2500).

**Point de comparabilité à noter** : `exp_001`/`exp_002` proviennent du
benchmark large de l'Étape 5 (`RepeatedStratifiedKFold`, 5 splits × 3
répétitions, sur `X_train`/`y_train`, n=2000) ; `exp_003`/`exp_004`
proviennent de l'arbitrage restreint à 3 candidats de l'Étape 6
(`StratifiedKFold`, 5 splits, un seul passage, mêmes `X_train`/`y_train`).
Ce sont deux validations croisées différentes (splits non identiques) —
comparables entre elles par paires (5.1 vs 5.1, 6.2 vs 6.2), mais pas
d'un groupe à l'autre. Les deux partagent le même `X_train`/`y_train`.

---

## exp_001 — Baseline (DummyClassifier)

- **Notebook** : §5.1 (`resultats_benchmark`, ligne `scenario=S1,
  modele=baseline` — identique sur S1-S4, la baseline n'utilise aucune
  feature)
- **Modèle** : `DummyClassifier(strategy="stratified", random_state=42)`
- **Split** : `X_train`/`y_train` (n=2000), `RepeatedStratifiedKFold(n_splits=5,
  n_repeats=3, random_state=42)`
- **Métriques (test interne, moyenne CV)** :
  - F1 macro : 0,341
  - Recall classe 2 : 0,159
  - Taux erreur 2→0 : 0,412
  - Coût moyen : 1,675
- **Temps d'entraînement** : 0,025 s
- **Verdict** : écarté — sert de plancher de référence, ne capte aucun signal.

## exp_002 — S1-LightGBM, décision argmax

- **Notebook** : §5.1 (`resultats_benchmark`, ligne `scenario=S1,
  modele=lightgbm`)
- **Modèle** : `LGBMClassifier(random_state=42, verbosity=-1)` sur S1
  (toutes variables, y compris `nationalite_hors_ue`)
- **Split** : identique à exp_001
- **Hyperparamètres** : tous par défaut (`random_state=42` uniquement)
- **Métriques (test interne, moyenne CV)** :
  - F1 macro : 0,695 (meilleur score brut avec S1-XGBoost, ex æquo)
  - Recall classe 2 : 0,581
  - Taux erreur 2→0 : 0,122
  - Coût moyen : 0,678
- **Temps d'entraînement** : 0,193 s — **Inférence** : 18,9 ms/1000
- **Verdict** : écarté — meilleur F1 macro absolu, mais l'audit d'équité
  montre que le modèle **amplifie** le biais historique sur
  les prédictions (DI sous décision à coût minimal : 0,167).

## exp_003 — S1-LightGBM, décision à coût minimal

- **Notebook** : §6.2 (`tableau_decision`, ligne `candidat=S1-lightgbm,
  regle=cout minimal`)
- **Modèle** : identique à exp_002, seule la règle de décision change
  (`decision_cout_minimal` au lieu de l'argmax)
- **Split** : `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
  sur `X_train`/`y_train` (même train qu'exp_001/002, CV différente —
  cf. point de comparabilité en tête de fichier)
- **Métriques (test interne, moyenne CV)** :
  - F1 macro : 0,688 (−0,013 vs argmax)
  - Recall classe 2 : 0,702
  - Taux erreur 2→0 : 0,050 (÷2,3 vs argmax)
  - Coût moyen : 0,534
  - **Par groupe (`nationalite_hors_ue`)** : le groupe hors-UE est *mieux*
    protégé que le groupe majoritaire sur ce candidat (recall 0,804 vs
    0,667 ; taux 2→0 0,011 vs 0,063 — détail §6.4)
- **Verdict** : écarté — la décision à coût minimal règle le problème de
  l'erreur critique, mais l'usage direct de `nationalite_hors_ue` comme
  feature reste un risque légal indépendant de cet effet mesuré (RGPD /
  non-discrimination).

## exp_004 — S2-LightGBM, décision à coût minimal + abstention 0,7 (RETENU)

- **Notebook** : §6.2/§6.6/§7.2 (`tableau_decision` ligne `S2-lightgbm,
  cout minimal` + seuil d'abstention appliqué en §7.2)
- **Modèle** : `LGBMClassifier(random_state=42, verbosity=-1)` sur S2
  (S1 privé de `nationalite_hors_ue`)
- **Split** : identique à exp_003
- **Métriques (test interne, moyenne CV, cas gardés hors abstention)** :
  - F1 macro : 0,720 — Recall classe 2 : 0,719 — Kappa pondéré : 0,645
  - Taux erreur 2→0 : 0,056 — Coût moyen : 0,490
  - Taux de revue humaine : 18,7 %
  - DI(classe 0 favorable, coût minimal) : **0,722**
  - DI(classe 2 favorable = accompagnement, coût minimal) : **0,825**
  - **Par groupe** : recall classe 2 majoritaire 0,704 / hors-UE **0,620** ;
    taux 2→0 majoritaire 0,041 / hors-UE **0,076** (7× celui d'exp_003
    pour ce même groupe — détail §6.4)
- **Verdict** : **retenu** — retenu pour le risque légal de l'usage direct d'une variable protégée,
  **pas** parce que le DI « prouverait » une correction — le coût réel pour le groupe hors-UE
  (taux 2→0 multiplié par 7 vs exp_003) doit être activement surveillé en production.

## exp_005 — S2-LightGBM, réentraîné train+test combinés (modèle de production)

- **Notebook** : §8.1-8.2 ; holdout = §7.4 (test scellé, exécuté une seule
  fois, **avant** ce réentraînement)
- **Modèle** : identique à exp_004 (même architecture, mêmes
  hyperparamètres), réentraîné sur `X`/`y` complet (n=2500, train+test
  combinés) — action irréversible, décidée séparément après le verdict du
  test scellé
- **Métriques (holdout, test scellé, exécuté sur le modèle
  d'exp_004 avant la fusion train+test — jamais recalculé depuis)** :
  - F1 macro : 0,722 — Recall classe 2 : 0,703 — Kappa pondéré : 0,688
  - Taux erreur 2→0 : **0,027** — Coût moyen : 0,441
  - Taux de revue humaine (test) : 17,0 %
  - DI(classe 0 favorable, argmax) : 0,808
  - **Par groupe (test scellé, n hors-UE=56, échantillon réduit)** :
    recall classe 2 majoritaire 0,707 / hors-UE 0,688 ; taux d'abstention
    majoritaire 15,5 % / **hors-UE 28,6 %** (détail §7.5)
- **Package** : `models/trajectoire_emploi_v1.joblib` +
  `models/trajectoire_emploi_v1.json` (métadonnées, 5 clés obligatoires)
- **Verdict** : **retenu** — modèle servi par l'API (`services/backend/`).
  Aucun holdout propre ne subsiste après cette fusion train+test ; toute
  évaluation future de ce modèle exact se fera en production pas par un nouveau split.
