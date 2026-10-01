"""Reloggue les 5 runs déjà documentés dans `experiments.md` vers MLflow.

Ce script transcrit fidèlement les chiffres déjà validés et versionnés dans
`experiments.md` vers un format comparable dans le temps (MLflow), en plus du markdown existant —
`experiments.md` reste la source de vérité lisible humainement, MLflow
ajoute la capacité de comparaison outillée.

Exécution :
    .venv/bin/python scripts/log_experiments_mlflow.py
    .venv/bin/mlflow ui   # http://localhost:5000

Idempotent : chaque exécution crée de nouveaux runs (MLflow ne déduplique
pas) — supprimer `mlruns/` avant de relancer si on veut repartir propre.
"""

from __future__ import annotations

import mlflow

DATASET_SHA256 = "2b9cb8c813b5f97540f0130b5e1cae2bad7ce5cbc1f57ffce51f54b498a7b34e"

mlflow.set_experiment("trajectoire_emploi")

# ---------------------------------------------------------------------------
# exp_001 — Baseline (DummyClassifier)
# ---------------------------------------------------------------------------
with mlflow.start_run(run_name="exp_001_baseline"):
    mlflow.log_params(
        {
            "modele": "DummyClassifier(strategy=stratified)",
            "scenario": "S1",
            "dataset_sha256": DATASET_SHA256,
            "split": "train/test 2000/500, stratify=y, random_state=42",
            "cv": "RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)",
            "random_state": 42,
        }
    )
    mlflow.log_metrics(
        {
            "f1_macro": 0.341,
            "recall_classe_2": 0.159,
            "taux_erreur_2_vers_0": 0.412,
            "cout_moyen": 1.675,
            "temps_entrainement_s": 0.025,
        }
    )
    mlflow.set_tags(
        {
            "verdict": "ecarte",
            "raison": "plancher de reference, ne capte aucun signal",
            "notebook_section": "§5.1",
        }
    )

# ---------------------------------------------------------------------------
# exp_002 — S1-LightGBM, décision argmax
# ---------------------------------------------------------------------------
with mlflow.start_run(run_name="exp_002_s1_lightgbm_argmax"):
    mlflow.log_params(
        {
            "modele": "LGBMClassifier",
            "scenario": "S1",
            "dataset_sha256": DATASET_SHA256,
            "split": "train/test 2000/500, stratify=y, random_state=42",
            "cv": "RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)",
            "random_state": 42,
            "regle_decision": "argmax",
        }
    )
    mlflow.log_metrics(
        {
            "f1_macro": 0.695,
            "recall_classe_2": 0.581,
            "taux_erreur_2_vers_0": 0.122,
            "cout_moyen": 0.678,
            "temps_entrainement_s": 0.193,
            "inference_ms_1k": 18.948,
        }
    )
    mlflow.set_tags(
        {
            "verdict": "ecarte",
            "raison": "meilleur F1 macro brut, mais amplifie le biais historique (DI=0.167 sous cout minimal, cf. exp_003)",
            "notebook_section": "§5.1",
        }
    )

# ---------------------------------------------------------------------------
# exp_003 — S1-LightGBM, décision à coût minimal
# ---------------------------------------------------------------------------
with mlflow.start_run(run_name="exp_003_s1_lightgbm_cout_minimal"):
    mlflow.log_params(
        {
            "modele": "LGBMClassifier",
            "scenario": "S1",
            "dataset_sha256": DATASET_SHA256,
            "split": "train/test 2000/500, stratify=y, random_state=42",
            "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "random_state": 42,
            "regle_decision": "cout_minimal",
        }
    )
    mlflow.log_metrics(
        {
            "f1_macro": 0.688,
            "recall_classe_2": 0.702,
            "taux_erreur_2_vers_0": 0.050,
            "cout_moyen": 0.534,
            "di_classe0_cout_minimal": 0.167,
            "recall_classe_2_groupe_majoritaire": 0.667,
            "recall_classe_2_groupe_hors_ue": 0.804,
            "taux_2_vers_0_groupe_majoritaire": 0.063,
            "taux_2_vers_0_groupe_hors_ue": 0.011,
        }
    )
    mlflow.set_tags(
        {
            "verdict": "ecarte",
            "raison": "regle corrigee, mais usage direct de nationalite_hors_ue = risque legal independant de l'effet mesure",
            "notebook_section": "§6.2",
        }
    )

# ---------------------------------------------------------------------------
# exp_004 — S2-LightGBM, décision à coût minimal + abstention 0,7 (RETENU)
# ---------------------------------------------------------------------------
with mlflow.start_run(run_name="exp_004_s2_lightgbm_abstention"):
    mlflow.log_params(
        {
            "modele": "LGBMClassifier",
            "scenario": "S2",
            "dataset_sha256": DATASET_SHA256,
            "split": "train/test 2000/500, stratify=y, random_state=42",
            "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "random_state": 42,
            "regle_decision": "cout_minimal",
            "seuil_abstention": 0.7,
        }
    )
    mlflow.log_metrics(
        {
            "f1_macro": 0.720,
            "recall_classe_2": 0.719,
            "kappa_pondere": 0.645,
            "taux_erreur_2_vers_0": 0.056,
            "cout_moyen": 0.490,
            "taux_revue_humaine": 0.187,
            "di_classe0_cout_minimal": 0.722,
            "di_classe2_accompagnement_cout_minimal": 0.825,
            "recall_classe_2_groupe_majoritaire": 0.704,
            "recall_classe_2_groupe_hors_ue": 0.620,
            "taux_2_vers_0_groupe_majoritaire": 0.041,
            "taux_2_vers_0_groupe_hors_ue": 0.076,
        }
    )
    mlflow.set_tags(
        {
            "verdict": "retenu",
            "raison": "risque legal de l'usage direct d'une variable protegee evite ; cout reel pour le groupe hors-UE documente et a surveiller en production",
            "notebook_section": "§6.2 / §6.6 / §7.2",
        }
    )

# ---------------------------------------------------------------------------
# exp_005 — S2-LightGBM, réentraîné train+test combinés (production)
# ---------------------------------------------------------------------------
with mlflow.start_run(run_name="exp_005_production"):
    mlflow.log_params(
        {
            "modele": "LGBMClassifier",
            "scenario": "S2",
            "dataset_sha256": DATASET_SHA256,
            "split": "train+test combines (n=2500), aucun holdout restant",
            "random_state": 42,
            "regle_decision": "cout_minimal",
            "seuil_abstention": 0.7,
            "model_version": "v1.0.0",
        }
    )
    mlflow.log_metrics(
        {
            # Holdout = verdict du test scellé (§7.4), calculé AVANT la
            # fusion train+test, jamais recalculé depuis (cf. règle anti-fuite)
            "f1_macro_holdout": 0.722,
            "recall_classe_2_holdout": 0.7027027027027027,
            "kappa_pondere_holdout": 0.68826705940108,
            "taux_erreur_2_vers_0_holdout": 0.02702702702702703,
            "cout_moyen_holdout": 0.44096385542168676,
            "taux_revue_humaine_holdout": 0.17,
            "di_classe0_argmax_holdout": 0.8080072793448589,
            "recall_classe_2_groupe_majoritaire_holdout": 0.707,
            "recall_classe_2_groupe_hors_ue_holdout": 0.688,
            "taux_abstention_groupe_majoritaire_holdout": 0.155,
            "taux_abstention_groupe_hors_ue_holdout": 0.286,
        }
    )
    mlflow.set_tags(
        {
            "verdict": "retenu",
            "raison": "modele de production, package dans models/trajectoire_emploi_v1.joblib, servi par l'API",
            "notebook_section": "§8.1-8.2",
        }
    )

print("5 runs loggués dans l'expérience 'trajectoire_emploi'.")
print("Lancer `mlflow ui` pour les comparer (http://localhost:5000).")
