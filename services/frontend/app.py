"""Interface conseiller — Streamlit.

Deux usages :
1. Scorer un dossier (consomme `POST /predict`) et afficher la décision.
2. Remonter la vraie classe observée après coup (`POST /feedback`), à partir
   du `request_id` affiché à l'étape 1 — alimente la boucle de
   réentraînement

Communication avec le backend via le nom de service Docker (`http://backend:8000`)
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

NIVEAUX_DIPLOME = ["Sans diplôme", "Bac", "Bac+2", "Bac+5"]
LIBELLE_CLASSE = {
    0: "retour_rapide",
    1: "retour_moyen",
    2: "risque_longue_duree",
}

st.set_page_config(page_title="Trajectoire Emploi — Conseiller", page_icon="🧭")
st.title("🧭 Trajectoire Emploi — Interface conseiller")

with st.sidebar:
    st.caption(f"API : {API_URL}")
    try:
        httpx.get(f"{API_URL}/health", timeout=3).raise_for_status()
        st.success("API joignable")
    except httpx.HTTPError:
        st.error("API injoignable")

tab_scorer, tab_feedback = st.tabs(["Scorer un dossier", "Remonter un feedback"])

# --- Onglet 1 : scorer un dossier -------------------------------------------------
with tab_scorer:
    st.subheader("Scorer un dossier")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Âge", min_value=16, max_value=70, value=35)
        anciennete = st.number_input(
            "Ancienneté dans le dernier poste (ans)", min_value=0.0, max_value=50.0, value=3.0
        )
        niveau_diplome = st.selectbox("Niveau de diplôme", NIVEAUX_DIPLOME, index=2)
    with col2:
        code_rome = st.text_input("Code ROME visé", max_chars=5, value="D1503")
        code_insee = st.text_input("Code INSEE commune", max_chars=5, value="18273")
        est_allocataire = st.checkbox("Allocataire chômage", value=True)

    synthese = st.text_area(
        "Synthèse de l'entretien", height=100, placeholder="Candidat motivé…"
    )

    if st.button("Scorer", type="primary"):
        payload = {
            "age": age,
            "anciennete_poste_ans": anciennete,
            "niveau_diplome": niveau_diplome,
            "code_rome_vise": code_rome,
            "code_insee_commune": code_insee,
            "est_allocataire": est_allocataire,
            "synthese_entretien": synthese,
        }
        try:
            with st.spinner("Scoring en cours…"):
                r = httpx.post(f"{API_URL}/predict", json=payload, timeout=10)
            r.raise_for_status()
            data = r.json()
        except httpx.TimeoutException:
            st.error("⏱️ API trop lente (>10s).")
        except httpx.HTTPStatusError as exc:
            st.error(f"HTTP {exc.response.status_code} : {exc.response.text}")
        except httpx.HTTPError as exc:
            st.error(f"Erreur réseau : {exc}")
        else:
            decision = data["decision"]
            display = {
                "retour_rapide": st.success,
                "retour_moyen": st.warning,
                "risque_longue_duree": st.error,
                "revue_humaine": st.warning,
            }
            display[decision](f"Décision : **{decision}**")
            st.bar_chart(data["probabilites"])
            st.caption(
                f"Coût attendu : {data['cout_attendu']:.3f} — "
                f"modèle : {data['model_version']}"
            )
            st.code(data["request_id"], language=None)
            st.info(
                "Conservez ce `request_id` : il permet de remonter la vraie "
                "classe observée plus tard, dans l'onglet « Remonter un feedback »."
            )

# --- Onglet 2 : remonter un feedback -----------------------------------------------
with tab_feedback:
    st.subheader("Remonter la vraie classe observée")

    request_id = st.text_input("request_id (affiché après un scoring)")
    vraie_classe = st.selectbox(
        "Vraie classe observée",
        options=list(LIBELLE_CLASSE.keys()),
        format_func=lambda c: f"{c} — {LIBELLE_CLASSE[c]}",
    )
    commentaire = st.text_area("Commentaire (optionnel)", height=80)

    if st.button("Envoyer le feedback", type="primary", disabled=not request_id.strip()):
        payload = {
            "request_id": request_id.strip(),
            "true_label": vraie_classe,
            "comments": commentaire or None,
        }
        try:
            r = httpx.post(f"{API_URL}/feedback", json=payload, timeout=10)
        except httpx.HTTPError as exc:
            st.error(f"Erreur réseau : {exc}")
        else:
            if r.status_code == 201:
                statut = r.json()["status"]
                if statut == "cree":
                    st.success("Feedback enregistré.")
                else:
                    st.info("Feedback déjà enregistré à l'identique (idempotent).")
            elif r.status_code == 404:
                st.error("request_id inconnu : aucune prédiction associée.")
            elif r.status_code == 409:
                st.error(
                    "Conflit : un feedback différent existe déjà pour ce "
                    "request_id. Arbitrage humain requis — contactez un référent."
                )
            else:
                st.error(f"Erreur inattendue : HTTP {r.status_code} — {r.text}")

    st.divider()
    try:
        compte = httpx.get(f"{API_URL}/feedback/count", timeout=3).json()
        st.caption(
            f"Feedbacks stockés : {compte['total']} "
            f"(dont {compte['new']} non encore utilisés pour un réentraînement)"
        )
    except httpx.HTTPError:
        st.caption("Compteur de feedbacks indisponible.")
