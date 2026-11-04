import base64

import requests
import streamlit as st

API_URL = "http://localhost:8000"

COULEURS_RISQUE = {
    "bénin": "#2e9e4f",
    "faible": "#2e9e4f",
    "précancéreux": "#f0a202",
    "modéré": "#f0a202",
    "malin": "#d62828",
    "élevé": "#d62828",
}

NOMS_CLASSES = {
    "akiec": "Kératose actinique / carcinome intra-épithélial",
    "bcc": "Carcinome basocellulaire",
    "bkl": "Kératose bénigne",
    "df": "Dermatofibrome",
    "mel": "Mélanome",
    "nv": "Naevus mélanocytaire",
    "vasc": "Lésion vasculaire",
}

st.set_page_config(page_title="Aide au dépistage cutané", layout="wide")
st.title("Aide au dépistage cutané")

st.warning(
    "Cet outil est une aide à la décision. Il ne remplace pas "
    "un diagnostic médical. Consultez un dermatologue."
)


def appeler_api(fichier):
    reponse = requests.post(
        f"{API_URL}/predict",
        files={"file": (fichier.name, fichier.getvalue(), fichier.type)},
        timeout=30,
    )
    reponse.raise_for_status()
    return reponse.json()


def afficher_resultat(resultat, fichier):
    colonne_images, colonne_resultat = st.columns(2)

    with colonne_images:
        st.subheader("Image analysée")
        st.image(fichier)
        st.subheader("Zones importantes pour la décision")
        if resultat.get("heatmap_base64"):
            st.image(base64.b64decode(resultat["heatmap_base64"]))
        else:
            st.info("Heatmap non disponible pour le moment.")

    with colonne_resultat:
        st.subheader("Résultat")
        st.markdown(f"**Lésion prédite :** {resultat['lesion']}")

        risque = resultat.get("risque", "inconnu")
        couleur = COULEURS_RISQUE.get(risque, "#888888")
        st.markdown(
            f"<div style='background:{couleur};color:white;padding:14px;"
            f"border-radius:8px;font-size:20px;text-align:center'>"
            f"Niveau de risque : {risque.upper()}</div>",
            unsafe_allow_html=True,
        )

        if resultat.get("alerte_incertitude"):
            st.warning(
                "Le modèle est incertain sur cette image. "
                "Une consultation médicale est fortement conseillée."
            )

        st.subheader("Probabilités par classe")
        probabilites = sorted(
            resultat["probabilites"].items(), key=lambda x: x[1], reverse=True
        )
        for code, p in probabilites:
            nom = NOMS_CLASSES.get(code, code)
            st.progress(float(p), text=f"{nom} : {p:.0%}")


fichier = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if fichier and st.button("Analyser"):
    try:
        with st.spinner("Analyse en cours..."):
            st.session_state["resultat"] = appeler_api(fichier)
            st.session_state["fichier"] = fichier
    except requests.exceptions.ConnectionError:
        st.error("Impossible de joindre l'API. Est-elle lancée ?")
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de l'API : {e}")

if "resultat" in st.session_state:
    afficher_resultat(st.session_state["resultat"], st.session_state["fichier"])