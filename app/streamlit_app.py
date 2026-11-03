import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Aide au dépistage cutané")
st.title("Aide au dépistage cutané")

st.warning(
    "Cet outil est une aide à la décision. Il ne remplace pas "
    "un diagnostic médical. Consultez un dermatologue."
)

fichier = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if fichier:
    st.image(fichier, caption="Image envoyée")

    if st.button("Analyser"):
        try:
            reponse = requests.post(
                f"{API_URL}/predict",
                files={"file": (fichier.name, fichier.getvalue(), fichier.type)},
                timeout=30,
            )
            reponse.raise_for_status()
            st.json(reponse.json())
        except requests.exceptions.ConnectionError:
            st.error("Impossible de joindre l'API. Est-elle lancée ?")
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur : {e}")