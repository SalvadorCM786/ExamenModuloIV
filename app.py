import re

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.write(''' # Nobel Prize Category Prediction ''')

try:
    st.image("Nobel.png", caption="Created by Swedish inventor Alfred Nobel through his 1895 will.")
except Exception:
    st.caption("Created by Swedish inventor Alfred Nobel through his 1895 will.")

st.header("Text")

if "texto_input" not in st.session_state:
    st.session_state.texto_input = ""

with st.form("prediction_form"):
    st.text_input("Enter the motivation text to evaluate", key="texto_input")
    col1, col2 = st.columns(2)
    with col1:
        enviar = st.form_submit_button("Enter", use_container_width=True)
    with col2:
        limpiar = st.form_submit_button("Clear", use_container_width=True)

if limpiar:
    st.session_state.texto_input = ""
    st.rerun()


def limpiar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).lower()
    texto = re.sub(r"[^a-z\s]", " ", texto)   # removes punctuation, numbers and special characters
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


nobel = pd.read_csv(
    "https://raw.githubusercontent.com/SalvadorCM786/ExamenModuloIV/refs/heads/main/nobelsalvador_limpio.csv"
)
nobel = nobel.dropna(subset=["Motivation"]).copy()
nobel["Motivation"] = nobel["Motivation"].astype(str)

nobel["Text"] = nobel["Motivation"].apply(limpiar_texto)
nobel = nobel[nobel["Text"].str.strip() != ""].copy()

mapa_categorias = {'physics': 0, 'medicine': 1, 'peace': 2, 'literature': 3, 'chemistry': 4, 'economics': 5}
nobel["Label"] = nobel["Category"].map(mapa_categorias)

X = nobel["Text"]
y = nobel["Label"]

vect = CountVectorizer()
X_dtm = vect.fit_transform(X)

nb = MultinomialNB()
nb.fit(X_dtm, y)

texto_usuario = st.session_state.texto_input

if enviar:
    if texto_usuario.strip():
        texto_limpio = limpiar_texto(texto_usuario)
        df_dtm = vect.transform([texto_limpio])
        prediction = nb.predict(df_dtm)

        st.subheader("Prediction")
        if prediction[0] == 0:
            st.success("Physics")
        elif prediction[0] == 1:
            st.success("Medicine")
        elif prediction[0] == 2:
            st.success("Peace")
        elif prediction[0] == 3:
            st.success("Literature")
        elif prediction[0] == 4:
            st.success("Chemistry")
        elif prediction[0] == 5:
            st.success("Economics")
        else:
            st.warning("No prediction")
    else:
        st.warning("Please enter some text before clicking Enter.")
else:
    st.info("Enter text above and click 'Enter' to get a prediction.")