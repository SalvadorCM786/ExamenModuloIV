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

texto_usuario = st.text_input("Enter the motivation text to evaluate")


def limpiar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).lower()
    texto = re.sub(r"[^a-z\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


nobel = pd.read_csv(
    "https://raw.githubusercontent.com/SalvadorCM786/ExamenModuloIV/refs/heads/main/nobel_limpio.csv"
)

nobel["Motivation"] = nobel["Motivation"].fillna("").astype(str)
nobel = nobel[nobel["Motivation"].str.strip().str.lower() != "nan"].copy()
nobel = nobel[nobel["Motivation"].str.strip() != ""].copy()

nobel["Text"] = nobel["Motivation"].apply(limpiar_texto)

nobel = nobel[nobel["Text"].str.strip() != ""].copy()

mapa_categorias = {'physics': 0, 'medicine': 1, 'peace': 2, 'literature': 3, 'chemistry': 4, 'economics': 5}
nobel["Label"] = nobel["Category"].map(mapa_categorias)

nobel = nobel.dropna(subset=["Label"]).copy()
nobel["Label"] = nobel["Label"].astype(int)

X = nobel["Text"]
y = nobel["Label"]

vect = CountVectorizer()
X_dtm = vect.fit_transform(X)

nb = MultinomialNB()
nb.fit(X_dtm, y)

if texto_usuario:
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
    st.info("Enter text above to get a prediction.")
