import re

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.write(''' # Nobel Prize Category Prediction ''')

try:
    st.image("Nobel.png", caption="It was established by the Swedish inventor Alfred Nobel through his 1895 will.")
except Exception:
    st.caption("It was established by the Swedish inventor Alfred Nobel through his 1895 will.")

st.header('Texto')


def user_input_features():
    texto = st.text_input("Enter the motivation text to evaluate")
    user_input_data = {'Text': texto}
    features = pd.DataFrame(user_input_data, index=[0])
    return features


df = user_input_features()


def limpiar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).lower()
    texto = re.sub(r"[^a-z\s]", " ", texto)   # quita puntuación, números y caracteres especiales
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

texto_usuario = df['Text'][0]

if texto_usuario:
    texto_limpio = limpiar_texto(texto_usuario)
    df_dtm = vect.transform([texto_limpio])
    prediction = nb.predict(df_dtm)


    st.subheader('Predicción')
    if prediction[0] == 0:
        st.write('Physics')
    elif prediction[0] == 1:
        st.write('Medicine')
    elif prediction[0] == 2:
        st.write('Peace')
    elif prediction[0] == 3:
        st.write('Literature')
    elif prediction[0] == 4:
        st.write('Chemistry')
    elif prediction[0] == 5:
        st.write('Economics')
    else:
        st.write('Sin predicción')
else:
    st.info("Escribe un texto arriba para obtener una predicción.")
