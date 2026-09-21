import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.write(''' # Predicción de categoría de Premio Nobel ''')
st.image("Nobel.png", caption="Su creador fue el inventor sueco Alfred Nobel mediante su testamento en 1895.")

st.header('Texto')


def user_input_features():
    # Entrada
    texto = st.text_input("Introduce el texto a evaluar")
    user_input_data = {'Text': texto}
    features = pd.DataFrame(user_input_data, index=[0])
    return features


df = user_input_features()

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^a-z\s]", " ", texto)   # quita puntuación, números y caracteres especiales
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

nobel = pd.read_csv(
    "https://raw.githubusercontent.com/SalvadorCM786/ExamenModuloIV/refs/heads/main/nobelsalvador_limpio.csv"
)
nobel = nobel.dropna(subset=["Motivation"]).copy()

# Antes: nobel.Text / nobel.Label no existían en ese csv -> se construyen aquí.
nobel["Text"] = nobel["Motivation"].apply(limpiar_texto)

# Mismo mapeo manual que ya tenías comentado, ahora aplicado de verdad con .map()
# (antes el modelo se entrenaba con un csv distinto, así que este mapeo nunca se usaba).
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
