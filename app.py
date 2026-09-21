import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

DATA_URL = (
    "https://raw.githubusercontent.com/SalvadorCM786/ExamenModuloIV/"
    "refs/heads/main/nobelsalvador_limpio.csv"
)


def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^a-z\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


@st.cache_resource
def entrenar_modelo():
    df = pd.read_csv(DATA_URL)
    df = df.dropna(subset=["Motivation"]).copy()
    df["Motivation_limpio"] = df["Motivation"].apply(limpiar_texto)

    vectorizador = TfidfVectorizer(
        stop_words="english", max_features=3000, min_df=2, ngram_range=(1, 2)
    )
    X = vectorizador.fit_transform(df["Motivation_limpio"])
    y = df["Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = MultinomialNB()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    etiquetas = sorted(y.unique())
    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "reporte": classification_report(y_test, y_pred, zero_division=0),
        "matriz": confusion_matrix(y_test, y_pred, labels=etiquetas),
        "etiquetas": etiquetas,
    }
    return vectorizador, modelo, metricas


st.title("Predicción de categoría de Premio Nobel")
st.caption("Su creador fue el inventor sueco Alfred Nobel mediante su testamento en 1895.")

with st.spinner("Entrenando el modelo (solo la primera vez)..."):
    vectorizador, modelo, metricas = entrenar_modelo()

st.header("Texto")
texto_usuario = st.text_input("Introduce el texto de una motivación de premio a evaluar")

if texto_usuario:
    texto_limpio = limpiar_texto(texto_usuario)
    vector = vectorizador.transform([texto_limpio])
    prediccion = modelo.predict(vector)[0]
    probabilidades = modelo.predict_proba(vector)[0]

    st.subheader("Predicción")
    st.success(f"Categoría predicha: **{prediccion.capitalize()}**")

    prob_df = pd.DataFrame(
        {"Categoría": modelo.classes_, "Probabilidad": probabilidades}
    ).sort_values("Probabilidad", ascending=False)
    st.bar_chart(prob_df.set_index("Categoría"))
else:
    st.info("Escribe un texto arriba para obtener una predicción.")

with st.expander("Ver desempeño del modelo (métricas y matriz de confusión)"):
    st.write(f"**Accuracy en el conjunto de prueba:** {metricas['accuracy']:.3f}")
    st.text(metricas["reporte"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=metricas["matriz"], display_labels=metricas["etiquetas"]
    )
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title("Matriz de confusión — Naive Bayes")
    st.pyplot(fig)
