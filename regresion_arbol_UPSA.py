import streamlit as st
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisioTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Titulo para que salga en la página web
st.title("Predicción de Aprobación de Estudiantes con Árbol de Decisión")
st.markdown("Este modelo usa notas: Parciales, Proyecto y Examen Final para predecir si un estudiante aprobaría la materia")

# Cargar los datos (cargar el conjunto de datos para su analisis)
@st.cache_data
def cargar_datos():
    return pd.read_csv("estudiantes_notas_finales.csv")

# Recibiendo los datos carcagdos en la variable df (antes llamado dataset)
df = cargar_datos()

# Mostrar los primeros datos (cino primeros datos)
st.subheader("Datos cargados") # Este es un titulo
st.write(df.head()) # Esta instrucción muestra los primeros cinco datos.

# Gráficos simples
st.subheader("Distribución de notas") # Titulo que aparece en la pagina web
st.bar_chart(df[["Primer_Parcial", "Segundo_Parcial", "Proyecto", "Examen_Final", "Nota_Final"]].mean())