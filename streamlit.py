from PIL import Image 
import streamlit as st
import matplotlib.pyplot as plt
from utils.modelisation import display_complete_modeling
from utils.introduction import display_intro
from utils.exploration import display_exploration
from utils.preprocessing import preprocessing_covid


st.title("Deep X-Vision Project")
st.sidebar.title("Sommaire")
pages=["Introduction", "Exploration", "Preprocessing", "Modélisation", "Resultat", "Analyse et Conclusion"]
page=st.sidebar.radio("Aller vers", pages)
# Sidebar : Informations complémentaires

st.sidebar.markdown("---")  # Ligne de séparation
st.sidebar.title("À propos")
st.sidebar.markdown("""
- **Auteurs** : Claudia FABRE, Gilles DOMENC, Romain JILIBERT
- **Promo** : Dec-2024
- **Source des données** : [COVID-19 Radiography Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **Outils** : Streamlit, Python, Scikit-learn, TensorFlow, PyTorch, Matplotlib
""")

if page == pages[0]:
    st.write("### Introduction")
    display_intro()


if page == pages[1]:
    st.write("### Exploration")
    display_exploration()

if page == pages[2]:
    st.write("### Preprocessing")
    preprocessing_covid()

if page == pages[3]:
    st.write("### Modelisation")

    display_complete_modeling()

if page == pages[4]:
    st.write("### Résultat Gilles 2025 10 01") 

if page == pages[5]:
    st.write("### Analyse et Conclusion")


