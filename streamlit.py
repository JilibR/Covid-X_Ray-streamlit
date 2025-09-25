import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Deep X-Vision Project")
st.sidebar.title("Sommaire")
pages=["Introduction", "Exploration", "Preprocessing", "Modélisation", "Resultat", "Analyse et Conclusion"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0]:
    st.write("### Introduction")

if page == pages[1]:
    st.write("### Exploration")

if page == pages[2]:
    st.write("### Preprocessing")

if page == pages[3]:
    st.write("### Modelisation")

if page == pages[4]:
    st.write("### Resultat")

if page == pages[5]:
    st.write("### Analyse et Conclusion")


