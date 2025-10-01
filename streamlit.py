from PIL import Image 
import streamlit as st
import matplotlib.pyplot as plt
from utils.modelisation import display_complete_modeling
import numpy as np
import pandas as pd
import os
import seaborn as sns

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
    st.markdown("""
    L’intelligence artificielle révolutionne aujourd’hui le domaine de la santé, notamment à travers l’automatisation et l’amélioration de l’interprétation des images médicales.
    **Ce projet s’inscrit à l’intersection de l’imagerie médicale et du Deep Learning**, avec pour objectif de concevoir un modèle capable d’analyser des radiographies pulmonaires, afin d’accélérer le diagnostic et de réduire la charge de travail des professionnels de santé.
    """)

    st.image("https://offices-appines-prod.s3.eu-west-1.amazonaws.com/60b8e6c746560a0152654dbe-64b501c5cce02a35918af8da-sd.png",
                 caption="Exemple de radiographie pulmonaire analysée par le modèle", use_container_width=True)

    st.markdown("""
    ---

    **Contexte et enjeux**
    En période de crise sanitaire, comme celle de la COVID-19, la pression sur les systèmes de santé est considérable.
    Les radiologues et médecins doivent traiter un volume élevé d’examens, souvent dans des délais très courts.
    L’automatisation de l’analyse des radiographies pulmonaires représente une opportunité majeure pour :
    - **Optimiser le temps médical** en réduisant le temps d’examen nécessaire,
    - **Améliorer la rapidité et la précision des diagnostics**, notamment dans les régions où l’accès aux soins est limité,
    - **Faciliter le tri des cas urgents** et réduire les erreurs liées à la surcharge de travail.
    """)
    try:
        image = Image.open("images/nom_de_ton_image.jpg")
        st.image(image, caption="Exemple de radiographie pulmonaire analysée par le modèle", use_column_width=True)
    except:
        # Option 2 : Image depuis une URL (exemple)
        st.image("https://marketing.webassets.siemens-healthineers.com/1800000005363926/a4e0eb057583/v/776ec3a71434/SIEMENS_HEALTHINEERS_Ysio_Max_Xray_Image_1800000005363926.jpg",
                 caption="Exemple de radiographie pulmonaire analysée par le modèle", use_container_width=True)

    st.markdown("""
    ---

    **Approche technique**
    Le projet repose sur une méthodologie rigoureuse de Deep Learning, combinant **classification** et **segmentation** :
    - **Classification** : identifier automatiquement la présence de pathologies (COVID-19, pneumonie, opacité pulmonaire, etc.) à partir d’une radiographie,
    - **Segmentation** : générer un masque binaire des poumons pour localiser précisément les zones affectées.

    Pour ce faire, nous utilisons un jeu de données varié et annoté, issu de la plateforme Kaggle (COVID-19 Radiography Database), comprenant plus de 21 000 images de radiographies pulmonaires, couvrant plusieurs pathologies.
    Ces données, bien que précieuses, soulèvent des questions éthiques et réglementaires, notamment en Europe avec le RGPD, qui encadre strictement l’utilisation des données de santé.

    ---

    **Impact et perspectives**
    Au-delà de l’aspect technique, ce projet vise à **démocratiser l’accès à un outil de diagnostic rapide et fiable**, particulièrement utile dans les zones rurales ou les pays en développement.
    L’intégration du modèle dans les systèmes hospitaliers, via une interface intuitive, permettrait aux praticiens de l’utiliser directement pour un diagnostic assisté.

    Enfin, ce travail s’inscrit dans une démarche de recherche appliquée, cherchant à concilier **innovation technologique, éthique et viabilité économique**, pour une médecine plus accessible et performante.
    """)


if page == pages[1]:
    st.write("### Exploration")
    st.markdown("""
    Ce projet vise à explorer un jeu de données d'images de radiographies pulmonaires, annotées selon quatre pathologies : **COVID-19**, **Pneumonie**, **Opacité pulmonaire** et **Non malade**.
    L'objectif est de préparer ces données pour l'entraînement d'un modèle de Deep Learning capable de discriminer ces pathologies.
    """)

    # Section 1 : Contexte et pertinence
    st.header("1. Contexte et pertinence du projet")
    st.markdown("""
    - **Origine des données** : [COVID-19 Radiography Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
    - **Volume** : 3 616 cas COVID-19, 10 192 sains, 6 012 opacités, 1 345 pneumonies.
    - **Objectif** : Automatiser l'interprétation des radiographies pour libérer du temps médical et améliorer le dépistage.
    - **Enjeux** : Déséquilibre des classes, qualité des données, éthique (RGPD).
    """)

    # Section 2 : Structure des données
    st.header("2. Structure des données")
    st.markdown("""
    Les données sont organisées en dossiers par pathologie, chacun contenant :
    - **images** : radiographies en PNG (299x299 pixels, niveaux de gris)
    - **masks** : masques de segmentation des poumons (binaire, noir/blanc)
    """)
    # Exemple de structure (à adapter selon vos chemins)
    st.code("""
    COVID/
    ├── images/
    │   ├── covid-1.png
    │   ├── covid-2.png
    │   └── ...
    ├── masks/
    │   ├── covid-1_mask.png
    │   ├── covid-2_mask.png
    │   └── ...
    ...
    """)

    # Section 3 : Visualisation des images
    st.header("3. Cadre des données pour l'exploitation")
    st.markdown("""        
        
        Les données sont des images au format PNG, où chaque pixel est représenté par une valeur numérique 
        indiquant son intensité lumineuse. Ces images peuvent être converties en tableaux numériques pour analyse.
                
        Les radiographies exploitent l’interaction des rayons X avec les tissus : l’air (noir) laisse passer les 
        rayons, tandis que les os et tissus denses (blancs) les absorbent. Pour optimiser l’analyse, les radiographies 
        pulmonaires sont réalisées avec un maximum d’air dans les poumons, afin de mieux visualiser leur structure interne.
                
        Les masques de segmentation sont des images binaires (noir et blanc) associées à chaque radiographie. Ils servent 
        à délimiter et identifier précisément les zones d’intérêt, comme les contours des poumons ou des lésions, pour 
        faciliter l’analyse automatique.
    """)
    # Exemple de chargement d'images (à adapter)
    cols = st.columns(2)
    classes = ["radiography", "mask"]
    filename = "Normal"
    for i, class_name in enumerate(classes):
        with cols[i]:
            st.subheader(class_name)
            # Remplacez par le chemin réel de vos images
            img_path = f"images/{class_name}/{filename}-1.png"
            if os.path.exists(img_path):
                img = Image.open(img_path)
                st.image(img, caption=f"{class_name} (299x299)", use_container_width=True)
            else:
                st.write("Image non disponible")

    # Section 4 : Distribution des classes
    st.header("4. Distribution des classes")
    st.markdown("""
    La distribution des classes est déséquilibrée, ce qui peut biaiser l'entraînement du modèle.
    """)
    # Exemple de données (à remplacer par vos vraies stats)
    data = {
        "Pathologie": ["COVID-19", "Non malade", "Opacité pulmonaire", "Pneumonie"],
        "Nombre d'images": [3616, 10192, 6012, 1345]
    }
    df = pd.DataFrame(data)
    fig, ax = plt.subplots()
    sns.barplot(x="Pathologie", y="Nombre d'images", data=df, ax=ax, palette="Set2")
    ax.set_title("Distribution des classes")
    st.pyplot(fig)
    st.markdown("""
    - **Test du chi-deux** : p-value = 0.0, χ²=8110 → distribution significativement déséquilibrée.
    - **Solution envisagée** : sous-échantillonnage ou pondération des classes.
    """)

    # Section 5 : Distribution de la valeurs des pixels pour chacune des classes (pour 100 chacune)
    st.header("5. Distribution de la valeurs des pixels pour chacune des classes (pour 100 elements chacune)")
    st.markdown("""
        Chaque pixel d’une radiographie, codé sur une échelle de 0 à 255, porte une information cruciale : 
        il reflète l’absorption des rayons X par les tissus, révélant ainsi des contrastes invisibles à l’œil nu
    """)
    img_path = f"images/pixels_value.png"
    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img, caption=f"Distribution des valeurs des pixels", use_container_width=True)
    else:
        st.write("Image distribution des pixels non disponible")


    # Section 6 : Réduction de dimension et clustering
    st.header("6. Réduction de dimension et clustering")
    st.markdown("""
    - **Réduction** : Passage de 256x256 à 199x199 pixels, puis application de PCA/UMAP.
    - **Clustering** : K-Means sur 54 composantes principales (90% de variance expliquée).
    - **Résultat** : Coefficient de silhouette moyen = 0.1455 → clusters peu distincts.
    """)
    # Exemple de visualisation UMAP (à adapter)
    st.markdown("""
    La visualisation UMAP et reduction de dimension montre un chevauchement important entre les classes, confirmant la nécessité d'un modèle plus complexe (CNN).
    """)
    img_path = f"images/umap.png"
    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img, caption=f"Umap Distribution des classes", use_container_width=True)
    else:
        st.write("Image umap des pixels non disponible")

    st.markdown("""
    K-Means sur 54 composantes principales (90% de variance expliquée)    """)
    img_path = f"images/clustering.png"
    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img, caption=f"Clustering K-means", use_container_width=True)
    else:
        st.write("Image clustering des pixels non disponible")

    st.markdown("""
    Coefficient de silhouette moyen   """)
    img_path = f"images/silhouettes.png"
    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img, caption=f"Distribution des silhouettes", use_container_width=True)
    else:
        st.write("Image distribution des silhouettes non disponible")


if page == pages[2]:
    st.write("### Preprocessing")

if page == pages[3]:
    st.write("### Modelisation")

    display_complete_modeling()

if page == pages[4]:
    st.write("### Résultat Gilles 2025 10 01") 

if page == pages[5]:
    st.write("### Analyse et Conclusion")


