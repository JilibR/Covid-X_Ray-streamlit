from PIL import Image 
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

def display_exploration():
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
            img_path = f"images/{class_name}/{filename}-1.png"
            if os.path.exists(img_path):
                img = Image.open(img_path)
                if class_name == "radiography":
                    st.image(img, caption=f"{class_name} (299x299)", use_container_width=True)
                else:  # mask
                    st.image(img, caption=f"{class_name} (256x256)", use_container_width=True)
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
    - **Réduction** : Passage de 256x256, puis application de PCA/UMAP.
    - **Clustering** : K-Means sur 54 composantes principales (90% de variance expliquée).
    - **Résultat** : Coefficient de silhouette moyen = 0.1455 → clusters peu distincts.
    """)

    graph_presenation()

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


def graph_presenation():
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    #st.markdown("""
    #Pipeline PCA + UMAP appliqué aux radiographies   """)
    st.subheader("Pipeline PCA + UMAP appliqué aux radiographies")

    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Cases (2 lignes, 3 colonnes)
    steps = [
        "Radiographie\n256x256",
        "Réduction\n199x199",
        "Vecteur\n39 601 dims",
        "PCA\n~54 dims",
        "UMAP\n2D",
        "Clusters\nvisibles"
    ]

    # Positions (x, y)
    positions = [
        (0, 2), (3, 2), (6, 2),   # ligne du haut
        (0, 0), (3, 0), (6, 0)    # ligne du bas
    ]

    # Couleurs
    colors = ["lightgray", "lightsteelblue", "lightblue",
            "lightgreen", "plum", "lightsalmon"]

    # Dessiner les boîtes
    for (step, (x, y), c) in zip(steps, positions, colors):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), 2.5, 1.2,
                                            boxstyle="round,pad=0.2",
                                            fc=c, ec="black", lw=1.5))
        ax.text(x+1.25, y+0.6, step, ha="center", va="center", fontsize=9, weight="bold")

    # Flèches horizontales (ligne du haut)
    ax.annotate("", xy=(3, 2.6), xytext=(2.5, 2.6), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(6, 2.6), xytext=(5.5, 2.6), arrowprops=dict(arrowstyle="->"))

    # Flèches horizontales (ligne du bas)
    ax.annotate("", xy=(3, 0.6), xytext=(2.5, 0.6), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(6, 0.6), xytext=(5.5, 0.6), arrowprops=dict(arrowstyle="->"))

    ax.set_xlim(-1, 9)
    ax.set_ylim(-1, 4)
    ax.axis("off")
    ax.set_title("Pipeline PCA + UMAP appliqué aux radiographies", fontsize=8, weight="bold")

    # Affichage dans Streamlit
    st.pyplot(fig)
