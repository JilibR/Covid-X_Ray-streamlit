import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from PIL import Image 

def preprocessing_covid():
    presention_preprocessing()
    apply_clahe()
    aumentation()
    normalisation()
    encodage_schemas()
    schemas_two_steps()

def schemas_two_steps():
    st.subheader("Pipeline de Classification COVID-19")

    fig, ax = plt.subplots(figsize=(7, 10))

    # --- Définition des étapes ---
    steps = {
        "step1": ("Etape 1 : Malade / Non malade", (2, 8), "lightgray", 3, 1.0),   # ligne 1
        "malade": ("Malade", (3.5, 6), "salmon", 2, 0.8),                # ligne 2 gauche
        "non_malade": ("Non malade", (0.5, 6), "lightgreen", 2, 0.8),    # ligne 2 droite
        "step2": ("Étape 2 : Covid / Non Covid", (3.5, 4.5), "lightblue", 3, 1.0),  # ligne 3
        "covid": ("Covid", (2.5, 3), "red", 2, 0.8),                     # ligne 4 gauche
        "non_covid": ("Non Covid", (4.5, 3), "skyblue", 2, 0.8),         # ligne 4 droite
    }

    # --- Dessiner les rectangles ---
    for key, (label, (x, y), color, w, h) in steps.items():
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                                            boxstyle="round,pad=0.2",
                                            fc=color, ec="black", lw=1.5))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=10, weight="bold")

    # --- Flèches (descendantes entre les étapes) ---
    # Ligne 1 -> Ligne 2 (vers "Malade")
    ax.annotate("", xy=(4.5, 8), xytext=(4.5, 7), arrowprops=dict(arrowstyle="<-"))

    # Ligne 2 Malade -> Ligne 3
    ax.annotate("", xy=(4.5, 6), xytext=(4.5, 5.5), arrowprops=dict(arrowstyle="<-"))

    # Ligne 3 -> Ligne 4 (Covid & Non Covid)
    ax.annotate("", xy=(3.5, 4.5), xytext=(3.5, 3.8), arrowprops=dict(arrowstyle="<-"))
    ax.annotate("", xy=(5.5, 4.5), xytext=(5.5, 3.8), arrowprops=dict(arrowstyle="<-"))

    # --- Ajustements ---
    ax.set_xlim(0, 7)
    ax.set_ylim(2, 9.5)
    ax.axis("off")

    st.pyplot(fig)


def presention_preprocessing():

    st.markdown("""
    Cette section explique comment nous préparons les images et leurs masques avant l'entraînement du modèle de deep learning.
    """)

    st.title("Prétraitement des données pour la classification de radiographies")

    st.header("1. Redimensionnement des images et des masques")
    st.write("""
    Les images et leurs masques de segmentation n'ont pas toujours la même taille.
    Nous redimensionnons les images pour qu'elles correspondent aux dimensions des masques,
    ce qui réduit les calculs tout en préservant la qualité nécessaire pour le modèle.
    """)

    st.header("2. Simplification des masques")
    st.write("""
    Les masques sont souvent en 3 canaux (RGB), mais ne contiennent que deux valeurs (noir et blanc).
    Nous les convertissons en un seul canal pour simplifier leur gestion et améliorer l'efficacité du modèle.
    """)

    st.markdown("""
    ---
    **Pourquoi c'est important ?**
    Ces étapes permettent d'optimiser la taille des données, de réduire la complexité des calculs
    et d'améliorer les performances du modèle de classification.
    """)

def encodage_schemas():

    st.title("Stratégie de classification en deux étapes")
    st.markdown("""
    Cette section détaille la méthode utilisée pour classer les patients en deux étapes successives.
    La première étape de eéparation des patients en deux groupes (malade/non-malade) Puis la 
    classification parmi les malades, classification en COVID/Non-COVID
    """)

    st.header("1. Première étape : Malade vs Non-malade")
    st.markdown("""
    **Encodage :**
    - **1** pour les patients **malades**
    - **0** pour les patients **non-malades**
    """)
    st.code("""
    # Exemple d'encodage pour la première étape
    labels_etape_1 = {"malade": 1, "non-malade": 0}
    """)

    st.header("2. Deuxième étape : COVID vs Non-COVID (pour les malades uniquement)")
    st.markdown("""
    **Encodage (appliqué uniquement aux patients classés comme malades) :**
    - **1** pour les patients **COVID**
    - **0** pour les patients **non-COVID**
    """)
    st.code("""
    # Exemple d'encodage pour la deuxième étape (uniquement pour les malades)
    labels_etape_2 = {"covid": 1, "non-covid": 0}
    """)

    
def apply_clahe():

    st.header("3. Filtre CLAHE")
    st.markdown("""CLAHE est un prétraitement essentiel pour les radiographies 
                car il améliore le contraste local, préserve les détails et 
                optimise les performances des modèles de classification.""")
    img_path = f"images/clahe.png"
    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img, caption=f"Apllication filtre CLAHE", use_container_width=True)
    else:
        st.write("Image CLAHE non disponible")


def aumentation():

    st.header("4. Augmentation des données")
    st.markdown("""
    **Objectif** : Pallier le déséquilibre des classes dans la variable cible.

    **Stratégie** :
    - Générer davantage d’exemples **pendant l’entraînement** du modèle.
    - Diversifier les images via des transformations :
    - Rotations
    - Ajout de bruit
    - Retournements
    - Zooms
    - Modifications de luminosité/contraste

    **Avantages** :
    - Augmente la taille du jeu de données.
    - Enrichit la variabilité des données, réduisant le surapprentissage.
    - Optimisée au fil de la modélisation pour affiner la méthode.
    """)
    img_path = f"images/augmentation.png"
    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img, caption=f"Exemple d'augmentation de données", use_container_width=True)
    else:
        st.write("Image augmentation non disponible")

def normalisation():
    st.header("5. Normalisation des images")
    st.markdown("""
    La normalisation est un méthode utile pour  améliorer la stabilité de l’apprentissage et 
    faciliter la convergence du modèle.

    **Méthode** :
    - Les valeurs de pixels sont initialement comprises entre **0 et 255**.
    - On divise chaque pixel par **255** pour les ramener dans l’intervalle **[0, 1]**.
    """)
    
    st.header("Resultat du préprocessing")
    st.markdown("""
    Le resultat de chacune des étapes mise en évidence pour la préparation des radiographies
    et qui seront ensuite coupées ensuite couplées avec leur masque respectif.          
    """)
    img_path = f"images/fin_preprocessing.png"
    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img, caption=f"Exemple du résultat du préprocessing", use_container_width=True)
    else:
        st.write("Image fin_preprocessing non disponible")
