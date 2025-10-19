from PIL import Image 
import streamlit as st
import matplotlib.pyplot as plt
from utils.modelisation import display_complete_modeling
from utils.introduction import display_intro
from utils.exploration import display_exploration
from utils.preprocessing import preprocessing_covid


st.title("Deep X-Vision Project")
st.sidebar.title("Sommaire")
pages=["Introduction", "Exploration", "Preprocessing", "Modélisation", "Résultats et Analyse", "Conclusion"]
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

elif page == pages[3]:
    st.write("### Modélisation")
    
    with st.expander("ℹ️ À propos de cette section", expanded=False):
        st.markdown("""
        Cette section présente les différentes approches de modélisation testées dans le projet :
        - **Stratégies de classification** : différentes approches (multiclasses, binaire, hiérarchique)
        - **Architectures** : modèles CNN utilisés (Transfer Learning et From Scratch)
        - **Paramètres** : hyperparamètres d'entraînement
        - **Métriques** : méthodes d'évaluation des performances
        """)
    
    display_complete_modeling()

if page == pages[4]:
    st.write("### Résultats et Analyse") 

    st.markdown("""
    Cette section présente l'analyse des performances de notre modèle de classification des radiographies pulmonaires.
    Nous évaluerons les résultats à travers trois outils complémentaires permettant une compréhension approfondie 
    des capacités et des limites du modèle.
    """)
    
    # Section 1 : Présentation des métriques d'évaluation
    st.header("1. Métriques d'évaluation")
  
    # Initialiser l'état du bouton si nécessaire
    if 'show_confusion_matrix' not in st.session_state:
        st.session_state.show_confusion_matrix = False

    # Bouton toggle
    if st.button("**Matrice de confusion**", key="toggle_confusion_matrix_exple"):
        st.session_state.show_confusion_matrix = not st.session_state.show_confusion_matrix

    # Affichage conditionnel
    if st.session_state.show_confusion_matrix:
        st.markdown("""  
        Comme nous l'avons vu, la **matrice de confusion** est un outil fondamental qui permet de :
        - **Évaluer les performances** du modèle de classification en comparant les valeurs prédites aux valeurs réelles
        - **Identifier les types d'erreurs** commises par le modèle :
            - **Faux négatifs** : cas pathologiques non détectés (risque médical élevé)  
            - **Faux positifs** : cas sains incorrectement classés comme pathologiques
        - **Comprendre les confusions** entre classes similaires (ex: COVID-19 vs Pneumonie)
        """)
        st.markdown("""
            Lecture de la matrice de confusion ci-dessous :
            - **Faux négatifs** (encadré en 🟡) : cas pathologiques non détectés
            - **Faux positifs** (encadré en 🔴) : cas sains incorrectement classés comme pathologiques
        """)
        img_path = "images/Exemple Matrice colorée.png"
        img = Image.open(img_path)
        st.image(img, caption="Matrice de confusion", use_container_width=True)

    #st.markdown("---")

    # Initialiser l'état du bouton si nécessaire
    if 'show_f1_details' not in st.session_state:
        st.session_state.show_f1_details = False

    # Bouton pour afficher/masquer les explications
    if st.button("F1-Score", key="toggle_F1_expl"):
        st.session_state.show_f1_details = not st.session_state.show_f1_details

    # Afficher les explications si l'état est True
    if st.session_state.show_f1_details:
        st.markdown("""
        Le **F1-score** représente la *moyenne harmonique* entre la **Précision** et le **Rappel** :

        ---
    
        **Précision** : Rapport entre le nombre de vrais positifs et le nombre total de positifs prédits
        - *Mesure la fiabilité des prédictions positives*
        - Formule : `Précision = VP / (VP + FP)`
    
        **Rappel** : Rapport entre le nombre de vrais positifs et le nombre total de positifs réels
        - *Mesure la capacité à détecter tous les cas positifs*
        - Formule : `Rappel = VP / (VP + FN)`
    
        ---

        **F1-Score** : `F1 = 2 × (Précision × Rappel) / (Précision + Rappel)`
    
        Le F1-score est compris entre **0 et 1** :
        - Plus il est proche de **1**, meilleurs sont le Rappel et la Précision
        - Il est particulièrement utile en cas de déséquilibre des classes
        - L'avantage principal de la moyenne harmonique est sa capacité à mettre en avant les valeurs
         plus petites, utile ici pour éviter qu'un score élevé dans une métrique ne masque un score faible dans l'autre.
        - En contexte médical, un F1-score élevé garantit un bon compromis entre détection et fiabilité
        """)


    # Rappel sur les GradCAM
    # Initialiser l'état du bouton
    if 'show_GradC' not in st.session_state:
        st.session_state.show_GradC = False

    # Bouton pour afficher/masquer les explications
    if st.button("GradCAM", key="toggle_GradCAM_expl"):
        st.session_state.show_GradC = not st.session_state.show_GradC

    # Afficher les explications si l'état est True
    if st.session_state.show_GradC:
        st.markdown("""
        Le **GradCAM** (**Grad**ient-weighted **C**lass **A**ctivation **M**apping) est une technique de visualisation qui révèle 
        les zones d'une image ayant le plus influencé la décision du réseau de neurones.

        ##### Interprétation des cartes de chaleur
        - **Zones rouges/chaudes** : Régions ayant fortement influencé la prédiction
        - **Zones bleues/froides** : Régions peu ou pas considérées par le modèle
    """)


    st.markdown("---")
    # Section 2 : Premiers tests CNN vs LeNet, avec et sans 
    st.header("2. Premiers tests CNN vs LeNet, sans et avec masques") 
    # Initialiser l'état du bouton
    if 'show_CNNLN' not in st.session_state:
        st.session_state.show_CNNLN = False

    # Bouton pour afficher/masquer les explications
    if st.button("F1 scores et GradCAM", key="toggle_CNNLN"):
        st.session_state.show_CNNLN = not st.session_state.show_CNNLN

    # Afficher les explications si l'état est True
    if st.session_state.show_CNNLN:
        st.subheader("Premiers F1 scores")
        img_path = "images/F1 CNN LeNet c.png"
        img = Image.open(img_path)
        st.image(img, caption="Comparatif des F1-score d'un CNN classique, d'un LeNet (sans et avec masque)", use_container_width=True)

        st.markdown("""
        Le modèle **CNN sans masque a le meilleur F1-score** (94,4 %)...
        """)

        #st.markdown("---")
        #st.subheader("GradCAM CNN sans masque")
        img_path = "images/Grad-CAM CNN No mask.png"
        img = Image.open(img_path)
        st.image(img, caption="GradCAM du CNN sans masque")

        st.markdown("""
        ...malheureusement le Grad-CAM indique que le modèle s’est 
        basé ***en-dehors des zones pulmonaires*** et en-dehors des zones périphériques des poumons (localisation classique des lésions COVID)
        """)
    st.markdown("---")

    st.markdown("### :arrow_forward: Nous avons choisi d'utiliser des réseaux de neurones convolutifs (CNN) plus récents qui sont plus profonds et plus efficaces.")

    st.markdown("---")

    # Section 3 : Architecture binaire : COVID vs Non-COVID
    st.header("3. Architecture binaire : COVID vs Non-COVID") 
    st.subheader("Matrice de confusion et F1 score")
    img_path = "images/Matrice - COVID vs all.png"
    img = Image.open(img_path)
    st.image(img, caption="Matrice de confusion - Stratégie COVID - Non COVID")

    st.markdown("""
    **F1-score de la classe COVID : 86,7 %**      
  
    """)
    # Initialiser l'état du bouton
    if 'show_confusion_matrixBCNC' not in st.session_state:
        st.session_state.show_confusion_matrixBCNC = False

    # Bouton 
    if st.button("Afficher / Masquer le GradCAM", key="toggle_confusion_GradC1"):
        st.session_state.show_confusion_matrixBCNC = not st.session_state.show_confusion_matrixBCNC

    # Affichage conditionnel
    if st.session_state.show_confusion_matrixBCNC:
        st.markdown("""
            On remarque que les zones qui ont servi aux prédictions sont **bien situées** dans les poumons : près de la plèvre et en bas des poumons.
        """)
        img_path = "images/GradCAM COVID vs All.png"
        img = Image.open(img_path)
        st.image(img, caption="GradCAM COVID vs Non-COVID") #, use_container_width=True)

    st.markdown("---")



    # Section 4 : Architecture à 2 étapes : différencier les Malades des Non-Malades, puis classifier les Malades
    st.header("4. Architecture à 2 étapes")
    st.subheader("On différencie d'abord les Malades des Non-Malades, puis on classifie au sein des Malades dans un 2e temps.") 
    st.markdown("---")

    # Initialiser l'état du bouton
    if 'show_confusion_matrix1' not in st.session_state:
        st.session_state.show_confusion_matrix1 = False

    # Bouton 
    if st.button("Matrice de confusion et F1 score", key="toggle_confusion_Mtrx1"):
        st.session_state.show_confusion_matrix1 = not st.session_state.show_confusion_matrix1

    # Affichage conditionnel
    if st.session_state.show_confusion_matrix1:
        img_path = "images/Matrice Malades - Non malades.png"
        img = Image.open(img_path)
        st.image(img, caption="Matrice de confusion - Stratégie à 2 étapes : Malades - Non Malades") #, use_container_width=True)
        st.markdown("""
            **F1-score de la classe Malade : 96,2 %**
     """)
        st.markdown("---")


    # Section 5.1 :Architecture en 2 étapes : mise en concurrence de deux stratégies
    st.header("5 :Architecture en 2 étapes : mise en concurrence de deux stratégies") 
    st.markdown("""
            **Deux** possibilités ont été envisagées :  
            - classification **binaire COVID vs Non-COVID** parmi les Malades
            - classification **multi-classes** (COVID, Pneumonie, Opacité pulmonaire) parmi les Malades
    """)
    st.subheader("5.1 Comparaison des matrices de confusion et des F1 scores")


    # Initialiser l'état du bouton
    if 'show_confusion_matrixD' not in st.session_state:
        st.session_state.show_confusion_matrixD = False
    # Bouton 
    if st.button("Afficher / Masquer les matrices de confusion", key="toggle_confusion_matrix51"):
        st.session_state.show_confusion_matrixD = not st.session_state.show_confusion_matrixD

    # Affichage conditionnel
    if st.session_state.show_confusion_matrixD:
        img_path = "images/D Matrices 2e étape.png"
        img = Image.open(img_path)
        st.image(img, caption="2e étape - Matrices de confusion COVID vs Non-COVID et Multi-classes")

    st.markdown("""
            F1-score, parmi les malades :  
             - classification **binaire COVID vs Non-COVID : 90,1 %**
             - classification **multi-classes : 93,4 %**  

            Au regard des F1-score, **la classification multi-classe s'avère plus pertinente que la classification binaire** (et elle permet de différencier les pathologies).   
            Les F1-score des Pneumonies et des Masses sombres semblent très bons (99 % et 95,9 %)
    """)
    
    # Initialiser l'état du bouton GradCAM
    if 'show_gradcam_multiclasse' not in st.session_state:
        st.session_state.show_gradcam_multiclasse = False
    
    # Bouton GradCAM
    if st.button("Afficher / Masquer le GradCAM multi-classes", key="toggle_gradcam_multiclasse"):
        st.session_state.show_gradcam_multiclasse = not st.session_state.show_gradcam_multiclasse
    
    # Affichage conditionnel du GradCAM
    if st.session_state.show_gradcam_multiclasse:
        img_gradcam_path = "images/Gradcam multiclassse.png"
        img_gradcam = Image.open(img_gradcam_path)
        st.image(img_gradcam, caption="GradCAM Analysis - EfficientNet Transfer Mask - Classification multi-classes")
        
        st.markdown("""
        **Analyse du GradCAM :**
        
        Cette visualisation montre les zones sur lesquelles le modèle se concentre pour prendre ses décisions :
        
        - **1er cas (COVID → Lung_Opacity)** : Le modèle détecte une opacité localisée plutôt que les patterns diffus typiques du COVID
        - **2e cas (Lung_Opacity → Lung_Opacity)** ✓ : Identification correcte des zones d'opacité dans les deux poumons
        - **3e cas (COVID → COVID)** ✓ : Détection correcte des patterns dans les régions périphériques et centrales
        - **4e cas (Lung_Opacity → COVID)** : Le modèle interprète l'opacité centrale comme un pattern COVID
        Le modèle semble se baser sur des caractéristiques pertinentes. Lordqu'il se trompe c'est qu'il fait des confusions entre opacités localisées et patterns diffus.
        """)
    
    st.markdown("---")


    # Section 5.2 : Matrice de confusion et F1-score globaux
    st.subheader("5.2 : Matrice de confusion et F1-score globaux") 
    st.markdown("""
            Pour pouvoir évaluer la performance de la stratégie à 2 étapes au regard de la stratégie 
            en une seule étape COVID - Non COVID, nous avons construit une matrice de confusion globale
            et calculé un F1-score. 
            """)
    st.subheader("Comparaison des matrices de confusion et des F1 scores")


    # Initialiser l'état du bouton
    if 'show_confusion_matrixF' not in st.session_state:
        st.session_state.show_confusion_matrixF = False
    # Bouton 
    if st.button("Afficher / Masquer les matrices de confusion", key="toggle_confusion_matrix_global"):
        st.session_state.show_confusion_matrixF = not st.session_state.show_confusion_matrixF

    # Affichage conditionnel
    if st.session_state.show_confusion_matrixF:
        img_path = "images/Matrice - COVID vs all.png"
        img = Image.open(img_path)
        st.image(img, caption="Matrice de confusion - Stratégie COVID - Non COVID")

        img_path = "images/Matrice globale.png"
        img = Image.open(img_path)
        st.image(img, caption="Matrice globale - Stratégie à 2 étapes, multi classes")

    st.markdown("""
        Le F1-score = 89,8 % de la stratégie en 2 étape reste meilleur que le F1-score de la classification binaire COVID/Non COVID sans 1ère étape (86,7 %)
    """)
 # Initialiser l'état du bouton
    if 'show_detl_mtrc' not in st.session_state:
        st.session_state.show_detl_mtrc = False
    # Bouton 
    if st.button("Détail des calculs de la matrice de confusion globale", key="toggle_dtl_mtrc"):
        st.session_state.show_detl_mtrc = not st.session_state.show_detl_mtrc

    # Affichage conditionnel
    if st.session_state.show_detl_mtrc:
        st.markdown("""               
**1. Attribution des cas "Normal" (Étape 1)**
Les 1 574 cas prédits comme "Normal" à l'étape 1 (84 faux négatifs + 1 490 
vrais négatifs) doivent être attribués à "Normal" dans la matrice globale,
car ils n'ont pas été soumis à l'étape 2.  

**2. Répartition des faux négatifs (84 cas)**    
Les **84 faux négatifs (classés Normal et en réalité Malade)** doivent être 
répartis parmi les classes malades (COVID, Lung_Opacity, Viral Pneumonia) 
selon leur **proportion réelle**.

Si les 1 646 vrais malades se répartissent ainsi :
- 40 % COVID (542 cas)
- 55 % Lung_Opacity (902 cas)  
- 12 % Viral Pneumonia (202 cas)

Alors les faux négatifs se répartissent :
- Faux négatifs pour COVID : 33 %
- Faux négatifs pour Lung_Opacity : 55 %
- Faux négatifs pour Viral Pneumonia : 12 %  

**3. Répartition des faux positifs (39 cas)**  
Les **39 Malades prédits à tort comme étant sains** doivent être répartis parmi les classes 
**prédites** à l'étape 2 (563 COVID, 879 Lung_Opacity, 204 Viral Pneumonia sur 900) :
- Faux positifs pour COVID : 34 %
- Faux positifs pour Lung_Opacity : 54 %
- Faux positifs pour Viral Pneumonia : 12 %

**4. Intégration de la matrice de l'étape 2**  
Les **1 562 malades prédits comme tels** doivent être répartis selon la répartition de la 
2ᵉ matrice de confusion (qui se base sur les 1 646 vrais malades de l'étape 1).

**Il faut multiplier la 2ᵉ matrice par 1 562 / 1 646 pour l'insérer dans la matrice globale.**
        """)


if page == pages[5]:
    st.write("### Conclusion")

    st.markdown("""
    Ce projet nous a permis de tester et de comparer différentes architectures de réseaux de neurones 
    et différentes stratégies de classification tout en mesurant l'importance des étapes de préprocessing.
    """)
    
    st.markdown("""
    Comme piste d'amélioration, nous aimerions conforter nos résultats en calculant la matrice de confusion globale
    et le F1-score global plutôt qu'en les reconstituant.
    """)

    st.markdown("""
    - **La stratégie en 2 étapes, multi classe parmi les malades, produit un F1-scode pour le COVID très intéressant (89,8 %)**
    """)

    st.markdown("""
    - **Cette stratégie en 2 étapes s'aligne avec la pratique clinique**, car souvent le radiologue 
    se demande *d'abord* s'il y a des anomalies sur la radiographie laissant penser que le 
    patient est malade (étape 1), puis, s'il y a des signes de pathologies pulmonaires, 
    quelle est cette maladie (étape 2 de la stratégie). On voit bien l'avantage de cette 
    stratégie qui va paraître pertinente aux interlocuteurs radiologues.
    """)
    
    st.markdown("""
    - Dans la stratégie en 2 étapes, la tâche d'apprentissage est simplifiée en divisant la tâche : d'abord 
    identifier les poumons sains, puis se focaliser sur les différences plus subtiles, 
    uniquement entre pathologies.
    """)
    
    st.markdown("""
    Enfin, le modèle est évolutif, car il permet la prise en compte de nouvelles pathologies.
    """)


