"""
Fonctions utilitaires pour la section Modélisation
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

# partie diagrammes
def create_single_strategy_diagram(strategy):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))    
    if strategy == "Classification Multiclasses (4 sorties)":
        # Dataset box
        dataset_box = FancyBboxPatch((0.2, 0.8), 0.6, 0.15, 
                                    boxstyle="round,pad=0.02", 
                                    facecolor='lightblue', 
                                    edgecolor='black', linewidth=2)
        ax.add_patch(dataset_box)
        ax.text(0.5, 0.875, 'Dataset Complet\n21,165 images', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Arrow down
        ax.arrow(0.5, 0.75, 0, -0.1, head_width=0.03, head_length=0.02, fc='black', ec='black', linewidth=2)
        
        # Model box
        model_box = FancyBboxPatch((0.2, 0.5), 0.6, 0.15, 
                                  boxstyle="round,pad=0.02", 
                                  facecolor='lightcoral', 
                                  edgecolor='black', linewidth=2)
        ax.add_patch(model_box)
        ax.text(0.5, 0.575, '"Classification Multiclasses (4 sorties)', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Arrow down
        ax.arrow(0.5, 0.45, 0, -0.1, head_width=0.03, head_length=0.02, fc='black', ec='black', linewidth=2)
        
        # Output boxes
        classes = [('COVID-19', '3,616'), ('Normal', '10,192'), ('Lung_Opacity', '6,012'), ('Viral Pneumonia', '1,345')]
        colors = ['#ff4444', '#44ff44', '#4444ff', '#ffaa44']
        for i, ((cls, count), color) in enumerate(zip(classes, colors)):
            x_pos = 0.05 + i * 0.225
            class_box = FancyBboxPatch((x_pos, 0.15), 0.2, 0.2, 
                                      boxstyle="round,pad=0.01", 
                                      facecolor=color, 
                                      edgecolor='black',
                                      alpha=0.8, linewidth=2)
            ax.add_patch(class_box)
            ax.text(x_pos + 0.1, 0.25, f'{cls}\n{count}', 
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    elif strategy == "Classificaiton Binaire (COVID vs Non-COVID)":
        # Dataset box
        dataset_box = FancyBboxPatch((0.2, 0.8), 0.6, 0.15, 
                                    boxstyle="round,pad=0.02", 
                                    facecolor='lightblue', 
                                    edgecolor='black', linewidth=2)
        ax.add_patch(dataset_box)
        ax.text(0.5, 0.875, 'Dataset Complet\n21,165 images', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Arrow down
        ax.arrow(0.5, 0.75, 0, -0.1, head_width=0.03, head_length=0.02, fc='black', ec='black', linewidth=2)
        
        # Model box
        model_box = FancyBboxPatch((0.2, 0.5), 0.6, 0.15, 
                                  boxstyle="round,pad=0.02", 
                                  facecolor='lightcoral', 
                                  edgecolor='black', linewidth=2)
        ax.add_patch(model_box)
        ax.text(0.5, 0.575, 'Modèles Classification Binaire', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Arrow down
        ax.arrow(0.5, 0.45, 0, -0.1, head_width=0.03, head_length=0.02, fc='black', ec='black', linewidth=2)
        
        # Output boxes
        covid_box = FancyBboxPatch((0.1, 0.15), 0.35, 0.2, 
                                  boxstyle="round,pad=0.02", 
                                  facecolor='#ff4444', 
                                  edgecolor='black',
                                  alpha=0.8, linewidth=2)
        ax.add_patch(covid_box)
        ax.text(0.275, 0.25, 'COVID-19\n3,616 images', 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        non_covid_box = FancyBboxPatch((0.55, 0.15), 0.35, 0.2, 
                                      boxstyle="round,pad=0.02", 
                                      facecolor='#44aa44', 
                                      edgecolor='black',
                                      alpha=0.8, linewidth=2)
        ax.add_patch(non_covid_box)
        ax.text(0.725, 0.25, 'Non-COVID\n17,549 images', 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    elif strategy == "Classificaiton Binaire - Étape 1 (Sains vs Malades)":
        # Dataset box
        dataset_box = FancyBboxPatch((0.2, 0.8), 0.6, 0.15, 
                                    boxstyle="round,pad=0.02", 
                                    facecolor='lightblue', 
                                    edgecolor='black', linewidth=2)
        ax.add_patch(dataset_box)
        ax.text(0.5, 0.875, 'Dataset Complet\n21,165 images', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Arrow down
        ax.arrow(0.5, 0.75, 0, -0.1, head_width=0.03, head_length=0.02, fc='black', ec='black', linewidth=2)
        
        # Model box
        model_box = FancyBboxPatch((0.2, 0.5), 0.6, 0.15, 
                                  boxstyle="round,pad=0.02", 
                                  facecolor='lightcoral', 
                                  edgecolor='black', linewidth=2)
        ax.add_patch(model_box)
        ax.text(0.5, 0.575, 'Modèles Étape 1\nSains vs Malades', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Arrow down
        ax.arrow(0.5, 0.45, 0, -0.1, head_width=0.03, head_length=0.02, fc='black', ec='black', linewidth=2)
        
        # Output boxes
        sain_box = FancyBboxPatch((0.1, 0.15), 0.35, 0.2, 
                                 boxstyle="round,pad=0.02", 
                                 facecolor='#44ff44', 
                                 edgecolor='black',
                                 alpha=0.8, linewidth=2)
        ax.add_patch(sain_box)
        ax.text(0.275, 0.25, 'SAINS\n10,192 images\n→ ARRÊT', 
                ha='center', va='center', fontsize=12, fontweight='bold', color='black')
        
        malade_box = FancyBboxPatch((0.55, 0.15), 0.35, 0.2, 
                                   boxstyle="round,pad=0.02", 
                                   facecolor='#ffaa44', 
                                   edgecolor='black',
                                   alpha=0.8, linewidth=2)
        ax.add_patch(malade_box)
        ax.text(0.725, 0.25, 'MALADES\n10,973 images\n→ Étape 2', 
                ha='center', va='center', fontsize=12, fontweight='bold', color='black')
    
    elif strategy == "Classification Multiclasses - Étape 2 (3 sorties)":
        ax.set_title("Stratégie Hiérarchique - Étape 2: Classification des Pathologies", fontsize=16, fontweight='bold', pad=20)
        
        # Input from step 1
        input_box = FancyBboxPatch((0.2, 0.8), 0.6, 0.15, 
                                  boxstyle="round,pad=0.02", 
                                  facecolor='#ffaa44', 
                                  edgecolor='black',
                                  alpha=0.8, linewidth=2)
        ax.add_patch(input_box)
        ax.text(0.5, 0.875, 'Images "Malade" (Étape 1)\n10,973 images', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Arrow down
        ax.arrow(0.5, 0.75, 0, -0.1, head_width=0.03, head_length=0.02, fc='black', ec='black', linewidth=2)
        
        # Model box
        model_box = FancyBboxPatch((0.2, 0.5), 0.6, 0.15, 
                                  boxstyle="round,pad=0.02", 
                                  facecolor='lightcoral', 
                                  edgecolor='black', linewidth=2)
        ax.add_patch(model_box)
        ax.text(0.5, 0.575, 'Modèle CNN - Étape 2\nClassification 3 Pathologies', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Arrow down
        ax.arrow(0.5, 0.45, 0, -0.1, head_width=0.03, head_length=0.02, fc='black', ec='black', linewidth=2)
        
        # Output boxes
        diseases = [('COVID-19', '3,616'), ('Lung_Opacity', '6,012'), ('Viral Pneumonia', '1,345')]
        colors = ['#ff4444', '#4444ff', '#aa44ff']
        for i, ((disease, count), color) in enumerate(zip(diseases, colors)):
            x_pos = 0.05 + i * 0.3
            disease_box = FancyBboxPatch((x_pos, 0.15), 0.25, 0.2, 
                                        boxstyle="round,pad=0.02", 
                                        facecolor=color, 
                                        edgecolor='black',
                                        alpha=0.8, linewidth=2)
            ax.add_patch(disease_box)
            ax.text(x_pos + 0.125, 0.25, f'{disease}\n{count}', 
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    elif strategy == "Classificaiton Binaire - Étape 2 (COVID vs Non-COVID)":
        ax.set_title("Stratégie Hiérarchique - Alternative: COVID vs Autres Pathologies", fontsize=16, fontweight='bold', pad=20)
        
        # Input from step 1
        input_box = FancyBboxPatch((0.2, 0.8), 0.6, 0.15, 
                                  boxstyle="round,pad=0.02", 
                                  facecolor='#ffaa44', 
                                  edgecolor='black',
                                  alpha=0.8, linewidth=2)
        ax.add_patch(input_box)
        ax.text(0.5, 0.875, 'Images "Malade" (Étape 1)\n10,973 images', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Arrow down
        ax.arrow(0.5, 0.75, 0, -0.1, head_width=0.03, head_length=0.02, fc='black', ec='black', linewidth=2)
        
        # Model box
        model_box = FancyBboxPatch((0.2, 0.5), 0.6, 0.15, 
                                  boxstyle="round,pad=0.02", 
                                  facecolor='lightcoral', 
                                  edgecolor='black', linewidth=2)
        ax.add_patch(model_box)
        ax.text(0.5, 0.575, 'Modèle CNN - Étape 2\nCOVID vs Autres Pathologies', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Arrow down
        ax.arrow(0.5, 0.45, 0, -0.1, head_width=0.03, head_length=0.02, fc='black', ec='black', linewidth=2)
        
        # Output boxes
        covid_box = FancyBboxPatch((0.1, 0.15), 0.35, 0.2, 
                                  boxstyle="round,pad=0.02", 
                                  facecolor='#ff4444', 
                                  edgecolor='black',
                                  alpha=0.8, linewidth=2)
        ax.add_patch(covid_box)
        ax.text(0.275, 0.25, 'COVID-19\n3,616 images', 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        other_box = FancyBboxPatch((0.55, 0.15), 0.35, 0.2, 
                                  boxstyle="round,pad=0.02", 
                                  facecolor='#6644ff', 
                                  edgecolor='black',
                                  alpha=0.8, linewidth=2)
        ax.add_patch(other_box)
        ax.text(0.725, 0.25, 'Autres Pathologies\n7,357 images', 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def get_strategy_description(strategy):
    descriptions = {
        "Classification Multiclasses (4 sorties)": """
        **Classification directe en 4 classes**
        
        **Avantages:**
        - Approche simple et directe
        - Un seul modèle à entraîner
        
        **Inconvénients:**
        - Déséquilibre important des classes
        - Complexité élevée pour le modèle
        """,
        
        "COVID vs Non-COVID": """
        **Classification binaire spécialisée COVID**
        
        **Avantages:**
        - Spécialisé pour la détection COVID
        - Plus simple (problème binaire)
        - Adapté au contexte pandémique
        
        **Inconvénients:**
        - Perte d'information sur les autres pathologies
        - Déséquilibre COVID (3,616) vs Non-COVID (17,549)
        - Pas de diagnostic différentiel
        """,
        
        "Hiérarchique - Étape 1 (Binary)": """
        **Première étape: Tri Sain vs Malade**
        
        **Avantages:**
        - Réduction drastique de la charge pour les cas sains
        - Classes plus équilibrées (10,192 vs 10,973)
        - Arrêt précoce pour 48% des cas
        
        **Utilité:**
        - Filtrage initial efficace
        - Réduction des faux positifs
        - Optimisation du temps de calcul
        """,
        
        "Hiérarchique - Étape 2 (Disease)": """
        **Deuxième étape: Classification des pathologies**
        
        **Avantages:**
        - Spécialisation sur les cas pathologiques
        - Meilleure discrimination entre maladies
        - Possibilité d'optimiser pour chaque pathologie
        
        **Défi:**
        - Toujours un déséquilibre entre les 3 classes
        - Nécessite deux modèles entraînés
        """,
        
        "Hiérarchique - COVID vs Disease": """
        **Alternative: COVID vs Autres pathologies**
        
        **Avantages:**
        - Focus sur la détection COVID
        - Classes plus équilibrées (3,616 vs 7,357)
        - Diagnostic différentiel COVID/Autres
        
        **Cas d'usage:**
        - Contexte pandémique
        - Besoin de distinguer COVID des autres pneumonies
        """
    }
    return descriptions.get(strategy, "")

def display_classification_strategies():
    """Fonction principale pour afficher les stratégies de classification"""
    
    st.header("1. Stratégies de Classification du Dataset")
    
    # Sélecteur de stratégie
    strategy = st.selectbox(
        "Choisissez la stratégie à visualiser:",
        [
            "Classification Multiclasses (4 sorties)",
            "Classificaiton Binaire (COVID vs Non-COVID)", 
            "Classificaiton Binaire - Étape 1 (Sains vs Malades)",
            "Classification Multiclasses - Étape 2 (3 sorties)",
            "Classificaiton Binaire - Étape 2 (COVID vs Non-COVID)"
        ]
    )
    
    # Affichage du diagramme correspondant
    fig = create_single_strategy_diagram(strategy)
    st.pyplot(fig)
    
    # Description de la stratégie
    description = get_strategy_description(strategy)
    if description:
        st.markdown(description)



# architecture de modèle 
def display_model_architectures():
    """Fonction pour afficher les architectures de modèles"""
    
    st.header("2. Architectures des Modèles")
    
    # Sélecteur d'approche
    approach = st.selectbox(
        "Choisissez l'approche d'entraînement:",
        ["Transfer Learning", "From Scratch", "Segmentation"]
    )
    
    # Sélecteur de modèle selon l'approche
    if approach == "Transfer Learning":
        model_options = ["DenseNet", "EfficientNet", "ResNet"]
        model_descriptions = {
            "DenseNet": """
            **DenseNet121 - Dense Convolutional Network**
            
            **Caractéristiques:**
       
            **Avantages:**
            """,
            
            "EfficientNet": """
            **EfficientNetB0 - Efficient Convolutional Neural Network**
            
            **Caractéristiques:**
            
            **Avantages:**
            """,
            
            "ResNet": """
            **ResNet50 - Residual Neural Network**
            
            **Caractéristiques:**
            
            **Avantages:**

            """
        }
    elif approach == "From Scratch":
        model_options = ["LeNet", "CNN Custom"]
        model_descriptions = {
            "LeNet": """
            **LeNet - Architecture Classique**
            
            **Caractéristiques:**
            - Architecture simple
            - Convolutions suivies de pooling
            - Environ 60K paramètres
            
            **Avantages:**
            - Rapidité d'entraînement
            - Faible complexité computationnelle
            """,
            
            "CNN Custom": """
            **CNN Custom - Architecture Personnalisée**
            
            **Caractéristiques:**
            - Architecture adaptée aux radiographies
            - Convolutions progressives (32→64→128→256)
            - Dropout pour la régularisation
            - Environ 2M paramètres
            
            **Avantages:**
            - Spécialisé pour notre tâche
            - Contrôle total de l'architecture
            """
        }
    else:  # Segmentation
        model_options = ["U-Net"]
        model_descriptions = {
            "U-Net": """
            **U-Net - Architecture pour Segmentation**
            
            **Caractéristiques:**
            - Architecture encoder-decoder en U
            - Skip connections pour préserver les détails
            - Sortie pixel-wise (masque binaire)
            - Environ 31M paramètres
            
            **Avantages:**
            - Excellent pour la segmentation médicale
            - Préservation des détails fins
            - Architecture éprouvée en imagerie médicale
            """
        }
    
    model_name = st.selectbox(f"Choisissez le modèle {approach}:", model_options)
    
    # Affichage du diagramme d'architecture
    fig = create_architecture_diagram(approach, model_name)
    st.pyplot(fig)
    
    # Description du modèle
    st.markdown(model_descriptions[model_name])


#parametres d'entrainement 
def display_training_parameters():
    """Fonction pour afficher les paramètres d'entraînement"""
    
    st.header("3. Paramètres d'Entraînement")
    
    # Sélecteur d'approche
    approach = st.selectbox(
        "Choisissez l'approche pour voir les paramètres:",
        ["Transfer Learning", "From Scratch", "Segmentation"],
        key="params_approach"
    )
    
    # Sélecteur de modèle selon l'approche
    if approach == "Transfer Learning":
        model_options = ["DenseNet", "EfficientNet", "ResNet"]
    elif approach == "From Scratch":
        model_options = ["LeNet", "CNN Custom"]
    else:  # Segmentation
        model_options = ["U-Net"]
    
    model_name = st.selectbox(f"Choisissez le modèle:", model_options, key="params_model")
    
    # Affichage du tableau des paramètres
    df_params = create_parameters_table(approach, model_name)
    
    # Utiliser des colonnes pour une meilleure présentation
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(f"Paramètres - {model_name}")
        st.dataframe(df_params, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Explications")
        
        explanations = {
            "Learning Rate": "Taux d'apprentissage - contrôle la vitesse de convergence",
            "Batch Size": "Taille du lot - nombre d'échantillons par mise à jour",
            "Epochs": "Nombre d'époques - passages complets sur les données",
            "Optimizer": "Optimiseur - algorithme de mise à jour des poids",
            "Loss Function": "Fonction de perte - mesure l'erreur du modèle",
            "Dropout": "Taux de dropout - régularisation pour éviter le surapprentissage",
            "Weight Decay": "Décroissance des poids - régularisation L2",
            "Scheduler": "Planificateur - ajuste le learning rate pendant l'entraînement",
            "Augmentation": "Augmentation des données - transformations pour enrichir le dataset",
            "Early Stopping": "Arrêt précoce - évite le surapprentissage"
        }
        
        for param in df_params["Paramètre"]:
            if param in explanations:
                st.write(f"**{param}**: {explanations[param]}")
    
    # Ajout d'informations contextuelles
    st.subheader("Justifications des Choix")
    
    if approach == "Transfer Learning":
        st.markdown("""
        **Stratégie Transfer Learning:**
        - **Learning Rate faible (1e-4)**: Les features pré-entraînées sont déjà bonnes
        - **Couches gelées**: Préservation des features ImageNet pertinentes
        - **Fine-tuning progressif**: Adaptation graduelle au domaine médical
        - **Dropout élevé (0.5)**: Prévention du surapprentissage sur les données médicales
        """)
    elif approach == "From Scratch":
        st.markdown("""
        **Stratégie From Scratch:**
        - **Learning Rate plus élevé (1e-3)**: Apprentissage complet des features
        - **Plus d'époques**: Temps nécessaire pour apprendre de zéro
        - **Augmentation intensive**: Compensation du manque de données pré-entraînées
        - **Régularisation forte**: Éviter le surapprentissage avec moins de données
        """)
    else:  # Segmentation
        st.markdown("""
        **Stratégie Segmentation:**
        - **Loss combinée**: Binary Crossentropy + Dice pour optimiser la segmentation
        - **Batch size réduit**: Images haute résolution (256x256)
        - **Augmentation élastique**: Simule les déformations anatomiques
        - **Patience élevée**: Convergence plus lente pour la segmentation précise
        """)


# metriques de performance
def display_evaluation_metrics():
    """Fonction pour afficher les métriques d'évaluation"""
    
    st.header("4. Métriques d'Évaluation")
    
    # Tabs pour organiser le contenu
    tab1, tab2, tab3 = st.tabs(["Métriques Classification", "Matrice de Confusion", "Métriques Segmentation"])
    
    with tab1:
        st.subheader("Métriques pour la Classification")
        
        # Affichage des formules des métriques
        fig_metrics = create_metrics_visualization()
        st.pyplot(fig_metrics)
        
        # Explications détaillées
        st.subheader("Importance de chaque métrique")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Accuracy (Exactitude)**
            - Métrique globale de performance
            -  Peut être trompeuse avec des classes déséquilibrées
            - Utile pour une vue d'ensemble
            
            **Precision (Précision)**
            - Important pour éviter les faux positifs
            - Focus sur la qualité des prédictions positives
            """)
        
        with col2:
            st.markdown("""
            **Recall/Sensitivity (Rappel/Sensibilité)**
            - Important pour éviter les faux négatifs
            - Critique en médecine (ne pas rater de cas positifs)
            - Focus sur la détection complète des cas positifs
            
            **F1-Score**
            - Équilibre entre Precision et Recall
            - Idéal pour les classes déséquilibrées
            - Métrique de référence pour notre projet
            """)
    
    with tab2:
        st.subheader("Matrice de Confusion")
        
        # Affichage de l'explication de la matrice de confusion
        fig_confusion = create_confusion_matrix_explanation()
        st.pyplot(fig_confusion)
        
        st.subheader("Interprétation en Contexte Médical")
        
        # Explication spécifique au contexte médical
        st.markdown("""
        **Dans le contexte du diagnostic COVID-19:**
        
        - **Vrai Positif (TP)**: Patient COVID correctement identifié 
          - Permet un traitement et isolement appropriés
        
        - **Faux Positif (FP)**: Patient sain diagnostiqué COVID 
          - Stress inutile, isolement non nécessaire
          - Surcharge du système de santé
        
        - **Faux Négatif (FN)**: Patient COVID non détecté 
          - **CRITIQUE**: Risque de propagation
          - Absence de traitement approprié
        
        - **Vrai Négatif (TN)**: Patient sain correctement identifié 
          - Évite l'anxiété et les procédures inutiles
        
        **En médecine, minimiser les FN prioritaire !**
        """)
    
    with tab3:
        st.subheader("Métriques pour la Segmentation")
        
        # Créer une visualisation pour les métriques de segmentation
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Dice Score
        ax1.set_title("Dice Score", fontsize=14, fontweight='bold')

def create_architecture_diagram(approach, model_name):
    """Créer un diagramme d'architecture pour un modèle spécifique"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    if approach == "Transfer Learning":
        if model_name == "DenseNet":
            ax.set_title("Architecture Transfer Learning - DenseNet121", fontsize=16, fontweight='bold', pad=20)
            
            # Input image
            input_box = FancyBboxPatch((0.05, 0.8), 0.15, 0.15, 
                                      boxstyle="round,pad=0.02", 
                                      facecolor='lightblue', 
                                      edgecolor='black', linewidth=2)
            ax.add_patch(input_box)
            ax.text(0.125, 0.875, 'Image\n224x224x3', ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Pre-trained DenseNet
            densenet_box = FancyBboxPatch((0.25, 0.7), 0.4, 0.3, 
                                         boxstyle="round,pad=0.02", 
                                         facecolor='lightgreen', 
                                         edgecolor='black', linewidth=2)
            ax.add_patch(densenet_box)
            ax.text(0.45, 0.85, 'DenseNet121\n(Pré-entraîné ImageNet)\n7.98M paramètres\nFeatures: 1024', 
                    ha='center', va='center', fontsize=11, fontweight='bold')
            
            # Frozen indicator
            ax.text(0.45, 0.72, '❄️ Couches gelées', ha='center', va='center', fontsize=9, color='blue')
            
            # Custom classifier
            classifier_box = FancyBboxPatch((0.7, 0.75), 0.25, 0.2, 
                                           boxstyle="round,pad=0.02", 
                                           facecolor='orange', 
                                           edgecolor='black', linewidth=2)
            ax.add_patch(classifier_box)
            ax.text(0.825, 0.85, 'Classificateur\nCustom\nDropout(0.5)\nDense(classes)', 
                    ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Output
            output_box = FancyBboxPatch((0.4, 0.4), 0.2, 0.15, 
                                       boxstyle="round,pad=0.02", 
                                       facecolor='lightcoral', 
                                       edgecolor='black', linewidth=2)
            ax.add_patch(output_box)
            ax.text(0.5, 0.475, 'Sortie\n(n classes)', ha='center', va='center', fontsize=11, fontweight='bold')
            
        elif model_name == "EfficientNet":
            ax.set_title("Architecture Transfer Learning - EfficientNetB0", fontsize=16, fontweight='bold', pad=20)
            
            # Input image
            input_box = FancyBboxPatch((0.05, 0.8), 0.15, 0.15, 
                                      boxstyle="round,pad=0.02", 
                                      facecolor='lightblue', 
                                      edgecolor='black', linewidth=2)
            ax.add_patch(input_box)
            ax.text(0.125, 0.875, 'Image\n224x224x3', ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Pre-trained EfficientNet
            efficientnet_box = FancyBboxPatch((0.25, 0.7), 0.4, 0.3, 
                                             boxstyle="round,pad=0.02", 
                                             facecolor='lightgreen', 
                                             edgecolor='black', linewidth=2)
            ax.add_patch(efficientnet_box)
            ax.text(0.45, 0.85, 'EfficientNetB0\n(Pré-entraîné ImageNet)\n5.29M paramètres\nFeatures: 1280', 
                    ha='center', va='center', fontsize=11, fontweight='bold')
            
            # Frozen indicator
            ax.text(0.45, 0.72, '❄️ Couches gelées', ha='center', va='center', fontsize=9, color='blue')
            
            # Custom classifier
            classifier_box = FancyBboxPatch((0.7, 0.75), 0.25, 0.2, 
                                           boxstyle="round,pad=0.02", 
                                           facecolor='orange', 
                                           edgecolor='black', linewidth=2)
            ax.add_patch(classifier_box)
            ax.text(0.825, 0.85, 'Classificateur\nCustom\nDropout(0.5)\nDense(classes)', 
                    ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Output
            output_box = FancyBboxPatch((0.4, 0.4), 0.2, 0.15, 
                                       boxstyle="round,pad=0.02", 
                                       facecolor='lightcoral', 
                                       edgecolor='black', linewidth=2)
            ax.add_patch(output_box)
            ax.text(0.5, 0.475, 'Sortie\n(n classes)', ha='center', va='center', fontsize=11, fontweight='bold')
            
        elif model_name == "ResNet":
            ax.set_title("Architecture Transfer Learning - ResNet50", fontsize=16, fontweight='bold', pad=20)
            
            # Input image
            input_box = FancyBboxPatch((0.05, 0.8), 0.15, 0.15, 
                                      boxstyle="round,pad=0.02", 
                                      facecolor='lightblue', 
                                      edgecolor='black', linewidth=2)
            ax.add_patch(input_box)
            ax.text(0.125, 0.875, 'Image\n224x224x3', ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Pre-trained ResNet
            resnet_box = FancyBboxPatch((0.25, 0.7), 0.4, 0.3, 
                                       boxstyle="round,pad=0.02", 
                                       facecolor='lightgreen', 
                                       edgecolor='black', linewidth=2)
            ax.add_patch(resnet_box)
            ax.text(0.45, 0.85, 'ResNet50\n(Pré-entraîné ImageNet)\n25.56M paramètres\nFeatures: 2048', 
                    ha='center', va='center', fontsize=11, fontweight='bold')
            
            # Frozen indicator
            ax.text(0.45, 0.72, '❄️ Couches gelées', ha='center', va='center', fontsize=9, color='blue')
            
            # Custom classifier
            classifier_box = FancyBboxPatch((0.7, 0.75), 0.25, 0.2, 
                                           boxstyle="round,pad=0.02", 
                                           facecolor='orange', 
                                           edgecolor='black', linewidth=2)
            ax.add_patch(classifier_box)
            ax.text(0.825, 0.85, 'Classificateur\nCustom\nDropout(0.5)\nDense(classes)', 
                    ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Output
            output_box = FancyBboxPatch((0.4, 0.4), 0.2, 0.15, 
                                       boxstyle="round,pad=0.02", 
                                       facecolor='lightcoral', 
                                       edgecolor='black', linewidth=2)
            ax.add_patch(output_box)
            ax.text(0.5, 0.475, 'Sortie\n(n classes)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    elif approach == "From Scratch":
        if model_name == "LeNet":
            ax.set_title("Architecture From Scratch - LeNet", fontsize=16, fontweight='bold', pad=20)
            
            layers = [
                ("Input\n224x224x1", 0.05, 'lightblue'),
                ("Conv2D\n6@5x5", 0.15, 'lightgreen'),
                ("MaxPool\n2x2", 0.25, 'yellow'),
                ("Conv2D\n16@5x5", 0.35, 'lightgreen'),
                ("MaxPool\n2x2", 0.45, 'yellow'),
                ("Flatten", 0.55, 'orange'),
                ("Dense\n120", 0.65, 'lightcoral'),
                ("Dense\n84", 0.75, 'lightcoral'),
                ("Output\nn classes", 0.85, 'red')
            ]
            
            for i, (layer_text, x_pos, color) in enumerate(layers):
                layer_box = FancyBboxPatch((x_pos, 0.4), 0.08, 0.2, 
                                          boxstyle="round,pad=0.01", 
                                          facecolor=color, 
                                          edgecolor='black', linewidth=1)
                ax.add_patch(layer_box)
                ax.text(x_pos + 0.04, 0.5, layer_text, ha='center', va='center', fontsize=8, fontweight='bold')
                
                if i < len(layers) - 1:
                    ax.arrow(x_pos + 0.08, 0.5, 0.06, 0, head_width=0.02, head_length=0.01, fc='black', ec='black')
            
        elif model_name == "CNN Custom":
            ax.set_title("Architecture From Scratch - CNN Custom", fontsize=16, fontweight='bold', pad=20)
            
            layers = [
                ("Input\n224x224x3", 0.05, 'lightblue'),
                ("Conv2D\n32@3x3", 0.13, 'lightgreen'),
                ("Conv2D\n64@3x3", 0.21, 'lightgreen'),
                ("MaxPool\n2x2", 0.29, 'yellow'),
                ("Conv2D\n128@3x3", 0.37, 'lightgreen'),
                ("Conv2D\n256@3x3", 0.45, 'lightgreen'),
                ("MaxPool\n2x2", 0.53, 'yellow'),
                ("Flatten", 0.61, 'orange'),
                ("Dense\n512", 0.69, 'lightcoral'),
                ("Dropout\n0.5", 0.77, 'pink'),
                ("Output\nn classes", 0.85, 'red')
            ]
            
            for i, (layer_text, x_pos, color) in enumerate(layers):
                layer_box = FancyBboxPatch((x_pos, 0.4), 0.06, 0.2, 
                                          boxstyle="round,pad=0.01", 
                                          facecolor=color, 
                                          edgecolor='black', linewidth=1)
                ax.add_patch(layer_box)
                ax.text(x_pos + 0.03, 0.5, layer_text, ha='center', va='center', fontsize=7, fontweight='bold')
                
                if i < len(layers) - 1:
                    ax.arrow(x_pos + 0.06, 0.5, 0.01, 0, head_width=0.02, head_length=0.005, fc='black', ec='black')
    
    elif approach == "Segmentation":
        ax.set_title("Architecture Segmentation - U-Net (From Scratch)", fontsize=16, fontweight='bold', pad=20)
        
        # Encoder path
        encoder_layers = [
            ("Input\n256x256x1", 0.05, 0.8, 'lightblue'),
            ("Conv Block\n64", 0.15, 0.7, 'lightgreen'),
            ("Conv Block\n128", 0.25, 0.6, 'lightgreen'),
            ("Conv Block\n256", 0.35, 0.5, 'lightgreen'),
            ("Conv Block\n512", 0.45, 0.4, 'lightgreen'),
        ]
        
        # Decoder path
        decoder_layers = [
            ("Conv Block\n256", 0.55, 0.5, 'orange'),
            ("Conv Block\n128", 0.65, 0.6, 'orange'),
            ("Conv Block\n64", 0.75, 0.7, 'orange'),
            ("Output\n256x256x1", 0.85, 0.8, 'red'),
        ]
        
        # Draw encoder
        for i, (layer_text, x_pos, y_pos, color) in enumerate(encoder_layers):
            layer_box = FancyBboxPatch((x_pos, y_pos), 0.08, 0.1, 
                                      boxstyle="round,pad=0.01", 
                                      facecolor=color, 
                                      edgecolor='black', linewidth=1)
            ax.add_patch(layer_box)
            ax.text(x_pos + 0.04, y_pos + 0.05, layer_text, ha='center', va='center', fontsize=8, fontweight='bold')
            
            if i < len(encoder_layers) - 1:
                ax.arrow(x_pos + 0.08, y_pos + 0.05, 0.06, -0.08, head_width=0.01, head_length=0.01, fc='black', ec='black')
        
        # Draw decoder
        for i, (layer_text, x_pos, y_pos, color) in enumerate(decoder_layers):
            layer_box = FancyBboxPatch((x_pos, y_pos), 0.08, 0.1, 
                                      boxstyle="round,pad=0.01", 
                                      facecolor=color, 
                                      edgecolor='black', linewidth=1)
            ax.add_patch(layer_box)
            ax.text(x_pos + 0.04, y_pos + 0.05, layer_text, ha='center', va='center', fontsize=8, fontweight='bold')
            
            if i < len(decoder_layers) - 1:
                ax.arrow(x_pos + 0.08, y_pos + 0.05, 0.06, 0.08, head_width=0.01, head_length=0.01, fc='black', ec='black')
        
        # Skip connections
        skip_connections = [(0.19, 0.75, 0.71, 0.75), (0.29, 0.65, 0.61, 0.65), (0.39, 0.55, 0.51, 0.55)]
        for x1, y1, x2, y2 in skip_connections:
            ax.plot([x1, x2], [y1, y2], 'r--', linewidth=2, alpha=0.7)
            ax.text((x1 + x2) / 2, y1 + 0.02, 'Skip Connection', ha='center', va='bottom', fontsize=7, color='red')
    
    # Arrows for transfer learning
    if approach == "Transfer Learning":
        # Input to pre-trained model
        ax.arrow(0.2, 0.875, 0.04, 0, head_width=0.02, head_length=0.01, fc='black', ec='black', linewidth=2)
        # Pre-trained to classifier
        ax.arrow(0.65, 0.85, 0.04, 0, head_width=0.02, head_length=0.01, fc='black', ec='black', linewidth=2)
        # Classifier to output
        ax.arrow(0.7, 0.8, -0.18, -0.25, head_width=0.02, head_length=0.01, fc='black', ec='black', linewidth=2)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 1)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_parameters_table(approach, model_name):
    """Créer un tableau des paramètres d'entraînement"""
    
    # Paramètres selon le modèle
    params_data = {
        "Transfer Learning": {
            "DenseNet": {
                "Learning Rate": "1e-4",
                "Batch Size": "32",
                "Epochs": "50",
                "Optimizer": "Adam",
                "Loss Function": "Categorical Crossentropy",
                "Dropout": "0.5",
                "Weight Decay": "1e-4",
                "Scheduler": "ReduceLROnPlateau",
                "Augmentation": "Rotation, Zoom, Flip",
                "Early Stopping": "Patience=10"
            },
            "EfficientNet": {
                "Learning Rate": "1e-4",
                "Batch Size": "32",
                "Epochs": "50",
                "Optimizer": "Adam",
                "Loss Function": "Categorical Crossentropy",
                "Dropout": "0.5",
                "Weight Decay": "1e-4",
                "Scheduler": "ReduceLROnPlateau",
                "Augmentation": "Rotation, Zoom, Flip",
                "Early Stopping": "Patience=10"
            },
            "ResNet": {
                "Learning Rate": "1e-4",
                "Batch Size": "32",
                "Epochs": "50",
                "Optimizer": "Adam",
                "Loss Function": "Categorical Crossentropy",
                "Dropout": "0.5",
                "Weight Decay": "1e-4",
                "Scheduler": "ReduceLROnPlateau",
                "Augmentation": "Rotation, Zoom, Flip",
                "Early Stopping": "Patience=10"
            }
        },
        "From Scratch": {
            "LeNet": {
                "Learning Rate": "1e-3",
                "Batch Size": "64",
                "Epochs": "100",
                "Optimizer": "Adam",
                "Loss Function": "Categorical Crossentropy",
                "Dropout": "0.3",
                "Weight Decay": "1e-5",
                "Scheduler": "StepLR",
                "Augmentation": "Basic (Flip, Rotation)",
                "Early Stopping": "Patience=15"
            },
            "CNN Custom": {
                "Learning Rate": "1e-3",
                "Batch Size": "32",
                "Epochs": "100",
                "Optimizer": "Adam",
                "Loss Function": "Categorical Crossentropy",
                "Dropout": "0.5",
                "Weight Decay": "1e-4",
                "Scheduler": "StepLR",
                "Augmentation": "Advanced (Rotation, Zoom, Flip, Brightness)",
                "Early Stopping": "Patience=15"
            }
        },
        "Segmentation": {
            "U-Net": {
                "Learning Rate": "1e-4",
                "Batch Size": "16",
                "Epochs": "75",
                "Optimizer": "Adam",
                "Loss Function": "Binary Crossentropy + Dice",
                "Dropout": "0.3",
                "Weight Decay": "1e-5",
                "Scheduler": "ReduceLROnPlateau",
                "Augmentation": "Rotation, Zoom, Flip, Elastic",
                "Early Stopping": "Patience=12"
            }
        }
    }
    
    if approach == "Segmentation":
        model_name = "U-Net"
    
    params = params_data[approach][model_name]
    
    # Créer le DataFrame
    df = pd.DataFrame({
        "Paramètre": list(params.keys()),
        "Valeur": list(params.values())
    })
    
    return df

def create_metrics_visualization():
    """Créer une visualisation des métriques d'évaluation"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy
    ax1.set_title("Accuracy", fontsize=14, fontweight='bold')
    accuracy_formula = r'$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$'
    ax1.text(0.5, 0.7, accuracy_formula, ha='center', va='center', fontsize=16, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax1.text(0.5, 0.3, "Proportion de prédictions correctes\nsur l'ensemble des prédictions", 
             ha='center', va='center', fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Precision
    ax2.set_title("Precision", fontsize=14, fontweight='bold')
    precision_formula = r'$Precision = \frac{TP}{TP + FP}$'
    ax2.text(0.5, 0.7, precision_formula, ha='center', va='center', fontsize=16,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax2.text(0.5, 0.3, "Proportion de vrais positifs\nparmi les prédictions positives", 
             ha='center', va='center', fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Recall (Sensitivity)
    ax3.set_title("Recall (Sensibilité)", fontsize=14, fontweight='bold')
    recall_formula = r'$Recall = \frac{TP}{TP + FN}$'
    ax3.text(0.5, 0.7, recall_formula, ha='center', va='center', fontsize=16,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="orange"))
    ax3.text(0.5, 0.3, "Proportion de vrais positifs\ncorrectement identifiés", 
             ha='center', va='center', fontsize=12)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # F1-Score
    ax4.set_title("F1-Score", fontsize=14, fontweight='bold')
    f1_formula = r'$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$'
    ax4.text(0.5, 0.7, f1_formula, ha='center', va='center', fontsize=14,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    ax4.text(0.5, 0.3, "Moyenne harmonique entre\nPrécision et Rappel", 
             ha='center', va='center', fontsize=12)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    return fig

def create_confusion_matrix_explanation():
    """Créer une explication visuelle de la matrice de confusion"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Titre
    ax.set_title("Matrice de Confusion - Explication", fontsize=16, fontweight='bold', pad=20)
    
    # Créer la matrice
    matrix_data = np.array([[85, 15], [10, 90]])
    
    # Couleurs pour la matrice
    colors = ['lightgreen', 'lightcoral', 'lightcoral', 'lightgreen']
    
    # Dessiner la matrice
    for i in range(2):
        for j in range(2):
            # Rectangle pour chaque case
            rect = FancyBboxPatch((0.3 + j*0.2, 0.5 + (1-i)*0.2), 0.18, 0.18,
                                 boxstyle="round,pad=0.02",
                                 facecolor=colors[i*2 + j],
                                 edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Valeur dans la case
            ax.text(0.39 + j*0.2, 0.59 + (1-i)*0.2, str(matrix_data[i, j]),
                   ha='center', va='center', fontsize=20, fontweight='bold')
            
            # Labels des cases
            if i == 0 and j == 0:
                ax.text(0.39 + j*0.2, 0.52 + (1-i)*0.2, 'TP\n(Vrai Positif)',
                       ha='center', va='center', fontsize=8, fontweight='bold')
            elif i == 0 and j == 1:
                ax.text(0.39 + j*0.2, 0.52 + (1-i)*0.2, 'FP\n(Faux Positif)',
                       ha='center', va='center', fontsize=8, fontweight='bold')
            elif i == 1 and j == 0:
                ax.text(0.39 + j*0.2, 0.52 + (1-i)*0.2, 'FN\n(Faux Négatif)',
                       ha='center', va='center', fontsize=8, fontweight='bold')
            else:
                ax.text(0.39 + j*0.2, 0.52 + (1-i)*0.2, 'TN\n(Vrai Négatif)',
                       ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Labels des axes
    ax.text(0.2, 0.69, 'Réel', ha='center', va='center', fontsize=14, fontweight='bold', rotation=90)
    ax.text(0.49, 0.45, 'Prédit', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Labels des classes
    ax.text(0.25, 0.69, 'Positif', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.25, 0.49, 'Négatif', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.39, 0.42, 'Positif', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.59, 0.42, 'Négatif', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Explication à droite
    ax.text(0.75, 0.8, "Interprétation:", ha='left', va='top', fontsize=14, fontweight='bold')
    ax.text(0.75, 0.75, "• TP: Modèle prédit positif, réalité positive ✓", ha='left', va='top', fontsize=10)
    ax.text(0.75, 0.7, "• FP: Modèle prédit positif, réalité négative ✗", ha='left', va='top', fontsize=10)
    ax.text(0.75, 0.65, "• FN: Modèle prédit négatif, réalité positive ✗", ha='left', va='top', fontsize=10)
    ax.text(0.75, 0.6, "• TN: Modèle prédit négatif, réalité négative ✓", ha='left', va='top', fontsize=10)
    
    ax.text(0.75, 0.5, "Calculs:", ha='left', va='top', fontsize=14, fontweight='bold')
    ax.text(0.75, 0.45, f"Accuracy = (85+90)/(85+15+10+90) = 87.5%", ha='left', va='top', fontsize=10)
    ax.text(0.75, 0.4, f"Precision = 85/(85+15) = 85%", ha='left', va='top', fontsize=10)
    ax.text(0.75, 0.35, f"Recall = 85/(85+10) = 89.5%", ha='left', va='top', fontsize=10)
    ax.text(0.75, 0.3, f"F1-Score = 2×(85×89.5)/(85+89.5) = 87.2%", ha='left', va='top', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.9)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

# Fonction principale pour orchestrer toutes les sections
def display_complete_modeling():
    """Fonction principale pour afficher toute la section modélisation"""
    
    # Section 1: Stratégies de classification
    display_classification_strategies()
    
    st.markdown("---")
    
    # Section 2: Architectures
    display_model_architectures()
    
    st.markdown("---")
    
    # Section 3: Paramètres d'entraînement
    display_training_parameters()
    
    st.markdown("---")
    
    # Section 4: Métriques d'évaluation
    display_evaluation_metrics()