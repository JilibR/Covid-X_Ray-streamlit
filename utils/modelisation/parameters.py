"""
Gestion et affichage des paramètres d'entraînement
"""

import streamlit as st
import pandas as pd


def display_training_parameters():
    """Affiche la section complète des paramètres d'entraînement"""
    st.header("3. Paramètres d'Entraînement")
    
    # Sélection de l'approche
    approach = st.selectbox(
        "Approche d'entraînement:",
        ["Transfer Learning", "From Scratch", "Segmentation"],
        key="params_approach"
    )
    
    # Sélection du modèle
    if approach == "Transfer Learning":
        model_options = ["DenseNet", "EfficientNet", "ResNet"]
    elif approach == "From Scratch":
        model_options = ["LeNet", "CNN Custom"]
    else:
        model_options = ["U-Net"]
    
    model_name = st.selectbox("Modèle:", model_options, key="params_model")
    
    # Affichage selon l'approche
    if approach == "Transfer Learning":
        _display_transfer_learning_details()
    elif approach == "From Scratch":
        _display_from_scratch_summary()
    else:
        _display_segmentation_details()
    
    st.markdown("---")



def _display_transfer_learning_details():
    """Affiche les détails du Transfer Learning"""
    
    st.subheader("Stratégie Transfer Learning en 3 Phases")
    
    # Phase 1
    st.markdown("#### Phase 1 - Warm-up (15 epochs)")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.info("""
        **Configuration:**
        - Classifier seul entraînable
        - Backbone gelé (frozen)
        - LR = 1e-3 (plus élevé)
        """)
    with col2:
        st.markdown("""
        **Objectif:** Adapter le classifier à notre tâche
        
        Le backbone pré-entraîné sur ImageNet est figé. Seule la dernière couche 
        (classifier) apprend à associer les features ImageNet à nos classes.
        """)
    
    # Phase 2
    st.markdown("#### Phase 2 - Fine-tuning partiel (30 epochs)")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.info("""
        **Configuration:**
        - Dégel des dernières couches
        - LR = 5e-4 (réduit)
        - Features bas niveau toujours gelées
        """)
    with col2:
        st.markdown("""
        **Objectif:** Ajuster les features haut niveau
        
        On permet aux couches profondes du backbone de s'adapter légèrement aux 
        caractéristiques spécifiques des images médicales (textures propres aux pathologies).
        """)
    
    # Phase 3
    st.markdown("#### Phase 3 - Fine-tuning complet (20 epochs)")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.info("""
        **Configuration:**
        - Tout le réseau entraînable
        - LR = Très faible (1e-5)
        - Ajustements fins globaux
        """)
    with col2:
        st.markdown("""
        **Objectif:** Optimisation fine globale
        
        Ajustement de l'ensemble du réseau pour maximiser les performances 
        finales sans dégrader les features pré-apprises.
        """)
    
    # Rationale globale
    st.success("""
    **Avantages:**
    
    - **Learning Rate faible:** Features pré-entraînées déjà de bonne qualité
    - **Dropout élevé (0.5):** Prévention du surapprentissage sur données médicales.
    - **Entraînement progressif:** Adaptation au domaine médical sans "casser" les features ImageNet
    - **ReduceLROnPlateau:** Réduction automatique si plateau de performance (factor=0.5, patience=5)
    """)


def _display_segmentation_details():
    """Affiche les détails de la segmentation U-Net"""
    
    st.subheader("Stratégie Segmentation U-Net")
    
    # Spécificités
    st.markdown("#### Spécificités de la Segmentation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Configuration:**
        - Loss: BCE + Dice (α=0.5)
        - Batch size: 16
        - Résolution: 256×256
        - Epochs: 75
        """)
    
    with col2:
        st.markdown("""
        **Pourquoi ces choix:**
        - Loss combinée optimise précision pixel ET forme globale
        - Batch réduit car images haute résolution
        """)
    
    # Loss fonction détaillée
    st.markdown("#### BCE + Dice Loss (α=0.5)")
    st.success("""
    **Combinaison optimale pour la segmentation:**
    
    - **Binary Crossentropy (BCE):** Précision pixel par pixel, pénalise chaque erreur localement
    - **Dice Loss:** Mesure la similitude des formes globales, favorise la cohérence de la région segmentée
    - **α=0.5:** Équilibre entre précision locale et cohérence globale
    
    Cette combinaison est particulièrement efficace pour prédire la région blanche beaucoup plus petite que la région noire.
    """)

def _display_from_scratch_summary():
    """Affiche le résumé du fonctionnement From Scratch"""
    
    st.subheader("Fonctionnement de l'Entraînement From Scratch")
    
    # Phase unique
    st.info("""
    **Phase Unique d'Entraînement**
    
    Contrairement au Transfer Learning (3 phases), une seule phase de **100 epochs** où 
    tous les poids sont appris depuis zéro.
    """)
    
    # Système de callbacks
    with st.container():
        st.success("""
        **Système de Callbacks Intelligent**
        
        Optimisation automatique pendant l'entraînement grâce à plusieurs callbacks:
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - **EarlyStopping**  
              `patience=12, min_delta=0.0005`  
              Arrêt si val_f1 ne progresse pas
              
            - **ReduceLROnPlateau**  
              `factor=0.6, patience=8`  
              Réduit LR de 40% si plateau
            """)
        
        with col2:
            st.markdown("""
            - **ModelCheckpoint**  
              Sauvegarde auto du meilleur modèle  
              (basé sur val_f1)
              
            - **CSVLogger**  
              Enregistre toutes les métriques  
              pour analyse post-entraînement
            """)
    
    # Paramètres adaptés
    st.markdown("### Paramètres Clés Adaptés")
    
    params_cols = st.columns(4)
    with params_cols[0]:
        st.metric("Learning Rate", "1e-3", help="Plus élevé car apprentissage complet")
    with params_cols[1]:
        st.metric("Batch Size", "32", help="Selon complexité du modèle")
    with params_cols[2]:
        st.metric("Augmentation", "Intensive", help="Compense absence pré-entraînement et le désequilibre classes")
    with params_cols[3]:
        st.metric("Dropout", "0.1→0.5", help="Progressif selon profondeur")
    
    # Différences vs Transfer Learning
    st.warning("""
    **Différences clés vs Transfer Learning:**
    
    | Aspect | From Scratch | Transfer Learning |
    |--------|--------------|-------------------|
    | Durée | 100 epochs | 50 epochs (3 phases) |
    | Learning Rate initial | 1e-3 (élevé) | 1e-4 → 5e-5 (faible) |
    | Sensibilité données | Très élevée | Moyenne |
    | Dépendance ImageNet | Aucune | Forte |
    | Meilleur pour | Données très spécifiques | Données génériques |
    """)
    
    # Déroulement détaillé
    with st.expander("Déroulement Détaillé de l'Entraînement", expanded=False):
        st.markdown("""
        **Étapes d'exécution:**
        
        1. **Initialisation (t=0)**
           - Poids aléatoires
           - Optimiseur Adam configuré (lr=1e-3)
           - Callbacks initialisés
        
        2. **Boucle d'entraînement (epochs 1-100)**
           - Pour chaque batch:
             * Forward pass
             * Calcul loss (CrossEntropy)
             * Backward pass + gradient clipping
             * Mise à jour poids
           - Validation après chaque epoch
           - Calcul F1-score macro (métrique principale)
        
        3. **Gestion automatique**
           - Si val_f1 stagne 8 epochs → LR × 0.6
           - Si val_f1 ne progresse pas 12 epochs → Arrêt
           - Sauvegarde continue du meilleur modèle
        
        4. **Fin d'entraînement**
           - Restauration automatique des meilleurs poids
           - Génération du log CSV complet
           - Retour du modèle optimisé
        """)