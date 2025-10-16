"""
Gestion et affichage des métriques d'évaluation
"""

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def display_evaluation_metrics():
    """Affiche la section complète des métriques d'évaluation"""
    st.header("4. Métriques d'Évaluation")
    
    # Tabs pour organiser
    tab1, tab2 = st.tabs([
        "Métriques Classification",
        "Métriques Segmentation"
    ])
    
    with tab1:
        _display_classification_metrics()
    
    with tab2:
        _display_segmentation_metrics()
    
    st.markdown("---")


def _display_classification_metrics():
    """Affiche les métriques de classification"""
    st.subheader("Métriques pour la Classification")
    
    # Visualisation des formules
    fig = _create_metrics_formulas()
    st.pyplot(fig)
    plt.close(fig)
    
    # Explications détaillées
    st.subheader("Importance de chaque métrique")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
**Accuracy (Exactitude)**
- Métrique globale de performance
        
**Precision (Précision)**
- Qualité des prédictions positives
- Ici: éviter l'isolement inutile des patients non atteints de la COVID-19
        """)
    
    with col2:
        st.markdown("""
**Recall/Sensitivity (Rappel/Sensibilité)**
- Capacité à détecter tous les positifs
- Éviter de rater des cas positifs
        
**F1-Score**
- Équilibre Precision/Recall
- `2 × (Precision × Recall) / (Precision + Recall)`
- Idéal pour classes déséquilibrées
- **Métrique principale du projet**
        """)


def _display_segmentation_metrics():
    """Affiche les métriques de segmentation"""
    st.subheader("Métriques pour la Segmentation")
    
    # Visualisation
    fig = _create_segmentation_metrics_diagram()
    st.pyplot(fig)
    plt.close(fig)
    
    # Explications
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
**Dice Score (Coefficient de Dice)**
```
Dice = 2 × |A ∩ B| / (|A| + |B|)
```
- Mesure similitude entre masque prédit et réel
- Valeurs: 0 (aucune overlap) à 1 (parfait)
- Équivalent à F1-Score pour segmentation
- **Métrique principale pour ce projet**
        """)
    
    with col2:
        st.markdown("""
**IoU (Intersection over Union)**
```
IoU = |A ∩ B| / |A ∪ B|
```
- Ratio zone commune / zone totale
- Plus strict que Dice Score
- Utilisé en computer vision
- Valeurs: 0 à 1
        """)
    
    # Relation entre les deux
    st.info("""
    **Relation Dice ↔ IoU**
    
    Ces deux métriques sont mathématiquement liées par la formule:
    ```
    Dice = 2 × IoU / (1 + IoU)
    ```
    
    **Implications:**
    - Le Dice Score est toujours **≥** IoU (plus optimiste)
    - Pour IoU = 0.9644 → Dice ≈ 0.9819
    - Le Dice donne plus de poids à la zone de chevauchement
    - L'IoU pénalise davantage les faux positifs et faux négatifs
    """)


def _create_metrics_formulas():
    """Crée la visualisation des formules de métriques"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_data = [
        (ax1, "Accuracy", r'$\frac{TP + TN}{TP + TN + FP + FN}$',
         "Proportion de prédictions\ncorrectes globales", "lightblue"),
        (ax2, "Precision", r'$\frac{TP}{TP + FP}$',
         "Proportion de vrais positifs\nparmi les prédictions positives", "lightgreen"),
        (ax3, "Recall", r'$\frac{TP}{TP + FN}$',
         "Proportion de vrais positifs\ncorrectement identifiés", "orange"),
        (ax4, "F1-Score", r'$2 \times \frac{Precision \times Recall}{Precision + Recall}$',
         "Moyenne harmonique\nPrecision et Recall", "lightcoral")
    ]
    
    for ax, title, formula, desc, color in metrics_data:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.text(0.5, 0.65, formula, ha='center', va='center',
               fontsize=16, bbox=dict(boxstyle="round,pad=0.4", facecolor=color))
        ax.text(0.5, 0.25, desc, ha='center', va='center', fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def _create_segmentation_metrics_diagram():
    """Crée le diagramme des métriques de segmentation"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Dice Score
    ax1.set_title("Dice Score", fontsize=14, fontweight='bold', pad=10)
    dice_formula = r'$Dice = \frac{2 \times |A \cap B|}{|A| + |B|}$'
    ax1.text(0.5, 0.65, dice_formula, ha='center', va='center',
            fontsize=16, bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue"))
    ax1.text(0.5, 0.25, "Similitude entre masques\nValeurs: 0 à 1\n(Métrique principale)",
            ha='center', va='center', fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # IoU
    ax2.set_title("IoU (Intersection over Union)", fontsize=14, fontweight='bold', pad=10)
    iou_formula = r'$IoU = \frac{|A \cap B|}{|A \cup B|}$'
    ax2.text(0.5, 0.65, iou_formula, ha='center', va='center',
            fontsize=16, bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen"))
    ax2.text(0.5, 0.25, "Ratio zone commune /\nzone totale\n(Plus strict)",
            ha='center', va='center', fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Relation Dice-IoU
    ax3.set_title("Relation Dice ↔ IoU", fontsize=14, fontweight='bold', pad=10)
    relation_formula = r'$Dice = \frac{2 \times IoU}{1 + IoU}$'
    ax3.text(0.5, 0.65, relation_formula, ha='center', va='center',
            fontsize=16, bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcoral"))
    ax3.text(0.5, 0.25, "Dice toujours ≥ IoU\n(plus optimiste)\nLien mathématique direct",
            ha='center', va='center', fontsize=11)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    plt.tight_layout()
    return fig