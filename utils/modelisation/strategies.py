"""
Gestion des stratégies de classification pour l'analyse de radiographies
"""

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def display_classification_strategies():
    st.header("1. Stratégies de Classification du Dataset")
    
    strategy = st.selectbox(
        "Choisissez la stratégie à visualiser:",
        [
            "Classification Multiclasses (4 sorties)",
            "Classification Binaire (COVID vs Non-COVID)",
            "Classification Binaire - Étape 1 (Sains vs Malades)",
            "Classification Multiclasses - Étape 2 (3 sorties)",
            "Classification Binaire - Étape 2 (COVID vs Non-COVID)"
        ]
    )
    
    # Diagramme
    fig = _create_strategy_diagram(strategy)
    st.pyplot(fig)
    plt.close(fig)
    
    # Description
    description = _get_strategy_description(strategy)
    if description:
        st.markdown(description)
    
    st.markdown("---")


def _create_strategy_diagram(strategy):
    """Crée le diagramme pour une stratégie donnée"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Mapping des stratégies vers les fonctions de dessin
    strategy_map = {
        "Classification Multiclasses (4 sorties)": _draw_multiclass_4,
        "Classification Binaire (COVID vs Non-COVID)": _draw_binary_covid,
        "Classification Binaire - Étape 1 (Sains vs Malades)": _draw_hierarchical_step1,
        "Classification Multiclasses - Étape 2 (3 sorties)": _draw_hierarchical_step2_multi,
        "Classification Binaire - Étape 2 (COVID vs Non-COVID)": _draw_hierarchical_step2_binary
    }
    
    # Appel de la fonction appropriée
    if strategy in strategy_map:
        strategy_map[strategy](ax)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def _draw_box(ax, x, y, width, height, text, color):
    """Fonction utilitaire pour dessiner une boîte"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02",
                         facecolor=color,
                         edgecolor='black',
                         linewidth=2,
                         alpha=0.8)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color='white' if color in ['#ff4444', '#4444ff', '#aa44ff', '#6644ff'] else 'black')


def _draw_arrow(ax, x1, y1, dx, dy):
    """Fonction utilitaire pour dessiner une flèche"""
    ax.arrow(x1, y1, dx, dy,
            head_width=0.03, head_length=0.02,
            fc='black', ec='black', linewidth=2)


def _draw_multiclass_4(ax):
    # Dataset
    _draw_box(ax, 0.2, 0.8, 0.6, 0.15, 'Dataset Complet\n21,165 images', 'lightblue')
    _draw_arrow(ax, 0.5, 0.75, 0, -0.1)
    
    # Modèle
    _draw_box(ax, 0.2, 0.5, 0.6, 0.15, 'Classification Multiclasses (4 sorties)', 'lightcoral')
    _draw_arrow(ax, 0.5, 0.45, 0, -0.1)
    
    # Classes de sortie
    classes = [
        ('COVID-19\n3,616', 0.05, '#ff4444'),
        ('Normal\n10,192', 0.275, '#44ff44'),
        ('Lung_Opacity\n6,012', 0.5, '#4444ff'),
        ('Viral Pneumonia\n1,345', 0.725, '#ffaa44'),
    ]
    for text, x, color in classes:
        _draw_box(ax, x, 0.15, 0.2, 0.2, text, color)


def _draw_binary_covid(ax):
    # Dataset
    _draw_box(ax, 0.2, 0.8, 0.6, 0.15, 'Dataset Complet\n21,165 images', 'lightblue')
    _draw_arrow(ax, 0.5, 0.75, 0, -0.1)
    
    # Modèle
    _draw_box(ax, 0.2, 0.5, 0.6, 0.15, 'Modèle Classification Binaire', 'lightcoral')
    _draw_arrow(ax, 0.5, 0.45, 0, -0.1)
    
    # Sorties
    _draw_box(ax, 0.1, 0.15, 0.35, 0.2, 'COVID-19\n3,616 images', '#ff4444')
    _draw_box(ax, 0.55, 0.15, 0.35, 0.2, 'Non-COVID\n17,549 images', '#44aa44')


def _draw_hierarchical_step1(ax):
    # Dataset
    _draw_box(ax, 0.2, 0.8, 0.6, 0.15, 'Dataset Complet\n21,165 images', 'lightblue')
    _draw_arrow(ax, 0.5, 0.75, 0, -0.1)
    
    # Modèle
    _draw_box(ax, 0.2, 0.5, 0.6, 0.15, 'Modèle Étape 1\nSains vs Malades', 'lightcoral')
    _draw_arrow(ax, 0.5, 0.45, 0, -0.1)
    
    # Sorties
    _draw_box(ax, 0.1, 0.15, 0.35, 0.2, 'SAINS\n10,192 images\n→ ARRÊT', '#44ff44')
    _draw_box(ax, 0.55, 0.15, 0.35, 0.2, 'MALADES\n10,973 images\n→ Étape 2', '#ffaa44')


def _draw_hierarchical_step2_multi(ax):
    ax.set_title("Stratégie Hiérarchique - Étape 2: Classification des Pathologies",
                fontsize=14, fontweight='bold', pad=15)
    
    # Input
    _draw_box(ax, 0.2, 0.8, 0.6, 0.15, 'Images "Malade" (Étape 1)\n10,973 images', '#ffaa44')
    _draw_arrow(ax, 0.5, 0.75, 0, -0.1)
    
    # Modèle
    _draw_box(ax, 0.2, 0.5, 0.6, 0.15, 'Modèle CNN - Étape 2\nClassification 3 Pathologies', 'lightcoral')
    _draw_arrow(ax, 0.5, 0.45, 0, -0.1)
    
    # Sorties
    diseases = [
        ('COVID-19\n3,616', 0.05, '#ff4444'),
        ('Lung_Opacity\n6,012', 0.375, '#4444ff'),
        ('Viral Pneumonia\n1,345', 0.7, '#aa44ff')
    ]
    for text, x, color in diseases:
        _draw_box(ax, x, 0.15, 0.25, 0.2, text, color)


def _draw_hierarchical_step2_binary(ax):
    ax.set_title("Stratégie Hiérarchique - Étape 2: COVID vs Autres Pathologies",
                fontsize=14, fontweight='bold', pad=15)
    
    # Input
    _draw_box(ax, 0.2, 0.8, 0.6, 0.15, 'Images "Malade" (Étape 1)\n10,973 images', '#ffaa44')
    _draw_arrow(ax, 0.5, 0.75, 0, -0.1)
    
    # Modèle
    _draw_box(ax, 0.2, 0.5, 0.6, 0.15, 'Modèle CNN - Étape 2\nCOVID vs Autres Pathologies', 'lightcoral')
    _draw_arrow(ax, 0.5, 0.45, 0, -0.1)
    
    # Sorties
    _draw_box(ax, 0.1, 0.15, 0.35, 0.2, 'COVID-19\n3,616 images', '#ff4444')
    _draw_box(ax, 0.55, 0.15, 0.35, 0.2, 'Autres Pathologies\n7,357 images', '#6644ff')


def _get_strategy_description(strategy):
    """Retourne la description détaillée d'une stratégie"""
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
        
        "Classification Binaire (COVID vs Non-COVID)": """
**Classification binaire spécialisée COVID**

**Avantages:**
- Problème binaire plus simple
- Adapté au contexte pandémique,avec détection du COVID

**Inconvénients:**
- Perte d'information sur les autres pathologies
- Déséquilibre important entre les deux classes
        """,
        
        "Classification Binaire - Étape 1 (Sains vs Malades)": """
**Première étape: Tri Sain vs Malade**

**Avantages:**
- Classes équilibrées (10,192 vs 10,973)
- Simplification du problème initial

**Inconvénients:**
- Nécessite un deuxième modèle pour la classification des maladies
- Risque d'erreurs cumulées entre les deux étapes
        """,
        
        "Classification Multiclasses - Étape 2 (3 sorties)": """
**Deuxième étape: Classification des pathologies**

**Avantages:**
- Spécialisation sur les cas pathologiques uniquement
- Meilleure discrimination entre maladies

**Inconvéninets:**
- Déséquilibre persistant entre les 3 classes
- Nécessite deux modèles entraînés
        """,
        
        "Classification Binaire - Étape 2 (COVID vs Non-COVID)": """
**Alternative Étape 2: COVID vs Autres pathologies**

**Avantages:**
- Focus maximal sur la détection COVID
- Problème binaire plus simple

**Inconvénients:**
- Perte d'information sur les autres maladies
- Modèle à deux étapes
        """
    }
    
    return descriptions.get(strategy, "")