"""
Gestion et affichage des architectures de modèles CNN
"""

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd


def display_model_architectures():
    st.header("2. Architectures des Modèles")
    
    # Sélection du type de tâche
    task_type = st.radio(
        "Type de classification:",
        ["binary", "multiclass"],
        format_func=lambda x: "Binaire" if x == "binary" else "Multi-Classes",
        horizontal=True
    )
    
    # Configuration du nombre de classes
    if task_type == 'binary':
        num_classes = 2
        task_name = "Binaire"
    else:
        num_classes = st.selectbox(
            "Nombre de classes:",
            [3, 4],
            format_func=lambda x: f"{x} Classes"
        )
        task_name = f"{num_classes} Classes"
    
    st.info(f"**Tâche:** {task_name} | **Sortie:** {num_classes} classes")
    
    # Sélection de l'approche
    approach = st.selectbox(
        "Approche d'entraînement:",
        ["Transfer Learning", "From Scratch", "Segmentation"]
    )
    
    # Affichage du contenu selon l'approche
    if approach == "Transfer Learning":
        _display_transfer_learning(num_classes)
    elif approach == "From Scratch":
        _display_from_scratch(num_classes)
    else:  # Segmentation
        _display_segmentation()
    
    st.markdown("---")


def _display_transfer_learning(num_classes):
    # Comparaison
    col1, col2,col3 = st.columns(3)
    with col1:
        st.metric("DenseNet169", "12.4M params")
        st.progress(0.7, text="Complexité")
    with col2:
        st.metric("EfficientNetB0", "4M params")
        st.progress(0.2, text="Complexité")
    with col3:
        st.metric("ResNet34", "21.2M params")
        st.progress(1.0, text="Complexité")
    
    # Sélection du modèle
    model_name = st.selectbox(
        "Choisissez le modèle:",
        ["DenseNet", "EfficientNet", "ResNet"]
    )
    
    # Diagramme
    fig = _create_transfer_learning_diagram(model_name, num_classes)
    st.pyplot(fig)
    plt.close(fig)
    
    # Description détaillée
    _show_transfer_learning_description(model_name, num_classes)


def _display_from_scratch(num_classes):
    # Comparaison
    col1, col2 = st.columns(2)
    with col1:
        st.metric("LeNet Modern", "61K params")
        st.progress(0.02, text="Complexité")
    with col2:
        st.metric("CNN Custom", "2.7M params")
        st.progress(1.0, text="Complexité")
    
    # Sélection du modèle
    model_name = st.selectbox(
        "Choisissez le modèle:",
        ["LeNet Modern", "CNN Custom"]
    )
    
    # Diagramme
    fig = _create_from_scratch_diagram(model_name, num_classes)
    st.pyplot(fig)
    plt.close(fig)
    
    # Description
    _show_from_scratch_description(model_name, num_classes)


def _display_segmentation():
    """Affiche l'architecture U-Net"""
    st.info("**U-Net** - Architecture spécialisée pour la segmentation")

    fig = _create_unet_diagram()
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("""
**U-Net - Architecture pour Segmentation**

**Caractéristiques:**
- Architecture encoder-decoder en U
- Skip connections préservant les détails

**Avantages:**
- Préservation des détails fins
- Architecture approuvée pour la segmentation médicale
    """)


def _create_transfer_learning_diagram(model_name, num_classes):
    """Crée le diagramme pour Transfer Learning"""
    model_info = {
        "DenseNet": ("DenseNet169", "7.98M", "1024"),
        "EfficientNet": ("EfficientNetB0", "5.3M", "1280"),
        "ResNet": ("ResNet34", "25.6M", "2048")
    }
    
    name, params, features = model_info[model_name]
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_title(f"Architecture Transfer Learning - {name}",
                fontsize=16, fontweight='bold', pad=20)
    
    # Input
    _draw_box(ax, 0.05, 0.8, 0.15, 0.15, 'Image\n224×224×3', 'lightblue')
    
    # Pre-trained backbone
    _draw_box(ax, 0.25, 0.7, 0.4, 0.3,
             f'{name}\n(Pré-entraîné ImageNet)\n{params} paramètres\nFeatures: {features}',
             'lightgreen')
    ax.text(0.45, 0.72, '❄️ Couches gelées',
            ha='center', va='center', fontsize=9, color='blue')
    
    # Custom classifier
    _draw_box(ax, 0.7, 0.75, 0.25, 0.2,
             'Classificateur\nCustom\nDropout(0.5)\nDense(classes)',
             'orange')
    
    # Output
    _draw_box(ax, 0.4, 0.4, 0.2, 0.15,
             f'Sortie\n{num_classes} classes',
             'lightcoral')
    
    # Arrows
    _draw_arrow(ax, 0.2, 0.875, 0.04, 0)
    _draw_arrow(ax, 0.65, 0.85, 0.04, 0)
    _draw_arrow(ax, 0.7, 0.8, -0.18, -0.25)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 1)
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def _create_from_scratch_diagram(model_name, num_classes):
    """Crée le diagramme pour From Scratch"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    
    if model_name == "LeNet":
        ax.set_title("Architecture From Scratch - LeNet Modern",
                    fontsize=16, fontweight='bold', pad=20)
        
        layers = [
            ("Input\n224×224×1", 'lightblue'),
            ("Conv2D\n6@5×5", 'lightgreen'),
            ("MaxPool\n2×2", 'yellow'),
            ("Conv2D\n16@5×5", 'lightgreen'),
            ("MaxPool\n2×2", 'yellow'),
            ("Flatten", 'orange'),
            ("Dense\n120", 'lightcoral'),
            ("Dense\n84", 'lightcoral'),
            (f"Output\n{num_classes}", 'red')
        ]
        
        x_start, spacing = 0.05, 0.105
        for i, (text, color) in enumerate(layers):
            x = x_start + i * spacing
            _draw_box(ax, x, 0.4, 0.09, 0.2, text, color)
            if i < len(layers) - 1:
                _draw_arrow(ax, x + 0.09, 0.5, 0.01, 0)
    
    else:  # CNN Custom
        ax.set_title("Architecture From Scratch - CNN Custom",
                    fontsize=16, fontweight='bold', pad=20)
        
        layers = [
            ("Input\n224×224×3", 'lightblue'),
            ("Conv\n32@3×3", 'lightgreen'),
            ("Conv\n64@3×3", 'lightgreen'),
            ("Pool", 'yellow'),
            ("Conv\n128@3×3", 'lightgreen'),
            ("Conv\n256@3×3", 'lightgreen'),
            ("Pool", 'yellow'),
            ("Flatten", 'orange'),
            ("Dense\n512", 'lightcoral'),
            ("Drop\n0.5", 'pink'),
            (f"Out\n{num_classes}", 'red')
        ]
        
        x_start, spacing = 0.02, 0.088
        for i, (text, color) in enumerate(layers):
            x = x_start + i * spacing
            _draw_box(ax, x, 0.4, 0.08, 0.2, text, color)
            if i < len(layers) - 1:
                _draw_arrow(ax, x + 0.08, 0.5, 0.005, 0)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.8)
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def _create_unet_diagram():
    """Crée le diagramme U-Net"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_title("Architecture U-Net - Segmentation",
                fontsize=16, fontweight='bold', pad=20)
    
    # Encoder
    encoder = [
        ("Input\n256×256×1", 0.05, 0.7, 'lightblue'),
        ("Conv\n64", 0.15, 0.6, 'lightgreen'),
        ("Conv\n128", 0.25, 0.5, 'lightgreen'),
        ("Conv\n256", 0.35, 0.4, 'lightgreen'),
        ("Conv\n512", 0.45, 0.3, 'lightgreen')
    ]
    
    # Decoder
    decoder = [
        ("Conv\n256", 0.55, 0.4, 'orange'),
        ("Conv\n128", 0.65, 0.5, 'orange'),
        ("Conv\n64", 0.75, 0.6, 'orange'),
        ("Output\n256×256×1", 0.85, 0.7, 'red')
    ]
    
    # Draw encoder
    for text, x, y, color in encoder:
        _draw_box(ax, x, y, 0.08, 0.1, text, color)
        
    # Draw decoder
    for text, x, y, color in decoder:
        _draw_box(ax, x, y, 0.08, 0.1, text, color)
    
    # Skip connections
    for i in range(3):
        ax.plot([0.19 + i*0.1, 0.71 - i*0.1],
               [0.65 - i*0.1, 0.65 - i*0.1],
               'r--', linewidth=2, alpha=0.6)
    
    ax.text(0.5, 0.2, 'Skip Connections →', ha='center',
            fontsize=10, color='red', style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.15, 0.85)
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def _draw_box(ax, x, y, w, h, text, color):
    """Dessine une boîte"""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02",
                         facecolor=color,
                         edgecolor='black',
                         linewidth=2, alpha=0.9)
    ax.add_patch(box)
    
    text_color = 'white' if color in ['red', '#ff4444', '#4444ff'] else 'black'
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            color=text_color)


def _draw_arrow(ax, x, y, dx, dy):
    """Dessine une flèche"""
    ax.arrow(x, y, dx, dy,
            head_width=0.015, head_length=0.01,
            fc='black', ec='black', linewidth=1.5)


def _show_transfer_learning_description(model_name, num_classes):
    """Affiche la description détaillée d'un modèle Transfer Learning"""
    
    descriptions = {
        "DenseNet": f"""
**DenseNet121 - Dense Convolutional Network**

**Structure:**
- ImageNet Backbone (121 couches)
- Dense Blocks avec skip connections
- Global Average Pooling
- Custom Classifier: 1024 → 512 → {num_classes}

**Caractéristiques:**
- Dense Connections: chaque couche connectée aux précédentes
- 12M paramètres pré-entraînés
- Feature reuse maximal
- ~500K paramètres entraînables

**Avantages pour radiographies:**
- Excellent gradient flow
- Moins d'overfitting
        """,
        
        "EfficientNet": f"""
**EfficientNetB0 - Efficient Convolutional Neural Network**

**Structure:**
- ImageNet Backbone (Compound Scaling)
- MBConv Blocks (Mobile Inverted Bottleneck)
- Global Average Pooling
- Custom Classifier: 1280 → 512 → {num_classes}

**Caractéristiques:**
- Compound Scaling: équilibre profondeur/largeur/résolution
- 4M paramètres (le plus léger)
- Squeeze-and-Excitation blocks

**Avantages:**
- Meilleur ratio performance/taille
- Inference ultra-rapide
- Attention mechanisms
        """,
        
        "ResNet": f"""
**ResNet50 - Residual Neural Network**

**Structure:**
- ImageNet Backbone (50 couches)
- Residual Blocks (skip connections)
- Global Average Pooling
- Custom Classifier: 2048 → 512 → {num_classes}

**Caractéristiques:**
- Residual Learning: skip connections (x + F(x))
- 21.2M paramètres (le plus profond)
- Architecture de référence (SOTA 2015)
- 50 couches sans degradation

**Avantages:**
- Très robuste en médical
- Capture features complexes
- Nombreux checkpoints disponibles

        """
    }
    
    st.markdown(descriptions[model_name])


def _show_from_scratch_description(model_name, num_classes):
    """Affiche la description détaillée d'un modèle From Scratch"""
    
    descriptions = {
        "LeNet Modern": f"""
**LeNet Modern - Architecture Classique Modernisée**

**Caractéristiques:**
- ~60K paramètres
- Inspiré de LeCun 1998, modernisé
- 2 couches convolutives (kernel 5×5)

**Modernisations:**
- Batch Normalization (stabilité)
- Dropout progressif (0.1→0.3)
- Adaptive Pooling

**Avantages:**
- Ultra-rapide (2-3 min/30 epochs)
- Faible mémoire (500 MB GPU)
- Baseline pour comparaison
        """,
        
        "CNN Custom": f"""
**CNN Custom - Architecture avec 4 blocs de convolutions**

**Caractéristiques:**
- ~2.7M paramètres
- 4 blocs convolutifs progressifs
- Double convolution par bloc (VGG-style)
- Dropout agressif (0.25-0.5)

**Design rationale:**
- Kernel 3×3: textures fines pulmonaires
- Progression 32→256: hiérarchie features
- Dropout élevé: anti-overfitting
- Classifier profond: séparation non-linéaire

**Avantages:**
- Contrôle total architecture
- Complexité plus adaptée à la tâche
        """
    }
    
    st.markdown(descriptions[model_name])