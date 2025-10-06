"""
Module de modélisation
Utilisation dans Streamlit: from utils.modelisation import display_complete_modeling
"""

from .strategies import display_classification_strategies
from .architectures import display_model_architectures
from .parameters import display_training_parameters
from .metrics import display_evaluation_metrics


def display_complete_modeling():
    """
    Fonction principale orchestrant l'affichage de toutes les sections de modélisation
    """
    # Section 1: Stratégies de classification
    display_classification_strategies()
    
    # Section 2: Architectures des modèles
    display_model_architectures()
    
    # Section 3: Paramètres d'entraînement
    display_training_parameters()
    
    # Section 4: Métriques d'évaluation
    display_evaluation_metrics()


__all__ = ['display_complete_modeling']
