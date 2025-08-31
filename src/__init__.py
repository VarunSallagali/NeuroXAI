"""
NeuroXAI - Parkinson's Disease Prediction using Explainable AI
A comprehensive machine learning and deep learning framework for Parkinson's disease diagnosis.

This package provides:
- Data preprocessing and feature selection
- Machine learning and deep learning model training
- Explainable AI techniques (SHAP, LIME)
- Visualization and evaluation utilities
- Web application interface
"""

__version__ = "1.0.0"
__author__ = "NeuroXAI Team"
__description__ = "Parkinson's Disease Prediction using XAI"

# Import main classes for easy access
from .preprocessing import DataPreprocessor
from .feature_selection import FeatureSelector
from .model_training import MLModelTrainer, DLModelTrainer
from .explainability import ModelExplainer
from .utils import DataVisualizer, ModelEvaluator

__all__ = [
    'DataPreprocessor',
    'FeatureSelector', 
    'MLModelTrainer',
    'DLModelTrainer',
    'ModelExplainer',
    'DataVisualizer',
    'ModelEvaluator'
]
