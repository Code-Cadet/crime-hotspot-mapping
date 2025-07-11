"""
Machine learning prediction package.

Provides ML models for crime hotspot prediction and evaluation tools.
"""

__version__ = "0.1.0"

from .model_train import CrimePredictor, FeatureEngineer
from .evaluate import ModelEvaluator, ModelComparison

__all__ = [
    'CrimePredictor',
    'FeatureEngineer',
    'ModelEvaluator',
    'ModelComparison'
]
