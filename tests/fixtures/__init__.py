"""
Test fixtures for Datalab testing.

This module provides both synthetic and real data generators:

Synthetic generators (fast, simple):
- DatalabInputGenerator: Generate synthetic data for quick testing

Real data generators (realistic, cached):
- RealDataGenerator: Use real datasets (Iris, Wine, MNIST, CIFAR-10, etc.)
- DatasetLoader: Download and cache real datasets
- FeatureExtractor: Extract features using real methods
- ModelTrainer: Train real models and generate predictions

All real data generators use joblib caching to avoid repeated computation.
"""

from .data_generators import DatalabInputGenerator, create_minimal_dataset
from .dataset_loaders import DatasetLoader
from .feature_extractors import FeatureExtractor
from .model_trainers import ModelTrainer
from .real_data_generators import RealDataGenerator

__all__ = [
    "DatalabInputGenerator",
    "DatasetLoader",
    "FeatureExtractor",
    "ModelTrainer",
    "RealDataGenerator",
    "create_minimal_dataset",
]
