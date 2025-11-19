"""
Real data generators for Datalab inputs using actual datasets.

This module integrates:
- Dataset loaders (real datasets from sklearn and Hugging Face)
- Feature extractors (real feature extraction methods)
- Model trainers (real model training and predictions)

All components use caching with joblib to avoid repeated computation.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .dataset_loaders import DatasetLoader
from .feature_extractors import FeatureExtractor
from .model_trainers import ModelTrainer

try:
    from datasets import Dataset, Image
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class RealDataGenerator:
    """
    Generates complete Datalab inputs using real datasets.
    
    Combines:
    - Real dataset loading (Iris, Wine, CIFAR-10, MNIST, etc.)
    - Real feature extraction
    - Real model training and predictions
    - All with caching via joblib
    """

    def __init__(self, cache_dir: Path, seed: int = 42):
        self.cache_dir = cache_dir
        self.seed = seed
        np.random.seed(seed)
        
        # Initialize submodules
        self.loader = DatasetLoader(cache_dir, seed)
        self.feature_extractor = FeatureExtractor(cache_dir, seed)
        self.trainer = ModelTrainer(cache_dir, seed)

    def generate_iris_dataset(
        self,
        n_samples: int = 80,
        noise_level: float = 0.1
    ) -> dict[str, Any]:
        """
        Generate Datalab inputs using Iris dataset.
        
        Returns:
            Dictionary with all Datalab inputs:
            - data: dict format {"X": features, "y": labels}
            - labels: numpy array of shape (n_samples,)
            - pred_probs: numpy array of shape (n_samples, n_classes)
            - features: numpy array of shape (n_samples, n_features)
            - knn_graph: sparse CSR matrix
            - label_name: name of label column
            - n_classes: number of classes
        """
        cache_file = self.cache_dir / f"iris_dataset_{n_samples}_{noise_level}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Load dataset
        iris = self.loader.load_iris(n_samples)
        
        # Extract features (standardized)
        features = self.feature_extractor.extract_tabular_features(
            iris["X"], method="standard"
        )
        
        # Train model and get predictions
        training_result = self.trainer.train_and_cache_all(
            features,
            iris["y"],
            task="classification",
            n_classes=iris["n_classes"],
            noise_level=noise_level,
            model_type="gradient_boosting"
        )

        result = {
            "data": {"X": features, "y": training_result["labels"]},
            "labels": training_result["labels"],
            "pred_probs": training_result["pred_probs"],
            "features": features,
            "knn_graph": training_result["knn_graph"],
            "label_name": "y",
            "n_classes": iris["n_classes"],
            "dataset_name": "iris",
        }

        joblib.dump(result, cache_file)
        return result

    def generate_wine_dataset(
        self,
        n_samples: int = 80,
        noise_level: float = 0.1
    ) -> dict[str, Any]:
        """Generate Datalab inputs using Wine dataset."""
        cache_file = self.cache_dir / f"wine_dataset_{n_samples}_{noise_level}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Load dataset
        wine = self.loader.load_wine(n_samples)
        
        # Extract features (PCA)
        features = self.feature_extractor.extract_tabular_features(
            wine["X"], method="pca", n_components=10
        )
        
        # Train model and get predictions
        training_result = self.trainer.train_and_cache_all(
            features,
            wine["y"],
            task="classification",
            n_classes=wine["n_classes"],
            noise_level=noise_level,
            model_type="gradient_boosting"
        )

        result = {
            "data": {"X": features, "y": training_result["labels"]},
            "labels": training_result["labels"],
            "pred_probs": training_result["pred_probs"],
            "features": features,
            "knn_graph": training_result["knn_graph"],
            "label_name": "y",
            "n_classes": wine["n_classes"],
            "dataset_name": "wine",
        }

        joblib.dump(result, cache_file)
        return result

    def generate_digits_dataset(
        self,
        n_samples: int = 80,
        noise_level: float = 0.1
    ) -> dict[str, Any]:
        """
        Generate Datalab inputs using Digits dataset (images).
        
        Returns both dict and DataFrame formats for compatibility.
        """
        cache_file = self.cache_dir / f"digits_dataset_{n_samples}_{noise_level}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Load dataset
        digits = self.loader.load_digits(n_samples)
        
        # Extract features from images
        features = self.feature_extractor.extract_image_features(
            digits["images"], method="combined"
        )
        
        # Train model and get predictions
        training_result = self.trainer.train_and_cache_all(
            features,
            digits["y"],
            task="classification",
            n_classes=digits["n_classes"],
            noise_level=noise_level,
            model_type="gradient_boosting"
        )

        # Create DataFrame format
        feature_cols = [f"feature_{i}" for i in range(features.shape[1])]
        dataframe = pd.DataFrame(features, columns=feature_cols)
        dataframe["label"] = training_result["labels"]

        result = {
            "data": {"X": features, "y": training_result["labels"]},
            "dataframe": dataframe,
            "images": digits["images"],
            "labels": training_result["labels"],
            "pred_probs": training_result["pred_probs"],
            "features": features,
            "knn_graph": training_result["knn_graph"],
            "label_name": "y",
            "label_name_df": "label",
            "n_classes": digits["n_classes"],
            "dataset_name": "digits",
        }

        joblib.dump(result, cache_file)
        return result

    def generate_mnist_dataset(
        self,
        n_samples: int = 80,
        noise_level: float = 0.1
    ) -> dict[str, Any]:
        """
        Generate Datalab inputs using MNIST dataset.
        
        Requires Hugging Face datasets.
        """
        if not HAS_DATASETS:
            raise ImportError(
                "Hugging Face datasets required. "
                "Install with: pip install datasets pillow"
            )

        cache_file = self.cache_dir / f"mnist_dataset_{n_samples}_{noise_level}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Load dataset
        mnist = self.loader.load_mnist(n_samples)
        
        # Extract features from images
        features = self.feature_extractor.extract_image_features(
            mnist["images"], method="combined"
        )
        
        # Train model and get predictions
        training_result = self.trainer.train_and_cache_all(
            features,
            mnist["y"],
            task="classification",
            n_classes=mnist["n_classes"],
            noise_level=noise_level,
            model_type="gradient_boosting"
        )

        # Create DataFrame format
        feature_cols = [f"feature_{i}" for i in range(features.shape[1])]
        dataframe = pd.DataFrame(features, columns=feature_cols)
        dataframe["label"] = training_result["labels"]

        result = {
            "data": {"X": features, "y": training_result["labels"]},
            "dataframe": dataframe,
            "images": mnist["images"],
            "labels": training_result["labels"],
            "pred_probs": training_result["pred_probs"],
            "features": features,
            "knn_graph": training_result["knn_graph"],
            "label_name": "y",
            "label_name_df": "label",
            "n_classes": mnist["n_classes"],
            "dataset_name": "mnist",
        }

        joblib.dump(result, cache_file)
        return result

    def generate_cifar10_dataset(
        self,
        n_samples: int = 80,
        noise_level: float = 0.1
    ) -> dict[str, Any]:
        """
        Generate Datalab inputs using CIFAR-10 dataset.
        
        Requires Hugging Face datasets.
        """
        if not HAS_DATASETS:
            raise ImportError(
                "Hugging Face datasets required. "
                "Install with: pip install datasets pillow"
            )

        cache_file = self.cache_dir / f"cifar10_dataset_{n_samples}_{noise_level}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Load dataset
        cifar10 = self.loader.load_cifar10(n_samples)
        
        # Extract features from images
        features = self.feature_extractor.extract_image_features(
            cifar10["images"], method="combined"
        )
        
        # Train model and get predictions
        training_result = self.trainer.train_and_cache_all(
            features,
            cifar10["y"],
            task="classification",
            n_classes=cifar10["n_classes"],
            noise_level=noise_level,
            model_type="gradient_boosting"
        )

        # Create DataFrame format
        feature_cols = [f"feature_{i}" for i in range(features.shape[1])]
        dataframe = pd.DataFrame(features, columns=feature_cols)
        dataframe["label"] = training_result["labels"]

        result = {
            "data": {"X": features, "y": training_result["labels"]},
            "dataframe": dataframe,
            "images": cifar10["images"],
            "labels": training_result["labels"],
            "pred_probs": training_result["pred_probs"],
            "features": features,
            "knn_graph": training_result["knn_graph"],
            "label_name": "y",
            "label_name_df": "label",
            "n_classes": cifar10["n_classes"],
            "dataset_name": "cifar10",
        }

        joblib.dump(result, cache_file)
        return result

    def generate_huggingface_image_dataset(
        self,
        n_samples: int = 80,
        noise_level: float = 0.1,
        dataset_name: str = "mnist",
        include_image_column: bool = True
    ) -> dict[str, Any]:
        """
        Generate Hugging Face Dataset for image classification.
        
        This format enables image-specific issue detection in Datalab.
        
        Args:
            n_samples: Number of samples
            noise_level: Fraction of labels to corrupt
            dataset_name: "mnist", "cifar10", or "fashion_mnist"
            include_image_column: Whether to include PIL images
            
        Returns:
            Dictionary with:
            - hf_dataset: Hugging Face Dataset object
            - labels: numpy array of class labels
            - pred_probs: predicted probabilities
            - features: extracted feature embeddings
            - knn_graph: k-nearest neighbors graph
            - image_key: name of image column
            - label_name: name of label column
        """
        if not HAS_DATASETS:
            raise ImportError(
                "Hugging Face datasets required. "
                "Install with: pip install datasets pillow"
            )

        cache_file = self.cache_dir / f"hf_{dataset_name}_{n_samples}_{noise_level}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Load appropriate dataset
        if dataset_name == "mnist":
            data = self.loader.load_mnist(n_samples)
        elif dataset_name == "cifar10":
            data = self.loader.load_cifar10(n_samples)
        elif dataset_name == "fashion_mnist":
            data = self.loader.load_fashion_mnist(n_samples)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Extract features
        features = self.feature_extractor.extract_image_features(
            data["images"], method="combined"
        )
        
        # Train model and get predictions
        training_result = self.trainer.train_and_cache_all(
            features,
            data["y"],
            task="classification",
            n_classes=data["n_classes"],
            noise_level=noise_level,
            model_type="gradient_boosting"
        )

        # Create Hugging Face Dataset
        dataset_dict = {
            "label": training_result["labels"].tolist(),
        }

        # Add feature columns
        for i in range(features.shape[1]):
            dataset_dict[f"feature_{i}"] = features[:, i].tolist()

        # Add PIL images if requested
        if include_image_column:
            dataset_dict["image"] = data["pil_images"]

        # Create HF Dataset
        hf_dataset = Dataset.from_dict(dataset_dict)

        # Cast image column to Image feature type
        if include_image_column:
            hf_dataset = hf_dataset.cast_column("image", Image())

        result = {
            "hf_dataset": hf_dataset,
            "images": data["images"],
            "labels": training_result["labels"],
            "pred_probs": training_result["pred_probs"],
            "features": features,
            "knn_graph": training_result["knn_graph"],
            "label_name": "label",
            "image_key": "image" if include_image_column else None,
            "n_classes": data["n_classes"],
            "dataset_name": dataset_name,
        }

        joblib.dump(result, cache_file)
        return result
