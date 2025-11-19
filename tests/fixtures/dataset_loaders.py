"""
Dataset loaders for real-world datasets.

Downloads and prepares popular datasets for testing:
- CIFAR-10 (images)
- MNIST (images)
- Iris (tabular)
- Wine (tabular)

All datasets are limited to ~80 samples for fast testing.
Results are cached with joblib to avoid repeated downloads/processing.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn import datasets as sklearn_datasets

try:
    from datasets import load_dataset
    from PIL import Image as PILImage
    HAS_HUGGINGFACE = True
except ImportError:
    HAS_HUGGINGFACE = False


class DatasetLoader:
    """
    Loads real-world datasets with caching.
    
    All datasets are limited to n_samples (default 80) for fast testing.
    Uses joblib for caching to avoid repeated downloads.
    """

    def __init__(self, cache_dir: Path, seed: int = 42):
        self.cache_dir = cache_dir
        self.seed = seed
        np.random.seed(seed)

    def load_iris(self, n_samples: int = 80) -> dict[str, Any]:
        """
        Load Iris dataset (classification, 3 classes).
        
        Returns:
            Dictionary with:
            - X: feature matrix (n_samples, 4)
            - y: labels (n_samples,)
            - feature_names: list of feature names
            - target_names: list of class names
            - n_classes: number of classes (3)
        """
        cache_file = self.cache_dir / f"iris_{n_samples}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Load full dataset
        iris = sklearn_datasets.load_iris()
        
        # Sample subset
        n_samples = min(n_samples, len(iris.data))
        indices = np.random.choice(len(iris.data), n_samples, replace=False)
        
        result = {
            "X": iris.data[indices].astype(np.float32),
            "y": iris.target[indices],
            "feature_names": iris.feature_names,
            "target_names": iris.target_names.tolist(),
            "n_classes": 3,
        }

        joblib.dump(result, cache_file)
        return result

    def load_wine(self, n_samples: int = 80) -> dict[str, Any]:
        """
        Load Wine dataset (classification, 3 classes).
        
        Returns:
            Dictionary with:
            - X: feature matrix (n_samples, 13)
            - y: labels (n_samples,)
            - feature_names: list of feature names
            - target_names: list of class names
            - n_classes: number of classes (3)
        """
        cache_file = self.cache_dir / f"wine_{n_samples}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Load full dataset
        wine = sklearn_datasets.load_wine()
        
        # Sample subset
        n_samples = min(n_samples, len(wine.data))
        indices = np.random.choice(len(wine.data), n_samples, replace=False)
        
        result = {
            "X": wine.data[indices].astype(np.float32),
            "y": wine.target[indices],
            "feature_names": wine.feature_names,
            "target_names": wine.target_names.tolist(),
            "n_classes": 3,
        }

        joblib.dump(result, cache_file)
        return result

    def load_digits(self, n_samples: int = 80) -> dict[str, Any]:
        """
        Load Digits dataset (classification, 10 classes, images).
        
        Returns:
            Dictionary with:
            - X: feature matrix (n_samples, 64) - flattened 8x8 images
            - y: labels (n_samples,)
            - images: image array (n_samples, 8, 8)
            - target_names: list of class names
            - n_classes: number of classes (10)
        """
        cache_file = self.cache_dir / f"digits_{n_samples}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Load full dataset
        digits = sklearn_datasets.load_digits()
        
        # Sample subset
        n_samples = min(n_samples, len(digits.data))
        indices = np.random.choice(len(digits.data), n_samples, replace=False)
        
        result = {
            "X": digits.data[indices].astype(np.float32),
            "y": digits.target[indices],
            "images": digits.images[indices].astype(np.float32),
            "target_names": [str(i) for i in range(10)],
            "n_classes": 10,
        }

        joblib.dump(result, cache_file)
        return result

    def load_breast_cancer(self, n_samples: int = 80) -> dict[str, Any]:
        """
        Load Breast Cancer dataset (binary classification).
        
        Returns:
            Dictionary with:
            - X: feature matrix (n_samples, 30)
            - y: labels (n_samples,)
            - feature_names: list of feature names
            - target_names: list of class names
            - n_classes: number of classes (2)
        """
        cache_file = self.cache_dir / f"breast_cancer_{n_samples}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Load full dataset
        cancer = sklearn_datasets.load_breast_cancer()
        
        # Sample subset
        n_samples = min(n_samples, len(cancer.data))
        indices = np.random.choice(len(cancer.data), n_samples, replace=False)
        
        result = {
            "X": cancer.data[indices].astype(np.float32),
            "y": cancer.target[indices],
            "feature_names": cancer.feature_names.tolist(),
            "target_names": cancer.target_names.tolist(),
            "n_classes": 2,
        }

        joblib.dump(result, cache_file)
        return result

    def load_mnist(self, n_samples: int = 80) -> dict[str, Any]:
        """
        Load MNIST dataset (classification, 10 classes, images).
        
        Uses Hugging Face datasets for easy access.
        
        Returns:
            Dictionary with:
            - images: image array (n_samples, 28, 28)
            - y: labels (n_samples,)
            - pil_images: list of PIL Images
            - target_names: list of class names
            - n_classes: number of classes (10)
        """
        if not HAS_HUGGINGFACE:
            raise ImportError(
                "Hugging Face datasets required for MNIST. "
                "Install with: pip install datasets pillow"
            )

        cache_file = self.cache_dir / f"mnist_{n_samples}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Load MNIST from Hugging Face
        dataset = load_dataset("mnist", split="train", trust_remote_code=True)
        
        # Sample subset
        n_samples = min(n_samples, len(dataset))
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        subset = dataset.select(indices)
        
        # Extract images and labels
        images = []
        labels = []
        pil_images = []
        
        for item in subset:
            img = item["image"]
            pil_images.append(img)
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
            labels.append(item["label"])
        
        images = np.array(images)
        labels = np.array(labels)
        
        result = {
            "images": images,
            "y": labels,
            "pil_images": pil_images,
            "target_names": [str(i) for i in range(10)],
            "n_classes": 10,
        }

        joblib.dump(result, cache_file)
        return result

    def load_cifar10(self, n_samples: int = 80) -> dict[str, Any]:
        """
        Load CIFAR-10 dataset (classification, 10 classes, RGB images).
        
        Uses Hugging Face datasets for easy access.
        
        Returns:
            Dictionary with:
            - images: image array (n_samples, 32, 32, 3)
            - y: labels (n_samples,)
            - pil_images: list of PIL Images
            - target_names: list of class names
            - n_classes: number of classes (10)
        """
        if not HAS_HUGGINGFACE:
            raise ImportError(
                "Hugging Face datasets required for CIFAR-10. "
                "Install with: pip install datasets pillow"
            )

        cache_file = self.cache_dir / f"cifar10_{n_samples}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Load CIFAR-10 from Hugging Face
        dataset = load_dataset("cifar10", split="train", trust_remote_code=True)
        
        # Sample subset
        n_samples = min(n_samples, len(dataset))
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        subset = dataset.select(indices)
        
        # Extract images and labels
        images = []
        labels = []
        pil_images = []
        
        for item in subset:
            img = item["img"]
            pil_images.append(img)
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
            labels.append(item["label"])
        
        images = np.array(images)
        labels = np.array(labels)
        
        # CIFAR-10 class names
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        
        result = {
            "images": images,
            "y": labels,
            "pil_images": pil_images,
            "target_names": class_names,
            "n_classes": 10,
        }

        joblib.dump(result, cache_file)
        return result

    def load_fashion_mnist(self, n_samples: int = 80) -> dict[str, Any]:
        """
        Load Fashion-MNIST dataset (classification, 10 classes, images).
        
        Uses Hugging Face datasets for easy access.
        
        Returns:
            Dictionary with:
            - images: image array (n_samples, 28, 28)
            - y: labels (n_samples,)
            - pil_images: list of PIL Images
            - target_names: list of class names
            - n_classes: number of classes (10)
        """
        if not HAS_HUGGINGFACE:
            raise ImportError(
                "Hugging Face datasets required for Fashion-MNIST. "
                "Install with: pip install datasets pillow"
            )

        cache_file = self.cache_dir / f"fashion_mnist_{n_samples}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Load Fashion-MNIST from Hugging Face
        dataset = load_dataset("fashion_mnist", split="train", trust_remote_code=True)
        
        # Sample subset
        n_samples = min(n_samples, len(dataset))
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        subset = dataset.select(indices)
        
        # Extract images and labels
        images = []
        labels = []
        pil_images = []
        
        for item in subset:
            img = item["image"]
            pil_images.append(img)
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
            labels.append(item["label"])
        
        images = np.array(images)
        labels = np.array(labels)
        
        # Fashion-MNIST class names
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        
        result = {
            "images": images,
            "y": labels,
            "pil_images": pil_images,
            "target_names": class_names,
            "n_classes": 10,
        }

        joblib.dump(result, cache_file)
        return result
