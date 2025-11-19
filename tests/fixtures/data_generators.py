"""
Simple data generators for Datalab inputs.

Generates the core inputs needed for Datalab.find_issues():
- labels: ground truth labels
- pred_probs: model predictions (probabilities)
- features: feature embeddings/representations
- knn_graph: k-nearest neighbors graph (sparse matrix)

For image data:
- Generates synthetic images using NumPy
- Extracts features from images
- Trains simple models for predictions
"""

from pathlib import Path
from typing import Any

# try:
from datasets import Dataset, Image
import joblib
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from scipy.sparse import csr_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors

HAS_DATASETS = True
# except ImportError:
# HAS_DATASETS = False


class DatalabInputGenerator:
    """
    Generates all necessary inputs for Datalab.find_issues().

    This class creates simple, reproducible synthetic data that matches
    the requirements for Datalab's find_issues() method:
    - pred_probs: (n_samples, n_classes) probability predictions
    - features: (n_samples, n_features) feature representations
    - knn_graph: sparse CSR matrix of k-nearest neighbor distances
    - labels: (n_samples,) ground truth labels
    """

    def __init__(self, cache_dir: Path, seed: int = 42):
        self.cache_dir = cache_dir
        self.seed = seed
        np.random.seed(seed)

    def generate_classification_data(
        self, n_samples: int = 100, n_features: int = 10, n_classes: int = 3, noise_level: float = 0.1
    ) -> dict[str, Any]:
        """
        Generate complete dataset for classification tasks.

        Returns a dictionary with all Datalab inputs:
        - data: dict format {"X": features, "y": labels}
        - labels: numpy array of shape (n_samples,)
        - pred_probs: numpy array of shape (n_samples, n_classes)
        - features: numpy array of shape (n_samples, n_features)
        - knn_graph: sparse CSR matrix of shape (n_samples, n_samples)
        """
        cache_file = self.cache_dir / f"classification_{n_samples}_{n_features}_{n_classes}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Generate features
        features = np.random.randn(n_samples, n_features).astype(np.float32)

        # Generate balanced labels
        labels = np.array([i % n_classes for i in range(n_samples)])
        np.random.shuffle(labels)

        # Add noise to some examples (to create potential label issues)
        if noise_level > 0:
            n_noisy = int(n_samples * noise_level)
            noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
            labels[noisy_indices] = np.random.randint(0, n_classes, n_noisy)

        # Generate pred_probs using simple model
        pred_probs = self._generate_pred_probs(features, labels, n_classes)

        # Generate knn_graph
        knn_graph = self._generate_knn_graph(features, n_neighbors=5)

        result = {
            "data": {"X": features, "y": labels},
            "labels": labels,
            "pred_probs": pred_probs,
            "features": features,
            "knn_graph": knn_graph,
            "label_name": "y",
            "n_classes": n_classes,
        }

        joblib.dump(result, cache_file)
        return result

    def generate_multilabel_data(
        self,
        n_samples: int = 100,
        n_features: int = 10,
        n_classes: int = 3,
    ) -> dict[str, Any]:
        """
        Generate complete dataset for multilabel classification tasks.

        Returns similar structure to classification data, but with multilabel format.
        """
        cache_file = self.cache_dir / f"multilabel_{n_samples}_{n_features}_{n_classes}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        features = np.random.randn(n_samples, n_features).astype(np.float32)

        # Multilabel: each example can have multiple labels
        labels = np.random.randint(0, 2, size=(n_samples, n_classes))

        # Ensure at least one label per example
        for i in range(n_samples):
            if labels[i].sum() == 0:
                labels[i, np.random.randint(0, n_classes)] = 1

        # pred_probs for multilabel: independent probabilities per class
        pred_probs = 1 / (1 + np.exp(-np.random.randn(n_samples, n_classes)))

        knn_graph = self._generate_knn_graph(features, n_neighbors=5)

        # Convert to list of lists for multilabel format
        labels_list = [list(np.where(row == 1)[0]) for row in labels]

        result = {
            "data": {"X": features, "y": labels_list},
            "labels": labels_list,
            "pred_probs": pred_probs,
            "features": features,
            "knn_graph": knn_graph,
            "label_name": "y",
            "n_classes": n_classes,
        }

        joblib.dump(result, cache_file)
        return result

    def generate_regression_data(
        self,
        n_samples: int = 100,
        n_features: int = 10,
    ) -> dict[str, Any]:
        """
        Generate complete dataset for regression tasks.

        For regression, pred_probs is a 1D array of predicted values.
        """
        cache_file = self.cache_dir / f"regression_{n_samples}_{n_features}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        features = np.random.randn(n_samples, n_features).astype(np.float32)

        # Generate continuous target values
        labels = features[:, 0] * 2 + features[:, 1] + np.random.randn(n_samples) * 0.1

        # pred_probs for regression: 1D array of predictions
        pred_probs = labels + np.random.randn(n_samples) * 0.2

        knn_graph = self._generate_knn_graph(features, n_neighbors=5)

        result = {
            "data": {"X": features, "y": labels},
            "labels": labels,
            "pred_probs": pred_probs,
            "features": features,
            "knn_graph": knn_graph,
            "label_name": "y",
        }

        joblib.dump(result, cache_file)
        return result

    def _generate_pred_probs(
        self, features: np.ndarray, labels: np.ndarray, n_classes: int, n_folds: int = 3
    ) -> np.ndarray:
        """
        Generate realistic pred_probs using cross-validation.

        Uses a simple HistGradientBoostingClassifier for speed.
        """
        cache_file = self.cache_dir / f"pred_probs_{len(features)}_{n_classes}_{n_folds}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        clf = HistGradientBoostingClassifier(max_iter=50, random_state=self.seed, verbose=0)

        pred_probs = cross_val_predict(clf, features, labels, cv=n_folds, method="predict_proba", n_jobs=-1)

        joblib.dump(pred_probs, cache_file)
        return pred_probs

    def _generate_knn_graph(self, features: np.ndarray, n_neighbors: int = 5, metric: str = "euclidean") -> csr_matrix:
        """
        Generate k-nearest neighbors graph in the format Datalab expects.

        Returns a sparse CSR matrix with:
        - Shape: (n_samples, n_samples)
        - Non-zero entries: distances to k nearest neighbors
        - Diagonal is zero (self-distances excluded)
        - Each row has exactly k non-zero entries (sorted by distance)
        """
        cache_file = self.cache_dir / f"knn_{len(features)}_{n_neighbors}_{metric}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        n_samples = len(features)
        k = min(n_neighbors, n_samples - 1)

        knn = NearestNeighbors(n_neighbors=k, metric=metric)
        knn.fit(features)

        # Get knn_graph in the correct format
        knn_graph = knn.kneighbors_graph(mode="distance")

        joblib.dump(knn_graph, cache_file)
        return knn_graph

    def generate_image_data(
        self, n_samples: int = 80, n_classes: int = 3, image_shape: tuple = (32, 32, 3), noise_level: float = 0.1
    ) -> dict[str, Any]:
        """
        Generate complete dataset for image classification tasks.

        Creates synthetic images and extracts features from them.
        Returns format compatible with Datalab for image tasks.

        Args:
            n_samples: Number of image samples to generate
            n_classes: Number of classes
            image_shape: Shape of images (height, width, channels)
            noise_level: Fraction of labels to corrupt

        Returns:
            Dictionary with:
            - data: dict with features and labels (for Datalab)
            - dataframe: pandas DataFrame with feature columns and label
            - images: numpy array of synthetic images
            - labels: numpy array of class labels
            - pred_probs: predicted probabilities
            - features: extracted feature embeddings
            - knn_graph: k-nearest neighbors graph
            - label_name: name of label column
        """
        cache_file = self.cache_dir / f"image_{n_samples}_{n_classes}_{image_shape[0]}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Generate synthetic images
        images = self._generate_synthetic_images(n_samples, n_classes, image_shape)

        # Generate balanced labels
        labels = np.array([i % n_classes for i in range(n_samples)])
        np.random.shuffle(labels)

        # Add noise to labels
        if noise_level > 0:
            n_noisy = int(n_samples * noise_level)
            noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
            labels[noisy_indices] = np.random.randint(0, n_classes, n_noisy)

        # Extract features from images
        features = self._extract_image_features(images)

        # Generate predictions using simple model
        pred_probs = self._generate_pred_probs(features, labels, n_classes)

        # Generate knn_graph
        knn_graph = self._generate_knn_graph(features, n_neighbors=5)

        # Create DataFrame format (with feature columns)
        feature_cols = [f"feature_{i}" for i in range(features.shape[1])]
        dataframe = pd.DataFrame(features, columns=feature_cols)
        dataframe["label"] = labels

        # Create dict format (for Datalab)
        dict_data = {
            "features": features.tolist(),
            "labels": labels.tolist(),
        }

        result = {
            "data": dict_data,
            "dataframe": dataframe,
            "images": images,
            "labels": labels,
            "pred_probs": pred_probs,
            "features": features,
            "knn_graph": knn_graph,
            "label_name": "labels",  # For dict format
            "label_name_df": "label",  # For DataFrame format
            "n_classes": n_classes,
        }

        joblib.dump(result, cache_file)
        return result

    def _generate_synthetic_images(self, n_samples: int, n_classes: int, image_shape: tuple) -> np.ndarray:
        """
        Generate simple synthetic images with class-specific patterns.

        Each class has a different visual pattern (color bias, shapes, etc.)
        """
        images = np.zeros((n_samples, *image_shape), dtype=np.float32)

        for i in range(n_samples):
            class_id = i % n_classes

            # Create base image with class-specific color bias
            base_color = (class_id / n_classes) * 0.5 + 0.25
            images[i] = base_color

            # Add class-specific patterns
            if class_id == 0:
                # Class 0: Add horizontal stripes
                images[i, ::4, :, :] += 0.3
            elif class_id == 1:
                # Class 1: Add vertical stripes
                images[i, :, ::4, :] += 0.3
            else:
                # Class 2+: Add checkerboard pattern
                images[i, ::3, ::3, :] += 0.3

            # Add random noise
            images[i] += np.random.randn(*image_shape) * 0.1

            # Clip to valid range [0, 1]
            images[i] = np.clip(images[i], 0, 1)

        return images

    def _extract_image_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract simple features from images.

        Uses basic statistics and hand-crafted features instead of deep learning.
        This keeps dependencies minimal and tests fast.
        """
        n_samples = len(images)
        features_list = []

        for img in images:
            # Flatten and take basic statistics
            flat = img.reshape(-1)

            feat = [
                flat.mean(),  # Mean intensity
                flat.std(),  # Standard deviation
                flat.min(),  # Min value
                flat.max(),  # Max value
                np.median(flat),  # Median
            ]

            # Add per-channel statistics if color image
            if img.shape[-1] == 3:
                for c in range(3):
                    channel = img[:, :, c].flatten()
                    feat.extend(
                        [
                            channel.mean(),
                            channel.std(),
                        ]
                    )

            features_list.append(feat)

        features = np.array(features_list, dtype=np.float32)

        # Add some random features to increase dimensionality
        random_feats = np.random.randn(n_samples, 10).astype(np.float32)
        features = np.hstack([features, random_feats])

        return features

    def generate_huggingface_image_dataset(
        self,
        n_samples: int = 80,
        n_classes: int = 3,
        image_shape: tuple = (32, 32, 3),
        noise_level: float = 0.1,
        include_image_column: bool = True,
    ) -> dict[str, Any] | None:
        """
        Generate Hugging Face Dataset for image classification.

        This format is required for using Datalab's image_key parameter,
        which enables image-specific issue detection (dark, blurry, etc.)

        Args:
            n_samples: Number of image samples to generate
            n_classes: Number of classes
            image_shape: Shape of images (height, width, channels)
            noise_level: Fraction of labels to corrupt
            include_image_column: Whether to include PIL images in dataset

        Returns:
            Dictionary with:
            - hf_dataset: Hugging Face Dataset object
            - labels: numpy array of class labels
            - pred_probs: predicted probabilities
            - features: extracted feature embeddings
            - knn_graph: k-nearest neighbors graph
            - image_key: name of image column (if include_image_column=True)
            - label_name: name of label column
        """
        if not HAS_DATASETS:
            raise ImportError(
                "Hugging Face datasets is required for this feature. Install with: pip install datasets pillow"
            )

        cache_file = self.cache_dir / f"hf_image_{n_samples}_{n_classes}_{image_shape[0]}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Generate synthetic images
        images = self._generate_synthetic_images(n_samples, n_classes, image_shape)

        # Generate balanced labels
        labels = np.array([i % n_classes for i in range(n_samples)])
        np.random.shuffle(labels)

        # Add noise to labels
        if noise_level > 0:
            n_noisy = int(n_samples * noise_level)
            noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
            labels[noisy_indices] = np.random.randint(0, n_classes, n_noisy)

        # Extract features from images
        features = self._extract_image_features(images)

        # Generate predictions using simple model
        pred_probs = self._generate_pred_probs(features, labels, n_classes)

        # Generate knn_graph
        knn_graph = self._generate_knn_graph(features, n_neighbors=5)

        # Create Hugging Face Dataset
        dataset_dict = {
            "label": labels.tolist(),
        }

        # Add feature columns
        for i in range(features.shape[1]):
            dataset_dict[f"feature_{i}"] = features[:, i].tolist()

        # Add PIL images if requested
        if include_image_column:
            pil_images = [self._numpy_to_pil(img) for img in images]
            dataset_dict["image"] = pil_images

        # Create HF Dataset
        hf_dataset = Dataset.from_dict(dataset_dict)

        # Cast image column to Image feature type
        if include_image_column:
            hf_dataset = hf_dataset.cast_column("image", Image())

        result = {
            "hf_dataset": hf_dataset,
            "images": images,
            "labels": labels,
            "pred_probs": pred_probs,
            "features": features,
            "knn_graph": knn_graph,
            "label_name": "label",
            "image_key": "image" if include_image_column else None,
            "n_classes": n_classes,
        }

        joblib.dump(result, cache_file)
        return result

    def _numpy_to_pil(self, img: np.ndarray):
        """
        Convert numpy array to PIL Image.

        Handles float [0, 1] or uint8 [0, 255] ranges.
        """
        # Convert to uint8 if in [0, 1] range
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)

        # Handle grayscale vs RGB
        if len(img.shape) == 2:
            mode = "L"
        elif img.shape[2] == 3:
            mode = "RGB"
        elif img.shape[2] == 4:
            mode = "RGBA"
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        return PILImage.fromarray(img, mode=mode)


def create_minimal_dataset(n_samples: int = 50, n_classes: int = 3) -> dict[str, Any]:
    """
    Create a minimal dataset for quick testing (no caching).

    Use this for simple unit tests that don't need caching.
    """
    np.random.seed(42)

    n_features = 5
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    labels = np.array([i % n_classes for i in range(n_samples)])
    np.random.shuffle(labels)

    # Simple pred_probs: add noise to one-hot encoding
    pred_probs = np.eye(n_classes)[labels] * 0.7 + np.random.rand(n_samples, n_classes) * 0.3
    pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)

    # Simple knn_graph
    knn = NearestNeighbors(n_neighbors=3, metric="euclidean")
    knn.fit(features)
    knn_graph = knn.kneighbors_graph(mode="distance")

    return {
        "data": {"X": features, "y": labels},
        "labels": labels,
        "pred_probs": pred_probs,
        "features": features,
        "knn_graph": knn_graph,
        "label_name": "y",
        "n_classes": n_classes,
    }
