"""
Feature extraction for different types of data.

Extracts meaningful features from:
- Tabular data (using statistics and transformations)
- Image data (using hand-crafted features and simple models)
- Text data (using embeddings)

All feature extractors use caching with joblib.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:
    """
    Extracts features from different data types with caching.

    Uses simple, fast methods that don't require deep learning.
    """

    def __init__(self, cache_dir: Path, seed: int = 42):
        self.cache_dir = cache_dir
        self.seed = seed
        np.random.seed(seed)

    def extract_tabular_features(self, X: np.ndarray, method: str = "pca", n_components: int | None = None) -> np.ndarray:
        """
        Extract features from tabular data.

        Methods:
        - "pca": Principal Component Analysis
        - "standard": Standardization only
        - "raw": Use raw features

        Args:
            X: feature matrix (n_samples, n_features)
            method: feature extraction method
            n_components: number of PCA components (None = keep all)

        Returns:
            Feature matrix (n_samples, n_output_features)
        """
        cache_key = f"{hash(X.tobytes())}_{method}_{n_components}"
        cache_file = self.cache_dir / f"features_tabular_{cache_key}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        if method == "raw":
            features = X.copy()
        elif method == "standard":
            scaler = StandardScaler()
            features = scaler.fit_transform(X).astype(np.float32)
        elif method == "pca":
            # First standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Then PCA
            n_components = n_components or min(X.shape)
            pca = PCA(n_components=n_components, random_state=self.seed)
            features = pca.fit_transform(X_scaled).astype(np.float32)
        else:
            raise ValueError(f"Unknown method: {method}")

        joblib.dump(features, cache_file)
        return features

    def extract_image_features(self, images: np.ndarray, method: str = "histogram") -> np.ndarray:
        """
        Extract features from images.

        Methods:
        - "histogram": Color histograms
        - "statistics": Basic statistics (mean, std, etc.)
        - "hog": Histogram of Oriented Gradients (simple version)
        - "combined": All of the above

        Args:
            images: image array (n_samples, height, width) or (n_samples, height, width, channels)
            method: feature extraction method

        Returns:
            Feature matrix (n_samples, n_features)
        """
        cache_key = f"{hash(images.tobytes())}_{method}"
        cache_file = self.cache_dir / f"features_image_{cache_key}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        features_list = []

        if method in ["histogram", "combined"]:
            hist_features = self._extract_histogram_features(images)
            features_list.append(hist_features)

        if method in ["statistics", "combined"]:
            stat_features = self._extract_statistical_features(images)
            features_list.append(stat_features)

        if method in ["hog", "combined"]:
            hog_features = self._extract_simple_hog_features(images)
            features_list.append(hog_features)

        if not features_list:
            raise ValueError(f"Unknown method: {method}")

        # Concatenate all features
        features = np.hstack(features_list).astype(np.float32)

        joblib.dump(features, cache_file)
        return features

    def _extract_histogram_features(self, images: np.ndarray) -> np.ndarray:
        """Extract color histogram features."""
        n_samples = len(images)
        n_bins = 16

        if len(images.shape) == 3:
            # Grayscale images
            features = np.zeros((n_samples, n_bins), dtype=np.float32)
            for i, img in enumerate(images):
                hist, _ = np.histogram(img.flatten(), bins=n_bins, range=(0, 1))
                features[i] = hist / hist.sum()
        else:
            # Color images
            n_channels = images.shape[-1]
            features = np.zeros((n_samples, n_bins * n_channels), dtype=np.float32)
            for i, img in enumerate(images):
                for c in range(n_channels):
                    hist, _ = np.histogram(img[:, :, c].flatten(), bins=n_bins, range=(0, 1))
                    features[i, c * n_bins : (c + 1) * n_bins] = hist / hist.sum()

        return features

    def _extract_statistical_features(self, images: np.ndarray) -> np.ndarray:
        """Extract statistical features (mean, std, min, max, median)."""
        n_samples = len(images)

        if len(images.shape) == 3:
            # Grayscale
            features = np.zeros((n_samples, 5), dtype=np.float32)
            for i, img in enumerate(images):
                flat = img.flatten()
                features[i] = [
                    flat.mean(),
                    flat.std(),
                    flat.min(),
                    flat.max(),
                    np.median(flat),
                ]
        else:
            # Color images - extract per channel
            n_channels = images.shape[-1]
            features = np.zeros((n_samples, 5 * n_channels), dtype=np.float32)
            for i, img in enumerate(images):
                for c in range(n_channels):
                    channel = img[:, :, c].flatten()
                    features[i, c * 5 : (c + 1) * 5] = [
                        channel.mean(),
                        channel.std(),
                        channel.min(),
                        channel.max(),
                        np.median(channel),
                    ]

        return features

    def _extract_simple_hog_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract simple HOG-like features (gradient statistics).

        Simplified version that doesn't require scikit-image.
        """
        n_samples = len(images)

        # Convert to grayscale if color
        if len(images.shape) == 4:
            gray_images = np.mean(images, axis=-1)
        else:
            gray_images = images

        features = np.zeros((n_samples, 8), dtype=np.float32)

        for i, img in enumerate(gray_images):
            # Compute gradients
            dy, dx = np.gradient(img)

            # Gradient magnitude and direction
            magnitude = np.sqrt(dx**2 + dy**2)
            direction = np.arctan2(dy, dx)

            # Simple statistics on gradients
            features[i] = [
                magnitude.mean(),
                magnitude.std(),
                magnitude.max(),
                direction.mean(),
                direction.std(),
                (magnitude > magnitude.mean()).sum() / magnitude.size,  # % strong edges
                np.percentile(magnitude, 75),
                np.percentile(magnitude, 95),
            ]

        return features

    def extract_flattened_images(self, images: np.ndarray) -> np.ndarray:
        """
        Flatten images to use as features.

        Useful for simple models. Applies PCA for dimensionality reduction.

        Args:
            images: image array of any shape

        Returns:
            Flattened and reduced features
        """
        cache_key = f"{hash(images.tobytes())}_flattened"
        cache_file = self.cache_dir / f"features_flattened_{cache_key}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Flatten
        n_samples = len(images)
        X_flat = images.reshape(n_samples, -1).astype(np.float32)

        # Apply PCA if too many features
        max_features = 100
        if X_flat.shape[1] > max_features:
            pca = PCA(n_components=max_features, random_state=self.seed)
            features = pca.fit_transform(X_flat).astype(np.float32)
        else:
            features = X_flat

        joblib.dump(features, cache_file)
        return features
