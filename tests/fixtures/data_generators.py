"""
Simple data generators for Datalab inputs.

Generates the core inputs needed for Datalab.find_issues():
- labels: ground truth labels
- pred_probs: model predictions (probabilities)
- features: feature embeddings/representations
- knn_graph: k-nearest neighbors graph (sparse matrix)
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors


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
        self,
        n_samples: int = 100,
        n_features: int = 10,
        n_classes: int = 3,
        noise_level: float = 0.1
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
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_classes: int,
        n_folds: int = 3
    ) -> np.ndarray:
        """
        Generate realistic pred_probs using cross-validation.
        
        Uses a simple HistGradientBoostingClassifier for speed.
        """
        cache_file = self.cache_dir / f"pred_probs_{len(features)}_{n_classes}_{n_folds}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        clf = HistGradientBoostingClassifier(
            max_iter=50,
            random_state=self.seed,
            verbose=0
        )

        pred_probs = cross_val_predict(
            clf, features, labels,
            cv=n_folds,
            method="predict_proba",
            n_jobs=-1
        )

        joblib.dump(pred_probs, cache_file)
        return pred_probs

    def _generate_knn_graph(
        self,
        features: np.ndarray,
        n_neighbors: int = 5,
        metric: str = "euclidean"
    ) -> csr_matrix:
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
