"""
Model training and prediction generation.

Trains real models to generate predictions for Datalab:
- Classification models (Random Forest, Gradient Boosting, etc.)
- Regression models
- Cross-validation for out-of-fold predictions

All training results are cached with joblib.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors


class ModelTrainer:
    """
    Trains models and generates predictions with caching.

    Uses fast sklearn models suitable for testing.
    All results are cached to avoid repeated training.
    """

    def __init__(self, cache_dir: Path, seed: int = 42):
        self.cache_dir = cache_dir
        self.seed = seed
        np.random.seed(seed)

    def train_classifier_and_predict(
        self, X: np.ndarray, y: np.ndarray, model_type: str = "gradient_boosting", n_folds: int = 3
    ) -> np.ndarray:
        """
        Train classifier and generate out-of-fold predictions.

        Uses cross-validation to generate pred_probs without data leakage.

        Args:
            X: feature matrix (n_samples, n_features)
            y: labels (n_samples,)
            model_type: "gradient_boosting" or "random_forest"
            n_folds: number of cross-validation folds

        Returns:
            pred_probs: probability predictions (n_samples, n_classes)
        """
        n_classes = len(np.unique(y))
        cache_key = f"{hash(X.tobytes())}_{hash(y.tobytes())}_{model_type}_{n_folds}"
        cache_file = self.cache_dir / f"pred_probs_clf_{cache_key}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Select model
        if model_type == "gradient_boosting":
            clf = HistGradientBoostingClassifier(max_iter=50, random_state=self.seed, verbose=0)
        elif model_type == "random_forest":
            clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=self.seed, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Generate predictions using cross-validation
        pred_probs = cross_val_predict(clf, X, y, cv=n_folds, method="predict_proba", n_jobs=-1)

        joblib.dump(pred_probs, cache_file)
        return pred_probs

    def train_regressor_and_predict(
        self, X: np.ndarray, y: np.ndarray, model_type: str = "gradient_boosting", n_folds: int = 3
    ) -> np.ndarray:
        """
        Train regressor and generate out-of-fold predictions.

        Args:
            X: feature matrix (n_samples, n_features)
            y: target values (n_samples,)
            model_type: "gradient_boosting" or "random_forest"
            n_folds: number of cross-validation folds

        Returns:
            predictions: predicted values (n_samples,)
        """
        cache_key = f"{hash(X.tobytes())}_{hash(y.tobytes())}_{model_type}_{n_folds}"
        cache_file = self.cache_dir / f"pred_reg_{cache_key}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Select model
        if model_type == "gradient_boosting":
            reg = HistGradientBoostingRegressor(max_iter=50, random_state=self.seed, verbose=0)
        elif model_type == "random_forest":
            reg = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=self.seed, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Generate predictions using cross-validation
        predictions = cross_val_predict(reg, X, y, cv=n_folds, n_jobs=-1)

        joblib.dump(predictions, cache_file)
        return predictions

    def compute_knn_graph(self, features: np.ndarray, n_neighbors: int = 5, metric: str = "euclidean") -> csr_matrix:
        """
        Compute k-nearest neighbors graph.

        Args:
            features: feature matrix (n_samples, n_features)
            n_neighbors: number of neighbors
            metric: distance metric

        Returns:
            knn_graph: sparse CSR matrix (n_samples, n_samples)
        """
        cache_key = f"{hash(features.tobytes())}_{n_neighbors}_{metric}"
        cache_file = self.cache_dir / f"knn_graph_{cache_key}.pkl"

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

    def add_label_noise(self, labels: np.ndarray, n_classes: int, noise_level: float = 0.1) -> np.ndarray:
        """
        Add noise to labels for testing label issue detection.

        Args:
            labels: original labels
            n_classes: number of classes
            noise_level: fraction of labels to corrupt

        Returns:
            noisy_labels: labels with added noise
        """
        if noise_level <= 0:
            return labels.copy()

        cache_key = f"{hash(labels.tobytes())}_{n_classes}_{noise_level}"
        cache_file = self.cache_dir / f"noisy_labels_{cache_key}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        noisy_labels = labels.copy()
        n_samples = len(labels)
        n_noisy = int(n_samples * noise_level)

        # Select random indices to corrupt
        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

        # Assign random incorrect labels
        noisy_labels[noisy_indices] = np.random.randint(0, n_classes, n_noisy)

        joblib.dump(noisy_labels, cache_file)
        return noisy_labels

    def train_and_cache_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = "classification",
        n_classes: int | None = None,
        noise_level: float = 0.1,
        model_type: str = "gradient_boosting",
        n_neighbors: int = 5,
    ) -> dict[str, Any]:
        """
        Train model and compute all necessary components for Datalab.

        This is a convenience method that computes everything at once:
        - Adds label noise (if classification)
        - Trains model and generates predictions
        - Computes KNN graph

        Args:
            X: feature matrix
            y: labels/targets
            task: "classification" or "regression"
            n_classes: number of classes (for classification)
            noise_level: fraction of labels to corrupt
            model_type: type of model to use
            n_neighbors: number of neighbors for KNN graph

        Returns:
            Dictionary with all Datalab inputs
        """
        result = {
            "features": X,
            "labels": y,
        }

        if task == "classification":
            # Add noise to labels
            noisy_labels = self.add_label_noise(y, n_classes, noise_level)
            result["labels"] = noisy_labels
            result["original_labels"] = y

            # Train classifier
            pred_probs = self.train_classifier_and_predict(X, noisy_labels, model_type)
            result["pred_probs"] = pred_probs
            result["n_classes"] = n_classes

        elif task == "regression":
            # Train regressor
            predictions = self.train_regressor_and_predict(X, y, model_type)
            result["pred_probs"] = predictions

        else:
            raise ValueError(f"Unknown task: {task}")

        # Compute KNN graph
        knn_graph = self.compute_knn_graph(X, n_neighbors)
        result["knn_graph"] = knn_graph

        return result
