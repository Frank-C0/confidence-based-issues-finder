"""
Simplified dataset generators for Datalab testing.

Creates small and medium synthetic datasets for each task type with real model predictions.
All datasets include labels, pred_probs, features, and knn_graph computed from real models.
Results are cached with joblib to avoid recomputation.

For images, uses real MNIST data with a simple PyTorch CNN.
For other tasks, uses synthetic data with sklearn models.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class SimpleDatasetGenerator:
    """
    Generates synthetic datasets with all Datalab requirements using real models.

    Each dataset includes:
    - labels: ground truth labels (with noise)
    - pred_probs: model predictions via cross-validation
    - features: feature embeddings
    - knn_graph: k-nearest neighbors graph

    For images, uses real MNIST data with a simple PyTorch CNN.
    For other tasks, uses synthetic sklearn data with appropriate models.
    """

    def __init__(self, cache_dir: Path, seed: int = 42):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.seed = seed
        np.random.seed(seed)

    def generate_classification_dataset(self, size: str = "small") -> dict[str, Any]:
        """
        Generate synthetic multiclass classification dataset.

        Args:
            size: "small" (50 samples) or "medium" (200 samples)

        Returns:
            Dictionary with all Datalab inputs
        """
        n_samples = 50 if size == "small" else 200
        cache_file = self.cache_dir / f"classification_{size}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Generate synthetic classification data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=15,
            n_informative=12,
            n_redundant=2,
            n_classes=5,
            n_clusters_per_class=1,
            class_sep=1.2,
            flip_y=0.02,  # 2% label noise
            random_state=self.seed,
        )

        # Add noise to features
        noise = np.random.randn(*X.shape) * 0.15
        X = X + noise

        # Add outliers
        n_outliers = max(2, int(n_samples * 0.025))
        outlier_idx = np.random.choice(len(X), n_outliers, replace=False)
        X[outlier_idx] += np.random.randn(n_outliers, X.shape[1]) * 3

        # Standardize features
        features = StandardScaler().fit_transform(X).astype(np.float32)

        # Add additional noise to labels (10% total)
        labels = self._add_label_noise(y, noise_rate=0.08)

        # Train classifier with cross-validation for pred_probs
        pred_probs = self._train_classifier(features, labels, n_classes=5)

        # Compute KNN graph
        knn_graph = self._compute_knn_graph(features)

        # Create HuggingFace dataset
        from datasets import Dataset

        hf_data = {"label": labels.tolist()}
        for i in range(features.shape[1]):
            hf_data[f"feature_{i}"] = features[:, i].tolist()

        hf_dataset = Dataset.from_dict(hf_data)

        result = {
            "data": hf_dataset,
            "labels": labels,
            "pred_probs": pred_probs,
            "features": features,
            "knn_graph": knn_graph,
            "label_name": "label",
            "n_classes": 5,
            "task": "classification",
        }

        joblib.dump(result, cache_file)
        return result

    def generate_multilabel_dataset(self, size: str = "small") -> dict[str, Any]:
        """
        Generate synthetic multilabel classification dataset.

        Args:
            size: "small" (50 samples) or "medium" (200 samples)

        Returns:
            Dictionary with all Datalab inputs
        """
        n_samples = 50 if size == "small" else 200
        cache_file = self.cache_dir / f"multilabel_{size}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Generate synthetic multilabel data
        n_features = 20
        n_classes = 8

        X, y = make_multilabel_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_labels=2,  # Average 2 active labels per sample
            allow_unlabeled=False,
            random_state=self.seed,
        )

        # Add noise to features
        noise = np.random.randn(*X.shape) * 0.15
        X = X + noise

        # Standardize features
        features = StandardScaler().fit_transform(X).astype(np.float32)

        # Add noise to labels
        noisy_labels = self._add_multilabel_noise(y, noise_rate=0.1)

        # Train multilabel classifier with cross-validation
        pred_probs = self._train_multilabel_classifier(features, noisy_labels)

        # Compute KNN graph
        knn_graph = self._compute_knn_graph(features)

        # Convert to list of lists format
        labels_list = [list(np.where(row == 1)[0]) for row in noisy_labels]

        # Create HuggingFace dataset
        from datasets import Dataset

        hf_data = {"label": labels_list}
        for i in range(features.shape[1]):
            hf_data[f"feature_{i}"] = features[:, i].tolist()

        hf_dataset = Dataset.from_dict(hf_data)

        result = {
            "data": hf_dataset,
            "labels": labels_list,
            "pred_probs": pred_probs,
            "features": features,
            "knn_graph": knn_graph,
            "label_name": "label",
            "n_classes": n_classes,
            "task": "multilabel",
        }

        joblib.dump(result, cache_file)
        return result

    def generate_regression_dataset(self, size: str = "small") -> dict[str, Any]:
        """
        Generate synthetic regression dataset.

        Args:
            size: "small" (50 samples) or "medium" (200 samples)

        Returns:
            Dictionary with all Datalab inputs
        """
        n_samples = 50 if size == "small" else 200
        cache_file = self.cache_dir / f"regression_{size}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Generate synthetic regression data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=12,
            n_informative=10,
            noise=10.0,
            bias=5.0,
            random_state=self.seed,
        )

        # Add heteroscedastic noise
        noise_mult = np.abs(X[:, 0]) * 0.5
        y = y + np.random.randn(len(y)) * noise_mult

        # Standardize features
        features = StandardScaler().fit_transform(X).astype(np.float32)
        labels = y.astype(np.float32)

        # Train regressor with cross-validation for predictions
        pred_probs = self._train_regressor(features, labels)

        # Compute KNN graph
        knn_graph = self._compute_knn_graph(features)

        # Create HuggingFace dataset
        from datasets import Dataset

        hf_data = {"label": labels.tolist()}
        for i in range(features.shape[1]):
            hf_data[f"feature_{i}"] = features[:, i].tolist()

        hf_dataset = Dataset.from_dict(hf_data)

        result = {
            "data": hf_dataset,
            "labels": labels,
            "pred_probs": pred_probs,
            "features": features,
            "knn_graph": knn_graph,
            "label_name": "label",
            "task": "regression",
        }

        joblib.dump(result, cache_file)
        return result

    def generate_image_dataset(self, size: str = "small") -> dict[str, Any]:
        """
        Generate image classification dataset using real MNIST data with PyTorch CNN.

        Args:
            size: "small" (50 samples) or "medium" (200 samples)

        Returns:
            Dictionary with all Datalab inputs including PIL images
        """
        n_samples = 50 if size == "small" else 200
        cache_file = self.cache_dir / f"image_{size}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Load MNIST dataset from HuggingFace
        from datasets import load_dataset

        dataset = load_dataset("mnist", split="train", trust_remote_code=True)

        # Sample balanced subset
        labels_array = np.array(dataset["label"])
        indices = self._balanced_sample(labels_array, n_samples)
        subset = dataset.select(indices)

        # Extract images and labels
        pil_images = []
        image_arrays = []
        labels = []

        for item in subset:
            img = item["image"]
            pil_images.append(img)
            img_array = np.array(img, dtype=np.float32) / 255.0
            image_arrays.append(img_array)
            labels.append(item["label"])

        labels = np.array(labels)

        # Add noise to labels (10%)
        noisy_labels = self._add_label_noise(labels, noise_rate=0.1)

        # Train CNN and extract features + predictions
        features, pred_probs = self._train_cnn_model(image_arrays, noisy_labels)

        # Compute KNN graph
        knn_graph = self._compute_knn_graph(features)

        # Create HuggingFace dataset with image column
        from datasets import Dataset, Image

        hf_data = {
            "image": pil_images,
            "label": noisy_labels.tolist(),
        }
        for i in range(features.shape[1]):
            hf_data[f"feature_{i}"] = features[:, i].tolist()

        hf_dataset = Dataset.from_dict(hf_data)
        hf_dataset = hf_dataset.cast_column("image", Image())

        result = {
            "data": hf_dataset,
            "labels": noisy_labels,
            "pred_probs": pred_probs,
            "features": features,
            "knn_graph": knn_graph,
            "label_name": "label",
            "image_key": "image",
            "n_classes": 10,
            "task": "classification",
        }

        joblib.dump(result, cache_file)
        return result

    # Helper methods

    def _balanced_sample(self, labels: list | np.ndarray, n_samples: int) -> list[int]:
        """Sample balanced subset of indices."""
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        samples_per_class = n_samples // len(unique_labels)

        indices = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            sampled = np.random.choice(label_indices, min(samples_per_class, len(label_indices)), replace=False)
            indices.extend(sampled)

        # Add remaining samples if needed
        if len(indices) < n_samples:
            remaining = n_samples - len(indices)
            all_indices = set(range(len(labels)))
            available = list(all_indices - set(indices))
            indices.extend(np.random.choice(available, remaining, replace=False))

        return indices[:n_samples]

    def _add_label_noise(self, labels: np.ndarray, noise_rate: float = 0.1) -> np.ndarray:
        """Add random noise to labels."""
        noisy_labels = labels.copy()
        n_samples = len(labels)
        n_noisy = int(n_samples * noise_rate)

        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
        unique_labels = np.unique(labels)

        for idx in noisy_indices:
            # Assign different label
            other_labels = unique_labels[unique_labels != labels[idx]]
            if len(other_labels) > 0:
                noisy_labels[idx] = np.random.choice(other_labels)

        return noisy_labels

    def _add_multilabel_noise(self, labels: np.ndarray, noise_rate: float = 0.1) -> np.ndarray:
        """Add noise to multilabel data by flipping bits."""
        noisy_labels = labels.copy()
        n_samples, n_classes = labels.shape
        n_flips = int(n_samples * n_classes * noise_rate)

        for _ in range(n_flips):
            i = np.random.randint(0, n_samples)
            j = np.random.randint(0, n_classes)
            noisy_labels[i, j] = 1 - noisy_labels[i, j]

        # Ensure at least one label per sample
        for i in range(n_samples):
            if noisy_labels[i].sum() == 0:
                noisy_labels[i, np.random.randint(0, n_classes)] = 1

        return noisy_labels

    def _train_classifier(self, features: np.ndarray, labels: np.ndarray, n_classes: int = 2) -> np.ndarray:
        """Train RandomForest classifier and get out-of-fold predictions via cross-validation."""
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=self.seed,
            n_jobs=-1,
        )

        pred_probs = cross_val_predict(
            clf, features, labels, cv=3, method="predict_proba", n_jobs=-1
        )

        return pred_probs

    def _train_multilabel_classifier(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Train multilabel classifier and get out-of-fold predictions via cross-validation."""
        clf = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.seed,
                n_jobs=-1,
            )
        )

        # For multilabel, we need to train and predict differently
        # Cross-validation with predict_proba for each output
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=3, shuffle=True, random_state=self.seed)
        n_samples, n_classes = labels.shape
        pred_probs = np.zeros((n_samples, n_classes))

        for train_idx, val_idx in kf.split(features):
            X_train, X_val = features[train_idx], features[val_idx]
            y_train = labels[train_idx]

            clf.fit(X_train, y_train)

            # Get probabilities for each class
            for i, estimator in enumerate(clf.estimators_):
                probs = estimator.predict_proba(X_val)
                # Get probability of class 1 (positive class)
                pred_probs[val_idx, i] = probs[:, 1]

        return pred_probs

    def _train_regressor(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Train RandomForest regressor and get out-of-fold predictions via cross-validation."""
        reg = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=self.seed,
            n_jobs=-1,
        )

        predictions = cross_val_predict(reg, features, labels, cv=3, n_jobs=-1)

        return predictions

    def _train_cnn_model(self, images: list, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Train a simple PyTorch CNN on MNIST images.

        Returns:
            features: Extracted CNN features (from penultimate layer)
            pred_probs: Predicted probabilities via cross-validation
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import KFold

        # Convert images to tensor
        images_array = np.array(images)  # Shape: (n_samples, 28, 28)
        images_tensor = torch.FloatTensor(images_array).unsqueeze(1)  # Add channel dim
        labels_tensor = torch.LongTensor(labels)

        # Define simple CNN
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, 2)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                features = F.relu(x)
                x = self.fc2(features)
                return x, features

        # Cross-validation
        kf = KFold(n_splits=3, shuffle=True, random_state=self.seed)
        n_samples = len(images)
        pred_probs = np.zeros((n_samples, 10))
        all_features = np.zeros((n_samples, 128))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for train_idx, val_idx in kf.split(images_tensor):
            X_train = images_tensor[train_idx]
            y_train = labels_tensor[train_idx]
            X_val = images_tensor[val_idx]

            # Create datasets
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            # Train model
            model = SimpleCNN().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            # Quick training (just 3 epochs for speed)
            model.train()
            for _ in range(3):
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs, _ = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Extract features and predictions for validation set
            model.eval()
            with torch.no_grad():
                X_val = X_val.to(device)
                logits, features = model(X_val)
                probs = F.softmax(logits, dim=1)

                pred_probs[val_idx] = probs.cpu().numpy()
                all_features[val_idx] = features.cpu().numpy()

        return all_features.astype(np.float32), pred_probs.astype(np.float32)

    def _compute_knn_graph(self, features: np.ndarray, n_neighbors: int = 5) -> csr_matrix:
        """Compute k-nearest neighbors graph."""
        n_samples = len(features)
        k = min(n_neighbors, n_samples - 1)

        knn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
        knn.fit(features)
        knn_graph = knn.kneighbors_graph(mode="distance")

        return knn_graph
