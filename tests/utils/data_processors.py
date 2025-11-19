"""
Procesadores de datos para pruebas - optimizados para velocidad.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# ML imports
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

# DL imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset


class TabularDataProcessor:
    """Procesador optimizado para datos tabulares."""

    def __init__(self, cache_dir: Path, seed: int = 42):
        self.cache_dir = cache_dir
        self.seed = seed
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def create_synthetic_tabular_data(
        self, n_samples: int = 100, n_features: int = 10, n_classes: int = 3
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Crea datos tabulares sintéticos balanceados."""
        cache_file = self.cache_dir / f"synthetic_tabular_{n_samples}_{n_features}_{n_classes}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Crear características
        np.random.seed(self.seed)
        X = np.random.randn(n_samples, n_features)

        # Crear labels balanceadas
        y = np.array([i % n_classes for i in range(n_samples)])
        np.random.shuffle(y)

        # Crear DataFrame realista
        feature_cols = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df["label"] = [f"class_{i}" for i in y]

        # CORRECCIÓN: Agregar ruido de manera correcta
        # Seleccionar 10% de las filas para agregar ruido
        noise_indices = np.random.choice(n_samples, size=max(1, n_samples // 10), replace=False)

        # Agregar ruido solo a las columnas de características (no a la columna 'label')
        noise = np.random.randn(len(noise_indices), n_features) * 0.5
        df.iloc[noise_indices, :n_features] += noise

        result = (df, X, y)
        joblib.dump(result, cache_file)
        return result

    def compute_tabular_pred_probs(self, X: np.ndarray, y: np.ndarray, n_folds: int = 3) -> np.ndarray:
        """Calcula pred_probs para datos tabulares."""
        cache_file = self.cache_dir / f"tabular_pred_probs_{X.shape[0]}_{n_folds}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        clf = HistGradientBoostingClassifier(
            max_iter=50,  # Reducido para velocidad
            random_state=self.seed,
        )

        pred_probs = cross_val_predict(clf, X, y, cv=n_folds, method="predict_proba", n_jobs=-1)

        joblib.dump(pred_probs, cache_file)
        return pred_probs

    def compute_knn_graph(self, X: np.ndarray, n_neighbors: int = 5) -> Any:
        """Calcula grafo KNN optimizado."""
        cache_file = self.cache_dir / f"knn_graph_{X.shape[0]}_{n_neighbors}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        knn = NearestNeighbors(
            n_neighbors=min(n_neighbors, len(X) - 1),  # Evitar n_neighbors > n_samples
            metric="euclidean",
        )
        knn.fit(X)
        knn_graph = knn.kneighbors_graph(mode="distance")

        joblib.dump(knn_graph, cache_file)
        return knn_graph


class SimpleCNN(nn.Module):
    """CNN minimalista para pruebas rápidas."""

    def __init__(self, num_classes: int = 3, input_channels: int = 1):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 32),  # Para imágenes 28x28
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.conv_layers(x))

    def get_embeddings(self, x):
        x = self.conv_layers(x)
        return torch.flatten(x, 1)


class ImageDataProcessor:
    """Procesador optimizado para datos de imágenes."""

    def __init__(self, cache_dir: Path, seed: int = 42):
        self.cache_dir = cache_dir
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_synthetic_image_data(
        self, n_samples: int = 100, img_size: tuple[int, int] = (28, 28), n_classes: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crea datos de imágenes sintéticos."""
        cache_file = self.cache_dir / f"synthetic_images_{n_samples}_{img_size[0]}x{img_size[1]}_{n_classes}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        np.random.seed(self.seed)

        # Crear imágenes sintéticas (canal 1 para escala de grises)
        images = np.random.rand(n_samples, 1, img_size[0], img_size[1]).astype(np.float32)

        # Labels balanceados
        labels = np.array([i % n_classes for i in range(n_samples)])
        np.random.shuffle(labels)

        # Hacer algunas imágenes más "reales" agregando patrones simples
        for i, label in enumerate(labels):
            if label == 0:  # Patrón vertical
                images[i, 0, :, img_size[1] // 2 - 2 : img_size[1] // 2 + 2] = 1.0
            elif label == 1:  # Patrón horizontal
                images[i, 0, img_size[0] // 2 - 2 : img_size[0] // 2 + 2, :] = 1.0
            # label == 2: ruido puro

        result = (images, labels)
        joblib.dump(result, cache_file)
        return result

    def train_simple_image_model(
        self, images: np.ndarray, labels: np.ndarray, n_epochs: int = 2, k_folds: int = 2
    ) -> tuple[nn.Module, np.ndarray, np.ndarray]:
        """Entrena modelo simple y obtiene predicciones."""
        cache_file = self.cache_dir / f"image_model_{images.shape[0]}_epochs{n_epochs}_folds{k_folds}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        n_classes = len(np.unique(labels))

        # Convertir a tensores
        X_tensor = torch.FloatTensor(images)
        y_tensor = torch.LongTensor(labels)
        dataset = TensorDataset(X_tensor, y_tensor)

        # Cross-validation simple
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.seed)

        all_pred_probs = []
        all_features = []
        all_indices = []

        for train_idx, val_idx in kf.split(images, labels):
            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            # Modelo simple
            model = SimpleCNN(num_classes=n_classes).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            # Entrenamiento rápido
            model.train()
            for _epoch in range(n_epochs):
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Predicciones
            model.eval()
            fold_pred_probs = []
            fold_features = []

            with torch.no_grad():
                for batch_X, _ in val_loader:
                    batch_X = batch_X.to(self.device)

                    # Predicciones
                    outputs = model(batch_X)
                    probs = torch.softmax(outputs, dim=1)
                    fold_pred_probs.append(probs.cpu().numpy())

                    # Embeddings
                    embeddings = model.get_embeddings(batch_X)
                    fold_features.append(embeddings.cpu().numpy())

            fold_pred_probs = np.vstack(fold_pred_probs)
            fold_features = np.vstack(fold_features)

            all_pred_probs.append(fold_pred_probs)
            all_features.append(fold_features)
            all_indices.append(val_idx)

        # Reordenar
        final_pred_probs = np.zeros((len(images), n_classes))
        final_features = np.zeros((len(images), 16 * 7 * 7))  # Tamaño de embeddings

        for pred_probs, features, indices in zip(all_pred_probs, all_features, all_indices):
            final_pred_probs[indices] = pred_probs
            final_features[indices] = features

        result = (model, final_pred_probs, final_features)
        joblib.dump(result, cache_file)
        return result

    def compute_image_knn_graph(self, features: np.ndarray, n_neighbors: int = 5) -> Any:
        """Calcula KNN graph para embeddings de imágenes."""
        cache_file = self.cache_dir / f"image_knn_{features.shape[0]}_{n_neighbors}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        knn = NearestNeighbors(n_neighbors=min(n_neighbors, len(features) - 1), metric="cosine")
        knn.fit(features)
        knn_graph = knn.kneighbors_graph(mode="distance")

        joblib.dump(knn_graph, cache_file)
        return knn_graph
