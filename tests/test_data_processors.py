"""
Pruebas para los procesadores de datos.
"""


import numpy as np
import pytest

from utils.data_processors import ImageDataProcessor, TabularDataProcessor


@pytest.mark.tabular
class TestTabularDataProcessor:
    """Pruebas para procesamiento de datos tabulares."""

    def test_synthetic_data_creation(self, cache_dir, random_seed):
        """Test creación de datos tabulares sintéticos."""
        processor = TabularDataProcessor(cache_dir, random_seed)
        df, X, y = processor.create_synthetic_tabular_data(n_samples=50, n_classes=3)

        assert len(df) == 50
        assert X.shape == (50, 10)
        assert len(y) == 50
        assert len(np.unique(y)) == 3

    def test_pred_probs_computation(self, cache_dir, random_seed):
        """Test cálculo de pred_probs."""
        processor = TabularDataProcessor(cache_dir, random_seed)
        _, X, y = processor.create_synthetic_tabular_data(n_samples=50)

        pred_probs = processor.compute_tabular_pred_probs(X, y, n_folds=2)

        assert pred_probs.shape == (50, 3)
        assert np.allclose(pred_probs.sum(axis=1), 1.0)  # Probabilidades suman 1

    def test_knn_graph_computation(self, cache_dir, random_seed):
        """Test cálculo de KNN graph."""
        processor = TabularDataProcessor(cache_dir, random_seed)
        _, X, _ = processor.create_synthetic_tabular_data(n_samples=50)

        knn_graph = processor.compute_knn_graph(X, n_neighbors=3)

        assert knn_graph.shape == (50, 50)
        # Verificar que cada muestra tiene exactamente n_neighbors conexiones
        assert (knn_graph.getnnz(axis=1) == 3).all()


@pytest.mark.image
class TestImageDataProcessor:
    """Pruebas para procesamiento de datos de imágenes."""

    def test_synthetic_image_creation(self, cache_dir, random_seed):
        """Test creación de imágenes sintéticas."""
        processor = ImageDataProcessor(cache_dir, random_seed)
        images, labels = processor.create_synthetic_image_data(n_samples=50)

        assert images.shape == (50, 1, 28, 28)
        assert len(labels) == 50
        assert images.dtype == np.float32

    def test_image_model_training(self, cache_dir, random_seed):
        """Test entrenamiento de modelo de imágenes."""
        processor = ImageDataProcessor(cache_dir, random_seed)
        images, labels = processor.create_synthetic_image_data(n_samples=50)

        _model, pred_probs, features = processor.train_simple_image_model(images, labels, n_epochs=1, k_folds=2)

        assert pred_probs.shape == (50, 3)
        assert features.shape[0] == 50
        assert np.allclose(pred_probs.sum(axis=1), 1.0)

    def test_image_knn_graph(self, cache_dir, random_seed):
        """Test KNN graph para imágenes."""
        processor = ImageDataProcessor(cache_dir, random_seed)
        images, labels = processor.create_synthetic_image_data(n_samples=50)
        _, _, features = processor.train_simple_image_model(images, labels)

        knn_graph = processor.compute_image_knn_graph(features, n_neighbors=3)

        assert knn_graph.shape == (50, 50)
        assert (knn_graph.getnnz(axis=1) == 3).all()
