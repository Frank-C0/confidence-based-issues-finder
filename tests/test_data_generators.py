"""
Tests for data generators.

These tests verify that the data generators produce valid inputs for Datalab.
"""

import numpy as np
import pytest
from scipy.sparse import issparse

from fixtures.data_generators import create_minimal_dataset


@pytest.mark.fast
class TestDataGeneratorOutputs:
    """Tests for verifying data generator outputs."""
    
    def test_classification_data_structure(self, classification_dataset):
        """Test classification dataset has correct structure."""
        data = classification_dataset
        
        # Check all required keys are present
        assert "data" in data
        assert "labels" in data
        assert "pred_probs" in data
        assert "features" in data
        assert "knn_graph" in data
        assert "label_name" in data
        
        # Check data shapes
        n_samples = len(data["labels"])
        n_classes = data["n_classes"]
        
        assert data["features"].shape[0] == n_samples
        assert data["pred_probs"].shape == (n_samples, n_classes)
        assert data["knn_graph"].shape == (n_samples, n_samples)
    
    def test_pred_probs_valid_probabilities(self, classification_dataset):
        """Test pred_probs are valid probability distributions."""
        pred_probs = classification_dataset["pred_probs"]
        
        # All values should be between 0 and 1
        assert np.all(pred_probs >= 0)
        assert np.all(pred_probs <= 1)
        
        # Each row should sum to approximately 1
        row_sums = pred_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)
    
    def test_knn_graph_format(self, classification_dataset):
        """Test knn_graph is in correct sparse format."""
        knn_graph = classification_dataset["knn_graph"]
        
        # Should be sparse matrix
        assert issparse(knn_graph)
        
        # Should be square
        assert knn_graph.shape[0] == knn_graph.shape[1]
        
        # Diagonal should be zero (no self-distances)
        assert knn_graph.diagonal().sum() == 0
    
    def test_labels_valid_range(self, classification_dataset):
        """Test labels are in valid range [0, K-1]."""
        labels = classification_dataset["labels"]
        n_classes = classification_dataset["n_classes"]
        
        assert np.all(labels >= 0)
        assert np.all(labels < n_classes)
    
    def test_data_dict_format(self, classification_dataset):
        """Test data dict is in correct format for Datalab."""
        data_dict = classification_dataset["data"]
        
        assert "X" in data_dict
        assert "y" in data_dict
        
        # X and y should have same length
        assert len(data_dict["X"]) == len(data_dict["y"])


@pytest.mark.fast
class TestMinimalDataset:
    """Tests for minimal dataset generator."""
    
    def test_minimal_dataset_creation(self):
        """Test minimal dataset can be created without caching."""
        data = create_minimal_dataset(n_samples=30, n_classes=3)
        
        assert len(data["labels"]) == 30
        assert data["n_classes"] == 3
        assert "pred_probs" in data
        assert "features" in data
        assert "knn_graph" in data
    
    def test_minimal_dataset_reproducibility(self):
        """Test minimal dataset is reproducible."""
        data1 = create_minimal_dataset(n_samples=30, n_classes=3)
        data2 = create_minimal_dataset(n_samples=30, n_classes=3)
        
        np.testing.assert_array_equal(data1["labels"], data2["labels"])
        np.testing.assert_array_almost_equal(data1["features"], data2["features"])


@pytest.mark.fast
class TestMultipleTaskTypes:
    """Tests for different task types."""
    
    def test_multilabel_data_structure(self, multilabel_dataset):
        """Test multilabel dataset has correct structure."""
        data = multilabel_dataset
        
        # Labels should be list of lists
        assert isinstance(data["labels"], list)
        assert isinstance(data["labels"][0], list)
        
        # pred_probs should be 2D
        assert len(data["pred_probs"].shape) == 2
    
    def test_regression_data_structure(self, regression_dataset):
        """Test regression dataset has correct structure."""
        data = regression_dataset
        
        # Labels should be 1D array of continuous values
        assert len(data["labels"].shape) == 1
        
        # pred_probs for regression should also be 1D
        assert len(data["pred_probs"].shape) == 1
        assert len(data["pred_probs"]) == len(data["labels"])


@pytest.mark.fast
class TestDataCaching:
    """Tests for data caching functionality."""
    
    def test_data_generator_uses_cache(self, data_generator):
        """Test that data generator uses cache for repeated calls."""
        # First call creates cache
        data1 = data_generator.generate_classification_data(
            n_samples=50,
            n_features=5,
            n_classes=3
        )
        
        # Second call should use cache (much faster)
        data2 = data_generator.generate_classification_data(
            n_samples=50,
            n_features=5,
            n_classes=3
        )
        
        # Should be identical
        np.testing.assert_array_equal(data1["labels"], data2["labels"])
        np.testing.assert_array_almost_equal(data1["features"], data2["features"])
