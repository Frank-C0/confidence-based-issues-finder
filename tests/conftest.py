"""
Simplified pytest fixtures for Datalab testing.

Provides fixtures for:
- Small datasets (50 samples) for fast tests
- Medium datasets (200 samples) for comprehensive tests
- All task types: classification, multilabel, regression, image

Datasets are generated synthetically with:
- Real model predictions (pred_probs) via cross-validation
- Real feature extraction
- KNN graphs
- Caching with joblib to avoid recomputation

For images: uses real MNIST data with PyTorch CNN
For other tasks: uses synthetic sklearn data with RandomForest models
"""

from pathlib import Path

import pytest

from fixtures import SimpleDatasetGenerator

# Cache configuration
CACHE_DIR = Path("./test_cache")
CACHE_DIR.mkdir(exist_ok=True)
SEED = 42


@pytest.fixture(scope="session")
def cache_dir():
    """Cache directory for test data."""
    return CACHE_DIR


@pytest.fixture(scope="session")
def random_seed():
    """Random seed for reproducibility."""
    return SEED


@pytest.fixture(scope="session")
def generator(cache_dir, random_seed):
    """Simplified dataset generator instance."""
    return SimpleDatasetGenerator(cache_dir, random_seed)


# Classification datasets
@pytest.fixture(scope="session")
def small_classification(generator):
    """Small classification dataset (50 samples, 2 classes) using IMDB."""
    return generator.generate_classification_dataset(size="small")


@pytest.fixture(scope="session")
def medium_classification(generator):
    """Medium classification dataset (200 samples, 2 classes) using IMDB."""
    return generator.generate_classification_dataset(size="medium")


# Multilabel datasets
@pytest.fixture(scope="session")
def small_multilabel(generator):
    """Small multilabel dataset (50 samples, 3 classes)."""
    return generator.generate_multilabel_dataset(size="small")


@pytest.fixture(scope="session")
def medium_multilabel(generator):
    """Medium multilabel dataset (200 samples, 3 classes)."""
    return generator.generate_multilabel_dataset(size="medium")


# Regression datasets
@pytest.fixture(scope="session")
def small_regression(generator):
    """Small regression dataset (50 samples) using California housing."""
    return generator.generate_regression_dataset(size="small")


@pytest.fixture(scope="session")
def medium_regression(generator):
    """Medium regression dataset (200 samples) using California housing."""
    return generator.generate_regression_dataset(size="medium")


# Image datasets
@pytest.fixture(scope="session")
def small_image(generator):
    """Small image dataset (50 samples, 10 classes) using MNIST."""
    return generator.generate_image_dataset(size="small")


@pytest.fixture(scope="session")
def medium_image(generator):
    """Medium image dataset (200 samples, 10 classes) using MNIST."""
    return generator.generate_image_dataset(size="medium")


# Configuration for pytest
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "fast: mark test as fast")
    config.addinivalue_line("markers", "classification: tests for classification tasks")
    config.addinivalue_line("markers", "multilabel: tests for multilabel tasks")
    config.addinivalue_line("markers", "regression: tests for regression tasks")
    config.addinivalue_line("markers", "image: tests for image tasks")
    config.addinivalue_line("markers", "validator: tests for EnhancedIssueTypesValidator")
