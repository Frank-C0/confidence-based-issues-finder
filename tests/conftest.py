"""
Pytest fixtures for all tests.

Provides shared fixtures for:
- Cache directory
- Random seed
- Data generators
- Pre-generated datasets for different tasks
"""

from pathlib import Path

import pytest

from fixtures.data_generators import DatalabInputGenerator

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
def data_generator(cache_dir, random_seed):
    """Data generator instance."""
    return DatalabInputGenerator(cache_dir, random_seed)


@pytest.fixture(scope="session")
def classification_dataset(data_generator):
    """Pre-generated classification dataset (80 samples, 3 classes)."""
    return data_generator.generate_classification_data(
        n_samples=80,
        n_features=10,
        n_classes=3,
        noise_level=0.1
    )


@pytest.fixture(scope="session")
def small_classification_dataset(data_generator):
    """Small classification dataset for fast tests (30 samples)."""
    return data_generator.generate_classification_data(
        n_samples=30,
        n_features=5,
        n_classes=3,
        noise_level=0.1
    )


@pytest.fixture(scope="session")
def multilabel_dataset(data_generator):
    """Pre-generated multilabel dataset."""
    return data_generator.generate_multilabel_data(
        n_samples=80,
        n_features=10,
        n_classes=3
    )


@pytest.fixture(scope="session")
def regression_dataset(data_generator):
    """Pre-generated regression dataset."""
    return data_generator.generate_regression_data(
        n_samples=80,
        n_features=10
    )


# Configuration for pytest
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "fast: mark test as fast")
    config.addinivalue_line("markers", "classification: tests for classification tasks")
    config.addinivalue_line("markers", "multilabel: tests for multilabel tasks")
    config.addinivalue_line("markers", "regression: tests for regression tasks")
    config.addinivalue_line("markers", "validator: tests for EnhancedIssueTypesValidator")
