"""
Pytest fixtures for all tests.

Provides shared fixtures for:
- Cache directory
- Random seed
- Data generators (both real and synthetic)
- Pre-generated datasets for different tasks

Uses real datasets (Iris, Wine, MNIST, CIFAR-10, etc.) with:
- Real feature extraction
- Real model training and predictions
- Caching with joblib to avoid repeated computation
"""

from pathlib import Path

import pytest

from fixtures.data_generators import DatalabInputGenerator
from fixtures.real_data_generators import RealDataGenerator

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
    """Synthetic data generator instance (for backward compatibility)."""
    return DatalabInputGenerator(cache_dir, random_seed)


@pytest.fixture(scope="session")
def real_data_generator(cache_dir, random_seed):
    """Real data generator instance using actual datasets."""
    return RealDataGenerator(cache_dir, random_seed)


@pytest.fixture(scope="session")
def classification_dataset(real_data_generator):
    """Pre-generated classification dataset using Wine dataset (80 samples, 3 classes)."""
    return real_data_generator.generate_wine_dataset(
        n_samples=80,
        noise_level=0.1
    )


@pytest.fixture(scope="session")
def small_classification_dataset(real_data_generator):
    """Small classification dataset using Iris (30 samples, 3 classes)."""
    return real_data_generator.generate_iris_dataset(
        n_samples=30,
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


@pytest.fixture(scope="session")
def image_dataset(real_data_generator):
    """Pre-generated image dataset using Digits dataset (80 samples, 10 classes)."""
    return real_data_generator.generate_digits_dataset(
        n_samples=80,
        noise_level=0.1
    )


@pytest.fixture(scope="session")
def small_image_dataset(real_data_generator):
    """Small image dataset using Digits for fast tests (30 samples)."""
    return real_data_generator.generate_digits_dataset(
        n_samples=30,
        noise_level=0.1
    )


@pytest.fixture(scope="session")
def huggingface_image_dataset(real_data_generator):
    """Hugging Face image dataset using MNIST (80 samples, 10 classes)."""
    try:
        return real_data_generator.generate_huggingface_image_dataset(
            n_samples=80,
            noise_level=0.1,
            dataset_name="mnist",
            include_image_column=True
        )
    except ImportError:
        pytest.skip("Hugging Face datasets not installed")


@pytest.fixture(scope="session")
def small_huggingface_image_dataset(real_data_generator):
    """Small Hugging Face image dataset using Fashion MNIST for fast tests (30 samples)."""
    try:
        return real_data_generator.generate_huggingface_image_dataset(
            n_samples=30,
            noise_level=0.1,
            dataset_name="fashion_mnist",
            include_image_column=True
        )
    except ImportError:
        pytest.skip("Hugging Face datasets not installed")


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
