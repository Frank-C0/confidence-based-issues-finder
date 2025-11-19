"""
Configuración global de pytest y fixtures compartidos.
"""

from pathlib import Path

import numpy as np
import pytest

# Configuración global
CACHE_DIR = Path("./test_cache")
CACHE_DIR.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)


# Registrar marcadores personalizados
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (skip with -m 'not slow')")
    config.addinivalue_line("markers", "tabular: mark test as using tabular data")
    config.addinivalue_line("markers", "image: mark test as using image data")


@pytest.fixture(scope="session")
def cache_dir():
    """Directorio para cache de datos de prueba."""
    return CACHE_DIR


@pytest.fixture(scope="session")
def random_seed():
    """Semilla para reproducibilidad."""
    return SEED
