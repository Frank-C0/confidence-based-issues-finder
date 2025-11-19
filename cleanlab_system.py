"""
Sistema de pruebas modular para CleanLab DataLab.
Datasets pequeños precomputados (~100 ejemplos) para pruebas rápidas.
Fixtures organizados para fácil acceso a diferentes configuraciones.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# CleanLab import
from cleanlab import Datalab
from datasets import load_dataset
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

# Machine Learning imports
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configuración
CACHE_DIR = Path("./test_cache")
CACHE_DIR.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# DATA CLASSES PARA ESTRUCTURAR LOS DATOS
# =============================================================================


@dataclass
class DatasetConfig:
    """Configuración para un dataset de prueba."""

    name: str
    task: str  # 'classification', 'regression', 'multilabel'
    sample_size: int = 100
    n_features: int | None = None
    n_classes: int | None = None


@dataclass
class ProcessedDataset:
    """Dataset procesado con todas las características necesarias."""

    name: str
    task: str
    data: Any  # Dataset original (pandas, huggingface, etc.)
    features: np.ndarray
    labels: np.ndarray
    pred_probs: np.ndarray | None = None
    knn_graph: Any = None
    label_name: str = "label"
    image_key: str | None = None

    def get_data_dict(self) -> dict[str, Any]:
        """Retorna diccionario en formato para Datalab."""
        if self.task == "classification" and hasattr(self.data, "features"):
            # Para datasets HuggingFace
            return self.data
        else:
            # Para datos tabulares
            return {"X": self.features, "y": self.labels}


# =============================================================================
# PROCESADOR BASE
# =============================================================================


class BaseDataProcessor:
    """Procesador base con funcionalidades comunes."""

    def __init__(self):
        self.scaler = StandardScaler()

    def compute_knn_graph(
        self, features: np.ndarray, n_neighbors: int = 10, metric: str = "euclidean"
    ) -> Any:
        """Calcula grafo KNN."""
        cache_file = CACHE_DIR / f"knn_{features.shape[0]}_{n_neighbors}_{metric}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        knn.fit(features)
        knn_graph = knn.kneighbors_graph(mode="distance")

        joblib.dump(knn_graph, cache_file)
        return knn_graph

    def sample_balanced_data(
        self, features: np.ndarray, labels: np.ndarray, sample_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Muestrea datos manteniendo el balance de clases."""
        if sample_size >= len(labels):
            return features, labels

        # Estratified sampling
        _, features_sampled, _, labels_sampled = train_test_split(
            features, labels, train_size=sample_size, stratify=labels, random_state=SEED
        )
        return features_sampled, labels_sampled


# =============================================================================
# PROCESADOR PARA DATOS TABULARES
# =============================================================================


class TabularDataProcessor(BaseDataProcessor):
    """Procesador especializado para datos tabulares."""

    def __init__(self):
        super().__init__()
        self.label_encoder = LabelEncoder()

    def create_synthetic_tabular_data(self, config: DatasetConfig) -> ProcessedDataset:
        """Crea datos tabulares sintéticos para pruebas."""
        cache_file = CACHE_DIR / f"synthetic_tabular_{config.sample_size}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Generar datos sintéticos
        n_features = config.n_features or 20
        n_classes = config.n_classes or 3

        X, y = make_classification(
            n_samples=config.sample_size * 2,  # Generar extra para sampling
            n_features=n_features,
            n_informative=int(n_features * 0.8),
            n_redundant=int(n_features * 0.2),
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=SEED,
        )

        # Samplear balanceadamente
        X, y = self.sample_balanced_data(X, y, config.sample_size)

        # Crear DataFrame para simular datos reales
        feature_cols = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df["target"] = y

        # Calcular pred_probs
        pred_probs = self.compute_tabular_pred_probs(X, y)

        # Calcular KNN graph
        knn_graph = self.compute_knn_graph(X)

        dataset = ProcessedDataset(
            name=config.name,
            task=config.task,
            data=df,
            features=X,
            labels=y,
            pred_probs=pred_probs,
            knn_graph=knn_graph,
            label_name="target",
        )

        joblib.dump(dataset, cache_file)
        return dataset

    def load_grades_data(self, config: DatasetConfig) -> ProcessedDataset:
        """Carga y procesa el dataset de Student Grades reducido."""
        cache_file = CACHE_DIR / f"grades_{config.sample_size}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Cargar datos completos
        grades_data = pd.read_csv("https://s.cleanlab.ai/grades-tabular-demo-v2.csv")

        # Preprocesamiento
        X_raw = grades_data[["exam_1", "exam_2", "exam_3", "notes"]]
        labels = grades_data["letter_grade"]

        # One-hot encoding
        cat_features = ["notes"]
        X_encoded = pd.get_dummies(X_raw, columns=cat_features, drop_first=True)

        # Estandarización
        numeric_features = ["exam_1", "exam_2", "exam_3"]
        X_processed = X_encoded.copy()
        X_processed[numeric_features] = self.scaler.fit_transform(
            X_encoded[numeric_features]
        )

        # Convertir labels a numéricas
        y_encoded = self.label_encoder.fit_transform(labels)

        # Samplear balanceadamente
        X_sampled, y_sampled = self.sample_balanced_data(
            X_processed.values, y_encoded, config.sample_size
        )

        # Calcular predicciones
        pred_probs = self.compute_tabular_pred_probs(X_sampled, y_sampled)
        knn_graph = self.compute_knn_graph(X_sampled)

        # Crear dataset procesado
        sampled_df = grades_data.iloc[
            train_test_split(
                range(len(grades_data)),
                train_size=config.sample_size,
                stratify=y_encoded,
                random_state=SEED,
            )[0]
        ]

        dataset = ProcessedDataset(
            name=config.name,
            task=config.task,
            data=sampled_df,
            features=X_sampled,
            labels=y_sampled,
            pred_probs=pred_probs,
            knn_graph=knn_graph,
            label_name="letter_grade",
        )

        joblib.dump(dataset, cache_file)
        return dataset

    def compute_tabular_pred_probs(
        self, X: np.ndarray, y: np.ndarray, n_folds: int = 3
    ) -> np.ndarray:
        """Calcula pred_probs para datos tabulares."""
        cache_file = CACHE_DIR / f"tabular_pred_probs_{X.shape[0]}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Usar modelo más simple para datasets pequeños
        if len(X) < 200:
            clf = RandomForestClassifier(n_estimators=50, random_state=SEED)
        else:
            clf = HistGradientBoostingClassifier(random_state=SEED)

        pred_probs = cross_val_predict(
            clf,
            X,
            y,
            cv=min(n_folds, len(np.unique(y))),
            method="predict_proba",
            n_jobs=-1,
        )

        joblib.dump(pred_probs, cache_file)
        return pred_probs


# =============================================================================
# PROCESADOR PARA DATOS DE IMÁGENES
# =============================================================================


class SimpleImageCNN(nn.Module):
    """CNN simple y rápida para imágenes pequeñas."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_embeddings(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ImageDataProcessor(BaseDataProcessor):
    """Procesador especializado para datos de imágenes."""

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_fashion_mnist_small(self, config: DatasetConfig) -> ProcessedDataset:
        """Carga un subset pequeño de Fashion-MNIST."""
        cache_file = CACHE_DIR / f"fashion_mnist_{config.sample_size}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Cargar dataset completo
        dataset = load_dataset("fashion_mnist", split="train")

        # Convertir a arrays numpy
        images = np.array(dataset["image"])
        labels = np.array(dataset["label"])

        # Normalizar y redimensionar
        images = images.astype(np.float32) / 255.0
        images = images.reshape(-1, 1, 28, 28)  # (N, 1, 28, 28)

        # Samplear balanceadamente
        images_sampled, labels_sampled = self.sample_balanced_data(
            images, labels, config.sample_size
        )

        # Seleccionar subset del dataset original
        sampled_indices = train_test_split(
            range(len(dataset)),
            train_size=config.sample_size,
            stratify=labels,
            random_state=SEED,
        )[0]
        sampled_dataset = dataset.select(sampled_indices)

        # Para pruebas rápidas, usar características simples (imágenes aplanadas)
        features = images_sampled.reshape(len(images_sampled), -1)

        # Calcular pred_probs simples (usando modelo pequeño)
        pred_probs = self.compute_simple_image_pred_probs(
            images_sampled, labels_sampled
        )
        knn_graph = self.compute_knn_graph(features, metric="cosine")

        processed_dataset = ProcessedDataset(
            name=config.name,
            task=config.task,
            data=sampled_dataset,
            features=features,
            labels=labels_sampled,
            pred_probs=pred_probs,
            knn_graph=knn_graph,
            label_name="label",
            image_key="image",
        )

        joblib.dump(processed_dataset, cache_file)
        return processed_dataset

    def compute_simple_image_pred_probs(
        self, images: np.ndarray, labels: np.ndarray, n_epochs: int = 2
    ) -> np.ndarray:
        """Calcula pred_probs simples para imágenes."""
        cache_file = CACHE_DIR / f"image_pred_probs_{images.shape[0]}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        # Convertir a tensores
        X_tensor = torch.FloatTensor(images)
        y_tensor = torch.LongTensor(labels)
        dataset = TensorDataset(X_tensor, y_tensor)

        # Entrenamiento simple
        model = SimpleImageCNN(num_classes=len(np.unique(labels))).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Split train/val simple (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED),
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        _val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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

        # Predicciones en todo el dataset
        model.eval()
        all_pred_probs = []
        with torch.no_grad():
            for batch_X, _ in DataLoader(dataset, batch_size=32):
                batch_X = batch_X.to(self.device)
                outputs = model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                all_pred_probs.append(probs.cpu().numpy())

        pred_probs = np.vstack(all_pred_probs)
        joblib.dump(pred_probs, cache_file)
        return pred_probs


# =============================================================================
# REGISTRO Y GESTIÓN DE DATASETS
# =============================================================================


class DatasetRegistry:
    """Registro central de datasets disponibles para pruebas."""

    def __init__(self):
        self.tabular_processor = TabularDataProcessor()
        self.image_processor = ImageDataProcessor()
        self._datasets = {}

        # Registrar configuraciones de datasets
        self._register_datasets()

    def _register_datasets(self):
        """Registra todas las configuraciones de datasets."""
        # Datos tabulares
        self._datasets["synthetic_tabular"] = DatasetConfig(
            name="synthetic_tabular",
            task="classification",
            sample_size=100,
            n_features=20,
            n_classes=3,
        )

        self._datasets["grades_tabular"] = DatasetConfig(
            name="grades_tabular", task="classification", sample_size=100
        )

        # Datos de imágenes
        self._datasets["fashion_mnist"] = DatasetConfig(
            name="fashion_mnist", task="classification", sample_size=100
        )

    def get_dataset(self, dataset_name: str) -> ProcessedDataset:
        """Obtiene un dataset procesado por nombre."""
        if dataset_name not in self._datasets:
            raise ValueError(
                f"Dataset '{dataset_name}' no encontrado. Disponibles: {list(self._datasets.keys())}"
            )

        config = self._datasets[dataset_name]

        if dataset_name == "synthetic_tabular":
            return self.tabular_processor.create_synthetic_tabular_data(config)
        elif dataset_name == "grades_tabular":
            return self.tabular_processor.load_grades_data(config)
        elif dataset_name == "fashion_mnist":
            return self.image_processor.load_fashion_mnist_small(config)
        else:
            raise ValueError(f"Processor para dataset '{dataset_name}' no implementado")

    def list_datasets(self) -> list[str]:
        """Lista todos los datasets disponibles."""
        return list(self._datasets.keys())


# =============================================================================
# FIXTURES DE PYTEST
# =============================================================================

# Instancia global del registro
_dataset_registry = DatasetRegistry()


@pytest.fixture(scope="session")
def dataset_registry():
    """Fixture del registro de datasets."""
    return _dataset_registry


@pytest.fixture(scope="session")
def available_datasets(dataset_registry):
    """Fixture con lista de datasets disponibles."""
    return dataset_registry.list_datasets()


# Fixtures para cada dataset individual
@pytest.fixture(scope="session")
def synthetic_tabular_data(dataset_registry):
    """Fixture con datos tabulares sintéticos."""
    return dataset_registry.get_dataset("synthetic_tabular")


@pytest.fixture(scope="session")
def grades_tabular_data(dataset_registry):
    """Fixture con datos de grades tabulares."""
    return dataset_registry.get_dataset("grades_tabular")


@pytest.fixture(scope="session")
def fashion_mnist_data(dataset_registry):
    """Fixture con datos de Fashion-MNIST."""
    return dataset_registry.get_dataset("fashion_mnist")


# Fixture parametrizado para todos los datasets
def pytest_generate_tests(metafunc):
    """Genera tests parametrizados para todos los datasets."""
    if "dataset_fixture" in metafunc.fixturenames:
        datasets = _dataset_registry.list_datasets()
        metafunc.parametrize("dataset_fixture", datasets, indirect=True)


@pytest.fixture(scope="session")
def dataset_fixture(request, dataset_registry):
    """Fixture parametrizado que proporciona cualquier dataset por nombre."""
    return dataset_registry.get_dataset(request.param)


# =============================================================================
# PRUEBAS BÁSICAS DE DATASETS
# =============================================================================


class TestDatasetBasics:
    """Pruebas básicas de integridad de datos."""

    def test_dataset_structure(self, synthetic_tabular_data):
        """Prueba que el dataset tiene la estructura correcta."""
        dataset = synthetic_tabular_data
        assert dataset.features.shape[0] == len(dataset.labels)
        assert dataset.pred_probs.shape[0] == len(dataset.labels)
        assert dataset.knn_graph.shape[0] == len(dataset.labels)

        print(f"Dataset: {dataset.name}")
        print(f"  Muestras: {len(dataset.labels)}")
        print(f"  Características: {dataset.features.shape[1]}")
        print(f"  Clases: {len(np.unique(dataset.labels))}")

    @pytest.mark.parametrize(
        "dataset_fixture", ["synthetic_tabular", "grades_tabular"], indirect=True
    )
    def test_tabular_data_quality(self, dataset_fixture):
        """Prueba calidad de datos tabulares."""
        dataset = dataset_fixture
        assert dataset.task == "classification"
        assert dataset.features.ndim == 2
        assert not np.any(np.isnan(dataset.features))
        assert len(np.unique(dataset.labels)) >= 2

    @pytest.mark.parametrize("dataset_fixture", ["fashion_mnist"], indirect=True)
    def test_image_data_quality(self, dataset_fixture):
        """Prueba calidad de datos de imágenes."""
        dataset = dataset_fixture
        assert dataset.task == "classification"
        assert dataset.image_key is not None
        assert hasattr(dataset.data, "features")


# =============================================================================
# PRUEBAS DE DATALAB BÁSICAS
# =============================================================================


class TestDatalabBasic:
    """Pruebas básicas de Datalab con diferentes datasets."""

    @pytest.mark.parametrize(
        "dataset_fixture", _dataset_registry.list_datasets(), indirect=True
    )
    def test_datalab_initialization(self, dataset_fixture):
        """Prueba que Datalab se inicializa correctamente con cualquier dataset."""
        dataset = dataset_fixture

        lab = Datalab(
            data=dataset.get_data_dict(), label_name=dataset.label_name, verbosity=4
        )

        assert lab is not None
        print(f"Datalab inicializado con {dataset.name}")

    @pytest.mark.parametrize(
        "dataset_fixture", ["synthetic_tabular", "grades_tabular"], indirect=True
    )
    def test_datalab_find_issues_tabular(self, dataset_fixture):
        """Prueba find_issues con datos tabulares."""
        dataset = dataset_fixture

        lab = Datalab(
            data=dataset.get_data_dict(), label_name=dataset.label_name, verbosity=4
        )

        # Probar con diferentes combinaciones de parámetros
        lab.find_issues(pred_probs=dataset.pred_probs, knn_graph=dataset.knn_graph)

        issues = lab.get_issues()
        assert len(issues) > 0
        assert "is_label_issue" in issues.columns

        print(f"Encontrados {len(issues)} issues en {dataset.name}")

    @pytest.mark.parametrize("dataset_fixture", ["fashion_mnist"], indirect=True)
    def test_datalab_find_issues_images(self, dataset_fixture):
        """Prueba find_issues con datos de imágenes."""
        dataset = dataset_fixture

        lab = Datalab(
            data=dataset.data,
            label_name=dataset.label_name,
            image_key=dataset.image_key,
            verbosity=4,
        )

        lab.find_issues(features=dataset.features, pred_probs=dataset.pred_probs)

        issues = lab.get_issues()
        assert len(issues) > 0

        print(f"Encontrados {len(issues)} issues en {dataset.name}")


# =============================================================================
# PRUEBAS DE VALIDACIÓN DE PARÁMETROS
# =============================================================================


class TestParameterValidation:
    """Pruebas del sistema de validación de parámetros con datos reales."""

    @pytest.fixture
    def issue_validator(self):
        """Fixture del validador de parámetros."""
        from cleanlab_plugins.enhanced_issue_validator import EnhancedIssueTypesValidator

        return EnhancedIssueTypesValidator(task="classification")

    def test_validation_with_real_data(self, synthetic_tabular_data, issue_validator):
        """Prueba validación con configuración realista."""
        test_config = {
            "label": {"k": 10, "clean_learning_kwargs": {"cv_n_folds": 3}},
            "outlier": {"threshold": 0.2, "k": 15},
            "near_duplicate": {"threshold": 0.1},
        }

        result = issue_validator.validate(test_config)
        assert result["is_valid"]
        assert "label" in result["validated_config"]
        assert "outlier" in result["validated_config"]

    @pytest.mark.parametrize(
        "config",
        [
            {"label": {"k": -1}},  # k negativo
            {"outlier": {"threshold": 1.5}},  # threshold > 1
            {"nonexistent_issue": {}},  # issue type no existente
        ],
    )
    def test_validation_invalid_params(self, issue_validator, config):
        """Prueba detección de parámetros inválidos."""
        result = issue_validator.validate(config)
        assert not result["is_valid"]
        assert len(result["errors"]) > 0


# =============================================================================
# UTILIDADES Y HERRAMIENTAS
# =============================================================================


def print_dataset_summary():
    """Imprime resumen de todos los datasets disponibles."""
    registry = DatasetRegistry()
    print("\n" + "=" * 50)
    print("DATASETS DISPONIBLES PARA PRUEBAS")
    print("=" * 50)

    for dataset_name in registry.list_datasets():
        dataset = registry.get_dataset(dataset_name)
        print(f"\n{dataset_name}:")
        print(f"  Tipo: {dataset.task}")
        print(f"  Muestras: {len(dataset.labels)}")
        print(f"  Características: {dataset.features.shape[1]}")
        print(f"  Clases: {len(np.unique(dataset.labels))}")
        print(f"  Label: {dataset.label_name}")
        if dataset.image_key:
            print(f"  Image key: {dataset.image_key}")


def clear_test_cache():
    """Limpia la cache de pruebas."""
    for file in CACHE_DIR.glob("*.pkl"):
        file.unlink()
    print("Cache de pruebas limpiada")


# =============================================================================
# EJECUCIÓN DIRECTA
# =============================================================================

if __name__ == "__main__":
    # Pre-cachear todos los datasets
    print("Pre-cacheando datasets para pruebas...")
    registry = DatasetRegistry()

    for dataset_name in registry.list_datasets():
        print(f"Procesando {dataset_name}...")
        dataset = registry.get_dataset(dataset_name)
        print(f"  ✓ {len(dataset.labels)} muestras")

    print_dataset_summary()

    # Ejecutar pruebas básicas
    print("\nEjecutando pruebas básicas...")
    pytest.main(
        [__file__, "-v", "--tb=short", "-k", "TestDatasetBasics or TestDatalabBasic"]
    )
