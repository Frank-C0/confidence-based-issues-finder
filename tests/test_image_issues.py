"""
Pruebas de Datalab con datos de imágenes - Versión simplificada.
"""

from cleanlab import Datalab
import pandas as pd
import pytest

from utils.data_processors import ImageDataProcessor
from utils.test_helpers import DatalabTestHelper, get_issue_types_combinations


@pytest.fixture
def issue_types_config(base_issue_types_config):
    from copy import deepcopy

    return deepcopy(base_issue_types_config)


@pytest.mark.image
class TestImageIssues:
    """Pruebas de issues con datos de imágenes usando features."""

    @pytest.fixture(scope="class")
    def image_data(self, cache_dir, random_seed):
        """Fixture con datos de imágenes precomputados - solo features."""
        processor = ImageDataProcessor(cache_dir, random_seed)

        # Crear dataset sintético de imágenes
        images, labels = processor.create_synthetic_image_data(n_samples=80, n_classes=3)
        _model, pred_probs, features = processor.train_simple_image_model(images, labels, n_epochs=1, k_folds=2)
        knn_graph = processor.compute_image_knn_graph(features, n_neighbors=5)

        # CORRECCIÓN: Usar solo formato dict con features, no imágenes raw
        # Datalab funciona mejor con características numéricas que con imágenes raw
        dict_data = {
            "features": features.tolist(),  # Convertir a lista para compatibilidad
            "labels": labels.tolist(),
        }

        # También crear un DataFrame simple sin columnas de imágenes
        # Solo usar características numéricas
        feature_cols = [f"feature_{i}" for i in range(features.shape[1])]
        dataframe = pd.DataFrame(features, columns=feature_cols)
        dataframe["label"] = labels

        return {
            "dataframe": dataframe,
            "dict_data": dict_data,
            "images": images,
            "labels": labels,
            "pred_probs": pred_probs,
            "features": features,
            "knn_graph": knn_graph,
            "label_name": "labels",  # Para dict
            "label_name_df": "label",  # Para DataFrame
        }

    def test_basic_image_datalab_with_dict(self, image_data):
        """Test básico de Datalab con formato dict y features."""
        lab = Datalab(data=image_data["dict_data"], label_name=image_data["label_name"], verbosity=4)

        lab.find_issues(
            pred_probs=image_data["pred_probs"], features=image_data["features"], knn_graph=image_data["knn_graph"]
        )

        issues = DatalabTestHelper.validate_datalab_results(lab, expected_columns=["is_label_issue", "label_score"])

        assert len(issues) == len(image_data["labels"])
        print(f"Encontrados {len(issues)} issues en datos de imágenes")

    def test_basic_image_datalab_with_dataframe(self, image_data):
        """Test básico de Datalab con DataFrame de características."""
        lab = Datalab(data=image_data["dataframe"], label_name=image_data["label_name_df"], verbosity=4)

        lab.find_issues(pred_probs=image_data["pred_probs"], features=image_data["features"])

        issues = DatalabTestHelper.validate_datalab_results(lab)
        assert len(issues) == len(image_data["labels"])

    @pytest.mark.parametrize("issue_types_config", get_issue_types_combinations())
    def test_image_with_different_issue_types(self, image_data, issue_types_config):
        """Test con diferentes combinaciones de issue types."""
        # from copy import deepcopy
        # issue_types_conf = deepcopy(issue_types_config)
        issue_types_conf = issue_types_config

        lab = Datalab(data=image_data["dict_data"], label_name=image_data["label_name"], verbosity=4)

        lab.find_issues(
            pred_probs=image_data["pred_probs"], features=image_data["features"], issue_types=issue_types_conf
        )

        issues = lab.get_issues()

        # Verificar columnas de issues
        for issue_name in issue_types_conf:
            issue_col = f"is_{issue_name}_issue"
            assert issue_col in issues.columns

    def test_image_with_features_only(self, image_data):
        """Test usando solo features (sin pred_probs)."""
        lab = Datalab(data=image_data["dict_data"], label_name=image_data["label_name"], verbosity=4)

        lab.find_issues(features=image_data["features"])

        issues = DatalabTestHelper.validate_datalab_results(lab)
        assert len(issues) == len(image_data["labels"])

    def test_image_with_pred_probs_only(self, image_data):
        """Test usando solo pred_probs (sin features)."""
        lab = Datalab(data=image_data["dict_data"], label_name=image_data["label_name"], verbosity=4)

        lab.find_issues(pred_probs=image_data["pred_probs"])

        issues = DatalabTestHelper.validate_datalab_results(lab)
        assert len(issues) == len(image_data["labels"])

    def test_image_with_knn_only(self, image_data):
        """Test usando solo knn_graph."""
        lab = Datalab(data=image_data["dict_data"], label_name=image_data["label_name"], verbosity=4)

        lab.find_issues(knn_graph=image_data["knn_graph"])

        issues = DatalabTestHelper.validate_datalab_results(lab)
        assert len(issues) == len(image_data["labels"])
