"""
Pruebas de Datalab con datos tabulares.
"""

from cleanlab import Datalab
import pytest

from utils.data_processors import TabularDataProcessor
from utils.test_helpers import DatalabTestHelper, get_issue_types_combinations


@pytest.mark.tabular
class TestTabularIssues:
    """Pruebas de issues con datos tabulares."""

    @pytest.fixture(scope="class")
    def tabular_data(self, cache_dir, random_seed):
        """Fixture con datos tabulares precomputados."""
        processor = TabularDataProcessor(cache_dir, random_seed)

        # Datos sintéticos pequeños para pruebas rápidas
        df, X, y = processor.create_synthetic_tabular_data(n_samples=80, n_classes=3)
        pred_probs = processor.compute_tabular_pred_probs(X, y, n_folds=2)
        knn_graph = processor.compute_knn_graph(X, n_neighbors=5)

        # Formato dict compatible con Datalab
        dict_data = {"X": X, "y": y}

        return {
            "dataframe": df,
            "dict_data": dict_data,
            "features": X,
            "labels": y,
            "pred_probs": pred_probs,
            "knn_graph": knn_graph,
            "label_name": "y",  # Datalab espera 'y' para el formato dict
        }

    def test_basic_tabular_datalab_with_dict(self, tabular_data):
        """Test básico de Datalab con formato dict."""
        lab = Datalab(data=tabular_data["dict_data"], label_name=tabular_data["label_name"], verbosity=4)

        lab.find_issues(pred_probs=tabular_data["pred_probs"], knn_graph=tabular_data["knn_graph"])

        issues = DatalabTestHelper.validate_datalab_results(lab, expected_columns=["is_label_issue", "label_score"])

        assert len(issues) == len(tabular_data["labels"])

    def test_basic_tabular_datalab_with_dataframe(self, tabular_data):
        """Test básico de Datalab con DataFrame."""
        lab = Datalab(data=tabular_data["dataframe"], label_name="label", verbosity=4)

        lab.find_issues(pred_probs=tabular_data["pred_probs"], knn_graph=tabular_data["knn_graph"])

        issues = DatalabTestHelper.validate_datalab_results(lab)
        assert len(issues) == len(tabular_data["labels"])

    @pytest.mark.parametrize("issue_types_config", get_issue_types_combinations())
    def test_tabular_with_different_issue_types(self, tabular_data, issue_types_config):
        """Test con diferentes combinaciones de issue types."""
        # from copy import deepcopy
        # issue_types_conf = deepcopy(issue_types_config)
        issue_types_conf = issue_types_config

        lab = Datalab(data=tabular_data["dict_data"], label_name=tabular_data["label_name"], verbosity=4)

        lab.find_issues(
            pred_probs=tabular_data["pred_probs"], knn_graph=tabular_data["knn_graph"], issue_types=issue_types_conf
        )

        issues = lab.get_issues()

        # Verificar que se generaron las columnas esperadas
        for issue_name in issue_types_conf:
            issue_col = f"is_{issue_name}_issue"
            score_col = f"{issue_name}_score"
            assert issue_col in issues.columns
            assert score_col in issues.columns

    def test_tabular_features_only(self, tabular_data):
        """Test usando solo features (sin pred_probs)."""
        lab = Datalab(data=tabular_data["dict_data"], label_name=tabular_data["label_name"], verbosity=4)

        lab.find_issues(features=tabular_data["features"])

        issues = DatalabTestHelper.validate_datalab_results(lab)
        assert len(issues) == len(tabular_data["labels"])

    def test_tabular_custom_parameters(self, tabular_data):
        """Test con parámetros personalizados."""
        lab = Datalab(data=tabular_data["dict_data"], label_name=tabular_data["label_name"], verbosity=4)

        custom_issue_types = {"label": {"k": 8}, "outlier": {"threshold": 0.15}, "near_duplicate": {"threshold": 0.1}}

        lab.find_issues(
            pred_probs=tabular_data["pred_probs"], knn_graph=tabular_data["knn_graph"], issue_types=custom_issue_types
        )

        issues = DatalabTestHelper.validate_datalab_results(lab)
        assert len(issues) == len(tabular_data["labels"])
