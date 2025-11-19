"""
Pruebas del sistema de validación de parámetros.
"""

import pytest

from cleanlab_plugins.enhanced_issue_validator import EnhancedIssueTypesValidator

# from enhanced_issue_validator import EnhancedIssueTypesValidator
from utils.test_helpers import get_issue_types_combinations


class TestParameterValidation:
    """Pruebas del validador de parámetros."""

    @pytest.mark.parametrize("task", ["classification", "regression", "multilabel"])
    def test_validator_initialization(self, task):
        """Test inicialización del validador para diferentes tasks."""
        validator = EnhancedIssueTypesValidator(task=task)

        assert validator.task == task
        available_issues = validator.get_available_issue_types()
        assert isinstance(available_issues, list)
        assert len(available_issues) > 0

    def test_comprehensive_parameter_validation(self):
        """Test de validación comprensiva de parámetros."""
        validator = EnhancedIssueTypesValidator(task="classification")

        complex_config = {
            "label": {
                "k": 15,
                "clean_learning_kwargs": {
                    "cv_n_folds": 3,
                    "find_label_issues_kwargs": {"filter_by": "prune_by_noise_rate", "frac_noise": 0.8},
                },
            },
            "outlier": {
                "k": 10,
                "threshold": 0.2,
            },
            "near_duplicate": {"threshold": 0.1, "metric": "cosine"},
        }

        result = validator.validate(complex_config)

        assert result["is_valid"]
        assert "label" in result["validated_config"]
        assert "outlier" in result["validated_config"]
        assert "near_duplicate" in result["validated_config"]
        assert len(result["errors"]) == 0

    def test_invalid_parameter_detection(self):
        """Test de detección de parámetros inválidos."""
        validator = EnhancedIssueTypesValidator(task="classification")

        invalid_config = {
            "label": {
                "k": -5,  # k debe ser > 0
            },
            "outlier": {
                "threshold": 1.5,  # debe estar entre 0 y 1
            },
            "nonexistent_issue": {  # issue type que no existe
                "invalid_param": "value"
            },
        }

        result = validator.validate(invalid_config)

        assert not result["is_valid"]
        assert len(result["errors"]) >= 2  # Al menos 2 errores

    # CORRECCIÓN: Cambiar el nombre del parámetro para evitar conflictos
    @pytest.mark.parametrize("issue_types_config", get_issue_types_combinations())
    def test_individual_issue_types(self, issue_types_config):
        # from copy import deepcopy
        # issue_types_conf = deepcopy(issue_types_config)
        issue_types_conf = issue_types_config

        """Test de validación individual de cada issue type."""
        print("=== DEBUG test_individual_issue_types ===")
        print(f"Tipo de issue_types_conf: {type(issue_types_conf)}")
        print(f"Contenido de issue_types_conf: {issue_types_conf}")
        print(f"Keys: {list(issue_types_conf.keys()) if hasattr(issue_types_conf, 'keys') else 'No tiene keys'}")

        validator = EnhancedIssueTypesValidator(task="classification")

        print("=== DEBUG test_individual_issue_types ===")
        print(f"Tipo de issue_types_conf: {type(issue_types_conf)}")
        print(f"Contenido de issue_types_conf: {issue_types_conf}")
        print(f"Keys: {list(issue_types_conf.keys()) if hasattr(issue_types_conf, 'keys') else 'No tiene keys'}")

        # Verificar que tenemos la estructura correcta
        assert isinstance(issue_types_conf, dict), f"Expected dict, got {type(issue_types_conf)}"

        # Verificar que no contiene datos de fixtures
        for issue_name, config in issue_types_conf.items():
            assert isinstance(config, dict), f"Config for {issue_name} should be dict, got {type(config)}"
            # Asegurarse de que no contiene arrays de numpy u otros datos de fixtures
            for key, value in config.items():
                assert not hasattr(value, "shape"), f"Config for {issue_name} contains numpy array in key {key}"

        result = validator.validate(issue_types_conf)

        # Cada issue type individual debería ser válido
        assert result["is_valid"], f"Validation failed: {result['errors']}"
        assert len(result["validated_config"]) == len(issue_types_conf)

    def test_empty_config_validation(self):
        """Test de validación con configuración vacía."""
        validator = EnhancedIssueTypesValidator(task="classification")

        result = validator.validate({})

        assert result["is_valid"]
        assert result["validated_config"] == {}
