"""
Tests for EnhancedIssueTypesValidator.

These tests focus on validating the parameter validation logic for all issue types.
"""

import pytest

from cleanlab_plugins.enhanced_issue_validator import EnhancedIssueTypesValidator


@pytest.mark.validator
@pytest.mark.fast
class TestValidatorBasics:
    """Basic tests for validator initialization and configuration."""
    
    @pytest.mark.parametrize("task", ["classification", "regression", "multilabel"])
    def test_validator_initialization(self, task):
        """Test validator can be initialized for all task types."""
        validator = EnhancedIssueTypesValidator(task=task)
        
        assert validator.task == task
        available_issues = validator.get_available_issue_types()
        assert isinstance(available_issues, list)
        assert len(available_issues) > 0
    
    def test_get_available_issue_types(self):
        """Test getting list of available issue types."""
        validator = EnhancedIssueTypesValidator(task="classification")
        available = validator.get_available_issue_types()
        
        # Should include common issue types
        assert "label" in available
        assert "outlier" in available
        assert "near_duplicate" in available
    
    def test_empty_config_is_valid(self):
        """Empty configuration should be valid."""
        validator = EnhancedIssueTypesValidator(task="classification")
        result = validator.validate({})
        
        assert result["is_valid"]
        assert result["validated_config"] == {}
        assert len(result["errors"]) == 0


@pytest.mark.validator
@pytest.mark.fast
class TestSingleIssueTypeValidation:
    """Tests for validating individual issue types."""
    
    def test_label_issue_basic(self):
        """Test basic label issue configuration."""
        validator = EnhancedIssueTypesValidator(task="classification")
        config = {"label": {}}
        
        result = validator.validate(config)
        
        assert result["is_valid"]
        assert "label" in result["validated_config"]
    
    def test_label_issue_with_parameters(self):
        """Test label issue with custom parameters."""
        validator = EnhancedIssueTypesValidator(task="classification")
        config = {
            "label": {
                "k": 10,
                "health_summary_parameters": {"verbose": False}
            }
        }
        
        result = validator.validate(config)
        
        assert result["is_valid"]
        assert result["validated_config"]["label"]["k"] == 10
    
    def test_outlier_issue_basic(self):
        """Test basic outlier issue configuration."""
        validator = EnhancedIssueTypesValidator(task="classification")
        config = {"outlier": {}}
        
        result = validator.validate(config)
        
        assert result["is_valid"]
        assert "outlier" in result["validated_config"]
    
    def test_outlier_issue_with_threshold(self):
        """Test outlier issue with custom threshold."""
        validator = EnhancedIssueTypesValidator(task="classification")
        config = {"outlier": {"threshold": 0.2}}
        
        result = validator.validate(config)
        
        assert result["is_valid"]
        assert result["validated_config"]["outlier"]["threshold"] == 0.2
    
    def test_near_duplicate_issue_basic(self):
        """Test basic near duplicate issue configuration."""
        validator = EnhancedIssueTypesValidator(task="classification")
        config = {"near_duplicate": {}}
        
        result = validator.validate(config)
        
        assert result["is_valid"]
        assert "near_duplicate" in result["validated_config"]
    
    def test_near_duplicate_with_metric(self):
        """Test near duplicate with custom metric."""
        validator = EnhancedIssueTypesValidator(task="classification")
        config = {"near_duplicate": {"metric": "cosine"}}
        
        result = validator.validate(config)
        
        assert result["is_valid"]
        assert result["validated_config"]["near_duplicate"]["metric"] == "cosine"


@pytest.mark.validator
@pytest.mark.fast
class TestMultipleIssueTypesValidation:
    """Tests for validating multiple issue types together."""
    
    def test_label_and_outlier(self):
        """Test validating label and outlier issues together."""
        validator = EnhancedIssueTypesValidator(task="classification")
        config = {
            "label": {"k": 10},
            "outlier": {"threshold": 0.2}
        }
        
        result = validator.validate(config)
        
        assert result["is_valid"]
        assert "label" in result["validated_config"]
        assert "outlier" in result["validated_config"]
        assert len(result["errors"]) == 0
    
    def test_all_common_issues(self):
        """Test validating all common issue types."""
        validator = EnhancedIssueTypesValidator(task="classification")
        config = {
            "label": {"k": 10},
            "outlier": {"threshold": 0.15},
            "near_duplicate": {"threshold": 0.1, "metric": "cosine"}
        }
        
        result = validator.validate(config)
        
        assert result["is_valid"]
        assert len(result["validated_config"]) == 3
        assert len(result["errors"]) == 0


@pytest.mark.validator
@pytest.mark.fast
class TestInvalidParameterDetection:
    """Tests for detecting invalid parameters."""
    
    def test_invalid_k_value(self):
        """Test detection of invalid k parameter (must be > 0)."""
        validator = EnhancedIssueTypesValidator(task="classification")
        config = {"label": {"k": -5}}
        
        result = validator.validate(config)
        
        assert not result["is_valid"]
        assert len(result["errors"]) > 0
    
    def test_invalid_threshold_range(self):
        """Test detection of invalid threshold (must be 0-1)."""
        validator = EnhancedIssueTypesValidator(task="classification")
        config = {"outlier": {"threshold": 1.5}}
        
        result = validator.validate(config)
        
        assert not result["is_valid"]
        assert len(result["errors"]) > 0
    
    def test_nonexistent_issue_type(self):
        """Test detection of non-existent issue type."""
        validator = EnhancedIssueTypesValidator(task="classification")
        config = {"nonexistent_issue": {"param": "value"}}
        
        result = validator.validate(config)
        
        assert not result["is_valid"]
        assert len(result["errors"]) > 0
    
    def test_multiple_invalid_parameters(self):
        """Test detection of multiple errors at once."""
        validator = EnhancedIssueTypesValidator(task="classification")
        config = {
            "label": {"k": -5},
            "outlier": {"threshold": 2.0},
            "invalid_type": {}
        }
        
        result = validator.validate(config)
        
        assert not result["is_valid"]
        assert len(result["errors"]) >= 2


@pytest.mark.validator
class TestComplexParameterValidation:
    """Tests for complex parameter configurations."""
    
    def test_nested_parameters(self):
        """Test validation of nested parameter structures."""
        validator = EnhancedIssueTypesValidator(task="classification")
        config = {
            "label": {
                "k": 15,
                "clean_learning_kwargs": {
                    "cv_n_folds": 3,
                    "find_label_issues_kwargs": {
                        "filter_by": "prune_by_noise_rate",
                        "frac_noise": 0.8
                    }
                }
            }
        }
        
        result = validator.validate(config)
        
        assert result["is_valid"], f"Validation failed: {result['errors']}"
        assert "clean_learning_kwargs" in result["validated_config"]["label"]
    
    def test_health_summary_parameters(self):
        """Test validation of health_summary_parameters."""
        validator = EnhancedIssueTypesValidator(task="classification")
        config = {
            "label": {
                "health_summary_parameters": {
                    "verbose": True,
                    "confidence_weight": 0.5
                }
            }
        }
        
        result = validator.validate(config)
        
        assert result["is_valid"]
        assert "health_summary_parameters" in result["validated_config"]["label"]


@pytest.mark.validator
@pytest.mark.fast
class TestTaskSpecificValidation:
    """Tests for task-specific validation logic."""
    
    def test_classification_task(self):
        """Test validation for classification task."""
        validator = EnhancedIssueTypesValidator(task="classification")
        config = {"label": {}, "outlier": {}}
        
        result = validator.validate(config)
        
        assert result["is_valid"]
    
    def test_regression_task(self):
        """Test validation for regression task."""
        validator = EnhancedIssueTypesValidator(task="regression")
        config = {"outlier": {}, "near_duplicate": {}}
        
        result = validator.validate(config)
        
        assert result["is_valid"]
    
    def test_multilabel_task(self):
        """Test validation for multilabel task."""
        validator = EnhancedIssueTypesValidator(task="multilabel")
        config = {"label": {}}
        
        result = validator.validate(config)
        
        assert result["is_valid"]
