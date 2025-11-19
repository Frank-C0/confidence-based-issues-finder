"""
Tests for Datalab integration with classification data.

These tests verify that Datalab.find_issues() works correctly with:
- Different input combinations (pred_probs, features, knn_graph)
- Different issue type configurations validated by EnhancedIssueTypesValidator
"""

from cleanlab import Datalab
import numpy as np
import pytest

from cleanlab_plugins.enhanced_issue_validator import EnhancedIssueTypesValidator


@pytest.mark.classification
@pytest.mark.fast
class TestDatalabBasicIntegration:
    """Basic integration tests for Datalab with classification data."""

    def test_datalab_with_pred_probs_only(self, small_classification):
        """Test Datalab with only pred_probs."""
        data = small_classification

        lab = Datalab(verbosity=4,
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )

        lab.find_issues(pred_probs=data["pred_probs"])

        issues = lab.get_issues()
        assert len(issues) == len(data["labels"])
        assert "is_label_issue" in issues.columns

    def test_datalab_with_features_only(self, small_classification):
        """Test Datalab with only features."""
        data = small_classification

        lab = Datalab(verbosity=4,
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )

        lab.find_issues(features=data["features"])

        issues = lab.get_issues()
        assert len(issues) == len(data["labels"])

    def test_datalab_with_knn_graph_only(self, small_classification):
        """Test Datalab with only knn_graph."""
        data = small_classification

        lab = Datalab(verbosity=4,
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )

        lab.find_issues(knn_graph=data["knn_graph"])

        issues = lab.get_issues()
        assert len(issues) == len(data["labels"])

    def test_datalab_with_all_inputs(self, small_classification):
        """Test Datalab with all inputs (pred_probs, features, knn_graph)."""
        data = small_classification

        lab = Datalab(verbosity=4,
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )

        lab.find_issues(
            pred_probs=data["pred_probs"],
            features=data["features"],
            knn_graph=data["knn_graph"]
        )

        issues = lab.get_issues()
        assert len(issues) == len(data["labels"])

        # Should detect multiple issue types
        issue_columns = [col for col in issues.columns if col.startswith("is_") and col.endswith("_issue")]
        assert len(issue_columns) > 0




@pytest.mark.classification
class TestDatalabWithValidatedIssueTypes:
    """Tests for Datalab using validated issue type configurations."""

    def test_label_issues_only(self, small_classification):
        """Test with only label issues enabled."""
        data = small_classification

        lab = Datalab(verbosity=4,
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )

        # Validate issue types
        validator = EnhancedIssueTypesValidator(task="classification")
        issue_types = {"label": {}}
        validated_issues = validator.validate(issue_types)
        assert validated_issues["is_valid"]

        lab.find_issues(
            pred_probs=data["pred_probs"],
            issue_types=validated_issues["validated_config"]
        )

        issues = lab.get_issues()
        assert "is_label_issue" in issues.columns
        assert "is_outlier_issue" not in issues.columns

    def test_outlier_issues_only(self, small_classification):
        """Test with only outlier issues enabled."""
        data = small_classification

        lab = Datalab(verbosity=4,
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )

        # Validate issue types
        validator = EnhancedIssueTypesValidator(task="classification")
        issue_types = {"outlier": {}}
        validated_issues = validator.validate(issue_types)
        assert validated_issues["is_valid"]

        lab.find_issues(
            features=data["features"],
            issue_types=validated_issues["validated_config"]
        )

        issues = lab.get_issues()
        assert "is_outlier_issue" in issues.columns
        assert "is_label_issue" not in issues.columns

    def test_multiple_issue_types(self, small_classification):
        """Test with multiple issue types enabled."""
        data = small_classification

        lab = Datalab(verbosity=4,
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )

        # Validate issue types
        validator = EnhancedIssueTypesValidator(task="classification")
        issue_types = {"label": {}, "outlier": {}}
        validated_issues = validator.validate(issue_types)
        assert validated_issues["is_valid"]

        lab.find_issues(
            pred_probs=data["pred_probs"],
            features=data["features"],
            issue_types=validated_issues["validated_config"]
        )

        issues = lab.get_issues()
        assert "is_label_issue" in issues.columns
        assert "is_outlier_issue" in issues.columns



@pytest.mark.classification
class TestDatalabWithCustomParameters:
    """Tests for Datalab with custom validated parameters."""

    def test_custom_k_parameter(self, small_classification):
        """Test with custom 'k' parameter for label issues."""
        data = small_classification

        lab = Datalab(verbosity=4,
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )

        # Validate issue types
        validator = EnhancedIssueTypesValidator(task="classification")
        issue_types = {"label": {"k": 5}}
        validated_issues = validator.validate(issue_types)
        assert validated_issues["is_valid"]

        lab.find_issues(
            pred_probs=data["pred_probs"],
            issue_types=validated_issues["validated_config"]
        )

        issues = lab.get_issues()
        assert "is_label_issue" in issues.columns

    def test_custom_threshold_parameter(self, small_classification):
        """Test with custom threshold for outlier issues."""
        data = small_classification

        lab = Datalab(verbosity=4,
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )

        # Validate issue types
        validator = EnhancedIssueTypesValidator(task="classification")
        issue_types = {"outlier": {"threshold": 0.8}}
        validated_issues = validator.validate(issue_types)
        assert validated_issues["is_valid"]

        lab.find_issues(
            features=data["features"],
            issue_types=validated_issues["validated_config"]
        )

        issues = lab.get_issues()
        assert "is_outlier_issue" in issues.columns



@pytest.mark.classification
class TestDatalabIssueDetection:
    """Tests verifying that Datalab actually detects issues in data."""

    def test_detects_label_issues(self, small_classification):
        """Test that label issues are detected."""
        data = small_classification

        lab = Datalab(verbosity=4,
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )
        lab.find_issues(pred_probs=data["pred_probs"])

        issues = lab.get_issues()
        assert issues["is_label_issue"].any()

    def test_issue_scores_valid_range(self, small_classification):
        """Test that issue scores are in valid range [0, 1]."""
        data = small_classification

        lab = Datalab(verbosity=4,
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )
        lab.find_issues(pred_probs=data["pred_probs"])

        issues = lab.get_issues()
        label_scores = issues["label_score"]

        assert np.all(label_scores >= 0)
        assert np.all(label_scores <= 1)
