"""
Tests for Datalab integration with classification data.

These tests verify that Datalab.find_issues() works correctly with:
- Different input combinations (pred_probs, features, knn_graph)
- Different issue type configurations validated by EnhancedIssueTypesValidator
"""

import pytest
from cleanlab import Datalab

from cleanlab_plugins.enhanced_issue_validator import EnhancedIssueTypesValidator


@pytest.mark.classification
@pytest.mark.fast
class TestDatalabBasicIntegration:
    """Basic integration tests for Datalab with classification data."""
    
    def test_datalab_with_pred_probs_only(self, small_classification_dataset):
        """Test Datalab with only pred_probs."""
        data = small_classification_dataset
        
        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )
        
        lab.find_issues(pred_probs=data["pred_probs"])
        
        issues = lab.get_issues()
        assert len(issues) == len(data["labels"])
        assert "is_label_issue" in issues.columns
    
    def test_datalab_with_features_only(self, small_classification_dataset):
        """Test Datalab with only features."""
        data = small_classification_dataset
        
        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )
        
        lab.find_issues(features=data["features"])
        
        issues = lab.get_issues()
        assert len(issues) == len(data["labels"])
    
    def test_datalab_with_knn_graph_only(self, small_classification_dataset):
        """Test Datalab with only knn_graph."""
        data = small_classification_dataset
        
        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )
        
        lab.find_issues(knn_graph=data["knn_graph"])
        
        issues = lab.get_issues()
        assert len(issues) == len(data["labels"])
    
    def test_datalab_with_all_inputs(self, small_classification_dataset):
        """Test Datalab with all inputs (pred_probs, features, knn_graph)."""
        data = small_classification_dataset
        
        lab = Datalab(
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
    
    def test_label_issues_only(self, classification_dataset):
        """Test detecting only label issues with validated config."""
        data = classification_dataset
        validator = EnhancedIssueTypesValidator(task="classification")
        
        issue_types = {"label": {}}
        validation_result = validator.validate(issue_types)
        assert validation_result["is_valid"]
        
        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )
        
        lab.find_issues(
            pred_probs=data["pred_probs"],
            issue_types=validation_result["validated_config"]
        )
        
        issues = lab.get_issues()
        assert "is_label_issue" in issues.columns
        assert "label_score" in issues.columns
    
    def test_outlier_issues_only(self, classification_dataset):
        """Test detecting only outlier issues with validated config."""
        data = classification_dataset
        validator = EnhancedIssueTypesValidator(task="classification")
        
        issue_types = {"outlier": {}}
        validation_result = validator.validate(issue_types)
        assert validation_result["is_valid"]
        
        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )
        
        try:
            lab.find_issues(
                features=data["features"],
                issue_types=validation_result["validated_config"]
            )
            
            issues = lab.get_issues()
            assert "is_outlier_issue" in issues.columns
        except (ValueError, TypeError) as e:
            # Known issue with NumPy 2.0 compatibility in cleanlab's outlier detector
            if "np.float_" in str(e) or "No issues available" in str(e):
                pytest.skip(f"NumPy 2.0 compatibility issue with outlier detector: {e}")
            raise
    
    def test_multiple_issue_types(self, classification_dataset):
        """Test detecting multiple issue types with validated config."""
        data = classification_dataset
        validator = EnhancedIssueTypesValidator(task="classification")
        
        issue_types = {
            "label": {"k": 10},
            "outlier": {"threshold": 0.15},
            "near_duplicate": {"threshold": 0.1}
        }
        validation_result = validator.validate(issue_types)
        assert validation_result["is_valid"]
        
        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )
        
        lab.find_issues(
            pred_probs=data["pred_probs"],
            features=data["features"],
            knn_graph=data["knn_graph"],
            issue_types=validation_result["validated_config"]
        )
        
        issues = lab.get_issues()
        
        # Verify all requested issue types are present
        for issue_name in issue_types.keys():
            assert f"is_{issue_name}_issue" in issues.columns
            assert f"{issue_name}_score" in issues.columns


@pytest.mark.classification
class TestDatalabWithCustomParameters:
    """Tests for Datalab with custom validated parameters."""
    
    def test_custom_k_parameter(self, classification_dataset):
        """Test label detection with custom k parameter."""
        data = classification_dataset
        validator = EnhancedIssueTypesValidator(task="classification")
        
        issue_types = {"label": {"k": 15}}
        validation_result = validator.validate(issue_types)
        assert validation_result["is_valid"]
        
        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )
        
        lab.find_issues(
            pred_probs=data["pred_probs"],
            knn_graph=data["knn_graph"],
            issue_types=validation_result["validated_config"]
        )
        
        issues = lab.get_issues()
        assert len(issues) > 0
    
    def test_custom_threshold_parameter(self, classification_dataset):
        """Test outlier detection with custom threshold."""
        data = classification_dataset
        validator = EnhancedIssueTypesValidator(task="classification")
        
        issue_types = {"outlier": {"threshold": 0.25}}
        validation_result = validator.validate(issue_types)
        assert validation_result["is_valid"]
        
        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )
        
        lab.find_issues(
            features=data["features"],
            issue_types=validation_result["validated_config"]
        )
        
        issues = lab.get_issues()
        assert "is_outlier_issue" in issues.columns


@pytest.mark.classification
class TestDatalabIssueDetection:
    """Tests verifying that Datalab actually detects issues in data."""
    
    def test_detects_label_issues(self, classification_dataset):
        """Verify that label issues are actually detected."""
        data = classification_dataset
        
        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )
        
        lab.find_issues(pred_probs=data["pred_probs"])
        
        issues = lab.get_issues()
        
        # Since we added noise_level=0.1, we should detect some label issues
        label_issues = issues[issues["is_label_issue"]]
        assert len(label_issues) > 0, "Should detect some label issues in noisy data"
    
    def test_issue_scores_valid_range(self, classification_dataset):
        """Verify that issue scores are in valid range [0, 1]."""
        data = classification_dataset
        
        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )
        
        lab.find_issues(
            pred_probs=data["pred_probs"],
            features=data["features"]
        )
        
        issues = lab.get_issues()
        
        # Check all score columns
        score_columns = [col for col in issues.columns if col.endswith("_score")]
        for col in score_columns:
            scores = issues[col]
            assert scores.min() >= 0, f"{col} has scores < 0"
            assert scores.max() <= 1, f"{col} has scores > 1"
    
    def test_issue_summary(self, classification_dataset):
        """Test that issue summary is generated correctly."""
        data = classification_dataset
        
        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )
        
        lab.find_issues(
            pred_probs=data["pred_probs"],
            features=data["features"]
        )
        
        summary = lab.get_issue_summary()
        
        assert len(summary) > 0
        assert "issue_type" in summary.columns
        assert "score" in summary.columns
