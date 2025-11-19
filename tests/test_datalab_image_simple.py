"""
Simplified tests for Datalab integration with image classification data.

These tests verify that Datalab.find_issues() works correctly with image data
using the available fixtures from conftest.py: small_image and medium_image.
"""

import pytest
from cleanlab import Datalab
from cleanlab_plugins.enhanced_issue_validator import EnhancedIssueTypesValidator


@pytest.mark.image
@pytest.mark.fast
class TestDatalabImageBasic:
    """Basic integration tests for Datalab with image data."""

    def test_datalab_image_basic_functionality(self, small_image):
        """Test basic Datalab functionality with image data."""
        data = small_image

        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task=data["task"]
        )

        lab.find_issues(
            pred_probs=data["pred_probs"],
            features=data["features"],
            knn_graph=data["knn_graph"]
        )

        issues = lab.get_issues()
        assert len(issues) == len(data["labels"])
        assert "is_label_issue" in issues.columns

    def test_datalab_image_with_features_only(self, small_image):
        """Test using only features extracted from images."""
        data = small_image

        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task=data["task"]
        )

        lab.find_issues(features=data["features"])

        issues = lab.get_issues()
        assert len(issues) == len(data["labels"])

    def test_datalab_image_with_pred_probs_only(self, small_image):
        """Test using only pred_probs (without features)."""
        data = small_image

        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task=data["task"]
        )

        lab.find_issues(pred_probs=data["pred_probs"])

        issues = lab.get_issues()
        assert len(issues) == len(data["labels"])
        assert "is_label_issue" in issues.columns


@pytest.mark.image
class TestDatalabImageWithValidator:
    """Tests for Datalab using the EnhancedIssueTypesValidator with image data."""

    def test_image_label_issues_only(self, small_image):
        """Test detecting only label issues with image data."""
        data = small_image
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

    def test_image_outlier_detection(self, small_image):
        """Test detecting outlier issues with image features."""
        data = small_image
        validator = EnhancedIssueTypesValidator(task="classification")

        issue_types = {"outlier": {}}
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

    def test_image_near_duplicate_detection(self, small_image):
        """Test detecting near duplicate issues in image data."""
        data = small_image
        validator = EnhancedIssueTypesValidator(task="classification")

        issue_types = {"near_duplicate": {"threshold": 0.1}}
        validation_result = validator.validate(issue_types)
        assert validation_result["is_valid"]

        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )

        lab.find_issues(
            features=data["features"],
            knn_graph=data["knn_graph"],
            issue_types=validation_result["validated_config"]
        )

        issues = lab.get_issues()
        assert "is_near_duplicate_issue" in issues.columns

    def test_image_multiple_issue_types(self, small_image):
        """Test detecting multiple issue types with image data."""
        data = small_image
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
        
        # Check that multiple issue types were detected
        issue_columns = [col for col in issues.columns if col.startswith("is_") and col.endswith("_issue")]
        assert len(issue_columns) >= 2


@pytest.mark.image
class TestDatalabImageCustomParameters:
    """Tests for Datalab with custom validated parameters on image data."""

    def test_image_custom_k_parameter(self, small_image):
        """Test custom k parameter for label issues."""
        data = small_image
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
            issue_types=validation_result["validated_config"]
        )

        issues = lab.get_issues()
        assert "is_label_issue" in issues.columns

    def test_image_custom_threshold_parameter(self, small_image):
        """Test custom threshold parameter for outliers."""
        data = small_image
        validator = EnhancedIssueTypesValidator(task="classification")

        issue_types = {"outlier": {"threshold": 0.2}}
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


@pytest.mark.image
class TestDatalabImageMediumDataset:
    """Tests using medium image dataset for more comprehensive testing."""

    def test_medium_image_dataset_basic(self, medium_image):
        """Test basic functionality with medium image dataset."""
        data = medium_image

        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task=data["task"]
        )

        lab.find_issues(
            pred_probs=data["pred_probs"],
            features=data["features"],
            knn_graph=data["knn_graph"]
        )

        issues = lab.get_issues()
        assert len(issues) == len(data["labels"])
        assert len(issues) == 200  # medium dataset size

    def test_medium_image_issue_detection(self, medium_image):
        """Test that issues are actually detected in medium dataset."""
        data = medium_image
        validator = EnhancedIssueTypesValidator(task="classification")

        issue_types = {
            "label": {},
            "outlier": {},
            "near_duplicate": {}
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
        
        # Check that some issues are detected
        label_issues = issues["is_label_issue"].sum()
        outlier_issues = issues["is_outlier_issue"].sum()
        duplicate_issues = issues["is_near_duplicate_issue"].sum()
        
        # At least one type should detect some issues
        assert (label_issues + outlier_issues + duplicate_issues) > 0


@pytest.mark.image
@pytest.mark.fast
class TestDatalabImageDataValidation:
    """Tests for validating the image data format and content."""

    def test_image_data_structure(self, small_image):
        """Test that image data has the expected structure."""
        data = small_image
        
        # Check required keys
        required_keys = ["data", "labels", "pred_probs", "features", "knn_graph", "label_name", "task"]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

        # Check data types and shapes
        assert data["task"] == "classification"
        assert len(data["labels"]) == 50  # small dataset size
        assert data["pred_probs"].shape[0] == 50
        assert data["features"].shape[0] == 50
        assert data["knn_graph"].shape[0] == 50

    def test_image_scores_valid_range(self, small_image):
        """Test that issue scores are in valid range [0, 1]."""
        data = small_image

        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task=data["task"]
        )

        lab.find_issues(
            pred_probs=data["pred_probs"],
            features=data["features"],
            knn_graph=data["knn_graph"]
        )

        issues = lab.get_issues()
        
        # Check score columns
        score_columns = [col for col in issues.columns if col.endswith("_score")]
        for col in score_columns:
            scores = issues[col]
            assert scores.min() >= 0.0, f"{col} has scores below 0"
            assert scores.max() <= 1.0, f"{col} has scores above 1"
