"""
Simple example tests using the new simplified fixtures.

This module demonstrates how to use the new fixtures with Datalab.
Use these tests as a guide to update other test files.
"""

from cleanlab import Datalab
import pytest


class TestSimpleClassification:
    """Tests for classification task using simplified fixtures."""

    @pytest.mark.fast
    @pytest.mark.classification
    def test_classification_with_pred_probs_only(self, small_classification):
        """Test classification with only pred_probs (minimal requirements)."""
        dataset = small_classification

        # Create Datalab instance
        lab = Datalab(data=dataset["data"], label_name=dataset["label_name"], task="classification")

        # Find issues using only pred_probs
        lab.find_issues(pred_probs=dataset["pred_probs"])

        # Verify issues were found
        assert lab.issues is not None
        assert len(lab.issues) == len(dataset["labels"])
        assert "is_label_issue" in lab.issues.columns

        # Verify summary
        summary = lab.get_issue_summary()
        assert summary is not None
        # assert "label" in summary.index

        assert "is_label_issue" in lab.issues.columns

    @pytest.mark.fast
    @pytest.mark.classification
    def test_classification_with_all_inputs(self, small_classification):
        """Test classification with all inputs (comprehensive audit)."""
        dataset = small_classification

        # Create Datalab instance
        lab = Datalab(data=dataset["data"], label_name=dataset["label_name"], task="classification")

        # Find issues with all inputs
        lab.find_issues(pred_probs=dataset["pred_probs"], features=dataset["features"], knn_graph=dataset["knn_graph"])

        # Verify multiple issue types were checked
        # summary = lab.get_issue_summary()
        # expected_issues = ["label", "outlier", "near_duplicate", "non_iid", "class_imbalance"]

        # for issue_type in expected_issues:
        #     assert issue_type in summary.index, f"Expected issue type '{issue_type}' not found"

        assert "is_label_issue" in lab.issues.columns
        assert "is_outlier_issue" in lab.issues.columns
        assert "is_near_duplicate_issue" in lab.issues.columns
        assert "is_non_iid_issue" in lab.issues.columns
        assert "is_class_imbalance_issue" in lab.issues.columns

    @pytest.mark.slow
    @pytest.mark.classification
    def test_classification_medium_dataset(self, medium_classification):
        """Test classification with medium dataset."""
        dataset = medium_classification

        lab = Datalab(data=dataset["data"], label_name=dataset["label_name"], task="classification")

        lab.find_issues(pred_probs=dataset["pred_probs"], features=dataset["features"])

        assert len(lab.issues) == len(dataset["labels"])
        assert lab.get_issue_summary() is not None


class TestSimpleMultilabel:
    """Tests for multilabel task using simplified fixtures."""

    @pytest.mark.fast
    @pytest.mark.multilabel
    def test_multilabel_with_pred_probs(self, small_multilabel):
        """Test multilabel classification."""
        dataset = small_multilabel

        lab = Datalab(data=dataset["data"], label_name=dataset["label_name"], task="multilabel")

        lab.find_issues(pred_probs=dataset["pred_probs"])

        assert lab.issues is not None
        assert len(lab.issues) == len(dataset["labels"])
        assert "is_label_issue" in lab.issues.columns

    @pytest.mark.fast
    @pytest.mark.multilabel
    def test_multilabel_with_features(self, small_multilabel):
        """Test multilabel with features."""
        dataset = small_multilabel

        lab = Datalab(data=dataset["data"], label_name=dataset["label_name"], task="multilabel")

        lab.find_issues(pred_probs=dataset["pred_probs"], features=dataset["features"])

        summary = lab.get_issue_summary()
        expected_issues = ["label", "outlier", "near_duplicate", "non_iid"]

        for issue_type in expected_issues:
            assert issue_type in summary.index


class TestSimpleRegression:
    """Tests for regression task using simplified fixtures."""

    @pytest.mark.fast
    @pytest.mark.regression
    def test_regression_with_predictions(self, small_regression):
        """Test regression with predictions only."""
        dataset = small_regression

        lab = Datalab(data=dataset["data"], label_name=dataset["label_name"], task="regression")

        # For regression, pred_probs is actually predictions (1D array)
        lab.find_issues(pred_probs=dataset["pred_probs"])

        assert lab.issues is not None
        assert len(lab.issues) == len(dataset["labels"])
        assert "is_label_issue" in lab.issues.columns

    @pytest.mark.fast
    @pytest.mark.regression
    def test_regression_with_features(self, small_regression):
        """Test regression with features."""
        dataset = small_regression

        lab = Datalab(data=dataset["data"], label_name=dataset["label_name"], task="regression")

        lab.find_issues(pred_probs=dataset["pred_probs"], features=dataset["features"])

        summary = lab.get_issue_summary()
        expected_issues = ["label", "outlier", "near_duplicate", "non_iid"]

        for issue_type in expected_issues:
            assert issue_type in summary.index


class TestSimpleImage:
    """Tests for image classification using simplified fixtures."""

    @pytest.mark.fast
    @pytest.mark.image
    def test_image_classification_basic(self, small_image):
        """Test image classification with basic inputs."""
        dataset = small_image

        lab = Datalab(data=dataset["data"], label_name=dataset["label_name"], task="classification")

        lab.find_issues(pred_probs=dataset["pred_probs"], features=dataset["features"])

        assert lab.issues is not None
        assert len(lab.issues) == len(dataset["labels"])

    @pytest.mark.fast
    @pytest.mark.image
    def test_image_classification_with_image_key(self, small_image):
        """Test image classification with image_key for image-specific issues."""
        dataset = small_image

        lab = Datalab(
            data=dataset["data"],
            label_name=dataset["label_name"],
            image_key=dataset["image_key"],  # This enables image-specific checks
            task="classification",
        )

        lab.find_issues(pred_probs=dataset["pred_probs"], features=dataset["features"])

        assert lab.issues is not None

        # When image_key is provided, additional image-specific issues may be detected
        summary = lab.get_issue_summary()
        # assert "label" in summary.index

        assert "is_label_issue" in lab.issues.columns

    @pytest.mark.slow
    @pytest.mark.image
    def test_image_medium_dataset(self, medium_image):
        """Test image classification with medium dataset."""
        dataset = medium_image

        lab = Datalab(
            data=dataset["data"],
            label_name=dataset["label_name"],
            image_key=dataset["image_key"],
            task="classification",
        )

        lab.find_issues(pred_probs=dataset["pred_probs"], features=dataset["features"], knn_graph=dataset["knn_graph"])

        assert len(lab.issues) == len(dataset["labels"])
        assert lab.get_issue_summary() is not None


class TestIssueTypes:
    """Tests for specific issue types detection."""

    @pytest.mark.fast
    @pytest.mark.classification
    def test_label_issues_only(self, small_classification):
        """Test detecting only label issues."""
        dataset = small_classification

        lab = Datalab(data=dataset["data"], label_name=dataset["label_name"], task="classification")

        # Specify only label issues
        issue_types = {"label": {}}

        lab.find_issues(pred_probs=dataset["pred_probs"], issue_types=issue_types)

        summary = lab.get_issue_summary()
        assert len(summary) == 1
        # assert "label" in summary.index

        assert "is_label_issue" in lab.issues.columns

    @pytest.mark.fast
    @pytest.mark.classification
    def test_multiple_specific_issues(self, small_classification):
        """Test detecting multiple specific issue types."""
        dataset = small_classification

        lab = Datalab(data=dataset["data"], label_name=dataset["label_name"], task="classification")

        # Specify multiple issue types
        issue_types = {"label": {}, "outlier": {}, "near_duplicate": {}}

        lab.find_issues(pred_probs=dataset["pred_probs"], features=dataset["features"], issue_types=issue_types)

        summary = lab.get_issue_summary()
        # assert "label" in summary.index
        # assert "outlier" in summary.index
        # assert "near_duplicate" in summary.index
        assert "is_label_issue" in lab.issues.columns
        assert "is_outlier_issue" in lab.issues.columns
        assert "is_near_duplicate_issue" in lab.issues.columns


class TestDataFormats:
    """Tests for different data formats."""

    @pytest.mark.fast
    @pytest.mark.classification
    def test_huggingface_dataset_format(self, small_classification):
        """Test with HuggingFace Dataset format."""
        dataset = small_classification

        # Verify it's a HuggingFace Dataset
        from datasets import Dataset

        assert isinstance(dataset["data"], Dataset)

        lab = Datalab(data=dataset["data"], label_name=dataset["label_name"], task="classification")

        lab.find_issues(pred_probs=dataset["pred_probs"])

        assert lab.issues is not None
        assert len(lab.issues) == len(dataset["data"])
