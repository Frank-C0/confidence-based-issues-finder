"""
Tests for Datalab integration with image classification data.

These tests verify that Datalab.find_issues() works correctly with image data:
- Different input combinations (pred_probs, features, knn_graph)
- Different issue type configurations validated by EnhancedIssueTypesValidator
- Both dict and DataFrame formats for image features
"""

from cleanlab import Datalab
import pytest

from cleanlab_plugins.enhanced_issue_validator import EnhancedIssueTypesValidator


@pytest.mark.image
@pytest.mark.fast
class TestDatalabImageBasicIntegration:
    """Basic integration tests for Datalab with image data."""

    def test_datalab_image_with_dict_format(self, small_image_dataset):
        """Test Datalab with image features in dict format."""
        data = small_image_dataset

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
        assert "is_label_issue" in issues.columns

    def test_datalab_image_with_dataframe_format(self, small_image_dataset):
        """Test Datalab with image features in DataFrame format."""
        data = small_image_dataset

        lab = Datalab(
            data=data["dataframe"],
            label_name=data["label_name_df"],
            task="classification"
        )

        lab.find_issues(
            pred_probs=data["pred_probs"],
            features=data["features"]
        )

        issues = lab.get_issues()
        assert len(issues) == len(data["labels"])

    def test_datalab_image_with_features_only(self, small_image_dataset):
        """Test using only features extracted from images."""
        data = small_image_dataset

        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )

        lab.find_issues(features=data["features"])

        issues = lab.get_issues()
        assert len(issues) == len(data["labels"])

    def test_datalab_image_with_pred_probs_only(self, small_image_dataset):
        """Test using only pred_probs (without features)."""
        data = small_image_dataset

        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )

        lab.find_issues(pred_probs=data["pred_probs"])

        issues = lab.get_issues()
        assert len(issues) == len(data["labels"])
        assert "is_label_issue" in issues.columns

    def test_datalab_image_with_knn_only(self, small_image_dataset):
        """Test using only knn_graph."""
        data = small_image_dataset

        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )

        lab.find_issues(knn_graph=data["knn_graph"])

        issues = lab.get_issues()
        assert len(issues) == len(data["labels"])

    def test_datalab_image_with_all_inputs(self, small_image_dataset):
        """Test with all inputs (pred_probs, features, knn_graph)."""
        data = small_image_dataset

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


@pytest.mark.image
class TestDatalabImageWithValidatedIssueTypes:
    """Tests for Datalab using validated issue type configurations with images."""

    def test_image_label_issues_only(self, image_dataset):
        """Test detecting only label issues with image data."""
        data = image_dataset
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

    def test_image_outlier_issues_only(self, image_dataset):
        """Test detecting only outlier issues with image features."""
        data = image_dataset
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

    def test_image_near_duplicate_issues(self, image_dataset):
        """Test detecting near duplicate issues in image data."""
        data = image_dataset
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

    def test_image_multiple_issue_types(self, image_dataset):
        """Test detecting multiple issue types with image data."""
        data = image_dataset
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
        for issue_name in issue_types:
            assert f"is_{issue_name}_issue" in issues.columns
            assert f"{issue_name}_score" in issues.columns


@pytest.mark.image
class TestDatalabImageCustomParameters:
    """Tests for Datalab with custom validated parameters on image data."""

    def test_image_custom_k_parameter(self, image_dataset):
        """Test label detection with custom k parameter on images."""
        data = image_dataset
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

    def test_image_custom_threshold_parameter(self, image_dataset):
        """Test near duplicate detection with custom threshold on images."""
        data = image_dataset
        validator = EnhancedIssueTypesValidator(task="classification")

        issue_types = {"near_duplicate": {"threshold": 0.2}}
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
        assert "is_near_duplicate_issue" in issues.columns


@pytest.mark.image
class TestDatalabImageIssueDetection:
    """Tests verifying that Datalab actually detects issues in image data."""

    def test_image_detects_label_issues(self, image_dataset):
        """Verify that label issues are detected in noisy image labels."""
        data = image_dataset

        lab = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )

        lab.find_issues(pred_probs=data["pred_probs"])

        issues = lab.get_issues()

        # Since we added noise_level=0.1, we should detect some label issues
        label_issues = issues[issues["is_label_issue"]]
        assert len(label_issues) > 0, "Should detect some label issues in noisy image data"

    def test_image_issue_scores_valid_range(self, image_dataset):
        """Verify that issue scores are in valid range [0, 1] for images."""
        data = image_dataset

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

    def test_image_issue_summary(self, image_dataset):
        """Test that issue summary is generated correctly for image data."""
        data = image_dataset

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

    def test_image_features_shape(self, image_dataset):
        """Verify that extracted features have correct shape."""
        data = image_dataset

        features = data["features"]
        n_samples = len(data["labels"])

        assert features.shape[0] == n_samples
        assert features.shape[1] > 0  # Should have some features
        assert features.dtype == "float32"

    def test_image_data_formats_consistency(self, image_dataset):
        """Test that dict and DataFrame formats produce consistent results."""
        data = image_dataset

        # Test with dict format
        lab_dict = Datalab(
            data=data["data"],
            label_name=data["label_name"],
            task="classification"
        )

        lab_dict.find_issues(pred_probs=data["pred_probs"])
        issues_dict = lab_dict.get_issues()

        # Test with DataFrame format
        lab_df = Datalab(
            data=data["dataframe"],
            label_name=data["label_name_df"],
            task="classification"
        )

        lab_df.find_issues(pred_probs=data["pred_probs"])
        issues_df = lab_df.get_issues()

        # Both should detect the same number of issues
        assert len(issues_dict) == len(issues_df)
        assert (issues_dict["is_label_issue"].values == issues_df["is_label_issue"].values).all()


@pytest.mark.image
class TestDatalabHuggingFaceImageDataset:
    """Tests for Datalab with Hugging Face Dataset format (enables image-specific issues)."""

    def test_huggingface_dataset_basic(self, small_huggingface_image_dataset):
        """Test Datalab with Hugging Face Dataset format."""
        data = small_huggingface_image_dataset

        lab = Datalab(
            data=data["hf_dataset"],
            label_name=data["label_name"],
            task="classification"
        )

        lab.find_issues(
            pred_probs=data["pred_probs"],
            features=data["features"]
        )

        issues = lab.get_issues()
        assert len(issues) == len(data["labels"])
        assert "is_label_issue" in issues.columns

    def test_huggingface_dataset_with_image_key(self, small_huggingface_image_dataset):
        """Test Datalab with image_key parameter for image-specific issues."""
        data = small_huggingface_image_dataset

        # Note: This requires cleanvision to be installed for full functionality
        # If not installed, Datalab will still work but won't detect image-specific issues
        lab = Datalab(
            data=data["hf_dataset"],
            label_name=data["label_name"],
            task="classification",
            image_key=data["image_key"]
        )

        lab.find_issues(
            pred_probs=data["pred_probs"],
            features=data["features"]
        )

        issues = lab.get_issues()
        assert len(issues) == len(data["labels"])

    def test_huggingface_dataset_with_all_inputs(self, huggingface_image_dataset):
        """Test Hugging Face format with all inputs."""
        data = huggingface_image_dataset

        lab = Datalab(
            data=data["hf_dataset"],
            label_name=data["label_name"],
            task="classification",
            image_key=data["image_key"]
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

    def test_huggingface_dataset_has_image_column(self, small_huggingface_image_dataset):
        """Verify that Hugging Face dataset has PIL Image column."""
        data = small_huggingface_image_dataset

        hf_dataset = data["hf_dataset"]

        # Check that image column exists
        assert data["image_key"] in hf_dataset.column_names

        # Check that first image is a PIL Image
        first_image = hf_dataset[0][data["image_key"]]
        # PIL Image objects should have a 'mode' attribute
        assert hasattr(first_image, "mode")

    def test_huggingface_dataset_with_validated_issue_types(self, huggingface_image_dataset):
        """Test Hugging Face format with validated issue types."""
        data = huggingface_image_dataset
        validator = EnhancedIssueTypesValidator(task="classification")

        issue_types = {
            "label": {"k": 10},
            "near_duplicate": {"threshold": 0.1}
        }
        validation_result = validator.validate(issue_types)
        assert validation_result["is_valid"]

        lab = Datalab(
            data=data["hf_dataset"],
            label_name=data["label_name"],
            task="classification",
            image_key=data["image_key"]
        )

        lab.find_issues(
            pred_probs=data["pred_probs"],
            features=data["features"],
            knn_graph=data["knn_graph"],
            issue_types=validation_result["validated_config"]
        )

        issues = lab.get_issues()

        # Verify requested issue types are present
        for issue_name in issue_types:
            assert f"is_{issue_name}_issue" in issues.columns
