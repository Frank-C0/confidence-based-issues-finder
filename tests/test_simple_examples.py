"""Minimal baseline tests for dataset types y combinaciones de find_issues.

Clasificación e Imágenes (task="classification"):
1. pred_probs -> label issues
2. features -> outlier / near_duplicate / non_iid (cualquiera que aparezca)
3. sin argumentos -> class imbalance
4. pred_probs + features -> mezcla (grupo subóptimo simplificado)
5. knn_graph -> posibles near_duplicate / otras

Regresión:
1. pred_probs
2. features (+ pred_probs para enriquecer)

Multilabel:
1. pred_probs
2. features (+ pred_probs)

Las aserciones son deliberadamente mínimas para ser base inicial.
"""

from cleanlab import Datalab
import pytest


def _assert_basic(issues_df, n_rows):
    assert issues_df is not None
    assert len(issues_df) == n_rows


@pytest.mark.fast
@pytest.mark.classification
def test_minimal_classification_combinations(small_classification):
    dataset = small_classification
    lab = Datalab(data=dataset["data"], label_name=dataset["label_name"], task="classification")

    # 1 pred_probs
    lab.find_issues(pred_probs=dataset["pred_probs"])
    issues = lab.get_issues()
    _assert_basic(issues, len(dataset["labels"]))
    assert "is_label_issue" in issues.columns

    # 2 features
    lab.find_issues(features=dataset["features"])
    issues = lab.get_issues()
    _assert_basic(issues, len(dataset["labels"]))
    assert any(c in issues.columns for c in ["is_outlier_issue", "is_near_duplicate_issue", "is_non_iid_issue"])

    # 3 sin argumentos
    lab.find_issues()
    issues = lab.get_issues()
    _assert_basic(issues, len(dataset["labels"]))
    assert "is_class_imbalance_issue" in issues.columns

    # 4 pred_probs + features
    lab.find_issues(pred_probs=dataset["pred_probs"], features=dataset["features"])
    issues = lab.get_issues()
    _assert_basic(issues, len(dataset["labels"]))
    assert "is_label_issue" in issues.columns

    # 5 knn_graph
    lab.find_issues(knn_graph=dataset["knn_graph"])
    issues = lab.get_issues()
    _assert_basic(issues, len(dataset["labels"]))
    assert any(c in issues.columns for c in ["is_near_duplicate_issue", "is_outlier_issue", "is_label_issue"])


@pytest.mark.fast
@pytest.mark.image
def test_minimal_image_classification_combinations(small_image):
    dataset = small_image
    lab = Datalab(data=dataset["data"], label_name=dataset["label_name"], task="classification")

    lab.find_issues(pred_probs=dataset["pred_probs"])
    issues = lab.get_issues()
    _assert_basic(issues, len(dataset["labels"]))
    assert "is_label_issue" in issues.columns

    lab.find_issues(features=dataset["features"])
    issues = lab.get_issues()
    _assert_basic(issues, len(dataset["labels"]))
    assert any(c in issues.columns for c in ["is_outlier_issue", "is_near_duplicate_issue", "is_non_iid_issue"])

    lab.find_issues()
    issues = lab.get_issues()
    _assert_basic(issues, len(dataset["labels"]))
    assert "is_class_imbalance_issue" in issues.columns

    lab.find_issues(pred_probs=dataset["pred_probs"], features=dataset["features"])
    issues = lab.get_issues()
    _assert_basic(issues, len(dataset["labels"]))
    assert "is_label_issue" in issues.columns

    lab.find_issues(knn_graph=dataset["knn_graph"])
    issues = lab.get_issues()
    _assert_basic(issues, len(dataset["labels"]))
    assert any(c in issues.columns for c in ["is_near_duplicate_issue", "is_outlier_issue", "is_label_issue"])


@pytest.mark.fast
@pytest.mark.regression
def test_minimal_regression_combinations(small_regression):
    dataset = small_regression
    lab = Datalab(data=dataset["data"], label_name=dataset["label_name"], task="regression")

    lab.find_issues(pred_probs=dataset["pred_probs"])
    issues = lab.get_issues()
    _assert_basic(issues, len(dataset["labels"]))
    assert "is_label_issue" in issues.columns

    lab.find_issues(features=dataset["features"], pred_probs=dataset["pred_probs"])
    issues = lab.get_issues()
    _assert_basic(issues, len(dataset["labels"]))
    assert any(
        c in issues.columns
        for c in ["is_outlier_issue", "is_near_duplicate_issue", "is_non_iid_issue", "is_label_issue"]
    )


@pytest.mark.fast
@pytest.mark.multilabel
def test_minimal_multilabel_combinations(small_multilabel):
    dataset = small_multilabel
    lab = Datalab(data=dataset["data"], label_name=dataset["label_name"], task="multilabel")

    lab.find_issues(pred_probs=dataset["pred_probs"])
    issues = lab.get_issues()
    _assert_basic(issues, len(dataset["labels"]))
    assert "is_label_issue" in issues.columns

    lab.find_issues(features=dataset["features"], pred_probs=dataset["pred_probs"])
    issues = lab.get_issues()
    _assert_basic(issues, len(dataset["labels"]))
    assert any(
        c in issues.columns
        for c in ["is_outlier_issue", "is_near_duplicate_issue", "is_non_iid_issue", "is_label_issue"]
    )
