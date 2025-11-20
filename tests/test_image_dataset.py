

from cleanlab.datalab.datalab import Datalab
import pytest


def _assert_len(issues_df, n_rows):
    assert issues_df is not None
    assert len(issues_df) == n_rows
# ====================================
# Classification Task
# ====================================
@pytest.mark.fast
@pytest.mark.classification
def test_image_classification_combinations(small_image):
    ds = small_image
    lab = Datalab(
        data=ds["data"],
        label_name=ds["label_name"],
        task="classification",
        image_key="image",
    )


    # # pred_probs -> Label Issue
    # lab.find_issues(pred_probs=ds["pred_probs"])
    # issues = lab.get_issues()
    # _assert_len(issues, len(ds["labels"]))
    # assert "is_label_issue" in issues.columns

    # # features -> Label Issue, Outlier, Near Duplicate, Non-IID, Null (lo que aplique)
    # lab.find_issues(features=ds["features"])
    # issues = lab.get_issues()
    # _assert_len(issues, len(ds["labels"]))
    # assert any(c in issues.columns for c in ["is_outlier_issue", "is_near_duplicate_issue", "is_non_iid_issue", "is_label_issue"])

    # # knn_graph -> Outlier, Near Duplicate, Non-IID, Data Valuation
    # lab.find_issues(knn_graph=ds["knn_graph"])
    # issues = lab.get_issues()
    # _assert_len(issues, len(ds["labels"]))
    # assert any(c in issues.columns for c in ["is_near_duplicate_issue", "is_outlier_issue", "is_non_iid_issue", "is_label_issue"])

    # # pred_probs + features -> Underperforming Group (simplificado sin pasar issue_types)
    # lab.find_issues(pred_probs=ds["pred_probs"], features=ds["features"])
    # issues = lab.get_issues()
    # _assert_len(issues, len(ds["labels"]))
    # assert "is_label_issue" in issues.columns

    # # pred_probs + knn_graph -> Underperforming Group (simplificado)
    # lab.find_issues(pred_probs=ds["pred_probs"], knn_graph=ds["knn_graph"])
    # issues = lab.get_issues()
    # _assert_len(issues, len(ds["labels"]))
    # assert "is_label_issue" in issues.columns

    # # pred_probs + cluster_ids -> Underperforming Group (REQUIERE issue_types) (NO IMPLEMENTADO)
    # # Ejemplo (no ejecutar):
    # # lab.find_issues(
    # #   pred_probs=ds["pred_probs"],
    # #   issue_types={"underperforming_group": {"cluster_ids": ds["cluster_ids"]}}
    # # )

    # # sin argumentos -> Class Imbalance
    # lab.find_issues()
    # issues = lab.get_issues()
    # _assert_len(issues, len(ds["labels"]))
    # # La columna puede aparecer según heurísticas
    # assert "is_class_imbalance_issue" in issues.columns

    print("Testing image dataset with all inputs and custom issue types...")
    print(lab.list_possible_issue_types())

    lab.find_issues(
        pred_probs=ds["pred_probs"],
        features=ds["features"],
        knn_graph=ds["knn_graph"],
        issue_types={
            "image_issue_types": {
                "dark": {"threshold": 0.32}, # `threshold` argument for dark issue type. Non-negative floating value between 0 and 1, lower value implies fewer samples will be marked as issue and vice versa.
                "light": {"threshold": 0.05}, # `threshold` argument for light issue type. Non-negative floating value between 0 and 1, lower value implies fewer samples will be marked as issue and vice versa.
                "blurry": {"threshold": 0.29}, # `threshold` argument for blurry issue type. Non-negative floating value between 0 and 1, lower value implies fewer samples will be marked as issue and vice versa.
                "low_information": {"threshold": 0.3}, # `threshold` argument for low_information issue type. Non-negative floating value between 0 and 1, lower value implies fewer samples will be marked as issue and vice versa.
                "odd_aspect_ratio": {"threshold": 0.35}, # `threshold` argument for odd_aspect_ratio issue type. Non-negative floating value between 0 and 1, lower value implies fewer samples will be marked as issue and vice versa.
                "odd_size": {"threshold": 10.0}, # `threshold` argument for odd_size issue type. Non-negative integer value between starting from 0, unlike other issues, here higher value implies fewer samples will be selected.
            }
        }
    )
