"""Tests extendidos de combinaciones de inputs para Datalab.

Se prueban las combinaciones solicitadas para cada tipo de dataset.
Se omiten (solo comentario) las combinaciones que requieren configurar
`issue_types` explícitamente (p.ej. cluster_ids, model en regresión).

Objetivo: servir como base para futuros tests más personalizados.
Asserts mínimos para verificar que se generan filas y algunas columnas clave.
"""

from cleanlab import Datalab
import pytest


def _assert_len(issues_df, n_rows):
    assert issues_df is not None
    assert len(issues_df) == n_rows


# ====================================
# Classification Task
# ====================================
@pytest.mark.fast
@pytest.mark.classification
def test_classification_combinations(small_classification):
    ds = small_classification
    lab = Datalab(data=ds["data"], label_name=ds["label_name"], task="classification")

    # pred_probs -> Label Issue
    lab.find_issues(pred_probs=ds["pred_probs"])
    issues = lab.get_issues()
    _assert_len(issues, len(ds["labels"]))
    assert "is_label_issue" in issues.columns

    # features -> Label Issue, Outlier, Near Duplicate, Non-IID, Null (lo que aplique)
    lab.find_issues(features=ds["features"])
    issues = lab.get_issues()
    _assert_len(issues, len(ds["labels"]))
    assert any(c in issues.columns for c in ["is_outlier_issue", "is_near_duplicate_issue", "is_non_iid_issue", "is_label_issue"])

    # knn_graph -> Outlier, Near Duplicate, Non-IID, Data Valuation
    lab.find_issues(knn_graph=ds["knn_graph"])
    issues = lab.get_issues()
    _assert_len(issues, len(ds["labels"]))
    assert any(c in issues.columns for c in ["is_near_duplicate_issue", "is_outlier_issue", "is_non_iid_issue", "is_label_issue"])

    # pred_probs + features -> Underperforming Group (simplificado sin pasar issue_types)
    lab.find_issues(pred_probs=ds["pred_probs"], features=ds["features"])
    issues = lab.get_issues()
    _assert_len(issues, len(ds["labels"]))
    assert "is_label_issue" in issues.columns

    # pred_probs + knn_graph -> Underperforming Group (simplificado)
    lab.find_issues(pred_probs=ds["pred_probs"], knn_graph=ds["knn_graph"])
    issues = lab.get_issues()
    _assert_len(issues, len(ds["labels"]))
    assert "is_label_issue" in issues.columns

    # pred_probs + cluster_ids -> Underperforming Group (REQUIERE issue_types) (NO IMPLEMENTADO)
    # Ejemplo (no ejecutar):
    # lab.find_issues(
    #   pred_probs=ds["pred_probs"],
    #   issue_types={"underperforming_group": {"cluster_ids": ds["cluster_ids"]}}
    # )

    # sin argumentos -> Class Imbalance
    lab.find_issues()
    issues = lab.get_issues()
    _assert_len(issues, len(ds["labels"]))
    # La columna puede aparecer según heurísticas
    assert "is_class_imbalance_issue" in issues.columns


# ====================================
# Regression Task
# ====================================
@pytest.mark.fast
@pytest.mark.regression
def test_regression_combinations(small_regression):
    ds = small_regression
    lab = Datalab(data=ds["data"], label_name=ds["label_name"], task="regression")

    # pred_probs -> Label Issue
    lab.find_issues(pred_probs=ds["pred_probs"])
    issues = lab.get_issues()
    _assert_len(issues, len(ds["labels"]))
    assert "is_label_issue" in issues.columns

    # features -> Label/Outlier/Near Duplicate/Non-IID/Null
    lab.find_issues(features=ds["features"])  # (puede requerir modelo para más robustez)
    issues = lab.get_issues()
    _assert_len(issues, len(ds["labels"]))
    assert any(c in issues.columns for c in ["is_outlier_issue", "is_near_duplicate_issue", "is_non_iid_issue", "is_label_issue"])

    # features + model (REQUIERE issue_types) (NO IMPLEMENTADO)
    # Ejemplo (no ejecutar):
    # lab.find_issues(
    #   features=ds["features"],
    #   issue_types={"label": {"clean_learning_kwargs": {"model": regression_model}}}
    # )

    # knn_graph -> Outlier/Near Duplicate/Non-IID
    lab.find_issues(knn_graph=ds["knn_graph"])
    issues = lab.get_issues()
    _assert_len(issues, len(ds["labels"]))
    assert any(c in issues.columns for c in ["is_outlier_issue", "is_near_duplicate_issue", "is_non_iid_issue", "is_label_issue"])


# ====================================
# Multilabel Task
# ====================================
@pytest.mark.fast
@pytest.mark.multilabel
def test_multilabel_combinations(small_multilabel):
    ds = small_multilabel
    lab = Datalab(data=ds["data"], label_name=ds["label_name"], task="multilabel")

    # pred_probs -> Label Issue
    lab.find_issues(pred_probs=ds["pred_probs"])
    issues = lab.get_issues()
    _assert_len(issues, len(ds["labels"]))
    assert "is_label_issue" in issues.columns

    # features -> Label/Outlier/Near Duplicate/Non-IID/Null
    lab.find_issues(features=ds["features"])
    issues = lab.get_issues()
    _assert_len(issues, len(ds["labels"]))
    assert any(c in issues.columns for c in ["is_outlier_issue", "is_near_duplicate_issue", "is_non_iid_issue", "is_label_issue"])

    # knn_graph -> Outlier/Near Duplicate/Non-IID
    lab.find_issues(knn_graph=ds["knn_graph"])
    issues = lab.get_issues()
    _assert_len(issues, len(ds["labels"]))
    assert any(c in issues.columns for c in ["is_outlier_issue", "is_near_duplicate_issue", "is_non_iid_issue", "is_label_issue"])
