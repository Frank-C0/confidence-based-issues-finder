"""
Validación simplificada para issue_types de CleanLab DataLab.
"""

from typing import Any

from .cleanlab_issue_types_validator import TASK_SCHEMA_REGISTRY


class EnhancedIssueTypesValidator:
    """
    Validador simplificado para issue_types de CleanLab DataLab.
    """

    def __init__(self, task: str = "classification"):
        self.task = task
        self.schema = TASK_SCHEMA_REGISTRY.get(task)

        if task not in TASK_SCHEMA_REGISTRY:
            raise ValueError(f"Task '{task}' no válida. Opciones: {list(TASK_SCHEMA_REGISTRY.keys())}")

    def validate(self, issue_types: dict[str, Any]) -> dict[str, Any]:
        """
        Valida la configuración de issue_types.

        Returns
        -------
        Dict con 'is_valid' (bool) y 'errors' (list[str])
        """
        result = {"is_valid": True, "errors": []}

        if not isinstance(issue_types, dict):
            result["is_valid"] = False
            result["errors"].append("issue_types debe ser un diccionario")
            return result

        try:
            self.schema(**issue_types)
        except Exception as e:
            result["is_valid"] = False
            if hasattr(e, "errors"):
                result["errors"] = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            else:
                result["errors"] = [str(e)]

        return result


def validate_issue_types_config(issue_types: dict[str, Any], task: str = "classification") -> dict[str, Any]:
    """
    Función de utilidad para validación rápida de configuración de issue_types.

    Returns
    -------
    Dict con 'is_valid' (bool) y 'errors' (list[str])
    """
    validator = EnhancedIssueTypesValidator(task=task)
    return validator.validate(issue_types)
