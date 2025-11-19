"""
Validación completa y mejorada para issue_types de CleanLab DataLab.
Basado en la documentación oficial completa de CleanLab v2.6.0.
Cubre todos los issue types, parámetros y configuraciones documentadas.
"""

from collections.abc import Callable
import logging
from typing import Any, ClassVar

from pydantic.v1 import BaseModel, Field, validator

# Configuración de logging
logger = logging.getLogger("cleanlab.validator")
logger.setLevel(logging.DEBUG)

# =============================================================================
# ESQUEMAS BASE Y UTILITARIOS
# =============================================================================


class BaseIssueSchema(BaseModel):
    """Esquema base con información de referencia y validaciones comunes."""

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True
        validate_assignment = True

    # Información de referencia para debugging y documentación
    _source_module: ClassVar[str] = ""
    _source_class: ClassVar[str] = ""
    _description: ClassVar[str] = ""
    _task_types: ClassVar[list[str]] = ["classification", "regression", "multilabel"]

    def get_reference_info(self) -> dict[str, Any]:
        """Retorna información de referencia completa."""
        return {
            "source_module": self._source_module,
            "source_class": self._source_class,
            "description": self._description,
            "task_types": self._task_types,
        }


class DistanceMetricSchema(BaseModel):
    """Esquema para validar métricas de distancia."""

    metric: str | Callable | None = Field(
        default=None,
        description="Métrica de distancia: 'cosine', 'euclidean', 'manhattan', 'l1', 'l2', o función callable personalizada",
    )

    @validator("metric")
    def validate_metric(cls, v):
        if v is None:
            return v
        if callable(v):
            return v
        valid_metrics = ["cosine", "euclidean", "manhattan", "l1", "l2"]
        if v not in valid_metrics:
            raise ValueError(f"metric debe ser uno de {valid_metrics} o una función callable")
        return v


class KNNParametersSchema(BaseModel):
    """Parámetros comunes para algoritmos basados en KNN."""

    k: int | None = Field(
        default=10, ge=1, le=1000, description="Número de vecinos más cercanos para algoritmos basados en KNN"
    )

    metric: str | Callable | None = Field(default=None, description="Métrica de distancia para KNN")


# =============================================================================
# ESQUEMAS PARA PARÁMETROS ANIDADOS DE CLEANLEARNING
# =============================================================================


class FindLabelIssuesKwargsSchema(BaseModel):
    """Parámetros para find_label_issues en CleanLearning."""

    filter_by: str | None = Field(
        default=None,
        description="Método para filtrar issues de labels: 'prune_by_noise_rate', 'prune_by_class', 'both', 'confident_learning', 'predicted_neq_given'",
    )

    frac_noise: float | None = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fracción de noise a remover (1.0 = remover todos los issues detectados)",
    )

    num_to_remove_per_class: list[int] | None = Field(
        default=None, description="Número específico de ejemplos a remover por clase"
    )

    min_examples_per_class: int | None = Field(
        default=1, ge=1, description="Mínimo número de ejemplos por clase requerido"
    )

    confident_joint: Any | None = Field(default=None, description="Matriz confident joint precomputada")

    n_jobs: int | None = Field(default=None, description="Número de trabajos paralelos")

    verbose: bool | None = Field(default=False, description="Modo verboso")

    @validator("filter_by")
    def validate_filter_by(cls, v):
        valid_methods = [
            "prune_by_noise_rate",
            "prune_by_class",
            "both",
            "confident_learning",
            "predicted_neq_given",
            None,
        ]
        if v not in valid_methods:
            raise ValueError(f"filter_by debe ser uno de: {valid_methods}")
        return v


class LabelQualityScoresKwargsSchema(BaseModel):
    """Parámetros para get_label_quality_scores en CleanLearning."""

    method: str | None = Field(
        default="self_confidence",
        description="Método para calcular calidad de labels: 'self_confidence', 'normalized_margin', 'confidence_weighted_entropy'",
    )

    adjust_pred_probs: bool | None = Field(
        default=True, description="Ajustar probabilidades predichas basado en confident joint"
    )

    weight_ensemble_members: bool | None = Field(
        default=False, description="Ponderar miembros del ensemble por su precisión"
    )

    @validator("method")
    def validate_method(cls, v):
        valid_methods = ["self_confidence", "normalized_margin", "confidence_weighted_entropy"]
        if v not in valid_methods:
            raise ValueError(f"method debe ser uno de: {valid_methods}")
        return v


class CleanLearningKwargsSchema(BaseModel):
    """Parámetros para el constructor de CleanLearning."""

    clf: Any | None = Field(default=None, description="Clasificador que implementa la API de scikit-learn")

    seed: int | None = Field(default=None, description="Semilla para reproducibilidad")

    cv_n_folds: int | None = Field(default=5, ge=2, le=20, description="Número de folds para cross-validation")

    converge_latent_estimates: bool | None = Field(
        default=False, description="Forzar consistencia numérica de estimaciones latentes"
    )

    pulearning: int | None | None = Field(
        default=None, description="None para aprendizaje supervisado normal, 0 o 1 para PU learning"
    )

    find_label_issues_kwargs: FindLabelIssuesKwargsSchema | None = Field(
        default=None, description="Parámetros para find_label_issues"
    )

    label_quality_scores_kwargs: LabelQualityScoresKwargsSchema | None = Field(
        default=None, description="Parámetros para label_quality_scores"
    )

    verbose: bool | None = Field(default=False, description="Modo verboso")

    low_memory: bool | None = Field(default=False, description="Modo de baja memoria para datasets grandes")

    # Parámetros adicionales documentados en Datalab
    thresholds: list[float] | None = Field(default=None, description="Umbrales para find_label_issues")

    noise_matrix: Any | None = Field(default=None, description="Matriz de noise para find_label_issues")

    inverse_noise_matrix: Any | None = Field(
        default=None, description="Matriz de noise inversa para find_label_issues"
    )

    save_space: bool | None = Field(default=False, description="Optimizar uso de memoria en find_label_issues")

    clf_kwargs: dict[str, Any] | None = Field(
        default=None, description="[ADVERTENCIA: Actualmente sin efecto] Parámetros para el clasificador"
    )

    validation_func: Any | None = Field(
        default=None, description="[ADVERTENCIA: Actualmente sin efecto] Función de validación personalizada"
    )

    @validator("cv_n_folds")
    def validate_cv_n_folds(cls, v):
        if v < 2:
            raise ValueError("cv_n_folds debe ser al menos 2")
        return v

    @validator("pulearning")
    def validate_pulearning(cls, v):
        if v not in [None, 0, 1]:
            raise ValueError("pulearning debe ser None, 0, o 1")
        return v

    @validator("find_label_issues_kwargs", "label_quality_scores_kwargs", pre=True)
    def convert_dict_to_schema(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
            if (
                "find_label_issues_kwargs" in cls.__fields__
                and cls.__fields__["find_label_issues_kwargs"].type_ == FindLabelIssuesKwargsSchema
            ):
                return FindLabelIssuesKwargsSchema(**v)
            elif (
                "label_quality_scores_kwargs" in cls.__fields__
                and cls.__fields__["label_quality_scores_kwargs"].type_ == LabelQualityScoresKwargsSchema
            ):
                return LabelQualityScoresKwargsSchema(**v)
        return v


class HealthSummaryParametersSchema(BaseModel):
    """Parámetros para health_summary function."""

    asymmetric: bool | None = Field(default=False, description="Usar matriz de noise asimétrica")

    class_names: list[str] | None = Field(default=None, description="Nombres de clases para display")

    num_examples: int | None = Field(default=None, ge=1, description="Número de ejemplos a mostrar")

    joint: Any | None = Field(default=None, description="Matriz de distribución conjunta")

    confident_joint: Any | None = Field(default=None, description="Matriz confident joint")

    multi_label: bool | None = Field(default=False, description="Indica si los datos son multi-label")

    verbose: bool | None = Field(default=True, description="Output verboso")


# =============================================================================
# ESQUEMAS PARA PARÁMETROS DE OUTLIER DETECTION
# =============================================================================


class OODParamsSchema(BaseModel):
    """Parámetros para OutOfDistribution dentro de outlier detection."""

    adjust_pred_probs: bool | None = Field(
        default=None, description="Ajustar probabilidades predichas para OOD detection"
    )

    method: str | None = Field(default=None, description="Método para OOD: 'entropy', 'least_confidence', 'gen'")

    confident_thresholds: Any | None = Field(default=None, description="Umbrales confidentes para OOD detection")

    @validator("method")
    def validate_method(cls, v):
        if v is not None and v not in ["entropy", "least_confidence", "gen"]:
            raise ValueError("method debe ser 'entropy', 'least_confidence', o 'gen'")
        return v


class OODKwargsSchema(BaseModel):
    """Wrapper para parámetros de OutOfDistribution."""

    params: OODParamsSchema | None = Field(default=None, description="Parámetros para OutOfDistribution")


# =============================================================================
# ESQUEMAS PRINCIPALES PARA ISSUE TYPES
# =============================================================================


class LabelIssueSchema(BaseIssueSchema):
    """
    Valida parámetros para LabelIssueManager.

    Detecta ejemplos cuyas labels pueden ser incorrectas debido a error de anotación.
    Requiere pred_probs de un modelo entrenado o features para calcularlos.
    """

    _source_module = "cleanlab.datalab.internal.issue_manager.label"
    _source_class = "LabelIssueManager"
    _description = "Manages label issues in a Datalab"
    _task_types = ["classification", "multilabel"]

    k: int | None = Field(
        default=10,
        ge=1,
        description="Número de vecinos para calcular pred_probs desde features (solo si features se proveen y pred_probs no)",
    )

    clean_learning_kwargs: CleanLearningKwargsSchema | None = Field(
        default=None, description="Keyword arguments para CleanLearning constructor"
    )

    health_summary_parameters: HealthSummaryParametersSchema | None = Field(
        default=None, description="Keyword arguments para health_summary function"
    )

    # Parámetros extraídos de _process_find_label_issues_kwargs
    thresholds: list[float] | None = Field(
        default=None, description="Umbrales para CleanLearning.find_label_issues()"
    )

    noise_matrix: Any | None = Field(
        default=None, description="Matriz de noise para CleanLearning.find_label_issues()"
    )

    inverse_noise_matrix: Any | None = Field(
        default=None, description="Matriz de noise inversa para CleanLearning.find_label_issues()"
    )

    save_space: bool | None = Field(
        default=None, description="Optimizar memoria en CleanLearning.find_label_issues()"
    )

    clf_kwargs: dict[str, Any] | None = Field(
        default=None, description="[ADVERTENCIA: Sin efecto actual] Parámetros para clasificador"
    )

    validation_func: Any | None = Field(
        default=None, description="[ADVERTENCIA: Sin efecto actual] Función de validación"
    )

    @validator("thresholds")
    def validate_thresholds(cls, v):
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("thresholds debe ser una lista")
            for threshold in v:
                if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                    raise ValueError("Cada threshold debe ser un número entre 0 y 1")
        return v


class RegressionLabelIssueSchema(BaseIssueSchema):
    """
    Valida parámetros para RegressionLabelIssueManager.

    Versión específica para regression tasks.
    """

    _source_module = "cleanlab.datalab.internal.issue_manager.regression.label"
    _source_class = "RegressionLabelIssueManager"
    _description = "Manages label issues in a Datalab for regression tasks"
    _task_types = ["regression"]

    clean_learning_kwargs: CleanLearningKwargsSchema | None = Field(
        default=None, description="Keyword arguments para regression.learn.CleanLearning"
    )

    threshold: float | None = Field(
        default=0.05,
        ge=0.0,
        description="Umbral para determinar label issues (multiplicador de la mediana del score de calidad)",
    )

    health_summary_parameters: HealthSummaryParametersSchema | None = Field(
        default=None, description="Keyword arguments para health_summary"
    )


class OutlierIssueSchema(BaseIssueSchema, KNNParametersSchema):
    """
    Valida parámetros para OutlierIssueManager.

    Detecta ejemplos que son muy diferentes del resto del dataset (potencialmente out-of-distribution).
    Puede basarse en features, knn_graph, o pred_probs.
    """

    _source_module = "cleanlab.datalab.internal.issue_manager.outlier"
    _source_class = "OutlierIssueManager"
    _description = "Manages issues related to out-of-distribution examples"
    _task_types = ["classification", "regression", "multilabel"]

    k: int | None = Field(default=10, ge=1, description="Número de vecinos para búsqueda de nearest neighbors")

    t: int | None = Field(
        default=1, ge=1, description="Entero para modular la transformación de distancias a scores [0,1]"
    )

    metric: str | Callable | None = Field(
        default=None, description="Métrica de distancia para nearest neighbors"
    )

    scaling_factor: float | None = Field(
        default=None, description="Factor para normalizar distancias antes de convertirlas a scores"
    )

    threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Umbral para detección de outliers (0-1, menor = más sensible)"
    )

    ood_kwargs: OODKwargsSchema | None = Field(default=None, description="Keyword arguments para OutOfDistribution")

    @validator("threshold")
    def validate_threshold(cls, v):
        if v is not None and not (0 <= v <= 1):
            raise ValueError("threshold debe estar entre 0 y 1")
        return v

    @validator("ood_kwargs", pre=True)
    def validate_ood_kwargs(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
            return OODKwargsSchema(**v)
        return v


class NearDuplicateIssueSchema(BaseIssueSchema, KNNParametersSchema):
    """
    Valida parámetros para NearDuplicateIssueManager.

    Detecta ejemplos que son duplicados exactos o casi duplicados en el dataset.
    Basado en features o knn_graph.
    """

    _source_module = "cleanlab.datalab.internal.issue_manager.duplicate"
    _source_class = "NearDuplicateIssueManager"
    _description = "Manages issues related to near-duplicate examples"
    _task_types = ["classification", "regression", "multilabel"]

    metric: str | Callable | None = Field(
        default=None, description="Métrica de distancia para detección de duplicados"
    )

    threshold: float | None = Field(
        default=0.13, ge=0.0, description="Umbral máximo de distancia relativa para considerar duplicados"
    )

    k: int | None = Field(
        default=10,
        ge=1,
        description="Número de vecinos para construcción del grafo KNN (no afecta resultados de duplicados)",
    )

    @validator("threshold")
    def validate_threshold(cls, v):
        if v is not None and v < 0:
            raise ValueError("threshold debe ser mayor o igual a 0")
        return v


class NonIIDIssueSchema(BaseIssueSchema, KNNParametersSchema):
    """
    Valida parámetros para NonIIDIssueManager.

    Detecta violaciones estadísticamente significativas de la suposición IID en el dataset.
    Basado en features o knn_graph.
    """

    _source_module = "cleanlab.datalab.internal.issue_manager.noniid"
    _source_class = "NonIIDIssueManager"
    _description = "Manages issues related to non-iid data distributions"
    _task_types = ["classification", "regression", "multilabel"]

    metric: str | Callable | None = Field(
        default=None, description="Métrica de distancia para cómputo del grafo KNN"
    )

    k: int | None = Field(default=10, ge=1, description="Número de vecinos para el grafo KNN")

    num_permutations: int | None = Field(
        default=25, ge=1, description="Número de pruebas de permutación para determinar IID"
    )

    seed: int | None = Field(default=0, description="Semilla para generador de números aleatorios")

    significance_threshold: float | None = Field(
        default=0.05, ge=0.0, le=1.0, description="Umbral de significancia estadística para tests IID"
    )

    @validator("significance_threshold")
    def validate_significance_threshold(cls, v):
        if v is not None and not (0 <= v <= 1):
            raise ValueError("significance_threshold debe estar entre 0 y 1")
        return v


class ClassImbalanceIssueSchema(BaseIssueSchema):
    """
    Valida parámetros para ClassImbalanceIssueManager.

    Detecta desbalance severo entre clases en el dataset.
    Solo requiere las labels proporcionadas.
    """

    _source_module = "cleanlab.datalab.internal.issue_manager.imbalance"
    _source_class = "ClassImbalanceIssueManager"
    _description = "Manages issues related to imbalance class examples"
    _task_types = ["classification"]

    threshold: float | None = Field(
        default=0.1, ge=0.0, le=1.0, description="Fracción mínima de muestras por clase para considerar balance"
    )

    @validator("threshold")
    def validate_threshold(cls, v):
        if v is not None and not (0 <= v <= 1):
            raise ValueError("threshold debe estar entre 0 y 1")
        return v


class UnderperformingGroupIssueSchema(BaseIssueSchema, KNNParametersSchema):
    """
    Valida parámetros para UnderperformingGroupIssueManager.

    Detecta grupos/clusters de ejemplos donde las predicciones del modelo son pobres.
    Basado en pred_probs + features, pred_probs + knn_graph, o pred_probs + cluster_ids.
    """

    _source_module = "cleanlab.datalab.internal.issue_manager.underperforming_group"
    _source_class = "UnderperformingGroupIssueManager"
    _description = "Manages issues related to underperforming group examples"
    _task_types = ["classification"]

    metric: str | Callable | None = Field(default=None, description="Métrica de distancia para clustering")

    threshold: float | None = Field(
        default=0.1, ge=0.0, le=1.0, description="Umbral para determinar grupos con bajo rendimiento"
    )

    k: int | None = Field(default=10, ge=1, description="Número de vecinos para construcción del grafo KNN")

    clustering_kwargs: dict[str, Any] | None = Field(
        default_factory=dict, description="Argumentos para el algoritmo de clustering (e.g., DBSCAN)"
    )

    min_cluster_samples: int | None = Field(
        default=5, ge=1, description="Mínimo número de ejemplos por cluster para ser considerado"
    )

    # Parámetro especial para find_issues (no constructor)
    cluster_ids: list[int] | None = Field(
        default=None, description="IDs de cluster precomputados para cada ejemplo (avanzaado)"
    )

    @validator("threshold")
    def validate_threshold(cls, v):
        if v is not None and not (0 <= v <= 1):
            raise ValueError("threshold debe estar entre 0 y 1")
        return v


class DataValuationIssueSchema(BaseIssueSchema, KNNParametersSchema):
    """
    Valida parámetros para DataValuationIssueManager.

    Detecta ejemplos con menor valor de datos (Data Shapley) que contribuyen menos al modelo.
    Basado en features o knn_graph.
    """

    _source_module = "cleanlab.datalab.internal.issue_manager.data_valuation"
    _source_class = "DataValuationIssueManager"
    _description = "Detect which examples in a dataset are least valuable via an approximate Data Shapely value"
    _task_types = ["classification", "regression", "multilabel"]

    metric: str | Callable | None = Field(default=None, description="Métrica de distancia para data valuation")

    threshold: float | None = Field(
        default=0.5, ge=0.0, description="Umbral para ejemplos con baja data valuation score"
    )

    k: int | None = Field(default=10, ge=1, description="Número de vecinos para cálculo de KNN-Shapley")

    @validator("threshold")
    def validate_threshold(cls, v):
        if v is not None and v < 0:
            raise ValueError("threshold debe ser mayor o igual a 0")
        return v


class NullIssueSchema(BaseIssueSchema):
    """
    Valida parámetros para NullIssueManager.

    Detecta ejemplos con valores null/missing across all feature columns.
    Solo requiere features.
    """

    _source_module = "cleanlab.datalab.internal.issue_manager.null"
    _source_class = "NullIssueManager"
    _description = "Manages issues related to null/missing values in the rows of features"
    _task_types = ["classification", "regression", "multilabel"]

    # No tiene parámetros configurables según documentación


class IdentifierColumnIssueSchema(BaseIssueSchema):
    """
    Valida parámetros para IdentifierColumnIssueManager.

    Detecta columnas numéricas secuenciales que probablemente sean identificadores.
    Solo requiere features.
    """

    _source_module = "cleanlab.datalab.internal.issue_manager.identifier_column"
    _source_class = "IdentifierColumnIssueManager"
    _description = "Flags sequential numerical columns in the features of a dataset"
    _task_types = ["classification", "regression", "multilabel"]

    # No tiene parámetros configurables según documentación


class MultilabelIssueSchema(BaseIssueSchema):
    """
    Valida parámetros para MultilabelIssueManager.

    Versión específica para multilabel classification tasks.
    """

    _source_module = "cleanlab.datalab.internal.issue_manager.multilabel.label"
    _source_class = "MultilabelIssueManager"
    _description = "Manages label issues in Datalab for multilabel tasks"
    _task_types = ["multilabel"]

    # No tiene parámetros específicos en constructor según documentación


# =============================================================================
# ESQUEMAS PARA IMAGE ISSUES Y SPURIOUS CORRELATIONS
# =============================================================================


class ImageIssueBaseSchema(BaseIssueSchema):
    """Esquema base para issues de imágenes."""

    _source_module = "cleanvision"
    _description = "Image-specific issue detection"
    _task_types = ["classification"]  # Principalmente para classification con imágenes

    threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Umbral para detección del issue (0-1, menor = menos sensibilidad)"
    )


class DarkImageIssueSchema(ImageIssueBaseSchema):
    """Detecta imágenes excesivamente oscuras."""

    _source_class = "DarkIssueManager"


class LightImageIssueSchema(ImageIssueBaseSchema):
    """Detecta imágenes excesivamente brillantes."""

    _source_class = "LightIssueManager"


class BlurryImageIssueSchema(ImageIssueBaseSchema):
    """Detecta imágenes borrosas."""

    _source_class = "BlurryIssueManager"


class LowInformationImageIssueSchema(ImageIssueBaseSchema):
    """Detecta imágenes con baja información/entropía."""

    _source_class = "LowInformationIssueManager"


class OddAspectRatioImageIssueSchema(ImageIssueBaseSchema):
    """Detecta imágenes con relaciones de aspecto inusuales."""

    _source_class = "OddAspectRatioIssueManager"


class OddSizeImageIssueSchema(ImageIssueBaseSchema):
    """Detecta imágenes con tamaños inusuales."""

    _source_class = "OddSizeIssueManager"

    threshold: float | None = Field(
        default=10.0, ge=0.0, description="Umbral para tamaño inusual (mayor = menos sensibilidad)"
    )


class SpuriousCorrelationsIssueSchema(BaseIssueSchema):
    """
    Valida parámetros para detección de correlaciones espurias.

    Detecta correlaciones entre propiedades de imagen y labels que pueden ser explotadas por modelos.
    """

    _source_module = "cleanlab.datalab.internal.issue_manager.spurious_correlations"
    _source_class = "SpuriousCorrelationsIssueManager"
    _description = "Manages spurious correlations between image properties and labels"
    _task_types = ["classification"]  # Para datasets de imágenes

    threshold: float | None = Field(
        default=0.3, ge=0.0, le=1.0, description="Umbral para correlaciones espurias (menor = más estricto)"
    )


# =============================================================================
# REGISTRO CENTRAL COMPLETO Y MEJORADO
# =============================================================================

SCHEMA_REGISTRY = {
    "classification": {
        "label": LabelIssueSchema,
        "outlier": OutlierIssueSchema,
        "near_duplicate": NearDuplicateIssueSchema,
        "non_iid": NonIIDIssueSchema,
        "class_imbalance": ClassImbalanceIssueSchema,
        "underperforming_group": UnderperformingGroupIssueSchema,
        "null": NullIssueSchema,
        "data_valuation": DataValuationIssueSchema,
        "identifier_column": IdentifierColumnIssueSchema,
        # Image issues
        "dark": DarkImageIssueSchema,
        "light": LightImageIssueSchema,
        "blurry": BlurryImageIssueSchema,
        "low_information": LowInformationImageIssueSchema,
        "odd_aspect_ratio": OddAspectRatioImageIssueSchema,
        "odd_size": OddSizeImageIssueSchema,
        # Spurious correlations
        "spurious_correlations": SpuriousCorrelationsIssueSchema,
    },
    "regression": {
        "label": RegressionLabelIssueSchema,
        "outlier": OutlierIssueSchema,
        "near_duplicate": NearDuplicateIssueSchema,
        "non_iid": NonIIDIssueSchema,
        "null": NullIssueSchema,
        "data_valuation": DataValuationIssueSchema,
        "identifier_column": IdentifierColumnIssueSchema,
    },
    "multilabel": {
        "label": MultilabelIssueSchema,
        "outlier": OutlierIssueSchema,
        "near_duplicate": NearDuplicateIssueSchema,
        "non_iid": NonIIDIssueSchema,
        "null": NullIssueSchema,
        "data_valuation": DataValuationIssueSchema,
        "identifier_column": IdentifierColumnIssueSchema,
    },
}


# =============================================================================
# VALIDADOR PRINCIPAL MEJORADO
# =============================================================================

class EnhancedIssueTypesValidator:
    """
    Validador mejorado para issue_types de CleanLab DataLab.

    Características:
    - Validación completa basada en documentación oficial
    - Soporte para todas las tareas: classification, regression, multilabel
    - Validación anidada de parámetros complejos
    - Información de referencia detallada
    - Mensajes de error descriptivos
    """

    def __init__(self, task: str = "classification"):
        self.task = task
        self.schemas = SCHEMA_REGISTRY.get(task, {})

        if task not in SCHEMA_REGISTRY:
            logger.error(f"Task '{task}' no válida. Opciones: {list(SCHEMA_REGISTRY.keys())}")
            raise ValueError(f"Task '{task}' no válida. Opciones: {list(SCHEMA_REGISTRY.keys())}")
        logger.info(f"Inicializado validador para task '{task}' con {len(self.schemas)} issue types disponibles")

    def validate_issue_type(self, issue_type: str, config: dict[str, Any]) -> dict[str, Any]:
        """
        Valida un solo issue type con su configuración.

        Parameters
        ----------
        issue_type : str
            Nombre del issue type a validar
        config : Dict[str, Any]
            Configuración de parámetros para el issue type

        Returns
        -------
        Dict con resultados de validación
        """
        logger.debug(f"Validando issue type '{issue_type}' con config: {config}")
        result = {"is_valid": True, "validated_config": {}, "errors": [], "warnings": [], "reference_info": None}

        # Verificar que el issue type existe para esta task
        if issue_type not in self.schemas:
            logger.warning(f"Issue type '{issue_type}' no válido para task '{self.task}'")
            result["is_valid"] = False
            result["errors"].append(
                f"Issue type '{issue_type}' no válido para task '{self.task}'. Válidos: {list(self.schemas.keys())}"
            )
            return result

        schema_class = self.schemas[issue_type]
        if schema_class is None:
            logger.warning(f"Issue type '{issue_type}' no implementado para task '{self.task}'")
            result["is_valid"] = False
            result["errors"].append(f"Issue type '{issue_type}' no implementado para task '{self.task}'")
            return result

        # Validar configuración con Pydantic
        try:
            # Pre-procesar config para convertir dicts anidados a esquemas
            processed_config = self._preprocess_config(config, schema_class)

            instance = schema_class(**processed_config)
            result["validated_config"] = instance.dict(exclude_unset=True, exclude_none=True, by_alias=False)
            result["reference_info"] = instance.get_reference_info()

            # Advertencias para parámetros con efectos limitados
            warnings = self._check_parameter_warnings(issue_type, config)
            result["warnings"].extend(warnings)
            logger.info(f"Issue type '{issue_type}' validado exitosamente")

        except Exception as e:
            logger.error(f"Error validando '{issue_type}': {e!s}")
            result["is_valid"] = False
            error_msg = f"Error validando '{issue_type}': {e!s}"
            if hasattr(e, "errors"):
                for err in e.errors():
                    error_msg += f"\n  - {err['loc']}: {err['msg']}"
            result["errors"].append(error_msg)

        return result

    def _preprocess_config(self, config: dict[str, Any], schema_class: type) -> dict[str, Any]:
        """Pre-procesa la configuración para convertir dicts anidados a esquemas."""
        if not isinstance(config, dict):
            return config

        processed = config.copy()
        schema_fields = schema_class.__fields__

        for field_name, field_info in schema_fields.items():
            if field_name in processed and processed[field_name] is not None:
                field_type = field_info.type_

                # Convertir dicts anidados a sus esquemas correspondientes
                if (
                    isinstance(processed[field_name], dict)
                    and hasattr(field_type, "__origin__")
                    and issubclass(field_type.__origin__, BaseModel)
                ):
                    try:
                        processed[field_name] = field_type(**processed[field_name])
                    except Exception as e:
                        logger.warning(f"Error procesando '{field_name}': {e!s}")
                        raise ValueError(f"Error procesando '{field_name}': {e!s}")  # noqa: B904

        return processed

    def _check_parameter_warnings(self, issue_type: str, config: dict[str, Any]) -> list[str]:
        """Genera advertencias para parámetros con efectos limitados o problemas conocidos."""
        warnings = []

        if issue_type == "label":
            if config.get("clf_kwargs"):
                warnings.append("clf_kwargs: Documentación indica 'Currently has no effect'")
            if config.get("validation_func"):
                warnings.append("validation_func: Documentación indica 'Currently has no effect'")

        return warnings

    def validate(self, issue_types: dict[str, Any]) -> dict[str, Any]:
        """
        Valida todos los issue types en el diccionario.

        Parameters
        ----------
        issue_types : Dict[str, Any]
            Diccionario con issue types y sus configuraciones

        Returns
        -------
        Dict con resultados completos de validación
        """
        logger.info(f"Iniciando validación de {len(issue_types)} issue types para task '{self.task}'")
        result = {
            "is_valid": True,
            "validated_config": {},
            "errors": [],
            "warnings": [],
            "reference_info": {},
            "summary": {"total_issues": 0, "valid_issues": 0, "invalid_issues": 0},
        }

        if not isinstance(issue_types, dict):
            logger.error("issue_types debe ser un diccionario")
            result["is_valid"] = False
            result["errors"].append("issue_types debe ser un diccionario")
            return result

        result["summary"]["total_issues"] = len(issue_types)

        for issue_type, config in issue_types.items():
            if config is None:
                config = {}

            validation_result = self.validate_issue_type(issue_type, config)

            if validation_result["is_valid"]:
                result["validated_config"][issue_type] = validation_result["validated_config"]
                result["reference_info"][issue_type] = validation_result["reference_info"]
                result["warnings"].extend(validation_result["warnings"])
                result["summary"]["valid_issues"] += 1
            else:
                result["is_valid"] = False
                result["errors"].extend(validation_result["errors"])
                result["warnings"].extend(validation_result["warnings"])
                result["summary"]["invalid_issues"] += 1

        logger.info(f"Validación completada: {result['summary']['valid_issues']} válidos, {result['summary']['invalid_issues']} inválidos")
        return result

    def get_available_issue_types(self) -> list[str]:
        """Retorna lista de issue types disponibles para la task actual."""
        logger.debug(f"Obteniendo issue types disponibles para task '{self.task}': {list(self.schemas.keys())}")
        return list(self.schemas.keys())

    def get_issue_type_info(self, issue_type: str) -> dict[str, Any]:
        """Retorna información detallada sobre un issue type específico."""
        logger.debug(f"Obteniendo información para issue type '{issue_type}'")
        if issue_type not in self.schemas:
            logger.warning(f"Issue type '{issue_type}' no disponible para task '{self.task}'")
            return {"error": f"Issue type '{issue_type}' no disponible para task '{self.task}'"}

        schema_class = self.schemas[issue_type]
        if schema_class is None:
            logger.warning(f"Schema no implementado para '{issue_type}'")
            return {"error": f"Schema no implementado para '{issue_type}'"}

        # Crear instancia vacía para obtener información
        instance = schema_class()
        info = instance.get_reference_info()

        # Agregar información de campos
        info["fields"] = {}
        for field_name, field_info in schema_class.__fields__.items():
            info["fields"][field_name] = {
                "type": str(field_info.type_),
                "default": field_info.default,
                "description": field_info.field_info.description,
                "required": field_info.required,
            }

        return info

# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================


def validate_issue_types_config(issue_types: dict[str, Any], task: str = "classification") -> dict[str, Any]:
    """
    Función de utilidad para validación rápida de configuración de issue_types.

    Parameters
    ----------
    issue_types : Dict[str, Any]
        Diccionario de configuración de issue_types
    task : str
        Tipo de task: 'classification', 'regression', o 'multilabel'

    Returns
    -------
    Dict con resultados de validación
    """
    validator = EnhancedIssueTypesValidator(task=task)
    return validator.validate(issue_types)


def get_task_requirements(task: str) -> dict[str, Any]:
    """
    Retorna requisitos y issue types disponibles por task.

    Parameters
    ----------
    task : str
        Tipo de task

    Returns
    -------
    Dict con información de requisitos
    """
    validator = EnhancedIssueTypesValidator(task=task)

    requirements = {
        "task": task,
        "available_issue_types": validator.get_available_issue_types(),
        "required_inputs": {
            "label_name": "Field con labels anotadas (classification)",
            "pred_probs": "Probabilidades predichas out-of-sample",
            "features": "Representaciones vectoriales de features",
            "knn_graph": "Grafo K nearest neighbors precomputado",
        },
    }

    return requirements


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de configuración para validar
    sample_config = {
        "label": {
            # "k": 15,
            # "clean_learning_kwargs": {
            #     "cv_n_folds": 3,
            #     "find_label_issues_kwargs": {"filter_by": "prune_by_noise_rate", "frac_noise": 0.8},
            # },
        },
        "outlier": {"k": 20, "threshold": 0.2, "ood_kwargs": {"params": {"method": "entropy"}}},
        "near_duplicate": {"threshold": 0.1, "metric": "cosine"},
    }

    # Validar la configuración
    result = validate_issue_types_config(sample_config, task="classification")

    print("Validación completada:")
    print(f"Válido: {result['is_valid']}")
    print(f"Errores: {result['errors']}")
    print(f"Advertencias: {result['warnings']}")
    print(f"Configuración validada: {result['validated_config'].keys()}")
