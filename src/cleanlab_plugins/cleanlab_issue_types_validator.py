from collections.abc import Callable
import logging
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Configuración de logging
logger = logging.getLogger("cleanlab.validator")
logger.setLevel(logging.DEBUG)

# =============================================================================
# ESQUEMAS BASE Y UTILITARIOS
# =============================================================================


class BaseIssueSchema(BaseModel):
    """Esquema base con información de referencia y validaciones comunes."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

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

    metric: Literal["cosine", "euclidean", "manhattan", "l1", "l2"] | Callable | None = Field(
        default=None,
        description="Métrica de distancia: 'cosine', 'euclidean', 'manhattan', 'l1', 'l2', o función callable personalizada",
    )


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

    filter_by: (
        Literal["prune_by_noise_rate", "prune_by_class", "both", "confident_learning", "predicted_neq_given"] | None
    ) = Field(
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


class LabelQualityScoresKwargsSchema(BaseModel):
    """Parámetros para get_label_quality_scores en CleanLearning."""

    method: Literal["self_confidence", "normalized_margin", "confidence_weighted_entropy"] | None = Field(
        default="self_confidence",
        description="Método para calcular calidad de labels: 'self_confidence', 'normalized_margin', 'confidence_weighted_entropy'",
    )

    adjust_pred_probs: bool | None = Field(
        default=True, description="Ajustar probabilidades predichas basado en confident joint"
    )

    weight_ensemble_members: bool | None = Field(
        default=False, description="Ponderar miembros del ensemble por su precisión"
    )


class CleanLearningKwargsSchema(BaseModel):
    """Parámetros para el constructor de CleanLearning."""

    clf: Any | None = Field(default=None, description="Clasificador que implementa la API de scikit-learn")

    seed: int | None = Field(default=None, description="Semilla para reproducibilidad")

    cv_n_folds: int | None = Field(default=5, ge=2, le=20, description="Número de folds para cross-validation")

    converge_latent_estimates: bool | None = Field(
        default=False, description="Forzar consistencia numérica de estimaciones latentes"
    )

    pulearning: Literal[0, 1] | None = Field(
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

    inverse_noise_matrix: Any | None = Field(default=None, description="Matriz de noise inversa para find_label_issues")

    save_space: bool | None = Field(default=False, description="Optimizar uso de memoria en find_label_issues")

    clf_kwargs: dict[str, Any] | None = Field(
        default=None, description="[ADVERTENCIA: Actualmente sin efecto] Parámetros para el clasificador"
    )

    validation_func: Any | None = Field(
        default=None, description="[ADVERTENCIA: Actualmente sin efecto] Función de validación personalizada"
    )

    @field_validator("find_label_issues_kwargs", mode="before")
    @classmethod
    def convert_find_label_issues_kwargs(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
            return FindLabelIssuesKwargsSchema(**v)
        return v

    @field_validator("label_quality_scores_kwargs", mode="before")
    @classmethod
    def convert_label_quality_scores_kwargs(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
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

    method: Literal["entropy", "least_confidence", "gen"] | None = Field(
        default=None, description="Método para OOD: 'entropy', 'least_confidence', 'gen'"
    )

    confident_thresholds: Any | None = Field(default=None, description="Umbrales confidentes para OOD detection")


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
    thresholds: Annotated[list[Annotated[float, Field(ge=0, le=1)]], Field(default=None)] = Field(
        default=None, description="Umbrales para CleanLearning.find_label_issues()"
    )

    noise_matrix: Any | None = Field(default=None, description="Matriz de noise para CleanLearning.find_label_issues()")

    inverse_noise_matrix: Any | None = Field(
        default=None, description="Matriz de noise inversa para CleanLearning.find_label_issues()"
    )

    save_space: bool | None = Field(default=None, description="Optimizar memoria en CleanLearning.find_label_issues()")

    clf_kwargs: dict[str, Any] | None = Field(
        default=None, description="[ADVERTENCIA: Sin efecto actual] Parámetros para clasificador"
    )

    validation_func: Any | None = Field(
        default=None, description="[ADVERTENCIA: Sin efecto actual] Función de validación"
    )


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

    metric: str | Callable | None = Field(default=None, description="Métrica de distancia para nearest neighbors")

    scaling_factor: float | None = Field(
        default=None, description="Factor para normalizar distancias antes de convertirlas a scores"
    )

    threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Umbral para detección de outliers (0-1, menor = más sensible)"
    )

    ood_kwargs: OODKwargsSchema | None = Field(default=None, description="Keyword arguments para OutOfDistribution")

    @field_validator("ood_kwargs", mode="before")
    @classmethod
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

    metric: str | Callable | None = Field(default=None, description="Métrica de distancia para detección de duplicados")

    threshold: float | None = Field(
        default=0.13, ge=0.0, description="Umbral máximo de distancia relativa para considerar duplicados"
    )

    k: int | None = Field(
        default=10,
        ge=1,
        description="Número de vecinos para construcción del grafo KNN (no afecta resultados de duplicados)",
    )


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

    metric: str | Callable | None = Field(default=None, description="Métrica de distancia para cómputo del grafo KNN")

    k: int | None = Field(default=10, ge=1, description="Número de vecinos para el grafo KNN")

    num_permutations: int | None = Field(
        default=25, ge=1, description="Número de pruebas de permutación para determinar IID"
    )

    seed: int | None = Field(default=0, description="Semilla para generador de números aleatorios")

    significance_threshold: float | None = Field(
        default=0.05, ge=0.0, le=1.0, description="Umbral de significancia estadística para tests IID"
    )


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


class ImageIssueTypesSchema(BaseModel):
    """Esquema que agrupa todos los tipos de issues de imágenes para validación."""

    dark: DarkImageIssueSchema | None = Field(default=None, description="Parámetros para detección de imágenes oscuras")
    light: LightImageIssueSchema | None = Field(
        default=None, description="Parámetros para detección de imágenes brillantes"
    )
    blurry: BlurryImageIssueSchema | None = Field(
        default=None, description="Parámetros para detección de imágenes borrosas"
    )
    low_information: LowInformationImageIssueSchema | None = Field(
        default=None, description="Parámetros para detección de imágenes con baja información"
    )
    odd_aspect_ratio: OddAspectRatioImageIssueSchema | None = Field(
        default=None, description="Parámetros para detección de imágenes con aspecto inusual"
    )
    odd_size: OddSizeImageIssueSchema | None = Field(
        default=None, description="Parámetros para detección de imágenes con tamaño inusual"
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


# Esquemas agrupados por tarea para validación simplificada
class ClassificationConfigSchema(BaseModel):
    """Esquema que agrupa todos los issue types para classification."""

    label: LabelIssueSchema | None = Field(default=None, description="Configuración para label issues")
    outlier: OutlierIssueSchema | None = Field(default=None, description="Configuración para outlier issues")
    near_duplicate: NearDuplicateIssueSchema | None = Field(
        default=None, description="Configuración para near duplicate issues"
    )
    non_iid: NonIIDIssueSchema | None = Field(default=None, description="Configuración para non-IID issues")
    class_imbalance: ClassImbalanceIssueSchema | None = Field(
        default=None, description="Configuración para class imbalance issues"
    )
    underperforming_group: UnderperformingGroupIssueSchema | None = Field(
        default=None, description="Configuración para underperforming group issues"
    )
    null: NullIssueSchema | None = Field(default=None, description="Configuración para null issues")
    data_valuation: DataValuationIssueSchema | None = Field(
        default=None, description="Configuración para data valuation issues"
    )
    identifier_column: IdentifierColumnIssueSchema | None = Field(
        default=None, description="Configuración para identifier column issues"
    )
    image_issue_types: ImageIssueTypesSchema | None = Field(default=None, description="Configuración para image issues")
    spurious_correlations: SpuriousCorrelationsIssueSchema | None = Field(
        default=None, description="Configuración para spurious correlations"
    )

    model_config = ConfigDict(extra="forbid")


class RegressionConfigSchema(BaseModel):
    """Esquema que agrupa todos los issue types para regression."""

    label: RegressionLabelIssueSchema | None = Field(default=None, description="Configuración para label issues")
    outlier: OutlierIssueSchema | None = Field(default=None, description="Configuración para outlier issues")
    near_duplicate: NearDuplicateIssueSchema | None = Field(
        default=None, description="Configuración para near duplicate issues"
    )
    non_iid: NonIIDIssueSchema | None = Field(default=None, description="Configuración para non-IID issues")
    null: NullIssueSchema | None = Field(default=None, description="Configuración para null issues")
    data_valuation: DataValuationIssueSchema | None = Field(
        default=None, description="Configuración para data valuation issues"
    )
    identifier_column: IdentifierColumnIssueSchema | None = Field(
        default=None, description="Configuración para identifier column issues"
    )

    model_config = ConfigDict(extra="forbid")


class MultilabelConfigSchema(BaseModel):
    """Esquema que agrupa todos los issue types para multilabel."""

    label: MultilabelIssueSchema | None = Field(default=None, description="Configuración para label issues")
    outlier: OutlierIssueSchema | None = Field(default=None, description="Configuración para outlier issues")
    near_duplicate: NearDuplicateIssueSchema | None = Field(
        default=None, description="Configuración para near duplicate issues"
    )
    non_iid: NonIIDIssueSchema | None = Field(default=None, description="Configuración para non-IID issues")
    null: NullIssueSchema | None = Field(default=None, description="Configuración para null issues")
    data_valuation: DataValuationIssueSchema | None = Field(
        default=None, description="Configuración para data valuation issues"
    )
    identifier_column: IdentifierColumnIssueSchema | None = Field(
        default=None, description="Configuración para identifier column issues"
    )

    model_config = ConfigDict(extra="forbid")


TASK_SCHEMA_REGISTRY = {
    "classification": ClassificationConfigSchema,
    "regression": RegressionConfigSchema,
    "multilabel": MultilabelConfigSchema,
}
