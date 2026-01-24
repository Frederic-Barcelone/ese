from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Generic

# Python 3.9-friendly TypeAlias
from typing_extensions import TypeAlias

from A_core.A01_domain_models import (
    GeneratorType,
    Candidate,
    ExtractedEntity,
)
from A_core.A14_extraction_result import EntityType, ExtractionResult

# Flexible doc model
DocumentModel: TypeAlias = Any


# -----------------------------------------------------------------------------
# Execution Context for Deterministic Orchestration
# -----------------------------------------------------------------------------


@dataclass
class ExecutionContext:
    """
    Shared context passed through all extraction steps.
    Strategies can READ prior outputs but NEVER gate execution.

    INVARIANT: All steps run unconditionally; this context is for data sharing only.
    """

    plan_id: str
    doc_id: str
    doc_fingerprint: str
    outputs_by_step: Dict[str, List["RawExtraction"]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.outputs_by_step is None:
            self.outputs_by_step = {}

    def get_prior_outputs(self, strategy_id: str) -> List["RawExtraction"]:
        """Get outputs from a prior step. Returns empty list if none."""
        return self.outputs_by_step.get(strategy_id, [])


# -----------------------------------------------------------------------------
# RawExtraction: Strategy Output with Features (NOT final confidence)
# -----------------------------------------------------------------------------


@dataclass
class RawExtraction:
    """
    Raw extraction output from strategies.
    Contains features but NOT final confidence.

    INVARIANT: Confidence is computed ONLY by UnifiedConfidenceCalculator.
    Strategies MUST NOT set confidence directly; they emit features instead.
    """

    # Required fields
    doc_id: str
    entity_type: EntityType
    field_name: str  # e.g., "disease", "drug_name", "abbreviation"
    value: str  # Primary extracted value
    page_num: int  # 1-based page number
    strategy_id: str  # Which strategy produced this

    # Optional fields
    normalized_value: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    node_ids: Tuple[str, ...] = ()  # Immutable tuple of block_id, table_id, etc.
    char_span: Optional[Tuple[int, int]] = None  # (start, end) within node
    strategy_version: str = "1.0.0"
    doc_fingerprint: str = ""
    lexicon_source: Optional[str] = None

    # Evidence
    evidence_text: str = ""
    supporting_evidence: Tuple[str, ...] = ()  # Immutable

    # Standard IDs
    standard_ids: Tuple[Tuple[str, str], ...] = ()  # e.g., (("ORPHA", "182090"),)
    extensions: Tuple[Tuple[str, Any], ...] = ()  # Domain-specific extras

    # -------------------------------------------------------------------------
    # FEATURES for confidence calculation (NOT final confidence)
    # These are used by UnifiedConfidenceCalculator to compute the final score.
    # -------------------------------------------------------------------------
    section_name: Optional[str] = None  # Which section was this found in
    from_table: bool = False  # Was this extracted from a table (higher quality)
    lexicon_matched: bool = False  # Did this match a curated lexicon
    externally_validated: bool = False  # PubTator, UMLS, etc.
    pattern_strength: float = 0.0  # How strong was the pattern match (0.0-1.0)
    negated: bool = False  # Was this in a negation context


# -----------------------------------------------------------------------------
# BaseExtractor: Universal Interface for All Extraction Strategies
# -----------------------------------------------------------------------------


class BaseExtractor(ABC):
    """
    Universal interface for all extraction strategies.

    INVARIANT: Extractors consume DocumentGraph only (via doc_graph parameter).
    INVARIANT: Extractors emit RawExtraction (features), NOT final confidence.
    INVARIANT: Extractors never skip execution; they may return empty list.

    Example:
        >>> class DiseaseExtractor(BaseExtractor):
        ...     @property
        ...     def strategy_id(self) -> str:
        ...         return "disease_lexicon_orphanet"
        ...
        ...     @property
        ...     def strategy_version(self) -> str:
        ...         return "1.0.0"
        ...
        ...     @property
        ...     def entity_type(self) -> EntityType:
        ...         return EntityType.DISEASE
        ...
        ...     def extract(self, doc_graph, ctx, config) -> List[RawExtraction]:
        ...         # Extraction logic here
        ...         return []
    """

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """
        Unique identifier for this strategy.
        Used in provenance and merge conflict resolution.
        Example: "disease_lexicon_orphanet", "abbreviation_syntax"
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def strategy_version(self) -> str:
        """
        Version string for reproducibility.
        Should be updated when strategy logic changes.
        Example: "1.0.0", "2.1.3"
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def entity_type(self) -> EntityType:
        """What type of entity this extractor produces."""
        raise NotImplementedError

    @abstractmethod
    def extract(
        self,
        doc_graph: DocumentModel,
        ctx: ExecutionContext,
        config: Dict[str, Any],
    ) -> List[RawExtraction]:
        """
        Extract entities from DocumentGraph.

        Args:
            doc_graph: Parsed document structure with stable node IDs.
                       This is a DocumentGraph from B_parsing/B02_doc_graph.py.
            ctx: Shared execution context. Use ctx.get_prior_outputs()
                 to read outputs from earlier steps. NEVER gate execution
                 based on ctx contents; always run and return results
                 (possibly empty list).
            config: Step-specific configuration from the extraction plan.

        Returns:
            List of RawExtraction with features (NOT final confidence).
            Return empty list if no entities found.

        IMPORTANT:
            - Do NOT set confidence directly; emit features instead.
            - Always return a list (possibly empty), never raise to skip.
            - Use doc_graph.iter_linear_blocks() or doc_graph.iter_tables()
              for iteration; never access raw text outside of DocGraph.
        """
        raise NotImplementedError

# Type variables for generic enricher
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseParser(ABC):
    """
    Interface for ingestion/parsing layer (PDF -> DocumentModel).
    """

    @abstractmethod
    def parse(self, file_path: str) -> DocumentModel:
        raise NotImplementedError


class BaseCandidateGenerator(ABC):
    """
    Interface for any strategy that generates candidates (high recall).
    """

    @property
    @abstractmethod
    def generator_type(self) -> GeneratorType:
        raise NotImplementedError

    @abstractmethod
    def extract(self, doc_structure: DocumentModel) -> List[Candidate]:
        raise NotImplementedError


class BaseVerifier(ABC):
    """
    Interface for the Judge (LLM or deterministic verifier).
    """

    @abstractmethod
    def verify(self, candidate: Candidate) -> ExtractedEntity:
        raise NotImplementedError


class BaseNormalizer(ABC):
    """
    Interface for post-verification standardization.
    """

    @abstractmethod
    def normalize(self, entity: ExtractedEntity) -> ExtractedEntity:
        raise NotImplementedError


class BaseEvaluationMetric(ABC):
    """
    Contract for metrics.
    """

    @abstractmethod
    def compute(
        self,
        predictions: List[ExtractedEntity],
        ground_truth: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        raise NotImplementedError


class BaseEnricher(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for all enrichers in the E_normalization module.

    Provides a common interface and shared functionality for enrichers that
    augment extracted entities with additional information from external
    sources (APIs, databases, models).

    Type Parameters:
        InputT: Input entity type (e.g., ExtractedDisease, str)
        OutputT: Output type (e.g., ExtractedDisease, NCTTrialInfo)

    Attributes:
        config: Configuration dictionary for the enricher.
        enabled: Whether the enricher is active.

    Example:
        >>> class MyEnricher(BaseEnricher[ExtractedEntity, ExtractedEntity]):
        ...     @property
        ...     def enricher_name(self) -> str:
        ...         return "my_enricher"
        ...
        ...     def enrich(self, entity: ExtractedEntity) -> ExtractedEntity:
        ...         # Add enrichment logic
        ...         return entity
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the enricher with configuration.

        Args:
            config: Configuration dictionary. Common keys include:
                - enabled: Enable/disable the enricher (default: True)
                - run_id: Pipeline run identifier
                - confidence_threshold: Minimum confidence for enrichment
        """
        self.config: Dict[str, Any] = config or {}
        self.enabled: bool = self.config.get("enabled", True)
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate the enricher configuration.

        Override in subclasses to add specific validation.
        Raise ConfigurationError if validation fails.

        Default implementation does nothing.
        """
        pass

    @property
    @abstractmethod
    def enricher_name(self) -> str:
        """
        Return the unique identifier for this enricher.

        Used for logging, metrics, and error messages.

        Returns:
            Enricher name (e.g., "pubtator", "nct", "genetic").
        """
        raise NotImplementedError

    @abstractmethod
    def enrich(self, entity: InputT) -> OutputT:
        """
        Enrich a single entity.

        Args:
            entity: Entity to enrich.

        Returns:
            Enriched entity or result.
        """
        raise NotImplementedError

    def enrich_batch(
        self,
        entities: List[InputT],
        verbose: bool = False,
    ) -> List[OutputT]:
        """
        Enrich a batch of entities.

        Default implementation calls enrich() for each entity.
        Override for optimized batch processing.

        Args:
            entities: List of entities to enrich.
            verbose: Print progress information.

        Returns:
            List of enriched entities/results.
        """
        if not self.enabled:
            return entities  # type: ignore[return-value]

        results = []
        for entity in entities:
            result = self.enrich(entity)
            results.append(result)

        return results

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(enabled={self.enabled})"
