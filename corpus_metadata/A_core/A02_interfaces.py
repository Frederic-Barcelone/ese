from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic

# Python 3.9-friendly TypeAlias
from typing_extensions import TypeAlias

from A_core.A01_domain_models import (
    GeneratorType,
    Candidate,
    ExtractedEntity,
)

# Flexible doc model
DocumentModel: TypeAlias = Any

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
