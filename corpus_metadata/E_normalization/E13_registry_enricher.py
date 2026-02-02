# corpus_metadata/E_normalization/E13_registry_enricher.py
"""
Registry Enricher for clinical trial feasibility extraction.

This module extracts patient registry information that provides real-world
cohort access, natural history data, and external control arms for rare-disease
trials. Registry data is critical for feasibility because it helps identify:

- Existing patient cohorts that could be recruited
- Natural history data for trial design
- Geographic distribution of patients
- Data availability for external control arms

Entity Types Extracted:
    - registry_name: Registry identifiers and names
      Examples: "RaDaR", "APPEAR-C3G", "NORD", "Orphanet"

    - registry_size: Cohort sizes and enrollment numbers
      Examples: "N=1,247 patients", "annual accrual 50 pts"

    - geographic_coverage: Geographic scope of registry
      Examples: "EU-wide", "US centers", "national registry"

    - data_types: Available data elements
      Examples: "genomics", "HPO phenotypes", "longitudinal labs"

    - access_policy: Data access terms and requirements
      Examples: "research-only", "IRB approval required", "federated query"

    - eligibility_criteria: Registry inclusion criteria
      Examples: "confirmed DMD genotype", "biopsy-proven C3G"

Technical Implementation:
    Uses ZeroShotBioNER's zero-shot capability (ProdicusII/ZeroShotBioNER)
    with registry-specific entity labels. Includes post-processing linkage
    to a database of known rare disease registries.

Known Registries Database:
    The module includes a curated database of known rare disease registries
    (KNOWN_REGISTRIES) for automatic linkage of extracted registry names
    to standardized registry information.

Example:
    >>> from E_normalization.E13_registry_enricher import RegistryEnricher
    >>> enricher = RegistryEnricher()
    >>> result = enricher.extract(clinical_text)
    >>> for reg in result.registry_names:
    ...     if reg.linked_registry:
    ...         print(f"{reg.text} -> {reg.linked_registry['full_name']}")

Dependencies:
    - E_normalization.E09_zeroshot_bioner: Zero-shot NER model
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from A_core.A00_logging import get_logger

if TYPE_CHECKING:
    from E_normalization.E09_zeroshot_bioner import ZeroShotBioNEREnricher

# Module logger
logger = get_logger(__name__)


# Entity labels for registry extraction
# These are passed to ZeroShotBioNER's extract_custom() method
REGISTRY_LABELS: List[str] = [
    "registry_name",        # Registry identifiers (RaDaR, APPEAR-C3G, Kidgo, GARD)
    "registry_size",        # Cohort sizes (N=1,247 patients)
    "geographic_coverage",  # Geographic scope (EU-wide, US centers, national)
    "data_types",           # Available data types (genomics, phenotypes, labs)
    "access_policy",        # Data access policy (research-only, federated query)
    "eligibility_criteria", # Registry inclusion criteria
]


# Known rare disease registries for post-processing linkage
# Maps lowercase registry identifiers to metadata
KNOWN_REGISTRIES: Dict[str, Dict[str, str]] = {
    # Rare disease registries
    "radar": {
        "full_name": "RaDaR - Rare Diseases Registry",
        "scope": "UK",
        "type": "renal",
    },
    "appear-c3g": {
        "full_name": "APPEAR-C3G Registry",
        "scope": "International",
        "type": "C3G",
    },
    "kidgo": {
        "full_name": "KDIGO Registry",
        "scope": "International",
        "type": "kidney",
    },
    "gard": {
        "full_name": "Genetic and Rare Diseases Information Center",
        "scope": "US",
        "type": "rare_disease",
    },
    "ordr": {
        "full_name": "Office of Rare Diseases Research",
        "scope": "US",
        "type": "rare_disease",
    },
    "eurordis": {
        "full_name": "EURORDIS Rare Diseases Europe",
        "scope": "EU",
        "type": "rare_disease",
    },
    "nord": {
        "full_name": "National Organization for Rare Disorders",
        "scope": "US",
        "type": "rare_disease",
    },
    "orphanet": {
        "full_name": "Orphanet",
        "scope": "International",
        "type": "rare_disease",
    },
    # Disease-specific registries
    "cftr2": {
        "full_name": "CFTR2 - CF Mutation Database",
        "scope": "International",
        "type": "cystic_fibrosis",
    },
    "cinrg": {
        "full_name": "CINRG Duchenne Natural History Study",
        "scope": "International",
        "type": "DMD",
    },
    "treat-nmd": {
        "full_name": "TREAT-NMD Alliance",
        "scope": "International",
        "type": "neuromuscular",
    },
}


@dataclass
class RegistryEntity:
    """
    Single entity extracted for registry information.

    Represents a text span identified as relevant to patient registry
    information. For registry names, includes optional linkage to
    the known registries database.

    Attributes:
        text: The extracted text span.
        entity_type: Category of registry information.
        score: Model confidence score (0.0-1.0).
        start: Character start position in source text.
        end: Character end position in source text.
        linked_registry: Matched registry metadata from KNOWN_REGISTRIES.
    """

    text: str
    entity_type: str
    score: float
    start: int = 0
    end: int = 0
    linked_registry: Optional[Dict[str, str]] = None

    def __repr__(self) -> str:
        """Return concise string representation."""
        linked_str = f" -> {self.linked_registry['full_name']}" if self.linked_registry else ""
        return f"RegistryEntity({self.entity_type}: '{self.text[:30]}'{linked_str})"


@dataclass
class RegistryResult:
    """
    Structured result from registry extraction.

    Groups extracted entities by category and provides methods
    for accessing linked registries and generating summaries.

    Attributes:
        registry_names: Registry identifiers and names.
        registry_sizes: Cohort size information.
        geographic_coverages: Geographic scope information.
        data_types: Available data type information.
        access_policies: Data access terms.
        eligibility_criteria: Registry inclusion criteria.
        raw_entities: Original extraction output for debugging.
        extraction_time_seconds: Processing time.

    Example:
        >>> result = enricher.extract(text)
        >>> linked = result.get_linked_registries()
        >>> for reg in linked:
        ...     print(f"{reg['extracted_text']} -> {reg['full_name']}")
    """

    # Core registry profile
    registry_names: List[RegistryEntity] = field(default_factory=list)
    registry_sizes: List[RegistryEntity] = field(default_factory=list)
    geographic_coverages: List[RegistryEntity] = field(default_factory=list)

    # Data & access
    data_types: List[RegistryEntity] = field(default_factory=list)
    access_policies: List[RegistryEntity] = field(default_factory=list)
    eligibility_criteria: List[RegistryEntity] = field(default_factory=list)

    # Raw entities for debugging
    raw_entities: List[Dict[str, Any]] = field(default_factory=list)

    # Timing
    extraction_time_seconds: float = 0.0

    def to_summary(self) -> Dict[str, Any]:
        """
        Convert to summary dictionary for logging/export.

        Returns:
            Dictionary with entity counts per category and timing.
        """
        return {
            "registry_name": len(self.registry_names),
            "registry_size": len(self.registry_sizes),
            "geographic_coverage": len(self.geographic_coverages),
            "data_types": len(self.data_types),
            "access_policy": len(self.access_policies),
            "eligibility_criteria": len(self.eligibility_criteria),
            "total": self.total_entities,
            "extraction_time_seconds": self.extraction_time_seconds,
        }

    @property
    def total_entities(self) -> int:
        """
        Total number of extracted entities across all categories.

        Returns:
            Sum of all entity list lengths.
        """
        return (
            len(self.registry_names)
            + len(self.registry_sizes)
            + len(self.geographic_coverages)
            + len(self.data_types)
            + len(self.access_policies)
            + len(self.eligibility_criteria)
        )

    def get_linked_registries(self) -> List[Dict[str, Any]]:
        """
        Get list of registries that were linked to known databases.

        Returns:
            List of dictionaries containing extracted text, confidence,
            and linked registry metadata.

        Example:
            >>> linked = result.get_linked_registries()
            >>> for reg in linked:
            ...     print(f"{reg['extracted_text']} matched {reg['full_name']}")
        """
        linked: List[Dict[str, Any]] = []
        for entity in self.registry_names:
            if entity.linked_registry:
                linked.append({
                    "extracted_text": entity.text,
                    "confidence": entity.score,
                    **entity.linked_registry,
                })
        return linked


class RegistryEnricher:
    """
    Enricher for patient registry extraction using ZeroShotBioNER.

    This class extracts information about patient registries mentioned
    in clinical documents, including registry names, sizes, geographic
    coverage, and access policies.

    Extracted registry names are automatically linked to a database
    of known rare disease registries when possible.

    The underlying ZeroShotBioNER model is loaded lazily on first use
    to avoid unnecessary overhead when this enricher is not needed.

    Attributes:
        run_id: Unique identifier for tracking.
        confidence_threshold: Minimum score to accept an entity.
        entity_labels: List of entity types to extract.
        link_registries: Whether to link to known registry database.

    Example:
        >>> enricher = RegistryEnricher({
        ...     "confidence_threshold": 0.6,
        ...     "link_registries": True
        ... })
        >>> result = enricher.extract(protocol_text)
        >>> for reg in result.get_linked_registries():
        ...     print(f"Found: {reg['full_name']}")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the registry enricher.

        Args:
            config: Optional configuration dictionary with keys:
                - run_id: Unique run identifier for tracking
                - confidence_threshold: Minimum entity confidence (default: 0.5)
                - entity_labels: Custom entity labels (default: REGISTRY_LABELS)
                - link_registries: Enable known registry linkage (default: True)
        """
        config = config or {}
        self.run_id: str = config.get("run_id", "unknown")
        self.confidence_threshold: float = config.get("confidence_threshold", 0.5)
        self.entity_labels: List[str] = config.get("entity_labels", REGISTRY_LABELS)
        self.link_registries: bool = config.get("link_registries", True)
        self._zeroshot: Optional["ZeroShotBioNEREnricher"] = None

        logger.debug(
            f"RegistryEnricher initialized with {len(self.entity_labels)} entity labels, "
            f"linkage={'enabled' if self.link_registries else 'disabled'}"
        )

    def _load_zeroshot(self) -> bool:
        """
        Lazy load ZeroShotBioNER enricher.

        Defers model loading until first extraction to minimize
        startup overhead.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self._zeroshot is not None:
            return True

        try:
            from E_normalization.E09_zeroshot_bioner import ZeroShotBioNEREnricher

            self._zeroshot = ZeroShotBioNEREnricher(
                config={
                    "confidence_threshold": self.confidence_threshold,
                }
            )
            logger.debug("ZeroShotBioNER loaded for registry extraction")
            return True

        except ImportError as e:
            logger.warning(f"Failed to import ZeroShotBioNEREnricher: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize ZeroShotBioNER: {e}")
            return False

    def _link_to_known_registry(self, text: str) -> Optional[Dict[str, str]]:
        """
        Attempt to link extracted registry name to known registry database.

        Uses both exact key matching and fuzzy matching on full names
        to identify known registries.

        Args:
            text: Extracted registry name text.

        Returns:
            Dictionary with linked registry metadata, or None if no match.
        """
        if not self.link_registries:
            return None

        text_lower = text.lower().strip()

        # Direct match on registry key
        for key, info in KNOWN_REGISTRIES.items():
            if key in text_lower or text_lower in info["full_name"].lower():
                logger.debug(f"Linked '{text}' to known registry: {info['full_name']}")
                return info

        # Partial match on full name words
        text_words: Set[str] = set(text_lower.split())
        for key, info in KNOWN_REGISTRIES.items():
            full_name_lower = info["full_name"].lower()
            name_words: Set[str] = set(full_name_lower.split())
            # Check if any significant word matches (excluding common words)
            common_words = {"the", "a", "an", "of", "for", "and", "-"}
            significant_overlap = (text_words & name_words) - common_words
            if significant_overlap:
                logger.debug(f"Linked '{text}' to known registry via word match: {info['full_name']}")
                return info

        return None

    def extract(self, text: str) -> RegistryResult:
        """
        Extract registry entities from text.

        Analyzes the input text for mentions of patient registries,
        including names, sizes, geographic coverage, and access policies.
        Registry names are automatically linked to known registries
        when possible.

        Args:
            text: Input text (document, section, or abstract).

        Returns:
            RegistryResult with categorized entities.

        Example:
            >>> result = enricher.extract(clinical_text)
            >>> for reg in result.registry_names:
            ...     print(f"Found registry: {reg.text}")
        """
        result = RegistryResult()
        start_time = time.time()

        if not text or not text.strip():
            logger.debug("Empty text provided, returning empty result")
            return result

        if not self._load_zeroshot():
            logger.warning("ZeroShotBioNER not available, returning empty result")
            return result

        assert self._zeroshot is not None  # Guaranteed after successful _load_zeroshot()

        try:
            # Use extract_custom with registry labels
            raw_results = self._zeroshot.extract_custom(text, self.entity_labels)

            # Categorize extracted entities
            for entity_type, entities in raw_results.items():
                for entity in entities:
                    # Attempt to link registry names to known registries
                    linked = None
                    if entity_type == "registry_name":
                        linked = self._link_to_known_registry(entity.text)

                    # Create RegistryEntity
                    reg_entity = RegistryEntity(
                        text=entity.text,
                        entity_type=entity_type,
                        score=entity.score,
                        start=entity.start,
                        end=entity.end,
                        linked_registry=linked,
                    )

                    # Store raw entity for debugging
                    result.raw_entities.append({
                        "text": entity.text,
                        "type": entity_type,
                        "score": entity.score,
                        "start": entity.start,
                        "end": entity.end,
                        "linked": linked,
                    })

                    # Route to appropriate list based on entity type
                    self._route_entity(result, entity_type, reg_entity)

            logger.debug(f"Extracted {result.total_entities} registry entities")

        except Exception as e:
            logger.error(f"Error during registry extraction: {e}")

        result.extraction_time_seconds = time.time() - start_time
        return result

    def _route_entity(
        self,
        result: RegistryResult,
        entity_type: str,
        entity: RegistryEntity,
    ) -> None:
        """
        Route an entity to the appropriate result list.

        Args:
            result: The result object to update.
            entity_type: Category of the entity.
            entity: The entity to route.
        """
        routing_map = {
            "registry_name": result.registry_names,
            "registry_size": result.registry_sizes,
            "geographic_coverage": result.geographic_coverages,
            "data_types": result.data_types,
            "access_policy": result.access_policies,
            "eligibility_criteria": result.eligibility_criteria,
        }

        target_list = routing_map.get(entity_type)
        if target_list is not None:
            target_list.append(entity)


def extract_registries(
    text: str,
    config: Optional[Dict[str, Any]] = None,
) -> RegistryResult:
    """
    Convenience function for quick registry extraction.

    Creates a temporary enricher instance and extracts registry
    entities from the provided text.

    Args:
        text: Input text to analyze.
        config: Optional configuration dictionary.

    Returns:
        RegistryResult with extracted entities.

    Example:
        >>> result = extract_registries(clinical_text)
        >>> linked = result.get_linked_registries()
        >>> print(f"Found {len(linked)} known registries")
    """
    enricher = RegistryEnricher(config)
    return enricher.extract(text)


__all__ = [
    "REGISTRY_LABELS",
    "KNOWN_REGISTRIES",
    "RegistryEntity",
    "RegistryResult",
    "RegistryEnricher",
    "extract_registries",
]
