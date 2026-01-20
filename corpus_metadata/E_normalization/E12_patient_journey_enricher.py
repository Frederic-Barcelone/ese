# corpus_metadata/E_normalization/E12_patient_journey_enricher.py
"""
Patient Journey Enricher for clinical trial feasibility extraction.

This module extracts longitudinal progression and real-world barriers that
determine how patients reach (or fail to reach) trial eligibility. Understanding
the patient journey is critical for feasibility assessment as it reveals:

- Time-to-diagnosis patterns affecting recruitment windows
- Prior treatment requirements impacting eligibility
- Healthcare system touchpoints for patient identification
- Barriers that cause patient dropout or non-enrollment

Entity Types Extracted:
    - diagnostic_delay: Time from symptom onset to diagnosis
      Examples: "average 3-year diagnostic delay", "median 18 months to diagnosis"

    - treatment_line: Prior therapy requirements (1L/2L/3L)
      Examples: "after failure of first-line therapy", "treatment-naive patients"

    - care_pathway_step: Healthcare journey milestones
      Examples: "referred to specialist center", "genetic testing completed"

    - surveillance_frequency: Monitoring intervals affecting trial burden
      Examples: "quarterly MRI monitoring", "monthly lab assessments"

    - pain_point: Patient barriers to trial participation
      Examples: "travel burden to study sites", "frequent monitoring requirements"

    - recruitment_touchpoint: Potential trial entry points
      Examples: "at time of diagnosis", "during routine follow-up"

Technical Implementation:
    Uses ZeroShotBioNER's zero-shot capability (ProdicusII/ZeroShotBioNER)
    with custom entity labels. The model is loaded lazily to minimize
    startup overhead when this enricher is not needed.

Example:
    >>> from E_normalization.E12_patient_journey_enricher import PatientJourneyEnricher
    >>> enricher = PatientJourneyEnricher()
    >>> result = enricher.extract(clinical_text)
    >>> print(f"Found {result.total_entities} patient journey entities")
    >>> for delay in result.diagnostic_delays:
    ...     print(f"  Diagnostic delay: {delay.text}")

Dependencies:
    - E_normalization.E09_zeroshot_bioner: Zero-shot NER model
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from A_core.A00_logging import get_logger

if TYPE_CHECKING:
    from E_normalization.E09_zeroshot_bioner import ZeroShotBioNEREnricher

# Module logger
logger = get_logger(__name__)


# Entity labels for patient journey extraction
# These are passed to ZeroShotBioNER's extract_custom() method
PATIENT_JOURNEY_LABELS: List[str] = [
    "diagnostic_delay",        # Time from symptom onset to diagnosis
    "treatment_line",          # Prior therapies (1L/2L/3L)
    "care_pathway_step",       # Healthcare journey milestones
    "surveillance_frequency",  # Monitoring intervals
    "pain_point",              # Patient barriers
    "recruitment_touchpoint",  # Trial entry points
]


@dataclass
class PatientJourneyEntity:
    """
    Single entity extracted for patient journey analysis.

    Represents a text span identified as relevant to understanding
    the patient's path to trial eligibility.

    Attributes:
        text: The extracted text span.
        entity_type: Category of patient journey information.
        score: Model confidence score (0.0-1.0).
        start: Character start position in source text.
        end: Character end position in source text.
    """

    text: str
    entity_type: str
    score: float
    start: int = 0
    end: int = 0

    def __repr__(self) -> str:
        """Return concise string representation."""
        return f"PatientJourneyEntity({self.entity_type}: '{self.text[:30]}...' conf={self.score:.2f})"


@dataclass
class PatientJourneyResult:
    """
    Structured result from patient journey extraction.

    Groups extracted entities by category and provides summary
    methods for logging and export.

    Attributes:
        diagnostic_delays: Time-to-diagnosis information.
        treatment_lines: Prior therapy requirements.
        care_pathway_steps: Healthcare journey milestones.
        surveillance_frequencies: Monitoring interval requirements.
        pain_points: Barriers to participation.
        recruitment_touchpoints: Potential trial entry points.
        raw_entities: Original extraction output for debugging.
        extraction_time_seconds: Processing time.

    Example:
        >>> result = enricher.extract(text)
        >>> if result.total_entities > 0:
        ...     for entity in result.diagnostic_delays:
        ...         print(f"Delay: {entity.text}")
    """

    # Diagnostic timeline
    diagnostic_delays: List[PatientJourneyEntity] = field(default_factory=list)

    # Treatment history
    treatment_lines: List[PatientJourneyEntity] = field(default_factory=list)

    # Care pathway
    care_pathway_steps: List[PatientJourneyEntity] = field(default_factory=list)

    # Trial burden
    surveillance_frequencies: List[PatientJourneyEntity] = field(default_factory=list)

    # Retention risks
    pain_points: List[PatientJourneyEntity] = field(default_factory=list)
    recruitment_touchpoints: List[PatientJourneyEntity] = field(default_factory=list)

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
            "diagnostic_delay": len(self.diagnostic_delays),
            "treatment_line": len(self.treatment_lines),
            "care_pathway_step": len(self.care_pathway_steps),
            "surveillance_frequency": len(self.surveillance_frequencies),
            "pain_point": len(self.pain_points),
            "recruitment_touchpoint": len(self.recruitment_touchpoints),
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
            len(self.diagnostic_delays)
            + len(self.treatment_lines)
            + len(self.care_pathway_steps)
            + len(self.surveillance_frequencies)
            + len(self.pain_points)
            + len(self.recruitment_touchpoints)
        )


class PatientJourneyEnricher:
    """
    Enricher for patient journey extraction using ZeroShotBioNER.

    This class extracts information about the patient's path to trial
    eligibility, including diagnostic timelines, treatment history,
    and barriers to participation.

    The underlying ZeroShotBioNER model is loaded lazily on first use
    to avoid unnecessary overhead when this enricher is not needed.

    Attributes:
        run_id: Unique identifier for tracking.
        confidence_threshold: Minimum score to accept an entity.
        entity_labels: List of entity types to extract.

    Example:
        >>> enricher = PatientJourneyEnricher({
        ...     "confidence_threshold": 0.6,
        ...     "run_id": "RUN_123"
        ... })
        >>> result = enricher.extract(document_text)
        >>> print(result.to_summary())
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the patient journey enricher.

        Args:
            config: Optional configuration dictionary with keys:
                - run_id: Unique run identifier for tracking
                - confidence_threshold: Minimum entity confidence (default: 0.5)
                - entity_labels: Custom entity labels (default: PATIENT_JOURNEY_LABELS)
        """
        config = config or {}
        self.run_id: str = config.get("run_id", "unknown")
        self.confidence_threshold: float = config.get("confidence_threshold", 0.5)
        self.entity_labels: List[str] = config.get("entity_labels", PATIENT_JOURNEY_LABELS)
        self._zeroshot: Optional["ZeroShotBioNEREnricher"] = None

        logger.debug(
            f"PatientJourneyEnricher initialized with {len(self.entity_labels)} entity labels"
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
            logger.debug("ZeroShotBioNER loaded for patient journey extraction")
            return True

        except ImportError as e:
            logger.warning(f"Failed to import ZeroShotBioNEREnricher: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize ZeroShotBioNER: {e}")
            return False

    def extract(self, text: str) -> PatientJourneyResult:
        """
        Extract patient journey entities from text.

        Analyzes the input text for mentions of diagnostic delays,
        treatment history, care pathways, and other patient journey
        information relevant to trial feasibility.

        Args:
            text: Input text (document, section, or abstract).

        Returns:
            PatientJourneyResult with categorized entities.

        Example:
            >>> result = enricher.extract(protocol_text)
            >>> for delay in result.diagnostic_delays:
            ...     print(f"Found delay mention: {delay.text}")
        """
        result = PatientJourneyResult()
        start_time = time.time()

        if not text or not text.strip():
            logger.debug("Empty text provided, returning empty result")
            return result

        if not self._load_zeroshot():
            logger.warning("ZeroShotBioNER not available, returning empty result")
            return result

        try:
            # Use extract_custom with patient journey labels
            raw_results = self._zeroshot.extract_custom(text, self.entity_labels)

            # Categorize extracted entities
            for entity_type, entities in raw_results.items():
                for entity in entities:
                    # Create PatientJourneyEntity
                    pj_entity = PatientJourneyEntity(
                        text=entity.text,
                        entity_type=entity_type,
                        score=entity.score,
                        start=entity.start,
                        end=entity.end,
                    )

                    # Store raw entity for debugging
                    result.raw_entities.append({
                        "text": entity.text,
                        "type": entity_type,
                        "score": entity.score,
                        "start": entity.start,
                        "end": entity.end,
                    })

                    # Route to appropriate list based on entity type
                    self._route_entity(result, entity_type, pj_entity)

            logger.debug(f"Extracted {result.total_entities} patient journey entities")

        except Exception as e:
            logger.error(f"Error during patient journey extraction: {e}")

        result.extraction_time_seconds = time.time() - start_time
        return result

    def _route_entity(
        self,
        result: PatientJourneyResult,
        entity_type: str,
        entity: PatientJourneyEntity,
    ) -> None:
        """
        Route an entity to the appropriate result list.

        Args:
            result: The result object to update.
            entity_type: Category of the entity.
            entity: The entity to route.
        """
        routing_map = {
            "diagnostic_delay": result.diagnostic_delays,
            "treatment_line": result.treatment_lines,
            "care_pathway_step": result.care_pathway_steps,
            "surveillance_frequency": result.surveillance_frequencies,
            "pain_point": result.pain_points,
            "recruitment_touchpoint": result.recruitment_touchpoints,
        }

        target_list = routing_map.get(entity_type)
        if target_list is not None:
            target_list.append(entity)


def extract_patient_journey(
    text: str,
    config: Optional[Dict[str, Any]] = None,
) -> PatientJourneyResult:
    """
    Convenience function for quick patient journey extraction.

    Creates a temporary enricher instance and extracts patient
    journey entities from the provided text.

    Args:
        text: Input text to analyze.
        config: Optional configuration dictionary.

    Returns:
        PatientJourneyResult with extracted entities.

    Example:
        >>> result = extract_patient_journey(clinical_text)
        >>> print(f"Found {result.total_entities} entities")
    """
    enricher = PatientJourneyEnricher(config)
    return enricher.extract(text)
