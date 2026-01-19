# corpus_metadata/E_normalization/E12_patient_journey_enricher.py
"""
Patient Journey Enricher for clinical trial feasibility extraction.

Extracts longitudinal progression and real-world barriers that determine
how patients reach (or fail to reach) trial eligibility.

Uses ZeroShotBioNER's zero-shot capability with custom entity labels:
- diagnostic_delay: Time from symptom onset to diagnosis
- treatment_line: Prior therapies (1L/2L/3L)
- care_pathway_step: Healthcare journey milestones
- surveillance_frequency: Monitoring intervals
- pain_point: Patient barriers
- recruitment_touchpoint: Trial entry points

Model: ProdicusII/ZeroShotBioNER (BioBERT-based zero-shot NER)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Entity labels for patient journey extraction
PATIENT_JOURNEY_LABELS = [
    "diagnostic_delay",        # Time from symptom onset to diagnosis
    "treatment_line",          # Prior therapies (1L/2L/3L)
    "care_pathway_step",       # Healthcare journey milestones
    "surveillance_frequency",  # Monitoring intervals
    "pain_point",              # Patient barriers
    "recruitment_touchpoint",  # Trial entry points
]


@dataclass
class PatientJourneyEntity:
    """Single entity extracted for patient journey."""

    text: str
    entity_type: str
    score: float
    start: int = 0
    end: int = 0


@dataclass
class PatientJourneyResult:
    """Structured result from patient journey extraction."""

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
        """Convert to summary dict for logging/export."""
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
        """Total number of extracted entities."""
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

    Reuses ZeroShotBioNER's extract_custom() method with patient journey
    entity labels to extract longitudinal progression information.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.run_id = config.get("run_id", "unknown")
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.entity_labels = config.get("entity_labels", PATIENT_JOURNEY_LABELS)
        self._zeroshot = None

    def _load_zeroshot(self) -> bool:
        """Lazy load ZeroShotBioNER enricher."""
        if self._zeroshot is not None:
            return True

        try:
            from E_normalization.E09_zeroshot_bioner import ZeroShotBioNEREnricher

            self._zeroshot = ZeroShotBioNEREnricher(
                config={
                    "confidence_threshold": self.confidence_threshold,
                }
            )
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

        Args:
            text: Input text (document, section, or abstract)

        Returns:
            PatientJourneyResult with categorized entities
        """
        result = PatientJourneyResult()
        start_time = time.time()

        if not text or not text.strip():
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

                    # Route to appropriate list
                    if entity_type == "diagnostic_delay":
                        result.diagnostic_delays.append(pj_entity)
                    elif entity_type == "treatment_line":
                        result.treatment_lines.append(pj_entity)
                    elif entity_type == "care_pathway_step":
                        result.care_pathway_steps.append(pj_entity)
                    elif entity_type == "surveillance_frequency":
                        result.surveillance_frequencies.append(pj_entity)
                    elif entity_type == "pain_point":
                        result.pain_points.append(pj_entity)
                    elif entity_type == "recruitment_touchpoint":
                        result.recruitment_touchpoints.append(pj_entity)

        except Exception as e:
            logger.error(f"Error during patient journey extraction: {e}")

        result.extraction_time_seconds = time.time() - start_time
        return result


def extract_patient_journey(
    text: str, config: Optional[Dict[str, Any]] = None
) -> PatientJourneyResult:
    """
    Convenience function for quick patient journey extraction.

    Args:
        text: Input text
        config: Optional configuration

    Returns:
        PatientJourneyResult with extracted entities
    """
    enricher = PatientJourneyEnricher(config)
    return enricher.extract(text)
