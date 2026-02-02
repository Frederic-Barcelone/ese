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
    Primary: Pattern-based extraction using curated regex patterns for each
    entity type. This approach is reliable and interpretable.

    Fallback: ZeroShotBioNER model (when available) for additional coverage.
    Note: The ZeroShotBioNER model was trained on specific entity types
    (ADE, Dosage, etc.) and may not recognize custom patient journey labels.

Example:
    >>> from E_normalization.E12_patient_journey_enricher import PatientJourneyEnricher
    >>> enricher = PatientJourneyEnricher()
    >>> result = enricher.extract(clinical_text)
    >>> print(f"Found {result.total_entities} patient journey entities")
    >>> for delay in result.diagnostic_delays:
    ...     print(f"  Diagnostic delay: {delay.text}")

Dependencies:
    - re: Regular expressions for pattern matching
    - E_normalization.E09_zeroshot_bioner: Zero-shot NER model (optional fallback)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, TYPE_CHECKING

from A_core.A00_logging import get_logger

if TYPE_CHECKING:
    from E_normalization.E09_zeroshot_bioner import ZeroShotBioNEREnricher

# Module logger
logger = get_logger(__name__)


# Entity labels for patient journey extraction
PATIENT_JOURNEY_LABELS: List[str] = [
    "diagnostic_delay",        # Time from symptom onset to diagnosis
    "treatment_line",          # Prior therapies (1L/2L/3L)
    "care_pathway_step",       # Healthcare journey milestones
    "surveillance_frequency",  # Monitoring intervals
    "pain_point",              # Patient barriers
    "recruitment_touchpoint",  # Trial entry points
]


# =============================================================================
# PATTERN-BASED EXTRACTION
# =============================================================================
# Primary extraction method using curated regex patterns for reliable detection.
# These patterns are designed to capture common clinical trial language.

# Diagnostic delay patterns
DIAGNOSTIC_DELAY_PATTERNS: List[str] = [
    # Time from symptoms to diagnosis
    r"(?:average|median|mean)?\s*(?:\d+[-–]?\d*\s*)?(?:year|month|week|day)s?\s*(?:diagnostic\s*)?delay",
    r"delay(?:ed)?\s+(?:in\s+)?diagnosis\s+(?:of\s+)?(?:up\s+to\s+)?(?:\d+\s*)?(?:year|month|week|day)s?",
    r"(?:time|duration)\s+(?:to|from|between)\s+(?:symptom\s+)?(?:onset\s+)?(?:to\s+)?diagnosis",
    r"diagnosed?\s+(?:after|within)\s+(?:\d+\s*)?(?:year|month|week|day)s?\s+(?:of\s+)?(?:symptom|presentation)",
    r"(?:symptom|disease)\s+onset\s+to\s+diagnosis",
    r"(?:early|late|delayed)\s+diagnosis",
    r"misdiagnos(?:is|ed)\s+(?:for|as)\s+(?:\d+\s*)?(?:year|month|week|day)s?",
    r"undiagnosed\s+(?:for\s+)?(?:\d+\s*)?(?:year|month|week|day)s?",
]

# Treatment line patterns
TREATMENT_LINE_PATTERNS: List[str] = [
    # Line of therapy
    r"(?:first|second|third|fourth|1st|2nd|3rd|4th|1L|2L|3L|4L)[-\s]*line\s+(?:therapy|treatment|regimen)",
    r"(?:after|following|post)\s+(?:failure\s+of\s+)?(?:first|second|third|1st|2nd|3rd)[-\s]*line",
    r"treatment[-\s]*(?:naive|naïve|experienced)\s+patients?",
    r"prior\s+(?:systemic\s+)?(?:therapy|treatment|medication|regimen)s?",
    r"(?:at\s+least\s+)?\d+\s+(?:prior|previous)\s+(?:lines?\s+of\s+)?(?:therapy|treatment)",
    r"(?:refractory|resistant|relapsed)\s+(?:to|after)\s+(?:\d+\s+)?(?:lines?\s+of\s+)?(?:therapy|treatment)",
    r"(?:standard|frontline|initial|induction|maintenance|salvage|consolidation)\s+(?:therapy|treatment|regimen)",
    r"(?:inadequate|insufficient|partial)\s+response\s+to\s+(?:\w+\s+)?(?:therapy|treatment)",
]

# Care pathway step patterns
CARE_PATHWAY_STEPS_PATTERNS: List[str] = [
    # Referral and care transitions
    r"(?:referred|referral)\s+(?:to|from)\s+(?:a\s+)?(?:specialist|nephrologist|center|clinic)",
    r"(?:genetic|biomarker|diagnostic)\s+(?:test(?:ing)?|screening)\s+(?:completed|performed|required)",
    r"(?:biopsy|histological|pathological)\s+(?:confirmation|diagnosis|examination)",
    r"(?:specialist|expert|tertiary)\s+(?:center|clinic|consultation|referral|care)",
    r"(?:primary|secondary|tertiary)\s+(?:care|healthcare)\s+(?:setting|provider|physician)",
    r"(?:hospitalization|admission|inpatient|outpatient)\s+(?:care|visit|treatment)",
    r"(?:emergency|urgent)\s+(?:room|department|care|presentation)",
    r"(?:follow[-\s]?up|surveillance|monitoring)\s+(?:visit|appointment|care)",
    r"multidisciplinary\s+(?:team|care|approach|management)",
    r"care\s+(?:coordination|transition|pathway|continuum)",
]

# Surveillance/monitoring frequency patterns
SURVEILLANCE_FREQUENCY_PATTERNS: List[str] = [
    # Monitoring intervals
    r"(?:daily|weekly|biweekly|monthly|quarterly|annually|every\s+\d+\s*(?:day|week|month|year)s?)\s+(?:monitoring|assessment|evaluation|visit|follow[-\s]?up)",
    r"(?:monitoring|assessment|evaluation|visit)\s+(?:every|each)\s+\d+\s*(?:day|week|month|year)s?",
    r"(?:frequent|regular|routine|periodic)\s+(?:monitoring|assessment|evaluation|laboratory|lab|blood)\s+(?:test|work|assessment)?",
    r"(?:MRI|CT|imaging|ultrasound|echocardiogram|ECG|EKG)\s+(?:every|each)\s+\d+\s*(?:day|week|month|year)s?",
    r"(?:urine|blood|serum|plasma)\s+(?:sample|collection|test)s?\s+(?:every|each)?\s*(?:\d+\s*)?(?:day|week|month)?",
    r"study\s+visit\s+(?:every|at)\s+(?:\d+\s*)?(?:day|week|month)s?",
    r"(?:screening|baseline|end[-\s]?of[-\s]?study)\s+visit",
]

# Pain points / barriers patterns
PAIN_POINT_PATTERNS: List[str] = [
    # Patient burden and barriers
    r"(?:travel|geographic|distance)\s+(?:burden|barrier|requirement|constraint)",
    r"(?:frequent|multiple|numerous)\s+(?:site|clinic|hospital)\s+visits?",
    r"(?:treatment|monitoring|visit)\s+(?:burden|fatigue|adherence)",
    r"(?:injection|infusion|administration)\s+(?:site|frequency|burden)",
    r"(?:quality\s+of\s+life|QoL)\s+(?:impact|burden|decline|impairment)",
    r"(?:adverse|side)\s+(?:event|effect)s?\s+(?:burden|risk|profile)",
    r"(?:caregiver|family)\s+(?:burden|impact|support)",
    r"(?:insurance|coverage|reimbursement|cost|financial)\s+(?:barrier|burden|challenge|constraint)",
    r"(?:work|employment|school)\s+(?:absence|disruption|impact|limitation)",
    r"(?:washout|run[-\s]?in)\s+period",
    r"(?:strict|stringent|complex)\s+(?:eligibility|inclusion|exclusion)\s+criteria",
]

# Recruitment touchpoint patterns
RECRUITMENT_TOUCHPOINT_PATTERNS: List[str] = [
    # Entry points for trial recruitment
    r"(?:at|during|upon|after)\s+(?:the\s+)?(?:time\s+of\s+)?(?:initial\s+)?diagnosis",
    r"(?:during|at)\s+(?:routine|regular|scheduled)\s+(?:follow[-\s]?up|visit|appointment|clinic)",
    r"(?:referral|referred)\s+(?:from|by)\s+(?:primary\s+care|specialist|physician|clinic)",
    r"(?:screening|identification|recruitment)\s+(?:at|in|through)\s+(?:clinic|hospital|center|registry)",
    r"(?:patient|disease)\s+(?:registry|registries|database)",
    r"(?:electronic\s+)?(?:health|medical)\s+record\s+(?:review|screening|identification)",
    r"(?:community|patient)\s+(?:outreach|engagement|advocacy|support\s+group)",
    r"(?:physician|site|investigator)\s+(?:referral|recommendation|identification)",
    r"(?:pre[-\s]?screening|eligibility)\s+(?:assessment|evaluation|visit)",
]

# Compile all patterns for efficiency
def _compile_patterns() -> Dict[str, List[Pattern]]:
    """Compile all regex patterns for each entity type."""
    return {
        "diagnostic_delay": [re.compile(p, re.IGNORECASE) for p in DIAGNOSTIC_DELAY_PATTERNS],
        "treatment_line": [re.compile(p, re.IGNORECASE) for p in TREATMENT_LINE_PATTERNS],
        "care_pathway_step": [re.compile(p, re.IGNORECASE) for p in CARE_PATHWAY_STEPS_PATTERNS],
        "surveillance_frequency": [re.compile(p, re.IGNORECASE) for p in SURVEILLANCE_FREQUENCY_PATTERNS],
        "pain_point": [re.compile(p, re.IGNORECASE) for p in PAIN_POINT_PATTERNS],
        "recruitment_touchpoint": [re.compile(p, re.IGNORECASE) for p in RECRUITMENT_TOUCHPOINT_PATTERNS],
    }

# Pre-compiled patterns (loaded once at module import)
COMPILED_PATTERNS: Dict[str, List[Pattern]] = _compile_patterns()


@dataclass
class PatientJourneyEntity:
    """
    Single entity extracted for patient journey analysis.

    Represents a text span identified as relevant to understanding
    the patient's path to trial eligibility.

    Attributes:
        text: The extracted text span (the entity itself).
        entity_type: Category of patient journey information.
        score: Confidence score (0.0-1.0).
        start: Character start position in source text.
        end: Character end position in source text.
        context: Surrounding text providing evidence for the extraction.
    """

    text: str
    entity_type: str
    score: float
    start: int = 0
    end: int = 0
    context: str = ""  # Evidence text around the match

    def __repr__(self) -> str:
        """Return concise string representation."""
        text_preview = self.text[:30] + "..." if len(self.text) > 30 else self.text
        return f"PatientJourneyEntity({self.entity_type}: '{text_preview}' conf={self.score:.2f})"


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
    Enricher for patient journey extraction using pattern matching.

    This class extracts information about the patient's path to trial
    eligibility, including diagnostic timelines, treatment history,
    and barriers to participation.

    Primary extraction uses curated regex patterns for reliable detection.
    ZeroShotBioNER can be used as an optional fallback, but note that
    the model was trained on specific entity types (ADE, Dosage, etc.)
    and may not recognize custom patient journey labels effectively.

    Attributes:
        run_id: Unique identifier for tracking.
        confidence_threshold: Minimum score to accept an entity.
        entity_labels: List of entity types to extract.
        use_zeroshot_fallback: Whether to use ZeroShotBioNER as fallback.
        context_window: Characters of context to include around matches.

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
                - use_zeroshot_fallback: Use ZeroShotBioNER as fallback (default: False)
                - context_window: Characters of context around matches (default: 150)
        """
        config = config or {}
        self.run_id: str = config.get("run_id", "unknown")
        self.confidence_threshold: float = config.get("confidence_threshold", 0.5)
        self.entity_labels: List[str] = config.get("entity_labels", PATIENT_JOURNEY_LABELS)
        self.use_zeroshot_fallback: bool = config.get("use_zeroshot_fallback", False)
        self.context_window: int = config.get("context_window", 150)
        self._zeroshot: Optional["ZeroShotBioNEREnricher"] = None

        logger.debug(
            f"PatientJourneyEnricher initialized with {len(self.entity_labels)} entity labels "
            f"(pattern-based extraction enabled)"
        )

    def _load_zeroshot(self) -> bool:
        """
        Lazy load ZeroShotBioNER enricher (optional fallback).

        Note: ZeroShotBioNER was trained on specific entity types
        (ADE, Dosage, Frequency, etc.) and may not recognize custom
        patient journey labels effectively. Use pattern-based extraction
        as the primary method.

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
            logger.debug("ZeroShotBioNER loaded as fallback for patient journey extraction")
            return True

        except ImportError as e:
            logger.debug(f"ZeroShotBioNEREnricher not available: {e}")
            return False
        except Exception as e:
            logger.debug(f"Failed to initialize ZeroShotBioNER fallback: {e}")
            return False

    def extract(self, text: str) -> PatientJourneyResult:
        """
        Extract patient journey entities from text.

        Uses pattern-based extraction as the primary method for reliable
        detection of patient journey information. Optionally uses
        ZeroShotBioNER as a fallback.

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

        # Track seen text spans to avoid duplicates
        seen_spans: set = set()

        # PRIMARY: Pattern-based extraction (reliable, interpretable)
        self._extract_with_patterns(text, result, seen_spans)

        # OPTIONAL FALLBACK: ZeroShotBioNER (if enabled and available)
        if self.use_zeroshot_fallback and self._load_zeroshot():
            self._extract_with_zeroshot(text, result, seen_spans)

        logger.debug(f"Extracted {result.total_entities} patient journey entities")
        result.extraction_time_seconds = time.time() - start_time
        return result

    def _extract_with_patterns(
        self,
        text: str,
        result: PatientJourneyResult,
        seen_spans: set,
    ) -> None:
        """
        Extract entities using pre-compiled regex patterns.

        Args:
            text: Input text to analyze.
            result: Result object to populate.
            seen_spans: Set of (start, end) tuples already seen.
        """
        for entity_type, patterns in COMPILED_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    start, end = match.start(), match.end()
                    matched_text = match.group(0).strip()

                    # Skip if we've seen this exact span
                    span_key = (start, end)
                    if span_key in seen_spans:
                        continue
                    seen_spans.add(span_key)

                    # Skip very short matches
                    if len(matched_text) < 5:
                        continue

                    # Extract context around the match (evidence)
                    context = self._extract_context(text, start, end)

                    # Create entity with high confidence (pattern match)
                    pj_entity = PatientJourneyEntity(
                        text=matched_text,
                        entity_type=entity_type,
                        score=0.90,  # High confidence for pattern matches
                        start=start,
                        end=end,
                        context=context,
                    )

                    # Store raw entity for debugging
                    result.raw_entities.append({
                        "text": matched_text,
                        "type": entity_type,
                        "score": 0.90,
                        "start": start,
                        "end": end,
                        "source": "pattern",
                        "context": context,
                    })

                    # Route to appropriate list
                    self._route_entity(result, entity_type, pj_entity)

    def _extract_with_zeroshot(
        self,
        text: str,
        result: PatientJourneyResult,
        seen_spans: set,
    ) -> None:
        """
        Extract entities using ZeroShotBioNER as fallback.

        Note: This is optional and may not yield results since
        ZeroShotBioNER was trained on different entity types.

        Args:
            text: Input text to analyze.
            result: Result object to populate.
            seen_spans: Set of (start, end) tuples already seen.
        """
        if self._zeroshot is None:
            return

        try:
            raw_results = self._zeroshot.extract_custom(text, self.entity_labels)

            for entity_type, entities in raw_results.items():
                for entity in entities:
                    # Skip if we've seen overlapping span
                    span_key = (entity.start, entity.end)
                    if span_key in seen_spans:
                        continue

                    # Check for overlap with existing spans
                    overlaps = any(
                        not (entity.end <= s[0] or entity.start >= s[1])
                        for s in seen_spans
                    )
                    if overlaps:
                        continue

                    seen_spans.add(span_key)

                    # Extract context
                    context = self._extract_context(text, entity.start, entity.end)

                    pj_entity = PatientJourneyEntity(
                        text=entity.text,
                        entity_type=entity_type,
                        score=entity.score,
                        start=entity.start,
                        end=entity.end,
                        context=context,
                    )

                    result.raw_entities.append({
                        "text": entity.text,
                        "type": entity_type,
                        "score": entity.score,
                        "start": entity.start,
                        "end": entity.end,
                        "source": "zeroshot",
                        "context": context,
                    })

                    self._route_entity(result, entity_type, pj_entity)

        except Exception as e:
            logger.debug(f"ZeroShotBioNER fallback failed: {e}")

    def _extract_context(self, text: str, start: int, end: int) -> str:
        """
        Extract context around a match for evidence.

        Args:
            text: Full text.
            start: Match start position.
            end: Match end position.

        Returns:
            Context string with match highlighted.
        """
        ctx_start = max(0, start - self.context_window)
        ctx_end = min(len(text), end + self.context_window)
        context = text[ctx_start:ctx_end].replace("\n", " ").strip()

        # Add ellipsis if truncated
        if ctx_start > 0:
            context = "..." + context
        if ctx_end < len(text):
            context = context + "..."

        return context

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


__all__ = [
    "PATIENT_JOURNEY_LABELS",
    "PatientJourneyEntity",
    "PatientJourneyResult",
    "PatientJourneyEnricher",
    "extract_patient_journey",
]
