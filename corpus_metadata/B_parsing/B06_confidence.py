# corpus_metadata/corpus_metadata/B_parsing/B06_confidence.py
"""
Feature-based confidence scoring framework for extraction candidates.

Provides a unified confidence calculation system for all extractors (C06-C08+).
Confidence is computed from multiple features rather than a single heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


# =============================================================================
# SPECULATION AND UNCERTAINTY CUES
# =============================================================================

# Words/phrases that indicate uncertainty (reduce confidence)
SPECULATION_CUES: Set[str] = {
    "may", "might", "could", "possibly", "potentially",
    "assuming", "estimated", "approximately", "about", "roughly",
    "perhaps", "likely", "unlikely", "probable", "uncertain",
    "suggested", "hypothesized", "speculated", "presumed",
}

# Words/phrases that indicate high certainty (boost confidence)
CERTAINTY_CUES: Set[str] = {
    "confirmed", "established", "demonstrated", "proven", "verified",
    "definitive", "conclusive", "documented", "validated",
}


# =============================================================================
# CONFIDENCE FEATURES DATACLASS
# =============================================================================


@dataclass
class ConfidenceFeatures:
    """
    Feature-based confidence score calculation.

    Each feature contributes to the final confidence score.
    Features can be positive (bonus) or negative (penalty).

    Usage:
        features = ConfidenceFeatures()
        features.section_match = 0.2  # In expected section
        features.pattern_strength = 0.15  # Strong pattern match
        features.negation_penalty = -0.1  # Contains speculation
        score = features.total()  # 0.5 + 0.2 + 0.15 - 0.1 = 0.75
    """

    # Section context (0.0 to 0.2)
    # Bonus when found in expected section for this field type
    section_match: float = 0.0

    # Pattern/anchor strength (0.0 to 0.3)
    # Higher for explicit patterns like "Primary endpoint:"
    pattern_strength: float = 0.0

    # Anchor proximity (0.0 to 0.2)
    # Bonus when found near key anchor phrases
    anchor_proximity: float = 0.0

    # Negation/speculation penalty (-0.2 to 0.0)
    # Penalty for uncertain language
    negation_penalty: float = 0.0

    # Context completeness (0.0 to 0.2)
    # For epi data: bonus when geography/time/setting present
    context_completeness: float = 0.0

    # Source quality (0.0 to 0.15)
    # Bonus for authoritative sources (tables, structured sections)
    source_quality: float = 0.0

    # Lexicon match (0.0 to 0.15)
    # Bonus when matched against curated lexicon
    lexicon_match: float = 0.0

    # External validation (0.0 to 0.2)
    # Bonus when validated by external source (PubTator, UMLS)
    external_validation: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert features to dictionary for storage/debugging."""
        return {
            "section_match": self.section_match,
            "pattern_strength": self.pattern_strength,
            "anchor_proximity": self.anchor_proximity,
            "negation_penalty": self.negation_penalty,
            "context_completeness": self.context_completeness,
            "source_quality": self.source_quality,
            "lexicon_match": self.lexicon_match,
            "external_validation": self.external_validation,
        }

    def total(self, base: float = 0.5, min_score: float = 0.1, max_score: float = 0.95) -> float:
        """
        Calculate total confidence score.

        Args:
            base: Base confidence (default 0.5)
            min_score: Minimum allowed score
            max_score: Maximum allowed score

        Returns:
            Clamped confidence score
        """
        score = base + sum([
            self.section_match,
            self.pattern_strength,
            self.anchor_proximity,
            self.negation_penalty,
            self.context_completeness,
            self.source_quality,
            self.lexicon_match,
            self.external_validation,
        ])
        return max(min_score, min(max_score, score))

    def apply_speculation_check(self, text: str, window_size: int = 100) -> None:
        """
        Check text prefix for speculation cues and apply penalty.

        Args:
            text: Text to check
            window_size: Number of characters to check from start
        """
        text_prefix = text[:window_size].lower()
        words = set(text_prefix.split())

        if words & SPECULATION_CUES:
            self.negation_penalty = -0.15
        elif words & CERTAINTY_CUES:
            self.pattern_strength += 0.05

    def apply_section_bonus(
        self,
        current_section: str,
        expected_sections: List[str],
        bonus: float = 0.2,
    ) -> None:
        """
        Apply section match bonus if in expected section.

        Args:
            current_section: Current document section
            expected_sections: List of expected sections for this field type
            bonus: Bonus to apply (default 0.2)
        """
        if current_section in expected_sections:
            self.section_match = bonus


# =============================================================================
# CONFIDENCE CALCULATOR CLASS
# =============================================================================


class ConfidenceCalculator:
    """
    Unified confidence calculator for all extractors.

    Encapsulates common confidence calculation logic.
    """

    def __init__(
        self,
        base_confidence: float = 0.5,
        min_confidence: float = 0.1,
        max_confidence: float = 0.95,
    ):
        self.base = base_confidence
        self.min = min_confidence
        self.max = max_confidence

        # Expected sections per field type
        self.expected_sections: Dict[str, List[str]] = {
            # Feasibility fields
            "ELIGIBILITY_INCLUSION": ["eligibility", "methods"],
            "ELIGIBILITY_EXCLUSION": ["eligibility", "methods"],
            "EPIDEMIOLOGY_PREVALENCE": ["epidemiology", "abstract"],
            "EPIDEMIOLOGY_INCIDENCE": ["epidemiology", "abstract"],
            "EPIDEMIOLOGY_DEMOGRAPHICS": ["epidemiology", "methods", "results"],
            "STUDY_ENDPOINT": ["endpoints", "methods"],
            "PATIENT_JOURNEY_PHASE": ["patient_journey", "methods"],
            "STUDY_SITE": ["methods"],
            # Disease/Drug fields
            "DISEASE": ["abstract", "methods", "results", "discussion"],
            "DRUG": ["abstract", "methods", "results"],
            # Abbreviation
            "ABBREVIATION": ["methods", "abstract"],
        }

    def create_features(self) -> ConfidenceFeatures:
        """Create new ConfidenceFeatures instance."""
        return ConfidenceFeatures()

    def calculate(
        self,
        field_type: str,
        section: str,
        text: str,
        match_text: str,
        lexicon_matched: bool = False,
        externally_validated: bool = False,
        from_table: bool = False,
    ) -> ConfidenceFeatures:
        """
        Calculate confidence features for a candidate.

        Args:
            field_type: Type of field being extracted
            section: Current document section
            text: Full context text
            match_text: Matched/extracted text
            lexicon_matched: Whether matched against lexicon
            externally_validated: Whether validated by external source
            from_table: Whether extracted from structured table

        Returns:
            ConfidenceFeatures with calculated values
        """
        features = ConfidenceFeatures()

        # Section bonus
        expected = self.expected_sections.get(field_type, [])
        features.apply_section_bonus(section, expected)

        # Speculation check
        features.apply_speculation_check(text)

        # Base pattern strength
        features.pattern_strength = 0.1

        # Source quality bonus for tables
        if from_table:
            features.source_quality = 0.15

        # Lexicon match bonus
        if lexicon_matched:
            features.lexicon_match = 0.1

        # External validation bonus
        if externally_validated:
            features.external_validation = 0.15

        return features

    def get_expected_sections(self, field_type: str) -> List[str]:
        """Get expected sections for a field type."""
        return self.expected_sections.get(field_type, [])


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Singleton calculator
_DEFAULT_CALCULATOR: Optional[ConfidenceCalculator] = None


def get_confidence_calculator() -> ConfidenceCalculator:
    """Get default confidence calculator instance."""
    global _DEFAULT_CALCULATOR
    if _DEFAULT_CALCULATOR is None:
        _DEFAULT_CALCULATOR = ConfidenceCalculator()
    return _DEFAULT_CALCULATOR


def calculate_confidence(
    field_type: str,
    section: str,
    text: str,
    match_text: str,
    **kwargs,
) -> float:
    """
    Simple function to calculate confidence score.

    Returns float confidence value.
    """
    calculator = get_confidence_calculator()
    features = calculator.calculate(field_type, section, text, match_text, **kwargs)
    return features.total()
