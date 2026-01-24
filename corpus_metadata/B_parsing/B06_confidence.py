# corpus_metadata/corpus_metadata/B_parsing/B06_confidence.py
"""
Feature-based confidence scoring framework for extraction candidates.

Provides a unified confidence calculation system for all extractors (C06-C08+).
Confidence is computed from multiple features rather than a single heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


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

    # Quote verification penalty (-0.5 to 0.0)
    # Penalty when quote cannot be verified in source document
    # Set to -0.5 (50% penalty) when quote is unverified
    quote_verification_penalty: float = 0.0

    # Numerical verification penalty (-0.3 to 0.0)
    # Penalty when numerical values cannot be verified in source document
    # Set to -0.3 (30% of current score) per unverified number
    numerical_verification_penalty: float = 0.0

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
            "quote_verification_penalty": self.quote_verification_penalty,
            "numerical_verification_penalty": self.numerical_verification_penalty,
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
        # Calculate additive score first
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

        # Apply verification penalties as multipliers
        # Quote verification: unverified quote multiplies by 0.5
        if self.quote_verification_penalty < 0:
            score *= (1.0 + self.quote_verification_penalty)

        # Numerical verification: each unverified number multiplies by 0.7
        if self.numerical_verification_penalty < 0:
            score *= (1.0 + self.numerical_verification_penalty)

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

    def apply_quote_verification_penalty(self, verified: bool) -> None:
        """
        Apply penalty for unverified quote.

        Unverified quotes result in a 50% confidence reduction.

        Args:
            verified: Whether the quote was verified in source document
        """
        if not verified:
            self.quote_verification_penalty = -0.5

    def apply_numerical_verification_penalty(
        self,
        unverified_count: int,
        penalty_per_number: float = 0.3,
    ) -> None:
        """
        Apply penalty for unverified numerical values.

        Each unverified number results in a 30% confidence reduction.

        Args:
            unverified_count: Number of unverified numerical values
            penalty_per_number: Penalty multiplier per unverified number (default 0.3)
        """
        if unverified_count > 0:
            # Cap at 0.7 total penalty (30% of original score)
            total_penalty = min(0.7, unverified_count * penalty_per_number)
            self.numerical_verification_penalty = -total_penalty


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
        quote_verified: Optional[bool] = None,
        unverified_numbers: int = 0,
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
            quote_verified: Whether quote was verified (None = not checked)
            unverified_numbers: Count of unverified numerical values

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

        # Quote verification penalty
        if quote_verified is not None:
            features.apply_quote_verification_penalty(quote_verified)

        # Numerical verification penalty
        if unverified_numbers > 0:
            features.apply_numerical_verification_penalty(unverified_numbers)

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


def apply_verification_penalty(
    base_confidence: float,
    quote_verified: Optional[bool] = None,
    unverified_numbers: int = 0,
    min_confidence: float = 0.1,
) -> float:
    """
    Apply verification penalties to a confidence score.

    This is a standalone function for applying anti-hallucination penalties
    without going through the full ConfidenceFeatures calculation.

    Args:
        base_confidence: Starting confidence score (0.0 - 1.0)
        quote_verified: Whether quote was verified (None = not checked, True/False = verified/unverified)
        unverified_numbers: Count of unverified numerical values
        min_confidence: Minimum confidence floor

    Returns:
        Adjusted confidence score with penalties applied.
        - Unverified quote: confidence × 0.5
        - Unverified numbers: confidence × 0.7 (per number, capped at 0.3)

    Example:
        >>> apply_verification_penalty(0.9, quote_verified=False)
        0.45
        >>> apply_verification_penalty(0.9, unverified_numbers=2)
        0.441  # 0.9 * 0.7 * 0.7
    """
    confidence = base_confidence

    # Apply quote verification penalty (50% reduction)
    if quote_verified is False:
        confidence *= 0.5

    # Apply numerical verification penalty (30% reduction per number)
    for _ in range(min(unverified_numbers, 3)):  # Cap at 3 numbers
        confidence *= 0.7

    return max(min_confidence, round(confidence, 3))


# =============================================================================
# CRITERION-SPECIFIC CONFIDENCE SCORING
# =============================================================================


# Learned confidence weights per eligibility category
# These weights reflect empirical trustworthiness by criterion type
CRITERION_TYPE_WEIGHTS: Dict[str, float] = {
    # High confidence - structured, verifiable criteria
    "lab_value": 0.9,  # Lab values are objective and verifiable
    "age": 0.85,  # Age criteria are straightforward
    "biomarker": 0.85,  # Biomarkers have clear thresholds
    "organ_function": 0.8,  # Organ function tests are standardized

    # Medium-high confidence - clinical standards
    "disease_definition": 0.75,  # Disease definitions can be verified against ICD/SNOMED
    "diagnosis_confirmation": 0.75,  # Biopsy/genetic testing requirements
    "prior_treatment": 0.7,  # Treatment history is documentable
    "vaccination_requirement": 0.7,  # Vaccination records exist

    # Medium confidence - some subjectivity
    "disease_severity": 0.65,  # Severity grades (NYHA, ECOG) have standards but vary
    "comorbidity": 0.6,  # Comorbidity assessment can be subjective
    "concomitant_medications": 0.6,  # Medication lists may be incomplete
    "background_therapy": 0.6,  # Background therapy requirements vary

    # Lower confidence - high subjectivity
    "disease_duration": 0.55,  # Duration estimates are often imprecise
    "consent": 0.5,  # Consent criteria are procedural
    "administrative": 0.5,  # Administrative criteria are site-specific
    "pregnancy": 0.5,  # Pregnancy exclusions are standard but not always explicit
}


class CriterionConfidenceCalculator:
    """
    Criterion-specific confidence calculator.

    Adjusts confidence based on criterion type, recognizing that
    some criterion types are inherently more reliable than others.
    """

    def __init__(self, type_weights: Optional[Dict[str, float]] = None):
        self.type_weights = type_weights or CRITERION_TYPE_WEIGHTS
        self.base_calculator = get_confidence_calculator()

    def calculate(
        self,
        criterion_category: str,
        section: str,
        text: str,
        match_text: str,
        has_structured_value: bool = False,
        has_normalization: bool = False,
        **kwargs,
    ) -> float:
        """
        Calculate criterion-specific confidence.

        Args:
            criterion_category: Category of eligibility criterion (e.g., "lab_value", "age")
            section: Document section
            text: Full context text
            match_text: Extracted text
            has_structured_value: Whether criterion has structured value (LabCriterion, SeverityGrade)
            has_normalization: Whether criterion has ontology normalization
            **kwargs: Additional arguments for base calculator

        Returns:
            Adjusted confidence score
        """
        # Get base confidence features
        features = self.base_calculator.calculate(
            f"ELIGIBILITY_{criterion_category.upper()}",
            section,
            text,
            match_text,
            **kwargs,
        )
        base_score = features.total()

        # Get type-specific weight
        type_weight = self.type_weights.get(criterion_category.lower(), 0.6)

        # Bonuses for structured/normalized criteria
        structure_bonus = 0.1 if has_structured_value else 0.0
        normalization_bonus = 0.05 if has_normalization else 0.0

        # Calculate adjusted score
        adjusted = base_score * type_weight + structure_bonus + normalization_bonus

        return min(0.95, max(0.1, adjusted))

    def get_type_weight(self, criterion_category: str) -> float:
        """Get the confidence weight for a criterion type."""
        return self.type_weights.get(criterion_category.lower(), 0.6)


# =============================================================================
# CROSS-CRITERIA CONTRADICTION DETECTION
# =============================================================================


class ContradictionType:
    """Types of contradictions between criteria."""

    VALUE_CONFLICT = "value_conflict"  # e.g., age >= 18 AND age <= 16
    LOGICAL_CONFLICT = "logical_conflict"  # e.g., must have X AND must not have X
    RANGE_OVERLAP = "range_overlap"  # e.g., overlapping but incompatible ranges
    SEVERITY_CONFLICT = "severity_conflict"  # e.g., NYHA I-II AND NYHA III-IV


@dataclass
class Contradiction:
    """A detected contradiction between criteria."""

    type: str
    criterion_a_text: str
    criterion_b_text: str
    criterion_a_id: Optional[str] = None
    criterion_b_id: Optional[str] = None
    description: str = ""
    severity: str = "error"  # "error", "warning", "info"


class ContradictionDetector:
    """
    Detects contradictions between eligibility criteria.

    Identifies logical inconsistencies that would make eligibility
    impossible or ambiguous.
    """

    def detect_lab_contradictions(
        self, criteria: List[Dict[str, Any]]
    ) -> List[Contradiction]:
        """
        Detect contradictions in lab value criteria.

        Args:
            criteria: List of criteria with lab_criterion fields

        Returns:
            List of detected contradictions
        """
        contradictions = []
        lab_criteria = [c for c in criteria if c.get("lab_criterion")]

        # Group by analyte
        by_analyte: Dict[str, List[Dict[str, Any]]] = {}
        for crit in lab_criteria:
            lab = crit["lab_criterion"]
            analyte = lab.get("analyte", "").lower()
            if analyte not in by_analyte:
                by_analyte[analyte] = []
            by_analyte[analyte].append(crit)

        # Check each analyte group for conflicts
        for analyte, group in by_analyte.items():
            if len(group) < 2:
                continue

            for i, crit_a in enumerate(group):
                for crit_b in group[i + 1:]:
                    lab_a = crit_a["lab_criterion"]
                    lab_b = crit_b["lab_criterion"]

                    conflict = self._check_lab_conflict(lab_a, lab_b)
                    if conflict:
                        contradictions.append(
                            Contradiction(
                                type=ContradictionType.VALUE_CONFLICT,
                                criterion_a_text=crit_a.get("text", ""),
                                criterion_b_text=crit_b.get("text", ""),
                                description=f"Conflicting {analyte} criteria: {conflict}",
                            )
                        )

        return contradictions

    def _check_lab_conflict(
        self, lab_a: Dict[str, Any], lab_b: Dict[str, Any]
    ) -> Optional[str]:
        """Check if two lab criteria conflict."""
        op_a, val_a = lab_a.get("operator"), lab_a.get("value")
        op_b, val_b = lab_b.get("operator"), lab_b.get("value")

        if None in (op_a, val_a, op_b, val_b):
            return None

        # After None check, assert for type narrowing
        assert val_a is not None and val_b is not None

        # Check for impossible combinations
        # e.g., >= 30 AND <= 20 is impossible
        if op_a == ">=" and op_b == "<=" and val_a > val_b:
            return f">= {val_a} conflicts with <= {val_b}"
        if op_a == "<=" and op_b == ">=" and val_a < val_b:
            return f"<= {val_a} conflicts with >= {val_b}"
        if op_a == ">" and op_b == "<" and val_a >= val_b:
            return f"> {val_a} conflicts with < {val_b}"
        if op_a == "<" and op_b == ">" and val_a <= val_b:
            return f"< {val_a} conflicts with > {val_b}"

        return None

    def detect_severity_contradictions(
        self, criteria: List[Dict[str, Any]]
    ) -> List[Contradiction]:
        """
        Detect contradictions in severity grade criteria.

        Args:
            criteria: List of criteria with severity_grade fields

        Returns:
            List of detected contradictions
        """
        contradictions = []
        severity_criteria = [c for c in criteria if c.get("severity_grade")]

        # Group by grade type
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for crit in severity_criteria:
            grade = crit["severity_grade"]
            grade_type = grade.get("grade_type", "")
            if grade_type not in by_type:
                by_type[grade_type] = []
            by_type[grade_type].append(crit)

        # Check each type group for conflicts
        for grade_type, group in by_type.items():
            if len(group) < 2:
                continue

            for i, crit_a in enumerate(group):
                for crit_b in group[i + 1:]:
                    grade_a = crit_a["severity_grade"]
                    grade_b = crit_b["severity_grade"]

                    # Check for non-overlapping ranges
                    min_a = grade_a.get("min_value") or grade_a.get("numeric_value", 0)
                    max_a = grade_a.get("max_value") or grade_a.get("numeric_value", 10)
                    min_b = grade_b.get("min_value") or grade_b.get("numeric_value", 0)
                    max_b = grade_b.get("max_value") or grade_b.get("numeric_value", 10)

                    if max_a < min_b or max_b < min_a:
                        contradictions.append(
                            Contradiction(
                                type=ContradictionType.SEVERITY_CONFLICT,
                                criterion_a_text=crit_a.get("text", ""),
                                criterion_b_text=crit_b.get("text", ""),
                                description=f"Non-overlapping {grade_type} ranges",
                            )
                        )

        return contradictions

    def detect_all(self, criteria: List[Dict[str, Any]]) -> List[Contradiction]:
        """Detect all types of contradictions."""
        contradictions = []
        contradictions.extend(self.detect_lab_contradictions(criteria))
        contradictions.extend(self.detect_severity_contradictions(criteria))
        return contradictions
