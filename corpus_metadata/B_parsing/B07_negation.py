# corpus_metadata/corpus_metadata/B_parsing/B07_negation.py
"""
Negation detection utilities for extraction pipelines.

Provides reusable negation detection for all extractors (C06-C08+).
Critical for eligibility criteria ("no prior treatment") and disease mentions
("no evidence of cancer").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set


# =============================================================================
# NEGATION CUE DEFINITIONS
# =============================================================================

# Direct negation words
NEGATION_CUES: Set[str] = {
    "no", "not", "without", "excluding", "except", "unless",
    "free of", "absence of", "lack of", "negative for", "non",
    "never", "none", "neither", "cannot", "unable",
    "denied", "denies", "deny", "rule out", "ruled out",
    "no evidence", "no history", "no sign", "no signs",
}

# Negation prefixes (attached to words)
NEGATION_PREFIXES: List[str] = [
    "non-", "non", "un-", "un", "in-", "im-", "a-",
]

# Exception/conditional phrases (may flip interpretation)
EXCEPTION_CUES: Set[str] = {
    "except", "unless", "other than", "apart from", "excluding",
    "with the exception", "provided that", "but not", "save for",
}

# Double negation patterns (negation of negation = positive)
DOUBLE_NEGATION_PATTERNS: List[str] = [
    r"no(?:t)?\s+(?:without|lacking|absent)",
    r"cannot\s+(?:be\s+)?ruled\s+out",
    r"not\s+excluded",
]


# =============================================================================
# NEGATION RESULT DATACLASS
# =============================================================================


@dataclass
class NegationResult:
    """Result of negation detection."""
    is_negated: bool
    negation_cue: Optional[str] = None  # The cue that triggered negation
    cue_position: Optional[int] = None  # Character position of cue
    has_exception: bool = False  # Contains exception phrase
    is_double_negation: bool = False  # Double negation detected


# =============================================================================
# NEGATION DETECTOR CLASS
# =============================================================================


class NegationDetector:
    """
    Detects negation in text for extraction validation.

    Handles:
    - Direct negation cues ("no", "not", "without")
    - Prefix negation ("non-small cell", "inability")
    - Exception clauses ("except", "unless")
    - Double negation ("cannot be ruled out")

    Example usage:
        detector = NegationDetector()
        result = detector.detect("No prior treatment with immunotherapy", 15)
        # result.is_negated = True, result.negation_cue = "no"
    """

    def __init__(
        self,
        window_size: int = 50,
        custom_cues: Optional[Set[str]] = None,
    ):
        """
        Initialize negation detector.

        Args:
            window_size: Character window to search before match
            custom_cues: Additional negation cues to include
        """
        self.window_size = window_size
        self.negation_cues = NEGATION_CUES.copy()
        if custom_cues:
            self.negation_cues.update(custom_cues)

        # Compile double negation patterns
        self.double_neg_patterns = [
            re.compile(p, re.IGNORECASE) for p in DOUBLE_NEGATION_PATTERNS
        ]

    def detect(
        self,
        text: str,
        match_start: Optional[int] = None,
        match_text: Optional[str] = None,
    ) -> NegationResult:
        """
        Detect negation in text.

        Args:
            text: Full text to analyze
            match_start: Start position of the matched phrase (for context window)
            match_text: The matched phrase itself (for prefix negation check)

        Returns:
            NegationResult with detection details
        """
        text_lower = text.lower()

        # Check for double negation first (overrides simple negation)
        for pattern in self.double_neg_patterns:
            if pattern.search(text_lower):
                return NegationResult(
                    is_negated=False,  # Double negation = positive
                    is_double_negation=True,
                )

        # Check exception clauses
        has_exception = self._has_exception(text_lower)

        # Check prefix negation on match text
        if match_text:
            prefix_neg = self._check_prefix_negation(match_text.lower())
            if prefix_neg:
                return NegationResult(
                    is_negated=True,
                    negation_cue=prefix_neg,
                    has_exception=has_exception,
                )

        # Check context window for negation cues
        if match_start is not None:
            window_start = max(0, match_start - self.window_size)
            prefix = text_lower[window_start:match_start]
        else:
            prefix = text_lower[:self.window_size]

        # Look for negation cues in prefix
        for cue in self.negation_cues:
            if cue in prefix:
                # Find position
                pos = prefix.rfind(cue)
                return NegationResult(
                    is_negated=True,
                    negation_cue=cue,
                    cue_position=window_start + pos if match_start else pos,
                    has_exception=has_exception,
                )

        return NegationResult(
            is_negated=False,
            has_exception=has_exception,
        )

    def detect_in_sentence(self, sentence: str) -> NegationResult:
        """
        Detect negation anywhere in a sentence.

        Simpler interface when you don't have a specific match position.
        """
        return self.detect(sentence, match_start=len(sentence) // 2)

    def _check_prefix_negation(self, text: str) -> Optional[str]:
        """Check if text starts with a negation prefix."""
        for prefix in NEGATION_PREFIXES:
            if text.startswith(prefix):
                return prefix
        return None

    def _has_exception(self, text: str) -> bool:
        """Check if text contains exception cues."""
        return any(cue in text for cue in EXCEPTION_CUES)

    def is_negated_context(
        self,
        text: str,
        match_start: int,
        match_end: int,
    ) -> bool:
        """
        Simple boolean check for negation.

        Convenience method for simple use cases.
        """
        result = self.detect(text, match_start)
        return result.is_negated and not result.is_double_negation


# =============================================================================
# ASSERTION CLASSIFICATION (Extended)
# =============================================================================


class AssertionType:
    """Assertion types for clinical text."""
    PRESENT = "present"
    ABSENT = "absent"
    POSSIBLE = "possible"
    CONDITIONAL = "conditional"
    HISTORICAL = "historical"
    FAMILY = "family"


# Historical/temporal context cues
HISTORICAL_CUES: Set[str] = {
    "history of", "prior", "previous", "past", "former",
    "had", "was diagnosed", "previously treated",
}

# Family history cues
FAMILY_CUES: Set[str] = {
    "family history", "familial", "hereditary", "mother",
    "father", "sibling", "relative", "genetic",
}

# Possibility/uncertainty cues
POSSIBILITY_CUES: Set[str] = {
    "possible", "probable", "suspected", "likely", "suggest",
    "consistent with", "consider", "differential",
}


class AssertionClassifier:
    """
    Extended assertion classifier for clinical text.

    Beyond simple negation, classifies assertions as:
    - Present (affirmed)
    - Absent (negated)
    - Possible (uncertain)
    - Conditional (depends on something)
    - Historical (past)
    - Family (family history, not patient)
    """

    def __init__(self):
        self.negation_detector = NegationDetector()

    def classify(
        self,
        text: str,
        match_start: Optional[int] = None,
    ) -> str:
        """
        Classify the assertion type of a mention.

        Returns AssertionType constant.
        """
        text_lower = text.lower()

        # Check family history first (overrides others)
        if any(cue in text_lower for cue in FAMILY_CUES):
            return AssertionType.FAMILY

        # Check historical
        if any(cue in text_lower for cue in HISTORICAL_CUES):
            return AssertionType.HISTORICAL

        # Check possibility/uncertainty
        if any(cue in text_lower for cue in POSSIBILITY_CUES):
            return AssertionType.POSSIBLE

        # Check negation
        neg_result = self.negation_detector.detect(text, match_start)
        if neg_result.is_negated and not neg_result.is_double_negation:
            if neg_result.has_exception:
                return AssertionType.CONDITIONAL
            return AssertionType.ABSENT

        return AssertionType.PRESENT


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Singleton instances
_DEFAULT_DETECTOR: Optional[NegationDetector] = None
_DEFAULT_CLASSIFIER: Optional[AssertionClassifier] = None


def get_negation_detector() -> NegationDetector:
    """Get default negation detector instance."""
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = NegationDetector()
    return _DEFAULT_DETECTOR


def get_assertion_classifier() -> AssertionClassifier:
    """Get default assertion classifier instance."""
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        _DEFAULT_CLASSIFIER = AssertionClassifier()
    return _DEFAULT_CLASSIFIER


def is_negated(text: str, match_start: Optional[int] = None) -> bool:
    """
    Simple function to check if text/match is negated.

    Returns True if negated, False otherwise.
    """
    detector = get_negation_detector()
    result = detector.detect(text, match_start)
    return result.is_negated and not result.is_double_negation


def classify_assertion(text: str, match_start: Optional[int] = None) -> str:
    """
    Simple function to classify assertion type.

    Returns AssertionType constant.
    """
    classifier = get_assertion_classifier()
    return classifier.classify(text, match_start)
