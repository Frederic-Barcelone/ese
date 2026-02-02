# corpus_metadata/B_parsing/B05_section_detector.py
"""
Layout-aware section detection for clinical document parsing.

This module provides reusable section detection for all entity extractors using
multiple signals: block metadata from PDF parsing, layout heuristics (ALL CAPS,
numbered sections, bold text), and pattern matching against known section headers
(Methods, Results, Eligibility Criteria, etc.).

Key Components:
    - SectionDetector: Main class for detecting section boundaries with multi-signal fusion
    - SectionInfo: Detected section with name, confidence, and triggering signals
    - SECTION_PATTERNS: Dict of compiled regex patterns by section type
    - detect_section: Simple function to detect section from text
    - get_section_detector: Get singleton detector instance

Example:
    >>> from B_parsing.B05_section_detector import SectionDetector
    >>> detector = SectionDetector()
    >>> for block in doc.iter_linear_blocks():
    ...     section = detector.detect(block.text, block)
    ...     if section:
    ...         print(f"Section: {section.name} (confidence: {section.confidence})")

Dependencies:
    - B_parsing.B02_doc_graph: ContentRole for block role checking
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from B_parsing.B02_doc_graph import ContentRole


# =============================================================================
# SECTION PATTERN DEFINITIONS
# =============================================================================

# Base section patterns used across extractors
SECTION_PATTERNS: Dict[str, List[str]] = {
    # Eligibility / Study Population
    "eligibility": [
        r"eligibility\s*criteria",
        r"inclusion\s*(?:and\s*)?exclusion\s*criteria",
        r"patient\s*selection",
        r"study\s*population",
        r"participants",
        r"patients\s*and\s*methods",
        r"subject\s*eligibility",
    ],
    # Methods / Study Design
    "methods": [
        r"methods",
        r"study\s*design",
        r"trial\s*design",
        r"materials?\s*and\s*methods",
        r"trial\s*design\s*and\s*oversight",
        r"study\s*overview",
    ],
    # Background / Epidemiology
    "epidemiology": [
        r"epidemiology",
        r"prevalence",
        r"incidence",
        r"demographics",
        r"background",
        r"introduction",
        r"disease\s*(?:background|overview)",
        r"epidemiology\s*and\s*natural\s*history",
    ],
    # Endpoints / Outcomes
    "endpoints": [
        r"endpoints?",
        r"outcomes?",
        r"efficacy",
        r"primary\s*(?:end)?points?",
        r"secondary\s*(?:end)?points?",
        r"efficacy\s*assessments?",
        r"objectives?\s*and\s*endpoints?",
        r"study\s*objectives?",
    ],
    # Patient Journey / Procedures
    "patient_journey": [
        r"study\s*procedures?",
        r"treatment\s*period",
        r"follow[\-\s]?up",
        r"screening",
        r"study\s*visits?",
        r"schedule\s*of\s*(?:assessments?|events?)",
        r"intervention\s*and\s*follow[\-\s]?up",
    ],
    # Results
    "results": [
        r"results",
        r"findings",
        r"efficacy\s*results",
        r"safety\s*results",
        r"outcomes?",
    ],
    # Discussion / Conclusions
    "discussion": [
        r"discussion",
        r"conclusions?",
        r"summary",
        r"limitations?",
    ],
    # References
    "references": [
        r"references",
        r"bibliography",
        r"citations?",
    ],
    # Abstract
    "abstract": [
        r"abstract",
        r"summary",
    ],
}

# Compiled patterns cache
_COMPILED_PATTERNS: Dict[str, List[re.Pattern]] = {}


def _get_compiled_patterns() -> Dict[str, List[re.Pattern]]:
    """Get or build compiled regex patterns."""
    global _COMPILED_PATTERNS
    if not _COMPILED_PATTERNS:
        _COMPILED_PATTERNS = {
            section: [re.compile(p, re.IGNORECASE) for p in patterns]
            for section, patterns in SECTION_PATTERNS.items()
        }
    return _COMPILED_PATTERNS


# =============================================================================
# SECTION DETECTOR CLASS
# =============================================================================


@dataclass
class SectionInfo:
    """Information about a detected section."""
    name: str  # Section name (e.g., "methods", "eligibility")
    start_text: str  # Text that triggered section detection
    confidence: float = 1.0  # Detection confidence
    signals: List[str] = field(default_factory=list)  # What triggered detection


class SectionDetector:
    """
    Layout-aware section detector for document parsing.

    Uses multiple signals to detect section boundaries:
    1. Block role metadata (from PDF parser)
    2. Layout heuristics (ALL CAPS, short text, colons, bold)
    3. Pattern matching against known section headers

    Example usage:
        detector = SectionDetector()
        for block in doc.iter_linear_blocks():
            section = detector.detect(block.text, block)
            if section:
                current_section = section.name
    """

    def __init__(
        self,
        custom_patterns: Optional[Dict[str, List[str]]] = None,
        max_header_length: int = 100,
    ):
        """
        Initialize section detector.

        Args:
            custom_patterns: Additional section patterns to merge
            max_header_length: Maximum text length to consider as header
        """
        self.max_header_length = max_header_length
        self.patterns = _get_compiled_patterns()

        # Merge custom patterns if provided
        if custom_patterns:
            for section, patterns in custom_patterns.items():
                compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
                if section in self.patterns:
                    self.patterns[section].extend(compiled)
                else:
                    self.patterns[section] = compiled

    def detect(
        self,
        text: str,
        block: Optional[Any] = None,
    ) -> Optional[SectionInfo]:
        """
        Detect if text represents a section header.

        Args:
            text: Text to analyze
            block: Optional TextBlock with metadata

        Returns:
            SectionInfo if section detected, None otherwise
        """
        if not text or not text.strip():
            return None

        text = text.strip()
        text_lower = text.lower()
        signals: List[str] = []

        # Signal 1: Check block role metadata
        if block is not None:
            if hasattr(block, 'role'):
                if block.role == ContentRole.SECTION_HEADER:
                    signals.append("block_role_header")
                    section = self._match_pattern(text_lower)
                    if section:
                        return SectionInfo(
                            name=section,
                            start_text=text,
                            confidence=0.95,
                            signals=signals,
                        )

        # Signal 2: Layout heuristics for likely headers
        is_likely_header = self._is_likely_header(text, block)
        if is_likely_header:
            signals.extend(is_likely_header)

        # Only check patterns if text looks like a header or is short
        if signals or len(text) < self.max_header_length:
            section = self._match_pattern(text_lower)
            if section:
                signals.append(f"pattern_match:{section}")
                confidence = 0.9 if signals else 0.7
                return SectionInfo(
                    name=section,
                    start_text=text,
                    confidence=confidence,
                    signals=signals,
                )

        return None

    def _is_likely_header(
        self,
        text: str,
        block: Optional[Any] = None,
    ) -> List[str]:
        """
        Check layout heuristics for header detection.

        Returns list of signals that indicate header-like text.
        """
        signals: list[str] = []

        # Too long to be a header
        if len(text) > self.max_header_length:
            return signals

        # ALL CAPS (common for section headers)
        if text.isupper() and len(text) > 3:
            signals.append("all_caps")

        # Ends with colon (often indicates header)
        if text.rstrip().endswith(':'):
            signals.append("ends_with_colon")

        # Title case with limited words
        words = text.split()
        if len(words) <= 5 and text.istitle():
            signals.append("title_case_short")

        # Check block metadata for bold/font info
        if block is not None:
            if hasattr(block, 'is_bold') and block.is_bold:
                signals.append("bold_text")
            if hasattr(block, 'font_size') and hasattr(block, 'avg_font_size'):
                if block.font_size > block.avg_font_size * 1.2:
                    signals.append("larger_font")

        # Numbered section (e.g., "1. Methods", "2.1 Study Design")
        if re.match(r'^[\d.]+\s+[A-Z]', text):
            signals.append("numbered_section")

        return signals

    def _match_pattern(self, text_lower: str) -> Optional[str]:
        """Match text against section patterns."""
        for section_name, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    return section_name
        return None

    def get_expected_sections(self, field_type: str) -> List[str]:
        """
        Get expected sections for a given field type.

        Useful for confidence scoring based on section context.
        """
        # Map field types to expected sections
        FIELD_SECTION_MAP = {
            "eligibility": ["eligibility", "methods"],
            "epidemiology": ["epidemiology", "abstract", "results"],
            "endpoints": ["endpoints", "methods"],
            "patient_journey": ["patient_journey", "methods"],
            "disease": ["abstract", "methods", "results", "discussion"],
            "drug": ["abstract", "methods", "results"],
            "abbreviation": ["methods", "abstract"],
        }
        return FIELD_SECTION_MAP.get(field_type, [])


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Singleton instance for simple usage
_DEFAULT_DETECTOR: Optional[SectionDetector] = None


def get_section_detector() -> SectionDetector:
    """Get default section detector instance."""
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = SectionDetector()
    return _DEFAULT_DETECTOR


def detect_section(text: str, block: Optional[Any] = None) -> Optional[str]:
    """
    Simple function to detect section from text.

    Returns section name or None.
    """
    detector = get_section_detector()
    result = detector.detect(text, block)
    return result.name if result else None
