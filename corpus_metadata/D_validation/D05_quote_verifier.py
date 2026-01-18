# corpus_metadata/D_validation/D05_quote_verifier.py
"""
Anti-hallucination verification for LLM-based extraction.

Provides quote and numerical value verification to detect hallucinated content.

Contains:
  - QuoteVerifier: Verifies exact/fuzzy quote matches against source text
  - NumericalVerifier: Verifies numerical values exist in source text
  - VerificationResult: Result container with verification status and details
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


# =============================================================================
# VERIFICATION RESULTS
# =============================================================================


@dataclass
class QuoteVerificationResult:
    """Result of quote verification."""

    verified: bool
    """Whether the quote was found in the context."""

    position: Optional[Tuple[int, int]] = None
    """(start, end) character offsets if found."""

    match_ratio: float = 0.0
    """Similarity ratio (1.0 = exact match, 0.0 = no match)."""

    matched_text: Optional[str] = None
    """The actual text matched in the context (for fuzzy matches)."""

    is_exact_match: bool = False
    """True if this was an exact substring match."""


@dataclass
class NumericalVerificationResult:
    """Result of numerical value verification."""

    verified: bool
    """Whether the number was found in the context."""

    positions: List[Tuple[int, int]] = field(default_factory=list)
    """List of (start, end) positions where the number was found."""

    matched_formats: List[str] = field(default_factory=list)
    """The actual formats found (e.g., "1,234", "1234", "1.2k")."""


@dataclass
class FieldVerificationResult:
    """Aggregated verification result for a single extracted field."""

    field_name: str
    """Name of the field being verified."""

    quote_result: Optional[QuoteVerificationResult] = None
    """Quote verification result if quote was provided."""

    numerical_results: Dict[str, NumericalVerificationResult] = field(default_factory=dict)
    """Numerical verification results keyed by value."""

    @property
    def is_verified(self) -> bool:
        """True if all provided evidence is verified."""
        if self.quote_result and not self.quote_result.verified:
            return False
        for nr in self.numerical_results.values():
            if not nr.verified:
                return False
        return True

    @property
    def confidence_penalty(self) -> float:
        """
        Calculate confidence penalty based on verification status.

        Returns:
            Multiplier between 0.0 and 1.0 (1.0 = no penalty).
        """
        penalty = 1.0

        # Unverified quote: 0.5 penalty
        if self.quote_result and not self.quote_result.verified:
            penalty *= 0.5

        # Unverified numbers: 0.7 penalty per unverified number
        for nr in self.numerical_results.values():
            if not nr.verified:
                penalty *= 0.7

        return penalty


# =============================================================================
# QUOTE VERIFIER
# =============================================================================


class QuoteVerifier:
    """
    Verifies that LLM-provided quotes actually exist in the source document.

    Uses exact match first, then falls back to fuzzy matching for minor
    variations (whitespace normalization, case differences, etc.).
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.90,
        max_fuzzy_window: int = 500,
    ):
        """
        Initialize quote verifier.

        Args:
            fuzzy_threshold: Minimum similarity ratio for fuzzy match (0.0-1.0).
            max_fuzzy_window: Maximum characters to search for fuzzy matches.
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.max_fuzzy_window = max_fuzzy_window

    def verify(
        self,
        quote: str,
        context: str,
        case_sensitive: bool = False,
    ) -> QuoteVerificationResult:
        """
        Verify that a quote exists in the context.

        First attempts exact match, then fuzzy match if exact fails.

        Args:
            quote: The quoted text to verify.
            context: The source document text to search.
            case_sensitive: Whether to use case-sensitive matching.

        Returns:
            QuoteVerificationResult with verification status and details.
        """
        if not quote or not context:
            return QuoteVerificationResult(verified=False, match_ratio=0.0)

        # Normalize whitespace in both quote and context
        normalized_quote = self._normalize_whitespace(quote)
        normalized_context = self._normalize_whitespace(context)

        # Check after normalization (handles whitespace-only quotes)
        if not normalized_quote:
            return QuoteVerificationResult(verified=False, match_ratio=0.0)

        # Try exact match first
        exact_result = self._try_exact_match(
            normalized_quote, normalized_context, case_sensitive
        )
        if exact_result.verified:
            return exact_result

        # Try fuzzy match
        return self._try_fuzzy_match(
            normalized_quote, normalized_context, case_sensitive
        )

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace: collapse multiple spaces, strip, normalize newlines."""
        # Replace various whitespace chars with single space
        text = re.sub(r'[\s\u00a0\u2003\u2002\u2009]+', ' ', text)
        return text.strip()

    def _try_exact_match(
        self,
        quote: str,
        context: str,
        case_sensitive: bool,
    ) -> QuoteVerificationResult:
        """Attempt exact substring match."""
        search_quote = quote if case_sensitive else quote.lower()
        search_context = context if case_sensitive else context.lower()

        pos = search_context.find(search_quote)
        if pos != -1:
            return QuoteVerificationResult(
                verified=True,
                position=(pos, pos + len(quote)),
                match_ratio=1.0,
                matched_text=context[pos : pos + len(quote)],
                is_exact_match=True,
            )

        return QuoteVerificationResult(verified=False, match_ratio=0.0)

    def _try_fuzzy_match(
        self,
        quote: str,
        context: str,
        case_sensitive: bool,
    ) -> QuoteVerificationResult:
        """
        Attempt fuzzy match using sliding window approach.

        Searches for the best matching substring of similar length to the quote.
        """
        if len(quote) < 10:
            # Too short for meaningful fuzzy matching
            return QuoteVerificationResult(verified=False, match_ratio=0.0)

        search_quote = quote if case_sensitive else quote.lower()
        search_context = context if case_sensitive else context.lower()

        quote_len = len(quote)
        best_ratio = 0.0
        best_pos = -1
        best_match = None

        # Window sizes to try (slightly smaller and larger than quote)
        window_sizes = [
            quote_len,
            int(quote_len * 0.9),
            int(quote_len * 1.1),
        ]

        # Limit search to avoid O(n*m) complexity
        max_search = min(len(search_context), self.max_fuzzy_window * 10)
        search_context = search_context[:max_search]

        for window_size in window_sizes:
            if window_size < 10:
                continue

            step = max(1, window_size // 4)  # Slide by 25% of window

            for i in range(0, len(search_context) - window_size + 1, step):
                window = search_context[i : i + window_size]

                # Quick filter: check if key words overlap
                if not self._quick_overlap_check(search_quote, window):
                    continue

                ratio = SequenceMatcher(None, search_quote, window).ratio()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pos = i
                    best_match = context[i : i + window_size]

        if best_ratio >= self.fuzzy_threshold:
            return QuoteVerificationResult(
                verified=True,
                position=(best_pos, best_pos + len(best_match)) if best_match else None,
                match_ratio=best_ratio,
                matched_text=best_match,
                is_exact_match=False,
            )

        return QuoteVerificationResult(
            verified=False,
            match_ratio=best_ratio,
        )

    def _quick_overlap_check(self, s1: str, s2: str) -> bool:
        """Quick check for word overlap to filter obviously non-matching windows."""
        words1 = set(s1.split()[:5])  # First 5 words
        words2 = set(s2.split())
        return len(words1 & words2) >= 2


# =============================================================================
# NUMERICAL VERIFIER
# =============================================================================


class NumericalVerifier:
    """
    Verifies that numerical values extracted by LLM exist in the source document.

    Handles various number formats:
    - Plain integers: 1234
    - Comma-separated: 1,234
    - Decimal: 1234.5
    - With units: 1234 mg, 12.5%
    - Ranges: 1234-5678
    """

    def __init__(self):
        # Pattern for finding numbers in text
        self._number_pattern = re.compile(
            r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\b'
        )

    def verify(
        self,
        value: int | float | str,
        context: str,
        tolerance: float = 0.0,
    ) -> NumericalVerificationResult:
        """
        Verify that a numerical value exists in the context.

        Args:
            value: The numerical value to verify (can be string, will be converted).
            context: The source document text to search.
            tolerance: Relative tolerance for matching (0.0 = exact, 0.1 = 10%).

        Returns:
            NumericalVerificationResult with verification status and details.
        """
        if context is None or value is None:
            return NumericalVerificationResult(verified=False)

        # Convert string to number if needed
        if isinstance(value, str):
            try:
                value = float(value.replace(',', ''))
            except (ValueError, AttributeError):
                return NumericalVerificationResult(verified=False)

        positions: List[Tuple[int, int]] = []
        matched_formats: List[str] = []

        # Find all numbers in context
        for match in self._number_pattern.finditer(context):
            matched_str = match.group(1)
            try:
                # Parse the matched number (remove commas)
                parsed_value = float(matched_str.replace(',', ''))

                # Check if it matches our target value
                if self._values_match(value, parsed_value, tolerance):
                    positions.append((match.start(), match.end()))
                    matched_formats.append(matched_str)
            except ValueError:
                continue

        return NumericalVerificationResult(
            verified=len(positions) > 0,
            positions=positions,
            matched_formats=matched_formats,
        )

    def _values_match(
        self,
        expected: int | float,
        actual: float,
        tolerance: float,
    ) -> bool:
        """Check if two values match within tolerance."""
        if tolerance == 0.0:
            # For integers, require exact match
            if isinstance(expected, int):
                return actual == expected or int(actual) == expected
            # For floats, allow small floating point error
            return abs(expected - actual) < 1e-9

        # Relative tolerance
        if expected == 0:
            return abs(actual) < tolerance
        return abs(expected - actual) / abs(expected) <= tolerance

    def verify_multiple(
        self,
        values: Dict[str, int | float],
        context: str,
        tolerance: float = 0.0,
    ) -> Dict[str, NumericalVerificationResult]:
        """
        Verify multiple numerical values at once.

        Args:
            values: Dict mapping field names to numerical values.
            context: The source document text to search.
            tolerance: Relative tolerance for matching.

        Returns:
            Dict mapping field names to verification results.
        """
        results = {}
        for field_name, value in values.items():
            if value is not None:
                results[field_name] = self.verify(value, context, tolerance)
        return results


# =============================================================================
# EXTRACTION VERIFIER (Composite)
# =============================================================================


class ExtractionVerifier:
    """
    Composite verifier for complete extraction verification.

    Combines quote and numerical verification to validate entire
    LLM extraction results.
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.90,
        numerical_tolerance: float = 0.0,
    ):
        self.quote_verifier = QuoteVerifier(fuzzy_threshold=fuzzy_threshold)
        self.numerical_verifier = NumericalVerifier()
        self.numerical_tolerance = numerical_tolerance

    def verify_field(
        self,
        field_name: str,
        context: str,
        quote: Optional[str] = None,
        numerical_values: Optional[Dict[str, int | float]] = None,
    ) -> FieldVerificationResult:
        """
        Verify all evidence for a single extracted field.

        Args:
            field_name: Name of the field being verified.
            context: Source document text.
            quote: Optional quote to verify.
            numerical_values: Optional dict of numerical values to verify.

        Returns:
            FieldVerificationResult with all verification details.
        """
        result = FieldVerificationResult(field_name=field_name)

        if quote:
            result.quote_result = self.quote_verifier.verify(quote, context)

        if numerical_values:
            result.numerical_results = self.numerical_verifier.verify_multiple(
                numerical_values, context, self.numerical_tolerance
            )

        return result

    def verify_extraction(
        self,
        extraction: Dict,
        context: str,
        quote_fields: Optional[List[str]] = None,
        numerical_fields: Optional[List[str]] = None,
    ) -> Dict[str, FieldVerificationResult]:
        """
        Verify a complete extraction result.

        Args:
            extraction: The extraction dict from LLM.
            context: Source document text.
            quote_fields: List of field names that contain quotes to verify.
            numerical_fields: List of field names that contain numbers to verify.

        Returns:
            Dict mapping field names to their verification results.
        """
        results: Dict[str, FieldVerificationResult] = {}
        quote_fields = quote_fields or []
        numerical_fields = numerical_fields or []

        # Verify quote fields
        for field in quote_fields:
            if field in extraction and extraction[field]:
                quote = extraction[field]
                if isinstance(quote, str):
                    results[field] = self.verify_field(field, context, quote=quote)

        # Verify numerical fields
        for field in numerical_fields:
            if field in extraction and extraction[field] is not None:
                value = extraction[field]
                if isinstance(value, (int, float)):
                    results[field] = self.verify_field(
                        field, context, numerical_values={field: value}
                    )

        return results

    def calculate_confidence_penalty(
        self,
        verification_results: Dict[str, FieldVerificationResult],
    ) -> float:
        """
        Calculate overall confidence penalty from verification results.

        Args:
            verification_results: Results from verify_extraction.

        Returns:
            Multiplier between 0.0 and 1.0 (1.0 = no penalty).
        """
        if not verification_results:
            return 1.0

        total_penalty = 1.0
        for result in verification_results.values():
            total_penalty *= result.confidence_penalty

        return total_penalty


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def verify_quote(quote: str, context: str, threshold: float = 0.90) -> bool:
    """
    Simple function to verify a quote exists in context.

    Args:
        quote: The quoted text to verify.
        context: The source document text.
        threshold: Minimum similarity for fuzzy match.

    Returns:
        True if quote is verified, False otherwise.
    """
    verifier = QuoteVerifier(fuzzy_threshold=threshold)
    result = verifier.verify(quote, context)
    return result.verified


def verify_number(value: int | float, context: str) -> bool:
    """
    Simple function to verify a number exists in context.

    Args:
        value: The numerical value to verify.
        context: The source document text.

    Returns:
        True if number is found, False otherwise.
    """
    verifier = NumericalVerifier()
    result = verifier.verify(value, context)
    return result.verified
