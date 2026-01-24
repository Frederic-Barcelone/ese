# corpus_metadata/Z_utils/Z03_text_normalization.py
"""
Text Normalization Utilities.

Shared functions for normalizing text extracted from PDFs:
- Whitespace normalization
- Dehyphenation of line-break artifacts
- Long form normalization for abbreviations

DRY principle: These functions are used across multiple generators.
"""

from __future__ import annotations

import re


def clean_whitespace(s: str) -> str:
    """
    Collapse all whitespace to single spaces and strip.

    Args:
        s: Input string

    Returns:
        Normalized string with single spaces
    """
    return " ".join((s or "").split()).strip()


def dehyphenate_long_form(lf: str) -> str:
    """
    Remove line-break hyphens from long forms.

    PDF extraction often produces hyphenated words where lines break:
    - "gastroin-testinal" -> "gastrointestinal"
    - "Vasculi-tis Study Group" -> "Vasculitis Study Group"

    Preserves intentional hyphens in compound words:
    - "anti-inflammatory" stays as-is
    - "non-profit" stays as-is

    Args:
        lf: Long form string potentially with line-break hyphens

    Returns:
        Dehyphenated string
    """
    if not lf:
        return lf

    lf = clean_whitespace(lf)

    # Pattern: hyphen + space + lowercase continuation
    # This catches line-break hyphenation where space remains after normalization
    lf = re.sub(r"-\s+([a-z])", r"\1", lf)

    # Common prefixes that form valid hyphenated compounds - preserve these
    compound_prefixes = (
        "anti", "non", "pre", "post", "re", "co", "sub", "inter",
        "intra", "extra", "multi", "semi", "self", "cross", "over",
        "under", "out", "well", "ill", "full", "half", "pro", "counter",
    )

    def maybe_dehyphenate(match: re.Match) -> str:
        """Decide whether to remove a hyphen."""
        before = match.group(1)
        after = match.group(2)

        # Check if this looks like a valid compound
        for prefix in compound_prefixes:
            if before.lower().endswith(prefix):
                return match.group(0)  # Keep hyphen

        # Otherwise, likely a line-break artifact - remove hyphen
        return before + after

    # Match: word-chars + hyphen + lowercase continuation
    lf = re.sub(r"(\w)-([a-z]{2,})", maybe_dehyphenate, lf)

    return lf


def truncate_at_clause_breaks(text: str) -> str:
    """
    Truncate text at obvious clause/sentence breaks.

    Stops at:
    - Punctuation marks (. ; : newlines)
    - Relative clause starters (which, that, who, etc.)
    - Verb phrases indicating narrative

    Used to prevent long forms from including unrelated text.

    Args:
        text: Input text

    Returns:
        Truncated text
    """
    t = (text or "").strip()

    # Stop at punctuation
    t = re.split(r"[.;:\n\r\)\]]", t, maxsplit=1)[0].strip()

    # Stop at relative clause starters
    t = re.split(
        r"\b(which|that|who|where|when|while|whose)\b",
        t,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    # Stop at verb phrases that indicate narrative
    t = re.split(
        r"\b(was|were|is|are|has|have|had|being|been|can|could|would|should|may|might)\s+",
        t,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    # Remove trailing connectors
    t = re.sub(r"\s+(and|or|the|a|an)\s*$", "", t, flags=re.IGNORECASE).strip()

    return t


def normalize_long_form(lf: str) -> str:
    """
    Full normalization for abbreviation long forms.

    Applies:
    1. Whitespace normalization
    2. Clause break truncation (to remove appended unrelated text)
    3. Dehyphenation of line-break artifacts

    Args:
        lf: Long form string

    Returns:
        Normalized long form
    """
    if not lf:
        return lf

    original = lf

    # First clean whitespace
    lf = clean_whitespace(lf)

    # Truncate at clause breaks
    truncated = truncate_at_clause_breaks(lf)

    # Only use truncated if it's not too short
    if truncated and len(truncated) >= 3:
        lf = truncated

    # Dehyphenate
    lf = dehyphenate_long_form(lf)

    # Ensure we return something meaningful
    if not lf or len(lf) < 2:
        return dehyphenate_long_form(clean_whitespace(original))

    return lf
