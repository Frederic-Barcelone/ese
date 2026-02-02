"""
Inline abbreviation definition detection using regex patterns.

This module extracts inline abbreviation definitions that complement scispacy's
Schwartz-Hearst detector. Catches additional patterns for mixed-case abbreviations,
reversed definitions, and alternative separators.

Key Components:
    - InlineDefinitionDetectorMixin: Mixin class for inline definition detection
    - Pattern categories:
        - "Long Form (ABBREV)" - standard with mixed-case abbreviations
        - "ABBREV (long form)" - reversed pattern
        - "ABBREV, the/or/i.e. long form" - comma-separated definitions
        - "ABBREV = long form" - equals-separated definitions

Example:
    >>> class MyExtractor(InlineDefinitionDetectorMixin):
    ...     pass
    >>> extractor = MyExtractor()
    >>> definitions = extractor._extract_inline_definitions("LoE, the Level of Evidence")
    >>> definitions
    [('LoE', 'Level of Evidence', 0, 28)]

Dependencies:
    - re: Regular expression matching
"""

from __future__ import annotations

import re
from typing import List, Tuple


class InlineDefinitionDetectorMixin:
    """
    Mixin class providing inline definition detection methods.

    Designed to be mixed into RegexLexiconGenerator to provide
    _extract_inline_definitions and related methods.
    """

    def _extract_inline_definitions(
        self, text: str
    ) -> List[Tuple[str, str, int, int]]:
        """
        Extract inline abbreviation definitions using regex patterns.

        Catches patterns that scispacy's Schwartz-Hearst detector might miss:
        1. "long form (ABBREV)" - standard pattern with mixed-case abbreviations
        2. "ABBREV (long form)" - reversed pattern
        3. "ABBREV, the/or/i.e. long form" - comma-separated definitions
        4. "ABBREV = long form" - equals-separated definitions

        Returns:
            List of (short_form, long_form, start, end) tuples
        """
        results: List[Tuple[str, str, int, int]] = []

        # Pattern 1a: "Long Form (ABBREV)" - title case long form
        # Catches mixed-case abbreviations like LoE, LoA
        # Updated to handle hyphenated words like "Five-Factor Score"
        pattern1a = re.compile(
            r"\b((?:[A-Z][a-z]+(?:-[A-Z]?[a-z]+)?\s+){1,7}[A-Za-z]+)\s*"  # Long form: capitalized words (with optional hyphen)
            r"\(([A-Za-z][A-Za-z0-9/-]{1,9})\)",  # (ABBREV) - mixed case allowed
            re.UNICODE
        )

        for match in pattern1a.finditer(text):
            lf = match.group(1).strip()
            sf = match.group(2).strip()

            # Validate: SF should have at least one uppercase
            if not any(c.isupper() for c in sf):
                continue

            # Validate: SF should look like an abbreviation (not a word)
            if sf.lower() == sf:  # All lowercase
                continue

            # Check if SF could plausibly be an abbreviation of LF
            if self._could_be_abbreviation(sf, lf):
                results.append((sf, lf, match.start(), match.end()))

        # Pattern 1b: "long form (ABBREV)" - lowercase long form (common in clinical text)
        # e.g., "level of agreement (LoA)", "eosinophilic GPA (EGPA)"
        # More restrictive: only capture 1-6 words before the parenthesis
        pattern1b = re.compile(
            r"\b([a-z][a-z\s/-]{3,60})\s*"  # Long form: lowercase words (allow hyphens/slashes)
            r"\(([A-Z][A-Za-z0-9/-]{1,9})\)",  # (ABBREV) - must start with uppercase
            re.UNICODE
        )

        for match in pattern1b.finditer(text):
            lf = match.group(1).strip()
            sf = match.group(2).strip()

            # Clean up long form - remove common lead-in phrases that aren't part of the term
            lead_in_patterns = [
                r"^(?:developed\s+for|known\s+as|called|termed|named|referred\s+to\s+as)\s+",
                r"^(?:including|such\s+as|like|e\.?g\.?)\s+",
                r"^(?:is\s+a|was\s+a|are|were)\s+",
                # NEW: Remove sentence fragments and connector phrases
                r"^.*?\b(?:with|without|and|or|from|between|patients|subjects)\s+",
                r"^.*?\b(?:activity|study|trial|treatment)\s+(?:of|in|for|with)\s+",
            ]
            for pattern in lead_in_patterns:
                lf = re.sub(pattern, "", lf, flags=re.IGNORECASE).strip()

            # NEW: Trim to last N words if still too long (likely captured sentence context)
            lf_words = lf.split()
            if len(lf_words) > 6:
                # Keep only the last 4 words closest to the abbreviation
                lf = " ".join(lf_words[-4:])
                lf_words = lf.split()

            # Validate: long form should have multiple words (or single word for some abbrevs)
            if len(lf_words) < 1:
                continue

            # Skip if LF looks like it could be part of a sentence (too long)
            if len(lf_words) > 6:
                continue

            # NEW: Skip if LF contains obvious sentence fragments
            sentence_indicators = ["patients with", "subjects with", "those with", "activity of"]
            if any(ind in lf.lower() for ind in sentence_indicators):
                continue

            # Check if SF could plausibly be an abbreviation of LF
            if self._could_be_abbreviation(sf, lf):
                results.append((sf, lf, match.start(), match.end()))

        # Pattern 2: "ABBREV, (the|or|i.e.|ie) long form"
        # e.g., "FV, the final vote" or "GPA, or granulomatosis with polyangiitis"
        pattern2 = re.compile(
            r"\b([A-Z][A-Za-z0-9/-]{1,9})"  # ABBREV
            r",?\s+(?:the|or|i\.?e\.?|namely|meaning)\s+"  # separator
            r"([a-z][a-z\s,/-]{5,60})"  # long form (lowercase start)
            r"(?=[.,;:\)\]\s]|$)",  # followed by punctuation or end
            re.UNICODE
        )

        for match in pattern2.finditer(text):
            sf = match.group(1).strip()
            lf = match.group(2).strip()

            # Clean up long form - remove trailing punctuation
            lf = re.sub(r"[.,;:]+$", "", lf).strip()

            if len(lf) < 5:
                continue

            if self._could_be_abbreviation(sf, lf):
                results.append((sf, lf, match.start(), match.end()))

        # Pattern 3: "ABBREV (long form)" - reversed Schwartz-Hearst
        # e.g., "LoE (level of evidence)"
        pattern3 = re.compile(
            r"\b([A-Z][A-Za-z0-9/-]{1,9})\s*"  # ABBREV
            r"\(([a-z][a-z\s,/-]{5,60})\)",  # (long form)
            re.UNICODE
        )

        for match in pattern3.finditer(text):
            sf = match.group(1).strip()
            lf = match.group(2).strip()

            if self._could_be_abbreviation(sf, lf):
                results.append((sf, lf, match.start(), match.end()))

        # Pattern 4: "ABBREV = long form" or "ABBREV: long form"
        pattern4 = re.compile(
            r"\b([A-Z][A-Za-z0-9/-]{1,9})\s*"  # ABBREV
            r"[=:]\s*"  # = or :
            r"([a-z][a-z\s,/-]{5,60})"  # long form
            r"(?=[.,;:\)\]\s]|$)",
            re.UNICODE
        )

        for match in pattern4.finditer(text):
            sf = match.group(1).strip()
            lf = match.group(2).strip()

            # Clean up long form
            lf = re.sub(r"[.,;:]+$", "", lf).strip()

            if len(lf) < 5:
                continue

            if self._could_be_abbreviation(sf, lf):
                results.append((sf, lf, match.start(), match.end()))

        return results

    def _could_be_abbreviation(self, sf: str, lf: str) -> bool:
        """
        Check if short form could plausibly be an abbreviation of long form.

        Uses a simple heuristic: at least half of the SF letters should
        appear in LF (in order), OR the SF appears as initials of LF words.
        """
        sf_upper = sf.upper()
        lf_lower = lf.lower()
        lf_words = lf_lower.split()

        # Check 1: Initials match
        # Get first letter of each word in LF
        initials = "".join(w[0] for w in lf_words if w)
        if initials.upper() == sf_upper:
            return True

        # Check 2: Partial initials match (for abbreviations that skip words)
        # Allow some flexibility - at least 50% of SF chars match LF initials
        matching = sum(1 for c in sf_upper if c.lower() in initials)
        if len(sf) >= 2 and matching >= len(sf) * 0.5:
            return True

        # Check 3: Letters appear in order in LF
        lf_idx = 0
        matches = 0
        for c in sf_upper:
            # Find this character in the remaining LF
            found = lf_lower.find(c.lower(), lf_idx)
            if found >= 0:
                matches += 1
                lf_idx = found + 1

        # At least half the SF letters should be found in order
        if matches >= len(sf) * 0.5:
            return True

        return False


__all__ = ["InlineDefinitionDetectorMixin"]
