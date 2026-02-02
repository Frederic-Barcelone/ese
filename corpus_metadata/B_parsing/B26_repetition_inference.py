# corpus_metadata/B_parsing/B26_repetition_inference.py
"""
Header/footer repetition inference for PDF parsing.

Provides:
- Repeated text detection across pages
- Zone-based header/footer classification
- Running header pattern matching

Extracted from B01_pdf_to_docgraph.py to reduce file size.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Optional, Set, Tuple

from B_parsing.B23_text_helpers import (
    KNOWN_FOOTER_RE,
    RUNNING_HEADER_RE,
)


def infer_repeated_headers_footers(
    norm_count: Counter,
    norm_pages: Dict[str, Set[int]],
    norm_zone_votes: Dict[str, Counter],
    norm_sample_text: Dict[str, str],
    min_repeat_count: int = 3,
    min_repeat_pages: int = 3,
    repeat_zone_majority: float = 0.60,
) -> Tuple[Set[str], Set[str]]:
    """
    Infer repeated headers/footers using repetition-based detection.

    This is the PRIMARY detection method (not pattern-based filtering).

    Algorithm:
    1. Collect normalized text from header/footer zones
    2. Text appearing on >= min_repeat_pages pages -> repeated
    3. Zone majority vote determines header vs footer
    4. Generic patterns (DOI, copyright, etc.) boost footer classification

    Normalization: lowercase, collapse whitespace, replace digits with #
    This groups "Page 1" and "Page 23" as the same normalized text.

    Args:
        norm_count: Counter of normalized text occurrences
        norm_pages: Dict mapping normalized text to set of page numbers
        norm_zone_votes: Dict mapping normalized text to Counter of zones
        norm_sample_text: Dict mapping normalized text to original sample
        min_repeat_count: Minimum occurrences to consider repeated
        min_repeat_pages: Minimum pages to consider repeated
        repeat_zone_majority: Fraction threshold for zone classification

    Returns:
        Tuple of (repeated_headers set, repeated_footers set)
    """
    repeated_headers: Set[str] = set()
    repeated_footers: Set[str] = set()

    for norm, total_c in norm_count.items():
        pages = norm_pages.get(norm, set())

        # Repetition threshold: must appear on multiple pages
        if total_c < min_repeat_count:
            continue
        if len(pages) < min_repeat_pages:
            continue

        sample = norm_sample_text.get(norm, "")

        # Generic footer patterns (truly generic, not publisher-specific)
        if looks_like_known_footer(sample):
            repeated_footers.add(norm)
            continue

        # Page-number-only lines are usually footers
        if sample.strip().isdigit():
            repeated_footers.add(norm)
            continue

        # Short repeated text in zones is likely header/footer noise
        # e.g., "Research Article", journal names, author names
        if len(sample) <= 50 and total_c >= min_repeat_count:
            zone_votes = norm_zone_votes.get(norm)
            if zone_votes:
                top_zone, top_votes = zone_votes.most_common(1)[0]
                frac = float(top_votes) / float(total_c) if total_c else 0.0

                # Lower threshold for short repeated text
                threshold = repeat_zone_majority * 0.8

                if top_zone == "HEADER" and frac >= threshold:
                    repeated_headers.add(norm)
                    continue
                elif top_zone == "FOOTER" and frac >= threshold:
                    repeated_footers.add(norm)
                    continue

        # Standard zone-based classification
        zone_votes = norm_zone_votes.get(norm)
        if not zone_votes:
            continue

        top_zone, top_votes = zone_votes.most_common(1)[0]
        frac = float(top_votes) / float(total_c) if total_c else 0.0
        if top_zone == "HEADER" and frac >= repeat_zone_majority:
            repeated_headers.add(norm)
        elif top_zone == "FOOTER" and frac >= repeat_zone_majority:
            repeated_footers.add(norm)

    return repeated_headers, repeated_footers


def looks_like_known_footer(text: str) -> bool:
    """Check if text matches known footer patterns (DOI, copyright, etc.)."""
    t = (text or "").strip()
    if not t:
        return False
    return bool(KNOWN_FOOTER_RE.search(t))


def is_short_repeated_noise(text: str) -> bool:
    """Check if text is short repeated noise (page numbers, footer patterns)."""
    t = (text or "").strip()
    if not t:
        return False
    if len(t) <= 3 and t.isdigit():
        return True
    if looks_like_known_footer(t):
        return True
    return False


def is_running_header(text: str, zone: Optional[str]) -> bool:
    """
    Detect running headers like 'Liao et al' or 'Smith et al.'

    Args:
        text: Text content
        zone: Position zone (HEADER, BODY, FOOTER)

    Returns:
        True if text appears to be a running header
    """
    t = (text or "").strip()
    if not t:
        return False
    # Must be short (author name + et al)
    if len(t) > 30:
        return False
    # Match "Author et al" pattern
    if RUNNING_HEADER_RE.match(t):
        return True
    # Also catch in header zone with page numbers like "Liao et al 18194"
    if zone == "HEADER" and re.match(
        r"^[A-Z][a-z]+\s+et\s+al\.?\s*\d*$", t, re.IGNORECASE
    ):
        return True
    return False


__all__ = [
    "infer_repeated_headers_footers",
    "looks_like_known_footer",
    "is_short_repeated_noise",
    "is_running_header",
]
