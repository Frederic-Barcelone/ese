# corpus_metadata/Z_utils/Z02_text_helpers.py
"""
Text processing helper functions for the extraction pipeline.

These utilities handle text normalization, context extraction, and quality scoring
for abbreviation candidates during the extraction and validation process.
"""

from __future__ import annotations

import re
from itertools import islice
from typing import TYPE_CHECKING, Optional, Set

if TYPE_CHECKING:
    from A_core.A01_domain_models import Candidate


# Pattern for normalizing various dash/hyphen characters
_DASH_PATTERN = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212\u00ad\-–—]")


def extract_context_snippet(
    full_text: str, match_start: int, match_end: int, window: int = 100
) -> str:
    """Extract context snippet around a match position.

    Args:
        full_text: The complete document text
        match_start: Start character index of the match
        match_end: End character index of the match
        window: Number of characters to include before and after

    Returns:
        String containing the match plus surrounding context
    """
    start = max(0, match_start - window)
    end = min(len(full_text), match_end + window)
    return full_text[start:end]


def normalize_lf_for_dedup(lf: str) -> str:
    """
    Normalize a long form for deduplication comparison.

    Normalizes:
    - All dash/hyphen variants to standard hyphen
    - Multiple spaces to single space
    - Case to lowercase
    - Leading/trailing whitespace

    This prevents duplicates like:
    - "renin-angiotensin" vs "renin–angiotensin" (hyphen vs en-dash)
    - "urine protein-creatinine" vs "urine protein– creatinine"

    Args:
        lf: Long form string to normalize

    Returns:
        Normalized string for comparison
    """
    if not lf:
        return ""
    # Normalize dashes/hyphens to standard hyphen
    normalized = _DASH_PATTERN.sub("-", lf)
    # Normalize whitespace around hyphens
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized)
    # Lowercase and strip
    return normalized.lower().strip()


def has_numeric_evidence(context: str, sf: str) -> bool:
    """Check if SF appears with numeric evidence (digits, %, =, :).

    Used to validate statistical abbreviations by checking if they
    appear in a numeric context (e.g., "CI 95%", "HR = 1.5").

    Args:
        context: The context text around the abbreviation
        sf: The short form abbreviation

    Returns:
        True if numeric evidence is found near the SF
    """
    if not context:
        return False
    ctx = context.lower()
    sf_lower = sf.lower()

    idx = ctx.find(sf_lower)
    if idx == -1:
        return False

    window = ctx[max(0, idx - 30) : min(len(ctx), idx + len(sf) + 30)]

    # Check for numeric patterns
    if re.search(r"\d", window) or "%" in window or "=" in window:
        return True
    if re.search(rf"{re.escape(sf_lower)}\s*[:=]?\s*[\d.\-]", window):
        return True
    if re.search(r"[\d.]+\s*-\s*[\d.]+", window):
        return True
    return False


def is_valid_sf_form(
    sf: str, context: str, allowed_2letter: Set[str], allowed_mixed: Set[str]
) -> bool:
    """Filter SF by form - reject non-abbreviation patterns.

    Applies various heuristics to determine if a short form looks like
    a valid abbreviation rather than a figure reference, author initials,
    or other non-abbreviation patterns.

    Args:
        sf: The short form to validate
        context: The context text around the abbreviation
        allowed_2letter: Set of allowed 2-letter abbreviations
        allowed_mixed: Set of allowed mixed-case abbreviations

    Returns:
        True if the SF appears to be a valid abbreviation form
    """
    sf_upper = sf.upper()

    if len(sf) == 2 and sf_upper in allowed_2letter:
        return True
    if sf_upper in allowed_mixed:
        return True

    # Reject author initials pattern (X.Y., A.B., M.C., etc.)
    if re.match(r"^[A-Z]\.[A-Z]\.$", sf):
        return False

    # Reject figure/table references (Figure 3B, Table S1, etc.)
    if re.match(r"^(Figure|Table|Fig)\s*\d+[A-Za-z]?$", sf, re.IGNORECASE):
        return False
    if re.match(r"^(Figure|Table|Fig)\s*S\d+$", sf, re.IGNORECASE):
        return False

    # Reject lowercase "al" from "et al."
    if sf == "al":
        return False

    # Reject DOI patterns (10.xxxx/yyyy)
    if re.match(r"^10\.\d{4,}", sf):
        return False

    # Reject statistical method names that look like abbreviations
    if sf_upper in {"COX", "KAPLAN", "MEIER"}:
        return False

    # Reject plural forms of common abbreviations (e.g., "CIs" when "CI" exists)
    # These are just plurals, not separate abbreviations
    if sf_upper.endswith("S") and len(sf) >= 3:
        base = sf_upper[:-1]
        if base in {"CI", "HR", "OR", "RR", "SD", "SE", "AUC", "ROC"}:
            return False

    # Reject author initials pattern in reference/citation context
    # Patterns like "Hoffman EA," or "Celermajer DS," in author lists
    if len(sf) == 2 and sf.isupper():
        # Check if SF appears right after a capitalized name (author pattern)
        author_pattern = re.search(
            rf"[A-Z][a-z]+\s+{re.escape(sf)}[,\.\s]",
            context
        )
        if author_pattern:
            # Check if this appears in reference-like context
            ctx_lower = context.lower()
            if any(ind in ctx_lower for ind in ["doi:", "et al", "10.", "pmid", "j ", "vol", "pp.", "issue"]):
                return False

    # Special case: IG only if near immunoglobulin context
    if sf_upper == "IG":
        ctx_lower = context.lower()
        return any(
            x in ctx_lower for x in ["igg", "iga", "igm", "ige", "immunoglobulin"]
        )

    # Reject lowercase words
    if sf.islower() and len(sf) > 4:
        return False
    if len(sf) > 6 and sf.islower():
        return False
    # Reject capitalized words (e.g., "Medications", "Crucially")
    if len(sf) > 5 and sf[0].isupper() and sf[1:].islower():
        return False
    return True


def score_lf_quality(
    candidate: "Candidate", full_text: str, full_text_lower: Optional[str] = None
) -> int:
    """Score LF quality for dedup ranking.

    Assigns a quality score to a candidate's long form to help choose
    the best expansion when multiple candidates exist for the same
    short form.

    Args:
        candidate: The candidate to score
        full_text: Original full text of the document
        full_text_lower: Pre-cached lowercase version (optional, avoids repeated .lower() calls)

    Returns:
        Integer quality score (higher is better)
    """
    score = 0
    lf = (candidate.long_form or "").lower()
    sf = candidate.short_form
    sf_upper = sf.upper()

    # Use cached lowercase or compute once
    text_lower = full_text_lower if full_text_lower is not None else full_text.lower()

    # PRIORITY BOOST: Stats abbreviations prefer canonical forms
    # This ensures CI->"confidence interval" beats CI->"Curie"
    STATS_CANONICAL = {
        "CI": "confidence interval",
        "SD": "standard deviation",
        "SE": "standard error",
        "OR": "odds ratio",
        "RR": "risk ratio",
        "HR": "hazard ratio",
        "IQR": "interquartile range",
        "AUC": "area under the curve",
        "ROC": "receiver operating characteristic",
    }
    if sf_upper in STATS_CANONICAL:
        canonical = STATS_CANONICAL[sf_upper]
        if canonical in lf:
            score += 200  # Strong boost for canonical stats LF
        else:
            score -= 100  # Penalize non-canonical stats LF

    if lf and sf in full_text:
        # Check if LF appears within 200 chars of SF - use islice to avoid materializing all matches
        for m in islice(re.finditer(re.escape(sf), full_text), 5):
            window_start = max(0, m.start() - 200)
            window_end = m.start() + 200
            window = text_lower[window_start:window_end]
            if lf in window:
                score += 100
                break

    if lf and lf in text_lower:
        score += 50

    if lf:
        if len(lf) > 60:
            score -= 20
        lf_words = set(lf.split())
        if lf_words & {"the", "a", "an", "is", "are", "was"}:
            score -= 10
        # Penalize LFs that look like partial text extractions
        if lf.startswith("and ") or lf.startswith("or ") or lf.startswith("the "):
            score -= 50

    if candidate.provenance and candidate.provenance.lexicon_source:
        if "umls" in candidate.provenance.lexicon_source.lower():
            score += 30

    return score


__all__ = [
    "extract_context_snippet",
    "normalize_lf_for_dedup",
    "has_numeric_evidence",
    "is_valid_sf_form",
    "score_lf_quality",
]
