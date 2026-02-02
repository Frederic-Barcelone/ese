# corpus_metadata/A_core/A20_unicode_utils.py
"""
Unicode Normalization Utilities

Handle PDF mojibake, ligatures, variant hyphens, and other Unicode issues
common in PDF text extraction.

Extracted from A04_heuristics_config.py for better modularity and testability.
"""

import re
import unicodedata


# ========================================
# UNICODE NORMALIZATION UTILITIES
# Handle PDF mojibake, ligatures, variant hyphens, etc.
# ========================================

# Various Unicode hyphen/dash characters to normalize
HYPHENS_PATTERN = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212\u00ad]")

# Common mojibake substitutions (e.g., from PDF extraction)
MOJIBAKE_MAP = {
    "Î±": "α",  # Greek alpha (common PDF encoding issue)
    "Î²": "β",  # Greek beta
    "Î³": "γ",  # Greek gamma
    "Î´": "δ",  # Greek delta
    "ﬁ": "fi",  # fi ligature
    "ﬂ": "fl",  # fl ligature
    "ﬀ": "ff",  # ff ligature
    "ﬃ": "ffi",  # ffi ligature
    "ﬄ": "ffl",  # ffl ligature
}


def normalize_sf(sf: str) -> str:
    """
    Normalize a short form for display/storage.

    - Applies NFKC Unicode normalization
    - Fixes common mojibake issues
    - Normalizes hyphens to standard ASCII hyphen
    - Collapses whitespace
    """
    s = unicodedata.normalize("NFKC", sf).strip()
    # Fix mojibake
    for bad, good in MOJIBAKE_MAP.items():
        s = s.replace(bad, good)
    # Normalize hyphens
    s = HYPHENS_PATTERN.sub("-", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_sf_key(sf: str) -> str:
    """
    Normalize a short form for use as dictionary/set key (comparison).

    Returns uppercase normalized form for consistent matching.
    """
    return normalize_sf(sf).upper()


def normalize_context(ctx: str) -> str:
    """
    Normalize context text for matching.

    - Applies NFKC Unicode normalization
    - Normalizes hyphens
    - Returns lowercase for case-insensitive matching
    """
    c = unicodedata.normalize("NFKC", ctx)
    c = HYPHENS_PATTERN.sub("-", c)
    return c.lower()


def clean_long_form(lf: str) -> str:
    """
    Clean up long form text extracted from PDFs.

    Fixes common PDF parsing artifacts:
    - Line-break hyphenation: "gastro-\nintestinal" -> "gastrointestinal"
    - Truncation detection: returns empty string if truncated
    - Extra whitespace: collapses multiple spaces
    - Mojibake: fixes common encoding issues

    Args:
        lf: Raw long form string from PDF extraction

    Returns:
        Cleaned long form, or empty string if invalid/truncated
    """
    if not lf:
        return ""

    # Apply NFKC normalization
    lf = unicodedata.normalize("NFKC", lf).strip()

    # Fix mojibake
    for bad, good in MOJIBAKE_MAP.items():
        lf = lf.replace(bad, good)

    # Fix line-break hyphenation: "gastro-\nintestinal" -> "gastrointestinal"
    # Pattern: hyphen followed by optional whitespace/newline, then lowercase letter
    lf = re.sub(r"-\s*\n\s*([a-z])", r"\1", lf)
    lf = re.sub(r"-\s+([a-z])", r"\1", lf)

    # Normalize all hyphens to standard ASCII
    lf = HYPHENS_PATTERN.sub("-", lf)

    # Collapse whitespace
    lf = re.sub(r"\s+", " ", lf).strip()

    # Detect truncation: word ending with just 1-3 consonants (likely incomplete)
    # e.g., "vasculi" (missing "tis"), "gastrointestin" (missing "al")
    words = lf.split()
    if words:
        last_word = words[-1].lower()
        # Check if last word looks truncated (ends abruptly)
        if len(last_word) >= 3:
            # Common truncation patterns: word ends with unusual consonant clusters
            truncation_endings = [
                "stin", "culi", "liti", "niti", "rati", "mati",  # -tion, -itis, etc.
                "gica", "logi", "path", "neur", "chem", "phar",  # -gical, -logy, etc.
            ]
            for ending in truncation_endings:
                if last_word.endswith(ending) and len(last_word) < 8:
                    # Likely truncated, return empty to trigger fallback
                    return ""

    return lf


def is_truncated_term(term: str) -> bool:
    """
    Check if a term appears to be truncated from PDF extraction.

    Returns True if term looks incomplete (likely PDF artifact).
    """
    if not term or len(term) < 4:
        return False

    term_lower = term.lower().strip()

    # Check for common incomplete suffixes
    incomplete_patterns = [
        r"vasculit?i?$",      # vasculitis
        r"glomeru?l?o?$",     # glomerulonephritis
        r"nephropa?t?h?$",    # nephropathy
        r"encepha?l?o?$",     # encephalopathy
        r"myopa?t?h?$",       # myopathy
        r"neuropa?t?h?$",     # neuropathy
        r"cardio?m?y?o?$",    # cardiomyopathy
        r"throm?b?o?$",       # thrombocytopenia
        r"pancre?a?t?$",      # pancreatitis
        r"hepat?i?t?$",       # hepatitis
    ]

    for pattern in incomplete_patterns:
        if re.search(pattern, term_lower):
            return True

    return False
