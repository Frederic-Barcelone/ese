# corpus_metadata/C_generators/C20_abbrev_patterns.py
"""
Pattern constants and helper functions for abbreviation extraction.

Contains:
- Regex patterns for abbreviation token validation
- Author initial detection patterns
- Schwartz-Hearst extraction helpers
- Long form normalization utilities
"""

from __future__ import annotations

import re
from typing import Optional


# =============================================================================
# PATTERN CONSTANTS
# =============================================================================

_ABBREV_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-\+/().]{0,14}$")

# Pattern to detect author initial contexts
# Matches patterns like "John Smith1,2, Jane Doe3" or "A. Smith, B. Jones"
_AUTHOR_LIST_PATTERN = re.compile(
    r"(?:[A-Z][a-z]+\s+[A-Z][a-z]+\s*[0-9,\s]*,?\s*){2,}",  # Multiple "First Last" names
    re.MULTILINE,
)

# Pattern for initials in author names (e.g., "J.H." or "JH" after a name)
_AUTHOR_INITIAL_PATTERN = re.compile(
    r"[A-Z][a-z]+\s+(?:[A-Z]\.?\s*)+[,\s0-9]"  # Name followed by initials
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _is_likely_author_initial(sf: str, context: str) -> bool:
    """
    Check if a short uppercase abbreviation is likely an author initial.

    Uses pattern + context detection (no hardcoded initial lists).

    Author initials appear in contexts like:
    - "John Smith1,2, BH contributed to..." (where BH = initials)
    - Author contribution sections
    - Name lists with superscript affiliations

    Detection strategy:
    1. Pattern: 2-3 letter all-uppercase alphabetic string
    2. Context: author-related keywords nearby
    3. Proximity: multiple similar initials nearby (author list pattern)

    Returns True if the SF is likely an author initial (should be filtered).
    """
    sf = (sf or "").strip()
    ctx = (context or "").strip()

    if not sf or not ctx:
        return False

    # Pattern check: 2-3 letter all-uppercase alphabetic (typical author initials)
    # Examples: "BH", "JM", "JLU" (but not "IL6", "TNF")
    if not (2 <= len(sf) <= 3 and sf.isupper() and sf.isalpha()):
        return False

    ctx_lower = ctx.lower()

    # Context clues that indicate author/contributor section
    author_clues = [
        "contributed", "author", "wrote", "drafted", "reviewed",
        "approved", "manuscript", "acknowledgement", "acknowledgment",
        "funding", "conflict", "interest", "affiliation", "department",
        "university", "hospital", "medical center", "school of",
        "corresponding", "equal contribution", "contributors",
        "conceptualization", "methodology", "investigation",
        "writing", "supervision", "declaration",
    ]

    # Strong author context: if any author clue present, likely an initial
    if any(clue in ctx_lower for clue in author_clues):
        return True

    # Author list patterns: names with superscript numbers/affiliations
    # e.g., "John Smith1,2, Jane Doe3, BH"
    if re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+\s*[0-9,*†‡§]+", ctx):
        return True

    # Proximity heuristic: multiple 2-3 letter uppercase sequences nearby
    # e.g., "BH, JH, MK contributed..." or "BH and JM designed..."
    # This catches author lists even without explicit author keywords
    initials_nearby = re.findall(r"\b[A-Z]{2,3}\b", ctx)
    # Filter out known non-initial abbreviations (common medical/scientific)
    known_abbrevs = {
        "TNF", "DNA", "RNA", "PCR", "MRI", "HIV", "HCV", "HBV",
        "ACE", "ARB", "GFR", "CKD", "AKI", "RRT", "PAH", "PH",
        "USA", "FDA", "EMA", "WHO", "BMI", "ICU", "CCU",
    }
    initials_nearby = [i for i in initials_nearby if i not in known_abbrevs]

    if len(initials_nearby) >= 3:
        return True

    # Check for comma-separated initial pattern: "BH, JM, and SK"
    if re.search(r"\b[A-Z]{2,3}\s*,\s*[A-Z]{2,3}\s*(?:,|and)\s*[A-Z]{2,3}\b", ctx):
        return True

    return False


def _clean_ws(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _dehyphenate_long_form(lf: str) -> str:
    """
    Remove line-break hyphens from long forms.

    PDF extraction often produces hyphenated words where lines break:
    - "gastroin-testinal" -> "gastrointestinal"
    - "Vasculi-tis Study Group" -> "Vasculitis Study Group"

    Pattern: hyphen followed by whitespace (from line break) then lowercase letter
    indicates a word was split across lines.
    """
    if not lf:
        return lf

    # First normalize whitespace
    lf = _clean_ws(lf)

    # Pattern: hyphen + space + lowercase continuation
    # This catches line-break hyphenation where space remains after normalization
    lf = re.sub(r"-\s+([a-z])", r"\1", lf)

    # Pattern: lowercase-hyphen-lowercase within a "word" that looks broken
    # Be careful to preserve valid compounds like "anti-inflammatory"
    # Only dehyphenate if it doesn't match a known compound prefix pattern

    # Common prefixes that form valid hyphenated compounds - don't dehyphenate these
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
    # Only process if it looks like a broken word (not at word boundary)
    lf = re.sub(r"(\w)-([a-z]{2,})", maybe_dehyphenate, lf)

    return lf


def _truncate_at_breaks(text: str) -> str:
    """
    For implicit LF captures: cut at punctuation / obvious clause starters.

    Improved to reduce false positives by stopping at:
    - Punctuation marks
    - Relative clause starters (which, that, who, etc.)
    - Verb phrases that indicate narrative rather than definition
    - Trailing connectors and articles
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

    # Stop at verb phrases that indicate narrative/commentary rather than definition
    # e.g., "enzyme was evaluated" -> "enzyme"
    # e.g., "receptor is a subtype" -> "receptor"
    t = re.split(
        r"\b(was|were|is|are|has|have|had|being|been|can|could|would|should|may|might)\s+",
        t,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    # Stop at common non-definition patterns
    # e.g., "protein also known as" -> handled by implicit patterns, not here
    t = re.split(
        r"\b(also|previously|formerly|sometimes|commonly|often)\b",
        t,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    # Drop trailing connector words and articles
    t = re.sub(
        r"\b(and|or|as|by|the|a|an|of|in|for|to)\s*$", "", t, flags=re.IGNORECASE
    ).strip()

    # Drop trailing punctuation that might have been left
    t = re.sub(r"[,\-]\s*$", "", t).strip()

    return _clean_ws(t)


def _normalize_long_form(lf: str) -> str:
    """
    Full normalization for long forms: clean whitespace, dehyphenate, truncate at breaks.
    """
    if not lf:
        return lf

    original = lf

    # First clean whitespace (collapses newlines to spaces)
    lf = _clean_ws(lf)

    # Truncate at obvious clause/sentence breaks
    # This prevents long forms from including unrelated text
    truncated = _truncate_at_breaks(lf)

    # Only use truncated version if it's not too short
    # (truncation might over-aggressively cut valid long forms)
    if truncated and len(truncated) >= 3:
        lf = truncated
    # Else keep the cleaned but non-truncated version

    # Dehyphenate line-break artifacts
    lf = _dehyphenate_long_form(lf)

    # Ensure we return something meaningful
    if not lf or len(lf) < 2:
        return _dehyphenate_long_form(_clean_ws(original))

    return lf


def _looks_like_short_form(sf: str, min_len: int = 2, max_len: int = 10) -> bool:
    sf = (sf or "").strip()
    if len(sf) < min_len or len(sf) > max_len:
        return False
    if sf.isdigit():
        return False

    # Handle space-separated abbreviations (e.g., "CC BY", "IL 2", "Type I")
    if " " in sf:
        # Allow only if it's 2-3 short uppercase tokens (like "CC BY", "IL 2")
        parts = sf.split()
        if len(parts) > 3:
            return False
        # Each part should be short and mostly uppercase/alphanumeric
        for part in parts:
            if len(part) > 6:  # Each token max 6 chars
                return False
            if not any(ch.isupper() for ch in part) and not part.isdigit():
                return False
        # Total length check (already done above, but be explicit)
        if len(sf) > max_len + 3:  # Allow slightly longer for spaces
            return False
        return True

    if not _ABBREV_TOKEN_RE.match(sf):
        return False
    # Must contain at least one uppercase letter (strong pharma heuristic)
    if not any(ch.isupper() for ch in sf):
        return False
    return True


def _context_window(text: str, start: int, end: int, window: int = 240) -> str:
    """
    Create a short context window around a match.
    """
    if not text:
        return ""
    s = max(0, start - window)
    e = min(len(text), end + window)
    return _clean_ws(text[s:e])


def _looks_like_measurement(text: str) -> bool:
    """
    Check if text looks like a measurement value rather than a definition.
    Filters false positives like "11.06 mg/L (UACR-I)".
    """
    if not text:
        return False

    text = text.strip()

    # Define measurement units (including compound units like mg/L, g/dL)
    units = r"(?:mg|g|L|mL|μg|ng|IU|%|°C|mmol|μmol|pg|kg|mm|cm|m|s|min|h|d|Hz|kDa|Da)"
    compound_unit = units + r"(?:/" + units + r")?"

    # Reject if ends with a measurement pattern: "NUMBER UNIT"
    # e.g., "11.06 mg/L", "76.79 mg/L", "100 %", "1.3 x 10^9/L"
    measurement_end_pattern = re.compile(
        r"\d+(?:\.\d+)?\s*(?:×\s*10[\^]?\d+)?/?\s*" + compound_unit + r"\s*$",
        re.IGNORECASE,
    )
    if measurement_end_pattern.search(text):
        return True

    # Reject if mostly numeric content (more digits than letters)
    letters = sum(1 for c in text if c.isalpha())
    digits = sum(1 for c in text if c.isdigit())
    if digits > 0 and digits >= letters:
        return True

    # Reject if looks like a numeric list/sequence
    if re.match(r"^[\d.,×\s\-/]+$", text):
        return True

    return False


def _space_sf_extract(short_form: str, preceding_text: str) -> Optional[str]:
    """
    Extract long form for space-separated abbreviations like "CC BY".

    Strategy: Match each letter in the SF to word initials in the preceding text.

    E.g., "CC BY" with preceding "Creative Commons Attribution":
    - C -> Creative, C -> Commons, B -> (no match), Y -> (no match)
    - But we find 2 matches for CC, which is good enough
    - Result: "Creative Commons Attribution" (take words that cover the matches)
    """
    sf = (short_form or "").strip()
    txt = (preceding_text or "").strip()

    if not sf or not txt or " " not in sf:
        return None

    # Get SF letters (uppercase, no spaces)
    sf_letters = [c.upper() for c in sf if c.isalpha()]
    if len(sf_letters) < 2:
        return None

    # Extract words from preceding text (filter out empty/punctuation-only)
    words = [w for w in txt.split() if w and any(c.isalpha() for c in w)]
    if len(words) < 2:
        return None

    # Find the best starting position by matching SF letters to word initials
    # Work backwards from the end of the text
    best_start = -1
    best_matches = 0

    for start_idx in range(len(words)):
        # Try to match SF letters starting from this word
        matches = 0
        sf_idx = 0
        for word_idx in range(start_idx, len(words)):
            if sf_idx >= len(sf_letters):
                break
            word_initial = words[word_idx][0].upper() if words[word_idx] else ""
            if word_initial == sf_letters[sf_idx]:
                matches += 1
                sf_idx += 1

        # Prefer matches that start with the first SF letter
        if matches > best_matches:
            # Verify first letter alignment
            word_initial = words[start_idx][0].upper() if words[start_idx] else ""
            if word_initial == sf_letters[0]:
                best_matches = matches
                best_start = start_idx

    # Need at least 2 letter matches for space-SF
    if best_matches < 2 or best_start < 0:
        return None

    # Extract from best_start to end
    lf = " ".join(words[best_start:])

    # Basic validation
    if 5 <= len(lf) <= 80:
        return lf

    return None


def _word_initial_extract(short_form: str, preceding_text: str) -> Optional[str]:
    """
    Word-initial letter matching for acronyms.
    Matches SF letters to the first letter of each word in preceding text.

    Works well for: "Acanthosis nigricans (AN)" where A=Acanthosis, N=nigricans
    """
    sf = re.sub(r"[^A-Za-z0-9]", "", short_form or "")
    if len(sf) < 2:
        return None

    txt = (preceding_text or "").rstrip()
    if not txt:
        return None

    # Extract words (alphanumeric sequences)
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", txt)
    if len(words) < len(sf):
        return None

    # Try to match SF letters to word-initial letters (backwards from end of words)
    # Look for consecutive words whose initials spell the SF
    sf_lower = sf.lower()

    # Sliding window: try to find len(sf) consecutive words matching SF
    for start_idx in range(len(words) - len(sf), -1, -1):
        candidate_words = words[start_idx : start_idx + len(sf)]
        initials = "".join(w[0].lower() for w in candidate_words)

        if initials == sf_lower:
            # Found a match! Build the long form
            lf = " ".join(candidate_words)

            # Validate: first letter must match
            if lf[0].lower() != sf[0].lower():
                continue

            # Avoid absurdly long LFs
            if len(lf) > 120:
                continue

            # Reject measurements
            if _looks_like_measurement(lf):
                continue

            return lf

    return None


def _schwartz_hearst_extract(short_form: str, preceding_text: str) -> Optional[str]:
    """
    Schwartz-Hearst-ish backward character alignment.
    Returns a best-effort long form (LF) extracted from preceding_text, or None.

    Works well for: "... Tumor Necrosis Factor (TNF)"

    Now tries word-initial matching first (better for disease acronyms),
    then falls back to classic character alignment.
    """
    # Try word-initial matching first (works better for disease acronyms)
    lf = _word_initial_extract(short_form, preceding_text)
    if lf:
        return lf

    # Fall back to classic Schwartz-Hearst character alignment
    sf = re.sub(r"[^A-Za-z0-9]", "", short_form or "")
    if len(sf) < 2:
        return None

    # Limit lookback window (characters)
    txt = (preceding_text or "").rstrip()
    if not txt:
        return None

    window_size = min(len(txt), len(sf) * 18 + 30)
    txt = txt[-window_size:]

    i = len(sf) - 1
    j = len(txt) - 1

    # Align SF chars backwards into txt backwards
    while i >= 0:
        c = sf[i].lower()
        while j >= 0 and txt[j].lower() != c:
            j -= 1
        if j < 0:
            return None
        i -= 1
        j -= 1

    # j is now before the earliest matched character; pick LF start at word boundary
    start = txt.rfind(" ", 0, j + 1) + 1
    lf = _clean_ws(txt[start:])

    if not lf:
        return None

    # Strong safety: first letter alignment (prevents many false positives)
    if lf[0].lower() != short_form[0].lower():
        return None

    # Avoid absurdly long LFs
    if len(lf) > 120:
        return None

    # Reject measurement values (e.g., "11.06 mg/L" is not a definition)
    if _looks_like_measurement(lf):
        return None

    return lf


def _extract_preceding_name(short_form: str, preceding_text: str) -> Optional[str]:
    """
    Extract a drug/compound name immediately before parentheses.

    Handles patterns like "iptacopan (LNP023)" where:
    - The SF is an alphanumeric code (LNP023, NCT04817618)
    - The LF is a proper name/word directly before the parentheses

    This is different from Schwartz-Hearst because the code letters
    don't need to appear in the name.
    """
    sf = (short_form or "").strip()
    txt = (preceding_text or "").rstrip()

    if not sf or not txt:
        return None

    # SF should look like an alphanumeric code (contains letters AND digits)
    has_letter = any(c.isalpha() for c in sf)
    has_digit = any(c.isdigit() for c in sf)
    is_compound_id = has_letter and has_digit and len(sf) >= 4

    if not is_compound_id:
        return None

    # Extract the last word before parentheses
    # Match a word that's either:
    # - A proper noun (capitalized) like "Iptacopan"
    # - A lowercase word like "iptacopan"
    # But NOT all-caps (that would be another abbreviation)
    word_match = re.search(r"([A-Za-z][a-z]{2,20})\s*$", txt)
    if not word_match:
        return None

    candidate_lf = word_match.group(1)

    # Skip if the candidate is too short or too long
    if len(candidate_lf) < 3 or len(candidate_lf) > 25:
        return None

    # Skip if it looks like a common word (articles, prepositions)
    skip_words = {
        "the", "and", "for", "with", "from", "that", "this",
        "which", "where", "when", "while", "being", "both"
    }
    if candidate_lf.lower() in skip_words:
        return None

    return candidate_lf


def _validate_sf_in_lf(short_form: str, long_form: str) -> bool:
    """
    Lightweight validation that SF letters appear in LF (in order, not necessarily contiguous).
    This is used for:
      - validating implicit captures
      - validating reverse explicit pattern: "SF (Long Form)"
    """
    sf = re.sub(r"[^A-Za-z0-9]", "", short_form or "")
    lf = re.sub(r"[^A-Za-z0-9 ]", " ", long_form or "")
    lf = _clean_ws(lf)

    if not sf or not lf:
        return False

    if lf[0].lower() != short_form[0].lower():
        return False

    # In-order scan across LF
    sf_idx = 0
    for ch in lf:
        if ch.lower() == sf[sf_idx].lower():
            sf_idx += 1
            if sf_idx == len(sf):
                return True

    return False


__all__ = [
    "_ABBREV_TOKEN_RE",
    "_AUTHOR_LIST_PATTERN",
    "_AUTHOR_INITIAL_PATTERN",
    "_is_likely_author_initial",
    "_clean_ws",
    "_dehyphenate_long_form",
    "_truncate_at_breaks",
    "_normalize_long_form",
    "_looks_like_short_form",
    "_context_window",
    "_looks_like_measurement",
    "_space_sf_extract",
    "_word_initial_extract",
    "_schwartz_hearst_extract",
    "_extract_preceding_name",
    "_validate_sf_in_lf",
]
