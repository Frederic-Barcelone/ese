# corpus_metadata/B_parsing/B01a_text_helpers.py
"""
Text normalization, cleaning, and pattern utilities for PDF parsing.

Provides:
- Text normalization for header/footer detection
- Abbreviation hyphen handling
- OCR garbage filtering
- Garbled flowchart detection
- Section detection patterns
- Document markdown conversion utilities
"""

from __future__ import annotations

import re
from typing import Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from A_core.A01_domain_models import BoundingBox


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================


def normalize_repeated_text(text: str) -> str:
    """
    Normalize text for detecting repeated headers/footers:
    - lowercase
    - collapse whitespace
    - replace digits with '#'
    """
    t = " ".join((text or "").split()).lower()
    t = re.sub(r"\d+", "#", t)
    return t.strip()


# Glue hyphens in abbreviation patterns only (MG- ADL -> MG-ADL)
ABBREV_HYPHEN_RE = re.compile(
    r"(?P<a>[A-Za-z0-9][A-Za-z0-9+\./]{0,12})\s*-\s*(?P<b>[A-Za-z0-9][A-Za-z0-9+\./]{0,12})"
)


def _looks_like_abbrev_token(tok: str) -> bool:
    if not tok:
        return False
    has_upper = any(c.isupper() for c in tok)
    has_digit = any(c.isdigit() for c in tok)
    has_plus = "+" in tok
    shortish = len(tok) <= 12
    return shortish and (has_upper or has_digit or has_plus)


def normalize_abbrev_hyphens(text: str) -> str:
    """
    Convert 'MG- ADL' -> 'MG-ADL' if both sides look like abbreviations.
    Avoid touching normal words like 'long-term' (lowercase).
    """

    def repl(m: re.Match) -> str:
        a = m.group("a")
        b = m.group("b")
        if _looks_like_abbrev_token(a) and _looks_like_abbrev_token(b):
            return f"{a}-{b}"
        return m.group(0)

    return ABBREV_HYPHEN_RE.sub(repl, text)


# =============================================================================
# NOISE PATTERNS
# =============================================================================

# Generic footer indicators (truly generic, not publisher-specific)
GENERIC_FOOTER_PATTERNS = [
    r"\bdownloaded from\b",              # Generic access notice
    r"\bterms and conditions\b",         # Legal boilerplate
    r"\bcreative commons\b",             # License text
    r"\bdoi:\s*10\.\d{4,9}/",            # DOI always noise in footer
    r"\bopen access\b",                  # OA notice
    r"\bcopyright\s*©?\s*\d{4}\b",       # Copyright notices
    r"\ball rights reserved\b",          # Copyright boilerplate
    r"\breceived:?\s*\d{1,2}\s+\w+\s+\d{4}\b",  # Received date
    r"\baccepted:?\s*\d{1,2}\s+\w+\s+\d{4}\b",  # Accepted date
    r"\bpublished:?\s*\d{1,2}\s+\w+\s+\d{4}\b", # Published date
]
KNOWN_FOOTER_RE = re.compile("|".join(GENERIC_FOOTER_PATTERNS), flags=re.IGNORECASE)

# Running header patterns (author names like "Liao et al", "Smith et al.")
RUNNING_HEADER_RE = re.compile(r"^[A-Z][a-z]+\s+et\s+al\.?$", flags=re.IGNORECASE)

# Numbered reference pattern (e.g., "7. Author A, ..." or "7. DCVAS Study Group, ...")
NUMBERED_REFERENCE_RE = re.compile(
    r"^\d{1,3}\.\s+[A-Z]",  # Starts with number, period, then capital letter
    flags=re.MULTILINE,
)

# OCR garbage patterns to remove
OCR_GARBAGE_PATTERNS = [
    r"\b\d+\s*[bBpP»«]+\s*$",  # "18194 bP»" -> page number garbage
    r'[""]\s*[*>]\s*$',  # ""*", ">" at end
    r"\bfo\)\s*$",  # "fo)" misread lock icon
    r"^\s*[+*~>]{2,}\s*$",  # lines of just symbols
    r"\s+[»«]+\s*$",  # trailing » or «
    r"^\s*[◆●○■□▪▫►◄▸◂]+\s*$",  # bullet-only lines
    r"\b[Il1|]{4,}\b",  # misread vertical bars
    r"^\s*[-_=]{5,}\s*$",  # horizontal rules
    r"^\s*\.{5,}\s*$",  # dotted lines
    r"\b\d{1,2}\s*[oO0]\s*[fF]\s*\d{1,3}\b",  # "1 of 10" page numbers
    r"^\s*[#*]{3,}\s*$",  # decorative symbol lines
    r"\[\s*[A-Z]?\s*\]",  # empty checkbox placeholders
    r"^\s*\d+\s*$",  # lone page numbers
]
OCR_GARBAGE_RE = re.compile("|".join(OCR_GARBAGE_PATTERNS))

# Pattern to detect percentage values in text (for chart classification)
PERCENTAGE_PATTERN = re.compile(r'\d+%|\d+\s*%')

# Pattern to detect garbled flowchart/diagram content
FLOWCHART_PATTERN = re.compile(
    r"^\d+[a-z]?\.\s+.{10,}\s+[+*|~>-]\s+.{10,}\s+[+*|~>-]\s+", re.IGNORECASE
)


def is_garbled_flowchart(text: str) -> bool:
    """
    Detect if text block is a garbled flowchart/diagram.
    These typically have numbered steps with symbols like + * | ~ separating them.
    """
    if not text or len(text) < 100:
        return False

    # Count flowchart-like symbols
    symbol_count = sum(1 for c in text if c in "+*|~>-")
    word_count = len(text.split())

    # Check for numbered step pattern with symbols
    has_numbered_steps = bool(
        re.search(r"\d+[a-z]?\.\s+[^|+*]+[+*|]\s+.*\d+[a-z]?\.\s+", text)
    )

    if word_count > 0:
        ratio = symbol_count / word_count
        if has_numbered_steps and ratio > 0.08:
            return True
        if ratio > 0.12:
            if FLOWCHART_PATTERN.search(text):
                return True
            if has_numbered_steps:
                return True

    return False


def extract_figure_reference(text: str) -> Optional[str]:
    """Extract figure number from garbled flowchart text."""
    match = re.search(r"(?:Figure|Fig\.?)\s*(\d+)", text, re.IGNORECASE)
    if match:
        return f"Figure {match.group(1)}"
    return None


# =============================================================================
# SECTION DETECTION PATTERNS
# =============================================================================

SECTION_NUM_RE = re.compile(r"^\s*\d+(\.\d+)*\s*[|\.]?\s*\S+")

# Expanded section patterns including clinical/medical sections
SECTION_PATTERNS = [
    # Standard academic sections
    "abstract", "introduction", "methods", "materials and methods",
    "results", "discussion", "conclusion", "conclusions", "references",
    "background", "summary", "acknowledgements", "acknowledgments",
    # Clinical trial sections
    "study design", "study population", "patient characteristics",
    "baseline characteristics", "demographics", "eligibility",
    "inclusion criteria", "exclusion criteria", "eligibility criteria",
    "primary outcomes", "secondary outcomes", "primary endpoint",
    "secondary endpoints", "endpoints", "assessments",
    "efficacy", "efficacy results", "efficacy analysis",
    "safety", "safety analysis", "safety results", "adverse events",
    "tolerability", "pharmacokinetics", "pharmacodynamics",
    "statistical analysis", "statistical methods", "sample size",
    # Regulatory sections
    "indications", "contraindications", "warnings", "precautions",
    "dosage", "dosage and administration", "overdosage",
    "clinical pharmacology", "nonclinical toxicology",
    # Review/meta-analysis sections
    "search strategy", "data extraction", "quality assessment",
    "risk of bias", "sensitivity analysis", "subgroup analysis",
]
SECTION_PLAIN_RE = re.compile(
    r"^(" + "|".join(re.escape(p) for p in SECTION_PATTERNS) + r")$",
    flags=re.IGNORECASE,
)

META_PREFIX_RE = re.compile(
    r"^(correspondence|received|revised|accepted|funding|keywords)\s*:",
    flags=re.IGNORECASE,
)

# Common affiliation tokens in biomedical PDFs
AFFIL_TOKENS_RE = re.compile(
    r"\b(university|hospital|college|centre|center|institute|department|school|foundation|clinic|irccs|charité)\b",
    flags=re.IGNORECASE,
)

EMAIL_RE = re.compile(r"\b\S+@\S+\b")

# Many author blocks have pipes separating names
PIPEY_RE = re.compile(r"\s\|\s")


# =============================================================================
# TEXT CLEANING UTILITIES
# =============================================================================

# Prefixes to keep hyphenated
KEEP_HYPHEN_PREFIXES: Set[str] = {
    "anti", "non", "pre", "post", "co", "multi", "bi", "tri",
    "long", "short", "open", "double", "single", "high", "low",
    "well", "self", "cross", "small", "medium", "large",
}

# Common suffix fragments that indicate broken words
COMMON_SUFFIX_FRAGS: Set[str] = {
    "ment", "tion", "tions", "sion", "sions", "tive", "tives",
    "ness", "less", "able", "ible", "ated", "ation", "ations",
    "ing", "ed", "ive", "ous", "ally", "ity", "ies", "es", "ly",
    "bulin", "blast", "cytes", "rhage", "lines", "tology",
    "alities", "ressive", "pressive", "globulin", "itis", "osis",
    "emia", "pathy", "plasty",
}

# Known hyphenated compound words to preserve
KEEP_HYPHENATED: Set[str] = {
    "anca-associated",
    "medium-sized",
    "end-stage",
}

# Words that need space preserved: "small- and" not "small-and"
HYPHEN_SPACE_PRESERVE: Set[str] = {
    "small", "medium", "large", "short", "long",
}


def clean_text(text: str) -> str:
    """
    Text cleaning:
    - Remove OCR garbage artifacts
    - Rejoin hyphenated words split across lines
    - Collapse whitespace
    - Fix abbreviation hyphens: 'MG- ADL' -> 'MG-ADL'
    """
    t = text or ""
    t = t.replace("\r", "\n")

    # 0) Remove OCR garbage patterns
    t = OCR_GARBAGE_RE.sub("", t)

    # 1) Join hyphenated line-break words: word-\n word -> wordword
    t = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", t)

    # 2) Handle "immuno- globulin" style (hyphen + space, broken word)
    def _dehyphen_repl(m: re.Match) -> str:
        a = m.group(1)
        b = m.group(2)
        a_l = a.lower()
        b_l = b.lower()
        combined = f"{a_l}-{b_l}"

        # Keep known hyphenated compounds
        if combined in KEEP_HYPHENATED:
            return f"{a}-{b}"

        # Preserve "small- and", "medium- and" patterns
        if a_l in HYPHEN_SPACE_PRESERVE and b_l == "and":
            return f"{a}- {b}"

        # Keep hyphen for known prefixes
        if a_l in KEEP_HYPHEN_PREFIXES:
            return f"{a}-{b}"

        # Remove hyphen when right side looks like a suffix fragment
        if b_l in COMMON_SUFFIX_FRAGS:
            return f"{a}{b}"

        # Remove hyphen for short left parts (likely broken words)
        if len(a) <= 5 and len(b) >= 3:
            return f"{a}{b}"

        return f"{a}-{b}"

    t = re.sub(r"\b([A-Za-z]{2,})-\s+([A-Za-z]{2,})\b", _dehyphen_repl, t)

    # 3) collapse whitespace
    t = " ".join(t.split()).strip()

    # 4) strip leading pipe artefact ("| Andreas..." -> "Andreas...")
    if re.match(r"^\|\s*[A-Za-z]", t):
        t = re.sub(r"^\|\s*", "", t)

    # 5) normalize abbrev hyphen spacing only for abbrev-like tokens
    t = normalize_abbrev_hyphens(t)

    return t.strip()


# =============================================================================
# DOCUMENT UTILITIES
# =============================================================================


def bbox_overlaps(
    bbox1: "BoundingBox", bbox2: "BoundingBox", threshold: float = 0.5
) -> bool:
    """Check if two bounding boxes overlap significantly."""
    x1_0, y1_0, x1_1, y1_1 = bbox1.coords
    x2_0, y2_0, x2_1, y2_1 = bbox2.coords

    # Calculate intersection
    ix0 = max(x1_0, x2_0)
    iy0 = max(y1_0, y2_0)
    ix1 = min(x1_1, x2_1)
    iy1 = min(y1_1, y2_1)

    if ix1 <= ix0 or iy1 <= iy0:
        return False

    intersection = (ix1 - ix0) * (iy1 - iy0)
    area1 = (x1_1 - x1_0) * (y1_1 - y1_0)

    if area1 <= 0:
        return False

    return (intersection / area1) >= threshold


def table_to_markdown(table) -> str:
    """Convert Table object to markdown table format."""
    if not table.logical_rows:
        return f"[Table: {table.caption or 'Empty'}]"

    # Get headers
    headers_map = table.metadata.get("headers", {})
    if headers_map:
        headers = [headers_map.get(i, f"Col{i}") for i in sorted(headers_map.keys())]
    elif table.logical_rows:
        headers = list(table.logical_rows[0].keys())
    else:
        return f"[Table: {table.caption or 'No data'}]"

    out = []
    if table.caption:
        out.append(f"**{table.caption}**")
        out.append("")

    # Header row
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Data rows
    for row in table.logical_rows:
        cells = [str(row.get(h, "")).replace("|", "\\|") for h in headers]
        out.append("| " + " | ".join(cells) + " |")

    return "\n".join(out)


__all__ = [
    # Text normalization
    "normalize_repeated_text",
    "normalize_abbrev_hyphens",
    # Noise patterns
    "KNOWN_FOOTER_RE",
    "RUNNING_HEADER_RE",
    "NUMBERED_REFERENCE_RE",
    "OCR_GARBAGE_RE",
    "PERCENTAGE_PATTERN",
    "is_garbled_flowchart",
    "extract_figure_reference",
    # Section patterns
    "SECTION_NUM_RE",
    "SECTION_PLAIN_RE",
    "META_PREFIX_RE",
    "AFFIL_TOKENS_RE",
    "EMAIL_RE",
    "PIPEY_RE",
    # Text cleaning
    "clean_text",
    "KEEP_HYPHEN_PREFIXES",
    "COMMON_SUFFIX_FRAGS",
    "KEEP_HYPHENATED",
    "HYPHEN_SPACE_PRESERVE",
    # Document utilities
    "bbox_overlaps",
    "table_to_markdown",
]
