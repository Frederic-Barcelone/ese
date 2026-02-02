"""
Noise filtering constants for abbreviation extraction.

This module provides constants for filtering obvious non-abbreviations and
invalid expansions. Contains curated sets of noise terms, blacklisted
SF/LF pairs, and the LexiconEntry data class for lexicon management.

Key Components:
    - OBVIOUS_NOISE: Set of terms that are obviously not abbreviations
    - MIN_ABBREV_LENGTH: Minimum length for valid abbreviations
    - WRONG_EXPANSION_BLACKLIST: SF->LF pairs that should never be used
    - BAD_LONG_FORMS: Long forms that are always wrong
    - LexiconEntry: Data class for compiled lexicon entries

Example:
    >>> from C_generators.C21_noise_filters import OBVIOUS_NOISE, WRONG_EXPANSION_BLACKLIST
    >>> "the" in OBVIOUS_NOISE
    True
    >>> ("US", "United States") in WRONG_EXPANSION_BLACKLIST
    True

Dependencies:
    - re: Regular expression matching
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple


# =============================================================================
# LIGHT NOISE FILTERING (High Recall - Let Validation Layer Judge)
# =============================================================================
# Philosophy: Generators should be EXHAUSTIVE. Only block OBVIOUS noise.
# Claude (D_validation) will handle borderline cases with context awareness.

# Obvious non-abbreviations: single letters, basic English function words
OBVIOUS_NOISE: Set[str] = {
    # Single letters (never valid abbreviations alone)
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    # Basic English function words (articles, prepositions, conjunctions)
    # NOTE: "or" and "us" removed - they can be valid abbreviations (Odds Ratio, United States)
    "an",
    "as",
    "at",
    "be",
    "by",
    "do",
    "go",
    "he",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "no",
    "of",
    "on",
    "so",
    "to",
    "up",
    "we",
    "the",
    "and",
    "for",
    "but",
    "not",
    "are",
    "was",
    "were",
    "been",
    "have",
    "has",
    "had",
    "will",
    "would",
    "could",
    "should",
    "this",
    "that",
    "these",
    "those",
    "with",
    "from",
    "into",
    # Citation artifacts
    "et",
    "al",
    # Name suffixes (part of author names, not abbreviations)
    "jr",
    "sr",
    # Month names and abbreviations (not domain abbreviations)
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    # Measurement units (not abbreviations, just units)
    "dl",
    "ml",
    "mg",
    "kg",
    "mm",
    "cm",
    "hz",
    "khz",
    "mhz",
    "mmhg",
    "kpa",
    "mol",
    "mmol",
    "umol",
    "nmol",
    # Full English words mistakenly in lexicons (NOT abbreviations)
    "investigator",
    "investigators",
    "sponsor",
    "protocol",
    "study",
    "patient",
    "patients",
    "subject",
    "subjects",
    "article",
    "articles",
    # Geographic (context-dependent, high FP rate in pharma docs)
    "nj",
    "ny",
    "ca",
    "tx",  # US states - usually location, not abbreviation
    # Company names (not abbreviations, even if in UMLS)
    "roche",
    "novartis",
    "pfizer",
    "merck",
    "bayer",
    "sanofi",
    "gsk",
    "astrazeneca",
    "amgen",
    "gilead",
    "biogen",
    "regeneron",
    "vertex",
    "alexion",
    "takeda",
    "abbvie",
    "lilly",
    "bristol",
    "johnson",
}

# Minimum length (allow 2-char if uppercase like CT, MR, IV)
MIN_ABBREV_LENGTH = 2


# =============================================================================
# WRONG EXPANSION BLACKLIST
# =============================================================================
# Some UMLS/lexicon entries have clearly wrong or contextually inappropriate
# expansions. These SF -> LF pairs should never be used.
#
# Format: (short_form_lower, bad_long_form_lower)
WRONG_EXPANSION_BLACKLIST: Set[Tuple[str, str]] = {
    # UMLS mapping errors
    ("task", "product"),  # TASK in protocols = schedule/activity, not SNOMED "product"
    ("musk", "musk secretion from musk deer"),  # MuSK = muscle-specific kinase
    # Clinical trial context - these expansions are wrong in protocol context
    ("et", "essential thrombocythemia"),  # ET in protocols = Early Termination
    ("sc", "subcutaneous"),  # Often correct, but sometimes wrong
    # Generic wrong mappings
    ("exam", "examination"),  # Too generic, not an abbreviation
    ("dose", "dosage"),  # Too generic
    ("task", "kcnk3 gene"),  # TASK is not usually a gene reference in protocols
}

# Long forms that are ALWAYS wrong (regardless of short form)
# These are UMLS artifacts or clearly incorrect expansions
BAD_LONG_FORMS: Set[str] = {
    "product",  # Too generic, SNOMED artifact
    "musk secretion from musk deer",  # Wrong MuSK expansion
    "essential thrombocythemia",  # Often wrong in clinical trial context
    "ambulatory care facilities",  # Wrong expansion for "Clinic"
    "simultaneous",  # Wrong expansion for "CONCOMITANT"
    "kit dosing unit",  # Wrong expansion for "Kits"
    "medical devices",  # Wrong expansion for "Device"
    "planum polare",  # Wrong expansion for "PP" (usually Per Protocol)
    "follicle stimulating hormone injectable",  # Wrong for FSH in most contexts
    "multiple sulfatase deficiency",  # MSD in pharma context = Merck Sharp & Dohme
}


class LexiconEntry:
    """Compiled lexicon entry with regex pattern and source provenance."""

    __slots__ = ("sf", "lf", "pattern", "source", "lexicon_ids", "preserve_case")

    def __init__(
        self,
        sf: str,
        lf: str,
        pattern: re.Pattern,
        source: str,
        lexicon_ids: Optional[List[Dict[str, str]]] = None,
        preserve_case: bool = True,
    ):
        self.sf = sf
        self.lf = lf
        self.pattern = pattern
        self.source = source  # Lexicon file name for provenance
        self.lexicon_ids = lexicon_ids or []  # External IDs [{source, id}, ...]
        self.preserve_case = preserve_case  # If True, use matched text as SF


def is_valid_abbreviation_match(term: str) -> bool:
    """
    Light filter - only block OBVIOUS noise.
    Let the Validation layer (Claude) handle borderline cases.
    """
    if not term:
        return False

    term_lower = term.lower().strip()

    # Block obvious noise (single letters, function words)
    if term_lower in OBVIOUS_NOISE:
        return False

    # Minimum length
    if len(term) < MIN_ABBREV_LENGTH:
        return False

    # Block pure numbers (not abbreviations)
    if term.isdigit():
        return False

    # Block short alphanumeric codes starting with digit (e.g., 1A, 2B, 45)
    if len(term) <= 3 and term[0].isdigit():
        return False

    return True


def is_wrong_expansion(sf: str, lf: str) -> bool:
    """
    Check if this SF -> LF mapping is a known wrong expansion.
    """
    sf_lower = sf.lower().strip()
    lf_lower = lf.lower().strip()

    # Check blacklisted pairs
    if (sf_lower, lf_lower) in WRONG_EXPANSION_BLACKLIST:
        return True

    # Check bad long forms
    if lf_lower in BAD_LONG_FORMS:
        return True

    return False
