"""
Noise filtering constants for abbreviation extraction.

This module provides constants for filtering obvious non-abbreviations and
invalid expansions. Contains curated sets of noise terms, blacklisted
SF/LF pairs, and the LexiconEntry data class for lexicon management.

Data is loaded from G_config/data/noise_filters.yaml.

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
    - Z_utils.Z12_data_loader: YAML data loading
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from Z_utils.Z12_data_loader import load_pair_list, load_term_set

# =============================================================================
# LIGHT NOISE FILTERING (High Recall - Let Validation Layer Judge)
# =============================================================================
# Philosophy: Generators should be EXHAUSTIVE. Only block OBVIOUS noise.
# Claude (D_validation) will handle borderline cases with context awareness.

OBVIOUS_NOISE: Set[str] = load_term_set("noise_filters.yaml", "obvious_noise")

# Minimum length (allow 2-char if uppercase like CT, MR, IV)
MIN_ABBREV_LENGTH = 2

# =============================================================================
# WRONG EXPANSION BLACKLIST
# =============================================================================
# Some UMLS/lexicon entries have clearly wrong or contextually inappropriate
# expansions. These SF -> LF pairs should never be used.
WRONG_EXPANSION_BLACKLIST: Set[Tuple[str, str]] = load_pair_list("noise_filters.yaml", "wrong_expansion_blacklist")

# Long forms that are ALWAYS wrong (regardless of short form)
# These are UMLS artifacts or clearly incorrect expansions
BAD_LONG_FORMS: Set[str] = load_term_set("noise_filters.yaml", "bad_long_forms")


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
