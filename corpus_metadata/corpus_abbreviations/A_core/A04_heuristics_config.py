# corpus_metadata/corpus_abbreviations/A_core/A04_heuristics_config.py
"""
Centralized Heuristics Configuration

All whitelists, blacklists, and configurable rules in one place.
Enables easy tuning without modifying pipeline logic.

Categories tracked for logging:
- recovered_by_stats_whitelist
- recovered_by_hyphen
- recovered_by_llm_sf_only
- blacklisted_fp_count
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional
import logging


@dataclass
class HeuristicsConfig:
    """
    Centralized configuration for abbreviation extraction heuristics.

    All rules are configurable to avoid hard-coding in multiple places.
    """

    # ========================================
    # STATS WHITELIST (PASO A) - Auto-approve with numeric evidence
    # Dict format: {SF: canonical_LF} - use .keys() for membership check
    # ========================================
    stats_abbrevs: Dict[str, str] = field(default_factory=lambda: {
        'CI': 'confidence interval',
        'SD': 'standard deviation',
        'SE': 'standard error',
        'OR': 'odds ratio',
        'RR': 'risk ratio',
        'HR': 'hazard ratio',
        'IQR': 'interquartile range',
        'AUC': 'area under the curve',
        'ROC': 'receiver operating characteristic',
    })

    # ========================================
    # COUNTRY CODES (PASO B) - Auto-approve
    # Dict format: {SF: canonical_LF} - use .keys() for membership check
    # ========================================
    country_abbrevs: Dict[str, str] = field(default_factory=lambda: {
        'US': 'United States',
        'UK': 'United Kingdom',
        'USA': 'United States of America',
        'EU': 'European Union',
    })

    # ========================================
    # BLACKLIST - Auto-reject (SURGICAL: keep small and targeted)
    # Only include clear FPs that are never valid medical abbreviations
    # ========================================
    sf_blacklist: Set[str] = field(default_factory=lambda: {
        # Author credentials (never medical abbreviations in context)
        'MD', 'PHD', 'MBBS', 'FRCP', 'MPH',

        # US state abbreviations (location context, not medical)
        'NY', 'NJ',

        # Ambiguous 2-letter that are rarely medical
        'IA',  # Iowa, intramural - ambiguous without strong context

        # Common words/months that get lexicon matches
        'AUG', 'INT',  # August, International - not abbreviations

        # Research/tool names that aren't medical abbreviations
        'IRCCS',  # Italian research institute
    })

    # Trial IDs - special handling (NCT\d+)
    # Option: include as identifier or exclude
    exclude_trial_ids: bool = False  # Set True to blacklist NCT04817618 etc.

    # ========================================
    # COMMON ENGLISH WORDS - Auto-reject
    # ========================================
    common_words: Set[str] = field(default_factory=lambda: {
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL',
        'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET',
        'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW',
        'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ANY',
        'DATA', 'WHITE', 'METHODS', 'STUDY', 'RESULTS', 'AGE',
        'YEARS', 'PATIENTS', 'TABLE', 'FIGURE', 'BASELINE',
    })

    # ========================================
    # ALLOWED SHORT SFs (2-letter exceptions)
    # ========================================
    allowed_2letter_sfs: Set[str] = field(default_factory=lambda: {
        'UK', 'US', 'EU', 'IV', 'IM', 'PO', 'SC', 'ID'  # Countries + routes
    })

    allowed_mixed_case: Set[str] = field(default_factory=lambda: {
        'MEDDRA', 'RADAR', 'SAS', 'SPSS', 'STATA'  # Software/registries
    })

    # ========================================
    # HYPHENATED ABBREVIATIONS (PASO C)
    # Often missed by standard generators
    # ========================================
    hyphenated_abbrevs: Dict[str, str] = field(default_factory=lambda: {
        'APPEAR-C3G': 'trial name',
        'CKD-EPI': 'Chronic Kidney Disease Epidemiology Collaboration',
        'FACIT-FATIGUE': 'Functional Assessment of Chronic Illness Therapy-Fatigue',
        'FACIT-Fatigue': 'Functional Assessment of Chronic Illness Therapy-Fatigue',
        'IL-6': 'interleukin-6',
        'IL-1': 'interleukin-1',
        'IL-5': 'interleukin-5',
        'TNF-α': 'tumor necrosis factor alpha',
        'sC5b-9': 'soluble terminal complement complex',
        'C5b-9': 'terminal complement complex',
        'IC-MPGN': 'immune complex membranoproliferative glomerulonephritis',
    })

    # ========================================
    # DIRECT SEARCH ABBREVIATIONS
    # Search in full text if not found by other generators
    # ========================================
    direct_search_abbrevs: Dict[str, str] = field(default_factory=lambda: {
        # Country codes (often not in medical lexicons)
        'UK': 'United Kingdom',
        # Stats that might be missed
        'RR': 'risk ratio',
        'HR': 'hazard ratio',
        # Multi-word
        'CC BY': 'Creative Commons Attribution',
        # Complement fragments
        'C3a': 'complement C3a anaphylatoxin',
        'C5a': 'complement C5a anaphylatoxin',
        # Brand names / special
        'DOI': 'digital object identifier',
        'FABHALTA': 'iptacopan (brand name)',
    })

    # ========================================
    # CONTEXTUAL RULES for ambiguous SFs
    # Option B: smarter filtering based on context
    # ========================================

    # SFs that need medical context to be valid
    context_required_sfs: Dict[str, Set[str]] = field(default_factory=lambda: {
        'IA': {'intramural', 'intra-arterial', 'ia nephropathy', 'igan'},  # Accept only if near these terms
        'IG': {'igg', 'iga', 'igm', 'ige', 'immunoglobulin'},  # Accept only if near immunoglobulin context
    })

    # SFs with case-sensitive matching requirements (targeted)
    # Only where canonical form differs significantly from common variants
    case_sensitive_sfs: Dict[str, str] = field(default_factory=lambda: {
        'SC5B9': 'sC5b-9',   # Soluble complement complex
        'SC5B-9': 'sC5b-9',
        'EGFR': 'eGFR',      # Estimated GFR (lowercase e)
    })

    # ========================================
    # LLM SF-ONLY EXTRACTOR (PASO D)
    # ========================================
    enable_llm_sf_extractor: bool = True
    llm_sf_max_chunks: int = 5
    llm_sf_chunk_size: int = 3000
    llm_sf_confidence: float = 0.75  # Lower confidence for LLM-extracted

    # ========================================
    # TIERED VALIDATION SETTINGS
    # ========================================
    enable_haiku_screening: bool = False
    batch_validation_size: int = 10
    lexicon_validation_batch_size: int = 15

    # Minimum occurrences for lexicon-only matches
    min_sf_occurrences: int = 2

    def eval_header(self, gold_file: str = "", gold_count: int = 0, scoring_mode: str = "sf_only_unique") -> str:
        """
        Generate one-line evaluation header for quick comparison.

        Args:
            gold_file: Name of gold file being used
            gold_count: Number of gold SFs in scope
            scoring_mode: 'sf_only_unique' or 'sf_occurrences'
        """
        trial_policy = "exclude" if self.exclude_trial_ids else "keep"
        blacklist_size = len(self.sf_blacklist)

        return (
            f"[EVAL] gold={gold_file} | mode={scoring_mode} | "
            f"trial_ids={trial_policy} | blacklist={blacklist_size} | "
            f"gold_sfs={gold_count}"
        )

    def __post_init__(self):
        """Validate config after initialization."""
        # Ensure all sets use uppercase for consistent matching
        self.sf_blacklist = {s.upper() for s in self.sf_blacklist}
        self.common_words = {s.upper() for s in self.common_words}
        self.allowed_2letter_sfs = {s.upper() for s in self.allowed_2letter_sfs}
        self.allowed_mixed_case = {s.upper() for s in self.allowed_mixed_case}

        # Ensure Dict keys are uppercase for consistent matching
        self.stats_abbrevs = {k.upper(): v for k, v in self.stats_abbrevs.items()}
        self.country_abbrevs = {k.upper(): v for k, v in self.country_abbrevs.items()}


@dataclass
class HeuristicsCounters:
    """
    Counters for tracking heuristics by category.
    Useful for debugging when corpus changes.
    """
    recovered_by_stats_whitelist: int = 0
    recovered_by_country_code: int = 0
    recovered_by_hyphen: int = 0
    recovered_by_direct_search: int = 0
    recovered_by_llm_sf_only: int = 0
    blacklisted_fp_count: int = 0
    common_word_rejected: int = 0
    context_rejected: int = 0
    form_filter_rejected: int = 0
    trial_id_excluded: int = 0

    def log_summary(self, logger: Optional[logging.Logger] = None):
        """Log summary of all counters."""
        summary = f"""
========================================
HEURISTICS COUNTERS SUMMARY
========================================
Auto-approved:
  - Stats whitelist:     {self.recovered_by_stats_whitelist}
  - Country codes:       {self.recovered_by_country_code}
  - Hyphenated:          {self.recovered_by_hyphen}
  - Direct search:       {self.recovered_by_direct_search}
  - LLM SF-only:         {self.recovered_by_llm_sf_only}

Auto-rejected:
  - Blacklist:           {self.blacklisted_fp_count}
  - Common words:        {self.common_word_rejected}
  - Context rejected:    {self.context_rejected}
  - Form filter:         {self.form_filter_rejected}
  - Trial IDs excluded:  {self.trial_id_excluded}
========================================
"""
        if logger:
            logger.info(summary)
        else:
            print(summary)

    def to_dict(self) -> Dict[str, int]:
        """Return counters as dict for JSON export."""
        return {
            'recovered_by_stats_whitelist': self.recovered_by_stats_whitelist,
            'recovered_by_country_code': self.recovered_by_country_code,
            'recovered_by_hyphen': self.recovered_by_hyphen,
            'recovered_by_direct_search': self.recovered_by_direct_search,
            'recovered_by_llm_sf_only': self.recovered_by_llm_sf_only,
            'blacklisted_fp_count': self.blacklisted_fp_count,
            'common_word_rejected': self.common_word_rejected,
            'context_rejected': self.context_rejected,
            'form_filter_rejected': self.form_filter_rejected,
            'trial_id_excluded': self.trial_id_excluded,
        }


# Default config instance (can be overridden)
DEFAULT_HEURISTICS_CONFIG = HeuristicsConfig()


def check_context_match(sf: str, context: str, config: HeuristicsConfig) -> bool:
    """
    Check if SF has required context to be valid.

    Returns True if:
    - SF is not in context_required_sfs, OR
    - SF is in context_required_sfs AND context contains required terms
    """
    sf_upper = sf.upper()

    if sf_upper not in config.context_required_sfs:
        return True  # No special context requirement

    required_terms = config.context_required_sfs[sf_upper]
    ctx_lower = context.lower()

    return any(term in ctx_lower for term in required_terms)


def check_trial_id(sf: str, config: HeuristicsConfig) -> bool:
    r"""
    Check if SF is a trial ID (NCT\d+) and should be excluded.

    Returns True if SF should be EXCLUDED (is a trial ID and exclude_trial_ids=True).
    """
    import re

    if not config.exclude_trial_ids:
        return False  # Don't exclude trial IDs

    # Match NCT followed by digits (ClinicalTrials.gov identifiers)
    return bool(re.match(r'^NCT\d+$', sf.upper()))


def get_canonical_case(sf: str, config: HeuristicsConfig) -> str:
    """
    Get the canonical case for a case-sensitive SF.

    E.g., SC5B9 -> sC5b-9
    """
    sf_upper = sf.upper()
    return config.case_sensitive_sfs.get(sf_upper, sf)
