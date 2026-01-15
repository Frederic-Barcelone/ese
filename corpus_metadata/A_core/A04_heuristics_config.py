# corpus_metadata/corpus_metadata/A_core/A04_heuristics_config.py
"""
Centralized Heuristics Configuration

All whitelists, blacklists, and configurable rules in one place.
Enables easy tuning without modifying pipeline logic.

IMPORTANT: All values are now loaded from config.yaml.
The hardcoded defaults are only used as fallback.

Categories tracked for logging:
- recovered_by_stats_whitelist
- recovered_by_hyphen
- recovered_by_llm_sf_only
- blacklisted_fp_count
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Set, Optional
import logging
import os
import re
import unicodedata
import yaml


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
    stats_abbrevs: Dict[str, str] = field(
        default_factory=lambda: {
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
    )

    # ========================================
    # COUNTRY CODES (PASO B) - DISABLED (moved to blacklist)
    # These are valid abbreviations but not domain-specific for clinical NLP.
    # Keeping empty dict for backward compatibility with code that references it.
    # ========================================
    country_abbrevs: Dict[str, str] = field(
        default_factory=lambda: {}
    )

    # ========================================
    # BLACKLIST - Auto-reject (SURGICAL: keep small and targeted)
    # Only include clear FPs that are never valid medical abbreviations
    # ========================================
    sf_blacklist: Set[str] = field(
        default_factory=lambda: {
            # ----------------------------------------
            # Country codes - SAFE (no medical conflict)
            # NOT included: AT, CA, HR, IL, IN, IT, MG, MS, NO, PE, PT, SA, SC, SE, TR
            # (these conflict with medical abbreviations)
            # ----------------------------------------
            # Major regions
            "US", "UK", "USA", "EU",
            # Europe (safe codes only)
            "FR",   # France
            "DE",   # Germany
            "ES",   # Spain
            "NL",   # Netherlands
            "BE",   # Belgium
            "PL",   # Poland
            "CZ",   # Czech Republic
            "DK",   # Denmark
            "FI",   # Finland
            "IE",   # Ireland
            "GR",   # Greece
            "HU",   # Hungary
            "RO",   # Romania
            "BG",   # Bulgaria
            "SK",   # Slovakia
            "SI",   # Slovenia
            "LT",   # Lithuania
            "LV",   # Latvia
            "EE",   # Estonia
            "CY",   # Cyprus
            "MT",   # Malta
            "LU",   # Luxembourg
            "RU",   # Russia
            "UA",   # Ukraine
            # Asia-Pacific (safe codes only)
            "JP",   # Japan
            "CN",   # China
            "KR",   # South Korea
            "TW",   # Taiwan
            "SG",   # Singapore
            "HK",   # Hong Kong
            "TH",   # Thailand
            "MY",   # Malaysia
            "VN",   # Vietnam
            "PK",   # Pakistan
            "AU",   # Australia
            "NZ",   # New Zealand
            # Americas (safe codes only)
            "BR",   # Brazil
            "MX",   # Mexico
            "AR",   # Argentina
            "CL",   # Chile
            "CO",   # Colombia
            # Africa/Middle East (safe codes only)
            "ZA",   # South Africa
            "EG",   # Egypt
            # 3-letter ISO codes (commonly seen)
            "GBR",  # Great Britain
            "FRA",  # France
            "DEU",  # Germany
            "JPN",  # Japan
            "CHN",  # China
            "AUS",  # Australia
            "ESP",  # Spain
            "NLD",  # Netherlands
            "BRA",  # Brazil
            "MEX",  # Mexico
            "KOR",  # South Korea
            "RUS",  # Russia
            # Author credentials (never medical abbreviations in context)
            "MD",
            "PHD",
            "MBBS",
            "FRCP",
            "MPH",
            # US state abbreviations (location context, not medical)
            "NY",
            "NJ",
            # UK postal code areas
            "NE",   # North East England (Newcastle)
            # Ambiguous 2-letter that are rarely medical
            "IA",  # Iowa, intramural - ambiguous without strong context
            # Common words/months that get lexicon matches
            "AUG",  # August (month)
            "INT",  # International - not a medical abbreviation
            # Research/tool names that aren't medical abbreviations
            "IRCCS",  # Italian research institute
            "CC BY",  # Creative Commons license
            # Database identifiers - these are references, not abbreviations
            "OMIM",  # Online Mendelian Inheritance in Man
            "MIM",  # Alternative OMIM prefix
            "ORPHA",  # Orphanet database
            "ORPHANET",  # Orphanet database
            "PMID",  # PubMed ID
            "PMC",  # PubMed Central
            "MESH",  # Medical Subject Headings
            "SNOMED",  # SNOMED CT
            "MONDO",  # Mondo Disease Ontology
            "ORCID",  # Researcher ID
            "ISRCTN",  # Trial registry
        }
    )

    # Trial IDs - special handling (NCT\d+)
    # Option: include as identifier or exclude
    exclude_trial_ids: bool = False  # Set True to blacklist NCT04817618 etc.

    # ========================================
    # COMMON ENGLISH WORDS - Auto-reject
    # Organized by category for maintainability
    # ========================================
    common_words: Set[str] = field(
        default_factory=lambda: {
            # Short common words (3 letters)
            "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN",
            "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM",
            "HIS", "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "SEE", "TWO",
            "WAY", "WHO", "BOY", "DID", "ANY", "AGE",
            # Paper section headers (common FPs in academic text)
            "ABSTRACT", "INTRODUCTION", "METHODS", "RESULTS", "DISCUSSION",
            "CONCLUSION", "CONCLUSIONS", "REFERENCES", "ACKNOWLEDGMENTS",
            "SUPPLEMENTARY", "APPENDIX", "BACKGROUND",
            # Table/figure labels
            "TABLE", "FIGURE", "FIG",
            # Common research terms (not abbreviations)
            "DATA", "STUDY", "YEARS", "PATIENTS", "BASELINE", "WHITE",
        }
    )

    # ========================================
    # ALLOWED SHORT SFs (2-letter exceptions)
    # ========================================
    allowed_2letter_sfs: Set[str] = field(
        default_factory=lambda: {
            "UK",
            "US",
            "EU",
            "IV",
            "IM",
            "PO",
            "SC",
            "ID",  # Countries + routes
        }
    )

    # Allowed non-medical tokens: don't reject these based on form/case
    # but also don't auto-approve as medical abbreviations
    # Categories: registries (MEDDRA), software (SAS, SPSS, STATA), acronyms (RADAR)
    allowed_mixed_case: Set[str] = field(
        default_factory=lambda: {
            # Medical registries (not abbreviations but valid identifiers)
            "MEDDRA",
            # Statistical software (common in methods sections)
            "SAS",
            "SPSS",
            "STATA",
            # Other acronyms
            "RADAR",
        }
    )

    # ========================================
    # HYPHENATED ABBREVIATIONS (PASO C)
    # Often missed by standard generators
    # Keys are CANONICAL forms; matching uses normalized comparison
    # ========================================
    hyphenated_abbrevs: Dict[str, str] = field(
        default_factory=lambda: {
            # Trial names
            "APPEAR-C3G": "trial name",
            # Clinical scores/tools
            "CKD-EPI": "Chronic Kidney Disease Epidemiology Collaboration",
            "FACIT-Fatigue": "Functional Assessment of Chronic Illness Therapy-Fatigue",
            # Interleukins
            "IL-1": "interleukin-1",
            "IL-5": "interleukin-5",
            "IL-6": "interleukin-6",
            # Cytokines (canonical form with Greek alpha)
            "TNF-α": "tumor necrosis factor alpha",
            # Complement system
            "C5b-9": "terminal complement complex",
            "sC5b-9": "soluble terminal complement complex",
            # Disease classifications
            "IC-MPGN": "immune complex membranoproliferative glomerulonephritis",
        }
    )

    # ========================================
    # DIRECT SEARCH ABBREVIATIONS
    # Search in full text if not found by other generators
    # ========================================
    direct_search_abbrevs: Dict[str, str] = field(
        default_factory=lambda: {
            # Country codes (often not in medical lexicons)
            "UK": "United Kingdom",
            # Stats that might be missed
            "RR": "risk ratio",
            "HR": "hazard ratio",
            # CC BY removed - license identifier, not medical abbreviation
            # Complement fragments
            "C3a": "complement C3a anaphylatoxin",
            "C5a": "complement C5a anaphylatoxin",
            # Brand names / special
            "DOI": "digital object identifier",
            "FABHALTA": "iptacopan (brand name)",
        }
    )

    # ========================================
    # CONTEXTUAL RULES for ambiguous SFs
    # Option B: smarter filtering based on context
    # ========================================

    # SFs that need medical context to be valid
    context_required_sfs: Dict[str, Set[str]] = field(
        default_factory=lambda: {
            "IA": {
                "intramural",
                "intra-arterial",
                "ia nephropathy",
                "igan",
            },  # Accept only if near these terms
            "IG": {
                "igg",
                "iga",
                "igm",
                "ige",
                "immunoglobulin",
            },  # Accept only if near immunoglobulin context
        }
    )

    # SFs with case-sensitive matching requirements (targeted)
    # Only where canonical form differs significantly from common variants
    case_sensitive_sfs: Dict[str, str] = field(
        default_factory=lambda: {
            "SC5B9": "sC5b-9",  # Soluble complement complex
            "SC5B-9": "sC5b-9",
            "EGFR": "eGFR",  # Estimated GFR (lowercase e)
        }
    )

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

    def eval_header(
        self,
        gold_file: str = "",
        gold_count: int = 0,
        scoring_mode: str = "sf_only_unique",
    ) -> str:
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
        """Validate config after initialization and normalize all keys."""
        # Normalize all sets using normalize_sf_key for consistent matching
        # This handles Unicode, hyphens, whitespace, and case
        self.sf_blacklist = {normalize_sf_key(s) for s in self.sf_blacklist}
        self.common_words = {normalize_sf_key(s) for s in self.common_words}
        self.allowed_2letter_sfs = {normalize_sf_key(s) for s in self.allowed_2letter_sfs}
        self.allowed_mixed_case = {normalize_sf_key(s) for s in self.allowed_mixed_case}

        # Normalize Dict keys for consistent matching
        self.stats_abbrevs = {normalize_sf_key(k): v for k, v in self.stats_abbrevs.items()}
        self.country_abbrevs = {normalize_sf_key(k): v for k, v in self.country_abbrevs.items()}
        self.hyphenated_abbrevs = {normalize_sf_key(k): v for k, v in self.hyphenated_abbrevs.items()}
        self.direct_search_abbrevs = {normalize_sf_key(k): v for k, v in self.direct_search_abbrevs.items()}
        self.case_sensitive_sfs = {normalize_sf_key(k): v for k, v in self.case_sensitive_sfs.items()}
        self.context_required_sfs = {normalize_sf_key(k): v for k, v in self.context_required_sfs.items()}

    @classmethod
    def from_yaml(cls, config_path: str) -> "HeuristicsConfig":
        """
        Load HeuristicsConfig from a YAML config file.

        Args:
            config_path: Path to the config.yaml file

        Returns:
            HeuristicsConfig instance with values from YAML
        """
        path = Path(config_path)
        if not path.exists():
            print(f"[WARN] Config file not found: {config_path}, using defaults")
            return cls()

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            print(
                f"[WARN] Failed to load config from {config_path}: {e}, using defaults"
            )
            return cls()

        heur = data.get("heuristics", {})
        if not heur:
            return cls()

        # Helper to convert list to set with type validation
        def to_set(val: Any, key: str) -> Set[str]:
            if val is None:
                return set()
            if isinstance(val, list):
                return set(val)
            if isinstance(val, set):
                return val
            print(f"[WARN] heuristics.{key}: expected list/set, got {type(val).__name__}, using empty set")
            return set()

        # Helper to convert dict with list values to dict with set values
        def to_dict_of_sets(val: Any, key: str) -> Dict[str, Set[str]]:
            if val is None:
                return {}
            if isinstance(val, dict):
                return {k: set(v) if isinstance(v, list) else v for k, v in val.items()}
            print(f"[WARN] heuristics.{key}: expected dict, got {type(val).__name__}, using empty dict")
            return {}

        # Build kwargs from YAML
        kwargs: Dict[str, Any] = {}

        # Stats abbrevs (dict)
        if "stats_abbrevs" in heur:
            kwargs["stats_abbrevs"] = heur["stats_abbrevs"]

        # Country abbrevs (dict)
        if "country_abbrevs" in heur:
            kwargs["country_abbrevs"] = heur["country_abbrevs"]

        # Blacklist (set)
        if "sf_blacklist" in heur:
            kwargs["sf_blacklist"] = to_set(heur["sf_blacklist"], "sf_blacklist")

        # Common words (set)
        if "common_words" in heur:
            kwargs["common_words"] = to_set(heur["common_words"], "common_words")

        # Allowed 2-letter SFs (set)
        if "allowed_2letter_sfs" in heur:
            kwargs["allowed_2letter_sfs"] = to_set(heur["allowed_2letter_sfs"], "allowed_2letter_sfs")

        # Allowed mixed case (set)
        if "allowed_mixed_case" in heur:
            kwargs["allowed_mixed_case"] = to_set(heur["allowed_mixed_case"], "allowed_mixed_case")

        # Hyphenated abbrevs (dict)
        if "hyphenated_abbrevs" in heur:
            kwargs["hyphenated_abbrevs"] = heur["hyphenated_abbrevs"]

        # Direct search abbrevs (dict)
        if "direct_search_abbrevs" in heur:
            kwargs["direct_search_abbrevs"] = heur["direct_search_abbrevs"]

        # Context required SFs (dict of sets)
        if "context_required_sfs" in heur:
            kwargs["context_required_sfs"] = to_dict_of_sets(
                heur["context_required_sfs"], "context_required_sfs"
            )

        # Case sensitive SFs (dict)
        if "case_sensitive_sfs" in heur:
            kwargs["case_sensitive_sfs"] = heur["case_sensitive_sfs"]

        # Boolean flags
        if "exclude_trial_ids" in heur:
            kwargs["exclude_trial_ids"] = bool(heur["exclude_trial_ids"])
        if "enable_llm_sf_extractor" in heur:
            kwargs["enable_llm_sf_extractor"] = bool(heur["enable_llm_sf_extractor"])
        if "enable_haiku_screening" in heur:
            kwargs["enable_haiku_screening"] = bool(heur["enable_haiku_screening"])

        # Numeric values
        if "llm_sf_max_chunks" in heur:
            kwargs["llm_sf_max_chunks"] = int(heur["llm_sf_max_chunks"])
        if "llm_sf_chunk_size" in heur:
            kwargs["llm_sf_chunk_size"] = int(heur["llm_sf_chunk_size"])
        if "llm_sf_confidence" in heur:
            kwargs["llm_sf_confidence"] = float(heur["llm_sf_confidence"])
        if "batch_validation_size" in heur:
            kwargs["batch_validation_size"] = int(heur["batch_validation_size"])
        if "lexicon_validation_batch_size" in heur:
            kwargs["lexicon_validation_batch_size"] = int(
                heur["lexicon_validation_batch_size"]
            )
        if "min_sf_occurrences" in heur:
            kwargs["min_sf_occurrences"] = int(heur["min_sf_occurrences"])

        return cls(**kwargs)


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
            "recovered_by_stats_whitelist": self.recovered_by_stats_whitelist,
            "recovered_by_country_code": self.recovered_by_country_code,
            "recovered_by_hyphen": self.recovered_by_hyphen,
            "recovered_by_direct_search": self.recovered_by_direct_search,
            "recovered_by_llm_sf_only": self.recovered_by_llm_sf_only,
            "blacklisted_fp_count": self.blacklisted_fp_count,
            "common_word_rejected": self.common_word_rejected,
            "context_rejected": self.context_rejected,
            "form_filter_rejected": self.form_filter_rejected,
            "trial_id_excluded": self.trial_id_excluded,
        }


# Default config path: prefer env var, fallback to relative path from this file
# Set ESE_CONFIG_PATH env var for custom config location
DEFAULT_CONFIG_PATH = os.getenv(
    "ESE_CONFIG_PATH",
    str(Path(__file__).resolve().parents[1] / "G_config" / "config.yaml")
)


def load_default_heuristics_config() -> HeuristicsConfig:
    """Load heuristics config from default YAML path, fallback to hardcoded defaults."""
    return HeuristicsConfig.from_yaml(DEFAULT_CONFIG_PATH)


# Default config instance (loaded from config.yaml)
DEFAULT_HEURISTICS_CONFIG = load_default_heuristics_config()


def check_context_match(sf: str, context: str, config: HeuristicsConfig) -> bool:
    """
    Check if SF has required context to be valid.

    Returns True if:
    - SF is not in context_required_sfs, OR
    - SF is in context_required_sfs AND context contains required terms

    Uses normalized comparison for both SF and context to handle
    Unicode variants, hyphens, etc.
    """
    sf_key = normalize_sf_key(sf)

    if sf_key not in config.context_required_sfs:
        return True  # No special context requirement

    required_terms = config.context_required_sfs[sf_key]
    ctx_normalized = normalize_context(context)

    return any(term in ctx_normalized for term in required_terms)


def check_trial_id(sf: str, config: HeuristicsConfig) -> bool:
    r"""
    Check if SF is a trial ID (NCT\d+) and should be excluded.

    Returns True if SF should be EXCLUDED (is a trial ID and exclude_trial_ids=True).
    """
    if not config.exclude_trial_ids:
        return False  # Don't exclude trial IDs

    # Match NCT followed by digits (ClinicalTrials.gov identifiers)
    sf_normalized = normalize_sf_key(sf)
    return bool(re.match(r"^NCT\d+$", sf_normalized))


def get_canonical_case(sf: str, config: HeuristicsConfig) -> str:
    """
    Get the canonical case for a case-sensitive SF.

    E.g., SC5B9 -> sC5b-9
    """
    sf_key = normalize_sf_key(sf)
    return config.case_sensitive_sfs.get(sf_key, sf)
