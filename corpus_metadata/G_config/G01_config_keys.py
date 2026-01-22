# corpus_metadata/G_config/G01_config_keys.py
"""
Configuration key constants for the ESE pipeline.

Provides type-safe configuration key enums that:
- Prevent typos in configuration keys
- Enable IDE autocomplete
- Document default values
- Centralize configuration schema

Usage:
    from G_config.G01_config_keys import ConfigKey, CacheConfig, ParserConfig

    # Instead of: config.get("timeout_seconds", 30)
    # Use: config.get(ConfigKey.TIMEOUT_SECONDS, ConfigKey.TIMEOUT_SECONDS.default)

    # For nested config:
    cache_dir = config.get("cache", {}).get(CacheConfig.DIRECTORY, CacheConfig.DIRECTORY.default)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class ConfigKeyBase(str, Enum):
    """
    Base class for configuration key enums.

    Inherits from str to allow direct use as dictionary keys.
    """

    _default: Any
    _description: str

    def __new__(cls, key: str, default: Any = None, description: str = "") -> "ConfigKeyBase":
        """Create enum member with key, default value, and description."""
        obj = str.__new__(cls, key)
        obj._value_ = key
        obj._default = default
        obj._description = description
        return obj

    @property
    def default(self) -> Any:
        """Get the default value for this config key."""
        return self._default

    @property
    def description(self) -> str:
        """Get the description for this config key."""
        return self._description


class ConfigKey(ConfigKeyBase):
    """
    Top-level configuration keys.

    These are the primary keys in the root of the configuration dictionary.
    """

    # Common metadata
    RUN_ID = ("run_id", None, "Unique identifier for the current run")
    PIPELINE_VERSION = ("pipeline_version", None, "Pipeline version (git hash)")
    DOC_FINGERPRINT = ("doc_fingerprint", "unknown-doc-fingerprint", "Document fingerprint")
    TIMESTAMP = ("timestamp", "", "Processing timestamp")

    # Processing settings
    CONTEXT_WINDOW = ("context_window", 300, "Characters of context around entities")
    ENABLED = ("enabled", True, "Enable/disable the component")
    DEVICE = ("device", -1, "Device for ML models (-1=CPU, 0+=GPU)")
    CONFIDENCE_THRESHOLD = ("confidence_threshold", 0.5, "Minimum confidence score")

    # API settings
    BASE_URL = ("base_url", "", "API base URL")
    TIMEOUT_SECONDS = ("timeout_seconds", 30, "Request timeout in seconds")
    RATE_LIMIT_PER_SECOND = ("rate_limit_per_second", 1.0, "Max requests per second")

    # Nested config sections
    PATHS = ("paths", {}, "File and directory paths")
    API = ("api", {}, "API configuration")
    HEURISTICS = ("heuristics", {}, "Heuristics configuration")
    EXTRACTION_PIPELINE = ("extraction_pipeline", {}, "Extraction pipeline settings")
    LEXICONS = ("lexicons", {}, "Lexicon paths")
    GENERATORS = ("generators", {}, "Generator settings")
    LLM = ("llm", {}, "LLM configuration")
    NORMALIZATION = ("normalization", {}, "Normalization settings")
    DOCUMENT_METADATA = ("document_metadata", {}, "Document metadata extraction")
    NCT_ENRICHER = ("nct_enricher", {}, "NCT enricher configuration")
    CACHE = ("cache", {}, "Cache configuration")
    ENRICHMENT = ("enrichment", {}, "Enrichment settings")


class CacheConfig(ConfigKeyBase):
    """
    Cache-related configuration keys.

    Used within the 'cache' configuration section.
    """

    ENABLED = ("enabled", True, "Enable disk caching")
    DIRECTORY = ("directory", "cache", "Cache directory path")
    TTL_HOURS = ("ttl_hours", 24, "Cache time-to-live in hours")
    TTL_DAYS = ("ttl_days", 30, "Cache time-to-live in days")


class ParserConfig(ConfigKeyBase):
    """
    PDF parsing configuration keys.

    Used for B_parsing module configuration.
    """

    # Unstructured settings
    EXTRACTION_METHOD = ("extraction_method", "unstructured", "PDF extraction method")
    UNSTRUCTURED_STRATEGY = ("unstructured_strategy", "hi_res", "Unstructured.io strategy")
    HI_RES_MODEL_NAME = ("hi_res_model_name", "yolox", "High-res model name")
    FORCE_FAST = ("force_fast", False, "Force fast extraction mode")
    FORCE_HF_OFFLINE = ("force_hf_offline", False, "Force HuggingFace offline mode")

    # Table extraction
    INFER_TABLE_STRUCTURE = ("infer_table_structure", True, "Infer table structure")
    STRATEGY = ("strategy", "hi_res", "Table extraction strategy")

    # Image extraction
    EXTRACT_IMAGES_IN_PDF = ("extract_images_in_pdf", True, "Extract images from PDF")
    EXTRACT_IMAGE_BLOCK_TO_PAYLOAD = ("extract_image_block_to_payload", True, "Include image blocks")
    EXTRACT_IMAGE_BLOCK_TYPES = ("extract_image_block_types", ["Image", "Figure"], "Image block types")

    # Layout settings
    USE_SOTA_LAYOUT = ("use_sota_layout", True, "Use SOTA layout analysis")
    LAYOUT_CONFIG = ("layout_config", None, "Layout configuration")
    DOCUMENT_TYPE = ("document_type", "academic", "Document type hint")
    LANGUAGES = ("languages", ["eng"], "OCR languages")
    INCLUDE_PAGE_BREAKS = ("include_page_breaks", True, "Include page break markers")
    DROP_CATEGORIES = ("drop_categories", [], "Element categories to drop")

    # Geometry tolerances
    Y_TOLERANCE = ("y_tolerance", 3.0, "Vertical alignment tolerance")
    HEADER_TOP_PCT = ("header_top_pct", 0.07, "Header zone percentage")
    FOOTER_BOTTOM_PCT = ("footer_bottom_pct", 0.90, "Footer zone start percentage")
    MIN_REPEAT_COUNT = ("min_repeat_count", 3, "Min repeated element count")
    MIN_REPEAT_PAGES = ("min_repeat_pages", 3, "Min pages for repeat detection")
    REPEAT_ZONE_MAJORITY = ("repeat_zone_majority", 0.60, "Repeat zone threshold")
    TWO_COL_MIN_SIDE_BLOCKS = ("two_col_min_side_blocks", 6, "Min blocks for two-column")


class GeneratorConfig(ConfigKeyBase):
    """
    Candidate generator configuration keys.

    Used for C_generators module configuration.
    """

    # Identifier extraction
    ENABLED_TYPES = ("enabled_types", [], "Enabled identifier types")
    EXTRACT_GENES = ("extract_genes", True, "Extract gene identifiers")

    # Abbreviation extraction
    MIN_SF_LENGTH = ("min_sf_length", 2, "Minimum short form length")
    MAX_SF_LENGTH = ("max_sf_length", 10, "Maximum short form length")
    CONTEXT_WINDOW_CHARS = ("context_window_chars", 400, "Context window in characters")
    CARRYOVER_CHARS = ("carryover_chars", 140, "Carryover characters between blocks")
    MAX_CANDIDATES_PER_BLOCK = ("max_candidates_per_block", 200, "Max candidates per block")

    # Layout extraction
    ZONE_MARGIN = ("zone_margin", 0.15, "Zone margin from page edges")
    MAX_VALUE_DISTANCE = ("max_value_distance", 200, "Max distance to value")
    MAX_VALUE_LENGTH = ("max_value_length", 100, "Max value text length")
    ROW_TOLERANCE = ("row_tolerance", 6.0, "Row alignment tolerance")
    MAX_SF_LEN = ("max_sf_len", 15, "Max short form length for layout")

    # Regex settings
    CTX_WINDOW = ("ctx_window", 60, "Regex context window")
    DEDUPE = ("dedupe", True, "Deduplicate results")

    # Table settings
    MAX_ROWS_PER_TABLE = ("max_rows_per_table", 500, "Max rows per table")

    # Lexicon settings
    LEXICON_BASE_PATH = ("lexicon_base_path", "ouput_datasources", "Base path for lexicons")


class EnricherConfig(ConfigKeyBase):
    """
    Enricher configuration keys.

    Used for E_normalization module configuration.
    """

    # Common settings
    RUN_ID = ("run_id", "unknown", "Run identifier")
    CONFIDENCE_THRESHOLD = ("confidence_threshold", 0.5, "Minimum confidence")
    CONTEXT_WINDOW = ("context_window", 150, "Context window size")

    # NER settings
    DEVICE = ("device", -1, "ML device (-1=CPU)")
    BATCH_SIZE = ("batch_size", 8, "Batch size for inference")
    ENTITY_TYPES = ("entity_types", [], "Entity types to extract")
    ENTITY_LABELS = ("entity_labels", [], "Entity labels to use")

    # Disambiguation
    MIN_CONTEXT_SCORE = ("min_context_score", 2, "Minimum context score")
    MIN_MARGIN = ("min_margin", 1, "Minimum disambiguation margin")
    ONLY_VALIDATED = ("only_validated", True, "Only process validated entities")
    FILL_LONG_FORM_FOR_ORPHANS = ("fill_long_form_for_orphans", True, "Fill orphan long forms")
    LOWERCASE = ("lowercase", True, "Lowercase for matching")
    AMBIGUITY_MAP = ("ambiguity_map", {}, "Custom ambiguity mappings")

    # Deduplication
    OVERLAP_THRESHOLD = ("overlap_threshold", 0.5, "Span overlap threshold")
    MIN_CONFIDENCE = ("min_confidence", 0.3, "Minimum confidence for spans")
    CONFIDENCE_MARGIN = ("confidence_margin", 0.1, "Confidence margin for dedup")
    STORE_ALTERNATIVES = ("store_alternatives", True, "Store alternative matches")
    MAX_ALTERNATIVES = ("max_alternatives", 5, "Max alternative matches")

    # Term mapping
    MAPPING_FILE = ("mapping_file", None, "Path to mapping file")
    ENABLE_FUZZY_MATCHING = ("enable_fuzzy_matching", False, "Enable fuzzy matching")
    FUZZY_CUTOFF = ("fuzzy_cutoff", 0.90, "Fuzzy match cutoff score")

    # Citation validation
    TIMEOUT = ("timeout", 10, "Request timeout")
    DELAY = ("delay", 0.5, "Delay between requests")
    VALIDATE_URLS = ("validate_urls", True, "Validate citation URLs")

    # Patient journey
    USE_ZEROSHOT_FALLBACK = ("use_zeroshot_fallback", False, "Use zero-shot fallback")

    # Registry enrichment
    LINK_REGISTRIES = ("link_registries", True, "Link to external registries")

    # Genetic enrichment
    VALIDATE_GENES = ("validate_genes", True, "Validate gene symbols")
    EXTRACT_HPO = ("extract_hpo", True, "Extract HPO terms")
    EXTRACT_ORDO = ("extract_ordo", True, "Extract Orphanet codes")
    ADDITIONAL_GENES = ("additional_genes", set(), "Additional gene symbols")

    # Drug enrichment
    ENRICH_MISSING_MESH = ("enrich_missing_mesh", True, "Enrich missing MeSH IDs")
    ADD_ALIASES = ("add_aliases", True, "Add aliases from lookup")


class PipelineConfig(ConfigKeyBase):
    """
    Pipeline extraction configuration keys.

    Used within the 'extraction_pipeline' section.
    """

    # Extractors toggles
    EXTRACTORS = ("extractors", {}, "Extractor enable/disable flags")

    # Options
    OPTIONS = ("options", {}, "Pipeline options")
    USE_LLM_VALIDATION = ("use_llm_validation", True, "Use LLM for validation")
    USE_LLM_FEASIBILITY = ("use_llm_feasibility", True, "Use LLM for feasibility")
    USE_VLM_TABLES = ("use_vlm_tables", False, "Use VLM for tables")
    USE_NORMALIZATION = ("use_normalization", True, "Enable normalization")


class LLMConfig(ConfigKeyBase):
    """
    LLM configuration keys.

    Used within the 'llm' section.
    """

    MODEL = ("model", "claude-sonnet-4-20250514", "LLM model identifier")
    TEMPERATURE = ("temperature", 0.0, "LLM temperature")
    MAX_TOKENS = ("max_tokens", 4096, "Maximum output tokens")


# Helper functions for type-safe config access

def get_config(
    config: Dict[str, Any],
    key: ConfigKeyBase,
    default: Optional[Any] = None,
) -> Any:
    """
    Get a configuration value with type-safe key.

    Args:
        config: Configuration dictionary.
        key: Configuration key enum.
        default: Override default value.

    Returns:
        Configuration value or default.
    """
    return config.get(key.value, default if default is not None else key.default)


def get_nested_config(
    config: Dict[str, Any],
    *keys: ConfigKeyBase,
    default: Optional[Any] = None,
) -> Any:
    """
    Get a nested configuration value.

    Args:
        config: Root configuration dictionary.
        *keys: Sequence of config keys to traverse.
        default: Default value if not found.

    Returns:
        Nested configuration value or default.

    Example:
        >>> get_nested_config(config, ConfigKey.CACHE, CacheConfig.TTL_HOURS)
        24
    """
    result = config
    for key in keys[:-1]:
        result = result.get(key.value, {})
        if not isinstance(result, dict):
            return default if default is not None else keys[-1].default

    final_key = keys[-1]
    return result.get(final_key.value, default if default is not None else final_key.default)


# Export all config classes
__all__ = [
    "ConfigKey",
    "CacheConfig",
    "ParserConfig",
    "GeneratorConfig",
    "EnricherConfig",
    "PipelineConfig",
    "LLMConfig",
    "get_config",
    "get_nested_config",
]
