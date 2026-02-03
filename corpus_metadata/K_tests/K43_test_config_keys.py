# corpus_metadata/K_tests/K43_test_config_keys.py
"""
Tests for G_config.G01_config_keys module.

Tests configuration key enums and helper functions.
"""

from __future__ import annotations

from typing import Any

from G_config.G01_config_keys import (
    ConfigKey,
    CacheConfig,
    ParserConfig,
    GeneratorConfig,
    EnricherConfig,
    PipelineConfig,
    LLMConfig,
    get_config,
    get_nested_config,
)


class TestConfigKeyBase:
    """Tests for ConfigKeyBase enum."""

    def test_enum_is_string(self):
        """Config keys should be usable as strings."""
        assert isinstance(ConfigKey.RUN_ID, str)
        assert ConfigKey.RUN_ID == "run_id"

    def test_default_property(self):
        """Config keys should have default values."""
        assert ConfigKey.CONTEXT_WINDOW.default == 300
        assert ConfigKey.ENABLED.default is True
        assert ConfigKey.DEVICE.default == -1

    def test_description_property(self):
        """Config keys should have descriptions."""
        assert ConfigKey.RUN_ID.description == "Unique identifier for the current run"
        assert "timeout" in ConfigKey.TIMEOUT_SECONDS.description.lower()


class TestConfigKey:
    """Tests for top-level ConfigKey enum."""

    def test_common_metadata_keys(self):
        assert ConfigKey.RUN_ID.value == "run_id"
        assert ConfigKey.PIPELINE_VERSION.value == "pipeline_version"
        assert ConfigKey.DOC_FINGERPRINT.value == "doc_fingerprint"

    def test_processing_settings_keys(self):
        assert ConfigKey.CONTEXT_WINDOW.value == "context_window"
        assert ConfigKey.ENABLED.value == "enabled"
        assert ConfigKey.CONFIDENCE_THRESHOLD.value == "confidence_threshold"

    def test_api_settings_keys(self):
        assert ConfigKey.BASE_URL.value == "base_url"
        assert ConfigKey.TIMEOUT_SECONDS.value == "timeout_seconds"
        assert ConfigKey.RATE_LIMIT_PER_SECOND.value == "rate_limit_per_second"

    def test_nested_config_section_keys(self):
        assert ConfigKey.PATHS.value == "paths"
        assert ConfigKey.API.value == "api"
        assert ConfigKey.LEXICONS.value == "lexicons"
        assert ConfigKey.CACHE.value == "cache"


class TestCacheConfig:
    """Tests for CacheConfig enum."""

    def test_cache_keys(self):
        assert CacheConfig.ENABLED.value == "enabled"
        assert CacheConfig.DIRECTORY.value == "directory"
        assert CacheConfig.TTL_HOURS.value == "ttl_hours"
        assert CacheConfig.TTL_DAYS.value == "ttl_days"

    def test_cache_defaults(self):
        assert CacheConfig.ENABLED.default is True
        assert CacheConfig.DIRECTORY.default == "cache"
        assert CacheConfig.TTL_HOURS.default == 24
        assert CacheConfig.TTL_DAYS.default == 30


class TestParserConfig:
    """Tests for ParserConfig enum."""

    def test_extraction_keys(self):
        assert ParserConfig.EXTRACTION_METHOD.value == "extraction_method"
        assert ParserConfig.UNSTRUCTURED_STRATEGY.value == "unstructured_strategy"

    def test_table_extraction_keys(self):
        assert ParserConfig.INFER_TABLE_STRUCTURE.value == "infer_table_structure"
        assert ParserConfig.STRATEGY.value == "strategy"

    def test_layout_keys(self):
        assert ParserConfig.USE_SOTA_LAYOUT.value == "use_sota_layout"
        assert ParserConfig.DOCUMENT_TYPE.value == "document_type"

    def test_geometry_tolerance_keys(self):
        assert ParserConfig.Y_TOLERANCE.value == "y_tolerance"
        assert ParserConfig.Y_TOLERANCE.default == 3.0
        assert ParserConfig.HEADER_TOP_PCT.default == 0.07


class TestGeneratorConfig:
    """Tests for GeneratorConfig enum."""

    def test_abbreviation_keys(self):
        assert GeneratorConfig.MIN_SF_LENGTH.value == "min_sf_length"
        assert GeneratorConfig.MAX_SF_LENGTH.value == "max_sf_length"
        assert GeneratorConfig.MIN_SF_LENGTH.default == 2
        assert GeneratorConfig.MAX_SF_LENGTH.default == 10

    def test_layout_keys(self):
        assert GeneratorConfig.ZONE_MARGIN.value == "zone_margin"
        assert GeneratorConfig.ZONE_MARGIN.default == 0.15

    def test_lexicon_keys(self):
        assert GeneratorConfig.LEXICON_BASE_PATH.value == "lexicon_base_path"


class TestEnricherConfig:
    """Tests for EnricherConfig enum."""

    def test_common_settings(self):
        assert EnricherConfig.RUN_ID.value == "run_id"
        assert EnricherConfig.CONFIDENCE_THRESHOLD.value == "confidence_threshold"
        assert EnricherConfig.CONFIDENCE_THRESHOLD.default == 0.5

    def test_ner_settings(self):
        assert EnricherConfig.DEVICE.value == "device"
        assert EnricherConfig.BATCH_SIZE.value == "batch_size"
        assert EnricherConfig.BATCH_SIZE.default == 8

    def test_disambiguation_settings(self):
        assert EnricherConfig.MIN_CONTEXT_SCORE.value == "min_context_score"
        assert EnricherConfig.MIN_MARGIN.value == "min_margin"

    def test_deduplication_settings(self):
        assert EnricherConfig.OVERLAP_THRESHOLD.value == "overlap_threshold"
        assert EnricherConfig.STORE_ALTERNATIVES.value == "store_alternatives"


class TestPipelineConfig:
    """Tests for PipelineConfig enum."""

    def test_extractor_keys(self):
        assert PipelineConfig.EXTRACTORS.value == "extractors"

    def test_option_keys(self):
        assert PipelineConfig.USE_LLM_VALIDATION.value == "use_llm_validation"
        assert PipelineConfig.USE_LLM_FEASIBILITY.value == "use_llm_feasibility"
        assert PipelineConfig.USE_VLM_TABLES.value == "use_vlm_tables"
        assert PipelineConfig.USE_NORMALIZATION.value == "use_normalization"

    def test_option_defaults(self):
        assert PipelineConfig.USE_LLM_VALIDATION.default is True
        assert PipelineConfig.USE_VLM_TABLES.default is False


class TestLLMConfig:
    """Tests for LLMConfig enum."""

    def test_llm_keys(self):
        assert LLMConfig.MODEL.value == "model"
        assert LLMConfig.TEMPERATURE.value == "temperature"
        assert LLMConfig.MAX_TOKENS.value == "max_tokens"

    def test_llm_defaults(self):
        assert "claude" in LLMConfig.MODEL.default.lower()
        assert LLMConfig.TEMPERATURE.default == 0.0
        assert LLMConfig.MAX_TOKENS.default == 4096


class TestGetConfig:
    """Tests for get_config helper function."""

    def test_get_existing_key(self):
        config = {"timeout_seconds": 60}
        result = get_config(config, ConfigKey.TIMEOUT_SECONDS)
        assert result == 60

    def test_get_missing_key_uses_default(self):
        config: dict[str, Any] = {}
        result = get_config(config, ConfigKey.TIMEOUT_SECONDS)
        assert result == 30  # ConfigKey.TIMEOUT_SECONDS.default

    def test_get_with_override_default(self):
        config: dict[str, Any] = {}
        result = get_config(config, ConfigKey.TIMEOUT_SECONDS, default=45)
        assert result == 45

    def test_get_none_value_returns_none(self):
        config = {"run_id": None}
        result = get_config(config, ConfigKey.RUN_ID)
        assert result is None


class TestGetNestedConfig:
    """Tests for get_nested_config helper function."""

    def test_get_two_level_nested(self):
        config = {
            "cache": {
                "ttl_hours": 48,
            }
        }
        result = get_nested_config(config, ConfigKey.CACHE, CacheConfig.TTL_HOURS)
        assert result == 48

    def test_get_missing_nested_uses_default(self):
        config: dict[str, Any] = {"cache": {}}
        result = get_nested_config(config, ConfigKey.CACHE, CacheConfig.TTL_HOURS)
        assert result == 24  # CacheConfig.TTL_HOURS.default

    def test_get_missing_section_uses_default(self):
        config: dict[str, Any] = {}
        result = get_nested_config(config, ConfigKey.CACHE, CacheConfig.TTL_HOURS)
        assert result == 24

    def test_get_with_override_default(self):
        config: dict[str, Any] = {}
        result = get_nested_config(
            config, ConfigKey.CACHE, CacheConfig.TTL_HOURS, default=12
        )
        assert result == 12


class TestConfigKeyAsDict:
    """Tests for using config keys as dictionary keys."""

    def test_config_key_as_dict_key(self):
        config: dict[str, str] = {ConfigKey.RUN_ID: "test-run"}
        assert config[ConfigKey.RUN_ID] == "test-run"
        # Also works with string
        assert config.get("run_id") == "test-run"

    def test_nested_config_access(self):
        config = {
            ConfigKey.CACHE: {
                CacheConfig.ENABLED: True,
                CacheConfig.TTL_HOURS: 12,
            }
        }
        cache_cfg = config.get(ConfigKey.CACHE, {})
        assert cache_cfg.get(CacheConfig.ENABLED) is True
        assert cache_cfg.get(CacheConfig.TTL_HOURS) == 12
