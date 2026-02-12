# corpus_metadata/K_tests/K01_base_provenance.py
"""Tests for BaseProvenanceMetadata generic class."""

from datetime import datetime
from enum import Enum

import pytest
from pydantic import ValidationError

from A_core.A01_domain_models import BaseProvenanceMetadata, LLMParameters


class CustomGeneratorType(str, Enum):
    """Custom enum for testing generator_name flexibility."""

    CUSTOM_STRATEGY = "gen:custom_strategy"
    ANOTHER_STRATEGY = "gen:another_strategy"


class AnotherEnumType(str, Enum):
    """Another enum type to test genericity."""

    TYPE_A = "type_a"
    TYPE_B = "type_b"


class TestBaseProvenanceMetadata:
    """Tests for the BaseProvenanceMetadata class."""

    def test_base_provenance_with_generator_type(self):
        """Test that BaseProvenanceMetadata works with any generator enum."""
        # Test with CustomGeneratorType
        prov1 = BaseProvenanceMetadata(
            pipeline_version="1.0.0",
            run_id="RUN_20250101_120000_abc123",
            doc_fingerprint="sha256_abc123",
            generator_name=CustomGeneratorType.CUSTOM_STRATEGY,
        )
        assert prov1.generator_name == CustomGeneratorType.CUSTOM_STRATEGY
        assert prov1.pipeline_version == "1.0.0"
        assert prov1.run_id == "RUN_20250101_120000_abc123"
        assert prov1.doc_fingerprint == "sha256_abc123"

        # Test with AnotherEnumType
        prov2 = BaseProvenanceMetadata(
            pipeline_version="2.0.0",
            run_id="RUN_20250102_130000_def456",
            doc_fingerprint="sha256_def456",
            generator_name=AnotherEnumType.TYPE_A,
        )
        assert prov2.generator_name == AnotherEnumType.TYPE_A

    def test_base_provenance_with_llm_config(self):
        """Test that BaseProvenanceMetadata works with LLM config."""
        llm_config = LLMParameters(
            model_name="claude-sonnet-4-5-20250929",
            temperature=0.0,
            max_tokens=1024,
            top_p=1.0,
            seed=42,
        )

        prov = BaseProvenanceMetadata(
            pipeline_version="1.0.0",
            run_id="RUN_20250101_120000_abc123",
            doc_fingerprint="sha256_abc123",
            generator_name=CustomGeneratorType.CUSTOM_STRATEGY,
            rule_version="v1.2.3",
            lexicon_source="2025_08_abbreviation_general.json",
            prompt_bundle_hash="hash_abc123",
            context_hash="ctx_hash_456",
            llm_config=llm_config,
        )

        assert prov.llm_config is not None
        assert prov.llm_config.model_name == "claude-sonnet-4-5-20250929"
        assert prov.llm_config.temperature == 0.0
        assert prov.llm_config.max_tokens == 1024
        assert prov.llm_config.seed == 42
        assert prov.rule_version == "v1.2.3"
        assert prov.lexicon_source == "2025_08_abbreviation_general.json"
        assert prov.prompt_bundle_hash == "hash_abc123"
        assert prov.context_hash == "ctx_hash_456"

    def test_base_provenance_is_frozen(self):
        """Test that BaseProvenanceMetadata is immutable (frozen)."""
        prov = BaseProvenanceMetadata(
            pipeline_version="1.0.0",
            run_id="RUN_20250101_120000_abc123",
            doc_fingerprint="sha256_abc123",
            generator_name=CustomGeneratorType.CUSTOM_STRATEGY,
        )

        # Attempting to modify any field should raise ValidationError
        with pytest.raises(ValidationError):
            prov.pipeline_version = "2.0.0"

        with pytest.raises(ValidationError):
            prov.run_id = "different_run_id"

        with pytest.raises(ValidationError):
            prov.generator_name = CustomGeneratorType.ANOTHER_STRATEGY

    def test_base_provenance_timestamp_default(self):
        """Test that timestamp has a default value."""
        prov = BaseProvenanceMetadata(
            pipeline_version="1.0.0",
            run_id="RUN_20250101_120000_abc123",
            doc_fingerprint="sha256_abc123",
            generator_name=CustomGeneratorType.CUSTOM_STRATEGY,
        )

        assert prov.timestamp is not None
        assert isinstance(prov.timestamp, datetime)

    def test_base_provenance_extra_forbid(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            BaseProvenanceMetadata(  # type: ignore[call-arg]
                pipeline_version="1.0.0",
                run_id="RUN_20250101_120000_abc123",
                doc_fingerprint="sha256_abc123",
                generator_name=CustomGeneratorType.CUSTOM_STRATEGY,
                unknown_field="should_fail",
            )
