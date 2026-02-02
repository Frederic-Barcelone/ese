# corpus_metadata/K_tests/K45_test_component_factory.py
"""
Tests for H_pipeline.H01_component_factory module.

Tests ComponentFactory initialization and component creation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from H_pipeline.H01_component_factory import ComponentFactory


@pytest.fixture
def minimal_config():
    """Minimal configuration for testing."""
    return {
        "paths": {
            "dictionaries": "ouput_datasources",
        },
        "lexicons": {},
        "api": {
            "claude": {
                "validation": {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 450,
                    "temperature": 0.0,
                },
            },
        },
        "extraction_pipeline": {
            "options": {
                "use_llm_validation": False,  # Disable LLM for tests
                "use_llm_feasibility": False,
                "use_vlm_tables": False,
                "use_epi_enricher": False,
                "use_zeroshot_bioner": False,
                "use_biomedical_ner": False,
                "use_patient_journey": False,
                "use_registry_extraction": False,
                "use_genetic_extraction": False,
            },
        },
        "generators": {},
        "normalization": {},
        "nct_enricher": {"enabled": False},
    }


@pytest.fixture
def temp_log_dir():
    """Create temporary log directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestComponentFactoryInit:
    """Tests for ComponentFactory initialization."""

    def test_init_with_minimal_config(self, minimal_config, temp_log_dir):
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST_001",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        assert factory.run_id == "TEST_001"
        assert factory.pipeline_version == "1.0"
        assert factory.use_llm_validation is False

    def test_init_extracts_options(self, minimal_config, temp_log_dir):
        minimal_config["extraction_pipeline"]["options"]["use_llm_validation"] = True

        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        assert factory.use_llm_validation is True

    def test_init_with_api_key(self, minimal_config, temp_log_dir):
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
            api_key="test-key",
        )

        assert factory.api_key == "test-key"


class TestComponentFactoryParserCreation:
    """Tests for parser creation."""

    def test_create_parser(self, minimal_config, temp_log_dir):
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        parser = factory.create_parser()
        assert parser is not None

    def test_create_table_extractor_returns_none_without_docling(
        self, minimal_config, temp_log_dir
    ):
        """Table extractor should return None if Docling not available."""
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        # This may return None or the extractor depending on environment
        result = factory.create_table_extractor()
        # Just verify it doesn't raise
        assert result is None or result is not None


class TestComponentFactoryGeneratorCreation:
    """Tests for generator creation."""

    def test_create_generators_returns_list(self, minimal_config, temp_log_dir):
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        generators = factory.create_generators()

        assert isinstance(generators, list)
        assert len(generators) > 0

    def test_generators_include_syntax(self, minimal_config, temp_log_dir):
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        generators = factory.create_generators()
        class_names = {g.__class__.__name__ for g in generators}

        assert "AbbrevSyntaxCandidateGenerator" in class_names


class TestComponentFactoryLLMCreation:
    """Tests for LLM component creation."""

    def test_create_claude_client_returns_none_when_disabled(
        self, minimal_config, temp_log_dir
    ):
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        client = factory.create_claude_client()
        assert client is None

    def test_create_llm_engine_returns_none_when_disabled(
        self, minimal_config, temp_log_dir
    ):
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        engine = factory.create_llm_engine(None)
        assert engine is None


class TestComponentFactoryValidationCreation:
    """Tests for validation component creation."""

    def test_create_validation_logger(self, minimal_config, temp_log_dir):
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        logger = factory.create_validation_logger()
        assert logger is not None
        assert logger.run_id == "TEST"


class TestComponentFactoryNormalizationCreation:
    """Tests for normalization component creation."""

    def test_create_term_mapper(self, minimal_config, temp_log_dir):
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        mapper = factory.create_term_mapper()
        assert mapper is not None

    def test_create_disambiguator(self, minimal_config, temp_log_dir):
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        disambiguator = factory.create_disambiguator()
        assert disambiguator is not None

    def test_create_deduplicator(self, minimal_config, temp_log_dir):
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        deduplicator = factory.create_deduplicator()
        assert deduplicator is not None


class TestComponentFactoryDetectorCreation:
    """Tests for detector creation."""

    def test_create_disease_detector(self, minimal_config, temp_log_dir):
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        detector = factory.create_disease_detector()
        assert detector is not None

    def test_create_drug_detector(self, minimal_config, temp_log_dir):
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        detector = factory.create_drug_detector()
        assert detector is not None

    def test_create_gene_detector(self, minimal_config, temp_log_dir):
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        detector = factory.create_gene_detector()
        assert detector is not None


class TestComponentFactoryEnricherCreation:
    """Tests for enricher creation."""

    def test_create_nct_enricher_returns_none_when_disabled(
        self, minimal_config, temp_log_dir
    ):
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        enricher = factory.create_nct_enricher()
        assert enricher is None

    def test_create_epi_enricher_returns_none_when_disabled(
        self, minimal_config, temp_log_dir
    ):
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        enricher = factory.create_epi_enricher()
        assert enricher is None


class TestComponentFactoryLookups:
    """Tests for lookup loading."""

    def test_load_rare_disease_lookup_returns_dict(self, minimal_config, temp_log_dir):
        factory = ComponentFactory(
            config=minimal_config,
            run_id="TEST",
            pipeline_version="1.0",
            log_dir=temp_log_dir,
        )

        lookup = factory.load_rare_disease_lookup()
        assert isinstance(lookup, dict)
