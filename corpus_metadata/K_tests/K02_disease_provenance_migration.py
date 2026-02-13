# corpus_metadata/K_tests/K02_disease_provenance_migration.py
"""
Tests for DiseaseProvenanceMetadata migration to inherit from BaseProvenanceMetadata.

These tests verify:
1. All base provenance fields exist on DiseaseProvenanceMetadata
2. Disease-specific lexicon_ids field works correctly with DiseaseIdentifier
3. Frozen/immutability is preserved after migration
"""

from datetime import datetime

import pytest

from A_core.A01_domain_models import LLMParameters
from A_core.A05_disease_models import (
    DiseaseGeneratorType,
    DiseaseIdentifier,
    DiseaseProvenanceMetadata,
)


class TestDiseaseProvenanceHasBaseFields:
    """Verify all base provenance fields exist on DiseaseProvenanceMetadata."""

    @pytest.fixture
    def disease_provenance(self) -> DiseaseProvenanceMetadata:
        """Create a DiseaseProvenanceMetadata instance with all fields."""
        return DiseaseProvenanceMetadata(
            pipeline_version="abc123",
            run_id="RUN_20250101_120000_test",
            doc_fingerprint="sha256_test_fingerprint",
            generator_name=DiseaseGeneratorType.LEXICON_ORPHANET,
            rule_version="1.0.0",
            lexicon_source="disease_lexicon_pah.json",
            prompt_bundle_hash="prompt_hash_123",
            context_hash="context_hash_456",
            llm_config=LLMParameters(
                model_name="claude-sonnet-4-5-20250929",
                temperature=0.0,
                max_tokens=1000,
                top_p=1.0,
            ),
        )

    def test_pipeline_version_exists(self, disease_provenance: DiseaseProvenanceMetadata):
        """Verify pipeline_version field exists."""
        assert disease_provenance.pipeline_version == "abc123"

    def test_run_id_exists(self, disease_provenance: DiseaseProvenanceMetadata):
        """Verify run_id field exists."""
        assert disease_provenance.run_id == "RUN_20250101_120000_test"

    def test_doc_fingerprint_exists(self, disease_provenance: DiseaseProvenanceMetadata):
        """Verify doc_fingerprint field exists."""
        assert disease_provenance.doc_fingerprint == "sha256_test_fingerprint"

    def test_generator_name_is_disease_type(self, disease_provenance: DiseaseProvenanceMetadata):
        """Verify generator_name is DiseaseGeneratorType."""
        assert disease_provenance.generator_name == DiseaseGeneratorType.LEXICON_ORPHANET
        assert isinstance(disease_provenance.generator_name, DiseaseGeneratorType)

    def test_rule_version_exists(self, disease_provenance: DiseaseProvenanceMetadata):
        """Verify rule_version field exists."""
        assert disease_provenance.rule_version == "1.0.0"

    def test_lexicon_source_exists(self, disease_provenance: DiseaseProvenanceMetadata):
        """Verify lexicon_source field exists."""
        assert disease_provenance.lexicon_source == "disease_lexicon_pah.json"

    def test_prompt_bundle_hash_exists(self, disease_provenance: DiseaseProvenanceMetadata):
        """Verify prompt_bundle_hash field exists."""
        assert disease_provenance.prompt_bundle_hash == "prompt_hash_123"

    def test_context_hash_exists(self, disease_provenance: DiseaseProvenanceMetadata):
        """Verify context_hash field exists."""
        assert disease_provenance.context_hash == "context_hash_456"

    def test_llm_config_exists(self, disease_provenance: DiseaseProvenanceMetadata):
        """Verify llm_config field exists."""
        assert disease_provenance.llm_config is not None
        assert disease_provenance.llm_config.model_name == "claude-sonnet-4-5-20250929"

    def test_timestamp_auto_populated(self, disease_provenance: DiseaseProvenanceMetadata):
        """Verify timestamp is auto-populated."""
        assert isinstance(disease_provenance.timestamp, datetime)
        assert (datetime.utcnow() - disease_provenance.timestamp).total_seconds() < 5


class TestDiseaseProvenanceWithLexiconIds:
    """Verify disease-specific lexicon_ids works with DiseaseIdentifier."""

    def test_lexicon_ids_with_disease_identifiers(self):
        """Verify lexicon_ids accepts DiseaseIdentifier list."""
        identifiers = [
            DiseaseIdentifier(system="ORPHA", code="ORPHA:182090", display="Pulmonary Arterial Hypertension"),
            DiseaseIdentifier(system="MONDO", code="MONDO_0011055"),
            DiseaseIdentifier(system="ICD-10", code="I27.0"),
        ]

        provenance = DiseaseProvenanceMetadata(
            pipeline_version="abc123",
            run_id="RUN_20250101_120000_test",
            doc_fingerprint="sha256_test_fingerprint",
            generator_name=DiseaseGeneratorType.LEXICON_ORPHANET,
            lexicon_ids=identifiers,
        )

        assert provenance.lexicon_ids is not None
        assert len(provenance.lexicon_ids) == 3
        assert provenance.lexicon_ids[0].system == "ORPHA"
        assert provenance.lexicon_ids[0].code == "ORPHA:182090"
        assert provenance.lexicon_ids[1].system == "MONDO"
        assert provenance.lexicon_ids[2].system == "ICD-10"

    def test_lexicon_ids_optional(self):
        """Verify lexicon_ids is optional (None by default)."""
        provenance = DiseaseProvenanceMetadata(
            pipeline_version="abc123",
            run_id="RUN_20250101_120000_test",
            doc_fingerprint="sha256_test_fingerprint",
            generator_name=DiseaseGeneratorType.SCISPACY_NER,
        )

        assert provenance.lexicon_ids is None

    def test_lexicon_ids_empty_list(self):
        """Verify lexicon_ids can be an empty list."""
        provenance = DiseaseProvenanceMetadata(
            pipeline_version="abc123",
            run_id="RUN_20250101_120000_test",
            doc_fingerprint="sha256_test_fingerprint",
            generator_name=DiseaseGeneratorType.LEXICON_GENERAL,
            lexicon_ids=[],
        )

        assert provenance.lexicon_ids == []


class TestDiseaseProvenanceIsFrozen:
    """Verify immutability is preserved after migration."""

    @pytest.fixture
    def disease_provenance(self) -> DiseaseProvenanceMetadata:
        """Create a DiseaseProvenanceMetadata instance."""
        return DiseaseProvenanceMetadata(
            pipeline_version="abc123",
            run_id="RUN_20250101_120000_test",
            doc_fingerprint="sha256_test_fingerprint",
            generator_name=DiseaseGeneratorType.LEXICON_ORPHANET,
        )

    def test_cannot_modify_pipeline_version(self, disease_provenance: DiseaseProvenanceMetadata):
        """Verify pipeline_version cannot be modified."""
        with pytest.raises(Exception):  # ValidationError for frozen models
            disease_provenance.pipeline_version = "new_version"

    def test_cannot_modify_run_id(self, disease_provenance: DiseaseProvenanceMetadata):
        """Verify run_id cannot be modified."""
        with pytest.raises(Exception):
            disease_provenance.run_id = "new_run_id"

    def test_cannot_modify_generator_name(self, disease_provenance: DiseaseProvenanceMetadata):
        """Verify generator_name cannot be modified."""
        with pytest.raises(Exception):
            disease_provenance.generator_name = DiseaseGeneratorType.SCISPACY_NER

    def test_cannot_modify_lexicon_ids(self, disease_provenance: DiseaseProvenanceMetadata):
        """Verify lexicon_ids cannot be modified."""
        with pytest.raises(Exception):
            disease_provenance.lexicon_ids = [
                DiseaseIdentifier(system="ORPHA", code="ORPHA:1234")
            ]

    def test_model_config_frozen_true(self):
        """Verify model_config has frozen=True."""
        assert DiseaseProvenanceMetadata.model_config.get("frozen") is True
