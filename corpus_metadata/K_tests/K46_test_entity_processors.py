# corpus_metadata/K_tests/K46_test_entity_processors.py
"""
Tests for I_extraction.I01_entity_processors module.

Tests EntityProcessor and entity creation helpers.
"""

from __future__ import annotations

import uuid

import pytest

from I_extraction.I01_entity_processors import EntityProcessor
from A_core.A01_domain_models import (
    Candidate,
    Coordinate,
    FieldType,
    GeneratorType,
    ProvenanceMetadata,
    ValidationStatus,
)


@pytest.fixture
def processor():
    """Create an EntityProcessor without detectors."""
    return EntityProcessor(
        run_id="TEST_001",
        pipeline_version="1.0.0",
    )


@pytest.fixture
def sample_candidate():
    """Create a sample candidate for testing."""
    return Candidate(
        id=uuid.uuid4(),
        doc_id="test.pdf",
        short_form="TNF",
        long_form="Tumor Necrosis Factor",
        field_type=FieldType.DEFINITION_PAIR,
        generator_type=GeneratorType.SYNTAX_PATTERN,
        initial_confidence=0.8,
        context_text="TNF (Tumor Necrosis Factor) is a cytokine",
        context_location=Coordinate(page_num=1),
        provenance=ProvenanceMetadata(
            run_id="TEST",
            pipeline_version="1.0",
            doc_fingerprint="abc123",
            generator_name=GeneratorType.SYNTAX_PATTERN,
        ),
    )


class TestEntityProcessorInit:
    """Tests for EntityProcessor initialization."""

    def test_init_with_minimal_args(self):
        processor = EntityProcessor(
            run_id="TEST",
            pipeline_version="1.0",
        )
        assert processor.run_id == "TEST"
        assert processor.pipeline_version == "1.0"
        assert processor.disease_detector is None
        assert processor.drug_detector is None

    def test_init_sets_enrichers_to_none(self):
        processor = EntityProcessor(
            run_id="TEST",
            pipeline_version="1.0",
        )
        assert processor.disease_enricher is None
        assert processor.drug_enricher is None


class TestCreateEntityFromCandidate:
    """Tests for create_entity_from_candidate method."""

    def test_creates_validated_entity(self, processor, sample_candidate):
        entity = processor.create_entity_from_candidate(
            candidate=sample_candidate,
            status=ValidationStatus.VALIDATED,
            confidence=0.95,
            reason="Exact match",
            flags=["auto_validated"],
            raw_response={"test": True},
        )

        assert entity.short_form == "TNF"
        assert entity.long_form == "Tumor Necrosis Factor"
        assert entity.status == ValidationStatus.VALIDATED
        assert entity.confidence_score == 0.95
        assert "auto_validated" in entity.validation_flags

    def test_creates_rejected_entity_with_reason(self, processor, sample_candidate):
        entity = processor.create_entity_from_candidate(
            candidate=sample_candidate,
            status=ValidationStatus.REJECTED,
            confidence=0.2,
            reason="Invalid expansion",
            flags=["rejected"],
            raw_response={},
        )

        assert entity.status == ValidationStatus.REJECTED
        assert entity.rejection_reason == "Invalid expansion"

    def test_long_form_override(self, processor, sample_candidate):
        entity = processor.create_entity_from_candidate(
            candidate=sample_candidate,
            status=ValidationStatus.VALIDATED,
            confidence=0.9,
            reason="",
            flags=[],
            raw_response={},
            long_form_override="Corrected Long Form",
        )

        assert entity.long_form == "Corrected Long Form"

    def test_preserves_provenance(self, processor, sample_candidate):
        entity = processor.create_entity_from_candidate(
            candidate=sample_candidate,
            status=ValidationStatus.VALIDATED,
            confidence=0.9,
            reason="",
            flags=[],
            raw_response={},
        )

        assert entity.provenance == sample_candidate.provenance


class TestCreateEntityFromSearch:
    """Tests for create_entity_from_search method."""

    def test_creates_entity_from_match(self, processor):
        import re

        text = "The abbreviation TNF appears in the document."
        match = re.search(r"TNF", text)

        entity = processor.create_entity_from_search(
            doc_id="test.pdf",
            full_text=text,
            match=match,
            long_form="Tumor Necrosis Factor",
            field_type=FieldType.SHORT_FORM_ONLY,
            confidence=0.7,
            flags=["lexicon_match"],
            rule_version="v1.0",
            lexicon_source="rare_disease_lexicon",
        )

        assert entity.short_form == "TNF"
        assert entity.long_form == "Tumor Necrosis Factor"
        assert entity.status == ValidationStatus.VALIDATED
        assert "lexicon_match" in entity.validation_flags
        assert entity.provenance.lexicon_source == "orchestrator:rare_disease_lexicon"

    def test_entity_has_uuid_candidate_id(self, processor):
        import re

        text = "TNF"
        match = re.search(r"TNF", text)

        entity = processor.create_entity_from_search(
            doc_id="test.pdf",
            full_text=text,
            match=match,
            long_form=None,
            field_type=FieldType.SHORT_FORM_ONLY,
            confidence=0.5,
            flags=[],
            rule_version="v1.0",
            lexicon_source="test",
        )

        # candidate_id should be a valid UUID
        assert entity.candidate_id is not None


class TestProcessDiseases:
    """Tests for process_diseases method."""

    def test_returns_empty_without_detector(self, processor):
        # No mock doc needed - should short-circuit
        result = processor.process_diseases(None, None)
        assert result == []


class TestProcessGenes:
    """Tests for process_genes method."""

    def test_returns_empty_without_detector(self, processor):
        result = processor.process_genes(None, None)
        assert result == []


class TestProcessDrugs:
    """Tests for process_drugs method."""

    def test_returns_empty_without_detector(self, processor):
        result = processor.process_drugs(None, None)
        assert result == []


class TestProcessPharma:
    """Tests for process_pharma method."""

    def test_returns_empty_without_detector(self, processor):
        result = processor.process_pharma(None, None)
        assert result == []


class TestProcessAuthors:
    """Tests for process_authors method."""

    def test_returns_empty_without_detector(self, processor):
        result = processor.process_authors(None, None, "")
        assert result == []


class TestProcessCitations:
    """Tests for process_citations method."""

    def test_returns_empty_without_detector(self, processor):
        result = processor.process_citations(None, None, "")
        assert result == []
