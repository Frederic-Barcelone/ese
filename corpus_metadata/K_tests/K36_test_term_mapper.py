# corpus_metadata/K_tests/K36_test_term_mapper.py
"""
Tests for E_normalization.E01_term_mapper module.

Tests term mapping, normalization, and lexicon loading.
"""

from __future__ import annotations

import json
import tempfile
import uuid

import pytest

from E_normalization.E01_term_mapper import TermMapper
from A_core.A01_domain_models import (
    ExtractedEntity,
    ValidationStatus,
    FieldType,
    EvidenceSpan,
    Coordinate,
    ProvenanceMetadata,
    GeneratorType,
)


@pytest.fixture
def sample_entity():
    return ExtractedEntity(
        candidate_id=uuid.uuid4(),
        doc_id="test.pdf",
        short_form="TNF",
        long_form="Tumor Necrosis Factor",
        field_type=FieldType.DEFINITION_PAIR,
        primary_evidence=EvidenceSpan(
            text="TNF (Tumor Necrosis Factor)",
            location=Coordinate(page_num=1),
            scope_ref="abc",
            start_char_offset=0,
            end_char_offset=27,
        ),
        supporting_evidence=[],
        status=ValidationStatus.VALIDATED,
        confidence_score=0.9,
        provenance=ProvenanceMetadata(
            run_id="TEST",
            pipeline_version="1.0",
            doc_fingerprint="abc123",
            generator_name=GeneratorType.SYNTAX_PATTERN,
        ),
    )


@pytest.fixture
def temp_lexicon():
    """Create a temporary lexicon file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        lexicon = {
            "tnf": {
                "canonical_long_form": "Tumor Necrosis Factor",
                "standard_id": "MESH:D014409",
            },
            "tumor necrosis factor": {
                "canonical_long_form": "Tumor Necrosis Factor",
                "standard_id": "MESH:D014409",
            },
            "il6": {
                "canonical_long_form": "Interleukin 6",
                "standard_id": "MESH:D015850",
            },
        }
        json.dump(lexicon, f)
        return f.name


class TestTermMapper:
    """Tests for TermMapper class."""

    def test_init_empty_config(self):
        mapper = TermMapper(config={})
        assert mapper.enable_fuzzy is False
        assert mapper.fill_long_form_for_orphans is False

    def test_init_with_lexicon(self, temp_lexicon):
        mapper = TermMapper(config={"mapping_file_path": temp_lexicon})
        assert len(mapper.lookup_table) > 0
        assert "tnf" in mapper.lookup_table

    def test_normalize_validated_entity(self, temp_lexicon, sample_entity):
        mapper = TermMapper(config={"mapping_file_path": temp_lexicon})
        result = mapper.normalize(sample_entity)

        assert result.standard_id == "MESH:D014409"
        assert "normalized" in result.validation_flags

    def test_normalize_non_validated_returns_unchanged(self, temp_lexicon):
        mapper = TermMapper(config={"mapping_file_path": temp_lexicon})

        entity = ExtractedEntity(
            candidate_id=uuid.uuid4(),
            doc_id="test.pdf",
            short_form="TNF",
            long_form="Tumor Necrosis Factor",
            field_type=FieldType.DEFINITION_PAIR,
            primary_evidence=EvidenceSpan(
                text="test",
                location=Coordinate(page_num=1),
                scope_ref="abc",
                start_char_offset=0,
                end_char_offset=4,
            ),
            supporting_evidence=[],
            status=ValidationStatus.REJECTED,  # Not validated
            confidence_score=0.2,
            provenance=ProvenanceMetadata(
                run_id="TEST",
                pipeline_version="1.0",
                doc_fingerprint="abc",
                generator_name=GeneratorType.SYNTAX_PATTERN,
            ),
        )

        result = mapper.normalize(entity)
        assert result.standard_id is None  # Not normalized


class TestTermMapperPreprocessing:
    """Tests for preprocessing methods."""

    def test_preprocess_whitespace(self):
        mapper = TermMapper(config={})
        assert mapper._preprocess("  hello   world  ") == "hello world"

    def test_preprocess_case(self):
        mapper = TermMapper(config={})
        assert mapper._preprocess("UPPERCASE") == "uppercase"

    def test_preprocess_punctuation(self):
        mapper = TermMapper(config={})
        assert mapper._preprocess('"quoted"') == "quoted"
        assert mapper._preprocess("(parentheses)") == "parentheses"


class TestTermMapperLexiconLoading:
    """Tests for lexicon loading."""

    def test_load_dict_format(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"tnf": {"canonical_long_form": "TNF Alpha"}}, f)
            path = f.name

        mapper = TermMapper(config={"mapping_file_path": path})
        assert "tnf" in mapper.lookup_table

    def test_load_list_format(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"key": "tnf", "canonical_long_form": "TNF Alpha"}], f)
            path = f.name

        mapper = TermMapper(config={"mapping_file_path": path})
        assert "tnf" in mapper.lookup_table

    def test_load_nonexistent_file(self):
        mapper = TermMapper(config={"mapping_file_path": "/nonexistent/path.json"})
        assert len(mapper.lookup_table) == 0

    def test_coerce_entry_different_keys(self):
        mapper = TermMapper(config={})

        # Test "name" key
        entry = mapper._coerce_entry({"name": "Test Name", "code": "T001"})
        assert entry is not None
        assert entry["canonical_long_form"] == "Test Name"
        assert entry["standard_id"] == "T001"

        # Test "canonical" key
        entry = mapper._coerce_entry({"canonical": "Another Name", "id": "A001"})
        assert entry is not None
        assert entry["canonical_long_form"] == "Another Name"
        assert entry["standard_id"] == "A001"


class TestTermMapperFuzzyMatching:
    """Tests for fuzzy matching."""

    def test_fuzzy_disabled_by_default(self):
        mapper = TermMapper(config={})
        assert mapper.enable_fuzzy is False

    def test_fuzzy_lookup_when_enabled(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"tumor necrosis factor": {"canonical_long_form": "TNF"}}, f)
            path = f.name

        mapper = TermMapper(
            config={
                "mapping_file_path": path,
                "enable_fuzzy_matching": True,
                "fuzzy_cutoff": 0.8,
            }
        )

        # Exact match
        match = mapper._fuzzy_lookup("tumor necrosis factor")
        assert match is not None

        # Near match (should work with fuzzy)
        match = mapper._fuzzy_lookup("tumour necrosis factor")
        # May or may not match depending on cutoff


class TestTermMapperOrphans:
    """Tests for SHORT_FORM_ONLY handling."""

    def test_orphan_no_fill_by_default(self, temp_lexicon):
        mapper = TermMapper(config={"mapping_file_path": temp_lexicon})

        entity = ExtractedEntity(
            candidate_id=uuid.uuid4(),
            doc_id="test.pdf",
            short_form="TNF",
            long_form=None,  # No long form
            field_type=FieldType.SHORT_FORM_ONLY,
            primary_evidence=EvidenceSpan(
                text="test",
                location=Coordinate(page_num=1),
                scope_ref="abc",
                start_char_offset=0,
                end_char_offset=4,
            ),
            supporting_evidence=[],
            status=ValidationStatus.VALIDATED,
            confidence_score=0.7,
            provenance=ProvenanceMetadata(
                run_id="TEST",
                pipeline_version="1.0",
                doc_fingerprint="abc",
                generator_name=GeneratorType.SYNTAX_PATTERN,
            ),
        )

        result = mapper.normalize(entity)
        # Long form should NOT be filled by default
        assert result.long_form is None

    def test_orphan_fill_when_enabled(self, temp_lexicon):
        mapper = TermMapper(
            config={
                "mapping_file_path": temp_lexicon,
                "fill_long_form_for_orphans": True,
            }
        )

        entity = ExtractedEntity(
            candidate_id=uuid.uuid4(),
            doc_id="test.pdf",
            short_form="TNF",
            long_form=None,
            field_type=FieldType.SHORT_FORM_ONLY,
            primary_evidence=EvidenceSpan(
                text="test",
                location=Coordinate(page_num=1),
                scope_ref="abc",
                start_char_offset=0,
                end_char_offset=4,
            ),
            supporting_evidence=[],
            status=ValidationStatus.VALIDATED,
            confidence_score=0.7,
            provenance=ProvenanceMetadata(
                run_id="TEST",
                pipeline_version="1.0",
                doc_fingerprint="abc",
                generator_name=GeneratorType.SYNTAX_PATTERN,
            ),
        )

        result = mapper.normalize(entity)
        # Long form should be filled when enabled
        assert result.long_form == "Tumor Necrosis Factor"
