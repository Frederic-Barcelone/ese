# corpus_metadata/K_tests/K37_test_deduplicator.py
"""
Tests for E_normalization.E07_deduplicator module.

Tests abbreviation deduplication with quality-based selection.
"""

from __future__ import annotations

import uuid
from typing import Optional

import pytest

from E_normalization.E07_deduplicator import Deduplicator, DeduplicationStats, _normalize_lf
from A_core.A01_domain_models import (
    ExtractedEntity,
    ValidationStatus,
    FieldType,
    GeneratorType,
    EvidenceSpan,
    Coordinate,
    ProvenanceMetadata,
)


def make_entity(
    sf: str,
    lf: str,
    confidence: float = 0.9,
    generator: GeneratorType = GeneratorType.SYNTAX_PATTERN,
    field_type: FieldType = FieldType.DEFINITION_PAIR,
    page: int = 1,
    entity_id: Optional[str] = None,
) -> ExtractedEntity:
    """Helper to create test entities."""
    return ExtractedEntity(
        candidate_id=uuid.uuid4(),
        doc_id="test.pdf",
        short_form=sf,
        long_form=lf,
        field_type=field_type,
        primary_evidence=EvidenceSpan(
            text=f"{lf} ({sf})",
            location=Coordinate(page_num=page),
            scope_ref="abc",
            start_char_offset=0,
            end_char_offset=len(f"{lf} ({sf})"),
        ),
        supporting_evidence=[],
        status=ValidationStatus.VALIDATED,
        confidence_score=confidence,
        provenance=ProvenanceMetadata(
            run_id="TEST",
            pipeline_version="1.0",
            doc_fingerprint="abc",
            generator_name=generator,
        ),
    )


class TestNormalizeLongForm:
    """Tests for _normalize_lf helper."""

    def test_basic_normalization(self):
        assert _normalize_lf("Tumor Necrosis Factor") == "tumor necrosis factor"

    def test_whitespace_normalization(self):
        assert _normalize_lf("Tumor  Necrosis   Factor") == "tumor necrosis factor"

    def test_dash_normalization(self):
        # Various dash characters should be normalized
        assert _normalize_lf("anti-inflammatory") == "anti-inflammatory"

    def test_empty_string(self):
        assert _normalize_lf("") == ""

    def test_none_handling(self):
        # Should handle None gracefully
        assert _normalize_lf(None) == ""  # type: ignore[arg-type]


class TestDeduplicator:
    """Tests for Deduplicator class."""

    @pytest.fixture
    def deduplicator(self):
        return Deduplicator(config={})

    def test_single_entity_unchanged(self, deduplicator):
        entities = [make_entity("TNF", "Tumor Necrosis Factor")]
        result = deduplicator.deduplicate(entities)

        assert len(result) == 1
        assert result[0].short_form == "TNF"
        assert "deduplicated" in result[0].validation_flags

    def test_dedup_same_sf_same_lf(self, deduplicator):
        """Multiple entities with same SF and LF should merge."""
        entities = [
            make_entity("TNF", "Tumor Necrosis Factor", page=1, entity_id="e1"),
            make_entity("TNF", "Tumor Necrosis Factor", page=2, entity_id="e2"),
            make_entity("TNF", "Tumor Necrosis Factor", page=5, entity_id="e3"),
        ]
        result = deduplicator.deduplicate(entities)

        assert len(result) == 1
        assert result[0].mention_count == 3
        assert set(result[0].pages_mentioned) == {1, 2, 5}

    def test_dedup_same_sf_different_lf(self, deduplicator):
        """Multiple LFs for same SF should select best and store alternatives."""
        entities = [
            make_entity(
                "TNF",
                "Tumor Necrosis Factor",
                confidence=0.95,
                generator=GeneratorType.GLOSSARY_TABLE,
            ),
            make_entity(
                "TNF",
                "Tumour Necrosis Factor",  # British spelling
                confidence=0.8,
                generator=GeneratorType.LEXICON_MATCH,
            ),
        ]
        result = deduplicator.deduplicate(entities)

        assert len(result) == 1
        # Glossary table should win (higher priority)
        assert result[0].long_form == "Tumor Necrosis Factor"

    def test_different_sf_not_merged(self, deduplicator):
        """Different short forms should not be merged."""
        entities = [
            make_entity("TNF", "Tumor Necrosis Factor"),
            make_entity("IL6", "Interleukin 6"),
        ]
        result = deduplicator.deduplicate(entities)

        assert len(result) == 2

    def test_non_validated_excluded(self, deduplicator):
        """Non-validated entities should be excluded by default."""
        entities = [
            make_entity("TNF", "Tumor Necrosis Factor"),
        ]
        # Add a rejected entity
        rejected = ExtractedEntity(
            candidate_id=uuid.uuid4(),
            doc_id="test.pdf",
            short_form="BAD",
            long_form="Bad Abbreviation",
            field_type=FieldType.DEFINITION_PAIR,
            primary_evidence=EvidenceSpan(
                text="test",
                location=Coordinate(page_num=1),
                scope_ref="abc",
                start_char_offset=0,
                end_char_offset=4,
            ),
            supporting_evidence=[],
            status=ValidationStatus.REJECTED,
            confidence_score=0.2,
            provenance=ProvenanceMetadata(
                run_id="TEST",
                pipeline_version="1.0",
                doc_fingerprint="abc",
                generator_name=GeneratorType.SYNTAX_PATTERN,
            ),
        )
        entities.append(rejected)

        result = deduplicator.deduplicate(entities)
        assert len(result) == 1  # Only validated entity


class TestDeduplicatorScoring:
    """Tests for entity scoring."""

    @pytest.fixture
    def deduplicator(self):
        return Deduplicator(config={})

    def test_glossary_beats_lexicon(self, deduplicator):
        """Glossary table source should beat lexicon match."""
        entities = [
            make_entity("TNF", "Wrong Name", generator=GeneratorType.LEXICON_MATCH),
            make_entity("TNF", "Correct Name", generator=GeneratorType.GLOSSARY_TABLE),
        ]
        result = deduplicator.deduplicate(entities)

        assert result[0].long_form == "Correct Name"

    def test_syntax_pattern_beats_lexicon(self, deduplicator):
        """Syntax pattern (Schwartz-Hearst) should beat lexicon."""
        entities = [
            make_entity("TNF", "Lexicon Name", generator=GeneratorType.LEXICON_MATCH),
            make_entity("TNF", "Document Name", generator=GeneratorType.SYNTAX_PATTERN),
        ]
        result = deduplicator.deduplicate(entities)

        assert result[0].long_form == "Document Name"

    def test_higher_confidence_wins(self, deduplicator):
        """Higher confidence should win when generators are equal."""
        entities = [
            make_entity("TNF", "Low Confidence", confidence=0.5),
            make_entity("TNF", "High Confidence", confidence=0.95),
        ]
        result = deduplicator.deduplicate(entities)

        assert result[0].long_form == "High Confidence"

    def test_generic_expansion_penalized(self, deduplicator):
        """Generic expansions like 'unknown' should be penalized."""
        entities = [
            make_entity("TNF", "unknown", confidence=0.9),
            make_entity("TNF", "Tumor Necrosis Factor", confidence=0.7),
        ]
        result = deduplicator.deduplicate(entities)

        # Real expansion should win despite lower confidence
        assert result[0].long_form == "Tumor Necrosis Factor"


class TestDeduplicatorConfig:
    """Tests for Deduplicator configuration."""

    def test_store_alternatives_enabled(self):
        deduplicator = Deduplicator(config={"store_alternatives": True})
        entities = [
            make_entity("TNF", "Primary Name", confidence=0.95),
            make_entity("TNF", "Alternative Name", confidence=0.8),
        ]
        result = deduplicator.deduplicate(entities)

        # Alternatives should be stored
        assert result[0].normalized_value is not None
        assert "deduplication" in result[0].normalized_value

    def test_max_alternatives(self):
        deduplicator = Deduplicator(config={"max_alternatives": 2})
        entities = [
            make_entity("TNF", f"Name {i}", confidence=0.9 - i * 0.1, entity_id=f"e{i}")
            for i in range(5)
        ]
        result = deduplicator.deduplicate(entities)

        # Should only store max_alternatives
        assert isinstance(result[0].normalized_value, dict)
        dedup_info = result[0].normalized_value.get("deduplication", {})
        assert isinstance(dedup_info, dict)
        assert len(dedup_info.get("alternatives", [])) <= 2


class TestDeduplicationStats:
    """Tests for DeduplicationStats class."""

    def test_stats_str(self):
        stats = DeduplicationStats()
        stats.total_input = 10
        stats.total_output = 5
        stats.groups_merged = 3
        stats.alternatives_found = 4

        output = str(stats)
        assert "10" in output
        assert "5" in output
        assert "merged" in output.lower()
