# corpus_metadata/K_tests/K38_test_span_deduplicator.py
"""
Tests for E_normalization.E11_span_deduplicator module.

Tests NER span deduplication from multiple sources.
"""

from __future__ import annotations

import pytest

from E_normalization.E11_span_deduplicator import (
    NERSpan,
    DeduplicationResult,
    SpanDeduplicator,
)


class TestNERSpan:
    """Tests for NERSpan dataclass."""

    def test_basic_span(self):
        span = NERSpan(
            text="diabetes mellitus",
            category="disease",
            confidence=0.95,
            source="BiomedicalNER",
        )
        assert span.text == "diabetes mellitus"
        assert span.confidence == 0.95

    def test_span_with_positions(self):
        span = NERSpan(
            text="diabetes",
            category="disease",
            confidence=0.9,
            source="EpiExtract4GARD-v2",
            start=100,
            end=108,
        )
        assert span.start == 100
        assert span.end == 108

    def test_overlaps_with_positions(self):
        span1 = NERSpan(
            text="diabetes mellitus",
            category="disease",
            confidence=0.9,
            source="A",
            start=0,
            end=17,
        )
        span2 = NERSpan(
            text="diabetes",
            category="disease",
            confidence=0.85,
            source="B",
            start=0,
            end=8,
        )

        assert span1.overlaps_with(span2)

    def test_no_overlap_positions(self):
        span1 = NERSpan(
            text="diabetes",
            category="disease",
            confidence=0.9,
            source="A",
            start=0,
            end=8,
        )
        span2 = NERSpan(
            text="hypertension",
            category="disease",
            confidence=0.85,
            source="B",
            start=100,
            end=112,
        )

        assert not span1.overlaps_with(span2)

    def test_text_overlap_exact(self):
        span1 = NERSpan(text="diabetes", category="disease", confidence=0.9, source="A")
        span2 = NERSpan(text="diabetes", category="disease", confidence=0.85, source="B")

        assert span1.overlaps_with(span2)

    def test_text_overlap_substring(self):
        span1 = NERSpan(
            text="diabetes mellitus type 2",
            category="disease",
            confidence=0.9,
            source="A",
        )
        span2 = NERSpan(
            text="diabetes mellitus",
            category="disease",
            confidence=0.85,
            source="B",
        )

        # One is substring of other
        overlap = span1._text_overlap(span2.text)
        assert overlap > 0.5

    def test_text_overlap_word_based(self):
        span1 = NERSpan(text="type 2 diabetes", category="disease", confidence=0.9, source="A")
        span2 = NERSpan(text="diabetes type 2", category="disease", confidence=0.85, source="B")

        # Word overlap (Jaccard)
        overlap = span1._text_overlap(span2.text)
        assert overlap > 0.5


class TestDeduplicationResult:
    """Tests for DeduplicationResult dataclass."""

    def test_empty_result(self):
        result = DeduplicationResult()
        assert result.unique_spans == []
        assert result.merged_count == 0

    def test_to_summary(self):
        result = DeduplicationResult(
            total_input=10,
            merged_count=3,
            unique_spans=[
                NERSpan(text="test", category="disease", confidence=0.9, source="A")
            ],
        )
        summary = result.to_summary()

        assert summary["total_input"] == 10
        assert summary["unique_spans"] == 1
        assert summary["merged_count"] == 3


class TestSpanDeduplicator:
    """Tests for SpanDeduplicator class."""

    @pytest.fixture
    def deduplicator(self):
        return SpanDeduplicator()

    def test_empty_input(self, deduplicator):
        result = deduplicator.deduplicate([])
        assert len(result.unique_spans) == 0

    def test_single_span(self, deduplicator):
        spans = [
            NERSpan(text="diabetes", category="disease", confidence=0.9, source="A")
        ]
        result = deduplicator.deduplicate(spans)

        assert len(result.unique_spans) == 1

    def test_dedup_overlapping_same_category(self, deduplicator):
        spans = [
            NERSpan(text="diabetes", category="disease", confidence=0.95, source="A"),
            NERSpan(text="diabetes", category="disease", confidence=0.8, source="B"),
        ]
        result = deduplicator.deduplicate(spans)

        # Should keep only one (highest confidence)
        assert len(result.unique_spans) == 1
        assert result.unique_spans[0].confidence == 0.95
        assert result.merged_count == 1

    def test_no_dedup_different_categories(self, deduplicator):
        spans = [
            NERSpan(text="diabetes", category="disease", confidence=0.9, source="A"),
            NERSpan(text="diabetes", category="symptom", confidence=0.8, source="B"),
        ]
        result = deduplicator.deduplicate(spans)

        # Different categories, no dedup
        assert len(result.unique_spans) == 2

    def test_filter_low_confidence(self):
        deduplicator = SpanDeduplicator(config={"min_confidence": 0.5})
        spans = [
            NERSpan(text="diabetes", category="disease", confidence=0.9, source="A"),
            NERSpan(text="mild symptom", category="symptom", confidence=0.2, source="B"),
        ]
        result = deduplicator.deduplicate(spans)

        # Low confidence should be filtered
        assert len(result.unique_spans) == 1

    def test_merge_source_tracking(self, deduplicator):
        spans = [
            NERSpan(text="diabetes", category="disease", confidence=0.95, source="A"),
            NERSpan(text="diabetes", category="disease", confidence=0.8, source="B"),
            NERSpan(text="diabetes", category="disease", confidence=0.7, source="C"),
        ]
        result = deduplicator.deduplicate(spans)

        assert len(result.unique_spans) == 1
        # Should track merged sources
        assert "merged_from" in result.unique_spans[0].metadata


class TestCategoryNormalization:
    """Tests for category normalization."""

    @pytest.fixture
    def deduplicator(self):
        return SpanDeduplicator()

    def test_normalize_epidemiology(self, deduplicator):
        assert deduplicator.normalize_category("prevalence") == "epidemiology"
        assert deduplicator.normalize_category("incidence") == "epidemiology"

    def test_normalize_adverse_event(self, deduplicator):
        assert deduplicator.normalize_category("ADE") == "adverse_event"
        assert deduplicator.normalize_category("side_effect") == "adverse_event"

    def test_normalize_drug_admin(self, deduplicator):
        assert deduplicator.normalize_category("drug_dosage") == "drug_admin"
        assert deduplicator.normalize_category("drug_frequency") == "drug_admin"

    def test_unknown_category_unchanged(self, deduplicator):
        assert deduplicator.normalize_category("unknown_category") == "unknown_category"


class TestBySourceTracking:
    """Tests for source tracking in results."""

    def test_by_source_counts(self):
        deduplicator = SpanDeduplicator()
        spans = [
            NERSpan(text="disease1", category="disease", confidence=0.9, source="A"),
            NERSpan(text="disease2", category="disease", confidence=0.9, source="A"),
            NERSpan(text="symptom1", category="symptom", confidence=0.9, source="B"),
        ]
        result = deduplicator.deduplicate(spans)

        assert result.by_source["A"] == 2
        assert result.by_source["B"] == 1
