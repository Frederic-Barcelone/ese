# corpus_metadata/K_tests/K47_test_feasibility_processor.py
"""
Tests for I_extraction.I02_feasibility_processor module.

Tests FeasibilityProcessor and NER enrichment.
"""

from __future__ import annotations

import pytest

from I_extraction.I02_feasibility_processor import FeasibilityProcessor


@pytest.fixture
def processor():
    """Create a FeasibilityProcessor without any enrichers."""
    return FeasibilityProcessor(run_id="TEST_001")


class TestFeasibilityProcessorInit:
    """Tests for FeasibilityProcessor initialization."""

    def test_init_with_minimal_args(self):
        processor = FeasibilityProcessor(run_id="TEST")
        assert processor.run_id == "TEST"
        assert processor.feasibility_detector is None
        assert processor.llm_feasibility_extractor is None

    def test_init_all_enrichers_none(self):
        processor = FeasibilityProcessor(run_id="TEST")
        assert processor.epi_enricher is None
        assert processor.zeroshot_bioner is None
        assert processor.biomedical_ner is None
        assert processor.patient_journey_enricher is None
        assert processor.registry_enricher is None
        assert processor.genetic_enricher is None


class TestFeasibilityProcessorProcess:
    """Tests for process method."""

    def test_returns_empty_without_detector(self, processor):
        """Should return empty list if no detector available."""
        result = processor.process(None, None, "")
        assert result == []


class TestFeasibilityProcessorEnrichment:
    """Tests for enrichment methods."""

    def test_enrich_with_epiextract_noop_without_enricher(self, processor):
        """Should return candidates unchanged if enricher not available."""
        candidates = []
        result = processor._enrich_with_epiextract(candidates, "sample text")
        assert result == candidates

    def test_enrich_with_zeroshot_noop_without_enricher(self, processor):
        """Should return candidates unchanged if enricher not available."""
        candidates = []
        result = processor._enrich_with_zeroshot(candidates, "sample text")
        assert result == candidates

    def test_enrich_with_biomedical_ner_noop_without_enricher(self, processor):
        """Should return candidates unchanged if enricher not available."""
        candidates = []
        result = processor._enrich_with_biomedical_ner(candidates, "sample text")
        assert result == candidates

    def test_enrich_with_patient_journey_noop_without_enricher(self, processor):
        """Should return candidates unchanged if enricher not available."""
        candidates = []
        result = processor._enrich_with_patient_journey(candidates, "sample text")
        assert result == candidates

    def test_enrich_with_registry_noop_without_enricher(self, processor):
        """Should return candidates unchanged if enricher not available."""
        candidates = []
        result = processor._enrich_with_registry(candidates, "sample text")
        assert result == candidates

    def test_enrich_with_genetic_noop_without_enricher(self, processor):
        """Should return candidates unchanged if enricher not available."""
        candidates = []
        result = processor._enrich_with_genetic(candidates, "sample text")
        assert result == candidates


class TestFeasibilityProcessorDeduplication:
    """Tests for span deduplication."""

    def test_deduplicate_empty_list(self, processor):
        """Should handle empty list."""
        result = processor._deduplicate_spans([])
        assert result == []
