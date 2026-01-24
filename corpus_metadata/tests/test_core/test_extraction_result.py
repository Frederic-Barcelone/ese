# corpus_metadata/tests/test_core/test_extraction_result.py
"""
Tests for the universal extraction output contract.

Tests cover:
- ExtractionResult immutability
- JSON-hash ID stability (same content = same ID)
- compute_regression_hash determinism
- MergeResolver determinism
"""

from __future__ import annotations

import pytest

from A_core.A14_extraction_result import (
    EntityType,
    ExtractionResult,
    Provenance,
    compute_result_id,
    compute_regression_hash,
)
from A_core.A02_interfaces import RawExtraction, ExecutionContext
from B_parsing.B06_confidence import UnifiedConfidenceCalculator
from H_pipeline.H04_merge_resolver import MergeResolver, MergeConfig


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_provenance() -> Provenance:
    """Create a sample Provenance for testing."""
    return Provenance(
        page_num=1,
        strategy_id="disease_lexicon_orphanet",
        bbox=(100.0, 200.0, 300.0, 250.0),
        node_ids=("block_123", "block_456"),
        char_span=(10, 50),
        strategy_version="1.0.0",
        doc_fingerprint="abc123",
    )


@pytest.fixture
def sample_result(sample_provenance: Provenance) -> ExtractionResult:
    """Create a sample ExtractionResult for testing."""
    return ExtractionResult(
        doc_id="doc_001",
        entity_type=EntityType.DISEASE,
        field_name="disease",
        value="pulmonary arterial hypertension",
        provenance=sample_provenance,
        normalized_value="Pulmonary arterial hypertension",
        confidence=0.85,
        confidence_features=(("lexicon_match", 0.1), ("section_match", 0.15)),
        evidence_text="patients with pulmonary arterial hypertension were enrolled",
        status="validated",
    )


@pytest.fixture
def sample_raw_extraction() -> RawExtraction:
    """Create a sample RawExtraction for testing."""
    return RawExtraction(
        doc_id="doc_001",
        entity_type=EntityType.DISEASE,
        field_name="disease",
        value="pulmonary arterial hypertension",
        page_num=1,
        strategy_id="disease_lexicon_orphanet",
        normalized_value="Pulmonary arterial hypertension",
        bbox=(100.0, 200.0, 300.0, 250.0),
        node_ids=("block_123",),
        evidence_text="patients with pulmonary arterial hypertension were enrolled",
        section_name="abstract",
        lexicon_matched=True,
        from_table=False,
    )


# =============================================================================
# ExtractionResult Immutability Tests
# =============================================================================


class TestExtractionResultImmutability:
    """Test that ExtractionResult is truly immutable."""

    def test_frozen_dataclass(self, sample_result: ExtractionResult):
        """ExtractionResult should be frozen (immutable)."""
        with pytest.raises(AttributeError):
            sample_result.value = "new value"  # type: ignore

    def test_provenance_frozen(self, sample_provenance: Provenance):
        """Provenance should be frozen (immutable)."""
        with pytest.raises(AttributeError):
            sample_provenance.page_num = 99  # type: ignore

    def test_with_confidence_returns_new_instance(self, sample_result: ExtractionResult):
        """with_confidence should return a new instance, not modify in place."""
        new_result = sample_result.with_confidence(0.95, (("new_feature", 0.2),))

        assert new_result is not sample_result
        assert new_result.confidence == 0.95
        assert sample_result.confidence == 0.85  # Original unchanged

    def test_with_status_returns_new_instance(self, sample_result: ExtractionResult):
        """with_status should return a new instance, not modify in place."""
        new_result = sample_result.with_status("rejected", "low confidence")

        assert new_result is not sample_result
        assert new_result.status == "rejected"
        assert new_result.rejection_reason == "low confidence"
        assert sample_result.status == "validated"  # Original unchanged


# =============================================================================
# Deterministic ID Tests
# =============================================================================


class TestDeterministicId:
    """Test that compute_result_id produces stable, deterministic IDs."""

    def test_same_content_same_id(self, sample_provenance: Provenance):
        """Same content should produce same ID."""
        result1 = ExtractionResult(
            doc_id="doc_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="test disease",
            provenance=sample_provenance,
        )
        result2 = ExtractionResult(
            doc_id="doc_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="test disease",
            provenance=sample_provenance,
        )

        assert result1.id == result2.id

    def test_different_value_different_id(self, sample_provenance: Provenance):
        """Different value should produce different ID."""
        result1 = ExtractionResult(
            doc_id="doc_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="disease A",
            provenance=sample_provenance,
        )
        result2 = ExtractionResult(
            doc_id="doc_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="disease B",
            provenance=sample_provenance,
        )

        assert result1.id != result2.id

    def test_id_is_16_char_hex(self, sample_result: ExtractionResult):
        """ID should be 16 character hex string."""
        assert len(sample_result.id) == 16
        assert all(c in "0123456789abcdef" for c in sample_result.id)

    def test_id_stable_across_calls(self, sample_result: ExtractionResult):
        """Multiple calls to .id should return same value."""
        id1 = sample_result.id
        id2 = sample_result.id
        id3 = compute_result_id(sample_result)

        assert id1 == id2 == id3

    def test_non_deterministic_fields_excluded(self, sample_provenance: Provenance):
        """Non-deterministic fields (run_id, timestamp) should not affect ID."""
        from datetime import datetime

        prov1 = Provenance(
            page_num=sample_provenance.page_num,
            strategy_id=sample_provenance.strategy_id,
            run_id="run_001",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )
        prov2 = Provenance(
            page_num=sample_provenance.page_num,
            strategy_id=sample_provenance.strategy_id,
            run_id="run_999",
            timestamp=datetime(2025, 6, 15, 18, 30, 0),
        )

        result1 = ExtractionResult(
            doc_id="doc_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="test",
            provenance=prov1,
        )
        result2 = ExtractionResult(
            doc_id="doc_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="test",
            provenance=prov2,
        )

        assert result1.id == result2.id


# =============================================================================
# Regression Hash Tests
# =============================================================================


class TestRegressionHash:
    """Test that compute_regression_hash is deterministic."""

    def test_same_results_same_hash(self, sample_result: ExtractionResult):
        """Same results should produce same hash."""
        results = [sample_result]

        hash1 = compute_regression_hash(results)
        hash2 = compute_regression_hash(results)

        assert hash1 == hash2

    def test_order_independent(self, sample_provenance: Provenance):
        """Hash should be independent of input order (internally sorted)."""
        result1 = ExtractionResult(
            doc_id="doc_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="disease A",
            provenance=sample_provenance,
        )
        result2 = ExtractionResult(
            doc_id="doc_002",
            entity_type=EntityType.DRUG,
            field_name="drug",
            value="drug B",
            provenance=sample_provenance,
        )

        hash_order1 = compute_regression_hash([result1, result2])
        hash_order2 = compute_regression_hash([result2, result1])

        assert hash_order1 == hash_order2

    def test_different_results_different_hash(self, sample_provenance: Provenance):
        """Different results should produce different hash."""
        result1 = ExtractionResult(
            doc_id="doc_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="disease A",
            provenance=sample_provenance,
        )
        result2 = ExtractionResult(
            doc_id="doc_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="disease B",
            provenance=sample_provenance,
        )

        hash1 = compute_regression_hash([result1])
        hash2 = compute_regression_hash([result2])

        assert hash1 != hash2

    def test_hash_is_16_char_hex(self, sample_result: ExtractionResult):
        """Hash should be 16 character hex string."""
        hash_val = compute_regression_hash([sample_result])

        assert len(hash_val) == 16
        assert all(c in "0123456789abcdef" for c in hash_val)


# =============================================================================
# UnifiedConfidenceCalculator Tests
# =============================================================================


class TestUnifiedConfidenceCalculator:
    """Test the unified confidence calculator."""

    def test_calculate_returns_tuple(self, sample_raw_extraction: RawExtraction):
        """calculate() should return (score, features) tuple."""
        calc = UnifiedConfidenceCalculator()
        result = calc.calculate(sample_raw_extraction)

        assert isinstance(result, tuple)
        assert len(result) == 2
        score, features = result
        assert isinstance(score, float)
        assert isinstance(features, tuple)

    def test_confidence_bounds(self, sample_raw_extraction: RawExtraction):
        """Confidence should be bounded between min and max."""
        calc = UnifiedConfidenceCalculator(min_confidence=0.1, max_confidence=0.95)
        score, _ = calc.calculate(sample_raw_extraction)

        assert 0.1 <= score <= 0.95

    def test_lexicon_match_increases_confidence(self):
        """Lexicon match should increase confidence."""
        calc = UnifiedConfidenceCalculator()

        raw_no_lexicon = RawExtraction(
            doc_id="doc",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="test",
            page_num=1,
            strategy_id="test",
            lexicon_matched=False,
        )
        raw_with_lexicon = RawExtraction(
            doc_id="doc",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="test",
            page_num=1,
            strategy_id="test",
            lexicon_matched=True,
        )

        score_no, _ = calc.calculate(raw_no_lexicon)
        score_with, _ = calc.calculate(raw_with_lexicon)

        assert score_with > score_no

    def test_to_extraction_result(self, sample_raw_extraction: RawExtraction):
        """to_extraction_result should produce valid ExtractionResult."""
        calc = UnifiedConfidenceCalculator()
        result = calc.to_extraction_result(sample_raw_extraction)

        assert isinstance(result, ExtractionResult)
        assert result.doc_id == sample_raw_extraction.doc_id
        assert result.value == sample_raw_extraction.value
        assert result.confidence > 0
        assert result.status == "pending"


# =============================================================================
# MergeResolver Tests
# =============================================================================


class TestMergeResolver:
    """Test the deterministic merge resolver."""

    def test_merge_empty_list(self):
        """Merging empty list should return empty list."""
        resolver = MergeResolver()
        result = resolver.merge([])

        assert result == []

    def test_merge_single_item(self, sample_raw_extraction: RawExtraction):
        """Merging single item should return that item."""
        resolver = MergeResolver()
        result = resolver.merge([sample_raw_extraction])

        assert len(result) == 1
        assert result[0].value == sample_raw_extraction.value

    def test_merge_deduplicates(self):
        """Identical extractions should be deduplicated."""
        raw1 = RawExtraction(
            doc_id="doc_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="test disease",
            page_num=1,
            strategy_id="strategy_a",
        )
        raw2 = RawExtraction(
            doc_id="doc_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="test disease",
            page_num=2,
            strategy_id="strategy_b",
        )

        resolver = MergeResolver()
        result = resolver.merge([raw1, raw2])

        assert len(result) == 1

    def test_merge_preserves_different_values(self):
        """Different values should not be merged."""
        raw1 = RawExtraction(
            doc_id="doc_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="disease A",
            page_num=1,
            strategy_id="strategy",
        )
        raw2 = RawExtraction(
            doc_id="doc_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="disease B",
            page_num=1,
            strategy_id="strategy",
        )

        resolver = MergeResolver()
        result = resolver.merge([raw1, raw2])

        assert len(result) == 2

    def test_merge_deterministic(self):
        """Same inputs should always produce same outputs."""
        raws = [
            RawExtraction(
                doc_id="doc_001",
                entity_type=EntityType.DISEASE,
                field_name="disease",
                value="disease",
                page_num=1,
                strategy_id="strategy_a",
            ),
            RawExtraction(
                doc_id="doc_001",
                entity_type=EntityType.DISEASE,
                field_name="disease",
                value="disease",
                page_num=2,
                strategy_id="strategy_b",
            ),
        ]

        resolver = MergeResolver()

        # Run multiple times
        result1 = resolver.merge(raws)
        result2 = resolver.merge(raws)
        result3 = resolver.merge(list(reversed(raws)))  # Different order

        # All should produce same output
        assert len(result1) == len(result2) == len(result3) == 1
        assert result1[0].strategy_id == result2[0].strategy_id == result3[0].strategy_id

    def test_strategy_priority_respected(self):
        """Higher priority strategy should win."""
        config = MergeConfig(
            strategy_priority={
                "high_priority": 10,
                "low_priority": 1,
            }
        )
        resolver = MergeResolver(config)

        raw_low = RawExtraction(
            doc_id="doc_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="disease",
            page_num=1,
            strategy_id="low_priority",
        )
        raw_high = RawExtraction(
            doc_id="doc_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="disease",
            page_num=1,
            strategy_id="high_priority",
        )

        result = resolver.merge([raw_low, raw_high])

        assert len(result) == 1
        assert result[0].strategy_id == "high_priority"

    def test_evidence_merged(self):
        """Supporting evidence from all candidates should be merged."""
        raw1 = RawExtraction(
            doc_id="doc_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="disease",
            page_num=1,
            strategy_id="strategy_a",
            evidence_text="evidence A",
        )
        raw2 = RawExtraction(
            doc_id="doc_001",
            entity_type=EntityType.DISEASE,
            field_name="disease",
            value="disease",
            page_num=2,
            strategy_id="strategy_b",
            evidence_text="evidence B",
        )

        resolver = MergeResolver()
        result = resolver.merge([raw1, raw2])

        assert len(result) == 1
        # The non-primary evidence should be in supporting_evidence
        assert "evidence A" in result[0].supporting_evidence or "evidence B" in result[0].supporting_evidence


# =============================================================================
# ExecutionContext Tests
# =============================================================================


class TestExecutionContext:
    """Test the execution context."""

    def test_get_prior_outputs_empty(self):
        """get_prior_outputs should return empty list for unknown strategy."""
        ctx = ExecutionContext(
            plan_id="plan_001",
            doc_id="doc_001",
            doc_fingerprint="abc123",
        )

        result = ctx.get_prior_outputs("unknown_strategy")

        assert result == []

    def test_get_prior_outputs_returns_stored(self, sample_raw_extraction: RawExtraction):
        """get_prior_outputs should return stored outputs."""
        ctx = ExecutionContext(
            plan_id="plan_001",
            doc_id="doc_001",
            doc_fingerprint="abc123",
        )
        ctx.outputs_by_step["test_strategy"] = [sample_raw_extraction]

        result = ctx.get_prior_outputs("test_strategy")

        assert len(result) == 1
        assert result[0] == sample_raw_extraction
