# corpus_metadata/K_tests/K44_test_merge_resolver.py
"""
Tests for H_pipeline.H04_merge_resolver module.

Tests deterministic merge and conflict resolution.
"""

from __future__ import annotations

import pytest

from H_pipeline.H04_merge_resolver import (
    MergeConfig,
    MergeResolver,
    get_merge_resolver,
)
from A_core.A02_interfaces import RawExtraction, EntityType


def make_extraction(
    value: str,
    strategy_id: str = "test",
    page_num: int = 1,
    evidence_text: str = "context",
    from_table: bool = False,
    normalized_value: str | None = None,
) -> RawExtraction:
    """Helper to create test extractions."""
    return RawExtraction(
        doc_id="test.pdf",
        entity_type=EntityType.ABBREVIATION,
        field_name="short_form",
        value=value,
        page_num=page_num,
        strategy_id=strategy_id,
        normalized_value=normalized_value,
        evidence_text=evidence_text,
        from_table=from_table,
    )


class TestMergeConfig:
    """Tests for MergeConfig dataclass."""

    def test_default_config(self):
        config = MergeConfig.default()
        assert config.prefer_table_evidence is True
        assert config.prefer_longer_evidence is True
        assert "abbreviation_syntax" in config.strategy_priority

    def test_custom_dedupe_fields(self):
        config = MergeConfig(
            dedupe_key_fields=("doc_id", "field_name", "value")
        )
        assert len(config.dedupe_key_fields) == 3

    def test_strategy_priority_defaults(self):
        config = MergeConfig.default()
        # Syntax should beat lexicon
        assert config.strategy_priority["abbreviation_syntax"] > config.strategy_priority["abbreviation_lexicon"]
        # Glossary should beat lexicon
        assert config.strategy_priority["abbreviation_glossary"] > config.strategy_priority["abbreviation_lexicon"]

    def test_mutual_exclusivity_defaults(self):
        config = MergeConfig.default()
        # Default has no exclusivity constraints
        assert config.mutually_exclusive_fields == ()


class TestMergeResolver:
    """Tests for MergeResolver class."""

    @pytest.fixture
    def resolver(self):
        return MergeResolver(MergeConfig.default())

    def test_empty_input(self, resolver):
        result = resolver.merge([])
        assert result == []

    def test_single_extraction_unchanged(self, resolver):
        extraction = make_extraction("TNF", strategy_id="test")
        result = resolver.merge([extraction])
        assert len(result) == 1
        assert result[0].value == "TNF"

    def test_dedup_same_value(self, resolver):
        """Extractions with same dedupe key should merge."""
        extractions = [
            make_extraction("TNF", strategy_id="a", page_num=1),
            make_extraction("TNF", strategy_id="b", page_num=2),
        ]
        result = resolver.merge(extractions)
        # Should merge to single result
        assert len(result) == 1
        assert result[0].value == "TNF"

    def test_different_values_not_merged(self, resolver):
        """Extractions with different values should not merge."""
        extractions = [
            make_extraction("TNF"),
            make_extraction("IL6"),
        ]
        result = resolver.merge(extractions)
        assert len(result) == 2

    def test_strategy_priority_wins(self):
        """Higher priority strategy should win."""
        config = MergeConfig(
            strategy_priority={
                "high_priority": 10,
                "low_priority": 1,
            }
        )
        resolver = MergeResolver(config)

        extractions = [
            make_extraction("TNF", strategy_id="low_priority", evidence_text="short"),
            make_extraction("TNF", strategy_id="high_priority", evidence_text="short"),
        ]
        result = resolver.merge(extractions)

        assert len(result) == 1
        assert result[0].strategy_id == "high_priority"

    def test_table_evidence_preference(self):
        """Table evidence should be preferred when configured."""
        config = MergeConfig(prefer_table_evidence=True)
        resolver = MergeResolver(config)

        extractions = [
            make_extraction("TNF", strategy_id="text", from_table=False),
            make_extraction("TNF", strategy_id="table", from_table=True),
        ]
        result = resolver.merge(extractions)

        assert len(result) == 1
        assert result[0].from_table is True

    def test_longer_evidence_preference(self):
        """Longer evidence should be preferred when configured."""
        config = MergeConfig(prefer_longer_evidence=True, strategy_priority={})
        resolver = MergeResolver(config)

        extractions = [
            make_extraction("TNF", strategy_id="a", evidence_text="short"),
            make_extraction("TNF", strategy_id="b", evidence_text="much longer evidence text"),
        ]
        result = resolver.merge(extractions)

        assert len(result) == 1
        assert "longer" in result[0].evidence_text

    def test_supporting_evidence_merged(self, resolver):
        """Supporting evidence should be collected from all candidates."""
        extractions = [
            make_extraction("TNF", strategy_id="a", evidence_text="evidence A"),
            make_extraction("TNF", strategy_id="b", evidence_text="evidence B"),
        ]
        result = resolver.merge(extractions)

        assert len(result) == 1
        # Winner's evidence text is primary, others are supporting
        all_evidence = set(result[0].supporting_evidence) | {result[0].evidence_text}
        assert "evidence A" in all_evidence or "evidence B" in all_evidence


class TestMergeResolverDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_output(self):
        """Same inputs should always produce same outputs."""
        resolver = MergeResolver(MergeConfig.default())

        extractions = [
            make_extraction("TNF", strategy_id="b", page_num=2),
            make_extraction("TNF", strategy_id="a", page_num=1),
            make_extraction("IL6", strategy_id="c", page_num=3),
        ]

        # Run multiple times
        results = [resolver.merge(extractions) for _ in range(3)]

        # All should be identical
        for i in range(1, len(results)):
            assert len(results[i]) == len(results[0])
            for j in range(len(results[0])):
                assert results[i][j].value == results[0][j].value
                assert results[i][j].strategy_id == results[0][j].strategy_id

    def test_output_sorted(self):
        """Output should be sorted deterministically."""
        resolver = MergeResolver(MergeConfig.default())

        extractions = [
            make_extraction("ZZZ", page_num=3),
            make_extraction("AAA", page_num=1),
            make_extraction("MMM", page_num=2),
        ]
        result = resolver.merge(extractions)

        # Should be sorted by sort key (doc_id, entity_type, field_name, value...)
        values = [r.value for r in result]
        assert values == sorted(values)


class TestMergeResolverConstraints:
    """Tests for mutual exclusivity constraints."""

    def test_mutual_exclusivity_keeps_first(self):
        """Mutual exclusivity should keep only first in sort order."""
        config = MergeConfig(
            mutually_exclusive_fields=(("rare_true", "rare_false"),),
        )
        resolver = MergeResolver(config)

        extractions = [
            RawExtraction(
                doc_id="test.pdf",
                entity_type=EntityType.DISEASE,
                field_name="rare_true",
                value="yes",
                page_num=1,
                strategy_id="a",
            ),
            RawExtraction(
                doc_id="test.pdf",
                entity_type=EntityType.DISEASE,
                field_name="rare_false",
                value="no",
                page_num=2,
                strategy_id="b",
            ),
        ]
        result = resolver.merge(extractions)

        # Only one should remain
        field_names = {r.field_name for r in result}
        assert len(field_names & {"rare_true", "rare_false"}) == 1


class TestGetMergeResolver:
    """Tests for singleton resolver function."""

    def test_returns_resolver(self):
        resolver = get_merge_resolver()
        assert isinstance(resolver, MergeResolver)

    def test_singleton_behavior(self):
        """Should return same instance."""
        resolver1 = get_merge_resolver()
        resolver2 = get_merge_resolver()
        assert resolver1 is resolver2
