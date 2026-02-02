# corpus_metadata/tests/test_visual_extraction/test_triage.py
"""
Tests for visual triage logic.
"""
import pytest

from A_core.A13_visual_models import (
    TableComplexitySignals,
    TriageDecision,
    VisualCandidate,
)
from B_parsing.B16_triage import (
    DocumentContext,
    TriageConfig,
    compute_area_ratio,
    compute_triage_statistics,
    get_cheap_path_candidates,
    get_skip_candidates,
    get_vlm_candidates,
    is_in_margin_zone,
    should_escalate_to_accurate,
    triage_batch,
    triage_visual,
)


# -------------------------
# Helper Functions
# -------------------------


def make_candidate(
    area_ratio: float = 0.10,
    has_caption: bool = False,
    docling_type: str = None,
    image_hash: str = None,
    in_margin: bool = False,
    has_grid: bool = False,
    is_referenced: bool = False,
    needs_accurate: bool = False,
    page_num: int = 1,
) -> VisualCandidate:
    """Create a visual candidate with specified properties."""
    return VisualCandidate(
        source="docling",
        docling_type=docling_type,
        page_num=page_num,
        bbox_pts=(100.0, 100.0, 300.0, 400.0),
        page_width_pts=612.0,
        page_height_pts=792.0,
        area_ratio=area_ratio,
        image_hash=image_hash,
        has_nearby_caption=has_caption,
        has_grid_structure=has_grid,
        is_referenced_in_text=is_referenced,
        in_margin_zone=in_margin,
        needs_accurate_rerun=needs_accurate,
    )


# -------------------------
# Geometry Helper Tests
# -------------------------


class TestIsInMarginZone:
    def test_in_header_zone(self):
        # Visual in top 10% of page (0-79.2 for 792pt page)
        bbox = (100.0, 10.0, 300.0, 70.0)  # Entirely in header
        assert is_in_margin_zone(bbox, 792.0)

    def test_in_footer_zone(self):
        # Visual in bottom 10% (712.8-792 for 792pt page)
        bbox = (100.0, 720.0, 300.0, 780.0)  # Entirely in footer
        assert is_in_margin_zone(bbox, 792.0)

    def test_in_body(self):
        # Visual in middle of page
        bbox = (100.0, 200.0, 300.0, 400.0)
        assert not is_in_margin_zone(bbox, 792.0)

    def test_spans_header_and_body(self):
        # Visual starts in header but extends into body
        bbox = (100.0, 50.0, 300.0, 150.0)
        assert not is_in_margin_zone(bbox, 792.0)


class TestComputeAreaRatio:
    def test_full_page(self):
        bbox = (0.0, 0.0, 612.0, 792.0)
        ratio = compute_area_ratio(bbox, 612.0, 792.0)
        assert ratio == pytest.approx(1.0)

    def test_quarter_page(self):
        bbox = (0.0, 0.0, 306.0, 396.0)  # Half width, half height
        ratio = compute_area_ratio(bbox, 612.0, 792.0)
        assert ratio == pytest.approx(0.25)

    def test_tiny_area(self):
        bbox = (0.0, 0.0, 50.0, 50.0)
        ratio = compute_area_ratio(bbox, 612.0, 792.0)
        assert ratio < 0.01

    def test_zero_page_area(self):
        bbox = (0.0, 0.0, 100.0, 100.0)
        ratio = compute_area_ratio(bbox, 0.0, 0.0)
        assert ratio == 0.0


# -------------------------
# Document Context Tests
# -------------------------


class TestDocumentContext:
    def test_build_from_candidates(self):
        candidates = [
            make_candidate(image_hash="hash1", page_num=1),
            make_candidate(image_hash="hash1", page_num=2),
            make_candidate(image_hash="hash1", page_num=3),
            make_candidate(image_hash="hash2", page_num=1),
        ]

        ctx = DocumentContext.build_from_candidates(candidates, repeat_threshold=3)

        assert ctx.image_hash_counts["hash1"] == 3
        assert ctx.image_hash_counts["hash2"] == 1
        assert "hash1" in ctx.repeated_image_hashes
        assert "hash2" not in ctx.repeated_image_hashes

    def test_body_references(self):
        ctx = DocumentContext()
        ctx.add_body_reference("Figure", 1)
        ctx.add_body_reference("Table", 2)

        assert ctx.is_referenced_in_body("Figure", 1)
        assert ctx.is_referenced_in_body("table", 2)  # Case insensitive
        assert not ctx.is_referenced_in_body("Figure", 2)


# -------------------------
# Triage Decision Tests
# -------------------------


class TestTriageVisual:
    def test_skip_tiny_area(self):
        candidate = make_candidate(area_ratio=0.01)
        ctx = DocumentContext()

        result = triage_visual(candidate, ctx)

        assert result.decision == TriageDecision.SKIP
        assert result.reason == "tiny_area"

    def test_skip_repeated_graphic(self):
        candidate = make_candidate(area_ratio=0.05, image_hash="logo123")
        ctx = DocumentContext(
            image_hash_counts={"logo123": 5},
            repeated_image_hashes={"logo123"},
        )

        result = triage_visual(candidate, ctx)

        assert result.decision == TriageDecision.SKIP
        assert result.reason == "repeated_graphic"

    def test_skip_margin_no_caption(self):
        candidate = make_candidate(area_ratio=0.05, in_margin=True, has_caption=False)
        ctx = DocumentContext()

        result = triage_visual(candidate, ctx)

        assert result.decision == TriageDecision.SKIP
        assert result.reason == "margin_no_caption"

    def test_vlm_has_caption(self):
        candidate = make_candidate(area_ratio=0.05, has_caption=True)
        ctx = DocumentContext()

        result = triage_visual(candidate, ctx)

        assert result.decision == TriageDecision.VLM_REQUIRED
        assert result.reason == "has_caption"

    def test_vlm_docling_table(self):
        candidate = make_candidate(area_ratio=0.05, docling_type="table")
        ctx = DocumentContext()

        result = triage_visual(candidate, ctx)

        assert result.decision == TriageDecision.VLM_REQUIRED
        assert result.reason == "docling_table"

    def test_vlm_body_reference(self):
        candidate = make_candidate(area_ratio=0.05, is_referenced=True)
        ctx = DocumentContext()

        result = triage_visual(candidate, ctx)

        assert result.decision == TriageDecision.VLM_REQUIRED
        assert result.reason == "body_reference"

    def test_vlm_grid_structure(self):
        candidate = make_candidate(area_ratio=0.05, has_grid=True)
        ctx = DocumentContext()

        result = triage_visual(candidate, ctx)

        assert result.decision == TriageDecision.VLM_REQUIRED
        assert result.reason == "grid_structure"

    def test_vlm_needs_accurate(self):
        candidate = make_candidate(area_ratio=0.05, needs_accurate=True)
        ctx = DocumentContext()

        result = triage_visual(candidate, ctx)

        assert result.decision == TriageDecision.VLM_REQUIRED
        assert result.reason == "complex_table"

    def test_vlm_large_uncaptioned(self):
        candidate = make_candidate(area_ratio=0.15)  # >10%
        ctx = DocumentContext()

        result = triage_visual(candidate, ctx)

        assert result.decision == TriageDecision.VLM_REQUIRED
        assert result.reason == "large_uncaptioned"

    def test_cheap_path_default(self):
        # Medium area, no special signals
        candidate = make_candidate(area_ratio=0.05)
        ctx = DocumentContext()

        result = triage_visual(candidate, ctx)

        assert result.decision == TriageDecision.CHEAP_PATH
        assert result.reason == "default"

    def test_margin_with_caption_is_vlm(self):
        # Even in margin zone, if has caption -> VLM
        candidate = make_candidate(area_ratio=0.05, in_margin=True, has_caption=True)
        ctx = DocumentContext()

        result = triage_visual(candidate, ctx)

        assert result.decision == TriageDecision.VLM_REQUIRED
        assert result.reason == "has_caption"


# -------------------------
# Batch Triage Tests
# -------------------------


class TestTriageBatch:
    def test_batch_triage(self):
        candidates = [
            make_candidate(area_ratio=0.01),  # Skip
            make_candidate(area_ratio=0.05, has_caption=True),  # VLM
            make_candidate(area_ratio=0.05),  # Cheap path
        ]

        results = triage_batch(candidates)

        assert len(results) == 3
        assert results[0][1].decision == TriageDecision.SKIP
        assert results[1][1].decision == TriageDecision.VLM_REQUIRED
        assert results[2][1].decision == TriageDecision.CHEAP_PATH


class TestFilterByDecision:
    def test_get_vlm_candidates(self):
        candidates = [
            make_candidate(area_ratio=0.01),  # Skip
            make_candidate(area_ratio=0.05, has_caption=True),  # VLM
            make_candidate(area_ratio=0.05),  # Cheap path
        ]

        triaged = triage_batch(candidates)
        vlm = get_vlm_candidates(triaged)

        assert len(vlm) == 1
        assert vlm[0].has_nearby_caption

    def test_get_skip_candidates(self):
        candidates = [
            make_candidate(area_ratio=0.01),
            make_candidate(area_ratio=0.005),
        ]

        triaged = triage_batch(candidates)
        skipped = get_skip_candidates(triaged)

        assert len(skipped) == 2

    def test_get_cheap_path_candidates(self):
        candidates = [
            make_candidate(area_ratio=0.05),
            make_candidate(area_ratio=0.08),
        ]

        triaged = triage_batch(candidates)
        cheap = get_cheap_path_candidates(triaged)

        assert len(cheap) == 2


# -------------------------
# Table Complexity Tests
# -------------------------


class TestShouldEscalateToAccurate:
    def test_multipage_escalates(self):
        signals = TableComplexitySignals(
            column_count=3,
            row_count=10,
            spans_multiple_pages=True,
        )

        should, reason = should_escalate_to_accurate(signals)

        assert should is True
        assert reason == "multipage_table"

    def test_deep_headers_escalates(self):
        signals = TableComplexitySignals(
            column_count=5,
            row_count=10,
            header_depth=4,  # Deep headers
        )

        should, reason = should_escalate_to_accurate(signals)

        assert should is True
        assert reason == "deep_headers"

    def test_many_merged_cells_escalates(self):
        signals = TableComplexitySignals(
            column_count=5,
            row_count=10,
            merged_cell_count=8,  # > 5
        )

        should, reason = should_escalate_to_accurate(signals)

        assert should is True
        assert reason == "many_merged_cells"

    def test_low_token_coverage_escalates(self):
        signals = TableComplexitySignals(
            column_count=5,
            row_count=10,
            token_coverage_ratio=0.60,  # < 0.70
        )

        should, reason = should_escalate_to_accurate(signals)

        assert should is True
        assert reason == "low_token_coverage"

    def test_vlm_flagged_escalates(self):
        signals = TableComplexitySignals(
            column_count=5,
            row_count=10,
            vlm_flagged_misparsed=True,
        )

        should, reason = should_escalate_to_accurate(signals)

        assert should is True
        assert reason == "vlm_flagged_misparsed"

    def test_large_complex_table_escalates(self):
        signals = TableComplexitySignals(
            column_count=10,  # >= 8
            row_count=20,  # >= 15
        )

        should, reason = should_escalate_to_accurate(signals)

        assert should is True
        assert reason == "large_complex_table"

    def test_simple_table_stays_fast(self):
        signals = TableComplexitySignals(
            column_count=3,
            row_count=5,
            header_depth=1,
            merged_cell_count=0,
            token_coverage_ratio=0.95,
        )

        should, reason = should_escalate_to_accurate(signals)

        assert should is False
        assert reason == "fast_sufficient"


# -------------------------
# Statistics Tests
# -------------------------


class TestComputeTriageStatistics:
    def test_statistics_computation(self):
        candidates = [
            make_candidate(area_ratio=0.01),  # Skip - tiny
            make_candidate(area_ratio=0.008),  # Skip - tiny
            make_candidate(area_ratio=0.05, has_caption=True),  # VLM - caption
            make_candidate(area_ratio=0.15),  # VLM - large
            make_candidate(area_ratio=0.05),  # Cheap path
        ]

        triaged = triage_batch(candidates)
        stats = compute_triage_statistics(triaged)

        assert stats.total_candidates == 5
        assert stats.skip_count == 2
        assert stats.vlm_required_count == 2
        assert stats.cheap_path_count == 1

        assert stats.skip_reasons["tiny_area"] == 2
        assert stats.vlm_reasons["has_caption"] == 1
        assert stats.vlm_reasons["large_uncaptioned"] == 1

        assert stats.skip_ratio == pytest.approx(0.4)
        assert stats.vlm_ratio == pytest.approx(0.4)


# -------------------------
# Configuration Tests
# -------------------------


class TestTriageConfig:
    def test_custom_config(self):
        config = TriageConfig(
            skip_area_ratio=0.05,  # Stricter - skip smaller
            vlm_area_threshold=0.20,  # More permissive for large
        )

        # This would normally be cheap path, but with stricter config -> skip
        candidate = make_candidate(area_ratio=0.04)
        ctx = DocumentContext()

        result = triage_visual(candidate, ctx, config)

        assert result.decision == TriageDecision.SKIP
        assert result.reason == "tiny_area"
