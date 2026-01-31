# corpus_metadata/tests/test_visual_extraction/test_caption_extractor.py
"""
Tests for multisource caption extraction.
"""
import pytest
from unittest.mock import MagicMock, patch

from A_core.A13_visual_models import (
    CaptionCandidate,
    CaptionProvenance,
    CaptionSearchZones,
    ReferenceSource,
    VisualReference,
)
from B_parsing.B15_caption_extractor import (
    ColumnLayout,
    detect_caption_pattern,
    get_column_for_x,
    get_relative_position,
    has_horizontal_overlap,
    has_vertical_overlap,
    infer_column_layout,
    is_continuation_caption,
    parse_reference_from_match,
    select_best_caption,
)


# -------------------------
# Caption Pattern Tests
# -------------------------


class TestDetectCaptionPattern:
    def test_figure_pattern_standard(self):
        result = detect_caption_pattern("Figure 1. Patient demographics")
        assert result is not None
        caption_type, match = result
        assert caption_type == "figure"
        assert match.group(1) == "1"

    def test_figure_pattern_abbreviated(self):
        result = detect_caption_pattern("Fig. 2 Kaplan-Meier curves")
        assert result is not None
        caption_type, match = result
        assert caption_type == "figure"
        assert match.group(1) == "2"

    def test_figure_pattern_no_period(self):
        result = detect_caption_pattern("Fig 3A Shows the results")
        assert result is not None
        caption_type, match = result
        assert caption_type == "figure"
        assert match.group(1) == "3"
        assert match.group(3) == "A"

    def test_figure_pattern_range(self):
        result = detect_caption_pattern("Figure 2-4. Multiple panels")
        assert result is not None
        caption_type, match = result
        assert caption_type == "figure"
        assert match.group(1) == "2"
        assert match.group(2) == "4"

    def test_table_pattern_standard(self):
        result = detect_caption_pattern("Table 1 Baseline characteristics")
        assert result is not None
        caption_type, match = result
        assert caption_type == "table"
        assert match.group(1) == "1"

    def test_table_pattern_with_suffix(self):
        result = detect_caption_pattern("Table 2B. Subgroup analysis")
        assert result is not None
        caption_type, match = result
        assert caption_type == "table"
        assert match.group(1) == "2"
        assert match.group(3) == "B"

    def test_exhibit_pattern(self):
        result = detect_caption_pattern("Exhibit 3. Regulatory summary")
        assert result is not None
        caption_type, match = result
        assert caption_type == "exhibit"

    def test_no_pattern_match(self):
        result = detect_caption_pattern("This is regular text")
        assert result is None

    def test_no_pattern_partial_match(self):
        result = detect_caption_pattern("The Figure shows...")
        assert result is None  # Doesn't start with Figure


class TestIsContinuationCaption:
    def test_continued_pattern(self):
        assert is_continuation_caption("Table 1 (continued)")
        assert is_continuation_caption("Figure 2 (cont.)")
        assert is_continuation_caption("Table 3 (cont'd)")
        assert is_continuation_caption("Table 4 (CONTINUED)")

    def test_not_continuation(self):
        assert not is_continuation_caption("Table 1. Demographics")
        assert not is_continuation_caption("Figure 2")


# -------------------------
# Reference Parsing Tests
# -------------------------


class TestParseReferenceFromMatch:
    def test_simple_figure_reference(self):
        import re
        pattern = re.compile(r"^(?:Figure|Fig\.?)\s*(\d+)(?:[.-](\d+))?([A-Za-z])?\.?\s*(.*)", re.IGNORECASE)
        match = pattern.match("Figure 1. Patient flow")

        ref = parse_reference_from_match(match, "Figure", ReferenceSource.CAPTION)

        assert ref.type_label == "Figure"
        assert ref.numbers == [1]
        assert ref.is_range is False
        assert ref.suffix is None
        assert ref.source == ReferenceSource.CAPTION

    def test_range_reference(self):
        import re
        pattern = re.compile(r"^(?:Figure|Fig\.?)\s*(\d+)(?:[.-](\d+))?([A-Za-z])?\.?\s*(.*)", re.IGNORECASE)
        match = pattern.match("Figure 2-4. Multiple panels")

        ref = parse_reference_from_match(match, "Figure", ReferenceSource.CAPTION)

        assert ref.numbers == [2, 3, 4]
        assert ref.is_range is True

    def test_reference_with_suffix(self):
        import re
        pattern = re.compile(r"^(?:Figure|Fig\.?)\s*(\d+)(?:[.-](\d+))?([A-Za-z])?\.?\s*(.*)", re.IGNORECASE)
        match = pattern.match("Figure 1A. Subgroup A")

        ref = parse_reference_from_match(match, "Figure", ReferenceSource.CAPTION)

        assert ref.numbers == [1]
        assert ref.suffix == "A"


# -------------------------
# Geometry Helper Tests
# -------------------------


class TestHasHorizontalOverlap:
    def test_full_overlap(self):
        bbox1 = (100.0, 0.0, 200.0, 50.0)
        bbox2 = (100.0, 60.0, 200.0, 100.0)
        assert has_horizontal_overlap(bbox1, bbox2)

    def test_partial_overlap(self):
        bbox1 = (100.0, 0.0, 200.0, 50.0)
        bbox2 = (150.0, 60.0, 250.0, 100.0)
        assert has_horizontal_overlap(bbox1, bbox2)

    def test_no_overlap(self):
        bbox1 = (100.0, 0.0, 200.0, 50.0)
        bbox2 = (300.0, 60.0, 400.0, 100.0)
        assert not has_horizontal_overlap(bbox1, bbox2)

    def test_minimal_overlap_below_threshold(self):
        bbox1 = (100.0, 0.0, 200.0, 50.0)
        bbox2 = (190.0, 60.0, 300.0, 100.0)  # Only 10pt overlap
        # 10 / 100 = 0.1 < 0.3 threshold
        assert not has_horizontal_overlap(bbox1, bbox2)


class TestHasVerticalOverlap:
    def test_full_overlap(self):
        bbox1 = (0.0, 100.0, 50.0, 200.0)
        bbox2 = (60.0, 100.0, 100.0, 200.0)
        assert has_vertical_overlap(bbox1, bbox2)

    def test_partial_overlap(self):
        bbox1 = (0.0, 100.0, 50.0, 200.0)
        bbox2 = (60.0, 150.0, 100.0, 250.0)
        assert has_vertical_overlap(bbox1, bbox2)

    def test_no_overlap(self):
        bbox1 = (0.0, 100.0, 50.0, 200.0)
        bbox2 = (60.0, 300.0, 100.0, 400.0)
        assert not has_vertical_overlap(bbox1, bbox2)


class TestGetRelativePosition:
    def test_caption_below(self):
        visual_bbox = (100.0, 100.0, 400.0, 300.0)  # Visual
        text_bbox = (100.0, 310.0, 400.0, 330.0)    # Caption below
        zones = CaptionSearchZones()

        position, distance = get_relative_position(visual_bbox, text_bbox, zones)

        assert position == "below"
        assert distance == 10.0  # 310 - 300

    def test_caption_above(self):
        visual_bbox = (100.0, 100.0, 400.0, 300.0)  # Visual
        text_bbox = (100.0, 70.0, 400.0, 90.0)      # Caption above
        zones = CaptionSearchZones()

        position, distance = get_relative_position(visual_bbox, text_bbox, zones)

        assert position == "above"
        assert distance == 10.0  # 100 - 90

    def test_caption_left(self):
        visual_bbox = (200.0, 100.0, 400.0, 300.0)  # Visual
        text_bbox = (150.0, 150.0, 190.0, 250.0)    # Caption to the left
        zones = CaptionSearchZones()

        position, distance = get_relative_position(visual_bbox, text_bbox, zones)

        assert position == "left"
        assert distance == 10.0  # 200 - 190

    def test_caption_too_far(self):
        visual_bbox = (100.0, 100.0, 400.0, 300.0)  # Visual
        text_bbox = (100.0, 500.0, 400.0, 520.0)    # Caption way below
        zones = CaptionSearchZones()  # Default 72pt below

        position, distance = get_relative_position(visual_bbox, text_bbox, zones)

        assert position is None
        assert distance == float("inf")

    def test_no_horizontal_overlap(self):
        visual_bbox = (100.0, 100.0, 200.0, 300.0)  # Visual
        text_bbox = (400.0, 310.0, 500.0, 330.0)    # No x overlap
        zones = CaptionSearchZones()

        position, distance = get_relative_position(visual_bbox, text_bbox, zones)

        assert position is None


# -------------------------
# Column Layout Tests
# -------------------------


class TestColumnLayout:
    def test_single_column(self):
        layout = ColumnLayout(
            columns=[(0, 612)],
            page_width=612.0,
            page_height=792.0,
        )
        assert layout.is_single_column
        assert not layout.is_two_column

    def test_two_column(self):
        layout = ColumnLayout(
            columns=[(0, 290), (320, 612)],
            page_width=612.0,
            page_height=792.0,
        )
        assert not layout.is_single_column
        assert layout.is_two_column


class TestGetColumnForX:
    def test_left_column(self):
        layout = ColumnLayout(
            columns=[(0, 290), (320, 612)],
            page_width=612.0,
            page_height=792.0,
        )
        assert get_column_for_x(100.0, layout) == 0

    def test_right_column(self):
        layout = ColumnLayout(
            columns=[(0, 290), (320, 612)],
            page_width=612.0,
            page_height=792.0,
        )
        assert get_column_for_x(400.0, layout) == 1

    def test_between_columns_nearest(self):
        layout = ColumnLayout(
            columns=[(0, 290), (320, 612)],
            page_width=612.0,
            page_height=792.0,
        )
        # 305 is between columns, closer to right column
        assert get_column_for_x(305.0, layout) == 0  # Nearer to col 0 center (145)


# -------------------------
# Caption Selection Tests
# -------------------------


class TestSelectBestCaption:
    def test_prefer_higher_confidence(self):
        candidates = [
            CaptionCandidate(
                text="Table 1",
                bbox_pts=(0, 0, 100, 20),
                provenance=CaptionProvenance.PDF_TEXT,
                position="below",
                distance_pts=50.0,
                confidence=0.6,
                parsed_reference=VisualReference(
                    raw_string="Table 1",
                    type_label="Table",
                    numbers=[1],
                    source=ReferenceSource.CAPTION,
                ),
            ),
            CaptionCandidate(
                text="Table 1. Demographics",
                bbox_pts=(0, 0, 200, 20),
                provenance=CaptionProvenance.PDF_TEXT,
                position="below",
                distance_pts=10.0,
                confidence=0.95,
                parsed_reference=VisualReference(
                    raw_string="Table 1",
                    type_label="Table",
                    numbers=[1],
                    source=ReferenceSource.CAPTION,
                ),
            ),
        ]

        best = select_best_caption(candidates)
        assert best.text == "Table 1. Demographics"

    def test_prefer_pdf_text_over_ocr(self):
        candidates = [
            CaptionCandidate(
                text="Figure 1",
                bbox_pts=(0, 0, 100, 20),
                provenance=CaptionProvenance.OCR,
                position="below",
                distance_pts=10.0,
                confidence=0.90,
                parsed_reference=VisualReference(
                    raw_string="Figure 1",
                    type_label="Figure",
                    numbers=[1],
                    source=ReferenceSource.CAPTION,
                ),
            ),
            CaptionCandidate(
                text="Figure 1. Results",
                bbox_pts=(0, 0, 150, 20),
                provenance=CaptionProvenance.PDF_TEXT,
                position="below",
                distance_pts=10.0,
                confidence=0.90,
                parsed_reference=VisualReference(
                    raw_string="Figure 1",
                    type_label="Figure",
                    numbers=[1],
                    source=ReferenceSource.CAPTION,
                ),
            ),
        ]

        best = select_best_caption(candidates)
        assert best.provenance == CaptionProvenance.PDF_TEXT

    def test_prefer_matching_visual_type(self):
        candidates = [
            CaptionCandidate(
                text="Table 1",
                bbox_pts=(0, 0, 100, 20),
                provenance=CaptionProvenance.PDF_TEXT,
                position="below",
                distance_pts=10.0,
                confidence=0.95,
                parsed_reference=VisualReference(
                    raw_string="Table 1",
                    type_label="Table",
                    numbers=[1],
                    source=ReferenceSource.CAPTION,
                ),
            ),
            CaptionCandidate(
                text="Figure 1",
                bbox_pts=(0, 0, 100, 20),
                provenance=CaptionProvenance.PDF_TEXT,
                position="below",
                distance_pts=10.0,
                confidence=0.95,
                parsed_reference=VisualReference(
                    raw_string="Figure 1",
                    type_label="Figure",
                    numbers=[1],
                    source=ReferenceSource.CAPTION,
                ),
            ),
        ]

        best = select_best_caption(candidates, visual_type_hint="figure")
        assert best.parsed_reference.type_label == "Figure"

    def test_prefer_closer_distance(self):
        ref = VisualReference(
            raw_string="Table 1",
            type_label="Table",
            numbers=[1],
            source=ReferenceSource.CAPTION,
        )
        candidates = [
            CaptionCandidate(
                text="Table 1 (far)",
                bbox_pts=(0, 0, 100, 20),
                provenance=CaptionProvenance.PDF_TEXT,
                position="below",
                distance_pts=50.0,
                confidence=0.95,
                parsed_reference=ref,
            ),
            CaptionCandidate(
                text="Table 1 (close)",
                bbox_pts=(0, 0, 100, 20),
                provenance=CaptionProvenance.PDF_TEXT,
                position="below",
                distance_pts=5.0,
                confidence=0.95,
                parsed_reference=ref,
            ),
        ]

        best = select_best_caption(candidates)
        assert "close" in best.text

    def test_empty_candidates(self):
        best = select_best_caption([])
        assert best is None

    def test_position_preference_for_tables(self):
        ref = VisualReference(
            raw_string="Table 1",
            type_label="Table",
            numbers=[1],
            source=ReferenceSource.CAPTION,
        )
        candidates = [
            CaptionCandidate(
                text="Table 1 (below)",
                bbox_pts=(0, 0, 100, 20),
                provenance=CaptionProvenance.PDF_TEXT,
                position="below",
                distance_pts=10.0,
                confidence=0.95,
                parsed_reference=ref,
            ),
            CaptionCandidate(
                text="Table 1 (above)",
                bbox_pts=(0, 0, 100, 20),
                provenance=CaptionProvenance.PDF_TEXT,
                position="above",
                distance_pts=10.0,
                confidence=0.95,
                parsed_reference=ref,
            ),
        ]

        # Tables prefer captions above
        best = select_best_caption(candidates, visual_type_hint="table")
        assert "above" in best.text
