# corpus_metadata/K_tests/K14_zone_expander.py
"""Tests for zone expander (whitespace-based bbox computation)."""
from typing import Any

import pytest
from A_core.A24_layout_models import (
    LayoutPattern,
    PageLayout,
    VisualPosition,
    VisualZone,
)
from B_parsing.B20_zone_expander import (
    compute_column_boundaries,
    expand_zone_to_whitespace,
    expand_all_zones,
    ExpandedVisual,
    _parse_vertical_zone,
)


class TestComputeColumnBoundaries:
    """Tests for column boundary detection from text blocks."""

    def test_two_column_detection(self):
        """Detect 2-column layout from text blocks."""
        # Mock text blocks: left column (x: 50-280), right column (x: 320-550)
        # Page width: 612 (US Letter)
        text_blocks = [
            {"bbox": (50, 100, 280, 120)},   # Left col
            {"bbox": (50, 130, 280, 150)},   # Left col
            {"bbox": (320, 100, 550, 120)},  # Right col
            {"bbox": (320, 130, 550, 150)},  # Right col
        ]
        page_width = 612.0

        boundaries = compute_column_boundaries(text_blocks, page_width)

        # Should find gap around x=300 (between 280 and 320)
        assert boundaries is not None
        assert 0.45 < boundaries["split"] < 0.55  # Around 50%

    def test_single_column_detection(self):
        """Detect single column layout."""
        # Text spans full width
        text_blocks = [
            {"bbox": (50, 100, 550, 120)},
            {"bbox": (50, 130, 550, 150)},
        ]
        page_width = 612.0

        boundaries = compute_column_boundaries(text_blocks, page_width)

        # No column split found
        assert boundaries is None

    def test_empty_text_blocks(self):
        """Handle empty text blocks list."""
        boundaries = compute_column_boundaries([], 612.0)
        assert boundaries is None

    def test_no_gap_in_middle(self):
        """No column boundary if gap is not in middle region."""
        # Gap at far left
        text_blocks = [
            {"bbox": (200, 100, 550, 120)},
            {"bbox": (200, 130, 550, 150)},
        ]
        page_width = 612.0

        boundaries = compute_column_boundaries(text_blocks, page_width)
        assert boundaries is None


class TestParseVerticalZone:
    """Tests for parsing vertical zone strings."""

    def test_parse_top(self):
        """Parse 'top' zone."""
        y0, y1 = _parse_vertical_zone("top", 792.0)
        assert y0 == 0
        assert y1 == pytest.approx(792.0 * 0.4)

    def test_parse_middle(self):
        """Parse 'middle' zone."""
        y0, y1 = _parse_vertical_zone("middle", 792.0)
        assert y0 == pytest.approx(792.0 * 0.3)
        assert y1 == pytest.approx(792.0 * 0.7)

    def test_parse_bottom(self):
        """Parse 'bottom' zone."""
        y0, y1 = _parse_vertical_zone("bottom", 792.0)
        assert y0 == pytest.approx(792.0 * 0.6)
        assert y1 == 792.0

    def test_parse_numeric_range(self):
        """Parse numeric range like '0.2-0.6'."""
        y0, y1 = _parse_vertical_zone("0.2-0.6", 792.0)
        assert y0 == pytest.approx(792.0 * 0.2)
        assert y1 == pytest.approx(792.0 * 0.6)

    def test_parse_invalid_defaults_to_full(self):
        """Invalid zone string defaults to full page."""
        y0, y1 = _parse_vertical_zone("invalid", 792.0)
        assert y0 == 0
        assert y1 == 792.0


class TestExpandZoneToWhitespace:
    """Tests for expanding visual zones to whitespace boundaries."""

    def test_expand_left_column_visual(self):
        """Expand visual in left column to whitespace."""
        zone = VisualZone(
            visual_type="figure",
            label="Figure 1",
            position=VisualPosition.LEFT,
            vertical_zone="0.2-0.6",
            confidence=0.9,
        )
        layout = PageLayout(
            page_num=3,
            pattern=LayoutPattern.TWO_COL,
            column_boundary=0.5,
            margin_left=0.05,
            margin_right=0.95,
            visuals=[zone],
        )
        page_width = 612.0
        page_height = 792.0
        text_blocks: list[Any] = []

        result = expand_zone_to_whitespace(
            zone=zone,
            layout=layout,
            page_width=page_width,
            page_height=page_height,
            text_blocks=text_blocks,
            other_visuals=[],
        )

        assert isinstance(result, ExpandedVisual)
        # Should be constrained to left column (margin to column_boundary)
        assert result.bbox_pts[0] >= 0  # Can have padding
        assert result.bbox_pts[2] <= page_width * 0.5 + 20  # Column boundary + padding
        assert result.layout_code == "2col"
        assert result.position_code == "L"

    def test_expand_right_column_visual(self):
        """Expand visual in right column."""
        zone = VisualZone(
            visual_type="table",
            label="Table 1",
            position=VisualPosition.RIGHT,
            vertical_zone="middle",
            confidence=0.9,
        )
        layout = PageLayout(
            page_num=3,
            pattern=LayoutPattern.TWO_COL,
            column_boundary=0.5,
            margin_left=0.05,
            margin_right=0.95,
            visuals=[zone],
        )
        page_width = 612.0
        page_height = 792.0

        result = expand_zone_to_whitespace(
            zone=zone,
            layout=layout,
            page_width=page_width,
            page_height=page_height,
            text_blocks=[],
            other_visuals=[],
        )

        # Should start from column boundary
        assert result.bbox_pts[0] >= page_width * 0.5 - 20  # Column boundary - padding
        assert result.position_code == "R"

    def test_expand_full_width_visual(self):
        """Expand full-width visual."""
        zone = VisualZone(
            visual_type="table",
            label="Table 1",
            position=VisualPosition.FULL,
            vertical_zone="bottom",
            confidence=0.9,
        )
        layout = PageLayout(
            page_num=5,
            pattern=LayoutPattern.TWO_COL_FULLBOT,
            column_boundary=0.5,
            margin_left=0.05,
            margin_right=0.95,
            visuals=[zone],
        )
        page_width = 612.0
        page_height = 792.0

        result = expand_zone_to_whitespace(
            zone=zone,
            layout=layout,
            page_width=page_width,
            page_height=page_height,
            text_blocks=[],
            other_visuals=[],
        )

        # Should span full width (margins)
        assert result.bbox_pts[0] <= page_width * 0.1   # Near left margin
        assert result.bbox_pts[2] >= page_width * 0.9   # Near right margin
        assert result.position_code == "F"

    def test_expanded_visual_has_normalized_bbox(self):
        """ExpandedVisual includes normalized bbox."""
        zone = VisualZone(
            visual_type="figure",
            position=VisualPosition.FULL,
            vertical_zone="middle",
        )
        layout = PageLayout(page_num=1, pattern=LayoutPattern.FULL)

        result = expand_zone_to_whitespace(
            zone=zone,
            layout=layout,
            page_width=612.0,
            page_height=792.0,
            text_blocks=[],
            other_visuals=[],
        )

        # Normalized bbox should be between 0 and 1
        x0, y0, x1, y1 = result.bbox_normalized
        assert 0 <= x0 <= 1
        assert 0 <= y0 <= 1
        assert 0 <= x1 <= 1
        assert 0 <= y1 <= 1


class TestExpandAllZones:
    """Tests for expanding all zones on a page."""

    def test_expand_multiple_zones(self):
        """Expand multiple zones on same page."""
        zones = [
            VisualZone(
                visual_type="figure",
                label="Figure 1",
                position=VisualPosition.LEFT,
                vertical_zone="top",
            ),
            VisualZone(
                visual_type="table",
                label="Table 1",
                position=VisualPosition.RIGHT,
                vertical_zone="middle",
            ),
        ]
        layout = PageLayout(
            page_num=3,
            pattern=LayoutPattern.TWO_COL,
            column_boundary=0.5,
            visuals=zones,
        )

        results = expand_all_zones(
            layout=layout,
            page_width=612.0,
            page_height=792.0,
            text_blocks=[],
        )

        assert len(results) == 2
        assert results[0].zone.label == "Figure 1"
        assert results[1].zone.label == "Table 1"

    def test_zones_sorted_by_position(self):
        """Zones are sorted top-to-bottom, left-to-right."""
        zones = [
            VisualZone(
                visual_type="figure",
                label="Bottom",
                position=VisualPosition.LEFT,
                vertical_zone="bottom",
            ),
            VisualZone(
                visual_type="figure",
                label="Top",
                position=VisualPosition.LEFT,
                vertical_zone="top",
            ),
        ]
        layout = PageLayout(
            page_num=1,
            pattern=LayoutPattern.TWO_COL,
            column_boundary=0.5,
            visuals=zones,
        )

        results = expand_all_zones(
            layout=layout,
            page_width=612.0,
            page_height=792.0,
            text_blocks=[],
        )

        # Top should come before Bottom
        assert results[0].zone.label == "Top"
        assert results[1].zone.label == "Bottom"

    def test_empty_visuals_returns_empty(self):
        """Empty visuals list returns empty results."""
        layout = PageLayout(
            page_num=1,
            pattern=LayoutPattern.FULL,
            visuals=[],
        )

        results = expand_all_zones(
            layout=layout,
            page_width=612.0,
            page_height=792.0,
            text_blocks=[],
        )

        assert results == []


class TestExpandedVisualDataclass:
    """Tests for ExpandedVisual dataclass."""

    def test_expanded_visual_creation(self):
        """Create ExpandedVisual with all fields."""
        zone = VisualZone(
            visual_type="figure",
            position=VisualPosition.LEFT,
            vertical_zone="top",
        )

        expanded = ExpandedVisual(
            zone=zone,
            bbox_pts=(50.0, 100.0, 280.0, 400.0),
            bbox_normalized=(0.08, 0.13, 0.46, 0.51),
            layout_code="2col",
            position_code="L",
        )

        assert expanded.zone == zone
        assert expanded.bbox_pts == (50.0, 100.0, 280.0, 400.0)
        assert expanded.bbox_normalized == (0.08, 0.13, 0.46, 0.51)
        assert expanded.layout_code == "2col"
        assert expanded.position_code == "L"
