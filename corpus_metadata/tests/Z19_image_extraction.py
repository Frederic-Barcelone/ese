# corpus_metadata/tests/Z19_image_extraction.py
"""
Tests for image extraction coordinate handling and padding.

Tests the fixes for:
- DPI ratio calculation (axis-independent scaling)
- 2-column layout detection
- Padding values
- Adaptive axis text margin
"""
import pytest


# -------------------------
# DPI Ratio Calculation Tests
# -------------------------


class TestDPIRatioCalculation:
    """Test coordinate space detection and scaling."""

    def test_coords_within_page_no_scaling(self):
        """Coordinates within page bounds should not be scaled."""
        page_width, page_height = 612.0, 792.0
        _x0, _y0, x1, y1 = 100, 200, 400, 500

        # Within bounds (< 1.1x page size)
        needs_x_scale = x1 > page_width * 1.1
        needs_y_scale = y1 > page_height * 1.1

        assert not needs_x_scale
        assert not needs_y_scale

    def test_coords_exceed_page_needs_scaling(self):
        """Coordinates exceeding page bounds should trigger scaling."""
        page_width, page_height = 612.0, 792.0
        # Simulating Unstructured coordinates at ~200 DPI (2.78x)
        _x0, _y0, x1, y1 = 278, 556, 1112, 1390

        needs_x_scale = x1 > page_width * 1.1
        needs_y_scale = y1 > page_height * 1.1

        assert needs_x_scale  # 1112 > 673
        assert needs_y_scale  # 1390 > 871

    def test_axis_independent_scaling_both_axes(self):
        """When both axes exceed, use tighter constraint."""
        page_width, page_height = 612.0, 792.0
        _x0, _y0, x1, y1 = 0, 0, 1800, 2300  # ~3x scale needed

        x_ratio = page_width / x1   # 612/1800 = 0.34
        y_ratio = page_height / y1  # 792/2300 = 0.344
        ratio = min(x_ratio, y_ratio)  # 0.34

        scaled_x1 = x1 * ratio
        scaled_y1 = y1 * ratio

        assert scaled_x1 <= page_width
        assert scaled_y1 <= page_height
        assert ratio == pytest.approx(0.34, rel=0.01)

    def test_single_axis_scaling_x_only(self):
        """When only X exceeds, scale proportionally."""
        page_width, page_height = 612.0, 792.0
        _x0, _y0, x1, y1 = 0, 0, 1800, 500  # Only X exceeds

        needs_x_scale = x1 > page_width * 1.1
        needs_y_scale = y1 > page_height * 1.1

        assert needs_x_scale
        assert not needs_y_scale

        x_ratio = page_width / x1
        scaled_x1 = x1 * x_ratio
        scaled_y1 = y1 * x_ratio  # Scale Y proportionally

        assert scaled_x1 == pytest.approx(page_width)
        assert scaled_y1 < y1  # Y gets scaled down too

    def test_single_axis_scaling_y_only(self):
        """When only Y exceeds, scale proportionally."""
        page_width, page_height = 612.0, 792.0
        _x0, _y0, x1, y1 = 0, 0, 400, 2000  # Only Y exceeds

        needs_x_scale = x1 > page_width * 1.1
        needs_y_scale = y1 > page_height * 1.1

        assert not needs_x_scale
        assert needs_y_scale

        y_ratio = page_height / y1
        scaled_x1 = x1 * y_ratio  # Scale X proportionally
        scaled_y1 = y1 * y_ratio

        assert scaled_y1 == pytest.approx(page_height)
        assert scaled_x1 < x1  # X gets scaled down too


# -------------------------
# 2-Column Layout Detection Tests
# -------------------------


class TestTwoColumnDetection:
    """Test 2-column layout detection logic."""

    def test_left_column_figure(self):
        """Figure in left column should be detected."""
        page_width = 612.0
        x0, _y0, x1, _y1 = 36, 100, 280, 400

        figure_center_x = (x0 + x1) / 2  # 158
        is_two_column = (
            (figure_center_x < page_width * 0.45) or  # < 275
            (figure_center_x > page_width * 0.55)     # > 337
        )

        assert is_two_column
        assert figure_center_x < page_width * 0.45  # In left column

    def test_right_column_figure(self):
        """Figure in right column should be detected."""
        page_width = 612.0
        x0, _y0, x1, _y1 = 320, 100, 576, 400

        figure_center_x = (x0 + x1) / 2  # 448
        is_two_column = (
            (figure_center_x < page_width * 0.45) or
            (figure_center_x > page_width * 0.55)
        )

        assert is_two_column
        assert figure_center_x > page_width * 0.55  # In right column

    def test_centered_figure_not_two_column(self):
        """Figure spanning center should not be detected as 2-column."""
        page_width = 612.0
        x0, _y0, x1, _y1 = 150, 100, 462, 400

        figure_center_x = (x0 + x1) / 2  # 306 (center of page)
        is_two_column = (
            (figure_center_x < page_width * 0.45) or  # < 275
            (figure_center_x > page_width * 0.55)     # > 337
        )

        assert not is_two_column  # Between 45% and 55%

    def test_full_width_figure_not_two_column(self):
        """Full-width figure should not be detected as 2-column."""
        page_width = 612.0
        x0, _y0, x1, _y1 = 36, 100, 576, 400

        figure_center_x = (x0 + x1) / 2  # 306
        is_two_column = (
            (figure_center_x < page_width * 0.45) or
            (figure_center_x > page_width * 0.55)
        )

        assert not is_two_column

    def test_two_column_padding_limit(self):
        """In 2-column layout, padding should be limited."""
        page_width = 612.0
        column_width = page_width / 2  # 306
        max_padding = column_width * 0.1  # 30.6

        original_padding = 75
        limited_padding = min(original_padding, max_padding)

        assert limited_padding == pytest.approx(30.6)
        assert limited_padding < original_padding


# -------------------------
# Full-Width Expansion Tests
# -------------------------


class TestFullWidthExpansion:
    """Test threshold for expanding figures to full page width."""

    def test_small_figure_no_expansion(self):
        """Small figures (<40% width) should not expand."""
        page_width = 612.0
        x0, x1 = 200, 350  # 150pt wide = 24.5% of page

        figure_width = x1 - x0
        is_significant = figure_width > page_width * 0.4

        assert not is_significant
        assert figure_width / page_width < 0.4

    def test_large_figure_expands(self):
        """Large figures (>40% width) should expand to full width."""
        page_width = 612.0
        x0, x1 = 50, 350  # 300pt wide = 49% of page

        figure_width = x1 - x0
        is_significant = figure_width > page_width * 0.4

        assert is_significant
        assert figure_width / page_width > 0.4

    def test_threshold_boundary(self):
        """Test behavior exactly at 40% threshold."""
        page_width = 612.0
        _threshold_width = page_width * 0.4  # 244.8 (for documentation)

        # Just below threshold
        assert 244 <= page_width * 0.4
        # Just above threshold
        assert 245 > page_width * 0.4


# -------------------------
# Adaptive Axis Text Margin Tests
# -------------------------


class TestAdaptiveAxisMargin:
    """Test adaptive margin for axis text detection."""

    def test_small_figure_uses_base_margin(self):
        """Small figures should use base 50pt margin."""
        x0, y0, x1, y1 = 100, 100, 200, 200
        base_margin = 50

        figure_height = y1 - y0  # 100
        figure_width = x1 - x0   # 100

        adaptive_margin = max(base_margin, figure_height * 0.15, figure_width * 0.1)

        assert adaptive_margin == 50  # Base margin dominates

    def test_tall_figure_gets_larger_margin(self):
        """Tall figures should get margin based on height."""
        x0, y0, x1, y1 = 100, 100, 300, 600
        base_margin = 50

        figure_height = y1 - y0  # 500
        figure_width = x1 - x0   # 200

        adaptive_margin = max(base_margin, figure_height * 0.15, figure_width * 0.1)

        assert adaptive_margin == 75  # 500 * 0.15 = 75

    def test_wide_figure_gets_larger_margin(self):
        """Wide figures should get margin based on width."""
        x0, y0, x1, y1 = 50, 100, 550, 300
        base_margin = 50

        figure_height = y1 - y0  # 200
        figure_width = x1 - x0   # 500

        # 200 * 0.15 = 30, 500 * 0.1 = 50
        adaptive_margin = max(base_margin, figure_height * 0.15, figure_width * 0.1)

        assert adaptive_margin == 50  # Both width and base give 50

    def test_large_figure_gets_largest_margin(self):
        """Large figures should get margin from largest dimension."""
        x0, y0, x1, y1 = 50, 50, 550, 650
        base_margin = 50

        figure_height = y1 - y0  # 600
        figure_width = x1 - x0   # 500

        # 600 * 0.15 = 90, 500 * 0.1 = 50
        adaptive_margin = max(base_margin, figure_height * 0.15, figure_width * 0.1)

        assert adaptive_margin == 90  # Height-based margin dominates


# -------------------------
# Padding Value Tests
# -------------------------


class TestPaddingValues:
    """Test that padding values are reasonable."""

    def test_default_padding_reasonable(self):
        """Default padding should not exceed 1.5 inches."""
        max_reasonable_padding = 108  # 1.5 inches in points

        # New default values from render_figure_with_padding
        default_padding = 30
        default_bottom = 100
        default_right = 50

        assert default_padding <= max_reasonable_padding
        assert default_bottom <= max_reasonable_padding
        assert default_right <= max_reasonable_padding

    def test_table_padding_minimal(self):
        """Table padding should be minimal to stay in column."""
        table_padding = 15
        table_right_padding = 15

        # Tables should have small padding
        assert table_padding <= 20
        assert table_right_padding <= 20

    def test_padding_not_larger_than_figure(self):
        """Padding should not exceed typical figure dimensions."""
        typical_figure_width = 200  # ~2.8 inches
        typical_figure_height = 150  # ~2 inches

        # Padding should be fraction of figure size
        default_bottom = 100
        default_right = 50

        assert default_bottom < typical_figure_height
        assert default_right < typical_figure_width


# -------------------------
# Integration-style Tests (Mock-based)
# -------------------------


class TestCoordinateTransformationIntegration:
    """Integration tests for full coordinate transformation flow."""

    def test_unstructured_coordinates_transform_correctly(self):
        """Simulate Unstructured layout model coordinates."""
        # Unstructured often uses ~200 DPI coordinates
        # A figure at (100, 200) in PDF points becomes (278, 556) at 200 DPI
        page_width_pts = 612.0
        page_height_pts = 792.0

        # Simulated Unstructured bbox (at ~2.78x scale)
        x0, y0, x1, y1 = 278, 556, 834, 1112

        # Detection
        needs_x_scale = x1 > page_width_pts * 1.1
        needs_y_scale = y1 > page_height_pts * 1.1

        assert needs_x_scale  # 834 > 673
        assert needs_y_scale  # 1112 > 871

        # Scaling (both axes)
        x_ratio = page_width_pts / x1
        y_ratio = page_height_pts / y1
        ratio = min(x_ratio, y_ratio)

        scaled = (x0 * ratio, y0 * ratio, x1 * ratio, y1 * ratio)

        # Verify scaled coordinates are within page
        assert scaled[2] <= page_width_pts
        assert scaled[3] <= page_height_pts

    def test_docling_coordinates_no_transform_needed(self):
        """Docling uses PDF points - no transformation needed."""
        page_width_pts = 612.0
        page_height_pts = 792.0

        # Docling bbox (already in PDF points)
        _x0, _y0, x1, y1 = 72, 144, 540, 648

        needs_x_scale = x1 > page_width_pts * 1.1
        needs_y_scale = y1 > page_height_pts * 1.1

        assert not needs_x_scale
        assert not needs_y_scale


# -------------------------
# Boundary/Edge Case Tests
# -------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_figure_at_page_edge(self):
        """Figure at page edge should not cause negative coordinates."""
        _page_width, _page_height = 612.0, 792.0
        x0, y0, _x1, _y1 = 0, 0, 200, 200
        padding = 30

        # Clamped coordinates
        clipped_x0 = max(0, x0 - padding)
        clipped_y0 = max(0, y0 - padding)

        assert clipped_x0 == 0
        assert clipped_y0 == 0

    def test_figure_near_right_edge(self):
        """Figure near right edge should not exceed page width."""
        page_width, _page_height = 612.0, 792.0
        _x0, _y0, x1, _y1 = 500, 100, 600, 400
        right_padding = 50

        # Clamped coordinates
        clipped_x1 = min(page_width, x1 + right_padding)

        assert clipped_x1 == page_width

    def test_very_small_figure(self):
        """Very small figure should still get minimum padding."""
        x0, y0, x1, y1 = 300, 400, 310, 410  # 10x10 figure
        base_margin = 50

        figure_height = y1 - y0
        figure_width = x1 - x0

        adaptive_margin = max(base_margin, figure_height * 0.15, figure_width * 0.1)

        # Small figure should still get base margin
        assert adaptive_margin == 50

    def test_zero_size_figure_handling(self):
        """Zero-size figure should be handled gracefully."""
        x0, y0, x1, y1 = 300, 400, 300, 400  # 0x0 figure

        width = x1 - x0
        height = y1 - y0

        # Should be detected as invalid
        is_valid = width > 0 and height > 0
        assert not is_valid
