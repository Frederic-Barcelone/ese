# corpus_metadata/tests/test_visual_extraction/test_renderer.py
"""
Tests for visual renderer with point-based padding.
"""
import pytest

from B_parsing.B14_visual_renderer import (
    RenderConfig,
    bbox_pts_to_pixels,
    compute_adaptive_dpi,
    compute_caption_padding,
    expand_bbox_with_padding,
    pixels_to_pts,
    pts_to_pixels,
)


# -------------------------
# Coordinate Conversion Tests
# -------------------------


class TestCoordinateConversion:
    def test_pts_to_pixels_at_72_dpi(self):
        # At 72 DPI, 1 point = 1 pixel
        assert pts_to_pixels(72.0, 72) == pytest.approx(72.0)
        assert pts_to_pixels(1.0, 72) == pytest.approx(1.0)

    def test_pts_to_pixels_at_300_dpi(self):
        # At 300 DPI, 1 inch (72 pts) = 300 pixels
        assert pts_to_pixels(72.0, 300) == pytest.approx(300.0)
        assert pts_to_pixels(36.0, 300) == pytest.approx(150.0)

    def test_pixels_to_pts_at_72_dpi(self):
        assert pixels_to_pts(72.0, 72) == pytest.approx(72.0)

    def test_pixels_to_pts_at_300_dpi(self):
        assert pixels_to_pts(300.0, 300) == pytest.approx(72.0)
        assert pixels_to_pts(150.0, 300) == pytest.approx(36.0)

    def test_roundtrip_conversion(self):
        original_pts = 100.0
        dpi = 300
        pixels = pts_to_pixels(original_pts, dpi)
        back_to_pts = pixels_to_pts(pixels, dpi)
        assert back_to_pts == pytest.approx(original_pts)

    def test_bbox_pts_to_pixels(self):
        bbox_pts = (72.0, 144.0, 216.0, 288.0)  # 1 inch x 2 inches at origin
        dpi = 300

        bbox_px = bbox_pts_to_pixels(bbox_pts, dpi)

        assert bbox_px == (300, 600, 900, 1200)


# -------------------------
# Adaptive DPI Tests
# -------------------------


class TestAdaptiveDPI:
    def test_small_visual_gets_high_dpi(self):
        # Very small visual (50pt x 50pt = 2500 sq pts)
        bbox = (100.0, 100.0, 150.0, 150.0)
        config = RenderConfig()

        dpi = compute_adaptive_dpi(bbox, config)

        assert dpi == config.max_dpi  # 400

    def test_large_visual_gets_low_dpi(self):
        # Large visual (400pt x 400pt = 160000 sq pts)
        bbox = (0.0, 0.0, 400.0, 400.0)
        config = RenderConfig()

        dpi = compute_adaptive_dpi(bbox, config)

        assert dpi == config.min_dpi  # 200

    def test_medium_visual_gets_default_dpi(self):
        # Medium visual (200pt x 200pt = 40000 sq pts)
        bbox = (0.0, 0.0, 200.0, 200.0)
        config = RenderConfig()

        dpi = compute_adaptive_dpi(bbox, config)

        assert dpi == config.default_dpi  # 300


# -------------------------
# Padding Tests
# -------------------------


class TestExpandBboxWithPadding:
    def test_simple_padding(self):
        bbox = (100.0, 100.0, 300.0, 400.0)
        page_width = 612.0
        page_height = 792.0

        expanded = expand_bbox_with_padding(
            bbox, page_width, page_height,
            padding_sides_pts=12.0,
            padding_top_pts=6.0,
            padding_bottom_pts=72.0,
        )

        assert expanded[0] == 88.0  # x0 - 12
        assert expanded[1] == 94.0  # y0 - 6
        assert expanded[2] == 312.0  # x1 + 12
        assert expanded[3] == 472.0  # y1 + 72

    def test_clamped_to_page_bounds(self):
        bbox = (10.0, 10.0, 600.0, 780.0)  # Near edges
        page_width = 612.0
        page_height = 792.0

        expanded = expand_bbox_with_padding(
            bbox, page_width, page_height,
            padding_sides_pts=50.0,
            padding_top_pts=50.0,
            padding_bottom_pts=50.0,
        )

        assert expanded[0] == 0.0  # Clamped to 0
        assert expanded[1] == 0.0  # Clamped to 0
        assert expanded[2] == 612.0  # Clamped to page width
        assert expanded[3] == 792.0  # Clamped to page height

    def test_zero_padding(self):
        bbox = (100.0, 100.0, 200.0, 200.0)
        page_width = 612.0
        page_height = 792.0

        expanded = expand_bbox_with_padding(
            bbox, page_width, page_height,
            padding_sides_pts=0.0,
            padding_top_pts=0.0,
            padding_bottom_pts=0.0,
        )

        assert expanded == bbox


class TestComputeCaptionPadding:
    def test_figure_default_caption_below(self):
        padding = compute_caption_padding("figure", caption_position=None)

        assert padding["bottom"] == 72.0  # Caption zone
        assert padding["top"] == 12.0  # Normal padding
        assert padding["left"] == 12.0
        assert padding["right"] == 12.0

    def test_table_default_caption_above(self):
        padding = compute_caption_padding("table", caption_position=None)

        assert padding["top"] == 72.0  # Caption zone
        assert padding["bottom"] == 12.0  # Normal padding

    def test_explicit_caption_above(self):
        padding = compute_caption_padding("figure", caption_position="above")

        assert padding["top"] == 72.0  # Caption zone
        assert padding["bottom"] == 12.0

    def test_explicit_caption_left(self):
        padding = compute_caption_padding("figure", caption_position="left")

        assert padding["left"] == 72.0  # Caption zone
        assert padding["right"] == 12.0


# -------------------------
# Render Config Tests
# -------------------------


class TestRenderConfig:
    def test_default_values(self):
        config = RenderConfig()

        assert config.default_dpi == 300
        assert config.min_dpi == 200
        assert config.max_dpi == 400
        assert config.padding_sides_pts == 12.0
        assert config.padding_caption_pts == 72.0
        assert config.image_format == "png"

    def test_custom_config(self):
        config = RenderConfig(
            default_dpi=200,
            padding_sides_pts=20.0,
            image_format="jpeg",
        )

        assert config.default_dpi == 200
        assert config.padding_sides_pts == 20.0
        assert config.image_format == "jpeg"
