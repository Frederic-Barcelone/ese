# corpus_metadata/K_tests/K18_detector.py
"""
Tests for visual detector with FAST/ACCURATE tiering.

Note: Full Docling integration tests require Docling to be installed.
These tests focus on the detection logic without Docling dependency.
"""
import pytest

from B_parsing.B13_visual_detector import (
    DetectorConfig,
    _compute_bbox_overlap,
    _has_nearby_caption,
)
from A_core.A13_visual_models import (
    CaptionCandidate,
    CaptionProvenance,
    ReferenceSource,
    VisualReference,
)


# -------------------------
# Configuration Tests
# -------------------------


class TestDetectorConfig:
    def test_default_config(self):
        config = DetectorConfig()

        assert config.default_table_mode == "fast"
        assert config.enable_escalation is True
        assert config.min_figure_area_ratio == 0.02
        assert config.filter_noise is True
        assert config.repeat_threshold == 3

    def test_custom_config(self):
        config = DetectorConfig(
            default_table_mode="accurate",
            enable_escalation=False,
            min_figure_area_ratio=0.05,
        )

        assert config.default_table_mode == "accurate"
        assert config.enable_escalation is False
        assert config.min_figure_area_ratio == 0.05


# -------------------------
# Bbox Overlap Tests
# -------------------------


class TestBboxOverlap:
    def test_full_overlap(self):
        bbox1 = (100.0, 100.0, 200.0, 200.0)
        bbox2 = (100.0, 100.0, 200.0, 200.0)

        overlap = _compute_bbox_overlap(bbox1, bbox2)

        assert overlap == pytest.approx(1.0)

    def test_no_overlap(self):
        bbox1 = (100.0, 100.0, 200.0, 200.0)
        bbox2 = (300.0, 300.0, 400.0, 400.0)

        overlap = _compute_bbox_overlap(bbox1, bbox2)

        assert overlap == 0.0

    def test_partial_overlap(self):
        bbox1 = (100.0, 100.0, 200.0, 200.0)  # 100x100 = 10000
        bbox2 = (150.0, 150.0, 250.0, 250.0)  # 100x100 = 10000
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500 / 17500 ≈ 0.143

        overlap = _compute_bbox_overlap(bbox1, bbox2)

        assert overlap == pytest.approx(0.143, rel=0.01)

    def test_contained_bbox(self):
        bbox1 = (100.0, 100.0, 300.0, 300.0)  # 200x200
        bbox2 = (150.0, 150.0, 250.0, 250.0)  # 100x100, inside bbox1
        # Intersection = 10000
        # Union = 40000 + 10000 - 10000 = 40000
        # IoU = 10000 / 40000 = 0.25

        overlap = _compute_bbox_overlap(bbox1, bbox2)

        assert overlap == pytest.approx(0.25)


# -------------------------
# Caption Detection Tests
# -------------------------


def make_caption(
    x0: float, y0: float, x1: float, y1: float, text: str = "Table 1"
) -> CaptionCandidate:
    """Helper to create caption candidates."""
    return CaptionCandidate(
        text=text,
        bbox_pts=(x0, y0, x1, y1),
        provenance=CaptionProvenance.PDF_TEXT,
        position="below",
        distance_pts=10.0,
        confidence=0.95,
        parsed_reference=VisualReference(
            raw_string=text,
            type_label="Table",
            numbers=[1],
            source=ReferenceSource.CAPTION,
        ),
    )


class TestHasNearbyCaption:
    def test_caption_below_visual(self):
        visual_bbox = (100.0, 100.0, 300.0, 200.0)
        caption = make_caption(100.0, 210.0, 300.0, 230.0)  # 10pt below

        has_caption = _has_nearby_caption(visual_bbox, [caption])

        assert has_caption is True

    def test_caption_above_visual(self):
        visual_bbox = (100.0, 100.0, 300.0, 200.0)
        caption = make_caption(100.0, 70.0, 300.0, 90.0)  # 10pt above

        has_caption = _has_nearby_caption(visual_bbox, [caption])

        assert has_caption is True

    def test_caption_too_far(self):
        visual_bbox = (100.0, 100.0, 300.0, 200.0)
        caption = make_caption(100.0, 400.0, 300.0, 420.0)  # 200pt below

        has_caption = _has_nearby_caption(visual_bbox, [caption])

        assert has_caption is False

    def test_caption_no_horizontal_overlap(self):
        visual_bbox = (100.0, 100.0, 200.0, 200.0)
        caption = make_caption(300.0, 210.0, 400.0, 230.0)  # Different column

        has_caption = _has_nearby_caption(visual_bbox, [caption])

        assert has_caption is False

    def test_no_captions(self):
        visual_bbox = (100.0, 100.0, 300.0, 200.0)

        has_caption = _has_nearby_caption(visual_bbox, [])

        assert has_caption is False

    def test_multiple_captions_one_nearby(self):
        visual_bbox = (100.0, 100.0, 300.0, 200.0)
        captions = [
            make_caption(400.0, 210.0, 500.0, 230.0, "Table 1"),  # Far
            make_caption(100.0, 210.0, 300.0, 230.0, "Table 2"),  # Near
        ]

        has_caption = _has_nearby_caption(visual_bbox, captions)

        assert has_caption is True

    def test_custom_distance_threshold(self):
        visual_bbox = (100.0, 100.0, 300.0, 200.0)
        caption = make_caption(100.0, 250.0, 300.0, 270.0)  # 50pt below

        # Default threshold is 72pt
        has_caption_default = _has_nearby_caption(visual_bbox, [caption])
        assert has_caption_default is True

        # Stricter threshold
        has_caption_strict = _has_nearby_caption(visual_bbox, [caption], max_distance_pts=30.0)
        assert has_caption_strict is False
