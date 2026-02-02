# corpus_metadata/tests/Z07_visual_models_layout.py
"""Tests for layout-aware fields in visual models."""

from A_core.A13_visual_models import (
    ExtractedVisual,
    PageLocation,
    VisualCandidate,
    VisualType,
)


class TestVisualCandidateLayoutFields:
    """Tests for layout-aware fields in VisualCandidate."""

    def test_layout_aware_source(self):
        """VisualCandidate accepts 'layout_aware' source."""
        candidate = VisualCandidate(
            source="layout_aware",
            page_num=1,
            bbox_pts=(50, 100, 280, 400),
            page_width_pts=612.0,
            page_height_pts=792.0,
            area_ratio=0.2,
        )
        assert candidate.source == "layout_aware"

    def test_layout_code_field(self):
        """VisualCandidate has layout_code field."""
        candidate = VisualCandidate(
            source="layout_aware",
            page_num=1,
            bbox_pts=(50, 100, 280, 400),
            page_width_pts=612.0,
            page_height_pts=792.0,
            area_ratio=0.2,
            layout_code="2col",
        )
        assert candidate.layout_code == "2col"

    def test_position_code_field(self):
        """VisualCandidate has position_code field."""
        candidate = VisualCandidate(
            source="layout_aware",
            page_num=1,
            bbox_pts=(50, 100, 280, 400),
            page_width_pts=612.0,
            page_height_pts=792.0,
            area_ratio=0.2,
            position_code="L",
        )
        assert candidate.position_code == "L"

    def test_layout_filename_field(self):
        """VisualCandidate has layout_filename field."""
        candidate = VisualCandidate(
            source="layout_aware",
            page_num=3,
            bbox_pts=(50, 100, 280, 400),
            page_width_pts=612.0,
            page_height_pts=792.0,
            area_ratio=0.2,
            layout_filename="article_figure_p3_2col_L_1.png",
        )
        assert candidate.layout_filename == "article_figure_p3_2col_L_1.png"

    def test_all_layout_fields_together(self):
        """All layout fields work together."""
        candidate = VisualCandidate(
            source="layout_aware",
            page_num=3,
            bbox_pts=(50, 100, 280, 400),
            page_width_pts=612.0,
            page_height_pts=792.0,
            area_ratio=0.2,
            docling_type="figure",
            confidence=0.9,
            vlm_label="Figure 1",
            layout_code="2col",
            position_code="L",
            layout_filename="article_figure_p3_2col_L_1.png",
        )
        assert candidate.source == "layout_aware"
        assert candidate.layout_code == "2col"
        assert candidate.position_code == "L"
        assert candidate.layout_filename == "article_figure_p3_2col_L_1.png"


class TestExtractedVisualLayoutFields:
    """Tests for layout-aware fields in ExtractedVisual."""

    def test_layout_code_in_extracted(self):
        """ExtractedVisual has layout_code field."""
        visual = ExtractedVisual(
            visual_type=VisualType.FIGURE,
            confidence=0.9,
            page_range=[3],
            bbox_pts_per_page=[
                PageLocation(page_num=3, bbox_pts=(50, 100, 280, 400))
            ],
            image_base64="base64data",
            extraction_method="layout_aware",
            source_file="test.pdf",
            layout_code="2col",
        )
        assert visual.layout_code == "2col"

    def test_position_code_in_extracted(self):
        """ExtractedVisual has position_code field."""
        visual = ExtractedVisual(
            visual_type=VisualType.FIGURE,
            confidence=0.9,
            page_range=[3],
            bbox_pts_per_page=[
                PageLocation(page_num=3, bbox_pts=(50, 100, 280, 400))
            ],
            image_base64="base64data",
            extraction_method="layout_aware",
            source_file="test.pdf",
            position_code="L",
        )
        assert visual.position_code == "L"

    def test_layout_filename_in_extracted(self):
        """ExtractedVisual has layout_filename field."""
        visual = ExtractedVisual(
            visual_type=VisualType.FIGURE,
            confidence=0.9,
            page_range=[3],
            bbox_pts_per_page=[
                PageLocation(page_num=3, bbox_pts=(50, 100, 280, 400))
            ],
            image_base64="base64data",
            extraction_method="layout_aware",
            source_file="test.pdf",
            layout_filename="article_figure_p3_2col_L_1.png",
        )
        assert visual.layout_filename == "article_figure_p3_2col_L_1.png"

    def test_full_layout_aware_extraction(self):
        """Full ExtractedVisual with all layout fields."""
        visual = ExtractedVisual(
            visual_type=VisualType.FIGURE,
            confidence=0.9,
            page_range=[3],
            bbox_pts_per_page=[
                PageLocation(page_num=3, bbox_pts=(50, 100, 280, 400))
            ],
            caption_text="Figure 1. Study design showing...",
            image_base64="base64data",
            extraction_method="layout_aware",
            source_file="article.pdf",
            layout_code="2col",
            position_code="L",
            layout_filename="article_figure_p3_2col_L_1.png",
        )

        assert visual.visual_type == VisualType.FIGURE
        assert visual.extraction_method == "layout_aware"
        assert visual.layout_code == "2col"
        assert visual.position_code == "L"
        assert visual.layout_filename == "article_figure_p3_2col_L_1.png"

    def test_layout_fields_optional(self):
        """Layout fields are optional for non-layout-aware extraction."""
        visual = ExtractedVisual(
            visual_type=VisualType.TABLE,
            confidence=0.85,
            page_range=[1],
            bbox_pts_per_page=[
                PageLocation(page_num=1, bbox_pts=(50, 100, 550, 400))
            ],
            image_base64="base64data",
            extraction_method="docling_only",
            source_file="test.pdf",
        )

        assert visual.layout_code is None
        assert visual.position_code is None
        assert visual.layout_filename is None

