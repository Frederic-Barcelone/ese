# corpus_metadata/K_tests/K09_filename_generator.py
"""Tests for layout-aware filename generation."""
from A_core.A24_layout_models import VisualPosition, VisualZone
from B_parsing.B20_zone_expander import ExpandedVisual
from B_parsing.B21_filename_generator import generate_visual_filename, sanitize_name


class TestSanitizeName:
    """Tests for document name sanitization."""

    def test_simple_name_unchanged(self):
        """Simple name passes through."""
        assert sanitize_name("article") == "article"

    def test_removes_pdf_extension(self):
        """Removes .pdf extension."""
        assert sanitize_name("document.pdf") == "document"
        assert sanitize_name("document.PDF") == "document"

    def test_replaces_spaces(self):
        """Replaces spaces with underscores."""
        assert sanitize_name("my document") == "my_document"

    def test_replaces_special_chars(self):
        """Replaces special characters with underscores."""
        result = sanitize_name("doc (2024) - final")
        assert "(" not in result
        assert ")" not in result
        assert "-" in result or "_" in result  # Dash is allowed

    def test_collapses_multiple_underscores(self):
        """Collapses multiple underscores to one."""
        result = sanitize_name("doc___name")
        assert "___" not in result

    def test_strips_leading_trailing_underscores(self):
        """Strips leading/trailing underscores."""
        result = sanitize_name("_document_")
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_limits_length(self):
        """Limits name to 50 characters."""
        long_name = "a" * 100
        result = sanitize_name(long_name)
        assert len(result) <= 50

    def test_empty_returns_default(self):
        """Empty or all-special returns 'document'."""
        assert sanitize_name("") == "document"
        assert sanitize_name("...") == "document"


class TestFilenameGeneration:
    """Tests for visual filename generation."""

    def test_two_column_left_figure(self):
        """Generate filename for figure in left column of 2-col layout."""
        zone = VisualZone(
            visual_type="figure",
            label="Figure 1",
            position=VisualPosition.LEFT,
            vertical_zone="top",
            confidence=0.9,
        )
        expanded = ExpandedVisual(
            zone=zone,
            bbox_pts=(50, 100, 280, 400),
            bbox_normalized=(0.08, 0.13, 0.46, 0.51),
            layout_code="2col",
            position_code="L",
        )

        filename = generate_visual_filename(
            doc_name="article",
            page_num=3,
            visual=expanded,
            index=1,
        )

        assert filename == "article_figure_p3_2col_L_1.png"

    def test_two_column_right_table(self):
        """Generate filename for table in right column."""
        zone = VisualZone(
            visual_type="table",
            label="Table 1",
            position=VisualPosition.RIGHT,
            vertical_zone="middle",
            confidence=0.9,
        )
        expanded = ExpandedVisual(
            zone=zone,
            bbox_pts=(320, 200, 550, 500),
            bbox_normalized=(0.52, 0.25, 0.90, 0.63),
            layout_code="2col",
            position_code="R",
        )

        filename = generate_visual_filename(
            doc_name="paper",
            page_num=5,
            visual=expanded,
            index=1,
        )

        assert filename == "paper_table_p5_2col_R_1.png"

    def test_hybrid_layout_full_width_table(self):
        """Generate filename for full-width table in hybrid layout."""
        zone = VisualZone(
            visual_type="table",
            label="Table 1",
            position=VisualPosition.FULL,
            vertical_zone="bottom",
            confidence=0.9,
        )
        expanded = ExpandedVisual(
            zone=zone,
            bbox_pts=(30, 500, 580, 750),
            bbox_normalized=(0.05, 0.63, 0.95, 0.95),
            layout_code="2col-fullbot",
            position_code="F",
        )

        filename = generate_visual_filename(
            doc_name="paper",
            page_num=5,
            visual=expanded,
            index=1,
        )

        assert filename == "paper_table_p5_2col-fullbot_F_1.png"

    def test_full_page_layout(self):
        """Generate filename for full-page layout."""
        zone = VisualZone(
            visual_type="figure",
            label="Figure 1",
            position=VisualPosition.FULL,
            vertical_zone="middle",
            confidence=0.9,
        )
        expanded = ExpandedVisual(
            zone=zone,
            bbox_pts=(30, 200, 580, 600),
            bbox_normalized=(0.05, 0.25, 0.95, 0.76),
            layout_code="full",
            position_code="F",
        )

        filename = generate_visual_filename(
            doc_name="report",
            page_num=7,
            visual=expanded,
            index=1,
        )

        assert filename == "report_figure_p7_full_F_1.png"

    def test_multiple_visuals_same_column(self):
        """Generate unique filenames for multiple visuals in same column."""
        zone1 = VisualZone(
            visual_type="figure",
            label="Figure 1",
            position=VisualPosition.LEFT,
            vertical_zone="top",
            confidence=0.9,
        )
        zone2 = VisualZone(
            visual_type="figure",
            label="Figure 2",
            position=VisualPosition.LEFT,
            vertical_zone="bottom",
            confidence=0.9,
        )

        expanded1 = ExpandedVisual(
            zone=zone1,
            bbox_pts=(0, 0, 0, 0),
            bbox_normalized=(0, 0, 0, 0),
            layout_code="2col",
            position_code="L",
        )
        expanded2 = ExpandedVisual(
            zone=zone2,
            bbox_pts=(0, 0, 0, 0),
            bbox_normalized=(0, 0, 0, 0),
            layout_code="2col",
            position_code="L",
        )

        filename1 = generate_visual_filename("doc", 3, expanded1, 1)
        filename2 = generate_visual_filename("doc", 3, expanded2, 2)

        assert filename1 == "doc_figure_p3_2col_L_1.png"
        assert filename2 == "doc_figure_p3_2col_L_2.png"

    def test_sanitize_document_name(self):
        """Sanitize document name with special characters."""
        zone = VisualZone(
            visual_type="figure",
            label="Figure 1",
            position=VisualPosition.FULL,
            vertical_zone="top",
            confidence=0.9,
        )
        expanded = ExpandedVisual(
            zone=zone,
            bbox_pts=(0, 0, 0, 0),
            bbox_normalized=(0, 0, 0, 0),
            layout_code="full",
            position_code="F",
        )

        filename = generate_visual_filename(
            doc_name="My Article (2024) - Final.pdf",
            page_num=1,
            visual=expanded,
            index=1,
        )

        # Should sanitize to safe filename
        assert " " not in filename
        assert "(" not in filename
        assert filename.endswith(".png")

    def test_custom_extension(self):
        """Support custom file extension."""
        zone = VisualZone(
            visual_type="figure",
            position=VisualPosition.FULL,
            vertical_zone="top",
        )
        expanded = ExpandedVisual(
            zone=zone,
            bbox_pts=(0, 0, 0, 0),
            bbox_normalized=(0, 0, 0, 0),
            layout_code="full",
            position_code="F",
        )

        filename = generate_visual_filename("doc", 1, expanded, 1, extension="jpg")

        assert filename.endswith(".jpg")

    def test_fulltop_2col_layout(self):
        """Generate filename for fulltop-2col layout."""
        zone = VisualZone(
            visual_type="figure",
            position=VisualPosition.FULL,
            vertical_zone="top",
        )
        expanded = ExpandedVisual(
            zone=zone,
            bbox_pts=(0, 0, 0, 0),
            bbox_normalized=(0, 0, 0, 0),
            layout_code="fulltop-2col",
            position_code="F",
        )

        filename = generate_visual_filename("doc", 2, expanded, 1)

        assert filename == "doc_figure_p2_fulltop-2col_F_1.png"
