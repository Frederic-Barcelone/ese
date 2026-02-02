# corpus_metadata/tests/test_parsing/test_layout_e2e.py
"""End-to-end tests for layout-aware visual extraction."""
import pytest
from unittest.mock import Mock, patch

from B_parsing.B12_visual_pipeline import (
    PipelineConfig,
    VisualExtractionPipeline,
)
from B_parsing.B13_visual_detector import DetectionResult
from B_parsing.B18_layout_models import (
    LayoutPattern,
    PageLayout,
    VisualPosition,
    VisualZone,
)
from B_parsing.B20_zone_expander import ExpandedVisual


class TestLayoutAwareE2E:
    """End-to-end tests for layout-aware detection."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = '''
        {
            "layout": "2col",
            "column_boundary": 0.48,
            "visuals": [
                {
                    "type": "figure",
                    "label": "Figure 1",
                    "position": "left",
                    "vertical_zone": "top",
                    "confidence": 0.95,
                    "caption_start": "Figure 1. Study design..."
                },
                {
                    "type": "table",
                    "label": "Table 1",
                    "position": "right",
                    "vertical_zone": "middle",
                    "confidence": 0.9,
                    "caption_start": "Table 1. Patient demographics"
                }
            ]
        }
        '''
        mock_client.messages.create.return_value = mock_response
        return mock_client

    @pytest.fixture
    def mock_fitz_doc(self):
        """Create a mock PyMuPDF document."""
        mock_page = Mock()
        mock_page.rect.width = 612.0
        mock_page.rect.height = 792.0
        mock_page.get_pixmap.return_value = Mock(tobytes=lambda fmt: b"fake_image_data")
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": (50, 100, 280, 120),
                    "lines": [{"spans": [{"text": "Sample text"}]}],
                }
            ]
        }

        mock_doc = Mock()
        mock_doc.page_count = 1
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        return mock_doc

    def test_layout_aware_detection_creates_candidates(
        self, mock_anthropic_client, mock_fitz_doc
    ):
        """Layout-aware detection creates VisualCandidates with layout metadata."""
        config = PipelineConfig(detection_mode="layout-aware")
        pipeline = VisualExtractionPipeline(config)

        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            with patch('fitz.open', return_value=mock_fitz_doc):
                # Mock the analyze_page_layout function
                with patch(
                    'B_parsing.B12_visual_pipeline.analyze_page_layout'
                ) as mock_analyze:
                    # Create a realistic layout response
                    mock_layout = PageLayout(
                        page_num=1,
                        pattern=LayoutPattern.TWO_COL,
                        column_boundary=0.48,
                        visuals=[
                            VisualZone(
                                visual_type="figure",
                                label="Figure 1",
                                position=VisualPosition.LEFT,
                                vertical_zone="top",
                                confidence=0.95,
                            ),
                            VisualZone(
                                visual_type="table",
                                label="Table 1",
                                position=VisualPosition.RIGHT,
                                vertical_zone="middle",
                                confidence=0.9,
                            ),
                        ],
                    )
                    mock_analyze.return_value = mock_layout

                    result = pipeline._detect_with_layout_awareness("/fake/path.pdf")

        assert isinstance(result, DetectionResult)
        assert result.detection_mode == "layout-aware"
        assert len(result.candidates) == 2
        assert result.figures_detected == 1
        assert result.tables_detected == 1

        # Check first candidate (figure in left column)
        fig_candidate = result.candidates[0]
        assert fig_candidate.source == "layout_aware"
        assert fig_candidate.layout_code == "2col"
        assert fig_candidate.position_code == "L"
        assert "figure" in fig_candidate.layout_filename.lower()
        assert "2col" in fig_candidate.layout_filename
        assert "_L_" in fig_candidate.layout_filename

        # Check second candidate (table in right column)
        table_candidate = result.candidates[1]
        assert table_candidate.layout_code == "2col"
        assert table_candidate.position_code == "R"
        assert "table" in table_candidate.layout_filename.lower()
        assert "_R_" in table_candidate.layout_filename

    def test_layout_aware_fallback_on_error(self, mock_fitz_doc):
        """Layout-aware detection falls back to VLM-only on error."""
        config = PipelineConfig(detection_mode="layout-aware")
        pipeline = VisualExtractionPipeline(config)

        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.side_effect = Exception("API Error")

            with patch.object(
                pipeline, '_detect_with_vlm'
            ) as mock_vlm_fallback:
                mock_vlm_fallback.return_value = DetectionResult(
                    candidates=[],
                    tables_detected=0,
                    figures_detected=0,
                    escalated_tables=0,
                    detection_mode="vlm-only",
                )

                _result = pipeline._detect_with_layout_awareness("/fake/path.pdf")

        # Should have fallen back to VLM-only
        mock_vlm_fallback.assert_called_once()


class TestLayoutCodeFilenames:
    """Tests for layout-aware filename generation in E2E flow."""

    def test_filename_encodes_layout_pattern(self):
        """Generated filename encodes layout pattern."""
        from B_parsing.B21_filename_generator import generate_visual_filename

        zone = VisualZone(
            visual_type="figure",
            position=VisualPosition.LEFT,
            vertical_zone="top",
        )
        expanded = ExpandedVisual(
            zone=zone,
            bbox_pts=(50, 100, 280, 400),
            bbox_normalized=(0.08, 0.13, 0.46, 0.51),
            layout_code="2col-fullbot",
            position_code="L",
        )

        filename = generate_visual_filename(
            doc_name="clinical_trial.pdf",
            page_num=5,
            visual=expanded,
            index=1,
        )

        assert "2col-fullbot" in filename
        assert "_L_" in filename
        assert "p5" in filename
        assert filename.endswith(".png")

    def test_filename_supports_all_layout_patterns(self):
        """Filename generation works for all layout patterns."""
        from B_parsing.B21_filename_generator import generate_visual_filename

        patterns = [
            ("full", "F"),
            ("2col", "L"),
            ("2col", "R"),
            ("2col-fullbot", "F"),
            ("fulltop-2col", "L"),
        ]

        for layout_code, position_code in patterns:
            position = {
                "F": VisualPosition.FULL,
                "L": VisualPosition.LEFT,
                "R": VisualPosition.RIGHT,
            }[position_code]

            zone = VisualZone(
                visual_type="table",
                position=position,
                vertical_zone="middle",
            )
            expanded = ExpandedVisual(
                zone=zone,
                bbox_pts=(0, 0, 100, 100),
                bbox_normalized=(0, 0, 0.16, 0.13),
                layout_code=layout_code,
                position_code=position_code,
            )

            filename = generate_visual_filename("doc", 1, expanded, 1)

            assert layout_code in filename
            assert f"_{position_code}_" in filename


class TestMultipleVisualsPerPage:
    """Tests for handling multiple visuals on the same page."""

    def test_multiple_visuals_get_unique_filenames(self):
        """Multiple visuals on same page get unique filenames with index."""
        from B_parsing.B21_filename_generator import generate_visual_filename

        # Create two figures in left column
        zone1 = VisualZone(
            visual_type="figure",
            position=VisualPosition.LEFT,
            vertical_zone="top",
        )
        zone2 = VisualZone(
            visual_type="figure",
            position=VisualPosition.LEFT,
            vertical_zone="bottom",
        )

        expanded1 = ExpandedVisual(
            zone=zone1,
            bbox_pts=(0, 0, 100, 200),
            bbox_normalized=(0, 0, 0.16, 0.25),
            layout_code="2col",
            position_code="L",
        )
        expanded2 = ExpandedVisual(
            zone=zone2,
            bbox_pts=(0, 400, 100, 600),
            bbox_normalized=(0, 0.5, 0.16, 0.76),
            layout_code="2col",
            position_code="L",
        )

        filename1 = generate_visual_filename("doc", 3, expanded1, 1)
        filename2 = generate_visual_filename("doc", 3, expanded2, 2)

        assert filename1 != filename2
        assert "_1.png" in filename1
        assert "_2.png" in filename2
        # Both should have same layout info since same column
        assert "2col_L" in filename1
        assert "2col_L" in filename2


class TestZoneExpansionIntegration:
    """Tests for zone expansion in E2E flow."""

    def test_zone_expander_creates_valid_bboxes(self):
        """Zone expander creates valid bounding boxes."""
        from B_parsing.B20_zone_expander import expand_all_zones

        zone = VisualZone(
            visual_type="figure",
            label="Figure 1",
            position=VisualPosition.LEFT,
            vertical_zone="top",
            confidence=0.9,
        )
        layout = PageLayout(
            page_num=1,
            pattern=LayoutPattern.TWO_COL,
            column_boundary=0.5,
            visuals=[zone],
        )

        results = expand_all_zones(
            layout=layout,
            page_width=612.0,
            page_height=792.0,
            text_blocks=[],
        )

        assert len(results) == 1
        expanded = results[0]

        # Verify bbox is valid
        x0, y0, x1, y1 = expanded.bbox_pts
        assert x0 >= 0
        assert y0 >= 0
        assert x1 > x0
        assert y1 > y0
        assert x1 <= 612.0  # Page width
        assert y1 <= 792.0  # Page height

        # Verify layout codes
        assert expanded.layout_code == "2col"
        assert expanded.position_code == "L"

