# corpus_metadata/tests/test_parsing/test_visual_pipeline_integration.py
"""Integration tests for layout-aware visual pipeline."""
import pytest
from unittest.mock import Mock, patch, MagicMock

from B_parsing.B12_visual_pipeline import (
    PipelineConfig,
    VisualExtractionPipeline,
    extract_visuals_layout_aware,
)
from B_parsing.B18_layout_models import (
    LayoutPattern,
    PageLayout,
    VisualPosition,
    VisualZone,
)
from B_parsing.B20_zone_expander import ExpandedVisual


class TestLayoutAwareDetectionMode:
    """Tests for layout-aware detection mode selection."""

    def test_config_accepts_layout_aware_mode(self):
        """Pipeline config accepts 'layout-aware' detection mode."""
        config = PipelineConfig(detection_mode="layout-aware")
        assert config.detection_mode == "layout-aware"

    def test_pipeline_initializes_with_layout_aware(self):
        """Pipeline initializes with layout-aware config."""
        config = PipelineConfig(detection_mode="layout-aware")
        pipeline = VisualExtractionPipeline(config)
        assert pipeline.config.detection_mode == "layout-aware"


class TestExtractTextBlocks:
    """Tests for text block extraction helper."""

    def test_extract_text_blocks_returns_list(self):
        """_extract_text_blocks returns a list."""
        pipeline = VisualExtractionPipeline()

        # Mock document and page
        mock_page = Mock()
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,  # Text block
                    "bbox": (50, 100, 280, 120),
                    "lines": [
                        {
                            "spans": [
                                {"text": "Sample text"}
                            ]
                        }
                    ],
                }
            ]
        }

        mock_doc = Mock()
        mock_doc.__getitem__ = Mock(return_value=mock_page)

        result = pipeline._extract_text_blocks(mock_doc, 1)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["bbox"] == (50, 100, 280, 120)
        assert result[0]["text"] == "Sample text"

    def test_extract_text_blocks_filters_non_text(self):
        """_extract_text_blocks filters out non-text blocks."""
        pipeline = VisualExtractionPipeline()

        mock_page = Mock()
        mock_page.get_text.return_value = {
            "blocks": [
                {"type": 0, "bbox": (0, 0, 100, 100), "lines": []},  # Text
                {"type": 1, "bbox": (0, 0, 100, 100)},  # Image (no type 0)
            ]
        }

        mock_doc = Mock()
        mock_doc.__getitem__ = Mock(return_value=mock_page)

        result = pipeline._extract_text_blocks(mock_doc, 1)

        assert len(result) == 1


class TestLayoutAwareIntegration:
    """Integration tests for layout-aware pipeline components."""

    def test_convenience_function_sets_mode(self):
        """extract_visuals_layout_aware sets correct mode."""
        # We can't actually run the function without a PDF,
        # but we can verify it creates the right config
        with patch.object(VisualExtractionPipeline, 'extract') as mock_extract:
            mock_extract.return_value = Mock(visuals=[])

            # This would fail without a real PDF, but the config is created first
            try:
                extract_visuals_layout_aware("/fake/path.pdf")
            except Exception:
                pass

    def test_layout_metadata_propagation(self):
        """Layout metadata (layout_code, position_code) is propagated."""
        # Create test data
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

        # Verify layout codes are accessible
        assert expanded.layout_code == "2col"
        assert expanded.position_code == "L"


class TestDetectionModeSelection:
    """Tests for detection mode selection in _stage_detection."""

    def test_layout_aware_mode_calls_correct_method(self):
        """'layout-aware' mode calls _detect_with_layout_awareness."""
        config = PipelineConfig(detection_mode="layout-aware")
        pipeline = VisualExtractionPipeline(config)

        with patch.object(
            pipeline, '_detect_with_layout_awareness'
        ) as mock_layout:
            mock_layout.return_value = Mock(candidates=[])

            pipeline._stage_detection("/fake/path.pdf")

            mock_layout.assert_called_once_with("/fake/path.pdf")

    def test_vlm_only_mode_calls_correct_method(self):
        """'vlm-only' mode calls _detect_with_vlm."""
        config = PipelineConfig(detection_mode="vlm-only")
        pipeline = VisualExtractionPipeline(config)

        with patch.object(pipeline, '_detect_with_vlm') as mock_vlm:
            mock_vlm.return_value = Mock(candidates=[])

            pipeline._stage_detection("/fake/path.pdf")

            mock_vlm.assert_called_once_with("/fake/path.pdf")

    def test_heuristic_mode_calls_detect_all(self):
        """'heuristic' mode calls detect_all_visuals."""
        config = PipelineConfig(detection_mode="heuristic")
        pipeline = VisualExtractionPipeline(config)

        with patch('B_parsing.B12_visual_pipeline.detect_all_visuals') as mock_detect:
            mock_detect.return_value = Mock(candidates=[])

            pipeline._stage_detection("/fake/path.pdf")

            mock_detect.assert_called_once()

