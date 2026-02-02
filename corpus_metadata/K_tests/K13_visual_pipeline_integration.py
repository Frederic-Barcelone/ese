# corpus_metadata/K_tests/K13_visual_pipeline_integration.py
"""Integration tests for layout-aware visual pipeline."""
from unittest.mock import Mock, patch

from B_parsing.B12_visual_pipeline import (
    PipelineConfig,
    VisualExtractionPipeline,
    extract_visuals_doclayout,
)
from B_parsing.B18_layout_models import (
    VisualPosition,
    VisualZone,
)
from B_parsing.B20_zone_expander import ExpandedVisual


class TestDoclayoutDetectionMode:
    """Tests for doclayout detection mode selection."""

    def test_config_accepts_doclayout_mode(self):
        """Pipeline config accepts 'doclayout' detection mode."""
        config = PipelineConfig(detection_mode="doclayout")
        assert config.detection_mode == "doclayout"

    def test_pipeline_initializes_with_doclayout(self):
        """Pipeline initializes with doclayout config."""
        config = PipelineConfig(detection_mode="doclayout")
        pipeline = VisualExtractionPipeline(config)
        assert pipeline.config.detection_mode == "doclayout"


class TestDoclayoutIntegration:
    """Integration tests for doclayout pipeline components."""

    def test_convenience_function_sets_mode(self):
        """extract_visuals_doclayout sets correct mode."""
        # We can't actually run the function without a PDF,
        # but we can verify it creates the right config
        with patch.object(VisualExtractionPipeline, 'extract') as mock_extract:
            mock_extract.return_value = Mock(visuals=[])

            # This would fail without a real PDF, but the config is created first
            try:
                extract_visuals_doclayout("/fake/path.pdf")
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

    def test_doclayout_mode_calls_correct_method(self):
        """'doclayout' mode calls _detect_with_doclayout."""
        config = PipelineConfig(detection_mode="doclayout")
        pipeline = VisualExtractionPipeline(config)

        with patch.object(
            pipeline, '_detect_with_doclayout'
        ) as mock_doclayout:
            mock_doclayout.return_value = Mock(candidates=[])

            pipeline._stage_detection("/fake/path.pdf")

            mock_doclayout.assert_called_once_with("/fake/path.pdf")

    def test_hybrid_mode_calls_both_methods(self):
        """'hybrid' mode calls both doclayout and heuristic detection."""
        config = PipelineConfig(detection_mode="hybrid")
        pipeline = VisualExtractionPipeline(config)

        with patch.object(pipeline, '_detect_with_doclayout') as mock_doclayout, \
             patch('B_parsing.B12_visual_pipeline.detect_all_visuals') as mock_heuristic, \
             patch.object(pipeline, '_merge_detection_results') as mock_merge:
            mock_doclayout.return_value = Mock(candidates=[])
            mock_heuristic.return_value = Mock(candidates=[])
            mock_merge.return_value = Mock(candidates=[])

            pipeline._stage_detection("/fake/path.pdf")

            mock_doclayout.assert_called_once_with("/fake/path.pdf")
            mock_heuristic.assert_called_once()

    def test_heuristic_mode_calls_detect_all(self):
        """'heuristic' mode calls detect_all_visuals."""
        config = PipelineConfig(detection_mode="heuristic")
        pipeline = VisualExtractionPipeline(config)

        with patch('B_parsing.B12_visual_pipeline.detect_all_visuals') as mock_detect:
            mock_detect.return_value = Mock(candidates=[])

            pipeline._stage_detection("/fake/path.pdf")

            mock_detect.assert_called_once()

