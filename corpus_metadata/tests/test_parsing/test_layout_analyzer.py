# corpus_metadata/tests/test_parsing/test_layout_analyzer.py
"""Tests for VLM layout analyzer."""
import pytest
from unittest.mock import Mock, patch
from B_parsing.B18_layout_models import LayoutPattern, VisualPosition
from B_parsing.B19_layout_analyzer import (
    parse_layout_response,
    VLM_LAYOUT_PROMPT,
)


class TestParseLayoutResponse:
    """Tests for parsing VLM layout response."""

    def test_parse_two_column_with_figure(self):
        """Parse 2-column layout with figure in left column."""
        raw_response = '''
        {
            "layout": "2col",
            "column_boundary": 0.48,
            "visuals": [
                {
                    "type": "figure",
                    "label": "Figure 1",
                    "position": "left",
                    "vertical_zone": "top",
                    "confidence": 0.95
                }
            ]
        }
        '''
        result = parse_layout_response(raw_response, page_num=3)

        assert result.pattern == LayoutPattern.TWO_COL
        assert result.column_boundary == 0.48
        assert len(result.visuals) == 1
        assert result.visuals[0].position == VisualPosition.LEFT

    def test_parse_full_page_layout(self):
        """Parse full page layout."""
        raw_response = '''
        {
            "layout": "full",
            "visuals": [
                {
                    "type": "table",
                    "label": "Table 1",
                    "position": "full",
                    "vertical_zone": "0.3-0.8",
                    "confidence": 0.9
                }
            ]
        }
        '''
        result = parse_layout_response(raw_response, page_num=5)

        assert result.pattern == LayoutPattern.FULL
        assert result.visuals[0].position == VisualPosition.FULL

    def test_parse_multiple_visuals(self):
        """Parse page with multiple visuals."""
        raw_response = '''
        {
            "layout": "2col",
            "column_boundary": 0.5,
            "visuals": [
                {"type": "figure", "label": "Figure 1", "position": "left", "vertical_zone": "top", "confidence": 0.9},
                {"type": "figure", "label": "Figure 2", "position": "left", "vertical_zone": "bottom", "confidence": 0.85},
                {"type": "table", "label": "Table 1", "position": "right", "vertical_zone": "middle", "confidence": 0.95}
            ]
        }
        '''
        result = parse_layout_response(raw_response, page_num=3)

        assert len(result.visuals) == 3
        assert result.visuals[0].label == "Figure 1"
        assert result.visuals[1].label == "Figure 2"
        assert result.visuals[2].label == "Table 1"

    def test_parse_hybrid_layout_fullbot(self):
        """Parse 2col-fullbot hybrid layout."""
        raw_response = '''
        {
            "layout": "2col-fullbot",
            "column_boundary": 0.5,
            "visuals": [
                {"type": "table", "label": "Table 1", "position": "full", "vertical_zone": "bottom", "confidence": 0.9}
            ]
        }
        '''
        result = parse_layout_response(raw_response, page_num=4)

        assert result.pattern == LayoutPattern.TWO_COL_FULLBOT
        assert result.visuals[0].position == VisualPosition.FULL

    def test_parse_hybrid_layout_fulltop(self):
        """Parse fulltop-2col hybrid layout."""
        raw_response = '''
        {
            "layout": "fulltop-2col",
            "column_boundary": 0.5,
            "visuals": [
                {"type": "figure", "label": "Figure 1", "position": "full", "vertical_zone": "top", "confidence": 0.92}
            ]
        }
        '''
        result = parse_layout_response(raw_response, page_num=2)

        assert result.pattern == LayoutPattern.FULLTOP_TWO_COL

    def test_parse_no_visuals(self):
        """Parse page with no visuals."""
        raw_response = '''
        {
            "layout": "2col",
            "column_boundary": 0.5,
            "visuals": []
        }
        '''
        result = parse_layout_response(raw_response, page_num=1)

        assert result.pattern == LayoutPattern.TWO_COL
        assert len(result.visuals) == 0

    def test_parse_with_markdown_code_block(self):
        """Parse response wrapped in markdown code block."""
        raw_response = '''
        ```json
        {
            "layout": "full",
            "visuals": [
                {"type": "figure", "label": "Figure 1", "position": "full", "vertical_zone": "middle", "confidence": 0.9}
            ]
        }
        ```
        '''
        result = parse_layout_response(raw_response, page_num=1)

        assert result.pattern == LayoutPattern.FULL
        assert len(result.visuals) == 1

    def test_parse_invalid_json_returns_default(self):
        """Invalid JSON returns default full layout."""
        raw_response = "This is not valid JSON"
        result = parse_layout_response(raw_response, page_num=1)

        assert result.pattern == LayoutPattern.FULL
        assert len(result.visuals) == 0

    def test_parse_unknown_layout_defaults_to_full(self):
        """Unknown layout code defaults to FULL."""
        raw_response = '''
        {
            "layout": "unknown_pattern",
            "visuals": []
        }
        '''
        result = parse_layout_response(raw_response, page_num=1)

        assert result.pattern == LayoutPattern.FULL

    def test_parse_with_caption_snippet(self):
        """Parse visual with caption snippet."""
        raw_response = '''
        {
            "layout": "full",
            "visuals": [
                {
                    "type": "figure",
                    "label": "Figure 1",
                    "position": "full",
                    "vertical_zone": "top",
                    "confidence": 0.9,
                    "caption_start": "Patient flow diagram showing..."
                }
            ]
        }
        '''
        result = parse_layout_response(raw_response, page_num=1)

        assert result.visuals[0].caption_snippet == "Patient flow diagram showing..."

    def test_parse_continuation_flags(self):
        """Parse visual with continuation flags."""
        raw_response = '''
        {
            "layout": "full",
            "visuals": [
                {
                    "type": "table",
                    "label": "Table 1",
                    "position": "full",
                    "vertical_zone": "0.0-0.9",
                    "confidence": 0.9,
                    "is_continuation": true,
                    "continues_next": false
                }
            ]
        }
        '''
        result = parse_layout_response(raw_response, page_num=2)

        assert result.visuals[0].is_continuation is True
        assert result.visuals[0].continues_next is False


class TestVLMLayoutPrompt:
    """Tests for VLM prompt structure."""

    def test_prompt_contains_layout_patterns(self):
        """Prompt mentions all layout patterns."""
        assert "full" in VLM_LAYOUT_PROMPT
        assert "2col" in VLM_LAYOUT_PROMPT
        assert "2col-fullbot" in VLM_LAYOUT_PROMPT
        assert "fulltop-2col" in VLM_LAYOUT_PROMPT

    def test_prompt_contains_position_codes(self):
        """Prompt mentions position codes."""
        assert "left" in VLM_LAYOUT_PROMPT.lower()
        assert "right" in VLM_LAYOUT_PROMPT.lower()
        assert "full" in VLM_LAYOUT_PROMPT.lower()

    def test_prompt_mentions_no_bbox(self):
        """Prompt explicitly says no bounding boxes."""
        assert "bounding box" in VLM_LAYOUT_PROMPT.lower() or "bbox" in VLM_LAYOUT_PROMPT.lower()

    def test_prompt_mentions_vertical_zone(self):
        """Prompt mentions vertical zone options."""
        assert "top" in VLM_LAYOUT_PROMPT.lower()
        assert "middle" in VLM_LAYOUT_PROMPT.lower()
        assert "bottom" in VLM_LAYOUT_PROMPT.lower()
