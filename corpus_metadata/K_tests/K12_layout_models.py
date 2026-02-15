# corpus_metadata/K_tests/K12_layout_models.py
"""Tests for layout models used in layout-aware visual extraction."""



class TestLayoutPatternEnum:
    """Tests for LayoutPattern enum."""

    def test_full_pattern_value(self):
        """Test FULL pattern has correct value."""
        from A_core.A24_layout_models import LayoutPattern

        assert LayoutPattern.FULL.value == "full"

    def test_two_col_pattern_value(self):
        """Test TWO_COL pattern has correct value."""
        from A_core.A24_layout_models import LayoutPattern

        assert LayoutPattern.TWO_COL.value == "2col"

    def test_two_col_fullbot_pattern_value(self):
        """Test TWO_COL_FULLBOT pattern has correct value."""
        from A_core.A24_layout_models import LayoutPattern

        assert LayoutPattern.TWO_COL_FULLBOT.value == "2col-fullbot"

    def test_fulltop_two_col_pattern_value(self):
        """Test FULLTOP_TWO_COL pattern has correct value."""
        from A_core.A24_layout_models import LayoutPattern

        assert LayoutPattern.FULLTOP_TWO_COL.value == "fulltop-2col"

    def test_all_patterns_are_strings(self):
        """Test all patterns inherit from str."""
        from A_core.A24_layout_models import LayoutPattern

        for pattern in LayoutPattern:
            assert isinstance(pattern.value, str)
            assert isinstance(pattern, str)


class TestVisualPositionEnum:
    """Tests for VisualPosition enum."""

    def test_left_position_value(self):
        """Test LEFT position has correct code."""
        from A_core.A24_layout_models import VisualPosition

        assert VisualPosition.LEFT.value == "L"

    def test_right_position_value(self):
        """Test RIGHT position has correct code."""
        from A_core.A24_layout_models import VisualPosition

        assert VisualPosition.RIGHT.value == "R"

    def test_full_position_value(self):
        """Test FULL position has correct code."""
        from A_core.A24_layout_models import VisualPosition

        assert VisualPosition.FULL.value == "F"

    def test_positions_are_single_char(self):
        """Test all positions are single character codes."""
        from A_core.A24_layout_models import VisualPosition

        for pos in VisualPosition:
            assert len(pos.value) == 1


class TestVisualZoneDataclass:
    """Tests for VisualZone dataclass."""

    def test_create_minimal_zone(self):
        """Test creating a zone with required fields only."""
        from A_core.A24_layout_models import VisualPosition, VisualZone

        zone = VisualZone(
            visual_type="table",
            position=VisualPosition.LEFT,
            vertical_zone="top",
        )

        assert zone.visual_type == "table"
        assert zone.position == VisualPosition.LEFT
        assert zone.vertical_zone == "top"
        # Check defaults
        assert zone.label is None
        assert zone.confidence == 0.9
        assert zone.caption_snippet is None
        assert zone.is_continuation is False
        assert zone.continues_next is False

    def test_create_full_zone(self):
        """Test creating a zone with all fields specified."""
        from A_core.A24_layout_models import VisualPosition, VisualZone

        zone = VisualZone(
            visual_type="figure",
            label="Figure 1",
            position=VisualPosition.FULL,
            vertical_zone="0.2-0.6",
            confidence=0.85,
            caption_snippet="Patient flow diagram...",
            is_continuation=True,
            continues_next=False,
        )

        assert zone.visual_type == "figure"
        assert zone.label == "Figure 1"
        assert zone.position == VisualPosition.FULL
        assert zone.vertical_zone == "0.2-0.6"
        assert zone.confidence == 0.85
        assert zone.caption_snippet == "Patient flow diagram..."
        assert zone.is_continuation is True
        assert zone.continues_next is False

    def test_zone_accepts_table_type(self):
        """Test zone accepts 'table' as visual_type."""
        from A_core.A24_layout_models import VisualPosition, VisualZone

        zone = VisualZone(
            visual_type="table",
            position=VisualPosition.RIGHT,
            vertical_zone="middle",
        )
        assert zone.visual_type == "table"

    def test_zone_accepts_figure_type(self):
        """Test zone accepts 'figure' as visual_type."""
        from A_core.A24_layout_models import VisualPosition, VisualZone

        zone = VisualZone(
            visual_type="figure",
            position=VisualPosition.FULL,
            vertical_zone="bottom",
        )
        assert zone.visual_type == "figure"

    def test_zone_vertical_zone_named_regions(self):
        """Test zone accepts named vertical regions."""
        from A_core.A24_layout_models import VisualPosition, VisualZone

        for region in ["top", "middle", "bottom"]:
            zone = VisualZone(
                visual_type="table",
                position=VisualPosition.LEFT,
                vertical_zone=region,
            )
            assert zone.vertical_zone == region

    def test_zone_vertical_zone_numeric_format(self):
        """Test zone accepts numeric range format for vertical_zone."""
        from A_core.A24_layout_models import VisualPosition, VisualZone

        zone = VisualZone(
            visual_type="table",
            position=VisualPosition.LEFT,
            vertical_zone="0.1-0.45",
        )
        assert zone.vertical_zone == "0.1-0.45"


class TestPageLayoutDataclass:
    """Tests for PageLayout dataclass."""

    def test_create_minimal_page_layout(self):
        """Test creating page layout with required fields only."""
        from A_core.A24_layout_models import LayoutPattern, PageLayout

        layout = PageLayout(
            page_num=1,
            pattern=LayoutPattern.FULL,
        )

        assert layout.page_num == 1
        assert layout.pattern == LayoutPattern.FULL
        # Check defaults
        assert layout.column_boundary is None
        assert layout.margin_left == 0.05
        assert layout.margin_right == 0.95
        assert layout.visuals == []
        assert layout.raw_vlm_response is None

    def test_create_two_column_layout(self):
        """Test creating a two-column page layout."""
        from A_core.A24_layout_models import LayoutPattern, PageLayout

        layout = PageLayout(
            page_num=3,
            pattern=LayoutPattern.TWO_COL,
            column_boundary=0.5,
        )

        assert layout.page_num == 3
        assert layout.pattern == LayoutPattern.TWO_COL
        assert layout.column_boundary == 0.5

    def test_create_layout_with_visuals(self):
        """Test creating page layout with visual zones."""
        from A_core.A24_layout_models import (
            LayoutPattern,
            PageLayout,
            VisualPosition,
            VisualZone,
        )

        table_zone = VisualZone(
            visual_type="table",
            label="Table 1",
            position=VisualPosition.LEFT,
            vertical_zone="top",
        )
        figure_zone = VisualZone(
            visual_type="figure",
            label="Figure 2",
            position=VisualPosition.RIGHT,
            vertical_zone="bottom",
        )

        layout = PageLayout(
            page_num=5,
            pattern=LayoutPattern.TWO_COL,
            column_boundary=0.48,
            visuals=[table_zone, figure_zone],
        )

        assert len(layout.visuals) == 2
        assert layout.visuals[0].label == "Table 1"
        assert layout.visuals[1].label == "Figure 2"

    def test_create_layout_with_custom_margins(self):
        """Test creating page layout with custom margins."""
        from A_core.A24_layout_models import LayoutPattern, PageLayout

        layout = PageLayout(
            page_num=1,
            pattern=LayoutPattern.FULL,
            margin_left=0.08,
            margin_right=0.92,
        )

        assert layout.margin_left == 0.08
        assert layout.margin_right == 0.92

    def test_create_layout_with_raw_vlm_response(self):
        """Test creating page layout with raw VLM response stored."""
        from A_core.A24_layout_models import LayoutPattern, PageLayout

        raw_response = '{"pattern": "full", "visuals": []}'
        layout = PageLayout(
            page_num=1,
            pattern=LayoutPattern.FULL,
            raw_vlm_response=raw_response,
        )

        assert layout.raw_vlm_response == raw_response

    def test_visuals_list_is_mutable(self):
        """Test that visuals list can be modified after creation."""
        from A_core.A24_layout_models import (
            LayoutPattern,
            PageLayout,
            VisualPosition,
            VisualZone,
        )

        layout = PageLayout(
            page_num=1,
            pattern=LayoutPattern.FULL,
        )

        assert layout.visuals == []

        zone = VisualZone(
            visual_type="table",
            position=VisualPosition.FULL,
            vertical_zone="middle",
        )
        layout.visuals.append(zone)

        assert len(layout.visuals) == 1

    def test_default_visuals_list_not_shared(self):
        """Test that default empty visuals list is not shared between instances."""
        from A_core.A24_layout_models import (
            LayoutPattern,
            PageLayout,
            VisualPosition,
            VisualZone,
        )

        layout1 = PageLayout(page_num=1, pattern=LayoutPattern.FULL)
        layout2 = PageLayout(page_num=2, pattern=LayoutPattern.FULL)

        zone = VisualZone(
            visual_type="table",
            position=VisualPosition.FULL,
            vertical_zone="top",
        )
        layout1.visuals.append(zone)

        # layout2 should not be affected
        assert len(layout1.visuals) == 1
        assert len(layout2.visuals) == 0


class TestLayoutModelsIntegration:
    """Integration tests for layout models working together."""

    def test_full_page_layout_scenario(self):
        """Test a realistic page layout with multiple visuals."""
        from A_core.A24_layout_models import (
            LayoutPattern,
            PageLayout,
            VisualPosition,
            VisualZone,
        )

        # Simulate a 2-column academic paper page with table and figure
        visuals = [
            VisualZone(
                visual_type="table",
                label="Table 1",
                position=VisualPosition.LEFT,
                vertical_zone="0.15-0.45",
                confidence=0.92,
                caption_snippet="Baseline characteristics...",
            ),
            VisualZone(
                visual_type="figure",
                label="Figure 1",
                position=VisualPosition.RIGHT,
                vertical_zone="0.55-0.85",
                confidence=0.88,
                caption_snippet="Patient disposition...",
            ),
        ]

        layout = PageLayout(
            page_num=3,
            pattern=LayoutPattern.TWO_COL,
            column_boundary=0.5,
            margin_left=0.05,
            margin_right=0.95,
            visuals=visuals,
            raw_vlm_response='{"pattern": "2col", ...}',
        )

        assert layout.pattern == LayoutPattern.TWO_COL
        assert len(layout.visuals) == 2
        assert all(v.confidence > 0.8 for v in layout.visuals)

    def test_continuation_table_scenario(self):
        """Test a table that spans multiple pages."""
        from A_core.A24_layout_models import (
            LayoutPattern,
            PageLayout,
            VisualPosition,
            VisualZone,
        )

        # Page 2: table starts
        page2_zone = VisualZone(
            visual_type="table",
            label="Table 2",
            position=VisualPosition.FULL,
            vertical_zone="0.3-0.95",
            is_continuation=False,
            continues_next=True,
        )
        page2 = PageLayout(
            page_num=2,
            pattern=LayoutPattern.FULL,
            visuals=[page2_zone],
        )

        # Page 3: table continues
        page3_zone = VisualZone(
            visual_type="table",
            label="Table 2 (continued)",
            position=VisualPosition.FULL,
            vertical_zone="0.05-0.6",
            is_continuation=True,
            continues_next=False,
        )
        page3 = PageLayout(
            page_num=3,
            pattern=LayoutPattern.FULL,
            visuals=[page3_zone],
        )

        assert page2.visuals[0].continues_next is True
        assert page3.visuals[0].is_continuation is True
