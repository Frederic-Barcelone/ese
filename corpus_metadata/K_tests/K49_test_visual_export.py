# corpus_metadata/K_tests/K49_test_visual_export.py
"""
Tests for J_export.J02_visual_export module.

Tests visual export functions and serialization.
"""

from __future__ import annotations

import base64
import json
import tempfile
from datetime import datetime
from pathlib import Path
import pytest

from J_export.J02_visual_export import (
    visual_to_dict,
    pipeline_result_to_dict,
    export_visuals_to_json,
    export_tables_only,
    export_figures_only,
)
from A_core.A13_visual_models import (
    ExtractedVisual,
    VisualType,
    PageLocation,
)


@pytest.fixture
def sample_figure():
    """Create a sample figure visual."""
    return ExtractedVisual(
        visual_id="fig_001",
        visual_type=VisualType.FIGURE,
        confidence=0.95,
        page_range=[1],
        bbox_pts_per_page=[
            PageLocation(page_num=1, bbox_pts=(100, 200, 400, 500)),
        ],
        image_base64=base64.b64encode(b"fake_image_data").decode(),
        caption_text="Figure 1: Test figure",
        extraction_method="native",
        source_file="test.pdf",
        extracted_at=datetime.now(),
    )


@pytest.fixture
def sample_table():
    """Create a sample table visual."""
    return ExtractedVisual(
        visual_id="tbl_001",
        visual_type=VisualType.TABLE,
        confidence=0.90,
        page_range=[2],
        bbox_pts_per_page=[
            PageLocation(page_num=2, bbox_pts=(50, 100, 500, 400)),
        ],
        image_base64=base64.b64encode(b"fake_table_image").decode(),
        caption_text="Table 1: Test table",
        extraction_method="docling",
        source_file="test.pdf",
        extracted_at=datetime.now(),
    )


@pytest.fixture
def sample_pipeline_result(sample_figure, sample_table):
    """Create a mock PipelineResult."""
    from B_parsing.B12_visual_pipeline import PipelineResult

    result = PipelineResult(
        visuals=[sample_figure, sample_table],
        source_file="test.pdf",
        extracted_at=datetime.now(),
        tables_detected=1,
        figures_detected=1,
        tables_escalated=0,
        vlm_enriched=0,
        merges_performed=0,
        duplicates_removed=0,
        extraction_time_seconds=1.5,
    )
    return result


class TestVisualToDict:
    """Tests for visual_to_dict function."""

    def test_figure_serialization(self, sample_figure):
        result = visual_to_dict(sample_figure)

        assert result["visual_id"] == "fig_001"
        assert result["visual_type"] == "figure"
        assert result["confidence"] == 0.95
        assert result["caption"] == "Figure 1: Test figure"

    def test_table_serialization(self, sample_table):
        result = visual_to_dict(sample_table)

        assert result["visual_id"] == "tbl_001"
        assert result["visual_type"] == "table"
        assert "table_data" in result

    def test_includes_locations(self, sample_figure):
        result = visual_to_dict(sample_figure)

        assert "locations" in result
        assert len(result["locations"]) == 1
        assert result["locations"][0]["page_num"] == 1

    def test_includes_provenance(self, sample_figure):
        result = visual_to_dict(sample_figure)

        assert "provenance" in result
        assert result["provenance"]["extraction_method"] == "native"
        assert result["provenance"]["source_file"] == "test.pdf"

    def test_includes_image_base64_when_enabled(self, sample_figure):
        result = visual_to_dict(sample_figure, include_image=True)
        assert "image_base64" in result

    def test_excludes_image_when_file_provided(self, sample_figure):
        result = visual_to_dict(sample_figure, include_image=True, image_file="test.png")
        assert result["image_file"] == "test.png"
        assert "image_base64" not in result

    def test_table_data_included(self, sample_table):
        result = visual_to_dict(sample_table)
        assert result["visual_type"] == "table"


class TestPipelineResultToDict:
    """Tests for pipeline_result_to_dict function."""

    def test_serializes_metadata(self, sample_pipeline_result):
        result = pipeline_result_to_dict(sample_pipeline_result)

        assert "metadata" in result
        assert result["metadata"]["source_file"] == "test.pdf"
        assert "extracted_at" in result["metadata"]

    def test_serializes_statistics(self, sample_pipeline_result):
        result = pipeline_result_to_dict(sample_pipeline_result)

        assert "statistics" in result
        assert result["statistics"]["final_count"] == 2

    def test_serializes_visuals(self, sample_pipeline_result):
        result = pipeline_result_to_dict(sample_pipeline_result)

        assert "visuals" in result
        assert len(result["visuals"]) == 2


class TestExportVisualsToJson:
    """Tests for export_visuals_to_json function."""

    def test_exports_to_file(self, sample_pipeline_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "visuals.json"

            result_path = export_visuals_to_json(
                sample_pipeline_result,
                output_path,
            )

            assert result_path.exists()
            with open(result_path) as f:
                data = json.load(f)

            assert "visuals" in data
            assert len(data["visuals"]) == 2

    def test_creates_parent_directories(self, sample_pipeline_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "path" / "visuals.json"

            result_path = export_visuals_to_json(
                sample_pipeline_result,
                output_path,
            )

            assert result_path.exists()

    def test_pretty_print_option(self, sample_pipeline_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "visuals.json"

            export_visuals_to_json(
                sample_pipeline_result,
                output_path,
                pretty_print=True,
            )

            with open(output_path) as f:
                content = f.read()

            # Pretty printed JSON has newlines
            assert "\n" in content


class TestExportTablesOnly:
    """Tests for export_tables_only function."""

    def test_exports_only_tables(self, sample_pipeline_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tables.json"

            result_path = export_tables_only(
                sample_pipeline_result,
                output_path,
                save_images=False,
            )

            assert result_path.exists()
            with open(result_path) as f:
                data = json.load(f)

            # Should only have the table
            assert data["count"] == 1
            assert data["tables"][0]["visual_type"] == "table"


class TestExportFiguresOnly:
    """Tests for export_figures_only function."""

    def test_exports_only_figures(self, sample_pipeline_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "figures.json"

            result_path = export_figures_only(
                sample_pipeline_result,
                output_path,
                save_images=False,
            )

            assert result_path.exists()
            with open(result_path) as f:
                data = json.load(f)

            # Should only have the figure
            assert data["count"] == 1
            assert data["figures"][0]["visual_type"] == "figure"
