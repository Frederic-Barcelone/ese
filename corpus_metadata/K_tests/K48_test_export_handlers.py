# corpus_metadata/K_tests/K48_test_export_handlers.py
"""
Tests for J_export.J01_export_handlers module.

Tests ExportManager and export functionality.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from J_export.J01_export_handlers import ExportManager


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def export_manager(temp_output_dir):
    """Create ExportManager for testing."""
    return ExportManager(
        run_id="TEST_001",
        pipeline_version="1.0.0",
        output_dir=temp_output_dir,
    )


class TestExportManagerInit:
    """Tests for ExportManager initialization."""

    def test_init_with_minimal_args(self):
        manager = ExportManager(
            run_id="TEST",
            pipeline_version="1.0",
        )
        assert manager.run_id == "TEST"
        assert manager.pipeline_version == "1.0"
        assert manager.output_dir_override is None
        assert manager.gold_json is None
        assert manager.claude_client is None

    def test_init_with_output_dir(self, temp_output_dir):
        manager = ExportManager(
            run_id="TEST",
            pipeline_version="1.0",
            output_dir=temp_output_dir,
        )
        assert manager.output_dir_override == temp_output_dir

    def test_init_with_gold_json(self):
        manager = ExportManager(
            run_id="TEST",
            pipeline_version="1.0",
            gold_json="/path/to/gold.json",
        )
        assert manager.gold_json == "/path/to/gold.json"


class TestExportManagerGetOutputDir:
    """Tests for get_output_dir method."""

    def test_uses_override_when_set(self, temp_output_dir):
        manager = ExportManager(
            run_id="TEST",
            pipeline_version="1.0",
            output_dir=temp_output_dir,
        )

        pdf_path = Path("/some/path/document.pdf")
        out_dir = manager.get_output_dir(pdf_path)

        assert out_dir == temp_output_dir

    def test_creates_pdf_named_folder_without_override(self):
        manager = ExportManager(
            run_id="TEST",
            pipeline_version="1.0",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test_document.pdf"
            pdf_path.touch()

            out_dir = manager.get_output_dir(pdf_path)

            assert out_dir.name == "test_document"
            assert out_dir.parent == Path(tmpdir)

    def test_creates_directory_if_not_exists(self, temp_output_dir):
        manager = ExportManager(
            run_id="TEST",
            pipeline_version="1.0",
            output_dir=temp_output_dir / "nested" / "path",
        )

        pdf_path = Path("/some/document.pdf")
        out_dir = manager.get_output_dir(pdf_path)

        assert out_dir.exists()


class TestExportManagerRenderFigure:
    """Tests for render_figure_with_padding method."""

    def test_returns_none_without_pymupdf(self, export_manager):
        """Should return None if fitz not available or file doesn't exist."""
        result = export_manager.render_figure_with_padding(
            pdf_path=Path("/nonexistent.pdf"),
            page_num=1,
            bbox=(0, 0, 100, 100),
        )
        # fitz (PyMuPDF) not available in test env + file doesn't exist
        assert result is None


class TestExportManagerAttributes:
    """Tests for ExportManager attributes."""

    def test_has_run_id(self, export_manager):
        assert hasattr(export_manager, "run_id")
        assert export_manager.run_id == "TEST_001"

    def test_has_pipeline_version(self, export_manager):
        assert hasattr(export_manager, "pipeline_version")
        assert export_manager.pipeline_version == "1.0.0"
