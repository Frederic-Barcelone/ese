# corpus_metadata/tests/test_parsing/test_docling_backend.py
"""Tests for Docling table extraction backend."""

from unittest import mock

import pytest


class TestDoclingBackendImport:
    """Tests for Docling backend import handling."""

    def test_import_when_docling_not_installed(self):
        """Test graceful handling when Docling is not installed."""
        from B_parsing.B28_docling_backend import DOCLING_AVAILABLE

        # DOCLING_AVAILABLE should be False if docling not installed
        # (or True if it is - either is valid)
        assert isinstance(DOCLING_AVAILABLE, bool)

    def test_docling_extractor_raises_when_not_available(self):
        """Test that DoclingTableExtractor raises ImportError when Docling not installed."""
        from B_parsing.B28_docling_backend import (
            DoclingTableExtractor,
            DOCLING_AVAILABLE,
        )

        if not DOCLING_AVAILABLE:
            with pytest.raises(ImportError, match="Docling is not installed"):
                DoclingTableExtractor()


class TestDoclingTableExtractorMocked:
    """Tests for DoclingTableExtractor with mocked Docling."""

    def test_table_classification_glossary(self):
        """Test table classification for glossary tables."""
        from B_parsing.B28_docling_backend import DoclingTableExtractor

        # Create a minimal mock extractor to test classification
        with mock.patch.object(
            DoclingTableExtractor, "__init__", lambda self, config=None: None
        ):
            extractor = DoclingTableExtractor()
            extractor.config = {}

            # Test glossary detection
            assert extractor._classify_table(["Abbreviation", "Definition"]) == "GLOSSARY"
            assert extractor._classify_table(["Acronym", "Meaning"]) == "GLOSSARY"
            assert extractor._classify_table(["Term", "Description"]) == "GLOSSARY"

            # Test non-glossary
            assert extractor._classify_table(["Name", "Value", "Unit"]) == "DATA_GRID"
            assert extractor._classify_table([]) == "UNKNOWN"

    def test_build_grid_from_cells(self):
        """Test building 2D grid from cell objects."""
        from B_parsing.B28_docling_backend import DoclingTableExtractor

        with mock.patch.object(
            DoclingTableExtractor, "__init__", lambda self, config=None: None
        ):
            extractor = DoclingTableExtractor()

            # Create mock cells
            class MockCell:
                def __init__(self, row, col, text, row_span=1, col_span=1):
                    self.row = row
                    self.col = col
                    self.text = text
                    self.row_span = row_span
                    self.col_span = col_span

            cells = [
                MockCell(0, 0, "Header1"),
                MockCell(0, 1, "Header2"),
                MockCell(1, 0, "Data1"),
                MockCell(1, 1, "Data2"),
            ]

            grid = extractor._build_grid_from_cells(cells)

            assert len(grid) == 2
            assert grid[0] == ["Header1", "Header2"]
            assert grid[1] == ["Data1", "Data2"]

    def test_build_grid_with_spans(self):
        """Test grid building with colspan/rowspan."""
        from B_parsing.B28_docling_backend import DoclingTableExtractor

        with mock.patch.object(
            DoclingTableExtractor, "__init__", lambda self, config=None: None
        ):
            extractor = DoclingTableExtractor()

            class MockCell:
                def __init__(self, row, col, text, row_span=1, col_span=1):
                    self.row = row
                    self.col = col
                    self.text = text
                    self.row_span = row_span
                    self.col_span = col_span

            # Header spans 2 columns
            cells = [
                MockCell(0, 0, "Wide Header", col_span=2),
                MockCell(1, 0, "A"),
                MockCell(1, 1, "B"),
            ]

            grid = extractor._build_grid_from_cells(cells)

            assert len(grid) == 2
            assert grid[0] == ["Wide Header", "Wide Header"]
            assert grid[1] == ["A", "B"]

    def test_empty_cells(self):
        """Test grid building with empty cells."""
        from B_parsing.B28_docling_backend import DoclingTableExtractor

        with mock.patch.object(
            DoclingTableExtractor, "__init__", lambda self, config=None: None
        ):
            extractor = DoclingTableExtractor()

            # Empty list should return empty grid
            grid = extractor._build_grid_from_cells([])
            assert grid == []


class TestTableExtractor:
    """Tests for TableExtractor class."""

    def test_table_extractor_requires_docling(self):
        """Test that TableExtractor requires Docling."""
        from B_parsing.B28_docling_backend import DOCLING_AVAILABLE

        if not DOCLING_AVAILABLE:
            from B_parsing.B03_table_extractor import TableExtractor

            with pytest.raises(ImportError):
                TableExtractor()

    def test_default_config_values(self):
        """Test default configuration values."""
        from B_parsing.B28_docling_backend import DOCLING_AVAILABLE

        if DOCLING_AVAILABLE:
            from B_parsing.B03_table_extractor import TableExtractor

            extractor = TableExtractor()
            # Check that defaults are applied
            assert extractor._extractor.mode == "accurate"
            assert extractor._extractor.do_cell_matching is True
            assert extractor._extractor.ocr_enabled is True


class TestConvenienceFunction:
    """Tests for convenience functions."""

    def test_extract_tables_docling_function(self):
        """Test the convenience function exists and handles missing Docling."""
        from B_parsing.B28_docling_backend import (
            extract_tables_docling,
            DOCLING_AVAILABLE,
        )

        if not DOCLING_AVAILABLE:
            with pytest.raises(ImportError):
                extract_tables_docling("/nonexistent/file.pdf")
