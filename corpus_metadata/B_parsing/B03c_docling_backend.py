# corpus_metadata/B_parsing/B03c_docling_backend.py
"""
Docling-based table extraction backend.

Uses IBM's Docling library with TableFormer model for high-accuracy
table structure recognition (95-98% TEDS score on complex tables).

This module provides an alternative to Unstructured's table extraction,
offering superior accuracy especially for:
- Complex nested tables
- Merged cells (colspan/rowspan)
- Tables with partial or no borders
- Multi-row headers

References:
- Docling: https://github.com/docling-project/docling
- TableFormer paper: https://arxiv.org/abs/2203.01017
- Docling technical report: https://arxiv.org/abs/2408.09869
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Check if Docling is available
DOCLING_AVAILABLE = False
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling_core.types.doc import TableItem

    DOCLING_AVAILABLE = True
except ImportError:
    logger.warning(
        "Docling not installed. Install with: pip install docling"
    )
    DocumentConverter = None  # type: ignore
    PdfFormatOption = None  # type: ignore
    InputFormat = None  # type: ignore
    PdfPipelineOptions = None  # type: ignore
    TableFormerMode = None  # type: ignore
    TableItem = None  # type: ignore


class DoclingTableExtractor:
    """
    Extract tables from PDFs using Docling's TableFormer model.

    TableFormer achieves state-of-the-art performance:
    - 98.5% TEDS on simple tables
    - 95% TEDS on complex tables

    Attributes:
        mode: TableFormer mode ('accurate' or 'fast')
        do_cell_matching: Whether to match cells to PDF coordinates
        converter: Docling DocumentConverter instance
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Docling table extractor.

        Args:
            config: Configuration dict with optional keys:
                - mode: 'accurate' (default) or 'fast'
                - do_cell_matching: bool (default False for better column separation)
                - ocr_enabled: bool (default True)
        """
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "Docling is not installed. Install with: pip install docling"
            )

        self.config = config or {}
        self.mode = self.config.get("mode", "accurate")
        self.do_cell_matching = self.config.get("do_cell_matching", True)
        self.ocr_enabled = self.config.get("ocr_enabled", True)

        # Initialize converter with optimized settings
        self.converter = self._create_converter()

    def _create_converter(self) -> "DocumentConverter":
        """Create and configure the Docling DocumentConverter."""
        # Configure pipeline options for optimal table extraction
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            do_ocr=self.ocr_enabled,
        )

        # Set TableFormer mode
        if self.mode == "accurate":
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        else:
            pipeline_options.table_structure_options.mode = TableFormerMode.FAST

        # Disable cell matching for better column separation
        # This uses TableFormer's predicted cells instead of PDF coordinates
        pipeline_options.table_structure_options.do_cell_matching = self.do_cell_matching

        # Create converter with PDF-specific options
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        return converter

    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of table dicts, each containing:
                - page_num: int
                - bbox: (x0, y0, x1, y1) in PDF points
                - headers: List[str]
                - rows: List[List[str]]
                - confidence: float (0-1)
                - extraction_method: 'docling'
        """
        file_path = str(Path(file_path).resolve())
        logger.info("Extracting tables with Docling from: %s", file_path)

        try:
            # Convert document
            result = self.converter.convert(file_path)
            document = result.document

            tables = []
            for table_idx, table in enumerate(document.tables):
                table_data = self._convert_table(table, document, table_idx)
                if table_data:
                    tables.append(table_data)

            logger.info("Docling extracted %d tables from %s", len(tables), file_path)
            return tables

        except Exception as e:
            logger.error("Docling table extraction failed: %s", e)
            return []

    def _convert_table(
        self,
        table: "TableItem",
        document: Any,
        table_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a Docling TableItem to our standard table dict format.

        Args:
            table: Docling TableItem object
            document: DoclingDocument for DataFrame export
            table_idx: Index of the table in the document

        Returns:
            Table dict or None if conversion fails
        """
        try:
            # Get provenance info (page number, bbox)
            page_num = 1
            bbox = (0.0, 0.0, 0.0, 0.0)

            if table.prov and len(table.prov) > 0:
                prov = table.prov[0]
                page_num = prov.page_no if hasattr(prov, "page_no") else 1

                if hasattr(prov, "bbox") and prov.bbox:
                    # Docling bbox has l, t, r, b properties
                    bbox = (
                        float(prov.bbox.l),
                        float(prov.bbox.t),
                        float(prov.bbox.r),
                        float(prov.bbox.b),
                    )

            # Export to DataFrame for structured data
            try:
                df = table.export_to_dataframe(doc=document)

                # Extract headers and rows from DataFrame
                headers = [str(col) for col in df.columns.tolist()]
                rows = []
                for _, row in df.iterrows():
                    rows.append([str(cell) if cell is not None else "" for cell in row.tolist()])

            except Exception as df_err:
                logger.warning(
                    "DataFrame export failed for table %d: %s, using fallback",
                    table_idx,
                    df_err,
                )
                # Fallback: try to extract from table data directly
                headers, rows = self._extract_from_table_data(table)

            # Skip empty tables
            if not headers and not rows:
                logger.debug("Skipping empty table %d on page %d", table_idx, page_num)
                return None

            # Get confidence from conversion result if available
            confidence = 0.95  # Default high confidence for Docling

            return {
                "page_num": page_num,
                "bbox": bbox,
                "headers": headers,
                "rows": rows,
                "table_type": self._classify_table(headers),
                "confidence": confidence,
                "extraction_method": "docling",
                "html": None,  # Docling doesn't use HTML intermediate
            }

        except Exception as e:
            logger.warning("Failed to convert table %d: %s", table_idx, e)
            return None

    def _extract_from_table_data(
        self, table: "TableItem"
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Fallback extraction from TableItem data structure.

        Args:
            table: Docling TableItem

        Returns:
            Tuple of (headers, rows)
        """
        headers = []
        rows = []

        try:
            # Try to access table.data if available
            if hasattr(table, "data") and table.data:
                data = table.data
                if hasattr(data, "table_cells") and data.table_cells:
                    # Build grid from cells
                    grid = self._build_grid_from_cells(data.table_cells)
                    if grid:
                        headers = grid[0] if grid else []
                        rows = grid[1:] if len(grid) > 1 else []
        except Exception as e:
            logger.debug("Fallback extraction failed: %s", e)

        return headers, rows

    def _build_grid_from_cells(self, cells: List[Any]) -> List[List[str]]:
        """
        Build a 2D grid from table cells.

        Args:
            cells: List of cell objects with row_span, col_span, text

        Returns:
            2D list of cell values
        """
        if not cells:
            return []

        # Find grid dimensions
        max_row = 0
        max_col = 0
        for cell in cells:
            row = getattr(cell, "row", 0) or 0
            col = getattr(cell, "col", 0) or 0
            row_span = getattr(cell, "row_span", 1) or 1
            col_span = getattr(cell, "col_span", 1) or 1
            max_row = max(max_row, row + row_span)
            max_col = max(max_col, col + col_span)

        if max_row == 0 or max_col == 0:
            return []

        # Initialize grid
        grid = [["" for _ in range(max_col)] for _ in range(max_row)]

        # Fill grid
        for cell in cells:
            row = getattr(cell, "row", 0) or 0
            col = getattr(cell, "col", 0) or 0
            text = getattr(cell, "text", "") or ""
            row_span = getattr(cell, "row_span", 1) or 1
            col_span = getattr(cell, "col_span", 1) or 1

            # Fill all cells covered by this span
            for r in range(row, min(row + row_span, max_row)):
                for c in range(col, min(col + col_span, max_col)):
                    grid[r][c] = str(text)

        return grid

    def _classify_table(self, headers: List[str]) -> str:
        """
        Classify table type based on headers.

        Args:
            headers: List of header strings

        Returns:
            Table type string
        """
        if not headers:
            return "UNKNOWN"

        header_text = " ".join(headers).lower()

        # Check for glossary/abbreviation table
        glossary_patterns = ["abbrev", "acronym", "definition", "term", "meaning"]
        if any(p in header_text for p in glossary_patterns):
            return "GLOSSARY"

        return "DATA_GRID"


def extract_tables_docling(
    file_path: str, config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function: Extract tables from PDF using Docling.

    Args:
        file_path: Path to PDF file
        config: Optional configuration dict

    Returns:
        List of table dicts
    """
    extractor = DoclingTableExtractor(config)
    return extractor.extract_tables(file_path)


__all__ = ["DoclingTableExtractor", "extract_tables_docling", "DOCLING_AVAILABLE"]
