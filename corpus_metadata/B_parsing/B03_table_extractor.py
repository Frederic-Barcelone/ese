# corpus_metadata/B_parsing/B03_table_extractor.py
"""
Table extraction from PDFs using Docling's TableFormer model.

This module provides high-accuracy table extraction (95-98% TEDS score) using
IBM's Docling library with the TableFormer model. It handles complex tables
with merged cells, multi-row headers, and borderless layouts. Tables are
rendered as images for VLM analysis and converted to structured logical rows.

Key Components:
    - TableExtractor: Main table extraction class with Docling backend
    - extract_tables_to_json: Convenience function for PDF to JSON table extraction
    - populate_document_graph: Integrate extracted tables into DocumentGraph

Example:
    >>> from B_parsing.B03_table_extractor import TableExtractor
    >>> extractor = TableExtractor(config={"mode": "accurate"})
    >>> tables = extractor.extract_tables("document.pdf")
    >>> for table in tables:
    ...     print(f"Page {table['page_num']}: {len(table['rows'])} rows")

Dependencies:
    - A_core.A01_domain_models: BoundingBox for table coordinates
    - B_parsing.B02_doc_graph: DocumentGraph, Table, TableCell, TableType
    - B_parsing.B14_visual_renderer: Table image rendering functions
    - B_parsing.B27_table_validation: MIN_TABLE_COLS validation constant
    - B_parsing.B28_docling_backend: DoclingTableExtractor backend
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from A_core.A01_domain_models import BoundingBox
from A_core.A23_doc_graph_models import DocumentGraph, Table, TableCell, TableType
from B_parsing.B28_docling_backend import DoclingTableExtractor

# Rendering module (B14) - visual rendering functions
from B_parsing.B14_visual_renderer import (
    render_table_as_image,
    render_full_page_from_path as render_full_page,
    render_multipage_table,
)

# Validation constants
from B_parsing.B27_table_validation import MIN_TABLE_COLS

# Minimum rows required for VLM extraction to be considered valid
MIN_VLM_DATA_ROWS = 2


class TableExtractor:
    """
    Extracts tables from PDF using Docling's TableFormer model.

    TableFormer achieves 95-98% TEDS score on complex tables.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

        # Initialize Docling backend
        docling_config = {
            "mode": self.config.get("mode", "accurate"),
            "do_cell_matching": self.config.get("do_cell_matching", True),
            "ocr_enabled": self.config.get("ocr_enabled", True),
        }
        self._extractor = DoclingTableExtractor(docling_config)
        logger.info("Using Docling TableFormer for table extraction")

    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF as list of dicts.

        Returns:
            List of table dicts with keys: page_num, bbox, headers, rows,
            table_type, confidence, extraction_method
        """
        return self._extractor.extract_tables(file_path)

    def populate_document_graph(
        self,
        doc: DocumentGraph,
        file_path: str,
        render_images: bool = True,
        use_vlm: bool = False,
        vlm_extractor: Optional[Any] = None,
    ) -> DocumentGraph:
        """
        Extract tables from PDF and add them to DocumentGraph.

        Args:
            doc: DocumentGraph to populate
            file_path: Path to PDF file
            render_images: Whether to render table images
            use_vlm: Whether to use VLM for additional extraction (optional)
            vlm_extractor: VLMTableExtractor instance
        """
        tables_data = self.extract_tables_with_images(
            file_path,
            render_images=render_images,
            use_vlm=use_vlm,
            vlm_extractor=vlm_extractor,
        )

        for t in tables_data:
            page_num = t["page_num"]
            if page_num not in doc.pages:
                continue

            page = doc.pages[page_num]
            bbox_coords = t["bbox"]

            # Build logical rows
            headers = t.get("headers", [])
            logical_rows = []
            for row in t.get("rows", []):
                row_dict = {}
                for i, cell in enumerate(row):
                    key = headers[i] if i < len(headers) else f"col_{i}"
                    row_dict[key] = cell
                logical_rows.append(row_dict)

            # Build cells
            cells = []
            all_rows = [headers] + t.get("rows", [])
            for row_idx, row in enumerate(all_rows):
                for col_idx, cell_text in enumerate(row):
                    cells.append(
                        TableCell(
                            text=cell_text,
                            row_index=row_idx,
                            col_index=col_idx,
                            row_span=1,
                            col_span=1,
                            is_header=(row_idx == 0),
                            bbox=BoundingBox(coords=bbox_coords),
                        )
                    )

            # Build metadata
            headers_map = {i: h for i, h in enumerate(headers)}
            glossary_cols = self._detect_glossary_cols(headers)

            table_metadata = {
                "headers": headers_map,
                "ordered_cols": list(range(len(headers))),
                "glossary_cols": glossary_cols,
                "extraction_method": t.get("extraction_method", "docling"),
            }

            if t.get("vlm_confidence"):
                table_metadata["vlm_confidence"] = t["vlm_confidence"]
            if t.get("vlm_warning"):
                table_metadata["vlm_warning"] = t["vlm_warning"]
            if t.get("notes"):
                table_metadata["notes"] = t["notes"]

            table = Table(
                page_num=page_num,
                reading_order_index=len(page.tables),
                table_type=TableType(t.get("table_type", "UNKNOWN")),
                cells=cells,
                logical_rows=logical_rows,
                bbox=BoundingBox(coords=bbox_coords),
                image_base64=t.get("image_base64"),
                image_format="png",
                page_nums=t.get("page_nums", [page_num]),
                is_multipage=t.get("is_multipage", False),
                extraction_method=t.get("extraction_method", "docling"),
                metadata=table_metadata,
            )

            page.tables.append(table)

        return doc

    def _detect_glossary_cols(self, headers: List[str]) -> Dict[str, int]:
        """Detect which columns are abbreviation/definition columns."""
        result = {}
        for i, h in enumerate(headers):
            h_lower = h.lower()
            if any(p in h_lower for p in ["abbr", "acronym", "symbol"]):
                result["sf_col_idx"] = i
            elif any(p in h_lower for p in ["definition", "meaning", "term", "description"]):
                result["lf_col_idx"] = i
        return result

    def extract_tables_with_images(
        self,
        file_path: str,
        render_images: bool = True,
        dpi: int = 300,
        use_vlm: bool = False,
        vlm_extractor: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract tables with rendered images.

        Args:
            file_path: Path to PDF
            render_images: Whether to render table images
            dpi: Resolution for image rendering
            use_vlm: Whether to use VLM for additional extraction
            vlm_extractor: VLMTableExtractor instance

        Returns:
            List of table dicts with image_base64 field
        """
        tables_data = self.extract_tables(file_path)

        if not tables_data:
            return []

        # Detect multi-page tables
        table_groups = self._detect_multipage_tables(tables_data)

        result = []
        for group in table_groups:
            if len(group) == 1:
                table = group[0]
                table["page_nums"] = [table["page_num"]]
                table["is_multipage"] = False

                if render_images:
                    if table["bbox"] != (0.0, 0.0, 0.0, 0.0):
                        table["image_base64"] = render_table_as_image(
                            file_path, table["page_num"], table["bbox"], dpi=dpi
                        )
                    else:
                        table["image_base64"] = render_full_page(
                            file_path, table["page_num"], dpi=dpi
                        )
                else:
                    table["image_base64"] = None

                # Optional VLM enhancement
                if use_vlm and vlm_extractor and table.get("image_base64"):
                    self._apply_vlm_extraction(table, vlm_extractor)

                result.append(table)
            else:
                # Multi-page table
                merged = self._merge_table_parts(group)

                if render_images:
                    parts = [
                        {
                            "page_num": t["page_num"],
                            "bbox": t["bbox"],
                            "full_page": t["bbox"] == (0.0, 0.0, 0.0, 0.0),
                        }
                        for t in group
                    ]
                    merged["image_base64"] = render_multipage_table(file_path, parts, dpi=dpi)
                else:
                    merged["image_base64"] = None

                if use_vlm and vlm_extractor and merged.get("image_base64"):
                    self._apply_vlm_extraction(merged, vlm_extractor)

                result.append(merged)

        return result

    def _apply_vlm_extraction(self, table: Dict[str, Any], vlm_extractor: Any) -> None:
        """Apply VLM extraction to enhance table data."""
        try:
            vlm_result = vlm_extractor.extract(table["image_base64"])
            if vlm_result and vlm_result.get("confidence", 0) > 0.3:
                vlm_rows = vlm_result.get("rows", [])
                vlm_headers = vlm_result.get("headers", [])

                if len(vlm_rows) < MIN_VLM_DATA_ROWS or len(vlm_headers) < MIN_TABLE_COLS:
                    logger.info(
                        "Rejecting VLM table on page %d: too small",
                        table.get("page_num", 0),
                    )
                    return

                table["headers"] = vlm_headers
                table["rows"] = vlm_rows
                table["extraction_method"] = "vlm"
                table["vlm_confidence"] = vlm_result.get("confidence", 0.95)
                if vlm_result.get("verification_warning"):
                    table["vlm_warning"] = vlm_result["verification_warning"]
                if vlm_result.get("notes"):
                    table["notes"] = vlm_result["notes"]

        except Exception as e:
            logger.warning("VLM extraction error: %s", e)

    def _detect_multipage_tables(
        self, tables_data: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Detect tables that span multiple pages."""
        if not tables_data:
            return []

        sorted_tables = sorted(
            tables_data,
            key=lambda t: (t["page_num"], t["bbox"][1] if t["bbox"] else 0),
        )

        groups: List[List[Dict[str, Any]]] = []
        current_group: List[Dict[str, Any]] = []

        for table in sorted_tables:
            if not current_group:
                current_group = [table]
                continue

            prev_table = current_group[-1]

            if self._is_table_continuation(prev_table, table):
                current_group.append(table)
            else:
                groups.append(current_group)
                current_group = [table]

        if current_group:
            groups.append(current_group)

        return groups

    def _is_table_continuation(
        self, prev_table: Dict[str, Any], curr_table: Dict[str, Any]
    ) -> bool:
        """Check if curr_table is a continuation of prev_table."""
        if curr_table["page_num"] != prev_table["page_num"] + 1:
            return False

        prev_headers = [h.lower().strip() for h in prev_table.get("headers", [])]
        curr_headers = [h.lower().strip() for h in curr_table.get("headers", [])]

        if prev_headers and curr_headers and prev_headers == curr_headers:
            return True

        # Check for "continued" markers
        if curr_table.get("rows"):
            first_row_text = " ".join(str(c) for c in curr_table["rows"][0])
            if any(m in first_row_text.lower() for m in ["continued", "cont."]):
                return True

        return False

    def _merge_table_parts(self, parts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple table parts into a single table dict."""
        if not parts:
            return {}

        first = parts[0]
        merged = {
            "page_num": first["page_num"],
            "page_nums": [p["page_num"] for p in parts],
            "is_multipage": True,
            "bbox": first["bbox"],
            "headers": first.get("headers", []),
            "table_type": first.get("table_type", "UNKNOWN"),
            "extraction_method": "docling",
            "confidence": first.get("confidence", 0.95),
        }

        all_rows = []
        first_headers = [h.lower().strip() for h in first.get("headers", [])]

        for i, part in enumerate(parts):
            rows = part.get("rows", [])
            if i > 0 and rows:
                first_row = [str(c).lower().strip() for c in rows[0]]
                if first_row == first_headers:
                    rows = rows[1:]
            all_rows.extend(rows)

        merged["rows"] = all_rows
        return merged


def extract_tables_to_json(
    file_path: str, config: Optional[dict] = None
) -> List[Dict[str, Any]]:
    """Convenience function: PDF -> JSON tables."""
    extractor = TableExtractor(config)
    return extractor.extract_tables(file_path)
