# corpus_metadata/corpus_metadata/B_parsing/B03_table_extractor.py
"""
Table extraction from PDF -> JSON/Table model.

Uses unstructured's table extraction (hi_res with infer_table_structure=True)
to populate Table objects in the DocumentGraph.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from unstructured.partition.pdf import partition_pdf

from A_core.A01_domain_models import BoundingBox
from B_parsing.B02_doc_graph import DocumentGraph, Table, TableCell, TableType


# Patterns to detect glossary/abbreviation tables
GLOSSARY_HEADER_PATTERNS = [
    r"abbrev",
    r"acronym",
    r"definition",
    r"term",
    r"meaning",
    r"description",
]
GLOSSARY_HEADER_RE = re.compile("|".join(GLOSSARY_HEADER_PATTERNS), re.IGNORECASE)


class TableExtractor:
    """
    Extracts tables from PDF and populates DocumentGraph.tables.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.strategy = self.config.get("strategy", "hi_res")
        self.hi_res_model_name = self.config.get("hi_res_model_name", "yolox")
        self.languages = self.config.get("languages", ["eng"])

    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF as list of dicts (JSON-friendly).
        """
        elements = partition_pdf(
            filename=file_path,
            strategy=self.strategy,
            hi_res_model_name=self.hi_res_model_name,
            infer_table_structure=True,
            languages=self.languages,
        )

        tables = []
        for el in elements:
            if el.category != "Table":
                continue

            table_data = self._element_to_dict(el)
            if table_data:
                tables.append(table_data)

        return tables

    def _element_to_dict(self, el) -> Optional[Dict[str, Any]]:
        """Convert unstructured table element to dict."""
        md = getattr(el, "metadata", None)
        if not md:
            return None

        # Get HTML table if available
        html = getattr(md, "text_as_html", None)
        page_num = getattr(md, "page_number", None) or 1

        # Get bounding box
        bbox = self._get_bbox(md)

        # Parse HTML to rows
        rows = self._parse_html_table(html) if html else []
        headers = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []

        # Classify table type
        table_type = self._classify_table(headers)

        return {
            "page_num": page_num,
            "bbox": bbox,
            "headers": headers,
            "rows": data_rows,
            "table_type": table_type,
            "html": html,
        }

    def _get_bbox(self, md) -> Tuple[float, float, float, float]:
        """Extract bounding box from metadata."""
        coords = getattr(md, "coordinates", None)
        if not coords:
            return (0.0, 0.0, 0.0, 0.0)

        points = getattr(coords, "points", None)
        if not points:
            return (0.0, 0.0, 0.0, 0.0)

        try:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            return (min(xs), min(ys), max(xs), max(ys))
        except (TypeError, IndexError):
            return (0.0, 0.0, 0.0, 0.0)

    def _parse_html_table(self, html: str) -> List[List[str]]:
        """Parse HTML table to list of rows."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        rows = []

        for tr in soup.find_all("tr"):
            cells = []
            for td in tr.find_all(["th", "td"]):
                cells.append(td.get_text(strip=True))
            if cells:
                rows.append(cells)

        return rows

    def _classify_table(self, headers: List[str]) -> str:
        """Classify table type based on headers."""
        if not headers:
            return TableType.UNKNOWN.value

        header_text = " ".join(headers).lower()
        if GLOSSARY_HEADER_RE.search(header_text):
            return TableType.GLOSSARY.value

        return TableType.DATA_GRID.value

    def populate_document_graph(
        self, doc: DocumentGraph, file_path: str
    ) -> DocumentGraph:
        """
        Extract tables from PDF and add them to DocumentGraph.
        """
        tables_data = self.extract_tables(file_path)

        for idx, t in enumerate(tables_data):
            page_num = t["page_num"]
            if page_num not in doc.pages:
                continue

            page = doc.pages[page_num]
            bbox_coords = t["bbox"]

            # Build logical rows
            headers = t["headers"]
            logical_rows = []
            for row in t["rows"]:
                row_dict = {}
                for i, cell in enumerate(row):
                    key = headers[i] if i < len(headers) else f"col_{i}"
                    row_dict[key] = cell
                logical_rows.append(row_dict)

            # Build cells
            cells = []
            all_rows = [headers] + t["rows"]
            for row_idx, row in enumerate(all_rows):
                for col_idx, cell_text in enumerate(row):
                    cells.append(
                        TableCell(
                            text=cell_text,
                            row_index=row_idx,
                            col_index=col_idx,
                            is_header=(row_idx == 0),
                            bbox=BoundingBox(coords=bbox_coords),
                        )
                    )

            # Build metadata
            headers_map = {i: h for i, h in enumerate(headers)}
            glossary_cols = self._detect_glossary_cols(headers)

            table = Table(
                page_num=page_num,
                reading_order_index=len(page.tables),
                table_type=TableType(t["table_type"]),
                cells=cells,
                logical_rows=logical_rows,
                bbox=BoundingBox(coords=bbox_coords),
                metadata={
                    "headers": headers_map,
                    "ordered_cols": list(range(len(headers))),
                    "glossary_cols": glossary_cols,
                },
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
            elif any(
                p in h_lower for p in ["definition", "meaning", "term", "description"]
            ):
                result["lf_col_idx"] = i
        return result


def extract_tables_to_json(
    file_path: str, config: Optional[dict] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function: PDF -> JSON tables.
    """
    extractor = TableExtractor(config)
    return extractor.extract_tables(file_path)
