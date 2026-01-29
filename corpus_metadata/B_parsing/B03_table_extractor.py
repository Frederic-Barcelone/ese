# corpus_metadata/B_parsing/B03_table_extractor.py
"""
Table extraction from PDF -> JSON/Table model.

Uses unstructured's table extraction (hi_res with infer_table_structure=True)
to populate Table objects in the DocumentGraph.

CHANGELOG v2.0:
- Extracted validation logic to B03a_table_validation.py
- Extracted rendering logic to B03b_table_rendering.py
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from unstructured.partition.pdf import partition_pdf

from A_core.A01_domain_models import BoundingBox
from B_parsing.B02_doc_graph import DocumentGraph, Table, TableCell, TableType

# Validation module (B03a)
from B_parsing.B03a_table_validation import (
    MIN_TABLE_CONFIDENCE,
    MIN_TABLE_ROWS,
    MIN_TABLE_COLS,
    is_valid_table,
    is_definition_table_candidate,
)

# Rendering module (B03b)
from B_parsing.B03b_table_rendering import (
    PYMUPDF_AVAILABLE,
    PIL_AVAILABLE,
    find_table_bbox_pymupdf,
    render_table_as_image,
    render_full_page,
    render_multipage_table,
)

# Optional pymupdf for coordinate handling
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore[assignment]


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

            table_data = self._element_to_dict(el, file_path)
            if table_data:
                tables.append(table_data)

        return tables

    def _element_to_dict(self, el, file_path: str = "") -> Optional[Dict[str, Any]]:
        """Convert unstructured table element to dict."""
        md = getattr(el, "metadata", None)
        if not md:
            return None

        # Check confidence score if available
        detection_confidence = getattr(md, "detection_class_prob", None)
        if detection_confidence is not None and detection_confidence < MIN_TABLE_CONFIDENCE:
            print(f"[INFO] Skipping low-confidence table detection: {detection_confidence:.2f}")
            return None

        # Get HTML table if available
        html = getattr(md, "text_as_html", None)
        page_num = getattr(md, "page_number", None) or 1

        # Parse HTML to rows FIRST to validate table structure
        rows = self._parse_html_table(html) if html else []

        # Validate table structure - reject false positives
        # Returns (is_valid, reason) where reason may indicate salvage info
        is_valid, validation_info = is_valid_table(rows, html)
        if not is_valid:
            print(f"[INFO] Skipping invalid table on page {page_num}: {validation_info}")
            return None

        # Check if this table was salvaged as a definition table
        is_salvaged = validation_info and "definition_table_salvaged" in validation_info
        if is_salvaged:
            print(f"[INFO] Salvaged definition table on page {page_num}: {validation_info}")

        headers = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []

        # Get bounding box with coordinate conversion
        bbox = self._get_bbox(md, file_path, page_num)

        # Classify table type (force GLOSSARY for salvaged definition tables)
        if is_salvaged:
            table_type = TableType.GLOSSARY.value
        else:
            table_type = self._classify_table(headers)

        return {
            "page_num": page_num,
            "bbox": bbox,
            "headers": headers,
            "rows": data_rows,
            "table_type": table_type,
            "html": html,
            "confidence": detection_confidence,
        }

    def _get_bbox(
        self, md, file_path: str = "", page_num: int = 1
    ) -> Tuple[float, float, float, float]:
        """
        Extract bounding box from metadata and convert coordinates.

        Unstructured's hi_res mode returns coordinates in pixel space (at ~200 DPI).
        We need to convert to PDF point space (72 DPI) for PyMuPDF rendering.

        If layout_width/layout_height are not available, we detect pixel-space
        coordinates by checking if they exceed PDF page dimensions and scale accordingly.
        """
        coords = getattr(md, "coordinates", None)
        if not coords:
            return (0.0, 0.0, 0.0, 0.0)

        points = getattr(coords, "points", None)
        if not points:
            return (0.0, 0.0, 0.0, 0.0)

        try:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

            # Get coordinate system info from Unstructured metadata
            layout_width = getattr(coords, "layout_width", None)
            layout_height = getattr(coords, "layout_height", None)

            # Always try to convert coordinates to PDF space
            if file_path and PYMUPDF_AVAILABLE and fitz is not None:
                try:
                    doc = fitz.open(file_path)
                    if 0 < page_num <= len(doc):
                        page = doc[page_num - 1]
                        pdf_width = page.rect.width
                        pdf_height = page.rect.height

                        # Determine if we need to scale coordinates
                        # If layout dimensions provided, use them
                        if layout_width and layout_height:
                            scale_x = pdf_width / layout_width
                            scale_y = pdf_height / layout_height
                        else:
                            # No layout dimensions - check if coords are in pixel space
                            # by seeing if they exceed PDF page dimensions
                            max_coord = max(x1, y1)
                            max_page = max(pdf_width, pdf_height)

                            if max_coord > max_page * 1.1:  # Coords are in pixel space
                                # Estimate scale based on typical Unstructured DPI (~200)
                                # Unstructured renders at 200 DPI, PDF is 72 DPI
                                # So pixel_coord * (72/200) = pdf_coord
                                estimated_dpi = 200
                                scale = 72.0 / estimated_dpi
                                scale_x = scale
                                scale_y = scale
                            else:
                                # Coords already in PDF space
                                scale_x = 1.0
                                scale_y = 1.0

                        # Apply scaling
                        x0 = x0 * scale_x
                        y0 = y0 * scale_y
                        x1 = x1 * scale_x
                        y1 = y1 * scale_y

                        # Clamp to page bounds with small margin
                        x0 = max(0, x0)
                        y0 = max(0, y0)
                        x1 = min(pdf_width, x1)
                        y1 = min(pdf_height, y1)

                    doc.close()
                except Exception as e:
                    print(f"[WARN] Coordinate conversion failed: {e}")

            return (x0, y0, x1, y1)
        except (TypeError, IndexError):
            return (0.0, 0.0, 0.0, 0.0)

    def _parse_html_table(self, html: str) -> List[List[str]]:
        """Parse HTML table to list of rows with colspan/rowspan support."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # First pass: determine grid size and track spans
        all_trs = soup.find_all("tr")
        if not all_trs:
            return []

        # Calculate max columns considering colspans
        max_cols = 0
        for tr in all_trs:
            col_count = sum(
                int(str(td.get("colspan") or 1)) for td in tr.find_all(["th", "td"])
            )
            max_cols = max(max_cols, col_count)

        # Build grid with span tracking
        grid: List[List[str]] = []
        rowspan_tracker: Dict[int, Tuple[str, int]] = {}  # col_idx -> (text, remaining_rows)

        for tr in all_trs:
            row: List[Optional[str]] = [None] * max_cols
            col_idx = 0

            for td in tr.find_all(["th", "td"]):
                # Skip columns occupied by rowspan from previous rows
                while col_idx < max_cols and col_idx in rowspan_tracker:
                    text, remaining = rowspan_tracker[col_idx]
                    row[col_idx] = text
                    if remaining <= 1:
                        del rowspan_tracker[col_idx]
                    else:
                        rowspan_tracker[col_idx] = (text, remaining - 1)
                    col_idx += 1

                if col_idx >= max_cols:
                    break

                cell_text = td.get_text(strip=True)
                colspan = int(str(td.get("colspan") or 1))
                rowspan = int(str(td.get("rowspan") or 1))

                # Fill cells for colspan
                for i in range(colspan):
                    if col_idx + i < max_cols:
                        row[col_idx + i] = cell_text

                # Track rowspan for future rows
                if rowspan > 1:
                    for i in range(colspan):
                        if col_idx + i < max_cols:
                            rowspan_tracker[col_idx + i] = (cell_text, rowspan - 1)

                col_idx += colspan

            # Fill any remaining rowspan cells
            while col_idx < max_cols:
                if col_idx in rowspan_tracker:
                    text, remaining = rowspan_tracker[col_idx]
                    row[col_idx] = text
                    if remaining <= 1:
                        del rowspan_tracker[col_idx]
                    else:
                        rowspan_tracker[col_idx] = (text, remaining - 1)
                col_idx += 1

            # Convert None to empty string and add row
            grid.append([cell or "" for cell in row])

        return grid

    def _classify_table(self, headers: List[str]) -> str:
        """Classify table type based on headers."""
        if not headers:
            return TableType.UNKNOWN.value

        header_text = " ".join(headers).lower()
        if GLOSSARY_HEADER_RE.search(header_text):
            return TableType.GLOSSARY.value

        return TableType.DATA_GRID.value

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
            render_images: Whether to render table images (default True)
            use_vlm: Whether to use VLM for table structure extraction
            vlm_extractor: VLMTableExtractor instance (required if use_vlm=True)
        """
        # Use the new method that includes images and multi-page detection
        tables_data = self.extract_tables_with_images(
            file_path,
            render_images=render_images,
            use_vlm=use_vlm,
            vlm_extractor=vlm_extractor,
        )

        for idx, t in enumerate(tables_data):
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

            # Build table metadata
            table_metadata = {
                "headers": headers_map,
                "ordered_cols": list(range(len(headers))),
                "glossary_cols": glossary_cols,
                "html": t.get("html", ""),
                "extraction_method": t.get("extraction_method", "html"),
            }

            # Add VLM-specific metadata if available
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
                # New image fields
                image_base64=t.get("image_base64"),
                image_format="png",
                page_nums=t.get("page_nums", [page_num]),
                is_multipage=t.get("is_multipage", False),
                # Extraction method tracking
                extraction_method=t.get("extraction_method", "html"),
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
            elif any(
                p in h_lower for p in ["definition", "meaning", "term", "description"]
            ):
                result["lf_col_idx"] = i
        return result

    # NOTE: Image rendering methods extracted to B03b_table_rendering.py

    def detect_multipage_tables(
        self, tables_data: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect tables that span multiple pages.

        Heuristics:
        - Tables on consecutive pages
        - Similar column count / header structure
        - Table at bottom of page followed by table at top of next page
        - "Continued" or "(cont.)" markers

        Args:
            tables_data: List of table dicts from extract_tables()

        Returns:
            List of table groups. Single-page tables are groups of 1.
            Multi-page tables are groups with multiple parts.
        """
        if not tables_data:
            return []

        # Sort by page number, then by vertical position (y0)
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

            # Check if this could be a continuation
            if self._is_table_continuation(prev_table, table):
                current_group.append(table)
            else:
                # Finalize current group and start new one
                groups.append(current_group)
                current_group = [table]

        # Don't forget the last group
        if current_group:
            groups.append(current_group)

        return groups

    def _is_table_continuation(
        self,
        prev_table: Dict[str, Any],
        curr_table: Dict[str, Any],
    ) -> bool:
        """
        Check if curr_table is a continuation of prev_table.
        """
        prev_page = prev_table["page_num"]
        curr_page = curr_table["page_num"]

        # Must be on consecutive pages
        if curr_page != prev_page + 1:
            return False

        # Check column count similarity
        prev_cols = len(prev_table.get("headers", []))
        curr_cols = len(curr_table.get("headers", []))

        # If both have headers and they match, likely continuation
        if prev_cols > 0 and curr_cols > 0:
            # Allow for slight differences (merged headers)
            if abs(prev_cols - curr_cols) <= 1:
                # Check if headers are similar (repeated on continuation)
                prev_headers = [h.lower().strip() for h in prev_table.get("headers", [])]
                curr_headers = [h.lower().strip() for h in curr_table.get("headers", [])]

                # If headers match exactly, it's a continuation with repeated headers
                if prev_headers == curr_headers:
                    return True

                # Check for "continued" marker in current table headers or rows
                all_text = " ".join(curr_headers)
                if curr_table.get("rows"):
                    first_row = curr_table["rows"][0] if curr_table["rows"] else []
                    all_text += " " + " ".join(str(c) for c in first_row)

                continued_markers = ["continued", "cont.", "cont'd", "(continued)"]
                if any(marker in all_text.lower() for marker in continued_markers):
                    return True

        # Check vertical position heuristics
        # Previous table should be near bottom of page
        # Current table should be near top of page
        prev_bbox = prev_table.get("bbox", (0, 0, 0, 0))
        curr_bbox = curr_table.get("bbox", (0, 0, 0, 0))

        # If prev table's y1 (bottom) is > 70% down the page
        # and curr table's y0 (top) is < 30% down the page
        # This is a rough heuristic; actual page dimensions would be better
        if prev_bbox[3] > 500 and curr_bbox[1] < 200:  # Assuming ~700pt page height
            # Additional check: similar column structure
            if prev_cols == curr_cols or (prev_cols > 0 and curr_cols == 0):
                return True

        return False

    def extract_tables_with_images(
        self,
        file_path: str,
        render_images: bool = True,
        dpi: int = 300,
        use_vlm: bool = False,
        vlm_extractor: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract tables with both structural data and rendered images.

        Pipeline:
        1. Get bbox from Unstructured (YOLOX detection)
        2. Render table image at 300 DPI
        3. If use_vlm=True, send to VLM for structure extraction
        4. Otherwise fall back to Unstructured HTML parsing

        Args:
            file_path: Path to PDF
            render_images: Whether to render table images
            dpi: Resolution for image rendering (default 300)
            use_vlm: Whether to use VLM for table structure extraction
            vlm_extractor: VLMTableExtractor instance (required if use_vlm=True)

        Returns:
            List of table dicts, each containing:
            - All fields from extract_tables()
            - 'image_base64': rendered image (if render_images=True)
            - 'page_nums': list of pages (for multi-page tables)
            - 'is_multipage': boolean
            - 'extraction_method': 'vlm', 'html', or 'html_fallback'
        """
        # First, extract structural data (bbox from Unstructured)
        tables_data = self.extract_tables(file_path)

        if not tables_data:
            return []

        # Detect multi-page tables
        table_groups = self.detect_multipage_tables(tables_data)

        result = []
        for group in table_groups:
            if len(group) == 1:
                # Single-page table
                table = group[0]
                table["page_nums"] = [table["page_num"]]
                table["is_multipage"] = False
                table["extraction_method"] = "html"  # Default

                if render_images:
                    if table["bbox"] != (0.0, 0.0, 0.0, 0.0):
                        # Render specific table region
                        table["image_base64"] = render_table_as_image(
                            file_path,
                            table["page_num"],
                            table["bbox"],
                            dpi=dpi,
                        )
                    else:
                        # Fallback: render full page when bbox is unavailable
                        table["image_base64"] = render_full_page(
                            file_path,
                            table["page_num"],
                            dpi=dpi,
                        )
                else:
                    table["image_base64"] = None

                # VLM extraction (replaces HTML parsing)
                if use_vlm and vlm_extractor and table.get("image_base64"):
                    try:
                        vlm_result = vlm_extractor.extract(table["image_base64"])
                        if vlm_result and vlm_result.get("confidence", 0) > 0.3:
                            vlm_rows = vlm_result.get("rows", [])
                            vlm_headers = vlm_result.get("headers", [])
                            vlm_row_count = len(vlm_rows)
                            vlm_col_count = len(vlm_headers)

                            # Validate VLM extraction meets minimum table requirements
                            # Require at least 2 data rows (not just 1) for a meaningful table
                            # This filters out false positives where VLM extracts a tiny snippet
                            min_vlm_data_rows = 2  # Stricter than MIN_TABLE_ROWS - 1
                            if vlm_row_count < min_vlm_data_rows or vlm_col_count < MIN_TABLE_COLS:
                                print(f"[INFO] Rejecting VLM table on page {table['page_num']}: "
                                      f"too small ({vlm_col_count} cols, {vlm_row_count} data rows) - "
                                      f"requires at least {MIN_TABLE_COLS} cols and {min_vlm_data_rows} data rows")
                                # Skip this table entirely - VLM confirmed it's not a real table
                                continue

                            # Use VLM results
                            table["headers"] = vlm_headers
                            table["rows"] = vlm_rows
                            table["extraction_method"] = "vlm"
                            table["vlm_confidence"] = vlm_result.get("confidence", 0.95)
                            if vlm_result.get("verification_warning"):
                                table["vlm_warning"] = vlm_result["verification_warning"]
                            if vlm_result.get("notes"):
                                table["notes"] = vlm_result["notes"]
                            print(f"[INFO] VLM extracted table on page {table['page_num']}: "
                                  f"{len(table['headers'])} cols, {len(table['rows'])} rows")
                        else:
                            # VLM failed, keep HTML fallback
                            table["extraction_method"] = "html_fallback"
                            print(f"[WARN] VLM extraction failed for table on page {table['page_num']}, "
                                  f"using HTML fallback")
                    except Exception as e:
                        print(f"[WARN] VLM extraction error on page {table['page_num']}: {e}")
                        table["extraction_method"] = "html_fallback"

                result.append(table)
            else:
                # Multi-page table - merge data and stitch image
                merged = self._merge_table_parts(group)
                merged["extraction_method"] = "html"  # Default for multi-page

                if render_images:
                    # Include all parts, marking those with empty bbox for full-page render
                    parts = [
                        {
                            "page_num": t["page_num"],
                            "bbox": t["bbox"],
                            "full_page": t["bbox"] == (0.0, 0.0, 0.0, 0.0),
                        }
                        for t in group
                    ]
                    merged["image_base64"] = render_multipage_table(
                        file_path, parts, dpi=dpi
                    )
                else:
                    merged["image_base64"] = None

                # VLM extraction for multi-page tables
                if use_vlm and vlm_extractor and merged.get("image_base64"):
                    try:
                        vlm_result = vlm_extractor.extract(merged["image_base64"])
                        if vlm_result and vlm_result.get("confidence", 0) > 0.3:
                            vlm_rows = vlm_result.get("rows", [])
                            vlm_headers = vlm_result.get("headers", [])
                            vlm_row_count = len(vlm_rows)
                            vlm_col_count = len(vlm_headers)

                            # Validate VLM extraction meets minimum table requirements
                            min_vlm_data_rows = 2  # Stricter than MIN_TABLE_ROWS - 1
                            if vlm_row_count < min_vlm_data_rows or vlm_col_count < MIN_TABLE_COLS:
                                print(f"[INFO] Rejecting VLM multi-page table on pages {merged['page_nums']}: "
                                      f"too small ({vlm_col_count} cols, {vlm_row_count} data rows)")
                                continue

                            merged["headers"] = vlm_headers
                            merged["rows"] = vlm_rows
                            merged["extraction_method"] = "vlm"
                            merged["vlm_confidence"] = vlm_result.get("confidence", 0.95)
                            if vlm_result.get("verification_warning"):
                                merged["vlm_warning"] = vlm_result["verification_warning"]
                            if vlm_result.get("notes"):
                                merged["notes"] = vlm_result["notes"]
                            print(f"[INFO] VLM extracted multi-page table: "
                                  f"{len(merged['headers'])} cols, {len(merged['rows'])} rows")
                        else:
                            merged["extraction_method"] = "html_fallback"
                            print("[WARN] VLM extraction failed for multi-page table, using HTML fallback")
                    except Exception as e:
                        print(f"[WARN] VLM extraction error for multi-page table: {e}")
                        merged["extraction_method"] = "html_fallback"

                result.append(merged)

        return result

    def _merge_table_parts(
        self, parts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge multiple table parts into a single table dict.
        """
        if not parts:
            return {}

        first = parts[0]
        merged = {
            "page_num": first["page_num"],  # Primary page
            "page_nums": [p["page_num"] for p in parts],
            "is_multipage": True,
            "bbox": first["bbox"],  # Bbox of first part
            "headers": first.get("headers", []),
            "table_type": first.get("table_type", "UNKNOWN"),
            "html": first.get("html", ""),
        }

        # Merge rows from all parts
        all_rows = []
        for i, part in enumerate(parts):
            rows = part.get("rows", [])
            if i > 0:
                # Skip header row on continuation pages if it's a repeat
                first_headers = [h.lower().strip() for h in first.get("headers", [])]
                if rows:
                    first_row = [str(c).lower().strip() for c in rows[0]]
                    if first_row == first_headers:
                        rows = rows[1:]  # Skip repeated header
            all_rows.extend(rows)

        merged["rows"] = all_rows

        return merged


def extract_tables_to_json(
    file_path: str, config: Optional[dict] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function: PDF -> JSON tables.
    """
    extractor = TableExtractor(config)
    return extractor.extract_tables(file_path)
