# corpus_metadata/corpus_metadata/B_parsing/B03_table_extractor.py
"""
Table extraction from PDF -> JSON/Table model.

Uses unstructured's table extraction (hi_res with infer_table_structure=True)
to populate Table objects in the DocumentGraph.

Also supports:
- Rendering tables as images for vision LLM analysis
- Multi-page table detection and stitching
"""

from __future__ import annotations

import base64
import io
import re
from typing import Any, Dict, List, Optional, Tuple

from unstructured.partition.pdf import partition_pdf

from A_core.A01_domain_models import BoundingBox
from B_parsing.B02_doc_graph import DocumentGraph, Table, TableCell, TableType

# Optional pymupdf for image rendering
try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False

# Optional PIL for image stitching
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    Image = None  # type: ignore[assignment]
    PIL_AVAILABLE = False


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

# Minimum confidence threshold for table detection (0.0-1.0)
MIN_TABLE_CONFIDENCE = 0.5

# Minimum requirements for a valid table
MIN_TABLE_ROWS = 2  # At least header + 1 data row
MIN_TABLE_COLS = 2  # At least 2 columns


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
        is_valid, rejection_reason = self._is_valid_table(rows, html)
        if not is_valid:
            print(f"[INFO] Skipping invalid table on page {page_num}: {rejection_reason}")
            return None

        headers = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []

        # Get bounding box with coordinate conversion
        bbox = self._get_bbox(md, file_path, page_num)

        # Classify table type
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

    def _is_valid_table(self, rows: List[List[str]], html: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate that the detected element is actually a table, not misclassified text.

        Returns (False, reason) for false positives like:
        - Multi-column article text
        - Single-column lists
        - Elements with no proper table structure
        - Tables that are mostly prose text

        Returns (True, None) for valid tables.
        """
        # Must have minimum rows
        if len(rows) < MIN_TABLE_ROWS:
            return False, f"too few rows ({len(rows)} < {MIN_TABLE_ROWS})"

        # Must have minimum columns
        if not rows or len(rows[0]) < MIN_TABLE_COLS:
            col_count = len(rows[0]) if rows else 0
            return False, f"too few columns ({col_count} < {MIN_TABLE_COLS})"

        # Check that HTML actually contains table structure
        if html:
            # Must have actual table tags (not just text wrapped in table tags)
            if "<tr" not in html.lower() or "<td" not in html.lower():
                return False, "missing HTML table tags (<tr>/<td>)"

            # Count actual table cells vs text length ratio
            # A real table has structured short cells, not long paragraphs
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            cells = soup.find_all(["td", "th"])

            if not cells:
                return False, "no table cells found in HTML"

            # Calculate average cell text length
            cell_texts = [cell.get_text(strip=True) for cell in cells]
            if cell_texts:
                avg_cell_length = sum(len(t) for t in cell_texts) / len(cell_texts)

                # If average cell length is very long (>100 chars), likely not a table
                # Real table cells are typically short (numbers, names, codes)
                if avg_cell_length > 100:
                    return False, f"avg cell length too long ({avg_cell_length:.0f} > 100 chars) - likely prose"

                # Check for paragraph-like content (multiple sentences in many cells)
                long_text_cells = sum(1 for t in cell_texts if len(t) > 80)
                long_pct = long_text_cells / len(cell_texts) * 100
                if long_text_cells > len(cell_texts) * 0.2:  # >20% cells have long text
                    return False, f"too many long cells ({long_pct:.0f}% > 80 chars) - likely paragraph text"

                # Check for sentence-like content (periods followed by capital letters)
                # This indicates prose text, not tabular data
                prose_indicators = 0
                for text in cell_texts:
                    # Count cells with multiple sentences (prose indicators)
                    if '. ' in text and len(text) > 50:
                        # Check if it's a sentence pattern (period followed by capital)
                        if re.search(r'\. [A-Z]', text):
                            prose_indicators += 1

                # If >15% of cells look like prose, reject
                prose_pct = prose_indicators / len(cell_texts) * 100 if cell_texts else 0
                if len(cell_texts) > 0 and prose_indicators / len(cell_texts) > 0.15:
                    return False, f"too many prose cells ({prose_pct:.0f}% contain sentences) - likely article text"

                # Check total text length - real tables shouldn't have huge amounts of text
                total_text = sum(len(t) for t in cell_texts)
                if total_text > 3000 and len(cell_texts) < 15:  # Lots of text, few cells
                    return False, f"too much text ({total_text} chars) in few cells ({len(cell_texts)}) - likely prose block"

                # Check for multi-column layout (article text split into columns)
                # If cells have very similar lengths and are long, it's likely prose
                if len(cell_texts) >= 4:
                    lengths = [len(t) for t in cell_texts if len(t) > 20]
                    if lengths:
                        avg_len = sum(lengths) / len(lengths)
                        # If most cells are similar length and long (>60 chars), likely prose columns
                        similar_long = sum(1 for length in lengths if abs(length - avg_len) < avg_len * 0.3 and length > 60)
                        if similar_long > len(lengths) * 0.6:
                            similar_pct = similar_long / len(lengths) * 100
                            return False, f"multi-column prose layout detected ({similar_pct:.0f}% cells have similar long text)"

        # Validate column consistency across rows
        col_counts = [len(row) for row in rows]
        if col_counts:
            # Allow some variation (for merged cells) but not wild inconsistency
            min_cols = min(col_counts)
            max_cols = max(col_counts)
            if max_cols > 0 and min_cols / max_cols < 0.5:  # >50% column count variation
                return False, f"inconsistent column counts ({min_cols}-{max_cols} cols across rows)"

        return True, None

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

    # =========================================================================
    # IMAGE RENDERING METHODS
    # =========================================================================

    def _find_table_bbox_pymupdf(
        self,
        file_path: str,
        page_num: int,
        hint_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Use PyMuPDF's table detection to find actual table boundaries.

        Args:
            file_path: Path to PDF
            page_num: 1-indexed page number
            hint_bbox: Optional hint bbox to find the nearest table

        Returns:
            Bbox tuple (x0, y0, x1, y1) or None if no table found
        """
        if not PYMUPDF_AVAILABLE or fitz is None:
            return None

        try:
            doc = fitz.open(file_path)
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return None

            page = doc[page_num - 1]
            page_width = page.rect.width
            page_height = page.rect.height

            # Use PyMuPDF's table finder
            tables = page.find_tables()

            if not tables or len(tables.tables) == 0:
                doc.close()
                return None

            # If we have a hint bbox, find the closest table
            if hint_bbox:
                # Check if hint_bbox is in a different coordinate space
                hint_out_of_bounds = (
                    hint_bbox[2] > page_width * 1.5 or
                    hint_bbox[3] > page_height * 1.5
                )

                if hint_out_of_bounds:
                    # Hint bbox is in different coordinate space - scale it
                    scale_x = page_width / max(hint_bbox[2], page_width)
                    scale_y = page_height / max(hint_bbox[3], page_height)
                    scale = min(scale_x, scale_y)
                    scaled_hint = (
                        hint_bbox[0] * scale,
                        hint_bbox[1] * scale,
                        hint_bbox[2] * scale,
                        hint_bbox[3] * scale,
                    )
                    hint_center_x = (scaled_hint[0] + scaled_hint[2]) / 2
                    hint_center_y = (scaled_hint[1] + scaled_hint[3]) / 2
                else:
                    hint_center_x = (hint_bbox[0] + hint_bbox[2]) / 2
                    hint_center_y = (hint_bbox[1] + hint_bbox[3]) / 2

                best_table = None
                best_distance = float('inf')

                for table in tables.tables:
                    table_bbox = table.bbox
                    table_center_x = (table_bbox[0] + table_bbox[2]) / 2
                    table_center_y = (table_bbox[1] + table_bbox[3]) / 2

                    distance = ((hint_center_x - table_center_x) ** 2 +
                               (hint_center_y - table_center_y) ** 2) ** 0.5

                    if distance < best_distance:
                        best_distance = distance
                        best_table = table

                if best_table:
                    doc.close()
                    return tuple(best_table.bbox)

            # Otherwise return the largest table
            largest_table = max(tables.tables, key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]))
            doc.close()
            return tuple(largest_table.bbox)

        except Exception as e:
            print(f"[WARN] PyMuPDF table detection failed: {e}")
            return None

    def render_table_as_image(
        self,
        file_path: str,
        page_num: int,
        bbox: Tuple[float, float, float, float],
        dpi: int = 300,
        padding: int = 15,
        bottom_padding: int = 120,
        use_pymupdf_detection: bool = True,
    ) -> Optional[str]:
        """
        Render a table region as a base64-encoded PNG image.

        Args:
            file_path: Path to PDF
            page_num: 1-indexed page number
            bbox: (x0, y0, x1, y1) bounding box in PDF points
            dpi: Resolution for rendering
            padding: Extra points around the table sides/top (default 15pt, ~5mm)
            bottom_padding: Extra points below table for footnotes/captions (default 70pt)
            use_pymupdf_detection: Try to refine bbox using PyMuPDF table detection

        Returns:
            Base64-encoded PNG string, or None if rendering fails
        """
        if not PYMUPDF_AVAILABLE or fitz is None:
            print("[WARN] PyMuPDF not available for table image rendering")
            return None

        try:
            doc = fitz.open(file_path)
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return None

            page = doc[page_num - 1]  # 0-indexed
            page_width = page.rect.width
            page_height = page.rect.height

            # Get bbox coordinates
            x0, y0, x1, y1 = bbox

            # Validate bbox dimensions
            if x1 <= x0 or y1 <= y0:
                doc.close()
                return self.render_full_page(file_path, page_num, dpi)

            # If bbox seems too large or out of bounds, it might be in wrong coordinate space
            if x1 > page_width * 1.1 or y1 > page_height * 1.1:
                print(f"[WARN] Bbox {bbox} seems out of bounds for page size {page_width}x{page_height}")

                # First, try PyMuPDF's native table detection for accurate bbox
                doc.close()
                pymupdf_bbox = self._find_table_bbox_pymupdf(file_path, page_num, (x0, y0, x1, y1))
                doc = fitz.open(file_path)
                page = doc[page_num - 1]

                if pymupdf_bbox:
                    x0, y0, x1, y1 = pymupdf_bbox
                    print(f"[INFO] Using PyMuPDF detected bbox: {pymupdf_bbox}")
                else:
                    # Fallback: scale coordinates from pixel space to PDF point space
                    # Calculate the DPI ratio - Unstructured typically uses 200-300 DPI
                    max_coord = max(x1, y1)
                    max_page = max(page_width, page_height)
                    dpi_ratio = max_page / max_coord  # Approximate scale factor

                    x0_scaled = x0 * dpi_ratio
                    y0_scaled = y0 * dpi_ratio
                    x1_scaled = x1 * dpi_ratio
                    y1_scaled = y1 * dpi_ratio

                    # Calculate table dimensions after scaling
                    table_width = x1_scaled - x0_scaled
                    table_height = y1_scaled - y0_scaled

                    # Be very generous with expansion - 30% of table size on each side
                    # minimum 40 points, capped at 100 points
                    expand_x = max(40, min(100, table_width * 0.3))
                    expand_y = max(40, min(100, table_height * 0.3))

                    x0 = max(0, x0_scaled - expand_x)
                    y0 = max(0, y0_scaled - expand_y)
                    x1 = min(page_width, x1_scaled + expand_x)
                    y1 = min(page_height, y1_scaled + expand_y)
                    print(f"[INFO] Scaled bbox with generous margins: {(x0, y0, x1, y1)}")

                    # Sanity check: if scaled table is suspiciously small, render full page
                    if (x1 - x0) < 100 or (y1 - y0) < 50:
                        print("[WARN] Scaled bbox too small, rendering full page")
                        doc.close()
                        return self.render_full_page(file_path, page_num, dpi)

            # Optionally try to refine bbox using PyMuPDF's table detection
            # Only do this if the original bbox seems unreliable (e.g., was auto-corrected)
            # This is disabled by default as PyMuPDF may find different/smaller tables
            if use_pymupdf_detection and False:  # Disabled - often finds fragments
                doc.close()  # Close before calling detection
                refined_bbox = self._find_table_bbox_pymupdf(file_path, page_num, (x0, y0, x1, y1))
                if refined_bbox:
                    # Check if refined bbox is reasonable
                    orig_area = (x1 - x0) * (y1 - y0)
                    new_area = (refined_bbox[2] - refined_bbox[0]) * (refined_bbox[3] - refined_bbox[1])

                    # Use refined bbox if it has reasonable dimensions (min 50pt in each direction)
                    # and is not tiny compared to page
                    new_width = refined_bbox[2] - refined_bbox[0]
                    new_height = refined_bbox[3] - refined_bbox[1]

                    if new_width > 50 and new_height > 30:
                        # Prefer PyMuPDF's detection - it's usually more accurate
                        # Only reject if the new area is extremely small (<5% of original)
                        if new_area > orig_area * 0.05 or new_area > 5000:
                            print(f"[INFO] Refined bbox from {(x0, y0, x1, y1)} to {refined_bbox}")
                            x0, y0, x1, y1 = refined_bbox
                        else:
                            print(f"[INFO] Skipping refined bbox {refined_bbox} - too small")

                # Reopen doc
                doc = fitz.open(file_path)
                page = doc[page_num - 1]

            # Create clip rectangle with padding
            # Use extra bottom padding to capture footnotes and table captions
            clip_rect = fitz.Rect(
                max(0, x0 - padding),
                max(0, y0 - padding),
                min(page_width, x1 + padding),
                min(page_height, y1 + bottom_padding),
            )

            # Check if clip rect has valid dimensions after clamping
            if clip_rect.width <= 0 or clip_rect.height <= 0:
                doc.close()
                return self.render_full_page(file_path, page_num, dpi)

            # Render to pixmap
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)

            # Convert to base64
            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            doc.close()
            return img_base64

        except Exception as e:
            print(f"[WARN] Failed to render table image: {e}")
            return None

    def render_full_page(
        self,
        file_path: str,
        page_num: int,
        dpi: int = 300,
    ) -> Optional[str]:
        """
        Render full page as image (fallback when bbox is unavailable).

        Args:
            file_path: Path to PDF
            page_num: 1-indexed page number
            dpi: Resolution for rendering

        Returns:
            Base64-encoded PNG string, or None if rendering fails
        """
        if not PYMUPDF_AVAILABLE or fitz is None:
            print("[WARN] PyMuPDF not available for page image rendering")
            return None

        try:
            doc = fitz.open(file_path)
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return None

            page = doc[page_num - 1]  # 0-indexed

            # Render full page to pixmap
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            # Convert to base64
            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            doc.close()
            return img_base64

        except Exception as e:
            print(f"[WARN] Failed to render full page image: {e}")
            return None

    def render_multipage_table(
        self,
        file_path: str,
        table_parts: List[Dict[str, Any]],
        dpi: int = 300,
        padding: int = 5,
    ) -> Optional[str]:
        """
        Render a multi-page table by stitching images vertically.

        Args:
            file_path: Path to PDF
            table_parts: List of dicts with 'page_num' and 'bbox' for each part
            dpi: Resolution for rendering
            padding: Extra pixels around each part

        Returns:
            Base64-encoded PNG string of stitched image, or None if fails
        """
        if not PYMUPDF_AVAILABLE or fitz is None:
            print("[WARN] PyMuPDF not available for table image rendering")
            return None

        if not PIL_AVAILABLE or Image is None:
            print("[WARN] PIL not available for image stitching")
            return None

        try:
            images = []
            doc = fitz.open(file_path)

            for part in table_parts:
                page_num = part["page_num"]
                bbox = part["bbox"]
                full_page = part.get("full_page", False)

                if page_num < 1 or page_num > len(doc):
                    continue

                page = doc[page_num - 1]
                mat = fitz.Matrix(dpi / 72, dpi / 72)

                x0, y0, x1, y1 = bbox
                page_width = page.rect.width
                page_height = page.rect.height

                invalid_bbox = (
                    full_page
                    or bbox == (0.0, 0.0, 0.0, 0.0)
                    or x1 <= x0
                    or y1 <= y0
                )

                if invalid_bbox:
                    # Render full page when bbox is unavailable or invalid
                    pix = page.get_pixmap(matrix=mat)
                else:
                    # Auto-correct coordinates if they seem out of bounds
                    if x1 > page_width * 1.1 or y1 > page_height * 1.1:
                        # Try PyMuPDF's native table detection
                        pymupdf_bbox = self._find_table_bbox_pymupdf(file_path, page_num, (x0, y0, x1, y1))
                        if pymupdf_bbox:
                            x0, y0, x1, y1 = pymupdf_bbox
                        else:
                            # Fallback: scale coordinates
                            scale_x = page_width / max(x1, page_width)
                            scale_y = page_height / max(y1, page_height)
                            scale = min(scale_x, scale_y)
                            x0, y0, x1, y1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
                            # Expand to capture full table
                            expand = 20
                            x0 = max(0, x0 - expand)
                            y0 = max(0, y0 - expand)
                            x1 = min(page_width, x1 + expand)
                            y1 = min(page_height, y1 + expand)

                    # Render specific table region
                    clip_rect = fitz.Rect(
                        max(0, x0 - padding),
                        max(0, y0 - padding),
                        min(page_width, x1 + padding),
                        min(page_height, y1 + padding),
                    )
                    # Check if clip rect has valid dimensions
                    if clip_rect.width <= 0 or clip_rect.height <= 0:
                        pix = page.get_pixmap(matrix=mat)
                    else:
                        pix = page.get_pixmap(matrix=mat, clip=clip_rect)

                # Convert to PIL Image
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                images.append(img)

            doc.close()

            if not images:
                return None

            # Stitch images vertically
            total_width = max(img.width for img in images)
            total_height = sum(img.height for img in images)

            stitched = Image.new("RGB", (total_width, total_height), (255, 255, 255))
            y_offset = 0
            for img in images:
                # Center horizontally if widths differ
                x_offset = (total_width - img.width) // 2
                stitched.paste(img, (x_offset, y_offset))
                y_offset += img.height

            # Convert to base64
            buffer = io.BytesIO()
            stitched.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return img_base64

        except Exception as e:
            print(f"[WARN] Failed to render multi-page table: {e}")
            return None

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
                        table["image_base64"] = self.render_table_as_image(
                            file_path,
                            table["page_num"],
                            table["bbox"],
                            dpi=dpi,
                        )
                    else:
                        # Fallback: render full page when bbox is unavailable
                        table["image_base64"] = self.render_full_page(
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
                    merged["image_base64"] = self.render_multipage_table(
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
