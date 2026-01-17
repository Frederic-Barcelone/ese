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
    fitz = None  # type: ignore[assignment]
    PYMUPDF_AVAILABLE = False

# Optional PIL for image stitching
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    Image = None  # type: ignore[assignment, misc]
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
                int(td.get("colspan", 1)) for td in tr.find_all(["th", "td"])
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
                colspan = int(td.get("colspan", 1))
                rowspan = int(td.get("rowspan", 1))

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
        self, doc: DocumentGraph, file_path: str, render_images: bool = True
    ) -> DocumentGraph:
        """
        Extract tables from PDF and add them to DocumentGraph.

        Args:
            doc: DocumentGraph to populate
            file_path: Path to PDF file
            render_images: Whether to render table images (default True)
        """
        # Use the new method that includes images and multi-page detection
        tables_data = self.extract_tables_with_images(
            file_path, render_images=render_images
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
                table_type=TableType(t.get("table_type", "UNKNOWN")),
                cells=cells,
                logical_rows=logical_rows,
                bbox=BoundingBox(coords=bbox_coords),
                # New image fields
                image_base64=t.get("image_base64"),
                image_format="png",
                page_nums=t.get("page_nums", [page_num]),
                is_multipage=t.get("is_multipage", False),
                metadata={
                    "headers": headers_map,
                    "ordered_cols": list(range(len(headers))),
                    "glossary_cols": glossary_cols,
                    "html": t.get("html", ""),
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

    # =========================================================================
    # IMAGE RENDERING METHODS
    # =========================================================================

    def render_table_as_image(
        self,
        file_path: str,
        page_num: int,
        bbox: Tuple[float, float, float, float],
        dpi: int = 150,
        padding: int = 10,
    ) -> Optional[str]:
        """
        Render a table region as a base64-encoded PNG image.

        Args:
            file_path: Path to PDF
            page_num: 1-indexed page number
            bbox: (x0, y0, x1, y1) bounding box
            dpi: Resolution for rendering
            padding: Extra pixels around the table

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

            # Create clip rectangle with padding
            x0, y0, x1, y1 = bbox
            clip_rect = fitz.Rect(
                max(0, x0 - padding),
                max(0, y0 - padding),
                min(page.rect.width, x1 + padding),
                min(page.rect.height, y1 + padding),
            )

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
        dpi: int = 150,
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
        dpi: int = 150,
        padding: int = 10,
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

                if full_page or bbox == (0.0, 0.0, 0.0, 0.0):
                    # Render full page when bbox is unavailable
                    pix = page.get_pixmap(matrix=mat)
                else:
                    # Render specific table region
                    x0, y0, x1, y1 = bbox
                    clip_rect = fitz.Rect(
                        max(0, x0 - padding),
                        max(0, y0 - padding),
                        min(page.rect.width, x1 + padding),
                        min(page.rect.height, y1 + padding),
                    )
                    pix = page.get_pixmap(matrix=mat, clip=clip_rect)

                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
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
        dpi: int = 150,
    ) -> List[Dict[str, Any]]:
        """
        Extract tables with both structural data and rendered images.

        Args:
            file_path: Path to PDF
            render_images: Whether to render table images
            dpi: Resolution for image rendering

        Returns:
            List of table dicts, each containing:
            - All fields from extract_tables()
            - 'image_base64': rendered image (if render_images=True)
            - 'page_nums': list of pages (for multi-page tables)
            - 'is_multipage': boolean
        """
        # First, extract structural data
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

                result.append(table)
            else:
                # Multi-page table - merge data and stitch image
                merged = self._merge_table_parts(group)

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
