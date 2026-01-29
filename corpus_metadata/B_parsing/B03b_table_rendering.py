# corpus_metadata/B_parsing/B03b_table_rendering.py
"""
Table image rendering using PyMuPDF.

Provides:
- Single table region rendering
- Full page rendering (fallback)
- Multi-page table stitching
- PyMuPDF table detection for bbox refinement

Extracted from B03_table_extractor.py to reduce file size.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

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
    Image = None  # type: ignore[assignment]
    PIL_AVAILABLE = False


def find_table_bbox_pymupdf(
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
        logger.warning("PyMuPDF table detection failed: %s", e)
        return None


def render_table_as_image(
    file_path: str,
    page_num: int,
    bbox: Tuple[float, float, float, float],
    dpi: int = 300,
    padding: int = 15,
    bottom_padding: int = 120,
) -> Optional[str]:
    """
    Render a table region as a base64-encoded PNG image.

    Args:
        file_path: Path to PDF
        page_num: 1-indexed page number
        bbox: (x0, y0, x1, y1) bounding box in PDF points
        dpi: Resolution for rendering
        padding: Extra points around the table sides/top (default 15pt, ~5mm)
        bottom_padding: Extra points below table for footnotes/captions (default 120pt)

    Returns:
        Base64-encoded PNG string, or None if rendering fails
    """
    if not PYMUPDF_AVAILABLE or fitz is None:
        logger.warning("PyMuPDF not available for table image rendering")
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
            return render_full_page(file_path, page_num, dpi)

        # If bbox seems too large or out of bounds, it might be in wrong coordinate space
        if x1 > page_width * 1.1 or y1 > page_height * 1.1:
            logger.warning("Bbox %s seems out of bounds for page size %.0fx%.0f", bbox, page_width, page_height)

            # First, try PyMuPDF's native table detection for accurate bbox
            doc.close()
            pymupdf_bbox = find_table_bbox_pymupdf(file_path, page_num, (x0, y0, x1, y1))
            doc = fitz.open(file_path)
            page = doc[page_num - 1]

            if pymupdf_bbox:
                x0, y0, x1, y1 = pymupdf_bbox
                logger.info("Using PyMuPDF detected bbox: %s", pymupdf_bbox)
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
                logger.info("Scaled bbox with generous margins: %s", (x0, y0, x1, y1))

                # Sanity check: if scaled table is suspiciously small, render full page
                if (x1 - x0) < 100 or (y1 - y0) < 50:
                    logger.warning("Scaled bbox too small, rendering full page")
                    doc.close()
                    return render_full_page(file_path, page_num, dpi)

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
            return render_full_page(file_path, page_num, dpi)

        # Render to pixmap
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, clip=clip_rect)

        # Convert to base64
        img_bytes = pix.tobytes("png")
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        doc.close()
        return img_base64

    except Exception as e:
        logger.warning("Failed to render table image: %s", e)
        return None


def render_full_page(
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
        logger.warning("PyMuPDF not available for page image rendering")
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
        logger.warning("Failed to render full page image: %s", e)
        return None


def render_multipage_table(
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
        logger.warning("PyMuPDF not available for table image rendering")
        return None

    if not PIL_AVAILABLE or Image is None:
        logger.warning("PIL not available for image stitching")
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
                    pymupdf_bbox = find_table_bbox_pymupdf(file_path, page_num, (x0, y0, x1, y1))
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
        logger.warning("Failed to render multi-page table: %s", e)
        return None


__all__ = [
    "PYMUPDF_AVAILABLE",
    "PIL_AVAILABLE",
    "find_table_bbox_pymupdf",
    "render_table_as_image",
    "render_full_page",
    "render_multipage_table",
]
