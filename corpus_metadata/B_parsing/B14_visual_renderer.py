# corpus_metadata/B_parsing/B14_visual_renderer.py
"""
Visual renderer with point-based padding and adaptive DPI.

This module renders visual elements (tables and figures) at high resolution
using PyMuPDF with strict coordinate space discipline. All coordinates are
in PDF points (canonical), padding is point-based, and DPI adapts based on
visual size for optimal quality/size tradeoffs.

Key Components:
    - RenderConfig: Configuration for DPI, padding, and image format
    - render_visual: Main rendering function with caption-aware padding
    - render_table_as_image: Render table region as base64 image
    - render_full_page_from_path: Render entire page as image
    - render_multipage_table: Render multi-page table as single image
    - pts_to_pixels: Convert PDF points to pixels at given DPI
    - pixels_to_pts: Convert pixels to PDF points at given DPI
    - bbox_pts_to_pixels: Convert bbox from PDF points to pixels

Example:
    >>> import fitz
    >>> from B_parsing.B14_visual_renderer import render_visual, RenderConfig
    >>> doc = fitz.open("paper.pdf")
    >>> config = RenderConfig(default_dpi=300)
    >>> result = render_visual(doc, page_num=1, bbox_pts=(100, 200, 400, 500))
    >>> print(f"Rendered at {result.dpi} DPI, {len(result.image_base64)} bytes")

Dependencies:
    - A_core.A13_visual_models: PageLocation
    - fitz (PyMuPDF): PDF rendering
"""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from A_core.A13_visual_models import PageLocation

logger = logging.getLogger(__name__)


# -------------------------
# Configuration
# -------------------------


@dataclass
class RenderConfig:
    """Configuration for visual rendering."""

    # Base DPI
    default_dpi: int = 300
    min_dpi: int = 200
    max_dpi: int = 400

    # Adaptive DPI thresholds (visual area in square points)
    small_visual_threshold: float = 10000.0  # ~1.4 inch square
    large_visual_threshold: float = 100000.0  # ~4.4 inch square

    # Padding in PDF points (minimal - VLM should provide precise bboxes)
    padding_sides_pts: float = 6.0  # ~0.08 inch - minimal margin
    padding_caption_pts: float = 12.0  # ~0.17 inch - small buffer for caption

    # Image format
    image_format: str = "png"
    jpeg_quality: int = 95


# -------------------------
# Coordinate Conversion
# -------------------------


def pts_to_pixels(pts: float, dpi: int) -> float:
    """
    Convert PDF points to pixels at given DPI.

    PDF points are 1/72 inch.

    Args:
        pts: Value in PDF points
        dpi: Dots per inch

    Returns:
        Value in pixels
    """
    return pts * dpi / 72.0


def pixels_to_pts(pixels: float, dpi: int) -> float:
    """
    Convert pixels to PDF points at given DPI.

    Args:
        pixels: Value in pixels
        dpi: Dots per inch

    Returns:
        Value in PDF points
    """
    return pixels * 72.0 / dpi


def bbox_pts_to_pixels(
    bbox_pts: Tuple[float, float, float, float],
    dpi: int,
) -> Tuple[int, int, int, int]:
    """
    Convert bbox from PDF points to pixels.

    Args:
        bbox_pts: Bounding box in PDF points (x0, y0, x1, y1)
        dpi: Dots per inch

    Returns:
        Bounding box in pixels (x0, y0, x1, y1)
    """
    return (
        int(pts_to_pixels(bbox_pts[0], dpi)),
        int(pts_to_pixels(bbox_pts[1], dpi)),
        int(pts_to_pixels(bbox_pts[2], dpi)),
        int(pts_to_pixels(bbox_pts[3], dpi)),
    )


# -------------------------
# Adaptive DPI
# -------------------------


def compute_adaptive_dpi(
    bbox_pts: Tuple[float, float, float, float],
    config: RenderConfig = RenderConfig(),
) -> int:
    """
    Compute adaptive DPI based on visual size.

    Small visuals get higher DPI to ensure readability.
    Large visuals get lower DPI to limit file size.

    Args:
        bbox_pts: Bounding box in PDF points
        config: Render configuration

    Returns:
        DPI to use for rendering
    """
    x0, y0, x1, y1 = bbox_pts
    area = (x1 - x0) * (y1 - y0)

    if area < config.small_visual_threshold:
        # Small visual - use higher DPI
        return config.max_dpi
    elif area > config.large_visual_threshold:
        # Large visual - use lower DPI
        return config.min_dpi
    else:
        # Medium visual - use default
        return config.default_dpi


# -------------------------
# Padding Functions
# -------------------------


def expand_bbox_with_padding(
    bbox_pts: Tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    padding_sides_pts: float = 12.0,
    padding_top_pts: float = 0.0,
    padding_bottom_pts: float = 72.0,
) -> Tuple[float, float, float, float]:
    """
    Expand bbox with point-based padding.

    Clamps to page bounds.

    Args:
        bbox_pts: Original bounding box in PDF points
        page_width: Page width in points
        page_height: Page height in points
        padding_sides_pts: Left/right padding in points
        padding_top_pts: Top padding in points
        padding_bottom_pts: Bottom padding in points (for caption)

    Returns:
        Expanded bounding box clamped to page
    """
    x0, y0, x1, y1 = bbox_pts

    # Expand with padding
    x0_padded = max(0, x0 - padding_sides_pts)
    y0_padded = max(0, y0 - padding_top_pts)
    x1_padded = min(page_width, x1 + padding_sides_pts)
    y1_padded = min(page_height, y1 + padding_bottom_pts)

    return (x0_padded, y0_padded, x1_padded, y1_padded)


def compute_caption_padding(
    visual_type: str,
    caption_position: Optional[str] = None,
    config: RenderConfig = RenderConfig(),
) -> Dict[str, float]:
    """
    Compute padding for a visual based on type and caption position.

    Args:
        visual_type: "table" or "figure"
        caption_position: Where caption is expected ("above", "below", etc.)
        config: Render configuration

    Returns:
        Dict with padding values for each side
    """
    padding = {
        "left": config.padding_sides_pts,
        "right": config.padding_sides_pts,
        "top": config.padding_sides_pts,
        "bottom": config.padding_sides_pts,
    }

    # Add extra padding for caption zone
    if caption_position == "above":
        padding["top"] = config.padding_caption_pts
    elif caption_position == "below":
        padding["bottom"] = config.padding_caption_pts
    elif caption_position == "left":
        padding["left"] = config.padding_caption_pts
    elif caption_position == "right":
        padding["right"] = config.padding_caption_pts
    else:
        # Default: tables often have captions above, figures below
        if visual_type == "table":
            padding["top"] = config.padding_caption_pts
        else:
            padding["bottom"] = config.padding_caption_pts

    return padding


# -------------------------
# PyMuPDF Table Detection
# -------------------------


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


# -------------------------
# Rendering Functions
# -------------------------


@dataclass
class RenderedVisual:
    """Result of rendering a visual."""

    image_base64: str
    image_format: str
    dpi: int
    rendered_bbox_pts: Tuple[float, float, float, float]
    original_bbox_pts: Tuple[float, float, float, float]
    page_num: int


def render_visual_region(
    doc: fitz.Document,
    page_num: int,
    bbox_pts: Tuple[float, float, float, float],
    dpi: int = 300,
    image_format: str = "png",
    jpeg_quality: int = 95,
) -> Optional[str]:
    """
    Render a region of a page as base64-encoded image.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        bbox_pts: Bounding box in PDF points to render
        dpi: Dots per inch for rendering
        image_format: Output format ("png" or "jpeg")
        jpeg_quality: JPEG quality if using jpeg format

    Returns:
        Base64-encoded image string, or None if rendering fails
    """
    try:
        page = doc[page_num - 1]

        # Create clip rect from bbox
        clip = fitz.Rect(bbox_pts)

        # Compute zoom factor from DPI (72 is PDF's native DPI)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        # Render the clipped region
        pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)

        # Convert to bytes
        if image_format.lower() == "jpeg":
            img_bytes = pixmap.tobytes(output="jpeg", jpg_quality=jpeg_quality)
        else:
            img_bytes = pixmap.tobytes(output="png")

        # Encode to base64
        return base64.b64encode(img_bytes).decode("utf-8")

    except Exception as e:
        logger.warning(f"Failed to render region on page {page_num}: {e}")
        return None


def render_visual(
    doc: fitz.Document,
    page_num: int,
    bbox_pts: Tuple[float, float, float, float],
    visual_type: str = "figure",
    caption_position: Optional[str] = None,
    config: RenderConfig = RenderConfig(),
) -> Optional[RenderedVisual]:
    """
    Render a visual with appropriate padding and DPI.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        bbox_pts: Visual bounding box in PDF points
        visual_type: "table" or "figure"
        caption_position: Where caption is located
        config: Render configuration

    Returns:
        RenderedVisual with image and metadata, or None if failed
    """
    page = doc[page_num - 1]
    page_width = page.rect.width
    page_height = page.rect.height

    # Compute padding
    padding = compute_caption_padding(visual_type, caption_position, config)

    # Expand bbox with padding
    expanded_bbox = expand_bbox_with_padding(
        bbox_pts,
        page_width,
        page_height,
        padding_sides_pts=padding["left"],  # Use symmetric for sides
        padding_top_pts=padding["top"],
        padding_bottom_pts=padding["bottom"],
    )

    # Compute adaptive DPI
    dpi = compute_adaptive_dpi(bbox_pts, config)

    # Render
    image_base64 = render_visual_region(
        doc,
        page_num,
        expanded_bbox,
        dpi=dpi,
        image_format=config.image_format,
        jpeg_quality=config.jpeg_quality,
    )

    if image_base64 is None:
        return None

    return RenderedVisual(
        image_base64=image_base64,
        image_format=config.image_format,
        dpi=dpi,
        rendered_bbox_pts=expanded_bbox,
        original_bbox_pts=bbox_pts,
        page_num=page_num,
    )


def render_full_page(
    doc: fitz.Document,
    page_num: int,
    dpi: int = 300,
    image_format: str = "png",
) -> Optional[str]:
    """
    Render a full page as base64-encoded image.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        dpi: Dots per inch
        image_format: Output format

    Returns:
        Base64-encoded image string
    """
    try:
        page = doc[page_num - 1]

        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        if image_format.lower() == "jpeg":
            img_bytes = pixmap.tobytes(output="jpeg")
        else:
            img_bytes = pixmap.tobytes(output="png")

        return base64.b64encode(img_bytes).decode("utf-8")

    except Exception as e:
        logger.warning(f"Failed to render page {page_num}: {e}")
        return None


# -------------------------
# Multi-Page Rendering
# -------------------------


def render_multipage_visual(
    doc: fitz.Document,
    locations: List[PageLocation],
    visual_type: str = "figure",
    config: RenderConfig = RenderConfig(),
) -> Optional[str]:
    """
    Render a multi-page visual by stitching images vertically.

    Args:
        doc: Open PyMuPDF document
        locations: List of PageLocation for each page
        visual_type: "table" or "figure"
        config: Render configuration

    Returns:
        Base64-encoded stitched image
    """
    try:
        from PIL import Image

        images: List[Image.Image] = []

        for loc in locations:
            # Render each part
            result = render_visual(
                doc,
                loc.page_num,
                loc.bbox_pts,
                visual_type=visual_type,
                config=config,
            )

            if result is None:
                continue

            # Decode from base64
            img_bytes = base64.b64decode(result.image_base64)
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)

        if not images:
            return None

        # Stitch vertically
        total_height = sum(img.height for img in images)
        max_width = max(img.width for img in images)

        stitched = Image.new("RGB", (max_width, total_height), (255, 255, 255))

        y_offset = 0
        for img in images:  # type: ignore[assignment]
            # Center horizontally if narrower
            x_offset = (max_width - img.width) // 2
            stitched.paste(img, (x_offset, y_offset))
            y_offset += img.height

        # Encode to base64
        buffer = io.BytesIO()
        stitched.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except ImportError:
        logger.warning("PIL not available for multi-page stitching")
        return None
    except Exception as e:
        logger.warning(f"Failed to stitch multi-page visual: {e}")
        return None


# -------------------------
# Utility Functions
# -------------------------


def get_image_dimensions_from_base64(image_base64: str) -> Optional[Tuple[int, int]]:
    """
    Get image dimensions from base64-encoded image.

    Args:
        image_base64: Base64-encoded image

    Returns:
        Tuple of (width, height) in pixels, or None if failed
    """
    try:
        from PIL import Image

        img_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_bytes))
        return img.size

    except Exception:
        return None


def resize_image_for_vlm(
    image_base64: str,
    max_dimension: int = 8000,
    max_bytes: int = 5 * 1024 * 1024,  # 5MB
) -> str:
    """
    Resize image to fit within VLM limits.

    Args:
        image_base64: Original base64-encoded image
        max_dimension: Maximum width or height in pixels
        max_bytes: Maximum file size in bytes

    Returns:
        Resized image as base64, or original if already fits
    """
    try:
        from PIL import Image

        img_bytes = base64.b64decode(image_base64)

        # Check current size
        if len(img_bytes) <= max_bytes:
            img = Image.open(io.BytesIO(img_bytes))
            if max(img.size) <= max_dimension:
                return image_base64

        # Need to resize
        img = Image.open(io.BytesIO(img_bytes))
        width, height = img.size

        # Scale to fit max dimension
        if max(width, height) > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)  # type: ignore[assignment]

        # Encode with quality reduction if needed
        quality = 95
        while quality >= 50:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")

            if buffer.tell() <= max_bytes:
                return base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Try JPEG with lower quality
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)

            if buffer.tell() <= max_bytes:
                return base64.b64encode(buffer.getvalue()).decode("utf-8")

            quality -= 10

        # Last resort: return whatever we have
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=50)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except Exception as e:
        logger.warning(f"Failed to resize image: {e}")
        return image_base64


# -------------------------
# File-Path Wrapper Functions
# -------------------------
# These functions provide a simpler API that opens/closes the PDF internally,
# used by B03_table_extractor and other modules that work with file paths.


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
    try:
        doc = fitz.open(file_path)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return None

        page = doc[page_num - 1]
        page_width = page.rect.width
        page_height = page.rect.height

        x0, y0, x1, y1 = bbox

        # Validate bbox dimensions
        if x1 <= x0 or y1 <= y0:
            doc.close()
            return render_full_page_from_path(file_path, page_num, dpi)

        # Try PyMuPDF's native table detection for accurate bbox
        doc.close()
        pymupdf_bbox = find_table_bbox_pymupdf(file_path, page_num, (x0, y0, x1, y1))
        doc = fitz.open(file_path)
        page = doc[page_num - 1]

        if pymupdf_bbox:
            x0, y0, x1, y1 = pymupdf_bbox
            logger.debug("Using PyMuPDF refined bbox: %s", pymupdf_bbox)
        elif x1 > page_width * 1.1 or y1 > page_height * 1.1:
            # Bbox seems out of bounds, scale coordinates
            logger.warning(
                "Bbox %s seems out of bounds for page size %.0fx%.0f, scaling coordinates",
                bbox, page_width, page_height
            )
            max_coord = max(x1, y1)
            max_page = max(page_width, page_height)
            dpi_ratio = max_page / max_coord

            x0_scaled = x0 * dpi_ratio
            y0_scaled = y0 * dpi_ratio
            x1_scaled = x1 * dpi_ratio
            y1_scaled = y1 * dpi_ratio

            table_width = x1_scaled - x0_scaled
            table_height = y1_scaled - y0_scaled

            expand_x = max(40, min(100, table_width * 0.3))
            expand_y = max(40, min(100, table_height * 0.3))

            x0 = max(0, x0_scaled - expand_x)
            y0 = max(0, y0_scaled - expand_y)
            x1 = min(page_width, x1_scaled + expand_x)
            y1 = min(page_height, y1_scaled + expand_y)
            logger.info("Scaled bbox with generous margins: %s", (x0, y0, x1, y1))

            if (x1 - x0) < 100 or (y1 - y0) < 50:
                logger.warning("Scaled bbox too small, rendering full page")
                doc.close()
                return render_full_page_from_path(file_path, page_num, dpi)

        # Create clip rectangle with padding
        clip_rect = fitz.Rect(
            max(0, x0 - padding),
            max(0, y0 - padding),
            min(page_width, x1 + padding),
            min(page_height, y1 + bottom_padding),
        )

        if clip_rect.width <= 0 or clip_rect.height <= 0:
            doc.close()
            return render_full_page_from_path(file_path, page_num, dpi)

        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, clip=clip_rect)

        img_bytes = pix.tobytes("png")
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        doc.close()
        return img_base64

    except Exception as e:
        logger.warning("Failed to render table image: %s", e)
        return None


def render_full_page_from_path(
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
    try:
        doc = fitz.open(file_path)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return None

        page = doc[page_num - 1]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        img_bytes = pix.tobytes("png")
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        doc.close()
        return img_base64

    except Exception as e:
        logger.warning("Failed to render full page image: %s", e)
        return None


def render_multipage_table(
    file_path: str,
    table_parts: List[Dict],
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
    try:
        from PIL import Image
    except ImportError:
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
                pix = page.get_pixmap(matrix=mat)
            else:
                if x1 > page_width * 1.1 or y1 > page_height * 1.1:
                    pymupdf_bbox = find_table_bbox_pymupdf(file_path, page_num, (x0, y0, x1, y1))
                    if pymupdf_bbox:
                        x0, y0, x1, y1 = pymupdf_bbox
                    else:
                        scale_x = page_width / max(x1, page_width)
                        scale_y = page_height / max(y1, page_height)
                        scale = min(scale_x, scale_y)
                        x0, y0, x1, y1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
                        expand = 20
                        x0 = max(0, x0 - expand)
                        y0 = max(0, y0 - expand)
                        x1 = min(page_width, x1 + expand)
                        y1 = min(page_height, y1 + expand)

                clip_rect = fitz.Rect(
                    max(0, x0 - padding),
                    max(0, y0 - padding),
                    min(page_width, x1 + padding),
                    min(page_height, y1 + padding),
                )
                if clip_rect.width <= 0 or clip_rect.height <= 0:
                    pix = page.get_pixmap(matrix=mat)
                else:
                    pix = page.get_pixmap(matrix=mat, clip=clip_rect)

            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)

        doc.close()

        if not images:
            return None

        total_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)

        stitched = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        y_offset = 0
        for img in images:
            x_offset = (total_width - img.width) // 2
            stitched.paste(img, (x_offset, y_offset))
            y_offset += img.height

        buffer = io.BytesIO()
        stitched.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return img_base64

    except Exception as e:
        logger.warning("Failed to render multi-page table: %s", e)
        return None


__all__ = [
    # Types
    "RenderConfig",
    "RenderedVisual",
    # Main functions
    "render_visual",
    "render_visual_region",
    "render_full_page",
    "render_multipage_visual",
    # Table detection
    "find_table_bbox_pymupdf",
    # File-path wrapper functions (for B03_table_extractor)
    "render_table_as_image",
    "render_full_page_from_path",
    "render_multipage_table",
    # Padding
    "expand_bbox_with_padding",
    "compute_caption_padding",
    # DPI
    "compute_adaptive_dpi",
    # Coordinate conversion
    "pts_to_pixels",
    "pixels_to_pts",
    "bbox_pts_to_pixels",
    # Utilities
    "get_image_dimensions_from_base64",
    "resize_image_for_vlm",
]
