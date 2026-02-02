# corpus_metadata/B_parsing/B09_pdf_native_figures.py
"""
Native PDF figure extraction using PyMuPDF for raster and vector graphics.

This module extracts figures directly from PDF using PyMuPDF primitives rather
than layout model detection. It handles raster figures (embedded images via
XObjects) and vector figures (drawings/plots via path detection), with noise
filtering for logos/headers/footers and deduplication via content hash.

Key Components:
    - EmbeddedFigure: Raster figure with page number, bbox, xref, and hash
    - VectorFigure: Vector plot with drawing count and axis text detection
    - extract_embedded_figures: Extract raster figures from PDF
    - detect_vector_figures: Detect vector plots using drawing primitives
    - filter_noise_images: Remove logos, headers, footers by repetition/size
    - detect_all_figures: Main entry point for both raster and vector extraction
    - render_figure_by_xref: Lazy render image with colorspace normalization
    - render_vector_figure: Export vector figure by rendering clipped region

Example:
    >>> from B_parsing.B09_pdf_native_figures import detect_all_figures
    >>> raster, vector = detect_all_figures("paper.pdf", min_area_ratio=0.03)
    >>> print(f"Found {len(raster)} raster and {len(vector)} vector figures")

Dependencies:
    - fitz (PyMuPDF): PDF rendering and extraction
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedFigure:
    """Raster figure extracted from PDF image XObjects."""

    page_num: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    xref: int  # Store xref, not Pixmap (lazy render later)
    image_hash: str  # sha1 of image bytes for dedup


@dataclass
class VectorFigure:
    """Vector-only figure detected from drawing primitives."""

    page_num: int
    bbox: Tuple[float, float, float, float]
    drawing_count: int  # Number of paths/lines in region
    has_axis_text: bool  # "months", "survival", tick labels detected


def extract_embedded_figures(
    pdf_path: str,
    min_area_ratio: float = 0.03,
) -> List[EmbeddedFigure]:
    """
    Extract raster figures from PDF image XObjects.

    Args:
        pdf_path: Path to PDF file
        min_area_ratio: Minimum image area as fraction of page area (default 3%)

    Returns:
        List of EmbeddedFigure objects
    """
    doc = fitz.open(pdf_path)
    figures: List[EmbeddedFigure] = []

    try:
        for page_num in range(1, doc.page_count + 1):
            page = doc[page_num - 1]
            page_area = page.rect.get_area()

            if page_area <= 0:
                continue

            for img in page.get_images(full=True):
                xref = img[0]

                # Get all rectangles where this image appears on the page
                try:
                    rects = page.get_image_rects(xref)
                except Exception as e:
                    logger.debug("Failed to get image rects for xref=%d: %s", xref, e)
                    continue

                for rect in rects:
                    area = rect.get_area()
                    if area <= 0 or area / page_area < min_area_ratio:
                        continue

                    # Get image hash for dedup (not Pixmap - too memory heavy)
                    try:
                        pix = fitz.Pixmap(doc, xref)
                        image_hash = hashlib.sha1(pix.tobytes()).hexdigest()[:12]
                        pix = None  # Release immediately
                    except Exception as e:
                        # Fallback: use xref as hash
                        logger.debug("Failed to create pixmap for xref=%d, using fallback hash: %s", xref, e)
                        image_hash = f"xref_{xref}"

                    figures.append(
                        EmbeddedFigure(
                            page_num=page_num,
                            bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                            xref=xref,
                            image_hash=image_hash,
                        )
                    )
    finally:
        doc.close()

    return figures


def extract_embedded_figures_from_doc(
    doc: fitz.Document,
    min_area_ratio: float = 0.03,
) -> List[EmbeddedFigure]:
    """
    Extract raster figures from an already-open document.

    Args:
        doc: Open PyMuPDF document
        min_area_ratio: Minimum image area as fraction of page area

    Returns:
        List of EmbeddedFigure objects
    """
    figures: List[EmbeddedFigure] = []

    for page_num in range(1, doc.page_count + 1):
        page = doc[page_num - 1]
        page_area = page.rect.get_area()

        if page_area <= 0:
            continue

        for img in page.get_images(full=True):
            xref = img[0]

            try:
                rects = page.get_image_rects(xref)
            except Exception as e:
                logger.debug("Failed to get image rects for xref=%d: %s", xref, e)
                continue

            for rect in rects:
                area = rect.get_area()
                if area <= 0 or area / page_area < min_area_ratio:
                    continue

                try:
                    pix = fitz.Pixmap(doc, xref)
                    image_hash = hashlib.sha1(pix.tobytes()).hexdigest()[:12]
                    pix = None
                except Exception as e:
                    logger.debug("Failed to create pixmap for xref=%d, using fallback hash: %s", xref, e)
                    image_hash = f"xref_{xref}"

                figures.append(
                    EmbeddedFigure(
                        page_num=page_num,
                        bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                        xref=xref,
                        image_hash=image_hash,
                    )
                )

    return figures


def render_figure_by_xref(doc: fitz.Document, xref: int) -> bytes:
    """
    Lazy render image with colorspace normalization.

    Args:
        doc: Open PyMuPDF document
        xref: Image XObject reference

    Returns:
        PNG bytes of the image
    """
    pix = fitz.Pixmap(doc, xref)

    # Handle CMYK/alpha - convert to RGB
    if pix.colorspace and pix.colorspace.n > 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)  # Remove alpha

    return pix.tobytes("png")


def detect_vector_figures(
    doc: fitz.Document,
    page_num: int,
    min_drawing_count: int = 20,
    dense_drawing_threshold: int = 50,
    min_height: float = 80,
    min_area_ratio: float = 0.02,
    header_zone_ratio: float = 0.12,
) -> List[VectorFigure]:
    """
    Detect vector-only plots using drawing primitives + axis text.

    Heuristic: dense linework regions + axis-like text nearby.
    This is critical for detecting Kaplan-Meier survival plots which
    are often rendered as vector graphics without embedded images.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        min_drawing_count: Minimum drawings to consider a region (default 20)
        dense_drawing_threshold: Threshold for "dense" region (default 50)
        min_height: Minimum height in points to be a figure (default 80)
        min_area_ratio: Minimum area as fraction of page (default 2%)
        header_zone_ratio: Top zone to exclude as header (default 12%)

    Returns:
        List of VectorFigure objects
    """
    page = doc[page_num - 1]
    page_height = page.rect.height
    page_area = page.rect.get_area()
    drawings = page.get_drawings()
    text_dict = page.get_text("dict")

    if not drawings or page_area <= 0:
        return []

    # Cluster drawings into regions
    regions = cluster_drawings_into_regions(drawings, min_gap=30)

    vector_figures: List[VectorFigure] = []
    for region_bbox, drawing_count in regions:
        if drawing_count < min_drawing_count:
            continue

        x0, y0, x1, y1 = region_bbox
        region_height = y1 - y0
        region_area = (x1 - x0) * region_height

        # Filter out narrow regions (likely headers/footers/decorations)
        if region_height < min_height:
            continue

        # Filter out small regions
        if region_area / page_area < min_area_ratio:
            continue

        # Filter out regions entirely in header zone (top of page)
        if y1 < page_height * header_zone_ratio:
            continue

        # Check for axis-like text nearby
        has_axis_text = check_axis_text_nearby(
            region_bbox,
            text_dict,
            keywords=["month", "week", "year", "day", "survival", "probability", "%", "time", "events", "risk"],
        )

        # Require either dense drawings OR axis text
        if drawing_count >= dense_drawing_threshold or has_axis_text:
            vector_figures.append(
                VectorFigure(
                    page_num=page_num,
                    bbox=region_bbox,
                    drawing_count=drawing_count,
                    has_axis_text=has_axis_text,
                )
            )

    return vector_figures


def cluster_drawings_into_regions(
    drawings: List[dict],
    min_gap: float = 30,
) -> List[Tuple[Tuple[float, float, float, float], int]]:
    """
    Group nearby drawings into regions using a simple grid-based clustering.

    Args:
        drawings: List of drawing dicts from page.get_drawings()
        min_gap: Minimum gap between regions to consider separate

    Returns:
        List of (bbox, drawing_count) tuples
    """
    # Extract all drawing bboxes
    bboxes: List[Tuple[float, float, float, float]] = []
    for d in drawings:
        rect = d.get("rect")
        if rect is not None:
            try:
                # Handle fitz.Rect or tuple
                if hasattr(rect, "x0"):
                    bboxes.append((rect.x0, rect.y0, rect.x1, rect.y1))
                elif len(rect) >= 4:
                    bboxes.append((float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])))
            except (TypeError, IndexError) as e:
                logger.debug("Failed to parse drawing rect: %s", e)
                continue

    if not bboxes:
        return []

    # Simple clustering: merge overlapping/adjacent bboxes
    clusters: List[List[Tuple[float, float, float, float]]] = []

    for bbox in bboxes:
        merged = False
        for cluster in clusters:
            # Check if bbox overlaps or is close to any bbox in cluster
            for cb in cluster:
                if _bboxes_close(bbox, cb, min_gap):
                    cluster.append(bbox)
                    merged = True
                    break
            if merged:
                break

        if not merged:
            clusters.append([bbox])

    # Merge clusters that became connected
    # Simple approach: iterate until no more merges
    changed = True
    while changed:
        changed = False
        new_clusters: List[List[Tuple[float, float, float, float]]] = []
        used = set()

        for i, c1 in enumerate(clusters):
            if i in used:
                continue

            merged_cluster = list(c1)
            for j, c2 in enumerate(clusters[i + 1 :], start=i + 1):
                if j in used:
                    continue

                # Check if any bbox in c1 is close to any in c2
                should_merge = False
                for b1 in c1:
                    for b2 in c2:
                        if _bboxes_close(b1, b2, min_gap):
                            should_merge = True
                            break
                    if should_merge:
                        break

                if should_merge:
                    merged_cluster.extend(c2)
                    used.add(j)
                    changed = True

            new_clusters.append(merged_cluster)
            used.add(i)

        clusters = new_clusters

    # Convert clusters to (bbox, count) tuples
    results: List[Tuple[Tuple[float, float, float, float], int]] = []
    for cluster in clusters:
        if not cluster:
            continue

        # Compute bounding box of cluster
        x0 = min(b[0] for b in cluster)
        y0 = min(b[1] for b in cluster)
        x1 = max(b[2] for b in cluster)
        y1 = max(b[3] for b in cluster)

        results.append(((x0, y0, x1, y1), len(cluster)))

    return results


def _bboxes_close(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
    gap: float,
) -> bool:
    """Check if two bboxes overlap or are within gap distance."""
    x1_0, y1_0, x1_1, y1_1 = b1
    x2_0, y2_0, x2_1, y2_1 = b2

    # Check horizontal proximity
    h_close = not (x1_1 + gap < x2_0 or x2_1 + gap < x1_0)
    # Check vertical proximity
    v_close = not (y1_1 + gap < y2_0 or y2_1 + gap < y1_0)

    return h_close and v_close


def check_axis_text_nearby(
    bbox: Tuple[float, float, float, float],
    text_dict: dict,
    keywords: List[str],
    margin: float = 50,
) -> bool:
    """
    Check if axis-like text exists within margin of bbox.

    Args:
        bbox: Region bounding box
        text_dict: Text dict from page.get_text("dict")
        keywords: Keywords indicating axis labels
        margin: Base search margin around bbox (adaptive based on figure size)

    Returns:
        True if axis-like text found nearby
    """
    x0, y0, x1, y1 = bbox

    # Adaptive margin: larger figures get larger search margins
    figure_height = y1 - y0
    figure_width = x1 - x0
    adaptive_margin = max(margin, figure_height * 0.15, figure_width * 0.1)

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # Only text blocks
            continue

        bx0, by0, bx1, by1 = block.get("bbox", (0, 0, 0, 0))

        # Check if text block is near the figure region
        if bx1 < x0 - adaptive_margin or bx0 > x1 + adaptive_margin:
            continue
        if by1 < y0 - adaptive_margin or by0 > y1 + adaptive_margin:
            continue

        # Extract text from block
        text = ""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text += " " + span.get("text", "")

        text_lower = text.lower()

        # Check for axis keywords
        if any(kw in text_lower for kw in keywords):
            return True

        # Check for numeric tick labels (common in axes)
        import re
        if re.search(r"\b\d+\.?\d*\s*%?\b", text):
            return True

    return False


def render_vector_figure(
    doc: fitz.Document,
    page_num: int,
    bbox: Tuple[float, float, float, float],
    dpi: int = 200,
    padding: float = 10,
) -> bytes:
    """
    Export vector figure by rendering clipped region.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        bbox: Region bounding box
        dpi: Render resolution
        padding: Extra padding around bbox

    Returns:
        PNG bytes of the rendered region
    """
    page = doc[page_num - 1]

    # Add padding
    x0, y0, x1, y1 = bbox
    clip_rect = fitz.Rect(
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(page.rect.width, x1 + padding),
        min(page.rect.height, y1 + padding),
    )

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip_rect)
    return pix.tobytes("png")


def is_text_heavy_region(
    doc: fitz.Document,
    page_num: int,
    bbox: Tuple[float, float, float, float],
    text_density_threshold: float = 0.6,
    min_text_blocks: int = 10,
) -> bool:
    """
    Check if a region is primarily text (not a figure).

    This helps filter false positive figure extractions where a text-heavy
    page or region was mistakenly identified as a figure.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        bbox: Region bounding box (x0, y0, x1, y1)
        text_density_threshold: Text area / region area threshold (default 0.6)
        min_text_blocks: Minimum text blocks to trigger text-heavy check

    Returns:
        True if the region appears to be primarily text
    """
    if page_num < 1 or page_num > doc.page_count:
        return False

    page = doc[page_num - 1]
    x0, y0, x1, y1 = bbox
    clip_rect = fitz.Rect(x0, y0, x1, y1)
    region_area = clip_rect.get_area()

    if region_area <= 0:
        return False

    # Get text blocks in the region
    text_dict = page.get_text("dict", clip=clip_rect)
    blocks = text_dict.get("blocks", [])

    # Count text blocks and calculate text coverage
    text_blocks = [b for b in blocks if b.get("type") == 0]  # type 0 = text

    if len(text_blocks) < min_text_blocks:
        return False

    # Calculate total text area
    text_area = 0
    for block in text_blocks:
        block_bbox = block.get("bbox", (0, 0, 0, 0))
        block_area = (block_bbox[2] - block_bbox[0]) * (block_bbox[3] - block_bbox[1])
        text_area += block_area

    text_density = text_area / region_area if region_area > 0 else 0

    # Also check character count - text pages have lots of characters
    char_count = sum(
        len(span.get("text", ""))
        for block in text_blocks
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    )

    # High text density or many characters indicates text page
    is_text_heavy = (
        text_density > text_density_threshold or
        char_count > 500  # More than 500 chars is likely a text page
    )

    return is_text_heavy


def filter_noise_images(
    figures: List[EmbeddedFigure],
    doc: fitz.Document,
    repeat_threshold: int = 3,
    small_area_ratio: float = 0.03,
    top_margin_ratio: float = 0.10,
    filter_text_heavy: bool = True,
) -> List[EmbeddedFigure]:
    """
    Filter logos/headers by:
    - Size (< 3% page area)
    - Position (top 10% of page)
    - Repetition (same image_hash OR xref across 3+ pages)

    Args:
        figures: List of extracted figures
        doc: Open PyMuPDF document
        repeat_threshold: Min occurrences to consider repeated (default 3)
        small_area_ratio: Size threshold for "small" images
        top_margin_ratio: Top zone threshold

    Returns:
        Filtered list of figures
    """
    # Count unique pages per hash/xref (for header/footer/logo detection)
    hash_pages: Dict[str, set] = {}
    xref_pages: Dict[int, set] = {}
    for fig in figures:
        hash_pages.setdefault(fig.image_hash, set()).add(fig.page_num)
        xref_pages.setdefault(fig.xref, set()).add(fig.page_num)

    filtered: List[EmbeddedFigure] = []
    for fig in figures:
        # Skip if same image appears on 3+ different pages (header/footer/logo)
        if len(hash_pages.get(fig.image_hash, set())) >= repeat_threshold:
            continue
        if len(xref_pages.get(fig.xref, set())) >= repeat_threshold:
            continue

        # Check size and position
        page = doc[fig.page_num - 1]
        page_area = page.rect.get_area()
        page_height = page.rect.height

        if page_area <= 0 or page_height <= 0:
            filtered.append(fig)
            continue

        # Calculate image area
        x0, y0, x1, y1 = fig.bbox
        img_area = (x1 - x0) * (y1 - y0)
        area_ratio = img_area / page_area

        # Calculate position (is it in top margin?)
        top_y = fig.bbox[1]
        in_top_margin = top_y < page_height * top_margin_ratio

        # Skip small images in top margin (likely logos)
        if area_ratio < small_area_ratio and in_top_margin:
            continue

        # Skip text-heavy regions (false positive figure extractions)
        if filter_text_heavy and is_text_heavy_region(doc, fig.page_num, fig.bbox):
            continue

        filtered.append(fig)

    return filtered


def detect_all_figures(
    pdf_path: str,
    min_area_ratio: float = 0.03,
    filter_noise: bool = True,
) -> Tuple[List[EmbeddedFigure], List[VectorFigure]]:
    """
    Main entry point: extract both raster and vector figures from PDF.

    Args:
        pdf_path: Path to PDF file
        min_area_ratio: Minimum image area as fraction of page
        filter_noise: Whether to filter noise images

    Returns:
        Tuple of (raster_figures, vector_figures)
    """
    doc = fitz.open(pdf_path)

    try:
        # Extract raster figures
        raster_figures = extract_embedded_figures_from_doc(doc, min_area_ratio)

        # Filter noise
        if filter_noise:
            raster_figures = filter_noise_images(raster_figures, doc)

        # Detect vector figures on each page
        vector_figures: List[VectorFigure] = []
        for page_num in range(1, doc.page_count + 1):
            page_vectors = detect_vector_figures(doc, page_num)
            vector_figures.extend(page_vectors)

        return raster_figures, vector_figures

    finally:
        doc.close()


def extract_text_from_region(
    doc: fitz.Document,
    page_num: int,
    bbox: Tuple[float, float, float, float],
    margin: float = 10,
) -> str:
    """
    Extract embedded text from a figure region using PyMuPDF.

    This provides OCR fallback when Vision LLM fails (e.g., image too large).
    For raster figures, this extracts any text overlays or nearby captions.
    For vector figures, this extracts axis labels, legends, and data annotations.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        bbox: Region bounding box (x0, y0, x1, y1)
        margin: Extra margin around bbox to capture captions (default 10pt)

    Returns:
        Extracted text from the region, or empty string if none found
    """
    if page_num < 1 or page_num > doc.page_count:
        return ""

    page = doc[page_num - 1]
    page_width = page.rect.width
    page_height = page.rect.height

    # Expand bbox with margin
    x0, y0, x1, y1 = bbox
    clip_rect = fitz.Rect(
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(page_width, x1 + margin),
        min(page_height, y1 + margin * 3),  # Extra margin below for captions
    )

    # Extract text from the clipped region
    try:
        text = page.get_text("text", clip=clip_rect)
        return text.strip() if text else ""
    except Exception as e:
        logger.debug("Failed to extract text from region on page %d: %s", page_num, e)
        return ""


def extract_text_from_figure_xref(
    doc: fitz.Document,
    page_num: int,
    xref: int,
) -> str:
    """
    Extract embedded text from around a figure identified by xref.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        xref: Image XObject reference

    Returns:
        Extracted text from the figure region
    """
    if page_num < 1 or page_num > doc.page_count:
        return ""

    page = doc[page_num - 1]

    # Get the image rectangle
    try:
        rects = page.get_image_rects(xref)
        if not rects:
            return ""

        # Use the first (usually only) rectangle
        rect = rects[0]
        return extract_text_from_region(
            doc, page_num, (rect.x0, rect.y0, rect.x1, rect.y1)
        )
    except Exception as e:
        logger.debug("Failed to extract text for xref=%d on page %d: %s", xref, page_num, e)
        return ""


__all__ = [
    "EmbeddedFigure",
    "VectorFigure",
    "extract_embedded_figures",
    "extract_embedded_figures_from_doc",
    "render_figure_by_xref",
    "detect_vector_figures",
    "cluster_drawings_into_regions",
    "check_axis_text_nearby",
    "render_vector_figure",
    "is_text_heavy_region",
    "filter_noise_images",
    "detect_all_figures",
    "extract_text_from_region",
    "extract_text_from_figure_xref",
]
