# corpus_metadata/B_parsing/B10_caption_detector.py
"""
Caption Detection and Column Layout Analysis

Detects Figure X and Table X captions with their positions,
infers page column layout, and links captions to figure regions.

Key capabilities:
- Caption detection via regex patterns
- Column layout inference from text distribution
- Caption-to-figure linking (nearest region above caption)
- Column-aware table region definition
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF

from B_parsing.B09_pdf_native_figures import EmbeddedFigure, VectorFigure


@dataclass
class Caption:
    """Detected figure or table caption."""

    text: str
    caption_type: str  # "figure" or "table"
    number: int  # Figure 1 -> 1
    bbox: Tuple[float, float, float, float]
    page_num: int
    column_idx: int  # Which column this caption is in


@dataclass
class TableRegion:
    """Defined region for a table based on caption and column bounds."""

    caption: Caption
    bbox: Tuple[float, float, float, float]
    page_num: int


# Caption patterns
FIGURE_CAPTION_RE = re.compile(
    r"^(?:Fig(?:ure)?\.?\s*)(\d+)\b",
    re.IGNORECASE,
)
TABLE_CAPTION_RE = re.compile(
    r"^(?:Table\s*)(\d+)\b",
    re.IGNORECASE,
)


def detect_all_captions(
    doc: fitz.Document,
    page_num: int,
    columns: Optional[List[Tuple[float, float]]] = None,
) -> List[Caption]:
    """
    Find all Figure X and Table X captions with their positions.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        columns: Optional pre-computed column boundaries

    Returns:
        List of Caption objects
    """
    page = doc[page_num - 1]
    text_dict = page.get_text("dict")

    # Infer columns if not provided
    if columns is None:
        columns = infer_page_columns(doc, page_num)

    captions: List[Caption] = []

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # Only text blocks
            continue

        for line in block.get("lines", []):
            # Combine all spans in line
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            text = text.strip()

            if not text:
                continue

            bbox = line.get("bbox", (0, 0, 0, 0))

            # Figure caption
            fig_match = FIGURE_CAPTION_RE.match(text)
            if fig_match:
                col_idx = get_column_index(bbox[0], columns)
                captions.append(
                    Caption(
                        text=text,
                        caption_type="figure",
                        number=int(fig_match.group(1)),
                        bbox=bbox,
                        page_num=page_num,
                        column_idx=col_idx,
                    )
                )
                continue

            # Table caption
            table_match = TABLE_CAPTION_RE.match(text)
            if table_match:
                col_idx = get_column_index(bbox[0], columns)
                captions.append(
                    Caption(
                        text=text,
                        caption_type="table",
                        number=int(table_match.group(1)),
                        bbox=bbox,
                        page_num=page_num,
                        column_idx=col_idx,
                    )
                )

    return captions


def infer_page_columns(
    doc: fitz.Document,
    page_num: int,
    gap_threshold_ratio: float = 0.08,
) -> List[Tuple[float, float]]:
    """
    Infer column boundaries from text block x-distribution.

    Uses the gap between text blocks to detect column gutters.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        gap_threshold_ratio: Gap size as fraction of page width to detect column boundary

    Returns:
        List of (col_x0, col_x1) tuples
    """
    page = doc[page_num - 1]
    text_dict = page.get_text("dict")
    page_width = page.rect.width

    # Collect x-centers of narrow text blocks (< 45% page width)
    x_coords: List[Tuple[float, float]] = []  # (x0, x1) of each block

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        bbox = block.get("bbox", (0, 0, 0, 0))
        x0, y0, x1, y1 = bbox
        width = x1 - x0

        # Only consider narrow blocks (likely column content)
        if width < page_width * 0.45:
            x_coords.append((x0, x1))

    if not x_coords:
        # Single column fallback
        return [(0, page_width)]

    # Sort by left edge
    x_coords.sort(key=lambda x: x[0])

    # Find gaps between blocks that indicate column boundaries
    # Use histogram-based approach: find x positions with low block density
    gap_threshold = page_width * gap_threshold_ratio

    # Collect all x-ranges covered by blocks
    all_x0 = [x[0] for x in x_coords]
    all_x1 = [x[1] for x in x_coords]

    # Find gaps in coverage
    # Sort all right edges and look for large gaps before next left edge
    edges = []
    for x0, x1 in x_coords:
        edges.append((x0, "start"))
        edges.append((x1, "end"))
    edges.sort(key=lambda e: e[0])

    # Track coverage and find gaps
    coverage = 0
    gap_starts = []
    gap_ends = []
    last_end = 0

    for x, edge_type in edges:
        if edge_type == "start":
            if coverage == 0 and x - last_end > gap_threshold:
                # Found a gap
                gap_starts.append(last_end)
                gap_ends.append(x)
            coverage += 1
        else:
            coverage -= 1
            if coverage == 0:
                last_end = x

    # Build column boundaries from gaps
    if not gap_starts:
        # Single column
        return [(0, page_width)]

    columns: List[Tuple[float, float]] = []
    prev_end = 0
    margin = 20  # Add small margin

    for gap_start, gap_end in zip(gap_starts, gap_ends):
        # Column from prev_end to gap_start
        if gap_start > prev_end + margin:
            columns.append((max(0, prev_end - margin), gap_start + margin))
        prev_end = gap_end

    # Final column from last gap to page width
    if prev_end < page_width - margin:
        columns.append((max(0, prev_end - margin), page_width))

    # If no columns detected, use simple left/right split
    if not columns:
        mid = page_width / 2
        return [(0, mid), (mid, page_width)]

    return columns


def get_column_index(x: float, columns: List[Tuple[float, float]]) -> int:
    """
    Get the column index for an x coordinate.

    Args:
        x: X coordinate to check
        columns: List of (col_x0, col_x1) tuples

    Returns:
        Column index (0-based)
    """
    for idx, (col_x0, col_x1) in enumerate(columns):
        if col_x0 <= x <= col_x1:
            return idx

    # Default to nearest column
    min_dist = float("inf")
    best_idx = 0
    for idx, (col_x0, col_x1) in enumerate(columns):
        center = (col_x0 + col_x1) / 2
        dist = abs(x - center)
        if dist < min_dist:
            min_dist = dist
            best_idx = idx

    return best_idx


def link_caption_to_figure(
    caption: Caption,
    raster_figures: List[EmbeddedFigure],
    vector_figures: List[VectorFigure],
    max_distance: float = 500,
    same_column_only: bool = False,
    columns: Optional[List[Tuple[float, float]]] = None,
) -> Optional[Union[EmbeddedFigure, VectorFigure]]:
    """
    Link caption to nearest figure region ABOVE the caption.

    This prevents grabbing tables/other content below the caption.

    Args:
        caption: Caption to link
        raster_figures: List of raster figures
        vector_figures: List of vector figures
        max_distance: Maximum vertical distance to search
        same_column_only: If True, only match figures in same column
        columns: Column boundaries (needed if same_column_only=True)

    Returns:
        Matched figure or None
    """
    cap_y = caption.bbox[1]  # Top of caption
    cap_x = (caption.bbox[0] + caption.bbox[2]) / 2  # Center x

    candidates: List[Tuple[float, str, Union[EmbeddedFigure, VectorFigure]]] = []

    # Check raster figures above caption
    for fig in raster_figures:
        if fig.page_num != caption.page_num:
            continue

        fig_bottom = fig.bbox[3]

        # Figure must be above caption
        if fig_bottom > cap_y:
            continue

        distance = cap_y - fig_bottom
        if distance > max_distance:
            continue

        # Check column constraint
        if same_column_only and columns:
            fig_x = (fig.bbox[0] + fig.bbox[2]) / 2
            fig_col = get_column_index(fig_x, columns)
            if fig_col != caption.column_idx:
                continue

        candidates.append((distance, "raster", fig))

    # Check vector figures above caption
    for fig in vector_figures:
        if fig.page_num != caption.page_num:
            continue

        fig_bottom = fig.bbox[3]

        if fig_bottom > cap_y:
            continue

        distance = cap_y - fig_bottom
        if distance > max_distance:
            continue

        if same_column_only and columns:
            fig_x = (fig.bbox[0] + fig.bbox[2]) / 2
            fig_col = get_column_index(fig_x, columns)
            if fig_col != caption.column_idx:
                continue

        candidates.append((distance, "vector", fig))

    if not candidates:
        return None

    # Return nearest figure above
    candidates.sort(key=lambda x: x[0])
    return candidates[0][2]


def get_table_region_column_aware(
    caption: Caption,
    columns: List[Tuple[float, float]],
    page: fitz.Page,
    next_element_y: Optional[float] = None,
    text_dict: Optional[Dict] = None,
) -> Tuple[float, float, float, float]:
    """
    Define table region locked to caption's column.

    Expands to full width only if table spans both columns.

    Args:
        caption: Table caption
        columns: Column boundaries
        page: PyMuPDF page object
        next_element_y: Y position of next caption/figure (if known)
        text_dict: Pre-computed text dict for span detection

    Returns:
        Table region bbox (x0, y0, x1, y1)
    """
    # Get column bounds for this caption
    if caption.column_idx < len(columns):
        cap_col = columns[caption.column_idx]
    else:
        cap_col = columns[0] if columns else (0, page.rect.width)

    col_x0, col_x1 = cap_col

    # Start below caption
    y_start = caption.bbox[3]  # Bottom of caption

    # End at next element or page bottom
    y_end = next_element_y if next_element_y else page.rect.height

    # Check if table content spans beyond column
    # (detect by looking for text/graphics in other columns within y range)
    spans_columns = False

    if text_dict is not None:
        for block in text_dict.get("blocks", []):
            block_bbox = block.get("bbox", (0, 0, 0, 0))
            bx0, by0, bx1, by1 = block_bbox

            # Check if block is in table region vertically
            if by0 < y_start or by1 > y_end:
                continue

            # Check if block extends significantly beyond column
            if bx0 < col_x0 - 20 or bx1 > col_x1 + 20:
                spans_columns = True
                break

    # If spanning, expand to full page width
    if spans_columns:
        margin = 36  # ~0.5 inch
        col_x0 = margin
        col_x1 = page.rect.width - margin

    return (col_x0, y_start, col_x1, min(y_end, page.rect.height - 50))


def detect_captions_all_pages(
    doc: fitz.Document,
) -> Tuple[List[Caption], Dict[int, List[Tuple[float, float]]]]:
    """
    Detect all captions across all pages.

    Args:
        doc: Open PyMuPDF document

    Returns:
        Tuple of (all_captions, columns_by_page)
    """
    all_captions: List[Caption] = []
    columns_by_page: Dict[int, List[Tuple[float, float]]] = {}

    for page_num in range(1, doc.page_count + 1):
        columns = infer_page_columns(doc, page_num)
        columns_by_page[page_num] = columns

        page_captions = detect_all_captions(doc, page_num, columns)
        all_captions.extend(page_captions)

    return all_captions, columns_by_page


def get_figure_captions(captions: List[Caption]) -> List[Caption]:
    """Filter to figure captions only."""
    return [c for c in captions if c.caption_type == "figure"]


def get_table_captions(captions: List[Caption]) -> List[Caption]:
    """Filter to table captions only."""
    return [c for c in captions if c.caption_type == "table"]


def find_next_caption_y(
    caption: Caption,
    all_captions: List[Caption],
) -> Optional[float]:
    """
    Find the Y position of the next caption on the same page.

    Args:
        caption: Current caption
        all_captions: All captions

    Returns:
        Y position of next caption, or None if none found
    """
    same_page = [c for c in all_captions if c.page_num == caption.page_num]

    # Sort by y position
    same_page.sort(key=lambda c: c.bbox[1])

    # Find current caption and get next
    for i, c in enumerate(same_page):
        if c.bbox == caption.bbox and c.caption_type == caption.caption_type:
            if i + 1 < len(same_page):
                return same_page[i + 1].bbox[1]
            break

    return None


__all__ = [
    "Caption",
    "TableRegion",
    "detect_all_captions",
    "infer_page_columns",
    "get_column_index",
    "link_caption_to_figure",
    "get_table_region_column_aware",
    "detect_captions_all_pages",
    "get_figure_captions",
    "get_table_captions",
    "find_next_caption_y",
]
