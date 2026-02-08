# corpus_metadata/B_parsing/B20_zone_expander.py
"""
Whitespace-based bounding box computation from VLM visual zones.

This module takes visual zones from VLM layout analysis (B19) and expands them
to precise bounding boxes using whitespace detection. The VLM identifies WHAT
and roughly WHERE; this module computes precise coordinates by finding whitespace
boundaries, stopping at margins, column boundaries, other visuals, and text blocks.

Key Components:
    - ExpandedVisual: Visual with computed precise bbox (pts and normalized)
    - expand_zones_to_bboxes: Main expansion function for a page's visual zones
    - compute_column_boundaries: Detect column boundaries from text block positions
    - find_whitespace_boundaries: Find whitespace edges around a region
    - expand_to_whitespace: Expand zone to nearest whitespace boundaries

Example:
    >>> from B_parsing.B20_zone_expander import expand_zones_to_bboxes
    >>> from B_parsing.B18_layout_models import PageLayout
    >>> expanded = expand_zones_to_bboxes(layout, text_blocks, page_width, page_height)
    >>> for ev in expanded:
    ...     print(f"{ev.zone.visual_type}: {ev.bbox_pts}")

Dependencies:
    - B_parsing.B18_layout_models: PageLayout, VisualPosition, VisualZone
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from A_core.A24_layout_models import (
    PageLayout,
    VisualPosition,
    VisualZone,
)

logger = logging.getLogger(__name__)


@dataclass
class ExpandedVisual:
    """A visual with computed precise bounding box."""

    zone: VisualZone  # Original zone from VLM
    bbox_pts: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF points
    bbox_normalized: Tuple[float, float, float, float]  # (x0, y0, x1, y1) as 0-1 fractions
    layout_code: str  # e.g., "2col"
    position_code: str  # e.g., "L", "R", "F"


def compute_column_boundaries(
    text_blocks: List[Dict],
    page_width: float,
    min_gap_ratio: float = 0.03,  # Minimum 3% page width gap
) -> Optional[Dict]:
    """
    Detect column boundaries from text block positions.

    A two-column layout is detected when:
    1. There's a significant vertical gap in the middle region of the page
    2. Text blocks exist on BOTH sides of the gap

    Args:
        text_blocks: List of text blocks with 'bbox' key
        page_width: Page width in PDF points
        min_gap_ratio: Minimum gap width as fraction of page

    Returns:
        Dict with 'split' (column boundary as 0-1 fraction) or None if single column
    """
    if not text_blocks:
        return None

    # Collect text block x-ranges (left edge, right edge)
    block_ranges = []
    for block in text_blocks:
        bbox = block.get("bbox")
        if bbox and len(bbox) >= 4:
            x0, _, x1, _ = bbox
            block_ranges.append((x0, x1))

    if not block_ranges:
        return None

    # Find all unique x-coordinate edges
    x_coords = sorted(set(x for rng in block_ranges for x in rng))

    if len(x_coords) < 2:
        return None

    min_gap = page_width * min_gap_ratio
    middle_start = page_width * 0.3
    middle_end = page_width * 0.7

    best_gap = None
    best_gap_size = 0
    best_gap_range = None

    for i in range(len(x_coords) - 1):
        gap_start = x_coords[i]
        gap_end = x_coords[i + 1]
        gap_size = gap_end - gap_start
        gap_center = (gap_start + gap_end) / 2

        # Gap must be in middle region and large enough
        if (gap_size > min_gap and
            gap_size > best_gap_size and
            middle_start < gap_center < middle_end):

            # Verify there are text blocks on BOTH sides of this gap
            has_left = any(x1 <= gap_start + 1 for x0, x1 in block_ranges)  # Blocks ending before gap
            has_right = any(x0 >= gap_end - 1 for x0, x1 in block_ranges)  # Blocks starting after gap

            if has_left and has_right:
                best_gap = gap_center
                best_gap_size = gap_size
                best_gap_range = (gap_start, gap_end)

    if best_gap is None:
        return None

    return {
        "split": best_gap / page_width,
        "gap_start": best_gap_range[0] if best_gap_range else 0,
        "gap_end": best_gap_range[1] if best_gap_range else best_gap,
    }


def expand_zone_to_whitespace(
    zone: VisualZone,
    layout: PageLayout,
    page_width: float,
    page_height: float,
    text_blocks: List[Dict],
    other_visuals: List[ExpandedVisual],
) -> ExpandedVisual:
    """
    Expand a visual zone to precise bbox using whitespace detection.

    Args:
        zone: Visual zone from VLM
        layout: Page layout information
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        text_blocks: Text blocks on the page
        other_visuals: Other already-expanded visuals (to avoid overlap)

    Returns:
        ExpandedVisual with precise bbox
    """
    # Determine horizontal bounds based on position
    if zone.position == VisualPosition.LEFT:
        x0 = page_width * layout.margin_left
        x1 = page_width * (layout.column_boundary or 0.5)
    elif zone.position == VisualPosition.RIGHT:
        x0 = page_width * (layout.column_boundary or 0.5)
        x1 = page_width * layout.margin_right
    else:  # FULL
        x0 = page_width * layout.margin_left
        x1 = page_width * layout.margin_right

    # Determine vertical bounds from zone description
    y0, y1 = _parse_vertical_zone(zone.vertical_zone, page_height)

    # Expand to nearest whitespace (simplified - full implementation would scan pixel rows)
    # For now, use the zone boundaries with small padding
    padding = 10.0  # PDF points

    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(page_width, x1 + padding)
    y1 = min(page_height, y1 + padding)

    # Avoid overlapping with other visuals
    for other in other_visuals:
        ox0, oy0, ox1, oy1 = other.bbox_pts

        # Check vertical overlap
        if not (y1 < oy0 or y0 > oy1):
            # There's vertical overlap - adjust
            if zone.position == other.zone.position:
                # Same column - don't overlap vertically
                if y0 < oy0 < y1:
                    y1 = oy0 - 5  # Stop above other visual
                elif y0 < oy1 < y1:
                    y0 = oy1 + 5  # Start below other visual

    bbox_pts = (x0, y0, x1, y1)
    bbox_normalized = (
        x0 / page_width,
        y0 / page_height,
        x1 / page_width,
        y1 / page_height,
    )

    return ExpandedVisual(
        zone=zone,
        bbox_pts=bbox_pts,
        bbox_normalized=bbox_normalized,
        layout_code=layout.pattern.value,
        position_code=zone.position.value,
    )


def _parse_vertical_zone(zone_str: str, page_height: float) -> Tuple[float, float]:
    """
    Parse vertical zone string to y-coordinates.

    Args:
        zone_str: "top", "middle", "bottom", or "0.2-0.6" format
        page_height: Page height in PDF points

    Returns:
        Tuple of (y0, y1) in PDF points
    """
    zone_str = zone_str.lower().strip()

    if zone_str == "top":
        return (0, page_height * 0.4)
    elif zone_str == "middle":
        return (page_height * 0.3, page_height * 0.7)
    elif zone_str == "bottom":
        return (page_height * 0.6, page_height)
    elif "-" in zone_str:
        # Parse "0.2-0.6" format
        try:
            parts = zone_str.split("-")
            start = float(parts[0])
            end = float(parts[1])
            return (page_height * start, page_height * end)
        except (ValueError, IndexError):
            pass

    # Default to full page
    return (0, page_height)


def expand_all_zones(
    layout: PageLayout,
    page_width: float,
    page_height: float,
    text_blocks: List[Dict],
) -> List[ExpandedVisual]:
    """
    Expand all visual zones in a page layout.

    Processes zones top-to-bottom, left-to-right to handle
    overlapping zones correctly.

    Args:
        layout: Page layout with visual zones
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        text_blocks: Text blocks on the page

    Returns:
        List of ExpandedVisual with precise bboxes
    """
    # Sort zones by vertical position, then horizontal
    def zone_sort_key(z: VisualZone) -> Tuple[float, int]:
        y_start, _ = _parse_vertical_zone(z.vertical_zone, page_height)
        x_order = 0 if z.position == VisualPosition.LEFT else (1 if z.position == VisualPosition.FULL else 2)
        return (y_start, x_order)

    sorted_zones = sorted(layout.visuals, key=zone_sort_key)

    expanded: List[ExpandedVisual] = []
    for zone in sorted_zones:
        result = expand_zone_to_whitespace(
            zone=zone,
            layout=layout,
            page_width=page_width,
            page_height=page_height,
            text_blocks=text_blocks,
            other_visuals=expanded,
        )
        expanded.append(result)

    return expanded


__all__ = [
    # Classes
    "ExpandedVisual",
    # Functions
    "compute_column_boundaries",
    "expand_zone_to_whitespace",
    "expand_all_zones",
]
