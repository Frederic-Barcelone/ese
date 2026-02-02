# corpus_metadata/B_parsing/B25_legacy_ordering.py
"""
Legacy block ordering for single and two-column PDF layouts.

Provides:
- Two-column layout detection
- Single-column ordering by Y position
- Legacy deterministic block ordering

NOTE: This is the fallback ordering when use_sota_layout=False.
For correct multi-column reading order, use B04_column_ordering (SOTA).

Extracted from B01_pdf_to_docgraph.py to reduce file size.
"""

from __future__ import annotations

from typing import Any, Dict, List


def is_two_column_page(
    raw_blocks: List[Dict[str, Any]],
    page_w: float,
    two_col_min_side_blocks: int = 6,
) -> bool:
    """
    Legacy two-column detection.

    Args:
        raw_blocks: List of block dicts with 'bbox' key
        page_w: Page width
        two_col_min_side_blocks: Minimum blocks on each side to detect two columns

    Returns:
        True if page appears to have two-column layout
    """
    if not raw_blocks or page_w <= 0:
        return False

    xs = []
    for rb in raw_blocks:
        x0, _, x1, _ = rb["bbox"].coords
        xc = (x0 + x1) / 2.0
        xs.append(xc)

    left = sum(1 for x in xs if x < page_w * 0.45)
    right = sum(1 for x in xs if x > page_w * 0.55)
    mid = sum(1 for x in xs if page_w * 0.45 <= x <= page_w * 0.55)

    return (
        left >= two_col_min_side_blocks
        and right >= two_col_min_side_blocks
        and mid <= max(2, int(0.08 * len(xs)))
    )


def order_single_column(
    items: List[Dict[str, Any]],
    y_tolerance: float = 3.0,
) -> List[Dict[str, Any]]:
    """
    Legacy single-column ordering by Y position.

    Groups blocks on same horizontal line (within y_tolerance),
    then orders left-to-right within each line.

    Args:
        items: List of block dicts with 'x0', 'y0' keys
        y_tolerance: Maximum Y difference to consider blocks on same line

    Returns:
        Ordered list of blocks
    """
    items = sorted(items, key=lambda r: (r["y0"], r["x0"]))

    ordered: List[Dict[str, Any]] = []
    current_line: List[Dict[str, Any]] = []
    current_y = None

    for it in items:
        if current_y is None:
            current_y = it["y0"]
            current_line = [it]
            continue

        if abs(it["y0"] - current_y) <= y_tolerance:
            current_line.append(it)
        else:
            current_line.sort(key=lambda r: r["x0"])
            ordered.extend(current_line)
            current_y = it["y0"]
            current_line = [it]

    if current_line:
        current_line.sort(key=lambda r: r["x0"])
        ordered.extend(current_line)

    return ordered


def order_blocks_deterministically(
    raw_blocks: List[Dict[str, Any]],
    page_w: float,
    two_col_min_side_blocks: int = 6,
    y_tolerance: float = 3.0,
) -> List[Dict[str, Any]]:
    """
    Legacy block ordering for single or two-column layouts.

    NOTE: This method has a known issue with two-column layouts:
    it reads ALL left column blocks, then ALL right column blocks,
    instead of interleaving by Y-bands. Use SOTA layout (B04) for
    correct multi-column reading order.

    Args:
        raw_blocks: List of block dicts with 'bbox', 'x0', 'y0' keys
        page_w: Page width
        two_col_min_side_blocks: Threshold for two-column detection
        y_tolerance: Y tolerance for same-line grouping

    Returns:
        Ordered list of blocks
    """
    if not raw_blocks:
        return []

    two_cols = is_two_column_page(
        raw_blocks, page_w=page_w, two_col_min_side_blocks=two_col_min_side_blocks
    )
    if not two_cols:
        return order_single_column(raw_blocks, y_tolerance=y_tolerance)

    full_width: List[Dict[str, Any]] = []
    left_items: List[Dict[str, Any]] = []
    right_items: List[Dict[str, Any]] = []

    for rb in raw_blocks:
        x0, _, x1, _ = rb["bbox"].coords
        width = x1 - x0

        if page_w > 0 and width >= page_w * 0.75:
            full_width.append(rb)
            continue

        xc = (x0 + x1) / 2.0
        if xc < page_w / 2.0:
            left_items.append(rb)
        else:
            right_items.append(rb)

    return (
        order_single_column(full_width, y_tolerance=y_tolerance)
        + order_single_column(left_items, y_tolerance=y_tolerance)
        + order_single_column(right_items, y_tolerance=y_tolerance)
    )


__all__ = [
    "is_two_column_page",
    "order_single_column",
    "order_blocks_deterministically",
]
