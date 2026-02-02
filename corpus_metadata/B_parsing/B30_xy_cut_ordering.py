# corpus_metadata/B_parsing/B30_xy_cut_ordering.py
"""
XY-Cut++ ordering algorithms for reading order detection.

This module provides the core XY-Cut ordering algorithm with XY-Cut++ enhancements:
density-driven axis selection, recursive partitioning with quality scoring,
semantic priority ordering, and multi-column body band interleaving.

Key Components:
    - xy_cut_order: Main XY-Cut ordering with density-driven axis selection
    - order_body_bands: Multi-column body band ordering with Y-interleaving
    - _xy_cut_recursive: Recursive XY-Cut partitioning implementation
    - _find_best_cut: Find best cut point with quality scoring
    - _compute_cut_quality: Score cut quality based on gap and balance

Example:
    >>> from B_parsing.B30_xy_cut_ordering import xy_cut_order
    >>> ordered_geoms = xy_cut_order(geoms, stats, config)
    >>> for g in ordered_geoms:
    ...     print(f"Block at ({g.x0}, {g.y0}): {g.block_ref.get('text', '')[:30]}")

Dependencies:
    - B_parsing.B04_column_ordering: BlockGeom, LayoutConfig, PageLayout (TYPE_CHECKING only)
    - B_parsing.B29_column_detection: PageStats (TYPE_CHECKING only)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from B_parsing.B04_column_ordering import (
        BlockGeom,
        LayoutConfig,
        PageLayout,
    )
    from B_parsing.B29_column_detection import PageStats


# =============================================================================
# XY-CUT ORDERING (XY-Cut++ Enhanced)
# =============================================================================


def xy_cut_order(
    geoms: List["BlockGeom"],
    stats: "PageStats",
    config: "LayoutConfig",
) -> List["BlockGeom"]:
    """
    Order blocks using recursive XY-Cut with density-driven axis selection (XY-Cut++).

    Key improvements over standard XY-Cut:
    - Density-driven axis selection (τ_d ratio)
    - Semantic priority sorting within groups
    - Configurable min_gap_factor
    """
    from B_parsing.B29_column_detection import detect_l_shaped_regions

    if len(geoms) <= 1:
        return geoms

    # Pre-process: detect L-shaped regions
    if config.enable_l_shape_masking:
        geoms = detect_l_shaped_regions(geoms, config)

    # Determine primary axis from density ratio
    use_xy = stats.should_use_xy_cut(config.density_ratio_threshold)

    return _xy_cut_recursive(geoms, stats, config, prefer_y_first=use_xy)


def _xy_cut_recursive(
    geoms: List["BlockGeom"],
    stats: "PageStats",
    config: "LayoutConfig",
    prefer_y_first: bool = True,
) -> List["BlockGeom"]:
    """
    Recursive XY-Cut implementation (XY-Cut++ enhanced).

    Args:
        prefer_y_first: If True, prefer Y-cuts (horizontal splits) for top-to-bottom reading.
                       Determined by density ratio at top level.
    """
    if len(geoms) <= config.xy_cut_min_elements:
        # Sort by semantic priority, then Y, then X
        if config.enable_semantic_priority:
            return sorted(geoms, key=lambda g: (g.semantic_priority.value, g.y0, g.x0))
        return sorted(geoms, key=lambda g: (g.y0, g.x0))

    # Find best cuts
    x_cut = _find_x_cut(geoms, stats, config)
    y_cut = _find_y_cut(geoms, stats, config)

    if x_cut is None and y_cut is None:
        if config.enable_semantic_priority:
            return sorted(geoms, key=lambda g: (g.semantic_priority.value, g.y0, g.x0))
        return sorted(geoms, key=lambda g: (g.y0, g.x0))

    # Choose axis based on XY-Cut++ density-driven selection
    if x_cut is None:
        cut_pos, axis = y_cut, "y"
    elif y_cut is None:
        cut_pos, axis = x_cut, "x"
    else:
        x_score = _cut_quality(geoms, x_cut, "x")
        y_score = _cut_quality(geoms, y_cut, "y")

        # XY-Cut++: Use density ratio to determine preference
        # τ_d > 0.9 → prefer Y-cuts (horizontal reading bands)
        if prefer_y_first:
            # Prefer Y-cuts when density suggests columnar content
            if y_score >= x_score * 0.7:  # More aggressive Y preference
                cut_pos, axis = y_cut, "y"
            else:
                cut_pos, axis = x_cut, "x"
        else:
            # Prefer X-cuts when density suggests row-based content
            if x_score >= y_score * 0.7:
                cut_pos, axis = x_cut, "x"
            else:
                cut_pos, axis = y_cut, "y"

    # Partition and recurse
    assert cut_pos is not None  # Guaranteed by logic above
    if axis == "x":
        left = [g for g in geoms if g.x_center < cut_pos]
        right = [g for g in geoms if g.x_center >= cut_pos]
        return _xy_cut_recursive(
            left, stats, config, prefer_y_first
        ) + _xy_cut_recursive(right, stats, config, prefer_y_first)
    else:
        top = [g for g in geoms if g.y_center < cut_pos]
        bottom = [g for g in geoms if g.y_center >= cut_pos]
        return _xy_cut_recursive(
            top, stats, config, prefer_y_first
        ) + _xy_cut_recursive(bottom, stats, config, prefer_y_first)


def _find_x_cut(
    geoms: List["BlockGeom"],
    stats: "PageStats",
    config: Optional["LayoutConfig"] = None,
) -> Optional[float]:
    """Find best vertical cut."""
    from B_parsing.B04_column_ordering import LayoutConfig

    if len(geoms) < 2:
        return None

    cfg = config or LayoutConfig()
    sorted_by_x = sorted(geoms, key=lambda g: g.x1)
    best_gap: float = 0.0
    best_pos: Optional[float] = None
    min_gap = stats.median_x_gap * cfg.min_gap_factor

    for i in range(len(sorted_by_x) - 1):
        gap = sorted_by_x[i + 1].x0 - sorted_by_x[i].x1
        if gap > best_gap and gap >= min_gap:
            best_gap = gap
            best_pos = (sorted_by_x[i].x1 + sorted_by_x[i + 1].x0) / 2

    return best_pos


def _find_y_cut(
    geoms: List["BlockGeom"],
    stats: "PageStats",
    config: Optional["LayoutConfig"] = None,
) -> Optional[float]:
    """Find best horizontal cut."""
    from B_parsing.B04_column_ordering import LayoutConfig

    if len(geoms) < 2:
        return None

    cfg = config or LayoutConfig()
    sorted_by_y = sorted(geoms, key=lambda g: g.y1)
    best_gap: float = 0.0
    best_pos: Optional[float] = None
    min_gap = stats.median_y_gap * cfg.min_gap_factor

    for i in range(len(sorted_by_y) - 1):
        gap = sorted_by_y[i + 1].y0 - sorted_by_y[i].y1
        if gap > best_gap and gap >= min_gap:
            best_gap = gap
            best_pos = (sorted_by_y[i].y1 + sorted_by_y[i + 1].y0) / 2

    return best_pos


def _cut_quality(geoms: List["BlockGeom"], cut_pos: float, axis: str) -> float:
    """
    Compute cut quality score (XY-Cut++).

    Score combines:
    - Balance: How evenly the cut divides elements (40%)
    - Separation: How few elements straddle the cut (60%)
    """
    if axis == "x":
        left = [g for g in geoms if g.x_center < cut_pos]
        right = [g for g in geoms if g.x_center >= cut_pos]
        straddling = sum(1 for g in geoms if g.x0 < cut_pos < g.x1)
    else:
        left = [g for g in geoms if g.y_center < cut_pos]
        right = [g for g in geoms if g.y_center >= cut_pos]
        straddling = sum(1 for g in geoms if g.y0 < cut_pos < g.y1)

    if not left or not right:
        return 0.0

    total = len(geoms)
    balance = 1.0 - abs(len(left) - len(right)) / total
    separation = 1.0 - straddling / total

    return balance * 0.4 + separation * 0.6


# =============================================================================
# MULTI-COLUMN ORDERING
# =============================================================================


def order_body_bands(
    geoms: List["BlockGeom"],
    layout: "PageLayout",
    config: "LayoutConfig",
) -> List["BlockGeom"]:
    """
    Order body blocks using Y-band interleaving (XY-Cut++ enhanced).

    Key: Process bands top-to-bottom.
    Within each band: spanning → left → center → right

    Uses content_key() for identity instead of id() to avoid fragility.
    """
    if not geoms:
        return []

    stats = layout.stats
    if stats is None:
        return sorted(geoms, key=lambda g: (g.y0, g.x0))
    band_height = max(
        stats.median_height * config.band_height_factor, config.y_tolerance * 3
    )

    # Build spanning set using content_key for stable identity
    spanning_keys = {g.content_key() for g in layout.spanning_blocks}

    def get_column(g: "BlockGeom") -> int:
        """0=spanning, 1=left, 2=center, 3=right"""
        # Use is_spanning flag or content_key lookup
        if g.is_spanning or g.content_key() in spanning_keys:
            return 0

        if not layout.gutters:
            return 0

        if len(layout.gutters) == 1:
            return 1 if g.x_center < layout.gutters[0].center else 3

        g1 = layout.gutters[0].center
        g2 = layout.gutters[1].center

        if g.x_center < g1:
            return 1
        elif g.x_center >= g2:
            return 3
        else:
            return 2

    def get_band(g: "BlockGeom") -> int:
        return int(g.y0 / band_height) if band_height > 0 else 0

    # Group by band
    bands: Dict[int, List[Tuple[int, "BlockGeom"]]] = defaultdict(list)
    for g in geoms:
        bands[get_band(g)].append((get_column(g), g))

    # Order within each band
    ordered: list["BlockGeom"] = []
    for band_idx in sorted(bands.keys()):
        band_items = bands[band_idx]
        # Sort by: column, then Y, then X
        band_items.sort(key=lambda item: (item[0], item[1].y0, item[1].x0))
        ordered.extend(g for _, g in band_items)

    return ordered


__all__ = [
    "xy_cut_order",
    "order_body_bands",
]
