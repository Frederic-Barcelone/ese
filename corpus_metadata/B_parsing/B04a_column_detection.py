# corpus_metadata/B_parsing/B04a_column_detection.py
"""
Column detection and layout analysis utilities.

Provides:
- PageStats computation for adaptive thresholds
- Gutter detection (Breuel sweep-line method)
- Column boundary clustering
- L-shaped region detection (XY-Cut++ pre-mask)
- Spanning element detection

Extracted from B04_column_ordering.py to reduce file size.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from B_parsing.B04_column_ordering import LayoutConfig, BlockGeom, SemanticPriority


# =============================================================================
# STATISTICS
# =============================================================================


@dataclass
class PageStats:
    """Computed statistics for adaptive thresholds (XY-Cut++ enhanced)."""

    page_width: float
    page_height: float

    # Block statistics
    median_width: float
    median_height: float
    median_x_gap: float
    median_y_gap: float

    # XY-Cut++ adaptive thresholds
    median_bbox_length: float  # For β×median spanning threshold
    density_ratio: float  # τ_d for axis selection

    # Content bounds
    content_left: float
    content_right: float
    content_top: float
    content_bottom: float

    # Counts
    num_blocks: int

    def adaptive_spanning_threshold(self, beta: float = 1.3) -> float:
        """
        Compute adaptive spanning threshold (XY-Cut++).

        T_l = β × median({l_i}) where β typically = 1.3
        Elements exceeding this are considered cross-layout.
        """
        return beta * self.median_bbox_length

    def should_use_xy_cut(self, threshold: float = 0.9) -> bool:
        """
        Determine cut direction based on density ratio (XY-Cut++).

        τ_d > threshold → horizontal-first (XY-Cut)
        τ_d ≤ threshold → vertical-first (YX-Cut)
        """
        return self.density_ratio > threshold

    @classmethod
    def compute(
        cls, geoms: List["BlockGeom"], page_w: float, page_h: float
    ) -> "PageStats":
        """Compute statistics from block geometries."""
        defaults = cls(
            page_width=page_w,
            page_height=page_h,
            median_width=page_w * 0.3,
            median_height=page_h * 0.02,
            median_x_gap=page_w * 0.05,
            median_y_gap=page_h * 0.01,
            median_bbox_length=page_w * 0.3,
            density_ratio=1.0,
            content_left=0,
            content_right=page_w,
            content_top=0,
            content_bottom=page_h,
            num_blocks=0,
        )

        if not geoms:
            return defaults

        widths = [g.width for g in geoms]
        heights = [g.height for g in geoms]

        # XY-Cut++: median of max(width, height) for each block
        bbox_lengths = [max(g.width, g.height) for g in geoms]

        # XY-Cut++: density ratio τ_d = column_area / row_area
        col_area = sum(g.width * g.height for g in geoms if g.width < g.height)
        row_area = sum(g.width * g.height for g in geoms if g.width >= g.height)
        density_ratio = col_area / row_area if row_area > 0 else 1.0

        # Compute horizontal gaps
        sorted_x = sorted(geoms, key=lambda g: g.x0)
        x_gaps = []
        for i in range(len(sorted_x) - 1):
            gap = sorted_x[i + 1].x0 - sorted_x[i].x1
            if gap > 0:
                x_gaps.append(gap)

        # Compute vertical gaps
        sorted_y = sorted(geoms, key=lambda g: g.y0)
        y_gaps = []
        for i in range(len(sorted_y) - 1):
            gap = sorted_y[i + 1].y0 - sorted_y[i].y1
            if gap > 0:
                y_gaps.append(gap)

        return cls(
            page_width=page_w,
            page_height=page_h,
            median_width=statistics.median(widths) if widths else defaults.median_width,
            median_height=statistics.median(heights)
            if heights
            else defaults.median_height,
            median_x_gap=statistics.median(x_gaps) if x_gaps else defaults.median_x_gap,
            median_y_gap=statistics.median(y_gaps) if y_gaps else defaults.median_y_gap,
            median_bbox_length=statistics.median(bbox_lengths)
            if bbox_lengths
            else defaults.median_bbox_length,
            density_ratio=density_ratio,
            content_left=min(g.x0 for g in geoms),
            content_right=max(g.x1 for g in geoms),
            content_top=min(g.y0 for g in geoms),
            content_bottom=max(g.y1 for g in geoms),
            num_blocks=len(geoms),
        )


# =============================================================================
# GUTTER DETECTION
# =============================================================================


@dataclass
class Gutter:
    """Detected column separator."""

    x_left: float  # Right edge of left content
    x_right: float  # Left edge of right content
    confidence: float

    @property
    def center(self) -> float:
        return (self.x_left + self.x_right) / 2

    @property
    def width(self) -> float:
        return self.x_right - self.x_left


def find_gutters(
    geoms: List["BlockGeom"], stats: PageStats, config: "LayoutConfig"
) -> List[Gutter]:
    """
    Find column gutters using whitespace analysis.

    Based on Breuel's sweep-line algorithm for maximal whitespace.
    """
    if len(geoms) < config.min_blocks_per_column * 2:
        return []

    # Adaptive minimum gutter width
    min_gutter = max(
        stats.median_x_gap * config.gutter_min_width_factor, config.min_gutter_width_abs
    )

    # Search region (middle portion of page)
    margin = config.column_search_margin
    search_left = stats.page_width * margin
    search_right = stats.page_width * (1 - margin)

    # Find gaps using sweep line
    events = []
    for g in geoms:
        events.append((g.x0, "start", g))
        events.append((g.x1, "end", g))

    events.sort(key=lambda e: (e[0], 0 if e[1] == "start" else 1))

    active: Set[int] = set()
    gaps = []
    last_x = stats.content_left

    for x, event_type, geom in events:
        if event_type == "start":
            if not active and x - last_x >= min_gutter:
                gap_center = (last_x + x) / 2
                if search_left <= gap_center <= search_right:
                    gaps.append((last_x, x))
            active.add(id(geom))
        else:
            active.discard(id(geom))
            if not active:
                last_x = x

    # Convert to Gutter objects and validate
    gutters = []
    for gap_left, gap_right in gaps:
        gutter = Gutter(
            x_left=gap_left,
            x_right=gap_right,
            confidence=min(1.0, (gap_right - gap_left) / (min_gutter * 2)),
        )

        # Validate: sufficient blocks on each side
        if _validate_gutter(gutter, geoms, stats, config):
            gutters.append(gutter)

    # Sort by position
    gutters.sort(key=lambda g: g.center)

    # Limit to 2 gutters max (for 3-column)
    if not config.enable_three_column:
        gutters = gutters[:1]
    else:
        gutters = gutters[:2]

    return gutters


def _validate_gutter(
    gutter: Gutter, geoms: List["BlockGeom"], stats: PageStats, config: "LayoutConfig"
) -> bool:
    """Validate gutter has sufficient content on both sides."""
    left_blocks = [g for g in geoms if g.x_center < gutter.center]
    right_blocks = [g for g in geoms if g.x_center > gutter.center]

    if (
        len(left_blocks) < config.min_blocks_per_column
        or len(right_blocks) < config.min_blocks_per_column
    ):
        return False

    # Check column widths
    min_col = stats.page_width * config.min_column_width_pct
    left_width = gutter.x_left - stats.content_left
    right_width = stats.content_right - gutter.x_right

    if left_width < min_col or right_width < min_col:
        return False

    return True


# =============================================================================
# X-COORDINATE CLUSTERING (Fallback Column Detection)
# =============================================================================


def find_columns_by_clustering(
    geoms: List["BlockGeom"], stats: PageStats, config: "LayoutConfig"
) -> List[Gutter]:
    """
    Detect column boundaries by clustering block x-coordinates.

    This is a fallback method when sweep-line gutter detection fails,
    particularly useful when Unstructured.io merges some content across columns.

    Algorithm:
    1. Collect x_center values for narrow blocks (width < 50% page)
    2. Sort and find gaps > min_column_gap
    3. Gaps between clusters indicate column boundaries

    Returns:
        List of Gutter objects representing column boundaries
    """
    if len(geoms) < config.min_blocks_per_column * 2:
        return []

    # Filter to narrow blocks (likely single-column content)
    max_width = stats.page_width * 0.45  # Less than 45% of page width
    narrow_geoms = [g for g in geoms if g.width < max_width]

    if len(narrow_geoms) < config.min_blocks_per_column * 2:
        return []

    # Collect x_center values
    x_centers = sorted(g.x_center for g in narrow_geoms)

    if len(x_centers) < 4:
        return []

    # Find gaps in x_centers
    # Use adaptive gap threshold based on page width
    min_gap = stats.page_width * 0.08  # At least 8% of page width gap

    gaps = []
    for i in range(len(x_centers) - 1):
        gap_distance = x_centers[i + 1] - x_centers[i]
        if gap_distance > min_gap:
            # Found a potential column boundary
            gap_center = (x_centers[i] + x_centers[i + 1]) / 2
            # Only consider gaps in middle portion of page
            if 0.2 * stats.page_width < gap_center < 0.8 * stats.page_width:
                gaps.append(
                    {
                        "x_left": x_centers[i],
                        "x_right": x_centers[i + 1],
                        "center": gap_center,
                        "width": gap_distance,
                    }
                )

    if not gaps:
        return []

    # Convert to Gutter objects
    gutters = []
    for gap in gaps[:2]:  # Max 2 gutters for 3-column
        gutter = Gutter(
            x_left=gap["x_left"],
            x_right=gap["x_right"],
            confidence=min(1.0, gap["width"] / (min_gap * 2)),
        )

        # Validate: count blocks on each side
        left_count = sum(1 for g in narrow_geoms if g.x_center < gap["center"])
        right_count = sum(1 for g in narrow_geoms if g.x_center > gap["center"])

        if (
            left_count >= config.min_blocks_per_column
            and right_count >= config.min_blocks_per_column
        ):
            gutters.append(gutter)

    return gutters


# =============================================================================
# L-SHAPED DETECTION (XY-Cut++ Pre-Mask)
# =============================================================================


def detect_l_shaped_regions(
    geoms: List["BlockGeom"], config: "LayoutConfig"
) -> List["BlockGeom"]:
    """
    Detect L-shaped regions that would break standard XY-cut (XY-Cut++ pre-mask).

    L-shaped elements overlap with 2+ other elements in perpendicular directions,
    causing XY-cut to produce incorrect splits.

    Returns:
        List of BlockGeom flagged as L-shaped (geom.is_l_shaped = True)
    """
    if not config.enable_l_shape_masking or len(geoms) < 3:
        return geoms

    # For each block, count perpendicular overlaps
    for geom in geoms:
        h_overlaps = 0  # Horizontal neighbors (same Y band, different X)
        v_overlaps = 0  # Vertical neighbors (same X band, different Y)

        for other in geoms:
            if other is geom:
                continue

            # Check horizontal overlap (Y ranges intersect)
            y_overlap = min(geom.y1, other.y1) - max(geom.y0, other.y0)
            # Check vertical overlap (X ranges intersect)
            x_overlap = min(geom.x1, other.x1) - max(geom.x0, other.x0)

            if y_overlap > 0 and x_overlap <= 0:
                # Same horizontal band, no X overlap → horizontal neighbor
                h_overlaps += 1
            elif x_overlap > 0 and y_overlap <= 0:
                # Same vertical band, no Y overlap → vertical neighbor
                v_overlaps += 1

        # L-shaped: has neighbors in BOTH directions
        if h_overlaps >= 1 and v_overlaps >= 1:
            total_overlaps = h_overlaps + v_overlaps
            if total_overlaps >= config.l_shape_overlap_threshold:
                geom.is_l_shaped = True

    return geoms


# =============================================================================
# SPANNING DETECTION (XY-Cut++ Enhanced)
# =============================================================================


def detect_spanning(
    geoms: List["BlockGeom"],
    gutters: List[Gutter],
    stats: PageStats,
    config: "LayoutConfig",
) -> Tuple[List["BlockGeom"], List["BlockGeom"]]:
    """
    Separate spanning elements from column-bound elements (XY-Cut++ enhanced).

    Uses adaptive β×median threshold instead of fixed percentage.

    Returns:
        (spanning_blocks, column_blocks)
    """
    # FIX: When no gutters, no elements are spanning (single-column)
    if not gutters:
        return [], geoms.copy()

    spanning = []
    column_bound = []

    # XY-Cut++ adaptive threshold: β × median(bbox_lengths)
    adaptive_threshold = stats.adaptive_spanning_threshold(config.spanning_beta)

    # Fallback: fixed percentage threshold
    fixed_threshold = stats.page_width * config.spanning_width_pct

    for geom in geoms:
        is_spanning = False

        # Check 1: Adaptive width-based (XY-Cut++)
        bbox_length = max(geom.width, geom.height)
        if bbox_length >= adaptive_threshold:
            is_spanning = True

        # Check 2: Width exceeds fixed threshold
        if not is_spanning and geom.width >= fixed_threshold:
            is_spanning = True

        # Check 3: Crosses gutter with significant overlap
        if not is_spanning:
            for gutter in gutters:
                if geom.x0 < gutter.center < geom.x1:
                    overlap_left = gutter.center - geom.x0
                    overlap_right = geom.x1 - gutter.center
                    min_overlap = geom.width * config.spanning_overlap_ratio

                    if overlap_left > min_overlap and overlap_right > min_overlap:
                        is_spanning = True
                        break

        # Check 4: Title/Header categories often span (Unstructured hint)
        if not is_spanning and config.use_unstructured_categories:
            if geom.category and geom.category.lower() in config.title_categories:
                if geom.width >= stats.page_width * 0.4:
                    is_spanning = True

        # Check 5: L-shaped elements are treated as spanning for safety
        if not is_spanning and geom.is_l_shaped:
            is_spanning = True

        # Update flag and classify
        geom.is_spanning = is_spanning
        if is_spanning:
            spanning.append(geom)
        else:
            column_bound.append(geom)

    return spanning, column_bound


__all__ = [
    "PageStats",
    "Gutter",
    "find_gutters",
    "find_columns_by_clustering",
    "detect_l_shaped_regions",
    "detect_spanning",
]
