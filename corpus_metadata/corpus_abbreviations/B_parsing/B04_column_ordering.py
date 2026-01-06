# B_parsing/B04_column_ordering.py
"""
SOTA Column Layout Detection and Reading Order Module
=====================================================

Compatible with Unstructured.io (hi_res, fast, auto strategies)

INTEGRATION:
    Drop-in replacement for PDFToDocGraphParser._order_blocks_deterministically()

    In B01_pdf_to_docgraph.py, change:
        ordered = self._order_blocks_deterministically(raw_pages[page_num], page_w=page_w)
    To:
        from B_parsing.B04_column_ordering import order_page_blocks
        ordered = order_page_blocks(raw_pages[page_num], page_w, page_h)

IMPLEMENTS:
    - XY-Cut++ style hierarchical segmentation (arxiv:2504.10258)
    - Whitespace-based gutter detection (Breuel method)
    - Cross-layout element detection with adaptive β×median threshold
    - Density-driven axis selection (τ_d ratio)
    - L-shaped region pre-masking
    - Semantic priority ordering (SPANNING > TITLE > TABLE > NARRATIVE)
    - Y-band interleaving for multi-column
    - Per-page adaptive layout detection
    - PPTX-to-PDF mode with inverted z-order handling

SUPPORTED LAYOUTS:
    - SINGLE_COLUMN: Standard single-column document
    - TWO_COLUMN: Academic paper style
    - THREE_COLUMN: Newsletter/magazine style
    - MIXED_HEADER: Single-col header + multi-col body
    - COMPLEX: Irregular with floating elements

INPUT FORMAT (from B01):
    raw_blocks = [
        {
            "text": str,
            "bbox": BoundingBox,      # from A01_domain_models
            "x0": float,              # bbox.coords[0]
            "y0": float,              # bbox.coords[1]
            "zone": str,              # "HEADER" | "BODY" | "FOOTER"
            "is_section_header": bool,
            # Optional Unstructured metadata:
            "category": str,          # "Title", "NarrativeText", "Table", etc.
            "element_id": str,
        },
        ...
    ]

OUTPUT FORMAT:
    Same list of dicts, reordered for correct reading sequence.

VERSION: 3.0.0 (XY-Cut++ / SOTA)
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
import logging

# Import from your core modules
try:
    from A_core.A01_domain_models import BoundingBox
except ImportError:
    # Fallback for standalone testing
    BoundingBox = None

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class LayoutType(str, Enum):
    """Page layout classification."""
    SINGLE_COLUMN = "single_column"
    TWO_COLUMN = "two_column"
    THREE_COLUMN = "three_column"
    MIXED_HEADER = "mixed_header"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


class BlockClass(str, Enum):
    """Block classification for ordering."""
    SPANNING = "spanning"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class SemanticPriority(int, Enum):
    """
    Semantic priority for reading order (XY-Cut++ CMM).
    Lower value = higher priority (read first).
    """
    CROSS_LAYOUT = 0   # Spanning elements (headers, full-width)
    TITLE = 1          # Titles, section headers
    VISION = 2         # Tables, figures, images
    NARRATIVE = 3      # Body text, paragraphs
    OTHER = 4          # Footers, captions, misc


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LayoutConfig:
    """
    Configuration for layout detection.
    Defaults tuned for Unstructured.io hi_res output on academic/clinical docs.
    Incorporates XY-Cut++ (arxiv:2504.10258) SOTA parameters.
    """
    # -------------------------------------------------------------------------
    # XY-Cut++ SOTA Parameters
    # -------------------------------------------------------------------------

    # Adaptive spanning threshold: β × median(bbox_lengths)
    # From XY-Cut++: β=1.3 for cross-layout detection
    spanning_beta: float = 1.3

    # Density ratio threshold for axis selection
    # τ_d > this → horizontal-first (XY), else vertical-first (YX)
    density_ratio_threshold: float = 0.9

    # Enable L-shaped region pre-masking
    enable_l_shape_masking: bool = True
    l_shape_overlap_threshold: int = 2  # Min overlaps to detect L-shape

    # Semantic priority ordering (XY-Cut++ CMM)
    enable_semantic_priority: bool = True

    # -------------------------------------------------------------------------
    # Gutter Detection (Breuel method)
    # -------------------------------------------------------------------------
    gutter_min_width_factor: float = 1.5      # Multiplier of median horizontal gap
    min_gutter_width_abs: float = 8.0         # Absolute minimum gutter width (points)
                                              # Academic journals often have 8-12pt gutters

    # -------------------------------------------------------------------------
    # Column Validation
    # -------------------------------------------------------------------------
    min_blocks_per_column: int = 3
    min_column_width_pct: float = 0.18        # Min column as fraction of page
    max_column_width_pct: float = 0.58        # Max single column width

    # -------------------------------------------------------------------------
    # Spanning Element Detection (fallback if adaptive disabled)
    # -------------------------------------------------------------------------
    spanning_width_pct: float = 0.55          # Width > this% = spanning
    spanning_overlap_ratio: float = 0.25      # Overlap ratio to consider spanning

    # -------------------------------------------------------------------------
    # Zone Detection (% of page height)
    # -------------------------------------------------------------------------
    header_zone_pct: float = 0.12             # Top 12% is header zone
    footer_zone_pct: float = 0.10             # Bottom 10% is footer zone

    # -------------------------------------------------------------------------
    # Y-band Parameters
    # -------------------------------------------------------------------------
    y_tolerance: float = 5.0                  # Blocks within this Y are "same line"
    band_height_factor: float = 1.8           # Multiplier of median block height

    # -------------------------------------------------------------------------
    # XY-Cut Parameters
    # -------------------------------------------------------------------------
    xy_cut_min_elements: int = 2
    min_gap_factor: float = 0.3               # Min gap as fraction of median gap

    # -------------------------------------------------------------------------
    # Column Search
    # -------------------------------------------------------------------------
    column_search_margin: float = 0.25        # Search for gutters in middle 50%

    # -------------------------------------------------------------------------
    # Three-column Support
    # -------------------------------------------------------------------------
    enable_three_column: bool = True
    three_col_min_blocks: int = 3

    # -------------------------------------------------------------------------
    # PPTX-to-PDF Mode
    # -------------------------------------------------------------------------
    pptx_mode: bool = False
    pptx_invert_z_order: bool = True          # PowerPoint reading order is inverted
    pptx_footer_patterns: Set[str] = field(default_factory=lambda: {
        "slide", "page", "©", "copyright", "confidential", "footer",
        "all rights reserved", "proprietary"
    })
    pptx_aspect_ratio_range: Tuple[float, float] = (1.2, 1.9)  # 4:3 to 16:9

    # -------------------------------------------------------------------------
    # Unstructured.io Categories
    # -------------------------------------------------------------------------
    use_unstructured_categories: bool = True
    title_categories: Set[str] = field(default_factory=lambda: {
        "title", "header", "headline", "sectionheader"
    })
    table_categories: Set[str] = field(default_factory=lambda: {
        "table", "tablecell", "figure", "image", "figurecaption"
    })
    narrative_categories: Set[str] = field(default_factory=lambda: {
        "narrativetext", "text", "listitem", "paragraph"
    })

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------
    debug_mode: bool = False


# =============================================================================
# GEOMETRY EXTRACTION
# =============================================================================

@dataclass
class BlockGeom:
    """Extracted geometry from a raw block dict."""
    block_ref: Dict[str, Any]
    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    height: float
    x_center: float
    y_center: float
    zone: str
    category: Optional[str] = None
    is_section_header: bool = False
    semantic_priority: SemanticPriority = SemanticPriority.OTHER
    is_l_shaped: bool = False  # Flagged by L-shape detection
    is_spanning: bool = False  # Flagged by spanning detection

    def content_key(self) -> Tuple[float, float, str]:
        """Content-based key for identity (avoids id() fragility)."""
        text = self.block_ref.get("text", "")[:50] if self.block_ref else ""
        return (round(self.x0, 1), round(self.y0, 1), text)


def extract_block_geometry(
    block: Dict[str, Any],
    config: Optional[LayoutConfig] = None
) -> Optional[BlockGeom]:
    """
    Extract geometry from B01 raw_block dict.

    Handles both BoundingBox objects and raw coordinate tuples.
    Assigns semantic priority based on category (XY-Cut++ CMM).
    """
    bbox = block.get("bbox")
    if bbox is None:
        return None

    # Extract coordinates from BoundingBox or tuple
    if hasattr(bbox, 'coords'):
        x0, y0, x1, y1 = bbox.coords
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x0, y0, x1, y1 = bbox[:4]
    else:
        # Try x0, y0 from block directly
        x0 = block.get("x0", 0)
        y0 = block.get("y0", 0)
        x1 = x0 + 10  # Fallback
        y1 = y0 + 10

    width = x1 - x0
    height = y1 - y0

    if width <= 0 or height <= 0:
        return None

    # Determine semantic priority from category
    category = block.get("category")
    semantic_priority = _get_semantic_priority(category, block, config)

    return BlockGeom(
        block_ref=block,
        x0=float(x0),
        y0=float(y0),
        x1=float(x1),
        y1=float(y1),
        width=float(width),
        height=float(height),
        x_center=(x0 + x1) / 2,
        y_center=(y0 + y1) / 2,
        zone=block.get("zone", "BODY"),
        category=category,
        is_section_header=block.get("is_section_header", False),
        semantic_priority=semantic_priority,
    )


def _get_semantic_priority(
    category: Optional[str],
    block: Dict[str, Any],
    config: Optional[LayoutConfig] = None
) -> SemanticPriority:
    """
    Determine semantic priority for reading order (XY-Cut++ CMM).

    Priority: CROSS_LAYOUT > TITLE > VISION > NARRATIVE > OTHER
    """
    cfg = config or LayoutConfig()

    if not cfg.use_unstructured_categories or not category:
        # Fallback: use is_section_header hint
        if block.get("is_section_header"):
            return SemanticPriority.TITLE
        return SemanticPriority.NARRATIVE

    cat_lower = category.lower()

    # Title/header elements
    if cat_lower in cfg.title_categories:
        return SemanticPriority.TITLE

    # Table/figure/image elements
    if cat_lower in cfg.table_categories:
        return SemanticPriority.VISION

    # Narrative text
    if cat_lower in cfg.narrative_categories:
        return SemanticPriority.NARRATIVE

    # Section headers (check is_section_header flag)
    if block.get("is_section_header"):
        return SemanticPriority.TITLE

    return SemanticPriority.OTHER


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
    median_bbox_length: float      # For β×median spanning threshold
    density_ratio: float           # τ_d for axis selection

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
    def compute(cls, geoms: List[BlockGeom], page_w: float, page_h: float) -> 'PageStats':
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
            median_height=statistics.median(heights) if heights else defaults.median_height,
            median_x_gap=statistics.median(x_gaps) if x_gaps else defaults.median_x_gap,
            median_y_gap=statistics.median(y_gaps) if y_gaps else defaults.median_y_gap,
            median_bbox_length=statistics.median(bbox_lengths) if bbox_lengths else defaults.median_bbox_length,
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
    x_left: float       # Right edge of left content
    x_right: float      # Left edge of right content
    confidence: float
    
    @property
    def center(self) -> float:
        return (self.x_left + self.x_right) / 2
    
    @property
    def width(self) -> float:
        return self.x_right - self.x_left


def find_gutters(
    geoms: List[BlockGeom],
    stats: PageStats,
    config: LayoutConfig
) -> List[Gutter]:
    """
    Find column gutters using whitespace analysis.
    
    Based on Breuel's sweep-line algorithm for maximal whitespace.
    """
    if len(geoms) < config.min_blocks_per_column * 2:
        return []
    
    # Adaptive minimum gutter width
    min_gutter = max(
        stats.median_x_gap * config.gutter_min_width_factor,
        config.min_gutter_width_abs
    )
    
    # Search region (middle portion of page)
    margin = config.column_search_margin
    search_left = stats.page_width * margin
    search_right = stats.page_width * (1 - margin)
    
    # Find gaps using sweep line
    events = []
    for g in geoms:
        events.append((g.x0, 'start', g))
        events.append((g.x1, 'end', g))
    
    events.sort(key=lambda e: (e[0], 0 if e[1] == 'start' else 1))
    
    active: Set[int] = set()
    gaps = []
    last_x = stats.content_left
    
    for x, event_type, geom in events:
        if event_type == 'start':
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
            confidence=min(1.0, (gap_right - gap_left) / (min_gutter * 2))
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
    gutter: Gutter,
    geoms: List[BlockGeom],
    stats: PageStats,
    config: LayoutConfig
) -> bool:
    """Validate gutter has sufficient content on both sides."""
    left_blocks = [g for g in geoms if g.x_center < gutter.center]
    right_blocks = [g for g in geoms if g.x_center > gutter.center]
    
    if (len(left_blocks) < config.min_blocks_per_column or
        len(right_blocks) < config.min_blocks_per_column):
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
    geoms: List[BlockGeom],
    stats: PageStats,
    config: LayoutConfig
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
        gap = x_centers[i + 1] - x_centers[i]
        if gap > min_gap:
            # Found a potential column boundary
            gap_center = (x_centers[i] + x_centers[i + 1]) / 2
            # Only consider gaps in middle portion of page
            if 0.2 * stats.page_width < gap_center < 0.8 * stats.page_width:
                gaps.append({
                    'x_left': x_centers[i],
                    'x_right': x_centers[i + 1],
                    'center': gap_center,
                    'width': gap,
                })

    if not gaps:
        return []

    # Convert to Gutter objects
    gutters = []
    for gap in gaps[:2]:  # Max 2 gutters for 3-column
        gutter = Gutter(
            x_left=gap['x_left'],
            x_right=gap['x_right'],
            confidence=min(1.0, gap['width'] / (min_gap * 2))
        )

        # Validate: count blocks on each side
        left_count = sum(1 for g in narrow_geoms if g.x_center < gap['center'])
        right_count = sum(1 for g in narrow_geoms if g.x_center > gap['center'])

        if (left_count >= config.min_blocks_per_column and
                right_count >= config.min_blocks_per_column):
            gutters.append(gutter)

    return gutters


# =============================================================================
# L-SHAPED DETECTION (XY-Cut++ Pre-Mask)
# =============================================================================

def detect_l_shaped_regions(
    geoms: List[BlockGeom],
    config: LayoutConfig
) -> List[BlockGeom]:
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
    geoms: List[BlockGeom],
    gutters: List[Gutter],
    stats: PageStats,
    config: LayoutConfig
) -> Tuple[List[BlockGeom], List[BlockGeom]]:
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

    # Use the more permissive threshold
    span_threshold = min(adaptive_threshold, fixed_threshold)

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

        # Update geom flag and categorize
        geom.is_spanning = is_spanning
        if is_spanning:
            geom.semantic_priority = SemanticPriority.CROSS_LAYOUT
            spanning.append(geom)
        else:
            column_bound.append(geom)

    return spanning, column_bound


# =============================================================================
# XY-CUT ORDERING (XY-Cut++ Enhanced)
# =============================================================================

def xy_cut_order(
    geoms: List[BlockGeom],
    stats: PageStats,
    config: LayoutConfig
) -> List[BlockGeom]:
    """
    Order blocks using recursive XY-Cut with density-driven axis selection (XY-Cut++).

    Key improvements over standard XY-Cut:
    - Density-driven axis selection (τ_d ratio)
    - Semantic priority sorting within groups
    - Configurable min_gap_factor
    """
    if len(geoms) <= 1:
        return geoms

    # Pre-process: detect L-shaped regions
    if config.enable_l_shape_masking:
        geoms = detect_l_shaped_regions(geoms, config)

    # Determine primary axis from density ratio
    use_xy = stats.should_use_xy_cut(config.density_ratio_threshold)

    return _xy_cut_recursive(geoms, stats, config, prefer_y_first=use_xy)


def _xy_cut_recursive(
    geoms: List[BlockGeom],
    stats: PageStats,
    config: LayoutConfig,
    prefer_y_first: bool = True
) -> List[BlockGeom]:
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
        cut_pos, axis = y_cut, 'y'
    elif y_cut is None:
        cut_pos, axis = x_cut, 'x'
    else:
        x_score = _cut_quality(geoms, x_cut, 'x')
        y_score = _cut_quality(geoms, y_cut, 'y')

        # XY-Cut++: Use density ratio to determine preference
        # τ_d > 0.9 → prefer Y-cuts (horizontal reading bands)
        if prefer_y_first:
            # Prefer Y-cuts when density suggests columnar content
            if y_score >= x_score * 0.7:  # More aggressive Y preference
                cut_pos, axis = y_cut, 'y'
            else:
                cut_pos, axis = x_cut, 'x'
        else:
            # Prefer X-cuts when density suggests row-based content
            if x_score >= y_score * 0.7:
                cut_pos, axis = x_cut, 'x'
            else:
                cut_pos, axis = y_cut, 'y'

    # Partition and recurse
    assert cut_pos is not None  # Guaranteed by logic above
    if axis == 'x':
        left = [g for g in geoms if g.x_center < cut_pos]
        right = [g for g in geoms if g.x_center >= cut_pos]
        return (_xy_cut_recursive(left, stats, config, prefer_y_first) +
                _xy_cut_recursive(right, stats, config, prefer_y_first))
    else:
        top = [g for g in geoms if g.y_center < cut_pos]
        bottom = [g for g in geoms if g.y_center >= cut_pos]
        return (_xy_cut_recursive(top, stats, config, prefer_y_first) +
                _xy_cut_recursive(bottom, stats, config, prefer_y_first))


def _find_x_cut(
    geoms: List[BlockGeom],
    stats: PageStats,
    config: Optional[LayoutConfig] = None
) -> Optional[float]:
    """Find best vertical cut."""
    if len(geoms) < 2:
        return None

    cfg = config or LayoutConfig()
    sorted_by_x = sorted(geoms, key=lambda g: g.x1)
    best_gap, best_pos = 0, None
    min_gap = stats.median_x_gap * cfg.min_gap_factor

    for i in range(len(sorted_by_x) - 1):
        gap = sorted_by_x[i + 1].x0 - sorted_by_x[i].x1
        if gap > best_gap and gap >= min_gap:
            best_gap = gap
            best_pos = (sorted_by_x[i].x1 + sorted_by_x[i + 1].x0) / 2

    return best_pos


def _find_y_cut(
    geoms: List[BlockGeom],
    stats: PageStats,
    config: Optional[LayoutConfig] = None
) -> Optional[float]:
    """Find best horizontal cut."""
    if len(geoms) < 2:
        return None

    cfg = config or LayoutConfig()
    sorted_by_y = sorted(geoms, key=lambda g: g.y1)
    best_gap, best_pos = 0, None
    min_gap = stats.median_y_gap * cfg.min_gap_factor

    for i in range(len(sorted_by_y) - 1):
        gap = sorted_by_y[i + 1].y0 - sorted_by_y[i].y1
        if gap > best_gap and gap >= min_gap:
            best_gap = gap
            best_pos = (sorted_by_y[i].y1 + sorted_by_y[i + 1].y0) / 2

    return best_pos


def _cut_quality(geoms: List[BlockGeom], cut_pos: float, axis: str) -> float:
    """
    Compute cut quality score (XY-Cut++).

    Score combines:
    - Balance: How evenly the cut divides elements (40%)
    - Separation: How few elements straddle the cut (60%)
    """
    if axis == 'x':
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
# LAYOUT DETECTION
# =============================================================================

@dataclass
class PageLayout:
    """Detected page layout."""
    page_width: float
    page_height: float
    layout_type: LayoutType
    gutters: List[Gutter] = field(default_factory=list)
    spanning_blocks: List[BlockGeom] = field(default_factory=list)
    column_blocks: List[BlockGeom] = field(default_factory=list)
    header_end_y: float = 0.0
    footer_start_y: float = float('inf')
    stats: Optional[PageStats] = None
    confidence: float = 0.0
    
    @property
    def num_columns(self) -> int:
        return len(self.gutters) + 1 if self.gutters else 1


def detect_layout(
    blocks: List[Dict[str, Any]],
    page_width: float,
    page_height: float,
    config: Optional[LayoutConfig] = None
) -> PageLayout:
    """
    Detect page layout from B01 raw_blocks (XY-Cut++ enhanced).

    Includes:
    - PPTX mode auto-detection
    - L-shaped region pre-masking
    - Adaptive spanning threshold
    """
    cfg = config or LayoutConfig()

    # Auto-detect PPTX mode from aspect ratio
    if not cfg.pptx_mode:
        aspect_ratio = page_width / page_height if page_height > 0 else 1.0
        if cfg.pptx_aspect_ratio_range[0] <= aspect_ratio <= cfg.pptx_aspect_ratio_range[1]:
            # Landscape aspect ratio suggests PPTX
            cfg = LayoutConfig(**{**cfg.__dict__, 'pptx_mode': True})
            logger.debug(f"Auto-detected PPTX mode (aspect ratio: {aspect_ratio:.2f})")

    # Extract geometries with semantic priority
    geoms = []
    for block in blocks:
        g = extract_block_geometry(block, cfg)
        if g:
            geoms.append(g)

    # Compute stats (includes XY-Cut++ metrics)
    stats = PageStats.compute(geoms, page_width, page_height)

    # Default layout
    layout = PageLayout(
        page_width=page_width,
        page_height=page_height,
        layout_type=LayoutType.SINGLE_COLUMN,
        stats=stats,
        confidence=1.0,
        header_end_y=page_height * cfg.header_zone_pct,
        footer_start_y=page_height * (1 - cfg.footer_zone_pct),
    )

    if len(geoms) < 4:
        return layout

    # L-shaped pre-masking (XY-Cut++)
    if cfg.enable_l_shape_masking:
        geoms = detect_l_shaped_regions(geoms, cfg)

    # Filter to body zone for column detection
    body_geoms = [
        g for g in geoms
        if g.y_center > layout.header_end_y and g.y_center < layout.footer_start_y
    ]

    if len(body_geoms) < cfg.min_blocks_per_column * 2:
        return layout

    # Find gutters using sweep-line method
    gutters = find_gutters(body_geoms, stats, cfg)

    # Fallback: try x-coordinate clustering if sweep-line fails
    if not gutters:
        gutters = find_columns_by_clustering(body_geoms, stats, cfg)

    if not gutters:
        return layout

    # Detect spanning elements (with adaptive threshold)
    spanning, column_bound = detect_spanning(body_geoms, gutters, stats, cfg)
    
    # Validate column layout
    if len(gutters) >= 2 and cfg.enable_three_column:
        # Check three-column validity
        g1, g2 = gutters[0], gutters[1]
        left = sum(1 for g in column_bound if g.x_center < g1.center)
        center = sum(1 for g in column_bound if g1.center <= g.x_center < g2.center)
        right = sum(1 for g in column_bound if g.x_center >= g2.center)
        
        if all(c >= cfg.three_col_min_blocks for c in [left, center, right]):
            layout.layout_type = LayoutType.THREE_COLUMN
            layout.gutters = gutters[:2]
        else:
            # Fall back to two-column with best gutter
            best = max(gutters, key=lambda g: g.confidence)
            layout.layout_type = LayoutType.TWO_COLUMN
            layout.gutters = [best]
    elif gutters:
        layout.layout_type = LayoutType.TWO_COLUMN
        layout.gutters = [gutters[0]]
    
    layout.spanning_blocks = spanning
    layout.column_blocks = column_bound
    layout.confidence = min(g.confidence for g in layout.gutters) if layout.gutters else 1.0
    
    # Check for mixed header (spanning elements in header area)
    if spanning and layout.num_columns > 1:
        header_spans = [
            g for g in spanning 
            if g.y_center < layout.header_end_y + stats.median_height * 2
        ]
        if header_spans:
            layout.layout_type = LayoutType.MIXED_HEADER
    
    return layout


# =============================================================================
# READING ORDER ENGINE
# =============================================================================

def order_by_layout(
    blocks: List[Dict[str, Any]],
    layout: PageLayout,
    config: Optional[LayoutConfig] = None
) -> List[Dict[str, Any]]:
    """
    Order blocks according to detected layout (XY-Cut++ enhanced).

    Supports:
    - Single-column XY-Cut ordering
    - Multi-column zone-aware band interleaving
    - PPTX mode with footer handling
    - Semantic priority ordering
    """
    cfg = config or LayoutConfig()

    if not blocks:
        return []

    # Extract geometries with config for semantic priority
    geoms = []
    for block in blocks:
        g = extract_block_geometry(block, cfg)
        if g:
            geoms.append(g)

    if not geoms:
        return blocks

    # PPTX mode: handle inverted reading order and footer detection
    if cfg.pptx_mode:
        geoms = _handle_pptx_ordering(geoms, layout, cfg)

    # Single column: XY-Cut ordering
    if layout.layout_type == LayoutType.SINGLE_COLUMN:
        if layout.stats is None:
            return blocks
        ordered = xy_cut_order(geoms, layout.stats, cfg)
        return [g.block_ref for g in ordered]

    # Multi-column: zone-aware, band-interleaved
    return _order_multicolumn(geoms, layout, cfg)


def _handle_pptx_ordering(
    geoms: List[BlockGeom],
    layout: PageLayout,
    config: LayoutConfig
) -> List[BlockGeom]:
    """
    Handle PPTX-specific ordering quirks.

    PowerPoint reading order:
    - Stored bottom-to-top in Selection Pane
    - Footer/page numbers should come last
    - Often chronological rather than visual order
    """
    if not config.pptx_mode:
        return geoms

    # Detect footer elements by content patterns
    footer_geoms = []
    body_geoms = []

    for g in geoms:
        text = g.block_ref.get("text", "").lower() if g.block_ref else ""

        # Check footer patterns
        is_footer = False
        for pattern in config.pptx_footer_patterns:
            if pattern in text:
                is_footer = True
                break

        # Also check position (bottom 15% of page)
        if g.y_center > layout.page_height * 0.85:
            is_footer = True

        if is_footer:
            g.semantic_priority = SemanticPriority.OTHER
            footer_geoms.append(g)
        else:
            body_geoms.append(g)

    # Return body first, then footers
    return body_geoms + footer_geoms


def _order_multicolumn(
    geoms: List[BlockGeom],
    layout: PageLayout,
    config: LayoutConfig
) -> List[Dict[str, Any]]:
    """
    Order blocks for multi-column layout with Y-band interleaving.
    """
    stats = layout.stats
    
    # Separate by zone
    header = [g for g in geoms if g.y_center <= layout.header_end_y]
    footer = [g for g in geoms if g.y_center >= layout.footer_start_y]
    body = [g for g in geoms 
            if g.y_center > layout.header_end_y and g.y_center < layout.footer_start_y]
    
    ordered = []
    
    # Header: top-to-bottom, left-to-right
    header.sort(key=lambda g: (g.y0, g.x0))
    ordered.extend(header)
    
    # Body: Y-band interleaved
    ordered.extend(_order_body_bands(body, layout, config))
    
    # Footer: top-to-bottom, left-to-right
    footer.sort(key=lambda g: (g.y0, g.x0))
    ordered.extend(footer)
    
    return [g.block_ref for g in ordered]


def _order_body_bands(
    geoms: List[BlockGeom],
    layout: PageLayout,
    config: LayoutConfig
) -> List[BlockGeom]:
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
        stats.median_height * config.band_height_factor,
        config.y_tolerance * 3
    )

    # Build spanning set using content_key for stable identity
    spanning_keys = {g.content_key() for g in layout.spanning_blocks}

    def get_column(g: BlockGeom) -> int:
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
    
    def get_band(g: BlockGeom) -> int:
        return int(g.y0 / band_height) if band_height > 0 else 0
    
    # Group by band
    bands: Dict[int, List[Tuple[int, BlockGeom]]] = defaultdict(list)
    for g in geoms:
        bands[get_band(g)].append((get_column(g), g))
    
    # Order within each band
    ordered = []
    for band_idx in sorted(bands.keys()):
        band_items = bands[band_idx]
        # Sort by: column, then Y, then X
        band_items.sort(key=lambda item: (item[0], item[1].y0, item[1].x0))
        ordered.extend(g for _, g in band_items)
    
    return ordered


# =============================================================================
# PUBLIC API (Drop-in replacement)
# =============================================================================

def order_page_blocks(
    blocks: List[Dict[str, Any]],
    page_width: float,
    page_height: float,
    config: Optional[LayoutConfig] = None,
    page_num: int = 1
) -> List[Dict[str, Any]]:
    """
    Main entry point: detect layout and order blocks.
    
    DROP-IN REPLACEMENT for PDFToDocGraphParser._order_blocks_deterministically()
    
    Args:
        blocks: List of raw_block dicts from B01 (with "bbox", "text", etc.)
        page_width: Page width in points (from PyMuPDF)
        page_height: Page height in points (from PyMuPDF)
        config: Optional LayoutConfig for tuning
        page_num: Page number (for logging only)
    
    Returns:
        Same blocks list, reordered for correct reading sequence.
    
    Usage in B01_pdf_to_docgraph.py:
        # Replace:
        ordered = self._order_blocks_deterministically(raw_pages[page_num], page_w=page_w)
        
        # With:
        from B_parsing.B04_column_ordering import order_page_blocks
        ordered = order_page_blocks(raw_pages[page_num], page_w, page_h)
    """
    cfg = config or LayoutConfig()
    
    # Detect layout
    layout = detect_layout(blocks, page_width, page_height, cfg)
    
    if cfg.debug_mode:
        logger.debug(
            f"Page {page_num}: {layout.layout_type.value}, "
            f"{layout.num_columns} cols, confidence={layout.confidence:.2f}"
        )
    
    # Order blocks
    return order_by_layout(blocks, layout, cfg)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_layout_info(
    blocks: List[Dict[str, Any]],
    page_width: float,
    page_height: float,
    config: Optional[LayoutConfig] = None
) -> Dict[str, Any]:
    """
    Get layout info without reordering (for debugging).
    """
    cfg = config or LayoutConfig()
    layout = detect_layout(blocks, page_width, page_height, cfg)

    # Compute XY-Cut++ stats
    geoms = [extract_block_geometry(b, cfg) for b in blocks]
    geoms = [g for g in geoms if g is not None]
    stats = PageStats.compute(geoms, page_width, page_height) if geoms else None

    info = {
        "layout_type": layout.layout_type.value,
        "num_columns": layout.num_columns,
        "num_gutters": len(layout.gutters),
        "gutter_positions": [round(g.center, 1) for g in layout.gutters],
        "num_spanning": len(layout.spanning_blocks),
        "num_column_bound": len(layout.column_blocks),
        "confidence": round(layout.confidence, 3),
        "header_end_y": round(layout.header_end_y, 1),
        "footer_start_y": round(layout.footer_start_y, 1),
        "total_blocks": len(blocks),
    }

    # Add XY-Cut++ metrics
    if stats:
        info["xy_cut_metrics"] = {
            "density_ratio": round(stats.density_ratio, 3),
            "use_xy_cut": stats.should_use_xy_cut(cfg.density_ratio_threshold),
            "adaptive_spanning_threshold": round(stats.adaptive_spanning_threshold(cfg.spanning_beta), 1),
            "median_bbox_length": round(stats.median_bbox_length, 1),
            "median_width": round(stats.median_width, 1),
            "median_height": round(stats.median_height, 1),
        }

    # Add L-shaped detection count
    l_shaped_count = sum(1 for g in geoms if g.is_l_shaped)
    if l_shaped_count > 0:
        info["l_shaped_blocks"] = l_shaped_count

    return info


def analyze_pdf_layout(
    pdf_path: str,
    config: Optional[LayoutConfig] = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Analyze layout of each page in a PDF (diagnostic tool).

    Args:
        pdf_path: Path to PDF file
        config: Optional LayoutConfig
        verbose: Print results to stdout

    Returns:
        List of layout info dicts, one per page
    """
    from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser

    cfg = config or LayoutConfig(debug_mode=True)
    parser = PDFToDocGraphParser(config={"use_sota_layout": True})

    # Parse PDF to get raw pages
    import fitz
    doc = fitz.open(pdf_path)

    results = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_w = page.rect.width
        page_h = page.rect.height

        # Get blocks from parser (simplified - just for layout analysis)
        # Use Unstructured if available, else basic extraction
        try:
            from unstructured.partition.pdf import partition_pdf
            elements = partition_pdf(
                pdf_path,
                strategy="hi_res",
                include_page_breaks=True,
            )
            # Filter to current page
            page_elements = [e for e in elements if getattr(e.metadata, 'page_number', 0) == page_num + 1]

            blocks = []
            for elem in page_elements:
                coords = getattr(elem.metadata, 'coordinates', None)
                if coords and hasattr(coords, 'points'):
                    pts = coords.points
                    x0 = min(p[0] for p in pts)
                    y0 = min(p[1] for p in pts)
                    x1 = max(p[0] for p in pts)
                    y1 = max(p[1] for p in pts)
                    blocks.append({
                        "text": str(elem),
                        "bbox": (x0, y0, x1, y1),
                        "category": elem.category,
                        "zone": "BODY",
                    })
        except Exception:
            # Fallback to PyMuPDF blocks
            blocks = []
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") == 0:  # Text block
                    bbox = block["bbox"]
                    text = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text += span.get("text", "") + " "
                    blocks.append({
                        "text": text.strip(),
                        "bbox": bbox,
                        "category": None,
                        "zone": "BODY",
                    })

        # Get layout info
        info = get_layout_info(blocks, page_w, page_h, cfg)
        info["page_num"] = page_num + 1
        info["page_size"] = f"{page_w:.0f}x{page_h:.0f}"
        results.append(info)

        if verbose:
            print(f"\n{'='*60}")
            print(f"PAGE {page_num + 1} ({page_w:.0f}x{page_h:.0f})")
            print(f"{'='*60}")
            print(f"  Layout: {info['layout_type']} ({info['num_columns']} columns)")
            print(f"  Blocks: {info['total_blocks']} total, {info['num_spanning']} spanning, {info['num_column_bound']} column-bound")
            if info['num_gutters'] > 0:
                print(f"  Gutters: {info['gutter_positions']}")
            if 'xy_cut_metrics' in info:
                m = info['xy_cut_metrics']
                print(f"  XY-Cut++: density_ratio={m['density_ratio']}, use_xy={m['use_xy_cut']}")
                print(f"            adaptive_threshold={m['adaptive_spanning_threshold']}")
            if 'l_shaped_blocks' in info:
                print(f"  L-shaped blocks: {info['l_shaped_blocks']}")

    doc.close()

    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        layout_counts = {}
        for r in results:
            lt = r['layout_type']
            layout_counts[lt] = layout_counts.get(lt, 0) + 1
        for lt, count in sorted(layout_counts.items()):
            print(f"  {lt}: {count} pages")

    return results


def create_config(
    document_type: str = "default",
    **overrides
) -> LayoutConfig:
    """
    Create configuration for specific document types.

    Args:
        document_type: "academic", "clinical", "regulatory", "newsletter", "pptx", "default"
        **overrides: Override any LayoutConfig field

    Returns:
        LayoutConfig instance
    """
    presets = {
        "academic": {
            "min_blocks_per_column": 4,
            "header_zone_pct": 0.15,
            "enable_three_column": False,
        },
        "clinical": {
            "min_blocks_per_column": 3,
            "footer_zone_pct": 0.12,
            "spanning_width_pct": 0.50,
        },
        "regulatory": {
            "min_blocks_per_column": 5,
            "enable_three_column": False,
        },
        "newsletter": {
            "min_blocks_per_column": 2,
            "enable_three_column": True,
            "three_col_min_blocks": 2,
        },
        "pptx": {
            "pptx_mode": True,
            "enable_three_column": False,
            "min_blocks_per_column": 2,
            "header_zone_pct": 0.15,
            "footer_zone_pct": 0.15,
        },
        "default": {},
    }

    preset = presets.get(document_type, {})
    preset.update(overrides)

    return LayoutConfig(**preset)


# =============================================================================
# INTEGRATION HELPER: Direct replacement in B01
# =============================================================================

class ColumnOrderingMixin:
    """
    Mixin class for PDFToDocGraphParser (XY-Cut++ enhanced).

    Usage:
        class PDFToDocGraphParser(BaseParser, ColumnOrderingMixin):
            ...

        # Then in parse():
        ordered = self.order_blocks_sota(raw_pages[page_num], page_w, page_h)

    Note: Uses instance attribute via getattr() to avoid class-level shared state.
    """

    def order_blocks_sota(
        self,
        blocks: List[Dict[str, Any]],
        page_width: float,
        page_height: float,
        page_num: int = 1
    ) -> List[Dict[str, Any]]:
        """
        SOTA replacement for _order_blocks_deterministically.

        Uses XY-Cut++ with:
        - Density-driven axis selection
        - Adaptive spanning threshold
        - L-shaped region pre-masking
        - Semantic priority ordering
        - PPTX mode support
        """
        # Use instance attribute to avoid shared state across instances
        config = getattr(self, '_layout_config', None) or LayoutConfig()
        return order_page_blocks(blocks, page_width, page_height, config, page_num)

    def set_layout_config(self, config: LayoutConfig) -> None:
        """Set layout configuration for this parser instance."""
        self._layout_config = config

    def set_document_type(self, document_type: str) -> None:
        """Set layout configuration from document type preset."""
        self._layout_config = create_config(document_type)
