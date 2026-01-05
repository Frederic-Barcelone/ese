# B_parsing/B04_column_ordering.py
"""
SOTA Column Layout Detection and Reading Order Module
=====================================================

Compatible with Unstructured.io (hi_res, fast, auto strategies)


IMPLEMENTS:
    - XY-Cut++ style hierarchical segmentation
    - Whitespace-based gutter detection (Breuel method)
    - Cross-layout element detection (spanning elements)
    - Density-driven axis selection
    - Y-band interleaving for multi-column
    - Per-page adaptive layout detection

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

VERSION: 2.1.0 (Unstructured.io Compatible)
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set
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


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LayoutConfig:
    """
    Configuration for layout detection.
    Defaults tuned for Unstructured.io hi_res output on academic/clinical docs.
    """
    # Gutter detection (adaptive based on content)
    gutter_min_width_factor: float = 1.5      # Multiplier of median horizontal gap
    min_gutter_width_abs: float = 15.0        # Absolute minimum gutter width (points)
    
    # Column validation
    min_blocks_per_column: int = 3
    min_column_width_pct: float = 0.18        # Min column as fraction of page
    max_column_width_pct: float = 0.58        # Max single column width
    
    # Spanning element detection
    spanning_width_pct: float = 0.55          # Width > this% = spanning
    spanning_overlap_ratio: float = 0.25      # Overlap ratio to consider spanning
    
    # Zone detection (% of page height)
    header_zone_pct: float = 0.12             # Top 12% is header zone
    footer_zone_pct: float = 0.10             # Bottom 10% is footer zone
    
    # Y-band parameters
    y_tolerance: float = 5.0                  # Blocks within this Y are "same line"
    band_height_factor: float = 1.8           # Multiplier of median block height
    
    # XY-Cut parameters
    xy_cut_min_elements: int = 2
    
    # Column search region (middle portion of page)
    column_search_margin: float = 0.25        # Search for gutters in middle 50%
    
    # Three-column
    enable_three_column: bool = True
    three_col_min_blocks: int = 3
    
    # Unstructured.io specific
    use_unstructured_categories: bool = True  # Use element categories for hints
    title_categories: Set[str] = field(default_factory=lambda: {
        "title", "header", "headline"
    })
    table_categories: Set[str] = field(default_factory=lambda: {
        "table", "tablecell"
    })
    
    # Debug
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


def extract_block_geometry(block: Dict[str, Any]) -> Optional[BlockGeom]:
    """
    Extract geometry from B01 raw_block dict.
    
    Handles both BoundingBox objects and raw coordinate tuples.
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
        category=block.get("category"),
        is_section_header=block.get("is_section_header", False),
    )


# =============================================================================
# STATISTICS
# =============================================================================

@dataclass
class PageStats:
    """Computed statistics for adaptive thresholds."""
    page_width: float
    page_height: float
    
    # Block statistics
    median_width: float
    median_height: float
    median_x_gap: float
    median_y_gap: float
    
    # Content bounds
    content_left: float
    content_right: float
    content_top: float
    content_bottom: float
    
    # Counts
    num_blocks: int
    
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
# SPANNING DETECTION
# =============================================================================

def detect_spanning(
    geoms: List[BlockGeom],
    gutters: List[Gutter],
    stats: PageStats,
    config: LayoutConfig
) -> Tuple[List[BlockGeom], List[BlockGeom]]:
    """
    Separate spanning elements from column-bound elements.
    
    Returns:
        (spanning_blocks, column_blocks)
    """
    if not gutters:
        return geoms.copy(), []
    
    spanning = []
    column_bound = []
    
    span_threshold = stats.page_width * config.spanning_width_pct
    
    for geom in geoms:
        is_spanning = False
        
        # Check 1: Width-based
        if geom.width >= span_threshold:
            is_spanning = True
        
        # Check 2: Crosses gutter with significant overlap
        if not is_spanning:
            for gutter in gutters:
                if geom.x0 < gutter.center < geom.x1:
                    overlap_left = gutter.center - geom.x0
                    overlap_right = geom.x1 - gutter.center
                    min_overlap = geom.width * config.spanning_overlap_ratio
                    
                    if overlap_left > min_overlap and overlap_right > min_overlap:
                        is_spanning = True
                        break
        
        # Check 3: Title/Header categories often span (Unstructured hint)
        if not is_spanning and config.use_unstructured_categories:
            if geom.category and geom.category.lower() in config.title_categories:
                if geom.width >= stats.page_width * 0.4:
                    is_spanning = True
        
        if is_spanning:
            spanning.append(geom)
        else:
            column_bound.append(geom)
    
    return spanning, column_bound


# =============================================================================
# XY-CUT ORDERING
# =============================================================================

def xy_cut_order(
    geoms: List[BlockGeom],
    stats: PageStats,
    config: LayoutConfig
) -> List[BlockGeom]:
    """
    Order blocks using recursive XY-Cut with density-driven axis selection.
    """
    if len(geoms) <= 1:
        return geoms
    
    return _xy_cut_recursive(geoms, stats, config)


def _xy_cut_recursive(
    geoms: List[BlockGeom],
    stats: PageStats,
    config: LayoutConfig
) -> List[BlockGeom]:
    """Recursive XY-Cut implementation."""
    if len(geoms) <= config.xy_cut_min_elements:
        return sorted(geoms, key=lambda g: (g.y0, g.x0))
    
    # Find best cuts
    x_cut = _find_x_cut(geoms, stats)
    y_cut = _find_y_cut(geoms, stats)
    
    if x_cut is None and y_cut is None:
        return sorted(geoms, key=lambda g: (g.y0, g.x0))
    
    # Choose axis with better separation
    # Prefer Y-cuts (top-to-bottom reading) when quality is similar
    if x_cut is None:
        cut_pos, axis = y_cut, 'y'
    elif y_cut is None:
        cut_pos, axis = x_cut, 'x'
    else:
        x_score = _cut_quality(geoms, x_cut, 'x')
        y_score = _cut_quality(geoms, y_cut, 'y')
        
        # Bias toward Y-cuts for reading order
        if y_score >= x_score * 0.8:
            cut_pos, axis = y_cut, 'y'
        else:
            cut_pos, axis = x_cut, 'x'
    
    # Partition and recurse
    if axis == 'x':
        left = [g for g in geoms if g.x_center < cut_pos]
        right = [g for g in geoms if g.x_center >= cut_pos]
        return (_xy_cut_recursive(left, stats, config) +
                _xy_cut_recursive(right, stats, config))
    else:
        top = [g for g in geoms if g.y_center < cut_pos]
        bottom = [g for g in geoms if g.y_center >= cut_pos]
        return (_xy_cut_recursive(top, stats, config) +
                _xy_cut_recursive(bottom, stats, config))


def _find_x_cut(geoms: List[BlockGeom], stats: PageStats) -> Optional[float]:
    """Find best vertical cut."""
    if len(geoms) < 2:
        return None
    
    sorted_by_x = sorted(geoms, key=lambda g: g.x1)
    best_gap, best_pos = 0, None
    min_gap = stats.median_x_gap * 0.3
    
    for i in range(len(sorted_by_x) - 1):
        gap = sorted_by_x[i + 1].x0 - sorted_by_x[i].x1
        if gap > best_gap and gap >= min_gap:
            best_gap = gap
            best_pos = (sorted_by_x[i].x1 + sorted_by_x[i + 1].x0) / 2
    
    return best_pos


def _find_y_cut(geoms: List[BlockGeom], stats: PageStats) -> Optional[float]:
    """Find best horizontal cut."""
    if len(geoms) < 2:
        return None
    
    sorted_by_y = sorted(geoms, key=lambda g: g.y1)
    best_gap, best_pos = 0, None
    min_gap = stats.median_y_gap * 0.3
    
    for i in range(len(sorted_by_y) - 1):
        gap = sorted_by_y[i + 1].y0 - sorted_by_y[i].y1
        if gap > best_gap and gap >= min_gap:
            best_gap = gap
            best_pos = (sorted_by_y[i].y1 + sorted_by_y[i + 1].y0) / 2
    
    return best_pos


def _cut_quality(geoms: List[BlockGeom], cut_pos: float, axis: str) -> float:
    """Compute cut quality score."""
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
    Detect page layout from B01 raw_blocks.
    """
    cfg = config or LayoutConfig()
    
    # Extract geometries
    geoms = []
    for block in blocks:
        g = extract_block_geometry(block)
        if g:
            geoms.append(g)
    
    # Compute stats
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
    
    # Filter to body zone for column detection
    body_geoms = [
        g for g in geoms
        if g.y_center > layout.header_end_y and g.y_center < layout.footer_start_y
    ]
    
    if len(body_geoms) < cfg.min_blocks_per_column * 2:
        return layout
    
    # Find gutters
    gutters = find_gutters(body_geoms, stats, cfg)
    
    if not gutters:
        return layout
    
    # Detect spanning elements
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
    Order blocks according to detected layout.
    """
    cfg = config or LayoutConfig()
    
    if not blocks:
        return []
    
    # Extract geometries
    geom_map: Dict[int, BlockGeom] = {}
    for block in blocks:
        g = extract_block_geometry(block)
        if g:
            geom_map[id(block)] = g
    
    if not geom_map:
        return blocks
    
    geoms = list(geom_map.values())
    
    # Single column: XY-Cut ordering
    if layout.layout_type == LayoutType.SINGLE_COLUMN:
        ordered = xy_cut_order(geoms, layout.stats, cfg)
        return [g.block_ref for g in ordered]
    
    # Multi-column: zone-aware, band-interleaved
    return _order_multicolumn(geoms, layout, cfg)


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
    Order body blocks using Y-band interleaving.
    
    Key: Process bands top-to-bottom.
    Within each band: spanning → left → center → right
    """
    if not geoms:
        return []
    
    stats = layout.stats
    band_height = max(
        stats.median_height * config.band_height_factor,
        config.y_tolerance * 3
    )
    
    # Build spanning set for quick lookup
    spanning_ids = {id(g.block_ref) for g in layout.spanning_blocks}
    
    def get_column(g: BlockGeom) -> int:
        """0=spanning, 1=left, 2=center, 3=right"""
        if id(g.block_ref) in spanning_ids:
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
    layout = detect_layout(blocks, page_width, page_height, config)
    
    return {
        "layout_type": layout.layout_type.value,
        "num_columns": layout.num_columns,
        "num_gutters": len(layout.gutters),
        "gutter_positions": [round(g.center, 1) for g in layout.gutters],
        "num_spanning": len(layout.spanning_blocks),
        "num_column_bound": len(layout.column_blocks),
        "confidence": round(layout.confidence, 3),
        "header_end_y": round(layout.header_end_y, 1),
        "footer_start_y": round(layout.footer_start_y, 1),
    }


def create_config(
    document_type: str = "default",
    **overrides
) -> LayoutConfig:
    """
    Create configuration for specific document types.
    
    Args:
        document_type: "academic", "clinical", "regulatory", "newsletter", "default"
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
    Mixin class for PDFToDocGraphParser.
    
    Usage:
        class PDFToDocGraphParser(BaseParser, ColumnOrderingMixin):
            ...
            
        # Then in parse():
        ordered = self.order_blocks_sota(raw_pages[page_num], page_w, page_h)
    """
    
    _layout_config: Optional[LayoutConfig] = None
    
    def order_blocks_sota(
        self,
        blocks: List[Dict[str, Any]],
        page_width: float,
        page_height: float
    ) -> List[Dict[str, Any]]:
        """SOTA replacement for _order_blocks_deterministically."""
        if self._layout_config is None:
            self._layout_config = LayoutConfig()
        
        return order_page_blocks(blocks, page_width, page_height, self._layout_config)
    
    def set_layout_config(self, config: LayoutConfig) -> None:
        """Set layout configuration."""
        self._layout_config = config