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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

# Import from your core modules

try:
    from A_core.A01_domain_models import BoundingBox
except ImportError:
    # Fallback for standalone testing
    BoundingBox = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

# Import from column detection submodule
from B_parsing.B04a_column_detection import (
    PageStats,
    Gutter,
    find_gutters,
    find_columns_by_clustering,
    detect_l_shaped_regions,
    detect_spanning,
)

# Import from XY-Cut ordering submodule
from B_parsing.B04b_xy_cut_ordering import (
    xy_cut_order,
    order_body_bands,
)


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

    CROSS_LAYOUT = 0  # Spanning elements (headers, full-width)
    TITLE = 1  # Titles, section headers
    VISION = 2  # Tables, figures, images
    NARRATIVE = 3  # Body text, paragraphs
    OTHER = 4  # Footers, captions, misc


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
    gutter_min_width_factor: float = 1.5  # Multiplier of median horizontal gap
    min_gutter_width_abs: float = 8.0  # Absolute minimum gutter width (points)
    # Academic journals often have 8-12pt gutters

    # -------------------------------------------------------------------------
    # Column Validation
    # -------------------------------------------------------------------------
    min_blocks_per_column: int = 3
    min_column_width_pct: float = 0.18  # Min column as fraction of page
    max_column_width_pct: float = 0.58  # Max single column width

    # -------------------------------------------------------------------------
    # Spanning Element Detection (fallback if adaptive disabled)
    # -------------------------------------------------------------------------
    spanning_width_pct: float = 0.55  # Width > this% = spanning
    spanning_overlap_ratio: float = 0.25  # Overlap ratio to consider spanning

    # -------------------------------------------------------------------------
    # Zone Detection (% of page height)
    # -------------------------------------------------------------------------
    header_zone_pct: float = 0.12  # Top 12% is header zone
    footer_zone_pct: float = 0.10  # Bottom 10% is footer zone

    # -------------------------------------------------------------------------
    # Y-band Parameters
    # -------------------------------------------------------------------------
    y_tolerance: float = 5.0  # Blocks within this Y are "same line"
    band_height_factor: float = 1.8  # Multiplier of median block height

    # -------------------------------------------------------------------------
    # XY-Cut Parameters
    # -------------------------------------------------------------------------
    xy_cut_min_elements: int = 2
    min_gap_factor: float = 0.3  # Min gap as fraction of median gap

    # -------------------------------------------------------------------------
    # Column Search
    # -------------------------------------------------------------------------
    column_search_margin: float = 0.25  # Search for gutters in middle 50%

    # -------------------------------------------------------------------------
    # Three-column Support
    # -------------------------------------------------------------------------
    enable_three_column: bool = True
    three_col_min_blocks: int = 3

    # -------------------------------------------------------------------------
    # PPTX-to-PDF Mode
    # -------------------------------------------------------------------------
    pptx_mode: bool = False
    pptx_invert_z_order: bool = True  # PowerPoint reading order is inverted
    pptx_footer_patterns: Set[str] = field(
        default_factory=lambda: {
            "slide",
            "page",
            "©",
            "copyright",
            "confidential",
            "footer",
            "all rights reserved",
            "proprietary",
        }
    )
    pptx_aspect_ratio_range: Tuple[float, float] = (1.2, 1.9)  # 4:3 to 16:9

    # -------------------------------------------------------------------------
    # Unstructured.io Categories
    # -------------------------------------------------------------------------
    use_unstructured_categories: bool = True
    title_categories: Set[str] = field(
        default_factory=lambda: {"title", "header", "headline", "sectionheader"}
    )
    table_categories: Set[str] = field(
        default_factory=lambda: {
            "table",
            "tablecell",
            "figure",
            "image",
            "figurecaption",
        }
    )
    narrative_categories: Set[str] = field(
        default_factory=lambda: {"narrativetext", "text", "listitem", "paragraph"}
    )

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
    block: Dict[str, Any], config: Optional[LayoutConfig] = None
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
    if hasattr(bbox, "coords"):
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
    config: Optional[LayoutConfig] = None,
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
    footer_start_y: float = float("inf")
    stats: Optional[PageStats] = None
    confidence: float = 0.0

    @property
    def num_columns(self) -> int:
        return len(self.gutters) + 1 if self.gutters else 1


def detect_layout(
    blocks: List[Dict[str, Any]],
    page_width: float,
    page_height: float,
    config: Optional[LayoutConfig] = None,
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
        if (
            cfg.pptx_aspect_ratio_range[0]
            <= aspect_ratio
            <= cfg.pptx_aspect_ratio_range[1]
        ):
            # Landscape aspect ratio suggests PPTX
            cfg = LayoutConfig(**{**cfg.__dict__, "pptx_mode": True})
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
        g
        for g in geoms
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
    layout.confidence = (
        min(g.confidence for g in layout.gutters) if layout.gutters else 1.0
    )

    # Check for mixed header (spanning elements in header area)
    if spanning and layout.num_columns > 1:
        header_spans = [
            g
            for g in spanning
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
    config: Optional[LayoutConfig] = None,
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
    geoms: List[BlockGeom], layout: PageLayout, config: LayoutConfig
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
    geoms: List[BlockGeom], layout: PageLayout, config: LayoutConfig
) -> List[Dict[str, Any]]:
    """
    Order blocks for multi-column layout with Y-band interleaving.
    """
    # Separate by zone
    header = [g for g in geoms if g.y_center <= layout.header_end_y]
    footer = [g for g in geoms if g.y_center >= layout.footer_start_y]
    body = [
        g
        for g in geoms
        if g.y_center > layout.header_end_y and g.y_center < layout.footer_start_y
    ]

    ordered = []

    # Header: top-to-bottom, left-to-right
    header.sort(key=lambda g: (g.y0, g.x0))
    ordered.extend(header)

    # Body: Y-band interleaved
    ordered.extend(order_body_bands(body, layout, config))

    # Footer: top-to-bottom, left-to-right
    footer.sort(key=lambda g: (g.y0, g.x0))
    ordered.extend(footer)

    return [g.block_ref for g in ordered]


# =============================================================================
# PUBLIC API (Drop-in replacement)
# =============================================================================


def order_page_blocks(
    blocks: List[Dict[str, Any]],
    page_width: float,
    page_height: float,
    config: Optional[LayoutConfig] = None,
    page_num: int = 1,
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
    config: Optional[LayoutConfig] = None,
) -> Dict[str, Any]:
    """
    Get layout info without reordering (for debugging).
    """
    cfg = config or LayoutConfig()
    layout = detect_layout(blocks, page_width, page_height, cfg)

    # Compute XY-Cut++ stats
    geoms_raw = [extract_block_geometry(b, cfg) for b in blocks]
    geoms: List[BlockGeom] = [g for g in geoms_raw if g is not None]
    stats = PageStats.compute(geoms, page_width, page_height) if geoms else None

    info: dict[str, Any] = {
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
            "adaptive_spanning_threshold": round(
                stats.adaptive_spanning_threshold(cfg.spanning_beta), 1
            ),
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
    pdf_path: str, config: Optional[LayoutConfig] = None, verbose: bool = True
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
    cfg = config or LayoutConfig(debug_mode=True)

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
            page_elements = [
                e
                for e in elements
                if getattr(e.metadata, "page_number", 0) == page_num + 1
            ]

            blocks = []
            for elem in page_elements:
                coords = getattr(elem.metadata, "coordinates", None)
                if coords and hasattr(coords, "points"):
                    pts = coords.points
                    x0 = min(p[0] for p in pts)
                    y0 = min(p[1] for p in pts)
                    x1 = max(p[0] for p in pts)
                    y1 = max(p[1] for p in pts)
                    blocks.append(
                        {
                            "text": str(elem),
                            "bbox": (x0, y0, x1, y1),
                            "category": elem.category,
                            "zone": "BODY",
                        }
                    )
        except Exception as e:
            # Fallback to PyMuPDF blocks
            logger.debug("Element processing failed, falling back to PyMuPDF: %s", e)
            blocks = []
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") == 0:  # Text block
                    bbox = block["bbox"]
                    text = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text += span.get("text", "") + " "
                    blocks.append(
                        {
                            "text": text.strip(),
                            "bbox": bbox,
                            "category": None,  # type: ignore[dict-item]
                            "zone": "BODY",
                        }
                    )

        # Get layout info
        info = get_layout_info(blocks, page_w, page_h, cfg)
        info["page_num"] = page_num + 1
        info["page_size"] = f"{page_w:.0f}x{page_h:.0f}"
        results.append(info)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"PAGE {page_num + 1} ({page_w:.0f}x{page_h:.0f})")
            print(f"{'=' * 60}")
            print(f"  Layout: {info['layout_type']} ({info['num_columns']} columns)")
            print(
                f"  Blocks: {info['total_blocks']} total, {info['num_spanning']} spanning, {info['num_column_bound']} column-bound"
            )
            if info["num_gutters"] > 0:
                print(f"  Gutters: {info['gutter_positions']}")
            if "xy_cut_metrics" in info:
                m = info["xy_cut_metrics"]
                print(
                    f"  XY-Cut++: density_ratio={m['density_ratio']}, use_xy={m['use_xy_cut']}"
                )
                print(
                    f"            adaptive_threshold={m['adaptive_spanning_threshold']}"
                )
            if "l_shaped_blocks" in info:
                print(f"  L-shaped blocks: {info['l_shaped_blocks']}")

    doc.close()

    if verbose:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        layout_counts: dict[str, int] = {}
        for r in results:
            lt = r["layout_type"]
            layout_counts[lt] = layout_counts.get(lt, 0) + 1
        for lt, count in sorted(layout_counts.items()):
            print(f"  {lt}: {count} pages")

    return results


def create_config(document_type: str = "default", **overrides) -> LayoutConfig:
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

    preset: dict = dict(presets.get(document_type, {}) or {})  # type: ignore[call-overload]
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
        page_num: int = 1,
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
        config = getattr(self, "_layout_config", None) or LayoutConfig()
        return order_page_blocks(blocks, page_width, page_height, config, page_num)

    def set_layout_config(self, config: LayoutConfig) -> None:
        """Set layout configuration for this parser instance."""
        self._layout_config = config

    def set_document_type(self, document_type: str) -> None:
        """Set layout configuration from document type preset."""
        self._layout_config = create_config(document_type)
