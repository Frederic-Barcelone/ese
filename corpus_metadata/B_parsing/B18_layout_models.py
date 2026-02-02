# corpus_metadata/B_parsing/B18_layout_models.py
"""
Layout models for layout-aware visual extraction.

Provides data structures for:
- LayoutPattern: Page layout patterns (single column, two-column, etc.)
- VisualPosition: Horizontal position codes for visuals
- VisualZone: Description of a visual element's location on a page
- PageLayout: Complete layout information for a single page

These models are used by the VLM layout analyzer to describe where
visuals (tables, figures) are located on a page, enabling the zone
expander to compute precise bounding boxes using whitespace detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class LayoutPattern(str, Enum):
    """
    Page layout pattern codes.

    Used to describe the overall column structure of a page.
    The pattern affects how horizontal zones are interpreted.
    """

    FULL = "full"  # Single column layout
    TWO_COL = "2col"  # Two-column layout
    TWO_COL_FULLBOT = "2col-fullbot"  # Two-column top, full-width bottom
    FULLTOP_TWO_COL = "fulltop-2col"  # Full-width top, two-column bottom


class VisualPosition(str, Enum):
    """
    Horizontal position codes for visuals within a page layout.

    Single-character codes for compact filename encoding.
    """

    LEFT = "L"  # Left column (in two-column layouts)
    RIGHT = "R"  # Right column (in two-column layouts)
    FULL = "F"  # Full width (spans entire page width)


@dataclass
class VisualZone:
    """
    Description of a visual element's location on a page.

    Represents the approximate zone where a table or figure is located,
    as identified by the VLM. The zone expander will later compute
    precise bounding boxes by finding whitespace boundaries.

    Attributes:
        visual_type: Type of visual - "table" or "figure"
        label: Optional label like "Table 1" or "Figure 2"
        position: Horizontal position (LEFT, RIGHT, or FULL)
        vertical_zone: Vertical location, either named ("top", "middle",
            "bottom") or as a normalized range ("0.2-0.6")
        confidence: VLM's confidence in this detection (0.0 to 1.0)
        caption_snippet: First few words of the caption if detected
        is_continuation: True if this continues from previous page
        continues_next: True if this continues to next page
    """

    visual_type: str  # "table" or "figure"
    position: VisualPosition
    vertical_zone: str  # "top", "middle", "bottom", or "0.2-0.6" format

    label: Optional[str] = None
    confidence: float = 0.9
    caption_snippet: Optional[str] = None
    is_continuation: bool = False
    continues_next: bool = False


@dataclass
class PageLayout:
    """
    Complete layout information for a single page.

    Combines the overall page pattern with a list of visual zones
    and margin information. Used as input to the zone expander.

    Attributes:
        page_num: 1-based page number
        pattern: Layout pattern (FULL, TWO_COL, etc.)
        column_boundary: Normalized x-coordinate of column separator
            (only relevant for two-column layouts)
        margin_left: Normalized left margin (default 0.05 = 5%)
        margin_right: Normalized right margin (default 0.95 = 95%)
        visuals: List of visual zones detected on this page
        raw_vlm_response: Raw JSON response from VLM for debugging
    """

    page_num: int
    pattern: LayoutPattern

    column_boundary: Optional[float] = None
    margin_left: float = 0.05
    margin_right: float = 0.95
    visuals: List[VisualZone] = field(default_factory=list)
    raw_vlm_response: Optional[str] = None
