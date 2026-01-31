# corpus_metadata/B_parsing/B15_caption_extractor.py
"""
Multisource Caption Extraction for Visual Pipeline.

Extracts captions using multiple sources in priority order:
1. PDF text blocks (preferred for born-digital documents)
2. OCR fallback (for scanned documents)
3. VLM extraction (handled downstream)

Key improvements over B10_caption_detector:
- Point-based search zones (not pixels)
- Multisource extraction with provenance tracking
- Handles above/below/left/right positions
- 2-column layout support
- Comprehensive caption patterns for clinical documents
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from A_core.A13_visual_models import (
    CaptionCandidate,
    CaptionProvenance,
    CaptionSearchZones,
    ReferenceSource,
    VisualReference,
)


# -------------------------
# Caption Patterns
# -------------------------


# Standard figure patterns
FIGURE_PATTERNS = [
    # "Figure 1", "Fig. 1", "Fig 1"
    re.compile(r"^(?:Figure|Fig\.?)\s*(\d+)(?:[.-](\d+))?([A-Za-z])?\.?\s*(.*)", re.IGNORECASE),
    # "FIGURE 1" (uppercase)
    re.compile(r"^FIGURE\s*(\d+)(?:[.-](\d+))?([A-Za-z])?\s*(.*)", re.IGNORECASE),
]

# Standard table patterns
TABLE_PATTERNS = [
    # "Table 1", "TABLE 1"
    re.compile(r"^(?:Table)\s*(\d+)(?:[.-](\d+))?([A-Za-z])?\.?\s*(.*)", re.IGNORECASE),
]

# Exhibit patterns (regulatory documents)
EXHIBIT_PATTERNS = [
    re.compile(r"^(?:Exhibit)\s*(\d+)(?:[.-](\d+))?([A-Za-z])?\.?\s*(.*)", re.IGNORECASE),
]

# Appendix patterns
APPENDIX_PATTERNS = [
    re.compile(r"^(?:Appendix)\s*([A-Za-z\d]+)(?:[.-](\d+))?\s*(.*)", re.IGNORECASE),
]

# Continuation patterns (any type)
CONTINUATION_PATTERNS = [
    re.compile(r"\(cont(?:inued)?\.?\)", re.IGNORECASE),
    re.compile(r"\(cont['']?d\.?\)", re.IGNORECASE),
]

# All patterns grouped by type
CAPTION_PATTERNS: Dict[str, List[re.Pattern]] = {
    "figure": FIGURE_PATTERNS,
    "table": TABLE_PATTERNS,
    "exhibit": EXHIBIT_PATTERNS,
    "appendix": APPENDIX_PATTERNS,
}


# -------------------------
# Helper Functions
# -------------------------


def parse_reference_from_match(
    match: re.Match,
    type_label: str,
    source: ReferenceSource,
) -> VisualReference:
    """
    Parse a VisualReference from a regex match.

    Args:
        match: Regex match with groups (number, range_end, suffix, caption_text)
        type_label: Type label (Figure, Table, etc.)
        source: Where the reference was parsed from

    Returns:
        VisualReference object
    """
    number_str = match.group(1)
    range_end_str = match.group(2) if match.lastindex >= 2 else None
    suffix = match.group(3) if match.lastindex >= 3 else None

    # Parse number(s)
    try:
        start_num = int(number_str)
    except ValueError:
        # Handle alphanumeric appendix references
        start_num = 0

    numbers = [start_num]
    is_range = False

    if range_end_str:
        try:
            end_num = int(range_end_str)
            numbers = list(range(start_num, end_num + 1))
            is_range = True
        except ValueError:
            pass

    # Build raw string
    raw_parts = [type_label, number_str]
    if range_end_str:
        raw_parts.append(f"-{range_end_str}")
    if suffix:
        raw_parts.append(suffix)
    raw_string = " ".join(raw_parts[:2])
    if len(raw_parts) > 2:
        raw_string = raw_parts[0] + " " + "".join(raw_parts[1:])

    return VisualReference(
        raw_string=raw_string,
        type_label=type_label.capitalize(),
        numbers=numbers if numbers[0] > 0 else [1],
        is_range=is_range,
        suffix=suffix,
        source=source,
    )


def has_horizontal_overlap(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
    min_overlap_ratio: float = 0.3,
) -> bool:
    """
    Check if two bboxes have horizontal overlap.

    Args:
        bbox1: First bbox (x0, y0, x1, y1)
        bbox2: Second bbox (x0, y0, x1, y1)
        min_overlap_ratio: Minimum overlap as fraction of smaller width

    Returns:
        True if significant horizontal overlap exists
    """
    x1_start, _, x1_end, _ = bbox1
    x2_start, _, x2_end, _ = bbox2

    overlap_start = max(x1_start, x2_start)
    overlap_end = min(x1_end, x2_end)

    if overlap_end <= overlap_start:
        return False

    overlap = overlap_end - overlap_start
    min_width = min(x1_end - x1_start, x2_end - x2_start)

    return overlap / max(min_width, 1) >= min_overlap_ratio


def has_vertical_overlap(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
    min_overlap_ratio: float = 0.3,
) -> bool:
    """
    Check if two bboxes have vertical overlap.

    Args:
        bbox1: First bbox (x0, y0, x1, y1)
        bbox2: Second bbox (x0, y0, x1, y1)
        min_overlap_ratio: Minimum overlap as fraction of smaller height

    Returns:
        True if significant vertical overlap exists
    """
    _, y1_start, _, y1_end = bbox1
    _, y2_start, _, y2_end = bbox2

    overlap_start = max(y1_start, y2_start)
    overlap_end = min(y1_end, y2_end)

    if overlap_end <= overlap_start:
        return False

    overlap = overlap_end - overlap_start
    min_height = min(y1_end - y1_start, y2_end - y2_start)

    return overlap / max(min_height, 1) >= min_overlap_ratio


def get_relative_position(
    visual_bbox: Tuple[float, float, float, float],
    text_bbox: Tuple[float, float, float, float],
    zones: CaptionSearchZones,
) -> Tuple[Optional[str], float]:
    """
    Determine position of text relative to visual and distance.

    Handles 2-column layouts where caption may be to the side.

    Args:
        visual_bbox: Visual bounding box in PDF points (x0, y0, x1, y1)
        text_bbox: Text bounding box in PDF points (x0, y0, x1, y1)
        zones: Search zones in PDF points

    Returns:
        Tuple of (position, distance) or (None, inf) if not in zone
    """
    vx0, vy0, vx1, vy1 = visual_bbox
    tx0, ty0, tx1, ty1 = text_bbox

    # Below visual (most common for figures)
    if ty0 >= vy1 and ty0 <= vy1 + zones.below:
        if has_horizontal_overlap(visual_bbox, text_bbox):
            return "below", ty0 - vy1

    # Above visual (common for tables)
    if ty1 <= vy0 and ty1 >= vy0 - zones.above:
        if has_horizontal_overlap(visual_bbox, text_bbox):
            return "above", vy0 - ty1

    # Left of visual (2-column layouts)
    if tx1 <= vx0 and tx1 >= vx0 - zones.left:
        if has_vertical_overlap(visual_bbox, text_bbox):
            return "left", vx0 - tx1

    # Right of visual (rare but possible)
    if tx0 >= vx1 and tx0 <= vx1 + zones.right:
        if has_vertical_overlap(visual_bbox, text_bbox):
            return "right", tx0 - vx1

    return None, float("inf")


def extract_block_text(block: Dict) -> str:
    """
    Extract text from a PyMuPDF text block dict.

    Args:
        block: Text block from page.get_text("dict")

    Returns:
        Combined text from all lines and spans
    """
    lines = []
    for line in block.get("lines", []):
        spans_text = []
        for span in line.get("spans", []):
            spans_text.append(span.get("text", ""))
        lines.append("".join(spans_text))
    return " ".join(lines).strip()


def detect_caption_pattern(text: str) -> Optional[Tuple[str, re.Match]]:
    """
    Detect if text matches any caption pattern.

    Args:
        text: Text to check

    Returns:
        Tuple of (caption_type, match) or None if no match
    """
    text = text.strip()

    for caption_type, patterns in CAPTION_PATTERNS.items():
        for pattern in patterns:
            match = pattern.match(text)
            if match:
                return caption_type, match

    return None


def is_continuation_caption(text: str) -> bool:
    """Check if caption text indicates a continuation."""
    for pattern in CONTINUATION_PATTERNS:
        if pattern.search(text):
            return True
    return False


# -------------------------
# Column Layout Detection
# -------------------------


@dataclass
class ColumnLayout:
    """Detected column layout for a page."""

    columns: List[Tuple[float, float]]  # List of (x0, x1) column bounds
    page_width: float
    page_height: float

    @property
    def is_single_column(self) -> bool:
        return len(self.columns) <= 1

    @property
    def is_two_column(self) -> bool:
        return len(self.columns) == 2


def infer_column_layout(
    doc: fitz.Document,
    page_num: int,
    gap_threshold_ratio: float = 0.08,
) -> ColumnLayout:
    """
    Infer column boundaries from text block x-distribution.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        gap_threshold_ratio: Gap size as fraction of page width to detect column

    Returns:
        ColumnLayout with detected columns
    """
    page = doc[page_num - 1]
    text_dict = page.get_text("dict")
    page_width = page.rect.width
    page_height = page.rect.height

    # Collect x-ranges of narrow text blocks
    x_ranges: List[Tuple[float, float]] = []

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # Only text blocks
            continue

        bbox = block.get("bbox", (0, 0, 0, 0))
        x0, y0, x1, y1 = bbox
        width = x1 - x0

        # Only consider narrow blocks (likely column content)
        if width < page_width * 0.45:
            x_ranges.append((x0, x1))

    if not x_ranges:
        return ColumnLayout(
            columns=[(0, page_width)],
            page_width=page_width,
            page_height=page_height,
        )

    # Sort by left edge
    x_ranges.sort(key=lambda x: x[0])

    # Find gaps using edge tracking
    gap_threshold = page_width * gap_threshold_ratio
    edges = []
    for x0, x1 in x_ranges:
        edges.append((x0, "start"))
        edges.append((x1, "end"))
    edges.sort(key=lambda e: e[0])

    # Track coverage and find gaps
    coverage = 0
    gaps: List[Tuple[float, float]] = []
    last_end = 0

    for x, edge_type in edges:
        if edge_type == "start":
            if coverage == 0 and x - last_end > gap_threshold:
                gaps.append((last_end, x))
            coverage += 1
        else:
            coverage -= 1
            if coverage == 0:
                last_end = x

    # Build columns from gaps
    if not gaps:
        return ColumnLayout(
            columns=[(0, page_width)],
            page_width=page_width,
            page_height=page_height,
        )

    columns: List[Tuple[float, float]] = []
    margin = 20  # Small margin

    prev_end = 0
    for gap_start, gap_end in gaps:
        if gap_start > prev_end + margin:
            columns.append((max(0, prev_end), gap_start))
        prev_end = gap_end

    # Final column
    if prev_end < page_width - margin:
        columns.append((prev_end, page_width))

    if not columns:
        # Fallback to two-column split
        mid = page_width / 2
        columns = [(0, mid), (mid, page_width)]

    return ColumnLayout(
        columns=columns,
        page_width=page_width,
        page_height=page_height,
    )


def get_column_for_x(x: float, layout: ColumnLayout) -> int:
    """Get the column index for an x coordinate."""
    for idx, (col_x0, col_x1) in enumerate(layout.columns):
        if col_x0 <= x <= col_x1:
            return idx

    # Find nearest column
    min_dist = float("inf")
    best_idx = 0
    for idx, (col_x0, col_x1) in enumerate(layout.columns):
        center = (col_x0 + col_x1) / 2
        dist = abs(x - center)
        if dist < min_dist:
            min_dist = dist
            best_idx = idx

    return best_idx


# -------------------------
# Main Caption Extraction
# -------------------------


@dataclass
class CaptionExtractionResult:
    """Result of caption extraction for a visual."""

    best_caption: Optional[CaptionCandidate]
    all_candidates: List[CaptionCandidate] = field(default_factory=list)
    extraction_attempted: bool = True


def extract_caption_from_pdf_text(
    doc: fitz.Document,
    page_num: int,
    visual_bbox_pts: Tuple[float, float, float, float],
    zones: CaptionSearchZones = CaptionSearchZones(),
    column_layout: Optional[ColumnLayout] = None,
) -> List[CaptionCandidate]:
    """
    Extract caption candidates from PDF text blocks.

    This is the preferred method for born-digital documents.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        visual_bbox_pts: Visual bounding box in PDF points
        zones: Search zones in PDF points
        column_layout: Pre-computed column layout (optional)

    Returns:
        List of caption candidates from PDF text
    """
    page = doc[page_num - 1]
    text_dict = page.get_text("dict")

    if column_layout is None:
        column_layout = infer_column_layout(doc, page_num)

    candidates: List[CaptionCandidate] = []

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # Only text blocks
            continue

        block_bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
        block_text = extract_block_text(block)

        if not block_text:
            continue

        # Check if in search zone relative to visual
        position, distance = get_relative_position(visual_bbox_pts, block_bbox, zones)

        if position is None:
            continue

        # Check for caption pattern
        pattern_result = detect_caption_pattern(block_text)

        if pattern_result:
            caption_type, match = pattern_result

            # Parse reference
            parsed_ref = parse_reference_from_match(
                match,
                caption_type.capitalize(),
                ReferenceSource.CAPTION,
            )

            # Higher confidence for pattern match
            confidence = 0.95

            candidates.append(
                CaptionCandidate(
                    text=block_text,
                    bbox_pts=block_bbox,
                    provenance=CaptionProvenance.PDF_TEXT,
                    position=position,
                    distance_pts=distance,
                    confidence=confidence,
                    parsed_reference=parsed_ref,
                )
            )
        else:
            # Check if text looks like it could be a caption continuation
            # (starts with uppercase, has reasonable length)
            if len(block_text) > 10 and block_text[0].isupper():
                # Lower confidence for non-pattern match
                candidates.append(
                    CaptionCandidate(
                        text=block_text,
                        bbox_pts=block_bbox,
                        provenance=CaptionProvenance.PDF_TEXT,
                        position=position,
                        distance_pts=distance,
                        confidence=0.3,  # Low confidence without pattern
                        parsed_reference=None,
                    )
                )

    return candidates


def extract_caption_via_ocr(
    doc: fitz.Document,
    page_num: int,
    visual_bbox_pts: Tuple[float, float, float, float],
    zones: CaptionSearchZones,
    dpi: int = 150,
) -> List[CaptionCandidate]:
    """
    Extract caption via OCR of the caption region.

    This is the fallback for scanned documents where PDF text is unreliable.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        visual_bbox_pts: Visual bounding box in PDF points
        zones: Search zones in PDF points
        dpi: DPI for OCR rendering

    Returns:
        List of caption candidates from OCR
    """
    # Note: This is a placeholder. Full OCR implementation would use
    # Tesseract, EasyOCR, or similar. For now, we return empty list
    # as the PDF text extraction should cover most clinical documents.

    # In a full implementation:
    # 1. Render caption zone at specified DPI
    # 2. Run OCR on the rendered region
    # 3. Parse results and create CaptionCandidate objects

    return []


def select_best_caption(
    candidates: List[CaptionCandidate],
    visual_type_hint: Optional[str] = None,
) -> Optional[CaptionCandidate]:
    """
    Select the best caption from candidates.

    Priority:
    1. Has parsed reference matching visual type
    2. Higher provenance priority (PDF_TEXT > OCR)
    3. Closer distance
    4. Position preference (below > above for figures, above > below for tables)

    Args:
        candidates: List of caption candidates
        visual_type_hint: Expected visual type ("table" or "figure")

    Returns:
        Best caption candidate or None
    """
    if not candidates:
        return None

    def score_candidate(c: CaptionCandidate) -> Tuple[int, int, float, int]:
        # Score components (higher is better)

        # 1. Has matching reference type
        ref_score = 0
        if c.parsed_reference:
            ref_score = 2
            if visual_type_hint:
                ref_type = c.parsed_reference.type_label.lower()
                if ref_type == visual_type_hint:
                    ref_score = 3

        # 2. Provenance priority
        prov_score = {
            CaptionProvenance.PDF_TEXT: 2,
            CaptionProvenance.OCR: 1,
            CaptionProvenance.VLM: 0,
        }.get(c.provenance, 0)

        # 3. Distance (inverted - smaller is better)
        dist_score = -c.distance_pts

        # 4. Position preference
        pos_score = 0
        if visual_type_hint == "table" and c.position == "above":
            pos_score = 1
        elif visual_type_hint == "figure" and c.position == "below":
            pos_score = 1

        return (ref_score, prov_score, dist_score, pos_score)

    # Sort by score (descending)
    sorted_candidates = sorted(candidates, key=score_candidate, reverse=True)
    return sorted_candidates[0]


def extract_caption_multisource(
    doc: fitz.Document,
    page_num: int,
    visual_bbox_pts: Tuple[float, float, float, float],
    zones: CaptionSearchZones = CaptionSearchZones(),
    column_layout: Optional[ColumnLayout] = None,
    visual_type_hint: Optional[str] = None,
    use_ocr_fallback: bool = True,
) -> CaptionExtractionResult:
    """
    Extract caption using multiple sources in priority order.

    Priority:
    1. PDF text blocks (born-digital, most reliable)
    2. OCR of expanded region (scanned docs)
    3. VLM extraction (handled downstream)

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        visual_bbox_pts: Visual bounding box in PDF points
        zones: Search zones in PDF points
        column_layout: Pre-computed column layout (optional)
        visual_type_hint: Expected visual type for scoring
        use_ocr_fallback: Whether to try OCR if PDF text fails

    Returns:
        CaptionExtractionResult with best caption and all candidates
    """
    all_candidates: List[CaptionCandidate] = []

    # Source 1: PDF text blocks
    pdf_candidates = extract_caption_from_pdf_text(
        doc, page_num, visual_bbox_pts, zones, column_layout
    )
    all_candidates.extend(pdf_candidates)

    # Check if we have good candidates from PDF text
    high_confidence_pdf = [c for c in pdf_candidates if c.confidence >= 0.8]

    # Source 2: OCR fallback (only if no high-confidence PDF candidates)
    if use_ocr_fallback and not high_confidence_pdf:
        ocr_candidates = extract_caption_via_ocr(
            doc, page_num, visual_bbox_pts, zones
        )
        all_candidates.extend(ocr_candidates)

    # Select best caption
    best_caption = select_best_caption(all_candidates, visual_type_hint)

    return CaptionExtractionResult(
        best_caption=best_caption,
        all_candidates=all_candidates,
        extraction_attempted=True,
    )


# -------------------------
# Batch Operations
# -------------------------


def extract_all_captions_on_page(
    doc: fitz.Document,
    page_num: int,
    zones: CaptionSearchZones = CaptionSearchZones(),
) -> List[CaptionCandidate]:
    """
    Extract all caption-like text blocks on a page.

    This is useful for pre-scanning a document to find all captions
    before linking them to visuals.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        zones: Search zones (not used for position filtering here)

    Returns:
        List of all caption candidates on the page
    """
    page = doc[page_num - 1]
    text_dict = page.get_text("dict")

    candidates: List[CaptionCandidate] = []

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        block_bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
        block_text = extract_block_text(block)

        if not block_text:
            continue

        # Check for caption pattern
        pattern_result = detect_caption_pattern(block_text)

        if pattern_result:
            caption_type, match = pattern_result

            parsed_ref = parse_reference_from_match(
                match,
                caption_type.capitalize(),
                ReferenceSource.CAPTION,
            )

            candidates.append(
                CaptionCandidate(
                    text=block_text,
                    bbox_pts=block_bbox,
                    provenance=CaptionProvenance.PDF_TEXT,
                    position="below",  # Default; will be updated when linked
                    distance_pts=0.0,
                    confidence=0.95,
                    parsed_reference=parsed_ref,
                )
            )

    return candidates


def extract_all_captions_in_document(
    doc: fitz.Document,
) -> Dict[int, List[CaptionCandidate]]:
    """
    Extract all captions across all pages.

    Args:
        doc: Open PyMuPDF document

    Returns:
        Dict mapping page_num -> list of captions on that page
    """
    captions_by_page: Dict[int, List[CaptionCandidate]] = {}

    for page_idx in range(doc.page_count):
        page_num = page_idx + 1
        captions = extract_all_captions_on_page(doc, page_num)
        if captions:
            captions_by_page[page_num] = captions

    return captions_by_page


# -------------------------
# Caption Linking
# -------------------------


def link_caption_to_visual(
    caption: CaptionCandidate,
    visual_bbox_pts: Tuple[float, float, float, float],
    zones: CaptionSearchZones = CaptionSearchZones(),
) -> Optional[CaptionCandidate]:
    """
    Check if a caption can be linked to a visual and update its position.

    Args:
        caption: Caption candidate
        visual_bbox_pts: Visual bounding box in PDF points
        zones: Search zones

    Returns:
        Updated caption with correct position/distance, or None if not linkable
    """
    position, distance = get_relative_position(
        visual_bbox_pts, caption.bbox_pts, zones
    )

    if position is None:
        return None

    # Create updated caption with correct position
    return CaptionCandidate(
        text=caption.text,
        bbox_pts=caption.bbox_pts,
        provenance=caption.provenance,
        position=position,
        distance_pts=distance,
        confidence=caption.confidence,
        parsed_reference=caption.parsed_reference,
    )


def find_caption_for_visual(
    visual_bbox_pts: Tuple[float, float, float, float],
    page_num: int,
    all_captions: Dict[int, List[CaptionCandidate]],
    zones: CaptionSearchZones = CaptionSearchZones(),
    visual_type_hint: Optional[str] = None,
) -> Optional[CaptionCandidate]:
    """
    Find the best caption for a visual from pre-extracted captions.

    Args:
        visual_bbox_pts: Visual bounding box in PDF points
        page_num: Page number where visual is located
        all_captions: Dict of page_num -> captions from extract_all_captions_in_document
        zones: Search zones
        visual_type_hint: Expected visual type

    Returns:
        Best matching caption or None
    """
    page_captions = all_captions.get(page_num, [])

    if not page_captions:
        return None

    # Link each caption to the visual
    linked_candidates: List[CaptionCandidate] = []

    for caption in page_captions:
        linked = link_caption_to_visual(caption, visual_bbox_pts, zones)
        if linked:
            linked_candidates.append(linked)

    return select_best_caption(linked_candidates, visual_type_hint)


__all__ = [
    # Types
    "CaptionCandidate",
    "CaptionExtractionResult",
    "CaptionSearchZones",
    "ColumnLayout",
    # Main extraction
    "extract_caption_multisource",
    "extract_caption_from_pdf_text",
    "extract_caption_via_ocr",
    "select_best_caption",
    # Batch operations
    "extract_all_captions_on_page",
    "extract_all_captions_in_document",
    # Caption linking
    "link_caption_to_visual",
    "find_caption_for_visual",
    # Column layout
    "infer_column_layout",
    "get_column_for_x",
    # Helpers
    "detect_caption_pattern",
    "parse_reference_from_match",
    "is_continuation_caption",
    "get_relative_position",
    "has_horizontal_overlap",
    "has_vertical_overlap",
]
