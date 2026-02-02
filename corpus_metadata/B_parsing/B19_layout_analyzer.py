# corpus_metadata/B_parsing/B19_layout_analyzer.py
"""
VLM Layout Analyzer.

Analyzes PDF pages using Claude Vision to detect:
1. Page layout pattern (full, 2col, hybrid)
2. Visual zones (rough location, not precise bbox)

The VLM returns zones, not coordinates. Precise extraction
is handled by whitespace expansion in B20_zone_expander.py.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from typing import Optional, Tuple

import anthropic
import fitz  # PyMuPDF

from B_parsing.B18_layout_models import (
    LayoutPattern,
    PageLayout,
    VisualPosition,
    VisualZone,
)

logger = logging.getLogger(__name__)


VLM_LAYOUT_PROMPT = """Analyze this PDF page layout and identify all visual elements.

## STEP 1: Identify the page layout pattern

- "full": Single column layout, content spans full page width
- "2col": Two-column layout throughout the page
- "2col-fullbot": Two columns at top, full-width section at bottom
- "fulltop-2col": Full-width section at top, two columns below

If 2-column, estimate where the column boundary is (0.0 to 1.0, where 0.5 = middle).

## STEP 2: Identify all visual elements (tables and figures)

For EACH visual, report:
- type: "table" or "figure"
- label: The label if visible (e.g., "Figure 1", "Table 2")
- position: "left" (left column), "right" (right column), or "full" (spans full width)
- vertical_zone: Where vertically on the page - "top", "middle", "bottom", or a range like "0.2-0.5"
- confidence: 0.0 to 1.0

## IMPORTANT RULES

1. Do NOT provide precise bounding boxes - just identify the zone
2. For multi-panel figures (A, B, C), report as ONE visual
3. Include the caption as part of the visual's zone
4. If a visual spans both columns, position = "full"
5. Multiple visuals can be in the same column (stacked vertically)

## Response format (JSON only)

{
    "layout": "2col",
    "column_boundary": 0.48,
    "visuals": [
        {
            "type": "figure",
            "label": "Figure 1",
            "position": "left",
            "vertical_zone": "top",
            "confidence": 0.95,
            "caption_start": "Figure 1. Study design..."
        }
    ],
    "notes": "optional observations"
}

If no visuals found: {"layout": "...", "visuals": [], "notes": "..."}"""


def render_page_for_analysis(
    doc: fitz.Document,
    page_num: int,
    max_dimension: int = 1568,
    dpi: int = 150,
) -> Tuple[str, float, float]:
    """
    Render a page as base64 PNG for VLM analysis.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        max_dimension: Maximum width or height in pixels
        dpi: Rendering DPI

    Returns:
        Tuple of (base64_image, page_width_pts, page_height_pts)
    """
    page = doc[page_num - 1]
    page_width = page.rect.width
    page_height = page.rect.height

    # Calculate zoom to fit within max_dimension
    zoom = dpi / 72.0
    rendered_width = page_width * zoom
    rendered_height = page_height * zoom

    if max(rendered_width, rendered_height) > max_dimension:
        scale = max_dimension / max(rendered_width, rendered_height)
        zoom *= scale

    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    img_bytes = pix.tobytes("png")
    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    return base64_image, page_width, page_height


def analyze_page_layout(
    doc: fitz.Document,
    page_num: int,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-20250514",
) -> PageLayout:
    """
    Analyze a page's layout and identify visual zones using VLM.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        client: Anthropic client
        model: Model to use

    Returns:
        PageLayout with pattern and visual zones
    """
    # Render page
    base64_image, page_width, page_height = render_page_for_analysis(doc, page_num)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": VLM_LAYOUT_PROMPT,
                        },
                    ],
                }
            ],
        )

        content_block = response.content[0]
        raw_text = content_block.text if hasattr(content_block, "text") else str(content_block)
        return parse_layout_response(raw_text, page_num)

    except Exception as e:
        logger.error(f"Layout analysis failed for page {page_num}: {e}")
        # Return default full-page layout with no visuals
        return PageLayout(
            page_num=page_num,
            pattern=LayoutPattern.FULL,
            visuals=[],
            raw_vlm_response=str(e),
        )


def parse_layout_response(raw_text: str, page_num: int) -> PageLayout:
    """
    Parse VLM JSON response into PageLayout.

    Args:
        raw_text: Raw text response from VLM
        page_num: Page number

    Returns:
        PageLayout with parsed data
    """
    try:
        # Extract JSON from response (may have markdown code blocks)
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if not json_match:
            logger.warning(f"No JSON found in layout response for page {page_num}")
            return PageLayout(page_num=page_num, pattern=LayoutPattern.FULL)

        data = json.loads(json_match.group())

        # Parse layout pattern
        layout_code = data.get("layout", "full").lower()
        pattern = _parse_layout_pattern(layout_code)

        # Parse column boundary
        column_boundary = data.get("column_boundary")
        if column_boundary is not None:
            column_boundary = float(column_boundary)

        # Parse visuals
        visuals = []
        for item in data.get("visuals", []):
            zone = _parse_visual_zone(item)
            if zone:
                visuals.append(zone)

        return PageLayout(
            page_num=page_num,
            pattern=pattern,
            column_boundary=column_boundary,
            visuals=visuals,
            raw_vlm_response=raw_text,
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse layout JSON for page {page_num}: {e}")
        return PageLayout(page_num=page_num, pattern=LayoutPattern.FULL, raw_vlm_response=raw_text)
    except Exception as e:
        logger.warning(f"Error parsing layout response for page {page_num}: {e}")
        return PageLayout(page_num=page_num, pattern=LayoutPattern.FULL, raw_vlm_response=raw_text)


def _parse_layout_pattern(code: str) -> LayoutPattern:
    """Parse layout code string to LayoutPattern enum."""
    mapping = {
        "full": LayoutPattern.FULL,
        "2col": LayoutPattern.TWO_COL,
        "2col-fullbot": LayoutPattern.TWO_COL_FULLBOT,
        "fulltop-2col": LayoutPattern.FULLTOP_TWO_COL,
    }
    return mapping.get(code, LayoutPattern.FULL)


def _parse_visual_zone(item: dict) -> Optional[VisualZone]:
    """Parse a visual item dict to VisualZone."""
    try:
        # Parse position
        pos_str = item.get("position", "full").lower()
        if pos_str == "left":
            position = VisualPosition.LEFT
        elif pos_str == "right":
            position = VisualPosition.RIGHT
        else:
            position = VisualPosition.FULL

        return VisualZone(
            visual_type=item.get("type", "figure"),
            label=item.get("label"),
            position=position,
            vertical_zone=item.get("vertical_zone", "middle"),
            confidence=float(item.get("confidence", 0.8)),
            caption_snippet=item.get("caption_start"),
            is_continuation=bool(item.get("is_continuation", False)),
            continues_next=bool(item.get("continues_next", False)),
        )
    except Exception as e:
        logger.warning(f"Failed to parse visual zone: {e}")
        return None


def analyze_document_layouts(
    pdf_path: str,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-20250514",
    pages: Optional[list[int]] = None,
) -> list[PageLayout]:
    """
    Analyze all pages in a document for layout and visuals.

    Args:
        pdf_path: Path to PDF file
        client: Anthropic client
        model: Model to use
        pages: Optional list of specific pages to analyze (1-indexed)

    Returns:
        List of PageLayout objects, one per page
    """
    doc = fitz.open(pdf_path)
    layouts = []

    try:
        page_nums = pages if pages else range(1, len(doc) + 1)

        for page_num in page_nums:
            if page_num < 1 or page_num > len(doc):
                logger.warning(f"Skipping invalid page number: {page_num}")
                continue

            logger.info(f"Analyzing layout for page {page_num}/{len(doc)}")
            layout = analyze_page_layout(doc, page_num, client, model)
            layouts.append(layout)

            logger.debug(
                f"Page {page_num}: {layout.pattern.value} layout, "
                f"{len(layout.visuals)} visuals"
            )

    finally:
        doc.close()

    return layouts
