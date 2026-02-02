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
import io
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import anthropic
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

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


# -------------------------
# Debug Mode: Visual Detection with Bounding Boxes
# -------------------------

VLM_BBOX_DEBUG_PROMPT = """Identify ALL tables and figures on this PDF page and provide their EXACT bounding boxes.

For EACH visual element (table or figure), provide:
- type: "table" or "figure"
- label: The label if visible (e.g., "Figure 1", "Table 2")
- bbox: [x0, y0, x1, y1] as normalized coordinates (0.0 to 1.0)
  - x0, y0 = top-left corner
  - x1, y1 = bottom-right corner
  - Include the caption in the bounding box
- confidence: 0.0 to 1.0

IMPORTANT:
1. The bounding box MUST include the caption
2. Use normalized coordinates (0.0 = left/top edge, 1.0 = right/bottom edge)
3. Be PRECISE - the box should tightly wrap the visual and its caption
4. For multi-panel figures (A, B, C), return ONE bounding box for the entire figure

Response format (JSON only):
{
    "visuals": [
        {
            "type": "figure",
            "label": "Figure 1",
            "bbox": [0.05, 0.15, 0.48, 0.55],
            "confidence": 0.95
        },
        {
            "type": "table",
            "label": "Table 1",
            "bbox": [0.52, 0.20, 0.95, 0.70],
            "confidence": 0.90
        }
    ]
}

If no visuals: {"visuals": []}"""


def analyze_page_with_bbox(
    doc: fitz.Document,
    page_num: int,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-20250514",
) -> List[dict]:
    """
    Analyze a page and get precise bounding boxes for visuals.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        client: Anthropic client
        model: Model to use

    Returns:
        List of dicts with type, label, bbox (normalized), confidence
    """
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
                            "text": VLM_BBOX_DEBUG_PROMPT,
                        },
                    ],
                }
            ],
        )

        content_block = response.content[0]
        raw_text = content_block.text if hasattr(content_block, "text") else str(content_block)

        # Parse JSON response
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if not json_match:
            logger.warning(f"No JSON found in bbox response for page {page_num}")
            return []

        data = json.loads(json_match.group())
        return data.get("visuals", [])

    except Exception as e:
        logger.error(f"Bbox detection failed for page {page_num}: {e}")
        return []


VLM_REFINE_PROMPT = """Look at the colored rectangles I drew:
- RED rectangles = figures
- BLUE rectangles = tables

CHECK EACH RECTANGLE CAREFULLY:

1. Does the rectangle include the ENTIRE visual?
   - Is the TOP of the figure/table inside the box? (y0 should be ABOVE the visual)
   - Is the BOTTOM fully included?
   - Are LEFT and RIGHT edges correct?

2. Does the rectangle include the CAPTION?
   - Figure captions are usually BELOW the figure
   - Table captions are usually ABOVE the table
   - The caption text MUST be inside the rectangle

3. Common problems to look for:
   - Rectangle starts too LOW (missing top of visual)
   - Rectangle ends too HIGH (missing caption below)
   - Rectangle on wrong column

COORDINATE SYSTEM:
- x0=0.0 is LEFT edge, x1=1.0 is RIGHT edge
- y0=0.0 is TOP of page, y1=1.0 is BOTTOM of page
- To include MORE at the top: DECREASE y0
- To include MORE at the bottom: INCREASE y1

For EACH visual, provide corrected coordinates:

{
    "corrections": [
        {
            "label": "Figure 1",
            "is_correct": false,
            "issue": "Missing top of figure and caption below",
            "corrected_bbox": [x0, y0, x1, y1]
        }
    ]
}

If a rectangle is correct: {"label": "...", "is_correct": true}"""


def refine_bboxes(
    annotated_image: Image.Image,
    visuals: List[dict],
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-20250514",
) -> List[dict]:
    """
    Ask VLM to refine bounding boxes by showing it the annotated image.

    Args:
        annotated_image: PIL Image with rectangles drawn
        visuals: Original visual detections
        client: Anthropic client
        model: Model to use

    Returns:
        List of refined visual dicts with corrected bboxes
    """
    # Convert image to base64
    buffer = io.BytesIO()
    annotated_image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

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
                            "text": VLM_REFINE_PROMPT,
                        },
                    ],
                }
            ],
        )

        content_block = response.content[0]
        raw_text = content_block.text if hasattr(content_block, "text") else str(content_block)

        # Parse corrections
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if not json_match:
            return visuals

        data = json.loads(json_match.group())
        corrections = data.get("corrections", [])

        if not corrections or data.get("all_correct"):
            return visuals

        # Apply corrections
        refined = []
        for visual in visuals:
            label = visual.get("label", "")
            # Find matching correction
            correction = next(
                (c for c in corrections if c.get("label") == label),
                None
            )
            if correction and not correction.get("is_correct", True):
                corrected_bbox = correction.get("corrected_bbox")
                if corrected_bbox:
                    refined.append({
                        **visual,
                        "bbox": corrected_bbox,
                        "refined": True,
                        "original_bbox": visual.get("bbox"),
                    })
                else:
                    refined.append(visual)
            else:
                refined.append(visual)

        return refined

    except Exception as e:
        logger.error(f"Bbox refinement failed: {e}")
        return visuals


def analyze_page_with_refinement(
    doc: fitz.Document,
    page_num: int,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-20250514",
    max_iterations: int = 2,
    output_dir: Optional[str] = None,
) -> Tuple[List[dict], Image.Image]:
    """
    Analyze a page with iterative bbox refinement.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        client: Anthropic client
        model: Model to use
        max_iterations: Maximum refinement iterations
        output_dir: Optional directory to save intermediate images

    Returns:
        Tuple of (refined visuals, final annotated image)
    """
    doc_name = Path(doc.name).stem if doc.name else "doc"

    # Initial detection
    visuals = analyze_page_with_bbox(doc, page_num, client, model)

    if not visuals:
        # No visuals found, return empty
        page = doc[page_num - 1]
        zoom = 150 / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return [], img

    # Iterative refinement
    for iteration in range(max_iterations):
        logger.info(f"Page {page_num} - Refinement iteration {iteration + 1}")

        # Draw current boxes
        img = draw_debug_rectangles(doc, page_num, visuals)

        # Save intermediate if requested
        if output_dir:
            out_path = Path(output_dir) / f"{doc_name}_p{page_num}_iter{iteration}.png"
            img.save(str(out_path))

        # Ask VLM to refine
        refined = refine_bboxes(img, visuals, client, model)

        # Check if any changes
        changes = sum(1 for v in refined if v.get("refined"))
        if changes == 0:
            logger.info(f"Page {page_num} - No changes in iteration {iteration + 1}, stopping")
            break

        logger.info(f"Page {page_num} - {changes} boxes refined in iteration {iteration + 1}")
        visuals = refined

    # Final image
    final_img = draw_debug_rectangles(doc, page_num, visuals)

    return visuals, final_img


# -------------------------
# Grid-Based Detection
# -------------------------

def draw_grid_overlay(
    doc: fitz.Document,
    page_num: int,
    rows: int = 10,
    cols: int = 10,
    dpi: int = 150,
    style: str = "margin",  # "margin", "overlay_dashed", "intersections"
) -> Tuple[Image.Image, dict]:
    """
    Draw a labeled grid on a page using various styles to avoid confusion with content.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        rows: Number of grid rows
        cols: Number of grid columns
        dpi: Rendering DPI
        style: Grid style - "margin" (rulers outside), "overlay_dashed", "intersections"

    Returns:
        Tuple of (PIL Image with grid, grid_info dict)
    """
    # Render page
    page = doc[page_num - 1]
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    original_img = Image.open(io.BytesIO(pix.tobytes("png")))

    orig_width, orig_height = original_img.size

    row_labels = [chr(ord('A') + i) for i in range(rows)]
    col_labels = [str(i + 1) for i in range(cols)]

    if style == "margin":
        # Add margins around the page for rulers - NO overlay on content
        margin = 50  # pixels for ruler margin
        new_width = orig_width + margin
        new_height = orig_height + margin

        # Create new image with margins
        img = Image.new("RGB", (new_width, new_height), "white")
        img.paste(original_img, (margin, margin))

        draw = ImageDraw.Draw(img)

        cell_width = orig_width / cols
        cell_height = orig_height / rows

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except Exception:
            font = ImageFont.load_default()

        # Draw column labels and tick marks at TOP margin
        for c in range(cols):
            cx = margin + int(c * cell_width + cell_width / 2)
            # Tick mark
            draw.line([(cx, margin - 10), (cx, margin)], fill="blue", width=2)
            # Label
            draw.text((cx, 20), col_labels[c], fill="blue", font=font, anchor="mm")

        # Draw column boundary lines (light cyan, dashed effect)
        for c in range(cols + 1):
            x = margin + int(c * cell_width)
            # Draw dashed line
            for y in range(margin, new_height, 20):
                draw.line([(x, y), (x, min(y + 10, new_height))], fill="cyan", width=1)

        # Draw row labels and tick marks on LEFT margin
        for r in range(rows):
            cy = margin + int(r * cell_height + cell_height / 2)
            # Tick mark
            draw.line([(margin - 10, cy), (margin, cy)], fill="blue", width=2)
            # Label
            draw.text((25, cy), row_labels[r], fill="blue", font=font, anchor="mm")

        # Draw row boundary lines (light cyan, dashed effect)
        for r in range(rows + 1):
            y = margin + int(r * cell_height)
            # Draw dashed line
            for x in range(margin, new_width, 20):
                draw.line([(x, y), (min(x + 10, new_width), y)], fill="cyan", width=1)

        # Draw border around original page area
        draw.rectangle([margin, margin, margin + orig_width, margin + orig_height],
                       outline="blue", width=3)

        grid_info = {
            "rows": rows,
            "cols": cols,
            "row_labels": row_labels,
            "col_labels": col_labels,
            "cell_width": cell_width,
            "cell_height": cell_height,
            "img_width": orig_width,
            "img_height": orig_height,
            "margin": margin,
        }

    elif style == "intersections":
        # Only draw small crosses at grid intersections
        img = original_img.copy()
        draw = ImageDraw.Draw(img)

        cell_width = orig_width / cols
        cell_height = orig_height / rows

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except Exception:
            font = ImageFont.load_default()

        cross_size = 8

        # Draw crosses at intersections with labels
        for r in range(rows + 1):
            for c in range(cols + 1):
                x = int(c * cell_width)
                y = int(r * cell_height)

                # Draw magenta cross
                draw.line([(x - cross_size, y), (x + cross_size, y)], fill="magenta", width=2)
                draw.line([(x, y - cross_size), (x, y + cross_size)], fill="magenta", width=2)

        # Labels on edges only
        for c in range(cols):
            cx = int(c * cell_width + cell_width / 2)
            draw.text((cx, 5), col_labels[c], fill="magenta", font=font, anchor="mt")

        for r in range(rows):
            cy = int(r * cell_height + cell_height / 2)
            draw.text((5, cy), row_labels[r], fill="magenta", font=font, anchor="lm")

        grid_info = {
            "rows": rows,
            "cols": cols,
            "row_labels": row_labels,
            "col_labels": col_labels,
            "cell_width": cell_width,
            "cell_height": cell_height,
            "img_width": orig_width,
            "img_height": orig_height,
            "margin": 0,
        }

    else:  # overlay_dashed - cyan dashed lines
        img = original_img.copy()
        draw = ImageDraw.Draw(img)

        cell_width = orig_width / cols
        cell_height = orig_height / rows

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except Exception:
            font = ImageFont.load_default()

        # Draw dashed cyan grid lines
        dash_length = 15
        gap_length = 10

        for c in range(cols + 1):
            x = int(c * cell_width)
            y = 0
            while y < orig_height:
                draw.line([(x, y), (x, min(y + dash_length, orig_height))], fill="cyan", width=2)
                y += dash_length + gap_length

        for r in range(rows + 1):
            y = int(r * cell_height)
            x = 0
            while x < orig_width:
                draw.line([(x, y), (min(x + dash_length, orig_width), y)], fill="cyan", width=2)
                x += dash_length + gap_length

        # Labels with background
        for c in range(cols):
            cx = int(c * cell_width + cell_width / 2)
            text_bbox = draw.textbbox((cx, 15), col_labels[c], font=font, anchor="mm")
            draw.rectangle([text_bbox[0]-3, text_bbox[1]-2, text_bbox[2]+3, text_bbox[3]+2],
                           fill="white", outline="cyan", width=2)
            draw.text((cx, 15), col_labels[c], fill="blue", font=font, anchor="mm")

        for r in range(rows):
            cy = int(r * cell_height + cell_height / 2)
            text_bbox = draw.textbbox((15, cy), row_labels[r], font=font, anchor="mm")
            draw.rectangle([text_bbox[0]-3, text_bbox[1]-2, text_bbox[2]+3, text_bbox[3]+2],
                           fill="white", outline="cyan", width=2)
            draw.text((15, cy), row_labels[r], fill="blue", font=font, anchor="mm")

        grid_info = {
            "rows": rows,
            "cols": cols,
            "row_labels": row_labels,
            "col_labels": col_labels,
            "cell_width": cell_width,
            "cell_height": cell_height,
            "img_width": orig_width,
            "img_height": orig_height,
            "margin": 0,
        }

    return img, grid_info


def parse_cell_reference(cell: str, grid_info: dict) -> Tuple[int, int]:
    """
    Parse a cell reference like 'C5' into (row_idx, col_idx).

    Args:
        cell: Cell reference string
        grid_info: Grid info dict

    Returns:
        Tuple of (row_index, col_index)
    """
    cell = cell.strip().upper()
    row_labels = grid_info["row_labels"]
    col_labels = grid_info["col_labels"]

    row_char = cell[0]
    col_str = cell[1:]

    row_idx = row_labels.index(row_char) if row_char in row_labels else 0
    col_idx = col_labels.index(col_str) if col_str in col_labels else 0

    return row_idx, col_idx


def grid_range_to_bbox(top_left: str, bottom_right: str, grid_info: dict) -> List[float]:
    """
    Convert grid cell range to normalized bbox.

    Args:
        top_left: Top-left cell reference like "C2"
        bottom_right: Bottom-right cell reference like "F8"
        grid_info: Grid info dict from draw_grid_overlay

    Returns:
        Normalized bbox [x0, y0, x1, y1]
    """
    rows = grid_info["rows"]
    cols = grid_info["cols"]

    top_row, left_col = parse_cell_reference(top_left, grid_info)
    bottom_row, right_col = parse_cell_reference(bottom_right, grid_info)

    # Convert to normalized coordinates
    x0 = left_col / cols
    y0 = top_row / rows
    x1 = (right_col + 1) / cols
    y1 = (bottom_row + 1) / rows

    return [x0, y0, x1, y1]


def grid_cells_to_bbox(cells: List[str], grid_info: dict) -> List[float]:
    """
    Convert grid cell references to normalized bbox.

    Args:
        cells: List of cell references like ["B2", "B3", "C2", "C3"]
        grid_info: Grid info dict from draw_grid_overlay

    Returns:
        Normalized bbox [x0, y0, x1, y1]
    """
    if not cells:
        return [0, 0, 1, 1]

    rows = grid_info["rows"]
    cols = grid_info["cols"]
    row_labels = grid_info["row_labels"]
    col_labels = grid_info["col_labels"]

    min_row, max_row = rows, 0
    min_col, max_col = cols, 0

    for cell in cells:
        cell = cell.strip().upper()
        if len(cell) < 2:
            continue

        row_char = cell[0]
        col_str = cell[1:]

        if row_char in row_labels:
            row_idx = row_labels.index(row_char)
            min_row = min(min_row, row_idx)
            max_row = max(max_row, row_idx)

        if col_str in col_labels:
            col_idx = col_labels.index(col_str)
            min_col = min(min_col, col_idx)
            max_col = max(max_col, col_idx)

    # Convert to normalized coordinates
    x0 = min_col / cols
    y0 = min_row / rows
    x1 = (max_col + 1) / cols
    y1 = (max_row + 1) / rows

    return [x0, y0, x1, y1]


def get_grid_prompt(rows: int, cols: int) -> str:
    """Generate grid prompt with correct row/col labels."""
    row_labels = [chr(ord('A') + i) for i in range(rows)]
    first_row = row_labels[0]
    last_row = row_labels[-1]

    return f"""I added a RED GRID OVERLAY on top of this PDF page to help you locate visuals.
The grid is NOT part of the original document - ignore the grid lines when identifying content.

GRID REFERENCE:
- ROWS: {first_row} to {last_row} (top to bottom) - labels on LEFT edge in yellow
- COLUMNS: 1 to {cols} (left to right) - labels on TOP edge in yellow

Look THROUGH the grid at the actual document content underneath.
Find any FIGURES (images, charts, graphs) or TABLES.

For each visual, identify which grid cells it occupies:

1. TOP EDGE: Which ROW letter does the TOP of the visual touch?
   (Look at where the figure/table STARTS - include title/header if above)

2. BOTTOM EDGE: Which ROW letter does the BOTTOM touch?
   (Include the caption at the bottom)

3. LEFT EDGE: Which COLUMN number does the LEFT side touch?

4. RIGHT EDGE: Which COLUMN number does the RIGHT side touch?

Response format (JSON only):
{{
    "visuals": [
        {{
            "type": "figure",
            "label": "Figure 1",
            "top_row": "C",
            "bottom_row": "G",
            "left_col": "3",
            "right_col": "7"
        }}
    ]
}}

IMPORTANT: Include the figure caption in your boundaries!
If no visuals found: {{"visuals": []}}"""


def get_grid_prompt_for_style(rows: int, cols: int, style: str) -> str:
    """Generate appropriate prompt based on grid style."""
    row_labels = [chr(ord('A') + i) for i in range(rows)]
    first_row = row_labels[0]
    last_row = row_labels[-1]

    if style == "margin":
        return f"""This image shows a PDF page with a COORDINATE SYSTEM added around it:
- The page content is INSIDE the blue border
- ROW labels ({first_row}-{last_row}) are on the LEFT margin (outside the page)
- COLUMN labels (1-{cols}) are on the TOP margin (outside the page)
- Light CYAN dashed lines show the grid divisions on the page

The grid lines are just for reference - they are NOT part of the actual document.
Look at the ACTUAL CONTENT inside the blue border.

Find all FIGURES and TABLES in the document content.

For each visual, tell me which grid cells it occupies:
- TOP ROW: Which row letter ({first_row}-{last_row}) does the TOP of the visual touch?
- BOTTOM ROW: Which row does the BOTTOM touch? (include caption)
- LEFT COLUMN: Which column number (1-{cols}) does the LEFT edge touch?
- RIGHT COLUMN: Which column does the RIGHT edge touch?

Response (JSON only):
{{
    "visuals": [
        {{
            "type": "figure",
            "label": "Figure 1",
            "top_row": "D",
            "bottom_row": "H",
            "left_col": "3",
            "right_col": "7"
        }}
    ]
}}

If no visuals: {{"visuals": []}}"""

    else:  # overlay styles
        return f"""This page has a grid overlay to help locate visuals:
- ROWS: {first_row} to {last_row} (top to bottom)
- COLUMNS: 1 to {cols} (left to right)

The grid is just for reference - ignore it when identifying actual content.
Find FIGURES and TABLES in the document.

For each visual, which grid cells does it occupy?

Response (JSON only):
{{
    "visuals": [
        {{
            "type": "figure",
            "label": "Figure 1",
            "top_row": "D",
            "bottom_row": "H",
            "left_col": "3",
            "right_col": "7"
        }}
    ]
}}

Include the caption in the bounding area!
If no visuals: {{"visuals": []}}"""


def analyze_page_with_grid(
    doc: fitz.Document,
    page_num: int,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-20250514",
    rows: int = 10,
    cols: int = 10,
    output_dir: Optional[str] = None,
    style: str = "margin",
) -> List[dict]:
    """
    Analyze a page using grid-based detection.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        client: Anthropic client
        model: Model to use
        rows: Grid rows
        cols: Grid columns
        output_dir: Optional directory to save debug images
        style: Grid style - "margin", "overlay_dashed", "intersections"

    Returns:
        List of visuals with bboxes derived from grid cells
    """
    # Create grid overlay
    grid_img, grid_info = draw_grid_overlay(doc, page_num, rows, cols, style=style)

    # Save grid image if requested
    if output_dir:
        doc_name = Path(doc.name).stem if doc.name else "doc"
        grid_path = Path(output_dir) / f"{doc_name}_p{page_num}_grid.png"
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        grid_img.save(str(grid_path))

    # Convert to base64
    buffer = io.BytesIO()
    grid_img.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Generate dynamic prompt based on grid size and style
    prompt = get_grid_prompt_for_style(rows, cols, style)

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
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        content_block = response.content[0]
        raw_text = content_block.text if hasattr(content_block, "text") else str(content_block)

        # Parse response
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if not json_match:
            logger.warning(f"No JSON found in grid response for page {page_num}")
            return []

        data = json.loads(json_match.group())
        raw_visuals = data.get("visuals", [])

        # Convert cells to bboxes
        result = []
        for v in raw_visuals:
            # Support multiple formats
            top_row = v.get("top_row")
            bottom_row = v.get("bottom_row")
            left_col = v.get("left_col")
            right_col = v.get("right_col")

            if top_row and bottom_row and left_col and right_col:
                # New explicit boundary format
                top_left = f"{top_row}{left_col}"
                bottom_right = f"{bottom_row}{right_col}"
                bbox = grid_range_to_bbox(top_left, bottom_right, grid_info)
            elif v.get("top_left") and v.get("bottom_right"):
                # Old top_left/bottom_right format
                bbox = grid_range_to_bbox(v.get("top_left"), v.get("bottom_right"), grid_info)
                top_left = v.get("top_left")
                bottom_right = v.get("bottom_right")
            else:
                # Fallback to cells list
                cells = v.get("cells", [])
                bbox = grid_cells_to_bbox(cells, grid_info)
                top_left = None
                bottom_right = None

            result.append({
                "type": v.get("type", "figure"),
                "label": v.get("label"),
                "bbox": bbox,
                "top_left": top_left if top_left else f"{top_row}{left_col}" if top_row else None,
                "bottom_right": bottom_right if bottom_right else f"{bottom_row}{right_col}" if bottom_row else None,
            })

        # Draw result boxes ON TOP of grid image so we can verify alignment
        if output_dir and result:
            # Draw thick blue rectangles on the grid image to show detected areas
            draw = ImageDraw.Draw(grid_img)
            margin = grid_info.get("margin", 0)
            for v in result:
                bbox = v.get("bbox", [])
                if len(bbox) == 4:
                    x0, y0, x1, y1 = bbox
                    # Convert to pixel coordinates, accounting for margin offset
                    px0 = margin + int(x0 * grid_info["img_width"])
                    py0 = margin + int(y0 * grid_info["img_height"])
                    px1 = margin + int(x1 * grid_info["img_width"])
                    py1 = margin + int(y1 * grid_info["img_height"])
                    # Draw thick blue rectangle
                    draw.rectangle([px0, py0, px1, py1], outline="blue", width=4)

            result_path = Path(output_dir) / f"{doc_name}_p{page_num}_result.png"
            grid_img.save(str(result_path))

        return result

    except Exception as e:
        logger.error(f"Grid detection failed for page {page_num}: {e}")
        return []


# -------------------------
# Two-Phase Detection (Column-First)
# -------------------------

PHASE1_COLUMN_PROMPT = """I drew a vertical RED LINE down the middle of this PDF page.
The page is divided into:
- LEFT side (left of the red line)
- RIGHT side (right of the red line)

Find any FIGURES or TABLES on this page.

For EACH visual, tell me:
1. type: "figure" or "table"
2. label: The label if visible (e.g., "Figure 1")
3. side: Is it on the "left" side, "right" side, or "spans_both"?
4. vertical_position: "top", "upper", "middle", "lower", or "bottom" of the page

Response JSON only:
{
    "visuals": [
        {
            "type": "figure",
            "label": "Figure 1",
            "side": "right",
            "vertical_position": "middle"
        }
    ]
}

If no visuals: {"visuals": []}"""


def get_column_grid_prompt(rows: int, cols: int, column_side: str) -> str:
    """Generate prompt for column-specific grid detection."""
    row_labels = [chr(ord('A') + i) for i in range(rows)]
    first_row = row_labels[0]
    last_row = row_labels[-1]

    return f"""I drew a GREEN GRID on the {column_side.upper()} COLUMN of this page.

The {column_side.upper()} COLUMN has a grid with:
- ROWS: {first_row} to {last_row} (top to bottom)
- COLUMNS: 1 to {cols} (left to right within the {column_side.upper()} column only)

The other column (marked in red) should be IGNORED.

Look at the {column_side.upper()} COLUMN ONLY.
Find the visual(s) in this column and tell me exactly which grid cells they occupy.

For each visual, provide:
- type: "figure" or "table"
- label: The label if visible
- top_row: Which row letter does the TOP of the visual touch? (include any title/header)
- bottom_row: Which row does the BOTTOM touch? (include the caption)
- left_col: Which column number (1-{cols}) does the LEFT edge touch?
- right_col: Which column number does the RIGHT edge touch?

Response JSON only:
{{
    "visuals": [
        {{
            "type": "figure",
            "label": "Figure 1",
            "top_row": "C",
            "bottom_row": "H",
            "left_col": "1",
            "right_col": "{cols}"
        }}
    ]
}}
"""


def analyze_page_two_phase(
    doc: fitz.Document,
    page_num: int,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-20250514",
    grid_rows: int = 10,
    grid_cols: int = 5,
    output_dir: Optional[str] = None,
    dpi: int = 150,
) -> List[dict]:
    """
    Analyze a page using two-phase column-aware detection.

    Phase 1: Identify which column (left/right/both) contains each visual
    Phase 2: Grid only the relevant column(s) to get precise coordinates

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        client: Anthropic client
        model: Model to use
        grid_rows: Number of grid rows for phase 2
        grid_cols: Number of grid columns for phase 2 (per column)
        output_dir: Optional directory to save debug images
        dpi: Rendering DPI

    Returns:
        List of visuals with normalized bboxes
    """
    page = doc[page_num - 1]
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    original_img = Image.open(io.BytesIO(pix.tobytes("png")))

    img_width, img_height = original_img.size
    doc_name = Path(doc.name).stem if doc.name else "doc"

    # Load fonts
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    # ===== PHASE 1: Column Detection =====
    img_phase1 = original_img.copy()
    draw = ImageDraw.Draw(img_phase1)

    # Draw red line at 50% to mark column boundary
    midpoint = img_width // 2
    draw.line([(midpoint, 0), (midpoint, img_height)], fill="red", width=5)
    draw.text((midpoint // 2, 20), "LEFT", fill="red", font=font, anchor="mm")
    draw.text((midpoint + midpoint // 2, 20), "RIGHT", fill="red", font=font, anchor="mm")

    if output_dir:
        phase1_path = Path(output_dir) / f"{doc_name}_p{page_num}_phase1_columns.png"
        phase1_path.parent.mkdir(parents=True, exist_ok=True)
        img_phase1.save(str(phase1_path))

    # Convert to base64 and send to VLM
    buffer = io.BytesIO()
    img_phase1.save(buffer, format="PNG")
    base64_phase1 = base64.b64encode(buffer.getvalue()).decode("utf-8")

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
                                "data": base64_phase1,
                            },
                        },
                        {
                            "type": "text",
                            "text": PHASE1_COLUMN_PROMPT,
                        },
                    ],
                }
            ],
        )

        content_block = response.content[0]
        raw_text = content_block.text if hasattr(content_block, "text") else str(content_block)

        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if not json_match:
            logger.warning(f"No JSON in phase 1 response for page {page_num}")
            return []

        phase1_data = json.loads(json_match.group())
        phase1_visuals = phase1_data.get("visuals", [])

        if not phase1_visuals:
            logger.info(f"Page {page_num}: No visuals found in phase 1")
            return []

        logger.info(f"Page {page_num} phase 1: {len(phase1_visuals)} visuals found")

    except Exception as e:
        logger.error(f"Phase 1 failed for page {page_num}: {e}")
        return []

    # ===== PHASE 2: Column-Specific Grid Detection =====
    result_visuals = []
    row_labels = [chr(ord('A') + i) for i in range(grid_rows)]
    col_labels = [str(i + 1) for i in range(grid_cols)]

    # Group visuals by column side
    by_side = {"left": [], "right": [], "spans_both": []}
    for v in phase1_visuals:
        side = v.get("side", "spans_both").lower()
        if side in by_side:
            by_side[side].append(v)
        else:
            by_side["spans_both"].append(v)

    # Process each column that has visuals
    for side in ["left", "right"]:
        if not by_side[side]:
            continue

        # Create column-specific grid image
        img_phase2 = original_img.copy()
        draw = ImageDraw.Draw(img_phase2)

        if side == "left":
            col_start = 0
            col_end = midpoint
            other_start = midpoint
            other_end = img_width
        else:
            col_start = midpoint
            col_end = img_width
            other_start = 0
            other_end = midpoint

        col_width = col_end - col_start
        cell_width = col_width / grid_cols
        cell_height = img_height / grid_rows

        # Mark other column as "ignore"
        draw.rectangle([other_start, 0, other_end, img_height], outline="red", width=3)
        other_label = "LEFT" if side == "right" else "RIGHT"
        draw.text(((other_start + other_end) // 2, 30), f"{other_label} COLUMN", fill="red", font=font, anchor="mm")
        draw.text(((other_start + other_end) // 2, 60), "(ignore)", fill="red", font=small_font, anchor="mm")

        # Draw green grid on target column
        for c in range(grid_cols + 1):
            x = col_start + int(c * cell_width)
            draw.line([(x, 0), (x, img_height)], fill="green", width=2)

        for r in range(grid_rows + 1):
            y = int(r * cell_height)
            draw.line([(col_start, y), (col_end, y)], fill="green", width=2)

        # Labels
        for c in range(grid_cols):
            cx = col_start + int(c * cell_width + cell_width / 2)
            draw.text((cx, 10), col_labels[c], fill="green", font=font, anchor="mm")

        for r in range(grid_rows):
            cy = int(r * cell_height + cell_height / 2)
            draw.text((col_start + 15, cy), row_labels[r], fill="green", font=small_font, anchor="mm")

        if output_dir:
            phase2_path = Path(output_dir) / f"{doc_name}_p{page_num}_phase2_{side}_grid.png"
            img_phase2.save(str(phase2_path))

        # Send to VLM
        buffer = io.BytesIO()
        img_phase2.save(buffer, format="PNG")
        base64_phase2 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        prompt = get_column_grid_prompt(grid_rows, grid_cols, side)

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
                                    "data": base64_phase2,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )

            content_block = response.content[0]
            raw_text = content_block.text if hasattr(content_block, "text") else str(content_block)

            json_match = re.search(r'\{[\s\S]*\}', raw_text)
            if not json_match:
                continue

            phase2_data = json.loads(json_match.group())

            for v in phase2_data.get("visuals", []):
                top_row = v.get("top_row", "A")
                bottom_row = v.get("bottom_row", row_labels[-1])
                left_col = v.get("left_col", "1")
                right_col = v.get("right_col", col_labels[-1])

                # Parse row/col indices
                top_idx = row_labels.index(top_row) if top_row in row_labels else 0
                bottom_idx = row_labels.index(bottom_row) if bottom_row in row_labels else grid_rows - 1
                left_idx = col_labels.index(left_col) if left_col in col_labels else 0
                right_idx = col_labels.index(right_col) if right_col in col_labels else grid_cols - 1

                # Convert to full-page normalized coordinates
                if side == "left":
                    x0_page = (left_idx / grid_cols) * 0.5
                    x1_page = ((right_idx + 1) / grid_cols) * 0.5
                else:
                    x0_page = 0.5 + (left_idx / grid_cols) * 0.5
                    x1_page = 0.5 + ((right_idx + 1) / grid_cols) * 0.5

                y0_page = top_idx / grid_rows
                y1_page = (bottom_idx + 1) / grid_rows

                result_visuals.append({
                    "type": v.get("type", "figure"),
                    "label": v.get("label"),
                    "bbox": [x0_page, y0_page, x1_page, y1_page],
                    "column": side,
                    "grid_cells": f"{top_row}{left_col} to {bottom_row}{right_col}",
                })

        except Exception as e:
            logger.error(f"Phase 2 ({side}) failed for page {page_num}: {e}")

    # Process full-width visuals (spans_both)
    for v in by_side["spans_both"]:
        # Use a simple vertical position mapping for full-width
        vert = v.get("vertical_position", "middle").lower()
        vert_map = {
            "top": (0.0, 0.4),
            "upper": (0.1, 0.5),
            "middle": (0.3, 0.7),
            "lower": (0.5, 0.9),
            "bottom": (0.6, 1.0),
        }
        y0, y1 = vert_map.get(vert, (0.2, 0.8))

        result_visuals.append({
            "type": v.get("type", "figure"),
            "label": v.get("label"),
            "bbox": [0.05, y0, 0.95, y1],
            "column": "full",
            "grid_cells": f"full-width ({vert})",
        })

    # Draw final result image
    if output_dir and result_visuals:
        result_img = original_img.copy()
        draw = ImageDraw.Draw(result_img)

        for v in result_visuals:
            bbox = v.get("bbox", [])
            if len(bbox) == 4:
                x0, y0, x1, y1 = bbox
                px0 = int(x0 * img_width)
                py0 = int(y0 * img_height)
                px1 = int(x1 * img_width)
                py1 = int(y1 * img_height)
                color = "red" if v.get("type") == "figure" else "blue"
                draw.rectangle([px0, py0, px1, py1], outline=color, width=4)

                # Label
                label = v.get("label") or v.get("type")
                draw.text((px0 + 5, py0 + 5), label, fill=color, font=font)

        result_path = Path(output_dir) / f"{doc_name}_p{page_num}_two_phase_result.png"
        result_img.save(str(result_path))

    return result_visuals


def draw_debug_rectangles(
    doc: fitz.Document,
    page_num: int,
    visuals: List[dict],
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> Image.Image:
    """
    Draw red rectangles around detected visuals on a page image.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        visuals: List of visual dicts with bbox field
        output_path: Optional path to save the annotated image
        dpi: Rendering DPI

    Returns:
        PIL Image with rectangles drawn
    """
    # Render page to image
    page = doc[page_num - 1]
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    # Convert to PIL Image
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    draw = ImageDraw.Draw(img)

    img_width, img_height = img.size

    # Try to load a font for labels
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()

    # Draw rectangles for each visual
    for visual in visuals:
        bbox = visual.get("bbox", [])
        if len(bbox) != 4:
            continue

        x0, y0, x1, y1 = bbox

        # Convert normalized coords to pixels
        px0 = int(x0 * img_width)
        py0 = int(y0 * img_height)
        px1 = int(x1 * img_width)
        py1 = int(y1 * img_height)

        # Draw rectangle
        visual_type = visual.get("type", "unknown")
        color = "red" if visual_type == "figure" else "blue"

        draw.rectangle([px0, py0, px1, py1], outline=color, width=3)

        # Draw label
        label = visual.get("label", visual_type)
        conf = visual.get("confidence", 0)
        label_text = f"{label} ({conf:.0%})"

        # Draw label background
        text_bbox = draw.textbbox((px0, py0 - 20), label_text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((px0, py0 - 20), label_text, fill="white", font=font)

    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        logger.info(f"Saved debug image: {output_path}")

    return img


def analyze_document_debug(
    pdf_path: str,
    output_dir: str,
    client: Optional[anthropic.Anthropic] = None,
    model: str = "claude-sonnet-4-20250514",
    pages: Optional[List[int]] = None,
) -> List[dict]:
    """
    Analyze a document and save debug images with bounding boxes.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save debug images
        client: Anthropic client (created if not provided)
        model: Model to use
        pages: Optional list of specific pages (1-indexed)

    Returns:
        List of results per page with visuals found
    """
    if client is None:
        client = anthropic.Anthropic()

    doc = fitz.open(pdf_path)
    doc_name = Path(pdf_path).stem
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    try:
        page_nums = pages if pages else list(range(1, len(doc) + 1))

        for page_num in page_nums:
            if page_num < 1 or page_num > len(doc):
                continue

            logger.info(f"Debug analyzing page {page_num}/{len(doc)}")

            # Get bounding boxes from VLM
            visuals = analyze_page_with_bbox(doc, page_num, client, model)

            # Draw debug image
            img_path = output_path / f"{doc_name}_debug_p{page_num}.png"
            draw_debug_rectangles(doc, page_num, visuals, str(img_path))

            results.append({
                "page": page_num,
                "visuals": visuals,
                "debug_image": str(img_path),
            })

            logger.info(f"Page {page_num}: found {len(visuals)} visuals")

    finally:
        doc.close()

    return results


__all__ = [
    # Constants
    "VLM_LAYOUT_PROMPT",
    "VLM_BBOX_DEBUG_PROMPT",
    "VLM_REFINE_PROMPT",
    "PHASE1_COLUMN_PROMPT",
    # Functions
    "render_page_for_analysis",
    "analyze_page_layout",
    "parse_layout_response",
    "analyze_document_layouts",
    "analyze_page_with_bbox",
    "refine_bboxes",
    "analyze_page_with_refinement",
    "draw_grid_overlay",
    "parse_cell_reference",
    "grid_range_to_bbox",
    "grid_cells_to_bbox",
    "get_grid_prompt",
    "get_grid_prompt_for_style",
    "analyze_page_with_grid",
    "get_column_grid_prompt",
    "analyze_page_two_phase",
    "draw_debug_rectangles",
    "analyze_document_debug",
]
