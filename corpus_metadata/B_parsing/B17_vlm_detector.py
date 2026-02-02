# corpus_metadata/B_parsing/B17_vlm_detector.py
"""
VLM-Assisted Visual Detection.

Uses Claude Vision to detect tables and figures on PDF pages,
returning structured bounding box information.

Key capabilities:
- Detects multiple visuals per page
- Identifies multi-page tables (continuation markers)
- Handles 2-column layouts and full-width visuals
- Returns normalized bounding boxes in PDF points
"""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class VLMDetectedVisual:
    """A visual element detected by VLM."""

    visual_type: str  # "table" or "figure"
    bbox_normalized: Tuple[float, float, float, float]  # (x0, y0, x1, y1) as 0-1 fractions
    bbox_pts: Tuple[float, float, float, float]  # Converted to PDF points
    page_num: int
    confidence: float
    label: Optional[str]  # e.g., "Table 1", "Figure 2A"
    caption_snippet: Optional[str]  # First ~100 chars of caption if visible
    is_continuation: bool  # Part of multi-page visual
    continues_to_next: bool  # Continues on next page


@dataclass
class VLMPageDetection:
    """Detection results for a single page."""

    page_num: int
    visuals: List[VLMDetectedVisual]
    raw_response: str
    tokens_used: int


def render_page_for_vlm(
    doc: fitz.Document,
    page_num: int,
    max_dimension: int = 1568,  # Claude's recommended size
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


VLM_DETECTION_PROMPT = """Analyze this PDF page and identify ALL tables and figures.

For EACH visual element found, provide:
1. type: "table" or "figure"
2. bbox: bounding box as [x0, y0, x1, y1] where coordinates are fractions (0.0 to 1.0) of page dimensions
3. label: the reference label if visible (e.g., "Table 1", "Figure 2A")
4. caption_start: first ~50 characters of the caption if visible
5. is_continuation: true if this continues from previous page
6. continues_next: true if this continues to next page
7. confidence: 0.0 to 1.0

BBOX RULES:

1. The bbox should contain ONLY the visual element and its caption:
   - The visual itself (chart, graph, flowchart, table, diagram, image)
   - The figure/table label and caption (e.g., "Figure 1: Description...")
   - Axis labels, legends, and annotations that are part of the visual

2. The bbox should NOT include:
   - Article body text paragraphs (running text discussing results, methods, etc.)
   - Section headings (e.g., "Methods", "Results", "Discussion")
   - Text from adjacent columns in multi-column layouts

3. For multi-column layouts:
   - Determine if the visual is in one column or spans multiple columns
   - If there is article text running alongside the visual, the visual is in ONE column only
   - Only use full page width if the visual clearly spans across all columns

4. Multi-panel figures (A, B, C, etc.):
   - Include ALL panels in a single bbox
   - Panels stacked vertically in one column = single column figure
   - Panels spread across columns = full width figure

5. PRECISION: Draw tight bboxes around the actual visual content.
   - Start y0 at the top edge of the visual (first chart border, flowchart box, table line)
   - End y1 at the bottom of the caption
   - Do not include surrounding whitespace or article text

Visual types:
- Tables: data tables, comparison tables, results tables
- Figures: charts, graphs, plots, flowcharts, diagrams, photos, illustrations

Return ONLY valid JSON:
{
  "visuals": [
    {
      "type": "figure",
      "bbox": [0.05, 0.20, 0.48, 0.75],
      "label": "Figure 1",
      "caption_start": "Study flowchart showing...",
      "is_continuation": false,
      "continues_next": false,
      "confidence": 0.95
    }
  ],
  "page_layout": "two_column",
  "notes": "optional observations"
}

If no tables or figures found: {"visuals": [], "page_layout": "...", "notes": "..."}"""


def detect_visuals_vlm_single_page(
    doc: fitz.Document,
    page_num: int,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-20250514",
) -> VLMPageDetection:
    """
    Use VLM to detect visuals on a single page.

    Args:
        doc: Open PyMuPDF document
        page_num: 1-indexed page number
        client: Anthropic client
        model: Model to use

    Returns:
        VLMPageDetection with detected visuals
    """
    # Render page
    base64_image, page_width, page_height = render_page_for_vlm(doc, page_num)

    # Call VLM
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
                            "text": VLM_DETECTION_PROMPT,
                        },
                    ],
                }
            ],
        )

        raw_text = response.content[0].text
        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        # Parse JSON response
        visuals = parse_vlm_detection_response(raw_text, page_num, page_width, page_height)

        return VLMPageDetection(
            page_num=page_num,
            visuals=visuals,
            raw_response=raw_text,
            tokens_used=tokens_used,
        )

    except Exception as e:
        logger.error(f"VLM detection failed for page {page_num}: {e}")
        return VLMPageDetection(
            page_num=page_num,
            visuals=[],
            raw_response=str(e),
            tokens_used=0,
        )


def parse_vlm_detection_response(
    raw_text: str,
    page_num: int,
    page_width: float,
    page_height: float,
) -> List[VLMDetectedVisual]:
    """
    Parse VLM JSON response into VLMDetectedVisual objects.

    Args:
        raw_text: Raw text response from VLM
        page_num: Page number
        page_width: Page width in PDF points
        page_height: Page height in PDF points

    Returns:
        List of detected visuals
    """
    visuals = []

    try:
        # Extract JSON from response (may have markdown code blocks)
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if not json_match:
            logger.warning(f"No JSON found in VLM response for page {page_num}")
            return []

        data = json.loads(json_match.group())

        for item in data.get("visuals", []):
            bbox_norm = item.get("bbox", [0, 0, 1, 1])

            # Validate bbox
            if len(bbox_norm) != 4:
                continue

            x0, y0, x1, y1 = bbox_norm

            # Clamp to valid range
            x0 = max(0.0, min(1.0, float(x0)))
            y0 = max(0.0, min(1.0, float(y0)))
            x1 = max(0.0, min(1.0, float(x1)))
            y1 = max(0.0, min(1.0, float(y1)))

            # Ensure x0 < x1, y0 < y1
            if x0 >= x1 or y0 >= y1:
                continue

            # Convert to PDF points
            bbox_pts = (
                x0 * page_width,
                y0 * page_height,
                x1 * page_width,
                y1 * page_height,
            )

            visuals.append(VLMDetectedVisual(
                visual_type=item.get("type", "figure"),
                bbox_normalized=(x0, y0, x1, y1),
                bbox_pts=bbox_pts,
                page_num=page_num,
                confidence=float(item.get("confidence", 0.8)),
                label=item.get("label"),
                caption_snippet=item.get("caption_start"),
                is_continuation=bool(item.get("is_continuation", False)),
                continues_to_next=bool(item.get("continues_next", False)),
            ))

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse VLM JSON for page {page_num}: {e}")
    except Exception as e:
        logger.warning(f"Error parsing VLM response for page {page_num}: {e}")

    return visuals


def detect_visuals_vlm_document(
    pdf_path: str,
    client: Optional[anthropic.Anthropic] = None,
    model: str = "claude-sonnet-4-20250514",
    skip_pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Detect all visuals in a document using VLM.

    Args:
        pdf_path: Path to PDF file
        client: Anthropic client (created if not provided)
        model: Model to use
        skip_pages: Page numbers to skip (e.g., cover pages)

    Returns:
        Dict with detection results and statistics
    """
    if client is None:
        client = anthropic.Anthropic()

    skip_pages = skip_pages or []

    doc = fitz.open(pdf_path)
    all_visuals: List[VLMDetectedVisual] = []
    page_results: List[VLMPageDetection] = []
    total_tokens = 0

    try:
        for page_idx in range(doc.page_count):
            page_num = page_idx + 1

            if page_num in skip_pages:
                continue

            print(f"  VLM detecting page {page_num}/{doc.page_count}...", end=" ", flush=True)

            result = detect_visuals_vlm_single_page(doc, page_num, client, model)
            page_results.append(result)
            all_visuals.extend(result.visuals)
            total_tokens += result.tokens_used

            print(f"found {len(result.visuals)} visuals")

    finally:
        doc.close()

    # Count by type
    tables = [v for v in all_visuals if v.visual_type == "table"]
    figures = [v for v in all_visuals if v.visual_type == "figure"]

    return {
        "visuals": all_visuals,
        "page_results": page_results,
        "tables_detected": len(tables),
        "figures_detected": len(figures),
        "total_visuals": len(all_visuals),
        "total_tokens": total_tokens,
        "pages_processed": len(page_results),
    }


def compare_detections(
    vlm_visuals: List[VLMDetectedVisual],
    heuristic_visuals: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compare VLM detections with heuristic detections.

    Args:
        vlm_visuals: Visuals detected by VLM
        heuristic_visuals: Visuals detected by heuristic method
        iou_threshold: IoU threshold for matching

    Returns:
        Comparison statistics
    """
    def compute_iou(bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute Intersection over Union."""
        x0 = max(bbox1[0], bbox2[0])
        y0 = max(bbox1[1], bbox2[1])
        x1 = min(bbox1[2], bbox2[2])
        y1 = min(bbox1[3], bbox2[3])

        if x1 <= x0 or y1 <= y0:
            return 0.0

        intersection = (x1 - x0) * (y1 - y0)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    # Match VLM detections to heuristic detections
    matched_vlm = set()
    matched_heuristic = set()
    matches = []

    for i, vlm_v in enumerate(vlm_visuals):
        best_iou = 0
        best_j = -1

        for j, heur_v in enumerate(heuristic_visuals):
            if j in matched_heuristic:
                continue
            if vlm_v.page_num != heur_v.get("page_num"):
                continue

            iou = compute_iou(vlm_v.bbox_pts, heur_v.get("bbox_pts", (0,0,0,0)))
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold:
            matched_vlm.add(i)
            matched_heuristic.add(best_j)
            matches.append({
                "vlm_idx": i,
                "heuristic_idx": best_j,
                "iou": best_iou,
                "vlm_bbox": vlm_v.bbox_pts,
                "heuristic_bbox": heuristic_visuals[best_j].get("bbox_pts"),
            })

    # Compute statistics
    vlm_only = [vlm_visuals[i] for i in range(len(vlm_visuals)) if i not in matched_vlm]
    heuristic_only = [heuristic_visuals[j] for j in range(len(heuristic_visuals)) if j not in matched_heuristic]

    return {
        "matched_count": len(matches),
        "vlm_only_count": len(vlm_only),
        "heuristic_only_count": len(heuristic_only),
        "vlm_total": len(vlm_visuals),
        "heuristic_total": len(heuristic_visuals),
        "matches": matches,
        "vlm_only": vlm_only,
        "heuristic_only": heuristic_only,
        "average_iou": sum(m["iou"] for m in matches) / len(matches) if matches else 0,
    }


__all__ = [
    "VLMDetectedVisual",
    "VLMPageDetection",
    "detect_visuals_vlm_single_page",
    "detect_visuals_vlm_document",
    "compare_detections",
    "render_page_for_vlm",
]
