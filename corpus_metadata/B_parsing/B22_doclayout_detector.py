# corpus_metadata/B_parsing/B22_doclayout_detector.py
"""
DocLayout-YOLO Visual Detector.

Uses DocLayout-YOLO model for accurate figure/table detection.
No VLM required - pure YOLO-based detection.

Key features:
1. Accurate bounding boxes from YOLO detection
2. Caption extraction and association
3. High-resolution cropping with PyMuPDF clip
4. Fast inference (~0.5s per page)
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Category mapping from DocLayout-YOLO
DOCLAYOUT_CATEGORIES = {
    0: "title",
    1: "plain_text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}

# Categories we want to extract
VISUAL_CATEGORIES = {"figure", "table"}
CAPTION_CATEGORIES = {"figure_caption", "table_caption"}


@dataclass
class DetectedCaption:
    """A caption detected by DocLayout-YOLO."""

    page_num: int
    caption_type: str  # "figure_caption" or "table_caption"
    confidence: float
    bbox_pts: Tuple[float, float, float, float]
    text: str = ""


@dataclass
class DocLayoutVisual:
    """A visual detected by DocLayout-YOLO."""

    page_num: int
    visual_type: str  # "figure" or "table"
    confidence: float
    bbox_pts: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF points
    bbox_normalized: Tuple[float, float, float, float]  # (x0, y0, x1, y1) as 0-1 fractions

    # Caption info
    caption_text: Optional[str] = None
    caption_bbox: Optional[Tuple[float, float, float, float]] = None
    caption_position: Optional[str] = None  # "above" or "below"


@dataclass
class DocLayoutResult:
    """Result of DocLayout-YOLO detection on a document."""

    visuals: List[DocLayoutVisual]
    tables_detected: int
    figures_detected: int
    pages_processed: int


def _load_doclayout_model():
    """Load DocLayout-YOLO model from HuggingFace Hub."""
    try:
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )
        model = YOLOv10(model_path)
        logger.info("DocLayout-YOLO model loaded successfully")
        return model
    except ImportError as e:
        logger.error(f"Failed to import DocLayout-YOLO: {e}")
        raise ImportError(
            "DocLayout-YOLO not installed. Install with: pip install doclayout-yolo"
        ) from e


# Lazy-loaded model singleton
_model = None


def get_model():
    """Get the DocLayout-YOLO model (lazy loading)."""
    global _model
    if _model is None:
        _model = _load_doclayout_model()
    return _model


def _extract_text_from_bbox(
    doc: fitz.Document,
    page_num: int,
    bbox_pts: Tuple[float, float, float, float],
) -> str:
    """Extract text from a bounding box using PyMuPDF."""
    page = doc[page_num - 1]
    rect = fitz.Rect(bbox_pts)
    text = page.get_text("text", clip=rect)
    return text.strip()


def _associate_captions_with_visuals(
    visuals: List[Dict],
    captions: List[Dict],
    page_height: float,
    max_distance_pts: float = 50.0,
) -> List[Dict]:
    """
    Associate captions with their corresponding visuals.

    Strategy:
    1. For each visual, find the closest caption of matching type
    2. Caption must be within max_distance_pts vertically
    3. Caption must overlap horizontally with the visual
    """
    for visual in visuals:
        v_type = visual["type"]
        v_bbox = visual["bbox_pts"]
        v_x0, v_y0, v_x1, v_y1 = v_bbox

        # Find matching caption type
        expected_caption_type = f"{v_type}_caption"

        best_caption = None
        best_distance = float("inf")

        for caption in captions:
            if caption["type"] != expected_caption_type:
                continue

            c_bbox = caption["bbox_pts"]
            c_x0, c_y0, c_x1, c_y1 = c_bbox

            # Check horizontal overlap
            h_overlap = min(v_x1, c_x1) - max(v_x0, c_x0)
            if h_overlap < 0:
                continue  # No horizontal overlap

            # Check vertical distance and position
            if c_y1 <= v_y0:
                # Caption is above visual
                distance = v_y0 - c_y1
                position = "above"
            elif c_y0 >= v_y1:
                # Caption is below visual
                distance = c_y0 - v_y1
                position = "below"
            else:
                # Caption overlaps with visual (unusual)
                continue

            if distance < max_distance_pts and distance < best_distance:
                best_distance = distance
                best_caption = {
                    "text": caption["text"],
                    "bbox_pts": c_bbox,
                    "position": position,
                }

        if best_caption:
            visual["caption_text"] = best_caption["text"]
            visual["caption_bbox"] = best_caption["bbox_pts"]
            visual["caption_position"] = best_caption["position"]

    return visuals


def detect_visuals_doclayout(
    pdf_path: str,
    detect_dpi: int = 144,
    confidence_threshold: float = 0.3,
    imgsz: int = 1024,
) -> DocLayoutResult:
    """
    Detect figures and tables using DocLayout-YOLO.

    Args:
        pdf_path: Path to PDF file
        detect_dpi: DPI for rendering pages for detection (lower = faster)
        confidence_threshold: Minimum confidence for detection
        imgsz: Image size for YOLO inference

    Returns:
        DocLayoutResult with detected visuals
    """
    model = get_model()
    doc = fitz.open(pdf_path)

    all_visuals: List[DocLayoutVisual] = []
    tables_detected = 0
    figures_detected = 0
    total_pages = doc.page_count

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for page_idx in range(doc.page_count):
                page_num = page_idx + 1
                page = doc[page_idx]

                # Get page dimensions
                page_width_pts = page.rect.width
                page_height_pts = page.rect.height

                # Render page for detection
                detect_matrix = fitz.Matrix(detect_dpi / 72, detect_dpi / 72)
                pix = page.get_pixmap(matrix=detect_matrix)
                img_path = Path(tmpdir) / f"page_{page_num}.png"
                pix.save(str(img_path))

                img_width_px = pix.width
                img_height_px = pix.height

                # Run YOLO detection
                results = model.predict(
                    str(img_path),
                    imgsz=imgsz,
                    conf=confidence_threshold,
                    verbose=False,
                )

                # Collect visuals and captions separately
                page_visuals = []
                page_captions = []

                # Process detections
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue

                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        category = DOCLAYOUT_CATEGORIES.get(cls_id, "unknown")
                        conf = float(boxes.conf[i].item())

                        # Get pixel coordinates
                        x0_px, y0_px, x1_px, y1_px = boxes.xyxy[i].tolist()

                        # Convert to PDF points
                        x0_pts = x0_px * page_width_pts / img_width_px
                        y0_pts = y0_px * page_height_pts / img_height_px
                        x1_pts = x1_px * page_width_pts / img_width_px
                        y1_pts = y1_px * page_height_pts / img_height_px

                        bbox_pts = (x0_pts, y0_pts, x1_pts, y1_pts)

                        if category in VISUAL_CATEGORIES:
                            page_visuals.append({
                                "type": category,
                                "confidence": conf,
                                "bbox_pts": bbox_pts,
                            })

                            if category == "table":
                                tables_detected += 1
                            else:
                                figures_detected += 1

                        elif category in CAPTION_CATEGORIES:
                            # Extract caption text
                            caption_text = _extract_text_from_bbox(
                                doc, page_num, bbox_pts
                            )
                            page_captions.append({
                                "type": category,
                                "confidence": conf,
                                "bbox_pts": bbox_pts,
                                "text": caption_text,
                            })

                # Associate captions with visuals
                page_visuals = _associate_captions_with_visuals(
                    page_visuals, page_captions, page_height_pts
                )

                # Create DocLayoutVisual objects
                for v in page_visuals:
                    bbox = v["bbox_pts"]
                    x0_norm = bbox[0] / page_width_pts
                    y0_norm = bbox[1] / page_height_pts
                    x1_norm = bbox[2] / page_width_pts
                    y1_norm = bbox[3] / page_height_pts

                    visual = DocLayoutVisual(
                        page_num=page_num,
                        visual_type=v["type"],
                        confidence=v["confidence"],
                        bbox_pts=bbox,
                        bbox_normalized=(x0_norm, y0_norm, x1_norm, y1_norm),
                        caption_text=v.get("caption_text"),
                        caption_bbox=v.get("caption_bbox"),
                        caption_position=v.get("caption_position"),
                    )
                    all_visuals.append(visual)

                    logger.debug(
                        f"Page {page_num}: {v['type']} @ "
                        f"({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}) "
                        f"conf={v['confidence']:.3f}"
                        + (f" caption='{v.get('caption_text', '')[:50]}...'" if v.get("caption_text") else "")
                    )

                logger.info(
                    f"Page {page_num}: {len(page_visuals)} visuals, "
                    f"{len(page_captions)} captions"
                )

    finally:
        doc.close()

    return DocLayoutResult(
        visuals=all_visuals,
        tables_detected=tables_detected,
        figures_detected=figures_detected,
        pages_processed=total_pages,
    )


def crop_visual_highres(
    pdf_path: str,
    page_num: int,
    bbox_pts: Tuple[float, float, float, float],
    crop_dpi: int = 300,
    padding_pts: float = 5.0,
) -> Tuple[bytes, int, int]:
    """
    Crop a visual at high resolution using PyMuPDF's clip parameter.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        bbox_pts: Bounding box in PDF points (x0, y0, x1, y1)
        crop_dpi: DPI for high-resolution crop
        padding_pts: Padding in PDF points

    Returns:
        Tuple of (PNG bytes, width_px, height_px)
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num - 1]

        # Add padding
        x0, y0, x1, y1 = bbox_pts
        x0 = max(0, x0 - padding_pts)
        y0 = max(0, y0 - padding_pts)
        x1 = min(page.rect.width, x1 + padding_pts)
        y1 = min(page.rect.height, y1 + padding_pts)

        clip_rect = fitz.Rect(x0, y0, x1, y1)

        # Render at high DPI with clip
        crop_matrix = fitz.Matrix(crop_dpi / 72, crop_dpi / 72)
        pix = page.get_pixmap(matrix=crop_matrix, clip=clip_rect)

        return pix.tobytes("png"), pix.width, pix.height

    finally:
        doc.close()


def detect_and_crop_all(
    pdf_path: str,
    output_dir: str,
    detect_dpi: int = 144,
    crop_dpi: int = 300,
    confidence_threshold: float = 0.3,
) -> Dict:
    """
    Detect all visuals and save high-resolution crops.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save crops and layout.json
        detect_dpi: DPI for detection
        crop_dpi: DPI for high-resolution crops
        confidence_threshold: Minimum confidence

    Returns:
        Dict with detection results and file paths
    """
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Detect visuals
    result = detect_visuals_doclayout(
        pdf_path,
        detect_dpi=detect_dpi,
        confidence_threshold=confidence_threshold,
    )

    # Get page sizes
    doc = fitz.open(pdf_path)
    page_sizes = {}
    for i in range(doc.page_count):
        page = doc[i]
        page_sizes[i + 1] = (page.rect.width, page.rect.height)
    doc.close()

    # Build layout.json structure
    layout = {
        "pdf_path": pdf_path,
        "pages": [],
    }

    # Group visuals by page
    page_visuals: Dict[int, List[DocLayoutVisual]] = {}
    for v in result.visuals:
        if v.page_num not in page_visuals:
            page_visuals[v.page_num] = []
        page_visuals[v.page_num].append(v)

    # Process each page
    for page_num in sorted(page_visuals.keys()):
        visuals_on_page = page_visuals[page_num]
        page_width, page_height = page_sizes[page_num]

        page_entry = {
            "page_num": page_num,
            "page_size_pts": [page_width, page_height],
            "visuals": [],
        }

        # Count visuals by type for filename
        type_counters = {"figure": 0, "table": 0}

        for v in visuals_on_page:
            type_counters[v.visual_type] += 1
            idx = type_counters[v.visual_type]

            # Generate filename
            crop_filename = f"p{page_num}_{v.visual_type}_{idx}.png"
            crop_path = output_path / crop_filename

            # Crop at high resolution
            png_bytes, width_px, height_px = crop_visual_highres(
                pdf_path,
                page_num,
                v.bbox_pts,
                crop_dpi=crop_dpi,
            )

            # Save crop
            crop_path.write_bytes(png_bytes)

            # Build visual entry with caption
            visual_entry = {
                "type": v.visual_type,
                "confidence": round(v.confidence, 3),
                "bbox_pts": [round(x, 1) for x in v.bbox_pts],
                "bbox_normalized": [round(x, 4) for x in v.bbox_normalized],
                "crop_file": crop_filename,
                "crop_size_px": [width_px, height_px],
                "crop_dpi": crop_dpi,
            }

            # Add caption info if available
            if v.caption_text:
                visual_entry["caption"] = {
                    "text": v.caption_text,
                    "position": v.caption_position,
                }
                if v.caption_bbox:
                    visual_entry["caption"]["bbox_pts"] = [
                        round(x, 1) for x in v.caption_bbox
                    ]

            page_entry["visuals"].append(visual_entry)

        layout["pages"].append(page_entry)

    # Save layout.json
    layout_path = output_path / "layout.json"
    with open(layout_path, "w") as f:
        json.dump(layout, f, indent=2)

    logger.info(
        f"Extracted {result.figures_detected} figures, "
        f"{result.tables_detected} tables to {output_dir}"
    )

    return {
        "layout_path": str(layout_path),
        "output_dir": str(output_dir),
        "figures_detected": result.figures_detected,
        "tables_detected": result.tables_detected,
        "visuals": result.visuals,
    }


__all__ = [
    "DocLayoutVisual",
    "DocLayoutResult",
    "DetectedCaption",
    "detect_visuals_doclayout",
    "crop_visual_highres",
    "detect_and_crop_all",
]
