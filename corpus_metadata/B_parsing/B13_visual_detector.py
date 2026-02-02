# corpus_metadata/B_parsing/B13_visual_detector.py
"""
Visual detector with FAST/ACCURATE tiering using Docling TableFormer.

This module detects tables and figures using Docling with tiered TableFormer
modes: FAST for initial extraction and ACCURATE for complex tables (escalated
based on complexity signals). It integrates native PDF figure detection and
provides multi-page table detection capabilities.

Key Components:
    - DetectorConfig: Configuration for TableFormer mode, escalation, and OCR
    - detect_tables_with_docling: Detect tables using Docling backend
    - detect_all_visuals: Main detection function returning DetectionResult
    - DetectionResult: Detection results with candidates and statistics
    - TableComplexitySignals: Signals for escalation decision (header depth, merged cells)

Example:
    >>> from B_parsing.B13_visual_detector import detect_all_visuals, DetectorConfig
    >>> config = DetectorConfig(default_table_mode="fast", enable_escalation=True)
    >>> result = detect_all_visuals("paper.pdf", config)
    >>> print(f"Detected {result.tables_detected} tables, {result.escalated_tables} escalated")

Dependencies:
    - A_core.A13_visual_models: TableComplexitySignals, TableStructure, VisualCandidate
    - B_parsing.B15_caption_extractor: ColumnLayout, caption extraction
    - B_parsing.B16_triage: is_in_margin_zone, should_escalate_to_accurate
    - fitz (PyMuPDF): PDF rendering
    - docling: TableFormer-based table extraction
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from A_core.A13_visual_models import (
    TableComplexitySignals,
    TableStructure,
    VisualCandidate,
)
from B_parsing.B15_caption_extractor import (
    ColumnLayout,
    extract_all_captions_on_page,
    infer_column_layout,
)
from B_parsing.B16_triage import is_in_margin_zone, should_escalate_to_accurate

logger = logging.getLogger(__name__)


# -------------------------
# Configuration
# -------------------------


@dataclass
class DetectorConfig:
    """Configuration for visual detection."""

    # TableFormer mode
    default_table_mode: str = "fast"  # "fast" or "accurate"
    enable_escalation: bool = True

    # Escalation thresholds
    escalation_config: Dict[str, Any] = field(default_factory=lambda: {
        "header_depth_threshold": 3,
        "merged_cell_threshold": 5,
        "token_coverage_threshold": 0.70,
        "large_table_cols": 8,
        "large_table_rows": 15,
    })

    # Figure detection
    min_figure_area_ratio: float = 0.02  # Minimum 2% of page
    filter_noise: bool = True
    repeat_threshold: int = 3

    # OCR settings (for scanned docs)
    enable_ocr: bool = True
    ocr_backend: str = "surya"  # "surya" or "easyocr"

    # Docling settings
    do_cell_matching: bool = True


# -------------------------
# Docling Integration
# -------------------------


def detect_tables_with_docling(
    pdf_path: str,
    mode: str = "fast",
    config: DetectorConfig = DetectorConfig(),
) -> List[Dict[str, Any]]:
    """
    Detect and extract tables using Docling.

    Args:
        pdf_path: Path to PDF file
        mode: "fast" or "accurate" TableFormer mode
        config: Detector configuration

    Returns:
        List of table detection results with structure
    """
    try:
        # Import Docling components
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableFormerMode,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        # Try to import SuryaOcrOptions for better OCR
        try:
            from docling_surya import SuryaOcrOptions
            has_surya = True
        except ImportError:
            has_surya = False
            logger.info("SuryaOCR not available, using default OCR")

        # Configure pipeline (follow pattern from B03c_docling_backend.py)
        if config.enable_ocr and has_surya:
            pipeline_options = PdfPipelineOptions(
                do_table_structure=True,
                do_ocr=True,
                allow_external_plugins=True,
                ocr_options=SuryaOcrOptions(lang=["en"]),
            )
        else:
            pipeline_options = PdfPipelineOptions(
                do_table_structure=True,
                do_ocr=config.enable_ocr,
            )

        # Set TableFormer mode on existing options object
        if mode == "accurate":
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        else:
            pipeline_options.table_structure_options.mode = TableFormerMode.FAST

        # Set cell matching option
        pipeline_options.table_structure_options.do_cell_matching = config.do_cell_matching

        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }

        # Create converter and process
        converter = DocumentConverter(format_options=format_options)
        result = converter.convert(pdf_path)

        # Extract tables from result
        tables = []
        document = result.document
        for item in document.tables:
            table_data = _extract_table_from_docling(item, document, mode)
            if table_data:
                tables.append(table_data)

        return tables

    except ImportError as e:
        logger.warning(
            f"Docling not available: {e}. "
            "Table extraction disabled - only native figure detection will work. "
            "Install 'docling' package to enable table extraction."
        )
        print(
            f"  [WARN] Docling not available: {e}\n"
            "         Table extraction disabled. Only figures will be extracted."
        )
        return []
    except Exception as e:
        logger.error(f"Docling table extraction failed: {e}")
        print(f"  [ERROR] Docling table extraction failed: {e}")
        return []


def _extract_table_from_docling(
    table_item: Any,
    document: Any,
    mode: str,
) -> Optional[Dict[str, Any]]:
    """
    Extract table data from a Docling TableItem.

    Args:
        table_item: Docling TableItem
        document: DoclingDocument for DataFrame export
        mode: Extraction mode used

    Returns:
        Dict with table data
    """
    try:
        # Import CoordOrigin for coordinate system check
        from docling_core.types.doc.base import CoordOrigin

        # Get provenance for page/bbox
        prov = table_item.prov[0] if table_item.prov else None
        if not prov:
            return None

        page_num = prov.page_no  # 1-indexed
        bbox = prov.bbox  # Docling bbox format

        # Get page height for coordinate conversion
        page_height = 792.0  # Default letter size
        if page_num in document.pages:
            page_info = document.pages[page_num]
            if hasattr(page_info, 'size') and page_info.size:
                page_height = page_info.size.height

        # Convert bbox to top-left origin (x0, y0, x1, y1) for PyMuPDF compatibility
        if hasattr(bbox, "l"):
            # Check coordinate origin and convert if needed
            if hasattr(bbox, 'coord_origin') and bbox.coord_origin == CoordOrigin.BOTTOMLEFT:
                # Convert from bottom-left to top-left origin
                # In bottom-left: y increases upward, t > b for a box
                # In top-left: y increases downward, t < b for a box
                x0 = min(bbox.l, bbox.r)
                x1 = max(bbox.l, bbox.r)
                # Flip Y coordinates: new_y = page_height - old_y
                y0 = page_height - max(bbox.t, bbox.b)  # top in top-left = page_height - top in bottom-left
                y1 = page_height - min(bbox.t, bbox.b)  # bottom in top-left = page_height - bottom in bottom-left
                bbox_pts = (x0, y0, x1, y1)
                logger.debug(f"Converted bbox from BOTTOMLEFT: ({bbox.l:.1f},{bbox.t:.1f},{bbox.r:.1f},{bbox.b:.1f}) -> ({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})")
            else:
                # Already top-left origin
                bbox_pts = (
                    min(bbox.l, bbox.r),
                    min(bbox.t, bbox.b),
                    max(bbox.l, bbox.r),
                    max(bbox.t, bbox.b),
                )
        else:
            bbox_pts = tuple(bbox)

        # Extract structure using export_to_dataframe (correct Docling API)
        headers = []
        rows = []

        try:
            df = table_item.export_to_dataframe(doc=document)
            headers = [[str(col) for col in df.columns.tolist()]]
            rows = [[str(cell) if cell is not None else "" for cell in row] for row in df.values.tolist()]
        except Exception as df_err:
            logger.debug(f"DataFrame export failed: {df_err}, trying fallback")

        # Count merged cells
        merged_count = 0
        header_depth = 1

        if hasattr(table_item, "table_cells"):
            for cell in table_item.table_cells:
                if cell.row_span > 1 or cell.col_span > 1:
                    merged_count += 1
                if cell.is_header:
                    header_depth = max(header_depth, cell.row_index + 1)

        return {
            "page_num": page_num,
            "bbox_pts": bbox_pts,
            "headers": headers,
            "rows": rows,
            "mode": mode,
            "merged_cell_count": merged_count,
            "header_depth": header_depth,
            "column_count": len(headers[0]) if headers else 0,
            "row_count": len(rows),
        }

    except Exception as e:
        logger.warning(f"Failed to extract table from Docling: {e}")
        return None


# -------------------------
# Native Figure Detection
# -------------------------


def detect_figures_native(
    pdf_path: str,
    config: DetectorConfig = DetectorConfig(),
) -> List[Dict[str, Any]]:
    """
    Detect figures using native PDF extraction (raster + vector).

    Args:
        pdf_path: Path to PDF file
        config: Detector configuration

    Returns:
        List of figure detection results
    """
    figures = []

    try:
        doc = fitz.open(pdf_path)

        # Track image hashes for deduplication
        hash_counts: Dict[str, int] = {}

        # Import vector detection from B09
        from B_parsing.B09_pdf_native_figures import detect_vector_figures

        for page_idx in range(doc.page_count):
            page_num = page_idx + 1
            page = doc[page_idx]
            page_width = page.rect.width
            page_height = page.rect.height
            page_area = page_width * page_height

            # Get embedded images
            image_list = page.get_images(full=True)

            for img_info in image_list:
                xref = img_info[0]

                try:
                    # Get image rect (position on page)
                    img_rects = page.get_image_rects(xref)
                    if not img_rects:
                        continue

                    rect = img_rects[0]
                    bbox_pts = (rect.x0, rect.y0, rect.x1, rect.y1)

                    # Compute area ratio
                    area = (rect.x1 - rect.x0) * (rect.y1 - rect.y0)
                    area_ratio = area / page_area

                    # Skip tiny images
                    if area_ratio < config.min_figure_area_ratio:
                        continue

                    # Compute image hash for deduplication
                    base_image = doc.extract_image(xref)
                    if base_image:
                        img_hash = hashlib.sha1(base_image["image"]).hexdigest()
                        hash_counts[img_hash] = hash_counts.get(img_hash, 0) + 1
                    else:
                        img_hash = None

                    # Check if in margin zone
                    in_margin = is_in_margin_zone(bbox_pts, page_height)

                    figures.append({
                        "page_num": page_num,
                        "bbox_pts": bbox_pts,
                        "area_ratio": area_ratio,
                        "xref": xref,
                        "image_hash": img_hash,
                        "in_margin_zone": in_margin,
                        "source": "native_raster",
                    })

                except Exception as e:
                    logger.debug(f"Failed to process image xref {xref}: {e}")
                    continue

            # Detect vector figures on this page (charts, flowcharts rendered as drawings)
            vector_figs = detect_vector_figures(doc, page_num)

            # Get text blocks for text density filtering
            text_dict = page.get_text("dict")
            text_blocks = [
                b for b in text_dict.get("blocks", [])
                if b.get("type") == 0  # Text blocks only
            ]

            # Detect column layout for column-aware filtering
            column_layout = infer_column_layout(doc, page_num)
            is_two_column = len(column_layout.columns) >= 2
            if is_two_column:
                col_gutter_x = (column_layout.columns[0][1] + column_layout.columns[1][0]) / 2

            for vf in vector_figs:
                fig_width = vf.bbox[2] - vf.bbox[0]
                fig_height = vf.bbox[3] - vf.bbox[1]
                area = fig_width * fig_height
                area_ratio = area / page_area

                # Text density filter: skip regions with high text coverage
                text_area_in_region = 0
                for tb in text_blocks:
                    tb_bbox = tb.get("bbox", (0, 0, 0, 0))
                    # Calculate intersection
                    ix0 = max(vf.bbox[0], tb_bbox[0])
                    iy0 = max(vf.bbox[1], tb_bbox[1])
                    ix1 = min(vf.bbox[2], tb_bbox[2])
                    iy1 = min(vf.bbox[3], tb_bbox[3])
                    if ix1 > ix0 and iy1 > iy0:
                        text_area_in_region += (ix1 - ix0) * (iy1 - iy0)

                text_density = text_area_in_region / area if area > 0 else 0
                # Skip if >40% of the region is covered by text blocks (likely article text)
                if text_density > 0.40:
                    logger.debug(f"Skipping vector region on page {page_num} - high text density ({text_density:.1%})")
                    continue

                # Column-aware constraint: for 2-column layouts, constrain to column width
                adjusted_bbox = vf.bbox
                if is_two_column:
                    # Check if figure spans the column gutter
                    if vf.bbox[0] < col_gutter_x < vf.bbox[2]:
                        # Figure spans gutter - only allow if it's wide enough (>60% page width)
                        # indicating it's intentionally full-width
                        if fig_width < page_width * 0.60:
                            # Constrain to the column with more drawing content
                            left_area = (min(col_gutter_x, vf.bbox[2]) - vf.bbox[0]) * fig_height
                            right_area = (vf.bbox[2] - max(col_gutter_x, vf.bbox[0])) * fig_height
                            if left_area >= right_area:
                                adjusted_bbox = (vf.bbox[0], vf.bbox[1], col_gutter_x - 10, vf.bbox[3])
                            else:
                                adjusted_bbox = (col_gutter_x + 10, vf.bbox[1], vf.bbox[2], vf.bbox[3])
                            logger.debug(f"Constrained vector figure to column: {vf.bbox} -> {adjusted_bbox}")

                # For full-width figures (>70% page width), constrain height
                fig_width = adjusted_bbox[2] - adjusted_bbox[0]
                fig_height = adjusted_bbox[3] - adjusted_bbox[1]
                if fig_width > page_width * 0.70:
                    # Limit height to 55% of page for full-width figures
                    max_height = page_height * 0.55
                    if fig_height > max_height:
                        new_y1 = adjusted_bbox[1] + max_height
                        adjusted_bbox = (adjusted_bbox[0], adjusted_bbox[1], adjusted_bbox[2], new_y1)

                # Recalculate after adjustment
                adj_width = adjusted_bbox[2] - adjusted_bbox[0]
                adj_height = adjusted_bbox[3] - adjusted_bbox[1]
                adj_area = adj_width * adj_height
                area_ratio = adj_area / page_area

                # Filter out if still too large (>65% of page area)
                if area_ratio > 0.65:
                    continue

                # Filter out very small adjusted regions
                if adj_area < page_area * 0.02:
                    continue

                figures.append({
                    "page_num": page_num,
                    "bbox_pts": adjusted_bbox,
                    "area_ratio": area_ratio,
                    "xref": None,
                    "image_hash": None,
                    "in_margin_zone": False,
                    "source": "native_vector",
                    "drawing_count": vf.drawing_count,
                    "has_axis_text": vf.has_axis_text,
                })

        doc.close()

        # Filter repeated images (logos, headers)
        if config.filter_noise:
            repeated_hashes = {
                h for h, c in hash_counts.items() if c >= config.repeat_threshold
            }
            figures = [
                f for f in figures
                if f["image_hash"] not in repeated_hashes
            ]

        return figures

    except Exception as e:
        logger.error(f"Native figure detection failed: {e}")
        print(f"  [ERROR] Native figure detection failed: {e}")
        return []


# -------------------------
# Detection Orchestration
# -------------------------


@dataclass
class DetectionResult:
    """Result of visual detection for a document."""

    candidates: List[VisualCandidate]
    tables_detected: int
    figures_detected: int
    escalated_tables: int
    detection_mode: str


def detect_all_visuals(
    pdf_path: str,
    config: DetectorConfig = DetectorConfig(),
) -> DetectionResult:
    """
    Detect all visuals (tables and figures) in a PDF.

    Uses tiered approach:
    1. FAST mode for all tables
    2. Escalate to ACCURATE for complex tables
    3. Native extraction for figures

    Args:
        pdf_path: Path to PDF file
        config: Detector configuration

    Returns:
        DetectionResult with all candidates
    """
    candidates: List[VisualCandidate] = []
    escalated_count = 0

    # Open PDF for page dimensions
    doc = fitz.open(pdf_path)

    try:
        # Pre-extract captions for all pages
        captions_by_page: Dict[int, List] = {}
        column_layouts: Dict[int, ColumnLayout] = {}

        for page_idx in range(doc.page_count):
            page_num = page_idx + 1
            column_layouts[page_num] = infer_column_layout(doc, page_num)
            captions_by_page[page_num] = extract_all_captions_on_page(doc, page_num)

        # Step 1: Detect tables with FAST mode
        tables = detect_tables_with_docling(pdf_path, mode="fast", config=config)

        for table in tables:
            page_num = table["page_num"]
            page = doc[page_num - 1]
            bbox_pts = table["bbox_pts"]

            # Compute complexity signals
            signals = TableComplexitySignals(
                merged_cell_count=table.get("merged_cell_count", 0),
                header_depth=table.get("header_depth", 1),
                token_coverage_ratio=table.get("token_coverage", 1.0),
                column_count=table.get("column_count", 0),
                row_count=table.get("row_count", 0),
            )

            # Check if escalation needed
            needs_accurate = False
            if config.enable_escalation:
                should_escalate, reason = should_escalate_to_accurate(
                    signals, config.escalation_config
                )
                if should_escalate:
                    needs_accurate = True
                    escalated_count += 1
                    logger.debug(f"Table on page {page_num} flagged for ACCURATE: {reason}")

            # Check for caption
            page_captions = captions_by_page.get(page_num, [])
            has_caption = _has_nearby_caption(bbox_pts, page_captions)

            # Create candidate
            area = (bbox_pts[2] - bbox_pts[0]) * (bbox_pts[3] - bbox_pts[1])
            page_area = page.rect.width * page.rect.height

            candidate = VisualCandidate(
                source="docling",
                docling_type="table",
                page_num=page_num,
                bbox_pts=bbox_pts,
                page_width_pts=page.rect.width,
                page_height_pts=page.rect.height,
                area_ratio=area / page_area if page_area > 0 else 0,
                has_nearby_caption=has_caption,
                has_grid_structure=True,
                needs_accurate_rerun=needs_accurate,
            )
            candidates.append(candidate)

        tables_count = len(tables)

        # Step 2: Detect figures
        figures = detect_figures_native(pdf_path, config)

        for fig in figures:
            page_num = fig["page_num"]
            page = doc[page_num - 1]
            bbox_pts = fig["bbox_pts"]

            # Check for caption
            page_captions = captions_by_page.get(page_num, [])
            has_caption = _has_nearby_caption(bbox_pts, page_captions)

            candidate = VisualCandidate(
                source=fig["source"],
                page_num=page_num,
                bbox_pts=bbox_pts,
                page_width_pts=page.rect.width,
                page_height_pts=page.rect.height,
                area_ratio=fig["area_ratio"],
                image_hash=fig.get("image_hash"),
                has_nearby_caption=has_caption,
                in_margin_zone=fig.get("in_margin_zone", False),
            )
            candidates.append(candidate)

        figures_count = len(figures)

    finally:
        doc.close()

    return DetectionResult(
        candidates=candidates,
        tables_detected=tables_count,
        figures_detected=figures_count,
        escalated_tables=escalated_count,
        detection_mode=config.default_table_mode,
    )


def _has_nearby_caption(
    bbox_pts: Tuple[float, float, float, float],
    captions: List,
    max_distance_pts: float = 72.0,
) -> bool:
    """
    Check if there's a caption nearby the visual.

    Args:
        bbox_pts: Visual bounding box
        captions: List of caption candidates on the page
        max_distance_pts: Maximum distance to consider "nearby"

    Returns:
        True if a caption is nearby
    """
    vx0, vy0, vx1, vy1 = bbox_pts

    for caption in captions:
        cx0, cy0, cx1, cy1 = caption.bbox_pts

        # Check if caption is below visual
        if cy0 >= vy1 and cy0 - vy1 <= max_distance_pts:
            # Check horizontal overlap
            if cx0 < vx1 and cx1 > vx0:
                return True

        # Check if caption is above visual
        if cy1 <= vy0 and vy0 - cy1 <= max_distance_pts:
            if cx0 < vx1 and cx1 > vx0:
                return True

    return False


# -------------------------
# Re-extraction with ACCURATE
# -------------------------


def reextract_table_accurate(
    pdf_path: str,
    page_num: int,
    bbox_pts: Tuple[float, float, float, float],
    config: DetectorConfig = DetectorConfig(),
) -> Optional[TableStructure]:
    """
    Re-extract a specific table using ACCURATE mode.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number of the table
        bbox_pts: Bounding box of the table
        config: Detector configuration

    Returns:
        TableStructure with ACCURATE extraction, or None
    """
    tables = detect_tables_with_docling(pdf_path, mode="accurate", config=config)

    # Find the matching table by bbox overlap
    for table in tables:
        if table["page_num"] != page_num:
            continue

        # Check bbox overlap
        t_bbox = table["bbox_pts"]
        overlap = _compute_bbox_overlap(bbox_pts, t_bbox)

        if overlap > 0.7:  # 70% overlap threshold
            return TableStructure(
                headers=table.get("headers", []),
                rows=table.get("rows", []),
                token_coverage=table.get("token_coverage", 1.0),
                structure_confidence=0.95,  # ACCURATE mode = high confidence
            )

    return None


def _compute_bbox_overlap(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
) -> float:
    """Compute overlap ratio between two bboxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


__all__ = [
    # Types
    "DetectorConfig",
    "DetectionResult",
    # Main functions
    "detect_all_visuals",
    "detect_tables_with_docling",
    "detect_figures_native",
    # Re-extraction
    "reextract_table_accurate",
]
