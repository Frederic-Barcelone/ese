# corpus_metadata/B_parsing/B24_native_figure_extraction.py
"""
Native PDF figure extraction integration for DocumentGraph population.

This module integrates the B09-B11 extraction pipeline into the DocumentGraph
builder. It extracts raster and vector figures from PDF XObjects, classifies
image types from caption/OCR text, and resolves extraction conflicts using
the deterministic resolver.

Key Components:
    - apply_native_figure_extraction: Main integration function for DocumentGraph
    - classify_image_type: Classify ImageType from caption and OCR text
    - merge_native_with_unstructured: Merge native figures with Unstructured extractions

Example:
    >>> from B_parsing.B24_native_figure_extraction import apply_native_figure_extraction
    >>> doc_graph, stats = apply_native_figure_extraction(
    ...     doc_graph, pdf_path, raw_images,
    ...     min_figure_area_ratio=0.03, filter_noise=True
    ... )
    >>> print(f"Extraction stats: {stats}")

Dependencies:
    - A_core.A01_domain_models: BoundingBox
    - B_parsing.B02_doc_graph: ImageBlock, ImageType, DocumentGraph
    - B_parsing.B09_pdf_native_figures: Figure extraction functions
    - B_parsing.B10_caption_detector: Caption detection
    - B_parsing.B11_extraction_resolver: Deterministic resolution
    - B_parsing.B23_text_helpers: PERCENTAGE_PATTERN
    - fitz (PyMuPDF): PDF rendering
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

import fitz  # PyMuPDF

from A_core.A01_domain_models import BoundingBox
from B_parsing.B02_doc_graph import ImageBlock, ImageType

from B_parsing.B23_text_helpers import PERCENTAGE_PATTERN

# B09-B11 modules
from B_parsing.B09_pdf_native_figures import (
    extract_embedded_figures_from_doc,
    detect_vector_figures,
    filter_noise_images,
    render_figure_by_xref,
    render_vector_figure,
    EmbeddedFigure,
    VectorFigure,
)
from B_parsing.B10_caption_detector import (
    detect_captions_all_pages,
)
from B_parsing.B11_extraction_resolver import (
    resolve_all,
    filter_duplicate_figures,
    get_resolution_stats,
)

if TYPE_CHECKING:
    from B_parsing.B02_doc_graph import DocumentGraph


def classify_image_type(
    caption: Optional[str],
    ocr_text: Optional[str],
) -> ImageType:
    """Classify image type from caption and OCR text."""
    check_text = (caption or "") + " " + (ocr_text or "")
    check_lower = check_text.lower()

    # Flowchart: CONSORT diagrams, patient flow, screening
    if any(kw in check_lower for kw in [
        "screened", "randomized", "enrolled", "consort", "flow",
        "excluded", "discontinued", "completed", "allocation"
    ]):
        return ImageType.FLOWCHART

    # Chart: various clinical trial result charts
    has_percentages = bool(PERCENTAGE_PATTERN.search(check_text))
    if has_percentages or any(kw in check_lower for kw in [
        "kaplan", "survival", "curve", "plot", "bar", "proportion",
        "percentage", "reduction", "change", "effect", "endpoint",
        "month", "week", "baseline", "placebo", "treatment",
        "pie", "distribution", "frequency", "symptoms", "serology",
        "fig.", "figure", "organ", "involvement"
    ]):
        return ImageType.CHART

    # Diagram: mechanism, pathway diagrams
    if any(kw in check_lower for kw in [
        "diagram", "schematic", "mechanism", "pathway", "cascade"
    ]):
        return ImageType.DIAGRAM

    return ImageType.UNKNOWN


def apply_native_figure_extraction(
    graph: "DocumentGraph",
    file_path: str,
    layout_model_images: Dict[int, List[Dict[str, Any]]],
    min_figure_area_ratio: float = 0.03,
    filter_noise: bool = True,
) -> tuple["DocumentGraph", Dict[str, Any]]:
    """
    Apply PDF-native figure extraction using B09-B11 modules.

    This extracts raster images and vector plots directly from PDF,
    links them to captions, and produces more accurate bounding boxes.

    Args:
        graph: DocumentGraph with existing images (from Unstructured)
        file_path: Path to PDF file
        layout_model_images: Images from Unstructured (as fallback signal)
        min_figure_area_ratio: Minimum image area as fraction of page
        filter_noise: Whether to filter noise images (logos, headers)

    Returns:
        Tuple of (updated DocumentGraph, resolution stats dict)
    """
    doc = fitz.open(file_path)
    resolution_stats: Dict[str, Any] = {}

    try:
        # Step 1: Extract raster figures from PDF XObjects
        raster_figures = extract_embedded_figures_from_doc(
            doc, min_area_ratio=min_figure_area_ratio
        )

        # Step 2: Filter noise (logos, repeated headers)
        if filter_noise:
            raster_figures = filter_noise_images(raster_figures, doc)

        # Step 3: Detect vector figures (Kaplan-Meier plots, etc.)
        vector_figures: List[VectorFigure] = []
        for page_num in range(1, doc.page_count + 1):
            page_vectors = detect_vector_figures(doc, page_num)
            vector_figures.extend(page_vectors)

        # Step 4: Detect all captions and column layout
        all_captions, columns_by_page = detect_captions_all_pages(doc)

        # Step 5: Convert Unstructured images to layout model signal
        layout_model_figures: List[ImageBlock] = []
        for page_num, images in layout_model_images.items():
            for img_data in images:
                # Only include if has actual image data
                if img_data.get("image_base64"):
                    layout_model_figures.append(
                        ImageBlock(
                            page_num=page_num,
                            reading_order_index=0,
                            image_base64=img_data.get("image_base64"),
                            ocr_text=img_data.get("ocr_text"),
                            bbox=img_data["bbox"],
                        )
                    )

        # Step 6: Resolve all signals
        resolved_figures, resolved_tables = resolve_all(
            raster_figures=raster_figures,
            vector_figures=vector_figures,
            layout_model_figures=layout_model_figures,
            captions=all_captions,
            columns_by_page=columns_by_page,
            doc=doc,
        )

        # Step 7: Deduplicate overlapping figures
        resolved_figures = filter_duplicate_figures(resolved_figures)

        # Step 8: Store resolution stats
        resolution_stats = get_resolution_stats(
            resolved_figures, resolved_tables
        )

        # Step 9: Convert resolved figures to ImageBlocks
        for page_num in sorted(graph.pages.keys()):
            page_obj = graph.pages[page_num]
            page_figures = [
                rf for rf in resolved_figures if rf.page_num == page_num
            ]

            new_images: List[ImageBlock] = []
            for idx, rf in enumerate(page_figures):
                # Render image from source
                image_base64 = None
                ocr_text = None

                if rf.figure_type == "raster" and isinstance(rf.figure, EmbeddedFigure):
                    # Render from xref
                    try:
                        img_bytes = render_figure_by_xref(doc, rf.figure.xref)
                        image_base64 = base64.b64encode(img_bytes).decode("utf-8")
                    except Exception as e:
                        logger.debug("Failed to render raster figure xref=%s: %s", rf.figure.xref, e)
                elif rf.figure_type == "vector" and isinstance(rf.figure, VectorFigure):
                    # Render vector region
                    try:
                        img_bytes = render_vector_figure(
                            doc, rf.page_num, rf.bbox, dpi=200
                        )
                        image_base64 = base64.b64encode(img_bytes).decode("utf-8")
                    except Exception as e:
                        logger.debug("Failed to render vector figure on page %d: %s", rf.page_num, e)
                elif rf.figure_type == "layout_model":
                    # Use existing base64 from layout model
                    if hasattr(rf.figure, "image_base64"):
                        image_base64 = rf.figure.image_base64
                    if hasattr(rf.figure, "ocr_text"):
                        ocr_text = rf.figure.ocr_text

                # Determine image type
                img_type = classify_image_type(
                    rf.caption_text, ocr_text
                )

                # Create bounding box
                x0, y0, x1, y1 = rf.bbox
                page_w, page_h = page_obj.width, page_obj.height
                bbox = BoundingBox(
                    coords=(x0, y0, x1, y1),
                    page_width=page_w,
                    page_height=page_h,
                )

                # Store metadata about source
                metadata = {
                    "source": rf.source,
                    "figure_type": rf.figure_type,
                }
                if rf.caption:
                    metadata["caption_number"] = str(rf.caption.number)

                new_images.append(
                    ImageBlock(
                        page_num=page_num,
                        reading_order_index=len(page_obj.blocks) + idx,
                        image_base64=image_base64,
                        ocr_text=ocr_text,
                        caption=rf.caption_text,
                        image_type=img_type,
                        bbox=bbox,
                        metadata=metadata,
                    )
                )

            # Replace page images with resolved ones
            page_obj.images = new_images

    finally:
        doc.close()

    return graph, resolution_stats


__all__ = [
    "classify_image_type",
    "apply_native_figure_extraction",
]
