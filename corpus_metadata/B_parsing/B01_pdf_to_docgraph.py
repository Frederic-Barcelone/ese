# corpus_metadata/B_parsing/B01_pdf_to_docgraph.py
"""
PDF to DocumentGraph parser with SOTA multi-column layout detection.

This module converts PDF documents into structured DocumentGraph representations
using Unstructured.io or PyMuPDF for text extraction. It implements state-of-the-art
column ordering (XY-Cut++), header/footer detection via repetition inference, and
integrates native figure extraction for accurate visual element handling.

Key Components:
    - PDFToDocGraphParser: Main parser class with configurable extraction strategies
    - document_to_markdown: Convert DocumentGraph to Markdown with table rendering
    - LayoutConfig: Configuration for SOTA column ordering (B04 integration)

Example:
    >>> from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser
    >>> parser = PDFToDocGraphParser(config={
    ...     "unstructured_strategy": "hi_res",
    ...     "use_sota_layout": True,
    ...     "document_type": "academic",
    ... })
    >>> doc_graph = parser.parse("paper.pdf")
    >>> for block in doc_graph.iter_linear_blocks():
    ...     print(block.text)

Dependencies:
    - A_core.A01_domain_models: BoundingBox
    - A_core.A02_interfaces: BaseParser
    - B_parsing.B02_doc_graph: DocumentGraph, Page, TextBlock, ContentRole, ImageBlock, ImageType
    - B_parsing.B04_column_ordering: SOTA column ordering and layout detection
    - B_parsing.B23_text_helpers: Text normalization and cleaning utilities
    - B_parsing.B24_native_figure_extraction: Native PDF figure extraction
    - B_parsing.B25_legacy_ordering: Fallback block ordering
    - B_parsing.B26_repetition_inference: Header/footer detection
"""

from __future__ import annotations

import logging
import os
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Suppress unstructured's pymupdf_layout suggestion
warnings.filterwarnings("ignore", message=".*pymupdf_layout.*")

import fitz  # PyMuPDF
from unstructured.partition.pdf import partition_pdf

from A_core.A02_interfaces import BaseParser
from A_core.A01_domain_models import BoundingBox
from B_parsing.B02_doc_graph import DocumentGraph, Page, TextBlock, ContentRole, ImageBlock, ImageType

# Text helpers (B23)
from B_parsing.B23_text_helpers import (
    normalize_repeated_text,
    NUMBERED_REFERENCE_RE,
    PERCENTAGE_PATTERN,
    is_garbled_flowchart,
    extract_figure_reference,
    SECTION_NUM_RE,
    SECTION_PLAIN_RE,
    META_PREFIX_RE,
    AFFIL_TOKENS_RE,
    EMAIL_RE,
    PIPEY_RE,
    clean_text,
    bbox_overlaps,
    table_to_markdown,
)

# Native figure extraction (B24)
from B_parsing.B24_native_figure_extraction import (
    apply_native_figure_extraction,
)

# Legacy ordering (B25)
from B_parsing.B25_legacy_ordering import (
    order_blocks_deterministically,
)

# Repetition inference (B26)
from B_parsing.B26_repetition_inference import (
    infer_repeated_headers_footers,
    looks_like_known_footer,
    is_short_repeated_noise,
    is_running_header,
)

# SOTA Column Ordering (B04)
from B_parsing.B04_column_ordering import (
    order_page_blocks,
    LayoutConfig,
    create_config as create_layout_config,
    get_layout_info,
)


def document_to_markdown(
    doc: DocumentGraph,
    include_tables: bool = True,
    skip_header_footer: bool = True,
) -> str:
    """
    Convert DocumentGraph to Markdown.
    - Renders tables as proper markdown tables (not flattened text)
    - Skips text blocks that overlap with table regions
    """
    lines: List[str] = []

    for pnum in sorted(doc.pages.keys()):
        page = doc.pages[pnum]
        tables = getattr(page, "tables", []) or []

        # Collect all content items (blocks + tables) for proper ordering
        content_items = []

        for b in page.blocks:
            if skip_header_footer and b.role in (
                ContentRole.PAGE_HEADER,
                ContentRole.PAGE_FOOTER,
            ):
                continue

            # Skip blocks that overlap with any table region
            if tables:
                overlaps_table = any(
                    bbox_overlaps(b.bbox, t.bbox, threshold=0.3) for t in tables
                )
                if overlaps_table:
                    continue

            y0 = b.bbox.coords[1] if b.bbox else 0
            content_items.append((b.reading_order_index, y0, "block", b))

        if include_tables:
            for t in tables:
                y0 = t.bbox.coords[1] if t.bbox else 0
                content_items.append((t.reading_order_index, y0, "table", t))

        # Sort by reading order, then y position
        content_items.sort(key=lambda x: (x[0], x[1]))

        for _, _, item_type, item in content_items:
            if item_type == "block":
                text = item.text

                # Detect and replace garbled flowcharts/diagrams
                if is_garbled_flowchart(text):
                    fig_ref = extract_figure_reference(text)
                    if fig_ref:
                        lines.append(
                            f"[{fig_ref}: Flowchart/Diagram - see PDF for visual]"
                        )
                    else:
                        lines.append("[Flowchart/Diagram - see PDF for visual]")
                    continue

                if item.role == ContentRole.SECTION_HEADER:
                    lines.append(f"## {text}")
                else:
                    lines.append(text)
            elif item_type == "table":
                # Render as markdown table
                lines.append("")
                lines.append(table_to_markdown(item))

        lines.append("")

    return "\n".join(lines).strip() + "\n"


class PDFToDocGraphParser(BaseParser):
    """
    PDF -> DocumentGraph usando Unstructured.

    Objetivos:
    - Orden de lectura determinista (SOTA multi-columna via B04)
    - Detectar headers/footers repetidos (robusto)
    - Detectar SECTION_HEADER evitando falsos positivos (autores/afiliaciones)
    - Reparar guiones de abreviaturas: 'MG- ADL' -> 'MG-ADL'
    - Limpieza de pipes iniciales: '| Andreas ...' -> 'Andreas ...'

    Config options:
    - unstructured_strategy: "hi_res" (default), "fast", "ocr_only", "auto"
    - hi_res_model_name: "yolox" (default), "detectron2_onnx"
    - infer_table_structure: True (default) - extract table structure in hi_res
    - extract_images_in_pdf: False (default) - extract images from PDF
    - languages: ["eng"] (default) - OCR language packs for better accuracy
    - include_page_breaks: True (default) - track page boundaries
    - force_fast: False (default) - force fast strategy regardless of config
    - force_hf_offline: False (default) - block HuggingFace model downloads

    NEW Config options (SOTA Layout - B04):
    - use_sota_layout: True (default) - use B04 SOTA column ordering
    - document_type: "academic" (default) - preset: "academic", "clinical", "regulatory", "newsletter"
    - layout_config: None (default) - custom LayoutConfig instance
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

        # Offline HF - disabled by default to allow model downloads for hi_res
        self.force_hf_offline = bool(self.config.get("force_hf_offline", False))
        if self.force_hf_offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # Unstructured knobs - default to hi_res for better extraction
        self.unstructured_strategy = (
            str(self.config.get("unstructured_strategy", "hi_res")).strip() or "hi_res"
        )
        self.force_fast = bool(self.config.get("force_fast", False))
        if self.force_fast:
            self.unstructured_strategy = "fast"

        # hi_res specific options
        self.hi_res_model_name = self.config.get("hi_res_model_name", "yolox")
        self.infer_table_structure = bool(
            self.config.get("infer_table_structure", True)
        )
        # Image extraction - enabled by default for hi_res
        self.extract_images_in_pdf = bool(
            self.config.get("extract_images_in_pdf", True)
        )
        # Extract base64 images into element payload
        self.extract_image_block_to_payload = bool(
            self.config.get("extract_image_block_to_payload", True)
        )
        # Which element types to extract as images
        self.extract_image_block_types = self.config.get(
            "extract_image_block_types", ["Image", "Table", "Figure"]
        )
        # Image output directory (set per-parse call)
        self._image_output_dir: Optional[str] = None

        # OCR and structure options
        self.languages = self.config.get("languages", ["eng"])  # improves OCR accuracy
        self.include_page_breaks = bool(
            self.config.get("include_page_breaks", True)
        )  # track page boundaries

        # categorías a descartar (case-insensitive)
        self.drop_categories = {
            c.strip().lower()
            for c in (self.config.get("drop_categories") or [])
            if isinstance(c, str)
        }

        # Orden / líneas
        self.y_tolerance = float(self.config.get("y_tolerance", 3.0))

        # Header/Footer por posición
        self.header_top_pct = float(self.config.get("header_top_pct", 0.07))  # top 7%
        self.footer_bottom_pct = float(
            self.config.get("footer_bottom_pct", 0.90)
        )  # bottom 10%

        # Header/Footer por repetición
        self.min_repeat_count = int(self.config.get("min_repeat_count", 3))
        self.min_repeat_pages = int(self.config.get("min_repeat_pages", 3))
        self.repeat_zone_majority = float(self.config.get("repeat_zone_majority", 0.60))

        # Legacy 2 columnas (used as fallback)
        self.two_col_min_side_blocks = int(
            self.config.get("two_col_min_side_blocks", 6)
        )

        # =============================================
        # Extraction Method Selection - NEW
        # =============================================
        # "unstructured" (default): Use Unstructured.io with ML models
        # "pymupdf": Use PyMuPDF direct text extraction (faster, better for native PDFs)
        # "auto": Use pymupdf for native PDFs, unstructured for scanned
        self.extraction_method = self.config.get("extraction_method", "unstructured")

        # =============================================
        # SOTA Column Ordering (B04) - NEW
        # =============================================
        self.use_sota_layout = bool(self.config.get("use_sota_layout", True))

        # Layout configuration - can be:
        # - None (use document_type preset)
        # - A LayoutConfig instance
        # - A string: "academic", "clinical", "regulatory", "newsletter"
        layout_cfg = self.config.get("layout_config")
        if layout_cfg is None:
            document_type = self.config.get("document_type", "academic")
            self.layout_config = create_layout_config(document_type)
        elif isinstance(layout_cfg, str):
            self.layout_config = create_layout_config(layout_cfg)
        elif isinstance(layout_cfg, LayoutConfig):
            self.layout_config = layout_cfg
        else:
            self.layout_config = LayoutConfig()

        # Store layout info per page (for debugging/export)
        self._page_layouts: Dict[int, Dict[str, Any]] = {}

        # =============================================
        # PDF Native Figure Extraction (B09-B11) - NEW
        # =============================================
        # Use native PDF extraction for figures (raster + vector)
        # This produces more accurate bounding boxes and catches vector plots
        self.use_native_figure_extraction = bool(
            self.config.get("use_native_figure_extraction", True)
        )
        # Minimum image area as fraction of page (default 3%)
        self.min_figure_area_ratio = float(
            self.config.get("min_figure_area_ratio", 0.03)
        )
        # Filter noise images (logos, headers)
        self.filter_noise_figures = bool(
            self.config.get("filter_noise_figures", True)
        )

        # Store resolution stats (for debugging/export)
        self._resolution_stats: Dict[str, Any] = {}

    # -----------------------
    # Public
    # -----------------------

    def parse(self, file_path: str, image_output_dir: Optional[str] = None) -> DocumentGraph:
        page_dims = self._get_page_dimensions(file_path)

        # Store image output directory for partition_pdf
        self._image_output_dir = image_output_dir

        # Select extraction method
        use_pymupdf = False
        if self.extraction_method == "pymupdf":
            use_pymupdf = True
        elif self.extraction_method == "auto":
            # Auto-detect: use PyMuPDF for native PDFs, Unstructured for scanned
            use_pymupdf = not self._is_scanned_pdf(file_path)

        # Extract elements using selected method
        if use_pymupdf:
            elements = self._extract_with_pymupdf(file_path)
        else:
            elements = self._call_partition_pdf(file_path)

        # Pass 1: raw blocks + repetition stats + images
        raw_pages: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        raw_images: Dict[int, List[Dict[str, Any]]] = defaultdict(list)  # Images per page
        raw_captions: Dict[int, List[Dict[str, Any]]] = defaultdict(list)  # Figure captions

        norm_count: Counter[str] = Counter()
        norm_pages: Dict[str, set] = defaultdict(set)
        norm_zone_votes: Dict[str, Counter[str]] = defaultdict(Counter)
        norm_sample_text: Dict[str, str] = {}

        for el in elements:
            # Handle both Unstructured elements and PyMuPDF dicts
            if use_pymupdf:
                # PyMuPDF returns dicts
                cat_norm = (el.get("category") or "").strip().lower()
                page_num_raw = el.get("page_num")
                page_num: Optional[int] = int(page_num_raw) if page_num_raw is not None else None
                text_raw = el.get("text", "")
                bbox_tuple = el.get("bbox", (0, 0, 0, 0))
                page_w, page_h = page_dims.get(page_num, (0.0, 0.0)) if page_num is not None else (0.0, 0.0)
                bbox = BoundingBox(coords=bbox_tuple, page_width=page_w, page_height=page_h)
            else:
                # Unstructured returns element objects
                cat = getattr(el, "category", None)
                cat_norm = (cat or "").strip().lower() if isinstance(cat, str) else ""
                cls_norm = el.__class__.__name__.strip().lower()

                if self.drop_categories and (
                    cat_norm in self.drop_categories or cls_norm in self.drop_categories
                ):
                    continue

                page_num = self._get_element_page_num(el)
                if page_num is None:
                    continue

                page_w, page_h = page_dims.get(page_num, (0.0, 0.0))
                text_raw = getattr(el, "text", "") or ""
                bbox = self._bbox_from_element(el, page_w=page_w, page_h=page_h)

            if page_num is None:
                continue

            # Handle Image and FigureCaption elements (Unstructured only)
            if not use_pymupdf:
                cls_name = el.__class__.__name__
                if cls_name == "Image":
                    # Extract image with base64 data
                    el_metadata = getattr(el, "metadata", None)
                    image_base64 = getattr(el_metadata, "image_base64", None) if el_metadata else None
                    raw_images[page_num].append({
                        "bbox": bbox,
                        "image_base64": image_base64,
                        "ocr_text": text_raw.strip() if text_raw else None,
                        "y0": float(bbox.coords[1]),
                    })
                    continue  # Don't add to text blocks
                elif cls_name == "FigureCaption":
                    raw_captions[page_num].append({
                        "text": text_raw.strip(),
                        "bbox": bbox,
                        "y0": float(bbox.coords[1]),
                    })
                    # Also add to text blocks for context

            text = self._clean_text(text_raw)
            if not text:
                continue

            zone = self._zone_from_bbox(bbox, page_h=page_h)

            # For PyMuPDF, pass category directly; for Unstructured, pass element
            is_section_header = self._is_section_header(
                el if not use_pymupdf else None,
                zone,
                text,
                category=cat_norm if use_pymupdf else None
            )

            # Include category for B04 spanning detection hints
            rb = {
                "text": text,
                "bbox": bbox,
                "x0": float(bbox.coords[0]),
                "y0": float(bbox.coords[1]),
                "zone": zone,
                "is_section_header": is_section_header,
                "category": cat_norm,  # For B04 hints
            }
            raw_pages[page_num].append(rb)

            norm = normalize_repeated_text(text)
            if norm:
                norm_count[norm] += 1
                norm_pages[norm].add(page_num)
                if zone in ("HEADER", "FOOTER"):
                    norm_zone_votes[norm][zone] += 1
                norm_sample_text.setdefault(norm, text)

        repeated_headers, repeated_footers = infer_repeated_headers_footers(
            norm_count=norm_count,
            norm_pages=norm_pages,
            norm_zone_votes=norm_zone_votes,
            norm_sample_text=norm_sample_text,
            min_repeat_count=self.min_repeat_count,
            min_repeat_pages=self.min_repeat_pages,
            repeat_zone_majority=self.repeat_zone_majority,
        )

        # Pass 2: build DocumentGraph
        graph = DocumentGraph(doc_id=str(file_path))
        self._page_layouts = {}  # Reset layout info

        for page_num in sorted(raw_pages.keys()):
            page_w, page_h = page_dims.get(page_num, (0.0, 0.0))
            page_obj = Page(number=page_num, width=page_w, height=page_h)

            # =============================================
            # SOTA Column Ordering (B04) - INTEGRATION
            # =============================================
            if self.use_sota_layout:
                # Get layout info for debugging/export
                self._page_layouts[page_num] = get_layout_info(
                    raw_pages[page_num], page_w, page_h, self.layout_config
                )

                # Apply SOTA ordering
                ordered = order_page_blocks(
                    raw_pages[page_num], page_w, page_h, self.layout_config, page_num
                )
            else:
                # Fallback to legacy ordering
                ordered = order_blocks_deterministically(
                    raw_pages[page_num],
                    page_w=page_w,
                    two_col_min_side_blocks=self.two_col_min_side_blocks,
                    y_tolerance=self.y_tolerance,
                )

            blocks: List[TextBlock] = []
            for idx, rb in enumerate(ordered):
                txt = rb["text"]
                if not txt:
                    continue

                role = ContentRole.BODY_TEXT
                norm = normalize_repeated_text(txt)

                # Check if this looks like a numbered reference (preserve as body text)
                is_reference = bool(NUMBERED_REFERENCE_RE.match(txt))

                # Hard override: known footer noise (but not references)
                if looks_like_known_footer(txt) and not is_reference:
                    role = ContentRole.PAGE_FOOTER
                # Running header pattern: "Author et al" or "Author et al."
                elif is_running_header(txt, rb.get("zone")):
                    role = ContentRole.PAGE_HEADER
                elif rb.get("zone") == "HEADER" and norm in repeated_headers:
                    role = ContentRole.PAGE_HEADER
                elif (
                    rb.get("zone") == "FOOTER"
                    and norm in repeated_footers
                    and not is_reference
                ):
                    role = ContentRole.PAGE_FOOTER
                # fallback repetition even if zone couldn't be trusted
                elif (
                    norm in repeated_footers
                    and is_short_repeated_noise(txt)
                    and not is_reference
                ):
                    role = ContentRole.PAGE_FOOTER
                elif rb.get("is_section_header"):
                    role = ContentRole.SECTION_HEADER

                blocks.append(
                    TextBlock(
                        text=txt,
                        page_num=page_num,
                        reading_order_index=idx,
                        role=role,
                        bbox=rb["bbox"],
                    )
                )

            page_obj.blocks = blocks

            # Add images to page (match captions by proximity)
            page_images: List[ImageBlock] = []
            page_caps = raw_captions.get(page_num, [])

            for img_idx, img_data in enumerate(raw_images.get(page_num, [])):
                # Find closest caption below the image (captions usually below figures)
                img_y = img_data["y0"]
                caption = None
                best_dist = float("inf")

                for cap in page_caps:
                    cap_y = cap["y0"]
                    # Caption should be below the image (larger Y in PDF coords)
                    # Allow large tolerance since captions can be far below in multi-column
                    if cap_y >= img_y:
                        dist = cap_y - img_y
                        if dist < best_dist and dist < 1500:  # Within 1500 units (generous)
                            best_dist = dist
                            caption = cap["text"]

                # If no caption found on same page, check if OCR text mentions "Figure X"
                if not caption and img_data.get("ocr_text"):
                    # Try to extract figure number from OCR text
                    import re
                    fig_match = re.search(r"Figure\s*(\d+)", img_data["ocr_text"], re.IGNORECASE)
                    if fig_match:
                        caption = f"Figure {fig_match.group(1)}"

                # Determine image type from caption or OCR text
                img_type = ImageType.UNKNOWN
                check_text = (caption or "") + " " + (img_data.get("ocr_text") or "")
                check_lower = check_text.lower()

                # Flowchart: CONSORT diagrams, patient flow, screening
                if any(kw in check_lower for kw in [
                    "screened", "randomized", "enrolled", "consort", "flow",
                    "excluded", "discontinued", "completed", "allocation"
                ]):
                    img_type = ImageType.FLOWCHART
                # Chart: various clinical trial result charts (including pie charts)
                # Also detect by percentage patterns in OCR text
                has_percentages = bool(PERCENTAGE_PATTERN.search(check_text))
                if has_percentages or any(kw in check_lower for kw in [
                    "kaplan", "survival", "curve", "plot", "bar", "proportion",
                    "percentage", "reduction", "change", "effect", "endpoint",
                    "month", "week", "baseline", "placebo", "treatment",
                    "pie", "distribution", "frequency", "symptoms", "serology",
                    "fig.", "figure", "organ", "involvement"
                ]):
                    img_type = ImageType.CHART
                # Diagram: mechanism, pathway diagrams
                elif any(kw in check_lower for kw in [
                    "diagram", "schematic", "mechanism", "pathway", "cascade"
                ]):
                    img_type = ImageType.DIAGRAM
                # Logo: small images on first page
                elif page_num == 1 and len(img_data.get("image_base64") or "") < 10000:
                    img_type = ImageType.LOGO

                page_images.append(ImageBlock(
                    page_num=page_num,
                    reading_order_index=len(blocks) + img_idx,  # After text blocks
                    image_base64=img_data.get("image_base64"),
                    ocr_text=img_data.get("ocr_text"),
                    caption=caption,
                    image_type=img_type,
                    bbox=img_data["bbox"],
                ))

            page_obj.images = page_images
            graph.pages[page_num] = page_obj

        # =============================================
        # PDF Native Figure Extraction (B09-B11)
        # =============================================
        if self.use_native_figure_extraction:
            graph, self._resolution_stats = apply_native_figure_extraction(
                graph,
                file_path,
                raw_images,
                min_figure_area_ratio=self.min_figure_area_ratio,
                filter_noise=self.filter_noise_figures,
            )

        return graph

    def get_page_layout_info(self, page_num: int) -> Optional[Dict[str, Any]]:
        """Get detected layout info for a specific page (after parsing)."""
        return self._page_layouts.get(page_num)

    def get_all_layout_info(self) -> Dict[int, Dict[str, Any]]:
        """Get detected layout info for all pages (after parsing)."""
        return self._page_layouts.copy()

    def get_resolution_stats(self) -> Dict[str, Any]:
        """Get figure/table resolution stats (after parsing)."""
        return self._resolution_stats.copy()

    # -----------------------
    # Unstructured call
    # -----------------------

    def _call_partition_pdf(self, file_path: str):
        """
        Partition PDF using unstructured. Supports hi_res with inference.
        """
        kwargs = {
            "filename": file_path,
            "strategy": self.unstructured_strategy,
            "languages": self.languages,
            "include_page_breaks": self.include_page_breaks,
        }

        # Add hi_res specific options when using hi_res strategy
        if self.unstructured_strategy == "hi_res":
            kwargs["hi_res_model_name"] = self.hi_res_model_name
            kwargs["infer_table_structure"] = self.infer_table_structure
            kwargs["extract_images_in_pdf"] = self.extract_images_in_pdf
            # Image extraction to payload (base64)
            if self.extract_image_block_to_payload:
                kwargs["extract_image_block_types"] = self.extract_image_block_types
                kwargs["extract_image_block_to_payload"] = True
            # Save extracted images to output directory
            if self._image_output_dir:
                kwargs["extract_image_block_output_dir"] = self._image_output_dir

        return partition_pdf(**kwargs)

    def _extract_with_pymupdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text blocks directly using PyMuPDF (fitz).

        Advantages over Unstructured:
        - Faster (no ML model overhead)
        - More reliable for native PDFs with embedded text
        - Better character preservation
        - Accurate bounding boxes in PDF coordinates

        Returns list of dicts with keys: text, bbox, page_num, category
        """
        doc = fitz.open(file_path)
        blocks = []

        try:
            for page_idx in range(doc.page_count):
                page = doc[page_idx]
                page_num = page_idx + 1

                # Extract text blocks with positions
                # flags: preserve whitespace and ligatures
                text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

                for block in text_dict.get("blocks", []):
                    # Only process text blocks (type 0), skip images (type 1)
                    if block.get("type") != 0:
                        continue

                    # Extract text from lines and spans
                    block_text = self._extract_block_text(block)
                    if not block_text or not block_text.strip():
                        continue

                    # Get bounding box (already in PDF coordinates)
                    bbox = block.get("bbox", (0, 0, 0, 0))

                    blocks.append({
                        "text": block_text,
                        "bbox": bbox,
                        "page_num": page_num,
                        "category": "NarrativeText",  # Default category
                    })
        finally:
            doc.close()

        return blocks

    def _extract_block_text(self, block: Dict[str, Any]) -> str:
        """Extract text from a PyMuPDF text block."""
        lines = []
        for line in block.get("lines", []):
            spans_text = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text:
                    spans_text.append(text)
            if spans_text:
                lines.append("".join(spans_text))
        return "\n".join(lines)

    def _is_scanned_pdf(self, file_path: str) -> bool:
        """
        Detect if PDF is scanned (image-based) vs native (text-based).

        Returns True if PDF appears to be scanned/OCR'd.
        """
        doc = fitz.open(file_path)
        try:
            # Check first few pages for text content
            text_chars = 0
            pages_checked = min(3, doc.page_count)

            for i in range(pages_checked):
                page = doc[i]
                text = page.get_text("text")
                text_chars += len(text.strip())

            # If very little text found, likely scanned
            avg_chars = text_chars / pages_checked if pages_checked > 0 else 0
            return avg_chars < 100  # Less than 100 chars per page = likely scanned
        finally:
            doc.close()

    # -----------------------
    # Element -> bbox, page, roles
    # -----------------------

    def _get_element_page_num(self, el) -> Optional[int]:
        md = getattr(el, "metadata", None)
        if md is None:
            return None
        pn = getattr(md, "page_number", None)
        if pn is None:
            pn = getattr(md, "page", None)
        try:
            return int(pn) if pn is not None else None
        except Exception as e:
            logger.debug("Failed to convert page number %r to int: %s", pn, e)
            return None

    def _bbox_from_element(self, el, page_w: float, page_h: float) -> BoundingBox:
        """
        Extract bounding box from Unstructured element.

        IMPORTANT: Unstructured hi_res uses PixelSpace coordinates (image pixels),
        not PDF coordinates. We must normalize to PDF space for B04 column detection.
        """
        md = getattr(el, "metadata", None)
        coords_meta = getattr(md, "coordinates", None) if md else None

        points = None
        if coords_meta is not None and hasattr(coords_meta, "points"):
            try:
                points = list(coords_meta.points)
            except Exception as e:
                logger.debug("Failed to extract coordinate points: %s", e)
                points = None

        if points is None and isinstance(coords_meta, dict) and "points" in coords_meta:
            points = coords_meta.get("points")

        if points:
            xs = [float(p[0]) for p in points if p and len(p) >= 2]
            ys = [float(p[1]) for p in points if p and len(p) >= 2]
            if xs and ys:
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)

                # Normalize from PixelSpace to PDF space
                # Unstructured hi_res renders at ~200-300 DPI, PDF is 72 DPI
                scale_x, scale_y = 1.0, 1.0
                if coords_meta is not None and hasattr(coords_meta, "system"):
                    system = getattr(coords_meta, "system", None)
                    if (
                        system is not None
                        and hasattr(system, "width")
                        and hasattr(system, "height")
                    ):
                        pixel_w = getattr(system, "width", 0)
                        pixel_h = getattr(system, "height", 0)
                        if pixel_w > 0 and pixel_h > 0:
                            scale_x = page_w / pixel_w
                            scale_y = page_h / pixel_h

                # Apply normalization
                x0 = max(0.0, x0 * scale_x)
                y0 = max(0.0, y0 * scale_y)
                x1 = max(0.0, x1 * scale_x)
                y1 = max(0.0, y1 * scale_y)

                return BoundingBox(
                    coords=(x0, y0, x1, y1),
                    page_width=page_w or None,
                    page_height=page_h or None,
                    is_normalized=False,
                )

        return BoundingBox(
            coords=(0.0, 0.0, 0.0, 0.0),
            page_width=page_w or None,
            page_height=page_h or None,
            is_normalized=False,
        )

    def _zone_from_bbox(self, bbox: BoundingBox, page_h: float) -> str:
        if page_h <= 0:
            return "BODY"
        x0, y0, x1, y1 = bbox.coords
        if (x0, y0, x1, y1) == (0.0, 0.0, 0.0, 0.0):
            return "BODY"
        header_y = page_h * self.header_top_pct
        footer_y = page_h * self.footer_bottom_pct
        if y1 <= header_y:
            return "HEADER"
        if y0 >= footer_y:
            return "FOOTER"
        return "BODY"

    # -----------------------
    # SECTION_HEADER detection (stricter)
    # -----------------------

    def _looks_like_author_or_affiliation(self, t: str) -> bool:
        if not t:
            return False

        # easy metadata exclusions
        if META_PREFIX_RE.search(t):
            return True
        if EMAIL_RE.search(t):
            return True

        # author blocks often have pipes
        if PIPEY_RE.search(t):
            return True

        # affiliations: university/hospital etc
        if AFFIL_TOKENS_RE.search(t):
            return True

        # lots of commas + digits (affiliations / footnotes)
        digits = sum(ch.isdigit() for ch in t)
        commas = t.count(",")
        if digits >= 2 and commas >= 1:
            return True

        return False

    def _is_section_header(
        self, el, zone: str, cleaned_text: str, category: Optional[str] = None
    ) -> bool:
        """
        Goal: mark true paper section headings, not author/affiliation/title fragments.

        Args:
            el: Unstructured element (or None if using PyMuPDF)
            zone: Position zone (HEADER, BODY, FOOTER)
            cleaned_text: Cleaned text content
            category: Optional category string (used when el is None)
        """
        if zone != "BODY":
            return False

        t = (cleaned_text or "").strip()
        if not t:
            return False

        tl = t.lower()

        # anti-ruido típico
        if tl.startswith(("figure", "fig.", "table", "supplementary", "appendix")):
            return False
        if "downloaded from" in tl or "wiley online library" in tl:
            return False
        if tl.startswith("doi") or "https://doi.org" in tl:
            return False

        # kill obvious metadata / author/affiliation lines
        if self._looks_like_author_or_affiliation(t):
            return False

        # section numbering style: "2 | Methods"
        if SECTION_NUM_RE.match(t):
            return True

        # plain section headings (rare but happens): "Methods"
        if SECTION_PLAIN_RE.match(t.strip()):
            return True

        # avoid sentence-like things
        if ". " in t or t.endswith("."):
            return False

        # avoid too long
        if len(t.split()) > 18:
            return False

        # category hints: only accept Title/Header if it *still* looks like a header
        # Support both Unstructured elements and direct category string
        cat_norm = category
        if cat_norm is None and el is not None:
            cat = getattr(el, "category", None)
            if isinstance(cat, str):
                cat_norm = cat.strip().lower()

        if cat_norm in {"title", "header"}:
            # must be short and "header-like"
            if t[0].isupper() or t[0].isdigit():
                return True
            return False

        # Check class name for Unstructured elements
        if el is not None:
            cls = el.__class__.__name__.lower()
            if cls in {"title", "header"}:
                if t[0].isupper() or t[0].isdigit():
                    return True
                return False

        return False

    # -----------------------
    # Utils
    # -----------------------

    def _clean_text(self, text: str) -> str:
        """Delegate to B23_text_helpers.clean_text."""
        return clean_text(text)

    def _get_page_dimensions(self, file_path: str) -> Dict[int, Tuple[float, float]]:
        dims: Dict[int, Tuple[float, float]] = {}
        doc = fitz.open(file_path)
        try:
            for i in range(doc.page_count):
                page = doc[i]
                dims[i + 1] = (float(page.rect.width), float(page.rect.height))
        finally:
            doc.close()
        return dims
