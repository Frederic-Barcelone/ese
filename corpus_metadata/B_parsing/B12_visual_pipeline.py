# corpus_metadata/B_parsing/B12_visual_pipeline.py
"""
Visual extraction pipeline orchestrator for tables and figures.

This module provides the main entry point for extracting visual elements from PDFs,
orchestrating a 4-stage pipeline: (1) Detection using DocLayout-YOLO or heuristics,
(2) Rendering with PyMuPDF point-based padding, (3) Triage and VLM enrichment,
(4) Document-level resolution with deduplication and multi-page merging.

Key Components:
    - VisualExtractionPipeline: Main pipeline class orchestrating all stages
    - PipelineConfig: Configuration for detection, rendering, triage, and resolution
    - PipelineResult: Extraction results with visuals and statistics
    - extract_visuals: Convenience function for full pipeline extraction
    - extract_visuals_doclayout: Extract using DocLayout-YOLO (recommended)
    - extract_tables_only: Extract only tables from PDF
    - extract_figures_only: Extract only figures from PDF

Example:
    >>> from B_parsing.B12_visual_pipeline import extract_visuals, PipelineConfig
    >>> config = PipelineConfig(detection_mode="doclayout", enable_vlm=True)
    >>> result = extract_visuals("paper.pdf", config)
    >>> print(f"Extracted {result.tables_detected} tables, {result.figures_detected} figures")

Dependencies:
    - A_core.A13_visual_models: ExtractedVisual, VisualCandidate, TriageDecision, etc.
    - B_parsing.B13_visual_detector: Visual detection with Docling
    - B_parsing.B14_visual_renderer: Visual rendering with PyMuPDF
    - B_parsing.B15_caption_extractor: Caption extraction and column layout
    - B_parsing.B16_triage: Triage decisions for VLM processing
    - B_parsing.B17_document_resolver: Document-level resolution
    - B_parsing.B22_doclayout_detector: DocLayout-YOLO detection
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from A_core.A13_visual_models import (
    CaptionProvenance,
    ExtractedVisual,
    PageLocation,
    TableStructure,
    TriageDecision,
    VisualCandidate,
    VisualType,
)
from B_parsing.B13_visual_detector import (
    DetectionResult,
    DetectorConfig,
    detect_all_visuals,
)
from B_parsing.B14_visual_renderer import (
    RenderConfig,
    render_visual,
)
from B_parsing.B15_caption_extractor import (
    CaptionSearchZones,
    extract_caption_multisource,
    infer_column_layout,
)
from B_parsing.B16_triage import (
    TriageConfig,
    get_vlm_candidates,
    triage_batch,
    compute_triage_statistics,
)
from B_parsing.B17_document_resolver import (
    ResolutionResult,
    resolve_document,
)
from B_parsing.B22_doclayout_detector import (
    detect_visuals_doclayout,
    generate_vlm_description,
)

logger = logging.getLogger(__name__)


# -------------------------
# Pipeline Configuration
# -------------------------


@dataclass
class PipelineConfig:
    """Configuration for the visual extraction pipeline."""

    # Detection mode: "doclayout" (recommended), "heuristic", or "hybrid"
    # "doclayout" uses DocLayout-YOLO for accurate detection without VLM
    detection_mode: str = "doclayout"
    detection_model: str = "claude-sonnet-4-20250514"  # Only used for VLM enrichment

    # DocLayout-YOLO settings
    doclayout_detect_dpi: int = 144
    doclayout_confidence: float = 0.3

    # Detection (used when mode is "heuristic" or "hybrid")
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    # Rendering
    render: RenderConfig = field(default_factory=RenderConfig)

    # Caption extraction
    caption_zones: CaptionSearchZones = field(default_factory=CaptionSearchZones)

    # Triage
    triage: TriageConfig = field(default_factory=TriageConfig)

    # VLM enrichment (separate from detection)
    enable_vlm: bool = True
    vlm_model: str = "claude-sonnet-4-20250514"
    validate_tables: bool = True
    generate_vlm_descriptions: bool = True  # Generate VLM title/description for visuals

    # Resolution
    merge_multipage: bool = True
    deduplicate: bool = True
    dedupe_threshold: float = 0.7


# -------------------------
# Pipeline Result
# -------------------------


@dataclass
class PipelineResult:
    """Result of visual extraction pipeline."""

    visuals: List[ExtractedVisual]

    # Statistics
    tables_detected: int
    figures_detected: int
    tables_escalated: int
    vlm_enriched: int
    merges_performed: int
    duplicates_removed: int

    # Timing
    extraction_time_seconds: float

    # Metadata
    source_file: str
    extracted_at: datetime


# -------------------------
# Pipeline
# -------------------------


class VisualExtractionPipeline:
    """
    Main visual extraction pipeline.

    Orchestrates the 4-stage extraction process:
    1. Detection (Docling + native)
    2. Rendering (PyMuPDF)
    3. Triage + VLM Enrichment
    4. Document Resolution
    """

    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self._vlm_client = None

    def _get_vlm_client(self):
        """Lazy initialization of VLM client."""
        if self._vlm_client is None and self.config.enable_vlm:
            try:
                from C_generators.C19_vlm_visual_enrichment import VLMClient, VLMConfig

                vlm_config = VLMConfig(model=self.config.vlm_model)
                self._vlm_client = VLMClient(vlm_config)
                logger.info(f"VLM client initialized with model: {self.config.vlm_model}")
            except ImportError as e:
                logger.warning(
                    f"VLM client not available: {e}. "
                    "Visual classification will use heuristics only (no VLM enrichment). "
                    "Install 'anthropic' package to enable VLM features."
                )
                print(
                    f"  [WARN] VLM not available: {e}\n"
                    "         Visual classification will use heuristics only."
                )
        return self._vlm_client

    def extract(self, pdf_path: str) -> PipelineResult:
        """
        Extract all visuals from a PDF document.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PipelineResult with extracted visuals
        """
        start_time = datetime.now()

        # Stage 1: Detection
        logger.info(f"Stage 1: Detecting visuals in {pdf_path}")
        detection_result = self._stage_detection(pdf_path)

        # Stage 2: Rendering + Caption Extraction
        logger.info(f"Stage 2: Rendering {len(detection_result.candidates)} candidates")
        rendered_candidates = self._stage_rendering(pdf_path, detection_result.candidates)

        # Stage 3: Triage + VLM Enrichment
        logger.info("Stage 3: Triage and VLM enrichment")
        enriched_visuals, vlm_count = self._stage_enrichment(
            pdf_path, rendered_candidates, detection_result.table_structures
        )

        # Stage 4: Document Resolution
        logger.info("Stage 4: Document-level resolution")
        resolution_result = self._stage_resolution(pdf_path, enriched_visuals)

        # Compute timing
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        return PipelineResult(
            visuals=resolution_result.visuals,
            tables_detected=detection_result.tables_detected,
            figures_detected=detection_result.figures_detected,
            tables_escalated=detection_result.escalated_tables,
            vlm_enriched=vlm_count,
            merges_performed=resolution_result.merges_performed,
            duplicates_removed=resolution_result.duplicates_removed,
            extraction_time_seconds=elapsed,
            source_file=pdf_path,
            extracted_at=start_time,
        )

    def _stage_detection(self, pdf_path: str) -> DetectionResult:
        """Stage 1: Detection."""
        mode = self.config.detection_mode

        if mode == "doclayout":
            # Use DocLayout-YOLO detection (recommended - no VLM needed)
            return self._detect_with_doclayout(pdf_path)
        elif mode == "hybrid":
            # Use DocLayout-YOLO with heuristic fallback
            doclayout_result = self._detect_with_doclayout(pdf_path)
            heuristic_result = detect_all_visuals(pdf_path, self.config.detector)
            # Merge results (DocLayout primary, heuristic fills gaps)
            return self._merge_detection_results(doclayout_result, heuristic_result)
        else:
            # Default: heuristic detection
            return detect_all_visuals(pdf_path, self.config.detector)

    def _detect_with_doclayout(self, pdf_path: str) -> DetectionResult:
        """Detect visuals using DocLayout-YOLO (no VLM required)."""
        try:
            result = detect_visuals_doclayout(
                pdf_path,
                detect_dpi=self.config.doclayout_detect_dpi,
                confidence_threshold=self.config.doclayout_confidence,
            )

            # Get page dimensions
            doc = fitz.open(pdf_path)
            page_dims = {}
            for i in range(doc.page_count):
                page = doc[i]
                page_dims[i + 1] = (page.rect.width, page.rect.height)
            doc.close()

            # Extract table structures using Docling for tables detected by DocLayout-YOLO
            table_structures: Dict[tuple, TableStructure] = {}
            try:
                from B_parsing.B13_visual_detector import detect_tables_with_docling
                docling_tables = detect_tables_with_docling(pdf_path, mode="fast")

                # Create a lookup map from Docling extraction
                for dt in docling_tables:
                    headers = dt.get("headers", [])
                    rows = dt.get("rows", [])
                    if headers or rows:
                        table_struct = TableStructure(
                            headers=headers,
                            rows=rows,
                            token_coverage=dt.get("token_coverage", 1.0),
                            structure_confidence=0.85,
                        )
                        table_structures[(dt["page_num"], tuple(dt["bbox_pts"]))] = table_struct
            except Exception as e:
                logger.warning(f"Docling table structure extraction failed: {e}")

            # Convert DocLayout detections to VisualCandidate format
            candidates = []
            for v in result.visuals:
                page_width, page_height = page_dims.get(v.page_num, (595, 842))
                page_area = page_width * page_height
                bbox = v.bbox_pts
                visual_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                candidate = VisualCandidate(
                    page_num=v.page_num,
                    bbox_pts=bbox,
                    page_width_pts=page_width,
                    page_height_pts=page_height,
                    area_ratio=visual_area / page_area if page_area > 0 else 0,
                    docling_type=v.visual_type,
                    confidence=v.confidence,
                    source="doclayout_yolo",
                )
                # Store caption info if available
                if v.caption_text:
                    candidate.caption_candidate = type('Caption', (), {
                        'text': v.caption_text,
                        'position': v.caption_position,
                        'parsed_reference': None,
                        'provenance': None,
                    })()

                candidates.append(candidate)

                # Match table structure by finding best overlapping Docling bbox
                if v.visual_type == "table":
                    best_match = self._find_best_table_match(
                        v.page_num, bbox, table_structures
                    )
                    if best_match:
                        # Store with DocLayout bbox as key for later lookup
                        table_structures[(v.page_num, tuple(bbox))] = best_match

            return DetectionResult(
                candidates=candidates,
                tables_detected=result.tables_detected,
                figures_detected=result.figures_detected,
                escalated_tables=0,
                detection_mode="doclayout",
                table_structures=table_structures,
            )

        except Exception as e:
            logger.error(f"DocLayout-YOLO detection failed: {e}, falling back to heuristic")
            return detect_all_visuals(pdf_path, self.config.detector)

    def _find_best_table_match(
        self,
        page_num: int,
        bbox: tuple,
        table_structures: Dict[tuple, TableStructure],
    ) -> Optional[TableStructure]:
        """Find the best matching table structure by IoU overlap."""
        best_iou = 0.3  # Minimum threshold
        best_match = None

        for (p, stored_bbox), table_struct in table_structures.items():
            if p != page_num:
                continue
            iou = self._compute_iou(bbox, stored_bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = table_struct

        return best_match

    def _merge_detection_results(
        self,
        vlm_result: DetectionResult,
        heuristic_result: DetectionResult,
    ) -> DetectionResult:
        """Merge VLM and heuristic detection results."""
        # Start with VLM candidates (primary)
        merged = list(vlm_result.candidates)
        vlm_bboxes = [(c.page_num, c.bbox_pts) for c in vlm_result.candidates]

        # Add heuristic candidates that don't overlap with VLM
        for hc in heuristic_result.candidates:
            is_duplicate = False
            for page_num, bbox in vlm_bboxes:
                if hc.page_num == page_num:
                    # Check IoU
                    iou = self._compute_iou(hc.bbox_pts, bbox)
                    if iou > 0.3:
                        is_duplicate = True
                        break
            if not is_duplicate:
                merged.append(hc)

        # Merge table structures from both sources
        merged_table_structures = dict(vlm_result.table_structures)
        merged_table_structures.update(heuristic_result.table_structures)

        return DetectionResult(
            candidates=merged,
            tables_detected=sum(1 for c in merged if c.docling_type == "table"),
            figures_detected=sum(1 for c in merged if c.docling_type != "table"),
            escalated_tables=heuristic_result.escalated_tables,
            detection_mode="hybrid",
            table_structures=merged_table_structures,
        )

    def _compute_iou(self, bbox1, bbox2) -> float:
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

    def _stage_rendering(
        self,
        pdf_path: str,
        candidates: List[VisualCandidate],
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Render visuals and extract captions.

        Returns list of dicts with candidate + rendered image + caption.
        """
        rendered = []

        doc = fitz.open(pdf_path)
        try:
            # Pre-compute column layouts
            column_layouts = {}
            for page_idx in range(doc.page_count):
                page_num = page_idx + 1
                column_layouts[page_num] = infer_column_layout(doc, page_num)

            for candidate in candidates:
                # Determine visual type for rendering
                visual_type = "table" if candidate.docling_type == "table" else "figure"

                # IMPORTANT: Detect caption BEFORE rendering to get correct padding
                caption = candidate.caption_candidate
                if caption is None:
                    caption_result = extract_caption_multisource(
                        doc,
                        candidate.page_num,
                        candidate.bbox_pts,
                        zones=self.config.caption_zones,
                        column_layout=column_layouts.get(candidate.page_num),
                        visual_type_hint=visual_type,
                    )
                    caption = caption_result.best_caption

                # Get caption position hint for padding
                caption_position = None
                if caption:
                    caption_position = caption.position

                # Render with correct caption padding
                rendered_result = render_visual(
                    doc,
                    candidate.page_num,
                    candidate.bbox_pts,
                    visual_type=visual_type,
                    caption_position=caption_position,
                    config=self.config.render,
                )

                if rendered_result is None:
                    continue

                rendered.append({
                    "candidate": candidate,
                    "rendered": rendered_result,
                    "caption": caption,
                })

        finally:
            doc.close()

        return rendered

    def _stage_enrichment(
        self,
        pdf_path: str,
        rendered_candidates: List[Dict[str, Any]],
        table_structures: Dict[tuple, TableStructure] = None,
    ) -> tuple[List[ExtractedVisual], int]:
        """
        Stage 3: Triage and VLM enrichment.

        Returns (list of ExtractedVisual, count of VLM-enriched).
        """
        if table_structures is None:
            table_structures = {}
        # Build candidates list for triage
        candidates = [r["candidate"] for r in rendered_candidates]

        # Triage
        triaged = triage_batch(candidates, config=self.config.triage)
        stats = compute_triage_statistics(triaged)

        logger.info(
            f"Triage: {stats.skip_count} skip, "
            f"{stats.cheap_path_count} cheap, "
            f"{stats.vlm_required_count} VLM"
        )

        # Get VLM candidates
        vlm_candidates = get_vlm_candidates(triaged)
        vlm_candidate_ids = {c.id for c in vlm_candidates}

        # VLM enrichment
        vlm_results = {}
        if self.config.enable_vlm and vlm_candidates:
            vlm_client = self._get_vlm_client()
            if vlm_client:
                try:
                    from C_generators.C19_vlm_visual_enrichment import classify_visual

                    logger.info(f"Running VLM enrichment on {len(vlm_candidates)} candidates")
                    for rendered_data in rendered_candidates:
                        candidate = rendered_data["candidate"]
                        if candidate.id not in vlm_candidate_ids:
                            continue

                        image_base64 = rendered_data["rendered"].image_base64
                        result = classify_visual(image_base64, vlm_client)
                        if result:
                            vlm_results[candidate.id] = result

                except Exception as e:
                    logger.error(f"VLM enrichment failed: {e}")
                    print(f"  [ERROR] VLM enrichment failed: {e}")
            else:
                # VLM client not available - warn user about fallback
                logger.warning(
                    f"VLM client not available. {len(vlm_candidates)} visuals "
                    "will be classified using heuristics only (less accurate)."
                )
                print(
                    f"  [WARN] VLM not available - {len(vlm_candidates)} visuals "
                    "classified using heuristics only"
                )
        elif vlm_candidates and not self.config.enable_vlm:
            logger.info(
                f"VLM disabled in config. {len(vlm_candidates)} visuals "
                "classified using heuristics only."
            )

        # Build ExtractedVisual objects
        visuals: List[ExtractedVisual] = []
        vlm_count = 0

        for i, rendered_data in enumerate(rendered_candidates):
            candidate = rendered_data["candidate"]
            rendered = rendered_data["rendered"]
            caption = rendered_data["caption"]

            # Get triage result
            triage_result = triaged[i][1] if i < len(triaged) else None

            # Skip if triaged to skip
            if triage_result and triage_result.decision == TriageDecision.SKIP:
                continue

            # Get VLM result if available
            vlm_result = vlm_results.get(candidate.id)

            # Determine visual type
            if vlm_result:
                visual_type = vlm_result.classification.visual_type
                confidence = vlm_result.classification.confidence
                vlm_count += 1
            elif candidate.docling_type == "table":
                visual_type = VisualType.TABLE
                confidence = 0.85
            else:
                visual_type = VisualType.FIGURE
                confidence = 0.80

            # Get reference
            reference = None
            if vlm_result and vlm_result.parsed_reference:
                reference = vlm_result.parsed_reference
            elif caption and caption.parsed_reference:
                reference = caption.parsed_reference

            # Get caption text
            caption_text = None
            caption_provenance = None
            if vlm_result and vlm_result.extracted_caption:
                caption_text = vlm_result.extracted_caption
                caption_provenance = CaptionProvenance.VLM
            elif caption:
                caption_text = caption.text
                caption_provenance = caption.provenance

            # Determine extraction method
            if candidate.source == "layout_aware":
                extraction_method = "layout_aware+vlm" if vlm_result else "layout_aware"
            else:
                extraction_method = "docling+vlm" if vlm_result else "docling_only"

            # Generate VLM title and description
            vlm_title = None
            vlm_description = None
            if self.config.generate_vlm_descriptions and self.config.enable_vlm:
                try:
                    image_bytes = base64.b64decode(rendered.image_base64)
                    type_str = "table" if visual_type == VisualType.TABLE else "figure"
                    vlm_desc_result = generate_vlm_description(
                        image_bytes=image_bytes,
                        visual_type=type_str,
                        caption_text=caption_text,
                        model=self.config.vlm_model,
                    )
                    vlm_title = vlm_desc_result.get("title")
                    vlm_description = vlm_desc_result.get("description")
                except Exception as e:
                    logger.warning(f"VLM description generation failed: {e}")

            # Look up table structure if this is a table
            docling_table = None
            if visual_type == VisualType.TABLE:
                table_key = (candidate.page_num, tuple(candidate.bbox_pts))
                docling_table = table_structures.get(table_key)

            # Create ExtractedVisual
            visual = ExtractedVisual(
                visual_type=visual_type,
                confidence=confidence,
                page_range=[candidate.page_num],
                bbox_pts_per_page=[
                    PageLocation(
                        page_num=candidate.page_num,
                        bbox_pts=candidate.bbox_pts,
                    )
                ],
                caption_text=caption_text,
                caption_provenance=caption_provenance,
                reference=reference,
                image_base64=rendered.image_base64,
                image_format=rendered.image_format,
                render_dpi=rendered.dpi,
                docling_table=docling_table,
                extraction_method=extraction_method,
                source_file=pdf_path,
                triage_decision=triage_result.decision if triage_result else None,
                triage_reason=triage_result.reason if triage_result else None,
                layout_code=candidate.layout_code,
                position_code=candidate.position_code,
                layout_filename=candidate.layout_filename,
                vlm_title=vlm_title,
                vlm_description=vlm_description,
            )
            visuals.append(visual)

        return visuals, vlm_count

    def _stage_resolution(
        self,
        pdf_path: str,
        visuals: List[ExtractedVisual],
    ) -> ResolutionResult:
        """Stage 4: Document-level resolution."""
        return resolve_document(
            visuals,
            pdf_path,
            merge_multipage=self.config.merge_multipage,
            deduplicate=self.config.deduplicate,
            dedupe_threshold=self.config.dedupe_threshold,
        )


# -------------------------
# Convenience Functions
# -------------------------


def extract_visuals(
    pdf_path: str,
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """
    Extract all visuals from a PDF document.

    This is a convenience function that creates a pipeline and runs extraction.

    Args:
        pdf_path: Path to PDF file
        config: Pipeline configuration (uses defaults if not provided)

    Returns:
        PipelineResult with extracted visuals
    """
    if config is None:
        config = PipelineConfig()

    pipeline = VisualExtractionPipeline(config)
    return pipeline.extract(pdf_path)


def extract_tables_only(pdf_path: str) -> List[ExtractedVisual]:
    """
    Extract only tables from a PDF document.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of table visuals
    """
    result = extract_visuals(pdf_path)
    return [v for v in result.visuals if v.is_table]


def extract_figures_only(pdf_path: str) -> List[ExtractedVisual]:
    """
    Extract only figures from a PDF document.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of figure visuals
    """
    result = extract_visuals(pdf_path)
    return [v for v in result.visuals if v.is_figure]


def extract_visuals_doclayout(pdf_path: str) -> PipelineResult:
    """
    Extract visuals using DocLayout-YOLO detection (recommended).

    This method uses DocLayout-YOLO for accurate figure/table detection
    without requiring VLM. Fast and accurate.

    Args:
        pdf_path: Path to PDF file

    Returns:
        PipelineResult with extracted visuals
    """
    config = PipelineConfig(detection_mode="doclayout")
    pipeline = VisualExtractionPipeline(config)
    return pipeline.extract(pdf_path)


__all__ = [
    # Types
    "PipelineConfig",
    "PipelineResult",
    "VisualExtractionPipeline",
    # Convenience functions
    "extract_visuals",
    "extract_visuals_doclayout",
    "extract_tables_only",
    "extract_figures_only",
]
