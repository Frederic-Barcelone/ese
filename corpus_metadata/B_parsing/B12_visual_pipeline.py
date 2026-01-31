# corpus_metadata/B_parsing/B12_visual_pipeline.py
"""
Visual Extraction Pipeline Orchestrator.

Main entry point for the visual extraction pipeline that orchestrates:
- Stage 1: Detection (Docling + native extraction)
- Stage 2: Rendering (PyMuPDF with point-based padding)
- Stage 3: Triage + VLM Enrichment
- Stage 4: Document-Level Resolution

Usage:
    pipeline = VisualExtractionPipeline()
    result = pipeline.extract(pdf_path)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from A_core.A13_visual_models import (
    CaptionProvenance,
    ExtractedVisual,
    PageLocation,
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

logger = logging.getLogger(__name__)


# -------------------------
# Pipeline Configuration
# -------------------------


@dataclass
class PipelineConfig:
    """Configuration for the visual extraction pipeline."""

    # Detection
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    # Rendering
    render: RenderConfig = field(default_factory=RenderConfig)

    # Caption extraction
    caption_zones: CaptionSearchZones = field(default_factory=CaptionSearchZones)

    # Triage
    triage: TriageConfig = field(default_factory=TriageConfig)

    # VLM
    enable_vlm: bool = True
    vlm_model: str = "claude-sonnet-4-20250514"
    validate_tables: bool = True

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
                from C_generators.C16_vlm_visual_enrichment import VLMClient, VLMConfig

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
            pdf_path, rendered_candidates
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
        return detect_all_visuals(pdf_path, self.config.detector)

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

                # Get caption position hint
                caption_position = None
                if candidate.caption_candidate:
                    caption_position = candidate.caption_candidate.position

                # Render
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

                # Extract caption if not already detected
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
    ) -> tuple[List[ExtractedVisual], int]:
        """
        Stage 3: Triage and VLM enrichment.

        Returns (list of ExtractedVisual, count of VLM-enriched).
        """
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
                    from C_generators.C16_vlm_visual_enrichment import classify_visual

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
                extraction_method="docling+vlm" if vlm_result else "docling_only",
                source_file=pdf_path,
                triage_decision=triage_result.decision if triage_result else None,
                triage_reason=triage_result.reason if triage_result else None,
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


__all__ = [
    # Types
    "PipelineConfig",
    "PipelineResult",
    "VisualExtractionPipeline",
    # Convenience functions
    "extract_visuals",
    "extract_tables_only",
    "extract_figures_only",
]
