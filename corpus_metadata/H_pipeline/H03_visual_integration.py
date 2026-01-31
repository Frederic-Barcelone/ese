# corpus_metadata/H_pipeline/H03_visual_integration.py
"""
Visual Pipeline Integration for Orchestrator.

Provides easy integration of the new visual extraction pipeline
with the existing orchestrator. Handles configuration loading
and result export.

Usage in orchestrator:
    from H_pipeline.H03_visual_integration import VisualPipelineIntegration

    # In Orchestrator.__init__
    self.visual_integration = VisualPipelineIntegration(config)

    # In process_single_pdf
    if self.visual_integration.enabled:
        visual_result = self.visual_integration.extract(pdf_path)
        self.visual_integration.export(visual_result, output_dir)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from B_parsing.B12_visual_pipeline import (
    PipelineConfig,
    PipelineResult,
    VisualExtractionPipeline,
)
from B_parsing.B13_visual_detector import DetectorConfig
from B_parsing.B14_visual_renderer import RenderConfig
from B_parsing.B15_caption_extractor import CaptionSearchZones
from B_parsing.B16_triage import TriageConfig
from J_export.J02_visual_export import (
    export_figures_only,
    export_images_separately,
    export_tables_only,
    export_visuals_to_json,
)

logger = logging.getLogger(__name__)


class VisualPipelineIntegration:
    """
    Integration wrapper for the visual extraction pipeline.

    Loads configuration from the orchestrator's config dict and
    provides simple methods for extraction and export.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize from orchestrator config.

        Args:
            config: Full config dict from orchestrator
        """
        self.config = config

        # Get visual extraction config
        pipeline_cfg = config.get("extraction_pipeline", {})
        visual_cfg = pipeline_cfg.get("visual_extraction", {})

        # Check if enabled
        self.enabled = visual_cfg.get("enabled", False)

        if not self.enabled:
            logger.info("Visual extraction pipeline disabled")
            self._pipeline = None
            return

        # Check dependencies and report status
        self._check_dependencies()

        # Build pipeline config from YAML
        pipeline_config = self._build_pipeline_config(visual_cfg)
        self._pipeline = VisualExtractionPipeline(pipeline_config)

        logger.info("Visual extraction pipeline initialized")

    def _check_dependencies(self) -> None:
        """Check and report status of optional dependencies."""
        # Check Docling
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401
            logger.info("Docling available - table extraction enabled")
        except ImportError:
            logger.warning(
                "Docling not installed - table extraction will be DISABLED. "
                "Only native figure detection will work. "
                "Install with: pip install docling"
            )
            print(
                "  [WARN] Docling not installed - table extraction DISABLED\n"
                "         Install with: pip install docling"
            )

        # Check PyMuPDF (required)
        try:
            import fitz  # noqa: F401
            logger.info("PyMuPDF available - rendering enabled")
        except ImportError:
            logger.error(
                "PyMuPDF (fitz) not installed - visual pipeline cannot run. "
                "Install with: pip install pymupdf"
            )
            print(
                "  [ERROR] PyMuPDF not installed - visual pipeline CANNOT RUN\n"
                "          Install with: pip install pymupdf"
            )
            self.enabled = False
            self._pipeline = None
            return

        # Check Anthropic (for VLM)
        try:
            import anthropic  # noqa: F401
            logger.info("Anthropic SDK available - VLM enrichment enabled")
        except ImportError:
            logger.warning(
                "Anthropic SDK not installed - VLM enrichment will be DISABLED. "
                "Visual classification will use heuristics only. "
                "Install with: pip install anthropic"
            )
            print(
                "  [WARN] Anthropic SDK not installed - VLM enrichment DISABLED\n"
                "         Classification will use heuristics only.\n"
                "         Install with: pip install anthropic"
            )

    def _build_pipeline_config(self, visual_cfg: Dict[str, Any]) -> PipelineConfig:
        """Build PipelineConfig from YAML config section."""
        detection_cfg = visual_cfg.get("detection", {})
        rendering_cfg = visual_cfg.get("rendering", {})
        triage_cfg = visual_cfg.get("triage", {})
        vlm_cfg = visual_cfg.get("vlm", {})
        resolution_cfg = visual_cfg.get("resolution", {})

        # Build detector config
        detector = DetectorConfig(
            default_table_mode=detection_cfg.get("table_mode_default", "fast"),
            enable_escalation=detection_cfg.get("enable_escalation", True),
            min_figure_area_ratio=detection_cfg.get("min_figure_area_ratio", 0.02),
            filter_noise=detection_cfg.get("filter_noise", True),
            repeat_threshold=detection_cfg.get("repeat_threshold", 3),
        )

        # Build render config
        render = RenderConfig(
            default_dpi=rendering_cfg.get("default_dpi", 300),
            min_dpi=rendering_cfg.get("min_dpi", 200),
            max_dpi=rendering_cfg.get("max_dpi", 400),
            padding_sides_pts=rendering_cfg.get("padding_sides_pts", 12.0),
            padding_caption_pts=rendering_cfg.get("padding_caption_pts", 72.0),
        )

        # Build triage config
        triage = TriageConfig(
            skip_area_ratio=triage_cfg.get("skip_area_ratio", 0.02),
            vlm_area_threshold=triage_cfg.get("vlm_area_threshold", 0.10),
            repeat_threshold=triage_cfg.get("repeat_threshold", 3),
        )

        # Caption zones use defaults
        caption_zones = CaptionSearchZones()

        return PipelineConfig(
            detector=detector,
            render=render,
            caption_zones=caption_zones,
            triage=triage,
            enable_vlm=vlm_cfg.get("enabled", True),
            vlm_model=vlm_cfg.get("model", "claude-sonnet-4-20250514"),
            validate_tables=vlm_cfg.get("validate_tables", True),
            merge_multipage=resolution_cfg.get("merge_multipage", True),
            deduplicate=resolution_cfg.get("deduplicate", True),
            dedupe_threshold=resolution_cfg.get("dedupe_threshold", 0.7),
        )

    def extract(self, pdf_path: str) -> Optional[PipelineResult]:
        """
        Extract visuals from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PipelineResult, or None if disabled
        """
        if not self.enabled or self._pipeline is None:
            return None

        try:
            result = self._pipeline.extract(pdf_path)
            logger.info(
                f"Extracted {len(result.visuals)} visuals "
                f"({result.tables_detected} tables, {result.figures_detected} figures)"
            )
            return result

        except Exception as e:
            logger.error(f"Visual extraction failed for {pdf_path}: {e}")
            return None

    def export(
        self,
        result: PipelineResult,
        output_dir: Path,
        doc_name: str,
        export_images: bool = False,
    ) -> Dict[str, Path]:
        """
        Export visual extraction results.

        Args:
            result: Pipeline result to export
            output_dir: Output directory
            doc_name: Document name (for filename generation)
            export_images: Whether to export images as separate files

        Returns:
            Dict of exported file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported: Dict[str, Path] = {}

        # Generate timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export all visuals
        visuals_path = output_dir / f"visuals_{doc_name}_{timestamp}.json"
        export_visuals_to_json(result, visuals_path)
        exported["visuals"] = visuals_path

        # Also export tables and figures separately for convenience
        if result.tables_detected > 0:
            tables_path = output_dir / f"tables_{doc_name}_{timestamp}.json"
            export_tables_only(result, tables_path)
            exported["tables"] = tables_path

        if result.figures_detected > 0:
            figures_path = output_dir / f"figures_{doc_name}_{timestamp}.json"
            export_figures_only(result, figures_path)
            exported["figures"] = figures_path

        # Optionally export images as files
        if export_images:
            images_dir = output_dir / f"visual_images_{doc_name}"
            export_images_separately(result, images_dir)
            exported["images_dir"] = images_dir

        logger.info(f"Exported visual results to {output_dir}")
        return exported


def create_visual_integration(config: Dict[str, Any]) -> VisualPipelineIntegration:
    """
    Factory function to create visual integration.

    Args:
        config: Orchestrator config dict

    Returns:
        VisualPipelineIntegration instance
    """
    return VisualPipelineIntegration(config)


__all__ = [
    "VisualPipelineIntegration",
    "create_visual_integration",
]
