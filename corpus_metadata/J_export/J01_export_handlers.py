# corpus_metadata/J_export/J01_export_handlers.py
"""
Export handlers for pipeline results.

Handles exporting extraction results to various JSON formats including
abbreviations, diseases, genes, drugs, pharma companies, authors, citations,
feasibility data, images, tables, and document metadata.

REFACTORED: Entity and metadata exporters extracted to:
- J01a_entity_exporters.py: Disease, gene, drug, pharma, author, citation exports
- J01b_metadata_exporters.py: Document metadata, care pathways, recommendations
"""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from Z_utils.Z04_image_utils import (
    get_image_size_bytes,
    is_image_oversized,
    extract_ocr_text_from_base64,
    PYTESSERACT_AVAILABLE,
)

# Extracted entity exporters
from J_export.J01a_entity_exporters import (
    export_disease_results as _export_disease_results,
    export_gene_results as _export_gene_results,
    export_drug_results as _export_drug_results,
    export_pharma_results as _export_pharma_results,
    export_author_results as _export_author_results,
    export_citation_results as _export_citation_results,
)

# Extracted metadata exporters
from J_export.J01b_metadata_exporters import (
    export_document_metadata as _export_document_metadata,
    export_care_pathways as _export_care_pathways,
    export_recommendations as _export_recommendations,
)

if TYPE_CHECKING:
    from B_parsing.B02_doc_graph import DocumentGraph
    from A_core.A01_domain_models import Candidate, ExtractedEntity
    from A_core.A04_heuristics_config import HeuristicsCounters
    from A_core.A05_disease_models import ExtractedDisease
    from A_core.A06_drug_models import ExtractedDrug
    from A_core.A19_gene_models import ExtractedGene
    from A_core.A09_pharma_models import ExtractedPharma
    from A_core.A10_author_models import ExtractedAuthor
    from A_core.A11_citation_models import ExtractedCitation
    from A_core.A07_feasibility_models import FeasibilityCandidate
    from A_core.A08_document_metadata_models import DocumentMetadata
    from A_core.A17_care_pathway_models import CarePathway
    from A_core.A18_recommendation_models import RecommendationSet
    from D_validation.D02_llm_engine import ClaudeClient


class ExportManager:
    """
    Manages export of extraction results to JSON files.

    Handles formatting and writing of all entity types extracted from documents,
    including images and tables with optional Vision LLM analysis.
    """

    def __init__(
        self,
        run_id: str,
        pipeline_version: str,
        output_dir: Optional[Path] = None,
        gold_json: Optional[str] = None,
        claude_client: Optional["ClaudeClient"] = None,
    ) -> None:
        """
        Initialize the export manager.

        Args:
            run_id: Unique identifier for this pipeline run
            pipeline_version: Version string for the pipeline
            output_dir: Optional override for output directory
            gold_json: Path to gold standard JSON for evaluation
            claude_client: Optional Claude client for Vision LLM analysis
        """
        self.run_id = run_id
        self.pipeline_version = pipeline_version
        self.output_dir_override = output_dir
        self.gold_json = gold_json
        self.claude_client = claude_client

    def get_output_dir(self, pdf_path: Path) -> Path:
        """Get output directory for a PDF file.

        Creates a folder with the same name as the PDF (without extension)
        in the same directory as the PDF file.

        Args:
            pdf_path: Path to the PDF file being processed

        Returns:
            Path to output directory (created if it doesn't exist)
        """
        if self.output_dir_override:
            out_dir = self.output_dir_override
        else:
            # Create folder named after PDF in same directory
            out_dir = pdf_path.parent / pdf_path.stem

        # Ensure directory exists
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def render_figure_with_padding(
        self,
        pdf_path: Path,
        page_num: int,
        bbox: Tuple[float, float, float, float],
        dpi: int = 200,
        padding: int = 30,
        bottom_padding: int = 100,
        right_padding: int = 50,
        top_padding: Optional[int] = None,
        is_table: bool = False,
    ) -> Optional[str]:
        """
        Re-render a figure from PDF with extra padding for captions/legends.

        Args:
            pdf_path: Path to PDF file
            page_num: 1-indexed page number
            bbox: (x0, y0, x1, y1) bounding box in PDF points
            dpi: Resolution for rendering
            padding: Extra points around left side (default 30pt ~0.4in)
            bottom_padding: Extra points below figure for captions (default 100pt ~1.4in)
            right_padding: Extra points to right for legends (default 50pt ~0.7in)
            top_padding: Extra points above figure (default same as padding)
            is_table: If True, use PyMuPDF table detection instead of full-width expansion

        Returns:
            Base64-encoded PNG string, or None if rendering fails
        """
        if top_padding is None:
            top_padding = padding
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return None

        try:
            doc = fitz.open(str(pdf_path))
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return None

            page = doc[page_num - 1]  # 0-indexed
            page_width = page.rect.width
            page_height = page.rect.height

            x0, y0, x1, y1 = bbox

            # Handle coordinate space issues (Unstructured uses higher DPI)
            # Scale each axis independently to avoid distortion
            needs_x_scale = x1 > page_width * 1.1
            needs_y_scale = y1 > page_height * 1.1

            if needs_x_scale or needs_y_scale:
                # Calculate scale ratio per axis
                if needs_x_scale and needs_y_scale:
                    # Both axes exceed - use consistent ratio from the more problematic axis
                    x_ratio = page_width / x1
                    y_ratio = page_height / y1
                    ratio = min(x_ratio, y_ratio)  # Use tighter constraint
                    x0, x1 = x0 * ratio, x1 * ratio
                    y0, y1 = y0 * ratio, y1 * ratio
                elif needs_x_scale:
                    x_ratio = page_width / x1
                    x0, x1 = x0 * x_ratio, x1 * x_ratio
                    y0, y1 = y0 * x_ratio, y1 * x_ratio  # Scale Y proportionally
                else:  # needs_y_scale only
                    y_ratio = page_height / y1
                    x0, x1 = x0 * y_ratio, x1 * y_ratio  # Scale X proportionally
                    y0, y1 = y0 * y_ratio, y1 * y_ratio

            # Detect 2-column layout by checking if figure is in left or right half
            figure_center_x = (x0 + x1) / 2
            is_two_column = (
                (figure_center_x < page_width * 0.45) or  # Left column
                (figure_center_x > page_width * 0.55)     # Right column
            )

            # For tables, use PyMuPDF's table detection to get accurate boundaries
            # This prevents capturing content from adjacent columns in 2-column layouts
            if is_table:
                from B_parsing.B14_visual_renderer import find_table_bbox_pymupdf
                doc.close()
                pymupdf_bbox = find_table_bbox_pymupdf(str(pdf_path), page_num, (x0, y0, x1, y1))
                doc = fitz.open(str(pdf_path))
                page = doc[page_num - 1]
                if pymupdf_bbox:
                    x0, y0, x1, y1 = pymupdf_bbox
                # Use minimal horizontal padding for tables to stay within column
                padding = 15
                right_padding = 15
            elif is_two_column:
                # In 2-column layout, don't expand horizontally - stay within column
                # Use moderate padding that won't cross column boundary
                column_width = page_width / 2
                max_padding = column_width * 0.1  # Max 10% of column width
                padding = min(padding, max_padding)
                right_padding = min(right_padding, max_padding)
            else:
                # Full-width figure (centered or spanning columns)
                figure_width = x1 - x0
                figure_height = y1 - y0
                is_significant_figure = (
                    figure_width > page_width * 0.4 or  # Must span >40% (not 15%)
                    figure_height > page_height * 0.25
                )
                if is_significant_figure:
                    # Expand to nearly full page width (small margins to avoid edge artifacts)
                    margin = 36  # ~0.5 inch margin
                    x0 = margin
                    x1 = page_width - margin

            # Create clip rectangle with generous padding to capture captions/legends
            clip_rect = fitz.Rect(
                max(0, x0 - padding),
                max(0, y0 - top_padding),
                min(page_width, x1 + right_padding),
                min(page_height, y1 + bottom_padding),
            )

            # Validate clip rect
            if clip_rect.width <= 0 or clip_rect.height <= 0:
                doc.close()
                return None

            # Render to pixmap
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)

            # Convert to base64 PNG
            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            doc.close()
            return img_base64

        except Exception as e:
            print(f"  [WARN] Failed to render figure with padding: {e}")
            return None

    def export_extracted_text(self, pdf_path: Path, doc: "DocumentGraph") -> None:
        """Export extracted text to file."""
        out_dir = self.get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = out_dir / f"{pdf_path.stem}_{timestamp}.txt"

        text_lines = []
        current_page = None

        for block in doc.iter_linear_blocks(skip_header_footer=False):
            if block.page_num != current_page:
                if current_page is not None:
                    text_lines.append("")
                text_lines.append(f"--- Page {block.page_num} ---")
                text_lines.append("")
                current_page = block.page_num

            text = (block.text or "").strip()
            if text:
                text_lines.append(text)

        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(text_lines))
            print(f"  Extracted text: {txt_path.name}")
        except Exception as e:
            print(f"  [WARN] Failed to export text: {e}")

    def export_results(
        self,
        pdf_path: Path,
        results: List["ExtractedEntity"],
        candidates: List["Candidate"],
        counters: Optional["HeuristicsCounters"] = None,
        disease_results: Optional[List["ExtractedDisease"]] = None,
        drug_results: Optional[List["ExtractedDrug"]] = None,
        pharma_results: Optional[List["ExtractedPharma"]] = None,
    ) -> None:
        """Export main results to JSON."""
        from A_core.A01_domain_models import ValidationStatus

        out_dir = self.get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        export_data: Dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "document": pdf_path.name,
            "document_path": str(pdf_path),
            "total_candidates": len(candidates),
            "total_validated": sum(
                1 for r in results if r.status == ValidationStatus.VALIDATED
            ),
            "total_rejected": sum(
                1 for r in results if r.status == ValidationStatus.REJECTED
            ),
            "total_ambiguous": sum(
                1 for r in results if r.status == ValidationStatus.AMBIGUOUS
            ),
            "heuristics_counters": counters.to_dict() if counters else None,
            "abbreviations": [],
            "diseases": [],
            "drugs": [],
            "pharma": [],
        }

        for entity in results:
            if entity.status == ValidationStatus.VALIDATED:
                lexicon_ids = None
                if entity.provenance.lexicon_ids:
                    lexicon_ids = [
                        {"source": lid.source, "id": lid.id}
                        for lid in entity.provenance.lexicon_ids
                    ]

                export_data["abbreviations"].append(
                    {
                        "short_form": entity.short_form,
                        "long_form": entity.long_form,
                        "confidence": entity.confidence_score,
                        "field_type": entity.field_type.value,
                        "page": entity.primary_evidence.location.page_num
                        if entity.primary_evidence
                        else None,
                        "context_text": entity.primary_evidence.text
                        if entity.primary_evidence
                        else None,
                        "lexicon_source": entity.provenance.lexicon_source,
                        "lexicon_ids": lexicon_ids,
                    }
                )

        # Add diseases to export
        if disease_results:
            for disease in disease_results:
                if disease.status == ValidationStatus.VALIDATED:
                    export_data["diseases"].append(
                        {
                            "name": disease.preferred_label,
                            "matched_text": disease.matched_text,
                            "abbreviation": disease.abbreviation,
                            "confidence": disease.confidence_score,
                            "is_rare": disease.is_rare_disease,
                            "icd10": disease.icd10_code,
                            "orpha": disease.orpha_code,
                        }
                    )

        # Add drugs to export
        if drug_results:
            for drug in drug_results:
                if drug.status == ValidationStatus.VALIDATED:
                    export_data["drugs"].append(
                        {
                            "name": drug.preferred_name,
                            "matched_text": drug.matched_text,
                            "compound_id": drug.compound_id,
                            "confidence": drug.confidence_score,
                            "is_investigational": drug.is_investigational,
                            "phase": drug.development_phase,
                        }
                    )

        # Add pharma companies to export
        if pharma_results:
            for pharma in pharma_results:
                if pharma.status == ValidationStatus.VALIDATED:
                    export_data["pharma"].append(
                        {
                            "name": pharma.canonical_name,
                            "matched_text": pharma.matched_text,
                            "full_name": pharma.full_name,
                            "headquarters": pharma.headquarters,
                            "parent_company": pharma.parent_company,
                            "confidence": pharma.confidence_score,
                        }
                    )

        out_file = out_dir / f"abbreviations_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"\n  Exported: {out_file}")

    def export_disease_results(
        self, pdf_path: Path, results: List["ExtractedDisease"]
    ) -> None:
        """Export disease detection results to separate JSON file."""
        out_dir = self.get_output_dir(pdf_path)
        _export_disease_results(
            out_dir, pdf_path, results, self.run_id, self.pipeline_version
        )

    def export_gene_results(
        self, pdf_path: Path, results: List["ExtractedGene"]
    ) -> None:
        """Export gene detection results to separate JSON file."""
        out_dir = self.get_output_dir(pdf_path)
        _export_gene_results(
            out_dir, pdf_path, results, self.run_id, self.pipeline_version
        )

    def export_drug_results(
        self, pdf_path: Path, results: List["ExtractedDrug"]
    ) -> None:
        """Export drug detection results to separate JSON file."""
        out_dir = self.get_output_dir(pdf_path)
        _export_drug_results(
            out_dir, pdf_path, results, self.run_id, self.pipeline_version
        )

    def export_pharma_results(
        self, pdf_path: Path, results: List["ExtractedPharma"]
    ) -> None:
        """Export pharma company detection results to separate JSON file."""
        out_dir = self.get_output_dir(pdf_path)
        _export_pharma_results(
            out_dir, pdf_path, results, self.run_id, self.pipeline_version
        )

    def export_author_results(
        self, pdf_path: Path, results: List["ExtractedAuthor"]
    ) -> None:
        """Export author detection results to separate JSON file."""
        out_dir = self.get_output_dir(pdf_path)
        _export_author_results(
            out_dir, pdf_path, results, self.run_id, self.pipeline_version
        )

    def export_citation_results(
        self, pdf_path: Path, results: List["ExtractedCitation"]
    ) -> None:
        """Export citation detection results to separate JSON file with API validation."""
        out_dir = self.get_output_dir(pdf_path)
        _export_citation_results(
            out_dir, pdf_path, results, self.run_id, self.pipeline_version
        )

    def export_feasibility_results(
        self, pdf_path: Path, results: List["FeasibilityCandidate"], doc: Optional["DocumentGraph"] = None
    ) -> None:
        """Export feasibility extraction results to JSON file."""
        from C_generators.C00_strategy_identifiers import IdentifierExtractor, IdentifierType
        from A_core.A07_feasibility_models import (
            FeasibilityExportDocument,
            FeasibilityExportEntry,
            EvidenceExport,
            TrialIdentifier,
        )

        out_dir = self.get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract trial identifiers (NCT, EudraCT, CTIS, etc.)
        trial_ids: List[TrialIdentifier] = []
        if doc is not None:
            try:
                id_extractor = IdentifierExtractor()
                extracted_ids = id_extractor.extract_identifiers(doc)

                # Registry URLs
                registry_urls = {
                    "NCT": "https://clinicaltrials.gov/study/",
                    "EudraCT": "https://www.clinicaltrialsregister.eu/ctr-search/search?query=",
                    "ISRCTN": "https://www.isrctn.com/",
                }

                for ext_id in extracted_ids:
                    # Only include trial-related identifiers
                    if ext_id.id_type in {IdentifierType.NCT, IdentifierType.EUDRACT, IdentifierType.ISRCTN}:
                        url = None
                        registry = None
                        if ext_id.id_type == IdentifierType.NCT:
                            registry = "ClinicalTrials.gov"
                            url = f"{registry_urls['NCT']}NCT{ext_id.normalized}"
                        elif ext_id.id_type == IdentifierType.EUDRACT:
                            registry = "EU Clinical Trials Register"
                            url = f"{registry_urls['EudraCT']}{ext_id.normalized}"
                        elif ext_id.id_type == IdentifierType.ISRCTN:
                            registry = "ISRCTN Registry"
                            url = f"{registry_urls['ISRCTN']}ISRCTN{ext_id.normalized}"

                        trial_ids.append(TrialIdentifier(
                            id_type=ext_id.id_type.value,
                            value=ext_id.value,
                            registry=registry,
                            url=url,
                            title=ext_id.long_form,
                        ))
            except Exception as e:
                print(f"  [WARN] Trial ID extraction failed: {e}")

        # Group by field type
        study_design_data = None
        operational_burden_data = None
        screening_flow_data = None
        eligibility_inclusion = []
        eligibility_exclusion = []
        epidemiology = []
        patient_journey = []
        endpoints = []
        sites = []

        def _convert_evidence(evidence_list) -> List[EvidenceExport]:
            """Convert EvidenceSpan list to EvidenceExport list for export."""
            result = []
            for ev in (evidence_list or []):
                result.append(EvidenceExport(
                    page=ev.page,
                    quote=ev.quote,
                    source_node_id=ev.source_node_id,
                    source_doc_id=ev.source_doc_id,
                ))
            return result

        for r in results:
            # Handle NERCandidate objects - merge epidemiology into main export
            if not hasattr(r, 'field_type') or r.field_type is None:
                # Check if this is an NERCandidate with epidemiology data
                if hasattr(r, 'epidemiology_data') and r.epidemiology_data is not None:
                    epi_entry = FeasibilityExportEntry(
                        field_type="EPIDEMIOLOGY_NER",
                        text=getattr(r, 'text', ''),
                        section="epidemiology",
                        page=None,
                        structured_data={
                            "data_type": r.epidemiology_data.data_type,
                            "value": r.epidemiology_data.value,
                            "population": r.epidemiology_data.population,
                            "source": getattr(r, 'source', 'NER'),
                        },
                        confidence=getattr(r, 'confidence', 0.8),
                        evidence=[],
                    )
                    epidemiology.append(epi_entry)
                continue

            # Handle study design separately (single object, not list)
            if r.field_type.value == "STUDY_DESIGN" and r.study_design:
                study_design_data = r.study_design.model_dump()
                continue

            # Handle operational burden (single object)
            if r.field_type.value == "OPERATIONAL_BURDEN" and r.operational_burden:
                operational_burden_data = r.operational_burden.model_dump()
                continue

            # Handle screening flow (single object)
            if r.field_type.value == "SCREENING_FLOW" and r.screening_flow:
                screening_flow_data = r.screening_flow.model_dump()
                continue

            # Convert evidence from the candidate
            evidence_export = _convert_evidence(r.evidence)

            # Propagate page from evidence if not set on parent
            page_num = r.page_number
            if page_num is None and evidence_export:
                page_num = evidence_export[0].page

            entry = FeasibilityExportEntry(
                field_type=r.field_type.value,
                text=r.matched_text,
                section=r.section_name,
                page=page_num,
                structured_data=None,
                confidence=r.confidence,
                evidence=evidence_export,
            )

            # Add structured data based on type, including evidence from nested objects
            if r.eligibility_criterion:
                entry.structured_data = {
                    "type": r.eligibility_criterion.criterion_type.value,
                    "category": r.eligibility_criterion.category,
                    "derived": r.eligibility_criterion.derived_variables,
                }
                # Add evidence from eligibility criterion if not already present
                if not entry.evidence and r.eligibility_criterion.evidence:
                    entry.evidence = _convert_evidence(r.eligibility_criterion.evidence)
                    # Also propagate page from this evidence
                    if entry.page is None and entry.evidence:
                        entry.page = entry.evidence[0].page
            elif r.epidemiology_data:
                entry.structured_data = {
                    "data_type": r.epidemiology_data.data_type,
                    "value": r.epidemiology_data.value,
                    "population": r.epidemiology_data.population,
                }
            elif r.patient_journey_phase:
                entry.structured_data = {
                    "phase": r.patient_journey_phase.phase_type.value,
                    "duration": r.patient_journey_phase.duration,
                }
            elif r.study_endpoint:
                entry.structured_data = {
                    "type": r.study_endpoint.endpoint_type.value,
                    "name": r.study_endpoint.name,
                    "measure": r.study_endpoint.measure,
                    "timepoint": r.study_endpoint.timepoint,
                    "analysis_method": r.study_endpoint.analysis_method,
                }
            elif r.study_site:
                entry.structured_data = {
                    "country": r.study_site.country,
                    "site_count": r.study_site.site_count,
                }

            # Categorize
            if "ELIGIBILITY_INCLUSION" in r.field_type.value:
                eligibility_inclusion.append(entry)
            elif "ELIGIBILITY_EXCLUSION" in r.field_type.value:
                eligibility_exclusion.append(entry)
            elif "EPIDEMIOLOGY" in r.field_type.value:
                epidemiology.append(entry)
            elif "PATIENT_JOURNEY" in r.field_type.value:
                patient_journey.append(entry)
            elif "ENDPOINT" in r.field_type.value:
                endpoints.append(entry)
            elif "SITE" in r.field_type.value:
                sites.append(entry)

        # Build export document
        export_doc = FeasibilityExportDocument(
            doc_id=pdf_path.stem,
            doc_filename=pdf_path.name,
            trial_ids=trial_ids,
            study_design=study_design_data,
            eligibility_inclusion=eligibility_inclusion,
            eligibility_exclusion=eligibility_exclusion,
            epidemiology=epidemiology,
            patient_journey=patient_journey,
            endpoints=endpoints,
            sites=sites,
            operational_burden=operational_burden_data,
            screening_flow=screening_flow_data,
        )

        # Write to file
        out_file = out_dir / f"feasibility_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(export_doc.model_dump_json(indent=2))

        print(f"  Feasibility export: {out_file.name}")

    def export_images(
        self, pdf_path: Path, doc: "DocumentGraph"
    ) -> None:
        """Export extracted images to JSON file with Vision LLM analysis."""
        from B_parsing.B02_doc_graph import ImageType
        from C_generators.C10_vision_image_analysis import VisionImageAnalyzer

        # Collect all images
        images = list(doc.iter_images())
        if not images:
            return

        out_dir = self.get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize Vision analyzer if LLM client available
        vision_analyzer = None
        if self.claude_client:
            vision_analyzer = VisionImageAnalyzer(self.claude_client)

        # Error tracking for processing issues
        processing_errors: List[Dict[str, Any]] = []
        analysis_stats = {
            "total": len(images),
            "successful": 0,
            "failed": 0,
            "compressed": 0,
            "skipped_oversized": 0,
            "vision_analyzed": 0,
            "ocr_extracted": 0,
            "ocr_failed": 0,
        }

        # Build export data
        export_data: Dict[str, Any] = {
            "doc_id": pdf_path.stem,
            "doc_filename": pdf_path.name,
            "total_images": len(images),
            "images": [],
            "processing_errors": processing_errors,
            "analysis_summary": analysis_stats,
        }

        vision_call_count = 0
        for img in images:
            # Check extraction source from metadata (B09-B11 native extraction)
            extraction_source = img.metadata.get("source") if img.metadata else None
            figure_type = img.metadata.get("figure_type") if img.metadata else None

            # Extract OCR text if not already present
            ocr_text = img.ocr_text
            ocr_method = None
            if not ocr_text and img.image_base64 and PYTESSERACT_AVAILABLE:
                try:
                    ocr_text, ocr_info = extract_ocr_text_from_base64(
                        img.image_base64,
                        lang="eng",
                        config="--psm 6"  # Assume uniform block of text
                    )
                    if ocr_info.get("success") and ocr_text:
                        ocr_method = ocr_info.get("method")
                        analysis_stats["ocr_extracted"] += 1
                        # Truncate very long OCR text to avoid bloating JSON
                        if len(ocr_text) > 5000:
                            ocr_text = ocr_text[:5000] + "...[truncated]"
                    else:
                        analysis_stats["ocr_failed"] += 1
                except Exception as e:
                    analysis_stats["ocr_failed"] += 1
                    print(f"    [DEBUG] OCR extraction failed for page {img.page_num}: {e}")

            img_data: Dict[str, Any] = {
                "page": img.page_num,
                "type": img.image_type.value,
                "caption": img.caption,
                "ocr_text": ocr_text,
                "ocr_method": ocr_method,
                "bbox": list(img.bbox.coords) if img.bbox else None,
                "image_base64": img.image_base64,
                "extraction_source": extraction_source,
                "figure_type": figure_type,
            }

            # Check image size before Vision analysis
            image_size = get_image_size_bytes(img.image_base64) if img.image_base64 else 0
            img_data["image_size_bytes"] = image_size

            # Run Vision LLM analysis based on image type
            if vision_analyzer and img.image_base64:
                # Check if image is oversized
                if is_image_oversized(img.image_base64):
                    # Will be auto-compressed by the Vision client
                    img_data["was_compressed"] = True
                    analysis_stats["compressed"] += 1

                # Add delay between Vision LLM calls to avoid rate limiting
                if vision_call_count > 0:
                    time.sleep(0.1)  # 100ms delay
                if img.image_type == ImageType.FLOWCHART:
                    try:
                        vision_call_count += 1
                        flow_result = vision_analyzer.analyze_flowchart(
                            img.image_base64, img.ocr_text
                        )
                        if flow_result:
                            img_data["vision_analysis"] = {
                                "analysis_type": "patient_flow",
                                "screened": flow_result.screened,
                                "screen_failures": flow_result.screen_failures,
                                "randomized": flow_result.randomized,
                                "completed": flow_result.completed,
                                "discontinued": flow_result.discontinued,
                                "arms": flow_result.arms,
                                "exclusion_reasons": [
                                    {"reason": e.reason, "count": e.count}
                                    for e in flow_result.exclusion_reasons
                                ],
                                "stages": [
                                    {"stage_name": s.stage_name, "count": s.count, "details": s.details}
                                    for s in flow_result.stages
                                ],
                                "notes": flow_result.notes,
                            }
                            analysis_stats["vision_analyzed"] += 1
                        else:
                            analysis_stats["failed"] += 1
                            processing_errors.append({
                                "type": "VISION_API_NO_RESULT",
                                "page": img.page_num,
                                "image_type": "flowchart",
                                "action": "ocr_fallback",
                            })
                    except Exception as e:
                        error_msg = str(e)
                        analysis_stats["failed"] += 1
                        error_type = "VISION_API_ERROR"
                        if "5 MB" in error_msg or "size" in error_msg.lower():
                            error_type = "SIZE_EXCEEDED"
                            analysis_stats["skipped_oversized"] += 1
                        processing_errors.append({
                            "type": error_type,
                            "page": img.page_num,
                            "image_type": "flowchart",
                            "error": error_msg[:200],
                            "action": "ocr_fallback",
                        })
                        print(f"    [WARN] Flowchart analysis failed: {e}")

                elif img.image_type == ImageType.CHART:
                    try:
                        vision_call_count += 1
                        chart_result = vision_analyzer.analyze_chart(
                            img.image_base64, img.caption
                        )
                        if chart_result:
                            vision_data: Dict[str, Any] = {
                                "analysis_type": "chart_data",
                                "chart_type": chart_result.chart_type,
                                "title": chart_result.title,
                                "x_axis": chart_result.x_axis,
                                "y_axis": chart_result.y_axis,
                                "legend_entries": [
                                    {
                                        "color": le.color,
                                        "marker": le.marker,
                                        "label": le.label,
                                        "series_id": le.series_id,
                                    }
                                    for le in chart_result.legend_entries
                                ] if chart_result.legend_entries else [],
                                "data_points": [
                                    {
                                        "label": dp.label,
                                        "value": dp.value,
                                        "unit": dp.unit,
                                        "group": dp.group,
                                    }
                                    for dp in chart_result.data_points
                                ],
                                "statistical_results": chart_result.statistical_results,
                            }
                            # Add taper schedule if present
                            if chart_result.taper_schedule:
                                vision_data["taper_schedule"] = {
                                    "drug_name": chart_result.taper_schedule.drug_name,
                                    "starting_dose": chart_result.taper_schedule.starting_dose,
                                    "target_dose": chart_result.taper_schedule.target_dose,
                                    "target_timepoint": chart_result.taper_schedule.target_timepoint,
                                }
                            img_data["vision_analysis"] = vision_data
                            analysis_stats["vision_analyzed"] += 1
                        else:
                            analysis_stats["failed"] += 1
                            processing_errors.append({
                                "type": "VISION_API_NO_RESULT",
                                "page": img.page_num,
                                "image_type": "chart",
                                "action": "ocr_fallback",
                            })
                    except Exception as e:
                        error_msg = str(e)
                        analysis_stats["failed"] += 1
                        error_type = "VISION_API_ERROR"
                        if "5 MB" in error_msg or "size" in error_msg.lower():
                            error_type = "SIZE_EXCEEDED"
                            analysis_stats["skipped_oversized"] += 1
                        processing_errors.append({
                            "type": error_type,
                            "page": img.page_num,
                            "image_type": "chart",
                            "error": error_msg[:200],
                            "action": "ocr_fallback",
                        })
                        print(f"    [WARN] Chart analysis failed: {e}")

                # Handle UNKNOWN type images - classify and analyze
                elif img.image_type == ImageType.UNKNOWN:
                    try:
                        vision_call_count += 1
                        detected_type, analysis_result = vision_analyzer.analyze_unknown_image(
                            img.image_base64, ocr_text, img.caption
                        )
                        # Update the image type based on classification
                        img_data["detected_type"] = detected_type

                        if detected_type == "flowchart" and analysis_result:
                            img_data["type"] = "FLOWCHART"  # Override UNKNOWN
                            img_data["vision_analysis"] = {
                                "analysis_type": "patient_flow",
                                "screened": analysis_result.screened,
                                "screen_failures": analysis_result.screen_failures,
                                "randomized": analysis_result.randomized,
                                "completed": analysis_result.completed,
                                "discontinued": analysis_result.discontinued,
                                "arms": analysis_result.arms,
                                "exclusion_reasons": [
                                    {"reason": e.reason, "count": e.count}
                                    for e in analysis_result.exclusion_reasons
                                ],
                                "stages": [
                                    {"stage_name": s.stage_name, "count": s.count, "details": s.details}
                                    for s in analysis_result.stages
                                ],
                                "notes": analysis_result.notes,
                            }
                            analysis_stats["vision_analyzed"] += 1

                        elif detected_type == "chart" and analysis_result:
                            img_data["type"] = "CHART"  # Override UNKNOWN
                            img_data["vision_analysis"] = {
                                "analysis_type": "chart_data",
                                "chart_type": analysis_result.chart_type,
                                "title": analysis_result.title,
                                "x_axis": analysis_result.x_axis,
                                "y_axis": analysis_result.y_axis,
                                "data_points": [
                                    {
                                        "label": dp.label,
                                        "value": dp.value,
                                        "unit": dp.unit,
                                        "group": dp.group,
                                    }
                                    for dp in analysis_result.data_points
                                ],
                                "statistical_results": analysis_result.statistical_results,
                            }
                            analysis_stats["vision_analyzed"] += 1

                        elif detected_type == "table" and analysis_result:
                            img_data["type"] = "TABLE"  # Override UNKNOWN
                            img_data["vision_analysis"] = {
                                "analysis_type": "table_data",
                                "title": analysis_result.title,
                                "headers": analysis_result.headers,
                                "rows": analysis_result.rows,
                                "table_type": analysis_result.table_type,
                                "notes": analysis_result.notes,
                            }
                            analysis_stats["vision_analyzed"] += 1

                        elif detected_type in ("text_page", "logo"):
                            # Mark as skipped - false positive figure extraction
                            img_data["type"] = detected_type.upper()
                            img_data["skipped_reason"] = "false_positive_figure"

                        else:
                            # Classification didn't yield analyzable content
                            analysis_stats["failed"] += 1

                    except Exception as e:
                        error_msg = str(e)
                        analysis_stats["failed"] += 1
                        error_type = "VISION_API_ERROR"
                        if "5 MB" in error_msg or "size" in error_msg.lower():
                            error_type = "SIZE_EXCEEDED"
                            analysis_stats["skipped_oversized"] += 1
                        processing_errors.append({
                            "type": error_type,
                            "page": img.page_num,
                            "image_type": "unknown",
                            "error": error_msg[:200],
                            "action": "ocr_fallback",
                        })
                        print(f"    [WARN] Unknown image analysis failed: {e}")

            # Save image as file - use original or re-render based on source
            if img.image_base64:
                # Use updated type from img_data (may have been classified from UNKNOWN)
                img_type = img_data.get("type", img.image_type.value if img.image_type else "image").lower()
                img_index = len([i for i in export_data['images'] if i.get('page') == img.page_num]) + 1
                img_filename = f"{pdf_path.stem}_{img_type}_page{img.page_num}_{img_index}.png"
                img_path = out_dir / img_filename

                try:
                    # Determine rendering strategy based on extraction source and figure type
                    # - Raster figures (native xref): use original image_base64 (already exact)
                    # - Vector figures: re-render region with minimal padding
                    # - Layout model: re-render with generous padding
                    rendered_base64 = None

                    if figure_type == "raster" and extraction_source in ("caption_linked", "orphan_native"):
                        # For raster figures, use the original image directly
                        # The xref rendering already gives us the exact figure
                        # No need to re-render from page (which would include adjacent content)
                        rendered_base64 = None  # Will use img.image_base64 below

                    elif figure_type == "vector" and extraction_source in ("caption_linked", "orphan_native"):
                        # Vector figures need re-rendering with caption padding
                        if img.bbox:
                            rendered_base64 = self.render_figure_with_padding(
                                pdf_path=pdf_path,
                                page_num=img.page_num,
                                bbox=img.bbox.coords,
                                padding=20,           # Minimal left padding
                                top_padding=20,       # Minimal top padding
                                bottom_padding=150,   # Include caption below
                                right_padding=30,     # Minimal right padding
                            )

                    elif img.bbox:
                        # Layout model or unknown source: moderate padding
                        # Reduced from 350/200 to avoid capturing adjacent content
                        rendered_base64 = self.render_figure_with_padding(
                            pdf_path=pdf_path,
                            page_num=img.page_num,
                            bbox=img.bbox.coords,
                            padding=40,           # ~0.5 inch left
                            top_padding=50,       # ~0.7 inch above
                            bottom_padding=120,   # ~1.7 inch for captions
                            right_padding=60,     # ~0.8 inch right
                        )

                    # Use re-rendered image if available, otherwise use original
                    if rendered_base64:
                        img_bytes = base64.b64decode(rendered_base64)
                        img_data["image_base64"] = rendered_base64  # Update for analysis
                    else:
                        img_bytes = base64.b64decode(img.image_base64)

                    with open(img_path, "wb") as img_file:
                        img_file.write(img_bytes)
                    img_data["saved_file"] = img_filename
                except Exception as e:
                    print(f"    [WARN] Failed to save image {img_filename}: {e}")

            export_data["images"].append(img_data)

        # Update successful count (total - failed)
        analysis_stats["successful"] = analysis_stats["total"] - analysis_stats["failed"]

        # Write JSON metadata
        out_file = out_dir / f"figures_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        # Count by source
        source_counts = {}
        for img in export_data["images"]:
            src = img.get("extraction_source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

        saved_count = sum(1 for img in export_data["images"] if "saved_file" in img)
        analyzed_count = sum(1 for img in export_data["images"] if "vision_analysis" in img)
        source_str = ", ".join(f"{k}:{v}" for k, v in source_counts.items()) if source_counts else ""

        # Include error summary in output
        error_summary = ""
        if processing_errors:
            error_summary = f", errors: {len(processing_errors)}"
        if analysis_stats["compressed"] > 0:
            error_summary += f", compressed: {analysis_stats['compressed']}"

        print(f"  Images export: {out_file.name} ({len(images)} images, {saved_count} saved, {analyzed_count} analyzed, sources: {source_str}{error_summary})")

    def export_tables(
        self, pdf_path: Path, doc: "DocumentGraph"
    ) -> None:
        """Export extracted tables as images to JSON file."""
        # Collect all tables with images
        tables = [t for t in doc.iter_tables() if t.image_base64]
        if not tables:
            return

        out_dir = self.get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build export data
        export_data: Dict[str, Any] = {
            "doc_id": pdf_path.stem,
            "doc_filename": pdf_path.name,
            "total_tables": len(tables),
            "tables": []
        }

        for idx, table in enumerate(tables):
            # Determine page number(s)
            if table.is_multipage and table.page_nums:
                page_str = f"pages{table.page_nums[0]}-{table.page_nums[-1]}"
                page_num = table.page_nums[0]
            else:
                page_str = f"page{table.page_num}"
                page_num = table.page_num

            # Calculate row/col counts from cells or logical_rows
            num_rows = len(table.logical_rows) if table.logical_rows else 0
            num_cols = len(table.logical_rows[0]) if table.logical_rows else 0

            # Extract headers from metadata if available
            headers = []
            if table.metadata and "headers" in table.metadata:
                headers_map = table.metadata["headers"]
                if isinstance(headers_map, dict):
                    headers = [headers_map.get(i, f"col_{i}") for i in range(num_cols)]

            table_data = {
                "page": page_num,
                "page_nums": table.page_nums if table.is_multipage else [table.page_num],
                "is_multipage": table.is_multipage,
                "type": table.table_type.value if table.table_type else "UNKNOWN",
                "caption": table.caption,
                "rows": num_rows,
                "cols": num_cols,
                "headers": headers,
                "data": table.logical_rows if table.logical_rows else [],
                "bbox": list(table.bbox.coords) if table.bbox else None,
                "image_base64": table.image_base64,
            }

            # Save table image as file - re-render with padding if bbox available
            if table.image_base64:
                table_type = table.table_type.value.lower() if table.table_type else "table"
                img_filename = f"{pdf_path.stem}_table_{table_type}_{page_str}_{idx + 1}.png"
                img_path = out_dir / img_filename
                try:
                    # Try to re-render with generous padding
                    rendered_base64 = None
                    if table.bbox and not table.is_multipage:
                        rendered_base64 = self.render_figure_with_padding(
                            pdf_path=pdf_path,
                            page_num=table.page_num,
                            bbox=table.bbox.coords,
                            padding=20,          # Minimal left (overridden to 15 for tables)
                            top_padding=80,      # ~1.1 inch for table title
                            bottom_padding=60,   # ~0.8 inch for footnotes
                            right_padding=20,    # Minimal right (overridden to 15 for tables)
                            is_table=True,       # Use PyMuPDF table detection for accurate bbox
                        )

                    # Use re-rendered image if successful, otherwise use original
                    if rendered_base64:
                        img_bytes = base64.b64decode(rendered_base64)
                        table_data["image_base64"] = rendered_base64
                    else:
                        img_bytes = base64.b64decode(table.image_base64)

                    with open(img_path, "wb") as img_file:
                        img_file.write(img_bytes)
                    table_data["saved_file"] = img_filename
                except Exception as e:
                    print(f"    [WARN] Failed to save table image {img_filename}: {e}")

            export_data["tables"].append(table_data)

        # Write JSON metadata
        out_file = out_dir / f"tables_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        saved_count = sum(1 for t in export_data["tables"] if "saved_file" in t)
        print(f"  Tables export: {out_file.name} ({len(tables)} tables, {saved_count} saved)")

    def export_document_metadata(
        self, pdf_path: Path, metadata: "DocumentMetadata"
    ) -> None:
        """Export document metadata to JSON file."""
        out_dir = self.get_output_dir(pdf_path)
        _export_document_metadata(
            out_dir, pdf_path, metadata, self.run_id, self.pipeline_version
        )

    def export_care_pathways(
        self, pdf_path: Path, pathways: List["CarePathway"]
    ) -> None:
        """Export care pathway graphs extracted from treatment algorithm figures."""
        out_dir = self.get_output_dir(pdf_path)
        _export_care_pathways(
            out_dir, pdf_path, pathways, self.run_id, self.pipeline_version
        )

    def export_recommendations(
        self, pdf_path: Path, recommendation_sets: List["RecommendationSet"]
    ) -> None:
        """Export guideline recommendations extracted from text and tables."""
        out_dir = self.get_output_dir(pdf_path)
        _export_recommendations(
            out_dir, pdf_path, recommendation_sets, self.run_id, self.pipeline_version
        )


__all__ = ["ExportManager"]
