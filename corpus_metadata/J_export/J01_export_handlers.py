# corpus_metadata/J_export/J01_export_handlers.py
"""
Export handlers for pipeline results.

Handles exporting extraction results to various JSON formats including
abbreviations, diseases, genes, drugs, pharma companies, authors, citations,
feasibility data, images, tables, and document metadata.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from B_parsing.B02_doc_graph import DocumentGraph
    from A_core.A01_domain_models import Candidate, ExtractedEntity
    from A_core.A04_heuristics_config import HeuristicsCounters
    from A_core.A05_disease_models import ExtractedDisease, DiseaseExportEntry
    from A_core.A06_drug_models import ExtractedDrug, DrugExportEntry
    from A_core.A12_gene_models import ExtractedGene, GeneExportEntry
    from A_core.A09_pharma_models import ExtractedPharma, PharmaExportEntry
    from A_core.A10_author_models import ExtractedAuthor, AuthorExportEntry
    from A_core.A11_citation_models import ExtractedCitation, CitationExportEntry
    from A_core.A07_feasibility_models import FeasibilityCandidate
    from A_core.A08_document_metadata_models import DocumentMetadata
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
        padding: int = 50,
        bottom_padding: int = 300,
        right_padding: int = 200,
        top_padding: Optional[int] = None,
    ) -> Optional[str]:
        """
        Re-render a figure from PDF with extra padding for captions/legends.

        Args:
            pdf_path: Path to PDF file
            page_num: 1-indexed page number
            bbox: (x0, y0, x1, y1) bounding box in PDF points
            dpi: Resolution for rendering
            padding: Extra points around left side (default 50pt)
            bottom_padding: Extra points below figure for captions (default 300pt)
            right_padding: Extra points to right for multi-panel figures (default 200pt)
            top_padding: Extra points above figure (default same as padding)

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
            if x1 > page_width * 1.1 or y1 > page_height * 1.1:
                # Scale from pixel space to PDF point space
                max_coord = max(x1, y1)
                max_page = max(page_width, page_height)
                dpi_ratio = max_page / max_coord

                x0 = x0 * dpi_ratio
                y0 = y0 * dpi_ratio
                x1 = x1 * dpi_ratio
                y1 = y1 * dpi_ratio

            # For figures that span a significant portion of page, expand to capture full width
            figure_width = x1 - x0
            figure_height = y1 - y0
            is_significant_figure = (
                figure_width > page_width * 0.15 or
                figure_height > page_height * 0.20
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
        from F_evaluation.F05_extraction_analysis import run_analysis

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
        if self.gold_json:
            run_analysis(export_data, self.gold_json)

    def export_disease_results(
        self, pdf_path: Path, results: List["ExtractedDisease"]
    ) -> None:
        """Export disease detection results to separate JSON file."""
        from A_core.A01_domain_models import ValidationStatus
        from A_core.A05_disease_models import DiseaseExportDocument, DiseaseExportEntry

        out_dir = self.get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        validated = [r for r in results if r.status == ValidationStatus.VALIDATED]
        rejected = [r for r in results if r.status == ValidationStatus.REJECTED]
        ambiguous = [r for r in results if r.status == ValidationStatus.AMBIGUOUS]

        # Build export entries
        disease_entries: List[DiseaseExportEntry] = []
        for entity in validated:
            codes = {
                "icd10": entity.icd10_code,
                "icd11": entity.icd11_code,
                "snomed": entity.snomed_code,
                "mondo": entity.mondo_id,
                "orpha": entity.orpha_code,
                "umls": entity.umls_cui,
                "mesh": entity.mesh_id,
            }

            all_identifiers = [
                {"system": i.system, "code": i.code, "display": i.display}
                for i in entity.identifiers
            ]

            entry = DiseaseExportEntry(
                matched_text=entity.matched_text,
                preferred_label=entity.preferred_label,
                abbreviation=entity.abbreviation,
                confidence=entity.confidence_score,
                is_rare_disease=entity.is_rare_disease,
                category=entity.disease_category,
                codes=codes,
                all_identifiers=all_identifiers,
                context=entity.primary_evidence.text
                if entity.primary_evidence
                else None,
                page=entity.primary_evidence.location.page_num
                if entity.primary_evidence
                else None,
                lexicon_source=entity.provenance.lexicon_source
                if entity.provenance
                else None,
                validation_flags=entity.validation_flags,
                mesh_aliases=entity.mesh_aliases,
                pubtator_normalized_name=entity.pubtator_normalized_name,
                enrichment_source=entity.enrichment_source,
            )
            disease_entries.append(entry)

        # Build export document
        export_doc = DiseaseExportDocument(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            document=pdf_path.name,
            document_path=str(pdf_path.absolute()),
            pipeline_version=self.pipeline_version,
            total_candidates=len(results),
            total_validated=len(validated),
            total_rejected=len(rejected),
            total_ambiguous=len(ambiguous),
            diseases=disease_entries,
        )

        # Write to file
        out_file = out_dir / f"diseases_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(export_doc.model_dump_json(indent=2))

        print(f"  Disease export: {out_file.name}")

    def export_gene_results(
        self, pdf_path: Path, results: List["ExtractedGene"]
    ) -> None:
        """Export gene detection results to separate JSON file."""
        from A_core.A01_domain_models import ValidationStatus
        from A_core.A12_gene_models import GeneExportDocument, GeneExportEntry

        out_dir = self.get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        validated = [r for r in results if r.status == ValidationStatus.VALIDATED]
        rejected = [r for r in results if r.status == ValidationStatus.REJECTED]

        # Build export entries
        gene_entries: List[GeneExportEntry] = []
        for entity in validated:
            codes = {
                "hgnc_id": entity.hgnc_id,
                "entrez": entity.entrez_id,
                "ensembl": entity.ensembl_id,
                "omim": entity.omim_id,
                "uniprot": entity.uniprot_id,
            }

            all_identifiers = [
                {"system": i.system, "code": i.code, "display": i.display}
                for i in entity.identifiers
            ]

            # Simplified disease associations
            disease_assocs = [
                {
                    "orphacode": d.orphacode,
                    "name": d.disease_name,
                    "association_type": d.association_type or "",
                }
                for d in entity.associated_diseases
            ]

            entry = GeneExportEntry(
                matched_text=entity.matched_text,
                hgnc_symbol=entity.hgnc_symbol,
                full_name=entity.full_name,
                confidence=entity.confidence_score,
                is_alias=entity.is_alias,
                locus_type=entity.locus_type,
                chromosome=entity.chromosome,
                codes=codes,
                all_identifiers=all_identifiers,
                associated_diseases=disease_assocs,
                context=entity.primary_evidence.text
                if entity.primary_evidence
                else None,
                page=entity.primary_evidence.location.page_num
                if entity.primary_evidence
                else None,
                lexicon_source=entity.provenance.lexicon_source
                if entity.provenance
                else None,
                validation_flags=entity.validation_flags,
            )
            gene_entries.append(entry)

        # Build export document
        export_doc = GeneExportDocument(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            document=pdf_path.name,
            document_path=str(pdf_path.absolute()),
            pipeline_version=self.pipeline_version,
            total_candidates=len(results),
            total_validated=len(validated),
            total_rejected=len(rejected),
            genes=gene_entries,
        )

        # Write to file
        out_file = out_dir / f"genes_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(export_doc.model_dump_json(indent=2))

        print(f"  Gene export: {out_file.name}")

    def export_drug_results(
        self, pdf_path: Path, results: List["ExtractedDrug"]
    ) -> None:
        """Export drug detection results to separate JSON file."""
        from A_core.A01_domain_models import ValidationStatus
        from A_core.A06_drug_models import DrugExportDocument, DrugExportEntry

        out_dir = self.get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        validated = [r for r in results if r.status == ValidationStatus.VALIDATED]
        rejected = [r for r in results if r.status == ValidationStatus.REJECTED]
        investigational = [r for r in validated if r.is_investigational]

        # Build export entries
        drug_entries: List[DrugExportEntry] = []
        for entity in validated:
            codes = {
                "rxcui": entity.rxcui,
                "mesh": entity.mesh_id,
                "ndc": entity.ndc_code,
                "drugbank": entity.drugbank_id,
                "unii": entity.unii,
            }

            all_identifiers = [
                {"system": i.system, "code": i.code, "display": i.display}
                for i in entity.identifiers
            ]

            entry = DrugExportEntry(
                matched_text=entity.matched_text,
                preferred_name=entity.preferred_name,
                brand_name=entity.brand_name,
                compound_id=entity.compound_id,
                confidence=entity.confidence_score,
                is_investigational=entity.is_investigational,
                drug_class=entity.drug_class,
                mechanism=entity.mechanism,
                development_phase=entity.development_phase,
                sponsor=entity.sponsor,
                conditions=entity.conditions,
                nct_id=entity.nct_id,
                dosage_form=entity.dosage_form,
                route=entity.route,
                marketing_status=entity.marketing_status,
                codes=codes,
                all_identifiers=all_identifiers,
                context=entity.primary_evidence.text
                if entity.primary_evidence
                else None,
                page=entity.primary_evidence.location.page_num
                if entity.primary_evidence
                else None,
                lexicon_source=entity.provenance.lexicon_source
                if entity.provenance
                else None,
                validation_flags=entity.validation_flags,
                mesh_aliases=entity.mesh_aliases,
                pubtator_normalized_name=entity.pubtator_normalized_name,
                enrichment_source=entity.enrichment_source,
            )
            drug_entries.append(entry)

        # Build export document
        export_doc = DrugExportDocument(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            document=pdf_path.name,
            document_path=str(pdf_path.absolute()),
            pipeline_version=self.pipeline_version,
            total_candidates=len(results),
            total_validated=len(validated),
            total_rejected=len(rejected),
            total_investigational=len(investigational),
            drugs=drug_entries,
        )

        # Write to file
        out_file = out_dir / f"drugs_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(export_doc.model_dump_json(indent=2))

        print(f"  Drug export: {out_file.name}")

    def export_pharma_results(
        self, pdf_path: Path, results: List["ExtractedPharma"]
    ) -> None:
        """Export pharma company detection results to separate JSON file."""
        from A_core.A01_domain_models import ValidationStatus
        from A_core.A09_pharma_models import PharmaExportDocument, PharmaExportEntry

        out_dir = self.get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        validated = [r for r in results if r.status == ValidationStatus.VALIDATED]

        # Build export entries
        pharma_entries: List[PharmaExportEntry] = []
        for entity in validated:
            entry = PharmaExportEntry(
                matched_text=entity.matched_text,
                canonical_name=entity.canonical_name,
                full_name=entity.full_name,
                headquarters=entity.headquarters,
                parent_company=entity.parent_company,
                subsidiaries=entity.subsidiaries,
                confidence=entity.confidence_score,
                context=entity.primary_evidence.text
                if entity.primary_evidence
                else None,
                page=entity.primary_evidence.location.page_num
                if entity.primary_evidence
                else None,
                lexicon_source=entity.provenance.lexicon_source
                if entity.provenance
                else None,
            )
            pharma_entries.append(entry)

        # Build export document
        unique_companies = set(e.canonical_name for e in pharma_entries)
        export_doc = PharmaExportDocument(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            document=pdf_path.name,
            document_path=str(pdf_path.absolute()),
            pipeline_version=self.pipeline_version,
            total_detected=len(results),
            unique_companies=len(unique_companies),
            companies=pharma_entries,
        )

        # Write to file
        out_file = out_dir / f"pharma_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(export_doc.model_dump_json(indent=2))

        print(f"  Pharma export: {out_file.name}")

    def export_author_results(
        self, pdf_path: Path, results: List["ExtractedAuthor"]
    ) -> None:
        """Export author detection results to separate JSON file."""
        from A_core.A01_domain_models import ValidationStatus
        from A_core.A10_author_models import AuthorExportDocument, AuthorExportEntry

        out_dir = self.get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        validated = [r for r in results if r.status == ValidationStatus.VALIDATED]

        # Build export entries
        author_entries: List[AuthorExportEntry] = []
        unique_names: set = set()
        for entity in validated:
            unique_names.add(entity.full_name.lower())
            entry = AuthorExportEntry(
                full_name=entity.full_name,
                role=entity.role.value,
                affiliation=entity.affiliation,
                email=entity.email,
                orcid=entity.orcid,
                confidence=entity.confidence_score,
                context=entity.primary_evidence.text
                if entity.primary_evidence
                else None,
                page=entity.primary_evidence.location.page_num
                if entity.primary_evidence
                else None,
            )
            author_entries.append(entry)

        # Build export document
        export_doc = AuthorExportDocument(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            document=pdf_path.name,
            document_path=str(pdf_path.absolute()),
            pipeline_version=self.pipeline_version,
            total_detected=len(results),
            unique_authors=len(unique_names),
            authors=author_entries,
        )

        # Write to file
        out_file = out_dir / f"authors_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(export_doc.model_dump_json(indent=2))

        print(f"  Author export: {out_file.name}")

    def export_citation_results(
        self, pdf_path: Path, results: List["ExtractedCitation"]
    ) -> None:
        """Export citation detection results to separate JSON file with API validation."""
        from A_core.A01_domain_models import ValidationStatus
        from A_core.A11_citation_models import (
            CitationExportDocument,
            CitationExportEntry,
            CitationValidation,
            CitationValidationSummary,
        )
        from E_normalization.E14_citation_validator import CitationValidator

        out_dir = self.get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        validated = [r for r in results if r.status == ValidationStatus.VALIDATED]

        # Build export entries
        citation_entries: List[CitationExportEntry] = []
        unique_ids: set = set()
        for entity in validated:
            # Track unique identifiers
            if entity.pmid:
                unique_ids.add(f"pmid:{entity.pmid}")
            if entity.pmcid:
                unique_ids.add(f"pmcid:{entity.pmcid}")
            if entity.doi:
                unique_ids.add(f"doi:{entity.doi}")
            if entity.nct:
                unique_ids.add(f"nct:{entity.nct}")

            entry = CitationExportEntry(
                pmid=entity.pmid,
                pmcid=entity.pmcid,
                doi=entity.doi,
                nct=entity.nct,
                url=entity.url,
                citation_text=entity.citation_text,
                citation_number=entity.citation_number,
                confidence=entity.confidence_score,
                page=entity.primary_evidence.location.page_num
                if entity.primary_evidence
                else None,
            )
            citation_entries.append(entry)

        # Run API validation on citations
        validation_summary = None
        if citation_entries:
            print("  Validating citations via API...")
            validator = CitationValidator({"validate_urls": False})  # Skip URL validation for speed
            valid_count = 0
            invalid_count = 0
            error_count = 0

            for entry in citation_entries:
                # Validate primary identifier (prefer DOI > NCT > PMID)
                validation_result = None

                if entry.doi:
                    result = validator.validate_doi(entry.doi)
                    validation_result = CitationValidation(
                        is_valid=result.is_valid,
                        resolved_url=result.resolved_url,
                        title=result.metadata.get("title"),
                        error=result.error_message,
                    )
                elif entry.nct:
                    result = validator.validate_nct(entry.nct)
                    validation_result = CitationValidation(
                        is_valid=result.is_valid,
                        resolved_url=result.resolved_url,
                        title=result.metadata.get("title"),
                        status=result.metadata.get("status"),
                        error=result.error_message,
                    )
                elif entry.pmid:
                    result = validator.validate_pmid(entry.pmid)
                    validation_result = CitationValidation(
                        is_valid=result.is_valid,
                        resolved_url=result.resolved_url,
                        title=result.metadata.get("title"),
                        error=result.error_message,
                    )

                if validation_result:
                    entry.validation = validation_result
                    if validation_result.is_valid:
                        valid_count += 1
                        print(f"    + {entry.doi or entry.nct or entry.pmid}: valid")
                    elif validation_result.error:
                        error_count += 1
                        print(f"    x {entry.doi or entry.nct or entry.pmid}: {validation_result.error}")
                    else:
                        invalid_count += 1
                        print(f"    x {entry.doi or entry.nct or entry.pmid}: not found")

            validation_summary = CitationValidationSummary(
                total_validated=valid_count + invalid_count + error_count,
                valid_count=valid_count,
                invalid_count=invalid_count,
                error_count=error_count,
            )

        # Build export document
        export_doc = CitationExportDocument(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            document=pdf_path.name,
            document_path=str(pdf_path.absolute()),
            pipeline_version=self.pipeline_version,
            total_detected=len(results),
            unique_identifiers=len(unique_ids),
            validation_summary=validation_summary,
            citations=citation_entries,
        )

        # Write to file
        out_file = out_dir / f"citations_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(export_doc.model_dump_json(indent=2))

        print(f"  Citation export: {out_file.name}")

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

        # Build export data
        export_data: Dict[str, Any] = {
            "doc_id": pdf_path.stem,
            "doc_filename": pdf_path.name,
            "total_images": len(images),
            "images": []
        }

        for img in images:
            # Check extraction source from metadata (B09-B11 native extraction)
            extraction_source = img.metadata.get("source") if img.metadata else None
            figure_type = img.metadata.get("figure_type") if img.metadata else None

            img_data: Dict[str, Any] = {
                "page": img.page_num,
                "type": img.image_type.value,
                "caption": img.caption,
                "ocr_text": img.ocr_text,
                "bbox": list(img.bbox.coords) if img.bbox else None,
                "image_base64": img.image_base64,
                "extraction_source": extraction_source,
                "figure_type": figure_type,
            }

            # Run Vision LLM analysis based on image type
            if vision_analyzer and img.image_base64:
                if img.image_type == ImageType.FLOWCHART:
                    try:
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
                    except Exception as e:
                        print(f"    [WARN] Flowchart analysis failed: {e}")

                elif img.image_type == ImageType.CHART:
                    try:
                        chart_result = vision_analyzer.analyze_chart(
                            img.image_base64, img.caption
                        )
                        if chart_result:
                            img_data["vision_analysis"] = {
                                "analysis_type": "chart_data",
                                "chart_type": chart_result.chart_type,
                                "title": chart_result.title,
                                "x_axis": chart_result.x_axis,
                                "y_axis": chart_result.y_axis,
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
                    except Exception as e:
                        print(f"    [WARN] Chart analysis failed: {e}")

            # Save image as file - re-render with padding if bbox available
            if img.image_base64:
                img_type = img.image_type.value.lower() if img.image_type else "image"
                img_index = len([i for i in export_data['images'] if i.get('page') == img.page_num]) + 1
                img_filename = f"{pdf_path.stem}_{img_type}_page{img.page_num}_{img_index}.png"
                img_path = out_dir / img_filename

                try:
                    # Determine padding based on extraction source
                    # Native extraction (caption_linked, orphan_native) has accurate bboxes
                    # Use minimal padding for these, generous padding for layout_model
                    rendered_base64 = None
                    if img.bbox:
                        if extraction_source in ("caption_linked", "orphan_native"):
                            # Minimal padding for accurate native bboxes
                            # Caption-linked figures: expand below for caption text
                            bottom_pad = 150 if extraction_source == "caption_linked" else 50
                            rendered_base64 = self.render_figure_with_padding(
                                pdf_path=pdf_path,
                                page_num=img.page_num,
                                bbox=img.bbox.coords,
                                padding=20,           # Minimal left padding
                                top_padding=20,       # Minimal top padding
                                bottom_padding=bottom_pad,  # Include caption below
                                right_padding=30,     # Minimal right padding
                            )
                        else:
                            # Generous padding for layout_model or unknown source
                            rendered_base64 = self.render_figure_with_padding(
                                pdf_path=pdf_path,
                                page_num=img.page_num,
                                bbox=img.bbox.coords,
                                padding=75,           # Extra space on left side
                                top_padding=100,      # Extra space above figure
                                bottom_padding=350,   # Extra space for captions/legends
                                right_padding=200,    # Extra space for multi-panel
                            )

                    # Use re-rendered image if successful, otherwise use original
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

        # Write JSON metadata
        out_file = out_dir / f"images_{pdf_path.stem}_{timestamp}.json"
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
        print(f"  Images export: {out_file.name} ({len(images)} images, {saved_count} saved, {analyzed_count} analyzed, sources: {source_str})")

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
                            padding=75,          # Extra space on left side
                            top_padding=150,     # Extra space above table for title
                            bottom_padding=200,  # Extra space below table for notes
                            right_padding=150,   # Extra space on right
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
        from A_core.A08_document_metadata_models import DocumentMetadataExport

        out_dir = self.get_output_dir(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build simplified export
        export = DocumentMetadataExport(
            doc_id=metadata.doc_id,
            doc_filename=metadata.doc_filename,
            file_size_bytes=metadata.file_metadata.size_bytes if metadata.file_metadata else None,
            file_size_human=metadata.file_metadata.size_human if metadata.file_metadata else None,
            file_extension=metadata.file_metadata.extension if metadata.file_metadata else None,
            pdf_title=metadata.pdf_metadata.title if metadata.pdf_metadata else None,
            pdf_author=metadata.pdf_metadata.author if metadata.pdf_metadata else None,
            pdf_page_count=metadata.pdf_metadata.page_count if metadata.pdf_metadata else None,
            pdf_creation_date=(
                metadata.pdf_metadata.creation_date.isoformat()
                if metadata.pdf_metadata and metadata.pdf_metadata.creation_date
                else None
            ),
            document_type_code=(
                metadata.classification.primary_type.code
                if metadata.classification
                else None
            ),
            document_type_name=(
                metadata.classification.primary_type.name
                if metadata.classification
                else None
            ),
            document_type_group=(
                metadata.classification.primary_type.group
                if metadata.classification
                else None
            ),
            classification_confidence=(
                metadata.classification.primary_type.confidence
                if metadata.classification
                else None
            ),
            title=metadata.description.title if metadata.description else None,
            short_description=(
                metadata.description.short_description if metadata.description else None
            ),
            long_description=(
                metadata.description.long_description if metadata.description else None
            ),
            document_date=(
                metadata.date_extraction.primary_date.date.isoformat()
                if metadata.date_extraction and metadata.date_extraction.primary_date
                else None
            ),
            document_date_source=(
                metadata.date_extraction.primary_date.source.value
                if metadata.date_extraction and metadata.date_extraction.primary_date
                else None
            ),
        )

        # Write to file
        out_file = out_dir / f"metadata_{pdf_path.stem}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(export.model_dump_json(indent=2))

        print(f"  Document metadata export: {out_file.name}")


__all__ = ["ExportManager"]
