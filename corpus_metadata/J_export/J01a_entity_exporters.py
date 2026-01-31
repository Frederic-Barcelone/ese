# corpus_metadata/J_export/J01a_entity_exporters.py
"""
Entity export functions for diseases, genes, drugs, pharma, authors, and citations.

Extracted from J01_export_handlers.py to reduce file size.
These functions are called by ExportManager.
"""
from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from A_core.A05_disease_models import ExtractedDisease
    from A_core.A19_gene_models import ExtractedGene
    from A_core.A06_drug_models import ExtractedDrug
    from A_core.A09_pharma_models import ExtractedPharma
    from A_core.A10_author_models import ExtractedAuthor
    from A_core.A11_citation_models import ExtractedCitation


def export_disease_results(
    out_dir: Path,
    pdf_path: Path,
    results: List["ExtractedDisease"],
    run_id: str,
    pipeline_version: str,
) -> None:
    """Export disease detection results to separate JSON file."""
    from A_core.A01_domain_models import ValidationStatus
    from A_core.A05_disease_models import DiseaseExportDocument, DiseaseExportEntry

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
            context=entity.primary_evidence.text if entity.primary_evidence else None,
            page=entity.primary_evidence.location.page_num if entity.primary_evidence else None,
            mention_count=entity.mention_count,
            pages_mentioned=entity.pages_mentioned,
            lexicon_source=entity.provenance.lexicon_source if entity.provenance else None,
            validation_flags=entity.validation_flags,
            mesh_aliases=entity.mesh_aliases,
            pubtator_normalized_name=entity.pubtator_normalized_name,
            enrichment_source=entity.enrichment_source,
        )
        disease_entries.append(entry)

    # Build export document
    export_doc = DiseaseExportDocument(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        document=pdf_path.name,
        document_path=str(pdf_path.absolute()),
        pipeline_version=pipeline_version,
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
    out_dir: Path,
    pdf_path: Path,
    results: List["ExtractedGene"],
    run_id: str,
    pipeline_version: str,
) -> None:
    """Export gene detection results to separate JSON file."""
    from A_core.A01_domain_models import ValidationStatus
    from A_core.A19_gene_models import GeneExportDocument, GeneExportEntry

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
            "uniprot": entity.uniprot_id,
            "omim": entity.omim_id,
        }

        all_identifiers = [
            {"system": i.system, "code": i.code, "display": i.display}
            for i in entity.identifiers
        ]

        # Convert associated diseases from GeneDiseaseLinkage to dict
        diseases_as_dicts = [
            {
                "orphacode": d.orphacode,
                "disease_name": d.disease_name,
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
            associated_diseases=diseases_as_dicts,
            context=entity.primary_evidence.text if entity.primary_evidence else None,
            page=entity.primary_evidence.location.page_num if entity.primary_evidence else None,
            mention_count=entity.mention_count,
            pages_mentioned=entity.pages_mentioned,
            lexicon_source=entity.provenance.lexicon_source if entity.provenance else None,
            validation_flags=entity.validation_flags,
        )
        gene_entries.append(entry)

    # Build export document
    export_doc = GeneExportDocument(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        document=pdf_path.name,
        document_path=str(pdf_path.absolute()),
        pipeline_version=pipeline_version,
        total_candidates=len(results),
        total_validated=len(validated),
        total_rejected=len(rejected),
        genes=gene_entries,
    )

    # Write to file
    out_file = out_dir / f"genes_{pdf_path.stem}_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(export_doc.model_dump_json(indent=2))

    logger.info("Gene export: %s", out_file.name)


def export_drug_results(
    out_dir: Path,
    pdf_path: Path,
    results: List["ExtractedDrug"],
    run_id: str,
    pipeline_version: str,
) -> None:
    """Export drug detection results to separate JSON file."""
    from A_core.A01_domain_models import ValidationStatus
    from A_core.A06_drug_models import DrugExportDocument, DrugExportEntry

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
            context=entity.primary_evidence.text if entity.primary_evidence else None,
            page=entity.primary_evidence.location.page_num if entity.primary_evidence else None,
            mention_count=entity.mention_count,
            pages_mentioned=entity.pages_mentioned,
            lexicon_source=entity.provenance.lexicon_source if entity.provenance else None,
            validation_flags=entity.validation_flags,
            mesh_aliases=entity.mesh_aliases,
            pubtator_normalized_name=entity.pubtator_normalized_name,
            enrichment_source=entity.enrichment_source,
        )
        drug_entries.append(entry)

    # Build export document
    export_doc = DrugExportDocument(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        document=pdf_path.name,
        document_path=str(pdf_path.absolute()),
        pipeline_version=pipeline_version,
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
    out_dir: Path,
    pdf_path: Path,
    results: List["ExtractedPharma"],
    run_id: str,
    pipeline_version: str,
) -> None:
    """Export pharma company detection results to separate JSON file."""
    from A_core.A01_domain_models import ValidationStatus
    from A_core.A09_pharma_models import PharmaExportDocument, PharmaExportEntry

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
            context=entity.primary_evidence.text if entity.primary_evidence else None,
            page=entity.primary_evidence.location.page_num if entity.primary_evidence else None,
            lexicon_source=entity.provenance.lexicon_source if entity.provenance else None,
        )
        pharma_entries.append(entry)

    # Build export document
    unique_companies = set(e.canonical_name for e in pharma_entries)
    export_doc = PharmaExportDocument(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        document=pdf_path.name,
        document_path=str(pdf_path.absolute()),
        pipeline_version=pipeline_version,
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
    out_dir: Path,
    pdf_path: Path,
    results: List["ExtractedAuthor"],
    run_id: str,
    pipeline_version: str,
) -> None:
    """Export author detection results to separate JSON file."""
    from A_core.A01_domain_models import ValidationStatus
    from A_core.A10_author_models import AuthorExportDocument, AuthorExportEntry

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
            context=entity.primary_evidence.text if entity.primary_evidence else None,
            page=entity.primary_evidence.location.page_num if entity.primary_evidence else None,
        )
        author_entries.append(entry)

    # Build export document
    export_doc = AuthorExportDocument(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        document=pdf_path.name,
        document_path=str(pdf_path.absolute()),
        pipeline_version=pipeline_version,
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
    out_dir: Path,
    pdf_path: Path,
    results: List["ExtractedCitation"],
    run_id: str,
    pipeline_version: str,
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
            page=entity.primary_evidence.location.page_num if entity.primary_evidence else None,
        )
        citation_entries.append(entry)

    # Run API validation on citations
    validation_summary = None
    if citation_entries:
        print("  Validating citations via API...")
        validator = CitationValidator({"validate_urls": False})
        valid_count = 0
        invalid_count = 0
        error_count = 0

        for entry in citation_entries:
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
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        document=pdf_path.name,
        document_path=str(pdf_path.absolute()),
        pipeline_version=pipeline_version,
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


__all__ = [
    "export_disease_results",
    "export_gene_results",
    "export_drug_results",
    "export_pharma_results",
    "export_author_results",
    "export_citation_results",
]
