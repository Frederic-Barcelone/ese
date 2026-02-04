# corpus_metadata/J_export/J01b_metadata_exporters.py
"""
Metadata export functions for document metadata, care pathways, and recommendations.

Provides export functions for document-level metadata including classification,
dates, care pathway graphs, and guideline recommendations. Extracted from
J01_export_handlers.py.

Key Components:
    - export_document_metadata: File info, PDF metadata, classification, dates
    - export_care_pathways: Treatment algorithm graphs (nodes, edges, phases)
    - export_recommendations: Guideline recommendations with dosing info

Example:
    >>> from J_export.J01b_metadata_exporters import export_document_metadata
    >>> export_document_metadata(
    ...     out_dir, pdf_path, metadata, run_id, pipeline_version
    ... )
    # Creates: metadata_{doc_name}_{timestamp}.json

Dependencies:
    - A_core.A08_document_metadata_models: DocumentMetadata, DocumentMetadataExport
    - A_core.A17_care_pathway_models: CarePathway
    - A_core.A18_recommendation_models: RecommendationSet
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from A_core.A08_document_metadata_models import DocumentMetadata
    from A_core.A17_care_pathway_models import CarePathway
    from A_core.A18_recommendation_models import RecommendationSet


def export_document_metadata(
    out_dir: Path,
    pdf_path: Path,
    metadata: "DocumentMetadata",
    run_id: str,
    pipeline_version: str,
    top_entities: Optional[list] = None,
) -> None:
    """Export document metadata to JSON file."""
    from A_core.A08_document_metadata_models import DocumentMetadataExport

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build simplified export
    export = DocumentMetadataExport(
        doc_id=metadata.doc_id,
        doc_filename=metadata.doc_filename,
        top_entities=top_entities,
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


def export_care_pathways(
    out_dir: Path,
    pdf_path: Path,
    pathways: List["CarePathway"],
    run_id: str,
    pipeline_version: str,
) -> None:
    """Export care pathway graphs extracted from treatment algorithm figures."""
    if not pathways:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build export data
    export_data: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "document": pdf_path.name,
        "document_path": str(pdf_path.absolute()),
        "pipeline_version": pipeline_version,
        "total_pathways": len(pathways),
        "pathways": [],
    }

    for pathway in pathways:
        # Export nodes with full structure
        nodes_export = []
        for node in pathway.nodes:
            nodes_export.append({
                "id": node.id,
                "type": node.type.value,
                "text": node.text,
                "phase": node.phase,
                "drugs": node.drugs,
                "dose": node.dose,
                "duration": node.duration,
            })

        # Export edges
        edges_export = []
        for edge in pathway.edges:
            edges_export.append({
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "condition": edge.condition,
                "condition_type": edge.condition_type,
            })

        pathway_data = {
            "pathway_id": pathway.pathway_id,
            "title": pathway.title,
            "condition": pathway.condition,
            "phases": pathway.phases,
            "nodes": nodes_export,
            "edges": edges_export,
            "entry_criteria": pathway.entry_criteria,
            "primary_drugs": pathway.primary_drugs,
            "alternative_drugs": pathway.alternative_drugs,
            "decision_points": pathway.decision_points,
            "target_dose": pathway.target_dose,
            "target_timepoint": pathway.target_timepoint,
            "maintenance_duration": pathway.maintenance_duration,
            "relapse_handling": pathway.relapse_handling,
            "source_figure": pathway.source_figure,
            "source_page": pathway.source_page,
            "extraction_confidence": pathway.extraction_confidence,
        }
        export_data["pathways"].append(pathway_data)

    # Write to file
    out_file = out_dir / f"care_pathways_{pdf_path.stem}_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    total_nodes = sum(len(p.nodes) for p in pathways)
    total_edges = sum(len(p.edges) for p in pathways)
    print(f"  Care pathways export: {out_file.name} ({len(pathways)} pathways, {total_nodes} nodes, {total_edges} edges)")


def export_recommendations(
    out_dir: Path,
    pdf_path: Path,
    recommendation_sets: List["RecommendationSet"],
    run_id: str,
    pipeline_version: str,
) -> None:
    """Export guideline recommendations extracted from text and tables."""
    if not recommendation_sets:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build export data
    export_data: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "document": pdf_path.name,
        "document_path": str(pdf_path.absolute()),
        "pipeline_version": pipeline_version,
        "total_recommendation_sets": len(recommendation_sets),
        "total_recommendations": sum(len(rs.recommendations) for rs in recommendation_sets),
        "recommendation_sets": [],
    }

    for rec_set in recommendation_sets:
        # Export individual recommendations
        recs_export = []
        for rec in rec_set.recommendations:
            # Export dosing info
            dosing_export = []
            for dose in rec.dosing:
                dosing_export.append({
                    "drug_name": dose.drug_name,
                    "dose_range": dose.dose_range,
                    "starting_dose": dose.starting_dose,
                    "maintenance_dose": dose.maintenance_dose,
                    "max_dose": dose.max_dose,
                    "route": dose.route,
                    "frequency": dose.frequency,
                })

            recs_export.append({
                "recommendation_id": rec.recommendation_id,
                "recommendation_type": rec.recommendation_type.value,
                "population": rec.population,
                "condition": rec.condition,
                "severity": rec.severity,
                "action": rec.action,
                "action_description": rec.action_description,
                "preferred": rec.preferred,
                "alternatives": rec.alternatives,
                "dosing": dosing_export,
                "taper_target": rec.taper_target,
                "duration": rec.duration,
                "stop_window": rec.stop_window,
                "evidence_level": rec.evidence_level.value,
                "strength": rec.strength.value,
                "references": rec.references,
                "source": rec.source,
                "source_text": rec.source_text,
                "page_num": rec.page_num,
            })

        set_data = {
            "guideline_name": rec_set.guideline_name,
            "guideline_year": rec_set.guideline_year,
            "organization": rec_set.organization,
            "target_condition": rec_set.target_condition,
            "target_population": rec_set.target_population,
            "recommendations": recs_export,
            "source_document": rec_set.source_document,
            "extraction_confidence": rec_set.extraction_confidence,
        }
        export_data["recommendation_sets"].append(set_data)

    # Write to file
    out_file = out_dir / f"recommendations_{pdf_path.stem}_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    total_recs = export_data["total_recommendations"]
    print(f"  Recommendations export: {out_file.name} ({len(recommendation_sets)} sets, {total_recs} recommendations)")


__all__ = [
    "export_document_metadata",
    "export_care_pathways",
    "export_recommendations",
]
