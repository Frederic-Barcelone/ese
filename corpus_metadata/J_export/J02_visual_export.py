# corpus_metadata/J_export/J02_visual_export.py
"""
Export handlers for the new visual extraction pipeline.

Exports ExtractedVisual objects to JSON format with:
- Full metadata (type, reference, caption, relationships)
- Base64 images (or external file references)
- Structured table data for table visuals
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from A_core.A13_visual_models import (
    ExtractedVisual,
)
from B_parsing.B12_visual_pipeline import PipelineResult

logger = logging.getLogger(__name__)


# -------------------------
# Serialization
# -------------------------


def visual_to_dict(
    visual: ExtractedVisual,
    include_image: bool = True,
    image_file: str | None = None,
) -> Dict[str, Any]:
    """
    Convert ExtractedVisual to serializable dict.

    Args:
        visual: ExtractedVisual to convert
        include_image: Whether to include base64 image data
        image_file: If provided, use this filename instead of base64

    Returns:
        Dict suitable for JSON serialization
    """
    result = {
        "visual_id": visual.visual_id,
        "visual_type": visual.visual_type.value,
        "confidence": visual.confidence,
        "page_range": visual.page_range,
        "is_multipage": visual.is_multipage,
    }

    # Location
    result["locations"] = [
        {
            "page_num": loc.page_num,
            "bbox_pts": list(loc.bbox_pts),
        }
        for loc in visual.bbox_pts_per_page
    ]

    # Top-level title, description, and caption fields
    result["title"] = visual.vlm_title
    result["description"] = visual.vlm_description
    result["caption"] = visual.caption_text

    # Reference (e.g., "Table 1", "Figure 2")
    if visual.reference:
        result["reference"] = visual.reference.raw_string

    # Image file (top-level for easy access)
    result["image_file"] = image_file

    # Image metadata
    if include_image and not image_file:
        # Fall back to base64 if no file saved
        result["image_base64"] = visual.image_base64

    # Table-specific
    if visual.is_table:
        result["table_data"] = {}

        if visual.validated_table:
            result["table_data"]["structure"] = {
                "headers": visual.validated_table.headers,
                "rows": visual.validated_table.rows,
                "confidence": visual.validated_table.structure_confidence,
            }
        elif visual.docling_table:
            result["table_data"]["structure"] = {
                "headers": visual.docling_table.headers,
                "rows": visual.docling_table.rows,
                "confidence": visual.docling_table.structure_confidence,
            }

        if visual.table_extraction_mode:
            result["table_data"]["extraction_mode"] = visual.table_extraction_mode.value

    # Relationships
    if visual.relationships:
        mentions = []
        for mention in visual.relationships.text_mentions:
            mentions.append({
                "text": mention.text,
                "page_num": mention.page_num,
                "char_offset": mention.char_offset,
            })

        result["relationships"] = {
            "text_mentions": mentions,
            "section_context": visual.relationships.section_context,
            "continued_from": visual.relationships.continued_from,
            "continues_to": visual.relationships.continues_to,
        }

    # Provenance
    result["provenance"] = {
        "extraction_method": visual.extraction_method,
        "source_file": visual.source_file,
        "extracted_at": visual.extracted_at.isoformat(),
    }

    # Triage info
    if visual.triage_decision:
        result["triage"] = {
            "decision": visual.triage_decision.value,
            "reason": visual.triage_reason,
        }

    return result


def pipeline_result_to_dict(result: PipelineResult) -> Dict[str, Any]:
    """
    Convert PipelineResult to serializable dict.

    Args:
        result: Pipeline result to convert

    Returns:
        Dict suitable for JSON serialization
    """
    return {
        "metadata": {
            "source_file": result.source_file,
            "extracted_at": result.extracted_at.isoformat(),
            "extraction_time_seconds": result.extraction_time_seconds,
        },
        "statistics": {
            "tables_detected": result.tables_detected,
            "figures_detected": result.figures_detected,
            "tables_escalated": result.tables_escalated,
            "vlm_enriched": result.vlm_enriched,
            "merges_performed": result.merges_performed,
            "duplicates_removed": result.duplicates_removed,
            "final_count": len(result.visuals),
        },
        "visuals": [visual_to_dict(v) for v in result.visuals],
    }


# -------------------------
# Export Functions
# -------------------------


def export_visuals_to_json(
    result: PipelineResult,
    output_path: Path,
    include_images: bool = True,
    pretty_print: bool = True,
) -> Path:
    """
    Export pipeline result to JSON file.

    Args:
        result: Pipeline result to export
        output_path: Output file path
        include_images: Whether to include base64 images
        pretty_print: Whether to format JSON with indentation

    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": {
            "source_file": result.source_file,
            "extracted_at": result.extracted_at.isoformat(),
            "extraction_time_seconds": result.extraction_time_seconds,
        },
        "statistics": {
            "tables_detected": result.tables_detected,
            "figures_detected": result.figures_detected,
            "tables_escalated": result.tables_escalated,
            "vlm_enriched": result.vlm_enriched,
            "merges_performed": result.merges_performed,
            "duplicates_removed": result.duplicates_removed,
            "final_count": len(result.visuals),
        },
        "visuals": [
            visual_to_dict(v, include_image=include_images)
            for v in result.visuals
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        if pretty_print:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)

    logger.info(f"Exported {len(result.visuals)} visuals to {output_path}")
    return output_path


def export_tables_only(
    result: PipelineResult,
    output_path: Path,
    output_dir: Path | None = None,
    doc_name: str = "",
    save_images: bool = True,
) -> Path:
    """
    Export only tables from pipeline result.

    Args:
        result: Pipeline result
        output_path: Path for JSON output
        output_dir: Directory for image files (defaults to output_path parent)
        doc_name: Document name prefix for image filenames
        save_images: Whether to save images as files (vs embed base64)

    Returns:
        Path to exported JSON file
    """
    import base64

    tables = [v for v in result.visuals if v.is_table]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_dir is None:
        output_dir = output_path.parent

    # Track page counts for numbering
    page_counts: Dict[int, int] = {}
    table_dicts = []

    for table in tables:
        page_num = table.primary_page
        if page_num not in page_counts:
            page_counts[page_num] = 0
        page_counts[page_num] += 1
        idx = page_counts[page_num]

        image_file = None
        image_save_failed = False
        if save_images and table.image_base64:
            # Generate filename and save image
            prefix = f"{doc_name}_" if doc_name else ""
            filename = f"{prefix}table_page{page_num}_{idx}.png"
            image_path = output_dir / filename

            try:
                img_bytes = base64.b64decode(table.image_base64)
                with open(image_path, "wb") as f:
                    f.write(img_bytes)
                image_file = filename
            except Exception as e:
                logger.warning(f"Failed to save table image: {e}, falling back to base64")
                image_save_failed = True

        # If save failed, fall back to embedding base64
        include_base64 = (not save_images) or image_save_failed
        table_dicts.append(visual_to_dict(
            table,
            include_image=include_base64,
            image_file=image_file,
        ))

    data = {
        "metadata": {
            "source_file": result.source_file,
            "extracted_at": result.extracted_at.isoformat(),
        },
        "count": len(tables),
        "tables": table_dicts,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported {len(tables)} tables to {output_path}")
    return output_path


def export_figures_only(
    result: PipelineResult,
    output_path: Path,
    output_dir: Path | None = None,
    doc_name: str = "",
    save_images: bool = True,
) -> Path:
    """
    Export only figures from pipeline result.

    Args:
        result: Pipeline result
        output_path: Path for JSON output
        output_dir: Directory for image files (defaults to output_path parent)
        doc_name: Document name prefix for image filenames
        save_images: Whether to save images as files (vs embed base64)

    Returns:
        Path to exported JSON file
    """
    import base64

    figures = [v for v in result.visuals if v.is_figure]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_dir is None:
        output_dir = output_path.parent

    # Track page counts for numbering
    page_counts: Dict[int, int] = {}
    figure_dicts = []

    for figure in figures:
        page_num = figure.primary_page
        if page_num not in page_counts:
            page_counts[page_num] = 0
        page_counts[page_num] += 1
        idx = page_counts[page_num]

        image_file = None
        image_save_failed = False
        if save_images and figure.image_base64:
            # Generate filename and save image
            prefix = f"{doc_name}_" if doc_name else ""
            filename = f"{prefix}figure_page{page_num}_{idx}.png"
            image_path = output_dir / filename

            try:
                img_bytes = base64.b64decode(figure.image_base64)
                with open(image_path, "wb") as f:
                    f.write(img_bytes)
                image_file = filename
            except Exception as e:
                logger.warning(f"Failed to save figure image: {e}, falling back to base64")
                image_save_failed = True

        # If save failed, fall back to embedding base64
        include_base64 = (not save_images) or image_save_failed
        figure_dicts.append(visual_to_dict(
            figure,
            include_image=include_base64,
            image_file=image_file,
        ))

    data = {
        "metadata": {
            "source_file": result.source_file,
            "extracted_at": result.extracted_at.isoformat(),
        },
        "count": len(figures),
        "figures": figure_dicts,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported {len(figures)} figures to {output_path}")
    return output_path


# -------------------------
# Image Export (Optional)
# -------------------------


def export_images_separately(
    result: PipelineResult,
    output_dir: Path,
    doc_name: str = "",
    format: str = "png",
) -> Dict[str, Path]:
    """
    Export visual images as separate files.

    Args:
        result: Pipeline result
        output_dir: Directory for image files
        doc_name: Document name prefix for filenames
        format: Image format (png, jpg)

    Returns:
        Dict mapping visual_id to image file path
    """
    import base64

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths: Dict[str, Path] = {}

    # Track counts per page for numbering
    page_counts: Dict[str, Dict[int, int]] = {"table": {}, "figure": {}}

    for visual in result.visuals:
        # Determine type
        type_prefix = "table" if visual.is_table else "figure"
        page_num = visual.primary_page

        # Increment count for this type/page
        if page_num not in page_counts[type_prefix]:
            page_counts[type_prefix][page_num] = 0
        page_counts[type_prefix][page_num] += 1
        idx = page_counts[type_prefix][page_num]

        # Generate filename: {doc_name}_{type}_page{N}_{idx}.png
        if doc_name:
            filename = f"{doc_name}_{type_prefix}_page{page_num}_{idx}.{format}"
        else:
            filename = f"{type_prefix}_page{page_num}_{idx}.{format}"

        image_path = output_dir / filename

        # Decode and write image
        try:
            img_bytes = base64.b64decode(visual.image_base64)
            with open(image_path, "wb") as f:
                f.write(img_bytes)

            image_paths[visual.visual_id] = image_path

        except Exception as e:
            logger.warning(f"Failed to export image {visual.visual_id}: {e}")

    logger.info(f"Exported {len(image_paths)} images to {output_dir}")
    return image_paths


__all__ = [
    # Serialization
    "visual_to_dict",
    "pipeline_result_to_dict",
    # Export
    "export_visuals_to_json",
    "export_tables_only",
    "export_figures_only",
    "export_images_separately",
]
