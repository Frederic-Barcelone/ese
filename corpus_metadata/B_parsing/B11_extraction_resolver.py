# corpus_metadata/B_parsing/B11_extraction_resolver.py
"""
Deterministic Extraction Resolver

Merges all figure/table extraction signals (raster, vector, layout model)
using fixed rules with caption anchoring as highest priority.

Resolution rules:
1. Caption-anchored figures (highest priority)
2. Orphan native figures (no caption match, not overlapping tables)
3. Layout model figures (only if no native coverage)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from B_parsing.B09_pdf_native_figures import EmbeddedFigure, VectorFigure
from B_parsing.B10_caption_detector import (
    Caption,
    TableRegion,
    link_caption_to_figure,
    get_figure_captions,
    get_table_captions,
    find_next_caption_y,
    get_table_region_column_aware,
)


@dataclass
class ResolvedFigure:
    """A resolved figure with provenance tracking."""

    figure: Union[EmbeddedFigure, VectorFigure, Any]  # Any for layout model figures
    caption: Optional[Caption]
    source: str  # "caption_linked", "orphan_native", "layout_model"
    figure_type: str = "unknown"  # "raster", "vector", "layout_model"

    @property
    def page_num(self) -> int:
        """Get page number from underlying figure."""
        if hasattr(self.figure, "page_num"):
            return self.figure.page_num
        # Layout model figures may have different attribute
        if hasattr(self.figure, "bbox") and hasattr(self.figure.bbox, "page_num"):
            return self.figure.bbox.page_num
        return 1

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Get bounding box from underlying figure."""
        if hasattr(self.figure, "bbox"):
            bbox = self.figure.bbox
            if isinstance(bbox, tuple):
                return bbox
            # Handle BoundingBox objects
            if hasattr(bbox, "coords"):
                return bbox.coords
        return (0, 0, 0, 0)

    @property
    def caption_text(self) -> Optional[str]:
        """Get caption text if available."""
        return self.caption.text if self.caption else None


@dataclass
class ResolvedTable:
    """A resolved table with provenance tracking."""

    region: TableRegion
    caption: Caption
    source: str = "caption_linked"

    @property
    def page_num(self) -> int:
        return self.caption.page_num

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return self.region.bbox


def bbox_overlaps(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
    threshold: float = 0.3,
) -> bool:
    """
    Check if two bounding boxes overlap significantly.

    Args:
        bbox1: First bounding box (x0, y0, x1, y1)
        bbox2: Second bounding box (x0, y0, x1, y1)
        threshold: Minimum overlap ratio to consider overlapping

    Returns:
        True if boxes overlap by at least threshold
    """
    x1_0, y1_0, x1_1, y1_1 = bbox1
    x2_0, y2_0, x2_1, y2_1 = bbox2

    # Calculate intersection
    ix0 = max(x1_0, x2_0)
    iy0 = max(y1_0, y2_0)
    ix1 = min(x1_1, x2_1)
    iy1 = min(y1_1, y2_1)

    if ix1 <= ix0 or iy1 <= iy0:
        return False

    intersection = (ix1 - ix0) * (iy1 - iy0)
    area1 = (x1_1 - x1_0) * (y1_1 - y1_0)
    area2 = (x2_1 - x2_0) * (y2_1 - y2_0)

    if area1 <= 0 or area2 <= 0:
        return False

    # Check overlap relative to smaller box
    min_area = min(area1, area2)
    return (intersection / min_area) >= threshold


def resolve_tables(
    captions: List[Caption],
    columns_by_page: Dict[int, List[Tuple[float, float]]],
    resolved_figures: List[ResolvedFigure],
    doc: Any,  # fitz.Document
) -> List[ResolvedTable]:
    """
    Resolve table regions based on captions.

    Args:
        captions: All detected captions
        columns_by_page: Column boundaries per page
        resolved_figures: Already resolved figures (to exclude overlap)
        doc: Open PyMuPDF document

    Returns:
        List of ResolvedTable objects
    """
    table_captions = get_table_captions(captions)
    resolved_tables: List[ResolvedTable] = []

    for cap in table_captions:
        page = doc[cap.page_num - 1]
        columns = columns_by_page.get(cap.page_num, [(0, page.rect.width)])

        # Find next caption Y for bounds
        next_y = find_next_caption_y(cap, captions)

        # Get text dict for span detection
        text_dict = page.get_text("dict")

        # Define table region
        region_bbox = get_table_region_column_aware(
            cap, columns, page, next_y, text_dict
        )

        # Check for overlap with resolved figures
        overlaps_figure = any(
            bbox_overlaps(region_bbox, rf.bbox, threshold=0.3)
            for rf in resolved_figures
            if rf.page_num == cap.page_num
        )

        if overlaps_figure:
            # Adjust region to exclude figure - truncate at figure top
            for rf in resolved_figures:
                if rf.page_num != cap.page_num:
                    continue
                if bbox_overlaps(region_bbox, rf.bbox, threshold=0.1):
                    # Truncate table region at figure top
                    fig_top = rf.bbox[1]
                    if fig_top > region_bbox[1]:
                        region_bbox = (
                            region_bbox[0],
                            region_bbox[1],
                            region_bbox[2],
                            min(region_bbox[3], fig_top - 10),
                        )

        table_region = TableRegion(
            caption=cap,
            bbox=region_bbox,
            page_num=cap.page_num,
        )

        resolved_tables.append(
            ResolvedTable(
                region=table_region,
                caption=cap,
                source="caption_linked",
            )
        )

    return resolved_tables


def resolve_all(
    raster_figures: List[EmbeddedFigure],
    vector_figures: List[VectorFigure],
    layout_model_figures: List[Any],  # ImageBlock from Unstructured
    captions: List[Caption],
    columns_by_page: Dict[int, List[Tuple[float, float]]],
    doc: Any,  # fitz.Document
) -> Tuple[List[ResolvedFigure], List[ResolvedTable]]:
    """
    Deterministic resolution with fixed rules.

    Figures:
    1. For each "Fig X" caption -> link to nearest region above (raster OR vector)
    2. Exclude any regions that overlap with table_regions
    3. Use layout_model_figures only for orphan regions (no caption match)

    Tables:
    1. For each "Table X" caption -> use column-aware region
    2. Exclude any overlap with resolved figures

    Args:
        raster_figures: Native raster images from PDF
        vector_figures: Detected vector plots
        layout_model_figures: Images from Unstructured/YOLOX
        captions: All detected captions
        columns_by_page: Column boundaries per page
        doc: Open PyMuPDF document

    Returns:
        Tuple of (resolved_figures, resolved_tables)
    """
    resolved_figures: List[ResolvedFigure] = []
    used_regions: set = set()

    # Step 1: Caption-anchored figures (highest priority)
    figure_captions = get_figure_captions(captions)
    for cap in figure_captions:
        columns = columns_by_page.get(cap.page_num)

        linked = link_caption_to_figure(
            cap,
            raster_figures,
            vector_figures,
            same_column_only=False,  # Allow cross-column linking
            columns=columns,
        )

        if linked:
            fig_type = "raster" if isinstance(linked, EmbeddedFigure) else "vector"
            resolved_figures.append(
                ResolvedFigure(
                    figure=linked,
                    caption=cap,
                    source="caption_linked",
                    figure_type=fig_type,
                )
            )
            used_regions.add(id(linked))

    # Compute preliminary table regions for exclusion
    table_captions = get_table_captions(captions)
    preliminary_table_bboxes: List[Tuple[int, Tuple[float, float, float, float]]] = []

    for cap in table_captions:
        page = doc[cap.page_num - 1]
        columns = columns_by_page.get(cap.page_num, [(0, page.rect.width)])
        next_y = find_next_caption_y(cap, captions)
        text_dict = page.get_text("dict")

        region_bbox = get_table_region_column_aware(
            cap, columns, page, next_y, text_dict
        )
        preliminary_table_bboxes.append((cap.page_num, region_bbox))

    # Step 2: Orphan raster/vector figures (no caption)
    all_native = list(raster_figures) + list(vector_figures)
    for fig in all_native:
        if id(fig) in used_regions:
            continue

        fig_bbox = fig.bbox
        page_num = fig.page_num

        # Check not overlapping with tables
        overlaps_table = any(
            pn == page_num and bbox_overlaps(fig_bbox, tb, threshold=0.3)
            for pn, tb in preliminary_table_bboxes
        )

        if overlaps_table:
            continue

        fig_type = "raster" if isinstance(fig, EmbeddedFigure) else "vector"
        resolved_figures.append(
            ResolvedFigure(
                figure=fig,
                caption=None,
                source="orphan_native",
                figure_type=fig_type,
            )
        )
        used_regions.add(id(fig))

    # Step 3: Layout model figures (only if no native coverage)
    for lm_fig in layout_model_figures:
        # Get bbox from layout model figure
        if hasattr(lm_fig, "bbox"):
            lm_bbox = lm_fig.bbox
            if hasattr(lm_bbox, "coords"):
                lm_bbox = lm_bbox.coords
        else:
            continue

        # Get page number
        lm_page = getattr(lm_fig, "page_num", 1)

        # Check if already covered by native figures
        covered_by_native = any(
            rf.page_num == lm_page and bbox_overlaps(lm_bbox, rf.bbox, threshold=0.5)
            for rf in resolved_figures
        )

        if covered_by_native:
            continue

        # Check overlap with tables
        overlaps_table = any(
            pn == lm_page and bbox_overlaps(lm_bbox, tb, threshold=0.3)
            for pn, tb in preliminary_table_bboxes
        )

        if overlaps_table:
            continue

        resolved_figures.append(
            ResolvedFigure(
                figure=lm_fig,
                caption=None,
                source="layout_model",
                figure_type="layout_model",
            )
        )

    # Step 4: Resolve tables with figure exclusion
    resolved_tables = resolve_tables(
        captions, columns_by_page, resolved_figures, doc
    )

    return resolved_figures, resolved_tables


def filter_duplicate_figures(
    figures: List[ResolvedFigure],
    overlap_threshold: float = 0.7,
) -> List[ResolvedFigure]:
    """
    Remove duplicate figures that overlap significantly.

    Priority: caption_linked > orphan_native > layout_model

    Args:
        figures: List of resolved figures
        overlap_threshold: Threshold for considering duplicates

    Returns:
        Deduplicated list
    """
    # Sort by priority (lower is better)
    priority_map = {"caption_linked": 0, "orphan_native": 1, "layout_model": 2}

    sorted_figures = sorted(
        figures,
        key=lambda f: (f.page_num, priority_map.get(f.source, 3)),
    )

    result: List[ResolvedFigure] = []
    for fig in sorted_figures:
        # Check if overlaps with any already-added figure
        is_duplicate = any(
            fig.page_num == existing.page_num
            and bbox_overlaps(fig.bbox, existing.bbox, overlap_threshold)
            for existing in result
        )

        if not is_duplicate:
            result.append(fig)

    return result


def get_resolution_stats(
    resolved_figures: List[ResolvedFigure],
    resolved_tables: List[ResolvedTable],
) -> Dict[str, Any]:
    """
    Get statistics about resolution results.

    Args:
        resolved_figures: Resolved figure list
        resolved_tables: Resolved table list

    Returns:
        Dictionary with statistics
    """
    figure_sources = {}
    for fig in resolved_figures:
        source = fig.source
        figure_sources[source] = figure_sources.get(source, 0) + 1

    figure_types = {}
    for fig in resolved_figures:
        ftype = fig.figure_type
        figure_types[ftype] = figure_types.get(ftype, 0) + 1

    return {
        "total_figures": len(resolved_figures),
        "total_tables": len(resolved_tables),
        "figures_by_source": figure_sources,
        "figures_by_type": figure_types,
        "figures_with_caption": sum(1 for f in resolved_figures if f.caption),
        "figures_without_caption": sum(1 for f in resolved_figures if not f.caption),
    }


__all__ = [
    "ResolvedFigure",
    "ResolvedTable",
    "bbox_overlaps",
    "resolve_all",
    "resolve_tables",
    "filter_duplicate_figures",
    "get_resolution_stats",
]
