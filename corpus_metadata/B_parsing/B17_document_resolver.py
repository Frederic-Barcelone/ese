# corpus_metadata/B_parsing/B17_document_resolver.py
"""
Document-level resolution for visual extraction pipeline.

This module performs document-level operations after VLM enrichment: body text
reference scanning (finding "Figure 1" mentions), caption-to-visual linking,
section context inference, multi-page visual merging for continued tables/figures,
and deduplication of overlapping extractions.

Key Components:
    - ResolutionResult: Resolution results with merged visuals and statistics
    - BodyTextReference: A reference found in body text (page, char offset, type)
    - resolve_document: Main resolution function with merging and deduplication
    - scan_body_text_references: Find visual references in document body text
    - infer_section_context: Determine section context for each visual
    - merge_multipage_visuals: Merge continued tables/figures across pages
    - deduplicate_visuals: Remove overlapping duplicate extractions
    - BODY_REFERENCE_PATTERNS: Regex patterns for body text references
    - SECTION_PATTERNS: Regex patterns for section header detection

Example:
    >>> from B_parsing.B17_document_resolver import resolve_document
    >>> result = resolve_document(
    ...     visuals, pdf_path,
    ...     merge_multipage=True, deduplicate=True
    ... )
    >>> print(f"Resolved to {len(result.visuals)} visuals, {result.merges_performed} merges")

Dependencies:
    - A_core.A13_visual_models: ExtractedVisual, ReferenceSource, TextMention,
      VisualReference, VisualRelationships
    - fitz (PyMuPDF): PDF text extraction for body text scanning
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import fitz  # PyMuPDF

from A_core.A13_visual_models import (
    ExtractedVisual,
    ReferenceSource,
    TextMention,
    VisualReference,
    VisualRelationships,
)


# -------------------------
# Reference Patterns
# -------------------------


# Patterns for finding references in body text
BODY_REFERENCE_PATTERNS = [
    # "see Figure 1", "in Figure 2-4", "(Figure 1A)"
    re.compile(
        r"(?:see|in|from|as shown in|according to|per|refer to)?\s*"
        r"(?:\()?(?:Figure|Fig\.?)\s*(\d+)(?:[.-](\d+))?([A-Za-z])?(?:\))?",
        re.IGNORECASE,
    ),
    # "Table 1 shows", "in Table 2", "(Table 1)"
    re.compile(
        r"(?:see|in|from|as shown in|according to|per|refer to)?\s*"
        r"(?:\()?(?:Table)\s*(\d+)(?:[.-](\d+))?([A-Za-z])?(?:\))?",
        re.IGNORECASE,
    ),
    # "Exhibit A", "Exhibit 1.1"
    re.compile(
        r"(?:see|in|per)?\s*(?:\()?(?:Exhibit)\s*([A-Za-z\d]+)(?:\.\d+)?(?:\))?",
        re.IGNORECASE,
    ),
]


# Section header patterns
SECTION_PATTERNS = [
    (re.compile(r"^(?:\d+\.?\s+)?Results?\b", re.IGNORECASE), "Results"),
    (re.compile(r"^(?:\d+\.?\s+)?Methods?\b", re.IGNORECASE), "Methods"),
    (re.compile(r"^(?:\d+\.?\s+)?Discussion\b", re.IGNORECASE), "Discussion"),
    (re.compile(r"^(?:\d+\.?\s+)?Introduction\b", re.IGNORECASE), "Introduction"),
    (re.compile(r"^(?:\d+\.?\s+)?Conclusions?\b", re.IGNORECASE), "Conclusions"),
    (re.compile(r"^(?:\d+\.?\s+)?Abstract\b", re.IGNORECASE), "Abstract"),
    (re.compile(r"^(?:\d+\.?\s+)?Background\b", re.IGNORECASE), "Background"),
    (re.compile(r"^(?:\d+\.?\s+)?Appendix\b", re.IGNORECASE), "Appendix"),
]


# -------------------------
# Body Text Scanning
# -------------------------


@dataclass
class BodyTextReference:
    """A reference found in body text."""

    text: str
    page_num: int
    char_offset: int
    ref_type: str  # "figure", "table", "exhibit"
    number: int
    range_end: Optional[int] = None
    suffix: Optional[str] = None


def scan_body_text_references(
    doc: fitz.Document,
) -> List[BodyTextReference]:
    """
    Scan document body text for visual references.

    Args:
        doc: Open PyMuPDF document

    Returns:
        List of references found in body text
    """
    references: List[BodyTextReference] = []

    for page_idx in range(doc.page_count):
        page_num = page_idx + 1
        page = doc[page_idx]
        text = page.get_text()

        for pattern in BODY_REFERENCE_PATTERNS:
            for match in pattern.finditer(text):
                # Determine reference type from pattern
                match_text = match.group(0).lower()
                if "figure" in match_text or "fig" in match_text:
                    ref_type = "figure"
                elif "table" in match_text:
                    ref_type = "table"
                else:
                    ref_type = "exhibit"

                # Parse number
                try:
                    number = int(match.group(1))
                except (ValueError, TypeError):
                    continue

                # Parse range end
                range_end = None
                if match.lastindex is not None and match.lastindex >= 2 and match.group(2):
                    try:
                        range_end = int(match.group(2))
                    except ValueError:
                        pass

                # Parse suffix
                suffix = None
                if match.lastindex is not None and match.lastindex >= 3 and match.group(3):
                    suffix = match.group(3)

                references.append(
                    BodyTextReference(
                        text=match.group(0),
                        page_num=page_num,
                        char_offset=match.start(),
                        ref_type=ref_type,
                        number=number,
                        range_end=range_end,
                        suffix=suffix,
                    )
                )

    return references


def build_reference_index(
    references: List[BodyTextReference],
) -> Dict[str, List[BodyTextReference]]:
    """
    Build index of references by key.

    Key format: "figure:1", "table:2", etc.

    Args:
        references: List of body text references

    Returns:
        Dict mapping reference key to list of mentions
    """
    index: Dict[str, List[BodyTextReference]] = {}

    for ref in references:
        # Add entry for each number in range
        numbers = [ref.number]
        if ref.range_end:
            numbers = list(range(ref.number, ref.range_end + 1))

        for num in numbers:
            key = f"{ref.ref_type}:{num}"
            if key not in index:
                index[key] = []
            index[key].append(ref)

    return index


# -------------------------
# Section Context
# -------------------------


def detect_section_headers(
    doc: fitz.Document,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Detect section headers on each page.

    Args:
        doc: Open PyMuPDF document

    Returns:
        Dict mapping page_num to list of (section_name, y_position)
    """
    sections_by_page: Dict[int, List[Tuple[str, float]]] = {}

    for page_idx in range(doc.page_count):
        page_num = page_idx + 1
        page = doc[page_idx]
        text_dict = page.get_text("dict")

        page_sections: List[Tuple[str, float]] = []

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                text = "".join(
                    span.get("text", "") for span in line.get("spans", [])
                ).strip()

                if not text:
                    continue

                # Check against section patterns
                for pattern, section_name in SECTION_PATTERNS:
                    if pattern.match(text):
                        y_pos = line.get("bbox", (0, 0, 0, 0))[1]
                        page_sections.append((section_name, y_pos))
                        break

        if page_sections:
            sections_by_page[page_num] = page_sections

    return sections_by_page


def infer_section_for_visual(
    page_num: int,
    y_position: float,
    sections_by_page: Dict[int, List[Tuple[str, float]]],
) -> Optional[str]:
    """
    Infer which section a visual belongs to.

    Args:
        page_num: Page number of visual
        y_position: Y position of visual on page
        sections_by_page: Section headers by page

    Returns:
        Section name, or None if unknown
    """
    # Look for most recent section header before this visual
    current_section = None

    # Check current page and previous pages
    for pn in range(page_num, 0, -1):
        if pn not in sections_by_page:
            continue

        for section_name, section_y in reversed(sections_by_page[pn]):
            # If on same page, section must be above visual
            if pn == page_num and section_y > y_position:
                continue

            current_section = section_name
            return current_section

    return current_section


# -------------------------
# Multi-Page Merging
# -------------------------


@dataclass
class MergeCandidate:
    """Candidate visuals to merge."""

    visuals: List[ExtractedVisual]
    merge_reason: str


def find_merge_candidates(
    visuals: List[ExtractedVisual],
) -> List[MergeCandidate]:
    """
    Find visuals that should be merged (multi-page).

    Args:
        visuals: List of extracted visuals

    Returns:
        List of merge candidates
    """
    candidates: List[MergeCandidate] = []

    # Group by reference number
    by_reference: Dict[str, List[ExtractedVisual]] = {}

    for visual in visuals:
        if visual.reference:
            key = f"{visual.reference.type_label}:{visual.reference.numbers[0]}"
            if key not in by_reference:
                by_reference[key] = []
            by_reference[key].append(visual)

    # Check each group for merge candidates
    for key, group in by_reference.items():
        if len(group) < 2:
            continue

        # Sort by page number
        group.sort(key=lambda v: v.primary_page)

        # Check for consecutive pages
        for i in range(len(group) - 1):
            v1 = group[i]
            v2 = group[i + 1]

            # Check if consecutive pages
            if v2.primary_page - v1.primary_page == 1:
                # Check for continuation markers
                has_continuation = False

                if v2.caption_text:
                    cont_patterns = ["continued", "cont.", "cont'd"]
                    for pattern in cont_patterns:
                        if pattern in v2.caption_text.lower():
                            has_continuation = True
                            break

                if has_continuation:
                    candidates.append(
                        MergeCandidate(
                            visuals=[v1, v2],
                            merge_reason="continuation_marker",
                        )
                    )

    return candidates


def merge_multipage_visuals(
    visuals: List[ExtractedVisual],
    candidates: List[MergeCandidate],
) -> List[ExtractedVisual]:
    """
    Merge multi-page visuals.

    Args:
        visuals: All extracted visuals
        candidates: Merge candidates

    Returns:
        Updated list with merged visuals
    """
    merged_ids: Set[str] = set()
    merged_visuals: List[ExtractedVisual] = []

    for candidate in candidates:
        # Mark all as merged
        for v in candidate.visuals:
            merged_ids.add(v.visual_id)

        # Create merged visual from first
        first = candidate.visuals[0]

        # Combine page ranges and bboxes
        all_pages = []
        all_locations = []

        for v in candidate.visuals:
            for page in v.page_range:
                if page not in all_pages:
                    all_pages.append(page)
            all_locations.extend(v.bbox_pts_per_page)

        # Update relationships
        relationships = VisualRelationships(
            text_mentions=first.relationships.text_mentions,
            section_context=first.relationships.section_context,
        )

        # Set continuation links
        for i, v in enumerate(candidate.visuals):
            if i > 0:
                # This visual continues from previous
                pass  # Could update continues_from/continues_to

        # Create merged visual
        merged = ExtractedVisual(
            visual_id=first.visual_id,
            visual_type=first.visual_type,
            confidence=first.confidence,
            page_range=all_pages,
            bbox_pts_per_page=all_locations,
            caption_text=first.caption_text,
            caption_provenance=first.caption_provenance,
            reference=first.reference,
            image_base64=first.image_base64,  # Could stitch images
            image_format=first.image_format,
            render_dpi=first.render_dpi,
            docling_table=first.docling_table,
            validated_table=first.validated_table,
            table_extraction_mode=first.table_extraction_mode,
            relationships=relationships,
            extraction_method=first.extraction_method + "_merged",
            source_file=first.source_file,
            triage_decision=first.triage_decision,
            triage_reason=first.triage_reason,
            vlm_title=first.vlm_title,
            vlm_description=first.vlm_description,
        )

        merged_visuals.append(merged)

    # Add non-merged visuals
    for visual in visuals:
        if visual.visual_id not in merged_ids:
            merged_visuals.append(visual)

    return merged_visuals


# -------------------------
# Deduplication
# -------------------------


def compute_visual_overlap(
    v1: ExtractedVisual,
    v2: ExtractedVisual,
) -> float:
    """
    Compute overlap between two visuals.

    Args:
        v1: First visual
        v2: Second visual

    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    # Must be on same page
    if v1.primary_page != v2.primary_page:
        return 0.0

    bbox1 = v1.primary_bbox_pts
    bbox2 = v2.primary_bbox_pts

    # Compute intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def deduplicate_visuals(
    visuals: List[ExtractedVisual],
    overlap_threshold: float = 0.7,
) -> List[ExtractedVisual]:
    """
    Remove duplicate visuals based on overlap.

    Prefers visuals with:
    1. Higher confidence
    2. Has caption
    3. Has reference

    Args:
        visuals: List of visuals to deduplicate
        overlap_threshold: Minimum overlap to consider duplicate

    Returns:
        Deduplicated list
    """

    def visual_priority(v: ExtractedVisual) -> Tuple[float, bool, bool]:
        return (
            v.confidence,
            v.caption_text is not None,
            v.reference is not None,
        )

    # Sort by priority (highest first)
    sorted_visuals = sorted(visuals, key=visual_priority, reverse=True)

    kept: List[ExtractedVisual] = []
    removed_ids: Set[str] = set()

    for visual in sorted_visuals:
        if visual.visual_id in removed_ids:
            continue

        # Check overlap with already kept visuals
        is_duplicate = False
        for kept_visual in kept:
            overlap = compute_visual_overlap(visual, kept_visual)
            if overlap >= overlap_threshold:
                is_duplicate = True
                removed_ids.add(visual.visual_id)
                break

        if not is_duplicate:
            kept.append(visual)

    return kept


# -------------------------
# Resolution Orchestration
# -------------------------


@dataclass
class ResolutionResult:
    """Result of document-level resolution."""

    visuals: List[ExtractedVisual]
    body_references: List[BodyTextReference]
    sections_detected: int
    merges_performed: int
    duplicates_removed: int


def resolve_document(
    visuals: List[ExtractedVisual],
    pdf_path: str,
    merge_multipage: bool = True,
    deduplicate: bool = True,
    dedupe_threshold: float = 0.7,
) -> ResolutionResult:
    """
    Perform document-level resolution on extracted visuals.

    Args:
        visuals: Extracted visuals from previous stages
        pdf_path: Path to PDF file
        merge_multipage: Whether to merge multi-page visuals
        deduplicate: Whether to remove duplicates
        dedupe_threshold: Overlap threshold for deduplication

    Returns:
        ResolutionResult with resolved visuals
    """
    doc = fitz.open(pdf_path)

    try:
        # Step 1: Scan body text references
        body_refs = scan_body_text_references(doc)
        ref_index = build_reference_index(body_refs)

        # Step 2: Detect section headers
        sections_by_page = detect_section_headers(doc)

        # Step 3: Link references and sections to visuals
        updated_visuals: List[ExtractedVisual] = []

        for visual in visuals:
            # Find body text mentions
            mentions: List[TextMention] = []

            if visual.reference:
                key = f"{visual.reference.type_label.lower()}:{visual.reference.numbers[0]}"
                if key in ref_index:
                    for ref in ref_index[key]:
                        mention = TextMention(
                            text=ref.text,
                            page_num=ref.page_num,
                            char_offset=ref.char_offset,
                            reference=VisualReference(
                                raw_string=ref.text,
                                type_label=ref.ref_type.capitalize(),
                                numbers=[ref.number],
                                source=ReferenceSource.BODY_TEXT,
                            ),
                        )
                        mentions.append(mention)

            # Infer section context
            section = infer_section_for_visual(
                visual.primary_page,
                visual.primary_bbox_pts[1],  # y0
                sections_by_page,
            )

            # Update relationships
            relationships = VisualRelationships(
                text_mentions=mentions,
                section_context=section,
                continued_from=visual.relationships.continued_from,
                continues_to=visual.relationships.continues_to,
            )

            # Create updated visual
            updated = ExtractedVisual(
                visual_id=visual.visual_id,
                visual_type=visual.visual_type,
                confidence=visual.confidence,
                page_range=visual.page_range,
                bbox_pts_per_page=visual.bbox_pts_per_page,
                caption_text=visual.caption_text,
                caption_provenance=visual.caption_provenance,
                caption_bbox_pts=visual.caption_bbox_pts,
                reference=visual.reference,
                image_base64=visual.image_base64,
                image_format=visual.image_format,
                render_dpi=visual.render_dpi,
                docling_table=visual.docling_table,
                validated_table=visual.validated_table,
                table_extraction_mode=visual.table_extraction_mode,
                relationships=relationships,
                extraction_method=visual.extraction_method,
                source_file=visual.source_file,
                triage_decision=visual.triage_decision,
                triage_reason=visual.triage_reason,
                vlm_title=visual.vlm_title,
                vlm_description=visual.vlm_description,
            )
            updated_visuals.append(updated)

        # Step 4: Merge multi-page visuals
        merges = 0
        if merge_multipage:
            merge_candidates = find_merge_candidates(updated_visuals)
            merges = len(merge_candidates)
            updated_visuals = merge_multipage_visuals(updated_visuals, merge_candidates)

        # Step 5: Deduplicate
        removed = 0
        if deduplicate:
            before_count = len(updated_visuals)
            updated_visuals = deduplicate_visuals(updated_visuals, dedupe_threshold)
            removed = before_count - len(updated_visuals)

    finally:
        doc.close()

    return ResolutionResult(
        visuals=updated_visuals,
        body_references=body_refs,
        sections_detected=len(sections_by_page),
        merges_performed=merges,
        duplicates_removed=removed,
    )


__all__ = [
    # Types
    "BodyTextReference",
    "MergeCandidate",
    "ResolutionResult",
    # Main functions
    "resolve_document",
    # Body text
    "scan_body_text_references",
    "build_reference_index",
    # Sections
    "detect_section_headers",
    "infer_section_for_visual",
    # Merging
    "find_merge_candidates",
    "merge_multipage_visuals",
    # Deduplication
    "compute_visual_overlap",
    "deduplicate_visuals",
]
