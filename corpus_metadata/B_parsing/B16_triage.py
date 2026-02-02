# corpus_metadata/B_parsing/B16_triage.py
"""
Visual triage logic for determining VLM processing requirements.

This module determines which visual candidates need VLM processing versus can be
skipped or handled cheaply. It uses cheap signals (area ratio, repeated image hash,
caption presence, grid structure, body text references) to minimize expensive VLM
calls while ensuring important visuals receive full enrichment.

Key Components:
    - TriageConfig: Configuration for area thresholds, repeat detection, margins
    - TriageResult: Triage decision with reason and confidence
    - TriageDecision: Enum (SKIP, CHEAP_PATH, VLM_REQUIRED)
    - DocumentContext: Document-level context for triage (repeated hashes, references)
    - triage_batch: Batch triage of visual candidates
    - get_vlm_candidates: Filter candidates requiring VLM processing
    - is_in_margin_zone: Check if visual is in header/footer margin
    - should_escalate_to_accurate: Check if table needs ACCURATE mode
    - compute_triage_statistics: Summary statistics for triage results

Example:
    >>> from B_parsing.B16_triage import triage_batch, TriageConfig
    >>> config = TriageConfig(skip_area_ratio=0.02, vlm_area_threshold=0.10)
    >>> triaged = triage_batch(candidates, config)
    >>> vlm_needed = [c for c, r in triaged if r.decision == TriageDecision.VLM_REQUIRED]

Dependencies:
    - A_core.A13_visual_models: TableComplexitySignals, TriageDecision, TriageResult, VisualCandidate
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from A_core.A13_visual_models import (
    TableComplexitySignals,
    TriageDecision,
    TriageResult,
    VisualCandidate,
)


# -------------------------
# Configuration
# -------------------------


@dataclass
class TriageConfig:
    """Configuration for visual triage."""

    # Area thresholds (as fraction of page area)
    skip_area_ratio: float = 0.02  # <2% of page = skip
    vlm_area_threshold: float = 0.10  # >10% without caption = VLM

    # Repeat detection
    repeat_threshold: int = 3  # Same image on 3+ pages = header/footer

    # Margin zones (as fraction of page)
    header_zone_ratio: float = 0.10  # Top 10%
    footer_zone_ratio: float = 0.10  # Bottom 10%

    # Grid detection
    min_grid_cells: int = 4  # Minimum cells to consider grid-like

    # Confidence thresholds
    high_confidence_skip: float = 0.90
    medium_confidence: float = 0.70


# -------------------------
# Document Context
# -------------------------


@dataclass
class DocumentContext:
    """
    Document-level context for triage decisions.

    Tracks repeated images and body text references across the document.
    """

    # Image hash -> count of pages it appears on
    image_hash_counts: Dict[str, int] = field(default_factory=dict)

    # Set of repeated image hashes (appears on 3+ pages)
    repeated_image_hashes: Set[str] = field(default_factory=set)

    # Set of reference numbers mentioned in body text
    # Format: "figure:1", "table:2", etc.
    body_text_references: Set[str] = field(default_factory=set)

    # Page dimensions
    page_dimensions: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def build_from_candidates(
        cls,
        candidates: List[VisualCandidate],
        repeat_threshold: int = 3,
    ) -> "DocumentContext":
        """
        Build document context from a list of candidates.

        Args:
            candidates: All visual candidates in document
            repeat_threshold: Number of pages for an image to be considered repeated

        Returns:
            DocumentContext with computed statistics
        """
        hash_counts: Counter = Counter()
        page_dims: Dict[int, Tuple[float, float]] = {}

        for c in candidates:
            if c.image_hash:
                hash_counts[c.image_hash] += 1
            page_dims[c.page_num] = (c.page_width_pts, c.page_height_pts)

        repeated = {h for h, count in hash_counts.items() if count >= repeat_threshold}

        return cls(
            image_hash_counts=dict(hash_counts),
            repeated_image_hashes=repeated,
            page_dimensions=page_dims,
        )

    def add_body_reference(self, ref_type: str, number: int) -> None:
        """Add a body text reference."""
        self.body_text_references.add(f"{ref_type.lower()}:{number}")

    def is_referenced_in_body(self, ref_type: str, number: int) -> bool:
        """Check if a reference is mentioned in body text."""
        return f"{ref_type.lower()}:{number}" in self.body_text_references


# -------------------------
# Triage Logic
# -------------------------


def is_in_margin_zone(
    bbox_pts: Tuple[float, float, float, float],
    page_height: float,
    header_ratio: float = 0.10,
    footer_ratio: float = 0.10,
) -> bool:
    """
    Check if visual is in header or footer zone.

    Args:
        bbox_pts: Visual bounding box in PDF points
        page_height: Page height in PDF points
        header_ratio: Top portion of page considered header
        footer_ratio: Bottom portion of page considered footer

    Returns:
        True if visual is in margin zone
    """
    _, y0, _, y1 = bbox_pts

    header_cutoff = page_height * header_ratio
    footer_cutoff = page_height * (1 - footer_ratio)

    # Check if entirely in header zone
    if y1 < header_cutoff:
        return True

    # Check if entirely in footer zone
    if y0 > footer_cutoff:
        return True

    return False


def compute_area_ratio(
    bbox_pts: Tuple[float, float, float, float],
    page_width: float,
    page_height: float,
) -> float:
    """
    Compute area ratio of visual to page.

    Args:
        bbox_pts: Visual bounding box in PDF points
        page_width: Page width in PDF points
        page_height: Page height in PDF points

    Returns:
        Area ratio (0.0 to 1.0)
    """
    x0, y0, x1, y1 = bbox_pts
    visual_area = (x1 - x0) * (y1 - y0)
    page_area = page_width * page_height

    if page_area <= 0:
        return 0.0

    return visual_area / page_area


def triage_visual(
    candidate: VisualCandidate,
    doc_context: DocumentContext,
    config: TriageConfig = TriageConfig(),
) -> TriageResult:
    """
    Determine triage decision for a visual candidate.

    Uses cheap signals to route visuals before expensive VLM calls.

    Args:
        candidate: Visual candidate to triage
        doc_context: Document-level context
        config: Triage configuration

    Returns:
        TriageResult with decision and reasoning
    """
    # === SKIP signals (noise filtering) ===

    # 1. Tiny area (logos, icons, separators)
    if candidate.area_ratio < config.skip_area_ratio:
        return TriageResult(
            decision=TriageDecision.SKIP,
            reason="tiny_area",
            confidence=config.high_confidence_skip,
        )

    # 2. Repeated across pages (headers/footers)
    if candidate.image_hash and candidate.image_hash in doc_context.repeated_image_hashes:
        count = doc_context.image_hash_counts.get(candidate.image_hash, 0)
        if count >= config.repeat_threshold:
            return TriageResult(
                decision=TriageDecision.SKIP,
                reason="repeated_graphic",
                confidence=config.high_confidence_skip,
            )

    # 3. In margin zone with no caption
    if candidate.in_margin_zone and not candidate.has_nearby_caption:
        return TriageResult(
            decision=TriageDecision.SKIP,
            reason="margin_no_caption",
            confidence=0.85,
        )

    # === VLM_REQUIRED signals ===

    # 4. Has caption-like text nearby
    if candidate.has_nearby_caption:
        return TriageResult(
            decision=TriageDecision.VLM_REQUIRED,
            reason="has_caption",
            confidence=0.95,
        )

    # 5. Docling detected as table
    if candidate.docling_type == "table":
        return TriageResult(
            decision=TriageDecision.VLM_REQUIRED,
            reason="docling_table",
            confidence=0.95,
        )

    # 6. Referenced in body text
    if candidate.is_referenced_in_text:
        return TriageResult(
            decision=TriageDecision.VLM_REQUIRED,
            reason="body_reference",
            confidence=0.90,
        )

    # 7. Dense grid structure (likely table)
    if candidate.has_grid_structure:
        return TriageResult(
            decision=TriageDecision.VLM_REQUIRED,
            reason="grid_structure",
            confidence=0.85,
        )

    # 8. Flagged for ACCURATE re-run
    if candidate.needs_accurate_rerun:
        return TriageResult(
            decision=TriageDecision.VLM_REQUIRED,
            reason="complex_table",
            confidence=0.90,
        )

    # 9. Large area but no caption (ambiguous - needs VLM)
    if candidate.area_ratio > config.vlm_area_threshold:
        return TriageResult(
            decision=TriageDecision.VLM_REQUIRED,
            reason="large_uncaptioned",
            confidence=config.medium_confidence,
        )

    # 10. Has continuation markers
    if candidate.continuation_markers:
        return TriageResult(
            decision=TriageDecision.VLM_REQUIRED,
            reason="continuation_detected",
            confidence=0.85,
        )

    # === CHEAP_PATH (medium area, no strong signals) ===
    return TriageResult(
        decision=TriageDecision.CHEAP_PATH,
        reason="default",
        confidence=0.60,
    )


def triage_batch(
    candidates: List[VisualCandidate],
    doc_context: Optional[DocumentContext] = None,
    config: TriageConfig = TriageConfig(),
) -> List[Tuple[VisualCandidate, TriageResult]]:
    """
    Triage a batch of visual candidates.

    Args:
        candidates: List of visual candidates
        doc_context: Document context (built from candidates if not provided)
        config: Triage configuration

    Returns:
        List of (candidate, triage_result) tuples
    """
    if doc_context is None:
        doc_context = DocumentContext.build_from_candidates(
            candidates, config.repeat_threshold
        )

    results = []
    for candidate in candidates:
        result = triage_visual(candidate, doc_context, config)
        results.append((candidate, result))

    return results


def filter_by_decision(
    triaged: List[Tuple[VisualCandidate, TriageResult]],
    decision: TriageDecision,
) -> List[VisualCandidate]:
    """
    Filter triaged candidates by decision.

    Args:
        triaged: List of (candidate, result) tuples from triage_batch
        decision: Decision to filter by

    Returns:
        List of candidates with the specified decision
    """
    return [c for c, r in triaged if r.decision == decision]


def get_vlm_candidates(
    triaged: List[Tuple[VisualCandidate, TriageResult]],
) -> List[VisualCandidate]:
    """Get candidates that need VLM processing."""
    return filter_by_decision(triaged, TriageDecision.VLM_REQUIRED)


def get_skip_candidates(
    triaged: List[Tuple[VisualCandidate, TriageResult]],
) -> List[VisualCandidate]:
    """Get candidates to skip."""
    return filter_by_decision(triaged, TriageDecision.SKIP)


def get_cheap_path_candidates(
    triaged: List[Tuple[VisualCandidate, TriageResult]],
) -> List[VisualCandidate]:
    """Get candidates for cheap path processing."""
    return filter_by_decision(triaged, TriageDecision.CHEAP_PATH)


# -------------------------
# Table Complexity Assessment
# -------------------------


def should_escalate_to_accurate(
    signals: TableComplexitySignals,
    config: Optional[Dict] = None,
) -> Tuple[bool, str]:
    """
    Determine if table needs ACCURATE mode re-run.

    Args:
        signals: Complexity signals from table
        config: Optional config overrides

    Returns:
        Tuple of (should_escalate, reason)
    """
    # Default thresholds
    header_depth_threshold = 3
    merged_cell_threshold = 5
    merge_ratio_threshold = 0.10
    token_coverage_threshold = 0.70
    large_table_cols = 8
    large_table_rows = 15

    if config:
        header_depth_threshold = config.get("header_depth_threshold", header_depth_threshold)
        merged_cell_threshold = config.get("merged_cell_threshold", merged_cell_threshold)
        merge_ratio_threshold = config.get("merge_ratio_threshold", merge_ratio_threshold)
        token_coverage_threshold = config.get("token_coverage_threshold", token_coverage_threshold)
        large_table_cols = config.get("large_table_cols", large_table_cols)
        large_table_rows = config.get("large_table_rows", large_table_rows)

    # 1. Multi-page tables always need ACCURATE
    if signals.spans_multiple_pages:
        return True, "multipage_table"

    # 2. Deep header stacks
    if signals.header_depth >= header_depth_threshold:
        return True, "deep_headers"

    # 3. Many merged cells (absolute count)
    if signals.merged_cell_count > merged_cell_threshold:
        return True, "many_merged_cells"

    # 4. High merge ratio
    total_cells = signals.column_count * signals.row_count
    if total_cells > 0:
        merge_ratio = signals.merged_cell_count / total_cells
        if merge_ratio > merge_ratio_threshold:
            return True, "high_merge_ratio"

    # 5. Low token coverage
    if signals.token_coverage_ratio < token_coverage_threshold:
        return True, "low_token_coverage"

    # 6. VLM flagged as misparsed
    if signals.vlm_flagged_misparsed:
        return True, "vlm_flagged_misparsed"

    # 7. Large complex tables
    if signals.column_count >= large_table_cols and signals.row_count >= large_table_rows:
        return True, "large_complex_table"

    # 8. Has continuation marker
    if signals.has_continuation_marker:
        return True, "has_continuation"

    return False, "fast_sufficient"


# -------------------------
# Statistics
# -------------------------


@dataclass
class TriageStatistics:
    """Statistics from triage operation."""

    total_candidates: int = 0
    skip_count: int = 0
    cheap_path_count: int = 0
    vlm_required_count: int = 0

    # Breakdown by reason
    skip_reasons: Dict[str, int] = field(default_factory=dict)
    vlm_reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def skip_ratio(self) -> float:
        return self.skip_count / max(self.total_candidates, 1)

    @property
    def vlm_ratio(self) -> float:
        return self.vlm_required_count / max(self.total_candidates, 1)


def compute_triage_statistics(
    triaged: List[Tuple[VisualCandidate, TriageResult]],
) -> TriageStatistics:
    """
    Compute statistics from triage results.

    Args:
        triaged: List of (candidate, result) tuples

    Returns:
        TriageStatistics with counts and breakdowns
    """
    stats = TriageStatistics(total_candidates=len(triaged))

    skip_reasons: Counter = Counter()
    vlm_reasons: Counter = Counter()

    for candidate, result in triaged:
        if result.decision == TriageDecision.SKIP:
            stats.skip_count += 1
            skip_reasons[result.reason] += 1
        elif result.decision == TriageDecision.CHEAP_PATH:
            stats.cheap_path_count += 1
        elif result.decision == TriageDecision.VLM_REQUIRED:
            stats.vlm_required_count += 1
            vlm_reasons[result.reason] += 1

    stats.skip_reasons = dict(skip_reasons)
    stats.vlm_reasons = dict(vlm_reasons)

    return stats


__all__ = [
    # Types
    "TriageConfig",
    "DocumentContext",
    "TriageStatistics",
    # Main functions
    "triage_visual",
    "triage_batch",
    # Filters
    "filter_by_decision",
    "get_vlm_candidates",
    "get_skip_candidates",
    "get_cheap_path_candidates",
    # Table complexity
    "should_escalate_to_accurate",
    # Helpers
    "is_in_margin_zone",
    "compute_area_ratio",
    "compute_triage_statistics",
]
