# corpus_metadata/A_core/A13_visual_models.py
"""
Data models for the visual extraction pipeline.

Provides Pydantic models for:
- ExtractedVisual: Unified model for tables and figures
- VisualReference: Parsed reference (e.g., "Table 1", "Figure 2-4")
- VisualRelationships: Cross-references and document context
- Triage and caption extraction models
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


# -------------------------
# Visual Type Enums
# -------------------------


class VisualType(str, Enum):
    """Type of extracted visual element."""

    TABLE = "table"
    FIGURE = "figure"
    OTHER = "other"  # Decorative, logo, separator


class CaptionProvenance(str, Enum):
    """Source of caption text extraction."""

    PDF_TEXT = "pdf_text"  # Native PDF text extraction (preferred)
    OCR = "ocr"  # OCR from rendered image
    VLM = "vlm"  # VLM extraction/correction


class TriageDecision(str, Enum):
    """Triage routing decision for visual processing."""

    SKIP = "skip"  # Noise - don't process further
    CHEAP_PATH = "cheap_path"  # Simple visual - minimal processing
    VLM_REQUIRED = "vlm"  # Needs full VLM enrichment


class TableExtractionMode(str, Enum):
    """TableFormer extraction mode."""

    FAST = "fast"
    ACCURATE = "accurate"


class ReferenceSource(str, Enum):
    """Source of reference parsing."""

    CAPTION = "caption"
    BODY_TEXT = "body_text"
    VLM = "vlm"


# -------------------------
# Geometry Models
# -------------------------


class PageLocation(BaseModel):
    """Location on a specific page with bbox in PDF points."""

    page_num: int = Field(..., ge=1, description="1-based page number")
    bbox_pts: Tuple[float, float, float, float] = Field(
        ..., description="Bounding box in PDF points (x0, y0, x1, y1)"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _validate_bbox(self):
        x0, y0, x1, y1 = self.bbox_pts
        if x0 < 0 or y0 < 0:
            raise ValueError("Bounding box coordinates cannot be negative.")
        if x1 < x0 or y1 < y0:
            raise ValueError(f"Invalid bbox geometry: ({x0}, {y0}, {x1}, {y1})")
        return self


# -------------------------
# Reference Models
# -------------------------


class VisualReference(BaseModel):
    """
    Parsed reference like 'Figure 2-4' or 'Table 1'.

    Supports:
    - Single references: "Table 1" -> numbers=[1]
    - Ranges: "Figure 2-4" -> numbers=[2, 3, 4], is_range=True
    - Letters: "Figure 1A" -> numbers=[1], suffix="A"
    """

    raw_string: str = Field(..., description="Original reference text")
    type_label: str = Field(..., description="Type label (Figure, Table, Exhibit)")
    numbers: List[int] = Field(..., description="Parsed reference numbers")
    is_range: bool = Field(default=False, description="True if range syntax (e.g., 2-4)")
    suffix: Optional[str] = Field(default=None, description="Letter suffix (e.g., 'A' in 'Figure 1A')")
    source: ReferenceSource = Field(..., description="Where reference was parsed from")

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _validate_reference(self):
        if not self.numbers:
            raise ValueError("VisualReference.numbers must contain at least one number.")
        if not self.type_label.strip():
            raise ValueError("VisualReference.type_label must be non-empty.")
        return self


class TextMention(BaseModel):
    """A reference to a visual found in body text."""

    text: str = Field(..., description="The mention text (e.g., 'see Figure 2')")
    page_num: int = Field(..., ge=1, description="Page where mention appears")
    char_offset: int = Field(..., ge=0, description="Character offset in page text")
    reference: Optional[VisualReference] = Field(
        default=None, description="Parsed reference from mention"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


# -------------------------
# Relationship Models
# -------------------------


class VisualRelationships(BaseModel):
    """Links to other document elements and cross-references."""

    text_mentions: List[TextMention] = Field(
        default_factory=list, description="Body text references to this visual"
    )
    section_context: Optional[str] = Field(
        default=None, description="Section name (e.g., 'Results', 'Methods')"
    )
    continued_from: Optional[str] = Field(
        default=None, description="visual_id of previous part (multi-page)"
    )
    continues_to: Optional[str] = Field(
        default=None, description="visual_id of next part (multi-page)"
    )

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Table Structure Models
# -------------------------


class TableCellStructure(BaseModel):
    """A cell in the table structure."""

    text: str
    row_index: int = Field(..., ge=0)
    col_index: int = Field(..., ge=0)
    row_span: int = Field(default=1, ge=1)
    col_span: int = Field(default=1, ge=1)
    is_header: bool = False

    model_config = ConfigDict(frozen=True, extra="forbid")


class TableStructure(BaseModel):
    """Structured table data from Docling or VLM."""

    headers: List[List[str]] = Field(
        default_factory=list, description="Header rows (supports multi-level)"
    )
    rows: List[List[str]] = Field(default_factory=list, description="Data rows")
    cells: List[TableCellStructure] = Field(
        default_factory=list, description="Physical cells with spans"
    )

    # Quality metrics
    token_coverage: float = Field(
        default=1.0, ge=0.0, le=1.0, description="% of bbox area with mapped tokens"
    )
    structure_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in structure accuracy"
    )

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Triage Models
# -------------------------


class TriageResult(BaseModel):
    """Result of visual triage decision."""

    decision: TriageDecision
    reason: str = Field(..., description="Reason for decision")
    confidence: float = Field(..., ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True, extra="forbid")


class TableComplexitySignals(BaseModel):
    """Signals indicating table complexity for FAST/ACCURATE decision."""

    merged_cell_count: int = Field(default=0, ge=0)
    header_depth: int = Field(default=1, ge=1, description="Nested header levels")
    token_coverage_ratio: float = Field(
        default=1.0, ge=0.0, le=1.0, description="% of area with mapped tokens"
    )
    column_count: int = Field(default=0, ge=0)
    row_count: int = Field(default=0, ge=0)
    spans_multiple_pages: bool = False
    has_continuation_marker: bool = Field(
        default=False, description="Has '(continued)' or '(cont.)' marker"
    )
    vlm_flagged_misparsed: bool = Field(
        default=False, description="VLM indicated structure is wrong"
    )

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Caption Models
# -------------------------


class CaptionCandidate(BaseModel):
    """A potential caption found near a visual."""

    text: str
    bbox_pts: Tuple[float, float, float, float] = Field(
        ..., description="Bounding box in PDF points"
    )
    provenance: CaptionProvenance
    position: Literal["above", "below", "left", "right"]
    distance_pts: float = Field(..., ge=0.0, description="Distance from visual edge")
    confidence: float = Field(..., ge=0.0, le=1.0)
    parsed_reference: Optional[VisualReference] = None

    model_config = ConfigDict(extra="forbid")


class CaptionSearchZones(BaseModel):
    """Search zones relative to visual bbox (in PDF points)."""

    above: float = Field(default=72.0, ge=0.0, description="Points above visual")
    below: float = Field(default=72.0, ge=0.0, description="Points below visual")
    left: float = Field(default=36.0, ge=0.0, description="Points left of visual")
    right: float = Field(default=36.0, ge=0.0, description="Points right of visual")

    model_config = ConfigDict(frozen=True, extra="forbid")


# -------------------------
# Visual Candidate (Pre-Triage)
# -------------------------


class VisualCandidate(BaseModel):
    """
    A detected visual before triage and VLM enrichment.

    Contains raw detection signals for triage decision.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Source detection
    source: Literal["docling", "native_raster", "native_vector", "layout_model", "vlm_detection", "layout_aware", "doclayout_yolo"]
    docling_type: Optional[str] = Field(
        default=None, description="Docling's classification if available"
    )
    confidence: float = Field(default=0.8, description="Detection confidence score")

    # Location
    page_num: int = Field(..., ge=1)
    bbox_pts: Tuple[float, float, float, float]
    page_width_pts: float = Field(..., gt=0)
    page_height_pts: float = Field(..., gt=0)

    # Detection signals for triage
    area_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Visual area / page area"
    )
    image_hash: Optional[str] = Field(
        default=None, description="SHA1 hash for deduplication"
    )
    has_nearby_caption: bool = False
    has_grid_structure: bool = False
    is_referenced_in_text: bool = False
    in_margin_zone: bool = False
    needs_accurate_rerun: bool = Field(
        default=False, description="Flagged for TableFormer ACCURATE"
    )

    # Initial caption (if detected in Stage 2)
    caption_candidate: Optional[CaptionCandidate] = None

    # VLM detection fields (when source="vlm_detection")
    vlm_label: Optional[str] = Field(
        default=None, description="Label from VLM detection (e.g., 'Table 1', 'Figure 2A')"
    )
    vlm_caption_snippet: Optional[str] = Field(
        default=None, description="Caption snippet from VLM detection"
    )

    # Layout-aware detection fields (when source="layout_aware")
    layout_code: Optional[str] = Field(
        default=None, description="Page layout pattern (e.g., 'full', '2col', '2col-fullbot')"
    )
    position_code: Optional[str] = Field(
        default=None, description="Visual position in layout (e.g., 'L', 'R', 'F')"
    )
    layout_filename: Optional[str] = Field(
        default=None, description="Generated filename with layout encoding"
    )

    # Continuation signals
    continuation_markers: List[str] = Field(
        default_factory=list, description="Detected continuation cues"
    )

    model_config = ConfigDict(extra="forbid")

    @property
    def area(self) -> float:
        """Calculate visual area in square points."""
        x0, y0, x1, y1 = self.bbox_pts
        return (x1 - x0) * (y1 - y0)

    @property
    def page_area(self) -> float:
        """Calculate page area in square points."""
        return self.page_width_pts * self.page_height_pts


# -------------------------
# Extracted Visual (Final Output)
# -------------------------


class ExtractedVisual(BaseModel):
    """
    Unified model for extracted tables and figures.

    This is the final output after all pipeline stages:
    - Detection (Docling)
    - Rendering (PyMuPDF)
    - VLM enrichment (Claude Vision)
    - Document-level resolution
    """

    # Identity
    visual_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    visual_type: VisualType
    confidence: float = Field(..., ge=0.0, le=1.0, description="VLM classification confidence")

    # Location (supports multi-page)
    page_range: List[int] = Field(..., min_length=1, description="Pages containing this visual")
    bbox_pts_per_page: List[PageLocation] = Field(
        ..., min_length=1, description="Bbox on each page"
    )

    # Caption
    caption_text: Optional[str] = None
    caption_provenance: Optional[CaptionProvenance] = None
    caption_bbox_pts: Optional[Tuple[float, float, float, float]] = None

    # Reference
    reference: Optional[VisualReference] = None

    # Rendered image
    image_base64: str = Field(..., description="Base64-encoded PNG image")
    image_format: str = "png"
    render_dpi: int = Field(default=300, ge=72, le=600)

    # Table-specific (only populated for visual_type == TABLE)
    docling_table: Optional[TableStructure] = Field(
        default=None, description="Raw Docling extraction"
    )
    validated_table: Optional[TableStructure] = Field(
        default=None, description="Post-VLM corrected structure"
    )
    table_extraction_mode: Optional[TableExtractionMode] = None

    # Relationships
    relationships: VisualRelationships = Field(default_factory=VisualRelationships)

    # Provenance
    extraction_method: str = Field(
        ..., description="Pipeline path (docling+vlm, docling_only, vlm_only)"
    )
    source_file: str
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

    # Triage metadata
    triage_decision: Optional[TriageDecision] = None
    triage_reason: Optional[str] = None

    # Layout metadata (when using layout-aware detection)
    layout_code: Optional[str] = Field(
        default=None, description="Page layout pattern (e.g., 'full', '2col', '2col-fullbot')"
    )
    position_code: Optional[str] = Field(
        default=None, description="Visual position in layout (e.g., 'L', 'R', 'F')"
    )
    layout_filename: Optional[str] = Field(
        default=None, description="Generated filename with layout encoding"
    )

    model_config = ConfigDict(extra="forbid")

    @property
    def is_multipage(self) -> bool:
        """Check if visual spans multiple pages."""
        return len(self.page_range) > 1

    @property
    def is_table(self) -> bool:
        """Check if visual is a table."""
        return self.visual_type == VisualType.TABLE

    @property
    def is_figure(self) -> bool:
        """Check if visual is a figure."""
        return self.visual_type == VisualType.FIGURE

    @property
    def primary_page(self) -> int:
        """Get the first page number."""
        return self.page_range[0]

    @property
    def primary_bbox_pts(self) -> Tuple[float, float, float, float]:
        """Get the bbox on the first page."""
        return self.bbox_pts_per_page[0].bbox_pts

    @model_validator(mode="after")
    def _validate_page_consistency(self):
        if len(self.page_range) != len(self.bbox_pts_per_page):
            raise ValueError(
                f"page_range length ({len(self.page_range)}) must match "
                f"bbox_pts_per_page length ({len(self.bbox_pts_per_page)})"
            )
        for i, loc in enumerate(self.bbox_pts_per_page):
            if loc.page_num != self.page_range[i]:
                raise ValueError(
                    f"bbox_pts_per_page[{i}].page_num ({loc.page_num}) must match "
                    f"page_range[{i}] ({self.page_range[i]})"
                )
        return self

    @model_validator(mode="after")
    def _validate_table_fields(self):
        if self.visual_type == VisualType.TABLE:
            # Tables should have at least docling_table or validated_table
            pass  # Allow empty for now; VLM may populate later
        else:
            # Non-tables shouldn't have table-specific fields populated
            if self.docling_table is not None or self.validated_table is not None:
                raise ValueError(
                    "docling_table and validated_table should only be set for TABLE visual_type"
                )
        return self


# -------------------------
# VLM Response Models
# -------------------------


class VLMClassificationResult(BaseModel):
    """VLM classification of a visual."""

    visual_type: VisualType
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: Optional[str] = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class VLMTableValidation(BaseModel):
    """VLM validation result for table structure."""

    is_misparsed: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    corrected_structure: Optional[TableStructure] = None

    model_config = ConfigDict(extra="forbid")


class VLMEnrichmentResult(BaseModel):
    """Full VLM enrichment result for a visual."""

    classification: VLMClassificationResult
    parsed_reference: Optional[VisualReference] = None
    extracted_caption: Optional[str] = None
    table_validation: Optional[VLMTableValidation] = None
    is_continuation: bool = False
    continuation_of_reference: Optional[str] = Field(
        default=None, description="Reference number this continues (e.g., 'Table 1')"
    )

    model_config = ConfigDict(extra="forbid")
