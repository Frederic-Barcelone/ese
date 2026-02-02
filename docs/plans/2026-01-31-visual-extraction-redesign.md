# Visual Extraction System Redesign

**Date:** 2026-01-31
**Status:** Approved for Implementation

## Overview

Complete redesign of table and figure extraction to achieve SOTA quality for clinical documents (scientific papers, protocols, advisory boards, regulatory documents).

## Requirements

- **All document types**: Scientific papers, protocols, advisory boards, regulatory docs
- **Maximum quality**: VLM on every meaningful visual
- **Full context**: Parsed references with cross-document mentions
- **300 DPI**: Fixed high resolution, tight cropping
- **VLM classification**: Let Claude Vision decide type with confidence

## Architecture

### Pipeline Stages

```
STAGE 1: Detection + FAST Structure (Docling v2.71.0)
    │
    ▼
STAGE 2: Rendering + Caption Extraction (PyMuPDF + PDF text)
    │
    ▼
STAGE 3: Triage + VLM Enrichment (Claude Sonnet Vision)
    │
    ▼
STAGE 4: Document-Level Resolution (Cross-reference + merge)
```

### Stage 1: Detection + FAST Structure

- Layout detection for figure regions
- TableFormer FAST for initial table structure
- Flag tables for ACCURATE re-run based on complexity signals
- All bboxes stored in PDF points (canonical coordinate space)

**ACCURATE escalation triggers:**
- Multi-page tables
- Deep header stacks (3+ levels)
- Many merged cells (>5 or >10% of cells)
- Low token coverage (<70%)
- VLM flags misparsed (>80% confidence)
- Large complex tables (8+ columns, 15+ rows)

### Stage 2: Rendering + Caption Extraction

- Point-based padding (6-12pt sides, 36-72pt for captions)
- Multisource caption extraction:
  1. PDF text blocks (preferred for born-digital)
  2. OCR fallback if no reliable PDF text
- Early continuation detection:
  - "(continued)" / "(cont.)" markers
  - Repeated header structure
  - Same reference across pages
  - Matching column geometry
- Render at 200-300 DPI (adaptive based on size)
- Caption provenance tracking: {pdf_text | ocr | vlm}

### Stage 3: Triage + VLM Enrichment

**SKIP (no VLM):**
- Tiny area (<2% page)
- Repeated graphics (3+ pages)
- Margin zone, no caption

**VLM PATH:**
- Has caption-like text
- Docling flagged as table
- Dense grid structure
- Referenced in body text
- Flagged for ACCURATE re-run
- Large uncaptioned (>10% page)

**VLM tasks:**
1. Classify: table vs figure (with confidence)
2. Parse reference from caption
3. Validate/correct caption text
4. Validate Docling structure, flag if misparsed
5. Trigger ACCURATE re-run if needed
6. Detect multi-page continuation

### Stage 4: Document-Level Resolution

1. Scan body text for mentions
2. Parse references from body text
3. Reconcile (caption wins, body fills missing)
4. Link mentions to visuals
5. Infer section context
6. Merge multi-page visuals
7. Deduplicate overlapping detections

## Data Models

### ExtractedVisual (Unified Model)

```python
class ExtractedVisual(BaseModel):
    # Identity
    visual_id: str
    visual_type: VisualType  # table, figure, other
    confidence: float

    # Location (multi-page support)
    page_range: List[int]
    bbox_pts_per_page: List[PageLocation]

    # Caption
    caption_text: Optional[str]
    caption_provenance: CaptionProvenance  # pdf_text, ocr, vlm
    caption_bbox_pts: Optional[Tuple[float, float, float, float]]

    # Reference
    reference: Optional[VisualReference]

    # Rendered image
    image_base64: str
    image_format: str = "png"
    render_dpi: int

    # Table-specific
    docling_table: Optional[TableStructure]
    validated_table: Optional[TableStructure]
    table_extraction_mode: Optional[Literal["fast", "accurate"]]

    # Relationships
    relationships: VisualRelationships

    # Provenance
    extraction_method: str
    source_file: str
    extracted_at: datetime
```

### VisualReference

```python
class VisualReference(BaseModel):
    raw_string: str          # "Figure 2-4"
    type_label: str          # "Figure", "Table", "Exhibit"
    numbers: List[int]       # [2, 3, 4]
    is_range: bool
    source: Literal["caption", "body_text", "vlm"]
```

### VisualRelationships

```python
class VisualRelationships(BaseModel):
    text_mentions: List[TextMention]
    section_context: Optional[str]
    continued_from: Optional[str]
    continues_to: Optional[str]
```

## File Structure

```
corpus_metadata/
├── A_core/
│   └── A13_visual_models.py          # ExtractedVisual, VisualReference, etc.
├── B_parsing/
│   ├── B12_visual_pipeline.py        # Main orchestrator
│   ├── B13_visual_detector.py        # Stage 1: Detection
│   ├── B14_visual_renderer.py        # Stage 2: Rendering
│   ├── B15_caption_extractor.py      # Multisource caption extraction
│   ├── B16_triage.py                 # Stage 3: Triage logic
│   └── B17_document_resolver.py      # Stage 4: Cross-reference
├── C_generators/
│   └── C16_vlm_visual_enrichment.py  # VLM enrichment for visuals
└── tests/
    └── test_visual_extraction/
        ├── test_visual_models.py
        ├── test_detector.py
        ├── test_renderer.py
        ├── test_caption_extractor.py
        ├── test_triage.py
        ├── test_document_resolver.py
        └── test_vlm_enrichment.py
```

## Configuration

```yaml
visual_extraction:
  # Stage 1
  table_mode_default: "fast"
  accurate_escalation:
    header_depth_threshold: 3
    merged_cell_threshold: 5
    token_coverage_threshold: 0.70
    large_table_cols: 8
    large_table_rows: 15

  # Stage 2
  render_dpi: 300
  padding_pts:
    sides: 12
    caption_zone: 72
  caption_sources: ["pdf_text", "ocr"]

  # Stage 3
  triage:
    skip_area_ratio: 0.02
    repeat_threshold: 3
    vlm_area_threshold: 0.10

  # Stage 4
  merge_multipage: true
  dedupe_overlap_threshold: 0.70
```

## Implementation Order

1. **A13_visual_models.py** - Data models (foundation)
2. **B15_caption_extractor.py** - Multisource caption extraction
3. **B16_triage.py** - Triage logic
4. **B13_visual_detector.py** - Detection with FAST/ACCURATE tiering
5. **B14_visual_renderer.py** - Point-based rendering
6. **C16_vlm_visual_enrichment.py** - VLM enrichment
7. **B17_document_resolver.py** - Cross-reference resolution
8. **B12_visual_pipeline.py** - Main orchestrator
9. **Integration** - Wire into existing pipeline

## Success Criteria

- Correct caption-to-visual linking (>95% accuracy)
- Correct reference parsing (>98% accuracy)
- Correct type classification (>95% accuracy)
- No missed tables/figures in test corpus
- Multi-page handling works correctly
- 300 DPI images with tight cropping
