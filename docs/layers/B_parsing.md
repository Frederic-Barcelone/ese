# Layer B: PDF Parsing

## Purpose

`B_parsing/` converts raw PDFs into a structured `DocumentGraph` that all downstream layers consume. Handles layout detection, reading order, table extraction, figure detection, and section identification. 31 modules.

See also: [Pipeline Overview](../architecture/01_overview.md) | [Data Flow](../architecture/02_data_flow.md)

---

## Key Modules

### Core Parsing

| Module | Description |
|--------|-------------|
| `B01_pdf_to_docgraph.py` | `PDFToDocGraphParser`. PDF to `DocumentGraph` via Unstructured.io or PyMuPDF. |
| `B02_doc_graph.py` | `DocumentGraph`, `Page`, `TextBlock`, `Table`, `ImageBlock`, `ContentRole`, `ImageType`, `TableType`. |

### Table Extraction

| Module | Description |
|--------|-------------|
| `B03_table_extractor.py` | Docling TableFormer (95-98% TEDS accuracy). |
| `B27_table_validation.py` | Row/column consistency, header detection. |
| `B28_docling_backend.py` | Docling PDF parsing backend. |
| `B29_column_detection.py` | Multi-column table span detection. |

### Layout Detection

| Module | Description |
|--------|-------------|
| `B04_column_ordering.py` | XY-Cut++ for multi-column reading order. |
| `B30_xy_cut_ordering.py` | Recursive page partitioning into strips. |
| `B18_layout_models.py` | `PageLayout`, `LayoutType`, `BlockClass`. |
| `B19_layout_analyzer.py` | Spatial layout classification. Uses `resolve_model_tier("layout_analysis")`. |
| `B25_legacy_ordering.py` | Fallback top-to-bottom reading order. |

### Section and Confidence

| Module | Description |
|--------|-------------|
| `B05_section_detector.py` | Multi-signal section detection (font, bold, numbering, keywords). |
| `B06_confidence.py` | `UnifiedConfidenceCalculator`. Feature-based confidence scoring. |
| `B07_negation.py` | Negation detection in extracted text. |
| `B08_eligibility_parser.py` | Inclusion/exclusion criteria parsing. |

### Figure Extraction

| Module | Description |
|--------|-------------|
| `B09_pdf_native_figures.py` | PyMuPDF native figure extraction (raster/vector). |
| `B24_native_figure_extraction.py` | Coordinate-based region extraction with overlap detection. |
| `B10_caption_detector.py` | Caption detection near figures/tables. |
| `B15_caption_extractor.py` | Caption extraction and association. |
| `B12_visual_pipeline.py` | Unified visual extraction orchestrator. |
| `B13_visual_detector.py` | VLM-based visual type detection. |
| `B14_visual_renderer.py` | Figure rendering from PDF pages. |
| `B31_vlm_detector.py` | VLM detector. Uses `resolve_model_tier("vlm_detection")`. |

### Triage and Resolution

| Module | Description |
|--------|-------------|
| `B16_triage.py` | Visual routing: `SKIP`, `CHEAP_PATH`, `VLM_REQUIRED`. |
| `B11_extraction_resolver.py` | Multi-page continuation detection. |
| `B17_document_resolver.py` | Document-level conflict resolution. |
| `B22_doclayout_detector.py` | YOLO layout detection. Uses `resolve_model_tier("vlm_visual_enrichment")`. |

### Utilities

| Module | Description |
|--------|-------------|
| `B20_zone_expander.py` | Bounding box expansion for context. |
| `B21_filename_generator.py` | Output filename generation. |
| `B23_text_helpers.py` | Text normalization for parsed content. |
| `B26_repetition_inference.py` | Header/footer detection and removal. |

---

## Public Interfaces

### DocumentGraph

```python
class DocumentGraph(BaseModel):
    doc_id: str
    pages: Dict[int, Page]
    metadata: Dict[str, str]

    def get_page(self, page_num: int) -> Page: ...
    def iter_images(self) -> Iterator[ImageBlock]: ...
    def iter_linear_blocks(self, skip_header_footer: bool = True) -> Iterator[TextBlock]: ...
    def iter_tables(self, table_type: Optional[TableType] = None) -> Iterator[Table]: ...
```

### TextBlock

```python
class TextBlock(BaseModel):     # frozen=True
    id: str
    text: str
    page_num: int               # 1-based
    reading_order_index: int    # 0-based per page
    role: ContentRole           # BODY_TEXT, SECTION_HEADER, etc.
    bbox: BoundingBox
```

**ContentRole**: `BODY_TEXT`, `SECTION_HEADER`, `PAGE_HEADER`, `PAGE_FOOTER`, `TABLE_CAPTION`, `TABLE_CELL`

---

## Usage Patterns

### Standard Parsing

```python
from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser
parser = PDFToDocGraphParser(config=config)
doc_graph = parser.parse("path/to/document.pdf")
for block in doc_graph.iter_linear_blocks():
    print(f"[p{block.page_num}] {block.role}: {block.text[:80]}")
```

### Visual Pipeline Stages

1. **Native extraction** (B09/B24): PyMuPDF extracts embedded images.
2. **Layout detection** (B22): YOLO identifies figure/table regions.
3. **Triage** (B16): Route to SKIP, CHEAP_PATH, or VLM_REQUIRED.
4. **Caption association** (B10/B15): Match captions by spatial proximity.
5. **VLM enrichment** (C layer): Claude Vision analysis.

### Reading Order

Multi-column PDFs use XY-Cut++ (B04/B30). Legacy algorithm (B25) as fallback.

### Table Extraction

Docling TableFormer (B03/B28) at 95-98% TEDS accuracy. Validation (B27) and column span detection (B29).
