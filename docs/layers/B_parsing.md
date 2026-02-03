# Layer B: PDF Parsing

## Purpose

`B_parsing/` converts raw PDF files into a structured `DocumentGraph` representation that all downstream layers consume. It handles layout detection, reading order, table extraction, figure detection, section identification, and confidence scoring. The output is a unified document model with pages, text blocks, tables, and images annotated with bounding boxes and content roles.

This layer contains 31 modules organized into functional groups.

See also: [Pipeline Overview](../architecture/01_overview.md) | [Data Flow](../architecture/02_data_flow.md)

---

## Key Modules

### Core Parsing

| Module | Description |
|--------|-------------|
| `B01_pdf_to_docgraph.py` | Main entry point. Converts PDF to `DocumentGraph` using Unstructured.io or PyMuPDF as backend. Orchestrates layout detection, table extraction, and section identification. |
| `B02_doc_graph.py` | Data models: `DocumentGraph`, `Page`, `TextBlock`, `Table`, `ImageBlock`. Defines `ContentRole` (BODY_TEXT, SECTION_HEADER, PAGE_HEADER, PAGE_FOOTER, TABLE_CAPTION, TABLE_CELL), `ImageType` (FIGURE, FLOWCHART, CHART, DIAGRAM), and `TableType` (DATA_GRID, GLOSSARY, LAYOUT_GRID). `BoundingBox` is imported from `A_core/A01_domain_models.py`. |

### Table Extraction

| Module | Description |
|--------|-------------|
| `B03_table_extractor.py` | Docling TableFormer integration for high-accuracy table structure recognition (95-98% TEDS accuracy). |
| `B27_table_validation.py` | Post-extraction table structure validation (row/column consistency, header detection). |
| `B28_docling_backend.py` | Docling PDF parsing backend implementation. |
| `B29_column_detection.py` | Column span detection for multi-column tables. |

### Layout Detection

| Module | Description |
|--------|-------------|
| `B04_column_ordering.py` | XY-Cut++ layout detection for determining correct reading order across multi-column pages. |
| `B30_xy_cut_ordering.py` | Recursive page partitioning into horizontal/vertical strips for multi-column reading order. |
| `B18_layout_models.py` | `PageLayout`, `LayoutType`, `BlockClass` data models. |
| `B19_layout_analyzer.py` | Spatial analysis for layout classification. |
| `B25_legacy_ordering.py` | Fallback top-to-bottom, left-to-right reading order for simple layouts. |

### Section Detection

| Module | Description |
|--------|-------------|
| `B05_section_detector.py` | Multi-signal section detection (font size, bold, numbering patterns, keyword matching). |
| `B08_eligibility_parser.py` | Structured parsing of inclusion/exclusion criteria sections into criterion objects. |

### Confidence and Assertions

| Module | Description |
|--------|-------------|
| `B06_confidence.py` | Feature-based confidence scoring for parsed elements. |
| `B07_negation.py` | Negation detection for extracted text (identifies negated assertions). |

### Figure Extraction

| Module | Description |
|--------|-------------|
| `B09_pdf_native_figures.py` | PyMuPDF-based native figure extraction (raster and vector images). |
| `B24_native_figure_extraction.py` | Coordinate-based figure region extraction with overlap detection and size filtering. |
| `B10_caption_detector.py` | Caption detection adjacent to figures and tables. |
| `B15_caption_extractor.py` | Caption text extraction and association with visual elements. |
| `B12_visual_pipeline.py` | Unified visual extraction pipeline orchestrating detection, rendering, and caption assignment. |
| `B13_visual_detector.py` | VLM-based visual type detection (figure vs. table vs. decorative). |
| `B14_visual_renderer.py` | Figure rendering and image generation from PDF pages. |
| `B31_vlm_detector.py` | VLM (Vision Language Model) detector integration. |

### Triage and Resolution

| Module | Description |
|--------|-------------|
| `B16_triage.py` | Routing decision for visual elements: `SKIP` (noise), `CHEAP_PATH` (minimal processing), `VLM_REQUIRED` (full VLM enrichment). Based on area ratio, grid structure, and caption presence. |
| `B11_extraction_resolver.py` | Multi-page continuation detection and resolution. |
| `B17_document_resolver.py` | Document-level conflict resolution. |
| `B22_doclayout_detector.py` | YOLO-based layout detection. |

### Utilities

| Module | Description |
|--------|-------------|
| `B20_zone_expander.py` | Bounding box zone expansion for context capture. |
| `B21_filename_generator.py` | Output filename generation for extracted assets. |
| `B23_text_helpers.py` | Text normalization and cleaning utilities for parsed content. |
| `B26_repetition_inference.py` | Repetitive header/footer detection and removal. |

---

## Public Interfaces

### DocumentGraph

Top-level container for a parsed PDF. All downstream layers consume this model.

```python
class DocumentGraph(BaseModel):
    doc_id: str
    pages: Dict[int, Page] = Field(default_factory=dict)
    metadata: Dict[str, str] = Field(default_factory=dict)

    def get_page(self, page_num: int) -> Page: ...
    def iter_images(self) -> Iterator[ImageBlock]: ...
    def iter_linear_blocks(self, skip_header_footer: bool = True) -> Iterator[TextBlock]: ...
    def iter_tables(self, table_type: Optional[TableType] = None) -> Iterator[Table]: ...
```

### Page

```python
class Page(BaseModel):
    number: int              # 1-based
    width: float
    height: float
    blocks: list[TextBlock]
    tables: list[Table]
    images: list[ImageBlock]
```

### TextBlock

Atomic text unit with content role classification and spatial coordinates.

```python
class TextBlock(BaseModel):     # frozen=True
    id: str
    text: str
    page_num: int               # 1-based
    reading_order_index: int    # 0-based per page
    role: ContentRole           # BODY_TEXT, SECTION_HEADER, etc.
    bbox: BoundingBox
```

### ContentRole Enum

`BODY_TEXT`, `SECTION_HEADER`, `PAGE_HEADER`, `PAGE_FOOTER`, `TABLE_CAPTION`, `TABLE_CELL`

### BaseParser Interface

From `A_core/A02_interfaces.py`:

```python
class BaseParser(ABC):
    def parse(self, file_path: str) -> DocumentModel: ...
```

---

## Usage Patterns

### Standard Parsing

```python
from B_parsing.B01_pdf_to_docgraph import PDFToDocGraphParser

parser = PDFToDocGraphParser(config=config)
doc_graph = parser.parse("path/to/document.pdf")

# Iterate over text blocks in reading order
for block in doc_graph.iter_linear_blocks():
    print(f"[p{block.page_num}] {block.role}: {block.text[:80]}")

# Access tables
for table in doc_graph.iter_tables():
    print(f"Table on page {table.page_num}: {table.num_rows}x{table.num_cols}")
```

### Visual Extraction Pipeline

The visual pipeline runs in stages:

1. **Native extraction** (`B09`/`B24`): PyMuPDF extracts embedded images from the PDF.
2. **Layout detection** (`B22`): YOLO model identifies figure and table regions.
3. **Triage** (`B16`): Each visual candidate is routed to SKIP, CHEAP_PATH, or VLM_REQUIRED.
4. **Caption association** (`B10`/`B15`): Captions are matched to their visual elements by spatial proximity.
5. **VLM enrichment** (C layer): Candidates requiring VLM analysis are sent to Claude Vision.

### Reading Order

Multi-column PDFs use the XY-Cut++ algorithm (`B04`/`B30`) to determine correct reading order. The algorithm recursively partitions the page into horizontal and vertical strips, then assigns `reading_order_index` to each `TextBlock`. The legacy algorithm (`B25`) serves as a fallback.

### Table Extraction

Tables are extracted via Docling's TableFormer model (`B03`/`B28`), which achieves 95-98% TEDS accuracy. Post-extraction validation (`B27`) checks row/column consistency. Column span detection (`B29`) handles merged cells.
