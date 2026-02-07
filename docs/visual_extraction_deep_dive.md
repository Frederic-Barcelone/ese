# Visual Extraction -- Tables & Figures

> **Pipeline version**: v0.8

---

## 1. Overview

```
PDF Pages -> B_parsing (PyMuPDF + Docling + Layout Analysis)
  -> Visual Triage (B16/B22): SKIP | CHEAP_PATH (Haiku OCR) | VLM_REQUIRED (Sonnet)
  -> VLM Analysis (C10, C19): classify + extract + caption match
  -> Export (J_export): PNG files + JSON metadata
```

---

## 2. PDF Parsing & Image Detection

### DocumentGraph (B02)

B_parsing converts each PDF into a `DocumentGraph` -- pages, text blocks, tables, images with bounding boxes in PDF points, ordered by reading sequence.

### PyMuPDF Extraction (B09/B24)

Extracts embedded images (inline + XObject) at original resolution, deduplicates across pages, provides bounding box coordinates. Each page yields: raster images, vector graphics, table regions, and decorative elements.

---

## 3. Visual Triage System

Three-path routing based on visual complexity:

| Path | Criteria | Processing |
|------|----------|------------|
| **SKIP** | Decorative: logos, icons, borders, <50x50px | None |
| **CHEAP_PATH** | Text-heavy, simple photos, single-panel | Haiku OCR (fast) |
| **VLM_REQUIRED** | Multi-panel, flowcharts, charts, complex tables | Sonnet vision (high quality) |

### Detection Methods

**DocLayout-YOLO (B22):** 144 DPI, ~0.5s/page. Detects figure/table/caption with confidence scores. Runs as pre-filter.

**VLM Detection (B31):** Claude Vision fallback. 150 DPI, max 1568px. Normalized bboxes converted to PDF points. Sonnet tier.

---

## 4. Table Extraction

### Docling TableFormer (Primary)

95-98% TEDS accuracy. Two modes: fast (simple tables) and accurate (merged cells, complex layouts). Output: `TableStructure` with headers, cells (row/col indices), merged cells, per-cell data types.

### Layout-Aware Detection (B19)

Detects page patterns: FULL, 2COL, 2COL_FULLBOT, FULLTOP_2COL. Two-phase column-aware detection with iterative bounding box refinement.

### VLM Fallback

When Docling fails: Claude Vision extracts headers, rows, cells from high-res image. Handles complex formatting, rotation, tables embedded in figures. `vlm_table_extraction` call type (Sonnet).

---

## 5. Figure Extraction

### Native Extraction

PyMuPDF extracts raster images (JPEG, PNG, TIFF) at original resolution. Vector graphics rendered at configurable DPI. Tight crop around content.

### Resolution Levels

Detection: 144 DPI. VLM analysis: 150 DPI, max 1568px. Export: original resolution. >5MB auto-compressed for VLM.

### Flowchart Detection

Classified by boxes/diamonds/arrows, routed to C17 for node/edge extraction.

### Chart Classification

Subtypes: bar, line, scatter, Kaplan-Meier, forest plot, heatmap -- each with specialized extraction.

---

## 6. Caption Management

### Detection

Regex patterns ("Figure 1:", "Table 2."), proximity (text above/below), font analysis (smaller/italic), sub-figure patterns ("(a)", "(b)").

### Matching Priority

DocLayout-YOLO bbox -> spatial proximity -> reference number -> cross-page. Each caption tracks source: `pdf_text`, `ocr`, or `vlm`.

---

## 7. VLM Analysis

### Processing

1. Base64 encode, auto-detect media type, compress >5MB, scale to max 1568px
2. Classify: table/figure/chart/flowchart/photograph/diagram (confidence >= 0.7)
3. Extract: summary, structured data, key findings, entity mentions

### Flowchart Analysis

Sonnet extracts nodes (text, type, position), edges (source-target, conditions), treatment phases, drug names. `flowchart_analysis` call type.

---

## 8. Content Extraction Modes

| Mode | Model | Use Case | call_type |
|------|-------|----------|-----------|
| OCR text fallback | Haiku | Simple text from images | `ocr_text_fallback` |
| Chart data | Sonnet | Series, axes, values | `chart_analysis` |
| Table structure | Sonnet | Headers, rows, merged cells | `vlm_table_extraction` |
| Flowchart graph | Sonnet | Nodes, edges, decisions, drugs | `flowchart_analysis` |

---

## 9. Visual Reference Resolution

Detects inline references ("Table 1", "Fig. 3a", "Figures 4-6") and links to extracted visuals. `VisualReference`: `visual_type`, `reference_number`, `reference_text`, `page_num`.

---

## 10. Export & Output

Naming: `{document}_figure_page{N}_{idx}.png` / `{document}_flowchart_page{N}_{idx}.png`.

### JSON Metadata

```json
{
  "visual_id": "fig_001",
  "visual_type": "figure",
  "page_range": [3],
  "bbox_pts": [72.0, 150.0, 540.0, 450.0],
  "caption": "Figure 1. Study design overview",
  "caption_provenance": "pdf_text",
  "classification_confidence": 0.95,
  "vlm_title": "Randomized controlled trial design",
  "vlm_description": "Flow diagram showing patient screening...",
  "image_path": "document_figure_page3_1.png"
}
```

### Table Structure JSON

```json
{
  "table_structure": {
    "headers": ["Treatment", "N", "Response Rate", "p-value"],
    "rows": [["Drug A", "150", "72%", "0.001"], ["Placebo", "148", "35%", "ref"]],
    "source": "docling"
  }
}
```

Bounding boxes in PDF points (absolute) and per-page for multi-page visuals.
