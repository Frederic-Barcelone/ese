# Visual Extraction -- Tables & Figures

> **Date**: February 2026
> **Pipeline version**: v0.8

Step-by-step explanation of how the ESE pipeline extracts, classifies, and enriches tables, figures, flowcharts, and charts from PDF documents.

---

## Table of Contents

1. [Overview](#1-overview)
2. [PDF Parsing & Image Detection](#2-pdf-parsing--image-detection)
3. [Visual Triage System](#3-visual-triage-system)
4. [Table Extraction](#4-table-extraction)
5. [Figure Extraction](#5-figure-extraction)
6. [Caption Management](#6-caption-management)
7. [VLM Analysis](#7-vlm-analysis)
8. [Content Extraction Modes](#8-content-extraction-modes)
9. [Visual Reference Resolution](#9-visual-reference-resolution)
10. [Export & Output](#10-export--output)

---

## 1. Overview

Visual extraction transforms embedded images, tables, and figures in PDFs into structured, searchable metadata. The pipeline handles everything from simple data tables to complex clinical flowcharts.

```
PDF Pages
    |
    v
PDF Parsing (B_parsing)
    |
    +-- Native Image Extraction (PyMuPDF)
    +-- Table Detection (Docling TableFormer)
    +-- Layout Analysis (B19, VLM)
    |
    v
Visual Triage (B22)
    |
    +-- SKIP        --> decorative, logos, icons
    +-- CHEAP_PATH  --> simple OCR (Haiku)
    +-- VLM_REQUIRED --> complex analysis (Sonnet)
    |
    v
VLM Analysis (C10, C19)
    |
    +-- Classification (table vs figure vs chart vs flowchart)
    +-- Content Extraction (structured data, node/edge graphs)
    +-- Caption Matching (link captions to visuals)
    |
    v
Export (J_export)
    |
    +-- PNG files + JSON metadata
    +-- Table structure JSON
    +-- Flowchart node/edge graphs
```

---

## 2. PDF Parsing & Image Detection

### DocumentGraph Construction

The B_parsing layer converts each PDF into a `DocumentGraph` -- a structured representation of pages, text blocks, tables, and images:

- Each page becomes a `PageNode` with text blocks, images, and layout metadata
- Text blocks are ordered by reading sequence (top-to-bottom, left-to-right)
- Images are extracted with bounding box coordinates in PDF points

### PyMuPDF Native Figure Extraction

PyMuPDF (fitz) handles primary image extraction:

- Extracts embedded images directly from the PDF object stream
- Preserves original image resolution and format (JPEG, PNG, TIFF)
- Provides bounding box coordinates for each extracted image
- Handles both inline images and XObject images
- Extracts images referenced by multiple pages only once

### Page-Level Image Identification

For each page, the parser identifies:

- **Embedded images**: Raster images stored in the PDF
- **Vector graphics**: Paths and shapes that may form figures
- **Table regions**: Areas with grid-like structure
- **Decorative elements**: Logos, headers, footers, page borders

---

## 3. Visual Triage System

### Three-Path Routing

Not every image deserves expensive VLM analysis. The triage system routes visuals to the appropriate processing path:

**SKIP** -- Decorative elements that carry no informational value:
- Logos and brand marks (small, typically in headers/footers)
- Page decorations and borders
- Icons and bullet graphics
- Images below minimum size threshold (< 50x50 pixels)

**CHEAP_PATH** -- Simple visuals that need only basic OCR:
- Text-heavy images with minimal structure
- Simple photographs with captions
- Single-panel images without complex layouts
- Processing: Haiku OCR text fallback (fast, inexpensive)

**VLM_REQUIRED** -- Complex visuals requiring vision model analysis:
- Multi-panel figures with subplots
- Clinical flowcharts and decision trees
- Data charts (bar, line, scatter, Kaplan-Meier)
- Complex tables with merged cells or spanning headers
- Processing: Sonnet vision analysis (slower, higher quality)

### DocLayout-YOLO Detection (B22)

A fast ML-based detector identifies visual elements on each page:

- Uses the DocLayout-YOLO model from HuggingFace
- Inference at 144 DPI (~0.5s per page)
- Detects categories: `figure`, `figure_caption`, `table`, `table_caption`
- Provides bounding boxes with confidence scores
- Runs as a pre-filter before VLM analysis

### VLM Detection (B31)

When DocLayout-YOLO is insufficient, Claude Vision detects visuals:

- Renders pages at 150 DPI for VLM input
- Scales to max 1568px on longest dimension
- Returns normalized bounding boxes (0-1 range) converted to PDF points
- Handles multi-column layouts and full-width visuals
- Tracks multi-page visuals (continuation detection)
- Uses `vlm_detection` call type (Sonnet tier)

---

## 4. Table Extraction

### Docling TableFormer

The primary table extraction engine uses Docling's TableFormer model:

- **Accuracy**: 95-98% TEDS (Tree-Edit Distance Score) on benchmark datasets
- Two modes:
  - **Fast mode**: Faster processing, suitable for simple tables
  - **Accurate mode**: Better handling of complex layouts, merged cells

### Table Structure Parsing

Extracted tables are represented as `TableStructure` objects:

- **Headers**: Column and row header detection with spanning support
- **Cells**: Individual cell content with row/column indices
- **Merged cells**: Cells spanning multiple rows or columns
- **Data types**: Numeric, text, percentage, date detection per cell

### Layout-Aware Table Detection (B19)

The layout analyzer provides additional table context:

- Detects page layout patterns: FULL (single column), 2COL (two columns), 2COL_FULLBOT, FULLTOP_2COL
- Identifies table zones within the page layout
- Grid-based detection with visual overlays for debugging
- Two-phase column-aware detection for complex layouts
- Iterative bounding box refinement for precise table boundaries

### VLM Table Extraction

When Docling fails or produces low-confidence results:

- Falls back to Claude Vision for table structure extraction
- Renders table region at high resolution
- Extracts headers, rows, and cell content from the image
- Particularly useful for:
  - Tables with complex formatting (colors, shading)
  - Rotated or skewed tables
  - Tables embedded in figures
- Uses `vlm_table_extraction` call type (Sonnet tier)

---

## 5. Figure Extraction

### Native PDF Figure Extraction

PyMuPDF extracts figures directly from the PDF:

- Raster images: JPEG, PNG, TIFF extracted at original resolution
- Vector graphics: Rendered to raster at configurable DPI
- Multi-page figures: Detected via continuation markers
- Cropping: Tight crop around actual figure content (excluding whitespace)

### Resolution Handling

Images are processed at appropriate resolutions:

- **Detection**: 144 DPI (fast, for DocLayout-YOLO)
- **VLM analysis**: 150 DPI scaled to max 1568px
- **Export**: Original resolution preserved
- **Compression**: Images exceeding 5MB are compressed before VLM calls

### Flowchart Detection

Flowcharts receive special handling:

- Detected by presence of boxes, diamonds, and connecting arrows
- Classified as "flowchart" rather than generic "figure"
- Routed to specialized flowchart analysis pipeline (C17)
- Node/edge extraction for structured pathway data

### Chart Type Classification

Charts are classified into subtypes:

- Bar charts, line graphs, scatter plots
- Kaplan-Meier survival curves
- Forest plots
- Heatmaps
- Each type has specialized extraction logic

---

## 6. Caption Management

### Caption Detection

Captions are identified through multiple strategies:

- **Regex patterns**: "Figure 1:", "Table 2.", "Fig. 3" etc.
- **Proximity detection**: Text blocks immediately above or below visual regions
- **Font analysis**: Captions often use smaller or italic fonts
- **Structured patterns**: "(a)", "(b)" for sub-figure captions

### Caption-to-Visual Matching

Matching captions to their visuals uses a priority system:

1. **DocLayout-YOLO**: Caption bounding boxes detected alongside figure/table boxes
2. **Spatial proximity**: Nearest caption within threshold distance
3. **Reference number matching**: "Table 1" caption matched to 1st table on page
4. **Cross-page handling**: Captions on one page matched to visuals on adjacent pages

### CaptionProvenance Tracking

Each caption tracks its detection source:

| Source | Description |
|--------|-------------|
| `pdf_text` | Extracted from PDF text layer |
| `ocr` | Extracted via OCR from image |
| `vlm` | Extracted by Vision LLM analysis |

---

## 7. VLM Analysis

### Vision LLM Processing

Claude Vision (Sonnet) provides deep analysis of complex visuals:

**Image Preprocessing:**
- Base64 encoding of image data
- Media type detection: JPEG, PNG, GIF, WebP
- Automatic compression for images > 5MB
- Resolution scaling for optimal VLM input (max 1568px)

**Classification:**
- Visual type: table, figure, chart, flowchart, photograph, diagram
- Confidence score (0.0-1.0) with minimum threshold of 0.7
- Reasoning text explaining classification decision

**Content Extraction:**
- Summary generation: 2-4 sentence description
- Structured data extraction based on visual type
- Key findings and notable data points
- Entity mentions (diseases, drugs, genes referenced in the visual)

### Flowchart Node/Edge Analysis

For flowcharts, Sonnet extracts a structured graph:

- **Nodes**: Text content, node type (action/decision/assessment), position
- **Edges**: Source-target connections with condition labels
- **Phases**: Treatment phases (induction, maintenance, relapse)
- **Drugs**: Drug names extracted from action nodes
- Uses `flowchart_analysis` call type (Sonnet tier)

---

## 8. Content Extraction Modes

### OCR Text Fallback (Haiku)

For simple images that just need text extraction:

- Image sent to Claude Haiku with OCR-focused prompt
- Returns plain text content from the image
- Fast and inexpensive (~$1/MTok input)
- Uses `ocr_text_fallback` call type

### Chart Data Extraction (Sonnet)

For data charts requiring numerical extraction:

- Identifies chart type (bar, line, scatter, etc.)
- Extracts data series with labels and values
- Reads axis labels, titles, and legends
- Returns structured data suitable for recreation
- Uses `chart_analysis` call type

### Table VLM Extraction (Sonnet)

For tables that Docling cannot parse:

- Full table structure extraction from image
- Header detection with spanning support
- Row/column content with data type inference
- Handles complex layouts: merged cells, nested headers, rotated text
- Uses `vlm_table_extraction` call type

### Flowchart Analysis (Sonnet)

For clinical decision trees and algorithms:

- Complete node/edge graph extraction
- Condition/action classification per node
- Drug, dose, and duration extraction per action node
- Decision point identification with branching logic
- Uses `flowchart_analysis` call type

---

## 9. Visual Reference Resolution

### Parsing Inline References

The pipeline detects references to visuals within document text:

- Patterns: "Table 1", "Figure 2", "Fig. 3a", "Figures 4-6"
- Case-insensitive matching with common abbreviations (Fig., Tab.)
- Sub-figure references: "Figure 1a", "Figure 1(b)"

### Linking Mentions to Extracted Visuals

References are linked to their corresponding extracted visuals:

1. Parse reference number from text mention
2. Match to visual with same reference number
3. Handle ambiguity (multiple "Table 1" across sections)
4. Create bidirectional links: text mention <-> visual metadata

### VisualReference Model

Each parsed reference contains:

- `visual_type`: "Table" or "Figure"
- `reference_number`: The number or label (e.g., "1", "2a")
- `reference_text`: The full reference text (e.g., "Table 1")
- `page_num`: Page where the reference appears

---

## 10. Export & Output

### PNG Files

Extracted visuals are saved as PNG files in the document output folder:

- Naming convention: `{document}_figure_page{N}_{idx}.png`
- Flowcharts: `{document}_flowchart_page{N}_{idx}.png`
- Original resolution preserved where possible
- Compressed for visuals exceeding size limits

### JSON Metadata

Visual metadata is exported as structured JSON:

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

Tables include structured content alongside the visual:

```json
{
  "table_structure": {
    "headers": ["Treatment", "N", "Response Rate", "p-value"],
    "rows": [
      ["Drug A", "150", "72%", "0.001"],
      ["Placebo", "148", "35%", "ref"]
    ],
    "merged_cells": [],
    "source": "docling"
  }
}
```

### Bounding Box Coordinates

All visuals include precise location data:

- **PDF points**: Absolute coordinates in the PDF coordinate system
- **Normalized**: 0-1 range relative to page dimensions
- **Per-page**: Multi-page visuals have bbox for each page
- Used for: visual highlighting, caption matching, reading order
