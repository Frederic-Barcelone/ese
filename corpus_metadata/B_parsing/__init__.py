# corpus_metadata/B_parsing/__init__.py
"""
PDF parsing and document structure extraction layer for the ESE pipeline.

This package provides comprehensive PDF parsing capabilities for clinical trial
and medical documents, converting PDFs into structured DocumentGraph representations
with text blocks, tables, figures, and metadata. It implements SOTA multi-column
layout detection, visual extraction pipelines, and confidence scoring frameworks.

Key Components:
    - B01_pdf_to_docgraph: PDF to DocumentGraph parser with SOTA column ordering
    - B02_doc_graph: Pydantic models for DocumentGraph, Page, TextBlock, Table, ImageBlock
    - B03_table_extractor: Table extraction using Docling TableFormer
    - B04_column_ordering: SOTA XY-Cut++ column layout detection and reading order
    - B05_section_detector: Layout-aware section detection
    - B06_confidence: Feature-based confidence scoring framework
    - B07_negation: Negation and assertion detection
    - B08_eligibility_parser: Eligibility criteria logical expression parser
    - B09_pdf_native_figures: Native PDF figure extraction (raster/vector)
    - B10_caption_detector: Caption detection and column layout analysis
    - B11_extraction_resolver: Deterministic figure/table extraction resolver
    - B12_visual_pipeline: Visual extraction pipeline orchestrator
    - B13_visual_detector: Visual detector with FAST/ACCURATE tiering
    - B14_visual_renderer: Visual renderer with point-based padding
    - B15_caption_extractor: Multisource caption extraction
    - B16_triage: Visual triage logic for VLM processing decisions
    - B17_document_resolver: Document-level resolution and deduplication
    - B18_layout_models: Layout data structures for visual extraction
    - B19_layout_analyzer: VLM-based layout analyzer
    - B20_zone_expander: Whitespace-based bbox computation
    - B21_filename_generator: Layout-aware filename generation
    - B22_doclayout_detector: DocLayout-YOLO visual detector
    - B23_text_helpers: Text normalization and cleaning utilities
    - B24_native_figure_extraction: Native figure extraction integration
    - B25_legacy_ordering: Legacy block ordering fallback
    - B26_repetition_inference: Header/footer repetition inference
    - B27_table_validation: Table validation and false positive filtering
    - B28_docling_backend: Docling-based table extraction backend
    - B29_column_detection: Column detection utilities for SOTA ordering
    - B30_xy_cut_ordering: XY-Cut++ ordering algorithms
    - B31_vlm_detector: VLM-assisted visual detection

Dependencies:
    - A_core: Domain models and interfaces
    - fitz (PyMuPDF): PDF rendering and extraction
    - docling: TableFormer-based table extraction
    - anthropic: Claude API for VLM features
"""
