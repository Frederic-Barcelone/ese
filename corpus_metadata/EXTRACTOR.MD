# ESE — Entity & Structure Extraction Pipeline

A production-grade 6-layer pipeline for extracting structured metadata from clinical trial and medical PDF documents. Focused on rare disease research.

**Extraction Capabilities:**

| Category | What is Extracted | Ontology Codes |
|----------|-------------------|----------------|
| Abbreviations | Acronyms with long-form definitions | — |
| Diseases | Rare diseases, conditions, syndromes | ICD-10, SNOMED, MONDO, ORPHA |
| Drugs | Approved, investigational, compounds | RxNorm, DrugBank, MeSH, ChEMBL |
| Genes | Symbols, aliases, disease associations | HGNC, Entrez, Ensembl, OMIM, UniProt |
| Pharma companies | Sponsors, manufacturers | — |
| Authors | Names, roles, affiliations | ORCID |
| Citations | References, identifiers | PMID, DOI, NCT, EudraCT |
| Feasibility | Eligibility, endpoints, patient journey | — |
| Recommendations | Guideline recommendations, evidence levels | — |
| Care pathways | Treatment algorithms, decision trees | — |
| Document metadata | Classification, dates, identifiers | — |
| Figures | Vision LLM analysis (flowcharts, KM plots) | — |
| Tables | Docling TableFormer + VLM validation | — |

---

## Architecture Overview

```
+---------------------------------------------------------------------------------+
|                           ORCHESTRATOR (orchestrator.py)                         |
|                              16-Stage Pipeline                                  |
|                                                                                 |
|  +----------+   +----------+   +----------+   +----------+   +----------+      |
|  | A_core   |-->| B_parsing|-->|C_generate|-->|D_validate|-->|E_normalize|     |
|  | Models   |   | PDF->Doc |   | Candidates|  | LLM/Rules|  | Enrich    |     |
|  | 25 files |   | 32 files |   | 35 files |  | 4 files  |  | 18 files  |     |
|  +----------+   +----------+   +----------+   +----------+   +----------+      |
|       |              |              |              |              |             |
|       v              v              v              v              v             |
|  +----------+   +----------+   +----------+   +----------+   +----------+      |
|  |H_pipeline|   |I_extract |   | J_export |   | Z_utils  |   | G_config |     |
|  | 4 files  |   | 2 files  |   | 4 files  |   | 11 files |   | 2 files  |     |
|  +----------+   +----------+   +----------+   +----------+   +----------+      |
|                                                                                 |
|  +----------+   +----------+                                                    |
|  |F_evaluate|   | K_tests  |                                                    |
|  | 3 files  |   | 59 tests |                                                    |
|  +----------+   +----------+                                                    |
+---------------------------------------------------------------------------------+
```

### Design Philosophy

| Layer | Goal | Strategy |
|-------|------|----------|
| **Generators (C)** | High Recall | Exhaustive extraction; noise acceptable |
| **Validation (D)** | High Precision | Claude LLM filters false positives |
| **Normalization (E)** | Standardization | Map to ontologies, deduplicate |

---

## Pipeline Stages

The orchestrator runs 16 sequential stages:

| # | Stage | Layer | Description |
|---|-------|-------|-------------|
| 1 | PDF Parsing | B | PDF → DocumentGraph via Unstructured.io + layout detection |
| 2 | Abbreviation Generation | C | Schwartz-Hearst, FlashText, glossary extraction |
| 3 | LLM Validation | D | Claude batch validation + PASO heuristics |
| 4 | Normalization | E | Deduplication, term mapping, disambiguation |
| 5 | Disease Detection | C+E | MONDO/Orphanet/scispacy → PubTator enrichment |
| 6 | Gene Detection | C+E | HGNC/Orphadata/scispacy → PubTator enrichment |
| 7 | Drug Detection | C+E | ChEMBL/RxNorm/FDA → PubTator enrichment |
| 8 | Pharma Detection | C | FlashText company name matching |
| 9 | Author Detection | C | Pattern-based name + credential extraction |
| 10 | Citation Detection | C+E | Identifier extraction + external validation |
| 11 | Feasibility Extraction | C+E | Rule-based + LLM + multi-NER enrichment |
| 12 | Care Pathway Extraction | C | Flowchart graph extraction from figures |
| 13 | Recommendation Extraction | C | LLM + VLM guideline recommendation extraction |
| 14 | Visual Extraction | B | 4-stage pipeline: detect → render → triage → resolve |
| 15 | Document Metadata | C | File info, PDF properties, LLM classification |
| 16 | Export & Summary | J | JSON export per entity type + metrics |

### Extraction Presets

```yaml
extraction_pipeline:
  preset: "standard"   # Default preset
```

| Preset | What it Extracts |
|--------|-----------------|
| `standard` | Abbreviations, diseases, drugs, genes, document metadata |
| `all` | Every entity type and visual extraction |
| `minimal` | Abbreviations only |
| `entities_only` | Drugs, diseases, genes, abbreviations |
| `clinical_entities` | Entities + feasibility + recommendations |
| `drugs_only` | Drug entities only |
| `diseases_only` | Disease entities only |
| `genes_only` | Gene entities only |
| `abbreviations_only` | Abbreviation entities only |
| `feasibility_only` | Feasibility data only |
| `metadata_only` | Document metadata only |
| `images_only` | Figure extraction only |
| `tables_only` | Table extraction only |

---

## Directory Structure

```
corpus_metadata/
│
├── orchestrator.py                          # Main 16-stage pipeline entry point
├── orchestrator_utils.py                    # StageTimer, warning suppression
│
├── A_core/                                  # LAYER 0: FOUNDATION (25 files)
│   ├── A00_logging.py                       # Centralized logging (ColoredFormatter, PipelineLogger)
│   ├── A01_domain_models.py                 # Core Pydantic schemas (Candidate, ExtractedEntity)
│   ├── A02_interfaces.py                    # ABCs (BaseExtractor, BaseCandidateGenerator, BaseEnricher)
│   ├── A03_provenance.py                    # Run ID, git hash, document fingerprint, prompt hashing
│   ├── A04_heuristics_config.py             # PASO heuristics (whitelists, blacklists, confidence)
│   ├── A05_disease_models.py                # Disease models (ICD-10, SNOMED, ORPHA, MONDO)
│   ├── A06_drug_models.py                   # Drug models (RxCUI, NDC, DrugBank, DevelopmentPhase)
│   ├── A07_feasibility_models.py            # Feasibility (eligibility, screening, epidemiology)
│   ├── A08_document_metadata_models.py      # Document classification, dates, identifiers
│   ├── A09_pharma_models.py                 # Pharma company models
│   ├── A10_author_models.py                 # Author/investigator models (roles, affiliations)
│   ├── A11_citation_models.py               # Citation models (PMID, DOI, NCT validation)
│   ├── A12_exceptions.py                    # Exception hierarchy (ESEPipelineError → 9 subtypes)
│   ├── A13_ner_models.py                    # Unified NER models (EntityCategory, NEREntity)
│   ├── A13_visual_models.py                 # Visual models (ExtractedVisual, TriageResult, VLM)
│   ├── A14_extraction_result.py             # Universal extraction result contract
│   ├── A15_domain_profile.py                # Domain profiles (nephrology, oncology, pulmonology)
│   ├── A16_pipeline_metrics.py              # Pipeline metrics (generation, validation, export)
│   ├── A17_care_pathway_models.py           # Care pathway nodes, edges, taper schedules
│   ├── A18_recommendation_models.py         # Guideline recommendations, evidence levels
│   ├── A19_gene_models.py                   # Gene models (HGNC, Entrez, Ensembl, OMIM, UniProt)
│   ├── A20_unicode_utils.py                 # Unicode normalization (mojibake, hyphens, SF keys)
│   ├── A21_clinical_criteria.py             # Computable eligibility criteria (lab, severity)
│   └── A22_logical_expressions.py           # Logical expression trees for criteria (AND/OR/NOT)
│
├── B_parsing/                               # LAYER 1: DOCUMENT STRUCTURE (32 files)
│   ├── B01_pdf_to_docgraph.py               # Main PDF parser (Unstructured.io + layout detection)
│   ├── B02_doc_graph.py                     # DocumentGraph, Page, TextBlock, Table, ImageBlock
│   ├── B03_table_extractor.py               # Table extraction orchestrator (95-98% TEDS)
│   ├── B04_column_ordering.py               # SOTA XY-Cut++ multi-column layout detection
│   ├── B05_section_detector.py              # Section/heading detection (Methods, Results, etc.)
│   ├── B06_confidence.py                    # Feature-based confidence scoring
│   ├── B07_negation.py                      # Negation/assertion detection
│   ├── B08_eligibility_parser.py            # Logical expression parser for eligibility criteria
│   ├── B09_pdf_native_figures.py             # Raster + vector figure extraction (PyMuPDF)
│   ├── B10_caption_detector.py              # Caption detection + column layout inference
│   ├── B11_extraction_resolver.py           # Multi-signal figure/table resolution
│   ├── B12_visual_pipeline.py               # 4-stage visual extraction orchestrator
│   ├── B13_visual_detector.py               # Docling FAST/ACCURATE detection tiering
│   ├── B14_visual_renderer.py               # PyMuPDF rendering at 200-400 DPI
│   ├── B15_caption_extractor.py             # Multisource caption extraction
│   ├── B16_triage.py                        # SKIP / CHEAP_PATH / VLM_REQUIRED routing
│   ├── B17_document_resolver.py             # Body text scanning, multi-page merge, dedup
│   ├── B18_layout_models.py                 # LayoutPattern, VisualPosition, VisualZone
│   ├── B19_layout_analyzer.py               # Claude Vision layout analysis
│   ├── B20_zone_expander.py                 # Whitespace-based bbox computation
│   ├── B21_filename_generator.py            # Layout-aware filename generation
│   ├── B22_doclayout_detector.py            # DocLayout-YOLO fast visual detection
│   ├── B23_text_helpers.py                  # Text normalization, pattern utilities
│   ├── B24_native_figure_extraction.py      # Native extraction integration with DocumentGraph
│   ├── B25_legacy_ordering.py               # Legacy single/two-column ordering (fallback)
│   ├── B26_repetition_inference.py          # Header/footer detection via repetition
│   ├── B27_table_validation.py              # Table false positive filtering
│   ├── B28_docling_backend.py               # Docling TableFormer backend
│   ├── B29_column_detection.py              # Gutter detection, column clustering
│   ├── B30_xy_cut_ordering.py               # XY-Cut++ core recursive algorithm
│   └── B31_vlm_detector.py                  # Claude Vision visual detection
│
├── C_generators/                            # LAYER 2: CANDIDATE GENERATION (35 files)
│   │
│   │ # ── Core Strategy Generators ──
│   ├── C00_strategy_identifiers.py          # Database IDs (OMIM, DOI, NCT, ORCID, EudraCT)
│   ├── C01_strategy_abbrev.py               # Schwartz-Hearst syntax patterns
│   ├── C02_strategy_regex.py                # Rigid patterns (trial IDs, DOIs, doses)
│   ├── C03_strategy_layout.py               # Spatial extraction (headers, footers, columns)
│   ├── C04_strategy_flashtext.py            # FlashText lexicon matching (600K+ terms) + scispacy
│   ├── C05_strategy_glossary.py             # Glossary table extractor
│   ├── C06_strategy_disease.py              # Disease detection (MONDO, Orphanet, scispacy)
│   ├── C07_strategy_drug.py                 # Drug detection (ChEMBL, RxNorm, FDA, scispacy)
│   ├── C08_strategy_feasibility.py          # Rule-based feasibility extraction
│   ├── C09_strategy_document_metadata.py    # Document metadata extraction
│   ├── C10_vision_image_analysis.py         # Vision LLM image analysis (CONSORT, KM plots)
│   ├── C11_llm_feasibility.py               # LLM-based feasibility extraction
│   ├── C12_guideline_recommendation_extractor.py  # Guideline recommendation extraction
│   ├── C13_strategy_author.py               # Author/investigator detection
│   ├── C14_strategy_citation.py             # Citation/reference detection
│   ├── C15_vlm_table_extractor.py           # VLM table structure extraction
│   ├── C16_strategy_gene.py                 # Gene detection (HGNC, Orphadata)
│   ├── C17_flowchart_graph_extractor.py     # Flowchart → care pathway graph extraction
│   ├── C18_strategy_pharma.py               # Pharma company detection
│   ├── C19_vlm_visual_enrichment.py         # Claude Vision enrichment (type, caption, ref)
│   │
│   │ # ── Support & Filter Modules ──
│   ├── C20_abbrev_patterns.py               # Abbreviation pattern constants
│   ├── C21_noise_filters.py                 # Noise filtering (blacklists, obvious noise)
│   ├── C22_lexicon_loaders.py               # FlashText lexicon loading mixin
│   ├── C23_inline_definition_detector.py    # Inline abbreviation definition detection
│   ├── C24_disease_fp_filter.py             # Disease confidence scoring + FP filter
│   ├── C25_drug_fp_filter.py                # Drug false positive filter
│   ├── C26_drug_fp_constants.py             # Drug FP exclusion sets (~16 curated sets)
│   ├── C27_feasibility_patterns.py          # Feasibility regex patterns
│   ├── C28_feasibility_fp_filter.py         # Feasibility false positive filter
│   ├── C29_feasibility_prompts.py           # LLM feasibility prompt templates
│   ├── C30_feasibility_response_parser.py   # LLM response parsing mixin
│   ├── C31_recommendation_patterns.py       # Recommendation pattern constants + prompts
│   ├── C32_recommendation_llm.py            # LLM recommendation extraction mixin
│   ├── C33_recommendation_vlm.py            # VLM recommendation extraction mixin
│   └── C34_gene_fp_filter.py                # Gene false positive filter
│
├── D_validation/                            # LAYER 3: LLM VERIFICATION (4 files)
│   ├── D01_prompt_registry.py               # Versioned prompts with hash tracking
│   ├── D02_llm_engine.py                    # Claude API client + batch validation
│   ├── D03_validation_logger.py             # JSONL audit trail
│   └── D04_quote_verifier.py                # Anti-hallucination quote verification
│
├── E_normalization/                         # LAYER 4: STANDARDIZATION (18 files)
│   ├── E01_term_mapper.py                   # Abbreviation long-form canonicalization
│   ├── E02_disambiguator.py                 # Context-based disambiguation (bag-of-words)
│   ├── E03_disease_normalizer.py            # Disease categorization + code extraction
│   ├── E04_pubtator_enricher.py             # PubTator3 API (MeSH, aliases) + caching
│   ├── E05_drug_enricher.py                 # PubTator3-based drug enrichment
│   ├── E06_nct_enricher.py                  # ClinicalTrials.gov NCT metadata
│   ├── E07_deduplicator.py                  # Quality-based abbreviation deduplication
│   ├── E08_epi_extract_enricher.py          # EpiExtract4GARD NER (LOC, EPI, STAT)
│   ├── E09_zeroshot_bioner.py               # ZeroShotBioNER (ADE, dosage, frequency)
│   ├── E10_biomedical_ner_all.py            # DistilBERT 84-entity-type NER
│   ├── E11_span_deduplicator.py             # Multi-source NER span deduplication
│   ├── E12_patient_journey_enricher.py      # Patient progression extraction
│   ├── E13_registry_enricher.py             # Clinical trial registry extraction
│   ├── E14_citation_validator.py            # External API citation validation (DOI, NCT, PMID)
│   ├── E15_genetic_enricher.py              # Genetic variant enrichment (HGVS, dbSNP)
│   ├── E16_drug_combination_parser.py       # Drug combination decomposition
│   ├── E17_entity_deduplicator.py           # Generic entity dedup (diseases, drugs, genes)
│   └── E18_gene_enricher.py                 # PubTator3-based gene enrichment
│
├── F_evaluation/                            # LAYER 5: METRICS & TESTING (3 files)
│   ├── F01_gold_loader.py                   # Load gold standard annotations (JSON/CSV)
│   ├── F02_scorer.py                        # Precision / Recall / F1 calculation
│   └── F03_evaluation_runner.py             # End-to-end evaluation runner
│
├── G_config/                                # CONFIGURATION (2 files)
│   ├── G01_config_keys.py                   # Type-safe configuration key enums
│   └── config.yaml                          # Central configuration (1,000+ lines)
│
├── H_pipeline/                              # PIPELINE COMPONENTS (4 files)
│   ├── H01_component_factory.py             # Factory for all pipeline components
│   ├── H02_abbreviation_pipeline.py         # Abbreviation pipeline (PASO A/B/C/D)
│   ├── H03_visual_integration.py            # Visual pipeline orchestrator integration
│   └── H04_merge_resolver.py               # Multi-source merge + deduplication
│
├── I_extraction/                            # ENTITY EXTRACTION (2 files)
│   ├── I01_entity_processors.py             # Unified entity processor (all entity types)
│   └── I02_feasibility_processor.py         # Feasibility + multi-NER enrichment
│
├── J_export/                                # OUTPUT HANDLERS (4 files)
│   ├── J01_export_handlers.py               # Central export manager
│   ├── J01a_entity_exporters.py             # Per-entity exporters (disease, drug, gene, ...)
│   ├── J01b_metadata_exporters.py           # Document metadata + care pathway export
│   └── J02_visual_export.py                 # Visual extraction JSON export
│
├── Z_utils/                                 # UTILITIES (11 files)
│   ├── Z01_api_client.py                    # Base API client (rate limiting, disk cache)
│   ├── Z02_text_helpers.py                  # Context extraction, dash normalization
│   ├── Z03_text_normalization.py            # Whitespace, dehyphenation, long-form cleanup
│   ├── Z04_image_utils.py                   # Image compression, OCR (pytesseract)
│   ├── Z05_path_utils.py                    # Base path resolution (CORPUS_BASE_PATH)
│   ├── Z06_usage_tracker.py                 # SQLite lexicon/API usage tracking
│   ├── Z07_console_output.py                # ANSI color formatting for terminal
│   ├── Z08_download_utils.py                # Shared lexicon download utilities
│   ├── Z09_download_gene_lexicon.py         # HGNC + Orphadata gene lexicon builder
│   ├── Z10_download_lexicons.py             # Public lexicon downloader (MONDO, ChEMBL)
│   └── Z11_entity_helpers.py                # Candidate → ExtractedEntity helpers
│
├── K_tests/                                 # TEST SUITE (59 test files)
│   ├── conftest.py                          # Shared fixtures (mock APIs, sample entities)
│   ├── K01-K07                              # Core model & provenance tests
│   ├── K08-K22                              # PDF parsing & visual pipeline tests
│   ├── K23-K29                              # Entity extraction pattern tests
│   ├── K30-K50                              # Infrastructure & pipeline tests
│   └── K51-K59                              # Utilities & advanced model tests
│
├── cache/                                   # Runtime cache
│   ├── api/                                 # General API response cache
│   ├── pubtator/                            # PubTator3 cache (7-day TTL)
│   └── clinicaltrials/                      # ClinicalTrials.gov cache (30-day TTL)
│
└── EXTRACTOR.MD                             # This documentation
```

**Module count: 140 implementation files + 59 tests = 199 total**

---

## Data Flow

```
PDF Input
    │
    v
┌──────────────────────────────────────────────────────┐
│ B_parsing: PDF → DocumentGraph                       │
│                                                      │
│  B01: Unstructured.io partition_pdf()                │
│  B04/B29/B30: XY-Cut++ multi-column layout           │
│  B05: Section detection (Methods, Results, ...)      │
│  B03/B28: Docling TableFormer (95-98% TEDS)          │
│  B09-B11: PDF-native figure extraction               │
│  B12-B17: Visual pipeline (detect→render→triage)     │
│  B22: DocLayout-YOLO fast detection                  │
│  B31: Claude Vision detection                        │
│  B18-B21: Layout-aware analysis + zone expansion     │
└──────────────────────────────────────────────────────┘
    │
    v
┌──────────────────────────────────────────────────────┐
│ C_generators: DocumentGraph → Candidates             │
│                                                      │
│  C00: Identifier extraction (OMIM, DOI, NCT, ORCID) │
│  C01: Schwartz-Hearst abbreviation patterns          │
│  C04: FlashText lexicon (600K+ terms) + scispacy     │
│  C06: Disease detection (MONDO, Orphanet)            │
│  C07: Drug detection (ChEMBL, RxNorm, FDA)           │
│  C08/C11: Feasibility (rule-based + LLM)             │
│  C10: Vision LLM image analysis                      │
│  C12: Guideline recommendation extraction            │
│  C13: Author/investigator detection                  │
│  C14: Citation/reference detection                   │
│  C16: Gene detection (HGNC, Orphadata)               │
│  C17: Flowchart → care pathway graph                 │
│  C18: Pharma company detection                       │
│  C19: VLM visual enrichment                          │
│  C24-C34: False positive filters per entity type     │
└──────────────────────────────────────────────────────┘
    │
    v
┌──────────────────────────────────────────────────────┐
│ D_validation: Candidates → Validated                 │
│                                                      │
│  D01: Versioned prompt registry (hash tracking)      │
│  D02: Claude API batch validation + retry logic      │
│  D03: JSONL audit trail                              │
│  D04: Anti-hallucination quote verification          │
│                                                      │
│  PASO Heuristic Shortcuts:                           │
│    A: Auto-approve statistical abbreviations         │
│    B: Country code blacklist                         │
│    C: Hyphenated abbreviation enrichment             │
│    D: LLM SF-only extraction for missing abbrevs     │
└──────────────────────────────────────────────────────┘
    │
    v
┌──────────────────────────────────────────────────────┐
│ E_normalization: Validated → Enriched                │
│                                                      │
│  E01-E02: Term mapping + disambiguation              │
│  E03: Disease categorization (ICD-10, SNOMED, MONDO) │
│  E04-E05: PubTator3 enrichment (diseases, drugs)     │
│  E06: ClinicalTrials.gov NCT metadata               │
│  E07/E11/E17: Multi-level deduplication              │
│  E08: EpiExtract4GARD NER (epidemiology)             │
│  E09: ZeroShotBioNER (ADE, dosage, frequency)        │
│  E10: DistilBERT biomedical NER (84 entity types)    │
│  E12: Patient journey extraction                     │
│  E13: Clinical trial registry enrichment             │
│  E14: External citation validation (DOI, NCT, PMID)  │
│  E15: Genetic variant enrichment (HGVS, dbSNP)       │
│  E16: Drug combination decomposition                 │
│  E18: Gene enrichment (PubTator3)                    │
└──────────────────────────────────────────────────────┘
    │
    v
Output Directory (named after PDF)
├── abbreviations_{stem}_{timestamp}.json
├── diseases_{stem}_{timestamp}.json
├── drugs_{stem}_{timestamp}.json
├── genes_{stem}_{timestamp}.json
├── pharma_{stem}_{timestamp}.json
├── authors_{stem}_{timestamp}.json
├── citations_{stem}_{timestamp}.json
├── feasibility_{stem}_{timestamp}.json
├── recommendations_{stem}_{timestamp}.json
├── figures_{stem}_{timestamp}.json
├── tables_{stem}_{timestamp}.json
├── visuals_{stem}_{timestamp}.json
├── metadata_{stem}_{timestamp}.json
├── {stem}_extracted_text_{timestamp}.txt
├── {stem}_{type}_page{N}_{index}.png
└── visual_images_{stem}/
```

---

## Table Extraction (Docling TableFormer)

```
PDF Input
    │
    v
┌──────────────────────────────────────────────────────┐
│ Docling TableFormer (B03, B28)                       │
│  FAST mode (default): Quick extraction               │
│  ACCURATE mode: Complex tables, merged cells         │
│  SuryaOCR integration for scanned documents          │
│  95-98% TEDS accuracy on complex tables              │
└──────────────────────────────────────────────────────┘
    │
    v
┌──────────────────────────────────────────────────────┐
│ Escalation Logic (B16)                               │
│  IF merged_cell_count > 5 OR                         │
│     header_depth > 3 OR                              │
│     token_coverage < 0.70:                           │
│     → Escalate to ACCURATE mode                      │
└──────────────────────────────────────────────────────┘
    │
    v
┌──────────────────────────────────────────────────────┐
│ Optional VLM Validation (C15, C19)                   │
│  Validate structure against image                    │
│  Correct misaligned cells                            │
│  Extract missing data                                │
└──────────────────────────────────────────────────────┘
```

### Graceful Dependency Handling

When Docling is not installed, table extraction is skipped:

```
[WARN] Docling not installed - table extraction DISABLED
       Install with: pip install docling docling-surya
```

---

## Visual Extraction Pipeline

A unified 4-stage pipeline (B12-B17, B18-B22, B31) for extracting tables and figures:

```
PDF Input
    │
    v
┌──────────────────────────────────────────────────────┐
│ STAGE 1: DETECTION                                   │
│                                                      │
│  Primary: Docling TableFormer FAST/ACCURATE (B13)    │
│  Alt:     DocLayout-YOLO fast detection (B22)        │
│  Alt:     Claude Vision detection (B31)              │
│  Figures: PyMuPDF native extraction (B09, B24)       │
│  Layout:  Claude Vision page analysis (B19)          │
│  All bboxes in PDF points (coordinate discipline)    │
└──────────────────────────────────────────────────────┘
    │
    v
┌──────────────────────────────────────────────────────┐
│ STAGE 2: RENDERING + CAPTION (B14, B15)              │
│                                                      │
│  Adaptive DPI: 200-400 based on visual size          │
│  Point-based padding (12pt sides, 72pt caption zone) │
│  Multisource caption: PDF text → OCR fallback        │
│  Reference parsing: "Table 1", "Figure 2-4"          │
│  Zone expansion: whitespace-based bbox (B20)         │
└──────────────────────────────────────────────────────┘
    │
    v
┌──────────────────────────────────────────────────────┐
│ STAGE 3: TRIAGE + VLM (B16, C19)                     │
│                                                      │
│  TRIAGE ROUTING:                                     │
│    SKIP:         Tiny logos, repeated headers, noise  │
│    CHEAP_PATH:   Simple visuals, heuristic classify   │
│    VLM_REQUIRED: Captioned, grid structure, refs      │
│                                                      │
│  VLM ENRICHMENT:                                     │
│    Type classification (table vs figure)             │
│    Table structure validation                        │
│    Caption extraction/correction                     │
│    Continuation detection                            │
└──────────────────────────────────────────────────────┘
    │
    v
┌──────────────────────────────────────────────────────┐
│ STAGE 4: RESOLUTION (B17)                            │
│                                                      │
│  Body text scanning for mentions                     │
│  Section context inference                           │
│  Multi-page visual merging                           │
│  Deduplication (70% overlap threshold)               │
│  Layout-aware filename generation (B21)              │
└──────────────────────────────────────────────────────┘
```

### Dependency Warnings

```
[WARN] Docling not installed - table extraction DISABLED
       Install with: pip install docling docling-surya

[WARN] SuryaOCR not available, using default OCR
       Install with: pip install docling-surya

[WARN] Anthropic SDK not installed - VLM enrichment DISABLED
       Classification will use heuristics only.

[WARN] DocLayout-YOLO not installed - fast detection unavailable
       Install with: pip install doclayout-yolo
```

---

## Core Models (A_core)

### Candidate → Entity Pattern

Every entity type follows a two-model pattern:

| Phase | Model | Purpose |
|-------|-------|---------|
| Generation | `*Candidate` | High-recall pre-validation output |
| Post-validation | `Extracted*` | High-precision enriched entity |

### A01 — Foundation Types

```python
class Candidate(BaseModel):
    """Raw extraction awaiting validation."""
    short_form: str                    # "TNF"
    long_form: Optional[str]           # "Tumor Necrosis Factor"
    evidence: List[EvidenceSpan]       # Source locations
    generator_type: GeneratorType      # SYNTAX_PATTERN, LEXICON_MATCH, etc.
    confidence_score: float            # 0.0 - 1.0
    context: str                       # Surrounding text

class ExtractedEntity(BaseModel):
    """Validated and enriched entity."""
    short_form: str
    long_form: Optional[str]
    status: ValidationStatus           # VALIDATED, REJECTED, AMBIGUOUS
    provenance: ProvenanceMetadata     # Version tracking
    normalized_value: Optional[Dict]   # Ontology mappings
```

### A19 — Gene Models

```python
class ExtractedGene(BaseModel):
    matched_text: str              # "BRCA1"
    hgnc_symbol: str               # Official HGNC symbol
    full_name: Optional[str]       # "BRCA1 DNA repair associated"
    hgnc_id: Optional[str]         # "HGNC:1100"
    entrez_id: Optional[str]       # "672"
    ensembl_id: Optional[str]      # "ENSG00000012048"
    omim_id: Optional[str]         # "113705"
    uniprot_id: Optional[str]      # "P38398"
    is_alias: bool = False
    locus_type: Optional[str]      # "gene with protein product"
    chromosome: Optional[str]      # "17q21.31"
    associated_diseases: List[GeneDiseaseLinkage]
```

### A18 — Guideline Recommendations

```python
class GuidelineRecommendation(BaseModel):
    recommendation_text: str           # Full recommendation text
    evidence_level: Optional[str]      # "Level A", "Grade 1"
    strength: Optional[str]            # "Strong", "Weak"
    source_guideline: Optional[str]    # Guideline name
    conditions: List[str]              # Target conditions
    interventions: List[str]           # Recommended interventions
```

### A13_visual — Visual Models

```python
class ExtractedVisual(BaseModel):
    visual_id: str
    visual_type: VisualType          # TABLE, FIGURE, OTHER
    confidence: float                 # VLM classification confidence
    page_range: List[int]
    bbox_pts_per_page: List[PageLocation]
    caption_text: Optional[str]
    caption_provenance: CaptionProvenance  # PDF_TEXT, OCR, VLM
    reference: Optional[VisualReference]   # Parsed "Table 1", etc.
    image_base64: str                 # Base64 PNG
    render_dpi: int                   # 200-400
```

### A21 — Clinical Criteria

```python
class LabCriterion(BaseModel):
    """Computable eligibility criterion with lab values."""
    analyte: str                      # "eGFR", "hemoglobin"
    operator: str                     # ">=", "<", "between"
    value: float
    unit: Optional[str]              # "mL/min/1.73m²"
    timepoints: List[LabTimepoint]

class LogicalExpression(BaseModel):
    """AND/OR/NOT tree for compound eligibility criteria."""
    operator: LogicalOperator
    children: List[CriterionNode]
```

### A12 — Exception Hierarchy

```
ESEPipelineError
├── ConfigurationError
├── ParsingError
├── ExtractionError
├── EnrichmentError
├── APIError
│   └── RateLimitError
├── ValidationError
├── CacheError
└── EvaluationError
```

---

## Lexicons (~617,000 terms)

| Lexicon | Terms | Source | Used By |
|---------|-------|--------|---------|
| Meta-Inventory | 65,048 | Clinical abbreviations | C04 |
| MONDO diseases | 97,313 | Unified disease ontology | C06 |
| Orphanet diseases | 9,468 | Rare diseases | C06 |
| ChEMBL drugs | 22,802 | Approved drugs | C07 |
| RxNorm terms | 131,961 | Drug vocabulary | C07 |
| UMLS biological | 97,308 | Biomedical terms | C04 |
| UMLS clinical | 19,937 | Clinical terms | C04 |
| Trial acronyms | 125,454 | ClinicalTrials.gov | C04 |
| HGNC genes | ~43,000 | Gene symbols + aliases | C16 |
| PRO scales | 301 | Patient-Reported Outcomes | C04 |
| Rare disease acronyms | 1,631 | Rare disease names | C04 |
| Abbreviation general | 5,392 | Curated abbreviations | C04 |

### Lexicon File Locations

Lexicon files are stored in `ouput_datasources/` at the project root:

| File | Size | Content |
|------|------|---------|
| `2025_08_lexicon_disease.json` | 4.9 MB | Unified disease ontology |
| `2025_08_lexicon_drug.json` | 19 MB | Approved drugs database |
| `2025_08_lexicon_medical_terms.json` | 16 MB | Medical terminology |
| `trial_acronyms_lexicon.json` | 80 MB | ClinicalTrials.gov trial acronyms |
| `pharma_companies_lexicon.json` | 8.7 KB | Pharmaceutical company names |
| `pro_scales_lexicon.json` | 161 KB | Patient-reported outcome scales |
| `clinical_staging_lexicon.json` | 6.5 KB | Clinical staging systems |
| `biomarker_thresholds_lexicon.json` | 5.9 KB | Biomarker reference ranges |
| `rare_disease_registries_lexicon.json` | 6.0 KB | Rare disease registries |
| `disease_lexicon_c3g.json` | 10 KB | C3 Glomerulopathy terms |
| `disease_lexicon_pah.json` | 21 KB | Pulmonary Arterial Hypertension |
| `disease_lexicon_anca.json` | 16 KB | ANCA-associated vasculitis |
| `disease_lexicon_igan.json` | 14 KB | IgA Nephropathy |

---

## Configuration (G_config/config.yaml)

The configuration file (~1,000 lines) controls all pipeline behavior.

### System Metadata

```yaml
system:
  name: "ESE"
  version: "15.0"
  pipeline_version: "0.7"
```

### Extraction Pipeline Flags

```yaml
extractors:
  drugs: true
  diseases: true
  genes: true
  abbreviations: true
  feasibility: true
  pharma_companies: true
  authors: true
  citations: true
  document_metadata: true
  tables: true
  visuals: true

options:
  use_llm_validation: true
  use_llm_feasibility: true
  use_vlm_tables: true
  use_normalization: true
  use_native_figure_extraction: true
```

### Visual Extraction Settings

```yaml
visual_extraction:
  enabled: true
  detection:
    table_mode_default: "fast"         # "fast" or "accurate"
    enable_escalation: true
    min_figure_area_ratio: 0.02
    enable_ocr: true
    ocr_backend: "surya"               # "surya" (recommended) or "easyocr"
    do_cell_matching: true
  rendering:
    default_dpi: 300
    min_dpi: 200
    max_dpi: 400
    padding_sides_pts: 12
    padding_caption_pts: 72
  triage:
    skip_area_ratio: 0.02
    vlm_area_threshold: 0.10
  vlm:
    enabled: true
    model: "claude-sonnet-4-20250514"
    validate_tables: true
  resolution:
    merge_multipage: true
    deduplicate: true
    dedupe_threshold: 0.7
```

### API Configuration

```yaml
api:
  pubtator:
    rate_limit_per_second: 3
    cache_ttl_days: 7
  claude:
    fast_model: "claude-sonnet-4-5-20250514"
    validation_model: "claude-sonnet-4-20250514"
    batch_delay_ms: 100
  nct:
    cache_ttl_days: 30
```

### PASO Heuristics

```yaml
heuristics:
  paso_a:  # Auto-approve statistical abbreviations (CI, HR, SD, OR, ...)
  paso_b:  # Country code blacklist
  paso_c:  # Hyphenated abbreviation enrichment from ClinicalTrials.gov
  paso_d:  # LLM SF-only extraction for missing abbreviations
  blacklist:  # Country codes, credentials, regulatory agencies
  common_words:  # English function words, academic terms
  contextual_rules:  # For ambiguous abbreviations (IA, IG)
  case_sensitive:  # For eGFR, sC5b-9
```

### Domain Profiles

Built-in profiles adjust extraction priors per therapeutic area:

| Profile | Focus |
|---------|-------|
| `generic` | Default, no domain bias |
| `nephrology` | Kidney-specific terms, biomarkers |
| `oncology` | Cancer staging, treatment lines |
| `pulmonology` | Respiratory conditions, lung function |

---

## Entity Detection Strategies

### Disease Detection (C06)

Layered lexicon priority:

```
1. Specialized lexicons (C3G, PAH, ANCA, IgAN)
2. MONDO disease ontology (97K terms)
3. Orphanet rare diseases (9.5K terms)
4. scispacy biomedical NER
→ Confidence-based FP filter (C24)
→ PubTator3 enrichment (E04)
→ Entity deduplication (E17)
```

### Drug Detection (C07)

Layered lexicon priority:

```
1. Alexion pipeline drugs
2. Investigational compounds
3. FDA-approved drugs
4. RxNorm vocabulary (132K terms)
5. ChEMBL approved drugs (23K terms)
6. scispacy chemical NER
→ FP filter with 16 exclusion sets (C25-C26)
→ PubTator3 enrichment (E05)
→ Drug combination parsing (E16)
→ Entity deduplication (E17)
```

### Gene Detection (C16)

Layered lexicon priority:

```
1. Orphadata rare disease genes
2. HGNC official symbols + aliases (~43K)
3. Gene name patterns (regex)
4. scispacy NER
→ Context-aware FP filter (C34)
→ PubTator3 enrichment (E18)
→ Entity deduplication (E17)
```

### Feasibility Extraction (C08, C11, I02)

Multi-source approach:

```
1. C08: Rule-based pattern extraction
   - Epidemiology anchors, inclusion/exclusion markers
   - Endpoint patterns, site patterns
2. C11: Claude LLM section-targeted extraction
   - Study design, eligibility, endpoints, sites
3. I02: Multi-NER enrichment stack
   - EpiExtract4GARD (epidemiology entities)
   - ZeroShotBioNER (ADE, dosage, frequency)
   - DistilBERT biomedical NER (84 entity types)
   - Patient journey extraction
   - Registry enrichment
   - Genetic variant enrichment
```

---

## Testing

### Test Suite Structure

59 test modules in `K_tests/` covering all pipeline layers:

| Range | Coverage | Layer |
|-------|----------|-------|
| K01-K07 | Core models, provenance, exceptions, NER | A_core |
| K08-K22 | PDF parsing, layout, visual pipeline, rendering | B_parsing |
| K23-K29 | Abbreviation, disease, drug, gene FP filters | C_generators |
| K30-K31 | Lexicon loaders, generator imports | C_generators |
| K32-K35 | Prompt registry, LLM engine, validation logger | D_validation |
| K36-K39 | Term mapper, deduplicator, span dedup | E_normalization |
| K40-K41 | Gold loader, scorer | F_evaluation |
| K42-K50 | Import validation, config, factory, processors, export | G-J |
| K51-K59 | Provenance, authors, citations, unicode, domain profiles | A_core, Z_utils |

### Test Fixtures (conftest.py)

Shared fixtures for all tests:

- **Config fixtures**: `test_config()`, `cache_enabled_config()`
- **Mock API responses**: PubTator3, ClinicalTrials.gov
- **Mock clients**: `mock_pubtator_client()`, `mock_clinicaltrials_client()`
- **Sample entities**: `sample_disease_dict()`, `sample_drug_dict()`
- **Assertion helpers**: `assert_no_warnings()`, `capture_logs()`

### Running Tests

```bash
# All tests
cd corpus_metadata && python -m pytest K_tests/ -v

# Specific test file
python -m pytest K_tests/K26_test_disease_fp_filter.py -v

# By marker
python -m pytest K_tests/ -m "not slow" -v

# With coverage
python -m pytest K_tests/ --cov=corpus_metadata --cov-report=html
```

### Test Configuration (pytest.ini)

```ini
testpaths = K_tests
python_files = test_*.py K*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: slow tests
    integration: integration tests
    requires_gpu: needs GPU
```

---

## Gold Standard Evaluation

### Datasets

| Dataset | Location | Purpose |
|---------|----------|---------|
| NLP4RARE | `gold_data/NLP4RARE/` | Rare disease NER benchmark |
| PAPERS | `gold_data/PAPERS/` | Clinical paper extraction gold standard |

### Evaluation Flow

```
F01_gold_loader.py   →  Load gold annotations (JSON/CSV)
F02_scorer.py        →  Precision / Recall / F1 (strict + lenient)
F03_evaluation_runner.py  →  Per-entity, per-dataset, overall metrics
```

---

## Environment & Setup

### Requirements

- Python 3.12+
- macOS / Linux

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r corpus_metadata/requirements.txt

# scispacy models (installed separately)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

### Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | Yes | Claude API access (validation, VLM) |
| `CORPUS_BASE_PATH` | No | Override default base path |

API key can also be set in a `.env` file at the project root.

### Dependencies

#### Core

| Package | Version | Purpose |
|---------|---------|---------|
| `anthropic` | >=0.50.0 | Claude API client |
| `pydantic` | >=2.0,<3.0 | Data validation |
| `pyyaml` | >=6.0 | Configuration |
| `python-dotenv` | >=1.0 | Environment variables |

#### NLP / Parsing

| Package | Version | Purpose |
|---------|---------|---------|
| `unstructured[pdf,pymupdf]` | >=0.10.0 | PDF text extraction |
| `scispacy` | >=0.5.0 | Biomedical NER + UMLS linking |
| `spacy` | >=3.7.4,<3.8.0 | NLP framework (pinned for scispacy) |
| `flashtext` | >=2.7 | Fast keyword matching (600K+ terms) |
| `pymupdf` | >=1.23.0 | PDF native extraction + rendering |
| `pymupdf4llm` | >=0.0.17 | LLM-friendly PDF extraction |

#### Visual / Table Extraction

| Package | Purpose |
|---------|---------|
| `docling` >=2.70.0 | Table detection with TableFormer |
| `docling-surya` >=0.1.0 | SuryaOCR integration (recommended) |
| `doclayout-yolo` | Fast DocLayout-YOLO detection (optional) |

#### Data Processing

| Package | Version | Purpose |
|---------|---------|---------|
| `requests` | >=2.25.0 | HTTP client (PubTator, NCT) |
| `pandas` | >=1.3.0 | Data manipulation |
| `Pillow` | >=9.0.0 | Image processing |
| `numpy` | >=1.19,<2.0 | Numerical computing |
| `tqdm` | >=4.60.0 | Progress bars |

#### Pinned for Compatibility

| Package | Version | Reason |
|---------|---------|--------|
| `spacy` | >=3.7.4,<3.8.0 | scispacy compatibility |
| `numpy` | >=1.19,<2.0 | Numeric stability |
| `thinc` | >=8.2.0,<8.3.0 | spacy dependency |
| `blis` | >=0.7.0,<0.8.0 | Linear algebra for thinc |

---

## Usage

### Command Line

```bash
# Run pipeline on configured PDF folder
cd corpus_metadata
python orchestrator.py

# Run tests
python -m pytest K_tests/ -v

# Type checking
mypy corpus_metadata

# Linting
ruff check corpus_metadata
```

### Python API

```python
from corpus_metadata.orchestrator import Orchestrator

# Initialize (loads config, NLP models, lexicons)
orch = Orchestrator()

# Process single PDF
results = orch.process_pdf("/path/to/document.pdf")

# Process all PDFs in folder
orch.process_folder()

# Access results
print(f"Abbreviations: {len(results.get('abbreviations', []))}")
print(f"Diseases: {len(results.get('diseases', []))}")
print(f"Drugs: {len(results.get('drugs', []))}")
print(f"Genes: {len(results.get('genes', []))}")
```

### Visual Pipeline Direct Usage

```python
from B_parsing.B12_visual_pipeline import extract_visuals

result = extract_visuals("document.pdf")
print(f"Found {len(result.visuals)} visuals")
print(f"Tables: {result.tables_detected}, Figures: {result.figures_detected}")
```

---

## Output Structure

Processing `ClinicalTrial.pdf` creates:

```
/path/to/ClinicalTrial/
├── abbreviations_ClinicalTrial_20260131_103045.json
├── diseases_ClinicalTrial_20260131_103045.json
├── drugs_ClinicalTrial_20260131_103045.json
├── genes_ClinicalTrial_20260131_103045.json
├── pharma_ClinicalTrial_20260131_103045.json
├── authors_ClinicalTrial_20260131_103045.json
├── citations_ClinicalTrial_20260131_103045.json
├── feasibility_ClinicalTrial_20260131_103045.json
├── recommendations_ClinicalTrial_20260131_103045.json
├── figures_ClinicalTrial_20260131_103045.json
├── tables_ClinicalTrial_20260131_103045.json
├── visuals_ClinicalTrial_20260131_103045.json
├── metadata_ClinicalTrial_20260131_103045.json
├── ClinicalTrial_extracted_text_20260131_103045.txt
├── ClinicalTrial_flowchart_page3_1.png
└── visual_images_ClinicalTrial/
    ├── table_1_page4.png
    └── figure_2_page5.png
```

---

## Caching

API responses are cached to disk to reduce latency and API costs:

| Cache | Location | TTL | Purpose |
|-------|----------|-----|---------|
| PubTator3 | `cache/pubtator/` | 7 days | Disease/drug/gene enrichment |
| ClinicalTrials.gov | `cache/clinicaltrials/` | 30 days | NCT metadata |
| General API | `cache/api/` | Varies | Other API responses |

Cache uses hash-based filenames (e.g., `autocomplete_chemical_135ae46a3720.json`). Cache is disabled by default in test runs for isolation.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.2 | 2026-01 | Graceful Docling handling, improved warnings, OCR configuration |
| 1.1 | 2026-01 | Unified visual extraction pipeline (B12-B17), Docling FAST/ACCURATE tiering, VLM enrichment |
| 1.0 | 2026-01 | PDF-native figure extraction (B09-B11), vector plot detection, caption linking |
| 0.9 | 2026-01 | Author/investigator extraction, citation/reference extraction, gene detection |
| 0.8 | 2026-01 | Output folder restructure, image file saving, Vision LLM analysis |
| 0.7 | 2025-12 | Image extraction, Vision LLM integration |
| 0.6 | 2025-11 | Feasibility extraction, document metadata |
| 0.5 | 2025-10 | Drug detection, PubTator enrichment |
| 0.4 | 2025-09 | Disease detection, NCT enrichment |
| 0.3 | 2025-08 | Multi-generator architecture |
| 0.2 | 2025-07 | Validation layer, heuristics |
| 0.1 | 2025-06 | Initial pipeline |

---

## References

- **NLP4RARE Corpus:** https://github.com/isegura/NLP4RARE-CM-UC3M
- **Schwartz-Hearst Algorithm:** DOI:10.1093/bioinformatics/btg014
- **PubTator3 API:** https://www.ncbi.nlm.nih.gov/research/pubtator3/api
- **Unstructured.io:** https://unstructured.io/
- **scispacy:** https://allenai.github.io/scispacy/
- **ClinicalTrials.gov API:** https://clinicaltrials.gov/data-api/api
- **PyMuPDF (fitz):** https://pymupdf.readthedocs.io/
- **HGNC:** https://www.genenames.org/
- **Docling:** https://github.com/DS4SD/docling
- **TableFormer:** https://arxiv.org/abs/2203.01017
- **MONDO:** https://mondo.monarchinitiative.org/
- **Orphanet:** https://www.orpha.net/
- **RxNorm:** https://www.nlm.nih.gov/research/umls/rxnorm/
- **EpiExtract4GARD:** https://github.com/ncats/epi4GARD
- **DocLayout-YOLO:** https://github.com/opendatalab/DocLayout-YOLO
