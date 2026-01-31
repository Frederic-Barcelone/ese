# Metadata Extraction Pipeline

A production-grade 6-layer pipeline for extracting structured metadata from clinical trial and medical PDF documents. Focused on rare disease research.

**Extraction Capabilities:**
- Abbreviations and acronyms with definitions
- Diseases (rare diseases, conditions, syndromes with ICD-10, SNOMED, MONDO, ORPHA codes)
- Drugs (approved, investigational, compounds with RxNorm, DrugBank, MeSH codes)
- Genes (HGNC symbols, aliases, Entrez, Ensembl, disease associations)
- Pharmaceutical companies (sponsors, manufacturers)
- Authors and investigators (names, roles, affiliations, ORCID)
- Citations and references (PMID, DOI, NCT, URLs)
- Clinical trial feasibility data (eligibility, endpoints, patient journey)
- Guideline recommendations with evidence levels
- Document metadata (classification, dates, identifiers)
- Figures with Vision LLM analysis (flowcharts, charts, Kaplan-Meier plots)
- Tables with Docling TableFormer and VLM validation

---

## Architecture Overview

```
+---------------------------------------------------------------------------------+
|                           ORCHESTRATOR (orchestrator.py)                         |
|                                                                                  |
|  +----------+   +----------+   +----------+   +----------+   +----------+       |
|  | A_core   |-->| B_parsing|-->|C_generate|-->|D_validate|-->|E_normalize|      |
|  | Models   |   | PDF->Doc |   | Candidates|   | LLM/Rules|   | Enrich   |       |
|  +----------+   +----------+   +----------+   +----------+   +----------+       |
|       |              |              |              |              |              |
|       |              |              |              |              v              |
|       |              |              |              |        +----------+         |
|       |              |              |              |        |F_evaluate|         |
|       |              |              |              |        | Metrics  |         |
|       |              |              |              |        +----------+         |
|       |              |              |              |              |              |
|       v              v              v              v              v              |
|  +----------+   +----------+   +----------+   +----------+   +----------+       |
|  |H_pipeline|   |I_extract |   | J_export |   | Z_utils  |   | G_config |       |
|  |Components|   |Processors|   | Handlers |   | Helpers  |   |  YAML    |       |
|  +----------+   +----------+   +----------+   +----------+   +----------+       |
+---------------------------------------------------------------------------------+
```

### Design Philosophy

| Layer | Goal | Strategy |
|-------|------|----------|
| **Generators (C)** | High Recall | Exhaustive extraction, noise acceptable |
| **Validation (D)** | High Precision | Claude LLM filters false positives |
| **Normalization (E)** | Standardization | Map to ontologies, deduplicate |

---

## Directory Structure

```
corpus_metadata/
|
+-- A_core/                                  # LAYER 0: FOUNDATION
|   +-- A00_logging.py                       # Centralized logging configuration
|   +-- A01_domain_models.py                 # Core Pydantic schemas (Candidate, ExtractedEntity)
|   +-- A02_interfaces.py                    # Abstract Base Classes (generators, validators)
|   +-- A03_provenance.py                    # Run ID generation, git hash, traceability
|   +-- A04_heuristics_config.py             # Centralized heuristics (whitelists/blacklists)
|   +-- A05_disease_models.py                # Disease domain models (ICD-10, SNOMED, ORPHA)
|   +-- A06_drug_models.py                   # Drug domain models (RxCUI, NDC, DrugBank)
|   +-- A07_feasibility_models.py            # Clinical trial feasibility models
|   +-- A08_document_metadata_models.py      # Document-level metadata models
|   +-- A09_pharma_models.py                 # Pharma company models
|   +-- A10_author_models.py                 # Author/investigator models
|   +-- A11_citation_models.py               # Citation/reference models
|   +-- A12_exceptions.py                    # Custom exception classes
|   +-- A13_ner_models.py                    # Named Entity Recognition models
|   +-- A13_visual_models.py                 # Visual extraction models (ExtractedVisual)
|   +-- A14_extraction_result.py             # Unified extraction result container
|   +-- A15_domain_profile.py                # Domain-specific configuration profiles
|   +-- A16_pipeline_metrics.py              # Pipeline performance metrics
|   +-- A17_care_pathway_models.py           # Patient care pathway models
|   +-- A18_recommendation_models.py         # Guideline recommendation models
|   +-- A19_gene_models.py                   # Gene/genetic entity models (HGNC, Entrez)
|
+-- B_parsing/                               # LAYER 1: DOCUMENT STRUCTURE
|   +-- B01_pdf_to_docgraph.py               # Main PDF parser (Unstructured.io + native figures)
|   +-- B01a_text_helpers.py                 # Text extraction helper functions
|   +-- B01b_native_figure_extraction.py     # PyMuPDF figure extraction integration
|   +-- B01c_legacy_ordering.py              # Legacy reading order algorithms
|   +-- B01d_repetition_inference.py         # Repeated element detection (headers/footers)
|   +-- B02_doc_graph.py                     # Internal model (Page -> Block -> Token)
|   +-- B03_table_extractor.py               # Table extraction orchestrator
|   +-- B03a_table_validation.py             # Table structure validation rules
|   +-- B03b_table_rendering.py              # Table image rendering (PyMuPDF)
|   +-- B03c_docling_backend.py              # Docling TableFormer backend (95-98% accuracy)
|   +-- B04_column_ordering.py               # SOTA multi-column layout detection
|   +-- B04a_column_detection.py             # Column boundary detection algorithms
|   +-- B04b_xy_cut_ordering.py              # XY-Cut++ reading order algorithm
|   +-- B05_section_detector.py              # Section/heading detection
|   +-- B06_confidence.py                    # Confidence scoring features
|   +-- B07_negation.py                      # Negation/assertion detection
|   +-- B08_eligibility_parser.py            # Eligibility criteria parsing
|   +-- B09_pdf_native_figures.py            # Raster + vector figure extraction
|   +-- B10_caption_detector.py              # Caption detection + column inference
|   +-- B11_extraction_resolver.py           # Multi-signal figure resolution
|   +-- B12_visual_pipeline.py               # Unified visual extraction pipeline
|   +-- B13_visual_detector.py               # Docling detection with FAST/ACCURATE tiering
|   +-- B14_visual_renderer.py               # PyMuPDF rendering at 200-400 DPI
|   +-- B15_caption_extractor.py             # Multisource caption extraction
|   +-- B16_triage.py                        # SKIP/CHEAP/VLM routing logic
|   +-- B17_document_resolver.py             # Body text scanning, multi-page merge
|
+-- C_generators/                            # LAYER 2: CANDIDATE GENERATION
|   +-- C00_strategy_identifiers.py          # Database IDs (OMIM, DOI, NCT, ORCID)
|   +-- C01_strategy_abbrev.py               # Schwartz-Hearst syntax patterns
|   +-- C01a_abbrev_patterns.py              # Abbreviation pattern definitions
|   +-- C02_strategy_regex.py                # Rigid patterns (Trial IDs, DOIs)
|   +-- C03_strategy_layout.py               # Spatial extraction from tables
|   +-- C04_strategy_flashtext.py            # Lexicon matching (600K+ terms)
|   +-- C04a_noise_filters.py                # False positive noise filtering
|   +-- C04b_lexicon_loaders.py              # Lexicon file loaders
|   +-- C04c_inline_definition_detector.py   # Inline abbreviation definitions
|   +-- C05_strategy_glossary.py             # Glossary table extractor
|   +-- C06_strategy_disease.py              # Disease mention detection
|   +-- C06a_disease_fp_filter.py            # Disease false positive filter
|   +-- C07_strategy_drug.py                 # Drug/chemical detection
|   +-- C07a_drug_fp_filter.py               # Drug false positive filter
|   +-- C07b_drug_fp_constants.py            # Drug FP filter constants
|   +-- C08_strategy_feasibility.py          # Rule-based feasibility extraction
|   +-- C08a_feasibility_patterns.py         # Feasibility regex patterns
|   +-- C08b_feasibility_fp_filter.py        # Feasibility false positive filter
|   +-- C09_strategy_document_metadata.py    # Document metadata extraction
|   +-- C10_vision_image_analysis.py         # Vision LLM image analysis
|   +-- C11_llm_feasibility.py               # LLM-based feasibility extraction
|   +-- C11a_feasibility_prompts.py          # LLM feasibility prompts
|   +-- C11b_feasibility_response_parser.py  # LLM response parsing
|   +-- C12_guideline_recommendation_extractor.py  # Guideline recommendation extraction
|   +-- C12a_recommendation_patterns.py      # Recommendation patterns
|   +-- C12b_recommendation_llm.py           # LLM recommendation extraction
|   +-- C12c_recommendation_vlm.py           # VLM recommendation extraction
|   +-- C13_strategy_author.py               # Author/investigator detection
|   +-- C14_strategy_citation.py             # Citation/reference detection
|   +-- C15_vlm_table_extractor.py           # VLM table extraction
|   +-- C16_strategy_gene.py                 # Gene/genetic entity detection
|   +-- C16_vlm_visual_enrichment.py         # Claude Vision enrichment
|   +-- C16a_gene_fp_filter.py               # Gene false positive filter
|   +-- C17_flowchart_graph_extractor.py     # Flowchart to graph extraction
|   +-- C18_strategy_pharma.py               # Pharma company detection
|
+-- D_validation/                            # LAYER 3: LLM VERIFICATION
|   +-- D01_prompt_registry.py               # Versioned prompts with hash tracking
|   +-- D02_llm_engine.py                    # LLM validation engine
|   +-- D02a_claude_client.py                # Claude API client wrapper
|   +-- D03_validation_logger.py             # JSONL audit trail
|   +-- D05_quote_verifier.py                # Quote verification against source
|
+-- E_normalization/                         # LAYER 4: STANDARDIZATION
|   +-- E01_term_mapper.py                   # Synonym resolution
|   +-- E02_disambiguator.py                 # Context-based disambiguation
|   +-- E03_disease_normalizer.py            # Disease ontology normalization
|   +-- E04_pubtator_enricher.py             # PubTator API (MeSH, aliases)
|   +-- E05_drug_enricher.py                 # Drug identifier enrichment
|   +-- E06_nct_enricher.py                  # NCT trial metadata enrichment
|   +-- E07_deduplicator.py                  # Merge duplicates, pick best LF
|   +-- E08_epi_extract_enricher.py          # EpiExtract4GARD enricher
|   +-- E09_zeroshot_bioner.py               # Zero-shot biomedical NER
|   +-- E10_biomedical_ner_all.py            # Comprehensive biomedical NER
|   +-- E11_span_deduplicator.py             # Span-level deduplication
|   +-- E12_patient_journey_enricher.py      # Patient journey extraction
|   +-- E13_registry_enricher.py             # Patient registry enricher
|   +-- E14_citation_validator.py            # Citation validation
|   +-- E15_genetic_enricher.py              # Genetic variant enrichment
|   +-- E16_drug_combination_parser.py       # Drug combination parsing
|
+-- F_evaluation/                            # LAYER 5: METRICS & TESTING
|   +-- F01_gold_loader.py                   # Load gold standard annotations
|   +-- F02_scorer.py                        # Precision/Recall/F1 calculation
|   +-- F03_generator_unit_test.py           # Generator unit tests (no LLM)
|   +-- F04_pipeline_test.py                 # Full pipeline evaluation
|   +-- F05_extraction_analysis.py           # Detailed extraction report
|
+-- G_config/                                # CONFIGURATION
|   +-- G01_config_keys.py                   # Configuration key constants
|   +-- config.yaml                          # Central configuration
|
+-- H_pipeline/                              # PIPELINE COMPONENTS
|   +-- H01_component_factory.py             # Factory for pipeline components
|   +-- H02_abbreviation_pipeline.py         # Abbreviation extraction pipeline
|   +-- H03_visual_integration.py            # Visual pipeline orchestrator integration
|   +-- H04_merge_resolver.py                # Multi-source merge resolution
|
+-- I_extraction/                            # ENTITY EXTRACTION
|   +-- I01_entity_processors.py             # Unified entity extraction
|   +-- I02_feasibility_processor.py         # Feasibility data extraction
|
+-- J_export/                                # OUTPUT HANDLERS
|   +-- J01_export_handlers.py               # JSON export orchestrator
|   +-- J01a_entity_exporters.py             # Entity-specific exporters
|   +-- J01b_metadata_exporters.py           # Metadata exporters
|   +-- J02_visual_export.py                 # Visual extraction JSON export
|
+-- Z_utils/                                 # UTILITIES & SCRIPTS
|   +-- Z01_api_client.py                    # Shared API client utilities
|   +-- Z02_text_helpers.py                  # Text processing helpers
|   +-- Z03_text_normalization.py            # Text normalization functions
|   +-- Z04_image_utils.py                   # Image processing utilities
|   +-- Z05_path_utils.py                    # Path resolution utilities
|   +-- Z09_download_gene_lexicon.py         # HGNC gene lexicon downloader
|   +-- Z10_download_lexicons.py             # Lexicon download scripts
|
+-- tests/                                   # TEST SUITE
|   +-- test_core/                           # Core model tests
|   +-- test_enrichers/                      # Enricher tests
|   +-- test_generators/                     # Generator tests
|   +-- test_parsing/                        # Parser tests
|   +-- test_utils/                          # Utility tests
|   +-- test_visual_extraction/              # Visual pipeline tests
|
+-- cache/                                   # Runtime cache (PubTator)
+-- orchestrator.py                          # Main pipeline entry point
+-- orchestrator_utils.py                    # Orchestrator helper functions
+-- EXTRACTOR.MD                             # This documentation
```

---

## Data Flow

```
PDF Input
    |
    v
+----------------------------------------------------------+
| B_parsing: PDF -> DocumentGraph                           |
| - Unstructured.io partition_pdf()                         |
| - Docling TableFormer (95-98% table accuracy)             |
| - PDF-native figure extraction (B09-B11)                  |
| - Visual pipeline (B12-B17)                               |
| - Layout detection, section detection                     |
+----------------------------------------------------------+
    |
    v
+----------------------------------------------------------+
| C_generators: DocumentGraph -> Candidates                 |
| - C01: Schwartz-Hearst syntax patterns                    |
| - C04: FlashText lexicon (600K+ terms)                    |
| - C06: Disease detection (MONDO, Orphanet)                |
| - C07: Drug detection (ChEMBL, RxNorm)                    |
| - C08/C11: Feasibility extraction                         |
| - C10: Vision LLM image analysis                          |
| - C12: Guideline recommendations                          |
| - C13: Author/investigator detection                      |
| - C14: Citation/reference detection                       |
| - C16: Gene detection (HGNC)                              |
| - C17: Flowchart graph extraction                         |
| - C18: Pharma company detection                           |
+----------------------------------------------------------+
    |
    v
+----------------------------------------------------------+
| D_validation: Candidates -> Validated                     |
| - Heuristic shortcuts (PASO A-D)                          |
| - Claude LLM batch validation                             |
| - Quote verification                                      |
| - Structured JSON responses                               |
+----------------------------------------------------------+
    |
    v
+----------------------------------------------------------+
| E_normalization: Validated -> Enriched                    |
| - PubTator API (MeSH, disease codes)                      |
| - NCT enrichment (trial metadata)                         |
| - Genetic enrichment (variants)                           |
| - Drug combination parsing                                |
| - Deduplication (merge same SF)                           |
+----------------------------------------------------------+
    |
    v
Output Directory (named after PDF)
+-- abbreviations_{stem}_{timestamp}.json
+-- diseases_{stem}_{timestamp}.json
+-- drugs_{stem}_{timestamp}.json
+-- genes_{stem}_{timestamp}.json
+-- pharma_{stem}_{timestamp}.json
+-- authors_{stem}_{timestamp}.json
+-- citations_{stem}_{timestamp}.json
+-- feasibility_{stem}_{timestamp}.json
+-- recommendations_{stem}_{timestamp}.json
+-- figures_{stem}_{timestamp}.json
+-- tables_{stem}_{timestamp}.json
+-- visuals_{stem}_{timestamp}.json
+-- metadata_{stem}_{timestamp}.json
+-- {stem}_extracted_text_{timestamp}.txt
+-- {stem}_{type}_page{N}_{index}.png
+-- visual_images_{stem}/
```

---

## Table Extraction (Docling TableFormer)

The pipeline uses IBM's Docling with TableFormer for high-accuracy table extraction:

```
PDF Input
    |
    v
+----------------------------------------------------------+
| Docling TableFormer                                       |
| - FAST mode (default): Quick extraction                   |
| - ACCURATE mode: Complex tables, merged cells             |
| - SuryaOCR integration for scanned documents              |
| - 95-98% TEDS accuracy on complex tables                  |
+----------------------------------------------------------+
    |
    v
+----------------------------------------------------------+
| Escalation Logic                                          |
| IF merged_cell_count > 5 OR                               |
|    header_depth > 3 OR                                    |
|    token_coverage < 0.70:                                 |
|    -> Escalate to ACCURATE mode                           |
+----------------------------------------------------------+
    |
    v
+----------------------------------------------------------+
| Optional VLM Validation                                   |
| - Validate structure against image                        |
| - Correct misaligned cells                                |
| - Extract missing data                                    |
+----------------------------------------------------------+
```

### Graceful Dependency Handling

When Docling is not installed, table extraction is skipped with a clear warning:

```
[WARN] Docling not installed - table extraction DISABLED
       Install with: pip install 'docling[docling-surya]'
```

---

## Visual Extraction Pipeline (v1.1)

A unified 4-stage pipeline for extracting tables and figures as high-quality images:

```
PDF Input
    |
    v
+----------------------------------------------------------+
| STAGE 1: DETECTION (Docling + PyMuPDF)                    |
| - Docling TableFormer FAST mode (default)                 |
| - Native figure extraction via PyMuPDF                    |
| - Escalate complex tables to ACCURATE mode                |
| - All bboxes in PDF points (coordinate discipline)        |
+----------------------------------------------------------+
    |
    v
+----------------------------------------------------------+
| STAGE 2: RENDERING (PyMuPDF + Caption Extraction)         |
| - Adaptive DPI: 200-400 based on visual size              |
| - Point-based padding (12pt sides, 72pt caption zone)     |
| - Multisource caption: PDF text preferred, OCR fallback   |
| - Reference parsing: "Table 1", "Figure 2-4", "Exhibit A" |
+----------------------------------------------------------+
    |
    v
+----------------------------------------------------------+
| STAGE 3: TRIAGE + VLM (Claude Vision)                     |
| TRIAGE ROUTING:                                           |
| - SKIP: tiny logos, repeated headers, noise               |
| - CHEAP_PATH: simple visuals, heuristic classification    |
| - VLM_REQUIRED: captioned, grid structure, references     |
|                                                           |
| VLM ENRICHMENT:                                           |
| - Type classification (table vs figure)                   |
| - Table structure validation                              |
| - Caption extraction/correction                           |
| - Continuation detection                                  |
+----------------------------------------------------------+
    |
    v
+----------------------------------------------------------+
| STAGE 4: RESOLUTION (Document-level)                      |
| - Body text scanning for mentions                         |
| - Section context inference                               |
| - Multi-page visual merging                               |
| - Deduplication (70% overlap threshold)                   |
+----------------------------------------------------------+
```

### Warning Messages

When dependencies are unavailable, clear warnings are displayed:

```
[WARN] Docling not installed - table extraction DISABLED
       Install with: pip install 'docling[docling-surya]'

[WARN] SuryaOCR not available, using default OCR
       Install with: pip install docling-surya

[WARN] Anthropic SDK not installed - VLM enrichment DISABLED
       Classification will use heuristics only.

[INFO] Table extraction skipped (Docling not available)
```

---

## Core Models (A_core)

### A01_domain_models.py - Foundation Types

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

### A19_gene_models.py - Gene Types

```python
class ExtractedGene(BaseModel):
    matched_text: str              # "BRCA1"
    hgnc_symbol: str               # Official HGNC symbol
    full_name: Optional[str]       # "BRCA1 DNA repair associated"

    # Identifiers
    hgnc_id: Optional[str]         # "HGNC:1100"
    entrez_id: Optional[str]       # "672"
    ensembl_id: Optional[str]      # "ENSG00000012048"
    omim_id: Optional[str]         # "113705"
    uniprot_id: Optional[str]      # "P38398"

    is_alias: bool = False
    locus_type: Optional[str]      # "gene with protein product"
    chromosome: Optional[str]      # "17q21.31"
    associated_diseases: List[GeneDisease]
```

### A18_recommendation_models.py - Guideline Recommendations

```python
class GuidelineRecommendation(BaseModel):
    """Extracted guideline recommendation."""
    recommendation_text: str           # Full recommendation text
    evidence_level: Optional[str]      # "Level A", "Grade 1"
    strength: Optional[str]            # "Strong", "Weak"
    source_guideline: Optional[str]    # Guideline name
    conditions: List[str]              # Target conditions
    interventions: List[str]           # Recommended interventions
```

### A13_visual_models.py - Visual Types

```python
class ExtractedVisual(BaseModel):
    """Unified model for tables and figures."""
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

---

## Lexicons (~617,000 terms)

| Lexicon | Terms | Source |
|---------|-------|--------|
| abbreviation_general | 5,392 | Curated abbreviations |
| **meta_inventory** | **65,048** | Clinical abbreviations |
| umls_biological | 97,308 | UMLS biomedical |
| umls_clinical | 19,937 | UMLS clinical |
| trial_acronyms | 125,454 | ClinicalTrials.gov |
| pro_scales | 301 | Patient-Reported Outcomes |
| rare_disease_acronyms | 1,631 | Rare disease names |
| **mondo_diseases** | **97,313** | Unified disease ontology |
| **chembl_drugs** | **22,802** | Approved drugs |
| RxNorm terms | 131,961 | Drug vocabulary |
| Orphanet diseases | 9,468 | Rare diseases |
| HGNC genes | ~43,000 | Gene symbols + aliases |

---

## Configuration (G_config/config.yaml)

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
    table_mode_default: "fast"
    enable_escalation: true
    min_figure_area_ratio: 0.02
    enable_ocr: true
    ocr_backend: "surya"    # "surya" (recommended) or "easyocr"
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

---

## Usage

### Command Line

```bash
# Run pipeline on PDF folder
cd corpus_metadata
python orchestrator.py

# Run tests
python -m pytest tests/ -v

# Type checking
mypy corpus_metadata

# Linting
ruff check corpus_metadata
```

### Python API

```python
from corpus_metadata.orchestrator import Orchestrator

# Initialize (loads config, NLP models)
orch = Orchestrator()

# Process PDF - outputs to /path/to/document/
results = orch.process_pdf("/path/to/document.pdf")

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

## Dependencies

### Core Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `anthropic` | >=0.18 | Claude API client |
| `pydantic` | >=2.0 | Data validation |
| `pyyaml` | >=6.0 | Configuration |
| `python-dotenv` | >=1.0 | Environment |

### NLP/Parsing

| Package | Version | Purpose |
|---------|---------|---------|
| `unstructured[pdf]` | >=0.10 | PDF parsing |
| `scispacy` | >=0.5 | Biomedical NER |
| `flashtext` | >=2.7 | Fast keyword matching |
| `pymupdf` (fitz) | >=1.23 | PDF native extraction |

### Table/Visual Extraction (Optional)

| Package | Purpose |
|---------|---------|
| `docling` | Table detection with TableFormer |
| `docling-surya` | SuryaOCR integration (recommended) |

---

## Output Structure

Processing `ClinicalTrial.pdf` creates:

```
/path/to/ClinicalTrial/
+-- abbreviations_ClinicalTrial_20260131_103045.json
+-- diseases_ClinicalTrial_20260131_103045.json
+-- drugs_ClinicalTrial_20260131_103045.json
+-- genes_ClinicalTrial_20260131_103045.json
+-- pharma_ClinicalTrial_20260131_103045.json
+-- authors_ClinicalTrial_20260131_103045.json
+-- citations_ClinicalTrial_20260131_103045.json
+-- feasibility_ClinicalTrial_20260131_103045.json
+-- recommendations_ClinicalTrial_20260131_103045.json
+-- figures_ClinicalTrial_20260131_103045.json
+-- tables_ClinicalTrial_20260131_103045.json
+-- visuals_ClinicalTrial_20260131_103045.json
+-- metadata_ClinicalTrial_20260131_103045.json
+-- ClinicalTrial_extracted_text_20260131_103045.txt
+-- ClinicalTrial_flowchart_page3_1.png
+-- visual_images_ClinicalTrial/
    +-- table_1_page4.png
    +-- figure_2_page5.png
```

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
