
# doc : https://chunkforge.com/blog/nlp-named-entity-recognition
# Ralph Wiggum: Continuous Agent Loops in Claude Code


# Metadata Extraction Pipeline

A production-grade 6-layer pipeline for extracting structured metadata from clinical trial and medical PDF documents.

**Extraction Capabilities:**
- Abbreviations and acronyms with definitions
- Diseases (rare diseases, conditions, syndromes)
- Drugs (approved, investigational, compounds)
- Genes (HGNC symbols, aliases, disease associations)
- Pharmaceutical companies (sponsors, manufacturers)
- Authors and investigators (names, roles, affiliations)
- Citations and references (PMID, DOI, NCT, URLs)
- Clinical trial feasibility data (eligibility, endpoints, patient journey)
- Document metadata (classification, dates, identifiers)
- Figures with Vision LLM analysis (flowcharts, charts, Kaplan-Meier plots)
- Tables with VLM extraction

---

## Architecture Overview

```
+-------------------------------------------------------------------------------+
|                           ORCHESTRATOR (orchestrator.py)                       |
|                                                                                |
|  +----------+   +----------+   +----------+   +----------+   +----------+     |
|  | A_core   |-->| B_parsing|-->|C_generate|-->|D_validate|-->|E_normalize|    |
|  | Models   |   | PDF->Doc |   | Candidates|   | LLM/Rules|   | Enrich   |     |
|  +----------+   +----------+   +----------+   +----------+   +----------+     |
|       |              |              |              |              |            |
|       |              |              |              |              v            |
|       |              |              |              |        +----------+       |
|       |              |              |              |        |F_evaluate|       |
|       |              |              |              |        | Metrics  |       |
|       |              |              |              |        +----------+       |
+-------------------------------------------------------------------------------+
```

### Design Philosophy

| Layer | Goal | Strategy |
|-------|------|----------|
| **Generators (C)** | High Recall | Exhaustive extraction, noise acceptable |
| **Validation (D)** | High Precision | Claude LLM filters false positives |
| **Normalization (E)** | Standardization | Map to ontologies, deduplicate |

---

## Data Flow

```
PDF Input
    |
    v
+----------------------------------------------------------+
| B_parsing: PDF -> DocumentGraph                           |
| - Unstructured.io partition_pdf()                         |
| - Layout detection (columns, tables)                      |
| - PDF-native figure extraction (B09-B11)                  |
| - Section detection                                       |
+----------------------------------------------------------+
    |
    v
+----------------------------------------------------------+
| C_generators: DocumentGraph -> Candidates                 |
| - C01: Schwartz-Hearst syntax patterns                    |
| - C04: FlashText lexicon (600K+ terms)                    |
| - C05: Glossary table extraction                          |
| - C06: Disease detection                                  |
| - C07: Drug detection                                     |
| - C08: Feasibility extraction                             |
| - C09: Document metadata                                  |
| - C10: Vision image analysis                              |
| - C12: Pharma company detection                           |
| - C13: Author/investigator detection                      |
| - C14: Citation/reference detection                       |
| - C16: Gene detection                                     |
+----------------------------------------------------------+
    |
    v
+----------------------------------------------------------+
| D_validation: Candidates -> Validated                     |
| - Heuristic shortcuts (PASO A-D)                          |
| - Claude LLM batch validation                             |
| - Structured JSON responses                               |
+----------------------------------------------------------+
    |
    v
+----------------------------------------------------------+
| E_normalization: Validated -> Enriched                    |
| - PubTator API (MeSH, disease codes)                      |
| - NCT enrichment (trial metadata)                         |
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
+-- figures_{stem}_{timestamp}.json
+-- tables_{stem}_{timestamp}.json
+-- metadata_{stem}_{timestamp}.json
+-- {stem}_extracted_text_{timestamp}.txt
+-- {stem}_{type}_page{N}_{index}.png
```

---

## PDF-Native Figure Extraction (NEW in v1.0)

The pipeline uses a multi-signal architecture for accurate figure extraction:

```
PDF Input
    |
    +-> [Signal 1] Raster Figures: get_images() + get_image_rects()
    |
    +-> [Signal 2] Vector Figures: get_drawings() + text density heuristics
    |
    +-> [Signal 3] Layout Model: Unstructured/YOLOX (fallback signal)
    |
    +-> Deterministic Resolver
            -> Caption anchoring (Fig X, Table X)
            -> Overlap exclusion
            -> Column-aware bounds
```

### B09_pdf_native_figures.py - Raster & Vector Extraction

```python
@dataclass
class EmbeddedFigure:
    page_num: int
    bbox: Tuple[float, float, float, float]
    xref: int       # Store xref for lazy rendering
    image_hash: str # sha1 for deduplication

@dataclass
class VectorFigure:
    page_num: int
    bbox: Tuple[float, float, float, float]
    drawing_count: int   # Number of paths/lines
    has_axis_text: bool  # Axis labels detected
```

**Key Functions:**
- `extract_embedded_figures()` - Extract raster images from PDF XObjects
- `detect_vector_figures()` - Detect Kaplan-Meier plots via get_drawings()
- `filter_noise_images()` - Remove logos/headers by repetition/size/position
- `render_figure_by_xref()` - Lazy render with colorspace normalization

**Vector Detection Filters:**
- Minimum height: 80pt (filters narrow headers)
- Minimum area: 2% of page (filters small decorations)
- Header zone exclusion: top 12% of page
- Requires 20+ drawings OR axis-like text nearby

### B10_caption_detector.py - Caption Linking

```python
@dataclass
class Caption:
    text: str
    caption_type: str  # "figure" or "table"
    number: int        # Figure 1 -> 1
    bbox: Tuple[float, float, float, float]
    page_num: int
    column_idx: int    # Which column
```

**Key Functions:**
- `detect_all_captions()` - Find "Fig X" and "Table X" patterns
- `infer_page_columns()` - Detect column layout from text distribution
- `link_caption_to_figure()` - Link caption to nearest region ABOVE
- `get_table_region_column_aware()` - Define table bounds in column

### B11_extraction_resolver.py - Deterministic Resolution

```python
@dataclass
class ResolvedFigure:
    figure: Union[EmbeddedFigure, VectorFigure, ImageBlock]
    caption: Optional[Caption]
    source: str        # "caption_linked", "orphan_native", "layout_model"
    figure_type: str   # "raster", "vector", "layout_model"
```

**Resolution Rules (Priority Order):**
1. **Caption-anchored figures** - Link each "Fig X" to nearest region above
2. **Orphan native figures** - Native figures without captions (exclude table overlaps)
3. **Layout model figures** - Unstructured detections (only if no native coverage)

**Overlap Handling:**
- Figures cannot overlap with table regions
- Tables are truncated if they overlap with resolved figures
- Deduplication removes figures with >70% overlap

### Problems Solved

| Problem | Solution |
|---------|----------|
| Fig 4 chart includes Table 5 | Caption anchoring links only to regions ABOVE caption |
| Vector Kaplan-Meier plots missed | get_drawings() detector finds dense linework + axis text |
| Logo has whitespace | Filtered by hash repetition + small size + top margin |
| Two tables merged | Column-aware regions lock each table to its column |
| Tables include figures | Resolver excludes figure regions from table bounds |

---

## Directory Structure

```
corpus_metadata/
|
+-- A_core/                              # LAYER 0: FOUNDATION
|   +-- A01_domain_models.py             # Core Pydantic schemas (Candidate, ExtractedEntity)
|   +-- A02_interfaces.py                # Abstract Base Classes (generators, validators)
|   +-- A03_provenance.py                # Run ID generation, git hash, traceability
|   +-- A04_heuristics_config.py         # Centralized heuristics (whitelists/blacklists)
|   +-- A05_disease_models.py            # Disease domain models (ICD-10, SNOMED, ORPHA)
|   +-- A06_drug_models.py               # Drug domain models (RxCUI, NDC, DrugBank)
|   +-- A07_feasibility_models.py        # Clinical trial feasibility models
|   +-- A08_document_metadata_models.py  # Document-level metadata models
|   +-- A09_pharma_models.py             # Pharma company models
|   +-- A10_author_models.py             # Author/investigator models
|   +-- A11_citation_models.py           # Citation/reference models
|   +-- A12_gene_models.py               # Gene/genetic entity models
|
+-- B_parsing/                           # LAYER 1: DOCUMENT STRUCTURE
|   +-- B01_pdf_to_docgraph.py           # PDF parsing with Unstructured.io + native figures
|   +-- B02_doc_graph.py                 # Internal model (Page -> Block -> Token)
|   +-- B03_table_extractor.py           # Table detection and JSON serialization
|   +-- B04_column_ordering.py           # SOTA multi-column layout detection (XY-Cut++)
|   +-- B05_section_detector.py          # Section/heading detection
|   +-- B06_confidence.py                # Confidence scoring features
|   +-- B07_negation.py                  # Negation/assertion detection
|   +-- B08_eligibility_parser.py        # Eligibility criteria parsing
|   +-- B09_pdf_native_figures.py        # Raster + vector figure extraction (NEW)
|   +-- B10_caption_detector.py          # Caption detection + column inference (NEW)
|   +-- B11_extraction_resolver.py       # Multi-signal figure resolution (NEW)
|
+-- C_generators/                        # LAYER 2: CANDIDATE GENERATION
|   +-- C00_strategy_identifiers.py      # Database IDs (OMIM, DOI, NCT, ORCID)
|   +-- C01_strategy_abbrev.py           # Schwartz-Hearst syntax patterns
|   +-- C02_strategy_regex.py            # Rigid patterns (Trial IDs, DOIs)
|   +-- C03_strategy_layout.py           # Spatial extraction from tables
|   +-- C04_strategy_flashtext.py        # Lexicon matching (600K+ terms)
|   +-- C05_strategy_glossary.py         # Glossary table extractor
|   +-- C06_strategy_disease.py          # Disease mention detection
|   +-- C07_strategy_drug.py             # Drug/chemical detection
|   +-- C08_strategy_feasibility.py      # Rule-based feasibility extraction
|   +-- C09_strategy_document_metadata.py # Document metadata extraction
|   +-- C10_vision_image_analysis.py     # Vision LLM image analysis
|   +-- C11_llm_feasibility.py           # LLM-based feasibility extraction
|   +-- C18_strategy_pharma.py           # Pharma company detection
|   +-- C13_strategy_author.py           # Author/investigator detection
|   +-- C14_strategy_citation.py         # Citation/reference detection
|   +-- C16_strategy_gene.py             # Gene/genetic entity detection
|
+-- D_validation/                        # LAYER 3: LLM VERIFICATION
|   +-- D01_prompt_registry.py           # Versioned prompts with hash tracking
|   +-- D02_llm_engine.py                # Claude API wrapper (JSON mode)
|   +-- D03_validation_logger.py         # JSONL audit trail
|
+-- E_normalization/                     # LAYER 4: STANDARDIZATION
|   +-- E01_term_mapper.py               # Synonym resolution
|   +-- E02_disambiguator.py             # Context-based disambiguation
|   +-- E03_disease_normalizer.py        # Disease ontology normalization
|   +-- E04_pubtator_enricher.py         # PubTator API (MeSH, aliases)
|   +-- E05_drug_enricher.py             # Drug identifier enrichment
|   +-- E06_nct_enricher.py              # NCT trial metadata enrichment
|   +-- E07_deduplicator.py              # Merge duplicates, pick best LF
|   +-- E15_genetic_enricher.py          # Genetic variant enrichment
|
+-- F_evaluation/                        # LAYER 5: METRICS & TESTING
|   +-- F01_gold_loader.py               # Load gold standard annotations
|   +-- F02_scorer.py                    # Precision/Recall/F1 calculation
|   +-- F03_generator_unit_test.py       # Generator unit tests (no LLM)
|   +-- F04_pipeline_test.py             # Full pipeline evaluation
|   +-- F05_extraction_analysis.py       # Detailed extraction report
|
+-- G_config/                            # CONFIGURATION
|   +-- config.yaml                      # Central configuration (v15.0)
|
+-- H_pipeline/                          # PIPELINE COMPONENTS
|   +-- H01_component_factory.py         # Factory for pipeline components
|   +-- H02_abbreviation_pipeline.py     # Abbreviation extraction pipeline
|   +-- H04_merge_resolver.py            # Multi-source merge resolution
|
+-- I_extraction/                        # ENTITY EXTRACTION
|   +-- I01_entity_processor.py          # Unified entity extraction
|   +-- I02_feasibility_processor.py     # Feasibility data extraction
|
+-- J_export/                            # OUTPUT HANDLERS
|   +-- J01_export_handlers.py           # JSON export for all entity types
|
+-- Z_utils/                             # UTILITIES & SCRIPTS
+-- cache/                               # Runtime cache (PubTator)
+-- orchestrator.py                      # Main pipeline entry point (v1.0)
+-- EXTRACTOR.MD                         # This documentation
```

---

## Output Structure

When processing a PDF, all outputs are saved to a dedicated folder named after the PDF file.

**Example:** Processing `/path/to/ClinicalTrial.pdf` creates:

```
/path/to/ClinicalTrial/
+-- abbreviations_ClinicalTrial_20260115_103045.json
+-- diseases_ClinicalTrial_20260115_103045.json
+-- drugs_ClinicalTrial_20260115_103045.json
+-- genes_ClinicalTrial_20260115_103045.json
+-- pharma_ClinicalTrial_20260115_103045.json
+-- authors_ClinicalTrial_20260115_103045.json
+-- citations_ClinicalTrial_20260115_103045.json
+-- feasibility_ClinicalTrial_20260115_103045.json
+-- figures_ClinicalTrial_20260115_103045.json
+-- tables_ClinicalTrial_20260115_103045.json
+-- metadata_ClinicalTrial_20260115_103045.json
+-- ClinicalTrial_extracted_text_20260115_103045.txt
+-- ClinicalTrial_flowchart_page3_1.png
+-- ClinicalTrial_chart_page5_1.png
+-- ClinicalTrial_chart_page7_1.png
```

### Output File Formats

| File | Content |
|------|---------|
| `abbreviations_*.json` | Validated abbreviation-expansion pairs with confidence, context, provenance |
| `diseases_*.json` | Disease entities with ICD-10, SNOMED, MONDO, ORPHA codes |
| `drugs_*.json` | Drug entities with RxCUI, NDC, MeSH, DrugBank identifiers |
| `genes_*.json` | Gene entities with HGNC symbol, Entrez, Ensembl, disease associations |
| `pharma_*.json` | Pharmaceutical company mentions with headquarters, subsidiaries |
| `authors_*.json` | Authors/investigators with roles, affiliations, ORCID |
| `citations_*.json` | References with PMID, PMCID, DOI, NCT identifiers |
| `feasibility_*.json` | Eligibility criteria, endpoints, epidemiology data |
| `figures_*.json` | Figure metadata with extraction source, Vision LLM analysis |
| `tables_*.json` | Table metadata with VLM extraction, cell structure |
| `metadata_*.json` | Document classification, dates, file info |
| `*_extracted_text_*.txt` | Full text extraction with timestamps |
| `*_{type}_page{N}_{index}.png` | Extracted figures (flowcharts, charts, Kaplan-Meier plots) |

---

## Figures JSON Schema (NEW)

```json
{
  "doc_id": "ClinicalTrial",
  "doc_filename": "ClinicalTrial.pdf",
  "total_images": 4,
  "images": [
    {
      "page": 3,
      "type": "FLOWCHART",
      "caption": "Fig. 1 Patient flow diagram...",
      "extraction_source": "caption_linked",
      "figure_type": "raster",
      "bbox": [119.0, 84.0, 476.2, 413.3],
      "saved_file": "ClinicalTrial_flowchart_page3_1.png",
      "vision_analysis": {
        "analysis_type": "patient_flow",
        "screened": 450,
        "randomized": 300,
        "completed": 275,
        "discontinued": 25,
        "arms": ["Treatment", "Placebo"],
        "exclusion_reasons": [
          {"reason": "Did not meet criteria", "count": 100},
          {"reason": "Declined", "count": 50}
        ]
      }
    },
    {
      "page": 7,
      "type": "CHART",
      "caption": "Fig. 4 Overall survival (left) and Relapse free survival (right)",
      "extraction_source": "caption_linked",
      "figure_type": "raster",
      "bbox": [56.7, 84.0, 538.6, 231.5],
      "saved_file": "ClinicalTrial_chart_page7_1.png",
      "vision_analysis": {
        "analysis_type": "chart_data",
        "chart_type": "kaplan_meier",
        "title": "Survival Analysis",
        "x_axis": "Time (months)",
        "y_axis": "Survival Probability",
        "data_points": [...]
      }
    }
  ]
}
```

**Extraction Source Values:**
| Source | Description |
|--------|-------------|
| `caption_linked` | Figure linked to "Fig X" caption (highest quality) |
| `orphan_native` | Native figure without caption match |
| `layout_model` | Unstructured/YOLOX detection (fallback) |

**Figure Type Values:**
| Type | Description |
|------|-------------|
| `raster` | Embedded image from PDF XObjects |
| `vector` | Vector plot (Kaplan-Meier, etc.) rendered from drawings |
| `layout_model` | Image from Unstructured detection |

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

class ValidationStatus(str, Enum):
    VALIDATED = "validated"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"
    ERROR = "error"

class GeneratorType(str, Enum):
    SYNTAX_PATTERN = "syntax_pattern"      # Schwartz-Hearst
    LEXICON_MATCH = "lexicon_match"        # FlashText dictionary
    TABLE_LAYOUT = "table_layout"          # Spatial extraction
    GLOSSARY_TABLE = "glossary_table"      # Author-provided glossary
    REGEX_PATTERN = "regex_pattern"        # Rigid patterns
```

### A12_gene_models.py - Gene Types

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

    is_alias: bool = False         # Whether matched text is an alias
    locus_type: Optional[str]      # "gene with protein product"
    chromosome: Optional[str]      # "17q21.31"

    associated_diseases: List[GeneDisease]  # OMIM/Orphanet associations
    identifiers: List[GeneIdentifier]
```

### A05_disease_models.py - Disease Types

```python
class ExtractedDisease(BaseModel):
    matched_text: str                  # "C3 glomerulopathy"
    preferred_label: str               # Normalized name
    abbreviation: Optional[str]        # "C3G"

    # Ontology codes
    icd10_code: Optional[str]          # "N05.8"
    snomed_code: Optional[str]         # "236504009"
    mondo_id: Optional[str]            # "MONDO:0020678"
    orpha_code: Optional[str]          # "ORPHA:329918"
    mesh_id: Optional[str]             # "D000081029"

    is_rare_disease: bool = False
    disease_category: Optional[str]    # "renal", "neurological"
    identifiers: List[DiseaseIdentifier]
```

### A06_drug_models.py - Drug Types

```python
class ExtractedDrug(BaseModel):
    matched_text: str                  # "iptacopan"
    preferred_label: str               # "Iptacopan"

    # Identifiers
    rxcui: Optional[str]               # RxNorm code
    ndc_code: Optional[str]            # National Drug Code
    mesh_id: Optional[str]             # MeSH term
    drugbank_id: Optional[str]         # DrugBank ID

    drug_class: Optional[str]          # "complement inhibitor"
    development_phase: Optional[DevelopmentPhase]
    identifiers: List[DrugIdentifier]
```

---

## Generators (C_generators)

### C04_strategy_flashtext.py - Lexicon Matching

**Lexicons Loaded (~617,000 terms):**

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

### C10_vision_image_analysis.py - Vision LLM

```python
class VisionImageAnalyzer:
    """Analyze images using Claude Vision."""

    def analyze_flowchart(self, image_b64: str) -> PatientFlowResult:
        """Extract patient flow data from CONSORT diagrams."""
        # Returns: screened, randomized, completed, discontinued
        # Plus: exclusion reasons, arm assignments

    def analyze_chart(self, image_b64: str) -> ChartDataResult:
        """Extract data points from bar/line/Kaplan-Meier charts."""
        # Returns: chart_type, axes, data_points, statistics
```

### C16_strategy_gene.py - Gene Detection

```python
class GeneDetector:
    """Multi-source gene extraction."""

    # Sources:
    # 1. HGNC lexicon (official symbols + aliases)
    # 2. scispacy NER (GENE semantic types)
    # 3. Pattern-based extraction

    # False positive filtering:
    # - Common words requiring context (e.g., "SET", "MET")
    # - Statistical terms (e.g., "OR", "HR")
    # - Chemical formulas
```

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

options:
  use_llm_validation: true
  use_llm_feasibility: true
  use_vlm_tables: true
  use_normalization: true
  use_native_figure_extraction: true  # NEW: B09-B11 extraction
```

### Figure Extraction Settings

```yaml
figure_extraction:
  min_figure_area_ratio: 0.03    # Minimum 3% of page area
  filter_noise_figures: true      # Remove logos/headers
  vector_detection:
    min_drawing_count: 20         # Minimum drawings for region
    dense_threshold: 50           # Dense region threshold
    min_height: 80                # Minimum height in points
    header_zone_ratio: 0.12       # Exclude top 12%
```

---

## Usage

### Single PDF Extraction

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
print(f"Authors: {len(results.get('authors', []))}")
print(f"Citations: {len(results.get('citations', []))}")
```

### Command Line

```bash
# Run pipeline on PDF folder
python orchestrator.py

# Check figure extraction stats
python -c "
from B_parsing.B09_pdf_native_figures import detect_all_figures
raster, vector = detect_all_figures('path/to/document.pdf')
print(f'Raster: {len(raster)}, Vector: {len(vector)}')
"
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

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
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




---
  Design Section 2 (Revised): Refined Pipeline Architecture

  ┌─────────────────────────────────────────────────────────────────────┐
  │                    STAGE 1: DETECTION + FAST STRUCTURE              │
  │                         (Docling v2.71.0)                           │
  ├─────────────────────────────────────────────────────────────────────┤
  │  PDF ──┬──► Layout detection ──► figure regions + bboxes (pts)      │
  │        │                                                             │
  │        └──► Table detection ──► TableFormer FAST ──► initial struct │
  │                                                                      │
  │  Flag tables for ACCURATE re-run if:                                 │
  │    • Many merged cells / deep header stacks                          │
  │    • Low token coverage in cells                                     │
  │    • Multi-page span detected                                        │
  │                                                                      │
  │  All bboxes stored in PDF points (canonical coordinate space)       │
  └─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │              STAGE 2: RENDERING + CAPTION EXTRACTION                │
  │                    (PyMuPDF + PDF text extraction)                  │
  ├─────────────────────────────────────────────────────────────────────┤
  │  For each candidate:                                                 │
  │    1. Expand bbox by point-based padding (6-12pt sides, 36-72pt     │
  │       bottom/top depending on expected caption location)             │
  │    2. Caption extraction (multisource):                              │
  │       a. PDF text blocks near visual (preferred for born-digital)   │
  │       b. OCR fallback only if no reliable PDF text found            │
  │    3. Detect continuation cues early:                                │
  │       • "(continued)" / "(cont.)" in caption                         │
  │       • Repeated header structure on next page                       │
  │       • Same reference number across pages                           │
  │       • Matching column geometry on consecutive pages                │
  │    4. Render at 200-300 DPI (adaptive based on visual size)         │
  │    5. Encode as base64 PNG                                           │
  │                                                                      │
  │  Caption provenance tracked: {pdf_text | ocr | vlm}                 │
  └─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    STAGE 3: TRIAGE + VLM ENRICHMENT                 │
  │                      (Claude Sonnet Vision)                         │
  ├─────────────────────────────────────────────────────────────────────┤
  │  TRIAGE (cheap signals first):                                       │
  │                                                                      │
  │  SKIP (no VLM):                     │  VLM PATH:                     │
  │  • Tiny logos/separators            │  • Has caption-like text       │
  │  • Repeated header/footer graphics  │  • Docling flagged as table    │
  │  • No caption + low area + no refs  │  • Dense grid structure        │
  │                                      │  • Referenced in body text    │
  │                                      │  • Flagged for ACCURATE re-run│
  ├──────────────────────────────────────┴───────────────────────────────┤
  │  VLM tasks (for candidates that pass triage):                        │
  │    1. Classify: table vs figure (with confidence)                    │
  │    2. Parse reference from caption: "Table 1" → {type, numbers}      │
  │    3. Validate/correct caption text                                  │
  │    4. For tables: validate Docling structure, flag if misparsed      │
  │    5. If misparsed + complex → trigger ACCURATE re-run               │
  │    6. Confirm/detect multi-page continuation                         │
  └─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                  STAGE 4: DOCUMENT-LEVEL RESOLUTION                 │
  │                      (Cross-reference + merge)                      │
  ├─────────────────────────────────────────────────────────────────────┤
  │  1. Scan body text for mentions ("see Figure 2", "Table 1 shows")   │
  │  2. Parse references from body text                                  │
  │  3. Reconcile: caption reference wins, body fills missing           │
  │  4. Link mentions to ExtractedVisual by reference number            │
  │  5. Infer section context from document structure                    │
  │  6. Merge multi-page visuals (using Stage 2/3 continuation flags)   │
  │  7. Deduplicate overlapping detections                               │
  │  8. Final validation pass                                            │
  └─────────────────────────────────────────────────────────────────────┘

  ---


  Escalation flow:

  FAST extraction
        │
        ▼
  Compute complexity signals
        │
        ▼
  should_escalate_to_accurate()?
        │
        ├─► No  ──► Use FAST result
        │
        └─► Yes ──► Run ACCURATE ──► Use ACCURATE result
                          │
                          ▼
                    (Optional) VLM validation
                          │
                          ▼
                    If still misparsed → flag for manual review