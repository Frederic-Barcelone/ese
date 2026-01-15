# Metadata Extraction Pipeline

A production-grade 6-layer pipeline for extracting structured metadata from clinical trial and medical PDF documents.

**Extraction Capabilities:**
- Abbreviations and acronyms with definitions
- Diseases (rare diseases, conditions, syndromes)
- Drugs (approved, investigational, compounds)
- Clinical trial feasibility data (eligibility, endpoints, patient journey)
- Document metadata (classification, dates, identifiers)
- Images with Vision LLM analysis (flowcharts, charts, figures)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATOR (orchestrator.py)                     │
│                                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │ A_core   │──▶│ B_parsing│──▶│C_generate│──▶│D_validate│──▶│E_normalize│ │
│  │ Models   │   │ PDF→Doc  │   │ Candidates│   │ LLM/Rules│   │ Enrich   │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│       │              │              │              │              │          │
│       │              │              │              │              ▼          │
│       │              │              │              │        ┌──────────┐    │
│       │              │              │              │        │F_evaluate│    │
│       │              │              │              │        │ Metrics  │    │
│       │              │              │              │        └──────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
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
    │
    ▼
┌─────────────────────────────────────────┐
│ B_parsing: PDF → DocumentGraph          │
│ • Unstructured.io partition_pdf()       │
│ • Layout detection (columns, tables)    │
│ • Image extraction (base64 embedded)    │
│ • Section detection                     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ C_generators: DocumentGraph → Candidates│
│ • C01: Schwartz-Hearst syntax patterns  │
│ • C04: FlashText lexicon (250K+ terms)  │
│ • C05: Glossary table extraction        │
│ • C06: Disease detection                │
│ • C07: Drug detection                   │
│ • C08: Feasibility extraction           │
│ • C09: Document metadata                │
│ • C10: Vision image analysis            │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ D_validation: Candidates → Validated    │
│ • Heuristic shortcuts (PASO A-D)        │
│ • Claude LLM batch validation           │
│ • Structured JSON responses             │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ E_normalization: Validated → Enriched   │
│ • PubTator API (MeSH, disease codes)    │
│ • NCT enrichment (trial metadata)       │
│ • Deduplication (merge same SF)         │
└─────────────────────────────────────────┘
    │
    ▼
Output Directory (named after PDF)
├── abbreviations_{stem}_{timestamp}.json
├── diseases_{stem}_{timestamp}.json
├── drugs_{stem}_{timestamp}.json
├── feasibility_{stem}_{timestamp}.json
├── images_{stem}_{timestamp}.json
├── metadata_{stem}_{timestamp}.json
├── {stem}_extracted_text_{timestamp}.txt
└── {stem}_{type}_page{N}_{index}.jpg
```

---

## Directory Structure

```
corpus_metadata/
│
├── A_core/                              # LAYER 0: FOUNDATION
│   ├── A01_domain_models.py             # Core Pydantic schemas (Candidate, ExtractedEntity)
│   ├── A02_interfaces.py                # Abstract Base Classes (generators, validators)
│   ├── A03_provenance.py                # Run ID generation, git hash, traceability
│   ├── A04_heuristics_config.py         # Centralized heuristics (whitelists/blacklists)
│   ├── A05_disease_models.py            # Disease domain models (ICD-10, SNOMED, ORPHA)
│   ├── A06_drug_models.py               # Drug domain models (RxCUI, NDC, DrugBank)
│   ├── A07_feasibility_models.py        # Clinical trial feasibility models
│   └── A08_document_metadata_models.py  # Document-level metadata models
│
├── B_parsing/                           # LAYER 1: DOCUMENT STRUCTURE
│   ├── B01_pdf_to_docgraph.py           # PDF parsing with Unstructured.io
│   ├── B02_doc_graph.py                 # Internal model (Page → Block → Token)
│   ├── B03_table_extractor.py           # Table detection and JSON serialization
│   ├── B04_column_ordering.py           # Multi-column layout detection
│   ├── B05_section_detector.py          # Section/heading detection
│   ├── B06_confidence.py                # Confidence scoring features
│   └── B07_negation.py                  # Negation/assertion detection
│
├── C_generators/                        # LAYER 2: CANDIDATE GENERATION
│   ├── C00_strategy_identifiers.py      # Database IDs (OMIM, DOI, NCT, ORCID)
│   ├── C01_strategy_abbrev.py           # Schwartz-Hearst syntax patterns
│   ├── C02_strategy_regex.py            # Rigid patterns (Trial IDs, DOIs)
│   ├── C03_strategy_layout.py           # Spatial extraction from tables
│   ├── C04_strategy_flashtext.py        # Lexicon matching (250K+ terms)
│   ├── C05_strategy_glossary.py         # Glossary table extractor
│   ├── C06_strategy_disease.py          # Disease mention detection
│   ├── C07_strategy_drug.py             # Drug/chemical detection
│   ├── C08_strategy_feasibility.py      # Rule-based feasibility extraction
│   ├── C11_llm_feasibility.py           # LLM-based feasibility extraction
│   ├── C09_strategy_document_metadata.py # Document metadata extraction
│   └── C10_vision_image_analysis.py     # Vision LLM image analysis
│
├── D_validation/                        # LAYER 3: LLM VERIFICATION
│   ├── D01_prompt_registry.py           # Versioned prompts with hash tracking
│   ├── D02_llm_engine.py                # Claude API wrapper (JSON mode)
│   └── D03_validation_logger.py         # JSONL audit trail
│
├── E_normalization/                     # LAYER 4: STANDARDIZATION
│   ├── E01_term_mapper.py               # Synonym resolution
│   ├── E02_disambiguator.py             # Context-based disambiguation
│   ├── E03_disease_normalizer.py        # Disease ontology normalization
│   ├── E04_pubtator_enricher.py         # PubTator API (MeSH, aliases)
│   ├── E05_drug_enricher.py             # Drug identifier enrichment
│   ├── E06_nct_enricher.py              # NCT trial metadata enrichment
│   └── E07_deduplicator.py              # Merge duplicates, pick best LF
│
├── F_evaluation/                        # LAYER 5: METRICS & TESTING
│   ├── F01_gold_loader.py               # Load gold standard annotations
│   ├── F02_scorer.py                    # Precision/Recall/F1 calculation
│   ├── F03_generator_unit_test.py       # Generator unit tests (no LLM)
│   ├── F04_pipeline_test.py             # Full pipeline evaluation
│   └── F05_extraction_analysis.py       # Detailed extraction report
│
├── G_config/                            # CONFIGURATION
│   └── config.yaml                      # Central configuration (v15.0)
│
├── Z_tests/                             # DEVELOPMENT TESTS
├── cache/                               # Runtime cache (PubTator)
├── orchestrator.py                      # Main pipeline entry point (v0.8)
└── EXTRACTOR.MD                         # This documentation
```

---

## Output Structure

When processing a PDF, all outputs are saved to a dedicated folder named after the PDF file.

**Example:** Processing `/path/to/ClinicalTrial.pdf` creates:

```
/path/to/ClinicalTrial/
├── abbreviations_ClinicalTrial_20260115_103045.json
├── diseases_ClinicalTrial_20260115_103045.json
├── drugs_ClinicalTrial_20260115_103045.json
├── feasibility_ClinicalTrial_20260115_103045.json
├── images_ClinicalTrial_20260115_103045.json
├── metadata_ClinicalTrial_20260115_103045.json
├── ClinicalTrial_extracted_text_20260115_103045.txt
├── ClinicalTrial_flowchart_page3_1.jpg
├── ClinicalTrial_chart_page5_1.jpg
└── ClinicalTrial_figure_page8_1.jpg
```

### Output File Formats

| File | Content |
|------|---------|
| `abbreviations_*.json` | Validated abbreviation-expansion pairs with confidence, context, provenance |
| `diseases_*.json` | Disease entities with ICD-10, SNOMED, MONDO, ORPHA codes |
| `drugs_*.json` | Drug entities with RxCUI, NDC, MeSH, DrugBank identifiers |
| `feasibility_*.json` | Eligibility criteria, endpoints, epidemiology data |
| `images_*.json` | Image metadata with Vision LLM analysis (flowcharts, charts) |
| `metadata_*.json` | Document classification, dates, file info |
| `*_extracted_text_*.txt` | Full text extraction with timestamps |
| `*_{type}_page{N}_{index}.jpg` | Extracted images (flowcharts, charts, figures) |

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

### A02_interfaces.py - Abstract Contracts

```python
class BaseCandidateGenerator(ABC):
    @abstractmethod
    def extract(self, doc: DocumentGraph) -> List[Candidate]:
        """Extract candidates from document."""

class BaseVerifier(ABC):
    @abstractmethod
    def verify(self, candidates: List[Candidate]) -> List[ExtractedEntity]:
        """Validate candidates using LLM."""

class BaseNormalizer(ABC):
    @abstractmethod
    def normalize(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Standardize and enrich entities."""
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

**Lexicons Loaded (~500,000+ terms):**

| Lexicon | Terms | Source | URL |
|---------|-------|--------|-----|
| abbreviation_general | 5,392 | Curated abbreviations | Internal |
| **meta_inventory** | **104,057** | **Clinical abbreviations (170K senses)** | [GitHub](https://github.com/lisavirginia/clinical-abbreviations) |
| rare_disease_acronyms | 1,630 | Rare disease names | Internal |
| umls_biological | 97,336 | UMLS biomedical | [NLM](https://www.nlm.nih.gov/research/umls/) |
| umls_clinical | 20,000 | UMLS clinical | [NLM](https://www.nlm.nih.gov/research/umls/) |
| trial_acronyms | 125,454 | ClinicalTrials.gov | [CT.gov](https://clinicaltrials.gov/) |
| pro_scales | 299 | Patient-Reported Outcomes | Internal |
| disease_pah/anca/igan | 167 | Indication-specific | Internal |
| **mondo_diseases** | **50,000+** | **Unified disease ontology** | [MONDO](https://mondo.monarchinitiative.org/) |
| **chembl_drugs** | **15,000+** | **Approved drugs with bioactivity** | [ChEMBL](https://www.ebi.ac.uk/chembl/) |

**New Public Lexicons (2025):**

1. **[Meta-Inventory](https://github.com/lisavirginia/clinical-abbreviations)** - 104,057 abbreviations with 170,426 senses
   - Most complete compilation of medical abbreviations in American English
   - Increases abbreviation coverage by 28-52% over previous resources
   - Source: [Nature Scientific Data](https://www.nature.com/articles/s41597-021-00929-4)

2. **[MONDO](https://mondo.monarchinitiative.org/)** - Unified disease ontology
   - Harmonizes OMIM, Orphanet, EFO, DOID, ICD-11, NCIt
   - Provides precise semantic mappings (1:1 equivalences)
   - Available in SSSOM format for interoperability

3. **[ChEMBL](https://www.ebi.ac.uk/chembl/)** - Open drug database
   - 2.4M+ compounds with bioactivity data
   - ~15K approved drugs (max_phase = 4)
   - CC0 license for open data subset

**Components:**
- FlashText: O(n) keyword matching
- scispacy: NER with UMLS EntityLinker
- Schwartz-Hearst: Abbreviation detector

### C06_strategy_disease.py - Disease Detection

```python
class DiseaseDetector:
    """Multi-source disease extraction."""

    def extract(self, doc: DocumentGraph) -> List[DiseaseCandidate]:
        # 1. FlashText lexicon matching
        # 2. scispacy NER (DISEASE semantic types)
        # 3. Orphanet rare disease lookup
        # 4. Pattern-based extraction
```

**False Positive Filtering:**
- Chromosome patterns (Chr1, 1p36, 22q11)
- Gene symbols (BRCA1, TP53)
- Chemical compounds (H2O, CO2)

### C07_strategy_drug.py - Drug Detection

```python
class DrugDetector:
    """Multi-source drug extraction."""

    # Lexicon priority (highest to lowest):
    # 1. Alexion pipeline drugs (specialized)
    # 2. Investigational drugs (ClinicalTrials.gov)
    # 3. FDA approved drugs
    # 4. RxNorm general terms
    # 5. scispacy CHEMICAL NER
```

### C10_vision_image_analysis.py - Vision LLM

```python
class VisionImageAnalyzer:
    """Analyze images using Claude Vision."""

    def analyze_flowchart(self, image_b64: str) -> PatientFlowResult:
        """Extract patient flow data from CONSORT diagrams."""
        # Returns: screened, randomized, completed, discontinued
        # Plus: exclusion reasons, arm assignments

    def analyze_chart(self, image_b64: str) -> ChartDataResult:
        """Extract data points from bar/line charts."""
        # Returns: chart_type, axes, data_points, statistics
```

---

## Validation (D_validation)

### D02_llm_engine.py - Claude Integration

```python
class LLMEngine:
    """Structured validation using Claude API."""

    def validate_batch(
        self,
        candidates: List[Candidate],
        context: str
    ) -> List[ValidationResult]:
        """
        Batch validation with JSON output.

        - Uses claude-sonnet-4 for validation
        - Structured prompts from D01_prompt_registry
        - Returns VALIDATED/REJECTED/AMBIGUOUS
        """
```

### Heuristic Shortcuts (PASO A-D)

| PASO | Rule | Example |
|------|------|---------|
| **A** | Auto-approve statistics with numeric context | CI 95%, SD 2.3, HR 0.65 |
| **B** | Auto-reject country codes | US, UK, EU, FR, DE |
| **C** | Detect hyphenated abbreviations | CKD-EPI, sC5b-9, FACIT-Fatigue |
| **D** | LLM extraction for undefined SFs | Abbreviations without definitions |

---

## Normalization (E_normalization)

### E04_pubtator_enricher.py - PubTator Integration

```python
class PubTator3Client:
    """NCBI PubTator3 API client."""

    # Rate limit: 3 req/sec (NCBI guidelines)
    # Cache: Disk-based, 7-day TTL

    def autocomplete(self, term: str, entity_type: str) -> List[Dict]:
        """Query PubTator for entity normalization."""
        # Returns: MeSH ID, normalized name, aliases
```

### E07_deduplicator.py - Duplicate Resolution

**Quality Ranking (highest priority first):**

1. `GLOSSARY_TABLE` - Author-provided definitions
2. `SYNTAX_PATTERN` - Schwartz-Hearst from text
3. `TABLE_LAYOUT` - Extracted from tables
4. `LEXICON_MATCH` - Dictionary lookup

```python
class Deduplicator:
    """Merge same-SF entries, pick best long_form."""

    def deduplicate(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        # Group by short_form
        # Select highest-quality long_form
        # Store alternatives in normalized_value.deduplication.alternatives
```

---

## Configuration (G_config/config.yaml)

### Key Sections

```yaml
# Feature flags
features:
  drug_detection: true
  disease_detection: true
  abbreviation_extraction: true
  pubtator_enrichment: true
  ai_validation: true

# Generator settings
generators:
  syntax_pattern:
    enabled: true
  lexicon:
    enabled: true
    context_window: 300
    min_abbrev_length: 2

# Heuristics
heuristics:
  stats_abbrevs:        # PASO A whitelist
    CI: "confidence interval"
    SD: "standard deviation"
  sf_blacklist:         # Auto-reject list
    - "US"
    - "UK"
    - "MD"
  hyphenated_abbrevs:   # PASO C patterns
    CKD-EPI: "Chronic Kidney Disease Epidemiology Collaboration"

# API configuration
api:
  pubtator:
    rate_limit_per_second: 3
    cache:
      ttl_hours: 168
  claude:
    validation:
      model: "claude-sonnet-4-20250514"
      temperature: 0
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
```

### Batch Processing

```python
from pathlib import Path

pdf_dir = Path("/path/to/pdfs")
for pdf in pdf_dir.glob("*.pdf"):
    results = orch.process_pdf(str(pdf))
```

### Pipeline Evaluation

```bash
# Run evaluation against gold standard
python -m corpus_metadata.F_evaluation.F04_pipeline_test

# Output:
# MICRO (global):
#   Precision: 85.2%
#   Recall:    78.4%
#   F1:        81.6%
```

### Environment Setup

```bash
# Required environment variable
export ANTHROPIC_API_KEY="your-api-key"

# Install dependencies
pip install anthropic pyyaml python-dotenv pydantic
pip install unstructured[pdf]  # PDF parsing
pip install scispacy           # NER
pip install flashtext          # Fast lexicon matching
```

---

## Dependencies

### Core Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `anthropic` | ≥0.18 | Claude API client |
| `pydantic` | ≥2.0 | Data validation |
| `pyyaml` | ≥6.0 | Configuration |
| `python-dotenv` | ≥1.0 | Environment |

### NLP/Parsing

| Package | Version | Purpose |
|---------|---------|---------|
| `unstructured[pdf]` | ≥0.10 | PDF parsing |
| `scispacy` | ≥0.5 | Biomedical NER |
| `flashtext` | ≥2.7 | Fast keyword matching |
| `pymupdf` (fitz) | ≥1.23 | PDF metadata |

### Optional

| Package | Purpose |
|---------|---------|
| `en_core_sci_lg` | scispacy large model |
| `scispacy-umls` | UMLS entity linker |

---

## Evaluation Metrics

### Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision** | TP / (TP + FP) | "Of extractions, how many correct?" |
| **Recall** | TP / (TP + FN) | "Of gold items, how many found?" |
| **F1** | 2 × (P × R) / (P + R) | Harmonic mean |

### Classification

- **True Positive (TP):** System found, Gold confirms
- **False Positive (FP):** System found, Gold denies
- **False Negative (FN):** System missed, Gold has it

### Target Performance

| Entity Type | Target F1 |
|-------------|-----------|
| Abbreviations | >80% |
| Diseases | >75% |
| Drugs | >85% |

---

## Extending the Pipeline

### Adding a New Generator

1. Create `C_generators/C11_strategy_new.py`
2. Implement `BaseCandidateGenerator` interface:

```python
from A_core.A02_interfaces import BaseCandidateGenerator
from A_core.A01_domain_models import Candidate

class NewGenerator(BaseCandidateGenerator):
    def extract(self, doc: DocumentGraph) -> List[Candidate]:
        candidates = []
        for page in doc.pages:
            for block in page.blocks:
                # Your extraction logic
                pass
        return candidates
```

3. Register in `orchestrator.py`:

```python
from C_generators.C11_strategy_new import NewGenerator

# In Orchestrator.__init__():
self.new_generator = NewGenerator(config)

# In Orchestrator._run_generators():
candidates.extend(self.new_generator.extract(doc))
```

### Adding a New Entity Type

1. Define models in `A_core/A0N_new_models.py`
2. Create generator in `C_generators/`
3. Add validation prompts in `D_validation/D01_prompt_registry.py`
4. Create enricher in `E_normalization/` (optional)
5. Add export method in `orchestrator.py`

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "scispacy model not found" | Model not installed | `pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz` |
| "API rate limit exceeded" | Too many Claude calls | Increase `batch_delay_ms` in config |
| "PubTator timeout" | API slow/unavailable | Check network; results cached locally |
| Empty disease results | Lexicons not loaded | Verify `paths.dictionaries` in config |

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set in config.yaml:
logging:
  level: "DEBUG"
  console: true
```

---

## References

- **NLP4RARE Corpus:** https://github.com/isegura/NLP4RARE-CM-UC3M
- **Schwartz-Hearst Algorithm:** DOI:10.1093/bioinformatics/btg014
- **PubTator3 API:** https://www.ncbi.nlm.nih.gov/research/pubtator3/api
- **Unstructured.io:** https://unstructured.io/
- **scispacy:** https://allenai.github.io/scispacy/

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.8 | 2026-01 | Output folder restructure, image file saving, Vision LLM analysis |
| 0.7 | 2025-12 | Image extraction, Vision LLM integration |
| 0.6 | 2025-11 | Feasibility extraction, document metadata |
| 0.5 | 2025-10 | Drug detection, PubTator enrichment |
| 0.4 | 2025-09 | Disease detection, NCT enrichment |
| 0.3 | 2025-08 | Multi-generator architecture |
| 0.2 | 2025-07 | Validation layer, heuristics |
| 0.1 | 2025-06 | Initial pipeline |
