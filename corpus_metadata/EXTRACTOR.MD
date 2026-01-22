
# doc : https://chunkforge.com/blog/nlp-named-entity-recognition
# Ralph Wiggum: Continuous Agent Loops in Claude Code


# Metadata Extraction Pipeline

A production-grade 6-layer pipeline for extracting structured metadata from clinical trial and medical PDF documents.

**Extraction Capabilities:**
- Abbreviations and acronyms with definitions
- Diseases (rare diseases, conditions, syndromes)
- Drugs (approved, investigational, compounds)
- Pharmaceutical companies (sponsors, manufacturers)
- Authors and investigators (names, roles, affiliations)
- Citations and references (PMID, DOI, NCT, URLs)
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
│ • C12: Pharma company detection         │
│ • C13: Author/investigator detection    │
│ • C14: Citation/reference detection     │
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
├── pharma_{stem}_{timestamp}.json
├── authors_{stem}_{timestamp}.json
├── citations_{stem}_{timestamp}.json
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
│   ├── A08_document_metadata_models.py  # Document-level metadata models
│   ├── A09_pharma_models.py             # Pharma company models
│   ├── A10_author_models.py             # Author/investigator models
│   └── A11_citation_models.py           # Citation/reference models
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
│   ├── C09_strategy_document_metadata.py # Document metadata extraction
│   ├── C10_vision_image_analysis.py     # Vision LLM image analysis
│   ├── C11_llm_feasibility.py           # LLM-based feasibility extraction
│   ├── C12_strategy_pharma.py           # Pharma company detection
│   ├── C13_strategy_author.py           # Author/investigator detection
│   └── C14_strategy_citation.py         # Citation/reference detection
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
├── Z_utils/                             # UTILITIES & SCRIPTS
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
├── pharma_ClinicalTrial_20260115_103045.json
├── authors_ClinicalTrial_20260115_103045.json
├── citations_ClinicalTrial_20260115_103045.json
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
| `pharma_*.json` | Pharmaceutical company mentions with headquarters, subsidiaries |
| `authors_*.json` | Authors/investigators with roles, affiliations, ORCID |
| `citations_*.json` | References with PMID, PMCID, DOI, NCT identifiers |
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

### A10_author_models.py - Author/Investigator Types

```python
class AuthorRoleType(str, Enum):
    """Role of the author in the document."""
    AUTHOR = "author"
    PRINCIPAL_INVESTIGATOR = "principal_investigator"
    CO_INVESTIGATOR = "co_investigator"
    CORRESPONDING_AUTHOR = "corresponding_author"
    STEERING_COMMITTEE = "steering_committee"
    STUDY_CHAIR = "study_chair"
    DATA_SAFETY_BOARD = "data_safety_board"
    UNKNOWN = "unknown"

class AuthorGeneratorType(str, Enum):
    """Tracks which strategy produced the author candidate."""
    HEADER_PATTERN = "gen:author_header"        # From document header
    AFFILIATION_BLOCK = "gen:author_affiliation" # From affiliation section
    CONTRIBUTION_SECTION = "gen:author_contribution"
    INVESTIGATOR_LIST = "gen:author_investigator"
    REGEX_PATTERN = "gen:author_regex"

class AuthorCandidate(BaseModel):
    """Pre-validation author mention."""
    full_name: str                     # "John Smith"
    role: AuthorRoleType               # principal_investigator
    affiliation: Optional[str]         # "Harvard Medical School"
    email: Optional[str]               # "jsmith@hms.harvard.edu"
    orcid: Optional[str]               # "0000-0001-2345-6789"
    context_text: str
    context_location: Coordinate
    initial_confidence: float

class ExtractedAuthor(BaseModel):
    """Validated author entity for output."""
    full_name: str
    role: AuthorRoleType
    affiliation: Optional[str]
    email: Optional[str]
    orcid: Optional[str]
    primary_evidence: EvidenceSpan
    status: ValidationStatus
    confidence_score: float

class AuthorExportEntry(BaseModel):
    """Simplified author entry for JSON export."""
    full_name: str
    role: str                          # "principal_investigator"
    affiliation: Optional[str]
    email: Optional[str]
    orcid: Optional[str]
    context: Optional[str]
    page: Optional[int]
    confidence: float
```

**Example Output:**
```json
{
  "run_id": "RUN_AUTHOR_20260118_120000_abc123",
  "timestamp": "2026-01-18T12:00:00",
  "document": "ClinicalTrial.pdf",
  "pipeline_version": "0.8",
  "total_detected": 5,
  "unique_authors": 5,
  "authors": [
    {
      "full_name": "John Smith",
      "role": "principal_investigator",
      "affiliation": "Harvard Medical School",
      "email": "jsmith@hms.harvard.edu",
      "orcid": "0000-0001-2345-6789",
      "context": "Principal Investigator: John Smith, MD, PhD...",
      "page": 1,
      "confidence": 0.95
    }
  ]
}
```

### A11_citation_models.py - Citation/Reference Types

```python
class CitationIdentifierType(str, Enum):
    """Type of citation identifier found."""
    PMID = "pmid"                      # PubMed ID (e.g., "12345678")
    PMCID = "pmcid"                    # PubMed Central ID (e.g., "PMC1234567")
    DOI = "doi"                        # Digital Object Identifier
    NCT = "nct"                        # ClinicalTrials.gov ID
    URL = "url"                        # Web URL

class CitationGeneratorType(str, Enum):
    """Tracks which strategy produced the citation candidate."""
    REGEX_PATTERN = "gen:citation_regex"
    REFERENCE_SECTION = "gen:citation_reference"
    INLINE_CITATION = "gen:citation_inline"

class CitationCandidate(BaseModel):
    """Pre-validation citation mention."""
    pmid: Optional[str]                # "12345678"
    pmcid: Optional[str]               # "PMC1234567"
    doi: Optional[str]                 # "10.1000/xyz123"
    nct: Optional[str]                 # "NCT01234567"
    url: Optional[str]                 # "https://..."
    citation_text: str                 # Full citation string
    citation_number: Optional[int]     # Reference number [1], [2], etc.
    identifier_types: List[CitationIdentifierType]
    context_text: str
    initial_confidence: float

class ExtractedCitation(BaseModel):
    """Validated citation entity for output."""
    pmid: Optional[str]
    pmcid: Optional[str]
    doi: Optional[str]
    nct: Optional[str]
    url: Optional[str]
    citation_text: str
    citation_number: Optional[int]
    primary_evidence: EvidenceSpan
    status: ValidationStatus
    confidence_score: float

class CitationExportEntry(BaseModel):
    """Simplified citation entry for JSON export."""
    pmid: Optional[str]
    pmcid: Optional[str]
    doi: Optional[str]
    nct: Optional[str]
    url: Optional[str]
    citation_text: str
    citation_number: Optional[int]
    page: Optional[int]
    confidence: float
```

**Example Output:**
```json
{
  "run_id": "RUN_CITATION_20260118_120000_def456",
  "timestamp": "2026-01-18T12:00:00",
  "document": "ClinicalTrial.pdf",
  "pipeline_version": "0.8",
  "total_detected": 25,
  "unique_identifiers": 22,
  "citations": [
    {
      "pmid": "12345678",
      "pmcid": null,
      "doi": "10.1056/NEJMoa2025314",
      "nct": "NCT04817618",
      "url": null,
      "citation_text": "Smith J et al. Iptacopan in C3G. N Engl J Med. 2024;390:1-12.",
      "citation_number": 1,
      "page": 15,
      "confidence": 0.98
    }
  ]
}
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

### C12_strategy_pharma.py - Pharma Company Detection

```python
class PharmaCompanyDetector:
    """Detect pharmaceutical company mentions using lexicon matching."""

    # Uses FlashText for fast keyword matching against pharma_companies_lexicon.json
    # Includes: canonical name, full legal name, headquarters, parent/subsidiaries

    def detect(self, doc_graph, doc_id, doc_fingerprint) -> List[PharmaCandidate]:
        # Matches company names and variants
        # Deduplicates by canonical name
        # Returns with provenance metadata
```

### C13_strategy_author.py - Author/Investigator Detection

```python
class AuthorDetector:
    """Detect author/investigator mentions using regex patterns."""

    # Name with credentials pattern
    NAME_WITH_CREDENTIALS = r"([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)(?:,\s*(?:MD|PhD|MPH|DO|PharmD|RN|FRCPC?))+,"

    # Role-prefixed patterns (higher confidence)
    ROLE_PATTERNS = [
        (r"Principal\s+[Ii]nvestigator[s]?[:\s]+(...)", PRINCIPAL_INVESTIGATOR),
        (r"Corresponding\s+[Aa]uthor[s]?[:\s]+(...)", CORRESPONDING_AUTHOR),
        (r"Co-?[Ii]nvestigator[s]?[:\s]+(...)", CO_INVESTIGATOR),
        (r"Study\s+[Cc]hair[:\s]+(...)", STUDY_CHAIR),
        (r"Steering\s+[Cc]ommittee[:\s]+(...)", STEERING_COMMITTEE),
    ]

    # Additional extractions
    EMAIL_PATTERN = r"[\w.+-]+@[\w.-]+\.\w+"
    ORCID_PATTERN = r"(\d{4}-\d{4}-\d{4}-\d{3}[\dX])"

    def detect(self, doc_graph, doc_id, doc_fingerprint, full_text) -> List[AuthorCandidate]:
        # 1. Detect role-prefixed names (high confidence: 0.95)
        # 2. Detect names with credentials (medium confidence: 0.85)
        # 3. Extract email and ORCID from context
        # 4. Infer role from surrounding text

    def validate_candidates(self, candidates) -> List[ExtractedAuthor]:
        # Auto-validate pattern matches
        # Create evidence spans

    def export_to_json(self, validated, doc_id) -> AuthorExportDocument:
        # Generate JSON export with unique author count
```

**Detection Examples:**
| Pattern | Example Match | Role |
|---------|---------------|------|
| Role-prefixed | "Principal Investigator: John Smith, MD" | `principal_investigator` |
| Credentials | "Jane Doe, MD, PhD" | `author` |
| Corresponding | "Corresponding Author: Bob Wilson" | `corresponding_author` |
| Committee | "Steering Committee: Alice Brown" | `steering_committee` |

### C14_strategy_citation.py - Citation/Reference Detection

```python
class CitationDetector:
    """Detect citation/reference mentions using regex patterns."""

    CITATION_PATTERNS = {
        "pmid": [
            r"PMID[:\s]*(\d{7,8})",
            r"PubMed\s*(?:ID)?[:\s]*(\d{7,8})",
        ],
        "pmcid": [
            r"PMC\s*(\d{6,8})",
            r"PMCID[:\s]*PMC?(\d{6,8})",
        ],
        "doi": [
            r"(?:doi[:\s]*)?10\.\d{4,9}/[^\s\])<>\"',;]+",
            r"https?://doi\.org/10\.\d{4,9}/[^\s\])<>\"',;]+",
        ],
        "nct": [
            r"NCT\s*(\d{8})",
            r"ClinicalTrials\.gov[:\s]*(?:NCT)?(\d{8})",
        ],
        "url": [
            r"https?://(?:www\.)?[^\s\])<>\"']+",
        ],
    }

    def detect(self, doc_graph, doc_id, doc_fingerprint, full_text) -> List[CitationCandidate]:
        # 1. Detect each identifier type with regex
        # 2. Validate identifier formats
        # 3. Extract citation text context
        # 4. Merge related citations (same reference, multiple IDs)

    def _merge_related_citations(self, candidates) -> List[CitationCandidate]:
        # Combine PMID + DOI from same reference
        # Use longest citation text
        # Take highest confidence

    def validate_candidates(self, candidates) -> List[ExtractedCitation]:
        # Auto-validate valid identifier formats
        # Build evidence with all identifiers

    def export_to_json(self, validated, doc_id) -> CitationExportDocument:
        # Generate JSON export with unique identifier count
```

**Identifier Confidence Scores:**
| Type | Confidence | Reason |
|------|------------|--------|
| DOI | 0.98 | Highly structured, unique |
| PMID | 0.95 | Standard 7-8 digit format |
| PMCID | 0.95 | PMC prefix is distinctive |
| NCT | 0.95 | ClinicalTrials.gov standard |
| URL | 0.80 | May be non-citation links |

**Detection Examples:**
| Pattern | Example Match | Type |
|---------|---------------|------|
| PMID | "PMID: 12345678" | `pmid` |
| DOI | "doi:10.1056/NEJMoa2025314" | `doi` |
| PMCID | "PMC7654321" | `pmcid` |
| NCT | "NCT04817618" | `nct` |
| URL DOI | "https://doi.org/10.1000/xyz" | `doi` |

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

### D01_prompt_registry.py - Validation Prompts

**Available Prompt Tasks:**

| Task | Purpose |
|------|---------|
| `VERIFY_DEFINITION_PAIR` | Validate SF→LF abbreviation mappings |
| `VERIFY_SHORT_FORM_ONLY` | Validate orphan abbreviations |
| `VERIFY_BATCH` | Batch abbreviation validation |
| `FAST_REJECT` | Haiku screening for obvious rejections |
| `VERIFY_DISEASE` | Single disease mention validation |
| `VERIFY_DISEASE_BATCH` | Batch disease validation |
| `VERIFY_AUTHOR_BATCH` | Batch author/investigator validation |
| `VERIFY_CITATION_BATCH` | Batch citation/reference validation |

### Heuristic Shortcuts (PASO A-D)

| PASO | Rule | Example |
|------|------|---------|
| **A** | Auto-approve statistics with numeric context | CI 95%, SD 2.3, HR 0.65 |
| **B** | Auto-reject country codes | US, UK, EU, FR, DE |
| **C** | Detect hyphenated abbreviations + trial enrichment | CKD-EPI, APPEAR-C3G |
| **D** | LLM extraction for undefined SFs | Abbreviations without definitions |

---

## Normalization (E_normalization)

### E06_nct_enricher.py - Trial Acronym Enrichment

```python
class TrialAcronymEnricher:
    """Enrich trial acronyms from ClinicalTrials.gov API."""

    def search_by_acronym(self, acronym: str) -> NCTTrialInfo:
        """Search for trial by acronym, returns official title."""
        # Returns: nct_id, official_title, brief_title, conditions, etc.

# Convenience function
def enrich_trial_acronym(acronym: str) -> Optional[str]:
    """Get official trial title for an acronym."""
```

**Example:**
```python
>>> enrich_trial_acronym("APPEAR-C3G")
"A Multicenter, Randomized, Double-blind Study to Evaluate
the Efficacy and Safety of Iptacopan in C3 Glomerulopathy"
```

**Features:**
- Searches ClinicalTrials.gov API by acronym
- 30-day disk cache to avoid repeated API calls
- Rate limiting (1 req/sec)
- Auto-enriches "trial name" placeholders in heuristics config

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
print(f"Authors: {len(results.get('authors', []))}")
print(f"Citations: {len(results.get('citations', []))}")
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
| Authors | >80% |
| Citations | >90% |

---

## Extending the Pipeline

### Adding a New Generator

1. Create `C_generators/C15_strategy_new.py`
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
from C_generators.C15_strategy_new import NewGenerator

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
| No authors detected | No credentials found | Check PDF has author section with MD/PhD |
| No citations detected | References in images | OCR not applied to reference images |

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
- **ClinicalTrials.gov API:** https://clinicaltrials.gov/data-api/api

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.9 | 2026-01 | Author/investigator extraction, citation/reference extraction |
| 0.8 | 2026-01 | Output folder restructure, image file saving, Vision LLM analysis |
| 0.7 | 2025-12 | Image extraction, Vision LLM integration |
| 0.6 | 2025-11 | Feasibility extraction, document metadata |
| 0.5 | 2025-10 | Drug detection, PubTator enrichment |
| 0.4 | 2025-09 | Disease detection, NCT enrichment |
| 0.3 | 2025-08 | Multi-generator architecture |
| 0.2 | 2025-07 | Validation layer, heuristics |
| 0.1 | 2025-06 | Initial pipeline |
