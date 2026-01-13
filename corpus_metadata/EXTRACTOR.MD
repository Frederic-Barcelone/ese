# Metadata Extraction Pipeline

A 6-layer pipeline for extracting structured metadata from PDFs: abbreviations, diseases, drugs, clinical trial feasibility data, and document metadata.

**Design Philosophy:**
- Generators (C) → High Recall (exhaustive, some noise OK)
- Validation (D) → High Precision (Claude filters noise)
- Layers: Parsing → Generation → Validation → Normalization → Evaluation

---

## Directory Structure

```
corpus_metadata/
│
├── A_core/                              # LAYER 0: FOUNDATION
│   ├── A01_domain_models.py             # Core Pydantic schemas (Candidate, ExtractedEntity)
│   ├── A02_interfaces.py                # Abstract Base Classes for all components
│   ├── A03_provenance.py                # Run ID generation, git hash, traceability
│   ├── A04_heuristics_config.py         # Centralized heuristics (whitelists/blacklists)
│   ├── A05_disease_models.py            # Disease-specific domain models
│   ├── A06_drug_models.py               # Drug/chemical entity models
│   ├── A07_feasibility_models.py        # Clinical trial feasibility models
│   └── A08_document_metadata_models.py  # Document-level metadata models
│
├── B_parsing/                           # LAYER 1: DOCUMENT STRUCTURE
│   ├── B01_pdf_to_docgraph.py           # PDF parsing with Docling/Unstructured
│   ├── B02_doc_graph.py                 # Internal model (Page → Block → Token)
│   ├── B03_table_extractor.py           # Table detection and JSON serialization
│   ├── B04_column_ordering.py           # Multi-column layout detection and ordering
│   ├── B05_section_detector.py          # Section/heading detection
│   ├── B06_confidence.py                # Confidence scoring features
│   └── B07_negation.py                  # Negation/assertion detection
│
├── C_generators/                        # LAYER 2: CANDIDATE GENERATION
│   ├── C00_strategy_identifiers.py      # Standardized database IDs (OMIM, DOI, NCT, etc.)
│   ├── C01_strategy_abbrev.py           # Schwartz-Hearst syntax patterns
│   ├── C02_strategy_regex.py            # Rigid patterns (Trial IDs, DOIs, doses)
│   ├── C03_strategy_layout.py           # Spatial extraction from table cells
│   ├── C04_strategy_flashtext.py        # Lexicon matching (UMLS, MeSH, trial acronyms)
│   ├── C05_strategy_glossary.py         # Glossary table extractor
│   ├── C06_strategy_disease.py          # Disease mention detection
│   ├── C07_strategy_drug.py             # Drug/chemical detection
│   ├── C08_strategy_feasibility.py      # Clinical trial feasibility extraction
│   └── C09_strategy_document_metadata.py # Document metadata extraction
│
├── D_validation/                        # LAYER 3: LLM VERIFICATION
│   ├── D01_prompt_registry.py           # Version control for prompts
│   ├── D02_llm_engine.py                # Claude API wrapper with JSON extraction
│   └── D03_validation_logger.py         # JSONL audit trail for validation events
│
├── E_normalization/                     # LAYER 4: STANDARDIZATION
│   ├── E01_term_mapper.py               # Synonym resolution (pembro → pembrolizumab)
│   ├── E02_disambiguator.py             # Context-based disambiguation (MS problem)
│   ├── E03_disease_normalizer.py        # Disease ontology normalization
│   ├── E04_pubtator_enricher.py         # PubTator API disease enrichment
│   └── E05_drug_enricher.py             # Drug identifier enrichment
│
├── F_evaluation/                        # LAYER 5: METRICS & TESTING
│   ├── F01_gold_loader.py               # Load gold standard annotations
│   ├── F02_scorer.py                    # Precision/Recall/F1 calculation
│   ├── F03_generator_unit_test.py       # Unit tests for generators (no LLM)
│   ├── F04_pipeline_test.py             # Full pipeline evaluation with scoring
│   └── F05_extraction_analysis.py       # Detailed extraction report (TP/FP/FN)
│
├── G_config/                            # CONFIGURATION
│   └── config.yaml                      # Central configuration file
│
├── Z_tests/                             # DEVELOPMENT TESTS
│   ├── small_script.py                  # Quick debugging scripts
│   └── test_parsing_with_unstructured.py # Parser testing
│
├── cache/                               # Runtime cache (PubTator data)
├── orchestrator.py                      # Main pipeline entry point
└── EXTRACTOR.MD                         # This documentation
```

---

## A_core (Foundation Layer)

### A01_domain_models.py - Core Data Structures
Defines all core data structures using Pydantic for strict validation.

**Key Models:**
- `Candidate`: Raw extraction with confidence score, awaiting validation
- `ExtractedEntity`: Validated and normalized entity
- `ValidationStatus`: Enum (VALIDATED, REJECTED, AMBIGUOUS, ERROR)
- `GeneratorType`: Enum (SYNTAX_PATTERN, LEXICON_MATCH, TABLE_LAYOUT, etc.)
- `BoundingBox`: Geometric model for spatial coordinates
- `ProvenanceMetadata`: Tracks pipeline version, run ID, lexicon source

**Purpose:** If a script tries to pass invalid data, Pydantic stops it immediately.

### A02_interfaces.py - Abstract Contracts
Abstract Base Classes that all components must implement.

**Key Interfaces:**
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

class BaseEvaluationMetric(ABC):
    @abstractmethod
    def compute(self, predictions, gold) -> Dict:
        """Calculate evaluation metrics."""
```

### A03_provenance.py - Traceability
Handles audit trail and reproducibility.

**Functions:**
- `generate_run_id()`: Creates unique ID like `RUN_20260106_143022_abc123`
- `get_git_revision_hash()`: Captures code version for reproducibility
- `hash_prompt()`: Fingerprints LLM prompts to track behavior changes

### A04_heuristics_config.py - The Tuning Panel
Centralizes all configurable rules in one place.

**Key Components:**
- `stats_whitelist`: Auto-approve statistical terms (CI, SD, OR, HR)
- `country_codes`: Auto-approve country abbreviations (US, UK, EU)
- `sf_blacklist`: Auto-reject known false positives (MD, PHD, NY)
- `hyphenated_abbrevs`: Handle hyphenated forms (CKD-EPI, sC5b-9)
- `HeuristicsCounters`: Track how many candidates hit each rule

### A05_disease_models.py - Disease Domain Models
Specialized models for disease entity extraction.

**Key Models:**
- `DiseaseCandidate`: Raw disease mention with context
- `ExtractedDisease`: Validated disease entity
- `DiseaseIdentifier`: Supports ICD-10, SNOMED-CT, MONDO, ORPHA codes
- `DiseaseGeneratorType`: Enum for disease lexicon sources

### A06_drug_models.py - Drug Domain Models
Models for drug and chemical entity extraction.

**Key Models:**
- `DrugCandidate`: Raw drug mention
- `ExtractedDrug`: Validated drug entity
- `DevelopmentPhase`: Enum (Preclinical, Phase I/II/III, Approved)
- Supports RxCUI, NDC, MeSH, DrugBank identifiers

### A07_feasibility_models.py - Clinical Trial Feasibility Models
Models for extracting clinical trial feasibility information.

**Key Models:**
- `FeasibilityFieldType`: Eligibility, epidemiology, patient journey, endpoints
- `CriterionType`: Inclusion/exclusion criteria types
- `PatientJourneyPhaseType`: Diagnosis, treatment, follow-up phases
- `EndpointType`: Primary, secondary, exploratory endpoints

### A08_document_metadata_models.py - Document Metadata Models
Models for document-level metadata extraction.

**Key Models:**
- `FileMetadata`: File system info (size, dates, permissions)
- `PDFMetadata`: PDF-specific fields (author, creator, creation date)
- `DocumentMetadata`: Classification, descriptions, date extraction

---

## B_parsing (Document Structure Layer)

### B01_pdf_to_docgraph.py - The Parser
Converts PDF to structured DocumentGraph using Docling.

**Features:**
- Layout-aware parsing (detects columns, headers, footers)
- Table detection with cell boundaries
- Handles multi-column academic papers
- Exports extracted text with timestamps

### B02_doc_graph.py - The Document Model
Internal representation of document structure.

**Hierarchy:** Document → Page → Block → Text
- Preserves spatial coordinates (bbox)
- Tags blocks as TITLE, PARAGRAPH, TABLE_CELL, etc.
- Enables spatial queries ("find text in upper-right corner")

### B03_table_extractor.py - Table Specialist
Dedicated module for table reconstruction.

**Capabilities:**
- Detects table boundaries from visual cues
- Reconstructs cell-to-cell relationships
- Serializes to JSON for downstream processing

### B04_column_ordering.py - Layout Intelligence
Handles multi-column document layouts.

**Purpose:** Academic papers often have 2-column layouts. This module ensures text is read in correct order (left column top-to-bottom, then right column).

### B05_section_detector.py - Section Detection
Identifies document sections and headings.

**Detects:**
- Standard sections (Abstract, Methods, Results, Discussion)
- Custom section headers
- Hierarchical section structure

### B06_confidence.py - Confidence Scoring
Calculates extraction confidence based on document context.

**Features:**
- Positional reliability scoring
- Source quality weighting
- Context-based confidence adjustment

### B07_negation.py - Negation/Assertion Detection
Classifies assertions as positive, negative, or uncertain.

**Purpose:** Avoids false positives from negated mentions:
- "No evidence of disease" → negative assertion
- "Patient has disease" → positive assertion
- "Possible disease" → uncertain assertion

---

## C_generators (Candidate Generation Layer)

Philosophy: **High Recall**. Better to propose something incorrect (filtered later) than miss a valid entity.

### C00_strategy_identifiers.py - Database Identifiers
Extracts standardized database identifiers.

**Extracts:**
- Disease IDs: OMIM, Orphanet
- Publication IDs: DOI, PMID, PMC
- Trial IDs: NCT, ISRCTN, EudraCT
- Author IDs: ORCID
- Ontology codes: UMLS CUI, MeSH, ICD-10, SNOMED CT

### C01_strategy_abbrev.py - Syntax Patterns
Schwartz-Hearst algorithm for explicit definitions.

**Patterns Detected:**
- `Long Form (SF)`: "Tumor Necrosis Factor (TNF)"
- `SF (Long Form)`: "TNF (Tumor Necrosis Factor)"
- Implicit definitions: "TNF, defined as..."

### C02_strategy_regex.py - Rigid Patterns
Pattern matching for structured identifiers.

**Extracts:**
- Trial IDs: NCT01234567
- DOIs: 10.1000/xyz123
- Compound IDs: LNP023, ABC-123

### C03_strategy_layout.py - Spatial Extraction
Uses document coordinates for positional extraction.

**Use Cases:**
- Table cells by position
- Form fields by label proximity
- Header/footer regions

### C04_strategy_flashtext.py - Lexicon Matching
High-speed dictionary matching using FlashText + regex.

**Lexicons Loaded (~250,278 terms):**
- Abbreviations (5,392)
- Rare disease acronyms (1,630)
- UMLS biological (97,336)
- UMLS clinical (20,000)
- ANCA/IgAN/PAH disease lexicons (167)
- **Trial acronyms (125,454)** - Clinical trial names from ClinicalTrials.gov
- **PRO scales (299)** - Patient-Reported Outcome instruments (SF-36, PHQ-9, EORTC-QLQ-C30, etc.)

**Also Includes:**
- scispacy NER with UMLS EntityLinker
- Schwartz-Hearst abbreviation detector

### C05_strategy_glossary.py - Glossary Tables
Extracts abbreviations from explicit glossary sections.

**Detects:**
- Sections titled "Abbreviations", "List of Abbreviations"
- Tables with columns "Abbreviation" / "Definition"
- High confidence (author-provided definitions)

### C06_strategy_disease.py - Disease Extraction
Specialized disease mention detection.

**Lexicons:**
- Disease-specific lexicons (PAH, ANCA, IgAN, C3G)
- General disease lexicon (29K+ diseases)
- Orphanet rare diseases (9.6K)
- scispacy NER with UMLS disease types

**False Positive Filtering:**
- Chromosome patterns (Chr1, 1p36)
- Gene symbols
- Chemical compounds

### C07_strategy_drug.py - Drug Extraction
Drug and chemical entity detection.

**Lexicons:**
- Investigational drug lexicons
- FDA-approved drugs
- RxNorm general terms
- Compound ID patterns (ABC-123)

**Methods:**
- FlashText lexicon matching
- scispacy CHEMICAL NER
- Regex compound patterns

### C08_strategy_feasibility.py - Feasibility Extraction
Clinical trial feasibility information extraction.

**Extracts:**
- Eligibility criteria (inclusion/exclusion)
- Epidemiology data (prevalence, incidence)
- Patient journey phases
- Study endpoints (primary, secondary, exploratory)

**Methods:**
- Rule-based extraction for structured fields
- LLM-based extraction for complex information

### C09_strategy_document_metadata.py - Document Metadata Extraction
Document-level metadata extraction.

**Extracts:**
- File metadata (size, dates)
- PDF metadata (author, creator, creation date)
- Document classification (protocol vs. article)
- Date extraction with fallback chain

---

## D_validation (LLM Verification Layer)

Philosophy: **High Precision**. Filter noise using Claude's reasoning.

### D01_prompt_registry.py - Prompt Version Control
Manages all LLM prompts with versioning.

**Prompt Tasks:**
- `VERIFY_DEFINITION_PAIR`: Validate abbreviation-expansion pairs
- `VERIFY_SHORT_FORM_ONLY`: Validate standalone abbreviations
- `VERIFY_BATCH`: Batch validation of multiple candidates
- `FAST_REJECT`: Quick rejection of obvious false positives
- Disease and drug validation tasks

**Features:**
- Unique prompt IDs: `verify_abbreviation:v2.1`
- Hash-based change detection
- Full audit trail for regulatory compliance

### D02_llm_engine.py - Claude Wrapper
Constrained LLM interaction with JSON output.

**Validation Modes:**
- Batch validation: Multiple explicit pairs at once
- Individual validation: Single lexicon-sourced candidates
- SF-only extraction (PASO D): Find undefined abbreviations

**Output Control:**
- Forces structured JSON responses
- Handles errors gracefully (marks as AMBIGUOUS)
- Retry logic for transient failures

### D03_validation_logger.py - Audit Trail
JSONL logging of all validation decisions.

**Logs:**
- Candidate submitted
- Prompt used
- Raw LLM response
- Final verdict and confidence

---

## E_normalization (Standardization Layer)

### E01_term_mapper.py - Synonym Resolution
Maps variants to canonical forms.

**Examples:**
- "pembro" → "pembrolizumab"
- "eGFR" → "estimated glomerular filtration rate"

**Features:**
- Fuzzy matching for typos
- Ontology linking (RxNorm, UMLS CUIs)

### E02_disambiguator.py - Context Resolution
Resolves polysemy using document context.

**The "MS" Problem:**
- "MS" + brain/lesion/relapse → Multiple Sclerosis
- "MS" + chromatography/ions → Mass Spectrometry

**Method:** Contextual voting based on surrounding terms.

### E03_disease_normalizer.py - Disease Normalization
Maps disease mentions to standard ontologies.

**Links to:**
- ICD-10 codes
- SNOMED-CT codes
- MONDO ontology
- Orphanet rare disease IDs

### E04_pubtator_enricher.py - PubTator Enrichment
Uses PubTator API for disease entity linking.

**Features:**
- Adds MeSH IDs
- Adds standard disease codes
- Cache-based to avoid redundant API calls

### E05_drug_enricher.py - Drug Enrichment
Maps drugs to standard identifiers.

**Links to:**
- RxNorm codes
- DrugBank IDs
- NDC codes
- MeSH terms

---

## F_evaluation (Metrics & Testing Layer)

### F01_gold_loader.py - Gold Standard Loader
Loads human-annotated ground truth.

**Supported Formats:**
- `papers_gold_v2.json`: defined_annotations, mentioned_annotations
- Standard format: annotations list
- Handles doc_id matching with fuzzy stem comparison

### F02_scorer.py - Metrics Calculator
Calculates Precision/Recall/F1.

**Classification:**
- **True Positive (TP):** System found, Gold confirms
- **False Positive (FP):** System found, Gold denies
- **False Negative (FN):** System missed, Gold has it

**Metrics:**
- Precision: TP / (TP + FP) - "How trustworthy?"
- Recall: TP / (TP + FN) - "How complete?"
- F1: Harmonic mean of P and R
- Micro (global) and macro (per-document) scoring

### F03_generator_unit_test.py - Generator Unit Tests
Tests generators without LLM (fast iteration).

**Purpose:**
- Validate regex patterns work correctly
- Measure recall before validation filtering
- Test synthetic sentences from gold pairs

### F04_pipeline_test.py - Full Pipeline Evaluation
End-to-end evaluation against gold standard.

**Process:**
1. Load annotated PDFs
2. Run full Orchestrator pipeline
3. Score against gold annotations
4. Report per-document and corpus metrics

**Output:**
```
MICRO (global):
  Precision: 40.0%
  Recall:    81.4%
  F1:        53.6%
  TP: 48 | FP: 72 | FN: 11
```

### F05_extraction_analysis.py - Detailed Analysis Report
Shows extraction results on screen (no file output).

**Sections:**
- Extraction summary (candidates, validated, rejected, ambiguous)
- Metrics (Precision/Recall/F1)
- Heuristics counters (auto-approved, auto-rejected)
- True Positives list
- False Positives list (with expansions)
- False Negatives list (with page numbers)
- Mentioned abbreviations status

**Called automatically** after each extraction by orchestrator.

---

## orchestrator.py - The Conductor

Main entry point that coordinates all layers.

**Pipeline Flow:**
1. **Parse** (B): PDF → DocumentGraph
2. **Generate** (C): DocumentGraph → Candidates
3. **Validate** (D): Candidates → Validated/Rejected
4. **Normalize** (E): Standardize and enrich entities
5. **Export**: JSON + Analysis report

**Key Features:**
- Configurable via `G_config/config.yaml`
- Heuristic shortcuts (PASO A-D) for efficiency
- Detailed logging to JSONL
- Automatic F05 analysis after extraction

**Heuristics (PASO):**
- **PASO A:** Auto-approve stats (CI, SD, OR with numeric context)
- **PASO B:** Auto-approve country codes (US, UK, EU)
- **PASO C:** Detect hyphenated abbreviations (CKD-EPI, sC5b-9)
- **PASO D:** LLM SF-only extractor for undefined terms

---

## G_config (Configuration)

All parameters centralized in `G_config/config.yaml`:

**Sections:**
- `system`: Pipeline name and version
- `paths`: Input/output directories, gold data location
- `lexicons`: 25+ lexicon file paths with version dating
- `databases`: Disease ontology databases
- `defaults`: Global parameters (confidence threshold, fuzzy match threshold)
- `features`: Feature flags for drug/disease/abbreviation detection
- `heuristics`: Tuning parameters (PASO A-D)
- `validation`: LLM settings, model selection, temperature

---

## Usage

### Single PDF Extraction
```python
from corpus_metadata.orchestrator import Orchestrator

orch = Orchestrator()
entities = orch.process_pdf("/path/to/document.pdf")
```

### Pipeline Evaluation
```bash
python -m corpus_metadata.F_evaluation.F04_pipeline_test
```

### Environment
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

---

## Gold Standard

Using NLP4RARE corpus for evaluation:
- Source: https://github.com/isegura/NLP4RARE-CM-UC3M
- Paper: https://www.sciencedirect.com/science/article/pii/S1532046421002902
- Contains 5,000+ rare diseases and 6,000 clinical manifestations
- Annotated entities: disease, rare disease, symptom, sign, anaphor
- Annotated relations: produces, is_a, is_acron, is_synon
