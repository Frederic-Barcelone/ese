# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ESE (Entity & Structure Extraction)** - A production-grade 6-layer pipeline for extracting structured metadata from clinical trial and medical PDF documents. Focused on rare disease research.

### Core Capabilities
- Abbreviations/acronyms with definitions
- Diseases (rare diseases, ICD-10, SNOMED, ORPHA, MONDO codes)
- Drugs (RxNorm, MeSH, DrugBank, development phase)
- Genes (HGNC symbols, Entrez, Ensembl, disease associations)
- Authors/investigators with affiliations and ORCID
- Citations/references (PMID, DOI, NCT identifiers)
- Clinical trial feasibility data
- Figures with Vision LLM analysis
- Tables with VLM extraction

## Commands

```bash
# Run pipeline on all PDFs in configured folder
python corpus_metadata/orchestrator.py

# Run tests
cd corpus_metadata && python -m pytest tests/ -v

# Type checking
mypy corpus_metadata

# Linting
ruff check corpus_metadata
```

### Environment Setup
```bash
# Requires Python 3.12+
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# API key required
export ANTHROPIC_API_KEY="your-key"
# Or add to .env file
```

## Architecture

```
PDF → B_parsing → C_generators → D_validation → E_normalization → F_evaluation
         ↓             ↓              ↓               ↓
     DocumentGraph  Candidates    Validated      Enriched+Deduplicated
```

### Layer Philosophy

| Layer | Goal | Strategy |
|-------|------|----------|
| **Generators (C)** | High Recall | Exhaustive extraction, noise acceptable |
| **Validation (D)** | High Precision | Claude LLM filters false positives |
| **Normalization (E)** | Standardization | Map to ontologies, deduplicate |

### Directory Structure (corpus_metadata/)

```
A_core/          # Domain models (Pydantic), interfaces, provenance
B_parsing/       # PDF→DocumentGraph, table/figure extraction, layout detection
C_generators/    # Candidate generation (syntax, regex, FlashText lexicons, LLM)
D_validation/    # LLM verification, prompt registry
E_normalization/ # Term mapping, disambiguation, PubTator/NCT enrichment
F_evaluation/    # Gold standard loading, precision/recall scoring
G_config/        # config.yaml (all pipeline parameters)
H_pipeline/      # Pipeline components, abbreviation pipeline
I_extraction/    # Entity and feasibility processors
J_export/        # JSON export handlers
Z_utils/         # Utilities
orchestrator.py  # Main entry point
```

## Configuration

All parameters in `corpus_metadata/G_config/config.yaml`. Key sections:

### Extraction Presets
```yaml
extraction_pipeline:
  preset: "standard"  # Options: drugs_only, diseases_only, abbreviations_only,
                      # feasibility_only, entities_only, all, minimal
```

### Entity Toggle
```yaml
extractors:
  drugs: true
  diseases: true
  genes: true
  abbreviations: true
  feasibility: true
  # etc.
```

### API Configuration
```yaml
api:
  claude:
    validation:
      model: "claude-sonnet-4-20250514"
```

## Key Patterns

### Adding a New Entity Type

1. Create domain model in `A_core/` (e.g., `A12_gene_models.py`)
2. Create generator in `C_generators/` (e.g., `C16_strategy_gene.py`)
3. Add detector/enricher in `E_normalization/` if needed
4. Register in `H_pipeline/H01_component_factory.py`
5. Add processing method to `I_extraction/I01_entity_processors.py`
6. Add export handler in `J_export/J01_export_handlers.py`
7. Wire up in `orchestrator.py`

### Generator Interface
```python
class CandidateGenerator(ABC):
    @abstractmethod
    def generate(self, doc: DocumentGraph) -> List[Candidate]:
        """Extract candidates from document."""
        pass
```

### Validation Flow (PASO heuristics)
- PASO A: Auto-approve statistical abbreviations (CI, HR, SD)
- PASO B: Country codes (handled via blacklist)
- PASO C: Hyphenated abbreviations (auto-enriched from ClinicalTrials.gov)
- PASO D: LLM SF-only extraction for missing abbreviations

## Output Structure

Processing `document.pdf` creates:
```
document/
├── abbreviations_document_YYYYMMDD_HHMMSS.json
├── diseases_document_*.json
├── drugs_document_*.json
├── genes_document_*.json
├── figures_document_*.json
├── tables_document_*.json
├── document_extracted_text_*.txt
└── document_flowchart_page3_1.png
```

## External Dependencies

- **Claude API** - Validation and Vision LLM
- **PubTator3** - MeSH codes, disease aliases
- **ClinicalTrials.gov** - NCT metadata enrichment
- **Unstructured.io** - PDF parsing
- **scispacy** - Biomedical NER with UMLS linking
- **FlashText** - Fast lexicon matching (600K+ terms)
- **PyMuPDF (fitz)** - PDF native figure extraction

## Lexicons Loaded (~617K terms)

| Source | Terms | Purpose |
|--------|-------|---------|
| Meta-Inventory | 65K | Clinical abbreviations |
| MONDO | 97K | Unified disease ontology |
| ChEMBL | 23K | Approved drugs |
| RxNorm | 132K | Drug vocabulary |
| Orphanet | 9.5K | Rare diseases |
| Trial acronyms | 125K | ClinicalTrials.gov |
