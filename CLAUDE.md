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

## Claude Code Plugins & Workflows

Claude Code MUST use these plugins proactively to ensure code quality, maintainability, and correct extraction logic.

### Development Workflow (Required Order)

```
New Feature/Entity Type:
  /brainstorming → /writing-plans → /test-driven-development → implement → /code-simplifier → /review-pr

Bug Fix:
  /systematic-debugging → /test-driven-development → fix → /code-simplifier → /verification-before-completion

Refactoring:
  /writing-plans → implement → /code-simplifier → /review-pr
```

### Plugin Reference

| Plugin | When to Use | ESE-Specific Notes |
|--------|-------------|-------------------|
| `/brainstorming` | **Before ANY new feature** - entity types, extraction strategies, validation rules | Explore recall vs precision tradeoffs for generators |
| `/writing-plans` | Multi-step tasks, new entity pipelines | Plan across all layers (A→C→D→E→J) |
| `/test-driven-development` | **Before writing implementation** | Write tests for generators, validators, normalizers first |
| `/systematic-debugging` | Test failures, extraction errors, false positives/negatives | Check each pipeline layer systematically |
| `/code-simplifier` | **After completing any code change** | Preserve layer separation, don't merge across A_core/C_generators/D_validation |
| `/review-pr` | Before merging, after major features | Verify extraction accuracy, API error handling |
| `/verification-before-completion` | **Before claiming work is done** | Run `pytest`, `mypy`, `ruff` - confirm extraction outputs |

### Code Quality Agents (Auto-Invoked)

| Agent | Purpose | ESE Focus |
|-------|---------|-----------|
| `code-reviewer` | Style, patterns, best practices | Pydantic model design, generator interfaces |
| `silent-failure-hunter` | Find swallowed errors, bad fallbacks | Critical for API calls (Claude, PubTator, NCT) |
| `type-design-analyzer` | Type invariants, encapsulation | A_core/ models must have strong types |
| `comment-analyzer` | Comment accuracy, maintainability | Extraction logic comments must match code |
| `pr-test-analyzer` | Test coverage gaps | Ensure edge cases for rare disease names |

### Parallel Work

| Plugin | When to Use |
|--------|-------------|
| `/dispatching-parallel-agents` | Independent tasks (e.g., add drug extractor + add gene extractor) |
| `/subagent-driven-development` | Execute plan steps in parallel within session |
| `/using-git-worktrees` | Isolate experimental extraction strategies |

### Mandatory Plugin Usage

**Claude MUST invoke these plugins automatically:**

1. **Starting new work**: `/brainstorming` before designing extraction logic
2. **Multi-file changes**: `/writing-plans` before touching code
3. **Any implementation**: `/test-driven-development` before writing production code
4. **After code changes**: `/code-simplifier` to clean up
5. **Before commits**: `/verification-before-completion` to run all checks
6. **Bugs/failures**: `/systematic-debugging` before proposing fixes

### ESE-Specific Quality Rules

When using plugins, enforce these patterns:

**Generators (C_generators/)**
- High recall, accept false positives
- Use FlashText for lexicon matching (not regex for large vocabularies)
- Every generator must implement `CandidateGenerator` interface

**Validators (D_validation/)**
- High precision, filter aggressively
- LLM prompts must be versioned in prompt registry
- Never auto-approve without explicit PASO rule

**Normalizers (E_normalization/)**
- Map to standard ontologies (MONDO, RxNorm, HGNC)
- Handle API failures gracefully (PubTator, NCT)
- Deduplicate by canonical ID, not string matching

**Models (A_core/)**
- Strong Pydantic types with validators
- Provenance tracking on all extracted entities
- No optional fields without default values

### Verification Checklist

Before marking any task complete, Claude MUST verify:

```bash
# All tests pass
cd corpus_metadata && python -m pytest tests/ -v

# Type checking passes
mypy corpus_metadata

# Linting passes
ruff check corpus_metadata

# Pipeline runs without errors (if extraction logic changed)
python corpus_metadata/orchestrator.py --dry-run
```
