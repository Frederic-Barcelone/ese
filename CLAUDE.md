# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ESE (Entity & Structure Extraction)** — Pipeline v0.8. Production-grade extraction of structured metadata from clinical trial and medical PDF documents. Focused on rare disease research.

### Core Capabilities
- Abbreviations/acronyms with definitions (PASO heuristics)
- Diseases (rare diseases, ICD-10, SNOMED, ORPHA, MONDO codes)
- Drugs (RxNorm, MeSH, DrugBank, development phase)
- Genes (HGNC symbols, Entrez, Ensembl, disease associations)
- Authors/investigators with affiliations and ORCID
- Citations/references (PMID, DOI, NCT identifiers)
- Clinical trial feasibility data (eligibility, epidemiology, study design)
- Clinical guideline recommendations (text + VLM extraction)
- Care pathways (patient journey mapping)
- Figures with Vision LLM analysis
- Tables with VLM extraction
- Document metadata (classification, dates, descriptions)

## Commands

```bash
# Run pipeline on all PDFs in configured folder
cd corpus_metadata && python orchestrator.py

# Run tests (59 test files)
cd corpus_metadata && python -m pytest K_tests/ -v

# Type checking
cd corpus_metadata && python -m mypy .

# Linting
cd corpus_metadata && python -m ruff check .
```

### Environment Setup
```bash
# Requires Python 3.12+
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# API key required
export ANTHROPIC_API_KEY="your-key"
# Or add to .env file in project root
```

## Architecture

```
                          orchestrator.py
                               |
PDF → B_parsing → C_generators → D_validation → E_normalization → J_export
         |             |              |               |
     DocumentGraph  Candidates    Validated      Enriched+Deduplicated
                       |
              H_pipeline (abbreviation pipeline, component factory)
              I_extraction (entity/feasibility processors)
              F_evaluation (gold standard scoring)
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
C_generators/    # Candidate generation (syntax, regex, FlashText lexicons, LLM, VLM)
D_validation/    # LLM verification, prompt registry, cost optimization
E_normalization/ # Term mapping, disambiguation, PubTator/NCT enrichment
F_evaluation/    # Gold standard loading, precision/recall scoring
G_config/        # config.yaml (all pipeline parameters)
H_pipeline/      # Component factory, abbreviation pipeline, merge resolver
I_extraction/    # Entity and feasibility processors
J_export/        # JSON export handlers
Z_utils/         # API client, text helpers, image utils, usage tracking
orchestrator.py  # Main entry point (v0.8)
```

## Operations

### Running the Pipeline
```bash
# Process all PDFs in configured input folder
cd corpus_metadata && python orchestrator.py

# To process specific PDFs, update config.yaml:
#   paths.pdfs: "path/to/pdf/folder"
# There is NO CLI argument support — all config is in config.yaml.
```

### Logs and Tracking
- All logs go to `corpus_log/` (configurable via `paths.logs` in config.yaml)
- LLM usage tracked in `corpus_log/usage_stats.db` (SQLite)
- Console output is tee'd to `corpus_log/console_YYYYMMDD_HHMMSS.log`

### Output Structure
Processing `document.pdf` creates a folder alongside the PDF:
```
document/
├── abbreviations_document_YYYYMMDD_HHMMSS.json
├── diseases_document_*.json
├── drugs_document_*.json
├── genes_document_*.json
├── recommendations_document_*.json
├── figures_document_*.json
├── tables_document_*.json
├── document_extracted_text_*.txt
└── document_flowchart_page3_1.png
```

## Configuration

All parameters in `corpus_metadata/G_config/config.yaml`. Key sections:

### Extraction Presets
```yaml
extraction_pipeline:
  preset: "all"
  # Options: standard, all, minimal,
  #          drugs_only, diseases_only, genes_only,
  #          abbreviations_only, feasibility_only,
  #          entities_only, clinical_entities,
  #          metadata_only, images_only, tables_only
```

### Entity Toggle
```yaml
extractors:
  drugs: true
  diseases: true
  genes: true
  abbreviations: true
  feasibility: true
  care_pathways: true
  recommendations: true
  figures: true
  tables: true
  # etc.
```

### API Configuration
```yaml
api:
  claude:
    validation:
      model: "claude-sonnet-4-20250514"  # Default model
    model_tiers:  # Per-task model routing (overrides default)
      abbreviation_batch_validation: "claude-haiku-4-5-20250901"
      feasibility_extraction: "claude-sonnet-4-20250514"
      # ... 17 call_types total
```

### LLM Cost Optimization

The pipeline routes 17 LLM call sites to different models based on task complexity via `model_tiers` in config.yaml. Simple tasks (classification, layout analysis) use Haiku ($1/$5 per MTok). Complex reasoning (feasibility, recommendations, visual extraction) uses Sonnet ($3/$15 per MTok).

**Key files:**
- `D02_llm_engine.py` — `resolve_model_tier()`, `record_api_usage()`, `calc_record_cost()`, `LLMUsageTracker`, `MODEL_PRICING`
- `G_config/config.yaml` — `model_tiers` maps `call_type` → model ID
- `Z06_usage_tracker.py` — `llm_usage` SQLite table for persistent tracking
- `orchestrator.py` — Per-document and batch cost summaries

**call_type conventions:**
Every LLM call site must pass a `call_type` string. For `ClaudeClient` calls, pass `call_type=` to `complete_json`/`complete_json_any`/`complete_vision_json`. For raw `anthropic.Anthropic()` calls, call `record_api_usage(response, model, call_type)` after each API call.

| call_type | Tier | Used By |
|-----------|------|---------|
| `abbreviation_batch_validation` | Haiku | D02 LLMEngine |
| `abbreviation_single_validation` | Haiku | D02 LLMEngine |
| `fast_reject` | Haiku | D02 LLMEngine |
| `document_classification` | Haiku | C09 document metadata |
| `description_extraction` | Haiku | C09 document metadata |
| `image_classification` | Haiku | C10 vision analysis |
| `sf_only_extraction` | Haiku | H02 abbreviation pipeline |
| `layout_analysis` | Haiku | B19 layout analyzer |
| `vlm_visual_enrichment` | Haiku | C19, B22 visual enrichment |
| `ocr_text_fallback` | Haiku | C10 vision analysis |
| `feasibility_extraction` | Sonnet | C11 feasibility |
| `recommendation_extraction` | Sonnet | C32 recommendation LLM |
| `recommendation_vlm` | Sonnet | C33 recommendation VLM |
| `flowchart_analysis` | Sonnet | C10, C17 flowchart |
| `chart_analysis` | Sonnet | C10 chart analysis |
| `vlm_table_extraction` | Sonnet | C10 table extraction |
| `vlm_detection` | Sonnet | B31 VLM detector |

**Adding a new LLM call site:**
1. Choose a descriptive `call_type` string
2. Add it to `config.yaml` under `model_tiers` with the appropriate model
3. For `ClaudeClient` calls: pass `call_type=` parameter
4. For raw API calls: import and call `record_api_usage(response, model, call_type)`
5. Add the model to `MODEL_PRICING` in `D02_llm_engine.py` if it's new

## Key Patterns

### Adding a New Entity Type

1. Create domain model in `A_core/` (e.g., `A19_gene_models.py`)
2. Create generator in `C_generators/` (e.g., `C16_strategy_gene.py`)
3. Optionally create false-positive filter (e.g., `C34_gene_fp_filter.py`)
4. Add detector/enricher in `E_normalization/` if needed
5. Register in `H_pipeline/H01_component_factory.py`
6. Add processing method to `I_extraction/I01_entity_processors.py`
7. Add export handler in `J_export/J01_export_handlers.py`
8. Wire up in `orchestrator.py`

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

## External Dependencies

- **Claude API** — Validation and Vision LLM (17 call sites, 2 model tiers)
- **PubTator3** — MeSH codes, disease aliases
- **ClinicalTrials.gov** — NCT metadata enrichment
- **Unstructured.io** — PDF parsing backend
- **scispacy** — Biomedical NER with UMLS linking
- **FlashText** — Fast lexicon matching (600K+ terms)
- **PyMuPDF (fitz)** — PDF native figure extraction
- **Docling** — TableFormer table extraction (95-98% TEDS accuracy)

## Lexicons Loaded (~617K terms)

| Source | Terms | Purpose |
|--------|-------|---------|
| Meta-Inventory | 65K | Clinical abbreviations |
| MONDO | 97K | Unified disease ontology |
| ChEMBL | 23K | Approved drugs |
| RxNorm | 132K | Drug vocabulary |
| Orphanet | 9.5K | Rare diseases |
| Trial acronyms | 125K | ClinicalTrials.gov |

## Claude Code Workflows

### When to Use Plugins

Scale plugin usage to task size:

**Small changes** (config fix, doc update, 1-3 line fix):
- Just do the work. Run verification commands before claiming done.

**Medium changes** (bug fix, add method, modify behavior):
- `/systematic-debugging` for bugs, `/verification-before-completion` before done.

**Large changes** (new entity type, multi-file refactor, new pipeline stage):
- `/brainstorming` → `/writing-plans` → `/test-driven-development` → implement → `/code-simplifier` → `/verification-before-completion`

### Plugin Reference

| Plugin | When to Use | ESE-Specific Notes |
|--------|-------------|-------------------|
| `/brainstorming` | New entity types, extraction strategies | Explore recall vs precision tradeoffs |
| `/writing-plans` | Multi-file changes, new entity pipelines | Plan across all layers (A→C→D→E→J) |
| `/test-driven-development` | Before implementing new features | Write tests for generators, validators, normalizers first |
| `/systematic-debugging` | Test failures, extraction errors | Check each pipeline layer systematically |
| `/code-simplifier` | After completing large code changes | Preserve layer separation |
| `/verification-before-completion` | Before claiming any work is done | Run pytest, mypy, ruff |

### ESE-Specific Quality Rules

**Generators (C_generators/)**
- High recall, accept false positives
- Use FlashText for lexicon matching (not regex for large vocabularies)
- Every generator must implement `CandidateGenerator` or `BaseExtractor` interface

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
cd corpus_metadata && python -m pytest K_tests/ -v

# Type checking passes
cd corpus_metadata && python -m mypy .

# Linting passes
cd corpus_metadata && python -m ruff check .
```

## Documentation

Comprehensive docs in `docs/` folder:
- `docs/architecture/` — Pipeline overview, data flow, domain models
- `docs/layers/` — Per-layer documentation (A through Z)
- `docs/guides/` — Getting started, adding entities, configuration, evaluation, cost optimization
- `docs/reference/` — Lexicons, external APIs, output format
- `docs/plans/` — Design documents for pipeline improvements
