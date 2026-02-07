# Pipeline Overview

ESE processes clinical PDF documents through a layered extraction pipeline, balancing high recall at generation time with high precision at validation time.

## Pipeline Diagram

```
                         +------------------+
                         |    Source PDF     |
                         +--------+---------+
                                  |
                                  v
                    +-------------+-------------+
                    |     B_parsing (Layer 2)    |
                    |   PDF -> DocumentGraph     |
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |   C_generators (Layer 3)   |
                    |   Candidate Generation     |
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |   D_validation (Layer 4)   |
                    |   LLM Verification         |
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |  E_normalization (Layer 5)  |
                    |   Standardization          |
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |  I_extraction (Layer 9)     |
                    |   Entity Processing        |
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |    J_export (Layer 10)      |
                    |   JSON Export              |
                    +-------------+-------------+
```

## Layer Philosophy

| Layer | Goal | Strategy |
|-------|------|----------|
| **B_parsing** | Faithful document representation | PDF parsing, layout detection, table/figure extraction |
| **C_generators** | **High Recall** | Exhaustive candidate extraction; noise acceptable |
| **D_validation** | **High Precision** | Claude LLM filters false positives aggressively |
| **E_normalization** | **Standardization** | Map to ontologies (MONDO, RxNorm, HGNC), deduplicate |
| **I_extraction** | Entity assembly | Per-entity-type processing pipelines |
| **J_export** | Structured output | JSON with provenance and audit trail |

## Directory Structure

All pipeline code resides under `corpus_metadata/`.

| Directory | Files | Purpose |
|-----------|-------|---------|
| `A_core/` | 24 | Domain models (Pydantic), interfaces, provenance, exceptions |
| `B_parsing/` | 31 | PDF to DocumentGraph, table extraction, figure detection, layout |
| `C_generators/` | 35 | Candidate generation (syntax, regex, FlashText, scispacy, LLM, VLM) |
| `D_validation/` | 4 | LLM verification, prompt registry, quote verifier |
| `E_normalization/` | 18 | Term mapping, disambiguation, PubTator/NCT enrichment, dedup |
| `F_evaluation/` | 3 | Gold standard loading, precision/recall scoring |
| `G_config/` | 1 | `G01_config_keys.py` + `config.yaml` |
| `H_pipeline/` | 4 | Component factory, abbreviation pipeline, visual integration, merge resolver |
| `I_extraction/` | 2 | Entity processors (`I01`) and feasibility processor (`I02`) |
| `J_export/` | 4 | Entity, metadata, and visual export handlers |
| `Z_utils/` | 11 | API client, text helpers, image utilities, download scripts |
| `K_tests/` | varies | Pytest test suite |
| `orchestrator.py` | 1 | Main entry point |

## Technology Stack

| Technology | Role |
|------------|------|
| Python 3.12+ | Runtime |
| Pydantic v2 | Domain models, validation, serialization |
| Claude API (Anthropic) | LLM validation, Vision analysis, feasibility extraction |
| Unstructured.io / Docling | PDF parsing backends |
| PyMuPDF (fitz) | Native PDF figure extraction, rendering |
| scispacy | Biomedical NER with UMLS linking |
| FlashText | Fast multi-pattern lexicon matching (600K+ terms) |
| Custom console (Z07) | Pipeline stage output, progress display |

## LLM Cost Optimization

Every LLM call site is tagged with a `call_type` string that maps to a specific model in `config.yaml`:

- **Haiku tier** ($1/$5 per MTok): Classification, layout analysis, abbreviation validation
- **Sonnet tier** ($3/$15 per MTok): Feasibility extraction, flowchart analysis, table extraction

Additional optimizations:
- **Prompt caching**: System prompts use `cache_control: {"type": "ephemeral"}` for 90% savings on repeated prompts
- **Fast-reject pre-screening**: Haiku pre-screens abbreviation candidates before full validation
- **Usage tracking**: All API calls tracked in-memory (`LLMUsageTracker`) and persisted to SQLite (`llm_usage` table) with per-document and batch cost summaries

See [Cost Optimization Guide](../guides/05_cost_optimization.md) for details.

## Orchestrator

`orchestrator.py` processes each PDF through 16 stages:

1. **PDF Parsing** -- PDF to DocumentGraph
2. **Candidate Generation** -- Abbreviation extraction (syntax, regex, lexicon, layout)
3. **LLM Validation** -- PASO heuristic filtering + Claude API validation
4. **Normalization** -- Abbreviation normalization and deduplication
5. **Disease Detection** -- FlashText + scispacy NER, FP filtering, PubTator3 enrichment
6. **Gene Detection** -- HGNC lexicon + pattern matching, PubTator3 enrichment
7. **Drug Detection** -- RxNorm/ChEMBL lexicon, FP filtering, enrichment
8. **Pharma Detection** -- Pharmaceutical company identification
9. **Author Detection** -- Author name and affiliation extraction
10. **Citation Detection** -- PMID, DOI, NCT extraction and API validation
11. **Feasibility Extraction** -- Pattern + LLM + NER pipeline
12. **Care Pathway Extraction** -- Clinical treatment algorithm extraction
13. **Recommendation Extraction** -- Guideline recommendations (LLM + VLM)
14. **Visual Extraction** -- Table and figure detection with VLM triage + Claude Vision
15. **Document Metadata** -- Type classification, title, date extraction
16. **Export & Summary** -- JSON export with metrics

Each stage is configurable through `G_config/config.yaml` and can be toggled via extraction presets.

## Related Documentation

- [Data Flow](02_data_flow.md) -- entity lifecycle and per-type flows
- [Domain Models](03_domain_models.md) -- Pydantic model reference
- [Configuration](../guides/03_configuration.md) -- config.yaml reference
