# Pipeline Overview

ESE processes clinical PDF documents through a layered extraction pipeline. Each layer has a distinct responsibility and quality goal, enabling the system to balance high recall at generation time with high precision at validation time.

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
                    |  pages, blocks, tables,    |
                    |  images, layout detection   |
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |   C_generators (Layer 3)   |
                    |   Candidate Generation     |
                    |  syntax, regex, FlashText,  |
                    |  scispacy NER, LLM, VLM    |
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |   D_validation (Layer 4)   |
                    |   LLM Verification         |
                    |  Claude API, PASO heuristics|
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |  E_normalization (Layer 5)  |
                    |   Standardization          |
                    |  PubTator3, NCT, ontology   |
                    |  mapping, deduplication     |
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |  I_extraction (Layer 9)     |
                    |   Entity Processing        |
                    |  entity + feasibility       |
                    |  processors                 |
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |    J_export (Layer 10)      |
                    |   JSON Export              |
                    |  per-entity type output     |
                    +-------------+-------------+
```

## Layer Philosophy

| Layer | Goal | Strategy |
|-------|------|----------|
| **B_parsing** | Faithful document representation | PDF parsing, layout detection, table/figure extraction |
| **C_generators** | **High Recall** | Exhaustive candidate extraction; noise is acceptable |
| **D_validation** | **High Precision** | Claude LLM filters false positives aggressively |
| **E_normalization** | **Standardization** | Map to ontologies (MONDO, RxNorm, HGNC), deduplicate |
| **I_extraction** | Entity assembly | Orchestrate per-entity-type processing pipelines |
| **J_export** | Structured output | JSON files with provenance, evidence, and audit trail |

## Directory Structure

All pipeline code resides under `corpus_metadata/`.

| Directory | Files | Purpose |
|-----------|-------|---------|
| `A_core/` | 22 | Domain models (Pydantic), interfaces, provenance, exceptions, enums |
| `B_parsing/` | 31 | PDF to DocumentGraph, table extraction, figure detection, layout analysis |
| `C_generators/` | 34 | Candidate generation strategies (syntax, regex, FlashText lexicons, LLM, VLM) |
| `D_validation/` | 4 | LLM verification engine, prompt registry, quote verifier, validation logger |
| `E_normalization/` | 18 | Term mapping, disambiguation, PubTator/NCT enrichment, deduplication |
| `F_evaluation/` | 3 | Gold standard loading, precision/recall scorer, evaluation runner |
| `G_config/` | 2 | `config.yaml` (all pipeline parameters) + `G01_config_keys.py` |
| `H_pipeline/` | 4 | Component factory, abbreviation pipeline, visual integration, merge resolver |
| `I_extraction/` | 2 | Entity processors (`I01`) and feasibility processor (`I02`) |
| `J_export/` | 4 | Export handlers: entity exporters, metadata exporters, visual export |
| `Z_utils/` | 11 | API client, text helpers, image utilities, path utilities, download scripts |
| `K_tests/` | varies | Pytest test suite |
| `orchestrator.py` | 1 | Main entry point |

## Technology Stack

| Technology | Version | Role |
|------------|---------|------|
| Python | 3.12+ | Runtime |
| Pydantic | v2 | Domain models, validation, serialization |
| Claude API (Anthropic) | claude-sonnet-4 | LLM validation, Vision analysis, feasibility extraction |
| Unstructured.io / Docling | latest | PDF parsing backends |
| PyMuPDF (fitz) | latest | Native PDF figure extraction, rendering |
| scispacy | latest | Biomedical NER with UMLS linking |
| FlashText | latest | Fast multi-pattern lexicon matching (600K+ terms) |
| Rich | latest | Console output, progress bars, tables |
| YAML | stdlib | Pipeline configuration |

## Orchestrator

The `orchestrator.py` entry point processes each PDF through 16+ stages:

1. Configuration loading and preset resolution
2. Run ID and provenance generation
3. PDF parsing to DocumentGraph (via Unstructured.io or Docling)
4. Native figure extraction (PyMuPDF)
5. Abbreviation candidate generation (syntax, regex, lexicon, layout)
6. PASO heuristic filtering (auto-approve/reject rules)
7. LLM abbreviation validation (Claude API)
8. Abbreviation normalization and deduplication
9. Disease candidate generation (FlashText + scispacy NER)
10. Disease false-positive filtering and PubTator3 enrichment
11. Drug candidate generation and enrichment
12. Gene candidate generation and enrichment
13. Author and citation extraction
14. Feasibility extraction (pattern-based + LLM-based)
15. Visual extraction (VLM triage + Claude Vision analysis)
16. JSON export for all entity types

Each stage is independently configurable through `G_config/config.yaml` and can be enabled or disabled via extraction presets.

## Related Documentation

- [Data Flow](02_data_flow.md) -- detailed entity lifecycle and per-type flows
- [Domain Models](03_domain_models.md) -- Pydantic model reference
- [Configuration](../guides/03_configuration.md) -- config.yaml reference
